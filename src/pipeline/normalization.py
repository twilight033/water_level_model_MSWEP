"""归一化统计量：只用训练期、只用原始未掩膜标签。

原实现（multi_task_lstm_wl2d.py 的 _load_data）在整条序列上计算强迫与
目标的均值方差，含验证期与测试期；消融脚本更是在打了掩膜的数据上计算，
导致不同缺失场景的统计量互不相同。这里统一为：

- 强迫：按变量在所有流域的**训练段**上汇总；
- 静态属性：按连续列在 86 个流域上计算，one-hot 列不标准化；
- 目标：按流域、按任务在该流域**训练段**上计算，且始终使用原始标签，
  与人工缺失场景无关，保证所有场景共用同一份统计量。
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

_ROOT = Path(__file__).resolve().parents[2]
for _p in (str(_ROOT / "src"), str(_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from pipeline.paths import NORM_STATS_FILE, SPLITS_DIR  # noqa: E402
from pipeline.loaders import FORCING_ORDER, TASKS, load_attributes, load_forcing, load_targets  # noqa: E402
from pipeline.splits import load_splits  # noqa: E402

_EPS = 1e-6


def train_mask(grid: pd.DatetimeIndex, entry: dict) -> np.ndarray:
    start = pd.Timestamp(entry["train_start"])
    end = pd.Timestamp(entry["train_end"])
    return (grid >= start) & (grid <= end)


def compute_norm_stats() -> dict:
    payload = load_splits()
    grid, forcing, basins = load_forcing()
    targets = load_targets(grid, basins)
    # 按**扩展集**（超集）计算属性统计量：基础集与扩展集共用同一份文件，
    # 已有列取值不变，因此两套属性集的结果严格可比
    attr_arr, attr_cols, onehot_cols = load_attributes(basins, attr_set="extended")

    # ── 强迫：汇总所有流域训练段 ────────────────────────────────────────────
    total_n = 0
    total_sum = np.zeros(len(FORCING_ORDER), dtype="float64")
    total_sq = np.zeros(len(FORCING_ORDER), dtype="float64")
    for bi, basin in enumerate(basins):
        if basin not in payload["splits"]:
            continue
        m = train_mask(grid, payload["splits"][basin])
        chunk = forcing[bi][m].astype("float64")
        total_n += chunk.shape[0]
        total_sum += chunk.sum(axis=0)
        total_sq += (chunk ** 2).sum(axis=0)
    mean = total_sum / total_n
    var = np.maximum(total_sq / total_n - mean ** 2, 0.0)
    std = np.sqrt(var)
    forcing_stats = {
        var_name: {"mean": float(mean[i]), "std": float(max(std[i], _EPS))}
        for i, var_name in enumerate(FORCING_ORDER)
    }

    # ── 静态属性：one-hot 列不标准化 ────────────────────────────────────────
    attr_stats = {}
    for j, col in enumerate(attr_cols):
        if col in onehot_cols:
            attr_stats[col] = {"mean": 0.0, "std": 1.0, "onehot": True}
        else:
            col_vals = attr_arr[:, j].astype("float64")
            s = float(col_vals.std(ddof=0))
            attr_stats[col] = {"mean": float(col_vals.mean()),
                               "std": float(s if s > _EPS else 1.0), "onehot": False}

    # ── 目标：按流域按任务，训练段 + 原始标签 ───────────────────────────────
    target_stats = {task: {} for task in TASKS}
    for bi, basin in enumerate(basins):
        if basin not in payload["splits"]:
            continue
        m = train_mask(grid, payload["splits"][basin])
        for task in TASKS:
            vals = targets[task][bi][m]
            vals = vals[~np.isnan(vals)].astype("float64")
            if vals.size == 0:
                target_stats[task][basin] = {"mean": 0.0, "std": 1.0, "n": 0}
                continue
            s = float(vals.std(ddof=0))
            target_stats[task][basin] = {
                "mean": float(vals.mean()),
                "std": float(s if s > _EPS else 1.0),
                "n": int(vals.size),
            }

    return {
        "meta": {
            "source": "训练段 + 原始未掩膜标签；与缺失场景无关",
            "forcing_order": list(FORCING_ORDER),
            "attr_columns": attr_cols,
            "attr_set": "extended（超集；base 子集读同一份文件）",
            "onehot_columns": onehot_cols,
            "n_forcing_steps_pooled": int(total_n),
            "n_basins": len(payload["splits"]),
        },
        "forcing": forcing_stats,
        "attr": attr_stats,
        **target_stats,
    }


def physical_flow_scale(attr_raw: pd.DataFrame, observed: dict,
                        fit_basins, all_basins) -> tuple:
    """仅用属性推出的径流尺度，不使用任何径流观测。

    动机
    ----
    默认的逐流域归一化用**实测 Q 的均值与标准差**，因此即使某流域的径流训练标签
    被全删，模型仍间接知道该流域流量的量级与变幅——而真正无资料流域拿不到这两个
    数。用本函数替代后，留出流域的目标尺度完全由属性推出，零径流观测。

    方法
    ----
    ``proxy = 流域面积 × 日均降水``，正比于年均径流体积。在 log–log 空间对
    观测的均值与标准差各拟合一条回归线。实测精度：均值 R²=0.987、中位相对误差
    12.5%；标准差 R²=0.953、误差 21.6%。

    Parameters
    ----------
    attr_raw : pd.DataFrame
        **未标准化**的属性表，需含 ``area`` 与 ``p_mean`` 两列。
    observed : dict
        {basin: {"mean": .., "std": ..}}，实测统计量，仅用于拟合回归。
    fit_basins : list
        参与拟合的流域。**必须只含该情景下仍有径流标签的流域**，否则又构成泄漏；
        本函数会断言 fit_basins ⊆ all_basins 且非空，越界由调用方保证。
    all_basins : list
        需要输出尺度的全部流域，顺序即返回数组的顺序。

    Returns
    -------
    (mean_array, std_array, info)
    """
    fit_basins = [str(b) for b in fit_basins]
    all_basins = [str(b) for b in all_basins]
    if not fit_basins:
        raise ValueError("physical_flow_scale 需要至少一个有径流标签的流域来拟合尺度")
    unknown = set(fit_basins) - set(all_basins)
    if unknown:
        raise ValueError(f"fit_basins 含未知流域: {sorted(unknown)[:5]}")

    for col in ("area", "p_mean"):
        if col not in attr_raw.columns:
            raise ValueError(f"物理尺度需要属性列 {col}")
    proxy = (attr_raw["area"].astype(float) * attr_raw["p_mean"].astype(float))
    if (proxy <= 0).any():
        raise ValueError("面积×降水出现非正值，无法取对数")
    log_proxy = np.log(proxy)

    out, info = {}, {}
    for key in ("mean", "std"):
        y = np.array([observed[b][key] for b in fit_basins], dtype="float64")
        if (y <= 0).any():
            raise ValueError(f"实测 {key} 出现非正值，无法在 log 空间拟合")
        x = log_proxy.reindex(fit_basins).to_numpy(dtype="float64")
        slope, intercept = np.polyfit(x, np.log(y), 1)
        pred = np.exp(intercept + slope * log_proxy.reindex(all_basins).to_numpy("float64"))
        out[key] = pred.astype("float32")
        resid = np.log(y) - (intercept + slope * x)
        info[key] = {"slope": float(slope), "intercept": float(intercept),
                     "r2": float(1 - resid.var() / np.log(y).var()),
                     "n_fit": len(fit_basins)}

    std = np.maximum(out["std"], _EPS).astype("float32")
    return out["mean"], std, info


def save_norm_stats(stats: dict, path: Path = NORM_STATS_FILE) -> Path:
    SPLITS_DIR.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(stats, indent=2, ensure_ascii=False), encoding="utf-8")
    return path


def load_norm_stats(path: Path = NORM_STATS_FILE) -> dict:
    if not path.exists():
        raise FileNotFoundError(
            f"归一化统计量不存在: {path}\n请先运行 python src/pipeline/normalization.py"
        )
    return json.loads(path.read_text(encoding="utf-8"))


if __name__ == "__main__":
    stats = compute_norm_stats()
    path = save_norm_stats(stats)
    print(f"强迫统计量（训练段汇总 {stats['meta']['n_forcing_steps_pooled']} 步）:")
    for name, v in stats["forcing"].items():
        print(f"  {name:<20s} mean={v['mean']:>10.4f}  std={v['std']:>10.4f}")
    n_onehot = sum(1 for v in stats["attr"].values() if v["onehot"])
    print(f"属性列: {len(stats['attr'])}（其中 one-hot {n_onehot} 列不标准化）")
    for task in TASKS:
        ns = [v["n"] for v in stats[task].values()]
        print(f"{task}: {len(ns)} 个流域，训练段有效标签数 中位 {int(np.median(ns))}，最小 {min(ns)}")
    print(f"已写入: {path}")
