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
    attr_arr, attr_cols, onehot_cols = load_attributes(basins)

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
            "onehot_columns": onehot_cols,
            "n_forcing_steps_pooled": int(total_n),
            "n_basins": len(payload["splits"]),
        },
        "forcing": forcing_stats,
        "attr": attr_stats,
        **target_stats,
    }


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
