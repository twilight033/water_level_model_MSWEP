"""配对统计检验与不确定性汇总。

回应审稿意见 Major 2：86 个流域构成配对结构，不能只用一次运行的平均
NSE 差异判断模型优劣。本模块提供：

- 逐流域差值的 Wilcoxon 符号秩检验（不假设正态）；
- 差值均值与中位数的 bootstrap 置信区间；
- 效应量（配对 Cohen's d 与秩双列相关 rank-biserial）；
- 跨种子汇总，且**模型种子方差与掩膜种子方差分开报告**——原实现的
  误差棒只反映固定模型种子下的掩膜变动，不能代表跨模型种子的不确定性。
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

_ROOT = Path(__file__).resolve().parents[2]
for _p in (str(_ROOT / "src"), str(_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

BOOTSTRAP_N = 10000


def paired_differences(df_a: pd.DataFrame, df_b: pd.DataFrame, task: str,
                       metric: str = "nse") -> pd.DataFrame:
    """按流域对齐两组结果，返回逐流域差值 (a - b)。"""
    # 用 groupby 而非 set_index：同一流域若有多行（多种子、或不慎混入多个配置），
    # set_index 会产生重复索引，.loc[common] 随之膨胀，构造 DataFrame 时抛
    # "All arrays must be of the same length"——该异常此前被上层 except 吞掉，
    # 导致"双头 vs 单任务"这组关键对比被静默丢弃。
    a = df_a[df_a["task"] == task].groupby("basin")[metric].mean()
    b = df_b[df_b["task"] == task].groupby("basin")[metric].mean()
    common = a.index.intersection(b.index)
    out = pd.DataFrame({"basin": common, f"{metric}_a": a.loc[common].to_numpy(),
                        f"{metric}_b": b.loc[common].to_numpy()})
    out["diff"] = out[f"{metric}_a"] - out[f"{metric}_b"]
    return out[np.isfinite(out["diff"])].reset_index(drop=True)


def bootstrap_ci(values: np.ndarray, statistic=np.mean, n_boot: int = BOOTSTRAP_N,
                 alpha: float = 0.05, seed: int = 0) -> tuple:
    """对配对差值做 bootstrap，返回 (点估计, 下界, 上界)。"""
    rng = np.random.default_rng(seed)
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if values.size < 2:
        return float("nan"), float("nan"), float("nan")
    idx = rng.integers(0, values.size, size=(n_boot, values.size))
    boot = statistic(values[idx], axis=1)
    lo, hi = np.percentile(boot, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(statistic(values)), float(lo), float(hi)


def paired_test(diffs: np.ndarray, seed: int = 0) -> dict:
    """Wilcoxon 符号秩检验 + bootstrap 置信区间 + 效应量。"""
    from scipy import stats as sps

    d = np.asarray(diffs, dtype=float)
    d = d[np.isfinite(d)]
    if d.size < 3:
        return {"n": int(d.size)}

    nonzero = d[d != 0]
    if nonzero.size >= 3:
        stat, p = sps.wilcoxon(nonzero, alternative="two-sided")
        # 秩双列相关：把检验统计量换算成 [-1, 1] 的效应量
        n = nonzero.size
        total_rank = n * (n + 1) / 2
        rank_biserial = float(2 * stat / total_rank - 1)
    else:
        stat, p, rank_biserial = float("nan"), float("nan"), float("nan")

    mean, mean_lo, mean_hi = bootstrap_ci(d, np.mean, seed=seed)
    med, med_lo, med_hi = bootstrap_ci(d, np.median, seed=seed)
    cohens_d = float(d.mean() / d.std(ddof=1)) if d.std(ddof=1) > 0 else float("nan")

    return {
        "n": int(d.size),
        "mean_diff": mean, "mean_ci_lo": mean_lo, "mean_ci_hi": mean_hi,
        "median_diff": med, "median_ci_lo": med_lo, "median_ci_hi": med_hi,
        "wilcoxon_stat": float(stat) if np.isfinite(stat) else float("nan"),
        "wilcoxon_p": float(p) if np.isfinite(p) else float("nan"),
        "cohens_d_paired": cohens_d,
        "rank_biserial": rank_biserial,
        "n_basins_improved": int((d > 0).sum()),
        "n_basins_worsened": int((d < 0).sum()),
    }


def compare_models(runs: pd.DataFrame, model_a: str, model_b: str, task: str,
                   metric: str = "nse", by_seed: bool = True) -> dict:
    """比较两个架构。

    runs 需含列: architecture, model_seed, basin, task, <metric>。
    by_seed=True 时先在同一 model_seed 内配对（同种子同划分，最严格），
    再把逐流域差值合并做检验。
    """
    a = runs[runs["architecture"] == model_a]
    b = runs[runs["architecture"] == model_b]
    if a.empty or b.empty:
        raise ValueError(f"缺少 {model_a} 或 {model_b} 的结果")

    pieces = []
    if by_seed:
        seeds = sorted(set(a["model_seed"]) & set(b["model_seed"]))
        for s in seeds:
            piece = paired_differences(a[a["model_seed"] == s], b[b["model_seed"] == s],
                                       task, metric)
            piece["model_seed"] = s
            pieces.append(piece)
    else:
        ma = a.groupby(["basin", "task"], as_index=False)[metric].mean()
        mb = b.groupby(["basin", "task"], as_index=False)[metric].mean()
        pieces.append(paired_differences(ma, mb, task, metric))

    merged = pd.concat(pieces, ignore_index=True)
    # 每个流域先在种子间取平均，再做跨流域配对检验，避免同一流域被重复计数
    per_basin = merged.groupby("basin", as_index=False)["diff"].mean()
    result = paired_test(per_basin["diff"].to_numpy())
    result.update({"model_a": model_a, "model_b": model_b, "task": task,
                   "metric": metric, "n_seeds": merged["model_seed"].nunique()
                   if "model_seed" in merged else 1})
    return {"summary": result, "per_basin": per_basin, "raw": merged}


def seed_variance(runs: pd.DataFrame, metric: str = "nse") -> pd.DataFrame:
    """模型种子方差与掩膜种子方差分开汇总。

    对每个 (architecture, scenario, task)：
    - across_model_seeds: 固定掩膜种子时，跨模型种子的流域平均指标标准差；
    - across_mask_seeds:  固定模型种子时，跨掩膜种子的流域平均指标标准差。
    """
    key = ["architecture", "scenario", "task"]
    basin_mean = (runs.groupby(key + ["model_seed", "mask_seed"], as_index=False)[metric]
                  .mean())

    rows = []
    for keys, sub in basin_mean.groupby(key):
        by_model = sub.groupby("mask_seed")[metric].std(ddof=1).mean()
        by_mask = sub.groupby("model_seed")[metric].std(ddof=1).mean()
        rows.append({
            **dict(zip(key, keys)),
            "mean": float(sub[metric].mean()),
            "std_across_model_seeds": float(by_model) if np.isfinite(by_model) else float("nan"),
            "std_across_mask_seeds": float(by_mask) if np.isfinite(by_mask) else float("nan"),
            "n_runs": int(len(sub)),
        })
    return pd.DataFrame(rows)


def degradation(runs: pd.DataFrame, baseline_scenario: str = "complete",
                metric: str = "nse") -> pd.DataFrame:
    """每个模型相对**自身**完整标签基线的下降幅度。

    这是审稿意见 Major 3 的核心比较口径：不能拿多任务的缺失结果去比
    多任务自己的完整基线就下结论，必须同时看单任务模型在同样缺失下
    相对其自身基线的下降。
    """
    base = (runs[runs["scenario"] == baseline_scenario]
            .groupby(["architecture", "task", "basin"], as_index=False)[metric]
            .mean().rename(columns={metric: "baseline"}))
    other = (runs[runs["scenario"] != baseline_scenario]
             .groupby(["architecture", "scenario", "task", "basin"], as_index=False)[metric]
             .mean())
    merged = other.merge(base, on=["architecture", "task", "basin"], how="inner")
    merged["degradation"] = merged["baseline"] - merged[metric]
    return merged
