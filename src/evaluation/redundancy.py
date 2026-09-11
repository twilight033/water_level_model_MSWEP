"""逐流域的 Q–h 冗余度分析。

动机
----
同一断面的径流与水位通过率定曲线近似单调对应，h ≈ f(Q)。多任务学习的收益
来自"相关但不同"的任务能提供互补信息；当第二个任务几乎是第一个的确定性
变换时，它提供的新信息接近零。因此"联合学习 Q 与 h 无收益"在理论上是可
预期的，单凭这一条零结果不构成发现。

真正有价值的问法是：**多任务的收益是否只出现在冗余被打破的流域**——存在
绳套效应（同一流量下涨水段与退水段水位不同）、回水顶托、结冰影响、低比降
断面的地方，h 携带 Q 以外的信息。本模块给出两个逐流域指标，供与逐流域多
任务增益做相关分析：

- ``nonredundancy``：用 Q 单调预测 h 之后剩余的方差占比（1 - R²）。越大说明
  h 越不能被 Q 解释，理论上多任务越可能有收益。
- ``hysteresis``：同一流量区间内涨水段与退水段水位均值之差，按 h 的标准差
  归一化。绳套效应的直接度量。

两个指标都只用**训练段**的成对观测计算，避免用测试期信息解释测试期表现。
实现不引入新依赖：用分位数分箱做非参数单调拟合，而不是 sklearn 的保序回归。
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

_ROOT = Path(__file__).resolve().parents[2]
for _p in (str(_ROOT / "src"), str(_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from evaluation.run_config import SUMMARY_DIR, load_summary, select  # noqa: E402
from pipeline.paths import COVERAGE_DIR  # noqa: E402

N_BINS = 50
MIN_PAIRS = 500          # 成对观测少于此数的流域不计算，避免噪声结论
MIN_PER_LIMB = 20        # 每个箱内涨/退水段各自至少这么多点才计入绳套指标


def _binned_monotone_fit(q: np.ndarray, h: np.ndarray, n_bins: int = N_BINS) -> tuple:
    """按 Q 的分位数分箱，用箱内 h 均值作为预测，返回 (预测值, 箱编号)。

    这是一个非参数拟合：不假设率定曲线的函数形式，只假设 h 主要由 Q 决定。
    """
    edges = np.quantile(q, np.linspace(0, 1, n_bins + 1))
    edges = np.unique(edges)
    if edges.size < 3:
        return None, None
    idx = np.clip(np.searchsorted(edges, q, side="right") - 1, 0, edges.size - 2)
    pred = np.full(q.shape, np.nan)
    for b in range(edges.size - 1):
        sel = idx == b
        if sel.any():
            pred[sel] = h[sel].mean()
    return pred, idx


def basin_redundancy(q: np.ndarray, h: np.ndarray) -> dict:
    """单个流域的冗余度与绳套指标。q、h 为训练段内成对有效的观测。"""
    if q.size < MIN_PAIRS:
        return {"n_pairs": int(q.size)}

    pred, bin_idx = _binned_monotone_fit(q, h)
    if pred is None:
        return {"n_pairs": int(q.size)}

    total_var = float(((h - h.mean()) ** 2).sum())
    resid_var = float(((h - pred) ** 2).sum())
    r2 = 1.0 - resid_var / total_var if total_var > 0 else np.nan
    h_std = float(h.std())

    # 绳套效应：同一流量箱内，涨水段与退水段的水位均值之差
    dq = np.diff(q, prepend=q[0])
    rising, falling = dq > 0, dq < 0
    gaps = []
    for b in np.unique(bin_idx):
        sel = bin_idx == b
        up, down = sel & rising, sel & falling
        if up.sum() >= MIN_PER_LIMB and down.sum() >= MIN_PER_LIMB:
            gaps.append(h[up].mean() - h[down].mean())
    hysteresis = float(np.mean(np.abs(gaps)) / h_std) if gaps and h_std > 0 else np.nan

    return {
        "n_pairs": int(q.size),
        "r2_h_given_q": float(r2),
        "nonredundancy": float(1.0 - r2) if np.isfinite(r2) else np.nan,
        "hysteresis": hysteresis,
        "n_hysteresis_bins": len(gaps),
        "spearman_q_h": float(pd.Series(q).corr(pd.Series(h), method="spearman")),
        "h_std": h_std,
    }


def build_redundancy_table(prepared=None) -> pd.DataFrame:
    """对全部流域计算冗余度指标。"""
    if prepared is None:
        from pipeline.dataset import PreparedData
        prepared = PreparedData()

    rows = []
    for basin in prepared.splits["splits"]:
        bi = prepared.basin_index[basin]
        lo, hi = prepared.split_range(basin, "train")
        q = prepared.targets_raw["flow"][bi, lo:hi + 1]
        h = prepared.targets_raw["waterlevel"][bi, lo:hi + 1]
        both = np.isfinite(q) & np.isfinite(h)
        rows.append({"basin": basin, **basin_redundancy(q[both], h[both])})
    return pd.DataFrame(rows)


def correlate_with_gain(redundancy: pd.DataFrame, metrics: pd.DataFrame,
                        task: str = "flow", metric: str = "nse",
                        config: str = "base", scenario: str = "complete",
                        held_out=None) -> dict:
    """检验多任务增益是否与 h 相对 Q 的冗余程度相关。

    增益定义为逐流域 (双头 - 对应单任务) 的差值，先在模型/掩膜种子上取平均。
    这是把"多任务没用"变成"多任务在什么条件下有用"的关键一步。

    必须限定在同一可比配置内：混入 4D（不同窗口）、4F（不同流域子集）、
    4G（不同任务权重）时，双头有这些配置的运行而单任务没有，算出来的"增益"
    其实是配置差异——实测会把一个 ρ=-0.006、p=0.96 的零结果变成 ρ=+0.18。

    ``r2_h_given_q`` 与 ``nonredundancy`` 互为 1 减，相关系数符号相反：前者是
    "h 作为 Q 的代理有多好"，替代命题要看的是它；只报后者容易把方向讲反。
    """
    from scipy import stats as sps

    single = {"flow": "single_flow", "waterlevel": "single_waterlevel"}[task]
    sub = select(metrics, config=config, scenario=scenario, task=task,
                 held_out=held_out)
    dual = sub[sub["architecture"] == "dual_head"].groupby("basin")[metric].mean()
    base = sub[sub["architecture"] == single].groupby("basin")[metric].mean()
    common = dual.index.intersection(base.index)
    gain = (dual.loc[common] - base.loc[common]).rename("gain")

    joined = redundancy.set_index("basin").join(gain, how="inner").dropna(subset=["gain"])

    out = {"config": config, "scenario": scenario, "task": task, "metric": metric,
           "held_out": held_out, "n": int(len(joined)),
           "mean_gain": float(joined["gain"].mean()) if len(joined) else float("nan")}
    if len(joined) < 5:
        return out
    for col in ("r2_h_given_q", "nonredundancy", "hysteresis"):
        vals = joined[[col, "gain"]].dropna()
        if len(vals) < 5:
            continue
        rho, p = sps.spearmanr(vals[col], vals["gain"])
        out[f"spearman_{col}"] = float(rho)
        out[f"p_{col}"] = float(p)
    return out


def gain_correlation_table(redundancy: pd.DataFrame, metrics: pd.DataFrame,
                           config: str = "base") -> pd.DataFrame:
    """完整标签与三档按流域留出各算一行。

    留出情景只取**被留出流域**——那里的径流训练标签被整段删除，增益才是
    "用水位替代径流"的直接度量；保留流域上的差值不回答这个问题。
    """
    rows = [correlate_with_gain(redundancy, metrics, task=task, config=config)
            for task in ("flow", "waterlevel")]
    for ratio in (30, 50, 70):
        rows.append(correlate_with_gain(redundancy, metrics, task="flow",
                                        config=config,
                                        scenario=f"q_holdout{ratio}", held_out=True))
    return pd.DataFrame([r for r in rows if r.get("n", 0) >= 5])


def main():
    table = build_redundancy_table()
    COVERAGE_DIR.mkdir(parents=True, exist_ok=True)
    path = COVERAGE_DIR / "basin_qh_redundancy.csv"
    table.to_csv(path, index=False, encoding="utf-8-sig")
    print(f"冗余度表: {path}  （{len(table)} 个流域）")

    ok = table.dropna(subset=["nonredundancy"])
    print()
    print(f"可用流域 {len(ok)}/{len(table)}")
    print("h 相对 Q 的非冗余度（1 - R²，越大说明 h 越携带 Q 以外的信息）:")
    print(ok["nonredundancy"].describe().round(4).to_string())
    print()
    print("绳套指标（涨退水段水位差 / h 标准差）:")
    print(ok["hysteresis"].describe().round(4).to_string())

    if not (SUMMARY_DIR / "all_metrics.csv").exists():
        print()
        print("尚无训练结果，跑完矩阵后重新运行本脚本即可得到相关性分析")
        return

    metrics, _ = load_summary()
    # 每个含留出情景的配置各算一份：base 与 physical 下的相关性可能不同
    configs = sorted(set(metrics[metrics["scenario"].str.startswith("q_holdout")]["config"]))
    corr = pd.concat([gain_correlation_table(table, metrics, config=c) for c in configs],
                     ignore_index=True) if configs else pd.DataFrame()
    if corr.empty:
        print()
        print("可配对的运行不足，跳过相关性分析")
        return
    out = SUMMARY_DIR / "redundancy_gain_correlation.csv"
    corr.to_csv(out, index=False, encoding="utf-8-sig")
    print()
    print("多任务增益与 Q–h 冗余度的相关性（同一可比配置内）:")
    cols = [c for c in ("config", "scenario", "task", "held_out", "n", "mean_gain",
                        "spearman_r2_h_given_q", "p_r2_h_given_q",
                        "spearman_hysteresis", "p_hysteresis") if c in corr]
    print(corr[cols].round(4).to_string(index=False))
    print(f"  已写入 {out}")


if __name__ == "__main__":
    main()
