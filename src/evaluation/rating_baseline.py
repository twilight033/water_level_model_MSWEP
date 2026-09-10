"""率定曲线相关的对照分析——回应"你不就是在学率定曲线吗"这一质疑。

背景
----
USGS 的流量本就由水位经率定曲线换算而来，因此审稿人必然会问：本文声称
"水位标签能替代径流标签"，是否只是同义反复？

本模块提供三项分析，全部不需要训练：

1. ``rating_stability``：逐流域自己的率定关系稳不稳。用训练期拟合、测试期
   应用，与"测试期自拟合"的上界比较。**不要指望靠"关系不稳"脱身**——实测
   中位 NSE 0.897、相比上界仅损失 0.019，老实承认它稳定才是稳的打法。
2. ``regionalized_rating``：只用**保留流域**的训练期数据拟合一条全局
   h→Q 曲线，迁移到留出流域。**推理时需要实测水位输入**，因此它解决的
   是另一个问题（把当下实测水位换算成流量），不能用天气预报驱动、不能
   重建已停测站点，而且建立率定曲线本身就需要成对实测流量——恰是本文
   假设不存在的东西。这个数必须主动报告，藏起来只会更被动。
3. 两阶段基线见 ``two_stage_baseline.py``，那才是真正的竞争者。

拟合方式：归一化空间的分位数分箱非参数单调拟合。之所以在归一化空间做，
是因为率定曲线是逐站的物理量纲关系（大河与小溪不在一个量级），标准化后
才可跨流域合并——它描述的是"水位比该流域常年水平高 N 个标准差时，流量
大约高多少个标准差"。
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

_ROOT = Path(__file__).resolve().parents[2]
for _p in (str(_ROOT / "src"), str(_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from pipeline.paths import RESULTS_ROOT  # noqa: E402

N_BINS = 50
MIN_PAIRS_FIT = 1000
MIN_PAIRS_EVAL = 500


def nse(obs: np.ndarray, pred: np.ndarray) -> float:
    ok = np.isfinite(obs) & np.isfinite(pred)
    if ok.sum() < 50:
        return float("nan")
    o, p = obs[ok], pred[ok]
    denom = ((o - o.mean()) ** 2).sum()
    return float(1 - ((p - o) ** 2).sum() / denom) if denom > 0 else float("nan")


def fit_rating(h: np.ndarray, q: np.ndarray, n_bins: int = N_BINS):
    """按 h 的分位数分箱、每箱取 q 均值的非参数单调拟合。"""
    edges = np.unique(np.quantile(h, np.linspace(0, 1, n_bins + 1)))
    if edges.size < 3:
        return None
    idx = np.clip(np.searchsorted(edges, h, "right") - 1, 0, edges.size - 2)
    table = np.full(edges.size - 1, np.nan)
    for b in range(edges.size - 1):
        sel = idx == b
        if sel.any():
            table[b] = q[sel].mean()
    return edges, table


def apply_rating(fit, h: np.ndarray) -> np.ndarray:
    edges, table = fit
    idx = np.clip(np.searchsorted(edges, h, "right") - 1, 0, edges.size - 2)
    return table[idx]


def _pairs(prepared, basin: str, split: str):
    bi = prepared.basin_index[basin]
    lo, hi = prepared.split_range(basin, split)
    q = prepared.targets["flow"][bi, lo:hi + 1].astype(float)
    h = prepared.targets["waterlevel"][bi, lo:hi + 1].astype(float)
    ok = np.isfinite(q) & np.isfinite(h)
    return h[ok], q[ok]


def rating_stability(prepared) -> pd.DataFrame:
    """逐流域率定关系的时间稳定性。"""
    rows = []
    for basin in prepared.splits["splits"]:
        h_tr, q_tr = _pairs(prepared, basin, "train")
        h_te, q_te = _pairs(prepared, basin, "test")
        if h_tr.size < MIN_PAIRS_FIT or h_te.size < MIN_PAIRS_EVAL:
            continue
        fit_tr = fit_rating(h_tr, q_tr)
        fit_te = fit_rating(h_te, q_te)
        if fit_tr is None or fit_te is None:
            continue
        rows.append({
            "basin": basin,
            "n_train_pairs": int(h_tr.size), "n_test_pairs": int(h_te.size),
            "train_fit_test_apply": nse(q_te, apply_rating(fit_tr, h_te)),
            "test_self_fit_upper_bound": nse(q_te, apply_rating(fit_te, h_te)),
        })
    df = pd.DataFrame(rows)
    df["drift_loss"] = df["test_self_fit_upper_bound"] - df["train_fit_test_apply"]
    return df


def regionalized_rating(prepared, held_out: set, fit_basins=None) -> pd.DataFrame:
    """只用保留流域拟合一条全局曲线，应用于留出流域的测试期实测水位。"""
    fit_basins = fit_basins or [b for b in prepared.splits["splits"] if b not in held_out]
    hs, qs = [], []
    for basin in fit_basins:
        h, q = _pairs(prepared, basin, "train")
        if h.size:
            hs.append(h)
            qs.append(q)
    if not hs:
        return pd.DataFrame()
    fit = fit_rating(np.concatenate(hs), np.concatenate(qs))
    if fit is None:
        return pd.DataFrame()

    rows = []
    for basin in sorted(held_out):
        h, q = _pairs(prepared, basin, "test")
        if h.size < MIN_PAIRS_EVAL:
            continue
        rows.append({"basin": basin, "n_pairs": int(h.size),
                     "nse": nse(q, apply_rating(fit, h))})
    return pd.DataFrame(rows)


def regionalized_rating_over_seeds(prepared, ratio: float = 0.70,
                                   mask_seeds=(42, 123, 456)) -> pd.DataFrame:
    """在各掩膜种子下重复区域化率定，逐流域取平均。"""
    from pipeline.masking import get_hidden

    collected = {}
    for seed in mask_seeds:
        _, stats = get_hidden(prepared, {"flow": ratio},
                             mechanism="basin_holdout", mask_seed=seed)
        held = set(stats["per_task"]["flow"]["held_out_basins"])
        for row in regionalized_rating(prepared, held).itertuples():
            collected.setdefault(row.basin, []).append(row.nse)
    return pd.DataFrame([{"basin": b, "nse": float(np.mean(v)), "n_seeds": len(v)}
                         for b, v in collected.items()])


def main():
    from pipeline.dataset import PreparedData

    out_dir = RESULTS_ROOT / "summary"
    out_dir.mkdir(parents=True, exist_ok=True)
    prep = PreparedData()

    stab = rating_stability(prep)
    stab.to_csv(out_dir / "rating_stability.csv", index=False, encoding="utf-8-sig")
    print(f"率定关系稳定性（{len(stab)} 个流域，归一化空间预测径流的 NSE）:")
    print(stab[["train_fit_test_apply", "test_self_fit_upper_bound", "drift_loss"]]
          .describe(percentiles=[.1, .5, .9]).round(3).to_string())
    print(f"  训练期拟合/测试期应用 NSE<0.5 的流域: "
          f"{int((stab.train_fit_test_apply < 0.5).sum())}/{len(stab)}")
    print(f"  已写入 {out_dir / 'rating_stability.csv'}")

    for ratio in (0.30, 0.50, 0.70):
        reg = regionalized_rating_over_seeds(prep, ratio)
        path = out_dir / f"regionalized_rating_holdout{int(ratio * 100)}.csv"
        reg.to_csv(path, index=False, encoding="utf-8-sig")
        print(f"\n区域化率定曲线（留出 {ratio:.0%}，需实测水位输入）: "
              f"n={len(reg)}  均值 {reg.nse.mean():.4f}  中位 {reg.nse.median():.4f}")
        print(f"  已写入 {path}")


if __name__ == "__main__":
    main()
