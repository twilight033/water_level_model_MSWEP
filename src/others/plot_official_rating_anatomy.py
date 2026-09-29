"""图解 USGS 官方率定曲线的构造：折点、偏移量与对数内插。

USGS 的 `base` 文件只给三样东西：若干**折点** $(h_i, Q_i)$、一个**偏移量** $h_0$
（RATING OFFSET，河床的"有效零点"）、以及展开方式（RATING EXPANSION="logarithmic"）。
两折点之间在 $\\log(h-h_0)$–$\\log Q$ 平面上用直线相连，等价于分段幂律

    Q = C_i (h - h_0)^{beta_i}

折点不是实测点，而是描述这条曲线所需的**最少控制点**——本例 5 个折点的四段
指数几乎相同（约 2.96），说明整条曲线在对数空间就是一条直线。

图中同时叠上 CAMELSH 的实测配对点：它们紧贴曲线（该站 NSE 0.996），直观印证
"CAMELSH 的径流由水位经这条曲线换算而来"。
"""

import sys
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parents[2]
for _p in (str(_ROOT / "src"), str(_ROOT),
           str(_ROOT / "hydro-multitask-paper" / "figures" / "revision" / "scripts")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from figcommon_rev import DOUBLE_COL, DUAL, INK, INK_SOFT, RATING, set_style  # noqa: E402
from evaluation.official_rating_check import (CACHE_DIR, FT_PER_M, apply_official_rating,  # noqa: E402
                                              parse_base_rating)
from pipeline.paths import RESULTS_ROOT  # noqa: E402

DEMO = "01085000"                 # CONTOOCOOK RIVER NEAR HENNIKER, NH；5 个折点，NSE 0.996
OUT_DIR = RESULTS_ROOT / "figures"


def segment_exponents(rating: dict) -> np.ndarray:
    """各段幂律指数 beta = dlog(Q) / dlog(h - h0)。"""
    h0 = rating["offsets"].get(1, 0.0)
    x, y = rating["indep"] - h0, rating["dep"]
    return np.diff(np.log(y)) / np.diff(np.log(x))


def observed_pairs(prepared, basin: str):
    """测试段的实测配对，换算到官方口径（ft, cfs）。"""
    bi = prepared.basin_index[basin]
    lo, hi = prepared.split_range(basin, "test")
    q = prepared.targets_raw["flow"][bi, lo:hi + 1].astype(float)
    h = prepared.targets_raw["waterlevel"][bi, lo:hi + 1].astype(float)
    ok = np.isfinite(q) & np.isfinite(h)
    return h[ok] * FT_PER_M, q[ok] / 0.0283168466


def main():
    from pipeline.dataset import PreparedData

    set_style("zh")
    text = (CACHE_DIR / f"{DEMO}.base.rdb").read_text(encoding="utf-8", errors="replace")
    rating = parse_base_rating(text)
    h0 = rating["offsets"].get(1, 0.0)
    hk, qk = rating["indep"], rating["dep"]
    betas = segment_exponents(rating)

    prepared = PreparedData()
    h_obs, q_obs = observed_pairs(prepared, DEMO)

    # 官方曲线：在折点范围内密集采样画光滑线
    h_line = np.linspace(hk.min(), hk.max(), 600)
    q_line = apply_official_rating(h_line / FT_PER_M, rating) / 0.0283168466

    fig, axes = plt.subplots(1, 2, figsize=(DOUBLE_COL, DOUBLE_COL * 0.44))

    # ── (a) 线性空间：看到的是一条上凹的幂律曲线 ──────────────────────────
    ax = axes[0]
    ax.scatter(h_obs, q_obs, s=2.0, color=RATING, alpha=0.18, linewidths=0, zorder=2,
               label=f"CAMELSH 实测配对（n={len(h_obs):,}）")
    ax.plot(h_line, q_line, color=INK, lw=1.5, zorder=4, label="官方率定曲线")
    ax.plot(hk, qk, "o", ms=5.5, mfc="white", mec=DUAL, mew=1.6, zorder=6,
            label=f"折点（{len(hk)} 个）")
    ax.axvline(h0, color=INK_SOFT, lw=0.8, ls="--", zorder=3)
    ax.annotate(f"偏移量 $h_0$ = {h0:g} ft\n（有效零点）", xy=(h0, ax.get_ylim()[1] * 0.62),
                xytext=(h0 + 0.5, ax.get_ylim()[1] * 0.62), fontsize=6.5,
                color=INK_SOFT, va="center")
    for x, y in zip(hk, qk):
        ax.annotate(f"({x:g}, {y:g})", xy=(x, y), xytext=(6, -8),
                    textcoords="offset points", fontsize=6, color=DUAL)
    ax.set_xlim(h0 - 0.6, hk.max() * 1.04)
    ax.set_title("(a) 线性空间：分段幂律", loc="left")
    ax.set_xlabel("水位 Gage height (ft)")
    ax.set_ylabel("径流 Discharge (ft³/s)")
    ax.legend(loc="upper left", markerscale=3)

    # ── (b) 对数空间：内插实际发生的地方，折点连成直线 ────────────────────
    ax = axes[1]
    keep = h_obs > h0
    ax.scatter(h_obs[keep] - h0, q_obs[keep], s=2.0, color=RATING, alpha=0.18,
               linewidths=0, zorder=2)
    ax.plot(h_line - h0, q_line, color=INK, lw=1.5, zorder=4)
    ax.plot(hk - h0, qk, "o", ms=5.5, mfc="white", mec=DUAL, mew=1.6, zorder=6)
    ax.set_xscale("log")
    ax.set_yscale("log")
    for i, beta in enumerate(betas):
        xm = np.sqrt((hk[i] - h0) * (hk[i + 1] - h0))
        ym = np.sqrt(qk[i] * qk[i + 1])
        ax.annotate(rf"$\beta$={beta:.2f}", xy=(xm, ym), xytext=(-2, 9),
                    textcoords="offset points", fontsize=6.5, color=INK_SOFT, ha="right")
    ax.set_title("(b) 对数空间：折点连成直线", loc="left")
    ax.set_xlabel("$h - h_0$ (ft)，对数轴")
    ax.set_ylabel("径流 Discharge (ft³/s)，对数轴")
    ax.text(0.96, 0.06, r"$Q = C\,(h-h_0)^{\beta}$", transform=ax.transAxes,
            fontsize=8, color=INK, ha="right", va="bottom")

    for ax in axes:
        ax.grid(True, which="both", lw=0.35, color=RATING, alpha=0.25)
        ax.set_axisbelow(True)

    station = "CONTOOCOOK RIVER NEAR HENNIKER, NH"
    fig.suptitle(f"USGS 官方率定曲线的构造（{DEMO} {station}，rating 23.0）",
                 fontsize=8.5, y=0.99)
    fig.tight_layout(rect=(0, 0, 1, 0.94))

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUT_DIR / "official_rating_anatomy_zh.png"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    fig.savefig(out.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)

    print(f"折点 {len(hk)} 个，偏移量 h0 = {h0:g} ft")
    for i, beta in enumerate(betas):
        print(f"  第 {i+1} 段 ({hk[i]:g} → {hk[i+1]:g} ft): beta = {beta:.4f}")
    print(f"  四段指数极差 {betas.max() - betas.min():.4f}"
          f" —— 越小说明整条曲线在对数空间越接近单一直线")
    print(f"  已写入 {out}")


if __name__ == "__main__":
    main()
