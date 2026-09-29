"""画出项目里率定曲线的实际形状——分位数分箱求均值得到的 50 级阶梯查表。

动机
----
论文的 Fig 6 只对比率定路径与双头的 NSE，没有任何一张图画出**这条曲线本身**长什么样。
被问到"现在用的率定曲线是怎么画的"时缺少可看的东西，故补一张说明图（非论文图）。

两个面板对应项目里两种口径：

- (a) 单流域自拟合：`rating_stability()` 用它做"训练期拟合、测试期应用"的时间稳定性检验；
- (b) 区域化全局曲线：`two_stage_baseline` 与 `regionalized_rating()` 用它——把所有流域的
  配对点混在一起拟合**一条**曲线，再迁移到留出流域。

之所以能把不同量级的河流混在一起，是因为 h 与 q 都已按逐流域训练段的均值/标准差做过
z-score；曲线的含义是"水位比该流域常年水平高 N 个标准差时，径流约高多少个标准差"。

配色：只用灰阶与墨色——单一序列靠明度区分，无色相依赖，天然对色觉缺陷安全，
不需要跑分类色板验证器。数据云取低透明度保持退让，拟合曲线用墨色突出。
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

from figcommon_rev import DOUBLE_COL, INK, INK_SOFT, RATING, set_style  # noqa: E402
from evaluation.rating_baseline import N_BINS, _pairs, apply_rating, fit_rating, nse  # noqa: E402
from pipeline.paths import RESULTS_ROOT  # noqa: E402

DEMO_BASIN = "01017000"          # Aroostook River at Washburn, ME；训练段 18 821 个配对点
OUT_DIR = RESULTS_ROOT / "figures"

# 少数流域的归一化水位有极端离群（可达 15 个标准差），不截断会把主体压成一条竖线。
# 截断只影响显示范围，曲线本身仍在全量数据上拟合。
CLIP_Q = (0.001, 0.999)


def _clip_limits(values: np.ndarray, pad: float = 0.06) -> tuple:
    lo, hi = np.quantile(values, CLIP_Q)
    margin = (hi - lo) * pad
    return lo - margin, hi + margin


def _panel_single(ax, prepared):
    """(a) 单流域：散点云 + 50 级阶梯，并标出分箱边界。"""
    h, q = _pairs(prepared, DEMO_BASIN, "train")
    edges, table = fit_rating(h, q)

    # 分箱边界：50 个箱画 51 条线太密，每 5 个画一条示意"等样本量分箱"
    for edge in edges[::5]:
        ax.axvline(edge, color=RATING, lw=0.35, alpha=0.45, zorder=1)

    # baseline=None：否则 stairs 会从末端画一条竖线回到 y=0，看着像伪影
    ax.scatter(h, q, s=2.0, color=RATING, alpha=0.18, linewidths=0, zorder=2,
               label=f"训练段配对观测（n={len(h):,}）")
    ax.stairs(table, edges, baseline=None, color=INK, lw=1.5, zorder=4,
              label="率定曲线（50 级阶梯）")

    # 箱内均值即该级的取值——挑一个中高水位的箱标出来
    mid = np.searchsorted(edges, np.quantile(h, 0.90), "right") - 1
    xc = (edges[mid] + edges[mid + 1]) / 2
    ax.plot([xc], [table[mid]], "o", ms=4.5, mfc="white", mec=INK, mew=1.2, zorder=5)
    ax.annotate(f"第 {mid + 1} 箱\n箱内径流均值 {table[mid]:.2f}",
                xy=(xc, table[mid]), xytext=(xc - 1.55, table[mid] + 1.5),
                fontsize=6.5, color=INK_SOFT, ha="left", va="bottom",
                arrowprops=dict(arrowstyle="-", color=INK_SOFT, lw=0.6,
                                shrinkA=0, shrinkB=3))

    ax.set_xlim(*_clip_limits(h))
    ax.set_ylim(*_clip_limits(q))
    ax.set_title(f"(a) 单流域自拟合（{DEMO_BASIN}）", loc="left")
    ax.set_xlabel("归一化水位（标准差）")
    ax.set_ylabel("归一化径流（标准差）")
    ax.legend(loc="upper left", markerscale=3.5)
    return len(h)


def _panel_regional(ax, prepared):
    """(b) 区域化：86 条逐流域曲线 + 混合拟合的全局曲线。"""
    basins = list(prepared.splits["splits"])
    hs, qs, n_drawn = [], [], 0
    for basin in basins:
        h, q = _pairs(prepared, basin, "train")
        if not h.size:
            continue
        hs.append(h)
        qs.append(q)
        fit = fit_rating(h, q)
        if fit is None:
            continue
        ax.stairs(fit[1], fit[0], baseline=None, color=RATING, lw=0.5, alpha=0.30,
                  zorder=2, label="逐流域曲线" if n_drawn == 0 else None)
        n_drawn += 1

    h_all, q_all = np.concatenate(hs), np.concatenate(qs)
    edges, table = fit_rating(h_all, q_all)
    ax.stairs(table, edges, baseline=None, color=INK, lw=1.8, zorder=4,
              label=f"区域化全局曲线（{n_drawn} 流域混合，n={len(h_all):,}）")

    ax.set_xlim(*_clip_limits(h_all))
    ax.set_ylim(*_clip_limits(q_all))
    ax.set_title("(b) 区域化全局曲线", loc="left")
    ax.set_xlabel("归一化水位（标准差）")
    ax.set_ylabel("归一化径流（标准差）")
    ax.legend(loc="upper left")
    return n_drawn, len(h_all), (edges, table)


def main():
    from pipeline.dataset import PreparedData

    set_style("zh")
    prepared = PreparedData()

    fig, axes = plt.subplots(1, 2, figsize=(DOUBLE_COL, DOUBLE_COL * 0.42))
    n_pairs = _panel_single(axes[0], prepared)
    n_basins, n_all, global_fit = _panel_regional(axes[1], prepared)

    for ax in axes:
        ax.grid(True, lw=0.35, color=RATING, alpha=0.25)
        ax.set_axisbelow(True)

    fig.suptitle("率定曲线的构造：按水位分位数分箱、每箱取径流均值（归一化空间）",
                 fontsize=8.5, y=0.99)
    fig.tight_layout(rect=(0, 0, 1, 0.95))

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUT_DIR / "rating_curve_shape_zh.png"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    fig.savefig(out.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)

    # 顺便把构造参数与拟合质量打出来，便于对照正文数字
    h_te, q_te = _pairs(prepared, DEMO_BASIN, "test")
    h_tr, q_tr = _pairs(prepared, DEMO_BASIN, "train")
    fit_tr = fit_rating(h_tr, q_tr)
    print(f"分箱数 N_BINS = {N_BINS}（等样本量，按 h 的分位数切）")
    print(f"(a) {DEMO_BASIN}: 训练段 {n_pairs:,} 对，每箱约 {n_pairs // N_BINS} 个点")
    print(f"    训练期拟合 → 测试期应用 NSE = {nse(q_te, apply_rating(fit_tr, h_te)):.4f}")
    print(f"(b) 区域化: {n_basins} 个流域混合 {n_all:,} 对配对观测")
    print(f"    全局曲线取值范围 {np.nanmin(global_fit[1]):.2f} ~ {np.nanmax(global_fit[1]):.2f}")
    print(f"  已写入 {out}")


if __name__ == "__main__":
    main()
