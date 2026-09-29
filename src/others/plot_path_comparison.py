"""强配置下四条径流预测路径的逐流域性能对比（70% 留出、被留出流域）。

四条路径的输入信息各不相同，这是读图的关键：

| 路径 | 用到的径流信息 | 是否合法 |
|---|---|---|
| 单任务径流 | 无（该流域零径流标签） | ✓ |
| 两阶段 + 自拟合曲线 | 保留流域的配对关系 + 该流域径流量级* | ✓（physical 口径下量级改由回归推算） |
| 两阶段 + 官方曲线 | **该站官方率定曲线**（由其历史实测径流定出） | ✗ 越界，仅作上界 |
| 双头 | 无（只有水位标签） | ✓ |

*observed 口径用实测统计量，physical 口径用 area×p_mean 回归推算——后者才是
留出设定下合法的做法，也正是自拟合曲线在右图塌下去的原因。官方曲线的量级
烧在折点数值里，剥不掉，所以两个配置下逐值相同。

配色沿用项目语义：单任务蓝、双头橙、率定路径灰；两条率定路径同色，用斜纹
区分自拟合与官方，以示它们是同一方法的两个变体而非两类方法。
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

_ROOT = Path(__file__).resolve().parents[2]
for _p in (str(_ROOT / "src"), str(_ROOT),
           str(_ROOT / "hydro-multitask-paper" / "figures" / "revision" / "scripts")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from figcommon_rev import DOUBLE_COL, DUAL, INK, INK_SOFT, RATING, SINGLE, set_style  # noqa: E402
from pipeline.paths import RESULTS_ROOT  # noqa: E402

CONFIGS = [("L480+extended", "(a) 强配置（60 天窗口 + 24 属性）"),
           ("L480+physical+extended", "(b) 强配置 + 物理归一化")]

# (标签, 颜色, 斜纹)；顺序即绘图顺序，固定不变
PATHS = [
    ("单任务径流", SINGLE, None),
    ("两阶段\n+自拟合曲线", RATING, None),
    ("两阶段\n+官方曲线", RATING, "///"),
    ("双头", DUAL, None),
]

OUT_DIR = RESULTS_ROOT / "figures"


def collect(config: str) -> pd.DataFrame:
    """四条路径的逐流域 NSE，对齐到同一批流域。"""
    from evaluation.run_config import load_summary, select

    rating = pd.read_csv(RESULTS_ROOT / "summary" / "two_stage_official_rating.csv",
                         dtype={"basin": str})
    rating = rating[rating["config"] == config].set_index("basin")
    rating = rating.dropna(subset=["nse_binned", "nse_official"])

    metrics, _ = load_summary()
    sub = select(metrics, config=config, scenario="q_holdout70",
                 task="flow", held_out=True)

    out = pd.DataFrame(index=rating.index)
    out["两阶段\n+自拟合曲线"] = rating["nse_binned"]
    out["两阶段\n+官方曲线"] = rating["nse_official"]
    for arch, label in (("single_flow", "单任务径流"), ("dual_head", "双头")):
        out[label] = sub[sub["architecture"] == arch].groupby("basin")["nse"].mean()
    return out.dropna()


def draw_panel(ax, data: pd.DataFrame, title: str):
    labels = [p[0] for p in PATHS]
    values = [data[label].to_numpy(dtype=float) for label in labels]

    bp = ax.boxplot(values, widths=0.55, patch_artist=True, showfliers=False,
                    medianprops=dict(color=INK, lw=1.2),
                    whiskerprops=dict(color=INK_SOFT, lw=0.7),
                    capprops=dict(color=INK_SOFT, lw=0.7),
                    boxprops=dict(lw=0.7))
    for patch, (_, color, hatch) in zip(bp["boxes"], PATHS):
        patch.set_facecolor(color)
        patch.set_alpha(0.30 if hatch is None else 0.16)
        patch.set_edgecolor(color)
        if hatch:
            patch.set_hatch(hatch)

    # 均值用菱形标出——箱线给的是中位，正文引用的是均值，两者都要能读到
    for i, v in enumerate(values, start=1):
        m = float(np.nanmean(v))
        ax.plot([i], [m], "D", ms=4, mfc="white", mec=INK, mew=1.1, zorder=5)
        ax.annotate(f"{m:.3f}", xy=(i, m), xytext=(0, 9), textcoords="offset points",
                    ha="center", fontsize=6.5, color=INK)

    ax.set_xticks(range(1, len(labels) + 1))
    ax.set_xticklabels(labels, fontsize=6.5)
    ax.set_ylabel("径流 NSE（被留出流域）")
    ax.set_title(title, loc="left")
    ax.grid(True, axis="y", lw=0.35, color=RATING, alpha=0.25)
    ax.set_axisbelow(True)


def main():
    set_style("zh")
    panels = [(collect(cfg), title) for cfg, title in CONFIGS]
    n = len(panels[0][0])

    fig, axes = plt.subplots(1, 2, figsize=(DOUBLE_COL, DOUBLE_COL * 0.46),
                             sharey=True)
    for ax, (data, title) in zip(axes, panels):
        draw_panel(ax, data, title)
    axes[1].set_ylabel("")

    fig.suptitle(f"70% 留出情景下四条路径的径流预测性能（被留出流域 n={n}，"
                 f"菱形为均值）", fontsize=8.5, y=0.985)
    fig.tight_layout(rect=(0, 0, 1, 0.94))

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUT_DIR / "strong_config_paths_zh.png"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    fig.savefig(out.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)

    for (data, title), (cfg, _) in zip(panels, CONFIGS):
        print(f"{cfg}（n={len(data)}）:")
        for label, *_ in PATHS:
            v = data[label]
            print(f"  {label.replace(chr(10), ''):18s} 均值 {v.mean():.4f}  中位 {v.median():.4f}")
        gap = data["双头"].mean() - data[PATHS[2][0]].mean()
        print(f"  双头 − 两阶段+官方曲线: {gap:+.4f}")
        print()
    print(f"  已写入 {out}")


if __name__ == "__main__":
    main()
