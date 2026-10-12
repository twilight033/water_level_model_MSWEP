"""绘制三张实验总结图，严格分开原86旧协议与99新协议。"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "hydro-multitask-paper/figures/experiment_summary_v2"
CONFIGS = ["base", "physical", "L480+extended", "L480+physical+extended"]
CONFIG_LABELS = ["21天 · observed", "21天 · physical", "60天+扩展属性 · observed", "60天+扩展属性 · physical"]
CONFIG_COLORS = ["#3373A5", "#38998C", "#D78334", "#8865AA"]
MODEL_COLORS = {"physical": ["#86ADC8", "#165F88"], "observed": ["#E7AA7B", "#B94F29"]}

plt.rcParams.update({
    "font.sans-serif": ["Microsoft YaHei", "SimHei", "DejaVu Sans"],
    "axes.unicode_minus": False,
    "svg.fonttype": "none",
    "font.size": 10,
    "axes.titlesize": 12,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "savefig.facecolor": "white",
})


def save(fig, name, note, bottom=.19):
    """预留标题、脚注区域，输出高清图和保留文字的矢量图。"""
    fig.tight_layout(rect=[.015, bottom, .985, .92], h_pad=2.3, w_pad=2.8)
    fig.text(.035, .025, note, va="bottom", fontsize=9, color="#444444", linespacing=1.6)
    for suffix in ("png", "svg"):
        fig.savefig(OUT / f"{name}.{suffix}", dpi=300, bbox_inches="tight")
    plt.close(fig)


def figure_original86():
    """复用已登记的逐流域配对差与置信区间，不跨配置混算。"""
    paired = pd.read_csv(ROOT / "results/summary/paired_tests.csv")
    held = pd.read_csv(ROOT / "results/summary/holdout_substitution.csv")
    fig, axes = plt.subplots(1, 2, figsize=(14, 6.6), gridspec_kw={"width_ratios": [1.05, 1.35]})
    fig.suptitle("原86流域：完整监督与径流训练标签留出的双头增益", fontsize=15, y=.98)
    a, b = axes
    for i, (cfg, color) in enumerate(zip(CONFIGS, CONFIG_COLORS)):
        row = paired[(paired.config == cfg) & (paired.model_a == "dual_head") &
                     (paired.model_b == "single_flow") & (paired.task == "flow") & (paired.metric == "nse")]
        assert len(row) == 1
        row = row.iloc[0]
        a.errorbar(row.mean_diff, i, xerr=[[row.mean_diff-row.mean_ci_lo], [row.mean_ci_hi-row.mean_diff]],
                   fmt="o", color=color, markersize=7, capsize=4, lw=1.8)
        a.text(.023, i, f"{row.mean_diff:+.3f}", ha="right", va="center", color=color, fontsize=11)
        vals = []
        for ratio in (30, 50, 70):
            v = held[(held.config == cfg) & (held.scenario == f"q_holdout{ratio}") & (held.basin_group == "被留出流域")]
            assert len(v) == 1
            vals.append(v.iloc[0])
        x = np.array([30, 50, 70]) + (i-1.5)*.8
        y = np.array([v.mean_gain for v in vals])
        b.errorbar(x, y, yerr=[[v.mean_gain-v.ci_lo for v in vals], [v.ci_hi-v.mean_gain for v in vals]],
                   fmt="o-", color=color, label=CONFIG_LABELS[i], markersize=6, capsize=3, lw=1.6)
        # 将重叠的两条基线曲线数值错开标注，不改变数据位置。
        offset = [(8, -18), (8, 9), (8, 0), (8, 0)][i]
        b.annotate(f"{y[-1]:+.3f}", (x[-1], y[-1]), xytext=offset, textcoords="offset points", color=color)
    a.set_yticks(range(4), CONFIG_LABELS)
    a.set_ylim(3.6, -.6)
    a.set_xlim(-.024, .025)
    a.axvline(0, color="#555555", lw=1)
    a.grid(axis="x", alpha=.18)
    a.set_title("(a) 完整监督：全部86流域")
    a.set_xlabel("平均ΔNSE（双头Q − 单任务Q）")
    b.axhline(0, color="#555555", lw=1)
    b.set_xticks([30, 50, 70], ["30%", "50%", "70%"])
    b.set_xlim(26, 78)
    b.set_ylim(-.015, .54)
    b.set_title("(b) 旧留出协议：只评价被留出流域")
    b.set_xlabel("随机整流域Q训练标签留出比例")
    b.set_ylabel("平均ΔNSE（双头Q − 单任务Q）")
    b.legend(loc="upper left", fontsize=9, frameon=False)
    b.grid(alpha=.18)
    save(fig, "fig01_original86_gain",
         "点为逐流域配对差的均值；误差棒为逐流域bootstrap 95%置信区间，不是种子波动。\n"
         "完整监督使用5或10个模型种子；留出使用3个模型种子×3个掩膜种子，先对每站在被留出时的结果平均。\n"
         "旧留出协议只屏蔽训练Q，验证Q仍参与选模，H保留。强配置70%留出的单任务结果具有较大种子波动。")


def load_new_runs():
    """读取每次实际名单并配对，避免汇总表缺少掩膜种子造成重复统计。"""
    rows = []
    for scale in MODEL_COLORS:
        root = ROOT / "results" / f"nonrating99_random_{scale}_v2"
        exp = json.loads((root / "experiment.json").read_text(encoding="utf-8-sig"))
        assert exp["random_holdout_protocol"] == "train_valid_q_hidden_v2"
        assert exp["model_seeds"] == [1, 2, 3] and exp["mask_seeds"] == [42, 123, 456]
        for ratio in (30, 50, 70):
            for ms in (1, 2, 3):
                for xs in (42, 123, 456):
                    previous = None
                    for arch in ("single_flow", "dual_head"):
                        run = root / f"random_q_holdout{ratio}_{arch}_ms{ms}_xs{xs}"
                        meta = json.loads((run / "run.json").read_text(encoding="utf-8-sig"))
                        assert meta["mask"]["meta"]["protocol"] == "train_valid_q_hidden_v2"
                        held = meta["mask"]["per_task"]["flow"]["held_out_basins"]
                        if previous is not None:
                            assert held == previous
                        previous = held
                        d = pd.read_csv(run / "metrics.csv", dtype={"basin": str})
                        v = d[(d.task == "flow") & d.basin.isin(held)].set_index("basin").nse
                        assert len(v) == len(held) == round(99*ratio/100)
                        assert v.index.is_unique and np.isfinite(v).all()
                        rows.append(dict(scale=scale, ratio=ratio, model_seed=ms, mask_seed=xs,
                                         arch=arch, mean=v.mean(), median=v.median()))
    assert len(rows) == 108
    return pd.DataFrame(rows)


def figure_new99():
    """均值与中位数并列展示，全部运行保留，不截断极端值。"""
    d = load_new_runs()
    fig, axes = plt.subplots(2, 2, figsize=(13, 9.5))
    fig.suptitle("99流域新协议：随机Q留出的绝对预测性能", fontsize=15, y=.98)
    for col, scale in enumerate(MODEL_COLORS):
        for row, metric in enumerate(("mean", "median")):
            a = axes[row, col]
            for i, arch in enumerate(("single_flow", "dual_head")):
                x = np.array([30, 50, 70]) + (i-.5)*1.7
                groups = [d[(d.scale == scale) & (d.arch == arch) & (d.ratio == r)][metric].to_numpy()
                          for r in (30, 50, 70)]
                y = np.array([g.mean() for g in groups])
                color = MODEL_COLORS[scale][i]
                a.plot(x, y, "o-", color=color, lw=2, markersize=7, label="单任务Q" if i == 0 else "双头Q")
                for xx, g in zip(x, groups):
                    assert len(g) == 9
                    a.scatter(xx+np.linspace(-.5, .5, 9), g, color=color, s=22, alpha=.55, zorder=3)
                if row == 0:
                    for xx, yy in zip(x, y):
                        at_left = xx < 35
                        offset_y = (15 if i == 0 else -20) if scale == "physical" else (-22 if i == 0 else 12)
                        a.annotate(f"{yy:.2f}", (xx, yy),
                                   xytext=(12 if at_left or i == 1 else -12, offset_y),
                                   textcoords="offset points", color=color,
                                   ha="left" if at_left or i == 1 else "right", fontsize=9)
            a.set_title(f"({'abcd'[row*2+col]}) {scale}：流域{'平均' if row == 0 else '中位'}NSE")
            a.set_xticks([30, 50, 70], ["30%\nn=30", "50%\nn=50", "70%\nn=69"])
            a.set_xlim(26, 75)
            a.set_xlabel("随机整流域Q留出比例（n为每次留出的流域数）")
            a.set_ylabel("测试Q NSE")
            if row == 0:
                a.axhline(0, color="#555555", lw=.9)
                if scale == "physical":
                    a.set_yscale("symlog", linthresh=1)
                    subset = d[d.scale == scale][metric]
                    a.set_ylim(min(subset)*1.4, 1)
                    a.set_title("(a) physical：流域平均NSE\n对称对数轴，−1至1线性", fontsize=11)
                else:
                    subset = d[d.scale == scale][metric]
                    a.set_ylim(min(subset)*1.15, .85)
            else:
                a.set_ylim(.18, .73)
                p = d[d.scale == scale].pivot(index=["ratio", "model_seed", "mask_seed"], columns="arch", values=metric)
                gain = (p.dual_head-p.single_flow).groupby("ratio").mean()
                a.text(.03, .95, "中位NSE差：" + " / ".join(f"{gain[r]:+.3f}" for r in (30, 50, 70)),
                       transform=a.transAxes, va="top", fontsize=9)
            a.legend(loc="lower left", frameon=False, fontsize=9)
            a.grid(alpha=.18)
    save(fig, "fig02_random99_new_protocol",
         "新协议：同一批流域的训练与验证Q均屏蔽，H保留；测试Q只用于最终评估。窗口60天，扩展属性。\n"
         "小散点为3个模型种子×3个掩膜种子的9次运行；大点与折线为9次运行的算术平均，并非置信区间。\n"
         "每次先对实际留出流域计算均值或中位数；下排为9个流域中位数的平均。observed仍使用真实训练期Q尺度。", bottom=.15)


def figure_fixed14():
    """两组独立固定留出实验，逐流域种子均值与配对增益。"""
    roots = {"physical": ROOT / "results/nonrating14_full_matrix",
             "observed": ROOT / "results/nonrating14_observed_matrix"}
    meta = json.loads((roots["physical"] / "experiment.json").read_text(encoding="utf-8-sig"))
    other = json.loads((roots["observed"] / "experiment.json").read_text(encoding="utf-8-sig"))
    assert meta["candidate_basins"] == other["candidate_basins"]
    assert meta["matched_controls"] == other["matched_controls"]
    data = {s: pd.read_csv(r / "metrics_all.csv", dtype={"basin": str}) for s, r in roots.items()}
    fig, axes = plt.subplots(2, 2, figsize=(13, 10))
    fig.suptitle("候选14与匹配对照14：绝对NSE和双头配对增益", fontsize=15, y=.98)
    for col, (scenario, key, title) in enumerate((
            ("candidate_fixed", "candidate_basins", "候选14流域"),
            ("matched_86_fixed", "matched_controls", "原86匹配对照14流域"))):
        ids = meta[key]
        arrays, labels = [], []
        deltas = []
        for scale in roots:
            d = data[scale]
            d = d[(d.scenario == scenario) & (d.task == "flow") & d.basin.isin(ids)]
            counts = d.groupby(["basin", "architecture"]).size()
            assert (counts == 3).all() and len(counts) == 28
            p = d.groupby(["basin", "architecture"]).nse.mean().unstack().reindex(ids)
            assert np.isfinite(p.to_numpy()).all()
            for arch, label in (("single_flow", "单任务"), ("dual_head", "双头")):
                v = p[arch].to_numpy()
                arrays.append(v)
                labels.append(f"{scale}\n{label}\n中位 {np.median(v):.3f}")
            deltas.append(p.dual_head-p.single_flow)
        a, b = axes[0, col], axes[1, col]
        colors = [c for cs in MODEL_COLORS.values() for c in cs]
        boxes = a.boxplot(arrays, patch_artist=True, showfliers=False, widths=.45,
                          medianprops={"color": "#222222", "linewidth": 1.5})
        for i, (v, color, patch) in enumerate(zip(arrays, colors, boxes["boxes"]), 1):
            patch.set_facecolor(color)
            patch.set_alpha(.3)
            a.scatter(i+np.linspace(-.13, .13, 14), v, color=color, s=24, zorder=3)
        a.set_title(title+"：绝对性能")
        a.set_xticks(range(1, 5), labels, fontsize=9)
        a.set_ylabel("测试Q NSE")
        a.axhline(0, color="#555555", lw=.9)
        if col == 0:
            a.set_yscale("symlog", linthresh=1)
            a.set_ylim(min(map(np.min, arrays))*1.5, 1)
            a.annotate(f"06843500\n双头physical = {arrays[1][ids.index('06843500')]:.3f}", (2+np.linspace(-.13, .13, 14)[ids.index("06843500")],
                       arrays[1][ids.index("06843500")]), xytext=(.35, .09), textcoords="axes fraction",
                       fontsize=9, arrowprops={"arrowstyle": "-", "color": "#555555"})
        else:
            a.set_ylim(-.38, .9)
        for j, (scale, delta) in enumerate(zip(roots, deltas), 1):
            b.scatter(j+np.linspace(-.13, .13, 14), delta, color=MODEL_COLORS[scale][1], s=30)
            b.plot([j-.25, j+.25], [delta.median()]*2, color="#222222", lw=2)
            b.text(j, 1.02, f"改善 {(delta > 0).sum()}/14", transform=b.get_xaxis_transform(), ha="center", fontsize=10)
        if col == 0:
            b.set_yscale("symlog", linthresh=1)
            b.set_ylim(min(v.min() for v in deltas)*1.5, max(v.max() for v in deltas)*1.5)
        else:
            b.set_ylim(-.04, .9)
        b.set_xlim(.5, 2.5)
        b.set_xticks([1, 2], [f"{s}\n中位增益 {v.median():+.3f}" for s, v in zip(roots, deltas)])
        b.set_ylabel("ΔNSE（双头Q − 单任务Q）")
        b.axhline(0, color="#555555", lw=.9)
        a.grid(axis="y", alpha=.18)
        b.grid(axis="y", alpha=.18)
    save(fig, "fig03_candidate_matched14",
         "两组来自各自独立的99流域固定留出训练：所选14流域训练与验证Q均屏蔽，H保留；测试Q仅用于评估。\n"
         "每个散点为单个流域的3个模型种子平均NSE；上下排均保留全部极端值。候选列使用对称对数轴，−1至1线性。\n"
         "箱线图黑线为中位数；下排黑横线为配对增益中位数。observed使用真实Q尺度；候选组尚不能认定为直接测流组。", bottom=.15)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    figure_original86()
    figure_new99()
    figure_fixed14()
    print(f"已生成3张PNG和3张SVG：{OUT}")


if __name__ == "__main__":
    main()
