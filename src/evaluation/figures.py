"""论文图表生成。

覆盖审稿意见要求的几类图：
- Major 4：各流域划分日期分布、每流域 × 任务 × 分段的有效标签数
- Major 9：真实缺口与人工缺失段的起点月份分布对比
- Major 2 / Minor 8：主实验对比必须同时给出跨种子误差棒与逐流域配对差值
  分布，避免窄纵轴把极小差异在视觉上放大
- Major 8：代表性流域的水文过程线与洪峰事件放大

配色取自经校验的分类色板（相邻对 CVD ΔE 9.1、正常视觉 22.9，均达标）。
青色与黄色对浅色底的对比度偏低，因此凡使用这两槽的图都配直接标注。
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

_ROOT = Path(__file__).resolve().parents[2]
for _p in (str(_ROOT / "src"), str(_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib import font_manager  # noqa: E402

from pipeline.paths import COVERAGE_DIR, RESULTS_ROOT  # noqa: E402

FIG_DIR = RESULTS_ROOT / "figures"
SUMMARY_DIR = RESULTS_ROOT / "summary"

# 经 validate_palette 校验的分类色板前 4 槽
SERIES = {"blue": "#2a78d6", "orange": "#eb6834", "aqua": "#1baf7a", "yellow": "#eda100"}
ARCH_COLOR = {
    "single_flow": SERIES["blue"],
    "single_waterlevel": SERIES["blue"],
    "dual_head": SERIES["orange"],
    "wl2d": SERIES["aqua"],
    "capacity_matched": SERIES["yellow"],
}
ARCH_LABEL = {
    "single_flow": "单任务 Q", "single_waterlevel": "单任务 h",
    "dual_head": "双头多任务", "wl2d": "WL2D 级联", "capacity_matched": "参数量匹配对照",
}
TASK_LABEL = {"flow": "径流 Q", "waterlevel": "水位 h"}

INK = "#0b0b0b"
INK_SOFT = "#52514e"
GRID = "#d9d8d4"


def setup_style():
    """统一的论文图风格：细线、弱化网格、无顶右边框。"""
    for name in ("Microsoft YaHei", "SimHei", "Source Han Sans SC", "Noto Sans CJK SC"):
        try:
            if font_manager.findfont(name, fallback_to_default=False):
                plt.rcParams["font.family"] = name
                break
        except ValueError:
            continue
    plt.rcParams.update({
        "axes.unicode_minus": False,
        "figure.dpi": 150, "savefig.dpi": 300, "savefig.bbox": "tight",
        "axes.edgecolor": GRID, "axes.linewidth": 0.8,
        "axes.labelcolor": INK, "axes.titlesize": 11, "axes.titleweight": "normal",
        "text.color": INK, "xtick.color": INK_SOFT, "ytick.color": INK_SOFT,
        "xtick.labelsize": 9, "ytick.labelsize": 9,
        "grid.color": GRID, "grid.linewidth": 0.6,
        "legend.frameon": False, "legend.fontsize": 9,
        "lines.linewidth": 1.6,
    })


def _finish(ax, title=None, xlabel=None, ylabel=None, grid_axis="y"):
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    if grid_axis:
        ax.grid(True, axis=grid_axis, alpha=0.7)
        ax.set_axisbelow(True)
    if title:
        ax.set_title(title, loc="left", pad=10)
    if xlabel:
        ax.set_xlabel(xlabel)
    if ylabel:
        ax.set_ylabel(ylabel)


def _suptitle(fig, text):
    """总标题抬到坐标区之上，避免压住子图标题（savefig 用 tight bbox 收边）。"""
    fig.suptitle(text, x=0.02, ha="left", y=1.04)


def _save(fig, name):
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    path = FIG_DIR / name
    fig.savefig(path)
    plt.close(fig)
    print(f"  已保存 {path}")
    return path


# ── Major 4：划分与覆盖率 ──────────────────────────────────────────────────────

def fig_split_timeline():
    """各流域训练/验证/测试期的时间跨度，按训练起始年排序。

    每流域测试期落在不同年份是按流域划分的固有代价，必须在论文中如实展示。
    """
    df = pd.read_csv(COVERAGE_DIR / "basin_split_bounds.csv", dtype={"basin": str})
    for col in ("train_start", "train_end", "valid_start", "valid_end",
                "test_start", "test_end"):
        df[col] = pd.to_datetime(df[col])
    df = df.sort_values("train_start").reset_index(drop=True)

    fig, ax = plt.subplots(figsize=(8, 9))
    segments = [("train", "train_start", "train_end", SERIES["blue"], "训练"),
                ("valid", "valid_start", "valid_end", SERIES["orange"], "验证"),
                ("test", "test_start", "test_end", SERIES["aqua"], "测试")]
    for _, start_col, end_col, color, label in segments:
        left = matplotlib.dates.date2num(df[start_col])
        width = matplotlib.dates.date2num(df[end_col]) - left
        ax.barh(df.index, width, left=left, height=0.7, color=color,
                edgecolor="white", linewidth=0.4, label=label)

    ax.set_yticks([])
    ax.invert_yaxis()
    ax.xaxis_date()
    # 晚开始的流域在左下留出大片空白，图例放进去，避免与标题争位置
    ax.legend(loc="lower left", ncol=1, borderpad=0.8)
    _finish(ax, title=f"各流域训练/验证/测试期（{len(df)} 个流域，按训练起始时间排序）",
            xlabel="年份", ylabel="流域", grid_axis="x")

    short = df.index[df["short_test"]] if "short_test" in df else []
    for idx in short:
        # x 用轴坐标固定在右侧页边，y 用数据坐标，四条标注互不遮挡也不压图
        ax.annotate("测试期不足 1 年", xy=(1.01, idx),
                    xycoords=("axes fraction", "data"),
                    va="center", fontsize=7, color=INK_SOFT, annotation_clip=False)
    return _save(fig, "fig_split_timeline.png")


def fig_label_coverage():
    """每流域 × 任务 × 分段的有效标签比例分布。"""
    cov = pd.read_csv(COVERAGE_DIR / "basin_task_split_counts.csv", dtype={"basin": str})
    splits = ["train", "valid", "test"]
    names = {"train": "训练", "valid": "验证", "test": "测试"}

    fig, axes = plt.subplots(1, 2, figsize=(9, 4), sharey=True)
    for ax, task, color in zip(axes, ("flow", "waterlevel"),
                               (SERIES["blue"], SERIES["orange"])):
        data = [cov.loc[cov["split"] == s, f"{task}_ratio"].to_numpy() for s in splits]
        parts = ax.boxplot(data, tick_labels=[names[s] for s in splits], widths=0.5,
                           patch_artist=True, showfliers=False,
                           medianprops=dict(color=INK, linewidth=1.4),
                           whiskerprops=dict(color=GRID, linewidth=1.0),
                           capprops=dict(color=GRID, linewidth=1.0),
                           boxprops=dict(facecolor=color, edgecolor="white", linewidth=0.8,
                                         alpha=0.85))
        for i, vals in enumerate(data, start=1):
            jitter = (np.random.default_rng(i).random(vals.size) - 0.5) * 0.18
            ax.scatter(i + jitter, vals, s=8, color=INK, alpha=0.35, linewidths=0)
            ax.text(i, 1.04, f"中位 {np.median(vals):.2f}", ha="center",
                    fontsize=8, color=INK_SOFT)
        ax.set_ylim(-0.03, 1.12)
        _finish(ax, title=TASK_LABEL[task], ylabel="有效标签比例" if task == "flow" else None)
    _suptitle(fig, "各流域在各分段的有效标签比例（每点一个流域）")
    return _save(fig, "fig_label_coverage.png")


# ── Major 9：缺失机制季节性 ────────────────────────────────────────────────────

def fig_gap_seasonality(month_tables: dict):
    """真实缺口与人工缺失段的起点月份分布对比。

    month_tables: {"真实缺口": counts12, "季节匹配段": counts12, "均匀段": counts12}
    """
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8), sharey=True)
    months = np.arange(1, 13)
    colors = [SERIES["blue"], SERIES["orange"], SERIES["aqua"]]

    for ax, task in zip(axes, ("flow", "waterlevel")):
        series = {k: v[task] for k, v in month_tables.items() if task in v}
        width = 0.8 / max(len(series), 1)
        for i, ((label, counts), color) in enumerate(zip(series.items(), colors)):
            share = np.asarray(counts, dtype=float)
            share = share / share.sum() * 100
            ax.bar(months + (i - (len(series) - 1) / 2) * width, share, width * 0.9,
                   color=color, edgecolor="white", linewidth=0.6, label=label)
        ax.set_xticks(months)
        _finish(ax, title=TASK_LABEL[task],
                xlabel="月份", ylabel="起点占比 %" if task == "flow" else None)
    axes[0].legend(loc="upper right")
    _suptitle(fig, "缺口起始月份分布：真实观测 vs 人工注入")
    return _save(fig, "fig_gap_seasonality.png")


# ── Major 2 / Minor 8：主实验与配对差值 ───────────────────────────────────────

def fig_main_experiment(metrics: pd.DataFrame, metric="nse", config="base"):
    """各架构的逐流域平均指标，误差棒为跨模型种子的标准差。

    只画同一可比配置内的运行：不同窗口长度、归一化方式、流域子集的结果混进
    同一根柱子会得到一个谁都不对应的均值。
    """
    sub = metrics[metrics["scenario"] == "complete"]
    if "config" in sub.columns:
        sub = sub[sub["config"] == config]
    if sub.empty:
        return None
    fig, axes = plt.subplots(1, 2, figsize=(9, 4))
    for ax, task in zip(axes, ("flow", "waterlevel")):
        rows = []
        for arch, grp in sub[sub["task"] == task].groupby("architecture"):
            per_seed = grp.groupby("model_seed")[metric].mean()
            rows.append((arch, per_seed.mean(),
                         per_seed.std(ddof=1) if len(per_seed) > 1 else 0.0, len(per_seed)))
        if not rows:
            continue
        rows.sort(key=lambda r: r[1])
        labels = [ARCH_LABEL.get(r[0], r[0]) for r in rows]
        means = [r[1] for r in rows]
        sds = [r[2] for r in rows]
        colors = [ARCH_COLOR.get(r[0], SERIES["blue"]) for r in rows]

        y = np.arange(len(rows))
        ax.barh(y, means, height=0.6, color=colors, edgecolor="white", linewidth=0.8)
        ax.errorbar(means, y, xerr=sds, fmt="none", ecolor=INK, elinewidth=1.2, capsize=3)
        for i, (m, sd, n) in enumerate(zip(means, sds, [r[3] for r in rows])):
            ax.text(m + sd + 0.004, i, f"{m:.4f} ± {sd:.4f}  (n={n})",
                    va="center", fontsize=8, color=INK_SOFT)
        ax.set_yticks(y, labels)
        # 纵轴从 0 起，避免窄轴把极小差异在视觉上放大（Minor 8）
        ax.set_xlim(0, max(np.array(means) + np.array(sds)) * 1.35)
        _finish(ax, title=TASK_LABEL[task], xlabel=metric.upper(), grid_axis="x")
    _suptitle(fig, "主实验：完整标签下各架构表现（误差棒为跨模型种子标准差）")
    suffix = "" if config == "base" else f"_{config}"
    return _save(fig, f"fig_main_experiment_{metric}{suffix}.png")


def fig_paired_difference(per_basin: pd.DataFrame, title: str, name: str):
    """逐流域配对差值的排序图与分布，直观展示效应量而非平均值。"""
    d = per_basin["diff"].to_numpy()
    d = d[np.isfinite(d)]
    order = np.argsort(d)

    fig, axes = plt.subplots(1, 2, figsize=(9, 3.6),
                             gridspec_kw={"width_ratios": [3, 1]})
    # 少数流域的单任务 NSE 极负，差值能到 +3 以上，按满量程画会把其余 80 多个
    # 流域压成一条线。截断纵轴并注明被截掉多少个，同时给出中位数——均值受这几
    # 个离群值影响很大，只报均值是不诚实的
    span = float(np.percentile(np.abs(d), 98)) if d.size else 0.0
    clipped = int((np.abs(d) > span).sum()) if span > 0 else 0
    use_clip = span > 0 and np.abs(d).max() > 3 * span

    ax = axes[0]
    colors = np.where(d[order] >= 0, SERIES["orange"], SERIES["blue"])
    ax.bar(np.arange(d.size), d[order], width=0.9, color=colors, linewidth=0)
    ax.axhline(0, color=INK, linewidth=0.9)
    ax.axhline(d.mean(), color=INK_SOFT, linewidth=1.0, linestyle="--")
    ax.text(d.size * 0.02, d.mean(), f"  均值 {d.mean():+.4f}",
            va="bottom", fontsize=8, color=INK_SOFT)
    ax.axhline(np.median(d), color=INK_SOFT, linewidth=1.0, linestyle=":")
    ax.text(d.size * 0.02, np.median(d), f"  中位数 {np.median(d):+.4f}",
            va="top", fontsize=8, color=INK_SOFT)
    if use_clip:
        ax.set_ylim(-span * 1.15, span * 1.15)
        ax.text(0.98, 0.95, f"{clipped} 个流域超出范围（最大 {d.max():+.2f}）",
                transform=ax.transAxes, ha="right", va="top", fontsize=8,
                color=INK_SOFT)
    _finish(ax, title=title, xlabel="流域（按差值排序）", ylabel="NSE 差值")

    ax2 = axes[1]
    bins = np.linspace(-span * 1.15, span * 1.15, 21) if use_clip else 20
    ax2.hist(np.clip(d, -span * 1.15, span * 1.15) if use_clip else d, bins=bins,
             color=SERIES["aqua"], edgecolor="white", linewidth=0.6,
             orientation="horizontal")
    ax2.axhline(0, color=INK, linewidth=0.9)
    if use_clip:
        ax2.set_ylim(-span * 1.15, span * 1.15)
    n_up = int((d > 0).sum())
    ax2.text(0.95, 0.02, f"改善 {n_up} / 变差 {d.size - n_up}",
             transform=ax2.transAxes, ha="right", fontsize=8, color=INK_SOFT)
    _finish(ax2, title="分布", xlabel="流域数", grid_axis="x")
    return _save(fig, name)


# ── Major 3：缺失下降曲线 ──────────────────────────────────────────────────────

def fig_missing_degradation(deg: pd.DataFrame, task="flow", name=None):
    """缺失比例 vs 相对自身完整标签基线的下降幅度。"""
    # 按任务取自己那一路的缺失情景。both50_seg 是"两种标签同时缺"的单点对照，
    # 不能标成 50 混进曲线——此前它与 q50_seg 落在同一横坐标上被平均
    prefix = {"flow": "q", "waterlevel": "h"}[task]
    ratio_of = {f"{prefix}{r}_seg": r for r in (30, 50, 70)}
    sub = deg[(deg["task"] == task) & deg["scenario"].isin(ratio_of)].copy()
    if sub.empty:
        return None
    sub["ratio"] = sub["scenario"].map(ratio_of)

    fig, ax = plt.subplots(figsize=(6, 4))
    for arch, grp in sub.groupby("architecture"):
        agg = grp.groupby("ratio")["degradation"].agg(["mean", "sem"]).reset_index()
        color = ARCH_COLOR.get(arch, SERIES["blue"])
        ax.plot(agg["ratio"], agg["mean"], marker="o", markersize=6, color=color,
                label=ARCH_LABEL.get(arch, arch))
        ax.fill_between(agg["ratio"], agg["mean"] - 1.96 * agg["sem"],
                        agg["mean"] + 1.96 * agg["sem"], color=color, alpha=0.15,
                        linewidth=0)
        last = agg.iloc[-1]
        ax.annotate(ARCH_LABEL.get(arch, arch), (last["ratio"], last["mean"]),
                    xytext=(6, 0), textcoords="offset points", va="center",
                    fontsize=9, color=color)
    ax.axhline(0, color=INK, linewidth=0.9)
    _finish(ax, title=f"{TASK_LABEL[task]}：相对自身完整标签基线的 NSE 下降",
            xlabel="人工缺失比例 %", ylabel="NSE 下降（越大越差）")
    return _save(fig, name or f"fig_missing_degradation_{task}.png")


# ── Major 8：水文过程线 ───────────────────────────────────────────────────────

def peak_window(series: pd.DataFrame, task="flow", days=30) -> tuple:
    """取观测最大值前后各 days/2 天，作为洪峰事件的放大窗口。"""
    frame = series.copy()
    frame["time"] = pd.to_datetime(frame["time"])
    obs = frame[f"obs_{task}"]
    if obs.notna().sum() == 0:
        return None
    center = frame.loc[obs.idxmax(), "time"]
    half = pd.Timedelta(days=days / 2)
    return (center - half, center + half)


def fig_hydrograph(series, basin: str, task="flow", window=None, name=None,
                   title=None):
    """代表性流域的观测与预测过程线，可选放大某次洪峰事件（Major 8）。

    series 既可以是单个 DataFrame，也可以是 {架构名: DataFrame}。传字典时
    多个架构画在同一张图上——审稿意见要的是"看得出差别在哪一段"，只画一条
    预测线无法回答这个问题。
    """
    frames = series if isinstance(series, dict) else {None: series}
    obs_col, pred_col = f"obs_{task}", f"pred_{task}"

    prepared = {}
    for label, raw in frames.items():
        # 单任务模型只导出自己那一路的时序，画另一个任务时直接跳过该条曲线，
        # 不能让它把整张图否掉
        if obs_col not in raw or pred_col not in raw:
            continue
        frame = raw.copy()
        frame["time"] = pd.to_datetime(frame["time"])
        if window:
            mask = ((frame["time"] >= pd.Timestamp(window[0]))
                    & (frame["time"] <= pd.Timestamp(window[1])))
            frame = frame[mask]
        if not frame.empty:
            prepared[label] = frame
    if not prepared:
        return None

    fig, ax = plt.subplots(figsize=(9, 3.4))
    first = next(iter(prepared.values()))
    ax.plot(first["time"], first[obs_col], color=INK, linewidth=1.6, label="观测")
    for label, frame in prepared.items():
        ax.plot(frame["time"], frame[pred_col], linewidth=1.3,
                color=ARCH_COLOR.get(label, SERIES["orange"]),
                label=ARCH_LABEL.get(label, "预测"))
    ax.legend(loc="upper right", ncol=len(prepared) + 1, frameon=False)
    unit = "m³/s" if task == "flow" else "m"
    _finish(ax, title=title or f"流域 {basin} — {TASK_LABEL[task]}",
            xlabel="时间", ylabel=f"{TASK_LABEL[task]}（{unit}）")
    fig.autofmt_xdate()
    return _save(fig, name or f"fig_hydrograph_{basin}_{task}.png")


RUNS_DIR = RESULTS_ROOT / "runs"


def _runs_with_timeseries() -> dict:
    """已导出测试期时序的运行：{架构: run_key}，只取完整标签的主实验运行。"""
    found = {}
    for series_dir in sorted(RUNS_DIR.glob("*/timeseries")):
        key = series_dir.parent.name
        if "_complete_" not in key:
            continue
        for arch in ("single_waterlevel", "single_flow", "capacity_matched",
                     "dual_head", "wl2d"):
            if f"_{arch}_" in key:
                found.setdefault(arch, key)
                break
    return found


def _load_series(run_key: str, basin: str):
    path = RUNS_DIR / run_key / "timeseries" / f"{basin}.csv"
    return pd.read_csv(path) if path.exists() else None


def _pick_basins(metrics: pd.DataFrame, run_key: str, task="flow", n=3) -> list:
    """按该运行的 NSE 分位挑代表流域：好、中、差各一个，避免只展示成功案例。"""
    sub = metrics[(metrics["run_key"] == run_key) & (metrics["task"] == task)]
    sub = sub[np.isfinite(sub["nse"])].sort_values("nse")
    if sub.empty:
        return []
    available = {p.stem for p in (RUNS_DIR / run_key / "timeseries").glob("*.csv")}
    sub = sub[sub["basin"].isin(available)]
    if sub.empty:
        return []
    idx = np.linspace(0.1, 0.9, n)
    return [sub.iloc[int(round(q * (len(sub) - 1)))]["basin"] for q in idx]


def _tag(config: str) -> str:
    """文件名后缀：base 配置不加后缀，其余配置以 _<config> 区分。"""
    return "" if config == "base" else "_" + config.replace("+", "_")


def figures_from_results():
    """依赖训练结果的图：主实验、缺失下降、配对差值、过程线。

    每个可比配置各出一套（base / physical / L480+extended …），文件名以配置
    区分；过程线只对 base 出，其他配置没有导出时序。
    """
    from evaluation.run_config import load_summary
    from evaluation.stats import degradation, paired_differences

    if not (SUMMARY_DIR / "all_metrics.csv").exists():
        print("尚无汇总结果，训练相关的图稍后生成")
        return
    metrics, _ = load_summary()

    # 只为"有完整标签 + 有留出情景"的配置出整套图，单点消融不出
    configs = [c for c in sorted(set(metrics["config"]))
               if (metrics["config"] == c).sum()
               and set(metrics[metrics["config"] == c]["scenario"]) >= {"complete", "q_holdout70"}]
    if not configs:
        print("没有任何配置同时含完整标签与留出情景，跳过依赖训练结果的图")
        return

    for config in configs:
        sub = metrics[metrics["config"] == config]
        tag = _tag(config)
        print(f"配置 {config}:")

        if sub[sub["scenario"] == "complete"]["model_seed"].nunique() >= 2:
            fig_main_experiment(sub, config=config)

        # Major 3：缺失下降曲线。基线是同一架构自己的完整标签结果
        deg = degradation(sub, baseline_scenario="complete", metric="nse")
        for task in ("flow", "waterlevel"):
            if fig_missing_degradation(deg, task=task,
                                       name=f"fig_missing_degradation_{task}{tag}.png") is None:
                print(f"  {TASK_LABEL[task]}：缺失情景不足，跳过下降图")

        # Minor 8：逐流域配对差值。第一张是完整标签下的多任务增益（效应量很小，
        # 必须让读者看到分布而不是只看均值）；后面几张是论文主命题——某些流域
        # 的径流标签被整段删除后，水位监督能不能顶上
        pairs = [("complete", None, f"完整标签：双头多任务 − 单任务 Q（{config}）",
                  f"fig_paired_complete_flow{tag}.png")]
        for ratio in (30, 50, 70):
            pairs.append((f"q_holdout{ratio}", True,
                          f"留出 {ratio}% 流域：被留出流域上的双头 − 单任务 Q（{config}）",
                          f"fig_paired_holdout{ratio}_flow{tag}.png"))
        for scenario, held, title, name in pairs:
            part = sub[sub["scenario"] == scenario]
            if held is not None and "held_out" in part.columns:
                flag = part["held_out"].astype(str).str.lower().isin(("true", "1"))
                part = part[flag == held]
            a = part[part["architecture"] == "dual_head"]
            b = part[part["architecture"] == "single_flow"]
            if a.empty or b.empty:
                print(f"  {scenario}：缺少 dual_head 或 single_flow，跳过")
                continue
            diff = paired_differences(a, b, "flow", "nse")
            if len(diff) < 3:
                print(f"  {scenario}：可配对流域不足，跳过")
                continue
            fig_paired_difference(diff, title, name)

    # Major 8：代表性流域过程线，同一流域上把两个架构画在一起（仅 base 有时序）
    runs = _runs_with_timeseries()
    if not runs:
        print("无已导出的测试期时序，跳过过程线图")
        return
    base = metrics[metrics["config"] == "base"]
    print("生成过程线图:")
    ref = runs.get("dual_head") or next(iter(runs.values()))
    for basin in _pick_basins(base, ref, task="flow"):
        frames = {arch: s for arch, key in runs.items()
                  if (s := _load_series(key, basin)) is not None}
        if not frames:
            continue
        nse = base[(base["run_key"] == ref) & (base["basin"] == basin)
                   & (base["task"] == "flow")]["nse"]
        tag = f"（NSE {nse.iloc[0]:.3f}）" if len(nse) else ""
        fig_hydrograph(frames, basin, task="flow",
                       title=f"流域 {basin} — {TASK_LABEL['flow']} 测试期{tag}",
                       name=f"fig_hydrograph_{basin}_flow.png")
        window = peak_window(next(iter(frames.values())), task="flow", days=30)
        if window:
            fig_hydrograph(frames, basin, task="flow", window=window,
                           title=f"流域 {basin} — 最大洪峰事件放大",
                           name=f"fig_hydrograph_{basin}_flow_peak.png")


def main():
    setup_style()
    print("生成不依赖训练结果的图（Major 4 / Major 9）:")
    fig_split_timeline()
    fig_label_coverage()

    from pipeline.dataset import PreparedData
    from pipeline.masking import build_hidden, real_gap_start_months
    prep = PreparedData()
    tables = {"真实缺口": {}, "季节匹配段": {}, "均匀段": {}}
    for task in ("flow", "waterlevel"):
        tables["真实缺口"][task] = real_gap_start_months(prep, task).tolist()
        for label, seasonal in (("季节匹配段", True), ("均匀段", False)):
            _, st = build_hidden(prep, {task: 0.5}, mechanism="segment",
                                 mask_seed=42, seasonal=seasonal)
            tables[label][task] = st["per_task"][task]["start_month_hist"]
    fig_gap_seasonality(tables)

    figures_from_results()


if __name__ == "__main__":
    main()
