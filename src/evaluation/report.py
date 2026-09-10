"""把实验矩阵结果汇总为面向论文的表格。

产出（写入 results/summary/）：
- main_experiment.csv       主实验：各架构跨种子的均值与标准差
- paired_tests.csv          配对检验：Wilcoxon + bootstrap CI + 效应量
- missing_degradation.csv   缺失实验：各模型相对自身完整标签基线的下降
- seed_variance.csv         模型种子方差与掩膜种子方差分开汇总
- short_test_sensitivity.csv 含/不含 4 个短测试期流域的聚合对比
- training_budget.csv       实际轮数、标签曝光、更新步数（训练预算透明度）
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

from evaluation.stats import compare_models, degradation, paired_test, seed_variance  # noqa: E402
from evaluation.run_config import (SUMMARY_DIR, add_config_column,  # noqa: E402
                                   load_summary)
from pipeline.splits import load_splits  # noqa: E402

METRICS = ("nse", "kge", "rmse", "mae", "pbias", "peak_time_mae_h", "peak_magnitude_pct")

# 主实验里，每个任务应该拿哪个单任务模型作对照
SINGLE_FOR = {"flow": "single_flow", "waterlevel": "single_waterlevel"}


def main_experiment_table(metrics: pd.DataFrame) -> pd.DataFrame:
    """各架构在完整标签下的跨种子表现：先按流域平均，再看种子间波动。

    **按可比配置分开统计**，不同窗口长度/属性集/归一化方式的运行绝不混入同一均值。
    """
    sub = metrics[metrics["scenario"] == "complete"]
    rows = []
    for (config, arch, task), grp in sub.groupby(["config", "architecture", "task"]):
        per_seed = grp.groupby("model_seed")[list(METRICS)].mean(numeric_only=True)
        row = {"config": config, "architecture": arch, "task": task,
               "n_seeds": len(per_seed), "n_basins": grp["basin"].nunique()}
        for metric in METRICS:
            if metric not in per_seed:
                continue
            vals = per_seed[metric].to_numpy(dtype=float)
            vals = vals[np.isfinite(vals)]
            if vals.size == 0:
                continue
            row[f"{metric}_mean"] = float(vals.mean())
            row[f"{metric}_sd_across_seeds"] = float(vals.std(ddof=1)) if vals.size > 1 else 0.0
        rows.append(row)
    return pd.DataFrame(rows).sort_values(
        ["config", "task", "architecture"]).reset_index(drop=True)


def paired_tests_table(metrics: pd.DataFrame) -> pd.DataFrame:
    """回应 Major 1 与 Major 2 的全部关键对比，逐可比配置分别进行。"""
    rows = []
    for config, sub in metrics[metrics["scenario"] == "complete"].groupby("config"):
        available = set(sub["architecture"])
        comparisons = []
        for task, single in SINGLE_FOR.items():
            # 多任务是否优于单任务
            if {"dual_head", single} <= available:
                comparisons.append(("dual_head", single, task))
            # 级联是否优于普通双头
            if {"wl2d", "dual_head"} <= available:
                comparisons.append(("wl2d", "dual_head", task))
            # 级联是否优于参数量匹配对照——这才是"级联本身有没有用"的判据
            if {"wl2d", "capacity_matched"} <= available:
                comparisons.append(("wl2d", "capacity_matched", task))
            if {"capacity_matched", "dual_head"} <= available:
                comparisons.append(("capacity_matched", "dual_head", task))

        for a, b, task in comparisons:
            for metric in ("nse", "kge"):
                res = compare_models(sub, a, b, task, metric=metric)
                if res["summary"].get("n", 0) < 3:
                    continue
                rows.append({"config": config, **res["summary"]})
    return pd.DataFrame(rows)


def missing_degradation_table(metrics: pd.DataFrame) -> pd.DataFrame:
    """Major 3 的核心口径：各模型相对**自身**完整标签基线的下降幅度，
    并对"双头下降是否显著小于单任务下降"做配对检验。"""
    frames = []
    for config, sub in metrics.groupby("config"):
        part = degradation(sub, baseline_scenario="complete", metric="nse")
        if not part.empty:
            part["config"] = config
            frames.append(part)
    if not frames:
        return pd.DataFrame()
    deg = pd.concat(frames, ignore_index=True)

    rows = []
    for (config, scenario, task), grp in deg.groupby(["config", "scenario", "task"]):
        single = SINGLE_FOR[task]
        by_arch = {arch: g.set_index("basin")["degradation"]
                   for arch, g in grp.groupby("architecture")}
        for arch, series in by_arch.items():
            rows.append({"config": config, "scenario": scenario,
                         "task": task, "architecture": arch,
                         "n_basins": int(series.size),
                         "degradation_mean": float(series.mean()),
                         "degradation_median": float(series.median()),
                         "baseline_nse_mean": float(
                             grp[grp["architecture"] == arch]["baseline"].mean()),
                         "missing_nse_mean": float(
                             grp[grp["architecture"] == arch]["nse"].mean())})
        # 双头的下降是否显著小于对应单任务的下降
        if "dual_head" in by_arch and single in by_arch:
            common = by_arch["dual_head"].index.intersection(by_arch[single].index)
            diff = (by_arch[single].loc[common] - by_arch["dual_head"].loc[common]).to_numpy()
            test = paired_test(diff)
            rows.append({"config": config, "scenario": scenario, "task": task,
                         "architecture": f"{single} 下降 - dual_head 下降",
                         "n_basins": test.get("n"),
                         "degradation_mean": test.get("mean_diff"),
                         "degradation_median": test.get("median_diff"),
                         "ci_lo": test.get("mean_ci_lo"), "ci_hi": test.get("mean_ci_hi"),
                         "wilcoxon_p": test.get("wilcoxon_p"),
                         "cohens_d_paired": test.get("cohens_d_paired"),
                         "n_basins_dual_head_better": test.get("n_basins_improved")})
    return pd.DataFrame(rows).sort_values(
        ["config", "task", "scenario", "architecture"]).reset_index(drop=True)


def holdout_table(metrics: pd.DataFrame) -> pd.DataFrame:
    """论文主命题的结果表：某些流域完全没有径流训练标签时，水位监督能否顶上。

    对每个留出情景，分别在**被留出流域**和**保留流域**上比较单任务 Q 与
    双头模型的径流表现，并对逐流域差值做配对检验。被留出流域上的差值才是
    "用廉价水位数据替代昂贵径流数据"的直接证据。
    """
    if "held_out" not in metrics.columns:
        return pd.DataFrame()
    sub = metrics[metrics["scenario"].str.startswith("q_holdout", na=False)]
    sub = sub[sub["task"] == "flow"]
    if sub.empty:
        return pd.DataFrame()

    rows = []
    for (config, scenario), grp in sub.groupby(["config", "scenario"]):
        for group_name, part in (("被留出流域", grp[grp["held_out"] == True]),
                                 ("保留流域", grp[grp["held_out"] == False])):
            if part.empty:
                continue
            dual = part[part["architecture"] == "dual_head"].groupby("basin")["nse"].mean()
            base = part[part["architecture"] == "single_flow"].groupby("basin")["nse"].mean()
            common = dual.index.intersection(base.index)
            if len(common) < 3:
                continue
            diff = (dual.loc[common] - base.loc[common]).to_numpy()
            test = paired_test(diff)
            rows.append({
                "config": config, "scenario": scenario, "basin_group": group_name,
                "n_basins": len(common),
                "nse_single_flow": float(base.loc[common].mean()),
                "nse_dual_head": float(dual.loc[common].mean()),
                "mean_gain": test.get("mean_diff"),
                "ci_lo": test.get("mean_ci_lo"), "ci_hi": test.get("mean_ci_hi"),
                "median_gain": test.get("median_diff"),
                "wilcoxon_p": test.get("wilcoxon_p"),
                "cohens_d_paired": test.get("cohens_d_paired"),
                "n_basins_improved": test.get("n_basins_improved"),
                "n_basins_worsened": test.get("n_basins_worsened"),
            })
    return pd.DataFrame(rows).sort_values(
        ["config", "scenario", "basin_group"]).reset_index(drop=True)


def short_test_sensitivity(metrics: pd.DataFrame) -> pd.DataFrame:
    """含 / 不含 4 个短测试期流域的聚合对比（按用户决定：保留但单独标注）。"""
    payload = load_splits()
    short = {b for b, v in payload["splits"].items() if v.get("short_test")}
    sub = metrics[metrics["scenario"] == "complete"]
    rows = []
    for (config, arch, task), grp in sub.groupby(["config", "architecture", "task"]):
        full = grp.groupby("model_seed")["nse"].mean()
        trimmed = grp[~grp["basin"].isin(short)].groupby("model_seed")["nse"].mean()
        rows.append({
            "config": config, "architecture": arch, "task": task,
            "n_short_test_basins": len(short & set(grp["basin"])),
            "nse_mean_all": float(full.mean()),
            "nse_mean_excluding_short": float(trimmed.mean()),
            "difference": float(trimmed.mean() - full.mean()),
        })
    return pd.DataFrame(rows)


def training_budget_table(runs: pd.DataFrame) -> pd.DataFrame:
    """训练预算透明度：实际轮数、是否撞上限、标签曝光、更新步数。"""
    def _get(row, col, key):
        raw = row.get(col)
        if isinstance(raw, str):
            try:
                raw = json.loads(raw.replace("'", '"'))
            except json.JSONDecodeError:
                return np.nan
        return raw.get(key, np.nan) if isinstance(raw, dict) else np.nan

    out = runs.copy()
    for task in ("flow", "waterlevel"):
        out[f"train_labels_{task}"] = out.apply(
            lambda r: _get(r, "train_label_counts", task), axis=1)
    cols = ["run_key", "group", "scenario", "architecture", "model_seed", "mask_seed",
            "epochs_run", "best_epoch", "hit_epoch_cap", "n_train_samples",
            "n_updates_per_epoch", "total_updates", "train_labels_flow",
            "train_labels_waterlevel", "best_val_score", "elapsed_sec"]
    return out[[c for c in cols if c in out.columns]]


def main():
    metrics, runs = load_summary()
    SUMMARY_DIR.mkdir(parents=True, exist_ok=True)

    outputs = {
        "main_experiment.csv": main_experiment_table(metrics),
        "paired_tests.csv": paired_tests_table(metrics),
        "missing_degradation.csv": missing_degradation_table(metrics),
        "holdout_substitution.csv": holdout_table(metrics),
        "short_test_sensitivity.csv": short_test_sensitivity(metrics),
        "training_budget.csv": training_budget_table(runs),
    }
    if {"scenario", "mask_seed"} <= set(metrics.columns):
        outputs["seed_variance.csv"] = seed_variance(metrics)

    for name, frame in outputs.items():
        if frame is None or frame.empty:
            print(f"跳过 {name}（暂无对应结果）")
            continue
        path = SUMMARY_DIR / name
        frame.to_csv(path, index=False, encoding="utf-8-sig")
        print(f"已写入 {path}  （{len(frame)} 行）")

    main_tab = outputs["main_experiment.csv"]
    if not main_tab.empty:
        print("\n主实验（完整标签，跨模型种子）:")
        cols = [c for c in ("config", "architecture", "task", "n_seeds", "nse_mean",
                            "nse_sd_across_seeds", "kge_mean") if c in main_tab]
        print(main_tab[cols].to_string(index=False))

    hold = outputs.get("holdout_substitution.csv")
    if hold is not None and not hold.empty:
        print("按流域留出（论文主命题：水位监督能否替代缺失的径流标签）:")
        cols = [c for c in ("config", "scenario", "basin_group", "n_basins", "nse_single_flow",
                            "nse_dual_head", "mean_gain", "ci_lo", "ci_hi",
                            "wilcoxon_p") if c in hold]
        print(hold[cols].to_string(index=False))

    paired = outputs["paired_tests.csv"]
    if not paired.empty:
        print("\n配对检验（逐流域差值）:")
        cols = [c for c in ("config", "model_a", "model_b", "task", "metric", "n", "mean_diff",
                            "mean_ci_lo", "mean_ci_hi", "wilcoxon_p",
                            "n_basins_improved") if c in paired]
        print(paired[cols].to_string(index=False))


if __name__ == "__main__":
    main()
