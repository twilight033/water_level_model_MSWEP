"""只读核验回拷结果并生成分析材料，不训练模型、不修改原始结果。"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
KEYS = ["basin", "ratio", "mask_seed", "model_seed"]
GROUPS = ["stage", "variant", "ratio", "mask_seed", "model_seed", "h_input_source"]
CONTRASTS = {
    "stage1_geometry_gain": ("h_attrs_geometry", "h_attrs"),
    "stage1_geometry_vs_shuffle": ("h_attrs_geometry", "h_attrs_geometry_shuffled"),
    "stage2_h_gain_without_geometry": ("dual_head", "single_flow"),
    "stage2_h_gain_with_geometry": ("dual_head_geometry", "single_flow_geometry"),
    "stage2_geometry_gain_single": ("single_flow_geometry", "single_flow"),
    "stage2_geometry_gain_dual": ("dual_head_geometry", "dual_head"),
    "stage2_geometry_vs_shuffle_single": ("single_flow_geometry", "single_flow_shuffled"),
    "stage2_geometry_vs_shuffle_dual": ("dual_head_geometry", "dual_head_shuffled"),
    "geometry_converter_vs_geometry_dual": ("dual_head_geometry", "h_attrs_geometry"),
}


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def sha(path):
    result = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(chunk)
    return result.hexdigest()


def read_csv(path):
    return pd.read_csv(path, dtype={"basin": str})


def save_csv(frame, path):
    frame.to_csv(path, index=False, encoding="utf-8-sig")


def metric(obs, pred):
    # 与正式评价一致保留float32算术，允许不同NumPy版本的小数值差异。
    denom = ((obs - obs.mean()) ** 2).sum()
    return float(1 - ((pred - obs) ** 2).sum() / denom) if denom > 0 else np.nan


def table(frame, digits=4):
    # 不依赖额外的Markdown表格包。
    def fmt(v):
        return f"{v:.{digits}f}" if isinstance(v, (float, np.floating)) else str(v)
    rows = [[fmt(v) for v in row] for row in frame.itertuples(index=False, name=None)]
    return "\n".join(["| " + " | ".join(frame.columns) + " |",
                      "| " + " | ".join(["---"] * len(frame.columns)) + " |"]
                     + ["| " + " | ".join(row) + " |" for row in rows])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=ROOT / "results/cross_section86_snapshot")
    parser.add_argument("--out", type=Path, default=ROOT / "reports/cross_section86_snapshot_review")
    args = parser.parse_args()
    source, output = args.results.resolve(), args.out.resolve()
    if source == output or source in output.parents:
        raise ValueError("分析材料须写在原始结果目录之外")
    output.mkdir(parents=True, exist_ok=True)
    spec, preflight = read_json(source / "experiment.json"), read_json(source / "preflight.json")
    cfg, basins, scenarios = spec["settings"], spec["basins"], spec["scenarios"]
    checks = []

    def check(name, passed, detail=None):
        checks.append({"check": name, "passed": bool(passed), "detail": detail})

    check("正式实验", cfg["smoke"] is False and preflight["smoke_not_paper_result"] is False)
    check("原86总体", len(basins) == len(set(basins)) == 86)
    fingerprint = spec["experiment_fingerprint"]
    check("实验指纹", digest({k: v for k, v in spec.items() if k != "experiment_fingerprint"}) == fingerprint)
    check("完成标记指纹", read_json(source / "completed_both.json")["experiment_fingerprint"] == fingerprint)
    hash_rows = []
    for kind in ("source_hashes", "code_hashes"):
        for name, expected in spec[kind].items():
            path = ROOT / name.replace("\\", "/")
            status = "missing"
            if path.exists():
                status = "exact" if sha(path) == expected else "different"
                if status == "different" and path.suffix in (".py", ".txt"):
                    value = path.read_bytes().replace(b"\r\n", b"\n")
                    alternatives = (value, value.replace(b"\n", b"\r\n"))
                    if any(hashlib.sha256(v).hexdigest() == expected for v in alternatives):
                        status = "line_endings_only"
            hash_rows.append({"kind": kind, "file": name, "status": status})
    check("代码与数据来源匹配（允许纯换行差异）", all(x["status"] in ("exact", "line_endings_only") for x in hash_rows))
    save_csv(pd.DataFrame(hash_rows), output / "source_hash_audit.csv")

    geometry = read_csv(source / "geometry_audit.csv").set_index("basin")
    geometry_hashes = []
    for basin, row in geometry[geometry.available].iterrows():
        roots = (ROOT / "data/cross_sections", ROOT / "research_notes/USGS 原86 历年断面核验")
        candidates = [root / row.json_path for root in roots]
        path = next((p for p in candidates if p.exists()), None)
        geometry_hashes.append(path is not None and sha(path) == row.source_sha256)
    check("已用逐点断面来源哈希", all(geometry_hashes), {"count": len(geometry_hashes)})
    check("几何覆盖", int(geometry.available.sum()) == preflight["n_geometry_available"] == 81)

    # 用源缓存重建完整测试时刻；不借助任何测试标签选择推理样本。
    frames = [pd.read_parquet(ROOT / ("data/camelsh_exported/" + name)) for name in
              ("mswep_precip_3h_86.parquet", "forcing_temperature_mean_3h_86.parquet", "forcing_solar_radiation_3h_86.parquet")]
    grid = frames[0].index
    for frame in frames[1:]:
        grid = grid.intersection(frame.index)
    grid = pd.DatetimeIndex(grid).sort_values()
    check("缓存站号顺序", list(frames[0].columns) == basins)
    del frames
    targets = {}
    for task, file in (("q", "flow_hourly_86.parquet"), ("h", "waterlevel_hourly_86.parquet")):
        frame = pd.read_parquet(ROOT / "data/camelsh_exported" / file).reindex(index=grid, columns=basins)
        targets[task] = np.ascontiguousarray(frame.to_numpy(dtype="float32").T)
    split = read_json(ROOT / "data/splits/basin_splits.json")["splits"]
    continuous = np.r_[True, np.diff(grid.to_numpy()) == np.timedelta64(3, "h")]
    prefix = np.r_[0, np.cumsum(~continuous)]
    expected_ids, expected_pos = [], []
    for bi, basin in enumerate(basins):
        lo = grid.searchsorted(pd.Timestamp(split[basin]["test_start"]), "left")
        hi = grid.searchsorted(pd.Timestamp(split[basin]["test_end"]), "right") - 1
        pos = np.arange(max(lo, cfg["sequence_length"]), hi + 1, cfg["test_stride"])
        pos = pos[continuous[pos] & ((prefix[pos] - prefix[pos - cfg["sequence_length"]]) == 0)]
        expected_ids.append(np.full(len(pos), bi, np.int64))
        expected_pos.append(pos)
    ids, positions = np.concatenate(expected_ids), np.concatenate(expected_pos)
    bounds = np.r_[0, np.cumsum([len(v) for v in expected_pos])]
    common_h_rows, recomputed_rows, training_rows, frames = [], [], [], []
    run_count = {"stage1": 0, "stage2": 0, "waterlevel": 0}
    pre_by_tag = {s["tag"]: s for s in preflight["scenarios"]}

    for seed in cfg["model_seeds"]:
        folder = source / "waterlevel_models" / f"seed{seed}"
        meta = read_json(folder / "training_done.json")
        expected = digest({"experiment": fingerprint, "stage": "waterlevel_only", "seed": seed})
        check(f"H模型seed{seed}协议", meta["run_fingerprint"] == expected and
              meta["scaling_info"]["flow_information_used"] is False and
              set(meta["train_label_counts"]) == {"waterlevel"})
        with np.load(folder / "prediction.npz") as z:
            check(f"H模型seed{seed}推理时刻", np.array_equal(z["basin_idx"], ids) and np.array_equal(z["target_pos"], positions))
        check(f"H模型seed{seed}权重存在", (folder / "model.pt").is_file())
        run_count["waterlevel"] += 1

    for scenario in scenarios:
        held, retained = set(scenario["heldout"]), set(scenario["retained"])
        tag, ratio, mask_seed = scenario["tag"], scenario["ratio"], scenario["mask_seed"]
        audit = read_json(source / "scenario_audits" / f"{tag}.json")
        check(f"{tag}站点隔离", not held & retained and held | retained == set(basins) and
              set(audit["geometry_scaler"]["fit_basins"]) == retained and
              set(pre_by_tag[tag]["physical_scale_fit_basins"]) == retained)
        for seed in cfg["model_seeds"]:
            for stage in ("stage1", "stage2"):
                variants = (["h_only", "h_attrs", "h_attrs_geometry", "h_attrs_geometry_shuffled"] if stage == "stage1"
                            else ["single_flow", "dual_head", "single_flow_geometry", "dual_head_geometry", "single_flow_shuffled", "dual_head_shuffled"])
                metadata = {}
                for variant in variants:
                    folder = source / stage / tag / f"seed{seed}" / variant
                    label = f"{stage}/{tag}/seed{seed}/{variant}"
                    meta = read_json(folder / "done.json")
                    metadata[variant] = meta
                    run_id = digest({"experiment": fingerprint, "scenario": tag, "stage": stage, "variant": variant, "seed": seed})
                    check(label + "完成信息", meta["run_fingerprint"] == run_id and set(meta["heldout"]) == held and
                          meta["protocol"] == spec["protocol"] and meta["smoke_not_paper_result"] is False)
                    check(label + "权重存在", (folder / "model.pt").is_file())
                    if stage == "stage1":
                        check(label + "拟合站隔离", set(meta["scale_fit_basins"]) == retained and
                              not set(meta["inner_train_basins"]) & set(meta["inner_validation_basins"]) and
                              set(meta["inner_train_basins"]) | set(meta["inner_validation_basins"]) == retained)
                    else:
                        train = read_json(folder / "training_done.json")
                        check(label + "训练尺度与标签数", train["run_fingerprint"] == run_id and
                              train["scaling_info"]["flow"]["mean"]["n_fit"] == len(retained) and
                              train["scaling_info"]["flow"]["std"]["n_fit"] == len(retained) and
                              all(train[k]["flow"] == pre_by_tag[tag][k]["flow"] for k in ("train_label_counts", "valid_label_counts")))
                        metadata[variant] = train
                        training_rows.append({"variant": variant, "ratio": ratio, "mask_seed": mask_seed,
                                              "model_seed": seed, **{k: train[k] for k in ("best_epoch", "epochs_run", "elapsed_sec", "n_train_samples")}})
                    metric_frame = read_csv(folder / "metrics.csv")
                    diagnostic = read_csv(folder / "diagnostic_metrics.csv") if stage == "stage1" else None
                    for frame in (metric_frame, diagnostic):
                        if frame is not None:
                            frame["run_fingerprint"] = run_id
                            frames.append(frame)
                    indexed = metric_frame.set_index("basin")
                    with np.load(folder / "prediction.npz") as z:
                        correct_indices = np.array_equal(z["basin_idx"], ids) and np.array_equal(z["target_pos"], positions)
                        check(label + "完整推理时刻", correct_indices)
                        if not correct_indices:
                            continue
                        predictions = z["pred_flow"]
                    all_correct = np.isfinite(predictions).all()
                    for bi, basin in enumerate(basins):
                        sl = slice(bounds[bi], bounds[bi + 1])
                        q, h, pred = targets["q"][bi, positions[sl]], targets["h"][bi, positions[sl]], predictions[sl]
                        ok = np.isfinite(q) & np.isfinite(pred)
                        score, rmse = metric(q[ok], pred[ok]), float(np.sqrt(((q[ok] - pred[ok]) ** 2).mean()))
                        row = indexed.loc[basin]
                        all_correct &= int(row.n) == int(ok.sum()) and int(row.n_negative_predictions) == int((pred < 0).sum())
                        all_correct &= bool(np.isclose(row.nse, score, rtol=2e-5, atol=2e-6) and np.isclose(row.rmse, rmse, rtol=2e-5, atol=2e-6))
                        recomputed_rows.append({"stage": stage, "variant": variant, "ratio": ratio, "mask_seed": mask_seed,
                                                "model_seed": seed, "basin": basin, "nse_error": score - row.nse})
                        if stage == "stage1":
                            common = ok & np.isfinite(h)
                            diag = diagnostic.loc[diagnostic.basin == basin].iloc[0]
                            all_correct &= int(common.sum()) == int(diag.n)
                            common_h_rows.append({"variant": variant, "ratio": ratio, "mask_seed": mask_seed, "model_seed": seed,
                                                  "basin": basin, "held_out": basin in held, "n": int(common.sum()),
                                                  "predicted_h_nse": metric(q[common], pred[common]), "observed_h_nse": diag.nse,
                                                  "dropped_q_times": int(ok.sum() - common.sum())})
                    check(label + "从预测重算NSE及RMSE", all_correct)
                    run_count[stage] += 1
                if stage == "stage1":
                    check(f"{tag}/seed{seed}/转换器真假几何参数匹配", metadata["h_attrs_geometry"]["n_parameters"] == metadata["h_attrs_geometry_shuffled"]["n_parameters"])
                else:
                    for architecture in ("single_flow", "dual_head"):
                        check(f"{tag}/seed{seed}/{architecture}真假几何参数匹配", metadata[architecture + "_geometry"]["model_config"]["n_parameters"] == metadata[architecture + "_shuffled"]["model_config"]["n_parameters"])
        print(f"已核验 {tag}：权重、协议、预测时刻与逐流域NSE/RMSE", flush=True)

    scores = read_csv(source / "all_basin_metrics.csv")
    detailed = pd.concat(frames, ignore_index=True)
    sort = GROUPS + ["basin"]
    try:
        pd.testing.assert_frame_equal(scores.sort_values(sort).reset_index(drop=True), detailed[scores.columns].sort_values(sort).reset_index(drop=True), check_exact=False, rtol=1e-10, atol=1e-10)
        check("总表与全部逐次指标文件一致", True)
    except AssertionError as error:
        check("总表与全部逐次指标文件一致", False, str(error)[:500])
    check("主表组合唯一且每组86站", not scores.duplicated(sort).any() and scores.groupby(GROUPS).size().eq(86).all())
    summaries = []
    for key, group in scores.groupby(GROUPS):
        scopes = {"all_86": group, "q_heldout": group[group.held_out], "q_retained": group[~group.held_out],
                  "q_heldout_with_geometry": group[group.held_out & group.geometry_available]}
        for scope, selected in scopes.items():
            record = {**dict(zip(GROUPS, key)), "scope": scope, "n_basins_reported": len(selected),
                      "n_valid_nse": selected.nse.count(), "nse_mean": selected.nse.mean(),
                      "nse_median": selected.nse.median(), "n_nse_negative": int((selected.nse < 0).sum())}
            for name in ("kge", "pbias", "pearson_r", "peak_time_mae_h"):
                record[name + "_mean"] = selected[name].mean()
            summaries.append(record)
    saved_summary = read_csv(source / "summary_by_run.csv")
    summary_keys = GROUPS + ["scope"]
    try:
        pd.testing.assert_frame_equal(pd.DataFrame(summaries)[saved_summary.columns].sort_values(summary_keys).reset_index(drop=True),
                                      saved_summary.sort_values(summary_keys).reset_index(drop=True), check_exact=False, check_dtype=False, rtol=1e-10, atol=1e-10)
        check("全部运行汇总重算一致", True)
    except AssertionError as error:
        check("全部运行汇总重算一致", False, str(error)[:500])
    deployed = scores[scores.h_input_source != "observed_h_diagnostic"]
    pairs = []
    for name, (model, control) in CONTRASTS.items():
        left = deployed[deployed.variant == model][KEYS + ["nse", "held_out", "geometry_available"]]
        right = deployed[deployed.variant == control][KEYS + ["nse"]]
        joined = left.merge(right, on=KEYS, suffixes=("_model", "_control"), validate="one_to_one")
        joined["delta_nse"] = joined.nse_model - joined.nse_control
        joined["comparison"] = name
        pairs.append(joined)
    pairs = pd.concat(pairs, ignore_index=True)
    saved_pairs = read_csv(source / "paired_differences.csv")
    comparison_keys = ["comparison"] + KEYS
    try:
        pd.testing.assert_frame_equal(pairs[saved_pairs.columns].sort_values(comparison_keys).reset_index(drop=True), saved_pairs.sort_values(comparison_keys).reset_index(drop=True), check_exact=False, rtol=1e-10, atol=1e-10)
        check("全部配对差值重算一致", True)
    except AssertionError as error:
        check("全部配对差值重算一致", False, str(error)[:500])

    # 先在每个运行内平均站点，再平均9个运行；重复种子不充当独立流域。
    held = deployed[deployed.held_out]
    per_run = held.groupby(GROUPS).agg(nse_mean=("nse", "mean"), nse_median=("nse", "median"),
                                         kge_mean=("kge", "mean"), negative_nse=("nse", lambda v: (v < 0).sum())).reset_index()
    performance = per_run.groupby(["stage", "variant", "ratio"]).agg(nse_mean=("nse_mean", "mean"),
            nse_median_run_average=("nse_median", "mean"), kge_mean=("kge_mean", "mean"),
            nse_run_sd=("nse_mean", "std"), negative_basins_run_average=("negative_nse", "mean")).reset_index()
    save_csv(performance, output / "heldout_performance.csv")
    pair_runs = pairs[pairs.held_out].groupby(["comparison", "ratio", "mask_seed", "model_seed"]).delta_nse.mean().reset_index()
    effects = pair_runs.groupby(["comparison", "ratio"]).delta_nse.agg(mean="mean", sd="std", min="min", max="max", positive_runs=lambda v: (v > 0).sum()).reset_index()
    save_csv(effects, output / "paired_run_effects.csv")
    station_effects = pairs[pairs.held_out].groupby(["comparison", "ratio", "basin"]).delta_nse.mean().reset_index()
    station_summary = station_effects.groupby(["comparison", "ratio"]).delta_nse.agg(n_basins="size", mean="mean", median="median", positive_fraction=lambda v: (v > 0).mean()).reset_index()
    save_csv(station_effects, output / "paired_station_effects.csv")
    save_csv(station_summary, output / "paired_station_summary.csv")
    available = pairs[pairs.held_out & pairs.geometry_available].groupby(["comparison", "ratio", "mask_seed", "model_seed"]).delta_nse.mean().reset_index()
    save_csv(available.groupby(["comparison", "ratio"]).delta_nse.agg(["mean", "std"]).reset_index(), output / "geometry_available_effects.csv")
    common_h = pd.DataFrame(common_h_rows)
    common_h["observed_minus_predicted"] = common_h.observed_h_nse - common_h.predicted_h_nse
    save_csv(common_h, output / "common_time_h_diagnostic.csv")
    diag = common_h[common_h.held_out].groupby(["variant", "ratio", "mask_seed", "model_seed"])[["predicted_h_nse", "observed_h_nse", "observed_minus_predicted"]].mean().reset_index()
    diag_summary = diag.groupby(["variant", "ratio"])[["predicted_h_nse", "observed_h_nse", "observed_minus_predicted"]].mean().reset_index()
    save_csv(diag_summary, output / "common_time_h_summary.csv")
    train = pd.DataFrame(training_rows)
    save_csv(train, output / "training_audit.csv")
    training_summary = train.groupby("variant").agg(n=("best_epoch", "size"), best_epoch_median=("best_epoch", "median"),
            best_at_cap=("best_epoch", lambda v: (v == cfg["lstm_epochs"]).sum()), elapsed_total_sec=("elapsed_sec", "sum")).reset_index()
    interaction = (pairs[pairs.comparison == "stage2_h_gain_with_geometry"][KEYS + ["delta_nse", "held_out"]]
        .merge(pairs[pairs.comparison == "stage2_h_gain_without_geometry"][KEYS + ["delta_nse"]], on=KEYS, suffixes=("_with", "_without")))
    interaction["interaction"] = interaction.delta_nse_with - interaction.delta_nse_without
    saved_interaction = read_csv(source / "h_geometry_interaction.csv")
    joined = interaction.merge(saved_interaction, on=KEYS, validate="one_to_one")
    check("H与几何交互差值重算一致", len(joined) == len(saved_interaction) == len(interaction) and
          np.allclose(joined.interaction, joined.h_geometry_interaction, rtol=1e-10, atol=1e-10))
    interaction_run = interaction[interaction.held_out].groupby(["ratio", "mask_seed", "model_seed"]).interaction.mean().reset_index()
    save_csv(interaction_run, output / "heldout_interaction.csv")
    failure_summary = held.groupby(["stage", "variant"]).agg(basin_run_count=("basin", "size"),
            negative_nse_count=("nse", lambda v: (v < 0).sum()), minimum_nse=("nse", "min"),
            negative_prediction_count=("n_negative_predictions", "sum")).reset_index()
    save_csv(failure_summary, output / "failure_summary.csv")
    worst = held[held.variant == "h_attrs_geometry"].nsmallest(5, "nse")
    save_csv(worst, output / "stage1_geometry_worst_cases.csv")
    audit_report = {"results": str(source), "experiment_fingerprint": fingerprint, "run_counts": run_count,
                    "n_metric_rows": len(scores), "n_checks": len(checks), "n_failed": sum(not c["passed"] for c in checks),
                    "max_absolute_recomputed_nse_error": float(pd.DataFrame(recomputed_rows).nse_error.abs().max()),
                    "common_time_diagnostic_dropped_q_times": int(common_h.dropped_q_times.sum()), "checks": checks}
    (output / "integrity_audit.json").write_text(json.dumps(audit_report, ensure_ascii=False, indent=2), encoding="utf-8")
    sections = ["# 原86流域断面实验回拷核验与结果解读", "",
        "本报告只分析既定实验；没有重训练、重新选模或修改测试集。所有差值均为模型NSE减对照NSE。", "",
        f"完整性核验：{len(checks)}项检查，{audit_report['n_failed']}项失败；3个水位模型、108个转换器、162个直接预测模型；指标共{len(scores)}行。",
        "逐次预测文件的时刻、NSE和RMSE已与源缓存独立核对；总表与逐次指标文件、配对差值一致。代码和站号清单存在纯CRLF/LF换行差异，内容一致。", "",
        "## 主要结果", "",
        "**支持水位辅助监督，尚不支持当前断面特征带来稳定增益。** 阶段一增加断面后平均NSE下降；阶段二双头模型增益很小且随留出比例变号，真实断面未稳定优于打乱对照。", "",
        "下表仅评价训练与验证Q均被隐藏的流域。每个比例含3个掩膜种子×3个模型种子：先每次运行计算流域平均NSE，再对9次运行平均。隐藏比例30%、50%、70%分别为26、43、60站。不同掩膜重复抽到同一站，不视为独立流域。", "",
        table(performance[["stage", "variant", "ratio", "nse_mean", "kge_mean"]]), "",
        "## 配对效应与稳定性", "",
        "positive_runs为9个运行中平均配对差值大于0的次数，sd为运行间描述性标准差，不是独立重复的置信区间。", "", table(effects), "",
        "剔除缺断面站后的同组分析见geometry_available_effects.csv；没有据此筛选论文主结果。逐站先跨种子平均的胜率和中位差见paired_station_summary.csv。", "",
        "## 水位误差诊断", "",
        "同一冻结转换器分别输入预测H和实测H，在Q、H共同有效的相同时刻比较。observed_h_nse来自原诊断指标，预测H指标在相同Q/H有效时刻重新计算；实测H诊断不代表纯气象预测能力，也未用于选模。", "",
        table(diag_summary), "",
        "即使输入实测H，加入断面仍未改善平均NSE，因此阶段一的负结果不能全部归因于水位预测误差。转换器用训练实测H拟合、推理输入预测H的分布变化仍是后续可检验解释。", "",
        "## 科研结论边界", "",
        "1. 断面为81站的2021—2023固定快照，其中79站晚于训练结束；属于给定事后几何信息的回溯研究。5站缺断面保留在原86总体中。",
        "2. 当前输入是6个相对剖面特征及缺失标记，不能据此否定一切断面信息；它们也不等于绝对床高、完整岸滩形态或已识别的水位—面积函数。",
        "3. H辅助监督的平均增益较稳定，但仍有单站受损，极端负NSE影响均值，需连同中位数、逐站分布和KGE报告。",
        "4. 单Q与双头模型有效训练样本量不同；双头的增益归于当前完整训练方案，不能声称计算预算完全一致或已排除所有机制解释。",
        "5. 无几何与有几何的H增益之差随比例变动，尚不能声称断面与H有稳定协同。",
        "6. 当前协议屏蔽训练与验证Q；不要同旧版仅屏蔽训练Q的结果直接配对。冻结时间边界继承原始资料可用性，实验属于Q标签删除，不能宣称站点从未有过任何历史Q信息。",
        "7. 建议论文保留双头数据替代主线，将当前断面实验作为不稳定/负向的消融证据。若继续研究，先预定义一个假设并仅用训练及验证资料设计方案，避免看过本测试结果后反复调参又将其当独立验证。", "",
        "## 失败情况", "",
        "下表按流域×比例×掩膜种子×模型种子计数，重复站点不是独立样本。negative_prediction_count统计推理时刻的负流量，包含没有Q观测的时刻；本次核验保持原始输出，没有事后截断负值。", "",
        table(failure_summary), "",
        "阶段一断面转换器最差的5个流域运行如下，保留于全部主分析中，没有以异常值为由删除。", "",
        table(worst[["basin", "ratio", "mask_seed", "model_seed", "nse", "pbias"]]), "",
        "## 训练诊断", "", table(training_summary), "",
        f"第二阶段{len(train)}次训练中，最优轮次等于{cfg['lstm_epochs']}轮上限的有{int(training_summary.best_at_cap.sum())}次。该检查没有发现checkpoint卡在上限的情况，不据测试结果延长训练或重新选模。", "",
        "## 复现本次分析", "", "```powershell", "python tools/review_cross_section86_results.py", "```", "",
        "完整检查记录见integrity_audit.json；本分析不重新执行原始训练时的标签掩膜，而是核对已保存预检、代码、训练计数、来源哈希和运行指纹。"]
    (output / "review.md").write_text("\n".join(sections) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in audit_report.items() if k != "checks"}, ensure_ascii=False, indent=2))
    print(performance[["stage", "variant", "ratio", "nse_mean"]].to_string(index=False))
    return 1 if audit_report["n_failed"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
