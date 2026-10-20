"""只读核验两套强配置回拷结果；仅计算分析统计，不调用训练或推理。"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from review_cross_section86_results import (
    CONTRASTS, ROOT, digest, metric, read_csv, read_json, save_csv, sha, table,
)


KEYS = ["basin", "ratio", "mask_seed", "model_seed", "target_scaling"]
GROUPS = ["stage", "variant", "ratio", "mask_seed", "model_seed", "h_input_source", "target_scaling"]
VARIANTS = {
    "stage1": ["h_only", "h_attrs", "h_attrs_geometry", "h_attrs_geometry_shuffled"],
    "stage2": ["single_flow", "dual_head", "single_flow_geometry", "dual_head_geometry",
               "single_flow_shuffled", "dual_head_shuffled"],
}
EXTRA_CONTRASTS = {
    "stage1_attributes_gain": ("h_attrs", "h_only"),
    "stage1_geometry_vs_h_only": ("h_attrs_geometry", "h_only"),
    "dual_vs_h_only": ("dual_head", "h_only"),
}


class Audit:
    def __init__(self):
        self.checks = []

    def check(self, name, passed, detail=None):
        self.checks.append({"check": name, "passed": bool(passed), "detail": detail})

    def frames(self, name, left, right, keys):
        try:
            pd.testing.assert_frame_equal(
                left[right.columns].sort_values(keys).reset_index(drop=True),
                right.sort_values(keys).reset_index(drop=True),
                check_dtype=False, check_exact=False, rtol=1e-9, atol=1e-9,
            )
            self.check(name, True)
        except AssertionError as error:
            self.check(name, False, str(error)[:800])


def load_cache(spec):
    """直接从冻结缓存重建时刻与原始标签，不读取模型选择分数。"""
    basins, cfg = spec["basins"], spec["settings"]
    frames = [pd.read_parquet(ROOT / "data/camelsh_exported" / name) for name in (
        "mswep_precip_3h_86.parquet", "forcing_temperature_mean_3h_86.parquet",
        "forcing_solar_radiation_3h_86.parquet")]
    grid = frames[0].index
    for frame in frames[1:]:
        grid = grid.intersection(frame.index)
    grid = pd.DatetimeIndex(grid).sort_values()
    targets = {}
    for task, name in (("q", "flow_hourly_86.parquet"), ("h", "waterlevel_hourly_86.parquet")):
        frame = pd.read_parquet(ROOT / "data/camelsh_exported" / name).reindex(index=grid, columns=basins)
        targets[task] = np.ascontiguousarray(frame.to_numpy(dtype="float32").T)
    splits = read_json(ROOT / "data/splits/basin_splits.json")["splits"]
    valid = np.r_[True, np.diff(grid.to_numpy()) == np.timedelta64(3, "h")]
    prefix = np.r_[0, np.cumsum(~valid)]
    positions = {}
    for split in ("train", "valid", "test"):
        stride = cfg[split + "_stride"]
        groups = []
        for basin in basins:
            lo = grid.searchsorted(pd.Timestamp(splits[basin][split + "_start"]), "left")
            hi = grid.searchsorted(pd.Timestamp(splits[basin][split + "_end"]), "right") - 1
            pos = np.arange(max(lo, cfg["sequence_length"]), hi + 1, stride)
            pos = pos[valid[pos] & ((prefix[pos] - prefix[pos - cfg["sequence_length"]]) == 0)]
            groups.append(pos)
        positions[split] = groups
    test_pos = np.concatenate(positions["test"])
    ids = np.concatenate([np.full(len(pos), bi, np.int64) for bi, pos in enumerate(positions["test"])])
    bounds = np.r_[0, np.cumsum([len(pos) for pos in positions["test"]])]
    return targets, positions, ids, test_pos, bounds


def label_counts(basins, positions, targets, retained, dual):
    """独立重算标签准入；隐藏站的训练和验证Q全部置为不可用。"""
    total, q_count, h_count = 0, 0, 0
    for bi, basin in enumerate(basins):
        pos = positions[bi]
        q = np.isfinite(targets["q"][bi, pos]) & (basin in retained)
        h = np.isfinite(targets["h"][bi, pos])
        keep = q | h if dual else q
        total += int(keep.sum())
        q_count += int((q & keep).sum())
        h_count += int((h & keep).sum())
    counts = {"flow": q_count}
    if dual:
        counts["waterlevel"] = h_count
    return total, counts


def audit_result(source, output):
    output.mkdir(parents=True, exist_ok=True)
    audit = Audit()
    check = audit.check
    spec, pre = read_json(source / "experiment.json"), read_json(source / "preflight.json")
    cfg, basins, fingerprint = spec["settings"], spec["basins"], spec["experiment_fingerprint"]
    scaling = cfg["target_scaling"]
    observed = scaling == "observed"
    check("原86正式强配置", len(basins) == len(set(basins)) == 86 and not cfg["smoke"]
          and not pre["smoke_not_paper_result"] and cfg["sequence_length"] == 480
          and cfg["attribute_set"] == "extended" and cfg["batch_size"] == 512)
    check("实验指纹", digest({k: v for k, v in spec.items() if k != "experiment_fingerprint"}) == fingerprint)
    check("完成及汇总标记", all(read_json(source / name)["experiment_fingerprint"] == fingerprint
          for name in ("completed_both.json", "summary_manifest.json")))
    check("Q损失与尺度信息边界", not pre["heldout_q_labels_used_in_loss_or_validation_score"]
          and pre["heldout_q_statistics_used_for_scaling"] == observed)
    manifest = read_json(source / "summary_manifest.json")
    check("测试分数不用于选择", manifest["selection_used_test_scores"] is False)

    hashes = []
    for kind in ("source_hashes", "code_hashes"):
        for name, expected in spec[kind].items():
            path = ROOT / name.replace("\\", "/")
            status = "missing"
            if path.is_file():
                status = "exact" if sha(path) == expected else "different"
                if status == "different" and path.suffix in (".py", ".txt", ".json"):
                    value = path.read_bytes().replace(b"\r\n", b"\n")
                    if any(hashlib.sha256(v).hexdigest() == expected
                           for v in (value, value.replace(b"\n", b"\r\n"))):
                        status = "line_endings_only"
            hashes.append({"kind": kind, "file": name, "status": status})
    check("代码与数据来源哈希", all(x["status"] in ("exact", "line_endings_only") for x in hashes))
    save_csv(pd.DataFrame(hashes), output / "source_hash_audit.csv")
    geometry = read_csv(source / "geometry_audit.csv").set_index("basin")
    matches = []
    for _, row in geometry[geometry.available].iterrows():
        candidates = [ROOT / root / row.json_path for root in (
            "data/cross_sections", "research_notes/USGS 原86 历年断面核验")]
        path = next((p for p in candidates if p.is_file()), None)
        matches.append(path is not None and sha(path) == row.source_sha256)
    check("81站断面逐文件来源", len(matches) == 81 and all(matches)
          and geometry.available.sum() == pre["n_geometry_available"] == 81)

    reference = read_json(source / "reference_model_audit.json")
    check("历史参考与批量", reference == spec["reference_model_audit"]
          and reference["original_batch_sizes"] == {"training": 512, "validation_and_inference": 4096})
    for ref in reference["models"]:
        folder = ROOT / ref["folder"].replace("\\", "/")
        check("原参考文件哈希/" + folder.name, all(sha(folder / filename) == ref[key] for filename, key in (
            ("model.pt", "source_model_sha256"), ("run.json", "source_run_sha256"),
            ("metrics.csv", "source_metrics_sha256"))))
    frozen_path = ROOT / cfg["reference_holdouts"]
    frozen = read_json(frozen_path)
    check("冻结原留出清单来源", frozen["basins"] == basins
          and sha(frozen_path) == reference["holdout_audit"]["sha256"])
    for tag, scene in frozen["scenarios"].items():
        check("历史留出文件/" + tag, all(sha(ROOT / item["run_json"]) == item["sha256"]
              and read_json(ROOT / item["run_json"])["mask_stats"]["flow"]["held_out_basins"] == scene["heldout"]
              for item in scene["source_runs"]))

    targets, positions, ids, test_pos, bounds = load_cache(spec)
    pre_by_tag = {s["tag"]: s for s in pre["scenarios"]}
    frames, common_rows, training_rows, converter_rows, errors = [], [], [], [], []
    counts = {"waterlevel": 0, "stage1": 0, "stage2": 0}
    for seed in cfg["model_seeds"]:
        folder = source / "waterlevel_models" / f"seed{seed}"
        meta = read_json(folder / "training_done.json")
        blob = torch.load(folder / "model.pt", map_location="cpu", weights_only=False)
        ref_folder = ROOT / cfg["reference_run_templates"]["single_waterlevel"].format(seed=seed)
        original = torch.load(ref_folder / "model.pt", map_location="cpu", weights_only=False)
        state, old_state = blob["state_dict"], original["state_dict"]
        check(f"H/seed{seed}/原单H权重逐张量一致", state.keys() == old_state.keys()
              and all(torch.equal(state[k], old_state[k]) for k in state)
              and blob["model_config"] == original["model_config"])
        expected = digest({"experiment": fingerprint, "stage": "waterlevel_only", "seed": seed})
        check(f"H/seed{seed}/无微调且无Q尺度", meta["training_performed_this_run"] is False
              and meta["run_fingerprint"] == blob["run_fingerprint"] == expected
              and meta["scaling_info"]["flow_information_used"] is False
              and blob["model_config"]["attr_size"] == 27
              and set(blob["target_mean"]) == {"waterlevel"})
        with np.load(folder / "prediction.npz") as z:
            check(f"H/seed{seed}/完整推理时刻", np.array_equal(z["basin_idx"], ids)
                  and np.array_equal(z["target_pos"], test_pos) and np.isfinite(z["pred_h_normalized"]).all())
        counts["waterlevel"] += 1

    for scenario in spec["scenarios"]:
        held, retained = set(scenario["heldout"]), set(scenario["retained"])
        tag, ratio, mask_seed = scenario["tag"], scenario["ratio"], scenario["mask_seed"]
        scene_audit = read_json(source / "scenario_audits" / f"{tag}.json")
        pre_scene = pre_by_tag[tag]
        check(tag + "/名单和尺度隔离", not held & retained and held | retained == set(basins)
              and scenario["heldout"] == frozen["scenarios"][tag]["heldout"]
              and set(scene_audit["geometry_scaler"]["fit_basins"]) == retained
              and set(pre_scene["physical_scale_fit_basins"]) == (set() if observed else retained)
              and set(pre_scene["observed_q_scale_basins"]) == (set(basins) if observed else set()))
        expected_counts = {(split, dual): label_counts(basins, positions[split], targets, retained, dual)
                           for split in ("train", "valid") for dual in (False, True)}
        check(tag + "/独立重建预检标签数", all(pre_scene[split + "_label_counts"] == expected_counts[(split, True)][1]
              for split in ("train", "valid")))
        for seed in cfg["model_seeds"]:
            for stage, variants in VARIANTS.items():
                parameter_counts = {}
                for variant in variants:
                    folder = source / stage / tag / f"seed{seed}" / variant
                    label = f"{stage}/{tag}/seed{seed}/{variant}"
                    meta = read_json(folder / "done.json")
                    run_id = digest({"experiment": fingerprint, "scenario": tag, "stage": stage, "variant": variant, "seed": seed})
                    blob = torch.load(folder / "model.pt", map_location="cpu", weights_only=False)
                    check(label + "/指纹", meta["run_fingerprint"] == blob["run_fingerprint"] == run_id
                          and set(meta["heldout"]) == held and not meta["smoke_not_paper_result"])
                    if stage == "stage1":
                        check(label + "/转换器拟合信息隔离", set(meta["training_q_pair_basins"]) == retained
                              and set(meta["scale_fit_basins"]) == (set() if observed else retained)
                              and set(meta["observed_q_scale_basins"]) == (set(basins) if observed else set())
                              and not set(meta["inner_train_basins"]) & set(meta["inner_validation_basins"])
                              and set(meta["inner_train_basins"]) | set(meta["inner_validation_basins"]) == retained
                              and meta["selected_by"] == "retained_basins_internal_whole_basin_validation")
                        parameter_counts[variant] = meta["n_parameters"]
                        converter_rows.append({"target_scaling": scaling, "ratio": ratio, "mask_seed": mask_seed,
                            "model_seed": seed, "variant": variant, **{k: meta[k] for k in (
                                "input_dim", "hidden", "n_parameters", "epochs_refit", "n_train_pairs")}})
                    else:
                        train = read_json(folder / "training_done.json")
                        train_cfg, model_cfg = blob["training_config"], blob["model_config"]
                        dual = variant.startswith("dual_head")
                        original_arch = "dual_head" if dual else "single_flow"
                        ref_folder = ROOT / cfg["reference_run_templates"][original_arch].format(seed=seed)
                        old_cfg = read_json(ref_folder / "run.json")["config"]
                        check(label + "/原训练设置", all(v == old_cfg.get(k) for k, v in train_cfg.items() if k != "device")
                              and train_cfg["batch_size"] == 512 and train_cfg["max_epochs"] == 30
                              and model_cfg["attr_size"] == (27 if variant == original_arch else 34))
                        check(label + "/独立重建损失标签数", all(
                            train[f"n_{split}_samples"] == expected_counts[(split, dual)][0]
                            and train[split + "_label_counts"] == expected_counts[(split, dual)][1]
                            for split in ("train", "valid")))
                        info = train["scaling_info"]
                        scale_ok = info["mode"] == scaling
                        if not observed:
                            scale_ok &= all(info["flow"][x]["n_fit"] == len(retained) for x in ("mean", "std"))
                        check(label + "/Q尺度", scale_ok and blob["scaling_info"] == info)
                        parameter_counts[variant] = model_cfg["n_parameters"]
                        training_rows.append({"target_scaling": scaling, "variant": variant, "ratio": ratio,
                            "mask_seed": mask_seed, "model_seed": seed, **{k: train[k] for k in (
                                "best_epoch", "epochs_run", "elapsed_sec", "n_train_samples", "n_valid_samples")}})
                    frame = read_csv(folder / "metrics.csv")
                    frames.append(frame.assign(run_fingerprint=run_id))
                    diag = read_csv(folder / "diagnostic_metrics.csv").set_index("basin") if stage == "stage1" else None
                    if diag is not None:
                        frames.append(diag.reset_index().assign(run_fingerprint=run_id))
                    indexed = frame.set_index("basin")
                    with np.load(folder / "prediction.npz") as z:
                        aligned = np.array_equal(z["basin_idx"], ids) and np.array_equal(z["target_pos"], test_pos)
                        predictions = z["pred_flow"]
                    check(label + "/完整测试时刻", aligned)
                    if not aligned:
                        continue
                    correct = bool(np.isfinite(predictions).all())
                    for bi, basin in enumerate(basins):
                        sl = slice(bounds[bi], bounds[bi + 1])
                        q, h, pred = targets["q"][bi, test_pos[sl]], targets["h"][bi, test_pos[sl]], predictions[sl]
                        ok = np.isfinite(q) & np.isfinite(pred)
                        score, rmse = metric(q[ok], pred[ok]), float(np.sqrt(((q[ok] - pred[ok]) ** 2).mean()))
                        row = indexed.loc[basin]
                        correct &= int(row.n) == int(ok.sum()) and int(row.n_negative_predictions) == int((pred < 0).sum())
                        correct &= bool(np.isclose(row.nse, score, rtol=2e-5, atol=2e-6)
                                        and np.isclose(row.rmse, rmse, rtol=2e-5, atol=2e-6))
                        errors.append(abs(score - row.nse))
                        if stage == "stage1":
                            common = ok & np.isfinite(h)
                            correct &= int(diag.loc[basin, "n"]) == int(common.sum())
                            common_rows.append({"target_scaling": scaling, "variant": variant, "ratio": ratio,
                                "mask_seed": mask_seed, "model_seed": seed, "basin": basin, "held_out": basin in held,
                                "n": int(common.sum()), "predicted_h_nse": metric(q[common], pred[common]),
                                "observed_h_nse": diag.loc[basin, "nse"], "dropped_q_times": int(ok.sum() - common.sum())})
                    check(label + "/从原预测重算NSE及RMSE", correct)
                    counts[stage] += 1
                pairs = [("h_attrs_geometry", "h_attrs_geometry_shuffled")] if stage == "stage1" else [
                    (x + "_geometry", x + "_shuffled") for x in ("single_flow", "dual_head")]
                check(f"{stage}/{tag}/seed{seed}/真假断面参数匹配", all(parameter_counts[a] == parameter_counts[b] for a, b in pairs))
        print(f"{scaling} 已核验 {tag}：标签、权重来源、原始预测及评价时刻", flush=True)

    check("正式运行数量", counts == {"waterlevel": 3, "stage1": 108, "stage2": 162})
    scores = read_csv(source / "all_basin_metrics.csv")
    audit.frames("总表与全部逐次指标一致", pd.concat(frames, ignore_index=True), scores, GROUPS + ["basin"])
    check("总表无重复且每组86站", not scores.duplicated(GROUPS + ["basin"]).any()
          and scores.groupby(GROUPS).size().eq(86).all())
    summaries = []
    for key, group in scores.groupby(GROUPS):
        for scope, selected in {"all_86": group, "q_heldout": group[group.held_out],
                "q_retained": group[~group.held_out], "q_heldout_with_geometry": group[group.held_out & group.geometry_available]}.items():
            row = {**dict(zip(GROUPS, key)), "scope": scope, "n_basins_reported": len(selected),
                   "n_valid_nse": selected.nse.count(), "nse_mean": selected.nse.mean(),
                   "nse_median": selected.nse.median(), "n_nse_negative": int((selected.nse < 0).sum())}
            row.update({m + "_mean": selected[m].mean() for m in ("kge", "pbias", "pearson_r", "peak_time_mae_h")})
            summaries.append(row)
    audit.frames("运行汇总重算一致", pd.DataFrame(summaries), read_csv(source / "summary_by_run.csv"), GROUPS + ["scope"])
    pairs = make_pairs(scores, CONTRASTS)
    audit.frames("配对差值重算一致", pairs, read_csv(source / "paired_differences.csv"), ["comparison"] + KEYS)
    with_g = pairs[pairs.comparison == "stage2_h_gain_with_geometry"]
    without_g = pairs[pairs.comparison == "stage2_h_gain_without_geometry"]
    interaction = with_g[KEYS + ["delta_nse"]].merge(without_g[KEYS + ["delta_nse"]], on=KEYS, suffixes=("_with", "_without"))
    interaction["h_geometry_interaction"] = interaction.delta_nse_with - interaction.delta_nse_without
    saved = read_csv(source / "h_geometry_interaction.csv")
    audit.frames("H与断面交互重算一致", interaction, saved[KEYS + ["h_geometry_interaction"]], KEYS)
    report = {"results": str(source), "target_scaling": scaling, "experiment_fingerprint": fingerprint,
              "run_counts": counts, "n_metric_rows": len(scores), "n_checks": len(audit.checks),
              "n_failed": sum(not c["passed"] for c in audit.checks),
              "max_absolute_recomputed_nse_error": float(max(errors)), "checks": audit.checks}
    (output / "integrity_audit.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    save_csv(pd.DataFrame(common_rows), output / "common_time_h_diagnostic.csv")
    save_csv(pd.DataFrame(training_rows), output / "training_audit.csv")
    save_csv(pd.DataFrame(converter_rows), output / "converter_audit.csv")
    return scores, pd.DataFrame(common_rows), pd.DataFrame(training_rows), report


def make_pairs(scores, contrasts):
    deployed = scores[scores.h_input_source != "observed_h_diagnostic"]
    pairs = []
    for name, (model, control) in contrasts.items():
        left = deployed[deployed.variant == model][KEYS + ["nse", "held_out", "geometry_available"]]
        right = deployed[deployed.variant == control][KEYS + ["nse"]]
        pair = left.merge(right, on=KEYS, suffixes=("_model", "_control"), validate="one_to_one")
        pairs.append(pair.assign(delta_nse=pair.nse_model - pair.nse_control, comparison=name))
    return pd.concat(pairs, ignore_index=True)


def analyze(scores, common, training, output):
    deployed = scores[scores.h_input_source != "observed_h_diagnostic"]
    held = deployed[deployed.held_out]
    run = held.groupby(GROUPS).agg(nse_mean=("nse", "mean"), nse_median=("nse", "median"),
        kge_mean=("kge", "mean"), negative_nse=("nse", lambda v: (v < 0).sum())).reset_index()
    perf = run.groupby(["target_scaling", "stage", "variant", "ratio"]).agg(
        nse_mean=("nse_mean", "mean"), nse_median_run_average=("nse_median", "mean"),
        kge_mean=("kge_mean", "mean"), run_sd=("nse_mean", "std"),
        negative_basins_run_average=("negative_nse", "mean")).reset_index()
    save_csv(perf, output / "heldout_performance.csv")
    pairs = make_pairs(scores, {**CONTRASTS, **EXTRA_CONTRASTS})
    pair_held = pairs[pairs.held_out]
    group = ["target_scaling", "comparison", "ratio"]
    pair_runs = pair_held.groupby(group + ["mask_seed", "model_seed"]).delta_nse.mean().reset_index()
    effects = pair_runs.groupby(group).delta_nse.agg(mean="mean", sd="std", min="min", max="max",
        positive_runs=lambda v: (v > 0).sum(), n_runs="size").reset_index()
    station = pair_held.groupby(group + ["basin"]).delta_nse.mean().reset_index()
    station_summary = station.groupby(group).delta_nse.agg(n_basins="size", mean="mean", median="median",
        positive_fraction=lambda v: (v > 0).mean()).reset_index()
    save_csv(effects, output / "paired_run_effects.csv")
    save_csv(pair_runs, output / "paired_run_values.csv")
    save_csv(station, output / "paired_station_effects.csv")
    save_csv(station_summary, output / "paired_station_summary.csv")
    available = pair_held[pair_held.geometry_available].groupby(group + ["mask_seed", "model_seed"]).delta_nse.mean().reset_index()
    save_csv(available.groupby(group).delta_nse.agg(["mean", "std"]).reset_index(), output / "geometry_available_effects.csv")
    common = common.assign(observed_minus_predicted=common.observed_h_nse - common.predicted_h_nse)
    diag = common[common.held_out].groupby(["target_scaling", "variant", "ratio", "mask_seed", "model_seed"])[
        ["predicted_h_nse", "observed_h_nse", "observed_minus_predicted"]].mean().reset_index()
    diag_summary = diag.groupby(["target_scaling", "variant", "ratio"])[
        ["predicted_h_nse", "observed_h_nse", "observed_minus_predicted"]].mean().reset_index()
    save_csv(diag_summary, output / "common_time_h_summary.csv")
    failure = held.groupby(["target_scaling", "variant", "ratio"]).agg(
        basin_run_count=("basin", "size"), negative_nse_count=("nse", lambda v: (v < 0).sum()),
        minimum_nse=("nse", "min"), negative_prediction_count=("n_negative_predictions", "sum")).reset_index()
    save_csv(failure, output / "failure_summary.csv")
    worst = held.sort_values("nse").groupby(["target_scaling", "variant", "ratio"], group_keys=False).head(3)
    save_csv(worst, output / "worst_cases.csv")
    # 逐站先跨所有已留出的种子平均，避免把重复站点当成独立样本。
    station_scores = held.groupby(["target_scaling", "stage", "variant", "ratio", "basin"]).nse.mean().reset_index()
    station_perf = station_scores.groupby(["target_scaling", "stage", "variant", "ratio"]).nse.agg(
        n_basins="size", mean="mean", median="median", negative_basins=lambda v: (v < 0).sum(), minimum="min").reset_index()
    save_csv(station_perf, output / "station_first_performance.csv")
    shared = ["stage", "variant", "ratio", "mask_seed", "model_seed", "basin"]
    scale = held[held.target_scaling == "observed"][shared + ["nse", "n", "obs_mean", "obs_std"]].merge(
        held[held.target_scaling == "physical"][shared + ["nse", "n", "obs_mean", "obs_std"]],
        on=shared, suffixes=("_observed", "_physical"), validate="one_to_one")
    scale["observed_minus_physical"] = scale.nse_observed - scale.nse_physical
    save_csv(scale, output / "normalization_paired_station_runs.csv")
    scale_run = scale.groupby(["stage", "variant", "ratio", "mask_seed", "model_seed"]).observed_minus_physical.mean().reset_index()
    save_csv(scale_run.groupby(["stage", "variant", "ratio"]).observed_minus_physical.agg(
        mean="mean", median="median", positive_runs=lambda v: (v > 0).sum()).reset_index(), output / "normalization_effects.csv")
    return perf, effects, station_summary, diag_summary, failure, station_perf


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--physical", type=Path, default=ROOT / "results/cross_section86_strong_physical_snapshot_v2")
    parser.add_argument("--observed", type=Path, default=ROOT / "results/cross_section86_strong_observed_snapshot_v2")
    parser.add_argument("--out", type=Path, default=ROOT / "reports/cross_section86_strong_v2_review")
    args = parser.parse_args()
    output = args.out.resolve()
    sources = [args.physical.resolve(), args.observed.resolve()]
    if any(output == source or source in output.parents for source in sources):
        raise ValueError("分析输出须位于原始结果目录之外")
    output.mkdir(parents=True, exist_ok=True)
    data = [audit_result(source, output / name) for source, name in zip(sources, ("physical", "observed"))]
    cross = Audit()
    for seed in (1, 2, 3):
        folders = [source / "waterlevel_models" / f"seed{seed}" for source in sources]
        with np.load(folders[0] / "prediction.npz") as a, np.load(folders[1] / "prediction.npz") as b:
            cross.check(f"两种归一化/seed{seed}/H预测完全一致", set(a.files) == set(b.files)
                        and all(np.array_equal(a[k], b[k], equal_nan=True) for k in a.files))
    scores = pd.concat([x[0] for x in data], ignore_index=True)
    keys = [k for k in GROUPS if k != "target_scaling"] + ["basin"]
    a, b = [scores[scores.target_scaling == name] for name in ("physical", "observed")]
    cross.frames("两种归一化评价样本与观测一致", a, b[keys + ["n", "obs_mean", "obs_std", "held_out", "geometry_available"]], keys)
    common = pd.concat([x[1] for x in data], ignore_index=True)
    training = pd.concat([x[2] for x in data], ignore_index=True)
    perf, effects, station, diag, failure, station_perf = analyze(scores, common, training, output)
    checks = [c for x in data for c in x[3]["checks"]] + cross.checks
    integrity = {"n_checks": len(checks), "n_failed": sum(not c["passed"] for c in checks),
        "n_metric_rows": len(scores), "n_prediction_files_recomputed": sum(x[3]["run_counts"][s] for x in data for s in ("stage1", "stage2")),
        "max_absolute_recomputed_nse_error": max(x[3]["max_absolute_recomputed_nse_error"] for x in data),
        "cross_scaling_checks": cross.checks, "failed_checks": [c for c in checks if not c["passed"]]}
    (output / "integrity_summary.json").write_text(json.dumps(integrity, ensure_ascii=False, indent=2), encoding="utf-8")
    selected = perf[perf.variant.isin(["h_only", "h_attrs", "h_attrs_geometry", "single_flow", "dual_head", "dual_head_geometry"])]
    train_summary = training.groupby(["target_scaling", "variant"]).agg(
        runs=("best_epoch", "size"), best_epoch_median=("best_epoch", "median"),
        best_at_cap=("best_epoch", lambda v: (v == 30).sum())).reset_index()
    save_csv(train_summary, output / "training_summary.csv")
    sections = ["# 原86强配置两阶段断面实验：回拷核验", "",
        "仅分析已完成的两套结果；没有启动训练或推理，没有重新选择模型，也没有改写原始结果。", "",
        f"核验 {integrity['n_checks']} 项，失败 {integrity['n_failed']} 项；独立重算 {integrity['n_prediction_files_recomputed']} 个预测文件的逐站NSE/RMSE，最大NSE绝对误差 {integrity['max_absolute_recomputed_nse_error']:.3g}。",
        "每套含3个复用单H模型、108个转换器、162个直接网络；两套合计65016行指标。H权重与各自原4J/4K单H模型逐张量相同，且两套的H预测完全一致。完整检查状态以integrity_summary.json及两套integrity_audit.json为准。", "",
        "## 结论", "",
        "当前两套结果支持水位辅助监督的稳定平均增益，尚不支持这6个相对断面特征具有稳定的额外增益。所有结论限于本次协议、模型与几何表示。", "",
        "阶段1应保留简单H-only转换器作强基线：两种归一化、三个留出比例下，其平均NSE均高于加入属性或属性加断面的转换器。不能只拿表现较弱的H+属性转换器，宣称断面有效。", "",
        "阶段2的真实断面与打乱断面参数量相同；真实断面没有跨比例稳定优于打乱对照，因此不能将少量分数改善解释为已识别的水力学信息。", "",
        "## 主结果与聚合口径", "",
        "仅评价训练与验证Q都隐藏的站。30%、50%、70%每次分别隐藏26、43、60站；每比例为3个模型种子×3个掩膜种子。先每次计算站平均，再平均9次。nse_median_run_average为各次站中位数的平均，非总体中位数。", "",
        table(selected[["target_scaling", "stage", "variant", "ratio", "nse_mean", "nse_median_run_average", "negative_basins_run_average"]]), "",
        "physical的Q尺度只由保留站拟合；observed使用全部86站训练期Q均值和标准差，包括隐藏Q站，是已知Q尺度对照，不能作为目标站Q信息完全未知的结果。", "",
        "## 配对效应", "",
        "差值均为模型减对照；geometry_converter_vs_geometry_dual的方向为双头加断面减转换器加断面。positive_runs是9次运行平均差大于0的次数，运行并非独立流域样本，sd不构成置信区间。", "", table(effects), "",
        "逐站先跨种子平均后的中位差和胜率如下；不同站被抽到的次数可能不同，所以此表与九次运行等权汇总有不同权重。没有用该表筛掉差站。", "", table(station), "",
        "## 水位误差诊断", "",
        "同一冻结转换器在Q/H共同有效的相同时刻，分别输入预测H和实测H。预测H指标从原预测重新计算；实测H指标来自已保存诊断输出。实测H测试输入不是纯气象预测路径，不能用于部署能力比较或模型选择。", "", table(diag), "",
        "阶段1用训练实测H拟合、推理用预测H，存在输入误差分布变化；并未实现折外预测H拟合。输入实测H也不能预先保证断面有效，应逐比例查看本表。", "",
        "## 稳健性与失败站", "",
        "NSE可有较大的负值；平均增益不能替代中位数与单站分布。下表保留所有负值；最差流域及运行见worst_cases.csv。negative_prediction_count包含全部推理时刻，没有事后截断。", "", table(failure), "",
        "geometry_available_effects.csv提供仅有断面站的敏感性分析；station_first_performance.csv先逐站跨种子聚合，再跨站汇总。normalization_effects.csv给出两种尺度的配对差，但差值不能解释为新增断面贡献。", "",
        "## 与原模型一致的部分及研究边界", "",
        "- H直接复用原60天窗口、extended属性的单任务水位LSTM，不微调、不输入断面；阶段2使用同架构同超参数重新训练。训练batch512，验证和推理batch4096。",
        "- 新训练与验证Q都隐藏；原4J/4K留出实验只隐藏训练Q。新训练/推理排除跨时间缺口窗口；复用H保留原训练窗口。阶段1预测不要求测试H有观测，因此不能直接把新旧绝对分数相减归因于断面。",
        "- 81站有合格相对剖面，79站测于训练结束后，是2021—2023断面快照提供的额外事后几何信息；并非完整绝对床高或已识别的A(H)。",
        "- 单Q与双头因标签并集准入而训练样本量不同，不能声称两者计算预算相同。断面真假对照匹配输入维数和参数量；加断面与不加断面参数量并不相同。",
        "- 冻结时间边界继承原始Q/H可用性。physical移除了目标站Q尺度使用，但不能称站点从未有过Q记录。",
        "- 这是一次预定方案的测试集分析，不据测试分数延长训练或调整宽度。若继续改方法，应重新预定义验证规则与独立验证，不能反复沿用这次测试集选择方案。", "",
        "## 训练与复现", "", table(train_summary), "",
        "下列命令只读结果并生成分析材料，不训练：", "", "```powershell",
        "python tools/review_cross_section86_strong_results.py", "```", "",
        "本次独立重建了冻结时刻、训练/验证样本准入与Q/H标签数；核对源代码哈希、尺度拟合站清单、旧模型来源和权重。没有重新执行正式训练；候选阶段更细的临时归一化过程不能单靠完成标记完全重放。"]
    (output / "review.md").write_text("\n".join(sections) + "\n", encoding="utf-8")
    print(json.dumps(integrity, ensure_ascii=False, indent=2))
    print(selected[["target_scaling", "variant", "ratio", "nse_mean"]].to_string(index=False))
    return int(integrity["n_failed"] > 0)


if __name__ == "__main__":
    raise SystemExit(main())
