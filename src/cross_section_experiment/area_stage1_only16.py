"""仅16个面积站参与阶段1转换器拟合和评价；原86水位模型保持冻结。"""

from __future__ import annotations

import argparse
import math
import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from . import area_stage1 as original
from .converters import PairData, predict_converter
from .features import file_sha256
from .protocol import fingerprint, write_json
from .runner import save_prediction
from evaluation.metrics import basin_metrics
from pipeline.dataset import PreparedData


ROOT = original.ROOT
AREA_BASINS, FOCUS_BASINS = original.AREA_BASINS, original.FOCUS_BASINS
VARIANTS = original.VARIANTS
PROTOCOL = "atgage16_only_stage1_train_valid_q_hidden_v2"
DEFAULT_OUTPUT = ROOT / "results/atgage16_area_stage1_only16_v2"
OLD_OUTPUT = ROOT / "results/atgage16_area_stage1_v1"
load_json = original.load_json


def restrict_scenarios(outer_scenarios):
    """与原冻结名单求交，不重新抽样；标签比例仍指原86的名义隐藏比例。"""
    result = []
    allowed = set(AREA_BASINS)
    for outer in outer_scenarios:
        retained = [b for b in outer["retained"] if b in allowed]
        heldout = [b for b in outer["heldout"] if b in allowed]
        if set(retained).intersection(heldout) or set(retained + heldout) != allowed:
            raise ValueError("16站交集与冻结Q名单不一致")
        if len(retained) < 3:
            raise ValueError("16站内至少需要3个Q保留站，才能保留整站内部验证")
        result.append({"tag": outer["tag"], "ratio": outer["ratio"], "mask_seed": outer["mask_seed"],
                       "retained": retained, "heldout": heldout,
                       "actual_q_hidden_ratio16": len(heldout) / 16,
                       "original86_retained": outer["retained"], "original86_heldout": outer["heldout"]})
    return result


class RestrictedQPrepared:
    """保留原86索引及H缓存，阻止Q尺度拟合越过当前16站内的Q保留名单。"""

    def __init__(self, prepared, retained):
        if not set(retained).issubset(AREA_BASINS):
            raise ValueError("转换器拟合名单含16站以外流域")
        self.base = prepared
        self.allowed = set(retained)
        self.scaling_calls = []

    def __getattr__(self, name):
        return getattr(self.base, name)

    def make_scaling(self, scaling, fit_basins=None):
        if not fit_basins or not set(fit_basins).issubset(self.allowed):
            raise ValueError("Q尺度拟合包含隐藏站或16站以外流域")
        self.scaling_calls.append({"target_scaling": scaling, "fit_basins": list(fit_basins)})
        return self.base.make_scaling(scaling, fit_basins=fit_basins)


def fit_converter(prepared, payload, scene, variant, seed, scaling, cfg, forced_width=None):
    """复用旧转换器结构和超参；所有训练标签、内部选模和尺度拟合均限定在新交集。"""
    retained, heldout = set(scene["retained"]), set(scene["heldout"])
    if (retained | heldout != set(AREA_BASINS) or retained.intersection(heldout)
            or len(retained) != len(scene["retained"]) or len(heldout) != len(scene["heldout"])):
        raise ValueError("转换器情景不是完整16站总体")
    guarded = RestrictedQPrepared(prepared, scene["retained"])
    model, scale, donors, audit = original.fit_converter(
        guarded, payload, scene, variant, seed, scaling, cfg, forced_width)
    if audit["training_q_pair_basins"] != scene["retained"]:
        raise ValueError("实际配对训练名单与16站交集不同")
    if any(not set(audit[key]).issubset(scene["retained"]) for key in
           ("inner_train_basins", "inner_validation_basins")):
        raise ValueError("内部选模使用了交集之外的Q标签")
    audit.update(converter_population=list(AREA_BASINS), waterlevel_base_population_size=86,
                 q_scaling_fit_calls=guarded.scaling_calls,
                 original86_q_retained=scene["original86_retained"],
                 original86_q_heldout=scene["original86_heldout"],
                 actual_q_hidden_ratio16=scene["actual_q_hidden_ratio16"],
                 observed_q_scale_basins=list(AREA_BASINS) if scaling == "observed" else [],
                 observed_q_statistics_outside16_used_for_fitting=False,
                 selected_by="retained_within16_internal_whole_basin_validation")
    return model, scale, donors, audit


def filter_prediction(prepared, prediction):
    allowed = np.array([prepared.basin_index[b] for b in AREA_BASINS])
    take = np.isin(prediction["basin_idx"], allowed)
    result = {key: prediction[key][take].copy() for key in ("basin_idx", "target_pos", "pred_h_normalized")}
    if set(np.unique(result["basin_idx"])) != set(allowed):
        raise ValueError("冻结H预测没有覆盖全部16站")
    return result


def metrics16(prepared, payload, ids, pos, values, scene, seed, variant, scaling, h_source):
    """仅访问16站测试Q；保持旧指标定义，不把其余70站带入分母。"""
    rows = []
    for basin in AREA_BASINS:
        bi = prepared.basin_index[basin]
        selected = ids == bi
        positions, pred = pos[selected], values[selected]
        obs = prepared.targets_raw["flow"][bi, positions]
        ok = np.isfinite(obs) & np.isfinite(pred)
        score = {"n": int(ok.sum()), "nse": float("nan")}
        if ok.sum() >= 2:
            full = np.arange(positions.min(), positions.max() + 1)
            obs_series, pred_series = np.full(len(full), np.nan), np.full(len(full), np.nan)
            obs_series[positions - full[0]], pred_series[positions - full[0]] = obs, pred
            score = basin_metrics(obs[ok], pred[ok], obs_series, pred_series)
        curve = payload["curves"][basin]
        rows.append({"basin": basin, "stage": "stage1_area_only16", "variant": variant,
                     "ratio": scene["ratio"], "mask_seed": scene["mask_seed"], "model_seed": seed,
                     "actual_q_hidden_ratio16": scene["actual_q_hidden_ratio16"],
                     "held_out": basin in scene["heldout"], "primary_evaluation": basin in scene["heldout"],
                     "geometry_available": True, "geometry_mode": "train_atgage_area_curve",
                     "h_input_source": h_source, "target_scaling": scaling, "smoke_not_paper_result": False,
                     "n_negative_predictions": int((pred[np.isfinite(pred)] < 0).sum()),
                     "area_analysis_group": "focus5" if basin in FOCUS_BASINS else "qualified11",
                     "geometry_lovo_r2": curve["lovo_r2"], "geometry_time_r2": curve["time_r2"],
                     "geometry_lovo_relative_mae": curve["lovo_relative_mae"], **score})
    return pd.DataFrame(rows)


def evaluate(prepared, payload, scene, seed, variant, cfg, model, scale, donors, prediction, folder, audit, run_id):
    ids, pos = prediction["basin_idx"], prediction["target_pos"]
    if not set(np.unique(ids)).issubset({prepared.basin_index[b] for b in AREA_BASINS}):
        raise ValueError("本轮预测/评价含16站外流域")
    for source in ("predicted_h", "observed_h_diagnostic"):
        h = prediction["pred_h_normalized"] if source == "predicted_h" else prepared.targets["waterlevel"][ids, pos]
        valid = np.isfinite(h)
        pairs = PairData(h[valid, None], np.zeros(valid.sum()), ids[valid], np.ones(valid.sum()), pos[valid])
        x = original.input_matrix(prepared, payload, pairs, variant, audit["final_area_scaler"], donors)
        normalized = predict_converter(model, x, cfg["device"])
        values = np.full(len(ids), np.nan)
        values[valid] = normalized * scale.std["flow"][ids[valid]] + scale.mean["flow"][ids[valid]]
        scores = metrics16(prepared, payload, ids, pos, values, scene, seed, variant, cfg["target_scaling"], source)
        _, _, outside, negative = original.area_values(prepared, payload, ids[valid], h[valid], np.arange(len(prepared.basins)))
        _, _, feature_outside, feature_negative = original.area_values(prepared, payload, ids[valid], h[valid], donors)
        for basin in AREA_BASINS:
            selected = ids[valid] == prepared.basin_index[basin]
            for name, flag in (("area_h_outside_fraction", outside), ("area_negative_clipped_fraction", negative),
                               ("feature_area_h_outside_fraction", feature_outside),
                               ("feature_area_negative_clipped_fraction", feature_negative)):
                scores.loc[scores.basin.eq(basin), name] = float(flag[selected].mean()) if selected.any() else np.nan
        scores.to_csv(folder / ("metrics.csv" if source == "predicted_h" else "diagnostic_metrics.csv"), index=False, encoding="utf-8-sig")
        if source == "predicted_h":
            if not np.isfinite(values).all():
                raise ValueError("正式Q预测含非有限值")
            save_prediction(folder, ids, pos, values)
    subset = [prepared.basin_index[b] for b in AREA_BASINS]
    torch.save({"protocol": PROTOCOL, "state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()},
                "input_dim": audit["input_dim"], "hidden": audit["hidden"], "variant": variant,
                "flow_scale_basins": list(AREA_BASINS), "flow_mean": scale.mean["flow"][subset],
                "flow_std": scale.std["flow"][subset], "flow_scale_info": scale.info,
                "original86_index_basins": prepared.basins, "training_q_pair_basins": scene["retained"],
                "area_scaler": audit["final_area_scaler"], "area_payload_fingerprint": payload["payload_fingerprint"],
                "area_curves": payload["curves"], "donor_basins": audit["donor_basins"], "run_fingerprint": run_id},
               folder / "model.pt")
    audit.update(protocol=PROTOCOL, run_fingerprint=run_id, heldout=scene["heldout"], ratio=scene["ratio"],
                 mask_seed=scene["mask_seed"], model_seed=seed, smoke_not_paper_result=False,
                 evaluation_basins=list(AREA_BASINS), primary_evaluation_basins=scene["heldout"])
    write_json(folder / "done.json", audit)


def summarize(output, experiment_id):
    frames, fits = [], []
    for done in sorted(output.glob("*/stage1/*/*/*/done.json")):
        meta = load_json(done)
        if meta["protocol"] != PROTOCOL or set(meta["training_q_pair_basins"]) - set(AREA_BASINS):
            raise ValueError("结果目录混入旧86转换器")
        for filename in ("metrics.csv", "diagnostic_metrics.csv"):
            frame = pd.read_csv(done.parent / filename, dtype={"basin": str})
            if len(frame) != 16 or set(frame.basin) != set(AREA_BASINS):
                raise ValueError("每次结果必须恰好包含16站")
            frames.append(frame)
        fits.append({k: meta.get(k) for k in ("variant", "target_scaling", "ratio", "mask_seed", "model_seed",
                     "hidden", "n_parameters", "epochs_refit", "n_area_train_basins", "elapsed_fit_sec")})
    if not frames:
        return 0
    data = pd.concat(frames, ignore_index=True)
    data.to_csv(output / "all_basin_metrics.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(fits).to_csv(output / "fit_audit.csv", index=False, encoding="utf-8-sig")
    groups = ["target_scaling", "variant", "ratio", "mask_seed", "model_seed", "h_input_source"]
    rows = []
    for keys, frame in data.groupby(groups):
        scopes = {"all16": np.ones(len(frame), bool), "q_heldout16": frame.held_out,
                  "q_retained16": ~frame.held_out, "qualified11": frame.area_analysis_group.eq("qualified11"),
                  "focus5": frame.area_analysis_group.eq("focus5"),
                  "q_heldout_qualified11": frame.held_out & frame.area_analysis_group.eq("qualified11"),
                  "q_heldout_focus5": frame.held_out & frame.area_analysis_group.eq("focus5")}
        for label, mask in scopes.items():
            selected = frame[mask]
            rows.append({**dict(zip(groups, keys)), "scope": label, "n_basins": len(selected),
                         "n_valid_nse": selected.nse.count(), "nse_mean": selected.nse.mean(),
                         "nse_median": selected.nse.median(), "negative_nse_count": int((selected.nse < 0).sum())})
    summary = pd.DataFrame(rows)
    summary.to_csv(output / "summary_by_run.csv", index=False, encoding="utf-8-sig")
    summary[summary.h_input_source.eq("predicted_h")].groupby(["target_scaling", "variant", "ratio", "scope"])[[
        "nse_mean", "nse_median", "negative_nse_count"]].mean().reset_index().to_csv(
        output / "group_summary.csv", index=False, encoding="utf-8-sig")
    join = ["target_scaling", "ratio", "mask_seed", "model_seed", "h_input_source", "basin"]
    baseline = data[data.variant.eq("h_attrs_area_available")][join + ["nse"]].rename(columns={"nse": "baseline_nse"})
    paired = data.merge(baseline, on=join, how="left", validate="many_to_one")
    paired["delta_nse_vs_same_availability_control"] = paired.nse - paired.baseline_nse
    paired.to_csv(output / "paired_differences.csv", index=False, encoding="utf-8-sig")
    write_json(output / "summary_manifest.json", {"experiment_fingerprint": experiment_id, "protocol": PROTOCOL,
               "n_completed": len(fits), "n_expected": 324, "n_converter_basins": 16,
               "training_and_selection_only16": True, "primary_scope": "q_heldout16",
               "all16_included": True, "focus5_retained": list(FOCUS_BASINS),
               "selection_used_test_scores": False, "stage2_executed": False})
    return len(fits)


def preflight(prepared, payload, specs):
    table = pd.DataFrame({"geom_missing": [int(b not in AREA_BASINS) for b in prepared.basins]}, index=prepared.basins)
    verified, full_predictions = original.verify_frozen_sources(prepared, specs, {"physical": table, "observed": table.copy()})
    scenarios = restrict_scenarios(verified["scenarios"])
    counts, inner_splits = [], []
    for scene in scenarios:
        for seed in (1, 2, 3):
            train, valid, audit = original.area_inner_split(scene["retained"], seed)
            assert set(train + valid) == set(scene["retained"]) and not audit["changed"]
            for split, basins in (("train", train), ("valid", valid)):
                for basin in basins:
                    bi = prepared.basin_index[basin]
                    lo, hi = prepared.split_range(basin, split)
                    q, h = prepared.targets_raw["flow"][bi, lo:hi + 1], prepared.targets["waterlevel"][bi, lo:hi + 1]
                    if not (np.isfinite(q) & np.isfinite(h)).any():
                        raise ValueError(f"{scene['tag']} seed{seed} {basin}的{split}无H—Q配对")
            # 检查真正的physical尺度拟合也只使用这组16站内的保留信息；不训练网络。
            for fit_basins in (train, scene["retained"]):
                prepared.make_scaling("physical", fit_basins=fit_basins)
            row = {"tag": scene["tag"], "model_seed": seed, "nominal_q_hidden_ratio86": scene["ratio"],
                   "actual_q_hidden_ratio16": scene["actual_q_hidden_ratio16"], "n_retained16": len(scene["retained"]),
                   "n_hidden16": len(scene["heldout"]), "n_inner_train16": len(train), "n_inner_valid16": len(valid)}
            counts.append(row)
            inner_splits.append({**row, "inner_train": train, "inner_valid": valid})
    records = pd.DataFrame(payload["training_records"])
    for basin in AREA_BASINS:
        when = pd.to_datetime(records.loc[records.basin.eq(basin), "measurement_time"])
        bounds = prepared.splits["splits"][basin]
        if not when.between(pd.Timestamp(bounds["train_start"]), pd.Timestamp(bounds["train_end"])).all():
            raise ValueError("面积曲线含原训练期以外记录")
    predictions = {scaling: {seed: filter_prediction(prepared, pred) for seed, pred in group.items()}
                   for scaling, group in full_predictions.items()}
    code_hashes = dict(verified["code_hashes"])
    for path in (Path(original.__file__), Path(__file__), ROOT / "tools/run_atgage16_area_stage1_only16.py"):
        code_hashes[str(path.relative_to(ROOT))] = file_sha256(path)
    spec = {"protocol": PROTOCOL, "waterlevel_base_basins": prepared.basins, "converter_basins": list(AREA_BASINS),
            "area_basins": list(AREA_BASINS), "focus_basins": list(FOCUS_BASINS),
            "scenarios": scenarios, "inner_splits": inner_splits, "sources": verified["sources"],
            "source_settings": {name: item["settings"] for name, item in specs.items()},
            "data_hashes": verified["data_hashes"], "code_hashes": code_hashes,
            "numerical_environment": verified["numerical_environment"],
            "area_payload_fingerprint": payload["payload_fingerprint"],
            "variants": list(VARIANTS), "scalings": ["physical", "observed"], "n_expected": 324,
            "training_and_selection_only16": True, "physical_scale_fit_only_retained16": True,
            "observed_q_scale_used_basins": list(AREA_BASINS),
            "primary_evaluation": "q_heldout_within16_original_test_period_predicted_h",
            "secondary_evaluation": "all16_and_q_retained16_temporal_test_diagnostic",
            "waterlevel_models_retrained": 0, "stage2_executed": False,
            "outer_split_rule": "intersect_original86_frozen_q_masks_with_confirmed16_no_resampling",
            "inner_split_rule": "same_whole_basin_split_for_all_variants_within_retained16",
            "training_h_source": "observed_training_h", "inference_h_source": "original_frozen_predicted_h",
            "area_feature": "unchanged_log1p_nonnegative_linear_area_inner_train_scaling",
            "area_availability_marker": "constant_one_for_all16_preserved_input_form",
            "converter_hyperparameters_unchanged": True,
            "new_experiment_after_prior_test_inspection": True}
    return spec, predictions, counts


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--curves", type=Path, default=original.CURVE_FILE)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--preflight", action="store_true")
    parser.add_argument("--scaling", choices=("physical", "observed"))
    parser.add_argument("--scenario")
    parser.add_argument("--model-seed", type=int, choices=(1, 2, 3))
    parser.add_argument("--variant", choices=VARIANTS)
    args = parser.parse_args()
    output = args.out_dir.resolve()
    for source in [OLD_OUTPUT, *original.SOURCE_PATHS.values()]:
        if output == source or source in output.parents or output in source.parents:
            raise ValueError("新结果目录不能覆盖或包含旧面积实验和原强配置结果")
    if not args.preflight and args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA不可用，不自动降级或改变批量")
    payload = original.load_curves(args.curves)
    prepared = PreparedData(attr_set="extended")
    specs = {name: load_json(path / "experiment.json") for name, path in original.SOURCE_PATHS.items()}
    spec, predictions, counts = preflight(prepared, payload, specs)
    if args.scenario and args.scenario not in {s["tag"] for s in spec["scenarios"]}:
        raise ValueError("指定情景不在原冻结名单内")
    experiment_id = fingerprint(spec)
    output.mkdir(parents=True, exist_ok=True)
    manifest = output / "experiment.json"
    if manifest.exists():
        previous = load_json(manifest)
        if previous["protocol"] != PROTOCOL:
            raise ValueError("输出目录属于另一套实验")
        if previous["experiment_fingerprint"] != experiment_id:
            if any(output.glob("*/stage1/*/*/*/model.pt")) or any(output.glob("*/stage1/*/*/*/done.json")):
                raise ValueError("输出目录已有不同代码或输入的模型，请换新目录")
            write_json(output / "preflight_history" / (previous["experiment_fingerprint"] + ".json"), previous)
    write_json(manifest, {**spec, "experiment_fingerprint": experiment_id})
    write_json(output / "area_curves.json", payload)
    pd.DataFrame(counts).to_csv(output / "area_training_coverage.csv", index=False, encoding="utf-8-sig")
    write_json(output / "preflight.json", {"status": "PASS", "protocol": PROTOCOL, "n_converter_basins": 16,
               "n_waterlevel_base_basins": 86, "training_and_selection_only16": True,
               "source_h_predictions_verified": 6, "n_planned_converter_fits": 324,
               "min_retained16": min(c["n_retained16"] for c in counts),
               "min_inner_train16": min(c["n_inner_train16"] for c in counts),
               "min_inner_valid16": min(c["n_inner_valid16"] for c in counts),
               "physical_scale_fit_only_retained16": True, "stage2_executed": False, "waterlevel_models_retrained": 0})
    print("预检通过：转换器仅16站、原86水位模型冻结；physical/observed共324项", flush=True)
    if args.preflight:
        return 0
    write_json(output / "process.json", {"pid": os.getpid(), "status": "running", "experiment_fingerprint": experiment_id})
    for scaling in ([args.scaling] if args.scaling else ["physical", "observed"]):
        cfg = {**specs[scaling]["settings"], "device": args.device, "target_scaling": scaling}
        for scene in spec["scenarios"]:
            if args.scenario and args.scenario != scene["tag"]:
                continue
            for seed in ([args.model_seed] if args.model_seed else [1, 2, 3]):
                for variant in ([args.variant] if args.variant else VARIANTS):
                    folder = output / scaling / "stage1" / scene["tag"] / f"seed{seed}" / variant
                    folder.mkdir(parents=True, exist_ok=True)
                    run_id = fingerprint({"experiment": experiment_id, "scaling": scaling,
                                          "scenario": scene["tag"], "seed": seed, "variant": variant})
                    if (folder / "done.json").is_file():
                        meta = load_json(folder / "done.json")
                        if meta["run_fingerprint"] != run_id or meta["protocol"] != PROTOCOL:
                            raise ValueError("已完成项的指纹或协议不一致")
                        if meta["training_q_pair_basins"] != scene["retained"]:
                            raise ValueError("已完成项没有按16站交集训练")
                        if not all((folder / name).is_file() for name in ("model.pt", "prediction.npz", "metrics.csv", "diagnostic_metrics.csv")):
                            raise ValueError("完成标记对应的文件缺失")
                        continue
                    forced, reference = None, None
                    if variant in ("h_attrs_area_shuffled", "h_attrs_capacity_control"):
                        ref_path = folder.parent / "h_attrs_area/done.json"
                        if not ref_path.exists():
                            raise ValueError("需先完成同情景h_attrs_area，才能运行打乱或容量对照")
                        reference = load_json(ref_path)
                        forced = reference["hidden"] if variant.endswith("_shuffled") else math.ceil(
                            (reference["n_parameters"] - 1) / (prepared.attrs.shape[1] + 4))
                    print(f"开始 {scaling} {scene['tag']} seed{seed} {variant}；训练{len(scene['retained'])}站，隐藏{len(scene['heldout'])}站", flush=True)
                    model, scale, donors, audit = fit_converter(prepared, payload, scene, variant, seed, scaling, cfg, forced)
                    if variant == "h_attrs_capacity_control":
                        audit["capacity_reference_parameters"] = reference["n_parameters"]
                        audit["extra_parameters_vs_area"] = audit["n_parameters"] - reference["n_parameters"]
                        if not 0 <= audit["extra_parameters_vs_area"] < audit["input_dim"] + 2:
                            raise ValueError("容量上界对照参数量不合约定")
                    evaluate(prepared, payload, scene, seed, variant, cfg, model, scale, donors,
                             predictions[scaling][seed], folder, audit, run_id)
                    write_json(output / "progress.json", {"last_completed": f"{scaling}/{scene['tag']}/seed{seed}/{variant}",
                               "n_completed": len(list(output.glob("*/stage1/*/*/*/done.json"))), "n_expected": 324})
            summarize(output, experiment_id)
    completed = summarize(output, experiment_id)
    write_json(output / "process.json", {"pid": os.getpid(), "status": "completed" if completed == 324 else "selected_runs_completed",
               "experiment_fingerprint": experiment_id, "n_completed": completed})
    if completed == 324:
        write_json(output / "completed.json", {"experiment_fingerprint": experiment_id, "protocol": PROTOCOL,
                   "n_completed": 324, "n_converter_basins": 16, "training_and_selection_only16": True,
                   "waterlevel_models_retrained": 0, "stage2_executed": False})
    print(f"指定拟合完成：{completed}/324；输出{output}", flush=True)
    return 0
