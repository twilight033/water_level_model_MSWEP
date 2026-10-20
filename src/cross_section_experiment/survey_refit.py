"""测量几何条件化H→Q重拟合；原H权重和测试预测全部复用。"""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import sys
import time

import numpy as np
import pandas as pd
import torch

from .converters import collect_pairs, fit_network, internal_basin_split, predict_converter
from .features import file_sha256
from .inference import InferenceWindows
from .protocol import fingerprint, validate_population, write_json
from .runner import DATA_FILES, flow_metrics, save_prediction
from .survey_geometry import (SCALE_COLUMNS, STAGE_COLUMN, fit_survey_scaler,
                              load_survey_geometry, survey_inputs)
from pipeline.dataset import PreparedData
from pipeline.paths import load_basin_ids


ROOT = Path(__file__).resolve().parents[2]
PROTOCOL = "original86_survey_geometry_refit_train_valid_q_hidden_v1"
VARIANTS = {
    "h_only": None,
    "h_geometry_scale": SCALE_COLUMNS,
    "h_survey_stage": [STAGE_COLUMN],
    "h_survey_geometry": SCALE_COLUMNS + [STAGE_COLUMN],
    "h_survey_geometry_shuffled": SCALE_COLUMNS + [STAGE_COLUMN],
    "h_only_capacity_control": None,
}
SOURCE_PATHS = {name: ROOT / "results" / f"cross_section86_strong_{name}_snapshot_v2"
                for name in ("physical", "observed")}


def load_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def columns_for_variant(variant):
    return VARIANTS[variant]


def survey_stage_alignment(table):
    """就近核验测量站点水位与小时缓存，不把缺测站误报为已完成对齐。"""
    frame = pd.read_parquet(ROOT / "data/camelsh_exported/waterlevel_hourly_86.parquet")
    roots = [ROOT / "data/cross_sections", ROOT / "research_notes/USGS 原86 历年断面核验"]
    records = []
    for basin, row in table[table.geom_missing == 0].iterrows():
        path = next(root / row.source_json for root in roots if (root / row.source_json).is_file())
        payload = load_json(path)
        instant = pd.Timestamp(payload["measurement_timestamp_utc"]).tz_convert(None)
        position = frame.index.get_indexer([instant], method="nearest")[0]
        nearest = frame.index[position]
        value = float(frame[basin].iloc[position])
        time_delta = abs((nearest - instant).total_seconds())
        usable = np.isfinite(value) and time_delta <= 3600
        error = abs(value - row.survey_stage_m) if usable else None
        # NWIS测量平均水位与就近整点水位允许有小差异；明显失配必须先停止核查。
        if usable and error > 0.1:
            raise ValueError(f"测量水位与缓存可能不同基准或单位: {basin}, 差={error:.3f}m")
        records.append({"basin": basin, "survey_utc": instant.isoformat(),
            "nearest_cache_time": nearest.isoformat(), "time_delta_sec": time_delta,
            "survey_stage_m": row.survey_stage_m, "cache_h_m": value,
            "absolute_stage_error_m": error,
            "alignment_status": "nearby_observation_checked" if usable else "nearby_h_missing_not_confirmed",
            "section_location": row.location_code,
            "section_water_surface_mapping_verified": False})
    return pd.DataFrame(records)


def make_static(table, prepared, fit_basins, variant, shuffle_seed):
    columns = columns_for_variant(variant)
    if columns is None:
        return None, None
    scaler = fit_survey_scaler(table, fit_basins, columns)
    values = survey_inputs(table, prepared.basins, scaler,
                           shuffle_seed if variant.endswith("_shuffled") else None)
    return values, scaler


def fit_converter(prepared, table, retained, variant, seed, mask_seed, scaling, cfg, forced_width=None):
    """内部整站验证选模；候选预处理和物理Q尺度都只拟合内部训练站。"""
    start = time.monotonic()
    inner_train, inner_valid = internal_basin_split(retained, seed + 1000)
    static, inner_scaler = make_static(table, prepared, inner_train, variant, mask_seed + 2000)
    inner_scale = prepared.make_scaling(scaling, fit_basins=inner_train)
    maximum = cfg["converter_max_pairs_per_basin"]
    train = collect_pairs(prepared, inner_train, "train", inner_scale, static, maximum)
    valid = collect_pairs(prepared, inner_valid, "valid", inner_scale, static, maximum)
    widths = [forced_width] if forced_width is not None else cfg["converter_widths"]
    candidates = []
    for width in widths:
        print(f"    内部验证 width={width}，训练配对={len(train.y)}，验证配对={len(valid.y)}", flush=True)
        _, info = fit_network(train, int(width), cfg["converter_epochs"], seed, cfg["device"], valid,
                              batch_size=2048, learning_rate=0.001)
        candidates.append({"width": int(width), **info})
        print(f"    width={width} 完成，验证NSE={info['best_score']:.5f}，选中轮次={info['best_epoch']}", flush=True)
    chosen = max(candidates, key=lambda candidate: candidate["best_score"])
    static, final_scaler = make_static(table, prepared, retained, variant, mask_seed + 2000)
    scale = prepared.make_scaling(scaling, fit_basins=retained)
    train = collect_pairs(prepared, retained, "train", scale, static, maximum)
    model, refit = fit_network(train, chosen["width"], chosen["best_epoch"], seed, cfg["device"],
                               batch_size=2048, learning_rate=0.001)
    audit = {"variant": variant, "input_dim": int(train.x.shape[1]), "hidden": chosen["width"],
        "n_parameters": model.n_parameters(), "epochs_refit": chosen["best_epoch"],
        "inner_train_basins": inner_train, "inner_validation_basins": inner_valid,
        "training_q_pair_basins": list(retained), "target_scaling": scaling,
        "scale_fit_basins": list(retained) if scaling == "physical" else [],
        "observed_q_scale_basins": prepared.basins if scaling == "observed" else [],
        "heldout_q_statistics_used_for_scaling": scaling == "observed",
        "inner_geometry_scaler": inner_scaler, "final_geometry_scaler": final_scaler,
        "training_h_source": "observed_training_h", "inference_h_source": "frozen_original_h_prediction",
        "selected_by": "retained_basins_internal_whole_basin_validation", "n_train_pairs": len(train.y),
        "candidates": candidates, "refit": refit, "elapsed_fit_sec": time.monotonic() - start,
        "converter_batch_size": 2048, "learning_rate": 0.001, "weight_decay": 0.0001}
    return model, scale, static, audit


def preflight(prepared, specs, tables):
    """核验来源、站点、冻结名单和推理时刻，不训练或重推理H。"""
    expected_basins = load_basin_ids()
    validate_population(prepared, expected_basins)
    if prepared.basins != expected_basins:
        raise ValueError("原86缓存站号顺序发生变化")
    frozen = load_json(ROOT / "configs/cross_section86_original_holdouts.json")
    predictions, sources = {}, {}
    data_hashes = {name: file_sha256(ROOT / name) for name in DATA_FILES}
    code_paths = [Path(__file__), Path(__file__).with_name("survey_geometry.py"),
                  ROOT / "tools/run_cross_section86_geometry_refit.py",
                  Path(__file__).with_name("converters.py"), Path(__file__).with_name("runner.py"),
                  ROOT / "src/pipeline/dataset.py", ROOT / "src/pipeline/normalization.py"]
    code_hashes = {str(path.relative_to(ROOT)): file_sha256(path) for path in code_paths}
    positions = InferenceWindows(prepared, "test", 480, 1, ("waterlevel",))
    for scaling, spec in specs.items():
        cfg = spec["settings"]
        if (cfg["sequence_length"] != 480 or cfg["attribute_set"] != "extended"
                or cfg["converter_widths"] != [4, 8, 16] or cfg["converter_epochs"] != 50
                or cfg["converter_max_pairs_per_basin"] != 4096 or cfg["smoke"]):
            raise ValueError("参考不是约定的原强配置")
        if spec["basins"] != prepared.basins:
            raise ValueError("参考不是原86总体")
        for name, expected in spec["source_hashes"].items():
            if data_hashes.get(name) != expected:
                path = ROOT / name
                value = path.read_bytes().replace(b"\r\n", b"\n")
                import hashlib
                alternatives = [hashlib.sha256(v).hexdigest() for v in (value, value.replace(b"\n", b"\r\n"))]
                if path.suffix != ".txt" or expected not in alternatives:
                    raise ValueError(f"源缓存哈希不一致: {name}")
        for scene in spec["scenarios"]:
            if scene["heldout"] != frozen["scenarios"][scene["tag"]]["heldout"]:
                raise ValueError("留出名单与冻结原86名单不一致")
            if set(scene["heldout"]) & set(scene["retained"]):
                raise ValueError("隐藏站与保留站发生交叉")
        predictions[scaling] = {}
        sources[scaling] = {"experiment_fingerprint": spec["experiment_fingerprint"],
            "source_dir": str(SOURCE_PATHS[scaling]), "H_predictions": {}}
        for seed in cfg["model_seeds"]:
            folder = SOURCE_PATHS[scaling] / "waterlevel_models" / f"seed{seed}"
            done = load_json(folder / "training_done.json")
            if done["training_performed_this_run"] or done["scaling_info"]["flow_information_used"]:
                raise ValueError("参考H不是冻结的原单任务H模型")
            with np.load(folder / "prediction.npz") as file:
                prediction = {key: file[key].copy() for key in file.files}
            if (not np.array_equal(prediction["basin_idx"], positions.basin_idx)
                    or not np.array_equal(prediction["target_pos"], positions.target_pos)
                    or not np.isfinite(prediction["pred_h_normalized"]).all()):
                raise ValueError("冻结H预测的完整推理时刻不一致")
            predictions[scaling][seed] = prediction
            sources[scaling]["H_predictions"][str(seed)] = {
                "prediction_sha256": file_sha256(folder / "prediction.npz"),
                "model_sha256": file_sha256(folder / "model.pt"), "source_checkpoint": done["source_checkpoint"]}
    if specs["physical"]["scenarios"] != specs["observed"]["scenarios"]:
        raise ValueError("两种归一化的留出情景不同")
    for seed in (1, 2, 3):
        if not np.array_equal(predictions["physical"][seed]["pred_h_normalized"],
                              predictions["observed"][seed]["pred_h_normalized"]):
            raise ValueError("两种归一化的冻结H预测不相同")
    if not tables["physical"].equals(tables["observed"]):
        raise ValueError("两种归一化选用的原始剖面不同")
    manifest = {"protocol": PROTOCOL, "sources": sources, "data_hashes": data_hashes,
        "code_hashes": code_hashes, "basins": prepared.basins,
        "scenarios": specs["physical"]["scenarios"], "model_seeds": [1, 2, 3],
        "scalings": ["physical", "observed"], "variants": list(VARIANTS),
        "widths": [4, 8, 16], "max_epochs": 50, "max_pairs_per_basin": 4096,
        "converter_batch_size": 2048, "waterlevel_training_performed": False,
        "geometry_has_absolute_bed_elevation": False, "geometry_mode": "snapshot",
        "survey_stage_is_extra_retrospective_observation": True,
        "geometry_stage_usage": "conditional_covariate_only_not_section_water_surface_elevation",
        "q_hidden_in_training_and_validation": True,
        "capacity_control": "H-only参数量不低于完整几何模型且最多多2个参数；宽度由完整模型验证选择确定",
        "geometry_preprocessing": "inner_train_only_for_selection_retained_only_for_refit",
        "training_h_source": "observed_training_h_unchanged_from_v2",
        "new_design_after_previous_test_inspection": True,
        "numerical_environment": {"python": sys.version.split()[0], "torch": torch.__version__,
            "numpy": np.__version__, "pandas": pd.__version__}}
    return manifest, predictions


def evaluate_and_save(prepared, table, cfg, scene, seed, variant, model, scale, static, prediction, folder, run_id, metadata):
    """选模结束后才进行测试评价；保存原始输出，不据测试结果截断或重选。"""
    ids, pos = prediction["basin_idx"], prediction["target_pos"]
    x = prediction["pred_h_normalized"][:, None]
    if static is not None:
        x = np.column_stack([x, static[ids]])
    values = predict_converter(model, x.astype(np.float32), cfg["device"])
    physical = values * scale.std["flow"][ids] + scale.mean["flow"][ids]
    if not np.isfinite(physical).all():
        raise ValueError("新转换器测试预测有非有限值")
    scores = flow_metrics(prepared, ids, pos, physical, scene, seed, "stage1_refit", variant, table, cfg)
    scores.to_csv(folder / "metrics.csv", index=False, encoding="utf-8-sig")
    save_prediction(folder, ids, pos, physical)
    observed_h = prepared.targets["waterlevel"][ids, pos]
    dx = observed_h[:, None]
    if static is not None:
        dx = np.column_stack([dx, static[ids]])
    valid = np.isfinite(dx).all(axis=1)
    diagnostic = np.full(len(ids), np.nan, np.float32)
    diagnostic[valid] = (predict_converter(model, dx[valid].astype(np.float32), cfg["device"])
                         * scale.std["flow"][ids[valid]] + scale.mean["flow"][ids[valid]])
    flow_metrics(prepared, ids, pos, diagnostic, scene, seed, "stage1_refit", variant, table, cfg,
                 "observed_h_diagnostic").to_csv(folder / "diagnostic_metrics.csv", index=False, encoding="utf-8-sig")
    torch.save({"state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()},
        "input_dim": metadata["input_dim"], "hidden": metadata["hidden"],
        "flow_mean": scale.mean["flow"], "flow_std": scale.std["flow"], "flow_scale_info": scale.info,
        "geometry_scaler": metadata["final_geometry_scaler"], "variant": variant,
        "run_fingerprint": run_id}, folder / "model.pt")
    metadata.update(run_fingerprint=run_id, protocol=PROTOCOL, heldout=scene["heldout"],
        ratio=scene["ratio"], mask_seed=scene["mask_seed"], model_seed=seed, smoke_not_paper_result=False)
    write_json(folder / "done.json", metadata)


def summarize(output, experiment_id):
    """只汇总已有固定模型，不使用测试分数选择保留哪些变体。"""
    frames, runs = [], []
    for done in sorted(output.glob("*/stage1/*/*/*/done.json")):
        meta = load_json(done)
        for name in ("metrics.csv", "diagnostic_metrics.csv"):
            frame = pd.read_csv(done.parent / name, dtype={"basin": str})
            frame["run_fingerprint"] = meta["run_fingerprint"]
            frames.append(frame)
        runs.append({k: meta.get(k) for k in ("variant", "target_scaling", "ratio", "mask_seed", "model_seed",
            "hidden", "input_dim", "n_parameters", "epochs_refit", "elapsed_fit_sec")})
    if not frames:
        return len(runs)
    all_scores = pd.concat(frames, ignore_index=True)
    all_scores.to_csv(output / "all_basin_metrics.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(runs).to_csv(output / "fit_audit.csv", index=False, encoding="utf-8-sig")
    group = ["target_scaling", "variant", "ratio", "mask_seed", "model_seed", "h_input_source"]
    rows = []
    for keys, frame in all_scores.groupby(group):
        for scope, selected in {"all_86": frame, "q_heldout": frame[frame.held_out],
                "q_heldout_with_geometry": frame[frame.held_out & frame.geometry_available]}.items():
            rows.append({**dict(zip(group, keys)), "scope": scope, "n_basins": len(selected),
                "n_valid_nse": selected.nse.count(), "nse_mean": selected.nse.mean(),
                "nse_median": selected.nse.median(), "negative_nse_count": int((selected.nse < 0).sum())})
    summary = pd.DataFrame(rows)
    summary.to_csv(output / "summary_by_run.csv", index=False, encoding="utf-8-sig")
    selected = summary[(summary.scope == "q_heldout") & (summary.h_input_source == "predicted_h")]
    selected.groupby(["target_scaling", "variant", "ratio"])[["nse_mean", "nse_median", "negative_nse_count"]].mean().reset_index().to_csv(
        output / "heldout_summary.csv", index=False, encoding="utf-8-sig")
    write_json(output / "summary_manifest.json", {"experiment_fingerprint": experiment_id,
        "n_completed": len(runs), "n_expected": 324, "selection_used_test_scores": False,
        "new_design_after_previous_test_inspection": True, "all_predefined_variants_reported": True})
    return len(runs)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=ROOT / "results/cross_section86_survey_refit_v1")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--preflight", action="store_true")
    parser.add_argument("--scaling", choices=("physical", "observed"))
    parser.add_argument("--scenario")
    parser.add_argument("--model-seed", type=int, choices=(1, 2, 3))
    parser.add_argument("--variant", choices=list(VARIANTS))
    args = parser.parse_args()
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA不可用，不自动降级为CPU")
    output = args.out_dir.resolve()
    for source in SOURCE_PATHS.values():
        if output == source or source in output.parents:
            raise ValueError("新结果不得写入旧结果目录")
    prepared = PreparedData(attr_set="extended")
    specs = {name: load_json(source / "experiment.json") for name, source in SOURCE_PATHS.items()}
    tables = {name: load_survey_geometry(source, prepared, ROOT) for name, source in SOURCE_PATHS.items()}
    spec, predictions = preflight(prepared, specs, tables)
    if args.scenario and args.scenario not in {scene["tag"] for scene in spec["scenarios"]}:
        raise ValueError("指定情景不在冻结的9组留出情景中")
    experiment_id = fingerprint(spec)
    output.mkdir(parents=True, exist_ok=True)
    manifest = output / "experiment.json"
    if manifest.exists() and load_json(manifest)["experiment_fingerprint"] != experiment_id:
        if (output / "process.json").exists() or any(output.glob("*/stage1/*/*/*/model.pt")):
            raise ValueError("新目录已包含不同代码、配置或来源的实验，请指定另一个输出目录")
        # 仅预检且未训练的目录可更新；保留旧预检清单，实际拟合后禁止变更指纹。
        previous = load_json(manifest)
        write_json(output / "preflight_history" / (previous["experiment_fingerprint"] + ".json"), previous)
    write_json(manifest, {**spec, "experiment_fingerprint": experiment_id})
    table = tables["physical"]
    table.to_csv(output / "survey_geometry.csv", encoding="utf-8-sig")
    alignment = survey_stage_alignment(table)
    alignment.to_csv(output / "survey_stage_alignment.csv", index=False, encoding="utf-8-sig")
    write_json(output / "preflight.json", {"status": "passed", "n_basins": 86,
        "n_geometry": int((table.geom_missing == 0).sum()), "n_h_models_retrained": 0,
        "source_h_predictions_verified": 6, "n_planned_converter_fits": 324,
        "converter_batch_size": 2048, "new_design_after_previous_test_inspection": True,
        "survey_stage_alignment_checked": int(alignment.absolute_stage_error_m.notna().sum()),
        "survey_stage_max_nearby_error_m": alignment.absolute_stage_error_m.max(),
        "section_water_surface_mapping_verified": False})
    print(f"预检通过：原86流域，81个测量几何，冻结H预测；计划324次转换器拟合；device={args.device}", flush=True)
    if args.preflight:
        return 0
    write_json(output / "process.json", {"pid": os.getpid(), "started_at": pd.Timestamp.now().isoformat(),
        "device": args.device, "experiment_fingerprint": experiment_id, "status": "running"})
    for scaling in ([args.scaling] if args.scaling else ["physical", "observed"]):
        cfg = {**specs[scaling]["settings"], "device": args.device}
        for scene in spec["scenarios"]:
            if args.scenario and scene["tag"] != args.scenario:
                continue
            for seed in ([args.model_seed] if args.model_seed else [1, 2, 3]):
                variants = [args.variant] if args.variant else list(VARIANTS)
                for variant in variants:
                    folder = output / scaling / "stage1" / scene["tag"] / f"seed{seed}" / variant
                    folder.mkdir(parents=True, exist_ok=True)
                    run_id = fingerprint({"experiment": experiment_id, "target_scaling": scaling,
                        "scenario": scene["tag"], "seed": seed, "variant": variant})
                    done = folder / "done.json"
                    if done.exists():
                        if load_json(done)["run_fingerprint"] != run_id:
                            raise ValueError("已完成拟合的指纹不一致")
                        if not all((folder / name).is_file() for name in ("model.pt", "prediction.npz", "metrics.csv", "diagnostic_metrics.csv")):
                            raise ValueError("完成标记对应的文件不完整")
                        continue
                    forced = None
                    if variant in ("h_survey_geometry_shuffled", "h_only_capacity_control"):
                        reference_done = folder.parent / "h_survey_geometry/done.json"
                        if not reference_done.exists():
                            raise ValueError("先拟合完整几何模型，才能确定打乱与容量对照的宽度")
                        reference = load_json(reference_done)
                        if variant.endswith("_shuffled"):
                            forced = reference["hidden"]
                        else:
                            forced = int(math.ceil((reference["n_parameters"] - 1) / 3))
                    print(f"开始 {scaling} {scene['tag']} seed{seed} {variant}", flush=True)
                    model, scale, static, info = fit_converter(prepared, table, scene["retained"], variant,
                        seed, scene["mask_seed"], scaling, cfg, forced)
                    if variant == "h_only_capacity_control":
                        info["capacity_reference_parameters"] = reference["n_parameters"]
                        info["extra_parameters_vs_geometry"] = info["n_parameters"] - reference["n_parameters"]
                        if not 0 <= info["extra_parameters_vs_geometry"] <= 2:
                            raise ValueError("H-only容量对照参数量不满足约定")
                    evaluate_and_save(prepared, table, cfg, scene, seed, variant, model, scale, static,
                        predictions[scaling][seed], folder, run_id, info)
                    write_json(output / "progress.json", {"pid": os.getpid(), "last_completed":
                        f"{scaling}/{scene['tag']}/seed{seed}/{variant}", "elapsed_fit_sec": info["elapsed_fit_sec"],
                        "n_completed": len(list(output.glob("*/stage1/*/*/*/done.json"))), "n_expected": 324})
                    print(f"完成 {scaling} {scene['tag']} seed{seed} {variant}，拟合耗时={info['elapsed_fit_sec']:.1f}s", flush=True)
            summarize(output, experiment_id)
    completed = summarize(output, experiment_id)
    if completed == 324:
        write_json(output / "completed.json", {"experiment_fingerprint": experiment_id, "n_completed": completed,
            "waterlevel_models_retrained": 0, "selection_used_test_scores": False})
    write_json(output / "process.json", {"pid": os.getpid(), "status": "completed" if completed == 324 else "selected_runs_completed",
        "n_completed": completed, "experiment_fingerprint": experiment_id})
    print(f"本次指定拟合完成，完整计划进度 {completed}/324；输出 {output}", flush=True)
    return 0
