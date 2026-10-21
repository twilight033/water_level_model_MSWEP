"""16站低维阶段1：H与H+A，分别支持岭回归和小型神经网络。"""

from __future__ import annotations

import argparse
from dataclasses import replace
import os
from pathlib import Path
import time

import numpy as np
import pandas as pd
import torch

from . import area_stage1 as area
from . import area_stage1_only16 as cohort
from .area_ridge16 import ALPHAS, DEFAULT_OUTPUT as OLD_RIDGE_OUTPUT, fit_ridge
from .converters import PairData, collect_pairs, fit_network, predict_converter, validation_score
from .features import file_sha256
from .protocol import fingerprint, write_json
from .runner import save_prediction
from evaluation.metrics import basin_metrics
from pipeline.dataset import PreparedData

ROOT = area.ROOT
DEFAULT_ROOT = ROOT / "results/atgage16_lowdim_stage1_v1"
VARIANTS = ("h_only", "h_area")
N_EXPECTED = 108


def protocol(estimator):
    if estimator not in ("ridge", "neural"):
        raise ValueError("未知估计器")
    return f"atgage16_only_stage1_lowdim_{estimator}_v1"


def feature_names(variant):
    if variant not in VARIANTS:
        raise ValueError("低维对照只允许H或H+A")
    return ["waterlevel_z"] + (["area_log1p_z"] if variant == "h_area" else [])


def input_matrix(prepared, payload, pairs, variant, scaler=None):
    """不读取prepared.attrs，不加入恒定标记，直接输入维数严格为1或2。"""
    names = feature_names(variant)
    h = pairs.x[:, 0]
    columns = [h]
    if variant == "h_area":
        if scaler is None:
            raise ValueError("H+A输入需要只由允许的训练站拟合的面积尺度")
        values, available, _, _ = area.area_values(prepared, payload, pairs.basin, h, np.arange(len(prepared.basins)))
        if not available.all():
            raise ValueError("H+A对照不能使用缺面积站")
        columns.append((values - scaler["mean_log1p_area"]) / scaler["std_log1p_area"])
    result = np.column_stack(columns).astype(np.float32)
    if result.shape[1] != len(names) or not np.isfinite(result).all():
        raise ValueError("低维输入维数错误或含非有限值")
    return result


def predict(model, x, estimator, device):
    return model.predict(x) if estimator == "ridge" else predict_converter(model, x, device)


def fit_converter(prepared, payload, scene, seed, scaling, variant, estimator, cfg):
    """标签只来自16站内Q保留站；使用相同内部整站划分选择参数。"""
    started = time.monotonic()
    protocol(estimator)
    retained, hidden = set(scene["retained"]), set(scene["heldout"])
    if retained & hidden or retained | hidden != set(area.AREA_BASINS):
        raise ValueError("情景必须正好覆盖16站且训练/隐藏互斥")
    guarded = cohort.RestrictedQPrepared(prepared, scene["retained"])
    train_basins, valid_basins, split_audit = area.area_inner_split(scene["retained"], seed)
    maximum = cfg["converter_max_pairs_per_basin"]
    inner_scale = guarded.make_scaling(scaling, fit_basins=train_basins)
    train = collect_pairs(guarded, train_basins, "train", inner_scale, None, maximum)
    valid = collect_pairs(guarded, valid_basins, "valid", inner_scale, None, maximum)
    if len(np.unique(train.basin)) != len(train_basins) or len(np.unique(valid.basin)) != len(valid_basins):
        raise ValueError("内部训练/验证存在无有效配对的站")
    identity = np.arange(len(prepared.basins))
    inner_area = area.area_scaler(guarded, payload, train, identity, train_basins) if variant == "h_area" else None
    train = replace(train, x=input_matrix(guarded, payload, train, variant, inner_area))
    valid = replace(valid, x=input_matrix(guarded, payload, valid, variant, inner_area))
    candidates = []
    if estimator == "ridge":
        for alpha in ALPHAS:
            model = fit_ridge(train.x, train.y, train.weight, alpha)
            candidates.append({"alpha": alpha, "best_score": validation_score(valid, model.predict(valid.x))})
        selected = max(candidates, key=lambda row: (row["best_score"], row["alpha"]))
    else:
        for width in cfg["converter_widths"]:
            _, info = fit_network(train, int(width), cfg["converter_epochs"], seed, cfg["device"], valid,
                                  batch_size=2048, learning_rate=.001)
            candidates.append({"width": int(width), **info})
        selected = max(candidates, key=lambda row: row["best_score"])
    scale = guarded.make_scaling(scaling, fit_basins=scene["retained"])
    final = collect_pairs(guarded, scene["retained"], "train", scale, None, maximum)
    if len(np.unique(final.basin)) != len(retained):
        raise ValueError("最终拟合名单存在无配对的站")
    final_area = area.area_scaler(guarded, payload, final, identity, scene["retained"]) if variant == "h_area" else None
    final = replace(final, x=input_matrix(guarded, payload, final, variant, final_area))
    if estimator == "ridge":
        model = fit_ridge(final.x, final.y, final.weight, selected["alpha"])
        refit = {"alpha": selected["alpha"]}
        n_parameters = len(model.coef) + 1
    else:
        model, refit = fit_network(final, selected["width"], selected["best_epoch"], seed, cfg["device"],
                                   batch_size=2048, learning_rate=.001)
        n_parameters = model.n_parameters()
    audit = {"protocol": protocol(estimator), "estimator": estimator, "variant": variant,
        "target_scaling": scaling, "input_dim": final.x.shape[1], "feature_names": feature_names(variant),
        "converter_static_attributes": [], "availability_marker_included": False,
        "n_parameters": n_parameters, "alpha": selected.get("alpha"), "hidden": selected.get("width"),
        "epochs_refit": selected.get("best_epoch"), "candidates": candidates, "refit": refit,
        "n_train_pairs": len(final.y), "training_q_pair_basins": scene["retained"],
        "inner_train_basins": train_basins, "inner_validation_basins": valid_basins,
        "inner_split_audit": split_audit, "inner_area_scaler": inner_area, "final_area_scaler": final_area,
        "q_scaling_fit_calls": guarded.scaling_calls, "scale_fit_basins": scene["retained"] if scaling == "physical" else [],
        "observed_q_scale_basins": list(area.AREA_BASINS) if scaling == "observed" else [],
        "heldout_q_statistics_used_for_scaling": scaling == "observed",
        "selected_by": "retained_within16_internal_whole_basin_validation",
        "converter_population": list(area.AREA_BASINS), "original86_q_retained": scene["original86_retained"],
        "original86_q_heldout": scene["original86_heldout"],
        "training_h_source": "observed_training_h", "inference_h_source": "frozen_original_predicted_h",
        "training_area_source": "A_of_observed_training_h", "test_area_source": "A_of_predicted_h",
        "area_transform": "unchanged_log1p_max_zero_linear_area_then_training_scaler",
        "physical_scale_still_uses_original_basin_area_and_p_mean": scaling == "physical",
        "q_predictions_clipped": False, "neural_batch_size": 2048 if estimator == "neural" else None,
        "neural_learning_rate": .001 if estimator == "neural" else None,
        "neural_weight_decay": .0001 if estimator == "neural" else None,
        "elapsed_fit_sec": time.monotonic() - started}
    return model, scale, audit, final


def retained_metrics(prepared, pairs, predicted_z, scale, scene, seed, scaling, variant, estimator, period):
    rows = []
    for basin in scene["retained"]:
        bi = prepared.basin_index[basin]
        take = pairs.basin == bi
        pos = pairs.target_pos[take]
        obs = prepared.targets_raw["flow"][bi, pos].astype(float)
        predicted = predicted_z[take] * scale.std["flow"][bi] + scale.mean["flow"][bi]
        rows.append({"basin": basin, "target_scaling": scaling, "ratio": scene["ratio"], "mask_seed": scene["mask_seed"],
            "model_seed": seed, "variant": variant, "estimator": estimator, "period": period,
            "h_input_source": "observed_h", "n_negative_predictions": int((predicted < 0).sum()),
            **basin_metrics(obs, predicted)})
    return pd.DataFrame(rows)


def evaluate(prepared, payload, scene, seed, scaling, variant, estimator, cfg, model, scale, audit, final, prediction, folder, run_id):
    ids, pos = prediction["basin_idx"], prediction["target_pos"]
    assert set(np.unique(ids)) == {prepared.basin_index[b] for b in area.AREA_BASINS}
    for source, filename in (("predicted_h", "metrics.csv"), ("observed_h_diagnostic", "diagnostic_metrics.csv")):
        h = prediction["pred_h_normalized"] if source == "predicted_h" else prepared.targets["waterlevel"][ids, pos]
        valid = np.isfinite(h)
        pairs = PairData(h[valid, None], np.zeros(valid.sum()), ids[valid], np.ones(valid.sum()), pos[valid])
        x = input_matrix(prepared, payload, pairs, variant, audit["final_area_scaler"])
        values = np.full(len(ids), np.nan)
        values[valid] = predict(model, x, estimator, cfg["device"]) * scale.std["flow"][ids[valid]] + scale.mean["flow"][ids[valid]]
        frame = cohort.metrics16(prepared, payload, ids, pos, values, scene, seed, variant, scaling, source)
        frame["stage"], frame["estimator"] = "stage1_lowdim_only16", estimator
        _, _, outside, negative = area.area_values(prepared, payload, ids[valid], h[valid], np.arange(len(prepared.basins)))
        for basin in area.AREA_BASINS:
            selected = ids[valid] == prepared.basin_index[basin]
            frame.loc[frame.basin.eq(basin), "area_h_outside_fraction"] = float(outside[selected].mean()) if selected.any() else np.nan
            frame.loc[frame.basin.eq(basin), "area_negative_clipped_fraction"] = float(negative[selected].mean()) if selected.any() else np.nan
        frame.to_csv(folder / filename, index=False, encoding="utf-8-sig")
        if source == "predicted_h":
            if not np.isfinite(values).all():
                raise ValueError("正式预测含非有限Q")
            save_prediction(folder, ids, pos, values)
    for period in ("train", "valid"):
        pairs = final if period == "train" else collect_pairs(prepared, scene["retained"], "valid", scale, None,
                                                             cfg["converter_max_pairs_per_basin"])
        if period == "valid":
            pairs = replace(pairs, x=input_matrix(prepared, payload, pairs, variant, audit["final_area_scaler"]))
        estimated_z = predict(model, pairs.x, estimator, cfg["device"])
        retained_metrics(prepared, pairs, estimated_z, scale, scene, seed, scaling, variant, estimator, period).to_csv(
            folder / f"retained_{period}_metrics.csv", index=False, encoding="utf-8-sig")
    subset = [prepared.basin_index[b] for b in area.AREA_BASINS]
    state = {"protocol": protocol(estimator), "variant": variant, "feature_names": feature_names(variant),
        "input_dim": audit["input_dim"], "flow_scale_basins": list(area.AREA_BASINS),
        "flow_mean": scale.mean["flow"][subset], "flow_std": scale.std["flow"][subset],
        "waterlevel_mean": prepared.target_mean["waterlevel"][subset], "waterlevel_std": prepared.target_std["waterlevel"][subset],
        "original86_index_basins": prepared.basins, "area_scaler": audit["final_area_scaler"],
        "area_payload_fingerprint": payload["payload_fingerprint"], "area_curves": payload["curves"],
        "training_q_pair_basins": scene["retained"], "run_fingerprint": run_id}
    if estimator == "ridge":
        np.savez_compressed(folder / "model.npz", coef=model.coef, intercept=model.intercept, alpha=model.alpha,
            feature_names=np.asarray(state["feature_names"]), flow_scale_basins=np.asarray(state["flow_scale_basins"]),
            flow_mean=state["flow_mean"], flow_std=state["flow_std"], waterlevel_mean=state["waterlevel_mean"],
            waterlevel_std=state["waterlevel_std"], original86_index_basins=np.asarray(prepared.basins))
        pd.DataFrame({"feature": ["intercept", *state["feature_names"]], "coefficient": [model.intercept, *model.coef]}).to_csv(
            folder / "coefficients.csv", index=False, encoding="utf-8-sig")
    else:
        torch.save({**state, "state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()},
                    "hidden": audit["hidden"]}, folder / "model.pt")
    audit.update(run_fingerprint=run_id, heldout=scene["heldout"], ratio=scene["ratio"], mask_seed=scene["mask_seed"],
        model_seed=seed, actual_q_hidden_ratio16=scene["actual_q_hidden_ratio16"], flow_scale_info=scale.info,
        area_payload_fingerprint=payload["payload_fingerprint"], evaluation_basins=list(area.AREA_BASINS),
        primary_evaluation_basins=scene["heldout"], waterlevel_models_retrained=0, stage2_executed=False)
    write_json(folder / "done.json", audit)


def summarize(output, estimator, experiment_id):
    frames, retained, fits = [], [], []
    for path in sorted(output.glob("*/stage1/*/*/*/done.json")):
        meta = area.load_json(path)
        assert meta["protocol"] == protocol(estimator)
        assert meta["feature_names"] == feature_names(meta["variant"]) and meta["input_dim"] in (1, 2)
        for filename in ("metrics.csv", "diagnostic_metrics.csv"):
            frame = pd.read_csv(path.parent / filename, dtype={"basin": str})
            assert len(frame) == 16 and set(frame.basin) == set(area.AREA_BASINS)
            assert set(frame.loc[frame.held_out, "basin"]) == set(meta["heldout"])
            frames.append(frame)
        for period in ("train", "valid"):
            frame = pd.read_csv(path.parent / f"retained_{period}_metrics.csv", dtype={"basin": str})
            assert set(frame.basin) == set(meta["training_q_pair_basins"])
            retained.append(frame)
        fits.append({k: meta[k] for k in ("estimator", "target_scaling", "ratio", "mask_seed", "model_seed", "variant",
            "input_dim", "n_parameters", "alpha", "hidden", "epochs_refit", "n_train_pairs", "elapsed_fit_sec")})
    if not frames:
        return 0
    data, familiar = pd.concat(frames, ignore_index=True), pd.concat(retained, ignore_index=True)
    data.to_csv(output / "all_basin_metrics.csv", index=False, encoding="utf-8-sig")
    familiar.to_csv(output / "retained_fit_metrics.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(fits).to_csv(output / "fit_audit.csv", index=False, encoding="utf-8-sig")
    keys = ["target_scaling", "ratio", "mask_seed", "model_seed", "h_input_source", "basin"]
    base = data[data.variant.eq("h_only")][keys + ["nse"]].rename(columns={"nse": "h_only_nse"})
    paired = data[data.variant.eq("h_area")].merge(base, on=keys, validate="one_to_one")
    paired["delta_nse"] = paired.nse - paired.h_only_nse
    paired.to_csv(output / "paired_area_differences.csv", index=False, encoding="utf-8-sig")
    rows = []
    for (scaling, ratio, source), scores in paired.groupby(["target_scaling", "ratio", "h_input_source"]):
        for scope, mask in (("q_heldout16", scores.held_out), ("q_retained16", ~scores.held_out),
                            ("all16", np.ones(len(scores), bool))):
            selected = scores[mask]
            runs = selected.groupby(["mask_seed", "model_seed"])[["h_only_nse", "nse", "delta_nse"]].mean()
            rows.append({"estimator": estimator, "target_scaling": scaling, "ratio": ratio, "h_input_source": source,
                "scope": scope, "h_only_nse": runs.h_only_nse.mean(), "h_area_nse": runs.nse.mean(),
                "delta_nse": runs.delta_nse.mean(), "area_better_runs": int((runs.delta_nse > 0).sum()), "n_runs": len(runs),
                "h_only_station_median": selected.h_only_nse.median(), "h_area_station_median": selected.nse.median(),
                "paired_station_delta_median": selected.delta_nse.median(), "n_station_runs": len(selected),
                "h_only_negative_nse_count": int((selected.h_only_nse < 0).sum()), "h_area_negative_nse_count": int((selected.nse < 0).sum())})
    summary = pd.DataFrame(rows)
    summary.to_csv(output / "comparison_by_scope.csv", index=False, encoding="utf-8-sig")
    main = summary[summary.scope.eq("q_heldout16") & summary.h_input_source.eq("predicted_h")]
    main.to_csv(output / "primary_comparison.csv", index=False, encoding="utf-8-sig")
    fkeys = ["target_scaling", "ratio", "mask_seed", "model_seed", "period", "h_input_source", "basin"]
    fbase = familiar[familiar.variant.eq("h_only")][fkeys + ["nse"]].rename(columns={"nse": "h_only_nse"})
    fpaired = familiar[familiar.variant.eq("h_area")].merge(fbase, on=fkeys, validate="one_to_one")
    fpaired["delta_nse"] = fpaired.nse - fpaired.h_only_nse
    fpaired.to_csv(output / "paired_retained_fit_differences.csv", index=False, encoding="utf-8-sig")
    fruns = fpaired.groupby(["target_scaling", "ratio", "mask_seed", "model_seed", "period"])[["h_only_nse", "nse", "delta_nse"]].mean().reset_index()
    fruns.to_csv(output / "retained_fit_by_run.csv", index=False, encoding="utf-8-sig")
    fsummary = fruns.groupby(["target_scaling", "period"])[["h_only_nse", "nse", "delta_nse"]].mean().reset_index()
    fsummary.to_csv(output / "retained_fit_summary.csv", index=False, encoding="utf-8-sig")
    lines = [f"# 16站低维{estimator}：H与H+A", "", f"完成{len(fits)}/{N_EXPECTED}个最终模型。转换器严格只输入1或2个特征，不加入27个属性和面积可用标记。", "",
        "前面的H模型仍为原86强配置，physical归一化仍使用原流域面积/降水尺度，因此是转换器的低维对照，不是整条流程完全不用属性。", "",
        "主要结果仅用当组Q隐藏站、原测试期、冻结预测H；先逐站平均NSE，再平均9次运行。比例沿用原86名义比例。", "",
        "| 归一化 | 隐藏比例 | 仅H | H+A | 增量 | H+A更好的次数 |", "| --- | ---: | ---: | ---: | ---: | ---: |"]
    for row in main.itertuples():
        lines.append(f"| {row.target_scaling} | {row.ratio:.0%} | {row.h_only_nse:.4f} | {row.h_area_nse:.4f} | {row.delta_nse:+.4f} | {row.area_better_runs}/{row.n_runs} |")
    lines += ["", "逐站中位数、负NSE站次数和全部保留站测试成绩见comparison_by_scope.csv。", "",
        "## Q保留站拟合诊断", "", "下表用实测H、每站最多4096个配对，每项平均27次运行；valid指已见过本站训练Q之后的后续时间段，不是内部整站未见Q验证。", "",
        "| 归一化 | 时段 | 仅H | H+A | 差值 |", "| --- | --- | ---: | ---: | ---: |"]
    for row in fsummary.itertuples():
        lines.append(f"| {row.target_scaling} | {row.period} | {row.h_only_nse:.4f} | {row.nse:.4f} | {row.delta_nse:+.4f} |")
    lines += ["", "训练输入实测H，测试输入预测H，A均由对应H代入原训练期直线计算并沿用log1p及训练面积标准化。面积不是逐时刻独立实测值。", "",
        "保留原16站、11/5分析标签、9个外层情景与相同内部整站划分。所有超参只由内部验证选取，不按测试Q改曲线或删除流域。", "",
        "本轮在查看旧测试结果之后提出，属于探索性分析，不能把这些重复使用的测试分数当作全新的独立验证。observed允许隐藏站训练Q尺度，属于已知尺度对照。", ""]
    (output / "report.md").write_text("\n".join(lines), encoding="utf-8")
    write_json(output / "summary_manifest.json", {"protocol": protocol(estimator), "experiment_fingerprint": experiment_id,
        "n_completed": len(fits), "n_expected": N_EXPECTED, "primary_scope": "q_heldout16", "test_used_for_selection": False})
    return len(fits)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--estimator", required=True, choices=("ridge", "neural"))
    parser.add_argument("--device")
    parser.add_argument("--out-dir", type=Path)
    parser.add_argument("--preflight", action="store_true")
    args = parser.parse_args()
    device = args.device or ("cpu" if args.estimator == "ridge" else "cuda:0")
    if args.estimator == "ridge" and device != "cpu":
        raise ValueError("岭回归使用NumPy CPU求解，请使用--device cpu")
    if not args.preflight and args.estimator == "neural" and device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA不可用，不自动降级或改变批量")
    output = (args.out_dir or DEFAULT_ROOT / args.estimator).resolve()
    for source in (cohort.OLD_OUTPUT, cohort.DEFAULT_OUTPUT, OLD_RIDGE_OUTPUT, *area.SOURCE_PATHS.values()):
        if output == source or source in output.parents or output in source.parents:
            raise ValueError("新低维目录不能覆盖已有结果和H模型")
    payload = area.load_curves(area.CURVE_FILE)
    prepared = PreparedData(attr_set="extended")
    sources = {name: area.load_json(path / "experiment.json") for name, path in area.SOURCE_PATHS.items()}
    base, predictions, coverage = cohort.preflight(prepared, payload, sources)
    hashes = dict(base["code_hashes"])
    for path in (Path(__file__), ROOT / "src/cross_section_experiment/area_ridge16.py", ROOT / "tools/run_atgage16_lowdim.py"):
        hashes[str(path.relative_to(ROOT))] = file_sha256(path)
    spec = {**base, "protocol": protocol(args.estimator), "estimator": args.estimator, "device": device,
        "variants": list(VARIANTS), "n_expected": N_EXPECTED, "code_hashes": hashes,
        "direct_converter_features": {v: feature_names(v) for v in VARIANTS}, "converter_static_attributes": [],
        "area_availability_marker": "removed_constant_marker", "converter_hyperparameters_unchanged": False,
        "non_input_hyperparameters_and_sampling_unchanged": True, "ridge_alphas": list(ALPHAS),
        "neural_widths": [4, 8, 16], "neural_max_epochs": 50, "neural_batch_size": 2048,
        "h_model_static_attributes_unchanged": True, "physical_scale_attributes_unchanged": True,
        "new_experiment_after_prior_test_inspection": True}
    experiment_id = fingerprint(spec)
    output.mkdir(parents=True, exist_ok=True)
    manifest = output / "experiment.json"
    if manifest.exists():
        previous = area.load_json(manifest)
        if previous["protocol"] != spec["protocol"] or previous["experiment_fingerprint"] != experiment_id:
            if any(output.glob("*/stage1/*/*/*/done.json")) or any(output.glob("*/stage1/*/*/*/model.*")):
                raise ValueError("输出已有不同配置/来源的低维模型，请换目录")
    write_json(manifest, {**spec, "experiment_fingerprint": experiment_id})
    write_json(output / "area_curves.json", payload)
    pd.DataFrame(coverage).to_csv(output / "area_training_coverage.csv", index=False, encoding="utf-8-sig")
    write_json(output / "preflight.json", {"status": "PASS", "protocol": spec["protocol"], "n_expected": N_EXPECTED,
        "direct_converter_features": spec["direct_converter_features"], "converter_static_attributes": [],
        "models_fitted_during_preflight": 0, "waterlevel_models_retrained": 0})
    print(f"预检通过：{args.estimator}，仅H或H+A，108项；原H与Q尺度协议冻结", flush=True)
    if args.preflight:
        return 0
    started = time.monotonic()
    write_json(output / "process.json", {"status": "running", "pid": os.getpid(), "experiment_fingerprint": experiment_id})
    try:
        for scaling in ("physical", "observed"):
            cfg = {**sources[scaling]["settings"], "device": device}
            for scene in spec["scenarios"]:
                for seed in (1, 2, 3):
                    for variant in VARIANTS:
                        folder = output / scaling / "stage1" / scene["tag"] / f"seed{seed}" / variant
                        folder.mkdir(parents=True, exist_ok=True)
                        run_id = fingerprint({"experiment": experiment_id, "scaling": scaling, "scenario": scene["tag"],
                                              "seed": seed, "variant": variant})
                        if (folder / "done.json").exists():
                            meta = area.load_json(folder / "done.json")
                            assert meta["protocol"] == spec["protocol"] and meta["run_fingerprint"] == run_id
                            expected_model = "model.npz" if args.estimator == "ridge" else "model.pt"
                            for name in (expected_model, "prediction.npz", "metrics.csv", "diagnostic_metrics.csv",
                                         "retained_train_metrics.csv", "retained_valid_metrics.csv"):
                                if not (folder / name).is_file():
                                    raise ValueError("已完成项缺文件")
                            continue
                        model, scale, audit, final = fit_converter(prepared, payload, scene, seed, scaling, variant, args.estimator, cfg)
                        evaluate(prepared, payload, scene, seed, scaling, variant, args.estimator, cfg, model, scale, audit, final,
                                 predictions[scaling][seed], folder, run_id)
                        count = len(list(output.glob("*/stage1/*/*/*/done.json")))
                        write_json(output / "progress.json", {"n_completed": count, "n_expected": N_EXPECTED,
                            "last_completed": str(folder.relative_to(output)), "elapsed_sec": time.monotonic() - started})
                        print(f"[{count}/108] {scaling} {scene['tag']} seed{seed} {variant}", flush=True)
                summarize(output, args.estimator, experiment_id)
        count = summarize(output, args.estimator, experiment_id)
        assert count == N_EXPECTED
        write_json(output / "completed.json", {"protocol": spec["protocol"], "experiment_fingerprint": experiment_id,
            "estimator": args.estimator, "n_completed": count, "elapsed_sec": time.monotonic() - started,
            "waterlevel_models_retrained": 0, "stage2_executed": False, "input_dims": [1, 2]})
        write_json(output / "process.json", {"status": "completed", "n_completed": count, "experiment_fingerprint": experiment_id})
        print(f"完成108/108：{output / 'report.md'}", flush=True)
        return 0
    except BaseException as exc:
        write_json(output / "process.json", {"status": "failed", "error": str(exc), "experiment_fingerprint": experiment_id})
        raise


if __name__ == "__main__":
    raise SystemExit(main())
