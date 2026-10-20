"""原86阶段1：冻结H，仅在已确认16站提供训练期测站处面积曲线。"""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import re
import sys
import time

import numpy as np
import pandas as pd
import torch

from .converters import collect_pairs, fit_network, internal_basin_split, predict_converter
from .features import file_sha256
from .protocol import fingerprint, write_json
from .runner import flow_metrics, save_prediction
from .survey_refit import preflight as verify_frozen_sources, SOURCE_PATHS
from pipeline.dataset import PreparedData


ROOT = Path(__file__).resolve().parents[2]
HANDOFF = ROOT / "reports/atgage16_area_experiment_handoff_20261020.md"
PAIR_FILE = ROOT / "reports/cross_section86_area_h_overlap/area_h_pairs_inventory.csv"
CURVE_FILE = ROOT / "data/cross_sections/atgage16_area_curves.json"
PAIR_HASH = "b685f5cffd7a95d4ca188ebeca08d1f055f812767737d817a84c20d8d06b1288"
AREA_BASINS = ("01030500", "01059000", "01068910", "01072800", "010735562", "01073587",
               "01076000", "01085500", "01092000", "01094000", "01096000", "01100505",
               "01100561", "01108410", "01109000", "01109403")
FOCUS_BASINS = ("01072800", "010735562", "01073587", "01092000", "01100505")
PROTOCOL = "original86_stage1_atgage16_train_area_v1"
# 缺失标记对照用于区分面积本身与16站身份信息，全部变体都重新拟合。
VARIANTS = ("h_only", "h_attrs", "h_attrs_area_available", "h_attrs_area",
            "h_attrs_area_shuffled", "h_attrs_capacity_control")


def load_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def visit_weights(frame):
    return 1 / frame.groupby("field_visit_id").field_visit_id.transform("size").to_numpy(float)


def fit_line(frame):
    x, y, w = frame.h_m.to_numpy(float), frame.area_m2.to_numpy(float), visit_weights(frame)
    mean = np.average(x, weights=w)
    scale = max(np.sqrt(np.average((x - mean) ** 2, weights=w)), 1e-8)
    design = np.column_stack([np.ones(len(x)), (x - mean) / scale])
    coef = np.linalg.lstsq(design * np.sqrt(w)[:, None], y * np.sqrt(w), rcond=None)[0]
    return {"intercept_m2": float(coef[0] - coef[1] * mean / scale),
            "slope_m": float(coef[1] / scale), "h_min_m": float(x.min()), "h_max_m": float(x.max())}


def curve_scores(frame, predictions):
    w, y = visit_weights(frame), frame.area_m2.to_numpy(float)
    mean = np.average(y, weights=w)
    variance = np.sum(w * (y - mean) ** 2)
    error = np.asarray(predictions) - y
    mae = float(np.average(abs(error), weights=w))
    return {"r2": float(1 - np.sum(w * error ** 2) / variance) if variance > 1e-15 else None,
            "mae_m2": mae, "relative_mae": mae / mean}


def prepare_curves(destination=CURVE_FILE):
    """严格重建交接中的396条记录；便携JSON只保存H、面积及追溯元数据。"""
    if file_sha256(PAIR_FILE) != PAIR_HASH:
        raise ValueError("面积配对源文件与已确认交接不一致，不能静默重新筛选")
    columns = ["basin", "record_id", "field_visit_id", "measurement_time", "model_period", "area_m2",
               "h_at_measurement_interpolated_m", "channel_location_direction", "channel_location_distance",
               "channel_name", "measurement_type"]
    data = pd.read_csv(PAIR_FILE, usecols=columns,
                       dtype={"basin": str, "record_id": str, "field_visit_id": str}, low_memory=False)
    name = data.channel_name.fillna("").str.strip().str.lower().str.replace(r"\s+", " ", regex=True)
    rejected = name.str.contains(r"check|\bchk\b|secondary|inflow|\bleft\b|\bright\b|channel 2|^2$", regex=True)
    distance = pd.to_numeric(data.channel_location_distance, errors="coerce")
    absent = data.channel_location_distance.isna() | data.channel_location_distance.astype(str).str.strip().eq("")
    keep = (data.model_period.eq("train") & np.isfinite(data.area_m2) & data.area_m2.gt(0)
            & np.isfinite(data.h_at_measurement_interpolated_m)
            & data.channel_location_direction.eq("AtTheGage") & (absent | distance.eq(0))
            & ~data.measurement_type.fillna("").str.strip().str.casefold().eq("ice") & ~rejected)
    data = data[keep].copy()
    data["h_m"] = data.h_at_measurement_interpolated_m
    counts = data.groupby("basin").agg(visits=("field_visit_id", "nunique"), h=("h_m", "nunique"))
    selected = set(counts.index[(counts.visits >= 15) & (counts.h >= 6)])
    if selected != set(AREA_BASINS):
        raise ValueError(f"重建名单与已确认16站不一致: {sorted(selected ^ set(AREA_BASINS))}")
    data = data[data.basin.isin(AREA_BASINS)].copy()
    if len(data) != 396 or data.field_visit_id.nunique() != 369 or data.field_visit_id.isna().any():
        raise ValueError("重建记录数/独立访问数不是396/369，或存在缺访问ID")
    assert data.record_id.is_unique
    data["measurement_time"] = pd.to_datetime(data.measurement_time, format="ISO8601", utc=True).dt.tz_localize(None)
    handoff = HANDOFF.read_text(encoding="utf-8-sig")
    declared = {}
    for line in handoff.splitlines():
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        if len(cells) == 5 and cells[0] in AREA_BASINS:
            declared[cells[0]] = list(map(float, cells[1:]))
    curves, records, rows, folds = {}, [], [], 0
    for basin in AREA_BASINS:
        frame = data[data.basin.eq(basin)].sort_values(["measurement_time", "record_id"]).reset_index(drop=True)
        curve = fit_line(frame)
        if not np.allclose([curve[k] for k in ("intercept_m2", "slope_m", "h_min_m", "h_max_m")],
                           declared[basin], rtol=1e-10, atol=1e-9):
            raise ValueError(f"面积曲线系数与交接不一致: {basin}")
        prediction = np.full(len(frame), np.nan)
        for visit in frame.field_visit_id.unique():
            test = frame.field_visit_id.eq(visit)
            train = frame[~test]
            assert set(train.field_visit_id).isdisjoint(frame.loc[test, "field_visit_id"])
            fitted = fit_line(train)
            prediction[test] = fitted["intercept_m2"] + fitted["slope_m"] * frame.loc[test, "h_m"]
            # 用独立协方差公式核对留出预测，避免重用拟合算法自证。
            w = visit_weights(train)
            x, y = train.h_m.to_numpy(), train.area_m2.to_numpy()
            mx, my = np.average(x, weights=w), np.average(y, weights=w)
            slope = np.sum(w * (x - mx) * (y - my)) / np.sum(w * (x - mx) ** 2)
            assert np.allclose(prediction[test], my + slope * (frame.loc[test, "h_m"] - mx), atol=1e-8)
            folds += 1
        order = frame.groupby("field_visit_id").measurement_time.min().sort_values(kind="stable").index
        n_test = max(5, math.ceil(.2 * len(order)))
        test = frame.field_visit_id.isin(order[-n_test:])
        cutoff = frame.loc[test, "measurement_time"].min()
        early = frame.groupby("field_visit_id").measurement_time.max()
        train = frame[frame.field_visit_id.isin(early[early < cutoff].index)]
        assert set(train.field_visit_id).isdisjoint(frame.loc[test, "field_visit_id"])
        assert train.measurement_time.max() < cutoff
        fitted = fit_line(train)
        time_score = curve_scores(frame[test], fitted["intercept_m2"] + fitted["slope_m"] * frame.loc[test, "h_m"])
        score = curve_scores(frame, prediction)
        group = "focus5" if basin in FOCUS_BASINS else "qualified11"
        curve.update(n_records=len(frame), n_visits=len(order), group=group,
                     lovo_r2=score["r2"], time_r2=time_score["r2"], lovo_relative_mae=score["relative_mae"])
        assert (score["r2"] >= .8 and time_score["r2"] >= .8) == (group == "qualified11")
        curves[basin] = curve
        rows.append({"basin": basin, **curve})
        for row in frame.itertuples(index=False):
            records.append({"basin": basin, "record_id": row.record_id, "field_visit_id": row.field_visit_id,
                            "measurement_time": row.measurement_time.isoformat(), "model_period": "train",
                            "h_m": row.h_m, "area_m2": row.area_m2})
        folds += 1
    payload = {"protocol": PROTOCOL, "area_basins": list(AREA_BASINS), "focus_basins": list(FOCUS_BASINS),
               "source_pair_sha256": PAIR_HASH, "frozen_splits_sha256": file_sha256(ROOT / "data/splits/basin_splits.json"),
               "handoff_sha256": file_sha256(HANDOFF), "curves": curves, "training_records": records,
               "n_records": len(records), "n_visits": 369, "independently_checked_geometry_folds": folds,
               "read_q_velocity_or_rating": False, "curve_type": "linear", "stage_units": "m", "area_units": "m2"}
    payload["payload_fingerprint"] = fingerprint(payload)
    write_json(destination, payload)
    report_dir = ROOT / "reports/atgage16_area_curve_rebuild"
    report_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(report_dir / "curve_parameters.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(records).to_csv(report_dir / "geometry_training_records.csv", index=False, encoding="utf-8-sig")
    write_json(report_dir / "integrity.json", {"status": "PASS", "n_basins": 16, "n_records": 396,
               "n_visits": 369, "validated_folds": folds, "parameters_match_handoff": True,
               "source_hash_unchanged": file_sha256(PAIR_FILE) == PAIR_HASH})
    return payload


def load_curves(path):
    payload = load_json(path) if path.exists() else prepare_curves(path)
    actual = {k: v for k, v in payload.items() if k != "payload_fingerprint"}
    if fingerprint(actual) != payload["payload_fingerprint"] or payload["area_basins"] != list(AREA_BASINS):
        raise ValueError("面积参数包内容或16站名单已改变")
    if payload["focus_basins"] != list(FOCUS_BASINS) or payload["source_pair_sha256"] != PAIR_HASH:
        raise ValueError("重点关注组或原始几何来源已改变")
    if file_sha256(ROOT / "data/splits/basin_splits.json") != payload["frozen_splits_sha256"]:
        raise ValueError("冻结时间划分与面积曲线包不一致")
    records = pd.DataFrame(payload["training_records"])
    for basin in AREA_BASINS:
        frame = records[records.basin.eq(basin)]
        curve = fit_line(frame)
        if not np.allclose([curve[k] for k in curve], [payload["curves"][basin][k] for k in curve], atol=1e-9):
            raise ValueError("面积参数不能从随包训练期几何记录重建")
    return payload


def area_inner_split(retained, seed):
    """保留原内部验证比例，必要时最少交换一站，确保至少2个面积训练站和1个面积验证站。"""
    train, valid = internal_basin_split(retained, seed + 1000)
    original = {"train": list(train), "valid": list(valid)}
    available = set(retained) & set(AREA_BASINS)
    if len(available) < 3:
        raise ValueError("该冻结情景不足3个面积可用Q保留站，不能进行整站内部验证")
    if not available.intersection(valid):
        into_valid = next(b for b in train if b in available)
        into_train = next(b for b in valid if b not in available)
        train[train.index(into_valid)] = into_train
        valid[valid.index(into_train)] = into_valid
    while len(available.intersection(train)) < 2:
        into_train = next(b for b in valid if b in available)
        into_valid = next(b for b in train if b not in available)
        train[train.index(into_valid)] = into_train
        valid[valid.index(into_train)] = into_valid
    assert set(train).isdisjoint(valid) and set(train + valid) == set(retained)
    return train, valid, {"original": original, "changed": train != original["train"] or valid != original["valid"]}


def donor_mapping(prepared, seed, shuffled):
    result = np.arange(len(prepared.basins))
    if shuffled:
        eligible = np.array([prepared.basin_index[b] for b in AREA_BASINS])
        rng = np.random.default_rng(seed)
        for _ in range(1000):
            donors = rng.permutation(eligible)
            if (donors != eligible).all():
                break
        else:
            donors = np.roll(eligible, 1)
        result[eligible] = donors
    return result


def area_values(prepared, payload, ids, h_normalized, donors):
    """实际或打乱曲线均连同H基准/尺度使用，不把其他站原始水位基准硬套过来。"""
    ids, h_normalized = np.asarray(ids, int), np.asarray(h_normalized, float)
    available = np.array([b in payload["curves"] for b in prepared.basins])[ids]
    source_ids = donors[ids]
    h_m = (h_normalized * prepared.target_std["waterlevel"][source_ids]
           + prepared.target_mean["waterlevel"][source_ids])
    raw, outside = np.zeros(len(ids)), np.zeros(len(ids), bool)
    for source in np.unique(source_ids[available]):
        curve = payload["curves"][prepared.basins[source]]
        selected = available & (source_ids == source)
        raw[selected] = curve["intercept_m2"] + curve["slope_m"] * h_m[selected]
        outside[selected] = (h_m[selected] < curve["h_min_m"]) | (h_m[selected] > curve["h_max_m"])
    # 物理输入保证面积非负；不截断水位范围，继续使用直线并单列外推和负值截断比例。
    area = np.maximum(raw, 0)
    return np.log1p(area), available, outside, (raw < 0) & available


def input_matrix(prepared, payload, pairs, variant, scaler, donors):
    ids, h = pairs.basin, pairs.x[:, 0]
    pieces = [h[:, None]]
    if variant != "h_only":
        pieces.append(prepared.attrs[ids])
    if variant not in ("h_only", "h_attrs"):
        values, available, _, _ = area_values(prepared, payload, ids, h, donors)
        if variant in ("h_attrs_area", "h_attrs_area_shuffled"):
            if scaler is None:
                raise ValueError("面积输入尚未按内部训练站拟合尺度")
            z = np.where(available, (values - scaler["mean_log1p_area"]) / scaler["std_log1p_area"], 0)
            pieces.append(z[:, None])
        pieces.append(available.astype(float)[:, None])
    result = np.column_stack(pieces).astype(np.float32)
    if not np.isfinite(result).all():
        raise ValueError("转换器输入含非有限H、属性或面积值")
    return result


def area_scaler(prepared, payload, pairs, donors, fit_basins):
    values, available, _, _ = area_values(prepared, payload, pairs.basin, pairs.x[:, 0], donors)
    if not available.any():
        raise ValueError("内部训练站没有可用面积")
    w = pairs.weight[available].astype(float)
    mean = np.average(values[available], weights=w)
    std = max(float(np.sqrt(np.average((values[available] - mean) ** 2, weights=w))), 1e-6)
    return {"mean_log1p_area": float(mean), "std_log1p_area": std,
            "fit_basins": sorted(set(fit_basins) & set(AREA_BASINS)),
            "fit_information": "retained_training_h_only_no_hidden_q_no_test_h"}


def transform_pairs(prepared, payload, pairs, variant, scaler, donors):
    from dataclasses import replace
    return replace(pairs, x=input_matrix(prepared, payload, pairs, variant, scaler, donors))


def fit_converter(prepared, payload, scene, variant, seed, scaling, cfg, forced_width=None):
    started = time.monotonic()
    retained = scene["retained"]
    train_basins, valid_basins, split_audit = area_inner_split(retained, seed)
    donors = donor_mapping(prepared, scene["mask_seed"] + 2000, variant.endswith("_shuffled"))
    maximum = cfg["converter_max_pairs_per_basin"]
    inner_scale = prepared.make_scaling(scaling, fit_basins=train_basins)
    train = collect_pairs(prepared, train_basins, "train", inner_scale, None, maximum)
    valid = collect_pairs(prepared, valid_basins, "valid", inner_scale, None, maximum)
    uses_area = variant in ("h_attrs_area", "h_attrs_area_shuffled")
    inner = area_scaler(prepared, payload, train, donors, train_basins) if uses_area else None
    train = transform_pairs(prepared, payload, train, variant, inner, donors)
    valid = transform_pairs(prepared, payload, valid, variant, inner, donors)
    area_indices = {prepared.basin_index[b] for b in AREA_BASINS}
    if len(set(train.basin) & area_indices) < 2 or not (set(valid.basin) & area_indices):
        raise ValueError("面积内部训练/验证站没有足够实际H—Q配对，不能继续选模")
    candidates = []
    for width in ([forced_width] if forced_width is not None else cfg["converter_widths"]):
        print(f"  内部验证 width={width}；面积训练站={len(set(train_basins) & set(AREA_BASINS))}，面积验证站={len(set(valid_basins) & set(AREA_BASINS))}", flush=True)
        _, info = fit_network(train, int(width), cfg["converter_epochs"], seed, cfg["device"], valid,
                              batch_size=2048, learning_rate=.001)
        candidates.append({"width": int(width), **info})
    selected = max(candidates, key=lambda v: v["best_score"])
    scale = prepared.make_scaling(scaling, fit_basins=retained)
    final_train = collect_pairs(prepared, retained, "train", scale, None, maximum)
    final = area_scaler(prepared, payload, final_train, donors, retained) if uses_area else None
    final_train = transform_pairs(prepared, payload, final_train, variant, final, donors)
    model, info = fit_network(final_train, selected["width"], selected["best_epoch"], seed, cfg["device"],
                               batch_size=2048, learning_rate=.001)
    audit = {"variant": variant, "target_scaling": scaling, "input_dim": final_train.x.shape[1],
             "hidden": selected["width"], "n_parameters": model.n_parameters(),
             "epochs_refit": selected["best_epoch"], "n_train_pairs": len(final_train.y),
             "training_q_pair_basins": retained, "inner_train_basins": train_basins,
             "inner_validation_basins": valid_basins, "inner_split_audit": split_audit,
             "n_area_train_basins": len(set(retained) & set(AREA_BASINS)),
             "n_area_inner_train_basins": len(set(train_basins) & set(AREA_BASINS)),
             "n_area_inner_validation_basins": len(set(valid_basins) & set(AREA_BASINS)),
             "inner_area_scaler": inner, "final_area_scaler": final,
             "donor_basins": {prepared.basins[i]: prepared.basins[donors[i]] for i in range(len(donors))
                              if prepared.basins[i] in AREA_BASINS},
             "scale_fit_basins": retained if scaling == "physical" else [],
             "observed_q_scale_basins": prepared.basins if scaling == "observed" else [],
             "heldout_q_statistics_used_for_scaling": scaling == "observed",
             "selected_by": "retained_internal_whole_basin_validation_area_stratified",
             "training_h_source": "observed_training_h_unchanged_from_previous_stage1",
             "inference_h_source": "frozen_original_single_waterlevel_prediction",
             "training_area_source": "A_of_observed_training_h", "test_area_source": "A_of_predicted_h",
             "input_error_distribution_shift": True, "converter_batch_size": 2048,
             "learning_rate": .001, "weight_decay": .0001, "candidates": candidates, "refit": info,
             "elapsed_fit_sec": time.monotonic() - started}
    return model, scale, donors, audit


def evaluate(prepared, payload, scene, seed, variant, cfg, model, scale, donors, prediction, folder, audit, run_id):
    from .converters import PairData
    ids, pos = prediction["basin_idx"], prediction["target_pos"]
    table = pd.DataFrame({"geom_missing": [int(b not in AREA_BASINS) for b in prepared.basins]}, index=prepared.basins)
    all_metrics = []
    for source in ("predicted_h", "observed_h_diagnostic"):
        h = prediction["pred_h_normalized"] if source == "predicted_h" else prepared.targets["waterlevel"][ids, pos]
        valid = np.isfinite(h)
        pairs = PairData(h[valid, None], np.zeros(valid.sum()), ids[valid], np.ones(valid.sum()), pos[valid])
        x = input_matrix(prepared, payload, pairs, variant, audit["final_area_scaler"], donors)
        normalized = predict_converter(model, x, cfg["device"])
        values = np.full(len(ids), np.nan)
        values[valid] = normalized * scale.std["flow"][ids[valid]] + scale.mean["flow"][ids[valid]]
        scores = flow_metrics(prepared, ids, pos, values, scene, seed, "stage1_area", variant, table, cfg, source)
        _, available, outside, negative = area_values(prepared, payload, ids[valid], h[valid], np.arange(len(prepared.basins)))
        _, _, feature_outside, feature_negative = area_values(prepared, payload, ids[valid], h[valid], donors)
        for bi, basin in enumerate(prepared.basins):
            selected = ids[valid] == bi
            scores.loc[scores.basin.eq(basin), "area_h_outside_fraction"] = float(outside[selected].mean()) if selected.any() and basin in AREA_BASINS else np.nan
            scores.loc[scores.basin.eq(basin), "area_negative_clipped_fraction"] = float(negative[selected].mean()) if selected.any() and basin in AREA_BASINS else np.nan
            scores.loc[scores.basin.eq(basin), "feature_area_h_outside_fraction"] = float(feature_outside[selected].mean()) if selected.any() and basin in AREA_BASINS else np.nan
            scores.loc[scores.basin.eq(basin), "feature_area_negative_clipped_fraction"] = float(feature_negative[selected].mean()) if selected.any() and basin in AREA_BASINS else np.nan
            if basin in payload["curves"]:
                curve = payload["curves"][basin]
                for key in ("lovo_r2", "time_r2", "lovo_relative_mae"):
                    scores.loc[scores.basin.eq(basin), "geometry_" + key] = curve[key]
        scores["area_analysis_group"] = scores.basin.map(lambda b: "focus5" if b in FOCUS_BASINS else "qualified11" if b in AREA_BASINS else "outside16")
        scores.to_csv(folder / ("metrics.csv" if source == "predicted_h" else "diagnostic_metrics.csv"), index=False, encoding="utf-8-sig")
        all_metrics.append(scores)
        if source == "predicted_h":
            if not np.isfinite(values).all():
                raise ValueError("冻结预测H的Q输出有非有限值")
            save_prediction(folder, ids, pos, values)
    torch.save({"state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()},
                "input_dim": audit["input_dim"], "hidden": audit["hidden"], "variant": variant,
                "flow_mean": scale.mean["flow"], "flow_std": scale.std["flow"], "flow_scale_info": scale.info,
                "area_scaler": audit["final_area_scaler"], "area_payload_fingerprint": payload["payload_fingerprint"],
                "area_curves": payload["curves"], "donor_basins": audit["donor_basins"],
                "run_fingerprint": run_id}, folder / "model.pt")
    audit.update(protocol=PROTOCOL, run_fingerprint=run_id, heldout=scene["heldout"], ratio=scene["ratio"],
                 mask_seed=scene["mask_seed"], model_seed=seed, smoke_not_paper_result=False)
    write_json(folder / "done.json", audit)


def summarize(output, experiment_id):
    frames, fits = [], []
    for done in sorted(output.glob("*/stage1/*/*/*/done.json")):
        meta = load_json(done)
        for filename in ("metrics.csv", "diagnostic_metrics.csv"):
            frames.append(pd.read_csv(done.parent / filename, dtype={"basin": str}))
        fits.append({k: meta.get(k) for k in ("variant", "target_scaling", "ratio", "mask_seed", "model_seed",
                     "hidden", "n_parameters", "epochs_refit", "n_area_train_basins", "elapsed_fit_sec")})
    if not frames:
        return 0
    all_scores = pd.concat(frames, ignore_index=True)
    all_scores.to_csv(output / "all_basin_metrics.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(fits).to_csv(output / "fit_audit.csv", index=False, encoding="utf-8-sig")
    groups = ["target_scaling", "variant", "ratio", "mask_seed", "model_seed", "h_input_source"]
    rows = []
    for key, frame in all_scores.groupby(groups):
        for label, mask in {"all86": np.ones(len(frame), bool), "q_heldout86": frame.held_out,
                            "all16": frame.geometry_available,
                            "qualified11": frame.area_analysis_group.eq("qualified11"),
                            "focus5": frame.area_analysis_group.eq("focus5"),
                            "q_heldout16": frame.held_out & frame.geometry_available,
                            "q_heldout_qualified11": frame.held_out & frame.area_analysis_group.eq("qualified11"),
                            "q_heldout_focus5": frame.held_out & frame.area_analysis_group.eq("focus5")}.items():
            selected = frame[mask]
            rows.append({**dict(zip(groups, key)), "scope": label, "n_basins": len(selected),
                         "nse_mean": selected.nse.mean(), "nse_median": selected.nse.median(),
                         "negative_nse_count": int((selected.nse < 0).sum())})
    summary = pd.DataFrame(rows)
    summary.to_csv(output / "summary_by_run.csv", index=False, encoding="utf-8-sig")
    subset = summary[summary.h_input_source.eq("predicted_h")]
    subset.groupby(["target_scaling", "variant", "ratio", "scope"])[["nse_mean", "nse_median", "negative_nse_count"]].mean().reset_index().to_csv(
        output / "group_summary.csv", index=False, encoding="utf-8-sig")
    join = ["target_scaling", "ratio", "mask_seed", "model_seed", "h_input_source", "basin"]
    baseline = all_scores[all_scores.variant.eq("h_attrs_area_available")][join + ["nse"]].rename(columns={"nse": "baseline_nse"})
    differences = all_scores.merge(baseline, on=join, how="left", validate="many_to_one")
    differences["delta_nse_vs_same_availability_control"] = differences.nse - differences.baseline_nse
    differences.to_csv(output / "paired_differences.csv", index=False, encoding="utf-8-sig")
    write_json(output / "summary_manifest.json", {"experiment_fingerprint": experiment_id, "n_completed": len(fits),
               "n_expected": 324, "selection_used_test_scores": False, "all16_included": True,
               "focus5_retained": list(FOCUS_BASINS), "stage2_executed": False})
    return len(fits)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=ROOT / "results/atgage16_area_stage1_v1")
    parser.add_argument("--curves", type=Path, default=CURVE_FILE)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--prepare-curves-only", action="store_true")
    parser.add_argument("--preflight", action="store_true")
    parser.add_argument("--scaling", choices=("physical", "observed"))
    parser.add_argument("--scenario")
    parser.add_argument("--model-seed", type=int, choices=(1, 2, 3))
    parser.add_argument("--variant", choices=VARIANTS)
    args = parser.parse_args()
    if args.prepare_curves_only:
        prepare_curves(args.curves)
        print(f"曲线重建通过：16站、396条记录、369次访问、385个验证折；{args.curves}", flush=True)
        return 0
    if not args.preflight and args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA不可用，不自动降级或改变批量")
    payload = load_curves(args.curves)
    prepared = PreparedData(attr_set="extended")
    specs = {name: load_json(path / "experiment.json") for name, path in SOURCE_PATHS.items()}
    table = pd.DataFrame({"geom_missing": [int(b not in AREA_BASINS) for b in prepared.basins]}, index=prepared.basins)
    verified, predictions = verify_frozen_sources(prepared, specs, {"physical": table, "observed": table.copy()})
    records = pd.DataFrame(payload["training_records"])
    for basin in AREA_BASINS:
        when = pd.to_datetime(records.loc[records.basin.eq(basin), "measurement_time"])
        split = prepared.splits["splits"][basin]
        if not when.between(pd.Timestamp(split["train_start"]), pd.Timestamp(split["train_end"])).all():
            raise ValueError("面积参数包含原训练期之外的几何记录")
    counts = []
    for scene in verified["scenarios"]:
        for seed in (1, 2, 3):
            train, valid, audit = area_inner_split(scene["retained"], seed)
            effective = {}
            for label, basins, split in (("train", train, "train"), ("valid", valid, "valid")):
                effective[label] = []
                for basin in set(basins) & set(AREA_BASINS):
                    bi = prepared.basin_index[basin]
                    lo, hi = prepared.split_range(basin, split)
                    h = prepared.targets["waterlevel"][bi, lo:hi + 1]
                    q = prepared.targets_raw["flow"][bi, lo:hi + 1]
                    if (np.isfinite(h) & np.isfinite(q)).any():
                        effective[label].append(basin)
            if len(effective["train"]) < 2 or len(effective["valid"]) < 1:
                raise ValueError(f"{scene['tag']} seed{seed}的面积内部验证没有足够实际H—Q配对")
            counts.append({"tag": scene["tag"], "model_seed": seed, "n_retained": len(scene["retained"]),
                           "n_area_retained": len(set(scene["retained"]) & set(AREA_BASINS)),
                           "n_area_heldout": len(set(scene["heldout"]) & set(AREA_BASINS)),
                           "n_area_inner_train": len(set(train) & set(AREA_BASINS)),
                           "n_area_inner_valid": len(set(valid) & set(AREA_BASINS)),
                           "n_effective_area_inner_train": len(effective["train"]),
                           "n_effective_area_inner_valid": len(effective["valid"]),
                           "inner_split_changed": audit["changed"], "inner_train": train, "inner_valid": valid})
    code_hashes = dict(verified["code_hashes"])
    for path in (Path(__file__), ROOT / "tools/run_atgage16_area_stage1.py"):
        code_hashes[str(path.relative_to(ROOT))] = file_sha256(path)
    spec = {"protocol": PROTOCOL, "basins": prepared.basins, "sources": verified["sources"],
            "data_hashes": verified["data_hashes"], "code_hashes": code_hashes,
            "numerical_environment": verified["numerical_environment"], "scenarios": verified["scenarios"],
            "source_settings": {k: v["settings"] for k, v in specs.items()},
            "area_payload_fingerprint": payload["payload_fingerprint"], "area_basins": list(AREA_BASINS),
            "focus_basins": list(FOCUS_BASINS), "variants": list(VARIANTS), "scalings": ["physical", "observed"],
            "n_expected": 324, "widths": [4, 8, 16], "converter_epochs": 50, "converter_batch_size": 2048,
            "converter_max_pairs_per_basin": 4096, "learning_rate": .001, "weight_decay": .0001,
            "area_feature": "log1p(max(a+b*H_m,0)); fit mean/std on eligible inner train then retained train",
            "outside_h_range_policy": "linear_extrapolation_with_recorded_outside_and_negative_clipped_fractions",
            "area_curve_fits_use_q": False, "area_curve_information": "original_training_period_atgage_geometry",
            "training_h_source": "observed_training_h", "test_h_source": "frozen_original_h_prediction",
            "waterlevel_models_retrained": 0, "stage2_executed": False,
            "q_hidden_in_training_and_validation": True,
            "observed_is_known_q_scale_control": True,
            "inner_split_policy": "original_retained_split_minimal_swaps_ensure_area_train2_valid1",
            "inner_splits": counts, "new_design_after_previous_test_inspection": True}
    experiment_id = fingerprint(spec)
    output = args.out_dir.resolve()
    for source in SOURCE_PATHS.values():
        if output == source or source in output.parents:
            raise ValueError("不得写入原强配置结果目录")
    if args.scenario and args.scenario not in {s["tag"] for s in spec["scenarios"]}:
        raise ValueError("指定情景不在原冻结名单内")
    output.mkdir(parents=True, exist_ok=True)
    manifest = output / "experiment.json"
    if manifest.exists() and load_json(manifest)["experiment_fingerprint"] != experiment_id:
        if any(output.glob("*/stage1/*/*/*/done.json")) or any(output.glob("*/stage1/*/*/*/model.pt")):
            raise ValueError("结果目录已有其他指纹模型，请换输出目录")
        previous = load_json(manifest)
        write_json(output / "preflight_history" / (previous["experiment_fingerprint"] + ".json"), previous)
    write_json(manifest, {**spec, "experiment_fingerprint": experiment_id})
    pd.DataFrame([{k: v for k, v in row.items() if k not in ("inner_train", "inner_valid")} for row in counts]).to_csv(
        output / "area_training_coverage.csv", index=False, encoding="utf-8-sig")
    write_json(output / "area_curves.json", payload)
    write_json(output / "preflight.json", {"status": "PASS", "n_basins": 86, "n_area_basins": 16,
               "n_planned_converter_fits": 324, "waterlevel_models_retrained": 0, "stage2_executed": False,
               "n_original_inner_splits_adjusted": sum(row["inner_split_changed"] for row in counts),
               "minimum_area_inner_train": min(row["n_area_inner_train"] for row in counts),
               "minimum_area_inner_valid": min(row["n_area_inner_valid"] for row in counts)})
    print(f"预检通过：原86总体、16站面积，冻结H；阶段1共324个转换器；内部验证调整{sum(r['inner_split_changed'] for r in counts)}/27组", flush=True)
    if args.preflight:
        return 0
    write_json(output / "process.json", {"pid": os.getpid(), "status": "running", "experiment_fingerprint": experiment_id})
    for scaling in ([args.scaling] if args.scaling else ["physical", "observed"]):
        cfg = {**specs[scaling]["settings"], "device": args.device, "target_scaling": scaling,
               "geometry_mode": "train_atgage_area_curve"}
        for scene in spec["scenarios"]:
            if args.scenario and args.scenario != scene["tag"]:
                continue
            for seed in ([args.model_seed] if args.model_seed else [1, 2, 3]):
                for variant in ([args.variant] if args.variant else VARIANTS):
                    folder = output / scaling / "stage1" / scene["tag"] / f"seed{seed}" / variant
                    folder.mkdir(parents=True, exist_ok=True)
                    run_id = fingerprint({"experiment": experiment_id, "scaling": scaling, "scenario": scene["tag"], "seed": seed, "variant": variant})
                    if (folder / "done.json").exists():
                        if load_json(folder / "done.json")["run_fingerprint"] != run_id:
                            raise ValueError("已完成转换器的指纹不一致")
                        if not all((folder / f).is_file() for f in ("model.pt", "prediction.npz", "metrics.csv", "diagnostic_metrics.csv")):
                            raise ValueError("转换器完成文件不完整")
                        continue
                    forced, reference = None, None
                    if variant in ("h_attrs_area_shuffled", "h_attrs_capacity_control"):
                        path = folder.parent / "h_attrs_area/done.json"
                        if not path.is_file():
                            raise ValueError("请先完成h_attrs_area，再运行打乱或容量对照")
                        reference = load_json(path)
                        forced = reference["hidden"] if variant.endswith("_shuffled") else math.ceil(
                            (reference["n_parameters"] - 1) / (1 + prepared.attrs.shape[1] + 1 + 2))
                    print(f"开始 {scaling} {scene['tag']} seed{seed} {variant}", flush=True)
                    model, scale, donors, audit = fit_converter(prepared, payload, scene, variant, seed, scaling, cfg, forced)
                    if variant == "h_attrs_capacity_control":
                        audit["capacity_reference_parameters"] = reference["n_parameters"]
                        audit["extra_parameters_vs_area"] = audit["n_parameters"] - reference["n_parameters"]
                        if not 0 <= audit["extra_parameters_vs_area"] < (audit["input_dim"] + 2):
                            raise ValueError("容量上界对照的参数量不满足约定")
                    evaluate(prepared, payload, scene, seed, variant, cfg, model, scale, donors,
                             predictions[scaling][seed], folder, audit, run_id)
                    write_json(output / "progress.json", {"last_completed": f"{scaling}/{scene['tag']}/seed{seed}/{variant}",
                               "n_completed": len(list(output.glob("*/stage1/*/*/*/done.json"))), "n_expected": 324})
            summarize(output, experiment_id)
    completed = summarize(output, experiment_id)
    write_json(output / "process.json", {"pid": os.getpid(), "status": "completed" if completed == 324 else "selected_runs_completed",
               "experiment_fingerprint": experiment_id, "n_completed": completed})
    if completed == 324:
        write_json(output / "completed.json", {"experiment_fingerprint": experiment_id, "n_completed": 324,
                   "stage2_executed": False, "waterlevel_models_retrained": 0})
    print(f"指定拟合完成；完整计划{completed}/324；输出{output}", flush=True)
    return 0
