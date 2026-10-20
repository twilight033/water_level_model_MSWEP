"""原86站水位—湿润长度/候选完整湿周诊断；不训练网络，不读取Q或流速字段。"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import isotonic_regression


ROOT = Path(__file__).resolve().parents[1]
PROFILE_ROOT = ROOT / "research_notes/USGS 原86 历年断面核验"
INDEX = PROFILE_ROOT / "profiles/profile_index.csv"
H_FILE = ROOT / "data/camelsh_exported/waterlevel_hourly_86.parquet"
SPLITS = ROOT / "data/splits/basin_splits.json"
MODELS = ("constant", "linear", "quadratic", "isotonic")
SCOPES = {
    "sampled_mixed": "全部合格剖面的已采样湿润长度，混合位置",
    "sampled_dominant_position": "已采样湿润长度，最多访问的已知位置",
    "endpoint_zero_mixed": "两端零水深候选湿周，混合位置",
    "endpoint_zero_known_positions": "两端零水深候选湿周，已知位置混合",
    "endpoint_zero_dominant_position": "两端零水深候选湿周，最多访问的已知位置",
    "endpoint_zero_dominant_position_method": "两端零水深候选湿周，同位置同方法",
}


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def clean_json(value):
    if isinstance(value, dict):
        return {str(k): clean_json(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean_json(v) for v in value]
    if isinstance(value, np.generic):
        return clean_json(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def write_json(path, value):
    Path(path).write_text(json.dumps(clean_json(value), ensure_ascii=False, indent=2,
                                    allow_nan=False), encoding="utf-8")


def number(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return np.nan


def polyline_length(x, depth):
    """只累加接触水体的河床折线；两个零深度点间的干地不计入湿周。"""
    x, depth = np.asarray(x, float), np.asarray(depth, float)
    wet = (depth[:-1] > 0) | (depth[1:] > 0)
    return float(np.hypot(np.diff(x), np.diff(depth))[wet].sum())


def downsample_length(x, depth, step):
    """保留两端和所有零深度点；降采样敏感性不当作独立精度验证。"""
    keep = np.unique(np.r_[np.arange(0, len(x), step), len(x) - 1,
                           np.flatnonzero(depth == 0)])
    return polyline_length(x[keep], depth[keep])


def pair_hourly_stage(series, when):
    """仅在相邻一小时均有效时插值，不跨缺口、不外推。"""
    left = when.floor("h")
    if left == when:
        value = series.get(left, np.nan)
        return number(value), "exact" if np.isfinite(number(value)) else "missing"
    right = left + pd.Timedelta(hours=1)
    a, b = number(series.get(left, np.nan)), number(series.get(right, np.nan))
    if np.isfinite([a, b]).all():
        fraction = (when - left).total_seconds() / 3600
        return a + fraction * (b - a), "adjacent_hour_interpolation"
    return np.nan, "missing"


def original_period(when, split):
    for period in ("train", "valid", "test"):
        if pd.Timestamp(split[f"{period}_start"]) <= when <= pd.Timestamp(split[f"{period}_end"]):
            return period
    return "other"


def inventory(basins, splits):
    index = pd.read_csv(INDEX, dtype={"station": str})
    index = index[index.station.isin(basins)]
    stage = pd.read_parquet(H_FILE, columns=basins)
    assert stage.index.is_unique and stage.index.is_monotonic_increasing
    rows, hashes = [], {}
    for item in index.itertuples(index=False):
        path = PROFILE_ROOT / item.json_path
        payload = json.loads(path.read_text(encoding="utf-8-sig"))
        hashes[item.json_path] = sha256(path)
        # 明确白名单；不访问NWIS_Discharge_cfs及任何流速值。
        fields = ("MeasurementQualityCode", "CrossSectionType", "DeploymentMethodCode",
                  "SectionLocationCode", "SectionDistanceMeasure_ft", "ChannelNumber",
                  "ChannelMaterialCode", "NXSDB_WettedDistance_ft", "NWIS_GageHeight_ft")
        record = {key: payload["record"].get(key) for key in fields}
        units = payload.get("units", {})
        assert units["Distance_ft"] == units["Depth_ft"] == "ft"
        assert units["NWIS_GageHeight_ft"] == units["NXSDB_WettedDistance_ft"] == "ft"
        points = np.array([[number(p.get("Distance_ft")), number(p.get("Depth_ft"))]
                           for p in payload["points"]], float)
        valid = np.isfinite(points).all(axis=1) & (points[:, 1] >= 0)
        points = points[valid]
        grouped = pd.DataFrame(points, columns=["x", "d"]).groupby("x").d
        conflict = int((grouped.nunique() > 1).sum())
        profile = grouped.median().sort_index()
        x, depth = profile.index.to_numpy(float) * .3048, profile.to_numpy(float) * .3048
        differences = np.diff(points[:, 0])
        backtracking = bool((differences > 0).any() and (differences < 0).any())
        geometry_valid = (len(x) >= 3 and bool(valid.all()) and not conflict and not backtracking
                          and bool((depth > 0).any()))
        length = polyline_length(x, depth) if len(x) >= 2 else np.nan
        endpoint_zero = bool(len(x) >= 2 and depth[0] == 0 and depth[-1] == 0)
        quality = str(record["MeasurementQualityCode"]).upper()
        ice = "ICE" in " ".join(str(record.get(k, "")) for k in
                                  ("ChannelMaterialCode", "DeploymentMethodCode", "CrossSectionType")).upper()
        accepted = geometry_valid and quality in ("EXCL", "GOOD", "FAIR") and not ice
        when = pd.Timestamp(payload["measurement_timestamp_utc"]).tz_convert("UTC").tz_localize(None)
        basin = str(payload["station"])
        h_local, pair_method = pair_hourly_stage(stage[basin], when)
        local_day = str(payload["measurement_date_local"])
        location = str(record["SectionLocationCode"] or "").upper()
        distance = number(record["SectionDistanceMeasure_ft"]) * .3048
        if location == "ATGA" and not np.isfinite(distance):
            distance = 0.0
        known = location in ("ATGA", "UPST", "DNST") and np.isfinite(distance) and distance >= 0
        channel = number(record["ChannelNumber"])
        position = f"channel{channel:g}|{location}|{distance:.6f}" if known else ""
        official = number(record["NXSDB_WettedDistance_ft"]) * .3048
        row = {"basin": basin, "record_id": str(payload["mmt_ID"]), "json_path": item.json_path,
               "source_sha256": hashes[item.json_path], "water_year": int(payload["water_year"]),
               "measurement_time": when, "measurement_date_local": local_day,
               "visit_key": f"{basin}|{local_day}", "quality": quality, "ice_flag": ice,
               "geometry_valid": geometry_valid, "accepted": accepted,
               "n_points": len(valid), "n_unique_x": len(x), "n_invalid_points": int((~valid).sum()),
               "n_conflicting_x": conflict, "backtracking_x": backtracking,
               "endpoint_zero": endpoint_zero,
               "left_endpoint_depth_m": depth[0] if len(x) else np.nan,
               "right_endpoint_depth_m": depth[-1] if len(x) else np.nan,
               "sampled_length_m": length, "official_sampled_length_m": official,
               "official_length_relative_difference": (length / official - 1) if official > 0 else np.nan,
               "span_m": x[-1] - x[0] if len(x) >= 2 else np.nan,
               "downsample2_relative_change": downsample_length(x, depth, 2) / length - 1 if length > 0 else np.nan,
               "downsample4_relative_change": downsample_length(x, depth, 4) / length - 1 if length > 0 else np.nan,
               "h_local_m": h_local, "h_pair_method": pair_method,
               "h_official_m": number(record["NWIS_GageHeight_ft"]) * .3048,
               "model_period": original_period(when, splits[basin]),
               "position_known": known, "position_key": position, "channel_number": channel,
               "location_code": location, "distance_m": distance,
               "method_key": f"{record['CrossSectionType']}|{record['DeploymentMethodCode']}"}
        rows.append(row)
    frame = pd.DataFrame(rows)
    assert frame.record_id.is_unique
    return frame, hashes


def weights(frame):
    return 1 / frame.groupby("visit_key").visit_key.transform("size").to_numpy(float)


def fit_models(frame):
    if frame.empty:
        return {}
    x, y, w = frame.h_m.to_numpy(), frame.sampled_length_m.to_numpy(), weights(frame)
    unique = np.unique(x).size
    models = {}
    center = float(np.average(x, weights=w))
    scale = max(float(np.sqrt(np.average((x - center) ** 2, weights=w))), 1e-8)
    for name, degree in (("constant", 0), ("linear", 1), ("quadratic", 2)):
        if unique <= degree:
            continue
        design = np.vander((x - center) / scale, N=degree + 1, increasing=True)
        coef = np.linalg.lstsq(design * np.sqrt(w)[:, None], y * np.sqrt(w), rcond=None)[0]
        model = {"model": name, "center_h_m": center, "scale_h_m": scale,
                 "coefficients_scaled_m": coef.tolist(), "h_min_m": float(x.min()), "h_max_m": float(x.max())}
        if name == "linear":
            model.update(intercept_m=float(coef[0] - coef[1] * center / scale),
                         slope_dimensionless=float(coef[1] / scale))
        models[name] = model
    if unique >= 2:
        levels, inverse = np.unique(x, return_inverse=True)
        mass = np.bincount(inverse, weights=w)
        target = np.bincount(inverse, weights=w * y) / mass
        fitted = isotonic_regression(target, weights=mass).x
        models["isotonic"] = {"model": "isotonic", "h_nodes_m": levels.tolist(),
                               "length_nodes_m": fitted.tolist(),
                               "h_min_m": float(x.min()), "h_max_m": float(x.max())}
    return models


def predict(model, x):
    x = np.asarray(x, float)
    if model["model"] == "isotonic":
        return np.interp(x, model["h_nodes_m"], model["length_nodes_m"])
    return np.polynomial.polynomial.polyval((x - model["center_h_m"]) / model["scale_h_m"],
                                            model["coefficients_scaled_m"])


def scores(y, prediction, w):
    y, prediction, w = map(lambda a: np.asarray(a, float), (y, prediction, w))
    mean = np.average(y, weights=w)
    error = prediction - y
    variance = np.sum(w * (y - mean) ** 2)
    mae = float(np.average(np.abs(error), weights=w))
    return {"r2": float(1 - np.sum(w * error ** 2) / variance) if variance > 1e-15 else np.nan,
            "mae_m": mae, "rmse_m": float(np.sqrt(np.average(error ** 2, weights=w))),
            "relative_mae": mae / mean if mean > 0 else np.nan}


def dominant(frame, key):
    if frame.empty:
        return frame
    ranks = frame.groupby(key).agg(n=("visit_key", "nunique"), records=("record_id", "size"))
    best = ranks.sort_values(["n", "records"], ascending=False, kind="stable").index[0]
    return frame[frame[key].eq(best)].copy()


def scope_frames(frame):
    known = frame[frame.position_known & frame.channel_number.eq(1)]
    complete = frame[frame.endpoint_zero]
    complete_known = known[known.endpoint_zero]
    main = dominant(complete_known, "position_key")
    return {"sampled_mixed": frame, "sampled_dominant_position": dominant(known, "position_key"),
            "endpoint_zero_mixed": complete, "endpoint_zero_known_positions": complete_known,
            "endpoint_zero_dominant_position": main,
            "endpoint_zero_dominant_position_method": dominant(main, "method_key")}


def validate(frame, basin, source, scope):
    """所有留出均按测量日分组；同日多条测线绝不跨拟合与验证部分。"""
    frame = frame.sort_values(["measurement_time", "record_id"]).reset_index(drop=True)
    metadata = {"basin": basin, "h_source": source, "scope": scope, "n_profiles": len(frame),
                "n_visits": frame.visit_key.nunique(), "n_unique_h": frame.h_m.nunique(),
                "n_positions": frame.loc[frame.position_known, "position_key"].nunique(),
                "h_min_m": frame.h_m.min(), "h_max_m": frame.h_m.max()}
    models = fit_models(frame)
    saved = {**metadata, "models": models}
    if "linear" in models:
        saved.update(intercept_m=models["linear"]["intercept_m"],
                     slope_dimensionless=models["linear"]["slope_dimensionless"])
    splits = {"fit": [(np.ones(len(frame), bool), np.ones(len(frame), bool), "fit")]}
    if metadata["n_visits"] >= 3:
        splits["leave_one_visit_out"] = []
        for visit in sorted(frame.visit_key.unique()):
            test = frame.visit_key.eq(visit).to_numpy()
            splits["leave_one_visit_out"].append((~test, test, visit))
    if metadata["n_visits"] >= 5:
        order = frame.groupby("visit_key").measurement_time.min().sort_values(kind="stable").index
        n_test = max(2, math.ceil(len(order) * .2))
        test = frame.visit_key.isin(order[-n_test:]).to_numpy()
        cutoff = frame.loc[test, "measurement_time"].min()
        early = frame.groupby("visit_key").measurement_time.max()
        train = frame.visit_key.isin(early[early < cutoff].index).to_numpy()
        if frame.loc[train, "visit_key"].nunique() >= 3:
            splits["time_holdout"] = [(train, test, "last_20pct_visits")]
    if scope == "endpoint_zero_known_positions" and metadata["n_positions"] >= 2:
        splits["leave_one_position_out"] = []
        for position in sorted(frame.position_key.unique()):
            test = frame.position_key.eq(position).to_numpy()
            test_visits = frame.loc[test, "visit_key"]
            train = (~test) & (~frame.visit_key.isin(test_visits).to_numpy())
            if frame.loc[train, "visit_key"].nunique() >= 3:
                splits["leave_one_position_out"].append((train, test, position))
    rows, predictions = [], []
    for scheme, folds in splits.items():
        held = []
        for train, test, fold in folds:
            if scheme != "fit":
                assert set(frame.loc[train, "visit_key"]).isdisjoint(frame.loc[test, "visit_key"])
            if scheme == "time_holdout":
                assert frame.loc[train, "measurement_time"].max() < frame.loc[test, "measurement_time"].min()
            if scheme == "leave_one_position_out":
                assert set(frame.loc[train, "position_key"]).isdisjoint(frame.loc[test, "position_key"])
            if not test.any() or not train.any():
                continue
            fitted = fit_models(frame.loc[train])
            out = frame.loc[test, ["record_id", "visit_key", "position_key", "measurement_time",
                                   "h_m", "sampled_length_m"]].copy()
            lo, hi = frame.loc[train, "h_m"].agg(["min", "max"])
            out["in_range"] = out.h_m.between(lo, hi)
            out["fold"] = fold
            for name in MODELS:
                out[name] = predict(fitted[name], out.h_m) if name in fitted else np.nan
            held.append(out)
        if not held:
            continue
        held = pd.concat(held, ignore_index=True)
        assert not held.record_id.duplicated().any()
        held = held.assign(basin=basin, h_source=source, scope=scope, scheme=scheme)
        if scheme != "fit":
            predictions.append(held)
        expected = int(sum(test.sum() for _, test, _ in folds))
        for name in MODELS:
            for area in ("all", "in_range"):
                part = held[np.isfinite(held[name]) & (held.in_range if area == "in_range" else True)]
                if part.empty:
                    continue
                w = weights(part)
                score = scores(part.sampled_length_m, part[name], w)
                baseline = scores(part.sampled_length_m, part.constant, w)
                rows.append({**metadata, "scheme": scheme, "range": area, "model": name,
                             "n_predictions": len(part), "n_validation_visits": part.visit_key.nunique(),
                             "coverage": len(part) / len(frame), "expected_validation_profiles": expected,
                             "prediction_coverage": len(part) / expected,
                             "in_range_fraction": float(held.in_range.mean()), **score,
                             "constant_mae_m": baseline["mae_m"],
                             "mae_improvement_vs_constant": 1 - score["mae_m"] / baseline["mae_m"]
                             if baseline["mae_m"] > 1e-12 else np.nan,
                             "negative_prediction_fraction": float((part[name] < 0).mean())})
    return saved, rows, predictions


def self_check():
    """用可手算断面和精确直线检查长度与回归，防止把水面或干地计入湿周。"""
    assert np.isclose(polyline_length([0, 3, 6], [0, 4, 0]), 10)
    assert np.isclose(polyline_length([0, 1, 2, 3, 4], [0, 1, 0, 0, 0]), 2 * np.sqrt(2))
    frame = pd.DataFrame({"h_m": np.arange(6.), "sampled_length_m": 10 + 2 * np.arange(6.),
                          "visit_key": list("abcdef")})
    model = fit_models(frame)["linear"]
    assert np.isclose(model["slope_dimensionless"], 2) and np.isclose(model["intercept_m"], 10)
    assert np.allclose(predict(model, [0, 1, 5]), [10, 12, 20])


def plot_station(basin, frame, fitted, table, destination):
    plt.rcParams.update({"font.family": ["Microsoft YaHei", "DejaVu Sans"], "axes.unicode_minus": False})
    fig, axes = plt.subplots(1, 2, figsize=(11.4, 4.1), constrained_layout=True)
    groups = scope_frames(frame)
    for ax, scope, title in zip(axes, ("sampled_mixed", "endpoint_zero_mixed"),
                                ("已采样湿润长度（可含缺岸边剖面）", "两端零水深的候选湿周")):
        group = groups[scope]
        if group.empty:
            ax.text(.5, .5, "没有可配对本地水位的合格剖面", ha="center", transform=ax.transAxes)
        else:
            if scope == "sampled_mixed":
                for zero, label, color in ((False, "端点仍有水深", "#b6b6b6"), (True, "两端零水深", "#2176ae")):
                    part = group[group.endpoint_zero.eq(zero)]
                    ax.scatter(part.h_m, part.sampled_length_m, s=22, color=color, label=label)
            else:
                ax.scatter(group.h_m, group.sampled_length_m, s=23, color="#2176ae", label="独立实测剖面")
            models = fitted.get((basin, "local", scope), {}).get("models", {})
            if len(group) and "linear" in models:
                grid = np.linspace(group.h_m.min(), group.h_m.max(), 150)
                ax.plot(grid, predict(models["linear"], grid), color="#e17c05", label="混合位置直线")
            specific = "sampled_dominant_position" if scope == "sampled_mixed" else "endpoint_zero_dominant_position"
            dominant_group = groups[specific]
            dm = fitted.get((basin, "local", specific), {}).get("models", {})
            if "linear" in dm:
                grid = np.linspace(dominant_group.h_m.min(), dominant_group.h_m.max(), 150)
                ax.scatter(dominant_group.h_m, dominant_group.sampled_length_m, marker="s", s=17,
                           facecolors="none", edgecolors="#32936f", label="最多访问位置的测点")
                ax.plot(grid, predict(dm["linear"], grid), "--", color="#32936f", label="最多访问位置的直线")
            selected = table[(table.basin == basin) & table.h_source.eq("local") & table.scope.eq(scope)
                             & table.scheme.eq("leave_one_visit_out") & table.range.eq("all") & table.model.eq("linear")]
            note = f"{len(group)}份剖面 / {group.visit_key.nunique()}个测量日"
            if not selected.empty:
                metric = selected.iloc[0]
                note += f"\n留出R²={metric.r2:.3f}，相对MAE={metric.relative_mae:.1%}"
            ax.text(.03, .97, note, va="top", transform=ax.transAxes, fontsize=9)
            ax.legend(loc="lower right", fontsize=8)
        ax.set(title=title, xlabel="同期本地测站水位 H（m）", ylabel="折线湿润长度（m）")
        ax.grid(alpha=.2)
    fig.suptitle(f"{basin}：2021—2023年剖面几何诊断；两端零水深仍需岸边完整性核查", fontsize=11)
    fig.savefig(destination, dpi=150)
    plt.close(fig)


def plot_examples(raw, fitted, table, destination):
    """同时展示同位置候选、混合位置较好及一般/较差案例。"""
    plt.rcParams.update({"font.family": ["Microsoft YaHei", "DejaVu Sans"], "axes.unicode_minus": False})
    eligible = table[table.h_source.eq("local") & table.scope.eq("endpoint_zero_mixed")
                     & table.scheme.eq("leave_one_visit_out") & table.range.eq("all")
                     & table.model.eq("linear") & table.n_visits.ge(8) & table.prediction_coverage.ge(.8)]
    choices = [("01031510", "endpoint_zero_dominant_position"),
               ("01054300", "endpoint_zero_mixed"), ("01073000", "endpoint_zero_mixed")]
    other = eligible[~eligible.basin.isin([x[0] for x in choices])].sort_values("r2")
    if len(other):
        for index in (0, len(other) // 2, len(other) - 1):
            item = (other.iloc[index].basin, "endpoint_zero_mixed")
            if item not in choices:
                choices.append(item)
    fig, axes = plt.subplots(2, 3, figsize=(12, 7), constrained_layout=True)
    for ax, (basin, scope) in zip(axes.flat, choices):
        group = raw[raw.basin.eq(basin) & raw.accepted & np.isfinite(raw.h_local_m)].copy()
        group["h_m"] = group.h_local_m
        group = scope_frames(group)[scope]
        ax.scatter(group.h_m, group.sampled_length_m, color="#346b8f", s=22)
        curves = fitted[(basin, "local", scope)]["models"]
        grid = np.linspace(group.h_m.min(), group.h_m.max(), 160)
        for name, label, color in (("linear", "直线", "#d77b13"), ("quadratic", "二次曲线", "#31866c"),
                                    ("isotonic", "单调插值", "#9561a5")):
            if name in curves:
                ax.plot(grid, predict(curves[name], grid), color=color, label=label, linewidth=1.4)
        selected = table[(table.basin == basin) & table.h_source.eq("local") & table.scope.eq(scope)
                         & table.scheme.eq("leave_one_visit_out") & table.range.eq("all") & table.model.eq("linear")].iloc[0]
        ax.set_title(f"{basin}  {'同位置' if 'dominant' in scope else '混合位置'}\n"
                     f"{group.visit_key.nunique()}测量日；直线留出R²={selected.r2:.3f}", fontsize=10)
        ax.set(xlabel="水位 H（m）", ylabel="候选湿周（m）")
        ax.grid(alpha=.2)
    for ax in list(axes.flat)[len(choices):]:
        ax.set_visible(False)
    axes.flat[0].legend(fontsize=8)
    fig.suptitle("水位—候选湿周：图中为全资料拟合曲线，R²来自独立测量日留出", fontsize=12)
    fig.savefig(destination, dpi=170)
    plt.close(fig)


def report(output, raw, models, table, predictions, basins, source_hashes):
    accepted = raw[raw.accepted]
    paired = accepted[np.isfinite(accepted.h_local_m)]
    zero = paired[paired.endpoint_zero]
    summary = {"n_basins": len(basins), "n_profiles": len(raw), "n_stations_with_profiles": raw.basin.nunique(),
               "n_accepted_profiles": len(accepted), "n_accepted_stations": accepted.basin.nunique(),
               "n_local_h_pairs": len(paired), "n_local_paired_stations": paired.basin.nunique(),
               "n_endpoint_zero_profiles": len(zero), "n_endpoint_zero_stations": zero.basin.nunique(),
               "n_visits_local": paired.visit_key.nunique(), "n_visits_endpoint_zero": zero.visit_key.nunique(),
               "date_start": str(raw.measurement_time.min()), "date_end": str(raw.measurement_time.max()),
               "accepted_model_period_counts": accepted.model_period.value_counts().to_dict(),
               "paired_model_period_counts": paired.model_period.value_counts().to_dict(),
               "geometry_exclusions": {"invalid_geometry": int((~raw.geometry_valid).sum()),
                                       "unaccepted_quality": int((~raw.quality.isin(["EXCL", "GOOD", "FAIR"])).sum()),
                                       "ice": int(raw.ice_flag.sum())}, "scopes": {}}
    cross = paired[np.isfinite(paired.h_official_m)]
    difference = cross.h_official_m - cross.h_local_m
    summary["official_vs_local_h"] = {"n": len(cross), "median_absolute_difference_m": difference.abs().median(),
                                        "p95_absolute_difference_m": difference.abs().quantile(.95),
                                        "n_abs_difference_over_0_1m": int((difference.abs() > .1).sum())}
    summary["length_check"] = {"official_relative_abs_median": accepted.official_length_relative_difference.abs().median(),
                                "official_relative_abs_p95": accepted.official_length_relative_difference.abs().quantile(.95),
                                "downsample2_relative_abs_median": accepted.downsample2_relative_change.abs().median(),
                                "downsample4_relative_abs_p95": accepted.downsample4_relative_change.abs().quantile(.95)}
    station_rows = []
    for basin in basins:
        a, b = raw[raw.basin.eq(basin)], zero[zero.basin.eq(basin)]
        row = {"basin": basin, "n_raw_profiles": len(a), "n_accepted_profiles": int(a.accepted.sum()),
               "n_local_h_pairs": int((a.accepted & np.isfinite(a.h_local_m)).sum()),
               "n_endpoint_zero_profiles": len(b), "n_endpoint_zero_visits": b.visit_key.nunique()}
        for scope in SCOPES:
            model = models.get((basin, "local", scope), {})
            row[scope + "_n_visits"] = model.get("n_visits", 0)
            for scheme, short in (("leave_one_visit_out", "lovo"), ("time_holdout", "time"),
                                  ("leave_one_position_out", "position")):
                part = table[(table.basin == basin) & table.h_source.eq("local") & table.scope.eq(scope)
                             & table.scheme.eq(scheme) & table.range.eq("all") & table.model.eq("linear")]
                for name in ("r2", "mae_m", "relative_mae", "mae_improvement_vs_constant", "prediction_coverage"):
                    row[f"{scope}_{short}_{name}"] = part.iloc[0][name] if len(part) else np.nan
        station_rows.append(row)
    stations = pd.DataFrame(station_rows)
    stations.to_csv(output / "all86_station_summary.csv", index=False, encoding="utf-8-sig")
    lines = ["# 原86站水位—湿周及已采样湿润长度拟合", "",
             "本轮只做几何诊断，不训练或修改两阶段网络。不读取流量、流速或评级曲线字段。", "",
             f"逐点档案共{len(raw):,}份、覆盖{raw.basin.nunique()}站；质量和点几何筛查后{len(accepted):,}份、{accepted.basin.nunique()}站。",
             f"可配对本地小时水位{len(paired):,}份、{paired.basin.nunique()}站；其中两端零水深{len(zero):,}份、{zero.basin.nunique()}站。", "",
             "两端零水深仅是岸边完整性的候选筛查，不能证明全部岸壁、测线、垂直基准均已核实。端点仍有水深的记录只分析已采样湿润长度S(H)，不称完整湿周P(H)。", "",
             "## 口径与验证", "",
             "- 使用所有现有2021—2023年剖面作事后几何诊断；按原冻结划分另标测量所属时段，不能称严格训练期内拟合。",
             "- 主输入为同站本地小时水位：整点直接匹配，非整点仅在相邻有效小时之间插值，不跨缺口。另用档案测量水位独立复算，绝不混填两种H。",
             "- 相同横距取深度中位数，但存在冲突深度、横距回折、无效点的剖面不进入主拟合；不补造到岸边的线段。两端零水深阈值为精确零。",
             "- 每个本地测量日期作为一组，同日多条测线不能跨拟合和留出；每个测量日总权重为1。",
             "- 同位置仅按channel=1、上下游方向及距离一致识别；不是已核实的真实固定测线。最多访问位置按访问数量选择，不按拟合分数选择。",
             "- 比较常数、直线、二次曲线、等序回归插值；不预设非线性一定更好，原始预测不截断以保留失效证据。",
             "- 至少3个测量日尝试逐日留出；至少5日尝试末20%时间留出（至少2个验证日且至少3个训练日）；小样本结果单列。整位置留出同时剔除重叠测量日。",
             "- 主汇总要求至少8个独立测量日，逐日留出预测覆盖率≥80%。8日、R²≥0.8、相对MAE≤5%均为探索性门槛，并非USGS标准。",
             "- 相对MAE=MAE/验证湿周加权均值，同时报告相对常数基线的MAE改善。低R²可能来自湿周变化很小，应结合误差解释。",
             "- 不从单份剖面合成多个水位点再拆分验证。降采样只检查分辨率敏感性，不代表独立精度。", "",
             "## 本地水位拟合结果（直线，逐测量日留出，含区间外点）", "",
             "| 数据口径 | 有数据站 | ≥8测量日且可验证 | R²≥0.8 | R²中位数 | 相对MAE中位数 | 较常数MAE改善中位数 |",
             "| --- | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for source in ("local", "official"):
        for scope, label in SCOPES.items():
            part = table[table.h_source.eq(source) & table.scope.eq(scope) & table.scheme.eq("leave_one_visit_out")
                         & table.range.eq("all") & table.model.eq("linear")]
            strong = part[part.n_visits.ge(8) & part.prediction_coverage.ge(.8)]
            times = table[table.h_source.eq(source) & table.scope.eq(scope) & table.scheme.eq("time_holdout")
                          & table.range.eq("all") & table.model.eq("linear") & table.prediction_coverage.ge(.8)]
            both = set(strong.loc[strong.r2.ge(.8), "basin"]) & set(times.loc[times.r2.ge(.8), "basin"])
            info = {"stations_with_data": sum(int(m["n_profiles"] > 0) for k, m in models.items() if k[1:] == (source, scope)),
                    "n_validated_ge8_visits": len(strong), "n_lovo_r2_ge08": int(strong.r2.ge(.8).sum()),
                    "median_r2": strong.r2.median(), "median_relative_mae": strong.relative_mae.median(),
                    "median_mae_improvement_vs_constant": strong.mae_improvement_vs_constant.median(),
                    "n_lovo_relative_mae_le05": int(strong.relative_mae.le(.05).sum()),
                    "n_lovo_and_time_r2_ge08": len(both), "lovo_and_time_r2_ge08_stations": sorted(both)}
            summary["scopes"][source + ":" + scope] = info
            if source == "local":
                fmt = lambda value: f"{value:.3f}" if pd.notna(value) else "无"
                pct = lambda value: f"{value:.1%}" if pd.notna(value) else "无"
                lines.append(f"| {label} | {info['stations_with_data']} | {len(strong)} | {info['n_lovo_r2_ge08']} | {fmt(info['median_r2'])} | {pct(info['median_relative_mae'])} | {pct(info['median_mae_improvement_vs_constant'])} |")
    lines.extend(["", "## 样本数量门槛的敏感性", "",
                  "同位置候选完整湿周的数据本身较少，不能把因样本不足未入选的站描述为拟合失败。", "",
                  "| 最少独立测量日 | 混合位置可验证站 | 同位置可验证站 |",
                  "| ---: | ---: | ---: |"])
    summary["visit_threshold_sensitivity"] = {}
    for minimum in (3, 5, 8, 10, 15):
        counts = {}
        for scope in ("endpoint_zero_mixed", "endpoint_zero_dominant_position"):
            part = table[table.h_source.eq("local") & table.scope.eq(scope) & table.scheme.eq("leave_one_visit_out")
                         & table.range.eq("all") & table.model.eq("linear") & table.n_visits.ge(minimum)
                         & table.prediction_coverage.ge(.8)]
            counts[scope] = len(part)
        summary["visit_threshold_sensitivity"][minimum] = counts
        lines.append(f"| {minimum} | {counts['endpoint_zero_mixed']} | {counts['endpoint_zero_dominant_position']} |")
    lines.extend(["", "## 非线性比较", "",
                  "下表在同一批至少8个测量日的混合位置候选湿周站上比较，全部使用独立留出预测。", "",
                  "| 模型 | 站数 | 留出R²中位数 | 相对MAE中位数 |",
                  "| --- | ---: | ---: | ---: |"])
    summary["nonlinear_comparison"] = {}
    for name in MODELS:
        part = table[table.h_source.eq("local") & table.scope.eq("endpoint_zero_mixed")
                     & table.scheme.eq("leave_one_visit_out") & table.range.eq("all")
                     & table.model.eq(name) & table.n_visits.ge(8) & table.prediction_coverage.ge(.8)]
        info = {"n_stations": len(part), "median_r2": part.r2.median(), "median_relative_mae": part.relative_mae.median()}
        summary["nonlinear_comparison"][name] = info
        lines.append(f"| {name} | {len(part)} | {info['median_r2']:.3f} | {info['median_relative_mae']:.1%} |")
    lines.extend(["", "## 直线留出R²较高的候选", "",
                  "仅列至少8个测量日、逐日留出R²≥0.8的候选；混合位置和同位置必须分开解释。", "",
                  "| 站点 | 口径 | 测量日 | 逐日留出R² | 相对MAE | 时间留出R² | 时间验证测量日 |",
                  "| --- | --- | ---: | ---: | ---: | ---: | ---: |"])
    candidate = table[table.h_source.eq("local") & table.scope.isin(["endpoint_zero_mixed", "endpoint_zero_dominant_position"])
                      & table.scheme.eq("leave_one_visit_out") & table.range.eq("all") & table.model.eq("linear")
                      & table.n_visits.ge(8) & table.prediction_coverage.ge(.8) & table.r2.ge(.8)]
    summary["linear_candidates"] = []
    for row in candidate.itertuples(index=False):
        time = table[(table.basin == row.basin) & table.h_source.eq("local") & table.scope.eq(row.scope)
                     & table.scheme.eq("time_holdout") & table.range.eq("all") & table.model.eq("linear")]
        t = time.iloc[0] if len(time) else None
        info = {"basin": row.basin, "scope": row.scope, "n_visits": row.n_visits, "lovo_r2": row.r2,
                "lovo_relative_mae": row.relative_mae, "time_r2": t.r2 if t is not None else np.nan,
                "time_validation_visits": int(t.n_validation_visits) if t is not None else 0}
        summary["linear_candidates"].append(info)
        lines.append(f"| {row.basin} | {'同位置' if 'dominant' in row.scope else '混合位置'} | {row.n_visits} | {row.r2:.3f} | {row.relative_mae:.1%} | {info['time_r2']:.3f} | {info['time_validation_visits']} |")
    one = models.get(("01031510", "local", "endpoint_zero_dominant_position"), {})
    if "linear" in one.get("models", {}):
        curve = one["models"]["linear"]
        lines.extend(["", f"01031510同位置直线为P(H)={curve['intercept_m']:.6f}+{curve['slope_dimensionless']:.6f}H（P、H均为m），已测H范围{curve['h_min_m']:.6f}—{curve['h_max_m']:.6f}m。",
                      "该站二次/单调模型在本次验证中也有改善，但只有11个测量日，时间验证仅3日，尚不足以确认可稳定推广。"])
    lines.extend(["", "## 时段、水位及分辨率检查", "",
                  f"合格且配对本地H的测量所属原划分：{summary['paired_model_period_counts']}。",
                  f"档案H与本地同期H的绝对差中位数{difference.abs().median():.4f}m，95%分位{difference.abs().quantile(.95):.4f}m；该差异不能自动解释为统一基准偏移。",
                  f"折线长度与档案WettedDistance相对绝对差中位数{summary['length_check']['official_relative_abs_median']:.3%}，95%分位{summary['length_check']['official_relative_abs_p95']:.3%}。该比对不是独立观测验证。",
                  f"每4点取1点且保留端点/零深度点，长度相对变化绝对值95%分位{summary['length_check']['downsample4_relative_abs_p95']:.3%}。", "",
                  "## 完整输出", "",
                  "profile_inventory.csv保留全部1230份剖面的筛查原因、端点、水位配对与原实验时段；all86_station_summary.csv逐站保留各类验证指标。",
                  "validation_metrics.csv包含两种H、所有数据口径、四种模型、范围内/全部留出指标；heldout_predictions.csv.gz可独立重算指标。",
                  "fitted_curves.json保存直线系数、非线性节点及有效H范围，仅供诊断，尚未接入流量模型。station_plots/为全部86站曲线。", "",
                  "![湿周关系示例](perimeter_examples.png)", "",
                  "示例图曲线使用全资料拟合以展示形状，标题R²来自独立留出预测，两者不可混作同一评价。", "",
                  "同位置混合测量方法、岸边遗漏、河床变化及站点H与测线水面的对应关系仍可能影响结果，不能仅凭高R²声明获得物理上完整可靠的P(H)。", "",
                  "复现：`python -X utf8 tools/fit_cross_section86_perimeter_h_curves.py`。"])
    write_json(output / "summary.json", summary)
    write_json(output / "source_manifest.json", {"profile_index_sha256": sha256(INDEX),
               "frozen_splits_sha256": sha256(SPLITS), "local_h_sha256": sha256(H_FILE),
               "script_sha256": sha256(__file__), "profiles": source_hashes,
               "read_discharge_velocity_or_ratings": False, "grouping": "basin_and_local_measurement_date"})
    (output / "review.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return summary, stations


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "reports/cross_section86_perimeter_h_curves")
    parser.add_argument("--skip-plots", action="store_true")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    self_check()
    splits = json.loads(SPLITS.read_text(encoding="utf-8-sig"))["splits"]
    basins = list(splits)
    print("读取全部剖面并配对两种水位……", flush=True)
    raw, source_hashes = inventory(basins, splits)
    raw.to_csv(args.output / "profile_inventory.csv", index=False, encoding="utf-8-sig")
    print(f"剖面{len(raw)}份，质量与几何合格{int(raw.accepted.sum())}份，两端零水深{int((raw.accepted & raw.endpoint_zero).sum())}份", flush=True)
    fitted, metric_rows, heldout = {}, [], []
    for source, column in (("local", "h_local_m"), ("official", "h_official_m")):
        usable = raw[raw.accepted & np.isfinite(raw[column])].copy()
        usable["h_m"] = usable[column]
        for i, basin in enumerate(basins):
            for scope, frame in scope_frames(usable[usable.basin.eq(basin)]).items():
                model, rows, predictions = validate(frame, basin, source, scope)
                fitted[(basin, source, scope)] = model
                metric_rows.extend(rows)
                heldout.extend(predictions)
            if (i + 1) % 20 == 0:
                print(f"{source}水位：完成{i + 1}/{len(basins)}站", flush=True)
    metrics = pd.DataFrame(metric_rows)
    predictions = pd.concat(heldout, ignore_index=True)
    metrics.to_csv(args.output / "validation_metrics.csv", index=False, encoding="utf-8-sig")
    predictions.to_csv(args.output / "heldout_predictions.csv.gz", index=False, compression="gzip")
    write_json(args.output / "fitted_curves.json", list(fitted.values()))
    summary, stations = report(args.output, raw, fitted, metrics, predictions, basins, source_hashes)
    if not args.skip_plots:
        plots = args.output / "station_plots"
        plots.mkdir(exist_ok=True)
        usable = raw[raw.accepted & np.isfinite(raw.h_local_m)].copy()
        usable["h_m"] = usable.h_local_m
        for i, basin in enumerate(basins):
            plot_station(basin, usable[usable.basin.eq(basin)], fitted, metrics, plots / f"{basin}.png")
            if (i + 1) % 20 == 0:
                print(f"作图完成{i + 1}/{len(basins)}站", flush=True)
        plot_examples(raw, fitted, metrics, args.output / "perimeter_examples.png")
    print(json.dumps(clean_json({k: v for k, v in summary.items() if k != "scopes"}), ensure_ascii=False), flush=True)
    print(f"完成：{args.output / 'review.md'}", flush=True)


if __name__ == "__main__":
    main()
