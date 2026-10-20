"""验证全部86站及混合测量位置的H–A关系，只读几何记录，不读取Q。"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import font_manager

from fit_cross_section86_area_h_curves import (
    ROOT, INPUT, digest, fit_curve, predict_curve, metrics, prepare_data, select_dominant, visit_weights,
)


OUTPUT = ROOT / "reports/cross_section86_area_h_position_validation"
BASE_MODELS = ("constant", "linear", "quadratic", "isotonic")
POSITION_MODELS = ("position_offset", "position_linear")
SCOPES = ("all_train_mixed", "openwater_mixed", "main_known_positions", "dominant_main_position")
LABELS = {"all_train_mixed": "全部候选记录", "openwater_mixed": "去冰期及明确非主河道/复核，位置混合",
          "main_known_positions": "明确主河道且位置已知，位置混合", "dominant_main_position": "最大主河道同位置组"}


def safe_json(value):
    if isinstance(value, dict):
        return {str(key): safe_json(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [safe_json(item) for item in value]
    if isinstance(value, np.generic):
        return safe_json(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def write_json(path, value):
    Path(path).write_text(json.dumps(safe_json(value), ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")


def fit_models(frame, position_models=False):
    """所有拟合和位置偏差估计只使用传入的拟合部分。"""
    if frame.empty:
        return {}
    x, y, w = frame.h_m.to_numpy(), frame.area_m2.to_numpy(), visit_weights(frame)
    unique = np.unique(x).size
    models = {"constant": fit_curve(x, y, w, "constant")}
    for name, required in (("linear", 2), ("quadratic", 3), ("isotonic", 2)):
        if unique >= required:
            models[name] = fit_curve(x, y, w, name)
    if position_models and "linear" in models:
        residual = y - predict_curve(models["linear"], x)
        offsets, separate = {}, {}
        for position, group in frame.groupby("position_key"):
            mask = frame.position_key.eq(position).to_numpy()
            offsets[position] = float(np.average(residual[mask], weights=w[mask]))
            if group.visit_key.nunique() >= 3 and group.h_m.nunique() >= 3:
                separate[position] = fit_curve(group.h_m, group.area_m2, visit_weights(group), "linear")
        # 位置偏移模型是简单残差均值诊断，不把位置效应解释成已识别的因果效应。
        models["position_offset"] = {"model": "position_offset", "pooled": models["linear"], "offsets": offsets}
        models["position_linear"] = {"model": "position_linear", "pooled": models["linear"], "curves": separate}
    return models


def predict_model(model, frame):
    if model["model"] not in POSITION_MODELS:
        return predict_curve(model, frame.h_m.to_numpy())
    result = predict_curve(model["pooled"], frame.h_m.to_numpy())
    if model["model"] == "position_offset":
        return result + frame.position_key.map(model["offsets"]).fillna(0).to_numpy()
    for position, curve in model["curves"].items():
        mask = frame.position_key.eq(position).to_numpy()
        result[mask] = predict_curve(curve, frame.loc[mask, "h_m"])
    return result


def validate(frame, basin, scope):
    """逐访问留出、原训练期内部时间留出和整位置留出均不接触Q。"""
    frame = frame.sort_values(["measurement_time", "visit_key", "record_id"]).reset_index(drop=True)
    names = BASE_MODELS + (POSITION_MODELS if scope == "main_known_positions" else ())
    n = len(frame)
    weights = visit_weights(frame) if n else np.array([])
    common = {"basin": basin, "scope": scope, "n_records": n, "n_visits": frame.visit_key.nunique(),
              "n_unique_h": frame.h_m.nunique(), "n_positions": frame.position_key.nunique()}
    common.update(h_min_m=frame.h_m.min(), h_max_m=frame.h_m.max())
    all_models = fit_models(frame, scope == "main_known_positions")
    schemes = {"leave_one_visit_out": [], "time_holdout": []}
    if scope == "main_known_positions":
        schemes["leave_one_position_out"] = []
    if common["n_visits"] >= 3:
        for visit in sorted(frame.visit_key.unique()):
            test = frame.visit_key.eq(visit).to_numpy()
            schemes["leave_one_visit_out"].append((~test, test))
    if common["n_visits"] >= 5:
        ordered = frame.groupby("visit_key").measurement_time.min().sort_values(kind="stable").index.tolist()
        n_test = max(5 if len(ordered) >= 15 else 2, math.ceil(len(ordered) * .2))
        test = frame.visit_key.isin(set(ordered[-n_test:])).to_numpy()
        # 若现场访问记录的时间区间交叠，剔除交叠的早期访问，保证真实时间顺序。
        cutoff = frame.loc[test, "measurement_time"].min()
        eligible_visits = frame.groupby("visit_key").measurement_time.max()
        eligible_visits = set(eligible_visits[eligible_visits <= cutoff].index) - set(ordered[-n_test:])
        train = frame.visit_key.isin(eligible_visits).to_numpy()
        if frame.loc[train, "visit_key"].nunique() >= 3:
            assert frame.loc[train, "measurement_time"].max() <= frame.loc[test, "measurement_time"].min()
            schemes["time_holdout"].append((train, test))
    if scope == "main_known_positions" and common["n_positions"] >= 2:
        for position in sorted(frame.position_key.unique()):
            test = frame.position_key.eq(position).to_numpy()
            test_visits = set(frame.loc[test, "visit_key"])
            train = ~frame.visit_key.isin(test_visits).to_numpy()
            if frame.loc[train, "visit_key"].nunique() >= 3:
                schemes["leave_one_position_out"].append((train, test))
    rows, predictions, integrity = [], [], []
    for scheme, splits in schemes.items():
        buffers = {name: np.full(n, np.nan) for name in names}
        inside = np.zeros(n, bool)
        seen = np.zeros(n, bool)
        fold_id = np.full(n, -1, int)
        for fold, (train, test) in enumerate(splits):
            training, testing = frame.loc[train], frame.loc[test]
            assert not set(training.visit_key) & set(testing.visit_key)
            if scheme == "leave_one_position_out":
                assert not set(training.position_key) & set(testing.position_key)
            if training.h_m.nunique() < 2:
                continue
            models = fit_models(training, scope == "main_known_positions")
            inside[test] = testing.h_m.between(training.h_m.min(), training.h_m.max()).to_numpy()
            seen[test] = testing.position_key.isin(set(training.position_key)).to_numpy()
            fold_id[test] = fold
            for name, model in models.items():
                buffers[name][test] = predict_model(model, testing)
        for name in names:
            prediction = buffers[name]
            valid = np.isfinite(prediction)
            row = {**common, "scheme": scheme, "model": name, "n_valid_prediction_visits": frame.loc[valid, "visit_key"].nunique(),
                   "prediction_record_coverage": float(valid.mean()) if n else 0,
                   "inside_training_h_fraction": float(inside[valid].mean()) if valid.any() else np.nan,
                   "seen_training_position_fraction": float(seen[valid].mean()) if valid.any() else np.nan,
                   "negative_predictions": int((prediction[valid] < 0).sum())}
            for label, mask in (("all", valid), ("inrange", valid & inside)):
                row.update({label + "_" + key: value for key, value in metrics(
                    frame.area_m2.to_numpy()[mask], prediction[mask], weights[mask]).items()})
            if "linear" in all_models:
                row.update(linear_slope_m=all_models["linear"]["slope_m"], linear_intercept_m2=all_models["linear"]["intercept_m2"])
            if name in all_models:
                row.update({"fit_" + key: value for key, value in metrics(
                    frame.area_m2, predict_model(all_models[name], frame), weights).items()})
            if name != "linear":
                matched = valid & np.isfinite(buffers["linear"])
                reference = metrics(frame.area_m2.to_numpy()[matched], buffers["linear"][matched], weights[matched])
                candidate = metrics(frame.area_m2.to_numpy()[matched], prediction[matched], weights[matched])
                row["paired_mae_reduction_vs_linear"] = 1 - candidate["mae"] / reference["mae"] if reference["mae"] > 1e-12 else np.nan
                inside_match = matched & inside
                reference_in = metrics(frame.area_m2.to_numpy()[inside_match], buffers["linear"][inside_match], weights[inside_match])
                candidate_in = metrics(frame.area_m2.to_numpy()[inside_match], prediction[inside_match], weights[inside_match])
                row["paired_inrange_mae_reduction_vs_linear"] = 1 - candidate_in["mae"] / reference_in["mae"] if reference_in["mae"] > 1e-12 else np.nan
            rows.append(row)
            for index in np.flatnonzero(valid):
                predictions.append({"basin": basin, "scope": scope, "scheme": scheme, "model": name,
                    "record_id": frame.record_id.iloc[index], "visit_key": frame.visit_key.iloc[index],
                    "position_key": frame.position_key.iloc[index], "fold": int(fold_id[index]),
                    "h_m": frame.h_m.iloc[index], "area_m2": frame.area_m2.iloc[index],
                    "prediction_m2": prediction[index], "weight": weights[index],
                    "inside_training_h_range": bool(inside[index]), "position_seen_in_training": bool(seen[index])})
        integrity.append({"basin": basin, "scope": scope, "scheme": scheme, "n_splits": len(splits),
                          "visit_groups_disjoint": True, "position_groups_disjoint": scheme == "leave_one_position_out"})
    return rows, predictions, all_models, integrity


def position_parameters(frame, basin):
    rows, pair = [], {}
    for position, group in frame.groupby("position_key"):
        n_visits = group.visit_key.nunique()
        row = {"basin": basin, "position_key": position, "n_records": len(group), "n_visits": n_visits,
               "h_min_m": group.h_m.min(), "h_max_m": group.h_m.max(), "n_methods": group.method_key.nunique()}
        if group.h_m.nunique() >= 2:
            curve = fit_curve(group.h_m, group.area_m2, visit_weights(group), "linear")
            row.update(intercept_m2=curve["intercept_m2"], slope_m=curve["slope_m"],
                       fit_r2=metrics(group.area_m2, predict_curve(curve, group.h_m), visit_weights(group))["r2"])
            if n_visits >= 5:
                pair[position] = (group, curve)
        rows.append(row)
    # 只按样本量选取最大的两个位置，不按拟合优劣选位置。
    ordered = sorted(pair, key=lambda key: (-pair[key][0].visit_key.nunique(), key))[:2]
    difference = {"basin": basin, "n_positions_at_least_5_visits": len(pair)}
    if len(ordered) == 2:
        (first, first_curve), (second, second_curve) = [pair[key] for key in ordered]
        lo, hi = max(first.h_m.min(), second.h_m.min()), min(first.h_m.max(), second.h_m.max())
        difference.update(position_1=ordered[0], position_2=ordered[1], common_h_low_m=lo, common_h_high_m=hi,
                          has_overlapping_h_range=bool(hi > lo), slope_1_m=first_curve["slope_m"], slope_2_m=second_curve["slope_m"])
        if hi > lo:
            h = (lo + hi) / 2
            a1, a2 = float(predict_curve(first_curve, [h])[0]), float(predict_curve(second_curve, [h])[0])
            difference.update(reference_h_m=h, area_1_m2=a1, area_2_m2=a2,
                              relative_area_difference=abs(a1 - a2) / ((a1 + a2) / 2) if min(a1, a2) > 0 else np.nan)
    return rows, difference


def md(frame):
    lines = ["| " + " | ".join(frame.columns) + " |", "| " + " | ".join(["---"] * len(frame.columns)) + " |"]
    lines += ["| " + " | ".join(map(str, values)) + " |" for values in frame.itertuples(index=False, name=None)]
    return "\n".join(lines)


def get_row(scores, basin, scope, scheme="leave_one_visit_out", model="linear"):
    return scores[(scores.basin == basin) & (scores.scope == scope) & (scores.scheme == scheme) & (scores.model == model)].iloc[0]


def draw_position_plot(ax, frame, models, title):
    if frame.empty:
        ax.text(.5, .5, "无明确主河道已知位置记录", transform=ax.transAxes, ha="center", va="center")
        ax.set_title(title)
        return
    colors = ["#be5a32", "#238c85", "#8560ac", "#af8b22"]
    top = frame.groupby("position_key").visit_key.nunique().sort_values(ascending=False).head(4).index
    ax.scatter(frame.h_m, frame.area_m2, color="#bbbbbb", s=20, alpha=.7, label="其他位置")
    for number, position in enumerate(top):
        group = frame[frame.position_key == position]
        ax.scatter(group.h_m, group.area_m2, color=colors[number], s=27, edgecolors="white", linewidths=.3,
                   label=f"位置{number + 1}，访问{group.visit_key.nunique()}", zorder=3)
        if group.visit_key.nunique() >= 3 and group.h_m.nunique() >= 3:
            curve = fit_curve(group.h_m, group.area_m2, visit_weights(group), "linear")
            grid = np.linspace(group.h_m.min(), group.h_m.max(), 90)
            ax.plot(grid, predict_curve(curve, grid), color=colors[number], linestyle="--", linewidth=1.3)
    if "linear" in models:
        grid = np.linspace(frame.h_m.min(), frame.h_m.max(), 180)
        ax.plot(grid, predict_curve(models["linear"], grid), color="#165da5", linewidth=2.1, label="合并位置的直线")
    ax.set_title(title, fontsize=11)
    ax.set_xlabel("测站水位 H（m）")
    ax.set_ylabel("测量面积 A（m²）")
    ax.grid(alpha=.15)


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    plots = OUTPUT / "station_plots"
    plots.mkdir(exist_ok=True)
    font = Path("C:/Windows/Fonts/msyh.ttc")
    if font.is_file():
        font_manager.fontManager.addfont(str(font))
        plt.rcParams["font.family"] = [font_manager.FontProperties(fname=str(font)).get_name(), "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    data = prepare_data()
    assert len(data) == 6618 and data.basin.nunique() == 86 and data.model_period.eq("train").all()
    old = pd.read_csv(ROOT / "reports/cross_section86_area_h_linearity/station_linearity.csv", dtype={"basin": str})
    previous14 = set(old.loc[(old.scope == "dominant_main_position") & old.cv_eligible, "basin"])
    assert len(previous14) == 14
    rows, prediction_rows, split_checks, parameters, differences = [], [], [], [], []
    frames, fitted = {}, {}
    for number, (basin, group) in enumerate(data.groupby("basin", sort=True), 1):
        dominant, _ = select_dominant(group, "position_key")
        scopes = {"all_train_mixed": group, "openwater_mixed": group[group.openwater_candidate],
                  "main_known_positions": group[group.openwater_candidate & group.explicit_main & group.position_known],
                  "dominant_main_position": dominant}
        for scope, frame in scopes.items():
            output, predictions, models, checks = validate(frame, basin, scope)
            rows.extend(output)
            prediction_rows.extend(predictions)
            split_checks.extend(checks)
            frames[(basin, scope)] = frame
            fitted[(basin, scope)] = models
        coefficients, difference = position_parameters(scopes["main_known_positions"], basin)
        parameters.extend(coefficients)
        differences.append(difference)
        if number % 15 == 0:
            print(f"已完成逐访问及位置留出验证 {number}/86 站", flush=True)
    scores, predictions = pd.DataFrame(rows), pd.DataFrame(prediction_rows)
    station_rows = []
    for basin in sorted(data.basin.unique()):
        record = {"basin": basin, "in_previous_primary14": basin in previous14}
        for scope in SCOPES:
            selected = get_row(scores, basin, scope)
            for key in ("n_records", "n_visits", "n_positions", "all_r2", "all_mae", "inrange_r2", "prediction_record_coverage", "linear_slope_m"):
                record[scope + "_" + key] = selected.get(key, np.nan)
            time = get_row(scores, basin, scope, "time_holdout")
            record[scope + "_time_r2"] = time.all_r2
            record[scope + "_time_mae"] = time.all_mae
            for model in ("quadratic", "isotonic"):
                chosen = get_row(scores, basin, scope, model=model)
                record[scope + "_" + model + "_mae_reduction"] = chosen.get("paired_mae_reduction_vs_linear", np.nan)
        for model in POSITION_MODELS:
            record[model + "_mae_reduction"] = get_row(scores, basin, "main_known_positions", model=model).get("paired_mae_reduction_vs_linear", np.nan)
        position = get_row(scores, basin, "main_known_positions", "leave_one_position_out")
        record["position_holdout_r2"] = position.all_r2
        record["position_holdout_coverage"] = position.prediction_record_coverage
        record["position_holdout_inrange_r2"] = position.inrange_r2
        clean = record["openwater_mixed_all_r2"]
        record["mixed_line_prediction_band"] = "较好：R²≥0.8" if clean >= .8 else "中等：0.5≤R²<0.8" if clean >= .5 else "较弱：0≤R²<0.5" if clean >= 0 else "差：R²<0" if np.isfinite(clean) else "无法验证"
        station_rows.append(record)
    stations = pd.DataFrame(station_rows)
    parameters, differences = pd.DataFrame(parameters), pd.DataFrame(differences)
    scores.to_csv(OUTPUT / "validation_metrics.csv", index=False, encoding="utf-8-sig")
    predictions.to_csv(OUTPUT / "heldout_area_predictions.csv.gz", index=False, encoding="utf-8-sig", compression="gzip")
    stations.to_csv(OUTPUT / "all86_station_validation.csv", index=False, encoding="utf-8-sig")
    parameters.to_csv(OUTPUT / "position_linear_parameters.csv", index=False, encoding="utf-8-sig")
    differences.to_csv(OUTPUT / "position_function_differences.csv", index=False, encoding="utf-8-sig")
    write_json(OUTPUT / "fitted_diagnostic_models.json", [{"basin": basin, "scope": scope, "models": models} for (basin, scope), models in fitted.items()])

    # 从保存的每条留出预测重新计算全部指标；不复用汇总表中的数值。
    n_rechecks = 0
    for keys, group in predictions.groupby(["basin", "scope", "scheme", "model"]):
        row = get_row(scores, keys[0], keys[1], keys[2], keys[3])
        check = metrics(group.area_m2, group.prediction_m2, group.weight)
        assert all(np.isclose(check[name], row["all_" + name], atol=1e-10, equal_nan=True) for name in ("r2", "mae", "rmse"))
        if keys[2] == "leave_one_visit_out":
            assert group.groupby("visit_key").fold.nunique().max() == 1
        if keys[2] == "leave_one_position_out":
            assert not group.position_seen_in_training.any()
        n_rechecks += 1
    assert stations.basin.nunique() == len(stations) == 86
    write_json(OUTPUT / "integrity.json", {"all_train_only": True, "n_recomputed_validation_groups": n_rechecks,
        "all_metric_rechecks_passed": True, "visit_groups_disjoint_for_every_split": True,
        "position_groups_disjoint_for_position_holdout": True, "test_q_or_area_used": False,
        "read_q_or_velocity_values": False, "n_stations": 86, "split_checks": split_checks})

    for number, basin in enumerate(stations.basin, 1):
        fig, axes = plt.subplots(1, 3, figsize=(16, 4.4), layout="constrained")
        for ax, scope, title in zip(axes[:2], ("openwater_mixed", "main_known_positions"), ("去冰期等，仍混合测量位置", "明确主河道的已知位置，按位置着色")):
            draw_position_plot(ax, frames[(basin, scope)], fitted[(basin, scope)], title)
            record = get_row(scores, basin, scope)
            ax.text(.02, .98, f"访问={record.n_visits}；位置={record.n_positions}\n留一次访问 R²={record.all_r2:.3f}",
                    transform=ax.transAxes, va="top", fontsize=8, bbox={"facecolor": "white", "edgecolor": "none", "alpha": .85})
        main = frames[(basin, "dominant_main_position")]
        model = fitted[(basin, "dominant_main_position")]
        draw_position_plot(axes[2], main, model, "最大主河道同位置组")
        record = get_row(scores, basin, "dominant_main_position")
        axes[2].text(.02, .98, f"访问={record.n_visits}\n留一次访问 R²={record.all_r2:.3f}", transform=axes[2].transAxes, va="top", fontsize=8)
        axes[0].legend(fontsize=7, loc="lower right")
        fig.suptitle(f"{basin}：水位—面积线性验证；蓝实线为合并直线，彩色虚线为分位置直线", fontsize=12)
        fig.savefig(plots / f"{basin}.png", dpi=145)
        plt.close(fig)
        if number % 20 == 0:
            print(f"已绘制位置对照图 {number}/86", flush=True)
    examples = ["01029500", "01017000", "01017290", "01060000", "01092000", "01115770"]
    fig, axes = plt.subplots(2, 3, figsize=(14, 8), layout="constrained")
    for ax, basin in zip(axes.ravel(), examples):
        draw_position_plot(ax, frames[(basin, "openwater_mixed")], fitted[(basin, "openwater_mixed")], basin)
        record = get_row(scores, basin, "openwater_mixed")
        ax.text(.02, .98, f"混合位置留一次访问 R²={record.all_r2:.3f}\n访问={record.n_visits}；位置组={record.n_positions}",
                transform=ax.transAxes, va="top", fontsize=9, bbox={"facecolor": "white", "edgecolor": "none", "alpha": .85})
    handles, labels = axes.ravel()[0].get_legend_handles_labels()
    # 总图各站的位置编号独立，图例不沿用第一站的访问次数。
    labels = [label.split("，访问")[0] for label in labels]
    fig.legend(handles, labels, loc="outside lower center", ncol=3, fontsize=9)
    fig.suptitle("位置不统一时的水位—过水面积关系示例", fontsize=14)
    fig.savefig(OUTPUT / "mixed_position_examples.png", dpi=150)
    plt.close(fig)

    summary_scopes = []
    for scope in SCOPES:
        subset = scores[(scores.scope == scope) & (scores.scheme == "leave_one_visit_out") & (scores.model == "linear")]
        valid = subset[(subset.prediction_record_coverage >= .8) & (subset.all_n >= 3)]
        summary_scopes.append({"scope": scope, "n_stations_with_records": int((subset.n_records > 0).sum()),
            "n_with_at_least_80pct_prediction_coverage": len(valid), "median_lovo_linear_r2": valid.all_r2.median(),
            "n_lovo_r2_at_least_080": int((valid.all_r2 >= .8).sum()), "n_lovo_r2_at_least_050": int((valid.all_r2 >= .5).sum()),
            "n_lovo_r2_below_zero": int((valid.all_r2 < 0).sum())})
    remaining = stations[~stations.in_previous_primary14]
    main_metrics = scores[(scores.scope == "main_known_positions") & (scores.scheme == "leave_one_visit_out")]
    offset = main_metrics[main_metrics.model == "position_offset"]
    separate = main_metrics[main_metrics.model == "position_linear"]
    position_holdout = scores[(scores.scope == "main_known_positions") & (scores.scheme == "leave_one_position_out") & (scores.model == "linear")]
    position_valid = position_holdout[(position_holdout.prediction_record_coverage >= .8) & (position_holdout.all_n >= 3)]
    summary = {"n_stations": 86, "n_training_pairs": len(data), "n_independent_visits": data.visit_key.nunique(),
        "scope_summaries": summary_scopes, "remaining72_bands": remaining.mixed_line_prediction_band.value_counts().to_dict(),
        "all86_bands": stations.mixed_line_prediction_band.value_counts().to_dict(),
        "remaining72_median_clean_lovo_r2": remaining.openwater_mixed_all_r2.median(),
        "position_offset_mae_reduction_at_least_10pct": int((offset.paired_mae_reduction_vs_linear >= .1).sum()),
        "position_separate_linear_mae_reduction_at_least_10pct": int((separate.paired_mae_reduction_vs_linear >= .1).sum()),
        "n_position_holdout_with_80pct_coverage": len(position_valid), "median_position_holdout_r2": position_valid.all_r2.median(),
        "n_position_holdout_r2_at_least_080": int((position_valid.all_r2 >= .8).sum()),
        "n_stations_with_two_positions_ge5_visits": int((differences.n_positions_at_least_5_visits >= 2).sum()),
        "n_two_positions_overlap_h": int(differences.get("has_overlapping_h_range", pd.Series(dtype=bool)).eq(True).sum()),
        "n_two_positions_area_difference_at_least_20pct": int((differences.get("relative_area_difference", pd.Series(dtype=float)) >= .2).sum()),
        "same_physical_cross_section_verified": False, "input_sha256": digest(INPUT), "script_sha256": digest(__file__),
        "helper_script_sha256": digest(ROOT / "tools/fit_cross_section86_area_h_curves.py"),
        "phase1_or_phase2_models_modified": False, "no_q_values_read": True,
        "n_metric_groups_independently_recomputed": n_rechecks}
    write_json(OUTPUT / "summary.json", summary)
    display = stations[["basin", "in_previous_primary14", "openwater_mixed_n_visits", "openwater_mixed_n_positions",
        "openwater_mixed_all_r2", "openwater_mixed_time_r2", "position_holdout_r2", "position_offset_mae_reduction",
        "dominant_main_position_n_visits", "dominant_main_position_all_r2", "mixed_line_prediction_band"]].copy()
    for column in display.columns:
        if column.endswith("r2"):
            display[column] = display[column].map(lambda value: f"{value:.3f}" if np.isfinite(value) else "不足")
    display["position_offset_mae_reduction"] = display.position_offset_mae_reduction.map(lambda value: f"{value:+.1%}" if np.isfinite(value) else "不足")
    display.columns = ["站点", "原14站", "混合访问数", "位置组数", "混合直线R²", "末期R²", "整位置留出R²", "位置偏移MAE降低", "最大位置访问数", "该位置R²", "浏览分组"]
    lines = ["# 全部86站及混合位置水位—面积关系验证", "",
        "本次在前次拟合基础上验证全部86站，不以15次访问作为统一排除条件。仅使用原训练期H–A及位置/方法元数据，不读取Q、流速或原验证/测试期面积，不修改阶段1或阶段2模型。", "",
        "## 主要结论", "",
        f"位置不统一不等于没有线性关系：清洗后全部86站的混合位置逐访问留出直线R²中位数为{stations.openwater_mixed_all_r2.median():.3f}，其中{int(stations.openwater_mixed_all_r2.ge(.8).sum())}站达到0.8。",
        "但混合位置的经验面积指标不等同于固定参考断面的真实过水面积；部分站的不同位置不能简单共用同一条几何曲线。",
        f"混合逐访问和末期时间留出R²同时达到0.8的有{int((stations.openwater_mixed_all_r2.ge(.8) & stations.openwater_mixed_time_r2.ge(.8)).sum())}站；再加整位置留出R²达到0.8且覆盖至少80%记录，有{int((stations.openwater_mixed_all_r2.ge(.8) & stations.openwater_mixed_time_r2.ge(.8) & stations.position_holdout_r2.ge(.8) & stations.position_holdout_coverage.ge(.8)).sum())}站。这里只是经验验证候选，尚未批准接入水文模型。", "",
        "## 核心口径", "",
        "1. 全记录、去冰期及明确非主河道/复核、明确主河道已知位置、最大同位置组四种口径分别报告。未知名称/位置不会被自动认定为同一断面。",
        "2. 逐次留出独立现场访问，同一次访问的重复记录不会跨训练和验证。至少3次访问尝试验证，但3～4次仅低置信度；1～2次不伪造验证结果。主汇总要求有效预测覆盖至少80%记录。",
        "3. 原训练期末20%访问（至少15次时末期至少5次，较小时至少2次）另作时间留出，并清除时间交叠访问。",
        "4. 对明确主河道且位置已知的记录，额外整组留出某个位置，并从训练中去掉该组涉及的全部现场访问。检验合并直线对未参与拟合的位置是否仍有预测力。",
        "5. 比较合并直线、二次、单调曲线；同位置条件另比较‘合并直线＋训练位置残差均值’和‘有至少3次训练访问时分别拟合直线，其他位置回退合并直线’。这些均为诊断，不是已批准的最终几何模型。",
        "6. R²≥0.8、0.5和MAE降低10%/面积差20%只是描述性浏览阈值，不是统计显著性或物理同断面的证明。主数值包括区间外点，另存区间内数值及覆盖率。", "",
        "## 混合位置直线验证概况", "",
        f"全部86站分组：{summary['all86_bands']}。",
        f"原先未达到主要验证门槛的72站：{summary['remaining72_bands']}；混合位置逐访问留出直线R²中位数{summary['remaining72_median_clean_lovo_r2']:.3f}。", ""]
    for item in summary_scopes:
        lines += [f"- {LABELS[item['scope']]}：{item['n_with_at_least_80pct_prediction_coverage']}站有至少80%记录可验证，R²中位数{item['median_lovo_linear_r2']:.3f}，R²≥0.8有{item['n_lovo_r2_at_least_080']}站。"]
    lines += ["", "## 位置变化是否影响共用曲线", "",
        f"在同样的主河道已知位置记录上，加入位置偏移后，{summary['position_offset_mae_reduction_at_least_10pct']}站的留出MAE至少降低10%；分位置直线有{summary['position_separate_linear_mae_reduction_at_least_10pct']}站达到该改善。",
        f"整位置留出有{len(position_valid)}站达到80%预测覆盖，直线R²中位数为{summary['median_position_holdout_r2']:.3f}，其中{summary['n_position_holdout_r2_at_least_080']}站R²≥0.8。",
        f"有{summary['n_stations_with_two_positions_ge5_visits']}站至少两个位置各具5次访问；选取最大两个位置后，{summary['n_two_positions_overlap_h']}站具有共同H范围，其中{summary['n_two_positions_area_difference_at_least_20pct']}站在共同水位中点的两条直线面积差至少20%。",
        "位置效应可能与水位范围、时间、测量方法和河床变化相关，以上比较不能单独识别因果原因。没有共同H范围时，不能靠不同位置的斜率差就判定几何差异。", "",
        "## 全部86站逐站结果", "", md(display), "",
        "最大同位置组样本很少时，R²可能剧烈波动；‘不足’表示没有足够独立访问或水位变化以形成有效留出预测，不计作零分。", "",
        "## 图及复现", "", "![混合位置示例](mixed_position_examples.png)", "",
        "station_plots/保留全部86站的混合位置、明确主河道位置及最大位置组对照图。",
        "all86_station_validation.csv为逐站总表；validation_metrics.csv含全部模型/验证口径；position_linear_parameters.csv含每个位置的直线参数；position_function_differences.csv含最大两个位置的共同水位比较。",
        "heldout_area_predictions.csv.gz保存全部留出预测；integrity.json保存独立指标重算和分组隔离检查。", "",
        "复现：`python -X utf8 tools/validate_cross_section86_area_h_positions.py`。", ""]
    (OUTPUT / "review.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps(safe_json(summary), ensure_ascii=False, indent=2))
    print("\n剩余72站：\n" + remaining[["basin", "openwater_mixed_all_r2", "openwater_mixed_time_r2", "position_holdout_r2", "mixed_line_prediction_band"]].to_string(index=False))


if __name__ == "__main__":
    main()
