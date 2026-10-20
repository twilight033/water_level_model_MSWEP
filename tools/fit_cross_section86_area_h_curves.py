"""探索原86站训练期H–A关系的线性程度，不读取Q，不训练水文网络。"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np
import pandas as pd
import scipy
from scipy.optimize import isotonic_regression


ROOT = Path(__file__).resolve().parents[1]
INPUT = ROOT / "reports/cross_section86_area_h_overlap/area_h_pairs_inventory.csv"
OUTPUT = ROOT / "reports/cross_section86_area_h_linearity"
MIN_VISITS = 15
MIN_UNIQUE_H = 6
SEED = 20261020
MODELS = ("constant", "linear", "quadratic", "isotonic")
MODEL_LABELS = {"constant": "常数面积", "linear": "直线", "quadratic": "二次曲线", "isotonic": "单调分段线性"}
SCOPE_LABELS = {
    "all_train_mixed": "所有训练期配对，混合位置",
    "openwater_mixed": "排除冰期及明确支汊/复核，仍混合位置",
    "dominant_main_position": "记录最多的主河道同位置元数据组",
    "dominant_main_position_method": "记录最多的主河道同位置同方法元数据组",
}


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def json_write(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")


def metrics(y, prediction, weights):
    y, prediction, weights = (np.asarray(value, float) for value in (y, prediction, weights))
    valid = np.isfinite(y) & np.isfinite(prediction) & np.isfinite(weights) & (weights > 0)
    y, prediction, weights = y[valid], prediction[valid], weights[valid]
    if not len(y):
        return {"n": 0, "r2": np.nan, "mae": np.nan, "rmse": np.nan}
    mean = np.average(y, weights=weights)
    error = prediction - y
    denominator = np.sum(weights * (y - mean) ** 2)
    return {"n": len(y), "r2": 1 - np.sum(weights * error ** 2) / denominator if denominator > 1e-15 else np.nan,
            "mae": np.average(np.abs(error), weights=weights),
            "rmse": np.sqrt(np.average(error ** 2, weights=weights))}


def fit_curve(x, y, weights, model):
    """拟合仅含H和面积的曲线；直线/二次曲线不加形状约束，用于诊断。"""
    x, y, weights = (np.asarray(value, float) for value in (x, y, weights))
    if model == "isotonic":
        # 相同H先按测量权重合并，防止同一自变量得到多个不同拟合值。
        levels, inverse = np.unique(x, return_inverse=True)
        mass = np.bincount(inverse, weights=weights)
        target = np.bincount(inverse, weights=weights * y) / mass
        fitted = isotonic_regression(target, weights=mass).x
        return {"model": model, "h_nodes_m": levels.tolist(), "area_nodes_m2": fitted.tolist()}
    center = float(np.average(x, weights=weights))
    scale = max(float(np.sqrt(np.average((x - center) ** 2, weights=weights))), 1e-8)
    degree = {"constant": 0, "linear": 1, "quadratic": 2}[model]
    design = np.vander((x - center) / scale, N=degree + 1, increasing=True)
    coefficient = np.linalg.lstsq(design * np.sqrt(weights)[:, None], y * np.sqrt(weights), rcond=None)[0]
    curve = {"model": model, "center_m": center, "scale_m": scale, "coefficients_scaled": coefficient.tolist()}
    if model == "linear":
        curve.update(slope_m=float(coefficient[1] / scale),
                     intercept_m2=float(coefficient[0] - coefficient[1] * center / scale))
    if model == "quadratic":
        derivative = (coefficient[1] + 2 * coefficient[2] * (np.array([x.min(), x.max()]) - center) / scale) / scale
        curve["nondecreasing_within_observed_range"] = bool((derivative >= -1e-9).all())
    return curve


def predict_curve(curve, x):
    x = np.asarray(x, float)
    if curve["model"] == "isotonic":
        return np.interp(x, curve["h_nodes_m"], curve["area_nodes_m2"])
    return np.polynomial.polynomial.polyval((x - curve["center_m"]) / curve["scale_m"], curve["coefficients_scaled"])


def visit_weights(frame):
    """每次独立现场访问总权重为1，不平均不同测线的原始记录。"""
    return 1.0 / frame.groupby("visit_key").visit_key.transform("size").to_numpy(float)


def prepare_data():
    data = pd.read_csv(INPUT, dtype={"basin": str, "field_visit_id": str, "record_id": str}, low_memory=False)
    data = data[(data.model_period == "train") & np.isfinite(data.h_at_measurement_interpolated_m)
                & np.isfinite(data.area_m2) & (data.area_m2 > 0)].copy()
    assert data.record_id.is_unique
    data["h_m"] = data.h_at_measurement_interpolated_m.astype(float)
    data["visit_key"] = data.field_visit_id.fillna(data.record_id)
    data["measurement_time"] = pd.to_datetime(data.measurement_time, format="ISO8601")
    data["name_normalized"] = data.channel_name.fillna("").str.strip().str.lower().str.replace(r"\s+", " ", regex=True)
    data["explicit_main"] = data.name_normalized.isin(["main", "main channel"])
    data["ice"] = data.measurement_type.fillna("").str.lower().eq("ice")
    # 只排除名称明确标注的复核、支汊和支流，不把未知名称自动当成主河道。
    data["explicit_nonmain_or_check"] = data.name_normalized.str.contains(
        r"check|\bchk\b|secondary|inflow|\bleft\b|\bright\b|channel 2|^2$", regex=True)
    data["openwater_candidate"] = ~data.ice & ~data.explicit_nonmain_or_check
    distance = pd.to_numeric(data.channel_location_distance, errors="coerce")
    units = data.channel_location_distance_unit.fillna("").str.lower()
    data["distance_m"] = np.where(units.eq("ft"), distance * 0.3048,
                                  np.where(units.isin(["m", "meter", "meters"]), distance, np.nan))
    at_gage = data.channel_location_direction.eq("AtTheGage")
    data.loc[at_gage & data.distance_m.isna(), "distance_m"] = 0.0
    data["position_known"] = data.distance_m.notna() & data.channel_location_direction.isin(
        ["Upstream", "Downstream", "AtTheGage"])
    data["position_key"] = data.channel_location_direction.fillna("Unspecified") + "|" + data.distance_m.map(
        lambda value: f"{value:.6f}" if np.isfinite(value) else "unknown")
    data["method_key"] = data.measurement_type.fillna("Unspecified") + "|" + data.channel_measurement_type.fillna("Unspecified")
    data["strict_context_key"] = data.position_key + "|" + data.method_key
    return data


def select_dominant(frame, key):
    candidates = frame[frame.openwater_candidate & frame.explicit_main & frame.position_known]
    if candidates.empty:
        return candidates.copy(), "无主河道已知位置记录"
    counts = candidates.groupby(key).agg(n_visits=("visit_key", "nunique"), n_records=("record_id", "size")).reset_index()
    chosen = counts.sort_values(["n_visits", "n_records", key], ascending=[False, False, True]).iloc[0][key]
    return candidates[candidates[key] == chosen].copy(), str(chosen)


def evaluate_scope(frame, basin, scope, context):
    common = {"basin": basin, "scope": scope, "context_key": context, "n_records": len(frame),
              "n_visits": frame.visit_key.nunique(), "n_unique_h": frame.h_m.nunique(),
              "n_positions": frame.position_key.nunique(), "n_methods": frame.method_key.nunique()}
    if frame.empty:
        return [{**common, "model": name, "cv_eligible": False} for name in MODELS], [], {}
    frame = frame.sort_values(["measurement_time", "visit_key", "record_id"]).reset_index(drop=True)
    x, y, weights = frame.h_m.to_numpy(), frame.area_m2.to_numpy(), visit_weights(frame)
    common.update(h_min_m=float(x.min()), h_max_m=float(x.max()), h_span_m=float(np.ptp(x)),
                  area_mean_m2=float(np.average(y, weights=weights)), area_min_m2=float(y.min()), area_max_m2=float(y.max()),
                  first_time=frame.measurement_time.min().isoformat(), last_time=frame.measurement_time.max().isoformat())
    eligible = common["n_visits"] >= MIN_VISITS and common["n_unique_h"] >= MIN_UNIQUE_H
    curves = {model: fit_curve(x, y, weights, model) for model in MODELS}
    oof = {model: np.full(len(frame), np.nan) for model in MODELS}
    in_range = np.zeros(len(frame), bool)
    folds = np.full(len(frame), -1, int)
    if eligible:
        visits = np.array(sorted(frame.visit_key.unique()), dtype=object)
        np.random.default_rng(SEED).shuffle(visits)
        for fold in range(3):
            holdout = set(visits[fold::3])
            test = frame.visit_key.isin(holdout).to_numpy()
            train = ~test
            assert not set(frame.loc[train, "visit_key"]) & set(frame.loc[test, "visit_key"])
            folds[test] = fold
            in_range[test] = (x[test] >= x[train].min()) & (x[test] <= x[train].max())
            for model in MODELS:
                curve = fit_curve(x[train], y[train], weights[train], model)
                oof[model][test] = predict_curve(curve, x[test])
    # 时间外推检查全部发生在原训练期内部，不使用原验证/测试期的面积。
    time_test = np.zeros(len(frame), bool)
    chronological = {}
    if eligible:
        ordered = frame.groupby("visit_key").measurement_time.min().sort_values(kind="stable").index.tolist()
        n_last = max(5, math.ceil(len(ordered) * 0.2))
        time_test = frame.visit_key.isin(set(ordered[-n_last:])).to_numpy()
        time_train = ~time_test
        assert frame.loc[time_train, "measurement_time"].max() <= frame.loc[time_test, "measurement_time"].min()
        for model in MODELS:
            curve = fit_curve(x[time_train], y[time_train], weights[time_train], model)
            chronological[model] = predict_curve(curve, x[time_test])
        time_in_range = (x[time_test] >= x[time_train].min()) & (x[time_test] <= x[time_train].max())
    rows, predictions = [], []
    for model in MODELS:
        curve = curves[model]
        row = {**common, "model": model, "cv_eligible": bool(eligible)}
        row.update({"train_" + key: value for key, value in metrics(y, predict_curve(curve, x), weights).items()})
        if model == "linear":
            row.update(linear_slope_m=curve["slope_m"], linear_intercept_m2=curve["intercept_m2"])
        if model == "quadratic":
            row["quadratic_nondecreasing_in_range"] = curve["nondecreasing_within_observed_range"]
        if eligible:
            for label, mask in (("cv_all", np.ones(len(frame), bool)), ("cv_inrange", in_range)):
                row.update({label + "_" + key: value for key, value in metrics(y[mask], oof[model][mask], weights[mask]).items()})
            row["cv_inrange_record_fraction"] = float(in_range.mean())
            row.update({"time_holdout_all_" + key: value for key, value in metrics(y[time_test], chronological[model], weights[time_test]).items()})
            row.update({"time_holdout_inrange_" + key: value for key, value in
                        metrics(y[time_test][time_in_range], chronological[model][time_in_range], weights[time_test][time_in_range]).items()})
            row["time_holdout_n_visits"] = int(frame.loc[time_test, "visit_key"].nunique())
            row["time_holdout_inrange_record_fraction"] = float(time_in_range.mean())
            for index in range(len(frame)):
                predictions.append({"basin": basin, "scope": scope, "model": model, "record_id": frame.record_id.iloc[index],
                    "visit_key": frame.visit_key.iloc[index], "fold": int(folds[index]), "h_m": float(x[index]),
                    "area_m2": float(y[index]), "prediction_m2": float(oof[model][index]),
                    "weight": float(weights[index]), "inside_fold_training_h_range": bool(in_range[index])})
        rows.append(row)
    return rows, predictions, curves


def markdown(frame):
    rows = ["| " + " | ".join(frame.columns) + " |", "| " + " | ".join(["---"] * len(frame.columns)) + " |"]
    for values in frame.itertuples(index=False, name=None):
        rows.append("| " + " | ".join(map(str, values)) + " |")
    return "\n".join(rows)


def draw_curves(ax, frame, curves, title, detail=""):
    if frame.empty:
        ax.text(.5, .5, "缺少明确主河道位置记录", ha="center", va="center", transform=ax.transAxes)
        ax.set_title(title)
        return
    years = frame.measurement_time.dt.year.to_numpy()
    ax.scatter(frame.h_m, frame.area_m2, c=years, cmap="viridis", s=25, edgecolors="white", linewidths=.35, zorder=3)
    if len(frame) >= 3 and frame.h_m.nunique() >= 2:
        grid = np.linspace(frame.h_m.min(), frame.h_m.max(), 180)
        for model, color, style in (("linear", "#1764ab", "-"), ("quadratic", "#e1812c", "--"), ("isotonic", "#318b55", ":")):
            ax.plot(grid, predict_curve(curves[model], grid), color=color, linestyle=style, linewidth=1.8, label=MODEL_LABELS[model])
    ax.set_title(title, fontsize=11)
    ax.set_xlabel("测站水位 H（m）")
    ax.set_ylabel("测量过水面积 A（m²）")
    ax.grid(alpha=.16)
    if detail:
        ax.text(.03, .97, detail, transform=ax.transAxes, va="top", fontsize=8,
                bbox={"facecolor": "white", "edgecolor": "none", "alpha": .83})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--skip-station-plots", action="store_true", help="输入未变时复用已有86站逐站图，仅更新汇总图和报告")
    args = parser.parse_args()
    OUTPUT.mkdir(parents=True, exist_ok=True)
    plots = OUTPUT / "station_plots"
    plots.mkdir(exist_ok=True)
    if Path("C:/Windows/Fonts/msyh.ttc").is_file():
        font_manager.fontManager.addfont("C:/Windows/Fonts/msyh.ttc")
        plt.rcParams["font.family"] = [font_manager.FontProperties(fname="C:/Windows/Fonts/msyh.ttc").get_name(), "DejaVu Sans"]
    plt.rcParams.update({"axes.unicode_minus": False, "figure.facecolor": "white", "savefig.facecolor": "white"})

    # 用已知直线检验系数恢复；检查等序曲线的约束及相同H处理。
    toy_x = np.array([0., 1., 2., 3.])
    toy = fit_curve(toy_x, 4 + 3 * toy_x, np.ones(4), "linear")
    assert np.allclose(predict_curve(toy, toy_x), 4 + 3 * toy_x, atol=1e-12)
    assert np.isclose(toy["slope_m"], 3) and np.isclose(toy["intercept_m2"], 4)
    toy_iso = fit_curve(np.array([0., 0., 1., 2.]), np.array([4., 2., 1., 8.]), np.ones(4), "isotonic")
    assert np.all(np.diff(toy_iso["area_nodes_m2"]) >= 0)
    assert np.isclose(predict_curve(toy_iso, [0])[0], 7 / 3)

    data = prepare_data()
    assert len(data) == 6618 and data.basin.nunique() == 86
    if args.skip_station_plots:
        previous = json.loads((OUTPUT / "summary.json").read_text(encoding="utf-8"))
        assert previous["input_sha256"] == digest(INPUT)
        assert all((plots / f"{basin}.png").is_file() for basin in data.basin.unique())
    selected_frames, all_curves, rows, oof_rows, selection = {}, {}, [], [], []
    for number, (basin, raw) in enumerate(data.groupby("basin", sort=True), 1):
        main_frame, context = select_dominant(raw, "position_key")
        strict_frame, strict_context = select_dominant(raw, "strict_context_key")
        scopes = {"all_train_mixed": (raw, "mixed"),
                  "openwater_mixed": (raw[raw.openwater_candidate], "mixed"),
                  "dominant_main_position": (main_frame, context),
                  "dominant_main_position_method": (strict_frame, strict_context)}
        for scope, (frame, signature) in scopes.items():
            result, oof, curves = evaluate_scope(frame, basin, scope, signature)
            rows.extend(result)
            oof_rows.extend(oof)
            selected_frames[(basin, scope)] = frame
            all_curves[(basin, scope)] = curves
            selection.append({"basin": basin, "scope": scope, "context_key": signature,
                "n_records": len(frame), "n_visits": frame.visit_key.nunique(), "n_unique_h": frame.h_m.nunique()})
        if number % 20 == 0:
            print(f"已拟合 {number}/86 站", flush=True)

    scores = pd.DataFrame(rows)
    oof = pd.DataFrame(oof_rows)
    linear = scores[scores.model == "linear"].copy()
    for model in ("quadratic", "isotonic"):
        columns = ["basin", "scope", "cv_inrange_mae", "cv_inrange_r2", "time_holdout_all_mae", "train_r2"]
        extra = scores[scores.model == model][columns].rename(columns={name: model + "_" + name for name in columns[2:]})
        linear = linear.merge(extra, on=["basin", "scope"], validate="one_to_one")
        linear[model + "_cv_mae_reduction"] = 1 - linear[model + "_cv_inrange_mae"] / linear.cv_inrange_mae
        linear[model + "_time_mae_reduction"] = 1 - linear[model + "_time_holdout_all_mae"] / linear.time_holdout_all_mae
    linear["cv_r2_at_least_080"] = linear.cv_inrange_r2 >= .8
    linear["positive_slope"] = linear.linear_slope_m > 0
    linear["best_nonlinear_cv_mae_reduction"] = linear[["quadratic_cv_mae_reduction", "isotonic_cv_mae_reduction"]].max(axis=1)
    linear["browse_label"] = "样本不足，仅画图"
    eligible = linear.cv_eligible
    linear.loc[eligible, "browse_label"] = "关系较弱或不稳定"
    linear.loc[eligible & linear.positive_slope & (linear.cv_inrange_r2 >= .8)
               & (linear.best_nonlinear_cv_mae_reduction < .1), "browse_label"] = "线性较强且非线性增益小"
    linear.loc[eligible & linear.positive_slope & (linear.best_nonlinear_cv_mae_reduction >= .1), "browse_label"] = "非线性候选，待时间稳定性核验"
    scores.to_csv(OUTPUT / "curve_metrics.csv", index=False, encoding="utf-8-sig")
    linear.to_csv(OUTPUT / "station_linearity.csv", index=False, encoding="utf-8-sig")
    linear[["basin", "scope", "context_key", "n_visits", "h_min_m", "h_max_m", "linear_intercept_m2",
            "linear_slope_m", "train_r2", "cv_eligible", "cv_inrange_r2"]].to_csv(
                OUTPUT / "linear_parameters.csv", index=False, encoding="utf-8-sig")
    oof.to_csv(OUTPUT / "geometry_oof_predictions.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(selection).to_csv(OUTPUT / "selected_contexts.csv", index=False, encoding="utf-8-sig")
    data.to_csv(OUTPUT / "training_pair_audit.csv", index=False, encoding="utf-8-sig")
    json_write(OUTPUT / "fitted_curves.json", [{"basin": basin, "scope": scope, "curves": curves}
                                               for (basin, scope), curves in all_curves.items()])

    # 逐条折外预测重算指标，确保表格中的分组和验证口径一致。
    rechecks = []
    for (basin, scope, model), frame in oof.groupby(["basin", "scope", "model"]):
        matched = scores[(scores.basin == basin) & (scores.scope == scope) & (scores.model == model)].iloc[0]
        inside = frame[frame.inside_fold_training_h_range]
        result = metrics(inside.area_m2, inside.prediction_m2, inside.weight)
        passed = all(np.isclose(result[key], matched["cv_inrange_" + key], equal_nan=True, atol=1e-10) for key in ("r2", "mae", "rmse"))
        assert passed
        rechecks.append({"basin": basin, "scope": scope, "model": model, "passed": passed})
    json_write(OUTPUT / "integrity.json", {"all_train_only": bool(data.model_period.eq("train").all()),
        "record_id_unique": bool(data.record_id.is_unique), "n_recomputed_cv_metric_groups": len(rechecks),
        "all_rechecks_passed": all(row["passed"] for row in rechecks), "read_q_or_velocity_values": False,
        "synthetic_linear_and_monotone_checks_passed": True})

    for number, basin in enumerate([] if args.skip_station_plots else sorted(data.basin.unique()), 1):
        fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.3), layout="constrained")
        for ax, scope in zip(axes, ("all_train_mixed", "dominant_main_position")):
            record = linear[(linear.basin == basin) & (linear.scope == scope)].iloc[0]
            detail = f"独立访问={record.n_visits}；测量位置组={record.n_positions if pd.notna(record.n_positions) else 0}\n直线拟合 R²={record.train_r2:.3f}" if record.n_records else ""
            if record.cv_eligible:
                detail += f"；折外 R²={record.cv_inrange_r2:.3f}"
            draw_curves(ax, selected_frames[(basin, scope)], all_curves[(basin, scope)],
                        "全部候选记录，位置混合" if scope == "all_train_mixed" else "主河道位置元数据一致的最大子组", detail)
            if scope == "dominant_main_position" and record.n_records:
                ax.text(.03, .02, str(record.context_key), transform=ax.transAxes, fontsize=8)
        axes[0].legend(loc="lower right", fontsize=8)
        fig.suptitle(f"{basin}：训练期水位—过水面积；散点颜色从深到浅表示年份较早到较晚", fontsize=11)
        fig.savefig(plots / f"{basin}.png", dpi=160)
        plt.close(fig)
        if number % 20 == 0:
            print(f"已绘制 {number}/86 站", flush=True)

    primary = linear[(linear.scope == "dominant_main_position") & linear.cv_eligible].sort_values("basin")
    columns = 4
    fig, axes = plt.subplots(math.ceil(len(primary) / columns), columns,
        figsize=(16, 3.4 * math.ceil(len(primary) / columns)), squeeze=False, layout="constrained")
    for ax, (_, record) in zip(axes.ravel(), primary.iterrows()):
        detail = f"访问={record.n_visits}，ΔH={record.h_span_m:.2f}m\n直线折外 R²={record.cv_inrange_r2:.2f}"
        draw_curves(ax, selected_frames[(record.basin, record.scope)], all_curves[(record.basin, record.scope)], record.basin, detail)
    for ax in axes.ravel()[len(primary):]:
        ax.set_visible(False)
    handles, labels = axes.ravel()[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside lower center", ncol=3)
    fig.suptitle("训练期同位置元数据组：至少15次独立访问；散点色阶表示测量年份", fontsize=14)
    fig.savefig(OUTPUT / "dominant_position_curves.png", dpi=160)
    plt.close(fig)

    examples = []
    for ordering, ascending in (("cv_inrange_r2", False), ("best_nonlinear_cv_mae_reduction", False), ("cv_inrange_r2", True)):
        for basin in primary.sort_values(ordering, ascending=ascending).basin:
            if basin not in examples:
                examples.append(basin)
                break
        remaining = primary[~primary.basin.isin(examples)].sort_values(ordering, ascending=ascending)
        if len(remaining):
            examples.append(remaining.iloc[0].basin)
    fig, axes = plt.subplots(2, 3, figsize=(13.5, 8), layout="constrained")
    for ax, basin in zip(axes.ravel(), examples):
        record = primary[primary.basin == basin].iloc[0]
        detail = f"访问={record.n_visits}，ΔH={record.h_span_m:.2f}m\n直线折外 R²={record.cv_inrange_r2:.2f}"
        draw_curves(ax, selected_frames[(basin, "dominant_main_position")], all_curves[(basin, "dominant_main_position")], basin, detail)
    for ax in axes.ravel()[len(examples):]:
        ax.set_visible(False)
    fig.legend(handles, labels, loc="outside lower center", ncol=3)
    fig.suptitle("水位—过水面积关系示例：线性较强、非线性候选与关系较弱的站点", fontsize=13)
    fig.savefig(OUTPUT / "curve_examples.png", dpi=160)
    plt.close(fig)

    summaries = []
    for scope, subset in linear.groupby("scope"):
        evaluated = subset[subset.cv_eligible]
        summaries.append({"scope": scope, "n_stations_with_records": int((subset.n_records > 0).sum()),
            "n_cv_eligible": len(evaluated), "n_visits_at_least_20": int((subset.n_visits >= 20).sum()),
            "median_train_linear_r2": float(evaluated.train_r2.median()),
            "median_all_record_groups_train_linear_r2": float(subset.train_r2.median()),
            "median_cv_linear_r2": float(evaluated.cv_inrange_r2.median()),
            "linear_cv_r2_at_least_080": int((evaluated.cv_inrange_r2 >= .8).sum()),
            "linear_cv_r2_below_zero": int((evaluated.cv_inrange_r2 < 0).sum()),
            "quadratic_cv_mae_reduction_at_least_10pct": int((evaluated.quadratic_cv_mae_reduction >= .1).sum()),
            "isotonic_cv_mae_reduction_at_least_10pct": int((evaluated.isotonic_cv_mae_reduction >= .1).sum()),
            "median_quadratic_cv_mae_reduction": float(evaluated.quadratic_cv_mae_reduction.median()),
            "median_isotonic_cv_mae_reduction": float(evaluated.isotonic_cv_mae_reduction.median()),
            "median_cv_inrange_fraction": float(evaluated.cv_inrange_record_fraction.median()),
            "median_time_holdout_linear_r2": float(evaluated.time_holdout_all_r2.median()),
            "median_time_quadratic_mae_reduction": float(evaluated.quadratic_time_mae_reduction.median()),
            "median_time_isotonic_mae_reduction": float(evaluated.isotonic_time_mae_reduction.median())})
    summary = {"population": 86, "n_training_pairs": len(data), "n_independent_visits": data.visit_key.nunique(),
        "n_ice_records": int(data.ice.sum()), "n_explicit_nonmain_or_check_records": int(data.explicit_nonmain_or_check.sum()),
        "seed": SEED, "min_visits_for_cv": MIN_VISITS, "min_unique_h_for_cv": MIN_UNIQUE_H,
        "cv_folds": 3, "scopes": summaries, "example_basins": examples,
        "same_physical_cross_section_verified": False, "no_q_or_velocity_values_read": True,
        "test_period_areas_or_q_used": False, "input_sha256": digest(INPUT), "script_sha256": digest(__file__),
        "environment": {"numpy": np.__version__, "pandas": pd.__version__, "scipy": scipy.__version__, "matplotlib": matplotlib.__version__}}
    json_write(OUTPUT / "summary.json", summary)

    display = primary[["basin", "n_visits", "h_span_m", "linear_slope_m", "train_r2", "cv_inrange_r2",
                       "quadratic_cv_mae_reduction", "isotonic_cv_mae_reduction", "time_holdout_all_r2"]].copy()
    for column in ("h_span_m", "linear_slope_m", "train_r2", "cv_inrange_r2", "time_holdout_all_r2"):
        display[column] = display[column].map(lambda value: f"{value:.3f}")
    for column in ("quadratic_cv_mae_reduction", "isotonic_cv_mae_reduction"):
        display[column] = display[column].map(lambda value: f"{value:+.1%}")
    display.columns = ["站点", "独立访问", "观测H跨度m", "直线斜率m", "拟合R²", "折外R²", "二次曲线MAE降低", "单调曲线MAE降低", "末期留出直线R²"]
    lines = ["# 原86站训练期水位—过水面积线性诊断", "",
        "这是几何关系的探索性拟合，不训练阶段1/阶段2水文网络，不改变原模型与结果。只读取已有几何配对清单的H、A、日期及测量位置/方法，不读取Q或流速数值。", "",
        f"原训练期共有{len(data):,}条H–A候选记录，对应{data.visit_key.nunique():,}次现场访问。",
        "测量面积来自多处断面；时间配对成功不等于同一固定断面。下面的‘同位置’仅指元数据中的主河道名称、上下游方向及距离一致，仍不能宣称真实测线已经核实。", "",
        "## 初步结论", "",
        f"在可做分组验证的{len(primary)}个主河道同位置元数据组中，直线折外R²中位数为{primary.cv_inrange_r2.median():.3f}，"
        f"其中{int((primary.cv_inrange_r2 >= .8).sum())}站达到0.8。数据支持多数这些站在已测范围内可用直线近似，不能推广为全部86站严格线性。",
        f"二次曲线的折外MAE降低中位数为{primary.quadratic_cv_mae_reduction.median():+.1%}；"
        f"单调曲线为{primary.isotonic_cv_mae_reduction.median():+.1%}。本批数据没有支持统一优先采用单调曲线。",
        "少数站存在值得继续检查的曲率，但也有小样本、年度差异和高水位覆盖不足；拟合R²较高不等于曲线严格为直线，非线性改善也不自动证明真实河槽曲率。", "",
        "## 筛选及验证", "",
        "- 原始训练期全量作为混合位置诊断；另排除Ice和名称明确标注的复核/支汊/支流记录，保留排除原因。",
        "- 主分析选择名称明确为Main或Main Channel、位置已知的主河道记录；距离统一为m，方向和距离必须完全一致。",
        "- 每站按独立访问数最多选择一个位置组，平局按记录数和字符串次序确定，不以H、A拟合效果或Q选择。另做同位置同方法敏感性分析。",
        "- 不跨位置平均面积。每次访问总拟合权重为1；同一次访问的所有记录进入同一验证折。",
        "- 比较常数、直线A=a+bH、二次曲线、等序回归后线性插值。直线和二次曲线为形状诊断，不强行限制斜率；非物理解单独标记。",
        f"- 至少{MIN_VISITS}次独立访问且{MIN_UNIQUE_H}个不同H时，进行固定种子的3折按访问分组验证。主比较只用各折训练H范围内的共同验证点；同时保留含区间外点的全量评价。",
        "- 另外使用原训练期最后20%访问（至少5次）作时间留出，检验关系的时间稳定性；这不接触原实验验证或测试期面积。",
        "- 等序回归区间外采用端点截断，仅作数值回退；二次曲线区间外可能不稳定，不能当可靠洪水面积。所有超界比例单独记录。", "",
        "## 主河道同位置元数据组的结果", "", markdown(display), "",
        "表中MAE降低为1−非线性MAE/直线MAE；正值表示非线性在相同折外点上更好。R²不是Q模型的NSE。",
        "‘R²≥0.8’和‘MAE降低≥10%’仅为浏览用的经验阈值，不是显著性检验或自动确认真实非线性的标准。", ""]
    sensitivity = primary[primary.best_nonlinear_cv_mae_reduction >= .1][[
        "basin", "quadratic_cv_mae_reduction", "quadratic_time_mae_reduction", "isotonic_cv_mae_reduction",
        "isotonic_time_mae_reduction", "time_holdout_inrange_record_fraction"]].copy()
    for column in sensitivity.columns[1:]:
        sensitivity[column] = sensitivity[column].map(lambda value: f"{value:+.1%}")
    sensitivity.columns = ["站点", "二次折外MAE降低", "二次末期MAE降低", "单调折外MAE降低", "单调末期MAE降低", "末期测量H在拟合范围内比例"]
    lines += ["## 非线性候选的时间留出检查", "", markdown(sensitivity), "",
        "末期误差包含超出早期测量H范围的点；单调曲线在区间外采用端点截断，因此不能将全部时间留出差异仅归因为关系随时间改变。区间内误差另外保存在curve_metrics.csv。", "",
        "## 各种筛选口径的汇总", ""]
    for item in summaries:
        lines += [f"### {SCOPE_LABELS[item['scope']]}", "",
            f"有记录{item['n_stations_with_records']}站，可做分组验证{item['n_cv_eligible']}站，至少20次访问{item['n_visits_at_least_20']}站。",
            f"直线拟合R²中位数{item['median_train_linear_r2']:.3f}；折外R²中位数{item['median_cv_linear_r2']:.3f}；折外R²≥0.8有{item['linear_cv_r2_at_least_080']}站。",
            f"二次曲线折外MAE至少降低10%有{item['quadratic_cv_mae_reduction_at_least_10pct']}站；单调曲线有{item['isotonic_cv_mae_reduction_at_least_10pct']}站。",
            f"非线性MAE降低中位数：二次{item['median_quadratic_cv_mae_reduction']:+.1%}，单调{item['median_isotonic_cv_mae_reduction']:+.1%}。",
            f"末期留出直线R²中位数{item['median_time_holdout_linear_r2']:.3f}；二次/单调MAE降低中位数分别为{item['median_time_quadratic_mae_reduction']:+.1%}/{item['median_time_isotonic_mae_reduction']:+.1%}。", ""]
    lines += ["## 解释边界", "",
        "混合位置散点的弯曲或散乱不能直接归因于真实A(H)非线性。即使同位置元数据一致，仍可能存在水位基准、实际测线、河床、测量方法或回水关系变化；本轮没有把这些问题视为已解决。",
        "高R²仅适用于本组已测H范围；每站H跨度和折外范围覆盖率保留在表中，不能推广到未测洪水范围。",
        "非线性曲线若只改善随机分组验证、没有改善时间留出，不能宣称获得稳定的几何关系。",
        "本轮仅拟合面积；历史宽度/面积清单没有湿周。没有用面积反推出湿周，也没有把已测湿润长度冒充完整动态P(H)。", "",
        "## 图与完整输出", "",
        "![同位置元数据组全部合格站](dominant_position_curves.png)", "",
        "station_plots/包含全部86站的混合位置与同位置对照图。curve_metrics.csv保存所有模型和验证口径；station_linearity.csv为逐站比较；geometry_oof_predictions.csv保留每条折外预测。",
        "[逐站直线参数](linear_parameters.csv)保存A=a+bH的截距、斜率和H有效范围；其中有充分样本的主分析组须同时满足scope=dominant_main_position和cv_eligible=True。",
        "fitted_curves.json保存全训练期拟合参数及节点，仅供诊断，尚未接入任何Q模型。", "",
        "复现：`python -X utf8 tools/fit_cross_section86_area_h_curves.py`。", "",
        "算法参考：[SciPy等序回归](https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.isotonic_regression.html)。", ""]
    (OUTPUT / "review.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print("\n主分析逐站表：\n" + primary[["basin", "n_visits", "h_span_m", "train_r2", "cv_inrange_r2",
          "quadratic_cv_mae_reduction", "isotonic_cv_mae_reduction", "time_holdout_all_r2"]].to_string(index=False))


if __name__ == "__main__":
    main()
