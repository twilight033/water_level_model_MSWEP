"""16站阶段1岭回归：同输入、同划分比较有无面积，CPU求解，不训练水位网络。"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, replace
import os
from pathlib import Path
import time

import numpy as np
import pandas as pd

from . import area_stage1 as area
from . import area_stage1_only16 as cohort
from .converters import PairData, collect_pairs, validation_score
from .features import file_sha256
from .protocol import fingerprint, write_json
from .runner import save_prediction
from pipeline.dataset import PreparedData


ROOT = area.ROOT
PROTOCOL = "atgage16_only_stage1_ridge_train_valid_q_hidden_v1"
DEFAULT_OUTPUT = ROOT / "results/atgage16_area_ridge_stage1_v1"
VARIANTS = ("h_attrs_area_available", "h_attrs_area")
# 先确定搜索范围，再运行测试；目标使用流域平衡的平均损失，避免样本数改变惩罚含义。
ALPHAS = tuple(10. ** exponent for exponent in range(-6, 4))
N_EXPECTED = 2 * 9 * 3 * len(VARIANTS)


@dataclass
class RidgeModel:
    coef: np.ndarray
    intercept: float
    alpha: float

    def predict(self, x):
        return np.asarray(x, dtype=np.float64) @ self.coef + self.intercept


def fit_ridge(x, y, weight, alpha):
    """最小化sum(w*残差²)/sum(w)+alpha*sum(斜率²)，截距不惩罚。"""
    x, y, w = (np.asarray(v, dtype=np.float64) for v in (x, y, weight))
    if x.ndim != 2 or y.shape != (len(x),) or w.shape != y.shape or len(x) == 0:
        raise ValueError("岭回归输入维数不一致或样本为空")
    if not all(np.isfinite(v).all() for v in (x, y, w)) or (w < 0).any() or w.sum() <= 0:
        raise ValueError("岭回归输入非有限或权重不合法")
    if not np.isfinite(alpha) or alpha <= 0:
        raise ValueError("正则强度必须为有限正数")
    w = w / w.sum()
    mx, my = np.sum(w[:, None] * x, axis=0), np.sum(w * y)
    xc, yc = x - mx, y - my
    gram = xc.T @ (w[:, None] * xc)
    coef = np.linalg.solve(gram + alpha * np.eye(x.shape[1]), xc.T @ (w * yc))
    intercept = float(my - mx @ coef)
    if not np.isfinite(coef).all() or not np.isfinite(intercept):
        raise ValueError("岭回归系数不是有限值")
    return RidgeModel(coef, intercept, float(alpha))


def feature_names(prepared, variant):
    columns = list(prepared.attrs_raw.columns)
    names = ["waterlevel_z", *columns]
    if variant == "h_attrs_area":
        names.append("area_log1p_z")
    names.append("area_available")
    return names


def fit_converter(prepared, payload, scene, seed, scaling, variant, maximum):
    """只在16站内的Q保留站选正则强度，再用保留站训练期重新拟合。"""
    started = time.monotonic()
    if variant not in VARIANTS:
        raise ValueError("岭回归只运行已确定的有面积、无面积两个对照")
    retained, hidden = set(scene["retained"]), set(scene["heldout"])
    if retained & hidden or retained | hidden != set(area.AREA_BASINS):
        raise ValueError("拟合总体必须是16站的互斥划分")
    guarded = cohort.RestrictedQPrepared(prepared, scene["retained"])
    train_basins, valid_basins, split_audit = area.area_inner_split(scene["retained"], seed)
    donors = np.arange(len(prepared.basins))
    inner_scale = guarded.make_scaling(scaling, fit_basins=train_basins)
    train = collect_pairs(guarded, train_basins, "train", inner_scale, None, maximum)
    valid = collect_pairs(guarded, valid_basins, "valid", inner_scale, None, maximum)
    inner_area = area.area_scaler(guarded, payload, train, donors, train_basins) if variant == "h_attrs_area" else None
    train = replace(train, x=area.input_matrix(guarded, payload, train, variant, inner_area, donors))
    valid = replace(valid, x=area.input_matrix(guarded, payload, valid, variant, inner_area, donors))
    if len(np.unique(train.basin)) != len(train_basins) or len(np.unique(valid.basin)) != len(valid_basins):
        raise ValueError("某个指定训练/验证站没有有效配对")
    candidates = []
    for alpha in ALPHAS:
        model = fit_ridge(train.x, train.y, train.weight, alpha)
        candidates.append({"alpha": alpha, "valid_basin_nse": validation_score(valid, model.predict(valid.x))})
    chosen = max(candidates, key=lambda row: (row["valid_basin_nse"], row["alpha"]))
    scale = guarded.make_scaling(scaling, fit_basins=scene["retained"])
    final = collect_pairs(guarded, scene["retained"], "train", scale, None, maximum)
    if len(np.unique(final.basin)) != len(retained):
        raise ValueError("某个指定重拟合站没有有效配对")
    final_area = area.area_scaler(guarded, payload, final, donors, scene["retained"]) if variant == "h_attrs_area" else None
    final = replace(final, x=area.input_matrix(guarded, payload, final, variant, final_area, donors))
    model = fit_ridge(final.x, final.y, final.weight, chosen["alpha"])
    names = feature_names(prepared, variant)
    if len(names) != final.x.shape[1]:
        raise ValueError("输入特征名与矩阵不一致")
    audit = {"protocol": PROTOCOL, "estimator": "numpy_weighted_ridge", "variant": variant,
             "target_scaling": scaling, "alpha": model.alpha, "alpha_grid": ALPHAS, "candidates": candidates,
             "objective": "sum(w*(y-intercept-X@coef)^2)/sum(w)+alpha*sum(coef^2)",
             "intercept_penalized": False, "input_dim": final.x.shape[1], "n_parameters": len(model.coef) + 1,
             "feature_names": names, "n_train_pairs": len(final.y), "n_inner_train_pairs": len(train.y),
             "n_inner_validation_pairs": len(valid.y), "training_q_pair_basins": scene["retained"],
             "inner_train_basins": train_basins, "inner_validation_basins": valid_basins,
             "inner_split_audit": split_audit, "q_scaling_fit_calls": guarded.scaling_calls,
             "inner_area_scaler": inner_area, "final_area_scaler": final_area,
             "scale_fit_basins": scene["retained"] if scaling == "physical" else [],
             "observed_q_scale_basins": list(area.AREA_BASINS) if scaling == "observed" else [],
             "heldout_q_statistics_used_for_scaling": scaling == "observed",
             "converter_population": list(area.AREA_BASINS), "waterlevel_base_population_size": 86,
             "original86_q_retained": scene["original86_retained"], "original86_q_heldout": scene["original86_heldout"],
             "selected_by": "retained_within16_internal_whole_basin_validation",
             "training_h_source": "observed_training_h", "inference_h_source": "original_frozen_predicted_h",
             "training_area_source": "A_of_observed_training_h", "test_area_source": "A_of_predicted_h",
             "q_predictions_clipped": False, "input_error_distribution_shift": True,
             "n_area_train_basins": len(retained), "elapsed_fit_sec": time.monotonic() - started}
    return model, scale, audit


def evaluate(prepared, payload, scene, seed, scaling, variant, model, scale, audit, prediction, folder, run_id):
    ids, pos = prediction["basin_idx"], prediction["target_pos"]
    allowed_ids = {prepared.basin_index[b] for b in area.AREA_BASINS}
    if set(np.unique(ids)) != allowed_ids:
        raise ValueError("评价预测必须正好覆盖16站")
    donors = np.arange(len(prepared.basins))
    for source, filename in (("predicted_h", "metrics.csv"), ("observed_h_diagnostic", "diagnostic_metrics.csv")):
        h = prediction["pred_h_normalized"] if source == "predicted_h" else prepared.targets["waterlevel"][ids, pos]
        valid = np.isfinite(h)
        pairs = PairData(h[valid, None], np.zeros(valid.sum()), ids[valid], np.ones(valid.sum()), pos[valid])
        x = area.input_matrix(prepared, payload, pairs, variant, audit["final_area_scaler"], donors)
        values = np.full(len(ids), np.nan)
        values[valid] = model.predict(x) * scale.std["flow"][ids[valid]] + scale.mean["flow"][ids[valid]]
        frame = cohort.metrics16(prepared, payload, ids, pos, values, scene, seed, variant, scaling, source)
        frame["stage"], frame["estimator"], frame["alpha"] = "stage1_area_ridge_only16", "ridge", model.alpha
        _, _, outside, negative = area.area_values(prepared, payload, ids[valid], h[valid], donors)
        for basin in area.AREA_BASINS:
            selected = ids[valid] == prepared.basin_index[basin]
            frame.loc[frame.basin.eq(basin), "area_h_outside_fraction"] = float(outside[selected].mean()) if selected.any() else np.nan
            frame.loc[frame.basin.eq(basin), "area_negative_clipped_fraction"] = float(negative[selected].mean()) if selected.any() else np.nan
        frame.to_csv(folder / filename, index=False, encoding="utf-8-sig")
        if source == "predicted_h":
            if not np.isfinite(values).all():
                raise ValueError("正式预测含非有限Q")
            save_prediction(folder, ids, pos, values)
    subset = [prepared.basin_index[b] for b in area.AREA_BASINS]
    np.savez_compressed(folder / "model.npz", coef=model.coef, intercept=model.intercept, alpha=model.alpha,
        feature_names=np.asarray(audit["feature_names"]), flow_scale_basins=np.asarray(area.AREA_BASINS),
        flow_mean=scale.mean["flow"][subset], flow_std=scale.std["flow"][subset],
        attrs=prepared.attrs[subset], waterlevel_mean=prepared.target_mean["waterlevel"][subset],
        waterlevel_std=prepared.target_std["waterlevel"][subset], original86_index_basins=np.asarray(prepared.basins))
    pd.DataFrame({"feature": ["intercept", *audit["feature_names"]],
                  "coefficient": [model.intercept, *model.coef]}).to_csv(folder / "coefficients.csv", index=False, encoding="utf-8-sig")
    audit.update(run_fingerprint=run_id, ratio=scene["ratio"], mask_seed=scene["mask_seed"], model_seed=seed,
                 actual_q_hidden_ratio16=scene["actual_q_hidden_ratio16"], heldout=scene["heldout"],
                 evaluation_basins=list(area.AREA_BASINS), primary_evaluation_basins=scene["heldout"],
                 area_payload_fingerprint=payload["payload_fingerprint"], flow_scale_info=scale.info,
                 smoke_not_paper_result=False, waterlevel_models_retrained=0, stage2_executed=False)
    write_json(folder / "done.json", audit)


def summarize(output, experiment_id):
    frames, fits = [], []
    for path in sorted(output.glob("*/stage1/*/*/*/done.json")):
        meta = area.load_json(path)
        if meta["protocol"] != PROTOCOL:
            raise ValueError("汇总目录混入其他协议")
        for filename in ("metrics.csv", "diagnostic_metrics.csv"):
            frame = pd.read_csv(path.parent / filename, dtype={"basin": str})
            if len(frame) != 16 or set(frame.basin) != set(area.AREA_BASINS):
                raise ValueError("每项指标必须恰好包含16站")
            if set(frame.loc[frame.held_out, "basin"]) != set(meta["heldout"]):
                raise ValueError("指标中的隐藏站与拟合审计不一致")
            frames.append(frame)
        fits.append({k: meta[k] for k in ("target_scaling", "variant", "ratio", "mask_seed", "model_seed",
                                         "alpha", "input_dim", "n_parameters", "n_train_pairs", "elapsed_fit_sec")})
    if not frames:
        return 0
    data = pd.concat(frames, ignore_index=True)
    data.to_csv(output / "all_basin_metrics.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(fits).to_csv(output / "fit_audit.csv", index=False, encoding="utf-8-sig")
    rows = []
    keys = ["target_scaling", "variant", "ratio", "mask_seed", "model_seed", "h_input_source"]
    for values, frame in data.groupby(keys):
        scopes = {"all16": np.ones(len(frame), dtype=bool), "q_heldout16": frame.held_out,
                  "q_retained16": ~frame.held_out, "q_heldout_qualified11": frame.held_out & frame.area_analysis_group.eq("qualified11"),
                  "q_heldout_focus5": frame.held_out & frame.area_analysis_group.eq("focus5")}
        for scope, mask in scopes.items():
            group = frame[mask]
            rows.append({**dict(zip(keys, values)), "scope": scope, "n_basins": len(group), "n_valid_nse": group.nse.count(),
                         "nse_mean": group.nse.mean(), "nse_median": group.nse.median(),
                         "negative_nse_count": int((group.nse < 0).sum()), "nse_min": group.nse.min()})
    summary = pd.DataFrame(rows)
    summary.to_csv(output / "summary_by_run.csv", index=False, encoding="utf-8-sig")
    summary[summary.h_input_source.eq("predicted_h")].groupby(["target_scaling", "variant", "ratio", "scope"])[[
        "nse_mean", "nse_median", "negative_nse_count", "nse_min"]].mean().reset_index().to_csv(
        output / "group_summary.csv", index=False, encoding="utf-8-sig")
    paired_keys = ["target_scaling", "ratio", "mask_seed", "model_seed", "h_input_source", "basin"]
    baseline = data[data.variant.eq(VARIANTS[0])][paired_keys + ["nse"]].rename(columns={"nse": "baseline_nse"})
    paired = data[data.variant.eq(VARIANTS[1])].merge(baseline, on=paired_keys, validate="one_to_one")
    paired["delta_nse"] = paired.nse - paired.baseline_nse
    paired.to_csv(output / "paired_area_differences.csv", index=False, encoding="utf-8-sig")
    main = summary[summary.scope.eq("q_heldout16") & summary.h_input_source.eq("predicted_h")]
    join = ["target_scaling", "ratio", "mask_seed", "model_seed"]
    run_pairs = main[main.variant.eq(VARIANTS[1])].merge(main[main.variant.eq(VARIANTS[0])], on=join, suffixes=("_area", "_no_area"), validate="one_to_one")
    run_pairs["delta_nse_mean"] = run_pairs.nse_mean_area - run_pairs.nse_mean_no_area
    run_pairs.to_csv(output / "paired_run_differences.csv", index=False, encoding="utf-8-sig")
    comparisons = []
    for (scaling, ratio), group in run_pairs.groupby(["target_scaling", "ratio"]):
        scores = paired[paired.target_scaling.eq(scaling) & paired.ratio.eq(ratio) & paired.held_out & paired.h_input_source.eq("predicted_h")]
        comparisons.append({"target_scaling": scaling, "ratio": ratio, "n_paired_runs": len(group),
            "no_area_nse_mean": group.nse_mean_no_area.mean(), "area_nse_mean": group.nse_mean_area.mean(),
            "delta_nse_mean": group.delta_nse_mean.mean(), "n_area_better_runs": int((group.delta_nse_mean > 0).sum()),
            "no_area_station_nse_median": scores.baseline_nse.median(), "area_station_nse_median": scores.nse.median(),
            "paired_station_delta_median": scores.delta_nse.median(),
            "no_area_negative_nse_count": int((scores.baseline_nse < 0).sum()),
            "area_negative_nse_count": int((scores.nse < 0).sum()), "n_station_runs": len(scores)})
    comparison = pd.DataFrame(comparisons)
    comparison.to_csv(output / "primary_comparison.csv", index=False, encoding="utf-8-sig")
    worst = paired[paired.held_out & paired.h_input_source.eq("predicted_h")].sort_values("delta_nse")
    worst.head(30).to_csv(output / "worst_area_differences.csv", index=False, encoding="utf-8-sig")
    report = ["# 16站阶段1岭回归结果", "", f"完成{len(fits)}/{N_EXPECTED}个转换器；CPU岭回归，原86水位模型冻结。", "",
              "主成绩仅使用每组16站中的Q隐藏站、原test期、预测H。先逐站算NSE，再每次运行等权平均，最后平均9次运行。", "",
              "| 归一化 | 原86名义Q隐藏比例 | 无面积平均NSE | 有面积平均NSE | 差值 | 有面积更好的运行数 |", "| --- | ---: | ---: | ---: | ---: | ---: |"]
    for row in comparison.itertuples():
        report.append(f"| {row.target_scaling} | {row.ratio:.0%} | {row.no_area_nse_mean:.4f} | {row.area_nse_mean:.4f} | {row.delta_nse_mean:+.4f} | {row.n_area_better_runs}/{row.n_paired_runs} |")
    report += ["", "同时查看primary_comparison.csv中的逐站中位数、负NSE站次数，以及worst_area_differences.csv。", "",
               "3个model_seed分别对应冻结H模型及内部整站划分；岭回归求解本身没有随机初始化，9次运行也不是9份互不重叠的独立数据。", "",
               "physical只用Q保留站训练尺度回归；observed额外允许隐藏站训练Q均值/标准差，是已知尺度对照。", "",
               "所有16站和原面积曲线均保留，未根据测试Q删站、调正则搜索范围或截断预测Q。", "",
               "最少情景只有3个Q保留站，内部2训1验；流域属性有共线性，回归系数不应解释为因果效应。", "",
               "本轮只有岭回归，没有追加二次回归。16站专用神经网络尚无完整匹配结果，不能据此断言岭回归优于神经网络；旧86混合拟合结果训练总体不同。", "",
               "这是在查看旧实验测试结果后提出的新探索，当前测试集并非未曾查看的独立最终验证集。", ""]
    (output / "report.md").write_text("\n".join(report), encoding="utf-8")
    write_json(output / "summary_manifest.json", {"experiment_fingerprint": experiment_id, "protocol": PROTOCOL,
        "n_completed": len(fits), "n_expected": N_EXPECTED, "primary_scope": "q_heldout16", "all16_included": True,
        "selection_used_test_scores": False, "new_experiment_after_prior_test_inspection": True})
    return len(fits)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--preflight", action="store_true")
    args = parser.parse_args()
    output = args.out_dir.resolve()
    for source in (cohort.OLD_OUTPUT, cohort.DEFAULT_OUTPUT, *area.SOURCE_PATHS.values()):
        if output == source or source in output.parents or output in source.parents:
            raise ValueError("岭回归目录不能覆盖旧结果或原H模型")
    payload = area.load_curves(area.CURVE_FILE)
    prepared = PreparedData(attr_set="extended")
    specs = {name: area.load_json(path / "experiment.json") for name, path in area.SOURCE_PATHS.items()}
    base, predictions, counts = cohort.preflight(prepared, payload, specs)
    hashes = dict(base["code_hashes"])
    for path in (Path(__file__), ROOT / "tools/run_atgage16_area_ridge.py"):
        hashes[str(path.relative_to(ROOT))] = file_sha256(path)
    spec = {**base, "protocol": PROTOCOL, "variants": VARIANTS, "n_expected": N_EXPECTED, "code_hashes": hashes,
            "estimator": "numpy_weighted_ridge", "device": "cpu", "ridge_alpha_grid": ALPHAS,
            "ridge_objective": "weighted_mean_squared_error_plus_alpha_times_squared_slopes_unpenalized_intercept",
            "converter_hyperparameters_unchanged": False, "neural_networks_trained": 0,
            "local_execution_authorized_in_current_conversation": True,
            "feature_and_sampling_protocol_unchanged": True, "polynomial_features_added": False}
    experiment_id = fingerprint(spec)
    output.mkdir(parents=True, exist_ok=True)
    manifest = output / "experiment.json"
    if manifest.exists():
        previous = area.load_json(manifest)
        if previous["protocol"] != PROTOCOL or previous["experiment_fingerprint"] != experiment_id:
            if any(output.glob("*/stage1/*/*/*/model.npz")) or any(output.glob("*/stage1/*/*/*/done.json")):
                raise ValueError("已有不同代码或输入的岭回归结果，请使用新目录")
    write_json(manifest, {**spec, "experiment_fingerprint": experiment_id})
    write_json(output / "area_curves.json", payload)
    pd.DataFrame(counts).to_csv(output / "area_training_coverage.csv", index=False, encoding="utf-8-sig")
    write_json(output / "preflight.json", {"status": "PASS", "protocol": PROTOCOL, "n_expected": N_EXPECTED,
        "converter_basins": list(area.AREA_BASINS), "waterlevel_models_retrained": 0, "device": "cpu"})
    print(f"预检通过：16站、CPU岭回归、{N_EXPECTED}项；原H预测冻结", flush=True)
    if args.preflight:
        return 0
    started = time.monotonic()
    write_json(output / "process.json", {"pid": os.getpid(), "status": "running"})
    try:
        for scaling in ("physical", "observed"):
            maximum = specs[scaling]["settings"]["converter_max_pairs_per_basin"]
            for scene in spec["scenarios"]:
                for seed in (1, 2, 3):
                    for variant in VARIANTS:
                        folder = output / scaling / "stage1" / scene["tag"] / f"seed{seed}" / variant
                        folder.mkdir(parents=True, exist_ok=True)
                        run_id = fingerprint({"experiment": experiment_id, "scaling": scaling,
                            "scenario": scene["tag"], "seed": seed, "variant": variant})
                        done = folder / "done.json"
                        if done.exists():
                            meta = area.load_json(done)
                            if meta["protocol"] != PROTOCOL or meta["run_fingerprint"] != run_id:
                                raise ValueError("完成项协议或指纹不一致")
                            if not all((folder / f).is_file() for f in ("model.npz", "prediction.npz", "metrics.csv", "diagnostic_metrics.csv", "coefficients.csv")):
                                raise ValueError("完成项缺少必要文件")
                            continue
                        model, scale, audit = fit_converter(prepared, payload, scene, seed, scaling, variant, maximum)
                        evaluate(prepared, payload, scene, seed, scaling, variant, model, scale, audit,
                                 predictions[scaling][seed], folder, run_id)
                        completed = len(list(output.glob("*/stage1/*/*/*/done.json")))
                        write_json(output / "progress.json", {"n_completed": completed, "n_expected": N_EXPECTED,
                            "last_completed": str(folder.relative_to(output)), "elapsed_sec": time.monotonic() - started})
                        print(f"[{completed}/{N_EXPECTED}] {scaling} {scene['tag']} seed{seed} {variant} alpha={model.alpha:g}", flush=True)
                summarize(output, experiment_id)
        completed = summarize(output, experiment_id)
        if completed != N_EXPECTED:
            raise ValueError("正式运行的完成项数不是108")
        write_json(output / "completed.json", {"protocol": PROTOCOL, "experiment_fingerprint": experiment_id,
            "n_completed": completed, "elapsed_sec": time.monotonic() - started,
            "device": "cpu", "waterlevel_models_retrained": 0, "neural_networks_trained": 0, "stage2_executed": False})
        write_json(output / "process.json", {"pid": os.getpid(), "status": "completed", "n_completed": completed})
        print(f"完成{completed}/{N_EXPECTED}，报告：{output / 'report.md'}", flush=True)
        return 0
    except BaseException as exc:
        write_json(output / "process.json", {"pid": os.getpid(), "status": "failed", "error": str(exc)})
        raise


if __name__ == "__main__":
    raise SystemExit(main())
