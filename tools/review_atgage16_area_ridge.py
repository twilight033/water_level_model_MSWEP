"""独立重建岭回归预测、重算指标并检查108项的Q信息隔离；不重新拟合模型。"""

import argparse
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from cross_section_experiment.area_ridge16 import DEFAULT_OUTPUT, N_EXPECTED, PROTOCOL
from cross_section_experiment.features import file_sha256
from cross_section_experiment.protocol import fingerprint, write_json
from cross_section_experiment.area_stage1 import load_json
from pipeline.dataset import PreparedData


def reconstruct(model, audit, curves, ids, h):
    """直接按保存的系数和单位变换重建，避免复用实验的输入/预测函数。"""
    original = model["original86_index_basins"].tolist()
    stations = model["flow_scale_basins"].tolist()
    lookup = {original.index(b): i for i, b in enumerate(stations)}
    local = np.array([lookup[int(bi)] for bi in ids])
    h = np.asarray(h, dtype=np.float64)
    pieces = [h[:, None], model["attrs"][local]]
    if audit["variant"] == "h_attrs_area":
        waterlevel = h * model["waterlevel_std"][local] + model["waterlevel_mean"][local]
        intercept = np.array([curves[b]["intercept_m2"] for b in stations])
        slope = np.array([curves[b]["slope_m"] for b in stations])
        log_area = np.log1p(np.maximum(intercept[local] + slope[local] * waterlevel, 0))
        scaler = audit["final_area_scaler"]
        pieces.append(((log_area - scaler["mean_log1p_area"]) / scaler["std_log1p_area"])[:, None])
    pieces.append(np.ones((len(h), 1)))
    x = np.column_stack(pieces).astype(np.float32)
    q = (x.astype(np.float64) @ model["coef"] + float(model["intercept"]))
    return q * model["flow_std"][local] + model["flow_mean"][local]


def write_interpretation(output, prepared, spec, predictions, curves, validation):
    """事后诊断最差运行，不重拟合、不修改搜索范围，也不删除任何站。"""
    comparison = pd.read_csv(output / "primary_comparison.csv")
    runs = pd.read_csv(output / "paired_run_differences.csv")
    worst = runs.loc[runs.delta_nse_mean.idxmin()]
    tag = next(scene["tag"] for scene in spec["scenarios"]
               if scene["ratio"] == worst.ratio and scene["mask_seed"] == worst.mask_seed)
    seed, scaling = int(worst.model_seed), worst.target_scaling
    folder = output / scaling / "stage1" / tag / f"seed{seed}" / "h_attrs_area"
    audit = load_json(folder / "done.json")
    train_area = []
    for basin in audit["training_q_pair_basins"]:
        bi = prepared.basin_index[basin]
        lo, hi = prepared.split_range(basin, "train")
        h = prepared.targets["waterlevel"][bi, lo:hi + 1].astype(float)
        q = prepared.targets_raw["flow"][bi, lo:hi + 1]
        h = h[np.isfinite(h) & np.isfinite(q)]
        maximum = spec["source_settings"][scaling]["converter_max_pairs_per_basin"]
        if maximum and len(h) > maximum:
            h = h[np.unique(np.linspace(0, len(h) - 1, maximum).astype(int))]
        h_m = h * prepared.target_std["waterlevel"][bi] + prepared.target_mean["waterlevel"][bi]
        curve = curves[basin]
        train_area.append(np.maximum(curve["intercept_m2"] + curve["slope_m"] * h_m, 0))
    train_area = np.concatenate(train_area)
    lower, upper = float(train_area.min()), float(train_area.max())
    metrics = pd.read_csv(folder / "metrics.csv", dtype={"basin": str}).set_index("basin")
    diagnostic = pd.read_csv(folder / "diagnostic_metrics.csv", dtype={"basin": str}).set_index("basin")
    baseline = pd.read_csv(folder.parent / "h_attrs_area_available/metrics.csv", dtype={"basin": str}).set_index("basin")
    frozen = predictions[(scaling, seed)]
    rows = []
    for basin in spec["converter_basins"]:
        bi = prepared.basin_index[basin]
        h = frozen["pred_h_normalized"][frozen["basin_idx"] == bi].astype(float)
        h_m = h * prepared.target_std["waterlevel"][bi] + prepared.target_mean["waterlevel"][bi]
        curve = curves[basin]
        estimated = np.maximum(curve["intercept_m2"] + curve["slope_m"] * h_m, 0)
        rows.append({"basin": basin, "heldout": metrics.loc[basin, "held_out"],
            "baseline_nse": baseline.loc[basin, "nse"], "area_nse": metrics.loc[basin, "nse"],
            "observed_h_area_nse": diagnostic.loc[basin, "nse"],
            "median_estimated_area_m2": float(np.median(estimated)),
            "outside_training_area_value_range_fraction": float(((estimated < lower) | (estimated > upper)).mean()),
            "outside_own_curve_h_range_fraction": metrics.loc[basin, "area_h_outside_fraction"],
            "training_area_min_m2": lower, "training_area_max_m2": upper})
    case = pd.DataFrame(rows)
    case.to_csv(output / "worst_run_diagnostic.csv", index=False, encoding="utf-8-sig")
    lines = ["# 16站岭回归：已运行结果与判断", "", "本机CPU完成108/108项，未重新训练水位模型；未追加神经网络或二次回归。", "",
             f"独立复核通过：从系数重建全部预测，最大绝对差为{validation['max_absolute_prediction_reconstruction_error']:.3g}；"
             f"重算{validation['n_checked_station_metrics']}个站次的NSE/RMSE/MAE，最大指标差{validation['max_absolute_metric_recomputation_error']:.3g}。", "",
             "以下仅统计当组Q隐藏站的原测试期，使用预测H；每项是9次运行平均NSE。隐藏比例仍指原86的名义比例。", "",
             "| 归一化 | Q隐藏比例 | 无面积 | 有面积 | 增量 | 面积更好次数 |", "| --- | ---: | ---: | ---: | ---: | ---: |"]
    for row in comparison.sort_values(["target_scaling", "ratio"], ascending=[False, True]).itertuples():
        lines.append(f"| {row.target_scaling} | {row.ratio:.0%} | {row.no_area_nse_mean:.4f} | {row.area_nse_mean:.4f} | {row.delta_nse_mean:+.4f} | {row.n_area_better_runs}/{row.n_paired_runs} |")
    lines += ["", "普通岭回归能完成跨站转换，但新增面积没有表现出普遍、稳定的提升。不能将这些分数解释成已超过同16站神经网络，因为其匹配运行目前没有完整结果。", "",
              "## 最差运行的诊断", "",
              f"最差面积增量出现在{scaling}、{tag}、seed{seed}：无面积平均NSE={worst.nse_mean_no_area:.4f}，"
              f"有面积={worst.nse_mean_area:.4f}，负NSE站数从{int(worst.negative_nse_count_no_area)}变成{int(worst.negative_nse_count_area)}。", "",
              f"该组最终保留站{len(audit['training_q_pair_basins'])}个，内部训练{len(audit['inner_train_basins'])}站、验证{len(audit['inner_validation_basins'])}站；"
              f"按内部验证选择alpha={audit['alpha']:g}，不是按这次测试失败后重选。", "",
              f"该组实际拟合样本的面积取值范围是{lower:.2f}—{upper:.2f}平方米。这里检查的是面积数值跨站外推，"
              "应与本站H是否超过自身面积曲线测量范围分开。", "",
              "| 隐藏站 | 无面积NSE | 有面积NSE | 改用观测H的面积模型NSE | 估算面积中位数/m² | 超过训练面积值域比例 |",
              "| --- | ---: | ---: | ---: | ---: | ---: |"]
    for row in case[case.heldout].sort_values("area_nse").head(5).itertuples():
        lines.append(f"| {row.basin} | {row.baseline_nse:.4f} | {row.area_nse:.4f} | {row.observed_h_area_nse:.4f} | {row.median_estimated_area_m2:.2f} | {row.outside_training_area_value_range_fraction:.1%} |")
    lines += ["", "观测H诊断失败的站不能仅归因于上游水位预测误差；少量站学习到的面积—流量关系跨站外推是需要关注的解释。"
              "这些是事后诊断，不能据此认定单一因果机制，也没有据此修改模型。", "",
              "physical70%的总体下降需结合paired_run_differences.csv核对，不能概括成每个站或每次运行都下降。"
              "逐站中位数、负NSE站次数和面积改善次数保存在primary_comparison.csv。", "",
              "## 结论的边界", "",
              "这轮证明了普通回归在技术上可实现，面积价值还缺乏稳定证据；它不能单独证明神经网络有必要，也不能单独证明普通回归更优。", "",
              "observed允许隐藏站训练Q尺度，只能作为已知尺度对照。所有16站及原曲线均保留。"
              "原测试集已在多轮探索中查看，需另行独立验证才能支持最终模型选择结论。", ""]
    (output / "analysis.md").write_text("\n".join(lines), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    output = args.out_dir.resolve()
    spec = load_json(output / "experiment.json")
    completed = load_json(output / "completed.json")
    assert spec["protocol"] == completed["protocol"] == PROTOCOL
    assert completed["n_completed"] == N_EXPECTED
    assert fingerprint({k: v for k, v in spec.items() if k != "experiment_fingerprint"}) == spec["experiment_fingerprint"]
    for filename, expected in spec["code_hashes"].items():
        assert file_sha256(ROOT / filename) == expected, f"代码已改变: {filename}"
    prepared = PreparedData(attr_set="extended")
    for filename, expected in spec["data_hashes"].items():
        assert file_sha256(ROOT / filename) == expected, f"数据已改变: {filename}"
    stations = spec["converter_basins"]
    allowed = {prepared.basin_index[b] for b in stations}
    predictions = {}
    for scaling, source in spec["sources"].items():
        for seed in (1, 2, 3):
            folder = Path(source["source_dir"]) / "waterlevel_models" / f"seed{seed}"
            expected = source["H_predictions"][str(seed)]
            assert file_sha256(folder / "prediction.npz") == expected["prediction_sha256"]
            assert file_sha256(folder / "model.pt") == expected["model_sha256"]
            with np.load(folder / "prediction.npz") as data:
                take = np.isin(data["basin_idx"], list(allowed))
                predictions[(scaling, seed)] = {key: data[key][take] for key in data.files}
    curves = load_json(output / "area_curves.json")["curves"]
    seen, errors, pairs = set(), [], {}
    max_prediction_error, max_metric_error = 0., 0.
    for path in sorted(output.glob("*/stage1/*/*/*/done.json")):
        meta = load_json(path)
        scaling, seed, variant = meta["target_scaling"], meta["model_seed"], meta["variant"]
        tag = path.parent.parent.parent.name
        scene = next(s for s in spec["scenarios"] if s["tag"] == tag)
        key = (scaling, tag, seed, variant)
        assert key not in seen
        seen.add(key)
        assert meta["protocol"] == PROTOCOL and meta["training_q_pair_basins"] == scene["retained"]
        assert meta["run_fingerprint"] == fingerprint({"experiment": spec["experiment_fingerprint"], "scaling": scaling,
            "scenario": tag, "seed": seed, "variant": variant})
        train, valid = set(meta["inner_train_basins"]), set(meta["inner_validation_basins"])
        assert not train & valid and train | valid == set(scene["retained"])
        assert all(set(call["fit_basins"]) <= set(scene["retained"]) for call in meta["q_scaling_fit_calls"])
        assert set(meta["q_scaling_fit_calls"][0]["fit_basins"]) == train
        assert set(meta["q_scaling_fit_calls"][1]["fit_basins"]) == set(scene["retained"])
        chosen = max(meta["candidates"], key=lambda row: (row["valid_basin_nse"], row["alpha"]))
        assert chosen["alpha"] == meta["alpha"]
        pairs.setdefault((scaling, tag, seed), []).append((meta["inner_train_basins"], meta["inner_validation_basins"]))
        if variant == "h_attrs_area":
            assert set(meta["inner_area_scaler"]["fit_basins"]) == train
            assert set(meta["final_area_scaler"]["fit_basins"]) == set(scene["retained"])
        frozen = predictions[(scaling, seed)]
        ids, positions = frozen["basin_idx"], frozen["target_pos"]
        with np.load(path.parent / "model.npz") as model, np.load(path.parent / "prediction.npz") as saved:
            assert set(np.unique(saved["basin_idx"])) == allowed
            np.testing.assert_array_equal(saved["basin_idx"], ids)
            np.testing.assert_array_equal(saved["target_pos"], positions)
            assert model["flow_scale_basins"].tolist() == stations
            np.testing.assert_array_equal(model["attrs"], prepared.attrs[[prepared.basin_index[b] for b in stations]])
            for source, filename in (("predicted_h", "metrics.csv"), ("observed_h_diagnostic", "diagnostic_metrics.csv")):
                h = frozen["pred_h_normalized"] if source == "predicted_h" else prepared.targets["waterlevel"][ids, positions]
                predicted = reconstruct(model, meta, curves, ids, h)
                if source == "predicted_h":
                    delta = float(np.max(np.abs(predicted - saved["pred_flow"])))
                    max_prediction_error = max(max_prediction_error, delta)
                    np.testing.assert_allclose(predicted, saved["pred_flow"], atol=1e-8, rtol=1e-10)
                frame = pd.read_csv(path.parent / filename, dtype={"basin": str}).set_index("basin")
                assert len(frame) == 16 and set(frame.index) == set(stations)
                assert set(frame.index[frame.held_out]) == set(scene["heldout"])
                for basin in stations:
                    bi = prepared.basin_index[basin]
                    mask = ids == bi
                    obs = prepared.targets_raw["flow"][bi, positions[mask]].astype(float)
                    pred = predicted[mask]
                    valid_pair = np.isfinite(obs) & np.isfinite(pred)
                    observed, estimated = obs[valid_pair], pred[valid_pair]
                    assert len(observed) == frame.loc[basin, "n"]
                    assert int((pred[np.isfinite(pred)] < 0).sum()) == frame.loc[basin, "n_negative_predictions"]
                    denominator = np.sum((observed - observed.mean()) ** 2)
                    calculated = {"nse": 1 - np.sum((observed - estimated) ** 2) / denominator,
                                  "rmse": np.sqrt(np.mean((observed - estimated) ** 2)),
                                  "mae": np.mean(np.abs(observed - estimated))}
                    for metric, actual in calculated.items():
                        reported = frame.loc[basin, metric]
                        difference = abs(actual - reported)
                        max_metric_error = max(max_metric_error, difference)
                        assert np.isclose(actual, reported, atol=2e-5, rtol=2e-6), (key, basin, metric, actual, reported)
                    errors.append({"target_scaling": scaling, "scenario": tag, "model_seed": seed,
                        "variant": variant, "source": source, "basin": basin,
                        "nse_recomputed": calculated["nse"], "nse_reported": frame.loc[basin, "nse"]})
    expected = {(s, c["tag"], seed, variant) for s in spec["scalings"] for c in spec["scenarios"]
                for seed in (1, 2, 3) for variant in spec["variants"]}
    assert seen == expected and len(seen) == N_EXPECTED
    assert all(len(pair) == 2 and pair[0] == pair[1] for pair in pairs.values())
    pd.DataFrame(errors).to_csv(output / "independent_metric_checks.csv", index=False, encoding="utf-8-sig")
    result = {"status": "PASS", "n_complete_models": len(seen), "n_checked_station_metrics": len(errors),
        "predictions_reconstructed_without_experiment_feature_builder": True,
        "max_absolute_prediction_reconstruction_error": max_prediction_error,
        "max_absolute_metric_recomputation_error": max_metric_error,
        "frozen_h_sources_unchanged": True, "q_retained16_isolation_passed": True,
        "paired_inputs_use_identical_inner_splits": True, "test_not_used_for_alpha_selection": True,
        "float32_original_metric_rounding_tolerance": 2e-5}
    write_json(output / "validation.json", result)
    write_interpretation(output, prepared, spec, predictions, curves, result)
    print(result)
    print(pd.read_csv(output / "primary_comparison.csv").to_string(index=False))


if __name__ == "__main__":
    main()
