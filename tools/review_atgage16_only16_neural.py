"""核验拷回的16站神经网络结果，并与同信息口径岭回归比较；不训练或修改模型。"""

from __future__ import annotations

import hashlib
import itertools
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from cross_section_experiment.area_stage1 import AREA_BASINS, FOCUS_BASINS, load_json
from cross_section_experiment.area_stage1_only16 import DEFAULT_OUTPUT as SOURCE, PROTOCOL
from cross_section_experiment.area_ridge16 import DEFAULT_OUTPUT as RIDGE
from cross_section_experiment.converters import collect_pairs
from cross_section_experiment.protocol import fingerprint, write_json
from pipeline.dataset import PreparedData

OUTPUT = ROOT / "reports/atgage16_only16_neural_review"
BASE, AREA = "h_attrs_area_available", "h_attrs_area"
KEYS = ["target_scaling", "ratio", "mask_seed", "model_seed", "h_input_source", "basin"]


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def check_hash(path, expected, line_endings=False):
    if digest(path) == expected:
        return "exact"
    if line_endings:
        raw = Path(path).read_bytes().replace(b"\r\n", b"\n")
        if expected in (hashlib.sha256(raw).hexdigest(), hashlib.sha256(raw.replace(b"\n", b"\r\n")).hexdigest()):
            return "lf_crlf_only"
    raise AssertionError(f"来源文件内容不一致: {path}")


def features(prepared, curves, ids, h, variant, scaler, donors=None):
    """独立拼接输入，仅在16站取值，不调用实验的input_matrix。"""
    h = np.asarray(h)
    pieces = [h[:, None]]
    if variant != "h_only":
        pieces.append(prepared.attrs[ids])
    if variant in (AREA, "h_attrs_area_shuffled"):
        donor_ids = np.array([prepared.basin_index[(donors or {}).get(prepared.basins[int(bi)], prepared.basins[int(bi)])] for bi in ids])
        h_m = h.astype(float) * prepared.target_std["waterlevel"][donor_ids] + prepared.target_mean["waterlevel"][donor_ids]
        a = np.array([curves[prepared.basins[int(bi)]]["intercept_m2"] for bi in donor_ids])
        b = np.array([curves[prepared.basins[int(bi)]]["slope_m"] for bi in donor_ids])
        log_area = np.log1p(np.maximum(a + b * h_m, 0))
        pieces.append(((log_area - scaler["mean_log1p_area"]) / scaler["std_log1p_area"])[:, None])
    if variant not in ("h_only", "h_attrs"):
        pieces.append(np.ones((len(h), 1)))
    return np.column_stack(pieces).astype(np.float32)


def neural_prediction(checkpoint, x):
    weights = checkpoint["state_dict"]
    w0, b0, w1, b1 = (weights[k].cpu().numpy() for k in ("net.0.weight", "net.0.bias", "net.2.weight", "net.2.bias"))
    return (np.maximum(x @ w0.T + b0, 0) @ w1.T + b1).reshape(-1)


def mean_basin_nse(obs, pred, ids):
    scores = []
    for bi in np.unique(ids):
        selected = ids == bi
        y, estimate = np.asarray(obs[selected], float), np.asarray(pred[selected], float)
        valid = np.isfinite(y) & np.isfinite(estimate)
        y, estimate = y[valid], estimate[valid]
        denom = np.square(y - y.mean()).sum()
        if len(y) >= 2 and denom > 0:
            scores.append(1 - np.square(y - estimate).sum() / denom)
    return float(np.mean(scores))


def audit():
    spec, ridge_spec = load_json(SOURCE / "experiment.json"), load_json(RIDGE / "experiment.json")
    experiment_id = spec["experiment_fingerprint"]
    assert spec["protocol"] == PROTOCOL and spec["n_expected"] == 324
    assert fingerprint({k: v for k, v in spec.items() if k != "experiment_fingerprint"}) == experiment_id
    for filename in ("completed.json", "process.json", "summary_manifest.json"):
        meta = load_json(SOURCE / filename)
        assert meta["experiment_fingerprint"] == experiment_id and meta["n_completed"] == 324
    line_endings = []
    for filename, expected in spec["code_hashes"].items():
        if check_hash(ROOT / filename, expected, True) == "lf_crlf_only":
            line_endings.append(filename)
    data_line_endings = []
    for filename, expected in spec["data_hashes"].items():
        if check_hash(ROOT / filename, expected, Path(filename).suffix == ".txt") == "lf_crlf_only":
            data_line_endings.append("neural:" + filename)
    for field in ("converter_basins", "area_basins", "focus_basins", "scenarios", "inner_splits", "area_payload_fingerprint"):
        assert spec[field] == ridge_spec[field], f"神经网络和岭回归的数据口径不同: {field}"
    assert set(spec["data_hashes"]) == set(ridge_spec["data_hashes"])
    for filename, expected in ridge_spec["data_hashes"].items():
        mode = check_hash(ROOT / filename, expected, filename == "86basin_ids.txt")
        if mode == "lf_crlf_only":
            data_line_endings.append("ridge:" + filename)
    curves = load_json(SOURCE / "area_curves.json")
    assert fingerprint({k: v for k, v in curves.items() if k != "payload_fingerprint"}) == curves["payload_fingerprint"] == spec["area_payload_fingerprint"]
    assert digest(SOURCE / "area_curves.json") == digest(RIDGE / "area_curves.json")
    prepared = PreparedData(attr_set="extended")
    assert prepared.basins == spec["waterlevel_base_basins"] and len(prepared.basins) == 86
    station_ids = [prepared.basin_index[b] for b in AREA_BASINS]
    local_index = {bi: i for i, bi in enumerate(station_ids)}
    frozen = {}
    for scaling in ("physical", "observed"):
        for seed in (1, 2, 3):
            folder = ROOT / "results" / f"cross_section86_strong_{scaling}_snapshot_v2/waterlevel_models/seed{seed}"
            item = spec["sources"][scaling]["H_predictions"][str(seed)]
            counterpart = ridge_spec["sources"][scaling]["H_predictions"][str(seed)]
            assert item["prediction_sha256"] == counterpart["prediction_sha256"]
            check_hash(folder / "prediction.npz", item["prediction_sha256"])
            check_hash(folder / "model.pt", item["model_sha256"])
            with np.load(folder / "prediction.npz") as prediction:
                take = np.isin(prediction["basin_idx"], station_ids)
                frozen[(scaling, seed)] = {k: prediction[k][take] for k in prediction.files}
    data = pd.read_csv(SOURCE / "all_basin_metrics.csv", dtype={"basin": str})
    assert len(data) == 324 * 16 * 2 and not data.duplicated(KEYS + ["variant"]).any()
    scenarios = {s["tag"]: s for s in spec["scenarios"]}
    expected = set(itertools.product(spec["scalings"], scenarios, (1, 2, 3), spec["variants"]))
    seen, models, diagnostics = set(), [], []
    checked, reconstructed, max_metric_error, max_prediction_error = 0, 0, 0., 0.
    done_files = sorted(SOURCE.glob("*/stage1/*/*/*/done.json"))
    assert len(done_files) == 324
    print("来源、冻结H、16站名单和岭回归配对口径通过；开始逐模型核验……", flush=True)
    for index, done in enumerate(done_files):
        scaling, _, tag, seed_dir, variant = done.relative_to(SOURCE).parts[:-1]
        seed = int(seed_dir[4:])
        run, scene, meta = (scaling, tag, seed, variant), scenarios[tag], load_json(done)
        assert run in expected and run not in seen
        seen.add(run)
        run_id = fingerprint({"experiment": experiment_id, "scaling": scaling, "scenario": tag, "seed": seed, "variant": variant})
        assert meta["protocol"] == PROTOCOL and meta["run_fingerprint"] == run_id
        assert meta["training_q_pair_basins"] == scene["retained"] and meta["heldout"] == scene["heldout"]
        assert set(meta["evaluation_basins"]) == set(AREA_BASINS) and meta["primary_evaluation_basins"] == scene["heldout"]
        train, valid = set(meta["inner_train_basins"]), set(meta["inner_validation_basins"])
        assert not train & valid and train | valid == set(scene["retained"])
        assert not (train | valid) & set(scene["heldout"])
        assert set(meta["converter_population"]) == set(AREA_BASINS)
        assert meta["observed_q_scale_basins"] == (list(AREA_BASINS) if scaling == "observed" else [])
        assert meta["scale_fit_basins"] == (scene["retained"] if scaling == "physical" else [])
        assert [set(c["fit_basins"]) for c in meta["q_scaling_fit_calls"]] == [train, set(scene["retained"])]
        for field, allowed in (("inner_area_scaler", train), ("final_area_scaler", set(scene["retained"]))):
            if meta[field] is not None:
                assert set(meta[field]["fit_basins"]) == allowed
        selected = max(meta["candidates"], key=lambda c: c["best_score"])
        assert selected["width"] == meta["hidden"] and selected["best_epoch"] == meta["epochs_refit"]
        assert selected["best_score"] == max(h["valid_basin_nse"] for h in selected["history"])
        assert meta["converter_batch_size"] == 2048 and meta["learning_rate"] == .001 and meta["weight_decay"] == .0001
        checkpoint = torch.load(done.parent / "model.pt", map_location="cpu", weights_only=False)
        assert checkpoint["protocol"] == PROTOCOL and checkpoint["run_fingerprint"] == run_id
        assert checkpoint["original86_index_basins"] == prepared.basins and checkpoint["flow_scale_basins"] == list(AREA_BASINS)
        assert checkpoint["training_q_pair_basins"] == scene["retained"]
        assert checkpoint["area_payload_fingerprint"] == spec["area_payload_fingerprint"]
        assert checkpoint["area_curves"] == curves["curves"] and checkpoint["area_scaler"] == meta["final_area_scaler"]
        assert checkpoint["donor_basins"] == meta["donor_basins"]
        n_parameters = sum(v.numel() for v in checkpoint["state_dict"].values())
        assert n_parameters == meta["n_parameters"] == meta["hidden"] * (meta["input_dim"] + 2) + 1
        if variant in ("h_attrs_area_shuffled", "h_attrs_capacity_control"):
            reference = load_json(done.parent.parent / AREA / "done.json")
            if variant.endswith("_shuffled"):
                assert meta["hidden"] == reference["hidden"]
                assert all(b != d for b, d in meta["donor_basins"].items())
            else:
                assert 0 <= n_parameters - reference["n_parameters"] < meta["input_dim"] + 2
        original = frozen[(scaling, seed)]
        ids, positions = original["basin_idx"], original["target_pos"]
        local = np.array([local_index[int(bi)] for bi in ids])
        with np.load(done.parent / "prediction.npz") as prediction:
            assert set(np.unique(prediction["basin_idx"])) == set(station_ids)
            np.testing.assert_array_equal(prediction["basin_idx"], ids)
            np.testing.assert_array_equal(prediction["target_pos"], positions)
            predicted = prediction["pred_flow"].astype(float)
        assert np.isfinite(predicted).all()
        sample = np.unique(np.linspace(0, len(ids) - 1, 1024).astype(int))
        x = features(prepared, curves["curves"], ids[sample], original["pred_h_normalized"][sample], variant,
                     meta["final_area_scaler"], meta["donor_basins"])
        reconstructed_z = neural_prediction(checkpoint, x)
        saved_z = (predicted[sample] - checkpoint["flow_mean"][local[sample]]) / checkpoint["flow_std"][local[sample]]
        prediction_error = float(np.max(np.abs(reconstructed_z - saved_z)))
        max_prediction_error = max(max_prediction_error, prediction_error)
        assert np.allclose(reconstructed_z, saved_z, rtol=3e-5, atol=3e-4), (run, prediction_error)
        reconstructed += len(sample)
        raw = pd.read_csv(done.parent / "metrics.csv", dtype={"basin": str}).set_index("basin")
        assert len(raw) == 16 and set(raw.index) == set(AREA_BASINS)
        assert set(raw.index[raw.held_out]) == set(scene["heldout"])
        for basin, bi in zip(AREA_BASINS, station_ids):
            mask = ids == bi
            y, estimate = prepared.targets_raw["flow"][bi, positions[mask]].astype(float), predicted[mask]
            ok = np.isfinite(y) & np.isfinite(estimate)
            y, estimate = y[ok], estimate[ok]
            assert len(y) == raw.loc[basin, "n"]
            scores = {"nse": 1 - np.square(y - estimate).sum() / np.square(y - y.mean()).sum(),
                      "rmse": np.sqrt(np.square(y - estimate).mean()), "mae": np.abs(y - estimate).mean()}
            for metric, calculated in scores.items():
                stored = raw.loc[basin, metric]
                max_metric_error = max(max_metric_error, abs(calculated - stored))
                assert np.isclose(calculated, stored, rtol=2e-6, atol=1e-5), (run, basin, metric, calculated, stored)
            # 原指标按全部预测计负值数，包含对应Q缺测的时刻，不只统计有效观测配对。
            assert int((predicted[mask] < 0).sum()) == raw.loc[basin, "n_negative_predictions"]
            checked += 1
        for source, filename in (("predicted_h", "metrics.csv"), ("observed_h_diagnostic", "diagnostic_metrics.csv")):
            original_rows = pd.read_csv(done.parent / filename, dtype={"basin": str}).set_index("basin")
            group = data[data.target_scaling.eq(scaling) & data.ratio.eq(scene["ratio"]) & data.mask_seed.eq(scene["mask_seed"])
                         & data.model_seed.eq(seed) & data.variant.eq(variant) & data.h_input_source.eq(source)].set_index("basin")
            assert set(group.index) == set(AREA_BASINS)
            for column in ("nse", "rmse", "mae", "n", "n_negative_predictions"):
                np.testing.assert_allclose(group.loc[list(AREA_BASINS), column], original_rows.loc[list(AREA_BASINS), column], equal_nan=True)
        models.append({"target_scaling": scaling, "scenario": tag, "ratio": scene["ratio"], "mask_seed": scene["mask_seed"],
            "model_seed": seed, "variant": variant, "hidden": meta["hidden"], "n_parameters": n_parameters,
            "epochs_refit": meta["epochs_refit"], "inner_validation_best_nse": selected["best_score"],
            "best_epoch_at_50": selected["best_epoch"] == 50,
            "validation_change_last5_epochs": selected["history"][-1]["valid_basin_nse"] - selected["history"][-6]["valid_basin_nse"],
            "refit_loss_first": meta["refit"]["history"][0]["loss"], "refit_loss_last": meta["refit"]["history"][-1]["loss"],
            "n_retained": len(scene["retained"]), "n_inner_valid": len(valid)})
        if variant in (BASE, AREA):
            ridge_folder = RIDGE / scaling / "stage1" / tag / f"seed{seed}" / variant
            ridge_meta = load_json(ridge_folder / "done.json")
            for field in ("training_q_pair_basins", "inner_train_basins", "inner_validation_basins"):
                assert meta[field] == ridge_meta[field]
            for field in ("inner_area_scaler", "final_area_scaler"):
                if meta[field] is not None:
                    for name in ("mean_log1p_area", "std_log1p_area"):
                        assert np.isclose(meta[field][name], ridge_meta[field][name], atol=1e-6)
            with np.load(ridge_folder / "model.npz") as ridge_model:
                assert ridge_model["flow_scale_basins"].tolist() == list(AREA_BASINS)
                np.testing.assert_allclose(checkpoint["flow_mean"], ridge_model["flow_mean"], rtol=1e-6, atol=1e-6)
                np.testing.assert_allclose(checkpoint["flow_std"], ridge_model["flow_std"], rtol=1e-6, atol=1e-6)
                scale = SimpleNamespace(mean={"flow": np.full(86, np.nan)}, std={"flow": np.full(86, np.nan)})
                scale.mean["flow"][station_ids], scale.std["flow"][station_ids] = checkpoint["flow_mean"], checkpoint["flow_std"]
                for split in ("train", "valid"):
                    pairs = collect_pairs(prepared, scene["retained"], split, scale, None,
                        spec["source_settings"][scaling]["converter_max_pairs_per_basin"])
                    x = features(prepared, curves["curves"], pairs.basin, pairs.x[:, 0], variant, meta["final_area_scaler"])
                    ridge_x = features(prepared, curves["curves"], pairs.basin, pairs.x[:, 0], variant, ridge_meta["final_area_scaler"])
                    nn_pred = neural_prediction(checkpoint, x)
                    linear_pred = ridge_x.astype(float) @ ridge_model["coef"] + float(ridge_model["intercept"])
                    diagnostics.append({"target_scaling": scaling, "ratio": scene["ratio"], "mask_seed": scene["mask_seed"],
                        "model_seed": seed, "variant": variant, "split": split, "n_basins": len(scene["retained"]),
                        "nn_final_model_nse": mean_basin_nse(pairs.y, nn_pred, pairs.basin),
                        "ridge_final_model_nse": mean_basin_nse(pairs.y, linear_pred, pairs.basin)})
        if (index + 1) % 54 == 0:
            print(f"已核验{index + 1}/324个模型，未训练任何模型", flush=True)
    assert seen == expected
    pd.DataFrame(models).to_csv(OUTPUT / "model_audit.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(diagnostics).to_csv(OUTPUT / "retained_fit_diagnostics.csv", index=False, encoding="utf-8-sig")
    status = {"status": "PASS", "n_completed_models": 324, "formal_station_metrics_recomputed": checked,
        "sampled_checkpoint_predictions_reconstructed": reconstructed, "max_sampled_normalized_prediction_error": max_prediction_error,
        "max_absolute_metric_recomputation_error": max_metric_error, "code_hash_differences_only_lf_crlf": line_endings,
        "data_hash_differences_only_lf_crlf": data_line_endings,
        "original_h_sources_unchanged": True, "q_retained16_pair_and_scaling_isolation_passed": True,
        "ridge_same_splits_inputs_h_predictions_and_flow_scales_verified": True, "models_retrained": 0,
        "neural_environment": spec["numerical_environment"], "ridge_environment": ridge_spec["numerical_environment"]}
    write_json(OUTPUT / "audit.json", status)
    return spec, data, pd.DataFrame(models), pd.DataFrame(diagnostics), status


def analyze(spec, data, models, diagnostics, audit_result):
    ridge = pd.read_csv(RIDGE / "all_basin_metrics.csv", dtype={"basin": str})
    baseline = data[data.variant.eq(BASE)][KEYS + ["nse"]].rename(columns={"nse": "baseline_nse"})
    area = data[data.variant.eq(AREA)].merge(baseline, on=KEYS, validate="one_to_one")
    area["delta_nse"] = area.nse - area.baseline_nse
    area.to_csv(OUTPUT / "paired_neural_area_differences.csv", index=False, encoding="utf-8-sig")
    join = KEYS + ["variant"]
    matched = data[data.variant.isin((BASE, AREA))].merge(ridge[join + ["nse"]], on=join, validate="one_to_one", suffixes=("_nn", "_ridge"))
    matched["delta_nn_minus_ridge"] = matched.nse_nn - matched.nse_ridge
    matched.to_csv(OUTPUT / "ridge_vs_neural_by_station.csv", index=False, encoding="utf-8-sig")
    run_keys = ["target_scaling", "ratio", "mask_seed", "model_seed", "variant", "h_input_source"]
    main = matched[matched.held_out & matched.h_input_source.eq("predicted_h")]
    runs = main.groupby(run_keys)[["nse_nn", "nse_ridge", "delta_nn_minus_ridge"]].mean().reset_index()
    runs.to_csv(OUTPUT / "ridge_vs_neural_by_run.csv", index=False, encoding="utf-8-sig")
    comparisons, controls = [], []
    for (scaling, ratio), group in main.groupby(["target_scaling", "ratio"]):
        no_area, with_area = group[group.variant.eq(BASE)], group[group.variant.eq(AREA)]
        run = runs[runs.target_scaling.eq(scaling) & runs.ratio.eq(ratio)]
        plain_run, area_run = run[run.variant.eq(BASE)], run[run.variant.eq(AREA)]
        paired = area[area.target_scaling.eq(scaling) & area.ratio.eq(ratio) & area.held_out & area.h_input_source.eq("predicted_h")]
        changes = paired.groupby(["mask_seed", "model_seed"])["delta_nse"].mean()
        h_only = data[data.target_scaling.eq(scaling) & data.ratio.eq(ratio) & data.variant.eq("h_only")
                      & data.held_out & data.h_input_source.eq("predicted_h")]
        comparisons.append({"target_scaling": scaling, "ratio": ratio,
            "nn_h_only_nse": h_only.groupby(["mask_seed", "model_seed"]).nse.mean().mean(),
            "ridge_no_area_nse": plain_run.nse_ridge.mean(), "ridge_area_nse": area_run.nse_ridge.mean(),
            "nn_no_area_nse": plain_run.nse_nn.mean(), "nn_area_nse": area_run.nse_nn.mean(),
            "nn_area_delta": changes.mean(), "nn_area_better_runs": int((changes > 0).sum()), "n_paired_runs": len(changes),
            "nn_no_area_median": no_area.nse_nn.median(), "nn_area_median": with_area.nse_nn.median(),
            "nn_no_area_negative_count": int((no_area.nse_nn < 0).sum()), "nn_area_negative_count": int((with_area.nse_nn < 0).sum()),
            "n_station_runs": len(with_area), "nn_no_area_beats_ridge_runs": int((plain_run.delta_nn_minus_ridge > 0).sum()),
            "nn_area_beats_ridge_runs": int((area_run.delta_nn_minus_ridge > 0).sum())})
        subset = data[data.target_scaling.eq(scaling) & data.ratio.eq(ratio) & data.held_out & data.h_input_source.eq("predicted_h")]
        for variant, scores in subset.groupby("variant"):
            controls.append({"target_scaling": scaling, "ratio": ratio, "variant": variant,
                "mean_run_nse": scores.groupby(["mask_seed", "model_seed"]).nse.mean().mean(),
                "station_nse_median": scores.nse.median(), "negative_nse_count": int((scores.nse < 0).sum()),
                "worst_nse": scores.nse.min()})
    comparison = pd.DataFrame(comparisons).sort_values(["target_scaling", "ratio"], ascending=[False, True])
    comparison.to_csv(OUTPUT / "primary_comparison.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(controls).to_csv(OUTPUT / "six_variant_controls.csv", index=False, encoding="utf-8-sig")
    worst = area[area.held_out & area.h_input_source.eq("predicted_h")].sort_values("delta_nse")
    diag = area[area.h_input_source.eq("observed_h_diagnostic")][[k for k in KEYS if k != "h_input_source"] + ["nse"]].rename(columns={"nse": "observed_h_area_nse"})
    worst = worst.merge(diag, on=[k for k in KEYS if k != "h_input_source"], validate="one_to_one")
    worst.head(30).to_csv(OUTPUT / "worst_station_runs.csv", index=False, encoding="utf-8-sig")
    fit = models[models.variant.isin((BASE, AREA))]
    convergence = fit.groupby(["target_scaling", "variant"]).agg(n_models=("hidden", "size"),
        mean_hidden=("hidden", "mean"), selected_width16_count=("hidden", lambda s: int((s == 16).sum())),
        selected_epoch50_count=("best_epoch_at_50", "sum"), median_selected_epoch=("epochs_refit", "median"),
        median_inner_validation_nse=("inner_validation_best_nse", "median"),
        median_refit_loss_last=("refit_loss_last", "median")).reset_index()
    convergence.to_csv(OUTPUT / "convergence_summary.csv", index=False, encoding="utf-8-sig")
    fit_summary = diagnostics.groupby(["target_scaling", "variant", "split"])[["nn_final_model_nse", "ridge_final_model_nse"]].agg(["mean", "median"])
    fit_summary.to_csv(OUTPUT / "retained_fit_summary.csv", encoding="utf-8-sig")
    geometry = data[data.held_out & data.h_input_source.eq("predicted_h") & data.variant.isin((BASE, AREA))]
    geometry_runs = geometry.groupby(["target_scaling", "ratio", "mask_seed", "model_seed", "variant", "area_analysis_group"]).nse.mean().reset_index()
    geometry_runs.groupby(["target_scaling", "ratio", "variant", "area_analysis_group"]).nse.agg(["mean", "count"]).reset_index().to_csv(
        OUTPUT / "geometry_group_comparison.csv", index=False, encoding="utf-8-sig")
    lines = ["# 16站神经网络与岭回归配对核验", "", "324个神经网络结果完整，数据、冻结H、16站训练/隐藏名单和归一化尺度与岭回归配对核验通过。未重新训练模型。", "",
        f"从保存预测独立重算{audit_result['formal_station_metrics_recomputed']}个正式逐站指标，最大绝对差{audit_result['max_absolute_metric_recomputation_error']:.3g}；"
        f"从权重和独立输入拼接抽样重建{audit_result['sampled_checkpoint_predictions_reconstructed']}个预测，最大标准化Q差{audit_result['max_sampled_normalized_prediction_error']:.3g}。", "",
        "主要成绩仅用当组16站内Q隐藏站、原test期、冻结预测H；先逐站算NSE、每次运行等权平均，再平均3个掩膜×3个模型种子。比例是原86名义比例，16站实际隐藏数不同。", "",
        "| 归一化 | Q隐藏比例 | 岭回归无面积 | 岭回归有面积 | 神经网络无面积 | 神经网络有面积 | 网络面积增量 |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for row in comparison.itertuples():
        lines.append(f"| {row.target_scaling} | {row.ratio:.0%} | {row.ridge_no_area_nse:.4f} | {row.ridge_area_nse:.4f} | {row.nn_no_area_nse:.4f} | {row.nn_area_nse:.4f} | {row.nn_area_delta:+.4f} |")
    lines += ["", "这次具体配置下，神经网络在六种设置中有面积和无面积的平均成绩均低于对应岭回归；加入面积也没有带来平均收益。"
              "这不能推广成所有神经网络均不如回归，也不能据此断定面积没有水文学价值。", "",
        "## 只输入水位的网络", "",
        "| 归一化 | Q隐藏比例 | 仅水位网络 | 水位+属性+恒定可用标记网络 | 再加面积网络 |",
        "| --- | ---: | ---: | ---: | ---: |"]
    for row in comparison.itertuples():
        lines.append(f"| {row.target_scaling} | {row.ratio:.0%} | {row.nn_h_only_nse:.4f} | {row.nn_no_area_nse:.4f} | {row.nn_area_nse:.4f} |")
    lines += ["", "仅水位网络在这六种设置中都高于带27个属性的两个主要网络，提示高维条件化和跨站迁移应优先检查。"
        "目前所有站面积可用标记恒1；h_attrs与h_attrs_area_available函数可表达的信息相同，但维度、初始化及正则参数化不同。"
        "两者成绩仍有差异，说明不能把每一项分数变化都归因于新增水文学信息。", "",
        "## 稳定性", "", "| 归一化 | 比例 | 网络面积更好次数 | 无面积逐站中位NSE | 有面积逐站中位NSE | 无面积/有面积负NSE站次数 |",
        "| --- | ---: | ---: | ---: | ---: | ---: |"]
    for row in comparison.itertuples():
        lines.append(f"| {row.target_scaling} | {row.ratio:.0%} | {row.nn_area_better_runs}/{row.n_paired_runs} | {row.nn_no_area_median:.4f} | {row.nn_area_median:.4f} | {row.nn_no_area_negative_count}/{row.nn_area_negative_count}，共{row.n_station_runs}站次 |")
    lines += ["", "负NSE站次数和中位数需与均值共同阅读，不能因少数极端差站直接删除流域。11达标/5重点组全部保留，逐站数据含分组。", "",
        "## 拟合与训练预算线索", "", "| 归一化 | 输入 | 选中50轮次数 | 选中宽16次数 | 内部验证NSE中位数 |",
        "| --- | --- | ---: | ---: | ---: |"]
    for row in convergence.itertuples():
        lines.append(f"| {row.target_scaling} | {row.variant} | {row.selected_epoch50_count}/{row.n_models} | {row.selected_width16_count}/{row.n_models} | {row.median_inner_validation_nse:.4f} |")
    lines += ["", "选中第50轮提示部分运行可能受训练轮数上限影响，不等于已证明网络容量不足。"
              "候选宽度4/8/16的完整曲线和最终重拟合损失来自原done.json，没有按当前测试结果重新选模型。", "",
        "## 已见训练站的拟合表现", "",
        "以下是最终模型在Q保留站上逐站等权的NSE，再平均27次运行。valid指这些站后续时段，不是未见过本站Q的整站验证。", "",
        "| 归一化 | 输入 | 网络训练期NSE | 岭回归训练期NSE | 网络后续valid期NSE | 岭回归后续valid期NSE |",
        "| --- | --- | ---: | ---: | ---: | ---: |"]
    for (scaling, variant), group in diagnostics.groupby(["target_scaling", "variant"]):
        train, valid = group[group.split.eq("train")], group[group.split.eq("valid")]
        lines.append(f"| {scaling} | {variant} | {train.nn_final_model_nse.mean():.4f} | {train.ridge_final_model_nse.mean():.4f} | {valid.nn_final_model_nse.mean():.4f} | {valid.ridge_final_model_nse.mean():.4f} |")
    lines += ["", "网络在这些已见过本站训练Q的流域上高于岭回归，而在Q隐藏站上更差，更符合跨站泛化不稳定这一解释。"
        "这不是已验证的单一因果机制，不能直接断言增大或缩小网络就能解决。", "",
        "retained_fit_diagnostics.csv额外评价已保存最终模型在Q保留站原train/valid期的拟合，仅用与正式拟合相同的最多4096个配对。"
        "这里的valid是最终模型在已见过本站train期Q后的时间验证，不能冒充内部整站未见站验证。"
        "内部选模best_score与最终重拟合来自不同模型，不能直接把二者之差解释为单一泛化差距。", "",
        "## 六个对照与逐站失败", "",
        "six_variant_controls.csv包含H单独、H+属性、可用标记、正确面积、打乱面积、参数量上界控制。"
        "容量控制改变宽度并采用内部验证选轮数，不能单独用其成绩证明某个因果机制。", "",
        "| 归一化 | 比例 | 掩膜 | 种子 | 流域 | 网络无面积NSE | 网络有面积NSE | 改用观测H的面积模型NSE |",
        "| --- | ---: | ---: | ---: | --- | ---: | ---: | ---: |"]
    for row in worst.head(8).itertuples():
        lines.append(f"| {row.target_scaling} | {row.ratio:.0%} | {row.mask_seed} | {row.model_seed} | {row.basin} | {row.baseline_nse:.4f} | {row.nse:.4f} | {row.observed_h_area_nse:.4f} |")
    lines += ["", "观测H诊断仍差的站，不能仅归因于前面的水位预测。当前面积由A(H)生成，与水位存在确定函数依赖；"
        "其潜在增益主要来自跨站几何差异，是否能迁移取决于少量Q保留站的代表性及面积外推。", "",
        "## 当前建议与限制", "",
        "优先保留岭回归为简单基线。先用内部验证判断训练预算或表示是否不足，再决定是否新增实验；不能看完当前测试集后选一个最好宽度或轮数并宣称独立泛化。", "",
        "observed允许隐藏站训练Q均值/标准差，是已知尺度对照；physical只用当前Q保留站拟合尺度。"
        "原测试集已经历多轮探索，尚需独立验证。当前匹配比较保留同样数据/输入/划分，但岭回归与神经网络的优化方式、正则和运行环境不同。", "",
        "原代码哈希若仅因CRLF/LF不同，audit.json单列记录；未忽略实际代码差异。", ""]
    (OUTPUT / "report.md").write_text("\n".join(lines), encoding="utf-8")
    print(comparison.to_string(index=False))
    print("convergence", convergence.to_string(index=False))
    print(fit_summary.to_string())
    return comparison


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    spec, data, models, diagnostics, result = audit()
    analyze(spec, data, models, diagnostics, result)
    print(f"核验通过；报告：{OUTPUT / 'report.md'}", flush=True)


if __name__ == "__main__":
    main()
