"""独立核验低维岭回归/神经网络结果，只读保存模型，不重新训练。"""

import argparse
import hashlib
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from cross_section_experiment.area_lowdim16 import DEFAULT_ROOT, N_EXPECTED, VARIANTS, feature_names, protocol
from cross_section_experiment.area_stage1 import AREA_BASINS, load_json
from cross_section_experiment.converters import collect_pairs
from cross_section_experiment.protocol import fingerprint, write_json
from pipeline.dataset import PreparedData


def verify_hash(path, expected):
    raw = Path(path).read_bytes()
    if hashlib.sha256(raw).hexdigest() == expected:
        return "exact"
    if Path(path).suffix in (".py", ".txt"):
        normalized = raw.replace(b"\r\n", b"\n")
        if expected in (hashlib.sha256(normalized).hexdigest(), hashlib.sha256(normalized.replace(b"\n", b"\r\n")).hexdigest()):
            return "lf_crlf_only"
    raise AssertionError(f"文件内容变化: {path}")


def matrix_and_scale(model, audit, curves, ids, h):
    original, basins = list(model["original86_index_basins"]), list(model["flow_scale_basins"])
    lookup = {original.index(b): i for i, b in enumerate(basins)}
    local = np.array([lookup[int(bi)] for bi in ids])
    h = np.asarray(h)
    columns = [h]
    if audit["variant"] == "h_area":
        h_m = h.astype(float) * model["waterlevel_std"][local] + model["waterlevel_mean"][local]
        a = np.array([curves[b]["intercept_m2"] for b in basins])[local]
        b = np.array([curves[b]["slope_m"] for b in basins])[local]
        log_area = np.log1p(np.maximum(a + b * h_m, 0))
        scaler = audit["final_area_scaler"]
        columns.append((log_area - scaler["mean_log1p_area"]) / scaler["std_log1p_area"])
    return np.column_stack(columns).astype(np.float32), model["flow_mean"][local], model["flow_std"][local]


def predict(model, x, estimator):
    if estimator == "ridge":
        return x.astype(float) @ model["coef"] + float(model["intercept"])
    w = {k: v.cpu().numpy() for k, v in model["state_dict"].items()}
    return (np.maximum(x @ w["net.0.weight"].T + w["net.0.bias"], 0) @ w["net.2.weight"].T + w["net.2.bias"]).reshape(-1)


def check_scores(frame, prepared, ids, positions, predicted, basins, negative_reference=None):
    frame = frame.set_index("basin")
    assert set(frame.index) == set(basins)
    maximum = 0.
    for basin in basins:
        bi = prepared.basin_index[basin]
        take = ids == bi
        obs = prepared.targets_raw["flow"][bi, positions[take]].astype(float)
        pred = predicted[take]
        ok = np.isfinite(obs) & np.isfinite(pred)
        obs, paired_pred = obs[ok], pred[ok]
        assert len(obs) == frame.loc[basin, "n"]
        scores = {"nse": 1 - np.square(obs - paired_pred).sum() / np.square(obs - obs.mean()).sum(),
                  "rmse": np.sqrt(np.square(obs - paired_pred).mean()), "mae": np.abs(obs - paired_pred).mean()}
        for metric, actual in scores.items():
            reported = frame.loc[basin, metric]
            maximum = max(maximum, abs(actual - reported))
            assert np.isclose(actual, reported, rtol=5e-6, atol=2e-5), (basin, metric, actual, reported)
        # 正式测试有原始预测文件时，离散符号计数按原输出精确核对。
        # NSE等连续指标始终使用独立重建值；CPU/GPU的近零符号差另行审计。
        signed_pred = pred if negative_reference is None else negative_reference[take]
        assert int((signed_pred[np.isfinite(signed_pred)] < 0).sum()) == frame.loc[basin, "n_negative_predictions"], basin
    return maximum


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--estimator", choices=("ridge", "neural"), required=True)
    parser.add_argument("--out-dir", type=Path)
    args = parser.parse_args()
    output = args.out_dir or DEFAULT_ROOT / args.estimator
    spec, complete = load_json(output / "experiment.json"), load_json(output / "completed.json")
    assert spec["protocol"] == complete["protocol"] == protocol(args.estimator)
    assert complete["n_completed"] == N_EXPECTED
    assert fingerprint({k: v for k, v in spec.items() if k != "experiment_fingerprint"}) == spec["experiment_fingerprint"]
    ending_only = []
    for filename, digest in {**spec["code_hashes"], **spec["data_hashes"]}.items():
        if verify_hash(ROOT / filename, digest) == "lf_crlf_only":
            ending_only.append(filename)
    prepared = PreparedData(attr_set="extended")
    curves = load_json(output / "area_curves.json")["curves"]
    allowed = [prepared.basin_index[b] for b in AREA_BASINS]
    frozen = {}
    for scaling, source in spec["sources"].items():
        for seed in (1, 2, 3):
            folder = ROOT / "results" / f"cross_section86_strong_{scaling}_snapshot_v2/waterlevel_models/seed{seed}"
            reference = source["H_predictions"][str(seed)]
            verify_hash(folder / "prediction.npz", reference["prediction_sha256"])
            verify_hash(folder / "model.pt", reference["model_sha256"])
            with np.load(folder / "prediction.npz") as f:
                selected = np.isin(f["basin_idx"], allowed)
                frozen[(scaling, seed)] = {k: f[k][selected] for k in f.files}
    scenes = {s["tag"]: s for s in spec["scenarios"]}
    seen, metric_count, max_metric_error, max_prediction_error = set(), 0, 0., 0.
    near_zero_sign_differences = []
    for done in sorted(output.glob("*/stage1/*/*/*/done.json")):
        meta = load_json(done)
        scaling, _, tag, seed_folder, variant = done.relative_to(output).parts[:-1]
        seed = int(seed_folder[4:])
        key, scene = (scaling, tag, seed, variant), scenes[tag]
        assert key not in seen
        seen.add(key)
        assert meta["protocol"] == protocol(args.estimator)
        assert meta["feature_names"] == feature_names(variant) and meta["input_dim"] == len(feature_names(variant))
        assert meta["converter_static_attributes"] == [] and not meta["availability_marker_included"]
        assert meta["training_q_pair_basins"] == scene["retained"] and meta["heldout"] == scene["heldout"]
        train, valid = set(meta["inner_train_basins"]), set(meta["inner_validation_basins"])
        assert not train & valid and train | valid == set(scene["retained"])
        assert [set(c["fit_basins"]) for c in meta["q_scaling_fit_calls"]] == [train, set(scene["retained"])]
        expected_id = fingerprint({"experiment": spec["experiment_fingerprint"], "scaling": scaling, "scenario": tag, "seed": seed, "variant": variant})
        assert meta["run_fingerprint"] == expected_id
        if args.estimator == "ridge":
            with np.load(done.parent / "model.npz") as f:
                model = {k: f[k].copy() for k in f.files}
            assert len(model["coef"]) == meta["input_dim"] and meta["n_parameters"] == meta["input_dim"] + 1
            chosen = max(meta["candidates"], key=lambda c: (c["best_score"], c["alpha"]))
            assert chosen["alpha"] == meta["alpha"]
        else:
            model = torch.load(done.parent / "model.pt", map_location="cpu", weights_only=False)
            assert model["run_fingerprint"] == expected_id
            assert sum(t.numel() for t in model["state_dict"].values()) == meta["n_parameters"]
        assert "attrs" not in model and list(model["feature_names"]) == feature_names(variant)
        assert list(model["flow_scale_basins"]) == list(AREA_BASINS)
        reference = frozen[(scaling, seed)]
        ids, positions = reference["basin_idx"], reference["target_pos"]
        for source, filename in (("predicted_h", "metrics.csv"), ("observed_h_diagnostic", "diagnostic_metrics.csv")):
            h = reference["pred_h_normalized"] if source == "predicted_h" else prepared.targets["waterlevel"][ids, positions]
            x, mean, std = matrix_and_scale(model, meta, curves, ids, h)
            predicted = predict(model, x, args.estimator) * std + mean
            saved_prediction = None
            if source == "predicted_h":
                with np.load(done.parent / "prediction.npz") as saved:
                    np.testing.assert_array_equal(ids, saved["basin_idx"])
                    np.testing.assert_array_equal(positions, saved["target_pos"])
                    error = float(np.max(np.abs((predicted - saved["pred_flow"]) / std)))
                    max_prediction_error = max(max_prediction_error, error)
                    assert error < 5e-4
                    saved_prediction = saved["pred_flow"].copy()
                    different_sign = (predicted < 0) != (saved_prediction < 0)
                    if np.any(different_sign):
                        # 浮点后端不同可能使接近零的值跨过零点，必须证明确实在舍入量级。
                        near_zero = float(max(np.max(np.abs(predicted[different_sign]) / std[different_sign]),
                                              np.max(np.abs(saved_prediction[different_sign]) / std[different_sign])))
                        assert args.estimator == "neural" and near_zero <= 32 * np.finfo(np.float32).eps
                        near_zero_sign_differences.append({"model": str(done.parent.relative_to(output)),
                            "n_values": int(different_sign.sum()), "max_absolute_normalized_value": near_zero,
                            "reconstructed_values": predicted[different_sign].tolist(),
                            "saved_values": saved_prediction[different_sign].tolist()})
            frame = pd.read_csv(done.parent / filename, dtype={"basin": str})
            max_metric_error = max(max_metric_error, check_scores(frame, prepared, ids, positions, predicted, AREA_BASINS,
                                                                 negative_reference=saved_prediction))
            metric_count += 16
        scale = SimpleNamespace(mean={"flow": np.full(86, np.nan)}, std={"flow": np.full(86, np.nan)})
        scale.mean["flow"][allowed], scale.std["flow"][allowed] = model["flow_mean"], model["flow_std"]
        for period in ("train", "valid"):
            pairs = collect_pairs(prepared, scene["retained"], period, scale, None,
                spec["source_settings"][scaling]["converter_max_pairs_per_basin"])
            x, mean, std = matrix_and_scale(model, meta, curves, pairs.basin, pairs.x[:, 0])
            predicted = predict(model, x, args.estimator) * std + mean
            frame = pd.read_csv(done.parent / f"retained_{period}_metrics.csv", dtype={"basin": str})
            max_metric_error = max(max_metric_error, check_scores(frame, prepared, pairs.basin, pairs.target_pos, predicted, scene["retained"]))
            metric_count += len(scene["retained"])
    expected = {(s, c, seed, v) for s in ("physical", "observed") for c in scenes for seed in (1, 2, 3) for v in VARIANTS}
    assert seen == expected and len(seen) == N_EXPECTED
    result = {"status": "PASS", "estimator": args.estimator, "n_models": len(seen), "n_station_metrics_recomputed": metric_count,
        "max_normalized_prediction_reconstruction_error": max_prediction_error, "max_absolute_metric_error": max_metric_error,
        "direct_input_dimensions_verified": [1, 2], "no_direct_static_attributes_or_availability_marker": True,
        "frozen_h_sources_unchanged": True, "q_retained16_isolation_passed": True, "models_retrained": 0,
        "hash_differences_only_lf_crlf": ending_only,
        "near_zero_backend_sign_differences": near_zero_sign_differences,
        "formal_negative_counts_verified_against_saved_predictions": True}
    write_json(output / "validation.json", result)
    print(result)
    print(pd.read_csv(output / "primary_comparison.csv").to_string(index=False))


if __name__ == "__main__":
    main()
