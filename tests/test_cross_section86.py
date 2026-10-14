"""断面实验的重要边界测试：标签隔离、几何来源与独立转换器。"""

import copy
import csv
import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import pandas as pd
import torch

from cross_section_experiment.converters import collect_pairs, fit_regional_converter
from cross_section_experiment.features import (
    FEATURE_COLUMNS, build_geometry, fit_geometry_scaler, profile_features,
    shuffled_geometry, transform_geometry,
)
from cross_section_experiment.inference import InferenceWindows
from cross_section_experiment.package import sanitized_profile
from cross_section_experiment.protocol import (
    WaterlevelOnlyPrepared, heldout_split, q_masks, validate_population,
)
from pipeline.dataset import PreparedData, TargetScaling, WindowDataset
from training.trainer import TrainConfig, train_model


def prepared_fixture(n_basins=6):
    """短序列仅用于测试，不写入真实缓存。"""
    p = object.__new__(PreparedData)
    p.basins = [f"01{i:06d}" for i in range(n_basins)]
    p.basin_index = {b: i for i, b in enumerate(p.basins)}
    p.grid = pd.date_range("2000-01-01", periods=100, freq="3h")
    p.n_time = len(p.grid)
    rng = np.random.default_rng(19)
    p.forcing = rng.normal(size=(n_basins, 100, 3)).astype(np.float32)
    p.attrs = rng.normal(size=(n_basins, 2)).astype(np.float32)
    p.attr_columns = ["area_scaled", "climate_scaled"]
    p.attrs_raw = pd.DataFrame({"area": np.arange(1, n_basins + 1) * 100.0,
                                "p_mean": np.arange(1, n_basins + 1) + 2.0}, index=p.basins)
    time = np.arange(100)
    h = np.array([2 + np.sin(time / 5 + i) for i in range(n_basins)], np.float32)
    q = np.array([(i + 1) * 10 + (i + 1) * np.sin(time / 5 + i)
                  for i in range(n_basins)], np.float32)
    p.targets_raw = {"flow": q, "waterlevel": h}
    p.stats = {task: {b: {"mean": float(values[i, :51].mean()),
                          "std": float(values[i, :51].std()), "n": 51}
                     for i, b in enumerate(p.basins)}
               for task, values in p.targets_raw.items()}
    p.splits = {"splits": {
        b: {"train_start": str(p.grid[0]), "train_end": str(p.grid[50]),
            "valid_start": str(p.grid[60]), "valid_end": str(p.grid[75]),
            "test_start": str(p.grid[86]), "test_end": str(p.grid[99])}
        for b in p.basins}}
    p.default_scaling = TargetScaling(p, "observed")
    p.target_mean, p.target_std = p.default_scaling.mean, p.default_scaling.std
    p.targets = p.default_scaling.normalized
    return p


class CrossSectionBoundaries(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_profile_features_ignore_q_velocity_and_uniform_depth_shift(self):
        profile = {"points": [{"Distance_ft": x, "Depth_ft": d}
                              for x, d in zip([0, 2, 4, 6, 8], [1, 2, 4, 3, 1])],
                   "record": {"NWIS_Discharge_cfs": 1.0, "MeanVelocity_fps": 2.0}}
        changed = copy.deepcopy(profile)
        changed["record"] = {"NWIS_Discharge_cfs": 1e12, "MeanVelocity_fps": -999}
        for point in changed["points"]:
            point["Depth_ft"] += 10
        np.testing.assert_allclose(profile_features(profile), profile_features(changed), atol=1e-6)
        mirrored = copy.deepcopy(profile)
        for point in mirrored["points"]:
            point["Distance_ft"] = 8 - point["Distance_ft"]
        np.testing.assert_allclose(profile_features(profile), profile_features(mirrored), atol=1e-6)

    def test_quality_time_filter_and_leading_zero_station(self):
        p = prepared_fixture()
        station = p.basins[0]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            profiles = root / "profiles"
            profiles.mkdir()
            rows = []
            for profile_id, date, quality in (
                ("poor", "1999-01-01", "POOR"), ("accepted", "2001-01-01", "GOOD"),
            ):
                file = profiles / f"{profile_id}.json"
                file.write_text(json.dumps({
                    "station": station, "points": [{"Distance_ft": 0, "Depth_ft": 0},
                    {"Distance_ft": 2, "Depth_ft": 2}, {"Distance_ft": 4, "Depth_ft": 0}]}))
                rows.append({"station": station, "mmt_ID": profile_id,
                             "measurement_date_local": date, "quality_code": quality,
                             "json_path": f"profiles/{profile_id}.json", "valid_profile": "True"})
            index = profiles / "profile_index.csv"
            with index.open("w", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)
            table, audit = build_geometry(index, p.basins, p.splits, "snapshot")
            self.assertEqual(audit.loc[0, "profile_id"], "accepted")
            self.assertTrue(audit.loc[0, "post_train_geometry"])
            self.assertEqual(table.index[0], station)
            causal, _ = build_geometry(index, p.basins, p.splits, "causal")
            self.assertEqual(causal.loc[station, "geom_missing"], 1)

    def test_q_is_hidden_in_train_and_valid_h_and_test_are_kept(self):
        p = prepared_fixture()
        held, retained = heldout_split(p.basins, 0.5, 42)
        train, valid = q_masks(p, held)
        scale = p.make_scaling("physical", retained)
        for split, mask in (("train", train), ("valid", valid)):
            ds = WindowDataset(p, split, 4, 1, hidden=mask, scaling=scale)
            for basin in held:
                indexes = np.flatnonzero(ds.basin_idx == p.basin_index[basin])
                self.assertGreater(len(indexes), 0)
                batch = ds[indexes]
                self.assertTrue(torch.all(batch[3][:, 0] == 0))
                self.assertTrue(torch.all(batch[3][:, 1] == 1))
        test = WindowDataset(p, "test", 4, 1, scaling=scale)
        held_indexes = np.isin(test.basin_idx, [p.basin_index[b] for b in held])
        self.assertTrue(torch.all(test[np.flatnonzero(held_indexes)][3] == 1))

    def test_scalers_and_converter_ignore_external_q_and_q_statistics(self):
        p = prepared_fixture()
        held, retained = heldout_split(p.basins, 0.33, 42)
        geometry = pd.DataFrame(np.arange(36).reshape(6, 6), index=p.basins,
                                columns=FEATURE_COLUMNS).astype(float)
        geometry["geom_missing"] = 0
        scaler = fit_geometry_scaler(geometry, retained)
        normalized = transform_geometry(geometry, p.basins, scaler)
        poisoned = copy.deepcopy(p)
        for basin in held:
            bi = p.basin_index[basin]
            poisoned.targets_raw["flow"][bi, :76] = 1e12
            poisoned.stats["flow"][basin] = {"mean": float("nan"), "std": float("nan"), "n": 0}
        scale = p.make_scaling("physical", retained)
        scale_poisoned = poisoned.make_scaling("physical", retained)
        np.testing.assert_array_equal(scale.mean["flow"], scale_poisoned.mean["flow"])
        a, _, info = fit_regional_converter(p, retained, normalized, [4], 2, 3, "cpu", 32)
        b, _, info_poisoned = fit_regional_converter(
            poisoned, retained, normalized, [4], 2, 3, "cpu", 32)
        for key in a.state_dict():
            torch.testing.assert_close(a.state_dict()[key], b.state_dict()[key], rtol=0, atol=0)
        self.assertEqual(info["inner_train_basins"], info_poisoned["inner_train_basins"])
        self.assertFalse(set(held) & set(info["inner_validation_basins"]))
        data = collect_pairs(p, retained, "train", scale, normalized, 32)
        for bi in np.unique(data.basin):
            self.assertAlmostEqual(float(data.weight[data.basin == bi].sum()),
                                   len(data.y) / len(np.unique(data.basin)), places=4)

    def test_h_model_has_no_q_scaling_and_is_invariant_to_q(self):
        p = prepared_fixture()
        poisoned = copy.deepcopy(p)
        poisoned.targets_raw["flow"][:] = np.nan
        poisoned.stats["flow"] = {}
        cfg = TrainConfig("single_waterlevel", seq_length=4, hidden_size=4, max_epochs=1,
                          batch_size=64, window_step_train=4, window_step_eval=2,
                          window_step_test=2, target_scaling="physical", model_seed=5, device="cpu")
        first = train_model(WaterlevelOnlyPrepared(p), cfg, fit_basins=p.basins, verbose=False)
        second = train_model(WaterlevelOnlyPrepared(poisoned), cfg, fit_basins=p.basins, verbose=False)
        self.assertNotIn("flow", first["scaling"].mean)
        for key in first["state_dict"]:
            torch.testing.assert_close(first["state_dict"][key], second["state_dict"][key],
                                       rtol=0, atol=0)

    def test_stage2_weights_and_selection_ignore_heldout_training_and_validation_q(self):
        p = prepared_fixture()
        held, retained = heldout_split(p.basins, 0.5, 42)
        hidden, hidden_valid = q_masks(p, held)
        poisoned = copy.deepcopy(p)
        for basin in held:
            bi = p.basin_index[basin]
            poisoned.targets_raw["flow"][bi, :76] = 1e12
            poisoned.stats["flow"][basin] = {"mean": float("nan"), "std": float("nan")}
        cfg = TrainConfig("dual_head", seq_length=4, hidden_size=4, max_epochs=1,
                          batch_size=64, window_step_train=4, window_step_eval=2,
                          window_step_test=2, target_scaling="physical", model_seed=7, device="cpu")
        original = train_model(p, cfg, hidden, hidden_valid, fit_basins=retained, verbose=False)
        altered = train_model(poisoned, cfg, hidden, hidden_valid,
                              fit_basins=retained, verbose=False)
        self.assertEqual(original["best_val_score"], altered["best_val_score"])
        for key in original["state_dict"]:
            torch.testing.assert_close(original["state_dict"][key], altered["state_dict"][key],
                                       rtol=0, atol=0)

    def test_inference_does_not_select_times_using_test_labels(self):
        p = prepared_fixture()
        before = InferenceWindows(p, "test", 4, 1, ("flow",))
        p.targets_raw["flow"][:] = np.nan
        p.targets["flow"][:] = np.nan
        p.targets_raw["waterlevel"][:] = np.nan
        p.targets["waterlevel"][:] = np.nan
        after = InferenceWindows(p, "test", 4, 1, ("flow",))
        np.testing.assert_array_equal(before.basin_idx, after.basin_idx)
        np.testing.assert_array_equal(before.target_pos, after.target_pos)
        shuffled = shuffled_geometry(np.column_stack([np.arange(6), np.zeros(6)]), 42)
        self.assertEqual(shuffled.shape, (6, 2))
        np.testing.assert_array_equal(shuffled[:, -1], 0)

    def test_missing_time_is_not_treated_as_an_adjacent_three_hour_step(self):
        p = prepared_fixture()
        p.grid = pd.DatetimeIndex(np.r_[p.grid[:90].to_numpy(),
                                       (p.grid[90:] + pd.Timedelta(hours=3)).to_numpy()])
        p.splits["splits"] = {
            b: {"train_start": str(p.grid[0]), "train_end": str(p.grid[50]),
                "valid_start": str(p.grid[60]), "valid_end": str(p.grid[75]),
                "test_start": str(p.grid[86]), "test_end": str(p.grid[99])}
            for b in p.basins}
        validate_population(p, p.basins, expected_count=6)
        self.assertFalse(p.time_valid[90])
        windows = InferenceWindows(p, "test", 4, 1, ("flow",))
        self.assertFalse(np.isin(windows.target_pos, [90, 91, 92, 93, 94]).any())

    def test_package_removes_discharge_velocity_and_derived_record_fields(self):
        original = {"station": "01017000", "mmt_ID": "test", "record": {
            "station": "01017000", "NWIS_Discharge_cfs": 1000,
            "MeanVelocity_fps": 5, "NWIS_Area_sf": 200, "NWIS_GageHeight_ft": 3},
            "points": [{"fid": 1, "Distance_ft": 0, "Depth_ft": 1, "Velocity_fps": 5}]}
        clean = sanitized_profile(original, "source_hash")
        text = json.dumps(clean)
        for forbidden in ("Discharge", "Velocity", "NWIS_Area", "GageHeight"):
            self.assertNotIn(forbidden, text)
        self.assertEqual(clean["points"][0]["Depth_ft"], 1)
        self.assertEqual(clean["source"]["original_json_sha256"], "source_hash")


if __name__ == "__main__":
    unittest.main()
