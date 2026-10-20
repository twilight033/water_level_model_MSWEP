"""面积阶段1的尺度、缺失信息和分组隔离检查；不运行网络训练。"""

import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from cross_section_experiment.area_stage1 import AREA_BASINS, area_inner_split, area_values, input_matrix
from cross_section_experiment.converters import PairData


def fixture():
    prepared = SimpleNamespace(basins=["01030500", "01059000", "outside"],
        target_mean={"waterlevel": np.array([2., 5., 99.])},
        target_std={"waterlevel": np.array([.5, 2., 1.])}, attrs=np.array([[1.], [2.], [3.]]))
    payload = {"curves": {"01030500": {"intercept_m2": 2., "slope_m": 3., "h_min_m": 1., "h_max_m": 4.},
                           "01059000": {"intercept_m2": 1., "slope_m": 4., "h_min_m": 3., "h_max_m": 8.}}}
    return prepared, payload


def test_area_uses_original_meters_not_normalized_h():
    prepared, payload = fixture()
    values, available, outside, negative = area_values(prepared, payload, [0, 1, 2], [2, 1, 0], np.arange(3))
    np.testing.assert_allclose(np.expm1(values), [11, 29, 0])
    assert available.tolist() == [True, True, False]
    assert not outside.any() and not negative.any()


def test_shuffled_curve_moves_its_h_reference_together():
    prepared, payload = fixture()
    values, _, outside, _ = area_values(prepared, payload, [0, 1], [2, 1], np.array([1, 0, 2]))
    np.testing.assert_allclose(np.expm1(values), [37, 9.5])
    assert outside.tolist() == [True, False]


def test_missing_is_zero_standardized_feature_plus_unavailable_flag():
    prepared, payload = fixture()
    pairs = PairData(np.array([[2.], [1.], [0.]]), np.zeros(3), np.arange(3), np.ones(3), np.arange(3))
    x = input_matrix(prepared, payload, pairs, "h_attrs_area", {"mean_log1p_area": 5., "std_log1p_area": 2.}, np.arange(3))
    np.testing.assert_allclose(x[2], [0., 3., 0., 0.])
    assert x[0, -1] == 1 and x[1, -1] == 1
    assert x[0, -2] != 0


def test_negative_linear_area_is_flagged_and_clipped_without_clipping_h():
    prepared, payload = fixture()
    value, _, outside, negative = area_values(prepared, payload, [0], [-10.], np.arange(3))
    assert value[0] == 0 and outside[0] and negative[0]


def test_all_original_scenarios_keep_their_population_and_have_area_validation():
    root = Path(__file__).resolve().parents[1]
    spec = json.loads((root / "configs/cross_section86_original_holdouts.json").read_text(encoding="utf-8-sig"))
    basins = json.loads((root / "data/splits/basin_splits.json").read_text(encoding="utf-8-sig"))["splits"]
    changed = 0
    for scenario in spec["scenarios"].values():
        retained = [b for b in basins if b not in scenario["heldout"]]
        for seed in (1, 2, 3):
            train, valid, audit = area_inner_split(retained, seed)
            assert set(train).isdisjoint(valid)
            assert set(train + valid) == set(retained)
            assert len(set(train) & set(AREA_BASINS)) >= 2
            assert len(set(valid) & set(AREA_BASINS)) >= 1
            assert len(valid) == len(audit["original"]["valid"])
            assert (train, valid) == area_inner_split(retained, seed)[:2]
            changed += audit["changed"]
    assert changed == 5


def test_converter_selection_and_refit_never_receive_hidden_q_pairs():
    from cross_section_experiment.area_stage1 import fit_converter
    basins = list(AREA_BASINS) + ["outside", "hidden"]
    count = len(basins)
    h = np.broadcast_to(np.linspace(.2, 1.2, 12), (count, 12)).copy()
    q = h * 3 + np.arange(count)[:, None]
    q[-1] = np.nan
    scale_calls, network_calls = [], []

    def scaling(name, fit_basins):
        assert name == "physical" and "hidden" not in fit_basins
        scale_calls.append(list(fit_basins))
        return SimpleNamespace(mean={"flow": np.zeros(count)}, std={"flow": np.ones(count)})

    prepared = SimpleNamespace(basins=basins, basin_index={b: i for i, b in enumerate(basins)},
        attrs=np.zeros((count, 3)), target_mean={"waterlevel": np.zeros(count)},
        target_std={"waterlevel": np.ones(count)}, targets={"waterlevel": h}, targets_raw={"flow": q},
        make_scaling=scaling, split_range=lambda basin, split: (0, 7) if split == "train" else (8, 11))
    payload = {"curves": {b: {"intercept_m2": 1., "slope_m": 2., "h_min_m": .2, "h_max_m": 1.2}
                           for b in AREA_BASINS}}

    def no_training(train, hidden, epochs, seed, device, valid=None, **kwargs):
        assert prepared.basin_index["hidden"] not in train.basin
        assert train.x.shape[1] == 6 and kwargs["batch_size"] == 2048
        if valid is not None:
            assert set(train.basin).isdisjoint(valid.basin)
            assert prepared.basin_index["hidden"] not in valid.basin
        network_calls.append((hidden, valid is not None))
        model = SimpleNamespace(n_parameters=lambda: hidden * 8 + 1)
        return model, {"best_score": float(hidden), "best_epoch": 2, "history": []}

    scene = {"retained": basins[:-1], "mask_seed": 42}
    cfg = {"converter_max_pairs_per_basin": 4, "converter_widths": [4, 8, 16],
           "converter_epochs": 50, "device": "cpu"}
    with patch("cross_section_experiment.area_stage1.fit_network", side_effect=no_training):
        _, _, _, audit = fit_converter(prepared, payload, scene, "h_attrs_area", 1, "physical", cfg)
    assert network_calls == [(4, True), (8, True), (16, True), (16, False)]
    assert set(scale_calls[-1]) == set(scene["retained"])
    assert set(audit["inner_area_scaler"]["fit_basins"]).isdisjoint(audit["inner_validation_basins"])
    assert set(audit["final_area_scaler"]["fit_basins"]) == set(AREA_BASINS)
