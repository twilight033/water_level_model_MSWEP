"""检查输入确为1/2维、Q信息隔离及神经网络配置；不训练神经网络。"""

from pathlib import Path
import runpy
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from cross_section_experiment import area_lowdim16 as low
from cross_section_experiment.converters import PairData


def fixture():
    old = runpy.run_path(str(Path(__file__).with_name("test_atgage16_area_stage1_only16.py")))
    scene = next(s for s in low.cohort.restrict_scenarios(old["frozen_scenarios"]()) if s["tag"] == "q70_mask123")
    prepared, payload, guarded_q, scale_calls = old["fixture"](scene["retained"])
    # 直接属性全部设NaN；低维构造不能接触这些值。
    prepared.attrs[:] = np.nan
    return prepared, payload, guarded_q, scale_calls, scene


def test_inputs_are_exactly_h_or_h_and_area_without_attributes_or_marker():
    prepared, payload, _, _, scene = fixture()
    ids = np.array([prepared.basin_index[b] for b in scene["retained"]])
    pairs = PairData(np.ones((len(ids), 1), np.float32), np.zeros(len(ids)), ids, np.ones(len(ids)), np.zeros(len(ids), int))
    scaler = {"mean_log1p_area": .2, "std_log1p_area": .5}
    h = low.input_matrix(prepared, payload, pairs, "h_only")
    ha = low.input_matrix(prepared, payload, pairs, "h_area", scaler)
    assert h.shape == (len(ids), 1) and ha.shape == (len(ids), 2)
    np.testing.assert_array_equal(h[:, 0], pairs.x[:, 0])
    np.testing.assert_allclose(ha[:, 1], (np.log1p(3) - .2) / .5)
    assert np.isfinite(ha).all()
    assert low.feature_names("h_only") == ["waterlevel_z"]
    assert low.feature_names("h_area") == ["waterlevel_z", "area_log1p_z"]


def test_neural_selection_and_refit_keep_original_settings_and_retained_q():
    cfg = {"converter_max_pairs_per_basin": 4, "converter_widths": [4, 8, 16], "converter_epochs": 50, "device": "cpu"}
    for variant, dim in (("h_only", 1), ("h_area", 2)):
        prepared, payload, q, scale_calls, scene = fixture()
        inner_train, inner_valid, _ = low.area.area_inner_split(scene["retained"], 1)
        calls = []

        def no_training(train, hidden, epochs, seed, device, valid=None, **kwargs):
            assert train.x.shape[1] == dim and np.isfinite(train.x).all()
            assert kwargs == {"batch_size": 2048, "learning_rate": .001}
            if valid is not None:
                assert set(train.basin).isdisjoint(valid.basin)
                assert (train.target_pos < 8).all() and (valid.target_pos >= 8).all()
                assert epochs == 50
            else:
                assert epochs == 3 and (train.target_pos < 8).all()
            calls.append((hidden, valid is not None))
            return SimpleNamespace(n_parameters=lambda: hidden * (dim + 2) + 1), {
                "best_score": float(hidden), "best_epoch": 3, "history": []}

        with patch("cross_section_experiment.area_lowdim16.fit_network", side_effect=no_training):
            _, _, audit, final = low.fit_converter(prepared, payload, scene, 1, "physical", variant, "neural", cfg)
        assert calls == [(4, True), (8, True), (16, True), (16, False)]
        assert scale_calls == [("physical", inner_train), ("physical", scene["retained"])]
        assert audit["inner_validation_basins"] == inner_valid and audit["converter_static_attributes"] == []
        assert not audit["availability_marker_included"] and final.x.shape[1] == dim
        assert len(q.calls) == 6


def test_ridge_uses_same_lowdim_features_and_q_boundary():
    cfg = {"converter_max_pairs_per_basin": 4, "device": "cpu"}
    for scaling in ("physical", "observed"):
        for variant, dim in (("h_only", 1), ("h_area", 2)):
            prepared, payload, q, calls, scene = fixture()
            model, _, audit, final = low.fit_converter(prepared, payload, scene, 1, scaling, variant, "ridge", cfg)
            assert len(model.coef) == dim and audit["n_parameters"] == dim + 1
            assert final.x.shape[1] == dim and len(q.calls) == 6
            assert all(set(basins).issubset(scene["retained"]) for _, basins in calls)
            assert audit["alpha"] in low.ALPHAS and audit["hidden"] is None
            if variant == "h_area":
                assert set(audit["inner_area_scaler"]["fit_basins"]) == set(audit["inner_train_basins"])
                assert set(audit["final_area_scaler"]["fit_basins"]) == set(scene["retained"])
