"""16站专用转换器的信息边界、固定划分和指标范围检查；替换网络训练，不拟合模型。"""

import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pandas as pd

from cross_section_experiment import area_stage1_only16 as experiment


def frozen_scenarios():
    root = Path(__file__).resolve().parents[1]
    spec = json.loads((root / "configs/cross_section86_original_holdouts.json").read_text(encoding="utf-8-sig"))
    return [{"tag": tag, **scene, "retained": [b for b in spec["basins"] if b not in scene["heldout"]]}
            for tag, scene in spec["scenarios"].items()]


class GuardedQ:
    """一旦读取禁用站的Q就报错，记录每次真正访问的站和时刻。"""

    def __init__(self, values, allowed):
        self.values, self.allowed = values, set(allowed)
        self.calls = []

    def __getitem__(self, key):
        bi, pos = key
        assert np.isscalar(bi) and bi in self.allowed, f"读取了不允许的站点Q: {bi}"
        self.calls.append((int(bi), np.arange(self.values.shape[1])[pos].copy()))
        return self.values[key]


def fixture(allowed_basins):
    basins = ["outside"] + list(experiment.AREA_BASINS)
    count = len(basins)
    h = np.broadcast_to(np.linspace(.2, 1.2, 12), (count, 12)).copy()
    q = h * 3 + np.arange(count)[:, None]
    index = {b: i for i, b in enumerate(basins)}
    guarded_q = GuardedQ(q, [index[b] for b in allowed_basins])
    scale_calls = []

    def scaling(name, fit_basins):
        assert set(fit_basins).issubset(allowed_basins)
        scale_calls.append((name, list(fit_basins)))
        return SimpleNamespace(mean={"flow": np.zeros(count)}, std={"flow": np.ones(count)}, info={})

    prepared = SimpleNamespace(basins=basins, basin_index=index, attrs=np.zeros((count, 3)),
        target_mean={"waterlevel": np.zeros(count)}, target_std={"waterlevel": np.ones(count)},
        targets={"waterlevel": h}, targets_raw={"flow": guarded_q}, make_scaling=scaling,
        split_range=lambda basin, split: (0, 7) if split == "train" else (8, 11))
    payload = {"curves": {b: {"intercept_m2": 1., "slope_m": 2., "h_min_m": .2, "h_max_m": 1.2,
                              "lovo_r2": .9, "time_r2": .85, "lovo_relative_mae": .1}
                           for b in experiment.AREA_BASINS}}
    return prepared, payload, guarded_q, scale_calls


def test_original_masks_are_intersected_without_resampling():
    outer = frozen_scenarios()
    scenes = experiment.restrict_scenarios(outer)
    expected = {(.3, 42): 11, (.3, 123): 7, (.3, 456): 12,
                (.5, 42): 9, (.5, 123): 8, (.5, 456): 9,
                (.7, 42): 6, (.7, 123): 3, (.7, 456): 5}
    assert len(scenes) == 9
    for source, scene in zip(outer, scenes):
        assert scene["retained"] == [b for b in source["retained"] if b in experiment.AREA_BASINS]
        assert scene["heldout"] == [b for b in source["heldout"] if b in experiment.AREA_BASINS]
        assert len(scene["retained"]) == expected[(scene["ratio"], scene["mask_seed"])]
        assert scene["actual_q_hidden_ratio16"] == len(scene["heldout"]) / 16
        assert set(scene["retained"] + scene["heldout"]) == set(experiment.AREA_BASINS)
        for seed in (1, 2, 3):
            train, valid, audit = experiment.original.area_inner_split(scene["retained"], seed)
            assert set(train).isdisjoint(valid) and not audit["changed"]
            assert set(train + valid) == set(scene["retained"])
            assert len(train) >= 2 and len(valid) >= 1


def test_scale_guard_rejects_outside_hidden_and_implicit_population():
    retained = experiment.AREA_BASINS[:3]
    prepared, _, _, calls = fixture(retained)
    guarded = experiment.RestrictedQPrepared(prepared, retained)
    for scaling in ("physical", "observed"):
        for denied in (None, [], ["outside"], [experiment.AREA_BASINS[-1]]):
            try:
                guarded.make_scaling(scaling, denied)
            except ValueError:
                pass
            else:
                raise AssertionError("尺度防护没有阻止不允许的拟合范围")
        guarded.make_scaling(scaling, list(retained[:2]))
    assert len(calls) == len(guarded.scaling_calls) == 2


def test_all_variants_use_same_retained16_and_no_forbidden_q_access():
    scene = next(s for s in experiment.restrict_scenarios(frozen_scenarios()) if s["tag"] == "q70_mask123")
    expected_train, expected_valid, _ = experiment.original.area_inner_split(scene["retained"], 1)
    cfg = {"converter_max_pairs_per_basin": 4, "converter_widths": [4, 8, 16],
           "converter_epochs": 50, "device": "cpu"}
    input_dims = {"h_only": 1, "h_attrs": 4, "h_attrs_area_available": 5,
                  "h_attrs_area": 6, "h_attrs_area_shuffled": 6, "h_attrs_capacity_control": 5}
    for scaling in ("physical", "observed"):
        for variant in experiment.VARIANTS:
            prepared, payload, q, scale_calls = fixture(scene["retained"])
            network_calls = []
            train_ids = {prepared.basin_index[b] for b in expected_train}
            valid_ids = {prepared.basin_index[b] for b in expected_valid}
            all_ids = {prepared.basin_index[b] for b in scene["retained"]}

            def no_training(train, hidden, epochs, seed, device, valid=None, **kwargs):
                assert train.x.shape[1] == input_dims[variant]
                assert kwargs == {"batch_size": 2048, "learning_rate": .001}
                if valid is not None:
                    assert set(train.basin) == train_ids and set(valid.basin) == valid_ids
                    assert (train.target_pos < 8).all() and (valid.target_pos >= 8).all()
                    assert epochs == 50
                else:
                    assert set(train.basin) == all_ids and (train.target_pos < 8).all()
                    assert epochs == 2
                if variant not in ("h_only", "h_attrs"):
                    assert (train.x[:, -1] == 1).all()
                network_calls.append((hidden, valid is not None))
                return SimpleNamespace(n_parameters=lambda: hidden * (input_dims[variant] + 2) + 1), {
                    "best_score": float(hidden), "best_epoch": 2, "history": []}

            forced = 16 if variant in ("h_attrs_area_shuffled", "h_attrs_capacity_control") else None
            with patch("cross_section_experiment.area_stage1.fit_network", side_effect=no_training), patch("builtins.print"):
                _, _, _, audit = experiment.fit_converter(prepared, payload, scene, variant, 1, scaling, cfg, forced)
            expected_calls = [(16, True), (16, False)] if forced else [(4, True), (8, True), (16, True), (16, False)]
            assert network_calls == expected_calls
            assert scale_calls == [(scaling, expected_train), (scaling, scene["retained"])]
            assert audit["training_q_pair_basins"] == scene["retained"]
            assert audit["inner_train_basins"] == expected_train and audit["inner_validation_basins"] == expected_valid
            assert {bi for bi, _ in q.calls} == all_ids
            assert len(q.calls) == 6
            assert audit["observed_q_scale_basins"] == (list(experiment.AREA_BASINS) if scaling == "observed" else [])
            if variant in ("h_attrs_area", "h_attrs_area_shuffled"):
                assert set(audit["inner_area_scaler"]["fit_basins"]) == set(expected_train)
                assert set(audit["final_area_scaler"]["fit_basins"]) == set(scene["retained"])


def test_h_prediction_filter_preserves_original_indices_and_values():
    prepared, _, _, _ = fixture(experiment.AREA_BASINS)
    ids = np.repeat(np.arange(len(prepared.basins)), 3)
    prediction = {"basin_idx": ids, "target_pos": np.tile([9, 3, 7], len(prepared.basins)),
                  "pred_h_normalized": np.arange(len(ids), dtype=float)}
    filtered = experiment.filter_prediction(prepared, prediction)
    keep = ids != prepared.basin_index["outside"]
    for key in prediction:
        np.testing.assert_array_equal(filtered[key], prediction[key][keep])
    assert set(filtered["basin_idx"]) == set(range(1, 17))


def test_metrics_read_only16_test_q_and_match_independent_nse():
    prepared, payload, q, _ = fixture(experiment.AREA_BASINS)
    scene = experiment.restrict_scenarios(frozen_scenarios())[0]
    ids = np.repeat([prepared.basin_index[b] for b in experiment.AREA_BASINS], 3)
    pos = np.tile([8, 9, 11], 16)
    obs = q.values[ids, pos]
    pred = obs + .25
    frame = experiment.metrics16(prepared, payload, ids, pos, pred, scene, 1, "h_attrs_area", "physical", "predicted_h")
    assert len(frame) == len(q.calls) == 16 and set(frame.basin) == set(experiment.AREA_BASINS)
    assert set(frame.loc[frame.primary_evaluation, "basin"]) == set(scene["heldout"])
    for i, row in enumerate(frame.itertuples()):
        observed, estimated = obs[i * 3:(i + 1) * 3], pred[i * 3:(i + 1) * 3]
        expected = 1 - np.sum((estimated - observed) ** 2) / np.sum((observed - observed.mean()) ** 2)
        assert row.n == 3 and np.isclose(row.nse, expected)
    assert all((positions >= 8).all() for _, positions in q.calls)


def test_summary_uses_hidden16_as_primary_and_pairs_equal_populations():
    prepared, payload, q, _ = fixture(experiment.AREA_BASINS)
    scene = experiment.restrict_scenarios(frozen_scenarios())[0]
    ids = np.repeat([prepared.basin_index[b] for b in experiment.AREA_BASINS], 3)
    pos = np.tile([8, 9, 11], 16)
    truth = q.values[ids, pos]
    with TemporaryDirectory() as temporary:
        output = Path(temporary)
        for variant, offset in (("h_attrs_area_available", .3), ("h_attrs_area", .2)):
            folder = output / "physical/stage1" / scene["tag"] / "seed1" / variant
            folder.mkdir(parents=True)
            meta = {"protocol": experiment.PROTOCOL, "training_q_pair_basins": scene["retained"],
                    "variant": variant, "target_scaling": "physical", "ratio": scene["ratio"],
                    "mask_seed": scene["mask_seed"], "model_seed": 1}
            (folder / "done.json").write_text(json.dumps(meta), encoding="utf-8")
            for source, filename in (("predicted_h", "metrics.csv"), ("observed_h_diagnostic", "diagnostic_metrics.csv")):
                frame = experiment.metrics16(prepared, payload, ids, pos, truth + offset, scene, 1, variant, "physical", source)
                frame.to_csv(folder / filename, index=False)
        assert experiment.summarize(output, "test_only_not_a_research_result") == 2
        summary = pd.read_csv(output / "summary_by_run.csv")
        assert set(summary.scope) == {"all16", "q_heldout16", "q_retained16", "qualified11", "focus5",
                                      "q_heldout_qualified11", "q_heldout_focus5"}
        assert (summary.loc[summary.scope.eq("q_heldout16"), "n_basins"] == len(scene["heldout"])).all()
        assert (summary.loc[summary.scope.eq("all16"), "n_basins"] == 16).all()
        paired = pd.read_csv(output / "paired_differences.csv", dtype={"basin": str})
        assert len(paired) == 64 and paired.baseline_nse.notna().all()
        assert (paired.loc[paired.variant.eq("h_attrs_area"), "delta_nse_vs_same_availability_control"] > 0).all()
        manifest = json.loads((output / "summary_manifest.json").read_text())
        assert manifest["primary_scope"] == "q_heldout16" and manifest["training_and_selection_only16"]
