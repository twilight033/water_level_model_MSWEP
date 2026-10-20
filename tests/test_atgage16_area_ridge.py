"""独立核对岭回归解，并验证真实配对收集过程的信息边界。"""

from pathlib import Path
import runpy

import numpy as np
import pandas as pd

from cross_section_experiment import area_ridge16 as ridge


def test_ridge_matches_independent_augmented_least_squares():
    rng = np.random.default_rng(142)
    x = rng.normal(size=(40, 5))
    x[:, -1] = 1
    x[:, -2] = x[:, 0] + 2 * x[:, 1]
    y, w, alpha = rng.normal(size=40) + 5, rng.uniform(.1, 2, 40), .07
    model = ridge.fit_ridge(x, y, w, alpha)
    design = np.column_stack([np.ones(len(x)), x])
    penalty = np.diag([0, *([np.sqrt(alpha)] * x.shape[1])])
    weighted_design = design * np.sqrt(w / w.sum())[:, None]
    weighted_y = y * np.sqrt(w / w.sum())
    coef = np.linalg.lstsq(np.vstack([weighted_design, penalty]), np.r_[weighted_y, np.zeros(design.shape[1])], rcond=None)[0]
    np.testing.assert_allclose(np.r_[model.intercept, model.coef], coef, atol=1e-11)
    assert abs(model.coef[-1]) < 1e-12
    scaled = ridge.fit_ridge(x, y, 100 * w, alpha)
    np.testing.assert_allclose(model.predict(x), scaled.predict(x), atol=1e-11)
    shifted = ridge.fit_ridge(x, y + 10, w, alpha)
    np.testing.assert_allclose(shifted.coef, model.coef, atol=1e-11)
    assert abs(shifted.intercept - model.intercept - 10) < 1e-11


def test_ridge_ignores_forbidden_basin_q_and_preserves_shared_split():
    fixture = runpy.run_path(str(Path(__file__).with_name("test_atgage16_area_stage1_only16.py")))
    scenes = ridge.cohort.restrict_scenarios(fixture["frozen_scenarios"]())
    scene = next(s for s in scenes if s["tag"] == "q70_mask123")
    train, valid, _ = ridge.area.area_inner_split(scene["retained"], 1)
    for scaling in ("physical", "observed"):
        results = []
        for variant in ridge.VARIANTS:
            prepared, payload, q, calls = fixture["fixture"](scene["retained"])
            prepared.attrs_raw = pd.DataFrame(prepared.attrs, columns=["x1", "x2", "x3"])
            model, _, audit = ridge.fit_converter(prepared, payload, scene, 1, scaling, variant, 4)
            assert audit["inner_train_basins"] == train and audit["inner_validation_basins"] == valid
            assert calls == [(scaling, train), (scaling, scene["retained"])]
            assert len(q.calls) == 6
            assert {bi for bi, _ in q.calls} == {prepared.basin_index[b] for b in scene["retained"]}
            assert model.alpha in ridge.ALPHAS and len(audit["candidates"]) == len(ridge.ALPHAS)
            assert audit["training_q_pair_basins"] == scene["retained"]
            assert audit["observed_q_scale_basins"] == (list(ridge.area.AREA_BASINS) if scaling == "observed" else [])
            assert audit["feature_names"][-1] == "area_available" and abs(model.coef[-1]) < 1e-12
            if variant == "h_attrs_area":
                assert set(audit["inner_area_scaler"]["fit_basins"]) == set(train)
                assert set(audit["final_area_scaler"]["fit_basins"]) == set(scene["retained"])
            results.append(audit["inner_split_audit"])
        assert results[0] == results[1]
