"""测量几何重拟合的科学边界：量纲、原始深度和隐藏Q隔离。"""

import copy
from pathlib import Path
import sys
import unittest

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tests"))

from test_cross_section86 import prepared_fixture
from cross_section_experiment.survey_geometry import (
    SCALE_COLUMNS, STAGE_COLUMN, fit_survey_scaler, measured_geometry, survey_inputs,
)
from cross_section_experiment.survey_refit import fit_converter


def payload():
    return {"record": {"NXSDB_Area_sf": 30., "NXSDB_WettedDistance_ft": 12.,
            "NXSDB_Width_ft": 10., "NWIS_GageHeight_ft": 5.,
            "NWIS_Discharge_cfs": 100., "MeanVelocity_fps": 3.},
            "units": {"NXSDB_Area_sf": "ft^2", "NXSDB_WettedDistance_ft": "ft",
                      "NXSDB_Width_ft": "ft", "NWIS_GageHeight_ft": "ft"},
            "points": [{"Distance_ft": x, "Depth_ft": d}
                       for x, d in zip([0., 5., 10.], [2., 4., 2.])]}


class SurveyRefitBoundaries(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_discharge_velocity_do_not_enter_geometry(self):
        original = payload()
        changed = copy.deepcopy(original)
        changed["record"]["NWIS_Discharge_cfs"] = 1e20
        changed["record"]["MeanVelocity_fps"] = -999
        self.assertEqual(measured_geometry(original), measured_geometry(changed))

    def test_true_depth_shift_is_not_erased(self):
        original = payload()
        changed = copy.deepcopy(original)
        for point in changed["points"]:
            point["Depth_ft"] += 2.
        changed["record"]["NXSDB_Area_sf"] += 20.
        a, b = measured_geometry(original), measured_geometry(changed)
        self.assertGreater(b["survey_log_area_m2"], a["survey_log_area_m2"])
        self.assertAlmostEqual(b["raw_profile_integral_m2"] - a["raw_profile_integral_m2"], 20 * .09290304)

    def test_units_must_be_explicit(self):
        original = payload()
        original["units"].pop("NWIS_GageHeight_ft")
        with self.assertRaises(ValueError):
            measured_geometry(original)

    def test_scaler_does_not_fit_hidden_geometry_and_shuffle_preserves_missing(self):
        table = pd.DataFrame({"a": [1., 3., 100., np.nan], "geom_missing": [0, 0, 0, 1]},
                             index=["01", "02", "03", "04"])
        one = fit_survey_scaler(table, ["01", "02"], ["a"])
        table.loc["03", "a"] = 1e10
        self.assertEqual(one, fit_survey_scaler(table, ["01", "02"], ["a"]))
        plain = survey_inputs(table, list(table.index), one)
        shuffled = survey_inputs(table, list(table.index), one, 42)
        np.testing.assert_array_equal(plain[:, -1], shuffled[:, -1])
        np.testing.assert_array_equal(plain[3], shuffled[3])
        np.testing.assert_array_equal(np.sort(plain[:3, 0]), np.sort(shuffled[:3, 0]))

    def test_hidden_q_labels_and_statistics_do_not_change_physical_fit(self):
        prepared = prepared_fixture()
        retained = prepared.basins[:4]
        table = pd.DataFrame({column: np.arange(6, dtype=float) + index
                              for index, column in enumerate(SCALE_COLUMNS + [STAGE_COLUMN])},
                             index=prepared.basins)
        table["geom_missing"] = 0
        cfg = {"converter_widths": [4], "converter_epochs": 2,
               "converter_max_pairs_per_basin": 32, "device": "cpu"}
        a, _, _, audit = fit_converter(prepared, table, retained, "h_survey_geometry", 1, 42, "physical", cfg)
        mutated = copy.deepcopy(prepared)
        for basin in prepared.basins[4:]:
            bi = prepared.basin_index[basin]
            mutated.targets_raw["flow"][bi] = 1e10
            mutated.stats["flow"][basin] = {"mean": 1e10, "std": 1e9, "n": 99}
        b, _, _, _ = fit_converter(mutated, table, retained, "h_survey_geometry", 1, 42, "physical", cfg)
        for key, value in a.state_dict().items():
            self.assertTrue(torch.equal(value, b.state_dict()[key]))
        self.assertEqual(set(audit["training_q_pair_basins"]), set(retained))
        self.assertEqual(set(audit["inner_geometry_scaler"]["fit_basins"]), set(audit["inner_train_basins"]))


if __name__ == "__main__":
    unittest.main()
