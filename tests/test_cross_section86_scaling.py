"""验证两种Q归一化的路由及信息范围；不启动任何模型训练。"""

import json
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from cross_section_experiment.converters import RegionalConverter, fit_regional_converter
from cross_section_experiment.runner import make_train_config, parser, settings_from_args
from cross_section_experiment.protocol import heldout_split
from test_cross_section86 import prepared_fixture


class ScalingConfigurationTests(unittest.TestCase):
    def test_holdout_ids_equal_frozen_historical_run_records(self):
        frozen = json.loads((ROOT / "configs/cross_section86_original_holdouts.json").read_text(encoding="utf-8"))
        for row in frozen["scenarios"].values():
            held, retained = heldout_split(frozen["basins"], row["ratio"], row["mask_seed"], "original86")
            self.assertEqual(held, row["heldout"])
            self.assertFalse(set(held) & set(retained))

    def test_two_strong_configurations_preserve_original_training_settings(self):
        configs = []
        for scaling in ("physical", "observed"):
            args = parser().parse_args(["--config", str(ROOT / f"configs/cross_section86_strong_{scaling}.json")])
            cfg = settings_from_args(args)
            result = make_train_config(cfg, "dual_head", 1, "cpu")
            self.assertEqual(result.target_scaling, scaling)
            self.assertEqual((result.seq_length, result.attr_set, result.batch_size), (480, "extended", 512))
            self.assertFalse(hasattr(result, "eval_batch_size"))
            self.assertNotIn("evaluation_batch_size", cfg)
            configs.append(cfg)
        ignored = {"target_scaling", "output_name", "reference_run_templates"}
        self.assertEqual({k: v for k, v in configs[0].items() if k not in ignored},
                         {k: v for k, v in configs[1].items() if k not in ignored})
        self.assertNotEqual(configs[0]["output_name"], configs[1]["output_name"])

    def test_converter_routes_both_scalings_and_records_observed_q_statistics(self):
        p = prepared_fixture()
        retained = p.basins[:4]

        def no_training(data, hidden, epochs, seed, device, valid=None, **kwargs):
            model = RegionalConverter(data.x.shape[1], hidden)
            return model, {"best_epoch": 1, "best_score": 0.5, "history": [], "n_parameters": model.n_parameters()}

        for scaling in ("physical", "observed"):
            with patch("cross_section_experiment.converters.fit_network", side_effect=no_training), \
                    patch.object(p, "make_scaling", wraps=p.make_scaling) as calls:
                _, scale, meta = fit_regional_converter(p, retained, None, [4], 1, 1, "cpu", target_scaling=scaling)
            self.assertEqual([call.args[0] for call in calls.call_args_list], [scaling, scaling])
            self.assertEqual(meta["training_q_pair_basins"], retained)
            self.assertEqual(meta["heldout_q_statistics_used_for_scaling"], scaling == "observed")
            self.assertEqual(meta["scale_fit_basins"], retained if scaling == "physical" else [])
            self.assertEqual(meta["observed_q_scale_basins"], p.basins if scaling == "observed" else [])
            if scaling == "observed":
                self.assertIs(scale, p.default_scaling)


if __name__ == "__main__":
    unittest.main()
