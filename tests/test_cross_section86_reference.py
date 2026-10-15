"""历史模型复用的边界：配置不匹配须拒绝，原权重不能被改写。"""

import copy
import json
import sys
import tempfile
import unittest
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from cross_section_experiment.reference import reuse_h_checkpoint, validate_reference
from models.lstm_models import build_model
from test_cross_section86 import prepared_fixture
from training.trainer import TrainConfig


class HistoricalReferenceTests(unittest.TestCase):
    def source(self, prepared):
        cfg = TrainConfig("single_waterlevel", seq_length=8, hidden_size=4,
                          target_scaling="physical", device="cpu")
        model = build_model("single_waterlevel", 3, prepared.attrs.shape[1], 4, 0.2)
        blob = {"config": asdict(cfg), "model_config": model.config(), "state_dict": model.state_dict()}
        record = {"config": asdict(cfg), "model_config": model.config(), "scenario": "complete",
                  "best_epoch": 1, "best_val_score": 0.5, "epochs_run": 1,
                  "n_train_samples": 20, "n_valid_samples": 10, "n_test_samples": 10,
                  "train_label_counts": {"waterlevel": 20}, "valid_label_counts": {"waterlevel": 10}}
        return cfg, blob, record

    def test_window_or_scaling_mismatch_is_rejected(self):
        p = prepared_fixture()
        cfg, blob, record = self.source(p)
        validate_reference(p, cfg, record, blob)
        for key, value in (("seq_length", 480), ("target_scaling", "observed")):
            changed = copy.deepcopy(cfg)
            setattr(changed, key, value)
            with self.assertRaisesRegex(ValueError, key):
                validate_reference(p, changed, record, blob)

    def test_reuse_preserves_exact_weights_and_never_constructs_q_scale(self):
        p = prepared_fixture()
        cfg, blob, record = self.source(p)
        p.stats["flow"] = {}
        p.targets_raw["flow"][:] = np.nan
        with tempfile.TemporaryDirectory() as directory:
            source, output = Path(directory) / "old", Path(directory) / "new"
            source.mkdir()
            torch.save(blob, source / "model.pt")
            (source / "run.json").write_text(json.dumps(record), encoding="utf-8")
            result = reuse_h_checkpoint(p, {"smoke": False}, cfg, source, output, "test-fingerprint")
            self.assertTrue(all(torch.equal(value, result["state_dict"][key]) for key, value in blob["state_dict"].items()))
            self.assertEqual(set(result["target_mean"]), {"waterlevel"})
            self.assertFalse(result["scaling_info"]["flow_information_used"])
            done = json.loads((output / "training_done.json").read_text(encoding="utf-8"))
            self.assertFalse(done["training_performed_this_run"])
            self.assertTrue(done["source_checkpoint"]["weights_reused_without_finetuning"])
            with self.assertRaisesRegex(ValueError, "指纹"):
                reuse_h_checkpoint(p, {"smoke": False}, cfg, source, output, "different-fingerprint")


if __name__ == "__main__":
    unittest.main()
