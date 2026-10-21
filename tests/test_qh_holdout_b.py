"""B组边界检查；不执行模型训练，也不启动GPU实验。"""

import contextlib
import copy
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import torch

from cross_section_experiment.inference import InferenceWindows
from pipeline.dataset import WindowDataset
from test_cross_section86 import prepared_fixture
from training.qh_holdout_b import (
    PROTOCOL, audit_datasets, hidden_label_view, load_reference, main, qh_masks, train_b,
    train_c_q, require_b_completed,
)
from training.trainer import TrainConfig, ValidationCheckpoint, train_model, validation_selection_score


def settings():
    return {"sequence_length": 4, "train_stride": 2, "valid_stride": 1,
            "test_stride": 1, "batch_size": 64, "hidden_size": 4,
            "lstm_epochs": 30, "attribute_set": "extended", "model_seeds": [1, 2],
            "target_scaling": "physical"}


def write_source(root):
    """构造小型C完成清单；文件占位只用于核对名单和入口安全性。"""
    basins = [f"01{i:06d}" for i in range(6)]
    scenario = {"tag": "q50_mask42", "ratio": .5, "mask_seed": 42,
                "heldout": basins[:3], "retained": basins[3:]}
    experiment = {"protocol": "original86_cross_section_train_valid_q_hidden_v1",
                  "settings": settings(), "basins": basins, "scenarios": [scenario],
                  "experiment_fingerprint": "fixture"}
    root.mkdir(parents=True)
    (root / "experiment.json").write_text(json.dumps(experiment), encoding="utf-8")
    (root / "completed_both.json").write_text("{}", encoding="utf-8")
    for seed in settings()["model_seeds"]:
        folder = root / "stage2" / scenario["tag"] / f"seed{seed}" / "dual_head"
        folder.mkdir(parents=True)
        (folder / "done.json").write_text(json.dumps({
            "heldout": scenario["heldout"], "model_seed": seed, "variant": "dual_head",
            "run_fingerprint": f"seed{seed}"}), encoding="utf-8")
        (folder / "training_done.json").write_text(json.dumps({
            "model_config": {"architecture": "dual_head"}, "run_fingerprint": f"seed{seed}"}),
            encoding="utf-8")
        for name in ("training_history.json", "model.pt", "prediction.npz", "metrics.csv"):
            (folder / name).touch()
    return experiment


class QHHoldoutBoundaries(unittest.TestCase):
    def setUp(self):
        self.prepared = prepared_fixture()
        self.held = self.prepared.basins[:3]
        self.retained = self.prepared.basins[3:]
        self.scenario = {"heldout": self.held, "retained": self.retained}

    def test_both_tasks_absent_in_train_and_validation_test_q_unchanged(self):
        p = self.prepared
        view = hidden_label_view(p, self.held, "physical")
        report = audit_datasets(view, self.scenario, settings(), "physical")
        for split in ("train", "valid"):
            self.assertFalse(set(report[split]["basins"]) & set(self.held))
        train, valid = qh_masks(p, self.held)
        for b in self.held:
            bi = p.basin_index[b]
            for split, masks in (("train", train), ("valid", valid)):
                lo, hi = p.split_range(b, split)
                for task in ("flow", "waterlevel"):
                    self.assertTrue(masks[task][bi, lo:hi + 1].all())
            lo, hi = p.split_range(b, "test")
            np.testing.assert_array_equal(view.targets_raw["flow"][bi, lo:hi + 1],
                                          p.targets_raw["flow"][bi, lo:hi + 1])
            self.assertFalse(train["flow"][bi, lo:hi + 1].any())
            self.assertFalse(valid["waterlevel"][bi, lo:hi + 1].any())
            self.assertTrue(np.isfinite(p.targets_raw["waterlevel"][bi]).all())

    def test_hidden_qh_and_h_statistics_cannot_change_b_training_data(self):
        p, poisoned = self.prepared, copy.deepcopy(self.prepared)
        for b in self.held:
            bi = p.basin_index[b]
            for task in ("flow", "waterlevel"):
                poisoned.targets_raw[task][bi, :76] = 1e12
                poisoned.stats[task][b] = {"mean": float("nan"), "std": float("nan")}
        views = [hidden_label_view(x, self.held, "physical") for x in (p, poisoned)]
        scales = [v.make_scaling("physical", self.retained) for v in views]
        for task in ("flow", "waterlevel"):
            np.testing.assert_array_equal(scales[0].mean[task], scales[1].mean[task])
            np.testing.assert_array_equal(scales[0].std[task], scales[1].std[task])
        for split in ("train", "valid"):
            datasets = [WindowDataset(v, split, 4, 1, scaling=s) for v, s in zip(views, scales)]
            np.testing.assert_array_equal(datasets[0].basin_idx, datasets[1].basin_idx)
            np.testing.assert_array_equal(datasets[0].target_pos, datasets[1].target_pos)
            ids = np.arange(len(datasets[0]))
            for left, right in zip(datasets[0][ids], datasets[1][ids]):
                np.testing.assert_array_equal(left.numpy(), right.numpy())

    def test_observed_retains_declared_q_scale_but_never_target_h_scale(self):
        p = self.prepared
        view = hidden_label_view(p, self.held, "observed")
        for b in self.held:
            self.assertEqual(view.stats["flow"][b], p.stats["flow"][b])
            self.assertEqual(view.stats["waterlevel"][b], {"mean": 0., "std": 1., "n": 0})
        before = InferenceWindows(p, "test", 4, 1, ("flow", "waterlevel"))
        after = InferenceWindows(view, "test", 4, 1, ("flow", "waterlevel"))
        np.testing.assert_array_equal(before.basin_idx, after.basin_idx)
        np.testing.assert_array_equal(before.target_pos, after.target_pos)

    def test_training_call_preserves_seed_architecture_and_masks_without_training(self):
        reference = {"scenario": self.scenario, "model_seed": 2}
        view = hidden_label_view(self.prepared, self.held, "physical")
        with patch("training.qh_holdout_b.train_model", return_value="mocked") as trainer:
            self.assertEqual(train_b(view, settings(), reference, "physical", "cpu"), "mocked")
        cfg = trainer.call_args.args[1]
        self.assertEqual((cfg.architecture, cfg.model_seed, cfg.max_epochs), ("dual_head", 2, 30))
        self.assertEqual(cfg.task_weights, {"flow": 1., "waterlevel": 1.})
        self.assertEqual(trainer.call_args.kwargs["track_selection_tasks"], ("flow",))
        self.assertEqual(trainer.call_args.kwargs["fit_basins"], self.retained)
        for key in ("hidden", "hidden_valid"):
            masks = trainer.call_args.kwargs[key]
            np.testing.assert_array_equal(masks["flow"], masks["waterlevel"])

    def test_reference_reuses_exact_saved_lists_and_default_entry_cannot_train(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            original = write_source(root / "C")
            with patch("numpy.random.default_rng", side_effect=AssertionError("禁止重新抽样")):
                _, runs = load_reference(root / "C", "physical")
            self.assertEqual([r["model_seed"] for r in runs], [1, 2])
            self.assertEqual(runs[0]["scenario"], original["scenarios"][0])
            config = root / "config.json"
            config.write_text(json.dumps({"protocol": PROTOCOL, "output_root": str(root / "B"),
                                          "sources": {"physical": str(root / "C")}}), encoding="utf-8")
            with patch("training.qh_holdout_b.PreparedData") as prepare, \
                    patch("training.qh_holdout_b.execute") as execute, \
                    contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(main(["--config", str(config)]), 0)
                prepare.assert_not_called()
                execute.assert_not_called()
            self.assertFalse((root / "B").exists())
            self.assertFalse((root / "B_q_selected_bc").exists())
            done = root / "C/stage2/q50_mask42/seed1/dual_head/done.json"
            data = json.loads(done.read_text(encoding="utf-8"))
            data["heldout"] = list(reversed(data["heldout"]))
            done.write_text(json.dumps(data), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "不一致"):
                load_reference(root / "C", "physical")

    def test_c_q_keeps_training_h_but_excludes_target_validation_h(self):
        view = hidden_label_view(self.prepared, self.held, "physical", hide_waterlevel=False)
        report = audit_datasets(view, self.scenario, settings(), "physical", group="C")
        self.assertTrue(set(report["train"]["basins"]) & set(self.held))
        self.assertFalse(set(report["valid"]["basins"]) & set(self.held))
        with patch("training.qh_holdout_b.train_model", return_value="mocked") as trainer:
            train_c_q(view, settings(), {"scenario": self.scenario, "model_seed": 2}, "physical", "cpu")
        self.assertEqual(trainer.call_args.kwargs["selection_tasks"], ("flow",))
        self.assertEqual(set(trainer.call_args.kwargs["hidden"]), {"flow"})
        self.assertEqual(set(trainer.call_args.kwargs["hidden_valid"]), {"flow", "waterlevel"})

    def test_second_stage_refuses_to_start_before_b_is_complete(self):
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp) / "C"
            experiment = write_source(source)
            _, runs = load_reference(source, "physical")
            references = [("physical", source, experiment, runs)]
            with self.assertRaisesRegex(FileNotFoundError, "B主实验尚未完成"):
                require_b_completed(references, Path(tmp) / "B", "not_started")

    def test_pipeline_orders_all_b_before_q_bc_without_training(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            write_source(root / "C")
            config = root / "config.json"
            config.write_text(json.dumps({"protocol": PROTOCOL, "output_root": str(root / "B"),
                                          "q_output_root": str(root / "Q_BC"),
                                          "sources": {"physical": str(root / "C")}}), encoding="utf-8")
            events = []
            with patch("training.qh_holdout_b.PreparedData", return_value=self.prepared), \
                    patch("training.qh_holdout_b.validate_population"), \
                    patch("training.qh_holdout_b.execute", side_effect=lambda *a: events.append("B_QH")), \
                    patch("training.qh_holdout_b.require_b_completed", side_effect=lambda *a: events.append("gate")), \
                    patch("training.qh_holdout_b.execute_q_selected", side_effect=lambda *a: events.append("Q_" + a[-1])), \
                    patch("training.qh_holdout_b.mark_completed"), \
                    contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(main(["--config", str(config), "--run", "--device", "cpu"]), 0)
            self.assertEqual(events, ["B_QH", "B_QH", "gate", "Q_B", "Q_B", "Q_C", "Q_C"])


class ValidationSelectionChecks(unittest.TestCase):
    def test_q_selection_ignores_h_and_task_average_is_preserved(self):
        self.assertAlmostEqual(validation_selection_score({"flow": .4, "waterlevel": .8},
                                                         ("flow", "waterlevel")), .6)
        self.assertEqual(validation_selection_score({"flow": .4, "waterlevel": np.nan}, ("flow",)), .4)
        with self.assertRaises(ValueError):
            validation_selection_score({"flow": .4}, ("waterlevel",))

    def test_alternate_checkpoint_is_an_independent_weight_snapshot(self):
        model = torch.nn.Linear(1, 1, bias=False)
        with torch.no_grad():
            model.weight.fill_(1.)
        q, qh = ValidationCheckpoint(("flow",)), ValidationCheckpoint(("flow", "waterlevel"))
        q.update({"flow": .8, "waterlevel": .2}, 1, model)
        qh.update({"flow": .8, "waterlevel": .2}, 1, model)
        with torch.no_grad():
            model.weight.fill_(2.)
        q.update({"flow": .7, "waterlevel": .9}, 2, model)
        qh.update({"flow": .7, "waterlevel": .9}, 2, model)
        self.assertEqual((q.best_epoch, qh.best_epoch), (1, 2))
        self.assertEqual(q.state_dict["weight"].item(), 1.)
        self.assertEqual(qh.state_dict["weight"].item(), 2.)

    def test_training_loop_routes_selection_using_stubs_no_optimization(self):
        class EpochMarker(torch.nn.Module):
            tasks = ("flow", "waterlevel")

            def __init__(self):
                super().__init__()
                self.marker = torch.nn.Parameter(torch.tensor(0.))

            def config(self):
                return {"architecture": "dual_head"}

        def fake_epoch(model, *args):
            # 仅写入轮次标记，不计算损失、梯度或执行优化器更新。
            with torch.no_grad():
                model.marker.add_(1.)
            return {"loss": 0., "n_updates": 0, "label_exposures": {"flow": 0, "waterlevel": 0}}

        def fake_nse(result, task):
            values = {1: {"flow": .8, "waterlevel": .2}, 2: {"flow": .7, "waterlevel": .9}}
            return np.array([values[result["epoch"]][task]])

        p = prepared_fixture()
        cfg = TrainConfig("dual_head", seq_length=4, max_epochs=2, lr_schedule="none", device="cpu")
        for selected, expected_epoch in ((None, 2), (("flow",), 1)):
            with patch("training.trainer.build_model", return_value=EpochMarker()), \
                    patch("training.trainer.train_epoch", side_effect=fake_epoch), \
                    patch("training.trainer.predict", side_effect=lambda model, *a: {"epoch": int(model.marker.item())}), \
                    patch("training.trainer._nse_by_basin", side_effect=fake_nse):
                result = train_model(p, cfg, verbose=False, selection_tasks=selected,
                                     track_selection_tasks=("flow",))
            self.assertEqual(result["best_epoch"], expected_epoch)
            self.assertEqual(result["tracked_checkpoint"]["best_epoch"], 1)
            self.assertEqual(result["tracked_checkpoint"]["state_dict"]["marker"].item(), 1.)


if __name__ == "__main__":
    unittest.main()
