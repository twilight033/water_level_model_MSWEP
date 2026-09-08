"""验证独立预测头、共享梯度和三个训练入口的兼容性。"""

import contextlib
import importlib
import io
import math
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src" / "training"), str(ROOT / "src" / "others")]
from parallel_multitask import ParallelMultiTaskLSTM, build_multitask_model, model_metadata
import train_multitask


class ParallelModelTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(12)
        self.model = ParallelMultiTaskLSTM(3, hidden_size=8, dropout_rate=0)
        self.x = torch.randn(4, 6, 3)

    def test_both_output_shapes_including_single_sample(self):
        for size in (1, 4):
            for output in self.model(self.x[:size]):
                self.assertEqual(output.shape, (size, 1))
                self.assertTrue(torch.isfinite(output).all())

    def test_changing_one_head_does_not_change_other_prediction(self):
        self.model.eval()
        before_q, before_h = self.model(self.x)
        with torch.no_grad():
            self.model.fc_waterlevel.bias.add_(2)
        after_q, after_h = self.model(self.x)
        torch.testing.assert_close(after_q, before_q)
        torch.testing.assert_close(after_h, before_h + 2)
        with torch.no_grad():
            self.model.fc_flow.bias.add_(3)
        final_q, final_h = self.model(self.x)
        torch.testing.assert_close(final_h, after_h)
        torch.testing.assert_close(final_q, after_q + 3)

    def test_each_loss_updates_encoder_and_own_head_only(self):
        for index, own, other in ((0, self.model.fc_flow, self.model.fc_waterlevel),
                                  (1, self.model.fc_waterlevel, self.model.fc_flow)):
            self.model.zero_grad(set_to_none=True)
            self.model(self.x)[index].square().mean().backward()
            self.assertGreater(self.model.lstm.weight_ih_l0.grad.abs().sum().item(), 0)
            self.assertGreater(own.weight.grad.abs().sum().item(), 0)
            self.assertIsNone(other.weight.grad)
            self.assertIsNone(other.bias.grad)

    def test_no_cascade_parameters_and_checkpoint_roundtrip(self):
        metadata = model_metadata(self.model, "parallel")
        expected = sum(p.numel() for p in self.model.lstm.parameters()) + 2 * (8 + 1)
        self.assertEqual(metadata["num_parameters"], expected)
        self.assertFalse(any("wl_proj" in key for key in self.model.state_dict()))
        buffer = io.BytesIO()
        torch.save({**metadata, "model_state_dict": self.model.state_dict()}, buffer)
        buffer.seek(0)
        checkpoint = torch.load(buffer, weights_only=True)
        restored = ParallelMultiTaskLSTM(**checkpoint["model_config"])
        restored.load_state_dict(checkpoint["model_state_dict"], strict=True)
        for expected_output, actual in zip(self.model(self.x), restored(self.x)):
            torch.testing.assert_close(actual, expected_output)

    def test_invalid_architecture_and_cascade_option_are_rejected(self):
        for options in ({"architecture": "unknown"},
                        {"architecture": "parallel", "stop_gradient": True}):
            with self.assertRaises(ValueError):
                build_multitask_model(wl2d_class=None, input_size=3, **options)


class TrainingIntegrationTests(unittest.TestCase):
    def test_single_baseline_does_not_report_zero_uncertainty(self):
        result = dict(experiment_name="baseline_both_complete_seed42",
                      test_nse_flow=0.6, test_nse_waterlevel=0.7,
                      best_epoch=1, flow_missing_ratio=0, waterlevel_missing_ratio=0,
                      n_basins_flow=2, n_basins_waterlevel=2)
        for phase in ("mcar", "segment"):
            module = importlib.import_module(train_multitask.EXPERIMENT_MODULES[phase])
            aggregated = module.aggregate_repeat_results([result])
            self.assertEqual(aggregated["n_repeats"], 1)
            self.assertTrue(math.isnan(aggregated["test_nse_flow_std"]))
            self.assertTrue(math.isnan(aggregated["test_nse_waterlevel_std"]))

    def test_existing_training_loops_accept_parallel_model(self):
        for phase, module_name in train_multitask.EXPERIMENT_MODULES.items():
            with self.subTest(phase=phase):
                module = importlib.import_module(module_name)
                torch.manual_seed(7)
                model = build_multitask_model("parallel", module.MultiTaskLSTM,
                                             input_size=3, hidden_size=8, dropout_rate=0)
                self.assertIsInstance(model, ParallelMultiTaskLSTM)
                x, y = torch.randn(8, 6, 3), torch.randn(8, 2)
                basins = torch.zeros(8, dtype=torch.long)
                if phase == "main":
                    dataset = TensorDataset(x, y, basins)
                    loss = nn.MSELoss()
                else:
                    # 批次依次只有 Q 标签、只有水位标签、都没有、都有。
                    masks = torch.tensor([[1., 0.]] * 2 + [[0., 1.]] * 2
                                         + [[0., 0.]] * 2 + [[1., 1.]] * 2)
                    dataset = TensorDataset(x, y, masks, basins)
                    loss = nn.MSELoss(reduction="none")
                loader = DataLoader(dataset, batch_size=2, shuffle=False)
                optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
                before = model.lstm.weight_ih_l0.detach().clone()
                with patch.object(module, "DEVICE", torch.device("cpu")), \
                     contextlib.redirect_stdout(io.StringIO()), \
                     contextlib.redirect_stderr(io.StringIO()):
                    losses = module.train_epoch(model, optimizer, loader, loss, 1)
                    outputs = module.eval_model(model, loader)
                self.assertTrue(all(torch.isfinite(torch.tensor(value)) for value in losses))
                self.assertFalse(torch.equal(before, model.lstm.weight_ih_l0))
                self.assertTrue(all(len(output) for output in outputs))
                # 旧入口仍然能重建原级联架构。
                legacy = build_multitask_model("wl2d", module.MultiTaskLSTM,
                                              input_size=3, hidden_size=8)
                self.assertTrue(hasattr(legacy, "wl_proj"))

    def test_launcher_selects_parallel_in_all_phases_without_loading_data(self):
        for phase, module_name in train_multitask.EXPERIMENT_MODULES.items():
            with self.subTest(phase=phase), \
                 patch.object(train_multitask, "import_module") as importer, \
                 contextlib.redirect_stdout(io.StringIO()):
                train_multitask.main(["--experiment", phase, "--model-seed", "42"])
                importer.assert_called_once_with(module_name)
                kwargs = importer.return_value.main.call_args.kwargs
                self.assertEqual(kwargs["architecture"], "parallel")
                self.assertEqual(kwargs["model_seed"], 42)
                self.assertIn("parallel", kwargs["output_root"].parts)


if __name__ == "__main__":
    torch.set_num_threads(1)
    unittest.main()
