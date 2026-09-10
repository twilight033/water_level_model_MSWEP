"""数据管线正确性测试。

对应修订方案的验证清单：在跑任何正式实验之前，这些断言必须全部通过。
每一条都直接对应一个曾经存在于旧实现中的缺陷。
"""

import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT)]

from models.lstm_models import ARCHITECTURES, build_model  # noqa: E402
from pipeline.dataset import PreparedData, WindowDataset  # noqa: E402
from pipeline.masking import assert_train_only, build_hidden  # noqa: E402
from pipeline.normalization import compute_norm_stats, load_norm_stats  # noqa: E402
from pipeline.splits import load_splits  # noqa: E402
from training.trainer import masked_loss  # noqa: E402

SEQ = 168
_PREPARED = None


def prepared():
    """PreparedData 载入约 1 秒，全部用例共用一份。"""
    global _PREPARED
    if _PREPARED is None:
        _PREPARED = PreparedData()
    return _PREPARED


class SplitFreezeTests(unittest.TestCase):
    """划分边界必须与窗口长度、掩膜、任务组合完全无关。"""

    def test_bounds_independent_of_seq_length_and_tasks(self):
        prep = prepared()
        basin = list(prep.splits["splits"])[0]
        reference = prep.split_range(basin, "train")
        for seq_length in (56, 168, 480):
            for tasks in (("flow",), ("waterlevel",), ("flow", "waterlevel")):
                ds = WindowDataset(prep, "train", seq_length, 8,
                                   tasks=tasks, basins=[basin])
                self.assertEqual(ds.prepared.split_range(basin, "train"), reference)

    def test_bounds_independent_of_mask(self):
        prep = prepared()
        hidden, _ = build_hidden(prep, {"flow": 0.7}, mechanism="mcar", mask_seed=1)
        for basin in list(prep.splits["splits"])[:5]:
            before = prep.split_range(basin, "test")
            WindowDataset(prep, "train", SEQ, 8, basins=[basin], hidden=hidden)
            self.assertEqual(prep.split_range(basin, "test"), before)

    def test_split_target_times_are_disjoint(self):
        """训练、验证、测试的目标时刻集合互不相交（旧实现会重叠）。"""
        prep = prepared()
        for basin in list(prep.splits["splits"])[:8]:
            seen = set()
            for split, step in (("train", 8), ("valid", 8), ("test", 1)):
                ds = WindowDataset(prep, split, SEQ, step, basins=[basin])
                positions = set(ds.target_pos.tolist())
                self.assertFalse(seen & positions,
                                 f"流域 {basin} 的 {split} 与前序分段目标时刻重叠")
                seen |= positions

    def test_embargo_keeps_test_inputs_off_training_targets(self):
        """禁运期保证测试样本的输入窗口不覆盖训练段目标时刻。"""
        prep = prepared()
        for basin in list(prep.splits["splits"])[:5]:
            _, train_end = prep.split_range(basin, "train")
            ds = WindowDataset(prep, "test", SEQ, 1, basins=[basin])
            earliest_input = int(ds.target_pos.min()) - SEQ
            self.assertGreater(earliest_input, train_end)


class NormalizationTests(unittest.TestCase):
    def test_stats_come_from_training_window_only(self):
        prep = prepared()
        stats = load_norm_stats()
        basin = list(prep.splits["splits"])[0]
        bi = prep.basin_index[basin]
        lo, hi = prep.split_range(basin, "train")
        vals = prep.targets_raw["flow"][bi, lo:hi + 1]
        vals = vals[~np.isnan(vals)]
        self.assertAlmostEqual(stats["flow"][basin]["mean"], float(vals.mean()), places=3)
        self.assertEqual(stats["flow"][basin]["n"], int(vals.size))

    def test_stats_identical_regardless_of_mask(self):
        """统计量只依赖原始标签，任何缺失场景下都必须相同。"""
        recomputed = compute_norm_stats()
        stored = load_norm_stats()
        for task in ("flow", "waterlevel"):
            for basin, v in stored[task].items():
                self.assertAlmostEqual(v["mean"], recomputed[task][basin]["mean"], places=6)
                self.assertAlmostEqual(v["std"], recomputed[task][basin]["std"], places=6)


class MaskingTests(unittest.TestCase):
    def test_mask_touches_training_period_only(self):
        prep = prepared()
        hidden, _ = build_hidden(prep, {"flow": 0.5, "waterlevel": 0.3},
                                 mechanism="mcar", mask_seed=5)
        assert_train_only(prep, hidden)          # 越界会抛异常

    def test_valid_and_test_labels_unchanged(self):
        prep = prepared()
        base_valid = WindowDataset(prep, "valid", SEQ, 8).label_counts()
        base_test = WindowDataset(prep, "test", SEQ, 8).label_counts()
        hidden, _ = build_hidden(prep, {"flow": 0.7}, mechanism="mcar", mask_seed=5)
        self.assertEqual(WindowDataset(prep, "valid", SEQ, 8, hidden=hidden).label_counts(),
                         base_valid)
        self.assertEqual(WindowDataset(prep, "test", SEQ, 8, hidden=hidden).label_counts(),
                         base_test)

    def test_realized_ratio_matches_target(self):
        prep = prepared()
        for ratio in (0.3, 0.5, 0.7):
            _, stats = build_hidden(prep, {"flow": ratio}, mechanism="mcar", mask_seed=9)
            got = stats["per_task"]["flow"]["realized_ratio_mean"]
            self.assertAlmostEqual(got, ratio, places=3)

    def test_basin_holdout_removes_whole_basins_only(self):
        """按流域留出：被留出流域该任务训练标签全删，其余流域一个不动，
        另一任务完全不受影响。"""
        prep = prepared()
        hidden, stats = build_hidden(prep, {"flow": 0.5},
                                     mechanism="basin_holdout", mask_seed=42)
        assert_train_only(prep, hidden)
        held = set(stats["per_task"]["flow"]["held_out_basins"])
        self.assertEqual(len(held), round(0.5 * len(prep.splits["splits"])))
        self.assertNotIn("waterlevel", hidden)      # 水位标签一个不删

        for basin in list(prep.splits["splits"])[:20]:
            bi = prep.basin_index[basin]
            lo, hi = prep.split_range(basin, "train")
            window = prep.targets["flow"][bi, lo:hi + 1]
            valid = ~np.isnan(window)
            masked = hidden["flow"][bi, lo:hi + 1]
            if basin in held:
                self.assertTrue(bool((masked == valid).all()),
                                f"{basin} 被留出但未删干净")
            else:
                self.assertFalse(bool(masked.any()), f"{basin} 未被留出却被删了标签")

    def test_basin_holdout_keeps_waterlevel_supervision(self):
        """留出情景下水位监督必须完好——这是替代命题成立的前提。"""
        prep = prepared()
        base = WindowDataset(prep, "train", SEQ, 8).label_counts()
        hidden, _ = build_hidden(prep, {"flow": 0.7},
                                 mechanism="basin_holdout", mask_seed=42)
        after = WindowDataset(prep, "train", SEQ, 8, hidden=hidden).label_counts()
        self.assertEqual(after["waterlevel"], base["waterlevel"])
        self.assertLess(after["flow"], base["flow"])

    def test_mask_is_reproducible_across_calls(self):
        """掩膜种子必须跨调用（以及跨进程）稳定，否则配对比较不成立。"""
        prep = prepared()
        a, _ = build_hidden(prep, {"flow": 0.4}, mechanism="mcar", mask_seed=11)
        b, _ = build_hidden(prep, {"flow": 0.4}, mechanism="mcar", mask_seed=11)
        self.assertTrue(np.array_equal(a["flow"], b["flow"]))


class SampleSetTests(unittest.TestCase):
    def test_single_task_samples_are_subset_of_multitask(self):
        """单任务模型的训练样本恰好是双头样本中该任务标签有效的子集。"""
        prep = prepared()
        basins = list(prep.splits["splits"])[:10]
        multi = WindowDataset(prep, "train", SEQ, 8, basins=basins)
        single = WindowDataset(prep, "train", SEQ, 8, tasks=("flow",), basins=basins)
        multi_pairs = set(zip(multi.basin_idx.tolist(), multi.target_pos.tolist()))
        single_pairs = set(zip(single.basin_idx.tolist(), single.target_pos.tolist()))
        self.assertTrue(single_pairs <= multi_pairs)
        self.assertEqual(single.label_counts()["flow"], multi.label_counts()["flow"])

    def test_no_nan_in_inputs_or_targets(self):
        prep = prepared()
        ds = WindowDataset(prep, "train", SEQ, 8, basins=list(prep.splits["splits"])[:5])
        x, c, y, m, _, _ = ds[np.arange(min(2048, len(ds)))]
        for tensor, name in ((x, "强迫"), (c, "属性"), (y, "目标"), (m, "掩膜")):
            self.assertFalse(bool(torch.isnan(tensor).any()), f"{name}含 NaN")


class TimeAxisTests(unittest.TestCase):
    """导出时序的时间戳必须来自真实数据，而不是人为构造的等间隔序列。"""

    def _predict_one_basin(self, basin):
        from pipeline.dataset import make_loader
        from training.trainer import predict

        prep = prepared()
        ds = WindowDataset(prep, "test", SEQ, 1, basins=[basin])
        model = build_model("dual_head", forcing_size=prep.forcing.shape[-1],
                            attr_size=prep.attrs.shape[-1])
        loader = make_loader(ds, 4096, shuffle=False)
        return prep, predict(model, loader, ("flow", "waterlevel"), "cpu")

    def test_exported_times_match_grid_and_raw_targets(self):
        from evaluation.metrics import build_basin_series

        prep = prepared()
        basin = list(prep.splits["splits"])[0]
        prep, result = self._predict_one_basin(basin)
        bi = prep.basin_index[basin]
        frame = build_basin_series(prep, result, "flow", bi)

        # 时间戳逐条来自强迫时间轴，而不是 pd.date_range 生成
        expected = prep.grid[frame["target_pos"].to_numpy()]
        self.assertTrue((frame["time"].to_numpy() == expected.to_numpy()).all())

        # 观测值逐条等于原始目标数组在该位置的取值
        raw = prep.targets_raw["flow"][bi, frame["target_pos"].to_numpy()]
        exported = frame["obs_flow"].to_numpy()
        both_valid = np.isfinite(raw) & np.isfinite(exported)
        self.assertTrue(both_valid.any())
        np.testing.assert_allclose(exported[both_valid], raw[both_valid],
                                   rtol=1e-4, atol=1e-4)

    def test_target_time_equals_window_end(self):
        """目标时刻恰为输入窗口右端点：t0 + L*3h，与文档算例一致。"""
        prep = prepared()
        basin = list(prep.splits["splits"])[0]
        ds = WindowDataset(prep, "test", SEQ, 1, basins=[basin])
        pos = int(ds.target_pos[0])
        first_input = prep.grid[pos + int(ds._offsets[0])]
        last_input = prep.grid[pos + int(ds._offsets[-1])]
        target_time = prep.grid[pos]
        self.assertEqual(target_time - first_input, pd.Timedelta(hours=SEQ * 3))
        self.assertEqual(target_time - last_input, pd.Timedelta(hours=3))


class AttributeSourceTests(unittest.TestCase):
    """直读 CSV 必须与既有缓存完全一致，且不得引入径流导出的属性。"""

    def test_base_set_reproduces_existing_cache(self):
        import pandas as pd
        from pipeline.attribute_sources import get_attribute_table
        from pipeline.paths import EXPORT_DIR

        cached = pd.read_parquet(EXPORT_DIR / "attributes_86.parquet")
        rebuilt, _ = get_attribute_table("base")
        rebuilt = rebuilt.reindex(cached.index)
        self.assertEqual(list(cached.columns), list(rebuilt.columns))
        np.testing.assert_allclose(cached.to_numpy(float), rebuilt.to_numpy(float),
                                   rtol=1e-6, atol=1e-6)

    def test_extended_is_strict_superset_of_base(self):
        from pipeline.attribute_sources import get_attribute_table

        base, _ = get_attribute_table("base")
        ext, _ = get_attribute_table("extended")
        self.assertTrue(set(base.columns) < set(ext.columns))
        for col in base.columns:
            np.testing.assert_allclose(base[col].to_numpy(float),
                                       ext[col].to_numpy(float), rtol=1e-6, atol=1e-6)

    def test_no_flow_derived_attributes(self):
        """由实测径流导出的属性绝不能进入输入——留出情景假设该流域无径流数据。"""
        from pipeline import attribute_sources as src

        for spec in src.EXTENDED_SPECS:
            self.assertNotIn(spec.column, src.FLOW_DERIVED_COLUMNS)
            self.assertNotIn(spec.file, src.FLOW_DERIVED_FILES)
            self.assertNotIn(spec.column, src.CONSTANT_COLUMNS)
        # 构造时即应拒绝
        with self.assertRaises(ValueError):
            src.AttrSpec("bfi", "attributes_gageii_Hydro.csv", "BFI_AVE")
        with self.assertRaises(ValueError):
            src.AttrSpec("x", "attributes_gageii_FlowRec.csv", "ACTIVE09")


class PhysicalScalingTests(unittest.TestCase):
    """物理归一化必须真正不含被留出流域的任何径流观测。"""

    def _fixture(self):
        import pandas as pd
        attr = pd.DataFrame({"area": [10.0, 100.0, 1000.0, 5000.0],
                             "p_mean": [2.0, 3.0, 2.5, 4.0]},
                            index=["a", "b", "c", "d"])
        observed = {"a": {"mean": 1.0, "std": 1.4}, "b": {"mean": 12.0, "std": 16.0},
                    "c": {"mean": 90.0, "std": 130.0}, "d": {"mean": 700.0, "std": 950.0}}
        return attr, observed

    def test_held_out_basin_observations_do_not_affect_its_scale(self):
        from pipeline.normalization import physical_flow_scale

        attr, observed = self._fixture()
        allb, fit = ["a", "b", "c", "d"], ["a", "b", "c"]      # d 被留出
        mean1, std1, _ = physical_flow_scale(attr, observed, fit, allb)

        tampered = {k: dict(v) for k, v in observed.items()}
        tampered["d"] = {"mean": 1.0, "std": 1.0}              # 篡改留出流域的观测
        mean2, std2, _ = physical_flow_scale(attr, tampered, fit, allb)
        np.testing.assert_allclose(mean1, mean2)
        np.testing.assert_allclose(std1, std2)

    def test_rejects_unknown_or_empty_fit_basins(self):
        from pipeline.normalization import physical_flow_scale

        attr, observed = self._fixture()
        with self.assertRaises(ValueError):
            physical_flow_scale(attr, observed, [], ["a", "b"])
        with self.assertRaises(ValueError):
            physical_flow_scale(attr, observed, ["zz"], ["a", "b"])

    def test_waterlevel_scale_unchanged_in_physical_mode(self):
        """水位仍用实测统计量——留出情景的前提就是该流域有水位记录。"""
        prep = prepared()
        physical = prep.make_scaling("physical", fit_basins=prep.basins[:40])
        np.testing.assert_allclose(physical.mean["waterlevel"],
                                   prep.default_scaling.mean["waterlevel"])
        np.testing.assert_allclose(physical.std["waterlevel"],
                                   prep.default_scaling.std["waterlevel"])
        # 径流尺度则必须改变
        self.assertFalse(np.allclose(physical.mean["flow"],
                                     prep.default_scaling.mean["flow"]))

    def test_sample_set_identical_across_scalings(self):
        """尺度只改目标数值，不改样本准入——两种模式必须给出相同的样本集。"""
        prep = prepared()
        basins = list(prep.splits["splits"])[:10]
        physical = prep.make_scaling("physical", fit_basins=prep.basins)
        a = WindowDataset(prep, "train", SEQ, 8, basins=basins)
        b = WindowDataset(prep, "train", SEQ, 8, basins=basins, scaling=physical)
        self.assertTrue(np.array_equal(a.basin_idx, b.basin_idx))
        self.assertTrue(np.array_equal(a.target_pos, b.target_pos))

    def test_gauged_basins_excludes_held_out(self):
        from training.trainer import gauged_basins

        prep = prepared()
        hidden, stats = build_hidden(prep, {"flow": 0.5},
                                     mechanism="basin_holdout", mask_seed=42)
        held = set(stats["per_task"]["flow"]["held_out_basins"])
        fit = set(gauged_basins(prep, hidden))
        self.assertFalse(fit & held, "留出流域混入了物理尺度的拟合样本")
        self.assertEqual(fit, set(prep.splits["splits"]) - held)


class MaskedLossGradientTests(unittest.TestCase):
    """掩膜为 0 的任务不得产生梯度，且不得污染另一任务。"""

    def _run(self, mask_row):
        torch.manual_seed(0)
        model = build_model("dual_head", forcing_size=3, attr_size=15)
        x, c, y = torch.randn(8, 32, 3), torch.randn(8, 15), torch.randn(8, 2)
        m = torch.tensor([mask_row] * 8)
        total, _ = masked_loss(model(x, c), y, m, ("flow", "waterlevel"),
                               {"flow": 1.0, "waterlevel": 1.0}, "per_valid")
        total.backward()
        return model

    @staticmethod
    def _no_grad(param):
        return param.grad is None or bool((param.grad == 0).all())

    def test_flow_missing(self):
        model = self._run([0.0, 1.0])
        self.assertTrue(self._no_grad(model.fc_flow.weight))
        self.assertTrue(bool((model.fc_waterlevel.weight.grad != 0).any()))
        self.assertTrue(bool(torch.isfinite(model.encoder.lstm.weight_ih_l0.grad).all()))

    def test_waterlevel_missing(self):
        model = self._run([1.0, 0.0])
        self.assertTrue(self._no_grad(model.fc_waterlevel.weight))
        self.assertTrue(bool((model.fc_flow.weight.grad != 0).any()))

    def test_both_missing_yields_finite_zero_loss(self):
        torch.manual_seed(0)
        model = build_model("dual_head", forcing_size=3, attr_size=15)
        x, c, y = torch.randn(4, 32, 3), torch.randn(4, 15), torch.randn(4, 2)
        m = torch.zeros(4, 2)
        total, _ = masked_loss(model(x, c), y, m, ("flow", "waterlevel"),
                               {"flow": 1.0, "waterlevel": 1.0}, "per_valid")
        self.assertTrue(torch.isfinite(total))
        self.assertEqual(float(total), 0.0)


class ModelTests(unittest.TestCase):
    def test_all_architectures_forward(self):
        for arch in ARCHITECTURES:
            model = build_model(arch, forcing_size=3, attr_size=15)
            out = model(torch.randn(2, 32, 3), torch.randn(2, 15))
            self.assertEqual(set(out), set(model.tasks))
            for value in out.values():
                self.assertEqual(value.shape, (2, 1))
                self.assertTrue(bool(torch.isfinite(value).all()))

    def test_dual_head_heads_are_independent(self):
        model = build_model("dual_head", forcing_size=3, attr_size=15).eval()
        x, c = torch.randn(4, 32, 3), torch.randn(4, 15)
        before = model(x, c)["flow"]
        with torch.no_grad():
            model.fc_waterlevel.bias.add_(2.0)
        torch.testing.assert_close(model(x, c)["flow"], before)

    def test_capacity_matched_matches_wl2d_shape(self):
        """参数量匹配对照与 WL2D 的分支形状一致，差额仅为额外的标量投影头。"""
        wl2d = build_model("wl2d", forcing_size=3, attr_size=15)
        matched = build_model("capacity_matched", forcing_size=3, attr_size=15)
        hidden_size = wl2d.encoder.lstm.hidden_size
        self.assertEqual(matched.n_parameters() - wl2d.n_parameters(), hidden_size + 1)
        self.assertEqual(matched.proj[0].weight.shape, wl2d.proj[0].weight.shape)

    def test_capacity_matched_flow_head_ignores_waterlevel_head(self):
        """对照模型的径流分支不读水位预测：改水位头不应影响径流输出。"""
        model = build_model("capacity_matched", forcing_size=3, attr_size=15).eval()
        x, c = torch.randn(4, 32, 3), torch.randn(4, 15)
        before = model(x, c)["flow"]
        with torch.no_grad():
            model.fc_waterlevel.bias.add_(3.0)
        torch.testing.assert_close(model(x, c)["flow"], before)

    def test_wl2d_flow_head_does_read_waterlevel_head(self):
        """作为对照的反证：WL2D 的径流输出确实依赖水位头。"""
        model = build_model("wl2d", forcing_size=3, attr_size=15).eval()
        x, c = torch.randn(4, 32, 3), torch.randn(4, 15)
        before = model(x, c)["flow"]
        with torch.no_grad():
            model.fc_up.bias.add_(3.0)
        self.assertFalse(torch.allclose(model(x, c)["flow"], before))


if __name__ == "__main__":
    unittest.main(verbosity=2)
