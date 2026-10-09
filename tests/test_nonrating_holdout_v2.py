"""验证99流域随机留出的训练、验证掩膜及抽样一致性。"""
import unittest
from types import SimpleNamespace

import numpy as np
import pandas as pd

from src.evaluation.train_nonrating14_external import (
    HOLDOUT_PROTOCOL, ExternalPrepared, fixed_mask, random_basin_holdout,
)


class HoldoutProtocolTest(unittest.TestCase):
    def test_actual_dataset_excludes_validation_q_but_keeps_h_and_test(self):
        from pipeline.dataset import WindowDataset
        from training.trainer import gauged_basins
        basins = [f"{i:08d}" for i in range(99)]
        grid = pd.date_range("2000-01-01", periods=30, freq="3h")
        raw = np.tile(np.arange(30, dtype=np.float32), (99, 1))
        splits = {b: {f"{s}_{edge}": str(grid[pos])
                      for s, lo, hi in (("train", 0, 9), ("valid", 10, 19), ("test", 20, 29))
                      for edge, pos in (("start", lo), ("end", hi))} for b in basins}
        prep = ExternalPrepared(grid, raw[:, :, None], pd.DataFrame({"area": np.ones(99)}, index=basins),
                                {"flow": raw.copy(), "waterlevel": raw.copy()}, basins, splits)
        train, _, held = random_basin_holdout(prep, .7, 42)
        _, valid = fixed_mask(prep, held, True)
        self.assertEqual(set(gauged_basins(prep, train)), set(basins) - set(held))
        for arch_tasks in (("flow",), ("flow", "waterlevel")):
            for split, mask in (("train", train), ("valid", valid)):
                counts = WindowDataset(prep, split, 2, 1, tasks=arch_tasks, hidden=mask).label_counts()
                gauged = [b for b in basins if b not in held]
                expected = WindowDataset(prep, split, 2, 1, tasks=("flow",), basins=gauged).label_counts()
                self.assertEqual(counts["flow"], expected["flow"])
                if "waterlevel" in arch_tasks:
                    held_counts = WindowDataset(prep, split, 2, 1, tasks=arch_tasks, basins=held, hidden=mask).label_counts()
                    self.assertEqual(held_counts["flow"], 0)
                    self.assertGreater(held_counts["waterlevel"], 0)
            counts = WindowDataset(prep, "test", 2, 1, tasks=arch_tasks, basins=held).label_counts()
            self.assertGreater(counts["flow"], 0)

    def test_same_basins_hidden_in_train_and_valid(self):
        basins = [f"{i:08d}" for i in range(99)]
        raw = np.ones((99, 12), dtype=np.float32)
        raw[:, 1] = np.nan
        raw[:, 6] = np.nan
        ranges = {"train": (0, 4), "valid": (5, 8), "test": (9, 11)}
        prep = SimpleNamespace(
            basins=basins, grid=np.arange(12),
            basin_index={b: i for i, b in enumerate(basins)},
            targets_raw={"flow": raw}, split_range=lambda b, s: ranges[s],
        )
        for ratio in (.3, .5, .7):
            for seed in (42, 123, 456):
                with self.subTest(ratio=ratio, seed=seed):
                    train, stats, held = random_basin_holdout(prep, ratio, seed)
                    expected_train, valid = fixed_mask(prep, held, True)
                    np.testing.assert_array_equal(train["flow"], expected_train["flow"])
                    self.assertEqual(len(held), round(99 * ratio))
                    self.assertEqual(stats["meta"]["protocol"], HOLDOUT_PROTOCOL)
                    self.assertEqual(set(train), {"flow"})
                    self.assertEqual(set(valid), {"flow"})
                    self.assertFalse(train["flow"][:, 5:].any())
                    self.assertFalse(valid["flow"][:, :5].any())
                    self.assertFalse(valid["flow"][:, 9:].any())
                    for b in basins:
                        bi = prep.basin_index[b]
                        expected = np.isfinite(raw[bi, 5:9]) if b in held else np.zeros(4, dtype=bool)
                        np.testing.assert_array_equal(valid["flow"][bi, 5:9], expected)
                    # 多架构、多模型种子共用该掩膜种子的同一名单。
                    self.assertEqual(held, random_basin_holdout(prep, ratio, seed)[2])


if __name__ == "__main__":
    unittest.main()
