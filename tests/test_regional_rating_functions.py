"""率定函数训练隔离、单调性与幂律恢复回归测试。"""
import unittest
from types import SimpleNamespace

import numpy as np
import pandas as pd

from src.evaluation.regional_rating_functions import training_pairs, fit_functions, apply_function


class RegionalRatingTest(unittest.TestCase):
    def test_original_onehot_and_constant_rules(self):
        from src.evaluation.train_nonrating14_external import ExternalPrepared
        grid = pd.date_range('2000-01-01', periods=30, freq='3h')
        basins = ['a', 'b']
        raw = np.tile(np.arange(30, dtype=np.float32), (2, 1))
        splits = {b: {f'{s}_{edge}': str(grid[pos])
                      for s, lo, hi in (('train', 0, 9), ('valid', 10, 19), ('test', 20, 29))
                      for edge, pos in (('start', lo), ('end', hi))} for b in basins}
        attrs = pd.DataFrame({'category__1': [0., 1.], 'constant': [4., 4.],
                              'area': [1., 3.]}, index=basins)
        prep = ExternalPrepared(grid, raw[:, :, None], attrs,
                                {'flow': raw.copy(), 'waterlevel': raw.copy()}, basins, splits)
        np.testing.assert_array_equal(prep.attrs[:, 0], [0., 1.])
        np.testing.assert_array_equal(prep.attrs[:, 1], [0., 0.])
        np.testing.assert_array_equal(prep.attrs[:, 2], [-1., 1.])
        self.assertEqual(prep.stats['attr']['constant']['std'], 1.)
        self.assertEqual(prep.stats['forcing']['0']['mean'], 4.5)

    def test_known_power_and_monotonic_piecewise(self):
        h = np.linspace(-2, 3, 500)
        q = 2.4 * (h + 3) ** 1.7
        models = fit_functions(h, q)
        np.testing.assert_allclose(apply_function(models['power'], h), q, rtol=1e-4)
        prediction = apply_function(models['piecewise'], np.linspace(-3, 4, 1000))
        self.assertTrue(np.all(np.diff(prediction) >= -1e-10))
        self.assertTrue(np.isfinite(apply_function(models['piecewise'], 0.)))

    def test_only_retained_training_pairs(self):
        h = np.tile(np.linspace(-2, 3, 100), (2, 1))
        q = 2.4 * (h + 3) ** 1.7
        prep = SimpleNamespace(basins=['a', 'held'], basin_index={'a': 0, 'held': 1},
                               targets_raw={'flow': q}, split_range=lambda b, s: (0, 49))
        scaling = SimpleNamespace(normalized={'waterlevel': h}, std={'flow': np.ones(2)})
        first = training_pairs(prep, scaling, ['a'])
        before = fit_functions(*first[:2])
        q[1, :] = 1e9
        h[1, :] = -1e9
        q[0, 50:] = -1e9
        h[0, 50:] = 1e9
        after = training_pairs(prep, scaling, ['a'])
        self.assertEqual(first[2], {'a': 50})
        self.assertEqual(before, fit_functions(*after[:2]))

    def test_negative_pairs_are_explicit(self):
        h = np.linspace(-2, 3, 100)
        q = h.copy()
        models = fit_functions(h, q)
        self.assertEqual(models['power']['n_negative_pairs_excluded'], int((q < 0).sum()))


if __name__ == '__main__':
    unittest.main()
