"""官方99对照名单与指标复现守卫测试，无训练及网络请求。"""
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from src.evaluation.official_rating_check import held_ids, verify_reproduction, apply_official_rating, FT_PER_M, CMS_PER_CFS


class Official99Test(unittest.TestCase):
    def test_exact_holdout_and_reject_old_random(self):
        record={'scenario':'random_q_holdout70','mask':{'meta':{'protocol':'train_valid_q_hidden_v2'},
                 'per_task':{'flow':{'held_out_basins':['b']}}}}
        self.assertEqual(held_ids(record,['a','b']),['b'])
        record['mask']['meta']['protocol']='old'
        with self.assertRaises(ValueError): held_ids(record,['a','b'])
        self.assertEqual(held_ids({'scenario':'complete','mask':None},['a','b']),['a','b'])

    def test_reproduction_guards(self):
        old=pd.DataFrame({'basin':['a'],'task':['flow'],'n':[100],'nse':[.5]})
        actual=old.copy()
        with patch('evaluation.metrics.evaluate_run',return_value=(actual,{})), patch('pandas.read_csv',return_value=old):
            verify_reproduction(None,None,None,'ignored')
        actual.loc[0,'nse']=.4
        with patch('evaluation.metrics.evaluate_run',return_value=(actual,{})), patch('pandas.read_csv',return_value=old):
            with self.assertRaises(ValueError): verify_reproduction(None,None,None,'ignored')
        actual.loc[0,'n']=99
        with patch('evaluation.metrics.evaluate_run',return_value=(actual,{})), patch('pandas.read_csv',return_value=old):
            with self.assertRaises(ValueError): verify_reproduction(None,None,None,'ignored')

    def test_official_units_and_no_zero_fill(self):
        rating={'indep':np.array([1.,2.,4.]),'dep':np.array([10.,40.,160.]),'offsets':{},'breakpoints':{}}
        h=np.array([1.,2.,0.])/FT_PER_M
        out=apply_official_rating(h,rating)
        np.testing.assert_allclose(out[:2],np.array([10.,40.])*CMS_PER_CFS)
        self.assertTrue(np.isnan(out[2]))


if __name__=='__main__': unittest.main()
