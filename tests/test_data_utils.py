# -*- coding: utf-8 -*-
import unittest
import pandas as pd
import numpy as np
from dym_estimate.data_utils import apply_stationarity_transforms

class TestDataUtils(unittest.TestCase):

    def test_apply_stationarity_transforms(self):
        """测试 apply_stationarity_transforms 函数"""
        dates = pd.date_range(start='2023-01-01', periods=100, freq='W-FRI')
        data = pd.DataFrame(
            {
                'target': np.random.randn(100), # 目标变量，应保持 level
                'stationary_var': np.random.randn(100), # 平稳变量，应保持 level
                'non_stationary_var': np.cumsum(np.random.randn(100)), # 非平稳变量，应变为 diff
                'constant_var': np.ones(100) * 5, # 常量，应保持 level
                'nan_var': np.full(100, np.nan), # 全 NaN，应保持全 NaN
                'mixed_nan_non_stationary': np.concatenate(([np.nan]*10, np.arange(90).astype(float) + np.random.randn(90) * 0.1)), # 新：带噪声的非平稳
                'mixed_nan_stationary': np.concatenate(([np.nan]*5, np.random.randn(95))), # 混合 NaN 的平稳，应保持 level
            },
            index=dates
        )
        target_variable = 'target'

        transformed_data, transform_log = apply_stationarity_transforms(data, target_variable, adf_p_threshold=0.05)

        # 检查返回类型
        self.assertIsInstance(transformed_data, pd.DataFrame)
        self.assertIsInstance(transform_log, dict)

        # 检查转换日志
        expected_log = {
            'stationary_var': 'level',
            'non_stationary_var': 'diff',
            'constant_var': 'level_constant',
            'nan_var': 'skipped_empty',
            'mixed_nan_non_stationary': 'diff',
            'mixed_nan_stationary': 'level', # 即使有 NaN，ADF 也会在 dropna() 后运行
            'target': 'target_level'
        }
        self.assertDictEqual(transform_log, expected_log)

        # 检查目标变量是否未变 (除了可能的类型转换)
        pd.testing.assert_series_equal(transformed_data['target'], data['target'], check_dtype=False)

        # 检查平稳变量是否未变
        pd.testing.assert_series_equal(transformed_data['stationary_var'], data['stationary_var'], check_dtype=False)

        # 检查非平稳变量是否被差分 (第一个值应为 NaN)
        self.assertTrue(pd.isna(transformed_data['non_stationary_var'].iloc[0]))
        # 差分后的非 NaN 值数量应该比原始少 1
        self.assertEqual(transformed_data['non_stationary_var'].notna().sum(), data['non_stationary_var'].notna().sum() - 1)

        # 检查常量变量是否未变
        pd.testing.assert_series_equal(transformed_data['constant_var'], data['constant_var'], check_dtype=False)

        # 检查全 NaN 变量是否仍然是全 NaN
        self.assertTrue(transformed_data['nan_var'].isna().all())

        # 检查混合 NaN 的非平稳变量
        # 转换后，第一个非 NaN 值的位置应该与原始数据不同（因为差分移除了第一个值）
        first_valid_index_orig = data['mixed_nan_non_stationary'].first_valid_index()
        first_valid_index_transformed = transformed_data['mixed_nan_non_stationary'].first_valid_index()
        self.assertNotEqual(first_valid_index_orig, first_valid_index_transformed)
        self.assertTrue(pd.isna(transformed_data['mixed_nan_non_stationary'].loc[first_valid_index_orig])) # 原始第一个有效值处现在是 NaN

        # 检查混合 NaN 的平稳变量是否未变 (除了 NaN 位置)
        pd.testing.assert_series_equal(transformed_data['mixed_nan_stationary'], data['mixed_nan_stationary'], check_dtype=False)


if __name__ == '__main__':
    unittest.main() 