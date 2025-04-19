# -*- coding: utf-8 -*-
import unittest
from unittest.mock import patch, MagicMock
import pandas as pd
import numpy as np
from dym_estimate.dfm_core import evaluate_dfm_params

# 创建一个模拟 DFM 结果的对象
class MockDFMResults:
    def __init__(self, k_factors, n_vars, n_obs, success=True):
        if success:
            # 模拟成功的 Lambda (n_vars x k_factors)
            self.Lambda = np.random.randn(n_vars, k_factors)
            # 模拟成功的 x_sm (n_obs x k_factors DataFrame with DatetimeIndex)
            dates = pd.date_range(end='2024-06-28', periods=n_obs, freq='W-FRI')
            self.x_sm = pd.DataFrame(np.random.randn(n_obs, k_factors), index=dates, columns=[f'Factor{i+1}' for i in range(k_factors)])
        else:
            # 模拟失败情况 (例如返回 None 或空对象)
            self.Lambda = None
            self.x_sm = None

# 创建一个模拟 DFM_EMalgo 类
class MockDFM_EMalgo:
    def __init__(self, observation, n_factors, n_shocks, n_iter, error='False'):
        self.observation = observation
        self.n_factors = n_factors
        self.n_shocks = n_shocks
        self.n_iter = n_iter
        self.error_flag = error # 模拟错误标志

        # 根据输入模拟成功或失败
        if isinstance(observation, pd.DataFrame) and not observation.empty:
             n_vars = observation.shape[1]
             n_obs = observation.shape[0]
             # 模拟 SVD 错误 (如果 error 标志被设置或满足特定条件)
             if self.error_flag == 'True' or "svd_error_trigger" in observation.columns:
                  # 引发类似 SVD 不收敛的错误
                  raise np.linalg.LinAlgError("Mock SVD did not converge")
             # 模拟成功结果
             self.results = MockDFMResults(n_factors, n_vars, n_obs, success=True)
        else:
             # 模拟一般失败结果
             self.results = MockDFMResults(n_factors, 0, 0, success=False)

    # DFM_EMalgo 调用后通常会访问其属性, 例如 Lambda 和 x_sm
    def __getattr__(self, name):
         # 将属性访问代理到内部的模拟结果对象
         if hasattr(self, 'results') and hasattr(self.results, name):
              return getattr(self.results, name)
         raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")

class TestDfmCore(unittest.TestCase):

    def setUp(self): # 设置一些通用的测试数据
        self.n_obs = 150
        self.dates = pd.date_range(start='2021-01-01', periods=self.n_obs, freq='W-FRI')
        self.variables = ['target', 'var1', 'var2', 'var3']
        self.target_variable = 'target'
        self.full_data = pd.DataFrame(
            np.random.randn(self.n_obs, len(self.variables)),
            index=self.dates,
            columns=self.variables
        )
        # 确保目标变量有合理的值
        self.full_data['target'] = np.sin(np.linspace(0, 10, self.n_obs)) * 5 + np.random.randn(self.n_obs) * 0.5

        self.params = {'k_factors': 2}
        self.var_type_map = {v: 'level' for v in self.variables} # 简化，实际不用于此函数
        self.validation_start = '2023-07-07'
        self.validation_end = '2023-12-29'
        self.train_end_date = '2023-06-30'
        self.target_mean_original = self.full_data[self.target_variable].mean()
        self.target_std_original = self.full_data[self.target_variable].std()
        self.max_iter = 10

    def test_placeholder(self):
        """一个简单的占位测试"""
        print("--- Running placeholder test ---") # 添加打印以确认执行
        self.assertTrue(True)

    # 使用 patch 来替换真实的 DFM_EMalgo 和 apply_stationarity_transforms
    @patch('dym_estimate.dfm_core.DFM_EMalgo', new=MockDFM_EMalgo)
    @patch('dym_estimate.dfm_core.apply_stationarity_transforms')
    def test_successful_evaluation(self, mock_apply_transforms):
        """测试 DFM 评估成功运行并返回指标和载荷"""
        # 配置 mock_apply_transforms 的返回值
        # 让它直接返回输入数据和空的转换日志 (简化测试)
        mock_apply_transforms.side_effect = lambda data, target: (data.copy(), {col: 'level' for col in data.columns})

        results = evaluate_dfm_params(
            variables=self.variables,
            full_data=self.full_data,
            target_variable=self.target_variable,
            params=self.params,
            var_type_map=self.var_type_map,
            validation_start=self.validation_start,
            validation_end=self.validation_end,
            target_freq='W-FRI',
            train_end_date=self.train_end_date,
            target_mean_original=self.target_mean_original,
            target_std_original=self.target_std_original,
            max_iter=self.max_iter
        )

        # 检查返回元组的长度 (8 项)
        self.assertEqual(len(results), 8)

        # 解包结果
        is_rmse, oos_rmse, is_mae, oos_mae, is_hit_rate, oos_hit_rate, is_svd_error, lambda_df = results

        # 检查指标是否为有限数值 (因为模拟了成功运行)
        self.assertTrue(np.isfinite(is_rmse))
        self.assertTrue(np.isfinite(oos_rmse))
        self.assertTrue(np.isfinite(is_mae))
        self.assertTrue(np.isfinite(oos_mae))
        self.assertTrue(np.isfinite(is_hit_rate))
        self.assertTrue(np.isfinite(oos_hit_rate))

        # 检查 SVD 错误标志
        self.assertFalse(is_svd_error)

        # 检查返回的 lambda_df 是否为 DataFrame
        self.assertIsInstance(lambda_df, pd.DataFrame)
        # 检查 lambda_df 的维度 (转换后变量数 x k_factors)
        # 注意：这里假设 apply_transforms 没有移除变量
        expected_vars = len(self.variables)
        self.assertEqual(lambda_df.shape, (expected_vars, self.params['k_factors']))
        # 检查索引是否匹配
        self.assertListEqual(lambda_df.index.tolist(), self.variables)

    @patch('dym_estimate.dfm_core.DFM_EMalgo', new=MockDFM_EMalgo)
    @patch('dym_estimate.dfm_core.apply_stationarity_transforms')
    def test_svd_error_handling(self, mock_apply_transforms):
        """测试 SVD 错误被正确捕获和标记"""
        mock_apply_transforms.side_effect = lambda data, target: (data.copy(), {col: 'level' for col in data.columns})

        # 修改输入数据以触发模拟的 SVD 错误
        error_data = self.full_data.copy()
        error_data['svd_error_trigger'] = 1 # 添加触发列
        error_vars = self.variables + ['svd_error_trigger']

        results = evaluate_dfm_params(
            variables=error_vars,
            full_data=error_data,
            target_variable=self.target_variable,
            params=self.params,
            var_type_map=self.var_type_map, # 传递完整的 map
            validation_start=self.validation_start,
            validation_end=self.validation_end,
            target_freq='W-FRI',
            train_end_date=self.train_end_date,
            target_mean_original=self.target_mean_original,
            target_std_original=self.target_std_original,
            max_iter=self.max_iter
        )

        self.assertEqual(len(results), 8)
        is_rmse, oos_rmse, is_mae, oos_mae, is_hit_rate, oos_hit_rate, is_svd_error, lambda_df = results

        # 检查指标是否为无穷大
        self.assertEqual(is_rmse, np.inf)
        self.assertEqual(oos_rmse, np.inf)
        self.assertEqual(is_mae, np.inf)
        self.assertEqual(oos_mae, np.inf)
        self.assertEqual(is_hit_rate, -np.inf)
        self.assertEqual(oos_hit_rate, -np.inf)

        # 检查 SVD 错误标志是否为 True
        self.assertTrue(is_svd_error)
        # 检查 lambda_df 是否为 None
        self.assertIsNone(lambda_df)

    @patch('dym_estimate.dfm_core.DFM_EMalgo', new=MockDFM_EMalgo)
    @patch('dym_estimate.dfm_core.apply_stationarity_transforms')
    def test_missing_target_in_vars(self, mock_apply_transforms):
        """测试当目标变量不在变量列表中时的失败处理"""
        mock_apply_transforms.side_effect = lambda data, target: (data.copy(), {col: 'level' for col in data.columns})

        results = evaluate_dfm_params(
            variables=['var1', 'var2', 'var3'], # 故意移除 target
            full_data=self.full_data,
            target_variable=self.target_variable,
            params=self.params,
            var_type_map=self.var_type_map,
            validation_start=self.validation_start,
            validation_end=self.validation_end,
            target_freq='W-FRI',
            train_end_date=self.train_end_date,
            target_mean_original=self.target_mean_original,
            target_std_original=self.target_std_original,
            max_iter=self.max_iter
        )
        self.assertEqual(len(results), 8)
        is_rmse, _, _, _, _, _, is_svd_error, lambda_df = results
        self.assertEqual(is_rmse, np.inf)
        self.assertFalse(is_svd_error)
        self.assertIsNone(lambda_df)

if __name__ == '__main__':
    unittest.main() 