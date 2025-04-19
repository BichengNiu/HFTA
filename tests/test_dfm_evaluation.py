# -*- coding: utf-8 -*-
import unittest
from unittest.mock import patch, MagicMock, call
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from dym_estimate.dfm_evaluation import evaluate_k_factors_rolling

class TestDfmEvaluation(unittest.TestCase):

    def setUp(self):
        # 创建足够长的日期索引
        self.start_dt = datetime(2020, 1, 3) # 周五
        self.n_total_weeks = 3 * 52 # 3年数据
        self.dates = pd.date_range(start=self.start_dt, periods=self.n_total_weeks, freq='W-FRI')

        # 创建模拟数据
        self.variables = ['target', 'var1', 'var2']
        self.target_variable = 'target'
        self.full_data = pd.DataFrame(
            np.random.randn(self.n_total_weeks, len(self.variables)),
            index=self.dates,
            columns=self.variables
        )

        self.k_factors = 2
        self.var_type_map = {v: 'level' for v in self.variables}
        self.target_freq = 'W-FRI'
        self.initial_train_weeks = 52 * 2 # 2年
        self.validation_weeks = 4
        self.step_weeks = 4
        self.target_mean_original = 0.5
        self.target_std_original = 1.5
        self.max_iter = 10

    @patch('dym_estimate.dfm_evaluation.evaluate_dfm_params')
    def test_rolling_evaluation_success_no_loadings(self, mock_evaluate_dfm):
        """测试滚动评估成功运行（不收集载荷）"""
        # 模拟 evaluate_dfm_params 的返回值 (RMSE, MAE, HitRate, SVD_err, Lambda)
        # 让它每次调用都返回稍微不同的 OOS 指标和 None 载荷
        mock_evaluate_dfm.side_effect = [
            (1.0, 0.1, 1.5, 0.15, 50.0, 60.0, False, None), # 窗口 1 OOS
            (1.1, 0.2, 1.6, 0.25, 45.0, 70.0, False, None), # 窗口 2 OOS
            (1.2, 0.3, 1.7, 0.35, 40.0, 80.0, False, None), # 窗口 3 OOS
            # ... 可以根据需要添加更多窗口的模拟返回值
        ] * 13 # 假设大约有 13 个窗口 (1年 / 4周步长)

        avg_rmse, avg_mae, avg_hit_rate, loadings = evaluate_k_factors_rolling(
            variables=self.variables,
            k_factors=self.k_factors,
            full_data=self.full_data,
            target_variable=self.target_variable,
            var_type_map=self.var_type_map,
            target_freq=self.target_freq,
            initial_train_weeks=self.initial_train_weeks,
            validation_weeks=self.validation_weeks,
            step_weeks=self.step_weeks,
            target_mean_original=self.target_mean_original,
            target_std_original=self.target_std_original,
            max_iter=self.max_iter,
            collect_loadings=False # 不收集载荷
        )

        # 检查返回值类型
        self.assertIsInstance(avg_rmse, float)
        self.assertIsInstance(avg_mae, float)
        self.assertIsInstance(avg_hit_rate, float)
        self.assertIsNone(loadings) # 确认载荷为 None

        # 检查 mock 是否被正确调用
        self.assertGreater(mock_evaluate_dfm.call_count, 0)
        # 检查第一次调用的参数是否符合预期
        first_call_args = mock_evaluate_dfm.call_args_list[0]
        expected_val_start_0 = (self.start_dt + timedelta(weeks=self.initial_train_weeks)).strftime('%Y-%m-%d')
        expected_val_end_0 = (self.start_dt + timedelta(weeks=self.initial_train_weeks + self.validation_weeks - 1)).strftime('%Y-%m-%d')
        self.assertEqual(first_call_args.kwargs['validation_start'], expected_val_start_0)
        self.assertEqual(first_call_args.kwargs['validation_end'], expected_val_end_0)

        # 检查计算的平均指标是否合理 (基于模拟的返回值)
        # (0.1 + 0.2 + 0.3) / 3 = 0.2 (假设只有3个有效窗口被模拟到)
        # (0.15 + 0.25 + 0.35) / 3 = 0.25
        # (60 + 70 + 80) / 3 = 70
        # 注意：实际调用次数取决于窗口计算逻辑，这里只是示意性检查
        num_calls = mock_evaluate_dfm.call_count
        if num_calls > 0:
             expected_avg_rmse = np.mean([0.1, 0.2, 0.3][:num_calls])
             expected_avg_mae = np.mean([0.15, 0.25, 0.35][:num_calls])
             expected_avg_hit = np.mean([60.0, 70.0, 80.0][:num_calls])
             self.assertAlmostEqual(avg_rmse, expected_avg_rmse)
             self.assertAlmostEqual(avg_mae, expected_avg_mae)
             self.assertAlmostEqual(avg_hit_rate, expected_avg_hit)

    @patch('dym_estimate.dfm_evaluation.evaluate_dfm_params')
    def test_rolling_evaluation_success_with_loadings(self, mock_evaluate_dfm):
        """测试滚动评估成功运行（收集载荷）"""
        # 模拟载荷 DataFrame
        mock_lambda_1 = pd.DataFrame(np.random.rand(len(self.variables), self.k_factors), index=self.variables)
        mock_lambda_2 = pd.DataFrame(np.random.rand(len(self.variables), self.k_factors), index=self.variables)
        # 模拟 evaluate_dfm_params 返回包含载荷的元组
        mock_evaluate_dfm.side_effect = [
            (1.0, 0.1, 1.5, 0.15, 50.0, 60.0, False, mock_lambda_1),
            (1.1, 0.2, 1.6, 0.25, 45.0, 70.0, False, mock_lambda_2),
        ] * 13

        avg_rmse, avg_mae, avg_hit_rate, loadings = evaluate_k_factors_rolling(
            variables=self.variables,
            k_factors=self.k_factors,
            full_data=self.full_data,
            target_variable=self.target_variable,
            var_type_map=self.var_type_map,
            target_freq=self.target_freq,
            initial_train_weeks=self.initial_train_weeks,
            validation_weeks=self.validation_weeks,
            step_weeks=self.step_weeks,
            target_mean_original=self.target_mean_original,
            target_std_original=self.target_std_original,
            max_iter=self.max_iter,
            collect_loadings=True # 收集载荷
        )
        
        self.assertGreater(mock_evaluate_dfm.call_count, 0)
        num_calls = mock_evaluate_dfm.call_count
        
        self.assertIsInstance(loadings, list)
        self.assertEqual(len(loadings), num_calls) # 载荷列表长度应等于调用次数
        if num_calls > 0:
            self.assertIsInstance(loadings[0], pd.DataFrame)
            pd.testing.assert_frame_equal(loadings[0], mock_lambda_1)
        if num_calls > 1:
             pd.testing.assert_frame_equal(loadings[1], mock_lambda_2)

    @patch('dym_estimate.dfm_evaluation.evaluate_dfm_params')
    def test_rolling_evaluation_insufficient_data(self, mock_evaluate_dfm):
        """测试数据不足无法进行滚动验证的情况"""
        short_data = self.full_data.iloc[:self.initial_train_weeks + self.validation_weeks - 5] # 故意缩短数据

        avg_rmse, avg_mae, avg_hit_rate, loadings = evaluate_k_factors_rolling(
            variables=self.variables,
            k_factors=self.k_factors,
            full_data=short_data,
            target_variable=self.target_variable,
            var_type_map=self.var_type_map,
            target_freq=self.target_freq,
            initial_train_weeks=self.initial_train_weeks,
            validation_weeks=self.validation_weeks,
            step_weeks=self.step_weeks,
            target_mean_original=self.target_mean_original,
            target_std_original=self.target_std_original,
            max_iter=self.max_iter,
            collect_loadings=False
        )

        # 检查是否返回了表示失败的无穷大值
        self.assertEqual(avg_rmse, np.inf)
        self.assertEqual(avg_mae, np.inf)
        self.assertEqual(avg_hit_rate, -np.inf)
        self.assertIsNone(loadings)
        # 确认 evaluate_dfm_params 没有被调用
        mock_evaluate_dfm.assert_not_called()

if __name__ == '__main__':
    unittest.main() 