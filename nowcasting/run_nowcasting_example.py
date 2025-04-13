# -*- coding: utf-8 -*-
"""
run_nowcasting_example.py

演示如何使用 DFM_Nowcasting.py 中的 DFMNowcastModel 类执行即时预测更新流程。
"""

import pandas as pd
import numpy as np
import pickle # 用于加载/保存模拟结果 (如果需要)

# --- 导入必要的类 ---
try:
    # 假设这些都在同一目录或 Python 路径中
    from DFM_Nowcasting import DFMNowcastModel
    # 需要 DFMEMResultsWrapper 来创建基线对象
    from DynamicFactorModel import DFMEMResultsWrapper
    # 如果需要创建模拟的 SKF/KF 结果
    from DiscreteKalmanFilter import SKFResultsWrapper, KalmanFilterResultsWrapper
except ImportError as e:
    print(f"错误：无法导入必要的模块: {e}")
    print("请确保 DynamicFactorModel.py, DiscreteKalmanFilter.py, DFM_Nowcasting.py, 和 Functions.py 在 Python 路径中。")
    import sys
    sys.exit(1)

# --- 模拟基线模型结果 ---

def simulate_baseline_model():
    """创建模拟的基线模型结果"""
    print("模拟基线模型结果...")
    n_factors = 2
    n_obs = 5
    n_shocks = n_factors # 假设冲击数等于因子数
    n_time_base = 100 # 基线数据的长度

    # 模拟参数 (简化)
    A = np.array([[0.8, 0.1], [0.05, 0.7]])
    B = np.eye(n_factors) * 0.1 # 简化
    Q = np.diag([0.1, 0.1])
    R = np.diag(np.random.uniform(0.1, 0.5, n_obs)) # 对角观测噪声
    Lambda = np.random.randn(n_obs, n_factors) * 0.5

    # 模拟状态和观测
    state_names = [f'Factor_{i+1}' for i in range(n_factors)]
    obs_names = [f'Obs_{i+1}' for i in range(n_obs)]
    base_index = pd.date_range(start='2022-01-01', periods=n_time_base, freq='MS')

    # 模拟平滑状态 x_sm (简单的 AR(1) 过程)
    x_sm_arr = np.zeros((n_time_base, n_factors))
    x_sm_arr[0, :] = np.random.randn(n_factors) * 0.5
    for t in range(1, n_time_base):
        x_sm_arr[t, :] = A @ x_sm_arr[t-1, :] + np.random.multivariate_normal(np.zeros(n_factors), Q)
    x_sm_df = pd.DataFrame(x_sm_arr, index=base_index, columns=state_names)

    # 模拟滤波状态 x (可以简单设为与 x_sm 类似，或者添加噪声)
    x_df = x_sm_df + np.random.randn(n_time_base, n_factors) * 0.05

    # 模拟中心化观测 z = Lambda @ x_sm + noise
    z_arr = x_sm_arr @ Lambda.T + np.random.multivariate_normal(np.zeros(n_obs), R, n_time_base)
    # 引入一些随机的 NaN
    nan_mask = np.random.rand(n_time_base, n_obs) < 0.1 # 10% 的概率为 NaN
    z_arr[nan_mask] = np.nan
    z_df = pd.DataFrame(z_arr, index=base_index, columns=obs_names)

    # 模拟观测均值 obs_mean
    obs_mean = pd.Series(np.random.randn(n_obs) * 10, index=obs_names)

    # 创建模拟结果对象 - 使用实际的 DFMEMResultsWrapper 类
    baseline_results = DFMEMResultsWrapper(
        A=A, B=B, Q=Q, R=R, Lambda=Lambda, x=x_df, x_sm=x_sm_df, z=z_df
    )

    # 模拟基线 SKF/KF 结果 (可选，但 DFMNowcastModel 会用到)
    # 这里我们直接使用模拟的数据填充
    # --- 重新定义 P_sm_list 用于模拟 KF/SKF 对象 ---
    P_sm_list = [Q.copy() + np.eye(Q.shape[0])*1e-6 for _ in range(n_time_base)] # 模拟平滑协方差 (加一点噪声)

    mock_kf_results = KalmanFilterResultsWrapper(
        x_minus=x_df, # 简化：用 x 代替 x_minus
        x=x_df,
        z=z_df,
        Kalman_gain=[np.random.rand(n_factors, n_obs) for _ in range(n_time_base)], # 模拟增益
        P=P_sm_list, # 简化：用 P_sm 代替 P
        P_minus=P_sm_list, # 简化：用 P_sm 代替 P_minus
        state_names=state_names,
        A=A
    )
    mock_smooth_results = SKFResultsWrapper(x_sm=x_sm_df, P_sm=P_sm_list, z=z_df)

    print("模拟基线模型结果完成。")
    return baseline_results, obs_mean, state_names, n_shocks, mock_kf_results, mock_smooth_results


# --- 模拟不同 Vintages 的数据 ---

def simulate_data_vintages(base_data: pd.DataFrame, obs_mean: pd.Series, num_vintages: int = 3):
    """基于基线数据创建模拟的数据 vintages"""
    print(f"模拟 {num_vintages} 个数据 vintages...")
    vintages = {}
    current_data = base_data.copy() # 从基线数据开始

    # 将基线数据（中心化后）转换回原始水平
    current_data_orig_level = current_data + obs_mean

    for i in range(num_vintages):
        vintage_date = current_data.index[-1] + pd.Timedelta(days=30 * (i + 1)) # 假设每月更新
        vintage_name = vintage_date.strftime('%Y-%m')
        print(f"  创建 vintage: {vintage_name}")

        # 模拟数据更新：增加一行新数据
        new_index = pd.date_range(start=current_data_orig_level.index[-1] + pd.Timedelta(days=1), periods=1, freq='MS')
        # 使用全局变量或传递的 baseline_results
        last_factors = baseline_results_for_sim.x_sm.iloc[-1].values
        next_factors = baseline_results_for_sim.A @ last_factors + np.random.multivariate_normal(np.zeros(baseline_results_for_sim.A.shape[0]), baseline_results_for_sim.Q)
        new_obs_centered = baseline_results_for_sim.Lambda @ next_factors + np.random.multivariate_normal(np.zeros(len(obs_mean)), baseline_results_for_sim.R)
        new_obs_orig_level = new_obs_centered + obs_mean.values
        new_row = pd.DataFrame([new_obs_orig_level], index=new_index, columns=obs_mean.index)

        updated_data_orig_level = pd.concat([current_data_orig_level, new_row])

        # 模拟数据修正：随机修改过去某个值
        if len(updated_data_orig_level) > 10: # 确保有足够历史数据
            random_row_idx = np.random.randint(0, len(updated_data_orig_level) - 2) # 不修改最新一行
            random_col_idx = np.random.randint(0, len(obs_mean))
            change_factor = 1 + np.random.randn() * 0.1 # +/- 10% 左右的变动
            updated_data_orig_level.iloc[random_row_idx, random_col_idx] *= change_factor

        # 引入新的随机 NaN
        nan_mask_new = np.random.rand(*updated_data_orig_level.shape) < 0.05 # 5% 概率引入新 NaN
        updated_data_orig_level[nan_mask_new] = np.nan

        vintages[vintage_name] = updated_data_orig_level
        current_data_orig_level = updated_data_orig_level # 更新为下一个 vintage 的基础

    print("模拟数据 vintages 完成。")
    return vintages

# --- 主程序 ---
if __name__ == "__main__":
    print("--- 开始 Nowcasting 示例 ---")

    # 1. 加载/模拟基线模型
    baseline_results, obs_mean, state_names, n_shocks, base_kf, base_smooth = simulate_baseline_model()

    # 2. 创建基线 DFMNowcastModel 实例
    baseline_vintage_name = baseline_results.x_sm.index[-1].strftime('%Y-%m') # 基线日期
    nowcast_models = {}
    nowcast_models[baseline_vintage_name] = DFMNowcastModel(
        baseline_results=baseline_results,
        obs_mean=obs_mean,
        state_names=state_names,
        n_shocks=n_shocks,
        baseline_kf_results=base_kf,       # 传入基线 KF 结果
        baseline_smooth_results=base_smooth # 传入基线平滑结果
    )
    print(f"基线模型 ({baseline_vintage_name}) 已创建。")

    # 3. 模拟后续数据 Vintages
    # 使用基线的 z (中心化数据) 作为模拟基础
    global baseline_results_for_sim # 使用全局变量或传递参数
    baseline_results_for_sim = baseline_results
    data_vintages = simulate_data_vintages(baseline_results.z, obs_mean, num_vintages=3)

    # 4. 迭代处理 Vintages
    forecast_results = {}
    news_results = {}
    previous_vintage_name = baseline_vintage_name

    for vintage_name, vintage_data in sorted(data_vintages.items()):
        print(f"\n--- 处理 Vintage: {vintage_name} ---")
        previous_model = nowcast_models[previous_vintage_name]

        # 4.1 应用模型到新数据
        # apply 方法内部会调用 smooth
        current_model = previous_model.apply(vintage_data)
        nowcast_models[vintage_name] = current_model
        print(f"模型已更新至 Vintage: {vintage_name}")

        # 4.2 生成预测 (例如，预测未来 3 步)
        forecast_steps = 3
        try:
            factor_forecasts, _ = current_model.forecast(steps=forecast_steps)
            forecast_results[vintage_name] = factor_forecasts
            print(f"因子预测 ({forecast_steps} 步):")
            print(factor_forecasts)

            # 也可以将因子预测转换为对某个观测变量的预测
            target_var_index = 0 # 假设预测第一个观测变量
            lambda_target = current_model.Lambda[target_var_index, :]
            obs_forecast_centered = factor_forecasts.values @ lambda_target
            obs_forecast = obs_forecast_centered + current_model.obs_mean.iloc[target_var_index]
            obs_forecast_series = pd.Series(obs_forecast, index=factor_forecasts.index, name=f'{current_model.obs_names[target_var_index]}_Forecast')
            print(f"\n观测变量 '{current_model.obs_names[target_var_index]}' 预测 ({forecast_steps} 步):")
            print(obs_forecast_series)

        except Exception as e:
            print(f"预测时出错: {e}")

        # 4.3 计算新闻影响 (使用占位符实现)
        impact_date = vintage_data.index[-1] + pd.DateOffset(months=1) # 假设关心下个月
        impacted_variable = obs_mean.index[0] # 假设关心第一个观测变量
        print(f"\n计算新闻影响 (目标: {impact_date.strftime('%Y-%m-%d')}, 变量: {impacted_variable})...")
        try:
            news_df = current_model.news(previous_model, impact_date, impacted_variable)
            news_results[vintage_name] = news_df
            if not news_df.empty:
                print("新闻影响 (前 5 大):")
                print(news_df.head())
            else:
                print("未计算出新闻影响。")
        except Exception as e:
            print(f"计算新闻时出错: {e}")

        # 更新 previous_vintage_name
        previous_vintage_name = vintage_name

    print("\n--- Nowcasting 示例完成 ---")

    # 可以进一步分析存储在 forecast_results 和 news_results 中的结果
    # 例如，绘制预测的演变图或新闻影响的累积图 