# -*- coding: utf-8 -*-
"""
DFM_Nowcasting.py

包含 DFMNowcastModel 类，用于基于已估计的 DFM 模型进行即时预测更新、
预测和新闻分析。
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Union

# 假设原始文件位于同一目录或 Python 路径中
try:
    from DynamicFactorModel import DFMEMResultsWrapper
    from DiscreteKalmanFilter import KalmanFilter, FIS, KalmanFilterResultsWrapper, SKFResultsWrapper
    from Functions import align_mixed_frequency_data # 如果需要对齐
except ImportError as e:
    print(f"错误：无法导入必要的模块: {e}")
    print("请确保 DynamicFactorModel.py, DiscreteKalmanFilter.py, 和 Functions.py 在 Python 路径中。")
    # 或者根据你的项目结构调整导入路径
    # 例如: from your_project.DynamicFactorModel import DFMEMResultsWrapper
    import sys
    sys.exit(1)

class DFMNowcastModel:
    """
    封装一个已估计的动态因子模型，并提供即时预测、更新和新闻分析功能。
    """
    def __init__(self,
                 baseline_results: DFMEMResultsWrapper,
                 obs_mean: Union[pd.Series, Dict[str, float]],
                 state_names: List[str],
                 n_shocks: int,
                 baseline_kf_results: Optional[KalmanFilterResultsWrapper] = None, # 用于存储基线的KF结果
                 baseline_smooth_results: Optional[SKFResultsWrapper] = None):   # 用于存储基线的平滑结果
        """
        初始化 DFMNowcastModel。

        Args:
            baseline_results: 从 DFM_EMalgo 返回的包含最终估计参数的对象。
            obs_mean: 用于中心化观测数据的均值 (Series 或字典)。
            state_names: 状态（因子）的名称列表。
            n_shocks: 模型中冲击的数量。
            baseline_kf_results: (可选) 运行基线数据得到的 KalmanFilterResultsWrapper。
            baseline_smooth_results: (可选) 运行基线数据得到的 SKFResultsWrapper。
        """
        if not isinstance(baseline_results, DFMEMResultsWrapper):
            raise TypeError("baseline_results 必须是 DFMEMResultsWrapper 的实例。")

        # --- 存储核心参数 ---
        self.A = np.array(baseline_results.A)
        self.B = np.array(baseline_results.B) # 注意：B 的估计可能很简单
        self.Q = np.array(baseline_results.Q)
        self.R = np.array(baseline_results.R)
        self.Lambda = np.array(baseline_results.Lambda) # H in KalmanFilter

        # --- 存储模型维度和名称 ---
        self.n_factors = self.A.shape[0]
        self.n_obs = self.Lambda.shape[0]
        self.n_shocks = n_shocks
        self.state_names = state_names
        self.obs_mean = pd.Series(obs_mean) if isinstance(obs_mean, dict) else obs_mean
        # 尝试从 Lambda 获取观测变量名称顺序（如果可用）
        self.obs_names = self.obs_mean.index.tolist() # 假设 obs_mean 的索引是正确的顺序

        # --- 存储初始条件 (来自基线模型的末尾或开始) ---
        # 优先使用传入的平滑结果，否则从 baseline_results 获取
        smoothed_states_base = baseline_smooth_results.x_sm if baseline_smooth_results else baseline_results.x_sm
        smoothed_cov_base = baseline_smooth_results.P_sm if baseline_smooth_results else getattr(baseline_results, 'P_sm', None) # 检查 P_sm 是否存在

        if smoothed_states_base is None or smoothed_states_base.empty:
             raise ValueError("无法获取基线平滑状态 (x_sm) 以设置初始条件。")

        self.x0 = smoothed_states_base.iloc[0].values.copy() # 初始状态用第一个平滑状态
        if smoothed_cov_base is not None and len(smoothed_cov_base) > 0:
            self.P0 = smoothed_cov_base[0].copy() # 初始协方差用第一个平滑协方差
        else:
            print("警告: 无法从 baseline_results 获取 P_sm。使用单位矩阵初始化 P0。")
            self.P0 = np.eye(self.n_factors)

        # --- 存储完整的基线结果供参考 ---
        self._baseline_em_results = baseline_results
        self.current_kf_results = baseline_kf_results # 如果传入，存储KF结果
        self.current_smooth_results = baseline_smooth_results if baseline_smooth_results else SKFResultsWrapper(x_sm=smoothed_states_base, P_sm=smoothed_cov_base, z=baseline_results.z)

        # --- 确保 B 矩阵形状正确 ---
        if self.B.shape != (self.n_factors, self.n_shocks):
             print(f"警告: 存储的 B 矩阵形状 {self.B.shape} 与预期的 ({self.n_factors}, {self.n_shocks}) 不符。将尝试重塑或使用零矩阵。")
             # 简单的处理：如果形状不匹配，创建一个零矩阵
             self.B = np.zeros((self.n_factors, self.n_shocks))


    def _preprocess_data(self, observation_data: pd.DataFrame) -> pd.DataFrame:
        """
        对输入的观测数据进行预处理（中心化）。

        Args:
            observation_data: 包含新观测数据的 DataFrame。

        Returns:
            中心化后的观测数据 DataFrame。
        """
        if not isinstance(observation_data, pd.DataFrame):
            raise TypeError("observation_data 必须是 Pandas DataFrame。")
        if not isinstance(observation_data.index, pd.DatetimeIndex):
             print("警告: observation_data 的索引不是 DatetimeIndex。")

        # 确保列顺序与 self.obs_mean 一致
        try:
            data_reordered = observation_data[self.obs_names].copy()
        except KeyError as e:
            missing_cols = set(self.obs_names) - set(observation_data.columns)
            extra_cols = set(observation_data.columns) - set(self.obs_names)
            msg = f"输入数据的列与模型期望的列不匹配。\n缺失: {missing_cols}\n多余: {extra_cols}"
            raise ValueError(msg) from e

        # 中心化
        centered_data = data_reordered - self.obs_mean
        return centered_data

    def smooth(self, observation_data: pd.DataFrame) -> tuple[KalmanFilterResultsWrapper, SKFResultsWrapper]:
        """
        使用存储的固定模型参数对新的观测数据运行卡尔曼滤波和平滑。

        Args:
            observation_data: 包含新观测数据的 DataFrame。

        Returns:
            一个元组，包含 KalmanFilterResultsWrapper 和 SKFResultsWrapper 对象，
            对应于在新数据上运行的结果。
        """
        print(f"对新数据运行滤波和平滑 (数据长度: {len(observation_data)})...")
        # 1. 预处理数据
        centered_data = self._preprocess_data(observation_data)

        # 2. 准备滤波器的输入
        Z_new = centered_data
        U_new = np.zeros((len(Z_new), self.n_shocks)) # 假设无外生输入
        error_df_new = pd.DataFrame(data=U_new, columns=[f'shock{i+1}' for i in range(self.n_shocks)], index=Z_new.index)

        # 使用存储的参数和初始条件
        # 注意：这里的 x0, P0 是基线模型的初始值，对于增量更新可能需要调整
        # 更稳健的方法可能是从上一个时间点的结果开始，但这需要更复杂的逻辑
        # 这里我们假设每次都从头开始滤波/平滑整个新数据集
        print("  调用 KalmanFilter...")
        kf_results = KalmanFilter(Z=Z_new, U=error_df_new, A=self.A, B=self.B, H=self.Lambda,
                                  state_names=self.state_names, x0=self.x0, P0=self.P0,
                                  Q=self.Q, R=self.R)

        print("  调用 FIS (平滑器)...")
        smooth_results = FIS(kf_results)
        print("滤波和平滑完成。")

        return kf_results, smooth_results

    def forecast(self, steps: int, last_state: Optional[np.ndarray] = None,
                 last_covariance: Optional[np.ndarray] = None) -> tuple[pd.DataFrame, List[np.ndarray]]:
        """
        从最后一个已知状态向前预测因子状态和协方差。

        Args:
            steps: 要预测的步数。
            last_state: 预测的起始状态 (n_factors,)。如果为 None，则使用最新平滑状态。
            last_covariance: 预测的起始状态协方差 (n_factors, n_factors)。如果为 None，则使用最新平滑协方差。

        Returns:
            一个元组，包含：
            - forecast_states: 包含预测状态的 DataFrame (steps x n_factors)。
            - forecast_covariances: 包含预测协方差矩阵的列表 (长度为 steps)。
        """
        if self.current_smooth_results is None:
            raise ValueError("无法进行预测，因为没有可用的平滑结果。请先运行 smooth 或 apply。")

        if last_state is None:
            current_state = self.current_smooth_results.x_sm.iloc[-1].values
        else:
            current_state = np.array(last_state)
            if current_state.shape != (self.n_factors,):
                raise ValueError(f"last_state 形状必须是 ({self.n_factors},)")

        if last_covariance is None:
            if self.current_smooth_results.P_sm is not None and len(self.current_smooth_results.P_sm) > 0:
                 current_cov = self.current_smooth_results.P_sm[-1]
            else:
                 raise ValueError("无法获取最后的平滑协方差用于预测。")
        else:
            current_cov = np.array(last_covariance)
            if current_cov.shape != (self.n_factors, self.n_factors):
                raise ValueError(f"last_covariance 形状必须是 ({self.n_factors}, {self.n_factors})")

        forecast_states_list = []
        forecast_covariances_list = []

        # 获取最后一个日期用于生成预测索引
        last_date = self.current_smooth_results.x_sm.index[-1]
        # 假设频率可以推断，或者需要用户指定
        freq = pd.infer_freq(self.current_smooth_results.x_sm.index)
        if freq is None:
            print("警告：无法推断原始数据的频率，预测日期可能不准确。")
            # 尝试使用 'D' 作为默认频率
            try:
                 forecast_index = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=steps, freq='D')
            except: # 更通用的异常捕获
                 forecast_index = pd.RangeIndex(start=len(self.current_smooth_results.x_sm), stop=len(self.current_smooth_results.x_sm) + steps)
        else:
             forecast_index = pd.date_range(start=last_date, periods=steps + 1, freq=freq)[1:] # 从下一个日期开始

        print(f"开始因子预测 {steps} 步...")
        for _ in range(steps):
            # 预测下一步状态 (忽略 B*u)
            next_state = self.A @ current_state
            # 预测下一步协方差
            next_cov = self.A @ current_cov @ self.A.T + self.Q

            forecast_states_list.append(next_state)
            forecast_covariances_list.append(next_cov)

            # 更新当前状态和协方差以进行下一步预测
            current_state = next_state
            current_cov = next_cov

        forecast_states_df = pd.DataFrame(forecast_states_list, index=forecast_index[:len(forecast_states_list)], columns=self.state_names)
        print("预测完成。")

        return forecast_states_df, forecast_covariances_list

    def apply(self, new_observation_data: pd.DataFrame) -> 'DFMNowcastModel':
        """
        将模型（固定参数）应用于新的观测数据集。

        这本质上是在新数据上运行 smooth，并返回一个新的 DFMNowcastModel 实例，
        该实例包含更新后的状态，但保留原始的基准参数。

        Args:
            new_observation_data: 新的观测数据 DataFrame。

        Returns:
            一个新的 DFMNowcastModel 实例，代表应用新数据后的模型状态。
        """
        print(f"应用模型到新数据 (数据长度: {len(new_observation_data)})...")
        # 运行滤波和平滑
        kf_results_new, smooth_results_new = self.smooth(new_observation_data)

        # 创建一个新的实例来代表这个 vintage
        # 它共享相同的参数 (A, Lambda, Q, R, B) 和 obs_mean, n_shocks 等
        # 但具有新的 kf_results 和 smooth_results
        new_model_instance = DFMNowcastModel(
            baseline_results=self._baseline_em_results, # 传递原始EM结果
            obs_mean=self.obs_mean,
            state_names=self.state_names,
            n_shocks=self.n_shocks,
            baseline_kf_results=kf_results_new,      # 存储新的KF结果
            baseline_smooth_results=smooth_results_new # 存储新的平滑结果
        )
        print("模型应用完成，返回新的模型实例。")
        return new_model_instance

    def news(self,
             previous_vintage_model: 'DFMNowcastModel',
             impact_date: Union[str, pd.Timestamp],
             impacted_variable: str,
             comparison_type: str = 'previous') -> pd.DataFrame:
        """
        计算新数据 vintage 相对于前一个 vintage 的 "新闻" 及其对特定变量预测的影响。

        Args:
            previous_vintage_model: 代表上一个数据 vintage 的 DFMNowcastModel 实例。
            impact_date: 要计算影响的目标日期。
            impacted_variable: 要计算影响的目标观测变量名称。
            comparison_type: 比较类型 ('previous' 或 'specific_date') - 当前实现主要关注 'previous'。

        Returns:
            一个 DataFrame，包含新闻分解结果，类似于 statsmodels 中的 news() 方法输出。
            列应包括 'update date', 'updated variable', 'observed', 'forecast (prev)',
            'news', 'weight', 'impact'。

        注意: 这是一个复杂的功能，当前实现是一个占位符或简化版本。
              完整的实现需要仔细遵循 Bańbura and Modugno (2014) 等文献。
        """
        print(f"开始计算'新闻'影响 (对比当前 vs 前一 vintage)...")
        print(f"  目标日期: {impact_date}, 目标变量: {impacted_variable}")
        print("警告: news() 方法当前是占位符，未完全实现复杂的新闻分解逻辑。")

        # --- 基本检查 ---
        if not isinstance(previous_vintage_model, DFMNowcastModel):
            raise TypeError("previous_vintage_model 必须是 DFMNowcastModel 的实例。")
        if impacted_variable not in self.obs_names:
             raise ValueError(f"目标变量 '{impacted_variable}' 不在模型观测变量列表中。")
        try:
             impact_date_ts = pd.to_datetime(impact_date)
        except ValueError:
             raise ValueError(f"无法将 impact_date '{impact_date}' 转换为时间戳。")

        # --- 简化逻辑占位符 ---
        # 1. 获取当前和先前 vintage 的数据 (中心化后)
        current_z = self.current_smooth_results.z
        previous_z = previous_vintage_model.current_smooth_results.z

        # 2. 获取先前 vintage 的滤波结果 (预测和卡尔曼增益)
        prev_kf_results = previous_vintage_model.current_kf_results
        if prev_kf_results is None:
             raise ValueError("无法计算新闻，因为前一个 vintage 的卡尔曼滤波结果不可用。")

        # 3. 识别新/修订的数据点
        # 合并索引以识别所有相关日期
        combined_index = current_z.index.union(previous_z.index).sort_values()
        results_list = []
        impacted_var_index = self.obs_names.index(impacted_variable)
        lambda_impacted_row = self.Lambda[impacted_var_index, :]

        # 迭代所有可能的更新日期
        for t_idx, timestamp in enumerate(combined_index):
            if timestamp > impact_date_ts: # 只考虑影响目标日期之前或当天的新闻
                continue

            # 检查当前和先前的数据
            z_t_current = current_z.loc[timestamp] if timestamp in current_z.index else pd.Series(index=self.obs_names, dtype=float) * np.nan
            z_t_previous = previous_z.loc[timestamp] if timestamp in previous_z.index else pd.Series(index=self.obs_names, dtype=float) * np.nan

            # 检查先前 vintage 在 t 时刻的预测（如果存在）
            if prev_kf_results.x_minus is None or timestamp not in prev_kf_results.x_minus.index:
                 continue # 没有前一期的预测，无法计算新闻

            x_minus_t_prev = prev_kf_results.x_minus.loc[timestamp].values
            forecast_z_t_prev = self.Lambda @ x_minus_t_prev # H @ x_minus_t

            # 检查先前 vintage 在 t 时刻的卡尔曼增益（如果存在）
            if prev_kf_results.Kalman_gain is None or t_idx >= len(prev_kf_results.Kalman_gain) or prev_kf_results.Kalman_gain[t_idx] is None:
                 K_t_prev = np.zeros((self.n_factors, self.n_obs)) # 如果没有增益，影响为0
            else:
                 K_t_prev = prev_kf_results.Kalman_gain[t_idx]

            # 迭代每个观测变量
            for j, var_name in enumerate(self.obs_names):
                observed_current = z_t_current.iloc[j]
                observed_previous = z_t_previous.iloc[j]
                forecast_previous = forecast_z_t_prev[j]

                # 定义"新闻"：仅当当前观测有效且与先前预测不同时发生
                is_news = pd.notna(observed_current) and (pd.isna(observed_previous) or observed_current != observed_previous)

                if is_news:
                    news_value = observed_current - forecast_previous
                    # 简化：计算对 t 时刻状态的影响 (需要处理哪些观测可用)
                    # 假设只有一个变量更新，这简化了计算
                    # 注意：真实实现需要处理多个同时更新，并使用完整的卡尔曼增益
                    # 这里我们使用一个非常简化的权重/影响计算作为占位符
                    # 获取 K_t 中与变量 j 对应的列
                    if j < K_t_prev.shape[1]: # 确保索引有效
                         K_t_j = K_t_prev[:, j] # (n_factors, 1)
                    else:
                         K_t_j = np.zeros(self.n_factors)

                    # 对 t 时刻状态的影响
                    impact_on_x_t = K_t_j * news_value

                    # 向前传播影响到 impact_date
                    steps_to_propagate = (impact_date_ts - timestamp).days # 简化为天数
                    # 注意：这假设 A 适用于天数，如果原始频率不同，需要调整
                    # 并且，这忽略了更精确的时间差计算
                    if steps_to_propagate < 0: continue # 更新发生在 impact_date 之后
                    if steps_to_propagate == 0:
                         A_pow_k = np.eye(self.n_factors)
                    else:
                         try:
                              # 如果频率不是天，这里需要调整
                              A_pow_k = np.linalg.matrix_power(self.A, steps_to_propagate)
                         except: # 可能频率非日导致 timedelta 不能直接用
                              print(f"警告：无法计算 A 的 {steps_to_propagate} 次幂，影响传播可能不准确。")
                              A_pow_k = np.eye(self.n_factors) # 简化处理

                    impact_on_x_impact_date = A_pow_k @ impact_on_x_t

                    # 转换为对目标观测变量的影响
                    impact_on_obs = lambda_impacted_row @ impact_on_x_impact_date

                    # 简化的权重（需要更复杂的计算）
                    # 这里的权重只是一个占位符
                    weight = lambda_impacted_row @ K_t_j # 非常粗略的近似

                    results_list.append({
                        'update date': timestamp,
                        'updated variable': var_name,
                        'observed': observed_current + self.obs_mean.get(var_name, 0), # 反中心化
                        'forecast (prev)': forecast_previous + self.obs_mean.get(var_name, 0), # 反中心化
                        'news': news_value,
                        'weight': weight, # 占位符
                        'impact': impact_on_obs
                    })

        if not results_list:
            print("未找到可计算的新闻。")
            return pd.DataFrame(columns=['update date', 'updated variable', 'observed', 'forecast (prev)', 'news', 'weight', 'impact'])

        news_df = pd.DataFrame(results_list)
        # 按影响大小排序 (绝对值)
        news_df['abs_impact'] = news_df['impact'].abs()
        news_df = news_df.sort_values(by='abs_impact', ascending=False).drop(columns='abs_impact')
        news_df = news_df.set_index(['update date', 'updated variable'])

        print("'新闻'影响计算（占位符）完成。")
        return news_df

# --- 可以在这里添加示例用法 ---
if __name__ == '__main__':
    # 这里可以添加一个简单的示例，说明如何使用 DFMNowcastModel
    # 例如：
    # 1. 加载一个预先估计好的 DFMEMResultsWrapper 对象 (可能需要从文件加载)
    # 2. 创建 DFMNowcastModel 实例
    # 3. 加载新的观测数据
    # 4. 调用 model.apply(new_data)
    # 5. 调用 model.forecast(steps)
    # 6. 调用 model.news(previous_model, ...)
    print("DFM_Nowcasting.py 脚本可以直接运行（包含示例用法）。")

    # 示例需要 DFMEMResultsWrapper 实例等，这里仅作结构演示
    pass 