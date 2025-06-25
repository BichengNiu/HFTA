# -*- coding: utf-8 -*-
# Test comment added by AI
import sys
import os

# --- BEGIN UPDATED SYS.PATH SETUP ---
# Get the directory of the current file (DFM_Nowcasting.py, which is in news_analysis folder)
current_script_dir = os.path.dirname(os.path.abspath(__file__))
# Get the DFM directory (parent of news_analysis)
dfm_directory = os.path.abspath(os.path.join(current_script_dir, '..'))
# Get the project root directory (parent of DFM)
project_root_dir = os.path.abspath(os.path.join(dfm_directory, '..', '..'))
# Get the dashboard directory (parent of DFM)
dashboard_actual_dir = os.path.abspath(os.path.join(dfm_directory, '..'))

# Add project root directory to sys.path
if project_root_dir not in sys.path:
    sys.path.insert(0, project_root_dir)
    print(f"[DFM_Nowcasting] Added project_root ('{project_root_dir}') to sys.path.")

# 🔥 关键修复：创建模块别名以兼容joblib文件中的旧路径
try:
    import train_model.DynamicFactorModel as DynamicFactorModel
    import train_model.DiscreteKalmanFilter as DiscreteKalmanFilter
    sys.modules['DynamicFactorModel'] = DynamicFactorModel
    sys.modules['DiscreteKalmanFilter'] = DiscreteKalmanFilter
    print("[DFM_Nowcasting] 模块别名已设置，可兼容旧的joblib文件")
except ImportError as e:
    print(f"[DFM_Nowcasting] 模块别名设置失败: {e}")

# Add DFM directory to sys.path for potential imports from DFM or other subdirectories like news_analysis
if dfm_directory not in sys.path:
    sys.path.insert(0, dfm_directory)
    # print(f"[DFM_Nowcasting] Added dfm_directory ('{dfm_directory}') to sys.path.")

# Add dashboard directory to sys.path
if dashboard_actual_dir not in sys.path:
    sys.path.insert(0, dashboard_actual_dir)
    # print(f"[DFM_Nowcasting] Added dashboard_actual_dir ('{dashboard_actual_dir}') to sys.path.")

# Add news_analysis directory (current_script_dir) for imports within news_analysis itself (e.g. if split further)
if current_script_dir not in sys.path:
    sys.path.insert(0, current_script_dir)
    # print(f"[DFM_Nowcasting] Added current_script_dir ('{current_script_dir}') to sys.path.")
# --- END UPDATED SYS.PATH SETUP ---

"""
DFM_Nowcasting.py

包含 DFMNowcastModel 类，用于基于已估计的 DFM 模型进行即时预测更新、
预测和新闻分析。
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Union

# 修改导入语句，从本地模块导入
try:
    # 尝试从train_model目录导入DFM模块 - 使用绝对路径导入
    from train_model.DynamicFactorModel import DFMEMResultsWrapper
    from train_model.DiscreteKalmanFilter import KalmanFilter, FIS, KalmanFilterResultsWrapper, SKFResultsWrapper
    print("[DFM_Nowcasting] 成功从本地 train_model 模块导入")
except ImportError as e_import:
    print(f"[DFM_Nowcasting] 导入本地DFM模块失败: {e_import}")
    print("确保 DynamicFactorModel.py 和 DiscreteKalmanFilter.py 在 train_model 目录中。")
    try:
        # 备用：尝试直接导入（如果sys.path设置正确）
        from DynamicFactorModel import DFMEMResultsWrapper
        from DiscreteKalmanFilter import KalmanFilter, FIS, KalmanFilterResultsWrapper, SKFResultsWrapper
        print("[DFM_Nowcasting] 成功使用备用导入路径")
    except ImportError as e_backup:
        print(f"[DFM_Nowcasting] 备用导入也失败: {e_backup}")
        raise ImportError("无法导入必需的DFM模块") from e_import

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

        # --- 打印关键参数以供调试 ---
        print("\n--- Debug: 打印模型参数 A 和 Q ---")
        try:
            print(f"  A 矩阵 (状态转移):\n{self.A}")
            print(f"  Q 矩阵 (状态噪声协方差):\n{self.Q}")
            print(f"  R 矩阵 (观测噪声协方差，部分对角线):\n{np.diag(self.R)[:10]}...") # 仅打印前10个对角线元素
            print(f"  R 矩阵对角线最小值: {np.min(np.diag(self.R))}")
            print(f"  R 矩阵对角线最大值: {np.max(np.diag(self.R))}")
            # 计算 A 的特征值
            eigenvalues_A = np.linalg.eigvals(self.A)
            print(f"  A 矩阵的特征值:\n{eigenvalues_A}")
            print(f"  A 矩阵特征值的绝对值:\n{np.abs(eigenvalues_A)}")
            # 检查是否有特征值的绝对值 >= 1
            if np.any(np.abs(eigenvalues_A) >= 1.0):
                print("  警告: A 矩阵存在绝对值大于等于 1 的特征值，模型可能不稳定导致预测发散！")
            else:
                print("  A 矩阵特征值绝对值均小于 1，模型状态转移看似稳定。")
        except Exception as e_param_print:
            print(f"  打印参数或计算特征值时出错: {e_param_print}")
        # --- End Debug ---

        self.current_kf_results = baseline_kf_results # 如果传入，存储KF结果
        self.current_smooth_results = baseline_smooth_results if baseline_smooth_results else SKFResultsWrapper(x_sm=smoothed_states_base, P_sm=smoothed_cov_base, z=baseline_results.z)

        # --- 确保 B 矩阵形状正确 ---
        if self.B.shape != (self.n_factors, self.n_shocks):
             print(f"警告: 存储的 B 矩阵形状 {self.B.shape} 与预期的 ({self.n_factors}, {self.n_shocks}) 不符。将尝试重塑或使用零矩阵。")
             # 简单的处理：如果形状不匹配，创建一个零矩阵
             self.B = np.zeros((self.n_factors, self.n_shocks))


    def _preprocess_data(self, observation_data: pd.DataFrame) -> pd.DataFrame:
        """
        对输入的观测数据进行预处理（确保列顺序正确）。
        注意：此版本不再执行中心化，假定输入数据已标准化。

        Args:
            observation_data: 包含观测数据的 DataFrame (假定已标准化)。

        Returns:
            列顺序与模型期望一致的数据 DataFrame。
        """
        if not isinstance(observation_data, pd.DataFrame):
            raise TypeError("observation_data 必须是 Pandas DataFrame。")
        if not isinstance(observation_data.index, pd.DatetimeIndex):
             print("警告: observation_data 的索引不是 DatetimeIndex。")

        # 确保列顺序与 self.obs_names (从 obs_mean.index 推断) 一致
        # 这是必要的，因为卡尔曼滤波器的 H 矩阵 (Lambda) 的行顺序是固定的
        try:
            # --- 修改：只进行列检查和重排，不进行中心化 ---
            # 检查是否存在所有需要的列
            missing_cols = set(self.obs_names) - set(observation_data.columns)
            if missing_cols:
                 raise ValueError(f"输入数据中缺少必要的列: {missing_cols}")
            
            # 确保列的顺序正确
            data_reordered = observation_data[self.obs_names].copy()
            
        except KeyError as e:
            # 这个 KeyError 可能在 observation_data[self.obs_names] 时触发 (虽然上面的检查应该能捕捉到)
            missing_cols_alt = set(self.obs_names) - set(observation_data.columns)
            extra_cols = set(observation_data.columns) - set(self.obs_names)
            msg = f"输入数据的列与模型期望的列不匹配。\\n缺失: {missing_cols_alt}\\n多余: {extra_cols}"
            raise ValueError(msg) from e

        # --- 移除中心化步骤 ---
        # centered_data = data_reordered - self.obs_mean

        # --- Debug (可选): 检查处理后的输入数据 ---
        # print("\n--- Debug: 检查预处理后的输入数据 (data_reordered, 应已标准化) ---\")
        # try:
        #     print("  data_reordered 整体描述:\")
        #     print(data_reordered.describe().to_string())
        #     max_abs_val = data_reordered.abs().max().max()
        #     print(f"  data_reordered 中的最大绝对值: {max_abs_val:.6f}\")
        # except Exception as e_prep_debug:
        #     print(f"  检查 data_reordered 时出错: {e_prep_debug}\")
        # --- End Debug ---

        # --- 修改：返回重排后的数据，而不是中心化后的 ---
        return data_reordered

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

        # --- 恢复：使用线性插值填充 NaN ---
        print("  正在对 centered_data (KalmanFilter 输入 Z) 进行线性插值以填充 NaN...")
        initial_nan_count = centered_data.isna().sum().sum()
        if initial_nan_count > 0:
            # 按时间轴（列）进行线性插值
            centered_data = centered_data.interpolate(method='linear', axis=0, limit_direction='both')
            remaining_nan_count = centered_data.isna().sum().sum()
            print(f"    插值完成。初始 NaN 数量: {initial_nan_count}, 剩余 NaN 数量: {remaining_nan_count}")
            if remaining_nan_count > 0:
                 print("    警告：插值后仍有剩余 NaN！可能是因为列的开头或结尾有连续 NaN。")
                 # 可以选择更复杂的填充策略，如前向/后向填充剩余 NaN
                 # centered_data = centered_data.ffill().bfill()
                 # print("    已尝试使用 ffill/bfill 填充剩余 NaN。")
        else:
            print("    输入数据中未发现 NaN，跳过插值。")
        # --- 结束恢复插值 ---

        # --- 新增：检查 centered_data 是否包含 NaN ---
        print("  检查 KalmanFilter 的输入 Z (centered_data)...")
        if centered_data.isnull().values.any():
            print("    错误: 输入到 KalmanFilter 的 centered_data 包含 NaN!")
            nan_counts_input = centered_data.isna().sum()
            print("    输入 Z 每列 NaN 数量 (非零部分):")
            print(nan_counts_input[nan_counts_input > 0].to_string())
            # 可以选择在这里引发错误或仅仅打印警告
            # raise ValueError("输入数据包含 NaN，无法继续进行 KalmanFilter。")
        else:
            print("    输入 Z (centered_data) 检查通过，未发现 NaN。")
        # --- 结束新增 ---

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

        # --- 新增：检查 kf_results --- 
        print("  检查 KalmanFilter 输出...")
        nan_found_kf = False
        # 移除对不存在的 x_hat 和 P_hat 的检查
        # if kf_results.x_hat is not None and kf_results.x_hat.isna().any().any():
        #     print("    警告: kf_results.x_hat 包含 NaN!")
        #     nan_found_kf = True
        if kf_results.x_minus is not None and kf_results.x_minus.isna().any().any():
            print("    警告: kf_results.x_minus 包含 NaN!")
            nan_found_kf = True
        # for i, p_hat in enumerate(kf_results.P_hat):
        #     if p_hat is not None and np.isnan(p_hat).any():
        #         print(f"    警告: kf_results.P_hat[{i}] 包含 NaN!")
        #         nan_found_kf = True
        #         break # 找到一个就停止检查
        for i, p_minus in enumerate(kf_results.P_minus):
            if p_minus is not None and np.isnan(p_minus).any():
                print(f"    警告: kf_results.P_minus[{i}] 包含 NaN!")
                nan_found_kf = True
                break
        if not nan_found_kf:
            print("    KalmanFilter 输出 (x_minus, P_minus) 检查通过，未发现 NaN。") # 修改打印信息
        # --- 结束新增 --- 

        print("  调用 FIS (平滑器)...")
        smooth_results = FIS(kf_results)
        print("滤波和平滑完成。")

        # --- 新增：检查 smooth_results (包括 z, x_sm, P_sm) --- 
        print("  检查 FIS 输出 (smooth_results)...)")
        nan_found_smooth = False
        if smooth_results.z is not None and smooth_results.z.isna().any().any():
            print("    错误: smooth_results.z 包含 NaN!")
            nan_counts_z = smooth_results.z.isna().sum()
            print("    平滑后 z 每列 NaN 数量 (非零部分):")
            print(nan_counts_z[nan_counts_z > 0].to_string())
            nan_found_smooth = True
        elif smooth_results.z is None:
            print("    错误: smooth_results.z is None!")
            nan_found_smooth = True
            
        if smooth_results.x_sm is not None and smooth_results.x_sm.isna().any().any():
            print("    错误: smooth_results.x_sm 包含 NaN!")
            nan_found_smooth = True
        elif smooth_results.x_sm is None:
            print("    错误: smooth_results.x_sm is None!")
            nan_found_smooth = True
            
        # 检查 P_sm (列表)
        if smooth_results.P_sm is not None:
            for i, p_sm in enumerate(smooth_results.P_sm):
                if p_sm is not None and np.isnan(p_sm).any():
                    print(f"    错误: smooth_results.P_sm[{i}] 包含 NaN!")
                    nan_found_smooth = True
                    break
        elif smooth_results.P_sm is None:
            print("    错误: smooth_results.P_sm is None!")
            nan_found_smooth = True
            
        if not nan_found_smooth:
            print("    FIS 输出检查通过，smooth_results (z, x_sm, P_sm) 不包含 NaN。") # 修改打印信息
        # --- 结束新增 ---

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
        
        # --- 修改：处理 steps = 0 的情况 ---
        if steps == 0:
            # 如果预测 0 步，直接返回最后一个已知状态 (平滑状态)
            # 使用 last_date 作为索引
            last_known_state = self.current_smooth_results.x_sm.iloc[-1].values
            forecast_states_df = pd.DataFrame([last_known_state], index=[last_date], columns=self.state_names)
            forecast_covariances_list = [] # 没有预测协方差
        else:
            forecast_states_df = pd.DataFrame(forecast_states_list, index=forecast_index[:len(forecast_states_list)], columns=self.state_names)
        # --- 结束修改 ---
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
             model_frequency: str = 'M') -> pd.DataFrame: # 默认月度频率
        """
        计算新数据 vintage 相对于前一个 vintage 的 "新闻" 及其对特定变量预测的影响。

        Args:
            previous_vintage_model: 代表上一个数据 vintage 的 DFMNowcastModel 实例。
            impact_date: 要计算影响的目标日期 (与模型频率对齐)。
            impacted_variable: 要计算影响的目标观测变量名称。
            model_frequency: 模型的基本频率 (从 config.py 获取默认值)。

        Returns:
            一个 DataFrame，包含新闻分解结果。列包括:
            - update date: 新闻发生的时间点 t。
            - updated variable: 发生新闻的变量 j。
            - observed: 在当前 vintage 中观测到的值 (反中心化)。
            - forecast (prev): 基于前一个 vintage 信息对 t 时刻变量 j 的预测 (反中心化)。
            - news: 新闻值 (observed - forecast (prev)，中心化)。
            - weight: 单位新闻对 impact_date 时刻 impacted_variable 预测的影响。
            - impact: 该新闻对 impact_date 时刻 impacted_variable 预测的总影响。

        注意: 此实现假设模型参数在两个 vintages 之间是固定的。
              它侧重于数据发布和修订的影响。
        """
        print(f"开始计算 '新闻' 影响 (对比当前 vs 前一 vintage)...")
        # 使用传入的 model_frequency 或来自 config 的默认值
        effective_frequency = model_frequency 
        print(f"  目标日期: {impact_date}, 目标变量: {impacted_variable}, 模型频率: {effective_frequency}")

        # --- 移除：跟踪点 1 --- 
        # print("[News Debug] Entering news method successfully.")

        # --- 基本检查和设置 ---
        if not isinstance(previous_vintage_model, DFMNowcastModel):
            raise TypeError("previous_vintage_model 必须是 DFMNowcastModel 的实例。")
        if impacted_variable not in self.obs_names:
             raise ValueError(f"目标变量 '{impacted_variable}' 不在模型观测变量列表中。")
        try:
            # 确保 impact_date 是 Timestamp 并与频率对齐
            impact_date_ts = pd.Timestamp(impact_date)
            # 可选：根据频率对齐 impact_date_ts，例如对于 'MS'，确保是月初
            # impact_date_ts = impact_date_ts.to_period(model_frequency).to_timestamp()
        except ValueError:
             raise ValueError(f"无法将 impact_date '{impact_date}' 转换为时间戳。")

        current_z = self.current_smooth_results.z
        previous_z = previous_vintage_model.current_smooth_results.z
        prev_kf_results = previous_vintage_model.current_kf_results

        # --- 新增 Debug: 检查传入的 Z 数据 ---
        print("\n--- [News Debug] 检查传入的 Z 数据 --- ")
        print(f"  current_z shape: {current_z.shape}")
        print(f"  previous_z shape: {previous_z.shape}")
        debug_var = 'MEG：产能利用率：中国（周）' # 选择一个已知有新闻的变量
        debug_dates = ['2024-12-20', '2024-12-27']
        if debug_var in current_z.columns:
            print(f"  Values for '{debug_var}' at key dates:")
            for dt_str in debug_dates:
                dt = pd.Timestamp(dt_str)
                val_curr = current_z.loc[dt, debug_var] if dt in current_z.index else '[Not Found]'
                val_prev = previous_z.loc[dt, debug_var] if dt in previous_z.index else '[Not Found]'
                print(f"    {dt_str}: current_z = {val_curr}, previous_z = {val_prev}")
        else:
            print(f"  调试变量 '{debug_var}' 不在 Z 的列中。")
        print("--- [News Debug] 结束检查 ---\n")
        # --- 结束新增 Debug ---

        if prev_kf_results is None or prev_kf_results.x_minus is None or prev_kf_results.Kalman_gain is None:
            raise ValueError("无法计算新闻，前一个 vintage 的卡尔曼滤波结果 (x_minus, Kalman_gain) 不可用。")

        impacted_var_index = self.obs_names.index(impacted_variable)
        lambda_impacted_row = self.Lambda[impacted_var_index, :]
        obs_mean_impacted = self.obs_mean.get(impacted_variable, 0)

        # 合并并排序索引，只考虑影响日期之前或当天的
        combined_index = current_z.index.union(previous_z.index).sort_values()
        relevant_index = combined_index[combined_index <= impact_date_ts]

        # --- 准备存储结果 ---
        results_list = []

        # --- 迭代时间步 t ---
        print(f"迭代 {len(relevant_index)} 个相关时间点进行新闻分析...")
        # --- 新增：获取旧 vintage 的最后时间和状态信息 ---
        last_date_prev = previous_z.index.max()
        last_x_sm_prev = None
        last_P_sm_prev = None
        if previous_vintage_model.current_smooth_results and previous_vintage_model.current_smooth_results.x_sm is not None:
             try:
                 last_x_sm_prev = previous_vintage_model.current_smooth_results.x_sm.loc[last_date_prev].values
                 last_P_sm_prev = previous_vintage_model.current_smooth_results.P_sm.loc[last_date_prev].values # 假设P_sm是DataFrame
             except (KeyError, AttributeError):
                 print(f"警告：无法从旧 vintage 获取最后的平滑状态 (x_sm) 或协方差 (P_sm) 在 {last_date_prev}。预测可能不准确。")
        # --- 结束新增 ---

        for timestamp in relevant_index:
            # --- 移除：跟踪点 2 ---
            # print(f"[News Debug] Processing timestamp: {timestamp}")
            # 获取前一个 vintage 在 t 时刻的预测状态 x_{t|t-1}
            x_minus_t_prev = None # 初始化
            try:
                # --- 修改：区分 timestamp 是否在旧 vintage 范围内 ---
                if timestamp <= last_date_prev:
                    # 如果在范围内，直接查找
                    x_minus_t_prev = prev_kf_results.x_minus.loc[timestamp].values
                elif last_x_sm_prev is not None:
                    # 如果超出范围，且我们有旧 vintage 的最后状态，则预测
                    # 计算需要预测的步数
                    steps_to_forecast_state = 0
                    try:
                        start_period_pred = pd.Period(last_date_prev, freq=effective_frequency)
                        end_period_pred = pd.Period(timestamp, freq=effective_frequency)
                        steps_to_forecast_state = pd.period_range(start=start_period_pred, end=end_period_pred).size - 1
                        if steps_to_forecast_state < 1: steps_to_forecast_state = 1 # 至少预测一步

                    except ValueError as e_step:
                         print(f"警告: 计算预测 x_minus_t_prev 的步数时出错 ({last_date_prev} -> {timestamp}): {e_step}。假设预测 1 步。")
                         steps_to_forecast_state = 1

                    # 预测状态 x_t = A^k * x_{t-k}
                    if steps_to_forecast_state > 0:
                        try:
                            A_pow_k_pred = np.linalg.matrix_power(self.A, steps_to_forecast_state)
                            # 注意：这里用的是旧 vintage 的最后 *平滑* 状态 x_sm 作为起点预测 x_minus
                            # 理论上更精确的是用旧 vintage 的最后 *预测* 状态 x_minus，但平滑状态通常更稳定
                            x_minus_t_prev = A_pow_k_pred @ last_x_sm_prev
                        except np.linalg.LinAlgError as e_pow:
                            print(f"警告: 计算 A^{steps_to_forecast_state} (用于预测 x_minus_t_prev) 时出错: {e_pow}。")
                        except Exception as e_pred:
                            print(f"警告: 预测 x_minus_t_prev (从 {last_date_prev} 到 {timestamp}) 时出错: {e_pred}。")
                    else: # steps <= 0 的情况，理论上不应发生，但作为回退
                         x_minus_t_prev = last_x_sm_prev # 直接使用最后状态
                # --- 结束修改 ---

                # --- 新增：如果 x_minus_t_prev 仍然是 None，则跳过 ---
                if x_minus_t_prev is None:
                    print(f"    [News Warning] t={timestamp.strftime('%Y-%m-%d')}: 无法获取或预测前一 vintage 的状态 x_minus_t_prev。跳过此时间点。")
                    continue
                # --- 结束新增 ---

            except KeyError:
                # --- 移除：旧的警告，已被上面的逻辑覆盖 ---
                # print(f"[News Debug] Processing timestamp: {timestamp}")
                # print(f"    [News Warning] t={timestamp.strftime('%Y-%m-%d')}: 未找到前一 vintage 的预测状态 x_minus_t_prev。跳过此时间点进行新闻计算。")
                # continue # 恢复 continue
                # --- 结束移除 ---
                # --- 新增：如果查找失败（理论上不应发生，因已被 if timestamp <= last_date_prev 处理），记录并跳过 ---
                print(f"    [News Error] t={timestamp.strftime('%Y-%m-%d')}: 尝试直接查找旧 vintage x_minus 时发生未预期的 KeyError。跳过。")
                continue
                # --- 结束新增 ---

            # 计算前一个 vintage 对 t 时刻观测值 z_t 的预测
            forecast_z_t_prev_centered = self.Lambda @ x_minus_t_prev
            forecast_z_t_prev_series = pd.Series(forecast_z_t_prev_centered, index=self.obs_names)

            # --- 修改：获取 *当前* vintage 在 t 时刻的卡尔曼增益 K_t ---
            # 获取卡尔曼增益 K_t (来自当前 vintage)
            current_kf_results = self.current_kf_results # 使用当前 vintage 的 KF 结果
            if current_kf_results is None or current_kf_results.Kalman_gain is None:
                 print(f"    [News Error] t={timestamp.strftime('%Y-%m-%d')}: 当前 vintage 的 Kalman_gain 不可用，无法计算权重。跳过此时间点。")
                 continue # 无法计算，跳过

            try:
                # 需要找到 timestamp 在 *当前* KF 结果索引中的位置
                t_idx = current_kf_results.x_minus.index.get_loc(timestamp) # 假设 x_minus 和 Kalman_gain 索引对齐

                if t_idx >= len(current_kf_results.Kalman_gain) or current_kf_results.Kalman_gain[t_idx] is None:
                    print(f"    [News Warning] t={timestamp.strftime('%Y-%m-%d')}: 在当前 vintage 的 Kalman_gain 列表索引 {t_idx} 处找不到有效增益，假设为零。")
                    K_t_current = np.zeros((self.n_factors, self.n_obs))
                else:
                    K_t_current = np.array(current_kf_results.Kalman_gain[t_idx])
                    if K_t_current.shape != (self.n_factors, self.n_obs):
                         print(f"    [News Warning] t={timestamp.strftime('%Y-%m-%d')}: 当前 vintage 在索引 {t_idx} 的卡尔曼增益形状 {K_t_current.shape} 不正确，应为 {(self.n_factors, self.n_obs)}。假设为零。")
                         K_t_current = np.zeros((self.n_factors, self.n_obs))
            except KeyError:
                 print(f"    [News Warning] t={timestamp.strftime('%Y-%m-%d')}: 时间戳不在当前 vintage 的 KF 结果索引中，无法获取卡尔曼增益，假设为零。")
                 K_t_current = np.zeros((self.n_factors, self.n_obs))
            except IndexError:
                 print(f"    [News Warning] t={timestamp.strftime('%Y-%m-%d')}: 计算出的索引 {t_idx} 超出了当前 vintage 卡尔曼增益列表的范围，假设为零。")
                 K_t_current = np.zeros((self.n_factors, self.n_obs))
            # --- 结束修改 ---


            # 获取当前和之前的观测值 (中心化)
            z_t_current = current_z.loc[timestamp] if timestamp in current_z.index else pd.Series(index=self.obs_names, dtype=float) * np.nan
            z_t_previous = previous_z.loc[timestamp] if timestamp in previous_z.index else pd.Series(index=self.obs_names, dtype=float) * np.nan

            # --- 识别并计算每个变量的新闻 ---
            news_items_at_t = []
            for j, var_name in enumerate(self.obs_names):
                # --- 移除：跟踪点 3 --- 
                # print(f"[News Debug] Processing variable j={j}, name={var_name}")
                obs_curr_centered = z_t_current.iloc[j]
                obs_prev_centered = z_t_previous.iloc[j]
                fcst_prev_centered = forecast_z_t_prev_series.iloc[j]

                is_news = False
                news_value = 0.0

                if pd.notna(obs_curr_centered):
                    # 检查是否为新发布或修正
                    if pd.isna(obs_prev_centered) or obs_curr_centered != obs_prev_centered:
                        is_news = True
                        # 修复符号错误：新闻定义为预测 - 实际观测，使符号与影响方向一致
                        news_value = fcst_prev_centered - obs_curr_centered
                        # --- 移除：过滤掉几乎为零的新闻 --- 
                        # if np.abs(news_value) < 1e-9:
                        #     is_news = False
                        # --- 结束移除 ---


                if is_news:
                    # 计算影响传播步数 k = T - t
                    try:
                        # 使用 pandas Period 来计算步数差异
                        start_period = pd.Period(timestamp, freq=effective_frequency)
                        end_period = pd.Period(impact_date_ts, freq=effective_frequency)
                        # PeriodIndex 可以计算整数差异
                        steps_to_propagate = pd.period_range(start=start_period, end=end_period).size -1

                    except ValueError as e:
                        print(f"警告: 无法使用频率 '{effective_frequency}' 计算 {timestamp} 和 {impact_date_ts} 之间的步数: {e}。将使用简化方法。")
                        # 简化：尝试按月计算（如果适用）
                        if 'M' in effective_frequency.upper():
                             steps_to_propagate = (impact_date_ts.year - timestamp.year) * 12 + (impact_date_ts.month - timestamp.month)
                        elif 'Q' in effective_frequency.upper():
                             steps_to_propagate = (impact_date_ts.year - timestamp.year) * 4 + (impact_date_ts.quarter - timestamp.quarter)
                        else: # 其他情况，例如日，使用天数（可能不准确）
                             steps_to_propagate = (impact_date_ts - timestamp).days

                        if steps_to_propagate < 0: # 确保非负
                             print(f"警告: 计算出的传播步数为负 ({steps_to_propagate})，将设为 0。")
                             steps_to_propagate = 0


                    if steps_to_propagate < 0:
                        # print(f"调试: 更新日期 {timestamp} 在影响日期 {impact_date_ts} 之后，跳过传播。")
                        continue # 新闻发生在影响日期之后

                    # 计算 A^k
                    if steps_to_propagate == 0:
                         A_pow_k = np.eye(self.n_factors)
                    else:
                         try:
                              A_pow_k = np.linalg.matrix_power(self.A, steps_to_propagate)
                         except np.linalg.LinAlgError as e:
                             print(f"警告: 计算 A^{steps_to_propagate} 时发生线性代数错误: {e}。影响计算可能不准确。")
                             A_pow_k = np.eye(self.n_factors) # 或者其他回退策略

                    # 获取 K_t 的第 j 列
                    # --- 增强检查：确保 K_t_prev 有效且索引 j 在范围内 ---
                    # --- 修改：检查 K_t_current ---
                    if not isinstance(K_t_current, np.ndarray) or K_t_current.shape[1] != self.n_obs:
                        print(f"    [News Warning] t={timestamp.strftime('%Y-%m-%d')}, var={var_name}: K_t_current 无效或列数 ({K_t_current.shape[1] if isinstance(K_t_current, np.ndarray) else 'N/A'}) 与 n_obs ({self.n_obs}) 不匹配, skipping.")
                        continue
                    if j >= K_t_current.shape[1]:
                        print(f"    [News Warning] t={timestamp.strftime('%Y-%m-%d')}, var={var_name}: Index j={j} 超出 K_t_current 列范围 ({K_t_current.shape[1]}), skipping.")
                        continue
                    # --- 结束增强检查 ---

                    try:
                        # --- 修改：使用 K_t_current ---
                        K_t_j = K_t_current[:, j]
                    except IndexError: # 冗余检查，以防万一
                        print(f"    [News Warning] t={timestamp.strftime('%Y-%m-%d')}, var={var_name}: Index j={j} out of bounds for K_t_current columns, skipping.")
                        continue

                    # 计算权重和影响
                    # --- 增强检查：确保 lambda_impacted_row, A_pow_k, K_t_j 有效 ---
                    valid_inputs = True
                    if not isinstance(lambda_impacted_row, np.ndarray) or lambda_impacted_row.shape != (self.n_factors,):
                        print(f"    [News Warning] t={timestamp.strftime('%Y-%m-%d')}, var={var_name}: lambda_impacted_row 无效或形状错误 ({lambda_impacted_row.shape if isinstance(lambda_impacted_row, np.ndarray) else 'N/A'}), skipping.")
                        valid_inputs = False
                    if not isinstance(A_pow_k, np.ndarray) or A_pow_k.shape != (self.n_factors, self.n_factors):
                        print(f"    [News Warning] t={timestamp.strftime('%Y-%m-%d')}, var={var_name}: A_pow_k 无效或形状错误 ({A_pow_k.shape if isinstance(A_pow_k, np.ndarray) else 'N/A'}), skipping.")
                        valid_inputs = False
                    if not isinstance(K_t_j, np.ndarray) or K_t_j.shape != (self.n_factors,):
                        print(f"    [News Warning] t={timestamp.strftime('%Y-%m-%d')}, var={var_name}: K_t_j 无效或形状错误 ({K_t_j.shape if isinstance(K_t_j, np.ndarray) else 'N/A'}), skipping.")
                        valid_inputs = False
                    
                    if not valid_inputs:
                        continue
                    # --- 结束增强检查 ---

                    weight = lambda_impacted_row @ A_pow_k @ K_t_j
                    item_impact = weight * news_value

                    # 存储结果（反中心化观测值和预测值）
                    obs_mean_j = self.obs_mean.get(var_name, 0)
                    results_list.append({
                        'update date': timestamp,
                        'updated variable': var_name,
                        'observed': obs_curr_centered + obs_mean_j, # 反中心化
                        'forecast (prev)': fcst_prev_centered + obs_mean_j, # 反中心化
                        'news': news_value, # 新闻值保持中心化
                        'weight': weight,
                        'impact': item_impact
                    })

        # --- Finalize DataFrame ---
        if not results_list:
            print("在此 vintage 更新中未找到可计算的新闻。")
            return pd.DataFrame(columns=['update date', 'updated variable', 'observed', 'forecast (prev)', 'news', 'weight', 'impact'])

        news_df = pd.DataFrame(results_list)

        # 按影响绝对值排序
        news_df['abs_impact'] = news_df['impact'].abs()
        # 优先按更新日期排序，然后按绝对影响排序
        news_df = news_df.sort_values(by=['update date', 'abs_impact'], ascending=[True, False])
        news_df = news_df.drop(columns='abs_impact')
        news_df = news_df.set_index(['update date', 'updated variable'])

        print(f"'新闻'影响计算完成，共找到 {len(news_df)} 条新闻。")
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