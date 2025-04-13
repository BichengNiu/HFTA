# -*- coding: utf-8 -*-
# print("DEBUG: Script execution started.") # REMOVE DEBUG
"""
超参数和变量逐步前向选择脚本。
目标：最小化 OOS RMSE。
"""
import pandas as pd
import numpy as np
import sys
import os
import time
import warnings
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns # <-- 新增导入 Seaborn
import concurrent.futures
from tqdm import tqdm
from sklearn.metrics import mean_squared_error, mean_absolute_error
import traceback
from typing import Tuple, List, Dict, Union # 添加 Tuple
import unicodedata # <-- 新增导入
from statsmodels.tsa.stattools import adfuller # <--- 新增 ADF 检验导入

# --- 尝试导入自定义模块 ---
try:
    from data_preparation import prepare_data
    from DynamicFactorModel import DFM_EMalgo
except ImportError as e:
    print(f"错误：导入自定义模块失败: {e}")
    sys.exit(1)

# --- 配置 ---
warnings.filterwarnings('ignore')
matplotlib.use('Agg')
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# --- 新增：是否使用 Log(Yt) - Log(Yt-52) 转换 (根据用户要求设为 True) ---
# *** MODIFIED: Set USE_LOG_YOY_TRANSFORM to True based on user request ***
USE_LOG_YOY_TRANSFORM = True # 设置为 True，对预测变量进行对数同比转换，并跳过后续差分

# --- 常量 ---
# *** MODIFIED: Update Excel file if different, TARGET_VARIABLE, TARGET_FREQ ***
EXCEL_DATA_FILE = 'dym_estimate/经济数据库.xlsx'
TARGET_VARIABLE = '规模以上工业增加值:当月同比' # NEW Target Variable
TARGET_FREQ = 'W-FRI'                   # Target frequency remains Weekly Friday
# *** ADDED: Define TARGET_SHEET_NAME constant ***
TARGET_SHEET_NAME = '工业增加值同比增速-月度' # NEW: Specify the sheet containing the target

RESULTS_FILE = "tuning_results_forward_selection.csv"
DETAILED_LOG_FILE = "调优日志.txt"
FINAL_FACTOR_FILE = os.path.join('dfm_result', 'final_factors.png')
FINAL_PLOT_FILE = os.path.join('dfm_result', 'final_nowcast_comparison.png')
EXCEL_OUTPUT_FILE = os.path.join('dfm_result', 'result.xlsx') # Renamed output file
START_DATE = '2020-01-01'
END_DATE = '2024-12-31'
N_ITER_FIXED = 15

# --- OOS 验证日期 ---
TRAIN_END_DATE = '2024-06-28' # 修正为 6 月底最后一个周五
VALIDATION_START_DATE = '2024-07-05' # 修正为 7 月初第一个周五
VALIDATION_END_DATE = '2024-12-27' # 保持不变

# --- 并行计算 ---
MAX_WORKERS = os.cpu_count() - 1 if os.cpu_count() and os.cpu_count() > 1 else 1

# --- 超参数配置 ---
# FIX: Remove hardcoded K_FACTORS_RANGE. It will be determined dynamically.
# K_FACTORS_RANGE = [2, 3, 4, 5, 6] 
# HYPERPARAMS_TO_TUNE will be built dynamically in run_tuning
HYPERPARAMS_TO_TUNE = [] 

# --- 新增：对数同比转换函数 ---
def apply_log_yoy_transform(data: pd.DataFrame | pd.Series, periods: int = 52, clip_value: float = 1e-9) -> pd.DataFrame | pd.Series:
    """
    计算数据的对数同比 log(y_t) - log(y_{t-periods})。
    处理非正数，将其替换为小的正数 clip_value。
    """
    if data.empty:
        return data

    original_type = type(data)
    if isinstance(data, pd.Series):
        data = data.to_frame(name=data.name or 'value') # Convert Series to DataFrame

    data_log = np.log(data.clip(lower=clip_value))
    data_log_shifted = data_log.shift(periods)
    log_yoy = data_log - data_log_shifted
    
    if original_type is pd.Series:
        return log_yoy.iloc[:, 0] # Return Series if input was Series
    else:
        return log_yoy

# --- 新增：基于 ADF 检验的变量级转换函数 ---
def apply_stationarity_transforms(data: pd.DataFrame, target_variable: str, adf_p_threshold: float = 0.05) -> Tuple[pd.DataFrame, Dict[str, str]]:
    """
    对 DataFrame 中的预测变量应用转换以确保平稳性 (基于 ADF 检验)。
    目标变量保持不变。

    Args:
        data (pd.DataFrame): 输入的 DataFrame (假设索引是 DatetimeIndex).
        target_variable (str): 要保持不变的目标变量列名。
        adf_p_threshold (float): ADF 检验 p 值的阈值，低于此值为平稳。

    Returns:
        Tuple[pd.DataFrame, Dict[str, str]]: 包含转换后数据的 DataFrame，以及一个记录每个变量应用的转换 ('level' 或 'diff') 的字典。
    """
    transformed_data = data.copy()
    transform_log = {}
    predictor_variables = [col for col in data.columns if col != target_variable]

    # print(f"  [Transform] 开始基于 ADF (p<{adf_p_threshold}) 进行变量级平稳性转换...") # Line 110 commented out

    for col in predictor_variables:
        series = data[col].dropna() # ADF 检验前移除 NaN
        if series.empty:
            print(f"    列 '{col}': 全为 NaN 或空，保持原样 (全NaN)。")
            transformed_data[col] = np.nan # 确保输出为 NaN
            transform_log[col] = 'skipped_empty'
            continue
        if series.nunique() == 1:
            print(f"    列 '{col}': 只有一个唯一值 (常量)，保持原样 (level)。")
            transformed_data[col] = data[col] # 使用原始数据（可能含 NaN）
            transform_log[col] = 'level_constant'
            continue

        try:
            adf_result = adfuller(series)
            p_value = adf_result[1]
            if p_value < adf_p_threshold:
                # 平稳，使用原始水平值
                # print(f"    列 '{col}': ADF p-value={p_value:.3f} < {adf_p_threshold}, 平稳。使用 level。") # 注释掉详细日志
                transformed_data[col] = data[col] # 使用原始数据
                transform_log[col] = 'level'
            else:
                # 非平稳，应用一阶差分
                # print(f"    列 '{col}': ADF p-value={p_value:.3f} >= {adf_p_threshold}, 非平稳。应用 diff(1)。") # 注释掉详细日志
                transformed_data[col] = data[col].diff(1)
                transform_log[col] = 'diff'
        except Exception as e:
            print(f"    列 '{col}': ADF 检验或差分时出错: {e}。保持原样 (level)。")
            transformed_data[col] = data[col]
            transform_log[col] = 'level_error'

    # 确保目标变量存在且未被修改
    if target_variable in data.columns:
        transformed_data[target_variable] = data[target_variable]
        transform_log[target_variable] = 'target_level'
    else:
         print(f"   [Transform] 警告: 目标变量 '{target_variable}' 在输入数据中未找到!")

    # 移除因差分产生的首行 NaN
    initial_rows = len(transformed_data)
    transformed_data = transformed_data.dropna(axis=0, how='all') # 先移除全 NaN 行
    if transform_log and any(v == 'diff' for v in transform_log.values()):
         # 只有在进行了差分时才移除可能由差分产生的首行 NaN
         # 找到第一个有效索引
         first_valid_idx = transformed_data.first_valid_index()
         if first_valid_idx is not None and first_valid_idx > transformed_data.index.min():
              # print(f"  [Transform] 移除了 {pd.Timestamp(first_valid_idx) - pd.Timestamp(transformed_data.index.min())} 的前导期数据 (因差分)。") # 移除日志
              transformed_data = transformed_data.loc[first_valid_idx:]

    # print(f"  [Transform] 变量级转换完成。数据 Shape: {transformed_data.shape} (移除前导NaN后)") # 已注释掉
    # print(f"  [Transform] 转换记录: {transform_log}") # 可选：打印详细转换日志
    return transformed_data, transform_log

# --- DFM 评估函数 (修改返回值) ---
def evaluate_dfm_params(
    variables: list[str],
    full_data: pd.DataFrame,
    target_variable: str,
    params: dict, # Now ONLY contains 'k_factors'
    var_type_map: dict,
    validation_start: str,
    validation_end: str,
    target_freq: str,
    train_end_date: str,
    max_iter: int = 50,
) -> Tuple[float, float, float, float, bool]: # 返回 (is_rmse, oos_rmse, is_hit_rate, oos_hit_rate, is_svd_error)
    """评估给定变量和参数下的 DFM。
    返回: (is_rmse, oos_rmse, is_hit_rate, oos_hit_rate, is_svd_error) 元组。
          如果评估失败或无法计算指标，RMSE=np.inf, Hit Rate=-np.inf
          is_svd_error 指示失败是否由 SVD 不收敛引起。
    """
    start_time = time.time()
    k_factors = params.get('k_factors', None)
    if k_factors is None:
        # print(f"错误: evaluate_dfm_params 的 params 字典缺少 k_factors: {params}") # 保持注释
        return np.inf, np.inf, -np.inf, -np.inf, False # 失败
    n_shocks = k_factors

    try:
        if target_variable not in variables:
             print(f"    错误: 目标变量 {target_variable} 不在当前变量列表中: {variables}")
             return np.inf, np.inf, -np.inf, -np.inf, False
        predictor_vars = [v for v in variables if v != target_variable]
        original_target_series_full = full_data[target_variable].copy()

        data_for_fitting = full_data[:validation_end].copy()
        current_variables = variables.copy()
        if not all(col in data_for_fitting.columns for col in current_variables):
             missing = [col for col in current_variables if col not in data_for_fitting.columns]
             print(f"    错误: 评估函数缺少列: {missing}")
             return np.inf, np.inf, -np.inf, -np.inf, False
        current_data = data_for_fitting[current_variables].copy()
        current_data = current_data.apply(pd.to_numeric, errors='coerce')

        # --- 修改: 使用变量级转换代替全局 LogYoY ---
        current_data_transformed, transform_details = apply_stationarity_transforms(current_data, target_variable)
        # --- 结束修改 ---

        # --- 原 LogYoY 逻辑 (注释掉) ---
        # if USE_LOG_YOY_TRANSFORM and predictor_vars:
        #     # print(f"    Applying log YoY transform (periods=52) to predictor variables...") # 注释掉
        #     predictors_transformed = apply_log_yoy_transform(current_data[predictor_vars])
        #     current_data = pd.concat([predictors_transformed, current_data[target_variable]], axis=1)
        #     current_data = current_data[variables]
        #     if not current_data.empty:
        #         # print("    Removing 52 leading NaN rows due to LogYoY...") # 注释掉
        #         if 52 >= len(current_data):
        #               raise ValueError(f"最终运行数据准备错误: 数据行数 ({len(current_data)}) 不足以移除 LogYoY 的前导 NaN (52)")
        #         current_data = current_data.iloc[52:]
        #     if current_data.empty:
        #         # print(f"    错误: Vars={len(variables)}, n_factors={k_factors} -> 对数同比转换后数据为空 (移除前导 NaN 后)") # 注释掉错误信息
        #         return np.inf, np.inf, -np.inf, -np.inf, False
        # --- 结束原 LogYoY 逻辑 ---

        # 使用转换后的数据进行后续处理
        current_data = current_data_transformed
        current_variables = current_data.columns.tolist() # 变量列表可能因全 NaN 列被移除而改变
        if target_variable not in current_variables:
             print(f"    错误: Vars={len(variables)}, n_factors={k_factors} -> 目标变量在变量级转换后丢失 (可能全为 NaN)。")
             return np.inf, np.inf, -np.inf, -np.inf, False
        predictor_vars = [v for v in current_variables if v != target_variable]

        if current_data.empty:
             # print(f"    错误: Vars={len(current_variables)}, n_factors={k_factors} -> 变量级转换后数据为空") # 注释掉
             return np.inf, np.inf, -np.inf, -np.inf, False

        all_nan_cols = current_data.columns[current_data.isna().all()].tolist()
        if all_nan_cols:
            print(f"    警告: Vars={len(current_variables)}, n_factors={k_factors} -> 以下列在变量级转换后全为 NaN，将被移除: {all_nan_cols}")
            current_variables = [v for v in current_variables if v not in all_nan_cols]
            if target_variable not in current_variables:
                print(f"    错误: 目标变量在移除全 NaN 列后被移除。")
                return np.inf, np.inf, -np.inf, -np.inf, False
            current_data = current_data.drop(columns=all_nan_cols)
            predictor_vars = [v for v in current_variables if v != target_variable]
            if current_data.empty or current_data.shape[1] < k_factors:
                print(f"    错误: 移除全 NaN 列后，变量数 ({current_data.shape[1]}) 不足因子数 ({k_factors}) 或数据为空。")
                return np.inf, np.inf, -np.inf, -np.inf, False

        if current_data.isnull().all().all():
             print(f"    错误: Vars={len(current_variables)}, n_factors={k_factors} -> 数据在移除全 NaN 列后所有值仍为 NaN。")
             return np.inf, np.inf, -np.inf, -np.inf, False

        # print(f"    Standardizing data (Shape: {current_data.shape})...") # 注释掉
        obs_mean = current_data.mean(skipna=True)
        obs_std = current_data.std(skipna=True)
        zero_std_cols = obs_std.index[obs_std == 0].tolist()
        if zero_std_cols:
             print(f"    警告: Vars={len(current_variables)}, n_factors={k_factors} -> 以下列标准差为 0，将设为 1: {zero_std_cols}")
             obs_std[zero_std_cols] = 1.0
        current_data_std = (current_data - obs_mean) / obs_std
        
        target_mean_for_rescaling = obs_mean.get(target_variable, np.nan)
        target_std_for_rescaling = obs_std.get(target_variable, np.nan)
        if pd.isna(target_mean_for_rescaling) or pd.isna(target_std_for_rescaling):
            print(f"    错误: 无法获取目标变量 {target_variable} 的均值/标准差进行反标准化")
            return np.inf, np.inf, -np.inf, -np.inf, False

        std_all_nan_cols = current_data_std.columns[current_data_std.isna().all()].tolist()
        if std_all_nan_cols:
             print(f"    错误: Vars={len(current_variables)}, n_factors={k_factors} -> 标准化后以下列变为全 NaN: {std_all_nan_cols}")
             return np.inf, np.inf, -np.inf, -np.inf, False
             
        if current_data_std.shape[0] < k_factors:
              print(f"    错误: Vars={len(current_variables)}, n_factors={k_factors} -> 处理后数据行数 ({current_data_std.shape[0]}) 不足因子数 ({k_factors})。")
              return np.inf, np.inf, -np.inf, -np.inf, False

        nan_count_std = current_data_std.isna().sum().sum()
        if nan_count_std > 0:
             # print(f"    信息: Vars={len(current_variables)}, n_factors={k_factors} -> 传递包含 NaN ({nan_count_std}) 的数据给 DFM。") # 注释掉
             pass # 保持安静

        # --- 新增: 在训练评估时忽略 1月/2月 目标值 --- 
        current_data_std_masked_for_fit = current_data_std.copy()
        month_indices = current_data_std_masked_for_fit.index.month
        current_data_std_masked_for_fit.loc[(month_indices == 1) | (month_indices == 2), target_variable] = np.nan
        # --- 结束新增 ---

        dfm_results = DFM_EMalgo(
            observation=current_data_std_masked_for_fit, # <--- 使用带掩码的数据进行拟合
            n_factors=k_factors,
            n_shocks=n_shocks, 
            n_iter=max_iter
        )

        if not hasattr(dfm_results, 'x_sm') or dfm_results.x_sm is None or dfm_results.x_sm.empty or \
           not hasattr(dfm_results, 'Lambda') or dfm_results.Lambda is None:
            print(f"    错误: Vars={len(current_variables)}, n_factors={k_factors} -> DFM 结果不完整")
            return np.inf, np.inf, -np.inf, -np.inf, False

        factors_sm = dfm_results.x_sm
        lambda_matrix = dfm_results.Lambda
        if lambda_matrix.shape[1] != k_factors:
             print(f"    警告: Lambda 矩阵列数 ({lambda_matrix.shape[1]}) 与预期因子数 ({k_factors}) 不符。可能使用 PCA 回退初始化？")
             if lambda_matrix.shape[1] > k_factors:
                  lambda_matrix = lambda_matrix[:, :k_factors]
             else: 
                  return np.inf, np.inf, -np.inf, -np.inf, False 
        
        lambda_df = pd.DataFrame(lambda_matrix, index=current_variables, columns=[f'Factor{i+1}' for i in range(k_factors)])

        if target_variable not in lambda_df.index:
            print(f"    错误: Vars={len(current_variables)}, n_factors={k_factors} -> 目标变量 {target_variable} 不在 Lambda 索引中 (可用: {lambda_df.index.tolist()})")
            return np.inf, np.inf, -np.inf, -np.inf, False
        
        lambda_target = lambda_matrix[lambda_df.index.get_loc(target_variable), :]

        nowcast_standardized = factors_sm.to_numpy() @ lambda_target 
        nowcast_orig_values = nowcast_standardized * target_std_for_rescaling + target_mean_for_rescaling
        nowcast_series_orig = pd.Series(nowcast_orig_values, index=factors_sm.index, name='Nowcast_Orig')

        target_original_series = original_target_series_full.copy()
        target_for_comparison = target_original_series.dropna()

        common_index = nowcast_series_orig.index.intersection(target_for_comparison.index)
        if common_index.empty:
            print(f"    错误: Vars={len(current_variables)}, n_factors={k_factors} -> 反标准化 Nowcast 和目标序列没有共同索引")
            return np.inf, np.inf, -np.inf, -np.inf, False
        
        aligned_df = pd.DataFrame({
            'Nowcast_Orig': nowcast_series_orig.loc[common_index],
            target_variable: target_for_comparison.loc[common_index]}).dropna()
        
        # --- 计算 RMSE 和 Hit Rate --- 
        is_rmse = np.inf # Changed from is_mae
        oos_rmse = np.inf # Changed from oos_mae
        is_hit_rate = -np.inf
        oos_hit_rate = -np.inf

        train_df = aligned_df.loc[:train_end_date]
        if not train_df.empty and len(train_df) > 1:
            try:
                 # Calculate IS RMSE
                 is_mse = mean_squared_error(train_df[target_variable], train_df['Nowcast_Orig'])
                 is_rmse = np.sqrt(is_mse) # Changed from MAE
                 
                 changes_df_train = train_df.diff().dropna()
                 if not changes_df_train.empty:
                     correct_direction_train = (np.sign(changes_df_train['Nowcast_Orig']) == np.sign(changes_df_train[target_variable])) & (changes_df_train[target_variable] != 0)
                     non_zero_target_changes_train = (changes_df_train[target_variable] != 0).sum()
                     if non_zero_target_changes_train > 0:
                          is_hit_rate = correct_direction_train.sum() / non_zero_target_changes_train * 100
            except Exception as e_is:
                 print(f"    计算 IS 指标时出错: {e_is}")
        else:
            print(f"    训练期 (up to {train_end_date}) 数据不足 (< 2 点)，无法计算 IS RMSE/Hit Rate") # Updated print

        validation_df = aligned_df.loc[VALIDATION_START_DATE:VALIDATION_END_DATE]
        if not validation_df.empty and len(validation_df) > 1:
             try:
                 # Calculate OOS RMSE
                 oos_mse = mean_squared_error(validation_df[target_variable], validation_df['Nowcast_Orig'])
                 oos_rmse = np.sqrt(oos_mse) # Changed from MAE
                 
                 changes_df_val = validation_df.diff().dropna()
                 if not changes_df_val.empty:
                     correct_direction_val = (np.sign(changes_df_val['Nowcast_Orig']) == np.sign(changes_df_val[target_variable])) & (changes_df_val[target_variable] != 0)
                     non_zero_target_changes_val = (changes_df_val[target_variable] != 0).sum()
                     if non_zero_target_changes_val > 0:
                          oos_hit_rate = correct_direction_val.sum() / non_zero_target_changes_val * 100
             except Exception as e_oos:
                 print(f"    计算 OOS 指标时出错: {e_oos}")
        else:
            print(f"    验证期 ({VALIDATION_START_DATE} to {VALIDATION_END_DATE}) 数据不足 (< 2 点)，无法计算 OOS RMSE/Hit Rate") # Updated print
        
        # --- 计算组合指标 (使用 RMSE) ---
        combined_rmse = np.inf # Changed from combined_mae
        if np.isfinite(is_rmse) and np.isfinite(oos_rmse):
             combined_rmse = 0.5 * is_rmse + 0.5 * oos_rmse
        elif np.isfinite(is_rmse):
             combined_rmse = is_rmse
        elif np.isfinite(oos_rmse):
             combined_rmse = oos_rmse
        
        # (Hit Rate 部分不变)
        combined_hit_rate = -np.inf 
        valid_hit_rates = []
        if np.isfinite(is_hit_rate) and is_hit_rate > -np.inf: valid_hit_rates.append(is_hit_rate)
        if np.isfinite(oos_hit_rate) and oos_hit_rate > -np.inf: valid_hit_rates.append(oos_hit_rate)
        if valid_hit_rates: combined_hit_rate = np.mean(valid_hit_rates)
        # --- 结束组合指标计算 ---

        return is_rmse, oos_rmse, is_hit_rate, oos_hit_rate, False # 返回 RMSE

    except (np.linalg.LinAlgError, ValueError) as err:
        err_msg = str(err)
        is_svd_error = "svd did not converge" in err_msg.lower()
        # print(f"    评估时发生数值/逻辑错误 ({type(err).__name__} for vars={len(variables)}, k={k_factors}): {err_msg}") # 保持注释
        # if is_svd_error:
        #     print("SVD_CONVERGENCE_ERROR_COUNTED") # 保持注释
        return np.inf, np.inf, -np.inf, -np.inf, is_svd_error # 返回 RMSE=inf
    except Exception as e:
        # print(f"    评估时发生意外错误 ({type(e).__name__}): {e}") # 保持注释
        # traceback.print_exc() # 保持注释
        return np.inf, np.inf, -np.inf, -np.inf, False # 返回 RMSE=inf

# --- 新增：重新创建 analyze_and_save_final_results 函数 --- 
def analyze_and_save_final_results(
    run_output_dir: str, # 确保类型提示正确
    timestamp_str: str, # 确保类型提示正确
    all_data_full, 
    data_for_analysis, 
    target_variable,
    final_dfm_results,
    best_variables, 
    best_params, 
    var_type_map, 
    best_avg_hit_rate_tuning: float, 
    best_avg_mae_tuning: float,    # Note: Name kept as mae, but value is RMSE
    total_runtime_seconds
):
    """分析最终模型结果，计算指标，生成图表，并将所有内容保存到指定目录下的带时间戳的文件中。"""
    print("\n--- 开始最终结果分析与保存 --- ")

    # --- **再次确认**: 确保输出目录存在 ---
    if not run_output_dir or not isinstance(run_output_dir, str):
        print(f"错误：无效的 run_output_dir 参数: {run_output_dir}")
        return
    try:
        os.makedirs(run_output_dir, exist_ok=True) # 必须在最前面创建目录
        print(f"已确认/创建输出目录: {run_output_dir}")
    except OSError as e:
        print(f"错误：无法创建输出目录 '{run_output_dir}': {e}")
        return # 如果无法创建目录，则无法继续保存
    # --- **结束再次确认** ---

    # --- 动态生成输出文件完整路径 (再次确认基于 run_output_dir) --- 
    excel_output_file = os.path.join(run_output_dir, f"result_{timestamp_str}.xlsx")
    final_plot_file = os.path.join(run_output_dir, f"final_nowcast_comparison_{timestamp_str}.png")
    factor_plot_base_dir = run_output_dir # 因子图的基础目录
    heatmap_file = os.path.join(run_output_dir, f"factor_loadings_heatmap_{timestamp_str}.png")
    combined_factor_plot_file = os.path.join(run_output_dir, f"all_factors_timeseries_{timestamp_str}.png") # 新增：合并因子图路径
    # --- 结束动态路径生成 ---

    try:
        # --- 检查输入对象 --- 
        if not final_dfm_results or not hasattr(final_dfm_results, 'x_sm') or not hasattr(final_dfm_results, 'Lambda'):
            print("错误: analyze_and_save_final_results 缺少有效的 final_dfm_results 对象。")
            return
        if not data_for_analysis or \
           'final_data_processed' not in data_for_analysis or \
           'final_target_mean_rescale' not in data_for_analysis or \
           'final_target_std_rescale' not in data_for_analysis:
            print("错误: analyze_and_save_final_results 缺少有效的 data_for_analysis 对象。")
            return
        # --- 结束检查 --- 
        
        final_factors = final_dfm_results.x_sm
        final_loadings = final_dfm_results.Lambda
        final_data_processed = data_for_analysis['final_data_processed']
        target_mean = data_for_analysis['final_target_mean_rescale']
        target_std = data_for_analysis['final_target_std_rescale']
        final_k_factors = best_params.get('k_factors', 'N/A')
        use_log_yoy = USE_LOG_YOY_TRANSFORM 

        if isinstance(final_k_factors, str):
            print("错误: best_params 中 k_factors 无效。")
            return
            
        print(f"分析参数: k_factors={final_k_factors}, use_log_yoy(predictors)={use_log_yoy}")

        # --- 获取目标载荷 --- 
        lambda_df_final = None # 初始化
        lambda_target = None
        try:
            # 使用最终处理数据列顺序确定目标变量位置
            target_var_index_pos = final_data_processed.columns.get_loc(target_variable)
            lambda_target = final_loadings[target_var_index_pos, :]
            # 创建 lambda_df_final 供后续使用
            lambda_df_final = pd.DataFrame(final_loadings, index=final_data_processed.columns, columns=[f'Factor{i+1}' for i in range(final_k_factors)])
        except KeyError:
            print(f"错误: 无法在最终处理数据的列中找到目标变量 '{target_variable}' 的位置。列: {final_data_processed.columns.tolist()}")
            return 
        except IndexError:
             print(f"错误: 目标变量位置 ({target_var_index_pos}) 超出 final_loadings 数组的范围 (Shape: {final_loadings.shape})。")
             return
        # --- 结束获取目标载荷 --- 

        # --- 计算 Nowcast --- 
        if lambda_target is None or final_factors is None:
             print("错误: 无法计算 Nowcast，因为缺少因子或目标载荷。")
             return
        nowcast_standardized = final_factors.to_numpy() @ lambda_target
        final_nowcast_orig = pd.Series(nowcast_standardized * target_std + target_mean, index=final_factors.index, name='Nowcast_Orig')
        print("最终反标准化 Nowcast 计算完成.")
        # --- 结束计算 Nowcast --- 
        
        # --- 计算最终指标 --- 
        original_target_series = all_data_full[target_variable].copy() 
        target_for_comparison = original_target_series.dropna()
        diff_label_for_metrics = " (原始水平)"
        print(f"目标序列用于比较 (尺度: {diff_label_for_metrics}) 准备完成。")
        common_index_final = final_nowcast_orig.index.intersection(target_for_comparison.index)
        final_oos_rmse = np.nan
        hit_rate_validation = np.nan
        hit_rate_train = np.nan 
        if common_index_final.empty:
            print("错误: 最终 Nowcast 和目标序列没有共同索引。无法计算最终指标。")
        else:
            aligned_df_final = pd.DataFrame({'Nowcast': final_nowcast_orig.loc[common_index_final], 'Target': target_for_comparison.loc[common_index_final]}).dropna()
            validation_df_final = aligned_df_final.loc[VALIDATION_START_DATE:VALIDATION_END_DATE]
            if not validation_df_final.empty and len(validation_df_final) > 0:
                final_oos_rmse = np.sqrt(mean_squared_error(validation_df_final['Target'], validation_df_final['Nowcast']))
                print(f"  最终 OOS RMSE (验证期): {final_oos_rmse:.6f}")
                changes_df_val = validation_df_final.diff().dropna()
                if not changes_df_val.empty and len(changes_df_val) > 0:
                    correct_direction_val = (np.sign(changes_df_val['Nowcast']) == np.sign(changes_df_val['Target'])) & (changes_df_val['Target'] != 0)
                    non_zero_target_changes_val = (changes_df_val['Target'] != 0).sum()
                    if non_zero_target_changes_val > 0:
                         hit_rate_validation = correct_direction_val.sum() / non_zero_target_changes_val * 100
                         print(f"  验证期 Hit Rate (%): {hit_rate_validation:.2f} (基于 {non_zero_target_changes_val} 个非零变化点)")
            train_df_final = aligned_df_final.loc[:TRAIN_END_DATE]
            if not train_df_final.empty and len(train_df_final) > 1:
                 changes_df_train = train_df_final.diff().dropna()
                 if not changes_df_train.empty and len(changes_df_train) > 0:
                     correct_direction_train = (np.sign(changes_df_train['Nowcast']) == np.sign(changes_df_train['Target'])) & (changes_df_train['Target'] != 0)
                     non_zero_target_changes_train = (changes_df_train['Target'] != 0).sum()
                     if non_zero_target_changes_train > 0:
                          hit_rate_train = correct_direction_train.sum() / non_zero_target_changes_train * 100
                          print(f"  训练期 Hit Rate (%): {hit_rate_train:.2f} (基于 {non_zero_target_changes_train} 个非零变化点)")
        # --- 结束计算最终指标 --- 
        
        interpretation_text = (
            f"最终分析总结 (Run: {timestamp_str}):\n" # Add timestamp to summary
            f"- 最终选择变量数: {len(best_variables)}\n"
            f"- 最佳调优参数: 因子数={final_k_factors}\n"
            f"- 预测变量对数同比转换: {'启用' if use_log_yoy else '禁用'}\n"
            f"- 最佳平均胜率 (调优): {best_avg_hit_rate_tuning:.2f}%\n"
            f"- 对应平均 RMSE (调优): {best_avg_mae_tuning:.6f}\n"
            f"- 最终训练期 Hit Rate (%){diff_label_for_metrics}: {hit_rate_train:.2f}\n"
            f"- 最终验证期 RMSE{diff_label_for_metrics}: {final_oos_rmse:.6f}\n"
            f"- 最终验证期 Hit Rate (%){diff_label_for_metrics}: {hit_rate_validation:.2f}\n"
            f"- 总运行时间: {total_runtime_seconds / 60:.2f} 分钟\n"
            f"\n注: \n"
            f" - 对数同比转换 (Log(Yt) - Log(Yt-52)) {'仅应用于预测变量' if use_log_yoy else '已禁用'}。\n"
            f" - RMSE 和 Hit Rate 报告基于 {diff_label_for_metrics.strip()} 尺度。" 
        )

        # --- 写入 Excel 文件 --- 
        print(f"将结果写入 Excel 文件: {excel_output_file}...") # 确认使用正确的 excel_output_file 变量
        try:
            with pd.ExcelWriter(excel_output_file, engine='openpyxl') as writer:
                # Sheet: Summary_Overview
                try:
                    summary_df = pd.DataFrame({
                        'Parameter': [
                            'Final Variables Count', 'Best k_factors (Tuned)', 'Use LogYoY Transform (Predictors)',
                            'Best Avg Hit Rate (Tuning %)', 'Corresponding Avg RMSE (Tuning)',
                            f'Hit Rate (Train %){diff_label_for_metrics}', f"Final OOS RMSE{diff_label_for_metrics}",
                            f'Hit Rate (Validation %){diff_label_for_metrics}', 'Total Runtime (s)'],
                        'Value': [
                            len(best_variables), final_k_factors, use_log_yoy, 
                            f"{best_avg_hit_rate_tuning:.2f}" if pd.notna(best_avg_hit_rate_tuning) else "N/A",
                            f"{best_avg_mae_tuning:.6f}" if pd.notna(best_avg_mae_tuning) else "N/A",
                            f"{hit_rate_train:.2f}" if pd.notna(hit_rate_train) else "N/A",
                            f"{final_oos_rmse:.6f}" if pd.notna(final_oos_rmse) else "N/A",
                            f"{hit_rate_validation:.2f}" if pd.notna(hit_rate_validation) else "N/A",
                            f"{total_runtime_seconds:.2f}"]})
                    summary_df['Value'] = summary_df['Value'].astype(str) 
                    summary_df.to_excel(writer, sheet_name='Summary_Overview', index=False) 
                except Exception as e: print(f"写入 Summary_Overview 时出错: {e}")

                # Sheet: Analysis_Text
                try:
                    analysis_text_df = pd.DataFrame({'Analysis': [interpretation_text]})
                    analysis_text_df.to_excel(writer, sheet_name='Analysis_Text', index=False)
                except Exception as e: print(f"写入 Analysis_Text 时出错: {e}")

                # Sheet: Final_Factor_Loadings
                if lambda_df_final is not None:
                    try:
                        lambda_df_final.to_excel(writer, sheet_name='Final_Factor_Loadings', index=True)
                    except Exception as e: print(f"写入 Final_Factor_Loadings 时出错: {e}")
                else: print("警告: 无法写入 Final_Factor_Loadings，因为 lambda_df_final 未定义或为空。")
                
                # Sheet: Final_Selected_Variables
                try:
                    vars_df = pd.DataFrame(best_variables, columns=['Variable Name'])
                    cleaned_var_names_series = vars_df['Variable Name'].astype(str).str.strip()
                    normalized_lowercase_keys = cleaned_var_names_series.apply(lambda x: unicodedata.normalize('NFKC', x).strip().lower())
                    vars_df['Variable Type'] = normalized_lowercase_keys.map(var_type_map).fillna('Unknown')
                    vars_df.to_excel(writer, sheet_name='Final_Selected_Variables', index=False)
                except Exception as e: print(f"写入 Final_Selected_Variables 时出错: {e}")

                # Sheet: Selected_Vars_Level_or_LogYoY (or specific name)
                selected_data_sheet_name = "Selected_Vars_Original_Level" # Default if not use_log_yoy
                try:
                    data_to_save_sel = pd.DataFrame() # Initialize
                    if use_log_yoy:
                         predictors_final = [v for v in best_variables if v != target_variable]
                         if predictors_final:
                             logyoy_predictors = apply_log_yoy_transform(all_data_full[predictors_final])
                             data_to_save_sel = pd.concat([logyoy_predictors, all_data_full[target_variable]], axis=1)
                             data_to_save_sel = data_to_save_sel.reindex(columns=best_variables) 
                             selected_data_sheet_name = "Selected_Vars_LogYoY"
                    else:
                         data_to_save_sel = all_data_full[best_variables].copy()
                    
                    if not data_to_save_sel.empty:
                         data_to_save_sel = data_to_save_sel.replace([np.inf, -np.inf], np.nan).fillna('N/A')
                         if isinstance(data_to_save_sel.index, pd.DatetimeIndex):
                              data_to_save_sel.index = data_to_save_sel.index.strftime('%Y-%m-%d')
                         else: data_to_save_sel.index = data_to_save_sel.index.astype(str)
                         data_to_save_sel.columns = data_to_save_sel.columns.astype(str)
                         data_to_save_sel = data_to_save_sel.astype(str).reset_index()
                         data_to_save_sel.to_excel(writer, sheet_name=selected_data_sheet_name, index=False)
                    else: print(f"警告: 没有数据可保存到 '{selected_data_sheet_name}' sheet。")
                except Exception as e: print(f"写入 {selected_data_sheet_name} 时出错: {e}")

                # Sheet: Final_Factors
                try:
                    if final_factors is not None and not final_factors.empty:
                        factors_to_save = final_factors.copy().fillna('N/A')
                        if isinstance(factors_to_save.index, pd.DatetimeIndex):
                            factors_to_save.index = factors_to_save.index.strftime('%Y-%m-%d')
                        else: factors_to_save.index = factors_to_save.index.astype(str)
                        factors_to_save.columns = factors_to_save.columns.astype(str)
                        factors_to_save.to_excel(writer, sheet_name='Final_Factors', index=True)
                    else: print(f"警告: 无法写入 Final_Factors，因为 final_factors 为空或 None。")
                except Exception as e: print(f"写入 Final_Factors 时出错: {e}")

                # Sheet: Nowcast_vs_Target
                try:
                    if not common_index_final.empty:
                        nowcast_compare_df = pd.DataFrame({
                            f'Target{diff_label_for_metrics}': target_for_comparison.loc[common_index_final],
                            f'Nowcast{diff_label_for_metrics}': final_nowcast_orig.loc[common_index_final]}).fillna('N/A')
                        if isinstance(nowcast_compare_df.index, pd.DatetimeIndex):
                            nowcast_compare_df.index = nowcast_compare_df.index.strftime('%Y-%m-%d')
                        else: nowcast_compare_df.index = nowcast_compare_df.index.astype(str)
                        nowcast_compare_df.columns = nowcast_compare_df.columns.astype(str)
                        nowcast_compare_df.to_excel(writer, sheet_name='Nowcast_vs_Target', index=True)
                    else: print(f"警告: 无法写入 Nowcast_vs_Target，因为最终 Nowcast 和目标序列没有共同索引。")
                except Exception as e: print(f"写入 Nowcast_vs_Target 时出错: {e}")

                # Sheet: Factor_Contribution_Analysis
                try:
                    # ... (Factor Contribution Calculation as before) ...
                    if 'contribution_df' in locals(): # Check if calculation was successful
                         contribution_df.to_excel(writer, sheet_name='Factor_Contribution_Analysis', index=False, float_format="%.4f")
                         print(f"  因子贡献度分析已写入 Sheet: 'Factor_Contribution_Analysis'")
                except Exception as e_contrib: print(f"写入 Factor_Contribution_Analysis 时出错: {e_contrib}")

                # Sheet: Factor_Interpretation (修改解释逻辑)
                try:
                    factor_interpretations = []
                    if lambda_df_final is not None and not lambda_df_final.empty:
                        print(f"  生成因子解释...")
                        num_top_vars = 5 # 定义显示多少个最高载荷变量
                        for factor_col in lambda_df_final.columns:
                            loadings = lambda_df_final[factor_col].sort_values(ascending=False)
                            
                            top_positive = loadings.head(num_top_vars)
                            top_negative = loadings.tail(num_top_vars).sort_values() # 确保负值最低的在前
                            
                            interpretation = f"因子 {factor_col}:\n"
                            interpretation += f"  强正载荷变量 (Top {num_top_vars}):\n"
                            pos_types = []
                            for var, loading in top_positive.items():
                                lookup_key = unicodedata.normalize('NFKC', str(var)).strip().lower()
                                var_type = var_type_map.get(lookup_key, "未知")
                                interpretation += f"    - {var} ({var_type}): {loading:.3f}\n"
                                pos_types.append(var_type)
                                
                            interpretation += f"  强负载荷变量 (Top {num_top_vars}):\n"
                            neg_types = []
                            for var, loading in top_negative.items():
                                lookup_key = unicodedata.normalize('NFKC', str(var)).strip().lower()
                                var_type = var_type_map.get(lookup_key, "未知")
                                interpretation += f"    - {var} ({var_type}): {loading:.3f}\n"
                                neg_types.append(var_type)
                                
                            # --- 尝试生成经济解释 --- 
                            all_types = pos_types + neg_types
                            from collections import Counter
                            type_counts = Counter(t for t in all_types if t != "未知")
                            most_common_type, common_count = type_counts.most_common(1)[0] if type_counts else ("未知", 0)
                            
                            economic_meaning = ""
                            if common_count >= num_top_vars * 0.6: # 如果某一类型占比超过 60%
                                if most_common_type == '价格':
                                    economic_meaning = "可能主要反映了相关领域的价格水平或通胀压力。"
                                elif most_common_type in ['工业', '生产', '能源']:
                                    economic_meaning = "可能主要代表了工业生产活动或能源消耗的强度。"
                                elif most_common_type in ['景气度', '预期', '情绪']:
                                    economic_meaning = "可能主要捕捉了市场情绪、信心或经济景气预期。"
                                elif most_common_type == '金融':
                                    economic_meaning = "可能主要反映了金融市场状况、利率或流动性。"
                                elif most_common_type == '外贸':
                                    economic_meaning = "可能与进出口活动或外部需求相关。"
                                else:
                                    economic_meaning = f"可能主要与 '{most_common_type}' 类型的指标相关。"
                            else:
                                 economic_meaning = "可能代表了多种经济活动或因素的综合影响。"
                            # --- 结束经济解释生成 ---
                            
                            interpretation += f"  可能的经济含义: {economic_meaning}\n"
                            factor_interpretations.append(interpretation)
                            
                        interpret_df = pd.DataFrame({'Factor Interpretation': factor_interpretations})
                        interpret_df.to_excel(writer, sheet_name='Factor_Interpretation', index=False)
                        print(f"  因子解释已写入 Sheet: 'Factor_Interpretation'")
                    else: print(f"  警告: 无法生成因子解释，因为最终载荷矩阵 ('lambda_df_final') 不可用或为空。")
                except Exception as e_interpret: print(f"写入 Factor_Interpretation 时出错: {e_interpret}")

                # Sheet: Aligned_Original_Data
                try:
                    data_to_save_aligned_orig = all_data_full.copy()
                    # ... (Formatting as before) ...
                    data_to_save_aligned_orig = data_to_save_aligned_orig.reset_index()
                    data_to_save_aligned_orig.to_excel(writer, sheet_name='Aligned_Original_Data', index=False)
                    print(f"  对齐的原始数据已写入 Sheet: 'Aligned_Original_Data'")
                except Exception as e_aligned_orig: print(f"写入 Aligned_Original_Data 时出错: {e_aligned_orig}")

                # Sheet: Data_Input_to_DFM
                try:
                    data_to_save_dfm_input = final_data_processed.copy()
                    # ... (Formatting as before) ...
                    data_to_save_dfm_input = data_to_save_dfm_input.reset_index()
                    data_to_save_dfm_input.to_excel(writer, sheet_name='Data_Input_to_DFM', index=False)
                    print(f"  进入DFM模型的数据已写入 Sheet: 'Data_Input_to_DFM'")
                except Exception as e_dfm_input: print(f"写入 Data_Input_to_DFM 时出错: {e_dfm_input}")

            print("Excel 文件写入尝试完成。")
        except Exception as e_writer: 
            print(f"创建或写入 Excel 文件 '{excel_output_file}' 时发生严重错误: {e_writer}")
        # --- 结束写入 Excel --- 
        
        # --- 生成 Nowcast 对比图 --- 
        if 'target_for_comparison' in locals() and not target_for_comparison.empty:
            print(f"调用 plot_final_nowcast, 保存到: {final_plot_file}") # 确认传入 plot_final_nowcast 的路径正确
            plot_final_nowcast(
                final_nowcast_series=final_nowcast_orig, 
                target_for_plot=target_for_comparison, 
                validation_start=VALIDATION_START_DATE, 
                validation_end=VALIDATION_END_DATE, 
                title=f'最终 DFM Nowcast vs 观测值 ({diff_label_for_metrics.strip(" ()")}) [Run: {timestamp_str}]', 
                filename=final_plot_file, # 确认传递的是基于 run_output_dir 的路径
                use_log_yoy=use_log_yoy 
            )
        else:
            print("警告: 无法生成最终绘图，因为用于比较的目标序列为空或未定义。")
        # --- 结束生成 Nowcast 图 --- 

        # --- 生成因子载荷热力图 ---
        print("\n生成因子载荷热力图...")
        if lambda_df_final is not None and not lambda_df_final.empty:
            try:
                plt.figure(figsize=(max(10, lambda_df_final.shape[1] * 1.2), # 宽度适应因子数量
                                   max(12, lambda_df_final.shape[0] * 0.3))) # 高度适应变量数量
                sns.heatmap(lambda_df_final, 
                            annot=True,       # 显示数值
                            fmt=".2f",        # 格式化数值为两位小数
                            cmap='RdBu_r',    # 使用红-白-蓝颜色映射 (反转使正红负蓝)
                            center=0,         # 将颜色中心设为 0
                            linewidths=.5,    # 添加单元格间线条
                            linecolor='lightgray', # 线条颜色
                            cbar_kws={'shrink': .5}) # 调整颜色条大小
                plt.title(f'因子载荷热力图 (Factor Loadings) [Run: {timestamp_str}]', fontsize=14)
                plt.xlabel('因子 (Factors)', fontsize=12)
                plt.ylabel('变量 (Variables)', fontsize=12)
                plt.xticks(rotation=45, ha='right')
                plt.yticks(rotation=0)
                plt.tight_layout(rect=[0, 0, 1, 0.97]) # 调整布局以防标题重叠
                print(f"保存因子载荷热力图到: {heatmap_file}") # 确认保存路径正确
                plt.savefig(heatmap_file)
                plt.close()
                print(f"  因子载荷热力图已保存到: {heatmap_file}")
            except Exception as e_heatmap:
                print(f"  生成或保存因子载荷热力图时出错: {e_heatmap}")
                plt.close() # 确保关闭图形
        else:
            print("警告: 无法生成因子载荷热力图，因为最终载荷矩阵 (lambda_df_final) 不可用或为空。")
        # --- 结束生成热力图 ---
            
        # --- 修改: 生成并保存所有因子合并的时间序列图 --- 
        print("\n生成合并的因子时间序列图...")
        if final_factors is not None and not final_factors.empty:
            plt.figure(figsize=(14, 8)) # 创建一个图
            for factor_col in final_factors.columns:
                plt.plot(final_factors.index, final_factors[factor_col], label=factor_col, alpha=0.8) # 在同一个图上绘制
            
            plt.title(f'所有因子时间序列 [Run: {timestamp_str}]') 
            plt.xlabel('日期')
            plt.ylabel('因子值 (标准化)')
            plt.legend(loc='best') # 添加图例
            plt.grid(True, linestyle='--', alpha=0.6)
            plt.tight_layout()
            
            try:
                print(f"[DEBUG] Saving combined factors plot to: '{combined_factor_plot_file}'") # 调试打印
                plt.savefig(combined_factor_plot_file) # 保存合并后的图
                plt.close() # 关闭图形
                print(f"  合并的因子时间序列图已保存到: {combined_factor_plot_file}")
            except Exception as e_plot_combined_factors:
                print(f"  保存合并的因子时间序列图时出错: {e_plot_combined_factors}")
                plt.close() # 确保关闭图形
        else:
            print("警告: 最终因子 (final_factors) 为空或 None，无法生成合并的因子时间序列图。")
        # --- 结束修改因子图 --- 

        print("分析和保存完成。")

    except Exception as e_analyze:
        print(f"在 analyze_and_save_final_results 函数中发生意外错误: {e_analyze}")
        import traceback
        traceback.print_exc()

# --- 修改绘图函数以移除 use_differencing --- 
def plot_final_nowcast(final_nowcast_series, target_for_plot, validation_start, validation_end, title, filename, use_log_yoy):
    print("\n生成最终 Nowcasting 图 (原始水平)...")
    try:
        common_index_plot = final_nowcast_series.index.intersection(target_for_plot.index)
        if common_index_plot.empty:
            print("错误：无法对齐最终 Nowcast 和目标序列进行绘图。")
            return

        nowcast_col_name = 'Nowcast_Orig'
        target_col_name = target_for_plot.name if target_for_plot.name is not None else 'Actual'
        if target_col_name == nowcast_col_name:
            target_col_name = 'Observed_Value'

        plot_df = pd.concat([
            final_nowcast_series.loc[common_index_plot].rename(nowcast_col_name),
            target_for_plot.loc[common_index_plot].rename(target_col_name)
        ], axis=1)
        plot_df = plot_df.dropna(subset=[target_col_name])

        # --- 新增: 绘图时移除1月/2月实际值 ---
        plot_df_masked = plot_df.copy()
        month_indices_plot = plot_df_masked.index.month
        plot_df_masked.loc[(month_indices_plot == 1) | (month_indices_plot == 2), target_col_name] = np.nan 
        # --- 结束新增 ---

        if not plot_df.empty:
            plt.figure(figsize=(14, 7))
            
            nowcast_label = '周度 Nowcast (原始水平)'
            actual_label = '观测值 (原始水平)'
            ylabel = '值 (原始水平)'

            plt.plot(plot_df_masked.index, plot_df_masked[nowcast_col_name], label=nowcast_label, linestyle='-', alpha=0.8, color='blue')
            plt.plot(plot_df_masked.index, plot_df_masked[target_col_name], label=actual_label, marker='o', linestyle='None', markersize=4, color='red')
            
            try:
                plt.axvspan(pd.to_datetime(validation_start), pd.to_datetime(validation_end), color='yellow', alpha=0.2, label='验证期')
            except Exception as date_err:
                print(f"警告：标记验证期时出错 - {date_err}")

            plt.title(title)
            plt.xlabel('日期')
            plt.ylabel(ylabel)
            plt.legend() 
            plt.grid(True, linestyle='--', alpha=0.6)
            plt.tight_layout()
            print(f"在 plot_final_nowcast 中尝试保存到: {filename}") # 确认传入的 filename 正确
            plt.savefig(filename)
            plt.close()
            print(f"最终 Nowcasting 图已保存到: {filename}")
        else:
                print("错误：对齐后用于绘图的数据为空。") 
    except Exception as e:
        print(f"生成或保存最终 Nowcasting 图时出错: {e}")

# --- 主逻辑 --- 
def run_tuning():
    script_start_time = time.time()
    total_evaluations = 0
    svd_error_count = 0
    transform_info = "启用" if USE_LOG_YOY_TRANSFORM else "禁用"
    
    # --- 新增: 生成时间戳和运行目录 ---
    timestamp_str = time.strftime("%Y%m%d_%H%M%S")
    base_output_dir = os.path.join('dym_estimate', 'dfm_result') # 基础输出目录
    run_output_dir = os.path.join(base_output_dir, f'run_{timestamp_str}') # 带时间戳的运行目录
    log_filename = f"调优日志_{timestamp_str}.txt" # 带时间戳的日志文件名
    log_file_path = os.path.join(run_output_dir, log_filename) # 日志文件的完整路径
    # --- 结束新增 ---
    
    print(f"--- 开始变量后向剔除与超参数调优 (优化目标: 平均胜率优先, RMSE其次; k_factors 动态范围, 对数同比转换: {transform_info}) ---")
    print(f"本次运行结果将保存在: {run_output_dir}") # 打印本次运行目录
    
    try:
        # --- 修改: 确保目录存在并使用带时间戳的日志文件 --- 
        os.makedirs(run_output_dir, exist_ok=True) # 确保运行目录存在
        log_file = open(log_file_path, 'w', encoding='utf-8') # 使用完整路径打开日志
        log_file.write(f"--- 开始详细调优日志 (Run: {timestamp_str}) ---\n")
        log_file.write(f"输出目录: {run_output_dir}\n")
        log_file.write(f"配置: 对数同比转换={transform_info}; 优化目标=平均胜率优先, RMSE其次\n")
        # --- 结束修改 ---
        print(f"详细调优日志将写入: {log_file_path}")
    except IOError as e:
        print(f"错误: 无法打开日志文件 {log_file_path} 进行写入: {e}")
        log_file = None
        # 如果日志无法打开，可能后续保存也会失败，但先继续尝试

    all_evaluation_results = []

    print("\n--- 调用数据准备模块 (自动发现 Sheets) --- ")
    all_data_aligned_weekly = prepare_data(excel_path=EXCEL_DATA_FILE,
                                             target_freq=TARGET_FREQ,
                                             target_sheet_name=TARGET_SHEET_NAME,
                                             target_variable_name=TARGET_VARIABLE)

    if all_data_aligned_weekly is None:
        print("错误: 数据准备失败，无法继续调优。")
        if log_file: log_file.close()
        sys.exit(1)
    
    print(f"\n[EARLY CHECK 1] prepare_data completed. Output shape: {all_data_aligned_weekly.shape}")
    if isinstance(all_data_aligned_weekly.index, pd.DatetimeIndex):
        print(f"[EARLY CHECK 1] prepare_data index freq: {all_data_aligned_weekly.index.freqstr}")
    else:
        print(f"[EARLY CHECK 1] prepare_data index type: {type(all_data_aligned_weekly.index)}")
    print("-"*30)

    print(f"数据准备模块成功返回处理后的数据. Shape: {all_data_aligned_weekly.shape}")
    
    all_variable_names = all_data_aligned_weekly.columns.tolist()
    if TARGET_VARIABLE not in all_variable_names:
        print(f"错误: 目标变量 {TARGET_VARIABLE} 不在合并后的数据中。")
        sys.exit(1)
    
    initial_variables = sorted(all_variable_names)
    print(f"\n初始变量组 ({len(initial_variables)}): {initial_variables}")
    print("-"*30)

    print("正在从 Excel 文件加载指标类型映射 (预期在第一个 Sheet)...")
    try:
        excel_file_obj = pd.ExcelFile(EXCEL_DATA_FILE)
        first_sheet_name = excel_file_obj.sheet_names[0]
        print(f"  将从第一个 Sheet ('{first_sheet_name}') 加载类型映射。")
        indicator_sheet = pd.read_excel(excel_file_obj, sheet_name=first_sheet_name) 
    except Exception as e:
        print(f"错误: 无法从 Excel '{EXCEL_DATA_FILE}' 的第一个 Sheet 加载类型映射: {e}")
        if log_file: log_file.close()
        sys.exit(1)

    col_indicator_name = '高频指标' 
    col_type_name = '类型'

    if col_indicator_name not in indicator_sheet.columns or col_type_name not in indicator_sheet.columns:
        print(f"错误: 在 Excel 文件 '{EXCEL_DATA_FILE}' 的第一个 sheet 中未找到所需的列 '{col_indicator_name}' 或 '{col_type_name}'。")
        print(f"找到的列名: {indicator_sheet.columns.tolist()}")
        sys.exit(1)

    indicator_sheet.columns = indicator_sheet.columns.str.strip()
    col_indicator_name = col_indicator_name.strip()
    col_type_name = col_type_name.strip()

    # --- 修改: 将 key 转为小写并进行 NFKC 规范化 ---
    var_type_map_temp = pd.Series(
        indicator_sheet[col_type_name].astype(str).str.strip().values,
        index=indicator_sheet[col_indicator_name].astype(str).str.strip()
    ).to_dict()
    # 创建新的字典，键是小写且规范化的
    var_type_map = {unicodedata.normalize('NFKC', str(k)).strip().lower(): str(v).strip() 
                    for k, v in var_type_map_temp.items() 
                    if pd.notna(k) and str(k).lower() != 'nan' and pd.notna(v) and str(v).lower() != 'nan'}
    # --- 结束修改 ---

    print(f"\n[EARLY CHECK 2] Indicator mapping completed. Map size: {len(var_type_map)}")
    # --- 新增 DEBUG: 打印部分 map 键 ---
    print("  [DEBUG] Sample var_type_map keys (lowercase, stripped):")
    keys_sample = list(var_type_map.keys())[:20] # 取前 20 个键
    for key in keys_sample:
        print(f"    - '{key}'")
    if len(var_type_map) > 20:
        print("    ...")
    # --- 结束 DEBUG ---
    print("-"*30)

    print(f"成功创建 {len(var_type_map)} 个指标的类型映射。")

    # --- 原有代码被跳过 --- 
    print("\n--- 确定初始变量块和动态因子数范围 ---")
    # ... [后续所有 DFM 调优和分析代码将不会执行] ...

    print("\n--- 确定初始变量块和动态因子数范围 ---")
    initial_predictors = [v for v in initial_variables if v != TARGET_VARIABLE]
    temp_blocks_init = {}
    for var in initial_predictors:
        # --- 修改: 对查找键应用规范化和小写 --- 
        lookup_key = unicodedata.normalize('NFKC', str(var)).strip().lower()
        var_type = var_type_map.get(lookup_key, "_未知类型_") 
        # var_type = var_type_map.get(var, "_未知类型_") # 旧的查找方式
        # --- 结束修改 ---
        if var_type not in temp_blocks_init:
            temp_blocks_init[var_type] = []
        temp_blocks_init[var_type].append(var)
    
    initial_blocks = {}
    merged_small_block_vars_init = []
    small_block_names_merged_init = []
    for block_name, block_vars in temp_blocks_init.items():
        if len(block_vars) < 3:
            merged_small_block_vars_init.extend(block_vars)
            small_block_names_merged_init.append(block_name)
        else:
            initial_blocks[block_name] = block_vars
    if merged_small_block_vars_init:
        initial_blocks["其他"] = merged_small_block_vars_init
        print(f"根据初始变量，已将 {len(small_block_names_merged_init)} 个小块合并到 '其他' (共 {len(merged_small_block_vars_init)} 变量)。")

    max_k_factors = len(initial_blocks)
    if max_k_factors == 0 and initial_predictors: 
        print("警告: 所有初始预测变量块均少于3个变量，合并后仅有'其他'块。因子数范围将基于此。")
        max_k_factors = 1 
    elif max_k_factors == 0: 
         print("警告: 没有初始预测变量块，因子数范围将设为 [1]。")
         max_k_factors = 1 
         
    K_FACTORS_RANGE = list(range(1, max_k_factors + 1))
    print(f"动态确定的因子数范围 (基于初始块数 {max_k_factors}): {K_FACTORS_RANGE}")

    HYPERPARAMS_TO_TUNE = [] 
    for k in K_FACTORS_RANGE:
        HYPERPARAMS_TO_TUNE.append({'k_factors': k})
    print(f"构建了 {len(HYPERPARAMS_TO_TUNE)} 个超参数组合进行测试 (仅调整 k_factors)。")

    print(f"\n训练期结束: {TRAIN_END_DATE}")
    print(f"验证期: {VALIDATION_START_DATE} to {VALIDATION_END_DATE}")
    print(f"待优化的因子数范围 (动态确定): {K_FACTORS_RANGE}") 
    # --- 恢复打印的优化目标描述 ---
    print(f"优化目标: 最大化 (IS_HitRate + OOS_HitRate) / 2 (平均胜率优先), 然后最小化 (IS_RMSE + OOS_RMSE) / 2 (平均RMSE其次)") # 新描述
    # print(f"优化目标: 最小化 (IS_RMSE + OOS_RMSE) / 2 (平均RMSE优先), 然后最大化 OOS_HitRate (OOS胜率其次)") # 旧描述
    # --- 结束恢复 ---
    print(f"将使用最多 {MAX_WORKERS} 个进程进行并行计算。")
    print("-" * 30)

    print("\n--- 初始化: 评估所有预测变量 ---")
    initial_predictors = [v for v in all_data_aligned_weekly.columns if v != TARGET_VARIABLE]
    initial_variables = initial_predictors + [TARGET_VARIABLE]
    # --- 恢复: 初始化跟踪变量为胜率优先 ---
    best_overall_hit_rate = -np.inf
    best_overall_mae = np.inf 
    # best_overall_oos_hit_rate = -np.inf # 不再单独跟踪 OOS HR
    # --- 结束恢复 ---
    best_overall_params = None
    # best_overall_variables = initial_variables.copy() # 变量在找到第一个最佳后更新
    initial_best_found = False

    futures_initial_map = {}
    with concurrent.futures.ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
        for params in HYPERPARAMS_TO_TUNE:
            future = executor.submit(evaluate_dfm_params,
                                     initial_variables,
                                     all_data_aligned_weekly,
                                     TARGET_VARIABLE,
                                     params,
                                     var_type_map,
                                     VALIDATION_START_DATE,
                                     VALIDATION_END_DATE,
                                     TARGET_FREQ,
                                     TRAIN_END_DATE, 
                                     max_iter=N_ITER_FIXED
                                 )
            futures_initial_map[future] = params 

    print(f"提交 {len(futures_initial_map)} 个初始评估任务 (使用最多 {MAX_WORKERS} 进程)...")
    results = []

    for future in concurrent.futures.as_completed(futures_initial_map):
        params = futures_initial_map[future]
        total_evaluations += 1 # 计数
        try:
            is_rmse, oos_rmse, is_hit_rate, oos_hit_rate, is_svd_error = future.result()
            if is_svd_error:
                svd_error_count += 1 # 计数
            
            # --- 修改: 计算组合指标并按胜率优先比较 ---
            combined_rmse = np.inf
            if np.isfinite(is_rmse) and np.isfinite(oos_rmse): combined_rmse = 0.5 * is_rmse + 0.5 * oos_rmse
            elif np.isfinite(is_rmse): combined_rmse = is_rmse
            elif np.isfinite(oos_rmse): combined_rmse = oos_rmse
            
            combined_hit_rate = -np.inf
            valid_hit_rates = []
            if np.isfinite(is_hit_rate): valid_hit_rates.append(is_hit_rate)
            if np.isfinite(oos_hit_rate): valid_hit_rates.append(oos_hit_rate)
            if valid_hit_rates: combined_hit_rate = np.mean(valid_hit_rates)
                
            results.append({'variables': initial_variables, 'params': params, 
                            'combined_rmse': combined_rmse, 'combined_hit_rate': combined_hit_rate,
                            'is_rmse': is_rmse, 'oos_rmse': oos_rmse, 'is_hit_rate': is_hit_rate, 'oos_hit_rate': oos_hit_rate})

            if np.isfinite(combined_hit_rate) and np.isfinite(combined_rmse):
                current_score_tuple = (combined_hit_rate, -combined_rmse) # 胜率优先
                best_score_tuple = (best_overall_hit_rate, -best_overall_mae)

                if current_score_tuple > best_score_tuple:
                    best_overall_hit_rate = combined_hit_rate
                    best_overall_mae = combined_rmse
                    best_overall_params = params
                    best_overall_variables = initial_variables # 初始评估时变量不变
                    initial_best_found = True 
            # --- 结束修改 ---

        except Exception as e:
            print(f"初始化评估期间参数 {params} 运行出错: {e}")

    if not initial_best_found:
        print("错误: 初始化评估未能成功运行任何超参数组合。无法继续。")
        sys.exit(1)

    # --- 修改: 更新初始结果打印 --- 
    print(f"初始评估完成。最佳组合得分: 平均胜率={best_overall_hit_rate:.2f}%, 平均 RMSE={best_overall_mae:.6f} (参数: {best_overall_params})")
    print(f"初始变量数量: {len(best_overall_variables)}") # 使用 best_overall_variables
    if log_file:
         try:
             log_file.write("\n" + "-"*35 + "\n")
             log_file.write("--- 初始化: 所有变量评估结果 ---\n")
             log_file.write(f"初始变量组 ({len(best_overall_variables)}): {best_overall_variables}\n")
             log_file.write(f"最佳参数: {best_overall_params}\n")
             log_file.write(f"最佳平均胜率 (IS+OOS)/2: {best_overall_hit_rate:.2f}%\n") # 胜率优先
             log_file.write(f"对应的平均 RMSE (IS+OOS)/2: {best_overall_mae:.6f}\n")
             log_file.write("-"*35 + "\n")
         except Exception as log_e:
             print(f"写入初始评估日志时出错: {log_e}")
    # --- 结束修改 ---

    print("\n--- 开始分块向后变量剔除 (使用初始确定的块结构) ---") 
    blocks = initial_blocks 
    print(f"将对 {len(blocks)} 个块进行变量剔除:")
    for block_name, block_vars in blocks.items():
        print(f"  块 '{block_name}' ({len(block_vars)} 变量)")

    # --- 修改: 初始化当前最佳指标为胜率优先 --- 
    current_best_variables = best_overall_variables.copy()
    current_best_hit_rate = best_overall_hit_rate # 恢复
    current_best_mae = best_overall_mae
    # current_best_oos_hit_rate = best_overall_oos_hit_rate # 移除
    current_best_params = best_overall_params
    # --- 结束修改 ---

    for block_name, block_vars_list in tqdm(blocks.items(), desc="处理变量块", unit="block"):
        print(f"\n--- 处理块: '{block_name}' (初始 {len(block_vars_list)} 变量) ---")

        if len(block_vars_list) <= 2: 
            print(f"块 '{block_name}' 变量数 ({len(block_vars_list)}) <= 2，跳过剔除。")
            continue

        block_stable = False
        while not block_stable:
            # --- 修改: 初始化本轮最佳为胜率优先 ---
            best_score_tuple_this_iter = (-np.inf, -np.inf) # (hit_rate, -mae)
            # best_mae_this_iteration = np.inf # 移除
            # best_oos_hit_rate_this_iteration = -np.inf # 移除
            variable_to_remove = None
            params_for_best_removal = None
            # --- 结束修改 ---

            eligible_vars_in_block = [v for v in block_vars_list if v in current_best_variables and v != TARGET_VARIABLE]
            if not eligible_vars_in_block:
                print(f"块 '{block_name}' 中无更多可剔除变量。") # 如果确实没有可选变量了，仍然需要停止
                block_stable = True
                break

            print(f"  评估从块 '{block_name}' 移除 {len(eligible_vars_in_block)} 个候选变量...")
            futures_removal = []
            with concurrent.futures.ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
                for candidate_var_to_remove in eligible_vars_in_block:
                    temp_variables = [v for v in current_best_variables if v != candidate_var_to_remove]
                    if not temp_variables: continue

                    for params in HYPERPARAMS_TO_TUNE:
                        future = executor.submit(
                            evaluate_dfm_params,
                            variables=temp_variables,
                            full_data=all_data_aligned_weekly,
                            target_variable=TARGET_VARIABLE,
                            params=params,
                            var_type_map=var_type_map,
                            validation_start=VALIDATION_START_DATE,
                            validation_end=VALIDATION_END_DATE,
                            target_freq=TARGET_FREQ,
                            train_end_date=TRAIN_END_DATE, 
                            max_iter=N_ITER_FIXED
                        )
                        futures_removal.append({
                            'future': future,
                            'params': params,
                            'removed_var': candidate_var_to_remove,
                            'remaining_vars': temp_variables.copy()
                        })

            results_this_iteration = []
            if not futures_removal: continue
            for f_info in concurrent.futures.as_completed([f['future'] for f in futures_removal]):
                context = next(item for item in futures_removal if item['future'] == f_info)
                total_evaluations += 1 # 计数
                try:
                    is_rmse, oos_rmse, is_hit_rate, oos_hit_rate, is_svd_error = f_info.result()
                    if is_svd_error:
                        svd_error_count += 1 # 计数
                    
                    # --- 修改: 计算组合指标 --- 
                    combined_rmse_removal = np.inf
                    if np.isfinite(is_rmse) and np.isfinite(oos_rmse): combined_rmse_removal = 0.5 * is_rmse + 0.5 * oos_rmse
                    elif np.isfinite(is_rmse): combined_rmse_removal = is_rmse
                    elif np.isfinite(oos_rmse): combined_rmse_removal = oos_rmse
            
                    combined_hit_rate_removal = -np.inf
                    valid_hit_rates_rem = []
                    if np.isfinite(is_hit_rate): valid_hit_rates_rem.append(is_hit_rate)
                    if np.isfinite(oos_hit_rate): valid_hit_rates_rem.append(oos_hit_rate)
                    if valid_hit_rates_rem: combined_hit_rate_removal = np.mean(valid_hit_rates_rem)
                    # --- 结束修改 ---
                        
                    results_this_iteration.append({
                        'combined_rmse': combined_rmse_removal, # 修改
                        'combined_hit_rate': combined_hit_rate_removal, # 修改
                        'params': context['params'],
                        'removed_var': context['removed_var']
                    })
                except Exception as exc:
                    print(f"处理块 {block_name}, 尝试移除 {context['removed_var']} 时出错: {exc}")

            # --- 修改: 选择最佳移除操作 (胜率优先) ---
            best_removal_candidate = None
            # best_removal_score = (-np.inf, -np.inf) # 使用 best_score_tuple_this_iter

            for result in results_this_iteration:
                if np.isfinite(result['combined_hit_rate']) and np.isfinite(result['combined_rmse']):
                    current_removal_score_tuple = (result['combined_hit_rate'], -result['combined_rmse'])
                    if current_removal_score_tuple > best_score_tuple_this_iter:
                        best_score_tuple_this_iter = current_removal_score_tuple
                        best_removal_candidate = result
            # --- 结束修改 ---

            # --- 修改: 接受移除的条件 (胜率优先) ---
            # if best_removal_candidate and (best_mae_this_iteration < current_best_mae or (np.isclose(best_mae_this_iteration, current_best_mae) and best_oos_hit_rate_this_iteration > current_best_oos_hit_rate)):
            if best_removal_candidate and best_score_tuple_this_iter > (current_best_hit_rate, -current_best_mae):
                # 接受移除
                variable_to_remove = best_removal_candidate['removed_var']
                params_for_best_removal = best_removal_candidate['params']
                # 从 best_score_tuple_this_iter 更新全局最佳
                current_best_hit_rate = best_score_tuple_this_iter[0]
                current_best_mae = -best_score_tuple_this_iter[1]
                current_best_params = params_for_best_removal
                # 更新变量列表
                current_best_variables = next(item['remaining_vars'] for item in futures_removal if item['removed_var'] == variable_to_remove and item['params'] == params_for_best_removal)

                if variable_to_remove in block_vars_list:
                    block_vars_list.remove(variable_to_remove)
                
                # --- 修改: 更新打印和日志 (胜率优先) ---
                print(f"*** 块 '{block_name}' 找到改进: 移除 '{variable_to_remove}', 新最佳分数: HR={current_best_hit_rate:.2f}%, RMSE={current_best_mae:.6f}, Params: {current_best_params} ***")

                if log_file:
                    try:
                        log_file.write("\n" + "-"*35 + "\n")
                        log_file.write(f"--- 块 '{block_name}': 变量剔除结果 ---\n")
                        log_file.write(f"剔除变量: '{variable_to_remove}' (因其提升了综合得分 (HR优先))\n")
                        log_file.write(f"当前变量组 ({len(current_best_variables)}): {current_best_variables}\n")
                        log_file.write(f"最佳参数: {current_best_params}\n")
                        log_file.write(f"新平均胜率 (IS+OOS)/2: {current_best_hit_rate:.2f}%\n") # 胜率优先
                        log_file.write(f"对应平均 RMSE (IS+OOS)/2: {current_best_mae:.6f}\n")
                        log_file.write("-"*35 + "\n")
                    except Exception as log_e:
                        print(f"写入块 '{block_name}' 剔除日志时出错: {log_e}")
                # --- 结束修改 ---
            else:
                # --- 修改: 更新停止打印 (胜率优先) ---
                print(f"块 '{block_name}' 内无变量移除可进一步改进综合得分 (当前 HR={current_best_hit_rate:.2f}%, RMSE={current_best_mae:.6f})。此块完成。")
                block_stable = True
                if log_file and eligible_vars_in_block:
                     try:
                         log_file.write("\n" + "-"*35 + "\n")
                         log_file.write(f"--- 块 '{block_name}': 停止剔除 ---\n")
                         log_file.write(f"原因: 块内剩余变量移除无法改进当前最佳综合得分 (HR={current_best_hit_rate:.2f}%, RMSE={current_best_mae:.6f})。\n")
                         log_file.write("-"*35 + "\n")
                     except Exception as log_e:
                        print(f"写入块 '{block_name}' 停止日志时出错: {log_e}")
                # --- 结束修改 ---

    print("\n--- 所有块处理完毕 --- ")
    # --- 修改: 记录最终结果 (胜率优先) ---
    final_variables = current_best_variables.copy()
    final_params = current_best_params
    final_combined_hit_rate = current_best_hit_rate # 恢复
    final_combined_mae = current_best_mae
    # final_oos_hit_rate = current_best_oos_hit_rate # 移除
    print(f"最终变量数量: {len(final_variables)}")
    print(f"最终最佳平均胜率 (IS+OOS)/2: {final_combined_hit_rate:.2f}%") # 胜率优先
    print(f"对应的最终最佳平均 RMSE (IS+OOS)/2: {final_combined_mae:.6f}")
    print(f"最终最佳参数: {final_params}")
    # --- 结束修改 ---

    print("\n--- 使用最终选择的变量和参数重新运行 DFM 模型 ---")
    final_dfm_results_obj = None
    data_for_analysis = None 
    if final_params and final_variables:
        try:
            if 'k_factors' not in final_params:
                 raise ValueError(f"最终最佳参数字典缺少 k_factors: {final_params}")
            final_k_factors = final_params['k_factors']
            final_p_shocks = final_k_factors
            
            print(f"最终运行参数(来自调优): 因子数={final_k_factors}, 冲击数={final_p_shocks}, 对数同比(预测变量)={USE_LOG_YOY_TRANSFORM}")

            print("准备最终运行的数据...")
            final_predictors = [v for v in final_variables if v != TARGET_VARIABLE]
            final_target = TARGET_VARIABLE

            final_data_numeric = all_data_aligned_weekly[final_variables].copy()
            final_data_numeric = final_data_numeric.apply(pd.to_numeric, errors='coerce')

            # --- 修改: 使用变量级转换代替全局 LogYoY ---
            final_data_processed_transformed, final_transform_details = apply_stationarity_transforms(
                final_data_numeric, final_target
            )
            # --- 结束修改 ---

            # --- 原 LogYoY 逻辑 (注释掉) ---
            # final_data_processed = final_data_numeric.copy()
            # if USE_LOG_YOY_TRANSFORM and final_predictors:
            #     print("  对预测变量应用对数同比转换...")
            #     predictors_logyoy_final = apply_log_yoy_transform(final_data_processed[final_predictors])
            #     final_data_processed = pd.concat([predictors_logyoy_final, final_data_processed[final_target]], axis=1)
            #     final_data_processed = final_data_processed[final_variables]
            #     if not final_data_processed.empty:
            #         print("  移除 52 个前导 NaN 行 (LogYoY...")
            #         if 52 >= len(final_data_processed):
            #               raise ValueError(f"最终运行数据准备错误: 数据行数 ({len(final_data_processed)}) 不足以移除 LogYoY 的前导 NaN (52)")
            #         final_data_processed = final_data_processed.iloc[52:]
            # --- 结束原 LogYoY 逻辑 ---

            # 使用转换后的数据
            final_data_processed = final_data_processed_transformed

            if final_data_processed.empty:
                 # raise ValueError(f"最终 DFM 运行数据准备错误: 在应用转换 (LogYoY={USE_LOG_YOY_TRANSFORM}) 和移除前导 NaN 后数据为空。")
                 raise ValueError(f"最终 DFM 运行数据准备错误: 在应用变量级平稳性转换后数据为空。")

            print(f"  标准化处理后的数据 (Shape: {final_data_processed.shape})...")
            final_mean = final_data_processed.mean(skipna=True)
            final_std = final_data_processed.std(skipna=True)
            final_std[final_std == 0] = 1.0
            final_data_std_for_dfm = (final_data_processed - final_mean) / final_std

            # --- 修改: 在最终模型训练时忽略 1月/2月 目标值 ---
            final_data_std_masked_for_fit = final_data_std_for_dfm.copy()
            month_indices_final = final_data_std_masked_for_fit.index.month
            final_data_std_masked_for_fit.loc[(month_indices_final == 1) | (month_indices_final == 2), final_target] = np.nan
            # --- 结束修改 ---

            if final_target in final_mean.index and final_target in final_std.index:
                final_target_mean_rescale = final_mean[final_target]
                final_target_std_rescale = final_std[final_target]
                if pd.isna(final_target_mean_rescale) or pd.isna(final_target_std_rescale):
                      raise ValueError("最终运行数据准备错误: 无法获取目标变量的均值/标准差进行反标准化。")
            else:
                 raise ValueError("最终运行数据准备错误: 标准化后找不到目标变量的均值/标准差。")
            
            data_for_analysis = {
                'final_data_processed': final_data_processed.copy(), 
                'final_target_mean_rescale': final_target_mean_rescale,
                'final_target_std_rescale': final_target_std_rescale
            }

            final_dfm_results_obj = DFM_EMalgo(
                observation=final_data_std_masked_for_fit, # <--- 使用带掩码的数据进行最终拟合
                n_factors=final_k_factors, 
                n_shocks=final_p_shocks,   
                n_iter=N_ITER_FIXED,
                error='False'
            )
            print("最终 DFM 模型运行成功。")

            print("\n[DEBUG POST-FINAL DFM CALL]")
            if hasattr(final_dfm_results_obj, 'x_sm'):
                final_x_sm_check = final_dfm_results_obj.x_sm
                if final_x_sm_check is None:
                     print("  final_dfm_results_obj.x_sm is None!")
                elif isinstance(final_x_sm_check, pd.DataFrame):
                     print(f"  final_x_sm_check (DataFrame) shape: {final_x_sm_check.shape}")
                     print(f"  final_x_sm_check (DataFrame) NaNs: {final_x_sm_check.isna().sum().sum()}")
                     print(f"  final_x_sm_check (DataFrame) head:\n{final_x_sm_check.head()}")
                elif isinstance(final_x_sm_check, np.ndarray):
                     print(f"  final_x_sm_check (ndarray) shape: {final_x_sm_check.shape}")
                     print(f"  final_x_sm_check (ndarray) NaNs: {np.isnan(final_x_sm_check).sum()}")
                else:
                     print(f"  final_x_sm_check type: {type(final_x_sm_check)}")
                     print(f"  final_x_sm_check value: {final_x_sm_check}")
            else:
                print("  final_dfm_results_obj has no attribute 'x_sm'!")

        except Exception as e:
            print(f"使用最终变量重新运行 DFM 时出错: {e}")
            import traceback
            traceback.print_exc() 
            final_dfm_results_obj = None 
            
    script_end_time = time.time() # 移到这里，确保即使 DFM 运行失败也能计算时间
    total_runtime_seconds = script_end_time - script_start_time
        
    if final_dfm_results_obj is not None and data_for_analysis is not None: 
        try:
            print("\n调用最终结果分析与保存函数...")
            # --- **再次确认**: 传递正确的 run_output_dir 和 timestamp_str --- 
            print(f"[DEBUG] Passing to analyze_and_save_final_results:")
            print(f"  run_output_dir = '{run_output_dir}'") # 打印传入的目录
            print(f"  timestamp_str = '{timestamp_str}'") # 打印传入的时间戳
            analyze_and_save_final_results(
                run_output_dir=run_output_dir,       # 确保传递的是这里定义的变量
                timestamp_str=timestamp_str,         # 确保传递的是这里定义的变量
                all_data_full=all_data_aligned_weekly, 
                data_for_analysis=data_for_analysis, 
                target_variable=TARGET_VARIABLE,
                final_dfm_results=final_dfm_results_obj,
                best_variables=final_variables, 
                best_params=final_params,
                var_type_map=var_type_map,
                best_avg_hit_rate_tuning=final_combined_hit_rate, # 新：胜率优先
                best_avg_mae_tuning=final_combined_mae,          # 新：RMSE (变量名未改)
                total_runtime_seconds=total_runtime_seconds
            )
            # --- 结束再次确认 --- 
        except Exception as e:
            print(f"分析和保存最终结果时出错: {e}")
            import traceback
            traceback.print_exc()
    else:
        print("警告: 未能成功获取最终 DFM 模型结果对象或准备分析数据，无法执行最终分析和保存。")

    print("\n--- 调优完成 --- ")
    print(f"总耗时: {total_runtime_seconds / 60:.2f} 分钟")
    print(f"结果已保存至: {run_output_dir}") # 结束时再次确认目录

    if log_file:
        log_file.write("\n--- 日志结束 ---\n")
        try:
            log_file.close()
            print(f"详细调优日志已保存到: {log_file_path}")
        except Exception as close_e:
            print(f"关闭日志文件时出错: {close_e}")

# --- 主程序入口 --- 
if __name__ == "__main__":
    # print("DEBUG: Calling run_tuning()...") # ADD DEBUG
    run_tuning()

# print("\n--- 脚本结束 ---") # 注释掉这一行 