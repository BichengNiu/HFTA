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
from sklearn.decomposition import PCA # <-- 新增：导入 PCA
from sklearn.impute import SimpleImputer # <-- 新增：导入 SimpleImputer

# --- 新增：动态构建数据文件路径 ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(SCRIPT_DIR) # 假设脚本在项目根目录的子文件夹下
# --- 结束新增 ---

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

# --- 全局执行标志 (防止重复运行) ---
_run_tuning_executed = False

# --- 新增：是否使用 Log(Yt) - Log(Yt-52) 转换 (根据用户要求设为 True) ---
# *** MODIFIED: Set USE_LOG_YOY_TRANSFORM to True based on user request ***

# --- 常量 ---
# *** MODIFIED: Update Excel file if different, TARGET_VARIABLE, TARGET_FREQ ***
# --- 修改: 使用动态构建的路径 --- 
EXCEL_DATA_FILE = os.path.join(BASE_DIR, 'data', '经济数据库.xlsx')
# --- 结束修改 --- 
TARGET_VARIABLE = '规模以上工业增加值:当月同比' # NEW Target Variable
TARGET_FREQ = 'W-FRI'                   # Target frequency remains Weekly Friday
# *** ADDED: Define TARGET_SHEET_NAME constant ***
TARGET_SHEET_NAME = '工业增加值同比增速-月度' # NEW: Specify the sheet containing the target

RESULTS_FILE = "tuning_results_forward_selection.csv"
DETAILED_LOG_FILE = "调优日志.txt"
FINAL_FACTOR_FILE = os.path.join('dfm_result', 'final_factors.png')
FINAL_PLOT_FILE = os.path.join('dfm_result', 'final_nowcast_comparison.png')
EXCEL_OUTPUT_FILE = os.path.join('dfm_result', 'result.xlsx') # Renamed output file

# --- 新增：控制是否移除长连续缺失变量 ---
REMOVE_VARS_WITH_CONSECUTIVE_NANS = True # 设置为 False 则不执行基于连续缺失的变量移除
CONSECUTIVE_NAN_THRESHOLD = 10           # 定义连续缺失的阈值

# --- 测试模式开关 和 迭代次数 ---
TEST_MODE = False # 设置为 True 以运行快速测试版
N_ITER_FIXED = 30 # 切换为完整版迭代次数
N_ITER_TEST = 5   # 测试版使用的迭代次数
n_iter_to_use = N_ITER_TEST if TEST_MODE else N_ITER_FIXED
# --- 结束 --- 

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
    target_mean_original: float, # <-- 新增：原始目标变量均值
    target_std_original: float,  # <-- 新增：原始目标变量标准差
    max_iter: int = 50,
) -> Tuple[float, float, float, float, float, float, bool]: # 返回 (is_rmse, oos_rmse, is_mae, oos_mae, is_hit_rate, oos_hit_rate, is_svd_error)
    """评估给定变量和参数下的 DFM。
    返回: (is_rmse, oos_rmse, is_mae, oos_mae, is_hit_rate, oos_hit_rate, is_svd_error) 元组。
          如果评估失败或无法计算指标，RMSE/MAE=np.inf, Hit Rate=-np.inf
          is_svd_error 指示失败是否由 SVD 不收敛引起。
    """
    start_time = time.time()
    k_factors = params.get('k_factors', None)
    if k_factors is None:
        # print(f"错误: evaluate_dfm_params 的 params 字典缺少 k_factors: {params}") # 保持注释
        return np.inf, np.inf, np.inf, np.inf, -np.inf, -np.inf, False # 失败
    n_shocks = k_factors

    try:
        if target_variable not in variables:
             print(f"    错误: 目标变量 {target_variable} 不在当前变量列表中: {variables}")
             return np.inf, np.inf, np.inf, np.inf, -np.inf, -np.inf, False
        predictor_vars = [v for v in variables if v != target_variable]
        original_target_series_full = full_data[target_variable].copy()

        data_for_fitting = full_data[:validation_end].copy()
        current_variables = variables.copy()
        if not all(col in data_for_fitting.columns for col in current_variables):
             missing = [col for col in current_variables if col not in data_for_fitting.columns]
             print(f"    错误: 评估函数缺少列: {missing}")
             return np.inf, np.inf, np.inf, np.inf, -np.inf, -np.inf, False
        current_data = data_for_fitting[current_variables].copy()
        current_data = current_data.apply(pd.to_numeric, errors='coerce')

        # --- 修改: 使用变量级转换代替全局 LogYoY ---
        current_data_transformed, transform_details = apply_stationarity_transforms(current_data, target_variable)
        # --- 结束修改 ---

        # 使用转换后的数据进行后续处理
        current_data = current_data_transformed
        current_variables = current_data.columns.tolist() # 变量列表可能因全 NaN 列被移除而改变
        if target_variable not in current_variables:
             print(f"    错误: Vars={len(variables)}, n_factors={k_factors} -> 目标变量在变量级转换后丢失 (可能全为 NaN)。")
             return np.inf, np.inf, np.inf, np.inf, -np.inf, -np.inf, False
        predictor_vars = [v for v in current_variables if v != target_variable]

        if current_data.empty:
             # print(f"    错误: Vars={len(current_variables)}, n_factors={k_factors} -> 变量级转换后数据为空") # 注释掉
             return np.inf, np.inf, np.inf, np.inf, -np.inf, -np.inf, False

        all_nan_cols = current_data.columns[current_data.isna().all()].tolist()
        if all_nan_cols:
            print(f"    警告: Vars={len(current_variables)}, n_factors={k_factors} -> 以下列在变量级转换后全为 NaN，将被移除: {all_nan_cols}")
            current_variables = [v for v in current_variables if v not in all_nan_cols]
            if target_variable not in current_variables:
                print(f"    错误: 目标变量在移除全 NaN 列后被移除。")
                return np.inf, np.inf, np.inf, np.inf, -np.inf, -np.inf, False
            current_data = current_data.drop(columns=all_nan_cols)
            predictor_vars = [v for v in current_variables if v != target_variable]
            if current_data.empty or current_data.shape[1] < k_factors:
                print(f"    错误: 移除全 NaN 列后，变量数 ({current_data.shape[1]}) 不足因子数 ({k_factors}) 或数据为空。")
                return np.inf, np.inf, np.inf, np.inf, -np.inf, -np.inf, False

        if current_data.isnull().all().all():
             print(f"    错误: Vars={len(current_variables)}, n_factors={k_factors} -> 数据在移除全 NaN 列后所有值仍为 NaN。")
             return np.inf, np.inf, np.inf, np.inf, -np.inf, -np.inf, False

        # print(f"    Standardizing data (Shape: {current_data.shape})...") # 注释掉
        obs_mean = current_data.mean(skipna=True)
        obs_std = current_data.std(skipna=True)
        zero_std_cols = obs_std.index[obs_std == 0].tolist()
        if zero_std_cols:
             print(f"    警告: Vars={len(current_variables)}, n_factors={k_factors} -> 以下列标准差为 0，将设为 1: {zero_std_cols}")
             obs_std[zero_std_cols] = 1.0
        current_data_std = (current_data - obs_mean) / obs_std
        
        # --- 修改: 使用传入的原始目标变量统计量进行反标准化 ---
        if pd.isna(target_mean_original) or pd.isna(target_std_original) or target_std_original == 0:
             print(f"    错误: 传入的原始目标变量统计量无效 (Mean: {target_mean_original}, Std: {target_std_original})，无法反标准化。")
             return np.inf, np.inf, np.inf, np.inf, -np.inf, -np.inf, False
        # --- 结束修改 ---

        std_all_nan_cols = current_data_std.columns[current_data_std.isna().all()].tolist()
        if std_all_nan_cols:
             print(f"    错误: Vars={len(current_variables)}, n_factors={k_factors} -> 标准化后以下列变为全 NaN: {std_all_nan_cols}")
             return np.inf, np.inf, np.inf, np.inf, -np.inf, -np.inf, False
             
        if current_data_std.shape[0] < k_factors:
              print(f"    错误: Vars={len(current_variables)}, n_factors={k_factors} -> 处理后数据行数 ({current_data_std.shape[0]}) 不足因子数 ({k_factors})。")
              return np.inf, np.inf, np.inf, np.inf, -np.inf, -np.inf, False

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

        if (not hasattr(dfm_results, 'x_sm') or dfm_results.x_sm is None or dfm_results.x_sm.empty or 
            not hasattr(dfm_results, 'Lambda') or dfm_results.Lambda is None):
            print(f"    错误: Vars={len(current_variables)}, n_factors={k_factors} -> DFM 结果不完整")
            return np.inf, np.inf, np.inf, np.inf, -np.inf, -np.inf, False

        factors_sm = dfm_results.x_sm
        lambda_matrix = dfm_results.Lambda
        if lambda_matrix.shape[1] != k_factors:
             print(f"    警告: Lambda 矩阵列数 ({lambda_matrix.shape[1]}) 与预期因子数 ({k_factors}) 不符。可能使用 PCA 回退初始化？")
             if lambda_matrix.shape[1] > k_factors:
                  lambda_matrix = lambda_matrix[:, :k_factors]
             else: 
                  return np.inf, np.inf, np.inf, np.inf, -np.inf, -np.inf, False 
        
        lambda_df = pd.DataFrame(lambda_matrix, index=current_variables, columns=[f'Factor{i+1}' for i in range(k_factors)])

        if target_variable not in lambda_df.index:
            print(f"    错误: Vars={len(current_variables)}, n_factors={k_factors} -> 目标变量 {target_variable} 不在 Lambda 索引中 (可用: {lambda_df.index.tolist()})")
            return np.inf, np.inf, np.inf, np.inf, -np.inf, -np.inf, False
        
        lambda_target = lambda_matrix[lambda_df.index.get_loc(target_variable), :]

        nowcast_standardized = factors_sm.to_numpy() @ lambda_target 
        # --- 修改: 使用传入的原始目标变量统计量进行反标准化 ---
        nowcast_orig_values = nowcast_standardized * target_std_original + target_mean_original
        # --- 结束修改 ---
        nowcast_series_orig = pd.Series(nowcast_orig_values, index=factors_sm.index, name='Nowcast_Orig')

        target_original_series = original_target_series_full.copy()
        target_for_comparison = target_original_series.dropna()

        common_index = nowcast_series_orig.index.intersection(target_for_comparison.index)
        if common_index.empty:
            print(f"    错误: Vars={len(current_variables)}, n_factors={k_factors} -> 反标准化 Nowcast 和目标序列没有共同索引")
            return np.inf, np.inf, np.inf, np.inf, -np.inf, -np.inf, False
        
        aligned_df = pd.DataFrame({
            'Nowcast_Orig': nowcast_series_orig.loc[common_index],
            target_variable: target_for_comparison.loc[common_index]}).dropna()
        
        # --- 计算指标 --- 
        is_rmse = np.inf
        oos_rmse = np.inf
        is_mae = np.inf  # <-- 新增
        oos_mae = np.inf # <-- 新增
        is_hit_rate = -np.inf
        oos_hit_rate = -np.inf

        train_df = aligned_df.loc[:train_end_date]
        if not train_df.empty and len(train_df) > 1:
            try:
                 # Calculate IS RMSE & MAE
                 is_mse = mean_squared_error(train_df[target_variable], train_df['Nowcast_Orig'])
                 is_rmse = np.sqrt(is_mse)
                 is_mae = mean_absolute_error(train_df[target_variable], train_df['Nowcast_Orig']) # <-- 新增

                 # Calculate IS Hit Rate (不变)
                 changes_df_train = train_df.diff().dropna()
                 if not changes_df_train.empty:
                     correct_direction_train = (np.sign(changes_df_train['Nowcast_Orig']) == np.sign(changes_df_train[target_variable])) & (changes_df_train[target_variable] != 0)
                     non_zero_target_changes_train = (changes_df_train[target_variable] != 0).sum()
                     if non_zero_target_changes_train > 0:
                          is_hit_rate = correct_direction_train.sum() / non_zero_target_changes_train * 100
            except Exception as e_is:
                 print(f"    计算 IS 指标时出错: {e_is}")
        else:
            print(f"    训练期 (up to {train_end_date}) 数据不足 (< 2 点)，无法计算 IS 指标") # 更新打印

        validation_df = aligned_df.loc[VALIDATION_START_DATE:VALIDATION_END_DATE]
        if not validation_df.empty and len(validation_df) > 1:
             try:
                 # Calculate OOS RMSE & MAE
                 oos_mse = mean_squared_error(validation_df[target_variable], validation_df['Nowcast_Orig'])
                 oos_rmse = np.sqrt(oos_mse)
                 oos_mae = mean_absolute_error(validation_df[target_variable], validation_df['Nowcast_Orig']) # <-- 新增

                 # Calculate OOS Hit Rate (不变)
                 changes_df_val = validation_df.diff().dropna()
                 if not changes_df_val.empty:
                     correct_direction_val = (np.sign(changes_df_val['Nowcast_Orig']) == np.sign(changes_df_val[target_variable])) & (changes_df_val[target_variable] != 0)
                     non_zero_target_changes_val = (changes_df_val[target_variable] != 0).sum()
                     if non_zero_target_changes_val > 0:
                          oos_hit_rate = correct_direction_val.sum() / non_zero_target_changes_val * 100
             except Exception as e_oos:
                 print(f"    计算 OOS 指标时出错: {e_oos}")
        else:
            print(f"    验证期 ({VALIDATION_START_DATE} to {VALIDATION_END_DATE}) 数据不足 (< 2 点)，无法计算 OOS 指标") # 更新打印

        # --- 计算组合指标 (用于可能的未来扩展或调试，优化逻辑不变) ---
        # (RMSE 和 Hit Rate 的组合指标计算保持不变)
        # ...
        combined_mae = np.inf # <-- 新增 (可选)
        if np.isfinite(is_mae) and np.isfinite(oos_mae): combined_mae = 0.5 * is_mae + 0.5 * oos_mae
        elif np.isfinite(is_mae): combined_mae = is_mae
        elif np.isfinite(oos_mae): combined_mae = oos_mae

        return is_rmse, oos_rmse, is_mae, oos_mae, is_hit_rate, oos_hit_rate, False # 返回 RMSE, MAE, HitRate

    except (np.linalg.LinAlgError, ValueError) as err:
        err_msg = str(err)
        is_svd_error = "svd did not converge" in err_msg.lower()
        return np.inf, np.inf, np.inf, np.inf, -np.inf, -np.inf, is_svd_error # 返回 RMSE/MAE=inf
    except Exception as e:
        return np.inf, np.inf, np.inf, np.inf, -np.inf, -np.inf, False # 返回 RMSE/MAE=inf

# --- 新增: 计算 Bai & Ng (2002) 信息准则 --- 
# Function definition removed as requested.

# --- 新增：重新创建 analyze_and_save_final_results 函数 --- 
def analyze_and_save_final_results(
    run_output_dir: str, # 确保类型提示正确
    timestamp_str: str, # 确保类型提示正确
    excel_output_path: str, # <-- 新增: Excel 文件路径参数
    all_data_full,
    data_for_analysis,
    target_variable,
    final_dfm_results,
    best_variables,
    best_params,
    var_type_map,
    best_avg_hit_rate_tuning: float,
    best_avg_mae_tuning: float,    # Note: Name kept as mae, but value is RMSE
    total_runtime_seconds,
    factor_contributions=None,
    final_transform_log=None,
    pca_results_df=None, # <-- 新增: PCA 结果
    contribution_results_df=None, # <-- 新增: 因子贡献结果
    # k_icp1_recommended=None, # Removed IC parameter
    # k_icp2_recommended=None,  # Removed IC parameter
    # ic_details=None, # Removed IC parameter
    var_industry_map=None # <-- 确认接收行业映射
):
    """分析最终DFM结果并保存到Excel和图中。"""
    print("\n--- [DEBUG] Entered analyze_and_save_final_results --- ") # DEBUG
    # --- 再次确认传递的参数 ---
    print(f"  [DEBUG] Received run_output_dir: {run_output_dir}") # DEBUG
    print(f"  [DEBUG] Received excel_output_path: {excel_output_path}") # DEBUG
    print(f"  [DEBUG] final_dfm_results type: {type(final_dfm_results)}") # DEBUG
    print(f"  [DEBUG] data_for_analysis keys: {list(data_for_analysis.keys()) if data_for_analysis else 'None'}") # DEBUG
    # print(f"    run_output_dir: {run_output_dir}")
    # print(f"    timestamp_str: {timestamp_str}")
    # print(f"    excel_output_path: {excel_output_path}")
    # print(f"    最终变量数: {len(best_variables)}")
    # print(f"    最终参数: {best_params}")
    # print(f"    最终转换日志: {final_transform_log}") # Optional: 可取消注释查看
    # --- 结束确认 ---

    # --- **再次确认**: 确保输出目录存在 ---
    if not run_output_dir or not isinstance(run_output_dir, str):
        print(f"错误：无效的 run_output_dir 参数: {run_output_dir}")
        return
    try:
        print(f"  [DEBUG] Attempting to create directory: {run_output_dir}") # DEBUG
        os.makedirs(run_output_dir, exist_ok=True) # 必须在最前面创建目录
        print(f"  [DEBUG] Directory confirmed/created: {run_output_dir}")
    except OSError as e:
        print(f"错误：无法创建输出目录 '{run_output_dir}': {e}")
        return # 如果无法创建目录，则无法继续保存
    # --- **结束再次确认** ---

    # --- 动态生成输出文件完整路径 (再次确认基于 run_output_dir) --- 
    final_plot_file = os.path.join(run_output_dir, f"final_nowcast_comparison_{timestamp_str}.png")
    factor_plot_base_dir = run_output_dir # 因子图的基础目录
    heatmap_file = os.path.join(run_output_dir, f"factor_loadings_heatmap_{timestamp_str}.png")
    combined_factor_plot_file = os.path.join(run_output_dir, f"all_factors_timeseries_{timestamp_str}.png") # 新增：合并因子图路径
    print(f"  [DEBUG] final_plot_file path: {final_plot_file}") # DEBUG
    print(f"  [DEBUG] heatmap_file path: {heatmap_file}") # DEBUG
    print(f"  [DEBUG] combined_factor_plot_file path: {combined_factor_plot_file}") # DEBUG
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
        # use_log_yoy = USE_LOG_YOY_TRANSFORM # 移除对已删除标志的引用

        if isinstance(final_k_factors, str):
            print("错误: best_params 中 k_factors 无效。")
            return
            
        # --- 移除 LogYoY 相关打印 ---
        # print(f"分析参数: k_factors={final_k_factors}, use_log_yoy(predictors)={use_log_yoy}")
        print(f"分析参数: k_factors={final_k_factors}")
        # --- 结束移除 ---

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

        final_is_rmse = np.nan
        final_oos_rmse = np.nan
        final_is_mae = np.nan  # <-- 新增
        final_oos_mae = np.nan # <-- 新增
        hit_rate_train = np.nan
        hit_rate_validation = np.nan

        if common_index_final.empty:
            print("错误: 最终 Nowcast 和目标序列没有共同索引。无法计算最终指标。")
        else:
            aligned_df_final = pd.DataFrame({'Nowcast': final_nowcast_orig.loc[common_index_final], 'Target': target_for_comparison.loc[common_index_final]}).dropna()
            
            # 计算验证期指标
            validation_df_final = aligned_df_final.loc[VALIDATION_START_DATE:VALIDATION_END_DATE]
            if not validation_df_final.empty and len(validation_df_final) > 0:
                final_oos_rmse = np.sqrt(mean_squared_error(validation_df_final['Target'], validation_df_final['Nowcast']))
                final_oos_mae = mean_absolute_error(validation_df_final['Target'], validation_df_final['Nowcast']) # <-- 新增
                print(f"  最终 OOS RMSE (验证期): {final_oos_rmse:.6f}")
                print(f"  最终 OOS MAE (验证期): {final_oos_mae:.6f}") # <-- 新增打印
                
                # OOS Hit Rate (逻辑不变)
                changes_df_val = validation_df_final.diff().dropna()
                if not changes_df_val.empty and len(changes_df_val) > 0:
                    correct_direction_val = (np.sign(changes_df_val['Nowcast']) == np.sign(changes_df_val['Target'])) & (changes_df_val['Target'] != 0)
                    non_zero_target_changes_val = (changes_df_val['Target'] != 0).sum()
                    if non_zero_target_changes_val > 0:
                         hit_rate_validation = correct_direction_val.sum() / non_zero_target_changes_val * 100
                         print(f"  验证期 Hit Rate (%): {hit_rate_validation:.2f} (基于 {non_zero_target_changes_val} 个非零变化点)")
            else:
                 print("警告: 验证期数据不足，无法计算 OOS 指标。")
            
            # 计算训练期指标
            train_df_final = aligned_df_final.loc[:TRAIN_END_DATE]
            if not train_df_final.empty and len(train_df_final) > 1:
                 try:
                    final_is_rmse = np.sqrt(mean_squared_error(train_df_final['Target'], train_df_final['Nowcast']))
                    final_is_mae = mean_absolute_error(train_df_final['Target'], train_df_final['Nowcast']) # <-- 新增
                    print(f"  最终 IS RMSE (训练期): {final_is_rmse:.6f}") # <-- 新增打印
                    print(f"  最终 IS MAE (训练期): {final_is_mae:.6f}")   # <-- 新增打印
                 
                    # IS Hit Rate (逻辑不变)
                    changes_df_train = train_df_final.diff().dropna()
                    if not changes_df_train.empty and len(changes_df_train) > 0:
                         correct_direction_train = (np.sign(changes_df_train['Nowcast']) == np.sign(changes_df_train['Target'])) & (changes_df_train['Target'] != 0)
                         non_zero_target_changes_train = (changes_df_train['Target'] != 0).sum()
                         if non_zero_target_changes_train > 0:
                              hit_rate_train = correct_direction_train.sum() / non_zero_target_changes_train * 100
                              print(f"  训练期 Hit Rate (%): {hit_rate_train:.2f} (基于 {non_zero_target_changes_train} 个非零变化点)")
                 except Exception as e_is_final:
                    print(f"计算最终 IS 指标时出错: {e_is_final}")
            else:
                print("警告: 训练期数据不足，无法计算 IS 指标。")
        # --- 结束计算最终指标 (包括 MAE) ---

        # --- 更新 interpretation_text (可选，但建议添加 MAE) ---
        interpretation_text = (
            f"最终分析总结 (Run: {timestamp_str}):\n"
            # ... (其他参数不变) ...
            f"- 最佳平均胜率 (调优): {best_avg_hit_rate_tuning:.2f}%\n"
            f"- 对应平均 RMSE (调优): {best_avg_mae_tuning:.6f}\n" # 注意: best_avg_mae_tuning 变量名虽为 mae，但实际存的是 RMSE
            # (如果需要，可以在 run_tuning 中计算并传递调优阶段的平均 MAE)
            f"- 最终 IS RMSE{diff_label_for_metrics}: {final_is_rmse:.6f}\n" # <-- 新增
            f"- 最终 IS MAE{diff_label_for_metrics}: {final_is_mae:.6f}\n"   # <-- 新增
            f"- 最终训练期 Hit Rate (%){diff_label_for_metrics}: {hit_rate_train:.2f}\n"
            f"- 最终 OOS RMSE{diff_label_for_metrics}: {final_oos_rmse:.6f}\n"
            f"- 最终 OOS MAE{diff_label_for_metrics}: {final_oos_mae:.6f}\n"   # <-- 新增
            f"- 最终验证期 Hit Rate (%){diff_label_for_metrics}: {hit_rate_validation:.2f}\n"
            # ... (剩余部分不变) ...
        )

        # --- 写入 Excel 文件 --- 
        print(f"\n[DEBUG] Starting Excel write process to: {excel_output_path}...")
        try:
            with pd.ExcelWriter(excel_output_path, engine='openpyxl') as writer:
                # --- 修改: Sheet: Summary_Overview (添加 MAE 指标) ---
                try:
                    summary_params = [
                        'Final Variables Count', 'Best k_factors (Tuned)',
                        'Best Avg Hit Rate (Tuning %)', 'Corresponding Avg RMSE (Tuning)',
                        f'Final IS RMSE{diff_label_for_metrics}', # <-- 更新标签
                        f'Final IS MAE{diff_label_for_metrics}',  # <-- 新增
                        f'Hit Rate (Train %){diff_label_for_metrics}', 
                        f"Final OOS RMSE{diff_label_for_metrics}",
                        f"Final OOS MAE{diff_label_for_metrics}", # <-- 新增
                        f'Hit Rate (Validation %){diff_label_for_metrics}', 'Total Runtime (s)',
                    ]
                    summary_values = [
                        len(best_variables), final_k_factors,
                        f"{best_avg_hit_rate_tuning:.2f}" if pd.notna(best_avg_hit_rate_tuning) else "N/A",
                        f"{best_avg_mae_tuning:.6f}" if pd.notna(best_avg_mae_tuning) else "N/A", # 仍然是 RMSE
                        f"{final_is_rmse:.6f}" if pd.notna(final_is_rmse) else "N/A", # <-- 新增
                        f"{final_is_mae:.6f}" if pd.notna(final_is_mae) else "N/A",   # <-- 新增
                        f"{hit_rate_train:.2f}" if pd.notna(hit_rate_train) else "N/A",
                        f"{final_oos_rmse:.6f}" if pd.notna(final_oos_rmse) else "N/A",
                        f"{final_oos_mae:.6f}" if pd.notna(final_oos_mae) else "N/A",   # <-- 新增
                        f"{hit_rate_validation:.2f}" if pd.notna(hit_rate_validation) else "N/A",
                        f"{total_runtime_seconds:.2f}",
                    ]
                    summary_df = pd.DataFrame({'Parameter': summary_params, 'Value': summary_values})
                    summary_df['Value'] = summary_df['Value'].astype(str)
                    summary_df.to_excel(writer, sheet_name='Summary_Overview', index=False)
                    current_row = writer.sheets['Summary_Overview'].max_row
                    
                    # 追加 Analysis Text 内容 (分行写入)
                    start_row_analysis = current_row + 2 # 加 2 留出空行
                    analysis_lines = interpretation_text.strip().split('\n')
                    analysis_df = pd.DataFrame({'Analysis Text': analysis_lines})
                    analysis_df.to_excel(writer, sheet_name='Summary_Overview', 
                                           startrow=start_row_analysis, index=False, header=True)
                    current_row = writer.sheets['Summary_Overview'].max_row
                    print(f"  Summary_Overview 和 Analysis_Text 已合并写入 Sheet: 'Summary_Overview'")

                    # <-- 新增: 追加 PCA 结果 -->
                    start_row_pca = current_row + 2 # 加 2 留出空行
                    if pca_results_df is not None and not pca_results_df.empty:
                        print(f"  追加 PCA 结果到 Sheet: 'Summary_Overview'...")
                        # 添加解释性注释
                        pd.DataFrame([["PCA 解释方差说明："],
                                      ["  - 解释方差 (%): 单个主成分解释原始数据总方差的百分比。"],
                                      ["  - 累计解释方差 (%): 从第一个主成分开始，到当前主成分为止，累计解释的总方差百分比。"]]).to_excel(writer, sheet_name='Summary_Overview', startrow=start_row_pca-1, index=False, header=False)
                        start_row_pca += 3 # 更新 PCA 表格的起始行
                        # 添加标题行
                        pd.DataFrame([["PCA Explained Variance"]]).to_excel(writer, sheet_name='Summary_Overview', startrow=start_row_pca-1, index=False, header=False)
                        pca_results_df.to_excel(writer, sheet_name='Summary_Overview', startrow=start_row_pca, index=False, header=True)
                        current_row = writer.sheets['Summary_Overview'].max_row
                    else:
                        print("  警告: PCA 结果不可用，无法追加到 Summary_Overview Sheet。")
                    # <-- 结束新增 -->
                    
                    # <-- 新增: 追加因子贡献度结果 -->
                    start_row_contrib = current_row + 2 # 加 2 留出空行
                    if contribution_results_df is not None and not contribution_results_df.empty:
                        print(f"  追加因子贡献度结果到 Sheet: 'Summary_Overview'...")
                        # 添加解释性注释
                        pd.DataFrame([["因子贡献度说明："],
                                      ["  - 载荷 (Loading): 因子与目标变量之间的相关性强度和方向。"],
                                      ["  - 对共同方差贡献 (%): 单个因子解释目标变量由所有因子共同解释那部分方差的百分比。"],
                                      ["  - 对总方差贡献(近似 %): 单个因子解释目标变量总方差的百分比（近似值）。"],
                                      ["  - 目标变量共同度 (Communality): 目标变量的方差能被所有提取出的因子共同解释的比例。"]]).to_excel(writer, sheet_name='Summary_Overview', startrow=start_row_contrib-1, index=False, header=False)
                        start_row_contrib += 5 # 更新贡献度表格的起始行
                        # 添加标题行
                        pd.DataFrame([["Factor Contribution to Target Variance"]]).to_excel(writer, sheet_name='Summary_Overview', startrow=start_row_contrib-1, index=False, header=False)
                        contribution_results_df.to_excel(writer, sheet_name='Summary_Overview', startrow=start_row_contrib, index=False, header=True)
                        print(f"  因子贡献度结果已追加写入 Sheet: 'Summary_Overview'")
                    else:
                        print("  警告: 因子贡献度结果不可用，无法追加到 Summary_Overview Sheet。")
                    # <-- 结束新增 -->
                    
                except Exception as e: print(f"写入 Summary_Overview (合并后) 时出错: {e}")
                
                # Sheet: Final_Selected_Variables (合并因子载荷)
                try:
                    vars_df = pd.DataFrame(best_variables, columns=['Variable Name'])
                    cleaned_var_names_series = vars_df['Variable Name'].astype(str).str.strip()
                    normalized_lowercase_keys = cleaned_var_names_series.apply(lambda x: unicodedata.normalize('NFKC', x).strip().lower())
                    vars_df['Variable Type'] = normalized_lowercase_keys.map(var_type_map).fillna('Unknown')
                    
                    # <-- 新增: 添加转换信息列 -->
                    if final_transform_log:
                        # <-- 修改: fillna('') -->
                        vars_df['Transformation'] = vars_df['Variable Name'].map(final_transform_log).fillna('')
                    else:
                        vars_df['Transformation'] = '' # <-- 修改: 保持一致用空字符串 -->
                    # <-- 结束新增 -->
                    
                    # <-- 新增: 合并因子载荷 -->
                    if lambda_df_final is not None and not lambda_df_final.empty:
                        print("  合并因子载荷到 Final_Selected_Variables Sheet...")
                        # 重置索引以便合并（如果 lambda_df_final 的索引是 Variable Name）
                        loadings_to_merge = lambda_df_final.reset_index().rename(columns={'index': 'Variable Name'})
                        # 执行左合并
                        vars_df = pd.merge(vars_df, loadings_to_merge, on='Variable Name', how='left')
                        
                        # 重新排序列顺序
                        factor_cols = [col for col in vars_df.columns if col.startswith('Factor')]
                        # <-- 修改: 调整列顺序 -->
                        desired_cols = ['Variable Name', 'Variable Type', 'Transformation'] + factor_cols
                        # <-- 结束修改 -->
                        # 确保所有期望的列都存在于DataFrame中
                        existing_cols = [col for col in desired_cols if col in vars_df.columns]
                        vars_df = vars_df[existing_cols]
                    else:
                         print("  警告: 因子载荷 (lambda_df_final) 不可用，无法合并到 Final_Selected_Variables Sheet。")
                    # <-- 结束新增 -->

                    vars_df.to_excel(writer, sheet_name='Final_Selected_Variables', index=False)
                    print(f"  已写入合并后的 Sheet: 'Final_Selected_Variables'") # 更新日志
                except Exception as e: print(f"写入 Final_Selected_Variables (合并后) 时出错: {e}")

                # <-- 新增: 生成 Selected_Vars_Transformed Sheet -->
                try:
                    print("  生成应用变量级转换后的选定变量数据 (Selected_Vars_Transformed)...")
                    if best_variables and not all_data_full.empty:
                        data_subset = all_data_full[best_variables].copy()
                        # 再次应用转换以获取用于表格的数据
                        transformed_data_for_sheet, _ = apply_stationarity_transforms(data_subset, target_variable) 
                        
                        if not transformed_data_for_sheet.empty:
                             # 格式化以便写入 Excel
                             # <-- 修改: fillna('') -->
                             data_to_save_transformed = transformed_data_for_sheet.copy().fillna('')
                             if isinstance(data_to_save_transformed.index, pd.DatetimeIndex):
                                  data_to_save_transformed.index = data_to_save_transformed.index.strftime('%Y-%m-%d')
                             else: data_to_save_transformed.index = data_to_save_transformed.index.astype(str)
                             data_to_save_transformed.columns = data_to_save_transformed.columns.astype(str)
                             data_to_save_transformed = data_to_save_transformed.reset_index()
                             data_to_save_transformed.to_excel(writer, sheet_name='Selected_Vars_Transformed', index=False)
                             print(f"  已写入 Sheet: 'Selected_Vars_Transformed'")
                        else:
                             print(f"  警告: 转换后的数据为空，无法写入 'Selected_Vars_Transformed' Sheet。")
                    else:
                        print(f"  警告: 缺少最终变量或原始数据，无法生成 'Selected_Vars_Transformed' Sheet。")
                except Exception as e: print(f"写入 Selected_Vars_Transformed 时出错: {e}")
                # <-- 结束新增 -->

                # Sheet: Nowcast_vs_Target (修改: 周度对齐，使用 all_data_full 中的目标)
                try:
                    print(f"  生成 Nowcast_vs_Target Sheet (周度对齐, 使用已处理目标)... ") # 更新日志
                    if final_nowcast_orig is not None and not final_nowcast_orig.empty:
                        # 开始于完整的周度 nowcast 序列
                        nowcast_weekly_df = final_nowcast_orig.to_frame(name=f'Nowcast{diff_label_for_metrics}')

                        # --- 修改: 直接从 all_data_full 获取目标列 (已包含延迟) ---
                        target_col_name = f'Target{diff_label_for_metrics}'
                        if target_variable in all_data_full.columns:
                            # 直接选择列，不需要重采样
                            target_delayed_weekly = all_data_full[[target_variable]].copy()
                            target_delayed_weekly = target_delayed_weekly.rename(columns={target_variable: target_col_name})
                        else:
                            print(f"  警告: 在 all_data_full 中找不到原始目标变量 '{target_variable}'，Target 列将为空。")
                            target_delayed_weekly = pd.DataFrame(index=nowcast_weekly_df.index, columns=[target_col_name]) # 创建空的 DF
                        # --- 结束修改 ---

                        # 左连接：保留所有周度 nowcast 点，匹配处理后的 target 值
                        combined_weekly_df = nowcast_weekly_df.join(target_delayed_weekly, how='left')

                        # <-- 新增: 合并因子时间序列 -->
                        if final_factors is not None and not final_factors.empty:
                            print("    合并因子时间序列...")
                            # 确保因子列名不与现有列冲突 (理论上不会)
                            factor_cols_to_add = final_factors.columns.difference(combined_weekly_df.columns)
                            if len(factor_cols_to_add) != len(final_factors.columns):
                                print("    警告: 因子列名可能与现有列冲突。")
                            combined_weekly_df = combined_weekly_df.join(final_factors[factor_cols_to_add], how='left')
                        else:
                            print("    警告: 最终因子序列不可用，无法添加到 Nowcast_vs_Target Sheet。")
                        # <-- 结束新增 -->

                        # <-- 修改: 直接创建稀疏的月度预测列 (此部分逻辑不变) -->
                        try:
                            print("    计算并添加月末周五对应的月度预测列...")
                            monthly_forecast_col_name = 'Monthly_Forecast'
                            # 确保 nowcast 索引是 DatetimeIndex
                            if not isinstance(final_nowcast_orig.index, pd.DatetimeIndex):
                                nowcast_index_dt = pd.to_datetime(final_nowcast_orig.index)
                            else:
                                nowcast_index_dt = final_nowcast_orig.index
                                
                            # 找到每个月最后一个周五的索引
                            last_friday_indices_m = final_nowcast_orig.groupby(nowcast_index_dt.to_period('M')).apply(lambda x: x.index.max())
                            
                            # 创建一个与 combined_weekly_df 索引相同的新 Series，初始为 NaN
                            monthly_forecast_sparse = pd.Series(index=combined_weekly_df.index, dtype=float)
                            
                            # 只在月末周五的位置填入对应的 nowcast 值
                            # 需要确保 last_friday_indices_m 存在于 combined_weekly_df 的索引中
                            valid_indices = combined_weekly_df.index.intersection(last_friday_indices_m)
                            if not valid_indices.empty:
                                monthly_forecast_sparse.loc[valid_indices] = combined_weekly_df.loc[valid_indices, f'Nowcast{diff_label_for_metrics}']
                            else:
                                print("    警告: 未能在 combined_weekly_df 索引中找到有效的月末周五索引。")

                            # 将新列添加到 DataFrame
                            combined_weekly_df[monthly_forecast_col_name] = monthly_forecast_sparse
                            print("    已添加稀疏的月度预测列。")
                        except Exception as e_add_monthly:
                            print(f"    警告: 添加月度预测列时出错: {e_add_monthly}")
                            # 即使出错，也继续，只是缺少这一列
                        # <-- 结束修改 -->
                            
                        # 格式化输出
                        combined_weekly_df = combined_weekly_df.fillna('') # 用空字符串代替 N/A
                        # <-- 确保索引转换在添加列之后 -->
                        if isinstance(combined_weekly_df.index, pd.DatetimeIndex):
                            combined_weekly_df.index = combined_weekly_df.index.strftime('%Y-%m-%d')
                        else: 
                            combined_weekly_df.index = combined_weekly_df.index.astype(str)
                        # <-- 结束确保 -->
                        combined_weekly_df.columns = combined_weekly_df.columns.astype(str)
                        combined_weekly_df = combined_weekly_df.reset_index()
                        # <-- 新增: 调整最终列顺序 (包含因子列) -->
                        factor_cols_in_df = [col for col in combined_weekly_df.columns if col.startswith('Factor')]
                        # <-- 修改：移除 Monthly_Forecast -->
                        desired_final_cols = ['index', f'Nowcast{diff_label_for_metrics}'] + factor_cols_in_df + [f'Target{diff_label_for_metrics}']
                        # desired_final_cols = ['index', f'Nowcast{diff_label_for_metrics}', f'Target{diff_label_for_metrics}', 'Monthly_Forecast'] # 旧顺序
                        # <-- 结束修改 -->
                        # <-- 结束新增 -->
                        existing_final_cols = [col for col in desired_final_cols if col in combined_weekly_df.columns]
                        combined_weekly_df = combined_weekly_df[existing_final_cols]
                        # <-- 结束新增 -->
                        combined_weekly_df.to_excel(writer, sheet_name='Nowcast_vs_Target', index=False)
                        print(f"  已写入 Sheet: 'Nowcast_vs_Target' (周度对齐, 包含到 {final_nowcast_orig.index.max()} 的预测)")
                    else: 
                        print(f"警告: 无法写入 Nowcast_vs_Target，因为最终 Nowcast 序列为空或 None。")
                except Exception as e: print(f"写入 Nowcast_vs_Target (周度对齐) 时出错: {e}")

                # Sheet: Factor_Contribution_Analysis (移除，已合并)
                # ... (removed) ...

                # Sheet: Factor_Interpretation (修改解释逻辑)
                try:
                    factor_interpretations = []
                    if lambda_df_final is not None and not lambda_df_final.empty:
                        print(f"  生成因子解释...")
                        num_top_vars = 5 # 定义显示多少个最高载荷变量
                        from collections import Counter # <--- 移到循环外

                        for factor_col in lambda_df_final.columns:
                            loadings = lambda_df_final[factor_col].sort_values(ascending=False)

                            top_positive = loadings.head(num_top_vars)
                            top_negative = loadings.tail(num_top_vars).sort_values() # 确保负值最低的在前

                            # <-- 修改: 使用 \n 替换 \\n -->
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
                            # <-- 结束修改 -->

                            # --- 修改: 增强经济解释生成 ---
                            pos_type_counts = Counter(t for t in pos_types if t != "未知")
                            neg_type_counts = Counter(t for t in neg_types if t != "未知")

                            economic_meaning = "可能的解释: "
                            dominant_pos_types = [f"{t} ({c})" for t, c in pos_type_counts.most_common(2)]
                            dominant_neg_types = [f"{t} ({c})" for t, c in neg_type_counts.most_common(2)]

                            if dominant_pos_types:
                                economic_meaning += f"正向主要由 [{', '.join(dominant_pos_types)}] 类型变量驱动。"
                            else:
                                economic_meaning += "正向无明显主导变量类型。"

                            if dominant_neg_types:
                                economic_meaning += f" 负向主要由 [{', '.join(dominant_neg_types)}] 类型变量驱动。"
                            else:
                                economic_meaning += " 负向无明显主导变量类型。"

                            # 尝试提供宏观/行业角度的总结 (可以根据具体类型进一步细化)
                            all_top_types = pos_types + neg_types
                            all_type_counts = Counter(t for t in all_top_types if t != "未知")
                            most_common_overall, common_overall_count = all_type_counts.most_common(1)[0] if all_type_counts else ("未知", 0)

                            if common_overall_count >= num_top_vars * 0.8: # 如果整体类型非常集中
                                if most_common_overall == '价格': economic_meaning += " 整体可能反映价格/通胀因素。"
                                elif most_common_overall in ['工业', '生产', '能源']: economic_meaning += " 整体可能反映工业生产活动强度。"
                                elif most_common_overall in ['景气度', '预期', '情绪']: economic_meaning += " 整体可能反映市场信心/预期。"
                                elif most_common_overall == '金融': economic_meaning += " 整体可能反映金融市场状况。"
                                elif most_common_overall == '外贸': economic_meaning += " 整体可能反映外贸活动。"
                                else: economic_meaning += f" 整体高度关联 '{most_common_overall}' 类型指标。"
                            else:
                                economic_meaning += " 该因子综合反映多种经济活动。"
                            # --- 结束修改 ---

                            # <-- 修改: 使用 \n 替换 \\n -->
                            interpretation += f"  {economic_meaning}\n" # 使用修改后的解释
                            # <-- 结束修改 -->
                            factor_interpretations.append(interpretation)

                        interpret_df = pd.DataFrame({'Factor Interpretation': factor_interpretations})
                        interpret_df.to_excel(writer, sheet_name='Factor_Interpretation', index=False)
                        print(f"  因子解释已写入 Sheet: 'Factor_Interpretation'")
                    else: print(f"  警告: 无法生成因子解释，因为最终载荷矩阵 ('lambda_df_final') 不可用或为空。")
                except Exception as e_interpret: print(f"写入 Factor_Interpretation 时出错: {e_interpret}")

                # Sheet: Data_Input_to_DFM
                try:
                    data_to_save_dfm_input = final_data_processed.copy()
                    # --- 修改: 保存前四舍五入到6位小数 --- 
                    data_to_save_dfm_input = data_to_save_dfm_input.round(6)
                    # --- 结束修改 ---
                    # --- 格式化索引和列名 (保持不变) ---
                    if isinstance(data_to_save_dfm_input.index, pd.DatetimeIndex):
                        data_to_save_dfm_input.index = data_to_save_dfm_input.index.strftime('%Y-%m-%d')
                    else: 
                        data_to_save_dfm_input.index = data_to_save_dfm_input.index.astype(str)
                    data_to_save_dfm_input.columns = data_to_save_dfm_input.columns.astype(str)
                    # --- 结束格式化 ---
                    data_to_save_dfm_input = data_to_save_dfm_input.reset_index()
                    data_to_save_dfm_input.to_excel(writer, sheet_name='Data_Input_to_DFM', index=False)
                    print(f"  进入DFM模型的数据已写入 Sheet: 'Data_Input_to_DFM' (格式化为6位小数)")
                except Exception as e_dfm_input: print(f"写入 Data_Input_to_DFM 时出错: {e_dfm_input}")

                # Sheet: Full_Aligned_Data_Orig
                try:
                    # 保存原始对齐数据 (保存前转换为字符串避免Excel自动格式化问题)
                    # <-- 修改: fillna('') -->
                    data_to_save_orig = all_data_full.copy().fillna('') 
                    if isinstance(data_to_save_orig.index, pd.DatetimeIndex):
                        data_to_save_orig.index = data_to_save_orig.index.strftime('%Y-%m-%d')
                    else: data_to_save_orig.index = data_to_save_orig.index.astype(str)
                    data_to_save_orig.columns = data_to_save_orig.columns.astype(str)
                    data_to_save_orig = data_to_save_orig.astype(str).reset_index()
                    data_to_save_orig.to_excel(writer, sheet_name='Full_Aligned_Data_Orig', index=False)
                except Exception as e: print(f"写入 Full_Aligned_Data_Orig 时出错: {e}")

            print(f"[DEBUG] Excel file base write completed.") # DEBUG
        except Exception as e_writer:
            print(f"[DEBUG] ERROR during ExcelWriter setup or saving: {e_writer}") # DEBUG
            print(f"创建或写入 Excel 文件 '{excel_output_path}' 时发生严重错误: {e_writer}")
        # --- 结束写入 Excel 文件 --- 

        # --- 生成 Nowcast 对比图 --- 
        print(f"\n[DEBUG] Checking conditions for final nowcast plot...") # DEBUG
        if 'target_for_comparison' in locals() and not target_for_comparison.empty:
            print(f"  [DEBUG] Conditions met. Calling plot_final_nowcast, save to: {final_plot_file}") # DEBUG
            # print(f"调用 plot_final_nowcast, 保存到: {final_plot_file}") # 确认传入 plot_final_nowcast 的路径正确
            plot_final_nowcast(
                final_nowcast_series=final_nowcast_orig, 
                target_for_plot=target_for_comparison, 
                validation_start=VALIDATION_START_DATE, 
                validation_end=VALIDATION_END_DATE, 
                title=f'最终 DFM Nowcast vs 观测值 ({diff_label_for_metrics.strip(" ()")}) [Run: {timestamp_str}]', 
                filename=final_plot_file, # 确认传递的是基于 run_output_dir 的路径
                # use_log_yoy=use_log_yoy # 移除参数
            )
        else:
            print("[DEBUG] Conditions NOT met for final nowcast plot.") # DEBUG
            print("警告: 无法生成最终绘图，因为用于比较的目标序列为空或未定义。")
        # --- 结束生成 Nowcast 图 --- 

        # --- 生成因子载荷热力图 (修改1: 排序) --- 
        print("\n[DEBUG] Checking conditions for heatmap with industry sorting...") # DEBUG
        lambda_df_for_heatmap = None # 初始化变量
        lambda_df_sorted = None # 初始化排序后的 df
        if lambda_df_final is not None and not lambda_df_final.empty and var_industry_map is not None:
            try:
                # 复制一份用于排序和绘图
                lambda_df_sorted_temp = lambda_df_final.copy()

                # 添加行业列
                default_industry = '未知行业'
                lambda_df_sorted_temp['行业'] = lambda_df_sorted_temp.index.map(
                    lambda var_name: var_industry_map.get(
                        unicodedata.normalize('NFKC', str(var_name)).strip().lower(), default_industry
                    )
                )

                # 按照行业和变量名排序
                index_name = lambda_df_sorted_temp.index.name if lambda_df_sorted_temp.index.name else 'index'
                if index_name == '行业': # Avoid conflict if index is already named '行业'
                     lambda_df_sorted_temp = lambda_df_sorted_temp.reset_index()
                     index_name = 'Original_Index' # Choose a temporary name
                     lambda_df_sorted_temp = lambda_df_sorted_temp.rename(columns={'index':index_name}).set_index(index_name)
                     
                lambda_df_sorted = lambda_df_sorted_temp.sort_values(by=['行业', index_name])

                # 准备用于绘图的数据（移除行业列）
                lambda_df_for_heatmap = lambda_df_sorted.drop(columns=['行业'])
                print("  [DEBUG] Loadings sorted by industry for heatmap.")

            except Exception as e_sort:
                 print(f"  [DEBUG] ERROR during heatmap data sorting: {e_sort}")
                 lambda_df_for_heatmap = lambda_df_final # 如果排序失败，回退到原始顺序
                 lambda_df_sorted = None # 排序失败，重置
        elif lambda_df_final is not None and not lambda_df_final.empty:
            lambda_df_for_heatmap = lambda_df_final # 如果没有行业信息，使用原始顺序
            lambda_df_sorted = None # 无排序
            print("  [DEBUG] Industry map not available, using original order for heatmap.")
        else:
            print("  [DEBUG] lambda_df_final is None or empty, cannot generate heatmap data.")
        
        # 继续使用 lambda_df_for_heatmap 绘制热力图 (后续步骤)
        if lambda_df_for_heatmap is not None:
            try:
                # 准备带贡献度的因子标签
                factor_labels_with_contrib = []
                if factor_contributions:
                    for factor_name in lambda_df_for_heatmap.columns:
                        contrib = factor_contributions.get(factor_name, None)
                        if contrib is not None:
                            factor_labels_with_contrib.append(f"{factor_name}\n({contrib:.1f}%)")
                        else:
                            factor_labels_with_contrib.append(factor_name)
                else:
                    factor_labels_with_contrib = lambda_df_for_heatmap.columns.tolist()
                
                # 简化 figsize 计算
                heatmap_width = max(10, lambda_df_for_heatmap.shape[1] * 1.5)
                heatmap_height = max(12, lambda_df_for_heatmap.shape[0] * 0.3)
                fig, ax = plt.subplots(figsize=(heatmap_width, heatmap_height))
                
                # 简化 heatmap 调用 (移除 ax=ax, 它会被自动使用)
                sns.heatmap(lambda_df_for_heatmap, 
                            annot=True,
                            fmt=".2f",
                            cmap='RdBu_r',
                            center=0,
                            linewidths=.5,
                            linecolor='lightgray', 
                            xticklabels=factor_labels_with_contrib, 
                            yticklabels=lambda_df_for_heatmap.index,
                            cbar_kws={'shrink': .5})
                
                ax.set_title(f'因子载荷热力图 [Run: {timestamp_str}]') # Removed trailing backslash
                ax.set_xlabel('因子及其贡献度 (%)')
                ax.set_ylabel('变量') 
                ax.tick_params(axis='x', rotation=0)
                ax.tick_params(axis='y', rotation=0)
                
                # --- 行业标签添加部分将在下一步加入 ---

                plt.tight_layout() # 调整布局
                print(f"  [DEBUG] Saving heatmap (potentially sorted) to: {heatmap_file}")
                plt.savefig(heatmap_file, bbox_inches='tight')
                plt.close(fig) # 关闭特定的 figure
                print(f"  [DEBUG] Heatmap saved successfully.")
            except Exception as e_heatmap:
                print(f"  [DEBUG] ERROR generating/saving heatmap: {e_heatmap}")
                print(f"  生成或保存热力图时出错: {e_heatmap}")
                # 尝试关闭可能未关闭的图形
                try:
                    plt.close(fig)
                except NameError:
                    pass # fig 可能未定义
                except Exception:
                    pass # 其他关闭错误
            # --- End Fix ---
        else:
            print("[DEBUG] Conditions NOT met for heatmap (lambda_df_for_heatmap is None).")
            print("警告: 无法生成因子载荷热力图，因为最终载荷矩阵不可用。")
        # --- 结束生成热力图 --- 

        # --- 修改: 生成并保存所有因子合并的时间序列图 --- 
        print("\n[DEBUG] Checking conditions for combined factors plot...") # DEBUG
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
                print(f"  [DEBUG] Saving combined factors plot to: '{combined_factor_plot_file}'") # 调试打印
                # print(f"[DEBUG] Saving combined factors plot to: '{combined_factor_plot_file}'") # 调试打印
                plt.savefig(combined_factor_plot_file) # 保存合并后的图
                plt.close() # 关闭图形
                print(f"  [DEBUG] Combined factors plot saved successfully.") # DEBUG
                # print(f"  合并的因子时间序列图已保存到: {combined_factor_plot_file}")
            except Exception as e_plot_combined_factors:
                print(f"  [DEBUG] ERROR saving combined factors plot: {e_plot_combined_factors}") # DEBUG
                print(f"  保存合并的因子时间序列图时出错: {e_plot_combined_factors}")
                plt.close() # 确保关闭图形
        else:
            print("[DEBUG] Conditions NOT met for combined factors plot (final_factors is None or empty).") # DEBUG
            print("警告: 最终因子 (final_factors) 为空或 None，无法生成合并的因子时间序列图。")
        # --- 结束修改因子图 --- 

        print("[DEBUG] analyze_and_save_final_results function finished.") # DEBUG

    except Exception as e_analyze:
        print(f"[DEBUG] UNEXPECTED ERROR in analyze_and_save_final_results: {e_analyze}") # DEBUG
        print(f"在 analyze_and_save_final_results 函数中发生意外错误: {e_analyze}")
        import traceback
        traceback.print_exc()
    # --- Fix: Add finally block if necessary, or handle exceptions within inner try blocks ---
    # If the main try was just for overall error catching, the inner try blocks should handle their specific errors.

# --- 修改绘图函数以绘制完整 Nowcast --- 
def plot_final_nowcast(final_nowcast_series, target_for_plot, validation_start, validation_end, title, filename):
    print("\n生成最终 Nowcasting 图 (原始水平 - 绘制完整预测)...") # 更新日志
    try:
        # 不再基于 common_index_plot 过滤，但需要目标序列的索引用于屏蔽
        target_index = target_for_plot.index

        nowcast_col_name = 'Nowcast_Orig'
        target_col_name = target_for_plot.name if target_for_plot.name is not None else 'Actual'
        if target_col_name == nowcast_col_name:
            target_col_name = 'Observed_Value'

        # --- 修改: 直接使用完整的 final_nowcast_series ---
        plot_df = final_nowcast_series.to_frame(name=nowcast_col_name)
        # 合并目标变量，只为了屏蔽和绘制，允许 NaN
        plot_df = plot_df.join(target_for_plot.rename(target_col_name), how='left') 

        # --- 新增: 绘图时移除1月/2月实际值 (如果目标值存在) ---
        if target_col_name in plot_df.columns:
            month_indices_plot = plot_df.index.month
            plot_df.loc[(month_indices_plot == 1) | (month_indices_plot == 2), target_col_name] = np.nan 
        # --- 结束新增 ---

        if not plot_df.empty:
            plt.figure(figsize=(14, 7))
            
            nowcast_label = '周度 Nowcast (原始水平)'
            actual_label = '观测值 (原始水平, 屏蔽1/2月)'
            ylabel = '值 (原始水平)'

            # --- 修改: 绘制完整的 Nowcast --- 
            plt.plot(plot_df.index, plot_df[nowcast_col_name], label=nowcast_label, linestyle='-', alpha=0.8, color='blue')
            # --- 结束修改 ---
            
            # 绘制实际观测点 (只绘制非 NaN 的点)
            if target_col_name in plot_df.columns:
                plt.plot(plot_df.index, plot_df[target_col_name], label=actual_label, marker='o', linestyle='None', markersize=4, color='red')
            
            try:
                # --- 修改: 确保验证期标记在图表范围内 --- 
                plot_start_date = plot_df.index.min()
                plot_end_date = plot_df.index.max()
                val_start = pd.to_datetime(validation_start)
                val_end = pd.to_datetime(validation_end)
                # 只在验证期与绘图范围有重叠时标记
                span_start = max(plot_start_date, val_start)
                span_end = min(plot_end_date, val_end)
                if span_start < span_end:
                    plt.axvspan(span_start, span_end, color='yellow', alpha=0.2, label='验证期')
                else:
                     plt.axvspan(val_start, val_end, color='yellow', alpha=0.2, label='验证期 (超出部分范围)') # 仍然添加图例标签
                     print("警告: 验证期与绘图范围无重叠或重叠无效，标记可能不完整。")
                # --- 结束修改 ---
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
                print("错误：无法准备用于绘图的数据。") 
    except Exception as e:
        print(f"生成或保存最终 Nowcasting 图时出错: {e}")

# --- 主逻辑 --- 
def run_tuning():
    global _run_tuning_executed # 引用全局标志
    if _run_tuning_executed:
        print("警告: run_tuning() 已被调用，跳过重复执行。")
        return # 如果已执行，则直接返回
    _run_tuning_executed = True # 标记为已执行

    script_start_time = time.time()
    total_evaluations = 0
    svd_error_count = 0
    # 移除对已删除标志的引用
    
    # --- 新增: 生成时间戳和运行目录 ---
    timestamp_str = time.strftime("%Y%m%d_%H%M%S")

    # 获取脚本所在的目录 (假设 tune_dfm.py 在 dym_estimate 文件夹下)
    script_dir = os.path.dirname(os.path.abspath(__file__)) # This gets the directory containing tune_dfm.py

    # 基础输出目录应该是脚本目录下的 'dfm_result' 子目录
    base_output_dir = os.path.join(script_dir, 'dfm_result')

    # 运行目录仍然是 base_output_dir 下的 run_{timestamp}
    run_output_dir = os.path.join(base_output_dir, f'run_{timestamp_str}')

    log_filename = f"调优日志_{timestamp_str}.txt" # 带时间戳的日志文件名
    log_file_path = os.path.join(run_output_dir, log_filename) # 日志文件的完整路径
    # --- 结束新增 ---
    
    # --- 新增: 定义 Excel 输出文件路径 ---
    excel_output_file = os.path.join(run_output_dir, f"result_{timestamp_str}.xlsx")
    # --- 结束新增 ---

    print(f"--- 开始变量后向剔除与超参数调优 (优化目标: 平均胜率优先, k_factors其次; 因子数范围动态确定) ---")
    # print(f"--- 开始变量后向剔除与超参数调优 (优化目标: 平均胜率优先, RMSE其次; k_factors 动态范围, 对数同比转换: {transform_info}) ---") # 旧打印
    print(f"本次运行结果将保存在: {run_output_dir}") # 打印本次运行目录
    
    try:
        # --- 修改: 确保目录存在并使用带时间戳的日志文件 --- 
        os.makedirs(run_output_dir, exist_ok=True) # 确保运行目录存在
        log_file = open(log_file_path, 'w', encoding='utf-8') # 使用完整路径打开日志
        log_file.write(f"--- 开始详细调优日志 (Run: {timestamp_str}) ---\n")
        log_file.write(f"输出目录: {run_output_dir}\n")
        # --- 修改: 更新日志，移除 LogYoY 相关描述 ---
        log_file.write(f"配置: 优化目标=平均胜率优先, k_factors其次\n")
        # log_file.write(f"配置: 对数同比转换={transform_info}; 优化目标=平均胜率优先, RMSE其次\n") # 旧日志
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

    # --- 新增: 加载 var_industry_map ---
    print("尝试从 Excel 文件加载指标行业映射 (预期在第一个 Sheet)...")
    var_industry_map = {} # 初始化为空字典
    col_industry_name = '行业' # 假设行业列名为 '行业'

    if col_industry_name in indicator_sheet.columns:
        try:
            # 清理列名
            col_industry_name = col_industry_name.strip()
            # 创建映射，键进行规范化和小写处理，处理 NaN
            industry_map_temp = pd.Series(
                indicator_sheet[col_industry_name].astype(str).str.strip().values,
                index=indicator_sheet[col_indicator_name].astype(str).str.strip() # 使用与 var_type_map 相同的键
            ).to_dict()

            var_industry_map = {unicodedata.normalize('NFKC', str(k)).strip().lower(): str(v).strip()
                               for k, v in industry_map_temp.items()
                               if pd.notna(k) and str(k).lower() != 'nan' and pd.notna(v) and str(v).lower() != 'nan'}
            print(f"  成功创建 {len(var_industry_map)} 个指标的行业映射。")
        except Exception as e_ind_map:
            print(f"  警告: 加载指标行业映射时出错: {e_ind_map}。将使用空映射。")
    else:
        print(f"  警告: 在 Excel 文件 '{EXCEL_DATA_FILE}' 的第一个 sheet 中未找到行业列 '{col_industry_name}'。热力图将无法按行业排序。")
    # --- 结束新增 ---

    # --- 新增: 计算原始目标变量的稳定均值和标准差 ---
    original_target_series_for_stats = all_data_aligned_weekly[TARGET_VARIABLE].copy().dropna()
    if original_target_series_for_stats.empty:
        print(f"错误: 原始目标变量 '{TARGET_VARIABLE}' 在移除 NaN 后为空，无法计算统计量。")
        sys.exit(1)
    target_mean_original = original_target_series_for_stats.mean()
    target_std_original = original_target_series_for_stats.std()
    if pd.isna(target_mean_original) or pd.isna(target_std_original) or target_std_original == 0:
        print(f"错误: 计算得到的原始目标变量统计量无效 (Mean: {target_mean_original}, Std: {target_std_original})。")
        sys.exit(1)
    print(f"已计算原始目标变量的稳定统计量 (用于反标准化): Mean={target_mean_original:.4f}, Std={target_std_original:.4f}")
    print("-"*30)
    # --- 结束新增 ---

    # --- 修改: 将连续缺失检查包裹在条件块中 --- 
    if REMOVE_VARS_WITH_CONSECUTIVE_NANS:
        print(f"\n--- (启用) 检查初始预测变量 ({len(initial_variables)}) 的连续缺失值 (阈值 >= {CONSECUTIVE_NAN_THRESHOLD})... ---")
        consecutive_nan_threshold_init = CONSECUTIVE_NAN_THRESHOLD # 使用常量
        vars_to_remove_consecutive_init = []

        # 检查的对象是 all_data_aligned_weekly 中对应的初始预测变量列
        for col in initial_variables:
            # --- 新增: 跳过目标变量的检查 ---
            if col == TARGET_VARIABLE:
                # print(f"    跳过对目标变量 '{col}' 的连续缺失值检查。") # 可以保持静默
                continue
            # --- 结束新增 ---

            # 计算连续 NaN
            is_na = all_data_aligned_weekly[col].isna()
            na_blocks = is_na.ne(is_na.shift()).cumsum()[is_na]
            if not na_blocks.empty:
                max_consecutive_nan = na_blocks.value_counts().max()
                if max_consecutive_nan >= consecutive_nan_threshold_init:
                    vars_to_remove_consecutive_init.append((col, max_consecutive_nan))

        if vars_to_remove_consecutive_init:
            print(f"  警告: 以下初始预测变量因存在 {consecutive_nan_threshold_init} 期或更长的连续缺失值，将被移除，不参与后续筛选:")
            removed_vars_log_init = []
            for var_rm, max_nan in vars_to_remove_consecutive_init:
                print(f"    - {var_rm} (最大连续缺失: {max_nan} 期)")
                removed_vars_log_init.append(var_rm)

            # 从 initial_predictors 和 initial_variables 中移除这些变量
            initial_predictors = [v for v in initial_variables if v != TARGET_VARIABLE and v not in removed_vars_log_init] # 确保目标变量始终在 initial_predictors 之外
            initial_variables = [v for v in initial_variables if v not in removed_vars_log_init] # 移除非目标变量

            print(f"  移除长连续缺失变量后，剩余初始预测变量数: {len(initial_predictors)}")
            # --- 新增: 确保目标变量仍在 initial_variables 中 --- 
            if TARGET_VARIABLE not in initial_variables:
                 print(f"  警告: 目标变量 '{TARGET_VARIABLE}' 在移除其他变量后丢失，正在重新添加...")
                 initial_variables.append(TARGET_VARIABLE)
                 initial_variables.sort() # 可选：保持排序
            # --- 结束新增 ---
            if log_file:
                 try:
                     log_file.write("\n" + "-"*35 + "\n")
                     log_file.write(f"--- (启用) 变量筛选前连续缺失值检查 (阈值 {consecutive_nan_threshold_init}期) ---\n")
                     log_file.write(f"移除变量: {removed_vars_log_init}\n")
                     log_file.write(f"剩余预测变量数: {len(initial_predictors)}\n")
                     log_file.write("-"*35 + "\n")
                 except Exception as log_e:
                     print(f"写入初始连续缺失检查日志时出错: {log_e}")
        else:
            print(f"  所有初始预测变量的最大连续缺失值均低于 {consecutive_nan_threshold_init} 期。所有变量均通过此检查！")
            if log_file:
                try:
                    log_file.write("\n" + "-"*35 + "\n")
                    log_file.write(f"--- (启用) 变量筛选前连续缺失值检查 (阈值 {consecutive_nan_threshold_init}期) ---\n")
                    log_file.write(f"所有初始预测变量均通过检查。\n")
                    log_file.write("-"*35 + "\n")
                except Exception as log_e:
                    print(f"写入初始连续缺失检查日志时出错: {log_e}")
    else:
        # 如果不执行移除，则打印信息
        print(f"\n--- (禁用) 跳过基于连续缺失值 (阈值 >= {CONSECUTIVE_NAN_THRESHOLD}) 的初始变量移除步骤。---")
        initial_predictors = [v for v in initial_variables if v != TARGET_VARIABLE] # 仍然需要定义 initial_predictors
        if log_file:
             try:
                 log_file.write("\n" + "-"*35 + "\n")
                 log_file.write(f"--- (禁用) 跳过变量筛选前连续缺失值检查 (阈值 {CONSECUTIVE_NAN_THRESHOLD}期) ---\n")
                 log_file.write("-"*35 + "\n")
             except Exception as log_e:
                 print(f"写入禁用连续缺失检查日志时出错: {log_e}")

    print("-"*30) # 添加分隔符
    # --- 结束修改 --- 

    # --- 原有代码被跳过 ---
    # print("\n--- 确定初始变量块和动态因子数范围 ---")
    # ... [后续所有 DFM 调优和分析代码将不会执行] ...

    print("\n--- 确定初始变量块和动态因子数范围 (基于筛选后的变量) ---") # 修改描述
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

    # --- 测试模式: 限制因子数范围 ---
    if TEST_MODE:
        K_FACTORS_RANGE_TEST = K_FACTORS_RANGE[:2] # 最多取前两个因子数
        print(f"*** 测试模式：限制因子数范围为: {K_FACTORS_RANGE_TEST} ***")
        k_factors_for_tuning = K_FACTORS_RANGE_TEST
    else:
        k_factors_for_tuning = K_FACTORS_RANGE
    # --- 结束测试模式限制 ---

    HYPERPARAMS_TO_TUNE = []
    for k in k_factors_for_tuning:
        HYPERPARAMS_TO_TUNE.append({'k_factors': k})
    print(f"构建了 {len(HYPERPARAMS_TO_TUNE)} 个超参数组合进行测试 (仅调整 k_factors)。")

    print(f"\n训练期结束: {TRAIN_END_DATE}")
    print(f"验证期: {VALIDATION_START_DATE} to {VALIDATION_END_DATE}")
    print(f"待优化的因子数范围 (动态确定): {k_factors_for_tuning}") # 使用限制后的范围
    print(f"优化目标: 最大化 (IS_HitRate + OOS_HitRate) / 2 (平均胜率优先), 然后最小化 (IS_RMSE + OOS_RMSE) / 2 (平均RMSE其次)")
    print(f"将使用最多 {MAX_WORKERS} 个进程进行并行计算。")
    print(f"优化目标: 1. 最大化平均胜率; 2. 最小化因子数; 3. 最小化平均 RMSE")
    print(f"使用的迭代次数: {n_iter_to_use}") # 打印使用的迭代次数
    print("-" * 30)

    print("\n--- 初始化: 评估所有预测变量 ---")
    best_overall_score_tuple = (-np.inf, np.inf, np.inf)
    best_overall_params = None
    best_overall_variables = initial_variables.copy()
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
                                     target_mean_original=target_mean_original,
                                     target_std_original=target_std_original,
                                     max_iter=n_iter_to_use # 使用 n_iter_to_use
                                 )
            futures_initial_map[future] = params

    print(f"提交 {len(futures_initial_map)} 个初始评估任务 (使用最多 {MAX_WORKERS} 进程)...")
    results = []

    for future in concurrent.futures.as_completed(futures_initial_map):
        params = futures_initial_map[future]
        total_evaluations += 1 # 计数
        try:
            # --- 修改: 解包时包含 MAE --- 
            is_rmse, oos_rmse, is_mae, oos_mae, is_hit_rate, oos_hit_rate, is_svd_error = future.result()
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
                
            # --- 修改: 保存结果时包含 MAE --- 
            results.append({'variables': initial_variables, 'params': params, 
                            'combined_rmse': combined_rmse, 'combined_hit_rate': combined_hit_rate,
                            'is_rmse': is_rmse, 'oos_rmse': oos_rmse, 'is_mae': is_mae, 'oos_mae': oos_mae, # 添加 MAE
                            'is_hit_rate': is_hit_rate, 'oos_hit_rate': oos_hit_rate})

            # --- 优化逻辑保持不变 (基于 Hit Rate, K, RMSE) --- 
            if np.isfinite(combined_hit_rate) and np.isfinite(combined_rmse) and 'k_factors' in params:
                current_k_factors = params['k_factors']
                if np.isfinite(current_k_factors):
                     current_score_tuple = (combined_hit_rate, -current_k_factors, -combined_rmse)

                     # 使用元组比较直接确定是否更新
                     if current_score_tuple > best_overall_score_tuple:
                         best_overall_score_tuple = current_score_tuple
                         best_overall_params = params
                         best_overall_variables = initial_variables.copy() # 确保使用当前的 initial_variables
                         initial_best_found = True

            # ... (旧的 score tuple 比较逻辑已移除)

        except Exception as e:
            print(f"初始化评估期间参数 {params} 运行出错: {e}")

    if not initial_best_found:
        print("错误: 初始化评估未能成功运行任何超参数组合。无法继续。")
        sys.exit(1)

    # --- 修改: 更新初始结果打印 (反映新逻辑) --- 
    best_init_hr, best_init_neg_k, best_init_neg_rmse = best_overall_score_tuple
    print(f"初始评估完成。最佳组合得分: 平均胜率={best_init_hr:.2f}%, 因子数={-best_init_neg_k}, 平均 RMSE={-best_init_neg_rmse:.6f} (参数: {best_overall_params})")
    print(f"初始变量数量: {len(best_overall_variables)}") # 使用 best_overall_variables
    if log_file:
         try:
             log_file.write("\n" + "-"*35 + "\n")
             log_file.write("--- 初始化: 所有变量评估结果 ---\n")
             log_file.write(f"初始变量组 ({len(best_overall_variables)}): {best_overall_variables}\n")
             log_file.write(f"最佳参数: {best_overall_params}\n")
             log_file.write(f"最佳得分 (HR, -K, -RMSE): {best_overall_score_tuple[0]:.2f}, {best_overall_score_tuple[1]}, {best_overall_score_tuple[2]:.6f}\n")
             log_file.write("-"*35 + "\n")
         except Exception as log_e:
             print(f"写入初始评估日志时出错: {log_e}")
    # --- 结束修改 ---

    print("\n--- 开始分块向后变量剔除 (使用初始确定的块结构) ---") 
    blocks = initial_blocks 
    print(f"将对 {len(blocks)} 个块进行变量剔除:")
    # for block_name, block_vars in blocks.items():  # <-- 这行是旧代码，确保不重复
    #     print(f"  块 '{block_name}' ({len(block_vars)} 变量)") # <-- 这行是旧代码

    # --- 修改: 初始化当前最佳指标为新的三层优化目标 --- 
    current_best_variables = best_overall_variables.copy()
    current_best_score_tuple = best_overall_score_tuple
    current_best_params = best_overall_params
    # current_best_hit_rate = best_overall_hit_rate # 移除
    # current_best_mae = best_overall_mae          # 移除
    # current_best_params = best_overall_params     # 移除
    # --- 结束修改 ---

    for block_name, block_vars_list in tqdm(blocks.items(), desc="处理变量块", unit="block"):
        print(f"\n--- 处理块: '{block_name}' (初始 {len(block_vars_list)} 变量) ---")

        # # --- 新增: 跳过非 '产能' 块 --- 
        # if block_name != '产能':
        #     print(f"  根据要求，跳过块 '{block_name}' 的变量剔除。")
        #     continue
        # # --- 结束新增 ---

        # --- 快速测试: 只处理第一个块 (逻辑移除) ---
        # is_first_block = (block_name == list(blocks.keys())[0]) 
        # if not is_first_block:                                 
        #      print(f"  [快速测试] 跳过块 '{block_name}'")        
        #      continue                                          
        # --- 结束快速测试 (逻辑移除) ---

        if len(block_vars_list) <= 2: 
            print(f"块 '{block_name}' 变量数 ({len(block_vars_list)}) <= 2，跳过剔除。")
            continue

        block_stable = False
        while not block_stable:
            # --- 修改: 初始化本轮最佳为新的三层优化目标 --- 
            # best_score_tuple_this_iter = (-np.inf, -np.inf) # 旧 (hit_rate, -mae)
            best_candidate_score_tuple_this_iter = (-np.inf, np.inf, np.inf) # (hit_rate, -k, -rmse)
            best_removal_candidate_this_iter = None
            # variable_to_remove = None # 移除
            # params_for_best_removal = None # 移除
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
                            max_iter=n_iter_to_use, # 使用 n_iter_to_use
                            target_mean_original=target_mean_original, # <-- 传递稳定值
                            target_std_original=target_std_original,  # <-- 传递稳定值
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
                    # --- 修改: 解包时包含 MAE --- 
                    is_rmse, oos_rmse, is_mae, oos_mae, is_hit_rate, oos_hit_rate, is_svd_error = f_info.result()
                    if is_svd_error:
                        svd_error_count += 1 # 计数
                    
                    # --- 计算组合指标 (RMSE 和 Hit Rate) --- 
                    combined_rmse_removal = np.inf
                    if np.isfinite(is_rmse) and np.isfinite(oos_rmse): combined_rmse_removal = 0.5 * is_rmse + 0.5 * oos_rmse
                    elif np.isfinite(is_rmse): combined_rmse_removal = is_rmse
                    elif np.isfinite(oos_rmse): combined_rmse_removal = oos_rmse
            
                    combined_hit_rate_removal = -np.inf
                    valid_hit_rates_rem = []
                    if np.isfinite(is_hit_rate): valid_hit_rates_rem.append(is_hit_rate)
                    if np.isfinite(oos_hit_rate): valid_hit_rates_rem.append(oos_hit_rate)
                    if valid_hit_rates_rem: combined_hit_rate_removal = np.mean(valid_hit_rates_rem)
                    # --- 结束组合指标计算 ---
                        
                    results_this_iteration.append({
                        'combined_rmse': combined_rmse_removal, # 修改
                        'combined_hit_rate': combined_hit_rate_removal, # 修改
                        'params': context['params'],
                        'removed_var': context['removed_var'],
                        'is_mae': is_mae, # <-- 添加
                        'oos_mae': oos_mae # <-- 添加
                    })
                except Exception as exc:
                    print(f"处理块 {block_name}, 尝试移除 {context['removed_var']} 时出错: {exc}")

            # --- 修改: 选择最佳移除操作 (使用新的三层优化目标) ---
            # best_removal_candidate = None # 移除
            # best_candidate_score_tuple = (-np.inf, np.inf) # 旧 (hit_rate, -k_factors)

            for result in results_this_iteration:
                if np.isfinite(result['combined_hit_rate']) and np.isfinite(result['combined_rmse']) and 'k_factors' in result['params']:
                    current_k = result['params']['k_factors']
                    if np.isfinite(current_k):
                        # 使用 (hit_rate, -k_factors, -rmse) 进行比较
                        score_tuple_for_result = (result['combined_hit_rate'], -current_k, -result['combined_rmse'])
                        if score_tuple_for_result > best_candidate_score_tuple_this_iter:
                            best_candidate_score_tuple_this_iter = score_tuple_for_result
                            best_removal_candidate_this_iter = result

            # --- 结束修改 ---

            # --- 修改: 接受移除的条件 (基于新的三层优化逻辑) ---
            accept_removal = False
            reason_for_update = ""
            if best_removal_candidate_this_iter:
                 # 只有当本轮找到的最佳分数严格优于当前全局最佳分数时，才接受移除
                 if best_candidate_score_tuple_this_iter > current_best_score_tuple:
                      accept_removal = True
                      # 构造更新原因的描述
                      old_hr, old_neg_k, old_neg_rmse = current_best_score_tuple
                      new_hr, new_neg_k, new_neg_rmse = best_candidate_score_tuple_this_iter
                      reason_for_update = f"找到更优解: HR={new_hr:.2f}%(>{old_hr:.2f}%), K={-new_neg_k}(vs {-old_neg_k}), RMSE={-new_neg_rmse:.6f}(vs {-old_neg_rmse:.6f})"

            # if best_removal_candidate:
            #     candidate_hit_rate = best_candidate_score_tuple[0]
            #     candidate_k_factors = -best_candidate_score_tuple[1] # 转回正数
            #     candidate_rmse = best_removal_candidate['combined_rmse'] # 获取对应的 RMSE
            #     benchmark_hit_rate = current_best_hit_rate
            #     benchmark_k_factors = current_best_params['k_factors']
            #     improvement_threshold = 1.0 # 胜率提升阈值
            #
            #     # 检查是否满足更新条件
            #     if np.isfinite(candidate_hit_rate) and np.isfinite(benchmark_hit_rate):
            #         if candidate_hit_rate >= benchmark_hit_rate + improvement_threshold:
            #             accept_removal = True
            #             reason_for_update = f"胜率提升 >= {improvement_threshold:.1f}%"
            #         elif (candidate_hit_rate > benchmark_hit_rate - improvement_threshold) and \
            #              (candidate_k_factors < benchmark_k_factors):
            #             accept_removal = True
            #             reason_for_update = f"胜率接近(>{benchmark_hit_rate - improvement_threshold:.2f}%)且因子数减少({candidate_k_factors}<{benchmark_k_factors})"

            if accept_removal and best_removal_candidate_this_iter:
                # 接受移除
                variable_to_remove = best_removal_candidate_this_iter['removed_var']
                params_for_best_removal = best_removal_candidate_this_iter['params']
                # 更新全局最佳指标
                current_best_score_tuple = best_candidate_score_tuple_this_iter
                current_best_params = params_for_best_removal
                # 更新变量列表
                current_best_variables = next(item['remaining_vars'] for item in futures_removal if item['removed_var'] == variable_to_remove and item['params'] == params_for_best_removal)

                if variable_to_remove in block_vars_list:
                    block_vars_list.remove(variable_to_remove)

                # --- 修改: 更新打印和日志 (反映新逻辑) ---
                new_hr, new_neg_k, new_neg_rmse = current_best_score_tuple
                print(f"*** 块 '{block_name}' 找到改进: 移除 '{variable_to_remove}' ({reason_for_update}), 新最佳分数: HR={new_hr:.2f}%, K={-new_neg_k}, RMSE={-new_neg_rmse:.6f}, Params: {current_best_params} ***")
                # print(f"*** 块 '{block_name}' 找到改进: 移除 '{variable_to_remove}' ({reason_for_update}), 新最佳分数: HR={current_best_hit_rate:.2f}%, RMSE={current_best_mae:.6f}, Params: {current_best_params} ***")

                if log_file:
                    try:
                        log_file.write("\n" + "-"*35 + "\n")
                        log_file.write(f"--- 块 '{block_name}': 变量剔除结果 ---\n")
                        log_file.write(f"剔除变量: '{variable_to_remove}' (原因: {reason_for_update})\n") # 添加原因
                        log_file.write(f"当前变量组 ({len(current_best_variables)}): {current_best_variables}\n")
                        log_file.write(f"最佳参数: {current_best_params}\n")
                        log_file.write(f"新最佳得分 (HR, -K, -RMSE): {current_best_score_tuple[0]:.2f}, {current_best_score_tuple[1]}, {current_best_score_tuple[2]:.6f}\n")
                        log_file.write("-"*35 + "\n")
                    except Exception as log_e:
                        print(f"写入块 '{block_name}' 剔除日志时出错: {log_e}")
                # --- 结束修改 ---
            else:
                # --- 修改: 更新停止打印 (反映新逻辑) ---
                stop_hr, stop_neg_k, stop_neg_rmse = current_best_score_tuple
                print(f"块 '{block_name}' 内无变量移除可获得严格更优解 (当前最佳: HR={stop_hr:.2f}%, K={-stop_neg_k}, RMSE={-stop_neg_rmse:.6f})。此块完成。")
                # print(f"块 '{block_name}' 内无变量移除可满足新的优化条件 (当前 HR={current_best_hit_rate:.2f}%, k={current_best_params['k_factors']})。此块完成。")
                block_stable = True
                if log_file and eligible_vars_in_block:
                     try:
                         log_file.write("\n" + "-"*35 + "\n")
                         log_file.write(f"--- 块 '{block_name}': 停止剔除 ---\n")
                         log_file.write(f"原因: 块内剩余变量移除无法找到严格更优的评分元组 (Benchmark HR={stop_hr:.2f}%, K={-stop_neg_k}, RMSE={-stop_neg_rmse:.6f})。\n")
                         log_file.write("-"*35 + "\n")
                     except Exception as log_e:
                        print(f"写入块 '{block_name}' 停止日志时出错: {log_e}")
                # --- 结束修改 ---

    print("\n--- 所有块处理完毕 --- ")
    # --- 修改: 记录最终结果 (反映新逻辑) ---
    final_variables = current_best_variables.copy()
    final_params = current_best_params
    final_score_tuple = current_best_score_tuple
    # final_combined_hit_rate = current_best_hit_rate # 移除
    # final_combined_mae = current_best_mae # 移除
    final_hr, final_neg_k, final_neg_rmse = final_score_tuple
    print(f"最终变量数量: {len(final_variables)}")
    print(f"最终最佳平均胜率 (IS+OOS)/2: {final_hr:.2f}%")
    print(f"对应的因子数: {-final_neg_k}")
    print(f"对应的最终最佳平均 RMSE (IS+OOS)/2: {-final_neg_rmse:.6f}")
    print(f"最终最佳参数: {final_params}")
    # print(f"最终变量数量: {len(final_variables)}")
    # print(f"最终最佳平均胜率 (IS+OOS)/2: {final_combined_hit_rate:.2f}%")
    # print(f"对应的最终最佳平均 RMSE (IS+OOS)/2: {final_combined_mae:.6f}")
    # print(f"最终最佳参数 (因子数最少优先): {final_params}")
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
            
            print(f"最终运行参数: 因子数={final_k_factors}, 冲击数={final_p_shocks}") # 使用更简洁的打印
            # 旧打印

            print("准备最终运行的数据...")
            # --- 删除: 移动到变量筛选前的连续缺失值检查 ---
            # print(f"  检查最终变量列表 ({len(final_variables)}) 的连续缺失值 (>=10期)...")
            # # 使用已对齐到周度的完整数据进行检查
            # final_data_for_check = all_data_aligned_weekly[final_variables].copy()
            # consecutive_nan_threshold = 10
            # vars_to_remove_consecutive = []

            # for col in final_variables:
            #     if col == TARGET_VARIABLE: # 跳过目标变量
            #         continue

            #     # 计算连续 NaN
            #     is_na = final_data_for_check[col].isna()
            #     # 使用 groupby 和 cumsum 来识别连续块
            #     na_blocks = is_na.ne(is_na.shift()).cumsum()[is_na]
            #     if not na_blocks.empty:
            #          # 计算每个连续 NaN 块的长度
            #         max_consecutive_nan = na_blocks.value_counts().max()
            #         if max_consecutive_nan >= consecutive_nan_threshold:
            #             vars_to_remove_consecutive.append((col, max_consecutive_nan))

            # if vars_to_remove_consecutive:
            #     print(f"  警告: 以下预测变量因存在 {consecutive_nan_threshold} 期或更长的连续缺失值，将从最终模型中移除:")
            #     removed_vars_log = []
            #     for var_rm, max_nan in vars_to_remove_consecutive:
            #         print(f"    - {var_rm} (最大连续缺失: {max_nan} 期)")
            #         removed_vars_log.append(var_rm)

            #     # 从 final_variables 中移除这些变量 (不包括目标变量)
            #     final_variables = [v for v in final_variables if v not in removed_vars_log]

            #     print(f"  移除长连续缺失变量后，剩余变量数: {len(final_variables)}")
            #     if TARGET_VARIABLE not in final_variables:
            #          # 理论上不会发生
            #          print(f"  严重错误: 目标变量 {TARGET_VARIABLE} 在连续缺失值检查后被移除! 无法继续。")
            #          if log_file: log_file.write(f"错误：目标变量 {TARGET_VARIABLE} 在连续缺失值检查后被移除。\n")
            #          sys.exit(1)
            #     # 检查剩余变量数是否足够
            #     if len(final_variables) -1 < final_k_factors: # -1 是因为要排除目标变量
            #         print(f"  警告: 移除长连续缺失变量后，剩余预测变量数 ({len(final_variables)-1}) 少于因子数 ({final_k_factors})。模型可能不稳定或失败。")
            #         if log_file: log_file.write(f"警告: 移除长连续缺失变量后，预测变量数 ({len(final_variables)-1}) 少于因子数 ({final_k_factors})。\n")
            # else:
            #     # 修改打印信息，使其更明确
            #     print(f"  所有最终预测变量的最大连续缺失值均低于 {consecutive_nan_threshold} 期。所有变量均通过此检查并进入最终模型！")
            #     if log_file: log_file.write(f"最终预测变量连续缺失值检查通过 (阈值 {consecutive_nan_threshold} 期)。\n")
            # --- 结束连续缺失值检查 ---

            # --- 修改: 确保 final_variables 包含 TARGET_VARIABLE 且预测变量列表正确 ---
            if TARGET_VARIABLE not in final_variables:
                 print(f"错误: 最终变量列表 '{final_variables}' 中不包含目标变量 '{TARGET_VARIABLE}'。正在尝试添加...")
                 final_variables.append(TARGET_VARIABLE)
                 final_variables = sorted(list(set(final_variables))) # 去重并排序

            final_predictors = [v for v in final_variables if v != TARGET_VARIABLE]
            final_target = TARGET_VARIABLE # 目标变量名称不变
            print(f"  最终确认变量数: {len(final_variables)} (其中预测变量 {len(final_predictors)})")
            # --- 结束修改 ---

            # 使用更新后的 final_variables 继续后续步骤
            if not final_variables or len(final_predictors) == 0:
                 print("错误: 经过连续缺失值检查后，没有剩余变量或预测变量可用于最终模型。")
                 if log_file: log_file.write("错误: 连续缺失值检查后无剩余变量或预测变量。\n")
                 sys.exit(1)
            
            # --- 修改: 最终运行时使用全时间范围数据 --- 
            # final_data_numeric = all_data_aligned_weekly[final_variables].copy() # 旧: 可能只包含到验证期结束
            # 直接使用 all_data_aligned_weekly 已经包含了所有时间
            final_data_numeric_full_range = all_data_aligned_weekly[final_variables].copy()
            print(f"  使用完整时间范围数据进行最终模型拟合 (原始数据 Shape: {final_data_numeric_full_range.shape})")
            final_data_numeric = final_data_numeric_full_range # 使用新变量名以清晰
            # --- 结束修改 ---
            
            final_data_numeric = final_data_numeric.apply(pd.to_numeric, errors='coerce')

            # --- 修改: 使用变量级转换代替全局 LogYoY --- 
            final_data_processed_transformed, final_transform_details = apply_stationarity_transforms(
                final_data_numeric, final_target
            )
            # --- 结束修改 ---

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
                n_iter=n_iter_to_use, # 使用 n_iter_to_use
                error='False'
            )
            print(f"最终 DFM 模型运行成功 (使用 {n_iter_to_use} 次迭代)。")

            # <-- 新增: 检查最终模型输入和输出时间范围 -->
            try:
                if not final_data_std_masked_for_fit.empty:
                    print(f"  [DEBUG TIME CHECK] Input data (final_data_std_masked_for_fit) end date: {final_data_std_masked_for_fit.index.max()}")
                else:
                    print("  [DEBUG TIME CHECK] Input data (final_data_std_masked_for_fit) is empty!")
                
                if hasattr(final_dfm_results_obj, 'x_sm') and final_dfm_results_obj.x_sm is not None and not final_dfm_results_obj.x_sm.empty:
                    print(f"  [DEBUG TIME CHECK] Smoothed factors (x_sm) end date: {final_dfm_results_obj.x_sm.index.max()}")
                else:
                    print("  [DEBUG TIME CHECK] Smoothed factors (x_sm) are missing or empty!")
            except Exception as e_debug_print:
                print(f"  [DEBUG TIME CHECK] Error printing time range info: {e_debug_print}")
            # <-- 结束新增 -->

            print("\\n[DEBUG POST-FINAL DFM CALL]")
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

    # <-- 移动 PCA 计算块到这里 -->
    # --- 计算 PCA 解释方差 (修改: 不写入，仅计算) ---
    print("\n计算基于最终输入数据的 PCA 解释方差...")
    pca_results_df = None # 保证变量存在
    try:
        # 获取最终用于 DFM 拟合的标准化数据
        if 'final_data_std_for_dfm' in locals() and isinstance(final_data_std_for_dfm, pd.DataFrame) and not final_data_std_for_dfm.empty:
             data_for_pca = final_data_std_for_dfm.copy()
             print(f"  使用 final_data_std_for_dfm (Shape: {data_for_pca.shape}) 进行 PCA 分析。")
        elif 'final_data_std_masked_for_fit' in locals() and isinstance(final_data_std_masked_for_fit, pd.DataFrame) and not final_data_std_masked_for_fit.empty:
             data_for_pca = final_data_std_masked_for_fit.copy()
             print(f"  警告: final_data_std_for_dfm 不可用，回退使用 final_data_std_masked_for_fit (Shape: {data_for_pca.shape}) 进行 PCA 分析。")
        elif 'final_data_processed' in locals() and isinstance(final_data_processed, pd.DataFrame) and not final_data_processed.empty:
             print(f"  警告: 标准化数据不可用，尝试使用处理后的非标准化数据进行 PCA 分析 (Shape: {final_data_processed.shape}).")
             # 需要先标准化
             pca_mean = final_data_processed.mean(skipna=True)
             pca_std = final_data_processed.std(skipna=True)
             pca_std[pca_std == 0] = 1.0
             data_for_pca = (final_data_processed - pca_mean) / pca_std
             print(f"  临时标准化完成。")
        else:
             print("  错误: 找不到合适的标准化或处理后数据进行 PCA 分析。")
             raise ValueError("无法进行 PCA 分析，缺少数据。")

        # 处理缺失值 (使用均值填充)
        print(f"  处理 PCA 输入数据的缺失值 (使用均值填充)... 原始 NaN 数量: {data_for_pca.isna().sum().sum()}")
        if data_for_pca.isna().any().any():
            data_pca_imputed = data_for_pca.fillna(data_for_pca.mean())
            print(f"  填充后 NaN 数量: {data_pca_imputed.isna().sum().sum()}")
        else:
            data_pca_imputed = data_for_pca
            print("  数据无缺失值，无需填充。")

        # 执行 PCA (确保 final_k_factors 已定义)
        if 'final_k_factors' not in locals() or not isinstance(final_k_factors, int) or final_k_factors <= 0:
             print("  错误: 无法执行 PCA，因为最终因子数 (final_k_factors) 无效或未定义。")
             raise ValueError("PCA 需要有效的 final_k_factors。")
        num_components = final_k_factors
        pca = PCA(n_components=num_components)
        print(f"  对填充后的数据执行 PCA (n_components={num_components})...")
        pca.fit(data_pca_imputed)

        explained_variance_ratio_pct = pca.explained_variance_ratio_ * 100
        cumulative_explained_variance_pct = np.cumsum(explained_variance_ratio_pct)

        pca_results_df = pd.DataFrame({
            '主成分 (Principal Component)': [f'PC{i+1}' for i in range(num_components)],
            '解释方差 (%)': explained_variance_ratio_pct,
            '累计解释方差 (%)': cumulative_explained_variance_pct
        })

        print("  PCA 解释方差计算完成:")
        print(pca_results_df.to_string(index=False))

    except Exception as e_pca_main:
        print(f"  计算 PCA 解释方差时发生错误: {e_pca_main}")
        import traceback
        traceback.print_exc()
    # --- 结束 PCA 计算 ---
    # <-- 移动 PCA 计算块结束 -->

    # <-- 移动因子贡献度计算块到这里 -->
    # --- 计算因子对目标变量贡献度 (修改: 不写入，仅计算) ---
    print("\n计算各因子对目标变量的贡献度...")
    contribution_results_df = None # 保证变量存在
    factor_contributions = None # 保证变量存在
    try:
        lambda_target = None
        # 尝试从 final_dfm_results_obj 获取载荷
        if final_dfm_results_obj and hasattr(final_dfm_results_obj, 'Lambda'):
            final_loadings = final_dfm_results_obj.Lambda
            # 需要 final_data_processed 来确定目标变量位置
            if 'final_data_processed' in locals() and final_data_processed is not None and TARGET_VARIABLE in final_data_processed.columns:
                try:
                    target_var_index_pos = final_data_processed.columns.get_loc(TARGET_VARIABLE)
                    if final_loadings is not None and target_var_index_pos < final_loadings.shape[0]:
                        lambda_target = final_loadings[target_var_index_pos, :]
                        print(f"  成功提取目标变量 '{TARGET_VARIABLE}' 的载荷向量用于贡献度计算。")
                    else:
                        print(f"  错误: 无法在最终载荷矩阵中定位目标变量索引 ({target_var_index_pos}) 或载荷矩阵为空。")
                except (KeyError, IndexError, AttributeError) as e_lambda:
                    print(f"  提取目标变量载荷时出错: {e_lambda}")
            else:
                print("  警告: 无法确定目标变量在最终载荷中的位置，因为 final_data_processed 不可用。")
        else:
             print("  警告: 无法提取目标载荷，最终模型结果或其载荷矩阵不可用。")

        if lambda_target is not None and 'final_k_factors' in locals() and final_k_factors > 0:
            lambda_target_sq = lambda_target ** 2
            sum_lambda_target_sq = np.sum(lambda_target_sq)
            
            if sum_lambda_target_sq > 1e-9:
                pct_contribution_common = (lambda_target_sq / sum_lambda_target_sq) * 100
            else:
                pct_contribution_common = np.zeros_like(lambda_target_sq) * np.nan
                print("  警告: 目标变量的平方载荷和过小，无法计算对共同方差的百分比贡献。")
            
            pct_contribution_total = lambda_target_sq * 100

            contribution_results_df = pd.DataFrame({
                '因子 (Factor)': [f'Factor{i+1}' for i in range(final_k_factors)],
                '载荷 (Loading)': lambda_target,
                '平方载荷 (Loading^2)': lambda_target_sq,
                '对共同方差贡献 (%)': pct_contribution_common,
                '对总方差贡献(近似 %)': pct_contribution_total
            })
            contribution_results_df = contribution_results_df.sort_values(by='对总方差贡献(近似 %)', ascending=False)

            print("  各因子对目标变量方差贡献度计算完成:")
            print(contribution_results_df.to_string(index=False, float_format="%.4f"))
            print(f"  目标变量共同度 (Communality): {sum_lambda_target_sq:.4f}")
            
            factor_contributions = contribution_results_df.set_index('因子 (Factor)')['对总方差贡献(近似 %)'].to_dict()
            print(f"[DEBUG] Factor contributions prepared: {factor_contributions}")
        elif lambda_target is None:
            print("  未能成功提取目标载荷，跳过贡献度计算。")
        else: # final_k_factors 无效
            print("  错误: 最终因子数无效，无法计算贡献度。")

    except Exception as e_contrib_main:
        print(f"  计算因子对目标变量贡献度时发生错误: {e_contrib_main}")
        import traceback
        traceback.print_exc()
    # --- 结束因子贡献度计算 ---
    # <-- 移动因子贡献度计算块结束 -->

    # --- 新增: 调用 Bai & Ng IC 计算 --- 
    # Block related to Bai & Ng IC calculation removed as requested.
    # --- 结束 IC 计算调用 ---

    if final_dfm_results_obj is not None and data_for_analysis is not None: 
        try:
            print("\n调用最终结果分析与保存函数...")
            # --- **再次确认**: 传递正确的 run_output_dir 和 timestamp_str --- 
            print(f"[DEBUG] Passing to analyze_and_save_final_results:")
            print(f"  run_output_dir = '{run_output_dir}'") # 打印传入的目录
            print(f"  timestamp_str = '{timestamp_str}'") # 打印传入的时间戳
            # --- 修改: 传递 IC 推荐的 k 值 --- 
            analyze_and_save_final_results(
                run_output_dir=run_output_dir,
                timestamp_str=timestamp_str,
                excel_output_path=excel_output_file,
                all_data_full=all_data_aligned_weekly,
                data_for_analysis=data_for_analysis,
                target_variable=TARGET_VARIABLE,
                final_dfm_results=final_dfm_results_obj,
                best_variables=final_variables,
                best_params=final_params,
                var_type_map=var_type_map,
                best_avg_hit_rate_tuning=final_hr,
                best_avg_mae_tuning=-final_neg_rmse, # 传递对应的最终 RMSE
                total_runtime_seconds=total_runtime_seconds,
                factor_contributions=factor_contributions,
                final_transform_log=final_transform_details,
                pca_results_df=pca_results_df,
                contribution_results_df=contribution_results_df,
                var_industry_map=var_industry_map # 确保这是最后一个传递的参数
            )
            # --- 结束修改 ---
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