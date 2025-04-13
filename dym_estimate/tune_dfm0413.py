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
import concurrent.futures
from tqdm import tqdm
from sklearn.metrics import mean_squared_error, mean_absolute_error
import traceback
from typing import Tuple, List, Dict, Union
import unicodedata
from statsmodels.tsa.stattools import adfuller

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

# --- Constants and Settings ---
INDUSTRY_MODE_ACTIVE = True # <-- NEW: 设置为 True 以运行行业模式
INDUSTRY_NAME_TO_RUN = '钢铁' # <-- NEW: 指定行业名称

EXCEL_DATA_FILE = '经济数据库.xlsx'
TARGET_FREQ = 'W-FRI'
# 以下变量仅在总模式下使用，或作为参考
TARGET_SHEET_NAME_DEFAULT = '工业增加值同比增速-月度'
TARGET_VARIABLE_DEFAULT = '规模以上工业增加值:当月同比'

# --- Adjust output filenames based on mode ---
if INDUSTRY_MODE_ACTIVE:
    run_identifier = f"industry_{INDUSTRY_NAME_TO_RUN}"
else:
    run_identifier = "total_mode"

RESULTS_FILE = f"tuning_results_{run_identifier}.csv"
DETAILED_LOG_FILE = f"调优日志_{run_identifier}.txt"
DFM_RESULT_DIR = os.path.join('dfm_result', run_identifier)
FINAL_FACTOR_FILE = os.path.join(DFM_RESULT_DIR, 'final_factors.png')
FINAL_PLOT_FILE = os.path.join(DFM_RESULT_DIR, 'final_nowcast_comparison.png')
EXCEL_OUTPUT_FILE = os.path.join(DFM_RESULT_DIR, f'dfm_results_{run_identifier}.xlsx')

# Create the specific output directory
os.makedirs(DFM_RESULT_DIR, exist_ok=True)
print(f"输出目录已创建/确认: {DFM_RESULT_DIR}")

START_DATE = '2020-01-01'
END_DATE = '2024-12-31' # 确保 END_DATE 已定义
N_ITER_FIXED = 15

# --- OOS 验证日期 ---
TRAIN_END_DATE = '2024-06-28'
VALIDATION_START_DATE = '2024-07-05'
VALIDATION_END_DATE = '2024-12-27'

# --- 并行计算 ---
MAX_WORKERS = os.cpu_count() - 1 if os.cpu_count() and os.cpu_count() > 1 else 1

# --- 超参数配置 ---
HYPERPARAMS_TO_TUNE = [] # 动态构建

# --- Transformation Switches ---
USE_ADF_TRANSFORM = True
ADF_P_VALUE_THRESHOLD = 0.05
APPLY_LOG_YOY_TO_PREDICTORS = False # <-- Set this based on desired strategy
LOG_YOY_PERIODS = 52
APPLY_DIFF_AFTER_LOGYOY = False # 控制目标变量的差分

# --- NEW: Validate STRICTLY MUTUALLY EXCLUSIVE Transformation Switches --- 
active_switches = [USE_ADF_TRANSFORM, APPLY_LOG_YOY_TO_PREDICTORS, APPLY_DIFF_AFTER_LOGYOY]
if sum(active_switches) > 1:
    print("错误：配置冲突！变换开关必须严格互斥。")
    print("以下三个开关中，最多只能有一个为 True：")
    print(f"  1. USE_ADF_TRANSFORM ({USE_ADF_TRANSFORM}): 基于 ADF 检验结果智能应用 LogYoY 到非平稳预测变量。")
    print(f"  2. APPLY_LOG_YOY_TO_PREDICTORS ({APPLY_LOG_YOY_TO_PREDICTORS}): 强制对所有可对数化预测变量应用 LogYoY。")
    print(f"  3. APPLY_DIFF_AFTER_LOGYOY ({APPLY_DIFF_AFTER_LOGYOY}): 对目标变量应用月度差分。")
    print("请修改配置，确保只有一个开关为 True。")
    sys.exit(1) # Exit if configuration is invalid
elif sum(active_switches) == 0:
    print("警告：所有变换开关均为 False。将不会应用任何自动变换 (LogYoY 或目标差分)。")

print("--- 变换开关配置 (严格互斥) --- ")
if USE_ADF_TRANSFORM:
    print("  激活策略: 使用 ADF 检验驱动预测变量变换。")
elif APPLY_LOG_YOY_TO_PREDICTORS:
    print("  激活策略: 强制对预测变量应用 LogYoY 变换。")
elif APPLY_DIFF_AFTER_LOGYOY:
    print("  激活策略: 仅对目标变量应用月度差分。")
else:
    print("  激活策略: 无自动变换 (所有开关均为 False)。")
# --- END NEW ---

# --- 辅助函数 (apply_log_yoy_transform) ---
def apply_log_yoy_transform(data: pd.DataFrame | pd.Series, periods: int = 52, clip_value: float = 1e-9) -> pd.DataFrame | pd.Series:
    """
    计算数据的对数同比 log(y_t) - log(y_{t-periods})。
    处理非正数，将其替换为小的正数 clip_value。
    """
    if data.empty:
        return data

    original_type = type(data)
    if isinstance(data, pd.Series):
        # Convert Series to DataFrame, handle potential unnamed Series
        data = data.to_frame(name=data.name if data.name is not None else 'value')

    data_log = np.log(data.clip(lower=clip_value))
    data_log_shifted = data_log.shift(periods)
    log_yoy = data_log - data_log_shifted

    if original_type is pd.Series:
        return log_yoy.iloc[:, 0] # Return Series if input was Series
    else:
        return log_yoy

# --- NEW Helper Function: Prepare Data for DFM Scenario ---
def prepare_data_for_dfm_scenario(
    full_data: pd.DataFrame,
    target_variable: str,
    predictor_vars: List[str],
    use_adf: bool,
    adf_threshold: float,
    apply_logyoy_manual: bool,
    logyoy_periods: int,
    apply_target_diff: bool
) -> Tuple[Union[pd.DataFrame, None], Union[Dict[str, str], None], Union[pd.Series, None], Union[pd.Series, None]]:
    """
    Applies transformations and standardization for a specific scenario.

    Returns:
        Tuple containing:
        - Standardized data (DataFrame) or None if processing fails.
        - Applied transformations (dict) or None.
        - Mean used for standardization (Series) or None.
        - Std used for standardization (Series) or None.
    """
    print(f"    --- Scenario Processing ---")
    print(f"    Settings: ADF={use_adf}, ManualLogYoY={apply_logyoy_manual}, TargetDiff={apply_target_diff}")

    fail_return = (None, None, None, None)
    current_data_numeric = full_data.apply(pd.to_numeric, errors='coerce')
    transformed_data = current_data_numeric.copy()
    applied_transforms = {}
    max_nan_rows_eval = 0

    # --- 1. 预测变量变换 ---
    if use_adf:
        print("      使用 ADF 检验确定预测变量变换...")
        for pred_var in predictor_vars:
            applied_transforms[pred_var] = 'adf_stationary' # Default
            series_clean = current_data_numeric[pred_var].dropna()
            if series_clean.empty or len(series_clean) < logyoy_periods + 5:
                applied_transforms[pred_var] = 'adf_too_short_or_empty'
                continue
            try:
                adf_result = adfuller(series_clean)
                p_value = adf_result[1]
                if p_value > adf_threshold:
                    can_log = (series_clean > 0).all()
                    if can_log:
                        try:
                            transformed_data[pred_var] = apply_log_yoy_transform(transformed_data[pred_var], periods=logyoy_periods)
                            applied_transforms[pred_var] = f'adf_log_yoy_{logyoy_periods}'
                            max_nan_rows_eval = max(max_nan_rows_eval, logyoy_periods)
                        except Exception as log_yoy_err:
                            print(f"      警告: 应用 LogYoY 到 '{pred_var}' 时出错: {log_yoy_err}")
                            applied_transforms[pred_var] = 'adf_log_yoy_error'
                    else:
                        applied_transforms[pred_var] = 'adf_non_stationary_non_loggable'
                # else: # Already stationary, keep 'adf_stationary'
            except Exception as adf_err:
                print(f"      警告: 对变量 '{pred_var}' 执行 ADF 检验时出错: {adf_err}. 跳过此变量的 ADF 变换。")
                applied_transforms[pred_var] = 'adf_error'
    elif apply_logyoy_manual: # Only if ADF is OFF
        print(f"      强制应用 log_yoy_{logyoy_periods} 到可对数化的预测变量...")
        for pred_var in predictor_vars:
            series_clean = current_data_numeric[pred_var].dropna()
            if series_clean.empty:
                 applied_transforms[pred_var] = 'manual_empty'
                 continue
            can_log = (series_clean > 0).all()
            if can_log:
                try:
                    transformed_data[pred_var] = apply_log_yoy_transform(transformed_data[pred_var], periods=logyoy_periods)
                    applied_transforms[pred_var] = f'manual_log_yoy_{logyoy_periods}'
                    max_nan_rows_eval = max(max_nan_rows_eval, logyoy_periods)
                except Exception as log_yoy_err:
                    print(f"      警告: 应用 LogYoY 到 '{pred_var}' 时出错: {log_yoy_err}")
                    applied_transforms[pred_var] = 'manual_log_yoy_error'
            else:
                applied_transforms[pred_var] = 'manual_non_loggable'
    else: # Neither ADF nor Manual LogYoY
        print("      未对预测变量应用 LogYoY 变换。")
        for pred_var in predictor_vars:
            applied_transforms[pred_var] = 'none'

    # --- 2. 目标变量变换 ---
    target_orig_aligned = transformed_data[target_variable].copy()
    if apply_target_diff:
        print(f"    应用月度差分到目标变量 {target_variable}...")
        target_monthly = target_orig_aligned.dropna()
        if len(target_monthly) < 2:
             print(f"    错误: 目标变量有效月度观测值不足 ({len(target_monthly)}) 无法计算月度差分。")
             return fail_return
        try:
            target_resampled_monthly = target_monthly.resample('M').last()
            monthly_diff = target_resampled_monthly.diff(1)
            target_final_for_model = pd.Series(np.nan, index=transformed_data.index, name=target_variable)
            for month_end_date, diff_value in monthly_diff.dropna().items():
                 relevant_weeks = transformed_data.index[
                     (transformed_data.index.year == month_end_date.year) &
                     (transformed_data.index.month == month_end_date.month)
                 ]
                 if not relevant_weeks.empty:
                     target_week = relevant_weeks.max()
                     if target_week in target_final_for_model.index: # Check index exists
                        target_final_for_model.loc[target_week] = diff_value
            transformed_data[target_variable] = target_final_for_model
            applied_transforms[target_variable] = "monthly_diff_1"
        except Exception as target_diff_err:
             print(f"    错误: 计算或映射目标变量月度差分时出错: {target_diff_err}")
             return fail_return
    else:
        print(f"    未对目标变量 {target_variable} 应用月度差分。")
        applied_transforms[target_variable] = applied_transforms.get(target_variable, 'none (original)')

    # --- 3. 移除因 *预测变量变换* 引入的初始 NaN 行 ---
    if max_nan_rows_eval > 0:
        if max_nan_rows_eval >= len(transformed_data):
            print(f"    错误: 预测变量变换后数据行数不足 ({len(transformed_data)} < {max_nan_rows_eval+1})。")
            return fail_return
        print(f"    移除前 {max_nan_rows_eval} 行数据。")
        transformed_data = transformed_data.iloc[max_nan_rows_eval:]
    else:
         print("    预测变量变换未引入需要移除的初始 NaN 行。")

    # --- 4. 后续处理 (检查空值、标准化等) ---
    if transformed_data.empty:
         print(f"    错误: 数据在转换和移除初始 NaN 行后为空。")
         return fail_return

    all_nan_cols = transformed_data.columns[transformed_data.isna().all()].tolist()
    if all_nan_cols:
        print(f"    警告: 以下列在转换后全为 NaN，将被移除: {all_nan_cols}")
        current_variables = [v for v in transformed_data.columns if v not in all_nan_cols]
        if target_variable not in current_variables:
            print(f"    错误: 目标变量在移除全 NaN 列后被移除。")
            return fail_return
        transformed_data = transformed_data.drop(columns=all_nan_cols)
        if transformed_data.empty:
            print(f"    错误: 移除全 NaN 列后数据为空。")
            return fail_return

    if transformed_data.isnull().all().all():
         print(f"    错误: 数据在移除全 NaN 列后所有值仍为 NaN。")
         return fail_return

    # --- 标准化数据 (带 NaN) ---
    obs_mean = transformed_data.mean(skipna=True)
    obs_std = transformed_data.std(skipna=True)
    zero_std_cols = obs_std.index[obs_std == 0].tolist()
    if zero_std_cols:
         print(f"    警告: 以下列标准差为零，将使用 1 代替: {zero_std_cols}")
         obs_std[zero_std_cols] = 1.0
    current_data_std = (transformed_data - obs_mean) / obs_std

    std_all_nan_cols = current_data_std.columns[current_data_std.isna().all()].tolist()
    if std_all_nan_cols:
         print(f"    错误: 标准化后以下列变为全 NaN: {std_all_nan_cols}")
         return fail_return

    print("    数据处理和标准化完成。")
    return current_data_std, applied_transforms, obs_mean, obs_std

# --- DFM 评估函数 (evaluate_dfm_params) ---
# (This function remains largely the same for now, but the data prep part is now handled by the helper)
# (We might remove/simplify it later if only saving preprocessed data)
def evaluate_dfm_params(
    variables: list[str],
    full_data: pd.DataFrame, # Now receives the already processed (standardized) data
    target_variable: str,
    params: dict,
    validation_start: str,
    validation_end: str,
    target_freq: str,
    train_end_date: str,
    max_iter: int = 50,
    # --- NEW Params for rescaling ---
    target_mean_for_rescaling: float = np.nan,
    target_std_for_rescaling: float = np.nan,
    original_target_series_full: Union[pd.Series, None] = None, # Pass original target for comparison
    is_target_differenced: bool = False # Flag to know comparison scale
    # --- END NEW ---
) -> Tuple[float, float, float, float, float, bool, Dict[str, str]]: # Applied transforms return empty dict now

    start_time = time.time()
    k_factors = params.get('k_factors', None)
    # Return empty dict for applied_transforms as it's done before calling this
    fail_return_value = (np.inf, np.inf, -np.inf, -np.inf, np.inf, False, {})

    # --- Initialize return metrics ---
    is_rmse, oos_rmse = np.inf, np.inf
    is_hit_rate, oos_hit_rate = -np.inf, -np.inf
    bic, is_svd_error = np.inf, False

    if k_factors is None: return fail_return_value
    n_shocks = k_factors

    try:
        # --- Data is assumed pre-processed (transformed and standardized) ---
        current_data_std_for_dfm = full_data.copy() # Input is already processed

        if current_data_std_for_dfm.empty:
             print("    错误 (Eval): 输入数据为空。")
             return fail_return_value
        if target_variable not in current_data_std_for_dfm.columns:
             print(f"    错误 (Eval): 目标变量 {target_variable} 不在输入数据列中。")
             return fail_return_value

        T_eff = len(current_data_std_for_dfm)
        N_vars = current_data_std_for_dfm.shape[1]

        if N_vars <= k_factors: # Check required *before* DFM call
            print(f"    错误 (Eval): 变量数 ({N_vars}) 小于或等于因子数 ({k_factors})。")
            return fail_return_value # DFM cannot run

        # --- DFM 运行 ---
        dfm_results = DFM_EMalgo(
            observation=current_data_std_for_dfm,
            n_factors=k_factors,
            n_shocks=n_shocks,
            n_iter=max_iter
        )

        # --- Check DFM results and LLF ---
        if dfm_results is None or not hasattr(dfm_results, 'x_sm') or dfm_results.x_sm is None or dfm_results.x_sm.empty or \
           not hasattr(dfm_results, 'Lambda') or dfm_results.Lambda is None or \
           not hasattr(dfm_results, 'log_likelihood'):
            print(f"    错误: Vars={N_vars}, n_factors={k_factors} -> DFM 运行失败或结果不完整/缺少 LLF。")
            return fail_return_value # Return empty transforms dict {}

        log_likelihood = dfm_results.log_likelihood
        if log_likelihood is None or not np.isfinite(log_likelihood):
            print(f"    警告: Vars={N_vars}, n_factors={k_factors} -> DFM 返回无效的 LLF ({log_likelihood})。无法计算 BIC。")
            bic = np.inf
        else:
            k_params = (N_vars * k_factors) + (k_factors**2) + (k_factors * (k_factors + 1) / 2) + N_vars
            if T_eff <= 0:
                 bic = np.inf
            else:
                 bic = -2 * log_likelihood + k_params * np.log(T_eff)

        # --- Nowcasting 和评估 ---
        factors_sm = dfm_results.x_sm
        lambda_matrix = dfm_results.Lambda

        if not isinstance(lambda_matrix, np.ndarray) or lambda_matrix.shape != (N_vars, k_factors):
             print(f"    错误: Lambda 矩阵维度不符。")
             return fail_return_value # Empty transforms dict

        lambda_df = pd.DataFrame(lambda_matrix, index=current_data_std_for_dfm.columns, columns=[f'Factor{i+1}' for i in range(k_factors)])

        if target_variable not in lambda_df.index:
             print(f"    错误: 目标变量不在 Lambda 索引中。")
             return fail_return_value # Empty transforms dict

        lambda_target = lambda_matrix[lambda_df.index.get_loc(target_variable), :]

        # --- Calculate standardized Nowcast ---
        nowcast_standardized = factors_sm.to_numpy() @ lambda_target

        # --- Rescale and Compare ---
        if pd.isna(target_mean_for_rescaling) or pd.isna(target_std_for_rescaling) or original_target_series_full is None:
            print("    错误 (Eval): 缺少反标准化或原始目标序列信息，无法计算指标。")
            return fail_return_value # Empty transforms dict

        nowcast_transformed = pd.Series(nowcast_standardized * target_std_for_rescaling + target_mean_for_rescaling, index=factors_sm.index, name='Nowcast_Transformed')

        # --- Prepare target for comparison based on differencing flag ---
        comparison_scale_label_for_metrics = ""
        if is_target_differenced:
            print("    使用原始数据的月度差分作为比较基准...")
            target_monthly_orig = original_target_series_full.dropna()
            if len(target_monthly_orig) < 2: return fail_return_value # Empty transforms dict
            target_resampled_monthly_orig = target_monthly_orig.resample('M').last()
            target_comparison_diff = target_resampled_monthly_orig.diff(1)
            target_for_comparison = pd.Series(np.nan, index=nowcast_transformed.index)
            for month_end_date, diff_value in target_comparison_diff.dropna().items():
                 relevant_weeks = nowcast_transformed.index[
                     (nowcast_transformed.index.year == month_end_date.year) &
                     (nowcast_transformed.index.month == month_end_date.month)
                 ]
                 if not relevant_weeks.empty:
                     target_week = relevant_weeks.max()
                     if target_week in target_for_comparison.index:
                         target_for_comparison.loc[target_week] = diff_value
            comparison_scale_label_for_metrics = " (月度差分尺度)"
        else:
            print("    使用原始水平的目标数据作为比较基准...")
            target_for_comparison = original_target_series_full.copy()
            comparison_scale_label_for_metrics = " (原始水平)"

        common_index = nowcast_transformed.index.intersection(target_for_comparison.index)
        if common_index.empty: return fail_return_value # Empty transforms dict

        aligned_df = pd.DataFrame({
            'Nowcast': nowcast_transformed.loc[common_index],
            'Target': target_for_comparison.loc[common_index]
        }).dropna()

        if aligned_df.empty: return fail_return_value # Empty transforms dict

        # --- Calculate IS/OOS RMSE and Hit Rate ---
        train_df = aligned_df.loc[:train_end_date]
        if not train_df.empty and len(train_df) > 1:
            # ... (IS RMSE/Hit Rate calculation - remains the same) ...
             try:
                 is_rmse = np.sqrt(mean_squared_error(train_df['Target'], train_df['Nowcast']))
                 changes_df_train = train_df.diff().dropna()
                 if not changes_df_train.empty:
                     correct_direction_train = (np.sign(changes_df_train['Nowcast']) == np.sign(changes_df_train['Target'])) & (changes_df_train['Target'] != 0)
                     non_zero_target_changes_train = (changes_df_train['Target'] != 0).sum()
                     if non_zero_target_changes_train > 0: is_hit_rate = correct_direction_train.sum() / non_zero_target_changes_train * 100
                     else: is_hit_rate = np.nan
             except Exception as e_is: print(f"    计算 IS 指标时出错: {e_is}")

        validation_df = aligned_df.loc[validation_start:validation_end]
        if not validation_df.empty and len(validation_df) > 1:
            # ... (OOS RMSE/Hit Rate calculation - remains the same) ...
            try:
                 oos_rmse = np.sqrt(mean_squared_error(validation_df['Target'], validation_df['Nowcast']))
                 changes_df_val = validation_df.diff().dropna()
                 if not changes_df_val.empty:
                     correct_direction_val = (np.sign(changes_df_val['Nowcast']) == np.sign(changes_df_val['Target'])) & (changes_df_val['Target'] != 0)
                     non_zero_target_changes_val = (changes_df_val['Target'] != 0).sum()
                     if non_zero_target_changes_val > 0: oos_hit_rate = correct_direction_val.sum() / non_zero_target_changes_val * 100
                     else: oos_hit_rate = np.nan
            except Exception as e_oos: print(f"    计算 OOS 指标时出错: {e_oos}")

        elapsed = time.time() - start_time
        # print(f"  [Eval End] k={k_factors}. OOS RMSE: {oos_rmse:.4f}, BIC: {bic:.2f}. Time: {elapsed:.2f}s")
        # Return empty dict {} for applied transforms
        return is_rmse, oos_rmse, is_hit_rate, oos_hit_rate, bic, is_svd_error, {}

    except Exception as e:
        import traceback
        print(f"!!! 在 evaluate_dfm_params 中发生未处理的错误: {e}")
        traceback.print_exc()
        return fail_return_value # Return empty transforms dict {}

# --- 新增：重新创建 analyze_and_save_final_results 函数 --- 
def analyze_and_save_final_results(
    excel_output_file,
    all_data_full, # 原始周度对齐数据 from prepare_data
    data_for_analysis, # 包含 'final_data_processed', 'final_target_mean_rescale', 'final_target_std_rescale'
    target_variable,
    final_dfm_results,
    best_variables,
    best_params, # Now only contains 'k_factors'
    var_type_map,
    best_avg_hit_rate_tuning: float, # 新：胜率优先
    best_avg_rmse_tuning: float,   # <--- 修改: 重命名为 RMSE
    total_runtime_seconds,
    applied_transforms: dict, # 新增: 接收变换信息
    apply_log_yoy_predictors: bool, # 新增: 记录 LogYoY 设置
    apply_diff_after_logyoy: bool # 新增: 记录差分设置
):
    """分析最终模型结果，计算指标，生成图表，并将所有内容保存到 Excel 文件。"""
    # print("\n--- 开始最终结果分析与保存 --- ") # 注释掉

    try:
        # ... initial checks ...
        # print(f"分析参数: k_factors={final_k_factors}, use_log_yoy(predictors)={use_log_yoy}") # 注释掉
        # ... calculations ...
        # print("最终反标准化 Nowcast 计算完成.") # 注释掉
        # print(f"目标序列用于比较 (尺度: {diff_label_for_metrics}) 准备完成。") # 注释掉

        if not final_dfm_results or not hasattr(final_dfm_results, 'x_sm') or not hasattr(final_dfm_results, 'Lambda'):
            print("错误: analyze_and_save_final_results 缺少有效的 final_dfm_results 对象。")
            return

        if not data_for_analysis or \
           'final_data_processed' not in data_for_analysis or \
           'final_target_mean_rescale' not in data_for_analysis or \
           'final_target_std_rescale' not in data_for_analysis:
            print("错误: analyze_and_save_final_results 缺少有效的 data_for_analysis 对象。")
            return

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
            
        # print(f"分析参数: k_factors={final_k_factors}, use_log_yoy(predictors)={use_log_yoy}")

        # --- 修正: 正确获取目标变量的载荷 --- 
        try:
            # 获取目标变量在 DFM 输入数据列中的整数位置
            target_var_index_pos = final_data_processed.columns.get_loc(target_variable)
            # 使用该位置从 loadings 数组中提取目标载荷向量
            lambda_target = final_loadings[target_var_index_pos, :]
        except KeyError:
            print(f"错误: 无法在最终处理数据的列中找到目标变量 '{target_variable}' 的位置。列: {final_data_processed.columns.tolist()}")
            return # 或者引发更具体的错误
        except IndexError:
             print(f"错误: 目标变量位置 ({target_var_index_pos}) 超出 final_loadings 数组的范围 (Shape: {final_loadings.shape})。")
             return

        # 使用正确的载荷向量进行计算
        # --- MODIFIED: Inverse transform to the potentially differenced scale --- 
        nowcast_standardized = final_factors.to_numpy() @ lambda_target
        final_nowcast_transformed = pd.Series(nowcast_standardized * target_std + target_mean, index=final_factors.index, name='Nowcast_Transformed')
        # --- END MODIFICATION ---
        # print("最终反标准化 Nowcast 计算完成.")

        # --- MODIFIED: Prepare target and label based on differencing setting --- 
        original_target_series = all_data_full[target_variable].copy() 
        final_index = final_factors.index # Get the final weekly index from DFM output

        if apply_diff_after_logyoy:
            # Calculate final target for comparison (monthly diff, forward-filled onto weekly index)
            print("  重新计算最终目标序列 (月度差分尺度，前向填充到周度)...")
            target_monthly = original_target_series.dropna()
            if len(target_monthly) < 2:
                 print("错误 (最终分析): 目标变量有效月度观测值不足，无法计算月度差分比较序列。")
                 target_for_comparison = pd.Series(dtype=float) # Empty series
            else:
                 monthly_diff = target_monthly.diff(1)
                 # Create a series with monthly diffs on their original monthly dates
                 # Then reindex to the final weekly index, forward fill, and reindex again to ensure alignment
                 target_diff_on_final_index = pd.Series(np.nan, index=final_index)
                 common_monthly_dates = final_index.intersection(monthly_diff.index)
                 target_diff_on_final_index.loc[common_monthly_dates] = monthly_diff.loc[common_monthly_dates]
                 
                 # Forward fill the monthly difference values
                 target_for_comparison = target_diff_on_final_index.ffill()
                 # Ensure the index matches exactly and drop any NaNs introduced at the beginning by ffill
                 target_for_comparison = target_for_comparison.loc[final_index].dropna() 
                 
            diff_label_for_metrics = " (月度差分尺度)"
            # print(f"目标序列用于比较 (尺度: {diff_label_for_metrics.strip()}) 准备完成。") # Reduced printing
        else:
            # If not differenced, align the original target to the final index
            target_for_comparison = original_target_series.loc[final_index].dropna()
            diff_label_for_metrics = " (原始水平)"
            # print(f"目标序列用于比较 (尺度: {diff_label_for_metrics.strip()}) 准备完成。") # Reduced printing
        print(f"  用于比较的目标序列 ({diff_label_for_metrics.strip()}) 准备完毕. Shape: {target_for_comparison.shape}")
        # --- END MODIFICATION ---

        common_index_final = final_nowcast_transformed.index.intersection(target_for_comparison.index)
        
        # --- 修改: 初始化 final_oos_rmse --- 
        final_oos_rmse = np.nan
        hit_rate_validation = np.nan
        hit_rate_train = np.nan

        if common_index_final.empty:
            print(f"错误: 最终 Nowcast 和目标序列 ({diff_label_for_metrics.strip()}) 没有共同索引。无法计算最终指标。")
        else:
            # --- MODIFIED: Align final Nowcast and Target on comparison scale --- 
            aligned_df_final = pd.DataFrame({
                'Nowcast': final_nowcast_transformed.loc[common_index_final],
                'Target': target_for_comparison.loc[common_index_final]
            }).dropna()
            # --- END MODIFICATION ---

            validation_df_final = aligned_df_final.loc[VALIDATION_START_DATE:VALIDATION_END_DATE]
            if not validation_df_final.empty and len(validation_df_final) > 0:
                # --- 修改: 计算 final_oos_rmse --- 
                final_oos_rmse = np.sqrt(mean_squared_error(validation_df_final['Target'], validation_df_final['Nowcast']))
                # print(f"  最终 OOS RMSE (验证期): {final_oos_rmse:.6f}")

                changes_df_val = validation_df_final.diff().dropna()
                if not changes_df_val.empty and len(changes_df_val) > 0:
                    correct_direction_val = (np.sign(changes_df_val['Nowcast']) == np.sign(changes_df_val['Target'])) & (changes_df_val['Target'] != 0)
                    non_zero_target_changes_val = (changes_df_val['Target'] != 0).sum()
                    if non_zero_target_changes_val > 0:
                         hit_rate_validation = correct_direction_val.sum() / non_zero_target_changes_val * 100
                         # print(f"  验证期 Hit Rate (%): {hit_rate_validation:.2f} (基于 {non_zero_target_changes_val} 个非零变化点)")
                    else:
                         print(f"  验证期目标变量无非零变化，无法计算 Hit Rate。")
                else:
                    print(f"  验证期变化数据不足 (<=1 点)，无法计算 Hit Rate。")
            else:
                # --- 修改: 更新打印信息为 RMSE --- 
                print(f"  验证期 ({VALIDATION_START_DATE} to {VALIDATION_END_DATE}) 数据不足 (0 点)，无法计算最终 OOS RMSE 和 Hit Rate。")

            train_df_final = aligned_df_final.loc[:TRAIN_END_DATE]
            if not train_df_final.empty and len(train_df_final) > 1:
                 changes_df_train = train_df_final.diff().dropna()
                 if not changes_df_train.empty and len(changes_df_train) > 0:
                     correct_direction_train = (np.sign(changes_df_train['Nowcast']) == np.sign(changes_df_train['Target'])) & (changes_df_train['Target'] != 0)
                     non_zero_target_changes_train = (changes_df_train['Target'] != 0).sum()
                     if non_zero_target_changes_train > 0:
                          hit_rate_train = correct_direction_train.sum() / non_zero_target_changes_train * 100
                          # print(f"  训练期 Hit Rate (%): {hit_rate_train:.2f} (基于 {non_zero_target_changes_train} 个非零变化点)")
                     else:
                          print(f"  训练期目标变量无非零变化，无法计算 Hit Rate。")
                 else:
                     print(f"  训练期变化数据为空，无法计算 Hit Rate。")
            else:
                print(f"  训练期 (up to {TRAIN_END_DATE}) 数据不足 (<=1 点)，无法计算 Hit Rate。")

        interpretation_text = (
            f"最终分析总结:\n"
            f"- 最终选择变量数: {len(best_variables)}\n"
            # f"- 预测变量对数同比转换: {'启用' if use_log_yoy else '禁用'}\n" # 移除 LogYoY 相关行
            f"- 最佳平均胜率 (调优): {best_avg_hit_rate_tuning:.2f}%\n"
            f"- 对应平均 RMSE (调优): {best_avg_rmse_tuning:.6f}\n"
            f"- 最终训练期 Hit Rate (%){diff_label_for_metrics}: {hit_rate_train:.2f}\n"
            f"- 最终验证期 RMSE{diff_label_for_metrics}: {final_oos_rmse:.6f}\n"
            f"- 最终验证期 Hit Rate (%){diff_label_for_metrics}: {hit_rate_validation:.2f}\n"
            f"- 总运行时间: {total_runtime_seconds / 60:.2f} 分钟\n"
            f"\n注: \n"
            # f" - 对数同比转换 (Log(Yt) - Log(Yt-52)) {'仅应用于预测变量' if use_log_yoy else '已禁用'}。\n" # 移除 LogYoY 相关行
            # --- MODIFIED: Update notes ---
            f" - 预测变量变换: {applied_transforms.get(best_variables[0] if best_variables else '', 'N/A')} (详见 'Final_Selected_Variables' sheet)。\\n" # Example, better to summarize transform usage
            f" - 目标变量 1/2 月数据: 已包含在模型中。\\n"
            f"- 预测变量 LogYoY(52) 变换: {'启用' if apply_log_yoy_predictors else '禁用'}\n" # 新增
            f"- 预测变量后续差分(1): {'启用' if apply_diff_after_logyoy else '禁用'}\n" # 新增
            f"- 目标变量月度差分(1): {'启用' if apply_diff_after_logyoy else '禁用'}\n" # 新增
            # --- END MODIFIED ---
            f"- RMSE 和 Hit Rate 报告基于 {diff_label_for_metrics.strip()} 尺度。" 
        )

        # print(f"将结果写入 Excel 文件: {excel_output_file}...") # 注释掉
        os.makedirs(os.path.dirname(excel_output_file), exist_ok=True)
        try:
            with pd.ExcelWriter(excel_output_file, engine='openpyxl') as writer:
                summary_sheet_name = 'Summary_Overview'
                try:
                    # --- 修改: 更新 summary_df 为 RMSE --- 
                    summary_df = pd.DataFrame({
                        'Parameter': [
                            'Final Variables Count',
                            'Best k_factors (Tuned)',
                            # 'Use LogYoY Transform (Predictors)', # 移除
                            'Predictor Transform Method', # 新增
                            'Target 1/2 Month Data Included', # 新增
                            'Best Avg Hit Rate (Tuning %)',
                            'Corresponding Avg RMSE (Tuning)',
                            f'Hit Rate (Train %){diff_label_for_metrics}',
                            f"Final OOS RMSE{diff_label_for_metrics}",
                            f'Hit Rate (Validation %){diff_label_for_metrics}',
                            'Total Runtime (s)'
                            ],
                        'Value': [
                            len(best_variables),
                            final_k_factors,
                            # use_log_yoy, # 移除
                            # --- MODIFIED: Update transform description ---
                            applied_transforms.get(best_variables[0] if best_variables else '', 'N/A'), # Example value
                            # --- END MODIFIED ---
                            'Yes', # Target 1/2 month data included
                            f"{best_avg_hit_rate_tuning:.2f}" if pd.notna(best_avg_hit_rate_tuning) else "N/A",
                            f"{best_avg_rmse_tuning:.6f}" if pd.notna(best_avg_rmse_tuning) else "N/A",
                            f"{hit_rate_train:.2f}" if pd.notna(hit_rate_train) else "N/A",
                            f"{final_oos_rmse:.6f}" if pd.notna(final_oos_rmse) else "N/A",
                            f"{hit_rate_validation:.2f}" if pd.notna(hit_rate_validation) else "N/A",
                            f"{total_runtime_seconds:.2f}"
                        ]
                    })
                    summary_df['Value'] = summary_df['Value'].astype(str)
                    summary_df.to_excel(writer, sheet_name=summary_sheet_name, index=False)
                except Exception as e: print(f"写入 {summary_sheet_name} 时出错: {e}")

                analysis_sheet_name = 'Analysis_Text'
                try:
                    analysis_text_df = pd.DataFrame({'Analysis': [interpretation_text]})
                    analysis_text_df.to_excel(writer, sheet_name=analysis_sheet_name, index=False)
                except Exception as e: print(f"写入 {analysis_sheet_name} 时出错: {e}")

                loadings_sheet_name = 'Final_Factor_Loadings'
                try:
                    lambda_df_final = pd.DataFrame(final_loadings, index=final_data_processed.columns, columns=[f'Factor{i+1}' for i in range(final_k_factors)])
                    lambda_df_final.to_excel(writer, sheet_name=loadings_sheet_name, index=True)
                except Exception as e: print(f"写入 {loadings_sheet_name} 时出错: {e}")

                vars_sheet_name = 'Final_Selected_Variables'
                try:
                    vars_df = pd.DataFrame(best_variables, columns=['Variable Name'])
                    # --- 修改: 将待查找的变量名也转为小写并规范化 ---
                    cleaned_var_names_series = vars_df['Variable Name'].astype(str).str.strip()
                    # 应用 NFKC 规范化 + 小写
                    normalized_lowercase_keys = cleaned_var_names_series.apply(lambda x: unicodedata.normalize('NFKC', x).strip().lower())
                    vars_df['Variable Type (From Excel)'] = normalized_lowercase_keys.map(var_type_map).fillna('Unknown')
                    # --- 新增: 添加应用的变换类型列 ---
                    vars_df['Applied Transformation'] = vars_df['Variable Name'].map(applied_transforms).fillna('none (target or error)')
                    # --- 结束新增 ---

                    # --- DEBUG 代码 (如果需要保留，查找键也需要更新) --- 
                    unknown_type_vars = vars_df[vars_df['Variable Type (From Excel)'] == 'Unknown']
                    if not unknown_type_vars.empty:
                        print("\n[DEBUG] Variables assigned 'Unknown' type (after normalization):")
                        for index, row in unknown_type_vars.iterrows():
                            original_name = row['Variable Name']
                            # 使用与查找时相同的规范化方法
                            lookup_key_debug = unicodedata.normalize('NFKC', str(original_name)).strip().lower()
                            print(f"  - Original: '{original_name}' -> Normalized Lookup Key: '{lookup_key_debug}'")
                    # --- 结束 DEBUG --- 
                    
                    # --- 结束修改 ---
                    vars_df.to_excel(writer, sheet_name=vars_sheet_name, index=False)
                except Exception as e:
                     print(f"写入 {vars_sheet_name} 时出错: {e}")

                selected_data_sheet_name = "Selected_Vars_Transformed"
                try:
                    # 保存变换后的数据（如果需要）或原始数据
                    # 注意：这里不再区分 LogYoY
                    # 决定保存哪些数据（原始？变换后？） - 暂时保存原始数据
                    data_to_save_sel = all_data_full[best_variables].copy()

                    # 或者，如果想保存变换后的数据 (需要重新应用变换，可能复杂)
                    # final_data_subset_orig = all_data_full[best_variables].copy().apply(pd.to_numeric, errors='coerce')
                    # transformed_data_to_save = final_data_subset_orig.copy()
                    # for var_name in best_variables:
                    #     transform_type = applied_transforms.get(var_name, 'none')
                    #     if transform_type == 'log_diff_1':
                    #         if (transformed_data_to_save[var_name].dropna() > 0).all():
                    #            transformed_data_to_save[var_name] = np.log(transformed_data_to_save[var_name].clip(lower=1e-9)).diff(1)
                    #     elif transform_type == 'diff_1':
                    #            transformed_data_to_save[var_name] = transformed_data_to_save[var_name].diff(1)
                    # data_to_save_sel = transformed_data_to_save

                    if not data_to_save_sel.empty:
                         data_to_save_sel = data_to_save_sel.replace([np.inf, -np.inf], np.nan).fillna('N/A')
                         if isinstance(data_to_save_sel.index, pd.DatetimeIndex):
                              data_to_save_sel.index = data_to_save_sel.index.strftime('%Y-%m-%d')
                         else:
                              data_to_save_sel.index = data_to_save_sel.index.astype(str)
                         data_to_save_sel.columns = data_to_save_sel.columns.astype(str)
                         data_to_save_sel = data_to_save_sel.astype(str)
                         data_to_save_sel = data_to_save_sel.reset_index()
                         # 使用固定的 Sheet 名称
                         data_to_save_sel.to_excel(writer, sheet_name="Selected_Vars_Original_Level", index=False)
                    else:
                         print(f"警告: 没有数据可保存到 'Selected_Vars_Original_Level' sheet。")
                except Exception as e: print(f"写入 Selected_Vars_Original_Level 时出错: {e}")
                # --- 结束修改 ---

                factors_sheet_name = 'Final_Factors'
                try:
                    if final_factors is not None and not final_factors.empty:
                        factors_to_save = final_factors.copy().fillna('N/A')
                        if isinstance(factors_to_save.index, pd.DatetimeIndex):
                            factors_to_save.index = factors_to_save.index.strftime('%Y-%m-%d')
                        else:
                            factors_to_save.index = factors_to_save.index.astype(str)
                        factors_to_save.columns = factors_to_save.columns.astype(str)
                        factors_to_save.to_excel(writer, sheet_name=factors_sheet_name, index=True)
                    else:
                         print(f"警告: 最终因子 (final_factors) 为空或 None，无法保存 '{factors_sheet_name}' sheet。")
                except Exception as e: print(f"写入 {factors_sheet_name} 时出错: {e}")

                nowcast_compare_sheet_name = 'Nowcast_vs_Target'
                try:
                    if not common_index_final.empty:
                        # --- MODIFIED: Save Nowcast/Target on the comparison scale --- 
                        nowcast_compare_df = pd.DataFrame({
                            f'Target{diff_label_for_metrics}': target_for_comparison.loc[common_index_final],
                            f'Nowcast{diff_label_for_metrics}': final_nowcast_transformed.loc[common_index_final]
                        }).fillna('N/A')
                        # --- END MODIFICATION ---
                        
                        if isinstance(nowcast_compare_df.index, pd.DatetimeIndex):
                            nowcast_compare_df.index = nowcast_compare_df.index.strftime('%Y-%m-%d')
                        else:
                             nowcast_compare_df.index = nowcast_compare_df.index.astype(str)
                        nowcast_compare_df.columns = nowcast_compare_df.columns.astype(str)
                        nowcast_compare_df.to_excel(writer, sheet_name=nowcast_compare_sheet_name, index=True)
                    else:
                         print(f"警告: 无法保存 '{nowcast_compare_sheet_name}' sheet，因为最终 Nowcast 和目标序列没有共同索引。")
                except Exception as e: print(f"写入 {nowcast_compare_sheet_name} 时出错: {e}")

                # --- 新增: 因子贡献度分析 ---
                contributions_sheet_name = 'Factor_Contribution_Analysis'
                try:
                    print(f"  计算因子贡献度...")
                    # final_loadings shape: (n_vars, n_factors)
                    # final_data_processed.columns should match the order of rows in final_loadings
                    var_names_for_loadings = final_data_processed.columns.tolist()
                    if len(var_names_for_loadings) != final_loadings.shape[0]:
                        print(f"  警告: 因子贡献度分析中，变量名数量 ({len(var_names_for_loadings)}) 与载荷矩阵行数 ({final_loadings.shape[0]}) 不匹配！跳过此分析。")
                    else:
                        loadings_sq = final_loadings ** 2
                        total_r2_common = loadings_sq.sum(axis=1) # Sum across factors for each variable
                        
                        # Avoid division by zero if a variable has zero loading on all factors
                        contributions_pct = np.zeros_like(loadings_sq)
                        non_zero_r2_mask = total_r2_common > 1e-9 # Avoid floating point issues
                        contributions_pct[non_zero_r2_mask] = (loadings_sq[non_zero_r2_mask] / total_r2_common[non_zero_r2_mask, np.newaxis]) * 100
                        
                        # Create DataFrame
                        analysis_data = {
                            'Variable Name': var_names_for_loadings
                        }
                        for k in range(final_k_factors):
                            analysis_data[f'Factor {k+1} Loading'] = final_loadings[:, k]
                        analysis_data['Total R2 (Common Variance)'] = total_r2_common
                        for k in range(final_k_factors):
                             analysis_data[f'Factor {k+1} Contribution (%)'] = contributions_pct[:, k]
                        
                        # Find dominant factor
                        dominant_factor_idx = np.argmax(contributions_pct, axis=1)
                        analysis_data['Dominant Factor'] = [f"Factor {idx+1}" for idx in dominant_factor_idx]
                        analysis_data['Dominant Factor Contribution (%)'] = np.max(contributions_pct, axis=1)
                        
                        contribution_df = pd.DataFrame(analysis_data)
                        contribution_df.to_excel(writer, sheet_name=contributions_sheet_name, index=False, float_format="%.4f")
                        print(f"  因子贡献度分析已写入 Sheet: '{contributions_sheet_name}'")
                except Exception as e_contrib:
                     print(f"写入 {contributions_sheet_name} 时出错: {e_contrib}")
                # --- 结束新增 ---

            print("Excel 文件写入尝试完成。")
        except Exception as e_writer: 
            print(f"创建或写入 Excel 文件 '{excel_output_file}' 时发生严重错误: {e_writer}")
        
        # --- MOVED PLOTTING LOGIC HERE ---
        os.makedirs(os.path.dirname(FINAL_PLOT_FILE), exist_ok=True)
        if 'target_for_comparison' in locals() and not target_for_comparison.empty and 'final_nowcast_transformed' in locals() and not final_nowcast_transformed.empty:
             plot_title = f'最终 DFM Nowcast vs 观测值 ({diff_label_for_metrics.strip(" ()")})'
             print(f"  生成最终结果图 (保存至: {FINAL_PLOT_FILE})...") # Add print statement
             plot_final_nowcast(
                final_nowcast_series=final_nowcast_transformed, 
                target_for_plot=target_for_comparison, # Use the target calculated within this function
                validation_start=VALIDATION_START_DATE, 
                validation_end=VALIDATION_END_DATE, 
                best_params=best_params, 
                filename=FINAL_PLOT_FILE, 
                apply_diff=apply_diff_after_logyoy # Pass the differencing flag
            )
        else:
            print("警告: 无法生成最终绘图，因为用于比较的目标序列或 Nowcast 序列为空或未定义。")
        # --- END MOVED PLOTTING LOGIC ---

    except Exception as e_analyze:
        print(f"在 analyze_and_save_final_results 函数中发生意外错误: {e_analyze}")
        import traceback
        traceback.print_exc()

# --- 主逻辑 ---
# USE_LOG_YOY_TRANSFORM = False # This global switch is less relevant now, logic moved inside evaluate

def run_tuning():
    script_start_time = time.time()
    total_evaluations = 0
    svd_error_count = 0
    log_file = None # Initialize log_file

    try:
        log_file = open(DETAILED_LOG_FILE, 'w', encoding='utf-8')
        log_file.write(f"--- 开始详细调优日志 (优化目标: RMSE优先, BIC其次) ---\\n")
    except IOError as e:
        print(f"错误: 无法打开日志文件 {DETAILED_LOG_FILE} 进行写入: {e}")
        log_file = None

    try:
        # --- Call prepare_data based on mode --- 
        current_target_variable = None
        if INDUSTRY_MODE_ACTIVE:
            # ... (industry mode logic - remains the same) ...
            print(f"\\n--- 调用数据准备模块 (行业模式: {INDUSTRY_NAME_TO_RUN}) ---")
            all_data_aligned_weekly, current_target_variable = prepare_data(
                excel_path=EXCEL_DATA_FILE,
                target_freq=TARGET_FREQ,
                mode='industry',
                industry_name=INDUSTRY_NAME_TO_RUN,
                start_date_filter=START_DATE,
                nan_threshold_predictors=0.30,
                exclude_first_sheet=False
            )
        else: # Total Mode
            print("\\n--- 调用数据准备模块 (总模式) ---")
            # --- FIX: Use defined default names for total mode --- 
            all_data_aligned_weekly, current_target_variable = prepare_data(
                excel_path=EXCEL_DATA_FILE,
                target_freq=TARGET_FREQ,
                mode='total',
                target_sheet_name_total=TARGET_SHEET_NAME_DEFAULT, # Use default name
                target_variable_name_total=TARGET_VARIABLE_DEFAULT, # Use default name
                start_date_filter=START_DATE,
                nan_threshold_predictors=0.30,
                exclude_first_sheet=True
            )
            # --- END FIX ---

        if all_data_aligned_weekly is None or all_data_aligned_weekly.empty:
            raise ValueError("数据准备失败，返回了空 DataFrame 或 None。")

        # --- Step 2: Restore var_type_map loading --- 
        print("正在从 Excel 文件加载指标类型映射...")
        try:
            excel_file_obj = pd.ExcelFile(EXCEL_DATA_FILE)
            first_sheet_name = excel_file_obj.sheet_names[0]
            print(f"  使用 Sheet '{first_sheet_name}' 作为指标类型映射来源。")
            indicator_sheet_full = pd.read_excel(excel_file_obj, sheet_name=first_sheet_name) # Load the full sheet first

            # --- MODIFIED: Robust column name finding --- 
            col_indicator_name = next((col for col in indicator_sheet_full.columns if '指标' in str(col).lower()), None)
            col_type_name = next((col for col in indicator_sheet_full.columns if '类型' in str(col).lower()), None)
            # --- NEW: Find '行业' column --- 
            col_industry_name = next((col for col in indicator_sheet_full.columns if '行业' == str(col).strip()), None) # Exact match for '行业'

            if not col_indicator_name or not col_type_name:
                 potential_cols = indicator_sheet_full.columns.tolist()
                 raise ValueError(f"指标类型映射 Sheet ('{first_sheet_name}') 中缺少必需的'指标'或'类型'列。可用列: {potential_cols}")
            print(f"  找到映射列: 指标='{col_indicator_name}', 类型='{col_type_name}'", end="") # Keep print on one line

            # --- NEW: Filter indicator_sheet if in industry mode ---
            indicator_sheet_to_use = indicator_sheet_full # Default to full sheet
            if INDUSTRY_MODE_ACTIVE:
                if not col_industry_name:
                     print("\n    警告: 未在指标类型映射 Sheet 中找到 '行业' 列，无法按行业筛选变量类型！将使用所有类型。")
                     if log_file: log_file.write("警告: 未在指标类型映射 Sheet 中找到 '行业' 列，无法按行业筛选变量类型！将使用所有类型。\\n")
                else:
                    print(f", 行业='{col_industry_name}'") # Complete the print from above
                    # Ensure INDUSTRY_NAME_TO_RUN is compared as string
                    indicator_sheet_to_use = indicator_sheet_full[
                        indicator_sheet_full[col_industry_name].astype(str).str.strip() == str(INDUSTRY_NAME_TO_RUN)
                    ].copy()
                    print(f"    已根据行业 '{INDUSTRY_NAME_TO_RUN}' 筛选指标类型，保留 {len(indicator_sheet_to_use)}/{len(indicator_sheet_full)} 条目。")
                    if log_file: log_file.write(f"已根据行业 '{INDUSTRY_NAME_TO_RUN}' 筛选指标类型，保留 {len(indicator_sheet_to_use)}/{len(indicator_sheet_full)} 条目。\\n")
            else:
                 print("") # End the line if not in industry mode

            # --- END NEW --- 

            # Clean column names before use (from the sheet we decided to use)
            indicator_sheet_to_use.columns = indicator_sheet_to_use.columns.str.strip()
            col_indicator_name = col_indicator_name.strip()
            col_type_name = col_type_name.strip()
            
            # Create the map from the filtered or full sheet
            var_type_map_temp = pd.Series(
                indicator_sheet_to_use[col_type_name].astype(str).str.strip().values,
                index=indicator_sheet_to_use[col_indicator_name].astype(str).str.strip()
            ).to_dict()
            # Normalize keys for the final map
            var_type_map = {unicodedata.normalize('NFKC', str(k)).strip().lower(): str(v).strip()
                            for k, v in var_type_map_temp.items()
                            if pd.notna(k) and str(k).strip().lower() not in ['nan', ''] and pd.notna(v) and str(v).strip().lower() not in ['nan', '']}
        except Exception as e_map:
            raise ValueError(f"加载指标类型映射失败: {e_map}")
        print(f"指标类型映射加载完成，最终包含 {len(var_type_map)} 个有效条目。") # Updated print

        # --- Step 3: Restore dynamic K_FACTORS_RANGE and HYPERPARAMS_TO_TUNE --- 
        print("\\n--- 确定动态因子数范围和超参数 ---") # <-- RESTORED PRINT
        all_variable_names = sorted(all_data_aligned_weekly.columns.tolist())
        initial_num_predictors = len(all_variable_names) - 1 # Exclude target
        max_k = min(10, initial_num_predictors) # Example: Cap K at 10 or num_predictors
        if max_k < 2:
            raise ValueError(f"初始预测变量数量 ({initial_num_predictors}) 太少，无法进行 DFM (至少需要 2)。")
        K_FACTORS_RANGE = list(range(2, max_k + 1))
        print(f"  动态 K 因子范围: {K_FACTORS_RANGE}")
        HYPERPARAMS_TO_TUNE = [{'k_factors': k} for k in K_FACTORS_RANGE]
        print(f"  待调优超参数组合数量: {len(HYPERPARAMS_TO_TUNE)}")

        # --- Step 4: Restore Initial Evaluation (Assuming forward selection or similar) ---
        # This part needs to be adapted based on the original script's logic.
        # Assuming a simple forward selection for now.
        # If it was backward elimination, the logic would be different.
        print("\\n--- 初始化评估 (简单示例: 评估所有超参数) ---") # <-- RESTORED PRINT
        
        best_params_overall = None
        best_variables_overall = all_variable_names.copy() # Start with all variables
        best_avg_rmse_overall = np.inf
        best_avg_hit_rate_overall = -np.inf
        best_bic_overall = np.inf
        best_applied_transforms = {} # Store transforms for the best model

        results_list = []
        with concurrent.futures.ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
            futures = {
                executor.submit(
                    evaluate_dfm_params,
                    variables=best_variables_overall, # Evaluate with all vars initially
                    full_data=all_data_aligned_weekly,
                    target_variable=current_target_variable, # <-- FIX: Use dynamic target variable
                    params=params,
                    validation_start=VALIDATION_START_DATE,
                    validation_end=VALIDATION_END_DATE,
                    target_freq=TARGET_FREQ,
                    train_end_date=TRAIN_END_DATE,
                    max_iter=N_ITER_FIXED # Use fixed iter for tuning
                ): params for params in HYPERPARAMS_TO_TUNE
            }
            
            for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures), desc="初始评估"):
                params = futures[future]
                try:
                    is_rmse, oos_rmse, is_hit_rate, oos_hit_rate, bic, svd_err, applied_transforms = future.result()
                    total_evaluations += 1
                    if svd_err: svd_error_count += 1
                    
                    # Combine metrics (RMSE priority, then BIC)
                    avg_rmse = np.nanmean([is_rmse, oos_rmse]) if np.isfinite(is_rmse) or np.isfinite(oos_rmse) else np.inf
                    avg_hit_rate = np.nanmean([is_hit_rate, oos_hit_rate]) if np.isfinite(is_hit_rate) or np.isfinite(oos_hit_rate) else -np.inf

                    results_list.append({
                        'k_factors': params['k_factors'],
                        'variables_count': len(best_variables_overall),
                        'avg_rmse': avg_rmse,
                        'avg_hit_rate': avg_hit_rate,
                        'bic': bic,
                        'is_rmse': is_rmse,
                        'oos_rmse': oos_rmse,
                        'is_hit_rate': is_hit_rate,
                        'oos_hit_rate': oos_hit_rate,
                        'svd_error': svd_err
                    })

                    # Update best based on RMSE first, then BIC
                    if avg_rmse < best_avg_rmse_overall:
                        best_avg_rmse_overall = avg_rmse
                        best_bic_overall = bic
                        best_params_overall = params
                        best_avg_hit_rate_overall = avg_hit_rate
                        best_applied_transforms = applied_transforms
                        print(f"\n  新最佳 (RMSE): k={params['k_factors']}, AvgRMSE={avg_rmse:.4f}, AvgHitRate={avg_hit_rate:.2f}%, BIC={bic:.2f}")
                    elif avg_rmse == best_avg_rmse_overall and bic < best_bic_overall:
                        best_bic_overall = bic
                        best_params_overall = params
                        best_avg_hit_rate_overall = avg_hit_rate
                        best_applied_transforms = applied_transforms
                        print(f"\n  新最佳 (BIC for same RMSE): k={params['k_factors']}, AvgRMSE={avg_rmse:.4f}, AvgHitRate={avg_hit_rate:.2f}%, BIC={bic:.2f}")

                except Exception as exc:
                    print(f"评估 k={params.get('k_factors')} 时生成异常: {exc}")
                    total_evaluations += 1
                    results_list.append({
                        'k_factors': params.get('k_factors'), 'variables_count': len(best_variables_overall),
                        'avg_rmse': np.inf, 'avg_hit_rate': -np.inf, 'bic': np.inf,
                        'is_rmse': np.inf, 'oos_rmse': np.inf, 'is_hit_rate': -np.inf, 'oos_hit_rate': -np.inf,
                        'svd_error': False
                    })
        
        # Save initial results
        if results_list:
            results_df_initial = pd.DataFrame(results_list)
            results_df_initial.sort_values(by=['avg_rmse', 'bic'], inplace=True)
            # Maybe save to a separate initial results file if needed
            # results_df_initial.to_csv("initial_evaluation_results.csv", index=False, encoding='utf-8-sig') 
            print(f"\n初始评估完成。最佳初始参数: {best_params_overall}，最佳平均 RMSE: {best_avg_rmse_overall:.4f}")
        else:
             raise ValueError("初始评估未能产生任何结果。")

        # --- Step 5: Restore Tuning Loop (Example: Forward Selection) ---
        # NOTE: The original script might have used backward elimination. 
        # This is a placeholder for a FORWARD selection logic. Adjust if needed.
        print("\\n--- 开始逐步前向变量选择 (示例) ---") 
        
        current_best_variables = [current_target_variable] # Start with only the target
        remaining_predictors = [v for v in all_variable_names if v != current_target_variable]
        
        # Re-evaluate the best k based on just the target (usually not meaningful, maybe skip?)
        # Or better: Start forward selection with the previously found best_params_overall

        if best_params_overall is None:
             raise ValueError("初始评估未能确定最佳参数，无法继续前向选择。")
             
        # Evaluate the starting point (only target) - might give inf RMSE
        # is_rmse_start, oos_rmse_start, _, _, bic_start, _, _ = evaluate_dfm_params(...)
        # current_best_avg_rmse = np.nanmean([is_rmse_start, oos_rmse_start])
        # current_best_bic = bic_start
        # Assume the initial best (with all variables) is the benchmark to beat FORWARD
        current_best_avg_rmse = best_avg_rmse_overall 
        current_best_bic = best_bic_overall
        print(f"  基准 RMSE (使用全部变量和最佳 k={best_params_overall['k_factors']}): {current_best_avg_rmse:.4f}")

        # Forward selection loop
        while remaining_predictors:
            best_predictor_to_add = None
            improvement_found = False
            results_this_step = []

            with concurrent.futures.ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
                futures_fs = {}
                for predictor in remaining_predictors:
                    candidate_vars = current_best_variables + [predictor]
                    futures_fs[executor.submit(
                        evaluate_dfm_params,
                        variables=candidate_vars,
                        full_data=all_data_aligned_weekly,
                        target_variable=current_target_variable, # <-- FIX: Use dynamic target variable
                        params=best_params_overall, # Use the best K found initially
                        validation_start=VALIDATION_START_DATE,
                        validation_end=VALIDATION_END_DATE,
                        target_freq=TARGET_FREQ,
                        train_end_date=TRAIN_END_DATE,
                        max_iter=N_ITER_FIXED
                    )] = predictor
                
                for future in tqdm(concurrent.futures.as_completed(futures_fs), total=len(futures_fs), desc=f"前向选择 (当前 {len(current_best_variables)} vars)"):
                     predictor_tried = futures_fs[future]
                     try:
                         is_rmse, oos_rmse, is_hit_rate, oos_hit_rate, bic, svd_err, applied_transforms_step = future.result()
                         total_evaluations += 1
                         if svd_err: svd_error_count += 1
                         
                         avg_rmse = np.nanmean([is_rmse, oos_rmse]) if np.isfinite(is_rmse) or np.isfinite(oos_rmse) else np.inf
                         avg_hit_rate = np.nanmean([is_hit_rate, oos_hit_rate]) if np.isfinite(is_hit_rate) or np.isfinite(oos_hit_rate) else -np.inf
                         
                         results_this_step.append({
                            'added_predictor': predictor_tried,
                            'variables_count': len(current_best_variables) + 1,
                             'k_factors': best_params_overall['k_factors'],
                             'avg_rmse': avg_rmse, 'avg_hit_rate': avg_hit_rate, 'bic': bic,
                             'svd_error': svd_err
                         })

                         # Check for improvement (lower RMSE, or same RMSE and lower BIC)
                         if avg_rmse < current_best_avg_rmse:
                             improvement_found = True
                             current_best_avg_rmse = avg_rmse
                             current_best_bic = bic
                             best_predictor_to_add = predictor_tried
                             best_applied_transforms = applied_transforms_step # Update transforms
                             best_avg_hit_rate_overall = avg_hit_rate # Update hit rate too
                             print(f"\\n    改进: 添加 '{predictor_tried}' -> 新 RMSE={avg_rmse:.4f}, BIC={bic:.2f}")
                         elif avg_rmse == current_best_avg_rmse and bic < current_best_bic:
                             improvement_found = True
                             current_best_bic = bic
                             best_predictor_to_add = predictor_tried
                             best_applied_transforms = applied_transforms_step # Update transforms
                             best_avg_hit_rate_overall = avg_hit_rate # Update hit rate too
                             print(f"\\n    改进 (BIC): 添加 '{predictor_tried}' -> 相同 RMSE, 新 BIC={bic:.2f}")

                     except Exception as exc_fs:
                        print(f"评估添加 '{predictor_tried}' 时出错: {exc_fs}")
                        total_evaluations += 1
                        results_this_step.append({
                             'added_predictor': predictor_tried, 'variables_count': len(current_best_variables) + 1,
                             'k_factors': best_params_overall['k_factors'], 'avg_rmse': np.inf, 
                             'avg_hit_rate': -np.inf, 'bic': np.inf, 'svd_error': False
                        })

            if improvement_found and best_predictor_to_add:
                print(f"  => 添加变量: {best_predictor_to_add}")
                current_best_variables.append(best_predictor_to_add)
                remaining_predictors.remove(best_predictor_to_add)
                best_variables_overall = current_best_variables.copy() # Update overall best vars
                best_avg_rmse_overall = current_best_avg_rmse # Update overall best RMSE
                # Save intermediate results if needed
                if results_this_step:
                    step_results_df = pd.DataFrame(results_this_step)
                    step_results_df.to_csv(f"tuning_step_{len(current_best_variables)}_vars.csv", index=False, encoding='utf-8-sig')
            else:
                print("\\n--- 前向选择完成：本轮未找到显著改进。---")
                break # No improvement found in this iteration

        if not remaining_predictors:
            print("\\n--- 前向选择完成：已尝试所有变量。---")

        # Final selected variables and parameters
        print(f"\\n--- 最终模型选择结果 --- ")
        print(f"最终变量数量: {len(best_variables_overall)}")
        print(f"最终选择的变量: {best_variables_overall}")
        print(f"最终最佳参数 (k_factors): {best_params_overall}")
        print(f"最终最佳平均 RMSE: {best_avg_rmse_overall:.6f}")
        print(f"最终最佳平均 Hit Rate: {best_avg_hit_rate_overall:.2f}%")
        print(f"最终最佳 BIC: {best_bic_overall:.2f}")

        # === REMOVE TEMPORARY MODIFICATION END ===

        # --- Existing code for final run (Should now use dynamically found best values) --- 
        print("\\n--- 使用最终参数重新运行 DFM 模型并进行分析 --- ") # 更新阶段信息
        final_dfm_obj = None
        data_for_analysis = None

        # --- 添加: 重新准备最终模型的数据 --- 
        print("  准备最终模型输入数据...")
        # --- MODIFIED: Use best_variables_overall found during tuning --- 
        final_data_subset = all_data_aligned_weekly[best_variables_overall].copy() 
        # --- END MODIFICATION ---
        final_data_subset = final_data_subset.apply(pd.to_numeric, errors='coerce')

        # --- MODIFIED: Re-apply the BEST transforms found during tuning ---
        # The transformations should ideally be reapplied here based on 'best_applied_transforms'
        # For simplicity, assuming prepare_data handles transforms OR reusing evaluate logic
        # Re-running evaluation logic fragment for final data prep:
        final_transformed_data = final_data_subset.copy()
        final_max_nan_rows = 0
        final_predictors = [v for v in best_variables_overall if v != current_target_variable]
        final_applied_transforms_run = {} 

        # Apply LogYoY if it was part of the best transform set
        # Note: This logic assumes transforms were consistent during tuning.
        # A more robust way is to store and reapply the exact transform type for each var.
        temp_apply_log_yoy_final = any('log_yoy' in t for t in best_applied_transforms.values())
        if temp_apply_log_yoy_final:
            print(f"  应用 LogYoY(52) 变换到最终预测变量 (基于最佳模型)...")
            for pred_var in final_predictors:
                # Re-check if log is possible AND if it was applied in the best run
                transform_type = best_applied_transforms.get(pred_var, 'none')
                if 'log_yoy' in transform_type:
                    try:
                        if (final_transformed_data[pred_var].dropna() > 0).all():
                            final_transformed_data[pred_var] = apply_log_yoy_transform(final_transformed_data[pred_var], periods=LOG_YOY_PERIODS)
                            final_applied_transforms_run[pred_var] = transform_type # Store the actual transform used
                            final_max_nan_rows = max(final_max_nan_rows, LOG_YOY_PERIODS)
                        else:
                             # This shouldn't happen if transform was possible during tuning
                             print(f"    警告 (最终运行): 变量 '{pred_var}' 不能 LogYoY，但最佳变换是 {transform_type}?")
                             final_applied_transforms_run[pred_var] = 'none (final run error)'
                    except Exception as log_e:
                         print(f"    警告 (最终运行): 应用 LogYoY 到 '{pred_var}' 时出错: {log_e}")
                         final_applied_transforms_run[pred_var] = 'none (final run error)'
                else:
                    final_applied_transforms_run[pred_var] = transform_type # Store 'none' or 'diff' etc.
            # Handle target transform from best run
            final_applied_transforms_run[current_target_variable] = best_applied_transforms.get(current_target_variable, 'none')
        else:
             # No LogYoY applied in best run
             for var_name in best_variables_overall:
                 final_applied_transforms_run[var_name] = best_applied_transforms.get(var_name, 'none')


        # Apply Diff if it was part of the best transform set
        temp_apply_diff_final = any('diff_1' in t for t in best_applied_transforms.values())
        if temp_apply_diff_final:
             print(f"  应用 diff(1) 变换 (基于最佳模型)...")
             diff_applied_final_predictors = []
             # Apply diff to predictors if needed
             for pred_var in final_predictors:
                 transform_type = best_applied_transforms.get(pred_var, 'none')
                 if 'diff_1' in transform_type:
                     final_transformed_data[pred_var] = final_transformed_data[pred_var].diff(1)
                     diff_applied_final_predictors.append(pred_var)
             # Apply diff to target if needed
             target_transform_type = best_applied_transforms.get(current_target_variable, 'none')
             if 'monthly_diff_1' in target_transform_type: # Specific check for target
                 print(f"    应用月度差分到目标变量 {current_target_variable} (最终运行)...")
                 target_monthly_final = final_transformed_data[current_target_variable].dropna()
                 if len(target_monthly_final) < 2:
                     print(f"    错误: 目标变量有效月度观测值不足 ({len(target_monthly_final)}) 无法计算月度差分。")
                     # Handle error - maybe fall back or raise
                 else:
                     monthly_diff_final = target_monthly_final.diff(1)
                     target_final_for_model_run = pd.Series(np.nan, index=final_transformed_data.index, name=current_target_variable)
                     target_final_for_model_run.loc[monthly_diff_final.index] = monthly_diff_final
                     final_transformed_data[current_target_variable] = target_final_for_model_run
             final_max_nan_rows = max(final_max_nan_rows, 1) # Diff adds at least 1 NaN
        
        # --- END MODIFICATION ---
        
        # Remove initial NaN rows based on ALL transforms applied
        if final_max_nan_rows > 0:
            if final_max_nan_rows >= len(final_transformed_data):
                print("错误 (最终运行): 变换后数据行数不足。")
                raise ValueError("Insufficient data after final transforms")
            print(f"  移除最终数据的前 {final_max_nan_rows} 行。")
            final_transformed_data = final_transformed_data.iloc[final_max_nan_rows:]

        # 移除全 NaN 列 (以防万一)
        final_all_nan_cols = final_transformed_data.columns[final_transformed_data.isna().all()].tolist()
        if final_all_nan_cols:
            print(f"    警告 (最终运行): 以下列在转换后全为 NaN，将被移除: {final_all_nan_cols}")
            final_transformed_data = final_transformed_data.drop(columns=final_all_nan_cols)
            if final_transformed_data.empty:
                 raise ValueError("Data became empty after removing all-NaN columns in final run.")
            # Update variable list if columns were dropped
            best_variables_overall = final_transformed_data.columns.tolist() 
            if current_target_variable not in best_variables_overall:
                 raise ValueError("Target variable removed after dropping all-NaN columns in final run.")

        # 标准化
        final_mean = final_transformed_data.mean(skipna=True)
        final_std = final_transformed_data.std(skipna=True)
        final_zero_std = final_std.index[final_std == 0].tolist()
        if final_zero_std:
            final_std[final_zero_std] = 1.0
        final_data_std = (final_transformed_data - final_mean) / final_std
        
        final_target_mean = final_mean.get(current_target_variable, np.nan)
        final_target_std = final_std.get(current_target_variable, np.nan)
        if pd.isna(final_target_mean) or pd.isna(final_target_std):
             raise ValueError("无法获取最终重新缩放的目标均值/标准差。")

        # 检查标准化后 NaN
        if final_data_std.isna().all().all():
             raise ValueError("最终运行中标准化后数据全为 NaN。")

        # 掩码 1/2 月目标值 (If this logic is still desired for the FINAL run)
        final_data_std_masked = final_data_std.copy()
        # --- OPTIONAL MASKING: Decide if Jan/Feb masking applies to the FINAL model run ---
        # If masking should NOT apply to the final run, comment out this block
        # --- START OPTIONAL MASKING BLOCK ---
        # if isinstance(final_data_std_masked.index, pd.DatetimeIndex):
        #     final_month_indices = final_data_std_masked.index.month
        #     final_target_mask = (final_month_indices == 1) | (final_month_indices == 2)
        #     final_data_std_masked.loc[final_target_mask, current_target_variable] = np.nan
        #     print("    注意：最终模型拟合时仍忽略目标变量的 1/2 月数据。") # Clarify if this happens
        # else:
        #     print("    警告：最终模型索引非日期时间格式，无法按月忽略目标值。")
        # --- END OPTIONAL MASKING BLOCK ---
        # If NO masking for final run, use: final_data_std_masked = final_data_std.copy() 
        
        print(f"  最终模型输入数据准备完成. Shape: {final_data_std_masked.shape}")
        # --- 结束数据准备 ---

        # --- 添加: 运行最终 DFM_EMalgo --- 
        print("  运行最终 DFM_EMalgo...")
        # --- MODIFIED: Use best_params_overall --- 
        if best_params_overall is None or 'k_factors' not in best_params_overall:
             raise ValueError("无法运行最终模型：未找到最佳参数。")
        final_k = best_params_overall['k_factors']
        # --- END MODIFICATION ---
        final_dfm_obj = DFM_EMalgo(
            observation=final_data_std_masked, # Use potentially masked data
            n_factors=final_k,
            n_shocks=final_k, # Assume n_shocks = n_factors
            # --- MODIFIED: Use a potentially higher iteration count for the final model ---
            n_iter=max(N_ITER_FIXED, 100) # Example: Use more iterations for final model
            # --- END MODIFICATION ---
        )
        if final_dfm_obj is None:
             print("错误: 最终 DFM_EMalgo 运行失败。无法进行分析。")
             # Handle error appropriately - maybe skip analysis?
        else:
             print("  最终 DFM_EMalgo 运行成功。")
             # 准备 data_for_analysis
             data_for_analysis = {
                 # --- MODIFIED: Pass the UNMASKED standardized data for analysis --- 
                 'final_data_processed': final_data_std, # Pass the unmasked data for analysis
                 # --- END MODIFICATION ---
                 'final_target_mean_rescale': final_target_mean,
                 'final_target_std_rescale': final_target_std
             }
        # --- 结束最终运行 --- 

        # --- 修改: 传递最终对象和数据给分析函数 --- 
        print("  调用 analyze_and_save_final_results...")
        script_end_time = time.time() # <-- 计算结束时间
        total_runtime_seconds = script_end_time - script_start_time # <-- 计算总时间
        
        # --- MODIFIED: Pass the dynamically found best values --- 
        analyze_and_save_final_results(
            excel_output_file=EXCEL_OUTPUT_FILE,
            all_data_full=all_data_aligned_weekly, # Pass the original loaded data
            data_for_analysis=data_for_analysis, # <-- 传递准备好的数据
            target_variable=current_target_variable, # <-- FIX: Use dynamic target variable
            final_dfm_results=final_dfm_obj, # <-- 传递最终 DFM 对象
            best_variables=best_variables_overall, # Use variables found from tuning
            best_params=best_params_overall, # Use params found from tuning
            var_type_map=var_type_map,
            best_avg_hit_rate_tuning=best_avg_hit_rate_overall, # Use hit rate found from tuning
            best_avg_rmse_tuning=best_avg_rmse_overall, # Use RMSE found from tuning
            total_runtime_seconds=total_runtime_seconds, # <-- 传递计算好的总时间
            applied_transforms=best_applied_transforms, # Use transforms from the best tuning run
            apply_log_yoy_predictors=temp_apply_log_yoy_final, # Reflect actual final run setting
            apply_diff_after_logyoy=temp_apply_diff_final # Reflect actual final run setting
        )
        # --- END MODIFICATION ---

        # --- 移除：绘图代码已移动到 analyze_and_save_final_results --- 
        # ... (Removed plotting code) ...
        # --- 结束移除 --- 

        # --- Exit after saving data ---
        print("\\n--- 所有场景数据已处理并尝试保存 ---")
        print(f"请检查输出目录: {DFM_RESULT_DIR}")
        if log_file:
            log_file.write("\\n--- 数据导出完成 ---\\n")
            script_end_time = time.time()
            total_runtime_seconds = script_end_time - script_start_time
    except Exception as final_run_e:
        print(f"脚本主逻辑执行过程中出错: {final_run_e}") # <-- Updated error message
        import traceback # <-- Add traceback
        traceback.print_exc() # <-- Print traceback

    # --- 移除：重复计算总运行时间 --- 
    # script_end_time = time.time()
    # total_runtime_seconds = script_end_time - script_start_time
    # print(f"\\n总评估次数: {total_evaluations}, SVD 错误次数: {svd_error_count}") # 保留 <-- 删除这行
    # print(f"总耗时: {total_runtime_seconds / 60:.2f} 分钟") # 保留 <-- 删除这行
    print(f"\\n总评估次数: {total_evaluations}, SVD 错误次数: {svd_error_count}") # 移到这里打印

    # --- MODIFIED: Add explicit log file closing logic at the end --- 
    if log_file and not log_file.closed:
        try:
            log_file.write("\n--- 日志结束 ---\n") # Corrected newline
            log_file.close()
            # print(f"详细调优日志已保存到: {DETAILED_LOG_FILE}") # 注释掉
        except Exception as close_e:
            print(f"关闭日志文件时出错: {close_e}") # 保留错误
    elif log_file:
        # If log_file exists but is already closed, just acknowledge
        print(f"日志文件 {DETAILED_LOG_FILE} 在函数末尾检查时已被关闭。")
    # --- END MODIFICATION --- 

# --- 修改绘图函数以移除 use_differencing, 添加参数和过滤 --- 
def plot_final_nowcast(final_nowcast_series, target_for_plot, validation_start, validation_end, best_params, filename, apply_diff):
    """生成最终的 Nowcast 对比图，实际值剔除1/2月数据，标题包含参数。"""
    try:
        common_index_plot = final_nowcast_series.index.intersection(target_for_plot.index)
        if common_index_plot.empty:
            print("错误：无法对齐最终 Nowcast 和目标序列进行绘图。") # 保留错误
            return

        nowcast_col_name = 'Nowcast_Orig' if not apply_diff else 'Nowcast_Transformed' # Use correct source name
        target_col_name = target_for_plot.name if target_for_plot.name is not None else 'Actual'
        if target_col_name == nowcast_col_name:
            target_col_name = 'Observed_Value'

        # --- MODIFIED: Ensure source data aligns with apply_diff flag ---
        # Nowcast source depends on apply_diff
        nowcast_data_for_plot = final_nowcast_series # This is now correct as passed
        plot_df = pd.concat([
            nowcast_data_for_plot.loc[common_index_plot].rename(nowcast_col_name), # Use the name determined above
            target_for_plot.loc[common_index_plot].rename(target_col_name)
        ], axis=1)
        plot_df = plot_df.dropna(subset=[target_col_name])
        # --- END MODIFICATION ---

        if not plot_df.empty:
            plt.figure(figsize=(14, 7))

            # --- MODIFIED: Adjust labels based on differencing --- 
            nowcast_label = '周度 Nowcast (原始水平)'
            actual_label = '观测值 (原始水平)' # Remove ", 剔除1/2月"
            ylabel = '值 (原始水平)'
            if apply_diff:
                nowcast_label = '周度 Nowcast (月度差分尺度)' # Updated label
                actual_label = '观测值 (月度差分尺度)' # Updated label
                ylabel = '值 (月度差分尺度)' # Updated label
            # --- END MODIFICATION ---

            plt.plot(plot_df.index, plot_df[nowcast_col_name], label=nowcast_label, linestyle='-', alpha=0.8, color='blue')

            # 筛选掉 1 月和 2 月的实际值再绘图
            actual_plot_df = plot_df[[target_col_name]].copy()
            if not isinstance(actual_plot_df.index, pd.DatetimeIndex):
                 try:
                     actual_plot_df.index = pd.to_datetime(actual_plot_df.index)
                 except Exception as date_conv_err:
                     print(f"警告：无法将绘图索引转换为日期时间格式，无法过滤 1/2 月数据。错误: {date_conv_err}")

            if isinstance(actual_plot_df.index, pd.DatetimeIndex):
                months_to_exclude = [1, 2]
                actual_plot_df_filtered = actual_plot_df[~actual_plot_df.index.month.isin(months_to_exclude)]
            else:
                actual_plot_df_filtered = actual_plot_df

            plt.plot(actual_plot_df_filtered.index, actual_plot_df_filtered[target_col_name], label=actual_label, marker='o', linestyle='None', markersize=4, color='red')

            try:
                plt.axvspan(pd.to_datetime(validation_start), pd.to_datetime(validation_end), color='yellow', alpha=0.2, label='验证期')
            except Exception as date_err:
                print(f"警告：标记验证期时出错 - {date_err}")

            # 构建包含参数的标题
            k_factors_str = best_params.get('k_factors', 'N/A')
            # --- MODIFIED: Adjust title based on differencing --- 
            scale_label_title = "月度差分尺度" if apply_diff else "原始水平" # Updated label
            title = f"最终 DFM Nowcast vs 观测值 ({scale_label_title}) - k={k_factors_str}"
            # --- END MODIFICATION ---
            plt.title(title)
            plt.xlabel('日期')
            plt.ylabel(ylabel)
            plt.legend()
            plt.grid(True, linestyle='--', alpha=0.6)
            plt.tight_layout()
            plt.savefig(filename)
            plt.close()
            print(f"最终 Nowcasting 图已保存到: {filename}") # 保留完成信息
        else:
                print("错误：对齐后用于绘图的数据为空。") # 保留错误
    except Exception as e:
        print(f"生成或保存最终 Nowcasting 图时出错: {e}") # 保留错误

# --- 主程序入口 --- 
if __name__ == "__main__":
    # print("DEBUG: Calling run_tuning()...") # ADD DEBUG
    run_tuning()

# print("\n--- 脚本结束 ---") # 注释掉这一行 

