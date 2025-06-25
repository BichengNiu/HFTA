# -*- coding: utf-8 -*-
"""
核心 DFM 模型评估功能
"""
import pandas as pd
# import suppress_prints  # 抑制子进程中的重复打印 - 暂时注释掉
import numpy as np
import time
import sys
import os
from typing import Tuple, List, Dict, Union
from sklearn.metrics import mean_squared_error, mean_absolute_error

# === 优化：添加静默控制机制 ===
_SILENT_WARNINGS = os.getenv('DFM_SILENT_WARNINGS', 'true').lower() == 'true'

def _conditional_print(*args, **kwargs):
    """条件化的print函数，可以通过环境变量控制"""
    if not _SILENT_WARNINGS:
        print(*args, **kwargs)

# 假设 apply_stationarity_transforms 在 data_utils.py 中定义
try:
    # from .data_utils import apply_stationarity_transforms # <-- 旧的导入
    # from .DynamicFactorModel import DFM_EMalgo # <-- 旧的导入
    # --- 修改为从 0416 导入 DFM ---
    try:
        from .DynamicFactorModel import DFM_EMalgo # <-- 改为导入新版 DFM
        from .analysis_utils import calculate_metrics_with_lagged_target
    except ImportError:
        from DynamicFactorModel import DFM_EMalgo
        from analysis_utils import calculate_metrics_with_lagged_target
    # <<< 新增：导入精确指标计算函数 >>>
    # <<< 结束新增 >>>
    # --- 保持导入新版 data_utils (如果不需要旧版) ---
    # --- 结束修改 ---
except ImportError as e:
    print(f"错误：导入 dfm_core 依赖失败: {e}")
    # 在实际应用中可能需要更健壮的错误处理
    raise

# --- DFM 评估函数 (修改返回值以包含 lambda_df) ---
def evaluate_dfm_params(
    variables: list[str],
    full_data: pd.DataFrame,
    target_variable: str,
    params: dict, # Now ONLY contains 'k_factors'
    var_type_map: dict, # 假设 var_type_map 仍被需要或将被使用
    validation_start: str,
    validation_end: str,
    target_freq: str, # 假设 target_freq 仍被需要或将被使用
    train_end_date: str,
    target_mean_original: float,
    target_std_original: float,
    max_iter: int = 50,
) -> Tuple[float, float, float, float, float, float, bool, pd.DataFrame | None, pd.DataFrame | None]: # 新增 aligned_df_monthly 返回类型
    """评估给定变量和参数下的 DFM。
    返回: (is_rmse, oos_rmse, is_mae, oos_mae, is_hit_rate, oos_hit_rate, is_svd_error, lambda_df, aligned_df_monthly) 元组。
          如果评估失败或无法计算指标，RMSE/MAE=np.inf, Hit Rate=-np.inf, lambda_df=None, aligned_df_monthly=None
          is_svd_error 指示失败是否由 SVD 不收敛引起。
          lambda_df 是包含因子载荷的 DataFrame，失败时为 None。
    """
    start_time = time.time()
    k_factors = params.get('k_factors', None)
    lambda_df = None # 初始化 lambda_df
    aligned_df_monthly = None # 初始化 aligned_df_monthly
    # --- NEW: 初始化 is_svd_error --- 
    is_svd_error = False # 默认不是 SVD 错误

    # --- 统一返回格式: (inf, inf, inf, inf, -inf, -inf, is_svd_error, None, None) --- 
    FAIL_RETURN = (np.inf, np.inf, np.inf, np.inf, -np.inf, -np.inf, is_svd_error, None, None)

    if k_factors is None:
        _conditional_print("    错误: 参数字典中未提供 'k_factors'。")
        return FAIL_RETURN # 使用统一格式

    n_shocks = k_factors # 假设 n_shocks 等于 k_factors

    try:
        if target_variable not in variables:
             _conditional_print(f"    错误: 目标变量 {target_variable} 不在当前变量列表中(len={len(variables)}): {variables[:5]}...")
             return FAIL_RETURN # 使用统一格式
        predictor_vars = [v for v in variables if v != target_variable]
        if not predictor_vars:
             _conditional_print(f"    错误: 只有目标变量，没有预测变量: {variables}")
             return FAIL_RETURN # 使用统一格式

        # 检查 full_data 是否有效
        if not isinstance(full_data, pd.DataFrame) or full_data.empty:
             _conditional_print("    错误: 传入的 full_data 无效或为空。")
             return FAIL_RETURN # 使用统一格式
        if not isinstance(full_data.index, pd.DatetimeIndex):
             _conditional_print("    错误: full_data 的索引必须是 DatetimeIndex。")
             return FAIL_RETURN # 使用统一格式

        # 截取拟合所需数据段
        try:
            data_for_fitting = full_data.loc[:validation_end].copy()
        except KeyError:
             print(f"    错误: 无法在 full_data 中定位结束日期 '{validation_end}'。可用范围: {full_data.index.min()} 到 {full_data.index.max()}")
             return FAIL_RETURN # 使用统一格式
        if data_for_fitting.empty:
             print(f"    错误: 截取到 '{validation_end}' 的数据为空。")
             return FAIL_RETURN # 使用统一格式

        current_variables = variables.copy()
        if not all(col in data_for_fitting.columns for col in current_variables):
             missing = [col for col in current_variables if col not in data_for_fitting.columns]
             print(f"    错误: 评估函数缺少列: {missing}。可用列: {data_for_fitting.columns.tolist()[:10]}...")
             return FAIL_RETURN # 使用统一格式

        current_data = data_for_fitting[current_variables].copy()
        current_data = current_data.apply(pd.to_numeric, errors='coerce')

        # # 应用平稳性转换
        # current_data_transformed, transform_details = apply_stationarity_transforms(current_data, target_variable)

        # # 使用转换后的数据进行后续处理
        # current_data = current_data_transformed
        
        # <<< 修改: 因为移除了上面的转换, 直接使用 current_data 并更新 current_variables >>>
        current_variables = current_data.columns.tolist() # 变量列表可能因之前的 to_numeric 等操作改变(如果列完全无效)
        if target_variable not in current_variables:
             print(f"    错误: Vars={len(variables)}, n_factors={k_factors} -> 目标变量在数据准备阶段丢失 (可能全为无效值)。")
             return FAIL_RETURN # 使用统一格式
        predictor_vars = [v for v in current_variables if v != target_variable]
        # <<< 结束修改 >>>

        if current_data.empty:
             print(f"    错误: Vars={len(current_variables)}, n_factors={k_factors} -> 数据准备阶段后数据为空") # 修改错误消息
             return FAIL_RETURN # 使用统一格式

        # 移除全 NaN 列
        all_nan_cols = current_data.columns[current_data.isna().all()].tolist()
        if all_nan_cols:
            print(f"    警告: Vars={len(current_variables)}, n_factors={k_factors} -> 以下列在数据准备阶段全为 NaN，将被移除: {all_nan_cols}")
            current_variables = [v for v in current_variables if v not in all_nan_cols]
            if target_variable not in current_variables:
                print(f"    错误: 目标变量在移除全 NaN 列后被移除。")
                return FAIL_RETURN # 使用统一格式
            current_data = current_data.drop(columns=all_nan_cols)
            predictor_vars = [v for v in current_variables if v != target_variable]
            if not predictor_vars:
                 print(f"    错误: 移除全 NaN 列后，没有预测变量剩下。")
                 return FAIL_RETURN # 使用统一格式
            if current_data.empty or current_data.shape[1] <= k_factors: 
                print(f"    错误: 移除全 NaN 列后，变量数 ({current_data.shape[1]}) 不足因子数+1 ({k_factors}+1) 或数据为空。")
                return FAIL_RETURN # 使用统一格式

        if current_data.isnull().all().all():
             print(f"    错误: Vars={len(current_variables)}, n_factors={k_factors} -> 数据在移除全 NaN 列后所有值仍为 NaN。")
             return FAIL_RETURN # 使用统一格式

        # --- 新增：移除有效值过少的列 (<=1) --- 
        valid_counts = current_data.count() # 计算每列的非 NaN 值数量
        insufficient_data_cols = valid_counts[valid_counts <= 1].index.tolist()
        if insufficient_data_cols:
            print(f"    警告: Vars={len(current_variables)}, n_factors={k_factors} -> 以下列有效值过少(<=1)，将被移除: {insufficient_data_cols}")
            # 移除列
            current_variables = [v for v in current_variables if v not in insufficient_data_cols]
            if target_variable not in current_variables:
                 print(f"    错误: 目标变量因有效值过少被移除。")
                 return FAIL_RETURN
            current_data = current_data.drop(columns=insufficient_data_cols)
            predictor_vars = [v for v in current_variables if v != target_variable]
            if not predictor_vars:
                 print(f"    错误: 移除有效值过少的列后，没有预测变量剩下。")
                 return FAIL_RETURN
            if current_data.empty or current_data.shape[1] <= k_factors:
                 print(f"    错误: 移除有效值过少的列后，变量数 ({current_data.shape[1]}) 不足因子数+1 ({k_factors}+1) 或数据为空。")
                 return FAIL_RETURN
        # --- 结束移除有效值过少的列 --- 

        # --- 标准化前检查并移除零标准差列 --- 
        obs_mean = current_data.mean(skipna=True)
        obs_std = current_data.std(skipna=True)
        zero_std_cols = obs_std.index[obs_std == 0].tolist()
        
        if zero_std_cols:
            print(f"    警告: Vars={len(current_variables)}, n_factors={k_factors} -> 以下列在标准化前标准差为0，将被移除: {zero_std_cols}")
            # 移除列
            current_variables_before_drop = current_variables[:]
            current_variables = [v for v in current_variables if v not in zero_std_cols]
            if target_variable not in current_variables:
                 print(f"    错误: 目标变量因标准差为0被移除。")
                 return FAIL_RETURN
            current_data = current_data.drop(columns=zero_std_cols)
            predictor_vars = [v for v in current_variables if v != target_variable]
            if not predictor_vars:
                 print(f"    错误: 移除零标准差列后，没有预测变量剩下。")
                 return FAIL_RETURN
            if current_data.empty or current_data.shape[1] <= k_factors:
                 print(f"    错误: 移除零标准差列后，变量数 ({current_data.shape[1]}) 不足因子数+1 ({k_factors}+1) 或数据为空。")
                 return FAIL_RETURN
            # 重新计算均值和标准差
            obs_mean = current_data.mean(skipna=True)
            obs_std = current_data.std(skipna=True)
            # 再次处理 std=0 (理论上不应再出现，但作为保险)
            obs_std[obs_std == 0] = 1.0 
        else:
            # 如果没有零标准差列，仍然要处理可能存在的 std=0 (原始逻辑)
            obs_std[obs_std == 0] = 1.0
        # --- 结束标准化前检查 ---

        # --- 标准化前再次检查并移除全 NaN 列 --- 
        all_nan_cols_before_std = current_data.columns[current_data.isna().all()].tolist()
        if all_nan_cols_before_std:
            print(f"    警告: Vars={len(current_variables)}, n_factors={k_factors} -> 以下列在标准化*前*为全 NaN，将被移除: {all_nan_cols_before_std}")
            # 移除列
            current_variables = [v for v in current_variables if v not in all_nan_cols_before_std]
            if target_variable not in current_variables:
                 print(f"    错误: 目标变量因在标准化前为全 NaN 被移除。")
                 return FAIL_RETURN
            current_data = current_data.drop(columns=all_nan_cols_before_std)
            predictor_vars = [v for v in current_variables if v != target_variable]
            if not predictor_vars:
                 print(f"    错误: 移除标准化前全 NaN 列后，没有预测变量剩下。")
                 return FAIL_RETURN
            if current_data.empty:
                 print(f"    错误: 移除标准化前全 NaN 列后，数据为空。")
                 return FAIL_RETURN
            # 需要重新获取 obs_mean 和 obs_std，因为数据变了
            obs_mean = current_data.mean(skipna=True)
            obs_std = current_data.std(skipna=True)
            obs_std[obs_std == 0] = 1.0 # 处理可能新出现的零标准差
        # --- 结束再次检查全 NaN 列 ---

        # --- 移除 Debugging: 打印特定列的统计信息 --- 
        # debug_col = 'MEG：社会库存：中国（周）'
        # if debug_col in current_data.columns:
        #     print(f"--- DEBUG STATS for '{debug_col}' BEFORE std --- K={k_factors}")
        #     print(f"  Data (Head):") # 分开打印标签
        #     print(current_data[debug_col].head()) # 单独打印 head()
        #     print(f"  Data (Tail):") # 分开打印标签
        #     print(current_data[debug_col].tail()) # 单独打印 tail()
        #     print(f"  Describe:") # 分开打印标签
        #     print(current_data[debug_col].describe()) # 单独打印 describe()
        #     print(f"  Calculated Mean (obs_mean): {obs_mean.get(debug_col)}")
        #     print(f"  Calculated Std (obs_std): {obs_std.get(debug_col)}")
        #     # 使用 np.isclose 处理可能的浮点数比较问题
        #     std_val = obs_std.get(debug_col)
        #     is_near_zero = np.isclose(std_val, 0.0) if pd.notna(std_val) else False
        #     print(f"  Is Std near zero? {is_near_zero}")
        #     print(f"---------------------------------------------")
        # --- 结束移除 Debugging ---

        # 标准化
        current_data_std = (current_data - obs_mean) / obs_std

        # 检查原始目标变量统计数据
        if pd.isna(target_mean_original) or pd.isna(target_std_original) or target_std_original == 0:
             print(f"    错误: 传入的原始目标变量统计量无效 (Mean: {target_mean_original}, Std: {target_std_original})，无法反标准化。")
             return FAIL_RETURN # 使用统一格式

        # 检查标准化后是否有全 NaN 列
        std_all_nan_cols = current_data_std.columns[current_data_std.isna().all()].tolist()
        if std_all_nan_cols:
             print(f"    错误: Vars={len(current_variables)}, n_factors={k_factors} -> 标准化后以下列变为全 NaN: {std_all_nan_cols}")
             return FAIL_RETURN # 使用统一格式

        # 检查行数是否足够
        if current_data_std.shape[0] < k_factors:
              print(f"    错误: Vars={len(current_variables)}, n_factors={k_factors} -> 处理后数据行数 ({current_data_std.shape[0]}) 不足因子数 ({k_factors})。")
              return FAIL_RETURN # 使用统一格式

        # 掩码1月/2月的目标值
        current_data_std_masked_for_fit = current_data_std.copy()
        month_indices = current_data_std_masked_for_fit.index.month
        mask_jan_feb = (month_indices == 1) | (month_indices == 2)
        if target_variable in current_data_std_masked_for_fit.columns:
            target_nan_before_mask = current_data_std_masked_for_fit[target_variable].isna().sum()
            current_data_std_masked_for_fit.loc[mask_jan_feb, target_variable] = np.nan
            target_nan_after_mask = current_data_std_masked_for_fit[target_variable].isna().sum()
        else:
            print(f"    内部错误: 目标变量 {target_variable} 在掩码前丢失。")
            return FAIL_RETURN # 使用统一格式

        # 运行 DFM
        try:
            # 这个 DFM_EMalgo 现在是来自 code0416 的版本
            dfm_results = DFM_EMalgo(
                observation=current_data_std_masked_for_fit,
                n_factors=k_factors,
                n_shocks=n_shocks,
                n_iter=max_iter
                # 注意：旧版 DFM_EMalgo 可能有 error= 参数，但这里调用时
                # 使用的是新 tune_dfm 的调用方式，没有传 error，会使用旧版的默认值 'False'
            )
        except Exception as dfm_e:
            print(f"    错误: Vars={len(current_variables)}, n_factors={k_factors} -> DFM 运行失败: {dfm_e}")
            # We don't know if it was SVD error, so keep is_svd_error=False
            return FAIL_RETURN # Use unified return

        # 检查 DFM 结果
        if (not hasattr(dfm_results, 'x_sm') or dfm_results.x_sm is None or
            not isinstance(dfm_results.x_sm, (pd.DataFrame, pd.Series)) or dfm_results.x_sm.empty or
            not hasattr(dfm_results, 'Lambda') or dfm_results.Lambda is None or
            not isinstance(dfm_results.Lambda, np.ndarray)):
            print(f"    错误: Vars={len(current_variables)}, n_factors={k_factors} -> DFM 结果不完整或类型错误")
            return (np.inf, np.inf, np.inf, np.inf, -np.inf, -np.inf, is_svd_error, None, None) # Use unified return, pass current is_svd_error

        factors_sm = dfm_results.x_sm
        lambda_matrix = dfm_results.Lambda

        # 确保 Lambda 维度正确
        if lambda_matrix.shape[1] != k_factors:
             print(f"    警告: Lambda 矩阵列数 ({lambda_matrix.shape[1]}) 与预期因子数 ({k_factors}) 不符。")
             return (np.inf, np.inf, np.inf, np.inf, -np.inf, -np.inf, is_svd_error, None, None) # Use unified return
        if lambda_matrix.shape[0] != len(current_variables):
             print(f"    警告: Lambda 矩阵行数 ({lambda_matrix.shape[0]}) 与变量数 ({len(current_variables)}) 不符。")
             return (np.inf, np.inf, np.inf, np.inf, -np.inf, -np.inf, is_svd_error, None, None) # Use unified return

        # 创建 lambda_df (成功获取载荷)
        try:
            lambda_df = pd.DataFrame(lambda_matrix, index=current_variables, columns=[f'Factor{i+1}' for i in range(k_factors)])
        except Exception as e_lambda_df:
             print(f"    错误: 创建 Lambda DataFrame 时出错: {e_lambda_df}")
             return (np.inf, np.inf, np.inf, np.inf, -np.inf, -np.inf, is_svd_error, None, None) # Use unified return

        # 提取目标变量载荷
        if target_variable not in lambda_df.index:
            print(f"    错误: Vars={len(current_variables)}, n_factors={k_factors} -> 目标变量 {target_variable} 不在 Lambda 索引中 (可用: {lambda_df.index.tolist()[:5]}...)")
            return (np.inf, np.inf, np.inf, np.inf, -np.inf, -np.inf, is_svd_error, lambda_df, None) # 返回已创建的 lambda_df
        lambda_target = lambda_matrix[lambda_df.index.get_loc(target_variable), :]

        # 计算 Nowcast
        if not isinstance(factors_sm, pd.DataFrame):
             print(f"    错误: DFM 返回的 factors_sm 不是 DataFrame (类型: {type(factors_sm)})")
             return (np.inf, np.inf, np.inf, np.inf, -np.inf, -np.inf, is_svd_error, lambda_df, None) # Use unified return

        try:
            nowcast_standardized = factors_sm.to_numpy() @ lambda_target
            # --- 检查反标准化参数 ---
            if pd.isna(target_std_original) or pd.isna(target_mean_original): # <-- 修正变量名
                 print(f"错误: 目标变量反标准化参数无效 (Mean={target_mean_original}, Std={target_std_original})。") # <-- 修正变量名
                 # 返回包含 lambda_df 但其他为 None 的失败
                 return (np.inf, np.inf, np.inf, np.inf, -np.inf, -np.inf, is_svd_error, lambda_df, None)
            # --- 结束检查 ---
            nowcast_series_orig = pd.Series(nowcast_standardized * target_std_original + target_mean_original, index=factors_sm.index, name='Nowcast_Orig') # <-- 修正变量名
            # print("最终反标准化 Nowcast 计算完成.") # <<< 注释掉
        except Exception as e_nowcast:
             print(f"错误: 计算最终 Nowcast 时出错: {e_nowcast}")
             return (np.inf, np.inf, np.inf, np.inf, -np.inf, -np.inf, is_svd_error, lambda_df, None) # Use unified return

        # --- 恢复：使用原始 target_for_comparison 进行对齐 --- 
        original_target_series_full = full_data[target_variable].copy()
        target_for_comparison = original_target_series_full.dropna()
        common_index = nowcast_series_orig.index.intersection(target_for_comparison.index)
        if common_index.empty:
            # 这个错误现在理论上不应该发生，除非数据本身完全没有重叠
            print(f"    错误: Vars={len(current_variables)}, n_factors={k_factors} -> Nowcast 和原始目标序列没有共同索引") 
            return (np.inf, np.inf, np.inf, np.inf, -np.inf, -np.inf, is_svd_error, lambda_df, None)

        # 对齐 Nowcast 和 Target (使用原始 Target)
        aligned_df = pd.DataFrame({
            'Nowcast_Orig': nowcast_series_orig.loc[common_index],
            target_variable: target_for_comparison.loc[common_index] # 使用原始 Target
        }).dropna()
        # --- 结束恢复 ---

        # 检查对齐后是否有足够数据
        if aligned_df.empty:
            print(f"    错误: 对齐 Nowcast 和 Target 后数据为空 (可能由于 NaN)。")
            return (np.inf, np.inf, np.inf, np.inf, -np.inf, -np.inf, is_svd_error, lambda_df, None)

        # --- 筛选每月最后一个预测值进行评估 --- 
        # (这部分逻辑现在作用于基于原始对齐的 aligned_df，其结果 aligned_df_monthly 实际上不会被最终返回的指标使用，但保留以避免破坏结构)
        aligned_df['YearMonth'] = aligned_df.index.to_period('M')
        last_indices = aligned_df.groupby('YearMonth').apply(lambda x: x.index.max())
        aligned_df_monthly_intermediate = aligned_df.loc[last_indices.values].copy() # 重命名以免混淆
        aligned_df_monthly_intermediate.drop(columns=['YearMonth'], inplace=True)

        if aligned_df_monthly_intermediate.empty:
             print(f"    错误: 筛选每月最后一个预测值后数据为空 (基于原始对齐)。")
             return (np.inf, np.inf, np.inf, np.inf, -np.inf, -np.inf, is_svd_error, lambda_df, None)
        # --- 结束筛选 --- 

        # --- 计算指标 (调用新函数，传递原始序列) --- 
        # print("  调用 calculate_metrics_with_lagged_target 计算指标...") # <<< 注释掉
        metrics_dict, aligned_df_for_metrics = calculate_metrics_with_lagged_target(
            nowcast_series=nowcast_series_orig, # 传递原始 Nowcast
            target_series=original_target_series_full, # 传递包含 NaN 的完整原始 Target
            validation_start=validation_start,
            validation_end=validation_end,
            train_end=train_end_date, # 使用 train_end_date
            target_variable_name=target_variable
        )
        
        # 从返回的字典中提取指标
        is_rmse = metrics_dict.get('is_rmse', np.nan)
        oos_rmse = metrics_dict.get('oos_rmse', np.nan)
        is_mae = metrics_dict.get('is_mae', np.nan)
        oos_mae = metrics_dict.get('oos_mae', np.nan)
        is_hit_rate = metrics_dict.get('is_hit_rate', np.nan)
        oos_hit_rate = metrics_dict.get('oos_hit_rate', np.nan)
        
        # 将新函数返回的对齐 DataFrame 赋值给函数返回值变量
        aligned_df_monthly = aligned_df_for_metrics # <--- 使用新函数返回的正确对齐的 DF

        # --- 最终返回 (成功路径) --- 
        return is_rmse, oos_rmse, is_mae, oos_mae, is_hit_rate, oos_hit_rate, is_svd_error, lambda_df, aligned_df_monthly

    except Exception as e:
        # --- Keep basic error printing, remove traceback print to stderr --- 
        print(f"评估函数 evaluate_dfm_params (k={k_factors}) 发生意外错误: {type(e).__name__}: {e}")
        # 失败路径也需要返回正确数量的值
        return (np.inf, np.inf, np.inf, np.inf, -np.inf, -np.inf, False, None, None) # is_svd_error is False here, lambda_df=None, aligned_df_monthly=None 