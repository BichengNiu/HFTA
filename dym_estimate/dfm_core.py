# -*- coding: utf-8 -*-
"""
核心 DFM 模型评估功能
"""
import pandas as pd
import numpy as np
import time
import sys
from typing import Tuple, List, Dict, Union
from sklearn.metrics import mean_squared_error, mean_absolute_error

# 假设 apply_stationarity_transforms 在 data_utils.py 中定义
try:
    from .data_utils import apply_stationarity_transforms
    # from DynamicFactorModel import DFM_EMalgo # 旧的导入
    from .DynamicFactorModel import DFM_EMalgo # <-- 修改为相对导入
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
) -> Tuple[float, float, float, float, float, float, bool, pd.DataFrame | None]: # 确认返回类型
    """评估给定变量和参数下的 DFM。
    返回: (is_rmse, oos_rmse, is_mae, oos_mae, is_hit_rate, oos_hit_rate, is_svd_error, lambda_df) 元组。
          如果评估失败或无法计算指标，RMSE/MAE=np.inf, Hit Rate=-np.inf, lambda_df=None
          is_svd_error 指示失败是否由 SVD 不收敛引起。
          lambda_df 是包含因子载荷的 DataFrame，失败时为 None。
    """
    start_time = time.time()
    k_factors = params.get('k_factors', None)
    lambda_df = None # 初始化 lambda_df
    # --- NEW: 初始化 is_svd_error --- 
    is_svd_error = False # 默认不是 SVD 错误

    # --- 统一返回格式: (inf, inf, inf, inf, -inf, -inf, is_svd_error, None) --- 
    FAIL_RETURN = (np.inf, np.inf, np.inf, np.inf, -np.inf, -np.inf, is_svd_error, None)

    if k_factors is None:
        print("    错误: 参数字典中未提供 'k_factors'。")
        return FAIL_RETURN # 使用统一格式

    n_shocks = k_factors # 假设 n_shocks 等于 k_factors

    try:
        if target_variable not in variables:
             print(f"    错误: 目标变量 {target_variable} 不在当前变量列表中(len={len(variables)}): {variables[:5]}...")
             return FAIL_RETURN # 使用统一格式
        predictor_vars = [v for v in variables if v != target_variable]
        if not predictor_vars:
             print(f"    错误: 只有目标变量，没有预测变量: {variables}")
             return FAIL_RETURN # 使用统一格式

        # 检查 full_data 是否有效
        if not isinstance(full_data, pd.DataFrame) or full_data.empty:
             print("    错误: 传入的 full_data 无效或为空。")
             return FAIL_RETURN # 使用统一格式
        if not isinstance(full_data.index, pd.DatetimeIndex):
             print("    错误: full_data 的索引必须是 DatetimeIndex。")
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

        # 应用平稳性转换
        current_data_transformed, transform_details = apply_stationarity_transforms(current_data, target_variable)

        # 使用转换后的数据进行后续处理
        current_data = current_data_transformed
        current_variables = current_data.columns.tolist() # 变量列表可能因全 NaN 列被移除而改变
        if target_variable not in current_variables:
             print(f"    错误: Vars={len(variables)}, n_factors={k_factors} -> 目标变量在变量级转换后丢失 (可能全为 NaN)。")
             return FAIL_RETURN # 使用统一格式
        predictor_vars = [v for v in current_variables if v != target_variable]

        if current_data.empty:
             print(f"    错误: Vars={len(current_variables)}, n_factors={k_factors} -> 变量级转换后数据为空")
             return FAIL_RETURN # 使用统一格式

        # 移除全 NaN 列
        all_nan_cols = current_data.columns[current_data.isna().all()].tolist()
        if all_nan_cols:
            print(f"    警告: Vars={len(current_variables)}, n_factors={k_factors} -> 以下列在变量级转换后全为 NaN，将被移除: {all_nan_cols}")
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

        # 标准化
        obs_mean = current_data.mean(skipna=True)
        obs_std = current_data.std(skipna=True)
        zero_std_cols = obs_std.index[obs_std == 0].tolist()
        if zero_std_cols:
             obs_std.loc[zero_std_cols] = 1.0
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
            dfm_results = DFM_EMalgo(
                observation=current_data_std_masked_for_fit,
                n_factors=k_factors,
                n_shocks=n_shocks,
                n_iter=max_iter
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
            return (np.inf, np.inf, np.inf, np.inf, -np.inf, -np.inf, is_svd_error, None) # Use unified return, pass current is_svd_error

        factors_sm = dfm_results.x_sm
        lambda_matrix = dfm_results.Lambda

        # 确保 Lambda 维度正确
        if lambda_matrix.shape[1] != k_factors:
             print(f"    警告: Lambda 矩阵列数 ({lambda_matrix.shape[1]}) 与预期因子数 ({k_factors}) 不符。")
             return (np.inf, np.inf, np.inf, np.inf, -np.inf, -np.inf, is_svd_error, None) # Use unified return
        if lambda_matrix.shape[0] != len(current_variables):
             print(f"    警告: Lambda 矩阵行数 ({lambda_matrix.shape[0]}) 与变量数 ({len(current_variables)}) 不符。")
             return (np.inf, np.inf, np.inf, np.inf, -np.inf, -np.inf, is_svd_error, None) # Use unified return

        # 创建 lambda_df (成功获取载荷)
        try:
            lambda_df = pd.DataFrame(lambda_matrix, index=current_variables, columns=[f'Factor{i+1}' for i in range(k_factors)])
        except Exception as e_lambda_df:
             print(f"    错误: 创建 Lambda DataFrame 时出错: {e_lambda_df}")
             return (np.inf, np.inf, np.inf, np.inf, -np.inf, -np.inf, is_svd_error, None) # Use unified return

        # 提取目标变量载荷
        if target_variable not in lambda_df.index:
            print(f"    错误: Vars={len(current_variables)}, n_factors={k_factors} -> 目标变量 {target_variable} 不在 Lambda 索引中 (可用: {lambda_df.index.tolist()[:5]}...)")
            return (np.inf, np.inf, np.inf, np.inf, -np.inf, -np.inf, is_svd_error, lambda_df) # 返回已创建的 lambda_df
        lambda_target = lambda_matrix[lambda_df.index.get_loc(target_variable), :]

        # 计算 Nowcast
        if not isinstance(factors_sm, pd.DataFrame):
             print(f"    错误: DFM 返回的 factors_sm 不是 DataFrame (类型: {type(factors_sm)})")
             return (np.inf, np.inf, np.inf, np.inf, -np.inf, -np.inf, is_svd_error, lambda_df) # Use unified return

        try:
            nowcast_standardized = factors_sm.to_numpy() @ lambda_target
            nowcast_orig_values = nowcast_standardized * target_std_original + target_mean_original
            nowcast_series_orig = pd.Series(nowcast_orig_values, index=factors_sm.index, name='Nowcast_Orig')
        except Exception as e_nowcast:
            print(f"    错误: 计算 Nowcast 时出错: {e_nowcast}")
            return (np.inf, np.inf, np.inf, np.inf, -np.inf, -np.inf, is_svd_error, lambda_df) # Use unified return

        # 准备比较数据
        original_target_series_full = full_data[target_variable].copy()
        target_for_comparison = original_target_series_full.dropna()

        common_index = nowcast_series_orig.index.intersection(target_for_comparison.index)
        if common_index.empty:
            print(f"    错误: Vars={len(current_variables)}, n_factors={k_factors} -> 反标准化 Nowcast 和目标序列没有共同索引")
            return (np.inf, np.inf, np.inf, np.inf, -np.inf, -np.inf, is_svd_error, lambda_df) # Use unified return

        # 对齐 Nowcast 和 Target
        aligned_df = pd.DataFrame({
            'Nowcast_Orig': nowcast_series_orig.loc[common_index],
            target_variable: target_for_comparison.loc[common_index]
        }).dropna()

        # 检查对齐后是否有足够数据
        if aligned_df.empty:
            print(f"    错误: 对齐 Nowcast 和 Target 后数据为空 (可能由于 NaN)。")
            return (np.inf, np.inf, np.inf, np.inf, -np.inf, -np.inf, is_svd_error, lambda_df) # Use unified return

        # --- 计算指标 ---
        is_rmse = np.inf
        oos_rmse = np.inf
        is_mae = np.inf
        oos_mae = np.inf
        is_hit_rate = -np.inf
        oos_hit_rate = -np.inf

        # Validation Period
        try:
            validation_df = aligned_df.loc[validation_start:validation_end]
            # validation_df = validation_df.dropna() # Dropna AFTER potentially using for plots if needed
            if not validation_df.empty and len(validation_df) > 1:
                oos_rmse = np.sqrt(mean_squared_error(validation_df[target_variable], validation_df['Nowcast_Orig']))
                oos_mae = mean_absolute_error(validation_df[target_variable], validation_df['Nowcast_Orig'])
                # Hit Rate
                changes_df_val = validation_df.diff().dropna()
                if not changes_df_val.empty:
                    correct_direction_val = (np.sign(changes_df_val['Nowcast_Orig']) == np.sign(changes_df_val[target_variable])) & (changes_df_val[target_variable] != 0)
                    non_zero_target_changes_val = (changes_df_val[target_variable] != 0).sum()
                    if non_zero_target_changes_val > 0:
                        oos_hit_rate = correct_direction_val.sum() / non_zero_target_changes_val * 100
                    else:
                        oos_hit_rate = np.nan # Undefined if no non-zero changes
        except KeyError:
            print(f"    警告: 无法定位验证期 ({validation_start} to {validation_end})。跳过 OOS 指标。")
        except Exception as e_oos:
            print(f"    计算 OOS 指标时出错: {e_oos}")

        # Training Period (IS)
        try:
            train_df = aligned_df.loc[:train_end_date]
            # train_df = train_df.dropna() # Dropna AFTER potentially using for plots if needed
            if not train_df.empty and len(train_df) > 1:
                is_rmse = np.sqrt(mean_squared_error(train_df[target_variable], train_df['Nowcast_Orig']))
                is_mae = mean_absolute_error(train_df[target_variable], train_df['Nowcast_Orig'])
                # Hit Rate
                changes_df_train = train_df.diff().dropna()
                if not changes_df_train.empty:
                    correct_direction_train = (np.sign(changes_df_train['Nowcast_Orig']) == np.sign(changes_df_train[target_variable])) & (changes_df_train[target_variable] != 0)
                    non_zero_target_changes_train = (changes_df_train[target_variable] != 0).sum()
                    if non_zero_target_changes_train > 0:
                        is_hit_rate = correct_direction_train.sum() / non_zero_target_changes_train * 100
                    else:
                        is_hit_rate = np.nan # Undefined if no non-zero changes
        except KeyError:
             print(f"    警告: 无法定位训练结束日期 ({train_end_date})。跳过 IS 指标。")
        except Exception as e_is:
            print(f"    计算 IS 指标时出错: {e_is}")

        # --- 最终返回 (成功路径) --- 
        return is_rmse, oos_rmse, is_mae, oos_mae, is_hit_rate, oos_hit_rate, is_svd_error, lambda_df

    except Exception as e:
        # --- Keep basic error printing, remove traceback print to stderr --- 
        print(f"评估函数 evaluate_dfm_params (k={k_factors}) 发生意外错误: {type(e).__name__}: {e}")
        return FAIL_RETURN # Return FAIL_RETURN, is_svd_error is False here 