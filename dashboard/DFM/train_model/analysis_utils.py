# -*- coding: utf-8 -*-
"""
包含 DFM 结果分析相关工具函数的模块，例如 PCA 和因子贡献度计算。
"""
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer # 如果 PCA 需要填充
from typing import Tuple, Dict, Optional, List, Any
import statsmodels.api as sm
from sklearn.metrics import mean_squared_error, mean_absolute_error # 确保导入
from collections import defaultdict
import unicodedata
import logging # Import logging
import time

# Get logger for this module
logger = logging.getLogger(__name__) # <<< 添加获取 logger 实例

def calculate_pca_variance(
    data_standardized: pd.DataFrame,
    n_components: int,
    impute_strategy: str = 'mean'
) -> Optional[pd.DataFrame]:
    """
    计算给定标准化数据的 PCA 解释方差。

    Args:
        data_standardized (pd.DataFrame): 输入的标准化数据 (行为时间，列为变量)。
        n_components (int): 要提取的主成分数量。
        impute_strategy (str): 处理缺失值的策略 ('mean', 'median', 'most_frequent', or None).

    Returns:
        Optional[pd.DataFrame]:
            - pca_results_df: 包含 PCA 结果的 DataFrame (主成分, 解释方差%, 累计解释方差%),
                              如果发生错误或无法计算则返回 None。
    """
    print("\n计算 PCA 解释方差...")
    pca_results_df = None

    try:
        if data_standardized is None or data_standardized.empty:
            print("  错误: 输入的标准化数据为空，无法计算 PCA。")
            return None
        if n_components <= 0:
            print(f"  错误: 无效的主成分数量 ({n_components})。")
            return None
        
        data_for_pca = data_standardized.copy()
        print(f"  使用数据 (Shape: {data_for_pca.shape}) 进行 PCA 分析。")

        # 处理缺失值
        nan_count = data_for_pca.isna().sum().sum()
        data_pca_imputed_array = None # 初始化
        if nan_count > 0:
            if impute_strategy:
                print(f"  处理 PCA 输入数据的缺失值 (共 {nan_count} 个，使用策略: {impute_strategy})...")
                imputer = SimpleImputer(strategy=impute_strategy)
                # 直接获取 NumPy 数组
                data_pca_imputed_array = imputer.fit_transform(data_for_pca)
                print(f"  填充后 NumPy 数组 Shape: {data_pca_imputed_array.shape}")
                # 检查填充后是否仍有 NaN (理论上 SimpleImputer 不会留下)
                if np.isnan(data_pca_imputed_array).sum() > 0:
                    print("  错误: SimpleImputer 填充后仍存在 NaN。PCA 无法进行。")
                    return None
            else:
                print(f"  警告: 数据包含 {nan_count} 个 NaN，但未指定填充策略。PCA 可能失败。")
                data_pca_imputed_array = data_for_pca.to_numpy() # 转换为 NumPy 继续尝试
        else:
            data_pca_imputed_array = data_for_pca.to_numpy() # 无缺失值，直接转 NumPy
            print("  数据无缺失值，无需填充。")

        # 检查最终数组是否有效
        if data_pca_imputed_array is None or data_pca_imputed_array.shape[1] == 0:
            print("  错误: 处理/填充后数据为空或没有列，无法执行 PCA。")
            return None
        if data_pca_imputed_array.shape[1] < n_components:
            print(f"  警告: 处理/填充后数据列数 ({data_pca_imputed_array.shape[1]}) 少于请求的主成分数 ({n_components})。将使用 {data_pca_imputed_array.shape[1]} 作为主成分数。")
            n_components = data_pca_imputed_array.shape[1]
            if n_components == 0:
                 print("  错误: 调整后主成分数为 0。无法执行 PCA。")
                 return None

        # 执行 PCA (在 NumPy 数组上)
        pca = PCA(n_components=n_components, random_state=42)  # 确保PCA结果的一致性
        print(f"  对处理/填充后的数据 (Shape: {data_pca_imputed_array.shape}) 执行 PCA (n_components={n_components})...")
        pca.fit(data_pca_imputed_array)

        explained_variance_ratio_pct = pca.explained_variance_ratio_ * 100
        cumulative_explained_variance_pct = np.cumsum(explained_variance_ratio_pct)

        pca_results_df = pd.DataFrame({
            '主成分 (PC)': [f'PC{i+1}' for i in range(n_components)],
            '解释方差 (%)': explained_variance_ratio_pct,
            '累计解释方差 (%)': cumulative_explained_variance_pct,
            '特征值 (Eigenvalue)': pca.explained_variance_
        })

        print("  PCA 解释方差计算完成:")
        print(pca_results_df.to_string(index=False))

    except Exception as e_pca_main:
        print(f"  计算 PCA 解释方差时发生错误: {e_pca_main}")
        import traceback
        traceback.print_exc()
        pca_results_df = None
        
    logger.debug(f"calculate_pca_variance 返回 pca_results_df 类型: {type(pca_results_df)}")
    return pca_results_df

def calculate_factor_contributions(
    dfm_results: object, 
    data_processed: pd.DataFrame, 
    target_variable: str, 
    n_factors: int
) -> Tuple[Optional[pd.DataFrame], Optional[Dict[str, float]]]:
    """
    计算 DFM 各因子对目标变量方差的贡献度。
    修正：使用 OLS 将原始尺度的目标变量对标准化因子回归，以获得正确的载荷。

    Args:
        dfm_results (object): DFM 模型运行结果对象 (需要包含 x_sm 属性, 即标准化因子)。
        data_processed (pd.DataFrame): DFM 模型输入的处理后数据 (包含原始/平稳化尺度的目标变量)。
        target_variable (str): 目标变量名称。
        n_factors (int): 模型使用的因子数量。

    Returns:
        Tuple[Optional[pd.DataFrame], Optional[Dict[str, float]]]: 
            - contribution_df: 包含各因子贡献度详情的 DataFrame，出错则为 None。
            - factor_contributions: 因子名称到总方差贡献度(%)的字典，出错则为 None。
    """
    print("\n计算各因子对目标变量的贡献度 (修正 OLS 方法)...")
    contribution_df = None
    factor_contributions_dict = None
    
    try:
        # 1. 提取标准化因子和原始目标变量
        if not (dfm_results and hasattr(dfm_results, 'x_sm') and isinstance(dfm_results.x_sm, pd.DataFrame)):
            print("  错误: DFM 结果对象无效或缺少 'x_sm' (标准化因子) 属性。")
            return None, None
        factors_std = dfm_results.x_sm
        print(f"  [DEBUG] Shape of factors_std: {factors_std.shape}")
        print(f"  [DEBUG] Value of n_factors passed to function: {n_factors}")
        # --- 结束调试打印 ---
        if not (data_processed is not None and target_variable in data_processed.columns):
            print("  错误: 'data_processed' 无效或不包含目标变量。")
            return None, None
        target_orig = data_processed[target_variable]

        # 确保因子数量有效 (修正类型检查 - 直接在 if 中修正)
        if not (isinstance(n_factors, (int, np.integer)) and n_factors > 0 and n_factors <= factors_std.shape[1]):
             print(f"  错误: 无效的因子数量 ({n_factors}, 类型: {type(n_factors)}) 或与因子矩阵维度不符 (Shape: {factors_std.shape})。")
             return None, None
        factors_std = factors_std.iloc[:, :n_factors] # 只选择实际使用的因子列

        print(f"  提取到标准化因子 (Shape: {factors_std.shape}) 和目标变量 (Length: {len(target_orig)})。")

        # 2. 对齐数据并处理缺失值以进行 OLS
        # 合并因子和目标变量，按索引对齐
        merged_data = pd.concat([target_orig, factors_std], axis=1).dropna()
        if merged_data.empty:
            print("  错误: 对齐因子和目标变量并移除 NaN 后数据为空，无法进行 OLS。")
            return None, None
            
        target_ols = merged_data[target_variable]
        factors_ols = merged_data[factors_std.columns]
        print(f"  OLS 使用的数据点数: {len(merged_data)}")

        # 添加常数项进行 OLS (因为 target_orig 未中心化)
        factors_ols_with_const = sm.add_constant(factors_ols)

        # 3. 执行 OLS: target_orig ~ const + factors_std
        print("  执行 OLS 回归: 目标变量 ~ 标准化因子...")
        ols_model = sm.OLS(target_ols, factors_ols_with_const)
        ols_results = ols_model.fit()
        
        # 提取因子对应的系数 (排除常数项)
        loadings_orig_scale = ols_results.params.drop('const', errors='ignore').values 
        if len(loadings_orig_scale) != n_factors:
             print(f"  错误: OLS 结果中的系数数量 ({len(loadings_orig_scale)}) 与预期因子数 ({n_factors}) 不匹配。")
             # 尝试从原始结果中按因子名提取？(更复杂，暂时先报错)
             return None, None 
        print(f"  成功从 OLS 提取 {len(loadings_orig_scale)} 个原始尺度载荷。")
        # print(f"  OLS R-squared: {ols_results.rsquared:.4f}") # 可选：打印 R 方

        # 4. 计算贡献度
        loading_sq_orig = loadings_orig_scale ** 2
        communality_orig_approx = np.sum(loading_sq_orig) # 近似共同度 (因子方差=1)
        target_variance_orig = np.nanvar(target_ols) # 使用 OLS 使用的数据计算方差
        
        if target_variance_orig < 1e-9:
            print("  错误: 目标变量在 OLS 数据点上的方差过小，无法计算贡献度。")
            return None, None
            
        pct_contribution_total_orig = (loading_sq_orig / target_variance_orig) * 100

        # 对共同方差的贡献
        if communality_orig_approx > 1e-9:
            pct_contribution_common_orig = (loading_sq_orig / communality_orig_approx) * 100
        else:
            pct_contribution_common_orig = np.zeros_like(loading_sq_orig) * np.nan
            print("  警告: 近似共同度过小，无法计算对共同方差的百分比贡献。")

        # 创建结果 DataFrame
        contribution_df = pd.DataFrame({
            '因子 (Factor)': [f'Factor{i+1}' for i in range(n_factors)],
            '原始尺度载荷 (OLS Coef)': loadings_orig_scale,
            '平方载荷 (原始尺度)': loading_sq_orig,
            '对共同方差贡献 (%)[近似]': pct_contribution_common_orig,
            '对总方差贡献 (%)[近似]': pct_contribution_total_orig
        })
        contribution_df = contribution_df.sort_values(by='对总方差贡献 (%)[近似]', ascending=False)

        print("  各因子对目标变量方差贡献度计算完成 (基于 OLS 原始尺度载荷):")
        print(contribution_df.to_string(index=False, float_format="%.4f"))
        print(f"  目标变量总方差 (OLS样本): {target_variance_orig:.4f}")
        print(f"  近似共同度 (OLS样本, 因子方差=1): {communality_orig_approx:.4f}")
        print(f"  OLS R-squared (总解释方差比例): {ols_results.rsquared:.4f}")
            
        factor_contributions_dict = contribution_df.set_index('因子 (Factor)')['对总方差贡献 (%)[近似]'].to_dict()

    except Exception as e_contrib_main:
        print(f"  计算因子对目标变量贡献度时发生错误: {e_contrib_main}")
        import traceback
        traceback.print_exc()
        contribution_df = None # 确保出错时返回 None
        factor_contributions_dict = None
        
    return contribution_df, factor_contributions_dict 

def calculate_individual_variable_r2(
    dfm_results: object, 
    data_processed: pd.DataFrame, 
    variable_list: List[str], 
    n_factors: int,
    timeout_seconds: int = 120  # 新增超时参数
) -> Optional[Dict[str, pd.DataFrame]]:
    """
    计算每个因子与每个单独变量回归的 R 平方值。

    Args:
        dfm_results (object): DFM 模型运行结果对象 (需要包含 x_sm)。
        data_processed (pd.DataFrame): DFM 模型输入的处理后数据。
        variable_list (List[str]): 要计算 R 平方的变量列表。
        n_factors (int): 模型使用的因子数量。
        timeout_seconds (int): 超时时间（秒），默认120秒

    Returns:
        Optional[Dict[str, pd.DataFrame]]: 一个字典，键是因子名称 (Factor1, ...),
            值是包含 'Variable' 和 'R2' 列的排序 DataFrame。出错则返回 None。
    """
    start_time = time.time()
    
    print("\n计算各因子对单个变量的解释力 (R-squared)...")
    r2_results_by_factor = {}
    
    try:
        # 1. 提取标准化因子
        if not (dfm_results and hasattr(dfm_results, 'x_sm') and isinstance(dfm_results.x_sm, pd.DataFrame)):
            print("  错误: DFM 结果对象无效或缺少 'x_sm' (标准化因子) 属性。")
            return None
        factors_std = dfm_results.x_sm
        print(f"  [DEBUG] Type of factors_std (dfm_results.x_sm): {type(factors_std)}")
        if isinstance(factors_std, pd.DataFrame):
            print(f"  [DEBUG] Shape of factors_std: {factors_std.shape}")
        print(f"  [DEBUG] Value of n_factors passed to function: {n_factors} (type: {type(n_factors)})")

        if not (isinstance(n_factors, (int, np.integer)) and n_factors > 0 and n_factors <= factors_std.shape[1]):
             print(f"  错误: 无效的因子数量 ({n_factors}, 类型: {type(n_factors)}) 或与因子矩阵维度不符 (Shape: {factors_std.shape})。")
             return None
        factors_std = factors_std.iloc[:, :n_factors] # 只选择实际使用的因子列
        factor_names = [f'Factor{i+1}' for i in range(n_factors)]
        factors_std.columns = factor_names # 重命名因子列
        
        # 过滤有效变量，避免不必要的计算
        valid_variables = [var for var in variable_list if var in data_processed.columns]
        print(f"  有效变量数: {len(valid_variables)}/{len(variable_list)}")
        
        # 2. 遍历每个因子
        for factor_idx, factor_name in enumerate(factor_names):
            print(f"  计算 {factor_name} 的 R-squared...")
            factor_series = factors_std[factor_name]
            factor_r2_list = []
            
            # 3. 遍历每个变量（加入进度监控）
            for var_idx, var in enumerate(valid_variables):
                # 检查超时
                elapsed_time = time.time() - start_time
                if elapsed_time > timeout_seconds:
                    print(f"    ⚠️ 计算超时 ({elapsed_time:.1f}s > {timeout_seconds}s)，停止计算")
                    break
                
                # 每10个变量打印一次进度
                if var_idx % 10 == 0 and var_idx > 0:
                    print(f"    进度: {var_idx}/{len(valid_variables)} 变量已处理 ({elapsed_time:.1f}s)")
                
                variable_series = data_processed[var]
                
                # 对齐因子和变量，移除 NaN
                merged = pd.concat([variable_series, factor_series], axis=1).dropna()
                
                # 需要至少两个点进行回归
                if len(merged) < 2:
                    continue
                    
                Y = merged.iloc[:, 0] # Variable
                X = merged.iloc[:, 1] # Factor
                
                # 添加常数项进行 OLS
                X_with_const = sm.add_constant(X)
                
                try:
                    model = sm.OLS(Y, X_with_const)
                    results = model.fit()
                    r_squared = results.rsquared
                    if pd.notna(r_squared):
                        factor_r2_list.append({'Variable': var, 'R2': r_squared})
                except Exception as e_ols:
                    # 静默跳过单个变量的OLS错误
                    pass
                    
            # 4. 排序并存储结果
            if factor_r2_list:
                factor_df = pd.DataFrame(factor_r2_list)
                factor_df.sort_values(by='R2', ascending=False, inplace=True)
                r2_results_by_factor[factor_name] = factor_df
                print(f"    {factor_name}: 计算了 {len(factor_df)} 个变量的 R-squared.")
            else:
                print(f"    {factor_name}: 未能计算任何变量的 R-squared.")
            
            # 检查是否因超时而中断
            if elapsed_time > timeout_seconds:
                print(f"  计算因超时而提前结束")
                break
                
    except Exception as e_main_r2:
        print(f"  计算因子对单个变量 R2 时发生主错误: {e_main_r2}")
        import traceback
        traceback.print_exc()
        return None # 出错时返回 None
    
    total_time = time.time() - start_time
    print(f"  R-squared计算完成，总耗时: {total_time:.1f}秒")
    return r2_results_by_factor

def calculate_metrics_with_lagged_target(
    nowcast_series: pd.Series,
    target_series: pd.Series,
    validation_start: str,
    validation_end: str,
    train_end: str,
    target_variable_name: str = 'Target'
) -> Tuple[Dict[str, Optional[float]], Optional[pd.DataFrame]]:
    """
    计算 IS/OOS RMSE, MAE (基于周度比较) 和 Hit Rate (基于修改后的月度方向一致性)。

    RMSE/MAE 核心逻辑: 将 t 月份的实际目标值与该月内 *所有周* 的预测值进行比较。(保持不变)
    Hit Rate 核心逻辑 (新): 对每个月 m，比较其内部每周预测值 nowcast_w 相对于上月实际值 actual_{m-1} 的方向变化 sign(nowcast_w - actual_{m-1})，
                             是否与本月实际值 actual_m 相对于上月实际值 actual_{m-1} 的方向变化 sign(actual_m - actual_{m-1}) 一致。
                             月度 Hit Rate = 方向一致的周数 / 总周数。然后对月度 Hit Rate 求 IS/OOS 平均。

    Args:
        nowcast_series: 周度 Nowcast 序列 (DatetimeIndex)。
        target_series: 原始月度 Target 序列 (DatetimeIndex 代表实际数据发生的月份)。
                       **重要假设**: target_series 的值是该索引月份的 *实际* 值。
        validation_start: OOS 周期开始日期字符串。
        validation_end: OOS 周期结束日期字符串。
        train_end: IS 周期结束日期字符串。
        target_variable_name: 输出 DataFrame 中目标变量列的名称。

    Returns:
        Tuple 包含:
            - 包含指标的字典: is_rmse, oos_rmse, is_mae, oos_mae (周度计算),
              is_hit_rate, oos_hit_rate (基于新的月度方向一致性计算)。 (计算失败则为 NaN)。
            - 用于 *RMSE/MAE* 计算的周度对齐 DataFrame (aligned_df_weekly)，对齐失败则为 None。
              (注意: 此 DataFrame 不直接用于新的 Hit Rate 计算)。
    """
    metrics = {
        'is_rmse': np.nan, 'oos_rmse': np.nan,
        'is_mae': np.nan, 'oos_mae': np.nan,
        'is_hit_rate': np.nan, 'oos_hit_rate': np.nan
    }
    aligned_df_weekly = None # 用于 RMSE/MAE 计算
    # aligned_df_monthly_for_hit_rate = None # 不再需要这个变量名

    try:
        # --- 输入验证 (基本保持不变) ---
        if nowcast_series is None or nowcast_series.empty:
            logger.error("Error (calc_metrics_new_hr): Nowcast series is empty.")
            return metrics, None
        if target_series is None or target_series.empty:
            logger.error("Error (calc_metrics_new_hr): Target series is empty.")
            return metrics, None
        # 确保索引是 DatetimeIndex
        for series, name in [(nowcast_series, 'nowcast'), (target_series, 'target')]:
            if not isinstance(series.index, pd.DatetimeIndex):
                try:
                    series.index = pd.to_datetime(series.index)
                except Exception as e:
                    logger.error(f"Error (calc_metrics_new_hr): Failed to convert {name} index to DatetimeIndex: {e}")
                    return metrics, None
        # --- 结束输入验证 ---

        # --- 准备时间点 ---
        try:
            oos_start_dt = pd.to_datetime(validation_start)
            oos_end_dt = pd.to_datetime(validation_end) if validation_end else nowcast_series.index.max() # Use nowcast end if None
            is_end_dt = pd.to_datetime(train_end)
        except Exception as e:
            logger.error(f"Error parsing date strings (train_end, validation_start, validation_end): {e}")
            return metrics, None

        # --- 计算 RMSE/MAE (基于周度比较 - 逻辑保持不变) ---
        # --- 准备月度目标数据 (仅用于 RMSE/MAE 的合并) ---
        df_target_monthly_for_rmse = target_series.to_frame(name=target_variable_name).copy()
        df_target_monthly_for_rmse['YearMonth'] = df_target_monthly_for_rmse.index.to_period('M')
        # 处理重复月 (保留最后一个) - 以防万一
        if df_target_monthly_for_rmse['YearMonth'].duplicated().any():
            df_target_monthly_for_rmse = df_target_monthly_for_rmse.groupby('YearMonth').last()
        else:
            df_target_monthly_for_rmse = df_target_monthly_for_rmse.set_index('YearMonth')

        try:
            # 1. 准备周度预测
            df_nowcast_weekly = nowcast_series.to_frame(name='Nowcast').copy()
            df_nowcast_weekly['YearMonth'] = df_nowcast_weekly.index.to_period('M')

            # 2. 合并周度预测和月度目标 (用于 RMSE/MAE)
            aligned_df_weekly = pd.merge(
                df_nowcast_weekly,
                df_target_monthly_for_rmse[[target_variable_name]], # 使用准备好的月度目标
                left_on='YearMonth',
                right_index=True,
                how='left'
            ).drop(columns=['YearMonth']) # YearMonth 列不再需要

            if aligned_df_weekly.empty or aligned_df_weekly[target_variable_name].isnull().all():
                logger.warning("Warning (calc_metrics_new_hr): Weekly alignment for RMSE/MAE resulted in empty data or all NaNs for target.")
                # 继续尝试计算 Hit Rate
            else:
                # 3. 分割 IS/OOS (周度)
                # 使用之前转换好的 datetime 对象
                aligned_is_weekly = aligned_df_weekly[aligned_df_weekly.index <= is_end_dt].dropna()
                aligned_oos_weekly = aligned_df_weekly[(aligned_df_weekly.index >= oos_start_dt) & (aligned_df_weekly.index <= oos_end_dt)].dropna()

                # 4. 计算周度 RMSE/MAE
                if not aligned_is_weekly.empty:
                    metrics['is_rmse'] = np.sqrt(mean_squared_error(aligned_is_weekly[target_variable_name], aligned_is_weekly['Nowcast']))
                    metrics['is_mae'] = mean_absolute_error(aligned_is_weekly[target_variable_name], aligned_is_weekly['Nowcast'])
                    # logger.debug(f"IS RMSE/MAE (weekly, {len(aligned_is_weekly)} pts): RMSE={metrics['is_rmse']:.4f}, MAE={metrics['is_mae']:.4f}")
                if not aligned_oos_weekly.empty:
                    metrics['oos_rmse'] = np.sqrt(mean_squared_error(aligned_oos_weekly[target_variable_name], aligned_oos_weekly['Nowcast']))
                    metrics['oos_mae'] = mean_absolute_error(aligned_oos_weekly[target_variable_name], aligned_oos_weekly['Nowcast'])
                    # logger.debug(f"OOS RMSE/MAE (weekly, {len(aligned_oos_weekly)} pts): RMSE={metrics['oos_rmse']:.4f}, MAE={metrics['oos_mae']:.4f}")
                else:
                    logger.warning("Warning (calc_metrics_new_hr): OOS period has no valid weekly aligned data points after dropna() for RMSE/MAE.")
        except Exception as e_rmse_mae:
             logger.error(f"Error calculating weekly RMSE/MAE: {type(e_rmse_mae).__name__}: {e_rmse_mae}", exc_info=True)
             # 即使 RMSE/MAE 计算失败，也继续尝试计算 Hit Rate


        # --- 计算 Hit Rate (基于新的月度方向一致性逻辑) ---
        try:
            # 1. 准备数据
            nowcast_df = nowcast_series.to_frame('Nowcast').copy()
            nowcast_df['NowcastMonth'] = nowcast_df.index.to_period('M') # 月份周期

            target_df = target_series.to_frame(target_variable_name).copy()
            target_df['TargetMonth'] = target_df.index.to_period('M')
            target_df = target_df.groupby('TargetMonth').last() # 确保每月只有一个目标值
            target_df_lagged = target_df.shift(1) # 获取上个月目标值
            target_df_lagged.columns = [f'{target_variable_name}_Lagged'] # 重命名

            # 2. 合并数据
            # 将本月目标和上月目标合并到周度预测数据中
            merged_hr = pd.merge(
                nowcast_df,
                target_df[[target_variable_name]],
                left_on='NowcastMonth',
                right_index=True,
                how='left' # 保留所有周预测
            )
            merged_hr = pd.merge(
                merged_hr,
                target_df_lagged[[f'{target_variable_name}_Lagged']],
                left_on='NowcastMonth',
                right_index=True,
                how='left'
            )

            # 3. 移除无法计算方向的行
            merged_hr.dropna(subset=['Nowcast', target_variable_name, f'{target_variable_name}_Lagged'], inplace=True)

            if merged_hr.empty:
                logger.warning("Warning (calc_metrics_new_hr): No valid data points after merging and dropping NaNs for Hit Rate calculation.")
            else:
                # 4. 计算方向
                actual_diff = merged_hr[target_variable_name] - merged_hr[f'{target_variable_name}_Lagged']
                predicted_diff = merged_hr['Nowcast'] - merged_hr[f'{target_variable_name}_Lagged']

                # 使用 np.sign 处理 0 值情况 (sign(0)=0)
                actual_direction = np.sign(actual_diff)
                predicted_direction = np.sign(predicted_diff)

                # 5. 判断方向是否一致 (注意: 0 == 0 会被算作命中)
                merged_hr['Hit'] = (actual_direction == predicted_direction).astype(int)

                # 6. 计算月度命中率
                monthly_hit_rate = merged_hr.groupby('NowcastMonth')['Hit'].mean() * 100

                # 7. 分割 IS/OOS (月度)
                # 转换 PeriodIndex 为 DatetimeIndex (月底) 以便比较
                monthly_hit_rate.index = monthly_hit_rate.index.to_timestamp(how='end')

                # 使用之前转换好的 datetime 对象进行分割
                is_monthly_hr = monthly_hit_rate[monthly_hit_rate.index <= is_end_dt]
                oos_monthly_hr = monthly_hit_rate[(monthly_hit_rate.index >= oos_start_dt) & (monthly_hit_rate.index <= oos_end_dt)]

                # 8. 计算平均月度命中率
                if not is_monthly_hr.empty:
                    metrics['is_hit_rate'] = is_monthly_hr.mean()
                    logger.debug(f"IS HitRate (new, avg monthly, {len(is_monthly_hr)} months): {metrics['is_hit_rate']:.2f}%")
                if not oos_monthly_hr.empty:
                    metrics['oos_hit_rate'] = oos_monthly_hr.mean()
                    logger.debug(f"OOS HitRate (new, avg monthly, {len(oos_monthly_hr)} months): {metrics['oos_hit_rate']:.2f}%")
                else:
                    logger.warning("Warning (calc_metrics_new_hr): OOS period has no valid monthly hit rates.")

        except Exception as e_hit_rate:
             logger.error(f"Error calculating new Hit Rate: {type(e_hit_rate).__name__}: {e_hit_rate}", exc_info=True)
             # Hit Rate 将保持 NaN

        # 格式化最终输出字典 (可选)
        metrics_formatted = {k: f"{v:.4f}" if pd.notna(v) else "N/A" for k, v in metrics.items()}
        # logger.info(f"Calculated Metrics (RMSE/MAE weekly, HitRate new monthly): {metrics_formatted}") # <<< 注释掉

    except Exception as e:
        logger.error(f"Error during metrics calculation: {type(e).__name__}: {e}", exc_info=True)
        # 返回初始 NaN 字典和 None DataFrame
        return {k: np.nan for k in metrics}, None

    # 返回包含 NaN 或计算值的指标字典，以及用于 RMSE/MAE 计算的周度对齐 DataFrame
    return metrics, aligned_df_weekly

# <<< 新增函数：计算行业 R2 >>>
def calculate_industry_r2(
    dfm_results: Any,
    data_processed: pd.DataFrame,
    variable_list: List[str],
    var_industry_map: Dict[str, str],
    n_factors: int,
    timeout_seconds: int = 120  # 新增超时参数
) -> Optional[pd.Series]:
    """
    计算估计出的因子对每个行业内变量群体的总体解释力度 (Pooled R²)。

    Args:
        dfm_results: DFM 模型结果对象，需要包含 .x_sm (估计的因子)。
        data_processed: 经过预处理（例如平稳化）的数据，用于 DFM 拟合。
        variable_list: 包含在 data_processed 中并用于最终模型的变量列表。
        var_industry_map: 变量名到行业名的映射字典。
        n_factors: 使用的因子数量。
        timeout_seconds: 超时时间（秒），默认120秒

    Returns:
        一个 Pandas Series，索引为行业名称，值为该行业的 Pooled R²，
        如果无法计算则返回 None。
    """
    start_time = time.time()
    print("\n计算行业 Pooled R²...")
    
    if not hasattr(dfm_results, 'x_sm'):
        print("  错误: DFM 结果对象缺少 'x_sm' (因子) 属性。")
        return None
    if data_processed is None or data_processed.empty:
        print("  错误: 提供的 'data_processed' 为空或 None。")
        return None
    if not variable_list:
        print("  错误: 'variable_list' 为空。")
        return None
    if not var_industry_map:
        print("  警告: 未提供 'var_industry_map'，无法按行业分组。")
        return None
    if n_factors <= 0:
        print(f"  错误: 因子数量 'n_factors' ({n_factors}) 无效。")
        return None

    try:
        factors = dfm_results.x_sm
        if factors.shape[1] != n_factors:
            print(f"  警告: DFM 结果中的因子数量 ({factors.shape[1]}) 与指定的 n_factors ({n_factors}) 不符。将使用结果中的因子数。")

        # 添加常数项用于回归截距
        factors_with_const = sm.add_constant(factors, prepend=True, has_constant='skip')

        # 规范化行业映射的键
        normalized_industry_map = {
            unicodedata.normalize('NFKC', str(k)).strip().lower(): str(v).strip()
            for k, v in var_industry_map.items()
            if pd.notna(k) and str(k).lower() != 'nan' and pd.notna(v) and str(v).lower() != 'nan'
        }
        print(f"  [DEBUG] 规范化后的行业映射数量: {len(normalized_industry_map)}")
        
        # 添加详细的PMI映射调试信息
        print(f"  [DEBUG] 原始行业映射数量: {len(var_industry_map)}")
        print(f"  [DEBUG] 原始映射中的PMI相关条目:")
        original_pmi_entries = {k: v for k, v in var_industry_map.items() if 'pmi' in str(k).lower() or 'PMI' in str(k)}
        if original_pmi_entries:
            for k, v in list(original_pmi_entries.items())[:10]:
                print(f"    原始: '{k}' -> '{v}'")
        else:
            print(f"    原始映射中未发现PMI相关条目")
        
        print(f"  [DEBUG] 规范化后的PMI相关条目:")
        normalized_pmi_entries = {k: v for k, v in normalized_industry_map.items() if 'pmi' in k}
        if normalized_pmi_entries:
            for k, v in list(normalized_pmi_entries.items())[:10]:
                print(f"    规范化: '{k}' -> '{v}'")
        else:
            print(f"    规范化后映射中未发现PMI相关条目")

        # 按行业分组变量
        industry_to_vars = defaultdict(list)
        processed_vars_set = set(data_processed.columns)
        print(f"  [DEBUG] 数据中的变量数量: {len(processed_vars_set)}")
        print(f"  [DEBUG] 要处理的变量列表数量: {len(variable_list)}")
        
        # 添加调试信息：显示前几个变量的映射过程
        unmapped_vars = []
        mapped_vars = []
        
        for var in variable_list:
            # 检查超时
            elapsed_time = time.time() - start_time
            if elapsed_time > timeout_seconds:
                print(f"    ⚠️ 行业分组超时 ({elapsed_time:.1f}s > {timeout_seconds}s)，停止处理")
                break
                
            if var not in processed_vars_set:
                continue # Skip if var not actually in the data used
            lookup_key = unicodedata.normalize('NFKC', str(var)).strip().lower()
            industry = normalized_industry_map.get(lookup_key, "_未知行业_")
            industry_to_vars[industry].append(var)
            
            # 调试信息：记录映射情况
            if industry == "_未知行业_":
                unmapped_vars.append((var, lookup_key))
            else:
                mapped_vars.append((var, lookup_key, industry))
        
        # 显示映射调试信息
        print(f"  [DEBUG] 成功映射的变量数: {len(mapped_vars)}")
        if mapped_vars:
            print(f"  [DEBUG] 前5个成功映射的变量:")
            for i, (var, key, industry) in enumerate(mapped_vars[:5]):
                print(f"    {i+1}. '{var}' -> 规范化: '{key}' -> 行业: '{industry}'")
        
        print(f"  [DEBUG] 未映射的变量数: {len(unmapped_vars)}")
        if unmapped_vars:
            print(f"  [DEBUG] 未映射的变量详情:")
            for i, (var, key) in enumerate(unmapped_vars):
                print(f"    {i+1}. 原名: '{var}' -> 规范化: '{key}'")
                # 检查是否有相似的键
                similar_keys = [k for k in normalized_industry_map.keys() if 'pmi' in k or var.replace('制造业', '').lower().strip() in k]
                if similar_keys:
                    print(f"       可能相似的映射键: {similar_keys[:3]}")
        
        # 显示映射表中的PMI相关条目
        pmi_entries = [(k, v) for k, v in normalized_industry_map.items() if 'pmi' in k]
        if pmi_entries:
            print(f"  [DEBUG] 映射表中的PMI相关条目:")
            for k, v in pmi_entries:
                print(f"    '{k}' -> '{v}'")
        else:
            print(f"  [DEBUG] 映射表中未找到PMI相关条目")
            # 显示映射表的前几个条目作为参考
            print(f"  [DEBUG] 映射表前10个条目:")
            for i, (k, v) in enumerate(list(normalized_industry_map.items())[:10]):
                print(f"    '{k}' -> '{v}'")

        if not industry_to_vars:
            print("  警告: 未能根据提供的映射将任何变量分配到行业。")
            return None
        if "_未知行业_" in industry_to_vars:
            print(f"  警告: 有 {len(industry_to_vars['_未知行业_'])} 个变量未能映射到已知行业。")

        print(f"  [DEBUG] 按行业分组后的行业数量: {len(industry_to_vars)}")

        # 计算每个行业的 Pooled R²
        industry_r2_results = {}
        processed_industries = 0
        
        for industry_name, industry_variables in industry_to_vars.items():
            # 检查超时
            elapsed_time = time.time() - start_time
            if elapsed_time > timeout_seconds:
                print(f"    ⚠️ 计算超时 ({elapsed_time:.1f}s > {timeout_seconds}s)，停止计算")
                break
                
            if not industry_variables:
                continue
                
            processed_industries += 1
            print(f"  处理行业 [{processed_industries}/{len(industry_to_vars)}]: '{industry_name}' ({len(industry_variables)} 个变量)...")
            
            industry_data_subset = data_processed[industry_variables].copy()

            total_tss_industry = 0.0
            total_rss_industry = 0.0
            valid_regressions = 0
            processed_vars = 0

            for var in industry_variables:
                # 检查超时
                elapsed_time = time.time() - start_time
                if elapsed_time > timeout_seconds:
                    print(f"      ⚠️ 变量处理超时 ({elapsed_time:.1f}s > {timeout_seconds}s)，停止处理该行业")
                    break
                    
                processed_vars += 1
                if processed_vars % 5 == 0:  # 每5个变量打印一次进度
                    print(f"    进度: {processed_vars}/{len(industry_variables)} 变量已处理")
                
                y_series = industry_data_subset[var].dropna()
                common_index = factors_with_const.index.intersection(y_series.index)

                if len(common_index) < n_factors + 2: # 需要足够点数进行回归
                    continue

                y = y_series.loc[common_index]
                X = factors_with_const.loc[common_index]

                # 再次检查 X 是否因对齐引入 NaN
                if X.isnull().any().any():
                     rows_before = len(X)
                     X = X.dropna()
                     y = y.loc[X.index]
                     if len(X) < n_factors + 2:
                         continue

                if y.var() == 0: # 如果因变量没有变动
                    continue # TSS 为 0，R² 无意义

                try:
                    tss = np.sum((y - y.mean())**2)
                    model = sm.OLS(y, X).fit()
                    rss = np.sum(model.resid**2)

                    if np.isfinite(tss) and np.isfinite(rss):
                        total_tss_industry += tss
                        total_rss_industry += rss
                        valid_regressions += 1
                    else:
                        print(f"      警告: 变量 '{var}' 计算出的 TSS 或 RSS 无效，跳过其贡献。")

                except Exception as e_ols:
                    print(f"      错误: 对变量 '{var}' 进行 OLS 回归时出错: {e_ols}")

            # 计算行业的 Pooled R²
            if valid_regressions > 0 and total_tss_industry > 1e-9: # 避免除零和无有效回归
                pooled_r2 = 1.0 - (total_rss_industry / total_tss_industry)
                industry_r2_results[industry_name] = pooled_r2
                print(f"  行业 '{industry_name}' Pooled R²: {pooled_r2:.4f} (基于 {valid_regressions} 个变量)")
            elif valid_regressions == 0:
                 print(f"  警告: 行业 '{industry_name}' 没有成功完成任何变量的回归，无法计算 R²。")
                 industry_r2_results[industry_name] = np.nan # Assign NaN
            else: # TSS 接近于 0
                 print(f"  警告: 行业 '{industry_name}' 的总平方和接近于零 ({total_tss_industry})，无法计算有意义的 R²。")
                 industry_r2_results[industry_name] = np.nan # Assign NaN

        if not industry_r2_results:
             print("  警告: 未能计算任何行业的 Pooled R²。")
             return None

        total_time = time.time() - start_time
        print(f"  行业R²计算完成，总耗时: {total_time:.1f}秒")
        return pd.Series(industry_r2_results, name="Industry_Pooled_R2")

    except Exception as e:
        print(f"  计算行业 Pooled R² 时发生意外错误: {e}")
        import traceback
        traceback.print_exc()
        return None

# <<< 新增函数：计算每个因子对每个行业的 R2 >>>
def calculate_factor_industry_r2(
    dfm_results: Any,
    data_processed: pd.DataFrame,
    variable_list: List[str],
    var_industry_map: Dict[str, str],
    n_factors: int,
    timeout_seconds: int = 120  # 新增超时参数
) -> Optional[Dict[str, pd.Series]]:
    """
    计算每个估计出的因子 Fᵢ 对每个行业 J 内变量群体的总体解释力度 (Pooled R²)。
    对每个行业，分别用单个因子进行回归。

    Args:
        dfm_results: DFM 模型结果对象，需要包含 .x_sm (估计的因子)。
        data_processed: 经过预处理（例如平稳化）的数据，用于 DFM 拟合。
        variable_list: 包含在 data_processed 中并用于最终模型的变量列表。
        var_industry_map: 变量名到行业名的映射字典。
        n_factors: 使用的因子数量。
        timeout_seconds: 超时时间（秒），默认120秒

    Returns:
        一个字典，键为因子名称 (e.g., 'Factor1'),
        值为 Pandas Series (索引为行业名称，值为该因子对该行业的 Pooled R²)。
        如果无法计算则返回 None。
    """
    start_time = time.time()
    print("\n计算单因子对行业的 Pooled R²...")
    
    if not hasattr(dfm_results, 'x_sm') or not isinstance(dfm_results.x_sm, pd.DataFrame):
        print("  错误: DFM 结果对象缺少 'x_sm' (因子) 属性或类型错误。")
        return None
    if data_processed is None or data_processed.empty:
        print("  错误: 提供的 'data_processed' 为空或 None。")
        return None
    if not variable_list:
        print("  错误: 'variable_list' 为空。")
        return None
    if not var_industry_map:
        print("  警告: 未提供 'var_industry_map'，无法按行业分组。")
        return None
    if n_factors <= 0 or n_factors > dfm_results.x_sm.shape[1]:
        print(f"  错误: 因子数量 'n_factors' ({n_factors}) 无效或超出范围 ({dfm_results.x_sm.shape[1]})。")
        return None

    try:
        factors_std = dfm_results.x_sm.iloc[:, :n_factors].copy() # 选择正确的因子数量
        factor_names = [f'Factor{i+1}' for i in range(n_factors)]
        factors_std.columns = factor_names # 确保列名正确

        # 规范化行业映射的键
        normalized_industry_map = {
            unicodedata.normalize('NFKC', str(k)).strip().lower(): str(v).strip()
            for k, v in var_industry_map.items()
            if pd.notna(k) and str(k).lower() != 'nan' and pd.notna(v) and str(v).lower() != 'nan'
        }
        print(f"  [DEBUG] 规范化后的行业映射数量: {len(normalized_industry_map)}")
        
        # 添加详细的PMI映射调试信息
        print(f"  [DEBUG] 原始行业映射数量: {len(var_industry_map)}")
        print(f"  [DEBUG] 原始映射中的PMI相关条目:")
        original_pmi_entries = {k: v for k, v in var_industry_map.items() if 'pmi' in str(k).lower() or 'PMI' in str(k)}
        if original_pmi_entries:
            for k, v in list(original_pmi_entries.items())[:10]:
                print(f"    原始: '{k}' -> '{v}'")
        else:
            print(f"    原始映射中未发现PMI相关条目")
        
        print(f"  [DEBUG] 规范化后的PMI相关条目:")
        normalized_pmi_entries = {k: v for k, v in normalized_industry_map.items() if 'pmi' in k}
        if normalized_pmi_entries:
            for k, v in list(normalized_pmi_entries.items())[:10]:
                print(f"    规范化: '{k}' -> '{v}'")
        else:
            print(f"    规范化后映射中未发现PMI相关条目")

        # 按行业分组变量
        industry_to_vars = defaultdict(list)
        processed_vars_set = set(data_processed.columns)
        
        # 添加调试信息：记录映射情况
        unmapped_vars = []
        mapped_vars = []
        
        for var in variable_list:
            # 检查超时
            elapsed_time = time.time() - start_time
            if elapsed_time > timeout_seconds:
                print(f"    ⚠️ 行业分组超时 ({elapsed_time:.1f}s > {timeout_seconds}s)，停止处理")
                break
                
            if var not in processed_vars_set:
                continue
            lookup_key = unicodedata.normalize('NFKC', str(var)).strip().lower()
            industry = normalized_industry_map.get(lookup_key, "_未知行业_")
            industry_to_vars[industry].append(var)
            
            # 调试信息：记录映射情况
            if industry == "_未知行业_":
                unmapped_vars.append((var, lookup_key))
            else:
                mapped_vars.append((var, lookup_key, industry))
        
        # 显示映射调试信息（简化版，用于因子-行业R²计算）
        print(f"  [DEBUG] 因子-行业R²: 成功映射变量数: {len(mapped_vars)}, 未映射变量数: {len(unmapped_vars)}")
        if unmapped_vars:
            print(f"  [DEBUG] 因子-行业R²计算中的未映射变量:")
            for i, (var, key) in enumerate(unmapped_vars[:5]):  # 只显示前5个
                print(f"    {i+1}. '{var}' -> '{key}'")

        if not industry_to_vars:
            print("  警告: 未能根据提供的映射将任何变量分配到行业。")
            return None
        if "_未知行业_" in industry_to_vars:
            print(f"  警告: 有 {len(industry_to_vars['_未知行业_'])} 个变量未能映射到已知行业。")

        print(f"  [DEBUG] 按行业分组后的行业数量: {len(industry_to_vars)}")

        # 对每个因子，计算其对各行业的 Pooled R²
        factor_industry_r2_dict = {}
        
        for factor_idx, factor_name in enumerate(factor_names):
            # 检查超时
            elapsed_time = time.time() - start_time
            if elapsed_time > timeout_seconds:
                print(f"    ⚠️ 计算超时 ({elapsed_time:.1f}s > {timeout_seconds}s)，停止计算")
                break
                
            print(f"  计算 {factor_name} 对各行业的 R²...")
            factor_series = factors_std[factor_name]
            factor_series_with_const = sm.add_constant(factor_series, prepend=True, has_constant='skip')

            industry_r2_for_this_factor = {}
            processed_industries = 0

            for industry_name, industry_variables in industry_to_vars.items():
                # 检查超时
                elapsed_time = time.time() - start_time
                if elapsed_time > timeout_seconds:
                    print(f"      ⚠️ 行业处理超时 ({elapsed_time:.1f}s > {timeout_seconds}s)，停止处理该因子")
                    break
                    
                if not industry_variables:
                    continue

                processed_industries += 1
                print(f"    处理行业 [{processed_industries}/{len(industry_to_vars)}]: '{industry_name}' ({len(industry_variables)} 个变量)...")

                total_tss_industry = 0.0
                total_rss_industry = 0.0
                valid_regressions = 0
                processed_vars = 0

                for var in industry_variables:
                    # 检查超时
                    elapsed_time = time.time() - start_time
                    if elapsed_time > timeout_seconds:
                        print(f"        ⚠️ 变量处理超时 ({elapsed_time:.1f}s > {timeout_seconds}s)，停止处理该行业")
                        break
                        
                    processed_vars += 1
                    if processed_vars % 10 == 0:  # 每10个变量打印一次进度
                        print(f"      进度: {processed_vars}/{len(industry_variables)} 变量已处理")
                    
                    y_series = data_processed[var].dropna()
                    common_index = factor_series_with_const.index.intersection(y_series.index)

                    if len(common_index) < 3: # 至少需要3个点用于单因子回归
                        continue

                    y = y_series.loc[common_index]
                    X = factor_series_with_const.loc[common_index]

                    # 再次检查 X 和 y 的 NaN
                    if X.isnull().any().any() or y.isnull().any():
                        aligned_data = pd.concat([y, X], axis=1, join='inner').dropna()
                        if len(aligned_data) < 3:
                            continue
                        y = aligned_data.iloc[:, 0]
                        X = aligned_data.iloc[:, 1:]

                    if y.var() == 0:
                        continue

                    try:
                        tss = np.sum((y - y.mean())**2)
                        # OLS: y ~ const + factor_i
                        model = sm.OLS(y, X).fit()
                        rss = np.sum(model.resid**2)

                        if np.isfinite(tss) and np.isfinite(rss) and tss > 1e-9: # 检查 TSS > 0
                            total_tss_industry += tss
                            total_rss_industry += rss
                            valid_regressions += 1

                    except Exception as e_ols:
                        print(f"        错误: 对变量 '{var}' 用因子 '{factor_name}' 进行 OLS 时出错: {e_ols}")
                        continue  # 继续处理下一个变量，不要中断整个过程

                # 计算该因子对该行业的 Pooled R²
                if valid_regressions > 0 and total_tss_industry > 1e-9:
                    pooled_r2 = max(0.0, 1.0 - (total_rss_industry / total_tss_industry)) # 确保 R2 非负
                    industry_r2_for_this_factor[industry_name] = pooled_r2
                    print(f"    => {factor_name} 对行业 '{industry_name}' 的 Pooled R²: {pooled_r2:.4f} (基于 {valid_regressions} 个变量)")
                elif valid_regressions == 0:
                     print(f"    => {factor_name} 对行业 '{industry_name}' 没有成功完成任何变量的回归，R² 设为 NaN。")
                     industry_r2_for_this_factor[industry_name] = np.nan
                else: # TSS 接近于 0
                     print(f"    => {factor_name} 对行业 '{industry_name}' 的总平方和接近零，R² 设为 NaN。")
                     industry_r2_for_this_factor[industry_name] = np.nan

            factor_industry_r2_dict[factor_name] = pd.Series(industry_r2_for_this_factor, name=f"{factor_name}_Industry_R2")

        total_time = time.time() - start_time
        print(f"  单因子对行业R²计算完成，总耗时: {total_time:.1f}秒")
        return factor_industry_r2_dict

    except Exception as e:
        print(f"  计算单因子对行业 Pooled R² 时发生意外错误: {e}")
        import traceback
        traceback.print_exc()
        return None

# --- <<< 新增：计算单因子对变量类型的汇总 R2 >>> ---
def calculate_factor_type_r2(
    dfm_results: Any,
    data_processed: pd.DataFrame,
    variable_list: List[str],
    var_type_map: Dict[str, str], # <-- 使用类型映射
    n_factors: int,
    timeout_seconds: int = 120  # 新增超时参数
) -> Optional[Dict[str, pd.Series]]:
    """
    计算每个因子对每个变量类型变量群体的汇总 R 平方值。
    采用与 calculate_factor_industry_r2 一致的逻辑：
    对每个类型内的变量，用单个因子进行 OLS 回归，累加 TSS 和 RSS，
    最后计算 Pooled R² = max(0.0, 1 - total_rss / total_tss)。

    Args:
        dfm_results: DFM 模型运行结果对象 (需要包含 x_sm)。
        data_processed: DFM 模型输入的处理后数据 (变量在列)。
        variable_list: 要考虑的变量列表。
        var_type_map: 变量名到类型名称的字典。
        n_factors: 模型使用的因子数量。
        timeout_seconds: 超时时间（秒），默认120秒

    Returns:
        一个字典，键是因子名称 (Factor1, ...)，
        值是 pandas Series (索引是类型名称, 值是 Pooled R²)。出错则返回 None。
    """
    start_time = time.time()
    print("\n计算单因子对变量类型的 Pooled R² (OLS-based)...")
    factor_type_r2_dict = {}

    # --- 输入验证 ---
    if not (dfm_results and hasattr(dfm_results, 'x_sm')):
        print("  错误: DFM 结果对象无效或缺少 'x_sm'。")
        return None
    if not isinstance(dfm_results.x_sm, pd.DataFrame):
        try:
            factors_std_df = pd.DataFrame(dfm_results.x_sm)
        except Exception as e:
            print(f"  错误: 无法将 dfm_results.x_sm 转换为 DataFrame: {e}")
            return None
    else:
        factors_std_df = dfm_results.x_sm

    if not isinstance(n_factors, (int, np.integer)) or n_factors <= 0 or n_factors > factors_std_df.shape[1]:
        print(f"  错误: 因子数量 'n_factors' ({n_factors}) 无效或超出范围 ({factors_std_df.shape[1]})。")
        return None
    factors_std = factors_std_df.iloc[:, :n_factors].copy()
    factor_cols = [f'Factor{i+1}' for i in range(n_factors)]
    factors_std.columns = factor_cols

    if data_processed is None or data_processed.empty:
        print("提供的 'data_processed' 为空或 None。")
        return None
    if not variable_list:
        print("'variable_list' 为空。")
        return None
    if not var_type_map:
        print("未提供变量类型映射 (var_type_map)，无法计算 Factor-Type R²。")
        return None
    # --- 结束验证 ---

    try:
        # --- 准备类型映射和分组 ---
        var_type_map_norm = {
            unicodedata.normalize('NFKC', str(k)).strip().lower(): str(v).strip()
            for k, v in var_type_map.items()
            if pd.notna(k) and str(k).lower() != 'nan' and pd.notna(v) and str(v).lower() != 'nan'
        }
        type_to_vars = defaultdict(list)
        processed_vars_set = set(data_processed.columns)
        unmapped_vars_list = []
        for var in variable_list:
            if var not in processed_vars_set:
                continue # Skip if var not in the processed data
            lookup_key = unicodedata.normalize('NFKC', str(var)).strip().lower()
            var_type = var_type_map_norm.get(lookup_key)
            if var_type:
                type_to_vars[var_type].append(var)
            else:
                unmapped_vars_list.append(var)
        if unmapped_vars_list:
             type_to_vars['_未知类型_'] = unmapped_vars_list
             print(f"有 {len(unmapped_vars_list)} 个变量未能映射到已知类型，归入 '_未知类型_'。")
        if not type_to_vars:
            print("未能根据提供的映射将任何变量分配到类型。")
            return None
        # --- 结束分组 ---

        # 过滤 data_processed 以匹配分组后的变量 (提高效率)
        all_vars_in_types = [var for vars_list in type_to_vars.values() for var in vars_list]
        data_subset = data_processed[all_vars_in_types]

        print(f"将为 {n_factors} 个因子和 {len(type_to_vars)} 个类型计算 Pooled R² (OLS-based)...")

        # --- 遍历因子和类型计算 R² ---
        for factor_idx, factor_name in enumerate(factor_cols):
            # 检查超时
            if time.time() - start_time > timeout_seconds:
                print(f"  ⚠️ 计算超时 ({timeout_seconds}秒)，在因子 {factor_name} 处停止")
                break
                
            type_r2_values = {}
            print(f"--- 计算因子: {factor_name} ({factor_idx+1}/{n_factors}) ---")
            factor_series = factors_std[[factor_name]] # 单个因子列
            factor_series_with_const = sm.add_constant(factor_series, prepend=True) # 添加常数项

            for type_idx, (var_type, type_variables) in enumerate(type_to_vars.items()):
                # 再次检查超时
                if time.time() - start_time > timeout_seconds:
                    print(f"  ⚠️ 计算超时，在类型 {var_type} 处停止")
                    break
                    
                if not type_variables:
                    continue # Skip empty types

                print(f"  处理类型: '{var_type}' ({len(type_variables)} 个变量) 对 {factor_name}...")
                type_data_subset = data_subset[type_variables]

                total_tss_type = 0.0
                total_rss_type = 0.0
                valid_regressions = 0
                
                # 限制每个类型内的变量数量，避免过长的计算
                max_vars_per_type = 20  # 限制每个类型最多处理20个变量
                if len(type_variables) > max_vars_per_type:
                    print(f"    ⚠️ 类型 '{var_type}' 有 {len(type_variables)} 个变量，限制处理前 {max_vars_per_type} 个")
                    type_variables = type_variables[:max_vars_per_type]

                # 遍历类型内的每个变量
                for var_idx, var in enumerate(type_variables):
                    # 定期检查超时
                    if var_idx > 0 and var_idx % 5 == 0:
                        if time.time() - start_time > timeout_seconds:
                            print(f"    ⚠️ 计算超时，在变量 {var} 处停止")
                            break
                    
                    try:
                        y_series = type_data_subset[var].dropna()
                        # 对齐当前变量和当前单个因子
                        common_index = factor_series_with_const.index.intersection(y_series.index)

                        if len(common_index) < 3: # OLS (k=1) needs at least 3 points
                            continue

                        y = y_series.loc[common_index]
                        X = factor_series_with_const.loc[common_index]

                        # 再次检查 NaN (理论上不应有, 但以防万一)
                        if X.isnull().any().any() or y.isnull().any():
                            combined = pd.concat([y, X], axis=1).dropna()
                            if len(combined) < 3:
                                continue
                            y = combined.iloc[:, 0]
                            X = combined.iloc[:, 1:]

                        if y.var() < 1e-9:
                            continue # Skip if variable has no variance

                        tss = np.sum((y - y.mean())**2)
                        # OLS: y ~ const + factor_k
                        model = sm.OLS(y, X).fit()
                        rss = np.sum(model.resid**2)

                        if np.isfinite(tss) and np.isfinite(rss) and tss > 1e-9:
                            total_tss_type += tss
                            total_rss_type += rss
                            valid_regressions += 1

                    except Exception as e_ols:
                        print(f"    OLS error for var '{var}' vs factor '{factor_name}': {e_ols}")
                        continue  # 继续处理下一个变量，不要中断整个过程

                # 计算该因子对该类型的 Pooled R²
                if valid_regressions > 0 and total_tss_type > 1e-9:
                    # ***** 与 Industry R2 算法一致，确保非负 *****
                    pooled_r2 = max(0.0, 1.0 - (total_rss_type / total_tss_type))
                    type_r2_values[var_type] = pooled_r2
                    print(f"  => {factor_name} 对类型 '{var_type}' 的 Pooled R²: {pooled_r2:.4f} (基于 {valid_regressions} 个变量)")
                elif valid_regressions == 0:
                     print(f"  => {factor_name} 对类型 '{var_type}' 没有成功完成任何变量的回归，R² 设为 NaN。")
                     type_r2_values[var_type] = np.nan
                else: # TSS 接近于 0
                     print(f"  => {factor_name} 对类型 '{var_type}' 的总平方和接近零，R² 设为 NaN。")
                     type_r2_values[var_type] = np.nan
            # --- 结束类型循环 ---

            if type_r2_values:
                factor_type_r2_dict[factor_name] = pd.Series(type_r2_values).sort_index()
            else:
                print(f"因子 {factor_name} 未能计算任何类型的 R²。")
        # --- 结束因子循环 ---

        if not factor_type_r2_dict:
            print("未能计算任何因子对任何类型的 Pooled R² (OLS-based)。")
            return None

        total_time = time.time() - start_time
        print(f"单因子对变量类型的 Pooled R² (OLS-based) 计算完成，耗时: {total_time:.1f}秒")
        return factor_type_r2_dict

    except Exception as e:
        print(f"计算单因子对类型 Pooled R² (OLS-based) 时发生意外错误: {e}")
        # 不打印完整traceback，避免卡住
        print(f"错误类型: {type(e).__name__}")
        return None
# --- 结束修改 ---

# --- 结束新增函数 --- 