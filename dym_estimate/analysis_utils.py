# -*- coding: utf-8 -*-
"""
包含 DFM 结果分析相关工具函数的模块，例如 PCA 和因子贡献度计算。
"""
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer # 如果 PCA 需要填充
from typing import Tuple, Dict, Optional, List

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
        Optional[pd.DataFrame]: 包含 PCA 结果的 DataFrame (主成分, 解释方差%, 累计解释方差%)，
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
        if nan_count > 0:
            if impute_strategy:
                print(f"  处理 PCA 输入数据的缺失值 (共 {nan_count} 个，使用策略: {impute_strategy})...")
                imputer = SimpleImputer(strategy=impute_strategy)
                data_pca_imputed = pd.DataFrame(imputer.fit_transform(data_for_pca), 
                                                index=data_for_pca.index, 
                                                columns=data_for_pca.columns)
                print(f"  填充后 NaN 数量: {data_pca_imputed.isna().sum().sum()}")
            else:
                print(f"  警告: 数据包含 {nan_count} 个 NaN，但未指定填充策略。PCA 可能失败。")
                data_pca_imputed = data_for_pca # 继续尝试，但可能出错
        else:
            data_pca_imputed = data_for_pca
            print("  数据无缺失值，无需填充。")

        # 执行 PCA
        pca = PCA(n_components=n_components)
        print(f"  对填充后的数据执行 PCA (n_components={n_components})...")
        pca.fit(data_pca_imputed)

        explained_variance_ratio_pct = pca.explained_variance_ratio_ * 100
        cumulative_explained_variance_pct = np.cumsum(explained_variance_ratio_pct)

        pca_results_df = pd.DataFrame({
            '主成分 (PC)': [f'PC{i+1}' for i in range(n_components)],
            '解释方差 (%)': explained_variance_ratio_pct,
            '累计解释方差 (%)': cumulative_explained_variance_pct
        })

        print("  PCA 解释方差计算完成:")
        print(pca_results_df.to_string(index=False))

    except Exception as e_pca_main:
        print(f"  计算 PCA 解释方差时发生错误: {e_pca_main}")
        import traceback
        traceback.print_exc()
        pca_results_df = None # 确保出错时返回 None
        
    return pca_results_df

def calculate_factor_contributions(
    dfm_results: object, 
    data_processed: pd.DataFrame, 
    target_variable: str, 
    n_factors: int
) -> Tuple[Optional[pd.DataFrame], Optional[Dict[str, float]]]:
    """
    计算 DFM 各因子对目标变量方差的贡献度。

    Args:
        dfm_results (object): DFM 模型运行结果对象 (需要包含 Lambda 属性)。
        data_processed (pd.DataFrame): DFM 模型输入的处理后数据 (用于定位目标变量)。
        target_variable (str): 目标变量名称。
        n_factors (int): 模型使用的因子数量。

    Returns:
        Tuple[Optional[pd.DataFrame], Optional[Dict[str, float]]]: 
            - contribution_df: 包含各因子贡献度详情的 DataFrame，出错则为 None。
            - factor_contributions: 因子名称到总方差贡献度(%)的字典，出错则为 None。
    """
    print("\n计算各因子对目标变量的贡献度...")
    contribution_df = None
    factor_contributions_dict = None
    try:
        lambda_target = None
        if dfm_results and hasattr(dfm_results, 'Lambda'):
            final_loadings = dfm_results.Lambda
            if data_processed is not None and target_variable in data_processed.columns:
                try:
                    target_var_index_pos = data_processed.columns.get_loc(target_variable)
                    if final_loadings is not None and target_var_index_pos < final_loadings.shape[0] and n_factors <= final_loadings.shape[1]:
                        lambda_target = final_loadings[target_var_index_pos, :n_factors] # 只取使用的因子数
                        print(f"  成功提取目标变量 '{target_variable}' 的 {n_factors} 个因子载荷。")
                    else:
                        print(f"  错误: 无法在最终载荷矩阵中定位目标变量索引 ({target_var_index_pos}) 或载荷/因子数不足 (Loadings shape: {final_loadings.shape}, n_factors: {n_factors})。")
                except (KeyError, IndexError, AttributeError, TypeError) as e_lambda:
                    print(f"  提取目标变量载荷时出错: {e_lambda}")
            else:
                print("  警告: 无法确定目标变量在最终载荷中的位置，因为 data_processed 不可用或不包含目标变量。")
        else:
             print("  警告: 无法提取目标载荷，最终模型结果或其 Lambda 属性不可用。")

        if lambda_target is not None and n_factors > 0:
            lambda_target_sq = lambda_target ** 2
            sum_lambda_target_sq = np.sum(lambda_target_sq)
            
            if sum_lambda_target_sq > 1e-9:
                pct_contribution_common = (lambda_target_sq / sum_lambda_target_sq) * 100
            else:
                pct_contribution_common = np.zeros_like(lambda_target_sq) * np.nan
                print("  警告: 目标变量的平方载荷和过小，无法计算对共同方差的百分比贡献。")
            
            pct_contribution_total = lambda_target_sq * 100

            contribution_df = pd.DataFrame({
                '因子 (Factor)': [f'Factor{i+1}' for i in range(n_factors)],
                '载荷 (Loading)': lambda_target,
                '平方载荷 (Loading^2)': lambda_target_sq,
                '对共同方差贡献 (%)': pct_contribution_common,
                '对总方差贡献(近似 %)': pct_contribution_total
            })
            contribution_df = contribution_df.sort_values(by='对总方差贡献(近似 %)', ascending=False)

            print("  各因子对目标变量方差贡献度计算完成:")
            print(contribution_df.to_string(index=False, float_format="%.4f"))
            print(f"  目标变量共同度 (Communality): {sum_lambda_target_sq:.4f}")
            
            factor_contributions_dict = contribution_df.set_index('因子 (Factor)')['对总方差贡献(近似 %)'].to_dict()
        
        elif lambda_target is None:
            print("  未能成功提取目标载荷，跳过贡献度计算。")
        else: # n_factors 无效
            print(f"  错误: 最终因子数无效 ({n_factors})，无法计算贡献度。")

    except Exception as e_contrib_main:
        print(f"  计算因子对目标变量贡献度时发生错误: {e_contrib_main}")
        import traceback
        traceback.print_exc()
        contribution_df = None # 确保出错时返回 None
        factor_contributions_dict = None
        
    return contribution_df, factor_contributions_dict 