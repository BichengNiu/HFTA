# -*- coding: utf-8 -*-
"""
数据处理相关工具函数
"""
import pandas as pd
import numpy as np
from statsmodels.tsa.stattools import adfuller
from typing import Tuple, Dict

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