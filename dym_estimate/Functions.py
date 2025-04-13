# -*- coding: utf-8 -*-
"""
Created on Mon Feb 17 10:01:28 2020

@author: Hogan
"""
from scipy.interpolate import interp1d
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
from datetime import *
import tkinter.filedialog
from datetime import timedelta
import math 
import time
import calendar
from numba import jit
from sklearn.metrics import mean_squared_error
from statsmodels.tsa.api import VAR
import scipy
import statsmodels.tsa.stattools as ts
import statsmodels.tsa as tsa
import matplotlib.pyplot as plt
import numpy as np
import sklearn
from sklearn.metrics import mean_squared_error, r2_score
import statsmodels.api as sm
from datetime import datetime,timedelta
from typing import List, Dict, Union
from collections import Counter

def expand_data(data, step, freq='M'):
    dates = pd.date_range(data.index[0], periods=len(data.index)+step, freq=freq)
    data_expanded = pd.DataFrame(data=np.nan,index=dates,columns=data.columns)
    data_expanded.iloc[:len(data.index)] = data.values
    
    return data_expanded

def import_data(file_name, sheet_name, start=0, interpolation=False, encoding='gb18030'):
    Temp = pd.read_excel(file_name, sheetname = sheet_name, encoding = encoding)
    res = Temp.iloc[start:,1:]
    res.index = Temp.iloc[start:,0]
    if interpolation==True:
        res = DataInterpolation(res, 0, len(res.index), 'cubic').dropna(axis=0,how='any')
    return res

def transform_data(data, method):
    transform=pd.DataFrame(data=np.nan,index=data.index,columns=data.columns)
    if method=='MoM':
        for i in range(5,len(data.index)):
            last_month = data[data.index<=(data.index[i]-timedelta(30))].iloc[-1]
            transform.iloc[i] = (data.iloc[i]-last_month)/last_month*100
    return transform

def plot_compare(history, forecast, title,fig_size=[24,16], line_width=3.0,font_size='xx-large'):
    plt.rcParams['figure.figsize'] = (fig_size[0], fig_size[1])
    plt.rcParams['lines.markersize'] = 6
    plt.rcParams['image.cmap'] = 'gray'
    
    plt.figure()
    plt.plot(history.index,history,color='r',label='observed', linewidth=line_width)
    plt.plot(forecast.index,forecast,color='k',label='predicted', linewidth=line_width)
    plt.legend()
    plt.xlabel('Date')
    plt.title(title, fontweight='bold', fontsize=font_size)
    plt.show()
    
    return
          


def DataInterpolation(data, start, end, method):
    # data must be a time series dataframe
    n_row = len(data.index)
    n_col = len(data.columns)
    res = np.array(np.zeros(shape=(n_row,n_col)))
    
    for i in range(n_col):
        res[:,i] = np.array(data.iloc[:,i]).T
        y=data.iloc[start:end,i]
        location = np.where(y.notnull())[0]
        upper_bound=max(location)
        lower_bound=min(location)
        f2 = interp1d(location, y[y.notnull()], kind=method)
        x = np.linspace(lower_bound, upper_bound, num=upper_bound-lower_bound, endpoint=False)
        res[lower_bound:upper_bound,i]=np.array(f2(x)).T
    
    res = pd.DataFrame(res, index=data.index, columns=data.columns)
    
    return res

def rand_Matrix(n_row, n_col):
    randArr = np.random.randn(n_row, n_col)
    randMat = np.array(randArr)
    return randMat


def calculate_factor_loadings(observables, factors):
    """Calculates factor loadings (Lambda) using OLS, handling NaNs.

    Regresses each observable series onto the factors using only non-missing data points
    for that specific series.

    Args:
        observables (pd.DataFrame): Observation data (n_time x n_obs), potentially with NaNs.
                                      Assumed to be already centered.
        factors (pd.DataFrame): Factor data (n_time x n_factors), assumed no NaNs.

    Returns:
        np.ndarray: Lambda matrix (n_obs x n_factors).
    """
    n_obs = observables.shape[1]
    n_factors = factors.shape[1]
    Lambda = np.full((n_obs, n_factors), np.nan) # Initialize with NaNs

    # Ensure factors is numpy array for efficiency if needed later, though OLS takes DataFrame
    F_np = factors.to_numpy()

    for i in range(n_obs):
        y_i = observables.iloc[:, i]
        valid_idx = y_i.notna() & factors.notna().all(axis=1) # Ensure both y_i and all factors are valid

        y_i_valid = y_i[valid_idx]
        F_valid = factors[valid_idx]

        # Check if enough data points remain for OLS (at least n_factors + 1 if adding constant, or n_factors if not)
        # We don't add constant as data is assumed centered
        if len(y_i_valid) > n_factors:
            try:
                # Perform OLS: y_i ~ F (no constant needed for centered data)
                ols_model = sm.OLS(y_i_valid, F_valid)
                ols_results = ols_model.fit()
                Lambda[i, :] = ols_results.params.values
            except Exception as e:
                print(f"Warning: OLS failed for observable {i} ('{observables.columns[i]}'). Loadings set to NaN. Error: {e}")
                # Lambda[i, :] remains NaN due to initialization
        else:
             # 如果有效数据点不足，打印警告并设置 Lambda 为 NaN
             # print(f"Warning: Not enough valid data points for observable {i} (\'{observables.columns[i]}\') after dropping NaNs ({len(y_i_valid)} <= {n_factors}). Loadings set to NaN.") # 注释掉
             # print("NOT_ENOUGH_DATA_WARNING_COUNTED") # <--- 在这里添加计数标记 # 注释掉
             pass # 保持安静
             # Lambda[i, :] 仍然是 NaN

    return Lambda # Shape (n_obs, n_factors)

def calculate_prediction_matrix(factors):
    n_time = len(factors.index)
    F = np.array(factors)

    # Correct matrix calculation for A = (F_t' F_{t-1})(F_{t-1}' F_{t-1})^-1
    F_t = F[1:, :]      # Shape (n_time-1, n_factors)
    F_tm1 = F[:-1, :]   # Shape (n_time-1, n_factors)

    Ft_Ftm1 = F_t.T @ F_tm1     # Shape: (n_factors, n_time-1) @ (n_time-1, n_factors) -> (n_factors, n_factors)
    Ftm1_Ftm1 = F_tm1.T @ F_tm1 # Shape: (n_factors, n_time-1) @ (n_time-1, n_factors) -> (n_factors, n_factors)

    # Use pseudo-inverse for stability, add small jitter
    A = Ft_Ftm1 @ np.linalg.pinv(Ftm1_Ftm1 + np.eye(Ftm1_Ftm1.shape[0]) * 1e-7)

    return A

def calculate_shock_matrix(factors, prediction_matrix, n_shocks):
    n_time = len(factors.index)
    F = np.array(factors)
    A = np.array(prediction_matrix)
    
    # Calculate F_{t-1}' F_{t-1} efficiently
    F_tm1 = F[:-1, :] # Shape (n_time-1, n_factors)
    temp = F_tm1.T @ F_tm1 # Shape (n_factors, n_factors)
    
    # Calculate F_t' F_t efficiently
    F_t = F[1:, :] # Shape (n_time-1, n_factors)
    term1 = F_t.T @ F_t # Shape (n_factors, n_factors)
    term1 = term1 / (n_time - 1)
    
    # Calculate Sigma = E[F_t F_t'] - A E[F_{t-1} F_{t-1}'] A'
    term2 = A @ (temp / (n_time - 1)) @ A.T
    Sigma = term1 - term2 # This is the estimated Q matrix
    
    # --- Ensure Sigma (Q) is positive semi-definite --- 
    try:
        eigenvalues, eigenvectors = np.linalg.eigh(Sigma) # Use eigh for symmetric matrices
        # Replace negative eigenvalues with a small positive number
        min_eig_val = 1e-7 # Floor for eigenvalues
        eigenvalues_corrected = np.maximum(eigenvalues, min_eig_val)
        
        # Reconstruct Sigma using corrected eigenvalues (optional, but good practice)
        Sigma_corrected = eigenvectors @ np.diag(eigenvalues_corrected) @ eigenvectors.T
        
        # Calculate B using the corrected eigenvalues
        # Sort eigenvalues descending to pick largest shocks
        sorted_indices = np.argsort(eigenvalues_corrected)[::-1]
        # Select top n_shocks eigenvalues and corresponding eigenvectors
        evalues_selected = eigenvalues_corrected[sorted_indices[:n_shocks]] 
        M = eigenvectors[:, sorted_indices[:n_shocks]]
        
        # Calculate B = M * sqrt(diag(selected eigenvalues))
        # Ensure sqrt is applied only to non-negative values (should be guaranteed by correction)
        B = M @ np.diag(np.sqrt(evalues_selected))
        
        # Use the corrected Sigma as the returned Q
        Q = Sigma_corrected

    except np.linalg.LinAlgError as e:
        print(f"Warning: Eigenvalue decomposition failed for Sigma (Q) in calculate_shock_matrix: {e}. Falling back to identity matrix for Q and B.")
        Q = np.eye(A.shape[0]) * 1e-6 # Small identity matrix as fallback Q
        # Fallback B: Adjust shape based on n_shocks
        B = np.zeros((A.shape[0], n_shocks))
        min_dim_fallback = min(A.shape[0], n_shocks)
        B[:min_dim_fallback, :min_dim_fallback] = np.eye(min_dim_fallback) * np.sqrt(1e-6)
    # --- End Ensure PSD --- 
    
    return B, Q # Return corrected Q (Sigma_corrected)

def calculate_pca(observables, n_factors):
    # syntax: 
    n_time = len(observables.index)
    x = np.array(observables - observables.mean())
    z = np.array((observables - observables.mean())/observables.std())
    
    # Calculate covariance matrix S from standardized data z
    # Correct calculation: S = (1/N) * Z'Z
    S = (z.T @ z) / n_time

    eigenvalues, eigenvectors = np.linalg.eigh(S) # Use eigh for covariance matrix
    sorted_indices = np.argsort(eigenvalues)[::-1] # Descending order
    evalues = eigenvalues[sorted_indices[:-n_factors-1:-1]]
    V = np.array(eigenvectors[:,sorted_indices[:-n_factors-1:-1]])
    D = np.diag(evalues)
    
    return D, V, S
    
def calculate_covariance(factors):
    n_time = len(factors.index)
    F = np.array(factors)
    temp = [factors.iloc[:,i] for i in range(len(factors.columns))]
    return np.cov(temp)

def align_mixed_frequency_data(
    data_list: Union[List[pd.DataFrame], Dict[str, pd.DataFrame]],
    target_freq: str = 'D'
) -> pd.DataFrame:
    """
    将多个可能具有不同频率和锯齿状边缘的时间序列 DataFrame 对齐到
    一个单一的高频网格上。

    参数:
        data_list: 包含 Pandas DataFrame 的列表或字典。每个 DataFrame
                   必须有一个 DatetimeIndex。如果提供字典，键可用于标识，
                   但除非列名冲突，否则不直接用于输出结构。
                   请确保所有 DataFrame 中的列名是唯一的。
        target_freq: 输出 DataFrame 的目标频率，
                     例如 'D' (日历日), 'B' (工作日)。

    返回:
        一个单一的 Pandas DataFrame，其索引为 target_freq 的 DatetimeIndex，
        包含所有输入 DataFrame 的列。由于频率差异或锯齿状边缘导致的缺失观测值
        将用 NaN 填充。

    异常:
        ValueError: 如果输入不是 DataFrame 的列表或字典，或者 DataFrame
                    没有 DatetimeIndex。
        ValueError: 如果输入 DataFrame 之间存在重复的列名。
    """
    if isinstance(data_list, list):
        dfs = data_list
    elif isinstance(data_list, dict):
        dfs = list(data_list.values())
    else:
        raise ValueError("输入 'data_list' 必须是 DataFrame 的列表或字典。")

    if not all(isinstance(df, pd.DataFrame) for df in dfs):
        raise ValueError(" 'data_list' 中的所有项必须是 Pandas DataFrame。")
    if not all(isinstance(df.index, pd.DatetimeIndex) for df in dfs):
        raise ValueError("所有输入的 DataFrame 必须具有 DatetimeIndex。")

    all_columns = []
    for df in dfs:
        all_columns.extend(df.columns)
    if len(all_columns) != len(set(all_columns)):
        duplicates = [item for item, count in Counter(all_columns).items() if count > 1]
        raise ValueError(f"在输入的 DataFrame 之间发现重复的列名: {duplicates}。请确保列名唯一。")

    # 确定整体日期范围
    valid_dfs = [df for df in dfs if not df.empty]
    if not valid_dfs:
        print("警告：所有输入的 DataFrame 都是空的。返回一个空的 DataFrame。")
        return pd.DataFrame(columns=all_columns)

    min_date = min(df.index.min() for df in valid_dfs)
    max_date = max(df.index.max() for df in valid_dfs)

    if pd.isna(min_date) or pd.isna(max_date):
         raise ValueError("无法确定有效的日期范围。请检查输入的 DataFrame。")


    # 创建高频索引
    target_index = pd.date_range(start=min_date, end=max_date, freq=target_freq)

    # 初始化合并后的 DataFrame
    # 先创建带有所有列名的空 DataFrame，以确保列序和处理空 DataFrame 的情况
    combined_df = pd.DataFrame(index=target_index, columns=all_columns)
    combined_df = combined_df.astype(np.float64) # 明确类型以存储 NaN

    # 重新索引每个 DataFrame 并合并
    for df in dfs:
        if not df.empty:
            # Reindex 确保对齐并将缺失日期填充为 NaN
            # 使用 fill_value=np.nan 明确指定填充值（虽然通常是默认值）
            df_reindexed = df.reindex(target_index, fill_value=np.nan)
            # 使用 update 更新值，这比逐列分配更安全，特别是对于大型数据集
            combined_df.update(df_reindexed)

    # 重新应用列顺序，因为 update 不保证顺序
    combined_df = combined_df[all_columns]

    return combined_df 