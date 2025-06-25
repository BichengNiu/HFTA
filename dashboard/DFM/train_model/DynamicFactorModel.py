# -*- coding: utf-8 -*-
"""
Created on Mon Feb 17 10:00:39 2020

@author: Hogan
"""
from scipy.interpolate import interp1d
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
from datetime import datetime
# import tkinter.filedialog  # 🔥 注释掉可能有问题的导入
from datetime import timedelta
import math
import time
import calendar
# from numba import jit  # 🔥 注释掉可能有问题的导入
from sklearn.metrics import mean_squared_error
from statsmodels.tsa.api import VAR
import scipy
import statsmodels.tsa.stattools as ts
import statsmodels.tsa as tsa
import sklearn
from sklearn.metrics import mean_squared_error, r2_score
import statsmodels.api as sm
from sklearn.decomposition import PCA
try:
    from .DiscreteKalmanFilter import calculate_factor_loadings, KalmanFilter, FIS, EMstep
except ImportError:
    from DiscreteKalmanFilter import calculate_factor_loadings, KalmanFilter, FIS, EMstep
from scipy.optimize import minimize
import numpy.linalg

# Helper function moved from Functions.py
def _calculate_pca(observables, n_factors):
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

def DFM(observation, n_factors):
    """Performs initial PCA for DFM initialization.
    Returns initial factors, mean, std dev, and idiosyncratic variance estimate.
    """
    if len(observation.columns) <= n_factors:
        raise ValueError('Error: number of common factors must be less than number of variables')

    n_time = len(observation.index)

    # Standardize data (handle potential NaNs from imputation residual)
    obs_mean = observation.mean()
    obs_std = observation.std()
    obs_std[obs_std == 0] = 1.0
    # Ensure standardization doesn't produce NaNs if mean/std calculation had issues
    z = ((observation - obs_mean) / obs_std).fillna(0) # Fill any residual NaNs with 0 after std

    # --- PCA using SVD for stability ---
    # Use np.linalg.svd directly on the standardized data z
    # U shape: (n_time, n_time), S shape: (min(n_time, n_obs),), Vh shape: (n_obs, n_obs)
    U, s, Vh = np.linalg.svd(z, full_matrices=False)

    # Factors are U[:, :k] * s[:k]
    factors = U[:, :n_factors] * s[:n_factors]
    CommonFactors = pd.DataFrame(data=factors, index=observation.index, columns=[f'Factor{i+1}' for i in range(n_factors)])

    # Estimate idiosyncratic variance Psi from PCA residuals
    # Residuals = Z - Factors @ Vh[:k, :].T (where V = Vh.T)
    V = Vh.T
    reconstructed_z = factors @ V[:, :n_factors].T
    residuals = z - reconstructed_z
    # Psi = diagonal variance of residuals
    # Use np.nanvar to be safe, though z should be imputed
    psi_diag = np.nanvar(residuals, axis=0)
    psi_diag = np.maximum(psi_diag, 1e-6) # Ensure positivity
    Psi = np.diag(psi_diag)
    # --- End PCA --- 

    # We don't calculate Lambda, A, B, Sigma here anymore

    return DFMResultsWrapperPCA(common_factors=CommonFactors,
                                idiosyncratic_covariance=Psi,
                                obs_mean=obs_mean)

# Simplified wrapper just for PCA results
class DFMResultsWrapperPCA():
    def __init__(self, common_factors, idiosyncratic_covariance, obs_mean):
        self.common_factors = common_factors
        self.idiosyncratic_covariance = idiosyncratic_covariance
        self.obs_mean = obs_mean

def DFM_EMalgo(observation, n_factors, n_shocks, n_iter, error='False'):

    n_obs = observation.shape[1]
    n_time = observation.shape[0]
    state_names = [f'Factor{i+1}' for i in range(n_factors)]

    # Step 1: Calculate Mean and Center Data
    # print("Step 1: Calculating mean and centering data...") # 注释掉
    obs_mean = observation.mean(skipna=True) # skipna=True default
    obs_centered = observation - obs_mean
    all_nan_cols = obs_centered.columns[obs_centered.isna().all()].tolist()
    if all_nan_cols:
        # print(f"Warning: Columns are all NaN after centering (likely originally all NaN): {all_nan_cols}") # 注释掉
        pass # 保持安静

    # === 修改开始: 基于 PCA 的初始化 ===
    # print("Step 2: Initializing parameters using PCA...") # 注释掉

    # 为 PCA 处理 NaN (标准化后用 0 填充)
    # print("  Standardizing data and filling NaNs with 0 for PCA initialization...") # 注释掉
    obs_std = observation.std(skipna=True)
    obs_std[obs_std == 0] = 1.0 # 避免除零
    z = (obs_centered / obs_std).fillna(0) # 使用中心化后的数据进行标准化并填充

    # 执行 PCA (SVD)
    # print("  Performing PCA via SVD...") # 注释掉
    try:
        U, s, Vh = np.linalg.svd(z, full_matrices=False)
        # 初始因子 F0 = U_k * S_k
        factors_init = U[:, :n_factors] * s[:n_factors]
        factors_init_df = pd.DataFrame(factors_init, index=observation.index, columns=state_names)
        # print(f"  Initial PCA factors calculated. Shape: {factors_init_df.shape}") # 注释掉
    except np.linalg.LinAlgError as pca_e:
        # print(f"  Error during PCA (SVD): {pca_e}. Cannot initialize with PCA. Stopping.") # 注释掉
        raise ValueError("PCA failed during initialization.") from pca_e

    # 初始化 Lambda (载荷矩阵)
    # print("  Calculating initial Lambda (Factor Loadings)...") # 注释掉
    try:
        # 使用 DiscreteKalmanFilter.py 中的函数
        Lambda_current = calculate_factor_loadings(obs_centered, factors_init_df)
        # 确保 Lambda 是 (n_obs, n_factors)
        if Lambda_current.shape == (n_factors, n_obs):
             Lambda_current = Lambda_current.T
        if Lambda_current.shape != (n_obs, n_factors):
             raise ValueError(f"Unexpected Lambda shape: {Lambda_current.shape}")
        # print(f"  Initial Lambda calculated. Shape: {Lambda_current.shape}") # 注释掉
        # print(f"  Initial Lambda contains NaN: {np.isnan(Lambda_current).any()}") # 注释掉
    except Exception as lambda_e:
        # print(f"  Error calculating initial Lambda: {lambda_e}. Initializing randomly as fallback.") # 注释掉
        Lambda_current = np.random.randn(n_obs, n_factors) * 0.1

    # 初始化 A (状态转移矩阵) 和 Q (过程噪声协方差)
    # print("  Calculating initial A (Transition Matrix) and Q (Process Noise Cov) via VAR(1)...") # 注释掉
    try:
        var_model = VAR(factors_init_df.dropna()) # 对初始因子拟合 VAR(1)
        var_results = var_model.fit(1)
        A_current = var_results.coefs[0]       # VAR(1) 系数
        Q_current = np.cov(var_results.resid, rowvar=False) # VAR 残差协方差
        # print(f"  Initial A calculated. Shape: {A_current.shape}") # 注释掉
        # print(f"  Initial A contains NaN: {np.isnan(A_current).any()}") # 注释掉
        # print(f"  Initial Q calculated. Shape: {Q_current.shape}") # 注释掉
        # print(f"  Initial Q contains NaN: {np.isnan(Q_current).any()}") # 注释掉
    except Exception as var_e:
        # print(f"  Error fitting VAR(1) for initial A/Q: {var_e}. Using simple initialization as fallback.") # 注释掉
        A_current = np.eye(n_factors) * 0.95
        Q_current = np.eye(n_factors) * 0.1

    # 初始化 R (观测噪声协方差)
    # print("  Calculating initial R (Observation Noise Cov) from PCA residuals...") # 注释掉
    try:
        V = Vh.T
        reconstructed_z = factors_init @ V[:, :n_factors].T # 用因子重构标准化数据
        residuals_z = z - reconstructed_z                   # 标准化数据的残差
        # R 的对角线元素 = var(标准化残差) * var(原始观测)
        # var(原始观测) = std(原始观测)^2
        psi_diag = np.nanvar(residuals_z, axis=0) # 标准化残差的方差
        original_std_sq = obs_std.fillna(1.0)**2 # 原始标准差的平方 (填充 NaN)
        R_diag_current = psi_diag * original_std_sq.to_numpy()
        R_diag_current = np.maximum(R_diag_current, 1e-6) # 确保正定
        R_current = np.diag(R_diag_current)
        # print(f"  Initial R calculated. Shape: {R_current.shape}") # 注释掉
        # print(f"  Initial R contains NaN: {np.isnan(R_current).any()}") # 注释掉
    except Exception as r_e:
         # print(f"  Error calculating initial R from PCA residuals: {r_e}. Using simple initialization as fallback.") # 注释掉
         R_current = np.eye(n_obs) * 0.1 # 初始猜测

    # 初始化 B (冲击矩阵) - 保持简单
    # print("  Initializing B simply...") # 注释掉
    if n_shocks != n_factors:
        B_current = np.zeros((n_factors, n_shocks))
        min_dim = min(n_factors, n_shocks)
        B_current[:min_dim, :min_dim] = np.eye(min_dim) * 0.1
    else:
         B_current = np.eye(n_factors) * 0.1

    # 初始化 x0 和 P0 - 保持简单
    # print("  Initializing x0 as zero vector and P0 as identity matrix.") # 注释掉
    x0_current = np.zeros(n_factors)
    P0_current = np.eye(n_factors)
    # <<< 新增：保存初始值 >>>
    initial_x0 = x0_current.copy()
    initial_P0 = P0_current.copy()
    # <<< 结束新增 >>>

    # 确保 Q, R 对角线为正
    epsilon = 1e-6
    # Ensure Q_current and R_current are numpy arrays before using np.diag
    if isinstance(Q_current, pd.DataFrame): Q_current = Q_current.to_numpy()
    if isinstance(R_current, pd.DataFrame): R_current = R_current.to_numpy()
    
    # Ensure Q_current is 2D before diag (can happen if VAR fails and fallback is used)
    if Q_current.ndim == 1: Q_current = np.diag(Q_current)
    if R_current.ndim == 1: R_current = np.diag(R_current) # Should always be diag from PCA

    Q_current = np.diag(np.maximum(np.diag(Q_current), epsilon))
    R_current = np.diag(np.maximum(np.diag(R_current), epsilon))
    # === 修改结束: 基于 PCA 的初始化 ===

    # Prepare U (error/shocks term for KF)
    if error:
        # u_data = rand_Matrix(len(observation.index), n_shocks) # Replace this call
        u_data = np.random.randn(len(observation.index), n_shocks)
    else:
        u_data = np.zeros(shape=(len(observation.index), n_shocks))
    error_df = pd.DataFrame(data=u_data, columns=[f'shock{i+1}' for i in range(n_shocks)], index=observation.index)

    # --- Start EM Loop ---
    # print(f"Step 3: Starting EM loop for {n_iter} iterations...") # 注释掉
    for i in range(n_iter):
        # E-Step: Run Kalman Filter and Smoother
        kf = KalmanFilter(Z=obs_centered, U=error_df, A=A_current, B=B_current, H=Lambda_current, state_names=state_names, x0=x0_current, P0=P0_current, Q=Q_current, R=R_current)
        fis = FIS(kf)

        # M-Step: Update parameters using smoothed factors
        em = EMstep(fis, n_shocks) # Should return arrays
        # print(f"    [EM Debug Iter {i+1}/{n_iter}] M-Step Completed.") # DEBUG REMOVED

        # Update parameters for next iteration
        A_current = np.array(em.A)
        B_current = np.array(em.B)
        Lambda_current = np.array(em.Lambda)
        Q_current = np.array(em.Q) # Make sure EMstep returns updated Q
        R_current = np.array(em.R) # Make sure EMstep returns updated R
        
        # --- EM Debug: Check updated parameters --- 
        # print(f"    [EM Debug Iter {i+1}] Updated A contains NaN/Inf: {np.isnan(A_current).any() or np.isinf(A_current).any()}") # DEBUG REMOVED
        # print(f"    [EM Debug Iter {i+1}] Updated Lambda contains NaN/Inf: {np.isnan(Lambda_current).any() or np.isinf(Lambda_current).any()}") # DEBUG REMOVED
        # print(f"    [EM Debug Iter {i+1}] Updated Q contains NaN/Inf: {np.isnan(Q_current).any() or np.isinf(Q_current).any()}") # DEBUG REMOVED
        # print(f"    [EM Debug Iter {i+1}] Updated R contains NaN/Inf: {np.isnan(R_current).any() or np.isinf(R_current).any()}") # DEBUG REMOVED
        
        # Check diagonal of Q and R
        # if np.any(np.diag(Q_current) <= 0): # DEBUG REMOVED
        #     print(f"    [EM Debug Iter {i+1}] WARNING: Updated Q diagonal has non-positive values: {np.diag(Q_current)[np.diag(Q_current) <= 0]}") # DEBUG REMOVED
        # if np.any(np.diag(R_current) <= 0): # DEBUG REMOVED
        #     print(f"    [EM Debug Iter {i+1}] WARNING: Updated R diagonal has non-positive values: {np.diag(R_current)[np.diag(R_current) <= 0]}") # DEBUG REMOVED
        # Check condition number of A (optional, can be slow)
        # cond_A = np.linalg.cond(A_current) # DEBUG REMOVED
        # print(f"    [EM Debug Iter {i+1}] Updated A condition number: {cond_A}") # DEBUG REMOVED
        # --- End EM Debug ---
        
        # Update initial state estimate for next KF run (optional, can use smoothed start)
        x0_current = np.array(em.x_sm.iloc[0])
        P0_current = fis.P_sm[0] # Use smoothed covariance at time 0
        # print(f"    [EM Debug Iter {i+1}] Updated x0 contains NaN: {np.isnan(x0_current).any()}") # DEBUG REMOVED
        # print(f"    [EM Debug Iter {i+1}] Updated P0 contains NaN: {np.isnan(P0_current).any()}") # DEBUG REMOVED
        # print(f"    [EM Debug Iter {i+1}] --------- EM Iteration End ---------") # DEBUG REMOVED

    # Final results use the parameters from the last EM step
    # print("Running final Kalman Filter with optimized parameters...") # 注释掉：减少输出
    kf_final = KalmanFilter(Z=obs_centered, U=error_df, A=A_current, B=B_current, H=Lambda_current, state_names=state_names, x0=x0_current, P0=P0_current, Q=Q_current, R=R_current)
    
    # The smoothed state `x_sm` is from the last M-step's input (`em.x_sm`)
    final_x_sm_to_return = em.x_sm 

    # --- ADD FINAL CHECK FOR NaNs in Smoothed Factors --- 
    nan_in_final_factors = False
    if isinstance(final_x_sm_to_return, pd.DataFrame):
        if final_x_sm_to_return.isnull().values.any():
            nan_in_final_factors = True
            print("\nERROR DETECTED in DFM_EMalgo: Final smoothed factors (x_sm) contain NaNs!")
            print(f"  NaN count per factor:\n{final_x_sm_to_return.isnull().sum()}")
            # Optional: Find first NaN index
            try:
                 first_nan_idx = final_x_sm_to_return[final_x_sm_to_return.isnull().any(axis=1)].index[0]
                 print(f"  First NaN occurred around index: {first_nan_idx}")
            except IndexError:
                 print("  Could not determine first NaN index.")
    elif isinstance(final_x_sm_to_return, np.ndarray):
        if np.isnan(final_x_sm_to_return).any():
             nan_in_final_factors = True
             print("\nERROR DETECTED in DFM_EMalgo: Final smoothed factors (x_sm as ndarray) contain NaNs!")
             print(f"  Total NaN count: {np.isnan(final_x_sm_to_return).sum()}")
    
    # Decide whether to raise an error or just return the NaN-containing object
    # Raising an error might be cleaner for the calling function (run_tuning)
    # if nan_in_final_factors:
    #     raise ValueError("Final smoothed factors computed by DFM_EMalgo contained NaNs.")
    # --- END FINAL CHECK ---

    # 在返回之前存储计算出的 obs_mean
    # <<< 修改：传递保存的初始值 initial_x0 和 initial_P0 >>>
    return DFMEMResultsWrapper(A=A_current, B=B_current, Q=Q_current, R=R_current, Lambda=Lambda_current, x=kf_final.x, x_sm=final_x_sm_to_return, z=kf_final.z, obs_mean=obs_mean, x0=initial_x0, P0=initial_P0)
    # <<< 结束修改 >>>

class DFMEMResultsWrapper():
    # 添加 obs_mean 到 __init__ 参数和属性
    def __init__(self, A, B, Q, R, Lambda, x, x_sm, z, obs_mean, x0, P0):
        self.A = A
        self.B = B
        self.Q = Q
        self.R = R
        self.Lambda = Lambda
        self.x = x
        self.x_sm = x_sm
        self.z = z
        self.obs_mean = obs_mean # 存储原始观测均值
        self.x0 = x0
        self.P0 = P0
    
def RevserseTranslate(Factors, miu, Lambda, names):
    # Factors is DataFrame, miu is Series, Lambda is array (n_obs, n_factors)
    # observation = Factors @ Lambda.T + miu
    factors_arr = np.array(Factors)
    lambda_arr = np.array(Lambda)
    # Ensure lambda_arr has shape (n_obs, n_factors)
    if lambda_arr.shape[0] == Factors.shape[1]: # Check if factors are columns in Lambda
        lambda_arr = lambda_arr.T

    # Perform calculation: Factors (time x n_factors) @ Lambda.T (n_factors x n_obs)
    translated_data = factors_arr @ lambda_arr.T
    # Add mean back (broadcasting)
    observation_arr = translated_data + miu.to_numpy() # Convert Series to array for broadcasting

    observation_df=pd.DataFrame(data=observation_arr, columns=names, index=Factors.index)
    return observation_df