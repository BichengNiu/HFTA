# -*- coding: utf-8 -*-
"""
Created on Mon Feb 17 11:00:22 2020

@author: Hogan
"""
from scipy.interpolate import interp1d
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
from datetime import datetime
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
from Functions import *
from scipy.linalg import solve_discrete_lyapunov # Added for potential future use
from scipy.stats import multivariate_normal # For LLF calculation


class KalmanFilterResultsWrapper():
    def __init__(self, x_minus, x, z, Kalman_gain, P, P_minus, state_names, A, log_likelihood):
        self.x_minus = x_minus
        self.x = x
        self.z = z
        self.Kalman_gain = Kalman_gain
        self.P = P
        self.state_names = state_names
        self.A = A
        self.P_minus = P_minus
        self.log_likelihood = log_likelihood
        
def KalmanFilter(Z, U, A, B, H, state_names, x0, P0, Q, R):
    #x_t = A*x_{t-1} + B*u_t + Q
    #z_t = H*x_t + R
    # Q is process noise covariance
    # R is measurement noice covariance
    
    measurement_names = Z.columns
    
    timestamp = Z.index
    n_time = len(Z.index)
    n_state = len(state_names)
    n_obs_total = Z.shape[1]
    # Convert to NumPy array instead of matrix
    z = np.array(Z.to_numpy(), dtype=float) # Ensure float
    u = np.array(U.to_numpy(), dtype=float) # Ensure float
    A = np.asarray(A, dtype=float) # Ensure float
    B = np.asarray(B, dtype=float) # Ensure float
    H = np.asarray(H, dtype=float) # Ensure float
    Q = np.asarray(Q, dtype=float) # Ensure float
    R = np.asarray(R, dtype=float) # Ensure float
    x0 = np.asarray(x0, dtype=float) # Ensure float
    P0 = np.asarray(P0, dtype=float) # Ensure float

    "out initialization"
    # predictions
    # Use np.zeros which returns an array
    x = np.zeros(shape=(n_time, n_state))
    x[0]=x0
    x_minus = np.zeros(shape=(n_time, n_state))
    x_minus[0] = x0
    
    # factor errors
    P = [np.zeros_like(P0) for _ in range(n_time)] # Initialize list of arrays
    P[0] = P0
    P_minus = [np.zeros_like(P0) for _ in range(n_time)] # Initialize list of arrays
    P_minus[0] = P0
    
    # Kalman gains
    K = [np.zeros((n_state, n_obs_total)) for _ in range(n_time)] # Initialize list of arrays, use n_obs_total initially
    
    # --- 新增: LLF 初始化 ---
    total_log_likelihood = 0.0
    llf_calculation_failed = False
    # --- 结束新增 ---

    "Kalman Filter"
    for i in range(1,n_time):
        # check if z is available
        ix = np.where(~np.isnan(z[i, :]))[0] # Get indices of non-NaN measurements at time i
        n_obs_avail = len(ix)
        
        if n_obs_avail == 0: # No observations at this time step
             # Just propagate prediction, no update, no LLF contribution from measurement
             x_minus[i] = A @ x[i-1] + B @ u[i]
             P_minus[i] = A @ P[i-1] @ A.T + Q
             x[i] = x_minus[i] # State update = state prediction
             P[i] = P_minus[i] # Covariance update = covariance prediction
             K[i] = np.zeros((n_state, n_obs_total)) # Kalman gain is zero conceptually
             continue # Skip to next time step

        z_t = z[i, ix]  # Select available observations
        H_t = H[ix, :]  # Select corresponding rows of H matrix
        R_t = R[ix][:, ix] # Select corresponding block of R matrix
        
        # --- 维度检查 --- 
        if H_t.shape != (n_obs_avail, n_state) or R_t.shape != (n_obs_avail, n_obs_avail):
            print(f"错误 KF iter {i}: H_t ({H_t.shape}) 或 R_t ({R_t.shape}) 维度不匹配 (预期 H:({n_obs_avail},{n_state}), R:({n_obs_avail},{n_obs_avail}) )")
            llf_calculation_failed = True
            break
        # --- 结束维度检查 --- 

        "prediction step"
        x_minus[i] = A @ x[i-1] + B @ u[i]
        P_minus[i] = A @ P[i-1] @ A.T + Q
        
        "update step"
        # Calculate innovation covariance: F_t = H_t @ P_minus[i] @ H_t.T + R_t
        try:
            # --- 健壮性: 确保 P_minus[i] 是有限的 --- 
            if not np.all(np.isfinite(P_minus[i])):
                 print(f"错误 KF iter {i}: P_minus[{i}] 包含 NaN 或 Inf。停止滤波。")
                 llf_calculation_failed = True
                 break
            # --- 结束健壮性 --- 
            temp = H_t @ P_minus[i] @ H_t.T + R_t 
        except ValueError as ve:
            print(f"错误 KF iter {i}: 计算 temp (H@P_@H.T+R) 时发生矩阵乘法错误: {ve}")
            print(f"  Shapes: H_t={H_t.shape}, P_minus={P_minus[i].shape}, R_t={R_t.shape}")
            llf_calculation_failed = True
            break

        # --- 新增: temp 矩阵检查 ---
        if not np.all(np.isfinite(temp)):
            print(f"警告 KF iter {i}: temp 矩阵包含 NaN 或 Inf。停止滤波。")
            llf_calculation_failed = True
            break
            
        cond_temp = np.linalg.cond(temp)
        if cond_temp > 1e12:
            print(f"警告 KF iter {i}: temp 矩阵条件数过高 ({cond_temp:.2e})，接近奇异。停止滤波。")
            llf_calculation_failed = True
            break
        # --- 结束 temp 检查 ---
        
        # Calculate Kalman Gain K and update state/covariance
        try:
            # temp_inv = temp.I # Old matrix inverse
            temp_inv = np.linalg.inv(temp)
            K_t = P_minus[i] @ H_t.T @ temp_inv # Calculate gain for available obs
            
            # --- 新增: 处理 K_t 存储 --- 
            # Store the calculated gain in the correct columns of the full K matrix
            # This part is tricky if n_obs_avail < n_obs_total. 
            # Storing full K might not be standard. Let's store K_t directly.
            # Reinitialize K as a list of potentially different shaped arrays
            if i == 1: # Only reinitialize K structure once if needed
                 K = [None] * n_time # Use None as placeholder
            K[i] = K_t # Store the potentially smaller K matrix
            # --- 结束 K_t 存储修改 ---
            
            P[i] = P_minus[i] - K_t @ H_t @ P_minus[i]
            # Calculate innovation (prediction error): v_t = z_t - H_t @ x_minus[i]
            v_t = z_t - H_t @ x_minus[i]
            x[i] = x_minus[i] + K_t @ v_t

            # --- 新增: LLF 计算 ---
            sign, logdet = np.linalg.slogdet(temp)
            if sign <= 0:
                 print(f"警告 KF iter {i}: temp 矩阵行列式非正 ({sign=}, {logdet=})。无法计算 LLF。停止滤波。")
                 llf_calculation_failed = True
                 break
            # Ensure v_t is treated as a column vector for quadratic form if needed
            # v_t shape is (n_obs_avail,), temp_inv is (n_obs_avail, n_obs_avail)
            # Quadratic form: v_t @ temp_inv @ v_t.T (if v_t is row vec)
            quadratic_term = v_t @ temp_inv @ v_t 
            llf_i = -0.5 * (n_obs_avail * np.log(2 * np.pi) + logdet + quadratic_term)
            if not np.isfinite(llf_i):
                 print(f"警告 KF iter {i}: 计算的 LLF 项非有限 ({llf_i=})。停止 LLF 累加。")
                 llf_calculation_failed = True # Mark failure but maybe continue filtering?
                 # Let's stop accumulating LLF but continue the filter if possible
                 # break # Or break here
            else:
                total_log_likelihood += llf_i
            # --- 结束 LLF 计算 ---

        except np.linalg.LinAlgError:
            print(f"错误 KF iter {i}: 求解逆(temp) 时发生奇异矩阵错误。停止滤波。")
            llf_calculation_failed = True
            break
        except ValueError as ve_update:
            print(f"错误 KF iter {i}: 更新状态/协方差时发生矩阵乘法错误: {ve_update}")
            print(f"  Shapes: K_t={K_t.shape}, H_t={H_t.shape}, P_minus={P_minus[i].shape}, v_t={v_t.shape}")
            llf_calculation_failed = True
            break
        except Exception as e_update:
             print(f"错误 KF iter {i}: 更新步骤中发生未知错误: {e_update}")
             llf_calculation_failed = True
             break
             
        # --- 新增: P[i] 对称性和正定性检查 (可选但推荐) ---
        P[i] = (P[i] + P[i].T) / 2 # Ensure symmetry
        # try:
        #     np.linalg.cholesky(P[i]) # Check positive definiteness
        # except np.linalg.LinAlgError:
        #     print(f"警告 KF iter {i}: 更新后的 P[{i}] 非正定。滤波可能不稳定。")
            # Optionally add handling here, e.g., reset P or stop
        # --- 结束 P[i] 检查 ---

    # Loop finished or broken
    # --- 修改: 处理 LLF 最终值和返回 --- 
    if llf_calculation_failed:
         print("Kalman Filter 提前终止或 LLF 计算失败。Log Likelihood 设置为 None。")
         final_log_likelihood = None
    else:
         final_log_likelihood = total_log_likelihood

    x = pd.DataFrame(data=x, index=Z.index, columns=state_names)
    x_minus = pd.DataFrame(data=x_minus, index=Z.index, columns=state_names)
    
    return KalmanFilterResultsWrapper(x_minus=x_minus, x=x, z=Z, Kalman_gain=K, P=P, P_minus=P_minus, state_names = state_names, A=A, log_likelihood=final_log_likelihood)

def FIS(res_KF):
    # Check if KF failed
    if res_KF is None or res_KF.x is None or res_KF.P is None:
        print("错误 FIS: 输入的 Kalman Filter 结果无效。")
        return None # Return None if KF failed
        
    N = len(res_KF.x.index)
    n_state = len(res_KF.x.columns)
    # Use asarray to handle potential DataFrame/array inputs
    x = np.asarray(res_KF.x)
    x_minus = np.asarray(res_KF.x_minus)
    P = res_KF.P # P is a list of arrays/matrices
    P_minus = res_KF.P_minus # P_minus is a list of arrays/matrices
    A = np.asarray(res_KF.A) # Ensure A is an array
    
    # Check if P or P_minus contain None or non-array elements after KF failure
    if any(p is None for p in P) or any(p_m is None for p_m in P_minus):
         print("错误 FIS: P 或 P_minus 包含 None 元素 (可能来自 KF 提前终止)。无法平滑。")
         return None

    # Use np.zeros
    x_sm = np.zeros(shape=(N, n_state))
    x_sm[N-1] = x[N-1]
    
    # P_sm is smoothed factors covariance matrix
    P_sm = [np.zeros((n_state, n_state)) for _ in range(N)] # Initialize list of arrays
    P_sm[N-1] = P[N-1] # P is P_{t|t} from Kalman Filter
    
    # --- ADDED: Initialize P_sm_lag1 --- 
    # P_sm_lag1[t] will store Cov(x_t, x_{t-1}|Z)
    P_sm_lag1 = [None] * N # Initialize with None, index 0 will remain None
    # --- END ADDED ---

    "fixed interval smoother,backward recursion"
    # Loop from t = N-2 down to 0
    for i in range(N-2, -1, -1):
        # Check for valid P_minus[i+1] (predicted covariance P_{t+1|t})
        P_pred_inv = None
        try:
            P_pred = P_minus[i+1] # P_{t+1|t}
            if P_pred is None or not np.all(np.isfinite(P_pred)):
                 print(f"错误 FIS iter {i}: P_minus[{i+1}] 无效 (None 或 non-finite)。停止平滑。")
                 return None # Stop smoothing if predicted covariance is invalid
            # --- ADDED: Regularization for inverse --- 
            epsilon_fis = 1e-8 # Small regularization term
            P_pred_reg = P_pred + epsilon_fis * np.identity(n_state)
            P_pred_inv = np.linalg.inv(P_pred_reg)
            # P_pred_inv = np.linalg.inv(P_pred) # Original inverse
            # --- END ADDED --- 
        except np.linalg.LinAlgError:
            print(f"错误 FIS iter {i}: P_minus[{i+1}] 求逆时奇异。停止平滑。")
            # Optionally check condition number:
            # cond_num_pred = np.linalg.cond(P_pred)
            # print(f"  P_minus[{i+1}] 的条件数: {cond_num_pred:.2e}")
            return None # Stop smoothing if matrix is singular

        P_filt = P[i] # P_{t|t}
        if P_filt is None or not np.all(np.isfinite(P_filt)):
             print(f"错误 FIS iter {i}: P[{i}] 无效 (None 或 non-finite)。停止平滑。")
             return None

        # Calculate smoother gain J_t = P_{t|t} @ A.T @ P_{t+1|t}^{-1}
        J = P_filt @ A.T @ P_pred_inv
        
        # Update smoothed state: x_sm_t = x_t + J @ (x_sm_{t+1} - x_minus_{t+1})
        x_sm[i] = x[i] + J @ (x_sm[i+1] - x_minus[i+1])
        
        # Update smoothed covariance: P_sm_t = P_t + J @ (P_sm_{t+1} - P_minus_{t+1}) @ J.T
        P_sm[i] = P_filt + J @ (P_sm[i+1] - P_pred) @ J.T
        # Ensure symmetry
        P_sm[i] = (P_sm[i] + P_sm[i].T) / 2
        
        # --- ADDED: Calculate P_sm_lag1 --- 
        # P_{t+1,t}^s = Cov(x_{t+1}, x_t | Z) = P_{t+1}^s @ J_t^T
        try:
            P_sm_tplus1_t = P_sm[i+1] @ J.T # Calculate Cov(x_{t+1}, x_t | Z)
            P_sm_lag1[i+1] = P_sm_tplus1_t # Store it at index t+1
        except Exception as e_lag1:
            print(f"错误 FIS iter {i}: 计算 P_sm_lag1[{i+1}] 时出错: {e_lag1}")
            return None # Stop if lag calculation fails
        # --- END ADDED ---

    # Store results in a pandas DataFrame
    x_sm_df = pd.DataFrame(data=x_sm, index=res_KF.x.index, columns=res_KF.x.columns)
    z_df = res_KF.z # Pass original observations
    
    # Return the smoothed results including P_sm and P_sm_lag1
    # --- MODIFIED: Pass P_sm_lag1 to wrapper --- 
    return SKFResultsWrapper(x_sm=x_sm_df, P_sm=P_sm, z=z_df, P_sm_lag1=P_sm_lag1)

class SKFResultsWrapper():
    # --- MODIFIED: Add P_sm_lag1 --- 
    def __init__(self, x_sm, P_sm, z, P_sm_lag1):
        self.x_sm = x_sm
        self.P_sm = P_sm
        self.z = z
        self.P_sm_lag1 = P_sm_lag1 # Store the smoothed lag-1 covariance

def EMstep(res_SKF, n_shocks):
    f = res_SKF.x_sm
    y = res_SKF.z
    
    Lambda = calculate_factor_loadings(y, f)
    A = calculate_prediction_matrix(f)
    Q = calculate_shock_matrix(f)
    
    y_arr = np.asarray(y)
    f_arr = np.asarray(f)
    Lambda_arr = np.asarray(Lambda)
    
    y_pred = f_arr @ Lambda_arr.T
    resid_arr = y_arr - y_pred
    
    resid_df = pd.DataFrame(resid_arr, index=y.index, columns=y.columns)
    R_pd = resid_df.cov()
    R = np.diag(np.diag(R_pd.fillna(0).to_numpy()))
    
    return EMstepResultsWrapper(Lambda=Lambda, A=A, B=None, Q=Q, R=R, x_sm=f, z=y)
    
class EMstepResultsWrapper():
    # --- IMPORTANT --- 
    # This class definition is likely duplicated from DynamicFactorModel.py
    # and should probably be REMOVED from this file (DiscreteKalmanFilter.py).
    # Assuming the primary wrapper is in DynamicFactorModel.py, we comment this out.
    pass # Commented out - See DynamicFactorModel.py for the active wrapper
    # def __init__(self, Lambda, A, B, Q, R, x_sm, z):
    #     self.Lambda = Lambda
    #     self.A = A
    #     self.B = B
    #     self.Q = Q
    #     self.R = R
    #     self.x_sm = x_sm # Include smoothed state for potential diagnostics
    #     self.z = z # Include observations for potential diagnostics

    