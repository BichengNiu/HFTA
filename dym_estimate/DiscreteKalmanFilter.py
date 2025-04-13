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
import Functions as funcs

# --- 全局调试标志 ---
DEBUG_KALMAN = False # 设置为 True 以打印详细的卡尔曼滤波器内部状态

class KalmanFilterResultsWrapper():
    def __init__(self, x_minus, x, z, Kalman_gain, P, P_minus, state_names, A):
        self.x_minus = x_minus
        self.x = x
        self.z = z
        self.Kalman_gain = Kalman_gain
        self.P = P
        self.state_names = state_names
        self.A = A
        self.P_minus = P_minus
        
def KalmanFilter(Z, U, A, B, H, state_names, x0, P0, Q, R):
    #x_t = A*x_{t-1} + B*u_t + Q
    #z_t = H*x_t + R
    # Q is process noise covariance
    # R is measurement noice covariance
    
    measurement_names = Z.columns
    timestamp = Z.index
    n_time = len(Z.index)
    n_state = len(state_names)
    # Convert inputs to NumPy arrays
    z = np.array(Z.to_numpy())
    u = np.array(U.to_numpy())
    A = np.array(A)
    B = np.array(B)
    H = np.array(H)
    x0 = np.array(x0)
    P0 = np.array(P0)
    Q = np.array(Q)
    R = np.array(R)

    "out initialization"
    # Use np.zeros directly which returns arrays
    x = np.zeros(shape=(n_time, n_state))
    x[0, :] = x0 # Assign initial state
    x_minus = np.zeros(shape=(n_time, n_state))
    x_minus[0, :] = x0 # Assign initial prediction

    # Factor errors - Store as list of arrays
    P = [np.zeros_like(P0) for _ in range(n_time)]
    P[0] = P0
    P_minus = [np.zeros_like(P0) for _ in range(n_time)]
    P_minus[0] = P0

    # Kalman gains - Store as list of arrays (or determine shape if fixed)
    # Assuming K shape is (n_state, n_measurements_available)
    # Size might vary, so list of arrays is safer
    K = [None] * n_time # Initialize with None or appropriate zeros

    for i in range(1, n_time):
        ix = np.where(~np.isnan(z[i, :]))[0]

        if len(ix) == 0:
            x_prev_col = x[i-1, :].reshape(-1, 1)
            u_col = u[i, :].reshape(-1, 1)
            x_minus_pred = A @ x_prev_col + B @ u_col
            x_minus[i, :] = x_minus_pred.flatten()
            P_minus_raw = A @ P[i-1] @ A.T + Q
            p_jitter = np.eye(P_minus_raw.shape[0]) * 1e-6 # Small jitter value
            P_minus[i] = P_minus_raw + p_jitter
            x[i, :] = x_minus[i, :]
            P[i] = P_minus[i]
            K[i] = np.zeros((n_state, H.shape[0]))
            continue

        z_t = z[i, ix]
        H_t = H[ix, :]
        R_t = R[np.ix_(ix, ix)]

        x_prev_col = x[i-1, :].reshape(-1, 1)
        u_col = u[i, :].reshape(-1, 1)

        "prediction step"
        x_minus_pred = A @ x_prev_col + B @ u_col
        x_minus[i, :] = x_minus_pred.flatten()
        P_minus_raw = A @ P[i-1] @ A.T + Q
        p_jitter = np.eye(P_minus_raw.shape[0]) * 1e-6 # Small jitter value
        P_minus[i] = P_minus_raw + p_jitter

        "update step"
        innovation_cov = H_t @ P_minus[i] @ H_t.T + R_t
        jitter = np.eye(innovation_cov.shape[0]) * 1e-4
        
        try:
            inv_innovation_cov = np.linalg.pinv(innovation_cov + jitter)
            K_t = P_minus[i] @ H_t.T @ inv_innovation_cov
            K[i] = K_t 
        except np.linalg.LinAlgError as svd_error:
            # --- 修改: 注释掉所有错误细节打印 --- 
            # print(f"\n--- SVD Error Details ---")
            # print(f"Timestamp: {timestamp[i]}")
            # print(f"Time step index i: {i}")
            # print(f"Available observation indices (ix): {ix}")
            # print(f"Shape of innovation_cov: {innovation_cov.shape}")
            # print(f"Innovation Covariance Matrix (before jitter):")
            # print(innovation_cov)
            # print(f"H_t shape: {H_t.shape}")
            # print(f"P_minus[{i}] shape: {P_minus[i].shape}")
            # print(f"R_t shape: {R_t.shape}")
            # print(f"Original SVD Error: {svd_error}")
            # print("--- End SVD Error Details ---\n")
            raise # Re-raise the exception

        x_minus_col = x_minus[i, :].reshape(-1, 1)
        z_t_col = z_t.reshape(-1, 1)
        innovation = z_t_col - H_t @ x_minus_col
        x_updated = x_minus_col + K_t @ innovation
        x[i, :] = x_updated.flatten()

        I_mat = np.eye(n_state)
        P[i] = (I_mat - K_t @ H_t) @ P_minus[i]

    x = pd.DataFrame(data=x, index=Z.index, columns=state_names)
    x_minus = pd.DataFrame(data=x_minus, index=Z.index, columns=state_names)

    
    return KalmanFilterResultsWrapper(x_minus=x_minus, x=x, z=Z, Kalman_gain=K, P=P, P_minus=P_minus, state_names = state_names, A=A)

def FIS(res_KF):
    N = len(res_KF.x.index)
    n_state = len(res_KF.x.columns)
    # Convert inputs to arrays
    x = np.array(res_KF.x)
    x_minus = np.array(res_KF.x_minus)
    # P and P_minus are lists of arrays from KF
    P = res_KF.P
    P_minus = res_KF.P_minus
    A = np.array(res_KF.A)

    # Initialize smoothed results as arrays
    x_sm = np.zeros((N, n_state))
    x_sm[N-1, :] = x[N-1, :]

    P_sm = [np.zeros_like(P[0]) for _ in range(N)]
    P_sm[N-1] = P[N-1]

    J = [None] * (N - 1) # Smoother gains

    for i in reversed(range(N-1)):
        try:
            P_minus_inv = np.linalg.pinv(P_minus[i+1])
            J_k = P[i] @ A.T @ P_minus_inv
            J[i] = J_k
        except np.linalg.LinAlgError as inv_error:
            print(f"  [FIS Smoother Iter {i}] Error inverting P_minus[{i+1}] for smoother: {inv_error}")
            raise

        P_sm[i] = P[i] + J_k @ (P_sm[i+1] - P_minus[i+1]) @ J_k.T

        x_col = x[i, :].reshape(-1, 1)
        x_sm_next_col = x_sm[i+1, :].reshape(-1, 1)
        x_minus_next_col = x_minus[i+1, :].reshape(-1, 1)
        x_sm_updated = x_col + J_k @ (x_sm_next_col - x_minus_next_col)
        x_sm[i, :] = x_sm_updated.flatten()

    x_sm = pd.DataFrame(data=x_sm, index=res_KF.x.index, columns=res_KF.x.columns)
    
    return SKFResultsWrapper(x_sm=x_sm, P_sm=P_sm,z=res_KF.z)
        
class SKFResultsWrapper():
    def __init__(self, x_sm, P_sm, z):
        self.x_sm = x_sm
        self.P_sm = P_sm
        self.z = z
    

    
def EMstep(res_SKF, n_shocks):
    """Performs the M-step of the EM algorithm, updating parameters.

    Handles NaNs in observables when calculating R.
    Assumes calculate_factor_loadings handles NaNs for Lambda.
    Might still face issues if calculate_shock_matrix produces non-PSD Q.
    """
    f = res_SKF.x_sm # Smoothed factors (DataFrame, n_time x n_factors)
    y = res_SKF.z    # Original centered observables (DataFrame, n_time x n_obs, with NaNs)
    n_obs = y.shape[1]
    n_time = y.shape[0]
    n_factors = f.shape[1]

    # --- REMOVE DEBUG CHECK ---
    # if f.isnull().values.any(): print("  [DEBUG EMstep] Input smoothed factors f contains NaN!")
    # --- END REMOVE DEBUG CHECK ---

    # print("  [DEBUG EMstep] Calculating Lambda...") # REMOVED
    Lambda = funcs.calculate_factor_loadings(y, f)
    # if np.isnan(Lambda).any(): print("  [DEBUG EMstep] Calculated Lambda contains NaN!") # REMOVED
    
    # print("  [DEBUG EMstep] Calculating A...") # REMOVED
    A = funcs.calculate_prediction_matrix(f)
    # if np.isnan(A).any(): print("  [DEBUG EMstep] Calculated A contains NaN!") # REMOVED

    # print("  [DEBUG EMstep] Calculating B and Q...") # REMOVED
    B, Q = funcs.calculate_shock_matrix(f, A, n_shocks)
    # if np.isnan(B).any(): print("  [DEBUG EMstep] Calculated B contains NaN!") # REMOVED
    # if np.isnan(Q).any(): print("  [DEBUG EMstep] Calculated Q contains NaN!") # REMOVED

    # print("  [DEBUG EMstep] Calculating R...") # REMOVED
    f_arr = f.to_numpy()
    y_arr = y.to_numpy()
    R_diag = np.full(n_obs, np.nan)
    lambda_valid = isinstance(Lambda, np.ndarray) and Lambda.shape == (n_obs, n_factors) and not np.isnan(Lambda).any()

    if lambda_valid:
        predicted_y = f_arr @ Lambda.T 
        resid = y_arr - predicted_y
        for j in range(n_obs):
            valid_idx = ~np.isnan(y_arr[:, j])
            if np.sum(valid_idx) > 1:
                var_j = np.var(resid[valid_idx, j])
                R_diag[j] = max(var_j, 1e-6)
            # else: # Keep commented unless debugging R
                # print(f"Warning: Not enough valid points ({np.sum(valid_idx)}) to estimate R[{j},{j}] reliably.")
    # else: # Keep commented unless debugging R
         # print(f"Warning: Initial Lambda calculation failed or produced NaNs. R matrix calculation might be unreliable. Using fallback for R.")

    if np.isnan(R_diag).any():
        nan_indices = np.where(np.isnan(R_diag))[0]
        # --- Use a slightly larger floor value --- 
        R_diag = np.nan_to_num(R_diag, nan=1e-5) # Increased floor
        R_diag = np.maximum(R_diag, 1e-5)      # Increased floor
        # --- End floor increase --- 

    R = np.diag(R_diag)

    # --- Ensure Q diagonal is also sufficiently positive --- 
    Q_diag = np.diag(Q).copy() # Get diagonal
    Q_diag = np.maximum(Q_diag, 1e-5) # Apply floor
    Q_stable = np.diag(Q_diag) # Reconstruct stable Q
    # --- End Q stabilization --- 
    
    # Return the stabilized Q
    return EMstepResultsWrapper(Lambda=Lambda, A=A, B=B, Q=Q_stable, R=R, x_sm=f, z=y)
    
class EMstepResultsWrapper():
    def __init__(self, Lambda, A, B, Q, R, x_sm, z):
        self.Lambda = Lambda
        self.A = A
        self.B = B
        self.Q = Q
        self.R = R
        self.x_sm = x_sm
        self.z = z

    