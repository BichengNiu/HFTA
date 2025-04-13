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
from sklearn.decomposition import PCA
from scipy.interpolate import interp1d
from DiscreteKalmanFilter import *
from Functions import *

def DFM(observation, n_factors, n_shocks):
    # model: y_it = miu_i + lambda_i*F_t + e_it
    # observables: columns are times series variables in interests, rows are time points
    #   *** ASSUMPTION: `observation` DataFrame is already standardized and NaN-handled ***
    # n_factors: number of common factors, can't be larger than number of input variables
    n_obs = observation.shape[1]
    if n_obs <= n_factors:
        print(f'Error: number of common factors ({n_factors}) >= number of variables ({n_obs})')
        return None # Or raise an error
    # n_time: number of time periods (observations)
    n_time = len(observation.index)
    # function returns two elements: common factors in the format of that of observables, transform matrix n_factors*n_observables

    "pca"
    # --- MODIFICATION: Simplify PCA call and factor calculation --- 
    # Remove internal standardization/demeaning
    # x = np.array(observation - observation.mean())
    # z = np.array((observation - observation.mean()) / observation.std())
    obs_array = np.array(observation.values, dtype=float) # Use the input directly
    
    # --- ADDED: Final NaN fill before PCA --- 
    if np.isnan(obs_array).any():
        # print("Warning: NaNs detected in input array just before PCA. Filling with 0.") # Commented out
        obs_array = np.nan_to_num(obs_array, nan=0.0)
    # --- END ADDED --- 
    
    # Call the updated calculate_pca function, which expects pre-processed data
    D, V, S = calculate_pca(obs_array, n_factors) 
    if D is None or V is None: # Check if PCA failed
         print("Error: calculate_pca failed within DFM.")
         return None

    # Calculate factors: F = X @ V (where X is the standardized observation data)
    # V contains the eigenvectors (principal components loadings)
    factors = obs_array @ V # Shape (n_time, n_factors)
    CommonFactors = pd.DataFrame(data=factors, index=observation.index, columns=[f'Factor{i+1}' for i in range(n_factors)])
    # --- END MODIFICATION ---

    # Ensure S, V, D are NumPy arrays for calculation (They should be from calculate_pca)
    # S_arr = np.array(S)
    # V_arr = np.array(V)
    # D_arr = np.array(D)
    # --- MODIFICATION: Calculate Psi directly from input data and factors --- 
    # Calculate residuals: e_it = y_it - lambda_i*F_t
    # First calculate Lambda using the *standardized* factors and observations
    Lambda_pca = calculate_factor_loadings(observation, CommonFactors)
    if Lambda_pca is None:
        print("Error: Failed to calculate factor loadings (Lambda_pca) after PCA.")
        return None
        
    # Calculate residuals based on standardized data
    predicted_obs_std = CommonFactors.values @ Lambda_pca.T # F @ Lambda^T
    residuals_std = obs_array - predicted_obs_std
    
    # Calculate Psi = diag(cov(residuals_std))
    if residuals_std.shape[0] < 2:
         # print("Warning: Not enough residuals after PCA to calculate Psi. Using identity matrix.") # Commented out
         Psi = np.identity(n_obs)
    else:
         try:
              cov_residuals = np.cov(residuals_std, rowvar=False)
              if cov_residuals.ndim == 0 and n_obs == 1: # Handle scalar case
                  Psi_diag_val = cov_residuals.item()
                  Psi = np.array([[max(Psi_diag_val, 1e-9)]]) # Ensure non-negative
              elif cov_residuals.shape == (n_obs, n_obs):
                  Psi_diag = np.diag(cov_residuals)
                  Psi = np.diag(np.maximum(Psi_diag, 1e-9)) # Ensure diagonal elements are positive
              else:
                  # print(f"Warning: Covariance matrix of PCA residuals has unexpected shape {cov_residuals.shape}. Using identity matrix for Psi.") # Commented out
                  Psi = np.identity(n_obs)
         except Exception as e_psi:
             # print(f"Error calculating Psi from PCA residuals: {e_psi}. Using identity matrix.") # Keep errors
             Psi = np.identity(n_obs)
    # --- END MODIFICATION ---
    
    # Psi_diag = np.diag(S_arr - V_arr @ D_arr @ V_arr.T)
    # Psi = np.diag(Psi_diag) # Create diagonal matrix from diagonal elements

    # Use @ for matrix multiplication
    # factors = (V_arr.T @ z.T).T # Old calculation using internally standardized z
    # CommonFactors = pd.DataFrame(data=factors, index=observation.index, columns=['Factor' + str(i+1) for i in range(n_factors)])

    "Factors loadings"
    # Recalculate Lambda using the derived factors and original (but standardized) observations
    Lambda = calculate_factor_loadings(observation, CommonFactors)
    if Lambda is None:
         print("Error: Failed to calculate final Lambda after PCA.")
         return None

    "VAR"
    # model: F_t = A*F_{t-1} + B*u_t
    # calculate matrix A and B
    A = calculate_prediction_matrix(CommonFactors)
    # --- MODIFICATION: Call updated shock matrix function and compute B ---
    # 1. Calculate Q (covariance of residuals) using the updated function
    Q = calculate_shock_matrix(CommonFactors) # Only returns Q (k x k) now
    if Q is None:
         print("Error: Failed to calculate Q matrix in DFM.")
         return None

    n_factors_q = Q.shape[0] # Use shape from Q
    if n_factors_q == 0:
        print("Error: Q matrix has zero dimensions.")
        return None
    
    # 2. Compute B from Q using eigenvalue decomposition (logic moved here)
    # Ensure n_shocks is defined and valid
    if n_shocks > n_factors_q:
         # print(f"Warning: n_shocks ({n_shocks}) > n_factors ({n_factors_q}). Setting n_shocks = n_factors.") # Commented out
         n_shocks = n_factors_q
    elif n_shocks <= 0:
         # print(f"Warning: n_shocks ({n_shocks}) is invalid. Setting n_shocks = n_factors ({n_factors_q}).") # Commented out
         n_shocks = n_factors_q

    try:
        eigenvalues, eigenvectors = np.linalg.eigh(Q) # Use eigh for symmetric matrices
        # Sort eigenvalues in descending order
        sorted_indices = np.argsort(eigenvalues)[::-1]
        evalues_sorted = eigenvalues[sorted_indices]
        evectors_sorted = eigenvectors[:, sorted_indices]

        # Select top n_shocks
        top_evalues = evalues_sorted[:n_shocks]
        # Ensure eigenvalues are non-negative before sqrt
        top_evalues_nonneg = np.maximum(top_evalues, 0)
        top_evectors = evectors_sorted[:, :n_shocks] # Shape (n_factors, n_shocks)

        # B = M * sqrt(diag(top_evalues))
        B = top_evectors @ np.diag(np.sqrt(top_evalues_nonneg)) # Shape (n_factors, n_shocks)

    except np.linalg.LinAlgError:
        print("Error: Eigen decomposition failed for residual covariance matrix Q in DFM.")
        print(f"Returning zero matrix for B with shape ({n_factors_q}, {n_shocks}).")
        B = np.zeros((n_factors_q, n_shocks))
    
    # Assign Q (calculated above) to Sigma for the wrapper object
    Sigma = Q 
    # --- END MODIFICATION ---
    # B, Sigma = calculate_shock_matrix(CommonFactors, A, n_shocks) # OLD CALL

    return DFMResultsWrapper(common_factors=CommonFactors, Lambda=Lambda, A=A, B=B, idiosyncratic_covariance=Psi, prediction_covariance=Sigma, obs_mean=observation.mean())

class DFMResultsWrapper():
    def __init__(self, common_factors, Lambda, A, B,  idiosyncratic_covariance, prediction_covariance, obs_mean):
        self.common_factors = common_factors
        self.Lambda = Lambda
        self.A = A
        self.B = B
        self.idiosyncratic_covariance = idiosyncratic_covariance
        self.prediction_covariance = prediction_covariance
        self.obs_mean = obs_mean

def DFM_EMalgo(observation, n_factors, n_shocks, n_iter, error='False'):
    dfm_pca = DFM(observation, n_factors, n_shocks)
    if dfm_pca is None: # Check if DFM failed
        print("DFM (PCA initialization) failed.")
        return None
    # --- Modification: Add check for common_factors ---
    if dfm_pca.common_factors is None or dfm_pca.common_factors.empty:
        print("Error: DFM (PCA initialization) did not produce valid common factors.")
        return None
    if dfm_pca.common_factors.shape[0] < 1: # Need at least one row for x0
        print("Error: DFM (PCA initialization) produced common factors with zero rows.")
        return None
    # --- End Modification ---

    if error == 'True':
        error_data = rand_Matrix(len(observation.index), n_shocks) # Assuming rand_Matrix returns np.array now
        error_df = pd.DataFrame(data=error_data, columns=['shock'+str(i+1) for i in range(n_shocks)], index=observation.index)
    else:
        error_df = pd.DataFrame(data=np.zeros(shape=(len(observation.index), n_shocks)), columns=['shock'+str(i+1) for i in range(n_shocks)], index=observation.index)

    # Use PCA results for initialization
    Lambda_init = np.array(dfm_pca.Lambda) # Ensure NumPy array
    A_init = np.array(dfm_pca.A)
    B_init = np.array(dfm_pca.B)
    Q_init = np.array(dfm_pca.prediction_covariance)
    R_init = np.array(dfm_pca.idiosyncratic_covariance)
    x0_init = np.array(dfm_pca.common_factors.iloc[0].values) # Get initial state as array
    P0_init = calculate_covariance(dfm_pca.common_factors) # Assuming returns np.array
    # --- Modification: Add check for P0_init --- 
    if P0_init is None:
        # ... error handling ...
        n_factors_check = dfm_pca.common_factors.shape[1]
        # print(f"Warning: Using identity matrix ({n_factors_check}x{n_factors_check}) as fallback for P0_init.") # Commented out
        P0_init = np.identity(n_factors_check)
        # return None # Alternatively, fail here
    # --- End Modification ---

    obs_demeaned = observation - observation.mean()

    # --- Run Initial Kalman Filter --- 
    kf = KalmanFilter(Z=obs_demeaned, U=error_df, A=A_init, B=B_init, H=Lambda_init, 
                      state_names=dfm_pca.common_factors.columns, x0=x0_init, P0=P0_init, Q=Q_init, R=R_init)
    if kf is None or kf.log_likelihood is None:
        print("Error: Initial Kalman Filter failed or did not produce LLF. Cannot start EM.")
        # Optionally return PCA results or None
        # return dfm_pca 
        return None

    # --- Run Initial Smoother --- 
    fis = FIS(kf)
    if fis is None:
         print("Error: Initial Smoother (FIS) failed. Cannot start EM.")
         return None

    em = None # Initialize em to handle potential loop issues
    # --- ADDED: Store previous parameters AND KF/FIS objects --- 
    previous_kf_params = {
        'H': Lambda_init.copy(),
        'A': A_init.copy(),
        'Q': Q_init.copy(),
        'R': R_init.copy(),
        'B': B_init.copy(),
        'x0': x0_init.copy(),
        'P0': P0_init.copy()
    }
    last_successful_fis = fis # Store initial FIS
    last_successful_kf = kf   # Store initial KF (which has the initial LLF)
    # --- END ADDED ---

    for i in range(n_iter):
        # print(f"--- EM Iteration {i+1} ---") # Commented out
        try:
            # E-step is implicitly done by FIS providing smoothed estimates from PREVIOUS iteration
            # M-step: Calculate updated parameters based on smoothed estimates
            em = EMstep(fis, n_shocks)
            if em is None:
                print(f"EM step calculation failed at iteration {i+1}. Stopping EM iterations.")
                # Return results from the *previous* successful iteration 
                print("  EMstep returned None. Returning results from previous successful iteration.")
                final_fis = last_successful_fis
                final_kf = last_successful_kf # Use stored kf
                llf_to_return = final_kf.log_likelihood if hasattr(final_kf, 'log_likelihood') else None
                return DFMEMResultsWrapper(A=previous_kf_params['A'], B=previous_kf_params['B'], Q=previous_kf_params['Q'],
                                            R=previous_kf_params['R'], Lambda=previous_kf_params['H'],
                                            x=final_kf.x, x_sm=final_fis.x_sm, z=final_fis.z, log_likelihood=llf_to_return)

            # --- Parameter Validation --- 
            Lambda_em = np.array(em.Lambda)
            A_em = np.array(em.A)
            B_em = np.array(em.B) if em.B is not None else None # Handle potential None B
            Q_em = np.array(em.Q)
            R_em = np.array(em.R)

            if not np.all(np.isfinite(Lambda_em)) or not np.all(np.isfinite(A_em)) or \
               not np.all(np.isfinite(Q_em)) or not np.all(np.isfinite(R_em)) or \
               (B_em is not None and not np.all(np.isfinite(B_em))): # Check B if not None
                print(f"Error: Non-finite values detected in estimated parameters at iteration {i+1}. Stopping.")
                # print(f"  NaNs: Lambda={np.isnan(Lambda_em).any()}, A={np.isnan(A_em).any()}, Q={np.isnan(Q_em).any()}, R={np.isnan(R_em).any()}, B={np.isnan(B_em).any() if B_em is not None else 'N/A'}") # Commented out
                print("  Non-finite params detected. Returning results from previous successful iteration.")
                final_fis = last_successful_fis
                final_kf = last_successful_kf # Use stored kf
                llf_to_return = final_kf.log_likelihood if hasattr(final_kf, 'log_likelihood') else None
                return DFMEMResultsWrapper(A=previous_kf_params['A'], B=previous_kf_params['B'], Q=previous_kf_params['Q'],
                                            R=previous_kf_params['R'], Lambda=previous_kf_params['H'],
                                            x=final_kf.x, x_sm=final_fis.x_sm, z=final_fis.z, log_likelihood=llf_to_return)

            # --- Calculate new x0 and P0 --- 
            if fis.x_sm is not None and not fis.x_sm.empty:
                x0_new = np.array(fis.x_sm.iloc[0].values)
            else:
                 x0_new = previous_kf_params['x0'].copy()
            # Use smoothed P from the last time step for P0
            P0_new = np.asarray(fis.P_sm[-1]) if fis.P_sm else previous_kf_params['P0'].copy() 
            if not np.all(np.isfinite(P0_new)):
                print(f"Warning: P0_new became non-finite at iteration {i+1}. Reverting to previous P0.")
                P0_new = previous_kf_params['P0'].copy()

            # --- Re-run Kalman Filter with NEW parameters to calculate LLF --- 
            kf_params_current = {
                'H': Lambda_em, 'A': A_em, 'B': B_em, 'Q': Q_em, 'R': R_em,
                'x0': x0_new, 'P0': P0_new, 'state_names': kf.state_names # Reuse state names
            }
            # print(f"  Re-running KF with updated params for LLF calc (Iter {i+1})...") # Commented out
            kf_current = KalmanFilter(Z=obs_demeaned, U=error_df, **kf_params_current)
            if kf_current is None or kf_current.log_likelihood is None:
                 print(f"Warning: Kalman Filter failed or did not produce LLF in iteration {i+1}. Stopping EM.")
                 print("  KF/LLF failed. Returning results from previous successful iteration.")
                 final_fis = last_successful_fis
                 final_kf = last_successful_kf # Use stored kf
                 llf_to_return = final_kf.log_likelihood if hasattr(final_kf, 'log_likelihood') else None
                 return DFMEMResultsWrapper(A=previous_kf_params['A'], B=previous_kf_params['B'], Q=previous_kf_params['Q'],
                                            R=previous_kf_params['R'], Lambda=previous_kf_params['H'],
                                            x=final_kf.x, x_sm=final_fis.x_sm, z=final_fis.z, log_likelihood=llf_to_return)
            
            current_llf = kf_current.log_likelihood
            # Optional: Check for LLF convergence
            prev_llf = last_successful_kf.log_likelihood
            if prev_llf is not None and np.isfinite(prev_llf) and np.isfinite(current_llf):
                llf_change = abs(current_llf - prev_llf) / (abs(prev_llf) + 1e-6)
                # print(f"  Iter {i+1} LLF: {current_llf:.4f} (Change: {llf_change:.6f})") # Commented out
                # Add convergence check here if needed
                # if llf_change < convergence_threshold:
                #     print(f"EM converged at iteration {i+1} based on LLF.")
                #     last_successful_kf = kf_current # Store the final KF result
                #     # Run FIS one last time
                #     final_fis = FIS(last_successful_kf)
                #     if final_fis is None:
                #          print("Warning: Final FIS run after convergence failed.")
                #          final_fis = last_successful_fis # Fallback
                #     break # Exit EM loop
            # else:
                # print(f"  Iter {i+1} LLF: {current_llf:.4f}") # Commented out

            # --- Run Smoother (FIS) for the *next* iteration's E-step --- 
            # print(f"  Running FIS with updated KF results for next iter (Iter {i+1})...") # Commented out
            fis = FIS(kf_current) # Run smoother on the KF results obtained with updated parameters
            if fis is None or fis.x_sm is None:
                 print(f"Error: FIS failed (e.g., smoothing produced None) in iteration {i+1}. Stopping.")
                 print("  FIS failed. Returning results from previous successful iteration.")
                 final_fis = last_successful_fis
                 final_kf = last_successful_kf # Use stored kf
                 llf_to_return = final_kf.log_likelihood if hasattr(final_kf, 'log_likelihood') else None
                 return DFMEMResultsWrapper(A=previous_kf_params['A'], B=previous_kf_params['B'], Q=previous_kf_params['Q'],
                                            R=previous_kf_params['R'], Lambda=previous_kf_params['H'],
                                            x=final_kf.x, x_sm=final_fis.x_sm, z=final_fis.z, log_likelihood=llf_to_return)

            # --- Store current successful state --- 
            previous_kf_params = {
                'H': Lambda_em.copy(), 'A': A_em.copy(), 'Q': Q_em.copy(),
                'R': R_em.copy(), 'B': B_em.copy() if B_em is not None else None, 
                'x0': x0_new.copy(), 'P0': P0_new.copy()
            }
            last_successful_fis = fis # Store the current FIS object
            last_successful_kf = kf_current # Store the KF object that produced the current LLF and FIS inputs
            # LLF is already stored within kf_current

        except Exception as e_iter:
             print(f"!!! Unhandled Exception in EM Iteration {i+1}: {e_iter}")
             import traceback
             traceback.print_exc()
             print("  Stopping EM iterations due to unexpected error.")
             print("  Exception occurred. Returning results from previous successful iteration.")
             final_fis = last_successful_fis
             final_kf = last_successful_kf # Use stored kf
             llf_to_return = final_kf.log_likelihood if hasattr(final_kf, 'log_likelihood') else None
             return DFMEMResultsWrapper(A=previous_kf_params['A'], B=previous_kf_params['B'], Q=previous_kf_params['Q'],
                                         R=previous_kf_params['R'], Lambda=previous_kf_params['H'],
                                         x=final_kf.x, x_sm=final_fis.x_sm, z=final_fis.z, log_likelihood=llf_to_return)

    # --- EM Loop Finished (or converged) --- 
    # print(f"EM algorithm finished {i+1} iterations. Returning results from last successful iteration.") # Commented out
    final_fis = last_successful_fis
    final_kf = last_successful_kf # Use the final stored KF
    final_llf = final_kf.log_likelihood if hasattr(final_kf, 'log_likelihood') else None
    return DFMEMResultsWrapper(A=previous_kf_params['A'], B=previous_kf_params['B'], Q=previous_kf_params['Q'],
                               R=previous_kf_params['R'], Lambda=previous_kf_params['H'],
                               x=final_kf.x, 
                               x_sm=final_fis.x_sm, 
                               z=final_fis.z, log_likelihood=final_llf) # Pass the final LLF


class DFMEMResultsWrapper():
    def __init__(self, A, B, Q, R, Lambda, x, x_sm, z, log_likelihood):
        self.A = A
        self.B = B
        self.Q = Q
        self.R = R
        self.Lambda = Lambda
        self.x = x
        self.x_sm = x_sm
        self.z = z
        self.log_likelihood = log_likelihood

def RevserseTranslate(Factors, miu, Lambda, names):
    # Ensure Lambda and Factors are NumPy arrays for multiplication
    Lambda_arr = np.array(Lambda)
    Factors_arr = np.array(Factors)
    # Use @ for matrix multiplication
    observation_data = (Lambda_arr @ Factors_arr.T).T
    observation = pd.DataFrame(data=observation_data, columns=names, index=Factors.index)
    # Ensure miu aligns correctly for addition (might need broadcasting or reshaping)
    if isinstance(miu, pd.Series):
        miu_arr = miu.values
    else:
        miu_arr = np.array(miu) # Assuming miu is array-like
    # Ensure shapes are compatible for broadcasting or align columns if miu is 1D array per column
    observation = observation + miu_arr # Add mean back
    return observation

def EMstep(res_SKF, n_shocks):
    # Expectation-Maximization step using Smoothed Kalman Filter results
    # E-step is handled by the Smoother (FIS) providing E[x_t|Z] = x_sm, etc.
    # M-step: Update parameters Lambda, A, Q, R based on smoothed estimates

    if res_SKF is None or res_SKF.x_sm is None or res_SKF.P_sm is None or res_SKF.P_sm_lag1 is None or res_SKF.z is None:
        print("Error: EMstep received invalid or incomplete Smoother results (res_SKF).")
        return None

    x_sm = np.array(res_SKF.x_sm) # Smoothed state E[x_t|Z]
    P_sm = np.array(res_SKF.P_sm) # Smoothed state covariance Var(x_t|Z)
    Z = np.array(res_SKF.z) # Original observations used by filter/smoother

    T = x_sm.shape[0] # Number of time points
    n_vars = Z.shape[1] # Number of observed variables
    n_factors = x_sm.shape[1] # Number of factors

    # --- ADDED: Check for NaN/Inf in smoothed factors BEFORE calculations ---
    if np.any(np.isnan(x_sm)) or np.any(np.isinf(x_sm)):
        print("Error: EMstep detected NaN or Inf in smoothed factors (x_sm). Cannot proceed.")
        return None
    # --- END ADDED ---

    # M-step calculations based on Hamilton (Time Series Analysis, 1994), Chapter 13.4

    # 1. Calculate terms needed for Lambda and R
    #    E[F_t F_t' | Z] = E[F_t|Z]E[F_t|Z]' + Var(F_t|Z) = x_sm[t] @ x_sm[t].T + P_sm[t]
    #    Sum over T: sum_FtFt = sum_{t=1}^T E[F_t F_t' | Z]
    #    Sum over T: sum_ZtFt = sum_{t=1}^T Z_t E[F_t | Z]' = sum_{t=1}^T Z_t @ x_sm[t].T (Handle NaNs in Z)
    #    Sum over T: sum_ZtZt = sum_{t=1}^T Z_t @ Z_t' (Handle NaNs in Z)
    sum_FtFt = np.zeros((n_factors, n_factors))
    sum_ZtFt = np.zeros((n_vars, n_factors))
    sum_ZtZt = np.zeros((n_vars, n_vars))
    T_valid_obs = 0 # Count time points with at least one valid observation

    for t in range(T):
        xt_sm = x_sm[t, :].reshape(-1, 1) # Shape (n_factors, 1)
        Pt_sm = P_sm[t, :, :]          # Shape (n_factors, n_factors)
        zt = Z[t, :].reshape(-1, 1)    # Shape (n_vars, 1)

        # Calculate E[F_t F_t' | Z] for this time step
        E_FtFt_t = xt_sm @ xt_sm.T + Pt_sm
        sum_FtFt += E_FtFt_t

        # Handle potential NaNs in observations (Z_t) for sum_ZtFt and sum_ZtZt
        valid_obs_mask = ~np.isnan(zt).flatten() # Mask for non-NaN observations at time t
        zt_valid = zt[valid_obs_mask] # Select valid observations
        if zt_valid.size > 0:
            T_valid_obs += 1
            xt_sm_T = xt_sm.T # Shape (1, n_factors)

            # Update sum_ZtFt considering only valid observations
            sum_ZtFt[valid_obs_mask, :] += zt_valid @ xt_sm_T

            # Update sum_ZtZt considering only valid observations
            outer_prod = zt_valid @ zt_valid.T
            valid_indices = np.ix_(valid_obs_mask, valid_obs_mask)
            sum_ZtZt[valid_indices] += outer_prod


    # 2. Update Lambda (Factor Loadings)
    #    Lambda = (sum_ZtFt) @ (sum_FtFt)^(-1)
    FtFt_sum = sum_FtFt
    ZtFt_sum = sum_ZtFt

    # --- MODIFIED: Add diagonal loading for numerical stability ---
    epsilon_lambda = 1e-8 # Small regularization term
    FtFt_sum_reg = FtFt_sum + epsilon_lambda * np.identity(n_factors)
    try:
        # --- ADDED: Check FtFt_sum_reg for NaN/Inf before inversion --- 
        if not np.all(np.isfinite(FtFt_sum_reg)):
            print("Error: FtFt_sum matrix contains NaN or Inf values BEFORE condition number calculation. Returning None.")
            return None
        # --- END ADDED --- 
        inv_FtFt_sum = np.linalg.inv(FtFt_sum_reg)
        Lambda_new = ZtFt_sum @ inv_FtFt_sum # Shape (n_vars, n_factors)
    except np.linalg.LinAlgError as e:
        print(f"Error: Failed to calculate factor loadings (Lambda) in EM step. Matrix inversion failed: {e}")
        # Check condition number if inversion fails
        try:
            cond_num = np.linalg.cond(FtFt_sum_reg)
            print(f"Condition number of FtFt_sum (regularized) was: {cond_num:.2e}")
        except Exception as cond_e:
            print(f"Could not calculate condition number for FtFt_sum_reg: {cond_e}")
        # Return None or previous Lambda? For now, return None to signal failure.
        return None
    # --- END MODIFIED ---

    # 3. Update R (Idiosyncratic Covariance) - Diagonal matrix
    #    R_ii = (1/T_eff) * sum_{t=1}^T [ (Z_it - Lambda_i * E[F_t|Z])^2 + (Lambda_i * Var(F_t|Z) * Lambda_i') ]
    #    Simplifies to: R_ii = (1/T_eff) * [ sum(Z_it^2) - 2*Lambda_i*sum(Z_it*E[F_t|Z]) + Lambda_i*sum(E[F_tF_t'|Z])*Lambda_i' ]_ii
    #    Which further simplifies using previous sums:
    #    R = (1/T_eff) * [ sum_ZtZt - Lambda_new @ sum_ZtFt.T - sum_ZtFt @ Lambda_new.T + Lambda_new @ sum_FtFt @ Lambda_new.T ]
    #    We only need the diagonal elements. T_eff is number of effective observations for each variable.
    #    Let's use T_valid_obs as an approximation for T_eff across all variables for simplicity.

    if T_valid_obs == 0:
        print("Error: No valid observations found across all time steps in EMstep. Cannot update R.")
        # Use previous R or an identity matrix? Returning None signals failure.
        return None

    term1 = sum_ZtZt
    term2 = Lambda_new @ ZtFt_sum.T
    term3 = ZtFt_sum @ Lambda_new.T
    term4 = Lambda_new @ FtFt_sum @ Lambda_new.T

    R_full = (term1 - term2 - term3 + term4) / T_valid_obs
    R_diag_new = np.diag(R_full)
    # Ensure non-negativity (variances cannot be negative)
    R_diag_new = np.maximum(R_diag_new, 1e-9) # Floor variance at a small positive number
    R_new = np.diag(R_diag_new)

    # 4. Calculate terms needed for A and Q
    #    E[F_t F_{t-1}' | Z] = E[F_t|Z]E[F_{t-1}|Z]' + Cov(F_t, F_{t-1}|Z) = x_sm[t] @ x_sm[t-1].T + P_sm_lag1[t]
    #    Sum over T (from t=2): sum_FtFtm1 = sum_{t=2}^T E[F_t F_{t-1}' | Z]
    #    Sum over T (from t=2): sum_Ftm1Ftm1 = sum_{t=2}^T E[F_{t-1} F_{t-1}' | Z] (Use sum_FtFt excluding last period)

    sum_FtFtm1 = np.zeros((n_factors, n_factors))
    sum_Ftm1Ftm1 = np.zeros((n_factors, n_factors))

    # Recalculate E[F_t F_t' | Z] for t=1 to T-1 for sum_Ftm1Ftm1
    for t in range(T - 1): # Sum from t=1 to T-1 (corresponding to index 0 to T-2)
        xt_sm = x_sm[t, :].reshape(-1, 1)
        Pt_sm = P_sm[t, :, :]
        E_FtFt_t = xt_sm @ xt_sm.T + Pt_sm
        sum_Ftm1Ftm1 += E_FtFt_t

    # Calculate sum_FtFtm1 for t=2 to T
    for t in range(1, T): # Sum from t=2 to T (corresponding to index 1 to T-1)
        xt_sm = x_sm[t, :].reshape(-1, 1)
        xtm1_sm = x_sm[t-1, :].reshape(-1, 1)
        # --- MODIFIED: Access P_sm_lag1 directly from list and check --- 
        # Pt_sm_lag1 = P_sm_lag1[t, :, :] # Old code using converted array
        Pt_sm_lag1 = res_SKF.P_sm_lag1[t] # Access directly from the list
        if Pt_sm_lag1 is None:
            print(f"Error: EMstep encountered None for P_sm_lag1 at index t={t}. Cannot update A/Q.")
            return None
        if not isinstance(Pt_sm_lag1, np.ndarray) or Pt_sm_lag1.shape != (n_factors, n_factors):
            print(f"Error: EMstep encountered invalid P_sm_lag1 at index t={t} (Type: {type(Pt_sm_lag1)}, Shape: {getattr(Pt_sm_lag1, 'shape', 'N/A')}).")
            return None
        # --- END MODIFIED ---
        E_FtFtm1_t = xt_sm @ xtm1_sm.T + Pt_sm_lag1
        sum_FtFtm1 += E_FtFtm1_t


    # 5. Update A (Transition Matrix)
    #    A = (sum_FtFtm1) @ (sum_Ftm1Ftm1)^(-1)
    Ftm1Ftm1_sum = sum_Ftm1Ftm1
    FtFtm1_sum = sum_FtFtm1

    # --- MODIFIED: Add diagonal loading for numerical stability ---
    epsilon_a = 1e-8 # Small regularization term
    Ftm1Ftm1_sum_reg = Ftm1Ftm1_sum + epsilon_a * np.identity(n_factors)
    try:
        # --- ADDED: Check Ftm1Ftm1_sum_reg for NaN/Inf before inversion --- 
        if not np.all(np.isfinite(Ftm1Ftm1_sum_reg)):
            print("Error: Ftm1Ftm1_sum matrix contains NaN or Inf values. Returning None.")
            return None
        # --- END ADDED --- 
        inv_Ftm1Ftm1_sum = np.linalg.inv(Ftm1Ftm1_sum_reg)
        A_new = FtFtm1_sum @ inv_Ftm1Ftm1_sum # Shape (n_factors, n_factors)
    except np.linalg.LinAlgError as e:
        print(f"Error: Failed to calculate transition matrix (A) in EM step. Matrix inversion failed: {e}")
        try:
            cond_num = np.linalg.cond(Ftm1Ftm1_sum_reg)
            print(f"Condition number of Ftm1Ftm1_sum (regularized) was: {cond_num:.2e}")
        except Exception as cond_e:
             print(f"Could not calculate condition number for Ftm1Ftm1_sum_reg: {cond_e}")
        return None
    # --- END MODIFIED ---


    # 6. Update Q (State Error Covariance)
    #    Q = (1/(T-1)) * [ sum_{t=2}^T E[F_tF_t'|Z] - A_new * sum_{t=2}^T E[F_{t-1}F_t'|Z] ]
    #    Q = (1/(T-1)) * [ sum_FtFt(exclude t=1) - A_new @ sum_FtFtm1.T ]
    #    Where sum_FtFt(exclude t=1) = sum_FtFt - E[F_1F_1'|Z]
    if T <= 1:
         print("Error: Not enough time points (T<=1) to update Q in EMstep.")
         return None

    x1_sm = x_sm[0, :].reshape(-1, 1)
    P1_sm = P_sm[0, :, :]
    E_F1F1_1 = x1_sm @ x1_sm.T + P1_sm
    sum_FtFt_excl_t1 = sum_FtFt - E_F1F1_1

    Q_term1 = sum_FtFt_excl_t1
    Q_term2 = A_new @ FtFtm1_sum.T
    Q_new = (Q_term1 - Q_term2) / (T - 1)
    # Ensure Q is symmetric (it should be theoretically, but enforce numerically)
    Q_new = (Q_new + Q_new.T) / 2
    # Ensure Q is positive semi-definite (e.g., by flooring eigenvalues, though simpler just to use it)
    # We might need to handle potential negative definite Q later if KF fails.


    # 7. Update B (Shock Matrix) - Assuming B is related to Q
    #    If Q = B @ B.T, we can find B via Cholesky or Eigen decomposition of Q_new.
    #    Using Eigen decomposition to select n_shocks:
    try:
        eigenvalues, eigenvectors = np.linalg.eigh(Q_new) # Use eigh for symmetric matrices
        # Sort eigenvalues in descending order
        sorted_indices = np.argsort(eigenvalues)[::-1]
        evalues_sorted = eigenvalues[sorted_indices]
        evectors_sorted = eigenvectors[:, sorted_indices]

        # Select top n_shocks (ensure n_shocks <= n_factors)
        n_shocks_eff = min(n_shocks, n_factors)
        top_evalues = evalues_sorted[:n_shocks_eff]
        # Ensure eigenvalues are non-negative before sqrt
        top_evalues_nonneg = np.maximum(top_evalues, 0)
        top_evectors = evectors_sorted[:, :n_shocks_eff] # Shape (n_factors, n_shocks_eff)

        # B = M * sqrt(diag(top_evalues))
        B_new = top_evectors @ np.diag(np.sqrt(top_evalues_nonneg)) # Shape (n_factors, n_shocks_eff)

        # If n_shocks was originally > n_factors, pad B with zeros? Or adjust n_shocks?
        # Let's assume the Kalman Filter part can handle B with n_shocks_eff columns.

    except np.linalg.LinAlgError as e:
        print(f"Error: Eigen decomposition failed for updated Q matrix in EM step: {e}. Cannot update B.")
        return None

    # Return updated parameters
    # Using EMstepResultsWrapper might be cleaner if defined elsewhere
    return EMstepResultsWrapper(Lambda=Lambda_new, A=A_new, B=B_new, Q=Q_new, R=R_new, x_sm=x_sm, z=Z) # Pass original Z


# --- MODIFIED: Define EMstepResultsWrapper ---
class EMstepResultsWrapper():
    def __init__(self, Lambda, A, B, Q, R, x_sm, z):
        self.Lambda = Lambda
        self.A = A
        self.B = B
        self.Q = Q
        self.R = R
        self.x_sm = x_sm # Include smoothed state for potential diagnostics
        self.z = z # Include observations for potential diagnostics
# --- END MODIFIED ---