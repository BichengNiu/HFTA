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
from scipy.linalg import solve_discrete_lyapunov
from scipy.stats import norm

def expand_data(data, step, freq='M'):
    dates = pd.date_range(data.index[0], periods=len(data.index)+step, freq=freq)
    data_expanded = pd.DataFrame(data=np.nan,index=dates,columns=data.columns)
    data_expanded.iloc[:len(data.index)] = data.values
    
    return data_expanded

def import_data(file_name, sheet_name, start=0, interpolation=False, encoding='gb18030'):
    try:
        # Use read_excel which is the current standard
        Temp = pd.read_excel(file_name, sheet_name=sheet_name) # engine='openpyxl' might be needed
    except FileNotFoundError:
        print(f"Error: File not found at {file_name}")
        return None
    except Exception as e:
        # Catch other potential errors like bad sheet name
        print(f"Error reading Excel file '{file_name}', sheet '{sheet_name}': {e}")
        return None

    if start >= len(Temp):
        print(f"Error: Start row {start} is out of bounds for sheet '{sheet_name}' with {len(Temp)} rows.")
        return None

    # Assuming the first column is the index and the rest are data
    res = Temp.iloc[start:, 1:]
    res.index = pd.to_datetime(Temp.iloc[start:, 0], errors='coerce') # Ensure datetime index
    res = res[res.index.notna()] # Remove rows where date conversion failed

    # Convert data columns to numeric, coercing errors
    for col in res.columns:
        res[col] = pd.to_numeric(res[col], errors='coerce')

    if interpolation:
        # Use the updated DataInterpolation function
        res_interpolated = DataInterpolation(res, 0, len(res.index), 'cubic')
        if res_interpolated is not None:
            res = res_interpolated.dropna(axis=0, how='any')
        else:
            print(f"Warning: Interpolation failed for sheet '{sheet_name}'. Returning original data after numeric conversion.")
            res = res.dropna(axis=0, how='any') # Drop rows with any NaNs if interpolation failed

    return res

def transform_data(data, method):
    transform=pd.DataFrame(data=np.nan,index=data.index,columns=data.columns)
    if method=='MoM':
        for i in range(5,len(data.index)):
            # Ensure index is datetime for comparison
            if isinstance(data.index, pd.DatetimeIndex):
                # Find the latest date that is at least 30 days before the current date i
                mask = data.index <= (data.index[i] - timedelta(days=30))
                if mask.any():
                    last_month_data = data.loc[mask].iloc[-1]
                    current_data = data.iloc[i]
                    # Avoid division by zero or near-zero
                    if np.abs(last_month_data).any() > 1e-9:
                         transform.iloc[i] = (current_data - last_month_data) / last_month_data * 100
                    else:
                         transform.iloc[i] = np.nan # Or handle differently
                else:
                    transform.iloc[i] = np.nan # Not enough history
            else:
                 print("Warning: Data index is not DatetimeIndex in transform_data. MoM transform skipped.")
                 return data # Return original if index is wrong type
    elif method == 'YoY': # Example: Add Year-over-Year
         if isinstance(data.index, pd.DatetimeIndex):
             transform = data.pct_change(periods=52) * 100 # Assuming weekly data for YoY
         else:
              print("Warning: Data index is not DatetimeIndex in transform_data. YoY transform skipped.")
              return data
    elif method == 'LogDiff': # Example: Add Log Difference
         transform = np.log(data.clip(lower=1e-9)).diff()
    # Add more methods as needed

    return transform

def plot_compare(history, forecast, title, fig_size=[24,16], line_width=3.0, font_size='xx-large'):
    plt.rcParams['figure.figsize'] = (fig_size[0], fig_size[1])
    plt.rcParams['lines.markersize'] = 6
    plt.rcParams['image.cmap'] = 'gray'
    
    plt.figure()
    plt.plot(history.index, history, color='r', label='observed', linewidth=line_width)
    plt.plot(forecast.index, forecast, color='k', label='predicted', linewidth=line_width)
    plt.legend()
    plt.xlabel('Date')
    plt.ylabel('Value') # Add Y-axis label
    plt.title(title, fontweight='bold', fontsize=font_size)
    plt.grid(True) # Add grid
    plt.tight_layout() # Adjust layout
    plt.show()
    
    # No return needed as plt.show() displays the plot

def DataInterpolation(data, start, end, method):
    # data must be a time series dataframe with a DatetimeIndex
    if not isinstance(data.index, pd.DatetimeIndex):
        print("Error in DataInterpolation: Data index must be a DatetimeIndex.")
        return None

    n_row, n_col = data.shape
    # Use np.array instead of np.mat
    res_array = data.values.copy() # Work on a copy of the underlying numpy array

    if start < 0 or end > n_row or start >= end:
        print(f"Error in DataInterpolation: Invalid start/end range ({start}, {end}) for data length {n_row}.")
        return None
    
    for i in range(n_col):
        col_data = data.iloc[start:end, i]
        # Need numeric index for interp1d, use integer positions
        numeric_index = np.arange(len(col_data))
        not_null_mask = col_data.notnull()
        valid_indices = numeric_index[not_null_mask]
        valid_values = col_data[not_null_mask].values

        if len(valid_indices) < 2: # Need at least two points to interpolate
            # print(f"Warning: Not enough non-NaN points ({len(valid_indices)}) to interpolate column {data.columns[i]}. Skipping.")
            continue # Keep original NaNs for this column

        lower_bound_idx = valid_indices.min()
        upper_bound_idx = valid_indices.max()

        try:
            # Create interpolation function using integer positions and valid values
            f_interp = interp1d(valid_indices, valid_values, kind=method, bounds_error=False, fill_value="extrapolate")

            # Indices to interpolate over (all integer positions within the valid range)
            indices_to_interpolate = np.arange(lower_bound_idx, upper_bound_idx + 1)

            # Interpolate using the function
            interpolated_values = f_interp(indices_to_interpolate)

            # Place the interpolated values back into the result array at correct positions
            # Ensure indices match the slice `start:end` used for `col_data`
            res_array[start + lower_bound_idx : start + upper_bound_idx + 1, i] = interpolated_values

        except ValueError as e:
            print(f"Error during interpolation for column {data.columns[i]}: {e}")
            continue # Skip this column if interpolation fails

    # Return as a DataFrame with the original index and columns
    res_df = pd.DataFrame(res_array, index=data.index, columns=data.columns)
    return res_df

def rand_Matrix(n_row, n_col):
    # Return a NumPy array directly
    return np.random.randn(n_row, n_col)

def calculate_factor_loadings(observables, factors):
    # Both dataframes should have the same time stamp and DatetimeIndex
    if not (isinstance(observables.index, pd.DatetimeIndex) and isinstance(factors.index, pd.DatetimeIndex)):
        print("Error: Inputs must have DatetimeIndex.")
        return None
        
    # --- ADDED: Check input NaNs before alignment ---
    # Removed DEBUG print statements
    # --- END ADDED ---

    common_index = observables.index.intersection(factors.index)
    if len(common_index) < 2: # Need sufficient overlap
        print("Error: Not enough overlapping time points.")
        return None

    obs_aligned = observables.loc[common_index]
    factors_aligned = factors.loc[common_index]

    n_time, n_obs = obs_aligned.shape
    _, n_factors = factors_aligned.shape

    # Use np.array, ensure consistent data types (float)
    # Demean observables *after* alignment and *before* NaN handling
    x = np.array(obs_aligned - obs_aligned.mean(axis=0), dtype=float)
    F = np.array(factors_aligned, dtype=float)

    # --- REVISED NaN HANDLING ---
    # Do not fill NaNs with 0 here. Calculations below will handle them.
    # We calculate FtF based on rows where F is fully non-NaN.
    # We calculate xtF based on points where BOTH x_i and F are non-NaN for each observable i.
    # print("DEBUG: Entering revised NaN handling") # Debug print

    # Find rows where F is entirely non-NaN
    f_valid_rows_mask = ~np.isnan(F).any(axis=1)
    # print(f"DEBUG: f_valid_rows_mask sum: {np.sum(f_valid_rows_mask)}") # Debug print
    if not np.any(f_valid_rows_mask):
        print("Error: No time points where all factors are non-NaN. Cannot calculate FtF sum.")
        return None
    F_valid = F[f_valid_rows_mask, :]
    # print(f"DEBUG: F_valid shape: {F_valid.shape}") # Debug print
    if F_valid.shape[0] < n_factors: # Need enough valid points
         print(f"Error: Not enough valid factor rows ({F_valid.shape[0]}) to potentially compute non-singular FtF for {n_factors} factors.")
         return None

    # Calculate F'F sum using only valid rows of F
    FtF_sum = F_valid.T @ F_valid # Result (n_factors, n_factors)
    # print(f"DEBUG: FtF_sum shape: {FtF_sum.shape}") # Debug print

    # --- 添加：检查 FtF_sum 是否包含 NaN/Inf ---
    if not np.all(np.isfinite(FtF_sum)):
        print(f"Error: FtF_sum matrix contains NaN or Inf values BEFORE condition number calculation. Returning None.")
        return None
    # --- 结束添加 ---

    # Check condition number for singularity
    cond_ftf = np.linalg.cond(FtF_sum)
    if cond_ftf > 1e12: # Threshold for near-singularity
        print(f"Error: F_valid\'F_valid matrix is near-singular (condition number {cond_ftf:.2e}) in factor loading calculation. Returning None.")
        return None

    try:
        FtF_sum_inv = np.linalg.inv(FtF_sum)
        # print(f"DEBUG: FtF_sum_inv shape: {FtF_sum_inv.shape}") # Debug print
    except np.linalg.LinAlgError:
        print("Error: F_valid\'F_valid matrix is singular (inv failed after cond check) in factor loading calculation. Returning None.")
        return None

    # Calculate sum(x_t^T * F_t) robustly for each observable
    # Initialize xtF_sum
    xtF_sum = np.zeros((n_obs, n_factors))
    # print("DEBUG: Calculating xtF_sum...") # Debug print

    for i in range(n_obs):
        x_col = x[:, i]
        # Find time points where BOTH x_col AND F (all factors) are non-NaN
        # Also need to ensure these indices are within the original common_index bounds
        x_col_non_nan_mask = ~np.isnan(x_col)
        valid_mask = x_col_non_nan_mask & f_valid_rows_mask # Combine masks
        
        # print(f"DEBUG Obs {i}: x_col_non_nan={np.sum(x_col_non_nan_mask)}, f_valid={np.sum(f_valid_rows_mask)}, common_valid={np.sum(valid_mask)}") # Debug

        if not np.any(valid_mask):
             print(f"Warning: No common valid time points for observable {i} and factors. Skipping its contribution to Lambda.")
             continue # Keep this row of xtF_sum as zeros

        x_valid = x_col[valid_mask]
        F_valid_common = F[valid_mask, :] # Use the same combined mask on F

        # print(f"DEBUG Obs {i}: x_valid len={len(x_valid)}, F_valid_common shape={F_valid_common.shape}") # Debug
        if x_valid.shape[0] < n_factors:
            print(f"Warning: Not enough common valid points ({x_valid.shape[0]}) for observable {i}. Skipping calculation.")
            continue
        
        # Check if F_valid_common is empty after filtering (shouldn't happen if np.any(valid_mask) is true, but defensive check)
        if F_valid_common.shape[0] == 0:
            print(f"Warning: F_valid_common became empty for observable {i} unexpectedly. Skipping.")
            continue

        # Calculate x_i.T @ F for valid points only
        # Ensure shapes are correct for matmul: (n_valid,) @ (n_valid, n_factors)
        # Result should be (n_factors,)
        try:
             xtF_sum[i, :] = x_valid @ F_valid_common # (n_valid,) @ (n_valid, n_factors) -> (n_factors,)
        except ValueError as e_matmul:
             print(f"Error during xtF calculation for observable {i}: {e_matmul}. Check shapes: x_valid {x_valid.shape}, F_valid_common {F_valid_common.shape}. Skipping.")
             xtF_sum[i, :] = np.nan # Mark as NaN if calculation fails

    # Lambda = sum(x_t^T * F_t) * [sum(F_t^T * F_t)]^-1
    # Lambda = (xtF_sum) @ FtF_sum_inv
    # print(f"DEBUG: Calculating final Lambda. xtF_sum shape={xtF_sum.shape}, FtF_sum_inv shape={FtF_sum_inv.shape}") # Debug
    Lambda = xtF_sum @ FtF_sum_inv # Result is (n_obs, n_factors)

    # --- Final Check for NaNs/Infs in Lambda ---
    if not np.all(np.isfinite(Lambda)):
        print("Warning: Final Lambda contains NaNs or Infs after robust calculation. Check xtF_sum and FtF_sum_inv.")
        print(f"  NaNs in xtF_sum: {np.isnan(xtF_sum).any()}, Infs: {np.isinf(xtF_sum).any()}")
        print(f"  NaNs in FtF_sum_inv: {np.isnan(FtF_sum_inv).any()}, Infs: {np.isinf(FtF_sum_inv).any()}")
        # Optionally return None or the Lambda with NaNs depending on desired behavior
        # For now, let's return None to signal failure clearly in the EM step
        return None
        
    return Lambda

def calculate_prediction_matrix(factors):
    if not isinstance(factors.index, pd.DatetimeIndex):
        print("Error: Factors must have DatetimeIndex.")
        return None

    n_time = len(factors.index)
    if n_time < 2:
        print("Error: Need at least 2 time points for prediction matrix.")
        return None

    # Use np.array
    F = np.array(factors, dtype=float)

    if np.isnan(F).any():
        print("Warning: NaNs detected in factors for prediction matrix calculation.")
        # --- Modification: Fill NaNs with 0 before calculation ---
        F = np.nan_to_num(F, nan=0.0)
        print("   Filled NaNs with 0 for calculation.")
        # --- End Modification ---

    # F_lagged is F_{t-1}, F_current is F_t
    F_lagged = F[:-1, :] # Shape (n_time-1, n_factors)
    F_current = F[1:, :]  # Shape (n_time-1, n_factors)

    # Calculate sum(F_{t-1}^T * F_{t-1})
    Ftm1tFtm1_sum = F_lagged.T @ F_lagged # Result (n_factors, n_factors)

    # --- ADDED: Check condition number for singularity ---
    cond_ftm1 = np.linalg.cond(Ftm1tFtm1_sum)
    if cond_ftm1 > 1e12: # Threshold for near-singularity
        print(f"Error: Lagged F\'F matrix is near-singular (condition number {cond_ftm1:.2e}) in prediction matrix calculation. Returning None.")
        return None
    # --- END ADDED ---

    try:
        # F_lagged.T @ F_lagged gives (n_factors, n_factors)
        # Ftm1tFtm1_sum = F_lagged.T @ F_lagged # Moved calculation up
        Ftm1tFtm1_sum_inv = np.linalg.inv(Ftm1tFtm1_sum)
    except np.linalg.LinAlgError:
        print("Error: Lagged F\'F matrix is singular (inv failed after cond check) in prediction matrix calculation. Returning None.")
        # --- REMOVED: Pseudo-inverse fallback ---
        # print("   Attempting to use pseudo-inverse for lagged F.T @ F.")
        # try:
        #     Ftm1tFtm1_sum = F_lagged.T @ F_lagged
        #     Ftm1tFtm1_sum_inv = np.linalg.pinv(Ftm1tFtm1_sum)
        #     print("   Successfully used pseudo-inverse.")
        # except np.linalg.LinAlgError:
        #     print("Error: Pseudo-inverse also failed for lagged F.T @ F. Returning None.")
        #     return None
        # --- End REMOVED ---
        return None # Return None if inv fails after condition check

    # Calculate sum(F_t^T * F_{t-1})
    # F_current.T @ F_lagged gives (n_factors, n_factors)
    FtTFtm1_sum = F_current.T @ F_lagged

    # A = sum(F_t^T * F_{t-1}) * [sum(F_{t-1}^T * F_{t-1})]^-1
    # A = (F_current.T @ F_lagged) @ Ftm1tFtm1_sum_inv
    A = FtTFtm1_sum @ Ftm1tFtm1_sum_inv # Result (n_factors, n_factors)
    
    return A

def calculate_shock_matrix(factors: pd.DataFrame) -> np.ndarray:
    """
    Estimates the shock covariance matrix Q by fitting an AR(1) model
    to each factor and calculating the covariance matrix of the residuals.

    Args:
        factors (pd.DataFrame): DataFrame of factors (time series in columns).

    Returns:
        np.ndarray: Estimated shock covariance matrix Q (k x k). Returns an
                    identity matrix if estimation fails.
    """
    if not isinstance(factors, pd.DataFrame):
        print("Warning: Input 'factors' should be a pandas DataFrame. Attempting conversion.")
        try:
            factors = pd.DataFrame(factors)
        except Exception as e:
            print(f"Error: Could not convert factors to DataFrame: {e}. Returning identity matrix.")
            # Cannot determine k without DataFrame structure or numpy array input
            # Assuming k=1 as a fallback might be wrong. Need a better way or raise error.
            # For now, let's return eye(1) if conversion fails early.
            # A better approach might require knowing expected k or raising an error.
            return np.eye(1) # Fallback, potentially incorrect k

    if factors.empty:
        print("Warning: Input 'factors' DataFrame is empty. Returning identity matrix.")
        return np.eye(1) # Or np.eye(0)? Returning eye(1) for now.

    # Ensure factors are numeric, convert if possible
    try:
        factors = factors.apply(pd.to_numeric, errors='coerce')
    except Exception as e:
         print(f"Warning: Could not convert factors to numeric: {e}. Proceeding, but NaNs may be introduced.")


    n_time, k = factors.shape

    if k == 0:
        print("Warning: Input 'factors' has 0 columns (k=0). Returning empty 0x0 matrix.")
        return np.empty((0, 0))

    if n_time <= 1:
        print(f"Warning: Not enough time points ({n_time}) to estimate AR(1) for shock matrix. Returning identity matrix.")
        return np.eye(k)

    residuals = np.full((n_time - 1, k), np.nan) # Initialize residuals array

    for i in range(k):
        factor_series = factors.iloc[:, i].values # Use .values for numpy array operations
        # Find the first non-NaN value's index
        valid_indices = np.where(~np.isnan(factor_series))[0]
        if len(valid_indices) == 0:
            print(f"Warning: Factor {i} contains only NaNs. Cannot fit AR(1). Residuals will be NaN.")
            continue # Skip this factor, residuals remain NaN

        start_idx = valid_indices[0]
        # Select data from the first valid index onwards
        series_from_start = factor_series[start_idx:]
        non_nan_mask = ~np.isnan(series_from_start)
        clean_factor = series_from_start[non_nan_mask]

        # Only proceed if there are at least 2 non-NaN values for AR(1)
        if len(clean_factor) < 2:
            print(f"Warning: Factor {i} has less than 2 non-NaN values ({len(clean_factor)}) after index {start_idx}. Cannot fit AR(1). Residuals will be NaN.")
            continue
        
        # --- ADDED: Handle constant factors explicitly ---
        is_constant = False
        if len(clean_factor) > 0:
             # Check if all non-NaN values are close to the first non-NaN value
             first_val = clean_factor[0]
             if np.allclose(clean_factor, first_val):
                 is_constant = True
        
        if is_constant:
             print(f"Warning: Factor {i} is constant. Setting residuals to 0.")
             # Find original positions corresponding to clean_factor to set residuals
             # The residuals array corresponds to time t=1 to n_time-1
             # We need to set the residuals corresponding to the time points where clean_factor was defined.
             # Since factor_series should now be NaN-free, the residuals correspond to indices 1 to n_time-1.
             residuals[:, i] = 0.0 
             continue # Skip OLS fitting for constant series
        # --- END ADDED --- 

        try:
            # Fit AR(1) model using statsmodels OLS on the cleaned data
            X = sm.add_constant(clean_factor[:-1]) # Lagged values as predictors
            y = clean_factor[1:]                    # Current values as dependent variable
            # Remove missing='drop' as input should be clean
            # model = sm.OLS(y, X, missing='drop') 
            model = sm.OLS(y, X) 
            results = model.fit()

            # Predict using the original factor series to align residuals correctly
            # Create lagged series including NaNs
            factor_lagged = np.roll(factor_series, 1)
            factor_lagged[0] = np.nan # First element has no lag

            # Prepare predictor matrix (constant and lagged factor) for original series
            X_orig = sm.add_constant(factor_lagged, prepend=True) # Shape (n_time, 2)

            # Predict where possible (where lagged value is not NaN)
            valid_prediction_indices = ~np.isnan(X_orig[:, 1]) & (np.arange(n_time) > 0) # Check lag is valid and not first row
            pred = np.full(n_time, np.nan)
            if np.any(valid_prediction_indices):
                pred[valid_prediction_indices] = results.predict(X_orig[valid_prediction_indices])

            # Calculate residuals: actual - predicted
            # Residuals are defined starting from the second time point (index 1)
            # Only calculate where actual value is not NaN
            actual_values = factor_series[1:]
            predicted_values = pred[1:] # Align predictions with actuals[1:]
            residuals_mask = ~np.isnan(actual_values) & ~np.isnan(predicted_values)

            residuals[residuals_mask, i] = actual_values[residuals_mask] - predicted_values[residuals_mask]

        except ValueError as e:
            print(f"Error fitting AR(1) or calculating residuals for factor {i}: {e}. Residuals will be NaN.")
            # Ensure residuals for this factor remain NaN if error occurs

    # --- ADDED: Debug print for residuals before cov ---
    # Removed DEBUG print statement
    # --- END ADDED ---

    # Calculate covariance matrix of residuals, handling NaNs
    if np.all(np.isnan(residuals)):
        print("Warning: All calculated residuals are NaN. Cannot compute covariance matrix. Returning identity matrix.")
        Q = np.eye(k)
    else:
        # Use pandas to calculate covariance, handling NaNs pairwise
        try:
            residuals_df = pd.DataFrame(residuals)
            Q_pd = residuals_df.cov() # Pandas handles NaNs by default (pairwise deletion)
            Q = Q_pd.to_numpy()

            # Check if Q contains NaNs after pandas calculation (can happen if a column is all NaN)
            if np.isnan(Q).any():
                print("Warning: Covariance matrix contains NaNs after calculation. Attempting to fill NaNs.")
                # Fill NaNs with 0 - maybe not the best strategy, but simple
                Q = np.nan_to_num(Q, nan=0.0)
                # Ensure diagonal elements corresponding to non-NaN residual columns are positive
                # If diagonal is zero after filling, set to 1 (or small positive number)
                for i in range(k):
                     # Check if original residuals for this factor were NOT all NaN
                    if not np.all(np.isnan(residuals[:, i])) and np.isclose(Q[i, i], 0):
                        print(f"  Setting diagonal element Q[{i},{i}] to 1.0 as it was zero after NaN fill.")
                        Q[i, i] = 1.0 # Or a small positive number like 1e-6

                # If the whole matrix became zero (e.g., k=1 and residual was NaN), return identity
                if np.allclose(Q, 0):
                    print("Warning: Covariance matrix calculation resulted in near-zero matrix after NaN handling. Returning identity matrix.")
                    Q = np.eye(k)

        except Exception as e:
             print(f"Error calculating covariance matrix from residuals: {e}. Returning identity matrix.")
             Q = np.eye(k)

    # --- Final Check and Reshaping ---
    # Ensure Q is always a 2D NumPy array of shape (k, k)
    if not isinstance(Q, np.ndarray):
        print(f"Warning: Q calculated is not a numpy array ({type(Q)}). Attempting conversion or returning identity.")
        try:
            Q = np.array(Q, dtype=float) # Ensure float type
        except Exception as e:
            print(f"   Conversion to numpy array failed: {e}. Returning identity matrix.")
            Q = np.eye(k)

    # Check shape after potential conversion
    expected_shape = (k, k)
    if Q.shape != expected_shape:
        print(f"Warning: Q shape is {Q.shape} instead of expected {expected_shape}. Attempting correction.")
        # Specific case: scalar result for k=1
        if k == 1 and Q.ndim == 0:
            print(f"   Detected scalar Q for k=1. Reshaping to {expected_shape}.")
            Q = np.array([[Q.item()]], dtype=float) # Reshape scalar to [[scalar]]
        # Check if the number of elements matches and reshape
        elif Q.size == k*k:
             try:
                 Q = Q.reshape(expected_shape)
                 print(f"   Reshaped Q to {Q.shape}.")
             except ValueError as e:
                 print(f"   Reshape failed: {e}. Returning identity matrix.")
                 Q = np.eye(k)
        # Shape and size mismatch
        else:
            print(f"   Shape ({Q.shape}) and size ({Q.size}) mismatch. Cannot correct. Returning identity matrix.")
            Q = np.eye(k)

    # Final check for NaNs/Infs, replace with identity if issues persist
    if not np.all(np.isfinite(Q)):
        print(f"Warning: Final Q matrix contains NaNs or Infs after all checks. Returning identity matrix.")
        Q = np.eye(k)

    # Ensure the matrix is symmetric (numerical issues might make it slightly non-symmetric)
    Q = (Q + Q.T) / 2

    return Q

def calculate_pca(observables, n_factors):
    """Performs PCA on the input data.

    Args:
        observables (pd.DataFrame or np.ndarray): Input data, assumed to be
            already standardized and appropriately NaN-handled.
        n_factors (int): Number of principal components to extract.

    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray] or Tuple[None, None, None]:
            D: Diagonal matrix of top eigenvalues.
            V: Matrix of corresponding eigenvectors (loadings).
            S: Covariance/Correlation matrix used for decomposition (X.T @ X).
            Returns (None, None, None) if PCA fails.
    """
    # --- MODIFICATION: Assume input is pre-processed --- 
    if isinstance(observables, pd.DataFrame):
        # Check for NaNs before proceeding
        if observables.isnull().values.any():
             print("Warning: NaNs detected in input data for calculate_pca. Results might be unstable.")
             # Optionally, add more robust handling or return None
             # For now, convert to array and proceed cautiously
        X = np.array(observables.values, dtype=float)
    elif isinstance(observables, np.ndarray):
        if np.isnan(observables).any():
             print("Warning: NaNs detected in input ndarray for calculate_pca. Results might be unstable.")
        X = observables.astype(float)
    else:
        print("Error: Input to calculate_pca must be DataFrame or ndarray.")
        return None, None, None
        
    if X.ndim != 2:
        print("Error: Input array for calculate_pca must be 2D.")
        return None, None, None

    n_time, n_obs = X.shape
    # --- END MODIFICATION ---

    if n_factors > n_obs:
        print(f"Warning: n_factors ({n_factors}) > n_observables ({n_obs}). Setting n_factors = n_obs.")
        n_factors = n_obs
    if n_factors <= 0:
         print(f"Error: n_factors ({n_factors}) must be positive. Cannot perform PCA.")
         return None, None, None

    # --- REMOVED: Internal standardization --- 
    # obs_mean = observables.mean(axis=0)
    # obs_std = observables.std(axis=0)
    # obs_std[obs_std == 0] = 1.0
    # z = np.array((observables - obs_mean) / obs_std, dtype=float)
    # if np.isnan(z).any():
    #     print("Warning: NaNs detected after standardizing data for PCA. Filling with 0.")
    #     z = np.nan_to_num(z, nan=0.0)
    # --- END REMOVED --- 

    # Calculate covariance matrix S = X'X / (n_time - 1) or correlation matrix
    # Using X'X directly as in the original code seems intended for PCA on Z'Z structure.
    # Let's compute the covariance matrix properly: np.cov(X.T)
    try:
        # Use np.cov for covariance matrix (variables are columns, so transpose X)
        # Handle potential NaNs by calculating covariance only on non-NaN rows/columns pairs if necessary
        if np.isnan(X).any():
            print("Warning: NaNs present in PCA input. Attempting covariance calculation.")
            # Using pandas again for robust covariance calculation on the array
            temp_df_pca = pd.DataFrame(X)
            S = temp_df_pca.cov().to_numpy() # Use pandas cov for NaN handling
            if np.isnan(S).any():
                 print("Warning: Covariance matrix for PCA contains NaNs after calculation. Trying to fill.")
                 S = np.nan_to_num(S, nan=0.0)
                 # Ensure diagonal is positive (e.g., 1e-9 or 1.0) if it became zero
                 diag_indices_pca = np.diag_indices_from(S)
                 current_diag_pca = S[diag_indices_pca]
                 new_diag_pca = np.where(np.isclose(current_diag_pca, 0), 1e-9, current_diag_pca)
                 S[diag_indices_pca] = new_diag_pca
                 if not np.all(np.isfinite(S)): # Check again after filling
                      print("Error: Covariance matrix for PCA still contains NaNs/Infs after filling. PCA failed.")
                      return None, None, None
        else:
            # If no NaNs, np.cov is fine
            S = np.cov(X.T) # Shape (n_obs, n_obs)
            
    except Exception as e_cov:
         print(f"Error calculating covariance matrix S in PCA: {e_cov}")
         return None, None, None

    # S = z.T @ z # OLD calculation (Z'Z structure) Shape (n_obs, n_obs)

    # Eigen decomposition of S
    try:
        eigenvalues, eigenvectors = np.linalg.eigh(S) # Use eigh for symmetric matrix
        # Sort eigenvalues in descending order
        sorted_indices = np.argsort(eigenvalues)[::-1]
        evalues_sorted = eigenvalues[sorted_indices]
        evectors_sorted = eigenvectors[:, sorted_indices]

        # Select top n_factors
        top_evalues = evalues_sorted[:n_factors]
        # Ensure eigenvalues are non-negative (covariance matrix should be positive semi-definite)
        top_evalues = np.maximum(top_evalues, 0)
        V = evectors_sorted[:, :n_factors] # Eigenvectors (loadings on principal components)
        D = np.diag(top_evalues) # Diagonal matrix of eigenvalues

    except np.linalg.LinAlgError:
        print("Error: Eigen decomposition failed for covariance matrix S in PCA.")
        return None, None, None

    # Return eigenvalues (D), eigenvectors (V), and the covariance matrix (S)
    # The factor calculation (X @ V) should happen outside this function.
    return D, V, S
    
def calculate_covariance(factors):
    """Calculates the covariance matrix of the factors, handling NaNs robustly.

    Args:
        factors (pd.DataFrame or np.ndarray): Input factors (time series in columns).

    Returns:
        np.ndarray: Covariance matrix (k x k). Returns identity matrix if calculation fails
                    or results contain NaNs/Infs.
    """
    # --- Convert input to DataFrame for robust NaN handling --- 
    try:
        if isinstance(factors, np.ndarray):
            # Attempt to create meaningful column names if possible
            n_factors_in = factors.shape[1]
            col_names = [f'Factor_{i}' for i in range(n_factors_in)]
            factors_df = pd.DataFrame(factors, columns=col_names)
        elif isinstance(factors, pd.DataFrame):
            factors_df = factors.copy()
        else:
             print("Error: Input to calculate_covariance must be a pandas DataFrame or NumPy array. Returning Identity(1).")
             return np.identity(1) # Fallback if input type is wrong
        
        # Ensure data is numeric, coerce errors
        factors_df = factors_df.apply(pd.to_numeric, errors='coerce')

    except Exception as e:
        print(f"Error converting input to DataFrame or numeric: {e}. Returning Identity(1).")
        return np.identity(1)
    
    if factors_df.empty:
         print("Warning: Input factors DataFrame is empty after conversion. Returning Identity(1).")
         return np.identity(1)

    n_time, n_factors = factors_df.shape

    if n_factors == 0:
        print("Warning: Input has 0 factors (columns). Returning empty 0x0 matrix.")
        return np.empty((0,0))

    # Define default return value (identity matrix)
    default_cov = np.identity(n_factors)

    if n_time < 2:
        print(f"Warning: Need at least 2 observations ({n_time}) to calculate covariance. Returning identity matrix ({n_factors}x{n_factors}).")
        return default_cov

    try:
        # Check for columns that are entirely NaN before calculating covariance
        all_nan_cols = factors_df.isnull().all()
        if all_nan_cols.any():
            print(f"Warning: calculate_covariance found factors that are entirely NaN: {all_nan_cols[all_nan_cols].index.tolist()}. Covariance calculation might produce NaNs or fail.")

        # Use pandas DataFrame.cov() for robust NaN handling (pairwise deletion by default)
        # ddof=1 is the default for sample covariance in pandas
        cov_matrix_pd = factors_df.cov(ddof=1)
        cov_matrix = cov_matrix_pd.to_numpy()

        # Check if the resulting matrix has the expected shape (pandas should handle k=1 correctly)
        if cov_matrix.shape != (n_factors, n_factors):
            # This case is less likely with pandas .cov() but check defensively
            print(f"Warning: Calculated covariance matrix shape {cov_matrix.shape} unexpected after pandas cov(). Expected ({n_factors},{n_factors}). Returning identity matrix.")
            return default_cov

        # Check for NaNs or Infs in the final covariance matrix
        # This can happen if a column was all NaN, or insufficient overlapping data for pairwise calculation
        if not np.all(np.isfinite(cov_matrix)):
            print("Warning: Final covariance matrix contains NaNs or Infs after pandas cov(). Attempting to fill NaNs.")
            # Try filling NaNs with 0 and ensuring diagonal is at least 1 for non-all-NaN columns
            nan_mask = np.isnan(cov_matrix)
            cov_matrix[nan_mask] = 0.0
            diag_indices = np.diag_indices_from(cov_matrix)
            # Find original columns that were *not* all NaN
            valid_cols_mask = ~all_nan_cols.values
            # For valid columns, ensure diagonal is at least 1.0 if it became 0
            current_diag = cov_matrix[diag_indices]
            new_diag = np.where(valid_cols_mask & np.isclose(current_diag, 0), 1.0, current_diag)
            cov_matrix[diag_indices] = new_diag
            
            # Re-check finiteness after filling
            if not np.all(np.isfinite(cov_matrix)):
                 print("Error: Covariance matrix still contains non-finite values after NaN filling. Returning identity matrix.")
                 return default_cov
            else:
                 print("   NaNs filled. Proceeding with the filled matrix.")

        # Ensure symmetry (numerical precision might cause slight asymmetry)
        cov_matrix = (cov_matrix + cov_matrix.T) / 2
        return cov_matrix

    except Exception as e:
        print(f"Error during covariance calculation using pandas: {e}. Returning identity matrix ({n_factors}x{n_factors}).")
        import traceback
        traceback.print_exc() # Print stack trace for unexpected errors
        return default_cov

def calculate_measurement_error_matrix(observables: pd.DataFrame, factors: pd.DataFrame, Lambda: np.ndarray) -> np.ndarray:
    """
    Estimates the measurement error covariance matrix R.
    Assumes errors are uncorrelated, so R is diagonal.
    Calculates variance of residuals: observable - Lambda @ factor.

    Args:
        observables (pd.DataFrame): DataFrame of observable series (time series in columns).
        factors (pd.DataFrame): DataFrame of estimated factors (time series in columns).
        Lambda (np.ndarray): Factor loading matrix (n_observables x k_factors).

    Returns:
        np.ndarray: Estimated measurement error covariance matrix R (n_observables x n_observables),
                    which is diagonal. Returns an identity matrix if estimation fails.
    """
    if not isinstance(observables, pd.DataFrame) or not isinstance(factors, pd.DataFrame):
        print("Warning: Observables and factors should be pandas DataFrames.")
        # Attempt conversion or return identity
        try:
            observables = pd.DataFrame(observables)
            factors = pd.DataFrame(factors)
        except Exception as e:
            print(f"Error converting inputs to DataFrames: {e}. Returning identity matrix.")
            # Determine n_observables if possible, otherwise fallback
            n_obs = Lambda.shape[0] if isinstance(Lambda, np.ndarray) and Lambda.ndim == 2 else 1
            return np.eye(n_obs)

    if not isinstance(Lambda, np.ndarray):
         print("Warning: Lambda should be a NumPy array. Returning identity matrix.")
         return np.eye(observables.shape[1] if not observables.empty else 1)

    n_time_obs, n_obs = observables.shape
    n_time_fac, k_fac = factors.shape
    n_lambda_obs, n_lambda_fac = Lambda.shape

    # Basic dimension checks
    if n_obs == 0:
        print("Warning: No observables provided. Returning 0x0 matrix.")
        return np.empty((0,0))
    if n_obs != n_lambda_obs:
        print(f"Error: Dimension mismatch between observables ({n_obs}) and Lambda ({n_lambda_obs}). Returning identity matrix.")
        return np.eye(n_obs)
    if k_fac != n_lambda_fac:
         print(f"Error: Dimension mismatch between factors ({k_fac}) and Lambda ({n_lambda_fac}). Returning identity matrix.")
         return np.eye(n_obs) # Return based on observables dimension
    if n_time_obs != n_time_fac:
        print(f"Warning: Different number of time points for observables ({n_time_obs}) and factors ({n_time_fac}). Aligning based on index.")
        # Align data using pandas index intersection
        common_index = observables.index.intersection(factors.index)
        if common_index.empty:
            print("Error: No common time index between observables and factors. Cannot calculate residuals. Returning identity matrix.")
            return np.eye(n_obs)
        observables = observables.loc[common_index]
        factors = factors.loc[common_index]
        n_time = len(common_index)
        if n_time == 0: # Check again after alignment
             print("Error: Alignment resulted in zero time points. Returning identity matrix.")
             return np.eye(n_obs)
    else:
        n_time = n_time_obs


    # Ensure numeric types, coerce errors
    try:
        obs_numeric = observables.apply(pd.to_numeric, errors='coerce').values
        factors_numeric = factors.apply(pd.to_numeric, errors='coerce').values
    except Exception as e:
        print(f"Warning: Could not convert observables/factors to numeric: {e}. Proceeding, NaNs may occur.")
        obs_numeric = observables.values
        factors_numeric = factors.values


    # Calculate predicted observables: Lambda @ factors.T
    # factors_numeric is (n_time, k_fac), Lambda is (n_obs, k_fac)
    # We need Lambda @ factors_numeric[t, :].T for each time t
    # Or calculate predicted = factors_numeric @ Lambda.T which gives (n_time, n_obs)
    predicted_observables = factors_numeric @ Lambda.T

    # Calculate residuals: actual - predicted
    residuals = obs_numeric - predicted_observables # Shape (n_time, n_obs)

    # Calculate variance of each residual series (column)
    # Use np.nanvar to ignore NaNs if present
    try:
        # ddof=0 for population variance, or ddof=1 for sample variance.
        # Using sample variance (ddof=1) is common.
        variances = np.nanvar(residuals, axis=0, ddof=1)
    except Exception as e:
        print(f"Error calculating residual variances: {e}. Returning identity matrix.")
        return np.eye(n_obs)

    # Handle cases where variance might be NaN (e.g., all NaNs in a residual column)
    # Replace NaN variances with a default value, e.g., 1.0 or a small positive number
    if np.isnan(variances).any():
        print("Warning: NaNs found in calculated residual variances. Replacing NaNs with 1.0.")
        variances = np.nan_to_num(variances, nan=1.0)

    # Handle cases where variance might be zero or negative (numerical issues)
    # Ensure variances are positive
    variances = np.maximum(variances, 1e-9) # Set a small floor for variance

    # Create diagonal matrix R
    R = np.diag(variances)

    return R 