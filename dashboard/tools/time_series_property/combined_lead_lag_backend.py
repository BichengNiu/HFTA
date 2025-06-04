import pandas as pd
import numpy as np
from tools.time_series_property.time_lag_corr_backend import calculate_time_lagged_correlation # For correlation part

# 从 kl_divergence_calculator.py 集成的函数
def series_to_distribution(series_a: pd.Series, series_b: pd.Series, bins: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Converts two time series into discrete probability distributions using a common binning strategy.

    Args:
        series_a: First pandas Series.
        series_b: Second pandas Series.
        bins: Number of bins for the histogram.

    Returns:
        A tuple containing:
            - p: Probability distribution for series_a.
            - q: Probability distribution for series_b.
            - bin_edges: The edges of the bins.
    """
    # Drop NaN values to avoid issues with histogram binning
    series_a_clean = series_a.dropna()
    series_b_clean = series_b.dropna()

    if series_a_clean.empty or series_b_clean.empty:
        raise ValueError("One or both series are empty after NaN removal, cannot create distributions.")

    # Explicitly check for no variance if multiple bins are requested.
    # This should be done *before* trying to determine common_min/common_max for binning,
    # as a constant series might lead to combined_min == combined_max, bypassing multi-bin logic.
    if bins > 1:
        if len(series_a_clean.unique()) == 1:
            raise ValueError(f"Series A has no variance (all values are '{series_a_clean.iloc[0]}' after NaN removal) but bins={bins} requested. Cannot form meaningful multi-bin distribution.")
        if len(series_b_clean.unique()) == 1:
            raise ValueError(f"Series B has no variance (all values are '{series_b_clean.iloc[0]}' after NaN removal) but bins={bins} requested. Cannot form meaningful multi-bin distribution.")

    # Determine common range for binning
    combined_min = min(series_a_clean.min(), series_b_clean.min())
    combined_max = max(series_a_clean.max(), series_b_clean.max())

    if combined_min == combined_max: # Handle case where all data points are the same
        # Create a single bin; both distributions will be 1.0 in this bin.
        # Add a small epsilon to max to ensure the single value falls within a bin.
        bin_edges = np.array([combined_min, combined_max + 1e-9 if combined_min == combined_max else combined_max])
        bins_actual = 1 
    else:
        bin_edges = np.linspace(combined_min, combined_max, bins + 1)
        bins_actual = bins

    # Calculate histograms (counts)
    counts_a, _ = np.histogram(series_a_clean, bins=bin_edges, density=False)
    counts_b, _ = np.histogram(series_b_clean, bins=bin_edges, density=False)

    # Convert counts to probabilities
    p = counts_a / counts_a.sum() if counts_a.sum() > 0 else np.zeros_like(counts_a, dtype=float)
    q = counts_b / counts_b.sum() if counts_b.sum() > 0 else np.zeros_like(counts_b, dtype=float)
    
    # Handle cases where sum is zero (e.g., series was all NaNs and then cleaned, or somehow only one value was passed and min=max)
    # If p or q sums to 0, and the other doesn't, it means one series had no data in the common range.
    # This will lead to KL divergence issues if not handled.
    # The smoothing step later should help, but ensuring they are valid distributions is important.
    if p.sum() == 0 and series_a_clean.shape[0] > 0: # Had data, but not in these bins
        p = np.ones_like(counts_a, dtype=float) / bins_actual # Uniform distribution as a fallback
    if q.sum() == 0 and series_b_clean.shape[0] > 0: # Had data, but not in these bins
        q = np.ones_like(counts_b, dtype=float) / bins_actual # Uniform distribution as a fallback
        
    return p, q, bin_edges


def kl_divergence(p: np.ndarray, q: np.ndarray, smoothing_alpha: float = 1e-9) -> float:
    """
    Calculates the Kullback-Leibler (KL) divergence between two discrete probability distributions.
    D_KL(P || Q) = sum(P(i) * log(P(i) / Q(i)))

    Args:
        p: Numpy array representing the first probability distribution.
        q: Numpy array representing the second probability distribution.
        smoothing_alpha: Small constant for Laplace smoothing to avoid log(0) or division by zero.
                        In Laplace smoothing, each count is typically incremented by alpha, and then renormalized.
                        However, simpler smoothing is often applied by just adding alpha to q.

    Returns:
        The KL divergence value.
    """
    # Basic validation
    if p.shape != q.shape:
        raise ValueError(f"Distributions p and q must have the same shape, but got p.shape={p.shape} and q.shape={q.shape}")
    
    # Zero treatment
    # Check if distribution sums are not close to 1 (allow for floating point error)
    if not np.isclose(p.sum(), 1.0, atol=1e-9):
        p = p / p.sum() if p.sum() != 0 else np.ones_like(p) / len(p)  # Normalize or use uniform if sum is 0
    if not np.isclose(q.sum(), 1.0, atol=1e-9):
        q = q / q.sum() if q.sum() != 0 else np.ones_like(q) / len(q)  # Normalize or use uniform if sum is 0
        
    # Apply smoothing to avoid division by zero and log(0)
    # Add a small value to q where q is zero or very small, to avoid division by zero
    # Add the same value to p to maintain proportionality
    # This is a common approach to ensure smoothing doesn't introduce excessive bias
    q_smooth = q + smoothing_alpha
    
    # Re-normalize to ensure q_smooth is still a probability distribution after smoothing (sums to 1)
    q_smooth = q_smooth / q_smooth.sum()
    
    # Compute KL divergence only where p > 0 (since 0 * log(anything) = 0)
    # This avoids the undefined case of 0 * log(0) in the sum, but log(0) for positive p is still a problem
    valid_indices = p > 0
    
    if not np.any(valid_indices):  # If no valid indices (all p is 0 or NaN), return 0 divergence
        return 0.0
    
    p_valid = p[valid_indices]
    q_smooth_valid = q_smooth[valid_indices]
    
    # Compute KL divergence as sum(p * log(p / q))
    # This reduces to sum(p * (log(p) - log(q))) to improve numerical stability
    
    log_ratio = np.log(p_valid) - np.log(q_smooth_valid)
    kl_value = np.sum(p_valid * log_ratio)
    
    # Handle potential numerical issues
    if np.isnan(kl_value) or np.isinf(kl_value):
        # If we still got NaN or infinite, check the cause
        if np.any(np.isnan(log_ratio)) or np.any(np.isinf(log_ratio)):
            problematic_indices = np.where(np.isnan(log_ratio) | np.isinf(log_ratio))[0]
            reason = [f"At index {i}: p={p_valid[i]}, q_smooth={q_smooth_valid[i]}, log_ratio={log_ratio[i]}" for i in problematic_indices[:5]]
            print(f"Warning: NaN/Inf in log ratios at {len(problematic_indices)} indices. Examples: {reason}")
            
            # Hard clipping to very negative log values for q near zero
            # This is a coarse approximation but prevents divergence to infinity
            log_ratio = np.clip(log_ratio, -30, 30)  # Reasonable bounds for log values
            return np.sum(p_valid * log_ratio)
    
    return max(0.0, kl_value)  # Ensure non-negative value (should be guaranteed mathematically anyway)

def get_overlapping_series(series_a: pd.Series, series_b: pd.Series, lag: int) -> tuple[pd.Series | None, pd.Series | None]:
    """
    Extracts overlapping, non-NaN segments of two series given a specific lag for series_b relative to series_a.
    lag > 0 means series_b is shifted to the past (series_b[t-lag] vs series_a[t])
    lag < 0 means series_b is shifted to the future (series_b[t+|lag|] vs series_a[t])

    Args:
        series_a (pd.Series): The reference series.
        series_b (pd.Series): The series to be lagged.
        lag (int): The lag to apply to series_b.
                   Positive lag: series_b is shifted "backwards" (potential leader).
                   Negative lag: series_b is shifted "forwards" (potential follower).

    Returns:
        tuple[pd.Series | None, pd.Series | None]: A tuple of the aligned, cleaned (NaN-dropped based on pair)
                                                  series_a segment and series_b segment. Returns (None, None) if
                                                  alignment results in empty or too short series (min length 2).
                                                  Returned series will have their original names preserved and
                                                  a fresh 0-based index resulting from the dropna operation.
    """
    s_a_original_name = series_a.name
    s_b_original_name = series_b.name

    # Work with copies to avoid modifying original series
    s_a = series_a.copy()
    s_b = series_b.copy()

    if lag == 0:
        a_slice = s_a
        b_slice = s_b
    elif lag > 0: # series_b is shifted to the past (X(t-k) vs A(t))
        if len(s_a) <= lag: # Not enough data in s_a to shift
            return None, None
        a_slice = s_a.iloc[lag:]
        b_slice = s_b.iloc[:-lag] if len(s_b) > lag else pd.Series(dtype=s_b.dtype) # Avoid negative slice if s_b too short
    else: # lag < 0 (series_b is shifted to the future (X(t+|k|) vs A(t)))
        abs_lag = abs(lag)
        if len(s_b) <= abs_lag: # Not enough data in s_b to shift
            return None, None
        a_slice = s_a.iloc[:-abs_lag] if len(s_a) > abs_lag else pd.Series(dtype=s_a.dtype) # Avoid negative slice if s_a too short
        b_slice = s_b.iloc[abs_lag:]

    # Ensure slices are not empty after initial lag application
    if a_slice.empty or b_slice.empty:
        return None, None

    # Trim to same length based on the shorter of the two slices
    min_len = min(len(a_slice), len(b_slice))
    if min_len == 0: # Should be caught by .empty above, but as a safeguard
        return None, None

    a_slice = a_slice.iloc[:min_len]
    b_slice = b_slice.iloc[:min_len]

    # Assign a common, fresh index before combining for dropna
    # This is crucial for correct row-wise NaN removal
    a_slice.index = pd.RangeIndex(start=0, stop=min_len)
    b_slice.index = pd.RangeIndex(start=0, stop=min_len)

    # Combine, drop NaNs based on pairs, then separate
    # The DataFrame constructor will use the common index
    combined_df = pd.DataFrame({'a_col': a_slice, 'b_col': b_slice}).dropna()
    
    if len(combined_df) < 2: # Need at least 2 points for distribution/correlation
        return None, None
        
    out_a = combined_df['a_col'].copy() # Use .copy() to avoid SettingWithCopyWarning later if names are assigned
    out_b = combined_df['b_col'].copy()

    # Restore original names
    out_a.name = s_a_original_name
    out_b.name = s_b_original_name

    # Reset index for consistent 0-based integer index output
    out_a.reset_index(drop=True, inplace=True)
    out_b.reset_index(drop=True, inplace=True)
        
    return out_a, out_b


def perform_combined_lead_lag_analysis(
    df_input: pd.DataFrame,
    target_variable_name: str,
    candidate_variable_names_list: list,
    max_lags_config: int,
    kl_bins_config: int
) -> tuple[list, list, list]:
    """
    Performs combined lead-lag analysis using time-lagged correlation and K-L divergence.

    Args:
        df_input (pd.DataFrame): The input DataFrame.
        target_variable_name (str): Name of the target (reference) variable.
        candidate_variable_names_list (list): List of candidate variable names.
        max_lags_config (int): Maximum number of lags (positive and negative) to check.
        kl_bins_config (int): Number of bins for K-L divergence histogram.

    Returns:
        tuple: (all_results, error_messages, warning_messages)
               all_results is a list of dictionaries, each containing analysis for one candidate.
    """
    all_results = []
    error_messages = []
    warning_messages = []

    if not candidate_variable_names_list:
        error_messages.append("Candidate variable list is empty. No analysis to perform.")
        return all_results, error_messages, warning_messages

    if target_variable_name not in df_input.columns:
        error_messages.append(f"Target variable '{target_variable_name}' not found in DataFrame.")
        return all_results, error_messages, warning_messages

    series_a_raw = df_input[target_variable_name]
    if not pd.api.types.is_numeric_dtype(series_a_raw):
        error_messages.append(f"Target variable '{target_variable_name}' is not numeric.")
        return all_results, error_messages, warning_messages
    series_a_clean = series_a_raw.dropna()
    if len(series_a_clean) < max_lags_config + 2 : # Need enough data
        error_messages.append(f"Target variable '{target_variable_name}' has insufficient data points after NaN removal for the given max_lags.")
        return all_results, error_messages, warning_messages


    for candidate_name in candidate_variable_names_list:
        try:
            if candidate_name not in df_input.columns:
                warning_messages.append(f"Candidate variable '{candidate_name}' not found. Skipping.")
                all_results.append({
                    'target_variable': target_variable_name,
                    'candidate_variable': candidate_name,
                    'k_corr': np.nan, 'corr_at_k_corr': np.nan, 'full_correlogram_df': pd.DataFrame(),
                    'k_kl': np.nan, 'kl_at_k_kl': np.nan, 'full_kl_divergence_df': pd.DataFrame(),
                    'notes': "Candidate variable not found in data."
                })
                continue

            series_x_raw = df_input[candidate_name]
            if not pd.api.types.is_numeric_dtype(series_x_raw):
                warning_messages.append(f"Candidate variable '{candidate_name}' is not numeric. Skipping.")
                all_results.append({
                    'target_variable': target_variable_name,
                    'candidate_variable': candidate_name,
                    'k_corr': np.nan, 'corr_at_k_corr': np.nan, 'full_correlogram_df': pd.DataFrame(),
                    'k_kl': np.nan, 'kl_at_k_kl': np.nan, 'full_kl_divergence_df': pd.DataFrame(),
                    'notes': "Candidate variable is not numeric."
                })
                continue
            
            series_x_clean = series_x_raw.dropna()
            if len(series_x_clean) < max_lags_config + 2:
                warning_messages.append(f"Candidate variable '{candidate_name}' has insufficient data points after NaN removal for the given max_lags. Skipping.")
                all_results.append({
                    'target_variable': target_variable_name,
                    'candidate_variable': candidate_name,
                    'k_corr': np.nan, 'corr_at_k_corr': np.nan, 'full_correlogram_df': pd.DataFrame(),
                    'k_kl': np.nan, 'kl_at_k_kl': np.nan, 'full_kl_divergence_df': pd.DataFrame(),
                    'notes': "Insufficient data for candidate variable."
                })
                continue

            # 1. Time-Lagged Correlation (Series A vs Series X)
            # calculate_time_lagged_correlation expects series1, series2, max_lags
            # Lag > 0 means series_x leads series_a
            # Lag < 0 means series_x lags series_a
            try:
                # Note: calculate_time_lagged_correlation internally handles NaNs and alignment for correlation
                # For this function, series1 is the one "being predicted" (target A), 
                # series2 is the "predictor" (candidate X).
                # So, a positive lag in its output means X(t-lag) is correlated with A(t), i.e., X leads A.
                correlogram_df = calculate_time_lagged_correlation(series_a_raw, series_x_raw, max_lags_config) # Pass raw series
                k_corr_val, corr_at_k_corr_val = np.nan, np.nan
                if not correlogram_df.empty and 'Correlation' in correlogram_df.columns and correlogram_df['Correlation'].notna().any():
                    # We want the lag of X relative to A.
                    # If correlogram_df['Lag'] is k, it means corr(A(t), X(t-k)).
                    # So positive k in correlogram_df means X is leading A.
                    optimal_corr_idx = correlogram_df['Correlation'].abs().idxmax()
                    k_corr_val = correlogram_df.loc[optimal_corr_idx, 'Lag']
                    corr_at_k_corr_val = correlogram_df.loc[optimal_corr_idx, 'Correlation']
            except Exception as e_corr:
                warning_messages.append(f"Error calculating correlation for {candidate_name}: {e_corr}")
                correlogram_df = pd.DataFrame({'Lag': range(-max_lags_config, max_lags_config + 1), 'Correlation': np.nan})
                k_corr_val, corr_at_k_corr_val = np.nan, np.nan


            # 2. Time-Lagged K-L Divergence D( P_A_slice || P_X_slice_lagged )
            kl_lags = []
            kl_values = []
            notes_kl = ""

            for k_lag in range(-max_lags_config, max_lags_config + 1):
                # get_overlapping_series: k_lag > 0 means series_x is shifted to past (X leads A)
                # So we are comparing A(t) with X(t-k_lag)
                a_slice, x_slice = get_overlapping_series(series_a_raw, series_x_raw, k_lag)

                # Ensure enough data points for the number of bins specified for K-L
                # For a meaningful distribution, typically need more points than bins.
                # Let's set a heuristic: at least 2 points per bin, or minimum kl_bins_config points.
                min_points_for_kl = max(kl_bins_config + 1, 3) # Require kl_bins_config + 1 points, or at least 3

                if a_slice is not None and x_slice is not None and len(a_slice) >= min_points_for_kl and len(x_slice) >= min_points_for_kl:
                    try:
                        p_dist, q_dist, _ = series_to_distribution(a_slice, x_slice, bins=kl_bins_config)
                        kl_val = kl_divergence(p_dist, q_dist)
                        kl_lags.append(k_lag)
                        kl_values.append(kl_val)
                    except ValueError as ve: # From series_to_distribution or kl_divergence
                        # This can happen if after slicing, one series becomes all same value, etc.
                        kl_lags.append(k_lag)
                        kl_values.append(np.inf) # Or np.nan, np.inf is more indicative of divergence
                        # notes_kl += f" KL calc warning at lag {k_lag}: {ve};"
                    except Exception as e_kl_calc:
                        kl_lags.append(k_lag)
                        kl_values.append(np.inf) # Or np.nan
                        notes_kl += f" KL calc error at lag {k_lag}: {e_kl_calc};"
                else:
                    kl_lags.append(k_lag)
                    kl_values.append(np.nan) # Not enough data for this lag

            kl_divergence_df = pd.DataFrame({'Lag': kl_lags, 'KL_Divergence': kl_values})
            k_kl_val, kl_at_k_kl_val = np.nan, np.nan
            
            # Revised K-L optimal finding logic
            if not kl_divergence_df.empty and 'KL_Divergence' in kl_divergence_df:
                kl_series_for_min = kl_divergence_df['KL_Divergence']
                
                # Check if there are any valid numbers (finite or infinite) to find a minimum from
                if kl_series_for_min.notna().any():
                    try:
                        # idxmin skips NaNs. If only Infs are present, it will pick the first Inf.
                        # If a mix of finite and Inf, it picks the minimum finite.
                        optimal_idx = kl_series_for_min.idxmin()
                        k_kl_val = -kl_divergence_df.loc[optimal_idx, 'Lag']
                        kl_at_k_kl_val = kl_series_for_min.loc[optimal_idx]
                    except ValueError: 
                        # This happens if kl_series_for_min contains only NaNs after some filtering, 
                        # or if idxmin itself has an issue with all-Inf and specific pandas version.
                        # Fallback: check if all non-NaNs are np.inf
                        non_nan_kl = kl_series_for_min.dropna()
                        if not non_nan_kl.empty and np.isinf(non_nan_kl).all():
                            # If all valid values are Inf, pick the first one's lag and value
                            first_inf_idx = non_nan_kl.index[0]
                            k_kl_val = -kl_divergence_df.loc[first_inf_idx, 'Lag']
                            kl_at_k_kl_val = non_nan_kl.loc[first_inf_idx]
                        # Else, k_kl_val and kl_at_k_kl_val remain np.nan
                # If kl_series_for_min contains only NaNs, k_kl_val and kl_at_k_kl_val remain np.nan

            all_results.append({
                'target_variable': target_variable_name,
                'candidate_variable': candidate_name,
                'k_corr': k_corr_val,
                'corr_at_k_corr': corr_at_k_corr_val,
                'full_correlogram_df': correlogram_df,
                'k_kl': k_kl_val,
                'kl_at_k_kl': kl_at_k_kl_val,
                'full_kl_divergence_df': kl_divergence_df,
                'notes': notes_kl.strip()
            })

        except Exception as e_loop:
            error_messages.append(f"Unexpected error processing candidate '{candidate_name}': {e_loop}")
            # Optionally, append a placeholder to all_results here too if the test expects one even for unexpected errors
            all_results.append({
                'target_variable': target_variable_name,
                'candidate_variable': candidate_name,
                'k_corr': np.nan, 'corr_at_k_corr': np.nan, 'full_correlogram_df': pd.DataFrame(),
                'k_kl': np.nan, 'kl_at_k_kl': np.nan, 'full_kl_divergence_df': pd.DataFrame(),
                'notes': f"Failed with unexpected error: {e_loop}"
            })
            # No continue here, let it go to the next candidate if possible, or error out if severe

    return all_results, error_messages, warning_messages

if __name__ == '__main__':
    # Example Usage (Illustrative)
    idx = pd.date_range('2023-01-01', periods=100, freq='D')
    data_dict = {
        'A': np.random.randn(100).cumsum(),
        'B': np.roll(np.random.randn(100).cumsum(), 5) + np.random.randn(100)*0.1, # B leads A by 5
        'C': np.roll(np.random.randn(100).cumsum(), -3) + np.random.randn(100)*0.1, # C lags A by 3
        'D': np.random.randn(100).cumsum(), # No clear relation
        'E_short': list(np.random.randn(10).cumsum()) + [np.nan]*90,
        'F_non_numeric': ['text'] * 100
    }
    test_df = pd.DataFrame(data_dict, index=idx)
    test_df['A'].iloc[10:15] = np.nan # Add some NaNs to target
    test_df['B'].iloc[20:25] = np.nan 

    target = 'A'
    candidates = ['B', 'C', 'D', 'E_short', 'F_non_numeric', 'G_not_exist']
    max_lags = 10
    bins = 10

    results, errors, warnings = perform_combined_lead_lag_analysis(
        test_df, target, candidates, max_lags, bins
    )

    print("--- ERRORS ---")
    for err in errors: print(err)
    print("\n--- WARNINGS ---")
    for warn in warnings: print(warn)
    print("\n--- RESULTS ---")
    for res in results:
        print(f"Candidate: {res['candidate_variable']}")
        print(f"  Optimal Lag (Corr): {res['k_corr']}, Corr: {res['corr_at_k_corr']:.3f} (if not nan)")
        print(f"  Optimal Lag (KL): {res['k_kl']}, KL: {res['kl_at_k_kl']:.3f} (if not nan)")
        print(f"  Notes: {res['notes']}")
        # print(f"  Corr DF: \n{res['full_correlogram_df'].head(3)}")
        # print(f"  KL DF: \n{res['full_kl_divergence_df'].head(3)}")
        print("-" * 20)

    # Test get_overlapping_series
    print("\n--- Testing get_overlapping_series ---")
    s1_test = pd.Series([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    s2_test = pd.Series([10,20,30,40,50,60,70,80,90,100])
    
    print("Lag = 0")
    a0, b0 = get_overlapping_series(s1_test, s2_test, 0)
    print(f"A0: {a0.values if a0 is not None else None}")
    print(f"B0: {b0.values if b0 is not None else None}")

    print("Lag = 2 (s2 leads s1 by 2)") # s1[t] vs s2[t-2] => a_slice = s1[2:], b_slice = s2[:-2]
    a_p2, b_p2 = get_overlapping_series(s1_test, s2_test, 2)
    print(f"A_p2: {a_p2.values if a_p2 is not None else None}") # Should be [3,4,5,6,7,8,9,10]
    print(f"B_p2: {b_p2.values if b_p2 is not None else None}") # Should be [10,20,30,40,50,60,70,80]
    
    print("Lag = -2 (s2 lags s1 by 2)") # s1[t] vs s2[t+2] => a_slice = s1[:-2], b_slice = s2[2:]
    a_n2, b_n2 = get_overlapping_series(s1_test, s2_test, -2)
    print(f"A_n2: {a_n2.values if a_n2 is not None else None}") # Should be [1,2,3,4,5,6,7,8]
    print(f"B_n2: {b_n2.values if b_n2 is not None else None}") # Should be [30,40,50,60,70,80,90,100]

    s1_nan = pd.Series([1, np.nan, 3, 4, np.nan, 6])
    s2_nan = pd.Series([np.nan, 20, 30, np.nan, 50, 60])
    print("Lag = 0 with NaNs")
    a_nan0, b_nan0 = get_overlapping_series(s1_nan, s2_nan, 0)
    print(f"A_nan0: {a_nan0.values if a_nan0 is not None else None}") # Expected: [3, 6]
    print(f"B_nan0: {b_nan0.values if b_nan0 is not None else None}") # Expected: [30, 60] 