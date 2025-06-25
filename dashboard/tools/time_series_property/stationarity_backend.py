import pandas as pd
import numpy as np
from statsmodels.tsa.stattools import adfuller, kpss
from statsmodels.tsa.statespace.tools import diff as statespace_diff
# Remove seasonal_decompose if confirmed unused
# from statsmodels.tsa.seasonal import seasonal_decompose 
# Remove io if only used in frontend
# import io 
import warnings

# --- ADF Test Helper ---
def _run_adf(series, alpha):
    """Runs ADF test and returns p-value and stationarity status."""
    # Drop NaNs that might result from transformations like differencing
    series_cleaned = series.dropna()
    if series_cleaned.empty or len(series_cleaned) < 5: # Need minimum samples for ADF
         print(f"[Backend ADF WARN] Series '{series.name}' is empty or too short after dropna. Cannot run ADF.")
         return None, '数据不足'
    try:
        # Suppress warnings during ADF test (e.g., about implicit frequency)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = adfuller(series_cleaned, regression='ct')
        p_value = result[1]
        is_stationary = '是' if p_value < alpha else '否'
        return p_value, is_stationary
    except Exception as e:
        print(f"[Backend ADF ERROR] Error running ADF for series '{series.name}': {e}")
        # Consider more specific error handling (e.g., LinAlgError) if needed
        if "Ensure that you have included enough lag variables" in str(e):
             return None, '计算失败(尝试减少滞后阶数)'
        elif "degrees of freedom" in str(e):
             return None, '计算失败(自由度不足)'
        else:
             return None, f'计算失败({type(e).__name__})'
             
# --- Removed _transformations dictionary as it's no longer needed ---

# --- Main Stationarity Testing and Processing Function ---
def test_and_process_stationarity(df_in, alpha=0.05, processing_method='keep', diff_order=1):
    """
    Tests stationarity for all numeric columns in a DataFrame using ADF test and processes 
    non-stationary series based on the chosen method.

    Args:
        df_in (pd.DataFrame): Input DataFrame, potentially with a datetime column/index.
        alpha (float): Significance level for ADF test.
        processing_method (str): Method to handle non-stationary series. 
                                 Options: 'keep' (default), 'diff', 'log_diff'.
        diff_order (int): Order of differencing to apply if method is 'diff' or 'log_diff'.

    Returns:
        tuple: (pd.DataFrame: summary_df, dict: processed_data_dict)
               summary_df contains test results for each original numeric column.
               processed_data_dict contains original numeric columns and any 
                                    newly generated stationary columns. 
                                    The datetime column is preserved as the index if possible.
    """
    print(f"[Backend] Entering test_and_process_stationarity. Input df shape: {df_in.shape}, alpha: {alpha}, method: {processing_method}, order: {diff_order}")
    df = df_in.copy()
    summary_results = []
    processed_data = {} # Dictionary to hold final data series

    # --- Identify and Preserve Datetime Column/Index ---
    original_datetime_col = None
    time_series_index = None

    if isinstance(df.index, pd.DatetimeIndex):
        original_datetime_col = df.index.name if df.index.name else "时间索引" # Use placeholder if index name is None
        time_series_index = df.index.copy()
        processed_data[original_datetime_col] = time_series_index # Add index to processed data directly
        print(f"[Backend] Identified datetime index: '{original_datetime_col}'")
    else:
        datetime_cols = df.select_dtypes(include=['datetime', 'datetime64']).columns
        if len(datetime_cols) == 1:
            original_datetime_col = datetime_cols[0]
            # Attempt to set it as index for processing, but keep original data column too
            try:
                 df = df.set_index(original_datetime_col, drop=False) # Keep the column
                 time_series_index = df.index.copy()
                 processed_data[original_datetime_col] = df[original_datetime_col] # Store the original column
                 print(f"[Backend] Found datetime column '{original_datetime_col}' and set as index.")
            except Exception as e_set_index:
                 print(f"[Backend WARN] Could not set column '{original_datetime_col}' as index: {e_set_index}. Proceeding without datetime index.")
                 # Keep original_datetime_col name, but time_series_index remains None
        elif len(datetime_cols) > 1:
            print(f"[Backend WARN] Multiple datetime columns found: {datetime_cols}. Cannot automatically determine time index.")
            # original_datetime_col remains None
        else:
            print("[Backend INFO] No datetime column/index found.")
            # original_datetime_col remains None
            
    if original_datetime_col:
         summary_results.append({
             '指标名称': original_datetime_col, 
             '原始P值': None, '原始是否平稳': '时间戳列', 
             '处理方法': '保留', '处理后P值': None, '最终是否平稳': '时间戳列'
         })
         print(f"[Backend] Preserving original datetime column/index '{original_datetime_col}' in processed data.")

    # --- Iterate through columns for testing and processing ---
    for name in df.columns:
        # Skip the datetime column/index if it was identified
        if name == original_datetime_col and time_series_index is not None: 
            print(f"[Backend] Skipping already handled datetime column/index: '{name}'")
            continue
            
        if pd.api.types.is_numeric_dtype(df[name]):
            print(f"\n[Backend] Processing column: '{name}'")
            series = df[name]
            series_cleaned = series.dropna() # Use cleaned version for initial test too

            # Store the original numeric series regardless of processing
            processed_data[name] = series.copy() 
            print(f"[Backend DEBUG] Stored original numeric series: {name}")

            # --- Initial ADF Test ---
            original_p_value, original_status = _run_adf(series_cleaned, alpha)
            print(f"[Backend] Initial ADF test for '{name}': p-value={original_p_value}, stationary={original_status}")

            processed_series = None
            processed_p_value_for_summary = None
            final_status_for_summary = original_status
            current_method_description = "原始序列" # Default description

            # --- Apply User-Specified Processing if Non-Stationary ---
            if original_status == '否' and processing_method != 'keep':
                print(f"[Backend] Column '{name}' is non-stationary. Applying method: '{processing_method}' with order {diff_order}")
                
                temp_series = series.copy() # Work on a copy
                transform_successful = False
                new_col_name = None

                try:
                    if processing_method == 'log_diff':
                         current_method_description = f"{diff_order}阶对数差分"
                         # Check if log is applicable
                         if (temp_series <= 0).any():
                             print(f"[Backend WARN] Cannot apply log transform to '{name}' as it contains non-positive values.")
                             current_method_description += " (无法应用对数)"
                         else:
                             temp_series = np.log(temp_series)
                             temp_series = statespace_diff(temp_series, k_diff=diff_order)
                             transform_successful = True
                             new_col_name = f"{name}_log_diff{diff_order}"
                             
                    elif processing_method == 'diff':
                         current_method_description = f"{diff_order}阶差分"
                         temp_series = statespace_diff(temp_series, k_diff=diff_order)
                         transform_successful = True
                         new_col_name = f"{name}_diff{diff_order}"

                    # Run ADF on the processed series if transformation was done
                    if transform_successful:
                         processed_p_value, processed_status = _run_adf(temp_series, alpha)
                         processed_p_value_for_summary = processed_p_value
                         final_status_for_summary = processed_status
                         print(f"[Backend]   Result for '{current_method_description}': p-value={processed_p_value}, stationary={processed_status}")
                         
                         if processed_status == '是':
                             processed_series = temp_series.copy() # Store the successful stationary series
                             # Add the NEW processed series to the output dict
                             processed_data[new_col_name] = processed_series 
                             print(f"[Backend] Storing NEW processed stationary series for '{name}' as '{new_col_name}' (Method: {current_method_description})")
                         # else: processed_series remains None if still not stationary
                    # else: (e.g., log failed), processed_series remains None
                
                except Exception as e_transform:
                     print(f"[Backend ERROR] Error applying {current_method_description} to '{name}': {e_transform}")
                     current_method_description += " (处理失败)"
                     # processed_series remains None
                     
            elif original_status == '否' and processing_method == 'keep':
                 current_method_description = "保留原始 (非平稳)"
                 # processed_series remains None, status remains '否'
                 
            # --- Append results to summary ---
            summary_results.append({
                '指标名称': name, 
                '原始P值': original_p_value,
                '原始是否平稳': original_status,
                '处理方法': current_method_description, 
                '处理后P值': processed_p_value_for_summary, # Will be None if no processing or failed
                '最终是否平稳': final_status_for_summary # Status after chosen processing (or original if stationary/keep)
            })
        
        else: # Non-numeric, non-datetime column
            print(f"[Backend] Skipping non-numeric/non-datetime column: '{name}'")
            # Optionally add to summary or processed_data if needed, e.g., as 'object' type
            # processed_data[name] = df[name].copy() # Example: store original non-numeric
            # summary_results.append({
            #     '指标名称': name, 
            #     '原始P值': None, '原始是否平稳': '非数值', 
            #     '处理方法': '跳过', '处理后P值': None, '最终是否平稳': '非数值'
            # })


    print(f"[Backend] Finished processing all columns. Summary rows: {len(summary_results)}, Processed data columns: {len(processed_data)}")
    summary_df = pd.DataFrame(summary_results)

    # --- Sort summary_df to put timestamp row first --- #
    if original_datetime_col and original_datetime_col in summary_df['指标名称'].values:
        print(f"[Backend DEBUG] Sorting summary_df to put '{original_datetime_col}' first.")
        # Create a boolean mask for the timestamp row
        is_time_col = summary_df['指标名称'] == original_datetime_col
        # Concatenate the timestamp row with the rest of the dataframe
        summary_df = pd.concat([summary_df[is_time_col], summary_df[~is_time_col]], ignore_index=True)
    # --- End sorting --- #

    # --- Create Final Processed DataFrame from Dictionary ---
    # Ensure consistent length by aligning index (important if differencing occurred)
    try:
        # Use the preserved time series index if available
        if time_series_index is not None:
            final_processed_df = pd.DataFrame(processed_data)
            # Try to set the datetime index FIRST
            if original_datetime_col in final_processed_df.columns:
                try:
                    final_processed_df = final_processed_df.set_index(original_datetime_col, drop=True) # Drop the column after setting index
                    print(f"[Backend DEBUG] Set index '{original_datetime_col}' on final DataFrame.")
                except KeyError: # Should not happen if logic above is correct
                     print(f"[Backend WARN] Datetime column '{original_datetime_col}' not found in final data dictionary keys.")
                     # Attempt reindex based on stored time_series_index
                     try:
                          final_processed_df = pd.DataFrame(processed_data, index=time_series_index)
                          print(f"[Backend DEBUG] Reindexed final DataFrame using stored datetime index.")
                     except Exception as e_reindex:
                          print(f"[Backend ERROR] Failed to reindex final DataFrame: {e_reindex}. Returning potentially unindexed/misaligned data.")
                          final_processed_df = pd.DataFrame(processed_data) # Fallback
                except Exception as e_set_index_final:
                    print(f"[Backend ERROR] Failed to set index '{original_datetime_col}' on final DataFrame: {e_set_index_final}. Returning potentially unindexed/misaligned data.")
                    final_processed_df = pd.DataFrame(processed_data) # Fallback
            else:
                 # Fallback if datetime col wasn't found in keys, use stored index directly
                 print(f"[Backend WARN] Datetime column '{original_datetime_col}' not found in final keys, attempting reindex.")
                 try:
                     final_processed_df = pd.DataFrame(processed_data, index=time_series_index)
                     print(f"[Backend DEBUG] Reindexed final DataFrame using stored datetime index (fallback).")
                 except Exception as e_reindex:
                      print(f"[Backend ERROR] Failed to reindex final DataFrame (fallback): {e_reindex}.")
                      final_processed_df = pd.DataFrame(processed_data) # Fallback
        else:
            # No time index identified, just create the DataFrame
             final_processed_df = pd.DataFrame(processed_data)
             print("[Backend DEBUG] Created final DataFrame without a datetime index.")

        # --- Final checks ---
        # Check for duplicate column names (shouldn't happen with new logic but good practice)
        if final_processed_df.columns.duplicated().any():
             print(f"[Backend WARN] Duplicate column names found in final processed DataFrame: {final_processed_df.columns[final_processed_df.columns.duplicated()].tolist()}")
             # Consider renaming duplicates if necessary
        
        print(f"[Backend DEBUG] Final processed_df columns before return: {final_processed_df.columns.tolist()}")
        print(f"[Backend] Created summary_df (shape: {summary_df.shape}) and final_processed_df (shape: {final_processed_df.shape}).")

        # Return the summary and the DataFrame directly
        return summary_df, final_processed_df 

    except Exception as e_final_df:
        print(f"[Backend ERROR] Failed to construct final processed DataFrame: {e_final_df}")
        # Return summary and an empty DataFrame or raise error
        return summary_df, pd.DataFrame()


# --- KPSS Test Helper (for residuals - keep as is for now) ---
def run_kpss_on_residuals(residuals, alpha=0.05):
    """
    Runs KPSS test on residuals to check for stationarity around a trend.
    H0: The series is stationary around a trend.
    HA: The series has a unit root (non-stationary).
    We want to FAIL to reject H0 (p-value > alpha).
    """
    if residuals is None or residuals.empty:
        return None, "无残差数据"
    try:
        # 'ct' tests for stationarity around a trend
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            kpss_stat, p_value, lags, crit = kpss(residuals.dropna(), regression='ct') 
        
        # Interpretation: If p > alpha, we fail to reject H0 (stationarity)
        is_stationary = '是' if p_value > alpha else '否' 
        print(f"[Backend KPSS] KPSS test on residuals: p-value={p_value}, stationary={is_stationary}")
        return p_value, is_stationary
    except Exception as e:
        print(f"[Backend KPSS ERROR] Error running KPSS on residuals: {e}")
        return None, f'计算失败({type(e).__name__})'

# No Streamlit code or test block in backend file 