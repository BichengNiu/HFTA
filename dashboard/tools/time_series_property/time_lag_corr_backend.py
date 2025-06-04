import pandas as pd
import numpy as np

def calculate_time_lagged_correlation(series1: pd.Series, series2: pd.Series, max_lags: int) -> pd.DataFrame:
    """
    计算两个时间序列之间的时差相关性。

    参数:
    series1 (pd.Series): 第一个时间序列。
    series2 (pd.Series): 第二个时间序列。
    max_lags (int): 最大滞后/超前阶数。

    返回:
    pd.DataFrame: 包含两列的DataFrame: 'Lag' 和 'Correlation'。
                  'Lag' 从 -max_lags 到 +max_lags。
                  'Correlation' 为对应滞后阶数的皮尔逊相关系数。
    """
    if not isinstance(series1, pd.Series):
        series1 = pd.Series(series1)
    if not isinstance(series2, pd.Series):
        series2 = pd.Series(series2)

    # Ensure series are float for correlation and handle potential all-NaN slices gracefully
    series1 = series1.astype(float)
    series2 = series2.astype(float)

    if series1.empty or series2.empty or series1.isnull().all() or series2.isnull().all():
        lags_range = range(-max_lags, max_lags + 1)
        correlations_val = [np.nan] * len(lags_range)
        return pd.DataFrame({'Lag': lags_range, 'Correlation': correlations_val})

    n = len(series1)
    m = len(series2)

    if n == 0 or m == 0: # Should be caught by series1.empty but as a safeguard
        lags_range = range(-max_lags, max_lags + 1)
        correlations_val = [np.nan] * len(lags_range)
        return pd.DataFrame({'Lag': lags_range, 'Correlation': correlations_val})

    lags = []
    correlations = []

    for lag in range(-max_lags, max_lags + 1):
        lags.append(lag)
        
        s1_slice = None
        s2_slice = None

        # Determine slices based on lag
        if lag == 0:
            # No shift, align based on common length from start
            common_len = min(n, m)
            s1_slice = series1.iloc[:common_len]
            s2_slice = series2.iloc[:common_len]
        elif lag > 0: # series2 is lagged (shifted forward) relative to series1
            # series1 starts at 0, series2 starts at lag
            # Effective length for s1_slice: n - lag
            # Effective length for s2_slice: m - lag (if m is length of series2.iloc[lag:])
            if n > lag: # series1 must be longer than the lag
                s1_slice = series1.iloc[:-lag] # series1 from start up to n-lag point
                s2_slice = series2.iloc[lag:] # series2 from lag up to its end
            else:
                correlations.append(np.nan)
                continue
        else: # lag < 0, series1 is lagged (shifted forward) relative to series2
            abs_lag = abs(lag)
            # series2 starts at 0, series1 starts at abs_lag
            if m > abs_lag: # series2 must be longer than the absolute lag
                s1_slice = series1.iloc[abs_lag:] # series1 from abs_lag to its end
                s2_slice = series2.iloc[:-abs_lag] # series2 from start up to m-abs_lag point
            else:
                correlations.append(np.nan)
                continue
        
        # Align slices to common length after shifting
        common_len_after_shift = min(len(s1_slice), len(s2_slice))
        if common_len_after_shift > 0:
            s1_slice = s1_slice.iloc[:common_len_after_shift]
            s2_slice = s2_slice.iloc[:common_len_after_shift]
        else:
            correlations.append(np.nan)
            continue

        if len(s1_slice) >= 2 and len(s2_slice) >= 2 and not s1_slice.isnull().all() and not s2_slice.isnull().all():
            # Reset index for .corr() to work correctly if original indices are misaligned
            s1_reset = s1_slice.reset_index(drop=True)
            s2_reset = s2_slice.reset_index(drop=True)
            
            # Check for variance after ensuring they are not all NaN
            if s1_reset.nunique(dropna=True) < 2 or s2_reset.nunique(dropna=True) < 2:
                 correlations.append(np.nan) # No variance, correlation is undefined or uninformative
            else:
                correlation = s1_reset.corr(s2_reset)
                correlations.append(correlation)
        else:
            correlations.append(np.nan)
            
    return pd.DataFrame({'Lag': lags, 'Correlation': correlations})

def perform_batch_time_lag_correlation(
    df_input: pd.DataFrame, 
    selected_lagged_variable: str, 
    selected_leading_variables_list: list, 
    selected_max_leading_periods: int
    ):
    """
    Performs batch time-lagged correlation analysis.

    Args:
        df_input (pd.DataFrame): Input dataframe.
        selected_lagged_variable (str): Name of the lagged variable column.
        selected_leading_variables_list (list): List of names for leading variable columns.
        selected_max_leading_periods (int): Maximum number of leading periods to check.

    Returns:
        tuple: (list_of_results, list_of_errors, list_of_warnings)
    """
    df = df_input.copy()
    batch_results = []
    error_messages = []
    warning_messages = []

    if not selected_lagged_variable:
        error_messages.append("滞后变量未选择。")
        return batch_results, error_messages, warning_messages
    
    if not selected_leading_variables_list:
        warning_messages.append("没有选择任何领先变量进行分析。") # Changed to warning as it might not be an error
        return batch_results, error_messages, warning_messages

    lagged_variable_data_raw = df.get(selected_lagged_variable)
    if lagged_variable_data_raw is None:
        error_messages.append(f"滞后变量 '{selected_lagged_variable}' 在数据集中未找到。")
        return batch_results, error_messages, warning_messages
        
    lagged_variable_numeric = pd.to_numeric(lagged_variable_data_raw, errors='coerce')

    if lagged_variable_numeric.dropna().empty or len(lagged_variable_numeric.dropna()) < 2:
        error_messages.append(f"滞后变量 '{selected_lagged_variable}' 数据不足或无效（少于2个非NaN值）。")
        # Return empty results but still allow UI to update with the error
        return batch_results, error_messages, warning_messages

    for leading_var_name in selected_leading_variables_list:
        leading_variable_data_raw = df.get(leading_var_name)
        if leading_variable_data_raw is None:
            warning_messages.append(f"领先变量 '{leading_var_name}' 在数据集中未找到，已跳过。")
            summary_entry = {
                '滞后变量': selected_lagged_variable,
                '领先变量': leading_var_name,
                '最优领先阶数': np.nan,
                '相关系数 (最优领先时)': np.nan,
                '备注': '领先变量数据未找到',
                'correlogram_df': pd.DataFrame()
            }
            batch_results.append(summary_entry)
            continue
            
        leading_variable_numeric = pd.to_numeric(leading_variable_data_raw, errors='coerce')

        # Prepare data for correlation: use common non-NaN period
        combined_df_for_pair = pd.DataFrame({
            'lagged': lagged_variable_numeric, 
            'leading': leading_variable_numeric
        }).dropna()
        
        s_lagged_processed = combined_df_for_pair['lagged']
        s_leading_processed = combined_df_for_pair['leading']
        num_common_points = len(s_lagged_processed)

        summary_entry = {
            '滞后变量': selected_lagged_variable,
            '领先变量': leading_var_name,
            '最优领先阶数': np.nan,
            '相关系数 (最优领先时)': np.nan,
            '备注': f"基于 {num_common_points} 个共同数据点计算",
            'correlogram_df': pd.DataFrame() # To store full results for plotting
        }

        if s_lagged_processed.empty or s_leading_processed.empty or num_common_points < 2:
            summary_entry['备注'] = f"数据预处理后序列为空或过短 (共同点: {num_common_points})"
        elif num_common_points < selected_max_leading_periods + 2 : # Need enough points for max lag + at least 2 for corr
            summary_entry['备注'] = f"共同数据点 ({num_common_points}) 不足以支持最大领先阶数 ({selected_max_leading_periods}) 的有效计算"
        else:
            full_correlogram_df = calculate_time_lagged_correlation(
                s_lagged_processed, 
                s_leading_processed, 
                selected_max_leading_periods
            )
            summary_entry['correlogram_df'] = full_correlogram_df

            if not full_correlogram_df.empty and 'Correlation' in full_correlogram_df.columns and full_correlogram_df['Correlation'].notna().any():
                # Filter for positive lags (Leading Variable leads Lagged Variable)
                # Lag > 0 means series2 (leading_variable) is shifted forward, thus leading series1 (lagged_variable)
                positive_lags_df = full_correlogram_df[full_correlogram_df['Lag'] > 0].copy()

                if not positive_lags_df.empty and positive_lags_df['Correlation'].notna().any():
                    positive_lags_df['AbsCorrelation'] = positive_lags_df['Correlation'].abs()
                    optimal_idx = positive_lags_df['AbsCorrelation'].idxmax()
                    
                    summary_entry['最优领先阶数'] = positive_lags_df.loc[optimal_idx, 'Lag']
                    summary_entry['相关系数 (最优领先时)'] = positive_lags_df.loc[optimal_idx, 'Correlation']
                else:
                    summary_entry['备注'] += "; 在考察的领先周期内未找到显著相关性"
            else:
                summary_entry['备注'] += "; 未能计算有效相关性或结果为空"
        
        batch_results.append(summary_entry)
            
    return batch_results, error_messages, warning_messages 