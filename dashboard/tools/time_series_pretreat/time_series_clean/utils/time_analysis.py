import pandas as pd
from datetime import date # For pd.Timestamp(date)
import numpy as np

# Assuming parse_time_column will be in a sibling module parsers.py
from .parsers import parse_time_column

# Extracted from time_series_clean_utils.py
def calculate_time_series_info(time_series_to_use):
    """
    Calculates frequency and start/end times for a non-empty datetime Series.
    Args:
        time_series_to_use (pd.Series): A pandas Series of datetime objects, NaNs already dropped.
    Returns:
        dict: Containing 'freq', 'inferred_freq_code', 'start', 'end'.
    """
    info = {'freq': '无法推断', 'inferred_freq_code': None, 'start': None, 'end': None}
    if time_series_to_use.empty:
        return info

    if not time_series_to_use.is_monotonic_increasing:
        print(f"--- DEBUG (calculate_time_series_info for {time_series_to_use.name if time_series_to_use.name else 'Index'}): Sorting time series before frequency inference. ---")
        time_series_to_use = time_series_to_use.sort_values()

    try:
        info['start'] = time_series_to_use.dt.strftime('%Y-%m-%d %H:%M:%S').min()
    except AttributeError:
        info['start'] = time_series_to_use.min()
    try:
        info['end'] = time_series_to_use.dt.strftime('%Y-%m-%d %H:%M:%S').max()
    except AttributeError:
        info['end'] = time_series_to_use.max()

    inferred_freq = pd.infer_freq(time_series_to_use)
    print(f"--- DEBUG (calculate_time_series_info for {time_series_to_use.name if time_series_to_use.name else 'Index'}): pd.infer_freq result: {inferred_freq} ---")
    if inferred_freq:
        freq_map = {'D': '日度', 'B': '工作日', 'W': '周度', 'M': '月度 (月末)', 'MS': '月度 (月初)', 'Q': '季度 (季末)', 'QS': '季度 (季初)', 'A': '年度 (年末)', 'AS': '年度 (年初)', 'ME': '月度 (月末)' }
        info['inferred_freq_code'] = inferred_freq
        display_freq_code = 'ME' if inferred_freq == 'M' else inferred_freq
        info['freq'] = freq_map.get(display_freq_code, display_freq_code)
    elif len(time_series_to_use) > 1:
        try:
            is_month_end = (time_series_to_use + pd.Timedelta(days=1)).dt.day == 1
            if is_month_end.all():
                info['freq'] = '月度 (月末推断)'
                info['inferred_freq_code'] = 'ME'
                print(f"--- DEBUG (calculate_time_series_info): Fallback - All dates are month ends. Setting inferred_freq_code to 'ME'. ---")
            else:
                median_diff = time_series_to_use.diff().median()
                print(f"--- DEBUG (calculate_time_series_info for {time_series_to_use.name if time_series_to_use.name else 'Index'}): median_diff for fallback freq: {median_diff} ---")
                if pd.Timedelta('6 days') <= median_diff <= pd.Timedelta('8 days'):
                    info['freq'] = '周度 (推断)'
                    info['inferred_freq_code'] = 'W-MON'
                elif pd.Timedelta('28 days') <= median_diff <= pd.Timedelta('32 days'):
                    if (time_series_to_use.dt.day == 1).mean() > 0.7:
                        info['freq'] = '月度 (月初推断)'
                        info['inferred_freq_code'] = 'MS'
                    else:
                        info['freq'] = '月度 (月末或月中推断)'
                        info['inferred_freq_code'] = 'ME'
                else:
                    info['freq'] = '不规则'
                    info['inferred_freq_code'] = None
        except Exception as e_fallback_freq:
            print(f"ERROR during fallback frequency inference: {e_fallback_freq}")
            info['freq'] = '不规则 (错误)'
            info['inferred_freq_code'] = None
    return info

def infer_series_frequency(series: pd.Series, series_name: str = "Series"):
    """
    Infers the frequency of a single time series.
    Args:
        series (pd.Series): Pandas Series with DatetimeIndex (or parsable to datetime).
                            Assumes NaNs in values are acceptable, but index should be datetime.
        series_name (str): Name of the series for logging/debugging.
    Returns:
        dict: {'freq_group': str, 'pandas_freq_code': str, 'details': str}
              freq_group can be "Daily", "Weekly", "Monthly", "Annual", "Irregular", "Undetermined"
    """
    if not isinstance(series.index, pd.DatetimeIndex):
        try:
            dt_index_values = pd.to_datetime(series.index, errors='coerce')
            if pd.isna(dt_index_values).all():
                return {'freq_group': "Undetermined", 'pandas_freq_code': None, 'details': "Index is not datetime and could not be converted (all NaT)."}
            series = series.set_axis(dt_index_values)
        except Exception as e:
            return {'freq_group': "Undetermined", 'pandas_freq_code': None, 'details': f"Index is not datetime and conversion failed: {e}"}

    series = series[series.index.notna()]
    if len(series) < 2:
        return {'freq_group': "Undetermined", 'pandas_freq_code': None, 'details': "Not enough data points to infer frequency."}

    if not series.index.is_monotonic_increasing:
        series = series.sort_index()
        print(f"[InferFreq - {series_name}] Series index was not monotonic, sorted.")

    # --- Begin Refactored Logic ---
    
    # 1. Analyze diffs first for strong weekly/monthly signals
    diffs_td = series.index.to_series().diff().dropna()
    
    if diffs_td.empty: # Only one unique data point or all NaT diffs
        # Try pd.infer_freq on the original index if it has at least two points.
        # This handles cases like [Timestamp('2022-01-01'), Timestamp('2022-01-01')] -> diff is NaT
        if len(series.index.unique()) == 1 and len(series.index) >=2:
             return {'freq_group': "Undetermined", 'pandas_freq_code': None, 'details': "Single unique timestamp with multiple entries."}
        # If truly not enough points for diff, let pd.infer_freq try, then give up.
        # pd.infer_freq might still give a result if the series itself has a very regular pattern of identical timestamps.
        # However, typically this means not enough info.
        _temp_inferred_code = pd.infer_freq(series.index)
        if _temp_inferred_code:
             print(f"[InferFreq - {series_name}] Diff empty, but pd.infer_freq gave '{_temp_inferred_code}'. Treating as Specific/Irregular.")
             # Fall through to pd.infer_freq logic below, but it will likely be Specific or Irregular.
        else:
            return {'freq_group': "Undetermined", 'pandas_freq_code': None, 'details': "Not enough distinct data points for diff analysis and pd.infer_freq failed."}


    diffs_days = diffs_td.dt.days
    
    # For very short series, diffs might be misleading.
    # Thresholds for pattern counts
    total_diffs = len(diffs_days)
    dominance_threshold_strict = 0.8 # For strong weekly/monthly signals
    dominance_threshold_moderate = 0.6 # For less strict signals
    
    # Counts for patterns
    weekly_pattern_count = diffs_days[diffs_days.between(6, 8)].count() # Primarily 7-day diffs
    bi_weekly_pattern_count = diffs_days[diffs_days.between(13,15)].count()
    
    monthly_pattern_count_strict = diffs_days[diffs_days.between(28, 31)].count() # Typical month lengths
    monthly_pattern_count_approx = diffs_days[diffs_days.between(27,32)].count() # More robust

    daily_pattern_count = diffs_days[diffs_days == 1].count()
    business_daily_pattern_count = diffs_days[diffs_days.isin([1, 3])].count() # Fri-Mon = 3 days

    # --- DETAILED DEBUG PRINTS ---
    print(f"[InferFreq DEBUG - {series_name}] --- Diff Analysis Stats ---")
    print(f"[InferFreq DEBUG - {series_name}] total_diffs: {total_diffs}")
    if total_diffs > 0:
        print(f"[InferFreq DEBUG - {series_name}] daily_pattern_count: {daily_pattern_count} ({(daily_pattern_count/total_diffs)*100:.2f}%)")
        # print(f"[InferFreq DEBUG - {series_name}] business_daily_pattern_count: {business_daily_pattern_count} ({(business_daily_pattern_count/total_diffs)*100:.2f}%)") # Less critical for now
        print(f"[InferFreq DEBUG - {series_name}] weekly_pattern_count (6-8 days): {weekly_pattern_count} ({(weekly_pattern_count/total_diffs)*100:.2f}%)")
        # print(f"[InferFreq DEBUG - {series_name}] bi_weekly_pattern_count (13-15 days): {bi_weekly_pattern_count} ({(bi_weekly_pattern_count/total_diffs)*100:.2f}%)")
        print(f"[InferFreq DEBUG - {series_name}] monthly_pattern_count_approx (27-32 days): {monthly_pattern_count_approx} ({(monthly_pattern_count_approx/total_diffs)*100:.2f}%)")
        
        # Print mode of dayofweek if series index is DatetimeIndex
        if isinstance(series.index, pd.DatetimeIndex):
            try:
                if not series.index.empty:
                    s_dayofweek = pd.Series(series.index.dayofweek)
                    # mode() can return multiple values if there's a tie, tolist() is good.
                    print(f"[InferFreq DEBUG - {series_name}] series.index.dayofweek.mode(): {s_dayofweek.mode().tolist()}")
                else:
                    print(f"[InferFreq DEBUG - {series_name}] series.index is empty, cannot get dayofweek.mode().")
            except Exception as e_dom:
                print(f"[InferFreq DEBUG - {series_name}] Error getting dayofweek.mode(): {e_dom}")
        else:
            print(f"[InferFreq DEBUG - {series_name}] series.index is not DatetimeIndex, cannot get dayofweek.mode(). Type: {type(series.index)}")

        print(f"[InferFreq DEBUG - {series_name}] diffs_days.max(): {diffs_days.max() if not diffs_days.empty else 'N/A'}")
        print(f"[InferFreq DEBUG - {series_name}] diffs_days.min(): {diffs_days.min() if not diffs_days.empty else 'N/A'}")
        print(f"[InferFreq DEBUG - {series_name}] diffs_days.median(): {diffs_days.median() if not diffs_days.empty else 'N/A'}")
        print(f"[InferFreq DEBUG - {series_name}] diffs_days.mean(): {diffs_days.mean() if not diffs_days.empty else 'N/A'}")
        # For more detail, print value counts of diffs if not too many unique values
        if not diffs_days.empty and diffs_days.nunique() < 20:
            print(f"[InferFreq DEBUG - {series_name}] diffs_days.value_counts():\\n{diffs_days.value_counts().sort_index()}")
        elif not diffs_days.empty:
            print(f"[InferFreq DEBUG - {series_name}] diffs_days.nunique(): {diffs_days.nunique()} (Too many unique diffs to print value_counts)")
            
    print(f"[InferFreq DEBUG - {series_name}] --- End Diff Analysis Stats ---")
    # --- END DETAILED DEBUG PRINTS ---

    # 2. Try pandas infer_freq
    inferred_code = pd.infer_freq(series.index)
    print(f"[InferFreq - {series_name}] pd.infer_freq result: {inferred_code}")

    # 3. Decision Logic based on combined info

    # Priority 1: Strong signals from pd.infer_freq for higher frequencies (Weekly, Monthly, etc.)
    if inferred_code:
        if any(c in inferred_code for c in ['AS', 'A', 'YS', 'Y']): # Annual
            return {'freq_group': "Annual", 'pandas_freq_code': inferred_code, 'details': f"Pandas inferred: {inferred_code}"}
        if any(c in inferred_code for c in ['QS', 'Q']): # Quarterly
            return {'freq_group': "Quarterly", 'pandas_freq_code': inferred_code, 'details': f"Pandas inferred: {inferred_code}"}
        if 'M' in inferred_code: # MS, M, ME - Monthly
             # If 'M' (month-end) or 'MS' (month-start) check diffs to confirm, as 'M' alone can be ambiguous.
            if monthly_pattern_count_approx / total_diffs > dominance_threshold_moderate or len(series) < 10 : # If month diffs are strong OR series is short
                return {'freq_group': "Monthly", 'pandas_freq_code': inferred_code, 'details': f"Pandas inferred: {inferred_code} (Diffs support monthly)"}
            # If 'M' but diffs don't strongly support monthly, it might be a misinterpretation by pandas for sparse weekly/daily.
            # Let it fall through to diff analysis.
            print(f"[InferFreq - {series_name}] Pandas gave '{inferred_code}', but monthly diffs not dominant. Will check diffs.")

        if 'W' in inferred_code: # W, W-MON, W-TUE etc. - Weekly
            # If 'W' type, check diffs to confirm, as 'W' alone can be from daily data with weekly seasonality.
            if weekly_pattern_count / total_diffs > dominance_threshold_moderate or len(series) < 20: # If weekly diffs are strong OR series is short
                 return {'freq_group': "Weekly", 'pandas_freq_code': inferred_code, 'details': f"Pandas inferred: {inferred_code} (Diffs support weekly)"}
            print(f"[InferFreq - {series_name}] Pandas gave '{inferred_code}', but weekly diffs not dominant. Will re-check diffs.")
            # If 'W' but diffs don't strongly support weekly, it might be a misinterpretation. Fall through.

    # Priority 2: Strong signals from diff analysis (if pd.infer_freq was None or inconclusive for W/M)
    if total_diffs > 0 : # Ensure there are diffs to analyze
        if weekly_pattern_count / total_diffs > dominance_threshold_strict:
            # Try to refine pandas_freq_code if original inferred_code was weekly-like but overridden
            best_w_code = 'W' # Generic weekly if cannot refine to specific day
            if inferred_code and 'W-' in inferred_code:
                best_w_code = inferred_code
            elif not series.index.empty: # Check if index is not empty
                dayofweek_series = pd.Series(series.index.dayofweek)
                if not dayofweek_series.mode().empty: # Check if mode is not empty
                    mode_day = dayofweek_series.mode().iloc[0]
                    day_map = {0: 'W-MON', 1: 'W-TUE', 2: 'W-WED', 3: 'W-THU', 4: 'W-FRI', 5: 'W-SAT', 6: 'W-SUN'}
                    if mode_day in day_map:
                        best_w_code = day_map[mode_day]
            return {'freq_group': "Weekly", 'pandas_freq_code': best_w_code, 'details': f"Diff analysis: ~7 day diffs dominant ({weekly_pattern_count}/{total_diffs})."}
        
        if monthly_pattern_count_approx / total_diffs > dominance_threshold_strict:
            best_m_code = 'M' # Default
            if inferred_code and 'M' in inferred_code : best_m_code = inferred_code
            elif (series.index.is_month_start).mean() > dominance_threshold_moderate : best_m_code = "MS"
            elif (series.index.is_month_end).mean() > dominance_threshold_moderate : best_m_code = "ME"
            return {'freq_group': "Monthly", 'pandas_freq_code': best_m_code, 'details': f"Diff analysis: ~30 day diffs dominant ({monthly_pattern_count_approx}/{total_diffs})."}

    # Priority 3: Daily signals (from pd.infer_freq or diffs)
    if inferred_code:
        if 'B' in inferred_code: # Business Daily
            # Check if diffs support this (mostly 1 or 3 days)
            if business_daily_pattern_count / total_diffs > dominance_threshold_moderate or daily_pattern_count / total_diffs > dominance_threshold_moderate :
                return {'freq_group': "Business Daily", 'pandas_freq_code': inferred_code, 'details': f"Pandas inferred: {inferred_code} (Diffs support B)"}
            print(f"[InferFreq - {series_name}] Pandas gave '{inferred_code}', but B-Day diffs not dominant. Re-evaluating.")
        if 'D' in inferred_code: # Daily
            # If pandas says 'D', ensure it's not actually a very regular weekly/monthly series that looks daily.
            # If weekly_pattern_count or monthly_pattern_count are high, it might be misclassified by pandas.
            # This was partially handled by Priority 1 & 2. Here, if it reaches this point and pandas says 'D', it's likely daily.
            if weekly_pattern_count / total_diffs < 0.5 and monthly_pattern_count_approx / total_diffs < 0.5: # Not strongly weekly or monthly
                 # Additional check: are values mostly unique day-to-day? Or just filled weekly data?
                 # This would require looking at series.values, which is beyond simple index frequency.
                 # For now, accept 'D' if not strongly weekly/monthly by diffs.
                return {'freq_group': "Daily", 'pandas_freq_code': inferred_code, 'details': f"Pandas inferred: {inferred_code} (Not strongly W/M by diffs)"}
            else:
                print(f"[InferFreq - {series_name}] Pandas inferred 'D', but diffs suggest W/M patterns might be present. Re-evaluating with diff priority.")
                 # Fall through to diff-based daily check if W/M from diffs wasn't dominant enough for Priority 2

    if total_diffs > 0: # Re-check daily based on diffs if pandas inference was overridden or None
        if daily_pattern_count / total_diffs > dominance_threshold_moderate and diffs_days.max() < 6 : # Max diff < 6 to avoid confusing with sparse weekly
             # Ensure it's not a misidentified weekly series that happens to have many 1-day diffs due to fill.
            if weekly_pattern_count / total_diffs < dominance_threshold_moderate : # If not moderately weekly
                return {'freq_group': "Daily", 'pandas_freq_code': "D", 'details': f"Diff analysis: Primarily 1-day diffs ({daily_pattern_count}/{total_diffs}), max diff < 6."}

    # Priority 4: Sub-daily from pd.infer_freq (if not caught by higher priorities)
    if inferred_code:
        if any(c in inferred_code for c in ['H', 'T', 'min', 'S', 'L', 'ms', 'U', 'us', 'N']):
            return {'freq_group': "Sub-Daily", 'pandas_freq_code': inferred_code, 'details': f"Pandas inferred: {inferred_code}"}
        # If inferred_code is not None but doesn't match any above, it's specific.
        return {'freq_group': "Specific", 'pandas_freq_code': inferred_code, 'details': f"Pandas inferred specific code: {inferred_code}"}

    # Priority 5: Fallback to irregular if all else fails
    return {'freq_group': "Irregular", 'pandas_freq_code': None, 'details': "No dominant regular pattern found by combined analysis."}
    # --- End Refactored Logic ---

def infer_dataframe_frequency(df: pd.DataFrame, time_col_name: str):
    """
    Infers frequencies for all data columns in a DataFrame based on a specified time column.
    Args:
        df (pd.DataFrame): The input DataFrame.
        time_col_name (str): The name of the column containing datetime information.
    Returns:
        dict: {
            'overall_inferred_freq_group': str, 
            'overall_pandas_freq_code': str,
            'column_analysis': {
                'col_name': {'freq_group': str, 'pandas_freq_code': str, 'details': str, 'non_nan_data_points': int}
            },
            'time_column_status': {'name': str, 'is_datetime_parseable': bool, 'parse_error': str or None}
        }
    """
    if time_col_name not in df.columns:
        return {
            'overall_inferred_freq_group': "Error", 
            'overall_pandas_freq_code': None,
            'column_analysis': {},
            'time_column_status': {'name': time_col_name, 'is_datetime_parseable': False, 'parse_error': "Time column not found in DataFrame."}
        }

    try:
        # Attempt to parse the time column robustly
        time_values_parsed = pd.to_datetime(df[time_col_name], errors='coerce')
        if time_values_parsed.isnull().all():
            return {
                'overall_inferred_freq_group': "Error", 
                'overall_pandas_freq_code': None,
                'column_analysis': {},
                'time_column_status': {'name': time_col_name, 'is_datetime_parseable': False, 'parse_error': "Time column could not be parsed to datetime (all values became NaT)."}
            }
        time_column_status = {'name': time_col_name, 'is_datetime_parseable': True, 'parse_error': None}
    except Exception as e:
        return {
            'overall_inferred_freq_group': "Error", 
            'overall_pandas_freq_code': None,
            'column_analysis': {},
            'time_column_status': {'name': time_col_name, 'is_datetime_parseable': False, 'parse_error': f"Error parsing time column: {e}"}
        }

    column_analysis = {}
    all_freq_groups = []
    all_pandas_codes = []

    data_columns = [col for col in df.columns if col != time_col_name]

    for col_name in data_columns:
        # Create a series with the parsed time index and the current data column
        # Drop rows where the data value is NaN, as these don't contribute to frequency of observation
        # Also drop rows where the parsed time index is NaT
        series_for_freq_analysis = pd.Series(df[col_name].values, index=time_values_parsed)
        series_for_freq_analysis = series_for_freq_analysis[series_for_freq_analysis.index.notna()] # Remove NaT indices
        series_for_freq_analysis = series_for_freq_analysis.dropna() # Remove NaN data values

        non_nan_points = len(series_for_freq_analysis)
        
        if non_nan_points < 2:
            analysis_result = {'freq_group': "Undetermined", 'pandas_freq_code': None, 'details': "Not enough non-NaN data points."}
        else:
            analysis_result = infer_series_frequency(series_for_freq_analysis, str(col_name))
        
        column_analysis[str(col_name)] = {**analysis_result, 'non_nan_data_points': non_nan_points}
        if analysis_result['freq_group'] not in ["Undetermined", "Irregular", "Error"]:
            all_freq_groups.append(analysis_result['freq_group'])
            if analysis_result['pandas_freq_code']:
                 all_pandas_codes.append(analysis_result['pandas_freq_code'])
    
    # Determine overall frequency
    # Simple majority vote for freq_group, more complex for pandas_freq_code if needed
    overall_freq_group = "Mixed/Irregular"
    overall_pandas_code = None

    if not all_freq_groups: # No valid frequencies found for any column
        overall_freq_group = "Undetermined"
    else:
        from collections import Counter
        freq_group_counts = Counter(all_freq_groups)
        most_common_group, count = freq_group_counts.most_common(1)[0]
        if count / len(all_freq_groups) > 0.5: # If more than 50% agree on a freq_group
            overall_freq_group = most_common_group
            # Try to get a consensus pandas_freq_code for this group
            # This is a simplification; might need more nuanced logic if codes within a group vary (e.g. W-MON, W-TUE)
            related_pandas_codes = [
                col_info['pandas_freq_code'] 
                for col_info in column_analysis.values() 
                if col_info['freq_group'] == overall_freq_group and col_info['pandas_freq_code'] is not None
            ]
            if related_pandas_codes:
                pandas_code_counts = Counter(related_pandas_codes)
                most_common_pandas_code, _ = pandas_code_counts.most_common(1)[0]
                overall_pandas_code = most_common_pandas_code
        elif len(set(all_freq_groups)) == 1: # All columns agree, even if not a majority of total columns (e.g. only one data column)
             overall_freq_group = all_freq_groups[0]
             if all_pandas_codes:
                pandas_code_counts = Counter(all_pandas_codes) # Recalculate for this specific case
                most_common_pandas_code, _ = pandas_code_counts.most_common(1)[0]
                overall_pandas_code = most_common_pandas_code


    return {
        'overall_inferred_freq_group': overall_freq_group,
        'overall_pandas_freq_code': overall_pandas_code,
        'column_analysis': column_analysis,
        'time_column_status': time_column_status
    }


def align_dataframe_frequency(df: pd.DataFrame, time_col_name: str, 
                              target_freq_group: str, # "Weekly" or "Monthly"
                              alignment_detail, # For Weekly: 0-6 (Mon-Sun). For Monthly: 1-31 or "L" (Last day)
                              duplicate_resolution_method: str = 'last'): # 'first', 'last', 'mean', 'median', 'sum'
    print("--- [DEBUG align_dataframe_frequency] --- ENTERING FUNCTION ---")
    print(f"Input df shape: {df.shape}")
    print(f"Input df dtypes:\n{df.dtypes}")
    print(f"Input df head:\n{df.head().to_string()}")
    print(f"Time column name: {time_col_name}")
    print(f"Target frequency group: {target_freq_group}")
    print(f"Alignment detail: {alignment_detail}")
    print(f"Duplicate resolution method: {duplicate_resolution_method}")

    if df.empty:
        print("[DEBUG align_dataframe_frequency] Input DataFrame is empty. Returning as is.")
        return df.copy() # Return a copy to be safe

    df_aligned = df.copy()

    try:
        # Ensure the time column is in datetime format
        df_aligned[time_col_name] = pd.to_datetime(df_aligned[time_col_name], errors='coerce')
        if df_aligned[time_col_name].isnull().all():
            print(f"[AlignFreq] Error: Time column '{time_col_name}' could not be parsed to datetime (all NaT).")
            return df.copy()
        # Drop rows that couldn't be parsed to datetime for the time column
        df_aligned.dropna(subset=[time_col_name], inplace=True)
        if df_aligned.empty:
            print(f"[AlignFreq] Info: DataFrame is empty after dropping NaT in time column.")
            return df_aligned
    except Exception as e:
        print(f"[AlignFreq] Error parsing time column '{time_col_name}' for alignment: {e}")
        return df.copy()

    def _normalize_timestamp(ts):
        if pd.isna(ts):
            return pd.NaT
        
        aligned_date = None
        if target_freq_group == "Weekly":
            if not isinstance(alignment_detail, int) or not (0 <= alignment_detail <= 6):
                # Default to Monday if alignment_detail is invalid for weekly
                print(f"[AlignFreq] Warning: Invalid weekly alignment_detail '{alignment_detail}'. Defaulting to Monday (0).")
                day_offset = 0
            else:
                day_offset = alignment_detail
            # ts.isocalendar().week gives the week number, but we need the start of the week.
            # ts.dayofweek is Monday=0, Sunday=6
            week_start_offset = ts.dayofweek 
            aligned_date = ts - pd.to_timedelta(week_start_offset, unit='D') + pd.to_timedelta(day_offset, unit='D')
        
        elif target_freq_group == "Monthly":
            year, month = ts.year, ts.month
            if alignment_detail == 'L':
                # Last day of the month
                # next_month_first_day = pd.Timestamp(year=year, month=month, day=1) + pd.offsets.MonthBegin(1)
                # aligned_date = next_month_first_day - pd.Timedelta(days=1)
                # A simpler way using MonthEnd offset:
                aligned_date = ts.replace(day=1) + pd.offsets.MonthEnd(0)
            elif isinstance(alignment_detail, int) and 1 <= alignment_detail <= 31:
                try:
                    aligned_date = pd.Timestamp(year=year, month=month, day=alignment_detail)
                except ValueError: # Day is out of range for month
                    print(f"[AlignFreq] Warning: Day {alignment_detail} is invalid for {year}-{month}. Aligning to last day of month instead.")
                    aligned_date = pd.Timestamp(year=year, month=month, day=1) + pd.offsets.MonthEnd(0) # pd.offsets.MonthEnd gives last day
            else:
                # Default to first day of month if alignment_detail is invalid
                print(f"[AlignFreq] Warning: Invalid monthly alignment_detail '{alignment_detail}'. Defaulting to 1st of month.")
                aligned_date = ts.replace(day=1)
        
        if aligned_date:
            return aligned_date.normalize() # Sets time to 00:00:00
        return pd.NaT # Should not happen if logic is correct

    df_aligned[time_col_name] = df_aligned[time_col_name].apply(_normalize_timestamp)
    df_aligned.dropna(subset=[time_col_name], inplace=True) # Drop if any normalization failed

    if df_aligned.empty:
        print(f"[AlignFreq] Info: DataFrame is empty after normalizing timestamps.")
        return df_aligned
        
    # Handle duplicates created by alignment
    # Identify numeric columns for methods like 'mean', 'median', 'sum'
    numeric_cols = df_aligned.select_dtypes(include=np.number).columns.tolist()
    if time_col_name in numeric_cols: # Should not happen if time_col_name is datetime
        numeric_cols.remove(time_col_name) 
    
    # Columns to group by (all non-numeric data columns + the time column)
    # This is a bit simplistic. If there are other categorical columns that should define unique series, they should be included.
    # For now, assuming all other columns are values to be aggregated or kept.
    
    # Create a list of original columns to keep, excluding the time column temporarily for grouping
    value_cols = [col for col in df_aligned.columns if col != time_col_name]
    
    if not value_cols: # Only time column left
        df_aligned = df_aligned.drop_duplicates(subset=[time_col_name], keep=duplicate_resolution_method if duplicate_resolution_method in ['first', 'last'] else 'last')
    elif duplicate_resolution_method in ['first', 'last']:
        df_aligned = df_aligned.sort_values(by=time_col_name).groupby(time_col_name, as_index=False).agg(duplicate_resolution_method)
        # Reorder columns to original order (approximately, groupby might change it)
        # df_aligned = df_aligned[[time_col_name] + [col for col in df.columns if col != time_col_name and col in df_aligned.columns]]

    elif duplicate_resolution_method in ['mean', 'median', 'sum'] and numeric_cols:
        agg_dict = {}
        for col in value_cols:
            if col in numeric_cols:
                agg_dict[col] = duplicate_resolution_method
            else: # For non-numeric columns, default to 'last' or 'first'
                agg_dict[col] = 'last' 
        
        df_aligned = df_aligned.groupby(time_col_name, as_index=False).agg(agg_dict)
        # df_aligned = df_aligned[[time_col_name] + [col for col in df.columns if col != time_col_name and col in df_aligned.columns]]
    else:
        print(f"[AlignFreq] Warning: Duplicate resolution method '{duplicate_resolution_method}' is not suitable or no numeric columns for it. Defaulting to 'last'.")
        df_aligned = df_aligned.sort_values(by=time_col_name).groupby(time_col_name, as_index=False).agg('last')
        # df_aligned = df_aligned[[time_col_name] + [col for col in df.columns if col != time_col_name and col in df_aligned.columns]]

    # Ensure original column order is preserved as much as possible
    # The groupby().agg() operation might change column order or drop non-aggregated columns if not handled carefully.
    # A robust way is to ensure all original value_cols are in the agg_dict or handled.
    
    # Re-construct columns to maintain original order as best as possible
    final_columns_ordered = [time_col_name] + [col for col in df.columns if col != time_col_name and col in df_aligned.columns]
    # Add any new columns created by agg if any (should not happen with simple agg funcs)
    # final_columns_ordered.extend([col for col in df_aligned.columns if col not in final_columns_ordered])
    df_aligned = df_aligned[final_columns_ordered]

    print(f"[AlignFreq] Frequency alignment completed. Method: {target_freq_group}, Detail: {alignment_detail}, Duplicates: {duplicate_resolution_method}. Shape: {df_aligned.shape}")
    print(f"Resampled df shape: {df_aligned.shape}")
    print(f"Resampled df dtypes:\n{df_aligned.dtypes}")
    print(f"Resampled df head:\n{df_aligned.head().to_string()}")
    print("--- [DEBUG align_dataframe_frequency] --- EXITING FUNCTION (SUCCESS) ---")
    return df_aligned

def identify_time_column(df_processed, manual_selection):
    """
    Identifies the time column (either manually selected or auto-detected), 
    parses it, and calculates its properties (frequency, start/end).
    Args:
        df_processed (pd.DataFrame): The DataFrame to analyze.
        manual_selection (str): The name of the manually selected column, 
                                or "(自动识别)" for auto-detection.
    Returns:
        dict: A dictionary containing time column info.
    """
    print(f"--- DEBUG (identify_time_column CALLED): df_processed shape: {df_processed.shape}, manual_selection: '{manual_selection}' ---")
    if not df_processed.empty:
        print(f"--- DEBUG (identify_time_column CALLED): df_processed.columns: {df_processed.columns.tolist()} ---")

    time_col_info = {
        'name': None, 'parsed_series': None, 'parse_format': None, 
        'freq': '未知', 'inferred_freq_code': None, 'start': None, 'end': None,
        'status_message': '', 'status_type': 'info'
    }
    identified_time_series = None
    identified_col_name_str = None
    parse_format_used = None
    parse_success = False

    if manual_selection != "(自动识别)":
        time_col_info['status_message'] = f"尝试解析手动选择的列: '{manual_selection}'..."
        time_col_info['status_type'] = 'info'
        selected_col_obj = None
        for col in df_processed.columns:
            if str(col) == manual_selection:
                selected_col_obj = col
                break
        if selected_col_obj is not None:
            parsed_series_manual, parse_format_manual = parse_time_column(df_processed[selected_col_obj])
            if parsed_series_manual is not None:
                identified_time_series = parsed_series_manual
                identified_col_name_str = manual_selection
                parse_format_used = parse_format_manual
                parse_success = True
                time_col_info['status_message'] = f"成功将手动选择的列 '{manual_selection}' 解析为时间格式 (方法: {parse_format_used})。"
                time_col_info['status_type'] = 'success'
            else:
                time_col_info['status_message'] = f"无法将手动选择的列 '{manual_selection}' 有效解析为时间格式。请检查数据或尝试自动识别。"
                time_col_info['status_type'] = 'warning'
        else:
            time_col_info['status_message'] = f"在数据中找不到手动选择的列 '{manual_selection}'。"
            time_col_info['status_type'] = 'error'
    else:
        time_col_info['status_message'] = "尝试自动识别时间列..."
        time_col_info['status_type'] = 'info'
        potential_time_cols_data = []
        for col_obj in df_processed.columns:
            col_name_str = str(col_obj)
            parsed_series_auto, parse_format_auto = parse_time_column(df_processed[col_obj])
            if parsed_series_auto is not None:
                potential_time_cols_data.append((parsed_series_auto, col_name_str, parse_format_auto))
        if potential_time_cols_data:
            common_time_names_lower = ['日期', '时间', 'date', 'time', '月份', '年份']
            preferred_cols = [data for data in potential_time_cols_data if any(tname.lower() in data[1].lower() for tname in common_time_names_lower)]
            if preferred_cols:
                identified_time_series, identified_col_name_str, parse_format_used = preferred_cols[0]
                time_col_info['status_message'] = f"自动识别到时间列 '{identified_col_name_str}' (方法: {parse_format_used})。"
                time_col_info['status_type'] = 'success'
            else:
                identified_time_series, identified_col_name_str, parse_format_used = potential_time_cols_data[0]
                time_col_info['status_message'] = f"自动识别到时间列 '{identified_col_name_str}' (基于解析成功, 方法: {parse_format_used})。"
                time_col_info['status_type'] = 'success'
            parse_success = True
        else:
            time_col_info['status_message'] = "未能自动识别到任何可以有效解析为时间格式的列。"
            time_col_info['status_type'] = 'info'

    if parse_success and identified_time_series is not None:
        time_col_info['name'] = identified_col_name_str
        time_col_info['parsed_series'] = identified_time_series
        time_col_info['parse_format'] = parse_format_used
        print(f"--- DEBUG (identify_time_column for '{identified_col_name_str}'): ---")
        print(f"identified_time_series non-NaT count: {identified_time_series.notna().sum()}, total length: {len(identified_time_series)}")
        time_series_to_use = identified_time_series.dropna()
        print(f"--- DEBUG (identify_time_column for '{identified_col_name_str}'): Passing to calculate_time_series_info ---")
        print(f"time_series_to_use name: {time_series_to_use.name}, non-NaT count: {time_series_to_use.notna().sum()}, total length: {len(time_series_to_use)}")
        if not time_series_to_use.empty:
            calculated_info = calculate_time_series_info(time_series_to_use)
            time_col_info.update(calculated_info)
        else:
            print(f"--- DEBUG (identify_time_column for '{identified_col_name_str}'): time_series_to_use is empty after dropna(). Skipping calculate_time_series_info. ---")
            time_col_info['status_message'] = f"列 '{identified_col_name_str}' 解析后无有效时间数据（可能全部为空或无法解析）。"
            time_col_info['status_type'] = 'warning'
            time_col_info['freq'] = '无法推断'
            time_col_info['inferred_freq_code'] = None
            time_col_info['start'] = None
            time_col_info['end'] = None
        
    debug_col_name_at_return = manual_selection if manual_selection != "(自动识别)" else (identified_col_name_str if identified_col_name_str else "AutoDetectFail")
    print(f'--- DEBUG (identify_time_column for "{debug_col_name_at_return}"): Final time_col_info before return: ---')
    info_to_print_at_return = {k: v for k, v in time_col_info.items() if k != 'parsed_series'}
    if time_col_info.get('parsed_series') is not None:
        info_to_print_at_return['parsed_series_stats'] = f"length={len(time_col_info['parsed_series'])}, non_na={time_col_info['parsed_series'].notna().sum()}"
    else:
        info_to_print_at_return['parsed_series_stats'] = "None"
    print(info_to_print_at_return)
    return time_col_info

def generate_final_data(df_processed, selected_cols, time_col_info, start_date, end_date, complete_time_index, manual_frequency_selection):
    """
    Generates the final DataFrame based on column selection, time filtering, and time series completion.
    """
    completion_status = {'completion_message': None}
    df_final = None
    if not selected_cols:
        return None, completion_status
    try:
        df_intermediate = df_processed[selected_cols].copy()
        time_col_name = time_col_info.get('name')
        is_time_col_kept = time_col_name and (time_col_name in [str(c) for c in selected_cols])

        if is_time_col_kept and (start_date or end_date):
            actual_time_col_obj_in_selected = None
            for col in df_intermediate.columns:
                if str(col) == time_col_name:
                    actual_time_col_obj_in_selected = col
                    break
            if actual_time_col_obj_in_selected is not None:
                parsed_time_for_filter, _ = parse_time_column(df_intermediate[actual_time_col_obj_in_selected])
                if parsed_time_for_filter is not None:
                    start_ts = pd.Timestamp(start_date) if start_date else None
                    end_ts = pd.Timestamp(end_date) + pd.Timedelta(days=1) if end_date else None
                    mask = pd.Series(True, index=df_intermediate.index)
                    if start_ts:
                        mask &= (parsed_time_for_filter.dt.normalize() >= start_ts)
                    if end_ts:
                        mask &= (parsed_time_for_filter.dt.normalize() < end_ts)
                    original_rows = len(df_intermediate)
                    df_intermediate = df_intermediate[mask]
                    filtered_rows = len(df_intermediate)
                    print(f"Time filtering applied ({start_date or '最早'} - {end_date or '最晚'}). Rows: {original_rows} -> {filtered_rows}")
                else:
                    print(f"Warning: Could not re-parse time column '{time_col_name}' for filtering.")
        
        df_final = df_intermediate

        if complete_time_index and df_final is not None and not df_final.empty and time_col_name and is_time_col_kept:
            # MODIFIED: Determine target_freq_code for asfreq
            # manual_frequency_selection from UI now directly provides the Pandas code or "自动"
            if manual_frequency_selection == "自动":
                target_freq_code = time_col_info.get('inferred_freq_code')
                print(f"Frequency completion mode: 自动. Using inferred_freq_code: {target_freq_code}")
            else:
                target_freq_code = manual_frequency_selection # This is now a direct Pandas frequency string like 'MS', 'M', etc.
                print(f"Frequency completion mode: 手动. Using selected freq_code: {target_freq_code}")
            
            if target_freq_code:
                actual_time_col_obj_for_reindex = next((col for col in df_final.columns if str(col) == time_col_name), None)
                if actual_time_col_obj_for_reindex is not None:
                    try:
                        df_copy = df_final.copy()
                        # <<< 调试打印开始 >>>
                        print("--- DEBUG: generate_final_data (inside complete_time_index block) ---")
                        print(f"Shape of df_copy (passed to asfreq logic): {df_copy.shape}")
                        print(f"df_copy.head(3) for asfreq:\n{df_copy.head(3)}")
                        print(f"Time column name for reindex: {actual_time_col_obj_for_reindex}")
                        if actual_time_col_obj_for_reindex in df_copy:
                            print(f"Content of time column in df_copy before re-parse:\n{df_copy[actual_time_col_obj_for_reindex].head(3)}")
                        else:
                            print(f"ERROR: Time column '{actual_time_col_obj_for_reindex}' not found in df_copy columns: {df_copy.columns.tolist()}")

                        parsed_time_for_reindex, parse_format_used = parse_time_column(df_copy[actual_time_col_obj_for_reindex])
                        print(f"Parse format used for reindex: {parse_format_used}")
                        print(f"Shape of parsed_time_for_reindex: {parsed_time_for_reindex.shape if parsed_time_for_reindex is not None else 'None'}")
                        if parsed_time_for_reindex is not None:
                            print(f"parsed_time_for_reindex.head(3):\n{parsed_time_for_reindex.head(3)}")
                            print(f"Number of NaTs in parsed_time_for_reindex: {parsed_time_for_reindex.isnull().sum()}")
                        # <<< 调试打印结束 >>>

                        if parsed_time_for_reindex is not None:
                            df_copy[actual_time_col_obj_for_reindex] = parsed_time_for_reindex

                            # --- REMOVED DATETIME ALIGNMENT FIX ---
                            # The logic to force month start alignment here is removed.
                            # Correct alignment should now be handled by using the precise target_freq_code in asfreq.
                            # --- END REMOVED DATETIME ALIGNMENT FIX ---
                            
                            df_indexed = df_copy.set_index(actual_time_col_obj_for_reindex)

                            # Ensure index is DatetimeIndex and sorted, and no duplicates
                            if not isinstance(df_indexed.index, pd.DatetimeIndex):
                                print("--- DEBUG: Converting index to DatetimeIndex before resample/asfreq. ---")
                                df_indexed.index = pd.to_datetime(df_indexed.index, errors='coerce')
                                df_indexed = df_indexed[df_indexed.index.notna()] 
                            
                            if df_indexed.empty: # If all index values became NaT and were dropped
                                print("--- DEBUG: df_indexed became empty after DatetimeIndex conversion and NaT drop. Skipping resample/asfreq. ---")
                                df_resampled = df_indexed.copy() # Will be an empty DataFrame
                            else:
                                if not df_indexed.index.is_monotonic_increasing:
                                    print("--- DEBUG: Time index not monotonic. Sorting before resample/asfreq. ---")
                                    df_indexed = df_indexed.sort_index()
                                
                                if df_indexed.index.has_duplicates:
                                    print("--- DEBUG: Dropping duplicate time index entries (keeping first) before resample/asfreq. ---")
                                    df_indexed = df_indexed[~df_indexed.index.duplicated(keep='first')]

                                original_row_count = len(df_indexed) # DEFINITION OF original_row_count

                                print(f"--- DEBUG: df_indexed BEFORE resample/asfreq ---")
                                print(f"Shape: {df_indexed.shape}")
                                print(f"Index type: {type(df_indexed.index)}")
                                print(f"Is monotonic increasing: {df_indexed.index.is_monotonic_increasing}")
                                print(f"Number of NaNs in index: {df_indexed.index.isna().sum()}")
                                if not df_indexed.empty and len(df_indexed.index) > 0 :
                                    try:
                                        index_sample_for_print = df_indexed.index[:20].tolist()
                                        if len(df_indexed.index) > 40:
                                            index_sample_for_print.append("...")
                                            index_sample_for_print.extend(df_indexed.index[-20:].tolist())
                                        print(f"Index values (first 20, last 20 if many):\\n{index_sample_for_print}")
                                    except Exception as e_print_idx:
                                        print(f"Error printing index sample: {e_print_idx}")
                                else:
                                    print("Index is empty or has no values to sample.")
                                print(f"Target frequency code for resample/asfreq: {target_freq_code}")

                                # --- MODIFIED LOGIC ---
                                df_resampled = df_indexed.resample(target_freq_code).first()
                                
                                print(f"--- DEBUG: df_resampled AFTER resample().first() ---")
                                print(f"Shape: {df_resampled.shape}")
                                print(f"Index type: {type(df_resampled.index)}")
                                if isinstance(df_resampled.index, pd.DatetimeIndex):
                                    print(f"Is monotonic increasing: {df_resampled.index.is_monotonic_increasing}")
                                    print(f"Number of NaTs in index: {df_resampled.index.isnull().sum()}")
                                    print(f"Index values (first 20, last 20 if many):")
                                    print(df_resampled.index.tolist()[:20])
                                    if len(df_resampled.index) > 40:
                                        print("...")
                                        print(df_resampled.index.tolist()[-20:])
                                    elif len(df_resampled.index) > 20:
                                        print(df_resampled.index.tolist()[20:])
                                
                                # Try to print the specific problematic range
                                try:
                                    start_slice = pd.Timestamp('2007-12-01')
                                    end_slice = pd.Timestamp('2008-02-28') # Use a valid end date for slicing
                                    # Check if index has data to slice before attempting
                                    if not df_resampled.empty and df_resampled.index.min() <= end_slice and df_resampled.index.max() >= start_slice:
                                        print(f"df_resampled around 2007-12 to 2008-02:\n{df_resampled.loc[start_slice:end_slice]}")
                                    else:
                                        print(f"df_resampled does not cover the range 2007-12 to 2008-02 for detailed view. Min: {df_resampled.index.min()}, Max: {df_resampled.index.max()}")
                                except KeyError as ke:
                                    print(f"KeyError when trying to slice df_resampled for 2007-12 to 2008-02: {ke}. This might mean the dates are not in the index after asfreq.")
                                except Exception as e_slice:
                                    print(f"Error slicing df_resampled for 2007-12 to 2008-02: {e_slice}")
                                # This else corresponds to: if isinstance(df_resampled.index, pd.DatetimeIndex)
                                else: 
                                    print("df_resampled.index is not DatetimeIndex after asfreq.")
                            
                            if not df_resampled.empty:
                                non_nan_rows_count = (~df_resampled.isnull().all(axis=1)).sum()
                                print(f"Number of non-NaN rows in df_resampled (sum over columns > 0): {non_nan_rows_count}")
                                completed_row_count_for_check = len(df_resampled)
                                if non_nan_rows_count == 0 and original_row_count > 0 and completed_row_count_for_check > original_row_count:
                                    print("CRITICAL WARNING: All original data appears lost after asfreq, but new rows were added!")
                                elif non_nan_rows_count < original_row_count and original_row_count > 0 and not df_indexed.empty and (~df_indexed.isnull().all(axis=1)).sum() > non_nan_rows_count :
                                    print(f"WARNING: Some data loss during asfreq? Non-NaN rows before asfreq (in df_indexed, approx): {(~df_indexed.isnull().all(axis=1)).sum()}, after: {non_nan_rows_count}")
                            else:
                                print("df_resampled is empty after asfreq.")
                            # <<< 增强的调试打印结束 >>>

                            completed_row_count = len(df_resampled) if df_resampled is not None else original_row_count
                            df_final = df_resampled # Assign before potential reset_index

                            # --- MODIFIED FIX: Reset index and rename time column ---
                            original_time_col_name = str(actual_time_col_obj_for_reindex) # Define before if

                            if df_final is not None: # Only proceed if asfreq returned something
                                df_final = df_final.reset_index() # Reset index unconditionally if df_final exists
                                
                                # Rename the new column (that was the index) to the original time column name.
                                # reset_index() names the new column with old index name, or 'index' if unnamed.
                                current_time_col_in_df = None
                                if actual_time_col_obj_for_reindex in df_final.columns: # Index had a name, and it was the time col
                                    current_time_col_in_df = actual_time_col_obj_for_reindex
                                elif 'index' in df_final.columns: # Index was unnamed, reset_index created 'index'
                                    current_time_col_in_df = 'index'
                                elif df_final.columns[0] == original_time_col_name: # Catch cases where it somehow already matches
                                    current_time_col_in_df = original_time_col_name
                                else: # Fallback if first col is different and not 'index' or original name
                                     if not df_final.empty:
                                        current_time_col_in_df = df_final.columns[0]
                                        print(f"Warning: Time column after reset_index is '{current_time_col_in_df}', expected something like '{original_time_col_name}' or 'index'.")
                                     else:
                                        print("Warning: df_final is empty after reset_index, cannot determine current time column name for rename.")

                                if current_time_col_in_df and current_time_col_in_df != original_time_col_name:
                                    df_final = df_final.rename(columns={current_time_col_in_df: original_time_col_name})
                                    print(f"Time column '{current_time_col_in_df}' successfully renamed to '{original_time_col_name}'.")
                                elif current_time_col_in_df == original_time_col_name:
                                    print(f"Time column '{original_time_col_name}' is already correct after reset_index.")
                                else:
                                    print(f"Could not reliably rename the time column after reset_index. Current columns: {df_final.columns.tolist()}")
                            else: # df_final was None (asfreq failed)
                                print("df_final is None after asfreq, skipping reset_index and rename.")
                                # df_final remains None or original df_intermediate if asfreq failed earlier
                            # --- END MODIFIED FIX ---

                            msg_type = 'success' if original_row_count != completed_row_count and df_final is not None else 'info'
                            msg_content = f"已根据频率 '{target_freq_code}' 自动补全时间序列。行数从 {original_row_count} 变为 {completed_row_count}。新行的变量值为NaN." if msg_type == 'success' else f"时间序列已连续（频率 '{target_freq_code}'），或补全未执行/失败。"
                            completion_status['completion_message'] = {'type': msg_type, 'content': msg_content}
                            
                            print(f"--- DEBUG: df_final after asfreq and potential reset_index (final state before return) ---")
                            print(f"Shape: {df_final.shape if df_final is not None else 'None'}")
                            if df_final is not None: print(f"Columns: {df_final.columns.tolist()}")
                            if df_final is not None and not df_final.empty and original_time_col_name in df_final.columns:
                                print(f"df_final['{original_time_col_name}'].head(3):\n{df_final[original_time_col_name].head(3)}")
                            else:
                                print(f"'{original_time_col_name}' not in columns or df_final is empty after freq completion (final state). Error or expected if asfreq failed.") 
                        else:
                            completion_status['completion_message'] = {'type': 'warning', 'content': f'无法解析时间列 \'{time_col_name}\' (可能值不规范)，无法进行频率补全。'}
                            print(f"Warning: Time column '{time_col_name}' could not be parsed for reindexing. Skipping asfreq.")
                            # df_final remains df_intermediate if reindex prep fails
                    except Exception as e_asfreq:
                        completion_status['completion_message'] = {'type': 'error', 'content': f'执行频率补全 (asfreq) 时出错: {e_asfreq}'}
                        print(f"Error during asfreq: {e_asfreq}")
                        # df_final might remain df_intermediate or be None depending on where error occurred
                else:
                    completion_status['completion_message'] = {'type': 'warning', 'content': f'在选定列中找不到时间列 \'{time_col_name}\' (可能未选择或已被重命名)，无法补全。'}
                    print(f"Warning: Time column '{time_col_name}' not found in df_final for reindexing. Skipping asfreq.")
            else:
                completion_status['completion_message'] = {'type': 'info', 'content': '未指定有效的目标频率或无法推断频率，跳过频率补全。'}
                print("Info: No valid target frequency for asfreq. Skipping.")
        else:
            # Conditions for completion not met (e.g., no time column, data empty, etc.)
            if not complete_time_index:
                pass # Not an error, user didn't request it
            elif df_final is None or df_final.empty:
                completion_status['completion_message'] = {'type': 'info', 'content': '数据为空，无法执行频率补全。'}
            elif not time_col_name:
                completion_status['completion_message'] = {'type': 'info', 'content': '未识别时间列，无法执行频率补全。'}
            elif not is_time_col_kept:
                completion_status['completion_message'] = {'type': 'info', 'content': f'时间列 \'{time_col_name}\' 未包含在选定列中，无法补全。'}

    except KeyError as e_key:
        print(f"KeyError in generate_final_data: {e_key}")
        completion_status['completion_message'] = {'type': 'error', 'content': f'处理数据时发生列错误: {e_key}，请检查列选择和重命名是否正确。'}
        return None, completion_status
    except Exception as e:
        print(f"Error in generate_final_data: {e}")
        completion_status['completion_message'] = {'type': 'error', 'content': f'生成最终数据时发生未知错误: {e}'}
        return None, completion_status

    return df_final, completion_status

# Helper function to get or infer frequency code for generate_final_data
# This might be more useful if generate_final_data is split or refactored.
def _get_frequency_code_for_completion(manual_selection, time_col_info):
    if manual_selection != "自动":
        options_map = {
            "月度 (M/MS)": 'MS', "周度 (W)": 'W-MON', "日度 (D/B)": 'D',
            "季度 (Q/QS)": 'QS', "年度 (A/AS)": 'AS'
        }
        return options_map.get(manual_selection)
    return time_col_info.get('inferred_freq_code') 