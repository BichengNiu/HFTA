import pandas as pd
import numpy as np
import datetime # Required for isinstance check with pd.Timestamp
from ..utils.time_analysis import identify_time_column # Assuming it's in the same utils or accessible

# --- FREQUENCY DEFINITIONS AND HELPER (Copied from time_column_ui.py for backend use) ---
FREQ_HIERARCHY = {
    'D': 0, 'B': 0, 'W': 1, 'M': 2, 'Q': 3, 'A': 4
}
def get_freq_level(pandas_freq_code):
    if not pandas_freq_code: return None
    base_code = pandas_freq_code.split('-')[0].upper()
    if base_code.startswith('W'): return FREQ_HIERARCHY['W']
    if base_code.startswith('M'): return FREQ_HIERARCHY['M']
    if base_code.startswith('Q'): return FREQ_HIERARCHY['Q']
    if base_code.startswith('A'): return FREQ_HIERARCHY['A']
    if base_code == 'D' or base_code == 'B': return FREQ_HIERARCHY['D']
    return None
# --- END OF FREQUENCY DEFINITIONS ---


def _align_single_dataframe(
    df_input: pd.DataFrame,
    time_col_name: str,
    original_pandas_freq_code: str | None,
    target_pandas_freq_code: str,
    alignment_mode: str,
    selected_stat_agg_method_code: str | None,
    parent_function_events_for_report: list # List to append report events to
) -> pd.DataFrame | None:
    """
    Helper function to align a single DataFrame based on the specified mode and frequency.
    Returns the aligned DataFrame or None if an error occurs.
    Modifies parent_function_events_for_report by appending new events.
    """
    if df_input is None or df_input.empty:
        # print("[_align_single_dataframe] Input DataFrame is None or empty.")
        return df_input # Return as is

    df_align_helper = df_input.copy()
    actual_events_for_report = [] # Ensure this is initialized for all paths
    temp_events_assign = [] # Initialize temp_events_assign for high-to-low custom path specifically


    # Ensure time column is DatetimeIndex
    if df_align_helper.index.name != time_col_name:
        if time_col_name not in df_align_helper.columns:
            # print(f"[_align_single_dataframe] Time column '{time_col_name}' not found in DataFrame. Skipping alignment for this df.")
            return df_input # Return original if time col is missing
        if not pd.api.types.is_datetime64_any_dtype(df_align_helper[time_col_name]):
            df_align_helper[time_col_name] = pd.to_datetime(df_align_helper[time_col_name], errors='coerce')
        df_align_helper = df_align_helper.dropna(subset=[time_col_name])
        if df_align_helper.empty:
            # print(f"[_align_single_dataframe] DataFrame empty after NaT drop for time column '{time_col_name}'.")
            # Return an empty df with target index structure if possible, or original columns
            # This part needs careful handling to ensure the outer function gets a consistently structured empty df
            return pd.DataFrame(columns=df_input.columns) # Simplified: return empty with original cols

        df_align_helper = df_align_helper.set_index(time_col_name)
        if df_align_helper is None or df_align_helper.empty: # Check again after set_index
            # print(f"[_align_single_dataframe] DataFrame empty or None after setting index '{time_col_name}'.")
            return pd.DataFrame(columns=df_input.columns)


    # Determine if high-to-low frequency conversion
    base_level = get_freq_level(original_pandas_freq_code)
    target_level = get_freq_level(target_pandas_freq_code)
    is_high_to_low = False
    if base_level is not None and target_level is not None and original_pandas_freq_code:
        if base_level < target_level:
            is_high_to_low = True

    df_final_resampled = None
    # actual_events_for_report = [] # Ensure this is initialized <- Moved up for clarity
    # temp_events_assign = [] # Initialize temp_events_assign <- Moved up for clarity

    # <<< DEBUG >>> Specific column to trace
    debug_col_name = "中国:生产率:焦炉:国内独立焦化厂(230家)"
    # print(f"DEBUG_V2 [{time_col_name}]: --- Entering _align_single_dataframe ---")
    # print(f"DEBUG_V2 [{time_col_name}]: OrigFreq: {original_pandas_freq_code}, TargetFreq: {target_pandas_freq_code}, Mode: {alignment_mode}, IsHighToLow: {is_high_to_low}")
    if df_input is not None and debug_col_name in df_input.columns:
        # print(f"DEBUG_V2 [{time_col_name}]: Input '{debug_col_name}' non-NaN count: {df_input[debug_col_name].notna().sum()}")
        pass
    else:
        # print(f"DEBUG_V2 [{time_col_name}]: Input df_input is None or '{debug_col_name}' not in columns.")
        pass

    # If not high-to-low conversion, ensure index frequency is None before resampling
    # to avoid resampling based on inferred anchors from original low frequency (e.g., 'ME' month-end).
    if not is_high_to_low and df_align_helper.index.freq is not None:
        # print(f"DEBUG_V2 [{time_col_name}]: Clearing index frequency (was {df_align_helper.index.freq}) for upsampling/same-frequency alignment.")
        df_align_helper.index.freq = None


    if alignment_mode == 'value_align' and is_high_to_low:
        # print(f"DEBUG_V2 [{time_col_name}]: Path: CUSTOM High-to-Low Value Alignment")
        temp_events_push = []
        
        if df_align_helper.empty:
            # print(f"DEBUG_V2 [{time_col_name}]: df_align_helper is empty before custom alignment. Returning early.")
            # ... (rest of the early return logic) ...
            return pd.DataFrame(columns=df_align_helper.columns)

        if not df_align_helper.index.is_monotonic_increasing:
            df_align_helper = df_align_helper.sort_index()

        # Generate a complete target frequency index based on the span of the original data
        min_orig_time = df_align_helper.index.min()
        max_orig_time = df_align_helper.index.max()
        # Use a helper frequency that is at least as fine as daily, or original if finer
        helper_index_freq = 'D'
        if original_pandas_freq_code and get_freq_level(original_pandas_freq_code) is not None and get_freq_level(original_pandas_freq_code) < get_freq_level('D'):
            helper_index_freq = original_pandas_freq_code
        
        try:
            full_span_helper_index = pd.date_range(start=min_orig_time, end=max_orig_time, freq=helper_index_freq)
            if full_span_helper_index.empty:
                 # Fallback if date_range is empty (e.g., min_orig_time > max_orig_time after some processing)
                 target_datetime_index = df_align_helper.resample(target_pandas_freq_code).asfreq().index
            else:
                 target_datetime_index = pd.Series(index=full_span_helper_index).resample(target_pandas_freq_code).asfreq().index
        except Exception as e_idx_gen:
            # print(f"[_align_single_dataframe] Error generating target_datetime_index: {e_idx_gen}. Falling back for index generation.")
            target_datetime_index = df_align_helper.resample(target_pandas_freq_code).asfreq().index

        if target_datetime_index.empty and not df_align_helper.empty : # If target index is empty but original data was not
             # print(f"DEBUG_V2 [{time_col_name}]: Target datetime index is empty despite non-empty source. Using resampled index from source itself.")
             target_datetime_index = df_align_helper.resample(target_pandas_freq_code).asfreq().index # One last attempt

        if target_datetime_index.empty : # If still empty, cannot proceed meaningfully
            # print(f"DEBUG_V2 [{time_col_name}]: CRITICAL: target_datetime_index is EMPTY. Cannot proceed.")
            return pd.DataFrame(columns=df_align_helper.columns)
        
        # print(f"DEBUG_V2 [{time_col_name}]: Generated target_datetime_index (first 5): {target_datetime_index[:5]}")


        aligned_data_all_cols = {}

        for col_name_iter in df_align_helper.columns:
            # <<< DEBUG >>> Only trace the specific column
            is_debug_column_iter = (col_name_iter == debug_col_name)

            # <<< NEW DEBUG V3 >>> Print is_high_to_low AT THE START OF THE COLUMN LOOP for the debug column
            if is_debug_column_iter:
                # print(f"DEBUG_V3 [{time_col_name}][{col_name_iter}]: COL_LOOP_START is_high_to_low: {is_high_to_low} (OrigFreqLvl: {get_freq_level(original_pandas_freq_code)}, TargetFreqLvl: {get_freq_level(target_pandas_freq_code)})")
                pass

            if is_debug_column_iter:
                # print(f"DEBUG_V2 [{time_col_name}][{col_name_iter}]: --- Processing column: {col_name_iter} ---")
                pass

            series_orig_col = df_align_helper[col_name_iter].dropna()
            if is_debug_column_iter:
                # print(f"DEBUG_V2 [{time_col_name}][{col_name_iter}]: Original series data for '{col_name_iter}' (dropna, head 5):\\n{series_orig_col.head()}")
                pass

            col_aligned_series_output = pd.Series(index=target_datetime_index, name=col_name_iter, dtype=series_orig_col.dtype if not series_orig_col.empty else np.float64)

            # <<< VALUE ALIGNMENT LOGIC PER COLUMN >>>
            # This is already inside "if alignment_mode == 'value_align' and is_high_to_low"
            # So no need to check alignment_mode again, just proceed with high-to-low logic.
            
            # CRITICAL PRE-CHECK FOR DEBUG COLUMN INSIDE VALUE_ALIGN BLOCK
            if is_debug_column_iter:
                # print(f"DEBUG_V3 [{time_col_name}][{col_name_iter}]: VALUE_ALIGN_PRE_BRANCH is_high_to_low: {is_high_to_low}")
                pass

            # This 'is_high_to_low' is the function-level one, which we are already inside the branch for.
            if is_debug_column_iter:
                # print(f"DEBUG_V3 [{time_col_name}][{col_name_iter}]: BRANCH: ENTERING High-to-Low (PUSH/ASSIGN) value alignment logic.")
                pass
                    
            # First pass: Handle potential pushes from T to T-1
            if is_debug_column_iter:
                # This print can be too verbose for many columns, only print overall summary later
                pass # # print(f"DEBUG_V2 [{time_col_name}][{col_name_iter}]: --- Starting PUSH PASS ---")
            original_series_resampler_for_push = series_orig_col.resample(target_pandas_freq_code)

            for idx_T, current_target_ts_T in enumerate(target_datetime_index):
                if idx_T == 0: 
                    continue

                prev_target_ts_T_minus_1 = target_datetime_index[idx_T - 1]
                if is_debug_column_iter:
                    # print(f"DEBUG_V2 [{time_col_name}][{col_name_iter}][PUSH]: Processing for T-1 ({prev_target_ts_T_minus_1}) by checking T ({current_target_ts_T})")
                    # print(f"DEBUG_V2 [{time_col_name}][{col_name_iter}][PUSH]: Value in col_aligned_series_output for T-1 ({prev_target_ts_T_minus_1}) BEFORE push: {col_aligned_series_output.get(prev_target_ts_T_minus_1)}")
                    pass

                try:
                    original_data_in_T_minus_1_window = original_series_resampler_for_push.get_group(prev_target_ts_T_minus_1).dropna()
                except KeyError: 
                    original_data_in_T_minus_1_window = pd.Series(dtype=series_orig_col.dtype if not series_orig_col.empty else np.float64)
                
                is_T_minus_1_originally_empty = original_data_in_T_minus_1_window.empty
                if is_debug_column_iter:
                    # print(f"DEBUG_V2 [{time_col_name}][{col_name_iter}][PUSH]: Original data in T-1 window ({prev_target_ts_T_minus_1}): count={len(original_data_in_T_minus_1_window)}, is_empty={is_T_minus_1_originally_empty}")
                    # if not original_data_in_T_minus_1_window.empty: # print(f"DEBUG_V2 [{time_col_name}][{col_name_iter}][PUSH]: T-1 original data (head(2)): {original_data_in_T_minus_1_window.head(2)}")
                    pass

                try:
                    original_data_in_T_window = original_series_resampler_for_push.get_group(current_target_ts_T).dropna()
                except KeyError:
                    original_data_in_T_window = pd.Series(dtype=series_orig_col.dtype if not series_orig_col.empty else np.float64)
                
                has_T_original_values = not original_data_in_T_window.empty
                if is_debug_column_iter:
                    # print(f"DEBUG_V2 [{time_col_name}][{col_name_iter}][PUSH]: Original data in T window ({current_target_ts_T}): count={len(original_data_in_T_window)}, has_values={has_T_original_values}")
                    # if not original_data_in_T_window.empty: # print(f"DEBUG_V2 [{time_col_name}][{col_name_iter}][PUSH]: T original data (head(2)): {original_data_in_T_window.head(2)}")
                    pass


                # 修改后的借调逻辑：只有当本周有2个以上值且上周全为空值时才执行借调
                if is_T_minus_1_originally_empty and has_T_original_values and len(original_data_in_T_window) >= 2:
                    value_to_push = original_data_in_T_window.iloc[0]
                    col_aligned_series_output.loc[prev_target_ts_T_minus_1] = value_to_push
                    if is_debug_column_iter: # Only print for the debug column
                        # print(f"DEBUG_V2 [{time_col_name}][{col_name_iter}][PUSH]: PUSHED {value_to_push} from T ({current_target_ts_T.date()}) to T-1 ({prev_target_ts_T_minus_1.date()})")
                        pass
                    # Event logging for push can be kept if needed for detailed report analysis later
                    push_event = {
                        "column": col_name_iter, "time": prev_target_ts_T_minus_1.strftime('%Y-%m-%d %H:%M:%S'), 
                        "event_type": "value_align_pushed_to_prev_period", 
                        "reason": f"从 {current_target_ts_T.strftime('%Y-%m-%d')} 借调首个有效值 {value_to_push} 到前一目标期 {prev_target_ts_T_minus_1.strftime('%Y-%m-%d')} 因为本周有{len(original_data_in_T_window)}个值且上周全为空。",
                        "pushed_value": value_to_push,
                        "from_source_period": current_target_ts_T.strftime('%Y-%m-%d %H:%M:%S')
                    }
                    temp_events_push.append(push_event)
                
            if is_debug_column_iter:
                # print(f"DEBUG_V2 [{time_col_name}][{col_name_iter}]: '{col_name_iter}' non-NaN count after PUSH PASS: {col_aligned_series_output.notna().sum()}")
                pass

            # Second pass: Determine values for each T
            # This pass iterates through each target timestamp again to assign final values.
            if is_debug_column_iter:
                # print(f"DEBUG_V2 [{time_col_name}][{col_name_iter}]: --- Starting ASSIGNMENT PASS (Corrected Loop) ---")
                pass

            # <<<< CORRECTED LOGIC: Loop for ASSIGNMENT PASS >>>>
            for idx_assign_T, current_assign_target_ts_T in enumerate(target_datetime_index):
                assign_reason = "未定义赋值原因" # Default reason
                value_for_T_output = np.nan # Default to NaN

                # Get original data for the current target window T (for assignment)
                try:
                    # Use original_series_resampler_for_push as it's already defined and resamples the original series
                    current_T_original_data = original_series_resampler_for_push.get_group(current_assign_target_ts_T).dropna()
                except KeyError:
                    current_T_original_data = pd.Series(dtype=series_orig_col.dtype if not series_orig_col.empty else np.float64)
                
                num_points_in_T_original = len(current_T_original_data)

                if is_debug_column_iter:
                    # print(f"DEBUG_V2 [{time_col_name}][{col_name_iter}][ASSIGN_LOOP]: Processing T_assign ({current_assign_target_ts_T}), Original points in window: {num_points_in_T_original}")
                    # if not current_T_original_data.empty:
                        # print(f"DEBUG_V2 [{time_col_name}][{col_name_iter}][ASSIGN_LOOP]: T_assign original data (head):\\n{current_T_original_data.head()}")
                    pass


                # Check if a value was already set by the PUSH pass
                if pd.notna(col_aligned_series_output.loc[current_assign_target_ts_T]):
                    value_for_T_output = col_aligned_series_output.loc[current_assign_target_ts_T]
                    assign_reason = "来自前期Push Pass的值被保留"
                    if is_debug_column_iter:
                        # print(f"DEBUG_V2 [{time_col_name}][{col_name_iter}][ASSIGN_LOOP]: T_assign ({current_assign_target_ts_T}) already has value (from push pass): {value_for_T_output}. Keeping it.")
                        pass
                elif num_points_in_T_original == 1:
                    value_for_T_output = current_T_original_data.iloc[0]
                    assign_reason = f"T_assign ({current_assign_target_ts_T.strftime('%Y-%m-%d')}) 原始周期内仅有一个数据点"
                    if is_debug_column_iter:
                        # print(f"DEBUG_V2 [{time_col_name}][{col_name_iter}][ASSIGN_LOOP]: T_assign ({current_assign_target_ts_T}) has 1 original point. Assigning: {value_for_T_output}")
                        pass
                elif num_points_in_T_original >= 2:
                    if is_debug_column_iter:
                        # print(f"DEBUG_V2 [{time_col_name}][{col_name_iter}][ASSIGN_LOOP]: T_assign ({current_assign_target_ts_T}) has >=2 original points. Checking T_assign-1's final state.")
                        pass
                    
                    final_T_minus_1_value_for_assign = np.nan
                    if idx_assign_T > 0:
                        prev_assign_target_ts_T_minus_1 = target_datetime_index[idx_assign_T-1]
                        # Get the value that was decided for T-1 in *this* assignment pass (or from PUSH if T-1 was not empty before this pass)
                        final_T_minus_1_value_for_assign = col_aligned_series_output.loc[prev_assign_target_ts_T_minus_1]
                        if is_debug_column_iter:
                            # print(f"DEBUG_V2 [{time_col_name}][{col_name_iter}][ASSIGN_LOOP]: T_assign-1 ({prev_assign_target_ts_T_minus_1}) final output value for assign logic: {final_T_minus_1_value_for_assign}")
                            pass
                    else: 
                        if is_debug_column_iter:
                            # print(f"DEBUG_V2 [{time_col_name}][{col_name_iter}][ASSIGN_LOOP]: At first target period (idx_assign_T=0), so no T_assign-1 to check final output for.")
                            pass
                    
                    if pd.notna(final_T_minus_1_value_for_assign):
                        value_for_T_output = current_T_original_data.iloc[-1] 
                        assign_reason = f"T_assign-1 ({target_datetime_index[idx_assign_T-1].strftime('%Y-%m-%d') if idx_assign_T > 0 else 'N/A'}) 有值, T_assign ({current_assign_target_ts_T.strftime('%Y-%m-%d')}) 取其原始周期内最后一个值"
                        if is_debug_column_iter:
                            # print(f"DEBUG_V2 [{time_col_name}][{col_name_iter}][ASSIGN_LOOP]: T_assign-1 had final value {final_T_minus_1_value_for_assign}. Assigning T_assign's LAST original value: {value_for_T_output}")
                            pass
                    else: 
                        value_for_T_output = current_T_original_data.iloc[0] 
                        assign_reason = f"T_assign-1 ({target_datetime_index[idx_assign_T-1].strftime('%Y-%m-%d') if idx_assign_T > 0 else 'N/A'}) 为空, T_assign ({current_assign_target_ts_T.strftime('%Y-%m-%d')}) 取其原始周期内第一个值"
                        if is_debug_column_iter:
                            # print(f"DEBUG_V2 [{time_col_name}][{col_name_iter}][ASSIGN_LOOP]: T_assign-1 was finally empty. Assigning T_assign's FIRST original value: {value_for_T_output}")
                            pass
                else: # num_points_in_T_original == 0 and not pd.notna(col_aligned_series_output.loc[current_assign_target_ts_T])
                    if is_debug_column_iter:
                        # print(f"DEBUG_V2 [{time_col_name}][{col_name_iter}][ASSIGN_LOOP]: T_assign ({current_assign_target_ts_T}) has 0 original points and no value from PUSH. Remains NaN.")
                        pass
                    assign_reason = f"T_assign ({current_assign_target_ts_T.strftime('%Y-%m-%d')}) 原始周期内无数据点且无Push值"
                
                # Assign the determined value to the output series for the current assignment timestamp
                col_aligned_series_output.loc[current_assign_target_ts_T] = value_for_T_output
                
                if pd.notna(value_for_T_output): # Log event only if a value was actually assigned or confirmed
                    # Check if this is a new assignment or just confirming a PUSHED value
                    is_newly_assigned_in_assign_pass = (assign_reason != "来自前期Push Pass的值被保留")

                    assign_event = {
                        "column": col_name_iter, "time": current_assign_target_ts_T.strftime('%Y-%m-%d %H:%M:%S'),
                        "event_type": "value_align_assigned_value",
                        "reason": assign_reason,
                        "assigned_value": value_for_T_output,
                        "num_original_points_in_T_window": num_points_in_T_original # num_points_in_T_original is now correctly scoped
                    }
                    temp_events_assign.append(assign_event)
                    if is_debug_column_iter and is_newly_assigned_in_assign_pass:
                        # print(f"DEBUG_V2 [{time_col_name}][{col_name_iter}][ASSIGN_LOOP]: Assigned {value_for_T_output} to T_assign ({current_assign_target_ts_T.date()}). Reason: {assign_reason}")
                        pass
            # <<<< END OF CORRECTED ASSIGNMENT LOOP >>>>
            
            if is_debug_column_iter:
                # print(f"DEBUG_V2 [{time_col_name}][{col_name_iter}]: '{col_name_iter}' non-NaN count after ASSIGNMENT PASS: {col_aligned_series_output.notna().sum()}")
                pass

            aligned_data_all_cols[col_name_iter] = col_aligned_series_output
            if is_debug_column_iter:
                # print(f"DEBUG_V2 [{time_col_name}][{col_name_iter}]: --- Finished column {col_name_iter} ---")
                # print(f"DEBUG_V2 [{time_col_name}][{col_name_iter}]: Final col_aligned_series_output for '{col_name_iter}' (head 5):\\n{col_aligned_series_output.dropna().head()}")
                pass

        # End of per-column loop for high-to-low value alignment

        # Consolidate events collected during per-column processing
        actual_events_for_report.extend(temp_events_push)
        actual_events_for_report.extend(temp_events_assign)

        if aligned_data_all_cols:
            df_final_resampled = pd.DataFrame(aligned_data_all_cols) # Automatically uses target_datetime_index from series
            if not df_final_resampled.index.equals(target_datetime_index):
                 # print("[WARN] Index of concatenated DataFrame does not match target_datetime_index. Reindexing.")
                 df_final_resampled = df_final_resampled.reindex(target_datetime_index)
        else:
            df_final_resampled = pd.DataFrame(index=target_datetime_index, columns=df_align_helper.columns)
        
        if not df_final_resampled.empty: # Ensure reset_index is called on non-empty or appropriately constructed empty df
            df_final_resampled = df_final_resampled.reset_index()
            new_time_col_name_val_align = df_final_resampled.columns[0] # Potentially 'index'
            if new_time_col_name_val_align != time_col_name:
                df_final_resampled = df_final_resampled.rename(columns={new_time_col_name_val_align: time_col_name})
        else: # If df_final_resampled became empty (e.g. target_datetime_index was empty)
            # Ensure it has the time_col_name, even if empty
            empty_cols = list(df_align_helper.columns)
            if time_col_name not in empty_cols and 'index' in df_final_resampled.columns: # after reset_index if it happened
                df_final_resampled = df_final_resampled.rename(columns={'index': time_col_name})
            elif time_col_name not in df_final_resampled.columns : # if it's truly empty df with no time_col
                df_final_resampled[time_col_name] = pd.Series(dtype='datetime64[ns]')


        # <<< FINAL DEBUG >>>
        if df_final_resampled is not None and debug_col_name in df_final_resampled.columns:
            # print(f"DEBUG_V2 [{time_col_name}]: FINAL '{debug_col_name}' non-NaN count in df_final_resampled (custom high-low): {df_final_resampled[debug_col_name].notna().sum()}")
            pass
        elif df_final_resampled is not None:
            # print(f"DEBUG_V2 [{time_col_name}]: FINAL df_final_resampled (custom high-low) is NOT empty, but '{debug_col_name}' is NOT in its columns.")
            pass
        else:
            # print(f"DEBUG_V2 [{time_col_name}]: FINAL df_final_resampled (custom high-low) is None or empty.")
            pass
        
        # print(f"DEBUG_V2 [{time_col_name}]: --- Exiting _align_single_dataframe (custom high-low path) --- Event report items generated: {len(actual_events_for_report)}")
        return df_final_resampled, actual_events_for_report

    elif alignment_mode == 'value_align' and not is_high_to_low:
        # print(f"DEBUG_V2 [{time_col_name}]: Path: Value Alignment (Same/Low-to-High Freq) using .asfreq()")
        if debug_col_name in df_align_helper.columns:
            # print(f"DEBUG_V2 [{time_col_name}]: Before .asfreq() for '{debug_col_name}', non-NaN count: {df_align_helper[debug_col_name].notna().sum()}")
            pass
        
        if not df_align_helper.empty:
            df_final_resampled = df_align_helper.resample(target_pandas_freq_code).asfreq()
        else: # df_align_helper is empty
            # Try to create an empty df with the target index structure based on original df_input's span
            target_idx_empty_asfreq = pd.DatetimeIndex([], name=time_col_name)
            if not df_input.empty and time_col_name in df_input.columns: # Use df_input (pre-indexing)
                min_orig_time_asfreq = df_input[time_col_name].min()
                max_orig_time_asfreq = df_input[time_col_name].max()
                if pd.notna(min_orig_time_asfreq) and pd.notna(max_orig_time_asfreq):
                    helper_freq_for_idx_asfreq = original_pandas_freq_code if original_pandas_freq_code and get_freq_level(original_pandas_freq_code) is not None else 'D'
                    try:
                        temp_idx_asfreq = pd.date_range(start=min_orig_time_asfreq, end=max_orig_time_asfreq, freq=helper_freq_for_idx_asfreq)
                        if not temp_idx_asfreq.empty:
                            target_idx_empty_asfreq = pd.Series(index=temp_idx_asfreq).resample(target_pandas_freq_code).asfreq().index
                    except Exception as e_idx_asfreq_empty:
                        # print(f"[_align_single_dataframe] Error generating target index for empty asfreq case: {e_idx_asfreq_empty}")
                        pass # Keep target_idx_empty_asfreq as empty
            df_final_resampled = pd.DataFrame(index=target_idx_empty_asfreq, columns=df_align_helper.columns) # Use columns from (empty) df_align_helper
        
        df_final_resampled = df_final_resampled.reset_index()
        if df_final_resampled is not None and debug_col_name in df_final_resampled.columns:
            # print(f"DEBUG_V2 [{time_col_name}]: After .asfreq() for '{debug_col_name}', non-NaN count: {df_final_resampled[debug_col_name].notna().sum()}")
            pass

    else: # alignment_mode == 'stat_align'
        # print(f"DEBUG_V2 [{time_col_name}]: Path: Statistical Alignment, method: '{selected_stat_agg_method_code}'")
        if df_align_helper is not None and debug_col_name in df_align_helper.columns:
             # print(f"DEBUG_V2 [{time_col_name}]: Before stat_align for '{debug_col_name}', non-NaN count: {df_align_helper[debug_col_name].notna().sum()}")
             pass
        
        if df_align_helper.empty:
            # print(f"DEBUG_V2 [{time_col_name}]: StatAlign Path: df_align_helper is empty. Creating empty DF with target index.")
            target_idx_empty_stat = pd.DatetimeIndex([], name=time_col_name) # Ensure name is set for consistency
            original_cols_for_empty_stat = df_align_helper.columns if df_align_helper is not None else df_input.columns
            if not df_input.empty and time_col_name in df_input.columns:
                min_orig_time_stat = df_input[time_col_name].min()
                max_orig_time_stat = df_input[time_col_name].max()
                if pd.notna(min_orig_time_stat) and pd.notna(max_orig_time_stat):
                    helper_freq_for_idx_stat = original_pandas_freq_code if original_pandas_freq_code and get_freq_level(original_pandas_freq_code) is not None else 'D'
                    try:
                        temp_idx_stat = pd.date_range(start=min_orig_time_stat, end=max_orig_time_stat, freq=helper_freq_for_idx_stat)
                        if not temp_idx_stat.empty:
                            target_idx_empty_stat = pd.Series(index=temp_idx_stat).resample(target_pandas_freq_code).asfreq().index
                    except Exception as e_idx_stat_empty:
                        # print(f"[_align_single_dataframe] Error generating target index for empty stat_align case: {e_idx_stat_empty}")
                        pass # Keep target_idx_empty_stat as empty
            df_final_resampled = pd.DataFrame(index=target_idx_empty_stat, columns=original_cols_for_empty_stat)

        elif selected_stat_agg_method_code:
            try:
                # print(f"DEBUG_V2 [{time_col_name}]: Attempting stat align with method: {selected_stat_agg_method_code}")
                numeric_cols = df_align_helper.select_dtypes(include=np.number).columns.tolist()
                non_numeric_cols = df_align_helper.select_dtypes(exclude=np.number).columns.tolist()
                
                df_resampled_numeric = pd.DataFrame()
                df_resampled_non_numeric = pd.DataFrame()

                if numeric_cols:
                    df_resampled_numeric = df_align_helper[numeric_cols].resample(target_pandas_freq_code).agg(selected_stat_agg_method_code)
                
                if non_numeric_cols:
                    # print(f"DEBUG_V2 [{time_col_name}]: Handling non-numeric columns ({non_numeric_cols}) for stat_align.")
                    if selected_stat_agg_method_code in ['first', 'last', 'min', 'max', 'count']: # Methods that might work for non-numeric
                        try:
                            df_resampled_non_numeric = df_align_helper[non_numeric_cols].resample(target_pandas_freq_code).agg(selected_stat_agg_method_code)
                        except Exception as e_non_num_agg:
                            # print(f"DEBUG_V2 [{time_col_name}]: Aggregation '{selected_stat_agg_method_code}' failed for non-numeric {non_numeric_cols}: {e_non_num_agg}. Falling back to .asfreq() for them.")
                            df_resampled_non_numeric = df_align_helper[non_numeric_cols].resample(target_pandas_freq_code).asfreq()
                            event_non_numeric_asfreq = {
                                "column": ", ".join(non_numeric_cols) if non_numeric_cols else "Non-numeric columns",
                                "time": "N/A",
                                "event_type": "stat_align_non_numeric_agg_fallback_asfreq",
                                "reason": f"对非数值列尝试统计方法 '{selected_stat_agg_method_code}' 失败，已回退到 .asfreq()。",
                                "original_method_attempted": selected_stat_agg_method_code,
                                "error_message": str(e_non_num_agg)
                            }
                            actual_events_for_report.append(event_non_numeric_asfreq)
                    else: # Default to asfreq for non-numeric if agg method isn't suitable (e.g. sum, mean)
                        df_resampled_non_numeric = df_align_helper[non_numeric_cols].resample(target_pandas_freq_code).asfreq()
                        event_non_numeric_asfreq = {
                            "column": ", ".join(non_numeric_cols) if non_numeric_cols else "Non-numeric columns",
                            "time": "N/A",
                            "event_type": "stat_align_non_numeric_fallback_asfreq",
                            "reason": f"非数值列使用 .asfreq() 进行对齐，因为统计方法 '{selected_stat_agg_method_code}' 不适用。",
                            "original_method_attempted": selected_stat_agg_method_code
                        }
                        actual_events_for_report.append(event_non_numeric_asfreq)
                
                if not df_resampled_numeric.empty and not df_resampled_non_numeric.empty:
                    df_final_resampled = pd.concat([df_resampled_numeric, df_resampled_non_numeric], axis=1).reindex(df_resampled_numeric.index.union(df_resampled_non_numeric.index))
                elif not df_resampled_numeric.empty:
                    df_final_resampled = df_resampled_numeric
                elif not df_resampled_non_numeric.empty:
                    df_final_resampled = df_resampled_non_numeric
                else: # Both empty, or only non-numeric columns failed, or no columns.
                    # Create an empty df with the target index
                    temp_target_index_stat = df_align_helper.resample(target_pandas_freq_code).asfreq().index # Get target index
                    df_final_resampled = pd.DataFrame(index=temp_target_index_stat, columns=df_align_helper.columns)

            except Exception as e_stat:
                # print(f"DEBUG_V2 [{time_col_name}]: Stat align with method '{selected_stat_agg_method_code}' FAILED: {e_stat}. Falling back to .asfreq()")
                df_final_resampled = df_align_helper.resample(target_pandas_freq_code).asfreq()
                event_stat_fallback = {
                    "column": "N/A", "time": "N/A",
                    "event_type": "stat_align_method_fallback_asfreq",
                    "reason": f"使用方法 '{selected_stat_agg_method_code}' 进行统计对齐失败 ({e_stat})。已回退到 .asfreq()。",
                    "original_method_attempted": selected_stat_agg_method_code,
                    "error_message": str(e_stat)
                }
                actual_events_for_report.append(event_stat_fallback)
        else:
            # print(f"DEBUG_V2 [{time_col_name}]: No stat_agg_method provided. Using .asfreq() for stat_align.")
            df_final_resampled = df_align_helper.resample(target_pandas_freq_code).asfreq()
            event_no_method = {
                "column": "N/A", "time": "N/A",
                "event_type": "stat_align_no_method_fallback_asfreq",
                "reason": "未提供统计聚合方法，已默认使用 .asfreq()。",
            }
            actual_events_for_report.append(event_no_method)
        
        df_final_resampled = df_final_resampled.reset_index()
        if df_final_resampled is not None and debug_col_name in df_final_resampled.columns:
            # print(f"DEBUG_V2 [{time_col_name}]: After stat_align for '{debug_col_name}', non-NaN count: {df_final_resampled[debug_col_name].notna().sum()}")
            pass

    # <<< FINAL DEBUG and Common Return Path for asfreq and stat_align paths >>>
    # Ensure time column name is correct after reset_index for asfreq/stat_align paths
    if alignment_mode != 'value_align' or not is_high_to_low: # If not the custom high-to-low path that returns early
        if df_final_resampled is not None and not df_final_resampled.empty:
            # After reset_index(), the time column is usually named 'index' or its original name if already index
            # We need to ensure it's time_col_name
            current_idx_name_after_reset = df_final_resampled.columns[0]
            if current_idx_name_after_reset != time_col_name:
                 df_final_resampled = df_final_resampled.rename(columns={current_idx_name_after_reset: time_col_name})
        elif df_final_resampled is not None and df_final_resampled.empty: # Handle empty DataFrame ensuring time col name
            if time_col_name not in df_final_resampled.columns and 'index' in df_final_resampled.columns:
                 df_final_resampled = df_final_resampled.rename(columns={'index': time_col_name})
            elif time_col_name not in df_final_resampled.columns:
                 df_final_resampled[time_col_name] = pd.Series(dtype='datetime64[ns]')


    if df_final_resampled is not None and debug_col_name in df_final_resampled.columns:
        # print(f"DEBUG_V2 [{time_col_name}]: FINAL (common path) '{debug_col_name}' non-NaN count in df_final_resampled: {df_final_resampled[debug_col_name].notna().sum()}")
        pass
    elif df_final_resampled is not None:
        # print(f"DEBUG_V2 [{time_col_name}]: FINAL (common path) df_final_resampled is NOT empty, but '{debug_col_name}' is NOT in its columns. Columns: {df_final_resampled.columns.tolist()}")
        pass
    else:
        # print(f"DEBUG_V2 [{time_col_name}]: FINAL (common path) df_final_resampled is None or empty.")
        pass
        
    # print(f"DEBUG_V2 [{time_col_name}]: --- Exiting _align_single_dataframe (common path) --- Event report items generated: {len(actual_events_for_report)}")
    
    # <<< CRITICAL STEP for column name consistency >>>
    if df_final_resampled is not None and not df_final_resampled.empty:
        original_data_cols = [col for col in df_input.columns if col != time_col_name and col in df_input.columns]
        # If df_input's time_col_name was its index, df_input.columns might not contain it.
        # In that case, original_data_cols would be all columns of df_input.
        # Let's refine original_data_cols based on df_align_helper which has time_col as index before this stage.
        if df_align_helper is not None: # df_align_helper has time_col as index
            original_data_cols_from_helper = df_align_helper.columns.tolist()
            # Check if the number of data columns matches.
            # df_final_resampled has 'time_col_name' + data_cols
            # original_data_cols_from_helper has only data_cols
            if len(df_final_resampled.columns) -1 == len(original_data_cols_from_helper):
                current_data_cols_in_final = [col for col in df_final_resampled.columns if col != time_col_name]
                if len(current_data_cols_in_final) == len(original_data_cols_from_helper):
                    rename_map = {old_col: new_col for old_col, new_col in zip(current_data_cols_in_final, original_data_cols_from_helper)}
                    df_final_resampled = df_final_resampled.rename(columns=rename_map)
                    # print(f"DEBUG_V2 [{time_col_name}]: Ensured data column names match original input. Rename map: {rename_map}")
    
    return df_final_resampled, actual_events_for_report


def perform_frequency_alignment(
    df_to_align: pd.DataFrame,
    time_col_name: str,
    original_pandas_freq_code: str | None, # NEW: Original frequency
    target_pandas_freq_code: str,
    alignment_mode: str,  # NEW: 'stat_align' or 'value_align'
    selected_stat_agg_method_code: str | None, # MODIFIED: for stat_align mode
    df_full_to_align: pd.DataFrame | None = None,
    df_before_align_for_stats: pd.DataFrame | None = None, 
    time_col_info_for_stats: dict | None = None, 
    selected_align_freq_display: str | None = None, 
    selected_agg_method_display: str | None = None,
    complete_time_index: bool = False  # 新增：是否启用频率补全
) -> dict:
    """
    Main function to perform frequency alignment on a DataFrame.
    Can handle a main DataFrame and an optional 'FULL' version.
    It uses a helper _align_single_dataframe for the core resampling logic.
    Returns a dictionary with aligned data, status, and report items.
    """
    # --- Input Type Validation ---
    if not isinstance(df_to_align, pd.DataFrame):
        error_msg = f"主输入数据 (df_to_align) 不是有效的DataFrame，而是 {type(df_to_align)} 类型。频率对齐中止。"
        print(f"[FREQ_ALIGN_ERROR] {error_msg}")
        return {
            "aligned_df": None, "aligned_df_full": None,
            "status_message": error_msg, "status_type": "error",
            "alignment_report_items": [{"column": "ALL", "reason": error_msg, "event_type": "input_error"}],
            "new_time_col_info": None
        }
    if df_full_to_align is not None and not isinstance(df_full_to_align, pd.DataFrame):
        error_msg = f"完整输入数据 (df_full_to_align) 不是有效的DataFrame，而是 {type(df_full_to_align)} 类型。频率对齐中止。"
        print(f"[FREQ_ALIGN_ERROR] {error_msg}")
        # Return df_to_align as is, if it was valid, or None if it also had issues (covered by above)
        return {
            "aligned_df": df_to_align if isinstance(df_to_align, pd.DataFrame) else None, 
            "aligned_df_full": None,
            "status_message": error_msg, "status_type": "error",
            "alignment_report_items": [{"column": "ALL", "reason": error_msg, "event_type": "input_error"}],
            "new_time_col_info": None
        }
    # --- End Input Type Validation ---

    print(f"DEBUG_PERFORM_FREQ_ALIGN: Entering with time_col='{time_col_name}', orig_freq='{original_pandas_freq_code}', target_freq='{target_pandas_freq_code}', mode='{alignment_mode}', agg_method='{selected_stat_agg_method_code}'")
    if df_to_align is not None:
        print(f"DEBUG_PERFORM_FREQ_ALIGN: df_to_align shape: {df_to_align.shape if hasattr(df_to_align, 'shape') else 'N/A'}")

    result = {
        'aligned_df': None,
        'aligned_df_full': None, # Initialize to None, will be set if df_full_to_align is processed
        'alignment_report_items': [],
        'status_message': '',
        'status_type': 'info',
        'new_time_col_info': None
    }
    
    # <<< FIX: Define debug_col_name here >>>
    debug_col_name = "中国:生产率:焦炉:国内独立焦化厂(230家)"

    # If df_full_to_align is None, but df_to_align is not, copy df_to_align structure for full if needed
    # This ensures aligned_df_full is somewhat consistent if only main df is passed.
    # However, typical use is df_full_to_align is the more complete version.
    if df_full_to_align is None and df_to_align is not None:
        result['aligned_df_full'] = df_to_align.copy() # Or an empty version of it
    elif df_full_to_align is not None:
        result['aligned_df_full'] = df_full_to_align.copy()


    current_events_for_report_main_call = [] # Events specifically from the main df alignment call

    try:
        # --- Align df_to_align (main DataFrame) ---
        # print(f"[FreqAlignLogic] Aligning main DataFrame. Mode: {alignment_mode}, Orig Freq: {original_pandas_freq_code}, Target Freq: {target_pandas_freq_code}, StatAgg: {selected_stat_agg_method_code}")
        # print(f"DEBUG_V2 [perform_frequency_alignment]: Initial non-NaN in df_to_align for '{debug_col_name}': {df_to_align[debug_col_name].notna().sum() if df_to_align is not None and debug_col_name in df_to_align.columns else 'N/A'}")
        
        aligned_main_df, events_from_main_align = _align_single_dataframe(
            df_input=df_to_align,
            time_col_name=time_col_name,
            original_pandas_freq_code=original_pandas_freq_code,
            target_pandas_freq_code=target_pandas_freq_code,
            alignment_mode=alignment_mode,
            selected_stat_agg_method_code=selected_stat_agg_method_code,
            parent_function_events_for_report=current_events_for_report_main_call # This list is not used by child, child returns its own
        )
        result['aligned_df'] = aligned_main_df
        if events_from_main_align: # Add events returned from the helper
            result['alignment_report_items'].extend(events_from_main_align)
            current_events_for_report_main_call.extend(events_from_main_align) # Keep a local copy too for the generic report section later
            
        # print(f"DEBUG_V2 [perform_frequency_alignment]: After _align_single_dataframe for main, aligned_main_df non-NaN for '{debug_col_name}': {aligned_main_df[debug_col_name].notna().sum() if aligned_main_df is not None and debug_col_name in aligned_main_df.columns else 'N/A'}")
        # print(f"DEBUG_V2 [perform_frequency_alignment]: current_events_for_report_main_call count after main alignment: {len(current_events_for_report_main_call)}")


        # --- Align df_full_to_align (backup/FULL DataFrame) if provided ---
        if df_full_to_align is not None:
            # print(f"[FreqAlignLogic] Aligning FULL DataFrame. Mode: {alignment_mode}, Orig Freq: {original_pandas_freq_code}, Target Freq: {target_pandas_freq_code}, StatAgg: {selected_stat_agg_method_code}")
            # original_pandas_freq_code for FULL df might be different if it's truly independent.
            # For now, assume it's aligned based on the same original_pandas_freq_code context from UI.
            # If FULL df has its own time_col_info, that should be used for its original_freq.
            # This simplification assumes the 'original_pandas_freq_code' applies to both.
            # full_df_events_for_report = [] # Separate list for full_df events - not needed if using returned events
            aligned_full_df, events_from_full_align = _align_single_dataframe(
                df_input=df_full_to_align,
                time_col_name=time_col_name,
                original_pandas_freq_code=original_pandas_freq_code, # Assuming same context
                target_pandas_freq_code=target_pandas_freq_code,
                alignment_mode=alignment_mode,
                selected_stat_agg_method_code=selected_stat_agg_method_code,
                parent_function_events_for_report=[] # Pass an empty list, not used by child
            )
            result['aligned_df_full'] = aligned_full_df
            # Note: events from aligning df_full are not currently added to the main report by default.
            # This could be changed if needed: result['alignment_report_items'].extend(events_from_full_align)
            # To include them:
            if events_from_full_align:
                 result['alignment_report_items'].extend(events_from_full_align)
        elif aligned_main_df is not None : # df_full_to_align was None, mirror structure from aligned_main_df
             result['aligned_df_full'] = aligned_main_df.copy()


        # --- Post-alignment checks and status ---
        if result['aligned_df'] is None or result['aligned_df'].empty:
            result['status_message'] = "数据对齐后为空或操作失败。"
            result['status_type'] = 'error'
            # Check if any specific error was already reported by _align_single_dataframe
            has_specific_error = any(event.get('event_type', '').startswith('stat_align_') and ('fallback' in event.get('event_type', '') or 'error' in event.get('event_type', '')) for event in current_events_for_report_main_call)
            
            if not current_events_for_report_main_call or not has_specific_error : # If no specific custom logic error was reported that explains emptiness
                general_failure_event = {
                    "column": "N/A", "time": "N/A", "event_type": "alignment_failed_empty_output",
                    "reason": "数据对齐后为空或操作失败，未报告具体原因。"
                }
                # Avoid duplicate general failure if a more specific one exists
                if not any(e['event_type'] == general_failure_event['event_type'] for e in result['alignment_report_items']):
                    result['alignment_report_items'].append(general_failure_event)
        else:
            display_method_for_status = selected_stat_agg_method_code if alignment_mode == 'stat_align' else alignment_mode
            # Determine is_high_to_low for status message context
            base_lvl_status = get_freq_level(original_pandas_freq_code)
            target_lvl_status = get_freq_level(target_pandas_freq_code)
            is_high_to_low_for_status_msg = False
            if base_lvl_status is not None and target_lvl_status is not None:
                is_high_to_low_for_status_msg = base_lvl_status < target_lvl_status

            if alignment_mode == 'value_align' and is_high_to_low_for_status_msg:
                 display_method_for_status = "自定义值对齐 (高频->低频)"
            elif alignment_mode == 'value_align': # Not high-to-low, or freqs are incomparable
                 display_method_for_status = "值对齐 (asfreq)"
            elif alignment_mode == 'stat_align' and selected_agg_method_display:
                 display_method_for_status = selected_agg_method_display
            elif alignment_mode == 'stat_align' and selected_stat_agg_method_code:
                 display_method_for_status = f"统计对齐 ({selected_stat_agg_method_code})"


            # 如果启用了频率补全，在对齐后对数据进行补全
            if complete_time_index and result['aligned_df'] is not None and not result['aligned_df'].empty and time_col_name in result['aligned_df'].columns:
                try:
                    # 获取原始数据范围
                    df_aligned = result['aligned_df']
                    min_date = df_aligned[time_col_name].min()
                    max_date = df_aligned[time_col_name].max()
                    
                    # 生成完整时间序列
                    full_date_range = pd.date_range(start=min_date, end=max_date, freq=target_pandas_freq_code)
                    
                    # 创建带有完整时间序列的新DataFrame
                    df_complete = pd.DataFrame({time_col_name: full_date_range})
                    
                    # 与原始数据合并
                    df_aligned_complete = pd.merge(df_complete, df_aligned, on=time_col_name, how='left')
                    
                    # 更新结果
                    result['aligned_df'] = df_aligned_complete
                    
                    # 如果有FULL版本的数据，也进行同样的补全
                    if result['aligned_df_full'] is not None and not result['aligned_df_full'].empty and time_col_name in result['aligned_df_full'].columns:
                        df_full = result['aligned_df_full']
                        df_full_complete = pd.merge(df_complete, df_full, on=time_col_name, how='left')
                        result['aligned_df_full'] = df_full_complete
                    
                    # 更新状态消息，增加频率补全信息
                    result['status_message'] = f"数据已成功按 '{selected_align_freq_display or target_pandas_freq_code}' 频率使用 '{display_method_for_status}' 逻辑对齐，并已补全缺失的时间点。"
                    
                    # 添加补全信息到报告项
                    result['alignment_report_items'].append({
                        "column": "ALL",
                        "event_type": "frequency_completion",
                        "reason": f"已根据目标频率 '{target_pandas_freq_code}' 补全时间序列。原有 {len(df_aligned)} 行，补全后 {len(full_date_range)} 行。"
                    })
                except Exception as e_completion:
                    # 如果补全过程出错，记录错误但继续使用原始对齐后的数据
                    print(f"[WARNING] 频率补全过程出错: {e_completion}")
                    result['alignment_report_items'].append({
                        "column": "ALL",
                        "event_type": "frequency_completion_error",
                        "reason": f"尝试频率补全时出错: {e_completion}"
                    })
                    # 保留原始状态消息
                    result['status_message'] = f"数据已成功按 '{selected_align_freq_display or target_pandas_freq_code}' 频率使用 '{display_method_for_status}' 逻辑对齐，但频率补全过程失败。"
            else:
                # 如果没有启用频率补全或数据不满足要求，使用原始状态消息
                result['status_message'] = f"数据已成功按 '{selected_align_freq_display or target_pandas_freq_code}' 频率使用 '{display_method_for_status}' 逻辑对齐。"
            
            result['status_type'] = 'success'
            # Ensure time_col_name exists before calling identify_time_column
            if time_col_name in result['aligned_df'].columns:
                result['new_time_col_info'] = identify_time_column(result['aligned_df'], time_col_name)
            else:
                # print(f"[WARN FreqAlignLogic] Time column '{time_col_name}' not found in aligned_df. Cannot identify new time col info.")
                result['new_time_col_info'] = None # Or some default error state


        # --- Generate Standard Alignment Report Data (counts, losses) ---
        # This part compares df_before_align_for_stats with the newly aligned result['aligned_df']
        # It should largely remain compatible, but ensure event_type from custom logic is handled in UI if needed.
        if df_before_align_for_stats is not None and time_col_info_for_stats and \
           time_col_info_for_stats.get('name') and result['aligned_df'] is not None and not result['aligned_df'].empty:
            
            df_before_copy_for_report = df_before_align_for_stats.copy()
            df_aligned_copy_for_report = result['aligned_df'].copy()
            
            # <<< DETAILED REPORT DEBUG >>>
            print(f"REPORT_DEBUG: Entering report generation block. time_col_name: {time_col_name}")
            if debug_col_name in df_before_copy_for_report.columns:
                print(f"REPORT_DEBUG: df_before_copy_for_report['{debug_col_name}'].dtype: {df_before_copy_for_report[debug_col_name].dtype}")
                print(f"REPORT_DEBUG: df_before_copy_for_report['{debug_col_name}'].notna().sum() (direct): {df_before_copy_for_report[debug_col_name].notna().sum()}")
                print(f"REPORT_DEBUG: df_before_copy_for_report['{debug_col_name}'].head():\n{df_before_copy_for_report[debug_col_name].head()}")
            else:
                print(f"REPORT_DEBUG: '{debug_col_name}' not in df_before_copy_for_report.columns")

            if debug_col_name in df_aligned_copy_for_report.columns:
                print(f"REPORT_DEBUG: df_aligned_copy_for_report['{debug_col_name}'].dtype: {df_aligned_copy_for_report[debug_col_name].dtype}")
                print(f"REPORT_DEBUG: df_aligned_copy_for_report['{debug_col_name}'].notna().sum() (direct): {df_aligned_copy_for_report[debug_col_name].notna().sum()}")
                print(f"REPORT_DEBUG: df_aligned_copy_for_report['{debug_col_name}'].head():\n{df_aligned_copy_for_report[debug_col_name].head()}")
            else:
                print(f"REPORT_DEBUG: '{debug_col_name}' not in df_aligned_copy_for_report.columns")
            # <<< END DETAILED REPORT DEBUG >>>

            report_cols = [col for col in df_aligned_copy_for_report.columns if col != time_col_name]
            
            processed_cols_for_summary_report = set()

            for col_name_report_iter in report_cols:
                original_non_nan = 0
                if col_name_report_iter in df_before_copy_for_report.columns:
                    try:
                        series_before = pd.to_numeric(df_before_copy_for_report[col_name_report_iter], errors='coerce')
                        original_non_nan = series_before.notna().sum()
                        if col_name_report_iter == debug_col_name:
                            print(f"REPORT_DEBUG_LOOP [{col_name_report_iter}]: original_non_nan (after to_numeric) = {original_non_nan}")
                    except Exception as e_orig_count:
                        print(f"REPORT_DEBUG_LOOP [{col_name_report_iter}]: Error counting original_non_nan with to_numeric: {e_orig_count}. Falling back.")
                        original_non_nan = df_before_copy_for_report[col_name_report_iter].notna().sum() # Fallback
                else: # Debugging if column is missing
                    if col_name_report_iter == debug_col_name:
                        print(f"REPORT_DEBUG_LOOP [{col_name_report_iter}]: Column not found in df_before_copy_for_report for original_non_nan count.")
                
                aligned_non_nan = 0
                if col_name_report_iter in df_aligned_copy_for_report.columns:
                    try:
                        series_aligned = pd.to_numeric(df_aligned_copy_for_report[col_name_report_iter], errors='coerce')
                        aligned_non_nan = series_aligned.notna().sum()
                        if col_name_report_iter == debug_col_name:
                            print(f"REPORT_DEBUG_LOOP [{col_name_report_iter}]: aligned_non_nan (after to_numeric) = {aligned_non_nan}")
                    except Exception as e_aligned_count:
                        print(f"REPORT_DEBUG_LOOP [{col_name_report_iter}]: Error counting aligned_non_nan with to_numeric: {e_aligned_count}. Falling back.")
                        aligned_non_nan = df_aligned_copy_for_report[col_name_report_iter].notna().sum() # Fallback
                else: # Debugging if column is missing (should not happen as report_cols is from its keys)
                    if col_name_report_iter == debug_col_name:
                        print(f"REPORT_DEBUG_LOOP [{col_name_report_iter}]: Column not found in df_aligned_copy_for_report for aligned_non_nan count (UNEXPECTED).")

                item_for_report = {
                    "column": col_name_report_iter,
                    "original_non_nan": int(original_non_nan),
                    "aligned_non_nan": int(aligned_non_nan),
                    "target_freq_display": selected_align_freq_display or target_pandas_freq_code,
                    "agg_method_display": selected_agg_method_display or display_method_for_status, # Use the determined display method
                    "lost_periods": [], # Original report logic for this can be complex, TBD if needed for custom.
                    "period_loss_info_flags": {'show_detailed_lost_periods': False, 'actual_value_loss_count': 0},
                    "custom_event_info": None # Clearing old custom_event_info, detailed events are separate items now
                }
                # Add this summary item if not already covered by a more specific event type for this column
                # This logic might be too simplistic; the UI will need to handle various event types.
                # For now, let's add all summary items. The UI can filter/group.
                result['alignment_report_items'].append(item_for_report)
                processed_cols_for_summary_report.add(col_name_report_iter)

            # Add any "N/A" column events (like overall fallback) if they were not for specific cols
            for event in current_events_for_report_main_call:
                 if event.get("column") == "N/A" and event not in result['alignment_report_items']:
                     result['alignment_report_items'].append(event)
        
        # <<< NEW DEBUG PRINT FOR FINAL REPORT ITEMS >>>
        # print(f"FINAL_REPORT_ITEMS_DEBUG: About to return from perform_frequency_alignment. Full report items list for debug:")
        if debug_col_name: # debug_col_name is defined at the start of this function
            found_debug_col_in_report = False
            for item_idx, item_content in enumerate(result.get('alignment_report_items', [])):
                if item_content.get('column') == debug_col_name:
                    # We are interested in the summary item, not just custom events.
                    # The summary item typically has 'original_non_nan' and 'aligned_non_nan' keys.
                    if 'original_non_nan' in item_content and 'aligned_non_nan' in item_content:
                        print(f"FINAL_REPORT_ITEMS_DEBUG [{debug_col_name} - Item Index {item_idx} - Summary]: {item_content}")
                        found_debug_col_in_report = True
            if not found_debug_col_in_report:
                print(f"FINAL_REPORT_ITEMS_DEBUG: Debug column '{debug_col_name}' (as a summary item) not found in final report items list.")
        # <<< END NEW DEBUG PRINT >>>
        
        # Ensure no duplicate report items if custom events were already added
        # A more robust way is to have _align_single_dataframe return structured data
        # and then build the final report items here.
        # For now, the custom events are specific. The generic report items are about counts.
        # Let's filter out the generic items if a custom one for the same column was logged by _align_single_dataframe.
        # This is simplified: custom events are just added. The UI can decide how to display.

        # <<< NEW DEBUG PRINT >>> Check value in processed_df after return from _align_single_dataframe
        # Use a common debug column name, ensure it's defined if used here.
        # For now, rely on _align_single_dataframe's internal debug_col_name
        local_debug_col_name_outer = "中国:生产率:焦炉:国内独立焦化厂(230家)" # Example, ensure this is consistent if used.
        if result['aligned_df'] is not None and local_debug_col_name_outer in result['aligned_df'].columns:
            # print(f"DEBUG_V2 [perform_frequency_alignment]: Final check of result['aligned_df'] for '{local_debug_col_name_outer}' non-NaN: {result['aligned_df'][local_debug_col_name_outer].notna().sum()}")
            target_debug_date = pd.to_datetime('2020-01-31')
            # Check if time_col_name is in result['aligned_df'] and then if target_debug_date is in its values
            if time_col_name in result['aligned_df'].columns and not result['aligned_df'][time_col_name].empty:
                # Ensure target_debug_date can be compared with the datetime objects in the column
                # Handle potential timezone differences if any, though not expected here with to_datetime
                try:
                    is_present = target_debug_date in pd.to_datetime(result['aligned_df'][time_col_name]).dt.tz_localize(None).values
                except TypeError: # Handle cases where tz_localize might not be needed or applicable
                    is_present = target_debug_date in pd.to_datetime(result['aligned_df'][time_col_name]).values

                if is_present:
                    target_row_outer = result['aligned_df'][pd.to_datetime(result['aligned_df'][time_col_name]).dt.tz_localize(None) == target_debug_date]
                    if not target_row_outer.empty:
                        val_in_processed_df = target_row_outer[local_debug_col_name_outer].iloc[0]
                        # print(f"DEBUG_V2 [perform_frequency_alignment] for '{local_debug_col_name_outer}' at {target_debug_date.date()}: {val_in_processed_df}")
                    else:
                        # print(f"DEBUG_V2 [perform_frequency_alignment]: Date {target_debug_date.date()} NOT FOUND in final result['aligned_df'] (row empty after filter).")
                        pass
                else:
                    # print(f"DEBUG_V2 [perform_frequency_alignment]: Date {target_debug_date.date()} not in time column of final result['aligned_df'].")
                    pass
            else:
                if time_col_name not in result['aligned_df'].columns:
                    # print(f"DEBUG_V2 [perform_frequency_alignment]: Time column '{time_col_name}' not found in final result['aligned_df'].")
                    pass
                else: # Time column exists but is empty
                    # print(f"DEBUG_V2 [perform_frequency_alignment]: Time column '{time_col_name}' is empty in final result['aligned_df'].")
                    pass

        else:
            if result['aligned_df'] is None:
                # print(f"DEBUG_V2 [perform_frequency_alignment]: result['aligned_df'] is None before returning.")
                pass
            elif local_debug_col_name_outer not in result['aligned_df'].columns:
                 # print(f"DEBUG_V2 [perform_frequency_alignment]: Column '{local_debug_col_name_outer}' NOT FOUND in final result['aligned_df']. Columns are: {result['aligned_df'].columns.tolist() if result['aligned_df'] is not None else 'N/A'}")
                 pass

    except Exception as e:
        result['status_message'] = f"频率对齐过程中发生主错误: {e}"
        result['status_type'] = 'error'
        # print(f"[ERROR FreqAlignLogic] Exception in perform_frequency_alignment: {e}")
        import traceback
        # print(traceback.format_exc())

    return result