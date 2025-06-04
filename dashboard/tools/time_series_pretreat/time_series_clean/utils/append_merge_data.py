# append_merge_data.py
import pandas as pd
import streamlit as st
from typing import List, Dict, Optional, Tuple, Any

def append_dataframes(dataframes: List[pd.DataFrame]) -> Optional[pd.DataFrame]:
    """
    Appends a list of DataFrames vertically.
    All DataFrames must have the same columns.

    Args:
        dataframes: A list of pandas DataFrames to append.

    Returns:
        A new DataFrame with all input DataFrames appended, or None if an error occurs.
    """
    if not dataframes:
        st.error("没有提供用于追加的数据框。")
        return None
    
    # Check if all dataframes have the same columns
    first_df_columns = dataframes[0].columns
    for i, df in enumerate(dataframes[1:], 1):
        if not df.columns.equals(first_df_columns):
            st.error(f"数据框 {i+1} 的列与第一个数据框的列不匹配。请确保所有数据框具有相同的列结构以便追加。")
            return None
            
    try:
        appended_df = pd.concat(dataframes, ignore_index=True)
        # st.success("数据框成功追加！") # Removed: UI layer will handle success message
        return appended_df
    except Exception as e:
        st.error(f"追加数据框时发生错误: {e}")
        return None

def _prepare_single_df_for_merge(df_raw: pd.DataFrame, time_col_name: str) -> Tuple[Optional[pd.DataFrame], Optional[str]]:
    """Prepares a single DataFrame for smart merge: sets DatetimeIndex, infers frequency."""
    if df_raw is None or time_col_name is None:
        st.error("提供给 _prepare_single_df_for_merge 的数据或时间列名为空。")
        return None, None
    
    df = df_raw.copy()
    if time_col_name not in df.columns:
        st.error(f"时间列 '{time_col_name}' 在数据集中未找到。")
        return None, None

    try:
        df[time_col_name] = pd.to_datetime(df[time_col_name], errors='coerce')
        if df[time_col_name].isnull().any():
            st.warning(f"时间列 '{time_col_name}' 包含无法解析为日期的值，这些行将被移除。")
            df.dropna(subset=[time_col_name], inplace=True)
        
        if df.empty:
            st.error(f"处理时间列 '{time_col_name}' 后，数据集为空。")
            return None, None
            
        df = df.set_index(time_col_name)
        df = df.sort_index()
    except Exception as e:
        st.error(f"处理时间列 '{time_col_name}' 并设置为索引时出错: {e}")
        return None, None
    
    inferred_freq = None
    if isinstance(df.index, pd.DatetimeIndex):
        if len(df.index) > 1:
            inferred_freq = pd.infer_freq(df.index)
            if inferred_freq:
                st.info(f"数据集 (原时间列: {time_col_name}) 推断频率为: {inferred_freq}")
            else:
                st.warning(f"无法为数据集 (原时间列: {time_col_name}) 推断规则频率。可能是不规则时间序列。")
        else:
            st.info(f"数据集 (原时间列: {time_col_name}) 只有一个数据点，无法推断频率。")
    else:
        st.error(f"列 '{time_col_name}' 未能成功转换成 DatetimeIndex。")
        return None, None
        
    return df, inferred_freq

def smart_merge_controller(left_df_raw: pd.DataFrame, 
                           left_time_col: str, 
                           right_df_raw: pd.DataFrame, 
                           right_time_col: str) -> Optional[pd.DataFrame]:
    """Controller function for smart merging two dataframes based on time frequency."""
    st.info(f"开始处理左侧数据集 (时间列: {left_time_col})...")
    left_df_processed, left_freq = _prepare_single_df_for_merge(left_df_raw, left_time_col)
    
    st.info(f"开始处理右侧数据集 (时间列: {right_time_col})...")
    right_df_processed, right_freq = _prepare_single_df_for_merge(right_df_raw, right_time_col)

    if left_df_processed is None or right_df_processed is None:
        st.error("一个或两个数据集在预处理阶段失败，无法进行智能合并。")
        return None

    # --- Placeholder for frequency comparison and actual merge/alignment logic ---
    st.subheader("频率分析结果 (占位符)")
    st.write(f"左侧数据集频率: {left_freq if left_freq else '无法推断或不规则'}")
    st.write(f"右侧数据集频率: {right_freq if right_freq else '无法推断或不规则'}")

    # TODO: Implement frequency comparison (e.g., using pd.tseries.frequencies.to_offset(freq).nanos)
    # TODO: Implement resampling strategy based on comparison (user input needed for strategy)
    # TODO: Implement final merge strategy

    st.warning("智能合并的频率对齐和具体合并逻辑尚未完全实现。下方为基于外部联接的简单合并结果作为占位符。")
    
    # Simple outer merge for now as a placeholder
    try:
        # Ensure no column name clashes before merge, or handle them with suffixes
        # Check for overlapping columns excluding the index
        common_cols = left_df_processed.columns.intersection(right_df_processed.columns)
        if not common_cols.empty:
            st.info(f"左右数据存在共同的列名: {list(common_cols)}。合并时将自动添加后缀。")
        
        merged_df = pd.merge(left_df_processed, right_df_processed, 
                             left_index=True, right_index=True, 
                             how='outer', suffixes=('_left', '_right'))
        st.success("占位符合并完成 (外部联接)。")
        return merged_df
    except Exception as e:
        st.error(f"占位符合并操作失败: {e}")
        return None

def _internal_merge_dataframes(
    left_df: pd.DataFrame, 
    right_df: pd.DataFrame, 
    how: str = 'inner', 
    left_on: Optional[List[str]] = None, 
    right_on: Optional[List[str]] = None,
    on: Optional[List[str]] = None,
    suffixes: Tuple[str, str] = ('_x', '_y')
) -> Optional[pd.DataFrame]:
    if left_df is None or right_df is None:
        # st.error is okay here as it indicates a fundamental issue passed to the backend
        st.error("INTERNAL ERROR: 用于合并的一个或两个内部数据框为空。") 
        return None
    try:
        if on:
            merged_df = pd.merge(left_df, right_df, how=how, on=on, suffixes=suffixes)
        elif left_on and right_on:
            merged_df = pd.merge(left_df, right_df, how=how, left_on=left_on, right_on=right_on, suffixes=suffixes)
        else: # Index merge if no keys are provided
            merged_df = pd.merge(left_df, right_df, how=how, left_index=True, right_index=True, suffixes=suffixes)
        # st.success("数据框成功合并！") # UI layer should handle this
        return merged_df
    except Exception as e:
        st.error(f"内部合并数据框时发生错误: {e}") # st.error okay for unexpected exceptions
        return None

def perform_merge_and_postprocess(
    left_df_raw: pd.DataFrame,
    left_time_col: str,
    left_df_name: str, # for error messages
    right_df_raw: pd.DataFrame,
    right_time_col: str,
    right_df_name: str, # for error messages
    merge_how: str,
    merge_on_cols: Optional[List[str]],
    status_container=st # For displaying intermediate messages in the UI
) -> Tuple[Optional[pd.DataFrame], Optional[str]]:
    """Handles resampling, merging, and post-processing (index, frequency)."""

    left_df_daily = resample_df_to_daily(df=left_df_raw, 
                                           time_col=left_time_col, 
                                           df_name_for_error=left_df_name, 
                                           output_container=status_container)
    
    right_df_daily = resample_df_to_daily(df=right_df_raw, 
                                            time_col=right_time_col, 
                                            df_name_for_error=right_df_name, 
                                            output_container=status_container)

    if left_df_daily is None or right_df_daily is None:
        status_container.warning("由于一个或两个数据集未能成功重采样至日度，合并操作未执行。")
        return None, None

    # Prepare for merge (convert daily resampled DFs to have columns if using merge_on_cols)
    df1_to_merge = left_df_daily
    df2_to_merge = right_df_daily
    merge_params = {'how': merge_how}

    if merge_on_cols:
        # If merging on columns, reset index for the merge operation
        # The time index will be restored/re-evaluated after the merge
        df1_to_merge = left_df_daily.reset_index()
        df2_to_merge = right_df_daily.reset_index()
        # Ensure the original time columns (if not part of merge_on_cols) are preserved or handled
        # For simplicity, this example assumes merge_on_cols includes the relevant join keys
        # and time columns might be part of them or will be handled post-merge.
        merge_params['on'] = merge_on_cols
        # Ensure merge_on_cols exist (resample_df_to_daily might change columns if only index existed)
        missing_keys_df1 = [key for key in merge_on_cols if key not in df1_to_merge.columns]
        missing_keys_df2 = [key for key in merge_on_cols if key not in df2_to_merge.columns]
        if missing_keys_df1 or missing_keys_df2:
            err_msg = []
            if missing_keys_df1: err_msg.append(f"左数据集缺少合并键: {missing_keys_df1}")
            if missing_keys_df2: err_msg.append(f"右数据集缺少合并键: {missing_keys_df2}")
            status_container.error("合并失败: " + ", ".join(err_msg) + "。这可能在重采样后发生，如果原始列中没有非数字数据。")
            return None, None
    else:
        # Index-based merge
        merge_params['left_index'] = True
        merge_params['right_index'] = True

    try:
        merged_df = _internal_merge_dataframes(
            left_df=df1_to_merge,
            right_df=df2_to_merge,
            **merge_params
        )

        if merged_df is None:
            status_container.error("数据合并步骤失败。")
            return None, None

        # --- Post-merge index and frequency logic (adapted from UI) ---
        final_time_col_for_index = None
        # Try to use original time columns if they still exist and make sense
        # This logic might need refinement based on how merge_on_cols interacts with time columns
        if left_time_col in merged_df.columns and pd.api.types.is_datetime64_any_dtype(merged_df[left_time_col]):
            final_time_col_for_index = left_time_col
        elif right_time_col in merged_df.columns and pd.api.types.is_datetime64_any_dtype(merged_df[right_time_col]):
            final_time_col_for_index = right_time_col
        # If merge was on index, the index should be datetime already from resample output
        elif not merge_on_cols and isinstance(merged_df.index, pd.DatetimeIndex):
            pass # Index is already good
        elif merge_on_cols: # If merged on columns, need to find and set a new DatetimeIndex
             # This part might need more robust logic to identify the correct time column post-merge
            potential_time_cols = [col for col in merged_df.columns if pd.api.types.is_datetime64_any_dtype(merged_df[col])]
            if potential_time_cols:
                final_time_col_for_index = potential_time_cols[0] # take the first one
                status_container.info(f"合并后，使用列 '{final_time_col_for_index}' 作为新的时间索引。")

        if final_time_col_for_index and final_time_col_for_index in merged_df.columns:
            try:
                merged_df[final_time_col_for_index] = pd.to_datetime(merged_df[final_time_col_for_index], errors='coerce')
                merged_df.dropna(subset=[final_time_col_for_index], inplace=True)
                if not merged_df.empty:
                    # merged_df.sort_values(by=final_time_col_for_index, inplace=True) # Sorting by time before setting index
                    # Drop duplicates on time column before setting index to avoid issues with non-unique index
                    merged_df.drop_duplicates(subset=[final_time_col_for_index], keep='first', inplace=True)
                    merged_df.set_index(final_time_col_for_index, inplace=True)
            except Exception as e_set_idx:
                status_container.warning(f"尝试从列 '{final_time_col_for_index}' 设置最终时间索引失败: {e_set_idx}")
        
        inferred_freq_str = "未知 (未能建立有效时间索引)"
        if isinstance(merged_df.index, pd.DatetimeIndex):
            if merged_df.index.name is None:
                merged_df.index.name = 'Time' # Default name for unnamed DatetimeIndex
            if not merged_df.index.is_monotonic_increasing and not merged_df.index.is_monotonic_decreasing:
                 merged_df.sort_index(inplace=True) # Sort if not already sorted
            # The UI had sort_index(ascending=False). Replicating that here. Consider if this is always desired.
            merged_df.sort_index(ascending=False, inplace=True) 
            inferred_freq_str = robust_infer_freq(merged_df.index.to_series())
        
        status_container.success("数据已成功合并并后处理。")
        return merged_df, inferred_freq_str

    except ValueError as ve: # Catch specific errors like from pd.merge itself
        status_container.error(f"合并或后处理失败: {ve}")
        return None, None
    except Exception as e_main:
        status_container.error(f"执行合并及后处理过程中发生意外错误: {e_main}")
        return None, None

def merge_dataframes(
    left_df: pd.DataFrame, 
    right_df: pd.DataFrame, 
    how: str = 'inner', 
    left_on: Optional[List[str]] = None, 
    right_on: Optional[List[str]] = None,
    on: Optional[List[str]] = None,
    suffixes: Tuple[str, str] = ('_x', '_y')
) -> Optional[pd.DataFrame]:
    """
    Merges two DataFrames based on specified keys and method.

    Args:
        left_df: The left DataFrame.
        right_df: The right DataFrame.
        how: Type of merge to be performed. 
             Can be one of 'left', 'right', 'outer', 'inner', 'cross'. Default is 'inner'.
        left_on: List of column names to join on in the left DataFrame.
        right_on: List of column names to join on in the right DataFrame.
        on: List of column names to join on. Must be found in both DataFrames.
            If 'on' is None and 'left_on' and 'right_on' are None, performs an index-based merge.
        suffixes: A tuple of string suffixes to apply to overlapping column names.

    Returns:
        A new DataFrame resulting from the merge, or None if an error occurs.
    """
    if left_df is None or right_df is None:
        st.error("用于合并的一个或两个数据框为空。")
        return None

    # Validate merge keys
    if on:
        if not all(col in left_df.columns for col in on):
            st.error(f"并非所有 'on' 指定的列 ({on}) 都存在于左侧数据框中。")
            return None
        if not all(col in right_df.columns for col in on):
            st.error(f"并非所有 'on' 指定的列 ({on}) 都存在于右侧数据框中。")
            return None
    elif left_on and right_on:
        if len(left_on) != len(right_on):
            st.error("'left_on' 和 'right_on' 的键数量必须相同。")
            return None
        if not all(col in left_df.columns for col in left_on):
            st.error(f"并非所有 'left_on' 指定的列 ({left_on}) 都存在于左侧数据框中。")
            return None
        if not all(col in right_df.columns for col in right_on):
            st.error(f"并非所有 'right_on' 指定的列 ({right_on}) 都存在于右侧数据框中。")
            return None
    elif (left_on and not right_on) or (not left_on and right_on):
        st.error("必须同时提供 'left_on' 和 'right_on'，或者只提供 'on'，或者都不提供以进行索引合并。")
        return None

    try:
        if on:
            merged_df = pd.merge(left_df, right_df, how=how, on=on, suffixes=suffixes)
        elif left_on and right_on:
            merged_df = pd.merge(left_df, right_df, how=how, left_on=left_on, right_on=right_on, suffixes=suffixes)
        else: # Index merge if no keys are provided
            merged_df = pd.merge(left_df, right_df, how=how, left_index=True, right_index=True, suffixes=suffixes)
        
        # st.success("数据框成功合并！") # Removed
        return merged_df
    except Exception as e:
        st.error(f"合并数据框时发生错误: {e}")
        return None

def align_dataframe_frequency(
    df: pd.DataFrame,
    target_freq_ui: str,  # e.g., "日", "周", "月", "季度", "年"
    align_to_ui: str,  # e.g., "周期第一天", "周期最后一天"
    day_type_ui: str,  # e.g., "自然日", "工作日"
    status_container=st  # For displaying messages
) -> Optional[pd.DataFrame]:
    """
    Aligns the DataFrame to a target frequency, rule, and day type.
    Assumes df has a DatetimeIndex.
    """
    if not isinstance(df.index, pd.DatetimeIndex):
        status_container.error("输入数据没有有效的DatetimeIndex，无法对齐频率。")
        return None
    if df.empty:
        status_container.warning("输入数据为空，无法对齐频率。")
        return df # Return empty df as is

    # --- Map UI selections to Pandas frequency codes and aggregation --- 
    # Default to daily if no specific frequency is chosen or if it's daily (no change)
    if target_freq_ui == "日":
        status_container.info("目标频率为 '日'，无需进行频率对齐。")
        # Even for daily, we might need to adjust for business days if selected
        # This part will be handled below if day_type_ui == "工作日"
        pass

    freq_pd = None
    agg_method = 'last' # Default for end of period

    if align_to_ui == "周期第一天":
        agg_method = 'first'

    if target_freq_ui == "周":
        # For weekly, pandas 'W' defaults to W-SUN. 
        # If '周期第一天', we want W-MON, so resample to 'W-MON' and take first.
        # If '周期最后一天', we want W-SUN (default 'W'), and take last.
        # Or, more generally, use 'W' and then adjust based on day_type_ui and align_to_ui for business days.
        # Let's use a base weekly frequency and adjust using dayofweek for precision if needed.
        # For simplicity, we can resample to 'W' (Sunday) and then adjust for '周期第一天' to Monday if needed.
        # Or, even better, use 'W-MON' for first day, 'W-TUE', etc.
        # For '周期第一天': 'W-MON', 'W-TUE', ..., 'W-FRI' for business days
        # For '周期最后一天': 'W-FRI', 'W-SAT', 'W-SUN' (but business day handling is key)
        
        # Let's map to specific end-of-week day if needed, default is W-SUN
        # 'W-MON', 'W-TUE', 'W-WED', 'W-THU', 'W-FRI', 'W-SAT', 'W-SUN'
        if align_to_ui == "周期第一天":
            freq_pd = 'W-MON' # Resample to weekly, anchor on Monday
            agg_method = 'first'
        else: # 周期最后一天
            freq_pd = 'W' # Resample to weekly, anchor on Sunday (default for 'W')
            agg_method = 'last'
            # The subsequent '工作日' logic will handle moving Sunday to Friday if day_type_ui is '工作日'.
            # Removed: if day_type_ui == "工作日": freq_pd = 'W-FRI'

    elif target_freq_ui == "月":
        freq_pd = 'MS' if align_to_ui == "周期第一天" else 'M'
    elif target_freq_ui == "季度":
        freq_pd = 'QS' if align_to_ui == "周期第一天" else 'Q'
    elif target_freq_ui == "年":
        freq_pd = 'AS' if align_to_ui == "周期第一天" else 'A'
    
    aligned_df = df
    if freq_pd: # If not '日' or unmapped
        try:
            aligned_df = df.resample(freq_pd).agg(agg_method)
            status_container.info(f"数据已重采样至频率 '{target_freq_ui}' ({freq_pd})，对齐到 '{align_to_ui}'。")
        except Exception as e_resample:
            status_container.error(f"重采样至频率 '{target_freq_ui}' ({freq_pd}) 失败: {e_resample}")
            return None
    
    # --- Handle '工作日' (Business Day) logic --- 
    if day_type_ui == "工作日":
        if aligned_df.empty:
            status_container.info("重采样后数据为空，工作日调整跳过。")
            return aligned_df
            
        original_index = aligned_df.index
        new_index = []
        for date_val in original_index:
            if date_val.weekday() >= 5: # 5 for Saturday, 6 for Sunday
                if align_to_ui == "周期第一天":
                    # Move to next business day (Monday normally)
                    new_date = date_val + pd.offsets.BusinessDay(n=1)
                    # Ensure it doesn't skip over a whole period for low frequencies (e.g. monthly)
                    # This simple adjustment is generally fine for daily->weekly/monthly
                else: # "周期最后一天"
                    # Move to previous business day (Friday normally)
                    new_date = date_val - pd.offsets.BusinessDay(n=1)
                new_index.append(new_date)
            else:
                new_index.append(date_val)
        
        aligned_df.index = pd.DatetimeIndex(new_index)
        
        # After business day adjustment, duplicates might arise if multiple non-BDays map to the same BDay.
        # Example: Sat and Sun (if they were somehow distinct after initial resample) both map to prev Friday for 'last day'.
        # Or if initial resample already created BDay, and non-BDay also maps to it.
        # Keep first for '周期第一天', keep last for '周期最后一天'. This mirrors the agg_method logic.
        keep_rule = 'first' if align_to_ui == "周期第一天" else 'last'
        aligned_df = aligned_df[~aligned_df.index.duplicated(keep=keep_rule)]
        aligned_df.sort_index(inplace=True) # Ensure sorted after index manipulation
        status_container.info(f"索引已调整为 '{day_type_ui}'。")

    # Drop rows that are all NaN after resampling/alignment
    aligned_df.dropna(axis=0, how='all', inplace=True)

    if aligned_df.empty:
        status_container.warning("频率对齐后，结果数据框为空。")
        return None
        
    return aligned_df

def robust_infer_freq(time_series: pd.Series) -> str:
    if not isinstance(time_series, pd.Series) or time_series.empty:
        # Consider logging instead of st.error for backend functions or return specific error codes/messages
        return "未知 (无时间数据)"
    if not pd.api.types.is_datetime64_any_dtype(time_series.dtype):
        return "未知 (非时间类型)"

    ts_sorted = time_series.dropna().sort_values().drop_duplicates()

    if len(ts_sorted) < 2:
        return "未知 (数据点不足)"

    try:
        pandas_freq = pd.infer_freq(ts_sorted)
        if pandas_freq:
            return pandas_freq
    except Exception:
        pass # Proceed to custom logic if pandas.infer_freq fails

    diffs = ts_sorted.diff().dropna()
    if diffs.empty:
        return "单一时间点或无法计算差异"

    diff_value_counts = diffs.value_counts()
    if diff_value_counts.empty:
        return "不规则 (无主导间隔)"

    modal_diff = diff_value_counts.index[0]
    count = diff_value_counts.iloc[0]
    total_diffs = len(diffs)

    freq_label = ""
    if modal_diff == pd.Timedelta(days=1): freq_label = "日度"
    elif modal_diff == pd.Timedelta(days=7): freq_label = "周度"
    elif modal_diff == pd.Timedelta(hours=1): freq_label = "小时"
    elif pd.Timedelta(days=28) <= modal_diff <= pd.Timedelta(days=31): freq_label = "月度近似"
    elif pd.Timedelta(days=89) <= modal_diff <= pd.Timedelta(days=92): freq_label = "季度近似"
    elif pd.Timedelta(days=360) <= modal_diff <= pd.Timedelta(days=370): freq_label = "年度近似"

    if freq_label:
        if count / total_diffs > 0.5:
            return f"{freq_label} (主导)"
        else:
            return f"{freq_label} (最常见: {modal_diff})"
    else:
        if count / total_diffs > 0.3:
            return f"主要间隔: {modal_diff}"
        else:
            return "不规则或混合频率"

def resample_df_to_daily(df: pd.DataFrame, time_col: str, df_name_for_error: str, output_container=st) -> pd.DataFrame | None:
    # Passing 'output_container=st' makes this function still dependent on Streamlit for UI messages.
    # A better long-term solution would be to return status and messages for the UI to display.
    try:
        df_resampled = df.copy()
        df_resampled[time_col] = pd.to_datetime(df_resampled[time_col], errors='coerce')
        df_resampled = df_resampled.dropna(subset=[time_col])
        if df_resampled.empty:
            output_container.error(f"数据集 '{df_name_for_error}' 在转换时间列 '{time_col}' 后为空或所有时间值无效，无法重采样。")
            return None
        df_resampled = df_resampled.set_index(time_col)

        agg_funcs = {}
        for col_name in df_resampled.columns:
            if pd.api.types.is_numeric_dtype(df_resampled[col_name]):
                agg_funcs[col_name] = 'mean'
            elif pd.api.types.is_datetime64_any_dtype(df_resampled[col_name]):
                agg_funcs[col_name] = 'first' # or 'min', 'max' as appropriate
            else: # Object, string, category, etc.
                agg_funcs[col_name] = 'first' 
    
        if not agg_funcs and not df_resampled.index.empty: # Only index, no other columns to aggregate
            daily_index = pd.date_range(start=df_resampled.index.min(), end=df_resampled.index.max(), freq='D')
            df_resampled_final = pd.DataFrame(index=daily_index)
            output_container.warning(f"数据集 '{df_name_for_error}' 重采样后仅剩时间索引，其他列丢失或不可聚合。")
        elif not agg_funcs and df_resampled.index.empty:
            output_container.error(f"数据集 '{df_name_for_error}' 为空或时间列无效，无法进行有效的日度重采样。")
            return None
        else:
            df_resampled_final = df_resampled.resample('D').agg(agg_funcs)
    
        output_container.success(f"数据集 '{df_name_for_error}' 已成功重采样至日度频率。")
        return df_resampled_final
    except Exception as e:
        output_container.error(f"重采样数据集 '{df_name_for_error}' 至日度频率失败: {e}")
        return None
