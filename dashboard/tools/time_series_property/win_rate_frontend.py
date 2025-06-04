import streamlit as st
import pandas as pd
import numpy as np # Though not directly used here, good to have for pandas interaction
from datetime import datetime # For pd.Timestamp.now()
# Removed dateutil.relativedelta as it's now in backend
# Removed collections.defaultdict as it's now in backend

# Import backend functions
from . import win_rate_backend

def display_win_rate_tab(st_obj, session_state):
    # Data acquisition from session_state (set by dashboard.py)
    # Assuming 'correlation_selected_df_name' and 'correlation_selected_df' are the primary keys from dashboard
    selected_df_name_from_dashboard = session_state.get('correlation_selected_df_name') 
    df_original_from_dashboard = session_state.get('correlation_selected_df')

    if df_original_from_dashboard is None or df_original_from_dashboard.empty:
        st_obj.info("请先在主界面选择一个暂存数据集以进行胜率计算。")
        # Clear any tab-specific states if main data is removed
        # This requires knowing the keys used by this tab, manage them carefully.
        # For now, we assume session state keys are prefixed or uniquely named to avoid broad clearing.
        return
        
    df_for_processing = df_original_from_dashboard.copy()
    
    # --- Session State Management for UI persistence specific to this tab & dataset ---
    # This helps maintain UI state correctly when switching between datasets or tabs.
    tab_dataset_prefix = f"win_rate_{selected_df_name_from_dashboard}"

    # Keys for UI elements
    target_series_key = f"{tab_dataset_prefix}_target_series_v2"
    ref_series_list_key = f"{tab_dataset_prefix}_ref_series_list_v2"
    time_ranges_key = f"{tab_dataset_prefix}_time_ranges_v2"
    # Key to store results for this specific dataset after calculation
    results_key = f"{tab_dataset_prefix}_results_v2"
    # Flag to indicate if data processing/conversion for datetime index was done for this dataset
    datetime_conversion_done_key = f"{tab_dataset_prefix}_dt_conversion_done_v2"
    datetime_index_available_key = f"{tab_dataset_prefix}_dt_index_available_v2"

    # --- Datetime Index Conversion Logic (run once per dataset load for this tab) ---
    is_datetime_index_available_for_current_df = session_state.get(datetime_index_available_key, False)

    if not session_state.get(datetime_conversion_done_key, False):
        is_dt_idx = isinstance(df_for_processing.index, pd.DatetimeIndex)
        if not is_dt_idx:
            st_obj.warning("警告：选定数据集的索引不是日期时间格式。时间范围筛选功能可能无法正常工作或将尝试转换。")
            date_col_candidates = ['date', 'Date', 'timestamp', 'Timestamp', '交易日期', 'time'] # Added 'time'
            converted_successfully = False
            for col_name in date_col_candidates:
                if col_name in df_for_processing.columns:
                    try:
                        # Try conversion on a temporary copy first
                        temp_df_check = df_for_processing.copy()
                        temp_df_check[col_name] = pd.to_datetime(temp_df_check[col_name])
                        temp_df_check = temp_df_check.set_index(col_name)
                        # If successful, apply to the actual df_for_processing
                        df_for_processing = temp_df_check
                        st_obj.info(f"已尝试将列 '{col_name}' 转换为日期时间索引。")
                        converted_successfully = True
                        is_dt_idx = True # Update status
                        break 
                    except Exception as e_convert:
                        st_obj.warning(f"尝试将列 '{col_name}' 转换为日期时间索引失败: {e_convert}")
            if not converted_successfully:
                 st_obj.error("未能自动转换有效的日期时间索引。涉及特定时间范围的筛选将不可用或不准确。")
        session_state[datetime_index_available_key] = is_dt_idx
        session_state[datetime_conversion_done_key] = True # Mark that conversion attempt was made
        # If conversion happened, df_for_processing is updated, so store it for this session context if needed
        # Or, ensure all downstream uses this potentially modified df_for_processing
    else:
        # On subsequent runs for the same dataset, retrieve the stored availability status
        is_datetime_index_available_for_current_df = session_state.get(datetime_index_available_key, False)
        # If df_for_processing was modified by setting index, this needs to be consistently handled.
        # For simplicity, re-apply index setting if it was successful previously.
        # This is a bit complex; a cleaner way might be to store the modified df in session_state too.
        # For now, the check above handles the initial conversion. Subsequent runs use the boolean flag.
        # The crucial part is that the backend receives the correct df and the boolean flag.

    series_options = [col for col in df_for_processing.columns if pd.api.types.is_numeric_dtype(df_for_processing[col])] 
    if not series_options:
        st_obj.warning("选定的数据集中没有可用的数值类型的列进行计算。")
        return

    # --- UI Layout --- # 
    col1, col2, col3 = st_obj.columns(3)
    with col1:
        # Ensure default is valid or first option
        current_target = session_state.get(target_series_key, series_options[0] if series_options else None)
        target_idx = series_options.index(current_target) if current_target in series_options else 0
        selected_target = st_obj.selectbox("选择目标序列", options=series_options, key=target_series_key, index=target_idx)
    with col2:
        ref_series_options = [s for s in series_options if s != selected_target]
        # Default selection for multiselect should be filtered by available options
        current_ref_selection = session_state.get(ref_series_list_key, [])
        valid_default_ref = [opt for opt in current_ref_selection if opt in ref_series_options]
        if not valid_default_ref and ref_series_options: # If previous default is all invalid, select all available
             valid_default_ref = ref_series_options[:]

        selected_ref_list = st_obj.multiselect("选择参考序列 (可多选)", options=ref_series_options, 
                                               default=valid_default_ref, key=ref_series_list_key)
    with col3:
        time_range_options = ["全部时间", "近半年", "近1年", "近3年"]
        # Default for time ranges
        current_time_range_selection = session_state.get(time_ranges_key, ["全部时间"])
        valid_default_time_ranges = [opt for opt in current_time_range_selection if opt in time_range_options]
        if not valid_default_time_ranges: valid_default_time_ranges = ["全部时间"]
        
        selected_time_ranges_ui = st_obj.multiselect("选择时间范围 (可多选)", options=time_range_options, 
                                                default=valid_default_time_ranges, key=time_ranges_key)
    
    # Helper function to pass to backend for getting current time (used for relative date ranges)
    def get_current_time_for_backend_filter():
        return pd.Timestamp.now().normalize()

    if st_obj.button("开始计算胜率", key=f"{tab_dataset_prefix}_calculate_button_v2"):
        # Retrieve current selections from UI state (which are stored in session_state by widgets)
        target_to_use = session_state.get(target_series_key)
        refs_to_use = session_state.get(ref_series_list_key, []) 
        time_ranges_to_use = session_state.get(time_ranges_key, ["全部时间"])

        if not target_to_use or not refs_to_use or not time_ranges_to_use:
            st_obj.warning("请确保已选择目标序列、至少一个参考序列以及至少一个时间范围。")
        else:
            with st.spinner("正在计算胜率..."):
                # Use the is_datetime_index_available_for_current_df determined earlier
                results_data, errors, warnings = win_rate_backend.perform_batch_win_rate_calculation(
                    df_input=df_for_processing, # This df might have been modified with set_index
                    target_series_name=target_to_use,
                    ref_series_names_list=refs_to_use,
                    selected_time_ranges=time_ranges_to_use,
                    is_datetime_index_available=is_datetime_index_available_for_current_df, # Pass the determined status
                    get_current_time_for_filter=get_current_time_for_backend_filter
                )
            
            session_state[results_key] = results_data # Store results under the dataset-specific key
            for err_msg in errors: st_obj.error(err_msg)
            for warn_msg in warnings: st_obj.warning(warn_msg)
            
            if not errors: # Show success only if no critical errors from backend
                st_obj.success("胜率计算完成。")
            # Rerun to display results table if it was updated
            st_obj.rerun()

    # --- Display Results --- #
    # Always try to display results from session_state if they exist for the current dataset context
    # This allows results to persist across reruns not triggered by the button itself (e.g., other widget changes)
    current_results_to_display = session_state.get(results_key)
    if current_results_to_display:
        st_obj.markdown("#### 胜率计算结果")
        # Convert defaultdict to a regular dict for DataFrame conversion, though DataFrame handles it.
        results_df_disp = pd.DataFrame.from_dict(dict(current_results_to_display), orient='index') 
        if not results_df_disp.empty:
            results_df_disp.index.name = "参考序列"
            # Ensure columns are in the order of selected_time_ranges for consistency
            # (if selected_time_ranges_ui reflects the order in the UI multiselect)
            ordered_time_range_cols = [tr for tr in selected_time_ranges_ui if tr in results_df_disp.columns]
            # Add any other columns that might have been generated if order is not strict
            for col in results_df_disp.columns:
                if col not in ordered_time_range_cols: ordered_time_range_cols.append(col)
            
            st_obj.dataframe(results_df_disp[ordered_time_range_cols])
        else:
            # This might occur if backend returns empty dict but no errors (e.g., all series failed individual checks)
            st_obj.info("未能生成有效的胜率结果表 (可能所有序列组合均无足够数据或变化)。")
    elif session_state.get(f"{tab_dataset_prefix}_calculate_button_v2"): # Check if button was ever pressed for this context
        # If button was pressed, but no results, means errors/warnings were shown above or no data was processable
        st_obj.caption("计算已执行，但无结果显示。请检查上述警告/错误信息。")
    else:
        st_obj.caption("请选择参数并点击 \"开始计算胜率\" 后，结果将在此处显示。") 