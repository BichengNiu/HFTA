import streamlit as st
import pandas as pd

# Import the new UI functions
from .processed_data_ui import display_column_selector_for_staging_ui
from .time_column_ui import display_time_filter_for_staging_ui

def display_data_staging_controls(st, session_state):
    """Displays UI for staging the currently processed data."""

    st.subheader("数据暂存")
    st.caption(
        "您可以将当前处理阶段的数据暂存起来，以便后续在其他工具或分析流程中使用。"
        "请先选择要保留的列和时间范围（可选），然后为暂存的数据指定一个唯一的名称。"
    )

    if session_state.get('ts_tool_data_processed') is None or session_state.ts_tool_data_processed.empty:
        st.info("当前没有已处理的数据可供暂存。请先完成数据加载和预处理步骤。")
        return

    # --- 1. Display Column Selector for Staging ---
    # This will store selected columns in session_state.ts_tool_staging_selected_columns
    display_column_selector_for_staging_ui(st, session_state, df_key='ts_tool_data_processed', selected_cols_session_key='ts_tool_staging_selected_columns')
    st.markdown("<br>", unsafe_allow_html=True) # Add some space

    # --- 2. Display Time Filter for Staging ---
    # This will store filter dates in session_state.ts_tool_staging_active_time_filter_start_date / _end_date
    display_time_filter_for_staging_ui(st, session_state, df_key='ts_tool_data_processed', time_col_info_key='ts_tool_time_col_info', active_filter_prefix='ts_tool_staging_active_time_filter')
    st.markdown("<br>", unsafe_allow_html=True) # Add some space

    # --- 3. Input for Staging Name and Staging Button ---
    stage_name = st.text_input("为处理后的数据命名以便暂存:", key="stage_data_name_input_main_v2", placeholder="例如：处理后的销售数据_Q1")
    
    col_stage_button, col_stage_info = st.columns([1, 3])

    with col_stage_button:
        if st.button("应用筛选并暂存数据", key="stage_processed_data_button_v2"):
            if not stage_name:
                st.error("请输入暂存数据的名称。")
            elif not session_state.get('ts_tool_staging_selected_columns'):
                st.error("请至少选择一列进行暂存。")
            else:
                if 'staged_data' not in session_state:
                    session_state.staged_data = {}
                
                if stage_name in session_state.staged_data:
                    st.warning(f"名称 '{stage_name}' 的暂存数据已存在。如果您继续，它将被覆盖。")
                
                # --- Prepare data for staging based on selections ---
                df_current_processed = session_state.ts_tool_data_processed.copy()
                
                # Apply column selection
                selected_cols_for_staging = session_state.get('ts_tool_staging_selected_columns', [])
                if not selected_cols_for_staging: # Should be caught by above check, but as a safeguard
                    st.error("错误：没有选择任何列用于暂存。")
                    return # Stop staging if no columns selected
                
                try:
                    df_to_stage = df_current_processed[selected_cols_for_staging].copy()
                except KeyError as e:
                    st.error(f"选择的列中存在错误: {e}。可能某些列已在之前的步骤中被移除或重命名。请重新选择列。")
                    # Filter out invalid columns from selection and retry or just error out
                    valid_cols = [col for col in selected_cols_for_staging if col in df_current_processed.columns]
                    if not valid_cols:
                         st.error("所有选定列均无效。无法暂存。")
                         return
                    df_to_stage = df_current_processed[valid_cols].copy()
                    session_state.ts_tool_staging_selected_columns = valid_cols # Update selection with valid ones
                    st.warning("暂存的列已更新为当前数据中实际存在的有效列。")

                # Apply time filter
                staging_filter_start_str = session_state.get('ts_tool_staging_active_time_filter_start_date')
                staging_filter_end_str = session_state.get('ts_tool_staging_active_time_filter_end_date')
                time_col_info = session_state.get('ts_tool_time_col_info', {})
                time_col_name_for_staging = time_col_info.get('name')

                if (staging_filter_start_str or staging_filter_end_str) and time_col_name_for_staging and time_col_name_for_staging in df_to_stage.columns:
                    try:
                        # Step 1: Convert time column from the current df_to_stage (which has selected columns)
                        original_time_series = df_to_stage[time_col_name_for_staging]
                        converted_time_col = pd.to_datetime(original_time_series, errors='coerce')
                        
                        # Step 2: Create a working copy for filtering. 
                        # Assign the converted time column to this copy.
                        df_for_filtering_and_staging = df_to_stage.copy()
                        df_for_filtering_and_staging[time_col_name_for_staging] = converted_time_col
                        
                        # Step 3: Remove rows with NaT dates from this working copy
                        initial_rows = len(df_for_filtering_and_staging)
                        df_for_filtering_and_staging.dropna(subset=[time_col_name_for_staging], inplace=True)
                        rows_after_nat_drop = len(df_for_filtering_and_staging)
                        
                        if rows_after_nat_drop < initial_rows:
                            print(f"[STAGING DEBUG] {initial_rows - rows_after_nat_drop} row(s) dropped due to NaT conversion in time column '{time_col_name_for_staging}' before date range filtering.")
                        
                        # Step 4: Apply date range filtering using the converted & cleaned time column
                        if staging_filter_start_str:
                            dt_start = pd.to_datetime(staging_filter_start_str).normalize()
                            df_for_filtering_and_staging = df_for_filtering_and_staging[df_for_filtering_and_staging[time_col_name_for_staging] >= dt_start]
                        if staging_filter_end_str:
                            dt_end = pd.to_datetime(staging_filter_end_str).normalize()
                            df_for_filtering_and_staging = df_for_filtering_and_staging[df_for_filtering_and_staging[time_col_name_for_staging] <= dt_end]
                        
                        # Step 5: Update df_to_stage to be this fully processed DataFrame
                        df_to_stage = df_for_filtering_and_staging
                        
                        print(f"[STAGING] Time filter processing complete on '{time_col_name_for_staging}'. Shape of df_to_stage for summary/storage: {df_to_stage.shape}")

                    except Exception as e_filter:
                        st.warning(f"应用时间筛选到暂存数据时出错: {e_filter}. 时间筛选可能未完全生效。")
                        print(f"[STAGING ERROR] Time filter error: {e_filter}")
                elif (staging_filter_start_str or staging_filter_end_str) and not time_col_name_for_staging:
                     st.warning("已设置时间筛选范围，但未识别时间列。时间筛选未应用于暂存数据。")
                elif (staging_filter_start_str or staging_filter_end_str) and time_col_name_for_staging and time_col_name_for_staging not in df_to_stage.columns:
                    st.warning(f"已设置时间筛选范围，但时间列 '{time_col_name_for_staging}' 不在选定要暂存的列中。时间筛选未应用于暂存数据。")

                if df_to_stage.empty:
                    st.error("筛选和列选择后，没有数据可供暂存。请调整您的选择。")
                else:
                    print(f"[STAGING SUMMARY] df_to_stage.shape: {df_to_stage.shape}") # DEBUG
                    print(f"[STAGING SUMMARY] df_to_stage.columns: {df_to_stage.columns.tolist()}") # DEBUG
                    
                    # Determine the time column to be recorded in summary
                    time_col_for_summary_final = None
                    if time_col_name_for_staging and time_col_name_for_staging in df_to_stage.columns:
                        time_col_for_summary_final = time_col_name_for_staging
                    
                    print(f"[STAGING SUMMARY] Initial time_col_name_for_staging: {time_col_name_for_staging}") # DEBUG
                    print(f"[STAGING SUMMARY] Final time_col_for_summary_final (in df_to_stage.columns?): {time_col_for_summary_final}") # DEBUG

                    summary = {
                        'rows': df_to_stage.shape[0],
                        'cols': df_to_stage.shape[1],
                        'columns': df_to_stage.columns.tolist(),
                        'staged_from_filter_start': staging_filter_start_str,
                        'staged_from_filter_end': staging_filter_end_str,
                        'time_col_at_staging': time_col_for_summary_final # Use the validated one
                    }
                    
                    # Add time range from actual data if time column is present in summary
                    if summary['time_col_at_staging']:
                        print(f"[STAGING SUMMARY] Attempting to extract time details using column: {summary['time_col_at_staging']}") # DEBUG
                        try:
                            time_series_for_summary = pd.to_datetime(df_to_stage[summary['time_col_at_staging']], errors='coerce').dropna()
                            print(f"[STAGING SUMMARY DEBUG] df_to_stage[time_col].tail():\n{df_to_stage[summary['time_col_at_staging']].tail()}") # ADDED DEBUG
                            print(f"[STAGING SUMMARY DEBUG] time_series_for_summary.tail():\n{time_series_for_summary.tail()}") # ADDED DEBUG
                            print(f"[STAGING SUMMARY DEBUG] time_series_for_summary.max() direct value: {time_series_for_summary.max()}") # ADDED DEBUG
                            print(f"[STAGING SUMMARY] time_series_for_summary is empty after to_datetime & dropna: {time_series_for_summary.empty}") # DEBUG
                            if not time_series_for_summary.empty:
                                summary['data_start_time'] = str(time_series_for_summary.min().date() if hasattr(time_series_for_summary.min(), 'date') else time_series_for_summary.min())
                                summary['data_end_time'] = str(time_series_for_summary.max().date() if hasattr(time_series_for_summary.max(), 'date') else time_series_for_summary.max())
                                summary['data_frequency'] = str(pd.infer_freq(time_series_for_summary.sort_values()) or '未知')
                                print(f"[STAGING SUMMARY] Extracted time details: Start={summary['data_start_time']}, End={summary['data_end_time']}, Freq={summary['data_frequency']}") # DEBUG
                            else:
                                print("[STAGING SUMMARY] time_series_for_summary was empty. Setting time details to '未知'.")
                                summary['data_start_time'] = '未知 (时间列解析后为空)'
                                summary['data_end_time'] = '未知 (时间列解析后为空)'
                                summary['data_frequency'] = '未知 (时间列解析后为空)'
                        except Exception as e_summary_time:
                            print(f"[STAGING SUMMARY ERROR] Error extracting time details for summary: {e_summary_time}")
                            summary['data_start_time'] = '未知 (提取时出错)'
                            summary['data_end_time'] = '未知 (提取时出错)'
                            summary['data_frequency'] = '未知 (提取时出错)'
                    else:
                        print("[STAGING SUMMARY] No time_col_at_staging. Setting time details to '未知'.")
                        summary['data_start_time'] = '未知 (无有效时间列)' # Clarified message
                        summary['data_end_time'] = '未知 (无有效时间列)' # Clarified message
                        summary['data_frequency'] = '未知 (无有效时间列)' # Clarified message

                    # Explicitly add first_col_display and last_col_display
                    if df_to_stage.shape[1] > 0:
                        summary['first_col_display'] = str(df_to_stage.columns[0])
                        summary['last_col_display'] = str(df_to_stage.columns[-1])
                        print(f"[STAGING SUMMARY] First col: {summary['first_col_display']}, Last col: {summary['last_col_display']}") # DEBUG
                    else:
                        summary['first_col_display'] = 'N/A (无列)'
                        summary['last_col_display'] = 'N/A (无列)'
                        print("[STAGING SUMMARY] No columns in df_to_stage for first/last col display.") # DEBUG
                    
                    print(f"[STAGING SUMMARY] Final summary object to be stored: {summary}") # DEBUG

                    # Store in session_state with a unified structure
                    session_state.staged_data[stage_name] = {
                        'df': df_to_stage,
                        'time_col': time_col_for_summary_final, # Key for unified structure
                        'source': 'staging', # Key for unified structure
                        'summary': summary # Original summary nested
                    }
                    st.success(f"数据 '{stage_name}' ({df_to_stage.shape[0]}行, {df_to_stage.shape[1]}列) 已成功应用筛选并暂存！")
                    # Clear inputs after successful staging
                    session_state.ts_tool_staging_selected_columns = df_current_processed.columns.tolist() # Reset column selector to all
                    session_state.ts_tool_staging_active_time_filter_start_date = None
                    session_state.ts_tool_staging_active_time_filter_end_date = None
                    st.rerun() # ADDED: Explicit rerun after successful staging
                    # Consider st.rerun() if the staging name input should clear or UI update needed immediately
                    # For now, rely on Streamlit's natural rerun for widget state if keys are managed.
    with col_stage_info:
        st.caption("点击按钮后，当前处理阶段的数据将根据上方选择的列和时间范围进行处理，然后保存到暂存区。")
    st.divider() 