import streamlit as st
import pandas as pd # Needed if any df operations happen here, though mostly for summary access
import datetime

# Helper function for CSV conversion, specific to sidebar staged data download
# 🔥 修复：移除缓存装饰器避免 ScriptRunContext 警告
# @st.cache_data
def _convert_df_to_csv_sidebar(df_in):
    return df_in.to_csv(index=False).encode('utf-8-sig')

def display_staged_data_sidebar(st, session_state):
    """Displays the sidebar UI for a unified current data preview and staged datasets."""

    with st.sidebar:
        # st.divider() # Removed divider, dashboard.py will handle it if needed before this call

        # --- Unified Current Data Preview ---
        st.subheader("当前数据预览")
        
        data_to_display = None
        data_label = ""
        
        # --- NEW: Check if Stationarity Tab preview data should be used --- #
        # Determine if the relevant exploration tool section is active
        is_exploration_tool_active = (
            session_state.get('selected_main_module') == "应用工具" and
            session_state.get('selected_sub_module') == "数据探索"
        )
        
        stationarity_preview_data = session_state.get('stationarity_tab_preview_df')
        print(f"[DEBUG Sidebar] is_exploration_tool_active: {is_exploration_tool_active}") # Check if exploration tool section is active
        print(f"[DEBUG Sidebar] stationarity_preview_data is None: {stationarity_preview_data is None}")
        if stationarity_preview_data is not None:
            print(f"[DEBUG Sidebar] stationarity_preview_data shape: {stationarity_preview_data.shape}")

        # Use stationarity preview IF the exploration section is active AND preview data exists
        if is_exploration_tool_active and stationarity_preview_data is not None and not stationarity_preview_data.empty:
             data_to_display = stationarity_preview_data
             data_label = "数据探索预览" # More general label
             print("[DEBUG Sidebar] Using preview data from stationarity tool section.")
        else:
            # --- Fallback Logic (Existing) --- #
            print("[DEBUG Sidebar] Exploration tool preview not active or empty. Falling back.")
            # Data from the Time Series Compute module
            compute_data = session_state.get('ts_compute_data')
            compute_data_source_name = session_state.get('ts_compute_file_name') 

            # Data from the Time Series Cleaning module
            processed_clean_data = session_state.get('ts_tool_data_processed')
            raw_clean_data = session_state.get('ts_tool_data_raw')

            # Prioritize data from the compute module if it seems to be the active context
            if compute_data is not None and not compute_data.empty and compute_data_source_name:
                data_to_display = compute_data
                # Clarify label based on source name content
                if isinstance(compute_data_source_name, str) and "来自暂存区:" in compute_data_source_name:
                    data_label = f"计算模块 ({compute_data_source_name.split(': ')[-1]})"
                else:
                    data_label = f"计算模块 ({compute_data_source_name})"
                print("[DEBUG Sidebar] Fallback: Using data from compute module.")
            elif processed_clean_data is not None and not processed_clean_data.empty:
                data_to_display = processed_clean_data
                data_label = "预处理后数据"
                print("[DEBUG Sidebar] Fallback: Using data from cleaning (processed).")
            elif raw_clean_data is not None and not raw_clean_data.empty:
                data_to_display = raw_clean_data
                data_label = "原始数据"
                print("[DEBUG Sidebar] Fallback: Using data from cleaning (raw).")
            # --- End Fallback Logic --- #

        # --- Display Logic (Common for all sources) ---
        if data_to_display is not None and not data_to_display.empty:
            # Create a copy for display modifications
            df_display_copy = data_to_display.copy()

            # Reset index if it has a name (often the time column after processing)
            # to make it a regular column for display purposes, matching other previews
            if df_display_copy.index.name is not None:
                print(f"[DEBUG Sidebar] Resetting index '{df_display_copy.index.name}' for display.")
                df_display_copy = df_display_copy.reset_index()
                
            # --- START ARROW COMPATIBILITY FIX ---
            # Ensure all datetime columns are ns precision for Arrow
            for col in df_display_copy.select_dtypes(include=['datetime64']).columns:
                try: # Add try-except for robustness
                     df_display_copy[col] = df_display_copy[col].astype('datetime64[ns]')
                except Exception as e_astype:
                     print(f"[DEBUG Sidebar] Error converting column {col} to datetime64[ns]: {e_astype}")
                     # Optionally handle the error, e.g., skip conversion for this column
                     pass 
            # No need to convert index here as we reset it above
            # --- END ARROW COMPATIBILITY FIX ---

            # Sort by Datetime column if it exists after potential reset_index
            time_col_for_sort = None
            for col in df_display_copy.columns:
                 if pd.api.types.is_datetime64_any_dtype(df_display_copy[col]):
                      time_col_for_sort = col
                      break
            if time_col_for_sort:
                try:
                    print(f"[DEBUG Sidebar] Sorting preview by column: {time_col_for_sort}")
                    df_display_copy = df_display_copy.sort_values(by=time_col_for_sort, ascending=False)
                except Exception as e_sort:
                    print(f"[DEBUG Sidebar] Error sorting by column {time_col_for_sort}: {e_sort}")
            else:
                 print("[DEBUG Sidebar] No datetime column found for sorting.")

            # Display the DataFrame without .head()
            st.dataframe(df_display_copy, height=250) 
            st.caption(f"当前显示: {data_label} (形状: {data_to_display.shape[0]}行, {data_to_display.shape[1]}列)")
        else:
            st.caption("当前无数据显示。请先上传文件或处理数据。")
        # --- End Unified Current Data Preview ---

        st.divider() # This divider should be before "暂存的数据集"
        st.subheader("暂存的数据集")
        if not session_state.get('staged_data'):
            st.info("暂存区为空。")
        else:
            datasets_to_delete = []
            for name, staged_item_dict in list(session_state.staged_data.items()): # Use list() for safe iteration if deleting
                # Ensure staged_item_dict is a dictionary and contains 'df' and 'summary'
                if not isinstance(staged_item_dict, dict) or 'df' not in staged_item_dict or 'summary' not in staged_item_dict:
                    st.error(f"暂存区中的项目 '{name}' 格式不正确，已跳过。")
                    continue
                    
                df_staged = staged_item_dict['df']
                summary = staged_item_dict['summary']

                expander_label = f"{name} (行: {summary.get('rows', 'N/A')}, 列: {summary.get('cols', 'N/A')})"
                with st.expander(expander_label, expanded=False):
                    # --- Improved First/Last Column Display --- 
                    num_cols_summary = summary.get('cols', 0)
                    first_col_summary = summary.get('first_col_display', 'N/A')
                    last_col_summary = summary.get('last_col_display', 'N/A')
                    if num_cols_summary == 1 and first_col_summary != 'N/A':
                        st.caption(f"列名: {first_col_summary}") # If only one column, show its name
                    elif num_cols_summary > 1:
                        st.caption(f"首列: {first_col_summary}")
                        st.caption(f"末列: {last_col_summary}")
                    else: # 0 columns
                        st.caption("首列: N/A")
                        st.caption("末列: N/A")
                    # --- End Improved Display --- 
                        
                    st.caption(f"起始时间: {summary.get('data_start_time', '未知')}")
                    st.caption(f"结束时间: {summary.get('data_end_time', '未知')}")
                    st.caption(f"推断频率: {summary.get('data_frequency', '未知')}")
                    
                    col_dl, col_del = st.columns(2)
                    with col_dl:
                        csv_data_staged = _convert_df_to_csv_sidebar(df_staged)
                        st.download_button(
                            label="下载", # Changed label
                            data=csv_data_staged,
                            file_name=f"{name}.csv",
                            mime='text/csv',
                            key=f"download_staged_{name}_{summary.get('rows', 'N')}", 
                            use_container_width=True
                        )
                    with col_del:
                        if st.button("删除", key=f"delete_staged_{name}_{summary.get('cols', 'N')}", use_container_width=True): # Changed label
                            datasets_to_delete.append(name)
            
            if datasets_to_delete:
                for name_to_del in datasets_to_delete:
                    if name_to_del in session_state.staged_data:
                        del session_state.staged_data[name_to_del]
                        print(f"Removed '{name_to_del}' from staging area.")
                st.rerun() 