import streamlit as st
import pandas as pd
import numpy as np
import io # For download functionality
from statsmodels.tsa.seasonal import seasonal_decompose # For decomposition plot

# Import backend functions
from . import stationarity_backend 

# --- Helper Function for Updating Sidebar Preview State ---
def _update_stationarity_preview_df(session_state, selected_df_name):
    """
    Calculates the DataFrame based on current selections and updates 
    session_state['stationarity_tab_preview_df'].
    Should be called via callbacks.
    """
    print("[DEBUG Stationarity CB] Entering _update_stationarity_preview_df")
    selection_key = f"stationarity_selected_cols_{selected_df_name}"
    
    # Get current selection (potentially updated by radio or multiselect)
    current_selection = session_state.get(selection_key, [])
    processed_staged_df = session_state.get('processed_staged_df')
    original_df_staged = session_state.get('stationarity_selected_staged_data_df') # Needed for time col check maybe

    # --- Replicate Time Column Identification Logic (needed here) --- #
    time_col_name = None
    if processed_staged_df is not None and isinstance(processed_staged_df.index, pd.DatetimeIndex):
        time_col_name = processed_staged_df.index.name
        if not time_col_name: time_col_name = "时间索引" 
    elif processed_staged_df is not None: # Check columns if no datetime index
        for col in processed_staged_df.columns:
            if pd.api.types.is_datetime64_any_dtype(processed_staged_df[col]):
                 time_col_name = col
                 break
    print(f"[DEBUG Stationarity CB] Identified time_col_name: {time_col_name}")
    # --- End Time Column Identification --- #

    df_selected_for_action = pd.DataFrame() # Initialize

    if processed_staged_df is None or processed_staged_df.empty:
        print("[DEBUG Stationarity CB] processed_staged_df is empty or None.")
        if 'stationarity_tab_preview_df' in session_state:
            del session_state['stationarity_tab_preview_df']
        return # Nothing to process

    if not current_selection:
        print("[DEBUG Stationarity CB] No columns selected.")
        if 'stationarity_tab_preview_df' in session_state:
            del session_state['stationarity_tab_preview_df']
        return # No selection

    # Logic from the action block to build df_selected_for_action
    try:
        cols_to_select_from_df = [col for col in current_selection if col != time_col_name]
        print(f"[DEBUG Stationarity CB] Columns to select from df: {cols_to_select_from_df}")
        
        valid_cols_for_df_selection = [col for col in cols_to_select_from_df if col in processed_staged_df.columns]
        print(f"[DEBUG Stationarity CB] Valid columns for df selection: {valid_cols_for_df_selection}")

        if not valid_cols_for_df_selection and time_col_name and time_col_name in current_selection:
            # Handle case where only the time column/index is selected
            df_selected_for_action = pd.DataFrame({time_col_name: processed_staged_df.index})
            print(f"[DEBUG Stationarity CB] Only time column '{time_col_name}' selected. Created preview df with just time.")
        elif valid_cols_for_df_selection:
             # Create the dataframe for action using validated *data* columns
             df_selected_for_action = processed_staged_df[valid_cols_for_df_selection].copy()
             # Add the time index back if it was selected by the user
             if time_col_name and time_col_name in current_selection:
                 df_selected_for_action.index.name = time_col_name # Ensure index name is set
                 df_selected_for_action.reset_index(inplace=True) # Move index to be a column
                 # Ensure time column is first for consistency?
                 if time_col_name in df_selected_for_action.columns:
                      cols_ordered = [time_col_name] + [col for col in df_selected_for_action.columns if col != time_col_name]
                      df_selected_for_action = df_selected_for_action[cols_ordered]
                 print(f"[DEBUG Stationarity CB] Added time index '{time_col_name}' back as a column.")
        else:
            # No valid data columns and time column not selected or invalid
             print("[DEBUG Stationarity CB] No valid data columns selected for preview.")
             df_selected_for_action = pd.DataFrame()

        # --- Store this selected df for potential sidebar preview --- #
        if not df_selected_for_action.empty:
            session_state['stationarity_tab_preview_df'] = df_selected_for_action.copy()
            print(f"[DEBUG Stationarity CB] Stored df_selected_for_action in session_state['stationarity_tab_preview_df']. Shape: {df_selected_for_action.shape}")
        else:
            if 'stationarity_tab_preview_df' in session_state:
                del session_state['stationarity_tab_preview_df']
            print(f"[DEBUG Stationarity CB] No valid selection/empty df, cleared stationarity_tab_preview_df.")

    except Exception as e:
        print(f"[ERROR Stationarity CB] Error creating preview df: {e}")
        if 'stationarity_tab_preview_df' in session_state:
            del session_state['stationarity_tab_preview_df']

# --- Display Function for Stationarity Tab ---
def display_stationarity_tab(st, session_state):
    """Displays the stationarity test results and processed data for a selected staged dataset."""
    # --- Set flag indicating this tab is active --- #
    session_state['currently_in_stationarity_tab'] = True
    print("[DEBUG StationarityFrontend] Set currently_in_stationarity_tab = True")
    
    # Get selected data name from session state (set by dashboard.py)
    selected_df_name = session_state.get('stationarity_selected_staged_data_name')
    selected_df_staged = session_state.get('stationarity_selected_staged_data_df') 
    print(f"[StationarityFrontend] Entered display_stationarity_tab for dataset: {selected_df_name}") # DEBUG

    # --- Check if data is selected from the main dashboard ---
    if selected_df_staged is None or selected_df_staged.empty:
        st.info("请先在主界面从暂存区选择一个数据集以进行平稳性分析。")
        # Clear previous results if no data is selected now
        session_state.pop('adf_results_staged', None)
        session_state.pop('processed_staged_df', None)
        session_state.pop('last_processed_staged_dataset_name_stationarity', None)
        session_state.pop('stationarity_tab_preview_df', None) # Clear preview df when no main data
        # Clear method/order states too
        session_state.pop(f'stationarity_processing_method_{selected_df_name}', None) 
        session_state.pop(f'stationarity_diff_order_{selected_df_name}', None)
        return

    # --- Layout Adjustments ---
    # Line 1: Dataset selection (handled by dashboard.py, info displayed here by dashboard.py)
    # We just proceed with the settings for the selected_df_name

    # Line 2: Settings (Alpha, Method, Order)
    st.markdown(f"**检验设置 (数据集: {selected_df_name})**")
    settings_cols = st.columns([1, 2, 1.5]) # Alpha | Method | Order
    
    with settings_cols[0]: # Alpha
        alpha_key = f"stationarity_alpha_radio_{selected_df_name}" 
        selected_alpha = st.radio(
            "Alpha:", 
            options=[0.01, 0.05, 0.10],
            index=1, 
            horizontal=True, # <<< CHANGE TO HORIZONTAL
            key=alpha_key,
            label_visibility="visible"
        )
    
    with settings_cols[1]: # Method
        method_key = f'stationarity_processing_method_{selected_df_name}'
        method_options = {"保留原始": "keep", "差分": "diff", "对数差分": "log_diff"}
        selected_method_label = st.radio(
            "非平稳序列处理方法:",
            options=list(method_options.keys()),
            index=0, 
            horizontal=True,
            key=method_key
        )
        processing_method = method_options[selected_method_label]

    with settings_cols[2]: # Order (conditional)
        diff_order_key = f'stationarity_diff_order_{selected_df_name}'
        diff_order = 1 # Default
        if processing_method in ['diff', 'log_diff']:
            diff_order = st.number_input(
                "阶数:", 
                min_value=1,
                max_value=5, 
                value=session_state.get(diff_order_key, 1), 
                step=1,
                key=diff_order_key,
                label_visibility="visible"
            )
        else:
            # Placeholder to keep layout consistent if needed, or leave empty
            st.write("") # Ensures the column takes up space

    # Line 3: Execute Button (far left)
    st.markdown("") # Add a little space before the button row if needed
    exec_button_cols = st.columns([1.5, 4.5]) # Button | Spacer
    execute_pressed = False
    with exec_button_cols[0]:
        if st.button("执行检验与处理", key=f"execute_stationarity_button_{selected_df_name}"):
            execute_pressed = True

    # --- Execute Logic (Triggered AFTER layout is defined, based on flag) ---
    if execute_pressed:
        session_state.pop('adf_results_staged', None)
        session_state.pop('processed_staged_df', None)
        session_state.pop('stationarity_tab_preview_df', None) 
        print(f"[StationarityFrontend EXECUTE] Cleared old results for '{selected_df_name}' before running.")

        alpha_to_use = session_state.get(alpha_key, 0.05) 
        print(f"[StationarityFrontend EXECUTE] Using alpha from widget ({alpha_key}): {alpha_to_use}")

        with st.spinner(f"正在对数据集 '{selected_df_name}' 执行检验与处理 (方法: {selected_method_label}, Alpha: {alpha_to_use}, 阶数: {diff_order if processing_method in ['diff', 'log_diff'] else 'N/A'})..."):
            try:
                # Call backend function WITH NEW PARAMETERS
                results_staged, processed_df_staged_data_dict = stationarity_backend.test_and_process_stationarity(
                    selected_df_staged.copy(), 
                    alpha=alpha_to_use, # Use alpha from widget state
                    processing_method=processing_method, # Pass selected method
                    diff_order=diff_order # Pass selected order (backend will ignore if not relevant)
                )
                
                # Convert the returned dictionary to DataFrame immediately
                processed_staged_df = pd.DataFrame(processed_df_staged_data_dict)
                print(f"[StationarityFrontend EXECUTE] Converted backend dict data to DataFrame. Shape: {processed_staged_df.shape}") 

                session_state.adf_results_staged = results_staged
                session_state.processed_staged_df = processed_staged_df # Store the DataFrame
                session_state.last_processed_staged_dataset_name_stationarity = selected_df_name # Store name of processed dataset
                # adf_results_staged = results_staged # Update local var not needed as we read from state below
                # processed_staged_df is already updated above

                # --- Initialize Selection State and Trigger Preview Update AFTER button click ---
                print("[StationarityFrontend EXECUTE] Initializing selection state and triggering preview update.")
                available_data_cols_init = processed_staged_df.columns.tolist()
                time_col_name_init = None
                if isinstance(processed_staged_df.index, pd.DatetimeIndex):
                     time_col_name_init = processed_staged_df.index.name if processed_staged_df.index.name else "时间索引"
                else: 
                    for col in available_data_cols_init:
                         if pd.api.types.is_datetime64_any_dtype(processed_staged_df[col]):
                              time_col_name_init = col
                              break
                available_options_init = available_data_cols_init[:]
                if time_col_name_init and time_col_name_init not in available_options_init:
                      available_options_init.insert(0, time_col_name_init)
                print(f"[DEBUG StationarityFrontend EXECUTE] available_options_init: {available_options_init}")
                
                selection_key = f"stationarity_selected_cols_{selected_df_name}"
                session_state[selection_key] = available_options_init[:] # Default to all
                session_state[f"{selection_key}_source"] = selected_df_name 
                print(f"[DEBUG StationarityFrontend EXECUTE] Initialized {selection_key} to all options.")
                
                # Call the update helper
                _update_stationarity_preview_df(session_state, selected_df_name)
                # --- End Update Block ---

                print(f"[StationarityFrontend EXECUTE] Processed stationarity for '{selected_df_name}'.")
                st.rerun() # Rerun to ensure results are displayed immediately below

            except Exception as e_process:
                st.error(f"处理数据集 '{selected_df_name}' 时发生错误: {e_process}")
                # Clear results on error to allow re-try
                session_state.pop('adf_results_staged', None)
                session_state.pop('processed_staged_df', None)
                session_state.pop('last_processed_staged_dataset_name_stationarity', None)
                session_state.pop('stationarity_tab_preview_df', None) 
                # Keep method/order states for user reference
                # session_state.pop(f'stationarity_processing_method_{selected_df_name}', None) 
                # session_state.pop(f'stationarity_diff_order_{selected_df_name}', None)
                return # Stop execution here if backend fails
                
    # --- Display Results (Only if available in session state) ---
    adf_results_staged = session_state.get('adf_results_staged', pd.DataFrame()) # Read from state
    processed_staged_df_display = session_state.get('processed_staged_df', pd.DataFrame()) # Read from state

    if not adf_results_staged.empty:
        st.markdown("---") # Keep separator before results table
     
        def style_stationarity(val):
            if val == '是': return 'color: green; font-weight: bold;'
            elif val == '否': return 'color: red; font-weight: bold;'
            else: return 'color: gray;'

        st.write("ADF 检验结果:")
        st.dataframe(adf_results_staged.style.map(style_stationarity, subset=['原始是否平稳', '最终是否平稳']))

        st.divider()

        # --- Display Processed Data Section ---
        processed_staged_df_display = session_state.get('processed_staged_df', pd.DataFrame())

        if not processed_staged_df_display.empty:
            st.markdown('**处理后的数据预览**')
            
            # --- Consistently define available columns FIRST --- #
            available_data_cols = processed_staged_df_display.columns.tolist()
            print(f"[DEBUG StationarityFrontend] Initial available_data_cols (from df.columns): {available_data_cols}")

            # --- Identify Time Column Name (index or column) --- #
            time_col_name = None
            if isinstance(processed_staged_df_display.index, pd.DatetimeIndex):
                time_col_name = processed_staged_df_display.index.name
                # If index has no name, we might need a placeholder or skip adding it
                if not time_col_name:
                    time_col_name = "时间索引" # Assign a placeholder if index name is None
                    print(f"[DEBUG StationarityFrontend] Identified unnamed DatetimeIndex. Using placeholder: {time_col_name}")
                else:
                    print(f"[DEBUG StationarityFrontend] Identified time column from index name: '{time_col_name}'")
            else:
                # Check columns for datetime type if no datetime index
                for col in available_data_cols:
                    if pd.api.types.is_datetime64_any_dtype(processed_staged_df_display[col]):
                        time_col_name = col
                        print(f"[DEBUG StationarityFrontend] Identified time column from column: '{time_col_name}'")
                        break 
            
            # --- Create the canonical list of ALL available options for multiselect --- #
            available_options = available_data_cols[:]
            if time_col_name and time_col_name not in available_options:
                 print(f"[DEBUG StationarityFrontend] Adding identified time column/index '{time_col_name}' to available options.")
                 available_options.insert(0, time_col_name) # Add time column/index name to the list of options
            
            print(f"[DEBUG StationarityFrontend] Final available_options for multiselect: {available_options}")

            # --- Column Selection for Preview/Download/Staging --- #
            st.markdown("选择要预览、下载或暂存的**处理后**变量:")
            
            selection_key = f"stationarity_selected_cols_{selected_df_name}"
            radio_key = f"stationarity_radio_select_{selected_df_name}"

            # --- Get original columns for comparison --- #
            original_df_staged = session_state.get('stationarity_selected_staged_data_df', pd.DataFrame())
            original_cols = original_df_staged.columns.tolist()
            # Identify the time column within the original columns if possible
            original_time_col = None
            for col in original_cols:
                 # Use the original staged df to check types
                 if pd.api.types.is_datetime64_any_dtype(original_df_staged[col]):
                      original_time_col = col
                      break
            print(f"[DEBUG StationarityFrontend] Identified original time column: {original_time_col}")
            # Exclude original time column from the list of *data* columns
            original_data_cols = [col for col in original_cols if col != original_time_col]

            # --- Prepare lists for Radio Button options --- #
            # Use 'available_options' and 'original_cols' to define these lists
            
            # Newly generated processed columns are those in available_options but NOT in original_cols
            newly_generated_cols = [col for col in available_options if col not in original_cols]
            print(f"[DEBUG StationarityFrontend] Newly generated columns: {newly_generated_cols}")
            
            # Define lists for radio options, ensuring time_col_name is handled correctly
            select_all_list = available_options[:] # All columns from the enhanced backend df
            
            # Start with original columns found in available options
            select_original_list = [col for col in available_options if col in original_cols] 
            # Ensure time column is included if it exists and isn't already there
            if time_col_name and time_col_name in available_options and time_col_name not in select_original_list:
                select_original_list.insert(0, time_col_name)

            # Start with newly generated columns
            select_processed_only_list = newly_generated_cols[:]
            # Ensure time column is included if it exists
            if time_col_name and time_col_name in available_options and time_col_name not in select_processed_only_list:
                 select_processed_only_list.insert(0, time_col_name)
            # Final check: ensure all items in processed_only list are actually in available_options (redundant but safe)
            select_processed_only_list = [col for col in select_processed_only_list if col in available_options]

            print(f"[DEBUG StationarityFrontend] Option List - All: {select_all_list}")
            print(f"[DEBUG StationarityFrontend] Option List - Original Only (forced time): {select_original_list}")
            print(f"[DEBUG StationarityFrontend] Option List - Processed Only (forced time): {select_processed_only_list}")

            option_dict = {
                "全选": select_all_list,
                "仅原始变量": select_original_list,
                "仅处理后变量": select_processed_only_list
            }

            # --- Radio button Callback MODIFIED --- #
            def update_multiselect_state_and_preview():
                """Callback to update the multiselect selection based on radio AND update sidebar preview."""
                chosen_option = session_state[radio_key]
                new_selection = option_dict.get(chosen_option, [])
                session_state[selection_key] = new_selection # Update selection state
                print(f"[DEBUG StationarityFrontend Radio CB] Radio '{chosen_option}' clicked. Updated {selection_key} to: {new_selection}")
                # --- Call the helper to update preview ---
                _update_stationarity_preview_df(session_state, selected_df_name) 

            st.radio(
                "快速选择变量组:", 
                options=list(option_dict.keys()), 
                key=radio_key,
                horizontal=True,
                label_visibility="collapsed",
                on_change=update_multiselect_state_and_preview # Use the MODIFIED callback 
            )

            # --- Multiselect with NEW on_change callback --- #
            current_selection = session_state.get(selection_key, [])
            # Filter against the canonical available_options list (good practice)
            valid_current_selection = [col for col in current_selection if col in available_options]
            if valid_current_selection != current_selection:
                 print(f"[WARN StationarityFrontend Multiselect] Correcting selection state. Was {current_selection}, now {valid_current_selection}")
                 session_state[selection_key] = valid_current_selection # Ensure state is valid before widget render

            print(f"[DEBUG StationarityFrontend Multiselect] Rendering. Options: {available_options}")
            print(f"[DEBUG StationarityFrontend Multiselect] Current selection from state ({selection_key}): {session_state.get(selection_key, [])}")

            selected_cols = st.multiselect(
                "处理后变量列表 (可调整):",
                options=available_options, 
                key=selection_key, 
                label_visibility="collapsed",
                # --- Add on_change to call the helper ---
                on_change=_update_stationarity_preview_df,
                args=(session_state, selected_df_name) # Pass necessary args to callback
            )

            # --- Action Block (Uses session_state['stationarity_tab_preview_df'] if needed, but mainly for staging/download) --- # 
            # The actual df_selected_for_action for staging/download can be derived again here, 
            # or we can potentially reuse session_state['stationarity_tab_preview_df'] if it's guaranteed to be up-to-date
            
            # Let's reuse the state for simplicity, assuming callbacks keep it fresh
            df_for_actions = session_state.get('stationarity_tab_preview_df', pd.DataFrame())
            
            if df_for_actions.empty:
                 # Check if the original processed df was empty or if selection resulted in empty
                 if processed_staged_df_display.empty:
                      st.info("没有处理后的数据可供操作。")
                 elif not current_selection: # Use the selection state
                      st.info("请从上方选择变量进行操作。")
                 else:
                      st.warning("当前选择未产生有效的可用数据进行操作。") # e.g., only invalid cols selected
            else:
                # --- 暂存与下载操作 --- #
                st.caption(f"已选择 {df_for_actions.shape[1]} 个有效变量进行操作。") # Use the preview df shape

                col_stage, col_download = st.columns(2)

                with col_stage:
                    st.markdown("##### 暂存选定数据")
                    new_staged_name = st.text_input(
                        "为暂存数据命名:",
                        value=f"{selected_df_name}_processed",
                        key=f"new_staged_name_input_{selected_df_name}"
                    ).strip()

                    if st.button("暂存选定数据", key=f"stage_selected_processed_button_{selected_df_name}"):
                        if not new_staged_name:
                            st.warning("请输入暂存数据集的名称。")
                        elif new_staged_name in session_state.get('staged_data', {}):
                            st.warning(f"名称 '{new_staged_name}' 已在暂存区存在，请输入不同名称。")
                        else:
                            # Use df_for_actions for staging
                            df_to_stage = df_for_actions.copy() 
                            
                            # --- Summary logic (already correct) --- #
                            timestamp_column = None
                            for col in df_to_stage.columns:
                                if col in df_to_stage and pd.api.types.is_datetime64_any_dtype(df_to_stage[col]):
                                    timestamp_column = col
                                    break
                            start_time, end_time, freq_desc = "未知", "未知", "未知"
                            if timestamp_column:
                                time_series = df_to_stage[timestamp_column].dropna()
                                if not time_series.empty:
                                    start_time = str(time_series.min().date()) 
                                    end_time = str(time_series.max().date())
                                    try:
                                        inferred = pd.infer_freq(time_series)
                                        freq_map = {'ME': '月度 (月末)', 'MS': '月度 (月初)', 'D': '每日', 'B': '工作日', 'W': '每周', 'QE': '季度 (月末)', 'QS': '季度 (月初)', 'YE': '年度 (年末)', 'YS': '年度 (年初)'}
                                        freq_desc = freq_map.get(inferred, inferred if inferred else "未知")
                                    except TypeError: freq_desc = "未知"
                            summary_info = {
                                'rows': df_to_stage.shape[0], 'cols': df_to_stage.shape[1],
                                'columns': df_to_stage.columns.tolist(), 'time_col_at_staging': timestamp_column or "未识别",
                                'data_start_time': start_time, 'data_end_time': end_time, 'data_frequency': freq_desc,
                                'first_col_display': df_to_stage.columns[0] if df_to_stage.shape[1] > 0 else 'N/A',
                                'last_col_display': df_to_stage.columns[-1] if df_to_stage.shape[1] > 0 else 'N/A'
                            }
                            # --- End summary logic --- #

                            if 'staged_data' not in session_state: session_state.staged_data = {}
                            session_state.staged_data[new_staged_name] = {
                                'df': df_to_stage, 'summary': summary_info, 'source': f'平稳性处理自: {selected_df_name}'
                            }
                            print(f"[StationarityFrontend] 成功暂存 '{new_staged_name}'，摘要信息: {summary_info}")
                            st.success(f"数据 '{new_staged_name}' 已成功暂存！")
                            st.rerun()

                with col_download:
                    st.markdown("##### 下载选定数据")
                    download_format_options = ("CSV", "Excel (.xlsx)")
                    download_format = st.radio("选择下载格式:", options=download_format_options, index=1, key=f"download_format_radio_processed_{selected_df_name}", horizontal=True)
                    base_filename_processed = st.text_input("下载文件名 (不含扩展名):", value=f"{selected_df_name}_selected_processed_stationary", key=f"download_filename_input_processed_{selected_df_name}").strip()

                    if not df_for_actions.empty and base_filename_processed:
                        df_to_download = df_for_actions.copy()
                        if download_format == "CSV":
                            file_name = f"{base_filename_processed}.csv"
                            data_bytes = df_to_download.to_csv(index=False).encode('utf-8-sig')
                            mime_type = "text/csv"
                            label_text = f"下载为 CSV"
                        else: # Excel
                            file_name = f"{base_filename_processed}.xlsx"
                            output_excel = io.BytesIO()
                            with pd.ExcelWriter(output_excel, engine='openpyxl') as writer:
                                df_to_download.to_excel(writer, sheet_name='选定处理后数据', index=False) 
                            output_excel.seek(0)
                            data_bytes = output_excel.getvalue()
                            mime_type = "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                            label_text = f"下载为 Excel (.xlsx)"

                        st.download_button(
                            label=label_text, data=data_bytes, file_name=file_name, mime=mime_type, 
                            key=f"download_button_{selected_df_name}_{download_format}"
                        )
                    else:
                        st.caption("请先选择变量并输入文件名。")
        else:
            st.caption("处理后的数据为空，无法进行选择或操作。")

    # --- 时间序列分解图 ---
    # Removed: Time series decomposition plot section as requested by the user.

# --- Test Block ---
if __name__ == '__main__':
    # This part is for standalone testing

    # Create a dummy session_state for testing
    class MockSessionState:
        def __init__(self):
            self._state = {}
        def get(self, key, default=None):
            return self._state.get(key, default)
        def __setitem__(self, key, value):
            self._state[key] = value
        def __getitem__(self, key):
            return self._state[key]
        def pop(self, key, default=None):
            return self._state.pop(key, default)
        # Add a way to check if a key is in st.session_state for the on_multiselect_change
        def __contains__(self, key):
            return key in self._state

    mock_st = st # Use the actual streamlit for UI elements in test
    # Patch st.session_state to use our mock for testing purposes
    # This is a bit tricky; usually st.session_state is managed by Streamlit itself.
    # For simple attribute access and item access, we might get away with direct assignment for testing.
    # However, Streamlit's actual SessionState is more complex.
    # A more robust way would be to pass mock_session_state directly where session_state is an argument
    # or use a more sophisticated patching mechanism if needed.
    
    # For this test, we will pass it explicitly where `session_state` is an argument
    # and for st.session_state, we hope direct item access in on_multiselect_change works with MockSessionState
    
    _original_session_state = st.session_state # Store original
    st.session_state = MockSessionState() # Patch with mock


    # Sample data
    data = {
        'NonStationary': np.cumsum(np.random.randn(100)),
        'Stationary': np.random.randn(100),
        'NeedsLog': np.exp(np.random.randn(100) * 0.1 + 5), # Positive values for log
        'DateTime': pd.date_range(start='2000-01-01', periods=100, freq='M') # Add a datetime column for testing default selection
    }
    sample_df = pd.DataFrame(data)
    # Set DateTime as index if you want it to be treated as index,
    # otherwise it will be a regular column and test_and_process_stationarity will handle it.
    # For this test, let's make it a regular column to test the logic inside test_and_process_stationarity
    # sample_df = sample_df.set_index('DateTime') 


    # Simulate data being selected in dashboard.py using the patched st.session_state
    st.session_state['stationarity_selected_staged_data_df'] = sample_df
    st.session_state['stationarity_selected_staged_data_name'] = "MySampleDatasetWithDate"
    
    st.set_page_config(layout="wide")
    st.sidebar.title("Test Stationarity Tab (Frontend)")
    test_alpha = st.sidebar.selectbox("Alpha", [0.01, 0.05, 0.10], index=1, key="test_alpha_selectbox")

    # Call display_stationarity_tab with the patched st and the mock_session_state instance
    display_stationarity_tab(st, st.session_state)

    st.sidebar.subheader("Session State Preview (Test):")
    # Access the internal _state dict of our MockSessionState for preview
    if hasattr(st.session_state, '_state'):
        st.sidebar.json(st.session_state._state, expanded=False)
    
    st.session_state = _original_session_state # Restore original 