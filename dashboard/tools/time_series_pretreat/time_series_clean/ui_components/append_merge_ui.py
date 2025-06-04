import streamlit as st
import pandas as pd
from typing import List, Dict, Optional, Tuple, Any

# Attempt to import backend functions
try:
    from ..utils.append_merge_data import append_dataframes, robust_infer_freq, resample_df_to_daily, perform_merge_and_postprocess, align_dataframe_frequency
    BACKEND_AVAILABLE = True
except ImportError as e:
    print(f"Import error: {e}") # For debugging in console
    BACKEND_AVAILABLE = False
    # This will be handled in the UI function

def show_append_merge_data_ui():
    """Main function to display the Append & Merge UI without internal tabs."""

    # Initialize session state keys
    if 'processed_data' not in st.session_state or not isinstance(st.session_state['processed_data'], dict):
        st.session_state['processed_data'] = {}
    if 'staged_data' not in st.session_state or not isinstance(st.session_state['staged_data'], dict):
        st.session_state['staged_data'] = {}
    # <<< 新增：初始化待上传文件区域 >>>
    if 'pending_upload_files' not in st.session_state or not isinstance(st.session_state['pending_upload_files'], dict):
        st.session_state['pending_upload_files'] = {}

    # Initialize a suffix for the file uploader key to allow resetting it
    st.session_state.setdefault('append_merge_uploader_key_suffix', 0)

    if not BACKEND_AVAILABLE:
        st.error("错误：数据追加/合并功能的后端模块未能成功导入。请检查项目结构和路径配置。")
        st.info("尝试导入路径: dashboard.tools.time_series_pretreat.time_series_clean.utils.append_merge_data")
        return

    # --- Helper Functions (moved to the beginning of the main UI function) ---
    def format_dataset_name_for_display(key_name: str) -> str:
        """Strips prefixes like '[暂存] ' and suffixes like ' (列: ...)' for cleaner display."""
        if isinstance(key_name, str):
            name_part = key_name
            if name_part.startswith("[暂存] "):
                name_part = name_part.replace("[暂存] ", "")
            elif name_part.startswith("[处理后] "):
                name_part = name_part.replace("[处理后] ", "")
        
            if " (" in name_part: # Remove the " (列: ...)" part
                return name_part.split(" (")[0]
            return name_part
        return key_name

    def get_dataset_metadata(selected_key: str, source_dfs_map: dict) -> dict:
        """Helper to retrieve DataFrame and time_col for a selected dataset key."""
        metadata = {'time_col': None, 'df': None}
        base_name = None
        df_from_source = source_dfs_map.get(selected_key)
        metadata['df'] = df_from_source # Get the actual DataFrame

        if selected_key.startswith("[暂存] "):
            base_name = selected_key.replace("[暂存] ", "").split(" (")[0]
            if base_name in st.session_state.staged_data:
                entry = st.session_state.staged_data[base_name]
                if isinstance(entry, dict):
                    metadata['time_col'] = entry.get('time_col')
        elif selected_key.startswith("[处理后] "):
            base_name = selected_key.replace("[处理后] ", "").split(" (")[0]
            if base_name in st.session_state.processed_data:
                entry = st.session_state.processed_data[base_name]
                if isinstance(entry, dict):
                    metadata['time_col'] = entry.get('time_col')
        return metadata

    # --- 1. Upload Files (Optional) ---
    st.subheader("上传数据文件（可选）")
    st.markdown("此区域上传的文件将首先显示在下方预览区，点击“添加至暂存区”后方可用于后续操作。") # <<< 修改说明文字
    
    # <<< 修改：重命名函数并调整其逻辑 >>>
    def _handle_files_to_pending_uploads(): 
        current_uploader_key = f"multi_file_uploader_to_pending_{st.session_state.get('append_merge_uploader_key_suffix', 0)}"
        uploaded_files_from_state = st.session_state.get(current_uploader_key)

        if not uploaded_files_from_state:
            return

        # Ensure pending_upload_files is initialized (should be done at the top of show_append_merge_data_ui)
        if 'pending_upload_files' not in st.session_state or not isinstance(st.session_state['pending_upload_files'], dict):
            st.session_state['pending_upload_files'] = {}

        for uploaded_file in uploaded_files_from_state:
            file_name = uploaded_file.name
            df_upload = None
            auto_detected_time_col = None
            error_occurred = False

            try:
                if file_name.endswith('.csv'):
                    df_upload = pd.read_csv(uploaded_file)
                elif file_name.endswith(('.xls', '.xlsx')):
                    df_upload = pd.read_excel(uploaded_file)
                else:
                    st.warning(f"文件 '{file_name}' 的格式不受支持，已跳过。")
                    continue 

                if df_upload is not None and not df_upload.empty:
                    if df_upload.shape[1] > 0:
                        potential_time_col = df_upload.columns[0]
                        try:
                            # Attempt to convert, but don't raise error here, just check
                            pd.to_datetime(df_upload[potential_time_col], errors='raise')
                            auto_detected_time_col = potential_time_col
                        except Exception:
                            # If first col is not time, we still load it, time_col will be None
                            # User can set it later if they add to staged area
                            auto_detected_time_col = None 
                            st.info(f"文件 '{file_name}' 的第一列 '{potential_time_col}' 未能自动识别为时间戳。您可以在添加到暂存区后手动指定时间列。")
                    else:
                        st.error(f"文件 '{file_name}' 为空或不包含任何列。此文件未加载到预览。")
                        error_occurred = True
                else:
                    st.error(f"文件 '{file_name}' 读取后为空。此文件未加载到预览。")
                    error_occurred = True

            except Exception as e_read:
                st.error(f"读取文件 '{file_name}' 失败: {e_read}")
                error_occurred = True
        
            if not error_occurred and df_upload is not None:
                # Store basic info, detailed summary can be when adding to staged
                st.session_state['pending_upload_files'][file_name] = {
                    'df': df_upload.copy(),
                    'detected_time_col_on_upload': auto_detected_time_col,
                    'original_file_name': uploaded_file.name # Store original for reference
                }
            elif not error_occurred and df_upload is None and not file_name.endswith(('.csv', '.xlsx', '.xls')):
                pass 
            elif not error_occurred: 
                 st.warning(f"文件 '{file_name}' 未能成功加载到预览区，原因未知。")

    # --- Dynamic key for file uploader ---
    # <<< 修改：uploader_key 名称对应新回调 >>>
    uploader_key = f"multi_file_uploader_to_pending_{st.session_state.get('append_merge_uploader_key_suffix', 0)}"

    uploaded_files = st.file_uploader(
        "选择一个或多个CSV/Excel文件进行预览:", # <<< 修改提示文字
        type=['csv', 'xlsx'],
        accept_multiple_files=True,
        key=uploader_key, 
        on_change=_handle_files_to_pending_uploads # <<< 修改：使用新的回调
    )

    # --- Reset Button --- 
    if st.button("重置", key="reset_append_merge_tab_state"): 
        st.session_state.append_merge_uploader_key_suffix += 1
        
        # <<< 新增：清空 pending_upload_files >>>
        if 'pending_upload_files' in st.session_state:
            st.session_state.pending_upload_files = {}

        keys_to_reset_to_none = [
            # Append related keys
            'append_df_left_key', 'append_df_right_key', # Potentially old, keeping for now
            'append_primary_df_key', 'append_secondary_df_key',
            'append_time_col_left', 'append_time_col_right',
            'append_how_method', 'append_fill_method',
            'appended_df_for_display', 'appended_df_name_suggestion',
            'download_appended_filename_input', 'new_appended_name_input',
            'append_resample_target_freq_for_display',
            
            # Merge related keys
            'merge_df1_key', 'merge_df2_key',
            'merge_time_col1', 'merge_time_col2',
            'merge_on_cols1_selection', 'merge_on_cols2_selection',
            'merge_how_method', 
            'temp_merged_df', 'temp_merged_df_freq',
            'new_merged_df_name_input_key',
            'merge_resample_target_freq_for_display',

            # UI interaction state (add more as identified)
            # These might be better deleted if they control visibility/expansion rather than selected values
            # 'append_options_expanded', 'merge_options_expanded',
            # 'upload_section_expanded', 
            # 'current_operation_am'
        ]
        for key in keys_to_reset_to_none:
            if key in st.session_state:
                st.session_state[key] = None # <<< 修改：设置为 None
        
        # Keys that are better deleted (e.g. for expanders, or if None is a valid selection)
        keys_to_actually_delete = [
            'append_options_expanded', 'merge_options_expanded',
            'upload_section_expanded', 'current_operation_am',
            'appended_df_for_display', # This stores a dataframe, should be deleted
            'temp_merged_df' # This stores a dataframe, should be deleted
        ]
        for key in keys_to_actually_delete:
            if key in st.session_state:
                del st.session_state[key]

        st.rerun()

    # --- Uploaded Data Preview and Add to Staged Area --- 
    # <<< 修改：标题和数据源，并添加“添加至暂存区”按钮 >>>
    if 'pending_upload_files' in st.session_state and st.session_state.pending_upload_files:
        st.subheader("上传数据预览") 
        pending_items = list(st.session_state.pending_upload_files.items())
        num_pending_items = len(pending_items)

        if num_pending_items > 0:
            # Display in multiple columns if many files, or single if few.
            # Max 3 columns for better readability of dataframe previews.
            cols_display = st.columns(min(num_pending_items, 3))
            item_idx = 0
            for name, pending_info in pending_items:
                with cols_display[item_idx % min(num_pending_items, 3)]:
                    st.markdown(f"**{name}**")
                    df_pending = pending_info['df']
                    st.dataframe(df_pending.head())
                    
                    # Button to add to staged area
                    add_to_staged_key = f"add_to_staged_btn_{name.replace('.', '_').replace(' ', '_')}" # Sanitize name for key
                    if st.button(f"添加到暂存区", key=add_to_staged_key):
                        if 'staged_data' not in st.session_state:
                            st.session_state.staged_data = {}
                        
                        # Basic processing for staged data (can be expanded)
                        staged_df = pending_info['df'].copy()
                        time_col_staged = pending_info.get('detected_time_col_on_upload')
                        
                        # Re-generate summary for staged data (or enhance existing)
                        all_columns_staged = staged_df.columns.tolist()
                        first_col_disp_staged = all_columns_staged[0] if all_columns_staged else 'N/A'
                        last_col_disp_staged = all_columns_staged[-1] if len(all_columns_staged) > 1 else first_col_disp_staged
                        if not all_columns_staged: last_col_disp_staged = 'N/A'

                        summary_staged = {
                            'rows': staged_df.shape[0],
                            'cols': staged_df.shape[1],
                            'columns': all_columns_staged,
                            'first_col_display': first_col_disp_staged,
                            'last_col_display': last_col_disp_staged,
                            'detected_time_col_on_upload': time_col_staged # Keep original detection
                        }
                        try:
                            if time_col_staged and time_col_staged in staged_df.columns:
                                ts_summary_staged = pd.to_datetime(staged_df[time_col_staged], errors='coerce').dropna()
                                if not ts_summary_staged.empty:
                                    summary_staged['data_start_time'] = str(ts_summary_staged.min().date() if hasattr(ts_summary_staged.min(), 'date') else ts_summary_staged.min())
                                    summary_staged['data_end_time'] = str(ts_summary_staged.max().date() if hasattr(ts_summary_staged.max(), 'date') else ts_summary_staged.max())
                                    summary_staged['data_frequency'] = str(pd.infer_freq(ts_summary_staged.sort_values()) or '未知')
                                else: # Fallback if time col is empty or all NaT
                                    summary_staged.update({'data_start_time': '未知', 'data_end_time': '未知', 'data_frequency': '未知'})
                            else: # No time col detected or provided
                                summary_staged.update({'data_start_time': 'N/A', 'data_end_time': 'N/A', 'data_frequency': 'N/A'})
                        except Exception:
                            summary_staged.update({'data_start_time': '错误', 'data_end_time': '错误', 'data_frequency': '错误'})

                        st.session_state.staged_data[name] = {
                            'df': staged_df,
                            'time_col': time_col_staged, # This might be None, user handles in clean/process tab
                            'source': 'upload_pending',
                            'summary': summary_staged
                        }
                        # Remove from pending after adding to staged
                        del st.session_state.pending_upload_files[name]
                        st.success(f"'{name}' 已添加到暂存区！")
                        st.rerun()
                    item_idx += 1
        else:
            st.info("当前没有待预览的文件。请通过上方按钮上传文件。")
    
    st.divider()

    # --- 2. Data Append Section --- 
    st.subheader("数据追加")
    st.markdown("从下方选择已加载或已暂存的数据集进行纵向追加。所有选中的数据集必须拥有完全相同的列结构。")

    available_dfs_for_append = {}
    options_for_multiselect = []

    if 'processed_data' in st.session_state:
        for name, data_dict in st.session_state.processed_data.items():
            internal_key = None
            df_to_store = None
            if isinstance(data_dict, dict) and 'df' in data_dict:
                df_to_store = data_dict['df']
                internal_key = f"[处理后] {name} (列: {', '.join(df_to_store.columns.tolist()) if df_to_store is not None else 'N/A'})"
            elif isinstance(data_dict, pd.DataFrame): # Direct DataFrame case
                df_to_store = data_dict
                internal_key = f"[处理后] {name} (列: {', '.join(df_to_store.columns.tolist())})"
            
            if internal_key and df_to_store is not None:
                available_dfs_for_append[internal_key] = df_to_store
                options_for_multiselect.append(internal_key)

    if 'staged_data' in st.session_state:
        for name, data_dict in st.session_state.staged_data.items():
            if isinstance(data_dict, dict) and 'df' in data_dict: # Ensure it's the expected dict structure
                df_to_store = data_dict['df']
                internal_key = f"[暂存] {name} (列: {', '.join(df_to_store.columns.tolist())})"
                available_dfs_for_append[internal_key] = df_to_store
                options_for_multiselect.append(internal_key)

    if not available_dfs_for_append:
        st.info("没有可用于追加的数据。请先在数据清洗模块处理数据或直接上传文件到暂存区。")
    else:
        left_append_controls, right_append_feedback = st.columns([2, 3])
        with left_append_controls:
            st.markdown("##### **选择数据集**")
            selected_dfs_internal_keys = st.multiselect(
                "选择要追加的数据集 (按选择顺序追加):",
                options=options_for_multiselect, 
                format_func=lambda internal_key: format_dataset_name_for_display(internal_key.split(' (列:')[0]), 
                key="append_multiselect_v2" # Changed key to avoid conflict if old one persists
            )

            if st.button("执行追加", key="execute_append_button_v2"):
                if len(selected_dfs_internal_keys) < 2:
                    right_append_feedback.error("请至少选择两个数据集进行追加操作。")
                    st.session_state.appended_df_for_display = None # Clear previous results
                else:
                    dfs_to_append = [available_dfs_for_append[key] for key in selected_dfs_internal_keys if key in available_dfs_for_append]
                    if len(dfs_to_append) == len(selected_dfs_internal_keys):
                        appended_df = append_dataframes(dfs_to_append)
                        if appended_df is not None:
                            right_append_feedback.success("数据追加成功！")
                            st.session_state.appended_df_for_display = appended_df
                            st.session_state.appended_df_name_suggestion = "_appended_".join([
                                format_dataset_name_for_display(key.split(' (列:')[0]) for key in selected_dfs_internal_keys
                            ])
                        else:
                            right_append_feedback.error("数据追加失败，内部函数未返回有效DataFrame。")
                            st.session_state.appended_df_for_display = None
                    else:
                        right_append_feedback.error("一个或多个选中的数据集在字典中未找到，追加失败。")
                        st.session_state.appended_df_for_display = None
            
        # Display area for appended data (moved to right column)
        if 'appended_df_for_display' in st.session_state and st.session_state.appended_df_for_display is not None:
            controls_col, preview_col = st.columns([2, 3]) # Adjust ratio as needed

            with controls_col: # All controls will now be in the left column
                st.markdown("##### **操作与保存**") # Title for the control section

                # --- Download Appended Data ---
                st.markdown("###### 下载追加数据")
                # NEW: Text input for download filename
                download_filename_input = st.text_input(
                    "输入下载文件名:",
                    value=st.session_state.get('appended_df_name_suggestion', 'appended_data'),
                    key='download_filename_input_append_v2'
                )
                
                csv_data_appended = st.session_state.appended_df_for_display.to_csv(index=False).encode('utf-8-sig')
                # Use the filename from the text input
                download_filename_to_use = f"{download_filename_input.strip()}.csv" if download_filename_input.strip() else "appended_data.csv"

                st.download_button(
                    label="下载追加数据 (CSV)",
                    data=csv_data_appended,
                    file_name=download_filename_to_use,
                    mime='text/csv',
                    key='download_appended_csv_v2_modified' # Ensure key is unique
                )
                
                               
                # --- Save to Staged Area ---
                st.markdown("###### 保存到暂存区")
                
                suggested_save_name = st.session_state.get('appended_df_name_suggestion', 'appended_data')
                new_appended_name_input = st.text_input(
                    "为追加后的数据集命名:", 
                    value=suggested_save_name,
                    key='appended_name_input_v2_modified' # Ensure key is unique
                )

                if st.button("保存追加结果到暂存区", key='save_appended_to_staged_v2_modified'): # Ensure key is unique
                    if not new_appended_name_input:
                        st.error("请输入数据集名称。")
                    elif new_appended_name_input in st.session_state.get('staged_data', {}) or \
                         new_appended_name_input in st.session_state.get('processed_data', {}):
                        st.error(f"名称 '{new_appended_name_input}' 已存在，请使用其他名称。")
                    elif 'appended_df_for_display' not in st.session_state or st.session_state.appended_df_for_display is None:
                        st.error("没有可保存的追加数据。")
                    else:
                        df_to_stage = st.session_state.appended_df_for_display.copy()
                        # Appended DFs from concat(ignore_index=True) will have RangeIndex.
                        # Try to identify a time column for metadata purposes.
                        time_col_for_metadata = None
                        inferred_freq_appended = "未知"
                        
                        if 'Time' in df_to_stage.columns and pd.api.types.is_datetime64_any_dtype(df_to_stage['Time']):
                            time_col_for_metadata = 'Time'
                        elif df_to_stage.shape[1] > 0 and pd.api.types.is_datetime64_any_dtype(df_to_stage.iloc[:, 0]):
                            # Fallback to first column if it's datetime
                            time_col_for_metadata = df_to_stage.columns[0]
                        
                        if time_col_for_metadata:
                            try:
                                # Ensure it's sorted if we infer freq based on it
                                time_series_for_freq = pd.to_datetime(df_to_stage[time_col_for_metadata], errors='coerce').dropna().sort_values()
                                if not time_series_for_freq.empty:
                                    inferred_freq_appended = robust_infer_freq(time_series_for_freq)
                            except Exception:
                                inferred_freq_appended = "未知 (推断出错)"
                        else: # If no time column identified, set to first col name for display consistency
                            time_col_for_metadata = df_to_stage.columns[0] if df_to_stage.shape[1] > 0 else 'N/A'

                        all_cols_appended = df_to_stage.columns.tolist()
                        first_col_appended = time_col_for_metadata
                        last_col_appended = all_cols_appended[-1] if len(all_cols_appended) > 1 else first_col_appended
                        if not all_cols_appended: last_col_appended = 'N/A'
                        
                        data_start_time_app = 'N/A'
                        data_end_time_app = 'N/A'
                        if time_col_for_metadata != 'N/A' and time_col_for_metadata in df_to_stage.columns and pd.api.types.is_datetime64_any_dtype(pd.to_datetime(df_to_stage[time_col_for_metadata], errors='coerce')):
                            valid_dates = pd.to_datetime(df_to_stage[time_col_for_metadata], errors='coerce').dropna()
                            if not valid_dates.empty:
                                data_start_time_app = str(valid_dates.min().date())
                                data_end_time_app = str(valid_dates.max().date())

                        st.session_state.staged_data[new_appended_name_input] = {
                            'df': df_to_stage, # Already has RangeIndex
                            'rows': df_to_stage.shape[0],
                            'cols': df_to_stage.shape[1],
                            'columns': all_cols_appended,
                            'time_col': time_col_for_metadata, 
                            'auto_detected_time_col': time_col_for_metadata, # Best guess for appended data
                            'user_selected_time_col': None, # User hasn't selected for appended data
                            'freq_inferred': inferred_freq_appended,
                            'file_name': new_appended_name_input,
                            'first_col_display': first_col_appended,
                            'last_col_display': last_col_appended,
                            'source': 'append_tool',
                            'summary': { 
                                'rows': df_to_stage.shape[0],
                                'cols': df_to_stage.shape[1],
                                'columns': all_cols_appended,
                                'time_col_for_summary': time_col_for_metadata, # Use the one identified
                                'data_start_time': data_start_time_app,
                                'data_end_time': data_start_time_app,
                                'data_frequency': inferred_freq_appended
                            }
                        }
                        st.success(f"追加后的数据集 '{new_appended_name_input}' 已保存到暂存区。")
                        st.session_state.appended_df_for_display = None # Clear after saving
                        st.session_state.appended_df_name_suggestion = None
                        st.rerun() # Rerun to refresh UI and clear selections

            with preview_col: # Preview will be in the right column
                if 'appended_df_for_display' in st.session_state and st.session_state.appended_df_for_display is not None: # Check again as it might be cleared by save
                    st.markdown("##### **追加结果预览**")
                    st.dataframe(st.session_state.appended_df_for_display) # Display full dataframe

    st.divider()

    # --- 3. Data Merge Section ---
    st.subheader("数据合并")
    st.markdown("从下方选择两个已加载或已暂存的数据集进行横向合并（类似SQL JOIN）。")

    # Define main columns for the merge section layout
    left_controls, right_feedback = st.columns([2, 3]) # Adjust ratio as needed

    # available_dfs_for_append is defined earlier in the function
    if len(available_dfs_for_append) < 1:
        left_controls.info("没有可用于合并的数据。请先处理或上传数据。")
        # To prevent NameError if available_dfs_for_append is empty later
        left_df_name = None
        right_df_name = None
        left_df_to_merge = None
        right_df_to_merge = None
        left_time_col = None
        right_time_col = None
        left_freq_str = "未知"
        right_freq_str = "未知"
        common_cols_str = "未选择"
    else:
        with left_controls:
            st.markdown("##### **选择数据集与合并参数**")
            left_df_name = st.selectbox(
                "选择左侧数据集:", 
                options=list(available_dfs_for_append.keys()), 
                key="left_df_merge_select",
                format_func=format_dataset_name_for_display,
                placeholder="选择左侧数据集...",
                index=None  # <<< 新增 index=None
            )
            right_df_name = st.selectbox(
                "选择右侧数据集:", 
                options=list(available_dfs_for_append.keys()), 
                key="right_df_merge_select",
                format_func=format_dataset_name_for_display,
                placeholder="选择右侧数据集...",
                index=None  # <<< 新增 index=None
            )

            # Immediately retrieve dataframes and calculate common_cols based on selections
            left_meta = get_dataset_metadata(left_df_name, available_dfs_for_append) if left_df_name else {'df': None, 'time_col': None}
            right_meta = get_dataset_metadata(right_df_name, available_dfs_for_append) if right_df_name else {'df': None, 'time_col': None}
            left_df_for_common_cols = left_meta['df']
            right_df_for_common_cols = right_meta['df']
            
            common_cols = []
            if left_df_for_common_cols is not None and right_df_for_common_cols is not None:
                common_cols = list(set(left_df_for_common_cols.columns) & set(right_df_for_common_cols.columns))
            # common_cols_str is for display in right_feedback, calculated later

            merge_how_options = ["left", "right", "outer", "inner"]
            merge_how = st.selectbox(
                "选择合并方式 (Merge Type):", 
                merge_how_options, 
                key="merge_how_select", 
                placeholder="选择合并类型...",
                index=None  # <<< 新增 index=None
            )
            
            merge_on_cols = st.multiselect(
                "选择合并键 (留空则基于重采样后的时间索引合并):", 
                options=common_cols, 
                default=[], 
                key="merge_on_cols_select"
            )

            # Removed: st.markdown("--- ")
            execute_merge_button = st.button("执行日度重采样与合并", key="execute_resample_merge_button")

        # This data retrieval is now partially duplicated from above for common_cols, 
        # but full df and time_col are needed for frequency display and merge logic.
        # We will use the 'left_meta' and 'right_meta' from above, but re-assign to 'left_df', 'right_df' etc. for clarity in existing code.
        left_df = left_meta['df']
        left_time_col = left_meta['time_col']
        right_df = right_meta['df']
        right_time_col = right_meta['time_col']

        # Frequency string calculation (can stay here for right_feedback)
        left_freq_str, right_freq_str = "未知", "未知"
        if left_df is not None and left_time_col and left_time_col in left_df.columns:
            try:
                time_series_left = pd.to_datetime(left_df[left_time_col], errors='coerce')
                left_freq_str = robust_infer_freq(time_series_left)
            except Exception:
                left_freq_str = "未知 (推断出错)"
        
        if right_df is not None and right_time_col and right_time_col in right_df.columns:
            try:
                time_series_right = pd.to_datetime(right_df[right_time_col], errors='coerce')
                right_freq_str = robust_infer_freq(time_series_right)
            except Exception:
                right_freq_str = "未知 (推断出错)"
        
        # common_cols_str for display in right_feedback
        # common_cols calculation is now done above for the multiselect widget
        common_cols_str = ", ".join(common_cols) if common_cols else "无共同列"

        # Removed redundant 'with left_controls:' block that previously contained another multiselect

    # Display pre-check info in the right column
    with right_feedback:
        st.markdown("##### **数据预览与合并状态**")
        if left_df_name and right_df_name and left_df is not None and right_df is not None:
            st.markdown(f"**左侧: {format_dataset_name_for_display(left_df_name)}**")
            st.markdown(f"&nbsp;&nbsp;&nbsp;&nbsp;时间频率: `{left_freq_str}`")
            
            st.markdown(f"**右侧: {format_dataset_name_for_display(right_df_name)}**")
            st.markdown(f"&nbsp;&nbsp;&nbsp;&nbsp;时间频率: `{right_freq_str}`")

            st.markdown(f"**共同列名:** `{common_cols_str}`")
        else:
            st.info("请从左侧选择有效的数据集以查看详情和执行合并。")
        # Removed: st.markdown("--- ") # Separator for feedback messages
    
    st.divider()

    # --- Logic after execute button is pressed ---
    if len(available_dfs_for_append) >= 1 and execute_merge_button:
        if left_df is not None and right_df is not None and left_time_col and right_time_col:
            
            # Backend function now handles resampling, merging, and post-processing
            merged_df, inferred_freq = perform_merge_and_postprocess(
                left_df_raw=left_df, 
                left_time_col=left_time_col,
                left_df_name=format_dataset_name_for_display(left_df_name),
                right_df_raw=right_df, 
                right_time_col=right_time_col,
                right_df_name=format_dataset_name_for_display(right_df_name),
                merge_how=st.session_state.merge_how_select.lower(), # ensure lowercase 'inner', 'left', etc.
                merge_on_cols=st.session_state.get('merge_on_cols_select', None),
                status_container=right_feedback # Pass the UI container for messages
            )

            if merged_df is not None:
                st.session_state.temp_merged_df = merged_df
                st.session_state.temp_merged_df_freq = inferred_freq
                # Success message is now handled by the backend or can be added here based on return
                # right_feedback.success("数据已成功合并。结果已暂存，请在左侧命名并保存。") # Example if backend doesn't show final success
                st.rerun() # Rerun to update UI with merged_df and hide/show relevant sections
            else:
                # Error messages are now mostly handled by the backend function within status_container
                # If merged_df is None, an error has occurred and should have been displayed.
                pass 

        else:
            left_controls.error("请确保已选择有效的左右数据集，并且它们的时间列已被正确识别。")

    # --- Save Merged DataFrame UI (conditionally displayed if temp_merged_df exists) ---
    if 'temp_merged_df' in st.session_state and st.session_state.temp_merged_df is not None:
        # Create two columns for layout: one for saving options, one for preview/feedback
        left_save_options, right_feedback = st.columns([2, 3])

        with left_save_options:
            # --- New section for Frequency Alignment ---
            st.markdown("##### **频率对齐 (可选)**")
            # UI for frequency alignment options
            target_freq_options = ["日", "周", "月", "季度", "年"]
            # Ensure session state keys exist before accessing them for default values
            if 'align_target_freq' not in st.session_state: st.session_state.align_target_freq = "周"
            if 'align_to_period' not in st.session_state: st.session_state.align_to_period = "周期最后一天"
            if 'align_day_type' not in st.session_state: st.session_state.align_day_type = "自然日"

            align_target_freq = st.selectbox(
                "选择对齐频率:", 
                target_freq_options, 
                index=target_freq_options.index(st.session_state.align_target_freq), 
                key='align_target_freq_select'
            )
            
            # Place "对齐到" and "日期类型" in the same row
            col_align_rule1, col_align_rule2 = st.columns(2)
            with col_align_rule1:
                align_to_period = st.radio(
                    "对齐到:", 
                    ("周期第一天", "周期最后一天"), 
                    index=["周期第一天", "周期最后一天"].index(st.session_state.align_to_period), 
                    key='align_to_period_radio'
                )
            with col_align_rule2:
                align_day_type = st.radio(
                    "日期类型:", 
                    ("自然日", "工作日"), 
                    index=["自然日", "工作日"].index(st.session_state.align_day_type), 
                    key='align_day_type_radio'
                )

            execute_align_button = st.button("执行频率对齐", key='execute_align_button')

            if execute_align_button and 'temp_merged_df' in st.session_state and st.session_state.temp_merged_df is not None:
                df_to_align = st.session_state.temp_merged_df
                if not df_to_align.empty:
                    # Use right_feedback (or a dedicated container for alignment feedback) 
                    # for messages from align_dataframe_frequency
                    aligned_df = align_dataframe_frequency(
                        df=df_to_align,
                        target_freq_ui=st.session_state.align_target_freq_select,
                        align_to_ui=st.session_state.align_to_period_radio,
                        day_type_ui=st.session_state.align_day_type_radio,
                        status_container=right_feedback # Use the existing feedback column for messages
                    )
                    if aligned_df is not None:
                        st.session_state.temp_merged_df = aligned_df
                        # Re-infer frequency after alignment
                        if not aligned_df.empty and isinstance(aligned_df.index, pd.DatetimeIndex):
                            st.session_state.temp_merged_df_freq = robust_infer_freq(aligned_df.index.to_series())
                        elif aligned_df.empty:
                            st.session_state.temp_merged_df_freq = "未知 (对齐后为空)"
                        else:
                            st.session_state.temp_merged_df_freq = "未知 (对齐后索引无效)"
                        
                        right_feedback.success("频率对齐成功。预览已更新。") # Or rely on backend message
                        st.rerun()
                    # else: error message should have been displayed by the backend function
                else:
                    right_feedback.warning("暂存的合并数据为空，无法执行频率对齐。")
            elif execute_align_button:
                 right_feedback.error("没有可供对齐的暂存合并数据。请先合并数据。")

            st.divider()

            st.markdown("##### **保存合并结果**")
            new_dataset_name_default = "merged_aligned_data"
            if 'current_merged_dataset_name_input' not in st.session_state:
                st.session_state.current_merged_dataset_name_input = new_dataset_name_default

            # Input for naming the dataset when saving to staged area
            staged_name_for_merged_df = st.text_input(
                "为保存在暂存区的结果命名:", 
                value=st.session_state.current_merged_dataset_name_input,
                key="new_merged_df_name_input_key" 
            )
            
            if st.button("保存合并结果到暂存区", key="save_merged_df_to_staged_button"):
                if staged_name_for_merged_df: # Use the correct variable name
                    if staged_name_for_merged_df in st.session_state.staged_data or staged_name_for_merged_df in st.session_state.processed_data:
                        right_feedback.error(f"名称 '{staged_name_for_merged_df}' 已存在于暂存区或已处理数据中，请使用其他名称。")
                    else:
                        # --- BEGIN MODIFIED LOGIC FOR STAGING --- 
                        df_temp_for_staging = st.session_state.temp_merged_df.copy()
                        
                        time_col_for_metadata = None
                        if isinstance(df_temp_for_staging.index, pd.DatetimeIndex):
                            if df_temp_for_staging.index.name is None:
                                df_temp_for_staging.index.name = 'Time' 
                            time_col_for_metadata = df_temp_for_staging.index.name
                        else:
                            temp_reset_df = df_temp_for_staging.reset_index()
                            if 'Time' in temp_reset_df.columns and pd.api.types.is_datetime64_any_dtype(temp_reset_df['Time']):
                                time_col_for_metadata = 'Time'
                            elif 'index' in temp_reset_df.columns and pd.api.types.is_datetime64_any_dtype(temp_reset_df['index']):
                                 time_col_for_metadata = 'index'
                            else: 
                                 time_col_for_metadata = temp_reset_df.columns[0] if len(temp_reset_df.columns) > 0 else 'N/A'

                        merged_df_to_save = df_temp_for_staging.reset_index()
                        if time_col_for_metadata not in merged_df_to_save.columns:
                            time_col_for_metadata = merged_df_to_save.columns[0] if len(merged_df_to_save.columns) > 0 else 'N/A'
                        
                        all_cols_merged = merged_df_to_save.columns.tolist()
                        first_col_merged = time_col_for_metadata 
                        last_col_merged = all_cols_merged[-1] if len(all_cols_merged) > 1 else first_col_merged
                        if not all_cols_merged: last_col_merged = 'N/A'

                        st.session_state.staged_data[staged_name_for_merged_df] = { # Use the correct variable
                            'df': merged_df_to_save, 
                            'rows': merged_df_to_save.shape[0],
                            'cols': merged_df_to_save.shape[1],
                            'columns': all_cols_merged,
                            'time_col': time_col_for_metadata, 
                            'auto_detected_time_col': time_col_for_metadata,
                            'user_selected_time_col': time_col_for_metadata, 
                            'freq_inferred': st.session_state.get('temp_merged_df_freq', '未知'),
                            'file_name': staged_name_for_merged_df, # Use the correct variable
                            'first_col_display': first_col_merged,
                            'last_col_display': last_col_merged,
                            'source': 'merge_tool',
                            'summary': { 
                                'rows': merged_df_to_save.shape[0],
                                'cols': merged_df_to_save.shape[1],
                                'columns': all_cols_merged,
                                'first_col_display': first_col_merged,
                                'last_col_display': last_col_merged,
                                'data_start_time': str(pd.to_datetime(merged_df_to_save[time_col_for_metadata], errors='coerce').min().date()) if time_col_for_metadata in merged_df_to_save and not merged_df_to_save.empty and pd.api.types.is_datetime64_any_dtype(pd.to_datetime(merged_df_to_save[time_col_for_metadata], errors='coerce')) else 'N/A',
                                'data_end_time': str(pd.to_datetime(merged_df_to_save[time_col_for_metadata], errors='coerce').max().date()) if time_col_for_metadata in merged_df_to_save and not merged_df_to_save.empty and pd.api.types.is_datetime64_any_dtype(pd.to_datetime(merged_df_to_save[time_col_for_metadata], errors='coerce')) else 'N/A',
                                'data_frequency': st.session_state.get('temp_merged_df_freq', '未知')
                            }
                        }
                        right_feedback.success(f"合并后的日度数据集 '{staged_name_for_merged_df}' 已保存到暂存区。") # Use the correct variable
                        del st.session_state.temp_merged_df
                        if 'temp_merged_df_freq' in st.session_state: del st.session_state.temp_merged_df_freq
                        st.rerun() # Rerun to reflect changes and remove temp_merged_df UI elements
                else:
                    right_feedback.error("请输入合并后数据集的名称。")
            
            # --- Download Button Section ---
            # Input for naming the downloaded CSV file
            download_filename_suggestion = staged_name_for_merged_df if staged_name_for_merged_df else new_dataset_name_default
            download_csv_filename = st.text_input(
                "为下载的CSV文件命名 (不含.csv后缀):",
                value=download_filename_suggestion.replace('.csv', ''), # Default to staged name, remove .csv if present
                key='download_merged_csv_filename_key'
            )

            df_for_csv_export = st.session_state.temp_merged_df.reset_index()
            original_index_name = st.session_state.temp_merged_df.index.name 
            if 'index' in df_for_csv_export.columns and original_index_name is None:
                new_time_column_name = 'Time' 
                if new_time_column_name in df_for_csv_export.columns and df_for_csv_export.columns.get_loc(new_time_column_name) != 0:
                     new_time_column_name = 'Original_Index_Time' 
                df_for_csv_export = df_for_csv_export.rename(columns={'index': new_time_column_name})
            
            csv_data_bytes = df_for_csv_export.to_csv(index=False).encode('utf-8-sig')

            actual_download_filename = download_csv_filename.strip() if download_csv_filename.strip() else download_filename_suggestion
            if not actual_download_filename.lower().endswith('.csv'):
                actual_download_filename += '.csv'

            st.download_button(
                label="下载合并结果 (CSV)",
                data=csv_data_bytes,
                file_name=actual_download_filename, # Use the name from the new input
                mime='text/csv',
                key='download_merged_df_as_csv_button'
            )

        # Display preview of temp_merged_df in the right column
        with right_feedback:
            # Ensure this preview is only shown if temp_merged_df still exists (e.g. not yet saved and deleted)
            if 'temp_merged_df' in st.session_state and st.session_state.temp_merged_df is not None:
                st.markdown("##### **暂存合并结果预览**")
                st.dataframe(st.session_state.temp_merged_df) 

if __name__ == "__main__":
    show_append_merge_data_ui()