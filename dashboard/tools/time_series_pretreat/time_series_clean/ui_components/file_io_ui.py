import streamlit as st
import pandas as pd
import io

# --- <<< 新增：重置数据导入状态的回调函数 >>> ---
def reset_data_import_state():
    """Resets all states related to data import to their initial values."""
    # Increment a counter to force re-creation of widgets with dynamic keys
    st.session_state.data_import_reset_counter = st.session_state.get('data_import_reset_counter', 0) + 1
    print(f"[CALLBACK reset_data_import_state] data_import_reset_counter incremented to: {st.session_state.data_import_reset_counter}")

    st.session_state.ts_tool_uploaded_file_name = None
    st.session_state.ts_tool_uploaded_file_content = None
    st.session_state.ts_tool_data_raw = None
    st.session_state.ts_tool_rows_to_skip_str = ""
    st.session_state.ts_tool_header_row_str = ""
    st.session_state.ts_tool_data_processed = None
    if 'ts_tool_data_processed_FULL' in st.session_state: # Only reset if exists
        st.session_state.ts_tool_data_processed_FULL = None
    st.session_state.ts_tool_processing_applied = False
    st.session_state.ts_tool_cols_to_keep = []
    st.session_state.ts_tool_data_final = None
    st.session_state.ts_tool_error_msg = None
    st.session_state.ts_tool_manual_time_col = "(自动识别)"
    st.session_state.processed_header_duplicates = {}
    st.session_state.ts_tool_auto_remove_duplicates = False
    st.session_state.ts_tool_manual_frequency = "自动"
    st.session_state.ts_tool_rename_rules = []
    st.session_state.ts_tool_filter_start_date = None
    st.session_state.ts_tool_filter_end_date = None
    st.session_state.ts_tool_merge_file_name = None
    st.session_state.ts_tool_merge_file_content = None
    st.session_state.ts_tool_merged_data = None
    st.session_state.ts_tool_merge_error = None
    st.session_state.ts_tool_time_col_info = {'name': None, 'parsed_series': None, 'status_message': '数据导入已重置', 'status_type': 'info'}
    st.session_state.ts_tool_horizontally_merged_data = None
    st.session_state.ts_tool_horizontal_merge_error = None
    st.session_state.ts_tool_last_operation_type = None
    st.session_state.ts_tool_complete_time_index = False
    st.session_state.ts_tool_completion_message = None
    st.session_state.ts_tool_sheet_names = None
    st.session_state.ts_tool_selected_sheet_name = None
    
    # Important for clearing the file uploader widget itself
    # if "time_series_tool_uploader" in st.session_state:
    #     st.session_state.time_series_tool_uploader = None # This line causes an error and should be removed

    # Reset snapshot keys if they exist from other modules that depend on initial data
    if 'df_processed_snapshot_at_entry' in st.session_state:
        st.session_state.df_processed_snapshot_at_entry = None
    if 'df_processed_FULL_snapshot_at_entry' in st.session_state:
        st.session_state.df_processed_FULL_snapshot_at_entry = None
    if 'ts_tool_df_at_time_ui_entry' in st.session_state:
        st.session_state.ts_tool_df_at_time_ui_entry = None
    if 'ts_tool_df_at_time_ui_entry_source_id' in st.session_state:
        st.session_state.ts_tool_df_at_time_ui_entry_source_id = None
    if 'ts_tool_df_at_missing_ui_entry' in st.session_state:
        st.session_state.ts_tool_df_at_missing_ui_entry = None
    if 'ts_tool_df_FULL_at_missing_ui_entry' in st.session_state:
        st.session_state.ts_tool_df_FULL_at_missing_ui_entry = None

    print("[CALLBACK reset_data_import_state] Data import state has been reset.")
    st.info("数据导入已重置。请重新上传文件。")
    # st.rerun() # Commented out to prevent 'no-op' warning
# --- <<< 结束新增 >>> ---

def _display_file_handling_and_sheet_selection(st, session_state):
    """Displays file uploader and handles new file logic."""
    
    # --- MODIFIED: Generate dynamic key for file_uploader ---
    current_reset_count = session_state.get('data_import_reset_counter', 0)
    uploader_key = f"time_series_tool_uploader_R{current_reset_count}"
    # --- END OF MODIFICATION ---

    col_title, col_reset_button = st.columns([0.8, 0.2]) # Adjust ratio as needed for alignment
    with col_title:
        st.subheader("上传数据")
    with col_reset_button:
        # --- MODIFIED: Removed CSS, simplified button ---
        if st.button("重置", # Changed button text
                     key="reset_data_import_button_file_io", 
                     on_click=reset_data_import_state, 
                     help="清除所有已上传的文件和相关设置，恢复到初始状态。"):
            pass # Callback handles the action and rerun
        # --- END OF MODIFICATION ---

    with st.container(border=False):
        # --- File Uploader ---
        uploaded_file = st.file_uploader(
            "**上传 Excel 或 CSV 数据文件**",
            type=["xlsx", "csv"],
            key=uploader_key # Use dynamic key
        )
        if uploaded_file: # 只有当文件上传后才尝试添加占位符
            st.empty() # <<< 尝试添加一个空的占位符来延长容器
    

    # --- Data Loading Logic (when a file is uploaded) ---
    if uploaded_file:
        if session_state.get('ts_tool_uploaded_file_name') != uploaded_file.name:
            session_state.ts_tool_uploaded_file_name = uploaded_file.name
            session_state.ts_tool_uploaded_file_content = uploaded_file.getvalue()
            
            # Reset relevant session_state variables for a new file
            session_state.ts_tool_data_raw = None
            session_state.ts_tool_rows_to_skip_str = ""
            session_state.ts_tool_header_row_str = ""
            session_state.ts_tool_data_processed = None
            session_state.ts_tool_processing_applied = False
            session_state.ts_tool_cols_to_keep = []
            session_state.ts_tool_data_final = None
            session_state.ts_tool_error_msg = None
            session_state.ts_tool_manual_time_col = "(自动识别)"
            session_state.processed_header_duplicates = {}
            session_state.ts_tool_auto_remove_duplicates = False
            session_state.ts_tool_manual_frequency = "自动"
            session_state.ts_tool_rename_rules = []
            session_state.ts_tool_filter_start_date = None
            session_state.ts_tool_filter_end_date = None
            session_state.ts_tool_merge_file_name = None
            session_state.ts_tool_merge_file_content = None
            session_state.ts_tool_merged_data = None
            session_state.ts_tool_merge_error = None
            session_state.ts_tool_time_col_info = {'name': None, 'parsed_series': None}
            session_state.ts_tool_horizontally_merged_data = None
            session_state.ts_tool_horizontal_merge_error = None
            session_state.ts_tool_last_operation_type = None
            session_state.ts_tool_complete_time_index = False
            session_state.ts_tool_completion_message = None
            session_state.ts_tool_sheet_names = None
            session_state.ts_tool_selected_sheet_name = None

            # --- Get Sheet Names for Excel files (but don't display selector here) ---
            try:
                file_extension = uploaded_file.name.split('.')[-1].lower()
                if file_extension == 'xlsx':
                    file_content_io = io.BytesIO(session_state.ts_tool_uploaded_file_content)
                    excel_file = pd.ExcelFile(file_content_io)
                    session_state.ts_tool_sheet_names = excel_file.sheet_names
                    if len(session_state.ts_tool_sheet_names) == 1:
                        session_state.ts_tool_selected_sheet_name = session_state.ts_tool_sheet_names[0]
                    print(f"[FileIO] File '{uploaded_file.name}' loaded, sheets: {session_state.ts_tool_sheet_names}")
                elif file_extension == 'csv':
                    session_state.ts_tool_sheet_names = ['csv']
                    session_state.ts_tool_selected_sheet_name = 'csv' 
                    print(f"[FileIO] CSV file '{uploaded_file.name}' loaded.")
                else:
                    session_state.ts_tool_error_msg = f"不支持的文件类型: {file_extension}"
                    session_state.ts_tool_uploaded_file_content = None
                    session_state.ts_tool_uploaded_file_name = None

            except Exception as e:
                session_state.ts_tool_error_msg = f"加载文件或读取Sheet名称时出错: {e}"
                session_state.ts_tool_uploaded_file_content = None
                session_state.ts_tool_uploaded_file_name = None
                session_state.ts_tool_sheet_names = None
                session_state.ts_tool_selected_sheet_name = None

    # Display Error Messages if any (related to file loading/sheet name reading)
    # Note: This error message might be better placed in the main tab if it's a general error display area.
    # For now, keeping it here as it's closely tied to the operations within this function.
    if session_state.ts_tool_error_msg: # Check if error_msg was set by file loading logic
        st.error(session_state.ts_tool_error_msg)
        # Potentially clear it after displaying, or let a more global error handler manage it
        # session_state.ts_tool_error_msg = None 

    # --- Sheet Selection UI and its on_change handler are REMOVED from here ---
    # --- They will be implemented in the calling tab (time_series_clean_tab.py) ---

# Note: The handle_sheet_change function is no longer needed in this file. 