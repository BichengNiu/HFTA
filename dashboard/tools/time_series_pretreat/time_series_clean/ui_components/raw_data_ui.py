import streamlit as st
import pandas as pd
import io
import datetime

# Assuming utils are in a sibling directory to ui_components
from ..utils.parsers import parse_indices # For rows_to_skip_str
from ..utils.data_processing import (
    load_and_preprocess_data, 
    check_processed_header_duplicates, 
    attempt_auto_datetime_conversion
)

# --- 新增回调函数 --- 
def _apply_preprocessing_settings_on_change(session_state):
    print(f"[DEBUG raw_data_ui CALLBACK] _apply_preprocessing_settings_on_change triggered. Selected sheet: {session_state.get('ts_tool_selected_sheet_name')}, Header row input: {session_state.get(f'ts_tool_header_row_str_R{session_state.get('data_import_reset_counter', 0)}','')}, Skip rows input: {session_state.get(f'ts_tool_rows_to_skip_str_R{session_state.get('data_import_reset_counter', 0)}','')}") # DEBUG ADDED
    session_state.ts_tool_error_msg = None 
    session_state.ts_tool_data_processed = None 
    session_state.ts_tool_processing_applied = False
    try:
        # --- MODIFIED: Read from dynamic keys ---
        current_reset_count = session_state.get('data_import_reset_counter', 0)
        skip_rows_input_key = f"ts_tool_rows_to_skip_str_R{current_reset_count}"
        header_row_input_key = f"ts_tool_header_row_str_R{current_reset_count}"

        rows_to_skip_str = session_state.get(skip_rows_input_key, "")
        header_row_str = session_state.get(header_row_input_key, "")
        
        rows_to_skip = parse_indices(rows_to_skip_str)
        header_row = None
        if header_row_str.strip():
            try:
                header_row = int(header_row_str.strip())
            except ValueError:
                raise ValueError("表头行索引必须是一个数字。")
        # --- END OF MODIFICATION ---
        
        if header_row is not None and header_row in rows_to_skip:
                raise ValueError(f"表头行 ({header_row}) 不能包含在要跳过的行 ({rows_to_skip}) 中。")

        if session_state.ts_tool_uploaded_file_content:
            selected_sheet = session_state.ts_tool_selected_sheet_name
            sheet_arg = selected_sheet if selected_sheet != 'csv' else 0 
            
            df_processed = load_and_preprocess_data(
                session_state,
                file_content_bytes=session_state.ts_tool_uploaded_file_content,
                file_name_with_extension=session_state.ts_tool_uploaded_file_name,
                rows_to_skip_list=rows_to_skip,
                header_row_int_or_none=header_row,
                sheet_name_to_load=sheet_arg
            )
            # --- BEGIN ADDED DEBUG PRINTS ---
            print(f"[DEBUG raw_data_ui] After load_and_preprocess_data:")
            print(f"[DEBUG raw_data_ui]   df_processed is None: {df_processed is None}")
            if df_processed is not None:
                print(f"[DEBUG raw_data_ui]   df_processed.empty: {df_processed.empty}")
                print(f"[DEBUG raw_data_ui]   df_processed shape: {df_processed.shape}")
                print(f"[DEBUG raw_data_ui]   df_processed columns: {df_processed.columns.tolist() if not df_processed.empty else 'N/A (empty)'}")
                if not df_processed.empty:
                    print(f"[DEBUG raw_data_ui]   df_processed head:")
                    try:
                        print(df_processed.head().to_string())
                    except Exception as e_head:
                        print(f"[DEBUG raw_data_ui]   Error printing df_processed head: {e_head}")
            # --- END ADDED DEBUG PRINTS ---
                
            if df_processed is None:
                # 使用 st.warning 或保留错误消息给主标签页显示
                # st.error(f"加载和预处理数据失败。 {session_state.get('ts_tool_error_message', '')}")
                session_state.ts_tool_data_processed = pd.DataFrame()
                print("[DEBUG raw_data_ui] Set ts_tool_data_processed to EMPTY DF because df_processed was None.") # DEBUG
                session_state.ts_tool_processing_applied = False
            elif df_processed.empty or df_processed.shape[1] <= 0:
                st.warning("应用处理后数据为空，请检查跳过行和表头行设置。")
                session_state.ts_tool_data_processed = pd.DataFrame() 
                print("[DEBUG raw_data_ui] Set ts_tool_data_processed to EMPTY DF because df_processed was empty or no columns.") # DEBUG
            else:
                df_processed_converted, potential_time_cols_detected = attempt_auto_datetime_conversion(df_processed)
                
                if df_processed_converted is not None and not df_processed_converted.empty:
                    if potential_time_cols_detected:
                        for col_name in potential_time_cols_detected:
                            if col_name in df_processed_converted.columns:
                                try:
                                    df_processed_converted[col_name] = pd.to_datetime(df_processed_converted[col_name], errors='coerce').astype('datetime64[ns]')
                                except Exception:
                                    pass # Log error if needed
                    if len(df_processed_converted.columns) > 0:
                        actual_first_col_name = df_processed_converted.columns[0] # Get actual column name
                        # Use string representation for checking against potential_time_cols_detected (which are strings)
                        if str(actual_first_col_name) not in (potential_time_cols_detected or []):
                            # Use actual_first_col_name (original type) for DataFrame access
                            if df_processed_converted[actual_first_col_name].dtype == 'object' or \
                               pd.api.types.is_datetime64_any_dtype(df_processed_converted[actual_first_col_name]):
                                try:
                                    # Use actual_first_col_name (original type) for DataFrame access
                                    if df_processed_converted[actual_first_col_name].apply(lambda x: isinstance(x, (datetime.datetime, datetime.date, pd.Timestamp))).any():
                                        # Use actual_first_col_name (original type) for DataFrame access
                                        df_processed_converted[actual_first_col_name] = pd.to_datetime(df_processed_converted[actual_first_col_name], errors='coerce').astype('datetime64[ns]')
                                except Exception:
                                    pass # Log error if needed
                    if isinstance(df_processed_converted.index, pd.DatetimeIndex):
                        df_processed_converted.index = df_processed_converted.index.astype('datetime64[ns]')

                session_state.ts_tool_data_processed_FULL = df_processed_converted.copy()
                session_state.ts_tool_data_processed = df_processed_converted.copy()
                # --- BEGIN ADDED DEBUG PRINTS FOR SUCCESS CASE ---
                print(f"[DEBUG raw_data_ui] SUCCESSFULLY updated ts_tool_data_processed.")
                print(f"[DEBUG raw_data_ui]   ts_tool_data_processed shape: {session_state.ts_tool_data_processed.shape}")
                print(f"[DEBUG raw_data_ui]   ts_tool_data_processed columns: {session_state.ts_tool_data_processed.columns.tolist()}")
                if not session_state.ts_tool_data_processed.empty:
                    print(f"[DEBUG raw_data_ui]   ts_tool_data_processed head:")
                    try:
                        print(session_state.ts_tool_data_processed.head().to_string())
                    except Exception as e_head_final:
                        print(f"[DEBUG raw_data_ui]   Error printing ts_tool_data_processed head: {e_head_final}")
                # --- END ADDED DEBUG PRINTS FOR SUCCESS CASE ---
                
                if potential_time_cols_detected:
                        # st.info(f"已自动尝试将列 {potential_time_cols_detected} 转换为时间类型。") # Commented out blue info message
                        pass # Keep the block for potential future logic if needed

                session_state.ts_tool_cols_to_keep = df_processed_converted.columns.tolist()
                session_state.ts_tool_data_final = df_processed_converted
                
                if not df_processed_converted.empty and len(df_processed_converted.columns) > 0:
                    session_state.ts_tool_manual_time_col = str(df_processed_converted.columns[0])
                else:
                    session_state.ts_tool_manual_time_col = "(自动识别)"
                
                # st.success("成功应用处理并重新加载数据！") # 消息可能过于频繁
                session_state.ts_tool_processing_applied = True
                # --- <<< 新增：设置标志位以强制刷新 processed_data_ui 中的快照 >>> ---
                session_state.force_processed_ui_snapshot_refresh_flag = True
                print("[DEBUG raw_data_ui] Set force_processed_ui_snapshot_refresh_flag to True.")
                # st.rerun() # Commented out to prevent 'no-op' warning and because state changes should trigger rerun
                # --- <<< 结束新增 >>> ---
        else:
            session_state.ts_tool_error_msg = "无法重新读取文件，文件内容丢失。请重新上传文件。"

    except ValueError as ve:
        session_state.ts_tool_error_msg = f"输入错误: {ve}"
        print(f"[ERROR raw_data_ui] ValueError in _apply_preprocessing_settings_on_change: {ve}") # DEBUG
    except Exception as e:
        session_state.ts_tool_error_msg = f"处理文件时出错: {e}"
        print(f"[ERROR raw_data_ui] Exception in _apply_preprocessing_settings_on_change: {e}") # DEBUG
        import traceback
        print(traceback.format_exc()) # Print full traceback for unexpected errors

def _display_raw_data_preview_and_controls(st, session_state):
    """Displays raw data preview, controls for rows to skip, header row, and a button to apply them."""
    if session_state.ts_tool_uploaded_file_content and session_state.ts_tool_selected_sheet_name:
        if session_state.ts_tool_data_raw is None:
            try:
                file_content_io_preview = io.BytesIO(session_state.ts_tool_uploaded_file_content)
                if session_state.ts_tool_selected_sheet_name == 'csv':
                    df_raw_display = pd.read_csv(file_content_io_preview, header=None)
                else: 
                    df_raw_display = pd.read_excel(file_content_io_preview, sheet_name=session_state.ts_tool_selected_sheet_name, header=None, engine='openpyxl')
                
                if df_raw_display is not None and not df_raw_display.empty:
                    for col in df_raw_display.columns:
                        if df_raw_display[col].dtype == 'object':
                            is_datetime_obj_series, is_float_obj_series, is_int_obj_series = False, False, False
                            try:
                                col_dropna = df_raw_display[col].dropna()
                                if not col_dropna.empty:
                                     is_datetime_obj_series = col_dropna.apply(lambda x: isinstance(x, datetime.datetime)).any()
                                     if not is_datetime_obj_series:
                                        is_float_obj_series = col_dropna.apply(lambda x: isinstance(x, float)).any()
                                        if not is_float_obj_series:
                                            is_int_obj_series = col_dropna.apply(lambda x: isinstance(x, int)).any()
                            except Exception: pass
                            if is_datetime_obj_series or is_float_obj_series or is_int_obj_series:
                                try: df_raw_display[col] = df_raw_display[col].astype(str) 
                                except Exception: pass
                
                session_state.ts_tool_data_raw = df_raw_display
                st.success(f"成功加载工作表 '{session_state.ts_tool_selected_sheet_name}' 用于预览。")
                st.rerun()
            except Exception as e:
                session_state.ts_tool_error_msg = f"加载工作表 '{session_state.ts_tool_selected_sheet_name}' 预览时出错: {e}"
                session_state.ts_tool_data_raw = None
        
    if session_state.ts_tool_data_raw is not None and not session_state.ts_tool_data_raw.empty:
        cols_config = st.columns(2)
        current_reset_count_raw = session_state.get('data_import_reset_counter', 0)
        skip_rows_key = f"ts_tool_rows_to_skip_str_R{current_reset_count_raw}"
        header_row_key = f"ts_tool_header_row_str_R{current_reset_count_raw}"

        with cols_config[0]:
            skip_rows_input_val = session_state.get(skip_rows_key, "")
            skip_rows_str_input = st.text_input(
                "**输入要跳过的行索引 (用逗号分隔, e.g., 0,1,2, 5-7):**",
                value=skip_rows_input_val,
                key=skip_rows_key
            )
        with cols_config[1]:
            header_row_input_val = session_state.get(header_row_key, "")
            header_row_str_input = st.text_input(
                "**输入包含表头的行索引 (单个数字, e.g., 3):**",
                value=header_row_input_val,
                key=header_row_key
            )
        
        st.caption("提示：如果文件不需要跳过行或指定特定表头行，请将上方两框留空，然后点击下方按钮。")

        if st.button("应用行处理并加载数据", key=f"ts_tool_apply_raw_processing_R{current_reset_count_raw}"):
            session_state.ts_tool_error_msg = None 
            session_state.ts_tool_data_processed = None 
            session_state.ts_tool_processing_applied = False
            
            rows_to_skip_str = skip_rows_str_input
            header_row_str = header_row_str_input
            
            try:
                rows_to_skip_list = []
                header_row_int = 0

                if rows_to_skip_str.strip():
                    rows_to_skip_list = parse_indices(rows_to_skip_str)
                
                if header_row_str.strip():
                    try:
                        header_row_int = int(header_row_str)
                    except ValueError:
                        raise ValueError("表头行索引必须是一个有效的数字。")
                else:
                    header_row_int = 0

                if header_row_int is not None and rows_to_skip_list and header_row_int in rows_to_skip_list:
                        raise ValueError(f"表头行 ({header_row_int}) 不能包含在要跳过的行 ({rows_to_skip_list}) 中。")

                if session_state.ts_tool_uploaded_file_content:
                    selected_sheet = session_state.ts_tool_selected_sheet_name
                    sheet_arg = selected_sheet if selected_sheet != 'csv' else 0 
                    
                    df_processed = load_and_preprocess_data(
                        session_state,
                        file_content_bytes=session_state.ts_tool_uploaded_file_content,
                        file_name_with_extension=session_state.ts_tool_uploaded_file_name,
                        rows_to_skip_list=rows_to_skip_list,
                        header_row_int_or_none=header_row_int,
                        sheet_name_to_load=sheet_arg
                    )
                        
                    if df_processed is None:
                        session_state.ts_tool_data_processed = pd.DataFrame()
                        session_state.ts_tool_processing_applied = False
                        if not session_state.get('ts_tool_error_msg'):
                           session_state.ts_tool_error_msg = "加载和预处理数据失败，返回了 None。"
                    elif df_processed.empty or df_processed.shape[1] <= 0:
                        st.warning("应用处理后数据为空，请检查跳过行和表头行设置。")
                        session_state.ts_tool_data_processed = pd.DataFrame() 
                    else:
                        df_processed_converted, potential_time_cols_detected = attempt_auto_datetime_conversion(df_processed)
                        if df_processed_converted is not None and not df_processed_converted.empty:
                            if potential_time_cols_detected:
                                for col_name in potential_time_cols_detected:
                                    if col_name in df_processed_converted.columns:
                                        try:
                                            df_processed_converted[col_name] = pd.to_datetime(df_processed_converted[col_name], errors='coerce').astype('datetime64[ns]')
                                        except Exception: pass 
                            if len(df_processed_converted.columns) > 0:
                                actual_first_col_name = df_processed_converted.columns[0] 
                                if str(actual_first_col_name) not in (potential_time_cols_detected or []):
                                    if df_processed_converted[actual_first_col_name].dtype == 'object' or \
                                    pd.api.types.is_datetime64_any_dtype(df_processed_converted[actual_first_col_name]):
                                        try:
                                            if df_processed_converted[actual_first_col_name].apply(lambda x: isinstance(x, (datetime.datetime, datetime.date, pd.Timestamp))).any():
                                                df_processed_converted[actual_first_col_name] = pd.to_datetime(df_processed_converted[actual_first_col_name], errors='coerce').astype('datetime64[ns]')
                                        except Exception: pass
                            if isinstance(df_processed_converted.index, pd.DatetimeIndex):
                                df_processed_converted.index = df_processed_converted.index.astype('datetime64[ns]')

                        session_state.ts_tool_data_processed_FULL = df_processed_converted.copy()
                        session_state.ts_tool_data_processed = df_processed_converted.copy()
                        session_state.ts_tool_cols_to_keep = df_processed_converted.columns.tolist()
                        session_state.ts_tool_data_final = df_processed_converted
                        if not df_processed_converted.empty and len(df_processed_converted.columns) > 0:
                            session_state.ts_tool_manual_time_col = str(df_processed_converted.columns[0])
                        else:
                            session_state.ts_tool_manual_time_col = "(自动识别)"
                        st.success("成功应用处理并加载数据！")
                        session_state.ts_tool_processing_applied = True
                        session_state.force_processed_ui_snapshot_refresh_flag = True
                        st.rerun()

                else:
                    session_state.ts_tool_error_msg = "无法重新读取文件，文件内容丢失。请重新上传文件。"
            
            except ValueError as ve:
                session_state.ts_tool_error_msg = f"输入错误: {ve}"
            except Exception as e:
                session_state.ts_tool_error_msg = f"处理文件时出错: {e}"
                import traceback
                st.error(f"详细错误: {traceback.format_exc()}")

            if session_state.get('ts_tool_error_msg'):
                 st.error(session_state.ts_tool_error_msg)

    elif session_state.ts_tool_uploaded_file_content and not session_state.ts_tool_selected_sheet_name:
        pass 