import streamlit as st
import pandas as pd
import io # For download functionality
import os
import copy # For staging deepcopy
from datetime import date # Import date for date input

def display_data_preview_and_download_section(st, session_state):
    """处理数据显示、筛选、暂存和下载的UI部分。"""
    
    # This section is shown only if data is loaded.
    # The check 'if session_state.ts_compute_data is not None:' is handled by the calling function in main tab.

    st.subheader("暂存和下载") # Renamed Subheader
    df_current_preview = session_state.ts_compute_data

    # --- Add Date Range Selection UI --- 
    start_date = None
    end_date = None
    date_range_available = isinstance(df_current_preview.index, pd.DatetimeIndex)

    if date_range_available:
        try:
            min_date_val = df_current_preview.index.min()
            max_date_val = df_current_preview.index.max()
            
            if pd.isna(min_date_val) or pd.isna(max_date_val):
                st.warning("数据索引中包含无效日期时间值，无法使用时间段选择。")
                date_range_available = False
            else:
                min_date = min_date_val.date() if hasattr(min_date_val, 'date') else min_date_val
                max_date = max_date_val.date() if hasattr(max_date_val, 'date') else max_date_val
                
                # Initialize session state keys if they don't exist
                if 'ts_compute_start_date_vc' not in session_state:
                     session_state.ts_compute_start_date_vc = min_date
                if 'ts_compute_end_date_vc' not in session_state:
                     session_state.ts_compute_end_date_vc = max_date

                st.caption("选择时间范围 (用于筛选要暂存/下载的数据):")
                col_date1, col_date2 = st.columns(2)
                with col_date1:
                    start_date_input = st.date_input("开始日期:",
                                               min_value=min_date,
                                               max_value=max_date,
                                               key="ts_compute_start_date_vc") # Key sets state
                with col_date2:
                    end_date_input = st.date_input("结束日期:",
                                             min_value=min_date, 
                                             max_value=max_date,
                                             key="ts_compute_end_date_vc") # Key sets state
                
                # Update local vars after getting input
                start_date = start_date_input
                end_date = end_date_input
                
                if start_date and end_date and start_date > end_date: 
                    st.error("错误：开始日期不能晚于结束日期。")
                    # Keep state as is, but maybe prevent actions?
                    # For now, filtering logic below will handle empty result
        except Exception as date_e:
            st.warning(f"处理日期范围时出错: {date_e}。时间段选择不可用。")
            date_range_available = False
    else:
        st.info("数据的索引不是有效的日期时间格式，无法使用时间段选择功能。")
        date_range_available = False # Ensure it's false if not DatetimeIndex
        
    # Re-update session state variable reflecting overall availability for other modules if needed
    session_state.ts_compute_date_range_available = date_range_available
    # --- End Date Range Selection UI --- 

    # --- Variable Selection --- 
    available_columns = df_current_preview.columns.tolist()
    # Initialize the key if it doesn't exist, defaulting to all columns
    if 'ts_compute_preview_multiselect_dp' not in session_state:
        session_state.ts_compute_preview_multiselect_dp = available_columns
        
    selected_columns = st.multiselect(
        "选择要暂存/下载的变量:", 
        options=available_columns, 
        key="ts_compute_preview_multiselect_dp",
        placeholder="点击选择或输入关键字筛选变量..." 
    )

    # --- Data Filtering based on selections --- 
    df_display = df_current_preview.copy() 

    if selected_columns: 
        df_display = df_display[selected_columns]
    else:
        # If no columns are selected, df_display becomes empty with original index
        # This prevents staging/downloading all columns if nothing is selected
        df_display = pd.DataFrame(index=df_display.index) 
        st.info("请至少选择一个变量以进行暂存或下载。")

    # Apply date filtering if applicable and valid
    # Use the start_date and end_date determined from the inputs above
    if date_range_available and start_date and end_date:
        if start_date <= end_date: 
            try:
                # 确保索引是单调递增的，避免切片报错
                if not df_display.index.is_monotonic_increasing:
                    df_display_sorted = df_display.sort_index()
                    df_display = df_display_sorted.loc[str(start_date):str(end_date)]
                else:
                    df_display = df_display.loc[str(start_date):str(end_date)]
                if df_display.empty and selected_columns: # Only warn if columns were selected but date range yielded empty
                    st.warning("选定的变量在当前时间范围内无数据。")
            except Exception as filter_e:
                st.warning(f"应用日期筛选时出错: {filter_e}")
        # else: invalid date range error shown near inputs
    
    # --- Removed st.dataframe(df_display) --- 

    # --- Staging and Download Functionality --- 
    if not df_display.empty:
        st.markdown("--- ") # Add a divider
        col_stage, col_download = st.columns(2) # Create two columns

        # --- Staging Column --- 
        with col_stage:
            st.markdown("**暂存数据:**")
            stage_name = st.text_input("输入暂存名称:", key="stage_name_input_dp", placeholder="例如: 计算结果_v1")
            if st.button("💾 暂存选定数据", key="stage_data_button_dp"):
                if not stage_name.strip():
                    st.error("请输入暂存数据的名称。")
                else:
                    stage_name = stage_name.strip()
                    if 'staged_data' not in st.session_state:
                        st.session_state.staged_data = {}
                    
                    if stage_name in st.session_state.staged_data:
                        st.warning(f"名称 '{stage_name}' 的暂存数据已存在，将被覆盖。") 
                        
                    # Prepare data and summary for staging
                    df_to_stage = df_display.copy()
                    # --- DEBUG for Staging ---
                    st.write("DEBUG (Staging): df_to_stage shape:", df_to_stage.shape)
                    st.write("DEBUG (Staging): df_to_stage columns:", df_to_stage.columns)
                    # --- END DEBUG ---
                    summary = {
                        'rows': df_to_stage.shape[0],
                        'cols': df_to_stage.shape[1],
                        'variables': selected_columns, # Store the list of selected columns
                        'date_range': f"{start_date} to {end_date}" if date_range_available and start_date and end_date else '全时间段',
                        'source': session_state.get('ts_compute_file_name', '未知来源')
                    }
                    # Add time info if index is DatetimeIndex
                    if isinstance(df_to_stage.index, pd.DatetimeIndex) and not df_to_stage.index.empty:
                         summary['start_time'] = str(df_to_stage.index.min().date() if hasattr(df_to_stage.index.min(), 'date') else df_to_stage.index.min())
                         summary['end_time'] = str(df_to_stage.index.max().date() if hasattr(df_to_stage.index.max(), 'date') else df_to_stage.index.max())
                         summary['frequency'] = str(pd.infer_freq(df_to_stage.index) or '未知')
                    else: 
                        summary['start_time'] = '非时间索引'
                        summary['end_time'] = '非时间索引'
                        summary['frequency'] = '未知'
                        
                    st.session_state.staged_data[stage_name] = {
                        'df': df_to_stage,
                        'summary': summary
                    }
                    st.success(f"数据 '{stage_name}' 已成功暂存！")
                    # Clear the input after successful staging? Maybe not needed due to rerun
                    st.rerun() # Rerun to update sidebar staging list

        # --- Download Column --- 
        with col_download:
            st.markdown("**下载处理后的数据:**")
            # Format selection
            file_format = st.radio("选择下载格式:", ("CSV", "Excel (.xlsx)"), key="download_format_dp", horizontal=True)
            
            # Filename suggestion
            file_name_suggestion = "computed_data"
            if session_state.get('ts_compute_file_name'):
                original_name, _ = os.path.splitext(session_state.ts_compute_file_name)
                file_name_suggestion = f"{original_name}_computed"
            if selected_columns:
                # Add first selected column name to suggestion if only one or two selected
                if len(selected_columns) <= 2:
                     file_name_suggestion += f"_{selected_columns[0]}"
                else:
                     file_name_suggestion += f"_{len(selected_columns)}vars"
            
            download_file_name = st.text_input("下载文件名 (不含扩展名):", value=file_name_suggestion, key="download_filename_dp")

            # Download buttons logic
            if download_file_name.strip():
                download_file_name = download_file_name.strip()
                df_for_download = df_display # Already filtered by var/date
                # Apply index formatting for download
                if isinstance(df_for_download.index, pd.DatetimeIndex):
                    df_for_download_formatted = df_for_download.copy()
                    df_for_download_formatted.index = df_for_download_formatted.index.strftime('%Y-%m-%d')
                else:
                    df_for_download_formatted = df_for_download
                
                dl_button_placeholder = st.empty() # Placeholder for the button
                if file_format == "CSV":
                    csv_data = df_for_download_formatted.to_csv(index=True).encode('utf-8-sig')
                    dl_button_placeholder.download_button(
                        label="📥 下载 CSV",
                        data=csv_data,
                        file_name=f"{download_file_name}.csv",
                        mime='text/csv',
                        key="download_csv_button_dp"
                    )
                elif file_format == "Excel (.xlsx)":
                    df_for_download = df_display.copy() 
                    df_to_write = df_for_download.copy()
                    original_index_name = df_to_write.index.name 
                    if isinstance(df_to_write.index, pd.DatetimeIndex):
                        try:
                            date_col_name = original_index_name if original_index_name else 'Date' 
                            df_to_write[date_col_name] = df_to_write.index.strftime('%Y-%m-%d')
                            df_to_write = df_to_write.reset_index(drop=True) 
                            cols = [date_col_name] + [col for col in df_to_write.columns if col != date_col_name]
                            df_to_write = df_to_write[cols]
                        except Exception as fmt_e:
                            st.warning(f"下载时处理日期索引出错: {fmt_e}. 将尝试不带格式化日期列下载。")
                            df_to_write = df_to_write.reset_index(drop=True) 
                    else:
                         df_to_write = df_to_write.reset_index() 
                         if original_index_name and original_index_name != 'index': 
                             df_to_write = df_to_write.rename(columns={'index': original_index_name})

                    # --- DEBUG for Excel ---
                    st.write("DEBUG (Excel): df_to_write shape:", df_to_write.shape)
                    st.write("DEBUG (Excel): df_to_write columns:", df_to_write.columns)
                    st.write("DEBUG (Excel): df_to_write index:", df_to_write.index)
                    st.write("DEBUG (Excel): df_to_write head:\n", df_to_write.head())
                    # --- END DEBUG ---

                    excel_buffer = io.BytesIO()
                    try:
                        with pd.ExcelWriter(excel_buffer, engine='xlsxwriter') as writer:
                            df_to_write.to_excel(writer, sheet_name='Sheet1', index=False) # Set index=False
                        excel_data = excel_buffer.getvalue()
                        
                        dl_button_placeholder = st.empty() 
                        dl_button_placeholder.download_button(
                            label="📥 下载 Excel",
                            data=excel_data,
                            file_name=f"{download_file_name}.xlsx",
                            mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
                            key="download_excel_button_dp"
                        )
                    except Exception as excel_e:
                        st.error(f"写入Excel文件时出错: {excel_e}")
            else:
                st.caption("请输入有效的文件名以启用下载按钮。")
                
    elif session_state.ts_compute_data is not None: 
        # Data is loaded, but df_display is empty (due to no var selected or date filter)
        # Message is already shown above if no vars selected
        if not selected_columns:
             pass # Message shown in selection area
        else: # Must be due to date filter yielding empty
             st.info("当前变量和时间范围组合下无数据显示，无法暂存或下载。")

    # It's assumed that the main tab handles the case where session_state.ts_compute_data is None initially. 