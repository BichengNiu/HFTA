import streamlit as st
import pandas as pd
import io

# Assuming utils are in a sibling directory to ui_components
from ..utils.time_analysis import generate_final_data

# --- Helper functions for download (can be part of this UI component if only used here) ---
# 🔥 修复：移除缓存装饰器避免 ScriptRunContext 警告
# @st.cache_data # Add cache decorator for potentially expensive conversion
def convert_final_df_to_csv(df):
    return df.to_csv(index=False).encode('utf-8-sig') # utf-8-sig for Excel compatibility

# @st.cache_data
def convert_final_df_to_excel(df):
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        df.to_excel(writer, index=False, sheet_name='FinalData')
    # output.seek(0) # Not strictly necessary for BytesIO for download data
    return output.getvalue()
# ---

def display_final_data_generation_and_controls(st, session_state):
    """Displays UI for final data generation, filtering, completion, display, and staging."""

    if not session_state.ts_tool_processing_applied or session_state.ts_tool_data_processed is None:
        # Show a message if there's no processed data to finalize
        # This state might be hit if processing failed or was never run.
        if session_state.ts_tool_uploaded_file_content: # Check if a file was at least uploaded
             st.info("请先完成数据预处理步骤，然后再生成最终数据。")
        return 
    
    if session_state.ts_tool_data_processed.empty and session_state.ts_tool_last_operation_type not in ["vertical_merge", "horizontal_merge"]:
        # If data_processed is empty and it wasn't due to a merge operation (which might legitimately result in empty), then inform.
        st.warning("当前已处理数据为空，无法生成最终数据。请检查之前的处理步骤。")
        return

    st.subheader("最终数据生成与预览")
    st.caption(
        "此部分允许您对选定的列和已识别的时间序列进行最终处理。 "
        "您可以按日期范围筛选数据。如果已在前面步骤中启用了频率补全，此数据将基于补全后的序列。"
    )


    # --- Generate Final Data Button ---
    if st.button("生成/刷新最终数据", key="generate_final_data_button"):
        if not session_state.ts_tool_cols_to_keep:
            st.warning("没有选择任何要保留的列。请在'已处理数据预览与列选择'部分至少选择一列。")
        elif not session_state.ts_tool_time_col_info or not session_state.ts_tool_time_col_info.get('name'):
             if session_state.ts_tool_complete_time_index: # Only warn if completion is checked but no time col
                st.warning("已勾选自动补全时间序列，但未能识别有效的时间列。请先在'时间与频率设置'部分成功分析时间列。")
             # Allow generation even without time column if completion is not checked
             df_final_generated, completion_status = generate_final_data(
                    session_state.ts_tool_data_processed, 
                    session_state.ts_tool_cols_to_keep,
                    session_state.ts_tool_time_col_info, 
                    session_state.ts_tool_filter_start_date, 
                    session_state.ts_tool_filter_end_date,
                    False, # <<< MODIFIED: complete_time_index is now False here
                    session_state.get('ts_tool_manual_frequency_fc', session_state.get('ts_tool_manual_frequency', '自动')) # Pass the one from fc_ui, or original default
                )
             session_state.ts_tool_data_final = df_final_generated
             # Completion message here is less relevant as asfreq is not done here
             # session_state.ts_tool_completion_message = completion_status.get('completion_message')
        else: # Columns and time info are available
            df_final_generated, completion_status = generate_final_data(
                session_state.ts_tool_data_processed, 
                session_state.ts_tool_cols_to_keep,
                session_state.ts_tool_time_col_info, 
                session_state.ts_tool_filter_start_date, 
                session_state.ts_tool_filter_end_date,
                False, # <<< MODIFIED: complete_time_index is now False here
                session_state.get('ts_tool_manual_frequency_fc', session_state.get('ts_tool_manual_frequency', '自动')) # Pass the one from fc_ui, or original default
            )
            session_state.ts_tool_data_final = df_final_generated
            # Completion message here is less relevant as asfreq is not done here
            # session_state.ts_tool_completion_message = completion_status.get('completion_message')
        
        if session_state.ts_tool_data_final is not None:
            st.success("最终数据已生成！")
        elif not session_state.ts_tool_completion_message: # If no specific completion message, but final data is None
            st.error("生成最终数据时出错，但未返回特定消息。")
        # No st.rerun() here, data editor below will update

    # --- Display Final Data ---
    if session_state.ts_tool_data_final is not None and not session_state.ts_tool_data_final.empty:
        st.markdown("**最终数据预览:**")
        # Display a data editor for the final data
        # This copy is important if edits are allowed and should not reflect back to ts_tool_data_final directly unless intended
        df_display_final = session_state.ts_tool_data_final.copy()
        
        # Sort by DatetimeIndex descending before display
        if isinstance(df_display_final.index, pd.DatetimeIndex):
            df_display_final = df_display_final.sort_index(ascending=False)
            
        session_state.ts_tool_data_final_edited = st.data_editor(
            df_display_final, 
            key="final_data_editor", 
            use_container_width=True,
            num_rows="dynamic" # Allow adding/deleting rows if needed, though typically just for viewing
        )
        # Here, decide if edits from final_data_editor should update ts_tool_data_final
        # For now, let's assume it's for display and download, staging uses ts_tool_data_final

        # --- Download Buttons ---
        st.markdown("**下载最终数据:**")
        dl_cols = st.columns(2)
        with dl_cols[0]:
            csv_data = convert_final_df_to_csv(session_state.ts_tool_data_final) 
            st.download_button(
                label="下载为 CSV 文件",
                data=csv_data,
                file_name=f"{session_state.ts_tool_uploaded_file_name}_final.csv" if session_state.ts_tool_uploaded_file_name else "final_data.csv",
                mime='text/csv',
                key="download_csv_button"
            )
        with dl_cols[1]:
            excel_data = convert_final_df_to_excel(session_state.ts_tool_data_final)
            st.download_button(
                label="下载为 Excel 文件",
                data=excel_data,
                file_name=f"{session_state.ts_tool_uploaded_file_name}_final.xlsx" if session_state.ts_tool_uploaded_file_name else "final_data.xlsx",
                mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
                key="download_excel_button"
            )
    elif session_state.ts_tool_data_final is not None and session_state.ts_tool_data_final.empty:
        st.info("生成的最终数据为空。这可能是由于筛选条件或原始数据本身导致。")
    
    st.divider()
    # --- <<< 新增：数据暂存功能 >>> ---
    st.subheader("数据暂存")
    st.caption(
        "您可以将当前生成的最终数据暂存起来，以便后续在其他工具或分析流程中使用。"
        "请为暂存的数据指定一个唯一的名称。"
    )
    # Check if there's final data to stage
    if session_state.ts_tool_data_final is not None and not session_state.ts_tool_data_final.empty:
        stage_name = st.text_input("输入暂存数据名称:", key="stage_data_name_input", placeholder="例如：处理后的销售数据_Q1")
        if st.button("暂存当前最终数据", key="stage_data_button"):
            if stage_name:
                if 'staged_data' not in st.session_state: # Corrected line
                    st.session_state.staged_data = {}
                
                # Check for name collision before overwriting
                if stage_name in st.session_state.staged_data:
                    st.warning(f"名称 '{stage_name}' 的暂存数据已存在。如果您继续，它将被覆盖。") # Corrected f-string
                    # Add a confirm button or a different workflow for overwriting if needed
                    # For now, direct overwrite on button press after warning
                
                # Store a copy to avoid issues if ts_tool_data_final is later modified
                df_to_stage = session_state.ts_tool_data_final.copy()
                summary = {
                    'rows': df_to_stage.shape[0],
                    'cols': df_to_stage.shape[1],
                    'first_col': str(df_to_stage.columns[0]) if df_to_stage.shape[1] > 0 else 'N/A',
                    'last_col': str(df_to_stage.columns[-1]) if df_to_stage.shape[1] > 0 else 'N/A',
                }
                
                time_col_name_from_info = session_state.get('ts_tool_time_col_info', {}).get('name')

                if isinstance(df_to_stage.index, pd.DatetimeIndex) and not df_to_stage.index.empty:
                    summary['start_time'] = str(df_to_stage.index.min().date() if hasattr(df_to_stage.index.min(), 'date') else df_to_stage.index.min())
                    summary['end_time'] = str(df_to_stage.index.max().date() if hasattr(df_to_stage.index.max(), 'date') else df_to_stage.index.max())
                    summary['frequency'] = str(pd.infer_freq(df_to_stage.index) or '未知')
                elif time_col_name_from_info and time_col_name_from_info in df_to_stage.columns and pd.api.types.is_datetime64_any_dtype(df_to_stage[time_col_name_from_info]):
                    time_series_for_summary = pd.to_datetime(df_to_stage[time_col_name_from_info]).dropna()
                    if not time_series_for_summary.empty:
                        summary['start_time'] = str(time_series_for_summary.min().date() if hasattr(time_series_for_summary.min(), 'date') else time_series_for_summary.min())
                        summary['end_time'] = str(time_series_for_summary.max().date() if hasattr(time_series_for_summary.max(), 'date') else time_series_for_summary.max())
                        summary['frequency'] = str(pd.infer_freq(time_series_for_summary.sort_values()) or '未知')
                    else:
                        summary['start_time'] = '未知'
                        summary['end_time'] = '未知'
                        summary['frequency'] = '未知'
                else: 
                    summary['start_time'] = '未知'
                    summary['end_time'] = '未知'
                    summary['frequency'] = '未知'

                st.session_state.staged_data[stage_name] = {
                    'df': df_to_stage,
                    'summary': summary
                }
                st.success(f"数据 '{stage_name}' 已成功暂存！") # Corrected f-string
                st.rerun() # <<< 新增：强制重新运行以更新侧边栏
                # Clear the input field after staging
                # st.session_state.stage_data_name_input = "" 
            else:
                st.error("请输入暂存数据的名称。")
    else:
        st.info("没有可用于暂存的最终数据。请先生成最终数据。")
    st.divider() # Final divider for this section 