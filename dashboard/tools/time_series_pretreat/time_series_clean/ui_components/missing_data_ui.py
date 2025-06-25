import streamlit as st
import pandas as pd
import numpy as np

# Assuming utils are in a sibling directory to ui_components
from ..utils.missing_data import (
    analyze_missing_data, 
    generate_missing_data_plot, 
    handle_missing_values
)

# --- <<< 新增：重置缺失数据处理状态的回调函数 >>> ---
def reset_missing_data_state():
    """Resets data to the state when entering the missing data UI and clears UI elements."""
    # <<< RERUN_DEBUG_MISSING_UI >>>
    print("RERUN_DEBUG_MISSING_UI: reset_missing_data_state callback initiated.")
    # Restore DataFrame from snapshot
    if 'ts_tool_df_at_missing_ui_entry' in st.session_state and \
       st.session_state.ts_tool_df_at_missing_ui_entry is not None:
        st.session_state.ts_tool_data_processed = st.session_state.ts_tool_df_at_missing_ui_entry.copy()
        print("[DEBUG missing_data_ui] ts_tool_data_processed has been reset from snapshot.")
    else:
        # Fallback or warning if snapshot is missing, though ideally it should always exist if logic is correct
        print("[WARN missing_data_ui] Snapshot 'ts_tool_df_at_missing_ui_entry' not found for reset.")

    if 'ts_tool_df_FULL_at_missing_ui_entry' in st.session_state and \
       st.session_state.ts_tool_df_FULL_at_missing_ui_entry is not None and \
       'ts_tool_data_processed_FULL' in st.session_state:
        st.session_state.ts_tool_data_processed_FULL = st.session_state.ts_tool_df_FULL_at_missing_ui_entry.copy()
        print("[DEBUG missing_data_ui] ts_tool_data_processed_FULL has been reset from snapshot.")
    else:
        print("[WARN missing_data_ui] Snapshot 'ts_tool_df_FULL_at_missing_ui_entry' not found for ts_tool_data_processed_FULL reset.")

    # Reset UI specific states for missing data handling
    st.session_state.missing_ui_selected_cols = []
    # Find the label for "无操作" to reset the selectbox to it
    fill_method_options_reset = {
        "无操作": "none", "向前填充 (ffill)": "ffill", "向后填充 (bfill)": "bfill",
        "用0填充 (仅数值型)": "zero", "用指定常量填充": "constant",
        "线性插值 (仅数值型)": "linear", "用均值填充 (仅数值型)": "mean",
        "用中位数填充 (仅数值型)": "median", "用众数填充": "mode"
    }
    default_fill_label = "无操作"
    for label, value in fill_method_options_reset.items():
        if value == "none":
            default_fill_label = label
            break
    if 'fill_method_selector_new_right_col' in st.session_state:
        st.session_state.fill_method_selector_new_right_col = default_fill_label 
    
    if 'fill_limit_input_new_right_col' in st.session_state:
        st.session_state.fill_limit_input_new_right_col = ""
    if 'constant_value_input_new_right_col' in st.session_state:
        st.session_state.constant_value_input_new_right_col = ""

    # Clear placeholders by re-assigning them if keys exist
    if session_state.get('analysis_placeholder_missing_ui_key') and session_state.get(session_state.analysis_placeholder_missing_ui_key):
        session_state[session_state.analysis_placeholder_missing_ui_key].empty()
    if session_state.get('plot_placeholder_missing_ui_key') and session_state.get(session_state.plot_placeholder_missing_ui_key):
        session_state[session_state.plot_placeholder_missing_ui_key].empty()
    
    # If there are other specific session state keys related to analysis results, clear them too
    # e.g., session_state.some_analysis_result_key = None

    print("[CALLBACK reset_missing_data_state] Missing data UI state and data have been reset.")
    st.info("缺失数据处理模块的数据和设置已重置到进入此部分时的状态。")
    # <<< RERUN_DEBUG_MISSING_UI >>>
    # print("RERUN_DEBUG_MISSING_UI: Calling st.rerun() at the end of reset_missing_data_state.")
    # st.rerun()
# --- <<< 结束新增 >>> ---

def display_missing_data_analysis_and_handling(st, session_state):
    """Displays UI for missing data analysis, visualization, and handling."""

    # --- <<< 新增：数据快照逻辑 >>> ---
    if session_state.get('force_missing_ui_snapshot_refresh_flag', False):
        if session_state.get('ts_tool_data_processed') is not None:
            session_state.ts_tool_df_at_missing_ui_entry = session_state.ts_tool_data_processed.copy()
            print("[DEBUG missing_data_ui] Snapshot 'ts_tool_df_at_missing_ui_entry' taken/updated.")
        else:
            session_state.ts_tool_df_at_missing_ui_entry = None
            print("[DEBUG missing_data_ui] ts_tool_data_processed is None, snapshot 'ts_tool_df_at_missing_ui_entry' set to None.")

        if session_state.get('ts_tool_data_processed_FULL') is not None:
            session_state.ts_tool_df_FULL_at_missing_ui_entry = session_state.ts_tool_data_processed_FULL.copy()
            print("[DEBUG missing_data_ui] Snapshot 'ts_tool_df_FULL_at_missing_ui_entry' taken/updated.")
        else:
            session_state.ts_tool_df_FULL_at_missing_ui_entry = None
            print("[DEBUG missing_data_ui] ts_tool_data_processed_FULL is None, snapshot 'ts_tool_df_FULL_at_missing_ui_entry' set to None.")
        
        session_state.force_missing_ui_snapshot_refresh_flag = False # Reset flag
        print("[DEBUG missing_data_ui] force_missing_ui_snapshot_refresh_flag reset to False.")
    # --- <<< 结束快照逻辑 >>> ---

    if not session_state.ts_tool_processing_applied or session_state.ts_tool_data_processed is None or session_state.ts_tool_data_processed.empty:
        return # Don't display if no processed data

    st.divider()
    col_title_missing, col_reset_button_missing = st.columns([0.7, 0.3]) # Adjust ratio as needed
    with col_title_missing:
        st.subheader("缺失数据分析与处理")
    with col_reset_button_missing:
        # --- MODIFIED: Removed CSS, simplified button ---
        if st.button("重置",  # Changed button text
                     key="reset_missing_data_button", 
                     on_click=reset_missing_data_state, 
                     help="将数据恢复到进入缺失数据处理模块时的状态，并清除此模块中的所有分析、绘图和填充设置。"):
            pass 
        # --- END OF MODIFICATION ---
    # st.subheader("缺失数据分析与处理") # This line is now part of the columns above
    st.caption("选择列进行缺失数据分析，可视化缺失模式，并选择合适的方法处理缺失值。")

    left_col_processing, right_col_analysis = st.columns(2)

    # Initialize placeholders in session_state if they don't exist
    if 'analysis_placeholder_missing_ui' not in session_state:
        # Create a unique key for the placeholder if it's specific to this instance/module
        session_state.analysis_placeholder_missing_ui_key = f"analysis_placeholder_{str(id(st))}"
        session_state[session_state.analysis_placeholder_missing_ui_key] = st.empty()
    if 'plot_placeholder_missing_ui' not in session_state:
        session_state.plot_placeholder_missing_ui_key = f"plot_placeholder_{str(id(st))}"
        session_state[session_state.plot_placeholder_missing_ui_key] = st.empty()

    # Retrieve the actual empty containers
    analysis_placeholder = session_state[session_state.analysis_placeholder_missing_ui_key]
    plot_placeholder = session_state[session_state.plot_placeholder_missing_ui_key]

    with right_col_analysis:
        st.markdown("##### 缺失值填充方法") 
        
        cols_selected_for_analysis = session_state.get('missing_ui_selected_cols', [])
        if not cols_selected_for_analysis:
            st.info('请先在左侧"分析与可视化"部分选择要操作的列，然后在此处选择标准填充方法。')
        
        fill_method_options = {
            "无操作": "none", "向前填充 (ffill)": "ffill", "向后填充 (bfill)": "bfill",
            "用0填充 (仅数值型)": "zero", "用指定常量填充": "constant",
            "线性插值 (仅数值型)": "linear", "用均值填充 (仅数值型)": "mean",
            "用中位数填充 (仅数值型)": "median", "用众数填充": "mode"
        }
        selected_fill_method_label = st.selectbox(
            label="选择标准填充方法 (作用于左侧选定的列):",
            options=list(fill_method_options.keys()), 
            index=0, 
            key="fill_method_selector_new_right_col",
            disabled=not cols_selected_for_analysis
        )
        selected_fill_method = fill_method_options[selected_fill_method_label]
        fill_limit = None
        constant_val_for_fill = None

        if selected_fill_method in ["ffill", "bfill"]:
            fill_limit_str = st.text_input("连续填充的最大数量 (可选):", key="fill_limit_input_new_right_col", disabled=not cols_selected_for_analysis)
            if fill_limit_str.strip():
                try:
                    fill_limit = int(fill_limit_str.strip())
                    if fill_limit <= 0: 
                        st.warning("填充限制必须是正整数。将忽略。")
                        fill_limit = None
                except ValueError: 
                    st.warning("填充限制必须是整数。将忽略。")
                    fill_limit = None
        elif selected_fill_method == "constant":
            constant_val_for_fill = st.text_input("输入用于填充的常量值:", key="constant_value_input_new_right_col", disabled=not cols_selected_for_analysis)

        if st.button("应用标准填充方法", key="apply_fill_button_new_right_col", disabled=not cols_selected_for_analysis):
            if selected_fill_method == "none":
                st.info("选择了'无操作'，未对数据进行更改。")
            elif selected_fill_method == "constant" and not constant_val_for_fill.strip():
                st.error("使用常量填充时，必须提供常量值。")
            else:
                try:
                    df_filled, message = handle_missing_values(
                        session_state.ts_tool_data_processed, 
                        cols_selected_for_analysis, 
                        method=selected_fill_method, 
                        limit=fill_limit, 
                        constant_value=constant_val_for_fill
                    )
                    session_state.ts_tool_data_processed = df_filled
                    session_state.ts_tool_data_final = df_filled
                    if "成功" in message: 
                        st.success(message)
                    else: 
                        st.info(message)
                    analysis_placeholder.empty()
                    plot_placeholder.empty() 
                    # <<< RERUN_DEBUG_MISSING_UI >>>
                    # ❌ 移除不必要的rerun调用
                    # print("RERUN_DEBUG_MISSING_UI: APPLY_FILL_RERUN: Calling st.rerun() after applying fill method.")
                    # st.rerun()
                except Exception as e_fill: 
                    st.error(f"处理缺失值时出错: {e_fill}")

    with left_col_processing:
        st.markdown("##### 分析与可视化")
        current_data_for_analysis = session_state.get('ts_tool_data_processed')
        all_available_cols_for_missing_left = current_data_for_analysis.columns.tolist() if current_data_for_analysis is not None and not current_data_for_analysis.empty else []

        if 'missing_ui_selected_cols' not in session_state: 
            session_state.missing_ui_selected_cols = []
        
        valid_current_selection_missing_left = [col for col in session_state.missing_ui_selected_cols if col in all_available_cols_for_missing_left]
        # <<< RERUN_DEBUG_MISSING_UI >>>
        if session_state.missing_ui_selected_cols != valid_current_selection_missing_left:
            # ❌ 移除过度的rerun调用，状态同步已完成
            # print("RERUN_DEBUG_MISSING_UI: VALIDATION_RERUN: Condition met.")
            # print(f"RERUN_DEBUG_MISSING_UI: VALIDATION_RERUN: session_state.missing_ui_selected_cols (before): {session_state.missing_ui_selected_cols}")
            # print(f"RERUN_DEBUG_MISSING_UI: VALIDATION_RERUN: valid_current_selection_missing_left: {valid_current_selection_missing_left}")
            # print(f"RERUN_DEBUG_MISSING_UI: VALIDATION_RERUN: all_available_cols_for_missing_left: {all_available_cols_for_missing_left}")
            session_state.missing_ui_selected_cols = valid_current_selection_missing_left
            # print(f"RERUN_DEBUG_MISSING_UI: VALIDATION_RERUN: session_state.missing_ui_selected_cols (after): {session_state.missing_ui_selected_cols}")
            # print("RERUN_DEBUG_MISSING_UI: VALIDATION_RERUN: Calling st.rerun().")
            # st.rerun() 

        action_buttons_col_left1, action_buttons_col_left2 = st.columns([1,1])
        with action_buttons_col_left1:
            if st.button("全选所有列 (分析)", key="select_all_cols_new_left_analyze", use_container_width=True, disabled=not all_available_cols_for_missing_left):
                session_state.missing_ui_selected_cols = all_available_cols_for_missing_left
                # <<< RERUN_DEBUG_MISSING_UI >>>
                # ❌ 移除不必要的rerun调用
                # print("RERUN_DEBUG_MISSING_UI: SELECT_ALL_RERUN: Calling st.rerun().")
                # st.rerun()
        with action_buttons_col_left2:
            if st.button("清除所有选择 (分析)", key="clear_all_cols_new_left_analyze", use_container_width=True, disabled=not session_state.missing_ui_selected_cols):
                session_state.missing_ui_selected_cols = []
                # <<< RERUN_DEBUG_MISSING_UI >>>
                # ❌ 移除不必要的rerun调用
                # print("RERUN_DEBUG_MISSING_UI: CLEAR_ALL_RERUN: Calling st.rerun().")
                # st.rerun()

        selected_cols_in_multiselect_left = st.multiselect(
            "选择要分析和绘图的列 (右侧的标准填充也将作用于这些列):",
            options=all_available_cols_for_missing_left,
            default=session_state.missing_ui_selected_cols, 
            key="missing_data_col_select_main_new_left_analyze",
            placeholder="点击选择列...",
            disabled=not all_available_cols_for_missing_left
        )
        if selected_cols_in_multiselect_left != session_state.missing_ui_selected_cols:
            # <<< RERUN_DEBUG_MISSING_UI >>>
            # ❌ 移除不必要的rerun调用，multiselect会自动处理状态变更
            # print("RERUN_DEBUG_MISSING_UI: MULTISELECT_RERUN: Multiselect changed.")
            # print(f"RERUN_DEBUG_MISSING_UI: MULTISELECT_RERUN: selected_cols_in_multiselect_left: {selected_cols_in_multiselect_left}")
            # print(f"RERUN_DEBUG_MISSING_UI: MULTISELECT_RERUN: session_state.missing_ui_selected_cols (before): {session_state.missing_ui_selected_cols}")
            session_state.missing_ui_selected_cols = selected_cols_in_multiselect_left
            # print("RERUN_DEBUG_MISSING_UI: MULTISELECT_RERUN: Calling st.rerun().")
            # st.rerun()
        
        if not all_available_cols_for_missing_left:
            st.info("当前没有可供分析的数据列。请先加载并处理数据。")
        elif not session_state.missing_ui_selected_cols and all_available_cols_for_missing_left :
            st.info("请选择至少一列以进行分析和可视化。")

        if st.button("执行缺失数据分析", key="analyze_missing_button_new_left_analyze", disabled=not session_state.missing_ui_selected_cols):
            if current_data_for_analysis is not None and not current_data_for_analysis.empty and session_state.missing_ui_selected_cols:
                with st.spinner("正在分析缺失数据并生成图表..."):
                    # Perform analysis
                    analysis_results = analyze_missing_data(current_data_for_analysis, session_state.missing_ui_selected_cols)
                    
                    # 准备分析结果 DataFrame 并进行格式化
                    analysis_df = pd.DataFrame(analysis_results).set_index('column')
                    
                    # 重新排列列顺序，把新增的字段放在合适的位置
                    if 'first_record_time' in analysis_df.columns and 'non_missing_count' in analysis_df.columns:
                        # 定义新的列顺序
                        new_columns = [
                            'first_record_time',  # 变量第一次记录的时间
                            'non_missing_count',  # 无缺失值的期数
                            'total_missing_pct',  # 总缺失百分比
                            'post_first_record_missing_pct',  # 首次记录后缺失百分比
                            'max_consecutive_missing'  # 最大连续缺失数
                        ]
                        # 确保所有列都存在
                        valid_cols = [col for col in new_columns if col in analysis_df.columns]
                        # 重新排序
                        analysis_df = analysis_df[valid_cols]
                    
                    # 显示分析结果
                    analysis_placeholder.dataframe(
                        analysis_df.style.format({
                            "total_missing_pct": "{:.2f}%",
                            "post_first_record_missing_pct": lambda x: f"{x:.2f}%" if isinstance(x, float) else x,
                            "max_consecutive_missing": "{:}",
                            "non_missing_count": "{:}"  # 无缺失值期数，整数格式
                        }), 
                        use_container_width=True
                    )

                    # Generate missing data pattern plot
                    # --- MODIFICATION: Ensure time column name is passed if available ---
                    time_col_name_for_plot = None
                    if 'ts_tool_time_col_info' in session_state and session_state.ts_tool_time_col_info:
                        time_col_name_for_plot = session_state.ts_tool_time_col_info.get('name')
                    
                    # --- Get explicit domain for the plot from session state if available ---
                    # These are set by the time filter UI and represent the active filter range
                    # Use the keys from the main time filter UI
                    explicit_start_for_plot = session_state.get("ts_tool_active_filter_start") 
                    explicit_end_for_plot = session_state.get("ts_tool_active_filter_end")
                    print(f"[Missing UI Plot Call] Time Col: {time_col_name_for_plot}, Start: {explicit_start_for_plot}, End: {explicit_end_for_plot}")


                    missing_pattern_plot = generate_missing_data_plot(
                        current_data_for_analysis, 
                        session_state.missing_ui_selected_cols,
                        time_col_name=time_col_name_for_plot,
                        explicit_domain_start=explicit_start_for_plot, # Pass explicit start
                        explicit_domain_end=explicit_end_for_plot     # Pass explicit end
                    )
                    
                    plot_placeholder.empty() # Clear previous plot/message
                    if missing_pattern_plot:
                        # --- MODIFIED LINE: Use pyplot to display the Matplotlib figure ---
                        plot_placeholder.pyplot(missing_pattern_plot, use_container_width=True)
                        # --- END OF MODIFICATION ---
                    else:
                        plot_placeholder.info("未能生成缺失数据模式图。可能是由于数据问题或没有有效的时间轴。")

                st.success("缺失数据分析完成。统计结果和模式图已更新。")
            else:
                st.warning("没有选择列或数据为空，无法执行分析。")

    st.divider() 