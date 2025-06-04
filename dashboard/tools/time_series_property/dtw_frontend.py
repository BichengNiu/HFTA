import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib # To ensure backend is suitable

# Import backend function
from . import dtw_backend 

# Helper function to plot DTW path (kept in frontend as it uses st_obj)
def plot_dtw_path(st_obj, s1_np, s2_np, path, s1_name, s2_name):
    fig, ax = plt.subplots(figsize=(10, 5))

    ax.plot(np.arange(len(s1_np)), s1_np, "o-", label=s1_name, markersize=4, linewidth=1.5)
    ax.plot(np.arange(len(s2_np)), s2_np, "s-", label=s2_name, markersize=4, linewidth=1.5)

    for idx1, idx2 in path:
        if idx1 < len(s1_np) and idx2 < len(s2_np): # Defensive check
            ax.plot([idx1, idx2], [s1_np[idx1], s2_np[idx2]], color='grey', linestyle='--', linewidth=0.8, alpha=0.7)

    ax.legend()
    ax.set_title(f"DTW Alignment: {s1_name} vs {s2_name}")
    ax.set_xlabel("Time Index")
    ax.set_ylabel("Value")
    ax.grid(True, linestyle=':', alpha=0.6)
    plt.tight_layout()
    st_obj.pyplot(fig)
    plt.close(fig) # Close the figure to free memory

def display_dtw_tab(st_obj, session_state):
    # Data acquisition from session_state (set by dashboard.py)
    selected_df_name = session_state.get('correlation_selected_df_name') # Assuming this is the relevant key
    df = session_state.get('correlation_selected_df')

    if df is None or df.empty:
        st_obj.info("请先在主界面选择一个暂存数据集以进行DTW分析。")
        return
    
    # Initialize/reset session state for batch results based on dataset name
    # Using a more specific key for DTW to avoid conflicts
    dtw_processed_dataset_key = f'dtw_processed_dataset_for_{selected_df_name}'
    if not session_state.get(dtw_processed_dataset_key, False):
        # Reset states specific to DTW tab when dataset changes or first load
        keys_to_reset_dtw = [
            'batch_dtw_results', 'batch_dtw_paths', 
            'dtw_target_series_for_results', 'dtw_plot_selection_corr_v1',
            'dtw_target_series_corr_v1', 'dtw_comparison_series_corr_v1',
            'dtw_window_type_corr_v1', 'dtw_window_size_corr_v1',
            'dtw_distance_metric_display_corr_v1'
        ]
        for key_to_reset in keys_to_reset_dtw:
            if hasattr(session_state, key_to_reset):
                del session_state[key_to_reset]
        
        # Mark this dataset as processed for DTW state initialization
        session_state[dtw_processed_dataset_key] = True
        # Clean up flags for other datasets if they exist
        for key in list(session_state.keys()):
            if key.startswith('dtw_processed_dataset_for_') and key != dtw_processed_dataset_key:
                del session_state[key]
        print(f"[DTW Frontend] Dataset changed to '{selected_df_name}' or first load. Resetting UI states.")

    col1, col2 = st_obj.columns(2)

    with col1:
        col1.markdown("### 参数输入")
        series_options_all = list(df.columns)
        if not series_options_all:
            col1.warning("选定的数据集中没有可用的列。")
            return

        current_target_val = session_state.get('dtw_target_series_corr_v1', series_options_all[0] if series_options_all else None)
        try:
            target_default_idx = series_options_all.index(current_target_val) if current_target_val in series_options_all else 0
        except ValueError:
            target_default_idx = 0
        
        session_state.dtw_target_series = col1.selectbox("选择目标变量:", series_options_all, 
                                                       key='dtw_target_series_corr_v1', 
                                                       index=target_default_idx)

        target_series_val = session_state.dtw_target_series
        comparison_options = [s for s in series_options_all if s != target_series_val]
        default_comparisons = session_state.get('dtw_comparison_series_corr_v1', comparison_options)
        default_comparisons = [opt for opt in default_comparisons if opt in comparison_options]

        if not comparison_options:
            col1.caption("数据集中没有其他可供对比的变量。")
            session_state.dtw_comparison_series = []
        else:
            session_state.dtw_comparison_series = col1.multiselect("选择对比变量:", comparison_options,
                                                                default=default_comparisons, 
                                                                key='dtw_comparison_series_corr_v1')
        col1.markdown("---")
        col1.markdown("#### DTW 参数设置")

        window_type_options = ["无限制", "固定大小窗口 (Sakoe-Chiba Band)"]
        current_window_type = session_state.get("dtw_window_type_corr_v1", window_type_options[0])
        try: wt_idx = window_type_options.index(current_window_type) 
        except ValueError: wt_idx = 0
        session_state.dtw_window_type = col1.radio("窗口限制类型:", window_type_options, key="dtw_window_type_corr_v1", horizontal=True, index=wt_idx)

        if session_state.dtw_window_type == "固定大小窗口 (Sakoe-Chiba Band)":
            df_len_for_default = len(df) if df is not None and not df.empty else 100
            default_window_size = max(1, int(df_len_for_default * 0.1))
            current_window_size = session_state.get("dtw_window_size_corr_v1", default_window_size)
            session_state.dtw_window_size = col1.number_input("窗口大小 (点数):", min_value=1, value=current_window_size, step=1, key="dtw_window_size_corr_v1")
        else:
            session_state.dtw_window_size = None

        distance_metric_options = {"欧氏距离 (Euclidean)":"euclidean", "曼哈顿距离 (Manhattan)":"manhattan", "平方欧氏距离 (Squared Euclidean)":"sqeuclidean"}
        metric_keys = list(distance_metric_options.keys())
        current_metric_key = session_state.get("dtw_distance_metric_display_corr_v1", metric_keys[0])
        try: mk_idx = metric_keys.index(current_metric_key)
        except ValueError: mk_idx = 0
        selected_metric_display = col1.selectbox("距离度量方法:", metric_keys, key="dtw_distance_metric_display_corr_v1", index=mk_idx)
        session_state.dtw_distance_metric = distance_metric_options[selected_metric_display]
        
        execute_button_clicked = col1.button("执行批量DTW分析", key="dtw_batch_calculate_button_v1")

    with col2:
        col2.markdown("### 计算结果")
        if execute_button_clicked:
            # Clear previous run results from session state before populating new ones
            session_state.batch_dtw_results = []
            session_state.batch_dtw_paths = {}
            session_state.dtw_target_series_for_results = None

            target_series_name_input = session_state.dtw_target_series
            comparison_series_names_input = session_state.dtw_comparison_series
            window_type_param_input = session_state.dtw_window_type
            window_size_param_input = session_state.dtw_window_size
            dist_metric_name_param_input = session_state.dtw_distance_metric
            dist_metric_display_param_input = selected_metric_display # From col1 selectbox

            # Call backend function
            with col2:
                with st.spinner(f"正在为目标 '{target_series_name_input}' 计算DTW..."):
                    results_list, paths_dict, errors, warnings = dtw_backend.perform_batch_dtw_calculation(
                        df_input=df,
                        target_series_name=target_series_name_input,
                        comparison_series_names=comparison_series_names_input,
                        window_type_param=window_type_param_input,
                        window_size_param=window_size_param_input,
                        dist_metric_name_param=dist_metric_name_param_input,
                        dist_metric_display_param=dist_metric_display_param_input
                    )
            
            # Store results in session state
            session_state.batch_dtw_results = results_list
            session_state.batch_dtw_paths = paths_dict
            if not errors and not warnings and results_list: # If calculation was successful
                 session_state.dtw_target_series_for_results = target_series_name_input
            
            for err_msg in errors:
                col2.error(err_msg)
            for warn_msg in warnings:
                col2.warning(warn_msg)
            
            if results_list and not errors: # Only show success if some results and no critical errors
                col2.success("批量DTW分析处理完成！")
            elif not results_list and not errors and not warnings: # e.g. no comparison series selected by user
                col2.info("没有对比序列被选择或有效，未执行计算。")

        # Display logic (always attempts to render based on current session_state)
        batch_results_data = session_state.get('batch_dtw_results')
        target_name_for_display = session_state.get('dtw_target_series_for_results')

        if batch_results_data and target_name_for_display:
            results_df = pd.DataFrame(batch_results_data)
            if 'DTW距离' in results_df.columns:
                results_df['DTW距离'] = pd.to_numeric(results_df['DTW距离'], errors='coerce')
                results_df = results_df.sort_values(by='DTW距离', ascending=True)

            cols_order = ['对比变量', 'DTW距离', '窗口类型', '窗口大小', '距离度量', '原因']
            ordered_cols_to_display = [col for col in cols_order if col in results_df.columns]
            results_df_display = results_df[ordered_cols_to_display]

            col2.markdown(f"#### DTW结果报告: 与目标 '{target_name_for_display}' 的比较")
            col2.dataframe(
                results_df_display,
                column_config={
                    "DTW距离": st.column_config.NumberColumn("DTW距离", format="%.2f")
                }
            )

            batch_paths_data = session_state.get('batch_dtw_paths')
            if batch_paths_data:
                plottable_compare_vars = [name for name, data in batch_paths_data.items() if data.get('path')]
                if plottable_compare_vars:
                    current_plot_selection_val = session_state.get('dtw_plot_selection_corr_v1')
                    if not current_plot_selection_val or current_plot_selection_val not in plottable_compare_vars:
                        current_plot_selection_val = plottable_compare_vars[0]
                    
                    try:
                        plot_select_idx = plottable_compare_vars.index(current_plot_selection_val)
                    except ValueError:
                        plot_select_idx = 0

                    # Use a different key for the widget to avoid potential conflicts if dtw_plot_selection_corr_v1 is set by non-widget logic
                    selected_compare_for_plot_widget = col2.selectbox(
                        "选择对比变量以显示DTW对齐图:", 
                        options=plottable_compare_vars, 
                        key='dtw_plot_selection_corr_v1_widget', 
                        index=plot_select_idx
                    )
                    # Update the primary session state variable from the widget's current value
                    session_state.dtw_plot_selection_corr_v1 = selected_compare_for_plot_widget
                    
                    # Read from the primary session state variable for plotting
                    selected_compare_for_plot_state = session_state.dtw_plot_selection_corr_v1

                    if selected_compare_for_plot_state and selected_compare_for_plot_state in batch_paths_data:
                        path_info = batch_paths_data[selected_compare_for_plot_state]
                        plot_dtw_path(col2, path_info['target_np'], path_info['compare_np'], 
                                      path_info['path'], target_name_for_display, selected_compare_for_plot_state)
                else:
                    col2.caption("没有可绘制的DTW路径（可能所有对比计算均失败或序列为空）。")
            elif not results_df.empty and results_df['DTW距离'].isna().all():
                 col2.caption("所有DTW计算均未成功生成路径数据。")
        
        elif execute_button_clicked: # Button was clicked but no valid results to display
            # Error/warning messages for invalid inputs are shown during the backend call handling
            # If batch_results_data is empty or target_name_for_display is None, it implies issues handled above.
            pass
        else:
            col2.caption("点击左侧 \"执行批量DTW分析\" 按钮后，结果将显示在此处。") 