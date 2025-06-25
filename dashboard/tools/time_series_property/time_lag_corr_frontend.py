import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Import backend functions
from . import time_lag_corr_backend

def plot_time_lag_correlogram(st_obj, results_df: pd.DataFrame, series1_name: str, series2_name: str):
    """绘制时差相关图 (Correlogram)"""
    if results_df.empty or 'Lag' not in results_df.columns or 'Correlation' not in results_df.columns:
        st_obj.warning("无法绘制时差相关图：结果数据不完整或为空。")
        return

    fig, ax = plt.subplots(figsize=(10, 5))
    
    ax.bar(results_df['Lag'], results_df['Correlation'], color='skyblue', width=0.8)
    
    # Find max absolute correlation for highlighting
    if 'Correlation' in results_df.columns and results_df['Correlation'].notna().any():
        # Consider only positive lags for "leading" relationship highlight as per original intent
        positive_lags_df = results_df[results_df['Lag'] > 0]
        if not positive_lags_df.empty and positive_lags_df['Correlation'].notna().any():
            max_corr_idx_positive = positive_lags_df['Correlation'].abs().idxmax()
            if pd.notna(max_corr_idx_positive) and max_corr_idx_positive in positive_lags_df.index: # Check index validity
                max_lag = positive_lags_df.loc[max_corr_idx_positive, 'Lag']
                max_corr_val = positive_lags_df.loc[max_corr_idx_positive, 'Correlation']
                ax.plot(max_lag, max_corr_val, 'ro', markersize=8, label=f'最优领先相关: {max_corr_val:.3f} (滞后 {max_lag})')
                ax.vlines(max_lag, 0, max_corr_val, colors='r', linestyles='dotted', alpha=0.7)
        elif results_df['Correlation'].notna().any(): # Fallback to overall max if no positive lag correlation
            max_corr_idx_overall = results_df['Correlation'].abs().idxmax()
            if pd.notna(max_corr_idx_overall) and max_corr_idx_overall in results_df.index:
                max_lag_overall = results_df.loc[max_corr_idx_overall, 'Lag']
                max_corr_val_overall = results_df.loc[max_corr_idx_overall, 'Correlation']
                ax.plot(max_lag_overall, max_corr_val_overall, 'go', markersize=6, label=f'最大绝对相关: {max_corr_val_overall:.3f} (滞后 {max_lag_overall})')

    ax.axhline(0, color='black', linewidth=0.8, linestyle='--')
    ax.set_xlabel("滞后/超前阶数 (Lag)")
    ax.set_ylabel("皮尔逊相关系数")
    ax.set_title(f"时差相关性: {series1_name} vs {series2_name}")
    ax.grid(True, linestyle=':', alpha=0.6)
    ax.legend()
    plt.tight_layout()
    st_obj.pyplot(fig)
    plt.close(fig)

def display_time_lag_corr_tab(st_obj, session_state):
    # Data acquisition from session_state (set by dashboard.py)
    # Using 'tlc_own_selected_df_name' and 'tlc_own_selected_df' as per original for this tab
    selected_df_name = session_state.get('tlc_own_selected_df_name') 
    df = session_state.get('tlc_own_selected_df')

    if df is None or df.empty:
        st_obj.info("请先在此标签页上方选择一个暂存数据集以进行时差相关性分析。")
        return
    
    # --- Session State Management for UI persistence --- 
    # Using a unique prefix for this tab's session state keys to ensure isolation
    tab_prefix = f"tlc_{selected_df_name}" 
    
    processed_dataset_flag_key = f"{tab_prefix}_processed_flag_v4"
    lagged_variable_key = f"{tab_prefix}_lagged_variable_v4"
    leading_variables_key = f"{tab_prefix}_leading_variables_v4"
    max_leading_periods_key = f"{tab_prefix}_max_periods_v4"
    batch_results_key = f"{tab_prefix}_batch_results_v4"
    selected_for_plot_key = f"{tab_prefix}_selected_for_plot_v4"
    calc_triggered_key = f"{tab_prefix}_calc_triggered_v4"
    # For defaulting leading vars when lagged var changes
    lagged_var_defaulting_tracker_key = f"{tab_prefix}_lagged_var_tracker_v4" 

    if not session_state.get(processed_dataset_flag_key, False):
        keys_to_reset = [
            lagged_variable_key, leading_variables_key, max_leading_periods_key,
            batch_results_key, selected_for_plot_key, calc_triggered_key,
            lagged_var_defaulting_tracker_key
        ]
        for key_to_del in keys_to_reset:
            if hasattr(session_state, key_to_del):
                del session_state[key_to_del]
        session_state[processed_dataset_flag_key] = True
        # Clean up flags for other datasets
        for key in list(session_state.keys()):
            if key.startswith("tlc_") and key.endswith("_processed_flag_v4") and key != processed_dataset_flag_key:
                del session_state[key]
        print(f"[TLC Frontend] Dataset changed to '{selected_df_name}' or first load for TLC. Resetting specific UI states.")

    col1, col2 = st_obj.columns(2)

    with col1:
        col1.markdown("### 参数输入")
        series_options_all = [col for col in df.columns if pd.api.types.is_numeric_dtype(df[col])]
        if not series_options_all:
            col1.warning("选定的数据集中没有可用的数值类型列。")
            return

        # 1. Lagged Variable
        lagged_var_current_selection = session_state.get(lagged_variable_key, series_options_all[0] if series_options_all else None)
        try: 
            lagged_var_idx = series_options_all.index(lagged_var_current_selection) if lagged_var_current_selection in series_options_all else 0
        except ValueError: lagged_var_idx = 0
        
        selected_lagged_variable_widget = col1.selectbox(
            "**选择滞后变量 (被预测的变量)**", 
            options=series_options_all, 
            key=f"{lagged_variable_key}_widget", # Distinct widget key
            index=lagged_var_idx
        )
        # Update main state key, and check if it changed for leading var defaulting
        if selected_lagged_variable_widget != session_state.get(lagged_variable_key):
            session_state[lagged_variable_key] = selected_lagged_variable_widget
            # Mark that lagged variable changed to reset leading variables multiselect default
            session_state[lagged_var_defaulting_tracker_key] = selected_lagged_variable_widget 
            if leading_variables_key in session_state: # Clear previous leading var selection
                del session_state[leading_variables_key]
            st_obj.rerun() # Rerun to update leading var options and default
        
        # Get the authoritative value from session_state for logic
        actual_selected_lagged_var = session_state.get(lagged_variable_key)

        # 2. Leading Variables
        leading_options = [s for s in series_options_all if s != actual_selected_lagged_var]
        default_leading_vars_for_widget = leading_options[:] # Default to all if lagged var just changed or no prior selection
        if session_state.get(lagged_var_defaulting_tracker_key) == actual_selected_lagged_var and leading_variables_key in session_state:
            # Lagged var hasn't changed since last default setting, and there's a previous selection
            current_selected_leading_state = session_state.get(leading_variables_key, [])
            default_leading_vars_for_widget = [opt for opt in current_selected_leading_state if opt in leading_options]
            if not default_leading_vars_for_widget and leading_options: # If all previous became invalid
                default_leading_vars_for_widget = leading_options[:]
        else: # Lagged var changed, or first time: set tracker and use all as default
             session_state[lagged_var_defaulting_tracker_key] = actual_selected_lagged_var

        # 添加领先变量快捷选择按钮
        if leading_options:
            col_select_all_leading, col_clear_all_leading = col1.columns(2)
            with col_select_all_leading:
                if col1.button("全选领先变量", key=f"{tab_prefix}_select_all_leading", use_container_width=True):
                    session_state[leading_variables_key] = leading_options[:]
                    col1.rerun()
            with col_clear_all_leading:
                if col1.button("清除全部选择", key=f"{tab_prefix}_clear_all_leading", use_container_width=True):
                    session_state[leading_variables_key] = []
                    col1.rerun()

        selected_leading_variables_widget = col1.multiselect(
            "**选择领先变量 (可多选)**", 
            options=leading_options, 
            default=default_leading_vars_for_widget, 
            key=f"{leading_variables_key}_widget", # Distinct widget key
            disabled=not leading_options,
            help="💡 提示：使用上方按钮可以快速选择多个领先变量"
        )
        session_state[leading_variables_key] = selected_leading_variables_widget # Update main state key
        actual_selected_leading_vars = session_state.get(leading_variables_key, [])

        # 3. Max Leading Periods
        max_len_for_input = max(1, len(df) - 1 if df is not None and not df.empty else 50)
        current_max_periods_val = session_state.get(max_leading_periods_key, min(12, max_len_for_input))
        current_max_periods_val = max(1, min(current_max_periods_val, max_len_for_input))
        selected_max_periods_widget = col1.number_input(
            "**最大领先阶数**", 
            min_value=1, max_value=max_len_for_input, 
            value=current_max_periods_val, step=1, 
            key=f"{max_leading_periods_key}_widget", # Distinct widget key
            help="希望向前探索多少个周期。"
        )
        session_state[max_leading_periods_key] = selected_max_periods_widget # Update main state key
        actual_max_periods = session_state.get(max_leading_periods_key, 1)

        if col1.button("计算领先关系", key=f"{tab_prefix}_calc_button_v4", 
                       disabled=not actual_selected_lagged_var or not actual_selected_leading_vars):
            session_state[calc_triggered_key] = True # Mark calculation attempt
            session_state[batch_results_key] = [] # Clear previous results
            
            with st.spinner("正在计算领先关系..."):
                results, errors, warnings = time_lag_corr_backend.perform_batch_time_lag_correlation(
                    df_input=df,
                    selected_lagged_variable=actual_selected_lagged_var,
                    selected_leading_variables_list=actual_selected_leading_vars,
                    selected_max_leading_periods=actual_max_periods
                )
            session_state[batch_results_key] = results
            for err_msg in errors: col1.error(err_msg) # Show errors in col1 directly
            for warn_msg in warnings: col1.warning(warn_msg)

            if results: # Set default for plot if results were generated
                first_plottable = next((res['领先变量'] for res in results if isinstance(res.get('correlogram_df'), pd.DataFrame) and not res.get('correlogram_df').empty), None)
                if first_plottable:
                    session_state[selected_for_plot_key] = first_plottable
                elif actual_selected_leading_vars: # Fallback if no correlograms but vars were selected
                    session_state[selected_for_plot_key] = actual_selected_leading_vars[0]
            else:
                session_state.pop(selected_for_plot_key, None) # Clear plot selection if no results
            st_obj.rerun() # Rerun to update col2 with results

    with col2:
        col2.markdown("### 计算结果汇总")
        calc_attempted = session_state.get(calc_triggered_key, False)
        current_batch_results = session_state.get(batch_results_key)

        if calc_attempted and current_batch_results is not None:
            if current_batch_results:
                results_summary_df = pd.DataFrame(current_batch_results)
                col2.markdown("#### 结果摘要")
                display_cols = ['领先变量', '最优领先阶数', '相关系数 (最优领先时)', '备注']
                actual_display_cols = [col for col in display_cols if col in results_summary_df.columns]
                col2.dataframe(
                    results_summary_df[actual_display_cols],
                    column_config={
                        "相关系数 (最优领先时)": st.column_config.NumberColumn(format="%.3f"),
                        "最优领先阶数": st.column_config.NumberColumn(format="%d 阶")
                    },
                    hide_index=True
                )

                plottable_vars = [res['领先变量'] for res in current_batch_results if isinstance(res.get('correlogram_df'), pd.DataFrame) and not res['correlogram_df'].empty]
                if plottable_vars:
                    current_plot_selection = session_state.get(selected_for_plot_key, plottable_vars[0] if plottable_vars else None)
                    try:
                        plot_idx = plottable_vars.index(current_plot_selection) if current_plot_selection in plottable_vars else 0
                    except ValueError: plot_idx = 0
                    
                    selected_var_for_plot_widget = col2.selectbox(
                        "选择领先变量以显示完整时差相关图:", 
                        options=plottable_vars, 
                        index=plot_idx,
                        key=f"{selected_for_plot_key}_widget"
                    )
                    if selected_var_for_plot_widget != session_state.get(selected_for_plot_key):
                        session_state[selected_for_plot_key] = selected_var_for_plot_widget
                        st_obj.rerun() # Rerun if selection changes
                    
                    # Plot based on the session state value
                    final_selected_var_for_plot = session_state.get(selected_for_plot_key)
                    if final_selected_var_for_plot:
                        plot_data_entry = next((item for item in current_batch_results if item['领先变量'] == final_selected_var_for_plot), None)
                        if plot_data_entry and isinstance(plot_data_entry.get('correlogram_df'), pd.DataFrame) and not plot_data_entry['correlogram_df'].empty:
                            lagged_var_for_plot_title = plot_data_entry.get('滞后变量', '滞后变量') # Fallback name
                            plot_time_lag_correlogram(col2, plot_data_entry['correlogram_df'], 
                                                      lagged_var_for_plot_title, final_selected_var_for_plot)
                        else:
                            col2.caption(f"领先变量 '{final_selected_var_for_plot}' 无相关图数据。")
                else:
                    col2.caption("没有可绘制时差相关图的结果。")
            else: # current_batch_results is empty list
                col2.info("计算完成，但未生成任何有效的领先关系结果。请检查输入。")
        elif not calc_attempted:
            col2.caption("请在左侧选择参数并点击 \"计算领先关系\"。") 