import streamlit as st
import pandas as pd
import numpy as np
import re # For frequency display from main tab if needed for context
import io

def display_visualization_section(st, session_state):
    """处理数据可视化的UI和逻辑。"""

    # This section is shown only if data is loaded.
    # The check 'if session_state.ts_compute_data is not None:' is handled by the calling function in main tab.

    st.subheader("查看时间序列") # Or appropriate section number
    df_current_viz = session_state.ts_compute_data
    numeric_cols_viz = df_current_viz.select_dtypes(include=np.number).columns.tolist()

    # --- Date range and frequency context (from other components or main tab) ---
    # These are ideally passed or reliably fetched if set by variable_calculations_ui
    date_range_available = session_state.get('ts_compute_date_range_available', False)
    start_date = session_state.get('ts_compute_start_date_vc', None)
    end_date = session_state.get('ts_compute_end_date_vc', None)
    
    # Infer frequency (this logic might be duplicated or passed from main tab/another utility)
    # For now, re-inferring for context within this component if needed for comparison plots
    freq = None
    freq_display_viz = "无法确定频率"
    if isinstance(df_current_viz.index, pd.DatetimeIndex):
        try:
            freq = pd.infer_freq(df_current_viz.index)
            if freq:
                # Simplified map for brevity, expand as in main tab if full detail needed
                freq_map_viz = {'A': "年度", 'Q': "季度", 'M': "月度", 'W': "周度", 'D': "日度", 'B': "工作日"}
                for prefix, display in freq_map_viz.items():
                    if freq.startswith(prefix): freq_display_viz = display; break
                else: freq_display_viz = f"检测到: {freq}"
            else:
                freq_display_viz = "不规则"
                # Could add more detailed irregularity info if essential for viz options here
        except Exception: # Broad except for frequency inference issues
            freq_display_viz = "推断频率出错"
    # --- End frequency context ---

    if not numeric_cols_viz:
        st.warning("数据中没有可用于可视化的数值列。")
        return

    st.write("选择变量和图表类型进行可视化：")
    selected_variables_viz = st.multiselect("选择绘制变量:", numeric_cols_viz, key="viz_selected_variables_viz") # Added _viz suffix
    plot_type_viz = st.selectbox("选择图表类型:", ["请选择...", "折线图", "折线图 (带数据点)", "堆积图", "同期对比图"], key="viz_plot_type_viz")

    y_axis_scale_viz = "线性"
    legend_visibility_viz = "自动"

    if selected_variables_viz and plot_type_viz not in ["请选择...", "堆积图", "同期对比图"]:
        y_axis_scale_viz = st.selectbox("Y轴刻度类型:", ["线性", "对数 (自然 e)", "对数 (底为10)"], key="viz_y_axis_scale_viz")
    
    if selected_variables_viz and plot_type_viz != "请选择...": 
        legend_visibility_viz = st.selectbox("图例显示:", ["自动", "隐藏"], key="viz_legend_visibility_viz")

    if plot_type_viz != "请选择..." and not selected_variables_viz:
        st.info("请至少选择一个变量进行绘图。")
        return

    if selected_variables_viz and plot_type_viz != "请选择...":
        df_plot_data_source = df_current_viz[selected_variables_viz].copy()

        if date_range_available and start_date and end_date and start_date <= end_date:
            try:
                # 确保索引是单调递增的，避免切片报错
                if not df_plot_data_source.index.is_monotonic_increasing:
                    df_plot_data_source_sorted = df_plot_data_source.sort_index()
                    df_plot_data_source = df_plot_data_source_sorted.loc[str(start_date):str(end_date)]
                else:
                    df_plot_data_source = df_plot_data_source.loc[str(start_date):str(end_date)]
                if df_plot_data_source.empty:
                    st.warning("选定时间范围内没有数据可供可视化。")
                    return # Stop if no data after slicing
            except Exception as viz_slice_e:
                st.error(f"根据日期范围筛选可视化数据时出错: {viz_slice_e}")
                return # Stop on error
        
        if df_plot_data_source.empty:
             st.warning("没有数据可用于绘制所选变量和时间范围。")
             return

        # --- Apply Y-axis scale transformation ---
        y_axis_label_suffix = ""
        if plot_type_viz not in ["堆积图", "同期对比图"]:
            if y_axis_scale_viz == "对数 (自然 e)":
                if (df_plot_data_source <= 0).any().any(): st.warning("数据含非正数，Log(e)转换前替换为NaN。")
                df_plot_data_source = df_plot_data_source.apply(lambda x: np.log(x.mask(x <= 0))); y_axis_label_suffix = " (Log e)"
            elif y_axis_scale_viz == "对数 (底为10)":
                if (df_plot_data_source <= 0).any().any(): st.warning("数据含非正数，Log10转换前替换为NaN。")
                df_plot_data_source = df_plot_data_source.apply(lambda x: np.log10(x.mask(x <= 0))); y_axis_label_suffix = " (Log10)"
        
        plot_args = {}
        if legend_visibility_viz == "隐藏": plot_args['legend'] = None 

        # Create a copy for plotting to modify column names without affecting original df_plot_data_source or debug info
        df_for_plotting = df_plot_data_source.copy()
        column_rename_map_viz = {
            col: col.replace(':', '_') # Replace colon with underscore
            for col in df_for_plotting.columns if ':' in col
        }
        if column_rename_map_viz:
            df_for_plotting.rename(columns=column_rename_map_viz, inplace=True)

        if plot_type_viz == "折线图" or plot_type_viz == "折线图 (带数据点)":
            if y_axis_label_suffix: st.caption(f"Y轴单位:{y_axis_label_suffix}")
            if plot_type_viz == "折线图 (带数据点)": st.caption("数据点标记显示依赖于数据密度和图表大小。")
            
            st.line_chart(df_for_plotting, **plot_args) # Use df_for_plotting with renamed columns
        
        elif plot_type_viz == "堆积图":
            st.area_chart(df_for_plotting, **plot_args) # Use df_for_plotting with renamed columns
        
        elif plot_type_viz == "同期对比图":
            if not isinstance(df_current_viz.index, pd.DatetimeIndex): st.warning("同期对比图需日期时间索引。") ; return
            if not freq: st.warning(f"无法推断数据频率 ({freq_display_viz})，难准确同期对比。") ; return
            
            primary_var_for_comp = selected_variables_viz[0]
            if len(selected_variables_viz) > 1: st.info(f"同期对比图推荐单变量，将用首选: {primary_var_for_comp}")
            
            source_series_for_shift = df_current_viz[primary_var_for_comp] 
            comparison_options = ["请选择...", "上一周期 (简单位移)", "去年同周期 (按频率位移)"]
            freq_str_comp_viz = str(freq).upper() # Renamed

            if freq_str_comp_viz.startswith('D') or freq_str_comp_viz.startswith('B'): comparison_options.append("月度对比 (日数据): 当月 vs 去年同月")
            elif freq_str_comp_viz.startswith('M'): comparison_options.append("季度对比 (月数据): 当季 vs 去年同季")
            
            comparison_basis = st.selectbox("选择对比基准:", comparison_options, key="viz_comparison_basis_viz")
            plot_df_comparison = None; shift_periods = 0; valid_comp_scenario = False; comp_label = ""; chart_ready = False # Renamed variables

            if comparison_basis == "上一周期 (简单位移)": shift_periods = 1; valid_comp_scenario = True; comp_label = "上一周期"
            elif comparison_basis == "去年同周期 (按频率位移)":
                if freq_str_comp_viz.startswith('A'): st.warning("年度数据此选项与'上一周期'同效。")
                elif freq_str_comp_viz.startswith('Q'): shift_periods = 4; valid_comp_scenario = True
                elif freq_str_comp_viz.startswith('M'): shift_periods = 12; valid_comp_scenario = True
                elif freq_str_comp_viz.startswith('W'): shift_periods = 52; valid_comp_scenario = True # Approx
                elif freq_str_comp_viz.startswith('D') or freq_str_comp_viz.startswith('B'): shift_periods = 365; valid_comp_scenario = True # Approx
                else: st.warning(f"未知频率({freq_display_viz})无法定去年同期周期数。")
                if valid_comp_scenario: comp_label = "去年同周期"
            
            elif comparison_basis == "月度对比 (日数据): 当月 vs 去年同月" and (freq_str_comp_viz.startswith('D') or freq_str_comp_viz.startswith('B')):
                if df_plot_data_source.empty: st.warning("无有效数据进行月度对比。"); return
                try:
                    latest_date = df_plot_data_source.index.max(); target_y = latest_date.year; target_m = latest_date.month; prev_y = target_y - 1
                    st.info(f"对比: {target_y}年{target_m}月 vs {prev_y}年{target_m}月 (基于当前数据显示范围末月)")
                    curr_m_data = source_series_for_shift[(source_series_for_shift.index.year == target_y) & (source_series_for_shift.index.month == target_m)]
                    prev_y_m_data = source_series_for_shift[(source_series_for_shift.index.year == prev_y) & (source_series_for_shift.index.month == target_m)]
                    if curr_m_data.empty or prev_y_m_data.empty: st.warning("当前月或去年同月数据不足。"); return
                    curr_m_data.index = curr_m_data.index.day; prev_y_m_data.index = prev_y_m_data.index.day
                    plot_df_comparison = pd.DataFrame({f"{target_y}-{target_m:02d}": curr_m_data, f"{prev_y}-{target_m:02d}": prev_y_m_data}); chart_ready = True
                except Exception as e_md_viz: st.error(f"月度对比(日数据)出错: {e_md_viz}")

            elif comparison_basis == "季度对比 (月数据): 当季 vs 去年同季" and freq_str_comp_viz.startswith('M'):
                if df_plot_data_source.empty: st.warning("无有效数据进行季度对比。"); return
                try:
                    latest_date = df_plot_data_source.index.max(); target_y = latest_date.year; target_q = pd.Timestamp(latest_date).quarter; prev_y = target_y - 1
                    st.info(f"对比: {target_y}年Q{target_q} vs {prev_y}年Q{target_q} (基于当前范围末季)")
                    curr_q_data = source_series_for_shift[(source_series_for_shift.index.year == target_y) & (pd.to_datetime(source_series_for_shift.index).quarter == target_q)]
                    prev_y_q_data = source_series_for_shift[(source_series_for_shift.index.year == prev_y) & (pd.to_datetime(source_series_for_shift.index).quarter == target_q)]
                    if curr_q_data.empty or prev_y_q_data.empty: st.warning("当前季或去年同季数据不足。"); return
                    month_map_q = {m: ((m-1)%3)+1 for q_idx in range(1,5) for m in range((q_idx-1)*3+1, q_idx*3+1)} # Corrected q_idx
                    curr_q_data.index = curr_q_data.index.month.map(lambda x: month_map_q.get(x,x))
                    prev_y_q_data.index = prev_y_q_data.index.month.map(lambda x: month_map_q.get(x,x))
                    plot_df_comparison = pd.DataFrame({f"{target_y}-Q{target_q}": curr_q_data, f"{prev_y}-Q{target_q}": prev_y_q_data}); chart_ready = True
                except Exception as e_qd_viz: st.error(f"季度对比(月数据)出错: {e_qd_viz}")
            
            if valid_comp_scenario and shift_periods > 0:
                current_series_comp = source_series_for_shift
                previous_series_comp = source_series_for_shift.shift(shift_periods)
                comp_data_unfiltered = pd.DataFrame({f"当前({primary_var_for_comp})": current_series_comp, f"{comp_label}({primary_var_for_comp})": previous_series_comp})
                # Apply date range filtering to the final comparison DataFrame for display
                if date_range_available and start_date and end_date and start_date <= end_date:
                    try: 
                        # 确保索引是单调递增的，避免切片报错
                        if not comp_data_unfiltered.index.is_monotonic_increasing:
                            comp_data_unfiltered_sorted = comp_data_unfiltered.sort_index()
                            plot_df_comparison = comp_data_unfiltered_sorted.loc[str(start_date):str(end_date)]
                        else:
                            plot_df_comparison = comp_data_unfiltered.loc[str(start_date):str(end_date)]
                    except Exception as comp_slice_e: st.error(f"筛选对比数据出错: {comp_slice_e}"); plot_df_comparison = pd.DataFrame()
                else: plot_df_comparison = comp_data_unfiltered
                chart_ready = True 
            
            if chart_ready:
                if plot_df_comparison is not None and not plot_df_comparison.empty: 
                    df_comp_for_plotting = plot_df_comparison.copy()
                    comp_column_rename_map = {
                        col: col.replace(':', '_')
                        for col in df_comp_for_plotting.columns if ':' in col
                    }
                    if comp_column_rename_map:
                        df_comp_for_plotting.rename(columns=comp_column_rename_map, inplace=True)
                    st.line_chart(df_comp_for_plotting) # Use df_comp_for_plotting
                elif comparison_basis != "请选择...": st.info("选定对比基准和时间范围无足够数据生成图表。")
            elif comparison_basis != "请选择..." and not valid_comp_scenario: pass # Warnings handled above 