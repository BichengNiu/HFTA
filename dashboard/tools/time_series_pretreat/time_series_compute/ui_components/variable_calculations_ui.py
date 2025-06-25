import streamlit as st
import pandas as pd
import numpy as np
from ..utils import calculations as calc_utils # Relative import
import re # For frequency display
import io # Needed for file parsing
import copy # Needed for deepcopy

def display_variable_calculations_section(st, session_state):
    """处理变量计算的UI部分，包括时间范围选择、单变量和多变量处理。"""

    # This section is shown only if data is loaded
    # The check 'if session_state.ts_compute_data is not None:' is handled by the calling function in main tab.
    df = session_state.ts_compute_data # Assume df is not None here

    # --- 时间段选择器 (移除UI控件，但保留 date_range_available 的计算供内部使用) --- 
    start_date = session_state.get('ts_compute_start_date_vc') # Get from session state set by data_preview_ui
    end_date = session_state.get('ts_compute_end_date_vc') # Get from session state set by data_preview_ui
    date_range_available = isinstance(df.index, pd.DatetimeIndex)
    if date_range_available:
        # Check for NaT in index to determine true availability
        if df.index.hasnans:
            st.warning("数据索引中包含无效日期时间值 (NaT)，日期范围筛选可能不准确或导致错误。")
            # We might still allow calculation, but filtering in data_preview_ui will handle NaTs during loc
        # Ensure start/end dates from state are valid relative to each other if they exist
        if start_date and end_date and start_date > end_date:
            # This state should ideally be caught where the inputs are, but double-check here
            date_range_available = False # Treat as unavailable for calculation logic if range invalid
    
    # Keep the state variable updated, though input is moved
    session_state.ts_compute_date_range_available = date_range_available

    # --- 频率推断 --- (This was here, keeping it for context, might be displayed or used by calculations)
    freq = None
    freq_display = "无法确定频率"
    if isinstance(df.index, pd.DatetimeIndex):
        try:
            freq = pd.infer_freq(df.index)
            if freq:
                freq_map = {'A': "年度", 'AS': "年度", 'Q': "季度", 'QS': "季度", 'M': "月度", 'MS': "月度", 'W': "周度", 'D': "日度", 'B': "工作日"}
                for prefix_map_key, display_val in freq_map.items(): # Corrected iteration
                    if freq.startswith(prefix_map_key): 
                        freq_display = display_val
                        break
                else: 
                    freq_display = f"检测到频率: {freq}"
            else:
                freq_display = "不规则时间序列"
                if len(df.index) > 1:
                    time_diffs = df.index.to_series().diff().dropna()
                    if not time_diffs.empty:
                        common_diffs = time_diffs.value_counts().head(3)
                        median_diff = time_diffs.median()
                        freq_display += f". 主要间隔: {', '.join([f'{d} ({c}次)' for d, c in common_diffs.items()])}. 中位数: {median_diff}."
        except Exception as e_freq_infer: # Renamed exception variable
            freq_display = f"推断频率出错: {e_freq_infer}"
    else:
        freq_display = "索引非日期时间格式，无法推断"

    compute_type = st.selectbox("选择要执行的计算:",
                                 ["请选择...", "单变量处理", "多变量处理"],
                                 key="ts_compute_type_select_vc") # Added _vc for key uniqueness

    # --- 单变量处理 UI 和逻辑 --- #
    if compute_type == "单变量处理":
        df_current = session_state.ts_compute_data
        numeric_cols = df_current.select_dtypes(include=np.number).columns.tolist()
        if not numeric_cols:
            st.warning("数据中未找到可用的数值列来进行单变量处理。")
        else:
            st.markdown("**定义单变量处理步骤:**")
            if 'sv_operations' not in session_state: session_state.sv_operations = []
            if not session_state.sv_operations:
                session_state.sv_operations.append({'variable': None, 'transform': None, 'params': {}})

            current_ui_operations = []
            transform_map = {
                "Log (自然对数)": "Log", "Exp (指数)": "Exp", "Abs (绝对值)": "Abs",
                "Cumsum (累加)": "Cumsum", "Cumprod (累乘)": "Cumprod",
                "Cummin (累积最小)": "Cummin", "Cummax (累积最大)": "Cummax",
                "Diff (差分)": "Diff", "Moving Average (移动平均)": "Moving Average",
                "Growth Rate (环比/同比)": "Growth Rate",
                "Grouped Calculation (分组计算)": "Grouped Calculation",
                "加常数": "AddConstant", "减常数": "SubtractConstant",
                "乘常数": "MultiplyConstant", "除常数": "DivideConstant",
                "HP Filter (趋势)": "HPFilter_Trend",
                "HP Filter (周期)": "HPFilter_Cycle",
                "STL 分解 (趋势)": "STL_Trend",
                "STL 分解 (季节性)": "STL_Seasonal",
                "STL 分解 (残差)": "STL_Resid",
            }
            transform_options_display = list(transform_map.keys())

            for i, op_state in enumerate(session_state.sv_operations):
                st.markdown(f"**步骤 {i+1}:**")
                cols_op = st.columns([3, 3, 4])
                current_op_values = {'variable': None, 'transform': None, 'params': {}}

                with cols_op[0]:
                    var_options_sv = ["请选择..."] + numeric_cols
                    selected_var_index = 0
                    if op_state.get('variable') in var_options_sv:
                         selected_var_index = var_options_sv.index(op_state.get('variable'))
                    selected_var = st.selectbox("选择变量", var_options_sv,
                                                key=f"sv_op_var_{i}_vc",
                                                index=selected_var_index)
                    current_op_values['variable'] = None if selected_var == "请选择..." else selected_var

                with cols_op[1]:
                    selected_transform_display = st.selectbox("选择处理类型",
                                                              ["请选择..."] + transform_options_display,
                                                              key=f"sv_op_transform_display_{i}_vc",
                                                              index= (transform_options_display.index(op_state.get('transform_display'))+1
                                                                     if op_state.get('transform_display') in transform_options_display else 0))
                    selected_transform_internal = transform_map.get(selected_transform_display)
                    current_op_values['transform'] = selected_transform_internal
                    current_op_values['transform_display'] = selected_transform_display

                with cols_op[2]:
                    st.write("参数:")
                    params_ui = {}
                    if selected_transform_internal == "Diff":
                        params_ui['periods'] = st.number_input("差分期数", min_value=1, value=op_state.get('params', {}).get('periods', 1), step=1, key=f"sv_op_diff_periods_{i}_vc")
                    elif selected_transform_internal == "Moving Average":
                        window_options_sv = [3, 5, 7, 12, 30]
                        custom_window_label_sv = "自定义..."
                        current_window_value = op_state.get('params', {}).get('window', None)
                        predefined_selection = "请选择..."
                        if current_window_value in window_options_sv: predefined_selection = current_window_value
                        elif current_window_value is not None: predefined_selection = custom_window_label_sv
                        selected_window_option_sv = st.selectbox("移动平均窗口", ["请选择..."] + window_options_sv + [custom_window_label_sv], key=f"sv_op_ma_window_select_{i}_vc", index=(["请选择..."] + window_options_sv + [custom_window_label_sv]).index(predefined_selection))
                        window_val = None
                        if selected_window_option_sv == custom_window_label_sv:
                            window_val = st.number_input("输入自定义窗口", min_value=1, step=1, key=f"sv_op_ma_window_custom_{i}_vc", value=current_window_value if predefined_selection==custom_window_label_sv else 1)
                        elif selected_window_option_sv != "请选择...":
                            window_val = selected_window_option_sv
                        params_ui['window'] = window_val
                    elif selected_transform_internal == "Growth Rate":
                        st.info(f"(频率: {freq_display})")
                        pop_label = "环比"; yoy_label = "同比"
                        if freq:
                             if freq.startswith('A'): pop_label, yoy_label = "环比(年)", "同比(基年)"
                             elif freq.startswith('Q'): pop_label, yoy_label = "环比(季)", "同比(年)"
                             elif freq.startswith('M'): pop_label, yoy_label = "环比(月)", "同比(年)"
                             elif freq.startswith('W'): pop_label, yoy_label = "环比(周)", "同比(年)"
                             elif freq.startswith('D') or freq.startswith('B'): pop_label, yoy_label = "环比(日)", "同比(年)"
                        rate_type_disp_opts = (pop_label, yoy_label)
                        rate_type_display_sv = st.radio("增长率类型", rate_type_disp_opts,
                                                        key=f"sv_op_gr_type_radio_{i}_vc",
                                                        index=rate_type_disp_opts.index(op_state.get('params', {}).get('rate_type_display', pop_label)),
                                                        horizontal=True)
                        is_yoy_sv = (rate_type_display_sv == yoy_label)
                        rate_type_internal_sv = 'year_over_year' if is_yoy_sv else 'period_over_period'
                        params_ui['rate_type_display'] = rate_type_display_sv
                        params_ui['rate_type_internal'] = rate_type_internal_sv
                        params_ui['is_yoy'] = is_yoy_sv
                        base_year_sv = None
                        periods_gr_sv = None
                        if freq and freq.startswith('A'):
                            if is_yoy_sv:
                                available_years = sorted(df_current.index.year.unique(), reverse=True)
                                if len(available_years) > 1: base_year_sv = st.selectbox("基准年", available_years[1:], key=f"sv_op_gr_base_year_select_{i}_vc", index=available_years[1:].index(op_state.get('params',{}).get('base_year')) if op_state.get('params',{}).get('base_year') in available_years[1:] else 0 )
                                else: st.warning("数据不足两年")
                            else: periods_gr_sv = 1
                        elif freq:
                            if is_yoy_sv:
                                if freq.startswith('Q'): periods_gr_sv = 4
                                elif freq.startswith('M'): periods_gr_sv = 12
                                elif freq.startswith('W'): periods_gr_sv = 52
                                elif freq.startswith('D') or freq.startswith('B'): periods_gr_sv = 365
                            else: periods_gr_sv = 1
                            if periods_gr_sv is None: periods_gr_sv = st.number_input("手动输期数", min_value=1, step=1, key=f"sv_op_gr_periods_input_manual_{i}_vc", value=op_state.get('params',{}).get('periods',1))
                        else:
                             periods_gr_sv = st.number_input("手动输期数", min_value=1, step=1, key=f"sv_op_gr_periods_input_manual_unknown_{i}_vc", value=op_state.get('params',{}).get('periods',1))
                        params_ui['periods'] = periods_gr_sv
                        params_ui['base_year'] = base_year_sv
                    elif selected_transform_internal == "Grouped Calculation":
                        if not isinstance(df_current.index, pd.DatetimeIndex):
                             st.error("分组计算需DateTime索引")
                        else:
                             freq_map_group = {"按年": "Y", "按季": "Q", "按月": "M", "按周": "W"}
                             group_freq_display = st.selectbox("分组频率", ["请选择..."] + list(freq_map_group.keys()), key=f"sv_op_group_freq_select_{i}_vc", index=(["请选择..."]+list(freq_map_group.keys())).index(op_state.get('params',{}).get('group_freq_display',"请选择...")) )
                             group_freq_sv = freq_map_group.get(group_freq_display)
                             params_ui['group_freq_display'] = group_freq_display
                             params_ui['group_freq'] = group_freq_sv
                             calc_map_group = {"累加": "sum", "累乘": "prod", "最小": "min", "最大": "max", "差分": "diff", "移动平均": "rolling"}
                             group_calc_type_display = st.selectbox("组内计算", ["请选择..."] + list(calc_map_group.keys()), key=f"sv_op_group_calc_type_select_{i}_vc", index=(["请选择..."]+list(calc_map_group.keys())).index(op_state.get('params',{}).get('group_calc_type_display',"请选择...")))
                             group_calc_type_sv = calc_map_group.get(group_calc_type_display)
                             params_ui['group_calc_type_display'] = group_calc_type_display
                             params_ui['group_calc_type'] = group_calc_type_sv
                             if group_calc_type_sv == "diff":
                                 params_ui['group_periods'] = st.number_input("差分期数", min_value=1, value=op_state.get('params', {}).get('group_periods', 1), step=1, key=f"sv_op_group_diff_periods_{i}_vc")
                             elif group_calc_type_sv == "rolling":
                                 params_ui['group_window'] = st.number_input("移动平均窗口", min_value=1, value=op_state.get('params', {}).get('group_window', 3), step=1, key=f"sv_op_group_ma_window_{i}_vc")
                    elif selected_transform_internal in ["AddConstant", "SubtractConstant", "MultiplyConstant", "DivideConstant"]:
                        constant_value = st.number_input("输入常数值", value=op_state.get('params', {}).get('constant_value', 0.0), key=f"sv_op_constant_value_{i}_vc", format="%f")
                        params_ui['constant_value'] = constant_value
                    elif selected_transform_internal in ["HPFilter_Trend", "HPFilter_Cycle"]:
                        lamb_tooltip = "Lambda (λ) 平滑参数。常用值：年度数据 6.25, 季度数据 1600, 月度数据 129600。"
                        default_lamb = 1600
                        if freq:
                            if 'A' in freq.upper(): default_lamb = 6.25
                            elif 'Q' in freq.upper(): default_lamb = 1600
                            elif 'M' in freq.upper(): default_lamb = 129600
                        params_ui['lamb'] = st.number_input("Lambda (λ)", min_value=0.1, value=float(op_state.get('params', {}).get('lamb', default_lamb)), step=0.1, key=f"sv_op_hp_lamb_{i}_vc", help=lamb_tooltip, format="%f")
                    elif selected_transform_internal in ["STL_Trend", "STL_Seasonal", "STL_Resid"]:
                        if not isinstance(df_current.index, pd.DatetimeIndex):
                            st.error("STL 分解需要数据具有 DatetimeIndex。")
                        else:
                            st.info(f"当前数据频率: {freq_display}")
                            params_ui['period'] = st.number_input("STL 周期 (Period)", min_value=2, value=op_state.get('params', {}).get('period', 12 if freq and 'M' in freq.upper() else (4 if freq and 'Q' in freq.upper() else 2)), step=1, key=f"sv_op_stl_period_{i}_vc", help="季节性的周期，例如月度数据为12，季度数据为4。")
                            params_ui['robust'] = st.checkbox("稳健拟合 (Robust)", value=op_state.get('params', {}).get('robust', False), key=f"sv_op_stl_robust_{i}_vc")
                    else:
                        st.write("(无需额外参数)")
                    current_op_values['params'] = params_ui
                current_ui_operations.append(current_op_values)
            

            cols_buttons = st.columns(5)
            with cols_buttons[0]:
                if st.button("➕ 添加步骤", key="sv_add_op_vc"):
                    session_state.sv_operations = current_ui_operations
                    session_state.sv_operations.append({'variable': None, 'transform': None, 'params': {}})
                    st.rerun()
            with cols_buttons[1]:
                 if len(session_state.sv_operations) > 1:
                     if st.button("➖ 移除最后步骤", key="sv_remove_op_vc"):
                        session_state.sv_operations = current_ui_operations
                        session_state.sv_operations.pop()
                        st.rerun()

            if st.button("执行所有单变量步骤", key="sv_execute_all_ops_button_vc"):
                df_target = session_state.ts_compute_data
                df_calc_slice = df_target
                params_valid_exec = True 
                if date_range_available and start_date and end_date and start_date <= end_date:
                    try:
                        # 确保索引是单调递增的，避免切片报错
                        if not df_target.index.is_monotonic_increasing:
                            df_target_sorted = df_target.sort_index()
                            df_calc_slice = df_target_sorted.loc[str(start_date):str(end_date)]
                        else:
                            df_calc_slice = df_target.loc[str(start_date):str(end_date)]
                        if df_calc_slice.empty:
                            st.warning("选定时间范围内没有数据，无法执行计算。")
                            params_valid_exec = False
                    except Exception as slice_e:
                        st.error(f"根据日期范围选择数据时出错: {slice_e}")
                        params_valid_exec = False
                elif date_range_available and start_date and end_date and start_date > end_date:
                    params_valid_exec = False
                
                if not params_valid_exec:
                     st.warning("时间范围无效或切片后无数据，无法继续执行计算。")
                else:
                    all_steps_successful = True
                    for i, op_config in enumerate(current_ui_operations):
                        st.markdown(f"--- 处理步骤 {i+1} ---")
                        variable = op_config.get('variable')
                        transform_type = op_config.get('transform')
                        transform_display = op_config.get('transform_display')
                        params = op_config.get('params', {})
                        if not variable or not transform_type:
                            st.warning(f"步骤 {i+1}：请选择变量和处理类型。跳过此步骤。")
                            all_steps_successful = False; continue
                        
                        step_params_valid_inner = True 
                        kwargs_for_calc = {}
                        new_col_name = None

                        #region Parameter Validation and New Column Naming
                        if transform_type == "Diff":
                            periods = params.get('periods'); step_params_valid_inner = periods is not None and periods >= 1
                            if step_params_valid_inner: kwargs_for_calc['periods'] = periods; new_col_name = f"{variable}_{transform_type}_{periods}"
                            else: st.warning(f"步骤 {i+1} ({variable}): Diff参数无效。"); all_steps_successful = False # Mark step as failed
                        elif transform_type == "Moving Average":
                            window = params.get('window'); step_params_valid_inner = window is not None and window >= 1
                            if step_params_valid_inner: kwargs_for_calc['window'] = window; new_col_name = f"{variable}_{transform_type}_{window}"
                            else: st.warning(f"步骤 {i+1} ({variable}): MA参数无效。"); all_steps_successful = False
                        elif transform_type == "Growth Rate":
                            periods_gr = params.get('periods'); base_year = params.get('base_year'); is_yoy = params.get('is_yoy'); rate_type_internal = params.get('rate_type_internal')
                            step_params_valid_inner = ((periods_gr is not None and periods_gr >= 1) or (is_yoy and freq and freq.startswith('A') and base_year is not None))
                            if step_params_valid_inner:
                                suffix = "PoP" if not is_yoy else "YoY"
                                if is_yoy and freq and freq.startswith('A') and base_year is not None: 
                                    new_col_name = f"{variable}_YoY_vs_{base_year}"; 
                                    st.warning(f"步骤 {i+1} ({variable}): 基年同比 ({base_year}) 功能待实现，暂时跳过此特定计算。请使用期数为1的环比作为替代，或等待功能更新。")
                                    step_params_valid_inner = False # Mark this specific sub-case as not ready
                                    all_steps_successful = False
                                elif periods_gr is not None: 
                                    new_col_name = f"{variable}_{suffix}_{periods_gr}p"; 
                                    kwargs_for_calc['periods'] = periods_gr; 
                                    # kwargs_for_calc['rate_type'] = rate_type_internal # This was passed to utils.calculate_growth_rate, ensure it's still used if that function is called directly
                                else: step_params_valid_inner = False; all_steps_successful = False
                            else: st.warning(f"步骤 {i+1} ({variable}): 增长率参数无效。"); all_steps_successful = False
                        elif transform_type == "Grouped Calculation":
                            group_freq_gc = params.get('group_freq') # Renamed to avoid conflict
                            group_calc_type_gc = params.get('group_calc_type') # Renamed
                            step_params_valid_inner = group_freq_gc and group_calc_type_gc
                            calc_suffix_gc = group_calc_type_gc
                            if step_params_valid_inner:
                                if group_calc_type_gc == 'diff':
                                    group_periods_gc = params.get('group_periods')
                                    if group_periods_gc is not None and group_periods_gc >= 1: kwargs_for_calc['group_periods'] = group_periods_gc; calc_suffix_gc += f"_{group_periods_gc}"
                                    else: st.warning(f"步骤 {i+1} ({variable}): 分组差分参数无效。"); step_params_valid_inner = False; all_steps_successful = False
                                elif group_calc_type_gc == 'rolling':
                                    group_window_gc = params.get('group_window')
                                    if group_window_gc is not None and group_window_gc >= 1: kwargs_for_calc['group_window'] = group_window_gc; calc_suffix_gc += f"_MA_{group_window_gc}"
                                    else: st.warning(f"步骤 {i+1} ({variable}): 分组移动平均参数无效。"); step_params_valid_inner = False; all_steps_successful = False
                                if step_params_valid_inner: new_col_name = f"{variable}_Grp{group_freq_gc}_{calc_suffix_gc}"
                            else: st.warning(f"步骤 {i+1} ({variable}): 分组计算参数无效。"); all_steps_successful = False
                        elif transform_type in ["AddConstant", "SubtractConstant", "MultiplyConstant", "DivideConstant"]:
                            constant_value = params.get('constant_value')
                            if isinstance(constant_value, (int, float)):
                                kwargs_for_calc['constant_value'] = constant_value
                                op_name_map = {"AddConstant": "add", "SubtractConstant": "sub", "MultiplyConstant": "mul", "DivideConstant": "div"}
                                op_suffix = op_name_map.get(transform_type, "op"); constant_str = str(constant_value); 
                                new_col_name = f"{variable}_{op_suffix}_{constant_str.replace('.0','') if constant_str.endswith('.0') else constant_str}"
                                if transform_type == "DivideConstant" and constant_value == 0: st.warning(f"步骤 {i+1} ({variable}): 除数为零。"); step_params_valid_inner = False; all_steps_successful = False
                            else: st.warning(f"步骤 {i+1} ({variable}): 常数无效。"); step_params_valid_inner = False; all_steps_successful = False
                        elif transform_type in ["HPFilter_Trend", "HPFilter_Cycle"]:
                            lamb_val = params.get('lamb'); step_params_valid_inner = lamb_val is not None and lamb_val > 0
                            if step_params_valid_inner: 
                                kwargs_for_calc['lamb'] = lamb_val; 
                                hp_part = "Trend" if transform_type == "HPFilter_Trend" else "Cycle"; 
                                lamb_str = str(lamb_val); 
                                new_col_name = f"{variable}_HP_{hp_part}_{lamb_str.replace('.0','') if lamb_str.endswith('.0') else lamb_str}"
                            else: st.warning(f"步骤 {i+1} ({variable}): HP Lambda无效。"); all_steps_successful = False
                        elif transform_type in ["STL_Trend", "STL_Seasonal", "STL_Resid"]:
                            period_val = params.get('period'); robust_val = params.get('robust', False)
                            if not isinstance(df_calc_slice.index, pd.DatetimeIndex): st.error(f"步骤 {i+1} ({variable}): STL需DatetimeIndex。"); step_params_valid_inner = False; all_steps_successful = False
                            elif period_val is None or period_val < 2: st.warning(f"步骤 {i+1} ({variable}): STL Period无效。"); step_params_valid_inner = False; all_steps_successful = False
                            else: 
                                kwargs_for_calc['period'] = period_val; kwargs_for_calc['robust'] = robust_val; 
                                stl_part_map = {"STL_Trend": "Trend", "STL_Seasonal": "Seasonal", "STL_Resid": "Resid"}; 
                                stl_part_name = stl_part_map.get(transform_type, "Comp"); 
                                robust_suffix = "_Robust" if robust_val else ""; 
                                new_col_name = f"{variable}_STL_{stl_part_name}_p{period_val}{robust_suffix}"
                        else: # Simple transforms like Log, Exp, Abs, Cumsum etc.
                            new_col_name = f"{variable}_{transform_type}"
                        #endregion

                        if not step_params_valid_inner: # If params for this step were invalid, skip its calculation
                            st.warning(f"步骤 {i+1} ({variable}) 因参数无效已跳过计算。")
                            all_steps_successful = False # Mark overall success as false
                            continue # Move to the next operation in the loop
                        
                        if new_col_name is not None: # Proceed only if new_col_name was set (implies params were somewhat valid)
                            with st.spinner(f"步骤 {i+1}: 对 '{variable}' 执行 '{transform_display}'..."):
                                result_series = None; warning_msg = None
                                try:
                                    if transform_type == "Growth Rate":
                                        # Ensure periods is correctly passed for Growth Rate as it's not 'rate_type' anymore for calc_utils.calculate_growth_rate
                                        result_series = calc_utils.calculate_growth_rate(df_calc_slice.copy(), variable, kwargs_for_calc['periods'])
                                    elif transform_type == "Grouped Calculation": 
                                        st.warning(f"'{transform_display}' 的完整逻辑应在此实现或移至utils。暂跳过。") 
                                        all_steps_successful = False # Mark step as failed if not implemented
                                        continue # Skip to next operation if not implemented
                                    else:
                                        result_series, warning_msg = calc_utils.apply_single_variable_transform(df_calc_slice.copy(), variable, transform_type, **kwargs_for_calc)
                                    
                                    if warning_msg: st.warning(f"步骤 {i+1} ({variable}): {warning_msg}")
                                    
                                    if result_series is not None:
                                        if new_col_name in df_target.columns: 
                                            st.warning(f"列 '{new_col_name}' 已存在，将被覆盖。")
                                            df_target.drop(columns=[new_col_name], inplace=True)
                                        try: 
                                            df_target.insert(df_target.columns.get_loc(variable) + 1, new_col_name, result_series)
                                        except KeyError: 
                                            df_target[new_col_name] = result_series 
                                        except Exception as e_insert: 
                                            st.error(f"插入列'{new_col_name}'时出错: {e_insert}")
                                            all_steps_successful = False
                                    # else: # If result_series is None from a transform (e.g. some internal error in utils not raising an exception)
                                        # st.warning(f"步骤 {i+1} ({variable}) 未能计算出结果 (转换函数返回 None)。")
                                        # all_steps_successful = False 

                                except (ValueError, TypeError, KeyError) as specific_e: 
                                    st.error(f"步骤 {i+1} ({variable}) 计算错误: {specific_e}")
                                    all_steps_successful = False
                                except Exception as general_e: 
                                    st.error(f"步骤 {i+1} ({variable}) 执行时发生意外错误: {general_e}")
                                    all_steps_successful = False
                        # No explicit else for `if new_col_name is not None` as step_params_valid_inner already covers it.

                    if all_steps_successful: 
                        st.success("所有单变量处理步骤已尝试完成。")
                        st.rerun()
                    else: 
                        st.warning("部分步骤未能成功完成或被跳过，请检查上方信息。界面未自动刷新。")

    # --- 多变量处理 UI 和逻辑 --- #
    elif compute_type == "多变量处理":
        df_current_mv = session_state.ts_compute_data
        numeric_cols_mv = df_current_mv.select_dtypes(include=np.number).columns.tolist()
        if len(numeric_cols_mv) < 1:
            st.warning("数据中没有可用的数值列进行处理。")
        else:
            mv_subtype = st.radio("选择多变量处理方式:", ("构建加权公式", "变量组运算"), key="mv_subtype_radio_vc", horizontal=True)
            st.markdown("---")
            if mv_subtype == "构建加权公式":
                st.caption("按照顺序构建公式： (权重1*变量1) 运算2 (权重2*变量2) 运算3 ...")

                # --- 新增：模板文件上传与加载 ---
                uploaded_template_file = st.file_uploader(
                    "上传公式模板定义文件 (Excel 或 CSV):", 
                    type=["xlsx", "csv"],
                    key="formula_template_uploader_vc"
                )
                st.caption("文件格式要求：第一列为指标名称 (对应下方'变量'选择框), 第二列为权重, 后续列为类别标签 (例如 类别1, 类别2, ...)。")
                
                # 处理上传的文件
                if uploaded_template_file is not None:
                    # 使用一个不同的 session_state key 来存储文件名，避免与之前的数据上传冲突
                    if session_state.get("formula_template_filename_vc") != uploaded_template_file.name:
                        session_state.formula_template_filename_vc = uploaded_template_file.name
                        try:
                            file_content = uploaded_template_file.getvalue()
                            file_type = uploaded_template_file.name.split('.')[-1]
                            
                            # 调用解析函数 (将在下面定义)
                            parsed_groups, all_labels = parse_template_file(file_content, file_type)
                            
                            if parsed_groups is not None:
                                session_state.loaded_template_groups_vc = parsed_groups
                                session_state.available_template_labels_vc = all_labels
                                st.success(f"模板文件 '{uploaded_template_file.name}' 解析成功！找到 {len(all_labels)} 个组标签。")
                                # 重置选择框以显示新选项
                                session_state.selected_template_label_vc = "请选择要加载的组标签..."
                            else:
                                # 解析失败，错误信息应在 parse_template_file 中通过 st.error 显示
                                session_state.loaded_template_groups_vc = {}
                                session_state.available_template_labels_vc = []
                                # Clear active template if parsing fails
                                session_state.active_template_label_vc = None 

                        except Exception as e:
                            st.error(f"处理模板文件时发生错误: {e}")
                            session_state.loaded_template_groups_vc = {}
                            session_state.available_template_labels_vc = []
                            # Clear active template on error
                            session_state.active_template_label_vc = None 
                
                # 加载模板下拉框 (仅当有可用模板时显示)
                available_labels = session_state.get("available_template_labels_vc", [])
                if available_labels:
                    # Ensure the key exists for selectbox, default to None or a placeholder
                    if 'selected_template_label_vc' not in session_state:
                        session_state.selected_template_label_vc = "请选择要加载的组标签..."
                        
                    def on_loaded_template_select_change():
                        selected_label = session_state.selected_template_label_vc
                        if selected_label != "请选择要加载的组标签..." and selected_label in session_state.get("loaded_template_groups_vc", {}):
                            template_data = session_state.loaded_template_groups_vc[selected_label]
                            # Check if template_data is a dict (new format) or list (old format fallback)
                            if isinstance(template_data, dict) and 'terms' in template_data:
                                session_state.mv_formula_terms = copy.deepcopy(template_data["terms"])
                                session_state.mv_formula_new_name_vc = template_data.get("default_name", f"{selected_label}_加权") # Use stored name or generate
                            else: # Fallback for old format if needed
                                session_state.mv_formula_terms = copy.deepcopy(template_data) # Assume it's just the terms list
                                session_state.mv_formula_new_name_vc = f"{selected_label}_加权"
                                
                            session_state.ts_compute_normalize_weights_checkbox_vc = False
                            # Record the successfully loaded template label
                            session_state.active_template_label_vc = selected_label 
                        else:
                            # If user selects "请选择...", clear the active template
                            session_state.active_template_label_vc = None
                    
                    # Determine the index based on the current session state value
                    current_selection_index = 0 # Default to the first option ("请选择...")
                    if session_state.selected_template_label_vc in available_labels:
                        # Find the index in the available_labels list and add 1 because 
                        # the full options list starts with "请选择..."
                        current_selection_index = available_labels.index(session_state.selected_template_label_vc) + 1 

                    st.selectbox(
                        "加载公式模板 (基于上传的文件):",
                        options=["请选择要加载的组标签..."] + available_labels,
                        key="selected_template_label_vc",
                        index=current_selection_index, # Set index explicitly
                        on_change=on_loaded_template_select_change
                    )
                # --- 结束 新增模板文件上传与加载 ---

                if 'mv_formula_terms' not in session_state: session_state.mv_formula_terms = [{'op': '+', 'var': None, 'weight': 1.0}]
                terms_to_render = session_state.mv_formula_terms; term_ui_inputs = []
                for i, term_state in enumerate(terms_to_render):
                    cols = st.columns([1, 4, 2]); current_term_values = {}
                    with cols[0]:
                        if i == 0: st.write(" "); current_term_values['op'] = '+'
                        else: op_index = ['+', '-', '*', '/'].index(term_state.get('op', '+')); current_term_values['op'] = st.selectbox(f"运算 {i + 1}", ['+', '-', '*', '/'], key=f"mv_term_op_{i}_vc", index=op_index)
                    with cols[1]:
                        var_options = ["请选择..."] + numeric_cols_mv; var_index = 0
                        if term_state.get('var') in numeric_cols_mv: var_index = var_options.index(term_state.get('var'))
                        current_term_values['var'] = st.selectbox(f"变量 {i + 1}", var_options, key=f"mv_term_var_{i}_vc", index=var_index)
                    with cols[2]:
                        # Weight memory logic is temporarily simplified/kept, might need adjustment based on new template loading
                        selected_var_for_weight_logic = current_term_values['var'] 
                        default_weight = 1.0
                        persisted_var_in_term = term_state.get('var')
                        persisted_weight_in_term = term_state.get('weight', default_weight)
                        weight_to_display = default_weight
                        if selected_var_for_weight_logic and selected_var_for_weight_logic != "请选择...":
                            remembered_weight_for_selected_var = session_state.get('remembered_weights_vc', {}).get(selected_var_for_weight_logic)
                            if remembered_weight_for_selected_var is not None:
                                if selected_var_for_weight_logic != persisted_var_in_term or abs(float(persisted_weight_in_term) - default_weight) < 1e-9:
                                    weight_to_display = remembered_weight_for_selected_var
                                else:
                                    weight_to_display = persisted_weight_in_term
                            else:
                                weight_to_display = default_weight if selected_var_for_weight_logic != persisted_var_in_term else persisted_weight_in_term
                        else:
                            weight_to_display = persisted_weight_in_term
                        current_term_values['weight'] = st.number_input(f"权重 {i + 1}", key=f"mv_term_weight_{i}_vc", value=float(weight_to_display), format="%f")
                    term_ui_inputs.append(current_term_values)
                
                control_cols = st.columns(6)
                with control_cols[0]:
                    if st.button("➕ 添加项", key="mv_add_term_vc"):
                        # Update session state before adding the new term
                        session_state.mv_formula_terms = [term.copy() for term in term_ui_inputs] 
                        session_state.mv_formula_terms.append({'op': '+', 'var': None, 'weight': 1.0}); 
                        st.rerun()
                with control_cols[1]:
                    if len(session_state.mv_formula_terms) > 1:
                        if st.button("➖ 移除末项", key="mv_remove_term_vc"):
                            # Update session state before removing the last term
                            session_state.mv_formula_terms = [term.copy() for term in term_ui_inputs] 
                            session_state.mv_formula_terms.pop(); 
                            st.rerun()
                with control_cols[2]:
                    if st.button("🔄 重置公式", key="mv_reset_formula_vc"):
                        active_label = session_state.get("active_template_label_vc")
                        loaded_groups = session_state.get("loaded_template_groups_vc", {})
                        
                        if active_label and active_label in loaded_groups:
                            # Reset to the active template state
                            template_data = loaded_groups[active_label]
                            if isinstance(template_data, dict) and 'terms' in template_data:
                                session_state.mv_formula_terms = copy.deepcopy(template_data["terms"])
                                session_state.mv_formula_new_name_vc = template_data.get("default_name", f"{active_label}_加权")
                            else: # Fallback
                                session_state.mv_formula_terms = copy.deepcopy(template_data)
                                session_state.mv_formula_new_name_vc = f"{active_label}_加权"
                            st.info(f"已重置为模板 '{active_label}' 的设定。")
                        else:
                            # Reset to the default empty state
                            session_state.mv_formula_terms = [{'op': '+', 'var': None, 'weight': 1.0}]
                            session_state.mv_formula_new_name_vc = ""
                            st.info("已重置为初始空白状态。")
                            
                        if 'ts_compute_normalize_weights_checkbox_vc' in session_state:
                            session_state.ts_compute_normalize_weights_checkbox_vc = False
                        # Removed trigger_template_select_reset_vc logic
                        st.rerun()
                
                st.markdown("---")

                # --- Weight check and normalization logic remains --- 
                # (Assuming term_ui_inputs is up-to-date when this part renders)
                current_weights_for_display = [term.get('weight', 1.0) for term in term_ui_inputs if isinstance(term.get('weight'), (int, float))]
                current_weight_sum_for_display = sum(current_weights_for_display) if current_weights_for_display else 0
                if 'ts_compute_normalize_weights_checkbox_vc' not in session_state:
                    session_state.ts_compute_normalize_weights_checkbox_vc = False
                normalize_weights_checkbox = st.checkbox(
                    "将权重和自动标准化为1进行计算", 
                    key="ts_compute_normalize_weights_checkbox_vc"
                )
                if abs(current_weight_sum_for_display - 1.0) < 1e-9 or abs(current_weight_sum_for_display - 100.0) < 1e-9:
                    st.success(f"当前权重之和为 {current_weight_sum_for_display:.4f}，将按此权重计算。")
                else:
                    st.warning(f"当前权重之和为 {current_weight_sum_for_display:.4f} (非1或100)。如需标准化为1，请勾选上方选项。否则将按当前输入权重计算。")
                
                new_col_name_formula = st.text_input("输入新变量名:", key="mv_formula_new_name_vc")
                if st.button("执行公式计算", key="mv_formula_compute_button_vc"):
                    # Initialize the flag *before* any checks that might assign it
                    calculation_valid_mv = True
                    
                    # --- Weight memory logic remains --- 
                    if 'remembered_weights_vc' not in session_state:
                        session_state.remembered_weights_vc = {}
                    for term_data in term_ui_inputs: 
                        var_name = term_data.get('var')
                        weight_val = term_data.get('weight')
                        if var_name and var_name != "请选择..." and isinstance(weight_val, (int, float)):
                            session_state.remembered_weights_vc[var_name] = weight_val
                    
                    # --- Normalization and calculation logic remains --- 
                    # (It operates on term_ui_inputs which reflect the current UI state) 
                    weights_for_calc = [term.get('weight', 1.0) for term in term_ui_inputs if isinstance(term.get('weight'), (int, float))]
                    actual_weight_sum_on_compute = sum(weights_for_calc) if weights_for_calc else 0
                    final_terms_for_calc = []
                    if session_state.ts_compute_normalize_weights_checkbox_vc:
                        st.info("将权重标准化为和等于1后进行计算。")
                        # --- New Debug Print ---
                        st.write("DEBUG (Before Norm Loop): len(term_ui_inputs):", len(term_ui_inputs)) 
                        # --- End New Debug Print ---
                        if actual_weight_sum_on_compute != 0 and not (abs(actual_weight_sum_on_compute - 1.0) < 1e-9 or abs(actual_weight_sum_on_compute - 100.0) < 1e-9) : # 仅在非0且非标准时标准化
                            for term_input_value in term_ui_inputs:
                                term_input_value['weight'] = term_input_value['weight'] / actual_weight_sum_on_compute
                            final_terms_for_calc = [t.copy() for t in term_ui_inputs]
                            st.info("按标准化后的权重进行计算。")
                        else:
                            final_terms_for_calc = [t.copy() for t in term_ui_inputs]
                            st.info("按用户输入的原始权重进行计算。")
                    else:
                        final_terms_for_calc = [t.copy() for t in term_ui_inputs]
                        st.info("按用户输入的原始权重进行计算。")

                    # --- Execute calculation only if name is provided and data slicing was valid ---
                    if not new_col_name_formula:
                        st.warning("请输入新变量的名称。")
                        # Set calculation_valid_mv to False if name is missing
                        calculation_valid_mv = False 
                    else:
                        # Name is provided, proceed with slicing checks
                        df_target_mv = session_state.ts_compute_data
                        df_calc_slice_mv = df_target_mv # Start with the full data
                        
                        # Apply date slicing if necessary
                        if date_range_available and start_date and end_date and start_date <= end_date:
                            try: 
                                # 确保索引是单调递增的，避免切片报错
                                if not df_target_mv.index.is_monotonic_increasing:
                                    df_target_mv_sorted = df_target_mv.sort_index()
                                    df_calc_slice_mv = df_target_mv_sorted.loc[str(start_date):str(end_date)]
                                else:
                                    df_calc_slice_mv = df_target_mv.loc[str(start_date):str(end_date)]
                                if df_calc_slice_mv.empty: 
                                    st.warning("选定时间范围内没有数据，无法计算。")
                                    calculation_valid_mv = False
                            except Exception as slice_e_mv: 
                                st.error(f"根据日期范围筛选数据时出错: {slice_e_mv}")
                                calculation_valid_mv = False
                        elif date_range_available and start_date and end_date and start_date > end_date: 
                            calculation_valid_mv = False 
                        
                        if df_calc_slice_mv.empty and calculation_valid_mv:
                             st.warning("用于计算的数据为空（可能由于日期筛选）。")
                             calculation_valid_mv = False

                    # --- Execute calculation only if still valid ---
                    if calculation_valid_mv: # Check if still valid after name and slicing checks
                        result_series_mv = None 
                        if not final_terms_for_calc: 
                            st.warning("没有有效的公式项用于计算。")
                            calculation_valid_mv = False
                        else:
                            # --- Process First Term --- 
                            first_term_values = final_terms_for_calc[0]
                            first_var = first_term_values.get('var')
                            first_weight = first_term_values.get('weight')

                            if not first_var or first_var == "请选择...": 
                                st.warning("公式第一项未选择变量。")
                                calculation_valid_mv = False
                            elif not isinstance(first_weight, (int, float)):
                                st.warning("公式第一项的权重不是有效数字。")
                                calculation_valid_mv = False
                            elif first_var not in df_calc_slice_mv.columns:
                                st.error(f"找不到变量 '{first_var}'。请检查变量名是否在当前数据中存在，或是否已被移除/重命名。")
                                calculation_valid_mv = False
                            else:
                                # Correctly indented try/except block
                                try: 
                                     result_series_mv = df_calc_slice_mv[first_var].astype(float) * float(first_weight)
                                except Exception as e_first_term: 
                                     st.error(f"计算公式第一项 ('{first_var}' * {first_weight}) 时出错: {e_first_term}")
                                     calculation_valid_mv = False
                        
                            # --- Process Subsequent Terms --- 
                            # Indentation corrected for the entire block below
                            if calculation_valid_mv and result_series_mv is not None: 
                                for i in range(1, len(final_terms_for_calc)):
                                    term_values = final_terms_for_calc[i]
                                    op = term_values.get('op')
                                    var = term_values.get('var')
                                    weight = term_values.get('weight')

                                    if not var or var == "请选择...": 
                                        st.warning(f"公式第 {i+1} 项未选择变量。")
                                        calculation_valid_mv = False; break
                                    if not isinstance(weight, (int, float)):
                                        st.warning(f"公式第 {i+1} 项的权重不是有效数字。")
                                        calculation_valid_mv = False; break
                                    if var not in df_calc_slice_mv.columns:
                                        st.error(f"找不到变量 '{var}' (在公式第 {i+1} 项)。请检查变量名。")
                                        calculation_valid_mv = False; break
                                        
                                    try: 
                                        current_term_series = df_calc_slice_mv[var].astype(float) * float(weight)
                                        if op == '+': result_series_mv += current_term_series
                                        elif op == '-': result_series_mv -= current_term_series
                                        elif op == '*': result_series_mv *= current_term_series
                                        elif op == '/':
                                            divisor = current_term_series.replace(0, np.nan)
                                            result_series_mv /= divisor
                                            if (current_term_series == 0).any():
                                                st.warning(f"公式第 {i+1} 项 ('{var}') 包含零值，除法结果中对应位置将为 NaN。")
                                        else:
                                            # Correctly indented warning
                                            st.warning(f"公式第 {i+1} 项使用了未知的运算符 '{op}'，已跳过此项。")
                                            
                                    except Exception as e_term: 
                                        st.error(f"计算公式第 {i+1} 项 ('{op}' '{var}' * {weight}) 时出错: {e_term}")
                                        calculation_valid_mv = False; break
                            
                            # --- Add Result to DataFrame --- 
                            # Indentation corrected for the entire block below
                            if calculation_valid_mv and result_series_mv is not None:
                                try: 
                                    df_target_mv = session_state.ts_compute_data # Re-fetch the target df reference
                                    if new_col_name_formula in df_target_mv.columns: 
                                        st.warning(f"列 '{new_col_name_formula}' 已存在，将被覆盖。")
                                    df_target_mv[new_col_name_formula] = result_series_mv 
                                    session_state.ts_compute_data = df_target_mv 
                                    st.success(f"公式计算完成！已添加/更新列 '{new_col_name_formula}'。");
                                    st.rerun() 
                                except Exception as e_add_res: 
                                    st.error(f"添加最终计算结果到数据时出错: {e_add_res}")
                            elif calculation_valid_mv and result_series_mv is None:
                                st.warning("未能计算出有效的结果序列。")
                    # --- End of moved calculation logic ---
                    # else corresponding to 'if calculation_valid_mv:' is implicitly pass

            elif mv_subtype == "变量组运算":
                st.caption("选择两组变量和一个操作符进行计算: (Sum(Group 1)) Operator (Sum(Group 2))") # This caption might need review for (Sum(...)) part for pair-wise
                col_group1, col_op, col_group2 = st.columns([5, 1, 5])

                # Ensure session state keys for multiselects are initialized
                if "mv_group1_select_vc" not in st.session_state:
                    st.session_state.mv_group1_select_vc = []
                if "mv_group2_select_vc" not in st.session_state:
                    st.session_state.mv_group2_select_vc = []

                with col_group1:
                    options_for_group1_widget = [
                        col for col in numeric_cols_mv 
                        if col not in st.session_state.mv_group2_select_vc # Use direct access
                    ]
                    selected_group1 = st.multiselect("选择左侧变量组 (Group 1)", options_for_group1_widget, key="mv_group1_select_vc")
                with col_op:
                    selected_op = st.selectbox("运算", ['+', '-', '*', '/'], key="mv_group_op_select_vc", label_visibility="collapsed")
                with col_group2:
                    options_for_group2_widget = [
                        col for col in numeric_cols_mv 
                        if col not in st.session_state.mv_group1_select_vc # Use direct access
                    ]
                    selected_group2 = st.multiselect("选择右侧变量组 (Group 2)", options_for_group2_widget, key="mv_group2_select_vc")
                st.markdown("---")
                if st.button("执行变量组运算", key="mv_group_op_compute_button_vc"):
                    if not selected_group1 or not selected_group2:
                        st.warning("请为左右两边变量组至少各选择一个变量。")
                    else:
                        op_name_map = {'+': 'plus', '-': 'minus', '*': 'times', '/': 'divby'}; selected_op_name = op_name_map.get(selected_op, selected_op)
                        df_target_mv_group = session_state.ts_compute_data; calculation_valid_group_op = True # Renamed
                        df_calc_slice_mv_group = df_target_mv_group
                        if date_range_available and start_date and end_date and start_date <= end_date:
                            try:
                                # 确保索引是单调递增的，避免切片报错
                                if not df_target_mv_group.index.is_monotonic_increasing:
                                    df_target_mv_group_sorted = df_target_mv_group.sort_index()
                                    df_calc_slice_mv_group = df_target_mv_group_sorted.loc[str(start_date):str(end_date)]
                                else:
                                    df_calc_slice_mv_group = df_target_mv_group.loc[str(start_date):str(end_date)]
                                if df_calc_slice_mv_group.empty: 
                                    st.warning("选定时间范围内无数据。")
                                    calculation_valid_group_op = False
                            except Exception as slice_e_group: 
                                st.error(f"日期范围切片错误: {slice_e_group}")
                                calculation_valid_group_op = False
                        elif date_range_available and start_date and end_date and start_date > end_date: 
                            calculation_valid_group_op = False
                        if df_calc_slice_mv_group.empty and calculation_valid_group_op: 
                            st.warning("计算数据为空。")
                            calculation_valid_group_op = False
                        if calculation_valid_group_op:
                            results_to_add = {}; all_ops_successful_group_op = True # Renamed
                            try:
                                g1_len = len(selected_group1); g2_len = len(selected_group2)
                                for col_list in [selected_group1, selected_group2]:
                                    for col_name in col_list:
                                        if df_calc_slice_mv_group[col_name].isnull().any(): st.warning(f"列 '{col_name}' 含NaN。"); break
                                with st.spinner(f"正在执行变量组运算..."):
                                    if g1_len > 1 and g2_len > 1:
                                        if g1_len != g2_len: st.error("两边多变量时，数量需相同以配对。"); all_ops_successful_group_op = False
                                        else:
                                            for i in range(g1_len):
                                                var1_name = selected_group1[i]; var2_name = selected_group2[i]; series1 = df_calc_slice_mv_group[var1_name]; series2 = df_calc_slice_mv_group[var2_name]; new_col_name = f"{var1_name}_{selected_op_name}_{var2_name}"
                                                if selected_op == '+': result_series = series1 + series2
                                                elif selected_op == '-': result_series = series1 - series2
                                                elif selected_op == '*': result_series = series1 * series2
                                                elif selected_op == '/':
                                                    if (series2 == 0).any(): st.warning(f"计算'{new_col_name}'时除数'{var2_name}'含零，结果NaN。"); divisor = series2.replace(0, np.nan); result_series = series1 / divisor
                                                    else: result_series = series1 / series2
                                                results_to_add[new_col_name] = result_series
                                    elif g1_len > 0 and g2_len > 0:
                                        iter_group, single_var_name_list = (selected_group1, selected_group2) if g1_len >= g2_len else (selected_group2, selected_group1)
                                        is_g1_iter = (g1_len >= g2_len); single_var_name = single_var_name_list[0]
                                        if g1_len == 1 and g2_len == 1: iter_group = selected_group1; single_var_name = selected_group2[0]; is_g1_iter = True
                                        for var_in_iter_group_name in iter_group:
                                            series_iter = df_calc_slice_mv_group[var_in_iter_group_name]; series_single = df_calc_slice_mv_group[single_var_name]
                                            if is_g1_iter: var1_name = var_in_iter_group_name; var2_name = single_var_name; series1, series2 = series_iter, series_single
                                            else: var1_name = single_var_name; var2_name = var_in_iter_group_name; series1, series2 = series_single, series_iter
                                            new_col_name = f"{var1_name}_{selected_op_name}_{var2_name}"
                                            if selected_op == '+': result_series = series1 + series2
                                            elif selected_op == '-': result_series = series1 - series2
                                            elif selected_op == '*': result_series = series1 * series2
                                            elif selected_op == '/':
                                                if (series2 == 0).any(): st.warning(f"计算'{new_col_name}'时除数含零，结果NaN。"); divisor = series2.replace(0, np.nan); result_series = series1 / divisor
                                                else: result_series = series1 / series2
                                            results_to_add[new_col_name] = result_series
                                    else: st.error("选择的变量组无效。"); all_ops_successful_group_op = False
                                    if all_ops_successful_group_op and results_to_add:
                                        num_cols_added = 0
                                        for col_name, series_data in results_to_add.items():
                                            if col_name in df_target_mv_group.columns: st.warning(f"列 '{col_name}' 已存在，将被覆盖。"); df_target_mv_group.drop(columns=[col_name], inplace=True)
                                            df_target_mv_group[col_name] = series_data; num_cols_added +=1
                                        st.success(f"变量组运算完成！已添加/更新 {num_cols_added} 列。"); st.rerun()
                                    elif not results_to_add and all_ops_successful_group_op: st.info("变量组运算未产生新列。")
                            except KeyError as ke_group: st.error(f"变量组运算KeyError: {ke_group}。"); import traceback; st.error(traceback.format_exc())
                            except Exception as group_op_e_outer: st.error(f"变量组运算意外错误: {group_op_e_outer}"); import traceback; st.error(traceback.format_exc())                                

def parse_template_file(file_content, file_type):
    """解析上传的模板文件 (Excel 或 CSV)。"""
    try:
        df = None
        if file_type == 'csv':
            df = pd.read_csv(io.BytesIO(file_content))
        elif file_type == 'xlsx':
            df = pd.read_excel(io.BytesIO(file_content), engine='openpyxl')
        else:
            st.error("不支持的文件类型。")
            return None, []

        if df is None or df.empty:
            st.warning("模板文件为空或无法读取。")
            return None, []

        # --- 假设列名或固定位置 --- 
        # 方案 A: 依赖列名 (更健壮)
        indicator_col = '指标'
        weight_col = '权重'
        category_cols = [col for col in df.columns if col.startswith('类别')] # 查找所有以'类别'开头的列
        
        required_cols = [indicator_col, weight_col]
        if not all(col in df.columns for col in required_cols):
            st.error(f"模板文件缺少必需的列：'{indicator_col}' 或 '{weight_col}'。")
            return None, []
        if not category_cols:
            st.error("模板文件必须至少包含一个以 '类别' 开头的列用于定义组标签。")
            return None, []
            
        # --- (如果需要按列位置，则修改这里) ---
        # 方案 B: 依赖列位置 (如果用户文件无表头或表头不固定)
        # if df.shape[1] < 3:
        #     st.error("模板文件至少需要3列：指标、权重、类别1。")
        #     return None, []
        # indicator_col_idx = 0
        # weight_col_idx = 1
        # category_col_indices = list(range(2, df.shape[1])) # 从第3列开始都是类别列
        # df.columns = [f'col_{i}' for i in range(df.shape[1])] # 强制重命名列以便索引
        # indicator_col = 'col_0'
        # weight_col = 'col_1'
        # category_cols = [f'col_{i}' for i in category_col_indices]
        # --- 结束方案 B ---

        all_labels = set()
        parsed_groups = {}

        # 1. 提取所有唯一的组标签
        for cat_col in category_cols:
            # 清理标签中的前后空格，处理NaN
            unique_in_col = df[cat_col].dropna().astype(str).str.strip().unique()
            all_labels.update(u for u in unique_in_col if u) # 添加非空标签
            
        sorted_labels = sorted(list(all_labels))

        if not sorted_labels:
            st.warning("在类别列中未能找到任何有效的组标签。")
            return None, []

        # 2. 为每个标签构建公式项
        for label in sorted_labels:
            formula_terms = []
            # 查找包含此标签的所有行 (在任何类别列中)
            mask = pd.Series(False, index=df.index)
            for cat_col in category_cols:
                mask = mask | (df[cat_col].astype(str).str.strip() == label)
            
            relevant_rows = df[mask]
            
            for _, row in relevant_rows.iterrows():
                indicator = row[indicator_col]
                weight = row[weight_col]
                
                # 数据校验
                if pd.isna(indicator) or not str(indicator).strip():
                    st.warning(f"标签 '{label}' 下找到空的指标名称，已跳过该行。")
                    continue
                if pd.isna(weight):
                    st.warning(f"标签 '{label}' 下指标 '{indicator}' 的权重为空，已跳过该行。")
                    continue
                try:
                    weight = float(weight)
                except (ValueError, TypeError):
                    st.warning(f"标签 '{label}' 下指标 '{indicator}' 的权重 '{weight}' 不是有效数字，已跳过该行。")
                    continue
                    
                formula_terms.append({
                    'op': '+', # 默认为加法
                    'var': str(indicator).strip(), 
                    'weight': weight
                })
            
            if formula_terms: # 只有当这个标签确实关联了有效的公式项时才添加到结果中
                # Store terms and default name
                parsed_groups[label] = {
                    "terms": formula_terms,
                    "default_name": f"{label}_加权" # Generate default name here
                }
            else:
                st.warning(f"标签 '{label}' 没有找到任何有效的指标和权重组合。")
                # 从最终的标签列表中移除无效的标签
                # Ensure removal works correctly if iterating while modifying
                # It's safer to build a new list or filter afterwards if removal is needed.
                # For now, let's keep it in parsed_groups but maybe not in sorted_labels?
                # Let's adjust the return logic slightly:
                # return parsed_groups, [lbl for lbl in sorted_labels if lbl in parsed_groups] # Return only labels with valid terms

        # Return only labels that have valid formula terms associated
        final_labels = [lbl for lbl in sorted_labels if lbl in parsed_groups]
        if not final_labels:
            st.warning("未能从文件中解析出任何包含有效公式项的组标签。")
            return None, []
            
        return parsed_groups, final_labels

    except Exception as e:
        st.error(f"解析模板文件时出错: {e}")
        import traceback
        st.error(traceback.format_exc()) # 打印详细错误供调试
        return None, []
                                