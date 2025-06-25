import streamlit as st
import pandas as pd
import numpy as np
# No direct calculation utilities needed from calc_utils here, but pandas is primary.

def display_pivot_table_section(st, session_state):
    """处理数据透视表生成的UI和逻辑。"""

    # This section is shown only if data is loaded.
    # The check 'if session_state.ts_compute_data is not None:' is handled by the calling function in main tab.
    
    st.subheader("数据透视表生成") # This might be "3." or "4." depending on final main tab structure
    df_current_pivot = session_state.ts_compute_data

    # --- Date range context (from variable_calculations_ui or main tab) ---
    # These would ideally be passed or reliably fetched from session_state if set by another component
    # For now, assume they might exist in session_state if set by variable_calculations_ui
    date_range_available = session_state.get('ts_compute_date_range_available', False)
    start_date = session_state.get('ts_compute_start_date_vc', None) # Key from variable_calculations_ui
    end_date = session_state.get('ts_compute_end_date_vc', None)     # Key from variable_calculations_ui

    if not isinstance(df_current_pivot.index, pd.DatetimeIndex):
         st.warning("警告：数据透视表的时间分组功能需要数据具有日期时间（Datetime）索引。当前索引不是日期时间格式，仅支持按现有列分组。")
         time_grouping_possible = False
    else:
         time_grouping_possible = True

    pivot_subtype = st.selectbox("选择透视类型:",
                                 ["请选择...", "单变量透视", "多变量透视"],
                                 key="pivot_subtype_select_pvt") # Added _pvt for key uniqueness

    if pivot_subtype == "单变量透视":
        st.info("按时间频率或现有分类列对单个数值变量进行聚合统计。")
        numeric_cols_pivot_sv = df_current_pivot.select_dtypes(include=np.number).columns.tolist()
        categorical_cols_pivot_sv = df_current_pivot.select_dtypes(exclude=np.number).columns.tolist()

        if not numeric_cols_pivot_sv:
            st.warning("数据中未找到可用的数值列进行单变量透视。")
        else:
            col_pv_sv1, col_pv_sv2, col_pv_sv3 = st.columns(3)
            with col_pv_sv1:
                selected_pivot_var_sv = st.selectbox("选择数值变量 (Values):", ["请选择..."] + numeric_cols_pivot_sv, key="pivot_var_select_sv_pvt")
            with col_pv_sv2:
                 grouping_options_sv = ["请选择..."]
                 if time_grouping_possible:
                      grouping_options_sv.extend(["按年 (Yearly)", "按季 (Quarterly)", "按月 (Monthly)", "按周 (Weekly)"])
                 if categorical_cols_pivot_sv:
                      grouping_options_sv.extend([f"按列: {col}" for col in categorical_cols_pivot_sv])
                 selected_grouping_sv = st.selectbox("选择分组依据 (Index):", grouping_options_sv, key="pivot_grouping_select_sv_pvt")
            with col_pv_sv3:
                agg_funcs_sv = {
                     "总和 (Sum)": "sum", "平均值 (Mean)": "mean", "计数 (Count)": "count",
                     "最小值 (Min)": "min", "最大值 (Max)": "max", "标准差 (Std)": "std",
                     "方差 (Var)": "var", "中位数 (Median)": "median"
                 }
                selected_agg_display_sv = st.multiselect("选择统计函数 (Aggfunc):", list(agg_funcs_sv.keys()), key="pivot_agg_select_sv_pvt")
                selected_agg_sv = [agg_funcs_sv[func] for func in selected_agg_display_sv]

            if st.button("执行单变量透视", key="pivot_compute_button_sv_pvt"):
                if selected_pivot_var_sv != "请选择..." and selected_grouping_sv != "请选择..." and selected_agg_sv:
                     st.info("正在计算单变量透视表...")
                     try:
                         # Apply date range filtering if available and valid
                         df_pivot_input_sv = df_current_pivot
                         if date_range_available and start_date and end_date and start_date <= end_date:
                             try:
                                 # 确保索引是单调递增的，避免切片报错
                                 if not df_current_pivot.index.is_monotonic_increasing:
                                     df_current_pivot_sorted = df_current_pivot.sort_index()
                                     df_pivot_input_sv = df_current_pivot_sorted.loc[str(start_date):str(end_date)]
                                 else:
                                     df_pivot_input_sv = df_current_pivot.loc[str(start_date):str(end_date)]
                                 if df_pivot_input_sv.empty:
                                     st.warning("选定时间范围内没有数据可用于透视。")
                                     raise ValueError("Empty DataFrame after date slice for pivot.")
                             except Exception as e_slice_sv_pvt:
                                 st.error(f"透视表日期切片时出错: {e_slice_sv_pvt}")
                                 raise # Re-raise to stop execution
                         
                         pivot_result_sv = None
                         index_key_sv = []

                         if selected_grouping_sv.startswith("按列: "):
                             group_col_name_sv = selected_grouping_sv.replace("按列: ", "")
                             if group_col_name_sv not in df_pivot_input_sv.columns: # Check in sliced df
                                 raise ValueError(f"选择的分组列 '{group_col_name_sv}' 不存在于（可能已切片的）数据中。")
                             index_key_sv = [group_col_name_sv]
                         elif time_grouping_possible and selected_grouping_sv != "请选择...": # Ensure a time grouping is actually selected
                             freq_map_pivot = {"按年 (Yearly)": "Y", "按季 (Quarterly)": "Q", "按月 (Monthly)": "M", "按周 (Weekly)": "W"}
                             pivot_freq_sv = freq_map_pivot.get(selected_grouping_sv)
                             if pivot_freq_sv:
                                 index_key_sv = [pd.Grouper(freq=pivot_freq_sv)]
                             else: # Should not happen if UI is correct, but defensive
                                  raise ValueError("选择了无效的时间分组频率。")
                         elif not time_grouping_possible and selected_grouping_sv != "请选择...": # Categorical only, not time
                             raise ValueError("选择了时间分组，但数据索引非日期时间格式。")
                         else: # "请选择..." or other invalid state for grouping
                              raise ValueError("需要选择有效的分组依据。")

                         pivot_result_sv = pd.pivot_table(df_pivot_input_sv,
                                                         values=selected_pivot_var_sv,
                                                         index=index_key_sv,
                                                         aggfunc=selected_agg_sv,
                                                         observed=False) # Explicitly set observed

                         if pivot_result_sv is not None:
                              st.dataframe(pivot_result_sv)
                              st.success("单变量透视表计算完成。")
                         else:
                              st.warning("未能生成透视表。")

                     except Exception as e_sv_pvt:
                         st.error(f"计算单变量透视表时出错: {e_sv_pvt}")
                else:
                    st.warning("请选择数值变量、分组依据和至少一个统计函数。")

    elif pivot_subtype == "多变量透视":
        st.info("按时间频率和/或现有分类列对多个数值变量进行聚合统计。可以指定变量作为行、列和值。")
        numeric_cols_pivot_mv = df_current_pivot.select_dtypes(include=np.number).columns.tolist()
        all_potential_grouping_cols = df_current_pivot.select_dtypes(exclude=np.number).columns.tolist()
        
        time_freq_options_mv = [] # Explicitly define for later use
        if time_grouping_possible:
            time_freq_options_mv = ["按年 (Yearly)", "按季 (Quarterly)", "按月 (Monthly)", "按周 (Weekly)"]

        if df_current_pivot.index.name:
             all_potential_grouping_cols.insert(0, df_current_pivot.index.name)
        elif isinstance(df_current_pivot.index, pd.DatetimeIndex):
             all_potential_grouping_cols.insert(0, "(时间索引)")

        if not numeric_cols_pivot_mv:
            st.warning("数据中未找到可用的数值列进行多变量透视。")
        else:
            col_pv_mv_val_select, col_pv_mv_val_all, col_pv_mv_agg = st.columns([4, 1, 5])
            with col_pv_mv_val_select:
                 if 'pivot_mv_select_all_values_pvt' not in session_state:
                     session_state.pivot_mv_select_all_values_pvt = False
                 def toggle_all_pivot_values_pvt():
                     if st.session_state.pivot_mv_select_all_values_checkbox_pvt:
                         session_state.pivot_mv_vals_select_mv_key_pvt = numeric_cols_pivot_mv
                     else:
                         session_state.pivot_mv_vals_select_mv_key_pvt = []
                     session_state.pivot_mv_select_all_values_pvt = st.session_state.pivot_mv_select_all_values_checkbox_pvt
                 if 'pivot_mv_vals_select_mv_key_pvt' not in session_state:
                      session_state.pivot_mv_vals_select_mv_key_pvt = []
                 selected_pivot_vals_mv = st.multiselect("选择数值变量 (Values):", numeric_cols_pivot_mv, key="pivot_mv_vals_select_mv_key_pvt")
            with col_pv_mv_val_all:
                 st.write("&nbsp;"); st.write("&nbsp;")
                 st.checkbox("全选", key="pivot_mv_select_all_values_checkbox_pvt", value=session_state.pivot_mv_select_all_values_pvt, on_change=toggle_all_pivot_values_pvt)
            with col_pv_mv_agg:
                 agg_funcs_mv = {"总和 (Sum)": "sum", "平均值 (Mean)": "mean", "计数 (Count)": "count",
                                 "最小值 (Min)": "min", "最大值 (Max)": "max", "标准差 (Std)": "std",
                                 "方差 (Var)": "var", "中位数 (Median)": "median"}
                 selected_agg_display_mv = st.multiselect("选择统计函数 (Aggfunc):", list(agg_funcs_mv.keys()), key="pivot_agg_select_mv_pvt")
                 selected_agg_mv = [agg_funcs_mv[func] for func in selected_agg_display_mv]

            st.write("选择透视表的行分组和列分组规则:")
            col_pv_mv_idx, col_pv_mv_col_rule = st.columns(2)
            index_keys_mv = []
            map_var_col_selected = None 
            map_group_col_selected = None
            mapping_df_for_pivot = None # Use a specific variable for the loaded mapping df

            with col_pv_mv_idx:
                 st.markdown("**行分组 (Index):**")
                 index_options_mv = ["(无行分组)"]
                 if time_grouping_possible: index_options_mv.extend(time_freq_options_mv)
                 if all_potential_grouping_cols: index_options_mv.extend([f"按列: {col}" for col in all_potential_grouping_cols])
                 selected_index_display_mv = st.multiselect("选择行分组依据:", index_options_mv, key="pivot_index_select_mv_pvt")

            with col_pv_mv_col_rule:
                 st.markdown("**列分组规则 (Columns):**")
                 uploaded_mapping_file = st.file_uploader("上传变量分组规则文件 (可选, CSV/Excel)", type=["csv", "xlsx"], key="pivot_mapping_uploader_pvt")
                 st.caption("文件需含两列: 一列匹配'数值变量', 另一列为自定义分组标签。")
                 mapping_columns = []
                 if uploaded_mapping_file is not None:
                     try:
                         file_name_map = uploaded_mapping_file.name # Renamed
                         if file_name_map.endswith('.csv'): mapping_df_header = pd.read_csv(uploaded_mapping_file, nrows=0)
                         elif file_name_map.endswith('.xlsx'): mapping_df_header = pd.read_excel(uploaded_mapping_file, engine='openpyxl', nrows=0)
                         else: mapping_df_header = None
                         if mapping_df_header is not None:
                             mapping_columns = mapping_df_header.columns.tolist(); uploaded_mapping_file.seek(0)
                         else: st.warning("无法识别规则文件类型。")
                     except Exception as header_e_pvt: # Renamed
                         st.error(f"读取规则文件列名出错: {header_e_pvt}"); mapping_columns = []; 
                         try: uploaded_mapping_file.seek(0) 
                         except: pass
                     if mapping_columns:
                         col_map1_pvt, col_map2_pvt = st.columns(2) # Renamed
                         with col_map1_pvt:
                             map_var_col_selected = st.selectbox("选择'变量名'列:", options=["请选择..."] + mapping_columns, key="pivot_map_var_col_select_pvt", help="含主数据数值变量匹配名称的列")
                         with col_map2_pvt:
                             map_group_col_selected = st.selectbox("选择'分组标签'列:", options=["请选择..."] + mapping_columns, key="pivot_map_group_col_select_pvt", help="含自定义分组标签的列")
                         if map_var_col_selected == "请选择...": map_var_col_selected = None
                         if map_group_col_selected == "请选择...": map_group_col_selected = None
            
            # --- Process selections and Load Mapping File ---
            valid_pivot_params = True
            if not selected_pivot_vals_mv: st.warning("请选数值变量(Values)。"); valid_pivot_params = False
            if not selected_agg_mv: st.warning("请选统计函数(Aggfunc)。"); valid_pivot_params = False

            freq_map_pivot_mv = {"按年 (Yearly)": "Y", "按季 (Quarterly)": "Q", "按月 (Monthly)": "M", "按周 (Weekly)": "W"}
            user_friendly_time_names_mv = {"按年 (Yearly)": "年份", "按季 (Quarterly)": "季度", "按月 (Monthly)": "月份", "按周 (Weekly)": "周数"} # 新增：用户友好的时间名称映射
            desired_index_names_mv = [] # 新增：用于存储期望的索引名称

            for item in selected_index_display_mv:
                if item == "(无行分组)": continue
                if item in time_freq_options_mv: 
                    freq_item = freq_map_pivot_mv.get(item) 
                    if freq_item:
                        index_keys_mv.append(pd.Grouper(freq=freq_item))
                        desired_index_names_mv.append(user_friendly_time_names_mv.get(item, item)) 
                    else: st.warning(f"无法识别时间频率: {item}"); valid_pivot_params = False
                elif item.startswith("按列: "):
                     col_name_item = item.replace("按列: ", "") 
                     actual_col_name_for_pivot = col_name_item 
                     if col_name_item == "(时间索引)" and isinstance(df_current_pivot.index, pd.DatetimeIndex):
                         if df_current_pivot.index.name:
                             actual_col_name_for_pivot = df_current_pivot.index.name
                         pass 

                     if actual_col_name_for_pivot == df_current_pivot.index.name or (actual_col_name_for_pivot == "(时间索引)" and isinstance(df_current_pivot.index, pd.DatetimeIndex)):
                         if df_current_pivot.index.name:
                             index_keys_mv.append(df_current_pivot.index.name)
                             desired_index_names_mv.append(df_current_pivot.index.name)
                         else: 
                             index_keys_mv.append(df_current_pivot.index) 
                             desired_index_names_mv.append("(时间索引)") 
                     elif actual_col_name_for_pivot in df_current_pivot.columns:
                         index_keys_mv.append(actual_col_name_for_pivot)
                         desired_index_names_mv.append(actual_col_name_for_pivot) 
                     else: st.warning(f"行分组列不存在: {actual_col_name_for_pivot}"); valid_pivot_params = False
                else: st.warning(f"无法识别行分组: {item}"); valid_pivot_params = False
            
            effective_index_keys_mv = index_keys_mv if index_keys_mv else None
            
            if uploaded_mapping_file:
                if not map_var_col_selected or not map_group_col_selected:
                    if mapping_columns: st.warning("请选择映射文件的'变量名'和'分组标签'列。"); valid_pivot_params = False
                else:
                    try:
                        file_name_map_load = uploaded_mapping_file.name 
                        if file_name_map_load.endswith('.csv'): mapping_df_for_pivot = pd.read_csv(uploaded_mapping_file)
                        elif file_name_map_load.endswith('.xlsx'): mapping_df_for_pivot = pd.read_excel(uploaded_mapping_file, engine='openpyxl')
                        else: st.error("无法识别规则文件类型。"); valid_pivot_params = False; mapping_df_for_pivot = None
                        if mapping_df_for_pivot is not None:
                            if map_var_col_selected not in mapping_df_for_pivot.columns: 
                                st.error(f"规则文件无列: '{map_var_col_selected}'"); valid_pivot_params = False; mapping_df_for_pivot = None
                            if map_group_col_selected not in mapping_df_for_pivot.columns: 
                                st.error(f"规则文件无列: '{map_group_col_selected}'"); valid_pivot_params = False; mapping_df_for_pivot = None
                            if mapping_df_for_pivot is not None: # Re-check after potential None assignment
                                missing_vars = [v for v in selected_pivot_vals_mv if v not in mapping_df_for_pivot[map_var_col_selected].tolist()]
                                if missing_vars: st.warning(f"警告:变量 {', '.join(missing_vars)} 在规则文件'{map_var_col_selected}'列中未找到,不参与列分组。")
                    except Exception as map_e_pvt: 
                        st.error(f"读取/处理变量分组规则文件出错: {map_e_pvt}"); valid_pivot_params = False; mapping_df_for_pivot = None
            
            if st.button("执行多变量透视", key="pivot_compute_button_mv_pvt"):
                if valid_pivot_params:
                    st.info("正在计算多变量透视表...")
                    try:
                        df_pivot_input_mv = df_current_pivot 
                        if date_range_available and start_date and end_date and start_date <= end_date:
                            try:
                                # 确保索引是单调递增的，避免切片报错
                                if not df_current_pivot.index.is_monotonic_increasing:
                                    df_current_pivot_sorted = df_current_pivot.sort_index()
                                    df_pivot_input_mv = df_current_pivot_sorted.loc[str(start_date):str(end_date)]
                                else:
                                    df_pivot_input_mv = df_current_pivot.loc[str(start_date):str(end_date)]
                                if df_pivot_input_mv.empty:
                                    st.warning("选定时间范围内没有数据可用于透视。")
                                    raise ValueError("Empty DataFrame after date slice for pivot.")
                            except Exception as e_slice_mv_pvt: 
                                st.error(f"透视表日期切片时出错: {e_slice_mv_pvt}")
                                raise 
                        
                        pivot_columns_arg_final = None 
                        
                        # --- Placeholder for complex column mapping logic ---
                        if mapping_df_for_pivot is not None and map_var_col_selected and map_group_col_selected:
                            # This is where the complex logic for using the mapping file to define 'columns' would go.
                            # It might involve melting, merging, and then setting pivot_columns_arg_final.
                            # For the current fix, we are focusing on index names, so this part remains a placeholder.
                            # Example: mapped_values dictionary was correctly built in the previous step.
                            # mapped_values = {}
                            # for _, row in mapping_df_for_pivot.iterrows():
                            # mapped_values[row[map_var_col_selected]] = row[map_group_col_selected]
                            # The actual transformation and setting of pivot_columns_arg_final is non-trivial.
                            st.info("列映射功能正在处理中（此部分为占位符）。") # Placeholder message

                        pivot_result_mv = pd.pivot_table(
                            df_pivot_input_mv,
                            values=selected_pivot_vals_mv,
                            index=effective_index_keys_mv, 
                            columns=pivot_columns_arg_final, 
                            aggfunc=selected_agg_mv,
                            observed=False, 
                        )

                        if pivot_result_mv is not None:
                            # --- 先设置索引名称 ---
                            if desired_index_names_mv: 
                                if isinstance(pivot_result_mv.index, pd.MultiIndex):
                                    if len(desired_index_names_mv) == pivot_result_mv.index.nlevels:
                                        pivot_result_mv.index.names = desired_index_names_mv
                                    else:
                                        st.warning(f"警告: 期望的索引级别数 ({len(desired_index_names_mv)}) 与实际生成的级别数 ({pivot_result_mv.index.nlevels}) 不匹配。部分重命名可能不准确。")
                                        current_names = list(pivot_result_mv.index.names)
                                        for i in range(min(len(desired_index_names_mv), len(current_names))):
                                            if desired_index_names_mv[i] is not None: # 确保只用非None名称覆盖
                                                current_names[i] = desired_index_names_mv[i]
                                        pivot_result_mv.index.names = current_names
                                elif not isinstance(pivot_result_mv.index, pd.MultiIndex) and len(desired_index_names_mv) == 1 and desired_index_names_mv[0] is not None:
                                    pivot_result_mv.index.name = desired_index_names_mv[0]
                            
                            # --- 新增：格式化时间相关的索引级别 ---
                            if isinstance(pivot_result_mv.index, pd.MultiIndex):
                                new_levels = []
                                for i, level_name in enumerate(pivot_result_mv.index.names):
                                    current_level_values = pivot_result_mv.index.levels[i]
                                    if isinstance(current_level_values, pd.DatetimeIndex):
                                        if level_name == user_friendly_time_names_mv.get("按年 (Yearly)", "年份"):
                                            new_levels.append(current_level_values.year) # 获取年份数字
                                        elif level_name == user_friendly_time_names_mv.get("按月 (Monthly)", "月份"):
                                            new_levels.append(current_level_values.strftime('%m')) # 格式化为 MM
                                        elif level_name == user_friendly_time_names_mv.get("按季 (Quarterly)", "季度"):
                                            new_levels.append(current_level_values.to_period('Q').strftime('%Y-Q%q')) # 格式化为 YYYY-Qq
                                        elif level_name == user_friendly_time_names_mv.get("按周 (Weekly)", "周数"):
                                             # PeriodIndex to string for Week. Using weekofyear and year.
                                            new_levels.append(current_level_values.strftime('%Y-W%U')) # YYYY-Www (Sunday as first day)
                                        else:
                                            new_levels.append(current_level_values) # 其他时间格式保持不变
                                    else:
                                        new_levels.append(current_level_values)
                                try:
                                    pivot_result_mv.index = pd.MultiIndex.from_arrays(
                                        [pivot_result_mv.index.get_level_values(i) for i in range(pivot_result_mv.index.nlevels)], 
                                        names=pivot_result_mv.index.names
                                    )
                                    # 在重建MultiIndex后，再次检查并应用格式化 (如果需要对get_level_values的结果操作)
                                    # 实际上，上面的 new_levels 应该直接用于 .set_levels() 或在 from_arrays 中变换
                                    # 更安全的做法是直接修改 Series/DataFrame 的索引值，如果它是DatetimeIndex
                                    # 鉴于 Grouper 的工作方式，levels 本身可能已经是 Timestamp 对象

                                    # 重构格式化逻辑：
                                    formatted_levels = []
                                    for i, name in enumerate(pivot_result_mv.index.names):
                                        level_values = pivot_result_mv.index.get_level_values(i)
                                        if isinstance(level_values, pd.DatetimeIndex) or pd.api.types.is_datetime64_any_dtype(level_values.dtype):
                                            # 确保转换为 DatetimeIndex 以使用 .dt accessor
                                            dt_level_values = pd.to_datetime(level_values)
                                            if name == user_friendly_time_names_mv.get("按年 (Yearly)", "年份"):
                                                formatted_levels.append(dt_level_values.year)
                                            elif name == user_friendly_time_names_mv.get("按月 (Monthly)", "月份"):
                                                formatted_levels.append(dt_level_values.strftime('%m'))
                                            elif name == user_friendly_time_names_mv.get("按季 (Quarterly)", "季度"):
                                                formatted_levels.append(dt_level_values.to_period('Q').strftime('%Y-Q%q'))
                                            elif name == user_friendly_time_names_mv.get("按周 (Weekly)", "周数"):
                                                formatted_levels.append(dt_level_values.strftime('%Y-W%U'))
                                            else:
                                                formatted_levels.append(level_values) # 保持原样
                                        else:
                                            formatted_levels.append(level_values)
                                    pivot_result_mv.index = pd.MultiIndex.from_arrays(formatted_levels, names=pivot_result_mv.index.names)

                                except Exception as e_format_idx:
                                    st.warning(f"格式化多级索引时出现问题: {e_format_idx}")
                            
                            elif isinstance(pivot_result_mv.index, pd.DatetimeIndex):
                                index_name = pivot_result_mv.index.name
                                if index_name == user_friendly_time_names_mv.get("按年 (Yearly)", "年份"):
                                    pivot_result_mv.index = pivot_result_mv.index.year
                                elif index_name == user_friendly_time_names_mv.get("按月 (Monthly)", "月份"):
                                    pivot_result_mv.index = pivot_result_mv.index.strftime('%m')
                                elif index_name == user_friendly_time_names_mv.get("按季 (Quarterly)", "季度"):
                                    pivot_result_mv.index = pivot_result_mv.index.to_period('Q').strftime('%Y-Q%q')
                                elif index_name == user_friendly_time_names_mv.get("按周 (Weekly)", "周数"):
                                    pivot_result_mv.index = pivot_result_mv.index.strftime('%Y-W%U')
                                # 重新设置名称，因为转换可能会丢失它
                                pivot_result_mv.index.name = index_name 
                            # --- 结束新增 ---

                            st.dataframe(pivot_result_mv)
                            st.success("多变量透视表计算完成。")
                    except Exception as e_mv_pvt:
                        st.error(f"计算多变量透视表时出错: {e_mv_pvt}")
                        import traceback; st.error(traceback.format_exc())
                else: st.warning("请检查并修正透视表参数。") 