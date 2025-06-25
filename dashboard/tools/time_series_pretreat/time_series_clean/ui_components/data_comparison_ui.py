import streamlit as st
import pandas as pd
import io

# 尝试导入后端数据处理函数
try:
    from ..utils.data_comparison_data import (
        handle_uploaded_file_for_comparison,
        add_pending_file_to_staged_data,
        reset_all_pending_uploads_comparison,
        compare_variables_in_dataset,
        compare_datasets_for_common_variables,
        update_variable_in_staged_data,
        make_staged_data_copy # <--- Add new import
    )
    BACKEND_FUNCTIONS_AVAILABLE = True
except ImportError as e:
    BACKEND_FUNCTIONS_AVAILABLE = False
    print(f"Data Comparison UI: Error importing utils.data_comparison_data: {e}")

def render_data_comparison_ui():
    """
    渲染数据比较标签页的UI界面。
    该标签页包含三个主要的功能模块。
    """

    # 初始化session_state键 (模块1: 上传至暂存区)
    st.session_state.setdefault('dc_m1_uploader_key_suffix', 0)
    st.session_state.setdefault('dc_m1_pending_uploads', {}) # {filename: {'file_obj': UploadedFile, 'df_preview': pd.DataFrame, 'time_col_details': dict, 'error': str}}

    # --- 初始化session_state键 (模块2: 变量比较) ---
    st.session_state.setdefault('dc_m2_selected_dataset_key', None)
    st.session_state.setdefault('dc_m2_selected_variables', [])
    st.session_state.setdefault('dc_m2_comparison_results', None) #  <--- 新增session_state

    # --- 初始化session_state键 (模块3: 数据集比较) ---
    st.session_state.setdefault('dc_m3_selected_datasets', [])      # For dataset comparison module
    st.session_state.setdefault('dc_m3_comparison_results', None)  # For dataset comparison module
    st.session_state.setdefault('dc_m3_update_execution_report', None)  # For update execution report

    # --- 模块一: 上传数据至暂存区 ---
    with st.container():
        st.subheader("上传数据文件（可选）")

        if not BACKEND_FUNCTIONS_AVAILABLE:
            st.error("错误：数据比较的核心后端处理模块未能成功导入。功能受限。")
            return

        # 文件上传组件
        uploader_key_dc_m1 = f"dc_m1_file_uploader_{st.session_state.get('dc_m1_uploader_key_suffix', 0)}"
        uploaded_files_dc_m1 = st.file_uploader(
            "选择或拖放一个或多个CSV或Excel文件：",
            type=['csv', 'xlsx', 'xls'],
            key=uploader_key_dc_m1,
            accept_multiple_files=True  # 允许上传多个文件
        )

        # --- Synchronize pending_uploads with file_uploader state (handles 'X' clicks) ---
        pending_uploads_dict = st.session_state.get('dc_m1_pending_uploads', {})
        active_original_filenames_in_uploader = set()
        if uploaded_files_dc_m1: # This is the list of UploadedFile objects currently in the uploader
            active_original_filenames_in_uploader = {f.name for f in uploaded_files_dc_m1}
        
        keys_in_pending_to_remove = []
        if pending_uploads_dict: # Only proceed if there's something to sync
            for key, details_in_pending in pending_uploads_dict.items():
                if details_in_pending.get('original_filename') not in active_original_filenames_in_uploader:
                    keys_in_pending_to_remove.append(key)

            if keys_in_pending_to_remove:
                for key_to_del in keys_in_pending_to_remove:
                    if key_to_del in pending_uploads_dict:
                        del pending_uploads_dict[key_to_del]
                # Streamlit should rerun due to file_uploader change, updating the UI.

        # --- Process newly uploaded files (or remaining files after 'X' clicks) ---
        if uploaded_files_dc_m1:
            batch_success_messages = []
            batch_error_messages = []

            for uploaded_file_obj in uploaded_files_dc_m1: 
                # Let handle_uploaded_file_for_comparison decide if it needs to add to pending.
                # It should internally avoid adding duplicates to dc_m1_pending_uploads and check staged_data.
                success, messages_from_handler = handle_uploaded_file_for_comparison(st.session_state, uploaded_file_obj)
                
                if success:
                    if isinstance(messages_from_handler, list):
                        batch_success_messages.extend(messages_from_handler)
                    elif messages_from_handler: # Non-empty string
                        batch_success_messages.append(messages_from_handler)
                else:
                    if isinstance(messages_from_handler, list):
                        batch_error_messages.extend(messages_from_handler)
                    elif messages_from_handler: # Non-empty string
                        batch_error_messages.append(messages_from_handler)
            
            # Display aggregated messages from this batch of uploads processing
            # Ensure unique messages if there's overlap or repeated info
            if batch_success_messages:
                unique_success_messages = list(dict.fromkeys(batch_success_messages)) # Remove duplicates while preserving order
                st.success("\n".join(unique_success_messages))
            if batch_error_messages:
                unique_error_messages = list(dict.fromkeys(batch_error_messages)) # Remove duplicates
                st.error("\n".join(unique_error_messages))
            
            # DO NOT increment uploader key or st.rerun() here.
            # This allows the pending files list below to display what was just added.

        # 显示待处理文件列表
        if st.session_state.dc_m1_pending_uploads:
            st.markdown("##### **上传数据预览**")
            
            pending_files_items = list(st.session_state.dc_m1_pending_uploads.items())
            num_files = len(pending_files_items)
            files_per_row = 2 # 你可以调整每行显示的预览数量

            for i in range(0, num_files, files_per_row):
                cols = st.columns(files_per_row)
                for j in range(files_per_row):
                    item_index = i + j
                    if item_index < num_files:
                        pending_file_key, details = pending_files_items[item_index]
                        pending_file_key_safe = pending_file_key.replace('.', '_').replace(' ', '_').replace('-', '_')
                        
                        with cols[j]: # 在当前列中渲染
                            st.markdown(f"**{pending_file_key}**") # 显示文件名
                            if details.get('error'):
                                st.error(f"加载错误: {details['error']}")
                            elif details.get('df_preview') is not None:
                                st.dataframe(details['df_preview']) # Show full DataFrame
                                st.caption(f"原始文件名: {details.get('original_filename', 'N/A')}, 工作表: {details.get('sheet_name', 'N/A') if details.get('sheet_name') else ' (单CSV文件)'}")
                                
                                if st.button(f"添加到暂存区", key=f"add_to_staged_dc_{pending_file_key_safe}", help="将此文件处理后添加到全局暂存数据中"):
                                    success_add, message_add = add_pending_file_to_staged_data(st.session_state, pending_file_key, details)
                                    if success_add:
                                        st.success(message_add)
                                        st.rerun()
                                    else:
                                        st.error(message_add)
                            else:
                                st.warning("没有可用的数据预览。")
               

    # --- 模块二: 比较暂存区数据 ---
    with st.container():
        st.markdown("--- ")
        st.subheader("数据集内变量比较")

        left_col, right_col = st.columns([1, 1]) # Or specify width ratios e.g. [2, 3]

        with left_col:
            st.markdown("##### **参数配置**")
            staged_data = st.session_state.get('staged_data', {})

            if not staged_data:
                st.info("暂存区目前没有可供比较的数据。请先在模块一上传并添加到暂存区。")
            else:
                dataset_options = list(staged_data.keys())
                current_selected_dataset_key = st.session_state.get('dc_m2_selected_dataset_key')
                if current_selected_dataset_key not in dataset_options:
                    current_selected_dataset_key = None 
                
                selected_dataset_key = st.selectbox(
                    "1. 选择要分析的数据集：",
                    options=[None] + dataset_options, 
                    format_func=lambda x: "请选择一个数据集" if x is None else x,
                    index=0 if current_selected_dataset_key is None else ([None] + dataset_options).index(current_selected_dataset_key),
                    key='dc_m2_selectbox_dataset'
                )

                if selected_dataset_key != st.session_state.get('dc_m2_selected_dataset_key'):
                    st.session_state['dc_m2_selected_dataset_key'] = selected_dataset_key
                    st.session_state['dc_m2_selected_variables'] = [] 
                    st.session_state['dc_m2_comparison_results'] = None 
                    # Consider st.rerun() if immediate feedback in right_col is needed upon dataset change
                    # For now, let button trigger comparison and result display

                if st.session_state['dc_m2_selected_dataset_key']:
                    selected_dataset_name = st.session_state['dc_m2_selected_dataset_key']
                    dataset_details = staged_data.get(selected_dataset_name)
                    
                    if dataset_details and 'df' in dataset_details:
                        df_selected = dataset_details['df']
                        available_variables = list(df_selected.columns)
                        
                        current_selected_vars = st.session_state.get('dc_m2_selected_variables', [])
                        valid_current_selected_vars = [var for var in current_selected_vars if var in available_variables]
                        
                        selected_variables = st.multiselect(
                            f"2. 从 '{selected_dataset_name}' 选择变量：",
                            options=available_variables,
                            default=valid_current_selected_vars,
                            key='dc_m2_multiselect_variables',
                            help="选择至少两个变量。第一个为基准，其余为比较对象。"
                        )
                        st.session_state['dc_m2_selected_variables'] = selected_variables

                        if len(selected_variables) >= 2:
                            base_var = selected_variables[0]
                            compare_vars = selected_variables[1:]
                            
                            if st.button(f"比较变量 (基准: {base_var})", key="dc_m2_start_comparison_button"):
                                if not BACKEND_FUNCTIONS_AVAILABLE or not hasattr(compare_variables_in_dataset, '__call__'):
                                    st.error("错误：变量比较功能所需的核心函数未能加载。")
                                    st.session_state['dc_m2_comparison_results'] = None
                                else:
                                    with st.spinner("正在比较变量，请稍候..."):
                                        comparison_output = compare_variables_in_dataset(df_selected, base_var, compare_vars)
                                    st.session_state['dc_m2_comparison_results'] = comparison_output
                                    # st.rerun() # Usually not needed as Streamlit reruns on widget interaction / state change that affects layout
                        
                        elif selected_variables:
                            st.info("请至少选择两个变量进行比较。")
                        else:
                            st.info("请选择变量以进行比较。")
                    else:
                        st.warning(f"无法加载数据集 '{selected_dataset_name}' 的详细信息或数据。")
                        st.session_state['dc_m2_comparison_results'] = None 
                elif selected_dataset_key is None and dataset_options: 
                     st.info("请先选择一个数据集以显示其可用变量。")
                     st.session_state['dc_m2_comparison_results'] = None 

        with right_col:
            st.markdown("##### **比较结果**")
            if st.session_state.get('dc_m2_selected_dataset_key') and st.session_state.get('dc_m2_selected_variables'):
                if st.session_state.get('dc_m2_comparison_results'):
                    results_data = st.session_state['dc_m2_comparison_results']
                    selected_variables_for_display = st.session_state.get('dc_m2_selected_variables', []) # Get current selected vars for display
                    base_variable_for_display = selected_variables_for_display[0] if selected_variables_for_display else "N/A"

                    if "_error_base_variable" in results_data:
                        st.error(results_data["_error_base_variable"]['message'])
                    else:
                        if not results_data: # Check if results_data itself is empty (e.g. no comparison_vars provided)
                            st.info("没有进行任何比较，或者比较变量列表为空。")
                        else:
                            for comp_var, result_detail in results_data.items():
                                status = result_detail.get('status', '未知状态')
                                message = result_detail.get('message', '无详细信息。')
                                differences_df = result_detail.get('differences_df')

                                expander_title = f"{comp_var} vs {base_variable_for_display}: {status}"
                                if status == "完全相同":
                                    expander_title = f"✅ {expander_title}"
                                elif status == "存在差异":
                                    expander_title = f"⚠️ {expander_title}"
                                elif status == "错误":
                                    expander_title = f"❌ {expander_title}"
                                else:
                                    expander_title = f"ℹ️ {expander_title}"

                                with st.expander(expander_title, expanded=True if status == "存在差异" else False):
                                    st.markdown(f"**{status}**: {message}")
                                    if status == "存在差异":
                                        if differences_df is not None and not differences_df.empty:
                                            st.dataframe(differences_df, use_container_width=True)
                                        elif differences_df is not None and differences_df.empty:
                                            st.info("差异分析表为空，但变量不完全相同（可能由于类型或特殊值差异）。")
                elif st.session_state.get('dc_m2_selected_dataset_key') and len(st.session_state.get('dc_m2_selected_variables', [])) >=2 :
                    st.info("参数已选择，请点击左侧“比较变量”按钮开始分析。")
                else:
                    st.info("请在左侧选择数据集和至少两个变量以进行比较。")
            else:
                st.info("请在左侧选择数据集和变量。")


    # --- 模块三: 比较数据集共同变量 ---
    with st.container():
        st.markdown("--- ")
        st.subheader("数据集间变量比较与更新")

        left_col_m3, right_col_m3 = st.columns([1, 1])

        with left_col_m3:
            st.markdown("##### **参数配置**")
            staged_data = st.session_state.get('staged_data', {})

            if not staged_data or len(staged_data) < 2:
                st.info("暂存区中需要至少有两个数据集才能进行比较。请先在模块一上传并添加至少两个数据集到暂存区。")
            else:
                dataset_options_m3 = list(staged_data.keys())
                
                # Ensure current selections are valid
                current_selected_datasets_m3 = st.session_state.get('dc_m3_selected_datasets', [])
                valid_current_selected_datasets_m3 = [ds for ds in current_selected_datasets_m3 if ds in dataset_options_m3]
                if len(valid_current_selected_datasets_m3) != len(current_selected_datasets_m3):
                    st.session_state['dc_m3_selected_datasets'] = valid_current_selected_datasets_m3
                    st.session_state['dc_m3_comparison_results'] = None # Reset results if selection changed due to invalidation

                selected_datasets_m3 = st.multiselect(
                    "1. 选择要比较的数据集（至少两个）：",
                    options=dataset_options_m3,
                    default=valid_current_selected_datasets_m3,
                    key='dc_m3_multiselect_datasets'
                )
                st.session_state['dc_m3_selected_datasets'] = selected_datasets_m3

                if len(selected_datasets_m3) >= 2:
                    if st.button("比较选定数据集的共同变量", key="dc_m3_start_comparison_button"):
                        # Clear any previous update report when starting a new comparison
                        st.session_state['dc_m3_update_execution_report'] = None 

                        if not BACKEND_FUNCTIONS_AVAILABLE or not hasattr(compare_datasets_for_common_variables, '__call__'):
                            st.error("错误：数据集比较功能所需的核心函数未能加载。")
                            st.session_state['dc_m3_comparison_results'] = None
                        else:
                            datasets_to_compare_data = {name: staged_data[name]['df'] for name in selected_datasets_m3 if name in staged_data and 'df' in staged_data[name]}
                            if len(datasets_to_compare_data) == len(selected_datasets_m3):
                                with st.spinner("正在比较数据集，请稍候..."):
                                    comparison_output_m3 = compare_datasets_for_common_variables(datasets_to_compare_data)
                                st.session_state['dc_m3_comparison_results'] = comparison_output_m3
                            else:
                                st.error("部分选定数据集无法加载数据，请检查暂存区。")
                                st.session_state['dc_m3_comparison_results'] = None
                    
                    # --- BEGIN: Variable Value Update Tool (Moved to left column) ---
                    results_m3_for_update_tool = st.session_state.get('dc_m3_comparison_results') # Use a distinct name if needed or reuse if context is clear
                    common_vars_list_for_update = results_m3_for_update_tool.get('common_variables', []) if results_m3_for_update_tool else []

                    if len(selected_datasets_m3) == 2 and results_m3_for_update_tool and results_m3_for_update_tool.get('status') == 'success' and common_vars_list_for_update:
                       
                        st.markdown("##### **变量值更新**")
                        staged_data_for_update = st.session_state.get('staged_data', {})
                        ds1_name_for_update, ds2_name_for_update = selected_datasets_m3[0], selected_datasets_m3[1]
                        
                        if ds1_name_for_update in staged_data_for_update and ds2_name_for_update in staged_data_for_update:
                            df1_for_update = staged_data_for_update[ds1_name_for_update].get('df')
                            df2_for_update = staged_data_for_update[ds2_name_for_update].get('df')

                            if isinstance(df1_for_update, pd.DataFrame) and isinstance(df2_for_update, pd.DataFrame):
                                st.info(f"当前比较的数据集: **{ds1_name_for_update}** 和 **{ds2_name_for_update}**")
                                update_var_key_suffix = f"left_col_{ds1_name_for_update}_{ds2_name_for_update}"
                                
                                selected_update_var = st.selectbox(
                                    "1. 选择要操作的共同变量：", 
                                    options=[None] + common_vars_list_for_update,
                                    format_func=lambda x: "请选择变量" if x is None else x,
                                    key=f'dc_m3_update_var_select_{update_var_key_suffix}'
                                )

                                if selected_update_var:
                                    if selected_update_var not in df1_for_update.columns or selected_update_var not in df2_for_update.columns:
                                        st.error(f"变量 '{selected_update_var}' 在其中一个数据集中未找到。请重新运行比较。")
                                    else:
                                        dataset_choices = [ds1_name_for_update, ds2_name_for_update]
                                        source_dataset_update = st.selectbox(
                                            "2. 选择源数据集 (提供新值)：",
                                            options=dataset_choices,
                                            key=f'dc_m3_source_ds_select_{update_var_key_suffix}'
                                        )
                                        target_dataset_update = st.selectbox(
                                            "3. 选择目标数据集 (被更新)：",
                                            options=dataset_choices,
                                            index=1 if source_dataset_update == dataset_choices[0] else 0, 
                                            key=f'dc_m3_target_ds_select_{update_var_key_suffix}'
                                        )

                                        if source_dataset_update == target_dataset_update:
                                            st.warning("源数据集和目标数据集不能相同。")
                                        else:
                                            update_mode = st.radio(
                                                "4. 选择更新模式：",
                                                options=[
                                                    ('fill_missing', f"用'{source_dataset_update}'的'{selected_update_var}'填补'{target_dataset_update}'的缺失"),
                                                    ('replace_all', f"用'{source_dataset_update}'的'{selected_update_var}'替换'{target_dataset_update}'的全部值"),
                                                    ('replace_specific_dates', f"用'{source_dataset_update}'的'{selected_update_var}'替换'{target_dataset_update}'的特定日期值")
                                                ],
                                                format_func=lambda x: x[1],
                                                key=f'dc_m3_update_mode_radio_{update_var_key_suffix}'
                                            )[0]

                                            specific_dates_to_update = []
                                            if update_mode == 'replace_specific_dates':
                                                try:
                                                    common_indices = df1_for_update.index.intersection(df2_for_update.index)
                                                    date_options_for_multiselect = sorted([str(idx.date()) if isinstance(idx, pd.Timestamp) else str(idx) for idx in common_indices])
                                                    if not date_options_for_multiselect:
                                                        st.info("源和目标数据集间无共同日期可选。")
                                                    else:
                                                        specific_dates_to_update_str = st.multiselect(
                                                            "5. 选择要更新的日期：",
                                                            options=date_options_for_multiselect,
                                                            key=f'dc_m3_specific_dates_multiselect_{update_var_key_suffix}'
                                                        )
                                                        if specific_dates_to_update_str:
                                                            specific_dates_to_update = pd.to_datetime(specific_dates_to_update_str).tolist()
                                                except Exception as e_idx:
                                                    st.error(f"获取日期选项时出错: {e_idx}")
                                            
                                            if st.button("执行更新", key=f'dc_m3_execute_update_button_{update_var_key_suffix}'):
                                                if update_mode == 'replace_specific_dates' and not specific_dates_to_update:
                                                    st.error("选择了按日期替换模式，但未选择任何日期。")
                                                else:
                                                    with st.spinner("正在更新数据..."):
                                                        success_update, msg_update, changes_df = update_variable_in_staged_data(
                                                            session_state=st.session_state, 
                                                            target_dataset_name=target_dataset_update, 
                                                            source_dataset_name=source_dataset_update, 
                                                            variable_name=selected_update_var, 
                                                            update_mode=update_mode, 
                                                            specific_dates=specific_dates_to_update
                                                        )
                                                    # Store report in session state instead of immediate display
                                                    st.session_state['dc_m3_update_execution_report'] = (success_update, msg_update, changes_df)
                                                    
                                                    # DO NOT RERUN OR CLEAR COMPARISON RESULTS HERE
                        # No 'else' needed for the df isinstance check, as it's handled by lack of UI elements.
                    # --- END: Variable Value Update Tool ---

                elif selected_datasets_m3: # i.e., len is 1
                    st.info("请至少选择两个数据集进行比较。")
                else: # No datasets selected
                    st.info("请选择数据集以进行比较。")
                    if st.session_state.get('dc_m3_comparison_results') is not None:
                         st.session_state['dc_m3_comparison_results'] = None # Clear previous results if selection becomes insufficient
        with right_col_m3:
            st.markdown("##### **比较结果**")
            results_m3 = st.session_state.get('dc_m3_comparison_results')

            if results_m3:
                status = results_m3.get('status', 'error')
                message = results_m3.get('message', '获取比较结果时出错。')
                common_vars = results_m3.get('common_variables')
                vars_per_dataset = results_m3.get('variables_per_dataset')
                compared_datasets_names = results_m3.get('compared_datasets', [])
                value_comp_results = results_m3.get('value_comparison_results') # <-- Get new results

                expander_title_m3 = f"数据集比较摘要 ({len(compared_datasets_names)}个数据集): {message}"
                icon_m3 = "✅" if status == "success" and common_vars else ("ℹ️" if status == "success" else "⚠️")
                if status == "error" or status == "no_datasets" or status == "insufficient_datasets":
                    icon_m3 = "❌"
                
                with st.expander(f"{icon_m3} {expander_title_m3}", expanded=True):
                    if status == "success":
                        if common_vars:
                            st.write("**共同变量列表：**")
                            st.code('\n'.join(common_vars), language='text')
                        else:
                            st.info("未在选定数据集中找到任何共同变量。")
                    else:
                        st.error(f"比较失败: {message}")
                
                # --- Expander for Value Comparison Results ---
                if status == "success" and common_vars and value_comp_results is not None:
                    # This expander is now OUTSIDE the one above, if you want it separate
                    # Or, if it should be inside the success case of the above, move it up.
                    # For now, keeping it separate as requested by "另起一个expander"
                    with st.expander("共同变量取值一致性检查", expanded=True):
                        comparison_summaries = value_comp_results
                        if not comparison_summaries:
                            st.info("没有共同变量可供进行取值比较或未能生成比较结果。")
                        else:
                            # Categorize variables
                            original_length_mismatches = []
                            post_dropna_length_mismatches = [] # Original lengths were same
                            value_mismatches = [] # Original and post-dropna lengths were same
                            type_mismatches = []
                            comparison_errors = []
                            identical_variables = []

                            for var_name, comp_detail in comparison_summaries.items():
                                comp_type = comp_detail.get('type')
                                comp_status = comp_detail.get('status')

                                if comp_type == 'original_length_mismatch':
                                    original_length_mismatches.append((var_name, comp_detail))
                                elif comp_type == 'type_mismatch':
                                    type_mismatches.append((var_name, comp_detail))
                                elif comp_type == 'exception_during_comparison' or comp_status == 'error': # Catch generic errors too
                                    comparison_errors.append((var_name, comp_detail))
                                elif comp_type == 'length_mismatch': # Post-dropna length mismatch
                                    post_dropna_length_mismatches.append((var_name, comp_detail))
                                elif comp_type == 'different_values':
                                    value_mismatches.append((var_name, comp_detail))
                                elif comp_type == 'identical' and comp_status == 'success':
                                    identical_variables.append((var_name, comp_detail))
                                else: # Fallback for any other unhandled cases
                                    # If comp_detail itself is None or not a dict, it would error earlier.
                                    # This handles cases where comp_type or comp_status is unexpected.
                                    unknown_detail = {'status': 'error', 'type': 'unknown_frontend_categorization', 'message': f"前端分类未知：类型='{comp_type}', 状态='{comp_status}'.原始消息: {comp_detail.get('message', 'N/A')}"}
                                    comparison_errors.append((var_name, unknown_detail))

                            # Display aggregated original index discrepancies
                            aggregated_original_discrepancies = results_m3.get('aggregated_original_index_discrepancies', [])
                            if aggregated_original_discrepancies:
                                st.write("**原始索引（日期）不一致的数据集对**")
                                for discrepancy in aggregated_original_discrepancies:
                                    ds1_name = discrepancy.get('dataset1_name', '数据集1')
                                    ds2_name = discrepancy.get('dataset2_name', '数据集2')
                                    ds1_unique_orig_str_list = discrepancy.get('dataset1_unique_indices_original', []) # Already strings YYYY-MM-DD
                                    ds2_unique_orig_str_list = discrepancy.get('dataset2_unique_indices_original', []) # Already strings YYYY-MM-DD
                                    
                                    ds1_part = f"{ds1_name}独有时间 {', '.join(ds1_unique_orig_str_list) if ds1_unique_orig_str_list else '无'}"
                                    ds2_part = f"{ds2_name}独有时间 {', '.join(ds2_unique_orig_str_list) if ds2_unique_orig_str_list else '无'}"
                                    
                                    st.info(f"{ds1_name} vs {ds2_name}：{ds1_part}；{ds2_part}")

                                st.markdown("---") # Separator after this section
                # --- End of Value Comparison Expander ---

                # --- Expander for Same Values, Different Names Analysis ---
                same_value_diff_names_results = results_m3.get('same_value_different_names_analysis')
                if status == "success" and same_value_diff_names_results is not None: # Ensure results exist
                    with st.expander("名称不同但取值相同的变量组", expanded=False): # Default to collapsed
                        if not same_value_diff_names_results: # Check if the list is empty
                            st.info("未找到名称不同但取值相同的变量组。")
                        else:
                            for idx, group in enumerate(same_value_diff_names_results):
                                st.markdown(f"**组 {idx + 1}**")
                                members = group.get('members', [])
                                preview = group.get('preview', {})
                                
                                if members:
                                    for member in members:
                                        st.markdown(f"  - 数据集: `{member['dataset_name']}`, 变量名: `{member['variable_name']}`")
                                
                                if preview:
                                    length = preview.get('length', 'N/A')
                                    dtype = preview.get('dtype', 'N/A')
                                    st.markdown(f"  - 长度: {length}, 类型: `{dtype}`")
                                st.markdown(" ") # Add a bit of space between groups
                # --- End of Same Values, Different Names Analysis Expander ---

                # --- BEGIN: Display Area for Update Execution Report (in right_col_m3) ---
                update_report = st.session_state.get('dc_m3_update_execution_report')
                if update_report:
                    st.markdown("##### **变量更新执行结果**")
                    success_update, msg_update, changes_df = update_report
                    if success_update:
                        st.success(msg_update)
                        if changes_df is not None and not changes_df.empty:
                            st.markdown("**具体变更详情：**")
                            st.dataframe(changes_df, use_container_width=True)
                        elif changes_df is not None and changes_df.empty:
                            # msg_update from backend should already cover this, but can add more info if needed
                            st.info("根据执行结果，操作已完成但未对数据值进行实际更改。") 
                    else:
                        st.error(msg_update)
                # --- END: Display Area for Update Execution Report ---

            elif st.session_state.get('dc_m3_selected_datasets') and len(st.session_state.get('dc_m3_selected_datasets',[])) >= 2:
                st.info("参数已选择，请点击左侧“比较选定数据集的共同变量”按钮开始分析。")
            else:
                st.info("请在左侧选择至少两个数据集以进行比较。")

    # --- 模块四: 数据暂存与导出 --- (New Module)
    with st.container():
        st.markdown("--- ") # Visual separator
        st.subheader("💾 数据暂存与导出")

        staged_data_keys = list(st.session_state.get('staged_data', {}).keys())

        if not staged_data_keys:
            st.info("暂存区为空。请先通过模块一上传或在上方模块操作数据。")
        else:
            col1_m4, col2_m4 = st.columns(2)

            with col1_m4:
                st.markdown("##### **1. 在暂存区创建数据集副本**")
                source_ds_for_copy = st.selectbox(
                    "选择源数据集：", 
                    options=[None] + staged_data_keys,
                    format_func=lambda x: "请选择" if x is None else x, 
                    key="m4_source_ds_copy"
                )
                new_copy_name = st.text_input("输入副本的新名称：", key="m4_new_copy_name")

                if st.button("创建副本", key="m4_create_copy_button"):
                    if not source_ds_for_copy:
                        st.error("请选择一个源数据集。")
                    elif not new_copy_name.strip():
                        st.error("副本名称不能为空。")
                    else:
                        if not BACKEND_FUNCTIONS_AVAILABLE or not hasattr(make_staged_data_copy, '__call__'):
                            st.error("错误：创建副本功能所需的核心函数未能加载。")
                        else:
                            success_copy, msg_copy = make_staged_data_copy(st.session_state, source_ds_for_copy, new_copy_name.strip())
                            if success_copy:
                                st.success(msg_copy)
                                # Update staged_data_keys for subsequent widgets if needed immediately
                                # This might require a rerun or careful state management if UI elements depend on it right after.
                                st.rerun() # Rerun to refresh selectbox options with new dataset
                            else:
                                st.error(msg_copy)
            
            with col2_m4:
                st.markdown("##### **2. 导出数据集**")
                datasets_to_export = st.multiselect(
                    "选择要导出的数据集：", 
                    options=staged_data_keys, 
                    key="m4_datasets_to_export"
                )
                export_format = st.radio("选择导出格式：", options=['CSV', 'Excel'], key="m4_export_format", horizontal=True)

                # Placeholder for advanced Excel options
                # excel_multi_sheet = False
                # if export_format == 'Excel' and datasets_to_export and len(datasets_to_export) > 1:
                #     excel_multi_sheet = st.checkbox("将多个数据集导出到同一个Excel文件的不同工作表", value=True, key="m4_excel_multi_sheet")

                if datasets_to_export:
                    if export_format == 'CSV':
                        if len(datasets_to_export) == 1:
                            ds_name_csv = datasets_to_export[0]
                            df_to_export_csv = st.session_state['staged_data'][ds_name_csv].get('df')
                            if isinstance(df_to_export_csv, pd.DataFrame):
                                csv_data = df_to_export_csv.to_csv(index=True).encode('utf-8-sig') # utf-8-sig for Excel compatibility with BOM
                                st.download_button(
                                    label=f"下载 {ds_name_csv}.csv",
                                    data=csv_data,
                                    file_name=f"{ds_name_csv}.csv",
                                    mime='text/csv',
                                    key=f"m4_download_csv_{ds_name_csv}"
                                )
                            else:
                                st.warning(f"数据集 '{ds_name_csv}' 中没有有效的数据表可导出。")
                        else: # Multiple CSVs - need to zip them
                            st.info("导出多个CSV文件将打包为ZIP。此功能待实现。") 
                            # TODO: Implement ZIP export for multiple CSVs
                    
                    elif export_format == 'Excel':
                        if len(datasets_to_export) == 1:
                            ds_name_excel = datasets_to_export[0]
                            df_to_export_excel = st.session_state['staged_data'][ds_name_excel].get('df')
                            if isinstance(df_to_export_excel, pd.DataFrame):
                                from io import BytesIO
                                excel_buffer = BytesIO()
                                with pd.ExcelWriter(excel_buffer, engine='openpyxl') as writer:
                                    df_to_export_excel.to_excel(writer, sheet_name=ds_name_excel[:31], index=True) # Sheet name limit 31 chars
                                excel_data = excel_buffer.getvalue()
                                st.download_button(
                                    label=f"下载 {ds_name_excel}.xlsx",
                                    data=excel_data,
                                    file_name=f"{ds_name_excel}.xlsx",
                                    mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
                                    key=f"m4_download_excel_{ds_name_excel}"
                                )
                            else:
                                st.warning(f"数据集 '{ds_name_excel}' 中没有有效的数据表可导出。")
                        else: # Multiple datasets for Excel
                            st.info("导出多个数据集到Excel（单文件多工作表或多文件）功能待实现。")
                            # TODO: Implement multi-sheet or multi-file Excel export
                else:
                    st.info("请选择至少一个数据集进行导出。")

if __name__ == '__main__':
    st.set_page_config(layout="wide", page_title="数据比较模块测试")
    if 'staged_data' not in st.session_state:
        st.session_state['staged_data'] = {}
    render_data_comparison_ui()
