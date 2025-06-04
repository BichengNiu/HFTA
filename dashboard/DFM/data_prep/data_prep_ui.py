import streamlit as st
from datetime import datetime

# 导入配置
try:
    from config import (
        DataDefaults, TrainDefaults, UIDefaults, VisualizationDefaults,
        FileDefaults, FormatDefaults, AnalysisDefaults
    )
    CONFIG_AVAILABLE = True
except ImportError as e:
    print(f"⚠️ 无法导入配置模块: {e}")
    CONFIG_AVAILABLE = False

def render_dfm_data_prep_tab(st, session_state):
    """Renders the DFM Model Data Preparation tab."""
    st.markdown("#### 上传数据")

    # 初始化 session_state 中的文件存储
    if 'dfm_training_data_file' not in session_state:
        session_state.dfm_training_data_file = None
    
    # --- NEW: Initialize session_state for direct data passing ---
    if 'dfm_prepared_data_df' not in session_state:
        session_state.dfm_prepared_data_df = None
    if 'dfm_transform_log_obj' not in session_state:
        session_state.dfm_transform_log_obj = None
    if 'dfm_industry_map_obj' not in session_state:
        session_state.dfm_industry_map_obj = None
    if 'dfm_removed_vars_log_obj' not in session_state:
        session_state.dfm_removed_vars_log_obj = None
    if 'dfm_var_type_map_obj' not in session_state:
        session_state.dfm_var_type_map_obj = None
    # --- END NEW ---

    uploaded_file = st.file_uploader(
        "选择训练数据集 (例如：.csv, .xlsx)", 
        type=["csv", "xlsx"], 
        key="dfm_training_data_uploader",
        help="请上传包含模型训练所需指标的表格数据。"
    )

    if uploaded_file is not None:
        session_state.dfm_training_data_file = uploaded_file
        
        # 自动检测数据的日期范围并设置默认的结束日期
        try:
            import io
            import pandas as pd
            
            # 读取上传的文件以获取日期范围
            uploaded_file_bytes = uploaded_file.getvalue()
            excel_file_like_object = io.BytesIO(uploaded_file_bytes)
            
            # 尝试读取第一个sheet来获取日期信息
            if uploaded_file.name.endswith('.xlsx'):
                # 获取所有sheet名称
                try:
                    xl_file = pd.ExcelFile(excel_file_like_object)
                    sheet_names = xl_file.sheet_names
                    
                    # 尝试从不同的sheet中找到日期列
                    date_info = None
                    for sheet_name in sheet_names:
                        try:
                            df_sample = pd.read_excel(excel_file_like_object, sheet_name=sheet_name, nrows=5)
                            # 寻找可能的日期列
                            date_cols = []
                            for col in df_sample.columns:
                                if 'date' in str(col).lower() or 'time' in str(col).lower() or '日期' in str(col):
                                    date_cols.append(col)
                                elif len(df_sample) > 0:
                                    # 尝试解析第一列是否为日期
                                    try:
                                        if col == df_sample.columns[0]:  # 通常第一列是日期
                                            sample_val = df_sample[col].iloc[0]
                                            if pd.notna(sample_val):
                                                pd.to_datetime(sample_val)
                                                date_cols.append(col)
                                    except:
                                        pass
                            
                            if date_cols:
                                # 读取完整的数据以获取日期范围
                                df_full = pd.read_excel(excel_file_like_object, sheet_name=sheet_name)
                                for date_col in date_cols:
                                    try:
                                        date_series = pd.to_datetime(df_full[date_col], errors='coerce')
                                        date_series_clean = date_series.dropna()
                                        if len(date_series_clean) > 0:
                                            min_date = date_series_clean.min().date()
                                            max_date = date_series_clean.max().date()
                                            date_info = (min_date, max_date, sheet_name, date_col)
                                            break
                                    except:
                                        continue
                                if date_info:
                                    break
                        except:
                            continue
                    
                    # 如果找到了日期信息，更新默认值
                    if date_info:
                        min_date, max_date, found_sheet, found_col = date_info
                        # 设置数据结束日期为数据的最后一个观测值日期
                        session_state.dfm_param_data_end_date = max_date
                        # 如果开始日期还没设置，也可以设置一个合理的开始日期
                        if session_state.dfm_param_data_start_date == datetime(2020, 1, 1).date():
                            session_state.dfm_param_data_start_date = min_date
                        
                        st.success(f"文件 '{uploaded_file.name}' 已上传并准备就绪。")
                        st.info(f"📅 检测到数据日期范围：{min_date} 至 {max_date} (来源：{found_sheet}表的{found_col}列)")
                    else:
                        st.success(f"文件 '{uploaded_file.name}' 已上传并准备就绪。")
                        st.warning("未能自动检测数据日期范围，请手动设置日期参数。")
                        
                except Exception as e:
                    st.success(f"文件 '{uploaded_file.name}' 已上传并准备就绪。")
                    st.warning(f"自动检测日期范围时出错：{e}")
            else:
                # CSV文件处理
                try:
                    df_csv = pd.read_csv(excel_file_like_object, nrows=5)
                    # 类似的日期检测逻辑...
                    st.success(f"文件 '{uploaded_file.name}' 已上传并准备就绪。")
                except:
                    st.success(f"文件 '{uploaded_file.name}' 已上传并准备就绪。")
                    
        except Exception as e:
            st.success(f"文件 '{uploaded_file.name}' 已上传并准备就绪。")
            st.warning(f"自动检测数据信息时出错：{e}")
            
    elif session_state.dfm_training_data_file is not None:
        st.info(f"当前已加载训练数据: {session_state.dfm_training_data_file.name}. 您可以上传新文件替换它。")
    else:
        st.info("请上传训练数据集。")

    st.markdown("**说明:**")
    st.markdown("- 此处上传的数据将用于DFM模型的训练或重新训练。")
    st.markdown("- 请确保数据格式符合模型要求。")

    
    # 添加重要提示
    st.info("📌 **重要说明**: 下面设置的日期范围将作为系统处理数据的**最大边界**。后续的训练期、验证期设置必须在此范围内。结果展示的Nowcasting默认覆盖此完整时间范围。")

    param_defaults = {
        'dfm_param_target_variable': '规模以上工业增加值:当月同比',
        'dfm_param_target_sheet_name': '工业增加值同比增速_月度_同花顺',
        'dfm_param_target_freq': 'W-FRI',
        'dfm_param_remove_consecutive_nans': True,
        'dfm_param_consecutive_nan_threshold': 10,
        'dfm_param_type_mapping_sheet': DataDefaults.TYPE_MAPPING_SHEET if CONFIG_AVAILABLE else '指标体系',
        'dfm_param_data_start_date': datetime(2020, 1, 1).date(),
        'dfm_param_data_end_date': None
    }
    for key, default_value in param_defaults.items():
        if key not in session_state:
            session_state[key] = default_value

    row1_col1, row1_col2 = st.columns(2)
    with row1_col1:
        session_state.dfm_param_data_start_date = st.date_input(
            "数据开始日期 (系统边界)",
            value=session_state.dfm_param_data_start_date,
            key="ss_dfm_data_start",
            help="设置系统处理数据的最早日期边界。训练期、验证期必须在此日期之后。"
        )
    with row1_col2:
        session_state.dfm_param_data_end_date = st.date_input(
            "数据结束日期 (系统边界)",
            value=session_state.dfm_param_data_end_date,
            key="ss_dfm_data_end",
            help="设置系统处理数据的最晚日期边界。训练期、验证期必须在此日期之前。"
        )

    row2_col1, row2_col2 = st.columns(2)
    with row2_col1:
        session_state.dfm_param_target_sheet_name = st.text_input(
            "目标工作表名称 (Target Sheet Name)", 
            value=session_state.dfm_param_target_sheet_name,
            key="ss_dfm_target_sheet"
        )
    with row2_col2:
        session_state.dfm_param_target_variable = st.text_input(
            "目标变量 (Target Variable)", 
            value=session_state.dfm_param_target_variable,
            key="ss_dfm_target_var"
        )

    row3_col1, row3_col2 = st.columns(2)
    with row3_col1:
        session_state.dfm_param_consecutive_nan_threshold = st.number_input(
            "连续 NaN 阈值 (Consecutive NaN Threshold)", 
            min_value=0, 
            value=session_state.dfm_param_consecutive_nan_threshold, 
            step=1,
            key="ss_dfm_nan_thresh"
        )
    with row3_col2:
        session_state.dfm_param_remove_consecutive_nans = st.checkbox(
            "移除过多连续 NaN 的变量", 
            value=session_state.dfm_param_remove_consecutive_nans,
            key="ss_dfm_remove_nans",
            help="移除列中连续缺失值数量超过阈值的变量"
        )

    row4_col1, row4_col2 = st.columns(2)
    with row4_col1:
        session_state.dfm_param_target_freq = st.text_input(
            "目标频率 (Target Frequency)", 
            value=session_state.dfm_param_target_freq,
            help="例如: W-FRI, D, M, Q",
            key="ss_dfm_target_freq"
        )
    with row4_col2:
        session_state.dfm_param_type_mapping_sheet = st.text_input(
            "指标映射表名称 (Type Mapping Sheet)", 
            value=session_state.dfm_param_type_mapping_sheet,
            key="ss_dfm_type_map_sheet"
        )

    st.markdown("--- ") # Separator before the new section
    st.markdown("#### 数据预处理与导出")

    # Initialize session_state for new UI elements if not already present
    if 'dfm_export_base_name' not in session_state:
        session_state.dfm_export_base_name = "dfm_prepared_output"
    if 'dfm_processed_outputs' not in session_state: # For storing results to persist downloads
        session_state.dfm_processed_outputs = None

    left_col, right_col = st.columns([1, 2]) # Left col for inputs, Right col for outputs/messages

    with left_col:
        session_state.dfm_export_base_name = st.text_input(
            "导出文件基础名称 (Export Base Filename)",
            value=session_state.dfm_export_base_name,
            key="ss_dfm_export_basename"
        )

        run_button_clicked = st.button("运行数据预处理并导出", key="ss_dfm_run_preprocessing")

    with right_col:
        if run_button_clicked:
            session_state.dfm_processed_outputs = None # Clear previous downloadable results
            # --- NEW: Clear previous direct data objects --- 
            session_state.dfm_prepared_data_df = None
            session_state.dfm_transform_log_obj = None
            session_state.dfm_industry_map_obj = None
            session_state.dfm_removed_vars_log_obj = None
            session_state.dfm_var_type_map_obj = None
            # --- END NEW ---

            if session_state.dfm_training_data_file is None:
                st.error("错误：请先上传训练数据集！")
            elif not session_state.dfm_export_base_name:
                st.error("错误：请指定有效的文件基础名称！")
            else:
                try:
                    import io 
                    import json
                    import pandas as pd
                    from .data_preparation import prepare_data, load_mappings

                    st.info("正在进行数据预处理... 详细日志请查看运行Streamlit的控制台。")
                    with st.spinner("数据预处理正在进行中，请稍候..."):
                        uploaded_file_bytes = session_state.dfm_training_data_file.getvalue()
                        excel_file_like_object = io.BytesIO(uploaded_file_bytes)
                        
                        start_date_str = session_state.dfm_param_data_start_date.strftime('%Y-%m-%d') \
                            if session_state.dfm_param_data_start_date else None
                        end_date_str = session_state.dfm_param_data_end_date.strftime('%Y-%m-%d') \
                            if session_state.dfm_param_data_end_date else None
                        
                        nan_threshold = session_state.dfm_param_consecutive_nan_threshold
                        nan_threshold_int = None
                        if not pd.isna(nan_threshold):
                            try:
                                nan_threshold_int = int(nan_threshold)
                            except ValueError:
                                st.warning(f"连续NaN阈值 '{nan_threshold}' 不是一个有效的整数。将忽略此阈值。")
                                nan_threshold_int = None

                        results = prepare_data(
                            excel_path=excel_file_like_object,
                            target_variable_name=session_state.dfm_param_target_variable,
                            target_sheet_name=session_state.dfm_param_target_sheet_name,
                            target_freq=session_state.dfm_param_target_freq,
                            consecutive_nan_threshold=nan_threshold_int,
                            data_start_date=start_date_str,
                            data_end_date=end_date_str,
                            reference_sheet_name=session_state.dfm_param_type_mapping_sheet,
                            # reference_column_name 使用配置的默认值
                        )

                        if results:
                            # 修复解包顺序：prepare_data返回 (data, industry_map, transform_log, removed_vars_log)
                            prepared_data, industry_map, transform_log, removed_variables_detailed_log = results
                            
                            # --- NEW: Also load var_type_map separately using load_mappings ---
                            try:
                                var_type_map, var_industry_map_loaded = load_mappings(
                                    excel_path=excel_file_like_object,
                                    sheet_name=session_state.dfm_param_type_mapping_sheet,
                                    indicator_col=DataDefaults.INDICATOR_COLUMN if CONFIG_AVAILABLE else '高频指标',
                                    type_col=DataDefaults.TYPE_COLUMN if CONFIG_AVAILABLE else '类型',
                                    industry_col=DataDefaults.INDUSTRY_COLUMN if CONFIG_AVAILABLE else '行业'
                                )
                                # Store var_type_map separately in session_state
                                session_state.dfm_var_type_map_obj = var_type_map
                                st.info(f"✅ 已成功加载变量类型映射：{len(var_type_map)} 个映射")
                            except Exception as e_load_maps:
                                st.warning(f"加载变量类型映射失败: {e_load_maps}")
                                session_state.dfm_var_type_map_obj = {}
                            # --- END NEW ---
                            
                            # --- NEW: Store Python objects in session_state for direct use ---
                            session_state.dfm_prepared_data_df = prepared_data
                            session_state.dfm_transform_log_obj = transform_log
                            session_state.dfm_industry_map_obj = industry_map
                            session_state.dfm_removed_vars_log_obj = removed_variables_detailed_log
                            
                            st.success("数据预处理完成！结果已准备就绪，可用于模型训练模块。")
                            # --- END NEW ---

                            # Prepare for download (existing logic)
                            session_state.dfm_processed_outputs = {
                                'base_name': session_state.dfm_export_base_name,
                                'data': None, 'industry_map': None, 'transform_log': None, 'removed_vars_log': None
                            }
                            if prepared_data is not None:
                                session_state.dfm_processed_outputs['data'] = prepared_data.to_csv(index=True, index_label='Date', encoding='utf-8-sig').encode('utf-8-sig')
                            
                            if industry_map:
                                try:
                                    df_industry_map = pd.DataFrame(list(industry_map.items()), columns=['Indicator', 'Industry'])
                                    session_state.dfm_processed_outputs['industry_map'] = df_industry_map.to_csv(index=False, encoding='utf-8-sig').encode('utf-8-sig')
                                except Exception as e_im:
                                    st.warning(f"行业映射转换到CSV时出错: {e_im}")
                                    session_state.dfm_processed_outputs['industry_map'] = None
                            
                            if removed_variables_detailed_log:
                                try:
                                    df_removed_log = pd.DataFrame(removed_variables_detailed_log)
                                    session_state.dfm_processed_outputs['removed_vars_log'] = df_removed_log.to_csv(index=False, encoding='utf-8-sig').encode('utf-8-sig')
                                except Exception as e_rl:
                                    st.warning(f"移除变量日志转换到CSV时出错: {e_rl}")
                                    session_state.dfm_processed_outputs['removed_vars_log'] = None

                            # Handling transform_log (it's a dict, potentially nested)
                            if transform_log:
                                formatted_log_data = []
                                # Attempt to flatten or nicely format the transform_log for CSV
                                # This is a simplified example; actual flattening might be more complex
                                for category, entries in transform_log.items():
                                    if isinstance(entries, dict):
                                        for var, details in entries.items():
                                            if isinstance(details, dict):
                                                log_entry = {'Category': category, 'Variable': var}
                                                log_entry.update(details) # Add all sub-details
                                                formatted_log_data.append(log_entry)
                                    elif isinstance(entries, list): # e.g. for 'removed_highly_correlated_vars'
                                         for item_pair in entries:
                                            if isinstance(item_pair, (list, tuple)) and len(item_pair) == 2:
                                                formatted_log_data.append({'Category': category, 'Variable1': item_pair[0], 'Variable2': item_pair[1]})
                                            else:
                                                formatted_log_data.append({'Category': category, 'Detail': str(item_pair)})
                                
                                if formatted_log_data:
                                    try:
                                        df_transformed_log_nice = pd.DataFrame(formatted_log_data)
                                        session_state.dfm_processed_outputs['transform_log'] = df_transformed_log_nice.to_csv(index=False, encoding='utf-8-sig').encode('utf-8-sig')
                                    except Exception as e_tl:
                                        st.warning(f"转换日志到CSV时出错: {e_tl}. 将尝试保存为JSON字符串。")
                                        try:
                                            session_state.dfm_processed_outputs['transform_log'] = json.dumps(transform_log, ensure_ascii=False, indent=4).encode('utf-8-sig')
                                        except Exception as e_json:
                                            st.warning(f"转换日志到JSON时也出错: {e_json}")
                                            session_state.dfm_processed_outputs['transform_log'] = None
                                else:
                                    session_state.dfm_processed_outputs['transform_log'] = None 
                                    st.info("转换日志为空或格式无法直接转换为简单CSV。")
                            else:
                                session_state.dfm_processed_outputs['transform_log'] = None
                        
                        else:
                            st.error("数据预处理失败或未返回数据。请检查控制台日志获取更多信息。")
                            session_state.dfm_processed_outputs = None
                            # Ensure direct data objects are also None on failure
                            session_state.dfm_prepared_data_df = None
                            session_state.dfm_transform_log_obj = None
                            session_state.dfm_industry_map_obj = None
                            session_state.dfm_removed_vars_log_obj = None
                            session_state.dfm_var_type_map_obj = None
                
                except ImportError as ie:
                    st.error(f"导入错误: {ie}. 请确保 'data_preparation.py' 文件与UI脚本在同一目录下或正确安装。")
                except FileNotFoundError as fnfe:
                    st.error(f"文件未找到错误: {fnfe}. 这可能与 'data_preparation.py' 内部的文件读取有关。")
                except Exception as e:
                    st.error(f"运行数据预处理时发生错误: {e}")
                    import traceback
                    st.text_area("详细错误信息:", traceback.format_exc(), height=200)
                    session_state.dfm_processed_outputs = None

        # Render download buttons if data is available in session_state
        if session_state.dfm_processed_outputs:
            outputs = session_state.dfm_processed_outputs
            base_name = outputs['base_name']

            st.download_button(
                label=f"下载处理后的数据 ({base_name}_data_v3.csv)",
                data=outputs['data'],
                file_name=f"{base_name}_data_v3.csv",
                mime='text/csv',
                key='download_data_csv'
            )

            if outputs['industry_map']:
                st.download_button(
                    label=f"下载行业映射 ({base_name}_industry_map_v3.csv)",
                    data=outputs['industry_map'],
                    file_name=f"{base_name}_industry_map_v3.csv",
                    mime='text/csv',
                    key='download_industry_map_csv'
                )
            
            if outputs['transform_log']:
                st.download_button(
                    label=f"下载转换日志 ({base_name}_transform_log_v3.csv)",
                    data=outputs['transform_log'],
                    file_name=f"{base_name}_transform_log_v3.csv",
                    mime='text/csv',
                    key='download_transform_log_csv'
                )

            if outputs['removed_vars_log']:
                st.download_button(
                    label=f"下载移除变量日志 ({base_name}_removed_log_v3.csv)",
                    data=outputs['removed_vars_log'],
                    file_name=f"{base_name}_removed_log_v3.csv",
                    mime='text/csv',
                    key='download_removed_log_csv'
                )
