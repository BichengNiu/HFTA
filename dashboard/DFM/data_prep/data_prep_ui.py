import streamlit as st

# --- 新增：导入状态管理器 ---
try:
    from ...core.state_manager import StateManager
    from ...core.compat import CompatibilityAdapter
    from ...core.state_keys import StateKeys
    DFM_STATE_MANAGER_AVAILABLE = True
except ImportError:
    DFM_STATE_MANAGER_AVAILABLE = False
    print("[DFM UI] Warning: State manager not available, using legacy state management")


def get_dfm_state_manager_instance():
    """获取状态管理器实例（DFM模块专用）"""
    if DFM_STATE_MANAGER_AVAILABLE and hasattr(st.session_state, 'state_manager_initialized'):
        try:
            state_manager = StateManager(st.session_state)
            compat_adapter = CompatibilityAdapter(st.session_state)
            return state_manager, compat_adapter
        except Exception as e:
            print(f"[DFM UI] Error getting state manager: {e}")
            return None, None
    return None, None


def get_dfm_state(key, default=None, session_state=None):
    """获取DFM状态值（兼容新旧状态管理）"""
    state_manager, compat_adapter = get_dfm_state_manager_instance()
    
    if compat_adapter is not None:
        return compat_adapter.get_value(key, default)
    else:
        # 回退到传统方式
        if session_state is not None:
            return getattr(session_state, key, default) if hasattr(session_state, key) else session_state.get(key, default)
        else:
            return st.session_state.get(key, default)


def set_dfm_state(key, value, session_state=None):
    """设置DFM状态值（兼容新旧状态管理）"""
    state_manager, compat_adapter = get_dfm_state_manager_instance()
    
    if compat_adapter is not None:
        compat_adapter.set_value(key, value, use_new_key=True)
    else:
        # 回退到传统方式
        if session_state is not None:
            if hasattr(session_state, key):
                setattr(session_state, key, value)
            else:
                session_state[key] = value
        else:
            st.session_state[key] = value
# --- 结束新增 ---

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
    if get_dfm_state('dfm_training_data_file', None, session_state) is None:
        set_dfm_state("dfm_training_data_file", None, session_state)
    
    # --- NEW: Initialize session_state for direct data passing ---
    if get_dfm_state('dfm_prepared_data_df', None, session_state) is None:
        set_dfm_state("dfm_prepared_data_df", None, session_state)
    if get_dfm_state('dfm_transform_log_obj', None, session_state) is None:
        set_dfm_state("dfm_transform_log_obj", None, session_state)
    if get_dfm_state('dfm_industry_map_obj', None, session_state) is None:
        set_dfm_state("dfm_industry_map_obj", None, session_state)
    if get_dfm_state('dfm_removed_vars_log_obj', None, session_state) is None:
        set_dfm_state("dfm_removed_vars_log_obj", None, session_state)
    if get_dfm_state('dfm_var_type_map_obj', None, session_state) is None:
        set_dfm_state("dfm_var_type_map_obj", None, session_state)
    # --- END NEW ---

    uploaded_file = st.file_uploader(
        "选择训练数据集 (例如：.csv, .xlsx)", 
        type=["csv", "xlsx"], 
        key="dfm_training_data_uploader",
        help="请上传包含模型训练所需指标的表格数据。"
    )

    if uploaded_file is not None:
        # 检查是否是新文件上传（避免重复处理）
        file_changed = (
            session_state.dfm_training_data_file is None or
            session_state.dfm_training_data_file.name != uploaded_file.name or
            get_dfm_state('dfm_file_processed', False, session_state) == False
        )

        set_dfm_state("dfm_training_data_file", uploaded_file, session_state)
        # 🔥 新增：保存Excel文件路径用于训练模块
        set_dfm_state("dfm_uploaded_excel_file_path", uploaded_file.name, session_state)
        set_dfm_state("dfm_use_full_data_preparation", True, session_state)

        # 只有在文件发生变化时才标记需要重新检测
        if file_changed:
            print(f"[UI] 检测到新文件上传: {uploaded_file.name}，标记需要重新检测...")
            set_dfm_state("dfm_file_processed", False, session_state)  # 重置处理标记
            set_dfm_state("dfm_date_detection_needed", True, session_state)  # 标记需要日期检测

            # 显示文件上传成功信息
            st.success(f"文件 '{uploaded_file.name}' 已上传并准备就绪。")
            st.info("📅 文件已加载，将自动检测日期范围。")

            # 标记文件已处理
            set_dfm_state("dfm_file_processed", True, session_state)
        else:
            # 文件没有变化，显示简单的状态信息
            st.success(f"文件 '{uploaded_file.name}' 已加载。")
            
    elif session_state.dfm_training_data_file is not None:
        st.info(f"当前已加载训练数据: {session_state.dfm_training_data_file.name}. 您可以上传新文件替换它。")
        
        # 添加文件结构检查工具
        with st.expander("🔍 文件结构诊断工具 (可选)", expanded=False):
            if st.button("检查文件结构", help="查看已上传文件的内部结构，帮助诊断格式问题"):
                try:
                    import io
                    import pandas as pd
                    
                    uploaded_file_bytes = session_state.dfm_training_data_file.getvalue()
                    excel_file_like_object = io.BytesIO(uploaded_file_bytes)
                    
                    if session_state.dfm_training_data_file.name.endswith('.xlsx'):
                        xl_file = pd.ExcelFile(excel_file_like_object)
                        sheet_names = xl_file.sheet_names
                        
                        st.write(f"**文件包含 {len(sheet_names)} 个工作表:**")
                        for i, sheet_name in enumerate(sheet_names):  # 显示所有工作表
                            with st.expander(f"工作表 {i+1}: {sheet_name}", expanded=(i==0)):
                                try:
                                    # 尝试使用格式检测
                                    try:
                                        from .data_preparation import detect_sheet_format
                                        format_info = detect_sheet_format(excel_file_like_object, sheet_name)
                                        st.write(f"**检测到的格式:** {format_info['format']} (来源: {format_info['source']})")
                                        st.write(f"**建议参数:** header={format_info['header']}, skiprows={format_info['skiprows']}")
                                    except:
                                        st.write("**格式检测:** 使用默认参数")
                                    
                                    # 读取前几行
                                    df_preview = pd.read_excel(excel_file_like_object, sheet_name=sheet_name, nrows=5)
                                    st.write(f"**数据形状:** {df_preview.shape}")
                                    st.write("**前5行预览:**")
                                    st.dataframe(df_preview)
                                    
                                    # 检查第一列是否可能是日期
                                    if len(df_preview.columns) > 0:
                                        first_col = df_preview.columns[0]
                                        first_col_values = df_preview[first_col].dropna().head(3)
                                        st.write(f"**第一列 '{first_col}' 的样本值:** {list(first_col_values)}")
                                        
                                        # 尝试转换为日期
                                        try:
                                            date_converted = pd.to_datetime(first_col_values, errors='coerce')
                                            if not date_converted.isna().all():
                                                st.success(f"✅ 第一列可以转换为日期: {date_converted.dropna().iloc[0]}")
                                            else:
                                                st.warning("⚠️ 第一列无法转换为日期")
                                        except:
                                            st.warning("⚠️ 第一列日期转换出错")
                                    
                                except Exception as e:
                                    st.error(f"读取工作表出错: {e}")
                        
                        # 移除限制显示的提示信息
                            
                except Exception as e:
                    st.error(f"文件结构检查出错: {e}")
    else:
        st.info("请上传训练数据集。")

    st.markdown("**说明:**")
    st.markdown("- 此处上传的数据将用于DFM模型的训练或重新训练。")
    st.markdown("- 请确保数据格式符合模型要求。")

    
    # 添加重要提示
    st.info("📌 **重要说明**: 下面设置的日期范围将作为系统处理数据的**最大边界**。后续的训练期、验证期设置必须在此范围内。结果展示的Nowcasting默认覆盖此完整时间范围。")

    # 🔥 正确修复：根据数据文件的实际日期范围进行检测
    def detect_data_date_range(uploaded_file):
        """从上传的文件中检测数据的真实日期范围"""
        try:
            if uploaded_file is None:
                return None, None

            import io
            import pandas as pd

            # 读取文件
            file_bytes = uploaded_file.getvalue()
            excel_file = io.BytesIO(file_bytes)

            all_dates_found = []

            # 获取所有工作表名称
            try:
                xl_file = pd.ExcelFile(excel_file)
                sheet_names = xl_file.sheet_names
                print(f"检测到工作表: {sheet_names}")
            except:
                sheet_names = [0]  # 回退到第一个工作表

            # 检查每个工作表寻找真实的日期数据
            for sheet_name in sheet_names:
                try:
                    excel_file.seek(0)  # 重置文件指针

                    # 跳过明显的元数据工作表
                    if any(keyword in str(sheet_name).lower() for keyword in ['指标体系', 'mapping', 'meta', 'info']):
                        print(f"跳过元数据工作表: {sheet_name}")
                        continue

                    # 读取工作表
                    df = pd.read_excel(excel_file, sheet_name=sheet_name, index_col=0)

                    if len(df) < 5:  # 跳过数据太少的工作表
                        continue

                    # 检查索引中的日期
                    datetime_indices = []
                    for idx in df.index:
                        # 检查是否是日期时间类型
                        if pd.api.types.is_datetime64_any_dtype(pd.Series([idx])):
                            datetime_indices.append(idx)
                        elif isinstance(idx, pd.Timestamp):
                            datetime_indices.append(idx)
                        elif hasattr(idx, 'year'):  # datetime对象
                            datetime_indices.append(idx)

                    if len(datetime_indices) > 10:  # 至少要有10个日期才认为是时间序列
                        dates_series = pd.to_datetime(datetime_indices)
                        all_dates_found.extend(dates_series.tolist())
                        print(f"✅ 工作表 '{sheet_name}': 找到 {len(dates_series)} 个日期")
                        print(f"   范围: {dates_series.min().date()} 到 {dates_series.max().date()}")

                except Exception as e:
                    print(f"⚠️ 工作表 '{sheet_name}': 处理失败 - {e}")
                    continue

            # 汇总所有真实日期，返回实际的数据范围
            if all_dates_found:
                all_dates = pd.to_datetime(all_dates_found)
                actual_start = all_dates.min().date()
                actual_end = all_dates.max().date()

                print(f"✅ 检测到数据的实际日期范围: {actual_start} 到 {actual_end}")
                print(f"   总共 {len(all_dates)} 个日期点，年份跨度: {actual_start.year}-{actual_end.year}")

                return actual_start, actual_end
            else:
                print("❌ 未能检测到任何日期数据")
                return None, None

        except Exception as e:
            print(f"❌ 检测日期范围失败: {e}")
            return None, None

    # 🔥 优化：增强缓存机制，避免重复检测
    # 检测上传文件的日期范围（只在文件变化时执行）
    if session_state.dfm_training_data_file:
        file_hash = hash(session_state.dfm_training_data_file.getvalue())
        cache_key = f"date_range_{session_state.dfm_training_data_file.name}_{file_hash}"
    else:
        cache_key = "date_range_none"

    # 检查缓存是否存在且有效
    cache_valid = (
        cache_key in session_state and
        not get_dfm_state('dfm_date_detection_needed', False, session_state)
    )

    if not cache_valid:
        # 需要重新检测
        if session_state.dfm_training_data_file:
            print(f"[UI] 执行日期检测: {session_state.dfm_training_data_file.name}")
            with st.spinner("🔍 正在检测数据日期范围..."):
                detected_start, detected_end = detect_data_date_range(session_state.dfm_training_data_file)
            # 缓存结果
            session_state[cache_key] = (detected_start, detected_end)
            set_dfm_state("dfm_date_detection_needed", False, session_state)

            # 清理旧的缓存
            old_keys = [k for k in session_state.keys() if k.startswith("date_range_") and k != cache_key]
            for old_key in old_keys:
                del session_state[old_key]
        else:
            detected_start, detected_end = None, None
            session_state[cache_key] = (None, None)
    else:
        # 使用缓存的结果
        detected_start, detected_end = session_state[cache_key]
        if detected_start and detected_end:
            print(f"[UI] 使用缓存的日期范围: {detected_start} 到 {detected_end}")

    # 设置默认值：优先使用检测到的日期，否则使用硬编码默认值
    default_start_date = detected_start if detected_start else datetime(2020, 1, 1).date()
    default_end_date = detected_end if detected_end else datetime(2025, 4, 30).date()

    param_defaults = {
        'dfm_param_target_variable': '规模以上工业增加值:当月同比',
        'dfm_param_target_sheet_name': '工业增加值同比增速_月度_同花顺',
        'dfm_param_target_freq': 'W-FRI',
        'dfm_param_remove_consecutive_nans': True,
        'dfm_param_consecutive_nan_threshold': 10,
        'dfm_param_type_mapping_sheet': DataDefaults.TYPE_MAPPING_SHEET if CONFIG_AVAILABLE else '指标体系',
        'dfm_param_data_start_date': default_start_date,
        'dfm_param_data_end_date': default_end_date
    }

    # 🔥 修复：只在首次初始化或文件更新时设置默认值
    for key, default_value in param_defaults.items():
        if key not in session_state:
            session_state[key] = default_value
        elif key in ['dfm_param_data_start_date', 'dfm_param_data_end_date'] and detected_start and detected_end:
            # 如果检测到新的日期范围，更新日期设置
            if key == 'dfm_param_data_start_date':
                session_state[key] = default_start_date
            elif key == 'dfm_param_data_end_date':
                session_state[key] = default_end_date

    # 显示检测结果
    if detected_start and detected_end:
        st.success(f"✅ 已自动检测文件日期范围: {detected_start} 到 {detected_end}")
    elif session_state.dfm_training_data_file:
        st.warning("⚠️ 无法自动检测文件日期范围，使用默认值。请手动调整日期设置。")

    row1_col1, row1_col2 = st.columns(2)
    with row1_col1:
        set_dfm_state("dfm_param_data_start_date", st.date_input(
            "数据开始日期 (系统边界)",
            value=session_state.dfm_param_data_start_date,
            key="ss_dfm_data_start",
            help="设置系统处理数据的最早日期边界。训练期、验证期必须在此日期之后。"
        ), session_state)
    with row1_col2:
        set_dfm_state("dfm_param_data_end_date", st.date_input(
            "数据结束日期 (系统边界)",
            value=session_state.dfm_param_data_end_date,
            key="ss_dfm_data_end",
            help="设置系统处理数据的最晚日期边界。训练期、验证期必须在此日期之前。"
        ), session_state)

    row2_col1, row2_col2 = st.columns(2)
    with row2_col1:
        set_dfm_state("dfm_param_target_sheet_name", st.text_input(
            "目标工作表名称 (Target Sheet Name)",
            value=session_state.dfm_param_target_sheet_name,
            key="ss_dfm_target_sheet"
        ), session_state)
    with row2_col2:
        set_dfm_state("dfm_param_target_variable", st.text_input(
            "目标变量 (Target Variable)",
            value=session_state.dfm_param_target_variable,
            key="ss_dfm_target_var"
        ), session_state)

    row3_col1, row3_col2 = st.columns(2)
    with row3_col1:
        set_dfm_state("dfm_param_consecutive_nan_threshold", st.number_input(
            "连续 NaN 阈值 (Consecutive NaN Threshold)",
            min_value=0,
            value=session_state.dfm_param_consecutive_nan_threshold,
            step=1,
            key="ss_dfm_nan_thresh"
        ), session_state)
    with row3_col2:
        set_dfm_state("dfm_param_remove_consecutive_nans", st.checkbox(
            "移除过多连续 NaN 的变量",
            value=session_state.dfm_param_remove_consecutive_nans,
            key="ss_dfm_remove_nans",
            help="移除列中连续缺失值数量超过阈值的变量"
        ), session_state)

    row4_col1, row4_col2 = st.columns(2)
    with row4_col1:
        set_dfm_state("dfm_param_target_freq", st.text_input(
            "目标频率 (Target Frequency)",
            value=session_state.dfm_param_target_freq,
            help="例如: W-FRI, D, M, Q",
            key="ss_dfm_target_freq"
        ), session_state)
    with row4_col2:
        set_dfm_state("dfm_param_type_mapping_sheet", st.text_input(
            "指标映射表名称 (Type Mapping Sheet)",
            value=session_state.dfm_param_type_mapping_sheet,
            key="ss_dfm_type_map_sheet"
        ), session_state)

    st.markdown("--- ") # Separator before the new section
    st.markdown("#### 数据预处理与导出")

    # Initialize session_state for new UI elements if not already present
    if get_dfm_state('dfm_export_base_name', None, session_state) is None:
        set_dfm_state("dfm_export_base_name", "dfm_prepared_output", session_state)
    if get_dfm_state('dfm_processed_outputs', None, session_state) is None: # For storing results to persist downloads
        set_dfm_state("dfm_processed_outputs", None, session_state)

    left_col, right_col = st.columns([1, 2]) # Left col for inputs, Right col for outputs/messages

    with left_col:
        set_dfm_state("dfm_export_base_name", st.text_input(
            "导出文件基础名称 (Export Base Filename)",
            value=session_state.dfm_export_base_name,
            key="ss_dfm_export_basename"
        ), session_state)

        run_button_clicked = st.button("运行数据预处理并导出", key="ss_dfm_run_preprocessing")

    with right_col:
        if run_button_clicked:
            set_dfm_state("dfm_processed_outputs", None, session_state)  # Clear previous downloadable results
            # --- NEW: Clear previous direct data objects --- 
            set_dfm_state("dfm_prepared_data_df", None, session_state)
            set_dfm_state("dfm_transform_log_obj", None, session_state)
            set_dfm_state("dfm_industry_map_obj", None, session_state)
            set_dfm_state("dfm_removed_vars_log_obj", None, session_state)
            set_dfm_state("dfm_var_type_map_obj", None, session_state)
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

                    # 🔥 优化：添加详细的进度指示器
                    progress_bar = st.progress(0)
                    status_text = st.empty()

                    try:
                        status_text.text("🔄 正在准备数据...")
                        progress_bar.progress(10)

                        uploaded_file_bytes = session_state.dfm_training_data_file.getvalue()
                        excel_file_like_object = io.BytesIO(uploaded_file_bytes)

                        start_date_str = session_state.dfm_param_data_start_date.strftime('%Y-%m-%d') \
                            if session_state.dfm_param_data_start_date else None
                        end_date_str = session_state.dfm_param_data_end_date.strftime('%Y-%m-%d') \
                            if session_state.dfm_param_data_end_date else None

                        status_text.text("📊 正在读取数据文件...")
                        progress_bar.progress(20)
                        
                        # 🔧 修复：只有在启用移除连续NaN功能时才传递阈值
                        nan_threshold_int = None
                        if session_state.dfm_param_remove_consecutive_nans:
                            nan_threshold = session_state.dfm_param_consecutive_nan_threshold
                            if not pd.isna(nan_threshold):
                                try:
                                    nan_threshold_int = int(nan_threshold)
                                except ValueError:
                                    st.warning(f"连续NaN阈值 '{nan_threshold}' 不是一个有效的整数。将忽略此阈值。")
                                    nan_threshold_int = None

                        # 🔥🔥🔥 新增：数据预处理参数调试
                        print(f"\n" + "="*80)
                        print(f"🔍🔍🔍 [数据预处理参数调试] 检查变量过滤设置")
                        print(f"="*80)
                        print(f"📊 预处理参数:")
                        print(f"   目标变量: {session_state.dfm_param_target_variable}")
                        print(f"   目标频率: {session_state.dfm_param_target_freq}")
                        print(f"   是否移除连续NaN变量: {session_state.dfm_param_remove_consecutive_nans}")
                        print(f"   连续NaN阈值: {session_state.dfm_param_consecutive_nan_threshold}")
                        print(f"   实际传递的阈值: {nan_threshold_int}")
                        print(f"   数据日期范围: {start_date_str} 到 {end_date_str}")

                        if nan_threshold_int is not None:
                            print(f"⚠️ 警告: 连续NaN阈值设置为 {nan_threshold_int}，可能会移除大量变量！")
                            print(f"   任何连续缺失值 ≥ {nan_threshold_int} 的变量都会被移除")
                        else:
                            print(f"✅ 连续NaN过滤已禁用，不会移除变量")
                        print(f"="*80)

                        status_text.text("🔧 正在执行数据预处理...")
                        progress_bar.progress(30)

                        # 修复参数顺序以匹配data_preparation.py中的函数签名
                        results = prepare_data(
                            excel_path=excel_file_like_object,
                            target_freq=session_state.dfm_param_target_freq,
                            target_sheet_name=session_state.dfm_param_target_sheet_name,
                            target_variable_name=session_state.dfm_param_target_variable,
                            consecutive_nan_threshold=nan_threshold_int,
                            data_start_date=start_date_str,
                            data_end_date=end_date_str,
                            reference_sheet_name=session_state.dfm_param_type_mapping_sheet,
                            # reference_column_name 使用配置的默认值
                        )

                        status_text.text("✅ 数据预处理完成，正在生成结果...")
                        progress_bar.progress(70)

                        if results:
                            status_text.text("📋 正在处理结果数据...")
                            progress_bar.progress(80)

                            # 修复解包顺序：prepare_data返回 (data, industry_map, transform_log, removed_vars_log)
                            prepared_data, industry_map, transform_log, removed_variables_detailed_log = results

                            # 🔥🔥🔥 新增：详细的变量移除分析
                            print(f"\n" + "="*80)
                            print(f"🔍🔍🔍 [数据预处理结果分析] 变量移除详情")
                            print(f"="*80)
                            print(f"📊 预处理结果:")
                            print(f"   最终数据形状: {prepared_data.shape}")
                            print(f"   最终变量数量: {len(prepared_data.columns)}")
                            print(f"   移除的变量数量: {len(removed_variables_detailed_log) if removed_variables_detailed_log else 0}")

                            if removed_variables_detailed_log:
                                print(f"\n📋 移除变量详细分析:")
                                removal_reasons = {}
                                for item in removed_variables_detailed_log:
                                    reason = item.get('Reason', 'unknown')
                                    if reason not in removal_reasons:
                                        removal_reasons[reason] = []
                                    removal_reasons[reason].append(item.get('Variable', 'unknown'))

                                for reason, vars_list in removal_reasons.items():
                                    print(f"   🔸 {reason}: {len(vars_list)} 个变量")
                                    if 'consecutive_nan' in reason.lower():
                                        print(f"      ❌ 因连续缺失值过多被移除: {vars_list[:5]}{'...' if len(vars_list) > 5 else ''}")
                                    else:
                                        print(f"      - 变量: {vars_list[:3]}{'...' if len(vars_list) > 3 else ''}")

                                # 统计连续NaN移除的变量
                                nan_removed = [item for item in removed_variables_detailed_log if 'consecutive_nan' in item.get('Reason', '').lower()]
                                if nan_removed:
                                    print(f"\n⚠️ 重要发现: {len(nan_removed)} 个变量因连续缺失值 ≥ {nan_threshold_int} 被移除！")
                                    print(f"   这就是为什么从76个变量变成{len(prepared_data.columns)}个的原因！")
                                    print(f"   💡 解决方案: 增加连续NaN阈值或禁用此功能")
                            else:
                                print(f"✅ 没有变量被移除")

                            print(f"="*80)

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
                                set_dfm_state("dfm_var_type_map_obj", var_type_map, session_state)
                                st.info(f"✅ 已成功加载变量类型映射：{len(var_type_map)} 个映射")
                            except Exception as e_load_maps:
                                st.warning(f"加载变量类型映射失败: {e_load_maps}")
                                set_dfm_state("dfm_var_type_map_obj", {}, session_state)
                            # --- END NEW ---
                            
                            # --- NEW: Store Python objects in session_state for direct use ---
                            set_dfm_state("dfm_prepared_data_df", prepared_data, session_state)
                            set_dfm_state("dfm_transform_log_obj", transform_log, session_state)
                            set_dfm_state("dfm_industry_map_obj", industry_map, session_state)
                            set_dfm_state("dfm_removed_vars_log_obj", removed_variables_detailed_log, session_state)
                            
                            st.success("数据预处理完成！结果已准备就绪，可用于模型训练模块。")
                            st.info(f"📊 预处理后数据形状: {prepared_data.shape}")

                            # 🔥 新增：在UI中显示变量移除警告
                            if removed_variables_detailed_log:
                                nan_removed_count = len([item for item in removed_variables_detailed_log if 'consecutive_nan' in item.get('Reason', '').lower()])
                                if nan_removed_count > 0:
                                    st.warning(f"⚠️ 注意: {nan_removed_count} 个变量因连续缺失值 ≥ {nan_threshold_int} 被移除！")
                                    st.info(f"💡 如需保留更多变量，可以增加连续NaN阈值或禁用此功能")

                                    with st.expander("🔍 查看被移除的变量详情", expanded=False):
                                        removal_reasons = {}
                                        for item in removed_variables_detailed_log:
                                            reason = item.get('Reason', 'unknown')
                                            if reason not in removal_reasons:
                                                removal_reasons[reason] = []
                                            removal_reasons[reason].append(item.get('Variable', 'unknown'))

                                        for reason, vars_list in removal_reasons.items():
                                            st.write(f"**{reason}**: {len(vars_list)} 个变量")
                                            if 'consecutive_nan' in reason.lower():
                                                st.error(f"因连续缺失值过多被移除: {vars_list[:10]}")
                                            else:
                                                st.write(f"变量: {vars_list[:5]}")
                            # --- END NEW ---

                            # Prepare for download (existing logic)
                            set_dfm_state("dfm_processed_outputs", {
                                'base_name': session_state.dfm_export_base_name,
                                'data': None, 'industry_map': None, 'transform_log': None, 'removed_vars_log': None
                            }, session_state)
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
                            progress_bar.progress(100)
                            status_text.text("❌ 处理失败")
                            st.error("数据预处理失败或未返回数据。请检查控制台日志获取更多信息。")
                            set_dfm_state("dfm_processed_outputs", None, session_state)
                            # Ensure direct data objects are also None on failure
                            set_dfm_state("dfm_prepared_data_df", None, session_state)
                            set_dfm_state("dfm_transform_log_obj", None, session_state)
                            set_dfm_state("dfm_industry_map_obj", None, session_state)
                            set_dfm_state("dfm_removed_vars_log_obj", None, session_state)
                            set_dfm_state("dfm_var_type_map_obj", None, session_state)

                        # 🔥 优化：完成进度指示器
                        if 'progress_bar' in locals():
                            progress_bar.progress(100)
                            status_text.text("🎉 处理完成！")
                            import time
                            time.sleep(1)
                            progress_bar.empty()
                            status_text.empty()

                    except ImportError as ie:
                        st.error(f"导入错误: {ie}. 请确保 'data_preparation.py' 文件与UI脚本在同一目录下或正确安装。")
                    except FileNotFoundError as fnfe:
                        st.error(f"文件未找到错误: {fnfe}. 这可能与 'data_preparation.py' 内部的文件读取有关。")
                    except Exception as e:
                        st.error(f"运行数据预处理时发生错误: {e}")
                        import traceback
                        st.text_area("详细错误信息:", traceback.format_exc(), height=200)
                        set_dfm_state("dfm_processed_outputs", None, session_state)

                except Exception as outer_e:
                    st.error(f"数据预处理过程中发生未预期的错误: {outer_e}")
                    import traceback
                    st.text_area("详细错误信息:", traceback.format_exc(), height=200)

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
