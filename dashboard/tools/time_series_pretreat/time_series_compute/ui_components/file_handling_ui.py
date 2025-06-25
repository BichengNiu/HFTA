import streamlit as st
import pandas as pd
import io # Not directly used here, but good practice
import re
from ..utils.data_loaders import process_uploaded_data

# Callback function to reset states when data source changes
def on_data_source_change():
    st.session_state.ts_compute_data = None
    st.session_state.ts_compute_original_data = None
    st.session_state.ts_compute_file_name = None
    # if 'ts_compute_uploader_state_reset_key' not in st.session_state:  # Temporarily comment out key reset
    #     st.session_state.ts_compute_uploader_state_reset_key = 0
    # st.session_state.ts_compute_uploader_state_reset_key += 1        # Temporarily comment out key reset
    st.session_state.ts_compute_selected_staged_key = None 

# --- 新增：辅助函数，用于解析 skiprows 字符串 ---
def parse_skiprows(skiprows_str: str):
    if not skiprows_str:
        return None
    rows_to_skip = set()
    parts = skiprows_str.split(',')
    for part in parts:
        part = part.strip()
        if not part:
            continue
        if '-' in part:
            try:
                start_end = part.split('-')
                if len(start_end) == 2:
                    start, end = map(int, start_end)
                    if start <= end:
                        rows_to_skip.update(range(start, end + 1))
                    else: 
                        st.warning(f"跳过行范围无效: {part}，已忽略。") 
                else: 
                    st.warning(f"跳过行范围格式错误: {part}，已忽略。")
            except ValueError: 
                st.warning(f"跳过行范围包含非数字: {part}，已忽略。")
                continue
        else:
            try:
                rows_to_skip.add(int(part))
            except ValueError: 
                st.warning(f"跳过行条目包含非数字: {part}，已忽略。")
                continue
    return sorted(list(rows_to_skip)) if rows_to_skip else None
# --- 结束新增 ---

# Helper function to display data overview (to reduce duplication)
def _display_data_overview_for_compute(st, df, source_name):
    st.subheader(f"数据概览 ({source_name})")
    rows, cols = df.shape 
    # cols here refers to df.shape[1], which is columns excluding index if index is set
    
    info_cols = st.columns(2)
    with info_cols[0]:
        st.metric("观测值数量 (行)", rows)
    with info_cols[1]:
        st.metric("变量数量 (列)", cols) # df.shape[1] correctly gives non-index columns

    time_info_str = "时间信息：\n"
    freq_display = "无法确定频率" # Renamed from freq_display_init
    if isinstance(df.index, pd.DatetimeIndex):
        time_info_str += f"  - 开始时间: {df.index.min()}\n"
        time_info_str += f"  - 结束时间: {df.index.max()}\n"
        try:
            inferred_freq = pd.infer_freq(df.index)
            if inferred_freq:
                freq_map = { # Renamed from freq_map_init
                    'A': "年度", 'AS': "年初开始的年度",
                    'Q': "季度", 'QS': "季初开始的季度",
                    'M': "月度", 'MS': "月初开始的月度",
                    'W': "周度", 'D': "日度", 'B': "工作日",
                    'H': "小时", 'T': "分钟", 'S': "秒",
                    'L': "毫秒", 'U': "微秒"
                }
                freq_prefix_match = re.match(r"^[A-Za-z]+", inferred_freq)
                if freq_prefix_match and freq_prefix_match.group(0) in freq_map:
                    freq_display = f"{freq_map[freq_prefix_match.group(0)]} ({inferred_freq})"
                else:
                    freq_display = f"推断频率: {inferred_freq}"
            else: # inferred_freq is None
                if len(df.index) > 1:
                    time_diffs = df.index.to_series().diff().dropna()
                    if not time_diffs.empty:
                        median_diff = time_diffs.median() # Renamed
                        common_diffs = time_diffs.value_counts().head(3) # Renamed
                        
                        freq_display = f"不规则时间序列 (中位时间差: {median_diff})\n"
                        freq_display += "  - 主要时间间隔 (前3): \n"
                        for diff_val, count_val in common_diffs.items():
                            freq_display += f"    - {diff_val} (出现 {count_val} 次)\n"
                        
                        is_likely_monthly = False
                        if not common_diffs.empty:
                            # Ensure index is not empty and is a Timedelta before accessing .days
                            if common_diffs.index.size > 0 and isinstance(common_diffs.index[0], pd.Timedelta):
                                first_common_diff_days = common_diffs.index[0].days
                                if 28 <= first_common_diff_days <= 31:
                                    is_likely_monthly = True
                        
                        if is_likely_monthly:
                            freq_display += "  - (数据可能具有月度特征，但日期不完全规整)"
                    else:
                        freq_display = "不规则时间序列 (无法计算时间差)"
                else:
                    freq_display = "不规则时间序列 (数据点不足)"
        except Exception as e_freq: # Renamed
            freq_display = f"推断频率时出错: {e_freq}"
        time_info_str += f"  - {freq_display}"
        st.info(f"数据 '{source_name}' 已加载。时间索引已识别。\n{time_info_str}")
    else:
        st.warning(f"数据 '{source_name}' 的索引不是标准日期时间格式，或未能将第一列转换为有效索引。部分时间信息和计算功能可能受限。\n{time_info_str}")


def display_file_upload_and_load(st, session_state):
    """处理文件上传、从暂存区加载数据及初始数据显示。"""

    st.subheader("选择数据来源与加载")

    data_source_options = ["上传新文件", "从暂存区选择"]
    if 'ts_compute_data_source_radio' not in session_state:
        session_state.ts_compute_data_source_radio = data_source_options[0]

    st.radio(
        "选择数据来源:",
        options=data_source_options,
        key="ts_compute_data_source_radio",
        horizontal=True,
        on_change=on_data_source_change # 暂时保留 on_change，但其内部重置 uploader key 的逻辑已注释
    )

    if session_state.ts_compute_data_source_radio == "上传新文件":
        uploaded_file = st.file_uploader(
            "上传 CSV 或 Excel 文件:",
            type=["csv", "xlsx"],
            key="ts_compute_uploader_FIXED_KEY", # 使用固定key，避免因重置导致的问题
            help="上传包含时间序列数据的文件。将自动以第一行为表头，第一列为时间索引进行处理。",
            on_change=on_data_source_change # 当文件改变时，也重置相关状态
        )

        if uploaded_file is not None:
            # 检查是否需要处理新文件 (避免 st.rerun() 导致的重复处理)
            # 如果当前文件名与session中记录的文件名不同，或者session中无数据，则处理
            needs_processing = (
                session_state.get('ts_compute_file_name') != uploaded_file.name or
                session_state.get('ts_compute_data') is None
            )

            if needs_processing:
                st.info(f"检测到新文件: {uploaded_file.name}，正在尝试自动处理...")
                with st.spinner(f"正在加载并处理文件: {uploaded_file.name} (默认第一行为表头，第一列为时间戳)..."):
                    # 将 uploaded_file 传递给 process_uploaded_data
                    # 注意: process_uploaded_data 需要能处理 BytesIO 或 UploadedFile 对象
                    
                    # 为了能重新读取文件，需要 seek(0)
                    if hasattr(uploaded_file, 'seek'):
                         uploaded_file.seek(0)
                    
                    processed_data = process_uploaded_data(
                        uploaded_file, # 直接传递 UploadedFile 对象
                        time_index_col=0,
                        skiprows=None, 
                        header=0
                    )

                    if processed_data is not None and not processed_data.empty:
                        if isinstance(processed_data.index, pd.DatetimeIndex) and not processed_data.index.isna().all():
                            # 检查NaT比例，例如超过30%则认为有问题
                            if processed_data.index.isna().sum() <= len(processed_data.index) * 0.3:
                                session_state.ts_compute_data = processed_data
                                session_state.ts_compute_original_data = processed_data.copy()
                                session_state.ts_compute_file_name = uploaded_file.name
                                st.success(f"文件 '{uploaded_file.name}' 自动加载和处理成功。")
                                # 不再直接调用 _display_data_overview_for_compute, 主tab会处理
                                st.rerun() # 成功加载后刷新整个UI
                            else:
                                st.error("非标准数据形式：时间戳列包含过多无效日期或无法解析。须使用数据清理模块标准化清理。")
                                session_state.ts_compute_data = None
                                session_state.ts_compute_original_data = None
                                session_state.ts_compute_file_name = uploaded_file.name # Keep name for re-attempt info
                        else:
                            st.error("非标准数据形式：第一列未能成功解析为有效的时间戳索引。须使用数据清理模块标准化清理。")
                            session_state.ts_compute_data = None
                            session_state.ts_compute_original_data = None
                            session_state.ts_compute_file_name = uploaded_file.name 
                    elif processed_data is not None and processed_data.empty:
                        st.error(f"文件 '{uploaded_file.name}' 加载后数据为空。请检查文件内容。")
                        session_state.ts_compute_data = None
                        session_state.ts_compute_original_data = None
                        session_state.ts_compute_file_name = uploaded_file.name
                    else: # processed_data is None
                        st.error(f"无法加载或处理文件 '{uploaded_file.name}'。文件可能已损坏、格式不支持或内容不符合预期。请检查文件或使用数据清理模块。")
                        session_state.ts_compute_data = None
                        session_state.ts_compute_original_data = None
                        session_state.ts_compute_file_name = uploaded_file.name
            
            # 如果处理失败，显示提示 (基于 session_state.ts_compute_data 为 None)
            # 只有当尝试加载的文件是当前上传的文件，并且数据加载失败时才显示
            if session_state.get('ts_compute_file_name') == uploaded_file.name and session_state.get('ts_compute_data') is None:
                # 避免在成功加载后（st.rerun之前）显示此消息
                # 通过检查是否有error消息（这比较hacky，更好的方式是引入一个加载失败的flag）
                # 暂时只在 needs_processing 为 True 且 data is None 时显示
                 if needs_processing: # 只有在刚才尝试处理失败时才显示
                    st.warning(f"文件 '{uploaded_file.name}' 自动处理失败或数据不符合标准。请检查上述错误提示或使用数据清理模块。")

    elif session_state.ts_compute_data_source_radio == "从暂存区选择":
        staged_data = session_state.get('staged_data', {})
        if not staged_data:
            st.warning('暂存区为空。请先在"数据清洗"或其他模块中处理并暂存数据，或选择"上传新文件"。')
            session_state.ts_compute_data = None
            session_state.ts_compute_original_data = None
            session_state.ts_compute_file_name = None
        else:
            staged_keys = list(staged_data.keys())
            if not staged_keys:
                st.info("暂存区当前没有可供选择的数据集。")
                session_state.ts_compute_selected_staged_key = None
                session_state.ts_compute_data = None
                session_state.ts_compute_original_data = None
                session_state.ts_compute_file_name = None
            else:
                selected_key_from_session = session_state.get('ts_compute_selected_staged_key')
                if not selected_key_from_session or selected_key_from_session not in staged_keys:
                    session_state.ts_compute_selected_staged_key = staged_keys[0]
                select_box_index = staged_keys.index(session_state.ts_compute_selected_staged_key)
                selected_in_selectbox = st.selectbox(
                    "从暂存区选择数据集:", 
                    options=staged_keys, 
                    index=select_box_index, 
                    key="ts_compute_selectbox_staged_key"
                )
                if selected_in_selectbox != session_state.ts_compute_selected_staged_key:
                    session_state.ts_compute_selected_staged_key = selected_in_selectbox
                    session_state.ts_compute_data = None
                    session_state.ts_compute_original_data = None
                    session_state.ts_compute_file_name = None
                    st.rerun()
                key_to_operate_on = session_state.ts_compute_selected_staged_key
                if key_to_operate_on:
                    is_data_already_loaded = (
                        session_state.get('ts_compute_data') is not None and 
                        session_state.get('ts_compute_file_name') == f"来自暂存区: {key_to_operate_on}"
                    )
                    if is_data_already_loaded:
                        _display_data_overview_for_compute(st, session_state.ts_compute_data, key_to_operate_on)
                    else:
                        if st.button(f"加载来自暂存区的 '{key_to_operate_on}'", key=f"load_staged_{key_to_operate_on}_button"):
                            with st.spinner(f"正在从暂存区加载 '{key_to_operate_on}'..."):
                                df_entry = staged_data.get(key_to_operate_on)
                                if df_entry and 'df' in df_entry and isinstance(df_entry['df'], pd.DataFrame):
                                    df_loaded = df_entry['df'].copy()
                                    if not df_loaded.empty:
                                        try: 
                                            if not isinstance(df_loaded.index, pd.DatetimeIndex):
                                                if len(df_loaded.columns) > 0:
                                                    time_col_name = df_loaded.columns[0]
                                                    if df_loaded.index.name != time_col_name or not isinstance(df_loaded.index, pd.RangeIndex):
                                                        df_loaded[time_col_name] = pd.to_datetime(df_loaded[time_col_name], errors='coerce')
                                                        df_loaded = df_loaded.set_index(time_col_name)
                                                        df_loaded = df_loaded[df_loaded.index.notna()]
                                                else: 
                                                    df_loaded.index = pd.to_datetime(df_loaded.index, errors='coerce')
                                                    df_loaded = df_loaded[df_loaded.index.notna()]                                         
                                            if df_loaded.empty: 
                                               st.error(f"暂存数据 '{key_to_operate_on}' 在时间索引处理后为空。")
                                               raise ValueError("Data empty after time index processing")
                                            session_state.ts_compute_data = df_loaded
                                            session_state.ts_compute_original_data = df_loaded.copy()
                                            session_state.ts_compute_file_name = f"来自暂存区: {key_to_operate_on}"
                                            st.success(f"成功从暂存区加载 '{key_to_operate_on}'。")
                                            st.rerun() 
                                        except (ValueError, TypeError, Exception) as e_inner_proc: 
                                            st.error(f"处理暂存数据 '{key_to_operate_on}' 时出错: {e_inner_proc}")
                                            session_state.ts_compute_data = None
                                            session_state.ts_compute_original_data = None
                                            session_state.ts_compute_file_name = None
                                    else: 
                                        st.error(f"从暂存区选择的数据 '{key_to_operate_on}' 为空。")
                                        session_state.ts_compute_data = None
                                        session_state.ts_compute_original_data = None
                                        session_state.ts_compute_file_name = None
                                else: 
                                    st.error(f"在暂存区中未找到 '{key_to_operate_on}' 的有效DataFrame数据。")
                                    session_state.ts_compute_data = None
                                    session_state.ts_compute_original_data = None
                                    session_state.ts_compute_file_name = None
    
    # After data source selection and potential loading,
    # The main tab (time_series_compute_tab.py) will check session_state.ts_compute_data
    # and then call other UI components like display_variable_calculations_section etc. 