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
    if 'ts_compute_uploader_state_reset_key' not in st.session_state:
        st.session_state.ts_compute_uploader_state_reset_key = 0
    st.session_state.ts_compute_uploader_state_reset_key += 1
    if st.session_state.get("ts_compute_data_source_radio") != "从暂存区选择":
      st.session_state.ts_compute_selected_staged_key = None

def _display_data_overview_for_compute(st, df, source_name):
    st.subheader(f"数据概览 ({source_name})")
    rows, cols = df.shape
    info_cols = st.columns(2)
    with info_cols[0]:
        st.metric("观测值数量 (行)", rows)
    with info_cols[1]:
        st.metric("变量数量 (列)", cols)

    time_info_str = "时间信息：\n"
    freq_display = "无法确定频率"
    if isinstance(df.index, pd.DatetimeIndex):
        time_info_str += f"  - 开始时间: {df.index.min()}\n"
        time_info_str += f"  - 结束时间: {df.index.max()}\n"
        try:
            inferred_freq = pd.infer_freq(df.index)
            if inferred_freq:
                freq_map = {
                    'A': "年度", 'AS': "年初开始的年度", 'Q': "季度", 'QS': "季初开始的季度",
                    'M': "月度", 'MS': "月初开始的月度", 'W': "周度", 'D': "日度", 'B': "工作日",
                    'H': "小时", 'T': "分钟", 'S': "秒", 'L': "毫秒", 'U': "微秒"
                }
                freq_prefix_match = re.match(r"^[A-Za-z]+", inferred_freq)
                if freq_prefix_match and freq_prefix_match.group(0) in freq_map:
                    freq_display = f"{freq_map[freq_prefix_match.group(0)]} ({inferred_freq})"
                else:
                    freq_display = f"推断频率: {inferred_freq}"
            else:
                if len(df.index) > 1:
                    time_diffs = df.index.to_series().diff().dropna()
                    if not time_diffs.empty:
                        median_diff = time_diffs.median()
                        common_diffs = time_diffs.value_counts().head(3)
                        freq_display = f"不规则时间序列 (中位时间差: {median_diff})\n"
                        freq_display += "  - 主要时间间隔 (前3): \n"
                        for diff_val, count_val in common_diffs.items():
                            freq_display += f"    - {diff_val} (出现 {count_val} 次)\n"
                        is_likely_monthly = False
                        if not common_diffs.empty:
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
        except Exception as e_freq:
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
        on_change=on_data_source_change
    )

    if session_state.ts_compute_data_source_radio == "上传新文件":
        if 'ts_compute_uploader_state_reset_key' not in session_state:
            session_state.ts_compute_uploader_state_reset_key = 0
        uploaded_file = st.file_uploader(
            "上传 CSV 或 Excel 文件:", type=["csv", "xlsx"],
            key=f"ts_compute_uploader_{session_state.ts_compute_uploader_state_reset_key}",
            help="上传包含时间序列数据的文件。建议第一列是日期/时间信息。"
        )
        if uploaded_file is not None:
            is_new_upload_action = (uploaded_file.name != session_state.get('ts_compute_file_name') or session_state.ts_compute_data is None)
            if is_new_upload_action:
                st.info(f"检测到文件: {uploaded_file.name}。将尝试使用第一列作为时间索引加载...")
                with st.spinner(f"正在加载文件: {uploaded_file.name}..."):
                    processed_data = process_uploaded_data(uploaded_file, time_index_col=0)
                    if processed_data is not None and not processed_data.empty:
                        session_state.ts_compute_data = processed_data
                        session_state.ts_compute_original_data = processed_data.copy()
                        session_state.ts_compute_file_name = uploaded_file.name
                        st.success(f"文件 '{uploaded_file.name}' 加载成功。")
                        _display_data_overview_for_compute(st, processed_data, uploaded_file.name)
                    elif processed_data is not None and processed_data.empty:
                        st.error(f"文件 '{uploaded_file.name}' 加载后为空。请检查文件内容。")
                        session_state.ts_compute_data = None
                        session_state.ts_compute_original_data = None
                        session_state.ts_compute_file_name = None
                    else:
                        st.error(f"无法加载或处理文件 '{uploaded_file.name}'。请检查文件格式、内容，并确保其包含有效的时间序列数据（建议时间列在第一列）。")
                        session_state.ts_compute_data = None
                        session_state.ts_compute_original_data = None
                        session_state.ts_compute_file_name = None
            elif session_state.ts_compute_data is not None:
                 _display_data_overview_for_compute(st, session_state.ts_compute_data, session_state.ts_compute_file_name)

    elif session_state.ts_compute_data_source_radio == "从暂存区选择":
        staged_data = session_state.get('staged_data', {})
        if not staged_data:
            st.warning("暂存区为空。请先在数据清洗或其他模块中处理并暂存数据，或选择上传新文件。")
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
                current_selected_staged_key = session_state.get('ts_compute_selected_staged_key')
                if not current_selected_staged_key or current_selected_staged_key not in staged_keys:
                    current_selected_staged_key = staged_keys[0]
                select_box_index = staged_keys.index(current_selected_staged_key)
                # Forcing st.selectbox to be a single line for Linter diagnosis
                new_selected_key = st.selectbox("从暂存区选择数据集:", options=staged_keys, index=select_box_index, key="ts_compute_selectbox_staged_key")

                if new_selected_key != current_selected_staged_key:
                    session_state.ts_compute_selected_staged_key = new_selected_key
                    session_state.ts_compute_data = None
                    session_state.ts_compute_original_data = None
                    session_state.ts_compute_file_name = None
                    st.rerun()
                if session_state.ts_compute_selected_staged_key:
                    selected_key_to_load = session_state.ts_compute_selected_staged_key
                    if st.button(f"加载来自暂存区的 '{selected_key_to_load}'", key="load_staged_data_compute_button"):
                        with st.spinner(f"正在从暂存区加载 '{selected_key_to_load}'..."):
                            df_entry = staged_data.get(selected_key_to_load)
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
                                                    df_loaded = df_loaded.dropna(subset=[df_loaded.index.name])
                                            else:
                                                st.error(f"暂存数据 '{selected_key_to_load}' 没有列可以作为时间索引。")
                                                raise ValueError("Staged data has no columns for DatetimeIndex")
                                        if df_loaded.empty: # after potential NaT drop
                                           st.error(f"暂存数据 '{selected_key_to_load}' 在时间索引处理后为空。")
                                           raise ValueError("Data empty after time index processing")
                                        session_state.ts_compute_data = df_loaded
                                        session_state.ts_compute_original_data = df_loaded.copy()
                                        session_state.ts_compute_file_name = f"来自暂存区: {selected_key_to_load}"
                                        st.success(f"成功从暂存区加载 '{selected_key_to_load}'。")
                                        _display_data_overview_for_compute(st, df_loaded, selected_key_to_load)
                                    except Exception as e_staged_load:
                                        st.error(f"处理或加载暂存数据 '{selected_key_to_load}' 时出错: {e_staged_load}")
                                        session_state.ts_compute_data = None
                                        session_state.ts_compute_original_data = None
                                        session_state.ts_compute_file_name = None
                                else:
                                    st.error(f"从暂存区选择的数据 '{selected_key_to_load}' 为空。")
                                    session_state.ts_compute_data = None
                                    session_state.ts_compute_original_data = None
                                    session_state.ts_compute_file_name = None
                            else:
                                st.error(f"在暂存区中未找到 '{selected_key_to_load}' 的有效DataFrame数据。")
                                session_state.ts_compute_data = None
                                session_state.ts_compute_original_data = None
                                session_state.ts_compute_file_name = None
                    elif session_state.ts_compute_data is not None and session_state.ts_compute_file_name == f"来自暂存区: {selected_key_to_load}":
                        _display_data_overview_for_compute(st, session_state.ts_compute_data, selected_key_to_load) 