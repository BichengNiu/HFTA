import streamlit as st
import pandas as pd
import io
import numpy as np # Added for parsing
import re # Added for regex operations
from datetime import date # 新增导入 date

# --- 新增：导入状态管理器 ---
try:
    import sys
    import os
    # 获取当前文件的目录
    current_dir = os.path.dirname(__file__)
    # 计算到dashboard根目录的路径
    dashboard_root = os.path.abspath(os.path.join(current_dir, '..', '..', '..'))
    if dashboard_root not in sys.path:
        sys.path.insert(0, dashboard_root)

    from core.state_manager import StateManager
    from core.compat import CompatibilityAdapter
    from core.state_keys import StateKeys
    TOOLS_STATE_MANAGER_AVAILABLE = True
    print("[Time Series Clean] State manager modules imported successfully")
except ImportError as e:
    TOOLS_STATE_MANAGER_AVAILABLE = False
    print(f"[Time Series Clean] Warning: State manager not available, using legacy state management: {e}")
# --- 结束新增 ---

from .utils.parsers import parse_time_column, parse_indices
from .utils.data_processing import (
    load_and_preprocess_data, check_processed_header_duplicates,
    apply_rename_rules, attempt_auto_datetime_conversion
)
from .utils.time_analysis import (
    identify_time_column, calculate_time_series_info, generate_final_data # calculate_time_series_info might be unused directly
)
# from .utils.merge_operations import (
#     perform_vertical_merge, perform_horizontal_merge
# ) # <<< 移除此导入块
from .utils.missing_data import (
    analyze_missing_data, generate_missing_data_plot, handle_missing_values
)
from .utils.stats_utils import calculate_descriptive_stats

# Import new UI components
from .ui_components.file_io_ui import _display_file_handling_and_sheet_selection
from .ui_components.raw_data_ui import _display_raw_data_preview_and_controls
from .ui_components.processed_data_ui import display_processed_data_and_renaming
from .ui_components.time_column_ui import display_time_column_settings
from .ui_components.missing_data_ui import display_missing_data_analysis_and_handling
# from .ui_components.descriptive_stats_ui import display_descriptive_statistics
# from .ui_components.merge_data_ui import display_merge_operations # <<< 注释掉导入
from .ui_components.final_data_ui import display_final_data_generation_and_controls
from .ui_components.sidebar_ui import display_staged_data_sidebar # Added import
from .ui_components.staging_ui import display_data_staging_controls # <<< 新增导入

# --- 工具状态管理辅助函数 ---
def get_tools_state_manager_instance():
    """获取状态管理器实例（工具模块专用）"""
    if TOOLS_STATE_MANAGER_AVAILABLE and hasattr(st.session_state, 'state_manager_initialized'):
        try:
            state_manager = StateManager(st.session_state)
            compat_adapter = CompatibilityAdapter(st.session_state)
            return state_manager, compat_adapter
        except Exception as e:
            print(f"[Time Series Clean] Error getting state manager: {e}")
            return None, None
    return None, None


def get_tools_state(key, default=None):
    """获取工具状态值（兼容新旧状态管理）"""
    state_manager, compat_adapter = get_tools_state_manager_instance()

    if compat_adapter is not None:
        return compat_adapter.get_value(key, default)
    else:
        # 回退到传统方式
        return st.session_state.get(key, default)


def set_tools_state(key, value):
    """设置工具状态值（兼容新旧状态管理）"""
    state_manager, compat_adapter = get_tools_state_manager_instance()

    if compat_adapter is not None:
        compat_adapter.set_value(key, value, use_new_key=True)
    else:
        # 回退到传统方式
        st.session_state[key] = value


# --- <<< 新增：处理重命名规则的回调函数 >>> ---
def add_rename_rule():
    """Callback function to add a rename rule to session state."""
    selected_orig = st.session_state.get('rename_select_orig')
    input_new = st.session_state.get('rename_input_new', '').strip()

    if not selected_orig:
        st.warning("请先选择一个要重命名的原始列。")
        return
    if not input_new:
        st.warning("请输入新列名。")
        return

    # 初始化规则列表（兼容新旧状态管理）
    rename_rules = get_tools_state('ts_tool_rename_rules', [])
    if not rename_rules:
        rename_rules = []

    # --- 验证 ---
    # 1. 检查新名称是否与现有规则中的新名称冲突
    for rule in rename_rules:
        if rule['new'] == input_new:
            st.warning(f"错误：新列名 '{input_new}' 与另一条规则中的新名称冲突。")
            return

    # 2. 检查新名称是否与未被重命名的现有列冲突
    current_data = get_tools_state('ts_tool_data_processed')
    if current_data is not None:
        current_columns = current_data.columns
        originals_in_rules = {rule['original'] for rule in rename_rules}
        existing_columns_not_in_rules = [col for col in current_columns if col not in originals_in_rules and col != selected_orig]
        if input_new in [str(c) for c in existing_columns_not_in_rules]:
             st.warning(f"错误：新列名 '{input_new}' 与一个现有的、未被重命名的列冲突。")
             return

    # --- 添加规则 ---
    rename_rules.append({'original': selected_orig, 'new': input_new})
    set_tools_state('ts_tool_rename_rules', rename_rules)

    # Clear the input field for the next rule (widget state might handle this on rerun)
    # st.session_state.rename_input_new = "" # Might not be necessary
    print(f"Added rename rule: {selected_orig} -> {input_new}") # Debug print

# --- <<< 结束新增 >>> ---

def display_time_series_tool_tab(st, session_state):
    """Displays the Time Series Calculation Tool tab."""

    # --- 工具状态初始化（兼容新旧状态管理） ---
    # 基础文件和数据状态
    if get_tools_state('ts_tool_uploaded_file_name') is None: set_tools_state('ts_tool_uploaded_file_name', None)
    if get_tools_state('ts_tool_uploaded_file_content') is None: set_tools_state('ts_tool_uploaded_file_content', None)
    if get_tools_state('ts_tool_data_raw') is None: set_tools_state('ts_tool_data_raw', None)
    if get_tools_state('ts_tool_rows_to_skip_str') is None: set_tools_state('ts_tool_rows_to_skip_str', "")
    if get_tools_state('ts_tool_header_row_str') is None: set_tools_state('ts_tool_header_row_str', "")
    if get_tools_state('ts_tool_data_processed') is None: set_tools_state('ts_tool_data_processed', None)
    if get_tools_state('ts_tool_processing_applied') is None: set_tools_state('ts_tool_processing_applied', False)
    if get_tools_state('ts_tool_preview_rows') is None: set_tools_state('ts_tool_preview_rows', 5)
    if get_tools_state('ts_tool_cols_to_keep') is None: set_tools_state('ts_tool_cols_to_keep', [])
    if get_tools_state('ts_tool_data_final') is None: set_tools_state('ts_tool_data_final', None)
    if get_tools_state('ts_tool_error_msg') is None: set_tools_state('ts_tool_error_msg', None)
    if get_tools_state('ts_tool_manual_time_col') is None: set_tools_state('ts_tool_manual_time_col', "(自动识别)")
    if get_tools_state('processed_header_duplicates') is None: set_tools_state('processed_header_duplicates', {})
    if get_tools_state('ts_tool_auto_remove_duplicates') is None: set_tools_state('ts_tool_auto_remove_duplicates', False)
    if get_tools_state('ts_tool_manual_frequency') is None: set_tools_state('ts_tool_manual_frequency', "自动")
    if get_tools_state('ts_tool_rename_rules') is None: set_tools_state('ts_tool_rename_rules', [])
    # 扩展状态变量（兼容新旧状态管理）
    if get_tools_state('ts_tool_filter_start_date') is None: set_tools_state('ts_tool_filter_start_date', None)
    if get_tools_state('ts_tool_filter_end_date') is None: set_tools_state('ts_tool_filter_end_date', None)
    if get_tools_state('ts_tool_merge_file_name') is None: set_tools_state('ts_tool_merge_file_name', None)
    if get_tools_state('ts_tool_merge_file_content') is None: set_tools_state('ts_tool_merge_file_content', None)
    if get_tools_state('ts_tool_merged_data') is None: set_tools_state('ts_tool_merged_data', None)
    if get_tools_state('ts_tool_merge_error') is None: set_tools_state('ts_tool_merge_error', None)
    if get_tools_state('ts_tool_time_col_info') is None:
        set_tools_state('ts_tool_time_col_info', {'name': None, 'parsed_series': None})
    if get_tools_state('ts_tool_horizontally_merged_data') is None: set_tools_state('ts_tool_horizontally_merged_data', None)
    if get_tools_state('ts_tool_horizontal_merge_error') is None: set_tools_state('ts_tool_horizontal_merge_error', None)
    if get_tools_state('ts_tool_last_operation_type') is None: set_tools_state('ts_tool_last_operation_type', None)
    if get_tools_state('ts_tool_complete_time_index') is None: set_tools_state('ts_tool_complete_time_index', False)
    if get_tools_state('ts_tool_completion_message') is None: set_tools_state('ts_tool_completion_message', None)
    if get_tools_state('ts_tool_frequency_completion_applied_flag') is None: set_tools_state('ts_tool_frequency_completion_applied_flag', False)
    # 工作表选择状态
    if get_tools_state('ts_tool_sheet_names') is None: set_tools_state('ts_tool_sheet_names', None)
    if get_tools_state('ts_tool_selected_sheet_name') is None: set_tools_state('ts_tool_selected_sheet_name', None)
    # 暂存区状态
    if get_tools_state('staged_data') is None: set_tools_state('staged_data', {})

    # --- 定义工作表改变时的回调函数 (移到这里) ---
    def handle_sheet_change_callback():
        dynamic_key = f"ts_tool_selected_sheet_name_selector_{session_state.get('data_import_reset_counter', 0)}" # Define dynamic key
        print(f"[TSC_TAB] Sheet changed to: {session_state.get(dynamic_key)}")
        # 将选择器的值同步到 ts_tool_selected_sheet_name 以便旧逻辑使用
        session_state.ts_tool_selected_sheet_name = session_state.get(dynamic_key)

        # Reset downstream states when sheet changes
        session_state.ts_tool_data_raw = None 
        session_state.ts_tool_rows_to_skip_str = ""
        session_state.ts_tool_header_row_str = ""
        session_state.ts_tool_data_processed = None
        session_state.ts_tool_processing_applied = False
        session_state.ts_tool_cols_to_keep = []
        session_state.ts_tool_data_final = None
        session_state.ts_tool_manual_time_col = "(自动识别)"
        session_state.processed_header_duplicates = {}
        session_state.ts_tool_auto_remove_duplicates = False 
        session_state.ts_tool_manual_frequency = "自动" 
        session_state.ts_tool_rename_rules = []
        session_state.ts_tool_filter_start_date = None
        session_state.ts_tool_filter_end_date = None
        session_state.ts_tool_merged_data = None
        session_state.ts_tool_merge_error = None
        session_state.ts_tool_time_col_info = {'name': None, 'parsed_series': None}
        session_state.ts_tool_horizontally_merged_data = None
        session_state.ts_tool_horizontal_merge_error = None
        session_state.ts_tool_last_operation_type = None
        session_state.ts_tool_complete_time_index = False
        session_state.ts_tool_completion_message = None
        # session_state.ts_tool_error_msg = None # 错误消息由主布局处理
        st.rerun() # ADDED: Force a rerun after sheet change and state reset

    # --- 创建两列布局 ---
    col_left, col_right = st.columns([2, 3]) # 左2右3比例，可调整，例如 [1,2] 或 [1,1.5]

    with col_left:
        _display_file_handling_and_sheet_selection(st, session_state) # 现在只包含文件上传
        st.markdown("---")  # 添加分隔线
    
    with col_right:
        # --- 工作表选择UI 和 预处理控件 合并到一个带边框的容器中 ---
        with st.container(border=False):
            if session_state.get('ts_tool_sheet_names'):
                if len(session_state.ts_tool_sheet_names) > 1:
                    default_idx = 0
                    selected_sheet_name_val = session_state.get('ts_tool_selected_sheet_name')
                    sheet_names_val = session_state.get('ts_tool_sheet_names')
                    if sheet_names_val and selected_sheet_name_val and selected_sheet_name_val in sheet_names_val:
                        try:
                            default_idx = sheet_names_val.index(selected_sheet_name_val)
                        except ValueError:
                            default_idx = 0 

                    st.selectbox(
                        label="**选择要处理的工作表 (Sheet):**",
                        options=sheet_names_val,
                        key=f"ts_tool_selected_sheet_name_selector_{session_state.get('data_import_reset_counter', 0)}",
                        on_change=handle_sheet_change_callback,
                        index=default_idx
                    )

                    # Use dynamic key here as well
                    dynamic_key_for_check = f"ts_tool_selected_sheet_name_selector_{session_state.get('data_import_reset_counter', 0)}"
                    if session_state.get(dynamic_key_for_check) != session_state.get('ts_tool_selected_sheet_name'):
                        session_state.ts_tool_selected_sheet_name = session_state.get(dynamic_key_for_check)

                elif len(session_state.ts_tool_sheet_names) == 1:
                    st.caption(f"文件只有一个工作表: {session_state.ts_tool_selected_sheet_name}") 
            
            # --- 预处理控件 (现在也在此容器内) ---
            if session_state.get('ts_tool_uploaded_file_content') and session_state.get('ts_tool_selected_sheet_name'):
                _display_raw_data_preview_and_controls(st, session_state) 
            # else: # 这个else分支的提示信息可能不再需要，或者需要调整位置
                # st.info("请先上传文件并在左侧选择工作表（如果文件包含多个工作表）。")

    # --- 中央错误消息显示区 --- (移出列布局)
    if session_state.get('ts_tool_error_msg'):
        st.error(session_state.ts_tool_error_msg)
        session_state.ts_tool_error_msg = None # Clear after displaying
    
    # Call the UI function for processed data display and renaming
    # 这个组件依赖于 ts_tool_data_processed，所以只有在预处理完成后才应该有内容
    if session_state.get('ts_tool_processing_applied') and session_state.get('ts_tool_data_processed') is not None:
        display_processed_data_and_renaming(st, session_state, add_rename_rule_callback=add_rename_rule)
    elif session_state.get('ts_tool_data_raw') is not None and not session_state.get('ts_tool_processing_applied'):
        st.info('请在上方输入跳过行和表头行，数据将自动应用预处理。')

    # --- Components are now displayed directly ---
    # Call the UI function for time column settings
    display_time_column_settings(st, session_state)

    # <<< 新增调试打印：检查 display_time_column_settings 调用后的 session_state >>>
    print("DEBUG_CLEAN_TAB_PY: After display_time_column_settings call.")
    if 'ts_tool_alignment_report_items' in session_state:
        print("DEBUG_CLEAN_TAB_PY: 'ts_tool_alignment_report_items' FOUND in session_state.")
        # 进一步检查我们关心的调试列
        target_debug_col_clean_tab = '中国:生产率:焦炉:国内独立焦化厂(230家)' # 与之前调试一致
        found_item_for_debug_col_clean_tab = False
        report_items_in_clean_tab = session_state['ts_tool_alignment_report_items']
        if isinstance(report_items_in_clean_tab, list):
            for item_idx, item_data in enumerate(report_items_in_clean_tab):
                if isinstance(item_data, dict) and item_data.get('column') == target_debug_col_clean_tab:
                    # 仅关注汇总行，它们通常不包含 'event_type' 或者 'event_type' 不是特定事件
                    if 'event_type' not in item_data or item_data.get('event_type') is None or not isinstance(item_data.get('event_type'), str) or "summary" in item_data.get('event_type','').lower() or item_data.get('event_type') == item_data.get('column'): # 假设汇总行的 event_type 可能为 None 或列名本身或包含 'summary'
                        original_val = item_data.get('original_non_nan', 'MISSING_original_non_nan')
                        aligned_val = item_data.get('aligned_non_nan', 'MISSING_aligned_non_nan')
                        print(f"DEBUG_CLEAN_TAB_PY [{target_debug_col_clean_tab} - Item Index {item_idx} - Summary]: original_non_nan={original_val} (type: {type(original_val)}), aligned_non_nan={aligned_val} (type: {type(aligned_val)})")
                        found_item_for_debug_col_clean_tab = True
                        break # 找到汇总行后即可停止
            if not found_item_for_debug_col_clean_tab:
                print(f"DEBUG_CLEAN_TAB_PY: Summary item for '{target_debug_col_clean_tab}' NOT found in report_items_in_clean_tab, or report_items_in_clean_tab is not a list of dicts.")
        else:
            print("DEBUG_CLEAN_TAB_PY: 'ts_tool_alignment_report_items' is not a list.")

    else:
        print("DEBUG_CLEAN_TAB_PY: 'ts_tool_alignment_report_items' NOT FOUND in session_state at this point.")

    # Call the UI function for missing data analysis and handling
    display_missing_data_analysis_and_handling(st, session_state)

    # Call the UI function for descriptive statistics
    # display_descriptive_statistics(st, session_state)

    # Call the UI function for final data generation, display, and staging
    # display_final_data_generation_and_controls(st, session_state) 

    # Call the UI function for data merging operations
    # display_merge_operations(st, session_state) # <<< 注释掉调用

    # --- <<< 新增：调用新的数据暂存UI >>> ---
    display_data_staging_controls(st, session_state) # <<< 新增调用

    # Final message if no file is uploaded yet or processing not started
    # This check should be at the very end of the main content area logic
    # Check if an uploaded_file object exists via the file_uploader's key
    # This is slightly different from checking session_state.ts_tool_uploaded_file_content directly
    # as the widget itself might hold the file object reference.
    # However, our current file_io_ui handles the uploaded_file object internally.
    # So, we rely on session_state here.

    if session_state.ts_tool_uploaded_file_content is None and not session_state.get('staged_data'):
        st.info("请在左上方上传一个 Excel 或 CSV 文件以开始。")


    # Any other trailing UI elements for the main tab can go here.
    # For now, we assume the sidebar is the last part of this function's UI definition.
