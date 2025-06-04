import streamlit as st
import pandas as pd
import numpy as np
import os
import sys
from datetime import datetime, timedelta, time
from streamlit import session_state
from collections import defaultdict
import threading
import traceback
from typing import Dict, List, Optional, Union

# 添加当前目录和父目录到路径
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)  # DFM目录
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

# 强制添加data_prep目录到路径
data_prep_dir = os.path.join(parent_dir, 'data_prep')
if data_prep_dir not in sys.path:
    sys.path.insert(0, data_prep_dir)

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

# --- Module Import Error Handling --- 
_CONFIG_MODULE = None
_DATA_PREPARATION_MODULE = None
_TRAIN_UI_IMPORT_ERROR_MESSAGE = None # Stores combined error messages

# 1. 尝试导入配置模块
try:
    if CONFIG_AVAILABLE:
        # 使用统一配置
        class ConfigWrapper:
            TYPE_MAPPING_SHEET = DataDefaults.TYPE_MAPPING_SHEET
            TARGET_VARIABLE = DataDefaults.TARGET_VARIABLE
            INDICATOR_COLUMN_NAME_IN_EXCEL = DataDefaults.INDICATOR_COLUMN
            INDUSTRY_COLUMN_NAME_IN_EXCEL = DataDefaults.INDUSTRY_COLUMN
            TYPE_COLUMN_NAME_IN_EXCEL = DataDefaults.TYPE_COLUMN
            
            # 输出目录设置
            DFM_OUTPUT_BASE = os.path.join(parent_dir, 'outputs')
            DFM_TRAIN_OUTPUT_DIR = DFM_OUTPUT_BASE
            DFM_DATA_OUTPUT_DIR = os.path.join(DFM_OUTPUT_BASE, 'data')
            DFM_MODEL_OUTPUT_DIR = os.path.join(DFM_OUTPUT_BASE, 'models')
            DFM_PLOTS_OUTPUT_DIR = os.path.join(DFM_OUTPUT_BASE, 'plots')
            DFM_REPORTS_OUTPUT_DIR = os.path.join(DFM_OUTPUT_BASE, 'reports')
        
        _CONFIG_MODULE = ConfigWrapper()
    else:
        # 创建基于项目结构的配置类作为后备
        class TrainModelConfig:
            # 基于项目结构的路径设置
            PROJECT_ROOT = os.path.abspath(os.path.join(current_dir, '..', '..', '..'))
            
            # UI默认配置值（使用配置模块或硬编码作为最后的后备）
            TYPE_MAPPING_SHEET = DataDefaults.TYPE_MAPPING_SHEET if 'DataDefaults' in globals() else '指标体系'
            TARGET_VARIABLE = DataDefaults.TARGET_VARIABLE if 'DataDefaults' in globals() else '规模以上工业增加值:当月同比'
            INDICATOR_COLUMN_NAME_IN_EXCEL = DataDefaults.INDICATOR_COLUMN if 'DataDefaults' in globals() else '高频指标'
            INDUSTRY_COLUMN_NAME_IN_EXCEL = DataDefaults.INDUSTRY_COLUMN if 'DataDefaults' in globals() else '行业'
            TYPE_COLUMN_NAME_IN_EXCEL = DataDefaults.TYPE_COLUMN if 'DataDefaults' in globals() else '类型'
            
            # 输出目录设置
            DFM_OUTPUT_BASE = os.path.join(parent_dir, 'outputs')
            DFM_TRAIN_OUTPUT_DIR = DFM_OUTPUT_BASE
            DFM_DATA_OUTPUT_DIR = os.path.join(DFM_OUTPUT_BASE, 'data')
            DFM_MODEL_OUTPUT_DIR = os.path.join(DFM_OUTPUT_BASE, 'models')
            DFM_PLOTS_OUTPUT_DIR = os.path.join(DFM_OUTPUT_BASE, 'plots')
            DFM_REPORTS_OUTPUT_DIR = os.path.join(DFM_OUTPUT_BASE, 'reports')
        
        _CONFIG_MODULE = TrainModelConfig()

except Exception as e_config:
    error_msg_config = (
        f"Failed to create configuration: {e_config}. "
        "Using fallback configuration. Functionality may be limited."
    )
    if _TRAIN_UI_IMPORT_ERROR_MESSAGE:
        _TRAIN_UI_IMPORT_ERROR_MESSAGE += f"\n{error_msg_config}"
    else:
        _TRAIN_UI_IMPORT_ERROR_MESSAGE = error_msg_config
    
    class FallbackConfig:
        TYPE_MAPPING_SHEET = DataDefaults.TYPE_MAPPING_SHEET if 'DataDefaults' in globals() else '指标体系'
        TARGET_VARIABLE = DataDefaults.TARGET_VARIABLE if 'DataDefaults' in globals() else '规模以上工业增加值:当月同比'
        INDICATOR_COLUMN_NAME_IN_EXCEL = DataDefaults.INDICATOR_COLUMN if 'DataDefaults' in globals() else '高频指标'
        INDUSTRY_COLUMN_NAME_IN_EXCEL = DataDefaults.INDUSTRY_COLUMN if 'DataDefaults' in globals() else '行业'
        TYPE_COLUMN_NAME_IN_EXCEL = DataDefaults.TYPE_COLUMN if 'DataDefaults' in globals() else '类型'
    _CONFIG_MODULE = FallbackConfig()

# 2. 尝试导入数据预处理模块
try:
    # 从data_prep目录导入
    data_prep_dir = os.path.join(parent_dir, 'data_prep')
    if data_prep_dir not in sys.path:
        sys.path.insert(0, data_prep_dir)
    
    from data_preparation import load_mappings
    
    class DataPreparationWrapper:
        @staticmethod
        def load_mappings(*args, **kwargs):
            return load_mappings(*args, **kwargs)
    
    _DATA_PREPARATION_MODULE = DataPreparationWrapper()

except ImportError as e_dp:
    error_msg_dp = (
        f"Failed to import data_preparation: {e_dp}. "
        "Using mock data preparation. Functionality may be limited."
    )
    if _TRAIN_UI_IMPORT_ERROR_MESSAGE:
        _TRAIN_UI_IMPORT_ERROR_MESSAGE += f"\n{error_msg_dp}"
    else:
        _TRAIN_UI_IMPORT_ERROR_MESSAGE = error_msg_dp
    
    class MockDataPreparation:
        @staticmethod
        def load_mappings(excel_path, sheet_name, indicator_col, type_col, industry_col):
            try:
                # 尝试直接读取Excel文件作为fallback
                if os.path.exists(excel_path):
                    df = pd.read_excel(excel_path, sheet_name=sheet_name)
                    if indicator_col in df.columns and industry_col in df.columns:
                        var_industry_map = dict(zip(df[indicator_col].fillna(''), df[industry_col].fillna('')))
                        var_type_map = {}
                        if type_col in df.columns:
                            var_type_map = dict(zip(df[indicator_col].fillna(''), df[type_col].fillna('')))
                        return var_type_map, var_industry_map
            except Exception as e:
                pass  # 静默处理错误
            return {}, {} # Return empty mappings if all fails
    
    _DATA_PREPARATION_MODULE = MockDataPreparation()

# Make the config and data_preparation (real or mock) available for the rest of the module
config = _CONFIG_MODULE
data_preparation = _DATA_PREPARATION_MODULE

# 3. 导入DFM训练脚本
try:
    import tune_dfm
    
    if hasattr(tune_dfm, 'train_and_save_dfm_results'):
        from tune_dfm import train_and_save_dfm_results
    else:
        raise ImportError("train_and_save_dfm_results function not found in tune_dfm module")
        
except ImportError as e_tune_dfm:
    error_msg_tune_dfm = (
        f"Failed to import 'tune_dfm.train_and_save_dfm_results': {e_tune_dfm}. "
        "Actual model training will not be possible."
    )
    if _TRAIN_UI_IMPORT_ERROR_MESSAGE:
        _TRAIN_UI_IMPORT_ERROR_MESSAGE += f"\n{error_msg_tune_dfm}"
    else:
        _TRAIN_UI_IMPORT_ERROR_MESSAGE = error_msg_tune_dfm
    # Define a mock function if import fails
    def train_and_save_dfm_results(*args, **kwargs):
        st.error("Mock train_and_save_dfm_results called due to import error. No training will occur.")
        # Simulate an error or empty result as training didn't happen
        raise RuntimeError("Model training function (train_and_save_dfm_results) is not available due to import error.")

# --- End Module Import Error Handling ---

# --- 辅助函数 ---
def convert_to_datetime(date_input):
    """将日期输入转换为datetime对象"""
    from datetime import date, datetime, time
    
    if date_input is None:
        return None
    if isinstance(date_input, datetime):
        return date_input
    if isinstance(date_input, date):
        # 如果是date对象，转换为datetime对象
        return datetime.combine(date_input, time())
    if isinstance(date_input, str):
        return pd.to_datetime(date_input).to_pydatetime()
    # 如果是其他pandas时间类型，转换为datetime
    if hasattr(date_input, 'to_pydatetime'):
        return date_input.to_pydatetime()
    return date_input

# --- 全局变量存储训练状态 ---
_training_state = {
    'status': '等待开始',
    'log': [],
    'results': None,
    'error': None
}

# --- 辅助函数：训练线程 ---
def _run_training_thread(params, st_instance_ref, log_callback_ref):
    """Helper function to run the training in a separate thread."""
    global _training_state
    
    try:
        # 设置初始状态
        _training_state['status'] = '正在训练...'
        _training_state['log'] = ['[训练开始] 开始DFM模型训练...']
        _training_state['results'] = None
        _training_state['error'] = None
        
        # 线程安全的日志回调函数
        def thread_log_callback(message):
            """完全线程安全的日志回调，避免任何Streamlit上下文调用"""
            try:
                # 忽略所有Streamlit相关的警告和消息
                if not message or not message.strip():
                    return
                    
                # 过滤掉所有不需要的消息类型
                skip_patterns = [
                    '[DEBUG]', '  [DEBUG]', '[TRAIN_LOG_ERROR]', '[UI_SYNC]',
                    'ScriptRunContext', 'Thread \'MainThread\':', 'missing ScriptRunContext',
                    'WARNING streamlit', '成功导入', '尝试导入', 'sys.path前3个',
                    'current_dir:', 'parent_dir:', 'data_prep_dir:', '✓ 成功创建配置模块',
                    '[OK] 成功导入', '模块导入状态总结:', 'DEBUG', 'No runtime found',
                    'Matplotlib Font Setup', 'Thread \'MainThread\': missing ScriptRunContext',
                    'MainThread', 'ScriptRunner', 'streamlit', 'can be ignored when running in bare mode'
                ]
                
                # 检查是否包含需要跳过的模式
                for pattern in skip_patterns:
                    if pattern in message:
                        return
                
                # 清理消息格式
                clean_message = message.replace('[TRAIN] ', '').replace('[TRAIN_LOG] ', '').strip()
                if clean_message and clean_message not in _training_state['log']:
                    # 添加时间戳
                    from datetime import datetime
                    timestamp = datetime.now().strftime('%H:%M:%S')
                    formatted_message = f"[{timestamp}] {clean_message}"
                    _training_state['log'].append(formatted_message)
                    # 只保留最近50条日志，避免内存占用过多
                    if len(_training_state['log']) > 50:
                        _training_state['log'] = _training_state['log'][-50:]
            except Exception:
                # 完全静默处理日志错误，避免干扰训练过程
                pass
        
        # 设置进度回调
        params_copy = params.copy()
        params_copy['progress_callback'] = thread_log_callback
        
        _training_state['log'].append("[训练参数] 开始准备训练数据...")
        thread_log_callback("开始DFM模型训练...")
        
        # 调用训练函数
        saved_files_paths = train_and_save_dfm_results(**params_copy)
        
        # 设置训练完成状态
        _training_state['status'] = "训练完成"
        _training_state['results'] = saved_files_paths
        
        thread_log_callback("✅ 训练成功完成！")
        thread_log_callback(f"📁 生成 {len(saved_files_paths)} 个文件")
        
        # 更新session_state以便UI显示
        if hasattr(st_instance_ref, 'session_state'):
            st_instance_ref.session_state.dfm_training_status = "训练完成"
            st_instance_ref.session_state.dfm_model_results_paths = saved_files_paths
        
    except Exception as e:
        _training_state['status'] = f"训练失败: {str(e)}"
        _training_state['error'] = str(e)
        error_msg = f"❌ 训练失败: {str(e)}"
        thread_log_callback(error_msg)
        
        # 更新session_state
        if hasattr(st_instance_ref, 'session_state'):
            st_instance_ref.session_state.dfm_training_status = f"训练失败: {str(e)}"
    
    # 训练线程结束，不输出到控制台

# --- 辅助函数：加载和解析指标体系 (保持不变) ---
# 🔥 修复：移除缓存装饰器避免 ScriptRunContext 警告
# @st.cache_data 

# 旧的基于本地文件路径的缓存函数已删除，现在使用基于上传文件的函数

@st.cache_data(ttl=3600)  # 缓存1小时，避免重复读取Excel文件
def load_indicator_mappings_from_data_prep(uploaded_excel_file=None, type_mapping_sheet=None):
    """从UI上传的Excel文件中加载行业及指标映射。
    Args:
        uploaded_excel_file: Streamlit上传的文件对象（来自data_prep模块）
        type_mapping_sheet: 指标体系工作表名称
    Returns: 
        unique_industries (list): 唯一行业名称列表。
        industry_to_indicators_map (dict): {'行业名': ['指标列表']}。
        all_indicators_flat (list): 所有映射中出现的指标的扁平化列表。
    """
    # 使用配置的默认值
    if type_mapping_sheet is None:
        if CONFIG_AVAILABLE:
            type_mapping_sheet = DataDefaults.TYPE_MAPPING_SHEET
        else:
            type_mapping_sheet = DataDefaults.TYPE_MAPPING_SHEET if 'DataDefaults' in globals() else '指标体系'
    if uploaded_excel_file is None:
        st.warning("⚠️ 尚未上传Excel数据文件。请先在「数据准备」模块上传Excel文件。")
        return [], {}, []
    
    try:
        import io
        # 重置文件指针
        uploaded_excel_file.seek(0)
        excel_file_like_object = io.BytesIO(uploaded_excel_file.getvalue())
        
        # 使用UI上传的文件而不是本地路径
        if CONFIG_AVAILABLE:
            indicator_col = DataDefaults.INDICATOR_COLUMN
            type_col = DataDefaults.TYPE_COLUMN
            industry_col = DataDefaults.INDUSTRY_COLUMN
        else:
            indicator_col = DataDefaults.INDICATOR_COLUMN if 'DataDefaults' in globals() else '高频指标'
            type_col = DataDefaults.TYPE_COLUMN if 'DataDefaults' in globals() else '类型'
            industry_col = DataDefaults.INDUSTRY_COLUMN if 'DataDefaults' in globals() else '行业'
            
        _, var_industry_map = _cached_load_mappings_silent_from_uploaded_file(
            excel_file_obj=excel_file_like_object,
            sheet_name=type_mapping_sheet,
            indicator_col=indicator_col, 
            type_col=type_col, 
            industry_col=industry_col
        )

        if not var_industry_map:
            st.warning(f"从上传文件的'{type_mapping_sheet}'工作表中未找到有效的行业映射数据。请检查工作表名称和列结构。")
            return [], {}, [] 

        all_indicators = list(var_industry_map.keys()) 
        industry_to_indicators_temp = defaultdict(list)
        for indicator, industry in var_industry_map.items():
            if indicator and industry: 
                industry_to_indicators_temp[str(industry).strip()].append(str(indicator).strip())
        
        industries = sorted(list(industry_to_indicators_temp.keys()))
        industry_to_indicators = {k: sorted(v) for k, v in industry_to_indicators_temp.items()}
        all_indicators = sorted(all_indicators)

        if not industries or not all_indicators:
            st.info(f"从上传文件的'{type_mapping_sheet}'工作表解析后未找到有效的行业或指标。")

        return industries, industry_to_indicators, all_indicators

    except Exception as e:
        st.error(f"从上传的Excel文件加载指标映射时出错：{e}")
        return [], {}, []

@st.cache_data(ttl=3600, show_spinner=False)  # 缓存1小时，不显示spinner
def _cached_load_mappings_silent_from_uploaded_file(excel_file_obj, sheet_name: str, indicator_col: str, type_col: str, industry_col: str):
    """从上传的Excel文件对象静默地加载映射数据"""
    import io
    from contextlib import redirect_stdout, redirect_stderr
    
    # 捕获并静默所有输出
    with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
        try:
            # 重置文件指针
            excel_file_obj.seek(0)
            
            # 直接使用pandas读取上传的文件
            excel_file = pd.ExcelFile(excel_file_obj)
            if sheet_name not in excel_file.sheet_names:
                return {}, {}
                
            indicator_sheet = pd.read_excel(excel_file, sheet_name=sheet_name)
            
            # 清理列名
            indicator_sheet.columns = indicator_sheet.columns.str.strip()
            
            # 检查必需列是否存在
            if indicator_col not in indicator_sheet.columns or industry_col not in indicator_sheet.columns:
                return {}, {}
            
            # 创建行业映射
            import unicodedata
            var_industry_map = {}
            for _, row in indicator_sheet.iterrows():
                indicator_name = row.get(indicator_col)
                industry_name = row.get(industry_col)
                
                if pd.notna(indicator_name) and pd.notna(industry_name):
                    # 标准化指标名称
                    norm_indicator = unicodedata.normalize('NFKC', str(indicator_name)).strip().lower()
                    norm_industry = str(industry_name).strip()
                    if norm_indicator and norm_industry:
                        var_industry_map[norm_indicator] = norm_industry
            
            # 创建类型映射（如果需要）
            var_type_map = {}
            if type_col in indicator_sheet.columns:
                for _, row in indicator_sheet.iterrows():
                    indicator_name = row.get(indicator_col)
                    type_name = row.get(type_col)
                    
                    if pd.notna(indicator_name) and pd.notna(type_name):
                        norm_indicator = unicodedata.normalize('NFKC', str(indicator_name)).strip().lower()
                        norm_type = str(type_name).strip()
                        if norm_indicator and norm_type:
                            var_type_map[norm_indicator] = norm_type
            
            return var_type_map, var_industry_map
            
        except Exception:
            return {}, {}

# --- 新增辅助函数：重置模型参数 --- 
def reset_model_parameters(ss):
    """将模型训练相关的参数重置为初始值。"""
    from datetime import datetime, timedelta
    
    # 时间窗口设置 - 使用配置的默认值
    today = datetime.now().date()
    
    # 使用配置中的默认值
    if CONFIG_AVAILABLE:
        years_back = TrainDefaults.TRAINING_YEARS_BACK
        val_end_year = TrainDefaults.VALIDATION_END_YEAR
        val_end_month = TrainDefaults.VALIDATION_END_MONTH
        val_end_day = TrainDefaults.VALIDATION_END_DAY
    else:
        # 后备硬编码值
        years_back = 5
        val_end_year = 2024
        val_end_month = 12
        val_end_day = 31
    
    # 默认训练期：从配置的年数前开始到1个月前
    ss.dfm_train_start_date_value = datetime(today.year - years_back, today.month, today.day)
    ss.dfm_train_end_date_value = datetime.combine(today - timedelta(days=30), datetime.min.time())
    
    # 默认验证期结束日期：使用配置
    ss.dfm_oos_validation_end_date_value = datetime(val_end_year, val_end_month, val_end_day)

    # 因子参数设置 - 使用配置
    if CONFIG_AVAILABLE:
        default_factor_strategy_options = list(UIDefaults.FACTOR_SELECTION_STRATEGY_OPTIONS.keys())
        default_strategy = TrainDefaults.FACTOR_SELECTION_STRATEGY
        max_iter = TrainDefaults.EM_MAX_ITER
        fixed_factors = TrainDefaults.FIXED_NUMBER_OF_FACTORS
        cum_threshold = TrainDefaults.CUM_VARIANCE_THRESHOLD
    else:
        # 后备硬编码值
        default_factor_strategy_options = ['information_criteria', 'fixed_number', 'cumulative_variance']
        default_strategy = 'information_criteria'
        max_iter = TrainDefaults.EM_MAX_ITER if 'TrainDefaults' in globals() else 100
        fixed_factors = TrainDefaults.FIXED_NUMBER_OF_FACTORS if 'TrainDefaults' in globals() else 3
        cum_threshold = 0.8
    
    ss.dfm_factor_selection_strategy_idx = 0
    ss.dfm_factor_selection_strategy_value = default_strategy
    ss.dfm_n_factors_manual_value = ""

    # 因子选择策略相关参数的默认值
    ss.dfm_information_criterion_threshold_value = 0.1
    ss.dfm_fixed_factor_number_value = fixed_factors
    ss.dfm_common_variance_contribution_threshold_value = cum_threshold
    ss.dfm_variance_threshold_value = 0.1

    # 高级参数
    ss.dfm_max_iter = max_iter
    default_em_criterion_options = ['params', 'likelihood']
    ss.dfm_em_convergence_criterion_idx = 0
    ss.dfm_em_convergence_criterion_value = default_em_criterion_options[0]
    
    # 如果有其他通过key直接在UI中设置的session_state变量，也需要在这里重置
    # 例如：ss.ss_dfm_max_iter = max_iter (如果key与dfm_max_iter不同)
    # 但通常我们会用一个统一的dfm_ prefixed key来存储值，UI的key只用于streamlit内部

# 已移除自动日期设置功能，用户完全控制日期输入

def render_dfm_train_model_tab(st_instance, session_state):
    # 确保datetime在函数开头就可用
    from datetime import datetime, time

    # --- NEW: Display import errors if they occurred --- 
    if _TRAIN_UI_IMPORT_ERROR_MESSAGE:
        st_instance.error(f"模块导入错误，部分功能可能受限:\n{_TRAIN_UI_IMPORT_ERROR_MESSAGE}")
        # Optionally, return here if critical modules failed to load
        # return 

    # --- Initialize session state variables for this tab if they don't exist ---    
    if 'dfm_training_status' not in session_state:
        session_state.dfm_training_status = "等待开始" # 初始化为等待开始状态
    if 'dfm_model_results' not in session_state:
        session_state.dfm_model_results = None # Initialize to None
    if 'dfm_training_log' not in session_state: # Initialize training log
        session_state.dfm_training_log = []
    if 'dfm_model_results_paths' not in session_state:  # 初始化结果路径
        session_state.dfm_model_results_paths = None
    
    # 同步全局状态到session_state
    global _training_state
    
    # 强制实时状态同步
    session_state.dfm_training_status = _training_state['status']
    session_state.dfm_training_log = _training_state['log'].copy()
    
    # 特别处理训练完成状态
    if _training_state['status'] == '训练完成' and _training_state['results']:
        session_state.dfm_model_results_paths = _training_state['results']
    elif _training_state['status'] == '训练失败':
        session_state.dfm_model_results_paths = None
    
    # === 自动检测功能已禁用 - 用户不希望自动恢复训练状态 ===
    
    # 仅在训练真正完成时刷新一次（避免重复）
    if (_training_state['status'] == '训练完成' and 
        session_state.dfm_training_status != '训练完成' and 
        'training_completed_refreshed' not in session_state):
        session_state.training_completed_refreshed = True
        st_instance.rerun()

    # 用户完全控制日期设置，不设置任何自动默认值

    # --- 数据加载与准备 ---
    input_df = session_state.get('dfm_prepared_data_df', None)
    available_target_vars = []
    if input_df is not None:
        # 从已加载数据中获取可选的目标变量
        available_target_vars = [col for col in input_df.columns if 'date' not in col.lower() and 'time' not in col.lower() and col not in getattr(config, 'EXCLUDE_COLS_FROM_TARGET', [])]
        if not available_target_vars and hasattr(config, 'TARGET_VARIABLE') and config.TARGET_VARIABLE in input_df.columns:
            available_target_vars = [config.TARGET_VARIABLE] # Fallback to config if filtering results in empty
        elif not available_target_vars:
            st_instance.warning("预处理数据中未找到合适的目标变量候选。")
    else:
        st_instance.warning("数据尚未准备，请先在\"数据准备\"选项卡中处理数据。变量选择功能将受限。")
        # 即使数据未准备好，也尝试加载映射，以便用户可以预先查看可选变量结构
        # 如果config中定义了默认目标变量，可以考虑在这里加入，但这通常依赖数据列存在
        if hasattr(config, 'TARGET_VARIABLE'):
             # We can't be sure it's a valid choice without data, so leave available_target_vars empty or add with caution
             pass 
    
    # 加载行业与指标映射
    # 获取上传的训练数据文件（来自data_prep模块）
    uploaded_training_file = session_state.get('dfm_training_data_file', None)
    
    # 使用配置的默认值
    if CONFIG_AVAILABLE:
        default_type_sheet = DataDefaults.TYPE_MAPPING_SHEET
    else:
        default_type_sheet = DataDefaults.TYPE_MAPPING_SHEET if 'DataDefaults' in globals() else '指标体系'
    
    type_mapping_sheet = session_state.get('dfm_param_type_mapping_sheet', default_type_sheet)
    
    # load_indicator_mappings_from_data_prep 现在返回: unique_industries, var_to_indicators_map_by_industry, _ (all_flat_indicators, not directly used here)
    map_data = load_indicator_mappings_from_data_prep(uploaded_training_file, type_mapping_sheet)
    if map_data and len(map_data) == 3:
        unique_industries, var_to_indicators_map_by_industry, _ = map_data # 解包三个值
        if not unique_industries or not var_to_indicators_map_by_industry:
            st_instance.warning("警告：加载的行业或指标映射不完整或为空。请检查 `load_indicator_mappings_from_data_prep` 函数和Excel文件。")
            # Fallback to empty to prevent errors, but UI will be limited
            if not unique_industries: unique_industries = []
            if not var_to_indicators_map_by_industry: var_to_indicators_map_by_industry = {}
    else:
        st_instance.error("错误：`load_indicator_mappings_from_data_prep`未能返回预期的映射数据 (预期3个元素)。")
        unique_industries = []
        var_to_indicators_map_by_industry = {}

    # 主布局：现在是上下结构，不再使用列
    # REMOVED: var_selection_col, param_col = st_instance.columns([1, 1.5]) 

    # --- 变量选择部分 (之前在 var_selection_col) ---

    # 1. 选择目标变量
    # 初始化目标变量 session state
    if 'dfm_target_variable' not in session_state and available_target_vars:
        session_state.dfm_target_variable = available_target_vars[0]
    elif not available_target_vars:
         session_state.dfm_target_variable = None

    selected_target_var = st_instance.selectbox(
        "**选择目标变量**",
        options=available_target_vars,
        index=available_target_vars.index(session_state.dfm_target_variable) if session_state.dfm_target_variable and session_state.dfm_target_variable in available_target_vars else 0,
        key="ss_dfm_target_variable",
        help="选择您希望模型预测的目标序列。"
    )
    session_state.dfm_target_variable = selected_target_var
    st_instance.markdown("--- ")

    # 2. 选择行业变量 (复选框形式，默认全选)
    st_instance.markdown("**选择行业**")
    if 'dfm_industry_checkbox_states' not in session_state and unique_industries:
        session_state.dfm_industry_checkbox_states = {industry: True for industry in unique_industries}
    elif not unique_industries:
        session_state.dfm_industry_checkbox_states = {}
    
    # 为了避免在没有行业时出错，检查 unique_industries
    if not unique_industries:
        st_instance.info("没有可用的行业数据。")
    else:
        # 创建列以更好地布局复选框
        if CONFIG_AVAILABLE:
            num_cols_industry = UIDefaults.NUM_COLS_INDUSTRY
        else:
            num_cols_industry = 3
        
        industry_cols = st_instance.columns(num_cols_industry)
        col_idx = 0
        for industry_name in unique_industries:
            with industry_cols[col_idx % num_cols_industry]:
                new_state = st_instance.checkbox(
                    industry_name, 
                    value=session_state.dfm_industry_checkbox_states.get(industry_name, True), 
                    key=f"ss_dfm_industry_cb_{industry_name}"
                )
                session_state.dfm_industry_checkbox_states[industry_name] = new_state
            col_idx += 1

        # 回调函数：取消全选行业
        def on_deselect_all_industries_change():
            if session_state.get('ss_dfm_deselect_all_industries', False):
                session_state.dfm_industry_checkbox_states = {industry: False for industry in unique_industries}
                # 重置"取消全选行业"复选框的状态，以便下次使用
                session_state.ss_dfm_deselect_all_industries = False 

        st_instance.checkbox(
            "取消全选行业", 
            key='ss_dfm_deselect_all_industries',
            on_change=on_deselect_all_industries_change,
            help="勾选此框将取消所有已选中的行业。"
        )

    # 更新当前选中的行业列表
    session_state.dfm_selected_industries = [ 
        industry for industry, checked in session_state.dfm_industry_checkbox_states.items() if checked
    ]
    st_instance.markdown("--- ")

    # 3. 根据选定行业选择预测指标 (每个行业一个多选下拉菜单，默认全选)
    st_instance.markdown("**选择预测指标**")
    if 'dfm_selected_indicators_per_industry' not in session_state:
        session_state.dfm_selected_indicators_per_industry = {}

    final_selected_indicators_flat = []
    if not session_state.dfm_selected_industries:
        st_instance.info("请先在上方选择至少一个行业。")
    else:
        for industry_name in session_state.dfm_selected_industries:
            st_instance.markdown(f"**行业: {industry_name}**")
            indicators_for_this_industry = var_to_indicators_map_by_industry.get(industry_name, [])
            
            if not indicators_for_this_industry:
                st_instance.text(f"  (该行业无可用指标)")
                session_state.dfm_selected_indicators_per_industry[industry_name] = []
                continue

            # 默认选中该行业下的所有指标
            default_selection_for_industry = session_state.dfm_selected_indicators_per_industry.get(
                industry_name, 
                indicators_for_this_industry # 默认全选
            )
            # 确保默认值是实际可选列表的子集
            valid_default = [item for item in default_selection_for_industry if item in indicators_for_this_industry]
            if not valid_default and indicators_for_this_industry: # 如果之前存的默认值无效了，且当前有可选指标，则全选
                valid_default = indicators_for_this_industry
            
            # 回调函数工厂：为特定行业的指标取消全选
            def make_on_deselect_all_indicators_change(current_industry_key):
                def on_change_callback():
                    checkbox_key = f'ss_dfm_deselect_all_indicators_{current_industry_key}'
                    multiselect_key = f'ss_dfm_indicators_{current_industry_key}'
                    if session_state.get(checkbox_key, False):
                        session_state[multiselect_key] = [] # 清空该行业已选指标
                        session_state[checkbox_key] = False # 重置取消全选复选框
                        # st_instance.rerun() # 同样，rerun需谨慎
                return on_change_callback

            st_instance.checkbox(
                f"取消全选 {industry_name} 指标",
                key=f'ss_dfm_deselect_all_indicators_{industry_name}',
                on_change=make_on_deselect_all_indicators_change(industry_name),
                help=f"勾选此框将取消所有已为 '{industry_name}' 选中的指标。"
            )

            selected_in_widget = st_instance.multiselect(
                f"为 '{industry_name}' 选择指标",
                options=indicators_for_this_industry,
                default=valid_default,
                key=f"ss_dfm_indicators_{industry_name}",
                help=f"从 {industry_name} 行业中选择预测指标。"
            )
            final_selected_indicators_flat.extend(selected_in_widget)
        
        # 清理 session_state.dfm_selected_indicators_per_industry 中不再被选中的行业条目
        industries_to_remove_from_state = [
            ind for ind in session_state.dfm_selected_indicators_per_industry 
            if ind not in session_state.dfm_selected_industries
        ]
        for ind_to_remove in industries_to_remove_from_state:
            del session_state.dfm_selected_indicators_per_industry[ind_to_remove]

    # 更新最终的扁平化预测指标列表 (去重)
    session_state.dfm_selected_indicators = sorted(list(set(final_selected_indicators_flat)))

    # 显示汇总信息 (可选)
    st_instance.markdown("--- ")    
    st_instance.text(f" - 目标变量: {session_state.dfm_target_variable if session_state.dfm_target_variable else '未选择'}")
    st_instance.text(f" - 选定行业数: {len(session_state.dfm_selected_industries)}")
    st_instance.text(f" - 选定预测指标总数: {len(session_state.dfm_selected_indicators)}")
    # with st_instance.expander("查看已选指标列表"):
    #     st_instance.json(session_state.dfm_selected_indicators if session_state.dfm_selected_indicators else [])

    # --- 模型参数配置部分 (之前在 param_col) ---
    st_instance.markdown("--- ") # 分隔线，将变量选择与参数配置分开
    st_instance.subheader("模型参数")

    # 创建三列布局
    col1_time, col2_factor_core, col3_factor_specific = st_instance.columns(3)

    # --- 第一列: 时间窗口设置 ---
    with col1_time:
        st_instance.markdown("###### 时间窗口设置")
        
        # 计算基于数据的智能默认值
        def get_data_based_date_defaults():
            """基于实际数据计算日期默认值，确保验证期不会是未来日期"""
            from datetime import datetime, timedelta
            today = datetime.now().date()
            
            # 静态默认值（备用）- 🔥 修复：验证期结束日期固定为2024年12月31日
            static_defaults = {
                'training_start': datetime(today.year - 5, 1, 1).date(),
                'validation_start': datetime(2024, 7, 1).date(),  # 2024年7月1日
                'validation_end': datetime(2024, 12, 31).date()   # 2024年12月31日
            }
            
            try:
                data_df = session_state.get('dfm_prepared_data_df')
                if data_df is not None and isinstance(data_df.index, pd.DatetimeIndex) and len(data_df.index) > 0:
                    # 从数据获取第一期和最后一期
                    data_first_date = data_df.index.min().date()  # 第一期数据
                    data_last_date = data_df.index.max().date()   # 最后一期数据
                    
                    # 重要：确保数据的最后日期不是未来日期
                    if data_last_date > today:
                        print(f"⚠️ 警告: 数据包含未来日期 {data_last_date}，将使用今天作为最后日期")
                        data_last_date = today
                    
                    # 计算验证期开始日期：使用数据时间范围的80%作为训练期
                    total_days = (data_last_date - data_first_date).days
                    training_days = int(total_days * 0.8)
                    validation_start_date = data_first_date + timedelta(days=training_days)
                    
                    # 确保验证期开始日期不是未来日期
                    if validation_start_date > today:
                        validation_start_date = today - timedelta(days=30)  # 1个月前
                    
                    # 🔥 修复：使用固定的验证期结束日期2024年12月31日
                    validation_end_date = datetime(2024, 12, 31).date()  # 固定使用2024年12月31日
                    
                    # 验证日期逻辑的合理性
                    if validation_start_date >= validation_end_date:
                        # 如果验证期开始晚于或等于结束，重新计算
                        validation_end_date = datetime(2024, 12, 31).date()  # 固定使用2024年12月31日
                        validation_start_date = validation_end_date - timedelta(days=90)  # 验证期3个月
                    
                    return {
                        'training_start': data_first_date,           # 训练开始日：第一期数据
                        'validation_start': validation_start_date,   # 验证开始日：计算得出
                        'validation_end': validation_end_date        # 验证结束日：不晚于昨天
                    }
                else:
                    return static_defaults
            except Exception as e:
                print(f"⚠️ 计算数据默认日期失败: {e}，使用静态默认值")
                return static_defaults
        
        # 获取智能默认值
        date_defaults = get_data_based_date_defaults()
        
        # 检查是否有数据，如果有则强制更新默认值
        has_data = session_state.get('dfm_prepared_data_df') is not None
        if has_data:
            data_df = session_state['dfm_prepared_data_df']
            if isinstance(data_df.index, pd.DatetimeIndex) and len(data_df.index) > 0:
                # 计算数据的实际日期范围用于比较
                actual_data_start = data_df.index.min().date()
                actual_data_end = data_df.index.max().date()
                
                # 强制更新session_state中的日期默认值（检查是否为静态默认值或与数据不匹配）
                current_training_start = session_state.get('dfm_training_start_date')
                if (current_training_start == datetime(2010, 1, 1).date() or 
                    'dfm_training_start_date' not in session_state or
                    current_training_start != actual_data_start):
                    session_state.dfm_training_start_date = date_defaults['training_start']
                
                current_validation_start = session_state.get('dfm_validation_start_date')  
                if (current_validation_start == datetime(2020, 12, 31).date() or 
                    'dfm_validation_start_date' not in session_state):
                    session_state.dfm_validation_start_date = date_defaults['validation_start']
                
                current_validation_end = session_state.get('dfm_validation_end_date')
                if (current_validation_end == datetime(2022, 12, 31).date() or 
                    'dfm_validation_end_date' not in session_state or
                    current_validation_end != actual_data_end):
                    session_state.dfm_validation_end_date = date_defaults['validation_end']
                
                # 简化数据范围信息
                data_start = data_df.index.min().strftime('%Y-%m-%d')
                data_end = data_df.index.max().strftime('%Y-%m-%d')
                data_count = len(data_df.index)
                st_instance.info(f"📊 数据: {data_start} 至 {data_end} ({data_count}点)")
                

        
        # 1. 训练期开始日期
        session_state.dfm_training_start_date = st_instance.date_input(
            "训练期开始日期 (Training Start Date)", 
            value=session_state.get('dfm_training_start_date', date_defaults['training_start']),
            key='dfm_training_start_date_input',
            help="选择模型训练数据的起始日期。默认为数据的第一期。"
        )
        # 2. 验证期开始日期
        session_state.dfm_validation_start_date = st_instance.date_input(
            "验证期开始日期 (Validation Start Date)", 
            value=session_state.get('dfm_validation_start_date', date_defaults['validation_start']),
            key='dfm_validation_start_date_input',
            help="选择验证期开始日期。默认为最后一期数据前3个月。"
        )
        # 3. 验证期结束日期
        session_state.dfm_validation_end_date = st_instance.date_input(
            "验证期结束日期 (Validation End Date)", 
            value=session_state.get('dfm_validation_end_date', date_defaults['validation_end']),
            key='dfm_validation_end_date_input',
            help="选择验证期结束日期。默认为数据的最后一期。"
        )

    # --- 第二列: 核心因子参数设置 ---
    with col2_factor_core:
        st_instance.markdown("###### 核心因子参数")
        
        # 🔥 新增：变量选择方法
        if CONFIG_AVAILABLE:
            variable_selection_options = UIDefaults.VARIABLE_SELECTION_OPTIONS
            default_var_method = TrainDefaults.VARIABLE_SELECTION_METHOD
        else:
            variable_selection_options = {
                'none': "无筛选 (使用全部变量)",
                'global_backward': "全局后向剔除"
            }
            default_var_method = 'global_backward'
        
        # 获取当前变量选择方法
        current_var_method = session_state.get('dfm_variable_selection_method', default_var_method)
        
        session_state.dfm_variable_selection_method = st_instance.selectbox(
            "变量选择方法",
            options=list(variable_selection_options.keys()),
            format_func=lambda x: variable_selection_options[x],
            index=list(variable_selection_options.keys()).index(current_var_method),
            key='dfm_variable_selection_method_input',
            help=(
                "选择变量筛选方法：\n"
                "- 无筛选: 使用所有已选择的变量\n"
                "- 全局后向剔除: 从所有变量开始，逐个剔除不重要的变量"
            )
        )
        
        # 启用变量选择的复选框
        enable_var_selection = session_state.dfm_variable_selection_method != 'none'
        session_state.dfm_enable_variable_selection = st_instance.checkbox(
            "启用变量选择",
            value=enable_var_selection,
            key='dfm_enable_variable_selection_input',
            help="启用后会根据上面选择的方法进行变量筛选，禁用则使用所有已选变量"
        )
        
        # 如果禁用了变量选择，强制设置方法为none
        if not session_state.dfm_enable_variable_selection:
            session_state.dfm_variable_selection_method = 'none'
        
        st_instance.markdown("---")
        
        # 1. 因子数量选择策略
        if CONFIG_AVAILABLE:
            factor_selection_strategy_options = UIDefaults.FACTOR_SELECTION_STRATEGY_OPTIONS
            default_strategy = TrainDefaults.FACTOR_SELECTION_STRATEGY
        else:
            factor_selection_strategy_options = {
                'information_criteria': "信息准则 (Information Criteria)",
                'fixed_number': "固定因子数量 (Fixed Number of Factors)",
                'cumulative_variance': "累积共同方差 (Cumulative Common Variance)"
            }
            default_strategy = 'information_criteria'
            
        session_state.dfm_factor_selection_strategy = st_instance.selectbox(
            "因子数量选择策略",
            options=list(factor_selection_strategy_options.keys()),
            format_func=lambda x: factor_selection_strategy_options[x],
            index=list(factor_selection_strategy_options.keys()).index(session_state.get('dfm_factor_selection_strategy', default_strategy)),
            key='dfm_factor_selection_strategy_input',
            help=(
                "选择确定模型中因子数量的方法：\n"
                "- 信息准则: 根据AIC/BIC等自动选择。\n"
                "- 固定因子数量: 直接指定因子数量。\n"
                "- 累积共同方差: 根据解释的方差比例确定因子数。"
            )
        )

        # 2. 最大迭代次数 (EM算法)
        if CONFIG_AVAILABLE:
            max_iter_default = UIDefaults.MAX_ITERATIONS_DEFAULT
            max_iter_min = UIDefaults.MAX_ITERATIONS_MIN
            max_iter_step = UIDefaults.MAX_ITERATIONS_STEP
        else:
            max_iter_default = UIDefaults.MAX_ITERATIONS_DEFAULT if 'UIDefaults' in globals() else 30
            max_iter_min = 1
            max_iter_step = 10
            
        session_state.dfm_max_iterations = st_instance.number_input(
            "最大迭代次数 (Max Iterations for EM)", 
            min_value=max_iter_min, 
            value=session_state.get('dfm_max_iterations', max_iter_default),
            step=max_iter_step,
            key='dfm_max_iterations_input',
            help="EM估计算法允许的最大迭代次数。"
        )

        # 3. 根据策略显示固定因子数量或手动覆盖
        if session_state.dfm_factor_selection_strategy == 'fixed_number':
            # 🔥 修复：使用配置的默认值而不是硬编码的3
            if CONFIG_AVAILABLE:
                default_fixed_factors = TrainDefaults.FIXED_NUMBER_OF_FACTORS
            else:
                default_fixed_factors = TrainDefaults.FIXED_NUMBER_OF_FACTORS if 'TrainDefaults' in globals() else 3
                
            session_state.dfm_fixed_number_of_factors = st_instance.number_input(
                "固定因子数量 (Fixed Number of Factors)", 
                min_value=1, 
                value=session_state.get('dfm_fixed_number_of_factors', default_fixed_factors),  # 🔥 使用配置值
                step=1,
                key='dfm_fixed_number_of_factors_input',
                help="直接指定模型中要使用的因子数量。"
            )
        elif session_state.dfm_factor_selection_strategy == 'manual_override': # 虽然移除了选项，但保留逻辑以防未来需要
            # 🔥 修复：使用配置的默认值而不是硬编码的3
            if CONFIG_AVAILABLE:
                default_manual_factors = TrainDefaults.FIXED_NUMBER_OF_FACTORS
            else:
                default_manual_factors = TrainDefaults.FIXED_NUMBER_OF_FACTORS if 'TrainDefaults' in globals() else 3
                
            session_state.dfm_manual_override_factors = st_instance.number_input(
                "因子数量 (手动覆盖)", 
                min_value=1, 
                value=session_state.get('dfm_manual_override_factors', default_manual_factors),  # 🔥 使用配置值
                step=1,
                key='dfm_manual_override_factors_input',
                help="手动指定模型中的因子数量，这将覆盖其他自动选择方法。"
            )

    # --- 第三列: 其他/特定因子参数设置 ---
    with col3_factor_specific:
        st_instance.markdown("###### 特定因子参数")
        if session_state.dfm_factor_selection_strategy == 'information_criteria':
            # a. 信息准则选择 (BIC, AIC等)
            info_criterion_options = {
                'bic': "BIC (Bayesian Information Criterion)",
                'aic': "AIC (Akaike Information Criterion)",
                # 'hqic': "HQIC (Hannan-Quinn Information Criterion)" # 视具体实现支持情况添加
            }
            session_state.dfm_info_criterion_method = st_instance.selectbox(
                "信息准则选择",
                options=list(info_criterion_options.keys()),
                format_func=lambda x: info_criterion_options[x],
                index=list(info_criterion_options.keys()).index(session_state.get('dfm_info_criterion_method', 'bic')),
                key='dfm_info_criterion_method_input',
                help="选择用于确定最佳因子数量的信息准则。"
            )
            # b. IC 最大因子数
            session_state.dfm_ic_max_factors = st_instance.number_input(
                "IC 最大因子数 (Max Factors for IC)", 
                min_value=1, 
                value=session_state.get('dfm_ic_max_factors', 10),
                step=1,
                key='dfm_ic_max_factors_input',
                help="使用信息准则时，允许测试的最大因子数量。"
            )

        elif session_state.dfm_factor_selection_strategy == 'cumulative_variance':
            # a. 累积共同方差阈值
            session_state.dfm_cum_variance_threshold = st_instance.slider(
                "累积共同方差阈值 (Cumulative Variance Threshold)",
                min_value=0.1, max_value=1.0, 
                value=session_state.get('dfm_cum_variance_threshold', 0.8),
                step=0.05,
                key='dfm_cum_variance_threshold_input',
                help="选择因子以解释至少此比例的共同方差。值在0到1之间。"
            )
        else:
            st_instance.empty() # 如果当前策略没有特定参数，则此列为空

    # --- 重置参数按钮 ---
    # 定义默认参数的函数，以便重置时调用
    def get_default_model_params():
        # 🔥 修复：使用配置值而不是硬编码
        if CONFIG_AVAILABLE:
            default_fixed_factors = TrainDefaults.FIXED_NUMBER_OF_FACTORS
            default_manual_factors = TrainDefaults.FIXED_NUMBER_OF_FACTORS
        else:
            default_fixed_factors = TrainDefaults.FIXED_NUMBER_OF_FACTORS if 'TrainDefaults' in globals() else 3
            default_manual_factors = TrainDefaults.FIXED_NUMBER_OF_FACTORS if 'TrainDefaults' in globals() else 3
            
        # 只重置非日期参数，日期由用户完全控制
        return {
            # 🔥 变量选择参数
            'dfm_variable_selection_method': 'global_backward',
            'dfm_enable_variable_selection': True,
            # 因子选择参数
            'dfm_factor_selection_strategy': 'information_criteria',
            'dfm_max_iterations': 30,
            'dfm_fixed_number_of_factors': default_fixed_factors,  # 🔥 使用配置值
            'dfm_manual_override_factors': default_manual_factors,  # 🔥 使用配置值
            'dfm_info_criterion_method': 'bic',
            'dfm_ic_max_factors': 10,
            'dfm_cum_variance_threshold': 0.8,
        }

    def reset_model_parameters():
        defaults = get_default_model_params()
        for key, value in defaults.items():
            # 直接更新 session_state 中的值
            # 对于 date_input, selectbox, number_input, slider, Streamlit 会在下次渲染时
            # 使用这些新的 session_state 值作为其 value/index/default
            session_state[key] = value

        st_instance.success("模型参数已重置为默认值（日期参数不受影响，由用户完全控制）。")
        # st_instance.rerun() # 可以考虑是否需要 rerun，通常状态更新后会自动重绘

    # 重置参数按钮
    col_reset, col_spacer = st_instance.columns([2, 8])
    
    with col_reset:
        st_instance.button("重置模型参数", on_click=reset_model_parameters, help="点击将所有模型参数恢复到预设的默认值。")

    # --- EM算法收敛标准 (通常不需要用户调整，设为默认值) ---
    # session_state.dfm_em_convergence_criterion = 1e-4 # 保持默认或从config加载

    st_instance.markdown("--- ") # 分隔线在因子参数设置模块之后

    # 新增"模型训练"模块
    st_instance.subheader("模型训练")
    
    # 开始训练按钮
    col_train_btn, col_spacer = st_instance.columns([2, 8])
    
    with col_train_btn:
        # 检查预处理数据是否存在
        input_df_check = session_state.get('dfm_prepared_data_df', None)
        can_train = input_df_check is not None and not input_df_check.empty
        
        if st_instance.button("🚀 开始训练模型", key="dfm_train_model_button", disabled=not can_train):
            # **新增：验证训练期、验证期是否在数据准备页面设置的边界内**
            validation_error = None
            
            # 获取数据准备页面设置的边界日期
            data_prep_start = session_state.get('dfm_param_data_start_date')
            data_prep_end = session_state.get('dfm_param_data_end_date')
            
            # 获取用户设置的训练期、验证期
            train_start = session_state.get('dfm_training_start_date')
            val_start = session_state.get('dfm_validation_start_date')
            val_end = session_state.get('dfm_validation_end_date')
            
            # 验证日期边界
            if data_prep_start and train_start and train_start < data_prep_start:
                validation_error = f"训练开始日期 ({train_start}) 不能早于数据准备页面设置的开始边界 ({data_prep_start})"
            elif data_prep_end and val_end and val_end > data_prep_end:
                validation_error = f"验证结束日期 ({val_end}) 不能晚于数据准备页面设置的结束边界 ({data_prep_end})"
            elif data_prep_start and val_start and val_start < data_prep_start:
                validation_error = f"验证开始日期 ({val_start}) 不能早于数据准备页面设置的开始边界 ({data_prep_start})"
            elif data_prep_end and train_start and train_start > data_prep_end:
                validation_error = f"训练开始日期 ({train_start}) 不能晚于数据准备页面设置的结束边界 ({data_prep_end})"
            
            if validation_error:
                st_instance.error(f"❌ 日期验证失败: {validation_error}")
                st_instance.info("💡 请调整训练期、验证期设置，确保在数据准备页面设置的日期边界内。")
                return  # 不执行训练
            
            # 重置全局状态
            _training_state['status'] = '准备启动训练...'
            _training_state['log'] = []
            _training_state['results'] = None
            _training_state['error'] = None
            
            # 设置训练启动标志
            session_state.dfm_should_start_training = True
            session_state.dfm_training_log = []
            session_state.dfm_training_status = "准备启动训练..."

    # --- Callback function for logging ---
    def training_log_callback(message, st_instance_ref=st_instance):
        # 这个函数现在只用于UI内部，不应该在训练线程中使用
        # 训练线程有自己的thread_log_callback
        pass

    # 检查是否需要启动训练
    should_start_training = session_state.get('dfm_should_start_training', False)
    
    # 执行训练逻辑（在布局之前）
    if should_start_training:
        # 重置启动标志
        session_state.dfm_should_start_training = False
        
        # 重置全局状态
        _training_state['status'] = '准备启动训练...'
        _training_state['log'] = []
        _training_state['results'] = None
        _training_state['error'] = None
        
        session_state.dfm_training_status = '准备启动训练...'
        session_state.dfm_training_log = []
        session_state.dfm_model_results_paths = None
        
        # 获取所有训练参数
        # 根据用户选择的策略确定因子数量
        if session_state.get('dfm_factor_selection_strategy') == 'fixed_number':
            n_factors = session_state.get('dfm_fixed_number_of_factors', TrainDefaults.FIXED_NUMBER_OF_FACTORS if CONFIG_AVAILABLE else 3)  # 🔥 使用配置值
        elif session_state.get('dfm_factor_selection_strategy') == 'manual_override':
            n_factors = session_state.get('dfm_manual_override_factors', TrainDefaults.FIXED_NUMBER_OF_FACTORS if CONFIG_AVAILABLE else 3)  # 🔥 使用配置值
        else:
            # 对于信息准则和累积方差策略，提供一个默认值，后端会自动优化
            n_factors = session_state.get('dfm_fixed_number_of_factors', TrainDefaults.FIXED_NUMBER_OF_FACTORS if CONFIG_AVAILABLE else 3)  # 🔥 使用配置值

        # 🔥 修复：使用配置的因子选择范围
        # 因子选择范围参数 - 根据用户的策略设置
        if session_state.get('dfm_factor_selection_strategy') == 'information_criteria':
            # 使用配置的因子选择范围
            if CONFIG_AVAILABLE:
                min_factors = TrainDefaults.K_FACTORS_RANGE_MIN
                max_factors = min(session_state.get('dfm_ic_max_factors', TrainDefaults.IC_MAX_FACTORS), TrainDefaults.K_FACTORS_RANGE_MAX)
            else:
                min_factors = 1
                max_factors = session_state.get('dfm_ic_max_factors', 10)
            k_factors_range = (min_factors, max_factors)
        else:
            # 对于固定数量策略，使用单一值
            k_factors_range = (n_factors, n_factors)

        params_for_training = {
            'input_df': session_state.get('dfm_prepared_data_df'),
            'target_variable': session_state.get('dfm_target_variable', '规模以上工业增加值:当月同比'),
            'selected_indicators': session_state.get('dfm_selected_indicators', []),
            'training_start_date': convert_to_datetime(session_state.get('dfm_training_start_date')),
            'training_end_date': None,  # 让后端自动计算训练结束日期
            'n_factors': n_factors,  # 使用计算得到的因子数量
            'em_max_iter': session_state.get('dfm_max_iterations', 50),
            'info_criterion_method': session_state.get('dfm_info_criterion_method', 'bic'),
            'cum_variance_threshold': session_state.get('dfm_cum_variance_threshold', 0.8),
            'output_base_dir': config.DFM_TRAIN_OUTPUT_DIR if hasattr(config, 'DFM_TRAIN_OUTPUT_DIR') else "dashboard/DFM/outputs",
            'progress_callback': training_log_callback,
            # 验证期参数
            'validation_start_date': convert_to_datetime(session_state.get('dfm_validation_start_date')),
            'validation_end_date': convert_to_datetime(session_state.get('dfm_validation_end_date')),
            # 因子选择策略参数
            'enable_hyperparameter_tuning': session_state.get('dfm_factor_selection_strategy') != 'fixed_number',
            # 🔥 修复：使用UI设置的变量选择参数
            'enable_variable_selection': session_state.get('dfm_enable_variable_selection', True),
            'variable_selection_method': session_state.get('dfm_variable_selection_method', 'global_backward'),
            'enable_detailed_analysis': True,  # 启用详细分析以支持Excel报告
            'generate_excel_report': True,  # 启用Excel报告生成
            # 🔥 修复：因子选择范围参数 - 使用配置值
            'k_factors_range': k_factors_range,
            # 映射参数 - 从session_state中获取data_prep模块的输出
            'var_industry_map': session_state.get('dfm_industry_map_obj'),
            'var_type_map': session_state.get('dfm_var_type_map_obj')
        }
        
        # 参数验证和调试信息
        print(f"[TRAIN_DEBUG] 训练参数检查:")
        print(f"  - 数据形状: {params_for_training['input_df'].shape if params_for_training['input_df'] is not None else 'None'}")
        print(f"  - 目标变量: {params_for_training['target_variable']}")
        print(f"  - 选择指标数量: {len(params_for_training['selected_indicators'])}")
        print(f"  - 训练开始日期: {params_for_training['training_start_date']}")
        print(f"  - 验证开始日期: {params_for_training['validation_start_date']}")
        print(f"  - 验证结束日期: {params_for_training['validation_end_date']}")
        # 🔥 显示变量选择参数
        print(f"  - 🔥 启用变量选择: {params_for_training['enable_variable_selection']}")
        print(f"  - 🔥 变量选择方法: {params_for_training['variable_selection_method']}")
        print(f"  - 因子数量参数: {params_for_training['n_factors']}")
        print(f"  - 因子选择策略: {session_state.get('dfm_factor_selection_strategy', 'unknown')}")
        print(f"  - 🔥🔥🔥 因子选择范围: {params_for_training['k_factors_range']} (这是系统会测试的因子数量范围)")
        print(f"  - 最大迭代次数: {params_for_training['em_max_iter']}")
        print(f"  - 信息准则方法: {params_for_training['info_criterion_method']}")

        # 🔥 增强配置参数验证信息
        if CONFIG_AVAILABLE:
            print(f"  - 🔥 配置K_FACTORS_RANGE_MIN: {TrainDefaults.K_FACTORS_RANGE_MIN}")
            print(f"  - 🔥 配置K_FACTORS_RANGE_MAX: {TrainDefaults.K_FACTORS_RANGE_MAX}")
            print(f"  - 🔥 配置FIXED_NUMBER_OF_FACTORS: {TrainDefaults.FIXED_NUMBER_OF_FACTORS}")
            print(f"  - 🔥 配置IC_MAX_FACTORS: {TrainDefaults.IC_MAX_FACTORS}")
        else:
            print(f"  - ⚠️ 配置文件不可用，使用硬编码默认值")
        
        # 在单独线程中运行训练
        training_thread = threading.Thread(
                    target=_run_training_thread, 
            args=(params_for_training, st_instance, training_log_callback),
            daemon=True
        )
        training_thread.start()

    # 创建紧凑的状态和结果显示区域
    col_status, col_results = st_instance.columns([3, 2])
    
    with col_status:
        st_instance.markdown("**训练状态**")
        
        # 实时状态显示，包含自动同步
        current_status = _training_state['status']  # 直接使用全局状态
        
        # 显示当前训练状态
        if current_status == '等待开始':
            st_instance.info("🔵 准备就绪")
        elif current_status == '准备启动训练...':
            st_instance.info("🟡 准备中...")
        elif current_status == '正在训练...':
            st_instance.warning("🟠 训练中...")
            # 训练中自动刷新页面
            import time
            time.sleep(0.5)
            st_instance.rerun()
        elif current_status == '训练完成':
            st_instance.success("🟢 训练完成")
        elif current_status.startswith('训练失败'):
            st_instance.error(f"🔴 训练失败")
        else:
            st_instance.info(f"📊 {current_status}")
        
        # 数据准备状态检查
        if input_df_check is None:
            st_instance.error("❌ 无数据")
        elif input_df_check.empty:
            st_instance.error("❌ 数据空")
        else:
            st_instance.success(f"✅ 数据 ({input_df_check.shape[0]}×{input_df_check.shape[1]})")
        
        # 实时训练进度显示
        current_log = _training_state['log']  # 直接使用全局日志
        
        st_instance.markdown("**训练日志**")
        
        if current_log:
            # 显示最新5条日志
            recent_logs = current_log[-5:] if len(current_log) > 5 else current_log
            log_content = "\n".join(recent_logs)
            
            # 使用动态key确保内容更新
            log_display_key = f"dfm_log_display_{len(current_log)}_{hash(log_content) % 10000}"
            st_instance.text_area(
                "", 
                value=log_content, 
                height=120, 
                disabled=True, 
                key=log_display_key,
                label_visibility="collapsed"
            )
            
            # 简化日志统计
            st_instance.caption(f"📝 {len(current_log)} 条日志")
        else:
            if current_status in ['正在训练...', '准备启动训练...']:
                st_instance.info("⏳ 等待日志...")
            else:
                st_instance.info("🔘 无日志")
    
    with col_results:
        st_instance.markdown("**训练结果**")
        
        # 检查训练是否完成 - 使用全局状态而不是session_state
        current_results = _training_state.get('results')
        
        if current_status == '训练完成' and current_results:
            # 计算实际存在的核心文件数量（模型文件、元数据和Excel报告）
            core_file_keys = ['final_model_joblib', 'model_joblib', 'metadata', 'simplified_metadata', 'excel_report']
            core_files = {k: v for k, v in current_results.items() if k in core_file_keys}
            existing_core_files = sum(1 for path in core_files.values() if path and os.path.exists(path))
            
            st_instance.success("✅ 训练完成")
            st_instance.info(f"📁 {existing_core_files} 个文件")
            
            # 同步到session_state以便下载功能正常工作
            session_state.dfm_training_status = "训练完成"
            session_state.dfm_model_results_paths = current_results
            
            # 紧凑的下载区域
            st_instance.markdown("**📥 下载文件**")
            
            # 核心文件：模型文件、元数据和Excel报告
            core_file_types = {
                'final_model_joblib': ('📦', '模型'),
                'model_joblib': ('📦', '模型'), 
                'metadata': ('📄', '元数据'),
                'simplified_metadata': ('📄', '元数据'),
                'excel_report': ('📊', 'Excel报告')
            }
            
            # 收集可用的下载文件
            available_downloads = []
            for file_key, file_path in current_results.items():
                if file_key in core_file_types and file_path and os.path.exists(file_path):
                    available_downloads.append((file_key, file_path))
            
            # 使用三列布局显示下载按钮
            if available_downloads:
                num_cols = min(3, len(available_downloads))  # 最多3列
                download_cols = st_instance.columns(num_cols)
                
                for idx, (file_key, file_path) in enumerate(available_downloads[:3]):  # 只显示前3个文件
                    with download_cols[idx % num_cols]:
                        try:
                            # 获取文件图标和显示名称
                            icon, display_name = core_file_types[file_key]
                            file_name = os.path.basename(file_path)
                            
                            # 根据文件类型确定MIME类型
                            if file_path.endswith('.pkl'):
                                mime_type = "application/octet-stream"
                            elif file_path.endswith('.joblib'):
                                mime_type = "application/octet-stream"
                            elif file_path.endswith('.xlsx'):
                                mime_type = "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                            else:
                                mime_type = "application/octet-stream"
                            
                            # 读取文件数据
                            with open(file_path, 'rb') as f:
                                file_data = f.read()
                            
                            # 创建下载按钮
                            st_instance.download_button(
                                label=f"{icon} {display_name}",
                                data=file_data,
                                file_name=file_name,
                                mime=mime_type,
                                key=f"download_{file_key}_{hash(file_path) % 1000}",
                                use_container_width=True
                            )
                            
                        except Exception as e:
                            st_instance.warning(f"⚠️ {display_name} 文件读取失败")
            
                # 显示下载统计
                if len(available_downloads) > 0:
                    st_instance.success(f"📁 {len(available_downloads)} 个文件可下载")
            else:
                st_instance.warning("❌ 暂无可下载的文件")
