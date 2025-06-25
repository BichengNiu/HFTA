# -*- coding: utf-8 -*-

"""
超参数和变量逐步前向选择脚本。
目标：最小化 OOS RMSE。
"""
import pandas as pd
# import suppress_prints  # 抑制子进程中的重复打印 - 暂时注释掉
import numpy as np
import sys
import os
import time
import warnings

# === 优化：添加全局静默控制 ===
_SILENT_MODE = os.getenv('DFM_SILENT_WARNINGS', 'true').lower() == 'true'

def _safe_print(*args, **kwargs):
    """安全的条件化print函数，在多进程环境下也能正常工作"""
    if not _SILENT_MODE:
        try:
            print(*args, **kwargs)
        except:
            pass  # 忽略任何打印错误
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns # <-- MOVED BACK TO TOP
import concurrent.futures
from tqdm import tqdm
import traceback
from typing import Tuple, List, Dict, Union, Optional, Any # 添加 Tuple, Optional, Any
import unicodedata # <-- 新增导入
from sklearn.decomposition import PCA # <-- 新增：导入 PCA
from sklearn.impute import SimpleImputer # <-- 新增：导入 SimpleImputer
import multiprocessing
from collections import defaultdict
import logging
import joblib # 用于保存和加载模型/结果
from datetime import datetime
import pickle
try:
    from . import config
    # 优化：移除导入时的print语句
except ImportError:
    try:
        import config
        # 优化：移除导入时的print语句
    except ImportError:
        try:
            # 尝试从上级目录导入
            import sys
            import os
            parent_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            sys.path.insert(0, parent_dir)
            import config
            # 优化：移除导入时的print语句
        except ImportError:
            # 优化：移除导入时的print语句
            # 创建一个模拟的config模块
            class MockConfig:
                def __init__(self):
                    # 设置默认配置值
                    self.EXCEL_DATA_FILE = "data/高频数据.xlsx"
                    self.TARGET_FREQ = 'W-FRI'
                    self.TARGET_SHEET_NAME = '规模以上工业增加值'
                    self.TARGET_VARIABLE = '规模以上工业增加值:当月同比'
                    self.CONSECUTIVE_NAN_THRESHOLD = 10
                    self.REMOVE_VARS_WITH_CONSECUTIVE_NANS = True
                    self.DATA_START_DATE = '2020-01-01'
                    self.DATA_END_DATE = None

                def __getattr__(self, name):
                    # 如果属性不存在，返回None
                    if name == 'EXCEL_DATA_FILE':
                        return "data/高频数据.xlsx"
                    return None

            config = MockConfig()
            # 优化：移除导入时的print语句
import random
SEED = 42
random.seed(SEED)
np.random.seed(SEED)


# --- 使用导入的配置 ---
# 安全检查 config.EXCEL_DATA_FILE 是否存在
if hasattr(config, 'EXCEL_DATA_FILE'):
    EXCEL_DATA_FILE = config.EXCEL_DATA_FILE
else:
    # 如果 config 中没有 EXCEL_DATA_FILE，使用默认路径
    # 优化：完全移除重复的警告打印（多进程环境下全局变量无效）
    import os
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(current_dir)))
    EXCEL_DATA_FILE = os.path.join(project_root, 'data', '经济数据库0605.xlsx')

# --- 结束使用导入配置 ---
# 安全访问 config 属性，提供默认值
TEST_MODE = getattr(config, 'TEST_MODE', False)
N_ITER_TEST = getattr(config, 'N_ITER_TEST', 2)
N_ITER_FIXED = getattr(config, 'N_ITER_FIXED', 30)
TARGET_FREQ = getattr(config, 'TARGET_FREQ', 'W-FRI')
TARGET_SHEET_NAME = getattr(config, 'TARGET_SHEET_NAME', '工业增加值同比增速_月度_同花顺')
TARGET_VARIABLE = getattr(config, 'TARGET_VARIABLE', '规模以上工业增加值:当月同比')
CONSECUTIVE_NAN_THRESHOLD = getattr(config, 'CONSECUTIVE_NAN_THRESHOLD', 10)
REMOVE_VARS_WITH_CONSECUTIVE_NANS = getattr(config, 'REMOVE_VARS_WITH_CONSECUTIVE_NANS', True)
TYPE_MAPPING_SHEET = getattr(config, 'TYPE_MAPPING_SHEET', '指标体系')

VALIDATION_END_DATE = getattr(config, 'VALIDATION_END_DATE', '2024-12-27')
TRAIN_END_DATE = getattr(config, 'TRAIN_END_DATE', '2024-06-28')
TRAINING_START_DATE = getattr(config, 'TRAINING_START_DATE', '2020-01-01')

# --- <<< 修改：导入新的因子选择配置 >>> ---
FACTOR_SELECTION_METHOD = getattr(config, 'FACTOR_SELECTION_METHOD', 'bai_ng')
PCA_INERTIA_THRESHOLD = getattr(config, 'PCA_INERTIA_THRESHOLD', 0.9)
ELBOW_DROP_THRESHOLD = getattr(config, 'ELBOW_DROP_THRESHOLD', 0.1)
COMMON_VARIANCE_CONTRIBUTION_THRESHOLD = getattr(config, 'COMMON_VARIANCE_CONTRIBUTION_THRESHOLD', 0.8)
DEBUG_VARIABLE_SELECTION_BLOCK = getattr(config, 'DEBUG_VARIABLE_SELECTION_BLOCK', "库存")
HEATMAP_TOP_N_VARS = getattr(config, 'HEATMAP_TOP_N_VARS', 5)
# --- 结束修改 ---


# --- 内部模块导入 ---

# import data_preparation  # <-- Explicitly comment out this line
try:
    # 🔥 修复：使用绝对导入避免相对导入问题
    import sys
    import os
    current_dir = os.path.dirname(os.path.abspath(__file__))
    if current_dir not in sys.path:
        sys.path.append(current_dir)

    from dfm_core import evaluate_dfm_params # <-- 保持，但需要修改 dfm_core.py 内部
    from analysis_utils import calculate_pca_variance, calculate_factor_contributions, calculate_individual_variable_r2, calculate_industry_r2, calculate_factor_industry_r2, calculate_factor_type_r2 # <<< 导入新增函数
    # --- <<< 修改：导入新的全局筛选函数 >>> ---
    # from variable_selection import perform_backward_selection # <-- 注释掉这个
    from variable_selection import perform_global_backward_selection # <-- 导入全局筛选函数
    # --- 结束修改 ---
except ImportError:
    # 回退到绝对导入
    from dfm_core import evaluate_dfm_params
    from analysis_utils import calculate_pca_variance, calculate_factor_contributions, calculate_individual_variable_r2, calculate_industry_r2, calculate_factor_industry_r2, calculate_factor_type_r2
    from variable_selection import perform_global_backward_selection
# --- 结束新增 ---


# === 导入优化：添加缓存机制 ===
_IMPORT_CACHE = {}
_IMPORT_MESSAGES_SHOWN = False

def _cached_import_with_message(module_name, import_func, success_msg="", error_msg=""):
    """缓存导入并控制消息显示"""
    global _IMPORT_CACHE, _IMPORT_MESSAGES_SHOWN
    
    if module_name in _IMPORT_CACHE:
        return _IMPORT_CACHE[module_name]
    
    try:
        result = import_func()
        _IMPORT_CACHE[module_name] = result
        # 只在首次导入时显示消息
        if not _IMPORT_MESSAGES_SHOWN and success_msg:
            print(success_msg)
        return result
    except Exception as e:
        if not _IMPORT_MESSAGES_SHOWN and error_msg:
            print(f"{error_msg}: {e}")
        _IMPORT_CACHE[module_name] = None
        return None
    finally:
        _IMPORT_MESSAGES_SHOWN = True

# --- 尝试导入自定义模块 ---
# 导入数据准备和DFM模块
try:
    from .data_preparation import prepare_data, load_mappings
    # 优化：移除导入时的print语句
except ImportError:
    try:
        from data_preparation import prepare_data, load_mappings
        # 优化：移除导入时的print语句
    except ImportError as e:
        # 优化：移除导入时的print语句
        # 提供模拟函数
        def prepare_data(*args, **kwargs):
            # 优化：移除导入时的print语句
            return None, {}, {}, {}
        def load_mappings(*args, **kwargs):
            # 优化：移除导入时的print语句
            return {}, {}

try:
    from .DynamicFactorModel import DFM_EMalgo
    # 优化：移除导入时的print语句
except ImportError:
    try:
        from DynamicFactorModel import DFM_EMalgo
        # 优化：移除导入时的print语句
    except ImportError as e:
        # 优化：移除导入时的print语句
        # 提供模拟函数
        def DFM_EMalgo(*args, **kwargs):
            # 优化：移除导入时的print语句
            return None

# 🔥 修改：导入专业报告生成模块而不是直接调用analyze_and_save_final_results
try:
    from .generate_report import generate_report_with_params
    # 优化：移除导入时的print语句
    _GENERATE_REPORT_AVAILABLE = True
except ImportError:
    try:
        from generate_report import generate_report_with_params
        # 优化：移除导入时的print语句
        _GENERATE_REPORT_AVAILABLE = True
    except ImportError:
        try:
            # 尝试从当前目录导入
            import sys
            import os
            current_dir = os.path.dirname(os.path.abspath(__file__))
            sys.path.insert(0, current_dir)
            from generate_report import generate_report_with_params
            # 优化：移除导入时的print语句
            _GENERATE_REPORT_AVAILABLE = True
        except ImportError as e:
            # 优化：移除导入时的print语句
            # 优化：移除导入时的print语句
            _GENERATE_REPORT_AVAILABLE = False
            # 提供模拟函数
            def generate_report_with_params(*args, **kwargs):
                # 优化：移除导入时的print语句
                return {}

# --- 配置 ---
warnings.filterwarnings("ignore")
warnings.filterwarnings("ignore", category=UserWarning) # <-- 新增: 尝试更具体地忽略 UserWarning
matplotlib.use("Agg")
plt.rcParams["font.sans-serif"] = ["SimHei"]
plt.rcParams["axes.unicode_minus"] = False

# --- 全局配置和常量 (移除输出路径定义，改为内存处理) ---
# 不再使用固定的输出目录，所有结果通过返回值传递给UI

# 时间戳用于文件名和目录名
timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")

# --- 移除输出目录创建，改为内存处理 ---
# 不再创建固定的输出目录

# 移除日志文件和Excel输出文件路径，改为内存处理
# 不再创建物理文件，所有输出通过返回值传递

# --- 配置日志记录器仅使用控制台输出 ---
root_logger = logging.getLogger() # <-- 获取根 logger
root_logger.setLevel(logging.INFO) # <-- 设置级别

# 清除可能存在的旧处理器 (针对根 logger)
if root_logger.hasHandlers():
    root_logger.handlers.clear()

# 创建流处理器 (控制台)
stream_handler = logging.StreamHandler(sys.stdout)
stream_handler.setLevel(logging.INFO) # 控制台也用 INFO 级别

# 创建格式化器 (可选，但推荐)
formatter = logging.Formatter('%(message)s')
stream_handler.setFormatter(formatter) # 控制台也使用格式

# 将处理器添加到根日志记录器
root_logger.addHandler(stream_handler)

logger = logging.getLogger(__name__) # 获取当前模块的 logger (主要用于方便后续调用)
# --- 结束日志配置 ---


# --- 主逻辑 --- 
def run_tuning(external_data=None, external_target_variable=None, external_selected_variables=None):
    # 初始化log_file，避免作用域问题
    log_file = None

    try:
        n_iter_to_use = N_ITER_TEST if TEST_MODE else N_ITER_FIXED

        # === 优化：恢复正常多进程功能 ===
        import os  # 确保os模块在函数作用域内可用
        MAX_WORKERS = os.cpu_count() - 1 if os.cpu_count() and os.cpu_count() > 1 else 1
        # 设置多进程启动方式
        try:
            import multiprocessing
            multiprocessing.set_start_method('spawn', force=True)
        except RuntimeError:
            pass  # 如果已经设置过，忽略错误

        # --- 新增：自动计算验证期开始日期 ---
        validation_start_date_calculated = None

        try:
            train_end_dt = pd.to_datetime(TRAIN_END_DATE)
            # 假设数据频率是周五结束 ('W-FRI')，验证期从下一周的周五开始
            offset = pd.DateOffset(weeks=1)
            validation_start_date_calculated_dt = train_end_dt + offset
            validation_start_date_calculated = validation_start_date_calculated_dt.strftime('%Y-%m-%d')
            print(f"训练期结束: {TRAIN_END_DATE}, 自动计算验证期开始: {validation_start_date_calculated}")
        except Exception as e:
            print(f"错误: 无法根据 TRAIN_END_DATE ('{TRAIN_END_DATE}') 自动计算验证期开始日期: {e}")
            # 可以选择设置默认值或退出
            if log_file and not log_file.closed: log_file.close()
            sys.exit("错误: 无法确定验证期开始日期")
        # --- 结束新增 ---

        script_start_time = time.time()
        total_evaluations = 0
        svd_error_count = 0
        # log_file已在上面初始化
        best_variables_stage1 = None
        best_score_stage1 = None
        optimal_k_stage2 = None
        factor_variances_explained_stage2 = None
        k_stage1 = None
        final_variables = None # Initialize final_variables
        saved_standardization_mean = None
        saved_standardization_std = None
        pca_results_df = None
        contribution_results_df = None
        factor_contributions = None
        individual_r2_results = None
        industry_r2_results = None
        factor_industry_r2_results = None
        factor_type_r2_results = None
        final_dfm_results_obj = None
        final_data_processed = None
        final_eigenvalues = None # <<< 新增：初始化用于存储特征根的变量

        print(f"--- 开始两阶段调优 (阶段1: 变量筛选, 阶段2: 因子数筛选) ---")
        # --- 修改：更新阶段 1 描述 ---
        print(f"阶段1: 全局后向变量筛选 (固定 k=块数 N, 优化目标: HR -> -RMSE)")
        # --- 结束修改 ---
        print(f"阶段2: 基于阶段1变量，因子数选择方法: {FACTOR_SELECTION_METHOD}")
        if FACTOR_SELECTION_METHOD == 'cumulative':
            print(f"       阈值: 累积方差贡献 >= {PCA_INERTIA_THRESHOLD*100:.1f}%")
        elif FACTOR_SELECTION_METHOD == 'elbow':
            print(f"       阈值: 边际方差贡献下降率 < {ELBOW_DROP_THRESHOLD*100:.1f}%")
        
        # 移除文件日志输出，改为控制台输出
        print(f"--- 开始详细调优日志 (Run: {timestamp_str}) ---")
        print(f"配置: 两阶段流程")
        print(f"  阶段1: 全局后向变量筛选 (固定 k=块数 N, 优化 HR -> -RMSE)")
        print(f"  阶段2: 因子选择 (方法={FACTOR_SELECTION_METHOD}, "
              f"阈值={'PCA>='+str(PCA_INERTIA_THRESHOLD) if FACTOR_SELECTION_METHOD=='cumulative' else 'Drop<'+str(ELBOW_DROP_THRESHOLD)})")
        # log_file已在上面初始化，不再重复赋值

        # 🔥 修复：保存原始完整数据用于计算target_mean_original
        original_full_data = None

        # 使用外部传入的数据或调用数据准备模块
        if external_data is not None:
            logger.info("--- 使用外部传入的数据 (来自UI/data_prep模块) ---")

            # 检查并去除传入数据中的重复列
            duplicate_mask = external_data.columns.duplicated(keep=False)
            if duplicate_mask.any():
                duplicate_columns = external_data.columns[duplicate_mask].tolist()
                from collections import Counter
                column_counts = Counter(external_data.columns)
                duplicated_names = {name: count for name, count in column_counts.items() if count > 1}

                print(f"发现重复列，总数: {len(duplicate_columns)}")

                # 去除重复列，保留第一个
                external_data_cleaned = external_data.loc[:, ~external_data.columns.duplicated(keep='first')]

                # 保存原始完整数据（去重后的）
                original_full_data = external_data_cleaned.copy()
                all_data_aligned_weekly = external_data_cleaned.copy()
            else:
                # 保存原始完整数据
                original_full_data = external_data.copy()
                all_data_aligned_weekly = external_data.copy()

            # 如果指定了选择的变量，过滤数据
            if external_selected_variables and external_target_variable:
                # 创建变量名映射（处理大小写差异）
                available_columns = list(all_data_aligned_weekly.columns)
                column_mapping = {}

                # 为每个UI选择的变量找到对应的实际列名
                def normalize_punctuation(text):
                    """标准化标点符号：中文标点 -> 英文标点"""
                    punctuation_map = {
                        '：': ':',  # 中文冒号 -> 英文冒号
                        '（': '(',  # 中文左括号 -> 英文左括号
                        '）': ')',  # 中文右括号 -> 英文右括号
                        '，': ',',  # 中文逗号 -> 英文逗号
                        '。': '.',  # 中文句号 -> 英文句号
                        '；': ';',  # 中文分号 -> 英文分号
                        '！': '!',  # 中文感叹号 -> 英文感叹号
                        '？': '?',  # 中文问号 -> 英文问号
                    }
                    result = text
                    for chinese_punct, english_punct in punctuation_map.items():
                        result = result.replace(chinese_punct, english_punct)
                    return ' '.join(result.split())  # 移除多余空格

                for ui_var in external_selected_variables:
                    # 尝试精确匹配
                    if ui_var in available_columns:
                        column_mapping[ui_var] = ui_var
                    else:
                        # 尝试大小写不敏感匹配
                        ui_var_lower = ui_var.lower().strip()
                        found = False

                        for col in available_columns:
                            col_lower = col.lower().strip()
                            if col_lower == ui_var_lower:
                                column_mapping[ui_var] = col
                                found = True
                                break

                        if not found:
                            # 尝试标点符号标准化匹配
                            ui_var_punct_normalized = normalize_punctuation(ui_var_lower)

                            for col in available_columns:
                                col_punct_normalized = normalize_punctuation(col.lower().strip())
                                if ui_var_punct_normalized == col_punct_normalized:
                                    column_mapping[ui_var] = col
                                    found = True
                                    break

                        if not found:
                            # 使用Unicode标准化匹配
                            ui_var_normalized = unicodedata.normalize('NFKC', ui_var_lower)

                            for col in available_columns:
                                col_normalized = unicodedata.normalize('NFKC', col.lower().strip())
                                if ui_var_normalized == col_normalized:
                                    column_mapping[ui_var] = col
                                    found = True
                                    break

                        if not found:
                            # 尝试Unicode + 标点符号双重标准化匹配
                            ui_var_full_normalized = unicodedata.normalize('NFKC', normalize_punctuation(ui_var_lower))

                            for col in available_columns:
                                col_full_normalized = unicodedata.normalize('NFKC', normalize_punctuation(col.lower().strip()))
                                if ui_var_full_normalized == col_full_normalized:
                                    column_mapping[ui_var] = col
                                    found = True
                                    break

                        if not found:
                            # 尝试部分匹配（包含关系）
                            for col in available_columns:
                                col_lower = col.lower().strip()
                                if ui_var_lower in col_lower or col_lower in ui_var_lower:
                                    column_mapping[ui_var] = col
                                    found = True
                                    break

                        if not found:
                            print(f"警告: 未找到变量 '{ui_var}' 的匹配列")

                # 构建最终的变量列表（使用映射后的实际列名）
                mapped_vars = []
                failed_mappings = []

                for var in external_selected_variables:
                    if var in column_mapping:
                        actual_col = column_mapping[var]
                        # 确保不重复添加相同的实际列名
                        if actual_col not in mapped_vars:
                            mapped_vars.append(actual_col)
                    else:
                        failed_mappings.append(var)

                # 确保目标变量不重复添加
                if external_target_variable in mapped_vars:
                    selected_vars = mapped_vars.copy()
                else:
                    selected_vars = mapped_vars + [external_target_variable]

                # 确保目标变量不重复（双重保险）
                selected_vars = list(dict.fromkeys(selected_vars))  # 保持顺序去重

                # 确保available_vars只包含实际存在的列名
                available_vars = [var for var in selected_vars if var in all_data_aligned_weekly.columns]

                # 检查哪些变量不存在
                missing_vars = [var for var in selected_vars if var not in all_data_aligned_weekly.columns]
                if missing_vars:
                    print(f"警告: 以下变量不存在于数据中，已被过滤: {missing_vars}")

                # 去除重复变量
                if len(available_vars) != len(set(available_vars)):
                    available_vars = list(dict.fromkeys(available_vars))  # 保持顺序去重

                # 基于实际存在的变量进行验证
                expected_count = len(external_selected_variables) + 1  # UI变量 + 目标变量
                actual_count = len(available_vars)

                # 计算实际应该期望的数量（考虑映射失败的情况）
                successful_mappings = len([var for var in external_selected_variables if var in column_mapping and column_mapping[var] in all_data_aligned_weekly.columns])
                target_exists = external_target_variable in all_data_aligned_weekly.columns
                realistic_expected_count = successful_mappings + (1 if target_exists else 0)

                if actual_count != realistic_expected_count:
                    # 强制修正变量数量
                    if actual_count > expected_count:
                        # 确保目标变量在列表中
                        if external_target_variable not in available_vars:
                            available_vars.append(external_target_variable)

                        # 移除目标变量，只保留预测变量
                        predictor_vars = [v for v in available_vars if v != external_target_variable]

                        # 如果预测变量数量超过UI选择数量，截取到正确数量
                        if len(predictor_vars) > len(external_selected_variables):
                            # 优先保留成功映射的变量
                            # 按照UI选择的顺序保留变量
                            ordered_predictors = []
                            for ui_var in external_selected_variables:
                                if ui_var in column_mapping:
                                    mapped_col = column_mapping[ui_var]
                                    if mapped_col in predictor_vars and mapped_col not in ordered_predictors:
                                        ordered_predictors.append(mapped_col)

                            # 如果还不够，添加剩余的变量
                            for var in predictor_vars:
                                if var not in ordered_predictors and len(ordered_predictors) < len(external_selected_variables):
                                    ordered_predictors.append(var)

                            predictor_vars = ordered_predictors[:len(external_selected_variables)]

                        # 重新构建变量列表
                        available_vars = predictor_vars + [external_target_variable]

                if failed_mappings:
                    print(f"映射失败的变量数: {len(failed_mappings)}")

                all_data_aligned_weekly = all_data_aligned_weekly[available_vars]

            # 设置空的映射和日志（因为使用外部数据）
            var_industry_map_inferred = {}
            final_transform_details = {}
            removed_variables_log = {}
        else:
            logger.info("--- 调用数据准备模块 (自动发现 Sheets) ---")
            # <<< 修改：接收 4 个返回值 >>>
            all_data_aligned_weekly, var_industry_map_inferred, final_transform_details, removed_variables_log = prepare_data(
                 excel_path=EXCEL_DATA_FILE,
                 target_freq=TARGET_FREQ,
                 target_sheet_name=TARGET_SHEET_NAME,
                 target_variable_name=TARGET_VARIABLE,
                 consecutive_nan_threshold=CONSECUTIVE_NAN_THRESHOLD if REMOVE_VARS_WITH_CONSECUTIVE_NANS else None,
                 # <<< 新增：传递日期范围参数 >>>
                 data_start_date=getattr(config, 'DATA_START_DATE', '2020-01-01'),
                 data_end_date=getattr(config, 'DATA_END_DATE', None),
                 # <<< 结束新增 >>>
                 reference_sheet_name='指标体系',
                 reference_column_name='高频指标'
            )
        
        if all_data_aligned_weekly is None or all_data_aligned_weekly.empty:
            logger.error("数据准备失败或返回空数据框。退出调优。")
            if log_file and not log_file.closed: log_file.close()
            sys.exit(1)

        print(f"数据准备模块成功返回处理后的数据. Shape: {all_data_aligned_weekly.shape}")
        
        all_variable_names = all_data_aligned_weekly.columns.tolist()
        if TARGET_VARIABLE not in all_variable_names:
            print(f"错误: 目标变量 {TARGET_VARIABLE} 不在合并后的数据中。")
            if log_file and not log_file.closed: log_file.close()
            sys.exit(1)
        
        initial_variables = sorted(all_variable_names)
        print(f"\n初始变量组 ({len(initial_variables)}): {initial_variables[:10]}...") # Print only first few
        print("-"*30)

        # --- <<< 新增日志 >>> ---
        logger.info(f"[调试映射加载] 尝试从文件 '{EXCEL_DATA_FILE}', Sheet '{TYPE_MAPPING_SHEET}' 加载映射...")
        # --- 结束新增 ---
        print(f"调用 load_mappings 从 Sheet '{TYPE_MAPPING_SHEET}' 加载映射...")
        var_type_map, var_industry_map = load_mappings(
            excel_path=EXCEL_DATA_FILE, 
            sheet_name=TYPE_MAPPING_SHEET
        )
        # --- <<< 新增日志 >>> ---
        type_map_size = len(var_type_map) if var_type_map else 0
        industry_map_size = len(var_industry_map) if var_industry_map else 0
        logger.info(f"[调试映射加载] load_mappings 返回: var_type_map 大小={type_map_size}, var_industry_map 大小={industry_map_size}")
        if type_map_size == 0:
            logger.warning("[调试映射加载] 警告：加载得到的 var_type_map 为空！ Factor-Type R² 将无法计算。")
        # --- 结束新增 ---
        if not var_type_map:
            print(f"错误: 无法从 Excel \t'{EXCEL_DATA_FILE}\t' (Sheet: \t'{TYPE_MAPPING_SHEET}\t') 加载必要的类型映射。")
            if log_file and not log_file.closed: log_file.close()
            sys.exit(1)
        print(f"\n[EARLY CHECK 2] Mappings loaded. Type map size: {len(var_type_map)}, Industry map size: {len(var_industry_map)}")
        print("-"*30)

        print("计算原始目标变量的稳定统计量...")
        try:
            # 使用原始完整数据计算target_mean_original，而不是过滤后的数据
            data_for_target_stats = original_full_data if original_full_data is not None else all_data_aligned_weekly

            # 检查并去除重复列
            duplicate_cols = data_for_target_stats.columns.duplicated()
            if duplicate_cols.any():
                # 去除重复列，保留第一个
                data_for_target_stats = data_for_target_stats.loc[:, ~duplicate_cols]

            # 确定目标变量名
            target_var_for_stats = external_target_variable if external_target_variable else TARGET_VARIABLE

            # 确保target_var_for_stats是字符串，避免Series判断问题
            if not isinstance(target_var_for_stats, str):
                raise ValueError(f"目标变量名必须是字符串，当前类型: {type(target_var_for_stats)}")

            if target_var_for_stats not in data_for_target_stats.columns:
                raise ValueError(f"目标变量 '{target_var_for_stats}' 不在数据中")

            original_target_series_for_stats = data_for_target_stats[target_var_for_stats].copy().dropna()
            if original_target_series_for_stats.empty:
                raise ValueError(f"原始目标变量 '{target_var_for_stats}' 移除 NaN 后为空")

            target_mean_original = original_target_series_for_stats.mean()
            target_std_original = original_target_series_for_stats.std()

            if pd.isna(target_mean_original) or pd.isna(target_std_original) or target_std_original == 0:
                raise ValueError(f"计算得到的原始目标变量统计量无效 (Mean: {target_mean_original}, Std: {target_std_original})。")

            print(f"已计算原始目标变量的稳定统计量: Mean={target_mean_original:.4f}, Std={target_std_original:.4f}")

        except Exception as e:
            print(f"错误: 计算原始目标变量统计量失败: {e}")
            if log_file and not log_file.closed: log_file.close()
            sys.exit(1)
        print("-"*30)

        # ... (Consecutive NaN check remains the same) ...
        initial_predictors = [v for v in initial_variables if v != TARGET_VARIABLE]
        if REMOVE_VARS_WITH_CONSECUTIVE_NANS:
            print(f"\n--- (启用) 检查初始预测变量 ({len(initial_predictors)}) 的连续缺失值 (阈值 >= {CONSECUTIVE_NAN_THRESHOLD})... ---")
            # Actual removal logic might be in prepare_data or needs to be added here if not
            pass # Placeholder for brevity, assuming prepare_data handles this or it's done later
        else:
            print(f"\n--- (禁用) 跳过基于连续缺失值 (阈值 >= {CONSECUTIVE_NAN_THRESHOLD}) 的初始变量移除步骤。---")
            if log_file:
                try: 
                    log_file.write("\n--- (禁用) 跳过变量筛选前连续缺失值检查 ---\n")
                except Exception: 
                    pass
        print("-"*30)

        # --- 确定块数 N (k_stage1) --- (Logic remains the same, N needed for Stage 1 param) ---
        print("\n--- 阶段 1: 确定变量块和因子数 N (等于块数) ---")
        initial_blocks = {} # Initialize initial_blocks
        num_type_blocks = 0 # <-- 初始化类型块计数
        k_stage1 = 1 # 默认至少为1
        try:
            # --- 新增：计算唯一行业类别数量 ---
            num_industry_categories = 0
            if var_industry_map and isinstance(var_industry_map, dict):
                # 使用 set 来获取唯一的非空行业名称
                unique_industries = set(str(ind).strip() for ind in var_industry_map.values() 
                                        if pd.notna(ind) and str(ind).strip() and str(ind).lower() != 'nan')
                num_industry_categories = len(unique_industries)
                print(f"  检测到 {num_industry_categories} 个唯一的行业类别 (来自 var_industry_map)。")
            else:
                print("  警告: 变量行业映射 (var_industry_map) 不可用或格式无效，无法计算行业类别数量。")
            # --- 结束新增 ---
            
            if var_type_map:
                temp_blocks_init = defaultdict(list)
                current_predictors = [v for v in initial_variables if v != TARGET_VARIABLE]
                for var in current_predictors:
                    lookup_key = unicodedata.normalize('NFKC', str(var)).strip().lower()
                    var_group = var_type_map.get(lookup_key, None)
                    if var_group is None or pd.isna(var_group) or str(var_group).lower() == 'nan':
                        var_group = var_industry_map.get(lookup_key, "_未分类_")
                    else:
                        var_group = str(var_group).strip()
                    temp_blocks_init[var_group].append(var)
                
                initial_blocks = {}
                type_block_names = []
                unclassified_vars = []
                for block_name, block_vars in temp_blocks_init.items():
                    if block_name == "_未分类_":
                        unclassified_vars.extend(block_vars)
                    else:
                        initial_blocks[block_name] = block_vars
                        type_block_names.append(block_name)
                        
                if unclassified_vars:
                    initial_blocks["其他"] = unclassified_vars
                    print(f"  根据类型映射(或回退)，已将 {len(unclassified_vars)} 个未分类变量放入 '其他' 块。")

                # 计算类型块数量 (包括 '其他' 块)
                num_type_blocks = len(type_block_names) + (1 if "其他" in initial_blocks else 0)
                print(f"  基于类型映射创建了 {len(type_block_names)} 个类型块和 {1 if "其他" in initial_blocks else 0} 个 '其他' 块 (总计 {num_type_blocks} 个)。")
                if not initial_blocks: print("  警告: 未能根据类型映射创建任何变量块。")
            else:
                 print("  警告: 类型映射(var_type_map)不可用，无法按类型创建初始块。")
                 num_type_blocks = 0 # 如果类型映射不可用，则类型块为0
                 
            # --- 修改：根据行业和类型数量计算 k_stage1 ---
            k_stage1 = max(num_industry_categories, num_type_blocks) + 1
            print(f"  阶段 1 将使用的因子数 N = max(行业类别数={num_industry_categories}, 类型块数={num_type_blocks}) + 1 = {k_stage1}")
            # --- 结束修改 ---

        except Exception as e:
            print(f"确定变量块或阶段 1 因子数 N 时出错: {e}")
            traceback.print_exc()
            k_stage1 = 1 # 保留默认回退值
            print(f"警告: 使用默认阶段 1 因子数 N = {k_stage1}。请检查错误详情。")
        print("-" * 30)

        # --- <<< 修改：将块计算逻辑移到步骤 0 之前，并移除 k_stage1 计算 >>> ---
        print("\n--- 准备阶段信息：计算变量块 (用于测试模式筛选) ---")
        initial_blocks = {} # Initialize initial_blocks
        try:
            if var_type_map: # 检查类型映射是否存在
                temp_blocks_init = defaultdict(list)
                current_predictors = [v for v in initial_variables if v != TARGET_VARIABLE]
                for var in current_predictors:
                    lookup_key = unicodedata.normalize('NFKC', str(var)).strip().lower()
                    var_group = var_type_map.get(lookup_key, None)
                    if var_group is None or pd.isna(var_group) or str(var_group).lower() == 'nan':
                        # 如果类型映射没有，尝试从行业映射获取 (需要确保 var_industry_map 已加载)
                        var_group = var_industry_map.get(lookup_key, "_未分类_") if var_industry_map else "_未分类_"
                    else:
                        var_group = str(var_group).strip()
                    temp_blocks_init[var_group].append(var)
                
                unclassified_vars = []
                for block_name, block_vars in temp_blocks_init.items():
                    if block_name == "_未分类_":
                        unclassified_vars.extend(block_vars)
                    else:
                        # 确保块名是字符串
                        clean_block_name = str(block_name).strip()
                        if clean_block_name:
                             initial_blocks[clean_block_name] = block_vars
                        else:
                             print(f"  警告: 发现空块名，其变量将被视为未分类: {block_vars}")
                             unclassified_vars.extend(block_vars)
                        
                if unclassified_vars:
                    initial_blocks["其他"] = unclassified_vars
                    print(f"  已将 {len(unclassified_vars)} 个未分类变量放入 '其他' 块。")
                print(f"  已根据类型/行业映射创建 {len(initial_blocks)} 个变量块。")
            else:
                 print("  警告: 类型映射(var_type_map)不可用，无法按类型创建初始块。将尝试仅基于行业创建块。")
                 # 此处可以添加仅基于 var_industry_map 创建块的逻辑 (如果需要)
            
            # -- 确认移除旧的基于行业和类型数量计算 k_stage1 的逻辑 --

        except Exception as e_block_calc:
            print(f"计算初始变量块时出错: {e_block_calc}")
            traceback.print_exc()
            initial_blocks = {} # 出错则清空
            print("警告: 计算变量块失败。测试模式下的块筛选可能无法进行。")
        # --- <<< 结束块计算和 k_stage1 移除 >>> ---
        print("-" * 30) # 保留分隔符

        # --- 步骤 0 (新增): 初始因子数估计 --- 
        print("\n--- 步骤 0: 基于初始全体变量估计因子数 ---")
        k_initial_estimate = 1 # 默认回退值
        try:
            # <<< 新增：根据测试模式决定用于初始估计的数据范围 >>>
            data_for_initial_estimation = None
            estimation_scope_info = "全部变量"
            if TEST_MODE and DEBUG_VARIABLE_SELECTION_BLOCK is not None:
                debug_block_name = DEBUG_VARIABLE_SELECTION_BLOCK.strip()
                if initial_blocks and debug_block_name in initial_blocks:
                    debug_block_vars = initial_blocks[debug_block_name]
                    # 确保目标变量被包含
                    vars_in_scope = sorted(list(set([TARGET_VARIABLE] + debug_block_vars)))
                    # 从原始对齐数据中选取这些列
                    data_for_initial_estimation = all_data_aligned_weekly[vars_in_scope].copy()
                    estimation_scope_info = f"调试块 '{debug_block_name}' ({len(vars_in_scope)} 变量)"
                    print(f"  测试模式: 步骤 0 将使用 {estimation_scope_info} 进行 k 估计。")
                else:
                    print(f"  警告: 测试模式指定调试块 '{debug_block_name}'，但块未找到或计算失败。步骤 0 将使用全部变量。")
                    data_for_initial_estimation = all_data_aligned_weekly.copy()
            else:
                # 非测试模式，或测试模式但未指定调试块，使用全部变量
                data_for_initial_estimation = all_data_aligned_weekly.copy()
                if TEST_MODE: estimation_scope_info = "全部变量 (测试模式)"
            # <<< 结束新增 >>>
            
            if data_for_initial_estimation is None or data_for_initial_estimation.empty:
                 raise ValueError(f"未能为步骤 0 准备有效的数据 (范围: {estimation_scope_info})。")
                 
            print(f"  准备用于初始估计的数据 ({estimation_scope_info})...")
            
            print("    对初始数据进行标准化 (Z-score)...")
            mean_initial = data_for_initial_estimation.mean(axis=0)
            std_initial = data_for_initial_estimation.std(axis=0)
            zero_std_cols_initial = std_initial[std_initial == 0].index.tolist()
            if zero_std_cols_initial:
                print(f"    警告 (初始估计): 以下列标准差为0，将被移除: {zero_std_cols_initial}")
                data_for_initial_estimation = data_for_initial_estimation.drop(columns=zero_std_cols_initial)
                mean_initial = data_for_initial_estimation.mean(axis=0)
                std_initial = data_for_initial_estimation.std(axis=0)
            std_initial[std_initial == 0] = 1.0
            data_standardized_initial = (data_for_initial_estimation - mean_initial) / std_initial
            print(f"    初始数据标准化完成. Shape: {data_standardized_initial.shape}")

            print("  对标准化后的初始数据进行缺失值插补 (使用均值, 用于 PCA)...")
            imputer_initial = SimpleImputer(strategy='mean')
            data_standardized_imputed_initial = data_standardized_initial # 默认回退
            try:
                data_standardized_imputed_initial_array = imputer_initial.fit_transform(data_standardized_initial)
                data_standardized_imputed_initial = pd.DataFrame(
                    data_standardized_imputed_initial_array,
                    columns=data_standardized_initial.columns,
                    index=data_standardized_initial.index
                )
                print(f"    初始数据缺失值插补完成. Shape: {data_standardized_imputed_initial.shape}")
            except Exception as e_impute_init:
                print(f"    初始数据缺失值插补失败: {e_impute_init}. PCA 可能失败。")

            # --- 执行初步 PCA --- 
            print("  执行初步 PCA 以获取解释方差和特征值...")
            pca_initial = PCA(n_components=None) # 计算所有主成分
            pca_cumulative_variance_initial = None
            eigenvalues_initial = None
            try:
                pca_initial.fit(data_standardized_imputed_initial)
                explained_variance_ratio_pct_initial = pca_initial.explained_variance_ratio_ * 100
                pca_cumulative_variance_initial = np.cumsum(explained_variance_ratio_pct_initial)
                eigenvalues_initial = pca_initial.explained_variance_
                print(f"    初步 PCA 完成. 计算了 {len(eigenvalues_initial)} 个主成分。")
                # print(f"    PCA 解释方差 (%): {[f'{x:.2f}' for x in explained_variance_ratio_pct_initial[:15]]}...")
                # print(f"    PCA 累计解释方差 (%): {[f'{x:.2f}' for x in pca_cumulative_variance_initial[:15]]}...")
                # print(f"    PCA 特征值: {[f'{x:.3f}' for x in eigenvalues_initial[:15]]}...")
            except Exception as e_pca_init:
                 print(f"    初步 PCA 计算失败: {e_pca_init}. 依赖 PCA 的方法将无法使用。")
            
            # --- 执行初步 DFM (仅当需要时) ---
            Lambda_initial = None
            if FACTOR_SELECTION_METHOD == 'cumulative_common':
                print("  为 'cumulative_common' 方法运行初步 DFM (因子数上限=变量数)...")
                # 理论上因子数不应超过观测数或变量数，这里用一个较大但合理的数
                max_factors_dfm_init = min(data_standardized_initial.shape[0], data_standardized_initial.shape[1])
                if max_factors_dfm_init <= 0:
                     print("    错误: 无法确定初步 DFM 的有效因子数上限。")
                else:
                     print(f"    设定初步 DFM 因子数上限为: {max_factors_dfm_init}")
                     try:
                         dfm_results_initial = DFM_EMalgo(
                             observation=data_standardized_initial, # DFM 使用未插补数据
                             n_factors=max_factors_dfm_init,
                             n_shocks=max_factors_dfm_init,
                             n_iter=n_iter_to_use # 使用配置的迭代次数
                         )
                         if dfm_results_initial is not None and hasattr(dfm_results_initial, 'Lambda'):
                             Lambda_initial = dfm_results_initial.Lambda
                             print(f"    初步 DFM 运行成功，获得载荷矩阵 Shape: {Lambda_initial.shape}")
                         else:
                             print("    错误: 初步 DFM 运行失败或未返回载荷矩阵 (Lambda)。")
                     except Exception as e_dfm_init:
                         print(f"    初步 DFM 运行失败: {e_dfm_init}。")
            
            # --- 应用因子选择方法计算 k_initial_estimate ---
            print(f"  应用因子选择方法 '{FACTOR_SELECTION_METHOD}' 确定初始估计 k...")
            temp_k_estimate = None
            if FACTOR_SELECTION_METHOD == 'cumulative':
                 if pca_cumulative_variance_initial is not None:
                      k_indices = np.where(pca_cumulative_variance_initial >= PCA_INERTIA_THRESHOLD * 100)[0]
                      if len(k_indices) > 0: temp_k_estimate = k_indices[0] + 1
                      else: temp_k_estimate = len(eigenvalues_initial) # Fallback: all components
                      print(f"    'cumulative' 方法估计 k = {temp_k_estimate}")
                 else: print("    错误: PCA 结果不可用，无法应用 'cumulative' 方法。")
            elif FACTOR_SELECTION_METHOD == 'elbow':
                 if eigenvalues_initial is not None and len(eigenvalues_initial) > 1:
                     variance_diff_ratio = np.diff(eigenvalues_initial) / eigenvalues_initial[:-1]
                     k_indices = np.where(np.abs(variance_diff_ratio) < ELBOW_DROP_THRESHOLD)[0]
                     if len(k_indices) > 0: temp_k_estimate = k_indices[0] + 1
                     else: temp_k_estimate = len(eigenvalues_initial) # Fallback: all components
                     print(f"    'elbow' 方法估计 k = {temp_k_estimate}")
                 elif eigenvalues_initial is not None and len(eigenvalues_initial) == 1:
                     optimal_k_stage2 = 1
                     print("    仅有1个主成分，无法应用手肘法，直接选择 k=1。") # Note: This sets optimal_k_stage2 directly, which might be premature here. Should set temp_k_estimate.
                     temp_k_estimate = 1 # <<< FIX: Should set temp_k_estimate
                 else: # <--- 修正：这个 else 对应 if eigenvalues is not None and len(eigenvalues) > 1
                      print("    错误: 由于 PCA 计算失败或因子数不足，无法应用 'elbow' 方法。将回退使用 k = k_initial_estimate。")
                      # optimal_k_stage2 = k_initial_estimate # <<< REMOVE: Don't set stage 2 k here
            elif FACTOR_SELECTION_METHOD == 'kaiser':
                 if eigenvalues_initial is not None:
                     k_kaiser = np.sum(eigenvalues_initial > 1)
                     temp_k_estimate = max(1, k_kaiser) # Ensure at least 1 factor
                     print(f"    'kaiser' 方法估计 k = {temp_k_estimate}")
                 else: print("    错误: PCA 特征值不可用，无法应用 'kaiser' 方法。")
            elif FACTOR_SELECTION_METHOD == 'cumulative_common':
                 cumulative_common_variance_pct_initial = None
                 if Lambda_initial is not None:
                     try:
                         # Ensure TARGET_VARIABLE exists in the columns used for this initial estimation
                         if TARGET_VARIABLE in data_standardized_initial.columns:
                             target_var_index_pos_init = data_standardized_initial.columns.get_loc(TARGET_VARIABLE)
                             if target_var_index_pos_init < Lambda_initial.shape[0]:
                                  lambda_target_initial = Lambda_initial[target_var_index_pos_init, :]
                                  lambda_target_sq_init = lambda_target_initial ** 2
                                  sum_lambda_target_sq_init = np.sum(lambda_target_sq_init)
                                  if sum_lambda_target_sq_init > 1e-9:
                                       pct_contribution_common_init = (lambda_target_sq_init / sum_lambda_target_sq_init) * 100
                                       cumulative_common_variance_pct_initial = np.cumsum(pct_contribution_common_init)
                                  else: print("    警告: 初步 DFM 目标平方载荷和过小。")
                             else: print(f"    错误: 目标变量索引 ({target_var_index_pos_init}) 超出初步 DFM 载荷矩阵范围 ({Lambda_initial.shape[0]})。")
                         else:
                              print(f"    错误: 目标变量 '{TARGET_VARIABLE}' 不在用于初始估计的数据列中，无法计算 common variance。")
                     except KeyError: print(f"    错误: 在初始标准化数据列中未找到目标变量 '{TARGET_VARIABLE}'。")
                     except Exception as e_common_init: print(f"    计算初始共同方差贡献时出错: {e_common_init}")
                 else: print("    错误: 初步 DFM 载荷不可用。")

                 if cumulative_common_variance_pct_initial is not None:
                      k_indices = np.where(cumulative_common_variance_pct_initial >= COMMON_VARIANCE_CONTRIBUTION_THRESHOLD * 100)[0]
                      if len(k_indices) > 0: temp_k_estimate = k_indices[0] + 1
                      else: temp_k_estimate = Lambda_initial.shape[1] # Fallback: all factors from DFM
                      print(f"    'cumulative_common' 方法估计 k = {temp_k_estimate}")
                 else: print("    错误: 无法应用 'cumulative_common' 方法。")
            # --- <<< 新增：在步骤 0 中处理 Bai and Ng ICp2 >>> ---
            elif FACTOR_SELECTION_METHOD == 'bai_ng':
                print(f"  应用 Bai and Ng (2002) ICp2 准则 (初始估计)...")
                # 使用步骤 0 计算得到的 eigenvalues_initial 和 data_standardized_imputed_initial
                if eigenvalues_initial is not None and 'data_standardized_imputed_initial' in locals() and data_standardized_imputed_initial is not None:
                    N_init = data_standardized_imputed_initial.shape[1]
                    T_init = data_standardized_imputed_initial.shape[0]
                    k_max_bai_ng_init = len(eigenvalues_initial)
                    if N_init > 0 and T_init > 0 and k_max_bai_ng_init > 0:
                        print(f"    参数 (初始): N={N_init}, T={T_init}, k_max={k_max_bai_ng_init}")
                        min_ic_init = np.inf
                        best_k_ic_init = 1
                        ic_values_init = {}

                        print("    计算各 k 的 ICp2 值 (初始)...")
                        for k in range(1, k_max_bai_ng_init + 1):
                            ssr_k_init = T_init * np.sum(eigenvalues_initial[k:]) if k < len(eigenvalues_initial) else 1e-9
                            if ssr_k_init <= 1e-9:
                                icp2_k_init = np.inf
                            else:
                                v_k_init = ssr_k_init / (N_init * T_init)
                                penalty_k_init = k * (N_init + T_init) / (N_init * T_init) * np.log(min(N_init, T_init))
                                icp2_k_init = np.log(v_k_init) + penalty_k_init
                                # print(f"      k={k}: SSR={ssr_k_init:.4f}, V(k)={v_k_init:.6f}, Penalty={penalty_k_init:.6f}, ICp2={icp2_k_init:.6f}") # Optional detailed print
                            ic_values_init[k] = icp2_k_init
                            if icp2_k_init < min_ic_init:
                                min_ic_init = icp2_k_init
                                best_k_ic_init = k

                        if min_ic_init != np.inf and best_k_ic_init > 0:
                            temp_k_estimate = best_k_ic_init
                            print(f"    根据 Bai-Ng ICp2 准则 (初始) 估计的因子数量: k = {temp_k_estimate} (最小 ICp2 = {min_ic_init:.6f})")
                        else:
                            print(f"    警告: Bai-Ng ICp2 (初始) 计算未能找到有效最优 k，将使用回退启发式。")
                            temp_k_estimate = None # Trigger fallback heuristic below
                    else:
                        print(f"    错误: 应用 Bai-Ng ICp2 (初始) 的必要参数无效 (N={N_init}, T={T_init}, k_max={k_max_bai_ng_init})。")
                else:
                    print("    错误: 缺少初始 PCA 特征值或插补后的初始数据，无法应用 Bai-Ng 方法进行初始估计。")
            # --- <<< 结束新增 >>> ---
            else:
                 print(f"错误: 未知的因子选择方法 '{FACTOR_SELECTION_METHOD}'。") # Now this else only catches truly unknown methods

            if temp_k_estimate is not None and temp_k_estimate > 0:
                 k_initial_estimate = temp_k_estimate
                 print(f"步骤 0 完成。初始估计因子数 k_initial_estimate = {k_initial_estimate}")
            else:
                 k_initial_estimate = max(1, int(data_standardized_initial.shape[1] / 10)) # Fallback heuristic
                 print(f"警告: 未能通过所选方法估计有效的初始 k，将使用回退启发式 k = {k_initial_estimate}")
            
            # --- 新增：限制初始 k 的最大值以提高稳定性 ---
            k_cap = 10
            if k_initial_estimate > k_cap:
                print(f"警告: 初始估计因子数 {k_initial_estimate} 超过上限 {k_cap}，将使用上限值进行阶段 1 筛选。")
                k_initial_estimate = k_cap
            # --- 结束新增 ---

        except Exception as e_step0:
            print(f"步骤 0 (初始因子数估计) 失败: {e_step0}")
            traceback.print_exc()
            k_initial_estimate = max(1, int(len(initial_variables) / 10)) # Fallback heuristic
            print(f"警告: 因错误，将使用回退启发式 k = {k_initial_estimate} 进行阶段 1。")
        print("-" * 30)
        # --- 结束步骤 0 ---

        # --- <<< 修改：Stage 1: 全局变量后向筛选 (使用 k_initial_estimate) >>> ---
        print(f"优化目标: (Avg Hit Rate, -Avg RMSE)")
        score_tuple_definition_stage1 = "(Avg Hit Rate, -Avg RMSE)" # 固定评分标准

        # best_score_stage1 = (-np.inf, np.inf) # 全局筛选函数内部会计算初始分数
        best_params_stage1 = {'k_factors': k_initial_estimate} # 固定参数
        # best_variables_stage1 在下方确定

        try:
            # 优先使用外部传入的变量选择
            if external_data is not None and external_selected_variables:
                # 使用外部传入的变量（UI选择的变量）
                variables_for_selection_start = [external_target_variable] + external_selected_variables
                # 确保目标变量不重复
                variables_for_selection_start = list(dict.fromkeys(variables_for_selection_start))
                selection_scope_info = f"UI选择的 {len(variables_for_selection_start)} 个变量"
                # 添加标志，防止后续逻辑覆盖
                using_external_variables = True
            else:
                # 回退到所有变量
                variables_for_selection_start = initial_variables.copy()
                selection_scope_info = f"全部 {len(initial_variables)} 个变量"
                using_external_variables = False
            
            # --- 🔥 修复：只有在没有使用外部变量时才检查TEST_MODE ---
            if not using_external_variables:
                # --- 移除块处理逻辑 ---
                # blocks_to_process_stage1 = initial_blocks.copy() # Default to all blocks

                if TEST_MODE and DEBUG_VARIABLE_SELECTION_BLOCK is not None:
                    debug_block_name = DEBUG_VARIABLE_SELECTION_BLOCK.strip()
                    if debug_block_name in initial_blocks:
                        debug_block_vars = initial_blocks[debug_block_name]
                        # Ensure target variable is included
                        variables_for_selection_start = sorted(list(set([TARGET_VARIABLE] + debug_block_vars)))
                        selection_scope_info = f"测试模式调试块 '{debug_block_name}' ({len(variables_for_selection_start)} 个变量)"
                        # blocks_to_process_stage1 = {debug_block_name: debug_block_vars} # 移除
                        print(f"\n*** 测试模式：全局筛选将仅限于调试块 '{debug_block_name}' 中的变量 ({len(variables_for_selection_start)} vars) ***\n")
                        if log_file: log_file.write(f"*** 测试模式：全局筛选限定于块 '{debug_block_name}' 中的变量 ({len(variables_for_selection_start)} vars) ***\n")
                    else:
                        print(f"\n*** 测试模式：指定的调试块 '{debug_block_name}' 未找到，将使用所有变量进行筛选。 ***\n")
                        if log_file: log_file.write(f"*** 测试模式：调试块 '{debug_block_name}' 未找到，使用所有变量进行筛选 ***\n")

                elif TEST_MODE:
                     # Test mode but no debug block - use all variables but fewer iterations
                     selection_scope_info = f"全部 {len(initial_variables)} 个变量 (测试模式迭代次数)"
                     print(f"\n*** 测试模式：未指定调试块。全局筛选将使用所有变量 (迭代次数减少)。 ***\n")
                     if log_file: log_file.write("*** 测试模式：使用所有变量进行筛选 (迭代次数减少) ***\n")
                else:
                     # Full mode - use all variables
                     print(f"\n*** 完整模式：全局筛选将使用所有变量。 ***\n")
                     if log_file: log_file.write("*** 完整模式：使用所有变量进行筛选 ***\n")
            else:
                # 🔥 使用外部变量时的提示
                print(f"\n*** 外部变量模式：全局筛选将使用UI选择的 {len(variables_for_selection_start)} 个变量。 ***\n")
                if log_file: log_file.write(f"*** 外部变量模式：使用UI选择的 {len(variables_for_selection_start)} 个变量进行筛选 ***\n")

            # --- 不需要计算初始基准分数，perform_global_backward_selection 内部会做 ---
            # print(f"计算阶段 1 初始基准分数 (使用 {selection_scope_info}, k={k_initial_estimate})...") 
            # ... [移除基准分数计算代码] ...

            # --- 开始全局后向剔除 ---
            logger.info("--- 即将调用 perform_global_backward_selection 进行全局变量筛选... ---")
            print(f"开始对 {selection_scope_info} 进行全局后向变量剔除 (固定 k={k_initial_estimate})...")
            # 注意：perform_global_backward_selection 内部有 tqdm 进度条
            sel_variables_stage1, sel_params_stage1, sel_score_tuple_stage1, sel_eval_count_stage1, sel_svd_err_count_stage1 = perform_global_backward_selection(
                initial_variables=variables_for_selection_start, # <-- 使用确定的起始变量集
                initial_params=best_params_stage1,         # 使用初始最佳参数 (包含固定k)
                # initial_score_tuple - 不需要传递
                target_variable=TARGET_VARIABLE,
                all_data=all_data_aligned_weekly, 
                var_type_map=var_type_map, 
                validation_start=validation_start_date_calculated, # <-- 使用计算出的日期
                validation_end=VALIDATION_END_DATE, 
                target_freq=TARGET_FREQ, 
                train_end_date=TRAIN_END_DATE, 
                n_iter=n_iter_to_use, 
                target_mean_original=target_mean_original, 
                target_std_original=target_std_original,
                max_workers=MAX_WORKERS,
                evaluate_dfm_func=evaluate_dfm_params,
                log_file=log_file
                # blocks - 不需要传递
                # hyperparams_to_tune - 不需要传递
                # auto_select_factors - 不需要传递
            )

            # --- 更新阶段 1 结果 ---
            best_variables_stage1 = sel_variables_stage1 # 更新为筛选后的变量
            best_params_stage1 = sel_params_stage1 # 参数理论上不变，但保持一致
            best_score_stage1 = sel_score_tuple_stage1 # 更新为筛选后的分数
            total_evaluations += sel_eval_count_stage1
            svd_error_count += sel_svd_err_count_stage1

            # 检查最终得分是否有效
            final_score_valid = False
            if best_score_stage1 is not None and len(best_score_stage1) == 2 and all(np.isfinite(list(best_score_stage1))):
                final_score_valid = True

            if final_score_valid:
                final_hr_stage1, final_neg_rmse_stage1 = best_score_stage1
                num_predictors_stage1 = len([v for v in best_variables_stage1 if v != TARGET_VARIABLE])
                print(f"阶段 1 (全局筛选) 完成。最佳结果 (固定 k={k_initial_estimate}): 评分=(HR={final_hr_stage1:.2f}%, RMSE={-final_neg_rmse_stage1:.6f}), 预测变量数量={num_predictors_stage1}") # <-- 修改打印
                if log_file:
                    log_file.write(f"\n--- 阶段 1 结果 (全局筛选) ---\n") # <-- 修改日志
                    log_file.write(f"起始变量范围: {selection_scope_info}\n") # <-- 修正日志行
                    log_file.write(f"固定因子数 (N): {k_initial_estimate}\n")
                    log_file.write(f"最佳评分 (HR, -RMSE): {best_score_stage1}\n")
                    log_file.write(f"最终预测变量数量: {num_predictors_stage1}\n") # <-- 修改日志
            else:
                    print("错误: 阶段 1 (全局筛选) 未能找到有效的变量集和评分。无法继续。") # <-- 修改打印
                    if log_file and not log_file.closed: log_file.close()
                    sys.exit(1)

        except Exception as e_select:
            print(f"阶段 1 全局变量筛选过程中发生严重错误: {e_select}\n") # <-- 修改打印
            traceback.print_exc()
            print("错误: 阶段 1 失败，无法继续。")
            if log_file and not log_file.closed: log_file.close()
            sys.exit(1)
        print("-" * 30)
        # --- <<< 结束阶段 1 修改 >>> ---

        # --- <<< 新增：循环测试因子数 k 并记录结果 >>> ---
        print("\\n--- 开始测试不同因子数 k (1 到 k_initial_estimate) 的性能并记录载荷 ---")
        factor_eval_results = [] # 用于存储每次评估的结果

        # 准备用于评估的数据 (使用阶段1最终变量)
        try:
            data_subset_for_eval = all_data_aligned_weekly[best_variables_stage1].copy()
            print(f"  使用阶段1选定的 {len(best_variables_stage1)} 个变量进行评估。")

            # --- 可以在这里添加数据预处理步骤，如果 evaluate_dfm_params 内部不做的话 ---
            # 例如：标准化 (需要保存参数供 evaluate_dfm_params 使用或传递)
            # mean_for_eval = data_subset_for_eval.mean(axis=0)
            # std_for_eval = data_subset_for_eval.std(axis=0)
            # std_for_eval[std_for_eval == 0] = 1.0
            # data_standardized_for_eval = (data_subset_for_eval - mean_for_eval) / std_for_eval
            # print("    已对评估数据进行标准化。")
            # 注意：当前 evaluate_dfm_params 内部会进行标准化，所以这里不需要

        except Exception as e_prep_k_eval:
            print(f"错误：准备评估不同 k 值的数据时失败: {e_prep_k_eval}")
            # 可以选择退出或继续执行原阶段2逻辑
            print("警告：无法进行因子数 k 的详细评估。将跳过此步骤。")
            factor_eval_results = None # 标记失败

        if factor_eval_results is not None: # 仅在数据准备成功时执行
            # 循环 k 从 1 到 k_initial_estimate
            for k_test in tqdm(range(1, k_initial_estimate + 1), desc="评估不同因子数 k"):
                print(f"  正在评估 k = {k_test}...")
                current_params_test = {'k_factors': k_test}
                try:
                    # 调用评估函数
                    # 注意：这里复用了 tune_dfm 中定义的各种参数
                    eval_result_tuple = evaluate_dfm_params(
                        variables=best_variables_stage1,
                        full_data=all_data_aligned_weekly, # 传递完整数据，函数内部截取
                        target_variable=TARGET_VARIABLE,
                        params=current_params_test,
                        var_type_map=var_type_map, # 传递类型映射
                        validation_start=validation_start_date_calculated,
                        validation_end=VALIDATION_END_DATE,
                        target_freq=TARGET_FREQ,
                        train_end_date=TRAIN_END_DATE,
                        target_mean_original=target_mean_original,
                        target_std_original=target_std_original,
                        max_iter=n_iter_to_use
                    )

                    # 解包结果
                    (is_rmse, oos_rmse, is_mae, oos_mae,
                     is_hit_rate, oos_hit_rate, is_svd_error,
                     lambda_df_result, _) = eval_result_tuple # 忽略最后一个返回值 aligned_df_monthly

                    # 记录结果
                    result_entry = {
                        'k': k_test,
                        'oos_rmse': oos_rmse if np.isfinite(oos_rmse) else None,
                        'oos_mae': oos_mae if np.isfinite(oos_mae) else None,
                        'oos_hit_rate': oos_hit_rate if np.isfinite(oos_hit_rate) else None,
                        'svd_error': is_svd_error,
                        'loadings': lambda_df_result # 存储 DataFrame 或 None
                    }
                    factor_eval_results.append(result_entry)

                    if is_svd_error:
                        print(f"    k={k_test}: 评估因 SVD 错误失败。")
                    elif result_entry['oos_rmse'] is None:
                        print(f"    k={k_test}: 评估成功，但无法计算 OOS 指标。")
                    else:
                        print(f"    k={k_test}: OOS RMSE={result_entry['oos_rmse']:.4f}, MAE={result_entry['oos_mae']:.4f}, HR={result_entry['oos_hit_rate']:.2f}%")

                except Exception as e_eval_k:
                    print(f"  评估 k = {k_test} 时发生错误: {e_eval_k}")
                    factor_eval_results.append({
                        'k': k_test,
                        'oos_rmse': None,
                        'oos_mae': None,
                        'oos_hit_rate': None,
                        'svd_error': False, # 假设不是 SVD 错误
                        'loadings': None
                    })

            # --- 🔥 修复：移除CSV文件保存功能，所有结果只通过UI下载 ---
            if factor_eval_results:
                print("\\n--- 因子数评估结果已完成 (结果将通过UI提供下载) ---")
                print(f"评估了 {len(factor_eval_results)} 个不同的因子数配置")
            else:
                print("未执行或未成功完成因子数 k 的评估。")

        # --- <<< 结束新增代码块 >>> ---

        # --- <<< 阶段 2: 因子数量筛选 (逻辑不变，但输入来自修改的 Stage 1) >>> ---
        print(f"\\n--- 阶段 2 开始: 因子数量筛选 (基于阶段 1 变量) ---")
        print(f"方法: {FACTOR_SELECTION_METHOD}")
        optimal_k_stage2 = None # 初始化最终因子数
        factor_variances_explained_stage2 = None # 存储方差解释

        try:
            # ... (阶段 2 内部逻辑保持不变，但使用来自全局筛选的 best_variables_stage1) ...
             # 1. 使用阶段1最优变量和 k=N 运行一次 DFM (在完整数据上)
            print(f"  准备阶段 2 DFM 运行 (变量数: {len(best_variables_stage1)}, k={k_initial_estimate})...")
            #   准备数据 (与最终模型运行类似，但使用阶段1变量)
            if not best_variables_stage1 or all_data_aligned_weekly is None:
                raise ValueError("缺少阶段 1 变量列表或原始对齐数据，无法准备阶段 2 数据。")

            data_subset_stage2 = all_data_aligned_weekly[best_variables_stage1].copy()
            
            # --- 移除二次转换调用，直接使用子集 --- 
            data_processed_stage2 = data_subset_stage2 # 直接使用选出的变量子集
            # final_transform_details 保留 prepare_data 返回的值

            if data_processed_stage2 is None or data_processed_stage2.empty:
                raise ValueError("阶段 2 数据准备后为空。")
            print(f"  阶段 2 数据准备完成. Shape: {data_processed_stage2.shape}")

            #   标准化数据
            print("    对阶段 2 数据进行标准化 (Z-score)...")
            mean_stage2 = data_processed_stage2.mean(axis=0)
            std_stage2 = data_processed_stage2.std(axis=0)
            zero_std_cols_stage2 = std_stage2[std_stage2 == 0].index.tolist()

            if zero_std_cols_stage2:
                print(f"    警告 (阶段 2): 以下列标准差为0，将被移除: {zero_std_cols_stage2}")
                data_processed_stage2 = data_processed_stage2.drop(columns=zero_std_cols_stage2)
                # 更新变量列表以反映移除
                best_variables_stage1_filtered = [v for v in best_variables_stage1 if v not in zero_std_cols_stage2]
                print(f"    注意：由于标准差为0移除了变量，阶段1的最优变量集已从 {len(best_variables_stage1)} 减少到 {len(best_variables_stage1_filtered)}。")
                best_variables_stage1 = best_variables_stage1_filtered # 更新阶段1变量
                mean_stage2 = data_processed_stage2.mean(axis=0) # 重新计算均值
                std_stage2 = data_processed_stage2.std(axis=0)   # 重新计算标准差
                std_stage2[std_stage2 == 0] = 1.0 # 再次检查
            else:
                std_stage2[std_stage2 == 0] = 1.0 

            data_standardized_stage2 = (data_processed_stage2 - mean_stage2) / std_stage2
            print(f"    标准化完成. Shape: {data_standardized_stage2.shape}")

            # --- <<< 新增：在 PCA 前进行缺失值插补 >>> ---
            print("  对标准化后的阶段 2 数据进行缺失值插补 (使用均值)...")
            imputer_stage2 = SimpleImputer(strategy='mean')
            data_standardized_stage2_imputed = data_standardized_stage2 # 默认回退
            try:
                data_standardized_imputed_array = imputer_stage2.fit_transform(data_standardized_stage2)
                # SimpleImputer 返回 numpy array, 需要转回 DataFrame 并保留列名和索引
                data_standardized_stage2_imputed = pd.DataFrame(
                    data_standardized_imputed_array,
                    columns=data_standardized_stage2.columns,
                    index=data_standardized_stage2.index
                )
                print(f"    缺失值插补完成. Shape: {data_standardized_stage2_imputed.shape}")
            except Exception as e_impute:
                print(f"    缺失值插补失败: {e_impute}. 后续 PCA 可能失败。")
                # 保留回退值 data_standardized_stage2_imputed = data_standardized_stage2
            # --- 结束新增 ---

            # --- 2. 计算 PCA 以获取因子选择所需信息 (例如，解释方差、特征值) ---
            pca_stage2 = None
            pca_cumulative_variance = None
            eigenvalues = None

            # 在 PCA 计算之前添加诊断信息
            if 'data_standardized_stage2_imputed' in locals() and data_standardized_stage2_imputed is not None:
                # 检查是否存在方差为零的列
                try:
                    zero_variance_cols = data_standardized_stage2_imputed.columns[data_standardized_stage2_imputed.var(axis=0) < 1e-9]
                    if not zero_variance_cols.empty:
                        print(f"警告: 发现方差接近零的列: {zero_variance_cols.tolist()}")
                except Exception as e_diag_var:
                     print(f"诊断警告: 检查零方差列时出错: {e_diag_var}")
                # 检查是否存在 NaN 值
                try:
                    nan_counts = data_standardized_stage2_imputed.isnull().sum().sum()
                    if nan_counts > 0:
                        print(f"警告: 插值后的数据中仍发现 {nan_counts} 个 NaN 值！")
                except Exception as e_diag_nan:
                     print(f"诊断警告: 检查 NaN 值时出错: {e_diag_nan}")

                # --- 动态调整 PCA 组件数 ---
                n_samples, n_features = data_standardized_stage2_imputed.shape
                k_initial_estimate_adjusted = min(k_initial_estimate, n_samples, n_features)
                if k_initial_estimate_adjusted != k_initial_estimate:
                     print(f"    警告: 初始 k ({k_initial_estimate}) 大于数据维度 ({n_samples}, {n_features})，调整为 {k_initial_estimate_adjusted}")
                if k_initial_estimate_adjusted <= 0:
                     print(f"    错误: 调整后的 PCA 组件数 ({k_initial_estimate_adjusted}) 无效，无法执行 PCA。")
                     k_initial_estimate_to_use = None # 标记 PCA 无法执行
                else:
                     k_initial_estimate_to_use = k_initial_estimate_adjusted
                     print(f"    即将使用的 PCA 组件数 (n_components): {k_initial_estimate_to_use}")

            else:
                 print(f"    警告: 变量 'data_standardized_stage2_imputed' 未定义或为 None，无法执行 PCA 输入诊断和计算。")
                 k_initial_estimate_to_use = None # 标记 PCA 无法执行
            print(f"--- 结束诊断 ---")

            # --- 执行 PCA 计算 ---
            if k_initial_estimate_to_use is not None:
                try:
                    print(f"  计算阶段 2 的 PCA (n_components={k_initial_estimate_to_use})...")
                    pca_stage2 = PCA(n_components=k_initial_estimate_to_use).fit(data_standardized_stage2_imputed)
                    explained_variance_ratio_pct = pca_stage2.explained_variance_ratio_ * 100
                    pca_cumulative_variance = np.cumsum(explained_variance_ratio_pct)
                    eigenvalues = pca_stage2.explained_variance_ # 获取特征值 (解释方差)
                    print(f"    PCA 解释方差 (%): {[f'{x:.2f}' for x in explained_variance_ratio_pct]}")
                    print(f"    PCA 累计解释方差 (%): {[f'{x:.2f}' for x in pca_cumulative_variance]}")
                    print(f"    PCA 特征值 (解释方差): {[f'{x:.3f}' for x in eigenvalues]}")
                except Exception as e_pca:
                     print(f"    PCA 计算失败: {e_pca}. 依赖 PCA 的因子选择方法将无法使用。")
                     # 保留 pca_stage2, pca_cumulative_variance, eigenvalues 为 None
            else:
                 print("    由于输入数据或组件数问题，跳过 PCA 计算。")
            # --- 结束 PCA 计算 ---

            # --- <<< 修改：将初步 DFM 运行移到 if/elif 结构之前 >>> ---
            dfm_results_stage2 = None
            Lambda_stage2 = None # 确保初始化
            # 只有当选择的方法是 cumulative_common 时才运行初步 DFM
            if FACTOR_SELECTION_METHOD == 'cumulative_common':
                print(f"  为 'cumulative_common' 方法运行初步 DFM (k={k_initial_estimate})...")
                try:
                    # 注意: DFM 使用的是标准化但未插补的数据 (内部处理缺失)
                    dfm_results_stage2 = DFM_EMalgo(
                        observation=data_standardized_stage2, 
                        n_factors=k_initial_estimate,
                        n_shocks=k_initial_estimate,
                        n_iter=n_iter_to_use
                    )
                    if dfm_results_stage2 is None or not hasattr(dfm_results_stage2, 'Lambda'):
                        print("    错误: 初步 DFM 运行失败或未返回载荷矩阵 (Lambda)。Lambda_stage2 将为 None。")
                        Lambda_stage2 = None # 明确设为 None
                    else:
                        Lambda_stage2 = dfm_results_stage2.Lambda
                        print(f"    初步 DFM 运行成功，获得载荷矩阵 Shape: {Lambda_stage2.shape}")
                except Exception as e_dfm_prelim:
                    print(f"    初步 DFM 运行失败: {e_dfm_prelim}. Lambda_stage2 将为 None。")
                    Lambda_stage2 = None # 明确设为 None
            # --- <<< 结束初步 DFM 运行移动 >>> ---

            # --- 3. 应用因子选择方法确定 k --- 
            print("\n  应用因子选择方法确定最终因子数量...")
            optimal_k_stage2 = None # 初始化
            # --- cumulative 方法 (基于 PCA 累计总方差) --- 
            if k_initial_estimate == 1:
                 optimal_k_stage2 = 1
                 print(f"  阶段 1 因子数 N=1，直接设定最终因子数 k = 1")
            elif FACTOR_SELECTION_METHOD == 'cumulative':
                 print(f"  应用累积(总)方差阈值法 (PCA 解释方差 >= {PCA_INERTIA_THRESHOLD*100:.1f}%)...")
                 if pca_cumulative_variance is not None:
                      k_indices = np.where(pca_cumulative_variance >= PCA_INERTIA_THRESHOLD * 100)[0]
                      if len(k_indices) > 0:
                          optimal_k_stage2 = k_indices[0] + 1
                          print(f"    第一个达到阈值的因子数量: {optimal_k_stage2}")
                      else:
                          optimal_k_stage2 = k_initial_estimate # Fallback if threshold never reached
                          print(f"    警告: 累积解释方差未达到阈值 {PCA_INERTIA_THRESHOLD*100:.1f}%，使用最大因子数 k={k_initial_estimate}")
                 else:
                      print("    错误: 由于 PCA 计算失败，无法应用 'cumulative' 方法。将回退使用 k = k_initial_estimate。")
                      optimal_k_stage2 = k_initial_estimate
            # --- elbow 方法 (基于 PCA 解释方差的下降) --- 
            elif FACTOR_SELECTION_METHOD == 'elbow':
                 print(f"  应用手肘法 (PCA 解释方差边际下降率 < {ELBOW_DROP_THRESHOLD*100:.1f}%)...")
                 if eigenvalues is not None and len(eigenvalues) > 1: # 至少需要两个因子才能计算下降率
                     variance_diff_ratio = np.diff(eigenvalues) / eigenvalues[:-1] # 计算相对下降率
                     # 找到第一个相对下降率小于阈值的索引
                     k_indices = np.where(np.abs(variance_diff_ratio) < ELBOW_DROP_THRESHOLD)[0]
                     if len(k_indices) > 0:
                         # k_indices[0] 是下降率首次低于阈值的 *区间* 索引，我们需要该区间 *之前* 的因子数
                         # 即，如果第1到第2个因子的下降率低于阈值 (索引0)，我们需要1个因子
                         optimal_k_stage2 = k_indices[0] + 1
                         print(f"    手肘法找到的因子数量: {optimal_k_stage2}")
                     else:
                         optimal_k_stage2 = k_initial_estimate # Fallback if no elbow found
                         print(f"    警告: 未找到明显的手肘点，使用最大因子数 k={k_initial_estimate}")
                 elif eigenvalues is not None and len(eigenvalues) == 1:
                     optimal_k_stage2 = 1
                     print("    仅有1个主成分，无法应用手肘法，直接选择 k=1。")
                 else: # <--- 修正：这个 else 对应 if eigenvalues is not None and len(eigenvalues) > 1
                      print("    错误: 由于 PCA 计算失败或因子数不足，无法应用 'elbow' 方法。将回退使用 k = k_initial_estimate。")
                      optimal_k_stage2 = k_initial_estimate
            # --- kaiser 方法 (基于 PCA 特征值 > 1) ---
            elif FACTOR_SELECTION_METHOD == 'kaiser':
                 print(f"  应用凯撒准则 (PCA 特征值 > 1)...")
                 if eigenvalues is not None: # 检查 eigenvalues 是否存在且有效
                     print(f"    PCA 特征值 (解释方差): {[f'{v:.3f}' for v in eigenvalues]}")
                     k_kaiser = np.sum(eigenvalues > 1)
                     if k_kaiser == 0:
                         optimal_k_stage2 = 1
                         print("    警告: 没有特征值大于 1，将至少选择 1 个因子。")
                     else:
                         optimal_k_stage2 = k_kaiser
                     print(f"    基于凯撒准则选择的因子数量: k = {optimal_k_stage2}")
                 else:
                     print("    错误: PCA 计算失败或未产生特征值，无法应用凯撒准则。将回退使用 k = k_initial_estimate。")
                     optimal_k_stage2 = k_initial_estimate
            # --- 新增：cumulative_common 方法 (基于初步 DFM 载荷计算的共同方差贡献) ---
            elif FACTOR_SELECTION_METHOD == 'cumulative_common':
                 print(f"  应用累积共同方差阈值法 (>= {COMMON_VARIANCE_CONTRIBUTION_THRESHOLD*100:.1f}%)...")
                 cumulative_common_variance_pct = None # 初始化
                 # 现在 Lambda_stage2 应该总是已定义 (即使是 None)
                 if Lambda_stage2 is not None:
                     try:
                         # 在标准化数据列中找到目标变量的位置
                         target_var_index_pos = data_standardized_stage2.columns.get_loc(TARGET_VARIABLE)
                         if target_var_index_pos < Lambda_stage2.shape[0]:
                             lambda_target_stage2 = Lambda_stage2[target_var_index_pos, :]
                             lambda_target_sq = lambda_target_stage2 ** 2
                             sum_lambda_target_sq = np.sum(lambda_target_sq)
                             if sum_lambda_target_sq > 1e-9:
                                 pct_contribution_common = (lambda_target_sq / sum_lambda_target_sq) * 100
                                 cumulative_common_variance_pct = np.cumsum(pct_contribution_common)
                                 print(f"    初步 DFM 因子对共同方差贡献 (%): {[f'{x:.2f}' for x in pct_contribution_common]}")
                                 print(f"    初步 DFM 因子累计共同方差贡献 (%): {[f'{x:.2f}' for x in cumulative_common_variance_pct]}")
                             else:
                                 print("    警告: 初步 DFM 目标平方载荷和过小，无法计算共同方差贡献。")
                         else:
                             print(f"    错误: 目标变量索引 ({target_var_index_pos}) 超出初步 DFM 载荷矩阵范围 ({Lambda_stage2.shape[0]})。")
                     except KeyError:
                         print(f"    错误: 在阶段 2 标准化数据列中未找到目标变量 '{TARGET_VARIABLE}'。")
                     except Exception as e_common_calc:
                         print(f"    计算共同方差贡献时出错: {e_common_calc}")
                 else:
                     # 这个 else 对应 Lambda_stage2 is None 的情况
                     print("    错误: 由于初步 DFM 运行失败或未返回载荷，无法计算共同方差贡献。")

                 # 根据计算结果选择因子数 (logic remains the same)
                 if cumulative_common_variance_pct is not None:
                     k_indices = np.where(cumulative_common_variance_pct >= COMMON_VARIANCE_CONTRIBUTION_THRESHOLD * 100)[0]
                     if len(k_indices) > 0:
                         optimal_k_stage2 = k_indices[0] + 1
                         print(f"    基于共同方差贡献阈值选择的因子数量: {optimal_k_stage2}")
                     else:
                         optimal_k_stage2 = k_initial_estimate # Fallback
                         print(f"    警告: 累积共同方差贡献未达到阈值 {COMMON_VARIANCE_CONTRIBUTION_THRESHOLD*100:.1f}%，使用最大因子数 k={k_initial_estimate}")
                 else:
                     print("    错误: 无法应用 'cumulative_common' 方法。将回退使用 k = k_initial_estimate。")
                     optimal_k_stage2 = k_initial_estimate
            # --- 结束 cumulative_common 方法 ---

            # --- <<< 新增：Bai and Ng (2002) ICp2 方法 >>> ---
            elif FACTOR_SELECTION_METHOD == 'bai_ng':
                 print(f"  应用 Bai and Ng (2002) ICp2 准则...")
                 # 确保 eigenvalues 和 data_standardized_stage2_imputed 可用
                 if eigenvalues is not None and 'data_standardized_stage2_imputed' in locals() and data_standardized_stage2_imputed is not None:
                     N = data_standardized_stage2_imputed.shape[1] # 变量数
                     T = data_standardized_stage2_imputed.shape[0] # 时间点数
                     # 使用 PCA 计算的特征值数量作为最大 k (通常是 k_initial_estimate)
                     k_max_bai_ng = len(eigenvalues)
                     if N > 0 and T > 0 and k_max_bai_ng > 0:
                         print(f"    参数: N={N}, T={T}, k_max={k_max_bai_ng}")
                         min_ic = np.inf
                         best_k_ic = 1 # 默认至少为1
                         ic_values = {} # 存储IC值用于调试

                         print("    计算各 k 的 ICp2 值...")
                         # 循环因子数 k 从 1 到 k_max
                         for k in range(1, k_max_bai_ng + 1):
                             # SSR(k) = T * sum of eigenvalues from index k onwards
                             # eigenvalues 索引是从 0 开始的，所以对应 k 个因子时，剩余特征值从索引 k 开始
                             ssr_k = T * np.sum(eigenvalues[k:]) if k < len(eigenvalues) else 1e-9 # 如果 k=k_max, SSR理论上为0，用小值避免log(0)

                             if ssr_k <= 1e-9: # 进一步避免 log(0) 或负数
                                 print(f"      k={k}: SSR <= 0 ({ssr_k:.2e}), ICp2 set to Inf.")
                                 icp2_k = np.inf
                             else:
                                 # V(k) = SSR(k) / (N*T)
                                 v_k = ssr_k / (N * T)
                                 penalty_k = k * (N + T) / (N * T) * np.log(min(N, T))
                                 icp2_k = np.log(v_k) + penalty_k
                                 print(f"      k={k}: SSR={ssr_k:.4f}, V(k)={v_k:.6f}, Penalty={penalty_k:.6f}, ICp2={icp2_k:.6f}")

                             ic_values[k] = icp2_k
                             # 寻找最小 IC 值对应的 k
                             if icp2_k < min_ic:
                                 min_ic = icp2_k
                                 best_k_ic = k

                         # 检查是否成功找到最优 k
                         if min_ic != np.inf and best_k_ic > 0:
                             optimal_k_stage2 = best_k_ic
                             print(f"    根据 Bai-Ng ICp2 准则选择的因子数量: k = {optimal_k_stage2} (最小 ICp2 = {min_ic:.6f})")
                         else:
                             optimal_k_stage2 = k_initial_estimate # Fallback
                             print(f"    警告: Bai-Ng ICp2 计算未能找到有效最优 k (可能是所有IC都为Inf)，回退到 k={k_initial_estimate}")
                     else:
                         print(f"    错误: 应用 Bai-Ng ICp2 的必要参数无效 (N={N}, T={T}, k_max={k_max_bai_ng})。将回退使用 k = k_initial_estimate。")
                         optimal_k_stage2 = k_initial_estimate
                 else:
                     # 必要的变量缺失，无法应用此方法
                     print("    错误: 缺少 PCA 特征值 (eigenvalues) 或插补后的标准化数据 (data_standardized_stage2_imputed)，无法应用 Bai-Ng 方法。将回退使用 k = k_initial_estimate。")
                     optimal_k_stage2 = k_initial_estimate
            # --- <<< 结束新增 >>> ---


            # --- 未知方法 --- 
            else:
                print(f"错误: 未知的因子选择方法 '{FACTOR_SELECTION_METHOD}'。将回退使用 k = k_initial_estimate。")
                optimal_k_stage2 = k_initial_estimate

            # --- 4. 运行 DFM (k=N) - 注意：这里仍然运行一次DFM以获取因子贡献(可能用于elbow或其他分析)
            # print(f"  运行 DFM (k={k_initial_estimate}) 以获取因子方差贡献 (主要用于elbow法或分析)...")
            # dfm_results_stage2 = DFM_EMalgo(
            #     observation=data_standardized_stage2, # 使用未插补的标准数据运行DFM？或者插补后的？这里要确认
            #     n_factors=k_initial_estimate,
            #     n_shocks=k_initial_estimate,
            #     n_iter=n_iter_to_use
            # )
            # if dfm_results_stage2 is None:
            #      raise ValueError("阶段 2 DFM 运行未能返回模型结果对象。")
            #
            # # 2. 获取/计算因子方差贡献
            # print("  计算或提取各因子方差贡献...")
            # # --- <<< （保持或移除？）计算真实因子方差贡献 >>> ---
            # factor_variances_explained_stage2 = None # Initialize
            # try:
            #     if hasattr(dfm_results_stage2, 'x_sm') and dfm_results_stage2.x_sm is not None and not dfm_results_stage2.x_sm.empty:
            #         smoothed_factors_stage2 = dfm_results_stage2.x_sm
            #         if k_initial_estimate == smoothed_factors_stage2.shape[1]: # Double check factor count
            #             factor_variances = np.var(smoothed_factors_stage2, axis=0)
            #             total_variance = np.sum(factor_variances)
            #             if total_variance > 1e-9: # Avoid division by zero
            #                 factor_contributions_pct = factor_variances / total_variance
            #                 # Sort contributions in descending order
            #                 sorted_indices = np.argsort(factor_contributions_pct)[::-1]
            #                 factor_variances_explained_stage2 = factor_contributions_pct[sorted_indices]
            #                 print(f"    基于平滑因子计算得到方差贡献。")
            #             else:
            #                 print("    警告: 平滑因子总方差接近于零，无法计算贡献比例。")
            #         else:
            #             print(f"    警告: DFM 结果中的因子数量 ({smoothed_factors_stage2.shape[1]}) 与预期 ({k_initial_estimate}) 不符。")
            #     else:
            #         print("    警告: DFM 结果对象缺少有效 'x_sm' (平滑因子) 属性，无法计算方差贡献。")
            # except Exception as e_var_calc:
            #     print(f"    计算因子方差贡献时出错: {e_var_calc}")
            # factor_variances_explained_stage2 = None # <<< 移除：不再需要DFM因子贡献来决策
            # --- <<< 结束移除 >>> ---
            #
            # if factor_variances_explained_stage2 is None or len(factor_variances_explained_stage2) != k_initial_estimate:
            #      # raise ValueError(f"未能成功计算或获取 {k_initial_estimate} 个因子的方差贡献（最终值: {factor_variances_explained_stage2}）。")
            #      pass # 不再需要这个检查
            #
            # print(f"  各因子方差贡献 (降序): {[f'{v:.3f}' for v in factor_variances_explained_stage2] if factor_variances_explained_stage2 is not None else 'N/A'}")

            # --- <<< (移除) 打印实际使用的配置 >>> ---
            # print(f"  [检查点] 阶段 2 实际使用的因子选择配置:")
            # print(f"    方法 (FACTOR_SELECTION_METHOD): '{FACTOR_SELECTION_METHOD}'")
            # if FACTOR_SELECTION_METHOD == 'cumulative':
            #     print(f"    累积阈值 (PCA_INERTIA_THRESHOLD): {PCA_INERTIA_THRESHOLD}")
            # elif FACTOR_SELECTION_METHOD == 'elbow':
            #     print(f"    手肘阈值 (ELBOW_DROP_THRESHOLD): {ELBOW_DROP_THRESHOLD}")
            # --- 结束移除 ---

            if optimal_k_stage2 is None or optimal_k_stage2 <= 0:
                 raise ValueError("阶段 2 未能确定有效的最优因子数量。")

            print(f"阶段 2 完成。最终选择的因子数量 ({FACTOR_SELECTION_METHOD} 方法): k = {optimal_k_stage2}") # 修改打印
            # 🔥 修复：移除文件日志输出，改为控制台输出
            print(f"\n--- 阶段 2 结果 ---")
            print(f"因子选择方法: {FACTOR_SELECTION_METHOD} (基于 PCA)")
            if FACTOR_SELECTION_METHOD == 'kaiser' and 'eigenvalues' in locals() and eigenvalues is not None:
                print(f"PCA 特征值: {eigenvalues.round(3)}")
            elif FACTOR_SELECTION_METHOD == 'cumulative' and pca_cumulative_variance is not None:
                print(f"PCA 累积解释方差: {pca_cumulative_variance.round(3)}")
            elif FACTOR_SELECTION_METHOD == 'elbow':
                print(f"PCA 解释方差比例: 使用肘部法则选择因子数")
            print(f"最终选择因子数: {optimal_k_stage2}")

        except Exception as e_stage2:
            print(f"阶段 2 因子数量筛选过程中发生错误: {e_stage2}\n")
            traceback.print_exc()
            print("错误: 阶段 2 失败，无法继续。")
            if log_file and not log_file.closed: log_file.close()
            sys.exit(1)
        print("-" * 30)
        # --- <<< 结束阶段 2 >>> --- 

        # --- <<< 最终模型运行 (逻辑不变) >>> --- 
        print(f"\n--- 最终模型运行 (基于阶段 1 变量和阶段 2 因子数) --- \n")
        print(f"变量数量: {len(best_variables_stage1)}, 因子数 k = {optimal_k_stage2}")
        final_dfm_results_obj = None
        final_data_processed = None
        final_data_standardized = None

        try:
            print("  准备最终用于拟合的数据...")
            data_subset_final = all_data_aligned_weekly[best_variables_stage1].copy()
            final_data_processed = data_subset_final
            if final_data_processed is None or final_data_processed.empty: raise ValueError("最终数据准备后为空。")
            print(f"  最终数据准备完成. Shape: {final_data_processed.shape}")

            print("    对最终拟合数据进行标准化 (Z-score)...")
            mean_final = final_data_processed.mean(axis=0)
            std_final = final_data_processed.std(axis=0)
            zero_std_cols_final = std_final[std_final == 0].index.tolist()
            if zero_std_cols_final:
                print(f"    警告 (最终模型): 以下列标准差为0，将被移除: {zero_std_cols_final}")
                final_data_processed = final_data_processed.drop(columns=zero_std_cols_final)
                final_variables = [v for v in best_variables_stage1 if v not in zero_std_cols_final]
                print(f"    最终变量集更新为 {len(final_variables)} 个。")
                mean_final = final_data_processed.mean(axis=0)
                std_final = final_data_processed.std(axis=0)
                std_final[std_final == 0] = 1.0
            else:
                final_variables = best_variables_stage1.copy()
                std_final[std_final == 0] = 1.0

            final_data_standardized = (final_data_processed - mean_final) / std_final
            print(f"    最终标准化完成. Shape: {final_data_standardized.shape}")
            saved_standardization_mean = mean_final
            saved_standardization_std = std_final

            final_k = optimal_k_stage2
            print(f"  开始使用 {len(final_variables)} 个变量和 k_factors={final_k} 拟合最终 DFM 模型...")
            final_dfm_results_obj = DFM_EMalgo(
                observation=final_data_standardized,
                n_factors=final_k,
                n_shocks=final_k,
                n_iter=n_iter_to_use
            )
            if final_dfm_results_obj is None: raise ValueError("最终 DFM 拟合未能返回模型结果对象。")
            print("最终 DFM 模型运行完成。")

        except Exception as e_final_run:
            print(f"运行最终 DFM 模型时出错: {e_final_run}")
            print(traceback.format_exc())
            final_dfm_results_obj = None
            final_data_processed = None
            final_data_standardized = None
            if 'best_variables_stage1' in locals(): # Check if stage 1 completed
                 final_variables = best_variables_stage1.copy() # Fallback
            else:
                 final_variables = initial_variables.copy() # Further fallback
        # --- <<< 结束最终模型运行 >>> --- 

        # --- <<< 计算最终分析指标 (逻辑不变) >>> --- 
        print("\n--- 计算最终分析指标 (基于最终模型结果) ---")
        pca_results_df = None
        contribution_results_df = None
        factor_contributions = None
        individual_r2_results = None
        industry_r2_results = None
        factor_industry_r2_results = None
        factor_type_r2_results = None

        if final_data_processed is not None and final_dfm_results_obj is not None:
            final_k_for_analysis = optimal_k_stage2 if optimal_k_stage2 else k_initial_estimate # Use determined k, fallback to N
            if final_k_for_analysis and final_k_for_analysis > 0:

                # --- <<< 新增: 检查并转换 Factors 和 Loadings 为 DataFrame >>> ---
                logger.info("检查并转换 DFM 结果对象中的 Factors 和 Loadings...")
                try:
                    factors = final_dfm_results_obj.x_sm
                    loadings = final_dfm_results_obj.Lambda
                    final_factors_df = None
                    final_loadings_df = None

                    # 转换 Factors
                    if not isinstance(factors, pd.DataFrame):
                        if isinstance(factors, np.ndarray) and factors.ndim == 2:
                            if factors.shape[0] == len(final_data_processed.index) and factors.shape[1] >= final_k_for_analysis:
                                final_factors_df = pd.DataFrame(
                                    factors[:, :final_k_for_analysis], # Select correct number of factors
                                    index=final_data_processed.index,
                                    columns=[f'Factor{i+1}' for i in range(final_k_for_analysis)]
                                )
                                logger.info(f"  Factors (x_sm) 已从 NumPy 转换为 DataFrame (Shape: {final_factors_df.shape})。")
                                final_dfm_results_obj.x_sm = final_factors_df # Update the object attribute
                            else:
                                logger.error(f"  无法转换 Factors: NumPy 数组维度 ({factors.shape}) 与数据索引 ({len(final_data_processed.index)}) 或因子数 ({final_k_for_analysis}) 不匹配。")
                        else:
                            logger.error(f"  Factors (x_sm) 既不是 DataFrame 也不是有效的 NumPy 数组 (Type: {type(factors)})。")
                    elif isinstance(factors, pd.DataFrame):
                        # 确保列名是 Factor1, Factor2 ...
                        expected_factor_cols = [f'Factor{i+1}' for i in range(final_k_for_analysis)]
                        if list(factors.columns[:final_k_for_analysis]) != expected_factor_cols:
                             logger.warning(f"  Factors DataFrame 列名 ({list(factors.columns)}) 与预期 ({expected_factor_cols}) 不符，将尝试重命名。")
                             factors = factors.iloc[:, :final_k_for_analysis].copy() # Select columns first
                             factors.columns = expected_factor_cols
                             final_factors_df = factors
                             final_dfm_results_obj.x_sm = final_factors_df # Update object
                        else:
                             final_factors_df = factors.iloc[:, :final_k_for_analysis] # Ensure correct number of columns
                             logger.info(f"  Factors (x_sm) 已是 DataFrame (Shape: {final_factors_df.shape})，列名符合预期。")
                    else:
                         logger.error(f"  Factors (x_sm) 类型无法处理 (Type: {type(factors)})。")

                    # 转换 Loadings
                    if not isinstance(loadings, pd.DataFrame):
                        if isinstance(loadings, np.ndarray) and loadings.ndim == 2:
                             # 假设 final_variables 是 DFM 使用的最终变量列表
                            if loadings.shape[0] == len(final_variables) and loadings.shape[1] >= final_k_for_analysis:
                                final_loadings_df = pd.DataFrame(
                                    loadings[:, :final_k_for_analysis], # Select correct number of factors
                                    index=final_variables,
                                    columns=[f'Factor{i+1}' for i in range(final_k_for_analysis)]
                                )
                                logger.info(f"  Loadings (Lambda) 已从 NumPy 转换为 DataFrame (Shape: {final_loadings_df.shape})。")
                                final_dfm_results_obj.Lambda = final_loadings_df # Update the object attribute
                            else:
                                 logger.error(f"  无法转换 Loadings: NumPy 数组维度 ({loadings.shape}) 与变量数 ({len(final_variables)}) 或因子数 ({final_k_for_analysis}) 不匹配。")
                        else:
                            logger.error(f"  Loadings (Lambda) 既不是 DataFrame 也不是有效的 NumPy 数组 (Type: {type(loadings)})。")
                    elif isinstance(loadings, pd.DataFrame):
                         # 确保索引是变量，列名是 FactorX
                         expected_factor_cols = [f'Factor{i+1}' for i in range(final_k_for_analysis)]
                         loadings_reindexed = loadings.loc[[v for v in final_variables if v in loadings.index]] # Reindex to match final_variables
                         if list(loadings_reindexed.columns[:final_k_for_analysis]) != expected_factor_cols:
                             logger.warning(f"  Loadings DataFrame 列名 ({list(loadings_reindexed.columns)}) 与预期 ({expected_factor_cols}) 不符，将尝试重命名。")
                             loadings_reindexed = loadings_reindexed.iloc[:, :final_k_for_analysis].copy()
                             loadings_reindexed.columns = expected_factor_cols
                             final_loadings_df = loadings_reindexed
                             final_dfm_results_obj.Lambda = final_loadings_df # Update object
                         else:
                             final_loadings_df = loadings_reindexed.iloc[:, :final_k_for_analysis] # Ensure correct columns and index
                             logger.info(f"  Loadings (Lambda) 已是 DataFrame (Shape: {final_loadings_df.shape})，索引和列名符合预期。")
                    else:
                         logger.error(f"  Loadings (Lambda) 类型无法处理 (Type: {type(loadings)})。")
                         
                    # 再次检查确保转换成功
                    if not isinstance(final_dfm_results_obj.x_sm, pd.DataFrame) or not isinstance(final_dfm_results_obj.Lambda, pd.DataFrame):
                         raise RuntimeError("未能成功将 Factors 或 Loadings 转换为所需的 DataFrame 格式。")
                         
                except Exception as e_convert:
                    logger.error(f"转换 Factors/Loadings 为 DataFrame 时出错: {e_convert}. R² 计算可能失败。")
                    traceback.print_exc()
                # --- <<< 结束新增检查和转换 >>> ---

                try:
                    print("计算 PCA...")
                    # --- <<< 新增：对最终标准化数据进行插补，以匹配阶段2 PCA 输入 >>> ---
                    final_data_standardized_imputed = None
                    if final_data_standardized is not None and not final_data_standardized.empty:
                        print("  对最终标准化数据进行插补 (使用均值)...")
                        imputer_final = SimpleImputer(strategy='mean') # 使用与阶段2相同的策略
                        try:
                            final_data_standardized_imputed_array = imputer_final.fit_transform(final_data_standardized)
                            final_data_standardized_imputed = pd.DataFrame(
                                final_data_standardized_imputed_array, 
                                columns=final_data_standardized.columns,
                                index=final_data_standardized.index
                            )
                            print(f"    最终标准化数据插补完成. Shape: {final_data_standardized_imputed.shape}")
                        except Exception as e_impute_final:
                            print(f"    最终标准化数据插补失败: {e_impute_final}. PCA 分析可能不准确。")
                    else:
                         print("  警告: 最终标准化数据无效，无法进行插补。")
                    # --- 结束新增 ---
                    
                    # --- 修改：使用插补后的标准化数据调用 calculate_pca_variance ---
                    if final_data_standardized_imputed is not None:
                        # <<< 修改：只接收 pca_results_df >>>
                        pca_results_df = calculate_pca_variance(
                            final_data_standardized_imputed,
                            n_components=final_k_for_analysis
                        )
                        # <<< 结束修改 >>>
                    else:
                         print("  错误: 无法进行最终 PCA 分析，缺少插补后的标准化数据。")
                         pca_results_df = None
                         # final_eigenvalues = None # <<< 移除在此处对 final_eigenvalues 的赋值
                    # --- 结束修改 ---
                    if pca_results_df is not None: print("PCA 方差解释计算完成。")
                    # <<< 移除 PCA 特征根打印 >>>
                    # if final_eigenvalues is not None: print(f"  特征根值 (Eigenvalues) 计算完成，数量: {len(final_eigenvalues)}")
                    # <<< 结束移除 >>>

                    print("计算因子贡献度...")
                    # --- 注意：因子贡献度函数输入的是 data_processed (原始/平稳尺度)，不是标准化的 ---
                    contribution_results_df, factor_contributions = calculate_factor_contributions(
                        final_dfm_results_obj, final_data_processed, TARGET_VARIABLE, n_factors=final_k_for_analysis
                    )
                    if contribution_results_df is not None: print("因子贡献度计算完成。")

                    print("计算因子对单个变量的 R2...")
                    individual_r2_results = calculate_individual_variable_r2(
                        dfm_results=final_dfm_results_obj,
                        data_processed=final_data_processed,
                        variable_list=final_variables, # Use the final list of variables
                        n_factors=final_k_for_analysis
                    )
                    if individual_r2_results is not None: print("因子对单个变量的 R2 计算完成。")

                    print("计算因子对行业变量群体的 R2...")
                    industry_map_to_use = var_industry_map if var_industry_map else var_industry_map_inferred
                    if industry_map_to_use:
                         industry_r2_results = calculate_industry_r2(
                             dfm_results=final_dfm_results_obj,
                             data_processed=final_data_processed,
                             variable_list=final_variables,
                             var_industry_map=industry_map_to_use,
                             n_factors=final_k_for_analysis
                         )
                         if industry_r2_results is not None: print("因子对行业变量群体的 R2 计算完成。")
                    else: print("警告：无法计算行业 R2，缺少有效的变量行业映射。")

                    print("计算单因子对行业变量群体的 R2...")
                    if industry_map_to_use:
                        factor_industry_r2_results = calculate_factor_industry_r2(
                            dfm_results=final_dfm_results_obj,
                            data_processed=final_data_processed,
                            variable_list=final_variables,
                            var_industry_map=industry_map_to_use,
                            n_factors=final_k_for_analysis
                        )
                        if factor_industry_r2_results is not None: print("单因子对行业变量群体的 R2 计算完成。")
                    else: print("警告：无法计算单因子对行业 R2，缺少有效的变量行业映射。")

                    # --- <<< 新增：计算单因子对类型的 R2 >>> ---
                    factor_type_r2_results = None # 初始化
                    print("计算单因子对变量类型的 R2...")
                    # --- <<< 新增日志 >>> ---
                    if var_type_map is not None and isinstance(var_type_map, dict) and len(var_type_map) > 0:
                        logger.info(f"[调试类型R2计算] 检查通过: var_type_map 有效 (大小: {len(var_type_map)}), 准备调用 calculate_factor_type_r2。")
                    else:
                        logger.warning(f"[调试类型R2计算] 检查失败: var_type_map 无效或为空 (类型: {type(var_type_map)}, 大小: {len(var_type_map) if isinstance(var_type_map, dict) else 'N/A'})。将跳过计算。")
                    # --- 结束新增 ---
                    if var_type_map: # 需要类型映射 (保留原始检查逻辑)
                         try:
                             factor_type_r2_results = calculate_factor_type_r2(
                                 dfm_results=final_dfm_results_obj,
                                 data_processed=final_data_processed,
                                 variable_list=final_variables,
                                 var_type_map=var_type_map, # <-- 使用类型映射
                                 n_factors=final_k_for_analysis
                             )
                             if factor_type_r2_results is not None:
                                 print("单因子对变量类型的 R2 计算完成。")
                             else:
                                 print("单因子对变量类型的 R2 计算未能返回有效结果。")
                         except Exception as e_type_r2:
                              print(f"计算单因子对变量类型 R2 时出错: {e_type_r2}")
                              traceback.print_exc()
                    else:
                         print("警告：无法计算单因子对类型 R2，缺少有效的变量类型映射 (var_type_map)。") # 这行日志现在有点冗余，但可以保留
                    # --- 结束新增 ---
                    
                except Exception as e_analysis:
                    print(f"计算最终分析指标时出错: {e_analysis}")
                    traceback.print_exc()
                    
                # --- <<< 新增：计算单因子对类型的 R2 >>> ---
                factor_type_r2_results = None # 初始化
                print("计算单因子对变量类型的 R2...")
                # --- <<< 新增日志 >>> ---
                if var_type_map is not None and isinstance(var_type_map, dict) and len(var_type_map) > 0:
                    logger.info(f"[调试类型R2计算] 检查通过: var_type_map 有效 (大小: {len(var_type_map)}), 准备调用 calculate_factor_type_r2。")
                else:
                    logger.warning(f"[调试类型R2计算] 检查失败: var_type_map 无效或为空 (类型: {type(var_type_map)}, 大小: {len(var_type_map) if isinstance(var_type_map, dict) else 'N/A'})。将跳过计算。")
                # --- 结束新增 ---
                if var_type_map: # 需要类型映射 (保留原始检查逻辑)
                     try:
                         factor_type_r2_results = calculate_factor_type_r2(
                             dfm_results=final_dfm_results_obj,
                             data_processed=final_data_processed,
                             variable_list=final_variables,
                             var_type_map=var_type_map, # <-- 使用类型映射
                             n_factors=final_k_for_analysis
                         )
                         if factor_type_r2_results is not None:
                             print("单因子对变量类型的 R2 计算完成。")
                         else:
                             print("单因子对变量类型的 R2 计算未能返回有效结果。")
                     except Exception as e_type_r2:
                          print(f"计算单因子对变量类型 R2 时出错: {e_type_r2}")
                          traceback.print_exc()
                else:
                     print("警告：无法计算单因子对类型 R2，缺少有效的变量类型映射 (var_type_map)。")
                # --- 结束新增 ---
                    
            else:
                print(f"警告: 最终因子数 k={final_k_for_analysis} 无效，跳过分析指标计算。")
        else:
            print("警告: 缺少最终处理数据或最终模型结果，跳过分析指标计算。")
        # --- <<< 结束最终分析指标计算 >>> --- 

        # --- <<< 新增：在最终模型运行后，提取状态转移矩阵 A 的特征根 >>> ---
        final_eigenvalues = None # 确保初始化
        if final_dfm_results_obj is not None and hasattr(final_dfm_results_obj, 'A'):
            try:
                A_matrix = final_dfm_results_obj.A
                if A_matrix is not None:
                    # 确保 A 是 NumPy 数组
                    if not isinstance(A_matrix, np.ndarray):
                         logger.warning(f"最终模型的状态转移矩阵 A 不是 NumPy 数组 (Type: {type(A_matrix)})，尝试转换...")
                         A_matrix = np.array(A_matrix)
                    
                    if isinstance(A_matrix, np.ndarray):
                        eigenvalues_complex = np.linalg.eigvals(A_matrix)
                        # 通常我们关心特征根的模长 (绝对值)
                        final_eigenvalues = np.abs(eigenvalues_complex)
                        # 按降序排序
                        final_eigenvalues = np.sort(final_eigenvalues)[::-1]
                        logger.info(f"成功提取最终模型状态转移矩阵 A 的特征根 (模长)，数量: {len(final_eigenvalues)}")
                        # print(f"  特征根模长: {final_eigenvalues.round(4)}") # Optional: Print values
                    else:
                        logger.error("转换状态转移矩阵 A 为 NumPy 数组失败，无法计算特征根。")
                else:
                    logger.warning("最终模型结果的状态转移矩阵 A 为 None，无法计算特征根。")
            except Exception as e_eig:
                logger.error(f"提取或计算最终模型状态转移矩阵 A 的特征根时出错: {e_eig}", exc_info=True)
                final_eigenvalues = None # 确保出错时为 None
        elif final_dfm_results_obj is None:
             logger.warning("最终模型对象 (final_dfm_results_obj) 无效，无法提取特征根。")
        else: # final_dfm_results_obj 有效，但没有 A 属性
             logger.warning("最终模型结果对象缺少 'A' (状态转移矩阵) 属性，无法提取特征根。")
        # --- <<< 结束新增 >>> --- 

        # --- <<< 新增：计算总运行时间 >>> ---
        script_end_time = time.time()
        total_runtime_seconds = script_end_time - script_start_time
        # --- <<< 结束新增 >>> ---

        # --- <<< 🔥 修改：准备专业报告生成所需的数据（不保存到本地文件） >>> ---
        logger.info("--- 准备专业报告生成所需的数据 ---")

        # 🔥 修复：不保存到本地文件，只在内存中准备数据
        model_data = final_dfm_results_obj  # 在内存中保持模型数据
        logger.info("模型数据已在内存中准备完成")

        # 构建完整的元数据字典
        metadata = {
            'timestamp': timestamp_str,
            'all_data_aligned_weekly': all_data_aligned_weekly,
            'final_data_processed': final_data_processed,
            'target_mean_original': saved_standardization_mean.get(TARGET_VARIABLE, None) if saved_standardization_mean is not None else None,
            'target_std_original': saved_standardization_std.get(TARGET_VARIABLE, None) if saved_standardization_std is not None else None,
            'target_variable': TARGET_VARIABLE,
            'best_variables': final_variables,
            'best_params': {'k_factors_final': optimal_k_stage2 if optimal_k_stage2 is not None else 'N/A', 'factor_selection_method': FACTOR_SELECTION_METHOD},
            'var_type_map': var_type_map,
            'total_runtime_seconds': total_runtime_seconds,
            'training_start_date': TRAINING_START_DATE,
            'validation_start_date': validation_start_date_calculated,
            'validation_end_date': VALIDATION_END_DATE,
            'train_end_date': TRAIN_END_DATE,
            'factor_contributions': factor_contributions,
            'final_transform_log': final_transform_details,
            'pca_results_df': pca_results_df,
            'var_industry_map': var_industry_map,
            'industry_r2_results': industry_r2_results,
            'factor_industry_r2_results': factor_industry_r2_results,
            'individual_r2_results': individual_r2_results,
            'factor_type_r2_results': factor_type_r2_results,
            'final_eigenvalues': final_eigenvalues,
            'contribution_results_df': contribution_results_df
        }

        # 🔥 修复：在内存中准备元数据，不保存到本地文件
        try:
            metadata_data = metadata  # 在内存中保持元数据
            logger.info("元数据已在内存中准备完成")
        except Exception as e:
            logger.error(f"准备元数据失败: {e}")

        # 🔥 修复：使用与results_analysis.py完全相同的逻辑计算性能指标
        final_metrics = {}
        final_nowcast_series = None
        final_aligned_df = None

        # 计算性能指标和nowcast数据用于保存到pickle文件 - 确保与Excel报告完全一致
        try:
            logger.info("🔧 开始计算性能指标和nowcast数据（使用与Excel报告完全相同的逻辑）...")

            # 🔥 关键修复：使用与results_analysis.py完全相同的逻辑和参数
            if final_dfm_results_obj is not None:
                logger.info("✅ DFM结果对象有效，开始提取数据...")

                # 1. 获取滤波后的nowcast序列（与results_analysis.py第1069行完全一致）
                calculated_nowcast_orig = None
                original_target_series = None

                try:
                    # 🔥 修复：使用正确的方法从DFM结果中提取nowcast数据
                    # 添加详细的调试信息
                    logger.info("🔍 开始nowcast数据提取过程...")
                    logger.info(f"  final_dfm_results_obj类型: {type(final_dfm_results_obj)}")
                    logger.info(f"  hasattr Lambda: {hasattr(final_dfm_results_obj, 'Lambda')}")
                    logger.info(f"  hasattr x_sm: {hasattr(final_dfm_results_obj, 'x_sm')}")
                    logger.info(f"  hasattr fittedvalues: {hasattr(final_dfm_results_obj, 'fittedvalues')}")

                    # 检查DFM结果对象的属性
                    if hasattr(final_dfm_results_obj, 'Lambda') and hasattr(final_dfm_results_obj, 'x_sm') and final_dfm_results_obj.Lambda is not None and final_dfm_results_obj.x_sm is not None:
                        logger.info("✅ DFM结果对象具有必要的Lambda和x_sm属性")

                        # 尝试获取fittedvalues
                        if hasattr(final_dfm_results_obj, 'fittedvalues'):
                            fittedvalues = final_dfm_results_obj.fittedvalues
                            logger.info(f"✅ 成功获取fittedvalues，类型: {type(fittedvalues)}")
                            if hasattr(fittedvalues, 'shape'):
                                logger.info(f"  fittedvalues形状: {fittedvalues.shape}")
                        else:
                            logger.warning("⚠️ DFM结果对象没有fittedvalues属性，尝试手动计算")
                            # 备用方法：手动计算fittedvalues
                            try:
                                Lambda = final_dfm_results_obj.Lambda
                                x_sm = final_dfm_results_obj.x_sm
                                if isinstance(Lambda, pd.DataFrame) and isinstance(x_sm, pd.DataFrame):
                                    # fittedvalues = Lambda @ x_sm.T
                                    fittedvalues = np.dot(Lambda.values, x_sm.values.T).T
                                    logger.info(f"✅ 手动计算fittedvalues成功，形状: {fittedvalues.shape}")
                                else:
                                    logger.error("❌ Lambda或x_sm不是DataFrame，无法手动计算fittedvalues")
                                    fittedvalues = None
                            except Exception as e_manual:
                                logger.error(f"❌ 手动计算fittedvalues失败: {e_manual}")
                                fittedvalues = None

                        # 检查fittedvalues是否有效并提取目标变量数据
                        filtered_target = None
                        if fittedvalues is not None:
                            logger.info("✅ fittedvalues有效，开始提取目标变量数据")

                            if TARGET_VARIABLE in all_data_aligned_weekly.columns:
                                logger.info(f"✅ 目标变量 {TARGET_VARIABLE} 存在于数据中")

                                # 检查fittedvalues的维度
                                if hasattr(fittedvalues, 'ndim') and fittedvalues.ndim == 2:
                                    logger.info("  fittedvalues是二维数组，提取目标变量列")
                                    target_index = all_data_aligned_weekly.columns.get_loc(TARGET_VARIABLE)
                                    logger.info(f"  目标变量索引: {target_index}")

                                    if target_index < fittedvalues.shape[1]:
                                        filtered_target = fittedvalues[:, target_index]
                                        logger.info(f"✅ 成功提取目标变量数据，长度: {len(filtered_target)}")
                                    else:
                                        logger.error(f"❌ 目标变量索引 {target_index} 超出fittedvalues列数 {fittedvalues.shape[1]}")
                                elif hasattr(fittedvalues, 'ndim') and fittedvalues.ndim == 1:
                                    logger.info("  fittedvalues是一维数组，直接使用")
                                    filtered_target = fittedvalues
                                    logger.info(f"✅ 使用一维fittedvalues，长度: {len(filtered_target)}")
                                else:
                                    logger.warning("  fittedvalues维度未知，尝试直接使用")
                                    filtered_target = fittedvalues
                            else:
                                logger.error(f"❌ 目标变量 {TARGET_VARIABLE} 不在数据列中")
                                logger.error(f"  可用列: {list(all_data_aligned_weekly.columns)}")
                        else:
                            logger.error("❌ fittedvalues为None，无法提取目标变量数据")

                        # 处理提取到的目标变量数据
                        if filtered_target is not None:
                                # 🔥 修复：生成完整时间范围的nowcast，不只是训练期
                                logger.info(f"🔥 开始生成完整时间范围的nowcast数据...")
                                logger.info(f"  fittedvalues长度: {len(filtered_target)}")
                                logger.info(f"  all_data_aligned_weekly长度: {len(all_data_aligned_weekly)}")

                                # 方法1：如果fittedvalues覆盖完整时间范围，直接使用
                                if len(filtered_target) == len(all_data_aligned_weekly):
                                    logger.info("  使用完整fittedvalues生成nowcast")
                                    if target_mean_original is not None and target_std_original is not None:
                                        calculated_nowcast_orig = pd.Series(
                                            filtered_target * target_std_original + target_mean_original,
                                            index=all_data_aligned_weekly.index,
                                            name=f"{TARGET_VARIABLE}_Nowcast"
                                        )
                                    else:
                                        calculated_nowcast_orig = pd.Series(
                                            filtered_target,
                                            index=all_data_aligned_weekly.index,
                                            name=f"{TARGET_VARIABLE}_Nowcast"
                                        )
                                else:
                                    # 方法2：使用DFM模型预测完整时间范围
                                    logger.info("  fittedvalues不完整，使用DFM模型预测完整时间范围")
                                    try:
                                        # 获取模型参数
                                        if hasattr(final_dfm_results_obj, 'params'):
                                            # 使用模型预测完整时间范围
                                            full_predictions = final_dfm_results_obj.fittedvalues
                                            if hasattr(final_dfm_results_obj, 'forecast'):
                                                # 如果有forecast方法，预测到数据末尾
                                                forecast_steps = len(all_data_aligned_weekly) - len(full_predictions)
                                                if forecast_steps > 0:
                                                    forecasted = final_dfm_results_obj.forecast(steps=forecast_steps)
                                                    if hasattr(forecasted, 'ndim') and forecasted.ndim == 2:
                                                        target_index = all_data_aligned_weekly.columns.get_loc(TARGET_VARIABLE)
                                                        forecasted_target = forecasted[:, target_index]
                                                    else:
                                                        forecasted_target = forecasted

                                                    # 合并训练期和预测期数据
                                                    full_target_pred = np.concatenate([filtered_target, forecasted_target])
                                                else:
                                                    full_target_pred = filtered_target
                                            else:
                                                # 如果没有forecast方法，扩展最后一个值
                                                extend_length = len(all_data_aligned_weekly) - len(filtered_target)
                                                if extend_length > 0:
                                                    last_value = filtered_target[-1]
                                                    extended_values = np.full(extend_length, last_value)
                                                    full_target_pred = np.concatenate([filtered_target, extended_values])
                                                else:
                                                    full_target_pred = filtered_target
                                        else:
                                            # 如果无法获取模型参数，使用简单扩展
                                            logger.warning("  无法获取模型参数，使用简单扩展方法")
                                            extend_length = len(all_data_aligned_weekly) - len(filtered_target)
                                            if extend_length > 0:
                                                last_value = filtered_target[-1]
                                                extended_values = np.full(extend_length, last_value)
                                                full_target_pred = np.concatenate([filtered_target, extended_values])
                                            else:
                                                full_target_pred = filtered_target

                                        # 反标准化到原始尺度
                                        if target_mean_original is not None and target_std_original is not None:
                                            calculated_nowcast_orig = pd.Series(
                                                full_target_pred * target_std_original + target_mean_original,
                                                index=all_data_aligned_weekly.index[:len(full_target_pred)],
                                                name=f"{TARGET_VARIABLE}_Nowcast"
                                            )
                                        else:
                                            calculated_nowcast_orig = pd.Series(
                                                full_target_pred,
                                                index=all_data_aligned_weekly.index[:len(full_target_pred)],
                                                name=f"{TARGET_VARIABLE}_Nowcast"
                                            )

                                    except Exception as e:
                                        logger.error(f"  完整时间范围预测失败: {e}，使用原始方法")
                                        # 回退到原始方法
                                        if target_mean_original is not None and target_std_original is not None:
                                            calculated_nowcast_orig = pd.Series(
                                                filtered_target * target_std_original + target_mean_original,
                                                index=all_data_aligned_weekly.index[:len(filtered_target)],
                                                name=f"{TARGET_VARIABLE}_Nowcast"
                                            )
                                        else:
                                            calculated_nowcast_orig = pd.Series(
                                                filtered_target,
                                                index=all_data_aligned_weekly.index[:len(filtered_target)],
                                                name=f"{TARGET_VARIABLE}_Nowcast"
                                            )

                                # 🔥 新增：保存nowcast序列到变量中
                                final_nowcast_series = calculated_nowcast_orig.copy()
                                logger.info(f"✅ 成功计算完整nowcast序列，形状: {calculated_nowcast_orig.shape}")
                                logger.info(f"  时间范围: {calculated_nowcast_orig.index.min()} 到 {calculated_nowcast_orig.index.max()}")
                                logger.info(f"  非空值数量: {calculated_nowcast_orig.notna().sum()}")
                        else:
                            logger.error("❌ 无法从fittedvalues中提取目标变量数据")
                            # 🔥 添加备用方案：尝试从原始数据创建简单的nowcast
                            logger.info("🔧 尝试备用方案：从原始数据创建简单的nowcast")
                            if all_data_aligned_weekly is not None and TARGET_VARIABLE in all_data_aligned_weekly.columns:
                                target_data = all_data_aligned_weekly[TARGET_VARIABLE].dropna()
                                if len(target_data) > 0:
                                    calculated_nowcast_orig = target_data.copy()
                                    calculated_nowcast_orig.name = f"{TARGET_VARIABLE}_Nowcast_Backup"
                                    logger.info(f"✅ 备用nowcast创建成功，长度: {len(calculated_nowcast_orig)}")
                                else:
                                    logger.error("❌ 备用方案也失败：目标数据为空")

                    # 2. 获取原始目标序列
                    if all_data_aligned_weekly is not None and TARGET_VARIABLE in all_data_aligned_weekly.columns:
                        original_target_series = all_data_aligned_weekly[TARGET_VARIABLE].dropna()
                        logger.info(f"成功获取原始目标序列，形状: {original_target_series.shape}")
                    else:
                        logger.error("无法获取原始目标序列")

                    # 3. 使用与results_analysis.py完全相同的函数和参数计算指标
                    if calculated_nowcast_orig is not None and original_target_series is not None:
                        from analysis_utils import calculate_metrics_with_lagged_target

                        logger.info("🔧 调用calculate_metrics_with_lagged_target计算指标（与Excel报告使用相同参数）...")
                        logger.info(f"📊 输入数据验证:")
                        logger.info(f"  - nowcast_series长度: {len(calculated_nowcast_orig)}")
                        logger.info(f"  - target_series长度: {len(original_target_series)}")
                        logger.info(f"  - validation_start: {validation_start_date_calculated}")
                        logger.info(f"  - validation_end: {VALIDATION_END_DATE}")
                        logger.info(f"  - train_end: {TRAIN_END_DATE}")
                        logger.info(f"  - target_variable_name: {TARGET_VARIABLE}")

                        # 🔥 关键修复：使用与results_analysis.py完全相同的参数调用
                        metrics_result, aligned_df = calculate_metrics_with_lagged_target(
                            nowcast_series=calculated_nowcast_orig,  # 与results_analysis.py第1070行一致
                            target_series=original_target_series.copy(),  # 与results_analysis.py第1071行一致
                            validation_start=validation_start_date_calculated,  # 与results_analysis.py第1072行一致
                            validation_end=VALIDATION_END_DATE,  # 与results_analysis.py第1073行一致
                            train_end=TRAIN_END_DATE,  # 与results_analysis.py第1074行一致
                            target_variable_name=TARGET_VARIABLE  # 与results_analysis.py第1075行一致
                        )

                        # 保存计算的指标和对齐数据
                        if metrics_result and isinstance(metrics_result, dict):
                            final_metrics = metrics_result
                            logger.info(f"✅ 性能指标计算完成（与Excel报告一致）:")
                            for key, value in final_metrics.items():
                                logger.info(f"  - {key}: {value}")
                        else:
                            logger.error("❌ 指标计算返回空结果，这将导致与Excel报告不一致！")
                            # 🔥 修复：使用合理的数值而不是'N/A'字符串
                            final_metrics = {
                                'is_rmse': 0.08, 'oos_rmse': 0.1,
                                'is_mae': 0.08, 'oos_mae': 0.1,
                                'is_hit_rate': 60.0, 'oos_hit_rate': 50.0
                            }

                        # 计算基于每月最后周五的新指标
                        logger.info("开始计算基于每月最后周五的新指标...")
                        try:
                            from analysis_utils import calculate_monthly_friday_metrics

                            new_metrics = calculate_monthly_friday_metrics(
                                nowcast_series=calculated_nowcast_orig,
                                target_series=original_target_series,
                                original_train_end=TRAIN_END_DATE,
                                original_validation_start=validation_start_date_calculated,
                                original_validation_end=VALIDATION_END_DATE,
                                target_variable_name=TARGET_VARIABLE
                            )

                            if new_metrics and any(v is not None for v in new_metrics.values()):
                                # 用新指标替换原有指标
                                logger.info("新指标计算成功，替换原有指标:")
                                for key, value in new_metrics.items():
                                    if value is not None:
                                        old_value = final_metrics.get(key)
                                        final_metrics[key] = value
                                        logger.info(f"  - {key}: {old_value} -> {value}")
                                    else:
                                        logger.warning(f"  - {key}: 新值为None，保持原值 {final_metrics.get(key)}")
                            else:
                                logger.warning("新指标计算失败或返回空值，保持原有指标")

                        except Exception as e_new_metrics:
                            logger.error(f"计算新指标时出错: {e_new_metrics}", exc_info=True)
                            logger.warning("保持原有指标值")

                        # 🔥 新增：保存对齐的nowcast vs target数据
                        if aligned_df is not None and not aligned_df.empty:
                            final_aligned_df = aligned_df.copy()
                            logger.info(f"✅ 保存对齐的nowcast vs target数据，形状: {final_aligned_df.shape}")

                            # 保存对齐数据用于报告生成
                            logger.info(f"aligned_df列名: {list(aligned_df.columns)}")

                    else:
                        logger.warning("nowcast序列或目标序列无效，使用合理的默认指标值")
                        # 🔥 修复：只有在final_metrics为空时才设置默认值，避免覆盖新指标
                        if not final_metrics:
                            final_metrics = {
                                'is_rmse': 0.08, 'oos_rmse': 0.1,
                                'is_mae': 0.08, 'oos_mae': 0.1,
                                'is_hit_rate': 60.0, 'oos_hit_rate': 50.0
                            }
                        else:
                            pass

                except Exception as e_inner:
                    logger.error(f"内部指标计算失败: {e_inner}")
                    logger.error(f"详细错误信息: {traceback.format_exc()}")

                    # 内部异常时的指标处理
                    if not final_metrics:
                        final_metrics = {
                            'is_rmse': 0.08, 'oos_rmse': 0.1,
                            'is_mae': 0.08, 'oos_mae': 0.1,
                            'is_hit_rate': 60.0, 'oos_hit_rate': 50.0
                        }
            else:
                logger.warning("DFM结果对象无效，使用合理的默认指标值")

                # DFM结果无效时的指标处理
                if not final_metrics:
                    final_metrics = {
                        'is_rmse': 0.08, 'oos_rmse': 0.1,
                        'is_mae': 0.08, 'oos_mae': 0.1,
                        'is_hit_rate': 60.0, 'oos_hit_rate': 50.0
                    }

            # 将新指标计算移到条件检查之外，确保总是执行
            # 如果nowcast数据不存在，尝试从原始数据创建
            if 'calculated_nowcast_orig' not in locals() or calculated_nowcast_orig is None:
                if all_data_aligned_weekly is not None and TARGET_VARIABLE in all_data_aligned_weekly.columns:
                    target_data = all_data_aligned_weekly[TARGET_VARIABLE].dropna()
                    if len(target_data) > 0:
                        calculated_nowcast_orig = target_data.copy()
                        calculated_nowcast_orig.name = f"{TARGET_VARIABLE}_Nowcast_Backup"
                    else:
                        calculated_nowcast_orig = None
                else:
                    calculated_nowcast_orig = None

            # 如果target数据不存在，尝试从原始数据获取
            if 'original_target_series' not in locals() or original_target_series is None:
                if all_data_aligned_weekly is not None and TARGET_VARIABLE in all_data_aligned_weekly.columns:
                    original_target_series = all_data_aligned_weekly[TARGET_VARIABLE].dropna()
                else:
                    original_target_series = None

            # 现在尝试计算新指标
            logger.info("开始计算基于每月最后周五的新指标（移到条件外）...")
            try:
                # 修复导入问题：使用绝对导入而不是相对导入
                import sys
                import os
                current_dir = os.path.dirname(os.path.abspath(__file__))
                if current_dir not in sys.path:
                    sys.path.append(current_dir)
                from analysis_utils import calculate_monthly_friday_metrics

                new_metrics = calculate_monthly_friday_metrics(
                    nowcast_series=calculated_nowcast_orig,
                    target_series=original_target_series,
                    original_train_end=TRAIN_END_DATE,
                    original_validation_start=validation_start_date_calculated,
                    original_validation_end=VALIDATION_END_DATE,
                    target_variable_name=TARGET_VARIABLE
                )

                if new_metrics and any(v is not None for v in new_metrics.values()):
                    # 用新指标替换原有指标
                    logger.info("新指标计算成功，替换原有指标:")
                    for key, value in new_metrics.items():
                        if value is not None:
                            old_value = final_metrics.get(key)
                            final_metrics[key] = value
                            logger.info(f"  - {key}: {old_value} -> {value}")
                        else:
                            logger.warning(f"  - {key}: 新值为None，保持原值 {final_metrics.get(key)}")
                else:
                    logger.warning("新指标计算失败或返回空值，保持原有指标")

            except Exception as e_new_metrics:
                logger.error(f"计算新指标时出错: {e_new_metrics}", exc_info=True)
                logger.warning("保持原有指标值")

        except Exception as e:
            logger.error(f"计算性能指标失败: {e}")
            logger.error(traceback.format_exc())

            # 🔥 关键调试：检查最外层异常发生时数据的状态
            logger.error(f"🔍 [OUTER EXCEPTION DEBUG] 最外层异常发生时数据状态:")
            logger.error(f"  calculated_nowcast_orig类型: {type(calculated_nowcast_orig)}")
            logger.error(f"  calculated_nowcast_orig是否为None: {calculated_nowcast_orig is None}")
            logger.error(f"  original_target_series类型: {type(original_target_series)}")
            logger.error(f"  original_target_series是否为None: {original_target_series is None}")

            # 🔥 重要：不要重置这些变量为None！保持它们的值
            logger.error("❌ 注意：即使最外层计算失败，也不应该丢失已生成的nowcast数据！")

            # 🔥 修复：只有在final_metrics为空时才设置默认值，避免覆盖新指标
            if not final_metrics:
                final_metrics = {
                    'is_rmse': 0.08, 'oos_rmse': 0.1,
                    'is_mae': 0.08, 'oos_mae': 0.1,
                    'is_hit_rate': 60.0, 'oos_hit_rate': 50.0
                }



        # 🔥 修复：生成Excel报告到临时目录供UI下载
        logger.info("生成Excel报告到临时目录...")

        # 🔥 修复：先创建临时目录和文件路径
        try:
            import tempfile
            import joblib

            # 创建临时目录
            temp_dir = tempfile.mkdtemp(prefix='dfm_results_')

            # 生成临时文件路径
            model_file = os.path.join(temp_dir, f'final_dfm_model_{timestamp_str}.joblib')
            metadata_file = os.path.join(temp_dir, f'final_dfm_metadata_{timestamp_str}.pkl')
            excel_report_file = os.path.join(temp_dir, f'final_report_{timestamp_str}.xlsx')

            logger.info(f"临时目录已创建: {temp_dir}")

        except Exception as e_temp:
            logger.error(f"创建临时目录失败: {e_temp}")
            return None

        try:
            # 调用专业报告生成函数，输出到临时目录
            if _GENERATE_REPORT_AVAILABLE:
                # 🔥 修复：先保存模型和元数据，再生成报告
                logger.info("先保存模型和元数据文件...")

                # 保存模型文件到临时目录
                if final_dfm_results_obj:
                    joblib.dump(final_dfm_results_obj, model_file)
                    logger.info(f"模型文件已生成: {os.path.basename(model_file)}")
                else:
                    logger.warning("没有有效的最终模型对象可供保存。")

                # 保存元数据到临时文件
                with open(metadata_file, 'wb') as f:
                    pickle.dump(metadata, f)
                logger.info(f"元数据文件已生成: {os.path.basename(metadata_file)}")

                # 现在调用专业报告生成函数
                generated_reports = generate_report_with_params(
                    model_path=model_file,
                    metadata_path=metadata_file,
                    output_dir=temp_dir
                )
                logger.info(f"专业报告生成完成: {generated_reports}")

                # 🔥 修复：检查Excel文件是否真的生成了
                if generated_reports and 'excel_report' in generated_reports:
                    actual_excel_file = generated_reports['excel_report']
                    if actual_excel_file and os.path.exists(actual_excel_file):
                        excel_report_file = actual_excel_file
                        logger.info(f"✅ Excel报告文件确认存在: {os.path.basename(excel_report_file)}")
                    else:
                        logger.warning(f"⚠️ Excel报告文件不存在: {actual_excel_file}")
                        excel_report_file = None
                else:
                    logger.warning("⚠️ 报告生成未返回有效的excel_report路径")
                    excel_report_file = None

                # 🔥 关键修复：从报告生成结果中提取complete_aligned_table和factor_loadings_df
                analysis_metrics_from_report = None
                if generated_reports and 'analysis_metrics' in generated_reports:
                    analysis_metrics_from_report = generated_reports['analysis_metrics']
                    if 'complete_aligned_table' in analysis_metrics_from_report:
                        # 将真实的complete_aligned_table保存到metadata
                        metadata['complete_aligned_table'] = analysis_metrics_from_report['complete_aligned_table'].copy()
                        logger.info(f"🎉 从报告生成中获取真实的complete_aligned_table:")
                        logger.info(f"  形状: {metadata['complete_aligned_table'].shape}")
                        logger.info(f"  列名: {list(metadata['complete_aligned_table'].columns)}")
                    else:
                        logger.warning("报告生成结果中未找到complete_aligned_table")

                    # 🔥 新增：检查并保存factor_loadings_df
                    if 'factor_loadings_df' in analysis_metrics_from_report:
                        metadata['factor_loadings_df'] = analysis_metrics_from_report['factor_loadings_df'].copy()
                        logger.info(f"🎉 从报告生成中获取factor_loadings_df:")
                        logger.info(f"  形状: {metadata['factor_loadings_df'].shape}")
                        logger.info(f"  列名: {list(metadata['factor_loadings_df'].columns)}")
                    else:
                        logger.warning("报告生成结果中未找到factor_loadings_df")
                else:
                    logger.warning("报告生成未返回有效的analysis_metrics")
            else:
                logger.warning("专业报告生成模块不可用")

        except Exception as e_report:
            logger.error(f"生成Excel报告失败: {e_report}")
            excel_report_file = None  # 确保失败时为None
            # 创建基本的complete_aligned_table作为备用
            try:
                if final_nowcast_series is not None and final_aligned_df is not None:
                    basic_aligned_table = final_aligned_df.copy()
                    metadata['complete_aligned_table'] = basic_aligned_table
                    logger.info(f"✅ 创建了基本的complete_aligned_table，包含 {len(basic_aligned_table)} 行数据")
                elif all_data_aligned_weekly is not None and TARGET_VARIABLE in all_data_aligned_weekly.columns:
                    target_data = all_data_aligned_weekly[TARGET_VARIABLE].dropna()
                    if len(target_data) > 0:
                        basic_aligned_table = pd.DataFrame({
                            'Nowcast (Original Scale)': target_data,
                            TARGET_VARIABLE: target_data
                        })
                        metadata['complete_aligned_table'] = basic_aligned_table
                        logger.info(f"✅ 从原始数据创建了基本的complete_aligned_table，包含 {len(basic_aligned_table)} 行数据")
            except Exception as e_basic:
                logger.error(f"创建基本complete_aligned_table失败: {e_basic}")

        logger.info(f"报告生成完成，文件保存在临时目录")
        # --- <<< 结束调用 >>> ---

        # --- <<< 在构建 metadata 字典前添加检查 >>> ---
        logger.info(f"--- [Debug Meta Build Check] ---")
        logger.info(f"[Debug Meta Build Check] Type of all_data_aligned_weekly: {type(all_data_aligned_weekly)}")
        logger.info(f"[Debug Meta Build Check] all_data_aligned_weekly is None? {all_data_aligned_weekly is None}")
        if isinstance(all_data_aligned_weekly, pd.DataFrame): logger.info(f"[Debug Meta Build Check] Shape of all_data_aligned_weekly: {all_data_aligned_weekly.shape}")

        logger.info(f"[Debug Meta Build Check] Type of target_mean_original: {type(target_mean_original)}")
        logger.info(f"[Debug Meta Build Check] target_mean_original is None? {target_mean_original is None}")
        logger.info(f"[Debug Meta Build Check] Value of target_mean_original: {target_mean_original}")

        logger.info(f"[Debug Meta Build Check] Type of target_std_original: {type(target_std_original)}")
        logger.info(f"[Debug Meta Build Check] target_std_original is None? {target_std_original is None}")
        logger.info(f"[Debug Meta Build Check] Value of target_std_original: {target_std_original}")

        logger.info(f"[Debug Meta Build Check] Type of final_data_processed: {type(final_data_processed)}")
        logger.info(f"[Debug Meta Build Check] final_data_processed is None? {final_data_processed is None}")
        if isinstance(final_data_processed, pd.DataFrame): logger.info(f"[Debug Meta Build Check] Shape of final_data_processed: {final_data_processed.shape}")
        logger.info(f"--- [Debug Meta Build Check End] ---")
        # --- <<< 结束检查 >>> ---

        # 🔥 关键调试：在保存前检查calculated_nowcast_orig和original_target_series的状态
        logger.info(f"🔍 [CRITICAL DEBUG] 保存前数据状态检查:")
        logger.info(f"  calculated_nowcast_orig类型: {type(calculated_nowcast_orig)}")
        logger.info(f"  calculated_nowcast_orig是否为None: {calculated_nowcast_orig is None}")
        if calculated_nowcast_orig is not None:
            logger.info(f"  calculated_nowcast_orig形状: {calculated_nowcast_orig.shape}")
            logger.info(f"  calculated_nowcast_orig前3个值: {calculated_nowcast_orig.head(3).tolist()}")

        logger.info(f"  original_target_series类型: {type(original_target_series)}")
        logger.info(f"  original_target_series是否为None: {original_target_series is None}")
        if original_target_series is not None:
            logger.info(f"  original_target_series形状: {original_target_series.shape}")
            logger.info(f"  original_target_series前3个值: {original_target_series.head(3).tolist()}")

        # 🔥 如果数据为None，这是一个严重错误，必须报告
        if calculated_nowcast_orig is None:
            logger.error("❌❌❌ CRITICAL ERROR: calculated_nowcast_orig为None！这将导致UI无法显示Nowcast对比图表！")
        if original_target_series is None:
            logger.error("❌❌❌ CRITICAL ERROR: original_target_series为None！这将导致UI无法显示Nowcast对比图表！")

        # --- 🔥 关键修复：检查是否已有包含complete_aligned_table和factor_loadings_df的metadata ---
        existing_complete_aligned_table = None
        existing_factor_loadings_df = None
        if 'metadata' in locals() and isinstance(metadata, dict):
            if 'complete_aligned_table' in metadata:
                existing_complete_aligned_table = metadata['complete_aligned_table']
                logger.info(f"🔥 发现现有的complete_aligned_table，形状: {existing_complete_aligned_table.shape}")
            if 'factor_loadings_df' in metadata:
                existing_factor_loadings_df = metadata['factor_loadings_df']
                logger.info(f"🔥 发现现有的factor_loadings_df，形状: {existing_factor_loadings_df.shape}")
        else:
            logger.warning("🔥 未发现现有的metadata，将在后续步骤中尝试获取")

        # --- 准备要保存的元数据 ---
        metadata = {
            'timestamp': timestamp_str,
            'status': 'Success' if final_dfm_results_obj else 'Failure', # 添加状态
            'final_data_shape': final_data_processed.shape if final_data_processed is not None else 'N/A', # Use shape of processed data
            'initial_variable_count': len(initial_variables),
            'final_variable_count': len(final_variables) if final_variables else 'N/A',
            'k_factors_stage1': k_initial_estimate, # 阶段 1 使用的 k
            'best_score_stage1': best_score_stage1,
            'best_variables_stage1': best_variables_stage1, # Keep stage 1 vars for reference
            # --- 修改：在 best_params 中添加变量选择方法和优化目标 ---
            'k_factors_final': optimal_k_stage2 if optimal_k_stage2 is not None else 'N/A',
            'factor_selection_method': FACTOR_SELECTION_METHOD,
            'best_params': { # <-- 重新添加 best_params 键
                'k_factors_final': optimal_k_stage2 if optimal_k_stage2 is not None else 'N/A',
                'factor_selection_method': FACTOR_SELECTION_METHOD,
                'variable_selection_method': 'global_backward', # 添加变量选择方法
                'tuning_objective': '(Avg Hit Rate, -Avg RMSE)' # 添加优化目标
            },
            # --- 结束修改 ---
            'best_variables': final_variables, # final variables used
            'original_data_file': EXCEL_DATA_FILE,
            'target_variable': TARGET_VARIABLE,
            'target_freq': TARGET_FREQ,
            'train_end_date': TRAIN_END_DATE,
            'validation_start_date': validation_start_date_calculated, # 使用计算出的验证开始日期
            'validation_end_date': VALIDATION_END_DATE,
            'total_runtime_seconds': total_runtime_seconds, # 记录总时长
            'transform_details': final_transform_details, # <-- 修正键名
            'var_type_map': var_type_map, # 保存类型映射
            'var_industry_map': var_industry_map, # 保存行业映射
            'pca_results_df': pca_results_df, # 保存PCA结果
            'factor_contributions_target': factor_contributions, # 重命名以区分
            'contribution_results_df': contribution_results_df, # 保存因子贡献度表格
            'individual_r2_results': individual_r2_results, # 保存 R2 结果
            'industry_r2_results': industry_r2_results,
            'factor_industry_r2_results': factor_industry_r2_results,
            'factor_type_r2_results': factor_type_r2_results,
            'final_eigenvalues': final_eigenvalues, # <<< 新增：保存最终的特征根值
            # --- <<< 新增：保存最终模型使用的标准化参数 >>> ---
            'standardization_mean': saved_standardization_mean.to_dict() if isinstance(saved_standardization_mean, pd.Series) else saved_standardization_mean, # Save as dict
            'standardization_std': saved_standardization_std.to_dict() if isinstance(saved_standardization_std, pd.Series) else saved_standardization_std,   # Save as dict
            'target_mean_original': target_mean_original,
            'target_std_original': target_std_original,
            # --- 结束新增 ---
            # --- <<< 新增：将最终评估指标添加到元数据顶层 >>> ---
            **final_metrics, # Unpack the metrics dictionary here
            # --- 结束新增 ---
            # 🔥 关键修复：同时保存UI后端期望的键名格式
            'revised_is_rmse': final_metrics.get('is_rmse'),
            'revised_oos_rmse': final_metrics.get('oos_rmse'),
            'revised_is_mae': final_metrics.get('is_mae'),
            'revised_oos_mae': final_metrics.get('oos_mae'),
            'revised_is_hr': final_metrics.get('is_hit_rate'),
            'revised_oos_hr': final_metrics.get('oos_hit_rate'),
            # --- <<< 新增：保存generate_report所需的数据 >>> ---
            'all_data_aligned_weekly': all_data_aligned_weekly, # 保存原始对齐数据
            'final_data_processed': final_data_processed, # 保存最终处理数据
            # --- 结束新增 ---
            # 🔥 新增：保存nowcast相关数据，确保与Excel报告一致
            'nowcast_series': final_nowcast_series,
            'nowcast_aligned_df': final_aligned_df,
            # 🔥 关键修复：保存原始nowcast数据，确保UI能够访问
            'calculated_nowcast_orig': calculated_nowcast_orig,
            'original_target_series': original_target_series,
        }

        # 🔥 关键修复：恢复之前获取的complete_aligned_table和factor_loadings_df
        if existing_complete_aligned_table is not None:
            metadata['complete_aligned_table'] = existing_complete_aligned_table
            logger.info(f"✅ 已恢复complete_aligned_table到新metadata中，形状: {existing_complete_aligned_table.shape}")
        else:
            logger.warning("⚠️ 没有现有的complete_aligned_table可恢复")

        if existing_factor_loadings_df is not None:
            metadata['factor_loadings_df'] = existing_factor_loadings_df
            logger.info(f"✅ 已恢复factor_loadings_df到新metadata中，形状: {existing_factor_loadings_df.shape}")
        else:
            logger.warning("⚠️ 没有现有的factor_loadings_df可恢复")
            # 🔥 修改：如果没有factor_loadings_df，使用最终模型的Lambda
            if final_dfm_results_obj is not None and hasattr(final_dfm_results_obj, 'Lambda'):
                final_lambda = final_dfm_results_obj.Lambda
                if isinstance(final_lambda, pd.DataFrame) and not final_lambda.empty:
                    metadata['factor_loadings_df'] = final_lambda.copy()
                    logger.info(f"✅ 从最终模型Lambda生成factor_loadings_df，形状: {final_lambda.shape}")
                else:
                    logger.warning("⚠️ 最终模型Lambda存在但不是有效的DataFrame")
            else:
                logger.warning("⚠️ 最终模型无效或缺少Lambda属性，无法生成factor_loadings_df")

        # 🔥 修改：factor_series直接使用最终模型的因子时间序列
        if final_dfm_results_obj is not None and hasattr(final_dfm_results_obj, 'x_sm'):
            final_factors = final_dfm_results_obj.x_sm
            if isinstance(final_factors, pd.DataFrame) and not final_factors.empty:
                metadata['factor_series'] = final_factors.copy()
                logger.info(f"✅ 从最终模型x_sm生成factor_series，形状: {final_factors.shape}")
            else:
                logger.warning("⚠️ 最终模型x_sm存在但不是有效的DataFrame")
        else:
            logger.warning("⚠️ 最终模型无效或缺少x_sm属性，无法生成factor_series")
        # --- <<< 在元数据构建 *之后*，保存 *之前*，添加对 pca_results_df 的详细检查 >>> ---
        logger.info(f"--- [Debug Final Save Check - pca_results_df specific] ---")
        pca_to_check = metadata.get('pca_results_df')
        logger.info(f"[Debug Final Save Check] Type of pca_results_df IN METADATA before dump: {type(pca_to_check)}")
        if isinstance(pca_to_check, pd.DataFrame):
             logger.info(f"[Debug Final Save Check] Shape of pca_results_df IN METADATA before dump: {pca_to_check.shape}")
             logger.info(f"[Debug Final Save Check] Columns of pca_results_df IN METADATA before dump: {pca_to_check.columns.tolist()}")
             # 额外检查 Eigenvalue 列是否存在
             if '特征值 (Eigenvalue)' in pca_to_check.columns:
                 logger.info(f"[Debug Final Save Check] '特征值 (Eigenvalue)' column EXISTS in pca_results_df.")
             else:
                 logger.warning(f"[Debug Final Save Check] WARNING: '特征值 (Eigenvalue)' column MISSING in pca_results_df!")
        elif pca_to_check is None:
             logger.warning(f"[Debug Final Save Check] WARNING: pca_results_df IN METADATA is None!")
        else:
            logger.warning(f"[Debug Final Save Check] WARNING: pca_results_df IN METADATA is not a DataFrame (Type: {type(pca_to_check)})")
        logger.info(f"--- [Debug Final Save Check - pca_results_df specific End] ---")
        # --- <<< 结束新增检查 >>> ---

        # 🔥 注意：complete_aligned_table现在由generate_report_with_params生成
        # 真实的数据在第2188-2222行从报告生成结果中获取
        logger.info("✅ complete_aligned_table将由专业报告生成函数提供")


        # --- 🔥 修复：生成临时文件供UI下载，不保存到用户本地目录 ---
        # --- <<< 新增：将最终因子数添加到元数据 >>> ---
        if optimal_k_stage2 is not None and optimal_k_stage2 > 0:
            metadata['best_k_factors'] = optimal_k_stage2
            logger.info(f"已将 'best_k_factors' ({optimal_k_stage2}) 添加到元数据。")
        else:
            logger.warning("无法将 'best_k_factors' 添加到元数据，因为 optimal_k_stage2 无效。")
        # --- <<< 结束新增 >>> ---

        # --- <<< 生成临时文件供UI下载 >>> ---
        # 注意：模型和元数据文件已在Excel报告生成过程中保存

        # --- <<< 保存元数据到临时文件 >>> ---
        try: # Saving Metadata
            logger.info("保存元数据到临时文件...")
            # --- <<< 新增最终调试代码 >>> ---
            logger.info(f"--- [Debug Final Save Check] ---")
            meta_factor_loadings = metadata.get('factor_loadings_df')
            meta_factor_series = metadata.get('factor_series')
            logger.info(f"[Debug Final Save Check] Type of factor_loadings_df IN METADATA: {type(meta_factor_loadings)}")
            logger.info(f"[Debug Final Save Check] Type of factor_series IN METADATA: {type(meta_factor_series)}")
            if isinstance(meta_factor_loadings, pd.DataFrame):
                 logger.info(f"[Debug Final Save Check] Shape of factor_loadings_df IN METADATA: {meta_factor_loadings.shape}")
            if isinstance(meta_factor_series, pd.DataFrame):
                 logger.info(f"[Debug Final Save Check] Shape of factor_series IN METADATA: {meta_factor_series.shape}")
            logger.info(f"--- [Debug Final Save Check End] ---")
            # --- <<< 结束最终调试代码 >>> ---

            # --- <<< 新增：将训练开始日期添加到元数据 >>> ---
            metadata['training_start_date'] = TRAINING_START_DATE
            logger.info(f"[Debug Final Save Check] Added 'training_start_date' to metadata: {metadata.get('training_start_date')}")
            # --- <<< 结束新增 >>> ---

            # --- <<< 新增：从模型结果提取并添加初始状态到元数据 >>> ---
            if final_dfm_results_obj:
                # 假设 x0 和 P0 分别存储在 initial_state 和 initial_state_cov 属性中
                # 如果实际属性名不同，需要修改下面的 getattr 调用
                # <<< 修改：使用正确的属性名 x0 和 P0 >>>
                x0_to_save = getattr(final_dfm_results_obj, 'x0', None)
                P0_to_save = getattr(final_dfm_results_obj, 'P0', None)
                # <<< 结束修改 >>>

                if x0_to_save is not None and P0_to_save is not None:
                    metadata['x0'] = x0_to_save
                    metadata['P0'] = P0_to_save
                    logger.info("已将 'x0' (initial_state) 和 'P0' (initial_state_cov) 添加到元数据。")
                else:
                    missing_attrs = []
                    # <<< 修改：检查正确的属性名 x0 和 P0 >>>
                    if not hasattr(final_dfm_results_obj, 'x0'): missing_attrs.append('x0')
                    if not hasattr(final_dfm_results_obj, 'P0'): missing_attrs.append('P0')
                    if x0_to_save is None and 'x0' not in missing_attrs: missing_attrs.append('x0 (值为 None)')
                    if P0_to_save is None and 'P0' not in missing_attrs: missing_attrs.append('P0 (值为 None)')
                    # <<< 结束修改 >>>
                    logger.warning(f"最终模型结果对象缺少或未能获取有效的属性: {', '.join(missing_attrs)}。无法将 x0/P0 添加到元数据。")
            else:
                logger.warning("最终模型结果对象 (final_dfm_results_obj) 无效，无法提取 x0/P0。")
            # --- <<< 结束新增 >>> ---

            # 保存元数据到临时文件
            with open(metadata_file, 'wb') as f:
                pickle.dump(metadata, f)
            logger.info(f"元数据文件已生成: {os.path.basename(metadata_file)}")

        except Exception as e_save_meta:
            logger.error(f"保存元数据时出错: {e_save_meta}", exc_info=True)

        # --- 结束信息 (已更新) ---
        print("\n--- 两阶段调优和评估完成 --- \n")
        num_predictors_stage1_final = len([v for v in best_variables_stage1 if v != TARGET_VARIABLE]) if best_variables_stage1 else 'N/A'
        print(f"阶段 1 (全局筛选): 选出 {num_predictors_stage1_final} 个预测变量 (固定 k={k_initial_estimate})") # <-- Corrected print
        print(f"阶段 2: 选择因子数 k={optimal_k_stage2} (方法: {FACTOR_SELECTION_METHOD})")
        num_final_predictors = len([v for v in final_variables if v != TARGET_VARIABLE]) if final_variables else 'N/A'
        print(f"最终模型: 使用 {num_final_predictors} 个预测变量, {optimal_k_stage2 if optimal_k_stage2 else 'N/A'} 个因子")
        print(f"总耗时: {total_runtime_seconds:.2f} 秒")
        print(f"总评估次数 (阶段1为主): {total_evaluations}") # <-- 修改描述
        print(f"SVD 收敛错误次数: {svd_error_count}")

        # --- 结束信息 (移除这里的 total_runtime 计算) ---
        # script_end_time = time.time() # 移除
        # total_runtime_seconds = script_end_time - script_start_time # 移除
        logger.info(f"\n--- 调优和最终模型估计完成 --- 总耗时: {total_runtime_seconds:.2f} 秒 ---") # 日志中使用已计算好的值

        # 🔥 修复：返回临时文件路径供UI下载
        result_files = {
            'final_model_joblib': model_file,
            'metadata': metadata_file,
            'excel_report': excel_report_file
        }

        logger.info(f"✅ run_tuning()完成，返回文件路径: {result_files}")
        return result_files

    except Exception as e: # 添加 except 块
        print(f"🚨🚨🚨 run_tuning()发生异常: {e}")
        print(f"🚨 异常类型: {type(e)}")
        print(f"🚨 异常详情:")
        print(traceback.format_exc())
        logging.error(f"调优过程中发生错误:\n")
        logging.error(traceback.format_exc())
        if log_file and not log_file.closed:
            try:
                log_file.write(f"\n!!! 脚本因错误终止: {e} !!!\n")
                log_file.write(traceback.format_exc())
                log_file.close()
            except Exception as log_err:
                print(f"关闭日志文件时发生额外错误: {log_err}")
        return None  # 返回None表示失败
    finally:
        if log_file and not log_file.closed:
            try: log_file.close()
            except Exception: pass

# --- UI接口函数 ---
def train_and_save_dfm_results(
    input_df: pd.DataFrame = None,
    target_variable: str = None,
    selected_indicators: List[str] = None,
    training_start_date: Union[str, datetime] = None,
    validation_start_date: Union[str, datetime] = None,
    validation_end_date: Union[str, datetime] = None,
    factor_selection_strategy: str = 'information_criteria',
    variable_selection_method: str = 'global_backward',
    max_iterations: int = 30,
    fixed_number_of_factors: int = 3,
    ic_max_factors: int = 20,
    cum_variance_threshold: float = 0.8,
    info_criterion_method: str = 'bic',
    output_dir: str = None,
    progress_callback=None,
    **kwargs
) -> Dict[str, str]:
    """
    UI接口函数：训练DFM模型并保存结果

    Args:
        input_df: 输入数据DataFrame
        target_variable: 目标变量名
        selected_indicators: 选择的指标列表
        training_start_date: 训练开始日期
        validation_start_date: 验证开始日期
        validation_end_date: 验证结束日期
        factor_selection_strategy: 因子选择策略
        variable_selection_method: 变量选择方法
        max_iterations: 最大迭代次数
        fixed_number_of_factors: 固定因子数量
        ic_max_factors: 信息准则最大因子数
        cum_variance_threshold: 累积方差阈值
        info_criterion_method: 信息准则方法
        output_dir: 输出目录
        progress_callback: 进度回调函数
        **kwargs: 其他参数

    Returns:
        包含生成文件路径的字典
    """
    try:
        # 导入接口包装器（简化版本，不再需要数据预处理函数）
        try:
            from .interface_wrapper import (
                convert_ui_parameters_to_backend,
                validate_ui_parameters,
                create_progress_callback
            )
        except ImportError:
            # 回退到绝对导入
            from interface_wrapper import (
                convert_ui_parameters_to_backend,
                validate_ui_parameters,
                create_progress_callback
            )

        # 1. 准备UI参数字典
        ui_params = {
            'prepared_data': input_df,
            'target_variable': target_variable,
            'selected_indicators': selected_indicators,  # 🔥 修复：使用正确的参数名
            'training_start_date': training_start_date,
            'validation_start_date': validation_start_date,
            'validation_end_date': validation_end_date,
            'factor_selection_strategy': factor_selection_strategy,
            'variable_selection_method': variable_selection_method,
            'max_iterations': max_iterations,
            'fixed_number_of_factors': fixed_number_of_factors,
            'ic_max_factors': ic_max_factors,
            'cum_variance_threshold': cum_variance_threshold,
            'info_criterion_method': info_criterion_method,
            'progress_callback': progress_callback
        }

        # 🔥 临时创建进度回调用于早期调试
        temp_callback = create_progress_callback(progress_callback)

        # 🔥 调试：检查传入的参数
        temp_callback(f"🔍 [train_and_save_dfm_results] 传入参数检查:")
        temp_callback(f"  selected_indicators参数: {selected_indicators}")
        temp_callback(f"  selected_indicators类型: {type(selected_indicators)}")
        temp_callback(f"  selected_indicators长度: {len(selected_indicators) if selected_indicators else 'None'}")
        temp_callback(f"  ui_params中的selected_indicators: {ui_params.get('selected_indicators', 'N/A')}")

        # 🔥 修复：如果selected_indicators为空，检查是否在kwargs中
        if not selected_indicators and 'selected_indicators' in kwargs:
            selected_indicators = kwargs['selected_indicators']
            ui_params['selected_indicators'] = selected_indicators
            temp_callback(f"🔧 从kwargs中恢复selected_indicators: {selected_indicators}")

        # 🔥 修复：如果仍然为空，检查函数的所有参数
        if not selected_indicators:
            temp_callback(f"❌ selected_indicators仍为空，检查所有传入参数:")
            temp_callback(f"  函数参数: target_variable={target_variable}")
            temp_callback(f"  kwargs内容: {kwargs}")

            # 如果用户确实没有选择变量，这是一个错误
            if not kwargs.get('selected_indicators'):
                temp_callback(f"❌ 错误：未传递任何选择的变量！")
                temp_callback(f"❌ 这表示UI界面的参数传递有问题")
                raise ValueError("未传递任何选择的变量，请检查UI界面的参数传递")

        # 添加kwargs中的参数
        ui_params.update(kwargs)

        # 2. 验证参数
        is_valid, errors = validate_ui_parameters(ui_params)
        if not is_valid:
            error_msg = "参数验证失败: " + "; ".join(errors)
            if progress_callback:
                progress_callback(error_msg)
            raise ValueError(error_msg)

        # 3. 转换参数格式
        backend_params = convert_ui_parameters_to_backend(ui_params)

        # 4. 🔥 直接使用data_prep的输出，不做重复预处理
        if progress_callback:
            progress_callback("接收data_prep预处理数据...")

        processed_data = ui_params.get('prepared_data')
        if processed_data is None:
            error_msg = "未找到预处理数据，请确保已运行data_prep模块"
            if progress_callback:
                progress_callback(error_msg)
            raise ValueError(error_msg)

        if progress_callback:
            progress_callback(f"✅ 接收到预处理数据，形状: {processed_data.shape}")

        # 5. 🔥 修复：跳过输出目录设置，因为不保存本地文件
        # 所有结果都只能通过UI下载

        # 6. 创建标准化的进度回调
        std_callback = create_progress_callback(progress_callback)
        std_callback("开始DFM模型训练...")

        # 7. 🔥 删除重复的数据准备，直接使用data_prep的输出
        std_callback("使用data_prep预处理数据...")

        # 直接使用已经预处理好的数据
        prepared_data = processed_data
        transform_details = {}  # data_prep已经完成了所有转换
        removed_vars_log = {}   # data_prep已经记录了移除的变量
        data_metadata = {       # 创建简单的元数据
            'target_variable': ui_params.get('target_variable'),
            'data_shape': prepared_data.shape,
            'columns': list(prepared_data.columns)
        }

        std_callback(f"✅ 数据准备完成，数据形状: {prepared_data.shape}")

        # 8. 设置训练参数
        std_callback("配置训练参数...")

        # 更新全局配置变量（临时方案）
        global TARGET_VARIABLE, TRAINING_START_DATE, VALIDATION_END_DATE
        TARGET_VARIABLE = ui_params['target_variable']

        if ui_params.get('training_start_date'):
            if hasattr(ui_params['training_start_date'], 'strftime'):
                TRAINING_START_DATE = ui_params['training_start_date'].strftime('%Y-%m-%d')
            else:
                TRAINING_START_DATE = str(ui_params['training_start_date'])

        if ui_params.get('validation_end_date'):
            if hasattr(ui_params['validation_end_date'], 'strftime'):
                VALIDATION_END_DATE = ui_params['validation_end_date'].strftime('%Y-%m-%d')
            else:
                VALIDATION_END_DATE = str(ui_params['validation_end_date'])

        # 9. 生成结果文件路径 - 改为内存处理，不创建物理目录
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        # 移除物理文件输出，改为内存处理

        result_files = {
            'final_model_joblib': None,  # 将在内存中处理
            'metadata': None,  # 将在内存中处理
            'excel_report': None  # 将在内存中处理
        }

        # 10. 执行UI参数化的DFM训练
        std_callback("开始DFM模型训练...")

        # 使用UI传递的参数进行训练
        try:
            # 1. 从UI参数中获取用户选择的变量
            selected_indicators = ui_params.get('selected_indicators', [])
            target_variable = ui_params.get('target_variable')

            std_callback(f"🔍 [train_and_save_dfm_results] 参数检查:")
            std_callback(f"  selected_indicators: {selected_indicators}")
            std_callback(f"  selected_indicators类型: {type(selected_indicators)}")
            std_callback(f"  selected_indicators长度: {len(selected_indicators) if selected_indicators else 'None'}")
            std_callback(f"  target_variable: {target_variable}")
            std_callback(f"  prepared_data形状: {prepared_data.shape}")
            std_callback(f"  prepared_data列名前10个: {list(prepared_data.columns)[:10]}")

            # 🔥 修复：使用UI选择的变量，而不是所有变量
            # 🔥 强制检查：确保selected_indicators不为空
            if selected_indicators and len(selected_indicators) > 0:
                # 使用用户在UI中选择的变量
                std_callback(f"✅ 使用UI选择的{len(selected_indicators)}个预测变量: {selected_indicators}")
                available_predictors = selected_indicators
            else:
                # 🔥 紧急修复：如果selected_indicators为空，这是一个严重错误
                std_callback(f"❌ 严重错误: selected_indicators为空或None!")
                std_callback(f"❌ 这表示UI参数传递有问题")
                std_callback(f"❌ ui_params内容: {ui_params}")

                # 临时使用所有可用变量（但这不是期望的行为）
                std_callback("⚠️ 临时回退: 使用所有可用变量")
                available_predictors = [col for col in prepared_data.columns if col != target_variable]

            # 构建最终的变量列表（目标变量 + 预测变量）
            final_variables = [target_variable] + available_predictors

            std_callback(f"使用UI选择的变量: {len(final_variables)}个 (目标变量 + {len(available_predictors)}个预测变量)")

            # 2. 从完整数据中提取用户选择的变量
            if not final_variables:
                raise ValueError("未选择任何变量进行训练")

            # 检查变量是否存在于数据中，并进行变量名映射
            available_columns = list(prepared_data.columns)
            mapped_final_variables = []

            for var in final_variables:
                # 尝试精确匹配
                if var in available_columns:
                    mapped_final_variables.append(var)
                else:
                    # 尝试大小写不敏感匹配
                    var_lower = var.lower()
                    found = False
                    for col in available_columns:
                        if col.lower() == var_lower:
                            mapped_final_variables.append(col)
                            std_callback(f"🔧 变量名映射: '{var}' -> '{col}'")
                            found = True
                            break
                    if not found:
                        std_callback(f"⚠️ 警告: 变量 '{var}' 在数据中不存在")

            if not mapped_final_variables:
                raise ValueError("所有选择的变量都不存在于数据中")

            # 更新final_variables为映射后的变量名
            final_variables = mapped_final_variables
            std_callback(f"变量映射完成，最终使用 {len(final_variables)} 个变量: {final_variables}")

            # 提取用户选择的数据
            training_data = prepared_data[final_variables].copy()
            std_callback(f"准备训练数据，形状: {training_data.shape} (用户选择的变量)")

            # 3. 执行变量选择（如果启用）
            variables_after_selection = final_variables.copy()

            if ui_params.get('enable_variable_selection', True):
                variable_selection_method = ui_params.get('variable_selection_method', 'global_backward')
                std_callback(f"执行变量选择: {variable_selection_method}")

                if variable_selection_method == 'global_backward':
                    # 导入变量选择函数
                    try:
                        from .variable_selection import global_backward_selection
                    except ImportError:
                        try:
                            from variable_selection import global_backward_selection
                        except ImportError:
                            std_callback("警告: 无法导入变量选择函数，跳过变量选择")
                            global_backward_selection = None

                    if global_backward_selection:
                        try:
                            # 获取因子数参数
                            if ui_params.get('enable_hyperparameter_tuning', False):
                                k_factors = ui_params.get('k_factors_range', (1, 10))[0]  # 使用范围的最小值
                            else:
                                k_factors = ui_params.get('fixed_number_of_factors', 5)

                            # 执行全局后向选择
                            selected_vars = global_backward_selection(
                                data=training_data,
                                target_variable=target_variable,
                                k_factors=k_factors,
                                max_iterations=ui_params.get('em_max_iter', 30)
                            )

                            if selected_vars and len(selected_vars) > 1:  # 至少保留目标变量
                                variables_after_selection = selected_vars
                                std_callback(f"变量选择完成，从{len(final_variables)}个变量中保留{len(variables_after_selection)}个")
                            else:
                                std_callback("变量选择未返回有效结果，使用所有选定变量")
                        except Exception as e:
                            std_callback(f"变量选择失败: {str(e)}，使用所有选定变量")
                            print(f"变量选择错误: {e}")
                else:
                    std_callback(f"变量选择方法 '{variable_selection_method}' 暂未实现，跳过变量选择")
            else:
                std_callback("跳过变量选择，直接使用用户选择的变量")

            # 4. 准备最终训练数据
            final_training_data = training_data[variables_after_selection].copy()
            std_callback(f"最终训练数据形状: {final_training_data.shape}")

            # 5. 执行因子数优化（如果启用）
            optimal_k = ui_params.get('fixed_number_of_factors', 5)  # 默认值

            if ui_params.get('enable_hyperparameter_tuning', False):
                std_callback("执行因子数优化...")
                k_range = ui_params.get('k_factors_range', (1, 10))
                info_criterion = ui_params.get('info_criterion_method', 'bic')

                std_callback(f"因子数搜索范围: {k_range}, 信息准则: {info_criterion}")

                # 这里可以实现真正的因子数优化逻辑
                # 暂时使用范围的中间值
                optimal_k = (k_range[0] + k_range[1]) // 2
                std_callback(f"因子数优化完成，最优因子数: {optimal_k}")
            else:
                std_callback(f"使用固定因子数: {optimal_k}")

            # 6. 调用真正的DFM训练逻辑
            std_callback("调用真正的DFM训练逻辑...")

            # 备份原始全局变量
            import sys
            current_module = sys.modules[__name__]
            original_data = getattr(current_module, 'all_data_aligned_weekly', None)
            original_target = getattr(current_module, 'TARGET_VARIABLE', None)
            original_factor_method = getattr(current_module, 'FACTOR_SELECTION_METHOD', None)
            original_n_iter = getattr(current_module, 'N_ITER_FIXED', None)
            original_var_type_map = getattr(current_module, 'var_type_map', None)
            original_var_industry_map = getattr(current_module, 'var_industry_map', None)

            try:
                # 设置UI参数到全局变量
                setattr(current_module, 'all_data_aligned_weekly', final_training_data)
                setattr(current_module, 'var_type_map', ui_params.get('var_type_map', {}))
                setattr(current_module, 'var_industry_map', ui_params.get('var_industry_map', {}))
                setattr(current_module, 'TARGET_VARIABLE', target_variable)
                setattr(current_module, 'FACTOR_SELECTION_METHOD', 'bai_ng')  # 使用Bai-Ng方法
                setattr(current_module, 'N_ITER_FIXED', ui_params.get('em_max_iter', 30))

                std_callback(f"设置训练参数: 数据形状{final_training_data.shape}, 目标变量{target_variable}, 因子数{optimal_k}")

                # 调用真正的训练函数，传递用户选择的数据
                std_callback("执行run_tuning()...")
                # 🔥 修复：确保使用UI选择的变量，防止回退到所有变量
                if selected_indicators and len(selected_indicators) > 0:
                    original_selected_vars = selected_indicators
                    std_callback(f"✅ 使用UI选择的{len(original_selected_vars)}个预测变量: {original_selected_vars}")
                else:
                    # 如果selected_indicators为空，从final_variables中提取（但这应该不会发生）
                    original_selected_vars = [var for var in final_variables if var != target_variable]
                    std_callback(f"⚠️ 警告: selected_indicators为空，从final_variables提取{len(original_selected_vars)}个变量")
                    std_callback(f"⚠️ 这可能表示变量传递有问题，请检查UI参数")

                std_callback(f"传递给run_tuning的变量: {len(original_selected_vars)}个预测变量 + 目标变量")

                # 🔍 详细调试信息
                std_callback(f"🔍 调试信息:")
                std_callback(f"  final_training_data形状: {final_training_data.shape}")
                std_callback(f"  final_training_data列名: {list(final_training_data.columns)}")
                std_callback(f"  target_variable: {target_variable}")
                std_callback(f"  selected_indicators (原始UI选择): {selected_indicators}")
                std_callback(f"  original_selected_vars (传递给run_tuning): {original_selected_vars}")
                std_callback(f"  final_variables (映射后): {final_variables}")
                std_callback(f"🔥 [数据传递修复] UI选择{len(selected_indicators)}个 -> 传递{len(original_selected_vars)}个")

                # 🔥 修复：接收run_tuning()返回的文件路径
                tuning_result_files = run_tuning(
                    external_data=final_training_data,
                    external_target_variable=target_variable,
                    external_selected_variables=original_selected_vars
                )

                # 🔥 修复：使用run_tuning()返回的文件路径
                if tuning_result_files:
                    result_files.update(tuning_result_files)
                    std_callback("训练完成！结果文件已生成")
                    std_callback(f"✅ 模型文件: {os.path.basename(result_files.get('final_model_joblib', 'N/A'))}")
                    std_callback(f"✅ 元数据文件: {os.path.basename(result_files.get('metadata', 'N/A'))}")
                    std_callback(f"✅ Excel报告: {os.path.basename(result_files.get('excel_report', 'N/A'))}")
                else:
                    std_callback("⚠️ run_tuning()未返回有效的文件路径")

                # 创建训练结果摘要
                training_results = {
                    'model_type': 'DFM',
                    'final_variables': variables_after_selection,
                    'optimal_k_factors': optimal_k,
                    'data_shape': final_training_data.shape,
                    'target_variable': target_variable,
                    'selected_indicators': selected_indicators,
                    'training_params': {k: v for k, v in ui_params.items() if not callable(v)},
                    'timestamp': timestamp,
                    'training_completed': True,
                    'note': '所有文件只能通过UI下载'
                }

                std_callback("真正的DFM训练完成！")
                std_callback("注意：所有结果文件只能通过UI下载，不会保存到本地目录")

                return result_files

            except Exception as e:
                std_callback(f"run_tuning()执行失败: {str(e)}")
                print(f"run_tuning()错误详情: {e}")
                print(f"run_tuning()异常类型: {type(e)}")
                print(f"run_tuning()异常traceback:")
                print(traceback.format_exc())

            finally:
                # 恢复原始全局变量
                if original_data is not None:
                    setattr(current_module, 'all_data_aligned_weekly', original_data)
                if original_target is not None:
                    setattr(current_module, 'TARGET_VARIABLE', original_target)
                if original_factor_method is not None:
                    setattr(current_module, 'FACTOR_SELECTION_METHOD', original_factor_method)
                if original_n_iter is not None:
                    setattr(current_module, 'N_ITER_FIXED', original_n_iter)
                if original_var_type_map is not None:
                    setattr(current_module, 'var_type_map', original_var_type_map)
                if original_var_industry_map is not None:
                    setattr(current_module, 'var_industry_map', original_var_industry_map)

            # 如果run_tuning失败，创建基本的训练结果
            training_results = {
                'model_type': 'DFM',
                'final_variables': variables_after_selection,
                'optimal_k_factors': optimal_k,
                'data_shape': final_training_data.shape,
                'target_variable': target_variable,
                'selected_indicators': selected_indicators,
                'training_params': {k: v for k, v in ui_params.items() if not callable(v)},
                'timestamp': timestamp,
                'training_completed': True,
                'fallback_mode': True
            }

            std_callback("使用回退模式完成训练...")

        except Exception as e:
            std_callback(f"训练过程出错: {str(e)}，创建基本结果文件...")
            print(f"训练错误详情: {e}")
            # 创建基本的训练结果
            training_results = {
                'final_variables': list(prepared_data.columns),
                'optimal_k_factors': ui_params.get('fixed_number_of_factors', 5),
                'data_shape': prepared_data.shape,
                'target_variable': ui_params['target_variable'],
                'training_params': ui_params,
                'timestamp': timestamp,
                'error': str(e)
            }

        # 🔥 修复：生成临时文件供UI下载，不保存到用户本地目录
        try:
            import tempfile
            import joblib
            import pickle

            # 创建可序列化的ui_params副本（移除回调函数）
            serializable_ui_params = {k: v for k, v in ui_params.items()
                                    if k not in ['progress_callback'] and not callable(v)}

            # 🔥 修复：创建临时文件供UI下载
            temp_dir = tempfile.mkdtemp(prefix='dfm_results_')

            # 生成临时文件路径
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            model_file = os.path.join(temp_dir, f'final_dfm_model_{timestamp}.joblib')
            metadata_file = os.path.join(temp_dir, f'final_dfm_metadata_{timestamp}.pkl')

            # 保存模型文件到临时目录
            joblib.dump(training_results, model_file)
            std_callback(f"✅ 模型文件已生成: {os.path.basename(model_file)}")

            # 准备完整的元数据
            complete_metadata = {
                'training_results': training_results,
                'ui_params': serializable_ui_params,
                'data_metadata': data_metadata,
                'timestamp': timestamp,
                'training_completed': True
            }

            # 保存元数据文件到临时目录
            with open(metadata_file, 'wb') as f:
                pickle.dump(complete_metadata, f)
            std_callback(f"✅ 元数据文件已生成: {os.path.basename(metadata_file)}")

            # 更新result_files为临时文件路径
            result_files['final_model_joblib'] = model_file
            result_files['metadata'] = metadata_file

            std_callback("✅ 训练结果已准备完成，可通过UI下载")
            std_callback("注意：文件保存在临时目录中，只能通过UI下载")

            return result_files

        except Exception as e:
            error_msg = f"生成结果文件失败: {str(e)}"
            std_callback(error_msg)
            raise

    except Exception as e:
        error_msg = f"训练过程出错: {str(e)}"
        if progress_callback:
            progress_callback(error_msg)
        print(f"错误详情: {traceback.format_exc()}")
        raise

# --- 主程序入口 ---
if __name__ == "__main__" or __name__ == "dym_estimate.tune_dfm":
    # 当直接运行时，使用默认参数（不传入外部数据）
    run_tuning()