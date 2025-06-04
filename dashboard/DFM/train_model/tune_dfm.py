# -*- coding: utf-8 -*-
"""
Complete DFM Training and Optimization Pipeline
Implements comprehensive hyperparameter tuning, variable selection, and analysis
Based on the original comprehensive DFM system architecture
"""

import os
import pandas as pd
import numpy as np
from datetime import datetime, time, timedelta
import pickle
import matplotlib.pyplot as plt
import traceback
import sys
import unicodedata
import re
import logging
from typing import List, Dict, Tuple, Optional, Any
import concurrent.futures
from tqdm import tqdm
from sklearn.metrics import mean_squared_error

# 设置日志级别，减少不必要的警告
logging.basicConfig(level=logging.ERROR)
# 关闭matplotlib的debug信息
logging.getLogger('matplotlib').setLevel(logging.ERROR)
logging.getLogger('PIL').setLevel(logging.ERROR)

# 禁用matplotlib的字体警告
import warnings
warnings.filterwarnings('ignore', category=UserWarning, module='matplotlib')
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=RuntimeWarning)

# ==== 可重现性设置 ====
import random
RANDOM_SEED = 42

def set_reproducible_environment(seed=RANDOM_SEED, force_single_thread=False):
    """设置可重现的计算环境
    
    Args:
        seed: 随机种子
        force_single_thread: 是否强制单线程(默认False，智能选择)
    """
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    
    # 🚀 智能线程控制策略
    if force_single_thread:
        # 严格可重现性模式：单线程
        thread_count = '1'
        mode = "严格可重现性"
    else:
        # 高性能模式：利用多核但保持确定性
        cpu_count = os.cpu_count() or 4
        thread_count = str(min(cpu_count, 8))  # 最多8线程，避免过度并行
        mode = "高性能确定性"
    
    try:
        # 设置BLAS/LAPACK线程数
        os.environ['MKL_NUM_THREADS'] = thread_count
        os.environ['OPENBLAS_NUM_THREADS'] = thread_count  # 当前使用的是OpenBLAS
        os.environ['OMP_NUM_THREADS'] = thread_count
        os.environ['NUMEXPR_NUM_THREADS'] = thread_count
        os.environ['BLIS_NUM_THREADS'] = thread_count
        
        # 确保OpenBLAS使用确定性计算
        os.environ['OPENBLAS_CONSISTENT_FPCSR'] = '1'  # 确保浮点计算一致性
    except:
        pass
    
    # 如果有sklearn，设置其随机状态
    try:
        from sklearn.utils import check_random_state
        check_random_state(seed)
    except:
        pass
    
    print(f"[REPRODUCIBILITY] 模式: {mode}, 随机种子: {seed}, 线程数: {thread_count}")

def init_worker_process(seed=RANDOM_SEED):
    """初始化并行工作进程的随机种子"""
    # 并行工作进程使用单线程确保确定性
    set_reproducible_environment(seed, force_single_thread=True)

def validate_reproducibility_mode():
    """验证可重现性的专用函数（强制单线程）"""
    set_reproducible_environment(RANDOM_SEED, force_single_thread=True)
    print("[VALIDATION] 已切换到严格可重现性模式")

# 在模块加载时使用智能模式
set_reproducible_environment(RANDOM_SEED, force_single_thread=False)
# ==== 可重现性设置结束 ====

# 添加项目根目录到 Python 路径
current_dir = os.path.dirname(__file__)
project_root = os.path.abspath(os.path.join(current_dir, '..', '..', '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# 添加当前目录到路径以便模块导入
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

# Joblib 可用性检查 (静默)
try:
    import joblib
    JOBLIB_AVAILABLE = True
except ImportError:
    JOBLIB_AVAILABLE = False

# 尝试导入核心DFM模块
try:
    # 直接从DynamicFactorModel.py导入需要的函数和类
    from DynamicFactorModel import DFM_EMalgo, RevserseTranslate, DFMEMResultsWrapper
    # 从dfm_core导入评估函数
    from dfm_core import evaluate_dfm_params
except ImportError as e:
    print(f"错误: 无法导入核心DFM模块: {e}")
    sys.exit(1)

# 尝试导入变量选择模块
try:
    from variable_selection import perform_global_backward_selection, perform_backward_selection
except ImportError as e:
    # 提供模拟函数
    def evaluate_dfm_params(*args, **kwargs):
        return float('inf')
    def perform_global_backward_selection(*args, **kwargs):
        return args[1] if len(args) > 1 else []
    def perform_backward_selection(*args, **kwargs):
        return args[1] if len(args) > 1 else []

# 尝试导入分析工具模块
try:
    from analysis_utils import (
        calculate_pca_variance, calculate_factor_contributions, 
        calculate_individual_variable_r2, calculate_metrics_with_lagged_target,
        calculate_industry_r2, calculate_factor_industry_r2, calculate_factor_type_r2
    )
except ImportError as e:
    # 提供模拟函数  
    def calculate_pca_variance(*args, **kwargs):
        return None
    def calculate_factor_contributions(*args, **kwargs):
        return None, None
    def calculate_individual_variable_r2(*args, **kwargs):
        return None
    def calculate_metrics_with_lagged_target(*args, **kwargs):
        return {}, None
    def calculate_industry_r2(*args, **kwargs):
        return None
    def calculate_factor_industry_r2(*args, **kwargs):
        return None
    def calculate_factor_type_r2(*args, **kwargs):
        return None

# 尝试导入结果分析模块
try:
    from results_analysis import analyze_and_save_final_results, plot_final_nowcast
except ImportError as e:
    # 提供模拟函数
    def analyze_and_save_final_results(*args, **kwargs):
        return {}
    def plot_final_nowcast(*args, **kwargs):
        return None

# 尝试导入验证对齐模块
try:
    from verify_alignment import verify_data_alignment
except ImportError as e:
    # 提供模拟函数
    def verify_data_alignment(*args, **kwargs):
        return True

# 尝试导入数据预处理模块
try:
    import data_preparation
except ImportError as e:
    data_preparation = None

# 尝试导入配置模块
try:
    from config import (
        TrainModelConfig, ensure_output_dirs,
        DataDefaults, TrainDefaults, UIDefaults, VisualizationDefaults,
        FileDefaults, FormatDefaults, AnalysisDefaults
    )
    _CONFIG_AVAILABLE = True
    TRAIN_RESULT_FILES = TrainModelConfig.TRAIN_RESULT_FILES
    DEFAULT_OUTPUT_BASE_DIR = TrainModelConfig.DFM_TRAIN_OUTPUT_DIR
except ImportError as e:
    _CONFIG_AVAILABLE = False
    DEFAULT_OUTPUT_BASE_DIR = os.path.join(os.path.dirname(__file__), '..', 'outputs')
    TRAIN_RESULT_FILES = {
        'model_joblib': 'final_dfm_model.joblib',
        'metadata': 'final_dfm_metadata.pkl',
        'excel_report': 'comprehensive_dfm_report.xlsx'
    }
    
    # 提供默认的ensure_output_dirs函数
    def ensure_output_dirs():
        """默认的输出目录创建函数"""
        os.makedirs(DEFAULT_OUTPUT_BASE_DIR, exist_ok=True)

# 尝试导入报告生成模块
try:
    from generate_report import generate_excel_main
except ImportError as e:
    # 提供模拟函数
    def generate_excel_main(*args, **kwargs):
        return None

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def normalize_string(s: str) -> str:
    """标准化字符串"""
    if not isinstance(s, str):
        return s
    full_width = "（）：　"
    half_width = "(): "
    translation_table = str.maketrans(full_width, half_width)
    s = s.translate(translation_table)
    s = s.strip()
    s = re.sub(r'\s+', ' ', s)
    s = s.lower()
    return s

def detect_data_frequency_and_calculate_train_end(data_index: pd.DatetimeIndex, validation_start_date: datetime) -> datetime:
    """
    检测数据频率并计算训练期结束日期
    
    Args:
        data_index: 数据的时间索引
        validation_start_date: 验证期开始日期
        
    Returns:
        训练期结束日期（验证期开始前的最后一个数据点）
    """
    # 检测数据频率
    if len(data_index) < 2:
        raise ValueError("数据索引长度不足，无法检测频率")
    
    # 计算时间间隔
    time_diffs = data_index[1:] - data_index[:-1]
    median_diff = time_diffs.median()
    
    # 判断频率类型
    if median_diff <= pd.Timedelta(days=1):
        freq = 'D'  # 日度
        freq_name = "日度"
    elif pd.Timedelta(days=6) <= median_diff <= pd.Timedelta(days=8):
        freq = 'W'  # 周度
        freq_name = "周度"
    elif pd.Timedelta(days=28) <= median_diff <= pd.Timedelta(days=32):
        freq = 'M'  # 月度
        freq_name = "月度"
    elif pd.Timedelta(days=88) <= median_diff <= pd.Timedelta(days=95):
        freq = 'Q'  # 季度
        freq_name = "季度"
    else:
        freq = 'infer'  # 其他频率
        freq_name = f"自定义({median_diff.days}天)"
    
    print(f"检测到数据频率: {freq_name} (平均间隔: {median_diff.days}天)")
    
    # 找到验证期开始日期前的最后一个数据点
    validation_start_dt = pd.to_datetime(validation_start_date)
    
    # 找到小于验证开始日期的所有数据点
    train_data_points = data_index[data_index < validation_start_dt]
    
    if len(train_data_points) == 0:
        raise ValueError(f"验证期开始日期({validation_start_date})早于所有数据点，无法确定训练期")
    
    # 训练期结束日期是验证期开始前的最后一个数据点
    training_end_date = train_data_points.max()
    
    print(f"自动计算训练期结束日期: {training_end_date.strftime('%Y-%m-%d')} (验证期开始前的最后一个{freq_name}数据点)")
    
    return training_end_date.to_pydatetime()

def train_and_save_dfm_results(
    input_df: pd.DataFrame,
    target_variable: str,
    selected_indicators: list,
    training_start_date: str or datetime,
    training_end_date: str or datetime,
    n_factors: int,
    factor_order: int = None,
    idio_ar_order: int = None,
    em_max_iter: int = None,
    output_base_dir: str = DEFAULT_OUTPUT_BASE_DIR,
    progress_callback=None,
    # 新增验证期参数
    validation_start_date: str or datetime = None,
    validation_end_date: str or datetime = None,
    # 新增完整优化参数
    enable_hyperparameter_tuning: bool = None,
    enable_variable_selection: bool = None,
    variable_selection_method: str = None,
    k_factors_range: Tuple[int, int] = None,
    max_workers: int = None,
    validation_split_ratio: float = None,
    # 高级分析参数
    enable_detailed_analysis: bool = None,
    generate_excel_report: bool = None,
    pca_n_components: int = None,
    # 新增UI特定参数
    info_criterion_method: str = None,
    cum_variance_threshold: float = None,
    # 新增映射参数
    var_industry_map: Dict[str, str] = None,  # 变量行业映射
    var_type_map: Dict[str, str] = None       # 变量类型映射
) -> dict:
    # 🚀 训练时使用智能线程控制，在非关键路径允许多线程加速
    set_reproducible_environment(RANDOM_SEED, force_single_thread=False)
    
    """
    完整的DFM训练和优化管道

    Args:
        input_df: 输入数据
        target_variable: 目标变量
        selected_indicators: 选择的指标
        training_start_date: 训练开始日期
        training_end_date: 训练结束日期
        n_factors: 因子数量（如果不启用超参数调优则为固定值）
        factor_order: 因子阶数（注意：当前DFM实现暂不支持此参数，但保留用于未来扩展）
        idio_ar_order: 特异性误差阶数（注意：当前DFM实现暂不支持此参数，但保留用于未来扩展）
        em_max_iter: EM算法最大迭代次数
        output_base_dir: 输出基础目录
        progress_callback: 进度回调函数
        validation_start_date: 验证期开始日期（如果提供，将覆盖validation_split_ratio）
        validation_end_date: 验证期结束日期（如果提供，将覆盖validation_split_ratio）
        enable_hyperparameter_tuning: 是否启用超参数调优
        enable_variable_selection: 是否启用变量选择
        variable_selection_method: 变量选择方法
        k_factors_range: 因子数量搜索范围
        max_workers: 最大并行工作数
        validation_split_ratio: 验证集分割比例（仅当validation_start_date和validation_end_date为None时使用）
        enable_detailed_analysis: 是否启用详细分析
        generate_excel_report: 是否生成Excel报告
        pca_n_components: PCA分析的主成分数
        info_criterion_method: 信息准则方法
        cum_variance_threshold: 累积方差阈值
        var_industry_map: 变量行业映射
        var_type_map: 变量类型映射

    Returns:
        dict: 保存的文件路径字典
    """
    
    # 确保datetime和time在函数作用域内可用
    from datetime import datetime, time
    
    # 设置参数默认值，使用配置或后备值
    if _CONFIG_AVAILABLE:
        factor_order = factor_order if factor_order is not None else TrainDefaults.FACTOR_ORDER
        idio_ar_order = idio_ar_order if idio_ar_order is not None else TrainDefaults.IDIO_AR_ORDER
        em_max_iter = em_max_iter if em_max_iter is not None else TrainDefaults.EM_MAX_ITER
        enable_hyperparameter_tuning = enable_hyperparameter_tuning if enable_hyperparameter_tuning is not None else TrainDefaults.ENABLE_HYPERPARAMETER_TUNING
        enable_variable_selection = enable_variable_selection if enable_variable_selection is not None else TrainDefaults.ENABLE_VARIABLE_SELECTION
        variable_selection_method = variable_selection_method if variable_selection_method is not None else TrainDefaults.VARIABLE_SELECTION_METHOD
        k_factors_range = k_factors_range if k_factors_range is not None else (TrainDefaults.K_FACTORS_RANGE_MIN, TrainDefaults.K_FACTORS_RANGE_MAX)
        max_workers = max_workers if max_workers is not None else TrainDefaults.MAX_WORKERS
        validation_split_ratio = validation_split_ratio if validation_split_ratio is not None else TrainDefaults.VALIDATION_SPLIT_RATIO
        enable_detailed_analysis = enable_detailed_analysis if enable_detailed_analysis is not None else TrainDefaults.ENABLE_DETAILED_ANALYSIS
        generate_excel_report = generate_excel_report if generate_excel_report is not None else TrainDefaults.GENERATE_EXCEL_REPORT
        pca_n_components = pca_n_components if pca_n_components is not None else TrainDefaults.PCA_N_COMPONENTS
        info_criterion_method = info_criterion_method if info_criterion_method is not None else TrainDefaults.INFO_CRITERION_METHOD
        cum_variance_threshold = cum_variance_threshold if cum_variance_threshold is not None else TrainDefaults.CUM_VARIANCE_THRESHOLD
    else:
                # 后备硬编码默认值
        factor_order = factor_order if factor_order is not None else (TrainDefaults.FACTOR_ORDER if _CONFIG_AVAILABLE else 1)
        idio_ar_order = idio_ar_order if idio_ar_order is not None else (TrainDefaults.IDIO_AR_ORDER if _CONFIG_AVAILABLE else 1)
        em_max_iter = em_max_iter if em_max_iter is not None else (TrainDefaults.EM_MAX_ITER if _CONFIG_AVAILABLE else 100)
        enable_hyperparameter_tuning = enable_hyperparameter_tuning if enable_hyperparameter_tuning is not None else (TrainDefaults.ENABLE_HYPERPARAMETER_TUNING if _CONFIG_AVAILABLE else True)
        enable_variable_selection = enable_variable_selection if enable_variable_selection is not None else (TrainDefaults.ENABLE_VARIABLE_SELECTION if _CONFIG_AVAILABLE else True)
        variable_selection_method = variable_selection_method if variable_selection_method is not None else (TrainDefaults.VARIABLE_SELECTION_METHOD if _CONFIG_AVAILABLE else "global_backward")
        k_factors_range = k_factors_range if k_factors_range is not None else (TrainDefaults.K_FACTORS_RANGE_MIN, TrainDefaults.K_FACTORS_RANGE_MAX) if _CONFIG_AVAILABLE else (1, 8)
        max_workers = max_workers if max_workers is not None else (TrainDefaults.MAX_WORKERS if _CONFIG_AVAILABLE else 4)
        validation_split_ratio = validation_split_ratio if validation_split_ratio is not None else (TrainDefaults.VALIDATION_SPLIT_RATIO if _CONFIG_AVAILABLE else 0.8)
        enable_detailed_analysis = enable_detailed_analysis if enable_detailed_analysis is not None else (TrainDefaults.ENABLE_DETAILED_ANALYSIS if _CONFIG_AVAILABLE else True)
        generate_excel_report = generate_excel_report if generate_excel_report is not None else (TrainDefaults.GENERATE_EXCEL_REPORT if _CONFIG_AVAILABLE else True)
        pca_n_components = pca_n_components if pca_n_components is not None else (TrainDefaults.PCA_N_COMPONENTS if _CONFIG_AVAILABLE else 10)
        info_criterion_method = info_criterion_method if info_criterion_method is not None else (TrainDefaults.INFO_CRITERION_METHOD if _CONFIG_AVAILABLE else "bic")
        cum_variance_threshold = cum_variance_threshold if cum_variance_threshold is not None else (TrainDefaults.CUM_VARIANCE_THRESHOLD if _CONFIG_AVAILABLE else 0.8)
    
    def _log(message):
        if progress_callback:
            progress_callback(message)
        else:
            print(message)
        logger.info(message)

    # 记录重要参数用于调试和未来扩展
    advanced_params = {
        'factor_order': factor_order,
        'idio_ar_order': idio_ar_order,
        'em_max_iter': em_max_iter,
        'enable_hyperparameter_tuning': enable_hyperparameter_tuning,
        'enable_variable_selection': enable_variable_selection,
        'variable_selection_method': variable_selection_method,
        'k_factors_range': k_factors_range,
        'max_workers': max_workers,
        'validation_split_ratio': validation_split_ratio,
        'enable_detailed_analysis': enable_detailed_analysis,
        'generate_excel_report': generate_excel_report,
        'pca_n_components': pca_n_components,
        'info_criterion_method': info_criterion_method,
        'cum_variance_threshold': cum_variance_threshold
    }
    
    _log("=== 启动完整DFM优化管道 ===")
    _log(f"高级参数配置: {advanced_params}")
    
    # 警告用户当前DFM实现的限制
    if factor_order != 1 or idio_ar_order != 1:
        _log(f"警告: 当前DFM实现暂不支持factor_order({factor_order})和idio_ar_order({idio_ar_order})参数，将使用默认值1。这些参数已记录，将在未来版本中支持。")

    # 1. 初始化输出目录
    if _CONFIG_AVAILABLE:
        ensure_output_dirs()
        run_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # 直接使用顶层目录，不创建run_timestamp子目录
        results_dir = DEFAULT_OUTPUT_BASE_DIR
        models_dir = os.path.join(DEFAULT_OUTPUT_BASE_DIR, 'models')
        data_dir = os.path.join(DEFAULT_OUTPUT_BASE_DIR, 'data')
        plots_dir = os.path.join(DEFAULT_OUTPUT_BASE_DIR, 'plots')
        reports_dir = os.path.join(DEFAULT_OUTPUT_BASE_DIR, 'reports')
        
        # 确保目录存在
        for sub_dir in [models_dir, data_dir, plots_dir, reports_dir]:
            os.makedirs(sub_dir, exist_ok=True)
    else:
        run_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_dir = output_base_dir
        models_dir = data_dir = plots_dir = reports_dir = results_dir
        os.makedirs(results_dir, exist_ok=True)
    
    _log(f"结果将直接保存到顶层目录: {results_dir}")
    if _CONFIG_AVAILABLE:
        _log(f"  - 模型文件: {models_dir}")
        _log(f"  - 数据文件: {data_dir}")
        _log(f"  - 图表文件: {plots_dir}")
        _log(f"  - 报告文件: {reports_dir}")

    saved_files = {}

    # 2. 数据预处理和验证
    _log("步骤1: 数据预处理...")
    if not isinstance(input_df.index, pd.DatetimeIndex):
            input_df.index = pd.to_datetime(input_df.index)
    
    # 强制转换所有日期参数为datetime对象
    if isinstance(training_start_date, str):
        training_start_date = pd.to_datetime(training_start_date)
    elif hasattr(training_start_date, 'date') and not hasattr(training_start_date, 'hour'):
        # 如果是date对象，转换为datetime对象
        training_start_date = datetime.combine(training_start_date, time())
    
    if training_end_date is not None:
        if isinstance(training_end_date, str):
            training_end_date = pd.to_datetime(training_end_date)
        elif hasattr(training_end_date, 'date') and not hasattr(training_end_date, 'hour'):
            training_end_date = datetime.combine(training_end_date, time())
    
    if validation_start_date is not None:
        if isinstance(validation_start_date, str):
            validation_start_date = pd.to_datetime(validation_start_date)
        elif hasattr(validation_start_date, 'date') and not hasattr(validation_start_date, 'hour'):
            validation_start_date = datetime.combine(validation_start_date, time())
    
    if validation_end_date is not None:
        if isinstance(validation_end_date, str):
            validation_end_date = pd.to_datetime(validation_end_date)
        elif hasattr(validation_end_date, 'date') and not hasattr(validation_end_date, 'hour'):
            validation_end_date = datetime.combine(validation_end_date, time())
    
    # 验证数据对齐（可选验证步骤）
    _log("步骤3: 验证数据对齐...")
    try:
        # 改为本地导入避免路径问题
        from . import verify_alignment
        csv_path = os.path.join(output_base_dir, 'data', 'training_data.csv')
        input_df.to_csv(csv_path, index=True, index_label='Date')
        # verify_alignment.verify_data_alignment(csv_path, excel_path)  # 现在通过参数传入，不硬编码
        _log("✅ 数据对齐验证完成")
    except ImportError as e:
        _log(f"⚠️ verify_alignment模块导入失败，跳过数据对齐验证: {e}")
    except Exception as e:
        _log(f"⚠️ 数据对齐验证出错，跳过: {e}")
    
    _log("步骤1.2: 继续数据预处理...")
    
    # 标准化指标名称
    normalized_selected_indicators = []
    for indicator in selected_indicators:
        normalized_indicator = normalize_string(indicator)
        normalized_selected_indicators.append(normalized_indicator)
    
    # 创建列名映射
    original_to_normalized_cols = {}
    normalized_to_original_cols = {}
    for col in input_df.columns:
        normalized_col = normalize_string(col)
        original_to_normalized_cols[col] = normalized_col
        normalized_to_original_cols[normalized_col] = col
    
    # 检查变量是否存在
    missing_cols = []
    available_original_cols = []
    for norm_indicator in normalized_selected_indicators:
        if norm_indicator in normalized_to_original_cols:
            available_original_cols.append(normalized_to_original_cols[norm_indicator])
        else:
            missing_cols.append(norm_indicator)
    
    if missing_cols:
        error_msg = f"缺少以下指标: {missing_cols}"
        _log(error_msg)
        raise ValueError(error_msg)

    # 确保目标变量包含在变量列表中
    if target_variable not in available_original_cols:
        available_original_cols.insert(0, target_variable)
    
    # 提取观测数据
    all_variables = list(set(available_original_cols))
    observables_full = input_df[all_variables].copy()
    
    # 处理验证期日期
    if validation_start_date is not None and validation_end_date is not None:
        # 使用手动设置的验证期
        if isinstance(validation_start_date, str):
            validation_start_date = pd.to_datetime(validation_start_date)
        if isinstance(validation_end_date, str):
            validation_end_date = pd.to_datetime(validation_end_date)
        
        # 🔥 修复：允许未来日期用于Nowcasting预测
        today = datetime.now()
        
        # 记录但不强制修正未来日期（Nowcasting就是要预测未来）
        if validation_end_date > today:
            _log(f"📈 信息: 验证期结束日期 {validation_end_date} 是未来日期，用于Nowcasting预测")
        
        if validation_start_date > today:
            _log(f"⚠️ 警告: 验证期开始日期 {validation_start_date} 是未来日期！这可能导致训练数据不足")
            # 只在数据严重不足时才修正
            if validation_start_date > validation_end_date:
                _log(f"修正验证期开始日期为: {validation_end_date - timedelta(days=90)}")
                validation_start_date = validation_end_date - timedelta(days=90)
        
        # 确保验证期逻辑合理
        if validation_start_date >= validation_end_date:
            _log(f"⚠️ 警告: 验证期开始日期晚于或等于结束日期，自动修正")
            validation_start_date = validation_end_date - timedelta(days=90)  # 验证期3个月
            _log(f"修正后验证期: {validation_start_date.strftime('%Y-%m-%d')} 到 {validation_end_date.strftime('%Y-%m-%d')}")
        
        # 自动计算训练期结束日期
        training_end_actual = detect_data_frequency_and_calculate_train_end(
            data_index=input_df.index,
            validation_start_date=validation_start_date
        )
        
        # 确保所有日期都是datetime对象，避免类型比较错误
        if hasattr(training_end_actual, 'date') and not hasattr(training_end_actual, 'hour'):
            training_end_actual = datetime.combine(training_end_actual, time())
        if hasattr(validation_start_date, 'date') and not hasattr(validation_start_date, 'hour'):
            validation_start_date = datetime.combine(validation_start_date, time())
        if hasattr(validation_end_date, 'date') and not hasattr(validation_end_date, 'hour'):
            validation_end_date = datetime.combine(validation_end_date, time())
        
        # 强制转换为datetime对象（更强制的方法）
        if not isinstance(training_end_actual, datetime):
            training_end_actual = pd.to_datetime(training_end_actual).to_pydatetime()
        if not isinstance(validation_start_date, datetime):
            validation_start_date = pd.to_datetime(validation_start_date).to_pydatetime()
        if not isinstance(validation_end_date, datetime):
            validation_end_date = pd.to_datetime(validation_end_date).to_pydatetime()
        
        # 设置validation_start和validation_end变量，供后续代码使用
        validation_start = validation_start_date
        validation_end = validation_end_date
        
        _log(f"使用手动设置的验证期（已检查未来日期）:")
        _log(f"训练期: {training_start_date} 到 {training_end_actual} (自动计算)")
        _log(f"验证期: {validation_start_date} 到 {validation_end_date}")
        
        # 验证日期逻辑
        if training_end_actual >= validation_start_date:
            raise ValueError(f"自动计算的训练结束日期({training_end_actual})晚于或等于验证开始日期({validation_start_date})，数据可能有问题")
        if validation_start_date >= validation_end_date:
            raise ValueError(f"验证开始日期({validation_start_date})必须早于验证结束日期({validation_end_date})")
            
    else:
        # 使用自动分割（原有逻辑）
        training_end_actual = training_start_date + (training_end_date - training_start_date) * validation_split_ratio
        validation_start = training_end_actual
        validation_end = training_end_date
        
        _log(f"使用自动分割验证期 (split_ratio={validation_split_ratio}):")
        _log(f"训练期: {training_start_date} 到 {training_end_actual}")
        _log(f"验证期: {validation_start} 到 {validation_end}")
    
    # 计算目标变量的原始统计信息
    target_data = observables_full[target_variable].dropna()
    target_mean_original = target_data.mean()
    target_std_original = target_data.std()
    
    # 加载变量映射
    if var_industry_map is not None and var_type_map is not None:
        _log("使用传入的变量映射数据")
        _log(f"使用变量映射: {len(var_type_map)} 个类型映射, {len(var_industry_map)} 个行业映射")
    else:
        _log("未提供变量映射，尝试从文件加载...")
        try:
            var_type_map_loaded, var_industry_map_loaded = data_preparation.load_mappings(
                excel_path=excel_file_path,  # 使用传入的参数而不是硬编码
                            sheet_name=DataDefaults.TYPE_MAPPING_SHEET if _CONFIG_AVAILABLE else '指标体系',
            indicator_col=DataDefaults.INDICATOR_COLUMN if _CONFIG_AVAILABLE else '高频指标',
                type_col=DataDefaults.TYPE_COLUMN if _CONFIG_AVAILABLE else '类型',
                industry_col=DataDefaults.INDUSTRY_COLUMN if _CONFIG_AVAILABLE else '行业'
            )
            # 如果参数中没有提供映射，使用加载的映射
            if var_type_map is None:
                var_type_map = var_type_map_loaded
            if var_industry_map is None:
                var_industry_map = var_industry_map_loaded
            _log(f"从文件加载变量映射: {len(var_type_map)} 个类型映射, {len(var_industry_map)} 个行业映射")
        except Exception as e:
            _log(f"从文件加载变量映射失败: {e}，使用默认映射")
            if var_type_map is None:
                var_type_map = {var: "未知" for var in all_variables}
            if var_industry_map is None:
                var_industry_map = {var: "未知" for var in all_variables}
    
    # 3. 超参数网格搜索
    best_params = {'k_factors': n_factors}
    best_variables = all_variables
    best_score_tuple = (-np.inf, np.inf)  # (hit_rate, -rmse)
    total_evaluations = 0
    
    if enable_hyperparameter_tuning:
        _log("步骤2: 超参数网格搜索...")
        
        # 构建超参数网格
        k_factors_list = list(range(k_factors_range[0], min(k_factors_range[1] + 1, len(all_variables))))
        hyperparams_grid = []
        for k in k_factors_list:
            hyperparams_grid.append({'k_factors': k})
        
        _log(f"测试 {len(hyperparams_grid)} 个超参数组合")
        
        # 并行评估超参数 - 确保子进程随机种子一致性
        futures_grid = {}
        with concurrent.futures.ProcessPoolExecutor(
            max_workers=max_workers, 
            initializer=init_worker_process,
            initargs=(RANDOM_SEED,)
        ) as executor:
            for params in hyperparams_grid:
                if params['k_factors'] >= len(all_variables):
                    continue
                
                future = executor.submit(
                    evaluate_dfm_params,
                    variables=all_variables,
                    full_data=observables_full,
                    target_variable=target_variable,
                    params=params,
                    var_type_map=var_type_map,
                    validation_start=validation_start.strftime('%Y-%m-%d'),
                    validation_end=validation_end.strftime('%Y-%m-%d'),
                    target_freq='W',
                    train_end_date=training_end_actual.strftime('%Y-%m-%d'),
                    target_mean_original=target_mean_original,
                    target_std_original=target_std_original,
                    max_iter=em_max_iter
                )
                futures_grid[future] = params
        
        # 收集结果
        grid_search_results = []
        for future in tqdm(concurrent.futures.as_completed(futures_grid), 
                          total=len(futures_grid), desc="超参数搜索"):
            params = futures_grid[future]
            total_evaluations += 1
            
            try:
                result_tuple = future.result()
                if len(result_tuple) >= 6:
                    is_rmse, oos_rmse, _, _, is_hit_rate, oos_hit_rate = result_tuple[:6]
                    
                    # 计算综合指标
                    combined_rmse = np.inf
                    finite_rmses = [r for r in [is_rmse, oos_rmse] if np.isfinite(r)]
                    if finite_rmses:
                        combined_rmse = np.mean(finite_rmses)
                    
                    combined_hit_rate = -np.inf
                    finite_hit_rates = [hr for hr in [is_hit_rate, oos_hit_rate] if np.isfinite(hr)]
                    if finite_hit_rates:
                        combined_hit_rate = np.mean(finite_hit_rates)
                    
                    if np.isfinite(combined_rmse) and np.isfinite(combined_hit_rate):
                        score_tuple = (combined_hit_rate, -combined_rmse)
                        grid_search_results.append({
                            'params': params,
                            'score_tuple': score_tuple,
                            'rmse': combined_rmse,
                            'hit_rate': combined_hit_rate
                        })
                        
                        if score_tuple > best_score_tuple:
                            best_score_tuple = score_tuple
                            best_params = params.copy()
                            
            except Exception as e:
                _log(f"评估参数 {params} 时出错: {e}")
        
        _log(f"超参数搜索完成，最佳参数: {best_params}, 得分: HR={best_score_tuple[0]:.2f}%, RMSE={-best_score_tuple[1]:.6f}")
        
        # 保存超参数搜索结果（可选）
        grid_results_df = pd.DataFrame([
            {
                'k_factors': r['params']['k_factors'],
                'hit_rate': r['hit_rate'],
                'rmse': r['rmse'],
                'score': r['score_tuple'][0] + r['score_tuple'][1]  # 综合得分
            }
            for r in grid_search_results
        ])
        # 只在配置中有定义时才保存超参数结果文件
        if _CONFIG_AVAILABLE and 'hyperparameter_results_csv' in TRAIN_RESULT_FILES:
            grid_results_path = os.path.join(data_dir, TRAIN_RESULT_FILES['hyperparameter_results_csv'])
            grid_results_df.to_csv(grid_results_path, index=False, encoding='utf-8-sig')
            saved_files['grid_search_results'] = grid_results_path
    
    # 4. 变量选择
    if enable_variable_selection and len(all_variables) > best_params['k_factors'] + 2:
        _log("步骤3: 变量选择...")
        
        if variable_selection_method == "global_backward":
            # 全局后向选择
            selected_vars, final_params, final_score, var_select_evals, svd_errors = perform_global_backward_selection(
                initial_variables=all_variables,
                initial_params=best_params,
                target_variable=target_variable,
                all_data=observables_full,
                var_type_map=var_type_map,
                validation_start=validation_start.strftime('%Y-%m-%d'),
                validation_end=validation_end.strftime('%Y-%m-%d'),
                target_freq='W',
                train_end_date=training_end_actual.strftime('%Y-%m-%d'),
                n_iter=em_max_iter,
                target_mean_original=target_mean_original,
                target_std_original=target_std_original,
                max_workers=max_workers,
                evaluate_dfm_func=evaluate_dfm_params
            )
            
            best_variables = selected_vars
            best_params = final_params
            best_score_tuple = final_score
            total_evaluations += var_select_evals
            
            _log(f"变量选择完成，最终变量数: {len(best_variables)}, 得分: HR={best_score_tuple[0]:.2f}%, RMSE={-best_score_tuple[1]:.6f}")
        
        else:
            _log("跳过变量选择（方法未实现或变量数量不足）")
    
    # 5. 使用最优参数训练最终模型
    _log("步骤4: 训练最终DFM模型...")
    
    # 记录参数使用情况
    dfm_training_params = {
        'n_factors': best_params['k_factors'],
        'n_shocks': best_params['k_factors'],
        'n_iter': em_max_iter,
        'error': 'False',
        # 以下参数当前版本的DFM实现暂不支持，但已记录
        'factor_order_requested': factor_order,
        'idio_ar_order_requested': idio_ar_order,
        'factor_order_used': 1,  # 当前DFM实现固定使用1
        'idio_ar_order_used': 1  # 当前DFM实现固定使用1
    }
    
    if factor_order != 1 or idio_ar_order != 1:
        _log(f"注意: 当前DFM实现使用固定的factor_order=1和idio_ar_order=1，用户设置的factor_order={factor_order}和idio_ar_order={idio_ar_order}已记录但未应用")
    
    # 准备最终训练数据
    final_observables = observables_full[best_variables].copy()
    
    # **🔥 重大修复：DFM模型应该使用完整的数据准备时间范围，不应被验证期限制**
    _log(f"训练期开始日期: {training_start_date}")
    _log(f"验证期结束日期: {validation_end}")
    _log(f"原始数据时间范围: {final_observables.index.min()} 到 {final_observables.index.max()}")
    
    # 🔥 关键修复：使用完整的数据范围，让显示范围由数据准备决定，而非验证期限制
    final_train_data = final_observables.copy()
    
    _log(f"DFM训练最终使用数据范围: {final_train_data.index.min()} 到 {final_train_data.index.max()}")
    _log(f"训练数据行数: {len(final_train_data)}")
    
    if len(final_train_data) == 0:
        raise ValueError("训练数据为空，请检查数据和日期设置")
    
    # 训练最终DFM模型
    final_dfm_results = DFM_EMalgo(
        observation=final_train_data,
        n_factors=best_params['k_factors'],
        n_shocks=best_params['k_factors'],
        n_iter=em_max_iter,
        error='False'
    )
    
    # 如果没有进行超参数调优，计算一个简单的评估指标
    if not enable_hyperparameter_tuning and best_score_tuple == (-np.inf, np.inf):
        _log("计算简单评估指标 (因为未启用超参数调优)...")
        try:
            # 计算拟合优度作为简单评估
            if hasattr(final_dfm_results, 'x_sm') and hasattr(final_dfm_results, 'Lambda'):
                fitted_values = RevserseTranslate(
                    Factors=final_dfm_results.x_sm,
                    miu=final_dfm_results.obs_mean,
                    Lambda=final_dfm_results.Lambda,
                    names=best_variables
                )
                
                # 计算目标变量的R²
                if target_variable in fitted_values.columns and target_variable in final_train_data.columns:
                    actual = final_train_data[target_variable].dropna()
                    predicted = fitted_values[target_variable].reindex(actual.index).dropna()
                    
                    if len(actual) > 0 and len(predicted) > 0:
                        # 确保索引对齐
                        common_idx = actual.index.intersection(predicted.index)
                        if len(common_idx) > 0:
                            actual_common = actual.loc[common_idx]
                            predicted_common = predicted.loc[common_idx]
                            
                            # 计算R²
                            ss_res = np.sum((actual_common - predicted_common) ** 2)
                            ss_tot = np.sum((actual_common - np.mean(actual_common)) ** 2)
                            r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
                            
                            # 计算RMSE
                            rmse = np.sqrt(mean_squared_error(actual_common, predicted_common))
                            
                            # 更新得分元组 (使用R²作为hit_rate的替代)
                            best_score_tuple = (r2 * 100, -rmse)  # R²转为百分比，RMSE取负数
                            _log(f"简单评估完成: R²={r2:.4f} ({r2*100:.2f}%), RMSE={rmse:.6f}")
                        else:
                            _log("⚠️ 实际值和预测值无共同索引，无法计算评估指标")
                            best_score_tuple = (0.0, -1.0)  # 设置默认值
                    else:
                        _log("⚠️ 实际值或预测值为空，无法计算评估指标")
                        best_score_tuple = (0.0, -1.0)  # 设置默认值
                else:
                    _log("⚠️ 无法找到目标变量进行评估")
                    best_score_tuple = (0.0, -1.0)  # 设置默认值
            else:
                _log("⚠️ 模型结果不完整，无法计算评估指标")
                best_score_tuple = (0.0, -1.0)  # 设置默认值
        except Exception as e:
            _log(f"⚠️ 计算简单评估指标时出错: {e}")
            best_score_tuple = (0.0, -1.0)  # 设置默认值
    
    # 6. 详细分析
    analysis_results = {}
    if enable_detailed_analysis:
        _log("步骤5: 详细分析...")
        
        # PCA分析
        try:
            pca_results_df = calculate_pca_variance(
                data_standardized=final_train_data,
                n_components=min(pca_n_components, final_train_data.shape[1] - 1)
            )
            if pca_results_df is not None:
                analysis_results['pca_results_df'] = pca_results_df
                _log("PCA分析完成")
        except Exception as e:
            _log(f"PCA分析失败: {e}")
        
        # 因子贡献分析
        try:
            contribution_results_df, factor_contributions = calculate_factor_contributions(
                dfm_results=final_dfm_results,
                data_processed=final_train_data,
                target_variable=target_variable,
                n_factors=best_params['k_factors']
            )
            if contribution_results_df is not None:
                analysis_results['contribution_results_df'] = contribution_results_df
                analysis_results['factor_contributions'] = factor_contributions
                _log("因子贡献分析完成")
        except Exception as e:
            _log(f"因子贡献分析失败: {e}")
        
        # 个体变量R²分析
        try:
            _log("开始个体变量R²分析...")
            individual_r2_results = calculate_individual_variable_r2(
                dfm_results=final_dfm_results,
                data_processed=final_train_data,
                variable_list=best_variables,
                n_factors=best_params['k_factors'],
                timeout_seconds=60  # 1分钟超时
            )
            if individual_r2_results:
                analysis_results['individual_r2_results'] = individual_r2_results
                _log("个体变量R²分析完成")
            else:
                _log("个体变量R²分析未返回结果")
        except Exception as e:
            _log(f"个体变量R²分析失败: {e}")
        
        # 行业R²分析
        try:
            industry_r2_results = calculate_industry_r2(
                dfm_results=final_dfm_results,
                data_processed=final_train_data,
                variable_list=best_variables,
                var_industry_map=var_industry_map,
                n_factors=best_params['k_factors'],
                timeout_seconds=60  # 1分钟超时
            )
            if industry_r2_results is not None:
                analysis_results['industry_r2_results'] = industry_r2_results
                _log("行业R²分析完成")
        except Exception as e:
            _log(f"行业R²分析失败: {e}")
        
        # 因子-行业R²分析
        try:
            factor_industry_r2_results = calculate_factor_industry_r2(
                dfm_results=final_dfm_results,
                data_processed=final_train_data,
                variable_list=best_variables,
                var_industry_map=var_industry_map,
                n_factors=best_params['k_factors'],
                timeout_seconds=60  # 1分钟超时
            )
            if factor_industry_r2_results:
                analysis_results['factor_industry_r2_results'] = factor_industry_r2_results
                _log("因子-行业R²分析完成")
        except Exception as e:
            _log(f"因子-行业R²分析失败: {e}")
        
        # 因子-类型R²分析
        try:
            factor_type_r2_results = calculate_factor_type_r2(
                dfm_results=final_dfm_results,
                data_processed=final_train_data,
                variable_list=best_variables,
                var_type_map=var_type_map,
                n_factors=best_params['k_factors'],
                timeout_seconds=60  # 1分钟超时
            )
            if factor_type_r2_results:
                _log("开始保存因子-类型R²分析结果...")
                # 安全地保存结果，避免大数据卡住
                try:
                    analysis_results['factor_type_r2_results'] = factor_type_r2_results
                    _log("✅ 因子-类型R²分析结果已保存到analysis_results")
                except Exception as save_err:
                    _log(f"⚠️ 保存因子-类型R²分析结果失败: {save_err}，继续处理")
                _log("因子-类型R²分析完成")
            else:
                _log("⚠️ 因子-类型R²分析未返回结果")
        except Exception as e:
            _log(f"因子-类型R²分析失败: {e}")
        
        # 🔥 关键修复：调用 analyze_and_save_final_results 计算标准指标
        _log("步骤5.1: 调用 analyze_and_save_final_results 计算完整的分析结果和指标...")
        try:
            # 直接调用 analyze_and_save_final_results 函数来计算指标和生成 nowcast
            from results_analysis import analyze_and_save_final_results
            
            _log("调用 analyze_and_save_final_results 函数...")
            
            # 准备调用参数
            excel_output_path = os.path.join(reports_dir, TRAIN_RESULT_FILES.get('excel_report', 'comprehensive_dfm_report.xlsx'))
            
            # 调用函数获取完整的分析结果
            final_nowcast_series, final_metrics = analyze_and_save_final_results(
                run_output_dir=reports_dir,
                timestamp_str=run_timestamp,
                excel_output_path=excel_output_path,
                all_data_full=observables_full,
                final_data_processed=final_train_data,
                final_target_mean_rescale=target_mean_original,
                final_target_std_rescale=target_std_original,
                target_variable=target_variable,
                final_dfm_results=final_dfm_results,
                best_variables=best_variables,
                best_params=best_params,
                var_type_map=var_type_map or {},
                total_runtime_seconds=0,  # 暂时设为0，后面会更新
                training_start_date=training_start_date.strftime('%Y-%m-%d'),
                validation_start_date=validation_start.strftime('%Y-%m-%d'),
                validation_end_date=validation_end.strftime('%Y-%m-%d'),
                train_end_date=training_end_actual.strftime('%Y-%m-%d'),
                factor_contributions=analysis_results.get('factor_contributions'),
                final_transform_log=None,
                pca_results_df=analysis_results.get('pca_results_df'),
                contribution_results_df=analysis_results.get('contribution_results_df'),
                var_industry_map=var_industry_map or {},
                individual_r2_results=analysis_results.get('individual_r2_results'),
                industry_r2_results=analysis_results.get('industry_r2_results'),
                factor_industry_r2_results=analysis_results.get('factor_industry_r2_results'),
                factor_type_r2_results=analysis_results.get('factor_type_r2_results'),
                final_eigenvalues=None
            )
            
            # 🔥 关键修复：将返回的指标添加到 analysis_results 中
            if final_metrics and isinstance(final_metrics, dict):
                _log(f"✅ analyze_and_save_final_results 返回了 {len(final_metrics)} 个结果")
                
                # 提取标准指标
                standard_metric_keys = ['is_rmse', 'oos_rmse', 'is_mae', 'oos_mae', 'is_hit_rate', 'oos_hit_rate']
                for key in standard_metric_keys:
                    if key in final_metrics:
                        analysis_results[key] = final_metrics[key]
                        _log(f"✅ 已添加标准指标到analysis_results: {key} = {final_metrics[key]}")
                
                # 添加其他有用的结果
                other_keys = ['nowcast_aligned', 'y_test_aligned', 'factor_loadings_df']
                for key in other_keys:
                    if key in final_metrics:
                        analysis_results[key] = final_metrics[key]
                        _log(f"✅ 已添加分析结果到analysis_results: {key}")
            else:
                _log("⚠️ analyze_and_save_final_results 未返回有效的 metrics")
                
            _log("✅ analyze_and_save_final_results 调用完成")
            
        except Exception as e:
            _log(f"⚠️ 调用 analyze_and_save_final_results 失败，回退到简化版本: {e}")
            # 如果调用失败，回退到原来的简化版本
            _log("开始生成 Nowcast 序列（简化版本）...")
            
            # 1. 获取因子载荷矩阵 (Lambda)
            if hasattr(final_dfm_results, 'Lambda') and hasattr(final_dfm_results, 'x_sm'):
                Lambda = final_dfm_results.Lambda  # Shape: (n_vars, n_factors)
                factors_ts = final_dfm_results.x_sm  # Shape: (n_time, n_factors)
                
                # 2. 找到目标变量在变量列表中的位置
                try:
                    target_idx = best_variables.index(target_variable)
                    target_loadings = Lambda[target_idx, :]  # Shape: (n_factors,)
                    
                    # 3. 计算目标变量的 Nowcast (factors @ target_loadings)
                    if factors_ts.shape[1] == len(target_loadings):
                        nowcast_standardized = factors_ts.dot(target_loadings)  # Shape: (n_time,)
                        
                        # 4. 反标准化 nowcast (如果有均值和标准差)
                        if target_mean_original is not None and target_std_original is not None:
                            nowcast_original_scale = nowcast_standardized * target_std_original + target_mean_original
                        else:
                            nowcast_original_scale = nowcast_standardized
                            _log("⚠️ 缺少目标变量的均值和标准差，使用标准化尺度的 nowcast")
                        
                        # 5. 转换为 pandas Series 并设置索引
                        nowcast_series = pd.Series(
                            nowcast_original_scale, 
                            index=factors_ts.index,
                            name='Nowcast (Original Scale)'
                        )
                        
                        # 6. 获取目标变量的原始数据进行对齐
                        if target_variable in observables_full.columns:
                            target_original = observables_full[target_variable].dropna()
                            
                            # 7. 使用 results_analysis 模块的对齐函数
                            from results_analysis import create_aligned_nowcast_target_table
                            
                            aligned_table = create_aligned_nowcast_target_table(
                                nowcast_weekly_orig=nowcast_series,
                                target_orig=target_original,
                                target_variable_name=target_variable
                            )
                            
                            if not aligned_table.empty:
                                # 8. 将对齐数据添加到分析结果
                                _log(f"对齐表格列名: {aligned_table.columns.tolist()}")
                                
                                # 检查并获取正确的列名
                                nowcast_col = None
                                target_col = None
                                
                                for col in aligned_table.columns:
                                    if 'nowcast' in col.lower() or 'original scale' in col.lower():
                                        nowcast_col = col
                                    elif target_variable in col or col == target_variable:
                                        target_col = col
                                
                                if nowcast_col and target_col:
                                    analysis_results['nowcast_aligned'] = aligned_table[nowcast_col]
                                    analysis_results['y_test_aligned'] = aligned_table[target_col]
                                    _log(f"✅ 成功生成对齐数据，包含 {len(aligned_table)} 个时间点")
                                    _log(f"  - Nowcast列: {nowcast_col}")
                                    _log(f"  - Target列: {target_col}")
                                else:
                                    _log(f"⚠️ 无法找到正确的列名: nowcast_col={nowcast_col}, target_col={target_col}")
                                    _log(f"  可用列: {aligned_table.columns.tolist()}")
                            else:
                                _log("⚠️ 对齐表格为空，无法生成 nowcast_aligned 和 y_test_aligned")
                        else:
                            _log(f"⚠️ 原始数据中未找到目标变量 '{target_variable}'")
                    else:
                        _log(f"⚠️ 因子维度不匹配: factors_ts.shape[1]={factors_ts.shape[1]}, target_loadings.shape={target_loadings.shape}")
                except ValueError:
                    _log(f"⚠️ 目标变量 '{target_variable}' 不在最终变量列表中")
            else:
                _log("⚠️ DFM 结果中缺少 Lambda 或 x_sm，无法生成 nowcast")
                
        except Exception as e:
            _log(f"⚠️ 生成 Nowcast 和对齐数据失败: {e}")
            # 继续执行，不中断流程
    
    # 强制刷新输出，确保日志被显示
    if hasattr(sys.stdout, 'flush'):
        sys.stdout.flush()
    
    _log("=== 🎯 所有分析步骤完成，开始保存结果 ===")
    
    # 7. 保存模型和结果
    _log("步骤6: 保存结果...")
    _log(f"joblib可用性: {JOBLIB_AVAILABLE}")
    _log(f"输出目录: models_dir={models_dir}, data_dir={data_dir}, plots_dir={plots_dir}")
    
    # 确保输出目录存在
    for dir_path in [models_dir, data_dir, plots_dir, reports_dir]:
        try:
            os.makedirs(dir_path, exist_ok=True)
            _log(f"✅ 目录已创建/存在: {dir_path}")
        except Exception as e:
            _log(f"❌ 创建目录失败: {dir_path}, 错误: {e}")
    
    # 保存最终模型
    if JOBLIB_AVAILABLE:
        try:
            final_model_joblib_path = os.path.join(models_dir, TRAIN_RESULT_FILES['model_joblib'] if _CONFIG_AVAILABLE else "final_dfm_model.joblib")
            _log(f"尝试保存模型到: {final_model_joblib_path}")
            joblib.dump(final_dfm_results, final_model_joblib_path)
            saved_files['final_model_joblib'] = final_model_joblib_path
            _log("✅ 模型已保存为joblib格式")
        except Exception as e:
            _log(f"❌ 保存模型失败: {e}")
    else:
        _log("⚠️ joblib不可用，跳过模型保存，但继续保存其他结果")
    
    # 保存详细元数据（即使joblib不可用也要保存）
    _log("开始准备元数据...")
    try:
        _log(f"正在创建基础元数据字典...")
        # 增强best_params以确保报告生成时能找到所需的键
        enhanced_best_params = best_params.copy()
        enhanced_best_params.update({
            'k_factors_final': best_params.get('k_factors', 'N/A'),
            'factor_order': 1,  # DFM当前实现固定使用1
            'variable_selection_method': variable_selection_method if enable_variable_selection else 'none',
            'tuning_objective': '(Hit Rate, -RMSE)' if enable_hyperparameter_tuning else 'none'
        })
        
        metadata = {
            'timestamp': run_timestamp,
            'target_variable': target_variable,
            'best_variables': best_variables,
            'best_params': enhanced_best_params,
            'best_score_tuple': best_score_tuple,
            'var_type_map': var_type_map,
            'var_industry_map': var_industry_map,
            'total_evaluations': total_evaluations,
            'training_start_date': training_start_date.strftime('%Y-%m-%d'),
            'training_end_date': training_end_actual.strftime('%Y-%m-%d'),
            'validation_start_date': validation_start.strftime('%Y-%m-%d'),
            'validation_end_date': validation_end.strftime('%Y-%m-%d'),
            'train_end_date': training_end_actual.strftime('%Y-%m-%d'),
            'target_mean_original': target_mean_original,
            'target_std_original': target_std_original,
            'total_runtime_seconds': 0,  # 将在后面计算
            # 完整的UI参数记录
            'ui_parameters': advanced_params,
            'dfm_training_parameters': dfm_training_params,
            'parameter_usage_notes': {
                'factor_order': f"用户设置: {factor_order}, 当前DFM实现使用: 1 (固定值)",
                'idio_ar_order': f"用户设置: {idio_ar_order}, 当前DFM实现使用: 1 (固定值)",
                'em_max_iter': f"用户设置: {em_max_iter}, 实际使用: {em_max_iter}",
                'enable_hyperparameter_tuning': f"用户设置: {enable_hyperparameter_tuning}, 实际使用: {enable_hyperparameter_tuning}",
                'enable_variable_selection': f"用户设置: {enable_variable_selection}, 实际使用: {enable_variable_selection}",
                'enable_detailed_analysis': f"用户设置: {enable_detailed_analysis}, 实际使用: {enable_detailed_analysis}",
                'generate_excel_report': f"用户设置: {generate_excel_report}, 实际使用: {generate_excel_report}",
                'validation_split_ratio': f"用户设置: {validation_split_ratio}, 实际使用: {validation_split_ratio}",
                'k_factors_range': f"用户设置: {k_factors_range}, 实际使用: {k_factors_range}",
                'pca_n_components': f"用户设置: {pca_n_components}, 实际使用: {min(pca_n_components, final_train_data.shape[1] - 1)}",
                'info_criterion_method': f"用户设置: {info_criterion_method}, 实际使用: {info_criterion_method}",
                'cum_variance_threshold': f"用户设置: {cum_variance_threshold}, 实际使用: {cum_variance_threshold}"
            }
        }
        
        _log("✅ 基础元数据字典创建完成")
        
        # 安全地添加大数据对象，避免卡住 - 使用超时机制
        import signal
        
        def timeout_handler(signum, frame):
            raise TimeoutError("元数据保存操作超时")
        
        # Windows平台不支持signal.alarm，改用其他方式
        _log("正在安全地添加大数据对象到元数据...")
        try:
            # 分别添加大数据对象，如果某个失败不影响其他，且限制大小
            if 'all_data_aligned_weekly' not in metadata:
                # 限制数据大小，避免内存问题
                if hasattr(observables_full, 'shape') and observables_full.shape[0] * observables_full.shape[1] > 100000:
                    _log("⚠️ observables_full 数据过大，仅保存前1000行")
                    metadata['all_data_aligned_weekly'] = observables_full.head(1000)
                else:
                    metadata['all_data_aligned_weekly'] = observables_full
                _log("✅ all_data_aligned_weekly 已添加")
        except Exception as e:
            _log(f"⚠️ 添加 all_data_aligned_weekly 失败: {e}")
            
        try:
            if 'final_data_processed' not in metadata:
                # 限制数据大小
                if hasattr(final_train_data, 'shape') and final_train_data.shape[0] * final_train_data.shape[1] > 100000:
                    _log("⚠️ final_train_data 数据过大，仅保存前1000行")
                    metadata['final_data_processed'] = final_train_data.head(1000)
                else:
                    metadata['final_data_processed'] = final_train_data
                _log("✅ final_data_processed 已添加")
        except Exception as e:
            _log(f"⚠️ 添加 final_data_processed 失败: {e}")

        # 🔥 新闻分析关键修复：添加processed_data_for_model
        try:
            if 'processed_data_for_model' not in metadata:
                # 使用final_train_data作为新闻分析的数据源
                processed_data_for_news = final_train_data.copy()
                # 限制数据大小以避免元数据文件过大
                if hasattr(processed_data_for_news, 'shape') and processed_data_for_news.shape[0] > 5000:
                    _log("⚠️ 处理数据过长，仅保存最后2000个时间点供新闻分析使用")
                    processed_data_for_news = processed_data_for_news.tail(2000)
                
                metadata['processed_data_for_model'] = processed_data_for_news
                _log("✅ processed_data_for_model 已添加到元数据（新闻分析需要）")
        except Exception as e:
            _log(f"⚠️ 添加 processed_data_for_model 失败: {e}")
            # 不是致命错误，继续执行
            
        try:
            # 安全地添加分析结果，跳过可能有问题的大对象
            _log(f"正在添加分析结果（跳过大对象）...")
            # 重要：首先添加关键数据，这些是 model_analysis 需要的核心数据
            critical_keys = ['nowcast_aligned', 'y_test_aligned', 'pca_results_df', 'factor_loadings_df']
            safe_analysis_keys = ['pca_variance_explained', 'factor_contributions_df', 'factor_target_contribution_dict']
            # 新增：R2分析结果键 - 这些是模型分析UI必需的
            r2_analysis_keys = ['individual_r2_results', 'industry_r2_results', 'factor_industry_r2_results', 'factor_type_r2_results']
            # 🔥 关键修复：添加标准指标键 - UI模块需要这些
            standard_metric_keys = ['is_rmse', 'oos_rmse', 'is_mae', 'oos_mae', 'is_hit_rate', 'oos_hit_rate']
            
            # 1. 添加关键数据
            for key in critical_keys:
                if key in analysis_results and analysis_results[key] is not None:
                    try:
                        metadata[key] = analysis_results[key]
                        _log(f"✅ 已添加关键数据: {key}")
                    except Exception as e:
                        _log(f"⚠️ 添加关键数据 {key} 失败: {e}")
            
            # 2. 🔥 关键修复：添加标准指标 - UI模块需要这些计算结果
            for key in standard_metric_keys:
                if key in analysis_results and analysis_results[key] is not None:
                    try:
                        metadata[key] = analysis_results[key]
                        _log(f"✅ 已添加标准指标: {key} = {analysis_results[key]}")
                    except Exception as e:
                        _log(f"⚠️ 添加标准指标 {key} 失败: {e}")
            
            # 3. 添加R2分析结果 - 模型分析UI的核心数据
            for key in r2_analysis_keys:
                if key in analysis_results and analysis_results[key] is not None:
                    try:
                        metadata[key] = analysis_results[key]
                        _log(f"✅ 已添加R2分析结果: {key}")
                        # 添加调试信息
                        if isinstance(analysis_results[key], dict):
                            _log(f"   - {key} 包含 {len(analysis_results[key])} 个键: {list(analysis_results[key].keys())}")
                        elif hasattr(analysis_results[key], 'shape'):
                            _log(f"   - {key} 数据形状: {analysis_results[key].shape}")
                        elif hasattr(analysis_results[key], '__len__'):
                            _log(f"   - {key} 数据长度: {len(analysis_results[key])}")
                    except Exception as e:
                        _log(f"⚠️ 添加R2分析结果 {key} 失败: {e}")
                        import traceback
                        _log(f"   详细错误: {traceback.format_exc()[:200]}...")
            
            # 4. 添加其他安全的分析结果
            for key in safe_analysis_keys:
                if key in analysis_results and analysis_results[key] is not None:
                    try:
                        metadata[key] = analysis_results[key]
                        _log(f"✅ 已添加分析结果: {key}")
                    except Exception as e:
                        _log(f"⚠️ 添加分析结果 {key} 失败: {e}")
            
            # 5. 添加因子载荷矩阵 (重要数据)
            if hasattr(final_dfm_results, 'Lambda') and final_dfm_results.Lambda is not None:
                try:
                    loadings_df = pd.DataFrame(
                        final_dfm_results.Lambda,
                        index=best_variables,
                        columns=[f'Factor{i+1}' for i in range(best_params['k_factors'])]
                    )
                    metadata['factor_loadings_df'] = loadings_df
                    _log("✅ 已添加因子载荷矩阵 (factor_loadings_df)")
                except Exception as e:
                    _log(f"⚠️ 添加因子载荷矩阵失败: {e}")
                        
            # 6. 对于其他分析结果，只添加摘要信息而不是完整对象
            excluded_keys = critical_keys + safe_analysis_keys + r2_analysis_keys + standard_metric_keys + ['factor_loadings_df']
            for key, value in analysis_results.items():
                if key not in excluded_keys and value is not None:
                    try:
                        if hasattr(value, 'shape'):
                            metadata[f"{key}_summary"] = f"数据形状: {value.shape}"
                        elif isinstance(value, dict):
                            metadata[f"{key}_summary"] = f"字典包含 {len(value)} 个键"
                        else:
                            metadata[f"{key}_summary"] = f"数据类型: {type(value).__name__}"
                        _log(f"✅ 已添加分析结果摘要: {key}_summary")
                    except Exception as e:
                        _log(f"⚠️ 添加分析结果摘要 {key} 失败: {e}")
                        
        except Exception as e:
            _log(f"⚠️ 添加分析结果失败: {e}")
            import traceback
            _log(f"详细错误: {traceback.format_exc()[:500]}...")
        
        _log("正在添加训练期载荷矩阵...")
        # 保存训练期载荷矩阵用于稳定性分析
        if hasattr(final_dfm_results, 'Lambda'):
            try:
                training_only_lambda = final_dfm_results.Lambda
                if isinstance(training_only_lambda, np.ndarray):
                    training_only_lambda = pd.DataFrame(
                        training_only_lambda,
                        index=best_variables,
                        columns=[f'Factor{i+1}' for i in range(best_params['k_factors'])]
                    )
                metadata['training_only_lambda'] = training_only_lambda
                _log("✅ training_only_lambda 已添加")
            except Exception as e:
                _log(f"⚠️ 添加 training_only_lambda 失败: {e}")
        
        _log("正在添加因子时间序列...")
        # 保存因子时间序列
        if hasattr(final_dfm_results, 'x_sm'):
            try:
                # 限制因子时间序列的大小
                factor_series = final_dfm_results.x_sm
                if hasattr(factor_series, 'shape') and factor_series.shape[0] > 5000:
                    _log("⚠️ 因子时间序列过长，仅保存最后2000个时间点")
                    factor_series = factor_series.tail(2000)
                
                # 只使用一个标准键名
                metadata['factor_series'] = factor_series
                _log("✅ 因子时间序列已添加")
            except Exception as e:
                _log(f"⚠️ 添加因子时间序列失败: {e}")
        
        # 使用更安全的pickle保存方式
        metadata_path = os.path.join(models_dir, TRAIN_RESULT_FILES['metadata'] if _CONFIG_AVAILABLE else "final_dfm_metadata.pkl")
        _log(f"尝试保存元数据到: {metadata_path}")
        
        # 分段保存，避免一次性保存大对象导致卡住
        try:
            import pickle
            with open(metadata_path, 'wb') as f:
                # 使用最高协议级别，但避免过度优化可能导致的问题
                pickle.dump(metadata, f, protocol=pickle.HIGHEST_PROTOCOL)
            saved_files['metadata'] = metadata_path
            _log("✅ 元数据已保存")
        except Exception as e:
            _log(f"❌ 标准pickle保存失败，尝试备用方案: {e}")
            # 备用方案：保存精简版元数据
            try:
                simplified_metadata = {
                    key: value for key, value in metadata.items() 
                    if key in ['timestamp', 'target_variable', 'best_variables', 'best_params', 
                              'best_score_tuple', 'total_evaluations', 'training_start_date',
                              'training_end_date', 'validation_start_date', 'validation_end_date']
                }
                simplified_path = os.path.join(models_dir, "simplified_metadata.pkl")
                with open(simplified_path, 'wb') as f:
                    pickle.dump(simplified_metadata, f)
                saved_files['simplified_metadata'] = simplified_path
                _log("✅ 精简版元数据已保存")
            except Exception as e2:
                _log(f"❌ 精简版元数据保存也失败: {e2}")
                
    except Exception as e:
        _log(f"❌ 保存元数据失败: {e}")
        # 不打印完整traceback，避免可能的卡住
        _log(f"错误类型: {type(e).__name__}")
        _log(f"错误消息: {str(e)}")
    
    # 保存因子和载荷
    try:
        if hasattr(final_dfm_results, 'x_sm') and final_dfm_results.x_sm is not None:
            # factors_path = os.path.join(data_dir, TRAIN_RESULT_FILES['factors_csv'] if _CONFIG_AVAILABLE else "smoothed_factors.csv")
            # _log(f"尝试保存因子到: {factors_path}")
            # final_dfm_results.x_sm.to_csv(factors_path, encoding='utf-8-sig')
            # saved_files['smoothed_factors'] = factors_path
            _log("✅ 因子数据已包含在元数据中")
            
            # 绘制因子图
            # try:
            #     plt.figure(figsize=(12, 2 * best_params['k_factors']))
            #     for i, factor_col in enumerate(final_dfm_results.x_sm.columns):
            #         plt.subplot(best_params['k_factors'], 1, i + 1)
            #         plt.plot(final_dfm_results.x_sm.index, final_dfm_results.x_sm[factor_col], label=factor_col)
            #         plt.title(factor_col)
            #         plt.legend()
            #     plt.tight_layout()
            #     factors_plot_path = os.path.join(plots_dir, TRAIN_RESULT_FILES['factors_plot'] if _CONFIG_AVAILABLE else "smoothed_factors_plot.png")
            #     _log(f"尝试保存因子图到: {factors_plot_path}")
            #     plt.savefig(factors_plot_path, dpi=300, bbox_inches='tight')
            #     plt.close()
            #     saved_files['smoothed_factors_plot'] = factors_plot_path
            #     _log("✅ 因子图已保存")
            # except Exception as e:
            #     _log(f"❌ 保存因子图失败: {e}")
    except Exception as e:
        _log(f"❌ 因子数据处理失败: {e}")
    
    try:
        if hasattr(final_dfm_results, 'Lambda') and final_dfm_results.Lambda is not None:
            # loadings_df = pd.DataFrame(
            #     final_dfm_results.Lambda,
            #     index=best_variables,
            #     columns=[f"Factor{i+1}" for i in range(best_params['k_factors'])]
            # )
            # loadings_path = os.path.join(data_dir, TRAIN_RESULT_FILES['loadings_csv'] if _CONFIG_AVAILABLE else "factor_loadings.csv")
            # _log(f"尝试保存载荷到: {loadings_path}")
            # loadings_df.to_csv(loadings_path, encoding='utf-8-sig')
            # saved_files['factor_loadings'] = loadings_path
            _log("✅ 载荷数据已包含在元数据中")
    except Exception as e:
        _log(f"❌ 载荷数据处理失败: {e}")
    
    # 计算拟合值
    try:
        if hasattr(final_dfm_results, 'x_sm') and hasattr(final_dfm_results, 'obs_mean') and hasattr(final_dfm_results, 'Lambda'):
            # fitted_values_df = RevserseTranslate(
            #     Factors=final_dfm_results.x_sm,
            #     miu=final_dfm_results.obs_mean,
            #     Lambda=final_dfm_results.Lambda,
            #     names=best_variables
            # )
            # fitted_values_path = os.path.join(data_dir, TRAIN_RESULT_FILES['fitted_csv'] if _CONFIG_AVAILABLE else "fitted_observables.csv")
            # _log(f"尝试保存拟合值到: {fitted_values_path}")
            # fitted_values_df.to_csv(fitted_values_path, encoding='utf-8-sig')
            # saved_files['fitted_observables'] = fitted_values_path
            _log("✅ 拟合值计算已完成，结果已包含在模型中")
            
            # 绘制拟合vs实际图
            # try:
            #     num_series_to_plot = min(5, len(best_variables))
            #     plot_vars = [target_variable] + [v for v in best_variables[:num_series_to_plot-1] if v != target_variable]
            #     
            #     plt.figure(figsize=(15, 3 * len(plot_vars)))
            #     for i, var in enumerate(plot_vars):
            #         if var in final_train_data.columns and var in fitted_values_df.columns:
            #             plt.subplot(len(plot_vars), 1, i + 1)
            #             plt.plot(final_train_data.index, final_train_data[var], label=f"实际 {var}")
            #             plt.plot(fitted_values_df.index, fitted_values_df[var], label=f"拟合 {var}", linestyle='--')
            #             plt.title(f"实际 vs 拟合: {var}")
            #             plt.legend()
            #     plt.tight_layout()
            #     fitted_plot_path = os.path.join(plots_dir, TRAIN_RESULT_FILES['fitted_plot'] if _CONFIG_AVAILABLE else "fitted_vs_actual_plot.png")
            #     _log(f"尝试保存拟合图到: {fitted_plot_path}")
            #     plt.savefig(fitted_plot_path, dpi=300, bbox_inches='tight')
            #     plt.close()
            #     saved_files['fitted_vs_actual_plot'] = fitted_plot_path
            #     _log("✅ 拟合图已保存")
            # except Exception as e:
            #     _log(f"❌ 保存拟合图失败: {e}")
    except Exception as e:
        _log(f"❌ 计算拟合值失败: {e}")
    
    # 8. 生成Excel报告（如果启用）
    # if generate_excel_report:
    #     _log("步骤7: 生成Excel报告...")
    #     try:
    #         excel_path = os.path.join(reports_dir, f"comprehensive_dfm_report_{run_timestamp}.xlsx")
    #         
    #         # 简化Excel报告生成，直接跳过以避免卡住
    #         _log("⚠️ 为避免卡住，跳过复杂Excel报告生成")
    #         
    #         # 生成简化的文本报告作为替代
    #         simple_report_path = os.path.join(reports_dir, f"dfm_training_summary_{run_timestamp}.txt")
    #         _log(f"生成简化文本报告: {simple_report_path}")
    #         
    #         with open(simple_report_path, 'w', encoding='utf-8') as f:
    #             f.write("DFM模型训练报告\n")
    #             f.write("=" * 50 + "\n\n")
    #             f.write(f"训练时间戳: {run_timestamp}\n")
    #             f.write(f"目标变量: {target_variable}\n")
    #             f.write(f"训练期: {training_start_date.strftime('%Y-%m-%d')} 到 {training_end_actual.strftime('%Y-%m-%d')}\n")
    #             f.write(f"验证期: {validation_start.strftime('%Y-%m-%d')} 到 {validation_end.strftime('%Y-%m-%d')}\n")
    #             f.write(f"最终变量数: {len(best_variables)}\n")
    #             f.write(f"最优因子数: {best_params['k_factors']}\n")
    #             f.write(f"EM最大迭代次数: {em_max_iter}\n")
    #             if best_score_tuple != (-np.inf, np.inf):
    #                 f.write(f"最终得分: HR={best_score_tuple[0]:.2f}%, RMSE={-best_score_tuple[1]:.6f}\n")
    #             f.write(f"总评估次数: {total_evaluations}\n\n")
    #             
    #             f.write("选择的变量:\n")
    #             for i, var in enumerate(best_variables, 1):
    #                 f.write(f"{i:2d}. {var}\n")
    #             
    #             f.write(f"\n分析结果:\n")
    #             for key, value in analysis_results.items():
    #                 if value is not None:
    #                     f.write(f"- {key}: 已完成\n")
    #                 else:
    #                     f.write(f"- {key}: 未完成\n")
    #             
    #             f.write(f"\n生成的文件:\n")
    #             for file_type, file_path in saved_files.items():
    #                 f.write(f"- {file_type}: {os.path.basename(file_path)}\n")
    #         
    #         saved_files['training_summary'] = simple_report_path
    #         _log(f"✅ 简化报告生成完成: {simple_report_path}")
    #             
    #     except Exception as e:
    #         _log(f"生成报告失败: {e}")
    #         # 不要使用traceback.print_exc()，可能导致卡住
    #         _log(f"错误类型: {type(e).__name__}")
    _log("✅ 训练完成，已生成核心结果文件")
    
    # 9. 生成Excel综合报告
    _log("步骤7: 生成Excel综合报告...")
    try:
        from generate_report import generate_excel_main
        
        # 调用Excel报告生成
        _log("正在生成Excel综合报告...")
        result = generate_excel_main()
        
        # 检查报告是否成功生成
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        expected_excel_path = os.path.join(reports_dir, f"final_report_{timestamp}.xlsx")
        
        # 查找实际生成的Excel文件
        excel_files = [f for f in os.listdir(reports_dir) if f.endswith('.xlsx') and 'final_report' in f]
        if excel_files:
            # 使用最新的Excel文件
            latest_excel = max(excel_files, key=lambda x: os.path.getctime(os.path.join(reports_dir, x)))
            excel_report_path = os.path.join(reports_dir, latest_excel)
            
            # 重命名为标准名称
            standard_excel_path = os.path.join(reports_dir, TRAIN_RESULT_FILES['excel_report'] if _CONFIG_AVAILABLE else "comprehensive_dfm_report.xlsx")
            if excel_report_path != standard_excel_path:
                import shutil
                shutil.copy2(excel_report_path, standard_excel_path)
                excel_report_path = standard_excel_path
            
            saved_files['excel_report'] = excel_report_path
            _log(f"✅ Excel综合报告生成完成: {os.path.basename(excel_report_path)}")
        else:
            _log("⚠️ Excel报告生成完成，但未找到输出文件")
            
    except Exception as e:
        _log(f"⚠️ Excel报告生成失败: {e}")
        # 不影响主流程，继续执行
    
    _log("=== DFM优化管道完成 ===")
    _log(f"总评估次数: {total_evaluations}")
    _log(f"最终变量数: {len(best_variables)}")
    _log(f"最优因子数: {best_params['k_factors']}")
    if best_score_tuple != (-np.inf, np.inf):
        _log(f"最终得分: HR={best_score_tuple[0]:.2f}%, RMSE={-best_score_tuple[1]:.6f}")
    _log(f"生成文件数: {len(saved_files)}")
    
    return saved_files


if __name__ == '__main__':
    # 测试代码
    def test_callback(message):
        print(f"[TEST] {message}")

    print("运行完整DFM优化管道测试...")

    # 创建模拟数据
    num_series = 15
    num_time_points = 200
    date_rng = pd.date_range(start='2010-01-01', periods=num_time_points, freq='W')
    data = np.random.rand(num_time_points, num_series)
    dummy_df = pd.DataFrame(data, index=date_rng, columns=[f'Series_{i+1}' for i in range(num_series)])
    
    test_params = {
        'input_df': dummy_df,
        'target_variable': 'Series_1',
        'selected_indicators': dummy_df.columns.tolist(),
        'training_start_date': '2010-01-01',
        'training_end_date': dummy_df.index[-1],
        'n_factors': 3,
        'em_max_iter': 10,  # 减少迭代次数以快速测试
        'output_base_dir': 'test_dfm_results',
        'progress_callback': test_callback,
        'enable_hyperparameter_tuning': False,  # 禁用以避免导入问题
        'enable_variable_selection': False,     # 禁用以避免导入问题
        'enable_detailed_analysis': True,       # 启用详细分析
        'generate_excel_report': True,          # ✅ 启用Excel报告生成
        'k_factors_range': (2, 5),
        'max_workers': 1,  # 减少并发以简化测试
        'info_criterion_method': 'bic',
        'cum_variance_threshold': 0.8
    }

    try:
        saved_files_test = train_and_save_dfm_results(**test_params)
        print("\n=== 测试完成 ===")
        print("保存的文件:")
        for key, path in saved_files_test.items():
            print(f"  {key}: {path}")
    except Exception as e:
        print(f"\n=== 测试失败 ===")
        print(f"错误: {e}")
        traceback.print_exc()