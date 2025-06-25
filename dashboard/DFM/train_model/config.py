# -*- coding: utf-8 -*-
"""
DFM 模型调优的配置常量。
"""

import os

# --- 路径设置 ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__)) # config.py 所在的目录 (train_model)
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(SCRIPT_DIR))) # 项目根目录

# 输入数据文件
# 检查几个可能的位置
data_file_name = '经济数据库0605.xlsx'
default_path = os.path.join(BASE_DIR, 'data', data_file_name)
alt_path_1 = os.path.join(BASE_DIR, data_file_name)
alt_path_2 = os.path.join(SCRIPT_DIR, 'data', data_file_name)
alt_path_3 = os.path.join(SCRIPT_DIR, data_file_name)

# 确保 EXCEL_DATA_FILE 总是被定义
EXCEL_DATA_FILE = None

if os.path.exists(default_path):
    EXCEL_DATA_FILE = default_path
elif os.path.exists(alt_path_1):
    EXCEL_DATA_FILE = alt_path_1
elif os.path.exists(alt_path_2):
    EXCEL_DATA_FILE = alt_path_2
elif os.path.exists(alt_path_3):
    EXCEL_DATA_FILE = alt_path_3
else:
    # 如果所有路径都不存在，保留默认路径但添加警告注释
    EXCEL_DATA_FILE = default_path
    print(f"警告: 数据文件 '{data_file_name}' 在所有预期位置都未找到，使用默认路径: {EXCEL_DATA_FILE}")
    # 在运行时将发出警告

# 输出目录基础路径 (在 tune_dfm.py 中会基于此创建带时间戳的运行子目录)
BASE_OUTPUT_DIR = os.path.join(SCRIPT_DIR, 'dfm_result')
# 注意: 具体的运行输出目录 (run_{timestamp}) 和日志/结果文件名将在主脚本中生成

# --- 目标变量和频率 ---
TARGET_VARIABLE = '规模以上工业增加值:当月同比'
TARGET_SHEET_NAME = '工业增加值同比增速_月度_同花顺'
TARGET_FREQ = 'W-FRI' # 目标频率: 周五

# --- 数据预处理 ---
REMOVE_VARS_WITH_CONSECUTIVE_NANS = True # 是否移除超过阈值的连续缺失变量
CONSECUTIVE_NAN_THRESHOLD = 10          # 连续缺失的阈值

# --- 新增：指标映射 Sheet ---
TYPE_MAPPING_SHEET = '指标体系' # 包含变量类型/行业映射的 Sheet 名称

# --- 测试与迭代 ---
TEST_MODE = False # True: 快速测试模式; False: 完整运行
N_ITER_FIXED = 30 # 完整运行的 DFM 迭代次数
N_ITER_TEST = 2   # 测试模式的 DFM 迭代次数
# n_iter_to_use 将在主脚本中根据 TEST_MODE 确定

# --- 新增: 因子选择策略 (两阶段流程) ---
FACTOR_SELECTION_METHOD = 'bai_ng' # 因子选择方法: 'cumulative', 'elbow', 'kaiser', 'cumulative_common', "bai_ng"
PCA_INERTIA_THRESHOLD = 0.9  # 累积(总)方差贡献率阈值 (当方法为 'cumulative' 时使用)
ELBOW_DROP_THRESHOLD = 0.1   # 手肘法边际下降率阈值 (当方法为 'elbow' 时使用)
COMMON_VARIANCE_CONTRIBUTION_THRESHOLD = 0.8 # 累积共同方差贡献阈值 (当方法为 'cumulative_common' 时使用)

# --- 变量选择 ---
DEBUG_VARIABLE_SELECTION_BLOCK = "库存" # 测试模式下仅剔除指定块 (None 则跳过变量剔除)

# --- 时间窗口 ---
# --- 新增: 数据整体时间范围控制 ---
DATA_START_DATE = '2020-01-01' # 'YYYY-MM-DD' 格式, None 表示使用数据最早日期
DATA_END_DATE = '2025-12-31'   # 🔥 修复：扩展到2025年，支持2025年数据
# --- 结束新增 ---

TRAINING_START_DATE = '2020-01-01'    # <-- 新增：手动指定训练期开始日期
TRAIN_END_DATE = '2024-06-28'         # 训练期结束日期 (最后一个周五)
VALIDATION_END_DATE = '2024-12-31'   # 🔥 修复：验证期结束日期应该是历史期间，不应该包含未来

# --- 其他绘图/分析参数 ---
# (如果需要，可以在这里添加，例如热力图显示的 top N 变量数等)
HEATMAP_TOP_N_VARS = 5 # 因子解释中显示 top N 载荷变量的数量

# --- 修改: 使用所有 CPU 核心 ---
# MAX_WORKERS_BACKWARD = 4 # Max workers for backward selection (adjust based on CPU)
MAX_WORKERS_BACKWARD = os.cpu_count() if os.cpu_count() else 8 # 使用所有核心，如果无法获取则回退到 4

# --- Nowcasting Evolution ---
# 最终 DFM 模型和元数据所在的目录 (相对于 dym_estimate)
NOWCAST_MODEL_INPUT_DIR = 'dfm_result' # 通常与 BASE_OUTPUT_DIR 相同
# 最终 DFM 模型文件名
NOWCAST_MODEL_FILENAME = "final_dfm_model.joblib"
# 最终 DFM 元数据文件名
NOWCAST_METADATA_FILENAME = "final_dfm_metadata.pkl"
# Nowcasting 演变结果输出目录 (相对于项目根目录)
NOWCAST_EVOLUTION_OUTPUT_DIR = 'nowcasting/nowcast_result'
# Nowcasting 演变分析的开始日期 (应为最终模型训练数据的最后日期或之后第一个 vintage 日期)
NOWCAST_EVOLUTION_START_DATE = '2024-12-27'
# Nowcasting 演变分析的目标/结束日期 (也是 nowcast 预测的目标日期)
# NOWCAST_EVOLUTION_TARGET_DATE = '2025-04-04' <-- 重命名
# Nowcasting 演变分析的时间范围结束日期
NOWCAST_EVOLUTION_END_DATE = '2025-04-25'
# Nowcasting 使用的频率 (应与模型训练一致)
NOWCAST_FREQ = 'W-FRI' # 覆盖 TARGET_FREQ 以防万一

# --- Plotting Configuration ---
PLOT_DEFAULT_OUTPUT_FILENAME = "news_decomposition_plot.png"
PLOT_FIGURE_SIZE = (16, 9) # Default figure size (width, height) in inches
PLOT_FONT_FAMILY = "SimHei" # Font family for Chinese characters (e.g., SimHei, Microsoft YaHei)
PLOT_DPI = 300            # Resolution for saved plots

# --- DFM Nowcasting Configuration ---
DEFAULT_MODEL_FREQUENCY_FOR_NEWS = 'W-FRI' # Default frequency assumption for news calculation (should match data)
NEWS_TARGET_MONTH = '2025-04' # Target month for month-internal analysis ('YYYY-MM' or None for auto)
NEWS_DECOMP_START_DATE = None  # Optional: Decomp analysis start date ('YYYY-MM-DD'), None for earliest data (IGNORED in month-internal mode)
NEWS_DECOMP_END_DATE_MODE = 'target_date' # How to end decomp: 'target_date' or 'latest_data' (IGNORED in month-internal mode)

# --- UI接口参数映射函数 ---
def map_ui_to_backend_params(ui_params: dict) -> dict:
    """
    将UI参数映射到后端配置参数

    Args:
        ui_params: UI参数字典

    Returns:
        后端配置参数字典
    """
    backend_params = {}

    # 因子选择策略映射
    strategy_mapping = {
        'information_criteria': 'bai_ng',
        'fixed_number': 'fixed',
        'cumulative_variance': 'cumulative'
    }

    if 'factor_selection_strategy' in ui_params:
        strategy = ui_params['factor_selection_strategy']
        backend_params['FACTOR_SELECTION_METHOD'] = strategy_mapping.get(strategy, strategy)

    # 变量选择方法映射
    if 'variable_selection_method' in ui_params:
        method = ui_params['variable_selection_method']
        backend_params['ENABLE_VARIABLE_SELECTION'] = (method != 'none')
        backend_params['VARIABLE_SELECTION_METHOD'] = method

    # 训练参数映射
    param_mappings = {
        'max_iterations': 'N_ITER_FIXED',
        'fixed_number_of_factors': 'FIXED_NUMBER_OF_FACTORS',
        'ic_max_factors': 'IC_MAX_FACTORS',
        'cum_variance_threshold': 'COMMON_VARIANCE_CONTRIBUTION_THRESHOLD',
        'info_criterion_method': 'INFO_CRITERION_METHOD'
    }

    for ui_key, backend_key in param_mappings.items():
        if ui_key in ui_params:
            backend_params[backend_key] = ui_params[ui_key]

    # 日期参数映射
    date_mappings = {
        'training_start_date': 'TRAINING_START_DATE',
        'validation_start_date': 'VALIDATION_START_DATE',
        'validation_end_date': 'VALIDATION_END_DATE'
    }

    for ui_key, backend_key in date_mappings.items():
        if ui_key in ui_params:
            date_value = ui_params[ui_key]
            if hasattr(date_value, 'strftime'):
                backend_params[backend_key] = date_value.strftime('%Y-%m-%d')
            elif isinstance(date_value, str):
                backend_params[backend_key] = date_value

    return backend_params

def validate_training_parameters(params: dict) -> tuple:
    """
    验证训练参数的有效性

    Args:
        params: 参数字典

    Returns:
        (是否有效, 错误信息列表)
    """
    errors = []

    # 必需参数检查
    required_params = ['target_variable', 'training_start_date', 'validation_end_date']
    for param in required_params:
        if param not in params or params[param] is None:
            errors.append(f"缺少必需参数: {param}")

    # 数值参数验证
    if 'max_iterations' in params:
        if not isinstance(params['max_iterations'], int) or params['max_iterations'] <= 0:
            errors.append("最大迭代次数必须是正整数")

    if 'fixed_number_of_factors' in params:
        if not isinstance(params['fixed_number_of_factors'], int) or params['fixed_number_of_factors'] <= 0:
            errors.append("固定因子数量必须是正整数")

    return len(errors) == 0, errors

