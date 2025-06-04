# -*- coding: utf-8 -*-
"""
DFM模块统一配置文件
定义所有输出路径和目录设置，以及所有默认参数值
"""

import os
from datetime import datetime
import shutil

# 项目根目录配置
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# === 数据相关默认配置 ===
class DataDefaults:
    """数据处理相关的默认配置"""
    # Excel工作表和列名
    TYPE_MAPPING_SHEET = '指标体系'
    TARGET_VARIABLE = '规模以上工业增加值:当月同比'
    INDICATOR_COLUMN = '高频指标'
    INDUSTRY_COLUMN = '行业'
    TYPE_COLUMN = '类型'
    
    # 数据处理参数
    ADF_P_THRESHOLD = 0.05  # ADF平稳性检验p值阈值
    CONSECUTIVE_NAN_THRESHOLD = None  # 连续缺失值阈值
    
    # 数据频率检测
    FREQ_DAILY = 'D'
    FREQ_WEEKLY = 'W'
    FREQ_MONTHLY = 'M'
    FREQ_QUARTERLY = 'Q'
    FREQ_INFER = 'infer'

# === 训练模型默认配置 ===
class TrainDefaults:
    """模型训练相关的默认配置"""
    # 基础DFM参数
    FACTOR_ORDER = 1
    IDIO_AR_ORDER = 1
    EM_MAX_ITER = 100
    
    # 因子选择策略
    FACTOR_SELECTION_STRATEGY = 'information_criteria'
    VARIABLE_SELECTION_METHOD = 'global_backward'
    INFO_CRITERION_METHOD = 'bic'
    
    # 因子选择参数
    IC_MAX_FACTORS = 10
    K_FACTORS_RANGE_MIN = 1
    K_FACTORS_RANGE_MAX = 8
    FIXED_NUMBER_OF_FACTORS = 3  # 固定因子数量策略使用的默认值
    CUM_VARIANCE_THRESHOLD = 0.8
    
    # 超参数调优参数
    ENABLE_HYPERPARAMETER_TUNING = True
    ENABLE_VARIABLE_SELECTION = True
    ENABLE_DETAILED_ANALYSIS = True
    GENERATE_EXCEL_REPORT = True
    MAX_WORKERS = 4
    VALIDATION_SPLIT_RATIO = 0.8  # 用于自动分割验证期，作为用户未指定日期时的后备机制
    
    # PCA分析参数
    PCA_N_COMPONENTS = 10
    
    # 默认日期设置 - 两套机制服务于不同场景
    TRAINING_YEARS_BACK = 5  # 动态计算训练开始日期：today - 5年（用于UI初始化和重置）
    # 固定验证期日期：确保模型训练的一致性和可重复性（用于生产环境）
    VALIDATION_END_YEAR = 2024
    VALIDATION_END_MONTH = 12
    VALIDATION_END_DAY = 31
    VALIDATION_START_YEAR = 2024
    VALIDATION_START_MONTH = 7
    VALIDATION_START_DAY = 1
    
    # 训练状态
    STATUS_WAITING = '等待开始'
    STATUS_PREPARING = '准备启动训练...'
    STATUS_TRAINING = '正在训练...'
    STATUS_COMPLETED = '训练完成'
    STATUS_FAILED_PREFIX = '训练失败'

# === UI界面默认配置 ===
class UIDefaults:
    """UI界面相关的默认配置"""
    # 变量选择方法选项
    VARIABLE_SELECTION_OPTIONS = {
        'none': "无筛选 (使用全部变量)",
        'global_backward': "全局后向剔除"
    }
    
    # 因子选择策略选项
    FACTOR_SELECTION_STRATEGY_OPTIONS = {
        'information_criteria': "信息准则 (Information Criteria)",
        'fixed_number': "固定因子数量 (Fixed Number of Factors)",
        'cumulative_variance': "累积共同方差 (Cumulative Common Variance)"
    }
    
    # 信息准则选项
    INFO_CRITERION_OPTIONS = {
        'bic': "BIC (Bayesian Information Criterion)",
        'aic': "AIC (Akaike Information Criterion)"
    }
    
    # UI组件默认值
    MAX_ITERATIONS_DEFAULT = 30
    MAX_ITERATIONS_MIN = 1
    MAX_ITERATIONS_STEP = 10
    
    FIXED_FACTORS_DEFAULT = 3
    FIXED_FACTORS_MIN = 1
    FIXED_FACTORS_STEP = 1
    
    IC_MAX_FACTORS_DEFAULT = 10
    IC_MAX_FACTORS_MIN = 1
    IC_MAX_FACTORS_STEP = 1
    
    CUM_VARIANCE_MIN = 0.1
    CUM_VARIANCE_MAX = 1.0
    CUM_VARIANCE_DEFAULT = 0.8
    CUM_VARIANCE_STEP = 0.05
    
    # 界面布局参数
    NUM_COLS_INDUSTRY = 3  # 行业选择列数
    NUM_COLS_DOWNLOAD = 3  # 下载按钮列数
    LOG_DISPLAY_LINES = 5  # 日志显示行数
    LOG_DISPLAY_HEIGHT = 120  # 日志显示高度

# === 可视化默认配置 ===
class VisualizationDefaults:
    """可视化相关的默认配置"""
    # 图表尺寸
    HEATMAP_HEIGHT_MIN = 600
    HEATMAP_HEIGHT_FACTOR = 35
    HEATMAP_HEIGHT_OFFSET = 200
    HEATMAP_WIDTH_MIN = 1000
    HEATMAP_WIDTH_FACTOR = 100
    HEATMAP_WIDTH_OFFSET = 50
    
    # 因子时间序列图布局
    FACTOR_PLOT_HEIGHT = 400
    FACTOR_PLOT_COLS_EVEN = 2  # 偶数个因子时每行列数
    FACTOR_PLOT_COLS_ODD = 3   # 奇数个因子时每行列数
    
    # 热力图聚类参数
    ENABLE_CLUSTERING = True
    MIN_VARS_FOR_CLUSTERING = 1
    
    # 图表标题和标签
    FACTOR_EVOLUTION_TITLE = "因子时间序列演变图"
    FACTOR_LOADINGS_TITLE = "因子载荷矩阵 (Lambda)"
    HEATMAP_TITLE_SUFFIX = " (变量聚类排序)"

# === 文件处理默认配置 ===
class FileDefaults:
    """文件处理相关的默认配置"""
    # MIME类型
    MIME_PICKLE = "application/octet-stream"
    MIME_JOBLIB = "application/octet-stream"
    MIME_EXCEL = "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    MIME_CSV = "text/csv"
    
    # 文件编码
    CSV_ENCODING = 'utf-8-sig'  # Excel兼容的UTF-8编码
    
    # 缓存设置
    CACHE_TTL_HOURS = 1  # 缓存时间1小时
    CACHE_TTL_SECONDS = 3600

# === 新闻分析默认配置 ===
class NewsAnalysisDefaults:
    """新闻分析相关的默认配置"""
    # 默认频率
    DEFAULT_MODEL_FREQUENCY = 'M'  # 月度
    
    # 图表文件名
    EVOLUTION_HTML_FILE = 'news_analysis_plot_backend_evo.html'
    DECOMPOSITION_HTML_FILE = 'news_analysis_plot_backend_decomp.html'
    EVOLUTION_DATA_FILE = 'nowcast_evolution_data_T.csv'
    DECOMPOSITION_DATA_FILE = 'news_decomposition_grouped.csv'
    BACKEND_PLOT_FILE = 'news_analysis_plot_backend.png'

# === 数值格式化默认配置 ===
class FormatDefaults:
    """数值格式化相关的默认配置"""
    # 精度设置
    PRECISION_DEFAULT = 2
    PRECISION_HIGH = 4
    PRECISION_PERCENTAGE = 2
    
    # 格式字符串
    NUMBER_FORMAT_DEFAULT = '0.0000'
    PERCENTAGE_FORMAT = '.2f'
    
    # 缺失值显示
    NA_REPRESENTATION = 'N/A'

# === 分析计算默认配置 ===
class AnalysisDefaults:
    """分析计算相关的默认配置"""
    # 超时设置
    TIMEOUT_SECONDS = 120  # R2计算等分析的超时时间
    
    # R2计算参数
    R2_MIN_VARIANCE = 0  # 最小方差阈值
    
    # 指标计算参数
    METRIC_PRECISION = 4
    METRIC_PCT_PRECISION = 2
    
    # 聚类参数
    LINKAGE_METHOD = 'ward'
    LINKAGE_METRIC = 'euclidean'

# === TrainModelConfig类定义（保持向后兼容） ===
class TrainModelConfig:
    """训练模型的统一配置类（向后兼容）"""
    
    # 项目根目录
    PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    
    # DFM输出目录
    DFM_TRAIN_OUTPUT_DIR = os.path.join(PROJECT_ROOT, "dashboard", "DFM", "outputs")
    
    # 训练结果文件名
    TRAIN_RESULT_FILES = {
        'model_joblib': 'final_dfm_model.joblib',
        'metadata': 'final_dfm_metadata.pkl',
        'excel_report': 'comprehensive_dfm_report.xlsx'
    }
    
    # Excel数据文件路径候选
    EXCEL_CANDIDATES = [
        os.path.join(PROJECT_ROOT, "data", "经济数据库0508.xlsx"),
        os.path.join(PROJECT_ROOT, "data", "wind数据", "经济数据库0508.xlsx"),
        os.path.join(PROJECT_ROOT, "dashboard", "经济数据库0508.xlsx")
    ]
    
    @classmethod
    def get_excel_path(cls):
        """获取可用的Excel文件路径"""
        for path in cls.EXCEL_CANDIDATES:
            if os.path.exists(path):
                return path
        return cls.EXCEL_CANDIDATES[0]  # 返回第一个作为默认值

# === 统一输出目录配置 ===
# 所有DFM相关输出都放在dashboard/DFM/outputs下
DFM_BASE_OUTPUT_DIR = os.path.join(PROJECT_ROOT, "dashboard", "DFM", "outputs")

# 各功能模块的子目录 - 直接保存到顶层，不使用train_results中间层
DFM_NEWS_OUTPUT_DIR = os.path.join(DFM_BASE_OUTPUT_DIR, "news_analysis")
DFM_EVOLUTION_OUTPUT_DIR = os.path.join(DFM_BASE_OUTPUT_DIR, "nowcast_evolution") 
DFM_REPORTS_OUTPUT_DIR = os.path.join(DFM_BASE_OUTPUT_DIR, "reports")
DFM_PLOTS_OUTPUT_DIR = os.path.join(DFM_BASE_OUTPUT_DIR, "plots")
DFM_DATA_OUTPUT_DIR = os.path.join(DFM_BASE_OUTPUT_DIR, "data")
DFM_MODELS_OUTPUT_DIR = os.path.join(DFM_BASE_OUTPUT_DIR, "models")

# === 数据准备输出配置 ===
DATA_PREP_OUTPUT_DIR = os.path.join(DFM_BASE_OUTPUT_DIR, "data_prep")

# === 文件命名规范 ===
def get_timestamped_filename(base_name: str, extension: str) -> str:
    """获取带时间戳的文件名"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{base_name}_{timestamp}.{extension}"

def get_timestamped_dir(base_name: str) -> str:
    """获取带时间戳的目录名"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{base_name}_{timestamp}"

# === 新闻分析文件名规范 ===
NEWS_RESULT_FILES = {
    'evolution_html': NewsAnalysisDefaults.EVOLUTION_HTML_FILE,
    'decomposition_html': NewsAnalysisDefaults.DECOMPOSITION_HTML_FILE,
    'evolution_data': NewsAnalysisDefaults.EVOLUTION_DATA_FILE,
    'decomposition_data': NewsAnalysisDefaults.DECOMPOSITION_DATA_FILE,
    'backend_plot': NewsAnalysisDefaults.BACKEND_PLOT_FILE
}

# === 确保输出目录存在 ===
def ensure_output_dirs():
    """确保所有输出目录存在"""
    dirs_to_create = [
        DFM_BASE_OUTPUT_DIR,
        DFM_NEWS_OUTPUT_DIR,
        DFM_EVOLUTION_OUTPUT_DIR,
        DFM_REPORTS_OUTPUT_DIR,
        DFM_PLOTS_OUTPUT_DIR,
        DFM_DATA_OUTPUT_DIR,
        DFM_MODELS_OUTPUT_DIR,
        DATA_PREP_OUTPUT_DIR
    ]
    
    for dir_path in dirs_to_create:
        os.makedirs(dir_path, exist_ok=True)

# === 清理和管理功能 ===
def get_output_summary():
    """获取输出目录汇总信息"""
    ensure_output_dirs()
    
    summary = {}
    dirs_to_check = [
        ('新闻分析', DFM_NEWS_OUTPUT_DIR),
        ('演化分析', DFM_EVOLUTION_OUTPUT_DIR),
        ('报告', DFM_REPORTS_OUTPUT_DIR),
        ('图表', DFM_PLOTS_OUTPUT_DIR),
        ('数据', DFM_DATA_OUTPUT_DIR),
        ('模型', DFM_MODELS_OUTPUT_DIR),
        ('数据准备', DATA_PREP_OUTPUT_DIR)
    ]
    
    for name, dir_path in dirs_to_check:
        if os.path.exists(dir_path):
            files = [f for f in os.listdir(dir_path) if os.path.isfile(os.path.join(dir_path, f))]
            subdirs = [d for d in os.listdir(dir_path) if os.path.isdir(os.path.join(dir_path, d))]
            summary[name] = {
                'path': dir_path,
                'files_count': len(files),
                'subdirs_count': len(subdirs),
                'files': files[:10],  # 只显示前10个文件
                'subdirs': subdirs[:10]  # 只显示前10个子目录
            }
        else:
            summary[name] = {
                'path': dir_path,
                'files_count': 0,
                'subdirs_count': 0,
                'files': [],
                'subdirs': []
            }
    
    return summary

# === 兼容性配置（向后兼容） ===
# 为了保持向后兼容，保留一些旧的配置名称
DEFAULT_OUTPUT_BASE_DIR = DFM_BASE_OUTPUT_DIR
NOWCAST_EVOLUTION_OUTPUT_DIR = DFM_EVOLUTION_OUTPUT_DIR

# === UI默认配置值（向后兼容，使用新的配置类） ===
UI_DEFAULT_TYPE_MAPPING_SHEET = DataDefaults.TYPE_MAPPING_SHEET
UI_DEFAULT_TARGET_VARIABLE = DataDefaults.TARGET_VARIABLE
UI_DEFAULT_INDICATOR_COLUMN = DataDefaults.INDICATOR_COLUMN
UI_DEFAULT_INDUSTRY_COLUMN = DataDefaults.INDUSTRY_COLUMN
UI_DEFAULT_TYPE_COLUMN = DataDefaults.TYPE_COLUMN

# 初始化时确保目录存在
ensure_output_dirs()

# 确保所有配置变量都被导出
__all__ = [
    # 配置类
    'DataDefaults',
    'TrainDefaults', 
    'UIDefaults',
    'VisualizationDefaults',
    'FileDefaults',
    'NewsAnalysisDefaults',
    'FormatDefaults',
    'AnalysisDefaults',
    
    # 向后兼容
    'TrainModelConfig',
    'DFM_TRAIN_OUTPUT_DIR',
    'DFM_DATA_OUTPUT_DIR', 
    'DFM_MODEL_OUTPUT_DIR',
    'DFM_PLOTS_OUTPUT_DIR',
    'DFM_REPORTS_OUTPUT_DIR',
    'DFM_NOWCAST_EVOLUTION_OUTPUT_DIR',
    'ensure_output_dirs',
    
    # UI默认值
    'UI_DEFAULT_TYPE_MAPPING_SHEET',
    'UI_DEFAULT_TARGET_VARIABLE', 
    'UI_DEFAULT_INDICATOR_COLUMN',
    'UI_DEFAULT_INDUSTRY_COLUMN',
    'UI_DEFAULT_TYPE_COLUMN'
]

# 添加默认配置以防万一
if 'DFM_TRAIN_OUTPUT_DIR' not in globals():
    import os
    current_dir = os.path.dirname(__file__)
    outputs_dir = os.path.join(current_dir, 'outputs')
    
    DFM_TRAIN_OUTPUT_DIR = outputs_dir
    DFM_DATA_OUTPUT_DIR = os.path.join(outputs_dir, 'data')
    DFM_MODEL_OUTPUT_DIR = os.path.join(outputs_dir, 'models')
    DFM_PLOTS_OUTPUT_DIR = os.path.join(outputs_dir, 'plots')
    DFM_REPORTS_OUTPUT_DIR = os.path.join(outputs_dir, 'reports')
    DFM_NOWCAST_EVOLUTION_OUTPUT_DIR = os.path.join(outputs_dir, 'nowcast_evolution')

if __name__ == "__main__":
    # 测试配置
    print("DFM模块配置信息:")
    print(f"基础输出目录: {DFM_BASE_OUTPUT_DIR}")
    print(f"新闻分析目录: {DFM_NEWS_OUTPUT_DIR}")
    print(f"演化分析目录: {DFM_EVOLUTION_OUTPUT_DIR}")
    
    print("\n默认配置测试:")
    print(f"默认目标变量: {DataDefaults.TARGET_VARIABLE}")
    print(f"默认因子选择策略: {TrainDefaults.FACTOR_SELECTION_STRATEGY}")
    print(f"默认最大迭代次数: {TrainDefaults.EM_MAX_ITER}")
    print(f"ADF检验阈值: {DataDefaults.ADF_P_THRESHOLD}")
    
    print("\n输出目录汇总:")
    summary = get_output_summary()
    for name, info in summary.items():
        print(f"{name}: {info['files_count']}个文件, {info['subdirs_count']}个子目录") 