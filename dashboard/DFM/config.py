# -*- coding: utf-8 -*-
"""
DFM模块统一配置文件
定义所有输出路径和目录设置，以及所有默认参数值
"""

import os
from datetime import datetime


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
    EM_MAX_ITER = 30
    
    # 🚀 变量筛选优化策略：先筛选变量，再选择因子数
    VARIABLE_SELECTION_FIXED_FACTORS = 10  # 变量筛选阶段使用的固定因子数
    
    # 🔥 修复：与老代码完全一致的因子选择策略
    FACTOR_SELECTION_METHOD = 'bai_ng'  # 与老代码一致
    FACTOR_SELECTION_STRATEGY = 'information_criteria'
    VARIABLE_SELECTION_METHOD = 'global_backward'
    INFO_CRITERION_METHOD = 'bic'
    
    # 🔥 修复：因子选择参数 - 与老代码完全一致
    IC_MAX_FACTORS = None  # 🔥 修复：无限制，与老代码k_max=len(eigenvalues)一致
    K_FACTORS_RANGE_MIN = 1
    K_FACTORS_RANGE_MAX = 30  # 🔥 修复：最终因子数评估范围：1-30
    FIXED_NUMBER_OF_FACTORS = 3  # 固定因子数量策略使用的默认值
    CUM_VARIANCE_THRESHOLD = 0.8
    
    # 超参数调优参数
    ENABLE_HYPERPARAMETER_TUNING = False  # 🔥 禁用超参数搜索，使用Bai-Ng方法
    USE_BAI_NG_FACTOR_SELECTION = True    # 🔥 启用Bai-Ng，与老代码保持一致
    ENABLE_VARIABLE_SELECTION = True
    ENABLE_DETAILED_ANALYSIS = True
    GENERATE_EXCEL_REPORT = True
    
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
    
    IC_MAX_FACTORS_DEFAULT = 20
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
    
    # 文件名配置
    REPORT_FILENAME = 'comprehensive_dfm_report.xlsx'
    MODEL_FILENAME = 'final_dfm_model.joblib'
    METADATA_FILENAME = 'final_dfm_metadata.pkl'
    
    # 数据文件名
    TRAINING_DATA_FILENAME = 'training_data.csv'
    EVOLUTION_DATA_FILENAME = 'nowcast_evolution_data_T.csv'
    DECOMPOSITION_DATA_FILENAME = 'news_decomposition_grouped.csv'
    
    # 图表文件名
    EVOLUTION_HTML_FILENAME = 'news_analysis_plot_backend_evo.html'
    DECOMPOSITION_HTML_FILENAME = 'news_analysis_plot_backend_decomp.html'
    BACKEND_PLOT_FILENAME = 'news_analysis_plot_backend.png'
    
    # 支持的文件扩展名
    EXCEL_EXTENSIONS = ['.xls', '.xlsx']
    IMAGE_EXTENSIONS = ['.png', '.jpg', '.jpeg']
    DATA_EXTENSIONS = ['.csv', '.txt']

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

# === 算法核心参数配置 ===
class AlgorithmDefaults:
    """算法核心计算相关的默认配置"""
    # 随机种子
    RANDOM_SEED = 42
    
    # EM算法初始化参数
    EM_FACTOR_INIT_SCALE = 0.1  # Lambda初始化缩放因子
    EM_AR_COEF_INIT = 0.95      # A矩阵对角线初始值
    EM_Q_INIT = 0.1             # Q矩阵初始值
    EM_R_INIT = 0.1             # R矩阵初始值
    EM_B_INIT = 0.1             # B矩阵初始值
    
    # 数值计算保护
    ZERO_PROTECTION = 1.0       # 除零保护值
    EIGENVALUE_STABILITY_THRESHOLD = 1.0  # 特征值稳定性阈值
    
    # Kalman滤波参数
    KALMAN_AR_FALLBACK = 0.95   # AR系数后备值

# === 性能和并发配置 ===
class PerformanceDefaults:
    """性能优化和并发处理相关的默认配置"""
    # 超时设置
    TIMEOUT_SHORT = 60          # 短期操作超时(秒)
    TIMEOUT_MEDIUM = 120        # 中期操作超时(秒) 
    TIMEOUT_LONG = 600          # 长期操作超时(秒)
    
    # 并发控制
    ANALYSIS_WORKERS = 1        # 分析计算工作进程数（单线程）
    
    # 批处理配置
    BATCH_SIZE_SMALL = 1        # 小批次处理
    BATCH_SIZE_MEDIUM = 2       # 中批次处理
    BATCH_SIZE_LARGE = 5        # 大批次处理

# === 可视化绘图配置 ===
class PlotDefaults:
    """图表绘制相关的默认配置"""
    # 图表尺寸配置
    FIGURE_SIZE_SMALL = (10, 6)     # 小图尺寸
    FIGURE_SIZE_MEDIUM = (14, 7)    # 中图尺寸
    FIGURE_SIZE_LARGE = (15, 10)    # 大图尺寸
    FIGURE_SIZE_COMPARISON = (15, 25)  # 对比图尺寸
    FIGURE_SIZE_HEATMAP = (12, 10)  # 热力图尺寸
    
    # 子图布局
    SUBPLOT_HEIGHT_PER_ROW = 4.5    # 每行子图高度
    SUBPLOT_WIDTH_PER_COL = 6       # 每列子图宽度
    
    # 行业因子图配置
    INDUSTRY_FACTOR_SUBPLOT_SIZE = (6, 4.5)  # 行业因子子图尺寸
    
    # 聚类热力图配置
    CLUSTERMAP_FIGSIZE = (12, 10)    # 聚类热力图尺寸
    
    # 载荷对比图配置  
    LOADING_COMPARISON_FIGSIZE = (15, 25)    # 载荷对比图尺寸
    LOADING_COMPARISON_THRESHOLD = 0.1       # 载荷对比阈值
    
    # 透明度配置
    ALPHA_MAIN_LINE = 0.8          # 主线条透明度
    ALPHA_BACKGROUND = 0.2         # 背景区域透明度
    ALPHA_GRID = 0.6               # 网格透明度
    ALPHA_GRID_LIGHT = 0.5         # 浅网格透明度
    ALPHA_SECONDARY = 0.7          # 辅助线条透明度
    
    # 线条配置
    LINEWIDTH_MAIN = 1.0           # 主线条宽度
    LINEWIDTH_SECONDARY = 0.8      # 辅助线条宽度
    
    # 布局配置
    TITLE_Y_POSITION = 1.02        # 标题Y位置
    TITLE_Y_POSITION_HIGH = 1.03   # 高标题Y位置
    LEGEND_Y_POSITION = -0.2       # 图例Y位置
    LEGEND_Y_POSITION_LOW = -0.4   # 低图例Y位置
    LEGEND_X_CENTER = 0.5          # 图例X中心位置
    
    # 颜色配置
    COLOR_MAIN = 'blue'            # 主要颜色
    COLOR_SECONDARY = 'grey'       # 次要颜色
    COLOR_VALIDATION = 'yellow'    # 验证期颜色
    COLOR_WHITE = 'white'          # 白色
    COLOR_BLACK = 'black'          # 黑色
    
    # 网格配置
    GRID_LINESTYLE = '--'          # 网格线样式
    GRID_LINESTYLE_LIGHT = ':'     # 浅网格线样式
    
    # 图表边距配置
    TIGHT_LAYOUT_RECT = [0, 0.03, 1, 0.98]  # 紧密布局矩形
    TIGHT_LAYOUT_RECT_TITLE = [0, 0, 1, 1.0]  # 带标题的紧密布局

# === 数据处理配置 ===
class ProcessingDefaults:
    """数据处理相关的默认配置"""
    # 数据分割比例
    TRAIN_SPLIT_RATIO = 0.8        # 训练验证分割比例
    
    # 缺失值处理
    HIGH_MISSING_THRESHOLD = 0.5   # 高缺失率阈值
    
    # 数值格式配置
    SCORE_DEFAULT_VALUE = (0.0, -1.0)  # 默认评分元组
    
    # PCA动态高度计算
    PCA_HEIGHT_SCALE = 0.3         # PCA图表高度缩放因子
    PCA_HEIGHT_MIN = 6             # PCA图表最小高度
    PCA_HEIGHT_MAX = 15            # PCA图表最大高度

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
    
    # 文件名配置
    REPORT_FILENAME = 'comprehensive_dfm_report.xlsx'
    MODEL_FILENAME = 'final_dfm_model.joblib'
    METADATA_FILENAME = 'final_dfm_metadata.pkl'
    
    # 数据文件名
    TRAINING_DATA_FILENAME = 'training_data.csv'
    EVOLUTION_DATA_FILENAME = 'nowcast_evolution_data_T.csv'
    DECOMPOSITION_DATA_FILENAME = 'news_decomposition_grouped.csv'
    
    # 图表文件名
    EVOLUTION_HTML_FILENAME = 'news_analysis_plot_backend_evo.html'
    DECOMPOSITION_HTML_FILENAME = 'news_analysis_plot_backend_decomp.html'
    BACKEND_PLOT_FILENAME = 'news_analysis_plot_backend.png'
    
    # 支持的文件扩展名
    EXCEL_EXTENSIONS = ['.xls', '.xlsx']
    IMAGE_EXTENSIONS = ['.png', '.jpg', '.jpeg']
    DATA_EXTENSIONS = ['.csv', '.txt']

# === TrainModelConfig类定义（保持向后兼容） ===
class TrainModelConfig:
    """训练模型的统一配置类（向后兼容）"""
    
    # 项目根目录
    PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

    # 🔥 移除：不再使用固定的输出目录，所有文件通过UI下载
    # DFM_TRAIN_OUTPUT_DIR = os.path.join(PROJECT_ROOT, "dashboard", "DFM", "outputs")
    
    # 训练结果文件名
    TRAIN_RESULT_FILES = {
        'model_joblib': FileDefaults.MODEL_FILENAME,
        'metadata': FileDefaults.METADATA_FILENAME,
        'excel_report': FileDefaults.REPORT_FILENAME
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

# === 移除统一输出目录配置 ===
# 不再使用固定的outputs目录，所有结果通过UI下载获得

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

# === 移除输出目录创建功能 ===
# 不再创建固定的输出目录

# === 移除清理和管理功能 ===
# 不再提供输出目录汇总功能

# 不再在初始化时创建目录

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
    'AlgorithmDefaults',
    'PerformanceDefaults',
    'PlotDefaults',
    'ProcessingDefaults',
    
    # 向后兼容
    'TrainModelConfig',
    
    # UI默认值
    'UI_DEFAULT_TYPE_MAPPING_SHEET',
    'UI_DEFAULT_TARGET_VARIABLE', 
    'UI_DEFAULT_INDICATOR_COLUMN',
    'UI_DEFAULT_INDUSTRY_COLUMN',
    'UI_DEFAULT_TYPE_COLUMN'
]

# 不再提供默认的outputs配置


if __name__ == "__main__":
    # 测试配置
    print("DFM模块配置信息:")
    print("不再使用固定的outputs目录")

    print("\n默认配置测试:")
    print(f"默认目标变量: {DataDefaults.TARGET_VARIABLE}")
    print(f"默认因子选择策略: {TrainDefaults.FACTOR_SELECTION_STRATEGY}")
    print(f"默认最大迭代次数: {TrainDefaults.EM_MAX_ITER}")
    print(f"ADF检验阈值: {DataDefaults.ADF_P_THRESHOLD}")