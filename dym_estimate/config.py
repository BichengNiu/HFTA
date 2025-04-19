  # -*- coding: utf-8 -*-
"""
DFM 模型调优的配置常量。
"""

import os

# --- 路径设置 ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__)) # config.py 所在的目录 (dym_estimate)
BASE_DIR = os.path.dirname(SCRIPT_DIR) # 项目根目录 (HFTA)

# 输入数据文件
EXCEL_DATA_FILE = os.path.join(BASE_DIR, 'data', '经济数据库.xlsx')

# 输出目录基础路径 (在 tune_dfm.py 中会基于此创建带时间戳的运行子目录)
BASE_OUTPUT_DIR = os.path.join(SCRIPT_DIR, 'dfm_result')
# 注意: 具体的运行输出目录 (run_{timestamp}) 和日志/结果文件名将在主脚本中生成

# --- 目标变量和频率 ---
TARGET_VARIABLE = '规模以上工业增加值:当月同比'
TARGET_SHEET_NAME = '工业增加值同比增速-月度' # 包含目标变量的 Sheet 名称
TARGET_FREQ = 'W-FRI' # 目标频率: 周五

# --- 数据预处理 ---
REMOVE_VARS_WITH_CONSECUTIVE_NANS = True # 是否移除超过阈值的连续缺失变量
CONSECUTIVE_NAN_THRESHOLD = 10          # 连续缺失的阈值

# --- 测试与迭代 ---
TEST_MODE = True # True: 快速测试模式; False: 完整运行
N_ITER_FIXED = 30 # 完整运行的 DFM 迭代次数
N_ITER_TEST = 2   # 测试模式的 DFM 迭代次数
# n_iter_to_use 将在主脚本中根据 TEST_MODE 确定

# --- 时间窗口 ---
TRAIN_END_DATE = '2024-06-28'        # 训练期结束日期 (最后一个周五)
VALIDATION_START_DATE = '2024-07-05' # 样本外验证期开始日期 (第一个周五)
VALIDATION_END_DATE = '2024-12-27'   # 样本外验证期结束日期

# --- 并行计算 ---
# MAX_WORKERS 将在主脚本中使用 os.cpu_count() 确定

# --- 超参数范围 (因子数) ---
# K_FACTORS_RANGE 将在主脚本中根据初始变量块动态确定
# HYPERPARAMS_TO_TUNE 将在主脚本中基于 K_FACTORS_RANGE 构建

# --- 其他绘图/分析参数 ---
# (如果需要，可以在这里添加，例如热力图显示的 top N 变量数等)
HEATMAP_TOP_N_VARS = 5 # 因子解释中显示 top N 载荷变量的数量

# --- NEW: Backward Selection Settings --- 
MIN_VARS = 10 # Minimum number of predictor variables to keep during backward selection
# --- 修改: 使用所有 CPU 核心 --- 
# MAX_WORKERS_BACKWARD = 4 # Max workers for backward selection (adjust based on CPU)
MAX_WORKERS_BACKWARD = os.cpu_count() if os.cpu_count() else 4 # 使用所有核心，如果无法获取则回退到 4

# Configuration constants for DFM tuning

# --- File Paths ---
# EXCEL_DATA_FILE = r'C:\Users\able7\Desktop\HFTA\data\经济数据库.xlsx' # Use raw string for windows path
# BASE_OUTPUT_DIR = r'C:\Users\able7\Desktop\HFTA\output'        # <<< 注释掉这行重复/错误的定义
# TYPE_MAPPING_SHEET = '指标体系' # <-- 新增: Sheet containing variable type/industry mapping

# --- Core Model Parameters ---
# ... existing params ...

# 可以在此处添加一些基本的配置验证或打印
# 例如，确认关键路径存在或打印运行模式
# print(f"Excel data file path: {EXCEL_DATA_FILE}") # DEBUG
# print(f"Base output directory: {BASE_OUTPUT_DIR}") # DEBUG

# --- 调试/确认打印 --- 
# 仅在模块被导入时打印一次，或者可以根据需要移动到使用配置的地方
# print(f"Config loaded: TARGET_VARIABLE='{TARGET_VARIABLE}', TEST_MODE={TEST_MODE}") # COMMENTED OUT 

# --- 旧的或重复的定义 (注释掉) ---
# EXCEL_DATA_FILE = r'C:\Users\able7\Desktop\HFTA\data\经济数据库.xlsx' # Use raw string for windows path
# BASE_OUTPUT_DIR = r'C:\Users\able7\Desktop\HFTA\output'        # <<< 注释掉这行重复/错误的定义
# TYPE_MAPPING_SHEET = '指标体系' # <-- 新增: Sheet containing variable type/industry mapping

# Configuration constants for DFM tuning

# --- File Paths ---
# EXCEL_DATA_FILE = r'C:\Users\able7\Desktop\HFTA\data\经济数据库.xlsx' # Use raw string for windows path
# BASE_OUTPUT_DIR = r'C:\Users\able7\Desktop\HFTA\output'        # <<< 注释掉这行重复/错误的定义
# TYPE_MAPPING_SHEET = '指标体系' # <-- 新增: Sheet containing variable type/industry mapping

# --- Core Model Parameters ---
# ... existing params ...

# 可以在此处添加一些基本的配置验证或打印
# 例如，确认关键路径存在或打印运行模式
# print(f"Excel data file path: {EXCEL_DATA_FILE}") # DEBUG
# print(f"Base output directory: {BASE_OUTPUT_DIR}") # DEBUG

# --- 调试/确认打印 --- 
# 仅在模块被导入时打印一次，或者可以根据需要移动到使用配置的地方
# print(f"Config loaded: TARGET_VARIABLE='{TARGET_VARIABLE}', TEST_MODE={TEST_MODE}") # COMMENTED OUT 