# -*- coding: utf-8 -*-
"""
经济运行数据分析平台 - 主dashboard
"""

# 🔥 第一步：环境变量级别抑制（在所有导入之前）
import os
import sys

# 🔥 设置环境变量完全禁用Streamlit日志和警告
os.environ['STREAMLIT_LOGGER_LEVEL'] = 'CRITICAL'
os.environ['STREAMLIT_CLIENT_TOOLBAR_MODE'] = 'minimal' 
os.environ['STREAMLIT_BROWSER_GATHER_USAGE_STATS'] = 'false'
os.environ['STREAMLIT_CLIENT_SHOW_ERROR_DETAILS'] = 'false'

# 🔥 Python日志级别设置
os.environ['PYTHONWARNINGS'] = 'ignore'

# 立即抑制所有警告（在任何其他导入之前）
import warnings
import logging

# 设置Python根日志级别
logging.getLogger().setLevel(logging.CRITICAL)

# 立即抑制ScriptRunContext警告
warnings.filterwarnings("ignore")  # 抑制所有警告
warnings.filterwarnings("ignore", message=".*missing ScriptRunContext.*")
warnings.filterwarnings("ignore", message=".*ScriptRunContext.*") 
warnings.filterwarnings("ignore", category=UserWarning, module="streamlit")
warnings.filterwarnings("ignore", category=UserWarning, module="streamlit.runtime")

# 立即设置日志级别
for logger_name in ["", "root", "streamlit", "streamlit.runtime", "streamlit.runtime.scriptrunner_utils"]:
    logging.getLogger(logger_name).setLevel(logging.CRITICAL)
    logging.getLogger(logger_name).disabled = True

# 获取dashboard目录路径
current_dir = os.path.dirname(__file__)
project_root = os.path.abspath(os.path.join(current_dir, '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# 🔥 第二步：路径设置

# 🔥 第三步：增强警告抑制设置（在导入streamlit之前）
try:
    # 🔥 强化基础警告抑制
    warnings.filterwarnings("ignore", category=UserWarning, module="streamlit.*")
    warnings.filterwarnings("ignore", message=".*missing ScriptRunContext.*")
    warnings.filterwarnings("ignore", message=".*ScriptRunContext.*")
    warnings.filterwarnings("ignore", message=".*No runtime found.*")
    warnings.filterwarnings("ignore", message=".*Session state does not function.*")
    warnings.filterwarnings("ignore", message=".*to view a Streamlit app.*")
    
    # 🔥 设置所有相关日志级别为CRITICAL
    loggers_to_silence = [
        "streamlit",
        "streamlit.runtime", 
        "streamlit.runtime.scriptrunner_utils",
        "streamlit.runtime.scriptrunner_utils.script_run_context",
        "streamlit.runtime.caching",
        "streamlit.runtime.caching.cache_data_api",
        "streamlit.runtime.state",
        "streamlit.runtime.state.session_state_proxy"
    ]
    
    for logger_name in loggers_to_silence:
        logger = logging.getLogger(logger_name)
        logger.setLevel(logging.CRITICAL)
        logger.propagate = False
    
    # 🔥 尝试导入并执行自定义的警告抑制
    try:
        from suppress_streamlit_warnings import suppress_streamlit_warnings
        # suppress_streamlit_warnings() 已在模块导入时自动执行
    except ImportError:
        pass  # 基础抑制已经设置
        
except Exception as e:
    pass  # 静默处理警告抑制错误

# 🔥 第四步：导入Streamlit（警告已被抑制）
import streamlit as st

# 🔥 第五步：页面配置（必须是第一个Streamlit命令）
st.set_page_config(
    page_title="经济运行数据分析平台",
    layout="wide"
)

# 其他导入放在 set_page_config 之后
import pandas as pd
import numpy as np
# Remove unused plotting imports if they aren't needed elsewhere
# import plotly.graph_objects as go
# import plotly.express as px
from datetime import datetime, timedelta
import os
import warnings
# Remove adfuller if only used in stationarity tab
# from statsmodels.tsa.stattools import adfuller 
# Remove io if only used in stationarity tab
# import io 
import re # Keep for regex
import sys # <<< 新增
import subprocess # <<< 新增
import shutil # <<< 新增
import logging
import altair as alt # Added import for Altair
# from PIL import Image # <<< REMOVED IMPORT FOR PILLOW

# Attempt to enable VegaFusion for Altair to handle larger datasets
try:
    alt.data_transformers.enable("vegafusion")
    # print("[Dashboard Startup] Successfully enabled Altair VegaFusion data transformer.") # 移除以减少重复日志
except ImportError:
    # print("[Dashboard Startup] WARNING: VegaFusion not available or import failed. Altair might struggle with large datasets.") # 移除以减少重复日志
    # Optionally, fall back to a different transformer or do nothing
    # alt.data_transformers.enable('json') # Default, but might be slow for large data
    pass

# --- Add NDRC Logo to Sidebar ---
# st.sidebar.image("dashboard/image/国家发改委图标.png", use_column_width='always') # <<< COMMENTED OUT

# --- Display New Icon at the top of the Sidebar ---
# st.sidebar.image("dashboard/image/图标.png", use_column_width='always') # <<< COMMENTED OUT THIS ICON AS WELL

# --- Sidebar Title ---
st.sidebar.title("📈 经济运行数据分析平台") # <<< RE-ADDED THE CHART ICON

# --- Initialize Session State ---
# (Moved session state initialization higher, after sidebar title but before other UI elements)
if 'active_tab' not in st.session_state:
    st.session_state.active_tab = None

# --- <<< 新增：将项目根目录添加到 sys.path >>> ---
# 获取当前脚本 (dashboard.py) 所在的目录
current_dir = os.path.dirname(__file__)
# 获取项目根目录 (dashboard 目录的上级目录)
project_root = os.path.abspath(os.path.join(current_dir, '..'))
# 如果项目根目录不在 sys.path 中，则添加它
if project_root not in sys.path:
    sys.path.insert(0, project_root)
    print(f"[Dashboard Startup] Added project root to sys.path: {project_root}")

import sys # Ensure sys is imported if not already for the debug print
# print("DEBUG SYS.PATH in dashboard.py:", sys.path) # <<< DEBUG PRINT 移除以减少控制台噪音

# --- <<< 新增：也尝试添加 dashboard 目录本身到 sys.path >>> ---
# if current_dir not in sys.path:
#     sys.path.insert(0, current_dir)
#     print(f"[Dashboard Startup] Added dashboard directory to sys.path: {current_dir}")
# --- <<< 结束新增 >>> ---

# --- 配置导入处理 ---
try:
    # 尝试导入本地配置（如果存在）
    # import config as local_config
    # print(f"[Dashboard Startup] Successfully imported local config")
    pass  # 暂时不导入任何配置
except ImportError as e:
    # 如果没有配置文件，使用默认设置
    print(f"[Dashboard Startup] 使用默认配置设置")
    pass

# --- 移除：不再需要直接导入后端函数，由前端模块处理 ---
# 新闻分析相关导入已移至 news_analysis_front_end.py 模块中

# 从重构的脚本中导入核心函数
# --- Revert imports from shared_utils --- 
from preview.data_loader import load_and_process_data
# Keep monthly summary import for pre-calculation
from preview.growth_calculator import calculate_monthly_growth_summary
# --- End Revert imports ---
# Remove unused growth functions if not needed elsewhere
# from growth_calculator import calculate_weekly_summary, calculate_weekly_growth_summary
# Remove unused plotting utils if not needed elsewhere
# from dashboard.preview.plotting_utils import calculate_historical_weekly_stats, plot_weekly_indicator, plot_monthly_indicator
from preview.industrial_data_tab import display_industrial_tabs # <<< 新增导入

# 导入所有 Tab 模块
from preview.weekly_data_tab import display_weekly_tab
from preview.monthly_data_tab import display_monthly_tab
from DFM.model_analysis.dfm_ui import render_dfm_tab 
from DFM.data_prep.data_prep_ui import render_dfm_data_prep_tab # <<< 新增：从新路径导入
from DFM.train_model.train_model_ui import render_dfm_train_model_tab # <<< 新增模型训练模块导入
# 延迟导入新闻分析模块，避免重复导入打印
# from DFM.news_analysis.news_analysis_front_end import render_news_analysis_tab # <<< 改为延迟导入
from preview.diffusion_analysis import display_diffusion_tab
# --- 更新：工具类模块导入路径 ---
from tools.time_series_pretreat.time_series_clean.time_series_clean_tab import display_time_series_tool_tab # <<< 更正路径
from tools.time_series_pretreat.time_series_compute.time_series_compute_tab import display_time_series_compute_tab # <<< 更正路径

# --- 更新：平稳性分析模块导入路径 ---
from tools.time_series_property.stationarity_frontend import display_stationarity_tab # <<< 更正路径

# --- 更新：时间序列性质分析模块导入路径 ---
from tools.time_series_property.win_rate_frontend import display_win_rate_tab
from tools.time_series_property.dtw_frontend import display_dtw_tab
from tools.time_series_property.time_lag_corr_frontend import display_time_lag_corr_tab
# from tools.time_series_property.kl_divergence_frontend import display_kl_divergence_analysis # <<< 注释掉旧的K-L单独导入
from tools.time_series_property.combined_lead_lag_frontend import display_combined_lead_lag_analysis_tab # <<< 新增综合分析前端导入
# --- 结束更新 ---

# --- 新增：数据合并与导出功能导入 (路径待确认或函数已整合) ---
# from dashboard.tools.time_series_clean_mod.ui_components.merge_data_ui import display_merge_operations # <<< 此文件已删除，旧路径注释
# 假设 display_merge_operations 现在可能在 time_series_clean_tab.py 或其子模块中，或者有新名称
# 暂时不导入，后续根据用户提供信息或报错来添加
# from tools.time_series_pretreat.time_series_clean.ui_components.merge_data_ui import display_merge_operations # <<< 移除此行

# --- 新增：通用侧边栏组件导入 ---
from tools.time_series_pretreat.time_series_clean.ui_components.sidebar_ui import display_staged_data_sidebar
from tools.time_series_pretreat.time_series_clean.ui_components.data_comparison_ui import render_data_comparison_ui # <<< 新增数据比较UI导入
# --- 结束更新 ---

# --- Helper Function for Extracting Industry Name ---
def extract_industry_name(source_string: str) -> str:
    """
    从 '文件名|工作表名' 格式的字符串中提取核心行业名称。
    例如: '经济数据库0424_带数据源标志|化学化工_周度_Mysteel' -> '化学化工'
          '经济数据库0424_带数据源标志|工业增加值同比增速_月度_同花顺' -> '工业增加值同比增速'
          '经济数据库0424_带数据源标志|钢铁_日度_Wind' -> '钢铁'
    """
    try:
        # 分割文件名和工作表名
        parts = source_string.split('|')
        if len(parts) < 2:
            # 如果格式不符，尝试直接清理整个字符串
            sheet_name_part = source_string 
        else:
            sheet_name_part = parts[1] # 取工作表名部分

        # 使用正则表达式查找常见的行业名称模式 (中文 + 可选英文/数字)
        # 或者，更简单地，按 '_' 分割并取第一个非空的有意义部分
        
        # 方案：按 '_' 分割，取第一个包含中文字符的部分
        sub_parts = sheet_name_part.split('_')
        for part in sub_parts:
            if re.search(r'[\u4e00-\u9fff]+', part): # 检查是否包含中文字符
                # 移除常见的后缀 (如 '行业', '产业' 等，如果需要的话)
                # part = re.sub(r'(行业|产业)$', '', part) 
                return part.strip() # 返回第一个有中文的部分
        
        # 如果上面没找到，尝试返回分割后的第一部分（可能不含中文）
        if sub_parts:
             first_part = sub_parts[0].strip()
             if first_part: # 确保不是空字符串
                 return first_part

        # 如果完全无法解析，返回原始工作表名部分或整个字符串
        return sheet_name_part.strip() if sheet_name_part else source_string.strip()

    except Exception as e:
        print(f"Error extracting industry name from '{source_string}': {e}")
        return source_string # Fallback to original string on error

# --- Custom CSS for Color Theme and Cursor ---
st.markdown("""
<style>
/* Sidebar */
[data-testid="stSidebar"] {
    background-color: #2c3e50; /* Lighter Slate Blue */
    color: #ffffff;
}
/* Ensure sidebar text elements are light */
[data-testid="stSidebar"] h1, 
[data-testid="stSidebar"] h2, 
[data-testid="stSidebar"] h3, 
[data-testid="stSidebar"] h4, 
[data-testid="stSidebar"] p, 
[data-testid="stSidebar"] label,
[data-testid="stSidebar"] .stMarkdown,
[data-testid="stSidebar"] .stButton>button, /* Button text in sidebar */
[data-testid="stSidebar"] .stFileUploader label, /* Uploader label */
/* Target success/info message text in sidebar */
[data-testid="stSidebar"] [data-testid="stAlert"] div[role="alert"] {
    color: white !important; /* Force text color to white */
}
/* General sidebar text color fallback */
[data-testid="stSidebar"] .st-emotion-cache-16txtl3 { color: #f0f0f0 !important; } /* Headers */
[data-testid="stSidebar"] .st-emotion-cache-1zhivh4 { color: #d0d0d0 !important; } /* Regular text */
[data-testid="stSidebar"] .st-emotion-cache-15txusw {{ color: #d0d0d0 !important; }} /* Uploader text */

/* Main App Area */
body, .main .block-container {{ /* Target body and main container */
    color: #f0f0f0; 
    background-color: #36454F; 
}}
div[data-testid="stAppViewContainer"] > section {{ /* Target main section */
    background-color: #36454F; 
}}
/* Ensure main area text elements are light */
h1, h2, h3, h4, h5, h6, p, label, .stMarkdown, .stAlert>div, .stSelectbox label, .stExpander header {{
    color: #f0f0f0 !important; 
}}
/* Tables */
.stDataFrame {{ /* Add subtle background and border */
    background-color: #4a5a66; 
    border: 1px solid #5a6a76;
    border-radius: 5px;
}}
.stDataFrame thead th {{ /* Table headers */
    background-color: #5a6a76;
    color: #f0f0f0 !important; 
    border-bottom: 2px solid #6a7a86;
}}
.stDataFrame tbody td {{ /* Table cells */
    color: #f0f0f0 !important; 
    border-bottom: 1px solid #5a6a76;
}}
/* Tabs - Button Style */
div[role="tablist"] { 
    border-bottom: none !important; 
    gap: 8px; 
}
button[data-baseweb="tab"] { 
    font-size: 1.05em !important; 
    font-weight: bold !important; 
    color: #333333 !important; /* Darker text for light background */
    background-color: #cccccc !important; /* Light Grey background for inactive tabs */
    border: 1px solid #bbbbbb !important; /* Slightly darker border */
    border-radius: 6px !important; 
    padding: 8px 18px !important; 
    margin-bottom: 5px !important; 
    transition: background-color 0.2s ease, border-color 0.2s ease;
    border-bottom: none !important; 
}
button[data-baseweb="tab"]:hover {
     background-color: #bbbbbb !important; /* Slightly darker grey on hover */
     border-color: #aaaaaa !important;
     color: #000000 !important; /* Darker text on hover */
}

button[data-baseweb="tab"][aria-selected="true"] { /* Active tab - Keep previous style */
    background-color: #0a9396 !important; /* Teal active color */
    color: #ffffff !important; /* White text for active tab */
    border: 1px solid #0a9396 !important; 
    border-bottom: none !important; 
}

/* Buttons */
.stButton>button {{ /* General buttons */
    background-color: #005f73;
    color: #ffffff !important;
    border: none;
    padding: 8px 16px;
    border-radius: 4px;
    transition: background-color 0.2s ease;
}}
.stButton>button:hover {{
    background-color: #0a9396;
}}
/* Download Buttons - Adjust font size and padding */
.stDownloadButton>button {{ 
     background-color: #0a9396;
     color: #ffffff !important;
     border: none;
     padding: 6px 14px !important; /* Slightly reduced padding */
     border-radius: 4px;
     transition: background-color 0.2s ease;
     font-size: 1em !important; /* Match default text size */
     font-weight: normal !important; /* Ensure not bold unless desired */
}}
 .stDownloadButton>button:hover {{
    background-color: #94d2bd;
}}
/* Selectbox Label and Hover Cursor */
.stSelectbox label {{
    color: #f0f0f0 !important;
    font-weight: bold; 
}}
/* Apply pointer cursor to the main clickable part of the selectbox */
div[data-baseweb="select"] > div:first-child, 
div[data-testid="stSelectbox"] div[data-baseweb="select"] {{ 
    cursor: pointer !important; 
}}

/* --- <<< 新增：侧边栏暂存区按钮样式 >>> --- */
/* 针对侧边栏展开项内的下载按钮 */
[data-testid="stSidebar"] [data-testid="stExpander"] .stDownloadButton button {
    background-color: #28a745 !important; /* 绿色背景 */
    color: white !important;              /* 白色文字 */
    border: none !important;
}
[data-testid="stSidebar"] [data-testid="stExpander"] .stDownloadButton button:hover {
    background-color: #218838 !important; /* 深一点的绿色 */
}

/* 针对侧边栏展开项内的普通按钮 (假设删除按钮是普通按钮) */
/* 注意：可能需要根据实际渲染出的类名或属性调整选择器 */
[data-testid="stSidebar"] [data-testid="stExpander"] .stButton button {
    background-color: #dc3545 !important; /* 红色背景 */
    color: white !important;              /* 白色文字 */
    border: none !important;
}
[data-testid="stSidebar"] [data-testid="stExpander"] .stButton button:hover {
    background-color: #c82333 !important; /* 深一点的红色 */
}
/* --- <<< 结束新增 >>> --- */

.uploadedFile {{cursor: default !important;}}
</style>
""", unsafe_allow_html=True)

# --- Module Configuration (NEW) ---
MODULE_CONFIG = {
    "数据预览": {
        "工业": None,
        "消费": None,
        # 可以根据需要添加更多领域，例如农业、服务业等
    },
    "行业分析": {
        "占位": None 
    },
    "模型分析": {
        "DFM 模型": ["数据准备", "模型训练", "模型结果分析", "新闻分析"], # <<< 修改：增加"模型训练"
        "其他模型 (占位)": None 
    },
    "应用工具": {
        "数据预处理": ["数据清洗", "变量计算", "数据追加与合并", "数据比较"], # <--- 修改此处，增加"数据比较"
        "数据探索": ["平稳性分析", "相关性分析", "领先滞后分析"]
    }
}

# --- Sidebar ---
with st.sidebar:
    # --- Initialize Session State for Navigation and DFM files ---
    if 'selected_main_module' not in st.session_state:
        st.session_state.selected_main_module = list(MODULE_CONFIG.keys())[0]
    if 'selected_sub_module' not in st.session_state:
        st.session_state.selected_sub_module = None

    # For resetting data import UI elements across relevant modules
    st.session_state.setdefault('data_import_reset_counter', 0)

    # DFM file states (ensure they are initialized if not already)
    if 'dfm_model_file_indep' not in st.session_state: st.session_state.dfm_model_file_indep = None
    if 'dfm_metadata_file_indep' not in st.session_state: st.session_state.dfm_metadata_file_indep = None
    # if 'dfm_data_file_indep' not in st.session_state: st.session_state.dfm_data_file_indep = None # <--- 移除相关数据会话状态初始化

    # --- Session states for "数据预览" data ---
    if 'preview_data_loaded_files' not in st.session_state: st.session_state.preview_data_loaded_files = None
    if 'preview_weekly_df' not in st.session_state: st.session_state.preview_weekly_df = pd.DataFrame()
    if 'preview_monthly_df' not in st.session_state: st.session_state.preview_monthly_df = pd.DataFrame()
    if 'preview_source_map' not in st.session_state: st.session_state.preview_source_map = {}
    if 'preview_indicator_industry_map' not in st.session_state: st.session_state.preview_indicator_industry_map = {}
    # Add other preview-specific states if needed

    # --- Session states for "应用工具" (examples) ---
    if 'ts_tool_uploaded_file' not in st.session_state: st.session_state.ts_tool_uploaded_file = None # For 数据清洗
    if 'stationarity_uploaded_file_tool' not in st.session_state: st.session_state.stationarity_uploaded_file_tool = None # For 平稳性分析


    # --- Main Module Selection ---
    st.subheader("选择功能模块") 
    main_module_options = list(MODULE_CONFIG.keys())
    try:
        current_main_module_index = main_module_options.index(st.session_state.selected_main_module)
    except ValueError: # Should not happen if initialized
        current_main_module_index = 0
        st.session_state.selected_main_module = main_module_options[0]

    selected_main_module_val = st.radio(
        "主模块:",
        main_module_options,
        index=current_main_module_index,
        key='main_module_radio_selector', # Unique key
        label_visibility="collapsed"
    )
    
    if selected_main_module_val != st.session_state.selected_main_module:
        st.session_state.selected_main_module = selected_main_module_val
        st.session_state.selected_sub_module = None # Reset sub-module choice
        # ❌ 移除这个rerun，Streamlit会自动重新渲染
        # st.rerun() # MODIFIED: Was st.experimental_rerun()

    # --- Sub-Module Selection (conditional) ---
    sub_config = MODULE_CONFIG[st.session_state.selected_main_module]
    if isinstance(sub_config, dict): # This main module has sub-modules
        sub_module_options = list(sub_config.keys())
        
        # --- Store previous sub-module for change detection --- #
        previous_sub_module = st.session_state.get('selected_sub_module')

        if not st.session_state.selected_sub_module or st.session_state.selected_sub_module not in sub_module_options:
            st.session_state.selected_sub_module = sub_module_options[0]
        
        try:
            current_sub_module_index = sub_module_options.index(st.session_state.selected_sub_module)
        except ValueError: 
            current_sub_module_index = 0
            st.session_state.selected_sub_module = sub_module_options[0]
        
        expander_label = f"{st.session_state.selected_main_module} 子选项"
        with st.expander(expander_label, expanded=True):
            selected_sub_module_val = st.radio(
                f"选择 {st.session_state.selected_main_module} 子项:",
                sub_module_options,
                index=current_sub_module_index,
                key=f"sub_module_radio_{st.session_state.selected_main_module.replace(' ', '_')}", 
                label_visibility="collapsed"
            )
            if selected_sub_module_val != st.session_state.selected_sub_module:
                st.session_state.selected_sub_module = selected_sub_module_val
                # --- Add cleanup logic here when sub-module changes --- #
                if previous_sub_module == "数据探索" and selected_sub_module_val != "数据探索":
                    st.session_state.pop('stationarity_tab_preview_df', None)
                    # print(f"[DEBUG Dashboard Nav] Left '数据探索', cleared stationarity_tab_preview_df.") # 移除debug打印
                # Add similar cleanups for other sub-modules if needed
                # ❌ 减少不必要的rerun调用，让Streamlit自然重新渲染
                # st.rerun() 
    else: 
        # --- Also clear if switching FROM a main module that HAD sub-modules --- #
        if st.session_state.selected_sub_module is not None: 
             if st.session_state.selected_sub_module == "数据探索": # Clear if we were in data explore
                 st.session_state.pop('stationarity_tab_preview_df', None)
                 # print(f"[DEBUG Dashboard Nav] Switched main module away from '应用工具' (while sub was '数据探索'), cleared stationarity_tab_preview_df.") # 移除debug打印
             st.session_state.selected_sub_module = None
             # Consider if rerun needed here

# --- 新增：通用侧边栏组件渲染逻辑 >>> ---
if st.session_state.get('selected_main_module') == "应用工具":
    st.sidebar.markdown("---") # 分隔线
    # # 1. 当前数据预览
    # st.sidebar.subheader("当前数据预览")
    # processed_data = st.session_state.get('ts_tool_data_processed')
    # if processed_data is not None and not processed_data.empty:
    #     st.sidebar.dataframe(processed_data.head())
    #     st.sidebar.caption(f"当前显示: 预处理后数据 (形状: {processed_data.shape[0]}行, {processed_data.shape[1]}列)")
    # else:
    #     st.sidebar.caption("当前无已处理数据可预览。")
    # 
    # st.sidebar.markdown(" ") # 增加一些间距
    
    # 2. 暂存的数据集 (调用原有的函数)
    # 注意：display_staged_data_sidebar 需要 st 作为第一个参数，session_state 作为第二个
    # 但在这里我们直接使用 st.sidebar 来确保它在侧边栏中渲染，并传递 st.session_state
    # 如果 display_staged_data_sidebar 内部也用 st.sidebar，则无需包装
    # 假设 display_staged_data_sidebar(st_object, session_state_object) 直接在传入的 st_object 上渲染
    try:
        with st.sidebar: # 确保在其上下文内渲染
             display_staged_data_sidebar(st, st.session_state) # 直接调用，它应该使用传入的 st 对象在其自己的 UI 上下文（这里是sidebar）渲染
    except Exception as e_sidebar_render:
        st.sidebar.error(f"加载暂存区侧边栏时出错: {e_sidebar_render}")
        
# --- <<< 结束新增 >>> ---    

# --- Main Area (NEW STRUCTURE) ---

# --- 1. 数据预览 ---
if st.session_state.selected_main_module == "数据预览":
    # st.header("数据概览") # 将由子模块处理标题
    # st.subheader("上传用于概览的数据文件") # 将由子模块处理

    # 根据选择的子模块展示不同内容
    selected_sub = st.session_state.get('selected_sub_module')

    if selected_sub == "工业":
        # 此处稍后将调用 display_industrial_tabs()
        # st.write("这里将展示工业数据的总体情况、日度、周度、月度、年度数据 Tabs。") # <<< 移除/注释掉
        display_industrial_tabs(st.session_state, extract_industry_name) # <<< 修改调用：传递 extract_industry_name


    elif selected_sub == "消费":
        st.header("消费数据预览") # 示例标题
        st.write("这里将展示消费数据的相关内容。")
        # (消费模块的UI和逻辑)
    
    else: # 当 "数据预览" 被选中但没有特定子模块被选中时 (例如，刚切换到 "数据预览")
        st.info("请在左侧选择一个数据预览的子领域（如：工业、消费）。")


# --- 2. 行业分析 ---
elif st.session_state.selected_main_module == "行业分析":
    st.header("行业分析")
    selected_exploration_tool = st.session_state.selected_sub_module
    
    if selected_exploration_tool: # 确保 selected_exploration_tool 不是 None
        st.subheader(f"{selected_exploration_tool}")
    else:
        # 如果没有选择子模块（例如，刚切换到"行业分析"），可以显示一个通用信息
        st.info("请在左侧选择一个行业分析的子工具。") 
        # 或者根据 MODULE_CONFIG 自动选择第一个子工具并 rerun，但这取决于期望行为

    if selected_exploration_tool == "扩散分析":
        # 以下是原扩散分析逻辑，取消注释
        # 注意：display_diffusion_tab 需要 session_state 中有 'data_loaded', 'weekly_summary_cache' 等
        # 这些数据通常是在 "数据概览" (现在是 "数据预览") 模块中加载的。
        # 如果直接在这里使用，需要确保这些数据已通过某种方式加载，或者 display_diffusion_tab 内部有自己的数据上传机制。
        # 原始 dashboard.py 中，这一部分似乎没有独立的文件上传器，它依赖于之前模块加载的数据。
        # 我们需要确认 display_diffusion_tab 的确切数据依赖。
        
        # 检查 diffusion_analysis_tab.py，它内部有如下检查:
        # if not session_state.get('data_loaded', False):
        #     st.info("请先在左侧侧边栏上传用于周度/月度分析的数据文件。")
        #     return
        # 这意味着它期望数据已在别处加载。
        
        # 在当前 dashboard.py 的结构下，数据主要在 "数据预览" -> "工业" 模块加载并存入 st.session_state.preview_... 
        # display_diffusion_tab 使用的是如 weekly_df, weekly_summary_cache 等没有 preview_ 前缀的键。
        # 因此，直接调用 display_diffusion_tab(st.session_state) 可能找不到它需要的数据。
        
        # 解决方案1: 修改 display_diffusion_tab 以使用 preview_ 前缀的键 (如果数据源一致)
        # 解决方案2: 在这里提供一个数据上传器，并适配 display_diffusion_tab
        # 解决方案3: 明确指引用户先去"数据预览"加载数据 (不太友好)
        
        # 暂时我们先恢复调用，并观察其行为。如果报错或提示数据未加载，则需要进一步适配。
        try:
            # 在调用前，我们可以尝试将 preview_系列数据 映射到 display_diffusion_tab 期望的 session_state 键
            # 这是一种临时的桥接方法
            if st.session_state.get('preview_data_loaded_files') is not None:
                st.session_state['data_loaded'] = True # 标志数据已加载
                st.session_state['weekly_df'] = st.session_state.get('preview_weekly_df', pd.DataFrame())
                st.session_state['monthly_df'] = st.session_state.get('preview_monthly_df', pd.DataFrame())
                st.session_state['source_map'] = st.session_state.get('preview_source_map', {})
                st.session_state['indicator_industry_map'] = st.session_state.get('preview_indicator_industry_map', {})
                st.session_state['weekly_industries'] = st.session_state.get('preview_weekly_industries', [])
                st.session_state['monthly_industries'] = st.session_state.get('preview_monthly_industries', [])
                st.session_state['clean_industry_map'] = st.session_state.get('preview_clean_industry_map', {})
                st.session_state['weekly_summary_cache'] = st.session_state.get('preview_weekly_summary_cache', {})
                # monthly_summary_cache 在 diffusion_analysis_tab 中似乎没有直接使用，但以防万一
                st.session_state['monthly_summary_cache'] = st.session_state.get('preview_monthly_summary_cache', {})
            else:
                # 如果 preview 数据不存在，确保 'data_loaded' 为 False，让 display_diffusion_tab 内部的检查起作用
                st.session_state['data_loaded'] = False

            display_diffusion_tab(st, st.session_state) 
        except Exception as e_diff_tab:
            st.error(f"加载扩散分析模块时出错: {e_diff_tab}")
            import traceback
            st.error(traceback.format_exc())

    # elif selected_exploration_tool: # 这部分已在上面处理过，避免重复
    #     st.info(f"'{selected_exploration_tool}' 功能正在开发中或未完全配置。")

# --- 3. 模型分析 ---
elif st.session_state.selected_main_module == "模型分析":
    selected_model_type = st.session_state.selected_sub_module

    if selected_model_type == "DFM 模型":
        dfm_tab_names = MODULE_CONFIG["模型分析"]["DFM 模型"]
        
        if len(dfm_tab_names) == 4: # 检查是否为更新后的4个选项卡
            tab_data_prep, tab_model_train, tab_results, tab_news = st.tabs(dfm_tab_names)

            with tab_data_prep:
                render_dfm_data_prep_tab(st, st.session_state)

            with tab_model_train: # 新增：模型训练选项卡
                render_dfm_train_model_tab(st, st.session_state)

            with tab_results:
                render_dfm_tab(st, st.session_state)

            with tab_news:
                # 延迟导入新闻分析模块，避免重复导入打印
                from DFM.news_analysis.news_analysis_front_end import render_news_analysis_tab
                render_news_analysis_tab(st, st.session_state)

    elif selected_model_type == "其他模型 (占位)":
        st.info(f"{selected_model_type} 功能正在开发中。")
    else:
        st.info("请在左侧选择一个模型进行分析。")

# --- 4. 应用工具 ---
elif st.session_state.selected_main_module == "应用工具":
    selected_tool = st.session_state.selected_sub_module

    if selected_tool == "数据预处理":
        preprocess_tab_names = MODULE_CONFIG["应用工具"]["数据预处理"]
        if preprocess_tab_names and len(preprocess_tab_names) == 4: # 确保列表不为空且有四个元素
            data_clean_tab, var_compute_tab, append_merge_tab, data_compare_tab = st.tabs(preprocess_tab_names)

            with data_clean_tab:
                try:
                    display_time_series_tool_tab(st, st.session_state)
                except Exception as e:
                    st.error(f"加载 {preprocess_tab_names[0]} 模块时出错: {e}")
                    # st.exception(e)
            
            with var_compute_tab:
                try:
                    display_time_series_compute_tab(st, st.session_state)
                except Exception as e:
                    st.error(f"加载 {preprocess_tab_names[1]} 模块时出错: {e}")
                    st.info("变量计算模块尚未完全集成。")

            with append_merge_tab:
                try:
                    from tools.time_series_pretreat.time_series_clean.ui_components.append_merge_ui import show_append_merge_data_ui
                    show_append_merge_data_ui()
                except ImportError as e_import:
                    st.error(f"无法导入 {preprocess_tab_names[2]} 模块: {e_import}。请检查路径和文件名。")
                except Exception as e:
                    st.error(f"加载 {preprocess_tab_names[2]} 模块时出错: {e}")
                    # st.exception(e)                      

            with data_compare_tab:
                try:
                    render_data_comparison_ui() # <<< 调用数据比较UI渲染函数
                except Exception as e:
                    st.error(f"加载 {preprocess_tab_names[3]} ({render_data_comparison_ui.__module__}) 模块时出错: {e}")
                    # st.exception(e)                      

        else:
            st.error(f"数据预处理的子模块配置错误（应包含四个标签页名称）。请检查 MODULE_CONFIG 设置。当前名称: {preprocess_tab_names}")

    elif selected_tool == "数据探索": # <<< 新增数据探索逻辑
        explore_tab_names = MODULE_CONFIG["应用工具"]["数据探索"]
        if explore_tab_names and len(explore_tab_names) == 3:
            tab_stationarity, tab_correlation, tab_lead_lag = st.tabs(explore_tab_names)

            with tab_stationarity:
                st.markdown("##### **设置与数据选择**")
                st.write("从下方列表选择暂存区的数据集进行平稳性分析:")

                # --- <<< 修改：持久化和恢复数据集选择 >>> --- 
                staged_data_options = [""] + list(st.session_state.get('staged_data', {}).keys())
                
                default_stationarity_idx = 0
                previous_stationarity_selection = st.session_state.get('stationarity_active_dataset_name', None)
                if previous_stationarity_selection and previous_stationarity_selection in staged_data_options:
                    default_stationarity_idx = staged_data_options.index(previous_stationarity_selection)

                selected_staged_data_name_stationarity = st.selectbox(
                    "选择一个数据集:",
                    options=staged_data_options,
                    index=default_stationarity_idx, 
                    key="stationarity_selectbox_main" 
                )

                selected_staged_df_stationarity = None

                if selected_staged_data_name_stationarity:
                    selected_staged_df_stationarity = st.session_state.staged_data[selected_staged_data_name_stationarity]['df'].copy()
                    st.caption(f"已选择数据集: **{selected_staged_data_name_stationarity}** (形状: {selected_staged_df_stationarity.shape}) 进行平稳性检验。")
                    
                    st.session_state.stationarity_active_dataset_name = selected_staged_data_name_stationarity
                    st.session_state.stationarity_active_dataset_df = selected_staged_df_stationarity
                    # print(f"[Dashboard - Stationarity] Saved to session_state: active_dataset_name = {selected_staged_data_name_stationarity}") # 移除debug打印 
                else:
                    st.info("请选择数据集以进行平稳性检验。") 
                    if 'stationarity_active_dataset_name' in st.session_state:
                        del st.session_state.stationarity_active_dataset_name
                        # print("[Dashboard - Stationarity] Cleared stationarity_active_dataset_name from session_state") # 移除debug打印 
                    if 'stationarity_active_dataset_df' in st.session_state:
                        del st.session_state.stationarity_active_dataset_df
                        # print("[Dashboard - Stationarity] Cleared stationarity_active_dataset_df from session_state") # 移除debug打印 
                
                st.divider()
                current_selected_df_for_tab = st.session_state.get('stationarity_active_dataset_df', pd.DataFrame())
                current_selected_name_for_tab = st.session_state.get('stationarity_active_dataset_name', None)
                
                # print(f"[Dashboard - Stationarity] About to call display_stationarity_tab with: name='{current_selected_name_for_tab}', df_empty={current_selected_df_for_tab.empty}") # 移除debug打印 
                
                st.session_state.stationarity_selected_staged_data_df = current_selected_df_for_tab
                st.session_state.stationarity_selected_staged_data_name = current_selected_name_for_tab
                
                display_stationarity_tab(st, st.session_state)

            with tab_correlation:
                st.markdown("##### **设置与数据选择**")
                st.write("从下方列表选择暂存区的数据集进行相关性分析:")

                # --- 为相关性分析标签页添加独立的数据选择器 ---
                staged_data_options_corr = [""] + list(st.session_state.get('staged_data', {}).keys())
                
                # 持久化选择
                default_corr_idx = 0
                previous_corr_selection = st.session_state.get('correlation_own_dataset_name', None)
                if previous_corr_selection and previous_corr_selection in staged_data_options_corr:
                    try:
                        default_corr_idx = staged_data_options_corr.index(previous_corr_selection)
                    except ValueError:
                        default_corr_idx = 0 # 如果找不到则使用默认索引

                selected_staged_data_name_corr = st.selectbox(
                    "选择一个数据集:",
                    options=staged_data_options_corr,
                    index=default_corr_idx,
                    key="correlation_selectbox_main"
                )

                selected_staged_df_corr = None

                if selected_staged_data_name_corr:
                    selected_staged_df_corr = st.session_state.staged_data[selected_staged_data_name_corr]['df'].copy()
                    st.caption(f"已选择数据集: **{selected_staged_data_name_corr}** (形状: {selected_staged_df_corr.shape}) 进行相关性分析。")
                    
                    # 更新session_state
                    st.session_state.correlation_own_dataset_name = selected_staged_data_name_corr
                    st.session_state.correlation_own_dataset_df = selected_staged_df_corr
                    # print(f"[Dashboard - Correlation] 已保存到session_state: correlation_own_dataset_name = {selected_staged_data_name_corr}") # 移除debug打印 
                else:
                    st.info("请选择数据集以进行相关性分析。")
                    if 'correlation_own_dataset_name' in st.session_state:
                        del st.session_state.correlation_own_dataset_name
                        # print("[Dashboard - Correlation] 已清除correlation_own_dataset_name") # 移除debug打印
                    if 'correlation_own_dataset_df' in st.session_state:
                        del st.session_state.correlation_own_dataset_df
                        # print("[Dashboard - Correlation] 已清除correlation_own_dataset_df") # 移除debug打印
                
                st.divider()
                current_selected_df_for_tab = st.session_state.get('correlation_own_dataset_df', pd.DataFrame())
                current_selected_name_for_tab = st.session_state.get('correlation_own_dataset_name', None)
                
                # print(f"[Dashboard - Correlation] 即将调用相关性分析子模块: name='{current_selected_name_for_tab}', df_empty={current_selected_df_for_tab.empty}") # 移除debug打印
                
                # 向下兼容以前的状态键（供子模块使用）
                if not current_selected_df_for_tab.empty:
                    st.session_state['correlation_selected_df'] = current_selected_df_for_tab
                    st.session_state['correlation_selected_df_name'] = current_selected_name_for_tab
                    
                    # 显示相关性分析子模块
                    st.markdown("---")
                    try:
                        display_win_rate_tab(st, st.session_state) 
                    except Exception as e_win_tab:
                        st.error(f"加载胜率计算模块时出错: {e_win_tab}")
                        import traceback; st.error(traceback.format_exc())
                    
                    st.markdown("---")
                    try:
                        display_dtw_tab(st, st.session_state) 
                    except Exception as e_dtw_tab:
                        st.error(f"加载动态规整模块时出错: {e_dtw_tab}")
                        import traceback; st.error(traceback.format_exc())
                    st.markdown("---") # End divider
                else:
                    st.warning("请选择一个数据集以进行相关性分析。")
            
            with tab_lead_lag:
                # --- 新增：为领先滞后分析提供独立的数据选择器 ---
                staged_data_options_tlc = [""] + list(st.session_state.get('staged_data', {}).keys())
                
                # 持久化选择
                default_tlc_idx = 0
                previous_tlc_selection_name = st.session_state.get('tlc_own_selected_df_name', None)
                if previous_tlc_selection_name and previous_tlc_selection_name in staged_data_options_tlc:
                    try:
                        default_tlc_idx = staged_data_options_tlc.index(previous_tlc_selection_name)
                    except ValueError:
                        default_tlc_idx = 0 # Fallback if somehow name is not in options

                selected_staged_data_name_tlc = st.selectbox(
                    "从暂存区选择数据集:",
                    options=staged_data_options_tlc,
                    index=default_tlc_idx,
                    key="tlc_own_selectbox_main"
                )

                selected_staged_df_tlc = None

                if selected_staged_data_name_tlc:
                    selected_staged_df_tlc = st.session_state.staged_data[selected_staged_data_name_tlc]['df'].copy()
                    st.caption(f"已选择数据集: **{selected_staged_data_name_tlc}** (形状: {selected_staged_df_tlc.shape})")
                    
                    # 更新session_state以供 time_lag_corr_tab.py 使用
                    # 检查数据集是否真的改变了，避免不必要的 session_state 重写和可能的 rerun
                    if st.session_state.get('tlc_own_selected_df_name') != selected_staged_data_name_tlc:
                        st.session_state.tlc_own_selected_df = selected_staged_df_tlc
                        st.session_state.tlc_own_selected_df_name = selected_staged_data_name_tlc
                        # 当数据集改变时，相关的状态将在 time_lag_corr_tab.py 内部通过比较新的 df_name 和旧的 df_name 来重置
                        # print(f"[Dashboard LeadLag] TLC own dataset changed to {selected_staged_data_name_tlc}. TLC tab will handle state resets.") # 移除debug打印
                        # No explicit rerun here needed, as display_time_lag_corr_tab will be called with updated session state.
                        # If selectbox change itself causes a rerun (Streamlit default for some widgets if key changes or on_change is set), that's fine.
                else:
                    # 清理, 如果之前有选择但现在没有
                    if 'tlc_own_selected_df' in st.session_state:
                        del st.session_state.tlc_own_selected_df
                    if 'tlc_own_selected_df_name' in st.session_state:
                        del st.session_state.tlc_own_selected_df_name

                # --- 结束新增 ---

                # 获取当前为本标签页选择的数据
                current_selected_df_for_tlc = st.session_state.get('tlc_own_selected_df')
                current_selected_df_name_for_tlc = st.session_state.get('tlc_own_selected_df_name')

                if current_selected_df_for_tlc is not None and not current_selected_df_for_tlc.empty:
                    try:
                        # --- <<< 修改：调用新的综合分析函数 >>> ---
                        display_combined_lead_lag_analysis_tab(st, st.session_state) 
                        # --- <<< 结束修改 >>> ---
                        
                    except Exception as e_lead_lag_tab: # <<< 可以考虑将此 try-except 移入新的前端模块内部，或者保持通用性
                        st.error(f"加载综合领先滞后分析模块时出错: {e_lead_lag_tab}")
                        import traceback
                        st.error(traceback.format_exc())
                else:
                    st.warning("请在上方选择一个暂存数据集以进行综合领先滞后分析。") # <<< 更新警告信息
                    # 清理旧的依赖于 correlation_selected_df 的 tlc_df, tlc_df_name (如果它们意外存在)
                    if 'tlc_df' in st.session_state: # old key used previously
                        del st.session_state['tlc_df']
                    if 'tlc_df_name' in st.session_state: # old key used previously
                        del st.session_state['tlc_df_name']

        else:
            st.error("数据探索的子模块配置错误（应包含三个主标签页）。")

    else:
        st.info("请在左侧选择一个应用工具。")

else:
    st.error("主模块选择无效或未实现。")

# (End of script) 