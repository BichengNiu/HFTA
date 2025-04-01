import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from DataProcessing import main as process_data
from stationarity import analyze_stationarity, make_stationary
import io
import base64
import os
import tempfile
import time
import gc
import numpy as np
import matplotlib.pyplot as plt
from io import BytesIO
# 导入扩散指数计算模块
from diffusion_index_calculator import calculate_diffusion_index
import traceback

# Page config
st.set_page_config(
    page_title="高频时间序列数据可视化分析",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 自定义CSS样式
st.markdown("""
<style>
    .main .block-container {
        padding-top: 2rem;
        background-color: #f7f7f8;
    }
    body {
        background-color: #f7f7f8;
    }
    .stTitle {
        font-size: 2rem !important;
        font-weight: bold !important;
        text-align: center !important;
        margin-bottom: 2rem !important;
        color: #1a162d;
    }
    .stSidebar {
        background-color: #1a162d;
        padding: 2rem 1rem;
    }
    .stSidebar .sidebar-content {
        padding: 0;
    }
    .stSidebar [data-testid="stMarkdown"] {
        color: white;
    }
    .stSidebar [data-testid="stHeader"] {
        color: white;
    }
    .stSidebar [data-testid="stVerticalBlock"] {
        background-color: #1a162d;
        color: white;
    }
    .stFileUploader {
        margin-top: 1rem;
    }
    .stFileUploader > div:first-child {
        display: block !important;
    }
    .stFileUploader > div:last-child {
        display: block !important;
    }
    .stMarkdown {
        margin-bottom: 1rem;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 2rem;
        background-color: #f7f7f8;
        border-bottom: 1px solid #dee2e6;
    }
    .stTabs [data-baseweb="tab"] {
        font-size: 1.1rem;
        font-weight: 500;
        color: #1a162d;
        padding: 10px 20px;
    }
    .stTabs [data-baseweb="tab-highlight"] {
        background-color: #4150f4;
    }
    .stTabs [data-baseweb="tab"][aria-selected="true"] {
        color: #4150f4;
    }
    .stPlotlyChart {
        background-color: white;
        border-radius: 5px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.12), 0 1px 2px rgba(0,0,0,0.24);
        padding: 15px;
        margin-bottom: 20px;
    }
    button[kind="primary"] {
        background-color: #4150f4;
        border-color: #4150f4;
    }
    button[kind="primary"]:hover {
        background-color: #3445e5;
        border-color: #3445e5;
    }
    .st-emotion-cache-1kyxreq {
        color: white;
    }
    .st-emotion-cache-16txtl3 h1, 
    .st-emotion-cache-16txtl3 h2, 
    .st-emotion-cache-16txtl3 h3, 
    .st-emotion-cache-16txtl3 h4 {
        color: white;
    }
    .stSelectbox label {
        font-size: 1.2rem;
        font-weight: bold;
    }
    .stExpander {
        font-size: 1.2rem;
        font-weight: bold;
        color: #000000;
    }
    .stApp {
        background-color: #f0f2f6;
    }
    .stButton>button {
        background-color: #1f1f1f;
        color: white;
        border: none;
        padding: 0.5rem 1rem;
        border-radius: 0.5rem;
        font-weight: bold;
    }
    .stButton>button:hover {
        background-color: #2f2f2f;
    }
    .stSelectbox label {
        color: white;
    }
    /* 下拉菜单样式 */
    .stSelectbox > div > div {
        cursor: pointer;
    }
    /* 表格样式 */
    .dataframe {
        font-size: 14px;
        border-collapse: collapse;
        width: 100%;
        margin: 1em 0;
    }
    .dataframe th {
        background-color: #f0f2f6;
        padding: 8px;
        text-align: left;
        border: 1px solid #ddd;
    }
    .dataframe td {
        padding: 8px;
        border: 1px solid #ddd;
    }
</style>
""", unsafe_allow_html=True)

# Title in sidebar
st.sidebar.markdown("<h1 style='text-align: center; font-size: 1.8rem; font-weight: bold; margin-bottom: 2rem; color: white;'>高频时间序列数据可视化分析</h1>", unsafe_allow_html=True)

# 侧边栏
with st.sidebar:
    uploaded_files = st.file_uploader("", type=['xlsx'], accept_multiple_files=True)
    
    # 显示上传状态
    if uploaded_files:
        status_placeholder = st.empty()

def process_data(file_path):
    """处理Excel文件数据"""
    try:
        # 显示进度信息
        st.sidebar.info("正在读取文件...")
        
        # 读取Excel文件
        df = pd.read_excel(file_path)
        st.sidebar.success(f"成功读取文件，包含 {len(df.columns)} 列")
        
        # 显示列名信息
        st.sidebar.info(f"列名: {', '.join(df.columns)}")
        
        # 确保日期列是datetime类型
        if '日期' in df.columns:
            df['日期'] = pd.to_datetime(df['日期'])
            df.set_index('日期', inplace=True)
        elif 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'])
            df.set_index('date', inplace=True)
        
        # 显示数据范围信息
        st.sidebar.info("数据加载信息:")
        st.sidebar.info(f"数据范围: {df.index.min()} 到 {df.index.max()}")
        st.sidebar.info(f"变量数量: {len(df.columns)}")
        
        return df
        
    except Exception as e:
        st.sidebar.error(f"处理数据时出错: {str(e)}")
        return None

def cleanup_temp_file(temp_file):
    """Clean up temporary file with retry mechanism"""
    if temp_file and os.path.exists(temp_file):
        max_retries = 5
        retry_delay = 2
        
        for attempt in range(max_retries):
            try:
                # 强制垃圾收集
                gc.collect()
                # 尝试关闭所有可能的打开文件句柄
                import pandas as pd
                pd.io.excel._base.ExcelFile.close(None)  # 尝试关闭所有Excel文件句柄
                time.sleep(1)  # 等待操作系统释放文件
                if os.path.exists(temp_file):
                    os.remove(temp_file)
                return True
            except Exception as e:
                if attempt < max_retries - 1:
                    time.sleep(retry_delay)
                    continue
                # 如果无法删除，记录警告但不显示
                return False

def process_single_file(file_content, file_name):
    """处理单个文件并返回处理后的数据框"""
    temp_path = None
    xls = None
    df_weekly = None
    df_monthly = None
    
    try:
        # 从文件名中提取行业信息，移除.xlsx后缀
        industry = file_name.replace('.xlsx', '')
        
        # 创建临时文件
        with tempfile.NamedTemporaryFile(delete=False, suffix='.xlsx') as temp_file:
            temp_file.write(file_content)
            temp_path = temp_file.name
        
        # 读取周度数据
        try:
            df_weekly = pd.read_excel(temp_path)
        except Exception as e:
            st.error(f"读取文件 {file_name} 的周度数据时出错: {str(e)}")
            
        # 读取月度数据（新增）
        try:
            df_monthly = pd.read_excel(temp_path, sheet_name="monthly")
            st.sidebar.success(f"成功读取 {file_name} 的月度数据")
        except Exception as e:
            st.sidebar.info(f"文件 {file_name} 没有monthly工作表或读取出错: {str(e)}")
            df_monthly = None
        
        # 检查周度数据有效性
        if df_weekly is None or df_weekly.empty:
            st.error(f"文件 {file_name} 的周度数据为空或格式不正确")
            if df_monthly is None or df_monthly.empty:
                return None, None, None
        
        # 处理周度数据
        if df_weekly is not None and not df_weekly.empty:
            # 确保日期列是datetime类型并设为索引
            if '日期' in df_weekly.columns:
                df_weekly['日期'] = pd.to_datetime(df_weekly['日期'])
                df_weekly.set_index('日期', inplace=True)
            elif 'date' in df_weekly.columns:
                df_weekly['date'] = pd.to_datetime(df_weekly['date'])
                df_weekly.set_index('date', inplace=True)
            else:
                # 尝试查找可能的日期列
                date_cols = [col for col in df_weekly.columns if any(kw in col.lower() for kw in ['日期', 'date', 'time', '时间'])]
                if date_cols:
                    df_weekly[date_cols[0]] = pd.to_datetime(df_weekly[date_cols[0]], errors='coerce')
                    df_weekly.set_index(date_cols[0], inplace=True)
                else:
                    st.error(f"文件 {file_name} 中未找到日期列")
                    if df_monthly is None or df_monthly.empty:
                        return None, None, None
            
            # 为weekly数据添加行业前缀
            if not df_weekly.empty:
                new_columns = {col: f"{industry}:{col}" for col in df_weekly.columns if col.lower() not in ['date', '日期']}
                df_weekly = df_weekly.rename(columns=new_columns)
        
        # 处理月度数据（新增）
        if df_monthly is not None and not df_monthly.empty:
            # 确保日期列是datetime类型并设为索引
            if '日期' in df_monthly.columns:
                df_monthly['日期'] = pd.to_datetime(df_monthly['日期'])
                df_monthly.set_index('日期', inplace=True)
            elif 'date' in df_monthly.columns:
                df_monthly['date'] = pd.to_datetime(df_monthly['date'])
                df_monthly.set_index('date', inplace=True)
            else:
                # 尝试查找可能的日期列
                date_cols = [col for col in df_monthly.columns if any(kw in col.lower() for kw in ['日期', 'date', 'time', '时间', '月份'])]
                if date_cols:
                    df_monthly[date_cols[0]] = pd.to_datetime(df_monthly[date_cols[0]], errors='coerce')
                    df_monthly.set_index(date_cols[0], inplace=True)
                else:
                    st.error(f"文件 {file_name} 月度工作表中未找到日期列")
                    df_monthly = None
            
            # 为monthly数据添加行业前缀
            if df_monthly is not None and not df_monthly.empty:
                # 筛选包含"规模以上工业增加值"和"同比"的列
                monthly_cols = [col for col in df_monthly.columns if "规模以上工业增加值" in col and "同比" in col]
                if not monthly_cols:
                    # 如果没有找到，则使用所有列
                    monthly_cols = df_monthly.columns
                
                monthly_new_columns = {col: f"{industry}:月度:{col}" for col in monthly_cols if col.lower() not in ['date', '日期', '月份']}
                # 只选择和重命名需要的列
                df_monthly = df_monthly[list(monthly_new_columns.keys())].rename(columns=monthly_new_columns)
        
        return df_weekly, df_monthly, industry
        
    except Exception as e:
        st.error(f"处理文件 {file_name} 时出错: {str(e)}")
        return None, None, None
    
    finally:
        # 清理临时文件
        if temp_path and os.path.exists(temp_path):
            try:
                os.remove(temp_path)
            except:
                pass

def get_industry_categories(industry_info):
    """获取行业大类"""
    return sorted(list(industry_info.keys()))

def get_sub_industries(df, category):
    """获取特定大类下的子行业"""
    sub_industries = set()
    for col in df.columns:
        if col.startswith(category + ':'):
            # 提取子行业（第二个冒号后的部分）
            parts = col.split(':')
            if len(parts) > 2:
                sub_industry = ':'.join(parts[1:]).strip()
                sub_industries.add(sub_industry)
    return sorted(list(sub_industries))

def create_time_series_plot(df, selected_var, title):
    """创建时间序列图"""
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=df.index,
        y=df[selected_var],
        mode='lines',
        name=selected_var,
        line=dict(width=2)
    ))

    fig.update_layout(
        title=title,
        xaxis_title='Date',
        yaxis_title=selected_var,
        legend=dict(x=1.05, y=1.0, xanchor='left', yanchor='top')
    )
    return fig

def process_weekly_data(df, indicator):
    """处理周度数据，按周统计数据"""
    # 确保索引是datetime类型
    df.index = pd.to_datetime(df.index)
    
    # 检查数据是否足够
    if len(df) < 2 or indicator not in df.columns:
        return pd.DataFrame()
    
    # 将0值和异常值转换为NaN（使用向量化操作）
    indicator_data = df[indicator].replace([0, float('inf'), float('-inf')], float('nan'))
    
    # 创建周数和年份列（使用向量化操作）
    weeks = df.index.isocalendar().week
    years = df.index.year
    
    # 获取当前年份和去年
    current_year = df.index.max().year
    last_year = current_year - 1
    
    # 获取历史年份列表（过去5年）- 确保不会超出数据范围
    min_year = df.index.min().year
    available_years = sorted(df.index.year.unique())
    historical_years = [y for y in range(current_year - 5, current_year) if y >= min_year]
    
    # 检查是否有足够的历史数据
    if len(historical_years) == 0:
        # 如果没有足够的历史数据，使用所有可用的年份作为历史数据
        historical_years = [y for y in available_years if y < current_year]
    
    # 创建结果数据框，索引为周数1-52
    result = pd.DataFrame(index=range(1, 53))
    
    # 为了提高性能，预先计算所有需要的过滤条件
    week_masks = {}
    for week in range(1, 53):
        week_masks[week] = weeks == week
    
    hist_mask = years.isin(historical_years)
    last_year_mask = years == last_year
    current_year_mask = years == current_year
    
    # 对每周计算统计值（使用向量化操作）
    for week in range(1, 53):
        # 获取该周的所有数据点
        week_mask = week_masks[week]
        
        # 过去5年的数据
        hist_data = indicator_data[week_mask & hist_mask]
        
        # 去年的数据（最近一年）
        last_year_data = indicator_data[week_mask & last_year_mask]
        
        # 今年的数据（当年）
        current_year_data = indicator_data[week_mask & current_year_mask]
        
        # 计算统计值并添加到结果中
        if not hist_data.empty:
            result.loc[week, 'min'] = hist_data.min()
            result.loc[week, 'max'] = hist_data.max()
            result.loc[week, 'mean'] = hist_data.mean()
        
        if not last_year_data.empty:
            result.loc[week, 'last_year'] = last_year_data.iloc[0]
            
        if not current_year_data.empty:
            result.loc[week, 'current_year'] = current_year_data.iloc[0]
    
    return result

def process_uploaded_files():
    if uploaded_files:
        status_placeholder.info("正在处理上传的文件...")
        
        # 创建进度条
        progress_bar = st.progress(0)
        progress_text = st.empty()
        
        # Process uploaded files
        weekly_dfs = []  # 存储所有处理后的周度数据框
        monthly_dfs = []  # 存储所有处理后的月度数据框（新增）
        industries = {}  # 存储所有行业信息
        
        # 处理每个文件
        for i, file in enumerate(uploaded_files):
            progress_percent = int((i / len(uploaded_files)) * 100)
            progress_bar.progress(progress_percent)
            progress_text.text(f"处理文件 {i+1}/{len(uploaded_files)}: {file.name}")
            
            try:
                # 读取文件内容
                file_bytes = file.getvalue()
                if len(file_bytes) == 0:
                    continue
                
                # 直接处理文件
                result = process_single_file(file_bytes, file.name)
                if result is not None:
                    df_weekly, df_monthly, industry = result
                    if df_weekly is not None and not df_weekly.empty:
                        weekly_dfs.append(df_weekly)
                    if df_monthly is not None and not df_monthly.empty:
                        monthly_dfs.append(df_monthly)
                    industries[industry] = {"weekly": df_weekly, "monthly": df_monthly}
            except Exception as e:
                st.error(f"处理文件 {file.name} 时出错: {str(e)}")
                continue
        
        # 完成处理
        progress_bar.progress(100)
        progress_text.empty()  # 清除进度文本
        
        # 检查是否有数据
        if not weekly_dfs and not monthly_dfs:
            status_placeholder.error("未能从上传的文件中提取有效数据。请检查文件格式是否正确。")
            return None, None, None
        else:
            # 移除成功处理的提示信息
            status_placeholder.empty()
            
            # 合并所有数据框
            df_weekly_all = pd.concat(weekly_dfs, axis=1) if weekly_dfs else pd.DataFrame()
            df_monthly_all = pd.concat(monthly_dfs, axis=1) if monthly_dfs else pd.DataFrame()
            
            return df_weekly_all, df_monthly_all, industries
    return None, None, None

try:
    # 初始状态信息
    status_placeholder = st.empty()
    
    # 文件处理区域
    if uploaded_files:
        # 处理上传的文件
        df_weekly, df_monthly, industries = process_uploaded_files()
        
        # 获取行业大类
        categories = get_industry_categories(industries) if industries else []
        
        # 创建数据可视化区域
        with st.container():
            if categories:  # 添加检查
                # 添加分析类型选择按钮（周度数据、月度数据和扩散指数）
                tab1, tab2, tab3 = st.tabs(["周度数据", "月度数据", "扩散指数"])
                
                # 在周度数据标签页中显示周度数据
                with tab1:
                    # 移动行业选择器到这里
                    st.markdown("""
                        <style>
                        .stSelectbox label {
                            font-size: 1.2rem;
                            font-weight: bold;
                        }
                        </style>
                    """, unsafe_allow_html=True)
                    st.markdown("### 选择行业大类")
                    selected_category = st.selectbox("", categories, key='industry_selector')
                    
                    if selected_category:
                        # 获取该大类的所有变量
                        category_cols = []
                        for col in df_weekly.columns:
                            if col.startswith(selected_category + ':'):
                                category_cols.append(col)
                        
                        # 获取周度指标（更改检测逻辑，使其更加灵活）
                        # 如果有包含"周"字样的指标，优先使用
                        week_containing_indicators = [col for col in category_cols if "周" in col]
                        
                        # 没有包含"周"字样的指标时，使用所有可用指标（除了明确标记为月度的）
                        monthly_keywords = ["月", "月度", "monthly"]
                        if not week_containing_indicators:
                            weekly_indicators = [col for col in category_cols if not any(keyword in col for keyword in monthly_keywords)]
                        else:
                            weekly_indicators = week_containing_indicators
                        
                        if not weekly_indicators:
                            st.warning(f"未找到行业 '{selected_category}' 的可用指标数据。请检查上传的数据文件。")
                            # 显示一些数据，帮助识别格式问题
                            if category_cols:
                                st.info(f"该行业可用指标: {', '.join(category_cols[:5] if len(category_cols) > 5 else category_cols)}")
                                
                                # 尝试从所有指标中获取替代指标
                                alternative_indicators = category_cols[:10] if len(category_cols) > 10 else category_cols
                                
                                st.subheader("数据预览")
                                st.dataframe(df_weekly[category_cols].head())
                                
                                # 显示指标数据表格，即使没有周度指标
                                st.subheader("行业指标数据")
                                
                                # 计算每个指标的最新值和上一年同期值
                                indicator_performance = []
                                current_year = df_weekly.index.max().year
                                last_year = current_year - 1
                                
                                for indicator in alternative_indicators:
                                    try:
                                        # 按年份筛选数据
                                        current_year_data = df_weekly[df_weekly.index.year == current_year][indicator].dropna()
                                        last_year_data = df_weekly[df_weekly.index.year == last_year][indicator].dropna()
                                        
                                        current_value = current_year_data.iloc[-1] if not current_year_data.empty else None
                                        last_year_value = last_year_data.iloc[-1] if not last_year_data.empty else None
                                        
                                        # 计算同比变化
                                        yoy_change = None
                                        if current_value is not None and last_year_value is not None:
                                            yoy_change = current_value - last_year_value
                                        
                                        # 计算同比变化百分比
                                        yoy_change_pct = None
                                        if current_value is not None and last_year_value is not None and last_year_value != 0:
                                            yoy_change_pct = (current_value - last_year_value) / last_year_value * 100
                                        
                                        # 提取指标名称
                                        indicator_name = ":".join(indicator.split(":")[1:]).strip()
                                        if ".xlsx" in indicator_name:
                                            indicator_name = indicator_name.replace(".xlsx", "")
                                        
                                        # 获取最新数据日期
                                        last_date = current_year_data.index[-1].strftime('%Y-%m-%d') if not current_year_data.empty else ""
                                        
                                        indicator_performance.append({
                                            'name': indicator_name,
                                            'current_value': current_value,
                                            'last_year_value': last_year_value,
                                            'yoy_change': yoy_change,
                                            'yoy_change_pct': yoy_change_pct,
                                            'last_date': last_date
                                        })
                                    except Exception as e:
                                        st.error(f"处理指标 {indicator} 时出错: {str(e)}")
                                        continue
                                
                                if indicator_performance:
                                    perf_df = pd.DataFrame(indicator_performance)
                                    
                                    # 重新排列列顺序
                                    perf_df = perf_df[['name', 'last_date', 'current_value', 'last_year_value', 'yoy_change', 'yoy_change_pct']]
                                    perf_df.columns = ['指标名称', '最新数据日期', '当前值', '去年同期', '同比变化', '同比变化(%)']
                                    
                                    # 按同比变化百分比排序
                                    perf_df = perf_df.sort_values('同比变化(%)', ascending=False)
                                    
                                    # 格式化数据，处理空值
                                    formatted_df = perf_df.copy()
                                    for col in ['当前值', '去年同期', '同比变化', '同比变化(%)']:
                                        try:
                                            formatted_df[col] = formatted_df[col].apply(lambda x: f"{float(x):.2f}" if pd.notna(x) else "")
                                        except Exception as e:
                                            formatted_df[col] = formatted_df[col].astype(str)
                                    
                                    # 单独处理同比变化百分比列，添加百分号
                                    try:
                                        formatted_df['同比变化(%)'] = formatted_df['同比变化(%)'].apply(
                                            lambda x: f"{float(x):.2f}%" if pd.notna(x) and x != "" else "")
                                    except Exception as e:
                                        pass
                                    
                                    # 自定义颜色映射
                                    def color_negative_red(val):
                                        if pd.isna(val) or val == "":
                                            return ''
                                        try:
                                            # 移除百分号再转换
                                            if isinstance(val, str) and '%' in val:
                                                val = float(val.replace('%', ''))
                                            else:
                                                val = float(val)
                                                
                                            if val > 0:
                                                return 'background-color: #ffcccc; color: black'
                                            else:
                                                return 'background-color: #ccffcc; color: black'
                                        except (ValueError, TypeError):
                                            return ''
                                    
                                    # 应用样式到同比变化和同比变化(%)列
                                    styled_df = formatted_df.style.applymap(color_negative_red, subset=['同比变化', '同比变化(%)'])
                                    st.dataframe(styled_df)
                        else:
                            # 计算每个指标最新一周的同比情况并排序
                            indicator_performance = []
                            
                            # 使用并行处理来加速数据处理
                            for indicator in weekly_indicators:
                                try:
                                    # 处理数据
                                    weekly_data = process_weekly_data(df_weekly, indicator)
                                    
                                    # 找到最新的非空周数据
                                    last_valid_week = None
                                    current_year_value = None
                                    last_year_value = None
                                    
                                    # 从最后一周开始往前查找，找到第一个有current_year数据的周
                                    for week in range(52, 0, -1):
                                        if week in weekly_data.index and 'current_year' in weekly_data.columns and pd.notna(weekly_data.loc[week, 'current_year']):
                                            last_valid_week = week
                                            current_year_value = weekly_data.loc[week, 'current_year']
                                            if 'last_year' in weekly_data.columns and pd.notna(weekly_data.loc[week, 'last_year']):
                                                last_year_value = weekly_data.loc[week, 'last_year']
                                            break
                                    
                                    # 计算同比变化
                                    yoy_change = None
                                    if current_year_value is not None and last_year_value is not None:
                                        yoy_change = current_year_value - last_year_value
                                    
                                    # 提取指标名称
                                    indicator_name = ":".join(indicator.split(":")[1:]).strip()
                                    if ".xlsx" in indicator_name:
                                        indicator_name = indicator_name.replace(".xlsx", "")
                                    
                                    indicator_performance.append({
                                        'indicator': indicator,
                                        'name': indicator_name,
                                        'yoy_change': yoy_change,
                                        'current_value': current_year_value,
                                        'last_year_value': last_year_value,
                                        'last_week': last_valid_week
                                    })
                                except Exception as e:
                                    st.error(f"处理指标 {indicator} 时出错: {str(e)}")
                                    continue
                            
                            # 优化：按同比变化降序排序，使同比上升的排在前面
                            # 只对具有有效同比变化值的指标进行排序
                            valid_indicators = [item for item in indicator_performance if item['yoy_change'] is not None]
                            no_change_indicators = [item for item in indicator_performance if item['yoy_change'] is None]
                            
                            # 按同比变化排序
                            valid_indicators.sort(key=lambda x: x['yoy_change'] if x['yoy_change'] is not None else float('-inf'), reverse=True)
                            
                            # 将排序后的有效指标和无变化指标合并
                            indicator_performance = valid_indicators + no_change_indicators
                            
                            # 显示排序后的指标
                            if indicator_performance:
                                with st.expander("指标同比情况", expanded=False):
                                    st.markdown("""
                                        <style>
                                        .stExpander {
                                            font-size: 1.2rem;
                                            font-weight: bold;
                                            color: #000000;
                                        }
                                        </style>
                                    """, unsafe_allow_html=True)
                                    perf_df = pd.DataFrame(indicator_performance)
                                    
                                    # 确保所有必需的列都存在
                                    for col in ['name', 'current_value', 'last_year_value', 'yoy_change', 'last_week']:
                                        if col not in perf_df.columns:
                                            perf_df[col] = None
                                    
                                    perf_df = perf_df[['name', 'current_value', 'last_year_value', 'yoy_change', 'last_week']]
                                    
                                    # 计算同比变化百分比（安全处理）
                                    perf_df['yoy_change_pct'] = perf_df.apply(
                                        lambda row: (row['current_value'] - row['last_year_value']) / row['last_year_value'] * 100 
                                        if pd.notna(row['current_value']) and pd.notna(row['last_year_value']) and row['last_year_value'] != 0 
                                        else None, axis=1)
                                    
                                    # 将周数转换为日期（安全处理）
                                    perf_df['last_week_date'] = perf_df['last_week'].apply(
                                        lambda x: (df_weekly.index.max() - pd.Timedelta(weeks=df_weekly.index.max().isocalendar().week - int(x))).strftime('%Y-%m-%d') 
                                        if pd.notna(x) and isinstance(x, (int, float)) and not pd.isna(df_weekly.index.max()) 
                                        else '')
                                    
                                    # 重新排列列顺序
                                    perf_df = perf_df[['name', 'last_week_date', 'current_value', 'last_year_value', 'yoy_change', 'yoy_change_pct']]
                                    perf_df.columns = ['指标名称', '最新数据日期', '当前值', '去年同期', '同比变化', '同比变化(%)']
                                    
                                    # 安全排序（处理空值）
                                    try:
                                        # 先筛选出有效值和无效值
                                        valid_df = perf_df.dropna(subset=['同比变化(%)'])
                                        invalid_df = perf_df[perf_df['同比变化(%)'].isna()]
                                        
                                        if not valid_df.empty:
                                            valid_df = valid_df.sort_values('同比变化(%)', ascending=False)
                                            # 合并排序后的数据
                                            perf_df = pd.concat([valid_df, invalid_df])
                                    except Exception as e:
                                        st.error(f"排序指标数据时出错: {str(e)}")
                                    
                                    # 格式化数据，处理空值
                                    formatted_df = perf_df.copy()
                                    for col in ['当前值', '去年同期', '同比变化', '同比变化(%)']:
                                        try:
                                            formatted_df[col] = formatted_df[col].apply(
                                                lambda x: f"{float(x):.2f}" if pd.notna(x) else "")
                                        except Exception as e:
                                            formatted_df[col] = formatted_df[col].astype(str)
                                    
                                    # 单独处理同比变化百分比列，添加百分号
                                    try:
                                        formatted_df['同比变化(%)'] = formatted_df['同比变化(%)'].apply(
                                            lambda x: f"{float(x):.2f}%" if pd.notna(x) and x != "" else "")
                                    except Exception as e:
                                        pass
                                    
                                    # 自定义颜色映射
                                    def color_negative_red(val):
                                        if pd.isna(val) or val == "":
                                            return ''
                                        try:
                                            # 移除百分号再转换
                                            if isinstance(val, str) and '%' in val:
                                                val = float(val.replace('%', ''))
                                            else:
                                                val = float(val)
                                                
                                            if val > 0:
                                                return 'background-color: #ffcccc; color: black'
                                            else:
                                                return 'background-color: #ccffcc; color: black'
                                        except (ValueError, TypeError):
                                            return ''
                                    
                                    # 应用样式到同比变化和同比变化(%)列
                                    styled_df = formatted_df.style.applymap(color_negative_red, subset=['同比变化', '同比变化(%)'])
                                    st.dataframe(styled_df)
                                
                                # 按排序后的顺序重新整理指标列表
                                sorted_indicators = [item['indicator'] for item in indicator_performance]
                                
                                # 显示排序后的周度指标，使用两列布局
                                cols = st.columns(2)
                                
                                # 将指标平均分配到两列中
                                half = len(sorted_indicators) // 2 + len(sorted_indicators) % 2
                                left_indicators = sorted_indicators[:half]
                                right_indicators = sorted_indicators[half:]
                                
                                # 左列指标
                                for indicator in left_indicators:
                                    with cols[0]:
                                        try:
                                            # 处理数据
                                            weekly_data = process_weekly_data(df_weekly, indicator)
                                            
                                            # 提取指标名称
                                            indicator_name = ":".join(indicator.split(":")[1:]).strip()
                                            if ".xlsx" in indicator_name:
                                                indicator_name = indicator_name.replace(".xlsx", "")
                                            
                                            fig = go.Figure()
                                            
                                            # 添加历史区间
                                            if 'max' in weekly_data.columns and 'min' in weekly_data.columns:
                                                fig.add_trace(go.Scatter(
                                                    x=list(weekly_data.index),
                                                    y=weekly_data['max'],
                                                    mode='lines',
                                                    line=dict(width=0),
                                                    name='历史区间',
                                                    showlegend=True,
                                                    hoverinfo='skip'
                                                ))
                                                
                                                fig.add_trace(go.Scatter(
                                                    x=list(weekly_data.index),
                                                    y=weekly_data['min'],
                                                    mode='lines',
                                                    line=dict(width=0),
                                                    fill='tonexty',
                                                    fillcolor='rgba(180, 180, 190, 0.65)',
                                                    name='历史区间',
                                                    showlegend=False,
                                                    hoverinfo='skip'
                                                ))
                                            
                                            # 添加历史均值线
                                            if 'mean' in weekly_data.columns:
                                                fig.add_trace(go.Scatter(
                                                    x=list(weekly_data.index),
                                                    y=weekly_data['mean'],
                                                    mode='lines',
                                                    line=dict(color='#404040', width=1, dash='dash'),
                                                    name='历史均值',
                                                    showlegend=True,
                                                    hovertemplate=f'W%{{x}}: %{{y:.2f}}<extra>{df_weekly.index.max().year-1}年</extra>'
                                                ))
                                            
                                            # 添加去年数据线
                                            if 'last_year' in weekly_data.columns:
                                                fig.add_trace(go.Scatter(
                                                    x=list(weekly_data.index),
                                                    y=weekly_data['last_year'],
                                                    mode='lines',
                                                    line=dict(color='#4150f4', width=2),
                                                    name=f'{df_weekly.index.max().year-1}年',
                                                    showlegend=True,
                                                    hovertemplate=f'W%{{x}}: %{{y:.2f}}<extra>{df_weekly.index.max().year-1}年</extra>'
                                                ))
                                            
                                            # 添加今年数据线
                                            if 'current_year' in weekly_data.columns:
                                                fig.add_trace(go.Scatter(
                                                    x=list(weekly_data.index),
                                                    y=weekly_data['current_year'],
                                                    mode='lines',
                                                    line=dict(color='#F15A29', width=2),
                                                    name=f'{df_weekly.index.max().year}年',
                                                    showlegend=True,
                                                    hovertemplate=f'W%{{x}}: %{{y:.2f}}<extra>{df_weekly.index.max().year}年</extra>'
                                                ))
                                            
                                            fig.update_layout(
                                                title=dict(
                                                    text=indicator_name,
                                                    x=0,
                                                    xanchor='left',
                                                    y=0.95,
                                                    yanchor='top',
                                                    font=dict(size=16, color='#1a162d')
                                                ),
                                                height=400,
                                                margin=dict(l=50, r=20, t=60, b=60),
                                                showlegend=True,
                                                legend=dict(
                                                    orientation="h",
                                                    yanchor="top",
                                                    y=-0.15,
                                                    xanchor="center",
                                                    x=0.5,
                                                    bgcolor='rgba(255,255,255,0)',
                                                    bordercolor='rgba(0,0,0,0)',
                                                    borderwidth=0,
                                                    font=dict(size=12, color='#1a162d'),
                                                    traceorder='reversed'
                                                ),
                                                xaxis=dict(
                                                    title=None,
                                                    showgrid=False,
                                                    showline=True,
                                                    linewidth=1,
                                                    linecolor='black',
                                                    tickmode='array',
                                                    tickvals=list(range(1, 53, 4)),
                                                    ticktext=[f'W{w}' for w in range(1, 53, 4)],
                                                    tickangle=0,
                                                    range=[0.5, 52.5],
                                                    tickfont=dict(size=12)
                                                ),
                                                yaxis=dict(
                                                    title=None,
                                                    showgrid=True,
                                                    gridwidth=1,
                                                    gridcolor='rgba(220, 220, 220, 0.4)',
                                                    showline=True,
                                                    linewidth=1,
                                                    linecolor='black',
                                                    tickfont=dict(size=12)
                                                ),
                                                plot_bgcolor='white',
                                                hoverlabel=dict(
                                                    bgcolor="white",
                                                    font_size=12,
                                                    font_family="Arial"
                                                )
                                            )
                                            st.plotly_chart(fig, use_container_width=True)
                                        except Exception as e:
                                            st.error(f"处理指标 {indicator} 时出错: {str(e)}")
                                            continue
                                
                                # 右列指标
                                for indicator in right_indicators:
                                    with cols[1]:
                                        try:
                                            # 处理数据
                                            weekly_data = process_weekly_data(df_weekly, indicator)
                                            
                                            # 提取指标名称
                                            indicator_name = ":".join(indicator.split(":")[1:]).strip()
                                            if ".xlsx" in indicator_name:
                                                indicator_name = indicator_name.replace(".xlsx", "")
                                            
                                            fig = go.Figure()
                                            
                                            # 添加历史区间
                                            if 'max' in weekly_data.columns and 'min' in weekly_data.columns:
                                                fig.add_trace(go.Scatter(
                                                    x=list(weekly_data.index),
                                                    y=weekly_data['max'],
                                                    mode='lines',
                                                    line=dict(width=0),
                                                    name='历史区间',
                                                    showlegend=True,
                                                    hoverinfo='skip'
                                                ))
                                                
                                                fig.add_trace(go.Scatter(
                                                    x=list(weekly_data.index),
                                                    y=weekly_data['min'],
                                                    mode='lines',
                                                    line=dict(width=0),
                                                    fill='tonexty',
                                                    fillcolor='rgba(180, 180, 190, 0.65)',
                                                    name='历史区间',
                                                    showlegend=False,
                                                    hoverinfo='skip'
                                                ))
                                            
                                            # 添加历史均值线
                                            if 'mean' in weekly_data.columns:
                                                fig.add_trace(go.Scatter(
                                                    x=list(weekly_data.index),
                                                    y=weekly_data['mean'],
                                                    mode='lines',
                                                    line=dict(color='#404040', width=1, dash='dash'),
                                                    name='历史均值',
                                                    showlegend=True,
                                                    hovertemplate=f'W%{{x}}: %{{y:.2f}}<extra>{df_weekly.index.max().year-1}年</extra>'
                                                ))
                                            
                                            # 添加去年数据线
                                            if 'last_year' in weekly_data.columns:
                                                fig.add_trace(go.Scatter(
                                                    x=list(weekly_data.index),
                                                    y=weekly_data['last_year'],
                                                    mode='lines',
                                                    line=dict(color='#4150f4', width=2),
                                                    name=f'{df_weekly.index.max().year-1}年',
                                                    showlegend=True,
                                                    hovertemplate=f'W%{{x}}: %{{y:.2f}}<extra>{df_weekly.index.max().year-1}年</extra>'
                                                ))
                                            
                                            # 添加今年数据线
                                            if 'current_year' in weekly_data.columns:
                                                fig.add_trace(go.Scatter(
                                                    x=list(weekly_data.index),
                                                    y=weekly_data['current_year'],
                                                    mode='lines',
                                                    line=dict(color='#F15A29', width=2),
                                                    name=f'{df_weekly.index.max().year}年',
                                                    showlegend=True,
                                                    hovertemplate=f'W%{{x}}: %{{y:.2f}}<extra>{df_weekly.index.max().year}年</extra>'
                                                ))
                                            
                                            fig.update_layout(
                                                title=dict(
                                                    text=indicator_name,
                                                    x=0,
                                                    xanchor='left',
                                                    y=0.95,
                                                    yanchor='top',
                                                    font=dict(size=16, color='#1a162d')
                                                ),
                                                height=400,
                                                margin=dict(l=50, r=20, t=60, b=60),
                                                showlegend=True,
                                                legend=dict(
                                                    orientation="h",
                                                    yanchor="top",
                                                    y=-0.15,
                                                    xanchor="center",
                                                    x=0.5,
                                                    bgcolor='rgba(255,255,255,0)',
                                                    bordercolor='rgba(0,0,0,0)',
                                                    borderwidth=0,
                                                    font=dict(size=12, color='#1a162d'),
                                                    traceorder='reversed'
                                                ),
                                                xaxis=dict(
                                                    title=None,
                                                    showgrid=False,
                                                    showline=True,
                                                    linewidth=1,
                                                    linecolor='black',
                                                    tickmode='array',
                                                    tickvals=list(range(1, 53, 4)),
                                                    ticktext=[f'W{w}' for w in range(1, 53, 4)],
                                                    tickangle=0,
                                                    range=[0.5, 52.5],
                                                    tickfont=dict(size=12)
                                                ),
                                                yaxis=dict(
                                                    title=None,
                                                    showgrid=True,
                                                    gridwidth=1,
                                                    gridcolor='rgba(220, 220, 220, 0.4)',
                                                    showline=True,
                                                    linewidth=1,
                                                    linecolor='black',
                                                    tickfont=dict(size=12)
                                                ),
                                                plot_bgcolor='white',
                                                hoverlabel=dict(
                                                    bgcolor="white",
                                                    font_size=12,
                                                    font_family="Arial"
                                                )
                                            )
                                            st.plotly_chart(fig, use_container_width=True)
                                        except Exception as e:
                                            st.error(f"处理指标 {indicator} 时出错: {str(e)}")
                                            continue
                        
                        # 在月度数据标签页中显示月度数据
                        with tab2:
                            # 移除"行业月度分析"标题
                            
                            if df_monthly is not None and not df_monthly.empty:
                                # 获取所有行业列
                                all_industries = set()
                                monthly_cols_by_industry = {}
                                
                                # 按行业分组整理数据
                                for col in df_monthly.columns:
                                    if ':月度:' in col:
                                        industry = col.split(':月度:')[0]
                                        if industry not in monthly_cols_by_industry:
                                            monthly_cols_by_industry[industry] = []
                                        monthly_cols_by_industry[industry].append(col)
                                        all_industries.add(industry)
                                
                                if not monthly_cols_by_industry:
                                    st.warning("未找到任何行业的月度规模以上工业增加值同比数据。")
                                else:
                                    # 处理月度数据
                                    def prepare_monthly_data(df, col):
                                        """处理月度数据，按月统计数据"""
                                        # 确保索引是datetime类型
                                        df.index = pd.to_datetime(df.index)
                                        
                                        # 创建月份和年份列
                                        months = df.index.month
                                        years = df.index.year
                                        
                                        # 获取当前年份和去年
                                        current_year = df.index.max().year
                                        last_year = current_year - 1
                                        
                                        # 获取历史年份列表（过去5年）
                                        min_year = df.index.min().year
                                        historical_years = [y for y in range(current_year - 5, current_year) if y >= min_year]
                                        
                                        # 创建结果数据框，索引为月份1-12
                                        result = pd.DataFrame(index=range(1, 13))
                                        
                                        # 对每月计算统计值
                                        for month in range(1, 13):
                                            month_mask = months == month
                                            
                                            # 当年的数据
                                            current_year_data = df[month_mask & (years == current_year)][col]
                                            if not current_year_data.empty:
                                                result.loc[month, 'current_year'] = current_year_data.iloc[0]
                                            
                                            # 去年的数据
                                            last_year_data = df[month_mask & (years == last_year)][col]
                                            if not last_year_data.empty:
                                                result.loc[month, 'last_year'] = last_year_data.iloc[0]
                                            
                                            # 历史数据（近5年）
                                            hist_mask = month_mask & years.isin(historical_years)
                                            hist_data = df[hist_mask][col]
                                            if not hist_data.empty:
                                                result.loc[month, 'min'] = hist_data.min()
                                                result.loc[month, 'max'] = hist_data.max()
                                        
                                        return result
                                    
                                    # 创建两列布局
                                    cols = st.columns(2)
                                    
                                    # 将行业平均分配到两列中
                                    industry_list = sorted(list(all_industries))
                                    half = len(industry_list) // 2 + len(industry_list) % 2
                                    left_industries = industry_list[:half]
                                    right_industries = industry_list[half:]
                                    
                                    # 月份名称映射
                                    month_names = {
                                        1: "Jan", 2: "Feb", 3: "Mar", 4: "Apr", 5: "May", 6: "Jun",
                                        7: "Jul", 8: "Aug", 9: "Sep", 10: "Oct", 11: "Nov", 12: "Dec"
                                    }
                                    
                                    # 左列行业
                                    for industry in left_industries:
                                        with cols[0]:
                                            # 获取该行业下第一个月度指标
                                            if industry in monthly_cols_by_industry and monthly_cols_by_industry[industry]:
                                                indicator = monthly_cols_by_industry[industry][0]
                                                
                                                # 格式化显示名称，去掉前缀
                                                format_name = indicator.split(':月度:')[1] if ':月度:' in indicator else indicator
                                                
                                                try:
                                                    # 处理数据
                                                    monthly_data = prepare_monthly_data(df_monthly, indicator)
                                                    
                                                    # 创建图表
                                                    fig = go.Figure()
                                                    
                                                    # 添加历史区间（近5年最大最小值）
                                                    if 'max' in monthly_data.columns and 'min' in monthly_data.columns:
                                                        fig.add_trace(go.Scatter(
                                                            x=list(monthly_data.index),
                                                            y=monthly_data['max'],
                                                            mode='lines',
                                                            line=dict(width=0),
                                                            name='历史区间',
                                                            showlegend=True,
                                                            hoverinfo='skip'
                                                        ))
                                                        
                                                        fig.add_trace(go.Scatter(
                                                            x=list(monthly_data.index),
                                                            y=monthly_data['min'],
                                                            mode='lines',
                                                            line=dict(width=0),
                                                            fill='tonexty',
                                                            fillcolor='rgba(180, 180, 190, 0.65)',
                                                            name='历史区间',
                                                            showlegend=False,
                                                            hoverinfo='skip'
                                                        ))
                                                    
                                                    # 添加去年数据线
                                                    if 'last_year' in monthly_data.columns:
                                                        fig.add_trace(go.Scatter(
                                                            x=list(monthly_data.index),
                                                            y=monthly_data['last_year'],
                                                            mode='lines+markers',
                                                            line=dict(color='#4150f4', width=2),
                                                            name=f'{df_monthly.index.max().year-1}年',
                                                            showlegend=True,
                                                            hovertemplate='%{x}月: %{y:.2f}%<extra></extra>'
                                                        ))
                                                    
                                                    # 添加今年数据线
                                                    if 'current_year' in monthly_data.columns:
                                                        fig.add_trace(go.Scatter(
                                                            x=list(monthly_data.index),
                                                            y=monthly_data['current_year'],
                                                            mode='lines+markers',
                                                            line=dict(color='#F15A29', width=2),
                                                            name=f'{df_monthly.index.max().year}年',
                                                            showlegend=True,
                                                            hovertemplate='%{x}月: %{y:.2f}%<extra></extra>'
                                                        ))
                                                    
                                                    # 设置图表布局
                                                    fig.update_layout(
                                                        title=dict(
                                                            text=f"{industry}",
                                                            x=0,
                                                            xanchor='left',
                                                            y=0.95,
                                                            yanchor='top',
                                                            font=dict(size=16, color='#1a162d')
                                                        ),
                                                        height=400,
                                                        margin=dict(l=50, r=20, t=60, b=60),
                                                        showlegend=True,
                                                        legend=dict(
                                                            orientation="h",
                                                            yanchor="top",
                                                            y=-0.15,
                                                            xanchor="center",
                                                            x=0.5,
                                                            bgcolor='rgba(255,255,255,0)',
                                                            bordercolor='rgba(0,0,0,0)',
                                                            borderwidth=0,
                                                            font=dict(size=12, color='#1a162d')
                                                        ),
                                                        xaxis=dict(
                                                            title=None,
                                                            showgrid=False,
                                                            showline=True,
                                                            linewidth=1,
                                                            linecolor='black',
                                                            tickmode='array',
                                                            tickvals=list(range(1, 13)),
                                                            ticktext=[month_names[m] for m in range(1, 13)],
                                                            tickangle=0,
                                                            range=[0.5, 12.5],
                                                            tickfont=dict(size=12)
                                                        ),
                                                        yaxis=dict(
                                                            title="同比增长率(%)",
                                                            showgrid=True,
                                                            gridwidth=1,
                                                            gridcolor='rgba(220, 220, 220, 0.4)',
                                                            showline=True,
                                                            linewidth=1,
                                                            linecolor='black',
                                                            tickfont=dict(size=12),
                                                            zeroline=True,
                                                            zerolinecolor='rgba(0, 0, 0, 0.2)',
                                                            zerolinewidth=1
                                                        ),
                                                        plot_bgcolor='white',
                                                        hoverlabel=dict(
                                                            bgcolor="white",
                                                            font_size=12,
                                                            font_family="Arial"
                                                        )
                                                    )
                                                    
                                                    # 显示图表
                                                    st.plotly_chart(fig, use_container_width=True)
                                                except Exception as e:
                                                    st.error(f"处理行业 {industry} 的月度数据时出错: {str(e)}")
                                    
                                    # 右列行业
                                    for industry in right_industries:
                                        with cols[1]:
                                            # 获取该行业下第一个月度指标
                                            if industry in monthly_cols_by_industry and monthly_cols_by_industry[industry]:
                                                indicator = monthly_cols_by_industry[industry][0]
                                                
                                                # 格式化显示名称，去掉前缀
                                                format_name = indicator.split(':月度:')[1] if ':月度:' in indicator else indicator
                                                
                                                try:
                                                    # 处理数据
                                                    monthly_data = prepare_monthly_data(df_monthly, indicator)
                                                    
                                                    # 创建图表
                                                    fig = go.Figure()
                                                    
                                                    # 添加历史区间（近5年最大最小值）
                                                    if 'max' in monthly_data.columns and 'min' in monthly_data.columns:
                                                        fig.add_trace(go.Scatter(
                                                            x=list(monthly_data.index),
                                                            y=monthly_data['max'],
                                                            mode='lines',
                                                            line=dict(width=0),
                                                            name='历史区间',
                                                            showlegend=True,
                                                            hoverinfo='skip'
                                                        ))
                                                        
                                                        fig.add_trace(go.Scatter(
                                                            x=list(monthly_data.index),
                                                            y=monthly_data['min'],
                                                            mode='lines',
                                                            line=dict(width=0),
                                                            fill='tonexty',
                                                            fillcolor='rgba(180, 180, 190, 0.65)',
                                                            name='历史区间',
                                                            showlegend=False,
                                                            hoverinfo='skip'
                                                        ))
                                                    
                                                    # 添加去年数据线
                                                    if 'last_year' in monthly_data.columns:
                                                        fig.add_trace(go.Scatter(
                                                            x=list(monthly_data.index),
                                                            y=monthly_data['last_year'],
                                                            mode='lines+markers',
                                                            line=dict(color='#4150f4', width=2),
                                                            name=f'{df_monthly.index.max().year-1}年',
                                                            showlegend=True,
                                                            hovertemplate='%{x}月: %{y:.2f}%<extra></extra>'
                                                        ))
                                                    
                                                    # 添加今年数据线
                                                    if 'current_year' in monthly_data.columns:
                                                        fig.add_trace(go.Scatter(
                                                            x=list(monthly_data.index),
                                                            y=monthly_data['current_year'],
                                                            mode='lines+markers',
                                                            line=dict(color='#F15A29', width=2),
                                                            name=f'{df_monthly.index.max().year}年',
                                                            showlegend=True,
                                                            hovertemplate='%{x}月: %{y:.2f}%<extra></extra>'
                                                        ))
                                                    
                                                    # 设置图表布局
                                                    fig.update_layout(
                                                        title=dict(
                                                            text=f"{industry}",
                                                            x=0,
                                                            xanchor='left',
                                                            y=0.95,
                                                            yanchor='top',
                                                            font=dict(size=16, color='#1a162d')
                                                        ),
                                                        height=400,
                                                        margin=dict(l=50, r=20, t=60, b=60),
                                                        showlegend=True,
                                                        legend=dict(
                                                            orientation="h",
                                                            yanchor="top",
                                                            y=-0.15,
                                                            xanchor="center",
                                                            x=0.5,
                                                            bgcolor='rgba(255,255,255,0)',
                                                            bordercolor='rgba(0,0,0,0)',
                                                            borderwidth=0,
                                                            font=dict(size=12, color='#1a162d')
                                                        ),
                                                        xaxis=dict(
                                                            title=None,
                                                            showgrid=False,
                                                            showline=True,
                                                            linewidth=1,
                                                            linecolor='black',
                                                            tickmode='array',
                                                            tickvals=list(range(1, 13)),
                                                            ticktext=[month_names[m] for m in range(1, 13)],
                                                            tickangle=0,
                                                            range=[0.5, 12.5],
                                                            tickfont=dict(size=12)
                                                        ),
                                                        yaxis=dict(
                                                            title="同比增长率(%)",
                                                            showgrid=True,
                                                            gridwidth=1,
                                                            gridcolor='rgba(220, 220, 220, 0.4)',
                                                            showline=True,
                                                            linewidth=1,
                                                            linecolor='black',
                                                            tickfont=dict(size=12),
                                                            zeroline=True,
                                                            zerolinecolor='rgba(0, 0, 0, 0.2)',
                                                            zerolinewidth=1
                                                        ),
                                                        plot_bgcolor='white',
                                                        hoverlabel=dict(
                                                            bgcolor="white",
                                                            font_size=12,
                                                            font_family="Arial"
                                                        )
                                                    )
                                                    
                                                    # 显示图表
                                                    st.plotly_chart(fig, use_container_width=True)
                                                except Exception as e:
                                                    st.error(f"处理行业 {industry} 的月度数据时出错: {str(e)}")
                            else:
                                st.warning("未找到任何月度数据。请确保上传的文件中包含monthly工作表。")


                        # 在扩散指数标签页中显示扩散指数分析
                        with tab3:
                            # 移除"扩散指数分析"标题
                            
                            # 选择行业大类
                            st.markdown("#### 选择行业大类")
                            selected_di_category = st.selectbox("", categories, key='di_industry_selector')
                            
                            if selected_di_category:
                                # 获取该大类的所有变量
                                category_cols = []
                                for col in df_weekly.columns:
                                    if col.startswith(selected_di_category + ':'):
                                        category_cols.append(col)
                                
                                # 显示可用指标数量
                                # st.write(f"找到 {len(category_cols)} 个指标用于计算扩散指数")
                                
                                # 确保有足够的数据用于扩散指数计算
                                if len(category_cols) < 2:
                                    st.warning(f"行业 '{selected_di_category}' 的可用指标数量不足，无法计算扩散指数。至少需要2个指标。")
                                else:
                                    # 筛选该行业的数据并创建独立副本
                                    industry_df = df_weekly[category_cols].copy()
                                    
                                    # 提示正在使用的数据形状
                                    # st.info(f"使用数据: {industry_df.shape[0]} 行 × {industry_df.shape[1]} 列")
                                    
                                    # 确保日期索引是日期类型
                                    industry_df.index = pd.to_datetime(industry_df.index)
                                    
                                    # 检查是否有足够的时间序列数据
                                    date_range = industry_df.index.max() - industry_df.index.min()
                                    if date_range.days < 365:
                                        st.warning(f"数据时间跨度仅有 {date_range.days} 天，不足一年，可能无法计算同比扩散指数。")
                                    
                                    # 先设置阈值
                                    col1, col2, col3, col4 = st.columns(4)
                                    with col1:
                                        di_threshold = st.slider("变化判定阈值", min_value=0.0, max_value=0.05, value=0.0, step=0.001, 
                                                                format="%.3f", help="指标正向变化超过该阈值才计入改善指标")
                                    
                                    # 尝试计算扩散指数
                                    try:
                                        # 使用简单直观的方法重新实现扩散指数计算（内联代码，不依赖外部函数）
                                        # 步骤1: 计算同比变化率（当前值与一年前的值相比）
                                        yoy_changes = industry_df.pct_change(periods=52)
                                        # 步骤2: 计算环比变化率（当前值与上一期的值相比）
                                        mom_changes = industry_df.pct_change(periods=4)
                                        
                                        # 步骤3: 计算组合变化率
                                        # 首先计算同比值
                                        yoy_relative = industry_df / industry_df.shift(52) - 1
                                        # 然后计算同比值的环比变化
                                        combined_changes = yoy_relative.pct_change(periods=4)
                                        
                                        # 显示变化率数据的有效性信息
                                        # st.write(f"同比变化率有效数据点: {yoy_changes.count().sum()} / {yoy_changes.size}") 
                                        # st.write(f"环比变化率有效数据点: {mom_changes.count().sum()} / {mom_changes.size}")
                                        # st.write(f"组合变化率有效数据点: {combined_changes.count().sum()} / {combined_changes.size}")
                                        
                                        # 步骤4: 创建结果DataFrame
                                        result_df = pd.DataFrame(index=industry_df.index)
                                        
                                        # 步骤5: 计算每一周的扩散指数
                                        def calculate_di_for_changes(changes):
                                            """计算特定变化率的每周扩散指数"""
                                            weekly_di = []
                                            valid_dates = []
                                            
                                            for date in changes.index:
                                                try:
                                                    # 获取当周的数据
                                                    week_data = changes.loc[date]
                                                    
                                                    # 检查非缺失值数量
                                                    non_na_count = week_data.count()
                                                    total_count = len(week_data)
                                                    
                                                    # 如果缺失值过多，则标记为无效
                                                    if non_na_count < total_count * 0.67 or non_na_count < 2:
                                                        weekly_di.append(np.nan)
                                                        valid_dates.append(False)
                                                        continue
                                                    
                                                    # 计算变化方向（考虑阈值）
                                                    improvements = (week_data > di_threshold).sum()
                                                    
                                                    # 计算扩散指数 (0-100范围)
                                                    di_value = (improvements / non_na_count) * 100
                                                    weekly_di.append(di_value)
                                                    valid_dates.append(True)
                                                except Exception as e:
                                                    weekly_di.append(np.nan)
                                                    valid_dates.append(False)
                                                    continue
                                            
                                            return pd.Series(weekly_di, index=changes.index), valid_dates
                                        
                                        # 计算三种扩散指数
                                        yoy_di, yoy_valid = calculate_di_for_changes(yoy_changes)
                                        mom_di, mom_valid = calculate_di_for_changes(mom_changes)
                                        combined_di, combined_valid = calculate_di_for_changes(combined_changes)
                                        
                                        # 添加到结果DataFrame
                                        result_df['yoy'] = yoy_di
                                        result_df['mom'] = mom_di
                                        result_df['combined'] = combined_di
                                        
                                        # 统计有效数据点的数量
                                        valid_yoy_count = sum(yoy_valid)
                                        valid_mom_count = sum(mom_valid)
                                        valid_combined_count = sum(combined_valid)
                                        
                                        # st.write(f"有效的同比扩散指数数据点: {valid_yoy_count}")
                                        # st.write(f"有效的环比扩散指数数据点: {valid_mom_count}")
                                        # st.write(f"有效的组合扩散指数数据点: {valid_combined_count}")
                                        
                                        # 检查是否有足够的有效数据点
                                        if valid_yoy_count < 1 and valid_mom_count < 1 and valid_combined_count < 1:
                                            st.error("没有足够的有效数据点来计算扩散指数。尝试减少阈值或检查数据质量。")
                                        else:
                                            # 获取最新的扩散指数值
                                            latest_yoy = result_df['yoy'].dropna().iloc[-1] if not result_df['yoy'].dropna().empty else np.nan
                                            latest_mom = result_df['mom'].dropna().iloc[-1] if not result_df['mom'].dropna().empty else np.nan
                                            latest_combined = result_df['combined'].dropna().iloc[-1] if not result_df['combined'].dropna().empty else np.nan
                                            
                                            # 获取前一期值（如果有足够数据点）
                                            previous_yoy = result_df['yoy'].dropna().iloc[-2] if len(result_df['yoy'].dropna()) > 1 else np.nan
                                            previous_mom = result_df['mom'].dropna().iloc[-2] if len(result_df['mom'].dropna()) > 1 else np.nan
                                            previous_combined = result_df['combined'].dropna().iloc[-2] if len(result_df['combined'].dropna()) > 1 else np.nan
                                            
                                            # 显示最新值
                                            with col2:
                                                if not np.isnan(latest_yoy):
                                                    delta = latest_yoy - previous_yoy if not np.isnan(previous_yoy) else None
                                                    delta_color = "inverse" if delta is not None else None
                                                    delta_formatted = f"{delta:.2f}" if delta is not None else None
                                                    st.metric("同比扩散指数", f"{latest_yoy:.2f}", delta=delta_formatted, delta_color=delta_color)
                                                else:
                                                    st.metric("同比扩散指数", "N/A")
                                            
                                            with col3:
                                                if not np.isnan(latest_mom):
                                                    delta = latest_mom - previous_mom if not np.isnan(previous_mom) else None
                                                    delta_color = "inverse" if delta is not None else None
                                                    delta_formatted = f"{delta:.2f}" if delta is not None else None
                                                    st.metric("环比扩散指数", f"{latest_mom:.2f}", delta=delta_formatted, delta_color=delta_color)
                                                else:
                                                    st.metric("环比扩散指数", "N/A")
                                            
                                            with col4:
                                                if not np.isnan(latest_combined):
                                                    delta = latest_combined - previous_combined if not np.isnan(previous_combined) else None
                                                    delta_color = "inverse" if delta is not None else None
                                                    delta_formatted = f"{delta:.2f}" if delta is not None else None
                                                    st.metric("同环比扩散指数", f"{latest_combined:.2f}", delta=delta_formatted, delta_color=delta_color)
                                                else:
                                                    st.metric("同环比扩散指数", "N/A")
                                            
                                            # 创建三个指数的标签页
                                            di_tab1, di_tab2, di_tab3 = st.tabs(["同比扩散指数", "环比扩散指数", "同环比扩散指数"])
                                            
                                            with di_tab1:
                                                # 创建同比扩散指数图表
                                                fig = go.Figure()
                                                
                                                # 添加扩散指数线
                                                fig.add_trace(go.Scatter(
                                                    x=result_df.index,
                                                    y=result_df['yoy'],
                                                    name='同比扩散指数',
                                                    mode='lines+markers',
                                                    line=dict(width=2, color='#1f77b4'),
                                                    marker=dict(size=4, color='#1f77b4')
                                                ))
                                                
                                                # 添加中性线
                                                fig.add_hline(y=50, line_dash="dash", line_color="red", opacity=0.5, annotation_text="中性值")
                                                
                                                # 更新布局
                                                fig.update_layout(
                                                    title=f"{selected_di_category} 同比扩散指数",
                                                    yaxis_title="扩散指数值",
                                                    yaxis=dict(range=[0, 100]),
                                                    showlegend=True,
                                                    hovermode='x unified',
                                                    template='plotly_white'
                                                )
                                                
                                                # 更新x轴
                                                fig.update_xaxes(
                                                    rangeslider_visible=True,
                                                    rangeselector=dict(
                                                        buttons=list([
                                                            dict(count=6, label="6月", step="month", stepmode="backward"),
                                                            dict(count=1, label="1年", step="year", stepmode="backward"),
                                                            dict(count=3, label="3年", step="year", stepmode="backward"),
                                                            dict(step="all", label="全部")
                                                        ])
                                                    )
                                                )
                                                
                                                # 显示图表
                                                st.plotly_chart(fig, use_container_width=True)
                                            
                                            with di_tab2:
                                                # 创建环比扩散指数图表
                                                fig = go.Figure()
                                                
                                                # 添加扩散指数线
                                                fig.add_trace(go.Scatter(
                                                    x=result_df.index,
                                                    y=result_df['mom'],
                                                    name='环比扩散指数',
                                                    mode='lines+markers',
                                                    line=dict(width=2, color='#ff7f0e'),
                                                    marker=dict(size=4, color='#ff7f0e')
                                                ))
                                                
                                                # 添加中性线
                                                fig.add_hline(y=50, line_dash="dash", line_color="red", opacity=0.5, annotation_text="中性值")
                                                
                                                # 更新布局
                                                fig.update_layout(
                                                    title=f"{selected_di_category} 环比扩散指数",
                                                    yaxis_title="扩散指数值",
                                                    yaxis=dict(range=[0, 100]),
                                                    showlegend=True,
                                                    hovermode='x unified',
                                                    template='plotly_white'
                                                )
                                                
                                                # 更新x轴
                                                fig.update_xaxes(
                                                    rangeslider_visible=True,
                                                    rangeselector=dict(
                                                        buttons=list([
                                                            dict(count=6, label="6月", step="month", stepmode="backward"),
                                                            dict(count=1, label="1年", step="year", stepmode="backward"),
                                                            dict(count=3, label="3年", step="year", stepmode="backward"),
                                                            dict(step="all", label="全部")
                                                        ])
                                                    )
                                                )
                                                
                                                # 显示图表
                                                st.plotly_chart(fig, use_container_width=True)
                                            
                                            with di_tab3:
                                                # 创建同环比扩散指数图表
                                                fig = go.Figure()
                                                
                                                # 添加扩散指数线
                                                fig.add_trace(go.Scatter(
                                                    x=result_df.index,
                                                    y=result_df['combined'],
                                                    name='同环比扩散指数',
                                                    mode='lines+markers',
                                                    line=dict(width=2, color='#2ca02c'),
                                                    marker=dict(size=4, color='#2ca02c')
                                                ))
                                                
                                                # 添加中性线
                                                fig.add_hline(y=50, line_dash="dash", line_color="red", opacity=0.5, annotation_text="中性值")
                                                
                                                # 更新布局
                                                fig.update_layout(
                                                    title=f"{selected_di_category} 同环比扩散指数",
                                                    yaxis_title="扩散指数值",
                                                    yaxis=dict(range=[0, 100]),
                                                    showlegend=True,
                                                    hovermode='x unified',
                                                    template='plotly_white'
                                                )
                                                
                                                # 更新x轴
                                                fig.update_xaxes(
                                                    rangeslider_visible=True,
                                                    rangeselector=dict(
                                                        buttons=list([
                                                            dict(count=6, label="6月", step="month", stepmode="backward"),
                                                            dict(count=1, label="1年", step="year", stepmode="backward"),
                                                            dict(count=3, label="3年", step="year", stepmode="backward"),
                                                            dict(step="all", label="全部")
                                                        ])
                                                    )
                                                )
                                                
                                                # 显示图表
                                                st.plotly_chart(fig, use_container_width=True)
                                            
                                            # 添加扩散指数解释
                                            with st.expander("扩散指数说明"):
                                                st.markdown("""
                                                ### 扩散指数简介
                                                
                                                扩散指数是一种衡量经济指标变化方向的综合指标，计算方法是将正向变化且超过阈值的指标数量占总指标数量的百分比。
                                                
                                                - **取值范围**: 0-100
                                                - **中性值**: 50（表示上升和下降的指标数量相等）
                                                - **大于50**: 表示正向变化的指标数量多于负向变化的指标数量，经济呈扩张趋势
                                                - **小于50**: 表示负向变化的指标数量多于正向变化的指标数量，经济呈收缩趋势
                                                
                                                ### 变化判定阈值
                                                
                                                变化判定阈值用于确定指标变化是否足够显著。只有正向变化且超过阈值的指标才会被计算为改善指标。
                                                
                                                ### 三种扩散指数的区别
                                                
                                                - **同比扩散指数**: 比较当前值与去年同期值的变化
                                                - **环比扩散指数**: 比较当前值与前期值的变化（通常为前一个月或季度）
                                                - **同环比扩散指数**: 结合同比和环比的变化趋势，先计算同比变化率再计算环比变化
                                                
                                                ### 扩散指数的应用
                                                
                                                扩散指数常用于:
                                                - 经济景气监测
                                                - 行业发展趋势分析
                                                - 宏观经济预测
                                                - 政策效果评估
                                                """)
                                    except Exception as e:
                                        st.error(f"计算扩散指数时出错: {str(e)}")
                                        st.write("错误详情:", traceback.format_exc())
                                        st.info("请检查数据格式和质量，确保有足够的历史数据用于计算")

except Exception as e:
    st.error(f"发生错误: {str(e)}")
    st.write("请确保数据文件可用且格式正确。") 