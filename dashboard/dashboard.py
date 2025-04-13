import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import os
import warnings
from statsmodels.tsa.stattools import adfuller
import io # Added for Excel download

# 从重构的脚本中导入核心函数
from data_loader import load_and_process_data
# Import the main calculation function and updated default values
from industry_diffusion_calculator import (
    calculate_all_diffusion_indices, 
    DEFAULT_MISSING_THRESHOLD, # Now defaults to 0.3
    DEFAULT_TOLERANCE_THRESHOLD
)
# Import BOTH summary functions
from growth_calculator import calculate_weekly_summary, calculate_monthly_growth_summary, calculate_weekly_growth_summary
# 导入新的绘图工具
from plotting_utils import calculate_historical_weekly_stats, plot_weekly_indicator, plot_monthly_indicator

# --- Helper Function for Stationarity Test ---
def _run_adf(series, alpha):
    """Helper to run ADF test and return p-value and status ('是', '否', '数据不足', '测试出错')."""
    series = series.dropna()
    if len(series) < 4:
        return np.nan, '数据不足'
    try:
        result = adfuller(series)
        p_value = result[1]
        # STRICT comparison with the passed alpha
        is_stationary = p_value <= alpha 
        status = '是' if is_stationary else '否'
        return p_value, status
    except Exception as e:
        return np.nan, f'测试出错'

def test_and_process_stationarity(df, alpha=0.05):
    """Performs ADF test, attempts various transformations, returns summary & processed data."""
    summary_results = []
    processed_data = {}

    for name, series in df.items():
        series_cleaned = series.dropna()
        
        original_p_value, original_status = _run_adf(series_cleaned, alpha)
        
        processed_p_value = np.nan
        final_status = original_status # Start assuming original status is the final one
        method = '原始序列'
        processed_series = series_cleaned 

        if original_status == '是':
            processed_p_value = original_p_value # P-value when original is stationary
            # No transformation needed, final_status remains '是'
            pass
        elif original_status == '否': # Only try transformations if original is definitively non-stationary
            can_log = (series_cleaned > 0).all() and len(series_cleaned) > 0
            found_stationary_method = None

            transformations = []
            if can_log: transformations.append(('对数变换', np.log(series_cleaned)))
            transformations.append(('一阶差分', series_cleaned.diff()))
            if can_log:
                 # Ensure log series exists for log diff
                log_series_for_diff = np.log(series_cleaned) if 'series_log' not in locals() else series_log
                transformations.append(('对数一阶差分', log_series_for_diff.diff()))
            transformations.append(('二阶差分', series_cleaned.diff().diff()))

            for m, transformed_series in transformations:
                p_val, status = _run_adf(transformed_series, alpha)
                if status == '是':
                    method = m
                    final_status = '是'
                    processed_p_value = p_val
                    processed_series = transformed_series.dropna() # Use the transformed series
                    found_stationary_method = m
                    break # Stop trying further transformations
            
            # If no transformation worked, final status remains '否'
            if not found_stationary_method:
                 method = '无有效方法' # Mark that no method worked
                 final_status = '否' # Explicitly set to No if original was No and nothing worked
                 processed_p_value = np.nan # No p-value resulted in stationarity
                 processed_series = series_cleaned # Revert to original

        # Final adjustments before appending
        # Ensure '处理后P值' is NaN if the method is '原始序列' or '无有效方法'
        p_value_to_report = processed_p_value if method not in ['原始序列', '无有效方法'] and final_status == '是' else np.nan
        
        summary_results.append({
            '指标名称': name,
            '原始P值': original_p_value,
            '原始是否平稳': original_status,
            '处理方法': method,
            '处理后P值': p_value_to_report, # Report p-value only if processed AND stationary
            '最终是否平稳': final_status
        })
        processed_data[name] = processed_series.dropna() # Ensure final processed data has no NaNs from processing

    summary_df = pd.DataFrame(summary_results)
    # Use outer join for concat to handle different lengths/indices after processing
    processed_df = pd.concat(processed_data, axis=1, join='outer') 

    return summary_df, processed_df

# --- Streamlit Page Configuration ---
st.set_page_config(
    page_title="高频数据洞察平台",
    page_icon="📈",
    layout="wide"
)

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
[data-testid="stSidebar"] .stFileUploader label {{ /* Uploader label */
    color: #f0f0f0 !important;
}}
[data-testid="stSidebar"] .st-emotion-cache-16txtl3 {{ color: #f0f0f0 !important; }} /* Headers */
[data-testid="stSidebar"] .st-emotion-cache-1zhivh4 {{ color: #d0d0d0 !important; }} /* Regular text */
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

.uploadedFile {{cursor: default !important;}}
</style>
""", unsafe_allow_html=True)

# --- Callback Functions for Sliders ---
def request_di_recalculation():
    st.session_state.di_recalculation_needed = True

# --- Sidebar --- 
with st.sidebar:
    st.title("📈 高频数据洞察平台")
    st.subheader("上传数据文件")
    uploaded_files = st.file_uploader(
        "拖放或选择 Excel 文件",
        type=["xlsx"],
        accept_multiple_files=True, 
        label_visibility="collapsed"
    )

    # Session State Initialization
    if 'data_loaded' not in st.session_state: st.session_state.data_loaded = False
    if 'weekly_df' not in st.session_state: st.session_state.weekly_df = pd.DataFrame()
    if 'monthly_df' not in st.session_state: st.session_state.monthly_df = pd.DataFrame()
    if 'source_map' not in st.session_state: st.session_state.source_map = {}
    if 'wow_di' not in st.session_state: st.session_state.wow_di = pd.DataFrame()
    if 'yoy_di' not in st.session_state: st.session_state.yoy_di = pd.DataFrame()
    if 'mix_di' not in st.session_state: st.session_state.mix_di = pd.DataFrame()
    if 'industries' not in st.session_state: st.session_state.industries = []
    if 'weekly_summary_cache' not in st.session_state: st.session_state.weekly_summary_cache = {}
    if 'processed_files' not in st.session_state: st.session_state.processed_files = []
    # Initialize master threshold values (DEFAULT_MISSING_THRESHOLD is now 0.3)
    if 'tolerance_threshold' not in st.session_state: st.session_state.tolerance_threshold = DEFAULT_TOLERANCE_THRESHOLD
    if 'missing_threshold' not in st.session_state: st.session_state.missing_threshold = DEFAULT_MISSING_THRESHOLD
    # Initialize the keys associated with the sliders (use updated default for missing)
    if 'tolerance_slider_tab3' not in st.session_state: st.session_state.tolerance_slider_tab3 = st.session_state.tolerance_threshold
    if 'missing_slider_tab3' not in st.session_state: st.session_state.missing_slider_tab3 = st.session_state.missing_threshold # Will be 0.3 initially
    # Initialize recalculation flag
    if 'di_recalculation_needed' not in st.session_state: st.session_state.di_recalculation_needed = False 
    # Add keys for stationarity results
    if 'adf_results_monthly' not in st.session_state: st.session_state.adf_results_monthly = pd.DataFrame()
    if 'adf_results_weekly' not in st.session_state: st.session_state.adf_results_weekly = pd.DataFrame()
    if 'processed_monthly_df' not in st.session_state: st.session_state.processed_monthly_df = pd.DataFrame()
    if 'processed_weekly_df' not in st.session_state: st.session_state.processed_weekly_df = pd.DataFrame()

    # --- Data Loading and Processing Logic --- 
    # This part remains, setting di_recalculation_needed on new data load
    if uploaded_files:
        uploaded_file_names = sorted([f.name for f in uploaded_files])
        if not st.session_state.data_loaded or st.session_state.processed_files != uploaded_file_names:
            with st.spinner('正在加载和处理上传的文件...'):
                try:
                    weekly_df, monthly_df, source_map = load_and_process_data(uploaded_files)
                    st.session_state.weekly_df = weekly_df
                    st.session_state.monthly_df = monthly_df
                    st.session_state.source_map = source_map
                    st.session_state.data_loaded = True
                    st.session_state.industries = sorted(list(set(source_map.values()))) if source_map else []
                    st.session_state.processed_files = uploaded_file_names 
                    st.session_state.weekly_summary_cache = {} 
                    # Clear stationarity results on new data load
                    st.session_state.adf_results_monthly = pd.DataFrame()
                    st.session_state.adf_results_weekly = pd.DataFrame()
                    st.session_state.processed_monthly_df = pd.DataFrame()
                    st.session_state.processed_weekly_df = pd.DataFrame()
                    st.success(f"成功处理 {len(uploaded_files)} 个文件！")
                    st.info(f"周度数据: {weekly_df.shape[0]} 行, {weekly_df.shape[1]} 列")
                    st.info(f"月度数据: {monthly_df.shape[0]} 行, {monthly_df.shape[1]} 列")
                    st.info(f"识别到 {len(st.session_state.industries)} 个行业")
                    st.session_state.di_recalculation_needed = True 
                except Exception as e:
                    st.error(f"处理文件时出错: {e}")
                    st.session_state.data_loaded = False
                    st.session_state.weekly_df = pd.DataFrame()
                    st.session_state.monthly_df = pd.DataFrame()
                    st.session_state.source_map = {}
                    st.session_state.industries = []
                    st.session_state.wow_di = pd.DataFrame()
                    st.session_state.yoy_di = pd.DataFrame()
                    st.session_state.mix_di = pd.DataFrame()
                    st.session_state.processed_files = []
                    st.session_state.weekly_summary_cache = {}
                    st.session_state.adf_results_monthly = pd.DataFrame()
                    st.session_state.adf_results_weekly = pd.DataFrame()
                    st.session_state.processed_monthly_df = pd.DataFrame()
                    st.session_state.processed_weekly_df = pd.DataFrame()

    # Removed the recalculation check block that was dependent on sidebar sliders
    # if st.session_state.data_loaded and (
    #     st.session_state.prev_tolerance != tolerance_threshold or 
    #     st.session_state.prev_missing != missing_threshold or 
    #     recalculate_di # This check is now handled outside the sidebar
    #    ): ...

# --- Recalculate Diffusion Indices (if needed) ---
# This block runs after sidebar processing and before tabs are drawn
if st.session_state.data_loaded and st.session_state.get('di_recalculation_needed', False):
    if not st.session_state.weekly_df.empty and st.session_state.source_map:
        current_tolerance = st.session_state.tolerance_slider_tab3
        current_missing = st.session_state.missing_slider_tab3 # This value is now unified

        st.session_state.tolerance_threshold = current_tolerance
        st.session_state.missing_threshold = current_missing
        
        with st.spinner(f'正在计算/重新计算扩散指数 (Tol={current_tolerance:.3f}, Miss={current_missing:.2f})...'):
            try:
                st.session_state.wow_di, st.session_state.yoy_di, st.session_state.mix_di = \
                    calculate_all_diffusion_indices(
                        st.session_state.weekly_df, 
                        st.session_state.source_map,
                        # Pass the unified thresholds
                        missing_threshold=current_missing, 
                        tolerance_threshold=current_tolerance
                    )
                st.session_state.di_recalculation_needed = False # Reset flag after successful calculation
            except Exception as e:
                 st.error(f"计算扩散指数时出错: {e}")
                 st.session_state.wow_di = pd.DataFrame() # Clear results on error
                 st.session_state.yoy_di = pd.DataFrame()
                 st.session_state.mix_di = pd.DataFrame()
                 st.session_state.di_recalculation_needed = False # Reset flag even on error to avoid loop
    else:
        st.session_state.di_recalculation_needed = False
        st.session_state.wow_di = pd.DataFrame()
        st.session_state.yoy_di = pd.DataFrame()
        st.session_state.mix_di = pd.DataFrame()
        # No warning here, handled in Tab 3

# --- Main Area --- 
if not st.session_state.data_loaded:
    # st.warning("请在左侧侧边栏上传 Excel 数据文件以开始分析。") # Commented out this warning
    st.stop()

# --- Main Area with Tabs ---
tab1, tab2, tab3, tab4 = st.tabs(["周度数据", "月度数据", "扩散指数", "平稳性检验"])

# --- 周度数据分析 ---
with tab1:
    if not st.session_state.weekly_df.empty:
        # 使用 Markdown 控制标签样式
        st.markdown("**选择行业大类**")
        selected_industry_w = st.selectbox(
            "select_industry_weekly", # Internal key, label is handled by markdown
            st.session_state.industries,
            key="industry_select_weekly",
            label_visibility="collapsed" # Hide the default label
        )

        if selected_industry_w:
            # --- 显示周度摘要 --- 
            if selected_industry_w not in st.session_state.weekly_summary_cache:
                with st.spinner(f"正在计算 '{selected_industry_w}' 的周度摘要..."):
                    # 筛选属于该行业的周度指标列
                    industry_indicator_cols = [ind for ind, src in st.session_state.source_map.items()
                                             if src == selected_industry_w and ind in st.session_state.weekly_df.columns]
                    if industry_indicator_cols:
                        industry_weekly_data = st.session_state.weekly_df[industry_indicator_cols]
                        # 调用新的摘要函数
                        try:
                            summary_table = calculate_weekly_growth_summary(industry_weekly_data)
                        except Exception as e:
                             st.error(f"计算周度摘要时出错 ({selected_industry_w}): {e}")
                             summary_table = pd.DataFrame()
                        st.session_state.weekly_summary_cache[selected_industry_w] = summary_table
                    else:
                        st.session_state.weekly_summary_cache[selected_industry_w] = pd.DataFrame()
            
            summary_table = st.session_state.weekly_summary_cache[selected_industry_w]
                 
            # 使用 Markdown 控制标题字体大小 (使用 bold)
            st.markdown(f"**{selected_industry_w} - 周度数据摘要**")
            if not summary_table.empty:
                # 添加颜色样式
                def highlight_positive_negative(val):
                    try:
                        val_float = float(str(val).replace('%', ''))
                        if val_float > 0:
                            return 'background-color: #ffebee'  # 浅红色
                        elif val_float < 0:
                            return 'background-color: #e8f5e9'  # 浅绿色
                        return ''
                    except (ValueError, TypeError):
                        return ''

                # 默认按"环比上周"降序排序
                try:
                    summary_table_sorted = summary_table.copy()
                    sort_col_name = '环比上周'  # 更新排序列名
                    summary_table_sorted[f'{sort_col_name}_numeric'] = pd.to_numeric(
                        summary_table_sorted[sort_col_name].astype(str).str.replace('%', ''), errors='coerce'
                    )
                    summary_table_sorted = summary_table_sorted.sort_values(
                        by=f'{sort_col_name}_numeric', ascending=False, na_position='last'
                    ).drop(columns=[f'{sort_col_name}_numeric'])
                except KeyError:
                    st.warning(f"无法按 '{sort_col_name}' 排序周度摘要，该列不存在。")
                    summary_table_sorted = summary_table
                except Exception as e:
                    st.warning(f"按 '{sort_col_name}' 排序周度摘要时出错: {e}")
                    summary_table_sorted = summary_table

                # 应用样式并显示表格
                try:
                    highlight_cols = ['环比上周', '环比上月', '同比上年']  # 更新高亮列名
                    # 更新格式化字典以匹配新函数可能返回的列
                    styled_table = summary_table_sorted.style.format({
                        '环比上周': '{:.2%}',   # 百分比格式
                        '环比上月': '{:.2%}',   # 百分比格式
                        '同比上年': '{:.2%}',   # 百分比格式
                        '上周值': '{:.2f}',    # 保留两位小数
                        '上月值': '{:.2f}',    # 保留两位小数
                        '上年值': '{:.2f}',    # 保留两位小数
                        '近5年最大值': '{:.2f}',  # 保留两位小数
                        '近5年最小值': '{:.2f}',  # 保留两位小数
                        '近5年平均值': '{:.2f}'   # 保留两位小数
                    }).apply(lambda x: x.map(highlight_positive_negative), subset=highlight_cols)
                    # Hide index
                    st.dataframe(styled_table, hide_index=True)
                except KeyError as e:
                    st.error(f"格式化/高亮周度摘要表时出错，列名可能不匹配: {e} (需要列: {highlight_cols})")
                    st.dataframe(summary_table_sorted, hide_index=True)
                except Exception as e:
                    st.error(f"格式化/高亮周度摘要表时出错: {e}")
                    st.dataframe(summary_table_sorted, hide_index=True)

                # 显示所有指标的时间序列图
                industry_indicators = [ind for ind, src in st.session_state.source_map.items()
                                    if src == selected_industry_w and ind in st.session_state.weekly_df.columns]
                
                if not industry_indicators: 
                    st.warning(f"行业 '{selected_industry_w}' 没有可供可视化的周度指标。")
                else:
                    current_year = datetime.now().year
                    previous_year = current_year - 1
                    
                    # 计算每个指标的周环比并排序 (用于图表分列)
                    indicator_changes = {}
                    for indicator in industry_indicators:
                        indicator_series = st.session_state.weekly_df[indicator].dropna()
                        if len(indicator_series) >= 2:
                            latest_value = indicator_series.iloc[-1]
                            previous_value = indicator_series.iloc[-2]
                            if previous_value != 0:
                                try:
                                     wow_change = (latest_value - previous_value) / previous_value
                                     indicator_changes[indicator] = wow_change
                                except ZeroDivisionError:
                                     indicator_changes[indicator] = np.inf
                            else:
                                indicator_changes[indicator] = np.inf
                        else:
                            indicator_changes[indicator] = 0
                    
                    # 按周环比排序指标 (用于图表分列)
                    sorted_indicators = sorted(industry_indicators,
                                            key=lambda x: indicator_changes.get(x, 0) if pd.notna(indicator_changes.get(x, 0)) else -np.inf, # Handle NaN/inf
                                            reverse=True)
                    
                    # 创建两列布局
                    col1, col2 = st.columns(2)
                    
                    # 在第一列显示周环比为正的指标
                    with col1:
                        for indicator in sorted_indicators:
                            change = indicator_changes.get(indicator, 0)
                            if pd.notna(change) and change > 0 and change != np.inf:
                                indicator_series = st.session_state.weekly_df[indicator].dropna()
                                if not indicator_series.empty:
                                    with st.spinner(f"正在生成 {indicator} 的图表..."):
                                        historical_stats = calculate_historical_weekly_stats(indicator_series, current_year)
                                        fig = plot_weekly_indicator(
                                            indicator_series=indicator_series,
                                            historical_stats=historical_stats,
                                            indicator_name=indicator,
                                            current_year=current_year,
                                            previous_year=previous_year
                                        )
                                        st.plotly_chart(fig, use_container_width=True)
                    
                    # 在第二列显示周环比为负/零/Inf 的指标
                    with col2:
                        for indicator in sorted_indicators:
                            change = indicator_changes.get(indicator, 0)
                            if not (pd.notna(change) and change > 0 and change != np.inf):
                                indicator_series = st.session_state.weekly_df[indicator].dropna()
                                if not indicator_series.empty:
                                    with st.spinner(f"正在生成 {indicator} 的图表..."):
                                        historical_stats = calculate_historical_weekly_stats(indicator_series, current_year)
                                        fig = plot_weekly_indicator(
                                            indicator_series=indicator_series,
                                            historical_stats=historical_stats,
                                            indicator_name=indicator,
                                            current_year=current_year,
                                            previous_year=previous_year
                                        )
                                        st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning(f"未能计算或找不到 '{selected_industry_w}' 的周度指标摘要。")
        # else: # Optional block if needed when no industry is selected
            # pass 
            
    else: 
        st.warning("未加载周度数据，请先在数据加载步骤中处理。")
        
with tab2:
    if not st.session_state.monthly_df.empty:
        
        # --- 显示月度摘要 --- #
        st.markdown("**月度数据摘要**")
        try:
            monthly_summary_table = calculate_monthly_growth_summary(st.session_state.monthly_df)
        except Exception as e:
            st.error(f"计算月度摘要时出错: {e}")
            monthly_summary_table = pd.DataFrame()

        if not monthly_summary_table.empty:
            # 颜色样式函数 (保持不变)
            def highlight_monthly_positive_negative(val):
                try:
                    # Convert to float, handle potential percentage sign if format applied before styling
                    val_float = float(str(val).replace('%', '')) 
                    if val_float > 0:
                        return 'background-color: #ffebee'  # 浅红色
                    elif val_float < 0:
                        return 'background-color: #e8f5e9'  # 浅绿色
                    return ''
                except (ValueError, TypeError):
                    return '' # Return empty string for non-numeric or error cases

            # 默认按"环比上月"降序排序
            try:
                monthly_summary_sorted = monthly_summary_table.copy()
                # 确保 '环比上月' 列存在
                sort_col_name = '环比上月' # Define column name
                # Use original (unformatted) data for sorting if possible, otherwise convert formatted string
                if pd.api.types.is_numeric_dtype(monthly_summary_sorted[sort_col_name]):
                     monthly_summary_sorted[f'{sort_col_name}_numeric'] = monthly_summary_sorted[sort_col_name]
                else: # If data is already string/formatted, attempt conversion
                    monthly_summary_sorted[f'{sort_col_name}_numeric'] = pd.to_numeric(
                        monthly_summary_sorted[sort_col_name].astype(str).str.replace('%', ''), errors='coerce'
                    )

                monthly_summary_sorted = monthly_summary_sorted.sort_values(
                    by=f'{sort_col_name}_numeric', ascending=False, na_position='last'
                ).drop(columns=[f'{sort_col_name}_numeric'])
            except KeyError:
                st.warning(f"无法按 '{sort_col_name}' 排序月度摘要，该列不存在。")
                monthly_summary_sorted = monthly_summary_table
            except Exception as e:
                st.warning(f"按 '{sort_col_name}' 排序月度摘要时出错: {e}")
                monthly_summary_sorted = monthly_summary_table

            # 应用样式并显示表格
            try:
                highlight_cols = ['环比上月', '同比上年']  # 高亮列
                # 格式化字典：差值列用 '{:.2f}%'，其他数值用 '{:.2f}'
                format_dict = {col: '{:.2f}' for col in monthly_summary_sorted.select_dtypes(include=np.number).columns}
                format_dict.update({
                    '环比上月': '{:.2f}%',
                    '同比上年': '{:.2f}%'
                })

                styled_monthly_table = monthly_summary_sorted.style.format(format_dict)\
                                         .apply(lambda x: x.map(highlight_monthly_positive_negative), subset=highlight_cols)
                # Hide index
                st.dataframe(styled_monthly_table, hide_index=True)
            except KeyError as e:
                st.error(f"格式化/高亮月度摘要表时出错，列名可能不匹配: {e} (需要列: {highlight_cols})")
                st.dataframe(monthly_summary_sorted, hide_index=True)
            except Exception as e:
                st.error(f"格式化/高亮月度摘要表时出错: {e}")
                st.dataframe(monthly_summary_sorted, hide_index=True)

            st.divider()

        # --- 显示月度图表 --- #
        # st.markdown("#### 月度工业增加值同比 (%)") # Remove the title

        # 查找工业增加值指标
        try:
            value_added_indicators = [
                col for col in st.session_state.monthly_df.columns
                if "工业增加值" in col
            ]
        except Exception as e:
            st.error(f"查找工业增加值指标时出错: {e}")
            value_added_indicators = []

        if not value_added_indicators:
            st.warning("未在月度数据中找到包含'工业增加值'的指标列。")
        else:
            current_year = datetime.now().year
            previous_year = current_year - 1

            num_columns = 2
            cols = st.columns(num_columns)
            col_index = 0

            for indicator in sorted(value_added_indicators):
                indicator_series = st.session_state.monthly_df[indicator].dropna()
                if not indicator_series.empty:
                    with cols[col_index % num_columns]:
                        with st.spinner(f"正在生成 {indicator} 的月度图表..."):
                            # 提取并清理行业名称作为标题
                            try:
                                # 尝试按常见分隔符分割，取第一部分
                                parts = indicator.split('-')
                                if len(parts) > 1:
                                     indicator_title = parts[0].strip()
                                else:
                                     # 如果没有分隔符，直接使用，但清理特定词语
                                     indicator_title = indicator.strip()
                                # 进一步清理标题
                                indicator_title = indicator_title.replace("规模以上工业增加值：", "").replace("当月同比", "").strip()
                            except Exception as title_e:
                                print(f"Error cleaning title for {indicator}: {title_e}")
                                indicator_title = indicator # Fallback to full name if cleaning fails

                            fig = plot_monthly_indicator(
                                indicator_series=indicator_series,
                                indicator_name=indicator_title,
                                current_year=current_year,
                                previous_year=previous_year
                            )
                            st.plotly_chart(fig, use_container_width=True)
                    col_index += 1
                else:
                    st.warning(f"指标 '{indicator}' 没有数据。")
    else:
        st.warning("未加载月度数据。")

with tab3:
    # Check if DI data is available (calculated or loaded)
    di_available = not st.session_state.wow_di.empty or not st.session_state.yoy_di.empty or not st.session_state.mix_di.empty

    if di_available:
        # --- Main content of Tab 3 --- 
        st.markdown("**选择行业大类**") 
        selected_industry_di = st.selectbox(
            "industry_select_di_label", 
            st.session_state.industries, 
            key="industry_select_di",
            label_visibility="collapsed" 
        )
        
        st.markdown("**同比/环比扩散指数计算参数**") 

        # --- Type Correction before rendering sliders --- 
        # Ensure slider values in session state are floats before passing them to widgets
        if not isinstance(st.session_state.tolerance_threshold, (int, float)):
            warnings.warn(f"修正 tolerance_threshold 类型从 {type(st.session_state.tolerance_threshold)} 为 float.")
            if isinstance(st.session_state.tolerance_threshold, (list, tuple)) and st.session_state.tolerance_threshold:
                try: st.session_state.tolerance_threshold = float(st.session_state.tolerance_threshold[0])
                except (ValueError, TypeError): st.session_state.tolerance_threshold = DEFAULT_TOLERANCE_THRESHOLD
            else: st.session_state.tolerance_threshold = DEFAULT_TOLERANCE_THRESHOLD

        if not isinstance(st.session_state.missing_threshold, (int, float)):
             warnings.warn(f"修正 missing_threshold 类型从 {type(st.session_state.missing_threshold)} 为 float.")
             if isinstance(st.session_state.missing_threshold, (list, tuple)) and st.session_state.missing_threshold:
                 try: st.session_state.missing_threshold = float(st.session_state.missing_threshold[0])
                 except (ValueError, TypeError): st.session_state.missing_threshold = DEFAULT_MISSING_THRESHOLD
             else: st.session_state.missing_threshold = DEFAULT_MISSING_THRESHOLD
        # --- End of Type Correction --- 

        # Ensure sliders are correctly indented within the if di_available block
        col_slider1, col_slider2 = st.columns(2)
        
        with col_slider1:
            tolerance_threshold_widget_val = st.slider( 
                "增长容忍度阈值", 
                min_value=0.0, max_value=0.2, 
                value=float(st.session_state.tolerance_threshold), 
                step=0.005,
                format="%.3f", key="tolerance_slider_tab3",
                on_change=request_di_recalculation, 
                help="指标增长率超过此值才被视为正增长。影响所有三种扩散指数。"
            )
        with col_slider2:
             missing_threshold_widget_val = st.slider( 
                "缺失值比例阈值", 
                min_value=0.1, max_value=1.0, 
                value=float(st.session_state.missing_threshold), 
                step=0.05,
                format="%.2f", key="missing_slider_tab3",
                on_change=request_di_recalculation,
                help="单个日期允许的最大缺失指标比例。超过此比例，该日期的扩散指数为 NaN (统一应用于所有指数)。"
            )
            
        st.divider()

        if selected_industry_di:
            # --- Updated Plotting Function --- 
            def plot_diffusion_index(series: pd.Series, title: str, color: str):
                if series is None or series.empty: return None 
                if series.empty: return None
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=series.index, y=series, mode='lines+markers',
                    name='扩散指数', connectgaps=False,
                    line=dict(color=color), marker=dict(color=color, size=4),
                    hovertemplate="%{x|%Y-%m-%d}：%{y:.2f}<extra></extra>"
                ))
                fig.add_hline(y=50, line_dash="dash", line_color="red")
                
                # Add rangeselector buttons and adjust positioning
                fig.update_layout(
                    title=f"{selected_industry_di} - {title}", 
                    showlegend=False,
                    xaxis_title="", yaxis_title="", xaxis_tickformat='%Y-%m',
                    margin=dict(l=20, r=20, t=80, b=40), # Increased top margin further for buttons below title
                    xaxis=dict(
                        rangeselector=dict(
                            buttons=list([
                                dict(count=3, label="3m", step="month", stepmode="backward"),
                                dict(count=6, label="6m", step="month", stepmode="backward"),
                                dict(count=1, label="1y", step="year", stepmode="backward"),
                                dict(count=3, label="3y", step="year", stepmode="backward"),
                                dict(step="all", label="All")
                            ]),
                            # Positioning the buttons to the top center inside the plot area (below title)
                            x=0.5,          # Center horizontally
                            y=0.95,         # Position close to the top edge, below title (adjust 0.9 to 0.98 as needed)
                            xanchor='center', # Anchor point is the center of the buttons
                            yanchor='top',    # Anchor point is the top of the buttons
                            bgcolor='rgba(255,255,255,0.7)', 
                            font=dict(size=10) 
                        ),
                        rangeslider=dict(visible=False),
                        type="date"
                    )
                )
                return fig

            # --- Get Data and Plot --- 
            wow_series = st.session_state.wow_di.get(selected_industry_di)
            yoy_series = st.session_state.yoy_di.get(selected_industry_di)
            mix_series = st.session_state.mix_di.get(selected_industry_di)

            fig_wow = plot_diffusion_index(wow_series, "环比扩散指数", color='cornflowerblue')
            if fig_wow: st.plotly_chart(fig_wow, use_container_width=True)
            else: st.warning(f"行业 '{selected_industry_di}' 没有有效的环比扩散指数数据。")

            fig_yoy = plot_diffusion_index(yoy_series, "同比扩散指数", color='mediumseagreen')
            if fig_yoy: st.plotly_chart(fig_yoy, use_container_width=True)
            else: st.warning(f"行业 '{selected_industry_di}' 没有有效的同比扩散指数数据。")
            
            fig_mix = plot_diffusion_index(mix_series, "同环比扩散指数", color='darkorange')
            if fig_mix: st.plotly_chart(fig_mix, use_container_width=True)
            else: st.warning(f"行业 '{selected_industry_di}' 没有有效的同环比扩散指数数据。")
            
    else: # DI data is not available
        if st.session_state.data_loaded:
            st.info("正在等待或计算扩散指数... 请确保已上传有效的周度数据。")
        else:
            st.warning("请先上传数据文件以计算扩散指数。")

    # --- Add Explanation Expander for Diffusion Indices --- 
    st.divider()
    with st.expander("扩散指数计算原理"):
        st.markdown("""
        扩散指数衡量在一组指标中表现出积极变化的指标所占的比例（通常以百分比表示，范围 0-100）。高于 50 通常表示总体扩张，低于 50 表示收缩。
        
        **计算中使用的参数:**
        *   **增长容忍度阈值 (Tolerance):** 指标的增长率需要超过此阈值才被视为正增长。
        *   **缺失值比例阈值 (Missing):** 单个时间点允许的最大缺失指标比例。若超过此阈值，该时间点的扩散指数记为缺失值 (NaN)。

        --- 
        
        **1. 周环比扩散指数 (WoW DI):**
        衡量本周相较于**上一周**表现出正增长（超过容忍度阈值）的指标比例。
        *   计算每个指标的周环比增长率: `WoW_Rate = (今值 - 上周值) / |上周值|` (分母取绝对值避免负基数问题，或根据实际业务调整)。
        *   统计满足 `WoW_Rate > Tolerance` 的指标数量 `N_wow_positive`。
        *   统计当前时间点有效的指标总数 `N_valid`。
        *   `WoW DI = (N_wow_positive / N_valid) * 100` (需考虑缺失值阈值)。
        
        **2. 年同比扩散指数 (YoY DI):**
        衡量本周相较于**去年同期**表现出正增长（超过容忍度阈值）的指标比例。
        *   计算每个指标的年同比生长率: `YoY_Rate = (今值 - 去年同期值) / |去年同期值|`。
        *   统计满足 `YoY_Rate > Tolerance` 的指标数量 `N_yoy_positive`。
        *   `YoY DI = (N_yoy_positive / N_valid) * 100` (需考虑缺失值阈值)。

        **3. 同环比扩散指数 (Mix DI):**
        衡量**同时满足**周环比和年同比均为正增长（均超过容忍度阈值）的指标比例。这代表了更强的增长信号。
        *   统计同时满足 `WoW_Rate > Tolerance` **且** `YoY_Rate > Tolerance` 的指标数量 `N_mix_positive`。
        *   `Mix DI = (N_mix_positive / N_valid) * 100` (需考虑缺失值阈值)。
        
        *注意：具体的增长率计算方式和 Mix DI 的确切定义可能因实际业务逻辑而异。*
        """, unsafe_allow_html=True)

# --- 平稳性检验 --- (MODIFIED TAB)
with tab4:
    # Removed header
    
    # Use standard label with markdown attempt for bolding
    alpha_stat = st.selectbox(
        label="**选择显著性水平 (α)**", # Try markdown bolding here
        options=[0.01, 0.05, 0.10], 
        index=1, 
        key="alpha_selectbox_stat"
        # Removed label_visibility="collapsed"
    )
    
    # --- Auto-run the test if data is loaded and results are missing --- 
    # Check if results for the current alpha are missing or data is newly loaded
    results_missing = (st.session_state.get('adf_results_monthly', pd.DataFrame()).empty and 
                       st.session_state.get('adf_results_weekly', pd.DataFrame()).empty)
    # Add a check for alpha change, maybe store last alpha used?
    # For simplicity now, we run if results are empty OR if button *was* present (logic removed)
    # We need a robust way to trigger recalculation if alpha changes, 
    # but for now, let's run if results are empty after data load.
    
    # Simple trigger: Run if data loaded and results are empty
    if st.session_state.data_loaded and results_missing:
        data_found = False
        with st.spinner("正在自动执行检验与处理..."):
            # Process Monthly Data
            if not st.session_state.monthly_df.empty:
                results_m, processed_m = test_and_process_stationarity(st.session_state.monthly_df, alpha=alpha_stat)
                st.session_state.adf_results_monthly = results_m
                st.session_state.processed_monthly_df = processed_m
                data_found = True
            else:
                st.session_state.adf_results_monthly = pd.DataFrame() # Ensure it's empty if no data
                st.session_state.processed_monthly_df = pd.DataFrame()
                # st.warning("未找到月度数据进行检验。", icon="⚠️") # Less noisy without button
            
            # Process Weekly Data
            if not st.session_state.weekly_df.empty:
                results_w, processed_w = test_and_process_stationarity(st.session_state.weekly_df, alpha=alpha_stat)
                st.session_state.adf_results_weekly = results_w
                st.session_state.processed_weekly_df = processed_w
                data_found = True
            else:
                st.session_state.adf_results_weekly = pd.DataFrame() # Ensure it's empty if no data
                st.session_state.processed_weekly_df = pd.DataFrame()
                # st.warning("未找到周度数据进行检验。", icon="⚠️") # Less noisy without button
        
        # if data_found: # Less noisy without button
            # st.success("平稳性检验与处理完成！")
            
    # Display Results and Download Buttons
    st.divider()

    # --- Display Monthly Results --- 
    adf_results_monthly = st.session_state.get('adf_results_monthly', pd.DataFrame())
    processed_monthly_df = st.session_state.get('processed_monthly_df', pd.DataFrame())
    
    if not adf_results_monthly.empty:
        st.markdown('**月度数据检验和处理结果**') 
        df_to_display_m = adf_results_monthly
        
        def style_stationarity(val):
            if val == '是': return 'color: green; font-weight: bold;'
            if val == '否': return 'color: red;'
            if '数据不足' in str(val) or '测试出错' in str(val): return 'color: orange;'
            return ''
            
        st.dataframe(df_to_display_m.style.format({
            '原始P值': '{:.4f}', # Display 4 decimals in UI
            '处理后P值': '{:.4f}',
        }).applymap(style_stationarity, subset=['原始是否平稳', '最终是否平稳']), 
        hide_index=True)

        # --- Excel Download Button for Monthly Data (with P-value formatting) ---
        if not processed_monthly_df.empty:
            output_m = io.BytesIO()
            # Format P-values specifically for Excel export (3 decimals)
            adf_results_monthly_excel = adf_results_monthly.copy()
            # Handle potential non-numeric values before formatting
            p_cols_to_format = ['原始P值', '处理后P值']
            for col in p_cols_to_format:
                 if col in adf_results_monthly_excel.columns:
                     adf_results_monthly_excel[col] = pd.to_numeric(adf_results_monthly_excel[col], errors='coerce')
                     adf_results_monthly_excel[col] = adf_results_monthly_excel[col].map('{:.3f}'.format, na_action='ignore')

            with pd.ExcelWriter(output_m, engine='openpyxl') as writer:
                adf_results_monthly_excel.to_excel(writer, sheet_name='数据检验和处理结果', index=False)
                processed_monthly_df.to_excel(writer, sheet_name='处理后的数据', index=True) 
            output_m.seek(0)
            st.download_button(
                label="下载月度结果 (Excel)",
                data=output_m,
                file_name=f'stationary_monthly_results_alpha_{alpha_stat:.2f}.xlsx',
                mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
                key='download_monthly_stat_xlsx'
            )
        st.divider()
    elif st.session_state.data_loaded: # Show message if data loaded but no monthly results
        st.info("未生成月度平稳性检验结果 (可能无月度数据)。")
        st.divider()

    # --- Display Weekly Results --- 
    adf_results_weekly = st.session_state.get('adf_results_weekly', pd.DataFrame())
    processed_weekly_df = st.session_state.get('processed_weekly_df', pd.DataFrame())

    if not adf_results_weekly.empty:
        st.markdown('**周度数据检验和处理结果**')
        df_to_display_w = adf_results_weekly
        st.dataframe(df_to_display_w.style.format({
            '原始P值': '{:.4f}', # Display 4 decimals in UI
            '处理后P值': '{:.4f}',
        }).applymap(style_stationarity, subset=['原始是否平稳', '最终是否平稳']), 
        hide_index=True)
        
        # ADD a divider before the download button for better spacing
        st.divider() 

        # --- Excel Download Button for Weekly Data (with P-value formatting) ---
        if not processed_weekly_df.empty:
            output_w = io.BytesIO()
            # Format P-values specifically for Excel export (3 decimals)
            adf_results_weekly_excel = adf_results_weekly.copy()
            # Handle potential non-numeric values before formatting
            for col in p_cols_to_format: # Reuse list from monthly
                 if col in adf_results_weekly_excel.columns:
                    adf_results_weekly_excel[col] = pd.to_numeric(adf_results_weekly_excel[col], errors='coerce')
                    adf_results_weekly_excel[col] = adf_results_weekly_excel[col].map('{:.3f}'.format, na_action='ignore')
                    
            with pd.ExcelWriter(output_w, engine='openpyxl') as writer:
                adf_results_weekly_excel.to_excel(writer, sheet_name='数据检验和处理结果', index=False)
                processed_weekly_df.to_excel(writer, sheet_name='处理后的数据', index=True)
            output_w.seek(0)
            st.download_button(
                label="下载周度结果 (Excel)",
                data=output_w,
                file_name=f'stationary_weekly_results_alpha_{alpha_stat:.2f}.xlsx',
                mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
                key='download_weekly_stat_xlsx'
            )
            
    elif st.session_state.data_loaded: # Show message if data loaded but no weekly results
         st.info("未生成周度平稳性检验结果 (可能无周度数据)。")

    # --- Explanation Expander (Moved to the end, styled label) ---
    st.divider()
    with st.expander("**检验与处理原理**"):
        st.markdown("""
        使用增广迪基-福勒检验 (ADF Test) 检验时间序列的平稳性，并按顺序尝试通过以下方法使非平稳序列变得平稳：
        1.  **对数变换:** `log(x)` (仅适用于严格为正的序列)
        2.  **一阶差分:** `x(t) - x(t-1)`
        3.  **对数一阶差分:** `log(x(t)) - log(x(t-1))` (仅适用于严格为正的序列)
        4.  **二阶差分:** `(x(t) - x(t-1)) - (x(t-1) - x(t-2))`
        
        检验基于以下假设：
        - **原假设 (H0):** 序列存在单位根 (非平稳)。
        - **备择假设 (H1):** 序列不存在单位根 (平稳)。
        
        如果 P 值小于选择的显著性水平 (α)，则拒绝原假设，认为序列是平稳的。处理将停在第一个成功使序列平稳的方法上。
        """, unsafe_allow_html=True)

# (End of script) 