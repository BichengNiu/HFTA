import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.colors
from datetime import datetime
import sys
import os
import unicodedata
import re
import io

# Get the directory of the current file
diffusion_mod_dir = os.path.dirname(__file__)
dashboard_dir = os.path.abspath(os.path.join(diffusion_mod_dir, '..'))
project_root_diff = os.path.abspath(os.path.join(dashboard_dir, '..'))

if project_root_diff not in sys.path:
    sys.path.insert(0, project_root_diff)

def normalize_string(s: str) -> str:
    """将字符串转换为标准化形式：半角，去除首尾空格，合并中间空格。"""
    if not isinstance(s, str):
        return s
    # 转换全角为半角 (常见标点和空格)
    full_width = "（）：　"
    half_width = "(): "
    translation_table = str.maketrans(full_width, half_width)
    s = s.translate(translation_table)
    # 特殊处理：确保冒号后面有空格
    s = re.sub(r':(?!\s)', ': ', s)
    # 去除首尾空格
    s = s.strip()
    # 合并中间多余空格
    s = re.sub(r'\s+', ' ', s)
    return s

# === 新增：缓存机制 ===
@st.cache_data(ttl=300)  # 缓存5分钟
def cached_calculate_diffusion_index(df_hash: str, 
                                   frequency: str,
                                   comparison_type: str,
                                   tolerance_threshold: float,
                                   df_values: list) -> pd.Series:
    """
    带缓存的扩散指数计算
    """
    # 重构DataFrame
    df = pd.DataFrame(df_values[0], index=df_values[1], columns=df_values[2])
    df.index = pd.to_datetime(df.index)
    
    return calculate_diffusion_index(df, frequency, comparison_type, tolerance_threshold)

def get_dataframe_hash(df: pd.DataFrame) -> str:
    """获取DataFrame的哈希值用于缓存"""
    return str(hash(str(df.shape) + str(df.columns.tolist()) + str(df.index[-1]) if not df.empty else "empty"))

# === 新增：扩散指数计算核心函数 ===
def calculate_diffusion_index(df: pd.DataFrame, 
                            frequency: str,
                            comparison_type: str,
                            tolerance_threshold: float) -> pd.Series:
    """
    计算指定频率和比较类型的扩散指数
    
    Args:
        df: 时间序列数据DataFrame
        frequency: 时间频率 ("daily", "weekly", "monthly", "annual")
        comparison_type: 比较类型 ("同比", "环比")
        tolerance_threshold: 容忍度阈值
        
    Returns:
        扩散指数时间序列
    """
    if df.empty:
        return pd.Series(dtype=float)
    
    # 确保索引是datetime类型
    df_copy = df.copy()
    df_copy.index = pd.to_datetime(df_copy.index)
    
    # 根据频率和比较类型设置periods参数
    periods_map = {
        "daily": {
            "同比": 365,  # 日度同比：与去年同一天比较
            "环比": 1     # 日度环比：与前一天比较
        },
        "weekly": {
            "同比": 52,   # 周度同比：与去年同一周比较
            "环比": 1     # 周度环比：与前一周比较
        },
        "monthly": {
            "同比": 12,   # 月度同比：与去年同月比较
            "环比": 1     # 月度环比：与前一月比较
        },
        "annual": {
            "同比": 1,    # 年度同比：与前一年比较
            "环比": 1     # 年度环比：与前一年比较（年度情况下同比环比相同）
        }
    }
    
    periods = periods_map.get(frequency, {}).get(comparison_type, 1)
    
    # 计算百分比变化
    df_pct_change = df_copy.pct_change(periods=periods, fill_method=None) * 100
    
    if df_pct_change.empty:
        return pd.Series(dtype=float)
    
    # 计算扩散指数：超过阈值的指标数量 / 总有效指标数量
    above_threshold = (df_pct_change > tolerance_threshold).sum(axis=1)
    total_non_nan = df_pct_change.notna().sum(axis=1)
    diffusion_index = (above_threshold / total_non_nan).fillna(0) * 100
    
    return diffusion_index

def get_frequency_display_info(frequency: str, comparison_type: str) -> dict:
    """获取频率显示相关信息"""
    info_map = {
        "daily": {
            "同比": {
                "title_prefix": "日度同比",
                "line_name": "日度同比扩散指数",
                "color": "blue",
                "tolerance_range": (-10.0, 30.0, 0.0, 0.5),
                "file_suffix": "daily_yoy"
            },
            "环比": {
                "title_prefix": "日度环比", 
                "line_name": "日度环比扩散指数",
                "color": "green",
                "tolerance_range": (-5.0, 15.0, 0.0, 0.1),
                "file_suffix": "daily_wow"
            }
        },
        "weekly": {
            "同比": {
                "title_prefix": "周度同比",
                "line_name": "周度同比扩散指数",
                "color": "royalblue",
                "tolerance_range": (-25.0, 50.0, 0.0, 0.5),
                "file_suffix": "weekly_yoy"
            },
            "环比": {
                "title_prefix": "周度环比",
                "line_name": "周度环比扩散指数", 
                "color": "darkorange",
                "tolerance_range": (-10.0, 20.0, 0.0, 0.1),
                "file_suffix": "weekly_wow"
            }
        },
        "monthly": {
            "同比": {
                "title_prefix": "月度同比",
                "line_name": "月度同比扩散指数",
                "color": "purple",
                "tolerance_range": (-20.0, 40.0, 0.0, 0.5),
                "file_suffix": "monthly_yoy"
            },
            "环比": {
                "title_prefix": "月度环比",
                "line_name": "月度环比扩散指数",
                "color": "red",
                "tolerance_range": (-8.0, 15.0, 0.0, 0.2),
                "file_suffix": "monthly_wow"
            }
        },
        "annual": {
            "同比": {
                "title_prefix": "年度同比",
                "line_name": "年度同比扩散指数",
                "color": "darkgreen",
                "tolerance_range": (-15.0, 25.0, 0.0, 0.5),
                "file_suffix": "annual_yoy"
            },
            "环比": {
                "title_prefix": "年度环比",
                "line_name": "年度环比扩散指数",
                "color": "brown", 
                "tolerance_range": (-15.0, 25.0, 0.0, 0.5),
                "file_suffix": "annual_wow"
            }
        }
    }
    
    return info_map.get(frequency, {}).get(comparison_type, {
        "title_prefix": "扩散指数",
        "line_name": "扩散指数",
        "color": "gray",
        "tolerance_range": (-10.0, 20.0, 0.0, 0.1),
        "file_suffix": "diffusion"
    })

def calculate_weekly_growth_summary(df_weekly: pd.DataFrame) -> pd.DataFrame:
    """
    计算周度数据的历史值、增长率及近5年统计，并生成汇总表。
    对每个指标，使用其自身最新的有效数据日期进行计算。

    参数:
    df_weekly: pd.DataFrame
        预处理后的周度数据 DataFrame (应包含 DatetimeIndex)。

    返回:
    pd.DataFrame: 包含指标名称、最新日期及各种最新增长率的汇总表。
                 如果输入 DataFrame 为空或无效，则返回空 DataFrame。
    """
    if df_weekly is None or df_weekly.empty or not isinstance(df_weekly.index, pd.DatetimeIndex):
        print("输入的周度 DataFrame 为空或索引无效，无法计算。")
        return pd.DataFrame()

    # 确保按时间排序
    df_weekly = df_weekly.sort_index()

    print("准备历史值...")
    summary_data = []
    print("生成汇总表中 (按指标最新日期计算)...")

    for indicator in df_weekly.columns:
        indicator_series = df_weekly[indicator].dropna()

        if indicator_series.empty:
            print(f"指标 '{indicator}' 无有效数据，跳过。")
            continue

        current_date = indicator_series.index.max()
        current_value = indicator_series.loc[current_date]

        last_week_date = current_date - pd.Timedelta(weeks=1)
        last_month_date = current_date - pd.Timedelta(weeks=4)
        last_year_date = current_date - pd.Timedelta(weeks=52)

        # 修正历史值获取逻辑
        try:
            val_last_week = df_weekly.loc[last_week_date, indicator] if last_week_date in df_weekly.index else np.nan
        except (KeyError, IndexError):
            val_last_week = np.nan
            
        try:
            val_last_month = df_weekly.loc[last_month_date, indicator] if last_month_date in df_weekly.index else np.nan
        except (KeyError, IndexError):
            val_last_month = np.nan
            
        try:
            val_last_year = df_weekly.loc[last_year_date, indicator] if last_year_date in df_weekly.index else np.nan
        except (KeyError, IndexError):
            val_last_year = np.nan

        latest_wow = (current_value - val_last_week) / abs(val_last_week) if pd.notna(current_value) and pd.notna(val_last_week) and val_last_week != 0 else np.nan
        latest_moy = (current_value - val_last_month) / abs(val_last_month) if pd.notna(current_value) and pd.notna(val_last_month) and val_last_month != 0 else np.nan
        latest_yoy = (current_value - val_last_year) / abs(val_last_year) if pd.notna(current_value) and pd.notna(val_last_year) and val_last_year != 0 else np.nan

        current_year = current_date.year 
        stats_start_year = current_year - 5
        stats_end_year = current_year - 1
        stats_start_date = pd.Timestamp(f'{stats_start_year}-01-01')
        stats_end_date = pd.Timestamp(f'{stats_end_year}-12-31')

        historical_5y_data = indicator_series.loc[
            (indicator_series.index >= stats_start_date) &
            (indicator_series.index <= stats_end_date)
        ]

        if not historical_5y_data.empty:
            stat_max = historical_5y_data.max()
            stat_min = historical_5y_data.min()
            stat_mean = historical_5y_data.mean()
        else:
            stat_max, stat_min, stat_mean = np.nan, np.nan, np.nan

        summary_data.append({
            '周度指标名称': indicator,
            '最新日期': current_date.strftime('%Y-%m-%d'), 
            '最新值': current_value, 
            '上周值': val_last_week,
            '环比上周': latest_wow,
            '上月值': val_last_month, 
            '环比上月': latest_moy,  
            '上年值': val_last_year, 
            '同比上年': latest_yoy, 
            '近5年最大值': stat_max,
            '近5年最小值': stat_min,
            '近5年平均值': stat_mean
        })

    summary_df = pd.DataFrame(summary_data)
    print(f"汇总表生成完成，共 {len(summary_df)} 个指标。")

    column_order = [
        '周度指标名称', '最新日期', '最新值',
        '上周值', '环比上周',
        '上月值', '环比上月',
        '上年值', '同比上年',
        '近5年最大值', '近5年最小值', '近5年平均值'
    ]
    existing_columns = [col for col in column_order if col in summary_df.columns]
    summary_df = summary_df[existing_columns]

    return summary_df

def display_diffusion_tab(st, session_state):
    """显示增强版扩散指数分析，支持多种时间频率"""
    
    # 检查数据是否已加载
    if not session_state.get('data_loaded', False):
        st.info("请先在左侧侧边栏上传数据文件。")
        return

    # 获取可用的数据 - 确保使用原始数据而非汇总数据
    available_frequencies = []
    frequency_data_map = {}
    
    # 日度数据：使用原始日度数据
    daily_df = None
    if hasattr(session_state, 'daily_df') and not session_state.daily_df.empty:
        daily_df = session_state.daily_df
    elif hasattr(session_state, 'preview_daily_df') and not session_state.preview_daily_df.empty:
        daily_df = session_state.preview_daily_df
    
    if daily_df is not None and not daily_df.empty:
        available_frequencies.append(("日度", "daily"))
        frequency_data_map["daily"] = daily_df
        
    # 周度数据：使用原始周度数据
    weekly_df = None
    if hasattr(session_state, 'weekly_df') and not session_state.weekly_df.empty:
        weekly_df = session_state.weekly_df
    elif hasattr(session_state, 'preview_weekly_df') and not session_state.preview_weekly_df.empty:
        weekly_df = session_state.preview_weekly_df
    
    if weekly_df is not None and not weekly_df.empty:
        available_frequencies.append(("周度", "weekly"))
        frequency_data_map["weekly"] = weekly_df
        
    # 月度数据：使用原始月度数据
    monthly_df = None
    if hasattr(session_state, 'monthly_df') and not session_state.monthly_df.empty:
        monthly_df = session_state.monthly_df
    elif hasattr(session_state, 'preview_monthly_df') and not session_state.preview_monthly_df.empty:
        monthly_df = session_state.preview_monthly_df
    
    if monthly_df is not None and not monthly_df.empty:
        available_frequencies.append(("月度", "monthly"))
        frequency_data_map["monthly"] = monthly_df

    # 年度数据：使用原始年度数据
    annual_df = None
    if hasattr(session_state, 'annual_df') and not session_state.annual_df.empty:
        annual_df = session_state.annual_df
    elif hasattr(session_state, 'preview_annual_df') and not session_state.preview_annual_df.empty:
        annual_df = session_state.preview_annual_df
    
    if annual_df is not None and not annual_df.empty:
        available_frequencies.append(("年度", "annual"))
        frequency_data_map["annual"] = annual_df

    if not available_frequencies:
        st.warning("未检测到有效的时间序列数据。请确保已上传包含日度、周度或月度数据的文件。")
        return

    # 用户界面设计
    st.markdown("#### 综合分析")
    
    # 左右布局：左侧参数设置，右侧数值显示
    left_col, right_col = st.columns([1, 1])
    
    with left_col:
        st.markdown("**参数设置**")
        
        # 第一步：选择时间频率
        freq_options = [f[0] for f in available_frequencies]
        selected_freq_display = st.selectbox(
            "选择数据频率",
            freq_options,
            key="diffusion_frequency_select",
            help="不同频率将使用对应的数据进行扩散指数计算"
        )
        
        # 获取对应的频率代码
        selected_frequency = None
        for freq_display, freq_code in available_frequencies:
            if freq_display == selected_freq_display:
                selected_frequency = freq_code
                break
        
        if not selected_frequency:
            st.error("未能找到对应的频率数据")
            return
        
        # 为不同频率提供说明
        comparison_help_text = {
            "daily": "同比：与上年同一天比较；环比：与前一天比较",
            "weekly": "同比：与上年同一周比较；环比：与前一周比较", 
            "monthly": "同比：与上年同月比较；环比：与前一月比较",
            "annual": "同比：与前一年比较；环比：与前一年比较"
        }
        
        # 第二步：选择比较类型
        comparison_type = st.radio(
            "计算方式",
            ("同比", "环比"),
            key="diffusion_comparison_select",
            horizontal=True,
            help=comparison_help_text.get(selected_frequency, "选择同比或环比计算方式")
        )
        
        # 获取当前选择的数据和显示信息
        current_data = frequency_data_map[selected_frequency]
        display_info = get_frequency_display_info(selected_frequency, comparison_type)
        
        # 第三步：设置容忍度阈值
        min_val, max_val, default_val, step_val = display_info["tolerance_range"]
        tolerance_threshold = st.slider(
            f"容忍度阈值 (%)",
            min_value=min_val,
            max_value=max_val,
            value=default_val,
            step=step_val,
            format="%.1f%%",
            help=f"指标{selected_freq_display}{comparison_type}增长率超过此阈值才被视为增长",
            key=f"tolerance_{selected_frequency}_{comparison_type}"
        )
        
                        # 添加时间范围筛选
        st.markdown("**选择时间范围**")
        
        if not current_data.empty:
            # 获取数据的时间范围
            min_date = current_data.index.min().date()
            max_date = current_data.index.max().date()
            
            # 默认显示最近2年的数据
            from datetime import timedelta
            try:
                default_start_date = max_date - timedelta(days=365*2)
                if default_start_date < min_date:
                    default_start_date = min_date
            except:
                default_start_date = min_date
            
            # 分离的时间选择器
            col_start, col_sep, col_end = st.columns([1, 0.1, 1])
            
            with col_start:
                start_date = st.date_input(
                    "开始时间",
                    value=default_start_date,
                    min_value=min_date,
                    max_value=max_date,
                    key="start_date_diffusion"
                )
                
            with col_sep:
                st.markdown("<div style='text-align: center; padding-top: 25px;'>-</div>", unsafe_allow_html=True)
                
            with col_end:
                end_date = st.date_input(
                    "结束时间", 
                    value=max_date,
                    min_value=min_date,
                    max_value=max_date,
                    key="end_date_diffusion"
                )
            
            # 确保结束时间不早于开始时间
            if end_date < start_date:
                st.error("结束时间不能早于开始时间")
                return
            
            # 根据比较类型确定所需的基期长度
            if comparison_type == "同比":
                if selected_frequency == "daily":
                    base_periods = 365  # 日度同比需要365天基期
                elif selected_frequency == "weekly": 
                    base_periods = 52   # 周度同比需要52周基期
                elif selected_frequency == "monthly":
                    base_periods = 12   # 月度同比需要12个月基期
                else:  # annual
                    base_periods = 1    # 年度同比需要1年基期
            else:  # 环比
                base_periods = 1        # 环比只需要1期基期
            
            # 计算需要包含的基期开始时间
            if selected_frequency == "daily":
                base_start_date = start_date - timedelta(days=base_periods)
            elif selected_frequency == "weekly":
                base_start_date = start_date - timedelta(weeks=base_periods) 
            elif selected_frequency == "monthly":
                base_start_date = start_date - timedelta(days=base_periods*30)  # 近似
            else:  # annual
                base_start_date = start_date - timedelta(days=base_periods*365)
            
            # 确保基期开始时间不早于数据最早时间
            if base_start_date < min_date:
                base_start_date = min_date
            
            # 显示基期提示信息
            if base_start_date < start_date:
                st.info(f"为计算{comparison_type}增长率，系统将自动使用 {base_start_date} 至 {start_date} 的数据作为基期（不在图表中显示），实际展示时间为 {start_date} 至 {end_date}")
            
            # 获取包含基期的完整数据用于计算
            full_data_for_calc = current_data.loc[base_start_date:end_date]
            
            # 获取仅用于显示的数据（排除基期）
            display_data = current_data.loc[start_date:end_date]
            
            # 更新current_data为包含基期的数据（用于计算）
            current_data = full_data_for_calc
    
    # 计算和显示扩散指数结果

    try:
        # 使用缓存机制优化性能
        df_hash = get_dataframe_hash(current_data)
        df_values = [current_data.values.tolist(), current_data.index.tolist(), current_data.columns.tolist()]
        
        with st.spinner(f"正在计算{selected_freq_display}{comparison_type}扩散指数..."):
            diffusion_index_full = cached_calculate_diffusion_index(
                df_hash,
                selected_frequency, 
                comparison_type, 
                tolerance_threshold,
                df_values
            )
        
        # 如果有时间范围选择，则只显示排除基期后的扩散指数
        if 'display_data' in locals() and not display_data.empty:
            # 筛选出显示时间范围内的扩散指数
            diffusion_index = diffusion_index_full.loc[start_date:end_date]
        else:
            diffusion_index = diffusion_index_full
        
        if not diffusion_index.empty:
            # 右侧显示数值信息
            with right_col:
                st.markdown("**摘要**")
                
                # 最新值统计信息
                col1, col2 = st.columns(2)
                
                with col1:
                    st.metric("最新值", f"{diffusion_index.iloc[-1]:.1f}%")
                    st.metric("最大值", f"{diffusion_index.max():.1f}%")
                    st.metric("最小值", f"{diffusion_index.min():.1f}%")
                    
                with col2:
                    st.metric("平均值", f"{diffusion_index.mean():.1f}%")
                    # 计算趋势
                    if len(diffusion_index) >= 2:
                        latest_change = diffusion_index.iloc[-1] - diffusion_index.iloc[-2]
                        trend_direction = "上升" if latest_change > 0 else "下降" if latest_change < 0 else "持平"
                        st.metric(f"变化 ({trend_direction})", f"{latest_change:+.1f}%")
                    else:
                        st.metric("数据点", f"{len(diffusion_index)}")
            
            st.markdown("---")
            
            # 绘制扩散指数图表
            fig_diffusion = go.Figure()
            fig_diffusion.add_trace(go.Scatter(
                x=diffusion_index.index,
                y=diffusion_index,
                mode='lines+markers',
                name=display_info["line_name"],
                line=dict(color=display_info["color"], width=2),
                marker=dict(size=4),
                hovertemplate='%{x|%Y-%m-%d}<br>扩散指数: %{y:.1f}%<extra></extra>'
            ))

            # 添加50%基准线
            fig_diffusion.add_hline(
                y=50,
                line_width=1.5,
                line_dash="dash",
                line_color="grey",
                annotation_text="50%",
                annotation_position="bottom right"
            )

            # 根据频率设置时间轴格式
            xaxis_format_map = {
                "daily": {"dtick": "M1", "tickformat": "%Y-%m"},      # 月度刻度
                "weekly": {"dtick": "M1", "tickformat": "%Y-%m"},     # 月度刻度  
                "monthly": {"dtick": "M3", "tickformat": "%Y-%m"},    # 季度刻度
                "annual": {"dtick": "M12", "tickformat": "%Y"}        # 年度刻度
            }
            
            xaxis_config = xaxis_format_map.get(selected_frequency, {"dtick": "M1", "tickformat": "%Y-%m"})
            
            fig_diffusion.update_layout(
                title=f'{display_info["title_prefix"]}扩散指数 (阈值: {tolerance_threshold:.1f}%)',
                xaxis=dict(
                    dtick=xaxis_config["dtick"],
                    tickformat=xaxis_config["tickformat"],
                    tickangle=45
                ),
                yaxis_title="扩散指数 (%)",
                yaxis_range=[0, 100],
                hovermode='x unified',
                height=450,
                showlegend=False
            )

            st.plotly_chart(fig_diffusion, use_container_width=True)
            
            # 提供数据下载
            if not diffusion_index.empty:
                try:
                    output_di = io.BytesIO()
                    df_di_download = diffusion_index.reset_index()
                    df_di_download.columns = ['日期', '扩散指数(%)']
                    excel_sheet_name_di = f'{display_info["title_prefix"]}扩散指数'
                    
                    with pd.ExcelWriter(output_di, engine='openpyxl', datetime_format='yyyy-mm-dd') as writer:
                        df_di_download.to_excel(writer, sheet_name=excel_sheet_name_di, index=False)
                    output_di.seek(0)
                    
                    st.download_button(
                        label=f"下载{display_info['title_prefix']}扩散指数数据",
                        data=output_di,
                        file_name=f'diffusion_index_{display_info["file_suffix"]}_thr{tolerance_threshold:.1f}.xlsx',
                        mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
                        key=f'download_diffusion_{selected_frequency}_{comparison_type}'
                    )
                except Exception as e_download:
                    st.error(f"生成下载文件时出错: {e_download}")
                
        else:
            st.warning(f"无法计算{selected_freq_display}{comparison_type}扩散指数，可能是数据不足或时间范围不够。")
            
    except Exception as e:
        st.error(f"计算扩散指数时出错: {e}")
        import traceback
        st.error(traceback.format_exc())

    # === 热力图功能（支持所有频率） ===
    indicator_industry_map = session_state.get('indicator_industry_map') or session_state.get('preview_indicator_industry_map')
    if indicator_industry_map:
                
        industry_map = indicator_industry_map
        # 使用包含基期的完整数据进行计算，但显示时排除基期
        data_full = current_data
        
        # 如果有时间范围选择，准备用于显示的数据
        if 'display_data' in locals() and not display_data.empty:
            data_display = display_data
        else:
            data_display = current_data
        
        if not industry_map:
            st.warning("缺少行业映射信息，无法按行业分组生成热力图。")
            return

        try:
            # 根据频率设置热力图参数
            freq_display_map = {
                "daily": "日",
                "weekly": "周", 
                "monthly": "月",
                "annual": "年"
            }
            freq_display = freq_display_map.get(selected_frequency, "")
            heatmap_type_display = f'{freq_display}{comparison_type}'
            
            # 根据频率和比较类型设置计算周期
            if comparison_type == "同比":
                calc_periods_map = {
                    "daily": 365,
                    "weekly": 52,
                    "monthly": 12,
                    "annual": 1
                }
                calc_periods = calc_periods_map.get(selected_frequency, 1)
            else:  # 环比
                calc_periods = 1
            
            hover_label_prefix = f"{freq_display}{comparison_type}"
            
            # 使用包含基期的完整数据计算增长率
            data_calc = data_full.copy()
            data_calc.index = pd.to_datetime(data_calc.index)
            
            with st.spinner(f"正在计算{heatmap_type_display}增长率..."):
                df_pct_change_full = data_calc.pct_change(periods=calc_periods, fill_method=None) * 100

            # 如果有时间范围筛选，只取显示时间范围内的数据
            if 'data_display' in locals() and not data_display.empty:
                # 筛选出显示时间范围内的增长率数据
                display_start = data_display.index.min()
                display_end = data_display.index.max()
                heatmap_data = df_pct_change_full.loc[display_start:display_end]
            else:
                heatmap_data = df_pct_change_full.copy()
            
            if not heatmap_data.empty and isinstance(heatmap_data.index, pd.DatetimeIndex):
                # 根据频率设置显示的期数
                periods_to_show_map = {
                    "daily": 20,    # 显示最近20天
                    "weekly": 15,   # 显示最近15周
                    "monthly": 12,  # 显示最近12月
                    "annual": 5     # 显示最近5年
                }
                n_periods_to_show = periods_to_show_map.get(selected_frequency, 15)
                heatmap_data_recent = heatmap_data.tail(n_periods_to_show)

                if not heatmap_data_recent.empty:
                    # 转置数据（指标作为行，时间作为列）
                    heatmap_data_transposed = heatmap_data_recent.transpose()
                    heatmap_data_transposed = heatmap_data_transposed.sort_index(axis=1, ascending=False)
                    latest_date_col = heatmap_data_transposed.columns[0]

                    # 🔍 根据数据源严格过滤：只保留对应频率的原始指标
                    all_indicators = heatmap_data_transposed.index.tolist()
                    
                    # 获取原始的source_map来判断指标真实来源
                    source_map = session_state.get('source_map') or session_state.get('preview_source_map', {})
                    
                    # 根据选择的频率确定过滤条件
                    freq_filter_map = {
                        "daily": "日度",
                        "weekly": "周度", 
                        "monthly": "月度",
                        "annual": "年度"
                    }
                    target_freq_name = freq_filter_map.get(selected_frequency, "")
                    
                    genuine_indicators = []
                    other_freq_indicators = []
                    
                    for indicator in all_indicators:
                        source = source_map.get(indicator, "")
                        # 检查数据源：只保留目标频率的原始指标
                        if target_freq_name in source:
                            genuine_indicators.append(indicator)
                        else:
                            # 可能是其他频率的指标
                            other_freq_indicators.append(indicator)
                    
                    # 只保留对应频率的原始指标
                    if genuine_indicators:
                        heatmap_data_transposed = heatmap_data_transposed.loc[genuine_indicators]
                    else:
                        st.warning(f"未找到纯{selected_freq_display}指标，无法生成热力图。")
                        return
                    
                    # 添加行业信息
                    heatmap_data_transposed['行业'] = heatmap_data_transposed.index.map(
                        lambda x: industry_map.get(normalize_string(x), '未分类')
                    )
                    
                    # 按行业分组数据
                    industries = heatmap_data_transposed['行业'].unique()
                    # 过滤掉空的或无指标的行业
                    valid_industries = []
                    industry_data_dict = {}
                    
                    for industry in industries:
                        industry_data = heatmap_data_transposed[heatmap_data_transposed['行业'] == industry]
                        if not industry_data.empty and len(industry_data) > 0:
                            valid_industries.append(industry)
                            # 按最新期数值排序
                            industry_data_sorted = industry_data.sort_values(
                                by=latest_date_col, 
                                ascending=False
                            )
                            industry_data_dict[industry] = industry_data_sorted.drop(columns=['行业'])
                    
                    if not valid_industries:
                        st.warning("没有有效的行业数据用于生成热力图。")
                        return
                    
                    # 每排2个热力图的布局
                    
                    # 计算需要的行数
                    n_industries = len(valid_industries)
                    n_rows = (n_industries + 1) // 2  # 向上取整
                    
                    for row in range(n_rows):
                        cols = st.columns(2)
                        
                        for col_idx in range(2):
                            industry_idx = row * 2 + col_idx
                            if industry_idx < n_industries:
                                industry = valid_industries[industry_idx]
                                industry_data = industry_data_dict[industry]
                                
                                with cols[col_idx]:
                                    # 准备该行业的绘图数据
                                    indicator_names = industry_data.index.tolist()
                                    y_labels_industry = [name.split(' - ')[-1] if ' - ' in name else name for name in indicator_names]
                                    x_labels_industry = [col.strftime('%Y-%m-%d') for col in industry_data.columns]
                                    
                                    z_values_industry = industry_data.values
                                    z_color_values_industry = np.sign(z_values_industry)
                                    z_text_values_industry = [[f'{val:.1f}' if pd.notna(val) else '' for val in row] for row in z_values_industry]
                                    
                                    # 创建该行业的热力图
                                    fig_industry = go.Figure(data=go.Heatmap(
                                        z=z_color_values_industry,
                                        x=x_labels_industry,
                                        y=y_labels_industry,
                                        colorscale=[[0.0, 'rgb(144,238,144)'], [0.5, 'rgb(255,255,255)'], [1.0, 'rgb(255,182,193)']],
                                        zmin=-1,
                                        zmax=1,
                                        showscale=False,
                                        text=z_text_values_industry,
                                        texttemplate="%{text}",
                                        textfont={"size": 9},
                                        customdata=z_values_industry,
                                        hovertemplate="<b>%{y}</b><br><i>%{x}</i><br>" + 
                                                      f"{hover_label_prefix}: %{{customdata:.1f}}%<extra></extra>",
                                        hoverongaps=False,
                                        xgap=1,
                                        ygap=1
                                    ))

                                    # 动态调整高度
                                    chart_height = max(300, len(indicator_names) * 25 + 100)
                                    
                                    fig_industry.update_layout(
                                        title=f"{industry}",
                                        title_font_size=14,
                                        xaxis={'type': 'category', 'tickfont': {'size': 9}},
                                        xaxis_side='top',
                                        yaxis_title="",
                                        yaxis={'type': 'category', 'tickfont': {'size': 9}},
                                        height=chart_height,
                                        margin=dict(l=20, r=20, t=40, b=20)
                                    )
                                    
                                    st.plotly_chart(fig_industry, use_container_width=True)
                            else:
                                # 如果是奇数个行业，最后一个位置留空
                                with cols[col_idx]:
                                    st.empty()

                    # 热力图数据下载
                    try:
                        output = io.BytesIO()
                        
                        # 重新构建完整的数据用于下载
                        all_data_for_download = []
                        for industry in valid_industries:
                            industry_data = industry_data_dict[industry].copy()
                            industry_data['行业'] = industry
                            all_data_for_download.append(industry_data)
                        
                        if all_data_for_download:
                            df_to_download = pd.concat(all_data_for_download)
                            # 重新排列列的顺序，把行业列放在最前面
                            cols = ['行业'] + [col for col in df_to_download.columns if col != '行业']
                            df_to_download = df_to_download[cols]
                            
                            # 格式化日期列
                            date_cols = [col for col in df_to_download.columns if isinstance(df_to_download[col].dtype, type(pd.Timestamp))]
                            for col in date_cols:
                                df_to_download[col] = df_to_download[col].dt.strftime('%Y-%m-%d')

                            with pd.ExcelWriter(output, engine='openpyxl') as writer:
                                df_to_download.to_excel(writer, sheet_name=f"{heatmap_type_display}数据", index=True)
                            output.seek(0)
                            
                            st.download_button(
                                label="下载热力图数据",
                                data=output,
                                file_name=f'diffusion_heatmap_{display_info["file_suffix"]}.xlsx',
                                mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
                                key='download_diffusion_heatmap'
                            )
                    except Exception as e_download:
                        st.error(f"生成热力图下载文件时出错: {e_download}")

        except Exception as e:
            st.error(f"生成热力图时出错: {e}")
            import traceback
            st.error(traceback.format_exc())

