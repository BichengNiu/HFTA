import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime
from .plotting_utils import calculate_historical_weekly_stats, plot_weekly_indicator
from .growth_calculator import calculate_weekly_growth_summary

# --- 新增：导入状态管理器 ---
try:
    from ..core.state_manager import StateManager
    from ..core.compat import CompatibilityAdapter
    from ..core.state_keys import StateKeys
    STATE_MANAGER_AVAILABLE = True
except ImportError:
    STATE_MANAGER_AVAILABLE = False
    print("[Weekly Tab] Warning: State manager not available, using legacy state management")
# --- 结束新增 ---


def get_weekly_state_manager_instance():
    """获取状态管理器实例（周度标签页专用）"""
    if STATE_MANAGER_AVAILABLE and hasattr(st.session_state, 'state_manager_initialized'):
        try:
            # 尝试从全局获取状态管理器实例
            state_manager = StateManager(st.session_state)
            compat_adapter = CompatibilityAdapter(st.session_state)
            return state_manager, compat_adapter
        except Exception as e:
            print(f"[Weekly Tab] Error getting state manager: {e}")
            return None, None
    return None, None


def get_weekly_preview_state(key, default=None, session_state=None):
    """获取预览状态值（周度标签页兼容新旧状态管理）"""
    state_manager, compat_adapter = get_weekly_state_manager_instance()

    if compat_adapter is not None:
        return compat_adapter.get_value(key, default)
    else:
        # 回退到传统方式，优先使用传入的session_state
        if session_state is not None:
            return getattr(session_state, key, default) if hasattr(session_state, key) else session_state.get(key, default)
        else:
            return st.session_state.get(key, default)


def set_weekly_preview_state(key, value, session_state=None):
    """设置预览状态值（周度标签页兼容新旧状态管理）"""
    state_manager, compat_adapter = get_weekly_state_manager_instance()

    if compat_adapter is not None:
        compat_adapter.set_value(key, value, use_new_key=True)
    else:
        # 回退到传统方式，优先使用传入的session_state
        if session_state is not None:
            if hasattr(session_state, key):
                setattr(session_state, key, value)
            else:
                session_state[key] = value
        else:
            st.session_state[key] = value


def display_weekly_tab(st, session_state):
    """Displays the weekly data analysis tab."""

    # 检查周度数据是否可用（兼容新旧状态管理）
    weekly_df = get_weekly_preview_state('weekly_df', pd.DataFrame(), session_state)
    if weekly_df is None or weekly_df.empty:
        # 尝试从preview前缀的键获取
        weekly_df = get_weekly_preview_state('preview_weekly_df', pd.DataFrame(), session_state)

    if weekly_df is None or weekly_df.empty:
        st.info("周度数据尚未加载或为空，请返回'数据概览'模块上传数据。")
        return

    # 获取预填充的周度行业列表（兼容新旧状态管理）
    weekly_industries = (get_weekly_preview_state('weekly_industries', [], session_state) or
                        get_weekly_preview_state('preview_weekly_industries', [], session_state))
    clean_industry_map = (get_weekly_preview_state('clean_industry_map', {}, session_state) or
                         get_weekly_preview_state('preview_clean_industry_map', {}, session_state))

    # --- Pre-calculation still needed here if not moved to dashboard.py ---
    # --- If moved, this block can be removed --- 
    if hasattr(session_state, 'data_loaded') and session_state.data_loaded and not session_state.weekly_summary_cache and session_state.weekly_industries:
        with st.spinner("正在计算周度行业摘要..."): # Keep spinner for feedback
            for industry_name in session_state.weekly_industries:
                if industry_name not in session_state.weekly_summary_cache:
                    original_sources = session_state.clean_industry_map.get(industry_name, [])
                    industry_indicator_cols = [ind for ind, src in session_state.source_map.items()
                                             if src in original_sources and ind in session_state.weekly_df.columns]
                    if industry_indicator_cols:
                        industry_weekly_data = session_state.weekly_df[industry_indicator_cols]
                        try:
                            summary_table = calculate_weekly_growth_summary(industry_weekly_data)
                            session_state.weekly_summary_cache[industry_name] = summary_table
                        except Exception as e:
                            print(f"Error pre-calculating summary for {industry_name}: {e}")
                            session_state.weekly_summary_cache[industry_name] = pd.DataFrame()
                    else:
                         session_state.weekly_summary_cache[industry_name] = pd.DataFrame()
            print("--- Weekly pre-calculation finished in weekly_data_tab.py ---") # Debug print

    # --- Middle Section: Industry Data Analysis ---
    
    
    if not session_state.weekly_df.empty:
        # 使用 Markdown 控制标签样式
        st.markdown("**选择行业大类**")
        # Use cleaned industry names
        selected_industry_w_clean = st.selectbox(
            "select_industry_weekly", # Internal key, label is handled by markdown
            session_state.weekly_industries, # Use the cleaned list
            key="industry_select_weekly",
            label_visibility="collapsed" # Hide the default label
        )

        if selected_industry_w_clean:
            # Get the original full source name(s) corresponding to the clean name
            original_sources = session_state.clean_industry_map.get(selected_industry_w_clean, [])

            # --- 显示周度摘要 ---
            # 获取缓存（兼容新旧状态管理）
            weekly_summary_cache = (get_weekly_preview_state('weekly_summary_cache', {}, session_state) or
                                   get_weekly_preview_state('preview_weekly_summary_cache', {}, session_state))
            source_map = (get_weekly_preview_state('source_map', {}, session_state) or
                         get_weekly_preview_state('preview_source_map', {}, session_state))

            # 使用清理后的名称作为缓存键
            if selected_industry_w_clean not in weekly_summary_cache:
                with st.spinner(f"正在计算 '{selected_industry_w_clean}' 的周度摘要..."):
                    # 筛选属于该行业 (匹配任何一个原始来源) 的周度指标列
                    industry_indicator_cols = [ind for ind, src in source_map.items()
                                             if src in original_sources and ind in weekly_df.columns]
                    if industry_indicator_cols:
                        industry_weekly_data = weekly_df[industry_indicator_cols]
                        try:
                            summary_table = calculate_weekly_growth_summary(industry_weekly_data)
                        except Exception as e:
                             st.error(f"计算周度摘要时出错 ({selected_industry_w_clean}): {e}")
                             summary_table = pd.DataFrame()
                        weekly_summary_cache[selected_industry_w_clean] = summary_table
                        # 更新缓存到状态管理器
                        set_weekly_preview_state('weekly_summary_cache', weekly_summary_cache, session_state)
                        set_weekly_preview_state('preview_weekly_summary_cache', weekly_summary_cache, session_state)
                    else:
                        weekly_summary_cache[selected_industry_w_clean] = pd.DataFrame()
                        # 更新缓存到状态管理器
                        set_weekly_preview_state('weekly_summary_cache', weekly_summary_cache, session_state)
                        set_weekly_preview_state('preview_weekly_summary_cache', weekly_summary_cache, session_state)

            summary_table = weekly_summary_cache[selected_industry_w_clean]

            # 使用 Markdown 控制标题字体大小 (使用 bold)
            st.markdown(f"**{selected_industry_w_clean} - 周度数据摘要**")

            # --- Calculate and Display Summary Sentence ---
            if not summary_table.empty:
                total_indicators = len(summary_table)

                # Convert percentage columns to numeric for comparison, handling errors
                wow_numeric = pd.to_numeric(summary_table['环比上周'].astype(str).str.replace('%', ''), errors='coerce')
                mom_numeric = pd.to_numeric(summary_table['环比上月'].astype(str).str.replace('%', ''), errors='coerce')
                yoy_numeric = pd.to_numeric(summary_table['同比上年'].astype(str).str.replace('%', ''), errors='coerce')

                # Convert other relevant columns to numeric
                latest_val = pd.to_numeric(summary_table['最新值'], errors='coerce')
                max_5y = pd.to_numeric(summary_table['近5年最大值'], errors='coerce')
                min_5y = pd.to_numeric(summary_table['近5年最小值'], errors='coerce')
                mean_5y = pd.to_numeric(summary_table['近5年平均值'], errors='coerce')

                # Count based on conditions, handling NaNs
                wow_increase_count = (wow_numeric > 0).sum()
                mom_increase_count = (mom_numeric > 0).sum()
                yoy_increase_count = (yoy_numeric > 0).sum()
                above_max_count = (latest_val > max_5y).sum()
                below_min_count = (latest_val < min_5y).sum()
                above_mean_count = (latest_val > mean_5y).sum()
                below_mean_count = (latest_val < mean_5y).sum()

                # REMOVED the summary sentence generation and display
                # summary_sentence = (
                #     f"{selected_industry_w_clean}行业最新周度高频指标共{total_indicators}个，"
                #     f"有{wow_increase_count}个环比上周增长，"
                #     f"{mom_increase_count}个环比上月同期增长，"
                #     f"{yoy_increase_count}个同比去年增长，"
                #     f"高于近5年最大值{above_max_count}个，"
                #     f"低于近5年最小值{below_min_count}个，"
                #     f"高于近5年平均值{above_mean_count}个，"
                #     f"低于近5年平均值{below_mean_count}个。"
                # )
                # st.markdown(f'<p style="color:red;">{summary_sentence}</p>', unsafe_allow_html=True)
            # --- End of Summary Sentence ---

            if not summary_table.empty:
                # 添加颜色样式
                def highlight_positive_negative(val):
                    try:
                        val_float = float(str(val).replace('%', ''))
                        if val_float > 0:
                            return 'background-color: #ffcdd2'  # 更深的红色
                        elif val_float < 0:
                            return 'background-color: #c8e6c9'  # 更深的绿色
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

                # 应用样式并显示表格 - 更新格式化
                try:
                    highlight_cols = ['环比上周', '环比上月', '同比上年']
                    # --- MODIFIED FORMATTING ---
                    # First, format all numeric columns to 2 decimal places
                    format_dict = {col: '{:.2f}' for col in summary_table_sorted.select_dtypes(include=np.number).columns}

                    # Then, specifically override format for percentage columns
                    format_dict['环比上周'] = '{:.2%}'
                    format_dict['环比上月'] = '{:.2%}'
                    format_dict['同比上年'] = '{:.2%}'

                    # Explicitly format known numeric cols again to ensure consistency if they were overwritten
                    known_numeric_cols = ['最新值', '上周值', '上月值', '上年值', '近5年最大值', '近5年最小值', '近5年平均值']
                    for col in known_numeric_cols:
                        if col in summary_table_sorted.columns:
                            format_dict[col] = '{:.2f}'

                    styled_table = summary_table_sorted.style.format(format_dict)\
                                         .apply(lambda x: x.map(highlight_positive_negative), subset=highlight_cols)
                    # Hide index
                    st.dataframe(styled_table, hide_index=True)
                    # --- END MODIFIED FORMATTING ---
                except KeyError as e:
                    st.error(f"格式化/高亮周度摘要表时出错，列名可能不匹配: {e} (需要列: {highlight_cols})")

            # --- 图表显示逻辑（移出 summary_table 判断块） ---
            # 显示所有指标的时间序列图 - 更新筛选逻辑
            # 使用原始来源在source_map中找到正确的指标
            industry_indicators = [ind for ind, src in source_map.items()
                                if src in original_sources and ind in weekly_df.columns]

            if not industry_indicators:
                st.warning(f"行业 '{selected_industry_w_clean}' 没有可供可视化的周度指标。")
            else:
                current_year = datetime.now().year
                previous_year = current_year - 1

                # 计算每个指标的周环比并排序 (用于图表分列)
                indicator_changes = {}
                for indicator in industry_indicators:
                    indicator_series = weekly_df[indicator].dropna()
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
                            indicator_series = weekly_df[indicator].dropna()
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
                            indicator_series = weekly_df[indicator].dropna()
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
            # --- End of 图表显示逻辑 ---
        # else: # Optional block if needed when no industry is selected
            # pass

    else:
        st.warning("未加载周度数据，请先在数据加载步骤中处理。")

# Ensure necessary libraries are imported, potentially adding more as needed
# Example: import plotly.express as px if it were used here. 