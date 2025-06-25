import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime
from .plotting_utils import plot_daily_indicator
from .growth_calculator import calculate_daily_growth_summary

def display_daily_tab(st, session_state):
    """Displays the daily data analysis tab."""

    # Check for daily_df specifically for this tab's operation
    if not hasattr(session_state, 'daily_df') or session_state.daily_df.empty:
        st.info("日度数据尚未加载或为空，请返回'数据概览'模块上传数据。")
        return

    # Use the pre-populated daily industries list
    daily_industries = session_state.get('daily_industries', [])
    clean_industry_map = session_state.get('clean_industry_map', {})

    # --- Pre-calculation still needed here if not moved to dashboard.py ---
    # --- If moved, this block can be removed --- 
    if hasattr(session_state, 'data_loaded') and session_state.data_loaded and not session_state.daily_summary_cache and session_state.daily_industries:
        with st.spinner("正在计算日度行业摘要..."): # Keep spinner for feedback
            for industry_name in session_state.daily_industries:
                if industry_name not in session_state.daily_summary_cache:
                    original_sources = session_state.clean_industry_map.get(industry_name, [])
                    industry_indicator_cols = [ind for ind, src in session_state.source_map.items()
                                             if src in original_sources and ind in session_state.daily_df.columns]
                    if industry_indicator_cols:
                        industry_daily_data = session_state.daily_df[industry_indicator_cols]
                        try:
                            summary_table = calculate_daily_growth_summary(industry_daily_data)
                            session_state.daily_summary_cache[industry_name] = summary_table
                        except Exception as e:
                            print(f"Error pre-calculating summary for {industry_name}: {e}")
                            session_state.daily_summary_cache[industry_name] = pd.DataFrame()
                    else:
                         session_state.daily_summary_cache[industry_name] = pd.DataFrame()
            print("--- Daily pre-calculation finished in daily_data_tab.py ---") # Debug print

    # --- Middle Section: Industry Data Analysis ---
    
    
    if not session_state.daily_df.empty:
        # 使用 Markdown 控制标签样式
        st.markdown("**选择行业大类**")
        # Use cleaned industry names
        selected_industry_d_clean = st.selectbox(
            "select_industry_daily", # Internal key, label is handled by markdown
            session_state.daily_industries, # Use the cleaned list
            key="industry_select_daily",
            label_visibility="collapsed" # Hide the default label
        )

        if selected_industry_d_clean:
            # Get the original full source name(s) corresponding to the clean name
            original_sources = session_state.clean_industry_map.get(selected_industry_d_clean, [])

            # --- 显示日度摘要 ---
            # Use clean name for cache key
            if selected_industry_d_clean not in session_state.daily_summary_cache:
                with st.spinner(f"正在计算 '{selected_industry_d_clean}' 的日度摘要..."):
                    # 筛选属于该行业 (匹配任何一个原始来源) 的日度指标列
                    industry_indicator_cols = [ind for ind, src in session_state.source_map.items()
                                             if src in original_sources and ind in session_state.daily_df.columns]
                    if industry_indicator_cols:
                        industry_daily_data = session_state.daily_df[industry_indicator_cols]
                        try:
                            summary_table = calculate_daily_growth_summary(industry_daily_data)
                        except Exception as e:
                             st.error(f"计算日度摘要时出错 ({selected_industry_d_clean}): {e}")
                             summary_table = pd.DataFrame()
                        session_state.daily_summary_cache[selected_industry_d_clean] = summary_table
                    else:
                        session_state.daily_summary_cache[selected_industry_d_clean] = pd.DataFrame()

            summary_table = session_state.daily_summary_cache[selected_industry_d_clean]

            # 使用 Markdown 控制标题字体大小 (使用 bold)
            st.markdown(f"**{selected_industry_d_clean} - 日度数据摘要**")

            # --- Calculate and Display Summary Sentence ---
            if not summary_table.empty:
                total_indicators = len(summary_table)

                # Convert percentage columns to numeric for comparison, handling errors
                dod_numeric = pd.to_numeric(summary_table['环比昨日'].astype(str).str.replace('%', ''), errors='coerce')
                wow_numeric = pd.to_numeric(summary_table['环比上周'].astype(str).str.replace('%', ''), errors='coerce')
                mom_numeric = pd.to_numeric(summary_table.get('环比上月', pd.Series(dtype=float)).astype(str).str.replace('%', ''), errors='coerce')
                yoy_numeric = pd.to_numeric(summary_table['同比上年'].astype(str).str.replace('%', ''), errors='coerce')

                # Convert other relevant columns to numeric
                latest_val = pd.to_numeric(summary_table['最新值'], errors='coerce')
                max_5y = pd.to_numeric(summary_table['近5年最大值'], errors='coerce')
                min_5y = pd.to_numeric(summary_table['近5年最小值'], errors='coerce')
                mean_5y = pd.to_numeric(summary_table['近5年平均值'], errors='coerce')

                # Count based on conditions, handling NaNs
                dod_increase_count = (dod_numeric > 0).sum()
                wow_increase_count = (wow_numeric > 0).sum()
                mom_increase_count = (mom_numeric > 0).sum()
                yoy_increase_count = (yoy_numeric > 0).sum()
                above_max_count = (latest_val > max_5y).sum()
                below_min_count = (latest_val < min_5y).sum()
                above_mean_count = (latest_val > mean_5y).sum()
                below_mean_count = (latest_val < mean_5y).sum()

                # 计算占比
                dod_pct = (dod_increase_count / total_indicators * 100) if total_indicators > 0 else 0
                wow_pct = (wow_increase_count / total_indicators * 100) if total_indicators > 0 else 0
                mom_pct = (mom_increase_count / total_indicators * 100) if total_indicators > 0 else 0
                yoy_pct = (yoy_increase_count / total_indicators * 100) if total_indicators > 0 else 0
                above_max_pct = (above_max_count / total_indicators * 100) if total_indicators > 0 else 0
                below_min_pct = (below_min_count / total_indicators * 100) if total_indicators > 0 else 0
                above_mean_pct = (above_mean_count / total_indicators * 100) if total_indicators > 0 else 0
                below_mean_pct = (below_mean_count / total_indicators * 100) if total_indicators > 0 else 0

                # 生成摘要句子
                summary_sentence = (
                    f"{selected_industry_d_clean}行业最新日度高频指标共{total_indicators}个，"
                    f"有{dod_increase_count}个环比昨日增长（占比{dod_pct:.1f}%），"
                    f"{wow_increase_count}个环比上周增长（占比{wow_pct:.1f}%），"
                    f"{mom_increase_count}个环比上月增长（占比{mom_pct:.1f}%），"
                    f"{yoy_increase_count}个同比去年增长（占比{yoy_pct:.1f}%），"
                    f"高于近5年最大值{above_max_count}个（占比{above_max_pct:.1f}%），"
                    f"低于近5年最小值{below_min_count}个（占比{below_min_pct:.1f}%），"
                    f"高于近5年平均值{above_mean_count}个（占比{above_mean_pct:.1f}%），"
                    f"低于近5年平均值{below_mean_count}个（占比{below_mean_pct:.1f}%）。"
                )
                st.markdown(f'<p style="color:red;">{summary_sentence}</p>', unsafe_allow_html=True)
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

                # 默认按"环比昨日"降序排序
                try:
                    summary_table_sorted = summary_table.copy()
                    sort_col_name = '环比昨日'  # 更新排序列名
                    summary_table_sorted[f'{sort_col_name}_numeric'] = pd.to_numeric(
                        summary_table_sorted[sort_col_name].astype(str).str.replace('%', ''), errors='coerce'
                    )
                    summary_table_sorted = summary_table_sorted.sort_values(
                        by=f'{sort_col_name}_numeric', ascending=False, na_position='last'
                    ).drop(columns=[f'{sort_col_name}_numeric'])
                except KeyError:
                    st.warning(f"无法按 '{sort_col_name}' 排序日度摘要，该列不存在。")
                    summary_table_sorted = summary_table
                except Exception as e:
                    st.warning(f"按 '{sort_col_name}' 排序日度摘要时出错: {e}")
                    summary_table_sorted = summary_table

                # 应用样式并显示表格 - 更新格式化
                try:
                    highlight_cols = ['环比昨日', '环比上周', '环比上月', '同比上年']
                    # --- MODIFIED FORMATTING ---
                    # First, format all numeric columns to 2 decimal places
                    format_dict = {col: '{:.2f}' for col in summary_table_sorted.select_dtypes(include=np.number).columns}

                    # Then, specifically override format for percentage columns
                    format_dict['环比昨日'] = '{:.2%}'
                    format_dict['环比上周'] = '{:.2%}'
                    format_dict['环比上月'] = '{:.2%}'
                    format_dict['同比上年'] = '{:.2%}'

                    # Explicitly format known numeric cols again to ensure consistency if they were overwritten
                    known_numeric_cols = ['最新值', '昨日值', '上周值', '上月值', '上年值', '近5年最大值', '近5年最小值', '近5年平均值']
                    for col in known_numeric_cols:
                        if col in summary_table_sorted.columns:
                            format_dict[col] = '{:.2f}'

                    styled_table = summary_table_sorted.style.format(format_dict)\
                                         .apply(lambda x: x.map(highlight_positive_negative), subset=highlight_cols)
                    # Hide index
                    st.dataframe(styled_table, hide_index=True)
                    # --- END MODIFIED FORMATTING ---
                except KeyError as e:
                    st.error(f"格式化/高亮日度摘要表时出错，列名可能不匹配: {e} (需要列: {highlight_cols})")

            # --- 图表显示逻辑（移出 summary_table 判断块） ---
            # 显示所有指标的时间序列图 - 更新筛选逻辑
            # Use original_sources to find the correct indicators in source_map
            industry_indicators = [ind for ind, src in session_state.source_map.items()
                                if src in original_sources and ind in session_state.daily_df.columns]

            if not industry_indicators:
                st.warning(f"行业 '{selected_industry_d_clean}' 没有可供可视化的日度指标。")
            else:
                current_year = datetime.now().year
                previous_year = current_year - 1

                # 计算每个指标的日环比并排序 (用于图表分列)
                indicator_changes = {}
                for indicator in industry_indicators:
                    indicator_series = session_state.daily_df[indicator].dropna()
                    if len(indicator_series) >= 2:
                        latest_value = indicator_series.iloc[-1]
                        previous_value = indicator_series.iloc[-2]
                        if previous_value != 0:
                            try:
                                 dod_change = (latest_value - previous_value) / previous_value
                                 indicator_changes[indicator] = dod_change
                            except ZeroDivisionError:
                                 indicator_changes[indicator] = np.inf
                        else:
                            indicator_changes[indicator] = np.inf
                    else:
                        indicator_changes[indicator] = 0

                # 按日环比排序指标 (用于图表分列)
                sorted_indicators = sorted(industry_indicators,
                                        key=lambda x: indicator_changes.get(x, 0) if pd.notna(indicator_changes.get(x, 0)) else -np.inf, # Handle NaN/inf
                                        reverse=True)

                # 创建两列布局
                col1, col2 = st.columns(2)

                # 在第一列显示日环比为正的指标
                with col1:
                    for indicator in sorted_indicators:
                        change = indicator_changes.get(indicator, 0)
                        if pd.notna(change) and change > 0 and change != np.inf:
                            indicator_series = session_state.daily_df[indicator].dropna()
                            if not indicator_series.empty:
                                with st.spinner(f"正在生成 {indicator} 的图表..."):
                                    fig = plot_daily_indicator(
                                        indicator_series=indicator_series,
                                        indicator_name=indicator,
                                        current_year=current_year,
                                        previous_year=previous_year
                                    )
                                    st.plotly_chart(fig, use_container_width=True)

                # 在第二列显示日环比为负/零/Inf 的指标
                with col2:
                    for indicator in sorted_indicators:
                        change = indicator_changes.get(indicator, 0)
                        if not (pd.notna(change) and change > 0 and change != np.inf):
                            indicator_series = session_state.daily_df[indicator].dropna()
                            if not indicator_series.empty:
                                with st.spinner(f"正在生成 {indicator} 的图表..."):
                                    fig = plot_daily_indicator(
                                        indicator_series=indicator_series,
                                        indicator_name=indicator,
                                        current_year=current_year,
                                        previous_year=previous_year
                                    )
                                    st.plotly_chart(fig, use_container_width=True)
            # --- End of 图表显示逻辑 ---
        # else: # Optional block if needed when no industry is selected
            # pass

    else:
        st.warning("未加载日度数据，请先在数据加载步骤中处理。")

# Ensure necessary libraries are imported, potentially adding more as needed
# Example: import plotly.express as px if it were used here. 