import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
from .plotting_utils import plot_monthly_indicator
from .growth_calculator import calculate_monthly_growth_summary

def display_monthly_tab(st, session_state):
    """显示月度数据标签页"""


    # Check for monthly_df specifically for this tab's operation
    # Try multiple sources for monthly data
    monthly_df = None
    if hasattr(session_state, 'monthly_df') and not session_state.monthly_df.empty:
        monthly_df = session_state.monthly_df
    elif hasattr(session_state, 'preview_monthly_df') and not session_state.preview_monthly_df.empty:
        monthly_df = session_state.preview_monthly_df
        # Update session_state for consistency
        session_state.monthly_df = monthly_df
    
    if monthly_df is None or monthly_df.empty:
        st.info("月度数据尚未加载或为空，请返回'数据概览'模块上传数据。") # Updated prompt
        return
    
    # Ensure session_state has the monthly_df for the rest of the function
    if not hasattr(session_state, 'monthly_df') or session_state.monthly_df.empty:
        session_state.monthly_df = monthly_df

    # Get monthly industries and map from session state
    # Try multiple sources for session state variables
    monthly_industries = (session_state.get('monthly_industries', []) or 
                         session_state.get('preview_monthly_industries', []))
    clean_industry_map = (session_state.get('clean_industry_map', {}) or 
                         session_state.get('preview_clean_industry_map', {}))
    source_map = (session_state.get('source_map', {}) or 
                 session_state.get('preview_source_map', {}))
    
    # Update session_state for consistency
    if not session_state.get('monthly_industries'):
        session_state.monthly_industries = monthly_industries
    if not session_state.get('clean_industry_map'):
        session_state.clean_industry_map = clean_industry_map  
    if not session_state.get('source_map'):
        session_state.source_map = source_map

    # --- <<< 新增：确保月度摘要缓存已计算 >>> ---
    if not hasattr(session_state, 'monthly_summary_cache'): 
        session_state.monthly_summary_cache = {}
        
    # Check if cache is empty or doesn't cover all industries (simple check for emptiness for now)
    # More robust check might be needed if industries can change dynamically without full reload
    if not session_state.monthly_summary_cache and monthly_industries:
        with st.spinner("正在计算月度行业摘要..."): 
            for industry_name_m in monthly_industries:
                # Avoid recalculating if already exists (though initial check is for empty cache)
                if industry_name_m not in session_state.monthly_summary_cache:
                    original_sources_m = clean_industry_map.get(industry_name_m, [])
                    industry_indicator_cols_m = [ind for ind, src in source_map.items()
                                               if src in original_sources_m and ind in monthly_df.columns]
                    if industry_indicator_cols_m:
                        industry_monthly_data = monthly_df[industry_indicator_cols_m]
                        try:
                            summary_table_m = calculate_monthly_growth_summary(industry_monthly_data)
                            session_state.monthly_summary_cache[industry_name_m] = summary_table_m
                        except Exception as e:
                            print(f"Error calculating monthly summary for {industry_name_m}: {e}")
                            # Store empty dataframe on error to prevent repeated attempts in the same session
                            session_state.monthly_summary_cache[industry_name_m] = pd.DataFrame()
                    else:
                         # Store empty if no indicators found
                         session_state.monthly_summary_cache[industry_name_m] = pd.DataFrame()
            print("--- Monthly summary calculation finished in monthly_data_tab.py ---")
            st.rerun() # Rerun to ensure the UI updates after calculation
    # --- <<< 结束新增 >>> ---

    # --- Start of existing Tab 2 content (Detailed Table & Charts) ---
    if not session_state.monthly_df.empty:

        # --- 显示月度摘要 (Detailed Table) --- #
        st.markdown("#### 月度数据摘要")
        try:
            # This calculation might be redundant if pre-calculation worked,
            # but keep it for now as a fallback or if specific filtering is added later.
            # Ideally, we'd fetch the already calculated full table here.
            # Let's fetch from cache if available, otherwise calculate.
            if not hasattr(session_state, 'full_monthly_summary'):
                 # Check if *any* monthly summary exists (from pre-calc)
                 if session_state.monthly_summary_cache:
                     # Attempt to rebuild full summary from cache (might be slow if many industries)
                     # This part needs careful thought - maybe just calculate the full one once?
                     # For now, let's recalculate the full one here if not cached specifically
                     print("Recalculating full monthly summary for detailed table...")
                     monthly_summary_table = calculate_monthly_growth_summary(session_state.monthly_df)
                     session_state.full_monthly_summary = monthly_summary_table # Cache it
                 else: # No cache and no full summary state -> calculate
                     print("Calculating full monthly summary for detailed table (no cache found)...")
                     monthly_summary_table = calculate_monthly_growth_summary(session_state.monthly_df)
                     session_state.full_monthly_summary = monthly_summary_table # Cache it
            else:
                 monthly_summary_table = session_state.full_monthly_summary

        except Exception as e:
            st.error(f"计算月度详细摘要时出错: {e}")
            monthly_summary_table = pd.DataFrame() # Assign empty df on error

        # 颜色样式函数 (更新颜色)
        def highlight_monthly_positive_negative(val):
            try:
                val_float = float(str(val).replace('%', ''))
                if val_float > 0:
                    return 'background-color: #ffcdd2'  # 更深的红色
                elif val_float < 0:
                    return 'background-color: #c8e6c9'  # 更深的绿色
                return ''
            except (ValueError, TypeError):
                return ''

        # Check if monthly_summary_table is not empty before proceeding
        if not monthly_summary_table.empty:
            # 默认按"环比上月"降序排序
            try:
                monthly_summary_sorted = monthly_summary_table.copy()
                # 确保 '环比上月' 列存在
                sort_col_name = '环比上月' # Define column name (FIXED)
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
                highlight_cols = ['环比上月', '同比上年']  # 高亮列 (更新列名以匹配 growth_calculator - REMOVED 差值)
                # --- MODIFIED FORMATTING ---
                # Format all numeric columns to 2 decimal places
                format_dict = {col: '{:.2f}' for col in monthly_summary_sorted.select_dtypes(include=np.number).columns}

                # No percentage columns here, so the above dict should be sufficient.
                # Ensure known columns are formatted correctly if needed (redundant but safe)
                known_numeric_cols_m = ['最新值', '上月值', '环比上月', '上年同月值', '同比上年', '近5年最大值', '近5年最小值', '近5年平均值']
                for col in known_numeric_cols_m:
                    if col in monthly_summary_sorted.columns:
                         format_dict[col] = '{:.2f}'

                styled_monthly_table = monthly_summary_sorted.style.format(format_dict)\
                                         .apply(lambda x: x.map(highlight_monthly_positive_negative), subset=highlight_cols)
                # Hide index
                st.dataframe(styled_monthly_table, hide_index=True)
                 # --- END MODIFIED FORMATTING ---
            except KeyError as e:
                st.error(f"格式化/高亮月度摘要表时出错，列名可能不匹配: {e} (需要列: {highlight_cols})")
                st.dataframe(monthly_summary_sorted, hide_index=True)
        else:
            st.info("月度详细摘要表为空或无法计算。")

        st.divider()

        # --- 显示月度图表 --- #
        st.markdown("#### 月度数据图")
        
        # 获取所有月度指标
        try: 
            monthly_indicators = list(session_state.monthly_df.columns)
        except Exception as e_find_monthly: 
            st.error(f"查找月度指标时出错: {e_find_monthly}")
            monthly_indicators = []
        
        if not monthly_indicators: 
            st.warning("未在月度数据中找到任何指标列。")
        else:
            current_year = datetime.now().year
            previous_year = current_year - 1
            
            # 根据指标数量决定每行显示的图表数
            num_indicators = len(monthly_indicators)
            if num_indicators % 2 == 0:  # 偶数个指标
                cols_per_row = 2
            else:  # 奇数个指标
                cols_per_row = 3
            
            # 创建列布局
            cols_m = st.columns(cols_per_row)
            col_idx_m = 0
            
            for indicator_m in sorted(monthly_indicators):
                indicator_series_m = session_state.monthly_df[indicator_m].dropna()
                if not indicator_series_m.empty:
                    with cols_m[col_idx_m % cols_per_row]:
                        with st.spinner(f"正在生成 {indicator_m} 的月度图表..."):
                            try:
                                # 清理标题
                                if "规模以上工业增加值:" in indicator_m:
                                    indicator_title_m = "工业增加值"
                                elif "中国:" in indicator_m and "发电量" in indicator_m:
                                    if "可再生能源" in indicator_m:
                                        indicator_title_m = "可再生能源发电量"
                                    elif "火力发电" in indicator_m:
                                        indicator_title_m = "火力发电量"
                                    else:
                                        indicator_title_m = indicator_m.replace("中国:", "").replace("(月)", "").strip()
                                else:
                                    indicator_title_m = indicator_m
                                
                                fig_m = plot_monthly_indicator(indicator_series=indicator_series_m, indicator_name=indicator_title_m, current_year=current_year, previous_year=previous_year)
                                st.plotly_chart(fig_m, use_container_width=True)
                            except Exception as e_plot_m: 
                                st.error(f"为指标 '{indicator_m}' 生成图表时出错: {e_plot_m}")
                    col_idx_m += 1
                else: 
                    st.warning(f"指标 '{indicator_m}' (月度) 没有数据。")
    else:
        st.warning("未加载月度数据。")

# Ensure necessary libraries are imported
# Example: import plotly.express as px if it were used here. 