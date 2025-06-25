import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
from .plotting_utils import plot_monthly_indicator
from .growth_calculator import calculate_monthly_growth_summary

# --- 新增：导入状态管理器 ---
try:
    from ..core.state_manager import StateManager
    from ..core.compat import CompatibilityAdapter
    from ..core.state_keys import StateKeys
    STATE_MANAGER_AVAILABLE = True
except ImportError:
    STATE_MANAGER_AVAILABLE = False
    print("[Monthly Tab] Warning: State manager not available, using legacy state management")
# --- 结束新增 ---


def get_monthly_state_manager_instance():
    """获取状态管理器实例（月度标签页专用）"""
    if STATE_MANAGER_AVAILABLE and hasattr(st.session_state, 'state_manager_initialized'):
        try:
            state_manager = StateManager(st.session_state)
            compat_adapter = CompatibilityAdapter(st.session_state)
            return state_manager, compat_adapter
        except Exception as e:
            print(f"[Monthly Tab] Error getting state manager: {e}")
            return None, None
    return None, None


def get_monthly_preview_state(key, default=None, session_state=None):
    """获取预览状态值（月度标签页兼容新旧状态管理）"""
    state_manager, compat_adapter = get_monthly_state_manager_instance()

    if compat_adapter is not None:
        return compat_adapter.get_value(key, default)
    else:
        # 回退到传统方式
        if session_state is not None:
            return getattr(session_state, key, default) if hasattr(session_state, key) else session_state.get(key, default)
        else:
            return st.session_state.get(key, default)


def set_monthly_preview_state(key, value, session_state=None):
    """设置预览状态值（月度标签页兼容新旧状态管理）"""
    state_manager, compat_adapter = get_monthly_state_manager_instance()

    if compat_adapter is not None:
        compat_adapter.set_value(key, value, use_new_key=True)
    else:
        # 回退到传统方式
        if session_state is not None:
            if hasattr(session_state, key):
                setattr(session_state, key, value)
            else:
                session_state[key] = value
        else:
            st.session_state[key] = value


def display_monthly_tab(st, session_state):
    """显示月度数据标签页"""


    # 检查月度数据是否可用（兼容新旧状态管理）
    monthly_df = get_monthly_preview_state('monthly_df', pd.DataFrame(), session_state)
    if monthly_df is None or monthly_df.empty:
        # 尝试从preview前缀的键获取
        monthly_df = get_monthly_preview_state('preview_monthly_df', pd.DataFrame(), session_state)

    if monthly_df is None or monthly_df.empty:
        st.info("月度数据尚未加载或为空，请返回'数据概览'模块上传数据。")
        return

    # 获取月度行业列表和映射（兼容新旧状态管理）
    monthly_industries = (get_monthly_preview_state('monthly_industries', [], session_state) or
                         get_monthly_preview_state('preview_monthly_industries', [], session_state))
    clean_industry_map = (get_monthly_preview_state('clean_industry_map', {}, session_state) or
                         get_monthly_preview_state('preview_clean_industry_map', {}, session_state))
    source_map = (get_monthly_preview_state('source_map', {}, session_state) or
                 get_monthly_preview_state('preview_source_map', {}, session_state))
    
    # 确保月度摘要缓存已计算（兼容新旧状态管理）
    monthly_summary_cache = (get_monthly_preview_state('monthly_summary_cache', {}, session_state) or
                            get_monthly_preview_state('preview_monthly_summary_cache', {}, session_state))

    # 检查缓存是否为空或不覆盖所有行业
    if not monthly_summary_cache and monthly_industries:
        with st.spinner("正在计算月度行业摘要..."):
            for industry_name_m in monthly_industries:
                # 避免重复计算已存在的缓存
                if industry_name_m not in monthly_summary_cache:
                    original_sources_m = clean_industry_map.get(industry_name_m, [])
                    industry_indicator_cols_m = [ind for ind, src in source_map.items()
                                               if src in original_sources_m and ind in monthly_df.columns]
                    if industry_indicator_cols_m:
                        industry_monthly_data = monthly_df[industry_indicator_cols_m]
                        try:
                            summary_table_m = calculate_monthly_growth_summary(industry_monthly_data)
                            monthly_summary_cache[industry_name_m] = summary_table_m
                        except Exception as e:
                            print(f"Error calculating monthly summary for {industry_name_m}: {e}")
                            # 存储空DataFrame以防止在同一会话中重复尝试
                            monthly_summary_cache[industry_name_m] = pd.DataFrame()
                    else:
                         # 如果没有找到指标则存储空DataFrame
                         monthly_summary_cache[industry_name_m] = pd.DataFrame()

            # 更新缓存到状态管理器
            set_monthly_preview_state('monthly_summary_cache', monthly_summary_cache, session_state)
            set_monthly_preview_state('preview_monthly_summary_cache', monthly_summary_cache, session_state)
            print("--- Monthly summary calculation finished in monthly_data_tab.py ---")
            st.rerun() # 重新运行以确保UI在计算后更新
    # --- <<< 结束新增 >>> ---

    # --- 开始现有的标签页2内容（详细表格和图表） ---
    if not monthly_df.empty:

        # --- 显示月度摘要 (Detailed Table) --- #
        st.markdown("#### 月度数据摘要")
        try:
            # 获取完整月度摘要（兼容新旧状态管理）
            full_monthly_summary = get_monthly_preview_state('full_monthly_summary', None, session_state)

            if full_monthly_summary is None:
                 # 检查是否存在任何月度摘要（来自预计算）
                 if monthly_summary_cache:
                     # 如果没有专门缓存，重新计算完整摘要
                     print("Recalculating full monthly summary for detailed table...")
                     monthly_summary_table = calculate_monthly_growth_summary(monthly_df)
                     set_monthly_preview_state('full_monthly_summary', monthly_summary_table, session_state)
                 else: # 没有缓存且没有完整摘要状态 -> 计算
                     print("Calculating full monthly summary for detailed table (no cache found)...")
                     monthly_summary_table = calculate_monthly_growth_summary(monthly_df)
                     set_monthly_preview_state('full_monthly_summary', monthly_summary_table, session_state)
            else:
                 monthly_summary_table = full_monthly_summary

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
            monthly_indicators = list(monthly_df.columns)
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
                indicator_series_m = monthly_df[indicator_m].dropna()
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