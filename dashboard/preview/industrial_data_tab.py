import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
import re

# 从当前目录或子目录导入模块
from .data_loader import load_and_process_data
from .diffusion_analysis import display_diffusion_tab
from .weekly_data_tab import display_weekly_tab
from .monthly_data_tab import display_monthly_tab
from .daily_data_tab import display_daily_tab
from .growth_calculator import calculate_weekly_growth_summary, calculate_monthly_growth_summary, calculate_daily_growth_summary

# --- 新增：导入状态管理器 ---
try:
    from ..core.state_manager import StateManager
    from ..core.compat import CompatibilityAdapter
    from ..core.state_keys import StateKeys
    STATE_MANAGER_AVAILABLE = True
except ImportError:
    STATE_MANAGER_AVAILABLE = False
    print("[Preview] Warning: State manager not available, using legacy state management")
# --- 结束新增 ---


def get_state_manager_instance():
    """获取状态管理器实例"""
    if STATE_MANAGER_AVAILABLE and hasattr(st.session_state, 'state_manager_initialized'):
        try:
            # 尝试从全局获取状态管理器实例
            state_manager = StateManager(st.session_state)
            compat_adapter = CompatibilityAdapter(st.session_state)
            return state_manager, compat_adapter
        except Exception as e:
            print(f"[Preview] Error getting state manager: {e}")
            return None, None
    return None, None


def clear_preview_data_with_state_manager():
    """使用状态管理器清理预览数据"""
    state_manager, compat_adapter = get_state_manager_instance()

    if state_manager is not None:
        # 使用状态管理器清理预览模块状态
        state_manager.clear_module_state('preview', exclude_shared=False)
        print("[Preview] Data cleared using state manager")
    else:
        # 回退到传统清理方式
        keys_to_clear = [
            # 主要数据
            'preview_data_loaded_files', 'preview_weekly_df', 'preview_monthly_df', 'preview_daily_df',
            'preview_source_map', 'preview_indicator_industry_map', 'preview_weekly_industries',
            'preview_monthly_industries', 'preview_daily_industries', 'preview_clean_industry_map',
            'preview_weekly_summary_cache', 'preview_monthly_summary_cache', 'preview_daily_summary_cache',
            'preview_monthly_growth_summary_df',
            # 扩散指数功能相关数据（向后兼容）
            'data_loaded', 'weekly_df', 'monthly_df', 'daily_df', 'annual_df',
            'source_map', 'indicator_industry_map', 'weekly_industries', 'monthly_industries',
            'clean_industry_map', 'weekly_summary_cache', 'monthly_summary_cache'
        ]
        for key in keys_to_clear:
            if key in st.session_state:
                del st.session_state[key]
        print("[Preview] Data cleared using legacy method")


def get_preview_state(key, default=None):
    """获取预览状态值（兼容新旧状态管理）"""
    state_manager, compat_adapter = get_state_manager_instance()

    if compat_adapter is not None:
        return compat_adapter.get_value(key, default)
    else:
        return st.session_state.get(key, default)


def set_preview_state(key, value):
    """设置预览状态值（兼容新旧状态管理）"""
    state_manager, compat_adapter = get_state_manager_instance()

    if compat_adapter is not None:
        compat_adapter.set_value(key, value, use_new_key=True)
    else:
        st.session_state[key] = value


def display_industrial_tabs(st_session_state, extract_industry_name_func):
    # === 侧边栏：文件上传功能 ===
    with st.sidebar:
        st.markdown("### 上传工业数据文件")
        
        # 文件上传器
        uploaded_industrial_file = st.file_uploader(
            "拖放或选择 Excel 数据文件", 
            type=["xlsx"], 
            accept_multiple_files=False,
            key="industrial_data_uploader_sidebar",
            help="支持上传 Excel 文件，请确保文件包含周度、月度或日度数据表"
        )
        
        # 显示已上传文件
        if uploaded_industrial_file:
            st.markdown("**已上传文件：**")
            st.markdown(f"{uploaded_industrial_file.name}")
                
        # 数据状态面板
        if get_preview_state('preview_data_loaded_files'):
            st.markdown("---")
            st.markdown("**数据状态：**")

            # 显示指标数量
            weekly_df = get_preview_state('preview_weekly_df', pd.DataFrame())
            monthly_df = get_preview_state('preview_monthly_df', pd.DataFrame())
            daily_df = get_preview_state('preview_daily_df', pd.DataFrame())

            weekly_count = len(weekly_df.columns) if not weekly_df.empty else 0
            monthly_count = len(monthly_df.columns) if not monthly_df.empty else 0
            daily_count = len(daily_df.columns) if not daily_df.empty else 0

            if weekly_count > 0:
                st.markdown(f"周度指标：{weekly_count} 个")
            if monthly_count > 0:
                st.markdown(f"月度指标：{monthly_count} 个")
            if daily_count > 0:
                st.markdown(f"日度指标：{daily_count} 个")

            # 显示行业数量
            weekly_industries = len(get_preview_state('preview_weekly_industries', []))
            monthly_industries = len(get_preview_state('preview_monthly_industries', []))
            if weekly_industries > 0 or monthly_industries > 0:
                st.markdown(f"涵盖行业：{max(weekly_industries, monthly_industries)} 个")
                

        
        else:
            st.markdown("---")
            st.markdown("**使用说明：**")
            st.markdown("• 支持 Excel (.xlsx) 格式文件")
            st.markdown("• 上传单个数据文件")
            st.markdown("• 文件应包含时间序列数据")
            st.markdown("• 支持周度、月度、日度数据")
            
    # === 主区域：数据处理和显示 ===

    # 检测文件是否被删除（用户点击了叉号）
    if not uploaded_industrial_file and get_preview_state('preview_data_loaded_files'):
        # 文件被删除，使用状态管理器清空所有相关数据
        clear_preview_data_with_state_manager()
        st.success("数据已清空！")
        st.rerun()

    if uploaded_industrial_file:
        uploaded_file_name = uploaded_industrial_file.name
        if get_preview_state('preview_data_loaded_files') != uploaded_file_name:
            with st.spinner('正在加载和处理工业数据...'):
                try:
                    weekly_df, monthly_df, daily_df, source_map, indicator_map = load_and_process_data([uploaded_industrial_file])

                    # 使用状态管理器设置数据
                    set_preview_state('preview_weekly_df', weekly_df)
                    set_preview_state('preview_monthly_df', monthly_df)
                    set_preview_state('preview_daily_df', daily_df)
                    set_preview_state('preview_source_map', source_map)
                    set_preview_state('preview_indicator_industry_map', indicator_map)
                    set_preview_state('preview_data_loaded_files', uploaded_file_name)

                    # 处理行业分类数据
                    weekly_clean_industries, monthly_clean_industries, daily_clean_industries, clean_map = set(), set(), set(), {}
                    if source_map:
                        for indicator, full_source in source_map.items():
                            clean_name = extract_industry_name_func(full_source)
                            if clean_name not in clean_map: clean_map[clean_name] = []
                            if full_source not in clean_map[clean_name]: clean_map[clean_name].append(full_source)
                            if indicator in weekly_df.columns: weekly_clean_industries.add(clean_name)
                            if indicator in monthly_df.columns: monthly_clean_industries.add(clean_name)
                            if indicator in daily_df.columns: daily_clean_industries.add(clean_name)

                    # 使用状态管理器设置行业数据
                    set_preview_state('preview_weekly_industries', sorted(list(weekly_clean_industries)))
                    set_preview_state('preview_monthly_industries', sorted(list(monthly_clean_industries)))
                    set_preview_state('preview_daily_industries', sorted(list(daily_clean_industries)))
                    set_preview_state('preview_clean_industry_map', clean_map)

                    # 初始化缓存
                    set_preview_state('preview_weekly_summary_cache', {})
                    set_preview_state('preview_monthly_summary_cache', {})
                    set_preview_state('preview_daily_summary_cache', {})
                    
                    # 计算月度摘要
                    if not monthly_df.empty:
                         set_preview_state('preview_monthly_growth_summary_df', calculate_monthly_growth_summary(monthly_df))
                         monthly_industries_list = get_preview_state('preview_monthly_industries', [])
                         current_monthly_df = get_preview_state('preview_monthly_df', pd.DataFrame())
                         current_clean_industry_map_m = get_preview_state('preview_clean_industry_map', {})
                         current_source_map_m = get_preview_state('preview_source_map', {})
                         temp_monthly_cache = {}
                         if monthly_industries_list:
                            print(f"[Industrial Tab] Populating monthly summary cache for {len(monthly_industries_list)} industries...")
                            for industry_name_m in monthly_industries_list:
                                original_sources_m = current_clean_industry_map_m.get(industry_name_m, [])
                                industry_indicator_cols_m = [ind for ind, src in current_source_map_m.items()
                                                           if src in original_sources_m and ind in current_monthly_df.columns]
                                if industry_indicator_cols_m:
                                    industry_monthly_data = current_monthly_df[industry_indicator_cols_m]
                                    try:
                                        summary_table_m = calculate_monthly_growth_summary(industry_monthly_data)
                                        temp_monthly_cache[industry_name_m] = summary_table_m
                                    except Exception as e_m_cache:
                                        print(f"Error calculating monthly summary for cache ({industry_name_m}): {e_m_cache}")
                                        temp_monthly_cache[industry_name_m] = pd.DataFrame()
                                else:
                                    temp_monthly_cache[industry_name_m] = pd.DataFrame()
                            set_preview_state('preview_monthly_summary_cache', temp_monthly_cache)
                            print(f"[Industrial Tab] Monthly summary cache populated with {len(temp_monthly_cache)} entries.")

                    # 预计算周度行业摘要缓存
                    if not weekly_df.empty:
                        weekly_industries_list_for_cache = get_preview_state('preview_weekly_industries', [])
                        current_weekly_df_for_cache = get_preview_state('preview_weekly_df', pd.DataFrame())
                        current_clean_industry_map_w = get_preview_state('preview_clean_industry_map', {})
                        current_source_map_w = get_preview_state('preview_source_map', {})
                        temp_weekly_cache = {}
                        if weekly_industries_list_for_cache:
                            print(f"[Industrial Tab] Pre-calculating weekly summary cache for {len(weekly_industries_list_for_cache)} industries...")
                            with st.spinner("正在计算周度行业摘要以备总览..."):
                                for industry_name_w_cache in weekly_industries_list_for_cache:
                                    original_sources_w = current_clean_industry_map_w.get(industry_name_w_cache, [])
                                    industry_indicator_cols_w = [ind for ind, src in current_source_map_w.items()
                                                               if src in original_sources_w and ind in current_weekly_df_for_cache.columns]
                                    if industry_indicator_cols_w:
                                        industry_weekly_data_cache = current_weekly_df_for_cache[industry_indicator_cols_w]
                                        try:
                                            summary_table_w_cache = calculate_weekly_growth_summary(industry_weekly_data_cache)
                                            temp_weekly_cache[industry_name_w_cache] = summary_table_w_cache
                                        except Exception as e_w_cache_calc:
                                            print(f"Error pre-calculating weekly summary for cache ({industry_name_w_cache}): {e_w_cache_calc}")
                                            temp_weekly_cache[industry_name_w_cache] = pd.DataFrame()
                                    else:
                                        temp_weekly_cache[industry_name_w_cache] = pd.DataFrame()
                                set_preview_state('preview_weekly_summary_cache', temp_weekly_cache)
                                print(f"[Industrial Tab] Weekly summary cache pre-populated with {len(temp_weekly_cache)} entries.")

                    # 预计算日度行业摘要缓存
                    if not daily_df.empty:
                        daily_industries_list_for_cache = get_preview_state('preview_daily_industries', [])
                        current_daily_df_for_cache = get_preview_state('preview_daily_df', pd.DataFrame())
                        current_clean_industry_map_d = get_preview_state('preview_clean_industry_map', {})
                        current_source_map_d = get_preview_state('preview_source_map', {})
                        temp_daily_cache = {}
                        if daily_industries_list_for_cache:
                            print(f"[Industrial Tab] Pre-calculating daily summary cache for {len(daily_industries_list_for_cache)} industries...")
                            with st.spinner("正在计算日度行业摘要以备总览..."):
                                for industry_name_d_cache in daily_industries_list_for_cache:
                                    original_sources_d = current_clean_industry_map_d.get(industry_name_d_cache, [])
                                    industry_indicator_cols_d = [ind for ind, src in current_source_map_d.items()
                                                               if src in original_sources_d and ind in current_daily_df_for_cache.columns]
                                    if industry_indicator_cols_d:
                                        industry_daily_data_cache = current_daily_df_for_cache[industry_indicator_cols_d]
                                        try:
                                            summary_table_d_cache = calculate_daily_growth_summary(industry_daily_data_cache)
                                            temp_daily_cache[industry_name_d_cache] = summary_table_d_cache
                                        except Exception as e_d_cache_calc:
                                            print(f"Error pre-calculating daily summary for cache ({industry_name_d_cache}): {e_d_cache_calc}")
                                            temp_daily_cache[industry_name_d_cache] = pd.DataFrame()
                                    else:
                                        temp_daily_cache[industry_name_d_cache] = pd.DataFrame()
                                set_preview_state('preview_daily_summary_cache', temp_daily_cache)
                                print(f"[Industrial Tab] Daily summary cache pre-populated with {len(temp_daily_cache)} entries.")

                    st.success("工业数据加载和预处理完成！")
                    print("[Industrial Tab] Data loaded and processed.")
                except Exception as e:
                    st.error(f"处理工业数据文件时出错: {e}")
                    print(f"[Industrial Tab] Error processing industrial data files: {e}")
                    # 使用状态管理器清理错误状态
                    clear_preview_data_with_state_manager()

    # 确保缓存已初始化
    if get_preview_state('preview_data_loaded_files') and not get_preview_state('preview_weekly_df', pd.DataFrame()).empty:
        # 确保 preview_weekly_summary_cache 已初始化为空字典
        if not get_preview_state('preview_weekly_summary_cache'):
            set_preview_state('preview_weekly_summary_cache', {})

    data_is_ready = get_preview_state('preview_data_loaded_files') is not None
    
    # 显示分析标签页
    tab_titles = ["摘要", "日度数据", "周度数据", "月度数据", "年度数据"]
    tab_overall, tab_daily, tab_weekly, tab_monthly, tab_yearly = st.tabs(tab_titles)

    with tab_overall:
       
        if not data_is_ready or (get_preview_state('preview_weekly_df', pd.DataFrame()).empty and get_preview_state('preview_monthly_df', pd.DataFrame()).empty):
            st.info("请先上传数据文件以查看总体概览。")
        else:
            # --- 日度数据总览 ---
            st.markdown("#### 日度数据总览")
            daily_summary_cache_for_overview = get_preview_state('preview_daily_summary_cache', {})
            preview_daily_df_exists = not get_preview_state('preview_daily_df', pd.DataFrame()).empty

            if preview_daily_df_exists and daily_summary_cache_for_overview:
                overall_total_d, overall_dod_d, overall_wow_d, overall_mom_d_from_daily, overall_yoy_d = 0, 0, 0, 0, 0
                overall_above_max_d, overall_below_min_d, overall_above_mean_d, overall_below_mean_d = 0, 0, 0, 0
                industry_summary_list_d = []
                daily_cache_valid_data_found = False

                for industry_name_d, summary_table_d in daily_summary_cache_for_overview.items():
                    if summary_table_d is not None and not summary_table_d.empty:
                        daily_cache_valid_data_found = True
                        try:
                            total_indicators_d = len(summary_table_d)
                            dod_numeric_d = pd.to_numeric(summary_table_d['环比昨日'].astype(str).str.rstrip('%').replace('nan', np.nan), errors='coerce') / 100.0
                            wow_numeric_d = pd.to_numeric(summary_table_d['环比上周'].astype(str).str.rstrip('%').replace('nan', np.nan), errors='coerce') / 100.0
                            mom_numeric_d = pd.to_numeric(summary_table_d.get('环比上月', pd.Series(dtype=float)).astype(str).str.rstrip('%').replace('nan', np.nan), errors='coerce') / 100.0 
                            yoy_numeric_d = pd.to_numeric(summary_table_d['同比上年'].astype(str).str.rstrip('%').replace('nan', np.nan), errors='coerce') / 100.0
                            latest_val_d = pd.to_numeric(summary_table_d['最新值'], errors='coerce')
                            max_5y_d = pd.to_numeric(summary_table_d['近5年最大值'], errors='coerce')
                            min_5y_d = pd.to_numeric(summary_table_d['近5年最小值'], errors='coerce')
                            mean_5y_d = pd.to_numeric(summary_table_d['近5年平均值'], errors='coerce')

                            dod_increase_count_d = (dod_numeric_d > 0).sum()
                            wow_increase_count_d = (wow_numeric_d > 0).sum()
                            mom_increase_count_d_from_daily = (mom_numeric_d > 0).sum()
                            yoy_increase_count_d = (yoy_numeric_d > 0).sum()
                            above_max_count_d = (latest_val_d > max_5y_d).sum()
                            below_min_count_d = (latest_val_d < min_5y_d).sum()
                            above_mean_count_d = (latest_val_d > mean_5y_d).sum()
                            below_mean_count_d = (latest_val_d < mean_5y_d).sum()

                            overall_total_d += total_indicators_d
                            overall_dod_d += dod_increase_count_d
                            overall_wow_d += wow_increase_count_d
                            overall_mom_d_from_daily += mom_increase_count_d_from_daily
                            overall_yoy_d += yoy_increase_count_d
                            overall_above_max_d += above_max_count_d; overall_below_min_d += below_min_count_d
                            overall_above_mean_d += above_mean_count_d; overall_below_mean_d += below_mean_count_d
                            
                            industry_summary_list_d.append({
                                '行业名称': industry_name_d, '指标总数': total_indicators_d,
                                '环比昨日增长': dod_increase_count_d,
                                '环比上周增长': wow_increase_count_d, 
                                '环比上月增长': mom_increase_count_d_from_daily, 
                                '同比去年增长': yoy_increase_count_d,
                                '高于近5年最大值': above_max_count_d, '低于近5年最小值': below_min_count_d,
                                '高于近5年平均值': above_mean_count_d, '低于近5年平均值': below_mean_count_d
                            })
                        except KeyError as e_d_key: print(f"Skipping daily overview for '{industry_name_d}' (KeyError): {e_d_key}")
                        except Exception as e_d_sum: print(f"Error in daily overview summary for '{industry_name_d}': {e_d_sum}")
                
                if daily_cache_valid_data_found and overall_total_d > 0:
                    # 计算占比
                    dod_pct = (overall_dod_d / overall_total_d * 100) if overall_total_d > 0 else 0
                    wow_pct = (overall_wow_d / overall_total_d * 100) if overall_total_d > 0 else 0
                    mom_pct = (overall_mom_d_from_daily / overall_total_d * 100) if overall_total_d > 0 else 0
                    yoy_pct = (overall_yoy_d / overall_total_d * 100) if overall_total_d > 0 else 0
                    above_max_pct = (overall_above_max_d / overall_total_d * 100) if overall_total_d > 0 else 0
                    below_min_pct = (overall_below_min_d / overall_total_d * 100) if overall_total_d > 0 else 0
                    above_mean_pct = (overall_above_mean_d / overall_total_d * 100) if overall_total_d > 0 else 0
                    below_mean_pct = (overall_below_mean_d / overall_total_d * 100) if overall_total_d > 0 else 0
                    
                    st.markdown(f"所有行业最新日度高频指标共 **{overall_total_d}** 个，"
                                f"其中 **{overall_dod_d}** 个环比昨日增长（占比{dod_pct:.1f}%），"
                                f"**{overall_wow_d}** 个环比上周增长（占比{wow_pct:.1f}%），"
                                f"**{overall_mom_d_from_daily}** 个环比上月增长（占比{mom_pct:.1f}%），"
                                f"**{overall_yoy_d}** 个同比去年增长（占比{yoy_pct:.1f}%）。 "
                                f"与近5年历史比， **{overall_above_max_d}** 个高于最大值（占比{above_max_pct:.1f}%），"
                                f"**{overall_below_min_d}** 个低于最小值（占比{below_min_pct:.1f}%），"
                                f"**{overall_above_mean_d}** 个高于平均值（占比{above_mean_pct:.1f}%），"
                                f"**{overall_below_mean_d}** 个低于平均值（占比{below_mean_pct:.1f}%）。")
                    if industry_summary_list_d:
                        industry_summary_df_d = pd.DataFrame(industry_summary_list_d)
                        
                        # --- BEGIN ADDED CODE FOR DAILY TOTAL ROW ---
                        numeric_cols_for_total_d = [
                            '指标总数', '环比昨日增长', '环比上周增长', '环比上月增长', '同比去年增长',
                            '高于近5年最大值', '低于近5年最小值', '高于近5年平均值', '低于近5年平均值'
                        ]
                        cols_present_to_sum_d = [col for col in numeric_cols_for_total_d if col in industry_summary_df_d.columns]

                        if cols_present_to_sum_d:
                            sums_d = industry_summary_df_d[cols_present_to_sum_d].sum()
                            total_row_dict_d = {'行业名称': '合计'}
                            total_row_dict_d.update(sums_d)
                            total_row_df_d = pd.DataFrame([total_row_dict_d])
                            industry_summary_df_d = pd.concat([industry_summary_df_d, total_row_df_d], ignore_index=True)
                        # --- END ADDED CODE FOR DAILY TOTAL ROW ---

                        cols_to_show_d = ['行业名称', '指标总数', '环比昨日增长', '环比上周增长', '环比上月增长', '同比去年增长', '高于近5年最大值', '低于近5年最小值', '高于近5年平均值', '低于近5年平均值']
                        st.dataframe(industry_summary_df_d[[col for col in cols_to_show_d if col in industry_summary_df_d.columns]], hide_index=True)
                    else: st.info("未能生成日度行业汇总表。")
                elif not preview_daily_df_exists: st.info("日度数据未加载，无法显示日度总览。")
                else: st.info("没有可用的日度行业摘要数据(缓存为空、处理错误或无日度数据)。")
            elif not preview_daily_df_exists: st.info("日度数据未加载，无法显示日度总览。")
            else: st.info("日度行业摘要缓存未初始化或为空。")
            st.markdown("---")

            # --- 周度数据总览 ---
            st.markdown("#### 周度数据总览")
            weekly_summary_cache_for_overview = get_preview_state('preview_weekly_summary_cache', {})
            preview_weekly_df_exists = not get_preview_state('preview_weekly_df', pd.DataFrame()).empty

            if preview_weekly_df_exists and weekly_summary_cache_for_overview:
                overall_total_w, overall_wow_w, overall_mom_w_from_weekly, overall_yoy_w = 0, 0, 0, 0
                overall_above_max_w, overall_below_min_w, overall_above_mean_w, overall_below_mean_w = 0, 0, 0, 0
                industry_summary_list_w = []
                weekly_cache_valid_data_found = False

                for industry_name_w, summary_table_w in weekly_summary_cache_for_overview.items():
                    if summary_table_w is not None and not summary_table_w.empty:
                        weekly_cache_valid_data_found = True
                        try:
                            total_indicators_w = len(summary_table_w)
                            wow_numeric_w = pd.to_numeric(summary_table_w['环比上周'].astype(str).str.rstrip('%').replace('nan', np.nan), errors='coerce') / 100.0
                            mom_numeric_w = pd.to_numeric(summary_table_w.get('环比上月', pd.Series(dtype=float)).astype(str).str.rstrip('%').replace('nan', np.nan), errors='coerce') / 100.0 
                            yoy_numeric_w = pd.to_numeric(summary_table_w['同比上年'].astype(str).str.rstrip('%').replace('nan', np.nan), errors='coerce') / 100.0
                            latest_val_w = pd.to_numeric(summary_table_w['最新值'], errors='coerce')
                            max_5y_w = pd.to_numeric(summary_table_w['近5年最大值'], errors='coerce')
                            min_5y_w = pd.to_numeric(summary_table_w['近5年最小值'], errors='coerce')
                            mean_5y_w = pd.to_numeric(summary_table_w['近5年平均值'], errors='coerce')

                            wow_increase_count_w = (wow_numeric_w > 0).sum()
                            mom_increase_count_w_from_weekly = (mom_numeric_w > 0).sum()
                            yoy_increase_count_w = (yoy_numeric_w > 0).sum()
                            above_max_count_w = (latest_val_w > max_5y_w).sum()
                            below_min_count_w = (latest_val_w < min_5y_w).sum()
                            above_mean_count_w = (latest_val_w > mean_5y_w).sum()
                            below_mean_count_w = (latest_val_w < mean_5y_w).sum()

                            overall_total_w += total_indicators_w
                            overall_wow_w += wow_increase_count_w
                            overall_mom_w_from_weekly += mom_increase_count_w_from_weekly
                            overall_yoy_w += yoy_increase_count_w
                            overall_above_max_w += above_max_count_w; overall_below_min_w += below_min_count_w
                            overall_above_mean_w += above_mean_count_w; overall_below_mean_w += below_mean_count_w
                            
                            industry_summary_list_w.append({
                                '行业名称': industry_name_w, '指标总数': total_indicators_w,
                                '环比上周增长': wow_increase_count_w, 
                                '环比上月增长': mom_increase_count_w_from_weekly, 
                                '同比去年增长': yoy_increase_count_w,
                                '高于近5年最大值': above_max_count_w, '低于近5年最小值': below_min_count_w,
                                '高于近5年平均值': above_mean_count_w, '低于近5年平均值': below_mean_count_w
                            })
                        except KeyError as e_w_key: print(f"Skipping weekly overview for '{industry_name_w}' (KeyError): {e_w_key}")
                        except Exception as e_w_sum: print(f"Error in weekly overview summary for '{industry_name_w}': {e_w_sum}")
                
                if weekly_cache_valid_data_found and overall_total_w > 0:
                    # 计算占比
                    wow_pct = (overall_wow_w / overall_total_w * 100) if overall_total_w > 0 else 0
                    mom_pct = (overall_mom_w_from_weekly / overall_total_w * 100) if overall_total_w > 0 else 0
                    yoy_pct = (overall_yoy_w / overall_total_w * 100) if overall_total_w > 0 else 0
                    above_max_pct = (overall_above_max_w / overall_total_w * 100) if overall_total_w > 0 else 0
                    below_min_pct = (overall_below_min_w / overall_total_w * 100) if overall_total_w > 0 else 0
                    above_mean_pct = (overall_above_mean_w / overall_total_w * 100) if overall_total_w > 0 else 0
                    below_mean_pct = (overall_below_mean_w / overall_total_w * 100) if overall_total_w > 0 else 0
                    
                    st.markdown(f"所有行业最新周度高频指标共 **{overall_total_w}** 个，"
                                f"其中 **{overall_wow_w}** 个环比上周增长（占比{wow_pct:.1f}%），"
                                f"**{overall_mom_w_from_weekly}** 个与四周前相比增长（占比{mom_pct:.1f}%），"
                                f"**{overall_yoy_w}** 个同比去年增长（占比{yoy_pct:.1f}%）。 "
                                f"与近5年历史比， **{overall_above_max_w}** 个高于最大值（占比{above_max_pct:.1f}%），"
                                f"**{overall_below_min_w}** 个低于最小值（占比{below_min_pct:.1f}%），"
                                f"**{overall_above_mean_w}** 个高于平均值（占比{above_mean_pct:.1f}%），"
                                f"**{overall_below_mean_w}** 个低于平均值（占比{below_mean_pct:.1f}%）。")
                    if industry_summary_list_w:
                        industry_summary_df_w = pd.DataFrame(industry_summary_list_w)
                        
                        # --- BEGIN ADDED CODE FOR TOTAL ROW ---
                        numeric_cols_for_total_w = [
                            '指标总数', '环比上周增长', '环比上月增长', '同比去年增长',
                            '高于近5年最大值', '低于近5年最小值', '高于近5年平均值', '低于近5年平均值'
                        ]
                        cols_present_to_sum_w = [col for col in numeric_cols_for_total_w if col in industry_summary_df_w.columns]

                        if cols_present_to_sum_w:
                            sums_w = industry_summary_df_w[cols_present_to_sum_w].sum()
                            total_row_dict_w = {'行业名称': '合计'}
                            total_row_dict_w.update(sums_w)
                            total_row_df_w = pd.DataFrame([total_row_dict_w])
                            industry_summary_df_w = pd.concat([industry_summary_df_w, total_row_df_w], ignore_index=True)
                        # --- END ADDED CODE FOR TOTAL ROW ---

                        cols_to_show_w = ['行业名称', '指标总数', '环比上周增长', '环比上月增长', '同比去年增长', '高于近5年最大值', '低于近5年最小值', '高于近5年平均值', '低于近5年平均值']
                        st.dataframe(industry_summary_df_w[[col for col in cols_to_show_w if col in industry_summary_df_w.columns]], hide_index=True)
                    else: st.info("未能生成周度行业汇总表。")
                elif not preview_weekly_df_exists: st.info("周度数据未加载，无法显示周度总览。")
                else: st.info("没有可用的周度行业摘要数据(缓存为空、处理错误或无周度数据)。")
            elif not preview_weekly_df_exists: st.info("周度数据未加载，无法显示周度总览。")
            else: st.info("周度行业摘要缓存未初始化或为空。")
            st.markdown("---")

            # --- 月度数据总览 ---
            st.markdown("#### 月度数据总览")
            monthly_summary_cache_for_overview = get_preview_state('preview_monthly_summary_cache', {})
            preview_monthly_df_exists = not get_preview_state('preview_monthly_df', pd.DataFrame()).empty

            if preview_monthly_df_exists and monthly_summary_cache_for_overview:
                overall_total_m, overall_mom_m, overall_yoy_m = 0, 0, 0
                overall_above_max_m, overall_below_min_m, overall_above_mean_m, overall_below_mean_m = 0, 0, 0, 0
                industry_summary_list_m = []
                monthly_cache_valid_data_found = False # Renamed from cache_empty_or_error

                for industry_name_m, summary_table_m in monthly_summary_cache_for_overview.items():
                    if summary_table_m is not None and not summary_table_m.empty:
                        monthly_cache_valid_data_found = True
                        try:
                            total_indicators_m = len(summary_table_m)
                            mom_diff = pd.to_numeric(summary_table_m['环比上月'], errors='coerce') 
                            yoy_diff = pd.to_numeric(summary_table_m['同比上年'], errors='coerce') 
                            latest_val_m = pd.to_numeric(summary_table_m['最新值'], errors='coerce')
                            max_5y_m = pd.to_numeric(summary_table_m['近5年最大值'], errors='coerce')
                            min_5y_m = pd.to_numeric(summary_table_m['近5年最小值'], errors='coerce')
                            mean_5y_m = pd.to_numeric(summary_table_m['近5年平均值'], errors='coerce')
                            mom_increase_count_m = (mom_diff > 0).sum()
                            yoy_increase_count_m = (yoy_diff > 0).sum()
                            above_max_count_m = (latest_val_m > max_5y_m).sum()
                            below_min_count_m = (latest_val_m < min_5y_m).sum()
                            above_mean_count_m = (latest_val_m > mean_5y_m).sum()
                            below_mean_count_m = (latest_val_m < mean_5y_m).sum()
                            overall_total_m += total_indicators_m
                            overall_mom_m += mom_increase_count_m; overall_yoy_m += yoy_increase_count_m
                            overall_above_max_m += above_max_count_m; overall_below_min_m += below_min_count_m
                            overall_above_mean_m += above_mean_count_m; overall_below_mean_m += below_mean_count_m
                            industry_summary_list_m.append({
                                '行业名称': industry_name_m, '指标总数': total_indicators_m,
                                '环比上月增长': mom_increase_count_m, '同比去年增长': yoy_increase_count_m,
                                '高于近5年最大值': above_max_count_m, '低于近5年最小值': below_min_count_m,
                                '高于近5年平均值': above_mean_count_m, '低于近5年平均值': below_mean_count_m
                            })
                        except KeyError as e_m_key: print(f"Skipping monthly overview for '{industry_name_m}' (KeyError): {e_m_key}")
                        except Exception as e_m_sum: print(f"Error in monthly overview for '{industry_name_m}': {e_m_sum}")
                
                if monthly_cache_valid_data_found and overall_total_m > 0:
                    # 计算月度占比
                    mom_pct_m = (overall_mom_m / overall_total_m * 100) if overall_total_m > 0 else 0
                    yoy_pct_m = (overall_yoy_m / overall_total_m * 100) if overall_total_m > 0 else 0
                    above_max_pct_m = (overall_above_max_m / overall_total_m * 100) if overall_total_m > 0 else 0
                    below_min_pct_m = (overall_below_min_m / overall_total_m * 100) if overall_total_m > 0 else 0
                    above_mean_pct_m = (overall_above_mean_m / overall_total_m * 100) if overall_total_m > 0 else 0
                    below_mean_pct_m = (overall_below_mean_m / overall_total_m * 100) if overall_total_m > 0 else 0
                    
                    st.markdown(f"所有行业最新月度高频指标共 **{overall_total_m}** 个，"
                                f"其中 **{overall_mom_m}** 个环比上月增长（占比{mom_pct_m:.1f}%），"
                                f"**{overall_yoy_m}** 个同比去年增长（占比{yoy_pct_m:.1f}%）。 "
                                f"与近5年历史比， **{overall_above_max_m}** 个高于最大值（占比{above_max_pct_m:.1f}%），"
                                f"**{overall_below_min_m}** 个低于最小值（占比{below_min_pct_m:.1f}%），"
                                f"**{overall_above_mean_m}** 个高于平均值（占比{above_mean_pct_m:.1f}%），"
                                f"**{overall_below_mean_m}** 个低于平均值（占比{below_mean_pct_m:.1f}%）。")
                    if industry_summary_list_m:
                        industry_summary_df_m = pd.DataFrame(industry_summary_list_m)
                        
                        # --- BEGIN ADDED CODE FOR MONTHLY TOTAL ROW ---
                        numeric_cols_for_total_m = [
                            '指标总数', '环比上月增长', '同比去年增长',
                            '高于近5年最大值', '低于近5年最小值', '高于近5年平均值', '低于近5年平均值'
                        ]
                        cols_present_to_sum_m = [col for col in numeric_cols_for_total_m if col in industry_summary_df_m.columns]

                        if cols_present_to_sum_m:
                            sums_m = industry_summary_df_m[cols_present_to_sum_m].sum()
                            total_row_dict_m = {'行业名称': '合计'}
                            total_row_dict_m.update(sums_m)
                            total_row_df_m = pd.DataFrame([total_row_dict_m])
                            industry_summary_df_m = pd.concat([industry_summary_df_m, total_row_df_m], ignore_index=True)
                        # --- END ADDED CODE FOR MONTHLY TOTAL ROW ---

                        cols_to_show_m = ['行业名称', '指标总数', '环比上月增长', '同比去年增长', '高于近5年最大值', '低于近5年最小值', '高于近5年平均值', '低于近5年平均值']
                        st.dataframe(industry_summary_df_m[[col for col in cols_to_show_m if col in industry_summary_df_m.columns]], hide_index=True)
                    else: st.info("未能生成月度行业汇总表。")
                elif not preview_monthly_df_exists: st.info("月度数据未加载，无法显示月度总览。")
                else: st.info("没有可用的月度行业摘要数据(缓存为空、处理错误或无月度数据)。")
            elif not preview_monthly_df_exists: st.info("月度数据未加载，无法显示月度总览。")
            else: st.info("月度行业摘要缓存未初始化或为空。")
            st.markdown("--- ")
            # --- <<< 结束迁移：月度数据总览 >>> ---

            # --- <<< 新增：扩散分析部分 >>> ---
           
            # 尝试调用 display_diffusion_tab
            # 我们需要确保 display_diffusion_tab 能够访问到它所需要的数据
            # 通过 session_state 传递。display_diffusion_tab 内部会检查 'data_loaded' 等键
            try:
                # 进行键名映射，确保 diffusion_analysis 模块能获取数据
                if st_session_state.get('preview_data_loaded_files') is not None:
                    st_session_state['data_loaded'] = True
                    st_session_state['weekly_df'] = st_session_state.get('preview_weekly_df', pd.DataFrame())
                    st_session_state['monthly_df'] = st_session_state.get('preview_monthly_df', pd.DataFrame()) # 尽管扩散分析主要用周度
                    st_session_state['source_map'] = st_session_state.get('preview_source_map', {})
                    st_session_state['indicator_industry_map'] = st_session_state.get('preview_indicator_industry_map', {})
                    st_session_state['weekly_industries'] = st_session_state.get('preview_weekly_industries', [])
                    # weekly_summary_cache 也需要被 diffusion_analysis_tab 内部的周度总览使用
                    st_session_state['weekly_summary_cache'] = st_session_state.get('preview_weekly_summary_cache', {})
                else:
                    st_session_state['data_loaded'] = False #确保 display_diffusion_tab 内部检查能工作
                
                # display_diffusion_tab 已在文件顶部导入
                display_diffusion_tab(st, st_session_state) # 直接传递 st 和 st_session_state
            
            except ImportError as e_imp_diff:
                st.error(f"无法导入扩散分析模块: {e_imp_diff}。请确保文件路径和名称正确。")
            except AttributeError as e_attr_diff:
                 st.error(f"调用扩散分析功能时出错: {e_attr_diff}。可能是函数未定义或模块问题。")
            except Exception as e_diff:
                st.error(f"加载或运行扩散分析时发生未知错误: {e_diff}")
                import traceback
                st.error(traceback.format_exc())
            # --- <<< 结束新增：扩散分析部分 >>> ---

    with tab_daily:
        # 调用独立的日度数据标签页函数
        # 准备session_state，确保兼容性
        class TempSessionState:
            def __init__(self):
                pass
            def get(self, key, default=None):
                return getattr(self, key, default)
        
        temp_session_state = TempSessionState()
        temp_session_state.daily_df = st_session_state.get('preview_daily_df', pd.DataFrame())
        temp_session_state.preview_daily_df = st_session_state.get('preview_daily_df', pd.DataFrame())
        temp_session_state.daily_industries = st_session_state.get('preview_daily_industries', [])
        temp_session_state.clean_industry_map = st_session_state.get('preview_clean_industry_map', {})
        temp_session_state.source_map = st_session_state.get('preview_source_map', {})
        temp_session_state.daily_summary_cache = st_session_state.get('preview_daily_summary_cache', {})
        temp_session_state.data_loaded = data_is_ready
        
        # 调用日度数据处理函数
        display_daily_tab(st, temp_session_state)
        
        # 将结果同步回原session_state
        if hasattr(temp_session_state, 'daily_summary_cache'):
            st_session_state.preview_daily_summary_cache = temp_session_state.daily_summary_cache

    with tab_weekly:
        # 调用独立的周度数据标签页函数
        # 准备session_state，确保兼容性
        class TempSessionState:
            def __init__(self):
                pass
            def get(self, key, default=None):
                return getattr(self, key, default)
        
        temp_session_state = TempSessionState()
        temp_session_state.weekly_df = st_session_state.get('preview_weekly_df', pd.DataFrame())
        temp_session_state.weekly_industries = st_session_state.get('preview_weekly_industries', [])
        temp_session_state.clean_industry_map = st_session_state.get('preview_clean_industry_map', {})
        temp_session_state.source_map = st_session_state.get('preview_source_map', {})
        temp_session_state.weekly_summary_cache = st_session_state.get('preview_weekly_summary_cache', {})
        temp_session_state.data_loaded = data_is_ready
        
        # 调用周度数据处理函数
        display_weekly_tab(st, temp_session_state)
        
        # 将结果同步回原session_state
        st_session_state.preview_weekly_summary_cache = temp_session_state.weekly_summary_cache

    with tab_monthly:
        # 调用独立的月度数据标签页函数
        # 准备session_state，确保兼容性
        class TempSessionState:
            def __init__(self):
                pass
            def get(self, key, default=None):
                return getattr(self, key, default)
        
        temp_session_state = TempSessionState()
        temp_session_state.monthly_df = st_session_state.get('preview_monthly_df', pd.DataFrame())
        temp_session_state.preview_monthly_df = st_session_state.get('preview_monthly_df', pd.DataFrame())
        temp_session_state.monthly_industries = st_session_state.get('preview_monthly_industries', [])
        temp_session_state.clean_industry_map = st_session_state.get('preview_clean_industry_map', {})
        temp_session_state.source_map = st_session_state.get('preview_source_map', {})
        temp_session_state.monthly_summary_cache = st_session_state.get('preview_monthly_summary_cache', {})
        temp_session_state.full_monthly_summary = st_session_state.get('preview_monthly_growth_summary_df', pd.DataFrame())
        
        # 调用月度数据处理函数
        display_monthly_tab(st, temp_session_state)
        
        # 将结果同步回原session_state
        st_session_state.preview_monthly_summary_cache = temp_session_state.monthly_summary_cache
        if hasattr(temp_session_state, 'full_monthly_summary'):
            st_session_state.preview_monthly_growth_summary_df = temp_session_state.full_monthly_summary

    with tab_yearly:
        st.header("工业年度数据")
        if not data_is_ready: 
            st.info("请先上传数据文件。")
        else:
            st.write("这里展示工业年度数据。")
        # TODO: 实现年度数据的展示逻辑
    
    if not uploaded_industrial_file and not data_is_ready:
        st.info("请在上方上传工业数据文件以开始分析。")

    st.markdown("---")
   