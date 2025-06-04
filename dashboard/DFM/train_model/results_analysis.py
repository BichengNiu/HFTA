# -*- coding: utf-8 -*-
"""
包含 DFM 结果分析相关的模块，包含原 old/results_analysis.py 和 old/generate_report.py 的所有功能
"""
import pandas as pd
import numpy as np
import os
import logging
import pickle
import joblib
import traceback
import sys
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
import unicodedata  # 🔥 添加缺失的导入
from typing import Dict, List, Optional, Tuple, Any, Union
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore', category=FutureWarning)
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from collections import Counter # 用于因子解释
from statsmodels.tsa.stattools import adfuller # <--- 新增 ADF 检验导入 (为了 apply_stationarity_transforms)
from statsmodels.tsa.statespace.dynamic_factor_mq import DynamicFactorMQ
import datetime
import statsmodels.api as sm # 用于 R2 计算
from openpyxl.utils import get_column_letter # <<< 新增导入
from openpyxl.styles import Font, Alignment, Border, Side, PatternFill # <<< 扩展导入
# --- 移除config导入，因为在新系统中不需要 ---
# <<< 新增：导入 tune_dfm 中的映射加载函数 (如果需要) >>>
# from .tune_dfm import load_mappings # 保持注释，因为映射应该从元数据加载
# --- 修改：导入新增的 Dominance Analysis 函数 --- 
try:
    from .analysis_utils import (
        calculate_metrics_with_lagged_target, 
        calculate_factor_contributions, 
        calculate_individual_variable_r2, 
        calculate_industry_r2, 
        calculate_factor_industry_r2, 
        calculate_factor_type_r2, 
        #calculate_dominance_analysis # <<< 移除导入
    )
except ImportError:
    # 尝试绝对导入作为fallback
    try:
        from analysis_utils import (
            calculate_metrics_with_lagged_target, 
            calculate_factor_contributions, 
            calculate_individual_variable_r2, 
            calculate_industry_r2, 
            calculate_factor_industry_r2, 
            calculate_factor_type_r2, 
        )
    except ImportError as e:
        # 如果仍然失败，创建fallback函数
        print(f"Warning: 无法导入analysis_utils模块: {e}")
        def calculate_metrics_with_lagged_target(*args, **kwargs):
            return None
        def calculate_factor_contributions(*args, **kwargs):
            return None, None
        def calculate_individual_variable_r2(*args, **kwargs):
            return None
        def calculate_industry_r2(*args, **kwargs):
            return None
        def calculate_factor_industry_r2(*args, **kwargs):
            return None
        def calculate_factor_type_r2(*args, **kwargs):
            return None
# --- 结束修改 ---
from scipy.optimize import linear_sum_assignment # <<< 新增：导入匈牙利算法

# Get logger for this module
logger = logging.getLogger(__name__)

# 尝试导入 openpyxl，如果 Excel 写入需要它 (虽然 pandas 会处理)
# try:
#     import openpyxl
# except ImportError:
#     print("警告: openpyxl 未安装，Excel (.xlsx) 文件写入可能失败。请运行 'pip install openpyxl'")

# --- 新增: 尝试设置中文支持 ---
try:
    plt.rcParams['font.sans-serif'] = ['SimHei'] # 或者 'Microsoft YaHei', 'WenQuanYi Micro Hei' 等
    plt.rcParams['axes.unicode_minus'] = False  # 解决负号'-'显示为方块的问题
except Exception as e:
    print(f"设置 Matplotlib 中文字体失败: {e}")
# --- 结束新增 ---


# <<< 添加从 tune_dfm.py 移动过来的函数 >>>

def write_r2_tables_to_excel(
    r2_results: Optional[Dict[str, pd.DataFrame]],
    excel_writer: pd.ExcelWriter,
    sheet_name: str = "Factor R2 Analysis",
    industry_r2: Optional[pd.Series] = None,
    factor_industry_r2: Optional[Dict[str, pd.Series]] = None,
    factor_type_r2: Optional[Dict[str, pd.Series]] = None,
    # dominance_industry_summary: Optional[pd.DataFrame] = None # <<< 移除 Dominance 参数
):
    """
    将各种 R2 表格写入 Excel 文件。
    - 单因子对各变量的 R2 (并排)
    - 整体因子对各行业的 R2 (单个表格)
    - 单因子对各行业的 Pooled R2 (单个表格)
    - 单因子对各类型的 Pooled R2 (单个表格)
    # - Dominance Analysis: 行业对因子贡献汇总 (单个表格) <-- 移除 Dominance

    Args:
        r2_results: 单因子对各变量 R2 的字典。
        excel_writer: pandas ExcelWriter 对象。
        sheet_name: 要写入的 Sheet 名称。
        industry_r2: 整体因子对各行业的 R2 Series。
        factor_industry_r2: 单因子对各行业的 Pooled R2 字典。
        factor_type_r2: 单因子对各类型的 Pooled R2 字典。
        # dominance_industry_summary: 行业对因子贡献的汇总 DataFrame。 <-- 移除 Dominance 参数
    """
    
    # --- 1. 写入单因子对各变量的 R2 (并排) --- 
    max_row_indiv_r2 = 0
    if r2_results is not None and r2_results:
        print(f"\n--- 将因子对单变量 R2 分析表写入 Excel Sheet: {sheet_name} (并排布局) ---")
        try:
            workbook = excel_writer.book
            if sheet_name in excel_writer.sheets:
                worksheet = excel_writer.sheets[sheet_name]
            else:
                worksheet = workbook.create_sheet(title=sheet_name)
                excel_writer.sheets[sheet_name] = worksheet

            bold_font = Font(bold=True)
            start_col = 1
            max_row_written = 0 

            for factor_name in sorted(r2_results.keys()):
                factor_df = r2_results.get(factor_name)
                if factor_df is None or factor_df.empty:
                    continue
                all_vars_df = factor_df.copy()
                all_vars_df.insert(0, '#', range(1, len(all_vars_df) + 1))
                num_cols = len(all_vars_df.columns)
                current_row = 1 # Start writing from row 1
                
                # Write title
                title_cell = worksheet.cell(row=current_row, column=start_col, value=f"All variables explained by {factor_name}")
                title_cell.font = bold_font
                try:
                    worksheet.merge_cells(start_row=current_row, start_column=start_col, end_row=current_row, end_column=start_col + num_cols - 1)
                except ValueError:
                    pass # Ignore merge cell errors if cells are already merged or invalid range
                current_row += 1
                
                # Write header
                header_row = current_row
                for c_idx, value in enumerate(all_vars_df.columns.values):
                    header_cell = worksheet.cell(row=header_row, column=start_col + c_idx, value=value)
                    header_cell.font = bold_font
                current_row += 1
                
                # Write data
                for r_idx, row_data in enumerate(all_vars_df.itertuples(index=False)):
                    data_current_row = current_row + r_idx
                    for c_idx, value in enumerate(row_data):
                        cell = worksheet.cell(row=data_current_row, column=start_col + c_idx)
                        if isinstance(value, (float, np.number)) and (np.isnan(value) or np.isinf(value)): cell.value = None
                        else: cell.value = value
                        if all_vars_df.columns[c_idx] == 'R2': cell.number_format = '0.0000'
                        elif all_vars_df.columns[c_idx] == '#': cell.alignment = cell.alignment.copy(horizontal='center')
                    if data_current_row > max_row_written: max_row_written = data_current_row
                    
                # Adjust column widths
                for c_idx, col_name in enumerate(all_vars_df.columns):
                    col_letter = get_column_letter(start_col + c_idx)
                    try:
                        col_data_str = all_vars_df.iloc[:, c_idx].astype(str)
                        max_len_data = col_data_str.map(len).max()
                        if pd.isna(max_len_data): max_len_data = 4
                        header_len = len(str(col_name))
                        adjusted_width = max(max_len_data, header_len) + 2
                        if col_name == 'Variable': adjusted_width = max(adjusted_width, 35)
                        elif col_name == 'R2': adjusted_width = max(adjusted_width, 10)
                        elif col_name == '#': adjusted_width = 5
                        worksheet.column_dimensions[col_letter].width = adjusted_width
                    except Exception: # Catch potential errors like empty data
                        worksheet.column_dimensions[col_letter].width = 15 # Default width
                        
                start_col += num_cols + 1
                
            max_row_indiv_r2 = max_row_written # 记录这部分写入的最大行
            print(f"--- 因子对单变量 R2 分析表写入完成 --- (Max Row: {max_row_indiv_r2})")
        except Exception as e_indiv:
            print(f"写入单因子对变量 R2 (并排布局) 时发生错误: {e_indiv}")
            # traceback.print_exc() # 移除 traceback 打印以减少日志噪音
            max_row_indiv_r2 = 1 # 假设至少写了标题行
    else:
        print(f"  未提供单变量 R2 结果 (r2_results)，跳过并排表格。")
        max_row_indiv_r2 = 0 # 没有写入行
        # Ensure sheet exists if other tables need to be written
        try:
            if sheet_name not in excel_writer.sheets:
                 workbook = excel_writer.book
                 worksheet = workbook.create_sheet(title=sheet_name)
                 excel_writer.sheets[sheet_name] = worksheet
        except Exception as e_sheet_create:
             print(f"  警告: 尝试为后续表格创建 Sheet '{sheet_name}' 失败: {e_sheet_create}")


    # --- 2. 写入行业和 Pooled R2 表格 (以及 Dominance 结果) --- 
    current_row = max_row_indiv_r2 + 3 # 在之前表格下方留出 2 行空白
    start_col_combined = 1

    # 获取 worksheet (可能在上面已创建或获取)
    worksheet = excel_writer.sheets.get(sheet_name)
    if worksheet is None:
        print(f"错误：无法获取 Worksheet '{sheet_name}' 以写入合并的 R2 表格。")
        return # 无法继续
    bold_font = Font(bold=True)

    # --- Helper function to write a DataFrame table (保持不变) ---
    def write_single_table(ws, df, title, start_r, start_c, bold_f, number_format='0.0000'): # Add number_format
        print(f"  正在写入表格: '{title}' (开始于 R{start_r}C{start_c})...")
        max_c_written = start_c -1 # Track max column written for this table
        # Write title
        try:
            title_cell = ws.cell(row=start_r, column=start_c, value=title)
            title_cell.font = bold_f
            current_r = start_r + 1
        except Exception as e_title:
            print(f"    错误: 写入表格标题 '{title}' 时出错: {e_title}")
            return start_r # Return original row if title fails

        # Write header
        try:
            index_header = df.index.name if df.index.name else "Index"
            ws.cell(row=current_r, column=start_c, value=index_header).font = bold_f
            max_c_written = start_c
            for c_idx, col_name in enumerate(df.columns):
                col_c = start_c + 1 + c_idx
                ws.cell(row=current_r, column=col_c, value=col_name).font = bold_f
                max_c_written = col_c
            current_r += 1
        except Exception as e_header:
            print(f"    错误: 写入表格 '{title}' 的表头时出错: {e_header}")
            return start_r + 1 # Return row after attempted title

        # Write data
        try:
            for r_idx, index_val in enumerate(df.index):
                data_r = current_r + r_idx
                ws.cell(row=data_r, column=start_c, value=index_val)
                for c_idx, col_name in enumerate(df.columns):
                    col_c = start_c + 1 + c_idx
                    value = df.iloc[r_idx, c_idx]
                    cell = ws.cell(row=data_r, column=col_c)
                    if isinstance(value, (float, np.number)) and (np.isnan(value) or np.isinf(value)): cell.value = None
                    else: cell.value = value
                    cell.number_format = number_format # Apply specified format
            final_row = current_r + len(df) - 1
        except Exception as e_data:
            print(f"    错误: 写入表格 '{title}' 的数据时出错: {e_data}")
            return current_r # Return row where data writing started

        # Adjust column widths for this table
        try:
            # Index column
            col_letter = get_column_letter(start_c)
            index_header = df.index.name if df.index.name else "Index"
            max_len_index = max(len(str(index_header)), df.index.astype(str).map(len).max()) + 2
            ws.column_dimensions[col_letter].width = max(max_len_index, 15) # Min width 15 for index
            # Data columns
            for c_idx, col_name in enumerate(df.columns):
                col_c = start_c + 1 + c_idx
                col_letter = get_column_letter(col_c)
                # Format number as string with target format to estimate width
                if number_format.endswith('%'):
                     col_data_str = df.iloc[:, c_idx].apply(lambda x: f"{x*100:.2f}%" if pd.notna(x) else "")
                elif '0' in number_format:
                     num_decimals = number_format.count('0', number_format.find('.')) if '.' in number_format else 0
                     col_data_str = df.iloc[:, c_idx].apply(lambda x: f"{x:.{num_decimals}f}" if pd.notna(x) else "")
                else:
                     col_data_str = df.iloc[:, c_idx].astype(str)

                max_len_data = col_data_str.map(len).max()
                if pd.isna(max_len_data): max_len_data = 6 # 'None' + buffer
                header_len = len(str(col_name))
                adjusted_width = max(max_len_data, header_len) + 2
                ws.column_dimensions[col_letter].width = max(adjusted_width, 12) # Min width 12 for R2/Contrib cols
        except Exception as e_width:
             print(f"    警告: 调整表格 '{title}' 列宽时出错: {e_width}")

        return final_row # Return the last row written by this table
    # --- End Helper ---
    
    last_row_written = current_row - 1 # Initialize from starting point
    
    # --- Write Industry R2 --- 
    if industry_r2 is not None and not industry_r2.empty:
        df_industry_r2 = industry_r2.to_frame(name="Industry R2 (All Factors)")
        df_industry_r2.index.name = "Industry"
        last_row_written = write_single_table(worksheet, df_industry_r2, "Industry R2 (All Factors)", current_row, start_col_combined, bold_font)
        current_row = last_row_written + 3 # Update for next table
    else:
        print("  未提供行业 R2 结果 (industry_r2)，跳过表格。")
        
    # --- Write Factor-Industry Pooled R2 --- 
    if factor_industry_r2 is not None and factor_industry_r2:
        try:
            df_factor_industry = pd.DataFrame(factor_industry_r2)
            df_factor_industry.index.name = "Industry"
            df_factor_industry = df_factor_industry.sort_index() # Sort by industry name
            # Ensure only factor columns, handling potential non-factor keys
            factor_cols = sorted([col for col in df_factor_industry.columns if col.startswith('Factor')], 
                                key=lambda x: int(x.replace('Factor','')))
            df_factor_industry = df_factor_industry[factor_cols] 
            last_row_written = write_single_table(worksheet, df_factor_industry, "Factor-Industry Pooled R²", current_row, start_col_combined, bold_font)
            current_row = last_row_written + 3
        except Exception as e_fi_prep:
            print(f"  准备或写入 Factor-Industry Pooled R2 表格时出错: {e_fi_prep}")
    else:
         print("  未提供因子-行业 Pooled R2 结果 (factor_industry_r2)，跳过表格。")

    # --- Write Factor-Type Pooled R2 --- 
    # ... (代码保持不变) ...
    logger.info(f"[Debug Write R2] Received 'factor_type_r2'. Type: {type(factor_type_r2)}. Is None or Empty: {factor_type_r2 is None or (isinstance(factor_type_r2, dict) and not factor_type_r2)}")
    if isinstance(factor_type_r2, dict) and factor_type_r2:
        logger.info(f"[Debug Write R2] 'factor_type_r2' keys: {list(factor_type_r2.keys())}")

    if factor_type_r2 is not None and factor_type_r2:
        try:
            df_factor_type = pd.DataFrame(factor_type_r2)
            df_factor_type.index.name = "Type" # Set index name to Type
            df_factor_type = df_factor_type.sort_index() # Sort by type name
            # Ensure only factor columns and sort them
            factor_cols = sorted([col for col in df_factor_type.columns if col.startswith('Factor')],
                                key=lambda x: int(x.replace('Factor','')))
            df_factor_type = df_factor_type[factor_cols]
            last_row_written = write_single_table(worksheet, df_factor_type, "Factor-Type Pooled R²", current_row, start_col_combined, bold_font)
            current_row = last_row_written + 3 # Update row counter
        except Exception as e_ft_prep:
            print(f"  准备或写入 Factor-Type Pooled R2 表格时出错: {e_ft_prep}")
            # traceback.print_exc() # Removed for brevity
    else:
        print("  未提供因子-类型 Pooled R2 结果 (factor_type_r2)，跳过表格。")


    print(f"--- 所有 R2 分析表格写入完成 (Sheet: {sheet_name}) ---") # <-- 更新结束消息
# <<< 结束添加函数 >>>

# <<< 新增: 创建对齐的 Nowcast vs Target 表格的辅助函数 >>>
def create_aligned_nowcast_target_table(
    nowcast_weekly_orig: pd.Series,
    target_orig: pd.Series,
    target_variable_name: str = "Target (Original Scale)" # 允许自定义目标列名
) -> pd.DataFrame:
    """
    将周度 Nowcast 与月度 Target 对齐。

    对齐规则：
    1. 选取每个月最后一个周五的 Nowcast 值。
    2. 将该 Nowcast 值与下一个月的 Target 值进行匹配。
    3. 返回的 DataFrame 以每月最后一个周五作为索引。

    Args:
        nowcast_weekly_orig: 周度频率的 Nowcast 时间序列 (原始尺度)。
        target_orig: 原始频率的目标变量时间序列 (原始尺度)。
        target_variable_name: 在输出 DataFrame 中用于目标列的名称。

    Returns:
        包含对齐后的 Nowcast 和 Target 的 DataFrame。
        索引：每月最后一个周五的日期。
        列：['Nowcast (Original Scale)', target_variable_name]
    """
    logger.info("开始创建对齐的 Nowcast vs Target 表格...")

    if not isinstance(nowcast_weekly_orig.index, pd.DatetimeIndex):
        nowcast_weekly_orig.index = pd.to_datetime(nowcast_weekly_orig.index)
    if not isinstance(target_orig.index, pd.DatetimeIndex):
        target_orig.index = pd.to_datetime(target_orig.index)

    # 🔥 新增：去除基于索引的重复项 (与老代码一致)
    if not nowcast_weekly_orig.index.is_unique:
        logger.info(f"原始 nowcast_weekly_orig 索引包含 {nowcast_weekly_orig.index.duplicated().sum()} 个重复项。将保留每个重复日期的最后一条记录。")
        nowcast_weekly_orig = nowcast_weekly_orig[~nowcast_weekly_orig.index.duplicated(keep='last')]
    
    if not target_orig.index.is_unique:
        logger.info(f"原始 target_orig 索引包含 {target_orig.index.duplicated().sum()} 个重复项。将保留每个重复日期的最后一条记录。")
        target_orig = target_orig[~target_orig.index.duplicated(keep='last')]
    # 🔥 结束新增

    # 🔥 修复：直接使用完整的 nowcast_weekly_orig (与老代码一致)
    # 不再进行周五筛选，使用完整的 nowcast 数据
    nowcast_aligned_base = nowcast_weekly_orig.copy()
    nowcast_aligned_base.name = 'Nowcast (Original Scale)' # 重命名 Series

    # 2. 准备 Target 数据
    # 假设 Target 数据频率至少是月度，如果不是，需要先聚合
    # 这里我们假设 target_orig 的索引代表了数据所属的月份 <-- 这个注释是误导性的，索引是发布日期
    # 例如 2023-01-31 代表 1 月的数据 <-- 这个例子也是误导性的
    # 我们需要能按年月访问它
    target_df = target_orig.dropna().to_frame(target_variable_name)
    target_df = target_df.sort_index(ascending=False) # 🔥 新增：按原始日期索引降序排序
    # --- 🔥 关键修复：Target数据索引已经是对齐后的日期（月末最后周五），不需要减1个月 ---
    # target_df['TargetPeriod'] = target_df.index.to_period('M') # <-- 旧的错误逻辑: 使用发布日期月份  
    # target_df['TargetPeriod'] = (target_df.index - pd.DateOffset(months=1)).to_period('M') # <-- 也是错误的：假设索引是发布日期
    # 正确逻辑：Target数据索引已经是对齐后的日期，直接使用索引的月份
    target_df['TargetPeriod'] = target_df.index.to_period('M')
    # --- 结束修改 ---
    
    logger.info(f"在对 TargetPeriod 去重前, target_df 有 {len(target_df)} 行。TargetPeriod 的唯一计数为: {target_df['TargetPeriod'].nunique()}")
    target_df = target_df.drop_duplicates(subset=['TargetPeriod'], keep='first') # 🔥 新增：基于 TargetPeriod 去重
    logger.info(f"在对 TargetPeriod 去重后, target_df 有 {len(target_df)} 行。")

    # 3. 创建对齐 DataFrame
    aligned_df = nowcast_aligned_base.to_frame()
    # Nowcast 在 M 月最后一个周五的值，对应 Target M 月份的值 (通常在 M+1 月发布)
    # 我们给 Nowcast 的 Period 加一个月，用于匹配  <- 旧注释，逻辑已修改
    # --- 修改：使用 Nowcast 的当前月份 Period 进行匹配 ---
    # aligned_df['TargetMatchPeriod'] = aligned_df.index.to_period('M') + 1 # <-- 旧逻辑
    aligned_df['TargetMatchPeriod'] = aligned_df.index.to_period('M') # <-- 新逻辑：使用当前月份
    # --- 结束修改 ---

    # 将 Target 的 Period 作为匹配键
    target_df_for_merge = target_df.set_index('TargetPeriod')

    # 合并
    final_aligned_table = pd.merge(
        aligned_df,
        target_df_for_merge[[target_variable_name]], # 只选择目标值列
        left_on='TargetMatchPeriod',
        right_index=True, # 使用 Target 的 Period 索引
        how='left' # 保留所有 Nowcast 点，即使没有对应的未来 Target
    )

    # 清理结果
    final_aligned_table = final_aligned_table.drop(columns=['TargetMatchPeriod'])
    final_aligned_table = final_aligned_table.sort_index() # 🔥 新增：按索引排序
    
    logger.info(f"成功创建对齐表格，包含 {len(final_aligned_table)} 行数据。")
    # logger.debug(f"对齐表示例:\\n{final_aligned_table.head()}") # Corrected string formatting

    return final_aligned_table
# <<< 结束新增 >>>

# --- <<< 新增：重新生成 plot_final_nowcast 函数 >>> ---
def plot_final_nowcast(
    final_nowcast_series: pd.Series,
    target_for_plot: pd.Series, # 原始目标序列 (dropna 后)
    validation_start: Union[str, pd.Timestamp],
    validation_end: Union[str, pd.Timestamp],
    title: str,
    filename: str
):
    """
    (重新生成) 绘制最终的周度 nowcast 与实际观测值的对比图（原始水平）。
    功能尽量模拟原始意图，绘制完整 nowcast，屏蔽 1/2 月实际值。
    """
    logger.info(f"\n[绘图函数] 生成最终 Nowcasting 图: {filename}...")
    try:
        # --- 确保索引为 DatetimeIndex ---
        try:
            if not isinstance(final_nowcast_series.index, pd.DatetimeIndex):
                final_nowcast_series.index = pd.to_datetime(final_nowcast_series.index)
            if not isinstance(target_for_plot.index, pd.DatetimeIndex):
                target_for_plot.index = pd.to_datetime(target_for_plot.index)
        except Exception as e_index_conv:
             logger.warning(f"警告: 将索引转换为 DatetimeIndex 时出错: {e_index_conv}")

        nowcast_col_name = 'Nowcast_Orig'
        target_col_name = target_for_plot.name if target_for_plot.name is not None else 'Actual'
        if target_col_name == nowcast_col_name: target_col_name = 'Observed_Value'

        # --- 准备绘图数据 --- 
        plot_df = final_nowcast_series.to_frame(name=nowcast_col_name)
        plot_df = plot_df.join(target_for_plot.rename(target_col_name), how='left')

        # --- 绘图时屏蔽 1/2 月实际值 --- 
        if target_col_name in plot_df.columns and isinstance(plot_df.index, pd.DatetimeIndex):
            month_indices_plot = plot_df.index.month
            plot_df.loc[(month_indices_plot == 1) | (month_indices_plot == 2), target_col_name] = np.nan
            logger.info("  已屏蔽 1月/2月 的实际观测值用于绘图。")

        if not plot_df.empty:
            plt.figure(figsize=(14, 7))
            nowcast_label = '周度 Nowcast (原始水平)'
            actual_label = '观测值 (原始水平, 屏蔽1/2月)'
            ylabel = '值 (原始水平)'

            plt.plot(plot_df.index, plot_df[nowcast_col_name], label=nowcast_label, linestyle='-', alpha=0.8, color='blue')

            if target_col_name in plot_df.columns:
                target_to_plot_filtered = plot_df[target_col_name].dropna()
                plt.plot(target_to_plot_filtered.index, target_to_plot_filtered.values, label=actual_label, marker='o', linestyle='None', markersize=4, color='red')

            # --- 标记验证期 --- 
            try:
                if isinstance(plot_df.index, pd.DatetimeIndex):
                    plot_start_date = plot_df.index.min()
                    plot_end_date = plot_df.index.max()
                    val_start_dt = pd.to_datetime(validation_start)
                    val_end_dt = pd.to_datetime(validation_end)
                    span_start = max(plot_start_date, val_start_dt)
                    span_end = min(plot_end_date, val_end_dt)
                    if span_start < span_end:
                        plt.axvspan(span_start, span_end, color='yellow', alpha=0.2, label='验证期')
                    else:
                        plt.axvspan(val_start_dt, val_end_dt, color='yellow', alpha=0.2, label='验证期 (超出范围)')
                else:
                     logger.warning("  警告: 绘图数据索引不是 DatetimeIndex，无法标记验证期。")
            except Exception as date_err:
                logger.warning(f"  警告：标记验证期时出错 - {date_err}")

            plt.title(title)
            plt.xlabel('日期')
            plt.ylabel(ylabel)
            plt.legend()
            plt.grid(True, linestyle='--', alpha=0.6)
            ax = plt.gca()
            ax.xaxis.set_major_locator(mdates.AutoDateLocator(minticks=8, maxticks=15))
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
            plt.xticks(rotation=45, ha='right')
            plt.tight_layout()
            plt.savefig(filename)
            plt.close()
            logger.info(f"最终 Nowcasting 图已保存到: {filename}")
        else:
             logger.error("错误：无法准备用于绘图的数据 (plot_df 为空)。")
    except Exception as e:
        logger.error(f"[绘图函数] 生成或保存最终 Nowcasting 图时出错: {e}", exc_info=True)
        try:
            plt.close() # 尝试关闭以防万一
        except Exception: pass
# --- 结束重新生成 --- 

def analyze_and_save_final_results(
    run_output_dir: str,
    timestamp_str: str,
    excel_output_path: str,
    # --- 修改：添加/确认所需数据 --- 
    all_data_full: pd.DataFrame, # 包含原始目标变量
    final_data_processed: pd.DataFrame, # DFM 输入数据 (处理后, 包含验证/预测期) <--- 确认包含验证期
    final_target_mean_rescale: Optional[float], # 用于 Nowcast 反标准化
    final_target_std_rescale: Optional[float], # 用于 Nowcast 反标准化
    # --- 结束修改 ---
    target_variable: str,
    final_dfm_results: Any, # DFM 结果对象, 假定为 statsmodels results wrapper OR DFMEMResultsWrapper
    best_variables: List[str],
    best_params: Dict[str, Any],
    var_type_map: Dict[str, str], # 用于变量类型注释 (key 需要规范化!)
    total_runtime_seconds: float,
    training_start_date: str,   # <-- 新增参数
    validation_start_date: str, # 验证期开始日期
    validation_end_date: str,   # 验证期结束日期
    train_end_date: str,        # 训练期结束日期 (用于指标计算分割)
    factor_contributions: Optional[Dict[str, float]] = None, # 因子对目标方差贡献度
    final_transform_log: Optional[Dict[str, str]] = None,   # 最终变量转换日志
    pca_results_df: Optional[pd.DataFrame] = None,          # PCA 结果 DataFrame
    contribution_results_df: Optional[pd.DataFrame] = None, # 因子贡献度结果 DataFrame (Note: Appears unused in provided snippet, maybe used later?)
    var_industry_map: Optional[Dict[str, str]] = None,      # 变量行业映射 (key 需要规范化!)
    individual_r2_results: Optional[Dict[str, pd.DataFrame]] = None, # <<< 添加新参数
    industry_r2_results: Optional[pd.Series] = None, # <<< 添加行业 R2 参数
    factor_industry_r2_results: Optional[Dict[str, pd.Series]] = None, # <<< 添加因子-行业 R2 参数
    factor_type_r2_results: Optional[Dict[str, pd.Series]] = None, # <<< 新增：添加因子-类型 R2 参数
    final_eigenvalues: Optional[np.ndarray] = None # <<< 新增：添加最终特征根值参数
) -> Tuple[Optional[pd.Series], Dict[str, Any]]: # <<< 修改返回类型注解
    """
    分析最终 DFM 模型结果并将其保存到 Excel 文件。
    (修改：计算实时滤波 Nowcast y(t|t) 并用于分析)
    (修改：兼容 DFMEMResultsWrapper 类型)
    """
    logger.info(f"开始分析最终结果并写入 Excel: {excel_output_path}")
    calculated_nowcast_orig = None # This will hold the FILTERED nowcast
    calculated_nowcast_smoothed_orig = None # Optional: To store the smoothed one
    metrics = {} # Initialize metrics dictionary
    aligned_df_for_metrics = None # Initialize aligned df
    loadings_df_final = None 
    filtered_state = None # Initialize filtered_state

    try:
        # --- 1. 提取必要信息 --- 
        # --- 修改：检查 DFMEMResultsWrapper 的属性 ---
        if not final_dfm_results or not hasattr(final_dfm_results, 'x_sm') or not hasattr(final_dfm_results, 'Lambda'):
             raise ValueError("提供的 DFM 结果对象无效或缺少必要的属性 (如 'x_sm', 'Lambda')。")
        # --- 结束修改 ---

        # --- 移除 filter 方法检查 ---
        # # 检查 filter 方法
        # # Adjusted check: Some results objects might have filter on the model object
        # results_filter_method = getattr(final_dfm_results, 'filter', None)
        # model_filter_method = getattr(getattr(final_dfm_results, 'model', None), 'filter', None)
        # if not callable(results_filter_method) and not callable(model_filter_method):
        #      raise AttributeError("DFM 结果对象及其关联的 model 对象都缺少可调用的 'filter' 方法。")
        # --- 结束移除 ---

        # --- 修改：调整载荷提取逻辑 (Lambda 是大写) ---
        if hasattr(final_dfm_results, 'Lambda') and final_dfm_results.Lambda is not None:
            loadings = final_dfm_results.Lambda # 使用大写 Lambda
        else:
            raise ValueError("无法从 DFM 结果对象获取载荷矩阵 ('Lambda' 属性)。")
        # --- 结束修改 ---


        # 检查载荷 (Lambda) 是否是 DataFrame 或 ndarray
        if not isinstance(loadings, pd.DataFrame):
            logger.warning(f"Loadings (Lambda) 不是 DataFrame (类型: {type(loadings)}), 尝试转换...")
            try:
                # 假设 loadings 是 ndarray [n_endog, k_factors]
                # 需要 endog_names 和 factor names
                # 尝试从 best_params 获取因子数，从 best_variables 获取变量名
                k_factors = best_params.get('k_factors_final', None) # 从 best_params 获取
                endog_names = best_variables # 使用 best_variables

                if k_factors is None and isinstance(loadings, np.ndarray):
                    k_factors = loadings.shape[1] # 从 ndarray 形状推断
                    logger.warning(f"无法从 best_params 获取因子数，从 Lambda 形状推断为 {k_factors}")

                if k_factors is not None and endog_names is not None:
                    if isinstance(loadings, np.ndarray): # Check if it's an ndarray first
                        if loadings.shape == (len(endog_names), k_factors):
                            loadings = pd.DataFrame(loadings, index=endog_names, columns=[f'Factor{i+1}' for i in range(k_factors)])
                            logger.info("Loadings 成功从 ndarray 转换为 DataFrame。")
                        else: # <<< This else handles the shape mismatch for ndarray
                            raise ValueError(f"Loadings (ndarray) 维度 ({getattr(loadings, 'shape', 'N/A')}) 与 endog_names ({len(endog_names)}) 或 k_factors ({k_factors}) 不匹配。")
                    # Removed the problematic else: here, as it doesn't pair with the try
                else: # This else pairs with 'if k_factors is not None and endog_names is not None:'
                    raise ValueError("无法获取 k_factors 或 endog_names (best_variables) 以转换 loadings ndarray。")
            # Correctly paired except block for the try block above
            except Exception as e_load_conv:
                logger.error(f"转换 Loadings 为 DataFrame 失败: {e_load_conv}", exc_info=True)
                raise TypeError(f"转换 Loadings 为 DataFrame 失败: {e_load_conv}")

        # Ensure loadings_df_final is assigned even if conversion wasn't needed or failed initially
        if not isinstance(loadings, pd.DataFrame):
            logger.warning("Loadings 既不是预期的 DataFrame 也未能成功从 ndarray 转换。")
            # Assign None or raise, depending on desired behavior if conversion fails.
            loadings_df_final = None
        else:
            loadings_df_final = loadings # Store the final loadings DataFrame

        # Add a check here to ensure loadings_df_final is usable AFTER the try-except block
        if loadings_df_final is None or not isinstance(loadings_df_final, pd.DataFrame):
            raise ValueError("未能获取或生成有效的 Loadings DataFrame。")


        # 检查目标变量是否在载荷矩阵中
        if target_variable not in loadings_df_final.index:
            # Attempt to normalize target_variable name if it exists in logs but not loadings
            if final_transform_log and target_variable in final_transform_log:
                transformed_target_name = final_transform_log[target_variable]
                if transformed_target_name in loadings_df_final.index:
                    logger.warning(f"目标变量 '{target_variable}' 不在载荷索引中，但转换后的名称 '{transformed_target_name}' 存在。将使用转换后的名称。")
                    target_variable_in_loadings = transformed_target_name
                else: # <<< Correct indentation for this else
                    raise ValueError(f"DFM 载荷矩阵索引中不包含目标变量 '{target_variable}' 或其转换后名称 '{transformed_target_name}'。Index: {loadings_df_final.index}")
            else:
                raise ValueError(f"DFM 载荷矩阵 (Loadings DataFrame) 的索引中不包含目标变量 '{target_variable}'。Index: {loadings_df_final.index}")
        else:
            target_variable_in_loadings = target_variable # Use original name
        
        target_loadings = loadings_df_final.loc[target_variable_in_loadings]

        # 获取反标准化参数
        target_mean = final_target_mean_rescale
        target_std = final_target_std_rescale
        if target_mean is None or target_std is None:
            raise ValueError("缺少用于反标准化的目标变量均值或标准差。")


        # --- 2. 计算实时滤波 Nowcast y(t|t) ---
        # --- 修改：直接使用 .x 属性作为滤波状态 ---
        # logger.info("计算实时滤波 Nowcast 序列 y(t|t) (使用 final_dfm_results.x)...") #<-- 旧日志
        # --- 修改：改为计算 y_{T(t)|t} ---
        logger.info("计算对月底目标的 Nowcast 序列 y_{T(t)|t} (使用 final_dfm_results.x 和 A)...")
        try:
            # 检查 final_dfm_results 是否有 .x 和 .A 属性
            if not hasattr(final_dfm_results, 'x') or final_dfm_results.x is None:
                raise AttributeError("DFM 结果对象缺少 'x' 属性 (假定为滤波状态) 或其值为 None。")
            if not hasattr(final_dfm_results, 'A') or final_dfm_results.A is None:
                 raise AttributeError("DFM 结果对象缺少 'A' 属性 (状态转移矩阵) 或其值为 None。")


            filtered_state_raw = final_dfm_results.x # 直接获取滤波状态
            A = final_dfm_results.A # 获取状态转移矩阵

            # 确保 A 是 NumPy array
            if not isinstance(A, np.ndarray):
                raise TypeError(f"状态转移矩阵 A 不是 NumPy array (Type: {type(A)})")
            n_factors = A.shape[0] # 获取因子数量

            # 确保 filtered_state 是 DataFrame 并具有正确的索引
            if not isinstance(filtered_state_raw, pd.DataFrame):
                 if isinstance(filtered_state_raw, np.ndarray):
                     logger.warning("Filtered state (x) 是 ndarray，尝试使用 final_data_processed 的索引转换为 DataFrame。")
                     try:
                         # 确保索引长度匹配状态数组长度
                         # 注意：final_data_processed 可能包含超出滤波期的数据，要对齐
                         index_for_state = final_data_processed.index[:len(filtered_state_raw)]
                         if len(index_for_state) != filtered_state_raw.shape[0]:
                              raise ValueError(f"用于转换 filtered_state 的索引长度 ({len(index_for_state)}) 与状态数组长度 ({filtered_state_raw.shape[0]}) 不匹配。")

                         k_factors_state = filtered_state_raw.shape[1]
                         if k_factors_state != n_factors:
                              raise ValueError(f"滤波状态的因子数 ({k_factors_state}) 与状态转移矩阵 A 的维度 ({n_factors}) 不匹配。")

                         filtered_state = pd.DataFrame(filtered_state_raw,
                                                       index=index_for_state,
                                                       columns=[f'Factor{i+1}' for i in range(k_factors_state)])
                     except Exception as e_fs_conv:
                         raise TypeError(f"将 filtered_state (x, ndarray) 转换为 DataFrame 失败: {e_fs_conv}")
                 else:
                     raise TypeError(f"Filtered state (x) 不是 Pandas DataFrame 或 Numpy Array。 Type: {type(filtered_state_raw)}")
            else:
                 filtered_state = filtered_state_raw # Already a DataFrame
                 if filtered_state.shape[1] != n_factors:
                      raise ValueError(f"滤波状态 DataFrame 的因子数 ({filtered_state.shape[1]}) 与状态转移矩阵 A 的维度 ({n_factors}) 不匹配。")

            # --- 确保索引是 DatetimeIndex (用于月份计算) ---
            if not pd.api.types.is_datetime64_any_dtype(filtered_state.index):
                 logger.warning("Filtered state index is not datetime, attempting conversion...")
                 try:
                     original_index = filtered_state.index
                     filtered_state.index = pd.to_datetime(filtered_state.index)
                     logger.info("Filtered state index converted to DatetimeIndex.")
                 except Exception as e_dt_conv:
                      logger.error(f"Failed to convert filtered state index to DatetimeIndex: {e_dt_conv}. Cannot calculate month-end targets.", exc_info=True)
                      raise TypeError(f"无法将 filtered_state 索引转换为 DatetimeIndex: {e_dt_conv}") from e_dt_conv

            logger.debug(f"获取的 filtered_state (来自 .x) 索引范围: {filtered_state.index.min()} to {filtered_state.index.max()}")

            # --- 计算对月底目标的预测 y_{T(t)|t} ---
            nowcast_list_T = []
            index_list_T = []
            # 假设数据频率是周五 ('W-FRI')，与 run_nowcasting_evolution.py 一致
            # 预计算 A^k 以提高效率 (如果 k 值重复度高)
            A_pow_k_cache = {}

            for t, x_t in filtered_state.iterrows():
                try:
                    # 1. 确定 t 所属月份的最后那个周五 T(t)
                    month_end_date = t + pd.offsets.MonthEnd(0) # 获取 t 所在月的最后一天
                    # 从 t 开始，向后找到当月的最后一个周五
                    # 找到该月所有日期
                    all_days_in_month = pd.date_range(start=t.to_period('M').start_time, end=month_end_date, freq='D')
                    # 筛选出周五
                    fridays_in_month = all_days_in_month[all_days_in_month.dayofweek == 4]
                    if fridays_in_month.empty:
                        # 如果当月没有周五（理论上不可能），或 t 之后没有周五，则 T(t)=t， k=0
                        target_date_T = t
                    else:
                        # 取当月最后一个周五作为目标 T(t)
                        target_date_T = fridays_in_month[-1]
                        # --- 重要：确保 T(t) >= t ---
                        if target_date_T < t:
                             # 如果 t 已经是当月最后一个周五之后了，则目标就是 t 本身
                             target_date_T = t

                    # 2. 计算预测步数 k = T(t) - t (以周为单位)
                    if target_date_T == t:
                        k = 0
                    else:
                        days_diff = (target_date_T - t).days
                        # 注意：这里假设了每周都有数据点，或者说状态转移是按周进行的
                        # 如果数据中有缺失的周五，直接除以 7 可能不精确
                        # 更稳妥的方式是计算两个日期在 filtered_state.index 中的位置差
                        try:
                             t_loc = filtered_state.index.get_loc(t)
                             T_loc = filtered_state.index.get_loc(target_date_T)
                             k = T_loc - t_loc
                             if k < 0: # Double check
                                 logger.warning(f"计算的步数 k < 0 ({k}) for t={t}, T(t)={target_date_T}. Setting k=0.")
                                 k = 0
                        except KeyError:
                             # 如果 T(t) 不在索引中，回退到使用天数计算（可能有风险）
                             logger.warning(f"Target date T(t)={target_date_T} not in filtered_state index for t={t}. Calculating k based on days/7.")
                             k = int(round(days_diff / 7.0))
                             if k < 0: k = 0 # Ensure k is not negative

                    # 3. 计算 A^k
                    if k == 0:
                        A_pow_k = np.eye(n_factors)
                    elif k in A_pow_k_cache:
                        A_pow_k = A_pow_k_cache[k]
                    else:
                        try:
                            A_pow_k = np.linalg.matrix_power(A, k)
                            A_pow_k_cache[k] = A_pow_k # Cache the result
                        except np.linalg.LinAlgError as e_power:
                            logger.error(f"计算 A^{k} (for t={t}, T(t)={target_date_T}) 时发生线性代数错误: {e_power}. 将使用单位矩阵代替。")
                            A_pow_k = np.eye(n_factors)

                    # 4. 计算预测状态 x_{T(t)|t}
                    x_T_given_t = A_pow_k @ x_t.values # x_t 是 Series

                    # 5. 计算标准化 Nowcast
                    # Ensure alignment between predicted state and target_loadings columns
                    # Assuming x_T_given_t is a numpy array [n_factors,]
                    # Assuming target_loadings is a pandas Series with factor names as index
                    nowcast_std = target_loadings.values @ x_T_given_t # Dot product

                    nowcast_list_T.append(nowcast_std)
                    index_list_T.append(t)

                except Exception as e_loop:
                    logger.error(f"计算 t={t} 的 y_(T(t)|t) 时出错: {e_loop}", exc_info=True)
                    # 选择跳过该点或添加 NaN
                    nowcast_list_T.append(np.nan)
                    index_list_T.append(t)

            # --- 结束循环 ---

            # 创建标准化 Nowcast Series
            nowcast_forecast_standardized = pd.Series(nowcast_list_T, index=index_list_T)
            nowcast_forecast_standardized.name = "Nowcast_Forecast_Standardized"

            # 反标准化
            calculated_nowcast_orig = (nowcast_forecast_standardized * target_std) + target_mean
            # --- 修改：更新列名以反映计算方式 ---
            calculated_nowcast_orig.name = "Nowcast_ForecastToMonthEnd"
            logger.info("对月底目标的 Nowcast 序列 y_{T(t)|t} 计算完成。")

        except Exception as e_filter:
            logger.error(f"计算对月底目标的 Nowcast 时出错: {e_filter}", exc_info=True)
            logger.warning("无法计算对月底目标的 Nowcast，后续指标将为 N/A。")
            calculated_nowcast_orig = None # 确保出错时为 None
        # --- 结束计算 y_{T(t)|t} 的修改 ---

        # --- (可选) 计算平滑 Nowcast 以供比较 (使用 .x_sm) ---
        logger.info("计算平滑 Nowcast 序列 (使用 .x_sm, 用于对比)...")
        try:
            # --- 修改：直接使用 .x_sm ---
            smoothed_state_attr = None
            if hasattr(final_dfm_results, 'x_sm') and final_dfm_results.x_sm is not None:
                 smoothed_state_attr = final_dfm_results.x_sm
            # --- 结束修改 ---

            if smoothed_state_attr is not None:
                factors_smoothed = smoothed_state_attr
                if not isinstance(factors_smoothed, pd.DataFrame):
                     if isinstance(factors_smoothed, np.ndarray):
                        k_factors_smooth = factors_smoothed.shape[1]
                        # Use index from final_data_processed, ensure length matches
                        index_for_smooth = final_data_processed.index[:len(factors_smoothed)]
                        factors_smoothed = pd.DataFrame(factors_smoothed,
                                                        index=index_for_smooth,
                                                        columns=[f'Factor{i+1}' for i in range(k_factors_smooth)])
                     else:
                         raise TypeError(f"Smoothed factors (x_sm) 不是 DataFrame 或 Array (type: {type(factors_smoothed)})")

                # Ensure alignment for dot product
                common_factors_smooth = factors_smoothed.columns.intersection(target_loadings.index)
                if len(common_factors_smooth) != len(target_loadings):
                    logger.warning(f"平滑状态因子 ({factors_smoothed.columns.tolist()}) 与目标载荷因子 ({target_loadings.index.tolist()}) 不完全匹配。仅使用共同因子: {common_factors_smooth.tolist()}")
                if common_factors_smooth.empty:
                     raise ValueError("平滑状态与目标载荷之间没有共同的因子名称，无法计算平滑 Nowcast。")

                nowcast_smoothed_standardized = factors_smoothed[common_factors_smooth].dot(target_loadings[common_factors_smooth])
                calculated_nowcast_smoothed_orig = (nowcast_smoothed_standardized * target_std) + target_mean
                calculated_nowcast_smoothed_orig.name = "Nowcast_OriginalScale_Smoothed"
                logger.info("平滑 Nowcast 序列计算完成。")
            else:
                logger.warning("结果对象缺少 'x_sm' 属性或其值为 None，无法计算平滑 Nowcast。")
                calculated_nowcast_smoothed_orig = None
        except Exception as e_smooth:
            logger.error(f"计算平滑 Nowcast 时出错: {e_smooth}", exc_info=True)
            calculated_nowcast_smoothed_orig = None
        # --- 结束可选平滑计算 ---


        # --- <<< 新增调试：检查 Nowcast (滤波后) 索引范围 >>> ---
        calculated_nowcast_for_metrics = calculated_nowcast_orig # 使用滤波结果进行评估

        if calculated_nowcast_for_metrics is not None and not calculated_nowcast_for_metrics.empty:
            logger.info(f"  [DEBUG] Filtered Nowcast (for metrics) Index Range: {calculated_nowcast_for_metrics.index.min()} to {calculated_nowcast_for_metrics.index.max()}")
            # --- 根据 training_start_date 过滤 Nowcast (用于报告，而非指标计算) ---
            calculated_nowcast_filtered_by_date = calculated_nowcast_for_metrics # Start with the full series
            if training_start_date:
                try:
                    start_dt = pd.to_datetime(training_start_date)
                    # Ensure index is datetime
                    if not pd.api.types.is_datetime64_any_dtype(calculated_nowcast_for_metrics.index):
                         logger.warning("Nowcast index is not datetime, attempting conversion for date filtering.")
                         temp_index = pd.to_datetime(calculated_nowcast_for_metrics.index, errors='coerce')
                         if temp_index.isna().any():
                             logger.error("Failed to convert Nowcast index to datetime for filtering. Skipping date filter.")
                             # Keep calculated_nowcast_filtered_by_date as the original unfiltered series
                         else:
                             calculated_nowcast_filtered_by_date = calculated_nowcast_for_metrics[temp_index >= start_dt]
                             logger.info(f"Filtered Nowcast series (for reporting) to start from {training_start_date}. Shape: {calculated_nowcast_filtered_by_date.shape}")
                    else:
                        calculated_nowcast_filtered_by_date = calculated_nowcast_for_metrics[calculated_nowcast_for_metrics.index >= start_dt]
                        logger.info(f"Filtered Nowcast series (for reporting) to start from {training_start_date}. Shape: {calculated_nowcast_filtered_by_date.shape}")

                except Exception as e_filter_date:
                    logger.warning(f"Could not filter Nowcast series by start date {training_start_date}: {e_filter_date}. Using original range for reporting.")
                    # Fallback to original if filtering fails - already assigned above
            else:
                logger.info("No training_start_date provided, using full Nowcast series for reporting.")
            # --- 结束过滤 --- <<< Removed trailing backslash causing issues
        else:
            logger.info("  [DEBUG] Filtered Nowcast (for metrics) is None or empty.")
            calculated_nowcast_filtered_by_date = None # Ensure it's None if original is None/empty
        # --- <<< 结束新增 >>> ---

        # --- 3. 计算最终评估指标 (使用滤波后的 Nowcast) ---
        logger.info("计算最终模型的 IS/OOS RMSE 和 Hit Rate (使用 Filtered Nowcast 和 analysis_utils)...")
        original_target_series = all_data_full.get(target_variable)
        if original_target_series is None or original_target_series.empty:
            logger.error("无法计算评估指标：在 all_data_full 中未找到或为空的原始目标序列。")
            metrics = {k: 'N/A' for k in ['is_rmse', 'oos_rmse', 'is_mae', 'oos_mae', 'is_hit_rate', 'oos_hit_rate']}
        elif calculated_nowcast_for_metrics is None or calculated_nowcast_for_metrics.empty: # 使用滤波结果检查
             logger.error("无法计算评估指标：未能成功计算 Filtered Nowcast 序列。")
             metrics = {k: 'N/A' for k in ['is_rmse', 'oos_rmse', 'is_mae', 'oos_mae', 'is_hit_rate', 'oos_hit_rate']}
        else:
            # 确保 nowcast_series 使用的是滤波结果
            logger.debug(f"传递给 calculate_metrics: Nowcast length={len(calculated_nowcast_for_metrics)}, Target length={len(original_target_series)}")
            metrics_raw, aligned_df_for_metrics = calculate_metrics_with_lagged_target(
                nowcast_series=calculated_nowcast_for_metrics, # <--- 确认使用滤波结果
                target_series=original_target_series.copy(),
                validation_start=validation_start_date,
                validation_end=validation_end_date,
                train_end=train_end_date,
                target_variable_name=target_variable # Pass original name here
            )
            metrics = metrics_raw
            logger.info(f"最终模型评估指标 (基于 Filtered Nowcast) 计算完成: {metrics}")
            if aligned_df_for_metrics is not None:
                logger.debug(f"Aligned dataframe for metrics created with shape: {aligned_df_for_metrics.shape}")
            else:
                 logger.warning("Aligned dataframe for metrics was not created.")

            # 🔥 关键修复：立即确保指标被保存到元数据中供UI模块使用
            # 直接在metrics计算完成后立即处理，避免作用域问题
            if metrics and isinstance(metrics, dict):
                logger.info(f"计算得到的标准指标已包含在metrics中: {list(metrics.keys())}")
            else:
                logger.warning("metrics计算失败或为空，UI模块将无法获取标准指标")
        
        # --- 5. 准备写入 Excel --- 
        # ... (加载元数据部分) ...
        logger.info(f"准备写入 Excel 文件: {excel_output_path}")
        metadata_loaded = {} # Initialize empty dict
        try:
             metadata_path = os.path.join(run_output_dir, 'final_dfm_metadata.pkl')
             if os.path.exists(metadata_path):
                 with open(metadata_path, 'rb') as f_meta:
                     metadata_loaded = pickle.load(f_meta) # Corrected indentation
                     logger.info(f"成功加载元数据: {metadata_path}") # Corrected indentation
             else:
                  logger.warning(f"元数据文件未找到: {metadata_path}")
        except Exception as e_meta_load:
            logger.warning(f"无法加载元数据文件以获取 Stage 1 信息: {e_meta_load}. 部分 Summary 信息将为 N/A。")
            # metadata_loaded remains {}

        original_data_file = metadata_loaded.get('original_data_file', 'N/A')

            
        # --- 创建解释内容 (保持不变) ---
        explanation_sheet_name = "指标解释"
        metric_explanations = [
            ["指标", "中文含义", "说明"],
            ["最终 IS RMSE (Filtered)", "样本内均方根误差 (基于滤波值)", "衡量模型在训练期间实时预测(滤波)值与实际值误差的平均大小。越小表示训练拟合越好。"],
            ["最终 OOS RMSE (Filtered)", "样本外均方根误差 (基于滤波值)", "衡量模型在验证期间实时预测(滤波)值与实际值误差的平均大小。越小表示模型泛化能力越强。"],
            ["最终 IS MAE (Filtered)", "样本内平均绝对误差 (基于滤波值)", "衡量模型在训练期间实时预测(滤波)值与实际值误差绝对值的平均大小。"],
            ["最终 OOS MAE (Filtered)", "样本外平均绝对误差 (基于滤波值)", "衡量模型在验证期间实时预测(滤波)值与实际值误差绝对值的平均大小。"],
            ["最终 IS Hit Rate (%) (Filtered)", "样本内命中率 (基于滤波值)", "衡量模型在训练期间预测目标变量实时变化方向(涨/跌)的准确率。"],
            ["最终 OOS Hit Rate (%) (Filtered)", "样本外命中率 (基于滤波值)", "衡量模型在验证期间预测目标变量实时变化方向的准确率。"],
            ["因子对单变量 R²", "(见 R2 Combined Sheet)", "表示单个因子能解释单个预测变量变动的百分比。"],
            ["行业汇总 R²", "(见 R2 Combined Sheet)", "表示所有因子共同能解释特定行业内所有变量整体变动的百分比。"],
            ["因子对行业汇总 R²", "(见 R2 Combined Sheet)", "表示单个因子能解释特定行业内所有变量整体变动的百分比。"],
            ["因子对类型汇总 R²", "(见 R2 Combined Sheet)", "表示单个因子能解释特定类型(e.g., M/Q/W)变量整体变动的百分比。"] # Added Type R2 explanation
        ]
        sheet_explanations = [
            ["Sheet 名称", "主要目的"],
            ["Summary", "提供模型运行总体概览信息(关键参数、评估指标、运行时间等)，并附加因子贡献度、PCA方差解释。"],
            ["Aligned Nowcast vs Target", "展示最终滤波Nowcast预测值与对齐后实际目标值的对比，用于直观评估预测效果。"], # Updated description
            ["Factor Time Series", "展示模型估计出的滤波后因子时间序列，以及最终的滤波Nowcast序列(可选包含平滑Nowcast对比)。"], # Updated description
            ["R2 Analysis Combined", "综合展示R²分析结果(因子对变量、行业、类型)。"],
            ["Variables and Loadings", "列出模型使用的最终预测变量及其类型/行业信息，以及每个变量在每个因子上的载荷。"],
            ["指标解释", "解释 Excel 中各 Sheet 和指标的含义。"] # Added self-explanation
        ]
        explanation_df_metrics = pd.DataFrame(metric_explanations[1:], columns=metric_explanations[0])
        explanation_df_sheets = pd.DataFrame(sheet_explanations[1:], columns=sheet_explanations[0])
            
        with pd.ExcelWriter(excel_output_path, engine='openpyxl', mode='w') as writer:
            # --- Sheet 0: 指标解释 (保持不变) ---
            logger.info(f"  正在写入 '{explanation_sheet_name}' Sheet...")
            try:
                explanation_df_metrics.to_excel(writer, sheet_name=explanation_sheet_name, startrow=1, index=False)
                startrow_sheets = explanation_df_metrics.shape[0] + 3 
                explanation_df_sheets.to_excel(writer, sheet_name=explanation_sheet_name, startrow=startrow_sheets, index=False)
                worksheet_exp = writer.sheets[explanation_sheet_name]
                title_metrics = worksheet_exp.cell(row=1, column=1, value="指标解释")
                title_metrics.font = Font(bold=True)
                title_sheets = worksheet_exp.cell(row=startrow_sheets + 1, column=1, value="Sheet 用途说明")
                title_sheets.font = Font(bold=True)
                format_excel_sheet(worksheet_exp, column_widths={'A': 35, 'B': 30, 'C': 70})
                logger.info(f"  '{explanation_sheet_name}' Sheet 写入完成。")
            except Exception as e_exp_write:
                logger.error(f"写入 '{explanation_sheet_name}' 时出错: {e_exp_write}", exc_info=True)
                 # Write a simple error message to the sheet if possible
                try: pd.DataFrame([f"Error writing explanations: {e_exp_write}"]).to_excel(writer, sheet_name=explanation_sheet_name, index=False, header=False)
                except: pass


            # --- Sheet 1: Summary (使用新 metrics) ---
            logger.info("  正在写入 'Summary' Sheet...")
            # --- 移除/注释掉错误的代码行 ---
            # summary_ws = writer.sheets['Summary'] #<-- 此行导致 KeyError，因为 sheet 此时尚未创建
            # bold_format = writer.book.add_format({'bold': True}) # 这个方式属于 xlsxwriter，不是 openpyxl

            # --- 写入基本信息 ---
            summary_data = {
                'Parameter': [
                    'Target Variable', 'Training Period Start', 'Training Period End', 'Validation Period Start', 'Validation Period End',
                     'Selected Variables Count', 'Selected Factors (Final)', 'Factor Order p (Final)',
                     'Model Estimation Runtime (seconds)',
                     # 'Best Tuning Avg Hit Rate (%)', #<-- 移除
                     # 'Best Tuning Avg MAE', #<-- 移除
                     'Final Model IS RMSE', 'Final Model OOS RMSE',
                     'Final Model IS MAE', 'Final Model OOS MAE', # <-- 添加 MAE
                     'Final Model IS Hit Rate (%)', 'Final Model OOS Hit Rate (%)'
                     ],
                'Value': [
                    target_variable, training_start_date, train_end_date, validation_start_date, validation_end_date,
                    len(best_variables), best_params.get('k_factors_final', best_params.get('k_factors', 'N/A')), best_params.get('factor_order', 1),
                     f"{total_runtime_seconds:.2f}" if total_runtime_seconds is not None else 'N/A',
                     # format_metric_pct(best_avg_hit_rate_tuning) if best_avg_hit_rate_tuning is not None else 'N/A', #<-- 移除
                     # format_metric(best_avg_rmse_tuning) if best_avg_rmse_tuning is not None else 'N/A', #<-- 移除
                     format_metric(metrics.get('is_rmse')), format_metric(metrics.get('oos_rmse')),
                     format_metric(metrics.get('is_mae')), format_metric(metrics.get('oos_mae')), # <-- 添加 MAE
                     format_metric_pct(metrics.get('is_hit_rate')), format_metric_pct(metrics.get('oos_hit_rate'))
                ]
            }
            summary_df = pd.DataFrame(summary_data)
            summary_start_row = 1 # Excel is 1-based, pandas to_excel startrow is 0-based for header

            # --- 首先写入数据，这会创建 'Summary' sheet ---
            summary_df.to_excel(writer, sheet_name='Summary', startrow=summary_start_row, index=False, header=True) # Ensure header is written

            # --- 然后，获取已创建的 sheet 对象 (使用 openpyxl 的 workbook) ---
            summary_ws = writer.book['Summary'] # 获取 worksheet 对象

            # --- 写入标题 (使用 openpyxl 的方式设置字体) ---
            # pandas to_excel 写入时，startrow=1 表示数据从第 2 行开始 (第 1 行是表头)
            # 因此，标题应该写在 Excel 的第 1 行 (row=1)
            title_row_excel = 1
            title_col_excel = 1
            title_cell = summary_ws.cell(row=title_row_excel, column=title_col_excel, value="Model Summary and Performance")
            # 确保 Font 已从 openpyxl.styles 导入
            try:
                title_cell.font = Font(bold=True)
            except ImportError:
                 logger.warning("无法导入 openpyxl.styles.Font，标题将不会加粗。")


            # --- 写入 PCA 相关信息 (如果提供) ---
            # 计算 PCA 表格在 Excel 中的起始行号
            # 数据在 summary_start_row+1 开始，共 len(summary_df) 行数据 + 1 行表头
            pca_title_start_row_excel = title_row_excel + len(summary_df) + 1 + 2 # Title row + data rows + header row + 2 blank rows
            pca_data_start_row_pandas = pca_title_start_row_excel # pandas startrow is the row *before* the header

            # --- 修改：增加对 pca_results_df 类型的检查 ---
            if isinstance(pca_results_df, pd.DataFrame) and not pca_results_df.empty:
            # --- 结束修改 ---
                # --- 检查 pca_results_df 是否包含 'Eigenvalue' 列 ---
                # <<< 修改：确保检查的列名与 analysis_utils.py 中添加的完全一致 >>>
                if '特征值 (Eigenvalue)' in pca_results_df.columns:
                     # 准备显示用的DataFrame
                     pca_results_df_display = pca_results_df.copy()
                     # 格式化数值列
                     pca_results_df_display['特征值 (Eigenvalue)'] = pca_results_df_display['特征值 (Eigenvalue)'].apply(lambda x: format_metric(x, precision=4))
                     pca_results_df_display['解释方差 (%)'] = pca_results_df_display['解释方差 (%)'].apply(lambda x: format_metric_pct(x, precision=2))
                     pca_results_df_display['累计解释方差 (%)'] = pca_results_df_display['累计解释方差 (%)'].apply(lambda x: format_metric_pct(x, precision=2))

                     # --- 写入 PCA 表格 ---
                     # <<< 修改：使用 format_excel_sheet 改进格式化 >>>
                     # pca_results_df_display.to_excel(writer, sheet_name='Summary', startrow=pca_data_start_row_pandas, index=True, index_label="Factor") # pandas startrow
                     pca_results_df_display.to_excel(writer, sheet_name='Summary', startrow=pca_data_start_row_pandas, index=False) # 不写入索引

                     # --- 写入 PCA 标题 (使用 summary_ws) ---
                     pca_title_cell = summary_ws.cell(row=pca_title_start_row_excel, column=1, value="PCA Results")
                     try: pca_title_cell.font = Font(bold=True)
                     except NameError: pass # Font not imported
                     
                     # --- 格式化写入的 PCA 表格 ---
                     try:
                        format_excel_sheet(summary_ws,
                                            column_widths={'A': 15, 'B': 15, 'C': 18, 'D': 20}, # 调整列宽
                                            # number_formats={'B': '0.00%', 'C': '0.00%', 'D': '0.0000'} # 尝试数字格式
                                            )
                     except Exception as e_format:
                        logger.warning(f"格式化 PCA 表格时出错: {e_format}")
                     # <<< 结束修改 >>>
                else:
                    logger.warning("PCA DataFrame provided but missing '特征值 (Eigenvalue)' column. Skipping PCA table write.") # <-- 更新警告信息中的列名
                    # --- 写入占位标题 (使用 summary_ws) ---
                    pca_title_cell = summary_ws.cell(row=pca_title_start_row_excel, column=1, value="PCA Results: (Missing '特征值 (Eigenvalue)' Column)") # <-- 更新占位符文本
                    try: pca_title_cell.font = Font(bold=True)
                    except NameError: pass
            else:
                # --- 修改：明确区分 None 和错误类型 --- 
                if pca_results_df is None:
                    logger.info("未提供 PCA 结果 (pca_results_df is None)，跳过写入 PCA 表格。")
                else:
                    logger.warning(f"提供的 pca_results_df 不是有效的 DataFrame (类型: {type(pca_results_df)})，跳过写入 PCA 表格。")
                # --- 结束修改 ---
                # --- 写入占位标题 (使用 summary_ws) ---
                pca_title_cell = summary_ws.cell(row=pca_title_start_row_excel, column=1, value="PCA Results: (Not Provided or Invalid)") # <-- 修改占位符文本
                try: pca_title_cell.font = Font(bold=True)
                except NameError: pass


            # --- 新增：写入最终特征根值 (如果提供) ---
            # 计算 Eigenvalue 表格的起始行
            # --- 修改：根据 pca_results_df_display 是否存在来计算 pca_table_rows ---
            pca_table_rows = (len(pca_results_df_display) + 1) if 'pca_results_df_display' in locals() and isinstance(pca_results_df_display, pd.DataFrame) else 1 # Data rows + header or just placeholder title row
            # --- 结束修改 ---
            eigenvalue_title_start_row_excel = pca_title_start_row_excel + pca_table_rows + 1 # After PCA table/placeholder + 1 blank row
            eigenvalue_data_start_row_pandas = eigenvalue_title_start_row_excel

            if final_eigenvalues is not None and len(final_eigenvalues) > 0:
                 eigen_df = pd.DataFrame({'Eigenvalue': final_eigenvalues})
                 eigen_df.index.name = 'Component'
                 eigen_df['Eigenvalue'] = eigen_df['Eigenvalue'].apply(lambda x: format_metric(x, precision=6))
                 eigen_df.to_excel(writer, sheet_name='Summary', startrow=eigenvalue_data_start_row_pandas, index=True) # pandas startrow
                 # --- 写入特征根标题 (使用 summary_ws) ---
                 eigen_title_cell = summary_ws.cell(row=eigenvalue_title_start_row_excel, column=1, value="Final Model Eigenvalues (State Transition Matrix)")
                 try: eigen_title_cell.font = Font(bold=True)
                 except NameError: pass
            else:
                 logger.warning("未提供特征根值 (final_eigenvalues)，未添加特征根列。")
                 # --- 写入占位标题 (使用 summary_ws) ---
                 eigen_title_cell = summary_ws.cell(row=eigenvalue_title_start_row_excel, column=1, value="Final Model Eigenvalues: (Not Provided)")
                 try: eigen_title_cell.font = Font(bold=True)
                 except NameError: pass
            # --- 结束新增 ---


            # --- 写入变量转换日志 (如果提供) ---
            # 计算 Transform Log 表格的起始行
            eigenvalue_table_rows = (len(eigen_df) + 1) if final_eigenvalues is not None and len(final_eigenvalues) > 0 else 1
            transform_title_start_row_excel = eigenvalue_title_start_row_excel + eigenvalue_table_rows + 1 # After Eigenvalue table/placeholder + 1 blank row
            transform_data_start_row_pandas = transform_title_start_row_excel

            if final_transform_log:
                # --- 修改：提取更有意义的转换信息 (status) --- 
                applied_transforms_list = []
                # Iterate through the original log to maintain order if needed
                for original_var, transform_details in final_transform_log.items():
                    # Check if a *meaningful* transformation occurred
                    # We might want to show all variables listed, even if status is 'level'?
                    # Let's refine the logic: show entries where status is not 'level' or where the key!=value if value is simple string
                    
                    is_dict_details = isinstance(transform_details, dict)
                    status = transform_details.get('status', 'N/A') if is_dict_details else 'N/A' # Extract status if it's a dict
                    transformed_name = transform_details.get('transformed_name', original_var) if is_dict_details else str(transform_details)

                    # Decide if this transformation is worth showing
                    # Show if: status is not 'level', OR if details are not a dict and original name != transformed name
                    if status != 'level' or (not is_dict_details and original_var != transformed_name):
                        applied_transforms_list.append({
                            'Original Variable': original_var,
                            'Transformation Status': status if status != 'N/A' else 'Renamed/Other',
                            'Resulting Name (if changed)': transformed_name if original_var != transformed_name else original_var
                        })
                        
                if applied_transforms_list:
                    transform_df = pd.DataFrame(applied_transforms_list)[['Original Variable', 'Transformation Status', 'Resulting Name (if changed)']] # Ensure column order
                    # --- 结束修改 ---
                    transform_df.to_excel(writer, sheet_name='Summary', startrow=transform_data_start_row_pandas, index=False) # pandas startrow
                    # --- 写入转换日志标题 (使用 summary_ws) ---
                    transform_title_cell = summary_ws.cell(row=transform_title_start_row_excel, column=1, value="Applied Variable Transformations") # openpyxl row
                else:
                    logger.info("没有适用的变量转换日志需要写入 Summary。")
                    # --- 写入占位标题 (使用 summary_ws) ---
                    transform_title_cell = summary_ws.cell(row=transform_title_start_row_excel, column=1, value="Applied Variable Transformations: (None Applied)")
                    try: transform_title_cell.font = Font(bold=True)
                    except NameError: pass
            else:
                logger.info("未提供变量转换日志。")
                # --- 写入占位标题 (使用 summary_ws) ---
                transform_title_cell = summary_ws.cell(row=transform_title_start_row_excel, column=1, value="Applied Variable Transformations: (Log Not Provided)")
                try: transform_title_cell.font = Font(bold=True)
                except NameError: pass


            # --- 调整 Summary Sheet 列宽 ---
            # (确保 format_excel_sheet 能正确处理 worksheet 对象)
            format_excel_sheet(summary_ws, column_widths={'A': 45, 'B': 45}) # Increased width slightly

            logger.info("  'Summary' Sheet 写入完成。")
            # --- 结束 Summary Sheet ---


            # --- Sheet 2: Aligned Nowcast vs Target (使用滤波结果) ---
            # --- 修改：重命名 Sheet 并使用 create_aligned_nowcast_target_table ---
            # logger.info("  正在写入 'Aligned Nowcast vs Target' Sheet...")
            monthly_sheet_name = "Monthly Forecast vs Target"
            logger.info(f"  正在写入 '{monthly_sheet_name}' Sheet...")
            # --- 移除旧的逻辑 (基于 aligned_df_for_metrics 和错误的索引转换) ---
            # if aligned_df_for_metrics is not None and not aligned_df_for_metrics.empty:
            #     try:
            #         aligned_df_monthly = aligned_df_for_metrics.copy()
            #         ...
            #         aligned_df_to_write.to_excel(writer, sheet_name="Aligned Nowcast vs Target", ...)
            #         ...
            #     except Exception as e_aligned_write:
            #         ...
            # else:
            #     ...
            # --- 结束移除 ---

            # --- 新逻辑：调用 create_aligned_nowcast_target_table --- 
            try:
                if calculated_nowcast_orig is not None and original_target_series is not None:
                    # 确保 create_aligned_nowcast_target_table 函数可用
                    if 'create_aligned_nowcast_target_table' in globals():
                        # 调用函数生成月度对齐表
                        monthly_aligned_df = create_aligned_nowcast_target_table(
                            nowcast_weekly_orig=calculated_nowcast_orig.copy(), # 传入 y_{T(t)|t} 序列
                            target_orig=original_target_series.copy(),
                            target_variable_name=f"{target_variable}_Actual_NextMonth"
                        )
                        
                        if monthly_aligned_df is not None and not monthly_aligned_df.empty:
                             # --- 修改：确保 Nowcast 列名为 'Nowcast_ForecastToMonthEnd' --- 
                             rename_map_monthly = {}
                             if 'Nowcast (Original Scale)' in monthly_aligned_df.columns:
                                 rename_map_monthly['Nowcast (Original Scale)'] = 'Nowcast_ForecastToMonthEnd'
                             monthly_aligned_df_to_write = monthly_aligned_df.rename(columns=rename_map_monthly)
                             # --- 结束修改 --- 

                             # 写入 Excel
                             monthly_aligned_df_to_write.to_excel(writer, sheet_name=monthly_sheet_name, index=True, index_label="Vintage (Last Friday of Month)")

                             # 格式化
                             ws_monthly = writer.sheets[monthly_sheet_name]
                             num_fmt_monthly = {get_column_letter(col_idx): '0.0000'
                                                for col_idx, cell in enumerate(ws_monthly[1], 1) if col_idx > 1}
                             format_excel_sheet(ws_monthly, column_widths={'A': 25}, number_formats=num_fmt_monthly)
                             logger.info(f"  '{monthly_sheet_name}' Sheet 写入完成。")
                        else: # This else belongs to: if monthly_aligned_df is not None...
                             logger.warning(f"  无法写入 '{monthly_sheet_name}': create_aligned_nowcast_target_table 返回空或无效 DataFrame。")
                             pd.DataFrame([[f"无法生成月度对齐数据"]]).to_excel(writer, sheet_name=monthly_sheet_name, index=False, header=False)
                    else: # This else belongs to: if 'create_aligned_nowcast_target_table' in globals()...
                         logger.error(f"无法写入 '{monthly_sheet_name}': 缺少辅助函数 'create_aligned_nowcast_target_table'。")
                         pd.DataFrame([[f"缺少必要的辅助函数"]]).to_excel(writer, sheet_name=monthly_sheet_name, index=False, header=False)
                else: # This else belongs to: if calculated_nowcast_orig is not None...
                    logger.warning(f"  无法写入 '{monthly_sheet_name}': Nowcast 或原始目标序列不可用。")
                    pd.DataFrame([["Nowcast 或原始目标序列不可用"]]).to_excel(writer, sheet_name=monthly_sheet_name, index=False, header=False)
            except Exception as e_monthly_write: # This except belongs to the try starting on line 1123
                logger.error(f"写入 '{monthly_sheet_name}' 时出错: {e_monthly_write}", exc_info=True)
                pd.DataFrame([f"写入月度对齐数据时出错: {e_monthly_write}"]).to_excel(writer, sheet_name=monthly_sheet_name, index=False, header=False)
            # --- 结束新逻辑 ---\


            # --- Sheet 3: Factor Time Series (包括 Filtered Nowcast) ---
            logger.info("  正在写入 'Factor Time Series' Sheet...")
            try:
                # Start with filtered_state if available
                if filtered_state is not None and isinstance(filtered_state, pd.DataFrame) and not filtered_state.empty:
                    factors_to_write = filtered_state.copy()
                    factors_to_write.columns = [f"Factor_{i+1}_Filtered" for i in range(factors_to_write.shape[1])]

                    # --- 修改：合并正确的 Nowcast 序列并使用正确名称 ---
                    # Merge ForecastToMonthEnd Nowcast (use the date-filtered version for reporting if needed, or full)
                    # We use calculated_nowcast_filtered_by_date which is derived from calculated_nowcast_orig
                    if calculated_nowcast_filtered_by_date is not None:
                        # nowcast_col_name = "Nowcast_OriginalScale_Filtered" # <-- 旧名称
                        nowcast_col_name = "Nowcast_ForecastToMonthEnd" # <-- 使用新名称
                        factors_to_write = factors_to_write.merge(
                             calculated_nowcast_filtered_by_date.rename(nowcast_col_name),
                             left_index=True, right_index=True, how='left'
                        )
                    # --- 结束修改 ---

                    # Merge Smoothed Nowcast (if available)
                    if calculated_nowcast_smoothed_orig is not None:
                        smoothed_col_name = "Nowcast_OriginalScale_Smoothed"
                        factors_to_write = factors_to_write.merge(
                            calculated_nowcast_smoothed_orig.rename(smoothed_col_name),
                            left_index=True, right_index=True, how='left'
                        )

                    factors_to_write.to_excel(writer, sheet_name="Factor Time Series", index=True, index_label="Date")
                    # Format sheet
                    ws_factors = writer.sheets["Factor Time Series"]
                    # --- Fix: Correct number_formats creation for ws_factors ---
                    num_fmt_factors = {get_column_letter(col_idx): '0.0000'
                                       for col_idx, cell in enumerate(ws_factors[1], 1) # Iterate header row (index 1)
                                       if col_idx > 1} # Skip first column (Date, index 1)
                    format_excel_sheet(ws_factors, column_widths={'A': 12}, number_formats=num_fmt_factors)

                    logger.info("  'Factor Time Series' Sheet 写入完成。")
                else:
                     logger.warning("  无法写入 'Factor Time Series': filtered_state 不可用或为空。")
                     pd.DataFrame([["无法生成因子时间序列 (可能是滤波失败)"]]).to_excel(writer, sheet_name="Factor Time Series", index=False, header=False)
            except Exception as e_factor_ts:
                 logger.error(f"  写入 'Factor Time Series' 时出错: {e_factor_ts}", exc_info=True)
                 pd.DataFrame([f"写入因子时间序列时出错: {e_factor_ts}"]).to_excel(writer, sheet_name="Factor Time Series", index=False, header=False)


            # --- Sheet 4: R2 Analysis Combined (需要 R2 结果传入) ---
            logger.info("  正在写入 'R2 Analysis Combined' Sheet...")
            try:
                # Ensure the helper function exists or is imported
                if 'write_r2_tables_to_excel' in globals():
                    write_r2_tables_to_excel( # Corrected indentation
                        r2_results=individual_r2_results,
                        excel_writer=writer,
                        sheet_name="R2 Analysis Combined",
                        industry_r2=industry_r2_results,
                        factor_industry_r2=factor_industry_r2_results,
                        factor_type_r2=factor_type_r2_results
                        # dominance_industry_summary=dominance_industry_summary_df # Removed
                    )
                    logger.info("  'R2 Analysis Combined' Sheet 写入完成。") # Corrected indentation
                else: # Corrected indentation level
                    logger.error("辅助函数 'write_r2_tables_to_excel' 未定义或未导入。无法写入 R2 分析。") # Corrected indentation
                    pd.DataFrame([["无法写入 R2 分析: 缺少 'write_r2_tables_to_excel' 函数"]]).to_excel(writer, sheet_name="R2 Analysis Combined", index=False, header=False) # Corrected indentation

            except Exception as e_r2: # Added missing except block
                logger.error(f"  写入 R2 分析结果时出错: {e_r2}", exc_info=True) # Corrected indentation
                pd.DataFrame([f"写入 R2 分析时出错: {e_r2}"]).to_excel(writer, sheet_name="R2 Analysis Combined", index=False, header=False) # Corrected indentation

            # --- Sheet 5: Variables and Loadings ---
            logger.info("  正在写入 'Variables and Loadings' Sheet...")
            try:
                if loadings_df_final is not None and not loadings_df_final.empty:
                    # 1. Create base DataFrame (Loadings)
                    loadings_to_write = loadings_df_final.copy()
                    loadings_to_write.index.name = 'Variable' # This index might be transformed name

                    # 2. Get variable type and industry info
                    # Assume var_type_map and var_industry_map use keys corresponding to loadings_to_write.index
                    # It's crucial that the keys match (e.g., both are transformed names, or a mapping exists)
                    var_types = [var_type_map.get(var, "N/A") for var in loadings_to_write.index] if var_type_map else ["N/A"] * len(loadings_to_write)
                    var_industries = [var_industry_map.get(var, "N/A") for var in loadings_to_write.index] if var_industry_map else ["N/A"] * len(loadings_to_write)

                    # 3. Insert type and industry columns
                    loadings_to_write.insert(0, 'Industry', var_industries)
                    loadings_to_write.insert(0, 'Type', var_types)

                    # 4. Write to Excel (reset index to make 'Variable' a column)
                    loadings_to_write.reset_index().to_excel(writer, sheet_name="Variables and Loadings", index=False)

                    # 5. Format sheet
                    try:
                        ws_loadings = writer.sheets["Variables and Loadings"]
                        num_fmt_loadings = {get_column_letter(col_idx): '0.0000'
                                            for col_idx in range(4, ws_loadings.max_column + 1)}
                        format_excel_sheet(ws_loadings,
                                           number_formats=num_fmt_loadings)
                    except Exception as e_fmt_load:
                        logger.warning(f"格式化 'Variables and Loadings' Sheet 时出错: {e_fmt_load}")
                    logger.info("  'Variables and Loadings' Sheet 写入完成。")
                else:
                    logger.warning("  无法写入 'Variables and Loadings': 最终载荷矩阵为空或不可用。")
                    pd.DataFrame([["无法获取最终载荷矩阵"]]).to_excel(writer, sheet_name="Variables and Loadings", index=False, header=False)
            except Exception as e_loadings_sheet:
                logger.error(f"  写入 'Variables and Loadings' 时出错: {e_loadings_sheet}", exc_info=True)
                pd.DataFrame([f"写入变量和载荷时出错: {e_loadings_sheet}"]).to_excel(writer, sheet_name="Variables and Loadings", index=False, header=False)


        logger.info(f"Excel 文件写入完成: {excel_output_path}")

        # --- 5. 绘制最终图形 (使用滤波结果) ---
        plot_output_dir = os.path.join(run_output_dir, "plots")
        os.makedirs(plot_output_dir, exist_ok=True)
        logger.info(f"开始绘制最终图形到目录: {plot_output_dir}")

        # 5.1 绘制 Nowcast vs Target (使用滤波结果)
        # Use calculated_nowcast_for_metrics (full range) for plotting against target
        if calculated_nowcast_for_metrics is not None and original_target_series is not None:
            logger.info("  绘制 Filtered Nowcast vs Target 图...")
            # --- START FIX for SyntaxError ---
            try: 
                 # Ensure plot function exists or is imported
                 if 'plot_final_nowcast' in globals():
                     plot_final_nowcast(
                         final_nowcast_series=calculated_nowcast_for_metrics, # Use full filtered series for plot
                         target_for_plot=original_target_series.copy(), # Pass original target
                         validation_start=validation_start_date,
                         validation_end=validation_end_date,
                         title=f"Final Filtered Nowcast vs Actual {target_variable}",
                         filename=os.path.join(plot_output_dir, f"{timestamp_str}_final_filtered_nowcast_vs_target.png")
                     )
                     logger.info("  Filtered Nowcast vs Target 图绘制完成。")
                 else: # Corrected indentation level
                      logger.error("无法绘制 Nowcast 图: 'plot_final_nowcast' 函数未定义或未导入。") # Corrected indentation
            except Exception as e_plot_nowcast: # Added missing except block
                 logger.error(f"绘制 Filtered Nowcast vs Target 图时出错: {e_plot_nowcast}", exc_info=True) # Corrected indentation
            # --- END FIX ---
        else:
            logger.warning("无法绘制 Nowcast vs Target 图: Filtered Nowcast 或原始目标序列不可用。")

        # 5.2 绘制因子载荷热力图/聚类图 (使用最终载荷)
        if loadings_df_final is not None and not loadings_df_final.empty:
            logger.info("  绘制最终因子载荷聚类图...")
            try:
                if 'plot_factor_loading_clustermap' in globals():
                    # Exclude target variable row for better visualization of predictor loadings
                    target_name_to_drop = target_variable_in_loadings # Use the name actually in the index
                    if target_name_to_drop in loadings_df_final.index:
                        loadings_plot = loadings_df_final.drop(target_name_to_drop, errors='ignore')
                        logger.debug(f"绘制载荷图时移除目标变量: {target_name_to_drop}")
                    else:
                         logger.warning(f"目标变量 '{target_name_to_drop}' 未在载荷矩阵中找到，绘制所有变量的载荷。")
                         loadings_plot = loadings_df_final.copy()

                    if not loadings_plot.empty:
                        plot_factor_loading_clustermap(
                            loadings_df=loadings_plot,
                            title="Factor Loadings Clustermap (Final Model, Predictors Only)",
                            filename=os.path.join(plot_output_dir, f"{timestamp_str}_final_factor_loadings_clustermap.png"),
                            # top_n_vars=50 # Optional
                        )
                        logger.info("  因子载荷聚类图绘制完成。")
                    else:
                        logger.warning("移除目标变量后没有剩余的预测变量载荷可供绘制。")
                else:
                     logger.error("无法绘制载荷图: 'plot_factor_loading_clustermap' 函数未定义或未导入。")

            except Exception as e_plot_loadings:
                logger.error(f"绘制因子载荷聚类图时出错: {e_plot_loadings}", exc_info=True)
        else:
            logger.warning("无法绘制因子载荷聚类图: 最终载荷矩阵不可用。")

        # 5.3 (可选) 绘制行业-驱动因子图 (如果提供了所需数据)
        # Use filtered_state (which should have the full time index)
        if factor_industry_r2_results and filtered_state is not None and not filtered_state.empty and var_industry_map and all_data_full is not None:
             logger.info("  绘制行业 vs 主要驱动因子图...")
             try:
                 if 'plot_industry_vs_driving_factor' in globals():
                     # Align original data to the filtered_state index for background lines
                     # Be careful if all_data_full has different frequency than filtered_state
                     try:
                         aligned_original_data = all_data_full.reindex(filtered_state.index, method='ffill') # Or appropriate resampling/alignment
                         logger.debug("Aligned original data to filtered state index for plotting.")
                     except Exception as e_align_orig:
                         logger.warning(f"无法将原始数据与滤波状态索引对齐以用于绘图背景: {e_align_orig}。将不绘制背景线。")
                         aligned_original_data = None

                     plot_industry_vs_driving_factor(
                         factor_industry_r2=factor_industry_r2_results,
                         factors_ts=filtered_state, # Use filtered factors
                         data_processed=final_data_processed.reindex(filtered_state.index), # Align processed data too if needed
                         data_original_aligned=aligned_original_data, # Pass aligned original data
                         var_industry_map=var_industry_map,
                         output_dir=plot_output_dir,
                         filename=f"{timestamp_str}_final_industry_driving_factors.png"
                     )
                     logger.info("  行业 vs 主要驱动因子图绘制完成。")
                 else:
                     logger.error("无法绘制行业驱动图: 'plot_industry_vs_driving_factor' 函数未定义或未导入。")

             except Exception as e_plot_industry:
                 logger.error(f"绘制行业 vs 主要驱动因子图时出错: {e_plot_industry}", exc_info=True)
        else:
             missing_comps = []
             if not factor_industry_r2_results: missing_comps.append("factor_industry_r2_results")
             if filtered_state is None or filtered_state.empty: missing_comps.append("filtered_state")
             if not var_industry_map: missing_comps.append("var_industry_map")
             if all_data_full is None: missing_comps.append("all_data_full (for alignment)")
             logger.warning(f"无法绘制行业 vs 主要驱动因子图: 缺少以下一个或多个组件: {', '.join(missing_comps)}。")


        logger.info("最终结果分析和保存完成。")
        # <<< 修改：返回 Nowcast 和 metrics 字典 >>>

        # 将 factor_loadings_df 添加到 metrics 字典
        if loadings_df_final is not None:
            metrics['factor_loadings_df'] = loadings_df_final
            logger.info("Added 'factor_loadings_df' to returned metrics.")
        else:
            metrics['factor_loadings_df'] = None # Ensure key exists
            logger.warning("'loadings_df_final' was None, 'factor_loadings_df' in metrics set to None.")

        # 从 aligned_df_for_metrics 提取 nowcast_aligned 和 y_test_aligned 并添加到 metrics 字典
        if aligned_df_for_metrics is not None and not aligned_df_for_metrics.empty:
            if 'Nowcast' in aligned_df_for_metrics.columns:
                metrics['nowcast_aligned'] = aligned_df_for_metrics['Nowcast']
                logger.info("Added 'nowcast_aligned' to returned metrics.")
            else:
                metrics['nowcast_aligned'] = None # Ensure key exists
                logger.warning("'Nowcast' column not found in 'aligned_df_for_metrics'. 'nowcast_aligned' in metrics set to None.")

            if target_variable in aligned_df_for_metrics.columns:
                metrics['y_test_aligned'] = aligned_df_for_metrics[target_variable]
                logger.info("Added 'y_test_aligned' to returned metrics.")
            else:
                metrics['y_test_aligned'] = None # Ensure key exists
                logger.warning(f"Target variable '{target_variable}' column not found in 'aligned_df_for_metrics'. 'y_test_aligned' in metrics set to None.")
        else:
            metrics['nowcast_aligned'] = None # Ensure key exists
            metrics['y_test_aligned'] = None # Ensure key exists
            logger.warning("'aligned_df_for_metrics' was None or empty. 'nowcast_aligned' and 'y_test_aligned' in metrics set to None.")

        # 🔥 关键修复：标准指标已经包含在metrics中，无需额外处理
        logger.info(f"返回的metrics包含以下键: {list(metrics.keys()) if metrics else 'None'}")
        
        return calculated_nowcast_orig, metrics

    except ValueError as ve:
        logger.error(f"值错误导致分析中止: {ve}", exc_info=True)
        return None, {}
    except TypeError as te:
        logger.error(f"类型错误导致分析中止: {te}", exc_info=True)
        return None, {}
    except AttributeError as ae:
         logger.error(f"属性错误导致分析中止: {ae}", exc_info=True)
         return None, {}
    except NotImplementedError as nie:
         logger.error(f"功能未实现导致分析中止: {nie}", exc_info=True)
         return None, {}
    except Exception as e:
        logger.error(f"分析和保存最终结果时发生意外错误: {e}", exc_info=True)
        # Try to save error message to Excel
        error_df = pd.DataFrame([f"Error during final analysis: {str(e)}"])
        try:
            with pd.ExcelWriter(excel_output_path, engine='openpyxl', mode='w') as writer:
                 error_df.to_excel(writer, sheet_name="ERROR", index=False, header=False)
            logger.info(f"错误信息已尝试写入 Excel 文件: {excel_output_path}")
        except Exception as e_write_err:
             logger.error(f"无法将错误信息写入 Excel: {e_write_err}")
        return None, {}

def get_var_attribute(var_name, mapping_dict, default_value="N/A"):
    """辅助函数：规范化变量名并在字典中查找属性。"""
    if not mapping_dict or not isinstance(mapping_dict, dict):
        return default_value
    # 规范化查询键
    lookup_key = unicodedata.normalize('NFKC', str(var_name)).strip().lower()
    # 尝试直接查找 (假设 mapping_dict 的键已规范化)
    value = mapping_dict.get(lookup_key, None)
    if value is not None and not pd.isna(value):
        return str(value).strip()
    # (可选)如果 mapping_dict 的键未规范化，可以尝试遍历查找
    # for map_key, map_value in mapping_dict.items():
    #     normalized_map_key = unicodedata.normalize('NFKC', str(map_key)).strip().lower()
    #     if normalized_map_key == lookup_key:
    #         if map_value is not None and not pd.isna(map_value):
    #             return str(map_value).strip()
    #     
    return default_value

# --- 新增: 通用 Excel 格式化辅助函数 ---
def format_excel_sheet(worksheet, column_widths: Dict[str, int] = None, number_formats: Dict[str, str] = None):
    """
    自动调整列宽、应用数字格式、设置对齐和标题样式。

    Args:
        worksheet: openpyxl worksheet 对象。
        column_widths: 可选字典，指定特定列的固定宽度 (e.g., {'A': 15, 'C': 10})。
        number_formats: 可选字典，指定特定列标题的数字格式 (e.g., {'RMSE': '0.0000', 'Hit Rate (%)': '0.00%'})。
    """
    logger.debug(f"开始格式化 Sheet: {worksheet.title}")
    thin_border = Border(left=Side(style='thin'), right=Side(style='thin'), top=Side(style='thin'), bottom=Side(style='thin'))
    header_font = Font(bold=True)
    header_fill = PatternFill(start_color="DDEEFF", end_color="DDEEFF", fill_type="solid") # Light blue fill
    center_alignment = Alignment(horizontal='center', vertical='center')
    left_alignment = Alignment(horizontal='left', vertical='center')
    right_alignment = Alignment(horizontal='right', vertical='center')

    # 存储需要特定数字格式的列字母
    col_format_map = {}
    if number_formats:
        for col_idx, cell in enumerate(worksheet[1], 1): # 假设表头在第一行
            if cell.value in number_formats:
                col_letter = get_column_letter(col_idx)
                col_format_map[col_letter] = number_formats[cell.value]
                logger.debug(f"  Sheet '{worksheet.title}', 列 '{cell.value}' ({col_letter}) 将使用格式: {number_formats[cell.value]}")

    for col_idx, column_cells in enumerate(worksheet.columns, 1):
        col_letter = get_column_letter(col_idx)
        max_length = 0

        # 确定数据类型 (简单检查第一行数据)
        is_numeric_col = False
        if len(column_cells) > 1 and isinstance(column_cells[1].value, (int, float, np.number)):
            is_numeric_col = True

        for cell in column_cells:
            # 设置边框
            # cell.border = thin_border # Optional: Apply border to all cells

            # 调整列宽
            try:
                # Check if cell has value and convert to string
                if cell.value is not None:
                    cell_text = str(cell.value)
                    # Add padding based on length
                    padding = 2 if len(cell_text) < 30 else 1
                    max_length = max(max_length, len(cell_text) + padding)
                else:
                    max_length = max(max_length, 4) # Min width for empty cells
            except Exception as e_width:
                logger.warning(f"调整列宽时读取单元格值出错: {e_width}")
                max_length = max(max_length, 8) # Default width on error

            # 设置对齐
            if cell.row == 1: # Header row
                cell.font = header_font
                cell.fill = header_fill
                cell.alignment = center_alignment
            else: # Data rows
                if is_numeric_col:
                    cell.alignment = right_alignment
                else:
                    cell.alignment = left_alignment
            
            # 应用数字格式
            if col_letter in col_format_map and cell.row > 1 and cell.value is not None and is_numeric_col:
                 try:
                     cell.number_format = col_format_map[col_letter]
                 except Exception as e_num_format:
                     logger.warning(f"应用数字格式 '{col_format_map[col_letter]}' 到单元格 {cell.coordinate} 失败: {e_num_format}")

        # 应用计算出的列宽，除非被覆盖
        final_width = max(8, max_length) # Minimum width 8
        if column_widths and col_letter in column_widths:
            final_width = column_widths[col_letter]
            logger.debug(f"  Sheet '{worksheet.title}', 列 {col_letter} 使用指定宽度: {final_width}")
        else:
            logger.debug(f"  Sheet '{worksheet.title}', 列 {col_letter} 自动计算宽度: {final_width}")
        
        # Add a bit extra padding, max width 70
        adjusted_width = min(final_width + 1, 70)
        worksheet.column_dimensions[col_letter].width = adjusted_width

    logger.debug(f"完成格式化 Sheet: {worksheet.title}")
# --- 结束新增 ---

# --- <<< 新增：行业与驱动因子绘图函数 >>> ---
def plot_industry_vs_driving_factor(
    factor_industry_r2: Dict[str, pd.Series],
    factors_ts: pd.DataFrame,
    data_processed: pd.DataFrame, # Keep for reference or potential future use, but primary data for grey lines will be from data_original_aligned
    data_original_aligned: Optional[pd.DataFrame], # <-- NEW: Add parameter for original (or less processed) data
    var_industry_map: Dict[str, str],
    output_dir: str,
    filename: str = "industry_driving_factors.png"
):
    """
    为每个行业绘制其变量时间序列与主要驱动因子时间序列的对比图。
    灰色线条将基于 data_original_aligned 进行 log 处理，绘制在左轴。
    彩色因子线将绘制原始因子序列，绘制在右轴。

    Args:
        factor_industry_r2: 单因子对行业 R2 的结果字典。
        factors_ts: 因子时间序列 DataFrame。
        data_processed: DFM模型使用的最终处理后的变量时间序列 DataFrame。
        data_original_aligned: 对齐频率后、进行平稳性处理之前的变量数据。
        var_industry_map: 变量到行业的映射。
        output_dir: 保存绘图的目录。
        filename: 输出文件名。
    """
    # --- 修改日志信息 ---
    logger.info("开始绘制行业与主要驱动因子对比图 (Log 处理行业变量(左轴), 原始因子(右轴))...")
    # --- 结束修改 ---
    save_path = os.path.join(output_dir, filename)

    # --- <<< 新增：检查输入数据 >>> ---
    if not factor_industry_r2 or not isinstance(factor_industry_r2, dict):
        logger.error("绘图错误：未提供有效的 factor_industry_r2 字典。")
        return
    if factors_ts is None or factors_ts.empty or not isinstance(factors_ts, pd.DataFrame):
        logger.error("绘图错误：未提供有效的 factors_ts 时间序列。")
        return
    # data_processed is no longer strictly required for the grey lines, but check data_original_aligned
    if data_original_aligned is None or data_original_aligned.empty or not isinstance(data_original_aligned, pd.DataFrame):
        logger.error("绘图错误：未提供有效的 data_original_aligned 用于绘制行业变量。")
        return
    if not var_industry_map or not isinstance(var_industry_map, dict):
        logger.error("绘图错误：未提供有效的 var_industry_map 字典。")
        return
    # --- <<< 结束检查 >>> ---

    # ... (rest of the initial setup: getting unique industries, finding driving factor, setting up subplots remains largely the same) ...
    # 1. 获取所有唯一的行业名称
    if not var_industry_map:
         logger.error("绘图错误：var_industry_map 为空，无法确定行业列表。")
         return
    unique_industries = sorted(list(set(var_industry_map.values())))
    logger.info(f"[绘图调试] 找到 {len(unique_industries)} 个唯一行业: {unique_industries[:10]}...")

    # 2. 为每个行业找到驱动因子 (R² 最高的因子)
    industry_to_driving_factor = {}
    industry_max_r2 = {}
    if not factor_industry_r2:
         logger.error("绘图错误：factor_industry_r2 数据缺失，无法确定驱动因子。")
         return

    for industry in unique_industries:
        max_r2 = -np.inf
        driving_factor = None
        for factor_name, r2_series in factor_industry_r2.items():
            if isinstance(r2_series, pd.Series) and industry in r2_series.index:
                current_r2 = r2_series.loc[industry]
                if pd.notna(current_r2) and current_r2 > max_r2:
                    max_r2 = current_r2
                    driving_factor = factor_name
        if driving_factor:
            industry_to_driving_factor[industry] = driving_factor
            industry_max_r2[industry] = max_r2
            logger.debug(f"  行业 '{industry}' 的驱动因子确定为: {driving_factor} (R²={max_r2:.4f})")
        else:
             logger.warning(f"  未能确定行业 '{industry}' 的驱动因子 (可能所有因子 R² 都无效或为负)。")

    if not industry_to_driving_factor:
        logger.error("绘图错误：未能为任何行业确定驱动因子。")
        return

    # 3. 确定子图布局
    industries_to_plot = list(industry_to_driving_factor.keys())
    num_industries_to_plot = len(industries_to_plot)
    ncols = 3
    nrows = int(np.ceil(num_industries_to_plot / ncols))
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(ncols * 6, nrows * 4.5), sharex=True) # Increased height slightly for second axis
    axes = axes.flatten() if nrows * ncols > 1 else [axes]

    # 4. 绘制子图
    # --- <<< CHANGE: Use data_original_aligned columns for variable finding >>> ---
    all_vars = data_original_aligned.columns.tolist()
    # (var_map_normalized and factor_color_map remain the same)
    var_map_normalized = {unicodedata.normalize('NFKC', str(k)).strip().lower(): v
                          for k, v in var_industry_map.items()} if var_industry_map else {}
    factor_colors = plt.cm.viridis(np.linspace(0, 1, factors_ts.shape[1]))
    factor_color_map = {col: factor_colors[i] for i, col in enumerate(factors_ts.columns)}

    plot_idx = 0
    plotted_industries_count = 0
    for industry in industries_to_plot:
        logger.info(f"  [Plot Loop] 开始处理行业: {industry}") # <-- 新增日志
        if plot_idx >= len(axes):
            logger.warning("子图数量不足以绘制所有行业，部分行业将被省略。")
            break

        ax = axes[plot_idx] # Primary Y-axis (Left)
        driving_factor = industry_to_driving_factor[industry]

        # --- <<< CHANGE: Find variables based on var_industry_map and all_vars from data_original_aligned >>> ---
        industry_vars = []
        # logger.info(f"-- [绘图调试] 开始查找行业 '{industry}' 的变量 (从 data_original_aligned) --")
        vars_checked_count = 0
        for var in all_vars:
            vars_checked_count += 1
            mapped_industry = get_var_attribute(var, var_industry_map, default_value=None)
            if mapped_industry == industry:
                industry_vars.append(var)
        # logger.info(f"-- [绘图调试] 行业 '{industry}' 查找结束. 共检查 {vars_checked_count} 个变量. 找到 {len(industry_vars)} 个匹配变量: {industry_vars[:10]}... --")

        if not industry_vars:
             # Handle case where no variables are found (same as before)
             logger.warning(f"行业 '{industry}' 未找到对应变量，在图中保留空位。")
             ax.set_title(f"行业: {industry}\n(无变量数据)")
             ax.tick_params(axis='both', which='both', bottom=False, top=False, left=False, right=False, labelbottom=False, labelleft=False)
        else:
             # --- <<< START MAJOR CHANGE: Process data_original_aligned using Log ONLY >>> ---
             # logger.debug(f"  行业 '{industry}': 对找到的 {len(industry_vars)} 个原始变量执行 Log ...")
             industry_vars_processed_temp = pd.DataFrame(index=data_original_aligned.index) # Initialize empty df with correct index
             processed_vars_count = 0
             for var in industry_vars:
                 if var in data_original_aligned.columns:
                     series = data_original_aligned[var].copy()
                     
                     # 1. Log transform (handle non-positive values)
                     series_log = series.copy()
                     non_positive_mask = series_log <= 0
                     if non_positive_mask.any():
                         # logger.warning(f"    变量 '{var}' 包含 {non_positive_mask.sum()} 个非正值，将在 Log 转换中设为 NaN。")
                         series_log[non_positive_mask] = np.nan
                     series_log = np.log(series_log)
                     
                     # Remove the differencing step
                     # series_log_diff = series_log.diff(1)
                     
                     # Check if result is valid before adding
                     if not series_log.isnull().all(): # Check log result directly
                         industry_vars_processed_temp[var] = series_log # Add log-transformed series
                         processed_vars_count += 1
                     # else:
                         # logger.warning(f"    变量 '{var}' Log 转换后结果全为 NaN，跳过添加。")

             # logger.debug(f"  行业 '{industry}': 成功处理了 {processed_vars_count} / {len(industry_vars)} 个变量。")

             # Check if any variables were successfully processed
             if industry_vars_processed_temp.empty or industry_vars_processed_temp.isnull().all().all():
                  logger.warning(f"  行业 '{industry}': Log 转换后没有有效的变量数据，跳过绘图。")
                  ax.set_title(f"行业: {industry}\n(处理后无有效数据)")
                  ax.tick_params(axis='both', which='both', bottom=False, top=False, left=False, right=False, labelbottom=False, labelleft=False)
             else:
                 # --- 移除计算处理后行业平均序列和统计量的部分 --- 
                 # ... [Removed code for calculating mean/std] ...

                 # --- 绘制处理后的行业变量曲线 (灰色) on primary axis --- 
                 lines_ax1 = []
                 labels_ax1 = []
                 num_plotted = 0
                 # logger.debug(f"    [Plot Loop - {industry}] 准备绘制 Log 变量线...") # <-- 新增
                 for var in industry_vars_processed_temp.columns: # Iterate through successfully processed vars
                     line, = ax.plot(industry_vars_processed_temp.index, industry_vars_processed_temp[var],
                                     color='grey', alpha=0.7, linewidth=1.0)
                     if num_plotted == 0: # Only add label once for grey lines
                         lines_ax1.append(line)
                         labels_ax1.append("Log(变量)")
                     num_plotted += 1
                 # logger.debug(f"    [Plot Loop - {industry}] 完成绘制 Log 变量线 ({num_plotted} 条)。") # <-- 新增
                 # --- 结束绘制变量曲线 --- 

                 # --- <<< 修改：创建次轴并绘制原始因子 >>> ---
                 # logger.debug(f"    [Plot Loop - {industry}] 准备创建次轴并绘制因子...") # <-- 新增
                 ax2 = ax.twinx() # Create secondary Y-axis (Right)
                 lines_ax2 = []
                 labels_ax2 = []
                 if driving_factor in factors_ts.columns:
                     factor_color = factor_color_map.get(driving_factor, 'black')
                     factor_ts_series = factors_ts[driving_factor].copy()
                     
                     # Plot the original factor directly on the secondary axis
                     if not factor_ts_series.isnull().all():
                         line, = ax2.plot(factor_ts_series.index, factor_ts_series,
                                          color=factor_color, linestyle='-',
                                          linewidth=1.5, label=f"{driving_factor} (右轴)") # Update label
                         lines_ax2.append(line)
                         labels_ax2.append(f"{driving_factor} (右轴)")
                         # logger.debug(f"      因子 '{driving_factor}' 已绘制在右轴。")
                         ax2.set_ylabel(f"因子值", fontsize=9, color=factor_color)
                         ax2.tick_params(axis='y', labelcolor=factor_color)
                     else:
                         # logger.warning(f"      因子 '{driving_factor}' 全为 NaN，无法绘制。")
                         ax2.set_ylabel(f"因子值", fontsize=9)
                         ax2.tick_params(axis='y')
                 else:
                     # logger.warning(f"    行业 '{industry}' 的驱动因子 '{driving_factor}' 不在因子时间序列中。")
                     ax2.set_ylabel(f"因子值", fontsize=9)
                     ax2.tick_params(axis='y')
                 # logger.debug(f"    [Plot Loop - {industry}] 完成次轴和因子绘制。") # <-- 新增
                 # --- <<< 结束次轴和因子绘制 >>> ---

                 # --- 设置此子图的标题、标签、网格、图例 --- 
                 ax.set_title(f"行业: {industry}\n(驱动因子: {driving_factor}) R2={industry_max_r2[industry]:.2f}")
                 # --- 修改主 Y 轴标签 ---
                 ax.set_ylabel("Log(变量值) (左轴)", fontsize=9)
                 # --- 结束修改主 Y 轴标签 ---
                 ax.grid(True, linestyle=':', alpha=0.5)
                 # --- 合并图例 --- 
                 lines = lines_ax1 + lines_ax2
                 labels = labels_ax1 + labels_ax2
                 # logger.debug(f"    [Plot Loop - {industry}] 准备设置图例...") # <-- 新增
                 ax.legend(lines, labels, loc='upper right', fontsize='small')
                 plotted_industries_count += 1
                 # logger.debug(f"    [Plot Loop - {industry}] 图例设置完成。") # <-- 新增
                 # --- 结束设置子图 --- 
             # --- <<< END MAJOR CHANGE >>> --- 
        logger.info(f"  [Plot Loop] 完成处理行业: {industry}") # <-- 新增日志
        # 无论是否绘制了数据，都要移动到下一个子图位置
        plot_idx += 1

    # 5. 隐藏未使用的子图轴
    logger.info("隐藏未使用的子图轴...") # <-- 新增日志
    for i in range(plot_idx, len(axes)):
        axes[i].axis('off')

    # 6. 调整整体布局和添加主标题
    # --- 修改主标题 ---
    logger.info("设置主标题...") # <-- 新增日志
    plt.suptitle("各行业(Log)与主要驱动因子(原始值)对比", fontsize=16, y=1.02)
    # --- 结束修改主标题 ---
    logger.info("调整布局 (tight_layout)...") # <-- 新增日志
    plt.tight_layout(rect=[0, 0.03, 1, 0.98]) # Adjust layout to prevent title overlap
    logger.info("布局调整完成。") # <-- 新增日志

    # 7. 保存图像
    logger.info(f"即将保存行业驱动因子图到: {save_path}") # <-- 新增日志 (重要)
    try:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"行业与驱动因子对比图已保存至: {save_path}") # <-- 这是成功保存的消息
    except Exception as e:
        logger.error(f"保存行业与驱动因子对比图时出错: {e}")
    finally:
        logger.info("关闭绘图对象...") # <-- 新增日志
        plt.close(fig) # 关闭图形，释放内存

    logger.info("绘制行业与驱动因子对比图完成。")
# --- <<< 结束新增/修改 >>> ---

# <<<-------------------- 新增因子载荷聚类热力图函数 -------------------->>>
def plot_factor_loading_clustermap(
    loadings_df: pd.DataFrame,
    title: str,
    filename: str,
    figsize: Tuple[int, int] = (12, 10), # 调整默认大小
    cmap: str = "coolwarm", # <<< 修改：默认使用发散色图
    annot: Optional[bool] = None, # <<< 修改：允许自动或手动设置
    fmt: str = ".2f", # 数值格式
    row_cluster: bool = True,
    col_cluster: bool = True,
    top_n_vars: Optional[int] = None, # <<< 新增：用于筛选 Top N 变量
    center: Optional[float] = 0 # <<< 新增：设置颜色中心
):
    """
    生成因子载荷的聚类热力图 (Clustermap)，可选只显示 Top N 变量。

    Args:
        loadings_df (pd.DataFrame): 因子载荷 DataFrame (行为变量, 列为因子)。
        title (str): 图表标题。
        filename (str): 保存图表的文件路径。
        figsize (Tuple[int, int], optional): 图表尺寸. Defaults to (12, 10).
        cmap (str, optional): Matplotlib 色图名称. Defaults to "coolwarm".
        annot (Optional[bool], optional): 是否在单元格中显示数值。如果为 None，则在 top_n_vars 生效时自动设为 True。
        fmt (str, optional): 数值显示格式. Defaults to ".2f".
        row_cluster (bool, optional): 是否对行进行聚类. Defaults to True.
        col_cluster (bool, optional): 是否对列进行聚类. Defaults to True.
        top_n_vars (Optional[int], optional): 如果指定，则只显示每个因子下载荷绝对值最高的 N 个变量（合并去重后的子集）。Defaults to None.
        center (Optional[float], optional): 色图的中心值。对于发散色图，通常设为 0。Defaults to 0.
    """
    logger.info(f"\\n[绘图函数] 开始生成因子载荷聚类热力图: {filename}...")
    if loadings_df.empty:
        logger.warning("因子载荷数据为空，无法生成聚类热力图。")
        return

    plot_df = loadings_df.copy()
    
    # 检查数据是否足够进行聚类分析
    min_rows_for_clustering = 2
    min_cols_for_clustering = 2
    
    if plot_df.shape[0] < min_rows_for_clustering or plot_df.shape[1] < min_cols_for_clustering:
        logger.warning(f"数据维度太小 {plot_df.shape}，无法进行聚类分析。需要至少 {min_rows_for_clustering} 行和 {min_cols_for_clustering} 列。")
        return

    # --- <<< 新增：筛选 Top N 变量 >>> ---
    if top_n_vars is not None and top_n_vars > 0:
        logger.info(f"筛选每个因子 Top {top_n_vars} 绝对载荷的变量...")
        top_vars_set = set()
        for factor in plot_df.columns:
            try:
                # 计算绝对载荷并找到 Top N 的索引（变量名）
                top_indices = plot_df[factor].abs().nlargest(top_n_vars).index
                top_vars_set.update(top_indices)
            except Exception as e_topn:
                logger.warning(f"为因子 {factor} 筛选 Top {top_n_vars} 变量时出错: {e_topn}")
        
        if not top_vars_set:
            logger.error("未能成功筛选出任何 Top N 变量，无法生成过滤后的热力图。")
            return
            
        filtered_vars = sorted(list(top_vars_set))
        logger.info(f"筛选后共保留 {len(filtered_vars)} 个唯一变量用于绘图。")
        plot_df = plot_df.loc[filtered_vars]
        
        # 筛选后再次检查数据是否足够进行聚类
        if plot_df.shape[0] < min_rows_for_clustering or plot_df.shape[1] < min_cols_for_clustering:
            logger.warning(f"筛选后数据维度太小 {plot_df.shape}，无法进行聚类分析。")
            return
        
        # 如果筛选生效，且 annot 未被显式设为 False，则自动开启 annot
        if annot is None:
            annot = True
            logger.info("由于已筛选 Top N 变量，自动启用数值标注 (annot=True)。")
        # 调整 figsize (可选，行数减少，可以减少高度)
        # new_height = max(6, min(15, len(filtered_vars) * 0.3)) # 动态调整高度示例
        # figsize = (figsize[0], new_height)
        # logger.info(f"调整 figsize 为: {figsize}")
        
    elif annot is None: # 如果未筛选且 annot 为 None，则根据总行数判断是否开启
         if len(plot_df) <= 50: # 行数不多时默认开启 annot
              annot = True
         else: # 行数多时默认关闭
              annot = False
              logger.info("变量数量较多 (>50) 且未指定 top_n_vars，默认禁用数值标注。")
    # --- <<< 结束筛选 >>> ---

    try:
        # 最终检查数据有效性
        if plot_df.empty or plot_df.isnull().all().all():
            logger.warning("绘图数据为空或全为NaN值，无法生成聚类热力图。")
            return
            
        # 再次确认数据维度足够
        if plot_df.shape[0] < min_rows_for_clustering or plot_df.shape[1] < min_cols_for_clustering:
            logger.warning(f"最终数据维度 {plot_df.shape} 不足以进行聚类分析。")
            return
        
        # 检查行数，如果最终行数过多且 annot 仍为 True，给出警告 (虽然筛选后不太可能)
        if len(plot_df) > 75 and annot:
            logger.warning("变量数量仍然较多 (>75)，数值标注可能会重叠。")
            # annot = False # 可以考虑强制关闭

        # 使用 seaborn 的 clustermap 同时进行聚类和绘图
        cluster_map = sns.clustermap(
            plot_df, # 使用筛选后的 DataFrame
            figsize=figsize,
            cmap=cmap,
            annot=annot, # 使用最终确定的 annot 值
            fmt=fmt,
            center=center, # <<< 新增：设置颜色中心
            linewidths=.5,
            linecolor='lightgray',
            row_cluster=row_cluster,
            col_cluster=col_cluster,
            dendrogram_ratio=(.15, .2) # 调整树状图比例，行减少可以适当减小行树状图比例
        )

        # 调整字体大小
        plt.setp(cluster_map.ax_heatmap.get_xticklabels(), rotation=45, ha='right', fontsize=8)
        # 调整 Y 轴标签字体大小，根据行数决定，行少字可以大点
        ytick_fontsize = 8 if len(plot_df) > 30 else 10
        plt.setp(cluster_map.ax_heatmap.get_yticklabels(), rotation=0, fontsize=ytick_fontsize)

        # 添加主标题
        plt.suptitle(title, y=1.03, fontsize=14) # 调整 y 位置

        # 保存图像
        cluster_map.savefig(filename, dpi=300, bbox_inches='tight')
        logger.info(f"因子载荷聚类热力图已保存至: {filename}")

        # 关闭图形，释放内存
        plt.close('all')

    except Exception as e:
        logger.error(f"生成因子载荷聚类热力图时出错: {e}", exc_info=True)
        plt.close('all')

# <<<-------------------- 结束新增/修改函数 -------------------->>>

# <<< --- 新增因子载荷稳定性对比函数 --- >>>
def plot_aligned_loading_comparison(
    lambda_full: pd.DataFrame,
    lambda_train: pd.DataFrame,
    variables: List[str],
    output_path: str,
    threshold: float = 0.1,
    figsize: Tuple[int, int] = (15, 25) # 调整 figsize 以适应更多变量
) -> None:
    """
    对齐两个因子载荷矩阵（例如，全样本 vs 仅训练期）并绘制对比图。

    Args:
        lambda_full: 全样本（或基准）因子载荷 DataFrame (Variables x Factors)。
        lambda_train: 训练期（或比较对象）因子载荷 DataFrame (Variables x Factors)。
        variables: 变量名列表，应与载荷矩阵的索引匹配。
        output_path: 图像输出路径。
        threshold: 仅绘制绝对载荷值大于此阈值的变量。
        figsize: Matplotlib 图形大小。
    """
    logger = logging.getLogger(__name__) # Use module logger or get a new one
    logger.info(f"开始绘制因子载荷稳定性对比图，保存至: {output_path}")

    # --- 数据验证和准备 ---
    if not isinstance(lambda_full, pd.DataFrame):
        # 如果是 numpy array, 尝试转换
        if isinstance(lambda_full, np.ndarray) and lambda_full.ndim == 2 and lambda_full.shape[0] == len(variables):
            logger.warning("lambda_full 是 NumPy 数组，将尝试转换为 DataFrame。")
            try:
                lambda_full = pd.DataFrame(lambda_full, index=variables, columns=[f"F{i+1}_Full_tmp" for i in range(lambda_full.shape[1])])
            except Exception as e_conv:
                logger.error(f"转换 lambda_full 到 DataFrame 失败: {e_conv}")
                return
        else:
            logger.error("全样本载荷 lambda_full 必须是 Pandas DataFrame 或可转换的 NumPy 数组。")
            return

    if not isinstance(lambda_train, pd.DataFrame):
        if isinstance(lambda_train, np.ndarray) and lambda_train.ndim == 2 and lambda_train.shape[0] == len(variables):
            logger.warning("lambda_train 是 NumPy 数组，将尝试转换为 DataFrame。")
            try:
                lambda_train = pd.DataFrame(lambda_train, index=variables, columns=[f"F{i+1}_Train_tmp" for i in range(lambda_train.shape[1])])
            except Exception as e_conv:
                 logger.error(f"转换 lambda_train 到 DataFrame 失败: {e_conv}")
                 return
        else:
            logger.error("训练期载荷 lambda_train 必须是 Pandas DataFrame 或可转换的 NumPy 数组。")
            return

    # 再次检查行数与变量数
    if lambda_full.shape[0] != len(variables) or lambda_train.shape[0] != len(variables):
         logger.error(f"转换后，载荷矩阵的行数仍与变量列表长度不匹配。 "
                      f"Lambda Full: {lambda_full.shape[0]}, Lambda Train: {lambda_train.shape[0]}, Variables: {len(variables)}")
         return

    # 确保索引是变量名
    lambda_full.index = variables
    lambda_train.index = variables

    k_factors = lambda_full.shape[1]
    if lambda_train.shape[1] != k_factors:
        logger.warning(f"两个载荷矩阵的因子数量不匹配 (Full: {k_factors}, Train: {lambda_train.shape[1]})。将尝试使用较少的因子数量进行对齐。")
        k_factors = min(k_factors, lambda_train.shape[1])
        lambda_full = lambda_full.iloc[:, :k_factors].copy() # Use .copy()
        lambda_train = lambda_train.iloc[:, :k_factors].copy() # Use .copy()

    # 重命名列以明确来源
    lambda_full.columns = [f"Factor{i+1}_Full" for i in range(k_factors)]
    lambda_train.columns = [f"Factor{i+1}_Train" for i in range(k_factors)]

    # --- 因子对齐 (使用匈牙利算法) ---
    lambda_train_aligned = None # Initialize
    try:
        # 计算成本矩阵 (负相关性，因为我们要最大化相关性)
        # Ensure no NaN values interfere with correlation calculation
        common_index = lambda_full.dropna().index.intersection(lambda_train.dropna().index)
        if len(common_index) < len(variables):
            logger.warning(f"载荷矩阵包含 NaN 值，仅在 {len(common_index)} 个共同非 NaN 变量上计算相关性。")
        if len(common_index) < 2:
             logger.error("计算相关性的共同有效变量不足 (<2)。无法执行因子对齐。")
             raise ValueError("Not enough common valid variables for correlation.")
             
        lambda_full_common = lambda_full.loc[common_index]
        lambda_train_common = lambda_train.loc[common_index]
        
        correlation_matrix = np.corrcoef(lambda_full_common.T, lambda_train_common.T)
        # 提取 Full vs Train 的相关性部分
        cost_matrix = -correlation_matrix[:k_factors, k_factors:]

        # 应用匈牙利算法找到最佳匹配
        row_ind, col_ind = linear_sum_assignment(cost_matrix)
        logger.info(f"因子对齐完成。匹配索引 (Full -> Train): {list(zip(row_ind, col_ind))}")

        # 根据匹配结果重新排列训练期因子列
        aligned_train_cols = [f"Factor{j+1}_Train" for j in col_ind]
        lambda_train_aligned = lambda_train[aligned_train_cols].copy()
        
        # --- 翻转符号以匹配 --- 
        # 检查匹配因子对之间的主要变量载荷符号是否一致
        for full_idx, train_aligned_idx in zip(row_ind, col_ind):
            full_factor_name = f"Factor{full_idx+1}_Full"
            train_factor_name = f"Factor{train_aligned_idx+1}_Train" # Original column name from lambda_train
            train_aligned_col_name = f"Factor{full_idx+1}_TrainAligned" # Target column name in lambda_train_aligned
            
            # 计算这对因子在共同变量上的相关性
            corr_val = np.corrcoef(lambda_full_common[full_factor_name], lambda_train_common[train_factor_name])[0, 1]
            logger.debug(f"检查对齐: {full_factor_name} vs {train_factor_name} (原始列 {train_aligned_idx+1}), 相关性: {corr_val:.3f}")
            if corr_val < 0:
                 logger.info(f"  因子 {full_idx+1} (Full) 与其匹配的训练期因子 {train_aligned_idx+1} 呈负相关。将翻转训练期因子 {train_aligned_idx+1} 的符号。")
                 lambda_train_aligned.iloc[:, list(col_ind).index(train_aligned_idx)] *= -1 # 翻转对齐后 DF 中对应的列
        # --- 结束符号翻转 --- 
        
        # 重命名对齐后的训练期因子列，以匹配全样本因子编号，并添加后缀
        lambda_train_aligned.columns = [f"Factor{i+1}_TrainAligned" for i in row_ind]
        # 确保顺序与 full 一致 (重要！因为 row_ind 可能不是 0, 1, 2...)
        final_aligned_cols = [f"Factor{i+1}_TrainAligned" for i in range(k_factors)]
        lambda_train_aligned = lambda_train_aligned.reindex(columns=final_aligned_cols)
        logger.debug(f"Aligned and renamed train columns: {lambda_train_aligned.columns.tolist()}")

    except Exception as e_align:
        logger.error(f"因子对齐过程中出错: {e_align}", exc_info=True)
        # 如果对齐失败，则不进行对齐，直接使用原始顺序比较
        lambda_train_aligned = lambda_train.copy()
        lambda_train_aligned.columns = [f"{col.replace('_Train', '')}_Train_Unaligned" for col in lambda_train_aligned.columns] # Add suffix
        logger.warning("由于对齐失败或错误，将按原始顺序比较因子 (列名已添加 '_Unaligned' 后缀)。")


    # --- 绘图 --- 
    n_factors = k_factors
    n_cols = 2 # 每行放两个因子对比图
    n_rows = (n_factors + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize, squeeze=False) # squeeze=False 保证 axes 是二维数组
    fig.suptitle('Factor Loading Comparison (Full Sample vs. Training Sample - Aligned)', fontsize=16, y=1.02)

    for i in range(n_factors):
        row = i // n_cols
        col = i % n_cols
        ax = axes[row, col]

        factor_label = f"Factor{i+1}"
        full_col = f"{factor_label}_Full"
        # --- 修改：处理对齐失败的列名 --- 
        aligned_train_col_name_aligned = f"{factor_label}_TrainAligned"
        aligned_train_col_name_unaligned = f"{factor_label}_Train_Unaligned"
        
        if aligned_train_col_name_aligned in lambda_train_aligned.columns:
            aligned_train_col = aligned_train_col_name_aligned
            train_label = "Training Sample (Aligned)"
        elif aligned_train_col_name_unaligned in lambda_train_aligned.columns:
            aligned_train_col = aligned_train_col_name_unaligned
            train_label = "Training Sample (Unaligned)"
        else:
             logger.error(f"无法在 lambda_train_aligned 中找到因子 {i+1} 的列。可用列: {lambda_train_aligned.columns.tolist()}")
             ax.text(0.5, 0.5, f'Error: Cannot find\ncolumn for Factor {i+1}\nin aligned train data.', ha='center', va='center', fontsize=10, color='red')
             ax.set_title(f'{factor_label} Comparison - Error')
             continue # 跳过这个子图
        # --- 结束修改 ---
        
        # 合并两个 Series 并筛选
        comparison_df = pd.DataFrame({
            'Full Sample': lambda_full[full_col],
            train_label: lambda_train_aligned[aligned_train_col]
        })
        # 筛选绝对值大于阈值的变量 (在任一样本中)
        filtered_df = comparison_df[
            (comparison_df['Full Sample'].abs() > threshold) |
            (comparison_df[train_label].abs() > threshold)
        ].copy() # Use .copy() to avoid SettingWithCopyWarning
        
        # 按全样本载荷的绝对值降序排序 (如果全样本载荷为 NaN 则移到后面)
        filtered_df['abs_full'] = filtered_df['Full Sample'].abs()
        filtered_df = filtered_df.sort_values(by='abs_full', ascending=False, na_position='last')
        filtered_df = filtered_df.drop(columns='abs_full')
        
        if filtered_df.empty:
            ax.text(0.5, 0.5, 'No variables above threshold', ha='center', va='center', fontsize=12)
            ax.set_title(f'{factor_label} Comparison')
            ax.set_xticks([])
            ax.set_yticks([])
            continue

        # 绘制并排条形图
        filtered_df.plot(kind='barh', ax=ax, width=0.8)

        ax.set_title(f'{factor_label} Comparison')
        ax.set_xlabel('Loading Value')
        ax.set_ylabel('Variable')
        ax.invert_yaxis() # 让绝对值最大的在上面
        ax.legend(title='Sample Period')
        ax.grid(axis='x', linestyle='--', alpha=0.7)
        ax.axvline(0, color='black', linewidth=0.8) # Add vertical line at 0

    # 如果因子数量不是 n_cols 的整数倍，隐藏多余的子图
    for j in range(n_factors, n_rows * n_cols):
        row = j // n_cols
        col = j % n_cols
        if row < axes.shape[0] and col < axes.shape[1]: # Check bounds
            fig.delaxes(axes[row, col])

    plt.tight_layout(rect=[0, 0, 1, 1.0]) # 调整布局以适应标题
    try:
        plt.savefig(output_path, bbox_inches='tight', dpi=150) # 增加 DPI
        logger.info(f"因子载荷稳定性对比图已成功保存至: {output_path}")
    except Exception as e:
        logger.error(f"保存因子载荷稳定性对比图时出错: {e}", exc_info=True)
    finally:
        plt.close(fig) # 关闭图形，释放内存

# <<< --- 结束新增 --- >>>

# <<< 新增：将 write_single_table 移到模块级别 >>>
def write_single_table(ws, df, title, start_r, start_c, bold_f, number_format='0.0000'):
    """辅助函数：将单个 DataFrame 写入 Excel Sheet 中的指定位置。"""
    print(f"  正在写入表格: '{title}' (开始于 R{start_r}C{start_c})...")
    max_c_written = start_c - 1
    try:
        title_cell = ws.cell(row=start_r, column=start_c, value=title)
        title_cell.font = bold_f
        current_r = start_r + 1
    except Exception as e_title:
        print(f"    错误: 写入表格标题 '{title}' 时出错: {e_title}")
        return start_r
    try:
        index_header = df.index.name if df.index.name else "Index"
        ws.cell(row=current_r, column=start_c, value=index_header).font = bold_f
        max_c_written = start_c
        for c_idx, col_name in enumerate(df.columns):
            col_c = start_c + 1 + c_idx
            ws.cell(row=current_r, column=col_c, value=col_name).font = bold_f
            max_c_written = col_c
        current_r += 1
    except Exception as e_header:
        print(f"    错误: 写入表格 '{title}' 的表头时出错: {e_header}")
        return start_r + 1
    try:
        for r_idx, index_val in enumerate(df.index):
            data_r = current_r + r_idx
            ws.cell(row=data_r, column=start_c, value=index_val)
            for c_idx, col_name in enumerate(df.columns):
                col_c = start_c + 1 + c_idx
                value = df.iloc[r_idx, c_idx]
                cell = ws.cell(row=data_r, column=col_c)
                if isinstance(value, (float, np.number)) and (np.isnan(value) or np.isinf(value)):
                    cell.value = None
                else:
                    cell.value = value
                cell.number_format = number_format
        final_row = current_r + len(df) - 1
    except Exception as e_data:
        print(f"    错误: 写入表格 '{title}' 的数据时出错: {e_data}")
        return current_r
    try:
        col_letter = get_column_letter(start_c)
        index_header = df.index.name if df.index.name else "Index"
        # Ensure index values are converted to string for length calculation
        index_lengths = df.index.astype(str).map(len)
        max_len_index = max(len(str(index_header)), index_lengths.max() if not index_lengths.empty else 0) + 2 # Handle empty index
        ws.column_dimensions[col_letter].width = max(max_len_index, 15)
        for c_idx, col_name in enumerate(df.columns):
            col_c = start_c + 1 + c_idx
            col_letter = get_column_letter(col_c)
            if number_format.endswith('%'):
                col_data_str = df.iloc[:, c_idx].apply(lambda x: f"{x*100:.2f}%" if pd.notna(x) else "")
            elif '0' in number_format:
                num_decimals = number_format.count('0', number_format.find('.')) if '.' in number_format else 0
                col_data_str = df.iloc[:, c_idx].apply(lambda x: f"{x:.{num_decimals}f}" if pd.notna(x) else "")
            else:
                col_data_str = df.iloc[:, c_idx].astype(str)
            # Handle case where col_data_str might be empty after formatting/filtering
            max_len_data = col_data_str.map(len).max() if not col_data_str.empty else 0 
            if pd.isna(max_len_data): max_len_data = 6
            header_len = len(str(col_name))
            adjusted_width = max(max_len_data, header_len) + 2
            ws.column_dimensions[col_letter].width = max(adjusted_width, 12)
    except Exception as e_width:
        print(f"    警告: 调整表格 '{title}' 列宽时出错: {e_width}")
    return final_row
# <<< 结束移动 >>>

def format_metric(value, precision=4, na_rep='N/A'):
    """简单格式化数字，保留指定精度。"""
    if value is None or pd.isna(value):
        return na_rep
    try:
        return f"{float(value):.{precision}f}"
    except (ValueError, TypeError):
        return str(value) # Fallback to string if conversion fails

def format_metric_pct(value, precision=2, na_rep='N/A'):
    """简单格式化为百分比字符串。"""
    if value is None or pd.isna(value):
        return na_rep
    try:
        # Assume input is already a percentage value (e.g., 75.0 for 75%)
        return f"{float(value):.{precision}f}%"
    except (ValueError, TypeError):
        return str(value) # Fallback
