import pandas as pd
import numpy as np
import warnings
from datetime import datetime, timedelta
import json
import os

def read_merged_data(filename: str = "merged_output.xlsx") -> tuple[pd.DataFrame | None, pd.DataFrame | None]:
    """
    读取包含合并数据的 Excel 文件。

    参数:
    filename: str, 默认 "merged_output.xlsx"
        要读取的 Excel 文件名。

    返回:
    tuple[pd.DataFrame | None, pd.DataFrame | None]:
        - df_weekly: 周度数据 DataFrame，如果 sheet 不存在或读取失败则为 None。
        - df_monthly: 月度数据 DataFrame，如果 sheet 不存在或读取失败则为 None。
    """
    df_weekly = None
    df_monthly = None

    try:
        print(f"尝试读取文件 '{filename}' 的 'WeeklyData' sheet...")
        # 读取时指定 index_col=0 和 parse_dates=True
        df_weekly = pd.read_excel(filename, sheet_name='WeeklyData', index_col=0, parse_dates=True)
        # 确保索引是 DatetimeIndex
        if not isinstance(df_weekly.index, pd.DatetimeIndex):
            warnings.warn("WeeklyData 的索引未能正确解析为日期，尝试再次转换...")
            df_weekly.index = pd.to_datetime(df_weekly.index, errors='coerce')
            df_weekly = df_weekly[df_weekly.index.notna()]
        df_weekly.sort_index(inplace=True)
        print("成功读取并处理 WeeklyData。")

    except FileNotFoundError:
        print(f"错误：文件 '{filename}' 未找到。")
        return None, None
    except ValueError as ve:
        if 'Worksheet named' in str(ve) and 'WeeklyData' in str(ve):
            warnings.warn(f"文件 '{filename}' 中未找到 'WeeklyData' sheet。")
        else:
            warnings.warn(f"读取 'WeeklyData' sheet 时出错: {ve}")
    except Exception as e:
        warnings.warn(f"读取或处理 'WeeklyData' sheet 时发生意外错误: {e}")

    try:
        print(f"尝试读取文件 '{filename}' 的 'MonthlyData' sheet...")
        df_monthly = pd.read_excel(filename, sheet_name='MonthlyData', index_col=0, parse_dates=True)
        if not isinstance(df_monthly.index, pd.DatetimeIndex):
             warnings.warn("MonthlyData 的索引未能正确解析为日期，尝试再次转换...")
             df_monthly.index = pd.to_datetime(df_monthly.index, errors='coerce')
             df_monthly = df_monthly[df_monthly.index.notna()]
        df_monthly.sort_index(inplace=True)
        print("成功读取并处理 MonthlyData。")

    except ValueError as ve:
        if 'Worksheet named' in str(ve) and 'MonthlyData' in str(ve):
            warnings.warn(f"文件 '{filename}' 中未找到 'MonthlyData' sheet。")
        else:
            warnings.warn(f"读取 'MonthlyData' sheet 时出错: {ve}")
    except Exception as e:
        warnings.warn(f"读取或处理 'MonthlyData' sheet 时发生意外错误: {e}")

    return df_weekly, df_monthly

def calculate_weekly_growth_summary(df_weekly: pd.DataFrame) -> pd.DataFrame:
    """
    以当前日期所在周为基准，计算周度数据的历史值、增长率及近5年统计，并生成汇总表。

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

    # --- 确定报告日期 --- 
    today = pd.Timestamp.now().normalize()
    # 计算本周的周五 (如果今天是周六/周日，则为刚过去的周五)
    # current_week_friday = today - pd.Timedelta(days=today.dayofweek) + pd.Timedelta(days=4)
    # 改为：计算本周对应的周五 (如果今天就是周五，则是今天；如果是周一，则是本周五)
    current_week_friday = today + pd.Timedelta(days=(4 - today.dayofweek + 7) % 7)

    if current_week_friday in df_weekly.index:
        reporting_friday = current_week_friday
    else:
        reporting_friday = df_weekly.index.max()
        warnings.warn(f"当前周的周五 ({current_week_friday.strftime('%Y-%m-%d')}) 不在数据中。使用最新的可用日期: {reporting_friday.strftime('%Y-%m-%d')}")
    print(f"使用报告日期: {reporting_friday.strftime('%Y-%m-%d')}")

    print("准备历史值...")
    # --- 预先计算历史值 DataFrame (用 shift) ---
    last_week_values_df = df_weekly.shift(periods=1)
    last_month_values_df = df_weekly.shift(periods=4) # 4周前的值
    last_year_values_df = df_weekly.shift(periods=52) # 52周前的值

    summary_data = []
    print("生成汇总表中...")

    for indicator in df_weekly.columns:
        # --- 获取当期和历史值 --- 
        current_value = df_weekly.loc[reporting_friday, indicator] if reporting_friday in df_weekly.index else np.nan
        val_last_week = last_week_values_df.loc[reporting_friday, indicator] if reporting_friday in last_week_values_df.index else np.nan
        val_last_month = last_month_values_df.loc[reporting_friday, indicator] if reporting_friday in last_month_values_df.index else np.nan
        val_last_year = last_year_values_df.loc[reporting_friday, indicator] if reporting_friday in last_year_values_df.index else np.nan

        # --- 计算当期增长率 --- 
        latest_wow = (current_value - val_last_week) / abs(val_last_week) if pd.notna(current_value) and pd.notna(val_last_week) and val_last_week != 0 else np.nan
        latest_moy = (current_value - val_last_month) / abs(val_last_month) if pd.notna(current_value) and pd.notna(val_last_month) and val_last_month != 0 else np.nan
        latest_yoy = (current_value - val_last_year) / abs(val_last_year) if pd.notna(current_value) and pd.notna(val_last_year) and val_last_year != 0 else np.nan
        
        # --- 计算近5年统计 (不含当前年) --- 
        current_year = pd.Timestamp.now().year
        stats_start_year = current_year - 5
        stats_end_year = current_year - 1
        stats_start_date = pd.Timestamp(f'{stats_start_year}-01-01')
        stats_end_date = pd.Timestamp(f'{stats_end_year}-12-31')

        # 筛选数据
        historical_5y_data = df_weekly.loc[(df_weekly.index >= stats_start_date) & (df_weekly.index <= stats_end_date), indicator].dropna()
        
        if not historical_5y_data.empty:
            stat_max = historical_5y_data.max()
            stat_min = historical_5y_data.min()
            stat_mean = historical_5y_data.mean()
        else:
            stat_max, stat_min, stat_mean = np.nan, np.nan, np.nan

        summary_data.append({
            '周度指标名称': indicator,
            '最新日期': reporting_friday.strftime('%Y-%m-%d'), # 只显示到日
            '上周值': val_last_week,
            '环比上周': latest_wow, # 改名
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

    # 确保列顺序符合要求
    column_order = [
        '周度指标名称', '最新日期',
        '上周值', '环比上周',
        '上月值', '环比上月', 
        '上年值', '同比上年', 
        '近5年最大值', '近5年最小值', '近5年平均值'
    ]
    # 过滤掉可能不存在的列
    existing_columns = [col for col in column_order if col in summary_df.columns]
    summary_df = summary_df[existing_columns]

    return summary_df

def calculate_monthly_growth_summary(df_monthly: pd.DataFrame) -> pd.DataFrame:
    """
    以最新月份为基准，计算月度数据的历史值、增长率及近5年统计，并生成汇总表。

    参数:
    df_monthly: pd.DataFrame
        预处理后的月度数据 DataFrame (应包含 DatetimeIndex)。

    返回:
    pd.DataFrame: 包含指标名称、最新月份及各种最新指标的汇总表。
                 如果输入 DataFrame 为空或无效，则返回空 DataFrame。
    """
    if df_monthly is None or df_monthly.empty or not isinstance(df_monthly.index, pd.DatetimeIndex):
        print("输入的月度 DataFrame 为空或索引无效，无法计算。")
        return pd.DataFrame()

    # 确保按时间排序
    df_monthly = df_monthly.sort_index()

    # --- 确定报告月份 (数据中最新的月份) --- 
    reporting_month_end = df_monthly.index.max()
    # 转换为 Period 以方便计算上月/上年
    try:
        reporting_period = reporting_month_end.to_period('M')
    except AttributeError:
         # 如果索引不是 DatetimeIndex（理论上read_merged_data已处理，但作为后备）
         print("月度数据索引无法转换为月度周期。")
         return pd.DataFrame()
         
    print(f"使用报告月份: {reporting_period}")

    print("准备月度历史值...")
    # --- 预先计算历史值 DataFrame (用 shift) ---
    # 月度数据 shift(1) 即为上月, shift(12) 即为上年同期
    last_month_values_df = df_monthly.shift(periods=1)
    last_year_values_df = df_monthly.shift(periods=12)

    summary_data = []
    print("生成月度汇总表中...")

    for indicator in df_monthly.columns:
        # --- 获取当期和历史值 --- 
        # 使用最新的日期索引来定位
        current_value = df_monthly.loc[reporting_month_end, indicator] if reporting_month_end in df_monthly.index else np.nan
        val_last_month = last_month_values_df.loc[reporting_month_end, indicator] if reporting_month_end in last_month_values_df.index else np.nan
        val_last_year = last_year_values_df.loc[reporting_month_end, indicator] if reporting_month_end in last_year_values_df.index else np.nan

        # --- 计算当期增长率 (使用差值) --- 
        latest_mom = (current_value - val_last_month) if pd.notna(current_value) and pd.notna(val_last_month) else np.nan
        latest_yoy = (current_value - val_last_year) if pd.notna(current_value) and pd.notna(val_last_year) else np.nan
        
        # --- 计算近5年统计 (不含当前年) --- 
        current_year = reporting_period.year # 使用报告期的年份
        stats_start_year = current_year - 5
        stats_end_year = current_year - 1
        stats_start_date = pd.Timestamp(f'{stats_start_year}-01-01')
        stats_end_date = pd.Timestamp(f'{stats_end_year}-12-31')

        # 筛选数据
        historical_5y_data = df_monthly.loc[(df_monthly.index >= stats_start_date) & (df_monthly.index <= stats_end_date), indicator].dropna()
        
        if not historical_5y_data.empty:
            stat_max = historical_5y_data.max()
            stat_min = historical_5y_data.min()
            stat_mean = historical_5y_data.mean()
        else:
            stat_max, stat_min, stat_mean = np.nan, np.nan, np.nan

        summary_data.append({
            '月度指标名称': indicator,
            '最新月份': reporting_period, 
            '上月值': val_last_month,
            '环比上月': latest_mom, # 结果是差值
            '上年值': val_last_year,
            '同比上年': latest_yoy, # 结果是差值
            '近5年最大值': stat_max,
            '近5年最小值': stat_min,
            '近5年平均值': stat_mean
        })

    summary_df = pd.DataFrame(summary_data)
    # 将 Period 转换为字符串以便 Excel 显示
    if '最新月份' in summary_df.columns:
        summary_df['最新月份'] = summary_df['最新月份'].astype(str)
        
    print(f"月度汇总表生成完成，共 {len(summary_df)} 个指标。")

    # 确保列顺序符合要求
    column_order = [
        '月度指标名称', '最新月份',
        '上月值', '环比上月', 
        '上年值', '同比上年', 
        '近5年最大值', '近5年最小值', '近5年平均值'
    ]
    existing_columns = [col for col in column_order if col in summary_df.columns]
    summary_df = summary_df[existing_columns]

    return summary_df

def calculate_weekly_summary(weekly_df: pd.DataFrame, reporting_date_str: str = None) -> pd.DataFrame:
    """
    Calculates the weekly growth summary table from the weekly data.

    Args:
        weekly_df (pd.DataFrame): DataFrame with weekly data, indexed by date.
        reporting_date_str (str, optional): The reporting date as 'YYYY-MM-DD'. 
                                            If None, uses the latest date in the data. Defaults to None.

    Returns:
        pd.DataFrame: The calculated weekly summary table.
    """
    if weekly_df.empty:
        warnings.warn("Input weekly DataFrame is empty. Cannot calculate weekly summary.")
        return pd.DataFrame()

    # Determine reporting date
    if reporting_date_str:
        try:
            reporting_date = pd.to_datetime(reporting_date_str)
            if reporting_date not in weekly_df.index:
                original_date = reporting_date.strftime('%Y-%m-%d')
                reporting_date = weekly_df.index.max() # Fallback to latest
                warnings.warn(f"指定报告日期 {original_date} 不在数据中，将使用最新可用日期 {reporting_date.strftime('%Y-%m-%d')}。")
        except ValueError:
            reporting_date = weekly_df.index.max() # Fallback to latest
            warnings.warn(f"无效的报告日期格式 '{reporting_date_str}', 将使用最新可用日期 {reporting_date.strftime('%Y-%m-%d')}。")
    else:
        reporting_date = weekly_df.index.max() # Default to latest date

    print(f"使用报告日期: {reporting_date.strftime('%Y-%m-%d')}")

    # Calculate necessary dates
    last_week_date = reporting_date - timedelta(weeks=1)
    last_year_date = reporting_date - timedelta(weeks=52)
    five_years_ago_date = reporting_date - timedelta(days=365 * 5)

    # --- Calculations ---
    summary_data = []
    for indicator in weekly_df.columns:
        data_series = weekly_df[indicator].dropna()
        if data_series.empty:
            continue

        # Get values for specific dates, handle missing dates
        current_value = data_series.get(reporting_date, np.nan)
        last_week_value = data_series.get(last_week_date, np.nan)
        last_year_value = data_series.get(last_year_date, np.nan)
        
        # Fallback if exact last year date is missing (find closest prior)
        if pd.isna(last_year_value) and last_year_date > data_series.index.min():
            closest_last_year_date = data_series.index[data_series.index <= last_year_date].max()
            if pd.notna(closest_last_year_date):
                last_year_value = data_series.get(closest_last_year_date, np.nan)
                # print(f"  Indicator '{indicator}': Using {closest_last_year_date.strftime('%Y-%m-%d')} for YoY comparison instead of {last_year_date.strftime('%Y-%m-%d')}")

        # Calculate growth rates
        wow_growth = (current_value / last_week_value - 1) if pd.notna(current_value) and pd.notna(last_week_value) and last_week_value != 0 else np.nan
        yoy_growth = (current_value / last_year_value - 1) if pd.notna(current_value) and pd.notna(last_year_value) and last_year_value != 0 else np.nan

        # Calculate 5-year stats
        past_5_years_data = data_series[data_series.index >= five_years_ago_date]
        past_5_years_data = past_5_years_data[past_5_years_data.index <= reporting_date] # Ensure we don't include future data if index goes beyond

        max_5y = past_5_years_data.max() if not past_5_years_data.empty else np.nan
        min_5y = past_5_years_data.min() if not past_5_years_data.empty else np.nan
        avg_5y = past_5_years_data.mean() if not past_5_years_data.empty else np.nan

        summary_data.append({
            '指标名称': indicator,
            '最新值': current_value,
            '最新日期': reporting_date.strftime('%Y-%m-%d'),
            '周环比': wow_growth,
            '年同比': yoy_growth,
            '近5年最大值': max_5y,
            '近5年最小值': min_5y,
            '近5年平均值': avg_5y
        })

    summary_table = pd.DataFrame(summary_data)
    if not summary_table.empty:
        summary_table.set_index('指标名称', inplace=True)
    
    print(f"Weekly summary table calculated with {len(summary_table)} indicators.")
    return summary_table

def save_summary_to_excel(summary_df: pd.DataFrame, output_filename: str, sheet_name: str = 'Summary'):
    """Saves the summary dataframe to an Excel sheet with basic formatting."""
    if summary_df.empty:
        print(f"Summary DataFrame for sheet '{sheet_name}' is empty. Skipping save.")
        return
       
    print(f"Saving summary to {output_filename}, sheet '{sheet_name}'...")
    try:
        with pd.ExcelWriter(output_filename, engine='xlsxwriter', datetime_format='yyyy-mm-dd') as writer:
            summary_df.to_excel(writer, sheet_name=sheet_name)
            worksheet = writer.sheets[sheet_name]
            workbook = writer.book

            # Basic Formatting
            header_format = workbook.add_format({'bold': True, 'align': 'center', 'valign': 'vcenter', 'border': 1})
            percent_format = workbook.add_format({'num_format': '0.00%', 'border': 1})
            number_format = workbook.add_format({'num_format': '#,##0.00', 'border': 1})
            date_format = workbook.add_format({'num_format': 'yyyy-mm-dd', 'border': 1})
            default_format = workbook.add_format({'border': 1})

            # Apply header format
            for col_num, value in enumerate(summary_df.reset_index().columns.values):
                worksheet.write(0, col_num, value, header_format)
                
            # Auto-adjust column widths and apply formats (example)
            for idx, col in enumerate(summary_df.reset_index().columns): 
                series = summary_df.reset_index()[col]
                max_len = max(series.astype(str).map(len).max(), len(str(series.name))) + 2
                worksheet.set_column(idx, idx, max_len, default_format) # Start with default
                
                # Apply specific formats based on column name heuristics
                col_name_lower = str(col).lower()
                if '日期' in col_name_lower:
                     worksheet.set_column(idx, idx, max_len, date_format)
                elif '环比' in col_name_lower or '同比' in col_name_lower or '率' in col_name_lower: 
                     worksheet.set_column(idx, idx, max_len, percent_format)
                elif pd.api.types.is_numeric_dtype(series.dtype) and not pd.api.types.is_integer_dtype(series.dtype):
                     # Apply number format to non-integer numeric columns
                     worksheet.set_column(idx, idx, max_len, number_format)

            print(f"Successfully saved and formatted sheet '{sheet_name}' in '{output_filename}'.")

    except Exception as e:
        print(f"Error saving or formatting Excel file '{output_filename}', sheet '{sheet_name}': {e}")

def main():
    # Configuration for standalone execution
    INPUT_FILE = 'merged_output.xlsx' # Assumes data_loader has run
    WEEKLY_SHEET = 'WeeklyData'
    # MONTHLY_SHEET = 'MonthlyData' # Uncomment if calculating monthly too
    OUTPUT_SUMMARY_FILE = 'summary_output.xlsx' # File for summary

    print("--- Running Growth Calculator Standalone ---")
    
    # --- Load Data --- 
    print(f"Loading data from {INPUT_FILE}...")
    try:
        weekly_df = pd.read_excel(INPUT_FILE, sheet_name=WEEKLY_SHEET, index_col=0)
        weekly_df.index = pd.to_datetime(weekly_df.index)
    except FileNotFoundError:
        print(f"Error: Input file '{INPUT_FILE}' not found. Run data_loader.py first.")
        return
    except KeyError:
         print(f"Error: Sheet '{WEEKLY_SHEET}' not found in '{INPUT_FILE}'.")
         return # Exit if weekly data is essential
    except Exception as e:
        print(f"Error reading weekly data: {e}")
        weekly_df = pd.DataFrame() # Allow continuing if monthly exists?

    # Load monthly data (optional, adapt as needed)
    # try:
    #     monthly_df = pd.read_excel(INPUT_FILE, sheet_name=MONTHLY_SHEET, index_col=0)
    #     monthly_df.index = pd.to_datetime(monthly_df.index)
    # except Exception as e:
    #     print(f"Warning: Could not read monthly data from sheet '{MONTHLY_SHEET}': {e}")
    #     monthly_df = pd.DataFrame()

    # --- Calculate Weekly Summary --- 
    if not weekly_df.empty:
        weekly_summary_table = calculate_weekly_summary(weekly_df)
        print("\nWeekly Summary Head:")
        print(weekly_summary_table.head())
        
        # Save Weekly Summary (only when run as script)
        save_summary_to_excel(weekly_summary_table, OUTPUT_SUMMARY_FILE, sheet_name='WeekSummary')
    else:
        print("Weekly data is empty, skipping weekly summary calculation and saving.")

    # --- Calculate and Save Monthly Summary (Example, adapt as needed) --- 
    # if not monthly_df.empty:
    #     monthly_summary_table = calculate_monthly_growth_summary(monthly_df) # Assume this function exists
    #     print("\nMonthly Summary Head:")
    #     print(monthly_summary_table.head())
    #     save_summary_to_excel(monthly_summary_table, OUTPUT_SUMMARY_FILE, sheet_name='MonthSummary') # Appends/Overwrites based on implementation
    # else:
    #      print("Monthly data is empty, skipping monthly summary calculation and saving.")
         
    print("--- Growth Calculator Standalone Finished ---")

if __name__ == '__main__':
    main() 