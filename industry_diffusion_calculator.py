import pandas as pd
import numpy as np
from datetime import datetime
import json # 引入 json 库
import os # 引入 os 库
import re # 引入正则表达式库
import warnings # Import warnings

# 统一缺失阈值常量
DEFAULT_MISSING_THRESHOLD = 0.3 # Changed default to 0.3
# DEFAULT_COMBINED_MISSING_THRESHOLD = 0.33 # Removed
DEFAULT_TOLERANCE_THRESHOLD = 0.01 # Reset tolerance default to 0.01 as 0 seems less useful

def calculate_wow_diffusion_index(data: pd.DataFrame, missing_threshold: float = DEFAULT_MISSING_THRESHOLD, tolerance_threshold: float = DEFAULT_TOLERANCE_THRESHOLD) -> pd.Series:
    """
    计算环比扩散指数

    Args:
        data (pd.DataFrame): 行业指标数据，索引为日期。
        missing_threshold (float): 允许的最大缺失值比例，超过则该日指数为 NaN。
        tolerance_threshold (float): 判断增长的容忍度阈值。大于此阈值才算增长。

    Returns:
        pd.Series: 环比扩散指数，索引为日期。
    """
    wow_growth = data.pct_change(periods=1, fill_method=None)
    diffusion_results = {}
    
    for date in wow_growth.index:
        # 检查缺失值比例 (基于原始数据, 使用传入的 missing_threshold)
        non_missing_ratio = data.loc[date].notna().mean()
        if non_missing_ratio < missing_threshold:
            diffusion_results[date] = np.nan
            continue

        # 计算环比增长率
        growth_this_week = wow_growth.loc[date]
        valid_growth = growth_this_week.dropna()
        valid_growth_count = len(valid_growth)
        
        if valid_growth_count == 0:
            diffusion_results[date] = np.nan
            continue

        # 计算增长指标数量 (使用传入的 tolerance_threshold)
        increasing_count = (valid_growth > tolerance_threshold).sum()
        diffusion_index = (increasing_count / valid_growth_count) * 100
        diffusion_results[date] = diffusion_index

    return pd.Series(diffusion_results, name="WoW_Diffusion_Index")

def calculate_yoy_diffusion_index(data: pd.DataFrame, missing_threshold: float = DEFAULT_MISSING_THRESHOLD, tolerance_threshold: float = DEFAULT_TOLERANCE_THRESHOLD) -> pd.Series:
    """
    计算同比扩散指数

    Args:
        data (pd.DataFrame): 行业指标数据，索引为日期。
        missing_threshold (float): 允许的最大缺失值比例。
        tolerance_threshold (float): 判断增长的容忍度阈值。

    Returns:
        pd.Series: 同比扩散指数，索引为日期。
    """
    yoy_growth = data.pct_change(periods=52, fill_method=None)
    diffusion_results = {}
    
    for date in yoy_growth.index:
        # 检查缺失值比例 (基于原始数据, 使用传入的 missing_threshold)
        non_missing_ratio = data.loc[date].notna().mean()
        if non_missing_ratio < missing_threshold:
            diffusion_results[date] = np.nan
            continue

        # 计算同比增长率
        growth_this_year = yoy_growth.loc[date]
        valid_growth = growth_this_year.dropna()
        valid_growth_count = len(valid_growth)
        
        if valid_growth_count == 0:
            diffusion_results[date] = np.nan
            continue

        # 计算增长指标数量 (使用传入的 tolerance_threshold)
        increasing_count = (valid_growth > tolerance_threshold).sum()
        diffusion_index = (increasing_count / valid_growth_count) * 100
        diffusion_results[date] = diffusion_index

    return pd.Series(diffusion_results, name="YoY_Diffusion_Index")

def calculate_mix_diffusion_index(data: pd.DataFrame, tolerance_threshold: float = DEFAULT_TOLERANCE_THRESHOLD, missing_threshold: float = DEFAULT_MISSING_THRESHOLD) -> pd.Series:
    """
    计算同环比扩散指数 (MixDI) - 使用新的逻辑。
    新逻辑：先同比变化(pct_change 52)，再对结果进行环比变化(pct_change 4)。
    Args:
        data (pd.DataFrame): 行业指标数据，索引为日期。
        tolerance_threshold (float): 判断增长的容忍度阈值 (> tolerance_threshold)。
        missing_threshold (float): 统一的缺失阈值 (要求非缺失比例 >= missing_threshold)。
    Returns:
        pd.Series: MixDI，索引为日期。
    """
    if not isinstance(data.index, pd.DatetimeIndex):
        # Attempt conversion if not already DatetimeIndex, warn if fails
        try:
            data.index = pd.to_datetime(data.index)
        except Exception as e:
             warnings.warn(f"MixDI: 输入数据框索引无法转换为时间索引: {e}")
             return pd.Series(index=data.index, name="Mix_Diffusion_Index", dtype=float) # Return empty Series
    
    # 确保数据按时间排序
    data = data.sort_index()

    # +++ 新增：将所有 0 值替换为 NaN，视为缺失 +++
    data_no_zero = data.replace(0, np.nan)
    
    # --- Start of new logic adaptation ---
    
    # 计算组合变化率：先计算同比变化，再计算环比变化
    # 1. 计算同比变化 (using data_no_zero to avoid inf from pct_change on 0)
    yoy_relative = data_no_zero.pct_change(52, fill_method=None)
    
    # 2. 计算同比变化的4周环比变化 (using data_no_zero based yoy_relative)
    combined_changes = yoy_relative.pct_change(periods=4, fill_method=None)
    
    # 检查组合变化率的缺失值比例 (使用传入的 missing_threshold)
    def check_missing_ratio(changes, threshold): 
        # Calculate NON-missing ratio and compare with the threshold
        non_missing_ratio = changes.notna().mean(axis=1)
        return non_missing_ratio >= threshold
    
    # Apply check using the passed missing_threshold
    valid_weeks = check_missing_ratio(combined_changes, missing_threshold)
    
    # 按周计算扩散指数 (使用传入的 tolerance_threshold)
    def calculate_weekly_di(changes, valid_weeks, tolerance): 
        weekly_di = []
        
        for i, week in enumerate(changes.index):
            # Also add initial check on original data missing ratio
            original_non_missing_ratio = data.loc[week].notna().mean()
            if original_non_missing_ratio < missing_threshold: # Use the same threshold
                 weekly_di.append(np.nan)
                 continue
                 
            # Check validity based on combined_changes missing ratio
            if not valid_weeks.iloc[i]:
                weekly_di.append(np.nan)
                continue
                
            week_data = changes.loc[week]
            # Check against tolerance_threshold instead of 0
            improvements = (week_data > tolerance).sum() 
            total = week_data.count()  # 非NaN的数量
            
            if total == 0: # 避免除以零
                di_value = np.nan
            else:
                di_value = (improvements / total) * 100
            
            # Use round only if not NaN
            weekly_di.append(round(di_value, 2) if pd.notna(di_value) else np.nan) 
                
        return weekly_di
    
    # Calculate the DI list using the passed tolerance_threshold
    mix_di_list = calculate_weekly_di(combined_changes, valid_weeks, tolerance_threshold)
    
    # Return as a named Series
    return pd.Series(mix_di_list, index=data.index, name="Mix_Diffusion_Index", dtype=float)
    # --- End of new logic adaptation ---

def calculate_all_diffusion_indices(
    weekly_data_numeric: pd.DataFrame, 
    indicator_source_map: dict,
    # Use unified thresholds
    missing_threshold: float = DEFAULT_MISSING_THRESHOLD,
    tolerance_threshold: float = DEFAULT_TOLERANCE_THRESHOLD
    # Removed separate mix/wow/yoy thresholds
    # wow_missing_threshold: float = DEFAULT_MISSING_THRESHOLD,
    # yoy_missing_threshold: float = DEFAULT_MISSING_THRESHOLD,
    # mix_combined_missing_threshold: float = DEFAULT_COMBINED_MISSING_THRESHOLD,
    # wow_tolerance: float = DEFAULT_TOLERANCE_THRESHOLD,
    # yoy_tolerance: float = DEFAULT_TOLERANCE_THRESHOLD,
    # mix_tolerance: float = DEFAULT_TOLERANCE_THRESHOLD
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """ 
    Orchestrates the calculation of WoW, YoY, and MixDI using unified thresholds.
    Args:
        weekly_data_numeric (pd.DataFrame): Weekly data.
        indicator_source_map (dict): Source map.
        missing_threshold (float): Unified missing threshold.
        tolerance_threshold (float): Unified tolerance threshold.
    Returns:
        tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]: WoW, YoY, MixDI results.
    """
    if weekly_data_numeric.empty or not indicator_source_map:
        warnings.warn("Input weekly data or source map is empty. Cannot calculate indices.")
        empty_df = pd.DataFrame()
        return empty_df, empty_df, empty_df

    # --- Group data by industry --- 
    print("Grouping data by industry based on source map...")
    industry_data_groups = {}
    all_sources = sorted(list(set(indicator_source_map.values())))
    
    if not all_sources:
         warnings.warn("Source map contains no valid sources. Cannot group data.")
         empty_df = pd.DataFrame()
         return empty_df, empty_df, empty_df
         
    print(f"Found {len(all_sources)} unique sources (industries) in map.")

    for source_name in all_sources:
        indicators_in_source = [indicator for indicator, source in indicator_source_map.items() 
                                if source == source_name and indicator in weekly_data_numeric.columns]
        
        if indicators_in_source:
            industry_data_groups[source_name] = weekly_data_numeric[indicators_in_source]
        else: 
            pass

    if not industry_data_groups:
        warnings.warn("Could not group data for any industry. Cannot calculate indices.")
        empty_df = pd.DataFrame()
        return empty_df, empty_df, empty_df

    # --- Calculate indices for each industry using unified thresholds --- 
    print(f"Calculating diffusion indices for {len(industry_data_groups)} industries...") 
    wow_results_df = pd.DataFrame(index=weekly_data_numeric.index)
    yoy_results_df = pd.DataFrame(index=weekly_data_numeric.index)
    mix_results_df = pd.DataFrame(index=weekly_data_numeric.index)

    for industry_name, industry_data in industry_data_groups.items():
        if industry_data.empty or industry_data.shape[1] == 0:
            continue

        # Pass the unified thresholds to each function
        wow_index = calculate_wow_diffusion_index(
            industry_data, missing_threshold=missing_threshold, tolerance_threshold=tolerance_threshold)
        yoy_index = calculate_yoy_diffusion_index(
            industry_data, missing_threshold=missing_threshold, tolerance_threshold=tolerance_threshold)
        mix_index = calculate_mix_diffusion_index(
            industry_data, missing_threshold=missing_threshold, tolerance_threshold=tolerance_threshold)

        wow_results_df[industry_name] = wow_index
        yoy_results_df[industry_name] = yoy_index
        mix_results_df[industry_name] = mix_index
        
    # Clean empty columns just in case
    wow_results_df.dropna(axis=1, how='all', inplace=True)
    yoy_results_df.dropna(axis=1, how='all', inplace=True)
    mix_results_df.dropna(axis=1, how='all', inplace=True)
    
    print("Diffusion index calculation complete.")
    return wow_results_df, yoy_results_df, mix_results_df

def save_indices_to_sheets(wow_df, yoy_df, mix_df, output_file): 
    """Saves the calculated diffusion indices to separate sheets in an Excel file."""
    print(f"\nPreparing to save results to {output_file} in separate sheets...")
    
    if wow_df.empty and yoy_df.empty and mix_df.empty:
        print("All calculation results are empty, skipping Excel save.")
        return

    try:
        with pd.ExcelWriter(output_file, engine='xlsxwriter', datetime_format='yyyy-mm-dd') as writer:
            num_format = writer.book.add_format({'num_format': '0.00'})

            # Write WoW Sheet
            if not wow_df.empty:
                wow_df.to_excel(writer, sheet_name='WoW_DI')
                worksheet_wow = writer.sheets['WoW_DI']
                worksheet_wow.set_column(0, 0, 15) # 日期列
                for i, col_name in enumerate(wow_df.columns):
                    col_width = max(15, len(str(col_name)) + 2)
                    worksheet_wow.set_column(i + 1, i + 1, col_width, num_format)
                print(f"  Written 'WoW_DI' sheet ({len(wow_df.columns)} industries).")
            else: print("WoW index results empty, skipping sheet.")

            # Write YoY Sheet
            if not yoy_df.empty:
                yoy_df.to_excel(writer, sheet_name='YoY_DI')
                worksheet_yoy = writer.sheets['YoY_DI']
                worksheet_yoy.set_column(0, 0, 15) # 日期列
                for i, col_name in enumerate(yoy_df.columns):
                    col_width = max(15, len(str(col_name)) + 2)
                    worksheet_yoy.set_column(i + 1, i + 1, col_width, num_format)
                print(f"  Written 'YoY_DI' sheet ({len(yoy_df.columns)} industries).")
            else: print("YoY index results empty, skipping sheet.")
                
            # Write Mix DI Sheet
            if not mix_df.empty:
                mix_df.to_excel(writer, sheet_name='Mix_DI')
                worksheet_mix = writer.sheets['Mix_DI']
                worksheet_mix.set_column(0, 0, 15) # 日期列
                for i, col_name in enumerate(mix_df.columns):
                    col_width = max(15, len(str(col_name)) + 2)
                    worksheet_mix.set_column(i + 1, i + 1, col_width, num_format)
                print(f"  Written 'Mix_DI' sheet ({len(mix_df.columns)} industries).")
            else: print("MixDI index results empty, skipping sheet.")

        print(f"Successfully saved results to {output_file}")
    except Exception as e:
        print(f"Error saving Excel file: {e}")

def main():
    # Configuration for standalone execution
    MAPPING_FILE = "indicator_source_mapping.json" 
    INPUT_FILE = 'merged_output.xlsx'
    INPUT_SHEET = 'WeeklyData'
    OUTPUT_FILE = 'industry_diffusion_indices_separate_sheets.xlsx' 
    
    print("--- Running Diffusion Calculator Standalone ---")
    # --- Load Data (only when run as script) --- 
    print(f"Loading indicator source map: {MAPPING_FILE}...")
    if not os.path.exists(MAPPING_FILE):
        print(f"Error: Source map file '{MAPPING_FILE}' not found. Run data_loader.py first.")
        return
    try:
        with open(MAPPING_FILE, 'r', encoding='utf-8') as f:
            indicator_source_map = json.load(f)
    except Exception as e:
        print(f"Error loading source map: {e}")
        return
        
    print(f"Loading weekly data: {INPUT_FILE}, sheet '{INPUT_SHEET}'...")
    try:
        weekly_data_numeric = pd.read_excel(INPUT_FILE, sheet_name=INPUT_SHEET, index_col=0)
        weekly_data_numeric.index = pd.to_datetime(weekly_data_numeric.index)
        weekly_data_numeric = weekly_data_numeric.sort_index()
        weekly_data_numeric = weekly_data_numeric.select_dtypes(include=np.number)
    except FileNotFoundError:
        print(f"Error: Input file '{INPUT_FILE}' not found. Run data_loader.py first.")
        return
    except KeyError:
         print(f"Error: Sheet '{INPUT_SHEET}' not found in '{INPUT_FILE}'.")
         return
    except Exception as e:
        print(f"Error reading or processing weekly data: {e}")
        return

    # --- Calculate Indices --- 
    wow_results, yoy_results, mix_results = calculate_all_diffusion_indices(
        weekly_data_numeric, indicator_source_map
    )
    
    # --- Save Results (only when run as script) ---
    save_indices_to_sheets(wow_results, yoy_results, mix_results, OUTPUT_FILE)
    
    print("--- Diffusion Calculator Standalone Finished ---")

if __name__ == '__main__':
    main() 