# data_preparation.py
import pandas as pd
import numpy as np
import sys
import os
from collections import Counter # Ensure Counter is imported
from collections import defaultdict # 引入 defaultdict

# --- NEW: Flag for testing with reduced variables ---
USE_REDUCED_VARIABLES_FOR_TESTING = False # <<--- 关闭测试模式，使用所有变量

# --- 控制开关 ---
CREATE_REDUCED_TEST_SET = False # 改回 False 以使用完整数据集
REDUCED_SET_SUFFIX = "_REDUCED" if CREATE_REDUCED_TEST_SET else "" # 根据开关调整后缀

# --- NEW: Helper function to parse frequency from sheet name ---
def get_freq_from_sheet_name(sheet_name: str) -> str | None:
    """Parses frequency ('日度', '周度', '月度') from sheet names like '行业-频率'."""
    if isinstance(sheet_name, str) and '-' in sheet_name:
        parts = sheet_name.split('-')
        if len(parts) >= 2: # Allow names like "Some-Industry-日度"
            freq_part = parts[-1].strip() # Take the last part
            if freq_part == '日度':
                return 'daily'
            elif freq_part == '周度':
                return 'weekly'
            elif freq_part == '月度':
                return 'monthly'
            # Add other potential frequencies if needed
    return None

# --- 辅助函数 ---
# (如果需要复杂的行业提取逻辑，可以在这里添加辅助函数)
def extract_industry(sheet_name):
    if '-' in sheet_name:
        parts = sheet_name.split('-')
        # 假设格式是 '行业-频率' 或 '行业-其他-频率'
        # 我们取第一个部分作为行业名，除非它是目标 sheet
        if len(parts) > 1 and parts[1] in ['日度', '周度', '月度']:
             # 可能是 '工业增加值同比增速-月度' 这种，需要特殊处理或指定行业
             if sheet_name == '工业增加值同比增速-月度': # 硬编码目标 sheet 的行业
                 return 'Macro' # 或者其他指定的名称
             else:
                return parts[0]
        elif len(parts) > 1: # 处理 '行业-其他-频率'
             return parts[0] # 仍然取第一部分
    # 如果无法解析，返回默认值或 None
    return "Uncategorized"

# --- REVISED: prepare_data function ---
def prepare_data(excel_path: str, target_freq: str, target_sheet_name: str, target_variable_name: str) -> pd.DataFrame | None:
    """
    Loads data from an Excel file, automatically identifies daily/weekly predictor sheets
    and the specified monthly target sheet, aligns all data to the target_freq
    (e.g., 'W-FRI') according to specific rules (daily=mean, weekly=last, monthly=assign_last_weekday),
    and returns the final aligned DataFrame.

    Args:
        excel_path (str): Path to the Excel data file.
        target_freq (str): The target frequency string (e.g., 'W-FRI').
        target_sheet_name (str): Name of the sheet containing the target variable.
        target_variable_name (str): Name of the target variable column.

    Returns:
        pd.DataFrame: A DataFrame containing the aligned weekly data,
                      or None if processing fails.
    """
    print(f"\n--- [Data Prep] 开始加载和处理数据 (目标频率: {target_freq}) ---")
    if CREATE_REDUCED_TEST_SET:
        print("  [!] 已启用缩小版测试集生成模式。")

    print(f"  [Data Prep] 目标 Sheet: '{target_sheet_name}', 目标变量: '{target_variable_name}'")

    try:
        excel_file = pd.ExcelFile(excel_path)
        available_sheets = excel_file.sheet_names
        print(f"  [Data Prep] Excel 文件中可用的 Sheets: {available_sheets}")

        data_parts = defaultdict(list) # 使用 defaultdict
        column_origins = {} # 用于追踪列的来源 (industry, type)

        # --- 步骤 1: 遍历所有 Sheets 加载数据 ---
        for sheet_name in available_sheets:
            print(f"    [Data Prep] 正在检查 Sheet: {sheet_name}...")
            df = None
            freq_type = None

            # 1a: 检查是否是目标 Sheet
            if sheet_name == target_sheet_name:
                freq_type = 'monthly_target' # Special type for target
                print(f"      检测到目标 Sheet ('{freq_type}')...")
                try:
                    df = pd.read_excel(excel_file, sheet_name=sheet_name, index_col=0, parse_dates=True)
                    if target_variable_name not in df.columns:
                        print(f"      错误: 目标变量 '{target_variable_name}' 在目标 Sheet '{sheet_name}' 中未找到。已跳过。")
                        continue # Skip this sheet
                    # Select only the target variable column initially
                    df = df[[target_variable_name]].copy()
                except Exception as e:
                    print(f"      加载目标 Sheet '{sheet_name}' 时出错: {e}. 已跳过。")
                    continue

            # 1b: 检查是否是日度预测变量 Sheet
            elif sheet_name.endswith('-日度'):
                freq_type = 'daily'
                print(f"      检测到预测变量 Sheet ('{freq_type}')...")
                try:
                    df = pd.read_excel(excel_file, sheet_name=sheet_name, index_col=0, parse_dates=True)
                except Exception as e:
                    print(f"      加载日度 Sheet '{sheet_name}' 时出错: {e}. 已跳过。")
                    continue

            # 1c: 检查是否是周度预测变量 Sheet
            elif sheet_name.endswith('-周度'):
                freq_type = 'weekly'
                print(f"      检测到预测变量 Sheet ('{freq_type}')...")
                try:
                    df = pd.read_excel(excel_file, sheet_name=sheet_name, index_col=0, parse_dates=True)
                except Exception as e:
                    print(f"      加载周度 Sheet '{sheet_name}' 时出错: {e}. 已跳过。")
                    continue

            # 1d: 其他 Sheet (忽略)
            else:
                print(f"      Sheet '{sheet_name}' 不符合预测变量命名规则 ('-日度' 或 '-周度') 且不是目标 Sheet，已跳过。")
                continue

            # --- 通用清理步骤 ---
            if df is not None:
                # print(f"      原始加载 Shape ('{sheet_name}'): {df.shape}") # REMOVE DEBUG
                if not isinstance(df.index, pd.DatetimeIndex):
                    print(f"      将 Sheet '{sheet_name}' 的索引转换为日期时间...")
                    df_index_original = df.index # Keep original for potential debug
                    try:
                        df.index = pd.to_datetime(df.index, errors='coerce')
                    except Exception as e_time:
                         print(f"      错误: pd.to_datetime 转换索引时出错: {e_time}。 跳过此 sheet。")
                         print(f"      问题索引示例: {df_index_original[:5]}") # Show problematic index
                         continue
                else:
                     print(f"      索引已是 DatetimeIndex。")

                # --- 修改后的清理步骤 ---
                # 1. 移除 NaT 索引行
                initial_rows_before_nat = len(df)
                df = df.loc[df.index.notna()]
                rows_after_nat = len(df)
                if initial_rows_before_nat > rows_after_nat:
                     print(f"      移除了 {initial_rows_before_nat - rows_after_nat} 行，因为索引是 NaT。")

                # 2. 移除全 NaN 的列
                initial_cols = df.shape[1]
                df = df.dropna(axis=1, how='all')
                cols_after_dropna = df.shape[1]
                if initial_cols > cols_after_dropna:
                     print(f"      移除了 {initial_cols - cols_after_dropna} 列，因为它们全为 NaN。")

                # 3. 移除全 NaN 的行
                initial_rows_before_nan_row = len(df)
                df = df.dropna(axis=0, how='all')
                rows_after_nan_row = len(df)
                if initial_rows_before_nan_row > rows_after_nan_row:
                     print(f"      移除了 {initial_rows_before_nan_row - rows_after_nan_row} 行，因为它们全为 NaN。")
                # --- 结束修改后的清理步骤 ---

                if not df.empty:
                     # Convert target/predictors to numeric *after* cleaning
                     # print(f"      清理后 Shape ('{sheet_name}'): {df.shape}") # REMOVE DEBUG
                     print(f"      将 Sheet '{sheet_name}' 数据转换为数值类型 (非数值转为 NaN)...")
                     df_numeric = df.apply(pd.to_numeric, errors='coerce')

                     # --- DEBUG: Check specific variable after loading and numeric conversion ---
                     if '期货结算价(连续):铁矿石' in df_numeric.columns:
                         print(f"      [DEBUG specific] NaN % for '期货结算价(连续):铁矿石' in loaded sheet '{sheet_name}': {df_numeric['期货结算价(连续):铁矿石'].isna().mean():.2%}")
                     # --- END DEBUG ---

                     # Check if numeric conversion resulted in empty DataFrame
                     if not df_numeric.empty and not df_numeric.isnull().all().all():
                         # --- 新增：记录列来源 ---
                         industry = extract_industry(sheet_name) # 使用辅助函数提取行业
                         current_sheet_origin = {'sheet_name': sheet_name, 'industry': industry, 'type': freq_type}
                         for col in df_numeric.columns:
                             # 如果列名已存在 (来自不同频率的同名指标?)，我们可能需要决定如何处理
                             # 暂时：如果列名已存在，保留第一个来源信息
                             if col not in column_origins:
                                 column_origins[col] = current_sheet_origin
                             # else:
                                 # print(f"      警告: 列 '{col}' 在之前的 Sheet 中已存在。将保留第一个来源信息 ({column_origins[col]['sheet_name']})。")
                         # --- 结束新增 ---

                         print(f"      Sheet '{sheet_name}' 加载和处理完成. Shape after potential reduction: {df_numeric.shape}.") # Updated print

                         # --- 修正后的追加逻辑 ---
                         if freq_type == 'monthly_target':
                              # Assign the single-column DataFrame to the monthly list
                              data_parts['monthly'].append(df_numeric)
                         elif freq_type == 'daily' or freq_type == 'weekly': # 直接检查类型
                              data_parts[freq_type].append(df_numeric) # 直接追加，defaultdict 会自动创建 list
                         # --- 结束修正 ---

                     else:
                        print(f"      Sheet '{sheet_name}' 在转换为数值后为空或全为 NaN，已跳过。")
                else:
                    print(f"      Sheet '{sheet_name}' 为空或索引无效，已跳过。")

        # --- 检查是否加载到必要数据 ---
        if not data_parts['monthly']:
             print(f"错误：[Data Prep] 未能成功加载目标变量 '{target_variable_name}' 从 Sheet '{target_sheet_name}'。")
             return None
        if not data_parts['daily'] and not data_parts['weekly']:
             print("警告：[Data Prep] 未能成功加载任何日度或周度预测变量数据。将仅使用目标变量。")
             # Allow proceeding with only the target

        # --- 新增: 步骤 1b: 移除原始数据中缺失过多的预测变量 ---
        print("\n--- [Data Prep] 步骤 1b: 移除原始数据中缺失率 > 50% 的预测变量 --- ")
        original_nan_threshold = 0.50
        cols_dropped_in_step1b = defaultdict(list)

        for freq_type in ['daily', 'weekly']:
            print(f"  检查 {freq_type} 数据...")
            if freq_type in data_parts:
                cleaned_dfs = []
                for i, df in enumerate(data_parts[freq_type]):
                    initial_cols = df.shape[1]
                    # 仅对预测变量执行 (理论上目标变量只在 monthly)
                    predictor_cols = [col for col in df.columns if col != target_variable_name]
                    if not predictor_cols:
                         print(f"    DataFrame {i} (Shape: {df.shape}) 不包含预测变量，跳过NaN检查。")
                         cleaned_dfs.append(df)
                         continue

                    nan_ratios = df[predictor_cols].isna().mean()
                    cols_to_drop = nan_ratios[nan_ratios > original_nan_threshold].index.tolist()

                    if cols_to_drop:
                        df = df.drop(columns=cols_to_drop)
                        cols_dropped_in_step1b[freq_type].extend(cols_to_drop)
                        print(f"    DataFrame {i} (原始 Shape: {(df.shape[0], initial_cols)}): 移除了 {len(cols_to_drop)} 列 (NaN > {original_nan_threshold:.0%}): {cols_to_drop}")
                    cleaned_dfs.append(df)
                data_parts[freq_type] = cleaned_dfs # 更新为清理后的 DataFrames

        if cols_dropped_in_step1b:
             for freq, cols in cols_dropped_in_step1b.items():
                 print(f"  [清理 1b] 在 {freq} 数据中总共移除了以下列: {list(set(cols))}") # 使用 set 去重
        else:
             print(f"  [清理 1b] 未在原始日度/周度数据中发现需要移除的高缺失率预测变量。")
        # --- 结束新增步骤 1b ---

        # --- 步骤 2: 合并相同频率的数据并处理 ---
        processed_parts = []
        all_loaded_indices = [] # To determine date range for monthly alignment
        predictor_dfs_before_resample = [] # 存储重采样前的预测变量 DF

        # 2a: 合并和处理日度数据
        if data_parts['daily']:
            print(f"\n--- [Data Prep] 步骤 2a: 合并和处理日度数据 -> {target_freq} (周均值) ---")
            df_daily_full = pd.concat(data_parts['daily'], axis=1)
            predictor_dfs_before_resample.extend(data_parts['daily']) # 添加到列表
            print(f"    合并了 {len(data_parts['daily'])} 个日度数据集。Shape: {df_daily_full.shape}")
            if df_daily_full.columns.duplicated().any():
                 print("    警告: 合并日度数据时发现重复列名，将保留第一个。")
                 df_daily_full = df_daily_full.loc[:, ~df_daily_full.columns.duplicated(keep='first')]
            df_daily_weekly_mean = df_daily_full.resample(target_freq).mean()
            print(f"  日度->周度(均值) 完成. Shape: {df_daily_weekly_mean.shape}")

            # --- DEBUG: Check specific variable after daily->weekly resample --- 
            if '期货结算价(连续):铁矿石' in df_daily_weekly_mean.columns:
                print(f"  [DEBUG specific] NaN % for '期货结算价(连续):铁矿石' after daily->weekly resample: {df_daily_weekly_mean['期货结算价(连续):铁矿石'].isna().mean():.2%}")
            # --- END DEBUG ---

            processed_parts.append(df_daily_weekly_mean)
            all_loaded_indices.append(df_daily_full.index) # Use original index for range
        else:
            print("\n[Data Prep] 未找到日度预测数据，跳过处理。")

        # 2b: 合并和处理周度数据
        if data_parts['weekly']:
            print(f"\n--- [Data Prep] 步骤 2b: 合并和处理周度数据 -> {target_freq} (取当周最后值对齐) ---")
            df_weekly_full = pd.concat(data_parts['weekly'], axis=1)
            predictor_dfs_before_resample.extend(data_parts['weekly']) # 添加到列表
            print(f"    合并了 {len(data_parts['weekly'])} 个周度数据集。Shape: {df_weekly_full.shape}")
            if df_weekly_full.columns.duplicated().any():
                 print("    警告: 合并周度数据时发现重复列名，将保留第一个。")
                 df_weekly_full = df_weekly_full.loc[:, ~df_weekly_full.columns.duplicated(keep='first')]
            df_weekly_aligned = df_weekly_full.resample(target_freq).last()
            print(f"  周度->周度(对齐) 完成. Shape: {df_weekly_aligned.shape}")

            # --- DEBUG: Check specific variable after weekly->weekly align --- 
            if '期货结算价(连续):铁矿石' in df_weekly_aligned.columns:
                print(f"  [DEBUG specific] NaN % for '期货结算价(连续):铁矿石' after weekly->weekly align: {df_weekly_aligned['期货结算价(连续):铁矿石'].isna().mean():.2%}")
            # --- END DEBUG ---

            processed_parts.append(df_weekly_aligned)
            all_loaded_indices.append(df_weekly_full.index) # Use original index for range
        else:
             print("\n[Data Prep] 未找到周度预测数据，跳过处理。")

        # 2c: 处理月度目标数据
        # Assume only one df in data_parts['monthly'] (the target)
        df_monthly_target = data_parts['monthly'][0]
        all_loaded_indices.append(df_monthly_target.index) # Use original index for range

        if not all_loaded_indices:
            print("警告: [Data Prep] 只有目标变量加载成功，但未加载任何预测变量或无法确定日期范围。")
            # Decide if we proceed with only target or return None
            # Let's try to proceed if target is valid
            min_date = df_monthly_target.index.min()
            max_date = df_monthly_target.index.max()
        else:
            min_date = min(idx.min() for idx in all_loaded_indices if not idx.empty)
            max_date = max(idx.max() for idx in all_loaded_indices if not idx.empty)
        print(f"  [Data Prep] 确定数据范围: {min_date} to {max_date}")

        target_weekly_index = pd.date_range(start=min_date, end=max_date, freq=target_freq)
        print(f"  [Data Prep] 创建目标周度索引范围: {target_weekly_index.min()} to {target_weekly_index.max()} ({len(target_weekly_index)} dates)")
        # --- DEBUG START ---
        # print(f"  [DEBUG] Monthly Target Index (first 5): {df_monthly_target.index[:5]}") # REMOVE DEBUG
        # print(f"  [DEBUG] Generated Weekly Index (first 5): {target_weekly_index[:5]}") # REMOVE DEBUG
        # --- DEBUG END ---

        print(f"\n--- [Data Prep] 步骤 2c: 处理月度目标数据 -> {target_freq} (分配到当月最后一个 {target_freq[-3:]} 日) ---")
        df_monthly_aligned = pd.DataFrame(np.nan, index=target_weekly_index, columns=df_monthly_target.columns)
        # Ensure correct dtype
        try:
            df_monthly_aligned = df_monthly_aligned.astype(df_monthly_target.dtypes.to_dict())
        except Exception as e:
             print(f"   [Data Prep] 警告: 设置月度对齐 DataFrame 的 dtypes 时出错: {e}。")

        # Determine the target weekday number (0=Mon, ..., 4=Fri, ...)
        target_weekday_map = {'MON': 0, 'TUE': 1, 'WED': 2, 'THU': 3, 'FRI': 4, 'SAT': 5, 'SUN': 6}
        target_day_str = target_freq.split('-')[-1][:3].upper()
        if target_day_str not in target_weekday_map:
             print(f"错误: [Data Prep] 无法识别的目标星期几 '{target_day_str}' 从频率 '{target_freq}'。")
             return None
        target_weekday = target_weekday_map[target_day_str]
        print(f"  [Data Prep] 目标星期几: {target_day_str} (数字: {target_weekday})")

        assignment_count = 0
        print(f"  [Data Prep] 正在将月度数据分配到对应目标周日期...")
        for month_date, row_data in df_monthly_target.iterrows():
            # print(f"    [DEBUG] Processing Month: {month_date.strftime('%Y-%m-%d')}") # REMOVE DEBUG
            last_day_current_month = month_date + pd.offsets.MonthEnd(0)
            weekday_of_last_day = last_day_current_month.weekday()

            days_to_subtract = (weekday_of_last_day - target_weekday + 7) % 7
            target_date = last_day_current_month - pd.Timedelta(days=days_to_subtract)
            target_date = target_date.normalize() # Keep normalization

            # --- 修改：使用 .loc 赋值，并添加边界检查和错误处理 ---
            is_in_index_exact = target_date in df_monthly_aligned.index # Keep exact check for debug info
            # print(f"      [DEBUG] Calculated Target Date: {target_date.strftime('%Y-%m-%d')}, In Exact Index: {is_in_index_exact}") # REMOVE DEBUG

            try:
                # 1. 检查计算出的日期是否在目标周度索引的范围内
                if target_date >= df_monthly_aligned.index.min() and target_date <= df_monthly_aligned.index.max():
                    # 2. 使用 .loc 进行赋值，Pandas 会尝试对齐
                    df_monthly_aligned.loc[target_date] = row_data
                    assignment_count += 1
                    # print(f"        [DEBUG] Assigned value for {target_date.strftime('%Y-%m-%d')} using .loc") # Optional success debug
                else:
                    # print(f"      [DEBUG] Target date {target_date.strftime('%Y-%m-%d')} 超出目标周度索引范围 ({df_monthly_aligned.index.min().strftime('%Y-%m-%d')} to {df_monthly_aligned.index.max().strftime('%Y-%m-%d')})") # REMOVE DEBUG
                    pass # 添加 pass 来修复缩进
            except KeyError:
                 # 如果 .loc 仍然找不到精确或可对齐的日期，会触发 KeyError
                 # print(f"      [DEBUG] KeyError: 无法使用 .loc 将值分配给目标日期 {target_date.strftime('%Y-%m-%d')} (可能日期不存在于索引中)") # REMOVE DEBUG
                 pass # 添加 pass
            except Exception as e_assign:
                 # 捕获其他潜在的赋值错误
                 # print(f"      [DEBUG] Assigning using .loc for {target_date.strftime('%Y-%m-%d')} failed: {e_assign}") # REMOVE DEBUG
                 pass # 添加 pass
            # --- 结束修改 ---

        print(f"  月度目标数据对齐完成. Shape: {df_monthly_aligned.shape}. 成功分配 {assignment_count} 个月度值。")
        # Check if assignment actually happened
        if assignment_count == 0:
             print("  [Data Prep] 警告: 未能将任何月度目标值成功分配到周度索引。最终数据集可能为空。")

        processed_parts.append(df_monthly_aligned)

        # --- DEBUG: Check for target name collision BEFORE concat ---
        # print("\n--- [Data Prep] DEBUG: Checking for target column name collision in predictors ---") # REMOVE DEBUG
        target_col_in_predictors = False
        target_col_source = []
        if 'df_daily_weekly_mean' in locals() and target_variable_name in df_daily_weekly_mean.columns:
            # print(f"  [DEBUG] 警告: 目标变量名 '{target_variable_name}' 存在于日度数据中!") # REMOVE DEBUG
            target_col_in_predictors = True
            target_col_source.append("日度")
        if 'df_weekly_aligned' in locals() and target_variable_name in df_weekly_aligned.columns:
            # print(f"  [DEBUG] 警告: 目标变量名 '{target_variable_name}' 存在于周度数据中!") # REMOVE DEBUG
            target_col_in_predictors = True
            target_col_source.append("周度")
        if not target_col_in_predictors:
            # print("  [DEBUG] 目标变量名未在预测变量列中找到。") # REMOVE DEBUG
            pass
        else:
            # print(f"  [DEBUG] 列名冲突来源: {target_col_source}") # REMOVE DEBUG
            pass
        # --- END DEBUG ---

        # --- 步骤 3: 合并所有处理后的数据 ---
        print("\n--- [Data Prep] 步骤 3: 合并所有处理后的周度数据 ---")
        if not processed_parts:
             print("错误：[Data Prep] 没有任何频率的数据被成功处理。")
             return None

        all_data_aligned_weekly = pd.concat(processed_parts, axis=1, join='outer')
        print(f"  [Data Prep] 合并后的周度数据 shape (before duplicate handling): {all_data_aligned_weekly.shape}")

        # Handle duplicate columns (could arise if a variable exists in multiple freq sheets)
        if all_data_aligned_weekly.columns.duplicated().any():
            duplicate_cols = all_data_aligned_weekly.columns[all_data_aligned_weekly.columns.duplicated()].unique().tolist()
            print(f"  [Data Prep] 警告: 合并后发现重复列名: {duplicate_cols}. 将保留第一次出现的列 (优先顺序: 日度均值 > 周度最后值 > 月度目标值)。")
            # --- DEBUG: Show columns before dropping duplicates ---
            # print(f"    [DEBUG] Columns before dropping duplicates: {all_data_aligned_weekly.columns.tolist()}") # REMOVE DEBUG
            # --- END DEBUG ---
            all_data_aligned_weekly = all_data_aligned_weekly.loc[:, ~all_data_aligned_weekly.columns.duplicated(keep='first')]
            print(f"  [Data Prep] 去除重复列后 shape: {all_data_aligned_weekly.shape}")
            # --- DEBUG: Show columns after dropping duplicates ---
            # print(f"    [DEBUG] Columns after dropping duplicates: {all_data_aligned_weekly.columns.tolist()}") # REMOVE DEBUG
            # --- END DEBUG ---

        # Ensure target variable column still exists AFTER duplicate removal
        if target_variable_name not in all_data_aligned_weekly.columns:
             print(f"错误: [Data Prep] 目标变量 '{target_variable_name}' 在去除重复列后丢失！")
             return None
        else:
             # --- DEBUG: Inspect target column BEFORE dropna ---
             # print(f"\n--- [Data Prep] DEBUG: Inspecting target column '{target_variable_name}' before final dropna ---") # REMOVE DEBUG
             target_col_data = all_data_aligned_weekly[target_variable_name]
             # print(f"  [DEBUG] Target column non-NaN count: {target_col_data.notna().sum()}") # REMOVE DEBUG
             # print(f"  [DEBUG] Target column value_counts (including NaN):\n{target_col_data.value_counts(dropna=False)}") # REMOVE DEBUG
             # print(f"  [DEBUG] Target column info:") # REMOVE DEBUG
             # target_col_data.info() # REMOVE DEBUG
             pass # Keep the block structure
             # --- END DEBUG ---

        print(f"  [Data Prep] 保留所有周度行，目标变量 NaN 将由 DFM 处理。当前行数: {len(all_data_aligned_weekly)}") # New message

        # --- Add detailed pre-threshold check ---
        print("\n--- [Data Prep] DEBUG: Pre-NaN Threshold Check ---")
        print(f"  Shape before cleaning: {all_data_aligned_weekly.shape}")
        print(f"  NaN count per column (Top 30):\n{all_data_aligned_weekly.isna().sum().sort_values(ascending=False).head(30)}")
        if '期货结算价(连续):铁矿石' in all_data_aligned_weekly.columns:
            fe_nan_ratio = all_data_aligned_weekly['期货结算价(连续):铁矿石'].isna().mean()
            print(f"  NaN ratio for '期货结算价(连续):铁矿石' BEFORE threshold check: {fe_nan_ratio:.2%}")
        # --- End pre-threshold check ---

        # --- NEW: Final Data Cleaning Steps ---
        print("\n--- [Data Prep] 步骤 5: 最终数据清理 --- ")
        initial_cols_final = all_data_aligned_weekly.shape[1]

        # 1. 移除 NaN 比例过高的列 (例如 > 30%)
        nan_threshold = 0.30 # <--- 使用用户指定的 30% 阈值
        print(f"  [清理] 检查 NaN 比例 > {nan_threshold:.0%} 的预测变量列...")
        nan_ratios = all_data_aligned_weekly.isna().mean() # This is the crucial calculation
        # --- DEBUG: Print NaN ratios being compared ---
        # print(f"  [DEBUG] Calculated NaN Ratios (Top 10):\n{nan_ratios.sort_values(ascending=False).head(10)}") # 注释掉调试信息
        # --- END DEBUG ---
        cols_to_drop_nan = nan_ratios[nan_ratios > nan_threshold].index.tolist()
        # 确保不删除目标变量列，即使它 NaN 很多
        if target_variable_name in cols_to_drop_nan:
            cols_to_drop_nan.remove(target_variable_name)

        if cols_to_drop_nan:
            # print(f"  [清理] **正在移除** NaN 比例 > {nan_threshold:.0%} 的预测变量列: {cols_to_drop_nan}") # 注释掉移除信息
            # all_data_aligned_weekly = all_data_aligned_weekly.drop(columns=cols_to_drop_nan) # **注释掉**以跳过移除
            print(f"  [清理] 注意: 检测到 {len(cols_to_drop_nan)} 个预测变量列的 NaN 比例 > {nan_threshold:.0%}，但已跳过移除步骤。") # 恢复跳过信息
            # print(f"  [清理] 移除后 Shape: {all_data_aligned_weekly.shape}") # 注释掉移除后形状信息
        else:
            print(f"  [清理] 未因高 NaN 比例移除任何预测变量列。")
        # --- END NEW Cleaning Steps ---

        # --- FINAL Check for emptiness (shouldn't be empty now) ---
        if all_data_aligned_weekly.empty:
            print(f"错误: [Data Prep] 最终数据集在清理后为空（异常情况）。")
            return None

        # --- 步骤 6 (原步骤4): 排序索引 ---
        print("\n--- [Data Prep] 步骤 6: 排序最终索引 --- ")
        all_data_aligned_weekly.sort_index(inplace=True)

        print(f"  [Data Prep] 索引排序完成. 最早日期: {all_data_aligned_weekly.index.min()}, 最晚日期: {all_data_aligned_weekly.index.max()}")

        # --- NEW: 步骤 7: 应用日期筛选 --- 
        start_date_filter = '2020-01-01'
        print(f"\n--- [Data Prep] 步骤 7: 应用日期筛选 (>= {start_date_filter}) ---")
        original_rows = len(all_data_aligned_weekly)
        all_data_aligned_weekly = all_data_aligned_weekly[all_data_aligned_weekly.index >= start_date_filter]
        filtered_rows = len(all_data_aligned_weekly)
        print(f"  [Data Prep] 筛选完成. 原始行数: {original_rows}, 筛选后行数: {filtered_rows}")

        # 检查筛选后是否为空
        if all_data_aligned_weekly.empty:
            print(f"警告: [Data Prep] 应用日期筛选 ({start_date_filter}) 后数据集为空！检查筛选日期或源数据。")
            return None # 如果为空则返回 None

        # --- 新增: 步骤 8: 根据行业/类型选择 Top K 缺失最少的变量 (如果开关打开) ---
        if CREATE_REDUCED_TEST_SET:
             print(f"\n--- [Data Prep] 步骤 8: 生成缩小版测试集 (Top 3 变量/行业/类型) ---")
             predictors_to_keep = []
             grouped_predictors = defaultdict(list)

             # 按行业和类型分组 (使用 column_origins)
             for col, origin in column_origins.items():
                 # 只考虑最终 DataFrame 中存在的预测变量
                 if col in all_data_aligned_weekly.columns and col != target_variable_name:
                     group_key = (origin.get('industry', 'Unknown'), origin.get('type', 'Unknown'))
                     grouped_predictors[group_key].append(col)

             print(f"  [筛选] 按 {len(grouped_predictors)} 个 (行业, 类型) 组进行筛选...")
             for (industry, type), cols in grouped_predictors.items():
                 print(f"    组: ({industry}, {type}), 包含 {len(cols)} 个变量")
                 if not cols: continue # 跳过空组

                 # 计算当前组内变量在 *筛选后* DataFrame 中的非缺失值数量
                 nan_counts_in_group = all_data_aligned_weekly[cols].notna().sum()
                 sorted_cols_in_group = nan_counts_in_group.sort_values(ascending=False)

                 # 选择 Top 3 (或全部，如果少于3个)
                 selected_for_group = sorted_cols_in_group.head(3).index.tolist()
                 predictors_to_keep.extend(selected_for_group)
                 print(f"      -> 已选择: {selected_for_group}")

             # 去重并添加目标变量
             final_selected_columns = list(dict.fromkeys([target_variable_name] + predictors_to_keep)) # 去重并保证目标变量在前面

             print(f"  [筛选] 最终选择 {len(final_selected_columns)} 列 (含目标变量)。")
             if len(final_selected_columns) <= 1 and target_variable_name in final_selected_columns:
                 print(f"  警告: 筛选后只剩下目标变量 '{target_variable_name}'。缩小版测试集将只包含此列。")
             elif target_variable_name not in final_selected_columns:
                  print(f"  错误: 目标变量 '{target_variable_name}' 在筛选后丢失了！这是一个逻辑错误。保留所有列以避免问题。")
                  # 不进行筛选，保留原样
             else:
                 all_data_aligned_weekly = all_data_aligned_weekly[final_selected_columns]
                 print(f"  [筛选] 缩小版数据集 Shape: {all_data_aligned_weekly.shape}")


        # --- 步骤 9 (原步骤8): 返回结果 ---
        print(f"\n--- [Data Prep] 数据准备完成 ({'缩小版' if CREATE_REDUCED_TEST_SET else '完整版'}) ---")
        return all_data_aligned_weekly

    except FileNotFoundError:
        print(f"错误: [Data Prep] Excel 数据文件 {excel_path} 未找到。")
        return None
    except Exception as err:
        print(f"错误: [Data Prep] 数据准备过程中发生意外错误: {err}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == '__main__':
    # Example usage (optional, for testing the module directly)
    print(f"Testing data_preparation module (Reduced Set Mode: {CREATE_REDUCED_TEST_SET})...") # 更新打印信息
    EXCEL_DATA_FILE_TEST = '经济数据库.xlsx'
    TARGET_FREQ_TEST = 'W-FRI'
    TARGET_SHEET_TEST = '工业增加值同比增速-月度'
    TARGET_VAR_TEST = '规模以上工业增加值:当月同比'

    prepared_data = prepare_data(excel_path=EXCEL_DATA_FILE_TEST,
                                 target_freq=TARGET_FREQ_TEST,
                                 target_sheet_name=TARGET_SHEET_TEST,
                                 target_variable_name=TARGET_VAR_TEST)

    if prepared_data is not None:
        output_suffix = REDUCED_SET_SUFFIX # 使用之前定义的后缀
        output_filename = f"test_prepared_weekly_auto_discovery{output_suffix}.csv" # 构建文件名
        output_path = os.path.join("test_output", output_filename)

        print(f"\n--- Prepared Weekly Data Info ({'Reduced' if CREATE_REDUCED_TEST_SET else 'Full'}) ---") # 更新打印信息
        prepared_data.info()
        print(f"\n--- Prepared Weekly Data Head ({'Reduced' if CREATE_REDUCED_TEST_SET else 'Full'}) ---") # 更新打印信息
        print(prepared_data.head())

        # Example: Save test output
        try:
            os.makedirs('test_output', exist_ok=True) # Create output dir if needed
            prepared_data.to_csv(output_path, encoding='utf-8-sig') # 使用构建的路径
            print(f"\nTest output saved to {output_path}") # 更新打印信息
        except Exception as e:
            print(f"\nError saving test output to {output_path}: {e}") # 更新打印信息
    else:
        print(f"\nWeekly data preparation failed during test ({'Reduced Set Mode' if CREATE_REDUCED_TEST_SET else 'Full Set Mode'}).") # 更新打印信息
