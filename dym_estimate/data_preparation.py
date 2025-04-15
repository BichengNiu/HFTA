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
    (e.g., 'W-FRI') according to specific rules (daily=mean, weekly=last, monthly=align_to_nearest_friday_from_col_A),
    and returns the final aligned DataFrame.

    Args:
        excel_path (str): Path to the Excel data file.
        target_freq (str): The target frequency string (e.g., 'W-FRI'). Must end in '-FRI'.
        target_sheet_name (str): Name of the sheet containing the target variable.
        target_variable_name (str): Name of the target variable column (expected in Col B).

    Returns:
        pd.DataFrame: A DataFrame containing the aligned weekly data,
                      or None if processing fails.
    """
    print(f"\n--- [Data Prep] 开始加载和处理数据 (目标频率: {target_freq}) ---")
    if CREATE_REDUCED_TEST_SET:
        print("  [!] 已启用缩小版测试集生成模式。")

    # <-- 新增: 验证目标频率 -->
    if not target_freq.upper().endswith('-FRI'):
        print(f"错误: [Data Prep] 当前目标对齐逻辑仅支持周五 (W-FRI)。提供的目标频率 '{target_freq}' 无效。")
        return None
    # <-- 结束新增 -->

    print(f"  [Data Prep] 目标 Sheet: '{target_sheet_name}', 目标变量: '{target_variable_name}' (预期在 B 列)")

    try:
        excel_file = pd.ExcelFile(excel_path)
        available_sheets = excel_file.sheet_names
        print(f"  [Data Prep] Excel 文件中可用的 Sheets: {available_sheets}")

        data_parts = defaultdict(list) # 使用 defaultdict
        column_origins = {} # 用于追踪列的来源 (industry, type)
        target_data_raw = None # 用于存储原始目标数据 (日期和值)

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
                    # <-- 修改: 读取时不设置索引，指定列类型 -->
                    # 读取时不解析日期，以便手动处理第一列的中文日期格式
                    df = pd.read_excel(excel_file, sheet_name=sheet_name, header=0) # header=0 假设第二行是数据

                    # 验证列是否存在 (假设 A 列是日期，B 列是目标值)
                    if df.shape[1] < 2:
                        print(f"      错误: 目标 Sheet '{sheet_name}' 的列数少于 2。无法识别日期和目标值列。")
                        continue

                    date_col_name = df.columns[0] # 获取第一列的实际名称 (可能是 'Unnamed: 0')
                    value_col_name_in_excel = df.columns[1] # 获取第二列的实际名称

                    # 检查第二列名称是否与预期的目标变量名匹配 (大小写/空格不敏感比较)
                    if value_col_name_in_excel.strip().lower() != target_variable_name.strip().lower():
                        print(f"      警告: 目标 Sheet '{sheet_name}' 的第二列名称 '{value_col_name_in_excel}' 与预期的目标变量名 '{target_variable_name}' 不完全匹配。将继续使用第二列作为目标值。")
                        # 仍然使用 value_col_name_in_excel 作为值的来源
                    else:
                        print(f"      确认目标变量 '{target_variable_name}' 在第二列。")
                        # 确保后续使用标准化的 target_variable_name

                    # 解析第一列的日期 (尝试多种格式)
                    print(f"      尝试解析第一列 ('{date_col_name}') 中的日期...")
                    publication_dates = pd.to_datetime(df[date_col_name], errors='coerce')
                    if publication_dates.isna().all():
                        print(f"      错误: 无法将第一列 ('{date_col_name}') 解析为日期。请检查格式。已跳过。")
                        continue

                    # 解析第二列的目标值
                    print(f"      尝试解析第二列 ('{value_col_name_in_excel}') 中的数值...")
                    target_values = pd.to_numeric(df[value_col_name_in_excel], errors='coerce')

                    # 合并解析后的数据
                    target_df_parsed = pd.DataFrame({
                        'PublicationDate': publication_dates,
                        target_variable_name: target_values
                    })

                    # 移除日期或数值无效的行
                    initial_target_rows = len(target_df_parsed)
                    target_data_raw = target_df_parsed.dropna()
                    if len(target_data_raw) < initial_target_rows:
                         print(f"      移除了 {initial_target_rows - len(target_data_raw)} 行，因为发布日期或目标值无效。")

                    if target_data_raw.empty:
                        print(f"      错误: 处理后目标 Sheet '{sheet_name}' 中没有有效的日期/数值对。已跳过。")
                        target_data_raw = None # 重置为 None
                        continue
                    else:
                        print(f"      目标 Sheet 数据加载和初步解析完成。Shape: {target_data_raw.shape}")
                    # <-- 结束修改 -->
                except Exception as e:
                    print(f"      加载或解析目标 Sheet '{sheet_name}' 时出错: {e}. 已跳过。")
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

            # --- 通用清理步骤 (预测变量) ---
            if df is not None and freq_type != 'monthly_target': # 只处理预测变量
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
        if target_data_raw is None:
             print(f"错误：[Data Prep] 未能成功加载或处理目标 Sheet '{target_sheet_name}' 的数据 (需要 '{target_variable_name}' 和 'PublicationDate' 列)。")
             return None
        if not data_parts['daily'] and not data_parts['weekly']:
             print("警告：[Data Prep] 未能成功加载任何日度或周度预测变量数据。将仅使用目标变量。")
             # Allow proceeding with only the target

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

        # 2c: 处理月度目标数据 (按发布日期对齐到最近周五)
        print(f"\n--- [Data Prep] 步骤 2c: 处理月度目标数据 -> {target_freq} (按发布日期对齐到最近周五) ---")
        target_weekly_data = []

        for _, row in target_data_raw.iterrows():
            month_value = row[target_variable_name]
            pub_date = row['PublicationDate'] # 使用解析后的日期列

            # 计算发布日期与当周周五的偏移量
            days_since_monday = pub_date.weekday() # Monday is 0, Friday is 4
            days_to_friday = 4 - days_since_monday

            # 根据规则确定目标周五日期
            if days_since_monday <= 1: # 周一或周二发布
                # 对齐到上周五 (当前周五日期 - 7天)
                target_aligned_date = pub_date + pd.Timedelta(days=days_to_friday - 7)
            else: # 周三、周四、周五 (或周末) 发布
                # 对齐到当前周五
                target_aligned_date = pub_date + pd.Timedelta(days=days_to_friday)

            target_weekly_data.append((target_aligned_date, month_value))

        if target_weekly_data:
            target_series_weekly = pd.Series(
                [val for idx, val in target_weekly_data],
                index=[idx for idx, val in target_weekly_data],
                name=target_variable_name
            ).sort_index()

            # 处理可能存在的重复目标日期 (例如，同一个周五发布了多个月的数据)，保留最后一个
            initial_len = len(target_series_weekly)
            target_series_weekly = target_series_weekly[~target_series_weekly.index.duplicated(keep='last')]
            if len(target_series_weekly) < initial_len:
                 print(f"    处理了 {initial_len - len(target_series_weekly)} 个重复的目标对齐日期，保留了每个日期的最后一个值。")

            # 确保索引是唯一的 DatetimeIndex
            target_series_weekly.index = pd.to_datetime(target_series_weekly.index)
            print(f"  月度目标数据按发布日期对齐到 {target_freq} 完成. Shape: {target_series_weekly.shape}")
            processed_parts.append(target_series_weekly)
            all_loaded_indices.append(target_series_weekly.index) # 使用对齐后的目标索引
        else:
            print("错误：[Data Prep] 未能从原始目标数据中生成任何有效的周度对齐数据点。")
            return None
        # --- 结束步骤 2c ---

        # --- 步骤 3: 最终合并所有处理过的数据 ---
        print("\n--- [Data Prep] 步骤 3: 最终合并所有处理过的数据 ---")
        if not processed_parts:
            print("错误：[Data Prep] 没有成功处理的数据部分可以合并。")
            return None

        # 合并所有周度数据
        combined_data = pd.concat(processed_parts, axis=1)
        print(f"  合并所有 {len(processed_parts)} 个处理后的数据部分. 初始合并 Shape: {combined_data.shape}")

        # 处理合并后可能产生的重复列名 (应该在加载时大部分已处理，但以防万一)
        if combined_data.columns.duplicated().any():
            print("    警告: 最终合并时发现重复列名，将保留第一个。")
            combined_data = combined_data.loc[:, ~combined_data.columns.duplicated(keep='first')]

        # 确定最终的完整日期范围
        if all_loaded_indices:
            min_date = min(idx.min() for idx in all_loaded_indices if not idx.empty)
            max_date = max(idx.max() for idx in all_loaded_indices if not idx.empty)
            # 使用 target_freq 生成完整的周度日期序列
            full_date_range = pd.date_range(start=min_date, end=max_date, freq=target_freq)
            print(f"  确定的完整日期范围: {min_date.date()} 到 {max_date.date()} (频率: {target_freq})")
        else:
            print("错误: 无法确定日期范围，因为没有加载任何有效数据。")
            return None

        # 重新索引以确保所有数据都在完整的周度网格上
        all_data_aligned_weekly = combined_data.reindex(full_date_range)
        print(f"  重新索引到完整日期范围完成. 最终 Shape: {all_data_aligned_weekly.shape}")

        # --- DEBUG: Check specific variable in final aligned data --- 
        if '期货结算价(连续):铁矿石' in all_data_aligned_weekly.columns:
            print(f"  [DEBUG specific] NaN % for '期货结算价(连续):铁矿石' in FINAL aligned data: {all_data_aligned_weekly['期货结算价(连续):铁矿石'].isna().mean():.2%}")
        # --- END DEBUG ---

        # --- 验证目标变量是否存在于最终数据中 --- 
        if target_variable_name not in all_data_aligned_weekly.columns:
            print(f"严重错误: 目标变量 '{target_variable_name}' 在最终对齐的数据中丢失了！")
            print(f"  可用列: {all_data_aligned_weekly.columns.tolist()[:20]}...")
            return None

        # --- 移除在合并/重采样后可能变成全 NaN 的列 --- 
        initial_final_cols = all_data_aligned_weekly.shape[1]
        all_data_aligned_weekly = all_data_aligned_weekly.dropna(axis=1, how='all')
        final_cols_after_dropna = all_data_aligned_weekly.shape[1]
        if initial_final_cols > final_cols_after_dropna:
             print(f"  在最终对齐和重采样后，移除了 {initial_final_cols - final_cols_after_dropna} 列，因为它们全为 NaN。")
             # 再次验证目标变量
             if target_variable_name not in all_data_aligned_weekly.columns:
                  print(f"严重错误: 目标变量 '{target_variable_name}' 在移除全 NaN 列后丢失了！")
                  return None

        print("\n--- [Data Prep] 数据准备完成 --- ")
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
