# data_preparation.py
import pandas as pd
# import suppress_prints  # 抑制子进程中的重复打印 - 暂时注释掉
import numpy as np
import sys
import os
import json
from collections import Counter, defaultdict
from typing import Tuple, Dict, Optional, List, Any
import unicodedata
from statsmodels.tsa.stattools import adfuller
import io

# --- NEW: Flag for testing with reduced variables ---
USE_REDUCED_VARIABLES_FOR_TESTING = False # <<--- 关闭测试模式，使用所有变量

# --- 控制开关 ---
CREATE_REDUCED_TEST_SET = False # 改回 False 以使用完整数据集
REDUCED_SET_SUFFIX = "_REDUCED" if CREATE_REDUCED_TEST_SET else "" # 根据开关调整后缀

# --- NEW: Function to detect target sheet format ---
def detect_sheet_format(excel_file, sheet_name: str) -> Dict[str, Any]:
    """
    Detects the format of any sheet to handle different database versions and data sources.
    Returns a dictionary with format information and reading parameters.
    
    Supports:
    - 同花顺格式: Row0(废弃), Row1(指标名称), Row2(频率), Row3(单位), Row4(指标ID), Row5+(数据)
    - Wind格式: Row0(废弃), Row1(指标名称), Row2+(数据)  
    - Mysteel格式: Row0(废弃), Row1(指标名称), Row2(频度), Row3(指标描述), Row4+(数据)
    - 旧格式: Row0(指标名称), Row1+(数据或元数据)
    """
    try:
        # Read first few rows to analyze format
        df_sample = pd.read_excel(excel_file, sheet_name=sheet_name, nrows=8, header=None)
        
        if df_sample.shape[0] < 2:
            print(f"      [格式检测] Sheet '{sheet_name}' 行数不足，使用默认格式")
            return {
                'format': 'default',
                'skiprows': None,
                'header': 0,
                'data_start_row': 1,
                'source': 'unknown'
            }
        
        # Check for new format patterns
        row_0 = df_sample.iloc[0, 0] if df_sample.shape[0] > 0 else None
        row_1 = df_sample.iloc[1, 0] if df_sample.shape[1] > 0 else None
        row_2 = df_sample.iloc[2, 0] if df_sample.shape[0] > 2 else None
        row_3 = df_sample.iloc[3, 0] if df_sample.shape[0] > 3 else None
        row_4 = df_sample.iloc[4, 0] if df_sample.shape[0] > 4 else None
        row_4_col1 = df_sample.iloc[4, 1] if df_sample.shape[0] > 4 and df_sample.shape[1] > 1 else None
        
        # 同花顺新格式检测 (更精确的检测条件)
        # 检查完整的5行格式：指标名称+频率+单位+指标ID+数据
        if (row_1 == '指标名称' and row_2 == '频率' and row_3 == '单位' and row_4 == '指标ID' and
            str(row_4_col1) not in [None, '', 'nan'] and (str(row_4_col1).startswith('M') or str(row_4_col1).startswith('S'))):  # 支持M和S开头的指标ID
            print(f"      [格式检测] 检测到同花顺完整新格式")
            return {
                'format': 'tonghuashun_new',
                'skiprows': [0, 2, 3, 4],  # Skip: 废弃行, 频率, 单位, 指标ID
                'header': 0,               # Row 1 becomes header
                'data_start_row': 5,
                'source': 'tonghuashun'
            }
        
        # Wind新格式检测 (检查row_0是否为数字0或NaN)
        elif (row_1 == '指标名称' and (row_0 == 0 or pd.isna(row_0)) and 
              row_2 != '频率' and row_2 != '频度' and 
              not pd.isna(row_2)):  # row_2是实际数据，不是元数据
            print(f"      [格式检测] 检测到Wind新格式")
            return {
                'format': 'wind_new',
                'skiprows': [0],           # Skip: 废弃行
                'header': 0,               # Row 1 becomes header
                'data_start_row': 2,
                'source': 'wind'
            }
        
        # Mysteel新格式检测 (检查是否以"钢联数据"开头)
        elif (row_0 == '钢联数据' and row_1 == '指标名称' and row_2 == '频度'):
            print(f"      [格式检测] 检测到Mysteel新格式")
            return {
                'format': 'mysteel_new',
                'skiprows': [0, 2, 3],     # Skip: 废弃行, 频度, 指标描述
                'header': 0,               # Row 1 becomes header
                'data_start_row': 4,
                'source': 'mysteel'
            }
        
        # 旧格式检测
        elif row_0 == '指标名称':
            print(f"      [格式检测] 检测到旧格式")
            # 检测是否有指标ID行需要跳过
            if (df_sample.shape[0] > 1 and 
                str(df_sample.iloc[1, 0]).startswith('S') and len(str(df_sample.iloc[1, 0])) > 5):
                return {
                    'format': 'old_with_id',
                    'skiprows': [1],       # Skip: 指标ID行
                    'header': 0,
                    'data_start_row': 2,
                    'source': 'old'
                }
            else:
                return {
                    'format': 'old_direct',
                    'skiprows': None,
                    'header': 0,
                    'data_start_row': 1,
                    'source': 'old'
                }
        
        # 如果都不匹配，使用默认策略
        else:
            print(f"      [格式检测] 未识别的格式，使用默认策略")
            print(f"      [Debug] Row 0: {row_0}, Row 1: {row_1}, Row 2: {row_2}")
            return {
                'format': 'default',
                'skiprows': None,
                'header': 0,
                'data_start_row': 1,
                'source': 'unknown'
            }
            
    except Exception as e:
        print(f"      [格式检测] 检测格式时出错: {e}，使用默认参数")
        return {
            'format': 'error',
            'skiprows': None,
            'header': 0,
            'data_start_row': 1,
            'source': 'error'
        }

# --- Backward compatibility function ---
def detect_target_sheet_format(excel_file, sheet_name: str) -> Dict[str, Any]:
    """
    Backward compatibility wrapper for detect_sheet_format.
    """
    return detect_sheet_format(excel_file, sheet_name)

# --- NEW: Helper function to parse frequency and industry from sheet name ---
def parse_sheet_info(sheet_name: str, target_sheet_name: str) -> Dict[str, Optional[str]]:
    """
    Parses sheet names like '行业_频率_数据来源' or handles target sheet.
    Returns a dictionary with 'industry', 'freq_type', 'source'.
    Frequency types: 'daily', 'weekly', 'monthly', None.
    Special freq_type: 'monthly_target' if sheet_name matches target_sheet_name.
    """
    # +++ 调试打印 +++
    # print(f"    [Debug parse_sheet_info] sheet_name='{sheet_name}', target_sheet_name='{target_sheet_name}'")
    # +++ 结束调试 +++
    info: Dict[str, Optional[str]] = {'industry': None, 'freq_type': None, 'source': None}
    if not isinstance(sheet_name, str):
        return info

    # --- 修改：进行大小写不敏感的比较，并确保 target_sheet_name 也是字符串 ---\
    is_target_sheet = False
    if isinstance(target_sheet_name, str):
        if sheet_name.lower() == target_sheet_name.lower():
            is_target_sheet = True

    if is_target_sheet:
    # --- 结束修改 ---\
         # +++ 调试打印 +++\n         # print(f"      [Debug parse_sheet_info] Matched target sheet!")\n         # +++ 结束调试 +++\n         info['freq_type'] = 'monthly_target'
         # 尝试从目标名称提取行业 (可选)
         parts_target = sheet_name.split('_')
         if len(parts_target) > 0:
             # 假设第一个部分是行业相关
             industry_part = parts_target[0].replace('-月度', '').replace('_月度','').strip() # 同时替换 - 和 _
             info['industry'] = industry_part if industry_part else 'Macro' # 默认为 Macro
         else:
             info['industry'] = 'Macro' # 默认
         return info
    # --- 结束目标 Sheet 检查 ---

    # --- 通用格式解析 ---
    parts = sheet_name.split('_')
    if len(parts) >= 2: # 至少需要 行业_频率
        info['industry'] = parts[0].strip()
        freq_part = parts[1].strip()
        if freq_part == '日度':
            info['freq_type'] = 'daily'
        elif freq_part == '周度':
            info['freq_type'] = 'weekly'
        elif freq_part == '月度':
            # We handle monthly predictors separately if they come from the target sheet
            # This type is for *other* monthly predictor sheets, if any exist.
            info['freq_type'] = 'monthly_predictor' # <--- 确保识别其他可能的月度Sheet
        # Add other potential frequencies if needed

        if len(parts) >= 3:
            info['source'] = '_'.join(parts[2:]).strip() # 允许来源包含下划线

    # 如果未能解析出行业，给个默认值
    if info['industry'] is None and info['freq_type'] is not None:
         info['industry'] = "Uncategorized"

    return info

# --- NEW: Function to load mappings ---
def load_mappings(
    excel_path: str,
    sheet_name: str,
    indicator_col: str = '高频指标',
    type_col: str = '类型',
    industry_col: Optional[str] = '行业' # Industry column is optional
) -> Tuple[Dict[str, str], Dict[str, str]]:
    """
    Loads variable type and industry mappings from a specified sheet in an Excel file.
    Normalizes indicator names (keys) to lowercase NFKC.
    (Implementation remains the same as before)
    """
    var_type_map = {}
    var_industry_map = {}
    print(f"\n--- [Mappings] Loading type/industry maps from: ")
    print(f"    Excel: {excel_path}")
    print(f"    Sheet: {sheet_name}")
    print(f"    Indicator Col: '{indicator_col}', Type Col: '{type_col}', Industry Col: '{industry_col}'")

    try:
        excel_file_obj = pd.ExcelFile(excel_path)
        if sheet_name not in excel_file_obj.sheet_names:
             raise FileNotFoundError(f"Sheet '{sheet_name}' not found in '{excel_path}'")

        indicator_sheet = pd.read_excel(excel_file_obj, sheet_name=sheet_name)

        # Normalize column names
        indicator_sheet.columns = indicator_sheet.columns.str.strip()
        indicator_col = indicator_col.strip()
        type_col = type_col.strip()
        if industry_col:
            industry_col = industry_col.strip()

        # Check required columns exist
        if indicator_col not in indicator_sheet.columns or type_col not in indicator_sheet.columns:
            raise ValueError(f"未找到必需的列 '{indicator_col}' 或 '{type_col}' 在 sheet '{sheet_name}'")

        # Create Type Map
        var_type_map_temp = pd.Series(
            indicator_sheet[type_col].astype(str).str.strip().values,
            index=indicator_sheet[indicator_col].astype(str).str.strip()
        ).to_dict()
        # Normalize keys and filter NaNs/empty strings
        var_type_map = {unicodedata.normalize('NFKC', str(k)).strip().lower(): str(v).strip()
                        for k, v in var_type_map_temp.items()
                        if pd.notna(k) and str(k).strip().lower() not in ['', 'nan']
                        and pd.notna(v) and str(v).strip().lower() not in ['', 'nan']}
        print(f"  [Mappings] Successfully created type map with {len(var_type_map)} entries.")

        # Create Industry Map (optional)
        if industry_col and industry_col in indicator_sheet.columns:
            industry_map_temp = pd.Series(
                indicator_sheet[industry_col].astype(str).str.strip().values,
                index=indicator_sheet[indicator_col].astype(str).str.strip() # Use indicator_col for index
            ).to_dict()
            # Normalize keys and filter NaNs/empty strings
            var_industry_map = {unicodedata.normalize('NFKC', str(k)).strip().lower(): str(v).strip()
                                for k, v in industry_map_temp.items()
                                if pd.notna(k) and str(k).strip().lower() not in ['', 'nan']
                                and pd.notna(v) and str(v).strip().lower() not in ['', 'nan']}
            print(f"  [Mappings] Successfully created industry map with {len(var_industry_map)} entries.")
        elif industry_col:
             print(f"  [Mappings] Warning: Industry column '{industry_col}' not found in sheet '{sheet_name}'. Industry map will be empty.")
        else:
             print(f"  [Mappings] Industry column not specified. Industry map will be empty.")

    except FileNotFoundError as e:
        print(f"Error loading mappings: {e}")
        # Return empty maps on file/sheet not found
    except ValueError as e:
        print(f"Error processing mapping sheet: {e}")
        # Return empty maps on column errors
    except Exception as e:
        print(f"An unexpected error occurred while loading mappings: {e}")
        # Return empty maps on other errors

    print(f"--- [Mappings] Loading finished. Type map size: {len(var_type_map)}, Industry map size: {len(var_industry_map)} ---")
    return var_type_map, var_industry_map

# --- REVISED: _ensure_stationarity function ---
def _ensure_stationarity(
    df: pd.DataFrame,
    skip_cols: Optional[set] = None, # <-- Added: skip columns
    adf_p_threshold: float = 0.05
) -> Tuple[pd.DataFrame, Dict, Dict]:
    """
    内部函数：检查并转换 DataFrame 中的变量以达到平稳性。
    新增: 可以通过 skip_cols 参数指定不进行检查的列。
    🔥 优化：添加批量处理和缓存机制以提高性能。

    处理逻辑:
    1. 对 df 中的每一列进行处理 (除非在 skip_cols 中指定)。
    2. 进行 ADF 检验 (Level)。
    3. 如果平稳 (p < adf_p_threshold)，保留 level。
    4. 如果不平稳，计算一阶差分，再次检验。
    5. 如果差分后平稳，使用差分序列。
    6. 如果差分后仍不平稳，仍使用差分序列，但标记状态。
    7. 移除原始为空/常量的列，或差分后为空/常量的列。
    8. 被跳过的列直接保留原始值。
    """
    print(f"\n--- [Stationarity Check] 开始检查和转换平稳性 (ADF p<{adf_p_threshold}) --- ")
    print(f"  🔧 优化模式：批量处理 {len(df.columns)} 个变量")

    transformed_data = pd.DataFrame(index=df.index)
    transform_log = {}
    removed_cols_info = defaultdict(list)

    # --- MODIFICATION: Normalize skip_cols for reliable matching ---
    skip_cols_normalized = set()
    if skip_cols:
        skip_cols_normalized = {unicodedata.normalize('NFKC', str(c)).strip().lower() for c in skip_cols}
        print(f"    [Stationarity Check] 标准化后的跳过列表 (首5项): {list(skip_cols_normalized)[:5]}")
    # --- END MODIFICATION ---

    for col in df.columns:
        # --- MODIFICATION: Check against normalized skip_cols ---
        col_normalized = unicodedata.normalize('NFKC', str(col)).strip().lower()
        # print(f"    [Debug Stationarity] Checking col: '{col}' (Normalized: '{col_normalized}') against skip list...") # Debug print
        if col_normalized in skip_cols_normalized:
            transformed_data[col] = df[col].copy() # Ensure we copy the data
            transform_log[col] = {'status': 'skipped_by_request'}
            print(f"    - {col}: 根据请求跳过平稳性检查 (匹配到规范化名称 '{col_normalized}').")
            continue
        # --- END MODIFICATION ---

        series = df[col]
        series_dropna = series.dropna()

        # --- NEW DEBUG for specific columns ---
        # debug_cols = ['中国：可再生能源：发电量（月）', '中国：火力发电：发电量（月）']
        # if col in debug_cols:
        #     print(f"      [DEBUG _ensure_stationarity] Processing column: {col}")
        #     print("        Original series (head):")
        #     print(series.head())
        #     print("        Original series (tail):")
        #     print(series.tail())
        #     print(f"        Original series length: {len(series)}")
        #     print("        Series after dropna() (head):")
        #     print(series_dropna.head())
        #     print("        Series after dropna() (tail):")
        #     print(series_dropna.tail())
        #     print(f"        Series after dropna() length: {len(series_dropna)}")
        # --- END NEW DEBUG ---

        if series_dropna.empty:
            transform_log[col] = {'status': 'skipped_empty'}
            removed_cols_info['skipped_empty'].append(col)
            print(f"    - {col}: 数据为空或全为 NaN，已移除.")
            continue

        if series_dropna.nunique() == 1:
            transform_log[col] = {'status': 'skipped_constant'}
            removed_cols_info['skipped_constant'].append(col)
            print(f"    - {col}: 列为常量，已移除.")
            continue

        original_pval = np.nan
        diff_pval = np.nan
        try:
            adf_result_level = adfuller(series_dropna)
            original_pval = adf_result_level[1]

            if original_pval < adf_p_threshold:
                transformed_data[col] = series
                transform_log[col] = {'status': 'level', 'original_pval': original_pval}
                # 🔥 优化：减少详细输出，只在必要时打印
                # print(f"    - {col}: Level 平稳 (p={original_pval:.3f}), 保留 Level.")
            else:
                # --- MODIFIED: Try Log Difference First ---
                series_orig = series # Keep original series
                series_transformed = None
                transform_type = 'diff' # Default to simple diff

                # Check if log difference is possible (all positive values)
                if (series_dropna > 0).all():
                    try:
                        series_transformed = np.log(series_orig).diff(1)
                        transform_type = 'log_diff'
                        # 🔥 优化：减少详细输出
                        # print(f"    - {col}: Level 不平稳 (p={original_pval:.3f}), 尝试对数差分...")
                    except Exception as e_log:
                         print(f"    - {col}: 对数差分出错: {e_log}. 回退到普通差分。")
                         series_transformed = series_orig.diff(1) # Fallback
                         transform_type = 'diff'
                else:
                    # print(f"    - {col}: Level 不平稳 (p={original_pval:.3f}), 包含非正值，使用普通一阶差分。")
                    series_transformed = series_orig.diff(1)
                    transform_type = 'diff'
                # --- END MODIFICATION ---

                series_transformed_dropna = series_transformed.dropna()

                # Check transformed series for empty or constant
                if series_transformed_dropna.empty:
                     transform_log[col] = {'status': f'skipped_{transform_type}_empty', 'original_pval': original_pval} # Use dynamic status
                     removed_cols_info[f'skipped_{transform_type}_empty'].append(col)
                     print(f"    - {col}: {transform_type.capitalize()} 后为空，已移除.") # Use dynamic message
                     continue
                if series_transformed_dropna.nunique() == 1:
                     transform_log[col] = {'status': f'skipped_{transform_type}_constant', 'original_pval': original_pval} # Use dynamic status
                     removed_cols_info[f'skipped_{transform_type}_constant'].append(col)
                     print(f"    - {col}: {transform_type.capitalize()} 后为常量，已移除.") # Use dynamic message
                     continue

                # Perform ADF test on the transformed series
                try:
                    adf_result_transformed = adfuller(series_transformed_dropna)
                    diff_pval = adf_result_transformed[1] # Store p-value from transformed series

                    transformed_data[col] = series_transformed # Assign the transformed series

                    if diff_pval < adf_p_threshold:
                        # Use dynamic status based on transform_type
                        transform_log[col] = {'status': transform_type, 'original_pval': original_pval, 'diff_pval': diff_pval}
                        print(f"    - {col}: {transform_type.capitalize()} 后平稳 (p={diff_pval:.3f}), 使用 {transform_type.capitalize()}.")
                    else:
                        # Use dynamic status based on transform_type
                        transform_log[col] = {'status': f'{transform_type}_still_nonstat', 'original_pval': original_pval, 'diff_pval': diff_pval}
                        print(f"    - {col}: {transform_type.capitalize()} 后仍不平稳 (p={diff_pval:.3f}), 使用 {transform_type.capitalize()}.")

                except Exception as e_diff:
                    print(f"    - {col}: 对 {transform_type.capitalize()} 序列 ADF 检验出错: {e_diff}. 保留 {transform_type.capitalize()} 序列.")
                    transformed_data[col] = series_transformed # Keep transformed series even if test fails
                    transform_log[col] = {'status': f'{transform_type}_test_error', 'original_pval': original_pval}

        except Exception as e_level:
            print(f"    - {col}: Level ADF 检验或处理时出错: {e_level}. 保留 Level (不推荐). ")
            transformed_data[col] = series
            transform_log[col] = {'status': 'level_test_error'}

    print(f"--- [Stationarity Check] 检查和转换完成. 输出 Shape: {transformed_data.shape} ---")
    total_removed = sum(len(v) for v in removed_cols_info.values())
    if total_removed > 0:
        print(f"  [!] 共移除了 {total_removed} 个变量:")
        for reason, cols in removed_cols_info.items():
             if cols:
                 print(f"      - 因 '{reason}' 移除 ({len(cols)} 个): {', '.join(cols[:5])}{'...' if len(cols)>5 else ''}")

    return transformed_data, transform_log, removed_cols_info

# --- NEW: Function to apply predefined stationarity transforms --- 
def apply_stationarity_transforms(
    data: pd.DataFrame,
    transform_rules: Dict[str, Dict[str, Any]] # 预期是 {var_name: {'status': 'level'/'diff'/'log_diff'/...}}
) -> pd.DataFrame:
    """
    根据提供的规则字典对 DataFrame 中的变量应用平稳性转换。
    如果某个变量在规则字典中找不到，则保留其原始值。

    Args:
        data: 包含原始（或预处理）数据的 DataFrame。
        transform_rules: 一个字典，键是变量名，值是包含转换状态的字典，
                         例如 {'status': 'level'}, {'status': 'diff'}, {'status': 'log_diff'}。
                         这个字典通常来自 _ensure_stationarity 的输出或模型元数据。

    Returns:
        一个包含应用转换后数据的新的 DataFrame，包含所有原始列。
    """
    print(f"\n--- [Apply Stationarity V2] 开始根据提供的规则应用平稳性转换 ---")
    transformed_data = pd.DataFrame(index=data.index) # 初始化空的 DataFrame
    applied_count = 0
    level_kept_count = 0 # 计数器：保留 Level (包括无规则或规则为 level)
    error_count = 0

    # --- 遍历输入数据的每一列 ---
    for col in data.columns:
        rule_info = transform_rules.get(col, None)
        status = 'level' # 默认保留 Level

        if rule_info and isinstance(rule_info, dict) and 'status' in rule_info:
            # 如果找到有效规则，则使用规则中的 status
            status = rule_info['status'].lower()
            # print(f"    - {col}: 找到规则 '{status}'.") # Debugging
        else:
            # 如果没有找到规则或规则无效，状态保持为 'level'
            # print(f"    - {col}: 未找到规则或规则无效，保留 Level.") # Debugging
            pass

        try:
            series = data[col]
            if status == 'diff':
                transformed_data[col] = series.diff(1)
                applied_count += 1
            elif status == 'log_diff':
                # 检查是否有非正值
                if (series <= 0).any():
                    print(f"    警告: 变量 '{col}' 包含非正值，无法应用 'log_diff'。将尝试普通 'diff'。")
                    transformed_data[col] = series.diff(1)
                    status = 'diff_fallback' # 标记实际操作
                    error_count += 1 # 算作一个需要注意的情况
                else:
                    transformed_data[col] = np.log(series).diff(1)
                    applied_count += 1
            else: # status == 'level' 或 其他未知/跳过状态
                transformed_data[col] = series.copy() # 保留原始序列
                level_kept_count += 1

        except Exception as e:
            print(f"    错误: 应用规则 '{status}' 到变量 '{col}' 时出错: {e}. 将保留原序列。")
            transformed_data[col] = data[col].copy() # 出错时保留原序列
            error_count += 1
            level_kept_count += 1 # 出错也算保留了 Level

    print(f"--- [Apply Stationarity V2] 转换应用完成. ---")
    print(f"    成功应用 'diff'/'log_diff': {applied_count} 个变量")
    print(f"    保留 Level (无规则或规则指示): {level_kept_count} 个变量")
    print(f"    转换时出错/回退 (保留 Level 或应用 Diff): {error_count} 个变量")
    print(f"    输入 Shape: {data.shape}, 输出 Shape: {transformed_data.shape}")

    # 移除转换后全为 NaN 的列 (这通常只发生在差分后的第一行，理论上不应移除整个列)
    # 但为了安全起见，保留此检查
    all_nan_cols = transformed_data.columns[transformed_data.isnull().all()].tolist()
    if all_nan_cols:
        print(f"    警告：以下列在转换后全为 NaN，将被移除: {all_nan_cols}")
        transformed_data = transformed_data.drop(columns=all_nan_cols)
        print(f"    移除全 NaN 列后 Shape: {transformed_data.shape}")

    # 确保输出包含所有原始列（即使转换失败也保留原列）
    if set(transformed_data.columns) != set(data.columns):
         print("    警告：输出列与输入列不完全匹配！正在尝试重新对齐...")
         transformed_data = transformed_data.reindex(columns=data.columns, fill_value=np.nan) # 可能需要更复杂的填充

    return transformed_data

# --- REVISED: prepare_data function (V3 Logic) ---
def prepare_data(
    excel_path: str,
    target_freq: str,
    target_sheet_name: str,
    target_variable_name: str, # Keep this as input for initial identification
    consecutive_nan_threshold: Optional[int] = None,
    data_start_date: Optional[str] = None,
    data_end_date: Optional[str] = None,
    reference_sheet_name: str = '指标体系',
    reference_column_name: str = '高频指标'
) -> Tuple[Optional[pd.DataFrame], Optional[Dict], Optional[Dict], Optional[List[Dict]]]:
    """
    Loads data, performs stationarity checks at appropriate frequencies (monthly vars
    at monthly freq), aligns all data to target_freq (weekly), performs NaN checks
    (skipping monthly vars), and weekly stationarity checks (skipping monthly vars).

    Args:
        (Parameters remain the same)

    Returns:
        Tuple[Optional[pd.DataFrame], Optional[Dict], Optional[Dict], Optional[List[Dict]]]:
            - Final aligned weekly data (DataFrame).
            - Variable-to-industry mapping (Dict).
            - Combined transformation log (Dict).
            - Detailed removal log (List[Dict]).
    """
    print(f"\n--- [Data Prep V3] 开始加载和处理数据 (目标频率: {target_freq}) ---")
    if CREATE_REDUCED_TEST_SET:
        print("  [!] 已启用缩小版测试集生成模式。")

    if not target_freq.upper().endswith('-FRI'):
        print(f"错误: [Data Prep] 当前目标对齐逻辑仅支持周五 (W-FRI)。提供的目标频率 '{target_freq}' 无效。")
        return None, None, None, None

    # Use original target name for clarity in logs, but read actual B col name later
    print(f"  [Data Prep] 目标 Sheet: '{target_sheet_name}', 目标变量名(预期B列): '{target_variable_name}'")

    var_industry_map = {}
    raw_columns_across_all_sheets = set() # Track normalized names of ALL predictors loaded
    reference_predictor_variables = set()
    target_sheet_cols = set() # Track ORIGINAL names from target sheet (B col + C+ cols)

    try:
        excel_file = pd.ExcelFile(excel_path)
        available_sheets = excel_file.sheet_names
        print(f"  [Data Prep] Excel 文件中可用的 Sheets: {available_sheets}")

        # --- Load Reference Variables (same as before) ---
        if reference_sheet_name in available_sheets:
            try:
                ref_df = pd.read_excel(excel_file, sheet_name=reference_sheet_name)
                ref_df.columns = ref_df.columns.str.strip()
                clean_reference_column_name = reference_column_name.strip()
                if clean_reference_column_name in ref_df.columns:
                    raw_reference_vars = (
                        ref_df[clean_reference_column_name]
                        .astype(str).str.strip().replace('nan', np.nan).dropna().unique()
                    )
                    raw_reference_vars = [v for v in raw_reference_vars if v]
                    reference_predictor_variables = set(
                        unicodedata.normalize('NFKC', var).strip().lower()
                        for var in raw_reference_vars
                    )
                    print(f"  [Data Prep] 从 '{reference_sheet_name}' 加载并规范化了 {len(reference_predictor_variables)} 个参考变量名。")
                else:
                    print(f"  [Data Prep] 警告: 在 '{reference_sheet_name}' 未找到参考列 '{clean_reference_column_name}'。")
            except Exception as e_ref:
                print(f"  [Data Prep] 警告: 读取参考 Sheet '{reference_sheet_name}' 出错: {e_ref}。")
        else:
             print(f"  [Data Prep] 警告: 未找到参考 Sheet '{reference_sheet_name}'。")

        data_parts = defaultdict(list)
        # --- Variables for monthly data processing --- 
        publication_dates_from_target = None
        raw_target_values = None # <-- Separate target Series
        # df_all_monthly_predictors_pubdate = pd.DataFrame() # OLD: Combine ALL non-target monthly predictors here
        df_other_monthly_predictors_pubdate = pd.DataFrame() # <-- NEW: Combine only predictors from OTHER monthly sheets
        df_target_sheet_predictors_pubdate = pd.DataFrame() # <-- NEW: Store predictors from target sheet separately
        target_sheet_industry_name = "Macro"
        actual_target_variable_name = None # Will store the actual name from Col B
        # monthly_predictor_cols_original = set() # <-- Track original names from predictor monthlies - Handled differently now
        target_sheet_cols = set() # Keep this to track target sheet columns for skipping later

        # --- Step 1: Load Data by Frequency ---
        print("\n--- [Data Prep V3] 步骤 1: 加载数据 ---")
        for sheet_name in available_sheets:
            print(f"    [Data Prep] 正在检查 Sheet: {sheet_name}...")
            is_target_sheet = (sheet_name == target_sheet_name)
            sheet_info = parse_sheet_info(sheet_name, target_sheet_name)
            freq_type = sheet_info['freq_type']
            industry_name = sheet_info['industry'] if sheet_info['industry'] else "Uncategorized"

            # 1a: Handle Target Sheet (Extract Monthly Data)
            if is_target_sheet:
                print(f"      检测到目标 Sheet ('{freq_type}', 行业: '{industry_name}')...")
                target_sheet_industry_name = industry_name
                try:
                    # --- NEW: Detect target sheet format and use appropriate parameters ---
                    format_info = detect_target_sheet_format(excel_file, sheet_name)
                    read_header = format_info['header']
                    skip_rows = format_info['skiprows']
                    # 目标Sheet默认不设index_col，日期在后续处理
                    print(f"    [目标Sheet读取参数] 格式: {format_info['format']}, header={read_header}, skiprows={skip_rows}")
                    # --- Read with detected format parameters ---
                    df_raw = pd.read_excel(excel_file, sheet_name=sheet_name, header=read_header, skiprows=skip_rows)
                    # --- END NEW ---
                    # --- <<< 新增：在正确位置打印更多原始读取信息 >>> ---
                    print("      [Debug Target Raw Read - FULL] 原始读取的目标 Sheet (前 5 行):")
                    print(df_raw.head())
                    print("      [Debug Target Raw Read - FULL] 原始读取的目标 Sheet (后 5 行):")
                    print(df_raw.tail())
                    print(f"      [Debug Target Raw Read - FULL] 原始读取的 Shape: {df_raw.shape}")
                    # --- <<< 结束新增 >>> ---

                    # --- <<< 新增：将 0 值替换为 NaN >>> ---
                    df_raw = df_raw.replace(0, np.nan)
                    print(f"      [处理] 将目标 Sheet '{sheet_name}' 中的 0 值替换为 NaN。")
                    # --- <<< 结束新增 >>> ---

                    if df_raw.shape[1] < 2:
                        print(f"      错误: 目标 Sheet '{sheet_name}' 列数 < 2。跳过。")
                        continue

                    date_col_name = df_raw.columns[0]
                    actual_target_variable_name = df_raw.columns[1] # Use actual name from sheet B
                    target_sheet_cols.add(actual_target_variable_name) # <-- Corrected variable name

                    print(f"      确认目标变量 (B列): '{actual_target_variable_name}'")
                    print(f"      解析发布日期 (A列: '{date_col_name}')...")
                    publication_dates_from_target = pd.to_datetime(df_raw[date_col_name], errors='coerce')
                    valid_date_mask = publication_dates_from_target.notna()
                    if not valid_date_mask.any():
                        print(f"      错误: 无法从列 '{date_col_name}' 解析任何有效日期。跳过目标Sheet。")
                        continue
                    publication_dates_from_target = publication_dates_from_target[valid_date_mask] # Keep only valid dates

                    # --- <<< 新增：打印原始读取的最后几行和日期解析情况 >>> ---
                    print("      [Debug Target Raw Read] 原始读取的目标 Sheet 最后 5 行:")
                    print(df_raw.tail()[[date_col_name, actual_target_variable_name]])
                    print("      [Debug Target Raw Read] 对应日期列解析结果 (valid_date_mask):")
                    print(valid_date_mask.tail())
                    # --- <<< 结束新增 >>> ---

                    print(f"      提取目标变量原始值...")
                    raw_target_values = pd.to_numeric(df_raw.loc[valid_date_mask, actual_target_variable_name], errors='coerce')
                    raw_target_values.index = publication_dates_from_target # Index by Pub Date

                    # Update maps for target var
                    norm_target_name = unicodedata.normalize('NFKC', actual_target_variable_name).strip().lower()
                    var_industry_map[norm_target_name] = target_sheet_industry_name

                    # Extract Monthly Predictors (Cols C+) indexed by PUBLICATION DATE
                    if df_raw.shape[1] > 2:
                        print(f"      提取目标 Sheet 的月度预测变量 (C列及以后)...")
                        temp_monthly_predictors = {}
                        monthly_preds_from_target_sheet = pd.DataFrame() # <-- Temp DF for this sheet's preds
                        # --- <<< 添加：在循环前移除原始 DataFrame 中的 Unnamed 列 (从第3列开始检查) >>> ---
                        unnamed_cols_target_pred = [col for col in df_raw.columns[2:] if isinstance(col, str) and col.startswith('Unnamed:')]
                        if unnamed_cols_target_pred:
                            print(f"      [清理 Target Sheet Predictors] 在 \'{sheet_name}\' 中发现并移除 Unnamed 列: {unnamed_cols_target_pred}")
                            df_raw = df_raw.drop(columns=unnamed_cols_target_pred)
                        # --- <<< 结束添加 >>> ---
                        for col_idx in range(2, df_raw.shape[1]): # 重新检查 shape[1] 因为列可能已被移除
                            col_name = df_raw.columns[col_idx]
                            target_sheet_cols.add(col_name) # Track original name
                            # Clean values
                            cleaned_series = df_raw[col_name].astype(str).str.replace('%', '', regex=False).str.replace(',', '', regex=False).str.strip()
                            predictor_values = pd.to_numeric(cleaned_series, errors='coerce')
                            # Create Series indexed by PUBLICATION DATE (aligned to valid dates)
                            temp_monthly_predictors[col_name] = pd.Series(
                                predictor_values[valid_date_mask].values,
                                index=publication_dates_from_target
                            )
                            # Update maps and tracking
                            norm_pred_col = unicodedata.normalize('NFKC', str(col_name)).strip().lower()
                            if norm_pred_col:
                                var_industry_map[norm_pred_col] = target_sheet_industry_name
                                raw_columns_across_all_sheets.add(norm_pred_col)

                        monthly_preds_from_target_sheet = pd.DataFrame(temp_monthly_predictors).sort_index()
                        monthly_preds_from_target_sheet = monthly_preds_from_target_sheet.dropna(axis=1, how='all')
                        # --- NEW: Add these predictors to the combined monthly predictor DF ---
                        if not monthly_preds_from_target_sheet.empty:
                            # df_all_monthly_predictors_pubdate = pd.concat([df_all_monthly_predictors_pubdate, monthly_preds_from_target_sheet], axis=1, join='outer') # OLD
                            df_target_sheet_predictors_pubdate = pd.concat([df_target_sheet_predictors_pubdate, monthly_preds_from_target_sheet], axis=1, join='outer')
                        # --- END NEW ---
                        print(f"      提取了 {monthly_preds_from_target_sheet.shape[1]} 个有效的月度预测变量 (按发布日期索引)。")
                    else:
                        print(f"      目标 Sheet 仅含 A, B 列。")

                except Exception as e:
                    print(f"      加载或处理目标 Sheet '{sheet_name}' 出错: {e}. 跳过。")
                    publication_dates_from_target = None
                    raw_target_values = None
                    # df_all_monthly_predictors_pubdate = None
                    continue

            # 1b, 1c, 1d: Load Daily, Weekly, Other Monthly Predictors (logic mostly unchanged)
            elif freq_type in ['daily', 'weekly']:
                print(f"      检测到预测变量 Sheet ('{freq_type}', 行业: '{industry_name}')...")
                try:
                    # --- NEW: 使用通用格式检测 ---
                    format_info = detect_sheet_format(excel_file, sheet_name)
                    read_header = format_info['header']
                    skip_rows = format_info['skiprows']
                    read_index_col = 0  # Always assume index is the first column
                    
                    print(f"        [格式检测结果] {format_info['format']} (来源: {format_info['source']})")
                    print(f"        [读取参数] header={read_header}, skiprows={skip_rows}, index_col={read_index_col}")
                    # --- End NEW Logic ---

                    df = pd.read_excel(excel_file, sheet_name=sheet_name, header=read_header, skiprows=skip_rows, index_col=read_index_col, parse_dates=True)

                    # --- <<< 新增：将 0 值替换为 NaN (日度/周度) >>> ---
                    df = df.replace(0, np.nan)
                    print(f"      [处理] 将 Sheet '{sheet_name}' 中的 0 值替换为 NaN。")
                    # --- <<< 结束新增 >>> ---

                    # --- <<< 添加：在处理前移除 Unnamed 列 >>> ---
                    unnamed_cols_dw = [col for col in df.columns if isinstance(col, str) and col.startswith('Unnamed:')]
                    if unnamed_cols_dw:
                        print(f"      [清理 Daily/Weekly] 在 \'{sheet_name}\' 中发现并移除 Unnamed 列: {unnamed_cols_dw}")
                        df = df.drop(columns=unnamed_cols_dw)
                    # --- <<< 结束添加 >>> ---

                    # --- <<< 新增：打印加载后的列名 >>> ---
                    print(f"      [Debug Columns Loaded] Sheet: '{sheet_name}', Loaded Columns: {df.columns.tolist()}")
                    # --- <<< 结束新增 >>> ---

                    # --- NEW: Force index to datetime and handle errors ---
                    print(f"      Attempting to convert index of '{sheet_name}' to datetime...")
                    original_index_len = len(df.index)
                    df.index = pd.to_datetime(df.index, errors='coerce')
                    # --- END NEW ---

                    if df is None or df.empty: continue
                    df = df.loc[df.index.notna()] # Filter rows where index conversion failed (became NaT)
                    filtered_index_len = len(df.index)
                    if filtered_index_len < original_index_len:
                        print(f"      警告: 在 '{sheet_name}' 中移除了 {original_index_len - filtered_index_len} 行，因为它们的索引无法解析为有效日期。")

                    df = df.dropna(axis=1, how='all')
                    if df.empty: continue
                    df_numeric = df.apply(pd.to_numeric, errors='coerce')
                    if df_numeric.empty or df_numeric.isnull().all().all(): continue
                    print(f"      Sheet '{sheet_name}' ({industry_name}, {freq_type}) 加载完成。 Shape: {df_numeric.shape}")
                    data_parts[freq_type].append(df_numeric)
                    for col in df_numeric.columns:
                        norm_col = unicodedata.normalize('NFKC', str(col)).strip().lower()
                        if norm_col:
                            var_industry_map[norm_col] = industry_name
                            raw_columns_across_all_sheets.add(norm_col)
                except Exception as e:
                    print(f"      加载或处理 {freq_type} Sheet '{sheet_name}' 时出错: {e}. 跳过。")
                    continue
            elif freq_type == 'monthly_predictor':
                print(f"      检测到非目标月度预测 Sheet ('{freq_type}', 行业: '{industry_name}')...")
                try:
                    # --- NEW: 使用通用格式检测 ---
                    format_info = detect_sheet_format(excel_file, sheet_name)
                    read_header = format_info['header']
                    skip_rows = format_info['skiprows']
                    
                    print(f"        [格式检测结果] {format_info['format']} (来源: {format_info['source']})")
                    print(f"        [读取参数] header={read_header}, skiprows={skip_rows}")
                    # --- End NEW Logic ---

                    df_raw_pred = pd.read_excel(excel_file, sheet_name=sheet_name, header=read_header, skiprows=skip_rows)

                    # --- <<< 添加：在处理前移除 Unnamed 列 >>> ---
                    unnamed_cols_m_pred = [col for col in df_raw_pred.columns if isinstance(col, str) and col.startswith('Unnamed:')]
                    if unnamed_cols_m_pred:
                        print(f"      [清理 Monthly Predictor] 在 \'{sheet_name}\' 中发现并移除 Unnamed 列: {unnamed_cols_m_pred}")
                        df_raw_pred = df_raw_pred.drop(columns=unnamed_cols_m_pred)
                    # --- <<< 结束添加 >>> ---

                    # --- <<< 新增：将 0 值替换为 NaN >>> ---
                    df_raw_pred = df_raw_pred.replace(0, np.nan)
                    print(f"      [处理] 将其他月度预测 Sheet '{sheet_name}' 中的 0 值替换为 NaN。")
                    # --- <<< 结束新增 >>> ---

                    if df_raw_pred.shape[1] < 2:
                        print(f"      错误: 月度预测 Sheet '{sheet_name}' 列数 < 2。跳过。")
                        continue

                    date_col_name_pred = df_raw_pred.columns[0]
                    print(f"      解析发布日期 (A列: '{date_col_name_pred}')...")
                    publication_dates_predictor = pd.to_datetime(df_raw_pred[date_col_name_pred], errors='coerce')
                    valid_date_mask_pred = publication_dates_predictor.notna()
                    if not valid_date_mask_pred.any():
                        print(f"      错误: 无法从列 '{date_col_name_pred}' 解析任何有效日期。跳过此Sheet。")
                        continue
                    publication_dates_predictor = publication_dates_predictor[valid_date_mask_pred]

                    print(f"      提取月度预测变量 (B列及以后)...")
                    temp_monthly_predictors_sheet = {}
                    for col_idx_pred in range(1, df_raw_pred.shape[1]): # Start from Col B (index 1)
                        col_name_pred = df_raw_pred.columns[col_idx_pred]
                        # Clean values
                        cleaned_series_pred = df_raw_pred[col_name_pred].astype(str).str.replace('%', '', regex=False).str.replace(',', '', regex=False).str.strip()
                        predictor_values_pred = pd.to_numeric(cleaned_series_pred, errors='coerce')
                        # Create Series indexed by PUBLICATION DATE (aligned to valid dates)
                        temp_monthly_predictors_sheet[col_name_pred] = pd.Series(
                            predictor_values_pred[valid_date_mask_pred].values,
                            index=publication_dates_predictor
                        )
                        # Update maps and tracking
                        norm_pred_col_p = unicodedata.normalize('NFKC', str(col_name_pred)).strip().lower()
                        if norm_pred_col_p:
                            var_industry_map[norm_pred_col_p] = industry_name # Use industry from this sheet
                            raw_columns_across_all_sheets.add(norm_pred_col_p)

                    df_monthly_pred_sheet = pd.DataFrame(temp_monthly_predictors_sheet).sort_index()
                    df_monthly_pred_sheet = df_monthly_pred_sheet.dropna(axis=1, how='all')
                    if not df_monthly_pred_sheet.empty:
                        # --- NEW: Add these predictors to the combined monthly predictor DF ---
                        # --- NEW: Add predictors from OTHER monthly sheets to their dedicated DF ---
                        df_other_monthly_predictors_pubdate = pd.concat([df_other_monthly_predictors_pubdate, df_monthly_pred_sheet], axis=1, join='outer')
                        # --- END NEW ---
                        print(f"      提取了 {df_monthly_pred_sheet.shape[1]} 个有效的月度预测变量 (按发布日期索引)。")
                    else:
                        print("      此 Sheet 未包含有效的月度预测变量数据。")

                except Exception as e_pred:
                    print(f"      加载或处理月度预测 Sheet '{sheet_name}' 出错: {e_pred}. 跳过。")
                    continue
            # 1e: Ignore other sheets
            else:
                 if sheet_name != reference_sheet_name:
                     print(f"      Sheet '{sheet_name}' 不符合要求或非目标 Sheet，已跳过。")
                 continue

        # --- Check if essential data was loaded ---
        if raw_target_values is None or raw_target_values.empty or publication_dates_from_target is None:
             print(f"错误：[Data Prep] 未能成功加载目标变量 '{target_variable_name}' 或其发布日期。")
             return None, None, None, None
        # Only warn if predictors are missing, target is essential  
        if (not data_parts['daily'] and not data_parts['weekly'] and 
            (df_other_monthly_predictors_pubdate is None or df_other_monthly_predictors_pubdate.empty) and
            (df_target_sheet_predictors_pubdate is None or df_target_sheet_predictors_pubdate.empty)):
             print("警告：[Data Prep] 未能加载任何有效的日度、周度或月度预测变量。")

        # +++++++++++++ DEBUG: Find missing variable +++++++++++++
        if reference_predictor_variables and raw_columns_across_all_sheets:
            missing_from_data_sheets = reference_predictor_variables - raw_columns_across_all_sheets
            if missing_from_data_sheets:
                print(f"\n+++ DEBUG: 变量存在于指标体系但未在数据Sheet中找到: {missing_from_data_sheets} +++\n")
            else:
                print("\n+++ DEBUG: 所有指标体系变量都在加载的数据中找到了。 +++\n")
        # +++++++++++++ END DEBUG +++++++++++++

        # Initialize lists/sets for tracking
        removed_variables_detailed_log = [] # List of dicts {'Variable': name, 'Reason': reason_code}
        all_indices_for_range = [] # Collect all datetime indices to determine full range

        # --- Step 2: Target Variable Alignment (Nearest Friday) ---
        print("\n--- [Data Prep V3] 步骤 2: 目标变量处理与对齐 (最近周五) ---")
        target_series_aligned_nearest_friday = pd.Series(dtype=float)
        if raw_target_values is not None and not raw_target_values.empty:
            temp_target_df = pd.DataFrame({'value': raw_target_values})
            # Calculate the nearest Friday for each publication date
            # If weekday is Mon, Tue, Wed -> go to upcoming Fri (4 - weekday)
            # If weekday is Thu, Fri, Sat, Sun -> go to previous Fri (4 - weekday)
            # Note: Python's weekday() is Mon=0, Tue=1, ..., Fri=4, Sat=5, Sun=6
            temp_target_df['nearest_friday'] = temp_target_df.index.map(lambda dt: dt + pd.Timedelta(days=4 - dt.weekday()))
            # Handle duplicates for the same target Friday: keep the one with the LATEST publication date
            # We sort by the original publication date index FIRST, then group and take the last.
            target_series_aligned_nearest_friday = temp_target_df.sort_index(ascending=True).groupby('nearest_friday')['value'].last()
            target_series_aligned_nearest_friday.index.name = 'Date'
            target_series_aligned_nearest_friday.name = actual_target_variable_name # Ensure Series has the correct name
            print(f"  目标变量对齐到最近周五完成。Shape: {target_series_aligned_nearest_friday.shape}")
            # --- <<< 新增：打印对齐后的最后几行 >>> ---
            print("    [Debug Target Align] 对齐后目标变量最后 5 行:")
            print(target_series_aligned_nearest_friday.tail())
            # --- <<< 结束新增 >>> ---
            if not target_series_aligned_nearest_friday.empty:
                all_indices_for_range.append(target_series_aligned_nearest_friday.index)
        else:
             print("  未加载目标变量，无法进行对齐。")
             # If target is essential, we should probably return error here, but we already checked above.

        # --- NEW Step 2.5: Target Sheet Predictors Alignment (Nearest Friday) ---
        print("\\n--- [Data Prep V3] 步骤 2.5: 目标 Sheet 预测变量处理与对齐 (最近周五) ---")
        target_sheet_predictors_aligned_nearest_friday = pd.DataFrame()
        target_sheet_predictor_cols = set() # Track original names from target sheet C+
        if df_target_sheet_predictors_pubdate is not None and not df_target_sheet_predictors_pubdate.empty:
            # Ensure no duplicate columns before processing
            cols_before_dedup_tsp = set(df_target_sheet_predictors_pubdate.columns)
            df_target_sheet_predictors_pubdate = df_target_sheet_predictors_pubdate.loc[:, ~df_target_sheet_predictors_pubdate.columns.duplicated(keep='first')]
            cols_after_dedup_tsp = set(df_target_sheet_predictors_pubdate.columns)
            removed_cols_dedup_tsp = cols_before_dedup_tsp - cols_after_dedup_tsp
            if removed_cols_dedup_tsp:
                 print(f"    警告: 在处理目标 Sheet 预测变量前因重复移除了 {len(removed_cols_dedup_tsp)} 列: {list(removed_cols_dedup_tsp)[:5]}...")
                 for col in removed_cols_dedup_tsp:
                     if not any(d['Variable'] == col for d in removed_variables_detailed_log):
                         removed_variables_detailed_log.append({'Variable': col, 'Reason': 'target_sheet_predictor_duplicate'})

            # Apply the same nearest Friday logic as the target variable
            temp_tsp_df = df_target_sheet_predictors_pubdate.copy()
            temp_tsp_df['nearest_friday'] = temp_tsp_df.index.map(lambda dt: dt + pd.Timedelta(days=4 - dt.weekday()))
            # Handle duplicates for the same target Friday: keep the one with the LATEST publication date
            target_sheet_predictors_aligned_nearest_friday = temp_tsp_df.sort_index(ascending=True).groupby('nearest_friday').last()
            target_sheet_predictors_aligned_nearest_friday.index.name = 'Date'
            # No need to rename columns, they keep their original names
            print(f"  目标 Sheet 预测变量 ({len(target_sheet_predictors_aligned_nearest_friday.columns)} 个) 对齐到最近周五完成。Shape: {target_sheet_predictors_aligned_nearest_friday.shape}")
            if not target_sheet_predictors_aligned_nearest_friday.empty:
                all_indices_for_range.append(target_sheet_predictors_aligned_nearest_friday.index)
                target_sheet_predictor_cols = set(target_sheet_predictors_aligned_nearest_friday.columns)
        else:
            print("  目标 Sheet 中未包含其他预测变量 (C列及以后)，或加载失败。")

        # --- Step 3: Other Monthly Predictors Processing (Last Friday of Month) ---
        print("\\n--- [Data Prep V3] 步骤 3: 其他月度预测变量处理与对齐 (月末最后周五) ---")
        monthly_predictors_aligned_last_friday = pd.DataFrame()
        monthly_transform_log = {}
        other_monthly_predictors_to_skip_weekly_stationarity = set() # Track columns from THIS step only
        # if df_all_monthly_predictors_pubdate is not None and not df_all_monthly_predictors_pubdate.empty: # OLD
        if df_other_monthly_predictors_pubdate is not None and not df_other_monthly_predictors_pubdate.empty:
            # 3a: Aggregate to Month End using last()
            print("  聚合其他来源的月度预测变量到月末 (取当月最后有效值)...")
            # Ensure no duplicate columns before processing
            # cols_before_dedup_monthly = set(df_all_monthly_predictors_pubdate.columns) # OLD
            cols_before_dedup_monthly = set(df_other_monthly_predictors_pubdate.columns)
            # df_all_monthly_predictors_pubdate = df_all_monthly_predictors_pubdate.loc[:, ~df_all_monthly_predictors_pubdate.columns.duplicated(keep='first')] # OLD
            df_other_monthly_predictors_pubdate = df_other_monthly_predictors_pubdate.loc[:, ~df_other_monthly_predictors_pubdate.columns.duplicated(keep='first')]
            # cols_after_dedup_monthly = set(df_all_monthly_predictors_pubdate.columns) # OLD
            cols_after_dedup_monthly = set(df_other_monthly_predictors_pubdate.columns)
            removed_cols_dedup_monthly = cols_before_dedup_monthly - cols_after_dedup_monthly
            if removed_cols_dedup_monthly:
                 print(f"    警告: 在聚合前因重复移除了 {len(removed_cols_dedup_monthly)} 个其他月度预测变量列: {list(removed_cols_dedup_monthly)[:5]}...")
                 for col in removed_cols_dedup_monthly:
                     if not any(d['Variable'] == col for d in removed_variables_detailed_log):
                         removed_variables_detailed_log.append({'Variable': col, 'Reason': 'other_monthly_predictor_duplicate'})

            # df_monthly_predictors_for_stat = df_all_monthly_predictors_pubdate.copy() # OLD
            df_monthly_predictors_for_stat = df_other_monthly_predictors_pubdate.copy()
            df_monthly_predictors_for_stat['MonthIndex'] = df_monthly_predictors_for_stat.index.to_period('M').to_timestamp('M')
            df_monthly_predictors_for_stat = df_monthly_predictors_for_stat.groupby('MonthIndex').last()
            df_monthly_predictors_for_stat = df_monthly_predictors_for_stat.sort_index()
            print(f"    聚合到月末完成. Shape: {df_monthly_predictors_for_stat.shape}")

            # 3b: Monthly NaN Check (Applied to OTHER monthly predictors)
            if consecutive_nan_threshold is not None and consecutive_nan_threshold > 0:
                print(f"  [月度检查] 开始检查其他来源月度预测变量的连续缺失值 (阈值 >= {consecutive_nan_threshold})...")
                # ... (NaN checking logic remains the same, applied to df_monthly_predictors_for_stat) ...
                initial_cols_monthly_nan = set(df_monthly_predictors_for_stat.columns)
                cols_to_remove_monthly_nan_pred = []
                for col in df_monthly_predictors_for_stat.columns:
                    series = df_monthly_predictors_for_stat[col]
                    first_valid_idx = series.first_valid_index()
                    if first_valid_idx is None: continue # Skip if column is all NaN already
                    series_after_first_valid = series.loc[first_valid_idx:]
                    is_na = series_after_first_valid.isna()
                    # Calculate consecutive NaNs
                    na_blocks = is_na.ne(is_na.shift()).cumsum()[is_na]
                    max_consecutive_nan = 0
                    if not na_blocks.empty:
                         try:
                             # Calculate counts of consecutive blocks
                             block_counts = na_blocks.value_counts()
                             if not block_counts.empty:
                                  max_consecutive_nan = block_counts.max()
                             else: # Handle edge case where na_blocks is not empty but value_counts is
                                 max_consecutive_nan = 0
                         except Exception as e_nan_count: # Catch potential errors in value_counts
                             print(f"    [月度检查] 警告: 计算 '{col}' 的 NaN 块时出错: {e_nan_count}. 跳过此列检查.")
                             continue # Skip this column if counting fails

                    if max_consecutive_nan >= consecutive_nan_threshold:
                        cols_to_remove_monthly_nan_pred.append(col)
                        print(f"    [月度检查] 标记移除变量: '{col}' (最大连续 NaN: {max_consecutive_nan} >= {consecutive_nan_threshold})", end='')
                        if not any(d['Variable'] == col for d in removed_variables_detailed_log):
                             removed_variables_detailed_log.append({'Variable': col, 'Reason': 'other_monthly_predictor_consecutive_nan'})
                             print(" - 已记录移除")
                        else: print(" - 已在其他步骤记录")

                if cols_to_remove_monthly_nan_pred:
                    print(f"\n    [月度检查] 正在移除 {len(cols_to_remove_monthly_nan_pred)} 个月度预测变量...")
                    df_monthly_predictors_for_stat = df_monthly_predictors_for_stat.drop(columns=cols_to_remove_monthly_nan_pred)
                    print(f"      移除后 Shape: {df_monthly_predictors_for_stat.shape}")
                else:
                     print(f"    [月度检查] 所有其他月度预测变量的连续缺失值均低于阈值。")
            else:
                print("  (跳过/禁用) 月度预测变量连续缺失值检查。")

            # 3c: Monthly Stationarity Check
            if not df_monthly_predictors_for_stat.empty:
                print("\n--- [月度预测变量平稳性检查] ---")
                df_monthly_predictors_stationary, monthly_transform_log, removed_cols_info_monthly_pred = _ensure_stationarity(
                    df_monthly_predictors_for_stat,
                    skip_cols=None, # Check all monthly predictors
                    adf_p_threshold=0.05
                )
                print(f"    月度预测变量平稳性处理完成。处理后 Shape: {df_monthly_predictors_stationary.shape}")
                # --- FIX: Record columns to skip BEFORE adding temporary columns ---
                if not df_monthly_predictors_stationary.empty:
                     other_monthly_predictors_to_skip_weekly_stationarity = set(df_monthly_predictors_stationary.columns)
                else:
                     other_monthly_predictors_to_skip_weekly_stationarity = set()
                # --- END FIX ---

                # Log stationarity removals
                for reason, cols in removed_cols_info_monthly_pred.items():
                    for col in cols:
                        if not any(d['Variable'] == col for d in removed_variables_detailed_log):
                            # Log reason as 'other_monthly_predictor_stationarity...'
                            removed_variables_detailed_log.append({'Variable': col, 'Reason': f'other_monthly_predictor_stationarity_{reason}'})

                # 3d: Align to Last Friday of the Month
                if not df_monthly_predictors_stationary.empty:
                    print("  对齐处理后的月度预测变量到当月最后周五...")
                    # Calculate the last Friday of the month for each index (which is month end)
                    df_monthly_predictors_stationary['last_friday'] = df_monthly_predictors_stationary.index.map(
                        lambda dt: dt - pd.Timedelta(days=(dt.weekday() - 4 + 7) % 7) # Go back to the last Friday
                    )
                    monthly_predictors_aligned_last_friday = df_monthly_predictors_stationary.set_index('last_friday', drop=True)
                    monthly_predictors_aligned_last_friday.index.name = 'Date'
                    # Handle potential duplicates for the same target Friday (if multiple month-ends map to the same last Friday)
                    # Keep the LATEST month's data in case of overlap (though unlikely with month-end index)
                    monthly_predictors_aligned_last_friday = monthly_predictors_aligned_last_friday[
                        ~monthly_predictors_aligned_last_friday.index.duplicated(keep='last')
                    ]
                    monthly_predictors_aligned_last_friday = monthly_predictors_aligned_last_friday.sort_index()
                    print(f"    对齐到最后周五完成。 Shape: {monthly_predictors_aligned_last_friday.shape}")
                    if not monthly_predictors_aligned_last_friday.empty:
                        all_indices_for_range.append(monthly_predictors_aligned_last_friday.index)
                    # Record columns that originated from OTHER monthly predictors to skip weekly stationarity later
                    # other_monthly_predictors_to_skip_weekly_stationarity = set(df_monthly_predictors_stationary.columns) # OLD BUGGY LINE
                    # --- Corrected log message using the set populated earlier ---
                    print(f"    将记录 {len(other_monthly_predictors_to_skip_weekly_stationarity)} 个来自其他月度源的列用于跳过后续周度平稳性检查。")
                else:
                    print("  没有来自其他月度源的平稳化预测变量可供对齐。")
            else: # If df_monthly_predictors_for_stat was empty after NaN check
                print("  没有月度预测变量进行平稳性检查 (可能因连续 NaN 被移除)。")
        else: # If df_other_monthly_predictors_pubdate was None or empty initially
            print("  没有其他月度预测变量需要处理。")

        # --- Step 4: Daily/Weekly Data Processing ---
        print(f"\n--- [Data Prep V3] 步骤 4: 日度和周度数据处理 ({target_freq}) ---")
        df_daily_weekly_mean = pd.DataFrame()
        df_weekly_aligned = pd.DataFrame()
        df_combined_dw_weekly = pd.DataFrame() # Initialize

        # 4a: Daily -> Weekly (Mean)
        if data_parts['daily']:
            print("  处理日度数据 -> 周度 (均值)...")
            df_daily_full = pd.concat(data_parts['daily'], axis=1)
            # Handle duplicate columns from different daily sheets before resampling
            cols_before_dedup_daily = set(df_daily_full.columns)
            df_daily_full = df_daily_full.loc[:, ~df_daily_full.columns.duplicated(keep='first')]
            cols_after_dedup_daily = set(df_daily_full.columns)
            removed_cols_dedup_daily = cols_before_dedup_daily - cols_after_dedup_daily
            if removed_cols_dedup_daily:
                 print(f"    警告: 在合并日度数据时因重复移除了 {len(removed_cols_dedup_daily)} 列: {list(removed_cols_dedup_daily)[:5]}...")
                 for col in removed_cols_dedup_daily:
                     if not any(d['Variable'] == col for d in removed_variables_detailed_log):
                         removed_variables_detailed_log.append({'Variable': col, 'Reason': 'daily_duplicate_column'})

            if not df_daily_full.empty:
                 # Add original daily index to range calculation *before* resampling
                 all_indices_for_range.append(df_daily_full.index)
                 df_daily_weekly_mean = df_daily_full.resample(target_freq).mean()
                 print(f"    日度->周度(均值) 完成. Shape: {df_daily_weekly_mean.shape}")
            else:
                print("    合并后的日度数据为空，无法进行周度转换。")
        else:
            print("  无日度数据。")

        # 4b: Weekly -> Weekly (Last value alignment)
        if data_parts['weekly']:
            print("  处理周度数据 -> 周度 (最后值)...")
            df_weekly_full = pd.concat(data_parts['weekly'], axis=1)
             # Handle duplicate columns from different weekly sheets before resampling
            cols_before_dedup_weekly = set(df_weekly_full.columns)
            df_weekly_full = df_weekly_full.loc[:, ~df_weekly_full.columns.duplicated(keep='first')]
            cols_after_dedup_weekly = set(df_weekly_full.columns)
            removed_cols_dedup_weekly = cols_before_dedup_weekly - cols_after_dedup_weekly
            if removed_cols_dedup_weekly:
                 print(f"    警告: 在合并周度数据时因重复移除了 {len(removed_cols_dedup_weekly)} 列: {list(removed_cols_dedup_weekly)[:5]}...")
                 for col in removed_cols_dedup_weekly:
                     if not any(d['Variable'] == col for d in removed_variables_detailed_log):
                         removed_variables_detailed_log.append({'Variable': col, 'Reason': 'weekly_duplicate_column'})

            if not df_weekly_full.empty:
                 # Add original weekly index to range calculation *before* resampling
                 all_indices_for_range.append(df_weekly_full.index)
                 df_weekly_aligned = df_weekly_full.resample(target_freq).last()
                 print(f"    周度->周度(对齐) 完成. Shape: {df_weekly_aligned.shape}")
            else:
                print("    合并后的周度数据为空，无法进行周度转换。")
        else:
            print("  无周度数据。")

        # 4c: Combine Daily(W) and Weekly(W) and perform NaN check
        print("\n  合并转换后的日度和周度数据...")
        parts_to_combine_dw = []
        if not df_daily_weekly_mean.empty: parts_to_combine_dw.append(df_daily_weekly_mean)
        if not df_weekly_aligned.empty: parts_to_combine_dw.append(df_weekly_aligned)

        if parts_to_combine_dw:
            df_combined_dw_weekly = pd.concat(parts_to_combine_dw, axis=1)
            # Handle duplicates arising from combining daily/weekly sources
            cols_before_dedup_dw = set(df_combined_dw_weekly.columns)
            df_combined_dw_weekly = df_combined_dw_weekly.loc[:, ~df_combined_dw_weekly.columns.duplicated(keep='first')]
            cols_after_dedup_dw = set(df_combined_dw_weekly.columns)
            removed_cols_dedup_dw = cols_before_dedup_dw - cols_after_dedup_dw
            if removed_cols_dedup_dw:
                 print(f"    警告: 在合并日度(周)和周度(周)数据时因重复移除了 {len(removed_cols_dedup_dw)} 列: {list(removed_cols_dedup_dw)[:5]}...")
                 for col in removed_cols_dedup_dw:
                     if not any(d['Variable'] == col for d in removed_variables_detailed_log):
                         removed_variables_detailed_log.append({'Variable': col, 'Reason': 'daily_weekly_combined_duplicate'})
            print(f"    合并后 Shape: {df_combined_dw_weekly.shape}")

            # Perform NaN check on the combined daily/weekly data
            if consecutive_nan_threshold is not None and consecutive_nan_threshold > 0:
                print(f"  [周度检查] 开始检查合并的日/周数据的连续缺失值 (阈值 >= {consecutive_nan_threshold})...")
                cols_to_remove_dw_nan = []
                for col in df_combined_dw_weekly.columns:
                    series = df_combined_dw_weekly[col]
                    first_valid_idx = series.first_valid_index()
                    if first_valid_idx is None: continue # Skip if column is all NaN
                    series_after_first_valid = series.loc[first_valid_idx:]
                    is_na = series_after_first_valid.isna()
                    na_blocks = is_na.ne(is_na.shift()).cumsum()[is_na]
                    max_consecutive_nan = 0
                    if not na_blocks.empty:
                        try:
                            block_counts = na_blocks.value_counts()
                            if not block_counts.empty: max_consecutive_nan = block_counts.max()
                        except Exception as e_nan_count_dw:
                             print(f"    [周度检查] 警告: 计算 '{col}' 的 NaN 块时出错: {e_nan_count_dw}. 跳过此列检查.")
                             continue

                    if max_consecutive_nan >= consecutive_nan_threshold:
                        cols_to_remove_dw_nan.append(col)
                        print(f"    [周度检查] 标记移除变量: '{col}' (最大连续 NaN: {max_consecutive_nan} >= {consecutive_nan_threshold})", end='')
                        if not any(d['Variable'] == col for d in removed_variables_detailed_log):
                             removed_variables_detailed_log.append({'Variable': col, 'Reason': 'dw_consecutive_nan'})
                             print(" - 已记录移除")
                        else: print(" - 已在其他步骤记录")

                if cols_to_remove_dw_nan:
                    print("\n    [周度检查] 正在移除 {len(cols_to_remove_dw_nan)} 个连续缺失值超标的日/周变量...")
                    df_combined_dw_weekly = df_combined_dw_weekly.drop(columns=cols_to_remove_dw_nan)
                    print(f"      移除后日/周数据 Shape: {df_combined_dw_weekly.shape}")
                else:
                    print(f"    [周度检查] 所有合并的日/周变量的连续缺失值均低于阈值。")
            else:
                print("  (跳过/禁用) 合并日/周数据的连续缺失值检查。")
        else:
            print("  没有有效的日度或周度数据可合并。")


        # --- Step 5: Combine All Aligned Weekly Data ---
        print("\n--- [Data Prep V3] 步骤 5: 合并所有对齐后的周度数据 --- ")
        # List of parts to combine: Target (nearest Fri), Target Sheet Predictors (nearest Fri), DW (cleaned), Other Monthly Preds (last Fri)
        all_final_weekly_parts = []

        # 🔥 修复：检查目标变量重复问题
        target_variable_added = False

        if target_series_aligned_nearest_friday is not None and not target_series_aligned_nearest_friday.empty:
            # Ensure the target series name is set correctly before appending
            target_series_aligned_nearest_friday.name = actual_target_variable_name
            all_final_weekly_parts.append(target_series_aligned_nearest_friday)
            target_variable_added = True
            print(f"  添加目标变量 '{actual_target_variable_name}' (最近周五对齐)...")

        # --- NEW: Add target sheet predictors ---
        if target_sheet_predictors_aligned_nearest_friday is not None and not target_sheet_predictors_aligned_nearest_friday.empty:
            # 🔥 修复：检查目标Sheet预测变量中是否包含目标变量
            target_sheet_cols = list(target_sheet_predictors_aligned_nearest_friday.columns)
            if actual_target_variable_name in target_sheet_cols:
                print(f"  ⚠️ 警告：目标Sheet预测变量中包含目标变量 '{actual_target_variable_name}'")
                if target_variable_added:
                    print(f"  🔧 移除目标Sheet中的重复目标变量")
                    target_sheet_predictors_aligned_nearest_friday = target_sheet_predictors_aligned_nearest_friday.drop(columns=[actual_target_variable_name])
                    target_sheet_cols = list(target_sheet_predictors_aligned_nearest_friday.columns)
                else:
                    print(f"  ✅ 目标变量将从目标Sheet中获取")
                    target_variable_added = True

            if not target_sheet_predictors_aligned_nearest_friday.empty:
                all_final_weekly_parts.append(target_sheet_predictors_aligned_nearest_friday)
                print(f"  添加目标 Sheet 预测变量 ({len(target_sheet_cols)} 个, 最近周五对齐)...")
        # --- END NEW ---

        if df_combined_dw_weekly is not None and not df_combined_dw_weekly.empty:
            # 🔥 修复：检查日度/周度数据中是否包含目标变量
            dw_cols = list(df_combined_dw_weekly.columns)
            if actual_target_variable_name in dw_cols:
                print(f"  ⚠️ 警告：日度/周度数据中包含目标变量 '{actual_target_variable_name}'")
                if target_variable_added:
                    print(f"  🔧 移除日度/周度数据中的重复目标变量")
                    df_combined_dw_weekly = df_combined_dw_weekly.drop(columns=[actual_target_variable_name])
                else:
                    print(f"  ✅ 目标变量将从日度/周度数据中获取")
                    target_variable_added = True

            if not df_combined_dw_weekly.empty:
                all_final_weekly_parts.append(df_combined_dw_weekly)
                print(f"  添加日度/周度预测变量 (Shape: {df_combined_dw_weekly.shape})...")

        if monthly_predictors_aligned_last_friday is not None and not monthly_predictors_aligned_last_friday.empty:
            # 🔥 修复：检查月度预测变量中是否包含目标变量
            monthly_cols = list(monthly_predictors_aligned_last_friday.columns)
            if actual_target_variable_name in monthly_cols:
                print(f"  ⚠️ 警告：月度预测变量中包含目标变量 '{actual_target_variable_name}'")
                if target_variable_added:
                    print(f"  🔧 移除月度预测变量中的重复目标变量")
                    monthly_predictors_aligned_last_friday = monthly_predictors_aligned_last_friday.drop(columns=[actual_target_variable_name])
                else:
                    print(f"  ✅ 目标变量将从月度预测变量中获取")
                    target_variable_added = True

            if not monthly_predictors_aligned_last_friday.empty:
                all_final_weekly_parts.append(monthly_predictors_aligned_last_friday)
                print(f"  添加其他月度预测变量 (月末最后周五对齐, Shape: {monthly_predictors_aligned_last_friday.shape})...")

        print(f"  🔍 目标变量添加状态: {'已添加' if target_variable_added else '未添加'}")

        if not all_final_weekly_parts:
            print("错误：[Data Prep] 没有成功处理的数据部分可以合并。无法继续。")
            return None, None, None, None

        # Determine full date range using collected indices
        if not all_indices_for_range or all(idx.empty for idx in all_indices_for_range):
            print("错误: 无法确定日期范围，因为没有加载任何有效数据索引。")
            return None, None, None, None

        # Filter out empty indices before finding min/max
        valid_indices = [idx for idx in all_indices_for_range if idx is not None and not idx.empty]
        if not valid_indices:
             print("错误: 所有收集到的索引都为空，无法确定日期范围。")
             return None, None, None, None

        min_date_orig = min(idx.min() for idx in valid_indices)
        max_date_orig = max(idx.max() for idx in valid_indices)
        print(f"  所有原始数据中的最小/最大日期: {min_date_orig.date()} / {max_date_orig.date()}")

        # Determine the final start/end dates considering config and data
        final_start_date = pd.to_datetime(data_start_date) if data_start_date else min_date_orig
        final_end_date = pd.to_datetime(data_end_date) if data_end_date else max_date_orig
        # Adjust if config dates are outside data range
        if data_start_date and pd.to_datetime(data_start_date) < min_date_orig:
            final_start_date = pd.to_datetime(data_start_date)
        if data_end_date and pd.to_datetime(data_end_date) > max_date_orig:
            final_end_date = pd.to_datetime(data_end_date)

        # Align final start/end dates to Friday frequency
        # Ensure start date is not pushed beyond the first available data point's Friday
        # Ensure end date is not pulled before the last available data point's Friday
        min_date_fri = min_date_orig - pd.Timedelta(days=(min_date_orig.weekday() - 4 + 7) % 7)
        max_date_fri = max_date_orig - pd.Timedelta(days=(max_date_orig.weekday() - 4 + 7) % 7) + pd.Timedelta(weeks=0 if max_date_orig.weekday()==4 else 1) # Go to next Friday if not Friday

        final_start_date_aligned = final_start_date - pd.Timedelta(days=(final_start_date.weekday() - 4 + 7) % 7)
        final_end_date_aligned = final_end_date - pd.Timedelta(days=(final_end_date.weekday() - 4 + 7) % 7)

        # Respect the actual data boundaries when creating the range
        actual_range_start = max(min_date_fri, final_start_date_aligned)
        actual_range_end = min(max_date_fri, final_end_date_aligned)

        if actual_range_start > actual_range_end:
            print(f"错误: 计算出的实际开始日期 ({actual_range_start.date()}) 晚于结束日期 ({actual_range_end.date()})。请检查数据范围和配置。")
            return None, None, None, None

        full_date_range = pd.date_range(start=actual_range_start, end=actual_range_end, freq=target_freq)
        print(f"  最终确定的完整周度日期范围 (对齐到 {target_freq}): {full_date_range.min().date()} 到 {full_date_range.max().date()}")

        # Combine and reindex
        # Use outer join initially to capture all data points on their respective Fridays
        combined_data_weekly_final = pd.concat(all_final_weekly_parts, axis=1, join='outer')
        print(f"  合并所有 {len(all_final_weekly_parts)} 个最终周度数据部分 (outer join). 初始合并 Shape: {combined_data_weekly_final.shape}")

        # 🔥 优化：高效的重复列检查和处理
        print(f"  🔍 合并前检查重复列...")

        # 使用更高效的重复检测方法
        columns = combined_data_weekly_final.columns
        duplicate_mask = columns.duplicated(keep=False)

        if duplicate_mask.any():
            # 只在有重复时才进行详细分析
            from collections import Counter
            column_counts = Counter(columns)
            duplicated_names = {name: count for name, count in column_counts.items() if count > 1}

            print(f"  🚨 发现重复列！总数: {duplicate_mask.sum()}")
            print(f"    重复的列名数量: {len(duplicated_names)}")

            # 只显示前5个重复列名以减少输出
            for i, (name, count) in enumerate(duplicated_names.items()):
                if i < 5:
                    print(f"      '{name}': {count} 次")
                elif i == 5:
                    print(f"      ... 还有 {len(duplicated_names) - 5} 个重复列名")

            # 特别检查目标变量重复
            if actual_target_variable_name in duplicated_names:
                target_count = duplicated_names[actual_target_variable_name]
                print(f"  ⚠️ 目标变量 '{actual_target_variable_name}' 重复了 {target_count-1} 次！")

        # 🔥 优化：高效处理重复列，减少内存使用
        original_col_count = len(combined_data_weekly_final.columns)
        duplicate_mask = combined_data_weekly_final.columns.duplicated(keep='first')

        if duplicate_mask.any():
            # 只在有重复时才进行处理
            removed_count = duplicate_mask.sum()
            print(f"  🔧 在最终合并后因重复移除了 {removed_count} 列")

            # 获取被移除的列名（仅用于日志记录）
            removed_cols = combined_data_weekly_final.columns[duplicate_mask].tolist()

            # 高效去除重复列
            combined_data_weekly_final = combined_data_weekly_final.iloc[:, ~duplicate_mask]

            # 批量记录移除的列，避免逐个检查
            for col in removed_cols:
                removed_variables_detailed_log.append({
                    'Variable': col,
                    'Reason': 'duplicate_column_final'
                })

            print(f"    最终列数: {len(combined_data_weekly_final.columns)} (减少了 {removed_count} 列)")
        else:
            print(f"  ✅ 未发现重复列")

        print(f"  🔧 去重后形状: {combined_data_weekly_final.shape}")

        # Reindex to the full calculated date range, this aligns everything to the target frequency grid
        all_data_aligned_weekly = combined_data_weekly_final.reindex(full_date_range)
        print(f"  重新索引到完整周度日期范围完成. Shape: {all_data_aligned_weekly.shape}")

        # --- Step 6: Final Weekly Data Processing ---
        print("\n--- [Data Prep V3] 步骤 6: 最终周度数据处理 --- ")

        # 6a: Apply Time Range Filter (already implicitly done by full_date_range calculation)
        print(f"  时间范围过滤已在确定日期范围时应用 ({full_date_range.min().date()} to {full_date_range.max().date()})。")

        # 6b: Drop any columns that became all NaN after all processing and reindexing
        print("  移除在最终处理和对齐后完全为 NaN 的列...")
        cols_before_final_dropna = set(all_data_aligned_weekly.columns)
        all_data_aligned_weekly = all_data_aligned_weekly.dropna(axis=1, how='all')
        removed_in_final_dropna = cols_before_final_dropna - set(all_data_aligned_weekly.columns)
        if removed_in_final_dropna:
             print(f"  [!] 移除了 {len(removed_in_final_dropna)} 个全 NaN 列: {list(removed_in_final_dropna)[:10]}{'...' if len(removed_in_final_dropna)>10 else ''}")
             # Log removals
             for col in removed_in_final_dropna:
                 if not any(d['Variable'] == col for d in removed_variables_detailed_log):
                     removed_variables_detailed_log.append({'Variable': col, 'Reason': 'all_nan_final'})
        print(f"    移除全 NaN 列后 Shape: {all_data_aligned_weekly.shape}")

        if all_data_aligned_weekly is None or all_data_aligned_weekly.empty:
            print("错误: [Data Prep] 最终合并和对齐后的数据在移除全 NaN 列后为空。")
            return None, None, None, None

        # 6c: Weekly Stationarity Check (Skipping MONTHLY originated predictors AND Target Variable)
        print("  执行最终周度数据平稳性检查 (跳过月度来源预测变量和目标变量)...")
        weekly_transform_log = {}
        final_data_stationary = all_data_aligned_weekly.copy() # Start with current data

        # --- Define columns to skip based on origin and target status ---
        # Start with the set of columns identified as monthly predictors after their processing
        # cols_to_skip_weekly_stationarity = monthly_predictors_to_skip_weekly_stationarity.copy() # OLD
        cols_to_skip_weekly_stationarity = other_monthly_predictors_to_skip_weekly_stationarity.copy() # Use vars from OTHER monthlies (last Fri aligned)
        print(f"    初始跳过列表基于其他月度来源变量: {len(cols_to_skip_weekly_stationarity)} 个")
        # Add the actual target variable name
        if actual_target_variable_name in final_data_stationary.columns:
             cols_to_skip_weekly_stationarity.add(actual_target_variable_name)
             print(f"    标记跳过目标变量: '{actual_target_variable_name}'")
        else:
             print(f"    警告：目标变量 '{actual_target_variable_name}' 不在最终数据中，无法在跳过列表中标记。")
        # --- NEW: Add target sheet predictors (Cols C+) to skip list ---
        target_sheet_predictor_cols_in_final = target_sheet_predictor_cols.intersection(final_data_stationary.columns)
        if target_sheet_predictor_cols_in_final:
             cols_to_skip_weekly_stationarity.update(target_sheet_predictor_cols_in_final)
             print(f"    标记跳过目标 Sheet 预测变量: {len(target_sheet_predictor_cols_in_final)} 个")
        # --- END NEW ---
        # print(f"    标记跳过 {len(monthly_predictors_to_skip_weekly_stationarity)} 个月度来源预测变量。") # OLD
        print(f"    总共将标记跳过 {len(cols_to_skip_weekly_stationarity)} 个变量进行周度平稳性检查。")

        # --- Normalize the skip list for reliable matching ---
        skip_cols_normalized = {unicodedata.normalize('NFKC', str(c)).strip().lower() for c in cols_to_skip_weekly_stationarity}
        print(f"    规范化后的跳过列表大小: {len(skip_cols_normalized)}")


        # --- Check for and apply pre-defined stationarity rules from config if available ---
        use_config_stationarity = False
        config_stationarity_rules = {}
        try:
            # Dynamically import config to get the latest version if it changes
            import importlib
            from dym_estimate import config
            importlib.reload(config) # Reload in case it was modified

            if hasattr(config, 'PREDEFINED_STATIONARITY_TRANSFORMS') and isinstance(config.PREDEFINED_STATIONARITY_TRANSFORMS, dict):
                 # Normalize keys in the loaded rules
                 config_stationarity_rules_raw = {
                      unicodedata.normalize('NFKC', str(k)).strip().lower(): v
                      for k, v in config.PREDEFINED_STATIONARITY_TRANSFORMS.items()
                      if isinstance(v, dict) and 'status' in v # Ensure value is dict with 'status'
                 }
                 if config_stationarity_rules_raw:
                      print(f"  检测到来自 config.py 的预定义平稳性转换规则 ({len(config_stationarity_rules_raw)} 条)。")

                      # --- CRITICAL FIX: Filter rules BEFORE applying ---
                      # Remove rules for columns that should be skipped (target + monthly)
                      config_stationarity_rules = {
                           k: v for k, v in config_stationarity_rules_raw.items()
                           if k not in skip_cols_normalized # Use normalized skip list
                      }
                      removed_rules_count = len(config_stationarity_rules_raw) - len(config_stationarity_rules)
                      if removed_rules_count > 0:
                          print(f"    已从预定义规则中移除 {removed_rules_count} 条，因为它们对应于需要跳过的目标变量或月度变量。")
                      # --- END CRITICAL FIX ---

                      if config_stationarity_rules: # Check if any rules remain after filtering
                           use_config_stationarity = True
                      else:
                           print("    过滤后，没有适用于日/周变量的预定义规则。将回退到 ADF 检验。")

                 else:
                      print("  config.py 中 PREDEFINED_STATIONARITY_TRANSFORMS 为空或格式无效，将执行 ADF 检验。")
            else:
                 print("  config.py 中未定义 PREDEFINED_STATIONARITY_TRANSFORMS，将执行 ADF 检验。")
        except ImportError:
            print("  无法导入 config.py 或其不存在，将执行 ADF 检验。")
        except Exception as e_cfg_stat:
             print(f"  加载或处理 config.py 中的平稳性规则时出错: {e_cfg_stat}。将执行 ADF 检验。")


        # --- Apply Stationarity Transformation ---
        removed_cols_info_weekly = {} # Initialize removal log specific to this step
        if use_config_stationarity:
             print(f"  应用过滤后的预定义平稳性规则 ({len(config_stationarity_rules)} 条规则)...")
             # Normalize column names of the data to match rule keys
             # Keep mapping to restore original names if needed later? Or assume rules use normalized?
             # Let's assume apply_stationarity_transforms handles matching normalized rule keys to potentially non-normalized df columns if necessary
             # Or better: normalize data columns before applying
             original_columns_map_cfg = {unicodedata.normalize('NFKC', str(c)).strip().lower(): c for c in final_data_stationary.columns}
             final_data_stationary.columns = list(original_columns_map_cfg.keys())

             final_data_stationary = apply_stationarity_transforms(
                 final_data_stationary,
                 config_stationarity_rules # Pass the FILTERED rules
             )
             # Restore original column names
             final_data_stationary.columns = final_data_stationary.columns.map(original_columns_map_cfg)

             # We don't have a detailed weekly_transform_log or removed_cols_info_weekly in this case from apply_stationarity_transforms
             weekly_transform_log = {"status": "Applied filtered rules from config"}
             # We rely on apply_stationarity_transforms to keep skipped cols untouched.
        else:
             print("  通过 ADF 检验确定平稳性 (仅日/周来源变量)...")
             # Use the normalized skip list directly with _ensure_stationarity
             # The function _ensure_stationarity now handles normalization internally for matching

             # Normalize column names before passing to ensure consistent matching inside the function
             original_columns_map_adf = {unicodedata.normalize('NFKC', str(c)).strip().lower(): c for c in final_data_stationary.columns}
             final_data_stationary.columns = list(original_columns_map_adf.keys())

             final_data_stationary, weekly_transform_log, removed_cols_info_weekly = _ensure_stationarity(
                 final_data_stationary,
                 skip_cols=skip_cols_normalized, # Pass the normalized skip set
                 adf_p_threshold=0.05
             )
             # Restore original column names
             final_data_stationary.columns = final_data_stationary.columns.map(original_columns_map_adf)

             # Log weekly stationarity removals (using original names)
             for reason, cols_norm in removed_cols_info_weekly.items():
                 for col_norm in cols_norm:
                     original_col_name = original_columns_map_adf.get(col_norm, col_norm) # Get original name back
                     if not any(d['Variable'] == original_col_name for d in removed_variables_detailed_log):
                         removed_variables_detailed_log.append({'Variable': original_col_name, 'Reason': f'weekly_stationarity_{reason}'})


        # --- Step 7: Final Checks and Log Combination ---
        print("\n--- [Data Prep V3] 步骤 7: 完成与检查 --- ")
        if final_data_stationary is None or final_data_stationary.empty:
            print("错误: [Data Prep] 最终数据在平稳性处理后为空。")
            return None, None, None, None

        print(f"  最终数据 Shape: {final_data_stationary.shape}")
        # Check for target variable existence using the ORIGINAL name
        target_exists = actual_target_variable_name in final_data_stationary.columns
        print(f"  目标变量 '{actual_target_variable_name}' 是否存在: {target_exists}")
        if not target_exists:
             # Also check normalized name in case it was normalized and not restored correctly
             norm_target_name_final_check = unicodedata.normalize('NFKC', actual_target_variable_name).strip().lower()
             temp_cols_lower = {unicodedata.normalize('NFKC', str(c)).strip().lower():c for c in final_data_stationary.columns}
             if norm_target_name_final_check in temp_cols_lower:
                 print(f"  注意：目标变量以规范化名称 '{norm_target_name_final_check}' 存在。")
                 # Attempt to rename back to original if found normalized
                 # final_data_stationary = final_data_stationary.rename(columns={temp_cols_lower[norm_target_name_final_check]: actual_target_variable_name})
             else:
                 print(f"  严重警告: 目标变量 '{actual_target_variable_name}' 在最终数据中不存在！")

        # Calculate predictor count after checking target existence
        final_predictor_count_output = final_data_stationary.shape[1] - (1 if target_exists else 0)

        # Combine Logs
        combined_transform_log = {
            "monthly_predictor_stationarity_checks": monthly_transform_log, # Log from monthly check
            "weekly_final_stationarity_checks": weekly_transform_log # Log from final weekly check
        }

        # --- Finalize Industry Map ---
        # Ensure map keys are normalized and only include columns present in the final data
        final_columns_in_data = set(final_data_stationary.columns)
        updated_var_industry_map = {}
        original_names_to_norm_final = {} # Map final original names to normalized

        for col_original in final_columns_in_data:
            col_norm = unicodedata.normalize('NFKC', str(col_original)).strip().lower()
            if col_norm:
                original_names_to_norm_final[col_original] = col_norm
                # Get industry from original map using normalized key, default to "Unknown"
                # The original var_industry_map was populated during loading
                industry = var_industry_map.get(col_norm, "Unknown")
                updated_var_industry_map[col_norm] = industry # Store with normalized key

        # --- Variable Count Comparison (Using normalized names for comparison) ---
        raw_predictor_count = len(raw_columns_across_all_sheets) # All predictors initially loaded (normalized)
        reference_count = len(reference_predictor_variables) # From reference sheet (normalized)
        print(f"\n--- [Data Prep] 变量数量与指标体系对比 ---")
        print(f"  指标体系变量数 (规范化): {reference_count}")
        print(f"  原始加载预测变量数 (规范化, 不含目标): {raw_predictor_count}")
        print(f"  最终输出预测变量数: {final_predictor_count_output}")

        if reference_predictor_variables:
            # Get normalized names of final predictors
            final_output_predictors_norm = {
                 norm_name for orig_name, norm_name in original_names_to_norm_final.items()
                 if orig_name != actual_target_variable_name # Exclude target variable using original name check
            }

            raw_loaded_predictors_norm = raw_columns_across_all_sheets # Already normalized, excludes target

            # --- Compare Reference vs Raw Loaded ---
            missing_in_data = reference_predictor_variables - raw_loaded_predictors_norm
            extra_in_data = raw_loaded_predictors_norm - reference_predictor_variables

            if missing_in_data:
                 # Filter out target var if it's in the reference list by mistake
                 norm_target_name_ref_check = unicodedata.normalize('NFKC', actual_target_variable_name).strip().lower()
                 missing_to_print_norm = [v for v in sorted(list(missing_in_data)) if v != norm_target_name_ref_check]
                 if missing_to_print_norm:
                      print(f"\n  [!] 以下 {len(missing_to_print_norm)} 个变量在指标体系中，但未在任何数据 Sheets 中加载:")
                      for i, var_norm in enumerate(missing_to_print_norm):
                           print(f"      {i+1}. {var_norm} (规范名)")
                 if len(missing_to_print_norm) != len(missing_in_data):
                      print("      (注: 目标变量 '{norm_target_name_ref_check}' 被从该列表忽略)")


            if extra_in_data:
                 print(f"\n  [!] 警告: 以下 {len(extra_in_data)} 个变量从数据 Sheets 加载，但不在指标体系中:")
                 for i, var_norm in enumerate(sorted(list(extra_in_data))):
                      # Try to find original name from the final data for display
                      original_name_guess = next((orig for orig, norm in original_names_to_norm_final.items() if norm == var_norm), var_norm)
                      print(f"      {i+1}. {original_name_guess} (规范名: {var_norm})")


            # --- Compare Reference vs Final Output ---
            missing_in_final_output = reference_predictor_variables - final_output_predictors_norm
            if missing_in_final_output:
                 missing_final_but_loaded = missing_in_final_output & raw_loaded_predictors_norm
                 missing_final_and_never_loaded = missing_in_final_output - raw_loaded_predictors_norm

                 if missing_final_but_loaded:
                      print(f"\n  [i] 以下 {len(missing_final_but_loaded)} 个指标体系中的变量在加载后、处理过程中被移除:")
                      count = 0
                      for var_norm in sorted(list(missing_final_but_loaded)):
                           count += 1
                           # Find original name if possible, fallback to norm name
                           original_name_guess = next((orig for orig, norm in original_names_to_norm_final.items() if norm == var_norm), var_norm)
                           # Find removal reason from the detailed log (match normalized name)
                           reason = "Unknown (移除原因未在日志中明确记录)"
                           for item in removed_variables_detailed_log:
                                logged_var_norm = unicodedata.normalize('NFKC', str(item.get('Variable',''))).strip().lower()
                                if logged_var_norm == var_norm:
                                    reason = item.get('Reason', '记录中原因缺失')
                                    break
                           print(f"      {count}. {original_name_guess} (规范名: {var_norm}) - 原因: {reason}")

                 # No need to print the never loaded ones again if already printed above
                 # if missing_final_and_never_loaded:
                 #     print(f"    (另有 {len(missing_final_and_never_loaded)} 个指标体系变量从未被加载)")

            else: # All reference vars (predictors) are in the final output
                 # Check if all RAW loaded predictors made it to the final output
                 removed_loaded_predictors = raw_loaded_predictors_norm - final_output_predictors_norm
                 if removed_loaded_predictors:
                      print(f"\n  [i] 所有指标体系中的变量均在最终输出中。但有 {len(removed_loaded_predictors)} 个加载的变量在处理中被移除:")
                      # (Optional: List removed variables here using similar logic as above)
                 else:
                      print("\\n  [i] 所有指标体系中的变量都存在于最终输出中，且所有加载的变量都未被移除。")


        else: # reference_predictor_variables is empty
             print("\\n  未能加载指标体系进行对比。")

        # --- Transformation Log Summary ---
        print(f"\n--- [Data Prep] 转换日志摘要 --- ")
        # Summarize monthly check results
        log_monthly = combined_transform_log.get("monthly_predictor_stationarity_checks", {})
        if isinstance(log_monthly, dict) and log_monthly:
            from collections import Counter
            monthly_statuses = Counter(log.get('status', 'unknown') for log in log_monthly.values())
            print(f"  月度预测变量检查状态 (ADF): {dict(monthly_statuses)}")
        else:
            print("  月度预测变量检查日志不可用或为空。")

        # Summarize weekly check results (handle both dict and string case)
        log_weekly = combined_transform_log.get("weekly_final_stationarity_checks", {})
        if isinstance(log_weekly, dict) and log_weekly.get("status") == "Applied filtered rules from config":
             print(f"  周度最终检查状态: 应用了来自 config 的过滤后预定义规则。")
        elif isinstance(log_weekly, dict) and log_weekly:
             weekly_statuses = Counter(log.get('status', 'unknown') for log in log_weekly.values())
             # Filter out 'skipped_by_request' from summary count
             filtered_weekly_statuses = {k:v for k,v in weekly_statuses.items() if k != 'skipped_by_request'}
             skipped_count = weekly_statuses.get('skipped_by_request', 0)
             print(f"  周度最终检查状态 (ADF, 仅日/周源): {dict(filtered_weekly_statuses)}")
             if skipped_count > 0: print(f"    (另有 {skipped_count} 个变量按计划被跳过)")
        elif isinstance(log_weekly, dict) and not log_weekly: # Empty dict means ADF run but nothing to check
             print(f"  周度最终检查状态 (ADF): 无需检查的日/周变量。")
        else: # Should be a dict, but handle unexpected cases
             print(f"  周度最终检查日志格式未知或不可用: {type(log_weekly)}")

        # --- Populate the detailed removal log (already done during process) ---
        print("\n--- [Data Prep] 正在生成移除变量日志 ---")
        print(f"  共记录了 {len(removed_variables_detailed_log)} 个移除事件。")

        print(f"\n--- [Data Prep V3] 数据准备完成 --- ")
        # --- <<< 新增：打印最终目标变量最后几行 >>> ---
        if actual_target_variable_name in final_data_stationary.columns:
            print(f"    [Debug Target Final] 最终数据中目标变量 '{actual_target_variable_name}' 最后 5 行:")
            print(final_data_stationary[actual_target_variable_name].tail())
        else:
            print(f"    [Debug Target Final] 警告：最终数据中未找到目标变量 '{actual_target_variable_name}'")
        # --- <<< 结束新增 >>> ---
        # Return data with original column names, and the updated industry map with normalized keys
        return final_data_stationary, updated_var_industry_map, combined_transform_log, removed_variables_detailed_log

    except FileNotFoundError:
        print(f"错误: [Data Prep] Excel 数据文件 {excel_path} 未找到。")
        return None, None, None, None
    except Exception as err:
        print(f"错误: [Data Prep] 数据准备过程中发生意外错误: {err}")
        import traceback
        traceback.print_exc()
        return None, None, None, None

if __name__ == '__main__':
    print(f"Testing data_preparation module (V3 Logic, Reduced Set Mode: {CREATE_REDUCED_TEST_SET})...")

    # --- MODIFICATION: Import config values --- 
    config_loaded = False
    
    # 先尝试直接从当前目录导入
    try:
        import config
        config_loaded = True
        config_source = "config.py"
    except ImportError:
        # 如果失败，尝试从包导入
        try:
            from dym_estimate import config
            config_loaded = True
            config_source = "dym_estimate.config"
        except ImportError:
            # 两种方式都失败，使用默认值
            config_loaded = False
    
    # 根据配置加载结果设置变量
    if config_loaded:
        DATA_START_DATE_TEST = config.DATA_START_DATE
        DATA_END_DATE_TEST = config.DATA_END_DATE
        TARGET_VAR_TEST = config.TARGET_VARIABLE 
        TARGET_SHEET_TEST = config.TARGET_SHEET_NAME
        TARGET_FREQ_TEST = config.TARGET_FREQ
        REMOVE_CONSECUTIVE_NAN_VARS_TEST = config.REMOVE_VARS_WITH_CONSECUTIVE_NANS
        EXCEL_DATA_FILE_TEST = config.EXCEL_DATA_FILE
        
        print(f"  成功从 {config_source} 加载设置:")
        print(f"    DATA_START_DATE = {DATA_START_DATE_TEST}")
        print(f"    DATA_END_DATE = {DATA_END_DATE_TEST}")
        print(f"    TARGET_VARIABLE = {TARGET_VAR_TEST}")
        print(f"    TARGET_SHEET = {TARGET_SHEET_TEST}")
        print(f"    EXCEL_DATA_FILE = {EXCEL_DATA_FILE_TEST}")
    else:
        print("  警告: 无法导入 dym_estimate.config。将使用硬编码的测试值。")
        DATA_START_DATE_TEST = '2020-01-01' # Fallback if config fails
        DATA_END_DATE_TEST = None
        TARGET_VAR_TEST = '规模以上工业增加值:当月同比'
        TARGET_SHEET_TEST = '工业增加值同比增速_月度_同花顺'
        TARGET_FREQ_TEST = 'W-FRI'  # 添加默认频率设置
        REMOVE_CONSECUTIVE_NAN_VARS_TEST = True  # 默认启用移除连续缺失变量
        # --- 添加：config 加载失败时的回退文件路径 ---
        EXCEL_DATA_FILE_TEST = os.path.join('data', '经济数据库0508.xlsx')
        # --- 结束添加 ---
    # --- END MODIFICATION ---

    # --- 删除：移除旧的硬编码文件路径设置 ---
    # Use hardcoded test values for file paths etc.
    # EXCEL_DATA_FILE_TEST = os.path.join('data', '经济数据库0424_带数据源标志.xlsx')
    # --- 结束删除 ---
    TARGET_FREQ_TEST = 'W-FRI'
    # TARGET_SHEET_TEST = '工业增加值同比增速_月度_同花顺' # Now loaded from config
    # TARGET_VAR_TEST_FROM_CONFIG = '规模以上工业增加值:当月同比' # Now loaded from config
    CONSECUTIVE_NAN_THRESHOLD_TEST = 10 # Set desired threshold
    REFERENCE_SHEET_TEST = '指标体系'
    REFERENCE_COL_TEST = '高频指标'

    # Check test file exists
    # --- 修改：移除备用路径检查 --- 
    if not os.path.exists(EXCEL_DATA_FILE_TEST):
         # alt_path = os.path.join('..', 'data', '经济数据库0424.xlsx')
         # if os.path.exists(alt_path):
         #     EXCEL_DATA_FILE_TEST = alt_path
         # else:
         print(f"错误: 测试文件未找到于 '{EXCEL_DATA_FILE_TEST}'。请检查 config.py 中的 EXCEL_DATA_FILE 设置或文件是否存在。")
         sys.exit(1)
    # --- 结束修改 ---

    print(f"  测试文件: {EXCEL_DATA_FILE_TEST}")
    print(f"  目标频率: {TARGET_FREQ_TEST}")
    print(f"  目标Sheet: {TARGET_SHEET_TEST}") # Use value from config/fallback
    print(f"  目标变量(预期B列): {TARGET_VAR_TEST}") # Use value from config/fallback
    print(f"  连续 NaN 阈值: {CONSECUTIVE_NAN_THRESHOLD_TEST}")
    print(f"  数据开始日期 (来自配置): {DATA_START_DATE_TEST}") # Print loaded date
    print(f"  数据结束日期 (来自配置): {DATA_END_DATE_TEST}") # Print loaded date

    # Call the revised prepare_data function
    prepared_data, industry_map, transform_log, removed_variables_detailed_log = prepare_data(
                                 excel_path=EXCEL_DATA_FILE_TEST,
                                 target_freq=TARGET_FREQ_TEST,
                                 target_sheet_name=TARGET_SHEET_TEST, # Use loaded value
                                 target_variable_name=TARGET_VAR_TEST, # Use loaded value
                                 consecutive_nan_threshold=CONSECUTIVE_NAN_THRESHOLD_TEST,
                                 data_start_date=DATA_START_DATE_TEST, # Use loaded value
                                 data_end_date=DATA_END_DATE_TEST,   # Use loaded value
                                 reference_sheet_name=REFERENCE_SHEET_TEST,
                                 reference_column_name=REFERENCE_COL_TEST
                                 )

    if prepared_data is not None and removed_variables_detailed_log is not None:
        # 移除文件输出功能，改为仅返回处理结果
        print("数据准备完成，结果将通过返回值传递，不再保存到文件")

        print(f"\n--- Prepared Weekly Data Info (V3 Logic, {'Reduced' if CREATE_REDUCED_TEST_SET else 'Full'}) ---")
        buffer = io.StringIO()
        prepared_data.info(buf=buffer)
        print(buffer.getvalue())

        print(f"\n--- Prepared Weekly Data Head (5 rows, 10 cols) ---")
        print(prepared_data.iloc[:5, :min(10, prepared_data.shape[1])])
        print(f"\n--- Prepared Weekly Data Tail (5 rows, 10 cols) ---")
        print(prepared_data.iloc[-5:, :min(10, prepared_data.shape[1])])

        print("数据准备完成，所有结果通过返回值传递，不保存到文件")
    else:
        print(f"\n(V3 Logic) 周度数据准备在测试期间失败。")

