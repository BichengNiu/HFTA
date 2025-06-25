import pandas as pd
import io
import re
from collections import defaultdict
# 我们需要从同一个 utils 目录下的 parsers.py 导入
from .parsers import parse_time_column 

# Extracted from time_series_clean_utils.py
def load_and_preprocess_data(session_state, file_content_bytes, file_name_with_extension, 
                             rows_to_skip_list, header_row_int_or_none, sheet_name_to_load=0):
    """
    Loads data, applies skip rows, sets header, removes duplicate columns, and performs initial preprocessing.
    """
    file_content_io = io.BytesIO(file_content_bytes)
    file_extension = file_name_with_extension.split('.')[-1].lower()
    df_raw = None
    final_df = None

    # Define a comprehensive list of NA values
    common_na_values = ['', '#N/A', '#N/A N/A', '#NA', '-1.#IND', '-1.#QNAN', '-NaN', '-nan', 
                        '1.#IND', '1.#QNAN', '<NA>', 'N/A', 'NA', 'NULL', 'NaN', 'None', 
                        'n/a', 'nan', 'null', 'none']

    try:
        # Step 1: Read entire sheet/csv with header=None to get all data
        if file_extension == 'xlsx':
            df_raw = pd.read_excel(file_content_io, sheet_name=sheet_name_to_load, header=None, engine='openpyxl', na_values=common_na_values)
        elif file_extension == 'csv':
            try:
                df_raw = pd.read_csv(file_content_io, header=None, encoding='utf-8', low_memory=False, na_values=common_na_values)
            except UnicodeDecodeError:
                print("UTF-8 decoding failed, trying cp936 for CSV.")
                file_content_io.seek(0)
                df_raw = pd.read_csv(file_content_io, header=None, encoding='cp936', low_memory=False, na_values=common_na_values)
        else:
            raise ValueError(f"不支持的文件类型: {file_extension}")

        if df_raw is None or df_raw.empty:
            session_state.ts_tool_error_message = "加载的原始文件为空或读取失败。"
            print("[DataProcessing] Error: Loaded raw file is empty or read failed.")
            return None
        print(f"[DEBUG DataProcessing Stage 0] df_raw loaded. Columns: {df_raw.columns.tolist()}, Shape: {df_raw.shape}") # DEBUG ADDED

        df_intermediate = df_raw.copy() 
        
        all_rows_to_remove_from_data = set(rows_to_skip_list) 
        new_columns = None

        if header_row_int_or_none is not None:
            if header_row_int_or_none in df_intermediate.index: 
                new_columns = df_intermediate.iloc[header_row_int_or_none].fillna('').astype(str).tolist()
                all_rows_to_remove_from_data.add(header_row_int_or_none)
                print(f"[DEBUG DataProcessing Stage 1] Extracted potential new_columns from original row {header_row_int_or_none}: {new_columns}") # DEBUG ADDED
            else:
                print(f"[DataProcessing] Warning: Specified header row {header_row_int_or_none} (0-indexed) not found in raw data with {len(df_intermediate)} rows. Proceeding with default integer column names.")
        else:
            print("[DEBUG DataProcessing Stage 1] header_row_int_or_none is None. No new_columns extracted by iloc.") # DEBUG ADDED
        
        rows_to_drop_indices = sorted([idx for idx in all_rows_to_remove_from_data if idx in df_intermediate.index])
        if rows_to_drop_indices:
            df_intermediate = df_intermediate.drop(index=rows_to_drop_indices).reset_index(drop=True)
            print(f"[DEBUG DataProcessing Stage 2] After dropping rows (original indices {rows_to_drop_indices}). df_intermediate columns: {df_intermediate.columns.tolist()}, Shape: {df_intermediate.shape}") # DEBUG ADDED
        else:
            print("[DEBUG DataProcessing Stage 2] No rows were dropped. df_intermediate columns: {df_intermediate.columns.tolist()}, Shape: {df_intermediate.shape}") # DEBUG ADDED


        if new_columns:
            if not df_intermediate.empty:
                print(f"[DEBUG DataProcessing Stage 3a] Attempting to assign new_columns. Current df_intermediate columns: {df_intermediate.columns.tolist()}, new_columns: {new_columns}") # DEBUG ADDED
                if len(new_columns) == len(df_intermediate.columns):
                    df_intermediate.columns = new_columns
                    print(f"[DEBUG DataProcessing Stage 3b] Assigned new_columns (exact match). df_intermediate columns now: {df_intermediate.columns.tolist()}") # DEBUG ADDED
                elif len(new_columns) > len(df_intermediate.columns):
                    df_intermediate.columns = new_columns[:len(df_intermediate.columns)]
                    print(f"[DEBUG DataProcessing Stage 3b] Assigned new_columns (trimmed header). df_intermediate columns now: {df_intermediate.columns.tolist()}") # DEBUG ADDED
                    print(f"[DataProcessing] Warning: Header has more columns ({len(new_columns)}) than data ({len(df_intermediate.columns)}). Trimmed header.")
                else: 
                    current_cols_list = list(df_intermediate.columns) 
                    for i in range(len(new_columns)):
                        current_cols_list[i] = new_columns[i]
                    df_intermediate.columns = current_cols_list[:len(df_intermediate.columns)] 
                    print(f"[DEBUG DataProcessing Stage 3b] Assigned new_columns (partial header). df_intermediate columns now: {df_intermediate.columns.tolist()}") # DEBUG ADDED
                    print(f"[DataProcessing] Warning: Header has fewer columns ({len(new_columns)}) than data ({len(df_intermediate.columns)}). Partially applied header.")
            elif new_columns: 
                 df_intermediate = pd.DataFrame(columns=new_columns)
                 print("[DEBUG DataProcessing Stage 3b] Data became empty, created new DataFrame with new_columns. df_intermediate columns now: {df_intermediate.columns.tolist()}") # DEBUG ADDED
                 print("[DataProcessing] Data became empty after row removal, header applied to an empty DataFrame.")
        else:
            print("[DEBUG DataProcessing Stage 3] new_columns is None or empty. df_intermediate columns remain: {df_intermediate.columns.tolist()}") # DEBUG ADDED

        final_df = df_intermediate
        
        # Check if dataframe is essentially empty (no data rows or no columns after processing)
        # But allow an empty dataframe that *has* columns (e.g. from a header but no data rows left)
        if final_df.empty and final_df.columns.empty:
            # Only set error if completely empty (no columns, no rows)
            session_state.ts_tool_error_message = "处理后数据为空 (可能由于跳过/表头行设置)。"
            print("[DataProcessing] Processed data is completely empty (no rows, no columns).")
            # It's important to return a DataFrameShell, even if empty, for astype(str) to not fail later
            # For now, we let it flow, astype(str) on empty df with no columns is fine.


        # Step 4: Apply duplicate column removal
        if final_df is not None and not final_df.empty: # Only if there's data
            final_df, removed_cols = remove_duplicates_and_report(final_df)
            session_state.ts_tool_removed_duplicate_cols = removed_cols
            print("--- DEBUG (load_and_preprocess_data): After remove_duplicates_and_report ---")
            print(f"Columns kept: {final_df.columns.tolist()}")
            if removed_cols: print(f"Columns REMOVED due to duplication: {removed_cols}")
            else: print("No columns were removed due to duplication.")
            print(f"Shape after duplicate removal: {final_df.shape}")
        elif final_df is not None: # Empty but possibly has columns
             session_state.ts_tool_removed_duplicate_cols = []
             print("--- DEBUG (load_and_preprocess_data): DataFrame is empty after header/skip processing, remove_duplicates_and_report skipped.")
        else: # final_df is None (shouldn't happen if we start with df_raw and modify)
            session_state.ts_tool_removed_duplicate_cols = []


        # Step 5: Restore session state updates
        if final_df is not None: # Check if final_df is not None before accessing it
            session_state.processed_header_duplicates = check_processed_header_duplicates(final_df)
            session_state.ts_tool_time_col_info = {}

        # Step 6: Convert all columns to string for Arrow compatibility
        # IMPORTANT: This global conversion to string is problematic for numerical/datetime processing.
        # Commenting out for now. If Arrow compatibility requires string types, this conversion 
        # should happen much later, ideally on a copy of the data just before it's needed by Arrow.
        # if final_df is not None and not final_df.empty:
        #     for col in final_df.columns:
        #         try:
        #             final_df[col] = final_df[col].astype(str)
        #         except Exception as e_astype:
        #             print(f"[DataProcessing] Warning: Could not convert column '{col}' to string: {e_astype}")
        #     print("[DataProcessing] Converted all columns of non-empty DataFrame to string type for Arrow compatibility.")
        # elif final_df is not None and final_df.columns.nlevels > 0: # Empty df but with columns (e.g. from header)
        #     final_df.columns = [str(c) for c in final_df.columns] # Ensure column names are strings
        #     print("[DataProcessing] DataFrame is empty but has columns. Ensured column names are strings.")
        # elif final_df is not None: # Empty and no columns
        #      print("[DataProcessing] DataFrame is empty and has no columns. String conversion for data skipped.")

        # <<< 添加调试代码：打印 final_df 中每一列的 dtype >>>
        if final_df is not None:
            print("--- [DEBUG DTYPES PRE-NUMERIC-CONVERSION in load_and_preprocess_data] ---") # MODIFIED
            for col in final_df.columns:
                try:
                    print(f"Column: '{col}', Original Dtype: {final_df[col].dtype}") # MODIFIED
                    # 仅对非 datetime64类型的列尝试转换为数值型
                    if not pd.api.types.is_datetime64_any_dtype(final_df[col]):
                        # 尝试转换为数值型前，可以先尝试替换一些已知的非数值但应视为NaN的字符串
                        # 例如，如果除了common_na_values外，还有其他特定文本代表空
                        # final_df[col] = final_df[col].replace(['一些特殊文本'], pd.NA) 
                        
                        original_non_na_count = final_df[col].notna().sum()
                        converted_series = pd.to_numeric(final_df[col], errors='coerce')
                        
                        # 检查转换后是否所有值都变成了NaN，而原始列有数据
                        # 这可能表示不应将此列视为纯数字
                        if converted_series.isna().all() and original_non_na_count > 0:
                            print(f"Column: '{col}', Conversion to numeric resulted in all NaNs. Reverting to original object type or attempting string.")
                            # 可以选择保留原始 object 类型，或者如果确定是文本，可以强制 astype(str)
                            # 为安全起见，如果全是NaN了，可能原始就是文本，这里我们先不改变它，让后续的datetime转换等逻辑处理
                            # 或者，如果需要强制为字符串： final_df[col] = final_df[col].astype(str)
                        else:
                            final_df[col] = converted_series
                            print(f"Column: '{col}', Dtype after to_numeric: {final_df[col].dtype}")

                except Exception as e_dtype_print:
                    print(f"Column: '{col}', Error during numeric conversion or dtype printing: {e_dtype_print}")
            print("--- [END DEBUG DTYPES POST-NUMERIC-CONVERSION] ---") # MODIFIED
        # <<< 结束调试代码 >>>

        session_state.ts_tool_error_message = None
        return final_df

    except Exception as e:
        error_message = f"在 data_processing.py 的 load_and_preprocess_data 中处理 '{file_name_with_extension}' (Sheet: '{sheet_name_to_load}') 时出错: {e}"
        session_state.ts_tool_error_message = error_message
        print(f"[ERROR DataProcessing] {error_message}")
        import traceback
        print(traceback.format_exc())
        return None

def check_processed_header_duplicates(df_processed):
    """
    Checks for duplicate column names in the processed DataFrame after handling potential suffixes.
    Args:
        df_processed (pd.DataFrame): The DataFrame after initial loading and header setting.
    Returns:
        dict: A dictionary where keys are normalized duplicate base names 
              and values are lists of the original full column names that are duplicates.
              Returns an empty dict if no duplicates are found.
    """
    duplicate_groups = {} 
    if df_processed.empty:
        return duplicate_groups

    final_columns = df_processed.columns.tolist()
    base_columns = [re.sub(r'\.\d+$', '', str(c)) for c in final_columns]
    normalized_base_cols = [bc.strip().lower() for bc in base_columns]
    
    normalized_base_series = pd.Series(normalized_base_cols)
    duplicates_mask = normalized_base_series.duplicated(keep=False)

    if duplicates_mask.any():
        duplicated_normalized_bases = normalized_base_series[duplicates_mask].unique()
        original_final_series = pd.Series(final_columns) 
        
        for norm_base in duplicated_normalized_bases:
            original_indices = normalized_base_series[normalized_base_series == norm_base].index
            original_duplicates = original_final_series.iloc[original_indices].tolist()
            if len(original_duplicates) > 1: 
                 duplicate_groups[norm_base] = original_duplicates
                 
    return duplicate_groups

def apply_rename_rules(df_processed, rename_rules):
    """
    Applies a list of rename rules to the DataFrame columns.
    Args:
        df_processed (pd.DataFrame): The DataFrame to rename columns on.
        rename_rules (list[dict]): A list of dictionaries, where each dict has 
                                   {'original': str, 'new': str}.
    Returns:
        pd.DataFrame: The DataFrame with columns renamed.
    Raises:
        ValueError: If duplicate new names are found within the rules or 
                    if an original column name from rules is not found in the DataFrame.
    """
    if not rename_rules:
        return df_processed 
        
    rename_map = {}
    final_new_names = set()
    originals_in_rules_set = {rule['original'] for rule in rename_rules}
    
    for rule in rename_rules:
        original_col_str = rule['original']
        new_name = rule['new']
        
        actual_original_col = None
        for col_obj in df_processed.columns:
             if str(col_obj) == original_col_str: 
                 actual_original_col = col_obj
                 break
                 
        if actual_original_col is None:
             raise ValueError(f"应用重命名规则时错误：在DataFrame中找不到原始列 '{original_col_str}'。")
             
        if new_name in final_new_names:
            raise ValueError(f"应用重命名规则时错误：规则中包含重复的新列名 '{new_name}'。")
            
        for existing_col in df_processed.columns:
             existing_col_str = str(existing_col)
             if existing_col_str not in originals_in_rules_set and existing_col_str == new_name:
                 raise ValueError(f"应用重命名规则时错误：新列名 '{new_name}' 与一个现有的、未被重命名的列 '{existing_col_str}' 冲突。")

        rename_map[actual_original_col] = new_name
        final_new_names.add(new_name)
        
    try:
        df_renamed = df_processed.rename(columns=rename_map)
        return df_renamed
    except Exception as e:
        raise ValueError(f"Pandas rename操作失败: {e}")

def attempt_auto_datetime_conversion(df_processed):
    """
    Attempts to automatically convert columns that look like date/time columns to datetime objects.
    Args:
        df_processed (pd.DataFrame): The DataFrame to process.
    Returns:
        tuple: (pd.DataFrame, list):
            - The DataFrame with potentially converted datetime columns.
            - A list of column names that were successfully converted.
    """
    if df_processed.empty:
        return df_processed, []
        
    df_converted = df_processed.copy()
    potential_time_cols_detected = []
    common_time_names_for_convert = ['日期', '时间', 'date', 'time', '月份', '年份', '指标']
    
    for col_name in df_converted.columns:
        should_try_convert = False
        if any(tname.lower() in str(col_name).lower() for tname in common_time_names_for_convert):
            should_try_convert = True
        
        if should_try_convert:
            try:
                original_dtype = df_converted[col_name].dtype
                parsed_series, _ = parse_time_column(df_converted[col_name]) 
                
                if parsed_series is not None and str(original_dtype) != str(parsed_series.dtype):
                    df_converted[col_name] = parsed_series
                    potential_time_cols_detected.append(col_name)
                    print(f"Auto-converted column '{col_name}' to datetime.") 
            except Exception as conv_err:
                print(f"Failed to auto-convert column '{col_name}': {conv_err}") 
                pass 
                
    return df_converted, potential_time_cols_detected 

def batch_rename_columns_by_text_replacement(df_input: pd.DataFrame, find_text: str, replace_text: str, time_col_name: str = None):
    """
    Performs batch renaming of DataFrame columns by replacing a specific substring.
    Args:
        df_input (pd.DataFrame): The input DataFrame.
        find_text (str): The substring to find in column names.
        replace_text (str): The substring to replace with. If empty, find_text is removed.
        time_col_name (str, optional): The name of the time column, which will be excluded from renaming.

    Returns:
        tuple: (pd.DataFrame or None, dict)
               - The DataFrame with modified column names if successful and no conflicts, otherwise None.
               - A status dictionary with keys like 'success' (bool), 'message' (str), 
                 'conflicts' (dict, if any), 'modified_map' (dict of old:new names).
    """
    if df_input is None or df_input.empty:
        return None, {'success': False, 'message': "输入数据为空，无法执行批量重命名。", 'conflicts': None, 'modified_map': {}}
    
    if not find_text: # find_text cannot be empty for str.replace to work meaningfully in this context
        return None, {'success': False, 'message': "“要查找的文本”不能为空。", 'conflicts': None, 'modified_map': {}}

    df = df_input.copy()
    original_columns = df.columns.tolist()
    new_column_names = []
    modified_map = {}
    has_changes = False

    for original_col_obj in original_columns:
        original_col_str = str(original_col_obj)
        
        if time_col_name and original_col_str == time_col_name:
            new_column_names.append(original_col_str) # Keep time column name as is
            modified_map[original_col_str] = original_col_str
            continue

        if find_text in original_col_str:
            new_col_str = original_col_str.replace(find_text, replace_text)
            if new_col_str != original_col_str:
                has_changes = True
        else:
            new_col_str = original_col_str
        
        new_column_names.append(new_col_str)
        modified_map[original_col_str] = new_col_str

    if not has_changes:
        return df_input, {'success': True, 'message': "没有列名包含要查找的文本，未做任何更改。", 'conflicts': None, 'modified_map': modified_map}

    # Check for conflicts
    from collections import Counter
    name_counts = Counter(new_column_names)
    conflicts = {name: count for name, count in name_counts.items() if count > 1}

    if conflicts:
        return None, {'success': False, 
                       'message': f"批量替换后产生重复的列名: {list(conflicts.keys())}。请调整规则。", 
                       'conflicts': conflicts, 
                       'modified_map': modified_map}
    
    # If no conflicts, apply new names
    df.columns = new_column_names
    return df, {'success': True, 'message': "批量文本替换成功完成。", 'conflicts': None, 'modified_map': modified_map} 

def remove_duplicates_and_report(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """
    Identifies and removes duplicate columns (those with pandas-generated .N suffixes).
    It keeps the first occurrence (e.g., 'X' or 'X.min_suffix') and removes others.
    
    Args:
        df (pd.DataFrame): Input DataFrame.

    Returns:
        tuple[pd.DataFrame, list[str]]: 
            - DataFrame with duplicate columns removed.
            - List of original column names that were removed due to duplication.
    """
    cols = df.columns.tolist()
    if not cols:
        return df, []

    base_groups = defaultdict(list)
    for col_name_obj in cols: # Iterate over original column objects
        col = str(col_name_obj) # Work with string representation
        match = re.fullmatch(r"(.+)\.(\d+)", col)
        if match:
            base = match.group(1)
            suffix_num = int(match.group(2))
            # Store original column name (as string) along with suffix info
            base_groups[base].append({'original_str': col, 'original_obj': col_name_obj, 'suffix': suffix_num, 'is_suffixed': True})
        else:
            base = col
            base_groups[base].append({'original_str': col, 'original_obj': col_name_obj, 'suffix': -1, 'is_suffixed': False})

    columns_to_keep_obj = [] # Store original column objects to keep
    columns_removed_str = []   # Store string names of removed columns

    for base, members in base_groups.items():
        if not members: # Should not happen with defaultdict
            continue

        # Sort members: non-suffixed first, then by suffix number.
        # This ensures 'X' comes before 'X.1', and 'X.1' before 'X.2'.
        members.sort(key=lambda x: (x['is_suffixed'], x['suffix']))
        
        # Keep the first one from the sorted list
        columns_to_keep_obj.append(members[0]['original_obj'])
        
        # If there were other members, they are duplicates and should be removed
        if len(members) > 1:
            for i in range(1, len(members)):
                columns_removed_str.append(members[i]['original_str'])
                
    # Create the new DataFrame with only the columns to keep
    # Using .loc to select columns by their original objects to handle non-string column names if any
    # However, pandas usually loads string column names. If they are guaranteed strings, df[cols_to_keep_str_list] is fine.
    # For safety with mixed types if they ever occur, or special characters, direct object indexing is safer.
    # But if cols_to_keep_obj contains non-string objects that are not directly in df.columns, this will fail.
    # It's safer to convert kept objects back to their string form IF all original columns were simple strings.
    # Given pandas typical behavior, original_obj should be usable if they are string-like or actual strings.
    
    # Let's assume columns are typically strings or can be reliably selected by their string representation if unique
    # If there were non-string column names that became strings via str(col_name_obj),
    # we need to ensure selection works. df.columns are the actual objects.
    
    # Simplest if all original columns were strings or behaved like unique strings for selection:
    final_columns_to_keep_for_selection = [member['original_obj'] for base, group in base_groups.items() if group for member in [sorted(group, key=lambda x: (x['is_suffixed'], x['suffix']))[0]]]
    
    # Ensure uniqueness in final_columns_to_keep_for_selection just in case (though logic implies it)
    # This step might be redundant if the grouping and selection logic is perfect.
    seen_for_final_selection = set()
    unique_final_columns_to_keep = []
    for col_obj_to_keep in final_columns_to_keep_for_selection:
        if col_obj_to_keep not in seen_for_final_selection:
            unique_final_columns_to_keep.append(col_obj_to_keep)
            seen_for_final_selection.add(col_obj_to_keep)

    df_deduplicated = df[unique_final_columns_to_keep]
    
    return df_deduplicated, columns_removed_str

def deduplicate_column_names(df: pd.DataFrame) -> pd.DataFrame:
    cols = df.columns.tolist()
    if not cols:
        return df

    # Step 1: Group original column names by their base name
    base_groups = defaultdict(list)
    for col in cols:
        match = re.fullmatch(r"(.+)\.(\d+)", col)
        if match:
            base = match.group(1)
            suffix_num = int(match.group(2))
            base_groups[base].append({'original': col, 'suffix': suffix_num, 'is_suffixed': True})
        else:
            base_groups[col].append({'original': col, 'suffix': -1, 'is_suffixed': False}) # -1 for non-suffixed

    rename_mapping = {}
    final_column_candidates = list(cols) # To check for new name clashes with original names not part of current renaming group

    for base, members in base_groups.items():
        # Step 2: Determine if deduplication is needed for this group
        needs_deduplication = False
        if len(members) > 1:
            needs_deduplication = True
        else: # len(members) == 1
            if members[0]['is_suffixed']:
                 # Edge case: a single column like 'X.1' but no 'X'. Treat as needing deduplication to become 'X_v1'.
                 needs_deduplication = True 
        
        if needs_deduplication:
            # Step 3: Sort members: non-suffixed first, then by suffix number
            members.sort(key=lambda x: (x['is_suffixed'], x['suffix']))
            
            for i, member_info in enumerate(members):
                original_col_name = member_info['original']
                current_new_name = f"{base}_v{i+1}"
                
                # Ensure new name is unique across all columns (originals and newly generated ones so far)
                # This check is against final_column_candidates which will be updated with new names
                # And also against already decided rename_mapping values
                k = i + 1
                temp_name_check_list = [name for name in final_column_candidates if name != original_col_name] + list(rename_mapping.values())
                while current_new_name in temp_name_check_list:
                    k += 1
                    current_new_name = f"{base}_v{k}"
                
                if original_col_name != current_new_name: # Only add to map if a change is made
                    rename_mapping[original_col_name] = current_new_name
    
    if rename_mapping:
        df = df.rename(columns=rename_mapping)
    
    return df 