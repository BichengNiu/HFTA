import pandas as pd
import os
import glob
import numpy as np
import warnings
import json # 引入 json 库
from datetime import datetime
import io # Import io for handling BytesIO from uploads
from typing import List, Tuple, Dict, Any, Optional
# --- <<< 新增：导入 re 用于正则替换 >>> ---
import re 
# --- <<< 结束新增 >>> ---

# --- <<< 新增：标准化字符串函数 >>> ---
def normalize_string(s: str) -> str:
    """将字符串转换为标准化形式：半角，去除首尾空格，合并中间空格。"""
    if not isinstance(s, str):
        return s # Return original if not a string
    # 转换全角为半角 (常见标点和空格)
    full_width = "（）：　"
    half_width = "(): "
    translation_table = str.maketrans(full_width, half_width)
    s = s.translate(translation_table)
    # 特殊处理：确保冒号后面有空格
    s = re.sub(r':(?!\s)', ': ', s)
    # 去除首尾空格
    s = s.strip()
    # 合并中间多余空格
    s = re.sub(r'\s+', ' ', s)
    return s
# --- <<< 结束新增 >>> ---

def load_and_process_data(excel_files_input: List[Any]) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Dict[str, str], Dict[str, str]]:
    """
    Reads Excel files (from paths or uploaded file objects), processes sheets
    containing '周度', '月度', or '日度' in their names, merges them,
    and returns the results. Handles specific header rows and averages daily data weekly.
    Also attempts to read an '指标体系' sheet for indicator categorization.

    Args:
        excel_files_input: A list containing file paths (str) or
                          Streamlit UploadedFile objects (BytesIO).

    Returns:
        tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, str], dict[str, str]]:
            - df_weekly_all: Merged weekly-frequency data (original weekly + averaged daily).
            - df_monthly_all: Merged monthly data.
            - df_daily_all: Merged daily data (original daily frequency).
            - indicator_source_map: Dictionary mapping indicator names to source 'filename|sheetname'.
            - indicator_industry_map: Dictionary mapping indicator names to industries from '指标体系'.
    """
    all_weekly_dfs = [] # Collect all weekly-frequency DFs here
    all_monthly_dfs = [] # Collect all monthly DFs here
    all_daily_dfs = [] # Collect all daily DFs here
    indicator_source_map = {} # Initialize indicator source map
    indicator_industry_map = {} # Initialize industry map
    type_mapping_sheet_name = '指标体系' # Default name for the category mapping sheet
    processed_files_count = 0
    first_file_processed = False # Flag to read mapping only from the first file

    print(f"Starting processing for {len(excel_files_input)} input(s)...")

    for file_input in excel_files_input:
        file_path_or_buffer = None
        file_name_for_display = "Unknown File"
        source_name_base = "unknown_source"
        excel_file_handler = None # To store the ExcelFile object

        try:
            # Determine if input is a path or an uploaded file object
            if isinstance(file_input, str):
                file_path_or_buffer = file_input
                file_name_for_display = os.path.basename(file_input)
                if file_name_for_display.startswith('~$'): # Skip temp files
                    print(f"Skipping temporary file: {file_name_for_display}")
                    continue
                source_name_base = os.path.splitext(file_name_for_display)[0]
                # Use pd.ExcelFile to get sheet names without reading all data yet
                excel_file_handler = pd.ExcelFile(file_path_or_buffer)
            elif hasattr(file_input, 'name') and hasattr(file_input, 'getvalue'): # Check for UploadedFile attributes
                # Read the uploaded file into memory first, as pd.ExcelFile might need it
                # Use BytesIO to keep it in memory
                file_buffer = io.BytesIO(file_input.getvalue())
                file_path_or_buffer = file_buffer # Use the buffer for reading
                file_name_for_display = file_input.name
                source_name_base = os.path.splitext(file_name_for_display)[0]
                excel_file_handler = pd.ExcelFile(file_path_or_buffer)
            else:
                warnings.warn(f"Skipping invalid input type: {type(file_input)}")
                continue

            print(f"--- Processing file: {file_name_for_display} ---")
            sheet_names = excel_file_handler.sheet_names
            print(f"  Found sheets: {', '.join(sheet_names)}")

            # 尝试读取指标体系 (仅从第一个文件)
            if not first_file_processed and type_mapping_sheet_name in sheet_names:
                print(f"  -- Attempting to read industry mapping from sheet: '{type_mapping_sheet_name}' --")
                try:
                    if isinstance(file_path_or_buffer, io.BytesIO):
                        file_path_or_buffer.seek(0)
                        
                    df_mapping = pd.read_excel(excel_file_handler, 
                                               sheet_name=type_mapping_sheet_name, 
                                               usecols=['高频指标', '行业'])
                    df_mapping['高频指标'] = df_mapping['高频指标'].apply(normalize_string)
                    df_mapping.dropna(subset=['高频指标', '行业'], inplace=True)
                    indicator_industry_map = pd.Series(df_mapping['行业'].values, index=df_mapping['高频指标']).to_dict()
                    print(f"     Successfully read {len(indicator_industry_map)} industry mappings.")
                except KeyError as ke:
                    warnings.warn(f"     Sheet '{type_mapping_sheet_name}' found, but required columns ('高频指标', '行业') not present or named differently: {ke}. Skipping mapping.")
                except Exception as e_map:
                    warnings.warn(f"     Error reading industry mapping sheet '{type_mapping_sheet_name}': {e_map}. Skipping mapping.")
            elif not first_file_processed:
                 print(f"  -- Industry mapping sheet '{type_mapping_sheet_name}' not found in this file. --")
            # Mark first file as processed regarding mapping sheet
            first_file_processed = True 

            processed_sheets_in_file = 0

            for sheet_name in sheet_names:
                # --- <<< 新增：跳过指标体系sheet本身的处理 >>> ---
                if sheet_name == type_mapping_sheet_name:
                    continue 
                # --- <<< 结束新增 >>> ---
                
                print(f"  -- Analyzing sheet: '{sheet_name}' --")
                # Determine header row based on source type inferred from sheet name
                header_row = 1 # Default: use row 1 (second row) as header for most sources
                skip_rows_after_header = 0 # Number of rows to skip after header row

                # CORRECTED: Check if 'mysteel' (case-insensitive) is in the name
                if "mysteel" in sheet_name.lower() or "myteel" in sheet_name.lower(): # Handle typo and case
                    header_row = 1 # Mysteel uses the second row (index 1) as header
                    skip_rows_after_header = 2 # Skip frequency and description rows
                    print(f"     Detected 'Mysteel/Myteel', using header row {header_row + 1}, skipping {skip_rows_after_header} metadata rows.")
                elif "同花顺" in sheet_name: # 同花顺 uses second row as header, skip metadata rows
                    header_row = 1 # Use row 1 as header (指标名称 row)
                    skip_rows_after_header = 3 # Skip 频率, 单位, 指标ID rows
                    print(f"     Detected '同花顺', using header row {header_row + 1}, skipping {skip_rows_after_header} metadata rows.")
                # Add elif for other specific formats if needed
                else: # Default case (e.g., Wind)
                    header_row = 1 # Use row 1 as header
                    skip_rows_after_header = 0 # No additional rows to skip
                    print(f"     Using default header row {header_row + 1}, skipping {skip_rows_after_header} additional rows.")


                data_type = None
                if "周度" in sheet_name:
                    data_type = "weekly"
                elif "月度" in sheet_name:
                    data_type = "monthly"
                elif "日度" in sheet_name:
                    data_type = "daily"
                else:
                    # Skip sheets not matching expected types (like '指标体系')
                    print(f"     Sheet name does not contain '周度', '月度', or '日度'. Skipping.")
                    continue # Make sure to skip processing for these sheets

                try:
                    # Reset buffer position if reading from BytesIO
                    if isinstance(file_path_or_buffer, io.BytesIO):
                        file_path_or_buffer.seek(0)

                    print(f"     Reading '{data_type}' data from sheet '{sheet_name}' with header={header_row}...")
                    # Force read as object first AND prevent default NA value interpretation
                    df_sheet = pd.read_excel(excel_file_handler, 
                                             sheet_name=sheet_name, 
                                             header=header_row, 
                                             dtype=object, 
                                             keep_default_na=False, 
                                             na_values=[])

                    # Handle specific row skipping AFTER reading header
                    if skip_rows_after_header > 0 and not df_sheet.empty:
                        # Skip metadata rows after header (for 同花顺, Mysteel, etc.)
                        print(f"     Skipping {skip_rows_after_header} metadata rows after header.")
                        if len(df_sheet) > skip_rows_after_header:
                            df_sheet = df_sheet.iloc[skip_rows_after_header:].reset_index(drop=True)
                        else:
                            print(f"     Warning: Not enough rows to skip {skip_rows_after_header} rows. Sheet has only {len(df_sheet)} rows.")
                            df_sheet = pd.DataFrame()  # Empty dataframe if not enough rows

                    if df_sheet.empty:
                        # Warning if empty after potential row skipping or initially
                        warnings.warn(f"     Sheet '{sheet_name}' in '{file_name_for_display}' is empty or became empty after skipping rows. Skipping.")
                        continue

                    # Validate column names - check for unnamed columns which indicate incorrect header detection
                    unnamed_cols = [col for col in df_sheet.columns if str(col).startswith('Unnamed')]
                    if len(unnamed_cols) > len(df_sheet.columns) * 0.5:  # If more than 50% columns are unnamed
                        warnings.warn(f"     Sheet '{sheet_name}': Too many unnamed columns ({len(unnamed_cols)}/{len(df_sheet.columns)}). This may indicate incorrect header row detection.")
                        print(f"     Column names: {list(df_sheet.columns)}")
                        print(f"     First few rows of data:")
                        print(df_sheet.head(3).to_string())
                    else:
                        print(f"     Successfully detected {len(df_sheet.columns)} columns: {list(df_sheet.columns)[:5]}{'...' if len(df_sheet.columns) > 5 else ''}")

                    # Common Preprocessing Steps
                    # 1. Handle Date Column (assuming it's the first column)
                    date_col_name = df_sheet.columns[0]
                    print(f"     Using '{date_col_name}' as date column.")
                    try:
                        df_sheet[date_col_name] = pd.to_datetime(df_sheet[date_col_name], errors='coerce')
                        original_rows = len(df_sheet)
                        df_sheet = df_sheet.dropna(subset=[date_col_name])
                        if len(df_sheet) < original_rows:
                            warnings.warn(f"     Sheet '{sheet_name}': Removed {original_rows - len(df_sheet)} rows with invalid dates.")
                        
                        if df_sheet.empty:
                            warnings.warn(f"     Sheet '{sheet_name}' became empty after removing invalid dates. Skipping.")
                            continue

                        df_sheet.set_index(date_col_name, inplace=True)
                        df_sheet.sort_index(inplace=True)

                        if not df_sheet.index.is_unique:
                            warnings.warn(f"     Sheet '{sheet_name}': Found duplicate dates, keeping last.")
                            df_sheet = df_sheet[~df_sheet.index.duplicated(keep='last')]

                    except Exception as e:
                        warnings.warn(f"     Error processing date column for sheet '{sheet_name}': {e}. Skipping sheet.")
                        continue
                        
                    # Attempt numeric conversion for all non-index columns
                    print(f"     Attempting numeric conversion for columns: {list(df_sheet.columns)}")
                    for col in df_sheet.columns:
                        df_sheet[col] = pd.to_numeric(df_sheet[col], errors='coerce')
                    print("     Numeric conversion attempt finished.")
                    
                    # Re-identify numeric columns AFTER coercion attempt
                    numeric_cols = df_sheet.select_dtypes(include=np.number).columns

                    # 2. Replace near-zero AND zero with NaN in numeric columns
                    small_threshold = 1e-9
                    if not numeric_cols.empty:
                        print(f"     Applying near-zero threshold ({small_threshold}) and replacing exact zeros in numeric columns: {list(numeric_cols)}")
                        for col in numeric_cols:
                            df_sheet[col] = df_sheet[col].mask(
                                (df_sheet[col].abs() < small_threshold) | (df_sheet[col] == 0),
                                np.nan
                            )
                        
                        # Check if all numeric data became NaN after coercion, thresholding and zero replacement
                        if df_sheet[numeric_cols].isnull().all().all():
                           warnings.warn(f"     Sheet '{sheet_name}' is all NaN after numeric conversion and replacements. Skipping sheet.")
                           continue
                    else:
                         print("     No numeric columns found after conversion attempt.")

                    # Type-Specific Processing
                    source_identifier_prefix = f"{source_name_base}|{sheet_name}"

                    # Date Adjustment for specific Monthly Indicators
                    if data_type == "monthly":
                        cols_to_adjust = [col for col in df_sheet.columns if "工业增加值" in col]
                        if cols_to_adjust:
                            print(f"     Adjusting date index for indicators containing '工业增加值' ({len(cols_to_adjust)} found). Shifting back one month (MonthEnd).")
                            if pd.api.types.is_datetime64_any_dtype(df_sheet.index):
                                try:
                                    df_sheet.index = df_sheet.index - pd.offsets.MonthEnd(1)
                                    print(f"     Adjusted index range: {df_sheet.index.min()} to {df_sheet.index.max()}")
                                except Exception as e_offset:
                                     warnings.warn(f"     Failed to apply MonthEnd offset for sheet '{sheet_name}': {e_offset}")
                            else:
                                warnings.warn(f"     Cannot adjust index for sheet '{sheet_name}' as it's not a DatetimeIndex.")

                    if data_type == "weekly":
                        print("     Processing as Weekly data...")
                        df_sheet = df_sheet.resample('W-FRI').last()
                        print(f"     Resampled to W-FRI (last). Last date: {df_sheet.index[-1].strftime('%Y-%m-%d') if not df_sheet.empty else 'N/A'}")
                        
                        # Map indicators
                        for col in df_sheet.columns:
                            map_key = normalize_string(col) 
                            if map_key not in indicator_source_map:
                                indicator_source_map[map_key] = source_identifier_prefix
                            else:
                                warnings.warn(f"Duplicate normalized indicator name '{map_key}' found (Original: '{col}'). Keeping source '{indicator_source_map[map_key]}'. Ignoring source '{source_identifier_prefix}'.")
                        
                        df_sheet.columns = [normalize_string(col) for col in df_sheet.columns]
                        all_weekly_dfs.append(df_sheet)

                    elif data_type == "monthly":
                        print("     Processing as Monthly data...")
                        print(f"     Processed monthly data. Last date: {df_sheet.index[-1].strftime('%Y-%m-%d') if not df_sheet.empty else 'N/A'}")
                        # Map indicators
                        for col in df_sheet.columns:
                            map_key = normalize_string(col) 
                            if map_key not in indicator_source_map:
                                indicator_source_map[map_key] = source_identifier_prefix
                            else:
                                warnings.warn(f"Duplicate normalized indicator name '{map_key}' found (Original: '{col}'). Keeping source '{indicator_source_map[map_key]}'. Ignoring source '{source_identifier_prefix}'.")
                        
                        df_sheet.columns = [normalize_string(col) for col in df_sheet.columns]
                        all_monthly_dfs.append(df_sheet)

                    elif data_type == "daily":
                        print("     Processing as Daily data...")
                        # Keep original daily data
                        df_daily_original = df_sheet.copy()
                        print(f"     Processed daily data. Last date: {df_daily_original.index[-1].strftime('%Y-%m-%d') if not df_daily_original.empty else 'N/A'}")
                        
                        # Also create weekly averaged version for weekly analysis
                        df_weekly_from_daily = df_sheet.resample('W-FRI').mean()
                        print(f"     Also created weekly average from daily. Last date: {df_weekly_from_daily.index[-1].strftime('%Y-%m-%d') if not df_weekly_from_daily.empty else 'N/A'}")
                        
                        # Map indicators for both daily and weekly
                        for col in df_sheet.columns:
                            map_key = normalize_string(col) 
                            if map_key not in indicator_source_map:
                                indicator_source_map[map_key] = source_identifier_prefix
                            else:
                                warnings.warn(f"Duplicate normalized indicator name '{map_key}' found (Original: '{col}'). Keeping source '{indicator_source_map[map_key]}'. Ignoring source '{source_identifier_prefix}'.")
                        
                        # Normalize column names
                        df_daily_original.columns = [normalize_string(col) for col in df_daily_original.columns]
                        df_weekly_from_daily.columns = [normalize_string(col) for col in df_weekly_from_daily.columns]
                        
                        # Add to respective collections
                        all_daily_dfs.append(df_daily_original)
                        all_weekly_dfs.append(df_weekly_from_daily)

                    processed_sheets_in_file += 1

                except Exception as e:
                    warnings.warn(f"     Error processing sheet '{sheet_name}' in '{file_name_for_display}': {e}. Skipping sheet.")

            if processed_sheets_in_file > 0:
                 processed_files_count += 1
            else:
                 warnings.warn(f"--- No valid data sheets found or processed in file: {file_name_for_display} ---")
                 
            print(f"--- File {file_name_for_display} processing finished ---")

        except Exception as e:
            warnings.warn(f"General error processing input {file_name_for_display}: {e}")
        finally:
            # Close the ExcelFile object if it was opened
             if excel_file_handler:
                 try:
                     # If it was created from a buffer, the buffer needs closing too
                     # if isinstance(file_path_or_buffer, io.BytesIO):
                     #    file_path_or_buffer.close()
                     excel_file_handler.close()
                 except Exception as e_close:
                      warnings.warn(f"Error closing resources for {file_name_for_display}: {e_close}")


    if processed_files_count == 0:
        warnings.warn("No files were successfully processed.")
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), {}, {}

    # Merge DataFrames
    print("\nMerging weekly-frequency data (weekly + averaged daily)...")
    df_weekly_all = pd.concat(all_weekly_dfs, axis=1) if all_weekly_dfs else pd.DataFrame()
    if not df_weekly_all.empty:
        df_weekly_all = df_weekly_all.loc[:, ~df_weekly_all.columns.duplicated(keep='first')]
        df_weekly_all.sort_index(inplace=True)
    print(f"Final weekly data shape: {df_weekly_all.shape}")

    print("Merging monthly data...")
    df_monthly_all = pd.concat(all_monthly_dfs, axis=1) if all_monthly_dfs else pd.DataFrame()
    if not df_monthly_all.empty:
        df_monthly_all = df_monthly_all.loc[:, ~df_monthly_all.columns.duplicated(keep='first')]
        df_monthly_all.sort_index(inplace=True)
        
        # Align monthly dates to month end
        print("     Aligning monthly data index to month end...")
        if pd.api.types.is_datetime64_any_dtype(df_monthly_all.index):
             # Note: Month-end alignment disabled based on user's data structure
             warnings.warn("Monthly index alignment to month-end is disabled to preserve data integrity.")
            
    print(f"Final monthly data shape: {df_monthly_all.shape}")

    print("Merging daily data...")
    df_daily_all = pd.concat(all_daily_dfs, axis=1) if all_daily_dfs else pd.DataFrame()
    if not df_daily_all.empty:
        df_daily_all = df_daily_all.loc[:, ~df_daily_all.columns.duplicated(keep='first')]
        df_daily_all.sort_index(inplace=True)
    print(f"Final daily data shape: {df_daily_all.shape}")

    print(f"\nData processing complete.")

    return df_weekly_all, df_monthly_all, df_daily_all, indicator_source_map, indicator_industry_map

def save_merged_data(df_weekly, df_monthly, output_file="merged_output.xlsx"):
    """Saves the merged weekly and monthly dataframes to an Excel file."""
    print(f"\nPreparing to save merged data to '{output_file}'...")
    try:
        with pd.ExcelWriter(output_file, engine='xlsxwriter', datetime_format='yyyy-mm-dd') as writer:
            if not df_weekly.empty:
                df_weekly.to_excel(writer, sheet_name="WeeklyData")
                print(f"  Weekly data written to sheet 'WeeklyData' ({df_weekly.shape[0]} rows, {df_weekly.shape[1]} columns).")
            else:
                print("  Weekly data is empty, skipping sheet 'WeeklyData'.")
            
            if not df_monthly.empty:
                df_monthly.to_excel(writer, sheet_name="MonthlyData")
                print(f"  Monthly data written to sheet 'MonthlyData' ({df_monthly.shape[0]} rows, {df_monthly.shape[1]} columns).")
            else:
                print("  Monthly data is empty, skipping sheet 'MonthlyData'.")
        print(f"Successfully saved merged data to '{output_file}'")
    except Exception as e:
        print(f"Error saving merged data to Excel: {e}")

def save_source_map(source_map, output_file="indicator_source_mapping.json"):
    """Saves the indicator source map to a JSON file."""
    print(f"\nPreparing to save indicator source map to '{output_file}'...")
    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(source_map, f, ensure_ascii=False, indent=4)
        print(f"Successfully saved indicator source map to '{output_file}' ({len(source_map)} indicators).")
    except Exception as e:
        print(f"Error saving source map to JSON: {e}")

# Keep the main execution block for standalone script usage
if __name__ == "__main__":
    start_time = datetime.now()
    print("Starting data_loader execution for specific file test...")
    
    # --- Configuration for standalone execution --- 
    # Explicitly target the single file for this test run
    target_file_path = os.path.join('data', '经济数据库0508.xlsx')
    # data_directory = '.' # No longer needed for directory scanning
    # excluded_files = [...] # No longer needed
    output_merged_excel = "merged_output_test.xlsx" # Use a test output file
    output_map_json = "indicator_source_mapping_test.json" # Use a test output file

    # Directly use the target file path
    valid_excel_paths = [target_file_path]

    if not os.path.exists(target_file_path):
         print(f"Error: Target file not found at '{target_file_path}'. Halting.")
    # elif not valid_excel_paths: # Simplified check
    #     print(f"No valid source Excel files found... Halting.") # Should not happen with direct path
    else:
        print(f"Processing single target file: '{target_file_path}'")
        # Call the main processing function with file paths
        merged_weekly_df, merged_monthly_df, merged_daily_df, source_map, industry_map = load_and_process_data(valid_excel_paths)
        
        # Save the results only when run as a script
        if not merged_weekly_df.empty or not merged_monthly_df.empty:
            save_merged_data(merged_weekly_df, merged_monthly_df, output_merged_excel)
        else:
            print("No data merged, skipping saving merged Excel file.")
            
        if source_map:
            save_source_map(source_map, output_map_json)
        else:
             print("No source map generated, skipping saving JSON file.")

    end_time = datetime.now()
    print(f"\ndata_loader finished execution. Duration: {end_time - start_time}")

