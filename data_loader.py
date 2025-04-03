import pandas as pd
import os
import glob
import numpy as np
import warnings
import json # 引入 json 库
from datetime import datetime
import io # Import io for handling BytesIO from uploads

def load_and_process_data(excel_files_input: list) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """
    Reads Excel files (from paths or uploaded file objects), processes 'weekly' 
    and 'monthly' sheets, merges them, and returns the results.

    Args:
        excel_files_input: A list containing file paths (str) or 
                          Streamlit UploadedFile objects (BytesIO).

    Returns:
        tuple[pd.DataFrame, pd.DataFrame, dict]:
            - df_weekly_all: Merged weekly data.
            - df_monthly_all: Merged monthly data.
            - indicator_source_map: Dictionary mapping indicator names to source filenames.
    """
    weekly_dfs = []
    monthly_dfs = []
    indicator_source_map = {} # Initialize indicator source map
    processed_files_count = 0

    print(f"Starting processing for {len(excel_files_input)} input(s)...")

    for file_input in excel_files_input:
        file_buffer = None
        file_name_for_display = "Unknown File"
        source_name_base = "unknown_source"

        try:
            # Determine if input is a path or an uploaded file object
            if isinstance(file_input, str):
                file_path_or_buffer = file_input
                file_name_for_display = os.path.basename(file_input)
                if file_name_for_display.startswith('~$'): # Skip temp files
                    print(f"Skipping temporary file: {file_name_for_display}")
                    continue
                source_name_base = os.path.splitext(file_name_for_display)[0]
            elif hasattr(file_input, 'name') and hasattr(file_input, 'getvalue'): # Check for UploadedFile attributes
                file_path_or_buffer = file_input
                file_name_for_display = file_input.name
                source_name_base = os.path.splitext(file_name_for_display)[0]
                # Ensure the buffer is reset if read multiple times (though pd.read_excel usually handles it)
                file_input.seek(0) 
            else:
                warnings.warn(f"Skipping invalid input type: {type(file_input)}")
                continue

            print(f"--- Processing file: {file_name_for_display} ---")

            # --- Process Weekly Sheet ---
            try:
                print("  Reading 'weekly' sheet...")
                df_w = pd.read_excel(file_path_or_buffer, sheet_name='weekly')
                
                # Check if sheet is essentially empty after reading
                if df_w.empty:
                     warnings.warn(f"File '{file_name_for_display}' 'weekly' sheet is empty. Skipping weekly data.")
                else:
                    # Replace near-zero with NaN
                    small_threshold = 1e-9 
                    numeric_cols_w = df_w.select_dtypes(include=np.number).columns
                    for col in numeric_cols_w:
                        df_w[col] = df_w[col].mask(df_w[col].abs() < small_threshold, np.nan)
                    print(f"  Applied near-zero threshold to '{file_name_for_display}' weekly sheet.")
                    
                    if df_w.drop(columns=df_w.select_dtypes(exclude=np.number).columns, errors='ignore').isnull().all().all():
                        warnings.warn(f"File '{file_name_for_display}' 'weekly' sheet is all NaN after thresholding. Skipping weekly data.")
                    else:
                        print("  Preprocessing 'weekly' data...")
                        date_col_name = df_w.columns[0]
                        try:
                            df_w[date_col_name] = pd.to_datetime(df_w[date_col_name], errors='coerce')
                            original_rows = len(df_w)
                            df_w = df_w.dropna(subset=[date_col_name])
                            if len(df_w) < original_rows:
                                warnings.warn(f"File '{file_name_for_display}' weekly: Removed {original_rows - len(df_w)} rows with invalid dates.")
                            df_w.set_index(date_col_name, inplace=True)
                            df_w.sort_index(inplace=True)
                            if not df_w.index.is_unique:
                                warnings.warn(f"File '{file_name_for_display}' weekly: Found duplicate dates, keeping last.")
                                df_w = df_w[~df_w.index.duplicated(keep='last')]
                            
                            # Resample to Friday ('W-FRI')
                            df_w = df_w.resample('W-FRI').last()
                            print(f"  Successfully processed weekly data. Last date: {df_w.index[-1].strftime('%Y-%m-%d') if not df_w.empty else 'N/A'}")
                            
                            # Add to weekly list and map indicators
                            weekly_dfs.append(df_w)
                            for col in df_w.columns:
                                if col not in indicator_source_map: # Avoid overwriting if column name exists in multiple files
                                    indicator_source_map[col] = source_name_base
                                else:
                                    warnings.warn(f"Duplicate column name '{col}' found in '{file_name_for_display}'. Source mapping kept from first occurrence.")

                        except Exception as e:
                            warnings.warn(f"Error preprocessing weekly data for '{file_name_for_display}': {e}. Skipping weekly data.")

            except ValueError as ve:
                if 'Worksheet named' in str(ve) and 'weekly' in str(ve):
                    warnings.warn(f"File '{file_name_for_display}' missing 'weekly' sheet. Skipping weekly data.")
                else:
                    warnings.warn(f"Error reading 'weekly' sheet from '{file_name_for_display}': {ve}. Skipping weekly data.")
            except Exception as e:
                warnings.warn(f"Unexpected error processing 'weekly' sheet from '{file_name_for_display}': {e}. Skipping weekly data.")

            # --- Process Monthly Sheet ---
            # Ensure the buffer is reset if it's an uploaded file
            if hasattr(file_input, 'seek'): file_input.seek(0)
                
            try:
                print("  Reading 'monthly' sheet...")
                # Need to pass the buffer/path again
                df_m = pd.read_excel(file_path_or_buffer, sheet_name='monthly')
                
                if df_m.empty:
                    warnings.warn(f"File '{file_name_for_display}' 'monthly' sheet is empty. Skipping monthly data.")
                else:
                    # Replace near-zero with NaN
                    small_threshold = 1e-9 
                    numeric_cols_m = df_m.select_dtypes(include=np.number).columns
                    for col in numeric_cols_m:
                        df_m[col] = df_m[col].mask(df_m[col].abs() < small_threshold, np.nan)
                    print(f"  Applied near-zero threshold to '{file_name_for_display}' monthly sheet.")

                    if df_m.drop(columns=df_m.select_dtypes(exclude=np.number).columns, errors='ignore').isnull().all().all():
                         warnings.warn(f"File '{file_name_for_display}' 'monthly' sheet is all NaN after thresholding. Skipping monthly data.")
                    else:
                        print("  Preprocessing 'monthly' data...")
                        date_col_name_m = df_m.columns[0]
                        try:
                            df_m[date_col_name_m] = pd.to_datetime(df_m[date_col_name_m], errors='coerce')
                            original_rows_m = len(df_m)
                            df_m = df_m.dropna(subset=[date_col_name_m])
                            if len(df_m) < original_rows_m:
                                warnings.warn(f"File '{file_name_for_display}' monthly: Removed {original_rows_m - len(df_m)} rows with invalid dates.")
                            df_m.set_index(date_col_name_m, inplace=True)
                            df_m.sort_index(inplace=True)
                            if not df_m.index.is_unique:
                                warnings.warn(f"File '{file_name_for_display}' monthly: Found duplicate dates, keeping last.")
                                df_m = df_m[~df_m.index.duplicated(keep='last')]
                            
                            print(f"  Successfully processed monthly data. Last date: {df_m.index[-1].strftime('%Y-%m-%d') if not df_m.empty else 'N/A'}")
                            
                            # Add to monthly list and map indicators
                            monthly_dfs.append(df_m)
                            for col in df_m.columns:
                                if col not in indicator_source_map:
                                    indicator_source_map[col] = source_name_base
                                else:
                                     # Only warn if it wasn't already warned for weekly
                                     if indicator_source_map[col] != source_name_base:
                                         warnings.warn(f"Duplicate column name '{col}' found in '{file_name_for_display}'. Source mapping kept from first occurrence.")

                        except Exception as e:
                            warnings.warn(f"Error preprocessing monthly data for '{file_name_for_display}': {e}. Skipping monthly data.")
            
            except ValueError as ve:
                if 'Worksheet named' in str(ve) and 'monthly' in str(ve):
                    warnings.warn(f"File '{file_name_for_display}' missing 'monthly' sheet. Skipping monthly data.")
                else:
                    warnings.warn(f"Error reading 'monthly' sheet from '{file_name_for_display}': {ve}. Skipping monthly data.")
            except Exception as e:
                warnings.warn(f"Unexpected error processing 'monthly' sheet from '{file_name_for_display}': {e}. Skipping monthly data.")
                
            processed_files_count += 1
            print(f"--- File {file_name_for_display} processing finished ---")
        
        except Exception as e:
            warnings.warn(f"General error processing input {file_name_for_display}: {e}")

    if processed_files_count == 0:
        warnings.warn("No files were successfully processed.")
        return pd.DataFrame(), pd.DataFrame(), {}

    # --- Merge DataFrames ---
    print("Merging weekly data...")
    df_weekly_all = pd.concat(weekly_dfs, axis=1) if weekly_dfs else pd.DataFrame()
    # Handle potential duplicate columns after concat (though mapping logic tries to prevent this)
    df_weekly_all = df_weekly_all.loc[:, ~df_weekly_all.columns.duplicated(keep='first')]

    print("Merging monthly data...")
    df_monthly_all = pd.concat(monthly_dfs, axis=1) if monthly_dfs else pd.DataFrame()
    df_monthly_all = df_monthly_all.loc[:, ~df_monthly_all.columns.duplicated(keep='first')]

    print(f"Data processing complete. Weekly shape: {df_weekly_all.shape}, Monthly shape: {df_monthly_all.shape}")

    return df_weekly_all, df_monthly_all, indicator_source_map

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
    print("Starting data_loader execution...")
    
    # Configuration for standalone execution
    data_directory = '.'
    excluded_files = [
        "merged_output.xlsx", 
        "weekly_summary.xlsx", 
        "weekly_summary_unformatted.xlsx", 
        "industry_diffusion_indices.xlsx", 
        "industry_diffusion_indices_by_keyword.xlsx", 
        "industry_diffusion_indices_by_source.xlsx", 
        "industry_diffusion_indices_combined.xlsx",
        "industry_diffusion_indices_separate_sheets.xlsx",
        "summary_output.xlsx",
        # Add verification output files if needed
        "black_metal_combined_verification.txt",
        "chemical_fiber_combined_verification.txt",
        "chemical_raw_materials_combined_verification.txt",
        "petroleum_coal_fuel_combined_verification.txt",
        "rubber_plastic_combined_verification.txt"
    ]
    output_merged_excel = "merged_output.xlsx"
    output_map_json = "indicator_source_mapping.json"

    # Find files in the directory
    search_path = os.path.join(data_directory, '*.xlsx')
    all_excel_files = glob.glob(search_path)
    
    # Filter out excluded files and temp files
    valid_excel_paths = [f for f in all_excel_files 
                         if os.path.basename(f) not in excluded_files and 
                         not os.path.basename(f).startswith('~$')]

    if not valid_excel_paths:
        print(f"No valid source Excel files found in directory '{data_directory}' (excluding {len(excluded_files)} specified files). Halting.")
    else:
        print(f"Found {len(valid_excel_paths)} potential source Excel files to process.")
        # Call the main processing function with file paths
        merged_weekly_df, merged_monthly_df, source_map = load_and_process_data(valid_excel_paths)
        
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

