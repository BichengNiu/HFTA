import streamlit as st
import pandas as pd
import io
from typing import Dict, Any, Tuple, Optional, List
import numpy as np # Ensure numpy is imported for np.nan handling
import json # For robust printing of complex structures
import copy # Ensure copy is imported

# --- Serialization Helper Functions ---
def _serialize_value(value):
    """Converts individual problematic pandas/numpy types to JSON-friendly formats."""
    if isinstance(value, (pd.Timestamp, pd.NaT.__class__)) or hasattr(value, 'to_period'):
        return str(value)
    if pd.isna(value) and not isinstance(value, pd.NaT.__class__): # Handles np.nan, pd.NA
        return None
    if isinstance(value, (np.integer, np.floating, np.bool_)):
        return value.item()
    if isinstance(value, (pd.Timedelta)):
        return str(value) # Convert Timedelta to string
    return value

def _serialize_complex_object(obj: Any) -> Any:
    """Recursively serializes complex objects (dicts, lists, DataFrames, Series) containing problematic types."""
    if isinstance(obj, dict):
        return {k: _serialize_complex_object(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [_serialize_complex_object(item) for item in obj]
    elif isinstance(obj, pd.DataFrame):
        df_reset = obj.reset_index() # Ensure index becomes a column
        # Convert to list of dicts, where each dict is a row, then serialize each dict
        return [_serialize_complex_object(row.to_dict()) for _, row in df_reset.iterrows()]
    elif isinstance(obj, pd.Series):
        # Convert Series to dict (respecting name for index if Series is named), then serialize dict
        # If series has a name, its index name is used. Otherwise, 'index' for index, 0 for values if no name.
        # A simple to_dict() might be sufficient if we just need key-value pairs.
        # For more structure, consider series.reset_index().to_dict('records') then serialize.
        s_dict = obj.to_dict()
        return _serialize_complex_object(s_dict) 
    else:
        return _serialize_value(obj)
# --- End of Serialization Helper Functions ---

# --- Constants for session_state keys (if not already defined elsewhere) ---
PENDING_UPLOADS_DC_KEY = 'dc_m1_pending_uploads'
GLOBAL_STAGED_DATA_KEY = 'staged_data' # Use the global staged data key

def handle_uploaded_file_for_comparison(session_state: Dict[str, Any], uploaded_file_obj: Dict[str, Any]) -> Tuple[bool, List[str]]:
    """
    Handles uploaded files for the data comparison module by directly reading them.
    If an Excel file has multiple sheets, each sheet is treated as a separate pending upload.
    The first column of each DataFrame is automatically marked as the 'selected_time_col'.
    Checks if the file/sheet is already in staged_data or pending_uploads before processing.
    Returns a tuple: (bool indicating if any operation was performed, List of messages).
    """
    messages: List[str] = []
    any_operation_performed_successfully = False # True if any part of the file is added to pending or confirmed as already handled.

    if not uploaded_file_obj:
        messages.append("没有提供上传文件。")
        return False, messages

    original_file_name = uploaded_file_obj.name
    file_content_bytes = uploaded_file_obj.getvalue()
    
    pending_uploads = session_state.setdefault(PENDING_UPLOADS_DC_KEY, {})
    staged_data = session_state.get(GLOBAL_STAGED_DATA_KEY, {}) # Use .get, default empty if not found

    # --- Inner function to process a given DataFrame (from a sheet or a CSV file) ---
    def _process_individual_df(df_to_process: pd.DataFrame, 
                               key_for_tracking: str, # This is the key for pending_uploads/staged_data
                               display_name_for_messages: str, 
                               sheet_name_to_store: Optional[str]): # Sheet name for the value dict, None for CSVs
        nonlocal any_operation_performed_successfully

        if key_for_tracking in staged_data:
            messages.append(f"{display_name_for_messages} 已存在于暂存区，将被跳过。")
            any_operation_performed_successfully = True # Recognized as handled
            return

        if key_for_tracking in pending_uploads:
            messages.append(f"{display_name_for_messages} 已在待处理列表中，将被跳过。")
            any_operation_performed_successfully = True # Recognized as handled
            return
        
        # Add to pending_uploads
        first_col_name = df_to_process.columns[0] if not df_to_process.empty and len(df_to_process.columns) > 0 else None
        pending_uploads[key_for_tracking] = {
            'original_filename': original_file_name, # original_file_name is from the uploaded_file_obj
            'sheet_name': sheet_name_to_store,
            'df_preview': df_to_process, 
            'selected_time_col': first_col_name, 
            'error': None
        }
        messages.append(f"{display_name_for_messages} 已添加到待处理列表。")
        any_operation_performed_successfully = True
    # --- End of inner function ---

    try:
        # Attempt to read as Excel (handles .xlsx, .xls)
        excel_file_io = io.BytesIO(file_content_bytes)
        excel_sheets_data = pd.read_excel(excel_file_io, sheet_name=None) # Read all sheets

        if isinstance(excel_sheets_data, pd.DataFrame): # Single-sheet Excel file
            _process_individual_df(df_to_process=excel_sheets_data, 
                                   key_for_tracking=original_file_name,
                                   display_name_for_messages=f"文件 '{original_file_name}' (单工作表)",
                                   sheet_name_to_store="Sheet1")
        
        elif isinstance(excel_sheets_data, dict): # Multi-sheet Excel file
            if not excel_sheets_data: # Excel file with no sheets
                messages.append(f"Excel文件 '{original_file_name}' 为空或不包含任何工作表。")
            else:
                for sheet_name, df_sheet in excel_sheets_data.items():
                    current_sheet_key = f"{original_file_name} - {sheet_name}"
                    _process_individual_df(df_to_process=df_sheet,
                                           key_for_tracking=current_sheet_key,
                                           display_name_for_messages=f"文件 '{original_file_name}' (工作表: {sheet_name})",
                                           sheet_name_to_store=sheet_name)
        else:
            messages.append(f"处理Excel文件 '{original_file_name}' 时返回了意外的数据结构。")

    except Exception as e_excel:
        try:
            csv_file_io = io.BytesIO(file_content_bytes)
            csv_file_io.seek(0) # Reset buffer position if excel reading attempt consumed it
            df_csv = pd.read_csv(csv_file_io)
            _process_individual_df(df_to_process=df_csv,
                                   key_for_tracking=original_file_name,
                                   display_name_for_messages=f"CSV文件 '{original_file_name}'",
                                   sheet_name_to_store=None)
        except Exception as e_csv:
            messages.append(f"无法将文件 '{original_file_name}' 解析为Excel或CSV。 Excel错误: {str(e_excel)}; CSV错误: {str(e_csv)}")

    if not any_operation_performed_successfully and not messages:
        messages.append(f"处理文件 '{original_file_name}' 未能成功或文件类型不受支持/为空。")

    return any_operation_performed_successfully, messages

def add_pending_file_to_staged_data(session_state: Dict[str, Any], pending_file_key: str, details: Dict[str, Any]) -> Tuple[bool, str]:
    """
    Processes a file from the pending list (after time column is selected) and adds it to the staged area.
    """
    df_preview = details.get('df_preview')
    time_col_to_process = details.get('selected_time_col') # This should now be the first column name or None
    original_filename = details.get('original_filename', pending_file_key)
    sheet_name = details.get('sheet_name')

    if df_preview is None:
        return False, f"错误: 文件 '{pending_file_key}' 没有可处理的预览数据。"

    if time_col_to_process is None:
        return False, f"错误: 文件 '{pending_file_key}' 为空或没有列，无法确定用作时间基准的第一列。"

    if time_col_to_process not in df_preview.columns:
        return False, f"错误: 预设的时间列 '{time_col_to_process}' (应为第一列) 在文件 '{pending_file_key}' 的数据中不存在。这可能是一个内部错误。"

    try:
        df_processed = df_preview.copy()
        
        # Convert to datetime, set index, sort
        df_processed[time_col_to_process] = pd.to_datetime(df_processed[time_col_to_process], errors='coerce')
        
        # Check for NaT values after conversion and remove rows with them
        if df_processed[time_col_to_process].isnull().any():
            original_rows = len(df_processed)
            df_processed.dropna(subset=[time_col_to_process], inplace=True)
            rows_after_dropna = len(df_processed)
            warning_msg = f"文件 '{pending_file_key}' 在其第一列 (用作时间列: '{time_col_to_process}') 中包含无效日期值，这些行 ({original_rows - rows_after_dropna}行) 已被移除。"
            if original_rows > rows_after_dropna:
                 st.warning(warning_msg)

        if df_processed.empty:
            return False, f"错误: 文件 '{pending_file_key}' 在处理其第一列 (用作时间列: '{time_col_to_process}') 后为空。请检查数据。"
        
        df_processed = df_processed.set_index(time_col_to_process)
        df_processed = df_processed.sort_index()

        # Infer frequency
        inferred_freq = None
        if isinstance(df_processed.index, pd.DatetimeIndex) and len(df_processed.index) > 1:
            inferred_freq = pd.infer_freq(df_processed.index)

        # --- Construct summary of the dataset --- 
        summary = {
            'rows': len(df_processed),
            'cols': len(df_processed.columns),
            'data_start_time': str(df_processed.index.min()) if not df_processed.empty else 'N/A',
            'data_end_time': str(df_processed.index.max()) if not df_processed.empty else 'N/A',
            'data_frequency': inferred_freq if inferred_freq else '未知',
            'first_col_display': df_processed.columns[0] if len(df_processed.columns) > 0 else 'N/A',
            'last_col_display': df_processed.columns[-1] if len(df_processed.columns) > 0 else 'N/A'
        }
        
        # Add to staged_data in session_state
        if GLOBAL_STAGED_DATA_KEY not in session_state:
            session_state[GLOBAL_STAGED_DATA_KEY] = {}
        
        current_staged_data = session_state[GLOBAL_STAGED_DATA_KEY]
        
        current_staged_data[pending_file_key] = {
            'df': df_processed,
            'original_filename': original_filename,
            'sheet_name': sheet_name,
            'time_col': time_col_to_process.strip() if isinstance(time_col_to_process, str) else time_col_to_process, # Ensure stripping whitespace
            'frequency': inferred_freq,
            'summary': summary,  # Add the summary here
            'source': 'data_comparison_module'
        }
        session_state[GLOBAL_STAGED_DATA_KEY] = current_staged_data

        # Remove from pending_uploads if it was there
        if PENDING_UPLOADS_DC_KEY in session_state and pending_file_key in session_state[PENDING_UPLOADS_DC_KEY]:
            del session_state[PENDING_UPLOADS_DC_KEY][pending_file_key]

        success_message = f"文件 '{original_filename}' (工作表: {sheet_name if sheet_name else 'N/A'}, 使用第一列 '{time_col_to_process}' 作为时间列) 已成功处理并添加到暂存区。"
        if sheet_name is None: # CSV file or single sheet Excel
             success_message = f"文件 '{original_filename}' (使用第一列 '{time_col_to_process}' 作为时间列) 已成功处理并添加到暂存区。"
        return True, success_message

    except Exception as e:
        error_message = f"处理文件 '{pending_file_key}' (时间列: '{time_col_to_process}') 时发生错误: {e}"
        # Also update the error in pending_uploads if it's still there (e.g., if it failed before removal)
        if PENDING_UPLOADS_DC_KEY in session_state and pending_file_key in session_state[PENDING_UPLOADS_DC_KEY]:
            session_state[PENDING_UPLOADS_DC_KEY][pending_file_key]['error'] = str(e)
        return False, error_message

def reset_all_pending_uploads_comparison(session_state: Dict[str, Any]) -> Tuple[bool, str]:
    """
    Resets all pending uploads for the data comparison module and updates the uploader's key.
    
    Args:
        session_state: Streamlit's session_state object.
    """
    session_state[PENDING_UPLOADS_DC_KEY] = {}
    session_state['dc_m1_uploader_key_suffix'] = session_state.get('dc_m1_uploader_key_suffix', 0) + 1
    return True, "所有待处理文件已重置。"

def compare_variables_in_dataset(
    df: pd.DataFrame, 
    base_variable_name: str, 
    comparison_variable_names: List[str]
) -> Dict[str, Dict[str, Any]]:
    """
    Compares a base variable against a list of other variables within the same DataFrame.

    Args:
        df: The DataFrame containing all variables. Assumes a DatetimeIndex.
        base_variable_name: The name of the column to be used as the base for comparison.
        comparison_variable_names: A list of column names to compare against the base variable.

    Returns:
        A dictionary where keys are the names of comparison variables.
        Each value is another dictionary with:
            - 'status': "完全相同" or "存在差异" or "错误".
            - 'differences_df': A DataFrame with (Index, Base_Value, Comparison_Value, Difference) 
                                for differing rows. None if '完全相同' or error.
            - 'message': A descriptive message.
    """
    results = {}

    if base_variable_name not in df.columns:
        # This case should ideally be caught by UI validation before calling
        results["_error_base_variable"] = {
            'status': "错误",
            'differences_df': None,
            'message': f"被比较变量 '{base_variable_name}' 不在数据集中。"
        }
        return results

    base_series = df[base_variable_name]

    for comp_var_name in comparison_variable_names:
        if comp_var_name not in df.columns:
            results[comp_var_name] = {
                'status': "错误",
                'differences_df': None,
                'message': f"比较变量 '{comp_var_name}' 不在数据集中。"
            }
            continue # Move to the next comparison variable

        comp_series = df[comp_var_name]
        
        # Attempt to convert to numeric for difference calculation, but keep original for display
        base_series_for_diff = pd.to_numeric(base_series, errors='coerce')
        comp_series_for_diff = pd.to_numeric(comp_series, errors='coerce')

        if base_series.equals(comp_series):
            results[comp_var_name] = {
                'status': "完全相同",
                'differences_df': None,
                'message': f"变量 '{comp_var_name}' 与 '{base_variable_name}' 的所有值完全相同。"
            }
        else:
            # Create a temporary DataFrame for finding differences
            temp_df = pd.DataFrame({
                f'{base_variable_name} (被比较)': base_series,
                f'{comp_var_name} (比较)': comp_series
            })
            
            # Calculate numeric difference if possible
            # Check if both series could be fully converted to numeric for meaningful subtraction
            if not base_series_for_diff.isna().all() and not comp_series_for_diff.isna().all():
                temp_df['差异'] = base_series_for_diff - comp_series_for_diff
            else:
                # If one or both are not fully numeric (e.g. object type, or all NaNs after coerce)
                temp_df['差异'] = "N/A (非数值或无法计算)"
            
            # Identify rows where original values are different
            # NaNs are tricky: (NaN == NaN) is False. For .equals(), two series with NaN at same spots are equal.
            # We need to find where they are visually different or one has value and other has NaN.
            # Correct mask: ((S1 != S2) & ~(S1.isnull() & S2.isnull())) | (S1.isnull() & ~S2.isnull()) | (~S1.isnull() & S2.isnull())
            mask_different_values = ((base_series != comp_series) & ~(base_series.isnull() & comp_series.isnull())) | (base_series.isnull() & ~comp_series.isnull()) | (~base_series.isnull() & comp_series.isnull())
            
            # Check if diff_mask leads to empty selection (e.g. only NaN differences not caught by simple !=)
            # This can happen if dtypes are object and NaNs are represented by different objects (None, np.nan)
            # or if .equals() is false due to dtype or other subtle reasons not captured by element-wise `!=` on raw values.
            # A robust way is to compare index-wise.
            diff_indices = base_series.index[mask_different_values]
            if diff_indices.empty and not base_series.empty:
                 # If .equals() is false but no differences found by `!=`, could be dtype or specific NaN issues.
                 # This path indicates a subtle difference detected by .equals() but not by the mask.
                 # For now, we'll provide a generic message for such cases.
                 # A more detailed diff here would require a row-by-row iteration or more complex comparison.
                pass # Will fall through to the message below

            differences_df = temp_df[mask_different_values].copy()
            
            # Rename columns for clarity in the output DataFrame
            differences_df.columns = ['被比较值', '比较值', '差异']

            results[comp_var_name] = {
                'status': "存在差异",
                'differences_df': differences_df,
                'message': f"变量 '{comp_var_name}' 与 '{base_variable_name}' 存在差异。详细差异见下表。"
            }
            
    return results

def compare_datasets_for_common_variables(
    datasets_data: Dict[str, pd.DataFrame] # Dict of dataset_name: dataframe
) -> Dict[str, Any]:
    """
    Compares selected datasets to find common variable names.

    Args:
        datasets_data: A dictionary where keys are dataset names (str)
                       and values are their pandas DataFrames.

    Returns:
        A dictionary containing:
        - 'status': str ("success", "error", "no_datasets", "insufficient_datasets")
        - 'message': str (A summary message for the comparison)
        - 'common_variables': Optional[List[str]] (List of common variable names)
        - 'variables_per_dataset': Optional[Dict[str, List[str]]] (Variables in each dataset)
        - 'compared_datasets': List[str] (Names of datasets that were compared)
        - 'value_comparison_results': Optional[Dict[str, str]] (Results of value comparison for common variables)
    """
    results: Dict[str, Any] = {
        'status': "error",
        'message': "比较过程中发生未知错误。",
        'common_variables': None,
        'variables_per_dataset': None,
        'compared_datasets': list(datasets_data.keys()),
        'value_comparison_results': None # New key for value comparison
    }

    if not datasets_data:
        results['status'] = "no_datasets"
        results['message'] = "没有选择任何数据集进行比较。"
        return results

    dataset_names = list(datasets_data.keys())
    if len(dataset_names) < 2:
        results['status'] = "insufficient_datasets"
        results['message'] = "请至少选择两个数据集进行比较。"
        return results

    variables_per_dataset: Dict[str, List[str]] = {}
    list_of_sets_of_variables: List[set[str]] = []

    for name, df in datasets_data.items():
        if not isinstance(df, pd.DataFrame):
            results['message'] = f"数据集 '{name}' 不是有效的 DataFrame。"
            return results
        current_vars = list(df.columns)
        variables_per_dataset[name] = sorted(current_vars) # Store sorted list
        list_of_sets_of_variables.append(set(current_vars))

    if not list_of_sets_of_variables: # Should not happen if len(dataset_names) >= 2 and DFs are valid
        results['message'] = "无法从选定数据集中提取变量列表。"
        return results

    # Find intersection of all sets of variables
    common_variables_set = list_of_sets_of_variables[0].copy() # Start with the first set
    for i in range(1, len(list_of_sets_of_variables)):
        common_variables_set.intersection_update(list_of_sets_of_variables[i])
    
    common_variables_list = sorted(list(common_variables_set))

    results['status'] = "success"
    results['common_variables'] = common_variables_list
    results['variables_per_dataset'] = variables_per_dataset
    
    # --- Perform value comparison for common variables ---
    value_comparison_output: Dict[str, Dict[str, Any]] = {} 
    aggregated_original_index_discrepancies: List[Dict[str, Any]] = [] # New list for aggregated discrepancies
    processed_original_length_pairs: set = set() # To track processed (ds1, ds2) pairs for original length

    if common_variables_list:
        dataset_name_list = list(datasets_data.keys())
        # First, iterate through dataset pairs to find original index discrepancies once per pair
        if len(dataset_name_list) >= 2:
            for i in range(len(dataset_name_list)):
                for j in range(i + 1, len(dataset_name_list)):
                    ds1_name = dataset_name_list[i]
                    ds2_name = dataset_name_list[j]
                    pair_key = frozenset({ds1_name, ds2_name})

                    if pair_key in processed_original_length_pairs:
                        continue # Already processed this pair

                    # Check if any common variable shows original length mismatch between this pair
                    # We only need one such variable to determine index diff for the pair
                    found_discrepancy_for_pair = False
                    temp_ref_indices_orig = None
                    temp_comp_indices_orig = None
                    
                    # Try to find one common variable to get indices. This assumes all common vars in a dataset have same original index.
                    # A more robust (but potentially slower) way would be to compare full dataset indices if no common vars exist or if they differ internally.
                    # For now, let's assume common_variables_list is representative if not empty.
                    # If common_variables_list is empty but datasets exist, this check might not be triggered as desired.
                    # However, original length check is primarily for *common* variables.

                    # Get the full indices of the two datasets
                    ds1_indices = set(datasets_data[ds1_name].index)
                    ds2_indices = set(datasets_data[ds2_name].index)

                    if len(ds1_indices) != len(ds2_indices) or ds1_indices != ds2_indices:
                        found_discrepancy_for_pair = True
                        temp_ref_indices_orig = ds1_indices
                        temp_comp_indices_orig = ds2_indices
                    
                    if found_discrepancy_for_pair and temp_ref_indices_orig is not None and temp_comp_indices_orig is not None:
                        ref_unique_orig = sorted(list(temp_ref_indices_orig - temp_comp_indices_orig))
                        comp_unique_orig = sorted(list(temp_comp_indices_orig - temp_ref_indices_orig))

                        if ref_unique_orig or comp_unique_orig: # Only add if there's actual difference
                            # Format timestamps to 'YYYY-MM-DD' strings
                            ref_unique_orig_str = [d.strftime('%Y-%m-%d') for d in ref_unique_orig]
                            comp_unique_orig_str = [d.strftime('%Y-%m-%d') for d in comp_unique_orig]

                            aggregated_original_index_discrepancies.append({
                                'dataset1_name': ds1_name,
                                'dataset2_name': ds2_name,
                                'dataset1_unique_indices_original': ref_unique_orig_str,
                                'dataset2_unique_indices_original': comp_unique_orig_str,
                                'dataset1_original_length': len(ds1_indices),
                                'dataset2_original_length': len(ds2_indices)
                            })
                        processed_original_length_pairs.add(pair_key)

        # Then, process common variables for other types of mismatches
        for cv_name in common_variables_list:
            current_var_comparison_result: Dict[str, Any] = { # Default/fallback structure
                'status': 'error', 
                'type': 'unknown_backend_state', 
                'message': 'Unknown error during value comparison for this variable.', 
                'details_df': None, 
                'diff_count': 0,
                'length_diff_details': None
            }
            try:
                all_series_for_cv = [(ds_name, datasets_data[ds_name][cv_name].copy()) for ds_name in dataset_name_list if cv_name in datasets_data[ds_name]]
                
                original_lengths_for_cv_consistent = True
                if all_series_for_cv:
                    first_len = len(all_series_for_cv[0][1]) # Length of the actual series data
                    for _, series_data in all_series_for_cv[1:]:
                        if len(series_data) != first_len:
                            original_lengths_for_cv_consistent = False
                            break
                else: # Should not happen if cv_name is in common_variables_list from valid datasets
                    original_lengths_for_cv_consistent = False 

                if not original_lengths_for_cv_consistent:
                    current_var_comparison_result = {
                        'status': 'original_length_mismatch',
                        'type': 'original_length_mismatch',
                        'message': '此变量在各数据集中原始长度/索引不一致。由于此基础结构差异，未进行后续的类型或逐值比较。详细的索引差异通常在“原始数据长度/索引差异总结”部分单独报告。',
                        'details_df': None, 
                        'diff_count': 0,
                        'length_diff_details': None # Details are in aggregated_original_index_discrepancies
                    }
                else: # Original lengths are consistent, proceed with type and value checks
                    series_objects = [s_tuple[1] for s_tuple in all_series_for_cv]
                    if len(set(s.dtype for s in series_objects)) > 1:
                        type_details = {ds_name: str(series_obj.dtype) for ds_name, series_obj in all_series_for_cv}
                        current_var_comparison_result = {
                            'status': 'type_mismatch', 
                            'type': 'type_mismatch',
                            'message': f"数据类型不一致。详情: {type_details}", 
                            'details_df': None, 
                            'diff_count': 0
                        }
                    else:
                        reference_dataset_name, reference_series_orig = all_series_for_cv[0]
                        reference_series_for_comp = reference_series_orig.dropna()
                        all_match_reference = True
                        final_comparison_message = ""
                        final_details_df = None
                        final_diff_count = 0
                        final_status = 'identical' 
                        final_type = 'identical' 
                        final_length_diff_details = None

                        for i in range(1, len(all_series_for_cv)):
                            current_dataset_name, current_series_orig = all_series_for_cv[i]
                            current_series_for_comp = current_series_orig.dropna()
                            
                            if len(reference_series_for_comp) != len(current_series_for_comp):
                                all_match_reference = False
                                final_status = 'length_mismatch' 
                                final_type = 'length_mismatch'   
                                
                                ref_index_set = set(reference_series_for_comp.index)
                                comp_index_set = set(current_series_for_comp.index)
                                ref_unique_indices = list(ref_index_set - comp_index_set)
                                comp_unique_indices = list(comp_index_set - ref_index_set)
                                ref_unique_rows = {idx: reference_series_for_comp.loc[idx] for idx in ref_unique_indices}
                                comp_unique_rows = {idx: current_series_for_comp.loc[idx] for idx in comp_unique_indices}

                                final_length_diff_details = {
                                    'reference_dataset_name': reference_dataset_name,
                                    'current_dataset_name': current_dataset_name,
                                    'ref_len_after_dropna': len(reference_series_for_comp),
                                    'comp_len_after_dropna': len(current_series_for_comp),
                                    'ref_unique_rows': ref_unique_rows,
                                    'comp_unique_rows': comp_unique_rows
                                }
                                final_length_diff_details = _serialize_complex_object(final_length_diff_details)
                                final_comparison_message = f"剔除NaN后长度不一致: '{reference_dataset_name}' (长度 {len(reference_series_for_comp)}) 与 '{current_dataset_name}' (长度 {len(current_series_for_comp)}) 对变量 '{cv_name}' 的非空值长度不同。"
                                break # Stop checking this cv_name against other datasets

                            # If lengths after dropna are same, proceed to check values
                            if not current_series_for_comp.equals(reference_series_for_comp):
                                all_match_reference = False
                                final_status = 'different_values' 
                                final_type = 'different_values'   
                                
                                common_index = reference_series_for_comp.index.intersection(current_series_for_comp.index)
                                ref_aligned = reference_series_for_comp.loc[common_index]
                                comp_aligned = current_series_for_comp.loc[common_index]
                                diff_mask_val = ((ref_aligned != comp_aligned) & ~(ref_aligned.isnull() & comp_aligned.isnull())) | \
                                              (ref_aligned.isnull() & ~comp_aligned.isnull()) | (~ref_aligned.isnull() & comp_aligned.isnull())
                                diff_indices_val = ref_aligned.index[diff_mask_val]
                                temp_df_val = pd.DataFrame({
                                    f'{reference_dataset_name.replace("::", "_")} (基准)': ref_aligned[diff_mask_val],
                                    f'{current_dataset_name.replace("::", "_")} (比较)': comp_aligned[diff_mask_val]
                                })
                                temp_df_val.index.name = 'Index/日期'
                                final_details_df = temp_df_val
                                final_diff_count = len(final_details_df) if final_details_df is not None else 0
                                message_detail_val = f"数值差异: '{reference_dataset_name}' 与 '{current_dataset_name}' 对变量 '{cv_name}' 在剔除NaN且长度一致后，存在数值差异。"
                                if final_diff_count > 0:
                                    message_detail_val += f" 共发现 {final_diff_count} 条观测值不同。"
                                else:
                                    message_detail_val += " .equals() 为False但未能通过掩码定位具体数值差异，可能源于数据类型或特殊NaN值。"
                                final_comparison_message = message_detail_val
                                break # Stop checking this cv_name against other datasets
                        
                        if all_match_reference:
                            current_var_comparison_result = {
                                'status': 'identical', 
                                'type': 'identical',
                                'message': '在所有选定数据集中，剔除NaN后数值完全相同。', 
                                'details_df': None, 
                                'diff_count': 0
                            }
                        else:
                            current_var_comparison_result = {
                                'status': final_status,
                                'type': final_type,
                                'message': final_comparison_message,
                                'details_df': final_details_df,
                                'diff_count': final_diff_count,
                                'length_diff_details': final_length_diff_details
                            }
            
            except Exception as e:
                current_var_comparison_result = {
                    'status': 'error', 
                    'type': 'exception_during_comparison', 
                    'message': f"对变量 '{cv_name}' 进行值比较时发生后端错误: {str(e)[:150]}...", 
                    'details_df': None, 
                    'diff_count': 0
                }
            
            value_comparison_output[cv_name] = _serialize_complex_object(current_var_comparison_result)
    
    results['value_comparison_results'] = value_comparison_output
    # Serialize aggregated_original_index_discrepancies as well
    results['aggregated_original_index_discrepancies'] = _serialize_complex_object(aggregated_original_index_discrepancies)
    
    if not common_variables_list and len(datasets_data) >= 2:
        results['message'] = f"在 {len(dataset_names)} 个数据集中未找到共同变量。"
        
    # --- Find variables with different names but same values ---
    same_value_different_names_groups = []
    all_series_infos = [] # list of dicts: {'dataset_name': ..., 'variable_name': ..., 'series': ...}
    for ds_name_key, df_data in datasets_data.items():
        for var_name_col in df_data.columns:
            all_series_infos.append({
                "dataset_name": ds_name_key,
                "variable_name": var_name_col,
                "series": df_data[var_name_col]
            })

    # Group series that are equal by iterating and checking .equals()
    # This list will store groups (lists) of series_info dicts that are identical to each other.
    grouped_identical_series_infos = []
    processed_indices = [False] * len(all_series_infos)

    for i in range(len(all_series_infos)):
        if processed_indices[i]:
            continue
        
        current_series_info = all_series_infos[i]
        current_group = [current_series_info]
        processed_indices[i] = True
        
        for j in range(i + 1, len(all_series_infos)):
            if processed_indices[j]:
                continue
            
            other_series_info = all_series_infos[j]
            try:
                if current_series_info["series"].equals(other_series_info["series"]):
                    current_group.append(other_series_info)
                    processed_indices[j] = True
            except Exception: # Broad exception for safety during comparison
                # If comparison fails, assume they are not equal for this purpose.
                pass
        
        if len(current_group) > 0: # Should always be at least 1 (itself)
             grouped_identical_series_infos.append(current_group)

    # From these groups, filter for those that have different variable names
    for group in grouped_identical_series_infos:
        if len(group) > 1: # Need at least two series in a group to check for different names
            var_names_in_group = set()
            for item in group:
                var_names_in_group.add(item['variable_name'])
            
            # Only consider this group if there's more than one unique variable name involved
            # OR if there's only one unique variable name BUT it appears in more than one dataset (already covered by common var check)
            # The specific request is "variable names different but values same"
            if len(var_names_in_group) > 1:
                representative_series = group[0]['series'] 
                # Ensure preview head is JSON serializable (convert from potential Series/Timestamp to basic types)
                try:
                    head_data = representative_series.head(3).to_dict()
                    # Convert Timestamps in dict values to string if they exist
                    for k, v in head_data.items():
                        if pd.api.types.is_datetime64_any_dtype(v) or isinstance(v, pd.Timestamp):
                            head_data[k] = str(v)
                        elif pd.isna(v):
                             head_data[k] = None # Explicitly set NaN to None for JSON
                except Exception:
                    head_data = {}

                preview_data = {
                    "length": len(representative_series),
                    "dtype": str(representative_series.dtype),
                    "head": head_data
                }

                same_value_different_names_groups.append({
                    "members": [{'dataset_name': item['dataset_name'], 'variable_name': item['variable_name']} for item in group],
                    "preview": preview_data
                })

    results['same_value_different_names_analysis'] = same_value_different_names_groups
    # --- End of find variables with different names but same values ---

    if common_variables_list:
        results['message'] = f"在 {len(dataset_names)} 个数据集中找到了 {len(common_variables_list)} 个共同变量。"
    else:
        results['message'] = f"在 {len(dataset_names)} 个数据集中未找到共同变量。"
        
    return results

def update_variable_in_staged_data(session_state: Dict[str, Any], target_dataset_name: str, source_dataset_name: str, variable_name: str, update_mode: str, specific_dates: Optional[List[Any]] = None):
    """
    Updates a variable in the target dataset using data from the source dataset based on the specified mode.
    Returns a tuple: (success_boolean, message_string, changes_dataframe_or_None)
    changes_dataframe will have columns: ['Date', 'Original Value', 'New Value']
    """
    staged_data = session_state.get('staged_data', {})
    changes_made_list = []

    if target_dataset_name not in staged_data or source_dataset_name not in staged_data:
        return False, "源数据集或目标数据集未在暂存区找到。", None

    target_df_container = staged_data[target_dataset_name]
    source_df_container = staged_data[source_dataset_name]

    if 'df' not in target_df_container or not isinstance(target_df_container['df'], pd.DataFrame):
        return False, f"目标数据集 '{target_dataset_name}' 未包含有效的数据表。", None
    if 'df' not in source_df_container or not isinstance(source_df_container['df'], pd.DataFrame):
        return False, f"源数据集 '{source_dataset_name}' 未包含有效的数据表。", None

    target_df = target_df_container['df'].copy() # Work on a copy
    source_df = source_df_container['df']

    if variable_name not in target_df.columns:
        return False, f"变量 '{variable_name}' 在目标数据集 '{target_dataset_name}' 中未找到。", None
    if variable_name not in source_df.columns:
        return False, f"变量 '{variable_name}' 在源数据集 '{source_dataset_name}' 中未找到。", None

    # Ensure indices are compatible for alignment (datetime if possible)
    try:
        target_df.index = pd.to_datetime(target_df.index)
        source_df.index = pd.to_datetime(source_df.index)
    except Exception as e:
        # If conversion fails, proceed with original indices but warn or restrict some operations?
        # For now, let's assume indices are compatible or user is aware.
        pass 

    original_target_series = target_df[variable_name].copy()
    target_series = target_df[variable_name] # This is a view, changes will reflect in target_df
    source_series = source_df[variable_name]

    common_index = target_series.index.intersection(source_series.index)
    
    if common_index.empty:
        return False, "源数据和目标数据没有共同的索引（日期），无法进行更新。", None

    if update_mode == 'fill_missing':
        fill_values = source_series.loc[common_index]
        target_subset_for_filling = target_series.loc[common_index]
        
        # Identify where target is NaN and source is not NaN
        condition = target_subset_for_filling.isnull() & fill_values.notnull()
        actual_fill_indices = target_subset_for_filling[condition].index
        
        if not actual_fill_indices.empty:
            for idx in actual_fill_indices:
                old_val = original_target_series.loc[idx]
                new_val = fill_values.loc[idx]
                if old_val != new_val: # Should always be true if old_val is NaN
                    changes_made_list.append({'Date': idx, 'Original Value': old_val, 'New Value': new_val})
            target_series.loc[actual_fill_indices] = fill_values.loc[actual_fill_indices]
        
    elif update_mode == 'replace_all':
        # Replace only on common_index where source is not NaN
        source_on_common = source_series.loc[common_index].dropna()
        update_indices = source_on_common.index

        if not update_indices.empty:
            for idx in update_indices:
                old_val = original_target_series.loc[idx]
                new_val = source_on_common.loc[idx]
                if pd.isna(old_val) or old_val != new_val: # Record if old was NaN or different
                    changes_made_list.append({'Date': idx, 'Original Value': old_val, 'New Value': new_val})
            target_series.loc[update_indices] = source_on_common

    elif update_mode == 'replace_specific_dates':
        if not specific_dates or not isinstance(specific_dates, list):
            return False, "按日期替换模式需要一个有效的日期列表。", None
        
        try:
            # Ensure specific_dates are in the same format/type as the DataFrame index
            specific_dates_idx = pd.to_datetime(specific_dates)
        except Exception as e:
            return False, f"提供的特定日期格式无效: {e}", None

        # Filter specific_dates to those that are also in common_index and where source has non-NaN values
        valid_update_dates = specific_dates_idx.intersection(common_index)
        source_on_valid_dates = source_series.loc[valid_update_dates].dropna()
        actual_update_indices = source_on_valid_dates.index
        
        if not actual_update_indices.empty:
            for idx in actual_update_indices:
                old_val = original_target_series.loc[idx]
                new_val = source_on_valid_dates.loc[idx]
                if pd.isna(old_val) or old_val != new_val:
                    changes_made_list.append({'Date': idx, 'Original Value': old_val, 'New Value': new_val})
            target_series.loc[actual_update_indices] = source_on_valid_dates
        else:
            # This case means either no specific dates were in common index, or source had NaNs on those dates
            # No changes_made_list entries would be added in this path naturally.
            pass 

    else:
        return False, f"未知的更新模式: {update_mode}", None

    # Commit changes to session_state
    staged_data[target_dataset_name]['df'] = target_df
    session_state['staged_data'] = staged_data

    changes_df = pd.DataFrame(changes_made_list)
    if not changes_df.empty:
        changes_df = changes_df.sort_values(by='Date').reset_index(drop=True)
        # Format 'Date' for better display if it's datetime
        if pd.api.types.is_datetime64_any_dtype(changes_df['Date']):
            changes_df['Date'] = changes_df['Date'].dt.strftime('%Y-%m-%d %H:%M:%S') # Or '%Y-%m-%d' if no time part

    num_changes = len(changes_df)
    if num_changes > 0:
        msg = f"成功更新变量 '{variable_name}' 在数据集 '{target_dataset_name}' 中的 {num_changes} 个值。"
        return True, msg, changes_df
    else:
        msg = f"操作完成。变量 '{variable_name}' 在数据集 '{target_dataset_name}' 中未发生实际更改（例如，没有符合条件的值，或源值与目标值已相同）。"
        return True, msg, pd.DataFrame(columns=['Date', 'Original Value', 'New Value'])

def make_staged_data_copy(session_state: Dict[str, Any], source_dataset_name: str, new_dataset_name: str) -> Tuple[bool, str]:
    """
    Creates a deep copy of a dataset in the staged_data with a new name.

    Args:
        session_state: The Streamlit session state.
        source_dataset_name: The name of the source dataset in 'staged_data'.
        new_dataset_name: The desired name for the copied dataset.

    Returns:
        A tuple (success_status, message).
    """
    staged_data = session_state.get('staged_data', {})

    if source_dataset_name not in staged_data:
        return False, f"错误：源数据集 '{source_dataset_name}' 未在暂存区找到。"
    
    if not new_dataset_name.strip():
        return False, "错误：副本名称不能为空。"
    
    if new_dataset_name == source_dataset_name:
        return False, "错误：副本名称不能与原名称相同。"
    
    if new_dataset_name in staged_data:
        return False, f"错误：名称 '{new_dataset_name}' 已在暂存区存在。请输入一个唯一的名称。"

    try:
        # Perform a deep copy to ensure the DataFrame and other metadata are fully independent
        staged_data[new_dataset_name] = copy.deepcopy(staged_data[source_dataset_name])
        
        # Optionally, update any internal 'name' or 'id' fields within the copied dataset's dict
        # For example, if the dict structure is {'name': ..., 'df': ..., 'uploaded_file_info': ...}
        # you might want to update staged_data[new_dataset_name]['name'] = new_dataset_name
        # This depends on how other parts of the application use these internal fields.
        # For now, we assume the primary identifier is the key in the staged_data dictionary.
        # If 'uploaded_file_info' exists and contains 'name', update it too for consistency if it represents the dataset identifier.
        if 'uploaded_file_info' in staged_data[new_dataset_name] and isinstance(staged_data[new_dataset_name]['uploaded_file_info'], dict):
            staged_data[new_dataset_name]['uploaded_file_info']['name'] = new_dataset_name # Update if this field is used as an ID

        session_state['staged_data'] = staged_data
        return True, f"成功从 '{source_dataset_name}' 创建副本 '{new_dataset_name}'。"
    except Exception as e:
        return False, f"创建副本时发生错误: {e}"