import pandas as pd
import numpy as np
from fastdtw import fastdtw

# --- Core DTW Calculation Function ---
def perform_batch_dtw_calculation(
    df_input: pd.DataFrame, 
    target_series_name: str, 
    comparison_series_names: list, 
    window_type_param: str, 
    window_size_param: int | None, 
    dist_metric_name_param: str,
    dist_metric_display_param: str # For reporting
    ):
    """
    Performs batch DTW calculations between a target series and multiple comparison series.

    Args:
        df_input (pd.DataFrame): The input DataFrame containing the time series data.
        target_series_name (str): The name of the target series column.
        comparison_series_names (list): A list of names of the comparison series columns.
        window_type_param (str): The DTW window type ("无限制" or "固定大小窗口 (Sakoe-Chiba Band)").
        window_size_param (int | None): The window size if using a fixed window, otherwise None.
        dist_metric_name_param (str): The internal name of the distance metric (e.g., "euclidean").
        dist_metric_display_param (str): The display name of the distance metric for reports.

    Returns:
        tuple: (list_of_results, dict_of_paths, list_of_errors, list_of_warnings)
               list_of_results contains dictionaries dificuldades for each comparison.
               dict_of_paths contains the path data for successful DTW computations.
               list_of_errors contains error messages encountered.
               list_of_warnings contains warning messages.
    """
    df = df_input.copy()
    results_list_temp = []
    paths_dict_temp = {}
    error_messages = []
    warning_messages = []

    if not target_series_name:
        error_messages.append("目标变量未选择。")
        return results_list_temp, paths_dict_temp, error_messages, warning_messages
    
    if not comparison_series_names:
        error_messages.append("对比变量未选择。")
        return results_list_temp, paths_dict_temp, error_messages, warning_messages

    s_target_data_numeric = pd.to_numeric(df[target_series_name], errors='coerce')
    s_target_data_clean = s_target_data_numeric.dropna()

    if s_target_data_clean.empty:
        error_messages.append(f"目标序列 '{target_series_name}' 无有效数据 (移除NaN后为空)。")
        return results_list_temp, paths_dict_temp, error_messages, warning_messages

    s_target_np = s_target_data_clean.to_numpy()
    dist_funcs = {
        "euclidean": lambda a, b: abs(a - b),
        "manhattan": lambda a, b: abs(a - b),
        "sqeuclidean": lambda a, b: (a - b)**2
    }
    selected_dist_func = dist_funcs.get(dist_metric_name_param)
    if selected_dist_func is None: # Should not happen if UI maps correctly
        error_messages.append(f"内部错误：未知的距离度量 '{dist_metric_name_param}'。")
        # Default to euclidean to prevent crash, but this indicates a bug
        selected_dist_func = dist_funcs["euclidean"]


    for compare_series_name in comparison_series_names:
        s_compare_data_numeric = pd.to_numeric(df[compare_series_name], errors='coerce')
        s_compare_data_clean = s_compare_data_numeric.dropna()
        
        current_result_entry = {
            '目标变量': target_series_name,
            '对比变量': compare_series_name,
            'DTW距离': np.nan,
            '原因': '-',
            '窗口类型': window_type_param,
            '窗口大小': window_size_param if window_type_param == "固定大小窗口 (Sakoe-Chiba Band)" else 'N/A',
            '距离度量': dist_metric_display_param
        }

        if s_compare_data_clean.empty:
            warning_messages.append(f"对比序列 '{compare_series_name}' 无有效数据，已跳过。")
            current_result_entry['原因'] = '对比序列为空'
            results_list_temp.append(current_result_entry)
            continue
        
        s_compare_np = s_compare_data_clean.to_numpy()
        
        dtw_radius_calc = 1 # Default radius
        if window_type_param == "固定大小窗口 (Sakoe-Chiba Band)":
            if window_size_param is not None and window_size_param > 0:
                dtw_radius_calc = int(window_size_param)
            else: # Default if somehow window_size_param is invalid
                dtw_radius_calc = 10 
                warning_messages.append(f"窗口大小参数无效 ({window_size_param})，已默认为10。")
        else: # "无限制"
            dtw_radius_calc = max(len(s_target_np), len(s_compare_np)) # Effectively no radius limit for fastdtw
            if dtw_radius_calc == 0 : dtw_radius_calc = 1 # if both series are somehow empty but not caught

        try:
            distance, path = fastdtw(s_target_np, s_compare_np, radius=dtw_radius_calc, dist=selected_dist_func)
            current_result_entry['DTW距离'] = distance
            paths_dict_temp[compare_series_name] = {
                'target_np': s_target_np,
                'compare_np': s_compare_np,
                'path': path
            }
        except Exception as e:
            error_msg = f"计算 '{target_series_name}' vs '{compare_series_name}' DTW时出错: {str(e)[:200]}"
            error_messages.append(error_msg)
            current_result_entry['原因'] = f'计算错误: {str(e)[:100]}'
        
        results_list_temp.append(current_result_entry)
            
    return results_list_temp, paths_dict_temp, error_messages, warning_messages 