import pandas as pd
import numpy as np
from datetime import datetime # Keep for type hinting if used
from dateutil.relativedelta import relativedelta
from collections import defaultdict

def _calculate_single_win_rate(target_series_data: pd.Series, ref_series_data: pd.Series) -> tuple[str | float, str]:
    """计算单个参考序列相对于目标序列的胜率。"""
    if target_series_data.empty or ref_series_data.empty:
        return "N/A", "输入序列为空"
    
    target_series_numeric = pd.to_numeric(target_series_data, errors='coerce')
    ref_series_numeric = pd.to_numeric(ref_series_data, errors='coerce')

    # Check for all NaNs after numeric conversion or if too few valid points
    target_valid = target_series_numeric.dropna()
    ref_valid = ref_series_numeric.dropna()

    if target_valid.empty or ref_valid.empty:
        return "N/A", "序列转换后无有效数值"
    if len(target_valid) < 2 or len(ref_valid) < 2:
        return "N/A", "有效数据点不足 (少于2个)"

    try:
        target_diff = target_valid.diff().iloc[1:] # Use valid data for diff
        ref_diff = ref_valid.diff().iloc[1:]     # Use valid data for diff
    except Exception as e: 
        return "N/A", f"计算diff时出错: {str(e)[:50]}"

    if target_diff.empty or ref_diff.empty or target_diff.isnull().all() or ref_diff.isnull().all():
        return "N/A", "计算变化量后序列为空或全为NaN"

    # Align based on common valid indices *after* diff
    common_index = target_diff.index.intersection(ref_diff.index)
    if common_index.empty:
        return "N/A", "序列变化量无共同索引周期"
        
    aligned_target_diff = target_diff.loc[common_index].dropna() # Drop NaNs that might arise from non-overlapping original NaNs
    aligned_ref_diff = ref_diff.loc[common_index].dropna()
    
    # Re-align after dropna on each aligned diff series
    final_common_index = aligned_target_diff.index.intersection(aligned_ref_diff.index)
    if final_common_index.empty:
        return "N/A", "对齐并移除NaN后无共同有效数据点"

    aligned_target_diff = aligned_target_diff.loc[final_common_index]
    aligned_ref_diff = aligned_ref_diff.loc[final_common_index]

    if aligned_target_diff.empty: # Final check after all alignments and NaN drops
        return "N/A", "最终对齐后无数据点"

    target_changed_mask = (aligned_target_diff != 0) 
    num_target_changes = target_changed_mask.sum()

    if num_target_changes == 0:
        # If remark needs more detail, it can be constructed here.
        return "N/A (目标无变化)", f"基于 {len(aligned_target_diff)} 个共同周期, 目标无变化"

    target_diff_when_changed = aligned_target_diff[target_changed_mask]
    ref_diff_when_target_changed = aligned_ref_diff[target_changed_mask]

    # Ensure same length for comparison (should be guaranteed by masks)
    if len(target_diff_when_changed) != len(ref_diff_when_target_changed):
        # This case should ideally not be reached if logic is correct
        return "N/A", "内部错误: 变化序列长度不匹配"

    same_direction = np.logical_or(
        np.logical_and(target_diff_when_changed > 0, ref_diff_when_target_changed > 0),
        np.logical_and(target_diff_when_changed < 0, ref_diff_when_target_changed < 0)
    )
    win_rate_val = (same_direction.sum() / num_target_changes) * 100
    remark = f"基于 {num_target_changes} 个目标变化周期 (共 {len(aligned_target_diff)} 周期)"
    return win_rate_val, remark

def perform_batch_win_rate_calculation(
    df_input: pd.DataFrame, 
    target_series_name: str, 
    ref_series_names_list: list, 
    selected_time_ranges: list,
    is_datetime_index_available: bool,
    get_current_time_for_filter: callable # Pass function to get current time
    ):
    """
    Performs batch win rate calculations.
    Returns: tuple: (defaultdict: results_accumulator, list: error_messages, list: warning_messages)
    """
    df_original = df_input.copy()
    results_accumulator = defaultdict(dict)
    error_messages = []
    warning_messages = []

    if not target_series_name:
        error_messages.append("目标序列未选择。")
        return results_accumulator, error_messages, warning_messages
    if not ref_series_names_list:
        warning_messages.append("没有选择任何参考序列。")
        return results_accumulator, error_messages, warning_messages
    if not selected_time_ranges:
        warning_messages.append("没有选择任何时间范围。")
        return results_accumulator, error_messages, warning_messages

    if target_series_name not in df_original.columns:
        error_messages.append(f"目标序列 '{target_series_name}' 在数据中未找到。")
        return results_accumulator, error_messages, warning_messages

    # Initial check on the full target series data
    base_target_series_check = pd.to_numeric(df_original[target_series_name], errors='coerce').dropna()
    if len(base_target_series_check) < 2:
        warning_messages.append(f"目标序列 '{target_series_name}' 的全局有效数据点不足 (少于2个)，所有胜率可能为N/A。")

    for ref_s_name in ref_series_names_list:
        if ref_s_name not in df_original.columns:
            warning_messages.append(f"参考序列 '{ref_s_name}' 在数据中未找到，已跳过。")
            for time_range_key in selected_time_ranges:
                results_accumulator[ref_s_name][time_range_key] = "N/A (序列不存在)"
            continue
        
        base_ref_series_check = pd.to_numeric(df_original[ref_s_name], errors='coerce').dropna()
        if len(base_ref_series_check) < 2:
            warning_messages.append(f"参考序列 '{ref_s_name}' 的全局有效数据点不足 (少于2个)。")
            # Fill N/A for this ref series for all time ranges and continue
            for time_range_key in selected_time_ranges:
                results_accumulator[ref_s_name][time_range_key] = "N/A (参考序列数据不足)"
            continue

        for time_range in selected_time_ranges:
            df_for_range_calc = df_original.copy()
            time_range_label_for_na = f"(@ {time_range})"
            
            start_date_filter = None
            if time_range != "全部时间":
                if is_datetime_index_available and isinstance(df_for_range_calc.index, pd.DatetimeIndex):
                    current_time = get_current_time_for_filter() # Get current time from passed function
                    if time_range == "近半年": start_date_filter = current_time - relativedelta(months=6)
                    elif time_range == "近1年": start_date_filter = current_time - relativedelta(years=1)
                    elif time_range == "近3年": start_date_filter = current_time - relativedelta(years=3)
                    
                    if start_date_filter:
                        df_for_range_calc = df_for_range_calc[df_for_range_calc.index >= start_date_filter]
                else:
                    results_accumulator[ref_s_name][time_range] = f"N/A (无法按'{time_range}'筛选)"
                    warning_messages.append(f"无法为 '{ref_s_name}' 应用时间范围 '{time_range}'，因无有效日期时间索引。")
                    continue # Skip to next time range

            if df_for_range_calc.empty:
                results_accumulator[ref_s_name][time_range] = f"N/A (无数据 {time_range_label_for_na})"
                continue

            current_target_data = df_for_range_calc.get(target_series_name, pd.Series(dtype=float))
            current_ref_data = df_for_range_calc.get(ref_s_name, pd.Series(dtype=float))

            win_rate_val, remark = _calculate_single_win_rate(current_target_data, current_ref_data)
            
            final_display_value = ""
            if isinstance(win_rate_val, float):
                final_display_value = f"{win_rate_val:.2f}% ({remark})"
            else: # win_rate_val is a string like "N/A" or "N/A (...)"
                # Append remark if it provides more context than the base N/A message
                if remark and remark not in win_rate_val: 
                    final_display_value = f"{win_rate_val} ({remark})"
                else:
                    final_display_value = str(win_rate_val)
            
            results_accumulator[ref_s_name][time_range] = final_display_value
            
    return results_accumulator, error_messages, warning_messages 