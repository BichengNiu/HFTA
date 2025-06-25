import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.dates as mdates # Added for date formatting
import altair as alt

# 🔥 修复：延迟字体设置，避免模块导入时的 ScriptRunContext 警告
def _setup_matplotlib_font():
    """延迟设置 Matplotlib 中文字体"""
    try:
        plt.rcParams['font.sans-serif'] = ['SimHei']  # Default to SimHei
        plt.rcParams['axes.unicode_minus'] = False # Ensure minus sign displays correctly
        # print("[Matplotlib Font Setup] Successfully set font to SimHei.")  # 移除打印减少日志
    except Exception:
        try:
            plt.rcParams['font.sans-serif'] = ['Microsoft YaHei'] # Fallback to Microsoft YaHei
            plt.rcParams['axes.unicode_minus'] = False
            # print("[Matplotlib Font Setup] Successfully set font to Microsoft YaHei.")  # 移除打印减少日志
        except Exception:
            pass  # 静默处理，避免日志污染

# Extracted from time_series_clean_utils.py
def analyze_missing_data(df, selected_cols):
    """
    Analyzes missing data for selected columns in a DataFrame.
    
    Args:
        df (pd.DataFrame): The input DataFrame containing the data.
        selected_cols (list): List of column names to analyze.
        
    Returns:
        list: List of dictionaries containing missing data analysis results for each column.
    """
    results = []
    if df.empty:
        return results
    
    # 检查是否有时间列信息
    time_col = None
    for col in df.columns:
        if pd.api.types.is_datetime64_any_dtype(df[col]) or pd.api.types.is_period_dtype(df[col]):
            time_col = col
            break
    
    for col_name in selected_cols:
        if col_name not in df.columns:
            results.append({
                'column': col_name,
                'total_missing_pct': '列不存在',
                'post_first_record_missing_pct': '列不存在',
                'max_consecutive_missing': '列不存在',
                'first_record_time': '列不存在',
                'non_missing_count': '列不存在'
            })
            continue

        series = df[col_name]
        total_obs = len(series)
        missing_obs = series.isnull().sum()
        non_missing_count = total_obs - missing_obs  # 计算无缺失值的期数
        
        total_missing_pct = (missing_obs / total_obs) * 100.0 if total_obs > 0 else 0.0

        post_first_record_missing_pct_val = "N/A (无有效记录)"
        first_valid_index = series.first_valid_index()
        first_record_time = "N/A (无有效记录)"
        
        if first_valid_index is not None:
            # 获取第一条记录的时间
            if time_col is not None and time_col in df.columns:
                first_record_time = df.loc[first_valid_index, time_col]
                # 如果是时间戳类型，转换为字符串格式
                if isinstance(first_record_time, (pd.Timestamp, np.datetime64)):
                    first_record_time = first_record_time.strftime('%Y-%m-%d %H:%M:%S')
                elif hasattr(first_record_time, 'strftime'):
                    first_record_time = first_record_time.strftime('%Y-%m-%d')
                else:
                    first_record_time = str(first_record_time)
            else:
                # 没有时间列，使用索引位置
                first_record_time = f"索引位置 {first_valid_index}"
            
            series_after_first_valid = series[first_valid_index:]
            total_obs_after_first = len(series_after_first_valid)
            missing_obs_after_first = series_after_first_valid.isnull().sum()
            post_first_record_missing_pct_val = (missing_obs_after_first / total_obs_after_first) * 100.0 if total_obs_after_first > 0 else 0.0
        
        max_consecutive_missing = 0
        current_consecutive_missing = 0
        for val in series:
            if pd.isnull(val):
                current_consecutive_missing += 1
            else:
                max_consecutive_missing = max(max_consecutive_missing, current_consecutive_missing)
                current_consecutive_missing = 0
        max_consecutive_missing = max(max_consecutive_missing, current_consecutive_missing) # Final check

        results.append({
            'column': col_name,
            'total_missing_pct': total_missing_pct,
            'post_first_record_missing_pct': post_first_record_missing_pct_val,
            'max_consecutive_missing': max_consecutive_missing,
            'first_record_time': first_record_time,
            'non_missing_count': non_missing_count
        })
    return results

# --- MODIFIED FUNCTION: generate_missing_data_plot using Matplotlib ---
def generate_missing_data_plot(df, selected_cols, time_col_name=None, explicit_domain_start=None, explicit_domain_end=None):
    """
    Generates a plot showing the missing data pattern for selected columns using Matplotlib.
    Args:
        df (pd.DataFrame): The input DataFrame.
        selected_cols (list): List of columns to plot.
        time_col_name (str, optional): The name of the time column in df. 
                                       If None, df.index will be used.
        explicit_domain_start (datetime.date or pd.Timestamp, optional): Explicit start for X-axis domain.
        explicit_domain_end (datetime.date or pd.Timestamp, optional): Explicit end for X-axis domain.
    Returns:
        matplotlib.figure.Figure or None: The Matplotlib figure object, or None if plotting is not possible.
    """
    # 🔥 修复：在需要时才设置字体
    _setup_matplotlib_font()
    print("\n--- DEBUG: Missing Data Plot (Matplotlib) - Inside generate_missing_data_plot ---")
    print(f"[MissingPlot_MPL] Received df is None: {df is None}")
    if df is not None:
        print(f"[MissingPlot_MPL] Received df.shape: {df.shape}")
    print(f"[MissingPlot_MPL] Received selected_cols: {selected_cols}")
    print(f"[MissingPlot_MPL] Received time_col_name: {time_col_name}")
    print(f"[MissingPlot_MPL] Received explicit_domain_start: {explicit_domain_start}")
    print(f"[MissingPlot_MPL] Received explicit_domain_end: {explicit_domain_end}")

    if df is None or df.empty or not selected_cols:
        print("[MissingPlot_MPL] DataFrame is None, empty, or no columns selected. Returning None.")
        return None

    valid_selected_cols = [col for col in selected_cols if col in df.columns]
    if not valid_selected_cols:
        print("[MissingPlot_MPL] None of the selected columns are present in the DataFrame. Returning None.")
        return None
    print(f"[MissingPlot_MPL] Valid selected_cols for plotting: {valid_selected_cols}")

    # Prepare data for plotting
    data_for_plot = df.copy() # Work with a copy

    # Determine and prepare the time axis
    time_axis_values = None
    is_datetime_axis = False
    
    if time_col_name and time_col_name in data_for_plot.columns:
        print(f"[MissingPlot_MPL] Using column '{time_col_name}' as time source.")
        time_axis_values = pd.to_datetime(data_for_plot[time_col_name], errors='coerce')
        if not time_axis_values.isnull().all():
            is_datetime_axis = True
            data_for_plot.index = time_axis_values # Set index for easy slicing
        else:
            print("[MissingPlot_MPL] Time column resulted in all NaT. Falling back to numeric index.")
            time_axis_values = pd.Series(np.arange(len(data_for_plot))) # Fallback
            data_for_plot.index = time_axis_values
    elif isinstance(data_for_plot.index, pd.DatetimeIndex):
        print("[MissingPlot_MPL] Using DataFrame's DatetimeIndex as time source.")
        time_axis_values = data_for_plot.index
        is_datetime_axis = True
        # Index is already set
    else:
        print("[MissingPlot_MPL] No valid time column or DatetimeIndex. Using numeric index.")
        time_axis_values = pd.Series(np.arange(len(data_for_plot)))
        data_for_plot.index = time_axis_values

    if time_axis_values.isnull().all():
        print("[MissingPlot_MPL] Final time axis contains all NaT/None. Cannot plot. Returning None.")
        return None

    # Filter data based on explicit domain if provided and valid
    min_data_time = data_for_plot.index.min()
    max_data_time = data_for_plot.index.max()
    
    plot_start_time = min_data_time
    plot_end_time = max_data_time

    if explicit_domain_start is not None:
        try:
            exp_start = pd.Timestamp(explicit_domain_start) if is_datetime_axis else float(explicit_domain_start)
            if (is_datetime_axis and not pd.isna(exp_start)) or (not is_datetime_axis and not np.isnan(exp_start)):
                if exp_start > min_data_time : plot_start_time = exp_start
                print(f"[MissingPlot_MPL] Applied explicit start: {plot_start_time}")
        except Exception as e:
            print(f"[MissingPlot_MPL] Error applying explicit_domain_start '{explicit_domain_start}': {e}. Using data min.")
            
    if explicit_domain_end is not None:
        try:
            exp_end = pd.Timestamp(explicit_domain_end) if is_datetime_axis else float(explicit_domain_end)
            if (is_datetime_axis and not pd.isna(exp_end)) or (not is_datetime_axis and not np.isnan(exp_end)):
                 if exp_end < max_data_time : plot_end_time = exp_end
                 print(f"[MissingPlot_MPL] Applied explicit end: {plot_end_time}")
        except Exception as e:
            print(f"[MissingPlot_MPL] Error applying explicit_domain_end '{explicit_domain_end}': {e}. Using data max.")

    if plot_start_time > plot_end_time:
        print(f"[MissingPlot_MPL] Domain error: plot_start_time ({plot_start_time}) is after plot_end_time ({plot_end_time}). Cannot plot. Returning None.")
        return None

    try:
        # Slice the DataFrame according to the determined plot domain
        # Ensure index is sorted for proper slicing if it's a DatetimeIndex
        if isinstance(data_for_plot.index, pd.DatetimeIndex) and not data_for_plot.index.is_monotonic_increasing:
            data_for_plot = data_for_plot.sort_index()
            print("[MissingPlot_MPL] Sorted DataFrame index for slicing.")

        # Perform slicing
        # Note: For pd.Series.loc or df.loc with DatetimeIndex, both start and end are inclusive.
        # For numeric index, .loc is also inclusive.
        data_to_display = data_for_plot.loc[plot_start_time:plot_end_time, valid_selected_cols]

    except Exception as e_slice:
        print(f"[MissingPlot_MPL] Error slicing data for display domain [{plot_start_time} to {plot_end_time}]: {e_slice}. Returning None.")
        return None
        
    if data_to_display.empty:
        print("[MissingPlot_MPL] Data for display is empty after time domain filtering. Returning None.")
        return None

    missing_matrix = data_to_display.isnull()

    # Matplotlib plotting
    fig, ax = plt.subplots(figsize=(10, max(5, len(valid_selected_cols) * 0.5))) # Dynamic height

    # Define colors: 0 for Present (False in missing_matrix), 1 for Missing (True in missing_matrix)
    # cmap = mcolors.ListedColormap(['#6BAED6', '#D3D3D3']) # Blue for Present, Light Grey for Missing
    cmap = mcolors.ListedColormap(['#6BAED6', '#E0E0E0']) # Blue for Present, Lighter Grey for Missing

    # imshow expects True/False to be mapped to 1/0 by default if cmap has 2 colors,
    # or use .astype(int) if specific mapping needed.
    # We want False (present) to be one color, True (missing) to be another.
    # Let's ensure missing_matrix.T is int so 0=Present, 1=Missing
    ax.imshow(missing_matrix.T.astype(int), aspect='auto', cmap=cmap, interpolation='nearest')

    # Set Y-axis (Variables)
    ax.set_yticks(np.arange(len(valid_selected_cols)))
    ax.set_yticklabels(valid_selected_cols) # Matplotlib should handle full names if figure is wide enough
    ax.set_ylabel("变量")

    # Set X-axis (Time)
    # For X-axis ticks, we need to be more careful, especially with datetime
    num_time_points_display = len(data_to_display.index)
    
    if is_datetime_axis:
        # Use matplotlib's date locators and formatters
        if num_time_points_display > 1:
             # Auto-locator and formatter for dates
            major_locator = mdates.AutoDateLocator(minticks=3, maxticks=10) # Adjust maxticks as needed
            major_formatter = mdates.ConciseDateFormatter(major_locator) # Nicer formatting
            ax.xaxis.set_major_locator(major_locator)
            ax.xaxis.set_major_formatter(major_formatter)
            # Set x-ticks to correspond to the actual indices in missing_matrix
            # imshow plots from 0 to N-1 for each dimension.
            # We need to map these back to date values for the labels if we want to be super precise,
            # but using the locators on the original date range should be fine if the imshow extent isn't explicitly set.
            # For now, let the locator handle it based on the data range implied by data_to_display.index
            ax.set_xticks(np.linspace(0, num_time_points_display - 1, num=min(num_time_points_display, 10), dtype=int))
            ax.set_xticklabels([data_to_display.index[i].strftime('%Y-%m-%d') for i in ax.get_xticks()], rotation=45, ha="right")

        elif num_time_points_display == 1: # Single data point
            ax.set_xticks([0])
            ax.set_xticklabels([data_to_display.index[0].strftime('%Y-%m-%d')], rotation=45, ha="right")
    else: # Numeric axis
        # For numeric axis, show a few ticks
        tick_positions = np.linspace(0, num_time_points_display - 1, num=min(num_time_points_display, 10), dtype=int)
        ax.set_xticks(tick_positions)
        ax.set_xticklabels([data_to_display.index[i] for i in tick_positions], rotation=45, ha="right")

    ax.set_xlabel("时间")
    ax.set_title("缺失数据模式图 (Missing Data Pattern)", pad=20)

    # Create legend
    # colors = {'存在': '#6BAED6', '缺失': '#E0E0E0'} # Swapped order to match imshow mapping (0=Present, 1=Missing)
    colors = {'存在': cmap.colors[0], '缺失': cmap.colors[1]}
    labels = list(colors.keys())
    handles = [plt.Rectangle((0,0),1,1, color=colors[label]) for label in labels]
    # Position legend below the plot
    fig.legend(handles, labels, loc='lower center', ncol=2, bbox_to_anchor=(0.5, -0.05), frameon=False)
    
    plt.tight_layout(rect=[0, 0.05, 1, 0.95]) # Adjust rect to make space for legend: [left, bottom, right, top]
    
    print("[MissingPlot_MPL] Matplotlib chart created. Returning fig.")
    return fig
# --- END OF MODIFIED FUNCTION ---

def handle_missing_values(df_processed, selected_cols, method, limit=None, order=None, constant_value=None):
    """
    Handles missing values in selected columns of a DataFrame using various methods.
    """
    if df_processed is None or df_processed.empty:
        return df_processed, "错误：数据为空，无法处理缺失值。"
    if not selected_cols:
        return df_processed, "没有选择任何列来处理缺失值。"

    df_copy = df_processed.copy()
    processed_count = 0
    skipped_cols_type = []
    skipped_cols_no_missing = []

    for col_name in selected_cols:
        if col_name not in df_copy.columns:
            print(f"警告：列 '{col_name}' 在数据中不存在，跳过。")
            skipped_cols_type.append(f"{col_name} (不存在)")
            continue

        if df_copy[col_name].isnull().sum() == 0:
            print(f"信息：列 '{col_name}' 没有缺失值，跳过。")
            skipped_cols_no_missing.append(col_name)
            continue 

        original_dtype = df_copy[col_name].dtype

        try:
            if method == 'ffill':
                df_copy[col_name] = df_copy[col_name].fillna(method='ffill', limit=limit)
            elif method == 'bfill':
                df_copy[col_name] = df_copy[col_name].fillna(method='bfill', limit=limit)
            elif method == 'zero':
                if pd.api.types.is_numeric_dtype(original_dtype) or pd.api.types.is_datetime64_any_dtype(original_dtype):
                    df_copy[col_name] = df_copy[col_name].fillna(0)
                else:
                    skipped_cols_type.append(f"{col_name} (非数值型，无法用0填充)")
                    continue 
            elif method == 'constant':
                if constant_value is not None:
                    try:
                        # Attempt to cast constant_value to the original dtype of the column
                        # This helps maintain data integrity, e.g., filling an int column with an int
                        typed_constant = pd.Series([constant_value]).astype(original_dtype).iloc[0]
                        df_copy[col_name] = df_copy[col_name].fillna(typed_constant)
                    except (ValueError, TypeError) as e_type:
                        # If casting fails (e.g., trying to cast "abc" to int), fallback to filling with the raw constant
                        # This might change the dtype of the column if it was strictly numeric before.
                        print(f"[MissingFill] Warning: Could not cast constant value '{constant_value}' to dtype '{original_dtype}' for column '{col_name}'. Error: {e_type}. Filling with raw value.")
                        df_copy[col_name] = df_copy[col_name].fillna(constant_value)
                else:
                    # This case should ideally be caught by UI validation before calling this function
                    skipped_cols_type.append(f"{col_name} (未提供常量值)")
                    continue
            elif method in ['linear', 'cubic', 'polynomial', 'spline', 'quadratic', 'slinear', 'pchip', 'akima']: # Expanded interpolation methods
                if pd.api.types.is_numeric_dtype(original_dtype):
                    # Ensure column is float for interpolation
                    df_copy[col_name] = df_copy[col_name].astype(float) 
                    
                    interp_kwargs = {}
                    if method in ['polynomial', 'spline']:
                        interp_kwargs['order'] = order if order is not None else (3 if method == 'spline' else 2) # Default order for spline/poly
                    
                    df_copy[col_name] = df_copy[col_name].interpolate(method=method, limit_direction='both', **interp_kwargs)
                else:
                    skipped_cols_type.append(f"{col_name} (非数值型，无法插值)")
                    continue
            elif method in ['mean', 'median']:
                if pd.api.types.is_numeric_dtype(original_dtype):
                    fill_value = df_copy[col_name].mean() if method == 'mean' else df_copy[col_name].median()
                    df_copy[col_name] = df_copy[col_name].fillna(fill_value)
                else:
                    skipped_cols_type.append(f"{col_name} (非数值型，无法用{method}填充)")
                    continue
            elif method == 'mode':
                mode_val = df_copy[col_name].mode()
                if not mode_val.empty:
                    df_copy[col_name] = df_copy[col_name].fillna(mode_val[0])
                else: 
                    # This can happen if all values are NaN, or all unique (no single mode)
                    skipped_cols_type.append(f"{col_name} (无法计算众数，可能全为缺失或值均不相同)")
                    continue
            # Removed 'none' as a processing method here, should be handled by UI not calling this.
            else: # Should not be reached if UI validates methods
                skipped_cols_type.append(f"{col_name} (未知或不支持的填充方法: {method})")
                continue
            
            # Check if dtype changed unintentionally, e.g., int to float after filling with NaN then a float mean
            if df_copy[col_name].dtype != original_dtype:
                print(f"[MissingFill] Info: Dtype of column '{col_name}' changed from '{original_dtype}' to '{df_copy[col_name].dtype}' after filling.")
            
            processed_count += 1
        except Exception as e:
            print(f"处理列 '{col_name}' 时出错 (方法: {method}): {e}")
            skipped_cols_type.append(f"{col_name} (处理时出错: {e})")
            continue
            
    message_parts = []
    if processed_count > 0:
        message_parts.append(f"成功对 {processed_count} 个选定列应用了缺失值处理方法。")
    
    # Consolidate messages for skipped columns
    skipped_any = False
    if skipped_cols_no_missing:
        message_parts.append(f"跳过 {len(skipped_cols_no_missing)} 个无缺失值的列: {', '.join(skipped_cols_no_missing)}。")
        skipped_any = True
    if skipped_cols_type: # This now includes more reasons like "不存在", "未提供常量", "无法计算众数" etc.
        message_parts.append(f"跳过 {len(skipped_cols_type)} 个因其他原因（如类型不符、方法不适用、处理错误等）的列: {', '.join(skipped_cols_type)}。")
        skipped_any = True
    
    if not processed_count and not skipped_any: # No processing and nothing explicitly skipped (e.g. no columns selected)
        message = "没有选择任何列或所选列均未进行缺失值处理。" # More generic
    elif not processed_count and skipped_any: # Nothing processed, but some were skipped
        message = " ".join(message_parts) + " 未处理任何其他列的缺失值。"
    else: # Some processing occurred
        message = " ".join(message_parts)
        
    return df_copy, message 