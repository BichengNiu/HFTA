import pandas as pd
import numpy as np
import re

# Extracted from time_series_clean_utils.py
def parse_time_column(series, success_threshold=0.5):
    """
    Tries to parse a pandas Series into datetime objects using various methods.

    Args:
        series (pd.Series): The column to parse.
        success_threshold (float): Minimum fraction of non-NaT values required
                                   for parsing to be considered successful.

    Returns:
        tuple: (parsed_series, format_used) if successful, else (None, None).
               format_used can be 'auto' or an explicit format string.
    """
    original_series = series.copy()
    if pd.api.types.is_datetime64_any_dtype(series):
        if series.notna().mean() >= success_threshold:
            print(f"Column '{series.name}' is already datetime64.")
            return series, 'datetime64'
        else:
            print(f"Column '{series.name}' is datetime64 but has too many NaNs after initial check.")
            return None, None

    if pd.api.types.is_numeric_dtype(series.dtype):
        try:
            is_likely_year_column = False
            col_name_lower = str(series.name).lower()
            if 'year' in col_name_lower or '年份' in col_name_lower:
                is_likely_year_column = True

            if is_likely_year_column and series.dropna().between(1800, 2200).all():
                parsed_year_as_date = pd.to_datetime(series.dropna().astype(int).astype(str), format='%Y', errors='coerce')
                parsed_year_as_date = parsed_year_as_date.reindex(series.index)
                if parsed_year_as_date.notna().mean() >= success_threshold:
                    print(f"Column '{series.name}' parsed as year numbers (format %Y).")
                    return parsed_year_as_date, '%Y'
            elif series.dropna().mean() > 10000: 
                parsed_excel_date = pd.to_datetime(series, unit='D', origin='1899-12-30', errors='coerce')
                if parsed_excel_date.notna().mean() >= success_threshold:
                    print(f"Column '{series.name}' parsed as Excel numeric date (origin 1899-12-30).")
                    return parsed_excel_date, 'excel_numeric'
        except Exception as e_excel:
            print(f"Attempt to parse column '{series.name}' as numeric (year/Excel) date failed: {e_excel}")
            pass 

    try:
        series_str = series.astype(str)
    except Exception:
        print(f"Column '{series.name}' could not be converted to string.")
        return None, None

    parsed_series = None
    format_used = None

    try:
        parsed_auto = pd.to_datetime(series_str, errors='coerce', infer_datetime_format=True)
        if parsed_auto.notna().mean() >= success_threshold:
            print(f"Column '{series.name}' parsed with auto-inference.")
            return parsed_auto, 'auto'
    except Exception as e:
        print(f"Auto-parsing for '{series.name}' failed: {e}")
        pass 

    common_formats = [
        '%Y-%m-%d', '%Y/%m/%d', '%Y%m%d', 
        '%Y-%m-%d %H:%M:%S', '%Y/%m/%d %H:%M:%S', 
        '%Y-%m', '%Y%m', '%Y年%m月', 
        '%Y', 
        '%Y年%m月%d日',
        '%d/%m/%Y', '%m/%d/%Y', 
        '%d-%b', 
    ]

    for fmt in common_formats:
        try:
            parsed_explicit = pd.to_datetime(series_str, format=fmt, errors='coerce')
            if parsed_explicit.notna().mean() >= success_threshold:
                print(f"Column '{series.name}' parsed with format: {fmt}")
                return parsed_explicit, fmt 
        except ValueError:
            continue
        except Exception as e:
            print(f"Error parsing '{series.name}' with format {fmt}: {e}")
            continue 

    print(f"Column '{series.name}' could not be reliably parsed as datetime.")
    print(f"--- DEBUG: Failed to parse column '{series.name}' ---")
    print(f"Original Dtype: {original_series.dtype}")
    print(f"Head of original series data (first 5 values):")
    print(original_series.head().to_string())
    print(f"Head of series_str data (first 5 values for string parsing attempts):")
    print(series_str.head().to_string())
    print(f"--- END DEBUG --- ")
    return None, None

def parse_indices(index_str):
    """Parses comma-separated string into a list of integers."""
    indices = []
    if not index_str:
        return indices
    parts = index_str.split(',')
    for part in parts:
        part = part.strip()
        if not part:
            continue
        try:
            if '-' in part:
                start, end = map(int, part.split('-'))
                if start > end:
                     raise ValueError("Range start cannot be greater than end")
                indices.extend(list(range(start, end + 1)))
            else:
                indices.append(int(part))
        except ValueError:
            raise ValueError(f"无法解析索引 '{part}'. 请输入数字、逗号或范围 (例如 '1, 3, 5-7').")
    return sorted(list(set(indices))) 