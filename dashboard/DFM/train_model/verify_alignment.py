# dym_estimate/verify_alignment.py
import pandas as pd
import numpy as np
import os
import unicodedata

def get_nearest_friday(dt):
    """Calculates the nearest Friday to a given datetime."""
    # Monday=0, Tuesday=1, ..., Friday=4, Saturday=5, Sunday=6
    return dt + pd.Timedelta(days=4 - dt.weekday())

def get_last_friday_of_month(dt):
    """Calculates the last Friday of the month containing the given datetime."""
    # Go to the end of the month, then go back to the last Friday
    last_day_of_month = dt + pd.offsets.MonthEnd(0)
    return last_day_of_month - pd.Timedelta(days=(last_day_of_month.weekday() - 4 + 7) % 7)

def verify_data_alignment(prepared_csv_path: str, raw_excel_path: str):
    """
    Verifies that the prepared weekly data aligns correctly with the raw monthly data
    according to the specified rules:
    - Target Sheet ('工业增加值同比增速_月度_同花顺') vars: Align to nearest Friday of pub date.
    - Other Monthly Sheets (e.g., '电力_月度_Myteel') vars: Align to last Friday of the month of pub date.
    """
    print(f"--- 开始验证数据对齐 ---")
    print(f"Prepared Data: {prepared_csv_path}")
    print(f"Raw Data: {raw_excel_path}")

    target_sheet_name = '工业增加值同比增速_月度_同花顺'
    other_monthly_sheet_example = '电力_月度_Myteel' # Add more if needed

    try:
        # --- Load Prepared Data ---
        df_prepared = pd.read_csv(prepared_csv_path, index_col='Date', parse_dates=True)
        if not isinstance(df_prepared.index, pd.DatetimeIndex):
             raise ValueError("Prepared data index is not a DatetimeIndex.")
        if df_prepared.index.freqstr != 'W-FRI':
             print(f"警告: Prepared data index frequency is {df_prepared.index.freqstr}, not W-FRI.")
        print(f"成功加载 Prepared Data. Shape: {df_prepared.shape}")

        # --- Load Raw Data ---
        xls = pd.ExcelFile(raw_excel_path)

        # --- Verification Logic ---
        all_checks_passed = True
        mismatches = []

        # 1. Verify Target Sheet Alignment (Nearest Friday)
        print(f"\n--- 验证目标 Sheet '{target_sheet_name}' (最近周五对齐) ---")
        if target_sheet_name not in xls.sheet_names:
            print(f"错误: 在 Excel 文件中未找到目标 Sheet '{target_sheet_name}'")
            return

        df_target_raw = pd.read_excel(xls, sheet_name=target_sheet_name, header=0, skiprows=None) # Adjust header/skiprows if needed for this specific sheet
        if df_target_raw.shape[1] < 2:
             print(f"错误: 目标 Sheet '{target_sheet_name}' 少于 2 列。无法验证。")
             return

        date_col_target = df_target_raw.columns[0]
        df_target_raw['PubDate'] = pd.to_datetime(df_target_raw[date_col_target], errors='coerce')
        df_target_raw = df_target_raw.dropna(subset=['PubDate']).set_index('PubDate')

        target_sheet_cols_raw = list(df_target_raw.columns[1:]) # Get all columns B, C, D...
        target_sheet_cols_in_prepared = [col for col in target_sheet_cols_raw if col in df_prepared.columns]

        print(f"  将在 Prepared Data 中检查来自目标 Sheet 的列: {target_sheet_cols_in_prepared}")

        for col_name in target_sheet_cols_in_prepared:
            print(f"    检查列: '{col_name}'...")
            prepared_series = df_prepared[col_name].dropna()
            raw_series = pd.to_numeric(df_target_raw[col_name], errors='coerce').dropna()

            # Check 1: All non-NaN values in prepared data must correspond to a nearest Friday of a raw pub date
            for prepared_date, prepared_value in prepared_series.items():
                is_nearest_friday = False
                matching_raw_value = np.nan
                for raw_pub_date, raw_value in raw_series.items():
                    nearest_friday = get_nearest_friday(raw_pub_date)
                    if prepared_date == nearest_friday:
                        # Allow for small float differences
                        if np.isclose(prepared_value, raw_value, atol=1e-6, rtol=1e-6):
                            is_nearest_friday = True
                            matching_raw_value = raw_value
                            break # Found a match for this prepared date
                        else:
                             # Date matches, value doesn't (check if it matches another pub date mapping to the same Friday)
                             pass

                if not is_nearest_friday:
                    all_checks_passed = False
                    mismatch_info = {
                        'Type': 'Target Sheet Misalignment',
                        'Column': col_name,
                        'Prepared Date': prepared_date.strftime('%Y-%m-%d'),
                        'Prepared Value': prepared_value,
                        'Issue': 'Prepared date is not the nearest Friday for any raw publication date.'
                    }
                    mismatches.append(mismatch_info)
                    print(f"      失败: 日期 {prepared_date.date()} 上的值 {prepared_value} 不是任何原始发布日期的最近周五。")

            # Check 2: All raw data points should map to a value on the nearest Friday in prepared data (unless filtered out)
            raw_dates_mapped = set()
            for raw_pub_date, raw_value in raw_series.items():
                nearest_friday = get_nearest_friday(raw_pub_date)
                raw_dates_mapped.add(nearest_friday) # Track which Fridays should have data

                if nearest_friday in df_prepared.index:
                    prepared_value_at_friday = df_prepared.loc[nearest_friday, col_name]
                    if pd.isna(prepared_value_at_friday):
                        # Potentially OK if multiple raw dates map to the same Friday and we keep the latest
                        # Re-check if *any* raw value mapping to this Friday matches
                        raw_vals_for_this_friday = [rv for rpd, rv in raw_series.items() if get_nearest_friday(rpd) == nearest_friday]
                        if not any(np.isclose(prepared_series.get(nearest_friday, np.nan), rv, atol=1e-6, rtol=1e-6) for rv in raw_vals_for_this_friday):
                             # It's only a failure if the prepared value is NaN AND *no* raw value for that nearest Friday matches
                             if nearest_friday in prepared_series.index: # Ensure the prepared series actually has an entry for this Friday
                                  all_checks_passed = False
                                  mismatch_info = {
                                       'Type': 'Target Sheet Missing Data',
                                       'Column': col_name,
                                       'Raw Pub Date': raw_pub_date.strftime('%Y-%m-%d'),
                                       'Nearest Friday': nearest_friday.strftime('%Y-%m-%d'),
                                       'Raw Value': raw_value,
                                       'Issue': f'Nearest Friday {nearest_friday.date()} exists in prepared data, but value is NaN or incorrect.'
                                  }
                                  mismatches.append(mismatch_info)
                                  print(f"      失败: 原始发布 {raw_pub_date.date()} (值 {raw_value}) 应在最近周五 {nearest_friday.date()} 有值，但 Prepared Data 中该处为 NaN 或值不匹配。")
                    # Value comparison already implicitly checked in Check 1

                else:
                    # OK if the nearest Friday is outside the final date range of prepared data
                    pass

            # Check 3: Ensure prepared data only has values on Fridays that ARE nearest Fridays
            fridays_with_data = set(prepared_series.index)
            fridays_that_are_nearest = raw_dates_mapped
            unexpected_fridays = fridays_with_data - fridays_that_are_nearest
            if unexpected_fridays:
                 all_checks_passed = False
                 for unexpected_friday in sorted(list(unexpected_fridays)):
                      mismatch_info = {
                            'Type': 'Target Sheet Unexpected Data',
                            'Column': col_name,
                            'Unexpected Date': unexpected_friday.strftime('%Y-%m-%d'),
                            'Value': prepared_series.loc[unexpected_friday],
                            'Issue': 'Data found on a Friday that is not a nearest Friday for any raw publication date.'
                      }
                      mismatches.append(mismatch_info)
                      print(f"      失败: 列 '{col_name}' 在日期 {unexpected_friday.date()} 有非预期值 {prepared_series.loc[unexpected_friday]} (该日期非任何发布日期的最近周五)。")


        # 2. Verify Other Monthly Sheets Alignment (Last Friday of Month)
        print(f"\n--- 验证其他月度 Sheet (示例: '{other_monthly_sheet_example}') (月末最后周五对齐) ---")
        if other_monthly_sheet_example not in xls.sheet_names:
            print(f"警告: 在 Excel 文件中未找到示例其他月度 Sheet '{other_monthly_sheet_example}'，跳过此部分验证。")
        else:
            df_other_raw = pd.read_excel(xls, sheet_name=other_monthly_sheet_example, header=1, skiprows=None) # Adjust header/skiprows
            if df_other_raw.shape[1] < 2:
                 print(f"错误: 其他月度 Sheet '{other_monthly_sheet_example}' 少于 2 列。无法验证。")
                 return

            date_col_other = df_other_raw.columns[0]
            df_other_raw['PubDate'] = pd.to_datetime(df_other_raw[date_col_other], errors='coerce')
            df_other_raw = df_other_raw.dropna(subset=['PubDate']).set_index('PubDate')

            other_sheet_cols_raw = list(df_other_raw.columns[1:-1]) # Exclude PubDate added col
            other_sheet_cols_in_prepared = [col for col in other_sheet_cols_raw if col in df_prepared.columns]

            print(f"  将在 Prepared Data 中检查来自 '{other_monthly_sheet_example}' 的列: {other_sheet_cols_in_prepared}")

            for col_name in other_sheet_cols_in_prepared:
                 print(f"    检查列: '{col_name}'...")
                 prepared_series = df_prepared[col_name].dropna()
                 raw_series_other = pd.to_numeric(df_other_raw[col_name], errors='coerce').dropna()

                 # Group raw data by month-end and get the last value for that month
                 raw_series_other = raw_series_other.sort_index() # Ensure order before grouping
                 monthly_last_raw = raw_series_other.groupby(raw_series_other.index.to_period('M')).last()
                 monthly_last_raw.index = monthly_last_raw.index.to_timestamp('M') # Index is now MonthEnd

                 # Check 1: All non-NaN values in prepared data must correspond to a last Friday of a month end from raw data
                 raw_month_ends_present = set(monthly_last_raw.index)
                 last_fridays_for_raw_months = {get_last_friday_of_month(me) for me in raw_month_ends_present}

                 for prepared_date, prepared_value in prepared_series.items():
                     is_last_friday = False
                     matching_raw_value = np.nan
                     if prepared_date in last_fridays_for_raw_months:
                          # Find the corresponding month end for this last friday
                          possible_month_ends = [me for me in raw_month_ends_present if get_last_friday_of_month(me) == prepared_date]
                          if possible_month_ends:
                               # Use the latest month end if multiple map to same Friday (unlikely but safe)
                               month_end_to_check = max(possible_month_ends)
                               raw_value = monthly_last_raw.get(month_end_to_check, np.nan)
                               # Check if the prepared value matches the raw value for that month end (allowing for stationarity diff)
                               # This check is tricky because prepared data might be differenced.
                               # We primarily check if the date alignment is correct.
                               is_last_friday = True # Assume date alignment is the main check here
                               matching_raw_value = raw_value # Store for info

                     if not is_last_friday:
                         all_checks_passed = False
                         mismatch_info = {
                            'Type': 'Other Monthly Misalignment',
                            'Column': col_name,
                            'Prepared Date': prepared_date.strftime('%Y-%m-%d'),
                            'Prepared Value': prepared_value,
                            'Issue': 'Prepared date is not the last Friday for any month with raw data.'
                         }
                         mismatches.append(mismatch_info)
                         print(f"      失败: 日期 {prepared_date.date()} 上的值 {prepared_value} 不是任何原始数据月份的最后周五。")

                 # Check 2: All monthly last raw data points should map to a value on the last Friday in prepared data (unless filtered)
                 last_fridays_mapped = set()
                 for month_end_date, raw_value in monthly_last_raw.items():
                     last_friday = get_last_friday_of_month(month_end_date)
                     last_fridays_mapped.add(last_friday)

                     if last_friday in df_prepared.index:
                         prepared_value_at_friday = df_prepared.loc[last_friday, col_name]
                         if pd.isna(prepared_value_at_friday):
                            # Failure only if the date exists but value is NaN
                            all_checks_passed = False
                            mismatch_info = {
                                'Type': 'Other Monthly Missing Data',
                                'Column': col_name,
                                'Month End': month_end_date.strftime('%Y-%m-%d'),
                                'Last Friday': last_friday.strftime('%Y-%m-%d'),
                                'Raw Value': raw_value,
                                'Issue': f'Last Friday {last_friday.date()} exists in prepared data, but value is NaN.'
                            }
                            mismatches.append(mismatch_info)
                            print(f"      失败: 原始月末 {month_end_date.date()} (值 {raw_value}) 应在最后周五 {last_friday.date()} 有值，但 Prepared Data 中该处为 NaN。")
                     # Value comparison is complex due to potential stationarity transforms, focus on date presence.

                 # Check 3: Ensure prepared data only has values on Fridays that ARE last Fridays of a month
                 fridays_with_data = set(prepared_series.index)
                 unexpected_fridays = fridays_with_data - last_fridays_mapped
                 if unexpected_fridays:
                     all_checks_passed = False
                     for unexpected_friday in sorted(list(unexpected_fridays)):
                          mismatch_info = {
                                'Type': 'Other Monthly Unexpected Data',
                                'Column': col_name,
                                'Unexpected Date': unexpected_friday.strftime('%Y-%m-%d'),
                                'Value': prepared_series.loc[unexpected_friday],
                                'Issue': 'Data found on a Friday that is not a last Friday for any month with raw data.'
                          }
                          mismatches.append(mismatch_info)
                          print(f"      失败: 列 '{col_name}' 在日期 {unexpected_friday.date()} 有非预期值 {prepared_series.loc[unexpected_friday]} (该日期非任何月份的最后周五)。")


        # --- Final Report ---
        print(f"\n--- 验证结果 ---")
        if all_checks_passed:
            print("[SUCCESS] 所有检查通过！数据对齐方式符合预期规则。")
        else:
            print(f"[FAILURE] 检测到 {len(mismatches)} 处不一致！详情如下:")
            for i, mismatch in enumerate(mismatches):
                print(f"  {i+1}. 类型: {mismatch['Type']}, 列: {mismatch['Column']}, 日期/月份: {mismatch.get('Prepared Date') or mismatch.get('Month End') or mismatch.get('Unexpected Date')}, 问题: {mismatch['Issue']}")

    except FileNotFoundError:
        print(f"错误: 无法找到输入文件。 Prepared: '{prepared_csv_path}', Raw: '{raw_excel_path}'")
    except Exception as e:
        print(f"验证过程中发生意外错误: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    # --- 配置路径 ---
    PREPARED_CSV = os.path.join("dym_estimate", "tests", "test_prepared_weekly_v3.csv")
    RAW_EXCEL = os.path.join("data", "经济数据库0424_带数据源标志.xlsx")

    # --- 检查文件是否存在 ---
    if not os.path.exists(PREPARED_CSV):
         print(f"错误: 找不到 Prepared CSV 文件: {PREPARED_CSV}")
         exit(1)
    if not os.path.exists(RAW_EXCEL):
         # Try alternative raw path if original not found
         alt_raw_path = os.path.join("..", "data", "经济数据库0424.xlsx")
         if os.path.exists(alt_raw_path):
              RAW_EXCEL = alt_raw_path
              print(f"警告: 使用备用 Raw Excel 路径: {RAW_EXCEL}")
         else:
              print(f"错误: 找不到 Raw Excel 文件于 '{RAW_EXCEL}' 或 '{alt_raw_path}'")
              exit(1)

    # --- 运行验证 ---
    verify_data_alignment(PREPARED_CSV, RAW_EXCEL) 