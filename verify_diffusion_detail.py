import pandas as pd
import numpy as np
import json
import os
from datetime import datetime

# 重新引入 print_and_log 函数
def print_and_log(message, file_handle):
    """Helper function to print to console and write to file."""
    print(message)
    file_handle.write(message + "\n")

# 修改函数签名，重新加入 output_log_file 参数
def verify_diffusion_for_industry(industry_name: str, verification_date_str: str = None, output_log_file: str = "diffusion_verification_log.txt"):
    """
    Performs detailed verification of WoW, YoY, and Mix diffusion indices for a specific industry,
    writing results to a specified log file.

    Args:
        industry_name (str): The exact name of the industry (source filename without extension).
        verification_date_str (str, optional): Specific date ('YYYY-MM-DD') to verify. 
                                             If None, uses the latest date with data. Defaults to None.
        output_log_file (str): Path to the output log file.
    """
    MAPPING_FILE = "indicator_source_mapping.json"
    WEEKLY_DATA_FILE = "merged_output.xlsx"
    WEEKLY_SHEET = "WeeklyData"
    DIFFUSION_RESULT_FILE = "industry_diffusion_indices_combined.xlsx"
    DIFFUSION_SHEET = "Diffusion Indices"
    MISSING_THRESHOLD = 0.7
    COMBINED_MISSING_THRESHOLD = 0.33
    TOLERANCE_THRESHOLD = 0.01

    with open(output_log_file, 'w', encoding='utf-8') as log_f:
        print_and_log(f"--- 开始验证行业: {industry_name} ---", log_f)
        print_and_log(f"验证时间: {datetime.now()}", log_f)
        print_and_log(f"使用增长阈值 (TOLERANCE_THRESHOLD): {TOLERANCE_THRESHOLD}", log_f)
        print_and_log(f"使用组合变化率缺失阈值 (COMBINED_MISSING_THRESHOLD): {COMBINED_MISSING_THRESHOLD}", log_f)
        print_and_log("-" * 50, log_f)

        # --- 1. 加载必要文件 ---
        try:
            print_and_log(f"加载指标来源映射: {MAPPING_FILE}", log_f)
            with open(MAPPING_FILE, 'r', encoding='utf-8') as f:
                indicator_source_map = json.load(f)
            print_and_log(f"  成功加载 {len(indicator_source_map)} 个指标映射。", log_f)

            print_and_log(f"加载周度数据: {WEEKLY_DATA_FILE} (Sheet: {WEEKLY_SHEET})", log_f)
            weekly_data_all = pd.read_excel(WEEKLY_DATA_FILE, sheet_name=WEEKLY_SHEET, index_col=0)
            weekly_data_all.index = pd.to_datetime(weekly_data_all.index)
            weekly_data_all = weekly_data_all.sort_index()
            print_and_log(f"  成功加载周度数据，形状: {weekly_data_all.shape}", log_f)

            print_and_log(f"加载扩散指数结果: {DIFFUSION_RESULT_FILE} (Sheet: {DIFFUSION_SHEET})", log_f)
            diffusion_results_all = pd.read_excel(DIFFUSION_RESULT_FILE, sheet_name=DIFFUSION_SHEET, index_col=0, header=[0, 1])
            diffusion_results_all.index = pd.to_datetime(diffusion_results_all.index)
            print_and_log("  成功加载预计算的扩散指数。", log_f)

            # 提取当前行业的预计算结果
            if industry_name in diffusion_results_all.columns.levels[0]:
                pre_calculated_results = diffusion_results_all[industry_name]
            else:
                print_and_log(f"错误: 在预计算结果文件中未找到行业 '{industry_name}' 的列。", log_f)
                return

        except FileNotFoundError as e:
            print_and_log(f"错误: 文件未找到 - {e}", log_f)
            return
        except Exception as e:
            print_and_log(f"错误: 加载文件时出错 - {e}", log_f)
            return

        # --- 2. 筛选行业指标和数据 ---
        print_and_log(f"\n--- 筛选行业指标: {industry_name} ---", log_f)
        industry_indicators = [indicator for indicator, source in indicator_source_map.items()
                               if source == industry_name and indicator in weekly_data_all.columns]

        if not industry_indicators:
            print_and_log(f"错误: 在来源映射中未找到行业 '{industry_name}' 的指标，或指标不在周度数据中。", log_f)
            return

        print_and_log(f"找到 {len(industry_indicators)} 个指标:", log_f)
        for ind in industry_indicators:
            print_and_log(f"  - {ind}", log_f)

        industry_data = weekly_data_all[industry_indicators]
        print_and_log(f"提取的行业数据形状: {industry_data.shape}", log_f)

        # --- 3. 确定验证日期 ---
        if verification_date_str:
            try:
                verify_date = pd.to_datetime(verification_date_str)
                if verify_date not in industry_data.index:
                    print_and_log(f"警告: 指定日期 {verification_date_str} 不在数据索引中，将使用最新日期。", log_f)
                    verify_date = industry_data.index.max() # Use latest date if specified one not found
            except ValueError:
                print_and_log(f"警告: 指定日期格式无效 '{verification_date_str}'，将使用最新日期。", log_f)
                verify_date = industry_data.index.max()
        else:
            verify_date = industry_data.index.max() # Use the latest date by default

        # Adjust if the chosen date has no data or too much missing data for calculation
        if verify_date not in industry_data.index or industry_data.loc[verify_date].isnull().all():
            # Go back until a date with some data is found
            available_dates = industry_data.dropna(how='all').index
            if available_dates.empty:
                print_and_log("错误：该行业所有日期的数据都为空。", log_f)
                return
            verify_date = available_dates.max()

        verify_date_str = verify_date.strftime('%Y-%m-%d')
        print_and_log(f"\n--- 选定验证日期: {verify_date_str} ---", log_f)

        # --- 4. 验证环比 (WoW) 扩散指数 ---
        print_and_log("\n" + "="*20 + f" 环比 (WoW) 验证: {verify_date_str} " + "="*20, log_f)
        calculated_wow_index = np.nan # 初始化
        if verify_date == industry_data.index.min():
            print_and_log("无法计算环比，因为这是数据的第一个日期。", log_f)
        else:
            try:
                prev_date = industry_data.index[industry_data.index.get_loc(verify_date) - 1]
                current_values = industry_data.loc[verify_date]
                prev_values = industry_data.loc[prev_date]
                # a. 环比增长率
                wow_growth_rates = (current_values - prev_values) / prev_values
                # b. 缺失比例检查 (基于原始数据)
                non_missing_ratio = current_values.notna().mean()
                print_and_log(f"缺失值检查 (原始数据 @ {verify_date_str}, 阈值 < {1-MISSING_THRESHOLD:.1f}): 非缺失比例 = {non_missing_ratio:.2f}", log_f)
                if non_missing_ratio >= MISSING_THRESHOLD:
                    valid_growth = wow_growth_rates.dropna()
                    valid_growth_count = len(valid_growth)
                    if valid_growth_count > 0:
                        increasing_count = (valid_growth > TOLERANCE_THRESHOLD).sum()
                        calculated_wow_index = (increasing_count / valid_growth_count) * 100
                        print_and_log(f"  计算 WoW: ({increasing_count} / {valid_growth_count}) * 100 = {calculated_wow_index:.2f}", log_f)
                    else: print_and_log("  无有效环比增长率计算 WoW 指数。", log_f)
                else: print_and_log(f"  原始数据缺失过多，WoW 指数记为 NaN。", log_f)
            except Exception as e: print_and_log(f"计算环比指数时发生错误: {e}", log_f)

        # 对比 WoW 结果
        print_and_log(f"\nWoW 结果对比:", log_f)
        print_and_log(f"  手动计算 WoW: {calculated_wow_index:.2f}" if not np.isnan(calculated_wow_index) else "NaN", log_f)
        if 'WoW_DI' in pre_calculated_results.columns and verify_date in pre_calculated_results.index:
            pre_calculated_wow = pre_calculated_results.loc[verify_date, 'WoW_DI']
            print_and_log(f"  预计算 WoW: {pre_calculated_wow:.2f}" if not np.isnan(pre_calculated_wow) else "NaN", log_f)
            if (np.isnan(calculated_wow_index) and np.isnan(pre_calculated_wow)) or np.isclose(calculated_wow_index, pre_calculated_wow, atol=0.01, equal_nan=True):
                print_and_log("  WoW 对比: 一致", log_f)
            else: print_and_log("  WoW 对比: 不一致", log_f)
        else: print_and_log(f"  未在预计算结果中找到 {verify_date_str} 的 WoW_DI 数据。", log_f)

        # --- 5. 验证同比 (YoY) 扩散指数 ---
        print_and_log("\n" + "="*20 + f" 同比 (YoY) 验证: {verify_date_str} " + "="*20, log_f)
        calculated_yoy_index = np.nan # 初始化
        try:
            prev_year_date = verify_date - pd.Timedelta(weeks=52)
            if prev_year_date not in industry_data.index:
                available_prev_dates = industry_data.index[industry_data.index <= prev_year_date]
                if not available_prev_dates.empty:
                    prev_year_date_actual = available_prev_dates.max()
                    print_and_log(f"注意: 精确的52周前日期 {prev_year_date.strftime('%Y-%m-%d')} 不存在，使用最近的日期 {prev_year_date_actual.strftime('%Y-%m-%d')}", log_f)
                    prev_year_date = prev_year_date_actual
                else: raise IndexError("无52周前数据")

            current_values_yoy = industry_data.loc[verify_date]
            prev_year_values = industry_data.loc[prev_year_date]
            # a. 同比增长率
            yoy_growth_rates = (current_values_yoy - prev_year_values) / prev_year_values
            # b. 缺失比例检查 (基于原始数据)
            non_missing_ratio_yoy = current_values_yoy.notna().mean()
            print_and_log(f"缺失值检查 (原始数据 @ {verify_date_str}, 阈值 < {1-MISSING_THRESHOLD:.1f}): 非缺失比例 = {non_missing_ratio_yoy:.2f}", log_f)
            if non_missing_ratio_yoy >= MISSING_THRESHOLD:
                valid_growth_yoy = yoy_growth_rates.dropna()
                valid_growth_count_yoy = len(valid_growth_yoy)
                if valid_growth_count_yoy > 0:
                    increasing_count_yoy = (valid_growth_yoy > TOLERANCE_THRESHOLD).sum()
                    calculated_yoy_index = (increasing_count_yoy / valid_growth_count_yoy) * 100
                    print_and_log(f"  计算 YoY: ({increasing_count_yoy} / {valid_growth_count_yoy}) * 100 = {calculated_yoy_index:.2f}", log_f)
                else: print_and_log("  无有效同比增长率计算 YoY 指数。", log_f)
            else: print_and_log(f"  原始数据缺失过多，YoY 指数记为 NaN。", log_f)
        except IndexError as e: print_and_log(f"计算同比指数时出错: {e}", log_f)
        except Exception as e: print_and_log(f"计算同比指数时发生意外错误: {e}", log_f)

        # 对比 YoY 结果
        print_and_log(f"\nYoY 结果对比:", log_f)
        print_and_log(f"  手动计算 YoY: {calculated_yoy_index:.2f}" if not np.isnan(calculated_yoy_index) else "NaN", log_f)
        if 'YoY_DI' in pre_calculated_results.columns and verify_date in pre_calculated_results.index:
            pre_calculated_yoy = pre_calculated_results.loc[verify_date, 'YoY_DI']
            print_and_log(f"  预计算 YoY: {pre_calculated_yoy:.2f}" if not np.isnan(pre_calculated_yoy) else "NaN", log_f)
            if (np.isnan(calculated_yoy_index) and np.isnan(pre_calculated_yoy)) or np.isclose(calculated_yoy_index, pre_calculated_yoy, atol=0.01, equal_nan=True):
                print_and_log("  YoY 对比: 一致", log_f)
            else: print_and_log("  YoY 对比: 不一致", log_f)
        else: print_and_log(f"  未在预计算结果中找到 {verify_date_str} 的 YoY_DI 数据。", log_f)

        # --- 6. 验证 Mix 扩散指数 ---
        print_and_log("\n" + "="*20 + f" MixDI 验证: {verify_date_str} " + "="*20, log_f)
        calculated_mix_index = np.nan # 初始化
        try:
            # 1. 替换 0 为 NaN
            industry_data_no_zero = industry_data.replace(0, np.nan)
            # 2. 计算同比变化
            yoy_relative = industry_data_no_zero.pct_change(periods=52, fill_method=None)
            # 3. 计算同比变化的4周环比变化
            combined_changes = yoy_relative.pct_change(periods=4, fill_method=None)
            
            # 检查是否能获取到当前日期的组合变化数据
            if verify_date in combined_changes.index:
                current_combined_changes = combined_changes.loc[verify_date]
                # 4. 检查组合变化率的缺失比例
                combined_missing_ratio = current_combined_changes.isna().mean()
                print_and_log(f"缺失值检查 (组合变化率 @ {verify_date_str}, 阈值 <= {COMBINED_MISSING_THRESHOLD:.2f}): 缺失比例 = {combined_missing_ratio:.2f}", log_f)
                if combined_missing_ratio <= COMBINED_MISSING_THRESHOLD:
                    valid_combined = current_combined_changes.dropna()
                    total_valid_combined = len(valid_combined)
                    if total_valid_combined > 0:
                        # 5. 计算增长指标数量
                        improvements_combined = (valid_combined > TOLERANCE_THRESHOLD).sum()
                        mix_di_value = (improvements_combined / total_valid_combined) * 100
                        calculated_mix_index = round(mix_di_value, 2) # 与计算脚本保持一致，四舍五入
                        print_and_log(f"  计算 MixDI: ({improvements_combined} / {total_valid_combined}) * 100 = {calculated_mix_index:.2f}", log_f)
                    else: print_and_log("  无有效组合变化率计算 MixDI 指数。", log_f)
                else: print_and_log(f"  组合变化率缺失过多，MixDI 指数记为 NaN。", log_f)
            else:
                print_and_log(f"  无法获取 {verify_date_str} 的组合变化率数据 (可能因前期数据不足)。", log_f)

        except Exception as e:
            print_and_log(f"计算 MixDI 时发生错误: {e}", log_f)

        # 对比 MixDI 结果
        print_and_log(f"\nMixDI 结果对比:", log_f)
        print_and_log(f"  手动计算 MixDI: {calculated_mix_index:.2f}" if not np.isnan(calculated_mix_index) else "NaN", log_f)
        if 'Mix_DI' in pre_calculated_results.columns and verify_date in pre_calculated_results.index:
            pre_calculated_mix = pre_calculated_results.loc[verify_date, 'Mix_DI']
            print_and_log(f"  预计算 MixDI: {pre_calculated_mix:.2f}" if not np.isnan(pre_calculated_mix) else "NaN", log_f)
            # MixDI 在计算时已 round(2)，比较时也需要考虑
            if (np.isnan(calculated_mix_index) and np.isnan(pre_calculated_mix)) or \
               (not np.isnan(calculated_mix_index) and not np.isnan(pre_calculated_mix) and \
                abs(calculated_mix_index - pre_calculated_mix) < 0.001): # 比较 round后的值
                print_and_log("  MixDI 对比: 一致", log_f)
            else: print_and_log("  MixDI 对比: 不一致", log_f)
        else: print_and_log(f"  未在预计算结果中找到 {verify_date_str} 的 Mix_DI 数据。", log_f)

        print_and_log("\n--- 验证结束 ---", log_f)

if __name__ == "__main__":
    # 设置阈值
    # TOLERANCE_THRESHOLD = 0.01 # 已在函数内定义
    # COMBINED_MISSING_THRESHOLD = 0.33 # 已在函数内定义
    # 
    # 修改目标行业和输出文件 
    TARGET_INDUSTRY = "黑色金属冶炼和压延加工业"
    OUTPUT_FILE = "black_metal_combined_verification.txt" 
    # 
    # 你可以在这里指定一个特定日期 'YYYY-MM-DD'，或者留空使用最新日期
    VERIFICATION_DATE = None 

    verify_diffusion_for_industry(TARGET_INDUSTRY, VERIFICATION_DATE, OUTPUT_FILE)
    # 添加最后打印文件路径的消息
    print(f"\n详细验证过程已写入文件: {OUTPUT_FILE}") 