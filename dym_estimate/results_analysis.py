# -*- coding: utf-8 -*-
"""
结果分析、保存和绘图相关函数
"""
import pandas as pd
import numpy as np
import os
import matplotlib
matplotlib.use('Agg') # Ensure Agg backend is used before importing pyplot
import matplotlib.pyplot as plt
import matplotlib.dates as mdates # <-- Add this import
import seaborn as sns
import unicodedata
from typing import Tuple, List, Dict, Union, Any # Added Any
from sklearn.metrics import mean_squared_error, mean_absolute_error

# 尝试导入 openpyxl，如果 Excel 写入需要它 (虽然 pandas 会处理)
# try:
#     import openpyxl
# except ImportError:
#     print("警告: openpyxl 未安装，Excel (.xlsx) 文件写入可能失败。请运行 'pip install openpyxl'")

def analyze_and_save_final_results(
    run_output_dir: str,
    timestamp_str: str,
    excel_output_path: str,
    all_data_full: pd.DataFrame, # 用于获取原始目标序列
    data_for_analysis: Dict[str, Any], # 包含 'final_data_processed', 'final_target_mean_rescale', 'final_target_std_rescale'
    target_variable: str,
    final_dfm_results: Any, # DFM 结果对象, 包含 'x_sm', 'Lambda'
    best_variables: List[str],
    best_params: Dict[str, Any],
    var_type_map: Dict[str, str], # 可能需要用于最终变量列表的注释
    best_avg_hit_rate_tuning: float,
    best_avg_mae_tuning: float, # 实际上是 RMSE
    total_runtime_seconds: float,
    validation_start_date: str,
    validation_end_date: str,
    train_end_date: str,
    heatmap_top_n_vars: int,     # <-- 新增
    factor_contributions: Dict[str, float] | None = None,
    final_transform_log: Dict[str, str] | None = None,
    pca_results_df: pd.DataFrame | None = None,
    contribution_results_df: pd.DataFrame | None = None,
    var_industry_map: Dict[str, str] | None = None,
    rolling_lambda_list: List[pd.DataFrame] | None = None # <-- 新增：滚动载荷列表
):
    """分析最终DFM结果并保存到Excel和图中。"""
    # print("\n--- [DEBUG] Entered analyze_and_save_final_results --- ")
    # print(f"  [DEBUG] Received run_output_dir: {run_output_dir}")
    # print(f"  [DEBUG] Received excel_output_path: {excel_output_path}")
    # print(f"  [DEBUG] final_dfm_results type: {type(final_dfm_results)}")
    # print(f"  [DEBUG] data_for_analysis keys: {list(data_for_analysis.keys()) if data_for_analysis else 'None'}")

    if not run_output_dir or not isinstance(run_output_dir, str):
        print(f"错误：无效的 run_output_dir 参数: {run_output_dir}")
        return
    try:
        os.makedirs(run_output_dir, exist_ok=True)
        # print(f"  [DEBUG] Output directory confirmed/created: {run_output_dir}")
    except OSError as e:
        print(f"错误：无法创建输出目录 \'{run_output_dir}\': {e}")
        return

    # 动态生成输出文件路径
    final_plot_file = os.path.join(run_output_dir, f"final_nowcast_comparison_{timestamp_str}.png")
    heatmap_file = os.path.join(run_output_dir, f"factor_loadings_heatmap_{timestamp_str}.png")
    combined_factor_plot_file = os.path.join(run_output_dir, f"all_factors_timeseries_{timestamp_str}.png")

    try:
        # 检查输入对象
        if not final_dfm_results or not hasattr(final_dfm_results, 'x_sm') or not hasattr(final_dfm_results, 'Lambda'):
            print("错误: analyze_and_save_final_results 缺少有效的 final_dfm_results 对象。")
            return
        if not data_for_analysis or \
           'final_data_processed' not in data_for_analysis or \
           'final_target_mean_rescale' not in data_for_analysis or \
           'final_target_std_rescale' not in data_for_analysis:
            print("错误: analyze_and_save_final_results 缺少有效的 data_for_analysis 对象。")
            return

        final_factors = final_dfm_results.x_sm
        final_loadings = final_dfm_results.Lambda
        final_data_processed = data_for_analysis['final_data_processed']
        target_mean = data_for_analysis['final_target_mean_rescale']
        target_std = data_for_analysis['final_target_std_rescale']
        final_k_factors = best_params.get('k_factors', 'N/A')

        if isinstance(final_k_factors, str):
            print("错误: best_params 中 k_factors 无效。")
            return

        print(f"分析参数: k_factors={final_k_factors}")

        # 获取目标载荷和 Lambda DataFrame
        lambda_df_final = None
        lambda_target = None
        try:
            target_var_index_pos = final_data_processed.columns.get_loc(target_variable)
            lambda_target = final_loadings[target_var_index_pos, :]
            lambda_df_final = pd.DataFrame(final_loadings, index=final_data_processed.columns, columns=[f'Factor{i+1}' for i in range(final_k_factors)])
        except (KeyError, IndexError) as e:
            print(f"错误: 获取目标载荷或创建 Lambda DataFrame 时出错: {e}")
            return

        # 计算最终 Nowcast
        if lambda_target is None or final_factors is None or not isinstance(final_factors, (pd.DataFrame, pd.Series)):
             print("错误: 无法计算 Nowcast，因子或目标载荷无效。")
             return
        nowcast_standardized = final_factors.to_numpy() @ lambda_target
        final_nowcast_orig = pd.Series(nowcast_standardized * target_std + target_mean, index=final_factors.index, name='Nowcast_Orig')
        print("最终反标准化 Nowcast 计算完成.")

        # 计算最终指标
        original_target_series = all_data_full[target_variable].copy()
        target_for_comparison = original_target_series.dropna()
        diff_label_for_metrics = " (原始水平)"
        print(f"目标序列用于比较 (尺度: {diff_label_for_metrics}) 准备完成。")
        common_index_final = final_nowcast_orig.index.intersection(target_for_comparison.index)

        final_is_rmse = np.nan
        final_oos_rmse = np.nan
        final_is_mae = np.nan
        final_oos_mae = np.nan
        hit_rate_train = np.nan
        hit_rate_validation = np.nan

        if common_index_final.empty:
            print("错误: 最终 Nowcast 和目标序列没有共同索引。无法计算最终指标。")
        else:
            aligned_df_final = pd.DataFrame({'Nowcast': final_nowcast_orig.loc[common_index_final], 'Target': target_for_comparison.loc[common_index_final]}).dropna()
            if aligned_df_final.empty:
                 print("错误: 对齐最终 Nowcast 和目标序列后数据为空。无法计算最终指标。")
            else:
                # OOS / Validation Period
                # 确保使用 tune_dfm.py 中定义的常量或传递这些日期
                # !! 注意: 这些常量不在此文件作用域内，理想情况下应作为参数传递 !!
                # 从 best_params 或其他地方获取？ 暂时硬编码（不推荐）
                # VALIDATION_START_DATE = '2024-07-05' # Placeholder - Should be passed
                # VALIDATION_END_DATE = '2024-12-27' # Placeholder - Should be passed
                # TRAIN_END_DATE = '2024-06-28'      # Placeholder - Should be passed

                validation_df_final = aligned_df_final.loc[validation_start_date:validation_end_date]
                if not validation_df_final.empty and len(validation_df_final) > 1:
                    try:
                        final_oos_rmse = np.sqrt(mean_squared_error(validation_df_final['Target'], validation_df_final['Nowcast']))
                        final_oos_mae = mean_absolute_error(validation_df_final['Target'], validation_df_final['Nowcast'])
                        print(f"  最终 OOS RMSE (验证期): {final_oos_rmse:.6f}")
                        print(f"  最终 OOS MAE (验证期): {final_oos_mae:.6f}")
                        changes_df_val = validation_df_final.diff().dropna()
                        if not changes_df_val.empty:
                            correct_direction_val = (np.sign(changes_df_val['Nowcast']) == np.sign(changes_df_val['Target'])) & (changes_df_val['Target'] != 0)
                            non_zero_target_changes_val = (changes_df_val['Target'] != 0).sum()
                            if non_zero_target_changes_val > 0:
                                hit_rate_validation = correct_direction_val.sum() / non_zero_target_changes_val * 100
                                print(f"  验证期 Hit Rate (%): {hit_rate_validation:.2f} (基于 {non_zero_target_changes_val} 个非零变化点)")
                    except Exception as e_oos: print(f"计算 OOS 指标时出错: {e_oos}")
                else: print("警告: 验证期数据不足或无效，无法计算 OOS 指标。")

                # IS / Training Period
                train_df_final = aligned_df_final.loc[:train_end_date]
                if not train_df_final.empty and len(train_df_final) > 1:
                     try:
                        final_is_rmse = np.sqrt(mean_squared_error(train_df_final['Target'], train_df_final['Nowcast']))
                        final_is_mae = mean_absolute_error(train_df_final['Target'], train_df_final['Nowcast'])
                        print(f"  最终 IS RMSE (训练期): {final_is_rmse:.6f}")
                        print(f"  最终 IS MAE (训练期): {final_is_mae:.6f}")
                        changes_df_train = train_df_final.diff().dropna()
                        if not changes_df_train.empty:
                            correct_direction_train = (np.sign(changes_df_train['Nowcast']) == np.sign(changes_df_train['Target'])) & (changes_df_train['Target'] != 0)
                            non_zero_target_changes_train = (changes_df_train['Target'] != 0).sum()
                            if non_zero_target_changes_train > 0:
                                hit_rate_train = correct_direction_train.sum() / non_zero_target_changes_train * 100
                                print(f"  训练期 Hit Rate (%): {hit_rate_train:.2f} (基于 {non_zero_target_changes_train} 个非零变化点)")
                     except Exception as e_is: print(f"计算 IS 指标时出错: {e_is}")
                else: print("警告: 训练期数据不足或无效，无法计算 IS 指标。")

        # 准备 Excel 输出
        # --- 修改：为每个指标创建格式化字符串 --- 
        fmt_is_rmse = f"{final_is_rmse:.6f}" if pd.notna(final_is_rmse) else 'N/A'
        fmt_is_mae = f"{final_is_mae:.6f}" if pd.notna(final_is_mae) else 'N/A'
        fmt_hit_train = f"{hit_rate_train:.2f}" if pd.notna(hit_rate_train) else 'N/A'
        fmt_oos_rmse = f"{final_oos_rmse:.6f}" if pd.notna(final_oos_rmse) else 'N/A'
        fmt_oos_mae = f"{final_oos_mae:.6f}" if pd.notna(final_oos_mae) else 'N/A'
        fmt_hit_val = f"{hit_rate_validation:.2f}" if pd.notna(hit_rate_validation) else 'N/A'
        fmt_avg_hit_tuning = f"{best_avg_hit_rate_tuning:.2f}%" if pd.notna(best_avg_hit_rate_tuning) else 'N/A'
        fmt_avg_mae_tuning = f"{best_avg_mae_tuning:.6f}" if pd.notna(best_avg_mae_tuning) else 'N/A'
        # --- 结束修改 --- 

        interpretation_text = (
            f"最终分析总结 (Run: {timestamp_str}):\n"
            f"- Target Variable: {target_variable}\n"
            f"- Final Variables Count: {len(best_variables)}\n"
            f"- Best k_factors (Final Model): {final_k_factors}\n"
            f"- Best Avg Hit Rate (Tuning): {fmt_avg_hit_tuning}\n" # <-- 使用格式化变量
            f"- Corresponding Avg RMSE (Tuning): {fmt_avg_mae_tuning}\n" # <-- 使用格式化变量
            f"- Final IS RMSE{diff_label_for_metrics}: {fmt_is_rmse}\n" # <-- 使用格式化变量
            f"- Final IS MAE{diff_label_for_metrics}: {fmt_is_mae}\n" # <-- 使用格式化变量
            f"- Hit Rate (Train %){diff_label_for_metrics}: {fmt_hit_train}\n" # <-- 使用格式化变量
            f"- Final OOS RMSE{diff_label_for_metrics}: {fmt_oos_rmse}\n" # <-- 使用格式化变量
            f"- Final OOS MAE{diff_label_for_metrics}: {fmt_oos_mae}\n" # <-- 使用格式化变量
            f"- Hit Rate (Validation %){diff_label_for_metrics}: {fmt_hit_val}\n" # <-- 使用格式化变量
            f"- Total Runtime (s): {total_runtime_seconds:.2f}\n"
        )

        print(f"\n写入 Excel 文件: {excel_output_path}...")
        # print("DEBUG: About to initialize ExcelWriter...")
        excel_writer_initialized = False
        try:
            with pd.ExcelWriter(excel_output_path, engine='openpyxl') as writer:
                # print("DEBUG: ExcelWriter initialized successfully.")
                excel_writer_initialized = True
                # Sheet: Summary_Overview
                # print("DEBUG: Attempting to write Summary_Overview sheet...")
                try:
                    summary_params = [
                        'Final Variables Count', 'Best k_factors (Final Model)',
                        'Best Avg Hit Rate (Tuning %)', 'Corresponding Avg RMSE (Tuning)',
                        f'Final IS RMSE{diff_label_for_metrics}', f'Final IS MAE{diff_label_for_metrics}',
                        f'Hit Rate (Train %){diff_label_for_metrics}', f"Final OOS RMSE{diff_label_for_metrics}",
                        f"Final OOS MAE{diff_label_for_metrics}", f'Hit Rate (Validation %){diff_label_for_metrics}', # <-- Fix typo: removed extra quote
                        'Total Runtime (s)',
                    ]
                    summary_values = [
                        len(best_variables), final_k_factors,
                        fmt_avg_hit_tuning, fmt_avg_mae_tuning,
                        fmt_is_rmse, fmt_is_mae,
                        fmt_hit_train, fmt_oos_rmse,
                        fmt_oos_mae, fmt_hit_val,
                        f"{total_runtime_seconds:.2f}",
                    ]
                    summary_df = pd.DataFrame({'Parameter': summary_params, 'Value': summary_values})
                    summary_df.to_excel(writer, sheet_name='Summary_Overview', index=False)
                    current_row = writer.sheets['Summary_Overview'].max_row

                    # 追加 Analysis Text
                    analysis_lines = interpretation_text.strip().split('\n')
                    analysis_df = pd.DataFrame({'Analysis Text': analysis_lines})
                    analysis_df.to_excel(writer, sheet_name='Summary_Overview', startrow=current_row + 2, index=False, header=True)
                    current_row = writer.sheets['Summary_Overview'].max_row
                    print("  Summary_Overview 和 Analysis_Text 已写入 Sheet: 'Summary_Overview'")

                    # 追加 PCA 结果
                    if pca_results_df is not None and not pca_results_df.empty:
                        start_row_pca = current_row + 2
                        pd.DataFrame([["PCA Explained Variance"]]).to_excel(writer, sheet_name='Summary_Overview', startrow=start_row_pca-1, index=False, header=False)
                        pca_results_df.to_excel(writer, sheet_name='Summary_Overview', startrow=start_row_pca, index=False, header=True)
                        current_row = writer.sheets['Summary_Overview'].max_row
                        print("  PCA 结果已追加写入 Sheet: 'Summary_Overview'")

                    # 追加因子贡献度结果
                    if contribution_results_df is not None and not contribution_results_df.empty:
                        start_row_contrib = current_row + 2
                        pd.DataFrame([["Factor Contribution to Target Variance"]]).to_excel(writer, sheet_name='Summary_Overview', startrow=start_row_contrib-1, index=False, header=False)
                        contribution_results_df.to_excel(writer, sheet_name='Summary_Overview', startrow=start_row_contrib, index=False, header=True)
                        current_row = writer.sheets['Summary_Overview'].max_row
                        print("  因子贡献度结果已追加写入 Sheet: 'Summary_Overview'")
                        # print("DEBUG: Successfully wrote Summary_Overview additions (PCA, Contrib).")
                except Exception as e:
                    print(f"写入 Summary_Overview (或追加 PCA/Contrib) 时出错: {e}")
                    # print("DEBUG: FAILED writing Summary_Overview additions.")

                # Sheet: Final_Selected_Variables
                # print("DEBUG: Attempting to write Final_Selected_Variables sheet...")
                try:
                    vars_df = pd.DataFrame(best_variables, columns=['Variable Name'])
                    # 添加转换信息
                    if final_transform_log:
                        vars_df['Transform'] = vars_df['Variable Name'].map(final_transform_log).fillna('N/A')
                    # 添加行业信息
                    if var_industry_map:
                        # Normalize keys for mapping
                        normalized_industry_map = {unicodedata.normalize('NFKC', str(k)).strip().lower(): v
                                                 for k, v in var_industry_map.items()}
                        vars_df['Industry'] = vars_df['Variable Name'].apply(
                            lambda x: normalized_industry_map.get(unicodedata.normalize('NFKC', str(x)).strip().lower(), 'Unknown')
                        ).fillna('Unknown')
                    # 添加最终载荷
                    if lambda_df_final is not None:
                        vars_df = vars_df.merge(lambda_df_final, left_on='Variable Name', right_index=True, how='left')
                    vars_df.to_excel(writer, sheet_name='Final_Selected_Variables', index=False)
                    print("  最终变量列表、转换、行业和载荷已写入 Sheet: 'Final_Selected_Variables'")
                    # print("DEBUG: Successfully wrote Final_Selected_Variables sheet.")
                except Exception as e:
                    print(f"写入 Final_Selected_Variables 时出错: {e}")
                    # print("DEBUG: FAILED writing Final_Selected_Variables sheet.")

                # Sheet: Final_Factors_Timeseries
                # print("DEBUG: Attempting to write Final_Factors_Timeseries sheet...")
                try:
                    if final_factors is not None and not final_factors.empty:
                         final_factors.to_excel(writer, sheet_name='Final_Factors_Timeseries')
                         print("  最终因子时间序列已写入 Sheet: 'Final_Factors_Timeseries'")
                         # print("DEBUG: Successfully wrote Final_Factors_Timeseries sheet.")
                    else:
                         print("  警告: 最终因子数据为空，无法写入 Final_Factors_Timeseries Sheet.")
                         # print("DEBUG: SKIPPED writing Final_Factors_Timeseries sheet (empty data).")
                except Exception as e:
                    print(f"写入 Final_Factors_Timeseries 时出错: {e}")
                    # print("DEBUG: FAILED writing Final_Factors_Timeseries sheet.")

                # Sheet: Final_Nowcast_vs_Actual
                # print("DEBUG: Attempting to write Final_Nowcast_vs_Actual sheet...")
                try:
                    # 重新计算对齐的数据用于保存
                    if not common_index_final.empty:
                         aligned_df_final_save = pd.DataFrame({
                             'Date': common_index_final,
                             'Nowcast_Orig': final_nowcast_orig.loc[common_index_final],
                             'Actual_Orig': target_for_comparison.loc[common_index_final]
                         }).reset_index(drop=True)
                         # Format Date column
                         if isinstance(aligned_df_final_save['Date'].iloc[0], pd.Timestamp):
                              aligned_df_final_save['Date'] = aligned_df_final_save['Date'].dt.strftime('%Y-%m-%d')
                         aligned_df_final_save.to_excel(writer, sheet_name='Final_Nowcast_vs_Actual', index=False)
                         print("  最终 Nowcast vs Actual 已写入 Sheet: 'Final_Nowcast_vs_Actual'")
                         # print("DEBUG: Successfully wrote Final_Nowcast_vs_Actual sheet.")
                    else:
                         print("  警告: 无法保存最终 Nowcast vs Actual，因为没有共同索引。")
                         # print("DEBUG: SKIPPED writing Final_Nowcast_vs_Actual sheet (no common index).")
                except Exception as e:
                    print(f"写入 Final_Nowcast_vs_Actual 时出错: {e}")
                    # print("DEBUG: FAILED writing Final_Nowcast_vs_Actual sheet.")

                # --- 新增: 分析和保存滚动载荷 --- 
                # print("DEBUG: Attempting to write Rolling_Loadings sheets...")
                if rolling_lambda_list is not None and rolling_lambda_list:
                    print(f"  分析并写入滚动载荷数据 ({len(rolling_lambda_list)} 个窗口)...")
                    try:
                        # 添加窗口标识符并合并
                        all_rolling_loadings = []
                        for i, df in enumerate(rolling_lambda_list):
                            if isinstance(df, pd.DataFrame) and not df.empty:
                                df_copy = df.copy()
                                df_copy['window'] = i + 1
                                all_rolling_loadings.append(df_copy.reset_index().rename(columns={'index': 'Variable'}))
                            else:
                                print(f"    警告: 跳过第 {i+1} 个窗口的空或无效滚动载荷 DataFrame。")

                        if all_rolling_loadings:
                            combined_loadings_df = pd.concat(all_rolling_loadings, ignore_index=True)

                            # 将合并后的原始滚动载荷写入新 Sheet
                            combined_loadings_df.to_excel(writer, sheet_name='Rolling_Loadings_Raw', index=False)
                            print(f"    原始滚动载荷数据已写入 Sheet: 'Rolling_Loadings_Raw'")

                            # 计算统计量 (按变量和因子分组)
                            factor_cols = [col for col in combined_loadings_df.columns if col.startswith('Factor')]
                            if factor_cols:
                                 # 融化 DataFrame 以便按 Variable 和 Factor 计算统计量
                                 melted_loadings = combined_loadings_df.melt(
                                     id_vars=['Variable', 'window'],
                                     value_vars=factor_cols,
                                     var_name='Factor',
                                     value_name='Loading'
                                 )
                                 # 计算每个变量在每个因子上的均值和标准差
                                 loading_stats = melted_loadings.groupby(['Variable', 'Factor'])['Loading'].agg(['mean', 'std']).reset_index()

                                 # 将统计结果写入新 Sheet
                                 loading_stats.to_excel(writer, sheet_name='Rolling_Loadings_Stats', index=False)
                                 print(f"    滚动载荷统计 (均值/标准差) 已写入 Sheet: 'Rolling_Loadings_Stats'")
                                 # print("DEBUG: Successfully wrote Rolling_Loadings_Stats sheet.")
                            else:
                                 print("    警告: 在合并的滚动载荷中未找到因子列，无法计算统计量。")
                                 # print("DEBUG: SKIPPED writing Rolling_Loadings_Stats sheet (no factor columns).")
                        else:
                            print("   警告：无法合并滚动载荷列表或列表为空。")
                            # print("DEBUG: SKIPPED writing Rolling_Loadings_Raw sheet (empty list).")

                    except Exception as e_roll_lambda:
                        print(f"    处理或写入滚动载荷时出错: {e_roll_lambda}")
                        # print("DEBUG: FAILED writing Rolling_Loadings sheets.")
                else:
                    print("  信息: 未提供滚动载荷数据或数据为空，跳过滚动载荷分析。")
                    # print("DEBUG: SKIPPED writing Rolling_Loadings sheets (no data provided).")
                # --- 结束滚动载荷分析 --- 

            # 在 with 块结束时检查 writer 是否成功初始化
            if excel_writer_initialized:
                 # print("DEBUG: Excel file should be saved now (after with block).")
                 pass
            else:
                 # print("DEBUG: ExcelWriter was never initialized, file not saved.")
                 pass
            print(f"Excel 文件写入完成: {excel_output_path}") # 这行可能在失败时也打印，所以 Debug 信息更重要

        except Exception as e_excel:
            # print("DEBUG: EXCEL WRITING FAILED! (Outer try block)")
            print(f"写入 Excel 文件时发生严重错误: {e_excel}")
            import traceback
            traceback.print_exc()

        # --- 绘图 --- 
        # (注释掉所有绘图代码) # <-- 这行注释可以移除或保留
        # 1. 最终 Nowcast 图
        # print("DEBUG: Skipping final nowcast plot (commented out).") # <-- 保留注释或移除
        # --- 取消下面对 plot_final_nowcast 调用的注释 --- 
        if final_factors is not None and lambda_target is not None: # Check if factors and loadings exist
             try:
                 # 重新计算 nowcast 和对齐的数据用于绘图
                 nowcast_standardized_plot = final_factors.to_numpy() @ lambda_target
                 final_nowcast_orig_plot = pd.Series(nowcast_standardized_plot * target_std + target_mean, index=final_factors.index, name='Nowcast_Orig')
                 original_target_series_plot = all_data_full[target_variable].copy().dropna()
                 common_index_plot = final_nowcast_orig_plot.index.intersection(original_target_series_plot.index)

                 if not common_index_plot.empty:
                     plot_data_aligned = pd.DataFrame({
                         'Nowcast_Orig': final_nowcast_orig_plot.loc[common_index_plot],
                         target_variable: original_target_series_plot.loc[common_index_plot]
                     })

                     plot_title = f'最终 DFM Nowcast vs 实际值 ({target_variable} - Run {timestamp_str})'
                     plot_filename = os.path.join(run_output_dir, f"final_nowcast_comparison_{timestamp_str}.png")

                     # 调用绘图函数
                     plot_final_nowcast(
                         final_nowcast_series=plot_data_aligned['Nowcast_Orig'],
                         target_for_plot=plot_data_aligned[target_variable],
                         validation_start=validation_start_date, # 使用传入的参数
                         validation_end=validation_end_date,     # 使用传入的参数
                         title=plot_title,
                         filename=plot_filename
                     )
                 else:
                     print("警告: 无法为绘图准备对齐的数据 (无共同索引)。")
             except Exception as e_plot_prep:
                 print(f"准备绘图数据或调用绘图函数时出错: {e_plot_prep}")
        else:
             print("警告: 缺少因子或目标载荷，无法生成最终 Nowcast 图。")
        # --- 结束取消注释 ---

        # 2. 因子载荷热力图 (保持注释)
        # print("DEBUG: Skipping heatmap plot (commented out).")
        # ... (代码保持注释)

        # 3. 因子时间序列图 (保持注释)
        # print("DEBUG: Skipping combined factors plot (commented out).")
        # ... (代码保持注释)

    except Exception as e_analyze:
        print(f"在 analyze_and_save_final_results 函数中发生意外错误: {e_analyze}")
        import traceback
        traceback.print_exc()

# 绘图函数 (依赖项移到本文件顶部)
def plot_final_nowcast(
    final_nowcast_series: pd.Series,
    target_for_plot: pd.Series,
    validation_start: str | pd.Timestamp,
    validation_end: str | pd.Timestamp,
    title: str,
    filename: str
):
    """绘制最终的周度 nowcast 与实际观测值的对比图（原始水平）。"""
    print("\n生成最终 Nowcasting 图 (原始水平 - 绘制完整预测)...")
    try:
        # --- 修改: 确保 final_nowcast_series 和 target_for_plot 索引为 DatetimeIndex --- 
        try:
            if not isinstance(final_nowcast_series.index, pd.DatetimeIndex):
                final_nowcast_series.index = pd.to_datetime(final_nowcast_series.index)
            if not isinstance(target_for_plot.index, pd.DatetimeIndex):
                target_for_plot.index = pd.to_datetime(target_for_plot.index)
        except Exception as e_index_conv:
             print(f"警告: 将索引转换为 DatetimeIndex 时出错: {e_index_conv}")
             # 如果转换失败，绘图可能出错或不准确
        # --- 结束修改 --- 

        nowcast_col_name = 'Nowcast_Orig'
        target_col_name = target_for_plot.name if target_for_plot.name is not None else 'Actual'
        if target_col_name == nowcast_col_name: target_col_name = 'Observed_Value'

        plot_df = final_nowcast_series.to_frame(name=nowcast_col_name)
        plot_df = plot_df.join(target_for_plot.rename(target_col_name), how='left')

        # 屏蔽 1/2 月实际值
        if target_col_name in plot_df.columns and isinstance(plot_df.index, pd.DatetimeIndex):
            month_indices_plot = plot_df.index.month
            plot_df.loc[(month_indices_plot == 1) | (month_indices_plot == 2), target_col_name] = np.nan

        if not plot_df.empty:
            plt.figure(figsize=(14, 7))
            nowcast_label = '周度 Nowcast (原始水平)'
            actual_label = '观测值 (原始水平, 屏蔽1/2月)'
            ylabel = '值 (原始水平)'

            plt.plot(plot_df.index, plot_df[nowcast_col_name], label=nowcast_label, linestyle='-', alpha=0.8, color='blue')
            if target_col_name in plot_df.columns:
                plt.plot(plot_df.index, plot_df[target_col_name], label=actual_label, marker='o', linestyle='None', markersize=4, color='red')

            try:
                val_start_dt = pd.to_datetime(validation_start)
                val_end_dt = pd.to_datetime(validation_end)
                plt.axvspan(val_start_dt, val_end_dt, color='yellow', alpha=0.2, label='验证期')
            except Exception as date_err:
                print(f"警告：标记验证期时出错 - {date_err}")

            plt.title(title)
            plt.xlabel('日期')
            plt.ylabel(ylabel)
            plt.legend()
            plt.grid(True, linestyle='--', alpha=0.6)
            plt.tight_layout()
            plt.savefig(filename)
            plt.close()
            print(f"最终 Nowcasting 图已保存到: {filename}")
        else:
             print("错误：无法准备用于绘图的数据。")
    except Exception as e:
        print(f"生成或保存最终 Nowcasting 图时出错: {e}")

# (The invalid tag should be removed entirely after this line) 