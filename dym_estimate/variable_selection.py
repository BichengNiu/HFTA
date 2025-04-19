# -*- coding: utf-8 -*-
"""
包含 DFM 变量选择相关功能的模块。
"""
import pandas as pd
import numpy as np
import concurrent.futures
from tqdm import tqdm
import unicodedata
from typing import List, Dict, Tuple, Callable, Optional

# 假设 evaluate_dfm_params 函数可以从某个地方导入
# from .dfm_core import evaluate_dfm_params 
# 或者它被作为参数传递进来

def perform_backward_selection(
    initial_variables: List[str],
    initial_params: Dict,
    initial_score_tuple: Tuple, # (hit_rate, -k, -rmse)
    blocks: Dict[str, List[str]],
    all_data: pd.DataFrame,
    target_variable: str,
    hyperparams_to_tune: List[Dict],
    var_type_map: Dict[str, str],
    validation_start: str,
    validation_end: str,
    target_freq: str,
    train_end_date: str,
    n_iter: int,
    target_mean_original: float,
    target_std_original: float,
    max_workers: int,
    evaluate_dfm_func: Callable, # 传递评估函数
    rmse_tolerance_factor: float = 1.0, # RMSE 容忍度因子
    log_file: Optional[object] = None # 可选的日志文件句柄
) -> Tuple[List[str], Dict, Tuple, int, int]:
    """
    执行分块向后变量剔除。

    Args:
        initial_variables: 初始变量列表。
        initial_params: 初始最佳参数。
        initial_score_tuple: 初始最佳得分元组 (hit_rate, -k, -rmse)。
        blocks: 变量块字典。
        all_data: 包含所有变量的完整 DataFrame。
        target_variable: 目标变量名称。
        hyperparams_to_tune: 要测试的超参数组合列表。
        var_type_map: 变量类型映射。
        validation_start: 验证期开始日期。
        validation_end: 验证期结束日期。
        target_freq: 目标频率。
        train_end_date: 训练期结束日期。
        n_iter: DFM 迭代次数。
        target_mean_original: 原始目标变量均值 (用于反标准化)。
        target_std_original: 原始目标变量标准差 (用于反标准化)。
        max_workers: 并行计算的最大进程数。
        evaluate_dfm_func: 用于评估 DFM 参数的函数。
        rmse_tolerance_factor: 允许 RMSE 增加的最大比例因子 (例如 1.01 表示允许 1% 的增加)。
        log_file: 可选的日志文件句柄。

    Returns:
        Tuple 包含:
            - final_variables: 最终选择的变量列表。
            - final_params: 最终选择的最佳参数。
            - final_score_tuple: 最终的最佳得分元组。
            - total_evaluations: 此过程中执行的评估总次数。
            - svd_error_count: 此过程中遇到的 SVD 错误次数。
    """
    current_best_variables = initial_variables.copy()
    current_best_score_tuple = initial_score_tuple
    current_best_params = initial_params
    total_evaluations_selection = 0
    svd_error_count_selection = 0

    print(f"开始向后剔除，初始得分: HR={current_best_score_tuple[0]:.2f}%, K={-current_best_score_tuple[1]}, RMSE={-current_best_score_tuple[2]:.6f}")

    for block_name, block_vars_list in tqdm(blocks.items(), desc="处理变量块", unit="block"):
        print(f"\n--- 处理块: 	'{block_name}	' (初始 {len(block_vars_list)} 变量) ---")

        if len(block_vars_list) <= 2: 
            print(f"块 	'{block_name}	' 变量数 ({len(block_vars_list)}) <= 2，跳过剔除。")
            continue

        block_stable = False
        while not block_stable:
            best_candidate_score_tuple_this_iter = (-np.inf, np.inf, np.inf) # (hit_rate, -k, -rmse)
            best_removal_candidate_this_iter = None

            eligible_vars_in_block = [v for v in block_vars_list if v in current_best_variables and v != target_variable]
            if not eligible_vars_in_block:
                print(f"块 	'{block_name}	' 中无更多可剔除变量。")
                block_stable = True
                break

            print(f"  评估从块 	'{block_name}	' 移除 {len(eligible_vars_in_block)} 个候选变量...")
            futures_removal = []
            with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
                for candidate_var_to_remove in eligible_vars_in_block:
                    temp_variables = [v for v in current_best_variables if v != candidate_var_to_remove]
                    if len(temp_variables) <= 1: 
                        continue 

                    for params in hyperparams_to_tune:
                        if params.get('k_factors', 1) >= len(temp_variables):
                            continue

                        future = executor.submit(
                            evaluate_dfm_func, # Use the passed function
                            variables=temp_variables, # Pass the list including target
                            full_data=all_data,
                            target_variable=target_variable,
                            params=params,
                            var_type_map=var_type_map,
                            validation_start=validation_start,
                            validation_end=validation_end,
                            target_freq=target_freq,
                            train_end_date=train_end_date, 
                            max_iter=n_iter,
                            target_mean_original=target_mean_original,
                            target_std_original=target_std_original,
                        )
                        futures_removal.append({
                            'future': future,
                            'params': params,
                            'removed_var': candidate_var_to_remove,
                            'remaining_vars': temp_variables.copy()
                        })

            results_this_iteration = []
            if not futures_removal: 
                print(f"块 	'{block_name}	' 无有效评估任务可提交，此块完成。")
                block_stable = True
                continue
                
            for f_info in concurrent.futures.as_completed([f['future'] for f in futures_removal]):
                context = next(item for item in futures_removal if item['future'] == f_info)
                total_evaluations_selection += 1
                try:
                    result_tuple = f_info.result()
                    if len(result_tuple) != 8:
                        print(f"警告: evaluate_dfm_func 返回了 {len(result_tuple)} 个值，预期为 8。跳过此结果。 ({context['removed_var']})")
                        continue
                    is_rmse, oos_rmse, is_mae, oos_mae, is_hit_rate, oos_hit_rate, is_svd_error, lambda_df = result_tuple
                    
                    if is_svd_error:
                        svd_error_count_selection += 1
                    
                    combined_rmse_removal = np.inf
                    if np.isfinite(is_rmse) and np.isfinite(oos_rmse): combined_rmse_removal = 0.5 * is_rmse + 0.5 * oos_rmse
                    elif np.isfinite(is_rmse): combined_rmse_removal = is_rmse
                    elif np.isfinite(oos_rmse): combined_rmse_removal = oos_rmse
            
                    combined_hit_rate_removal = -np.inf
                    valid_hit_rates_rem = []
                    if np.isfinite(is_hit_rate): valid_hit_rates_rem.append(is_hit_rate)
                    if np.isfinite(oos_hit_rate): valid_hit_rates_rem.append(oos_hit_rate)
                    if valid_hit_rates_rem: combined_hit_rate_removal = np.mean(valid_hit_rates_rem)
                        
                    results_this_iteration.append({
                        'combined_rmse': combined_rmse_removal,
                        'combined_hit_rate': combined_hit_rate_removal,
                        'params': context['params'],
                        'removed_var': context['removed_var'],
                        'is_mae': is_mae,
                        'oos_mae': oos_mae
                    })
                except Exception as exc:
                    print(f"处理块 {block_name}, 尝试移除 {context['removed_var']} 时出错: {exc}")

            # --- 添加详细日志 --- 
            print(f"  块 	'{block_name}	' 本轮评估完成，比较 {len(results_this_iteration)} 个移除结果与当前最佳 {current_best_score_tuple}...")
            # --- 结束添加详细日志 --- 

            for result in results_this_iteration:
                if np.isfinite(result['combined_hit_rate']) and np.isfinite(result['combined_rmse']) and 'k_factors' in result['params']:
                    current_k = result['params']['k_factors']
                    if np.isfinite(current_k):
                        score_tuple_for_result = (result['combined_hit_rate'], -current_k, -result['combined_rmse'])
                        
                        # --- 添加详细日志: 打印每个候选移除的得分 --- 
                        print(f"    -> 尝试移除 '{result['removed_var']}', 参数: {result['params']}, 得分: {score_tuple_for_result} (HR={result['combined_hit_rate']:.2f}%, K={current_k}, RMSE={result['combined_rmse']:.6f})")
                        # --- 结束添加详细日志 --- 

                        if score_tuple_for_result > best_candidate_score_tuple_this_iter:
                            best_candidate_score_tuple_this_iter = score_tuple_for_result
                            best_removal_candidate_this_iter = result

            accept_removal = False
            reason_for_update = ""
            if best_removal_candidate_this_iter:
                cand_hr, cand_neg_k, cand_neg_rmse = best_candidate_score_tuple_this_iter
                curr_hr, curr_neg_k, curr_neg_rmse = current_best_score_tuple
                
                rmse_within_tolerance = False
                if curr_neg_rmse < 0: # Ensure current positive RMSE is > 0
                     rmse_within_tolerance = (-cand_neg_rmse <= (-curr_neg_rmse * rmse_tolerance_factor))
                elif curr_neg_rmse == 0 and cand_neg_rmse == 0:
                     rmse_within_tolerance = True # Both zero, so within tolerance
                
                if cand_hr > curr_hr:
                    accept_removal = True
                    reason_for_update = f"HR 提升: {cand_hr:.2f}% > {curr_hr:.2f}%"
                elif cand_hr == curr_hr and cand_neg_k > curr_neg_k:
                    accept_removal = True
                    reason_for_update = f"HR 持平, K 减少: {-cand_neg_k} < {-curr_neg_k}"
                elif cand_hr == curr_hr and cand_neg_k == curr_neg_k and rmse_within_tolerance:
                     accept_removal = True
                     reason_for_update = f"HR/K 持平, RMSE 在容忍 ({rmse_tolerance_factor*100:.1f}%) 范围内: {-cand_neg_rmse:.6f} vs {-curr_neg_rmse:.6f}"
                 
            if accept_removal and best_removal_candidate_this_iter:
                variable_to_remove = best_removal_candidate_this_iter['removed_var']
                params_for_best_removal = best_removal_candidate_this_iter['params']
                current_best_score_tuple = best_candidate_score_tuple_this_iter
                current_best_params = params_for_best_removal
                current_best_variables = next(item['remaining_vars'] for item in futures_removal if item['removed_var'] == variable_to_remove and item['params'] == params_for_best_removal)

                if variable_to_remove in block_vars_list:
                    block_vars_list.remove(variable_to_remove)

                new_hr, new_neg_k, new_neg_rmse = current_best_score_tuple
                print(f"*** 块 	'{block_name}	' 找到改进: 移除 	'{variable_to_remove}	' ({reason_for_update}), 新最佳分数: HR={new_hr:.2f}%, K={-new_neg_k}, RMSE={-new_neg_rmse:.6f}, Params: {current_best_params} ***")

                if log_file and not log_file.closed:
                    try:
                        log_file.write("\n" + "-"*35 + "\n")
                        log_file.write(f"--- 块 	'{block_name}	': 变量剔除结果 ---\
")
                        log_file.write(f"剔除变量: 	'{variable_to_remove}	' (原因: {reason_for_update})\
")
                        log_file.write(f"当前变量组 ({len(current_best_variables)}): {current_best_variables}\
")
                        log_file.write(f"最佳参数: {current_best_params}\
")
                        log_file.write(f"新最佳得分 (HR, -K, -RMSE): {current_best_score_tuple[0]:.2f}, {current_best_score_tuple[1]}, {current_best_score_tuple[2]:.6f}\
")
                        log_file.write("-"*35 + "\n")
                    except Exception as log_e:
                        print(f"写入块 	'{block_name}	' 剔除日志时出错: {log_e}")
            else:
                stop_hr, stop_neg_k, stop_neg_rmse = current_best_score_tuple
                print(f"块 	'{block_name}	' 内无变量移除可获得严格更优解 (当前最佳: HR={stop_hr:.2f}%, K={-stop_neg_k}, RMSE={-stop_neg_rmse:.6f})。此块完成。")
                block_stable = True
                if log_file and not log_file.closed and eligible_vars_in_block:
                     try:
                         log_file.write("\n" + "-"*35 + "\n")
                         log_file.write(f"--- 块 	'{block_name}	': 停止剔除 ---\
")
                         log_file.write(f"原因: 块内剩余变量移除无法找到严格更优的评分元组 (Benchmark HR={stop_hr:.2f}%, K={-stop_neg_k}, RMSE={-stop_neg_rmse:.6f})。\
")
                         log_file.write("-"*35 + "\n")
                     except Exception as log_e:
                        print(f"写入块 	'{block_name}	' 停止日志时出错: {log_e}")

    print("\n--- 所有块处理完毕 --- ")
    final_variables = current_best_variables.copy()
    final_params = current_best_params
    final_score_tuple = current_best_score_tuple
    final_hr, final_neg_k, final_neg_rmse = final_score_tuple
    print(f"最终变量数量: {len(final_variables)}")
    print(f"最终最佳平均胜率 (IS+OOS)/2: {final_hr:.2f}%")
    print(f"对应的因子数: {-final_neg_k}")
    print(f"对应的最终最佳平均 RMSE (IS+OOS)/2: {-final_neg_rmse:.6f}")
    print(f"最终最佳参数: {final_params}")

    return final_variables, final_params, final_score_tuple, total_evaluations_selection, svd_error_count_selection 