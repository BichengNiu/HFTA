# -*- coding: utf-8 -*-
"""
包含 DFM 变量选择相关功能的模块。
"""
import pandas as pd
import numpy as np
import concurrent.futures
from tqdm import tqdm
import unicodedata
import time # <-- 新增导入 time
import logging # <-- 新增导入 logging
import sys # <-- 新增导入 sys
import traceback # <-- 新增导入 traceback
from typing import List, Dict, Tuple, Callable, Optional, Any
import os

# 导入多进程包装器
try:
    from .multiprocess_wrapper import create_silent_executor
except ImportError:
    # 如果包装器不可用，使用标准的ProcessPoolExecutor
    def create_silent_executor(max_workers=None):
        return concurrent.futures.ProcessPoolExecutor(max_workers=max_workers)

# 假设 evaluate_dfm_params 函数可以从某个地方导入
# from .dfm_core import evaluate_dfm_params 
# 或者它被作为参数传递进来

logger = logging.getLogger(__name__) # <-- 获取 logger

# <<<-------------------- 新增全局后向筛选函数 -------------------->>>

def perform_global_backward_selection(
    initial_variables: List[str],
    initial_params: Dict, # 包含固定 k=N 的参数
    # initial_score_tuple: Tuple, # 不再需要初始分数，函数内部会计算
    target_variable: str, # 需要明确目标变量
    all_data: pd.DataFrame,
    var_type_map: Dict[str, str],
    validation_start: str,
    validation_end: str,
    target_freq: str,
    train_end_date: str,
    n_iter: int,
    target_mean_original: float,
    target_std_original: float,
    max_workers: int,
    evaluate_dfm_func: Callable,
    log_file: Optional[object] = None
) -> Tuple[List[str], Dict, Tuple, int, int]:
    """
    执行全局后向变量剔除。

    从所有预测变量开始，每次迭代评估移除每个变量后的性能，
    移除性能提升最大的那个变量（基于 HR -> -RMSE 优化）。
    当没有单个变量的移除能提升性能时停止。

    Args:
        initial_variables: 包含目标变量和所有初始预测变量的列表。
        initial_params: 固定的 DFM 参数 (包含 k=N)。
        target_variable: 目标变量名称。
        all_data: 包含所有变量的完整 DataFrame。
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
        log_file: 可选的日志文件句柄。

    Returns:
        Tuple 包含:
            - final_variables: 最终选择的变量列表 (包含目标变量)。
            - final_params: 最终选择的最佳参数 (与 initial_params 相同)。
            - final_score_tuple: 最终的最佳得分元组 (HR, -RMSE)。
            - total_evaluations: 此过程中执行的评估总次数。
            - svd_error_count: 此过程中遇到的 SVD 错误次数。
    """
    total_evaluations_global = 0
    svd_error_count_global = 0

    # 1. 初始化当前最优变量集 (仅预测变量)
    current_best_predictors = sorted([v for v in initial_variables if v != target_variable])
    if not current_best_predictors:
        logger.error("全局后向筛选：初始预测变量列表为空，无法进行筛选。")
        return initial_variables, initial_params, (-np.inf, np.inf), 0, 0

    # 2. 计算初始基准性能
    logger.info("全局后向筛选：计算初始基准性能...")
    initial_vars_for_eval = [target_variable] + current_best_predictors
    try:
        result_tuple_base = evaluate_dfm_func(
             variables=initial_vars_for_eval,
             full_data=all_data,
             target_variable=target_variable,
             params=initial_params,
             var_type_map=var_type_map,
             validation_start=validation_start,
             validation_end=validation_end,
             target_freq=target_freq,
             train_end_date=train_end_date,
             max_iter=n_iter,
             target_mean_original=target_mean_original,
             target_std_original=target_std_original,
        )
        total_evaluations_global += 1
        # --- 修正：确保接收正确数量的返回值 ---
        if len(result_tuple_base) != 9:
             logger.error(f"全局后向筛选：初始评估返回了 {len(result_tuple_base)} 个值 (预期 9)。无法计算基准分数。")
             return initial_variables, initial_params, (-np.inf, np.inf), total_evaluations_global, svd_error_count_global
        is_rmse_base, oos_rmse_base, _, _, is_hit_rate_base, oos_hit_rate_base, is_svd_error_base, _, _ = result_tuple_base
        # --- 结束修正 ---
        if is_svd_error_base: svd_error_count_global += 1

        combined_rmse_base = np.inf
        finite_rmses_base = [r for r in [is_rmse_base, oos_rmse_base] if r is not None and np.isfinite(r)]
        if finite_rmses_base: combined_rmse_base = np.mean(finite_rmses_base)

        combined_hit_rate_base = -np.inf
        finite_hit_rates_base = [hr for hr in [is_hit_rate_base, oos_hit_rate_base] if hr is not None and np.isfinite(hr)]
        if finite_hit_rates_base: combined_hit_rate_base = np.mean(finite_hit_rates_base)

        if not (np.isfinite(combined_rmse_base) and np.isfinite(combined_hit_rate_base)):
            logger.error("全局后向筛选：初始基准评估未能计算有效分数。无法继续。")
            return initial_variables, initial_params, (-np.inf, np.inf), total_evaluations_global, svd_error_count_global

        current_best_score_tuple = (combined_hit_rate_base, -combined_rmse_base)
        logger.info(f"初始基准得分 (HR={current_best_score_tuple[0]:.2f}%, RMSE={-current_best_score_tuple[1]:.6f})，变量数: {len(current_best_predictors)}")
        if log_file:
            log_file.write(f"\n--- 全局后向筛选开始 ---\n")
            log_file.write(f"初始预测变量数: {len(current_best_predictors)}\n")
            log_file.write(f"初始基准得分 (HR, -RMSE): {current_best_score_tuple}\n")

    except Exception as e_base:
        logger.error(f"全局后向筛选：计算初始基准性能时出错: {e_base}")
        traceback.print_exc(file=sys.stderr) # 打印 traceback 到 stderr
        return initial_variables, initial_params, (-np.inf, np.inf), total_evaluations_global, svd_error_count_global

    # 3. 初始化进度条
    # 总迭代次数最多是初始预测变量数 - 1 (至少保留一个预测变量)
    max_possible_removals = len(current_best_predictors) - 1 if len(current_best_predictors) > 1 else 0
    pbar = tqdm(total=max_possible_removals, desc="全局变量后向剔除", unit="var")

    # 4. 迭代移除变量
    iteration = 0
    while True:
        iteration += 1
        logger.info(f"\n--- 全局后向筛选：第 {iteration} 轮 (当前变量数: {len(current_best_predictors)}) ---")
        if len(current_best_predictors) <= 1:
            logger.info("剩余预测变量数已达下限 (1)，停止筛选。")
            break

        best_candidate_score_this_iter = (-np.inf, np.inf) # (HR, -RMSE) -> 越前面越大越好，越后面越小越好
        best_removal_candidate_var_this_iter = None
        results_this_iteration_map = {} # 存储移除变量 -> 结果

        futures_removal = {}
        # 使用静默的ProcessPoolExecutor
        with create_silent_executor(max_workers=max_workers) as executor:
            # logger.info(f"提交 {len(current_best_predictors)} 个移除评估任务...")
            for var_to_remove in current_best_predictors:
                temp_predictors = [v for v in current_best_predictors if v != var_to_remove]
                if not temp_predictors: # 如果移除后没有预测变量了，跳过
                    continue

                temp_variables_for_eval = [target_variable] + temp_predictors
                # 检查因子数是否仍然小于变量数 (N > K)
                if initial_params.get('k_factors', 1) >= len(temp_variables_for_eval):
                     # logger.debug(f"跳过移除 {var_to_remove}：移除后变量数 ({len(temp_variables_for_eval)}) <= 因子数 ({initial_params.get('k_factors')})")
                     continue

                future = executor.submit(
                    evaluate_dfm_func,
                    variables=temp_variables_for_eval,
                    full_data=all_data,
                    target_variable=target_variable,
                    params=initial_params, # 固定参数 k=N
                    var_type_map=var_type_map,
                    validation_start=validation_start,
                    validation_end=validation_end,
                    target_freq=target_freq,
                    train_end_date=train_end_date,
                    max_iter=n_iter,
                    target_mean_original=target_mean_original,
                    target_std_original=target_std_original,
                )
                futures_removal[future] = var_to_remove

        if not futures_removal:
             logger.info("本轮无可行的评估任务，筛选结束。")
             break

        # 使用 tqdm 处理 future 完成情况，提供评估进度
        eval_pbar_desc = f"迭代 {iteration} 评估"
        eval_pbar = tqdm(total=len(futures_removal), desc=eval_pbar_desc, unit="eval", leave=False)

        for future in concurrent.futures.as_completed(futures_removal):
            var_removed = futures_removal[future]
            total_evaluations_global += 1
            eval_pbar.update(1)
            try:
                result_tuple = future.result()
                # --- 修正：确保接收正确数量的返回值 ---
                if len(result_tuple) != 9:
                    logger.warning(f"评估函数返回了 {len(result_tuple)} 个值 (预期 9)，跳过移除 {var_removed} 的结果。")
                    continue
                is_rmse, oos_rmse, _, _, is_hit_rate, oos_hit_rate, is_svd_error, _, _ = result_tuple
                # --- 结束修正 ---
                if is_svd_error:
                    svd_error_count_global += 1

                combined_rmse_removal = np.inf
                finite_rmses = [r for r in [is_rmse, oos_rmse] if r is not None and np.isfinite(r)]
                if finite_rmses: combined_rmse_removal = np.mean(finite_rmses)

                combined_hit_rate_removal = -np.inf
                finite_hit_rates = [hr for hr in [is_hit_rate, oos_hit_rate] if hr is not None and np.isfinite(hr)]
                if finite_hit_rates: combined_hit_rate_removal = np.mean(finite_hit_rates)

                if np.isfinite(combined_rmse_removal) and np.isfinite(combined_hit_rate_removal):
                    current_score_tuple_eval = (combined_hit_rate_removal, -combined_rmse_removal) # 使用临时变量名
                    results_this_iteration_map[var_removed] = current_score_tuple_eval

                    # 实时比较，找到本轮最佳移除候选
                    if current_score_tuple_eval > best_candidate_score_this_iter:
                        best_candidate_score_this_iter = current_score_tuple_eval
                        best_removal_candidate_var_this_iter = var_removed
                else:
                    # logger.debug(f"移除 {var_removed} 的结果无效 (RMSE={combined_rmse_removal}, HR={combined_hit_rate_removal})")
                    pass # 不记录无效结果

            except Exception as exc:
                logger.error(f"处理移除 {var_removed} 的评估结果时出错: {exc}")
                traceback.print_exc(file=sys.stderr) # 添加 traceback 打印
        eval_pbar.close() # 关闭评估进度条

        # 5. 检查本轮结果，决定是否移除变量
        if best_removal_candidate_var_this_iter is not None:
             # 比较本轮找到的最佳分数与全局当前最佳分数
             # logger.debug(f"比较本轮最佳移除 ({best_removal_candidate_var_this_iter}) 得分 {best_candidate_score_this_iter} 与当前最佳得分 {current_best_score_tuple}")
             if best_candidate_score_this_iter > current_best_score_tuple:
                 # 找到了更好的解，执行移除
                 removed_var = best_removal_candidate_var_this_iter
                 old_score_str = f"(HR={current_best_score_tuple[0]:.2f}%, RMSE={-current_best_score_tuple[1]:.6f})"
                 new_score_str = f"(HR={best_candidate_score_this_iter[0]:.2f}%, RMSE={-best_candidate_score_this_iter[1]:.6f})"

                 logger.info(f"接受移除: '{removed_var}'，性能提升: {old_score_str} -> {new_score_str}")
                 if log_file:
                     log_file.write(f"Iter {iteration}: 移除 '{removed_var}'，得分 {old_score_str} -> {new_score_str}\n")

                 # 更新最优变量集和分数
                 current_best_predictors.remove(removed_var)
                 current_best_score_tuple = best_candidate_score_this_iter
                 pbar.update(1) # 更新总进度条

                 # 继续下一轮迭代
                 continue
             else:
                 logger.info("本轮最佳移除候选未优于当前最佳得分，筛选稳定。")
                 if log_file:
                     log_file.write(f"Iter {iteration}: 未找到更优移除，筛选结束。\n")
                 break # 跳出 while 循环
        else:
             logger.info("本轮未找到任何有效的移除候选，筛选结束。")
             if log_file:
                 log_file.write(f"Iter {iteration}: 未找到有效移除候选，筛选结束。\n")
             break # 跳出 while 循环

    pbar.close() # 关闭总进度条

    # 6. 返回最终结果
    final_variables = sorted([target_variable] + current_best_predictors)
    logger.info(f"\n全局后向筛选完成。最终选择 {len(current_best_predictors)} 个预测变量。")
    logger.info(f"最终得分 (HR, -RMSE): {current_best_score_tuple}")
    if log_file:
        log_file.write(f"\n--- 全局后向筛选结束 ---\n")
        log_file.write(f"最终预测变量数: {len(current_best_predictors)}\n")
        log_file.write(f"最终得分 (HR, -RMSE): {current_best_score_tuple}\n")
        # log_file.write(f"最终变量列表: {final_variables}\n") # 可能太长

    return final_variables, initial_params, current_best_score_tuple, total_evaluations_global, svd_error_count_global

# <<<-------------------- 结束新增全局后向筛选函数 -------------------->>>


# <<<-------------------- 保留原有的分块后向筛选函数 -------------------->>>
# （如果你确定不再需要它，可以将其注释掉或删除）

def perform_backward_selection(
    initial_variables: List[str],
    initial_params: Dict,
    initial_score_tuple: Tuple, # 根据 auto_select_factors 可能是 (HR, -K, -RMSE) 或 (HR, -RMSE)
    blocks: Dict[str, List[str]],
    all_data: pd.DataFrame,
    target_variable: str,
    hyperparams_to_tune: List[Dict], # 可能只包含一个元素
    var_type_map: Dict[str, str],
    validation_start: str,
    validation_end: str,
    target_freq: str,
    train_end_date: str,
    n_iter: int,
    target_mean_original: float,
    target_std_original: float,
    max_workers: int,
    evaluate_dfm_func: Callable,
    log_file: Optional[object] = None,
    auto_select_factors: bool = True # <-- 新增参数，默认 True
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
        log_file: 可选的日志文件句柄。
        auto_select_factors: 是否自动选择因子数。

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

    # --- 修改：根据模式调整初始得分打印 --- 
    score_len = len(current_best_score_tuple)
    if auto_select_factors and score_len == 3: # (HR, -RMSE, -K)
        logger.info(f"开始分块向后剔除 (自动选择因子)，初始得分: HR={current_best_score_tuple[0]:.2f}%, K={-current_best_score_tuple[2]}, RMSE={-current_best_score_tuple[1]:.6f}")
    elif not auto_select_factors and score_len == 2: # (HR, -RMSE)
        fixed_k_init = current_best_params.get('k_factors', '未知') # 获取初始固定 K
        logger.info(f"开始分块向后剔除 (固定因子 K={fixed_k_init})，初始得分: HR={current_best_score_tuple[0]:.2f}%, RMSE={-current_best_score_tuple[1]:.6f}")
    else:
        # 添加警告，因为得分元组长度与模式不匹配
        mode_str = "自动选择因子" if auto_select_factors else "固定因子"
        expected_len = 3 if auto_select_factors else 2
        logger.warning(f"分块向后剔除 ({mode_str}) 模式下，接收到的初始得分元组长度为 {score_len}，预期为 {expected_len}。")
        logger.info(f"开始分块向后剔除，初始得分: {current_best_score_tuple}")
    # --- 结束修改 ---

    for block_name, block_vars_list in tqdm(blocks.items(), desc="处理变量块", unit="block"):
        # 使用 list() 创建副本，以便安全地从中移除元素
        current_block_vars = list(block_vars_list)
        # logger.info(f"\n--- 处理块: \t'{block_name}'\t (初始 {len(current_block_vars)} 变量) ---")
        # 使用 logger 替代 print
        logger.info(f"\n--- 处理块: '{block_name}' (初始 {len(current_block_vars)} 变量) ---")


        # --- 移除块内最小变量保护 ---
        # if len(current_block_vars) <= 2: 
        #     print(f"块 \t'{block_name}'\t 变量数 ({len(current_block_vars)}) <= 2，跳过剔除。")
        #     continue
        # --- 结束移除 ---

        block_stable = False
        while not block_stable:
            # --- 修改：根据模式初始化本轮最佳得分元组 ---
            if auto_select_factors:
                 best_candidate_score_tuple_this_iter = (-np.inf, np.inf, np.inf) # (HR, -RMSE, -K), HR 最大化，K 和 RMSE 最小化
            else:
                 best_candidate_score_tuple_this_iter = (-np.inf, np.inf) # (HR, -RMSE), HR 最大化，RMSE 最小化
            # --- 结束修改 ---
            best_removal_candidate_this_iter = None

            # 查找当前块中仍然存在于全局最佳变量列表中的变量
            eligible_vars_in_block = [v for v in current_block_vars if v in current_best_variables and v != target_variable]

            if not eligible_vars_in_block:
                # logger.info(f"块 \t'{block_name}'\t 中无更多可剔除变量。")
                logger.info(f"块 '{block_name}' 中无更多可剔除变量。")
                block_stable = True
                break

            futures_removal = []
            with create_silent_executor(max_workers=max_workers) as executor:
                for candidate_var_to_remove in eligible_vars_in_block:
                    # 创建移除候选变量后的临时列表
                    temp_variables = [v for v in current_best_variables if v != candidate_var_to_remove]
                    if len(temp_variables) <= 1: # 至少需要一个预测变量+目标变量
                        continue

                    # 遍历需要测试的超参数（在固定模式下只有一个）
                    for params in hyperparams_to_tune:
                        # 确保因子数小于变量数 (N > K)
                        if params.get('k_factors', 1) >= len(temp_variables):
                            # logger.debug(f"Skipping K={params.get('k_factors')} >= N={len(temp_variables)} for removal of {candidate_var_to_remove}")
                            continue # 跳过 K >= N 的情况

                        future = executor.submit(
                            evaluate_dfm_func,
                            variables=temp_variables,
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
                            'removed_var': candidate_var_to_remove
                            # 'remaining_vars' 不再需要在此存储，因为 best_removal_candidate_this_iter 会保存被移除的变量
                        })

            results_this_iteration = []
            if not futures_removal:
                # logger.info(f"块 \t'{block_name}'\t 无有效评估任务可提交，此块完成。")
                logger.info(f"块 '{block_name}' 无有效评估任务可提交，此块完成。")
                block_stable = True
                continue
                
            for f_info in concurrent.futures.as_completed([f['future'] for f in futures_removal]):
                context = next(item for item in futures_removal if item['future'] == f_info)
                total_evaluations_selection += 1
                try:
                    result_tuple = f_info.result()
                    # --- 修正：确保接收正确数量的返回值 ---
                    if len(result_tuple) != 9:
                        logger.warning(f"评估函数返回了 {len(result_tuple)} 个值 (预期 9)，跳过移除 {context['removed_var']} 的结果。")
                        continue
                    is_rmse, oos_rmse, is_mae, oos_mae, is_hit_rate, oos_hit_rate, is_svd_error, _, _ = result_tuple
                    # --- 结束修正 ---
                    
                    if is_svd_error:
                        svd_error_count_selection += 1
                    
                    # 计算综合RMSE和胜率
                    combined_rmse_removal = np.inf
                    finite_rmses = [r for r in [is_rmse, oos_rmse] if r is not None and np.isfinite(r)]
                    if finite_rmses: combined_rmse_removal = np.mean(finite_rmses)

                    combined_hit_rate_removal = -np.inf
                    finite_hit_rates = [hr for hr in [is_hit_rate, oos_hit_rate] if hr is not None and np.isfinite(hr)]
                    if finite_hit_rates: combined_hit_rate_removal = np.mean(finite_hit_rates)

                    # 仅当 RMSE 和 Hit Rate 都有效时才添加到结果列表
                    if np.isfinite(combined_rmse_removal) and np.isfinite(combined_hit_rate_removal):
                        results_this_iteration.append({
                            'combined_rmse': combined_rmse_removal,
                            'combined_hit_rate': combined_hit_rate_removal,
                            'params': context['params'],
                            'removed_var': context['removed_var'],
                            'is_mae': is_mae, # 保留用于信息打印
                            'oos_mae': oos_mae # 保留用于信息打印
                        })
                    else:
                         # logger.debug(f"调试：跳过无效结果（RMSE={combined_rmse_removal}, HR={combined_hit_rate_removal}）对于移除 {context['removed_var']}，参数 {context['params']}")
                         pass

                except Exception as exc:
                    # 使用 logger 记录错误
                    logger.error(f"处理块 {block_name}, 尝试移除 {context['removed_var']} 时出错: {exc}")
                    # traceback.print_exc() # 可以考虑在 debug 模式下打印 traceback

            # 在本轮所有评估完成后，比较结果以找到最佳移除候选
            for result in results_this_iteration:
                # --- 修改：根据模式构建得分元组 ---
                if auto_select_factors:
                    current_k = result['params'].get('k_factors')
                    if current_k is not None and np.isfinite(current_k):
                         score_tuple_for_result = (result['combined_hit_rate'], -result['combined_rmse'], -int(current_k))
                    else:
                         # logger.debug(f"调试：跳过，因为在自动模式下无法获取有效的 k_factors: {result['params']}")
                         continue # 跳过无效结果
                else: # 固定模式
                    score_tuple_for_result = (result['combined_hit_rate'], -result['combined_rmse'])
                # --- 结束修改 ---

                # --- 比较得分 ---
                # 使用元组比较，它会按顺序比较元素
                # logger.debug(f"调试：比较 {score_tuple_for_result} vs {best_candidate_score_tuple_this_iter}")
                if score_tuple_for_result > best_candidate_score_tuple_this_iter:
                    best_candidate_score_tuple_this_iter = score_tuple_for_result
                    best_removal_candidate_this_iter = result # 存储整个结果字典

            # 检查是否找到了比当前全局最佳解更好的解
            accept_removal = False
            reason_for_update = ""
            if best_removal_candidate_this_iter:
                # --- 修改：比较本轮最佳和当前全局最佳 ---
                # 元组比较会自动处理不同长度的情况吗？否，如果长度不同会抛出 TypeError。
                # 我们需要确保比较发生在相同模式（即相同长度）的元组之间。
                # current_best_score_tuple 的长度应该已经由初始传入的值和 accept_removal 逻辑保证了与当前模式一致。
                # best_candidate_score_tuple_this_iter 的长度由本轮开始时的初始化保证了与当前模式一致。
                # 所以可以直接比较。
                if best_candidate_score_tuple_this_iter > current_best_score_tuple:
                    accept_removal = True
                    # --- 修改：构造原因描述 (根据模式) ---
                    removed_var_accepted = best_removal_candidate_this_iter['removed_var']
                    if auto_select_factors:
                        # 确保 score_tuple 长度正确
                        if len(current_best_score_tuple) == 3 and len(best_candidate_score_tuple_this_iter) == 3:
                            old_hr, old_neg_rmse, old_neg_k = current_best_score_tuple
                            new_hr, new_neg_rmse, new_neg_k = best_candidate_score_tuple_this_iter
                            new_params_accepted = best_removal_candidate_this_iter['params']
                            reason_for_update = f"找到更优解 (移除 {removed_var_accepted}, 参数 {new_params_accepted}): (HR={new_hr:.2f}%, RMSE={-new_neg_rmse:.6f}, K={-new_neg_k}) > (HR={old_hr:.2f}%, RMSE={-old_neg_rmse:.6f}, K={-old_neg_k})"
                        else:
                             reason_for_update = f"找到更优解（移除 {removed_var_accepted}，得分元组格式不匹配，使用新元组）" # 异常情况
                             logger.warning(f"自动模式下比较得分元组时长度不匹配: current={current_best_score_tuple}, new={best_candidate_score_tuple_this_iter}")
                    else: # 固定模式
                        if len(current_best_score_tuple) == 2 and len(best_candidate_score_tuple_this_iter) == 2:
                            old_hr, old_neg_rmse = current_best_score_tuple
                            new_hr, new_neg_rmse = best_candidate_score_tuple_this_iter
                            # 获取移除操作对应的固定 K (它应该总是与 current_best_params 中的 K 相同)
                            fixed_k_used = best_removal_candidate_this_iter['params'].get('k_factors', '未知')
                            reason_for_update = f"找到更优解 (移除 {removed_var_accepted}, 固定 K={fixed_k_used}): (HR={new_hr:.2f}%, RMSE={-new_neg_rmse:.6f}) > (HR={old_hr:.2f}%, RMSE={-old_neg_rmse:.6f})"
                        else:
                            reason_for_update = f"找到更优解（移除 {removed_var_accepted}，得分元组格式不匹配，使用新元组）" # 异常情况
                            logger.warning(f"固定模式下比较得分元组时长度不匹配: current={current_best_score_tuple}, new={best_candidate_score_tuple_this_iter}")
                    # --- 结束修改 ---

                    # logger.info(f"块 {block_name}: {reason_for_update}")
                    logger.info(f"{reason_for_update}") # 简化日志输出
                    if log_file:
                        log_file.write(f"块 {block_name}: {reason_for_update}\n")

                    # 更新全局最佳解
                    current_best_variables = [v for v in current_best_variables if v != removed_var_accepted]
                    current_best_score_tuple = best_candidate_score_tuple_this_iter
                    current_best_params = best_removal_candidate_this_iter['params'] # 更新参数
                
                    # 从当前块的列表中也移除，防止重复评估
                    if removed_var_accepted in current_block_vars:
                        current_block_vars.remove(removed_var_accepted)

                else: # 本轮最佳结果不优于当前全局最佳
                    # logger.info(f"块 {block_name}: 未找到比当前全局最佳 ({current_best_score_tuple}) 更好的移除候选 (本轮最佳: {best_candidate_score_tuple_this_iter})，此块稳定。")
                    logger.info(f"块 '{block_name}': 未找到更优移除候选，此块稳定。")
                    if log_file:
                        log_file.write(f"块 {block_name}: 未找到更优移除，此块稳定。\n")
                    block_stable = True # 结束当前块的循环
            else: # 本轮未找到任何有效的移除候选
                # logger.info(f"块 {block_name}: 本轮迭代未找到任何有效的移除候选，此块稳定。")
                logger.info(f"块 '{block_name}': 本轮迭代未找到任何有效的移除候选，此块稳定。")
                if log_file:
                    log_file.write(f"块 {block_name}: 本轮迭代未找到有效的移除候选，此块稳定。\n")
                block_stable = True # 结束当前块的循环

    logger.info(f"分块向后剔除完成。最终变量数: {len(current_best_variables)}")
    return current_best_variables, current_best_params, current_best_score_tuple, total_evaluations_selection, svd_error_count_selection
