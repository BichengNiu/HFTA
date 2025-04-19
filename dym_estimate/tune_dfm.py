# -*- coding: utf-8 -*-

# print("DEBUG: Script execution started.") # REMOVE DEBUG
"""
超参数和变量逐步前向选择脚本。
目标：最小化 OOS RMSE。
"""
import pandas as pd
import numpy as np
import sys
import os
import time
import warnings
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns # <-- MOVED BACK TO TOP
import concurrent.futures
from tqdm import tqdm
import traceback
from typing import Tuple, List, Dict, Union # 添加 Tuple
import unicodedata # <-- 新增导入
from sklearn.decomposition import PCA # <-- 新增：导入 PCA
from sklearn.impute import SimpleImputer # <-- 新增：导入 SimpleImputer
import multiprocessing
from collections import defaultdict
import logging
import joblib # 用于保存和加载模型/结果
from datetime import datetime

# --- 内部模块导入 ---
from .data_utils import apply_stationarity_transforms # <--- 新增导入
from .dfm_core import evaluate_dfm_params # <-- 新增导入
from .results_analysis import analyze_and_save_final_results, plot_final_nowcast # <-- 新增导入
# --- 新增: 直接在 tune_dfm.py 中定义常量 (使用 config.py 中的值) ---
TARGET_VARIABLE = '规模以上工业增加值:当月同比'
TARGET_FREQ = 'W-FRI'
TARGET_SHEET_NAME = '工业增加值同比增速-月度'
EXCEL_DATA_FILE = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', '经济数据库.xlsx') # 相对路径
TYPE_MAPPING_SHEET = '指标体系'
REMOVE_VARS_WITH_CONSECUTIVE_NANS = True
CONSECUTIVE_NAN_THRESHOLD = 10
OPTIMIZATION_PRIORITY = 'hit_rate_first' # 从旧代码中恢复，config.py 中未包含此项
ROLLING_VALIDATION_WINDOW_SIZE = 52 # 从旧代码中恢复
ROLLING_VALIDATION_STEP = 4 # 从旧代码中恢复
N_ITER_FIXED = 30 # 使用 config.py 中的值
N_ITER_TEST = 2   # 使用 config.py 中的值
TEST_MODE = True # <-- 修改为 True
DEBUG_VARIABLE_SELECTION_BLOCK = '其他' if TEST_MODE else None
HEATMAP_TOP_N_VARS = 5 # 使用 config.py 中的值 (注意 config.py 末尾定义为 30，此处用了 5)
VALIDATION_START_DATE = '2024-07-05'
VALIDATION_END_DATE = '2024-12-27'
TRAIN_END_DATE = '2024-06-28'
ROLLING_VALIDATION_ENABLED = True
INITIAL_TRAIN_WEEKS = 208 # 必须定义，即使 ROLLING_VALIDATION_ENABLED 为 False
VALIDATION_WEEKS = 4      # 必须定义
ROLLING_STEP_WEEKS = 4    # 必须定义
# --- 结束新增常量定义 ---

# --- 新增: 从 variable_selection 导入 --- 
from .variable_selection import perform_backward_selection
# --- 新增: 从 analysis_utils 导入 --- 
from .analysis_utils import calculate_pca_variance, calculate_factor_contributions
# --- 结束新增 ---

# --- 尝试导入自定义模块 --- 
try:
    # --- 修改: 从 data_preparation 导入 load_mappings --- 
    from .data_preparation import prepare_data, load_mappings 
    from .DynamicFactorModel import DFM_EMalgo # Correct relative import
except ImportError as e:
    print(f"错误：导入自定义模块失败: {e}")
    sys.exit(1)

# --- 配置 ---
warnings.filterwarnings("ignore")
matplotlib.use("Agg")
plt.rcParams["font.sans-serif"] = ["SimHei"]
plt.rcParams["axes.unicode_minus"] = False

# --- 全局执行标志 --- 
# _run_tuning_executed = False # <-- REMOVED

# --- 配置日志记录 ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- 全局配置和常量 (保留输出路径定义) ---
# BASE_OUTPUT_DIR = 'output' # 旧的基础输出目录
BASE_OUTPUT_DIR = os.path.join('dym_estimate', 'dfm_result') # <<< 新的基础目录
EXCEL_INPUT_PATH = 'data/经济数据库.xlsx' # 这个看起来是重复的，但保留以防万一

# 时间戳用于文件名和目录名
timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")

# --- 修改: 直接使用 BASE_OUTPUT_DIR 作为运行输出目录 ---
# run_output_dir = os.path.join(BASE_OUTPUT_DIR, f'run_{timestamp_str}') # 旧的带时间戳子目录
run_output_dir = BASE_OUTPUT_DIR # <<< 直接使用基础目录
os.makedirs(run_output_dir, exist_ok=True) # 确保目录存在
# print(f"本次运行结果将保存在: {os.path.abspath(run_output_dir)}") # <-- 注释掉这一行

# 日志文件路径
tuning_log_file = os.path.join(run_output_dir, f"调优日志_{timestamp_str}.txt")
# ... (设置日志记录器，应在 logging.basicConfig 之前或作为其一部分)

# Excel 输出文件路径 (文件名保持时间戳，路径使用新的 run_output_dir)
excel_output_file = os.path.join(run_output_dir, f"final_results_{timestamp_str}.xlsx")

# --- (重新) 配置日志记录器以使用文件 --- 
logger = logging.getLogger(__name__)
# 清除可能存在的旧处理器
if logger.hasHandlers():
    logger.handlers.clear()
# 添加文件处理器和流处理器
logger.addHandler(logging.FileHandler(tuning_log_file, encoding='utf-8'))
logger.addHandler(logging.StreamHandler(sys.stdout))
logger.setLevel(logging.INFO)

# --- 函数定义 --- 
# (analyze_and_save_final_results 和 plot_final_nowcast 定义需要更新以接收日期常量)
def plot_final_nowcast(
    run_output_dir: str, # 新增
    timestamp_str: str,  # 新增
    all_data_full: pd.DataFrame, # 新增: 用于获取原始目标
    data_for_analysis: dict, # 新增: 用于获取反标准化参数
    target_variable: str, # 新增
    final_dfm_results: object, # 新增: 用于计算 nowcast
    best_k: int, # 新增
    best_variables: List[str], # 新增 (可选，如果需要重新计算)
    target_mean_original: float, # 新增
    target_std_original: float,  # 新增
    validation_start: str,
    validation_end: str
    # title 和 filename 将在内部生成
):
    """绘制最终 Nowcast 对比图。"""
    # ... (函数体保持不变，但内部使用传入的 validation_start/end)
    # title 和 filename 在内部基于 run_output_dir 和 timestamp_str 生成
    pass # Placeholder for brevity

# --- 主逻辑 --- 
def run_tuning():
    try: # 添加 try 块
        # import seaborn as sns # <-- REMOVED from here
        # global _run_tuning_executed # <-- REMOVED
        # # Force reset the flag at the beginning of the function
        # _run_tuning_executed = False # <-- REMOVED
        # if _run_tuning_executed: # This check will now always be false initially # <-- REMOVED
        #     print("警告: run_tuning() 已被调用，跳过重复执行。") # <-- REMOVED
        #     return # <-- REMOVED
        # _run_tuning_executed = True # Set the flag after ensuring it runs # <-- REMOVED

        n_iter_to_use = N_ITER_TEST if TEST_MODE else N_ITER_FIXED
        MAX_WORKERS = os.cpu_count() - 1 if os.cpu_count() and os.cpu_count() > 1 else 1

        script_start_time = time.time()
        total_evaluations = 0
        svd_error_count = 0
        log_file = None # 初始化 log_file
        best_k_rolling_loadings = None # 初始化滚动载荷列表
        final_transform_details = None # 初始化转换日志
        pca_results_df = None # 初始化 PCA 结果
        contribution_results_df = None # 初始化贡献度结果
        factor_contributions = None # 初始化贡献度字典
        final_dfm_results_obj = None # 初始化 DFM 结果对象
        data_for_analysis = None # 初始化分析数据
        
        print(f"--- 开始变量后向剔除与超参数调优 (优化目标: 平均胜率优先, k_factors其次; 因子数范围动态确定) ---")
        # print(f"本次运行结果将保存在: {run_output_dir}") # <-- 注释掉这一行
        
        try:
            os.makedirs(run_output_dir, exist_ok=True)
            log_file = open(tuning_log_file, 'w', encoding='utf-8')
            log_file.write(f"--- 开始详细调优日志 (Run: {timestamp_str}) ---\
")
            log_file.write(f"输出目录: {run_output_dir}\
")
            log_file.write(f"配置: 优化目标=平均胜率优先, k_factors其次\
")
            print(f"详细调优日志将写入: {tuning_log_file}")
        except IOError as e:
            print(f"错误: 无法打开日志文件 {tuning_log_file} 进行写入: {e}")
            # 不退出，但后续日志写入会失败

        all_evaluation_results = []

        print("\n--- 调用数据准备模块 (自动发现 Sheets) --- ")
        # 假设 prepare_data 来自导入的模块
        all_data_aligned_weekly, var_industry_map_inferred = prepare_data(
             excel_path=EXCEL_DATA_FILE,
             target_freq=TARGET_FREQ,
             target_sheet_name=TARGET_SHEET_NAME,
             target_variable_name=TARGET_VARIABLE,
             consecutive_nan_threshold=CONSECUTIVE_NAN_THRESHOLD if REMOVE_VARS_WITH_CONSECUTIVE_NANS else None
             # Ensure all required args for prepare_data are passed if its signature changed
        )
        if all_data_aligned_weekly is None:
            print("错误: 数据准备失败，无法继续调优。")
            if log_file and not log_file.closed: log_file.close()
            sys.exit(1)
        
        print(f"\n[EARLY CHECK 1] prepare_data completed. Output shape: {all_data_aligned_weekly.shape}")
        print(f"  Industry map inferred from sheet names: {len(var_industry_map_inferred)} entries.") # Log inferred map size
        print("-"*30)

        print(f"数据准备模块成功返回处理后的数据. Shape: {all_data_aligned_weekly.shape}")
        
        all_variable_names = all_data_aligned_weekly.columns.tolist()
        if TARGET_VARIABLE not in all_variable_names:
            print(f"错误: 目标变量 {TARGET_VARIABLE} 不在合并后的数据中。")
            if log_file and not log_file.closed: log_file.close()
            sys.exit(1)
        
        initial_variables = sorted(all_variable_names)
        print(f"\n初始变量组 ({len(initial_variables)}): {initial_variables}")
        print("-"*30)

        # --- NEW: 调用 load_mappings 加载类型和行业映射 --- 
        print(f"调用 load_mappings 从 Sheet '{TYPE_MAPPING_SHEET}' 加载映射...")
        var_type_map, var_industry_map = load_mappings(
            excel_path=EXCEL_DATA_FILE, 
            sheet_name=TYPE_MAPPING_SHEET
            # Assuming default indicator_col, type_col, industry_col names are correct
        )
        if not var_type_map: # Exit if type map loading fails critically
            print(f"错误: 无法从 Excel 	'{EXCEL_DATA_FILE}	' (Sheet: 	'{TYPE_MAPPING_SHEET}	') 加载必要的类型映射。")
            if log_file and not log_file.closed: log_file.close()
            sys.exit(1)
        print(f"\n[EARLY CHECK 2] Mappings loaded. Type map size: {len(var_type_map)}, Industry map size: {len(var_industry_map)}")
        print("-"*30)
        # --- 结束新增调用 --- 

        print("计算原始目标变量的稳定统计量...")
        target_mean_original = np.nan
        target_std_original = np.nan
        try:
            original_target_series_for_stats = all_data_aligned_weekly[TARGET_VARIABLE].copy().dropna()
            if original_target_series_for_stats.empty:
                raise ValueError(f"原始目标变量 '{TARGET_VARIABLE}' 移除 NaN 后为空")
            target_mean_original = original_target_series_for_stats.mean()
            target_std_original = original_target_series_for_stats.std()
            if pd.isna(target_mean_original) or pd.isna(target_std_original) or target_std_original == 0:
                raise ValueError(f"计算得到的原始目标变量统计量无效 (Mean: {target_mean_original}, Std: {target_std_original})。")
            print(f"已计算原始目标变量的稳定统计量 (用于反标准化): Mean={target_mean_original:.4f}, Std={target_std_original:.4f}")
        except Exception as e:
            print(f"错误: 计算原始目标变量统计量失败: {e}")
            if log_file and not log_file.closed: log_file.close()
            sys.exit(1)
        print("-"*30)

        # --- 初始变量连续缺失检查 --- 
        initial_predictors = [v for v in initial_variables if v != TARGET_VARIABLE]
        if REMOVE_VARS_WITH_CONSECUTIVE_NANS:
            print(f"\n--- (启用) 检查初始预测变量 ({len(initial_variables)}) 的连续缺失值 (阈值 >= {CONSECUTIVE_NAN_THRESHOLD})... ---")
            # ... (保留检查逻辑) ...
            pass # Placeholder for brevity
        else:
            print(f"\n--- (禁用) 跳过基于连续缺失值 (阈值 >= {CONSECUTIVE_NAN_THRESHOLD}) 的初始变量移除步骤。---")
            if log_file:
                 try: log_file.write("\n--- (禁用) 跳过变量筛选前连续缺失值检查 ---\
")
                 except Exception: pass
        print("-"*30)

        # --- 确定 K_FACTORS_RANGE 和 HYPERPARAMS_TO_TUNE --- 
        print("\n--- 确定初始变量块和动态因子数范围 (基于筛选后的变量) ---")
        K_FACTORS_RANGE = [] # Placeholder
        initial_blocks = {} # Initialize initial_blocks
        try:
            # --- FIX: Add logic to create initial_blocks --- 
            if var_type_map:
                temp_blocks_init = defaultdict(list)
                current_predictors = [v for v in initial_variables if v != TARGET_VARIABLE] # Use current initial_variables
                for var in current_predictors:
                    lookup_key = unicodedata.normalize('NFKC', str(var)).strip().lower()
                    var_type = var_type_map.get(lookup_key, "_未知类型_")
                    temp_blocks_init[var_type].append(var)
                
                merged_small_block_vars_init = []
                small_block_names_merged_init = []
                initial_blocks = {} # Reset here before populating
                for block_name, block_vars in temp_blocks_init.items():
                    if len(block_vars) < 3:
                        merged_small_block_vars_init.extend(block_vars)
                        small_block_names_merged_init.append(block_name)
                    else:
                        initial_blocks[block_name] = block_vars
                if merged_small_block_vars_init:
                    initial_blocks["其他"] = merged_small_block_vars_init
                    print(f"  根据初始变量，已将 {len(small_block_names_merged_init)} 个小块 ({small_block_names_merged_init}) 合并到 '其他' (共 {len(merged_small_block_vars_init)} 变量)。")
                print(f"  基于类型映射创建了 {len(initial_blocks)} 个初始变量块。")
            else:
                 print("  警告: 类型映射不可用，无法按类型创建初始块。将无法动态确定 K 范围。")
            # --- END FIX ---

            max_k_factors = len(initial_blocks) # 假设 initial_blocks 已计算
            if max_k_factors == 0: max_k_factors = 1
            K_FACTORS_RANGE = list(range(1, max_k_factors + 1))
            print(f"动态确定的因子数范围 (基于初始块数 {max_k_factors}): {K_FACTORS_RANGE}")
        except Exception as e: # FIX: Add except block to handle errors during block/k_range determination
            print(f"确定 K_FACTORS_RANGE 时出错: {e}")
            K_FACTORS_RANGE = [1, 2, 3] # 提供一个默认值以继续
            print(f"警告: 使用默认 K_FACTORS_RANGE: {K_FACTORS_RANGE}")
      
        if TEST_MODE:
            max_test_factors = 5     
            K_FACTORS_RANGE_TEST = K_FACTORS_RANGE[:max_test_factors] 
            print(f"*** 测试模式：限制因子数范围为: {K_FACTORS_RANGE_TEST} ***")
            k_factors_for_tuning = K_FACTORS_RANGE_TEST
        else:
            k_factors_for_tuning = K_FACTORS_RANGE

        HYPERPARAMS_TO_TUNE = []
        for k in k_factors_for_tuning:
            HYPERPARAMS_TO_TUNE.append({'k_factors': k})
        print(f"构建了 {len(HYPERPARAMS_TO_TUNE)} 个超参数组合进行测试 (仅调整 k_factors)。")

        # ... (保留打印配置信息) ...
        print("-" * 30)

        # --- 初始化评估 --- 
        print("\n--- 初始化: 评估所有预测变量 ---")
        initial_best_found = False # Initialize flag before loop
        best_overall_score_tuple = (-np.inf, np.inf, np.inf) # Initialize best score
        best_overall_params = None
        best_overall_variables = initial_variables.copy() # Initialize with all vars
        
        # --- FIX: Add the actual loop for processing initial evaluation results --- 
        futures_initial_map = {}
        with concurrent.futures.ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
            print(f"提交 {len(HYPERPARAMS_TO_TUNE)} 个初始评估任务 (使用最多 {MAX_WORKERS} 进程)...")
            for params_init in HYPERPARAMS_TO_TUNE:
                future = executor.submit(evaluate_dfm_params,
                                         variables=initial_variables, # Evaluate with all initial variables
                                         full_data=all_data_aligned_weekly, # <-- FIX: Changed all_data to full_data
                                         target_variable=TARGET_VARIABLE,
                                         params=params_init, # Pass current hyperparameter set
                                         var_type_map=var_type_map,
                                         validation_start=VALIDATION_START_DATE,
                                         validation_end=VALIDATION_END_DATE,
                                         target_freq=TARGET_FREQ,
                                         train_end_date=TRAIN_END_DATE,
                                         target_mean_original=target_mean_original,
                                         target_std_original=target_std_original,
                                         max_iter=n_iter_to_use
                                     )
                futures_initial_map[future] = params_init

        print("处理初始评估结果...")
        for future in concurrent.futures.as_completed(futures_initial_map):
            params = futures_initial_map[future]
            total_evaluations += 1 # Increment evaluation count
            try:
                # THIS IS THE LINE TO MODIFY: Expect 8 values now
                # FIX: Correct unpacking order
                is_rmse, oos_rmse, is_mae, oos_mae, is_hit_rate, oos_hit_rate, is_svd_error, lambda_df_result = future.result()
                if is_svd_error:
                    svd_error_count += 1 

                # Calculate combined metrics
                combined_rmse = np.inf
                if np.isfinite(is_rmse) and np.isfinite(oos_rmse): combined_rmse = 0.5 * is_rmse + 0.5 * oos_rmse
                elif np.isfinite(is_rmse): combined_rmse = is_rmse
                elif np.isfinite(oos_rmse): combined_rmse = oos_rmse
                
                combined_hit_rate = -np.inf
                valid_hit_rates = []
                if np.isfinite(is_hit_rate): valid_hit_rates.append(is_hit_rate)
                if np.isfinite(oos_hit_rate): valid_hit_rates.append(oos_hit_rate)
                if valid_hit_rates: combined_hit_rate = np.mean(valid_hit_rates)
                    
                # Store results (optional, can be removed if not needed later)
                all_evaluation_results.append({
                    'variables': initial_variables, 'params': params, 
                    'combined_rmse': combined_rmse, 'combined_hit_rate': combined_hit_rate,
                    'is_rmse': is_rmse, 'oos_rmse': oos_rmse, 'is_mae': is_mae, 'oos_mae': oos_mae,
                    'is_hit_rate': is_hit_rate, 'oos_hit_rate': oos_hit_rate
                })

                # Compare with current best
                if np.isfinite(combined_hit_rate) and np.isfinite(combined_rmse) and 'k_factors' in params:
                    current_k_factors = params['k_factors']
                    if np.isfinite(current_k_factors):
                         current_score_tuple = (combined_hit_rate, -current_k_factors, -combined_rmse)
                         if current_score_tuple > best_overall_score_tuple:
                             best_overall_score_tuple = current_score_tuple
                             best_overall_params = params
                             # best_overall_variables remains initial_variables during this phase
                             initial_best_found = True # FIX: Set the flag here
                             
            except Exception as e_init_eval:
                print(f"处理初始评估参数 {params} 时出错: {type(e_init_eval).__name__}: {e_init_eval}")
                # --- REMOVE DETAILED TRACEBACK PRINTING --- 
                # print(\"--- Error Traceback START ---\")
                # traceback.print_exc() # Print traceback from the caught exception
                # print(\"--- Error Traceback END ---\")
                # --- END REMOVE --- 
                if log_file and not log_file.closed: 
                    # Keep logging the basic error message, but no need for full traceback now
                    log_file.write(f"\n!!! 初始评估错误 (参数 {params}): {e_init_eval}\n")
        # --- END FIX ---

        # 确保 best_overall_variables, best_overall_params, best_overall_score_tuple 被正确赋值
        if not initial_best_found: # 现在这个检查应该能正确工作了
            print("错误: 初始化评估未能成功运行任何超参数组合。无法继续。")
            if log_file and not log_file.closed: log_file.close()
            sys.exit(1)
        
        best_init_hr, best_init_neg_k, best_init_neg_rmse = best_overall_score_tuple
        print(f"初始评估完成。最佳组合得分: 平均胜率={best_init_hr:.2f}%, 因子数={-best_init_neg_k}, 平均 RMSE={-best_init_neg_rmse:.6f} (参数: {best_overall_params})")
        # ... (保留日志写入) ...
        print("-" * 30)

        # --- 变量后向剔除 --- 
        print("\n--- 开始分块向后变量剔除 (调用 variable_selection 模块) ---")
        final_variables = best_overall_variables.copy() # 默认使用初始最佳
        final_params = best_overall_params
        final_score_tuple = best_overall_score_tuple
        try:
            # --- 调用新的函数 --- 
            # --- 修改：根据 TEST_MODE 决定要选择的块 ---
            if TEST_MODE:
                blocks_for_selection = {k: v for k, v in initial_blocks.items() if k == '其他'} 
                if not blocks_for_selection:
                    print("警告：(测试模式) 未找到 '其他' 块，跳过变量后向剔除。")
                    # 初始化 sel_* 变量以避免 UnboundLocalError
                    sel_variables, sel_params, sel_score_tuple, sel_eval_count, sel_svd_err_count = (final_variables, final_params, final_score_tuple, 0, 0)
                else:
                    print(f"--- (测试模式) 仅对块 {list(blocks_for_selection.keys())} 进行变量剔除 ---") 
            else:
                blocks_for_selection = initial_blocks # 使用所有块
                print(f"--- (完整模式) 将对所有 {len(blocks_for_selection)} 个块进行变量剔除: {list(blocks_for_selection.keys())} ---") 

            # 只有在确定了要选择的块时才运行选择过程
            if blocks_for_selection:
                 sel_variables, sel_params, sel_score_tuple, sel_eval_count, sel_svd_err_count = perform_backward_selection(
                     initial_variables=best_overall_variables, 
                     initial_params=best_overall_params,
                     initial_score_tuple=best_overall_score_tuple,
                     blocks=blocks_for_selection, # <-- 使用过滤后的块
                     all_data=all_data_aligned_weekly, 
                     target_variable=TARGET_VARIABLE, 
                     hyperparams_to_tune=HYPERPARAMS_TO_TUNE, 
                     var_type_map=var_type_map, 
                     validation_start=VALIDATION_START_DATE, 
                     validation_end=VALIDATION_END_DATE, 
                     target_freq=TARGET_FREQ, 
                     train_end_date=TRAIN_END_DATE, 
                     n_iter=n_iter_to_use, 
                     target_mean_original=target_mean_original, 
                     target_std_original=target_std_original,
                     max_workers=MAX_WORKERS,
                     evaluate_dfm_func=evaluate_dfm_params, # 传递核心评估函数
                     log_file=log_file
                 )
            # else 分支已经在 if TEST_MODE 块中处理了，如果测试模式下找不到'其他'块，sel_* 会被初始化

            # --- 结束修改 --- 
            # --- 更新结果 --- 
            final_variables = sel_variables
            final_params = sel_params
            final_score_tuple = sel_score_tuple
            total_evaluations += sel_eval_count # 累加评估次数
            svd_error_count += sel_svd_err_count # 累加 SVD 错误次数
            print(f"变量剔除过程完成。") # 函数内部已有详细打印
            # --- 结束调用和更新 --- 
        except Exception as e_select:
            print(f"变量后向剔除过程中发生严重错误: {e_select}")
            traceback.print_exc()
            print("警告: 变量剔除失败，将使用初始评估的最佳结果继续。")
            # 保持 final_variables, final_params, final_score_tuple 为初始值
        print("-" * 30)

        # --- 使用变量选择后的最佳结果 ---
        print("--- 使用变量选择阶段的最佳结果进行最终模型运行 ---")
        if final_params is None or 'k_factors' not in final_params:
             print("警告: 变量选择后未获得有效参数，将尝试使用初始评估的最佳参数。")
             final_params = best_overall_params # 回退到初始最佳参数
             final_variables = best_overall_variables # 确保变量也回退
             final_score_tuple = best_overall_score_tuple # 确保得分也回退
             if final_params is None or 'k_factors' not in final_params:
                  print("错误: 变量选择和初始评估均未提供有效 k_factors，无法继续。")
                  if log_file and not log_file.closed: log_file.close()
                  sys.exit(1)
             else:
                 print(f"  已回退到初始评估结果: k={final_params.get('k_factors')}, HR={final_score_tuple[0]:.2f}, RMSE={-final_score_tuple[2]:.6f}")
        else:
             print(f"  将使用变量选择的最佳结果: k={final_params.get('k_factors')}, HR={final_score_tuple[0]:.2f}, RMSE={-final_score_tuple[2]:.6f}")

        # --- 最终模型运行 --- 
        print("\n--- 使用最终选择的变量和参数重新运行 DFM 模型 --- ")
        final_dfm_results_obj = None
        data_for_analysis = {}
        final_data_std_for_dfm = None # Initialize for PCA later
        final_data_std_masked_for_fit = None # Initialize for PCA fallback

        if final_variables and final_params and 'k_factors' in final_params:
            try:
                final_k_factors = final_params['k_factors']
                print(f"  最终变量数: {len(final_variables)}, 最终因子数: {final_k_factors}")

                # 1. 准备最终数据 (类似 evaluate_dfm_params 但使用 final_variables)
                # 使用包含完整时间范围的数据进行最终拟合和分析
                final_data = all_data_aligned_weekly[final_variables].copy()
                final_data = final_data.apply(pd.to_numeric, errors='coerce')
                print(f"  最终数据初始 Shape: {final_data.shape}")

                # 应用平稳性转换
                final_data_transformed, final_transform_details = apply_stationarity_transforms(final_data, TARGET_VARIABLE)
                current_final_vars = final_data_transformed.columns.tolist()
                print(f"  转换后 Shape: {final_data_transformed.shape}")

                # 移除全 NaN 列 (以防万一)
                nan_cols_final = final_data_transformed.columns[final_data_transformed.isna().all()].tolist()
                if nan_cols_final:
                    print(f"    警告 (最终模型): 移除全 NaN 列: {nan_cols_final}")
                    final_data_transformed = final_data_transformed.drop(columns=nan_cols_final)
                    current_final_vars = [v for v in current_final_vars if v not in nan_cols_final]
                    if TARGET_VARIABLE not in current_final_vars:
                         raise ValueError("目标变量在最终模型准备中因全 NaN 被移除")
                print(f"  移除 NaN 列后 Shape: {final_data_transformed.shape}")

                if final_data_transformed.empty or final_data_transformed.shape[1] <= final_k_factors:
                    raise ValueError("最终模型数据变量不足或为空")

                # 标准化 (使用整个序列的 mean/std)
                final_mean = final_data_transformed.mean(skipna=True)
                final_std = final_data_transformed.std(skipna=True)
                final_std[final_std == 0] = 1.0
                final_data_std = (final_data_transformed - final_mean) / final_std
                final_data_std_for_dfm = final_data_std.copy() # Save for PCA
                print(f"  标准化后 Shape: {final_data_std.shape}, NaN 总数: {final_data_std.isna().sum().sum()}")

                if final_data_std.isna().all().all():
                     raise ValueError("最终标准化数据全为 NaN")

                # 掩码目标变量 (用于拟合)
                final_data_std_masked_for_fit = final_data_std.copy()
                month_indices_final = final_data_std_masked_for_fit.index.month
                mask_jan_feb_final = (month_indices_final == 1) | (month_indices_final == 2)
                if TARGET_VARIABLE in final_data_std_masked_for_fit.columns:
                    final_data_std_masked_for_fit.loc[mask_jan_feb_final, TARGET_VARIABLE] = np.nan
                else:
                     raise ValueError("目标变量在最终掩码前丢失")
                print(f"  掩码后 Shape: {final_data_std_masked_for_fit.shape}, NaN 总数: {final_data_std_masked_for_fit.isna().sum().sum()}")

                # 2. 运行最终 DFM
                print(f"  开始运行最终 DFM (n_iter={n_iter_to_use})...")
                final_dfm_results_obj = DFM_EMalgo(
                    observation=final_data_std_masked_for_fit, 
                    n_factors=final_k_factors,
                    n_shocks=final_k_factors, # Assuming n_shocks = n_factors
                    n_iter=n_iter_to_use
                )
                print("  最终 DFM 模型运行完成。")

                # 3. 准备 data_for_analysis
                data_for_analysis = {
                    "final_data_processed": final_data_transformed, # 转换后，标准化前的数据
                    "final_data_standardized": final_data_std,       # 标准化后的数据 (未掩码)
                    "final_data_std_masked_for_fit": final_data_std_masked_for_fit, # 标准化+掩码 (用于拟合)
                    "final_mean": final_mean,                      # 标准化参数
                    "final_std": final_std,                        # 标准化参数
                    "final_transform_details": final_transform_details, # 转换日志
                    # 新增: 反标准化所需的目标变量统计量
                    "final_target_mean_rescale": target_mean_original, 
                    "final_target_std_rescale": target_std_original
                    # 可以根据需要添加更多信息
                }
                print("  data_for_analysis 字典已填充。")

            except Exception as e_final_run:
                print(f"!!! 运行最终 DFM 模型时发生错误: {e_final_run}")
                traceback.print_exc()
                final_dfm_results_obj = None # Ensure it's None on failure
                data_for_analysis = None     # Ensure it's None on failure
        else:
            print("警告: 缺少最终变量或参数，无法运行最终 DFM 模型。")

        print("-" * 30)

        # --- 计算 PCA 和因子贡献度 --- 
        print("\n--- 计算最终分析指标 (调用 analysis_utils 模块) ---")
        try:
            # --- 调用新的函数 --- 
            if data_for_analysis and "final_data_processed" in data_for_analysis and final_params and "k_factors" in final_params:
                 # 使用 DFM 拟合前的标准化数据进行 PCA (可能含 NaN，函数内部处理)
                 # 需要确保 final_data_std_for_dfm 在最终模型运行部分被正确计算和保留
                if "final_data_std_for_dfm" in locals() and final_data_std_for_dfm is not None:
                    pca_results_df = calculate_pca_variance(
                        data_standardized=final_data_std_for_dfm, # 使用这个
                        n_components=final_params["k_factors"]
                    )
                elif "final_data_std_masked_for_fit" in locals() and final_data_std_masked_for_fit is not None:
                     print("警告：final_data_std_for_dfm 未找到，回退使用 final_data_std_masked_for_fit 进行 PCA 计算。")
                     pca_results_df = calculate_pca_variance(
                        data_standardized=final_data_std_masked_for_fit, # 回退选项
                        n_components=final_params["k_factors"]
                     )
                else:
                     print("警告：无法找到合适的标准化数据进行 PCA 计算。")
            else:
                print("警告：缺少必要数据 (data_for_analysis, final_params) 无法计算 PCA。")
        except Exception as e_pca_call:
            print(f"调用 calculate_pca_variance 时出错: {e_pca_call}")

        try:
            # --- 调用新的函数 --- 
            if final_dfm_results_obj and data_for_analysis and "final_data_processed" in data_for_analysis and final_params and "k_factors" in final_params:
                contribution_results_df, factor_contributions = calculate_factor_contributions(
                    dfm_results=final_dfm_results_obj,
                    data_processed=data_for_analysis["final_data_processed"],
                    target_variable=TARGET_VARIABLE,
                    n_factors=final_params["k_factors"]
                )
            else:
                print("警告：缺少必要数据 (final_dfm_results_obj, data_for_analysis, final_params) 无法计算因子贡献度。")
        except Exception as e_contrib_call:
            print(f"调用 calculate_factor_contributions 时出错: {e_contrib_call}")
        print("-" * 30)
        
        # --- 最终结果分析与保存 --- 
        script_end_time = time.time() # 移到这里，覆盖掉之前的定义
        total_runtime_seconds = script_end_time - script_start_time
        if final_dfm_results_obj is not None and data_for_analysis is not None: 
            try:
                print("\n调用最终结果分析与保存函数...")
                # print("DEBUG: About to call analyze_and_save_final_results...") # 添加调用前打印
                analyze_and_save_final_results(
                    run_output_dir=run_output_dir,
                    timestamp_str=timestamp_str,
                    excel_output_path=excel_output_file,
                    all_data_full=all_data_aligned_weekly,
                    data_for_analysis=data_for_analysis,
                    target_variable=TARGET_VARIABLE,
                    final_dfm_results=final_dfm_results_obj,
                    best_variables=final_variables,
                    best_params=final_params,
                    var_type_map=var_type_map,
                    best_avg_hit_rate_tuning=final_score_tuple[0], # 使用最终得分
                    best_avg_mae_tuning=-final_score_tuple[2], # 使用最终得分 (MAE/RMSE 取负)
                    total_runtime_seconds=total_runtime_seconds,
                    validation_start_date=VALIDATION_START_DATE,
                    validation_end_date=VALIDATION_END_DATE,
                    train_end_date=TRAIN_END_DATE,
                    heatmap_top_n_vars=HEATMAP_TOP_N_VARS,
                    factor_contributions=factor_contributions,
                    final_transform_log=final_transform_details,
                    pca_results_df=pca_results_df,
                    contribution_results_df=contribution_results_df,
                    var_industry_map=var_industry_map
                )
                # print("DEBUG: Returned from analyze_and_save_final_results.") # 添加调用后打印
            except Exception as e:
                print(f"分析和保存最终结果时出错: {e}")
                import traceback
                if log_file and not log_file.closed: log_file.write(f"\n!!! 分析保存错误: {e}")
                traceback.print_exc() # 打印错误信息
        else:
            print("警告: 无法执行最终分析和保存，缺少 DFM 结果或分析数据。")
            if log_file and not log_file.closed: log_file.write("\n!!! 警告: 无法执行最终分析保存。")

        # --- 最终绘图 --- 
        if final_dfm_results_obj is not None and data_for_analysis is not None:
            try:
                print("\n调用最终 Nowcast 绘图函数...")
                plot_final_nowcast(
                     run_output_dir=run_output_dir,
                     timestamp_str=timestamp_str,
                     all_data_full=all_data_aligned_weekly,
                     data_for_analysis=data_for_analysis,
                     target_variable=TARGET_VARIABLE,
                     final_dfm_results=final_dfm_results_obj,
                     best_k=final_params.get('k_factors', '未知'),
                     best_variables=final_variables,
                     target_mean_original=target_mean_original,
                     target_std_original=target_std_original,
                     validation_start=VALIDATION_START_DATE,
                     validation_end=VALIDATION_END_DATE
                )
            except Exception as e_final_plot:
                print(f"生成最终 Nowcast 图表时发生严重错误: {e_final_plot}")
                if log_file and not log_file.closed: log_file.write(f"\n!!! 生成图表错误: {e_final_plot}")
        else:
            print("警告: 无法生成最终 Nowcast 图表，缺少 DFM 结果或分析数据。")
            if log_file and not log_file.closed: log_file.write("\n!!! 警告: 无法生成图表。")

        # --- 结束信息 --- 
        print("\n--- 调优完成 --- ")
        print(f"总耗时: {total_runtime_seconds / 60:.2f} 分钟")
        print(f"结果已保存至: {run_output_dir}")
        print(f"总计进行了 {total_evaluations} 次参数评估。") # 移除滚动验证计数
        if svd_error_count > 0:
            print(f"警告: 评估期间发生了 {svd_error_count} 次 SVD 相关的错误。")
        # ... (其他最终打印信息)

        # 关闭日志文件
        if log_file and not log_file.closed:
            log_file.write("\n--- 日志结束 ---\
")
            try: log_file.close() 
            except Exception as close_e: print(f"关闭日志文件时出错: {close_e}")

        logging.info("DFM tuning process completed.")

    except Exception as e: # 添加 except 块
        logging.error(f"An error occurred during the tuning process:")
        logging.error(traceback.format_exc()) # 打印完整的追溯信息
        raise # 重新引发异常，以便脚本以非零退出代码结束

# --- 主程序入口 ---
if __name__ == "__main__":
    # print("DEBUG: Entering __main__ block...") # REMOVED
    # Ensure multiprocessing support for Windows freezing
    multiprocessing.freeze_support() # Keep this if needed
    # print("DEBUG: Multiprocessing support checked.") # REMOVED
    # print("DEBUG: Calling run_tuning()...") # REMOVED
    start_time = time.time()
    run_tuning() # Call the main function
    end_time = time.time()
    logging.info(f"Total execution time: {end_time - start_time:.2f} seconds")
    # print("DEBUG: run_tuning() finished.") # REMOVED