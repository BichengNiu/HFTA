# -*- coding: utf-8 -*-
"""
加载已保存的 DFM 模型和元数据，生成最终报告（Excel 和图表）。
(版本：硬编码文件路径)
"""
# import argparse # <-- 移除
import os
import pickle
import joblib
import pandas as pd
import numpy as np
import sys
from datetime import datetime
import logging
from typing import Optional # <-- 确保导入 Optional

# --- 配置日志 ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- 尝试导入分析模块 ---
try:
    # 直接使用绝对导入避免相对导入问题
    import sys
    import os
    current_script_dir = os.path.dirname(os.path.abspath(__file__))
    if current_script_dir not in sys.path:
        sys.path.insert(0, current_script_dir)
    
    from results_analysis import (
        analyze_and_save_final_results, 
        plot_final_nowcast, 
        write_r2_tables_to_excel, # Keep this if it's used elsewhere, though called by analyze_...
        plot_industry_vs_driving_factor,
        plot_factor_loading_clustermap, # <<< 新增导入
        plot_aligned_loading_comparison # <<< 新增导入
    )
    logger.info("Successfully imported results_analysis using absolute import")
    _RESULTS_ANALYSIS_AVAILABLE = True
except ImportError as e:
    logger.error(f"无法导入 results_analysis 模块: {e}")
    _RESULTS_ANALYSIS_AVAILABLE = False
    
    # 创建基础的mock函数
    def analyze_and_save_final_results(*args, **kwargs):
        logger.error("analyze_and_save_final_results not available due to import error")
        # 返回适当的类型
        return None, {}
    
    def plot_final_nowcast(*args, **kwargs):
        logger.warning("plot_final_nowcast not available due to import error")
        pass
        
    def write_r2_tables_to_excel(*args, **kwargs):
        logger.warning("write_r2_tables_to_excel not available")
        pass
        
    def plot_industry_vs_driving_factor(*args, **kwargs):
        logger.warning("plot_industry_vs_driving_factor not available")
        pass
        
    def plot_factor_loading_clustermap(*args, **kwargs):
        logger.warning("plot_factor_loading_clustermap not available")
        pass
        
    def plot_aligned_loading_comparison(*args, **kwargs):
        logger.warning("plot_aligned_loading_comparison not available")
        pass

def main(): # <-- 移除函数参数
    """
    主函数，加载文件并生成报告。
    """
    # --- 修复硬编码文件路径 ---
    # 获取当前脚本所在目录的上层目录，即dashboard/DFM目录
    current_script_dir = os.path.dirname(os.path.abspath(__file__))  # train_model目录
    dfm_dir = os.path.dirname(current_script_dir)  # DFM目录
    
    # 正确的输出目录路径
    default_result_dir = os.path.join(dfm_dir, 'outputs')
    model_path = os.path.join(default_result_dir, 'models', 'final_dfm_model.joblib')
    metadata_path = os.path.join(default_result_dir, 'models', 'final_dfm_metadata.pkl')
    output_dir = os.path.join(default_result_dir, 'reports')  # 输出到reports目录
    # --- 结束修复 ---

    logger.info(f"开始生成报告...")
    logger.info(f"  模型文件: {model_path}")
    logger.info(f"  元数据文件: {metadata_path}")
    logger.info(f"  输出目录: {output_dir}")

    # --- 路径检查 ---
    if not os.path.exists(model_path):
        logger.error(f"错误: 模型文件未找到: {model_path}")
        return
    if not os.path.exists(metadata_path):
        logger.error(f"错误: 元数据文件未找到: {metadata_path}")
        return
    os.makedirs(output_dir, exist_ok=True) # 确保输出目录存在
    # --- 结束路径检查 ---

    # --- 加载文件 ---
    try:
        final_dfm_results_obj = joblib.load(model_path)
        logger.info("成功加载模型文件 (.joblib)。")
    except Exception as e:
        logger.error(f"加载模型文件 '{model_path}' 时出错: {e}")
        return

    try:
        with open(metadata_path, 'rb') as f:
            metadata = pickle.load(f)
        logger.info("成功加载元数据文件 (.pkl)。")
    except Exception as e:
        logger.error(f"加载元数据文件 '{metadata_path}' 时出错: {e}")
        return

    # --- 调试代码：检查加载后的 metadata ---
    logger.debug("--- [Debug Load Check] 开始检查加载的 metadata ---")
    if isinstance(metadata, dict):
        logger.debug(f"  加载的 metadata 是字典，包含键: {list(metadata.keys())}")
        
        keys_to_check = ['all_data_aligned_weekly', 'target_mean_original', 'target_std_original', 'final_data_processed', 'training_only_lambda', 'training_only_factors_ts']
        for key in keys_to_check:
            if key in metadata:
                value = metadata[key]
                value_type = type(value)
                logger.debug(f"  键 '{key}': 存在, 类型: {value_type}")
                if hasattr(value, 'shape'):
                    logger.debug(f"    Shape: {value.shape}")
                elif not isinstance(value, (dict, list, tuple)): # Avoid printing large collections
                     logger.debug(f"    Value: {value}")
                if value is None:
                     logger.warning(f"    警告: 键 '{key}' 的值为 None!")
            else:
                logger.error(f"  错误: 键 '{key}' 在加载的 metadata 中不存在!")
    else:
        logger.error(f"  错误: 加载的对象不是字典类型! 类型为: {type(metadata)}")
    logger.debug("--- [Debug Load Check] 检查加载的 metadata 结束 ---")
    # --- 调试代码结束 ---

    # --- 确定输出文件名 (使用元数据时间戳) ---
    timestamp_str = metadata.get('timestamp', datetime.now().strftime("%Y%m%d_%H%M%S"))
    excel_output_file = os.path.join(output_dir, f"final_report_{timestamp_str}.xlsx")
    plot_output_file = os.path.join(output_dir, f"final_nowcast_comparison_{timestamp_str}.png")
    heatmap_output_file = os.path.join(output_dir, f"factor_loading_clustermap_{timestamp_str}.png") # <<< 新增：热力图文件名
    comparison_plot_output_file = os.path.join(output_dir, f"factor_loading_comparison_{timestamp_str}.png") # <<< 新增：稳定性对比图文件名

    # --- 提取所需参数 --- (这部分逻辑不变)
    logger.info("从元数据中提取参数...")
    try:
        # ... (提取参数的代码保持不变)
        # 必需参数检查
        required_keys = [
            'all_data_aligned_weekly', 'target_variable', 'best_variables', #'best_k_factors',
             #<-- best_k_factors 似乎不是硬性必须的，因为 best_params 里有? 暂时保留
             'best_params', # <-- 添加 best_params 检查
            'var_type_map', #'best_avg_hit_rate_tuning', 'best_avg_rmse_tuning',
            # <-- 移除对预计算指标的硬性依赖
            'total_runtime_seconds', 'validation_start_date', 'validation_end_date',
            'train_end_date', 'target_mean_original', 'target_std_original',
            'final_data_processed' # 用于构建 data_for_analysis
        ]
        missing_keys = [key for key in required_keys if key not in metadata or metadata.get(key) is None] # Use .get() for safer check
        if missing_keys:
            logger.error(f"元数据缺少以下必需键或其值为 None: {missing_keys}")
            return

        all_data_full = metadata['all_data_aligned_weekly']
        target_variable = metadata['target_variable']
        best_variables = metadata['best_variables']
        best_params = metadata['best_params'] # 已经检查过存在
        # best_k_factors = metadata['best_k_factors'] # <-- best_params 里有，可以不单独依赖
        var_type_map = metadata['var_type_map']
        # best_avg_hit_rate_tuning = metadata['best_avg_hit_rate_tuning'] # <-- 移除读取
        # best_avg_rmse_tuning = metadata['best_avg_rmse_tuning'] # <-- 移除读取 (注意 generate_report 写的是 rmse, 但调用时是 mae)
        total_runtime_seconds = metadata['total_runtime_seconds']
        validation_start_date = metadata['validation_start_date']
        validation_end_date = metadata['validation_end_date']
        train_end_date = metadata['train_end_date']
        target_mean_original = metadata['target_mean_original']
        target_std_original = metadata['target_std_original']
        final_data_processed = metadata['final_data_processed']
        final_transform_log = metadata.get('transform_details')
        pca_results_df = metadata.get('pca_results_df')
        contribution_results_df = metadata.get('contribution_results_df')
        factor_contributions = metadata.get('factor_contributions')
        var_industry_map = metadata.get('var_industry_map')
        individual_r2_results = metadata.get('individual_r2_results')
        industry_r2_results = metadata.get('industry_r2_results')
        factor_industry_r2_results = metadata.get('factor_industry_r2_results')
        factor_type_r2_results = metadata.get('factor_type_r2_results')
        logger.info(f"[Debug Report Gen] Loaded 'factor_type_r2_results'. Type: {type(factor_type_r2_results)}. Is None or Empty: {factor_type_r2_results is None or (isinstance(factor_type_r2_results, dict) and not factor_type_r2_results)}")
        if isinstance(factor_type_r2_results, dict) and factor_type_r2_results:
            logger.info(f"[Debug Report Gen] 'factor_type_r2_results' keys: {list(factor_type_r2_results.keys())}") # Log keys
            # logger.info(f\"[Debug Report Gen] \'factor_type_r2_results\' head: {dict(list(factor_type_r2_results.items())[:2])}\") # Log first few items
        all_data_aligned_weekly = metadata.get('all_data_aligned_weekly')
        training_start_date = metadata.get('training_start_date') # <-- 新增：提取训练开始日期

        # # --- DEBUG: 打印 contribution_results_df 类型 ---
        # print(f"DEBUG generate_report: Type of contribution_results_df from metadata.get: {type(contribution_results_df)}")
        # # --- DEBUG END ---

        data_for_analysis = {
            "final_data_processed": final_data_processed,
            "final_target_mean_rescale": target_mean_original,
            "final_target_std_rescale": target_std_original
        }
        logger.info("参数提取完成。")

    except KeyError as e:
        logger.error(f"从元数据中提取参数时出错，缺少键: {e}")
        return
    except Exception as e:
        logger.error(f"准备参数时发生意外错误: {e}")
        return

    # --- <<< 新增调试代码 开始 >>> ---
    logger.info("--- [Debug Stability Check] ---")
    loaded_training_lambda = metadata.get('training_only_lambda')
    loaded_training_factors = metadata.get('training_only_factors_ts')
    logger.info(f"[Debug Stability Check] Type of loaded_training_lambda: {type(loaded_training_lambda)}")
    logger.info(f"[Debug Stability Check] Type of loaded_training_factors: {type(loaded_training_factors)}")
    
    is_lambda_valid = isinstance(loaded_training_lambda, pd.DataFrame) and not loaded_training_lambda.empty
    is_factors_valid = isinstance(loaded_training_factors, pd.DataFrame) and not loaded_training_factors.empty
    logger.info(f"[Debug Stability Check] Is loaded_training_lambda a non-empty DataFrame? {is_lambda_valid}")
    logger.info(f"[Debug Stability Check] Is loaded_training_factors a non-empty DataFrame? {is_factors_valid}")
    
    if is_lambda_valid:
        logger.info(f"[Debug Stability Check] Shape of loaded_training_lambda: {loaded_training_lambda.shape}")
    if is_factors_valid:
         logger.info(f"[Debug Stability Check] Shape of loaded_training_factors: {loaded_training_factors.shape}")
         
    # 检查后续 if 条件所需的其他变量 (从模型对象或元数据获取最终载荷)
    final_loadings_for_check = None
    if final_dfm_results_obj and hasattr(final_dfm_results_obj, 'Lambda'):
        final_loadings_for_check = getattr(final_dfm_results_obj, 'Lambda')
        # 确保是DataFrame
        if not isinstance(final_loadings_for_check, pd.DataFrame) and isinstance(final_loadings_for_check, np.ndarray):
            try:
                final_variables_check = metadata.get('best_variables', []) # 需要最终变量列表做索引
                k_factors_check = final_loadings_for_check.shape[1]
                final_loadings_for_check = pd.DataFrame(
                    final_loadings_for_check,
                    index=final_variables_check,
                    columns=[f'Factor{i+1}' for i in range(k_factors_check)]
                )
            except Exception as e_conv:
                logger.warning(f"[Debug Stability Check] Failed to convert final loadings from ndarray: {e_conv}")
                final_loadings_for_check = None
        elif not isinstance(final_loadings_for_check, pd.DataFrame):
            final_loadings_for_check = None # 如果不是DF也不是ndarray，设为None
            
    logger.info(f"[Debug Stability Check] Type of final_model_loadings for check: {type(final_loadings_for_check)}")
    is_final_loadings_valid = isinstance(final_loadings_for_check, pd.DataFrame) and not final_loadings_for_check.empty
    logger.info(f"[Debug Stability Check] Is final_model_loadings a non-empty DataFrame? {is_final_loadings_valid}")
    if is_final_loadings_valid:
         logger.info(f"[Debug Stability Check] Shape of final_model_loadings: {final_loadings_for_check.shape}")
    
    condition_check = is_lambda_valid and is_final_loadings_valid # 模拟 if 条件的主要部分
    logger.info(f"[Debug Stability Check] Result of condition (lambda valid AND final loadings valid): {condition_check}")
    logger.info("--- [Debug Stability Check End] ---")
    # --- <<< 新增调试代码 结束 >>> ---

    # --- 调用分析和保存函数 --- 
    logger.info(f"调用 analyze_and_save_final_results 将 Excel 报告保存至: {excel_output_file}")
    calculated_nowcast = None # <-- 初始化返回值捕获变量
    main_excel_success = False # 标记主 Excel 是否成功
    try:
        # --- 捕获返回值 ---
        calculated_nowcast = analyze_and_save_final_results(
            run_output_dir=output_dir,
            timestamp_str=timestamp_str,
            excel_output_path=excel_output_file,
            all_data_full=all_data_full,
            final_data_processed=final_data_processed,
            final_target_mean_rescale=target_mean_original,
            final_target_std_rescale=target_std_original,
            target_variable=target_variable,
            final_dfm_results=final_dfm_results_obj,
            best_variables=best_variables,
            best_params=best_params,
            var_type_map=var_type_map,
            total_runtime_seconds=total_runtime_seconds,
            validation_start_date=validation_start_date,
            validation_end_date=validation_end_date,
            train_end_date=train_end_date,
            factor_contributions=factor_contributions,
            final_transform_log=final_transform_log,
            pca_results_df=pca_results_df,
            contribution_results_df=contribution_results_df,
            var_industry_map=var_industry_map,
            industry_r2_results=industry_r2_results,
            factor_industry_r2_results=factor_industry_r2_results,
            factor_type_r2_results=factor_type_r2_results,
            individual_r2_results=individual_r2_results,
            final_eigenvalues=metadata.get('final_eigenvalues'),
            training_start_date=training_start_date
        )
        if os.path.exists(excel_output_file):
            logger.info("主 Excel 报告生成完成。")
            main_excel_success = True
        else:
            logger.warning("analyze_and_save_final_results 调用完成，但未找到预期的 Excel 文件。")

    except Exception as e:
        logger.error(f"调用 analyze_and_save_final_results 时出错: {e}", exc_info=True)
        # 即使 Excel 生成失败，如果 nowcast 计算成功，仍然尝试绘图

    # --- 调用绘图函数 --- 
    logger.info(f"调用 plot_final_nowcast 生成对比图至: {plot_output_file}...")
    # --- 使用 analyze_and_save_final_results 返回的 nowcast --- 
    # 修复类型检查：calculated_nowcast可能是tuple、None或DataFrame
    nowcast_series_for_plot = None
    if calculated_nowcast is not None:
        # 如果是tuple，尝试提取第一个元素（通常是nowcast series）
        if isinstance(calculated_nowcast, tuple) and len(calculated_nowcast) > 0:
            nowcast_series_for_plot = calculated_nowcast[0]
        # 如果直接是DataFrame或Series
        elif hasattr(calculated_nowcast, 'empty'):
            nowcast_series_for_plot = calculated_nowcast
        else:
            logger.warning(f"calculated_nowcast 类型未知: {type(calculated_nowcast)}")
    
    # 检查是否有有效的nowcast数据用于绘图
    if (nowcast_series_for_plot is not None and 
        hasattr(nowcast_series_for_plot, 'empty') and 
        not nowcast_series_for_plot.empty):
        try:
            # 准备绘图所需的目标序列 (这个仍然需要)
            target_for_plot = all_data_full[target_variable].copy().dropna()

            plot_final_nowcast(
                final_nowcast_series=nowcast_series_for_plot, # <-- 使用处理后的 nowcast
                target_for_plot=target_for_plot,
                validation_start=validation_start_date,
                validation_end=validation_end_date,
                title=f'最终 DFM Nowcast vs 观测值 (原始水平) [Run: {timestamp_str}]',
                filename=plot_output_file
            )
            logger.info("最终 Nowcast 对比图生成完成。")
        except Exception as e:
            logger.error(f"调用 plot_final_nowcast 或准备绘图数据时出错: {e}", exc_info=True)
    else:
        logger.warning("无法生成 Nowcast 对比图，因为未能获取到有效的 Nowcast 序列数据。")

    # --- 新增：调用行业-驱动因子绘图函数 --- 
    logger.info("尝试生成行业与驱动因子对比图...")
    try:
        # 从元数据或模型结果中获取因子时间序列
        factors_ts = None
        if final_dfm_results_obj and hasattr(final_dfm_results_obj, 'x_sm'):
            factors_ts = final_dfm_results_obj.x_sm
            # 确保是 DataFrame
            if not isinstance(factors_ts, pd.DataFrame) and isinstance(factors_ts, np.ndarray):
                 k_factors_plot = factors_ts.shape[1]
                 # 尝试使用 final_data_processed 的索引，如果不行可能需要从元数据获取
                 index_for_factors = metadata.get('final_data_processed', pd.DataFrame()).index
                 factors_ts = pd.DataFrame(factors_ts, index=index_for_factors, columns=[f'Factor{i+1}' for i in range(k_factors_plot)])
            elif not isinstance(factors_ts, pd.DataFrame):
                 logger.warning("因子时间序列 (x_sm) 不是 DataFrame 或无法转换，无法生成行业图。")
                 factors_ts = None
                 
        # 检查所需数据是否齐全
        if factor_industry_r2_results and factors_ts is not None and final_data_processed is not None and var_industry_map and all_data_aligned_weekly is not None:
             plot_industry_vs_driving_factor(
                 factor_industry_r2=factor_industry_r2_results,
                 factors_ts=factors_ts,
                 data_processed=final_data_processed,
                 data_original_aligned=all_data_aligned_weekly,
                 var_industry_map=var_industry_map,
                 output_dir=output_dir # 保存到报告输出目录
             )
        else:
             # 打印具体哪个数据缺失
             missing_plot_data = []
             if not factor_industry_r2_results: missing_plot_data.append("Factor-Industry R2 结果")
             if factors_ts is None: missing_plot_data.append("因子时间序列")
             if final_data_processed is None: missing_plot_data.append("处理后的数据")
             if not var_industry_map: missing_plot_data.append("行业映射")
             if all_data_aligned_weekly is None: missing_plot_data.append("对齐后的原始数据")
             logger.warning(f"缺少绘制行业-驱动因子图所需的数据: {', '.join(missing_plot_data)}，跳过绘图。")
             
    except Exception as e_plot_industry:
         logger.error(f"生成行业-驱动因子对比图时出错: {e_plot_industry}", exc_info=True)
    # --- 结束新增 --- 

    # --- 新增：调用因子载荷聚类热力图函数 --- 
    logger.info("尝试生成因子载荷聚类热力图...")
    try:
        if hasattr(final_dfm_results_obj, 'Lambda') and final_dfm_results_obj.Lambda is not None:
            loadings_obj = final_dfm_results_obj.Lambda # 使用更通用的名字
            loadings_df = None # 初始化 DataFrame 变量

            # --- <<< 修改：检查 loadings_obj 是 DataFrame 或 2D NumPy Array >>> ---
            if isinstance(loadings_obj, pd.DataFrame):
                # 如果已经是 DataFrame，直接使用，但要确保索引和列名正确
                if isinstance(loadings_obj.index, pd.Index) and list(loadings_obj.index) == best_variables:
                    loadings_df = loadings_obj
                    # 确保列名是 FactorN 格式 (可选，但建议)
                    k_factors_loadings = loadings_df.shape[1]
                    expected_cols = [f'Factor{i+1}' for i in range(k_factors_loadings)]
                    if list(loadings_df.columns) != expected_cols:
                        logger.warning("因子载荷 DataFrame 列名不是预期的 FactorN 格式，将尝试重命名。")
                        try:
                           loadings_df.columns = expected_cols
                        except ValueError:
                           logger.error("重命名因子载荷 DataFrame 列名失败，因子数量可能不匹配。跳过热力图。")
                           loadings_df = None # 设为 None 以跳过绘图
                else:
                    logger.warning(f"因子载荷是 DataFrame，但其索引与 best_variables ({len(best_variables)}个) 不匹配。Index: {loadings_obj.index[:5]}...，无法生成载荷热力图。")

            elif isinstance(loadings_obj, np.ndarray) and loadings_obj.ndim == 2:
                # 如果是 NumPy 数组，执行之前的转换逻辑
                loadings_matrix = loadings_obj # 明确变量名
                k_factors_loadings = loadings_matrix.shape[1]
                if len(best_variables) == loadings_matrix.shape[0]:
                    factor_names_loadings = [f'Factor{i+1}' for i in range(k_factors_loadings)]
                    loadings_df = pd.DataFrame(
                        loadings_matrix, 
                        index=best_variables,
                        columns=factor_names_loadings
                    )
                    logger.info("因子载荷是 NumPy 数组，已成功转换为 DataFrame。")
                else:
                    logger.warning(f"因子载荷是 NumPy 数组，但其行数 ({loadings_matrix.shape[0]}) 与 best_variables 数量 ({len(best_variables)}) 不匹配，无法生成载荷热力图。")
            else:
                 logger.warning(f"因子载荷矩阵 (Lambda) 既不是 Pandas DataFrame 也不是 2D NumPy 数组 (实际类型: {type(loadings_obj)})，无法生成载荷热力图。")
            # --- <<< 结束修改 >>> ---
            
            # 只有成功获得了有效的 loadings_df 才进行绘图
            if loadings_df is not None and not loadings_df.empty:
                plot_factor_loading_clustermap(
                    loadings_df=loadings_df,
                    title=f'Factor Loadings Clustermap (Top 15 per Factor) [Run: {timestamp_str}]',
                    filename=heatmap_output_file,
                    top_n_vars=15
                )
                logger.info(f"因子载荷聚类热力图调用完成，尝试保存至: {heatmap_output_file}")
            elif loadings_df is None: # 如果是因为之前的检查失败而设置为 None
                 pass # 警告已经在之前的逻辑中打印
            else: # loadings_df 不为 None 但可能为空 (虽然不太可能发生在此逻辑中)
                 logger.warning("因子载荷 DataFrame 为空，跳过生成热力图。")

        else:
            logger.warning("模型结果中未找到有效的因子载荷矩阵 (Lambda)，无法生成载荷热力图。")
            
    except Exception as e_plot_heatmap:
         logger.error(f"生成因子载荷聚类热力图时出错: {e_plot_heatmap}", exc_info=True)
    # --- 结束新增 ---

    # --- <<< 新增：绘制全样本与仅训练期因子载荷对比图 >>> ---
    logger.info(f"绘制因子载荷稳定性对比图到 {comparison_plot_output_file}...")
    try:
        # 检查 metadata 中是否存在所需的训练期载荷信息
        training_only_lambda = metadata.get('training_only_lambda') # 从 metadata 获取训练期载荷
        variables = metadata.get('best_variables') # 获取变量名列表

        if training_only_lambda is not None and variables is not None:
            # --- 修改：从模型结果对象获取全样本载荷 ---
            lambda_final = None
            if final_dfm_results_obj and hasattr(final_dfm_results_obj, 'Lambda'):
                lambda_final = getattr(final_dfm_results_obj, 'Lambda')
                if lambda_final is None:
                    logger.warning("从 final_dfm_results_obj.Lambda 获取的全样本载荷为 None。")
            else:
                logger.warning("模型结果对象 final_dfm_results_obj 无效或缺少 'Lambda' 属性。")
            # --- 结束修改 ---

            # 检查所有必需信息是否都已获取
            if lambda_final is not None:
                # 此时 lambda_final 可能是 DataFrame 或 ndarray
                # plot_aligned_loading_comparison 函数内部会处理 ndarray 的转换
                # 但我们仍需确保变量列表长度与载荷矩阵的行数匹配
                
                # 获取 lambda_final 的行数 (无论它是 DF 还是 ndarray)
                lambda_final_rows = lambda_final.shape[0] if hasattr(lambda_final, 'shape') else 0
                training_lambda_rows = training_only_lambda.shape[0] if hasattr(training_only_lambda, 'shape') else 0
                
                if len(variables) == lambda_final_rows and len(variables) == training_lambda_rows:
                    plot_aligned_loading_comparison(
                        lambda_full=lambda_final,      # 直接传递获取到的对象
                        lambda_train=training_only_lambda, # 从 metadata 获取的对象
                        variables=variables,            # 变量名列表
                        output_path=comparison_plot_output_file
                    )
                    logger.info(f"因子载荷稳定性对比图函数调用完成，尝试保存至: {comparison_plot_output_file}")
                else:
                    logger.warning(f"无法绘制因子载荷稳定性对比图，变量数量 ({len(variables)}) 与载荷矩阵维度不匹配 (全样本: {lambda_final_rows}, 训练期: {training_lambda_rows})。")

            else:
                # lambda_final 为 None 的情况 (获取失败)
                logger.warning(f"无法绘制因子载荷稳定性对比图，因为未能从模型结果中获取有效的全样本载荷。")
        else:
            # training_only_lambda 或 variables 为 None 的情况
            missing_meta_info = []
            if training_only_lambda is None: missing_meta_info.append("训练期载荷 ('training_only_lambda')")
            if variables is None: missing_meta_info.append("变量名列表 ('best_variables')")
            logger.warning(f"无法绘制因子载荷稳定性对比图，因为元数据中缺少以下信息: {', '.join(missing_meta_info)}。")

    except Exception as e:
        logger.error(f"绘制因子载荷稳定性对比图时发生意外错误: {e}", exc_info=True)
    # --- <<< 结束新增 >>> ---

    logger.info("报告生成过程结束。")

# --- 添加generate_excel_main函数作为main的别名 ---
def generate_excel_main(*args, **kwargs):
    """
    生成Excel报告的主函数（main函数的别名）
    为保持向后兼容性而提供
    """
    try:
        return main()
    except Exception as e:
        logger.error(f"生成Excel报告失败: {e}")
        return None

# --- 直接调用 main --- 
if __name__ == "__main__":
    main() 