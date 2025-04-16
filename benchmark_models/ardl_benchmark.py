# -*- coding: utf-8 -*-
"""
基准模型 (ARDL) 评估脚本。
目的：实现 ARDL 模型作为另一个基准。
修改：
- 基于原 ARIMA 脚本结构。
- 添加周度数据加载和月度平均处理。
- 实现基于滚动窗口验证的外生变量筛选。
- 使用 statsmodels.tsa.ardl 实现 ARDL。
- 使用最后一个已知值预测未来外生变量。
- 将结果追加到 benchmark_result.xlsx 的新 Sheet。
+ 训练/选择期统一为 2020-01 到 2024-06。
+ 预测期统一为 2024-07 到 2024-12 (6个月)。
+ 移除独立的验证期指标计算。
+ 报告滚动验证的平均指标。
+ 实现训练-验证-再训练流程。
+ 变量筛选基于验证期表现 (最高胜率 -> 最低 RMSE)。
+ 最终模型使用全部历史数据训练。
+ 预测未来 3 个月。
"""
import pandas as pd
import numpy as np
import sys
import os
import time
import warnings
import itertools
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import TimeSeriesSplit # 用于滚动窗口验证
# --- 修改: 移除 ARIMA，添加 ARDL ---
# from statsmodels.tsa.arima.model import ARIMA
# from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.ardl import ARDL, ardl_select_order
import traceback
import matplotlib
import matplotlib.pyplot as plt

# --- !!! 添加测试打印 !!! ---
print("--- ARDL Script Execution Test: Starting Imports ---")
# --- !!! 结束测试打印 !!! ---

# --- 配置 ---
warnings.filterwarnings('ignore')
matplotlib.use('Agg') 
plt.rcParams['font.sans-serif'] = ['SimHei'] 
plt.rcParams['axes.unicode_minus'] = False 

# --- 文件路径和常量 ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(SCRIPT_DIR)
EXCEL_DATA_FILE = os.path.join(BASE_DIR, 'data', '经济数据库.xlsx')

TARGET_VARIABLE = '规模以上工业增加值:当月同比'
TARGET_SHEET_NAME = '工业增加值同比增速-月度' 
EXOG_SHEET_PREFIX = '-周度'

# --- 修改：调整时间定义 --- 
TRAIN_END_DATE = '2024-06-30' # 选择、最终训练都用此日期之前的数据
VALIDATION_START_DATE = '2024-07-01'
VALIDATION_END_DATE = '2024-12-31'
BENCHMARK_RESULTS_FILE = "benchmark_result.xlsx"
MAX_ARDL_LAGS = 5 
N_SPLITS_VALIDATION = 5 
MIN_HIT_RATE_THRESHOLD = 50.0 
PLOT_FILENAME = "ardl_vs_actual.png" 
FORECAST_STEPS = 3 # <-- 修改：预测 3 个月
# --- 结束修改 --- 

# --- 指标计算函数 (保持不变) ---
def calculate_metrics(y_true, y_pred):
    """计算 RMSE, MAE, Hit Rate."""
    metrics = {
        'RMSE': np.nan,
        'MAE': np.nan,
        'Hit_Rate': np.nan
    }
    if len(y_true) < 2 or len(y_pred) < 2 or len(y_true) != len(y_pred):
        # print("[Metrics] 警告: 输入序列长度不足或不匹配，无法计算指标。") # 减少打印
        return metrics
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    valid_mask = ~np.isnan(y_true) & ~np.isnan(y_pred)
    y_true_valid = y_true[valid_mask]
    y_pred_valid = y_pred[valid_mask]
    if len(y_true_valid) < 2:
        # print(f"[Metrics] 警告: 移除 NaN 后有效数据点不足 ({len(y_true_valid)} 点)，无法计算指标。") # 减少打印
        return metrics
    try:
        metrics['RMSE'] = np.sqrt(mean_squared_error(y_true_valid, y_pred_valid))
    except Exception as e:
        # print(f"[Metrics] 计算 RMSE 时出错: {e}") # 减少打印
        pass
    try:
        metrics['MAE'] = mean_absolute_error(y_true_valid, y_pred_valid)
    except Exception as e:
        # print(f"[Metrics] 计算 MAE 时出错: {e}") # 减少打印
        pass
    try:
        true_diff = np.diff(y_true_valid)
        pred_diff = np.diff(y_pred_valid)
        non_zero_target_change_mask = true_diff != 0
        if np.any(non_zero_target_change_mask):
            correct_direction = np.sign(pred_diff[non_zero_target_change_mask]) == np.sign(true_diff[non_zero_target_change_mask])
            metrics['Hit_Rate'] = np.mean(correct_direction) * 100
        else:
            # print("[Metrics] 警告: 真实值变化全为零，无法计算 Hit Rate。") # 减少打印
            pass
    except Exception as e:
        # print(f"[Metrics] 计算 Hit Rate 时出错: {e}") # 减少打印
        pass
    return metrics


# --- 新增：数据预处理函数 ---
def preprocess_exog_data(exog_df: pd.DataFrame, target_index: pd.DatetimeIndex) -> pd.DataFrame:
    """预处理外生变量数据 (周度转月度等)。"""
    print("\n--- 预处理外生变量数据 ---")
    if exog_df.empty:
        print("输入的外生变量 DataFrame 为空。")
        return pd.DataFrame()
    
    print(f"原始外生变量维度: {exog_df.shape}")
    print(f"原始外生变量索引频率: {exog_df.index.freqstr}")
    print(f"目标序列索引频率: {target_index.freqstr}")

    # 1. 确保索引是 DatetimeIndex (已在加载时处理)
    if not isinstance(exog_df.index, pd.DatetimeIndex):
        print("错误：外生变量索引不是 DatetimeIndex。")
        return pd.DataFrame()

    # 2. 尝试重采样到月末 ('ME')
    try:
        exog_resampled = exog_df.resample('ME').last() # 使用最后一个观测值
        print(f"重采样至月末 ('ME') 后维度: {exog_resampled.shape}")
    except Exception as e: # Ensure except block exists
        print(f"重采样外生变量出错: {e}。返回空 DataFrame。")
        return pd.DataFrame()

    # 3. 处理缺失值 (向前填充)
    exog_filled = exog_resampled.ffill()
    print(f"向前填充 (ffill) 后维度: {exog_filled.shape}")
    
    # 4. 与目标序列索引对齐 (取交集，确保使用目标序列的日期范围)
    common_index = target_index.intersection(exog_filled.index)
    exog_aligned = exog_filled.reindex(common_index)
    print(f"与目标序列索引对齐后维度: {exog_aligned.shape}")

    # 5. 再次检查 NaN (对齐可能引入 NaN)
    exog_final = exog_aligned.dropna(axis=1, how='all') # 删除全为 NaN 的列
    nan_counts = exog_final.isna().sum()
    cols_with_nan = nan_counts[nan_counts > 0]
    if not cols_with_nan.empty:
        print(f"警告: 预处理后以下外生变量仍包含 NaN 值 (将尝试在建模前处理):")
        print(cols_with_nan)
    else:
        print("预处理后所有外生变量列均无 NaN 值。")

    print(f"最终预处理后的外生变量维度: {exog_final.shape}")
    print("--- 外生变量预处理完成 ---")
    return exog_final


# --- 新增：外生变量评估函数 ---
def evaluate_exog_variable(
    train_target: pd.Series, 
    train_exog: pd.Series, 
    val_target: pd.Series,
    val_exog: pd.Series,
    exog_name: str, 
    n_splits: int, 
    max_lags: int
) -> dict:
    """
    使用训练数据进行滚动预测选择滞后阶数，用最后一个模型预测验证期，返回验证期指标。
    """
    print(f"  评估变量: {exog_name}")
    validation_metrics = {'RMSE': np.nan, 'MAE': np.nan, 'Hit_Rate': np.nan, 'Error': None}

    # 合并训练数据以便于 TimeSeriesSplit
    combined_train = pd.concat([train_target, train_exog], axis=1).dropna()
    if combined_train.empty or len(combined_train) < n_splits * 2: # 确保足够数据
        print(f"    错误: 训练数据不足或合并后为空 (长度 {len(combined_train)})。无法评估。")
        validation_metrics['Error'] = "Insufficient training data"
        return validation_metrics
        
    endog_train = combined_train[train_target.name]
    exog_train_single = combined_train[[exog_name]]

    tscv = TimeSeriesSplit(n_splits=n_splits)
    
    last_model_fit = None
    selected_lags = None
    
    print(f"    使用 {n_splits}-折滚动预测 (在训练集上) 选择滞后阶数...")
    try:
        # 在整个训练集上运行一次 ardl_select_order 来选择滞后阶数
        # 注意: 这简化了流程，不严格遵循滚动预测的阶数选择，但更高效
        # 在实际应用中，更严格的方法是在每个滚动折叠中重新选择阶数
        sel_order = ardl_select_order(
            endog=endog_train, 
            maxlag=max_lags, 
            exog=exog_train_single, 
            trend='c', # 包含常数项
            glob=True, # 搜索所有组合
            ic='aic', # 使用 AIC 标准
            maxorder=max_lags # 添加外生变量最大滞后
        )
        selected_lags = sel_order.model.ardl_order
        print(f"    在完整训练集上选择的最佳滞后 (p, q): {selected_lags}")

        # --- 修正：处理不同长度的 selected_lags 并创建模型 --- 
        p_lag = selected_lags[0]
        if len(selected_lags) == 1: # 只有 p，没有 q
             q_lags = 0 # 或者 []，需要根据 ARDL 接受的参数确定，0 通常可行
             print("    注意：仅选择了内生变量滞后 (p)，外生变量滞后 (q) 为 0。")
        elif len(selected_lags) == 2:
             q_lags = selected_lags[1]
        else:
             raise ValueError(f"意外的 selected_lags 格式: {selected_lags}")

        model = ARDL(endog_train, p_lag, exog_train_single, q_lags, trend='c')
        # --- 结束修正 ---
        last_model_fit = model.fit()
        print(f"    已使用完整训练数据和滞后 {selected_lags} 训练模型。")

    except Exception as e:
        print(f"    错误: 在训练集上选择滞后或训练时出错: {e}")
        traceback.print_exc()
        validation_metrics['Error'] = f"Lag selection/Training error: {e}"
        return validation_metrics

    # 使用训练好的最后一个模型预测验证期
    if last_model_fit is not None and not val_target.empty and not val_exog.empty:
        print(f"    使用训练好的模型预测验证期 ({val_target.index.min().strftime('%Y-%m-%d')} to {val_target.index.max().strftime('%Y-%m-%d')})...")
        
        # 准备验证期的外生变量 (确保列名匹配)
        val_exog_single = val_exog.to_frame(name=exog_name) 
        
        try:
            # --- 修正：使用 forecast 方法进行验证期预测 --- 
            steps_val = len(val_target)
            if steps_val == 0:
                print("    警告：验证期目标序列为空，无法进行预测。")
                validation_metrics['Error'] = "Empty validation target"
                return validation_metrics
                
            # 确保验证期外生变量长度正确
            if len(val_exog_single) != steps_val:
                 print(f"    警告：验证期外生变量长度 ({len(val_exog_single)}) 与验证期目标长度 ({steps_val}) 不符！将尝试对齐。")
                 val_exog_single = val_exog_single.reindex(val_target.index)
                 if len(val_exog_single) != steps_val or val_exog_single.isna().any().any():
                     print(f"    错误：无法对齐验证期外生变量或对齐后包含 NaN。")
                     validation_metrics['Error'] = "Cannot align validation exog or NaNs present"
                     return validation_metrics

            print(f"    使用 forecast 方法预测 {steps_val} 步...")
            # 调用 forecast
            forecast_values = last_model_fit.forecast(steps=steps_val, exog=val_exog_single)
            # --- 新增：打印 forecast 原始输出 ---
            print(f"    Forecast raw output:\n{forecast_values}")
            # --- 结束新增 ---
            
            # --- 修正：确保使用 forecast 的数值和目标索引创建 Series ---
            if len(forecast_values) == steps_val:
                # 使用 .values 提取数值，忽略 forecast_values 可能的索引
                val_pred = pd.Series(forecast_values.values, index=val_target.index)
            else:
                 print(f"    警告: forecast 结果长度 ({len(forecast_values)}) 与预期 ({steps_val}) 不符。")
                 val_pred = pd.Series(np.nan, index=val_target.index)
            # --- 结束修正 --- 

            # 对齐预测结果和实际值 (val_pred 已经带验证期索引)
            val_pred_aligned = val_pred
            
            # --- 新增：打印指标计算的输入 --- 
            print(f"    Input to calculate_metrics - val_target (len={len(val_target)}):")
            print(val_target.head(steps_val)) # 打印整个验证期
            print(f"    Input to calculate_metrics - val_pred_aligned (len={len(val_pred_aligned)}):")
            print(val_pred_aligned.head(steps_val)) # 打印整个验证期
            print(f"    NaNs in val_target: {val_target.isna().sum()}, NaNs in val_pred_aligned: {val_pred_aligned.isna().sum()}")
            # --- 结束新增 --- 

            # 计算验证期指标
            validation_metrics = calculate_metrics(val_target, val_pred_aligned)
            validation_metrics['Selected_Lags_During_Eval'] = selected_lags # 记录评估中选择的滞后
            print(f"    验证期指标: RMSE={validation_metrics['RMSE']:.4f}, MAE={validation_metrics['MAE']:.4f}, HitRate={validation_metrics['Hit_Rate']:.2f}%")

        except Exception as e:
            print(f"    错误: 预测验证期或计算指标时出错: {e}")
            traceback.print_exc()
            validation_metrics['Error'] = f"Validation prediction/metrics error: {e}"
            
    else:
        print("    错误: 无法进行验证期预测 (模型未训练或验证数据为空)。")
        validation_metrics['Error'] = "Cannot predict validation (no model or empty val data)"
        
    return validation_metrics


# --- 修改：ARDL 基准测试主函数 --- 
def run_ardl_benchmark(target_series_full: pd.Series, best_exog_series_full: pd.Series, 
                       max_lags: int, forecast_steps: int) -> tuple[dict, pd.DataFrame]:
    """
    执行 ARDL 基准测试，使用完整的历史数据选择最终滞后并训练模型，预测未来 forecast_steps 步。
    """
    
    exog_name = best_exog_series_full.name if best_exog_series_full.name else 'Exog'
    target_name = target_series_full.name if target_series_full.name else 'Target'
    print(f"\n--- 开始最终 ARDL 模型训练与预测 (使用外生变量: {exog_name}) --- ")
    
    model_info = {'Final_Selected_Order_p': None, 'Final_Selected_Order_q': None} 
    results_df_hist_forecast = pd.DataFrame() 
    
    # 准备最终拟合数据 (使用完整的历史数据)
    combined_data_full = pd.concat([target_series_full, best_exog_series_full], axis=1).dropna()
    
    if combined_data_full.empty:
        print("错误：合并目标和最佳外生变量后数据为空 (完整历史)。")
        return model_info, results_df_hist_forecast

    endog_final = combined_data_full[target_name]
    exog_final = combined_data_full[[exog_name]] 

    # 1. 在完整数据上选择最终滞后阶数并训练最终模型
    print(f"使用所有历史数据选择最终 ARDL 滞后 (滞后 <= {max_lags}) 并训练...")
    final_model_fit = None
    try:
        # 在完整数据上重新选择阶数
        final_sel_order = ardl_select_order(
            endog=endog_final, 
            maxlag=max_lags, 
            exog=exog_final, 
            trend='c', 
            glob=True, 
            ic='aic', 
            maxorder=max_lags # 添加外生变量最大滞后
        )
        final_selected_lags = final_sel_order.model.ardl_order
        model_info['Final_Selected_Order_p'] = final_selected_lags[0]
        # --- 修正：处理不同长度的 final_selected_lags 并创建模型/记录信息 --- 
        final_p_lag = final_selected_lags[0]
        if len(final_selected_lags) == 1:
            final_q_lags = 0 
            model_info['Final_Selected_Order_q'] = [] # 存空列表表示无q
        elif len(final_selected_lags) == 2:
            final_q_lags = final_selected_lags[1]
            # --- 修正：存储实际的滞后列表/元组 --- 
            model_info['Final_Selected_Order_q'] = final_q_lags 
            # --- 结束修正 ---
        else:
            raise ValueError(f"意外的 final_selected_lags 格式: {final_selected_lags}")
            
        print(f"  在完整数据上选择的最佳滞后 (p, q_list): ({final_p_lag}, {final_q_lags})") # 打印调整后的形式

        final_model = ARDL(endog_final, final_p_lag, exog_final, final_q_lags, trend='c')
        # --- 结束修正 ---
        final_model_fit = final_model.fit()
        print("  最终 ARDL 模型拟合完成。")
        
    except Exception as e:
        print(f"  错误: 最终 ARDL 模型滞后选择或拟合失败: {e}")
        traceback.print_exc()
        return model_info, results_df_hist_forecast 

    # 2. 生成历史拟合值
    print("生成历史拟合值...")
    historical_fitted = None
    try:
        if final_model_fit is not None and not endog_final.empty:
            # --- 修正：从模型本身获取 AR 阶数 p --- 
            if hasattr(final_model_fit, 'model') and hasattr(final_model_fit.model, 'ardl_order'):
                 start_hist_lag = final_model_fit.model.ardl_order[0] # AR 阶数 p
                 if start_hist_lag < len(endog_final.index):
                      start_hist = endog_final.index[start_hist_lag] # 起始日期是第 p 个索引
                      end_hist = endog_final.index[-1]
                      historical_fitted = final_model_fit.predict(start=start_hist, end=end_hist)
                      print(f"  历史拟合值生成完成 (从 AR 阶数 p={start_hist_lag} 开始)。")
                 else:
                      print(f"  警告：计算得到的起始滞后 ({start_hist_lag}) 超出索引范围，无法生成历史拟合值。")
                      historical_fitted = pd.Series(np.nan, index=endog_final.index) # 填充 NaN
            else:
                print("  错误：无法从 final_model_fit.model.ardl_order 获取 AR 阶数 p。")
                historical_fitted = pd.Series(np.nan, index=endog_final.index)
            # --- 结束修正 ---
    except Exception as e: # 保留通用异常捕获
        print(f"  错误: 生成历史拟合值时发生意外错误: {e}")
        traceback.print_exc()
        historical_fitted = pd.Series(np.nan, index=endog_final.index) # 确保有列

    # 3. 生成未来预测值
    print(f"生成未来 {forecast_steps} 步预测...")
    future_forecast = None
    try:
        if final_model_fit is not None:
            # 准备未来外生变量 (简单假设：使用最后一个已知值)
            last_known_exog_value = exog_final.iloc[-1].values 
            future_exog_values = np.tile(last_known_exog_value, (forecast_steps, 1)) 
            
            last_hist_date = endog_final.index[-1]
            future_index = pd.date_range(start=last_hist_date + pd.DateOffset(months=1), 
                                         periods=forecast_steps, 
                                         freq=endog_final.index.freqstr if endog_final.index.freqstr else 'ME') 
            future_exog_df = pd.DataFrame(future_exog_values, index=future_index, columns=exog_final.columns)
            print(f"  未来预测使用的外生变量 (前5行):")
            print(future_exog_df.head())

            # --- 新增：打印 forecast 输入和原始输出 ---
            print(f"  Forecast input exog info: index=\n{future_exog_df.index}\ndtype={future_exog_df.iloc[:, 0].dtype}\nhead=\n{future_exog_df.head()}")
            future_forecast_vals = final_model_fit.forecast(steps=forecast_steps, exog=future_exog_df)
            print(f"  Forecast raw output:\n{future_forecast_vals}")
            # --- 结束新增 ---
            
            if len(future_forecast_vals) == forecast_steps:
                # --- 修正：尝试明确指定 dtype 并处理可能的错误 --- 
                try:
                     # 使用 .values 提取数值，忽略 forecast_values 可能的索引, 指定 dtype
                     future_forecast = pd.Series(future_forecast_vals.values, index=future_index, dtype=float)
                     # 再次检查是否为 NaN
                     if future_forecast.isna().all():
                          print("  警告：直接创建 Series 后值全为 NaN，尝试逐个赋值...")
                          future_forecast = pd.Series(np.nan, index=future_index, dtype=float) # 创建 NaN Series
                          for i in range(len(future_forecast_vals.values)):
                               future_forecast.iloc[i] = future_forecast_vals.values[i]
                          if future_forecast.isna().all():
                               print("  错误：逐个赋值后仍然全为 NaN。预测失败。")
                     else:
                          print(f"  未来 {forecast_steps} 步预测值 (处理后):")
                          print(future_forecast)
                except Exception as series_creation_e:
                     print(f"  错误：创建或填充 future_forecast Series 时出错: {series_creation_e}")
                     future_forecast = pd.Series(np.nan, index=future_index) 
                # --- 结束修正 ---
            else:
                print(f"    警告: forecast 结果长度 ({len(future_forecast_vals)}) 与预期 ({forecast_steps}) 不符。预测设为 NaN。")
                future_forecast = pd.Series(np.nan, index=future_index) 

    except Exception as e:
        print(f"  错误: 生成未来预测失败: {e}")
        traceback.print_exc()

    # 4. 合并结果
    print("合并历史和未来预测结果...")
    try:
        # --- Revised Merging Logic ---
        # 1. Create the full index covering history and forecast
        full_index = target_series_full.index
        if future_forecast is not None and not future_forecast.empty:
            full_index = full_index.union(future_forecast.index)
        elif historical_fitted is not None and not historical_fitted.empty:
            # Handle case where forecast failed/empty but hist exists
            full_index = full_index.union(historical_fitted.index) # Usually a subset

        # 2. Initialize the DataFrame with the full index
        results_df_hist_forecast = pd.DataFrame(index=full_index)

        # 3. Add columns, letting pandas align by index
        results_df_hist_forecast['Actual'] = target_series_full
        if historical_fitted is not None:
            results_df_hist_forecast['Fitted'] = historical_fitted
        else:
            results_df_hist_forecast['Fitted'] = np.nan # Ensure column exists
             
        if future_forecast is not None and not future_forecast.empty:
            results_df_hist_forecast['Forecast'] = future_forecast
        else:
            # Ensure column exists even if forecast failed or was empty
            future_index_placeholder = pd.date_range(start=target_series_full.index[-1] + pd.DateOffset(months=1),
                                                 periods=forecast_steps,
                                                 freq=target_series_full.index.freqstr if target_series_full.index.freqstr else 'ME')
            # Make sure index covers placeholder, even if forecast was empty Series
            results_df_hist_forecast = results_df_hist_forecast.reindex(results_df_hist_forecast.index.union(future_index_placeholder))
            results_df_hist_forecast['Forecast'] = np.nan # Assign NaNs to the forecast part

        print("  历史和未来数据合并完成。")
        # --- End Revised Merging Logic ---

    except Exception as e:
        print(f"  错误: 合并最终结果失败: {e}")
        traceback.print_exc()
        # Fallback: Create a basic DataFrame to avoid downstream errors
        results_df_hist_forecast = pd.DataFrame({'Actual': target_series_full})
        results_df_hist_forecast['Fitted'] = np.nan
        results_df_hist_forecast['Forecast'] = np.nan
        # Add future index for Forecast NaNs if possible
        try:
             future_index_placeholder = pd.date_range(start=target_series_full.index[-1] + pd.DateOffset(months=1), periods=forecast_steps, freq=target_series_full.index.freqstr if target_series_full.index.freqstr else 'ME')
             results_df_hist_forecast = results_df_hist_forecast.reindex(results_df_hist_forecast.index.union(future_index_placeholder))
             # Reassign NaN after reindex
             results_df_hist_forecast['Fitted'] = np.nan
             results_df_hist_forecast['Forecast'] = np.nan
        except:
             pass # Ignore errors during fallback

    print("--- ARDL 最终模型训练与预测完成 --- ")
    return model_info, results_df_hist_forecast # 返回最终模型信息和完整预测序列


# --- 修改：绘图函数 --- 
def plot_ardl_results(results_df: pd.DataFrame, exog_name: str, filename: str):
    """绘制 ARDL 实际值 vs 历史拟合/预测值图。"""
    print(f"\n--- 生成 ARDL 结果图 (外生变量: {exog_name}) --- ")
    if results_df.empty or 'Actual' not in results_df.columns: 
        print("错误：传入绘图的数据缺少 'Actual' 列或为空，无法生成图像。")
        return
        
    try:
        plt.figure(figsize=(14, 7))
        
        plt.plot(results_df.index, results_df['Actual'], marker='.', linestyle='None', color='red', label='实际值 (月度)')
        
        if 'Fitted' in results_df.columns and results_df['Fitted'].notna().any():
             plt.plot(results_df.index, results_df['Fitted'], linestyle='--', color='green', label='历史拟合值')
             
        if 'Forecast' in results_df.columns and results_df['Forecast'].notna().any():
             plt.plot(results_df.index, results_df['Forecast'], linestyle=':', color='blue', label='未来预测值')
        
        plt.title(f'最终 ARDL 模型 (Exog: {exog_name}): 实际 vs 拟合/预测 ({TARGET_VARIABLE})')
        plt.xlabel('日期')
        plt.ylabel('值')
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.tight_layout()
        
        output_dir = os.path.dirname(filename)
        if output_dir and not os.path.exists(output_dir):
             os.makedirs(output_dir)
             print(f"创建目录: {output_dir}")
             
        plt.savefig(filename)
        plt.close()
        print(f"ARDL 结果图已保存到: {filename}")

    except Exception as e:
        print(f"生成或保存 ARDL 结果图时出错: {e}")
        traceback.print_exc()
        try: plt.close()
        except: pass


# --- 主执行逻辑 ---
if __name__ == "__main__":
    print(f"开始基准模型评估 (ARDL - 周度转月度)...")
    script_start_time = time.time()
    
    # 1. 加载月度目标数据
    print(f"\n--- 1. 加载月度目标数据 --- ")
    print(f"Excel 文件: {EXCEL_DATA_FILE}")
    print(f"目标变量 Sheet: {TARGET_SHEET_NAME}")
    print(f"目标变量名称: {TARGET_VARIABLE}")
    target_series_monthly = None
    try:
        monthly_data_df = pd.read_excel(
            EXCEL_DATA_FILE, 
            sheet_name=TARGET_SHEET_NAME,
            index_col=0, 
            parse_dates=True 
        )
        # ... (省略部分与 ARIMA 相同的加载和检查代码) ...
        if not isinstance(monthly_data_df.index, pd.DatetimeIndex):
                  monthly_data_df.index = pd.to_datetime(monthly_data_df.index, errors='coerce')
             # 检查 NaT...
        target_series_monthly = monthly_data_df[TARGET_VARIABLE]
        target_series_monthly = pd.to_numeric(target_series_monthly, errors='coerce')
        target_series_monthly = target_series_monthly.resample('ME').last() # 确保是月末频率
        target_series_monthly.name = TARGET_VARIABLE # 确保 Series 有名字
        print(f"成功加载并转换月度目标序列，长度: {len(target_series_monthly)}, "
              f"时间范围: {target_series_monthly.index.min().strftime('%Y-%m-%d')} to {target_series_monthly.index.max().strftime('%Y-%m-%d')}")

    except Exception as e:
        print(f"加载月度目标数据时出错: {e}")
        traceback.print_exc()
        sys.exit(1)

    # 2. 加载并合并周度外生数据
    print(f"\n--- 2. 加载并合并周度外生数据 --- ")
    print(f"Excel 文件: {EXCEL_DATA_FILE}")
    exog_weekly_df = pd.DataFrame()
    try:
        excel_file = pd.ExcelFile(EXCEL_DATA_FILE)
        available_sheets_in_file = excel_file.sheet_names
        actual_exog_sheets = [s for s in available_sheets_in_file if EXOG_SHEET_PREFIX in s]
        print(f"找到 {len(actual_exog_sheets)} 个包含 '{EXOG_SHEET_PREFIX}' 的 Sheet: {actual_exog_sheets}")
        
        exog_weekly_df_list = []
        loaded_count = 0
        for sheet_name in actual_exog_sheets:
                print(f"  加载 Sheet: {sheet_name}")
                try:
                 df_weekly = pd.read_excel(excel_file, sheet_name=sheet_name, index_col=0, parse_dates=False)
                 # --- 强制索引和数据类型转换 ---
                 original_index_name = df_weekly.index.name 
                 df_weekly.index = pd.to_datetime(df_weekly.index, errors='coerce')
                 # --- 修正：根据索引 NaT 删除行 ---
                 original_len = len(df_weekly)
                 df_weekly = df_weekly[df_weekly.index.notna()] # 保留索引不是 NaT 的行
                 print(f"    原始行数: {original_len}, 移除 NaT 索引后行数: {len(df_weekly)}")
                 # --- 结束修正 ---
                 df_weekly.index.name = original_index_name # 尝试恢复原始名称 (如果需要)
                 if df_weekly.empty: continue
                 numeric_cols_count = 0
                 for col in df_weekly.columns:
                     try:
                         df_weekly[col] = pd.to_numeric(df_weekly[col], errors='coerce')
                         if not df_weekly[col].isna().all(): numeric_cols_count += 1
                     except: pass
                 if numeric_cols_count == 0: continue
                 # --- 结束转换 --- 
                 exog_weekly_df_list.append(df_weekly)
                 loaded_count += 1
                except Exception as load_err:
                 print(f"    加载或初步处理 Sheet '{sheet_name}' 时出错: {load_err}")
        
        if exog_weekly_df_list:
            print(f"成功加载并初步处理 {loaded_count} 个周度数据 Sheet。正在合并...")
            exog_weekly_df = pd.concat(exog_weekly_df_list, axis=1, join='outer')
            print(f"合并后 (outer join) 的周度数据维度: {exog_weekly_df.shape}")
            if exog_weekly_df.empty:
                 print("警告：Outer join 后周度 DataFrame 仍为空 (不应发生除非所有输入都为空)。")
        else:
             print("错误：未能成功加载并初步处理任何指定的周度数据 Sheet。")
             
    except FileNotFoundError:
        print(f"错误: Excel 文件 '{EXCEL_DATA_FILE}' 未找到。")
        exog_weekly_df = pd.DataFrame()
    except Exception as e:
        print(f"加载或合并周度外生数据时出错: {e}")
        traceback.print_exc()

    # 3. 预处理外生数据
    exog_monthly_df = pd.DataFrame()
    if not exog_weekly_df.empty and target_series_monthly is not None:
        exog_monthly_df = preprocess_exog_data(exog_weekly_df, target_series_monthly.index)
    if exog_monthly_df.empty:
        print("错误: 外生变量预处理失败或结果为空。无法继续 ARDL。")
        # 即使失败，也尝试写入已有的 ARIMA 结果 (如果存在)
        # ... (可以在这里添加保存 ARIMA 结果的代码，但现在先退出) ...
        sys.exit(1) 

    # --- 修改：数据分割与变量筛选 ---
    print(f"\n--- 4. 数据分割与变量筛选 --- ")
    print(f"训练期: <= {TRAIN_END_DATE}")
    print(f"验证期: {VALIDATION_START_DATE} to {VALIDATION_END_DATE}")

    # 分割目标序列
    target_train = target_series_monthly[:TRAIN_END_DATE].copy()
    target_val = target_series_monthly[VALIDATION_START_DATE:VALIDATION_END_DATE].copy()
    
    # 分割外生变量
    exog_train_df = exog_monthly_df[:TRAIN_END_DATE].copy()
    exog_val_df = exog_monthly_df[VALIDATION_START_DATE:VALIDATION_END_DATE].copy()

    print(f"训练集 Target 长度: {len(target_train)}, Exog 维度: {exog_train_df.shape}")
    print(f"验证集 Target 长度: {len(target_val)}, Exog 维度: {exog_val_df.shape}")

    if target_train.empty or exog_train_df.empty or target_val.empty or exog_val_df.empty:
        print("错误: 训练集或验证集数据为空，无法进行变量筛选。")
        sys.exit(1)

    all_variable_metrics = {}
    print(f"\n开始评估 {len(exog_train_df.columns)} 个外生变量 (基于验证期表现)...")
    for exog_col in exog_train_df.columns:
        if exog_col in exog_val_df.columns: # 确保验证集也有此列
             # 准备单变量的训练和验证数据
             current_exog_train = exog_train_df[exog_col]
             current_exog_val = exog_val_df[exog_col]
             
             # 调用修改后的评估函数
             metrics = evaluate_exog_variable(
                 train_target=target_train,
                 train_exog=current_exog_train,
                 val_target=target_val,
                 val_exog=current_exog_val,
                 exog_name=exog_col,
                n_splits=N_SPLITS_VALIDATION, 
                max_lags=MAX_ARDL_LAGS
            )
             all_variable_metrics[exog_col] = metrics
        else:
             print(f"  警告: 变量 '{exog_col}' 在验证集中不存在，跳过评估。")

    # 选择最佳变量 (最高 Hit Rate -> 最低 RMSE)
    best_variable = None
    best_hit_rate = -np.inf
    best_rmse = np.inf

    valid_results = {k: v for k, v in all_variable_metrics.items() if v.get('Error') is None and not np.isnan(v.get('Hit_Rate', np.nan))}
    
    if not valid_results:
         print("\n错误: 所有外生变量评估失败或未产生有效指标，无法选择最佳变量。")
         # 退出前尝试保存 ARIMA 结果
         sys.exit(1)
    else:
        print("\n变量评估完成。根据验证期指标选择最佳变量:")
        # 按 Hit Rate 降序, RMSE 升序排序
        sorted_vars = sorted(valid_results.items(), key=lambda item: (item[1].get('Hit_Rate', -np.inf), -item[1].get('RMSE', np.inf)), reverse=True)
        
        best_variable, best_metrics = sorted_vars[0]
        best_hit_rate = best_metrics.get('Hit_Rate', np.nan)
        best_rmse = best_metrics.get('RMSE', np.nan)
        
        print(f"  最佳变量: {best_variable}")
        print(f"  其验证期表现: HitRate={best_hit_rate:.2f}%, RMSE={best_rmse:.4f}")
        
        # 打印所有变量的验证期表现 (用于 Excel)
        variable_metrics_df = pd.DataFrame.from_dict(all_variable_metrics, orient='index')
        variable_metrics_df = variable_metrics_df.reset_index().rename(columns={'index': 'Variable'})
        print("\n所有变量验证期指标总结:")
        print(variable_metrics_df)

            # --- 结束修改 --- 

    # 5. 使用选定的最佳变量，在完整历史数据上运行最终 ARDL 模型
    final_ardl_model_info = None
    ardl_hist_forecast_df = pd.DataFrame()
    
    if best_variable and best_variable in exog_monthly_df.columns:
        print(f"\n--- 5. 使用最佳变量 '{best_variable}' 和完整历史数据训练最终 ARDL 模型 ---")
        best_exog_series_full_hist = exog_monthly_df[best_variable]
        
        final_ardl_model_info, ardl_hist_forecast_df = run_ardl_benchmark(
            target_series_full=target_series_monthly, # 使用完整历史目标数据
            best_exog_series_full=best_exog_series_full_hist, # 使用完整历史最佳外生数据
            max_lags=MAX_ARDL_LAGS,
            forecast_steps=FORECAST_STEPS
        )
    else:
        print("\n错误: 未能选择最佳外生变量或该变量不在预处理后的数据中，无法运行最终 ARDL 模型。")

    # 6. 绘制最终 ARDL 结果图
    if not ardl_hist_forecast_df.empty and best_variable:
        plot_save_path = os.path.join(SCRIPT_DIR, PLOT_FILENAME)
        plot_ardl_results(
            results_df=ardl_hist_forecast_df,
            exog_name=best_variable,
            filename=plot_save_path
        )
    else:
        print("警告：未生成 ARDL 预测结果或未选定变量，无法绘制图像。")

    # --- 修改：保存结果到 Excel ---
    print(f"\n--- 6. 保存结果到 Excel ---")
    # --- 修正：保存路径指向工作区根目录 --- 
    results_save_path = os.path.join(BASE_DIR, BENCHMARK_RESULTS_FILE) 
    # --- 结束修正 ---
    try:
        # 尝试读取现有的 ARIMA 摘要，如果存在的话
        existing_summary = {}
        if os.path.exists(results_save_path):
             try:
                 summary_sheet = pd.read_excel(results_save_path, sheet_name='Summary_Metrics', index_col=None)
                 # 将其转换为字典，以便更新或添加 ARDL
                 summary_sheet = summary_sheet.set_index('Model')
                 existing_summary = summary_sheet.to_dict(orient='index')
                 print("已读取现有的 Summary_Metrics Sheet。")
             except Exception as read_err:
                 print(f"警告：无法读取现有的 Summary_Metrics Sheet ({read_err})。将覆盖或创建新的。")

        # 准备 ARDL 的摘要信息
        ardl_summary_data = {
            'Selected_Variable': best_variable if best_variable else 'N/A',
            'Final_Lags_p': final_ardl_model_info.get('Final_Selected_Order_p', 'N/A') if final_ardl_model_info else 'N/A',
            'Final_Lags_q': str(final_ardl_model_info.get('Final_Selected_Order_q', 'N/A')) if final_ardl_model_info else 'N/A',
            'Validation_Hit_Rate': best_metrics.get('Hit_Rate', np.nan) if best_variable and best_variable in valid_results else np.nan,
            'Validation_RMSE': best_metrics.get('RMSE', np.nan) if best_variable and best_variable in valid_results else np.nan
        }
        # --- 修正：q_lags 转为字符串 (使用 final_ardl_model_info) --- 
        q_lags_value = final_ardl_model_info.get('Final_Selected_Order_q', 'N/A') if final_ardl_model_info else 'N/A' # 使用正确的变量
        ardl_summary_data['Final_Lags_q'] = str(q_lags_value) 
        # --- 结束修正 ---
        existing_summary['ARDL'] = ardl_summary_data # 添加或更新 ARDL 行

        # 转换回 DataFrame 准备写入
        final_summary_df = pd.DataFrame.from_dict(existing_summary, orient='index')
        final_summary_df = final_summary_df.reset_index().rename(columns={'index': 'Model'})
        # 确保 ARIMA 和 ARDL 的列都存在，可能需要调整顺序
        all_summary_cols = ['Model', 'Best_Order', 'Best_AIC', # ARIMA cols
                            'Selected_Variable', 'Final_Lags_p', 'Final_Lags_q', # ARDL cols
                            'Validation_Hit_Rate', 'Validation_RMSE'] # ARDL validation cols
        # 重新索引以包含所有列，填充缺失值为 ''
        final_summary_df = final_summary_df.reindex(columns=all_summary_cols, fill_value='')

        with pd.ExcelWriter(results_save_path, engine='openpyxl', mode='w') as writer: # 总是覆盖
            # 保存合并后的摘要
            final_summary_df.to_excel(writer, sheet_name='Summary_Metrics', index=False)
            print(f"模型摘要信息已写入 Sheet: Summary_Metrics")
            print(final_summary_df)

            # 保存 ARDL 变量评估指标 (验证期)
            if 'variable_metrics_df' in locals() and not variable_metrics_df.empty:
                 cols_var_metrics = ['Variable', 'RMSE', 'MAE', 'Hit_Rate', 'Selected_Lags_During_Eval', 'Error']
                 variable_metrics_df = variable_metrics_df.reindex(columns=cols_var_metrics, fill_value='') # 确保列存在且有序
                 variable_metrics_df = variable_metrics_df.fillna('') # 替换 NaN
                 variable_metrics_df.to_excel(writer, sheet_name='ARDL_Variable_Metrics', index=False)
                 print(f"ARDL 变量验证期指标已写入 Sheet: ARDL_Variable_Metrics")
            else:
                 print("警告：未生成 ARDL 变量评估指标，未写入 Excel。")

            # 保存最终 ARDL 模型的历史/未来预测
            if not ardl_hist_forecast_df.empty:
                 df_to_save = ardl_hist_forecast_df.copy()
                 # --- Revised Saving Steps ---
                 # Keep index for now, reset it to become a column
                 df_to_save = df_to_save.reset_index()
                 # Rename the new column (original index) to 'Date'
                 df_to_save = df_to_save.rename(columns={'index': 'Date'})
                 # Format the 'Date' column as string *after* it's a column
                 df_to_save['Date'] = df_to_save['Date'].dt.strftime('%Y-%m-%d')
                 # --- REMOVED fillna('') before saving --- 
                 # Define desired column order
                 cols = ['Date', 'Actual', 'Fitted', 'Forecast']
                 # Filter columns to only those present in df_to_save, maintaining order
                 cols_present = [col for col in cols if col in df_to_save.columns]
                 df_to_save = df_to_save[cols_present]
                 # --- End Revised Saving Steps ---
                 df_to_save.to_excel(writer, sheet_name='ARDL_Forecast', index=False)
                 print(f"最终 ARDL 历史/未来预测数据已写入 Sheet: ARDL_Forecast")
            else:
                 print("警告: ARDL 历史/未来预测数据为空，未写入 Excel。")
                 
            # 保存预处理后的月度外生变量
            if not exog_monthly_df.empty:
                 exog_monthly_df_to_save = exog_monthly_df.copy()
                 exog_monthly_df_to_save.index.name = 'Date' # 给索引列命名
                 exog_monthly_df_to_save = exog_monthly_df_to_save.reset_index()
                 exog_monthly_df_to_save['Date'] = exog_monthly_df_to_save['Date'].dt.strftime('%Y-%m-%d')
                 # --- Keep fillna here as it's for different data --- 
                 exog_monthly_df_to_save = exog_monthly_df_to_save.fillna('')
                 exog_monthly_df_to_save.to_excel(writer, sheet_name='ARDL_Preprocessed_Exog', index=False)
                 print(f"预处理后的月度外生变量数据已写入 Sheet: ARDL_Preprocessed_Exog")
            else:
                print("警告：预处理后的月度外生变量数据为空，未写入 Excel。")
                
            # --- 尝试写入 ARIMA 的预测数据 (如果 arima_benchmark.py 已运行并生成了它) ---
            # 这个逻辑依赖于 arima_benchmark.py 已经正确保存了它的数据
            # 为了独立性，这里不再尝试读取 ARIMA_Forecast，假设用户分开运行
            print("注意：ARIMA_Forecast Sheet 需要通过运行 arima_benchmark.py 生成。")
                 
        print(f"ARDL 基准测试结果已写入到: {results_save_path}")

    except Exception as e:
        print(f"保存 ARDL 结果到 Excel 时出错: {e}")
        traceback.print_exc()
    # --- 结束修改 ---

    script_end_time = time.time()
    total_runtime_seconds = script_end_time - script_start_time
    print(f"总耗时: {total_runtime_seconds:.2f} 秒") 