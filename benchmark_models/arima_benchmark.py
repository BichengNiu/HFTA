# -*- coding: utf-8 -*-
"""
基准模型 (ARIMA) 评估脚本。
目的：为 DFM 提供比较基准。
修改：
- 移除 MAPE 指标计算。
- 直接加载和处理月度数据。
- 添加诊断打印语句。
- 使用 resample('ME').last() 处理不规则月度数据。
- 修改 ARIMA 预测步骤以使用整数索引。
- 分开预测样本内和样本外以生成完整序列用于绘图。
- 动态构建数据文件路径。
- 添加 ARIMA 结果绘图功能。
+ 使用所有可用历史数据训练最终模型。
+ 预测未来 3 个月。
+ 添加训练集/验证集划分和对应绘图功能。
+ 生成两张图：训练/验证对比图 和 全样本拟合/预测图。
"""
import pandas as pd
import numpy as np
import sys
import os
import time
import warnings
import itertools
from sklearn.metrics import mean_squared_error, mean_absolute_error
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller
import traceback
import matplotlib
import matplotlib.pyplot as plt

# --- 移除 data_preparation 导入 ---
# try:
#     from data_preparation import prepare_data
# except ImportError as e:
#     print(f"错误：导入自定义模块 data_preparation 失败: {e}")
#     print("请确保 benchmark_models.py 可以找到 data_preparation.py")
#     sys.exit(1)

# --- 配置 ---
warnings.filterwarnings('ignore')
matplotlib.use('Agg') # 使用 Agg 后端，避免 GUI 问题
plt.rcParams['font.sans-serif'] = ['SimHei'] # 设置中文字体
plt.rcParams['axes.unicode_minus'] = False # 解决负号显示问题

# --- 构建数据文件路径 ---
# 获取 benchmark_models.py 文件所在的目录
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
# 获取 HFTA 根目录 (SCRIPT_DIR 的上一级目录)
BASE_DIR = os.path.dirname(SCRIPT_DIR)
# 构建数据文件的绝对路径
EXCEL_DATA_FILE = os.path.join(BASE_DIR, 'data', '经济数据库.xlsx')
# --- 结束构建路径 ---

# --- 常量 (部分与 tune_dfm.py 保持一致) ---
# EXCEL_DATA_FILE = '../data/经济数据库.xlsx' # <-- 移除旧路径
TARGET_VARIABLE = '规模以上工业增加值:当月同比'
TARGET_SHEET_NAME = '工业增加值同比增速-月度' # 直接从此 Sheet 加载月度数据
# TARGET_FREQ = 'W-FRI' # 不再需要，使用月度数据
# TRAIN_END_DATE = '2024-06-30' # <-- 移除
# VALIDATION_START_DATE = '2024-07-01' # <-- 移除
# VALIDATION_END_DATE = '2024-12-31' # <-- 移除
BENCHMARK_RESULTS_FILE = "arima_benchmark_result.xlsx"
ADF_P_THRESHOLD = 0.05
MAX_ARIMA_ORDER = 2 # 限制 p 和 q 的最大阶数，避免过长时间搜索
# --- 新增：绘图相关常量 ---
PLOT_FILENAME = "arima_vs_actual.png" # 图像保存文件名
FORECAST_STEPS = 3 # <-- 修改：预测 3 个月
# --- 新增：训练/验证划分 ---
VALIDATION_START_DATE = '2024-07-01' # <-- 修改：根据用户要求，验证集从24年7月开始
# --- 新增：绘图相关常量 ---
PLOT_FULL_SAMPLE_FILENAME = "arima_full_sample_fit_forecast.png" # 全样本图文件名
PLOT_TRAIN_VALID_FILENAME = "arima_train_validation.png" # 训练验证图文件名

# --- 指标计算函数 (移除 MAPE) ---
def calculate_metrics(y_true, y_pred):
    """计算 RMSE, MAE, Hit Rate."""
    metrics = {
        'RMSE': np.nan,
        'MAE': np.nan,
        'Hit_Rate': np.nan
    }
    if len(y_true) < 2 or len(y_pred) < 2 or len(y_true) != len(y_pred):
        print("[Metrics] 警告: 输入序列长度不足或不匹配，无法计算指标。")
        return metrics
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    valid_mask = ~np.isnan(y_true) & ~np.isnan(y_pred)
    y_true_valid = y_true[valid_mask]
    y_pred_valid = y_pred[valid_mask]
    if len(y_true_valid) < 2:
        print(f"[Metrics] 警告: 移除 NaN 后有效数据点不足 ({len(y_true_valid)} 点)，无法计算指标。")
        return metrics
    try:
        metrics['RMSE'] = np.sqrt(mean_squared_error(y_true_valid, y_pred_valid))
    except Exception as e:
        print(f"[Metrics] 计算 RMSE 时出错: {e}")
    try:
        metrics['MAE'] = mean_absolute_error(y_true_valid, y_pred_valid)
    except Exception as e:
        print(f"[Metrics] 计算 MAE 时出错: {e}")
    try:
        true_diff = np.diff(y_true_valid)
        pred_diff = np.diff(y_pred_valid)
        non_zero_target_change_mask = true_diff != 0
        if np.any(non_zero_target_change_mask):
            correct_direction = np.sign(pred_diff[non_zero_target_change_mask]) == np.sign(true_diff[non_zero_target_change_mask])
            metrics['Hit_Rate'] = np.mean(correct_direction) * 100
        else:
            print("[Metrics] 警告: 真实值变化全为零，无法计算 Hit Rate。")
    except Exception as e:
        print(f"[Metrics] 计算 Hit Rate 时出错: {e}")
    return metrics

# --- ARIMA 辅助函数 (添加诊断) ---
def get_arima_integration_order(series: pd.Series, p_threshold: float) -> int:
    """使用 ADF 检验确定差分阶数 d."""
    d = 0
    print(f"\n[ADF Check] 输入序列 (前5行):\n{series.head()}")
    print(f"[ADF Check] 输入序列信息:")
    series.info()
    print(f"[ADF Check] 输入序列 NaN 数量: {series.isna().sum()}")
    temp_series = series.dropna()
    print(f"[ADF Check] 调用 dropna() 后序列长度: {len(temp_series)}")
    if temp_series.empty:
        print("[ADF Check] 错误: dropna() 后序列为空！")
    else:
        print(f"[ADF Check] dropna() 后序列 (前5行):\n{temp_series.head()}")
    if temp_series.empty:
        print(f"[ADF Check] 由于 dropna() 后序列为空，无法执行 ADF 检验。将假定 d=0.")
        return 0
    try:
        adf_result = adfuller(temp_series)
        p_value = adf_result[1]
        if p_value >= p_threshold:
            d = 1
            temp_series_diff = temp_series.diff().dropna()
            if not temp_series_diff.empty:
                adf_result_diff = adfuller(temp_series_diff)
                p_value_diff = adf_result_diff[1]
                if p_value_diff >= p_threshold:
                    d = 2
                    print("警告: 一阶差分后仍非平稳 (ADF p-value >= {}), 尝试 d=2.".format(p_threshold))
            else:
                print("警告: 一阶差分后序列为空，无法进一步检验，使用 d=1.")
    except Exception as e:
        print(f"[ADF Check] ADF 检验出错: {e}. 将假定 d=0.")
        return 0
    print(f"根据 ADF 检验 (p<{p_threshold}), 确定的差分阶数 d = {d}")
    return d

def find_best_arima_order(train_series: pd.Series, d: int, max_order: int) -> dict:
    """在训练集上通过 AIC 选择最佳的 p, q 阶数，返回包含阶数和 AIC 的字典。"""
    best_aic = np.inf
    best_order = (0, d, 0)
    p_range = range(max_order + 1)
    q_range = range(max_order + 1)
    if max_order == 0 and d == 0:
         print("警告: max_order 为 0 且 d 为 0，无法搜索 ARIMA 模型。返回 (0,0,0) 和 Inf AIC。")
         return {'order': (0, 0, 0), 'aic': np.inf}
    candidates = list(itertools.product(p_range, q_range))
    print(f"[ARIMA Order Search] 搜索 ARIMA(p, {d}, q) 最佳阶数 (p, q <= {max_order})...共 {len(candidates)} 种组合")
    for p, q in candidates:
        if d == 0 and p == 0 and q == 0: continue
        order = (p, d, q)
        try:
            model = ARIMA(train_series, order=order)
            model_fit = model.fit(method_kwargs={"warn_convergence": False})
            current_aic = model_fit.aic
            if not np.isinf(current_aic) and current_aic < best_aic:
                best_aic = current_aic
                best_order = order
        except np.linalg.LinAlgError:
            continue
        except ValueError as ve:
            continue
        except Exception as e:
            continue
    print(f"[ARIMA Order Search] 在训练集上找到的最佳 ARIMA 阶数: {best_order} (AIC: {best_aic:.2f})")
    return {'order': best_order, 'aic': best_aic}

# --- 修改: ARIMA 基准测试主函数 (用于全样本拟合和未来预测) ---
def run_arima_benchmark(target_series: pd.Series,
                        adf_p_threshold: float, max_order: int, forecast_steps: int) -> tuple[dict, pd.DataFrame]:
    """执行 ARIMA 基准测试，使用所有历史数据选择阶数和训练，预测未来 forecast_steps 步。"""
    print("\n--- 开始 ARIMA 基准测试 (全样本拟合与预测) --- ")

    results = {'Best_Order': None, 'Best_AIC': np.inf}
    results_df_hist_forecast = pd.DataFrame()

    print(f"[ARIMA Benchmark Full] 传入的 target_series (前5行):\n{target_series.head()}")
    print(f"[ARIMA Benchmark Full] 传入的 target_series 信息:")
    target_series.info()
    print(f"[ARIMA Benchmark Full] 传入的 target_series NaN 数量: {target_series.isna().sum()}")

    # 1. 数据准备 (使用所有可用数据)
    target_series_processed = target_series.dropna() # 使用全部数据
    print(f"[ARIMA Benchmark Full] 对目标序列调用 dropna() 后长度: {len(target_series_processed)}")
    if target_series_processed.empty:
        print("[ARIMA Benchmark Full] 错误: 用于 ARIMA 的目标序列为空。")
        return results, results_df_hist_forecast

    if not isinstance(target_series_processed.index, pd.DatetimeIndex):
        print("[ARIMA Benchmark Full] 错误: 月度目标序列的索引不是 DatetimeIndex。")
        try:
            target_series_processed.index = pd.to_datetime(target_series_processed.index)
            print("[ARIMA Benchmark Full] 索引已成功转换为 DatetimeIndex。")
        except Exception as e:
            print(f"[ARIMA Benchmark Full] 无法将索引转换为 DatetimeIndex: {e}")
            return results, results_df_hist_forecast

    original_index_full = target_series_processed.index
    try:
        # 重采样整个序列
        target_series_resampled = target_series_processed.resample('ME').last()
        print(f"[ARIMA Benchmark Full] 数据已重采样至月末频率 ('ME')。")
        print(f"[ARIMA Benchmark Full] 重采样后数据 NaN 数量: {target_series_resampled.isna().sum()} (重采样可能引入 NaN)")
    except Exception as e:
        print(f"[ARIMA Benchmark Full] 警告: 重采样出错 ({e})。将使用原始未重采样数据。")
        target_series_resampled = target_series_processed

    # --- 使用所有重采样后的非 NaN 数据进行建模 ---
    data_for_modeling = target_series_resampled.dropna()
    print(f"[ARIMA Benchmark Full] 用于模型选择和最终拟合的数据点数: {len(data_for_modeling)}")
    if data_for_modeling.empty:
        print("[ARIMA Benchmark Full] 错误：最终用于建模的数据为空。")
        return results, results_df_hist_forecast

    # 2. 确定 ARIMA 阶数 (使用 data_for_modeling)
    try:
        integration_order_d = get_arima_integration_order(data_for_modeling, adf_p_threshold)
        best_order_aic = find_best_arima_order(data_for_modeling, integration_order_d, max_order)
        final_arima_order = best_order_aic['order']
        results['Best_Order'] = final_arima_order
        results['Best_AIC'] = best_order_aic['aic']
    except Exception as e:
         print(f"[ARIMA Benchmark Full] 确定 ARIMA 阶数时出错: {e}，无法继续。")
         return results, results_df_hist_forecast

    # 3. 使用所有 data_for_modeling 训练最终模型
    print(f"\n[ARIMA Benchmark Full] 使用阶数 {final_arima_order} 拟合最终模型 (基于所有可用数据)...")
    final_model_fit = None
    try:
        final_model = ARIMA(data_for_modeling, order=final_arima_order)
        final_model_fit = final_model.fit()
        print("[ARIMA Benchmark Full] 最终 ARIMA 模型拟合完成.")
    except Exception as e:
        print(f"[ARIMA Benchmark Full] 最终 ARIMA 模型拟合失败: {e}")
        return results, results_df_hist_forecast

    # 4. 生成历史拟合值和未来预测值
    print(f"[ARIMA Benchmark Full] 生成历史拟合值和未来 {forecast_steps} 步预测...")
    historical_fitted = None
    future_forecast = None

    try:
        # 获取历史拟合值 (覆盖所有用于建模的数据)
        if final_model_fit is not None and not data_for_modeling.empty:
            start_hist = data_for_modeling.index[0]
            end_hist = data_for_modeling.index[-1]
            historical_fitted = final_model_fit.predict(start=start_hist, end=end_hist)
            print("[ARIMA Benchmark Full] 历史拟合值生成完成。")

        # 获取未来预测值 (从最后历史点之后开始)
        if final_model_fit is not None:
            future_forecast = final_model_fit.forecast(steps=forecast_steps)
            print(f"[ARIMA Benchmark Full] 未来 {forecast_steps} 步预测完成。")
            print("[ARIMA Benchmark Full] 未来预测值:")
            print(future_forecast)

        # 合并结果到 DataFrame (基础是 data_for_modeling)
        results_df_hist_forecast = pd.DataFrame({'Actual': data_for_modeling, 'Fitted': historical_fitted})
        if future_forecast is not None:
            future_df = pd.DataFrame({'Forecast': future_forecast})
            # --- 修改：确保未来预测的索引正确 ---
            last_hist_date = data_for_modeling.index[-1]
            future_index = pd.date_range(start=last_hist_date + pd.DateOffset(months=1),
                                         periods=forecast_steps,
                                         freq=data_for_modeling.index.freqstr if data_for_modeling.index.freqstr else 'ME')
            future_df.index = future_index
            # --- 结束修改 ---
            results_df_hist_forecast = pd.concat([results_df_hist_forecast, future_df])
        else:
             results_df_hist_forecast['Forecast'] = np.nan
        print("[ARIMA Benchmark Full] 历史和未来数据合并完成。")

    except Exception as e:
        print(f"[ARIMA Benchmark Full] ARIMA 生成历史拟合或未来预测失败: {e}")
        traceback.print_exc()
        if results_df_hist_forecast.empty:
             results_df_hist_forecast = pd.DataFrame({'Actual': data_for_modeling, 'Fitted': np.nan, 'Forecast': np.nan})

    print("--- ARIMA 基准测试 (全样本拟合与预测) 完成 --- ")
    return results, results_df_hist_forecast # 返回选择结果和预测 DataFrame

# --- 新增：执行训练/验证集划分和预测的函数 ---
def run_arima_train_validation(target_series: pd.Series,
                               validation_start_date: str,
                               adf_p_threshold: float,
                               max_order: int) -> pd.DataFrame:
    """在训练集上选择和训练ARIMA，在验证集上预测，返回合并结果DataFrame。"""
    print("\n--- 开始 ARIMA 训练/验证评估 ---")
    results_df_train_val = pd.DataFrame()
    validation_start = pd.to_datetime(validation_start_date)

    # 1. 数据准备和划分
    print(f"[ARIMA Train/Val] 传入的 target_series (前5行):\n{target_series.head()}")
    target_series_processed = target_series.dropna()
    print(f"[ARIMA Train/Val] 调用 dropna() 后长度: {len(target_series_processed)}")

    if not isinstance(target_series_processed.index, pd.DatetimeIndex):
        print("[ARIMA Train/Val] 错误: 索引不是 DatetimeIndex。")
        try:
            target_series_processed.index = pd.to_datetime(target_series_processed.index)
            print("[ARIMA Train/Val] 索引已转换为 DatetimeIndex。")
        except Exception as e:
            print(f"[ARIMA Train/Val] 转换索引失败: {e}")
            return results_df_train_val

    try:
        target_series_resampled = target_series_processed.resample('ME').last()
        print(f"[ARIMA Train/Val] 数据已重采样至 'ME'。")
        print(f"[ARIMA Train/Val] 重采样后 NaN 数量: {target_series_resampled.isna().sum()}")
    except Exception as e:
        print(f"[ARIMA Train/Val] 警告: 重采样出错 ({e})。使用原始数据。")
        target_series_resampled = target_series_processed

    data_for_modeling = target_series_resampled.dropna()
    print(f"[ARIMA Train/Val] 用于建模和划分的数据点数: {len(data_for_modeling)}")
    if data_for_modeling.empty:
        print("[ARIMA Train/Val] 错误: 无可用数据进行划分。")
        return results_df_train_val

    # 划分训练集和验证集
    train_data = data_for_modeling[data_for_modeling.index < validation_start]
    validation_data = data_for_modeling[data_for_modeling.index >= validation_start]

    print(f"[ARIMA Train/Val] 训练集范围: {train_data.index.min()} to {train_data.index.max()} ({len(train_data)}点)")
    print(f"[ARIMA Train/Val] 验证集范围: {validation_data.index.min()} to {validation_data.index.max()} ({len(validation_data)}点)")

    if train_data.empty:
        print("[ARIMA Train/Val] 错误: 训练集为空，无法继续。")
        return results_df_train_val
    if validation_data.empty:
        print("[ARIMA Train/Val] 警告: 验证集为空，仅执行训练集拟合。")

    # 2. 在训练集上确定 ARIMA 阶数
    best_order = None
    try:
        integration_order_d = get_arima_integration_order(train_data, adf_p_threshold)
        best_order_aic = find_best_arima_order(train_data, integration_order_d, max_order)
        best_order = best_order_aic['order']
        print(f"[ARIMA Train/Val] 在训练集上找到的最佳阶数: {best_order} (AIC: {best_order_aic['aic']:.2f})")
    except Exception as e:
         print(f"[ARIMA Train/Val] 确定 ARIMA 阶数时出错: {e}，无法继续。")
         return results_df_train_val

    # 3. 使用训练集训练模型
    print(f"\n[ARIMA Train/Val] 使用阶数 {best_order} 在训练集上训练模型...")
    model_fit = None
    train_fitted = None
    try:
        model = ARIMA(train_data, order=best_order)
        model_fit = model.fit()
        print("[ARIMA Train/Val] 模型训练完成.")
        # 获取训练集拟合值
        train_fitted = model_fit.predict(start=train_data.index[0], end=train_data.index[-1])
    except Exception as e:
        print(f"[ARIMA Train/Val] ARIMA 模型训练失败: {e}")
        # 即使训练失败，也尝试返回包含实际值的数据框
        results_df_train_val = pd.DataFrame({'Actual': data_for_modeling})
        results_df_train_val['Fitted'] = np.nan
        results_df_train_val['Predicted'] = np.nan
        return results_df_train_val

    # 4. 在验证集上进行预测
    validation_predicted = None
    if model_fit is not None and not validation_data.empty:
        print(f"[ARIMA Train/Val] 在验证集上进行预测...")
        print(f"[DEBUG] validation_data (len={len(validation_data)}):\n{validation_data}")
        try:
            # --- 修改：使用步数进行样本外预测 ---
            n_validation = len(validation_data)
            # forecast 方法用于样本外预测
            validation_predicted_values = model_fit.forecast(steps=n_validation)
            print(f"[DEBUG] Raw forecast output (len={len(validation_predicted_values)}):\n{validation_predicted_values}")
            # 创建一个带有正确验证集索引的 Series
            validation_predicted = pd.Series(validation_predicted_values.values, index=validation_data.index)
            print(f"[DEBUG] validation_predicted Series (len={len(validation_predicted)}):\n{validation_predicted}")
            # --- 结束修改 ---
            print("[ARIMA Train/Val] 验证集预测完成。")
        except Exception as e:
            print(f"[ARIMA Train/Val] 验证集预测失败: {e}")
            traceback.print_exc()

    # 5. 合并结果
    print("[ARIMA Train/Val] 合并训练集和验证集结果...")
    results_df_train_val = pd.DataFrame({'Actual': data_for_modeling}) # 以完整数据为基础
    # 添加训练拟合值
    if train_fitted is not None:
        results_df_train_val['Fitted'] = train_fitted # 会自动对齐索引
    else:
        results_df_train_val['Fitted'] = np.nan
    # 添加验证预测值
    if validation_predicted is not None:
        results_df_train_val['Predicted'] = validation_predicted # 会自动对齐索引
    else:
        results_df_train_val['Predicted'] = np.nan

    # --- DEBUG: 打印合并后的验证期数据 ---
    print("[DEBUG] results_df_train_val in validation period:")
    print(results_df_train_val.loc[validation_start:])
    # --- END DEBUG ---

    # 验证指标计算 (可选，这里只做打印)
    if validation_predicted is not None and not validation_data.empty:
         val_metrics = calculate_metrics(validation_data, validation_predicted)
         print("[ARIMA Train/Val] 验证集指标:")
         print(val_metrics)


    print("--- ARIMA 训练/验证评估 完成 ---")
    return results_df_train_val


# --- 修改：绘图函数 (用于全样本拟合和未来预测图) ---
def plot_arima_results(results_df: pd.DataFrame, filename: str, title_suffix: str):
    """绘制 ARIMA 实际值 vs 历史拟合/预测值图。"""
    print(f"\n--- 生成 ARIMA 全样本拟合/预测图 --- ")
    # 检查基本列是否存在
    if results_df.empty or 'Actual' not in results_df.columns:
        print("[Plot Full] 错误：传入绘图的数据缺少 'Actual' 列，无法生成图像。")
        return

    try:
        plt.figure(figsize=(14, 7))

        # 绘制实际值 (仅绘制有实际值的时期)
        actual_data = results_df['Actual'].dropna()
        plt.plot(actual_data.index, actual_data, marker='.', linestyle='None', color='red', label='实际值 (月度)')

        # 绘制历史拟合值 (如果存在)
        if 'Fitted' in results_df.columns and results_df['Fitted'].notna().any():
             fitted_data = results_df['Fitted'].dropna()
             plt.plot(fitted_data.index, fitted_data, linestyle='--', color='green', label='历史拟合值 (全样本)')

        # 绘制未来预测值 (如果存在)
        if 'Forecast' in results_df.columns and results_df['Forecast'].notna().any():
             forecast_data = results_df['Forecast'].dropna()
             plt.plot(forecast_data.index, forecast_data, linestyle=':', color='blue', label=f'未来 {len(forecast_data)} 个月预测值')

        plt.title(f'ARIMA 模型: 实际 vs 全样本拟合/预测 ({title_suffix})')
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
        print(f"ARIMA 全样本拟合/预测图已保存到: {filename}")

    except Exception as e:
        print(f"[Plot Full] 生成或保存 ARIMA 全样本图时出错: {e}")
        traceback.print_exc()
        try: plt.close()
        except: pass

# --- 新增：绘图函数 (用于训练/验证对比图) ---
def plot_train_validation_results(results_df: pd.DataFrame, validation_start_date: str, filename: str, title_suffix: str):
    """绘制 ARIMA 训练集拟合 vs 验证集预测图。"""
    print(f"\n--- 生成 ARIMA 训练/验证对比图 --- ")
    if results_df.empty or 'Actual' not in results_df.columns:
        print("[Plot Train/Val] 错误：传入绘图的数据缺少 'Actual' 列，无法生成图像。")
        return

    validation_start = pd.to_datetime(validation_start_date)

    try:
        plt.figure(figsize=(14, 7))

        # 绘制实际值
        plt.plot(results_df.index, results_df['Actual'], marker='.', linestyle='None', color='red', label='实际值 (月度)')

        # 绘制训练集拟合值
        if 'Fitted' in results_df.columns and results_df['Fitted'].notna().any():
            fitted_data = results_df['Fitted'].dropna()
            plt.plot(fitted_data.index, fitted_data, linestyle='--', color='green', label='训练集拟合值')

        # 绘制验证集预测值
        if 'Predicted' in results_df.columns and results_df['Predicted'].notna().any():
            predicted_data = results_df['Predicted'].dropna()
            plt.plot(predicted_data.index, predicted_data, linestyle=':', color='blue', label='验证集预测值')

        # 添加垂直线标记验证集开始
        plt.axvline(validation_start, color='gray', linestyle='--', linewidth=1, label=f'验证集开始 ({validation_start_date})')

        plt.title(f'ARIMA 模型: 训练集拟合 vs 验证集预测 ({title_suffix})')
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
        print(f"ARIMA 训练/验证对比图已保存到: {filename}")

    except Exception as e:
        print(f"[Plot Train/Val] 生成或保存 ARIMA 训练/验证图时出错: {e}")
        traceback.print_exc()
        try: plt.close()
        except: pass


# --- 主执行逻辑 (修改以生成两张图和 Excel 输出) ---
if __name__ == "__main__":
    print(f"开始基准模型评估 (ARIMA - 月度数据)...")
    script_start_time = time.time()
    print(f"\n--- 加载月度数据 --- ")
    print(f"Excel 文件: {EXCEL_DATA_FILE}")
    print(f"目标变量 Sheet: {TARGET_SHEET_NAME}")
    print(f"目标变量名称: {TARGET_VARIABLE}")
    target_series_monthly = None
    # --- 数据加载逻辑 (保持不变) ---
    try:
        monthly_data_df = pd.read_excel(
            EXCEL_DATA_FILE,
            sheet_name=TARGET_SHEET_NAME,
            index_col=0,
            parse_dates=True
        )
        print(f"成功从 Sheet '{TARGET_SHEET_NAME}' 加载数据。")
        print("\n[Data Loading] 加载的 DataFrame 信息:")
        monthly_data_df.info()
        print(f"\n[Data Loading] 加载的 DataFrame (前5行):\n{monthly_data_df.head()}")
        print(f"\n[Data Loading] 加载的 DataFrame (后5行):\n{monthly_data_df.tail()}")
        if not isinstance(monthly_data_df.index, pd.DatetimeIndex):
             print("[Data Loading] 警告: 加载后索引不是 DatetimeIndex，尝试转换...")
             try:
                  monthly_data_df.index = pd.to_datetime(monthly_data_df.index, errors='coerce')
                  if monthly_data_df.index.isna().any():
                       print("[Data Loading] 错误: 转换日期时部分索引变为 NaT，请检查 Excel 日期格式。")
                       sys.exit(1)
                  print("[Data Loading] 索引已成功转换为 DatetimeIndex。")
             except Exception as date_e:
                  print(f"[Data Loading] 错误: 尝试转换索引为日期时失败: {date_e}")
                  sys.exit(1)
        else:
             print("[Data Loading] 加载后索引是 DatetimeIndex。")
        if TARGET_VARIABLE not in monthly_data_df.columns:
             print(f"[Data Loading] 错误: 在 Sheet '{TARGET_SHEET_NAME}' 中找不到目标变量列 '{TARGET_VARIABLE}'。")
             print(f"[Data Loading] 可用列: {monthly_data_df.columns.tolist()}")
             if len(monthly_data_df.columns) == 1:
                  original_target = TARGET_VARIABLE
                  TARGET_VARIABLE = monthly_data_df.columns[0]
                  print(f"[Data Loading] 警告: 目标列名 '{original_target}' 不匹配，但只有一列数据，将使用列 '{TARGET_VARIABLE}' 作为目标。")
             else:
                  sys.exit(1)
        target_series_monthly = monthly_data_df[TARGET_VARIABLE]
        print(f"\n[Data Loading] 提取的目标序列 '{TARGET_VARIABLE}' (前5行):\n{target_series_monthly.head()}")
        print(f"[Data Loading] 目标序列数据类型: {target_series_monthly.dtype}")
        original_nan_count = target_series_monthly.isna().sum()
        target_series_monthly = pd.to_numeric(target_series_monthly, errors='coerce')
        new_nan_count = target_series_monthly.isna().sum()
        print(f"[Data Loading] 目标序列转换为数值类型 (errors='coerce')。")
        print(f"[Data Loading] 原始 NaN 数量: {original_nan_count}")
        print(f"[Data Loading] 转换后 NaN 数量: {new_nan_count}")
        if new_nan_count > original_nan_count:
             print(f"[Data Loading] 警告: 转换过程中增加了 {new_nan_count - original_nan_count} 个 NaN，表明原始数据中存在非数值内容。")
        print(f"已提取月度目标序列。数据范围: {target_series_monthly.index.min().strftime('%Y-%m-%d')} to {target_series_monthly.index.max().strftime('%Y-%m-%d')}")
    except FileNotFoundError:
        print(f"错误: Excel 文件 '{EXCEL_DATA_FILE}' 未找到。")
        sys.exit(1)
    except ValueError as ve:
         print(f"错误: 加载或处理月度数据时出错: {ve}")
         sys.exit(1)
    except Exception as e:
        print(f"加载月度数据时发生未知错误: {e}")
        traceback.print_exc()
        sys.exit(1)

    # 初始化结果变量
    arima_full_results = None # 全样本模型选择结果
    arima_hist_forecast_df = pd.DataFrame() # 全样本拟合+预测数据
    arima_train_val_df = pd.DataFrame() # 训练/验证数据

    if target_series_monthly is not None and not target_series_monthly.dropna().empty:
        # --- 新增：计算验证集指标 --- 
        validation_metrics = {'Validation_RMSE': np.nan, 'Validation_MAE': np.nan, 'Validation_Hit_Rate': np.nan}

        # --- 1. 执行训练/验证评估并生成对应图 ---
        arima_train_val_df = run_arima_train_validation(
            target_series=target_series_monthly.copy(), # 传入副本以防修改
            validation_start_date=VALIDATION_START_DATE,
            adf_p_threshold=ADF_P_THRESHOLD,
            max_order=MAX_ARIMA_ORDER
        )

        if not arima_train_val_df.empty:
            plot_train_val_save_path = os.path.join(SCRIPT_DIR, PLOT_TRAIN_VALID_FILENAME)
            plot_train_validation_results(
                results_df=arima_train_val_df,
                validation_start_date=VALIDATION_START_DATE,
                filename=plot_train_val_save_path,
                title_suffix=TARGET_VARIABLE
            )
        else:
            print("警告：未生成 ARIMA 训练/验证结果，无法绘制对应图像。")

        # --- 在生成训练/验证结果后计算指标 ---
        if not arima_train_val_df.empty and 'Actual' in arima_train_val_df.columns and 'Predicted' in arima_train_val_df.columns:
            validation_actual = arima_train_val_df['Actual'][arima_train_val_df['Predicted'].notna()]
            validation_predicted = arima_train_val_df['Predicted'].dropna()
            if not validation_actual.empty and not validation_predicted.empty and len(validation_actual) == len(validation_predicted):
                # 确保索引对齐
                validation_predicted = validation_predicted.reindex(validation_actual.index)
                val_metrics_calculated = calculate_metrics(validation_actual, validation_predicted)
                validation_metrics['Validation_RMSE'] = val_metrics_calculated.get('RMSE', np.nan)
                validation_metrics['Validation_MAE'] = val_metrics_calculated.get('MAE', np.nan)
                validation_metrics['Validation_Hit_Rate'] = val_metrics_calculated.get('Hit_Rate', np.nan)
                print("\n[Main] 验证集指标计算完成:")
                print(validation_metrics)
            else:
                print("[Main] 警告: 由于验证集实际值或预测值为空或长度不匹配，无法计算验证集指标。")

        # --- 2. 执行全样本拟合与未来预测并生成对应图 ---
        arima_full_results, arima_hist_forecast_df = run_arima_benchmark(
            target_series=target_series_monthly.copy(), # 传入副本
            adf_p_threshold=ADF_P_THRESHOLD,
            max_order=MAX_ARIMA_ORDER,
            forecast_steps=FORECAST_STEPS
        )

        if not arima_hist_forecast_df.empty:
            plot_full_save_path = os.path.join(SCRIPT_DIR, PLOT_FULL_SAMPLE_FILENAME)
            plot_arima_results(
                results_df=arima_hist_forecast_df,
                filename=plot_full_save_path,
                title_suffix=TARGET_VARIABLE
            )
        else:
            print("警告：未生成 ARIMA 全样本拟合/预测结果，无法绘制对应图像。")

    else:
        print("错误：无法运行 ARIMA，因为月度目标序列未能成功加载或为空。")
        arima_full_results = {'Best_Order': None, 'Best_AIC': np.inf}

    # --- 修改: 将结果保存到 Excel (仅保存全样本选择结果和预测) ---
    results_summary_dict = {}
    if arima_full_results:
        # 只保存全样本选择过程的结果 (阶数和 AIC)
        # --- 修改：合并全样本选择结果和验证指标 --- 
        summary_entry = {
            'Best_Order': str(arima_full_results.get('Best_Order', 'N/A')),
            'Best_AIC': arima_full_results.get('Best_AIC', np.nan),
            **validation_metrics # 使用解包合并验证指标
        }
        results_summary_dict['ARIMA'] = summary_entry # 模型名统一为 ARIMA
    else:
        # --- 修改：即使全样本失败也包含验证指标 --- 
        results_summary_dict['ARIMA'] = {
            'Best_Order': 'Failed',
            'Best_AIC': np.nan,
            **validation_metrics
        }

    results_save_path = os.path.join(SCRIPT_DIR, BENCHMARK_RESULTS_FILE)
    try:
        mode = 'a' if os.path.exists(results_save_path) else 'w'
        # --- 修改：仅在追加模式下使用 if_sheet_exists --- 
        excel_writer_kwargs = {'engine': 'openpyxl', 'mode': mode}
        if mode == 'a':
            excel_writer_kwargs['if_sheet_exists'] = 'replace'
        # --- 结束修改 --- 

        with pd.ExcelWriter(results_save_path, **excel_writer_kwargs) as writer:
            # 保存指标摘要 (基于全样本模型选择)
            summary_df = pd.DataFrame.from_dict(results_summary_dict, orient='index')
            summary_df = summary_df.reset_index().rename(columns={'index': 'Model'})
            # --- 修改：确保包含所有需要的列 --- 
            summary_cols = ['Model', 'Best_Order', 'Best_AIC', 'Validation_RMSE', 'Validation_MAE', 'Validation_Hit_Rate']
            # 使用 reindex 保证列存在且顺序正确，缺失填充 NaN
            summary_df = summary_df.reindex(columns=summary_cols, fill_value=np.nan)
            # --- 结束修改 ---
            summary_df.to_excel(writer, sheet_name='Summary_Metrics', index=False)
            print(f"\nARIMA 全样本模型选择结果已写入 Sheet: Summary_Metrics")
            print(summary_df)

            # 保存全样本历史和未来预测
            if not arima_hist_forecast_df.empty:
                 df_to_save = arima_hist_forecast_df.copy()
                 # --- 修改：确保 Date 列处理正确 ---
                 # 检查索引是否为 DatetimeIndex，如果是则格式化，否则转为字符串
                 if isinstance(df_to_save.index, pd.DatetimeIndex):
                     df_to_save['Date'] = df_to_save.index.strftime('%Y-%m-%d')
                 else:
                     df_to_save['Date'] = df_to_save.index.astype(str)
                 # --- 结束修改 ---
                 df_to_save = df_to_save.reset_index(drop=True) # 移除索引，使用 Date 列
                 df_to_save = df_to_save.fillna('')
                 cols = ['Date'] + [col for col in df_to_save.columns if col != 'Date']
                 df_to_save = df_to_save[cols]
                 df_to_save.to_excel(writer, sheet_name='ARIMA_Forecast', index=False)
                 print(f"全样本历史和未来预测数据已写入 Sheet: ARIMA_Forecast")
            else:
                 print("警告: ARIMA 全样本历史/未来预测数据为空，未写入 Excel。")

            # 保存训练/验证集数据 (可选，但绘图已完成)
            if not arima_train_val_df.empty:
                df_tv_to_save = arima_train_val_df.copy()
                if isinstance(df_tv_to_save.index, pd.DatetimeIndex):
                    df_tv_to_save['Date'] = df_tv_to_save.index.strftime('%Y-%m-%d')
                else:
                    df_tv_to_save['Date'] = df_tv_to_save.index.astype(str)
                df_tv_to_save = df_tv_to_save.reset_index(drop=True)
                df_tv_to_save = df_tv_to_save.fillna('')
                cols_tv = ['Date'] + [col for col in df_tv_to_save.columns if col != 'Date']
                df_tv_to_save = df_tv_to_save[cols_tv]
                df_tv_to_save.to_excel(writer, sheet_name='ARIMA_Train_Validation', index=False)
                print(f"训练/验证集数据已写入 Sheet: ARIMA_Train_Validation")
            else:
                print("警告: ARIMA 训练/验证数据为空，未写入 Excel。")


        print(f"\n基准测试结果已追加或写入到: {results_save_path}")

    except Exception as e:
        print(f"保存结果到 Excel 时出错: {e}")
        traceback.print_exc()

    script_end_time = time.time()
    total_runtime_seconds = script_end_time - script_start_time
    print(f"\n总耗时: {total_runtime_seconds:.2f} 秒") 