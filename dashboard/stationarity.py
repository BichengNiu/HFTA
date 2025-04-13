import pandas as pd
import numpy as np
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.seasonal import seasonal_decompose
from scipy import stats

def test_stationarity(series):
    """
    对时间序列进行 ADF 检验
    """
    # 首先检查数据是否有效
    if series is None or len(series) == 0:
        return {
            'adf_statistic': None,
            'p_value': None,
            'is_stationary': False,
            'error': 'Empty series'
        }
    
    # 移除 NaN 值
    series = series.dropna()
    
    # 检查处理后的数据是否足够进行分析
    if len(series) < 10:  # 设置最小样本量要求
        return {
            'adf_statistic': None,
            'p_value': None,
            'is_stationary': False,
            'error': 'Insufficient data points'
        }
    
    try:
        # 进行 ADF 检验
        result = adfuller(series, autolag='AIC')
        return {
            'adf_statistic': result[0],
            'p_value': result[1],
            'is_stationary': result[1] < 0.05,
            'error': None
        }
    except Exception as e:
        return {
            'adf_statistic': None,
            'p_value': None,
            'is_stationary': False,
            'error': str(e)
        }

def make_stationary(series, method='difference'):
    """
    将时间序列转换为平稳序列
    """
    if series is None or len(series) == 0:
        return pd.Series()
    
    # 移除 NaN 值
    series = series.dropna()
    
    if len(series) < 2:
        return pd.Series()
        
    try:
        if method == 'difference':
            return series.diff().dropna()
        elif method == 'log_difference':
            # 确保所有值都是正数
            if (series <= 0).any():
                # 如果有非正数，使用简单差分
                return series.diff().dropna()
            return (np.log(series)).diff().dropna()
        elif method == 'percentage_change':
            return series.pct_change().dropna()
        else:
            return series.diff().dropna()
    except Exception:
        # 如果转换失败，返回简单差分
        return series.diff().dropna()

def analyze_stationarity(df):
    """
    分析数据框中所有列的平稳性
    """
    results = []
    
    for column in df.columns:
        series = df[column]
        
        # 检查数据是否有效
        if series.empty or series.isna().all():
            results.append({
                'Variable': column,
                'ADF Statistic': None,
                'P-value': None,
                'Is Stationary': False,
                'Recommended Method': None,
                'Error': 'No valid data'
            })
            continue
            
        # 移除 NaN 值
        clean_series = series.dropna()
        
        if len(clean_series) < 10:
            results.append({
                'Variable': column,
                'ADF Statistic': None,
                'P-value': None,
                'Is Stationary': False,
                'Recommended Method': None,
                'Error': 'Insufficient data points'
            })
            continue
        
        # 测试原始序列
        test_result = test_stationarity(clean_series)
        
        if test_result['error']:
            results.append({
                'Variable': column,
                'ADF Statistic': None,
                'P-value': None,
                'Is Stationary': False,
                'Recommended Method': None,
                'Error': test_result['error']
            })
            continue
        
        if test_result['is_stationary']:
            method = 'none'
        else:
            # 尝试不同的转换方法
            methods = ['difference', 'log_difference', 'percentage_change']
            best_method = None
            best_p_value = 1.0
            
            for method in methods:
                transformed = make_stationary(clean_series, method)
                if len(transformed) >= 10:
                    try:
                        method_result = test_stationarity(transformed)
                        if method_result['is_stationary'] and method_result['p_value'] < best_p_value:
                            best_method = method
                            best_p_value = method_result['p_value']
                    except:
                        continue
            
            method = best_method if best_method else 'difference'
        
        results.append({
            'Variable': column,
            'ADF Statistic': test_result['adf_statistic'],
            'P-value': test_result['p_value'],
            'Is Stationary': test_result['is_stationary'],
            'Recommended Method': method,
            'Error': None
        })
    
    return pd.DataFrame(results) 