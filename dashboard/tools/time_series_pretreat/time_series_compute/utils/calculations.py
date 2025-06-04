import pandas as pd
import numpy as np

# --- Calculation Utilities ---

def calculate_moving_average(data: pd.DataFrame, variable: str, window: int) -> pd.Series:
    """计算指定变量的移动平均值

    Args:
        data (pd.DataFrame): 输入的数据帧。
        variable (str): 需要计算移动平均的列名。
        window (int): 移动平均的窗口大小。

    Returns:
        pd.Series: 计算得到的移动平均序列。

    Raises:
        KeyError: 如果列 'variable' 不存在。
        TypeError: 如果列 'variable' 不是数值类型。
        ValueError: 如果窗口大小小于 1。
        Exception: 如果滚动计算时发生其他错误。
    """
    if variable not in data.columns:
        raise KeyError(f"错误：列 '{variable}' 不存在于数据中。")
    if not pd.api.types.is_numeric_dtype(data[variable]):
        raise TypeError(f"错误：列 '{variable}' 不是数值类型，无法计算移动平均。")
    if window < 1:
        raise ValueError("错误：移动平均窗口必须大于等于 1。")
    # 窗口大于数据长度的情况 pandas rolling 会自动处理（返回 NaN），无需在此处特别处理或警告

    try:
        # min_periods=1 确保即使窗口内数据不足，只要有1个有效值也计算
        return data[variable].rolling(window=window, min_periods=1).mean()
    except Exception as e:
        # 可以选择重新抛出更具体的异常或原始异常
        raise Exception(f"计算移动平均时发生内部错误: {e}") from e


def calculate_growth_rate(data: pd.DataFrame, variable: str, periods: int) -> pd.Series:
    """计算指定变量的期间增长率 (period-over-period)

    Args:
        data (pd.DataFrame): 输入的时间序列数据。
        variable (str): 需要计算增长率的变量名。
        periods (int): 计算增长率的期数 (例如，环比为1，同比通常根据数据频率设定)。

    Returns:
        pd.Series: 计算得到的增长率序列 (小数形式)。

    Raises:
        KeyError: 如果列 'variable' 不存在。
        TypeError: 如果列 'variable' 不是数值类型。
        ValueError: 如果期数 'periods' 小于 1。
        Exception: 如果计算时发生其他错误。
    """
    if variable not in data.columns:
        raise KeyError(f"错误：列 '{variable}' 不存在于数据中。")
    if not pd.api.types.is_numeric_dtype(data[variable]):
        raise TypeError(f"错误：列 '{variable}' 不是数值类型，无法计算增长率。")
    if periods < 1:
        raise ValueError("错误：计算增长率的期数必须大于等于 1。")
    # periods 大于等于数据长度的情况 pandas shift/division 会自动处理（返回 NaN），无需特别处理

    try:
        original_series = data[variable] # 直接操作原 Series 即可
        shifted_series = original_series.shift(periods)
        growth_series = (original_series - shifted_series) / shifted_series

        # 处理除零或其他计算问题导致的 inf/-inf
        growth_series.replace([np.inf, -np.inf], np.nan, inplace=True)

        return growth_series
    except Exception as e:
        raise Exception(f"计算增长率时发生内部错误: {e}") from e


def apply_single_variable_transform(data: pd.DataFrame, variable: str, transform_type: str, **kwargs) -> tuple[pd.Series, str | None]:
    """对单个变量应用指定的转换

    Args:
        data (pd.DataFrame): 输入数据帧。
        variable (str): 要转换的列名。
        transform_type (str): 转换类型 ('Log', 'Exp', 'Abs', 'Cumsum', 'Cumprod', 'Cummin', 'Cummax', 'Diff', 'Moving Average', 'AddConstant', 'SubtractConstant', 'MultiplyConstant', 'DivideConstant')。
        **kwargs: 转换所需的额外参数 (例如 'periods' for Diff, 'window' for Moving Average, 'constant_value' for AddConstant, SubtractConstant, MultiplyConstant, DivideConstant)。

    Returns:
        tuple[pd.Series, str | None]: 包含转换后序列和可选警告信息的元组。

    Raises:
        KeyError: 如果列 'variable' 不存在。
        TypeError: 如果列 'variable' 不是数值类型 (除 Abs 外)。
        ValueError: 如果转换类型未知，或参数无效 (如 periods<1, window<1)。
        Exception: 如果计算时发生其他错误。
    """
    if variable not in data.columns:
        raise KeyError(f"错误：列 '{variable}' 不存在于数据中。")

    series = data[variable]
    warning_message = None

    # Abs 可以处理非数值类型，其他需要检查
    if transform_type != 'Abs' and not pd.api.types.is_numeric_dtype(series):
         raise TypeError(f"错误：列 '{variable}' 不是数值类型，无法进行 '{transform_type}' 转换。")

    try:
        if transform_type == 'Log':
            # 检查非正数
            if (series <= 0).any():
                warning_message = f"警告：列 '{variable}' 包含非正数，Log 转换结果中这些位置将为 NaN。"
                # 使用 mask 替换非正数为 NaN 以允许计算
                series_masked = series.mask(series <= 0)
                result = np.log(series_masked)
            else:
                 result = np.log(series)
            return result, warning_message
        elif transform_type == 'Exp':
            return np.exp(series), None
        elif transform_type == 'Abs':
            return series.abs(), None
        elif transform_type == 'Cumsum':
            return series.cumsum(), None
        elif transform_type == 'Cumprod':
            return series.cumprod(), None
        elif transform_type == 'Cummin':
            return series.cummin(), None
        elif transform_type == 'Cummax':
            return series.cummax(), None
        elif transform_type == 'Diff':
            periods = kwargs.get('periods', 1)
            if not isinstance(periods, int) or periods < 1:
                raise ValueError("错误：差分期数必须是大于等于 1 的整数。")
            return series.diff(periods=periods), None
        elif transform_type == 'Moving Average':
            window = kwargs.get('window', 3) # 默认窗口为 3
            # calculate_moving_average 会自行处理 window < 1 的 ValueError
            # 以及 variable not in data 或非数值型的 KeyError/TypeError
            return calculate_moving_average(data, variable, window), None
        elif transform_type == 'AddConstant':
            constant = kwargs.get('constant_value', 0)
            if not isinstance(constant, (int, float)):
                raise TypeError("错误：加法运算的常数值必须是数字。")
            return series + constant, None
        elif transform_type == 'SubtractConstant':
            constant = kwargs.get('constant_value', 0)
            if not isinstance(constant, (int, float)):
                raise TypeError("错误：减法运算的常数值必须是数字。")
            return series - constant, None
        elif transform_type == 'MultiplyConstant':
            constant = kwargs.get('constant_value', 1)
            if not isinstance(constant, (int, float)):
                raise TypeError("错误：乘法运算的常数值必须是数字。")
            return series * constant, None
        elif transform_type == 'DivideConstant':
            constant = kwargs.get('constant_value', 1)
            if not isinstance(constant, (int, float)):
                raise TypeError("错误：除法运算的常数值必须是数字。")
            if constant == 0:
                warning_message = f"警告：除数为零。变量 '{variable}' 除以常数 0 的结果将全部为 NaN。"
                return pd.Series(np.nan, index=series.index), warning_message
            return series / constant, None
        else:
            raise ValueError(f"错误：未知的单变量转换类型 '{transform_type}'。")

    except Exception as e:
        raise Exception(f"在对列 '{variable}' 执行 '{transform_type}' 转换时发生内部错误: {e}") from e

# 可以在此添加其他纯计算函数，如 variable_arithmetic, weighted_operation 等的实现
# def variable_arithmetic(...) -> pd.Series: ...
# def weighted_operation(...) -> pd.Series: ...
# def grouped_statistics(...) -> pd.DataFrame: ...
# def resample_frequency(...) -> pd.DataFrame: ... 