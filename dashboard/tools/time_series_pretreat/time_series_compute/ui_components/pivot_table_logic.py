import pandas as pd
import numpy as np # 虽然当前直接用到的不多，但pandas依赖，且可能未来扩展计算会用到

def calculate_multi_variable_pivot_table(
    df_input_data: pd.DataFrame,
    selected_values: list[str],
    index_keys: list | None, # 可以是列名列表，也可以包含 pd.Grouper 对象
    column_keys, # 可以是列名，也可能是更复杂的结构如果列映射实现
    agg_functions: list[str],
    date_range_active: bool,
    filter_start_date,
    filter_end_date
) -> pd.DataFrame:
    """
    执行多变量数据透视表计算。

    参数:
    - df_input_data: 输入的DataFrame。
    - selected_values: 要聚合的数值变量列表。
    - index_keys: 用于行索引的键列表 (含pd.Grouper对象或列名)。
    - column_keys: 用于列索引的键。
    - agg_functions: 聚合函数列表 (字符串形式，如 ['mean', 'sum'])。
    - date_range_active: 日期范围筛选是否激活。
    - filter_start_date: 筛选的开始日期。
    - filter_end_date: 筛选的结束日期。

    返回:
    - pd.DataFrame: 计算得到的数据透视表。
    """
    
    df_pivot_input = df_input_data.copy() # 操作副本以避免修改原始数据

    # 1. 应用日期范围筛选 (如果激活且有效)
    if date_range_active and filter_start_date and filter_end_date and filter_start_date <= filter_end_date:
        if not isinstance(df_pivot_input.index, pd.DatetimeIndex):
            # 如果索引不是 DatetimeIndex，但尝试了日期筛选，这是一个潜在问题。
            # 根据UI设计，此时 date_range_active 可能不应该为True，或者应有警告。
            # 为保持计算函数纯粹性，这里不发 st.warning，但实际应用中应考虑。
            pass # 或者根据需求决定是否抛出错误
        else:
            try:
                df_pivot_input = df_pivot_input.loc[str(filter_start_date):str(filter_end_date)]
                if df_pivot_input.empty:
                    # 返回空 DataFrame，让调用方处理UI提示
                    # 或者可以抛出一个特定异常
                    raise ValueError("选定时间范围内没有数据可用于透视。") 
            except Exception as e_slice:
                # 同样，可以将错误传递给调用方处理
                raise ValueError(f"透视表日期切片时出错: {e_slice}")

    # 2. 执行数据透视表操作
    # 注意: column_keys (pivot_columns_arg_final in UI) 的复杂映射逻辑 (如从文件读取)
    # 仍需在UI层准备好，或者将该映射逻辑也移入此模块的其他辅助函数。
    # 当前假设 column_keys 是可以直接被 pivot_table 使用的参数。

    pivot_result = pd.pivot_table(
        df_pivot_input,
        values=selected_values,
        index=index_keys, 
        columns=column_keys, 
        aggfunc=agg_functions,
        observed=False, 
    )
    
    return pivot_result

# 如果未来单变量透视有显著不同的计算逻辑，也可以在此处添加：
# def calculate_single_variable_pivot_table(...):
#     pass 