"""
扩散指数计算模块

此模块提供了一个核心函数用于计算扩散指数。
扩散指数用于衡量经济指标的变化趋势，取值范围为0-100。

主要功能：
- 计算扩散指数（支持同比、环比和组合计算）
- 支持自定义变化判定阈值
- 返回标准化的扩散指数值（0-100）
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os

def preprocess_black_metal_data(file_path: str) -> pd.DataFrame:
    """
    预处理黑色金属冶炼行业数据
    
    参数:
    file_path: str
        Excel文件路径，包含weekly sheet
    
    返回:
    pd.DataFrame: 处理后的数据框，包含时间索引
    """
    try:
        # 读取Excel文件中的weekly sheet
        df = pd.read_excel(file_path, sheet_name='weekly')
        
        # 确保第一列是日期列
        if not pd.api.types.is_datetime64_any_dtype(df.iloc[:, 0]):
            df.iloc[:, 0] = pd.to_datetime(df.iloc[:, 0])
        
        # 将第一列设置为索引
        df.set_index(df.columns[0], inplace=True)

        # 确保数据按时间排序
        df.sort_index(inplace=True)
        
        return df
        
    except Exception as e:
        raise ValueError(f"数据预处理失败: {str(e)}")

def calculate_diffusion_index(
    df: pd.DataFrame,
    threshold: float = 0
) -> pd.DataFrame:
    """
    计算扩散指数
    
    参数:
    df: pd.DataFrame
        输入数据框，必须包含时间索引
    threshold: float, 默认值 0
        变化判定阈值，用于判断指标变化绝对值相对于基期值的百分比是否超过阈值
    
    返回:
    pd.DataFrame: 包含三种扩散指数的数据框
        - 'yoy': 同比扩散指数
        - 'mom': 环比扩散指数
        - 'combined': 同环比扩散指数
        如果缺失值过多，对应的值将为NaN
    """
    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("输入数据框必须包含时间索引")
    
    # 确保数据按时间排序
    df = df.sort_index()
    
    # 创建结果DataFrame，使用时间索引
    result_df = pd.DataFrame(index=df.index)
    
    # 计算变化率
    yoy_changes = df.pct_change(periods=52)
    mom_changes = df.pct_change(periods=4)
    
    # 计算组合变化率：先计算同比变化，再计算环比变化
    # 1. 计算同比变化
    yoy_relative = df / df.shift(52) - 1
    # 2. 计算同比变化的环比变化
    combined_changes = yoy_relative.pct_change(periods=4)
    
    # 分别检查每种变化率的缺失值比例
    def check_missing_ratio(changes):
        missing_ratio = changes.isna().mean(axis=1)
        return missing_ratio <= 0.33  # 缺失值不超过三分之一
    
    valid_weeks_yoy = check_missing_ratio(yoy_changes)
    valid_weeks_mom = check_missing_ratio(mom_changes)
    valid_weeks_combined = check_missing_ratio(combined_changes)
    
    # 按周计算扩散指数
    def calculate_weekly_di(changes, valid_weeks):
        weekly_di = []
        
        for i, week in enumerate(changes.index):
            if not valid_weeks.iloc[i]:
                weekly_di.append(np.nan)
                continue
                
            week_data = changes.loc[week]
            # 修改判断条件：变化的绝对值（相对于基期的百分比）是否大于阈值
            improvements = (week_data.abs() > threshold).sum()
            total = week_data.count()  # 非NaN的数量
            
            di_value = (improvements / total) * 100
            weekly_di.append(round(di_value, 2))
                
        return weekly_di
    
    # 计算每周的扩散指数
    result_df['yoy'] = calculate_weekly_di(yoy_changes, valid_weeks_yoy)
    result_df['mom'] = calculate_weekly_di(mom_changes, valid_weeks_mom)
    result_df['combined'] = calculate_weekly_di(combined_changes, valid_weeks_combined)
    
    return result_df

def plot_diffusion_index(
    di_df: pd.DataFrame,
    save_dir: str = None
) -> None:
    """
    绘制交互式扩散指数时间序列图，每种指数一张图
    
    参数:
    di_df: pd.DataFrame
        包含扩散指数的数据框，必须包含'yoy'、'mom'和'combined'列
    save_dir: str, 默认值 None
        图表保存目录，如果为None则不保存
    """
    # 定义每种指数的标题、标签和颜色
    index_info = {
        'yoy': ('同比扩散指数', '同比', '#1f77b4'),  # 蓝色
        'mom': ('环比扩散指数', '环比', '#ff7f0e'),  # 橙色
        'combined': ('同环比扩散指数', '同环比', '#2ca02c')  # 绿色
    }
    
    # 创建保存目录
    if save_dir and not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    # 为每种指数创建单独的图表
    for index_type, (title, label, color) in index_info.items():
        # 创建图形
        fig = go.Figure()
        
        # 添加扩散指数线
        fig.add_trace(go.Scatter(
            x=di_df.index,
            y=di_df[index_type],
            name=label,
            mode='lines+markers',
            line=dict(width=2, color=color),
            marker=dict(size=4, color=color)
        ))
        
        # 添加中性线
        fig.add_hline(y=50, line_dash="dash", line_color="red", opacity=0.5)
        
        # 更新布局
        fig.update_layout(
            yaxis_title="扩散指数",
            yaxis=dict(range=[0, 100]),
            showlegend=True,
            hovermode='x unified',
            template='plotly_white'
        )
        
        # 更新x轴
        fig.update_xaxes(
            rangeslider_visible=True,  # 添加范围滑块
            rangeselector=dict(  # 添加范围选择器
                buttons=list([
                    dict(count=6, label="6月", step="month", stepmode="backward"),
                    dict(count=1, label="1年", step="year", stepmode="backward"),
                    dict(count=3, label="3年", step="year", stepmode="backward"),
                    dict(step="all", label="全部")
                ])
            )
        )
        
        # 保存图表
        if save_dir:
            save_path = f"{save_dir}/{title}.html"
            fig.write_html(save_path)
            print(f"图表已保存至: {save_path}")
        
        # 显示图表
        fig.show()

