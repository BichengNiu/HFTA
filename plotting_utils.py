import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime

def calculate_historical_weekly_stats(indicator_series: pd.Series, current_year: int) -> pd.DataFrame:
    """
    Calculates historical weekly statistics (5-year min, max, mean) for a given indicator.

    Args:
        indicator_series (pd.Series): Time series data for a single indicator, indexed by datetime.
        current_year (int): The current year to exclude from historical calculation.

    Returns:
        pd.DataFrame: DataFrame indexed by week number (1-53) with columns 'hist_min', 'hist_max', 'hist_mean'.
    """
    if indicator_series.empty:
        return pd.DataFrame(columns=['hist_min', 'hist_max', 'hist_mean'])
        
    # Filter out current year and future data if any
    historical_data = indicator_series[indicator_series.index.year < current_year].copy()
    
    if historical_data.empty:
        return pd.DataFrame(columns=['hist_min', 'hist_max', 'hist_mean'])

    # Get week number (ISO week, 1-53)
    historical_data.index = pd.to_datetime(historical_data.index)
    
    # 按周分组，对每个周分别计算近5年的统计值
    all_weeks = pd.Index(range(1, 54), name='week')
    historical_stats = pd.DataFrame(index=all_weeks, columns=['hist_min', 'hist_max', 'hist_mean'], dtype=float) # Ensure float dtype
    
    for week in all_weeks:
        # 获取该周的所有历史数据
        week_data = historical_data[historical_data.index.isocalendar().week == week]
        if not week_data.empty:
            # 计算该周的历史统计值
            historical_stats.loc[week, 'hist_min'] = week_data.min()
            historical_stats.loc[week, 'hist_max'] = week_data.max()
            historical_stats.loc[week, 'hist_mean'] = week_data.mean()
    
    return historical_stats

def get_friday_date(year, week):
    # ... (函数内容不变) ...
    try:
        first_day = pd.Timestamp(f'{year}-01-01')
        first_thursday = first_day + pd.Timedelta(days=(3 - first_day.weekday() + 7) % 7)
        target_date = first_thursday + pd.Timedelta(weeks=week - 1, days=1)
        if target_date.year != year and week == 1:
             return pd.Timestamp(f'{year-1}-12-31') - pd.Timedelta(days=(pd.Timestamp(f'{year-1}-12-31').weekday() - 4 + 7) % 7)
        elif target_date.year != year and week > 50:
             return pd.Timestamp(f'{year+1}-01-01') + pd.Timedelta(days=(4 - pd.Timestamp(f'{year+1}-01-01').weekday() + 7) % 7)
        return target_date
    except Exception as e:
        print(f"Error calculating Friday date for year {year}, week {week}: {e}")
        return pd.NaT

def plot_weekly_indicator(
    indicator_series: pd.Series,
    historical_stats: pd.DataFrame,
    indicator_name: str,
    current_year: int,
    previous_year: int
) -> go.Figure:

    fig = go.Figure()

    indicator_series.index = pd.to_datetime(indicator_series.index)
    current_year_data = indicator_series[indicator_series.index.year == current_year].copy()
    previous_year_data = indicator_series[indicator_series.index.year == previous_year].copy()

    all_weeks = pd.Index(range(1, 54), name='week')
    week_labels = [f"W{w}" for w in all_weeks]
    week_indices = all_weeks.values

    # --- 准备绘图数据 --- # (Slightly simplified)
    plot_data = pd.DataFrame(index=all_weeks)
    plot_data = plot_data.join(historical_stats) # hist_min, hist_max, hist_mean
    current_year_plot_data = current_year_data.groupby(current_year_data.index.isocalendar().week).last().reindex(all_weeks)
    previous_year_plot_data = previous_year_data.groupby(previous_year_data.index.isocalendar().week).last().reindex(all_weeks)
    plot_data[f'{current_year}年'] = current_year_plot_data
    plot_data[f'{previous_year}年'] = previous_year_plot_data

    # --- 添加 Traces --- (调整 mode 和 hovertemplate, 移除 customdata)

    # 1. 历史区间 (灰色区域)
    fig.add_trace(go.Scatter(
        x=np.concatenate([week_indices, week_indices[::-1]]),
        y=np.concatenate([plot_data['hist_max'].values, plot_data['hist_min'].values[::-1]]),
        fill='toself',
        fillcolor='rgba(211, 211, 211, 0.5)',
        line=dict(color='rgba(255,255,255,0)'),
        hoverinfo='skip',
        showlegend=True,
        name='历史区间'
    ))

    # 2. 历史均值 (灰色虚线) - lines+markers, individual hover
    fig.add_trace(go.Scatter(
        x=week_indices,
        y=plot_data['hist_mean'],
        mode='lines+markers',
        line=dict(color='grey', dash='dash'),
        name='近5年均值',
        showlegend=True,
        hovertemplate=f'近5年均值 (W%{{x}}): %{{y:.2f}}<extra></extra>'
        # hoverinfo='skip' # REMOVE skip
    ))

    # 3. 上年数据 (蓝色实线) - lines+markers, individual hover
    fig.add_trace(go.Scatter(
        x=week_indices,
        y=plot_data[f'{previous_year}年'],
        mode='lines+markers',
        name=f'{previous_year}年',
        line=dict(color='blue'),
        hovertemplate=f'{previous_year}年 (W%{{x}}): %{{y:.2f}}<extra></extra>'
        # hoverinfo='skip' # REMOVE skip
    ))

    # 4. 当年数据 (红色实线) - lines+markers, individual hover
    fig.add_trace(go.Scatter(
        x=week_indices,
        y=plot_data[f'{current_year}年'],
        mode='lines+markers',
        name=f'{current_year}年',
        line=dict(color='red'),
        hovertemplate=f'{current_year}年 (W%{{x}}): %{{y:.2f}}<extra></extra>'
        # customdata=... # REMOVE customdata
        # hovertemplate=... # USE simplified template
    ))

    # --- Layout and Styling --- (Remove hovermode='x unified')
    fig.update_layout(
        title=indicator_name,
        # hovermode="x unified", # REMOVE unified hover
        hovermode='closest', # Use closest mode for individual hovers
        hoverlabel=dict(
            bgcolor="white",
            font_size=12,
        ),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=-0.2,
            xanchor="center",
            x=0.5
        ),
        margin=dict(l=20, r=20, t=40, b=80),
        xaxis=dict(
            tickmode='array',
            tickvals=week_indices[::4],
            ticktext=week_labels[::4],
        )
    )

    return fig 

def get_month_end_date(year, month):
    """Helper to get the last day of a given month and year."""
    try:
        # Create the first day of the next month, then subtract one day
        if month == 12:
            return pd.Timestamp(year, 12, 31)
        else:
            return pd.Timestamp(year, month + 1, 1) - pd.Timedelta(days=1)
    except ValueError:
        return pd.NaT # Handle invalid year/month combo if necessary

def plot_monthly_indicator(
    indicator_series: pd.Series,
    indicator_name: str,
    current_year: int,
    previous_year: int
) -> go.Figure:
    """
    Generates an interactive Plotly chart for a monthly indicator.

    Args:
        indicator_series (pd.Series): Time series data for the indicator (monthly).
        indicator_name (str): Name of the indicator for the title.
        current_year (int): The year to plot as the current year (red line).
        previous_year (int): The year to plot as the previous year (blue line).

    Returns:
        go.Figure: Plotly figure object.
    """
    if indicator_series.empty:
        return go.Figure().update_layout(title=f"{indicator_name} - 无数据")

    indicator_series.index = pd.to_datetime(indicator_series.index)
    plot_data = pd.DataFrame({
        'value': indicator_series,
        'year': indicator_series.index.year,
        'month': indicator_series.index.month
    })
    prior_years_data = plot_data[plot_data['year'] < previous_year]
    historical_stats = pd.DataFrame(index=range(1, 13), columns=['hist_min', 'hist_max', 'hist_mean'])
    if not prior_years_data.empty:
        # Calculate min, max, and mean for historical data
        monthly_stats = prior_years_data.groupby('month')['value'].agg(['min', 'max', 'mean'])
        historical_stats.loc[monthly_stats.index, 'hist_min'] = monthly_stats['min']
        historical_stats.loc[monthly_stats.index, 'hist_max'] = monthly_stats['max']
        historical_stats.loc[monthly_stats.index, 'hist_mean'] = monthly_stats['mean']

    current_year_monthly = plot_data[plot_data['year'] == current_year].set_index('month')['value'].reindex(range(1, 13))
    previous_year_monthly = plot_data[plot_data['year'] == previous_year].set_index('month')['value'].reindex(range(1, 13))

    # Prepare customdata for hover (date, current_val, hist_min, hist_max, hist_mean)
    customdata_monthly = []
    for month in range(1, 13):
        date_obj = get_month_end_date(current_year, month)
        date_str = date_obj.strftime('%Y-%m-%d') if pd.notna(date_obj) else "日期无效"
        current_val = current_year_monthly.get(month, np.nan)
        hist_min = historical_stats.loc[month, 'hist_min']
        hist_max = historical_stats.loc[month, 'hist_max']
        hist_mean = historical_stats.loc[month, 'hist_mean']
        customdata_monthly.append([date_str, current_val, hist_min, hist_max, hist_mean])

    # Define monthly hover template
    hover_template_monthly = (
        f"%{{x}}月: %{{customdata[0]}}<br>"  # Month: Date from customdata[0]
        f"当月值: %{{customdata[1]:.2f}}%<br>" # Current val from customdata[1]
        f"近5年区间: [%{{customdata[2]:.2f}}%, %{{customdata[3]:.2f}}%]<br>" # Min/Max from customdata[2]/[3]
        f"近5年均值: %{{customdata[4]:.2f}}%"   # Mean from customdata[4]
        "<extra></extra>"
    )

    # --- 创建图表 ---
    fig = go.Figure()
    months_indices = list(range(1, 13))
    month_labels = [f"M{m}" for m in months_indices]

    # 1. 历史区间 (灰色区域)
    hist_min_values = historical_stats['hist_min'].astype(float).values
    hist_max_values = historical_stats['hist_max'].astype(float).values
    fig.add_trace(go.Scatter(
        x=np.concatenate([months_indices, months_indices[::-1]]),
        y=np.concatenate([hist_max_values, hist_min_values[::-1]]),
        fill='toself',
        fillcolor='rgba(211, 211, 211, 0.5)',
        line=dict(color='rgba(255,255,255,0)'),
        hoverinfo='skip',
        showlegend=False,
        name='历史区间'
    ))

    # 2. 去年数据 (蓝色) - lines+markers, individual hover
    fig.add_trace(go.Scatter(
        x=months_indices,
        y=previous_year_monthly.values,
        mode='lines+markers',
        name=f'{previous_year}年',
        line=dict(color='blue'),
        # hoverinfo='skip' # REMOVE skip to allow individual hover
        hovertemplate=f'{previous_year}年 (%{{x}}月): %{{y:.2f}}%<extra></extra>' # Add individual template
    ))

    # 3. 当年数据 (红色) - lines+markers, individual hover (use simplified template)
    fig.add_trace(go.Scatter(
        x=months_indices,
        y=current_year_monthly.values,
        mode='lines+markers',
        name=f'{current_year}年',
        line=dict(color='red'),
        # customdata=customdata_monthly, # REMOVE customdata, not needed for simple hover
        hovertemplate=f'{current_year}年 (%{{x}}月): %{{y:.2f}}%<extra></extra>' # Use simple template
        # hovertemplate=hover_template_monthly # REMOVE detailed template
    ))

    # --- Layout and Styling --- (Change yaxis_title, remove hovermode unified)
    fig.update_layout(
        title=indicator_name,
        yaxis_title="当月同比 (%)", # Change Y-axis title
        # hovermode="x unified", # REMOVE unified hover
        hovermode="closest",     # Use closest for individual hovers
        hoverlabel=dict(bgcolor="white", font_size=12),
        legend=dict(orientation="h", yanchor="bottom", y=-0.2, xanchor="center", x=0.5),
        margin=dict(l=20, r=20, t=40, b=80),
        xaxis=dict(
            tickmode='array',
            tickvals=months_indices,
            ticktext=month_labels,
            showgrid=False
        ),
        yaxis=dict(
            ticksuffix="%"
        )
    )

    return fig 