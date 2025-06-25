import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime
from typing import List, Optional

def calculate_historical_weekly_stats(indicator_series: pd.Series, current_year: int) -> pd.DataFrame:
    """
    Calculate historical weekly statistics for an indicator.
    """
    if indicator_series.empty:
        return pd.DataFrame(index=range(1, 54), columns=['hist_min', 'hist_max', 'hist_mean'])

    # Filter data to only include years before current_year
    indicator_series = indicator_series[indicator_series.index.year < current_year]
    
    if indicator_series.empty:
        return pd.DataFrame(index=range(1, 54), columns=['hist_min', 'hist_max', 'hist_mean'])

    # Initialize result DataFrame
    historical_stats = pd.DataFrame(index=range(1, 54), columns=['hist_min', 'hist_max', 'hist_mean'])
    
    # Calculate stats for each week
    for week in range(1, 54):
        # Get data for this specific week across all historical years
        week_mask = indicator_series.index.isocalendar().week == week
        week_data = indicator_series[week_mask]
        
        if not week_data.empty:
            historical_stats.loc[week, 'hist_min'] = week_data.min()
            historical_stats.loc[week, 'hist_max'] = week_data.max()
            historical_stats.loc[week, 'hist_mean'] = week_data.mean()
            
    return historical_stats

def get_friday_date(year, week):
    """Helper function to get Friday date for a given year and week."""
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
    """
    Generates an interactive Plotly chart for a weekly indicator.
    """
    fig = go.Figure()

    indicator_series.index = pd.to_datetime(indicator_series.index)
    
    current_year_data = indicator_series[indicator_series.index.year == current_year].copy()
    previous_year_data = indicator_series[indicator_series.index.year == previous_year].copy()

    # Prepare plot data based on week number for X-axis alignment
    all_weeks = pd.Index(range(1, 54), name='week')
    week_labels = [f"W{w}" for w in all_weeks]
    week_indices = all_weeks.values

    plot_data = pd.DataFrame(index=all_weeks)

    # Join historical_stats directly as it's already indexed by week
    plot_data = plot_data.join(historical_stats)
    
    # Current and previous year data mapping
    current_year_plot_data = current_year_data.groupby(current_year_data.index.isocalendar().week).last().reindex(all_weeks)
    previous_year_plot_data = previous_year_data.groupby(previous_year_data.index.isocalendar().week).last().reindex(all_weeks)
    plot_data[f'{current_year}年'] = current_year_plot_data
    plot_data[f'{previous_year}年'] = previous_year_plot_data

    # Add Traces
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

    # 2. 历史均值 (灰色虚线)
    fig.add_trace(go.Scatter(
        x=week_indices,
        y=plot_data['hist_mean'],
        mode='lines+markers',
        line=dict(color='grey', dash='dash'),
        name='近5年均值',
        showlegend=True,
        hovertemplate=f'近5年均值 (W%{{x}}): %{{y:.2f}}<extra></extra>'
    ))

    # 3. 上年数据 (蓝色实线)
    fig.add_trace(go.Scatter(
        x=week_indices,
        y=plot_data[f'{previous_year}年'],
        mode='lines+markers',
        name=f'{previous_year}年',
        line=dict(color='blue'),
        hovertemplate=f'{previous_year}年 (W%{{x}}): %{{y:.2f}}<extra></extra>'
    ))

    # 4. 当年数据 (红色实线)
    fig.add_trace(go.Scatter(
        x=week_indices,
        y=plot_data[f'{current_year}年'],
        mode='lines+markers',
        name=f'{current_year}年',
        line=dict(color='red'),
        hovertemplate=f'{current_year}年 (W%{{x}}): %{{y:.2f}}<extra></extra>'
    ))

    # Layout and Styling
    fig.update_layout(
        title=indicator_name,
        hovermode='closest',
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
        if month == 12:
            return pd.Timestamp(year, 12, 31)
        else:
            return pd.Timestamp(year, month + 1, 1) - pd.Timedelta(days=1)
    except ValueError:
        return pd.NaT

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

    # Prepare customdata for hover
    customdata_monthly = []
    for month in range(1, 13):
        date_obj = get_month_end_date(current_year, month)
        date_str = date_obj.strftime('%Y-%m-%d') if pd.notna(date_obj) else "日期无效"
        current_val = current_year_monthly.iloc[month - 1] if pd.notna(current_year_monthly.iloc[month - 1]) else "无数据"
        hist_min = historical_stats.loc[month, 'hist_min'] if pd.notna(historical_stats.loc[month, 'hist_min']) else "无数据"
        hist_max = historical_stats.loc[month, 'hist_max'] if pd.notna(historical_stats.loc[month, 'hist_max']) else "无数据"
        hist_mean = historical_stats.loc[month, 'hist_mean'] if pd.notna(historical_stats.loc[month, 'hist_mean']) else "无数据"
        customdata_monthly.append([date_str, current_val, hist_min, hist_max, hist_mean])

    fig = go.Figure()

    # 1. 历史区间 (灰色区域)
    fig.add_trace(go.Scatter(
        x=list(range(1, 13)) + list(range(12, 0, -1)),
        y=list(historical_stats['hist_max']) + list(historical_stats['hist_min'][::-1]),
        fill='toself',
        fillcolor='rgba(211, 211, 211, 0.5)',
        line=dict(color='rgba(255,255,255,0)'),
        hoverinfo='skip',
        showlegend=True,
        name='历史区间'
    ))

    # 2. 历史均值 (灰色虚线)
    fig.add_trace(go.Scatter(
        x=list(range(1, 13)),
        y=historical_stats['hist_mean'],
        mode='lines+markers',
        line=dict(color='grey', dash='dash'),
        name='近5年均值',
        hovertemplate='近5年均值 (%{x}月): %{y:.2f}<extra></extra>'
    ))

    # 3. 上年数据 (蓝色实线)
    fig.add_trace(go.Scatter(
        x=list(range(1, 13)),
        y=previous_year_monthly,
        mode='lines+markers',
        name=f'{previous_year}年',
        line=dict(color='blue'),
        hovertemplate=f'{previous_year}年 (%{{x}}月): %{{y:.2f}}<extra></extra>'
    ))

    # 4. 当年数据 (红色实线)
    fig.add_trace(go.Scatter(
        x=list(range(1, 13)),
        y=current_year_monthly,
        mode='lines+markers',
        name=f'{current_year}年',
        line=dict(color='red'),
        customdata=customdata_monthly,
        hovertemplate=f'{current_year}年 (%{{x}}月): %{{y:.2f}}<br>' +
                      '日期: %{customdata[0]}<br>' +
                      '近5年最小值: %{customdata[2]}<br>' +
                      '近5年最大值: %{customdata[3]}<br>' +
                      '近5年平均值: %{customdata[4]}<extra></extra>'
    ))

    fig.update_layout(
        title=indicator_name,
        yaxis_title='数值',
        hovermode='closest',
        hoverlabel=dict(bgcolor="white", font_size=12),
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
            tickvals=list(range(1, 13)),
            ticktext=[f'{m}月' for m in range(1, 13)]
        )
    )

    return fig 

def plot_daily_indicator(
    indicator_series: pd.Series,
    indicator_name: str,
    current_year: int,
    previous_year: int
) -> go.Figure:
    """
    Generates an interactive Plotly chart for a daily indicator.

    Args:
        indicator_series (pd.Series): Time series data for the indicator (daily).
        indicator_name (str): Name of the indicator for the title.
        current_year (int): The year to plot as the current year (red line).
        previous_year (int): The year to plot as the previous year (blue line).

    Returns:
        go.Figure: Plotly figure object.
    """
    if indicator_series.empty:
        return go.Figure().update_layout(title=f"{indicator_name} - 无数据")

    indicator_series.index = pd.to_datetime(indicator_series.index)
    
    # Filter data for the years we want to display
    current_year_data = indicator_series[indicator_series.index.year == current_year].copy()
    previous_year_data = indicator_series[indicator_series.index.year == previous_year].copy()
    
    # Historical data (for range display, exclude current and previous year)
    historical_data = indicator_series[indicator_series.index.year < previous_year]
    
    # Calculate historical stats by day of year for comparison
    historical_stats = pd.DataFrame(index=range(1, 367), columns=['hist_min', 'hist_max', 'hist_mean'])
    
    if not historical_data.empty:
        for dayofyear in range(1, 367):
            # Get data for this specific day of year across all historical years
            day_mask = historical_data.index.dayofyear == dayofyear
            day_data = historical_data[day_mask]
            
            if not day_data.empty:
                historical_stats.loc[dayofyear, 'hist_min'] = day_data.min()
                historical_stats.loc[dayofyear, 'hist_max'] = day_data.max()
                historical_stats.loc[dayofyear, 'hist_mean'] = day_data.mean()

    fig = go.Figure()

    # Prepare x-axis: use day of year for alignment
    if not current_year_data.empty:
        current_x = current_year_data.index.dayofyear
        current_y = current_year_data.values
        current_dates = current_year_data.index.strftime('%Y-%m-%d')
    else:
        current_x, current_y, current_dates = [], [], []

    if not previous_year_data.empty:
        prev_x = previous_year_data.index.dayofyear
        prev_y = previous_year_data.values
        prev_dates = previous_year_data.index.strftime('%Y-%m-%d')
    else:
        prev_x, prev_y, prev_dates = [], [], []

    # 1. 历史区间 (灰色区域) - if we have historical data
    if not historical_stats.isna().all().all():
        days_range = list(range(1, 367))
        fig.add_trace(go.Scatter(
            x=days_range + days_range[::-1],
            y=list(historical_stats['hist_max'].ffill().bfill()) + 
              list(historical_stats['hist_min'].ffill().bfill()[::-1]),
            fill='toself',
            fillcolor='rgba(211, 211, 211, 0.5)',
            line=dict(color='rgba(255,255,255,0)'),
            hoverinfo='skip',
            showlegend=True,
            name='历史区间'
        ))

        # 2. 历史均值 (灰色虚线)
        fig.add_trace(go.Scatter(
            x=days_range,
            y=historical_stats['hist_mean'].ffill().bfill(),
            mode='lines',
            line=dict(color='grey', dash='dash'),
            name='历史均值',
            hovertemplate='历史均值 (第%{x}天): %{y:.2f}<extra></extra>'
        ))

    # 3. 上年数据 (蓝色实线)
    if len(prev_x) > 0:
        fig.add_trace(go.Scatter(
            x=prev_x,
            y=prev_y,
            mode='lines+markers',
            name=f'{previous_year}年',
            line=dict(color='blue'),
            customdata=prev_dates,
            hovertemplate=f'{previous_year}年 (%{{customdata}}): %{{y:.2f}}<extra></extra>'
        ))

    # 4. 当年数据 (红色实线)
    if len(current_x) > 0:
        fig.add_trace(go.Scatter(
            x=current_x,
            y=current_y,
            mode='lines+markers',
            name=f'{current_year}年',
            line=dict(color='red'),
            customdata=current_dates,
            hovertemplate=f'{current_year}年 (%{{customdata}}): %{{y:.2f}}<extra></extra>'
        ))

    # Layout and Styling
    fig.update_layout(
        title=indicator_name,
        yaxis_title='数值',
        hovermode='closest',
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
            tickvals=[1, 32, 60, 91, 121, 152, 182, 213, 244, 274, 305, 335],
            ticktext=['1月', '2月', '3月', '4月', '5月', '6月', '7月', '8月', '9月', '10月', '11月', '12月']
        )
    )

    return fig 