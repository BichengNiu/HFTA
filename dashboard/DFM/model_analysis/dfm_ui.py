import streamlit as st
import pandas as pd
import logging # Keep logging for UI-side info/errors if needed
import plotly.graph_objects as go # <--- 新增 Plotly
import numpy as np
import joblib
import pickle
from typing import Optional, Dict, Any
from datetime import datetime # <<< 新增导入 for date input defaults
# --- 新增导入 ---
import os # <<< 新增导入 os

# 导入配置
try:
    from config import (
        DataDefaults, TrainDefaults, UIDefaults, VisualizationDefaults,
        FileDefaults, FormatDefaults, AnalysisDefaults
    )
    CONFIG_AVAILABLE = True
except ImportError as e:
    print(f"⚠️ 无法导入配置模块: {e}")
    CONFIG_AVAILABLE = False

# Import backend functions
# Note: Use absolute import to avoid relative import issues in standalone execution
try:
    from .dfm_backend import load_dfm_results_from_uploads, create_aligned_nowcast_target_table
except ImportError:
    # Fallback for standalone execution
    from dfm_backend import load_dfm_results_from_uploads, create_aligned_nowcast_target_table

logger = logging.getLogger(__name__)


def load_dfm_data(_session_state: Dict) -> tuple[Optional[Any], Optional[Dict]]:
    """从 session_state 加载模型结果和元数据。"""
    model_file = _session_state.get('dfm_model_file_indep')
    metadata_file = _session_state.get('dfm_metadata_file_indep')
    
    model_results = None
    metadata = None

    if model_file:
        try:
            model_file.seek(0) # 重置文件指针
            model_results = joblib.load(model_file)
            print("[DFM UI] Model loaded successfully from session state.")
        except Exception as e:
            st.error(f"加载模型文件 ('{model_file.name}') 时出错: {e}")
    
    if metadata_file:
        try:
            metadata_file.seek(0) # 重置文件指针
            metadata = pickle.load(metadata_file)
            print("[DFM UI] Metadata loaded successfully from session state.")
        except Exception as e:
            st.error(f"加载元数据文件 ('{metadata_file.name}') 时出错: {e}")

    return model_results, metadata

# --- Helper Function to Plot Factor Evolution ---
def plot_factor_evolution(factor_df: pd.DataFrame, title: str = "因子时间序列演变图"):
    """绘制因子随时间变化的曲线图。"""
    if not isinstance(factor_df, pd.DataFrame) or factor_df.empty:
        st.warning("因子数据无效，无法绘制演变图。")
        return
    
    fig = go.Figure()
    
    for col in factor_df.columns:
        fig.add_trace(go.Scatter(
            x=factor_df.index,
            y=factor_df[col],
            mode='lines',
            name=col,
            hovertemplate=(
                f"日期: %{{x}}<br>" +
                f"{col}: %{{y:.4f}}<extra></extra>"
            )
        ))

    fig.update_layout(
        title=title,
        xaxis_title="日期",
        yaxis_title="因子值",
        legend_title_text='因子',
        hovermode='x unified',
        height=400,
        margin=dict(t=50, b=50, l=50, r=50)
    )

    st.plotly_chart(fig, use_container_width=True)

# --- Helper Function to Plot Loadings Heatmap (修改后) ---
def plot_loadings_heatmap(loadings_df: pd.DataFrame, title: str = "因子载荷矩阵 (Lambda)", cluster_vars: bool = True):
    """
    绘制因子载荷矩阵的热力图 (因子在X轴, 变量在Y轴, 可选聚类)。

    Args:
        loadings_df: 包含因子载荷的 DataFrame (原始形式：变量为行，因子为列)。
        title: 图表标题。
        cluster_vars: 是否对变量进行聚类排序。
    """
    if not isinstance(loadings_df, pd.DataFrame) or loadings_df.empty:
        st.warning(f"无法绘制热力图：提供的载荷数据无效 ({title})。")
        return

    data_for_clustering = loadings_df.copy() # 变量是行
    variable_names = data_for_clustering.index.tolist()
    factor_names = data_for_clustering.columns.tolist()

    # 1. (如果需要) 对变量进行聚类
    if cluster_vars:
        try:
            if data_for_clustering.shape[0] <= 1: # 如果只有一个变量，无法聚类
                 print("[DFM UI] 只有一个变量，跳过聚类。")
            else:
                from scipy.cluster import hierarchy as sch
                linked = sch.linkage(data_for_clustering.values, method='ward', metric='euclidean')
                dendro = sch.dendrogram(linked, no_plot=True)
                clustered_indices = dendro['leaves']
                data_for_clustering = data_for_clustering.iloc[clustered_indices, :]
                variable_names = data_for_clustering.index.tolist() # 获取聚类后的变量顺序
                title += " (变量聚类排序)"
        except Exception as e_cluster:
            st.warning(f"变量聚类失败: {e_cluster}. 将按原始顺序显示。")
            # 如果聚类失败，variable_names 保持原始顺序

    # 2. 转置数据以便绘图 (因子在 X 轴, 变量在 Y 轴)
    plot_data_transposed = data_for_clustering.T # 转置后：因子是行，（聚类后）变量是列
    
    # 确保轴标签列表是最新的
    y_axis_labels = variable_names # 聚类后的变量名
    x_axis_labels = factor_names   # 原始因子名

    # 3. 创建热力图
    fig = go.Figure(data=go.Heatmap(
        z=plot_data_transposed.values, # 使用转置后的数据
        x=x_axis_labels,          # X 轴是因子
        y=y_axis_labels,          # Y 轴是变量 (按聚类顺序)
        colorscale='RdBu',
        zmid=0,
        hovertemplate=(
            "变量 (Variable): %{y}<br>" +
            "因子 (Factor): %{x}<br>" +
            "载荷值 (Loading): %{z:.4f}<extra></extra>"
        )
    ))

    # 4. 更新布局
    fig.update_layout(
        title=title,
        xaxis_title="因子 (Factors)",
        yaxis_title="变量 (Predictors)",
        xaxis_tickangle=-45, 
        # Y轴使用类别类型，并直接指定顺序
        yaxis=dict(
            type='category', 
            categoryorder='array', # 明确指定使用下面提供的数组顺序
            categoryarray=y_axis_labels # 确保Y轴按聚类顺序显示
        ), 
        height=max(600, len(y_axis_labels) * 20), # 调整高度计算
        margin=dict(l=150, r=30, t=80, b=100) # <<< 减小左右边距
    )
    
    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='LightGrey')
    fig.update_xaxes(showgrid=False) 

    st.plotly_chart(fig, use_container_width=True)

def render_file_upload_section(st_instance, session_state):
    """
    渲染文件上传区域
    """
    st_instance.markdown("##### 📁 上传模型文件")
    st_instance.markdown("请上传训练完成的DFM模型文件和元数据文件以进行结果分析。")
    
    # 创建两列布局
    col_model, col_metadata = st_instance.columns(2)
    
    with col_model:
        st_instance.markdown("**DFM 模型文件 (.joblib)**")
        uploaded_model_file = st_instance.file_uploader(
            "选择模型文件",
            type=['joblib'],
            key="dfm_model_upload_independent",
            help="上传训练好的DFM模型文件，通常为.joblib格式"
        )
        
        if uploaded_model_file:
            session_state.dfm_model_file_indep = uploaded_model_file
            st_instance.success(f"✅ 已上传: {uploaded_model_file.name}")
        elif session_state.get('dfm_model_file_indep'):
            st_instance.info(f"📎 当前文件: {session_state.dfm_model_file_indep.name}")
    
    with col_metadata:
        st_instance.markdown("**元数据文件 (.pkl)**")
        uploaded_metadata_file = st_instance.file_uploader(
            "选择元数据文件", 
            type=['pkl'],
            key="dfm_metadata_upload_independent",
            help="上传包含训练元数据的.pkl文件"
        )
        
        if uploaded_metadata_file:
            session_state.dfm_metadata_file_indep = uploaded_metadata_file
            st_instance.success(f"✅ 已上传: {uploaded_metadata_file.name}")
        elif session_state.get('dfm_metadata_file_indep'):
            st_instance.info(f"📎 当前文件: {session_state.dfm_metadata_file_indep.name}")
    
    # 文件状态总结
    model_file = session_state.get('dfm_model_file_indep')
    metadata_file = session_state.get('dfm_metadata_file_indep')
    
    if model_file and metadata_file:
        st_instance.success("🎉 所有必需文件已上传完成，可以进行模型分析。")
        return True
    else:
        missing_files = []
        if not model_file:
            missing_files.append("模型文件")
        if not metadata_file:
            missing_files.append("元数据文件")
        
        st_instance.warning(f"⚠️ 缺少文件: {', '.join(missing_files)}。请上传所有文件后再进行分析。")
        return False

def render_dfm_tab(st, session_state):
    """Renders the DFM Model Results tab using independently uploaded files."""
        
    # 添加文件上传区域
    st.markdown("---")
    files_ready = render_file_upload_section(st, session_state)
    
    if not files_ready:
        st.info("💡 请先上传模型文件和元数据文件以继续分析。")
        return

    model_results, metadata = load_dfm_data(session_state)

    if model_results is None or metadata is None:
        st.error("❌ 无法加载模型数据，请检查文件格式和内容。")
        return

    # --- 添加 DEBUG 打印 --- 
    # st.write("[DEBUG RENDER TAB] 准备调用 load_dfm_results_from_uploads...")
    # --- 结束 DEBUG 打印 --- 
    model, metadata, load_errors = load_dfm_results_from_uploads(model_results, metadata)
    # --- 添加 DEBUG 打印 --- 
    # st.write(f"[DEBUG RENDER TAB] 加载结果: model 类型={type(model)}, metadata 类型={type(metadata)}, load_errors={load_errors}")
    # --- 结束 DEBUG 打印 --- 
    
    all_errors = load_errors

    if all_errors:
        st.error("加载 DFM 相关文件时遇到错误:")
        for error in all_errors:
            st.error(f"- {error}")
        if model is None or metadata is None: 
            # --- 添加 DEBUG 打印 --- 
            # st.write("[DEBUG RENDER TAB] 因加载错误导致 model 或 metadata 为 None，函数提前返回。")
            # --- 结束 DEBUG 打印 --- 
            return
            
    if model is None or metadata is None:
        st.warning("未能成功加载 DFM 模型或元数据。请检查文件内容或格式。")
        # --- 添加 DEBUG 打印 --- 
        # st.write("[DEBUG RENDER TAB] 因 model 或 metadata 为 None (但无加载错误)，函数提前返回。")
        # --- 结束 DEBUG 打印 --- 
        return
    
    st.success("成功加载 DFM 模型和元数据！")



    # --- 关键结果摘要 (移到此处) ---
    st.write(f"- **目标变量:** {metadata.get('target_variable', 'N/A')}")
    train_start = metadata.get('training_start_date', 'N/A')
    # 修正训练结束日期键名 - 训练模块使用 'training_end_date'
    train_end = metadata.get('training_end_date', metadata.get('train_end_date', 'N/A'))
    val_start = metadata.get('validation_start_date', 'N/A')
    val_end = metadata.get('validation_end_date', 'N/A')
    st.write(f"- **训练期:** {train_start} 至 {train_end}")
    st.write(f"- **验证期:** {val_start} 至 {val_end}")
    
    best_params_dict = metadata.get('best_params', {})
    var_select_method = best_params_dict.get('variable_selection_method', '未指定') 
    tuning_objective = best_params_dict.get('tuning_objective', '未指定')
    st.write(f"- **变量选择方法:** {var_select_method} (优化目标: {tuning_objective})") # Removed trailing space from original
    if var_select_method == '未指定' or tuning_objective == '未指定':
        st.caption(":grey[注：未能从元数据 'best_params' 字典中找到 'variable_selection_method' 或 'tuning_objective'。]")
    st.markdown("---") # 分隔线
    # --- 结束关键结果摘要 ---
    
    # 从元数据获取指标
    # 修正因子数量的获取方式 - 训练模块使用 'k_factors' 作为key
    k_factors = metadata.get('best_params', {}).get('k_factors', metadata.get('n_factors', 'N/A'))
    if k_factors == 'N/A':
        # 从DFM结果中推断
        factor_loadings = metadata.get('factor_loadings_df')
        if factor_loadings is not None and hasattr(factor_loadings, 'columns'):
            k_factors = len(factor_loadings.columns)
    n_vars = len(metadata.get('best_variables', []))

    # --- 使用后端计算的修正后指标 ---
    revised_is_hr = metadata.get('revised_is_hr')
    revised_oos_hr = metadata.get('revised_oos_hr')
    revised_is_rmse = metadata.get('revised_is_rmse')
    revised_oos_rmse = metadata.get('revised_oos_rmse')
    revised_is_mae = metadata.get('revised_is_mae')
    revised_oos_mae = metadata.get('revised_oos_mae')
    # --- 结束使用修正后指标 ---

    def format_value(val, is_percent=False, precision=2):
        if isinstance(val, (int, float)) and pd.notna(val):
            if is_percent:
                # MODIFIED: Assume val is already the percentage value if is_percent is True
                # e.g., if val is 72.3, it represents 72.3%
                return f"{val:.{precision}f}%" 
            return f"{val:.{precision}f}"
        return 'N/A' if val == 'N/A' or pd.isna(val) else str(val)

    # --- 第一行 ---
    row1_col1, row1_col2 = st.columns(2)
    with row1_col1:
        st.metric("最终因子数 (k)", k_factors if isinstance(k_factors, int) else 'N/A')
    with row1_col2:
        st.metric("最终变量数 (N)", n_vars if isinstance(n_vars, int) else 'N/A')

    # --- 第二行 ---
    row2_col1, row2_col2 = st.columns(2)
    with row2_col1:
        st.metric("训练期胜率", format_value(revised_is_hr, is_percent=True))
    with row2_col2:
        st.metric("验证期胜率", format_value(revised_oos_hr, is_percent=True))
        
    # --- 第三行 ---
    row3_col1, row3_col2, row3_col3, row3_col4 = st.columns(4)
    with row3_col1:
        st.metric("样本内 RMSE", format_value(revised_is_rmse))
    with row3_col2:
        st.metric("样本外 RMSE", format_value(revised_oos_rmse))
    with row3_col3:
        st.metric("样本内 MAE", format_value(revised_is_mae))
    with row3_col4:
        st.metric("样本外 MAE", format_value(revised_oos_mae))

    # --- "Nowcast 与实际值对比图及详细数据" - 直接显示内容 ---
    # 原 with st.expander("Nowcast 与实际值对比图及详细数据", expanded=True):
    
    # --- 添加 DEBUG 打印 --- 
    # st.write("[DEBUG RENDER TAB] 准备获取 nowcast_aligned 和 y_test_aligned...")
    # --- 结束 DEBUG 打印 --- 
    nowcast_series = metadata.get('nowcast_aligned')
    actual_series = metadata.get('y_test_aligned')
    # 使用配置的默认值或从元数据获取
    if CONFIG_AVAILABLE:
        default_target_name = DataDefaults.TARGET_VARIABLE
    else:
        default_target_name = '规模以上工业增加值当月同比'
    target_variable_name_for_plot = metadata.get('target_variable', default_target_name)
    # --- 添加 DEBUG 打印 --- 
    # st.write(f"[DEBUG RENDER TAB] 获取的序列: nowcast_aligned 类型={type(nowcast_series)}, y_test_aligned 类型={type(actual_series)}")
    # if isinstance(nowcast_series, pd.Series): st.write(f"[DEBUG RENDER TAB] nowcast_series.shape={nowcast_series.shape}")
    # if isinstance(actual_series, pd.Series): st.write(f"[DEBUG RENDER TAB] actual_series.shape={actual_series.shape}")
    # --- 结束 DEBUG 打印 --- 

    if nowcast_series is not None and actual_series is not None:
        nowcast_display_name = "Nowcast值"
        target_display_name = target_variable_name_for_plot 
        
        # --- 添加 DEBUG 打印 --- 
        # st.write("[DEBUG RENDER TAB] 准备调用 create_aligned_nowcast_target_table...")
        # --- 结束 DEBUG 打印 --- 
        comparison_df = create_aligned_nowcast_target_table(
            nowcast_series=nowcast_series,
            target_series=actual_series,
            nowcast_col_name=nowcast_display_name, 
            target_col_name=target_display_name
        )
        # --- 添加 DEBUG 打印 --- 
        # st.write(f"[DEBUG RENDER TAB] comparison_df 生成完成，Shape={comparison_df.shape if comparison_df is not None else 'None'}")
        # --- 结束 DEBUG 打印 --- 
        
        if not comparison_df.empty:
            if not isinstance(comparison_df.index, pd.DatetimeIndex):
                try:
                    comparison_df.index = pd.to_datetime(comparison_df.index)
                    comparison_df = comparison_df.sort_index()
                except Exception as e:
                    st.error(f"无法将对比数据的索引转换为日期时间格式: {e}")
                    st.warning("由于索引格式问题，回退到显示表格。")
                    st.dataframe(comparison_df, use_container_width=True)
            # --- 添加 DEBUG 打印 --- 
            # st.write("[DEBUG RENDER TAB] 准备进入 Matplotlib 绘图逻辑...")
            # --- 结束 DEBUG 打印 --- 
            # --- <<< 开始使用 Plotly 绘图 >>> --- 
            logger.info("开始使用 Plotly 绘制 Nowcast vs 实际值图表...")
            fig = go.Figure()

            columns_plotted = [] # Track which columns are actually added
            # Add Nowcast trace (Line + Markers)
            if nowcast_display_name in comparison_df.columns and comparison_df[nowcast_display_name].notna().any():
                fig.add_trace(go.Scatter(
                    x=comparison_df.index,
                    y=comparison_df[nowcast_display_name],
                    mode='lines+markers',
                    name=nowcast_display_name,
                    line=dict(color='blue'),
                    marker=dict(size=5),
                    hovertemplate=
                    f'<b>日期</b>: %{{x|%Y/%m/%d}}<br>' +
                    f'<b>{nowcast_display_name}</b>: %{{y:.2f}}<extra></extra>' # <extra></extra> removes trace info
                ))
                columns_plotted.append(nowcast_display_name)

            # Add Actual trace (Markers only) - 直接使用后端对齐后的数据
            if target_display_name in comparison_df.columns and comparison_df[target_display_name].notna().any():
                # 筛选出非空的实际值数据点用于绘图
                actual_plot_data = comparison_df[target_display_name].dropna()

                if not actual_plot_data.empty: #确保筛选后仍有数据
                    fig.add_trace(go.Scatter(
                        x=actual_plot_data.index, # 使用对齐后的数据索引
                        y=actual_plot_data.values, # 使用对齐后的数据值
                        mode='markers',
                        name=target_display_name,
                        marker=dict(color='red', size=7),
                        hovertemplate=
                        f'<b>日期</b>: %{{x|%Y/%m/%d}}<br>' +
                        f'<b>{target_display_name}</b>: %{{y:.2f}}<extra></extra>'
                    ))
                    columns_plotted.append(target_display_name)
                else:
                    logger.info(f"'{target_display_name}' 列数据为空，将不会在图表中显示红点。")
            else:
                if target_display_name not in columns_plotted: # Avoid double warning if column missing entirely
                     st.caption(f"提示: '{target_display_name}' 列数据无效或为空，将不会在图表中显示。")

            if columns_plotted:
                # --- Generate Quarterly Ticks --- 
                quarterly_ticks = []
                quarterly_labels = []
                min_date_for_ticks = None
                max_date_for_ticks = None
                try:
                    if comparison_df.index.is_monotonic_increasing or comparison_df.index.is_monotonic_decreasing:
                         min_date_for_ticks = comparison_df.index.min()
                         max_date_for_ticks = comparison_df.index.max()
                    
                    if min_date_for_ticks and max_date_for_ticks:
                        # Generate quarterly dates spanning the data range
                        # Ensure start date is the first day of its quarter
                        start_quarter = pd.Timestamp(min_date_for_ticks.year, 3 * ((min_date_for_ticks.month - 1) // 3) + 1, 1)
                        
                        quarterly_dates = pd.date_range(start=start_quarter, end=max_date_for_ticks, freq='QS') # QS for Quarter Start
                        
                        # Filter out quarters that might be slightly outside the actual data range after generation
                        quarterly_dates = quarterly_dates[(quarterly_dates >= min_date_for_ticks) & (quarterly_dates <= max_date_for_ticks) | \
                                                          (quarterly_dates.to_period('Q') >= min_date_for_ticks.to_period('Q')) & \
                                                          (quarterly_dates.to_period('Q') <= max_date_for_ticks.to_period('Q'))]
                        
                        # If the first data point is not a quarter start, add it to ensure the first tick is sensible
                        if not quarterly_dates.empty and quarterly_dates[0] > min_date_for_ticks and min_date_for_ticks.to_period('Q') < quarterly_dates[0].to_period('Q'):
                            quarterly_dates = quarterly_dates.insert(0, pd.Timestamp(min_date_for_ticks.year, 3 * ((min_date_for_ticks.month - 1) // 3) + 1, 1))

                        # Ensure unique ticks
                        quarterly_dates = quarterly_dates.unique()

                        quarterly_ticks = quarterly_dates.tolist()
                        # Format labels as YYYYQn
                        quarterly_labels = [f"{dt.year}Q{(dt.month - 1) // 3 + 1}" for dt in quarterly_dates]
                        logger.info(f"生成季度刻度完成，共 {len(quarterly_ticks)} 个刻度: {quarterly_labels}")
                    else:
                        logger.info("最小/最大日期无效，无法生成季度刻度。")
                        
                except Exception as e_qticks:
                    logger.warning(f"生成季度 X 轴刻度失败: {e_qticks}")
                    quarterly_ticks = None # Fallback
                    quarterly_labels = None
                # --- End Generate Quarterly Ticks --- 
                    
                # Update layout
                fig.update_layout(
                    title=f'月度 {nowcast_display_name} vs. {target_display_name}',
                    xaxis_title="日期",
                    yaxis_title="(%)",
                    # --- 修改 X 轴刻度和图例位置 --- 
                    xaxis=dict(
                        tickmode='array' if quarterly_ticks and quarterly_labels else 'auto', # Use array mode if ticks generated
                        tickvals=quarterly_ticks if quarterly_ticks and quarterly_labels else None,
                        ticktext=quarterly_labels if quarterly_ticks and quarterly_labels else None,
                        tickangle=0 # Set to 0 for horizontal, or 45 if labels overlap
                        # tickformat='%Y/%m', # Remove this if using tickvals/ticktext
                    ),
                    legend=dict(
                        orientation="h", # Horizontal legend
                        yanchor="bottom",
                        y=-0.4, # Position legend below x-axis title and ticks
                        xanchor="center",
                        x=0.5
                    ),
                    # --- 结束修改 ---\
                    hovermode='x unified',
                    height=600, # Increased height for legend space and ticks
                    margin=dict(t=50, b=150, l=50, r=50) # Adjusted margins (t=top, b=bottom, l=left, r=right)
                )

                # Add validation period shading
                try:
                    val_start_dt = pd.to_datetime(val_start) if pd.notna(val_start) and val_start != 'N/A' else None
                    val_end_dt = pd.to_datetime(val_end) if pd.notna(val_end) and val_end != 'N/A' else None
                    if val_start_dt and val_end_dt:
                        fig.add_vrect(
                            x0=str(val_start_dt), x1=str(val_end_dt),
                            fillcolor="yellow", opacity=0.2, 
                            layer="below", line_width=0,
                            # Remove annotation text from vrect to avoid cluttering legend
                        )
                    else:
                        logger.warning("验证期开始或结束日期无效，无法标黄。")
                except Exception as e_vrect:
                    st.warning(f"标记验证期时出错: {e_vrect}")
                    logger.error(f"标记验证期时出错: {e_vrect}")

                # Display Plotly chart
                st.plotly_chart(fig, use_container_width=True)
                logger.info("Plotly 图表显示成功。")
                # --- <<< 结束使用 Plotly 绘图 >>> --- 
                
                # --- 下载按钮代码：直接使用后端对齐后的数据 --- 
                try:
                    # 直接使用对齐后的数据，无需额外处理
                    csv_data = comparison_df.to_csv(index=True).encode('utf-8-sig')
                    st.download_button(
                        label="数据下载", 
                        data=csv_data,
                        file_name=f"nowcast_vs_{target_variable_name_for_plot}_aligned.csv",
                        mime="text/csv",
                    )
                except Exception as e:
                    st.error(f"生成下载文件时出错: {e}")
                    
            else: 
                st.warning("处理后无可绘制的数据列。检查 nowcast_aligned 和 y_test_aligned 是否存在于元数据中，并且包含有效数值。")
                st.dataframe(comparison_df, use_container_width=True)

        else:
            st.info("未能生成 Nowcast 与实际值的对比数据 (可能为空或对齐失败)。")
    else:
        reason = []
        if nowcast_series is None: reason.append("'nowcast_aligned' (预测值序列) 未在元数据中找到或为None")
        if actual_series is None: reason.append("'y_test_aligned' (实际值序列) 未在元数据中找到或为None")
        st.warning(f"无法显示对比图。原因: { '; '.join(reason) if reason else '未知（序列可能为None）'}。请确保生成 .pkl 的脚本已包含这些有效的 Pandas Series 数据。")

    # --- "详细分析结果" - 直接显示内容 ---
    # 原 with st.expander("详细分析结果", expanded=False):
    st.markdown("**PCA结果分析**")
    pca_results = metadata.get('pca_results_df')
    # 修正因子数量获取，与训练模块的元数据键匹配
    k = metadata.get('best_params', {}).get('k_factors', metadata.get('n_factors', 0))
    if not isinstance(k, int) or k <= 0:
        if not isinstance(k, int) or k <= 0:
            logger.warning("无法确定最终因子数 k，将尝试从 PCA 数据推断。")
            k = len(pca_results.index) if pca_results is not None and isinstance(pca_results, pd.DataFrame) else 0
    
    if pca_results is not None and isinstance(pca_results, pd.DataFrame) and not pca_results.empty:
        pca_df_display = pca_results.head(k if k > 0 else len(pca_results.index)).copy()
        if '主成分 (PC)' in pca_df_display.columns:
            pca_df_display = pca_df_display.drop(columns=['主成分 (PC)'])
        pca_df_display.insert(0, '主成分 (PC)', [f"PC{i+1}" for i in range(len(pca_df_display.index))])
        if not isinstance(pca_df_display.index, pd.RangeIndex):
            pca_df_display = pca_df_display.reset_index()
            if 'index' in pca_df_display.columns:
                pca_df_display = pca_df_display.rename(columns={'index': 'Original Index'})
        pca_df_display = pca_df_display.rename(columns={
            '解释方差 (%)': '解释方差(%)',
            '累计解释方差 (%)': '累计解释方差(%)',
            '特征值 (Eigenvalue)': '特征值(Eigenvalue)'
        })
        st.dataframe(pca_df_display, use_container_width=True)
    else:
        st.write("未找到 PCA 结果。")
    
    st.markdown("--- ")
    st.markdown("**R² 分析**")
    
    # 创建两列布局显示R²分析表格
    r2_col1, r2_col2 = st.columns(2)
    
    with r2_col1:
        st.markdown("**整体 R² (按行业)**")
        industry_r2 = metadata.get('industry_r2_results')
        if industry_r2 is not None and isinstance(industry_r2, pd.Series) and not industry_r2.empty:
            st.dataframe(industry_r2.to_frame(name="Industry R2 (All Factors)"), use_container_width=True)
            # --- 添加解释 ---
            st.caption("附注：衡量所有因子共同解释该行业内所有变量整体变动的百分比。计算方式为对行业内各变量分别对所有因子进行OLS回归后，汇总各变量的总平方和(TSS)与残差平方和(RSS)，计算 Pooled R² = 1 - (Sum(RSS) / Sum(TSS))。")
        else:
            st.write("未找到行业整体 R² 数据。")
         
    with r2_col2:
        st.markdown("**因子对行业 Pooled R²**")
        factor_industry_r2 = metadata.get('factor_industry_r2_results')
        if factor_industry_r2 and isinstance(factor_industry_r2, dict):
            try:
                factor_industry_df = pd.DataFrame(factor_industry_r2)
                st.dataframe(factor_industry_df, use_container_width=True)
                # --- 添加解释 ---
                st.caption("附注：衡量单个因子解释该行业内所有变量整体变动的百分比。计算方式为对行业内各变量分别对单个因子进行OLS回归后，汇总TSS与RSS，计算 Pooled R² = 1 - (Sum(RSS) / Sum(TSS))。")
            except ValueError as ve:
                st.warning(f"因子对行业 Pooled R² 数据格式错误，无法转换为DataFrame: {ve}")
                logger.warning(f"Error converting factor_industry_r2 to DataFrame: {factor_industry_r2}")
            except Exception as e:
                st.error(f"显示因子对行业 Pooled R² 时发生未知错误: {e}")
        elif factor_industry_r2 is not None: # It's not a dict or it's an empty dict but not None
            st.write("因子对行业 Pooled R² 数据格式不正确或为空。")
        else:
            st.write("未找到因子对行业 Pooled R² 数据。")

    st.markdown("---") # Add a separator
    # --- 移除标题1 ---
    # st.markdown("**因子载荷分析 (Factor Loadings)**")
    
    factor_loadings_df = metadata.get('factor_loadings_df') # Assuming this key exists

    if factor_loadings_df is not None and isinstance(factor_loadings_df, pd.DataFrame) and not factor_loadings_df.empty:
        # --- 移除标题2 ---
        # st.markdown("###### 因子载荷热力图")
        
        # --- 修改热力图逻辑以实现 因子在X轴, 聚类变量在Y轴 ---
        data_for_clustering = factor_loadings_df.copy() # 变量为行，因子为列
        variable_names_original = data_for_clustering.index.tolist()
        factor_names_original = data_for_clustering.columns.tolist()
        
        # --- 初始化 y_labels_heatmap 并改进聚类逻辑中的赋值 ---
        y_labels_heatmap = variable_names_original # 默认使用原始顺序
        clustering_performed_successfully = False

        # 1. 对变量进行聚类 (如果变量多于1个)
        if data_for_clustering.shape[0] > 1:
            try:
                from scipy.cluster import hierarchy as sch
                linked = sch.linkage(data_for_clustering.values, method='ward', metric='euclidean')
                dendro = sch.dendrogram(linked, no_plot=True)
                clustered_indices = dendro['leaves']
                data_for_clustering = data_for_clustering.iloc[clustered_indices, :]
                y_labels_heatmap = data_for_clustering.index.tolist() # 聚类成功后更新
                clustering_performed_successfully = True
                logger.info("因子载荷热力图：变量聚类成功。")
            except Exception as e_cluster_heatmap:
                st.warning(f"因子载荷热力图的变量聚类失败: {e_cluster_heatmap}. 将按原始顺序显示变量。")
                logger.warning(f"因子载荷热力图的变量聚类失败: {e_cluster_heatmap}")
                # y_labels_heatmap 保持为之前初始化的 variable_names_original
        else:
            logger.info("因子载荷热力图：只有一个变量，跳过聚类。")
            # y_labels_heatmap 保持为之前初始化的 variable_names_original

        # 2. 准备绘图数据 (z值, x轴标签, y轴标签)
        z_values = data_for_clustering.values # (num_clustered_vars, num_factors)
        x_labels_heatmap = factor_names_original # 因子名作为X轴

        fig_heatmap = go.Figure(data=go.Heatmap(
            z=z_values,
            x=x_labels_heatmap,
            y=y_labels_heatmap, # <--- 确保这里使用的是聚类后的 y_labels_heatmap
            colorscale='RdBu_r',
            zmid=0,
            colorbar=dict(title='载荷值'),
            xgap=1,
            ygap=1,
            hovertemplate=(
                "变量 (Variable): %{y}<br>" +
                "因子 (Factor): %{x}<br>" +
                "载荷值 (Loading): %{z:.4f}<extra></extra>"
            )
        ))

        # Annotate heatmap cells
        annotations = []
        for i, var_name in enumerate(y_labels_heatmap): # 遍历变量 (Y轴)
            for j, factor_name in enumerate(x_labels_heatmap): # 遍历因子 (X轴)
                val = z_values[i][j]
                annotations.append(
                    go.layout.Annotation(
                        text=f"{val:.2f}",
                        x=factor_name,  # X轴是因子
                        y=var_name,     # Y轴是变量
                        xref='x1',
                        yref='y1',
                        showarrow=False,
                        font=dict(color='white' if abs(val) > 0.5 else 'black')
                    )
                )
            
        fig_heatmap.update_layout(
            title="因子载荷热力图", # <--- 修改标题
            xaxis_title="因子 (Factors)",
            yaxis_title="变量 (Predictors)",
            yaxis=dict( # 确保Y轴按聚类顺序显示
                type='category',
                categoryorder='array', # 强制使用 categoryarray 的顺序
                categoryarray=y_labels_heatmap # <--- 再次确认这里使用的是聚类后的 y_labels_heatmap
            ),
            height=max(600, len(y_labels_heatmap) * 35 + 200),  # 增加高度
            # --- 修改宽度、边距，并将X轴移到顶部 -- -
            width=max(1000, len(x_labels_heatmap) * 100 + max(200, max(len(name) for name in y_labels_heatmap)*8 if y_labels_heatmap else 200) + 50),  # 增加宽度
            margin=dict(l=max(200, max(len(name) for name in y_labels_heatmap)*8 if y_labels_heatmap else 200), r=50, t=100, b=200),  # 增加边距
            annotations=annotations,
            xaxis=dict(
                side='top',       # 将X轴移到顶部
                tickangle=-45    # 保持X轴标签旋转角度
            )
        )
            
        # 使用居中显示的容器
        heatmap_col1, heatmap_col2, heatmap_col3 = st.columns([1, 8, 1])
        with heatmap_col2:
            st.plotly_chart(fig_heatmap, use_container_width=True)

        # Download button for factor loadings data
        try:
            # --- 下载原始（未转置）数据 ---
            csv_loadings = factor_loadings_df.to_csv(index=True).encode('utf-8-sig') # utf-8-sig for Excel compatibility
            st.download_button(
                label="数据下载",
                data=csv_loadings,
                file_name="factor_loadings.csv",
                mime="text/csv",
            )
        except Exception as e_csv_loadings:
            st.error(f"生成因子载荷下载文件时出错: {e_csv_loadings}")
        
        # --- 因子时间序列图展示 ---
        st.markdown("---")
        st.markdown("**因子时间序列演变图**")
        
        # 获取因子时间序列数据
        factor_series_data = metadata.get('factor_series')
        
        if factor_series_data is not None and isinstance(factor_series_data, pd.DataFrame) and not factor_series_data.empty:
            factor_names = factor_series_data.columns.tolist()
            num_factors = len(factor_names)
            
            if num_factors > 0:
                # 确定每行显示的图表数量
                if CONFIG_AVAILABLE:
                    cols_per_row = VisualizationDefaults.FACTOR_PLOT_COLS_EVEN if num_factors % 2 == 0 else VisualizationDefaults.FACTOR_PLOT_COLS_ODD
                else:
                    cols_per_row = 2 if num_factors % 2 == 0 else 3
                
                # 计算需要的行数
                num_rows = (num_factors + cols_per_row - 1) // cols_per_row
                
                # 为每个因子创建时间序列图
                for row in range(num_rows):
                    # 创建列布局
                    cols = st.columns(cols_per_row)
                    
                    for col_idx in range(cols_per_row):
                        factor_idx = row * cols_per_row + col_idx
                        
                        if factor_idx < num_factors:
                            factor_name = factor_names[factor_idx]
                            
                            with cols[col_idx]:
                                # 创建单个因子的时间序列图
                                factor_data = factor_series_data[factor_name].dropna()
                                
                                if not factor_data.empty:
                                    fig_factor = go.Figure()
                                    
                                    fig_factor.add_trace(go.Scatter(
                                        x=factor_data.index,
                                        y=factor_data.values,
                                        mode='lines+markers',
                                        name=factor_name,
                                        line=dict(width=2),
                                        marker=dict(size=4),
                                        hovertemplate=(
                                            f"日期: %{{x|%Y/%m/%d}}<br>" +
                                            f"{factor_name}: %{{y:.4f}}<extra></extra>"
                                        )
                                    ))
                                    
                                    fig_factor.update_layout(
                                        title=f"{factor_name}",
                                        xaxis_title="日期",
                                        yaxis_title="因子值",
                                        height=400,
                                        margin=dict(t=60, b=80, l=60, r=30),
                                        showlegend=False,  # 隐藏图例以节省空间
                                        hovermode='x unified'
                                    )
                                    
                                    # 添加零轴线
                                    fig_factor.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.5)
                                    
                                    st.plotly_chart(fig_factor, use_container_width=True)
                                else:
                                    st.warning(f"{factor_name}数据为空，无法绘制图表。")
                
                # 提供所有因子数据的统一下载
                
                try:
                    all_factors_csv = factor_series_data.to_csv(index=True).encode('utf-8-sig')
                    st.download_button(
                        label="数据下载",
                        data=all_factors_csv,
                        file_name="所有因子时间序列.csv",
                        mime="text/csv",
                        key="download_all_factors"
                    )
                except Exception as e_all_factors:
                    st.error(f"生成所有因子下载文件时出错: {e_all_factors}")
            else:
                st.write("未找到有效的因子数据。")
        else:
            st.write("未在元数据中找到因子时间序列数据。预期的键名: 'factor_series'。")
    
    elif factor_loadings_df is not None and not isinstance(factor_loadings_df, pd.DataFrame):
        st.warning("因子载荷数据 (factor_loadings_df) 存在但不是有效的 DataFrame 格式。")
    else:
        st.write("未在元数据中找到因子载荷数据 (expected key: 'factor_loadings_df')。")


