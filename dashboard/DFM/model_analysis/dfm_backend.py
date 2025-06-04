import pandas as pd
import joblib
import pickle
import io
import logging
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error
from scipy.cluster import hierarchy as sch

logger = logging.getLogger(__name__)

# 🔥 已移除 _calculate_revised_monthly_metrics 函数
# UI模块现在直接使用训练模块中通过 calculate_metrics_with_lagged_target 计算的标准指标
# 这确保了指标计算方法的一致性

# @st.cache_data(ttl=3600) # Streamlit caching is UI-specific, remove from backend
def load_dfm_results_from_uploads(loaded_model_object, loaded_metadata_object):
    """
    Receives already loaded DFM model and metadata objects.
    The actual loading (joblib.load, pickle.load) is expected to have happened
    before calling this function (e.g., in the UI layer with caching).
    """
    model = loaded_model_object
    metadata = loaded_metadata_object
    load_errors = []

    if model is None:
        error_msg = "接收到的 DFM 模型对象为 None。"
        logger.warning(error_msg)
        load_errors.append(error_msg)
    else:
        logger.info("成功接收 DFM 模型对象。")

    if metadata is None:
        error_msg = "接收到的 DFM 元数据对象为 None。"
        logger.warning(error_msg)
        load_errors.append(error_msg)
    else:
        logger.info("成功接收 DFM 元数据对象。")
        
    # --- 🔥 修复：直接使用训练模块已计算的标准指标，不再重新计算 ---
    logger.info("直接使用训练模块已计算的标准指标...")
    
    # 检查元数据中是否包含训练模块计算的指标
    standard_metric_keys = ['is_rmse', 'oos_rmse', 'is_mae', 'oos_mae', 'is_hit_rate', 'oos_hit_rate']
    has_standard_metrics = all(key in metadata for key in standard_metric_keys)
    
    if has_standard_metrics:
        logger.info("发现训练模块计算的标准指标，直接使用...")
        # 直接使用训练模块的标准指标，保持键名一致以供UI使用
        metadata['revised_is_hr'] = metadata.get('is_hit_rate')
        metadata['revised_is_rmse'] = metadata.get('is_rmse')
        metadata['revised_is_mae'] = metadata.get('is_mae')
        metadata['revised_oos_hr'] = metadata.get('oos_hit_rate')
        metadata['revised_oos_rmse'] = metadata.get('oos_rmse')
        metadata['revised_oos_mae'] = metadata.get('oos_mae')
        
        logger.info(f"已加载标准指标: IS胜率={metadata['revised_is_hr']}, OOS胜率={metadata['revised_oos_hr']}")
        logger.info(f"                 IS_RMSE={metadata['revised_is_rmse']}, OOS_RMSE={metadata['revised_oos_rmse']}")
    else:
        logger.warning("未在元数据中找到训练模块计算的标准指标，将尝试从原始数据重新计算...")
        # 回退到原始计算逻辑（保留原代码作为备份）
        nowcast_aligned_full = metadata.get('nowcast_aligned') 
        y_test_aligned_full = metadata.get('y_test_aligned')   

        def parse_date_robust(date_input):
            if pd.isna(date_input) or date_input == 'N/A' or date_input is None:
                return None
            try:
                return pd.to_datetime(date_input)
            except Exception as e_parse:
                logger.warning(f"日期解析失败 '{date_input}': {e_parse}")
                return None

        train_start_date = parse_date_robust(metadata.get('training_start_date'))
        train_end_date = parse_date_robust(metadata.get('train_end_date')) or parse_date_robust(metadata.get('training_end_date'))
        val_start_date = parse_date_robust(metadata.get('validation_start_date'))
        val_end_date = parse_date_robust(metadata.get('validation_end_date'))

        # 初始化指标键
        metadata['revised_is_hr'] = None
        metadata['revised_is_rmse'] = None
        metadata['revised_is_mae'] = None
        metadata['revised_oos_hr'] = None
        metadata['revised_oos_rmse'] = None
        metadata['revised_oos_mae'] = None

        if nowcast_aligned_full is not None and y_test_aligned_full is not None:
            try:
                # 确保索引是 datetime 类型
                if not isinstance(nowcast_aligned_full.index, pd.DatetimeIndex):
                    nowcast_aligned_full.index = pd.to_datetime(nowcast_aligned_full.index)
                if not isinstance(y_test_aligned_full.index, pd.DatetimeIndex):
                    y_test_aligned_full.index = pd.to_datetime(y_test_aligned_full.index)

                # 样本内 (训练期) 指标
                if train_start_date and train_end_date:
                    nowcast_is = nowcast_aligned_full.loc[train_start_date:train_end_date].copy()
                    actual_is = y_test_aligned_full.loc[train_start_date:train_end_date].copy()
                    
                    # 🔥 回退：由于没有标准指标，使用简化计算
                    logger.warning("使用简化指标计算作为回退方案（不推荐）")
                    hr_is, rmse_is, mae_is = None, None, None
                    metadata['revised_is_hr'] = hr_is
                    metadata['revised_is_rmse'] = rmse_is
                    metadata['revised_is_mae'] = mae_is
                else:
                    logger.warning(f"训练期起止日期未在元数据中完全提供。无法计算样本内指标。")
                    
                # 样本外 (验证期) 指标
                if val_start_date and val_end_date:
                    nowcast_oos = nowcast_aligned_full.loc[val_start_date:val_end_date].copy()
                    actual_oos = y_test_aligned_full.loc[val_start_date:val_end_date].copy()

                    # 🔥 回退：由于没有标准指标，使用简化计算
                    logger.warning("使用简化指标计算作为回退方案（不推荐）")
                    hr_oos, rmse_oos, mae_oos = None, None, None
                    metadata['revised_oos_hr'] = hr_oos
                    metadata['revised_oos_rmse'] = rmse_oos
                    metadata['revised_oos_mae'] = mae_oos
                else:
                    logger.warning(f"验证期起止日期未在元数据中完全提供。无法计算样本外指标。")
            except Exception as e_calc:
                error_msg = f"在分割数据或调用指标计算时出错: {e_calc}"
                logger.error(error_msg, exc_info=True)
                load_errors.append(error_msg)
        else:
            error_msg = "'nowcast_aligned' 或 'y_test_aligned' 未在元数据中找到。无法计算指标。"
            logger.error(error_msg)
            load_errors.append(error_msg)
        
    return model, metadata, load_errors

# Placeholder for future DFM data processing logic related to the third (data) file
def process_dfm_data(uploaded_data_file):
    """
    Processes the uploaded DFM-related data file (Excel/CSV).
    Placeholder: Implement actual data processing logic here.
    """
    df = None
    processing_errors = []
    if uploaded_data_file is not None:
        try:
            file_name = uploaded_data_file.name
            if file_name.endswith('.csv'):
                df = pd.read_csv(uploaded_data_file)
            elif file_name.endswith(('.xls', '.xlsx')):
                df = pd.read_excel(uploaded_data_file)
            else:
                processing_errors.append(f"不支持的文件类型: {file_name}。请上传 CSV 或 Excel 文件。")
            
            if df is not None:
                logger.info(f"成功处理数据文件 '{file_name}'。")
                # Placeholder for further processing if needed
        except Exception as e:
            error_msg = f"处理数据文件 '{uploaded_data_file.name}' 时出错: {e}"
            logger.error(error_msg)
            processing_errors.append(error_msg)
    else:
        processing_errors.append("未提供 DFM 相关数据文件。")
        
    return df, processing_errors 

def create_aligned_nowcast_target_table(nowcast_series, target_series, nowcast_col_name="Nowcast值", target_col_name="实际值"):
    """
    🔥 修复实际值对齐逻辑：实际值发布日期需要向前偏移一个月对应预测值
    🔥 新增：每月只显示最后一个值，去除重复
    
    对齐规则：
    1. Nowcast 数据每月只保留最后一个值（月末）
    2. Target 数据索引向前偏移一个月，因为实际值是滞后发布的
       例如：4月25日发布的7.7%应该对应3月的预测值
    3. 按调整后的月份Period匹配 Nowcast 和 Target 数据
    """
    logger.info("开始创建对齐的 Nowcast vs Target 表格...")
    
    if nowcast_series is None or target_series is None: 
        return pd.DataFrame(columns=[nowcast_col_name, target_col_name])
    if not isinstance(nowcast_series, pd.Series): 
        nowcast_series = pd.Series(nowcast_series)
    if not isinstance(target_series, pd.Series): 
        target_series = pd.Series(target_series)
    if not isinstance(nowcast_series.index, pd.DatetimeIndex): 
        nowcast_series.index = pd.to_datetime(nowcast_series.index)
    if not isinstance(target_series.index, pd.DatetimeIndex): 
        target_series.index = pd.to_datetime(target_series.index)

    # --- 去除基于索引的重复项 ---
    if not nowcast_series.index.is_unique:
        logger.info(f"原始 nowcast_series 索引包含 {nowcast_series.index.duplicated().sum()} 个重复项。将保留每个重复日期的最后一条记录。")
        nowcast_series = nowcast_series[~nowcast_series.index.duplicated(keep='last')]
    
    if not target_series.index.is_unique:
        logger.info(f"原始 target_series 索引包含 {target_series.index.duplicated().sum()} 个重复项。将保留每个重复日期的最后一条记录。")
        target_series = target_series[~target_series.index.duplicated(keep='last')]

    # 🔥 修正：保持 nowcast 的周度数据，不进行月度聚合
    logger.info(f"原始 nowcast_series 包含 {len(nowcast_series)} 个数据点")
    
    # 保持完整的周度 nowcast 数据
    nowcast_aligned_base = nowcast_series.dropna()
    nowcast_aligned_base.name = nowcast_col_name
    
    logger.info(f"保持周度频率，nowcast 包含 {len(nowcast_aligned_base)} 个数据点")

    # 2. 🔥 修复：实际值发布日期向前偏移一个月对应预测值
    target_df = target_series.dropna().to_frame(target_col_name)
    target_df = target_df.sort_index(ascending=False) # 按日期索引降序排序
    # 🔥 关键修复：将索引减1个月，因为实际值是滞后发布的
    # 例如：4月25日发布的数据对应3月的经济情况
    target_df['TargetPeriod'] = (target_df.index - pd.DateOffset(months=1)).to_period('M')
    
    logger.info(f"在对 TargetPeriod 去重前, target_df 有 {len(target_df)} 行。TargetPeriod 的唯一计数为: {target_df['TargetPeriod'].nunique()}")
    target_df = target_df.drop_duplicates(subset=['TargetPeriod'], keep='first') # 基于 TargetPeriod 去重
    logger.info(f"在对 TargetPeriod 去重后, target_df 有 {len(target_df)} 行。")

    # 3. 🔥 修正：将月度实际值对应到该月最后一周的nowcast
    # 保持所有周度nowcast数据
    final_aligned_table = nowcast_aligned_base.to_frame()
    final_aligned_table[target_col_name] = np.nan
    
    # 为每个目标期间找到对应月份的最后一周
    for target_period, target_value in zip(target_df['TargetPeriod'], target_df[target_col_name]):
        # 找到该月份内的所有nowcast数据点
        month_nowcast = final_aligned_table[final_aligned_table.index.to_period('M') == target_period]
        
        if not month_nowcast.empty:
            # 选择该月份的最后一个数据点（最后一周）
            last_week_date = month_nowcast.index.max()
            final_aligned_table.loc[last_week_date, target_col_name] = target_value
            logger.info(f"月度实际值 {target_value} 对应到 {target_period} 的最后一周: {last_week_date}")
    
    final_aligned_table = final_aligned_table.sort_index()
    
    logger.info(f"成功创建对齐表格，包含 {len(final_aligned_table)} 行数据。")
    logger.info(f"对齐表格包含实际值的数据点: {final_aligned_table[target_col_name].notna().sum()} 个")
    
    return final_aligned_table

def perform_loadings_clustering(loadings_df: pd.DataFrame, cluster_vars: bool = True):
    """
    对因子载荷矩阵进行变量聚类计算。
    
    Args:
        loadings_df: 包含因子载荷的 DataFrame (原始形式：变量为行，因子为列)
        cluster_vars: 是否对变量进行聚类排序
    
    Returns:
        tuple: (clustered_loadings_df, variable_order, clustering_success)
            - clustered_loadings_df: 聚类后的载荷矩阵
            - variable_order: 聚类后的变量顺序列表
            - clustering_success: 聚类是否成功的布尔值
    """
    if not isinstance(loadings_df, pd.DataFrame) or loadings_df.empty:
        logger.warning("无法进行聚类：提供的载荷数据无效。")
        return loadings_df, loadings_df.index.tolist() if not loadings_df.empty else [], False

    data_for_clustering = loadings_df.copy()  # 变量是行
    variable_names_original = data_for_clustering.index.tolist()
    clustering_success = False

    if not cluster_vars:
        logger.info("跳过变量聚类，使用原始顺序。")
        return data_for_clustering, variable_names_original, False

    # 对变量进行聚类 (如果变量多于1个)
    if data_for_clustering.shape[0] > 1:
        try:
            linked = sch.linkage(data_for_clustering.values, method='ward', metric='euclidean')
            dendro = sch.dendrogram(linked, no_plot=True)
            clustered_indices = dendro['leaves']
            data_for_clustering = data_for_clustering.iloc[clustered_indices, :]
            variable_order = data_for_clustering.index.tolist()  # 聚类成功后更新
            clustering_success = True
            logger.info("因子载荷变量聚类成功。")
        except Exception as e_cluster:
            logger.warning(f"因子载荷变量聚类失败: {e_cluster}. 将按原始顺序显示变量。")
            variable_order = variable_names_original
            data_for_clustering = loadings_df.copy()  # 恢复原始数据
    else:
        logger.info("只有一个变量，跳过聚类。")
        variable_order = variable_names_original

    return data_for_clustering, variable_order, clustering_success 