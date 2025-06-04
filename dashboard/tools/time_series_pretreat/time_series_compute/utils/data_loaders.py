import pandas as pd
import streamlit as st # For st.error/st.warning
import io

def process_uploaded_data(uploaded_file, time_index_col=0, skiprows=None, header=0):
    """识别上传的文件格式并加载数据, 尝试使用指定列作为时间索引。
    
    Args:
        uploaded_file: Streamlit UploadedFile object.
        time_index_col: 尝试用作时间索引的列号 (基于最终列名)。默认为0。
        skiprows: 要跳过的行列表 (传递给 pandas read 函数)。
        header: 包含表头的行号 (传递给 pandas read 函数)。
    """
    try:
        file_name = uploaded_file.name
        bytes_data = uploaded_file.getvalue()
        data = None

        # `index_col_arg` 将在读取后用于set_index，如果 time_index_col 是数字
        # 或者直接在 read_csv/excel 中使用，如果 Pandas 对其与 header 的组合处理符合预期
        # 当前的策略是让 pandas 在读取时就尝试设置索引（如果 time_index_col 是列名或可接受的数字）
        # 但更稳妥的做法可能是先读取数据，然后再根据 time_index_col 设置索引，尤其是当 time_index_col 是数字时。
        # 为了与之前逻辑（强制第一列）兼容，并引入skiprows/header，我们让pandas先读取，然后再处理索引

        read_params = {
            "skiprows": skiprows,
            "header": header,
            # "index_col": time_index_col, # 暂时不在这里指定index_col，在读取后处理更灵活
        }

        if file_name.endswith('.csv'):
            # 使用 io.BytesIO 是好的，因为它适用于多种来源的字节流
            data = pd.read_csv(io.BytesIO(bytes_data), **read_params)
        elif file_name.endswith('.xlsx'):
             data = pd.read_excel(io.BytesIO(bytes_data), engine='openpyxl', **read_params)
        else:
            st.error("不支持的文件格式。请上传 CSV 或 Excel 文件。")
            return None

        if data is None or data.empty:
             st.error(f"未能成功读取数据，或文件 '{file_name}' 为空 (在应用skiprows/header后)。")
             return None # 返回None，让UI层知道是空的或读取失败

        # --- 尝试将指定列设置为索引并转换为 DatetimeIndex --- #
        if isinstance(time_index_col, int):
            if 0 <= time_index_col < len(data.columns):
                target_index_column_name = data.columns[time_index_col]
            else:
                st.error(f"指定的 time_index_col ({time_index_col}) 超出了列范围。将尝试使用第一列。")
                target_index_column_name = data.columns[0] if len(data.columns) > 0 else None
        elif isinstance(time_index_col, str):
            if time_index_col in data.columns:
                target_index_column_name = time_index_col
            else:
                st.error(f"指定的 time_index_col 名 '{time_index_col}' 不存在。将尝试使用第一列。")
                target_index_column_name = data.columns[0] if len(data.columns) > 0 else None
        else:
            st.warning("time_index_col 参数类型无效，将尝试使用第一列。")
            target_index_column_name = data.columns[0] if len(data.columns) > 0 else None

        if target_index_column_name:
            try:
                # 确保目标列在设置索引前转换为datetime
                data[target_index_column_name] = pd.to_datetime(data[target_index_column_name], errors='coerce')
                data = data.set_index(target_index_column_name)
                # 移除因日期转换失败而产生的 NaT 索引行
                data = data[data.index.notna()]
                if data.empty:
                    st.error(f"在将列 '{target_index_column_name}' 设置为时间索引并移除无效日期后，数据为空。")
                    return None # 返回None，表示处理后数据为空
            except Exception as e_set_index:
                st.error(f"将列 '{target_index_column_name}' 设置为时间索引时出错: {e_set_index}")
                # 如果设置索引失败，可以考虑是否返回原始 data 或 None
                # 为了安全，返回 None，因为时间索引是后续操作的关键
                return None
        elif not isinstance(data.index, pd.DatetimeIndex): # 如果没有目标列，但当前索引也不是DatetimeIndex
            # 这是一个回退或警告情况，表明未能按预期设置时间索引
            st.warning("未能找到或设置有效的时间索引列。数据的第一列未被自动识别为日期时间。")
        # 如果 target_index_column_name 为 None，且当前索引已经是 DatetimeIndex，则什么也不做，使用现有索引


        # --- 旧的索引处理逻辑 (现已整合到上面的 set_index 逻辑中) ---
        # if not isinstance(data.index, pd.DatetimeIndex):
        #     original_index_name = data.index.name # 保存原始名字
        #     try:
        #         converted_index = pd.to_datetime(data.index, errors='coerce')
        #         if converted_index.isnull().all():
        #              pass
        #         else:
        #              data.index = converted_index
        #              data.index.name = original_index_name
        #     except Exception as e:
        #          print(f"索引转换为日期时间时出错: {e}") 
        #          pass
        # --- 结束索引处理 ---

        print(f"数据加载成功，共 {len(data)} 条记录。索引类型: {type(data.index)}")

        if data.isnull().values.any():
            st.warning("**注意：数据中包含缺失值 (NaN)。** 部分计算可能会受影响或产生 NaN 结果。")

        return data
    except Exception as e:
        print(f"加载数据时出错: {e}") # 用于服务器端日志
        st.error(f"加载或解析文件时出错: {e}") # 显示给用户
        return None 