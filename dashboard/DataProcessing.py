import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import calendar
import os
import warnings
warnings.filterwarnings('ignore')

def load_data(wd, filename,
              week_sheet = "weekly",
              month_sheet = "monthly",
              start = "2020-01-01",
              end = "2025-02-28"):
    
    try:
        # 读取文件
        file_path = filename  # 直接使用完整文件路径
        file = pd.ExcelFile(file_path)

        # 读取周度数据
        df_weekly = pd.read_excel(file, sheet_name = week_sheet) 
        df_weekly['date'] = pd.to_datetime(df_weekly['date'])  
        df_weekly.set_index("date", inplace = True)
        
        # 读取月度数据
        df_monthly = pd.read_excel(file, sheet_name = month_sheet) 
        df_monthly['date'] = pd.to_datetime(df_monthly['date'])
        df_monthly.set_index("date", inplace = True)

        # 截取数据
        df_monthly = df_monthly.loc[start:end]
        df_weekly = df_weekly.loc[start:end]

        # 打印数据信息
        print("\n数据加载信息:")
        print(f"周度数据范围: {df_weekly.index.min()} 到 {df_weekly.index.max()}")
        print(f"周度变量数量: {len(df_weekly.columns)}")
        print(f"月度数据范围: {df_monthly.index.min()} 到 {df_monthly.index.max()}")
        print(f"月度变量数量: {len(df_monthly.columns)}")

        return df_weekly, df_monthly
    except Exception as e:
        print(f"Error loading data: {str(e)}")
        return None, None

def process_weekly_data(df):
    """处理周度数据"""
    try:
        # 检查必要的列是否存在（支持中英文列名）
        date_column = None
        if 'date' in df.columns:
            date_column = 'date'
        elif '日期' in df.columns:
            date_column = '日期'
        else:
            raise ValueError("数据中缺少日期列（date或日期）")
            
        # 将日期列转换为datetime类型
        df[date_column] = pd.to_datetime(df[date_column])
        
        # 检查日期格式
        if df[date_column].isna().any():
            raise ValueError("日期列包含无效的日期格式")
            
        # 设置日期为索引
        df.set_index(date_column, inplace=True)
        
        # 按日期排序
        df.sort_index(inplace=True)
        
        return df
        
    except Exception as e:
        print(f"处理周度数据时出错: {str(e)}")
        return None

def process_monthly_data(df):
    """处理月度数据"""
    try:
        # 检查必要的列是否存在（支持中英文列名）
        date_column = None
        if 'date' in df.columns:
            date_column = 'date'
        elif '日期' in df.columns:
            date_column = '日期'
        else:
            raise ValueError("数据中缺少日期列（date或日期）")
            
        # 将日期列转换为datetime类型
        df[date_column] = pd.to_datetime(df[date_column])
        
        # 检查日期格式
        if df[date_column].isna().any():
            raise ValueError("日期列包含无效的日期格式")
            
        # 设置日期为索引
        df.set_index(date_column, inplace=True)
        
        # 按日期排序
        df.sort_index(inplace=True)
        
        return df
        
    except Exception as e:
        print(f"处理月度数据时出错: {str(e)}")
        return None

def compute_change(df, period = 52):
    """计算同比增长率"""
    # 计算yoy周度增长率
    df_pct = df.apply(lambda x: x.pct_change(period, fill_method = None).mul(100))
    
    # 删除第一年数据
    df_pct = df_pct.loc[str(df_pct.index.year.min() + 1):]
    
    # 打印增长率统计信息
    print("\n增长率统计:")
    print(f"数据范围: {df_pct.index.min()} 到 {df_pct.index.max()}")
    print(f"变量数量: {len(df_pct.columns)}")
    
    return df_pct

def last_friday_of_month(date):
    """计算指定月份最后一个星期五"""
    if isinstance(date, str):
        date_obj = datetime.strptime(str(date), '%Y-%m-%d')
    else:
        date_obj = date.to_pydatetime()
    
    # 获取指定月份的最后一天
    last_day = date_obj.replace(day=calendar.monthrange(date_obj.year, date_obj.month)[1])
    
    # 计算最后一天是星期几
    last_day_weekday = last_day.weekday()
    
    # 计算到最后一个星期五的天数差
    days_diff = (last_day_weekday - 4) % 7
    last_friday = last_day - timedelta(days=days_diff)
    
    return last_friday

def merge_data(df_weekly, df_monthly):
    """合并周度和月度数据"""
    # 合并数据
    df_merged = df_weekly.join(df_monthly, how='left')
    
    # 打印合并后的数据信息
    print("\n数据合并信息:")
    print(f"合并后数据范围: {df_merged.index.min()} 到 {df_merged.index.max()}")
    print(f"合并后变量总数: {len(df_merged.columns)}")
    
    # 统计各行业的变量数量
    industry_stats = {}
    for col in df_merged.columns:
        industry = col.split('_')[0]
        if industry not in industry_stats:
            industry_stats[industry] = 0
        industry_stats[industry] += 1
    
    print("\n各行业变量数量:")
    for industry, count in industry_stats.items():
        print(f"{industry}: {count}个变量")
    
    return df_merged

def main(file_path=None):
    """主函数"""
    try:
        if file_path:
            print(f"\n正在处理文件: {os.path.basename(file_path)}")
            
            # 读取Excel文件
            try:
                df = pd.read_excel(file_path)
                print(f"成功读取文件，包含 {len(df.columns)} 列")
                print(f"列名: {', '.join(df.columns)}")
            except Exception as e:
                print(f"读取Excel文件时出错: {str(e)}")
                return None
            
            # 检查数据是否为空
            if df.empty:
                print("数据为空")
                return None
                
            # 检查必要的列
            if 'date' not in df.columns and '日期' not in df.columns:
                print("错误：数据中缺少日期列（date或日期）")
                return None
            
            # 处理数据
            df = process_weekly_data(df)
            if df is None:
                return None
                
            # 打印数据加载信息
            print("\n数据加载信息:")
            print(f"数据范围: {df.index.min()} 到 {df.index.max()}")
            print(f"变量数量: {len(df.columns)}")
            
            return df
            
    except Exception as e:
        print(f"处理数据时出错: {str(e)}")
        return None

if __name__ == "__main__":
    main()
