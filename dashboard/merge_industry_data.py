import pandas as pd
import os
from datetime import datetime
import glob

def load_industry_data(file_path, start_date='2020-01-01'):
    """Load data from a single industry Excel file"""
    try:
        # 读取文件
        file = pd.ExcelFile(file_path)
        
        # 读取周度数据
        df_weekly = pd.read_excel(file, sheet_name="weekly")
        df_weekly['date'] = pd.to_datetime(df_weekly['date'])
        df_weekly.set_index("date", inplace=True)
        # 过滤2020年后的数据
        df_weekly = df_weekly[df_weekly.index >= start_date]
        
        # 读取月度数据
        df_monthly = pd.read_excel(file, sheet_name="monthly")
        df_monthly['date'] = pd.to_datetime(df_monthly['date'])
        df_monthly.set_index("date", inplace=True)
        # 过滤2020年后的数据
        df_monthly = df_monthly[df_monthly.index >= start_date]
        
        # 获取行业名称
        industry_name = os.path.basename(file_path).replace('_data.xlsx', '')
        
        # 重命名列以包含行业信息
        df_weekly.columns = [f"{industry_name}_{col}" for col in df_weekly.columns]
        df_monthly.columns = [f"{industry_name}_{col}" for col in df_monthly.columns]
        
        # 打印每个行业的数据信息
        print(f"\n{industry_name} 数据信息:")
        print(f"周度数据范围: {df_weekly.index.min()} 到 {df_weekly.index.max()}")
        print(f"周度变量数量: {len(df_weekly.columns)}")
        print(f"月度数据范围: {df_monthly.index.min()} 到 {df_monthly.index.max()}")
        print(f"月度变量数量: {len(df_monthly.columns)}")
        
        return df_weekly, df_monthly
    except Exception as e:
        print(f"Error loading {file_path}: {str(e)}")
        return None, None

def merge_all_industry_data(start_date='2020-01-01'):
    """Merge data from all industry Excel files"""
    # 获取所有行业数据文件
    data_files = glob.glob("*_data.xlsx")
    
    if not data_files:
        print("No industry data files found!")
        return None, None
    
    # 初始化合并后的数据框
    merged_weekly = None
    merged_monthly = None
    
    # 处理每个行业的数据
    for file_path in data_files:
        print(f"\nProcessing {file_path}...")
        df_weekly, df_monthly = load_industry_data(file_path, start_date)
        
        if df_weekly is not None and df_monthly is not None:
            if merged_weekly is None:
                merged_weekly = df_weekly
                merged_monthly = df_monthly
            else:
                merged_weekly = merged_weekly.join(df_weekly, how='outer')
                merged_monthly = merged_monthly.join(df_monthly, how='outer')
    
    # 按日期排序
    if merged_weekly is not None:
        merged_weekly.sort_index(inplace=True)
    if merged_monthly is not None:
        merged_monthly.sort_index(inplace=True)
    
    return merged_weekly, merged_monthly

def save_merged_data(weekly_df, monthly_df):
    """Save merged data to Excel file"""
    if weekly_df is None or monthly_df is None:
        print("No data to save!")
        return
    
    # 创建输出文件名，包含数据时间范围
    start_date = weekly_df.index.min().strftime("%Y%m%d")
    end_date = weekly_df.index.max().strftime("%Y%m%d")
    output_file = f"merged_industry_data_{start_date}_to_{end_date}.xlsx"
    
    # 保存到Excel
    with pd.ExcelWriter(output_file) as writer:
        weekly_df.to_excel(writer, sheet_name='weekly')
        monthly_df.to_excel(writer, sheet_name='monthly')
    
    print(f"\nData saved to {output_file}")
    return output_file

def main():
    print("Starting data merge process...")
    start_date = '2020-01-01'
    print(f"Filtering data from {start_date}")
    
    # 合并所有行业数据
    weekly_df, monthly_df = merge_all_industry_data(start_date)
    
    if weekly_df is not None and monthly_df is not None:
        # 保存合并后的数据
        output_file = save_merged_data(weekly_df, monthly_df)
        
        if output_file:
            print("\nData merge completed successfully!")
            print("\n合并后数据统计:")
            print(f"周度数据范围: {weekly_df.index.min()} 到 {weekly_df.index.max()}")
            print(f"周度变量总数: {len(weekly_df.columns)}")
            print(f"月度数据范围: {monthly_df.index.min()} 到 {monthly_df.index.max()}")
            print(f"月度变量总数: {len(monthly_df.columns)}")
            
            # 显示每个行业的变量数量
            print("\n各行业变量统计:")
            industry_stats = {}
            for col in weekly_df.columns:
                industry = col.split('_')[0]
                if industry not in industry_stats:
                    industry_stats[industry] = {'weekly': 0, 'monthly': 0}
                industry_stats[industry]['weekly'] += 1
            
            for col in monthly_df.columns:
                industry = col.split('_')[0]
                if industry not in industry_stats:
                    industry_stats[industry] = {'weekly': 0, 'monthly': 0}
                industry_stats[industry]['monthly'] += 1
            
            for industry, stats in industry_stats.items():
                print(f"{industry}:")
                print(f"  周度变量: {stats['weekly']}")
                print(f"  月度变量: {stats['monthly']}")
    else:
        print("Failed to merge data!")

if __name__ == "__main__":
    main() 