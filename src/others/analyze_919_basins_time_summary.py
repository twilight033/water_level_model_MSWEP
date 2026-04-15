"""
快速分析919个验证流域的时间范围
直接从runtime_basin_time_ranges_simple.csv读取
"""

import pandas as pd
import numpy as np
from datetime import datetime

def main():
    print("="*80)
    print("919个验证流域的CAMELSH数据时间范围分析")
    print("="*80)
    
    # 读取已有的per-basin时间范围文件
    print("\n[1] 读取per-basin时间范围文件...")
    df = pd.read_csv("runtime_basin_time_ranges_simple.csv", dtype={'basin_id': str})
    
    # 转换时间列为datetime
    time_cols = ['train_start', 'train_end', 'valid_start', 'valid_end', 'test_start', 'test_end']
    for col in time_cols:
        df[col] = pd.to_datetime(df[col])
    
    print(f"  流域数量: {len(df)}")
    
    # 读取919个验证流域列表
    print("\n[2] 读取919个验证流域列表...")
    with open("validated_basin_ids_919_simple.txt", 'r') as f:
        validated_basins = [line.strip() for line in f if line.strip()]
    print(f"  验证流域数量: {len(validated_basins)}")
    
    # 过滤出919个验证流域的数据
    df_validated = df[df['basin_id'].isin(validated_basins)].copy()
    print(f"  匹配的流域数量: {len(df_validated)}")
    
    if len(df_validated) < len(validated_basins):
        missing = set(validated_basins) - set(df_validated['basin_id'])
        print(f"  警告: 有 {len(missing)} 个流域在时间范围文件中不存在")
        if len(missing) <= 10:
            print(f"    缺失流域: {list(missing)}")
    
    # 统计整体时间范围
    print(f"\n" + "="*80)
    print("时间范围统计")
    print("="*80)
    
    # 计算每个流域的完整数据范围（从train_start到test_end）
    df_validated['full_start'] = df_validated['train_start']
    df_validated['full_end'] = df_validated['test_end']
    df_validated['full_days'] = (df_validated['full_end'] - df_validated['full_start']).dt.days + 1
    df_validated['full_years'] = df_validated['full_days'] / 365.25
    
    # 计算训练集时间范围
    df_validated['train_days'] = (df_validated['train_end'] - df_validated['train_start']).dt.days + 1
    df_validated['train_years'] = df_validated['train_days'] / 365.25
    
    print(f"\n【完整数据范围（Train + Valid + Test）】")
    print(f"  所有流域最早开始: {df_validated['full_start'].min()}")
    print(f"  所有流域最晚开始: {df_validated['full_start'].max()}")
    print(f"  所有流域最早结束: {df_validated['full_end'].min()}")
    print(f"  所有流域最晚结束: {df_validated['full_end'].max()}")
    print(f"  平均时间跨度: {df_validated['full_years'].mean():.1f} 年")
    print(f"  中位数跨度: {df_validated['full_years'].median():.1f} 年")
    print(f"  最短跨度: {df_validated['full_years'].min():.1f} 年")
    print(f"  最长跨度: {df_validated['full_years'].max():.1f} 年")
    
    print(f"\n【训练集时间范围】")
    print(f"  所有流域最早开始: {df_validated['train_start'].min()}")
    print(f"  所有流域最晚开始: {df_validated['train_start'].max()}")
    print(f"  所有流域最早结束: {df_validated['train_end'].min()}")
    print(f"  所有流域最晚结束: {df_validated['train_end'].max()}")
    print(f"  平均时间跨度: {df_validated['train_years'].mean():.1f} 年")
    print(f"  中位数跨度: {df_validated['train_years'].median():.1f} 年")
    
    print(f"\n【验证集时间范围】")
    print(f"  所有流域最早开始: {df_validated['valid_start'].min()}")
    print(f"  所有流域最晚开始: {df_validated['valid_start'].max()}")
    print(f"  所有流域最早结束: {df_validated['valid_end'].min()}")
    print(f"  所有流域最晚结束: {df_validated['valid_end'].max()}")
    
    print(f"\n【测试集时间范围】")
    print(f"  所有流域最早开始: {df_validated['test_start'].min()}")
    print(f"  所有流域最晚开始: {df_validated['test_start'].max()}")
    print(f"  所有流域最早结束: {df_validated['test_end'].min()}")
    print(f"  所有流域最晚结束: {df_validated['test_end'].max()}")
    
    # 按开始年份统计
    print(f"\n【按开始年份统计】")
    df_validated['start_year'] = df_validated['full_start'].dt.year
    year_counts = df_validated['start_year'].value_counts().sort_index()
    print(f"  年份范围: {year_counts.index.min()} - {year_counts.index.max()}")
    print(f"\n  各年份流域数量:")
    for year, count in year_counts.items():
        print(f"    {year}: {count} 个流域 ({count/len(df_validated)*100:.1f}%)")
    
    # 按结束年份统计
    print(f"\n【按结束年份统计】")
    df_validated['end_year'] = df_validated['full_end'].dt.year
    year_counts_end = df_validated['end_year'].value_counts().sort_index()
    print(f"  年份范围: {year_counts_end.index.min()} - {year_counts_end.index.max()}")
    print(f"\n  各年份流域数量:")
    for year, count in year_counts_end.items():
        print(f"    {year}: {count} 个流域 ({count/len(df_validated)*100:.1f}%)")
    
    # 时间跨度分布
    print(f"\n【时间跨度分布】")
    bins = [0, 5, 10, 15, 20]
    labels = ['<5年', '5-10年', '10-15年', '>=15年']
    df_validated['years_bin'] = pd.cut(df_validated['full_years'], bins=bins, labels=labels, include_lowest=True)
    span_dist = df_validated['years_bin'].value_counts().sort_index()
    for label, count in span_dist.items():
        print(f"  {label}: {count} 个流域 ({count/len(df_validated)*100:.1f}%)")
    
    # 数据充足性分析
    print(f"\n【数据充足性分析】")
    long_data = df_validated[df_validated['full_years'] >= 10]
    print(f"  >= 10年数据: {len(long_data)} 个流域 ({len(long_data)/len(df_validated)*100:.1f}%)")
    
    long_data_15 = df_validated[df_validated['full_years'] >= 15]
    print(f"  >= 15年数据: {len(long_data_15)} 个流域 ({len(long_data_15)/len(df_validated)*100:.1f}%)")
    
    recent_data = df_validated[df_validated['full_end'] >= pd.Timestamp('2024-01-01')]
    print(f"  数据到2024年: {len(recent_data)} 个流域 ({len(recent_data)/len(df_validated)*100:.1f}%)")
    
    # 典型流域示例
    print(f"\n【典型流域示例】")
    print(f"\n时间跨度最长的10个流域:")
    for idx, row in df_validated.nlargest(10, 'full_years').iterrows():
        print(f"  {row['basin_id']}: {row['full_start'].strftime('%Y-%m-%d')} 到 "
              f"{row['full_end'].strftime('%Y-%m-%d')} ({row['full_years']:.1f}年)")
    
    print(f"\n时间跨度最短的10个流域:")
    for idx, row in df_validated.nsmallest(10, 'full_years').iterrows():
        print(f"  {row['basin_id']}: {row['full_start'].strftime('%Y-%m-%d')} 到 "
              f"{row['full_end'].strftime('%Y-%m-%d')} ({row['full_years']:.1f}年)")
    
    print(f"\n开始时间最早的10个流域:")
    for idx, row in df_validated.nsmallest(10, 'full_start').iterrows():
        print(f"  {row['basin_id']}: {row['full_start'].strftime('%Y-%m-%d')} 到 "
              f"{row['full_end'].strftime('%Y-%m-%d')} ({row['full_years']:.1f}年)")
    
    print(f"\n结束时间最晚的10个流域:")
    for idx, row in df_validated.nlargest(10, 'full_end').iterrows():
        print(f"  {row['basin_id']}: {row['full_start'].strftime('%Y-%m-%d')} 到 "
              f"{row['full_end'].strftime('%Y-%m-%d')} ({row['full_years']:.1f}年)")
    
    # 保存详细报告
    print(f"\n[3] 保存详细报告...")
    output_report = "validated_919_basins_timerange_summary.txt"
    with open(output_report, 'w', encoding='utf-8') as f:
        f.write("="*80 + "\n")
        f.write("919个验证流域的CAMELSH数据时间范围摘要\n")
        f.write("="*80 + "\n\n")
        f.write(f"生成时间: {datetime.now()}\n")
        f.write(f"数据来源: runtime_basin_time_ranges_simple.csv\n")
        f.write(f"流域数量: {len(df_validated)}\n\n")
        
        f.write("="*80 + "\n")
        f.write("完整数据范围（Train + Valid + Test）\n")
        f.write("="*80 + "\n")
        f.write(f"所有流域最早开始: {df_validated['full_start'].min()}\n")
        f.write(f"所有流域最晚开始: {df_validated['full_start'].max()}\n")
        f.write(f"所有流域最早结束: {df_validated['full_end'].min()}\n")
        f.write(f"所有流域最晚结束: {df_validated['full_end'].max()}\n")
        f.write(f"平均时间跨度: {df_validated['full_years'].mean():.1f} 年\n")
        f.write(f"中位数跨度: {df_validated['full_years'].median():.1f} 年\n")
        f.write(f"最短跨度: {df_validated['full_years'].min():.1f} 年\n")
        f.write(f"最长跨度: {df_validated['full_years'].max():.1f} 年\n\n")
        
        f.write("="*80 + "\n")
        f.write("训练集时间范围\n")
        f.write("="*80 + "\n")
        f.write(f"所有流域最早开始: {df_validated['train_start'].min()}\n")
        f.write(f"所有流域最晚开始: {df_validated['train_start'].max()}\n")
        f.write(f"所有流域最早结束: {df_validated['train_end'].min()}\n")
        f.write(f"所有流域最晚结束: {df_validated['train_end'].max()}\n")
        f.write(f"平均时间跨度: {df_validated['train_years'].mean():.1f} 年\n")
        f.write(f"中位数跨度: {df_validated['train_years'].median():.1f} 年\n\n")
        
        f.write("="*80 + "\n")
        f.write("按开始年份统计\n")
        f.write("="*80 + "\n")
        for year, count in year_counts.items():
            f.write(f"{year}: {count} 个流域 ({count/len(df_validated)*100:.1f}%)\n")
        
        f.write("\n" + "="*80 + "\n")
        f.write("时间跨度分布\n")
        f.write("="*80 + "\n")
        for label, count in span_dist.items():
            f.write(f"{label}: {count} 个流域 ({count/len(df_validated)*100:.1f}%)\n")
        
        f.write("\n" + "="*80 + "\n")
        f.write("数据充足性\n")
        f.write("="*80 + "\n")
        f.write(f">= 10年数据: {len(long_data)} 个流域 ({len(long_data)/len(df_validated)*100:.1f}%)\n")
        f.write(f">= 15年数据: {len(long_data_15)} 个流域 ({len(long_data_15)/len(df_validated)*100:.1f}%)\n")
        f.write(f"数据到2024年: {len(recent_data)} 个流域 ({len(recent_data)/len(df_validated)*100:.1f}%)\n\n")
        
        f.write("="*80 + "\n")
        f.write("典型流域示例\n")
        f.write("="*80 + "\n\n")
        
        f.write("时间跨度最长的10个流域:\n")
        f.write("-"*80 + "\n")
        for idx, row in df_validated.nlargest(10, 'full_years').iterrows():
            f.write(f"{row['basin_id']}: {row['full_start'].strftime('%Y-%m-%d')} 到 "
                   f"{row['full_end'].strftime('%Y-%m-%d')} ({row['full_years']:.1f}年)\n")
        
        f.write("\n时间跨度最短的10个流域:\n")
        f.write("-"*80 + "\n")
        for idx, row in df_validated.nsmallest(10, 'full_years').iterrows():
            f.write(f"{row['basin_id']}: {row['full_start'].strftime('%Y-%m-%d')} 到 "
                   f"{row['full_end'].strftime('%Y-%m-%d')} ({row['full_years']:.1f}年)\n")
        
        f.write("\n开始时间最早的10个流域:\n")
        f.write("-"*80 + "\n")
        for idx, row in df_validated.nsmallest(10, 'full_start').iterrows():
            f.write(f"{row['basin_id']}: {row['full_start'].strftime('%Y-%m-%d')} 到 "
                   f"{row['full_end'].strftime('%Y-%m-%d')} ({row['full_years']:.1f}年)\n")
        
        f.write("\n结束时间最晚的10个流域:\n")
        f.write("-"*80 + "\n")
        for idx, row in df_validated.nlargest(10, 'full_end').iterrows():
            f.write(f"{row['basin_id']}: {row['full_start'].strftime('%Y-%m-%d')} 到 "
                   f"{row['full_end'].strftime('%Y-%m-%d')} ({row['full_years']:.1f}年)\n")
    
    print(f"  已保存: {output_report}")
    
    # 保存CSV
    output_csv = "validated_919_basins_timerange.csv"
    df_validated[['basin_id', 'train_start', 'train_end', 'valid_start', 'valid_end', 
                  'test_start', 'test_end', 'full_start', 'full_end', 'full_days', 
                  'full_years', 'train_days', 'train_years']].to_csv(output_csv, index=False)
    print(f"  已保存: {output_csv}")
    
    print(f"\n" + "="*80)
    print("完成！")
    print("="*80)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n用户中断操作")
    except Exception as e:
        print(f"\n发生错误: {e}")
        import traceback
        traceback.print_exc()
