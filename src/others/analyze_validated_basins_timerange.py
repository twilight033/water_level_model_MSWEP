"""
分析919个验证流域的CAMELSH数据时间范围
"""

import pandas as pd
import numpy as np
from datetime import datetime
from pathlib import Path
import sys

# 添加项目根目录到路径
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# 不再需要导入这些，直接读取CSV文件


def load_validated_basins(file_path="validated_basin_ids_919_simple.txt"):
    """读取验证通过的流域列表"""
    with open(file_path, 'r') as f:
        basins = [line.strip() for line in f if line.strip()]
    return basins


def analyze_basin_timerange(flow_df, waterlevel_df, basin_id):
    """分析单个流域的时间范围"""
    try:
        # 检查流域是否在数据中
        if basin_id not in flow_df.columns or basin_id not in waterlevel_df.columns:
            return None
        
        flow_series = flow_df[basin_id]
        waterlevel_series = waterlevel_df[basin_id]
        
        # 找到非NaN数据的时间范围
        flow_valid = flow_series[~np.isnan(flow_series)]
        waterlevel_valid = waterlevel_series[~np.isnan(waterlevel_series)]
        
        if len(flow_valid) == 0 or len(waterlevel_valid) == 0:
            return None
        
        # 计算各自的时间范围
        flow_start = flow_valid.index.min()
        flow_end = flow_valid.index.max()
        waterlevel_start = waterlevel_valid.index.min()
        waterlevel_end = waterlevel_valid.index.max()
        
        # 计算交集时间范围（同时有有效数据）
        common_start = max(flow_start, waterlevel_start)
        common_end = min(flow_end, waterlevel_end)
        
        # 统计有效数据点数
        flow_count = len(flow_valid)
        waterlevel_count = len(waterlevel_valid)
        
        # 计算交集范围内的有效数据
        common_mask = (flow_series.index >= common_start) & (flow_series.index <= common_end)
        both_valid = (~np.isnan(flow_series[common_mask])) & (~np.isnan(waterlevel_series[common_mask]))
        both_valid_count = both_valid.sum()
        
        return {
            'basin_id': basin_id,
            'flow_start': flow_start,
            'flow_end': flow_end,
            'flow_count': flow_count,
            'waterlevel_start': waterlevel_start,
            'waterlevel_end': waterlevel_end,
            'waterlevel_count': waterlevel_count,
            'common_start': common_start,
            'common_end': common_end,
            'common_days': (common_end - common_start).days + 1,
            'both_valid_count': both_valid_count
        }
        
    except Exception as e:
        print(f"[ERROR] Basin {basin_id}: {e}")
        return None


def main():
    print("="*80)
    print("分析919个验证流域的CAMELSH数据时间范围")
    print("="*80)
    
    # 读取验证流域列表
    print("\n[1] 读取验证流域列表...")
    basins = load_validated_basins()
    print(f"  流域数量: {len(basins)}")
    
    # 加载CAMELSH导出的CSV数据
    print("\n[2] 加载CAMELSH数据集...")
    flow_file = project_root / "camelsh_exported" / "flow_hourly.csv"
    waterlevel_file = project_root / "camelsh_exported" / "waterlevel_hourly.csv"
    
    if not flow_file.exists() or not waterlevel_file.exists():
        print(f"  错误: CSV文件不存在！")
        print(f"  请先运行: uv run python qualifiers_fetcher/export_camelsh_data.py")
        return
    
    print(f"  正在读取径流数据: {flow_file}")
    flow_df = pd.read_csv(flow_file, index_col=0, parse_dates=True)
    print(f"    形状: {flow_df.shape}")
    
    print(f"  正在读取水位数据: {waterlevel_file}")
    waterlevel_df = pd.read_csv(waterlevel_file, index_col=0, parse_dates=True)
    print(f"    形状: {waterlevel_df.shape}")
    
    # 时间范围
    default_range = [flow_df.index.min().strftime('%Y-%m-%d'), flow_df.index.max().strftime('%Y-%m-%d')]
    print(f"  数据集整体时间范围: {default_range}")
    
    # 分析每个流域的时间范围
    print(f"\n[3] 分析每个流域的实际数据时间范围...")
    print(f"  这可能需要一些时间...")
    
    results = []
    failed = []
    
    for i, basin_id in enumerate(basins, 1):
        if i % 50 == 0 or i == 1:
            print(f"  进度: {i}/{len(basins)} ({i/len(basins)*100:.1f}%)")
        
        result = analyze_basin_timerange(flow_df, waterlevel_df, basin_id)
        if result:
            results.append(result)
        else:
            failed.append(basin_id)
    
    print(f"\n  分析完成!")
    print(f"  成功: {len(results)} 个")
    print(f"  失败: {len(failed)} 个")
    
    if failed:
        print(f"\n  失败的流域: {failed[:10]}")
        if len(failed) > 10:
            print(f"  ... 还有 {len(failed)-10} 个")
    
    # 转换为DataFrame
    df = pd.DataFrame(results)
    
    # 统计分析
    print(f"\n" + "="*80)
    print("时间范围统计分析")
    print("="*80)
    
    print(f"\n【径流数据时间范围】")
    print(f"  最早开始: {df['flow_start'].min()}")
    print(f"  最晚开始: {df['flow_start'].max()}")
    print(f"  最早结束: {df['flow_end'].min()}")
    print(f"  最晚结束: {df['flow_end'].max()}")
    print(f"  平均数据点数: {df['flow_count'].mean():.0f}")
    
    print(f"\n【水位数据时间范围】")
    print(f"  最早开始: {df['waterlevel_start'].min()}")
    print(f"  最晚开始: {df['waterlevel_start'].max()}")
    print(f"  最早结束: {df['waterlevel_end'].min()}")
    print(f"  最晚结束: {df['waterlevel_end'].max()}")
    print(f"  平均数据点数: {df['waterlevel_count'].mean():.0f}")
    
    print(f"\n【交集时间范围（同时有有效数据）】")
    print(f"  最早开始: {df['common_start'].min()}")
    print(f"  最晚开始: {df['common_start'].max()}")
    print(f"  最早结束: {df['common_end'].min()}")
    print(f"  最晚结束: {df['common_end'].max()}")
    print(f"  平均天数: {df['common_days'].mean():.0f}")
    print(f"  中位数天数: {df['common_days'].median():.0f}")
    print(f"  最短天数: {df['common_days'].min()}")
    print(f"  最长天数: {df['common_days'].max()}")
    
    # 按年份统计
    print(f"\n【按开始年份统计】")
    df['start_year'] = df['common_start'].dt.year
    year_counts = df['start_year'].value_counts().sort_index()
    print(f"  年份范围: {year_counts.index.min()} - {year_counts.index.max()}")
    print(f"\n  主要年份分布:")
    for year, count in year_counts.head(10).items():
        print(f"    {year}: {count} 个流域 ({count/len(df)*100:.1f}%)")
    
    print(f"\n【按结束年份统计】")
    df['end_year'] = df['common_end'].dt.year
    year_counts_end = df['end_year'].value_counts().sort_index()
    print(f"  年份范围: {year_counts_end.index.min()} - {year_counts_end.index.max()}")
    print(f"\n  主要年份分布:")
    for year, count in year_counts_end.tail(10).items():
        print(f"    {year}: {count} 个流域 ({count/len(df)*100:.1f}%)")
    
    # 时间跨度分布
    print(f"\n【时间跨度分布】")
    df['years'] = df['common_days'] / 365.25
    bins = [0, 5, 10, 20, 30, 40, 50]
    labels = ['<5年', '5-10年', '10-20年', '20-30年', '30-40年', '40-50年']
    df['years_bin'] = pd.cut(df['years'], bins=bins, labels=labels, include_lowest=True)
    span_dist = df['years_bin'].value_counts().sort_index()
    for label, count in span_dist.items():
        print(f"  {label}: {count} 个流域 ({count/len(df)*100:.1f}%)")
    
    # 保存结果
    print(f"\n[4] 保存结果...")
    
    # 详细数据
    output_csv = "validated_basins_timerange_919.csv"
    df_save = df.copy()
    df_save['flow_start'] = df_save['flow_start'].dt.strftime('%Y-%m-%d')
    df_save['flow_end'] = df_save['flow_end'].dt.strftime('%Y-%m-%d')
    df_save['waterlevel_start'] = df_save['waterlevel_start'].dt.strftime('%Y-%m-%d')
    df_save['waterlevel_end'] = df_save['waterlevel_end'].dt.strftime('%Y-%m-%d')
    df_save['common_start'] = df_save['common_start'].dt.strftime('%Y-%m-%d')
    df_save['common_end'] = df_save['common_end'].dt.strftime('%Y-%m-%d')
    df_save.to_csv(output_csv, index=False)
    print(f"  已保存详细数据: {output_csv}")
    
    # 摘要报告
    output_report = "validated_basins_timerange_919_report.txt"
    with open(output_report, 'w', encoding='utf-8') as f:
        f.write("="*80 + "\n")
        f.write("919个验证流域的CAMELSH数据时间范围报告\n")
        f.write("="*80 + "\n\n")
        f.write(f"生成时间: {datetime.now()}\n")
        f.write(f"流域数量: {len(results)}\n")
        f.write(f"数据集整体范围: {default_range}\n\n")
        
        f.write("="*80 + "\n")
        f.write("径流数据时间范围\n")
        f.write("="*80 + "\n")
        f.write(f"最早开始: {df['flow_start'].min()}\n")
        f.write(f"最晚开始: {df['flow_start'].max()}\n")
        f.write(f"最早结束: {df['flow_end'].min()}\n")
        f.write(f"最晚结束: {df['flow_end'].max()}\n")
        f.write(f"平均数据点数: {df['flow_count'].mean():.0f}\n\n")
        
        f.write("="*80 + "\n")
        f.write("水位数据时间范围\n")
        f.write("="*80 + "\n")
        f.write(f"最早开始: {df['waterlevel_start'].min()}\n")
        f.write(f"最晚开始: {df['waterlevel_start'].max()}\n")
        f.write(f"最早结束: {df['waterlevel_end'].min()}\n")
        f.write(f"最晚结束: {df['waterlevel_end'].max()}\n")
        f.write(f"平均数据点数: {df['waterlevel_count'].mean():.0f}\n\n")
        
        f.write("="*80 + "\n")
        f.write("交集时间范围（同时有有效数据）\n")
        f.write("="*80 + "\n")
        f.write(f"最早开始: {df['common_start'].min()}\n")
        f.write(f"最晚开始: {df['common_start'].max()}\n")
        f.write(f"最早结束: {df['common_end'].min()}\n")
        f.write(f"最晚结束: {df['common_end'].max()}\n")
        f.write(f"平均天数: {df['common_days'].mean():.0f} 天 ({df['common_days'].mean()/365.25:.1f} 年)\n")
        f.write(f"中位数天数: {df['common_days'].median():.0f} 天 ({df['common_days'].median()/365.25:.1f} 年)\n")
        f.write(f"最短: {df['common_days'].min()} 天 ({df['common_days'].min()/365.25:.1f} 年)\n")
        f.write(f"最长: {df['common_days'].max()} 天 ({df['common_days'].max()/365.25:.1f} 年)\n\n")
        
        f.write("="*80 + "\n")
        f.write("按开始年份统计\n")
        f.write("="*80 + "\n")
        for year, count in df['start_year'].value_counts().sort_index().items():
            f.write(f"{year}: {count} 个流域 ({count/len(df)*100:.1f}%)\n")
        
        f.write("\n" + "="*80 + "\n")
        f.write("按结束年份统计\n")
        f.write("="*80 + "\n")
        for year, count in df['end_year'].value_counts().sort_index().items():
            f.write(f"{year}: {count} 个流域 ({count/len(df)*100:.1f}%)\n")
        
        f.write("\n" + "="*80 + "\n")
        f.write("时间跨度分布\n")
        f.write("="*80 + "\n")
        for label, count in span_dist.items():
            f.write(f"{label}: {count} 个流域 ({count/len(df)*100:.1f}%)\n")
        
        # 添加典型流域示例
        f.write("\n" + "="*80 + "\n")
        f.write("典型流域示例\n")
        f.write("="*80 + "\n\n")
        
        f.write("时间跨度最长的10个流域:\n")
        f.write("-"*80 + "\n")
        for idx, row in df.nlargest(10, 'common_days').iterrows():
            f.write(f"{row['basin_id']}: {row['common_start'].strftime('%Y-%m-%d')} 到 "
                   f"{row['common_end'].strftime('%Y-%m-%d')} ({row['common_days']}天, {row['years']:.1f}年)\n")
        
        f.write("\n时间跨度最短的10个流域:\n")
        f.write("-"*80 + "\n")
        for idx, row in df.nsmallest(10, 'common_days').iterrows():
            f.write(f"{row['basin_id']}: {row['common_start'].strftime('%Y-%m-%d')} 到 "
                   f"{row['common_end'].strftime('%Y-%m-%d')} ({row['common_days']}天, {row['years']:.1f}年)\n")
        
        f.write("\n开始时间最早的10个流域:\n")
        f.write("-"*80 + "\n")
        for idx, row in df.nsmallest(10, 'common_start').iterrows():
            f.write(f"{row['basin_id']}: {row['common_start'].strftime('%Y-%m-%d')} 到 "
                   f"{row['common_end'].strftime('%Y-%m-%d')} ({row['years']:.1f}年)\n")
        
        f.write("\n结束时间最晚的10个流域:\n")
        f.write("-"*80 + "\n")
        for idx, row in df.nlargest(10, 'common_end').iterrows():
            f.write(f"{row['basin_id']}: {row['common_start'].strftime('%Y-%m-%d')} 到 "
                   f"{row['common_end'].strftime('%Y-%m-%d')} ({row['years']:.1f}年)\n")
    
    print(f"  已保存摘要报告: {output_report}")
    
    # 统计摘要
    print(f"\n" + "="*80)
    print("完成！")
    print("="*80)
    print(f"\n关键发现:")
    print(f"  1. 数据集整体范围: {default_range[0]} 到 {default_range[1]}")
    print(f"  2. 流域实际范围: {df['common_start'].min().strftime('%Y-%m-%d')} 到 "
          f"{df['common_end'].max().strftime('%Y-%m-%d')}")
    print(f"  3. 平均时间跨度: {df['common_days'].mean()/365.25:.1f} 年")
    print(f"  4. 中位数跨度: {df['common_days'].median()/365.25:.1f} 年")
    
    # 数据充足性分析
    long_data = df[df['years'] >= 20]
    print(f"\n  数据充足性:")
    print(f"    >= 20年数据: {len(long_data)} 个流域 ({len(long_data)/len(df)*100:.1f}%)")
    
    recent_data = df[df['common_end'] >= pd.Timestamp('2020-01-01')]
    print(f"    数据到2020年后: {len(recent_data)} 个流域 ({len(recent_data)/len(df)*100:.1f}%)")
    
    print(f"\n生成的文件:")
    print(f"  1. {output_csv} - 详细时间范围数据")
    print(f"  2. {output_report} - 摘要报告")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n用户中断操作")
    except Exception as e:
        print(f"\n发生错误: {e}")
        import traceback
        traceback.print_exc()
