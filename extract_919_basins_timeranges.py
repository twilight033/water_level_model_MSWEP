"""
为919个验证流域提取per-basin时间范围
"""

import os
import sys
import pandas as pd
import numpy as np
from datetime import datetime
from pathlib import Path

# 添加项目根目录到路径
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from config import CAMELSH_DATA_PATH, TRAIN_RATIO, VALID_RATIO, TEST_RATIO
from improved_camelsh_reader import ImprovedCAMELSHReader


def load_validated_basins(file_path="validated_basin_ids_919_simple.txt"):
    """读取验证通过的流域列表"""
    with open(file_path, 'r') as f:
        basins = [line.strip() for line in f if line.strip()]
    return basins


def extract_basin_time_range(flow_data, waterlevel_data, basin_id):
    """提取单个流域的有效数据时间范围"""
    try:
        if basin_id not in flow_data.columns or basin_id not in waterlevel_data.columns:
            return None
        
        flow_series = flow_data[basin_id]
        waterlevel_series = waterlevel_data[basin_id]
        
        # 找到同时有效的数据范围
        valid_mask = (~pd.isna(flow_series)) & (~pd.isna(waterlevel_series))
        valid_data = flow_series[valid_mask]
        
        if len(valid_data) == 0:
            return None
        
        # 有效数据的时间范围
        first_valid = valid_data.index.min()
        last_valid = valid_data.index.max()
        total_hours = len(valid_data)
        
        # 按比例划分训练/验证/测试集
        train_end_idx = int(total_hours * TRAIN_RATIO)
        valid_end_idx = train_end_idx + int(total_hours * VALID_RATIO)
        
        # 转换为时间戳
        valid_times = valid_data.index
        train_start = valid_times[0]
        train_end = valid_times[min(train_end_idx - 1, len(valid_times) - 1)]
        valid_start = valid_times[min(train_end_idx, len(valid_times) - 1)]
        valid_end = valid_times[min(valid_end_idx - 1, len(valid_times) - 1)]
        test_start = valid_times[min(valid_end_idx, len(valid_times) - 1)]
        test_end = valid_times[-1]
        
        return {
            'basin_id': basin_id,
            'first_valid': first_valid,
            'last_valid': last_valid,
            'total_valid_hours': total_hours,
            'train_start': train_start,
            'train_end': train_end,
            'valid_start': valid_start,
            'valid_end': valid_end,
            'test_start': test_start,
            'test_end': test_end
        }
    except Exception as e:
        print(f"[ERROR] Basin {basin_id}: {e}")
        return None


def main():
    print("="*80)
    print("为919个验证流域提取时间范围")
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
        print(f"  正在生成CSV文件...")
        
        # 运行导出脚本
        os.system("uv run python qualifiers_fetcher/export_camelsh_data.py")
        
        if not flow_file.exists() or not waterlevel_file.exists():
            print(f"  错误: 生成失败！")
            return
    
    print(f"  正在读取径流数据...")
    print(f"  (这可能需要几分钟，请耐心等待...)")
    
    # 只读取需要的流域列
    flow_df = pd.read_csv(flow_file, index_col=0, parse_dates=True, usecols=['time'] + basins)
    print(f"    径流数据形状: {flow_df.shape}")
    
    print(f"  正在读取水位数据...")
    waterlevel_df = pd.read_csv(waterlevel_file, index_col=0, parse_dates=True, usecols=['time'] + basins)
    print(f"    水位数据形状: {waterlevel_df.shape}")
    
    # 提取每个流域的时间范围
    print(f"\n[3] 提取每个流域的时间范围...")
    print(f"  使用时间划分比例: Train {TRAIN_RATIO:.0%}, Valid {VALID_RATIO:.0%}, Test {TEST_RATIO:.0%}")
    
    results = []
    failed = []
    
    for i, basin_id in enumerate(basins, 1):
        if i % 50 == 0 or i == 1:
            print(f"  进度: {i}/{len(basins)} ({i/len(basins)*100:.1f}%)")
        
        result = extract_basin_time_range(flow_df, waterlevel_df, basin_id)
        if result:
            results.append(result)
        else:
            failed.append(basin_id)
    
    print(f"\n  提取完成!")
    print(f"  成功: {len(results)} 个")
    print(f"  失败: {len(failed)} 个")
    
    if failed:
        print(f"\n  失败的流域: {failed[:10]}")
        if len(failed) > 10:
            print(f"  ... 还有 {len(failed)-10} 个")
    
    # 转换为DataFrame
    df = pd.DataFrame(results)
    
    # 保存结果
    print(f"\n[4] 保存结果...")
    
    # 详细CSV
    output_csv = "validated_919_basins_time_ranges.csv"
    df.to_csv(output_csv, index=False)
    print(f"  已保存详细数据: {output_csv}")
    
    # 简化CSV（只包含train/valid/test时间）
    output_simple = "validated_919_basins_time_ranges_simple.csv"
    df[['basin_id', 'train_start', 'train_end', 'valid_start', 'valid_end', 
        'test_start', 'test_end']].to_csv(output_simple, index=False)
    print(f"  已保存简化数据: {output_simple}")
    
    # 统计报告
    output_report = "validated_919_basins_time_ranges_report.txt"
    with open(output_report, 'w', encoding='utf-8') as f:
        f.write("="*80 + "\n")
        f.write("919个验证流域的时间范围报告\n")
        f.write("="*80 + "\n\n")
        f.write(f"生成时间: {datetime.now()}\n")
        f.write(f"流域数量: {len(results)}\n")
        f.write(f"时间划分比例: Train {TRAIN_RATIO:.0%}, Valid {VALID_RATIO:.0%}, Test {TEST_RATIO:.0%}\n\n")
        
        f.write("="*80 + "\n")
        f.write("完整数据范围统计\n")
        f.write("="*80 + "\n")
        f.write(f"最早开始: {df['first_valid'].min()}\n")
        f.write(f"最晚开始: {df['first_valid'].max()}\n")
        f.write(f"最早结束: {df['last_valid'].min()}\n")
        f.write(f"最晚结束: {df['last_valid'].max()}\n")
        f.write(f"平均数据点数: {df['total_valid_hours'].mean():.0f}\n")
        f.write(f"中位数数据点数: {df['total_valid_hours'].median():.0f}\n\n")
        
        # 计算时间跨度
        df['days'] = (df['last_valid'] - df['first_valid']).dt.days + 1
        df['years'] = df['days'] / 365.25
        
        f.write("="*80 + "\n")
        f.write("时间跨度统计\n")
        f.write("="*80 + "\n")
        f.write(f"平均: {df['years'].mean():.1f} 年\n")
        f.write(f"中位数: {df['years'].median():.1f} 年\n")
        f.write(f"最短: {df['years'].min():.1f} 年\n")
        f.write(f"最长: {df['years'].max():.1f} 年\n\n")
        
        # 按年份统计
        df['start_year'] = df['first_valid'].dt.year
        year_counts = df['start_year'].value_counts().sort_index()
        
        f.write("="*80 + "\n")
        f.write("按开始年份统计\n")
        f.write("="*80 + "\n")
        for year, count in year_counts.items():
            f.write(f"{year}: {count} 个流域 ({count/len(df)*100:.1f}%)\n")
        
        # 时间跨度分布
        bins = [0, 5, 10, 15, 20]
        labels = ['<5年', '5-10年', '10-15年', '>=15年']
        df['years_bin'] = pd.cut(df['years'], bins=bins, labels=labels, include_lowest=True)
        span_dist = df['years_bin'].value_counts().sort_index()
        
        f.write("\n" + "="*80 + "\n")
        f.write("时间跨度分布\n")
        f.write("="*80 + "\n")
        for label, count in span_dist.items():
            f.write(f"{label}: {count} 个流域 ({count/len(df)*100:.1f}%)\n")
        
        # 示例流域
        f.write("\n" + "="*80 + "\n")
        f.write("典型流域示例\n")
        f.write("="*80 + "\n\n")
        
        f.write("时间跨度最长的10个流域:\n")
        f.write("-"*80 + "\n")
        for idx, row in df.nlargest(10, 'years').iterrows():
            f.write(f"{row['basin_id']}: {row['first_valid'].strftime('%Y-%m-%d')} 到 "
                   f"{row['last_valid'].strftime('%Y-%m-%d')} ({row['years']:.1f}年)\n")
        
        f.write("\n时间跨度最短的10个流域:\n")
        f.write("-"*80 + "\n")
        for idx, row in df.nsmallest(10, 'years').iterrows():
            f.write(f"{row['basin_id']}: {row['first_valid'].strftime('%Y-%m-%d')} 到 "
                   f"{row['last_valid'].strftime('%Y-%m-%d')} ({row['years']:.1f}年)\n")
        
        # 完整列表
        f.write("\n" + "="*80 + "\n")
        f.write("所有流域时间范围\n")
        f.write("="*80 + "\n\n")
        for idx, row in df.iterrows():
            f.write(f"{row['basin_id']}: {row['first_valid'].strftime('%Y-%m-%d')} 到 "
                   f"{row['last_valid'].strftime('%Y-%m-%d')} ({row['years']:.1f}年)\n")
            f.write(f"  Train: {row['train_start'].strftime('%Y-%m-%d')} 到 {row['train_end'].strftime('%Y-%m-%d')}\n")
            f.write(f"  Valid: {row['valid_start'].strftime('%Y-%m-%d')} 到 {row['valid_end'].strftime('%Y-%m-%d')}\n")
            f.write(f"  Test: {row['test_start'].strftime('%Y-%m-%d')} 到 {row['test_end'].strftime('%Y-%m-%d')}\n\n")
    
    print(f"  已保存报告: {output_report}")
    
    # 显示摘要
    print(f"\n" + "="*80)
    print("完成！")
    print(f"="*80)
    print(f"\n时间范围摘要:")
    print(f"  最早开始: {df['first_valid'].min()}")
    print(f"  最晚开始: {df['first_valid'].max()}")
    print(f"  最早结束: {df['last_valid'].min()}")
    print(f"  最晚结束: {df['last_valid'].max()}")
    df['years'] = ((df['last_valid'] - df['first_valid']).dt.days + 1) / 365.25
    print(f"  平均时间跨度: {df['years'].mean():.1f} 年")
    print(f"  中位数跨度: {df['years'].median():.1f} 年")
    
    print(f"\n生成的文件:")
    print(f"  1. {output_csv} - 详细时间范围")
    print(f"  2. {output_simple} - 简化时间范围（Train/Valid/Test）")
    print(f"  3. {output_report} - 详细报告")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n用户中断操作")
    except Exception as e:
        print(f"\n发生错误: {e}")
        import traceback
        traceback.print_exc()
