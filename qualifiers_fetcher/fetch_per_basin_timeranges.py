"""
根据每个流域的实际训练时间范围获取USGS Qualifiers数据
从runtime_basin_time_ranges_simple.csv读取配置
"""

import sys
from pathlib import Path
import pandas as pd
from datetime import datetime
from tqdm import tqdm
import time

# 添加项目路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(Path(__file__).parent))

from usgs_qualifiers_fetcher import USGSQualifiersFetcher


def load_basin_time_ranges(csv_file="runtime_basin_time_ranges_simple.csv"):
    """
    从CSV文件加载每个流域的时间范围
    
    Parameters
    ----------
    csv_file : str
        CSV文件路径
    
    Returns
    -------
    pd.DataFrame
        包含流域ID和时间范围的DataFrame
    """
    try:
        # 重要：指定basin_id为字符串类型，保留前导零
        df = pd.read_csv(csv_file, dtype={'basin_id': str})
        print(f"成功加载 {len(df)} 个流域的时间配置")
        print(f"\n列: {list(df.columns)}")
        return df
    except FileNotFoundError:
        print(f"错误: 找不到文件 {csv_file}")
        print("请先运行 extract_per_basin_config.py 生成此文件")
        raise


def fetch_qualifiers_per_basin(
    basin_time_df,
    time_period="train",
    output_dir="qualifiers_output_per_basin",
    cache_dir="qualifiers_cache_per_basin",
    max_basins=None
):
    """
    为每个流域使用其独立的时间范围获取qualifiers
    
    Parameters
    ----------
    basin_time_df : pd.DataFrame
        包含流域时间范围的DataFrame
    time_period : str
        使用哪个时间段 ('train', 'valid', 'test', 'all')
    output_dir : str
        输出目录
    cache_dir : str
        缓存目录
    max_basins : int, optional
        最多处理的流域数量（用于测试）
    
    Returns
    -------
    dict
        {basin_id: {'discharge': df, 'gage_height': df}}
    """
    
    # 创建fetcher
    fetcher = USGSQualifiersFetcher(
        output_dir=output_dir,
        cache_dir=cache_dir
    )
    
    # 限制处理的流域数量
    if max_basins:
        basin_time_df = basin_time_df.head(max_basins)
    
    print(f"\n{'='*80}")
    print(f"开始为 {len(basin_time_df)} 个流域获取qualifiers")
    print(f"时间段: {time_period}")
    print(f"{'='*80}\n")
    
    all_qualifiers = {}
    success_count = 0
    fail_count = 0
    
    # 为每个流域独立获取
    for idx, row in tqdm(basin_time_df.iterrows(), total=len(basin_time_df), desc="获取qualifiers"):
        basin_id = str(row['basin_id']).zfill(8)  # 确保是8位，前导零补齐
        
        # 根据time_period选择时间范围
        if time_period == "train":
            start_date = row['train_start'][:10]  # 只取日期部分
            end_date = row['train_end'][:10]
        elif time_period == "valid":
            start_date = row['valid_start'][:10]
            end_date = row['valid_end'][:10]
        elif time_period == "test":
            start_date = row['test_start'][:10]
            end_date = row['test_end'][:10]
        elif time_period == "all":
            # 使用从训练开始到测试结束的完整范围
            start_date = row['train_start'][:10]
            end_date = row['test_end'][:10]
        else:
            raise ValueError(f"未知的时间段: {time_period}")
        
        # 计算时间跨度（年）
        start_dt = pd.to_datetime(start_date)
        end_dt = pd.to_datetime(end_date)
        years_span = (end_dt - start_dt).days / 365.25
        
        tqdm.write(f"\n流域 {basin_id}:")
        tqdm.write(f"  时间范围: {start_date} 至 {end_date} ({years_span:.1f}年)")
        
        try:
            # 如果时间跨度大于1年，按年份分批获取
            if years_span > 1:
                tqdm.write(f"  时间跨度较长，按年份分批获取...")
                basin_qualifiers = fetch_basin_by_years(
                    fetcher, basin_id, start_date, end_date
                )
            else:
                # 直接获取
                tqdm.write(f"  直接获取...")
                result = fetcher.fetch_multiple_gauges(
                    gauge_ids=[basin_id],
                    start_date=start_date,
                    end_date=end_date,
                    use_cache=True
                )
                basin_qualifiers = result.get(basin_id, {})
            
            # 检查是否成功获取数据
            has_discharge = 'discharge' in basin_qualifiers and \
                          basin_qualifiers['discharge'] is not None and \
                          not basin_qualifiers['discharge'].empty
            has_gage_height = 'gage_height' in basin_qualifiers and \
                            basin_qualifiers['gage_height'] is not None and \
                            not basin_qualifiers['gage_height'].empty
            
            if has_discharge or has_gage_height:
                all_qualifiers[basin_id] = basin_qualifiers
                discharge_count = len(basin_qualifiers['discharge']) if has_discharge else 0
                gage_height_count = len(basin_qualifiers['gage_height']) if has_gage_height else 0
                tqdm.write(f"  [成功] 径流: {discharge_count}, 水位: {gage_height_count}")
                success_count += 1
            else:
                tqdm.write(f"  [警告] 未获取到任何数据")
                fail_count += 1
            
        except Exception as e:
            tqdm.write(f"  [错误] {str(e)[:100]}")
            fail_count += 1
        
        # 短暂延迟，避免API限流
        time.sleep(0.5)
    
    print(f"\n{'='*80}")
    print(f"获取完成统计")
    print(f"{'='*80}")
    print(f"  成功: {success_count}/{len(basin_time_df)}")
    print(f"  失败: {fail_count}/{len(basin_time_df)}")
    
    return all_qualifiers, fetcher


def fetch_basin_by_years(fetcher, basin_id, start_date, end_date):
    """
    按年份分批获取单个流域的qualifiers
    
    Parameters
    ----------
    fetcher : USGSQualifiersFetcher
        Fetcher实例
    basin_id : str
        流域ID
    start_date : str
        开始日期 'YYYY-MM-DD'
    end_date : str
        结束日期 'YYYY-MM-DD'
    
    Returns
    -------
    dict
        {'discharge': df, 'gage_height': df}
    """
    start_year = int(start_date[:4])
    end_year = int(end_date[:4])
    
    discharge_dfs = []
    gage_height_dfs = []
    
    for year in range(start_year, end_year + 1):
        # 确定该年的起止日期
        year_start = f"{year}-01-01" if year > start_year else start_date
        year_end = f"{year}-12-31" if year < end_year else end_date
        
        try:
            result = fetcher.fetch_multiple_gauges(
                gauge_ids=[basin_id],
                start_date=year_start,
                end_date=year_end,
                use_cache=True
            )
            
            basin_data = result.get(basin_id, {})
            
            if 'discharge' in basin_data and basin_data['discharge'] is not None:
                discharge_dfs.append(basin_data['discharge'])
            
            if 'gage_height' in basin_data and basin_data['gage_height'] is not None:
                gage_height_dfs.append(basin_data['gage_height'])
                
        except Exception as e:
            tqdm.write(f"    年份 {year} 失败: {str(e)[:50]}")
            continue
    
    # 合并所有年份的数据
    result = {}
    if discharge_dfs:
        result['discharge'] = pd.concat(discharge_dfs, ignore_index=False).sort_index()
    if gage_height_dfs:
        result['gage_height'] = pd.concat(gage_height_dfs, ignore_index=False).sort_index()
    
    return result


def main():
    """主函数"""
    
    print("=" * 80)
    print("按每个流域独立时间范围获取USGS Qualifiers")
    print("=" * 80)
    
    # ==================== 1. 加载流域时间配置 ====================
    print("\n[1] 加载流域时间配置...")
    
    config_file = "runtime_basin_time_ranges_simple.csv"
    
    try:
        basin_time_df = load_basin_time_ranges(config_file)
    except Exception as e:
        print(f"\n错误: {e}")
        return
    
    print(f"\n前5个流域示例:")
    print(basin_time_df[['basin_id', 'train_start', 'train_end']].head())
    
    # ==================== 2. 配置参数 ====================
    print("\n[2] 配置参数...")
    
    # 选择时间段
    TIME_PERIOD = "train"  # 可选: 'train', 'valid', 'test', 'all'
    
    # 输出配置
    OUTPUT_DIR = f"qualifiers_output_per_basin_{TIME_PERIOD}"
    CACHE_DIR = f"qualifiers_cache_per_basin_{TIME_PERIOD}"
    
    # 测试选项：只处理前N个流域（设为None处理全部）
    MAX_BASINS = 2  # 设置为 5 进行快速测试
    
    print(f"  时间段: {TIME_PERIOD}")
    print(f"  输出目录: {OUTPUT_DIR}")
    print(f"  缓存目录: {CACHE_DIR}")
    if MAX_BASINS:
        print(f"  [测试模式] 只处理前 {MAX_BASINS} 个流域")
    else:
        print(f"  处理全部 {len(basin_time_df)} 个流域")
    
    # 估算耗时
    avg_years_per_basin = (
        (pd.to_datetime(basin_time_df[f'{TIME_PERIOD}_end']) - 
         pd.to_datetime(basin_time_df[f'{TIME_PERIOD}_start'])).dt.days / 365.25
    ).mean()
    
    basins_to_process = MAX_BASINS if MAX_BASINS else len(basin_time_df)
    estimated_minutes = basins_to_process * avg_years_per_basin * 5 / 60
    
    print(f"\n预计信息:")
    print(f"  平均时间跨度: {avg_years_per_basin:.1f} 年/流域")
    print(f"  预计耗时: 约 {estimated_minutes:.1f} 分钟")
    
    # ==================== 3. 用户确认 ====================
    print(f"\n{'='*80}")
    print("重要提示")
    print(f"{'='*80}")
    print(f"将为每个流域使用其独立的{TIME_PERIOD}时间范围获取qualifiers")
    print(f"不同流域的时间范围可能不同")
    print(f"这可能需要较长时间，但会自动使用缓存加速")
    
    response = input("\n确认开始? (y/n): ")
    if response.lower() != 'y':
        print("已取消")
        return
    
    # ==================== 4. 获取qualifiers ====================
    start_time = time.time()
    
    qualifiers_data, fetcher = fetch_qualifiers_per_basin(
        basin_time_df=basin_time_df,
        time_period=TIME_PERIOD,
        output_dir=OUTPUT_DIR,
        cache_dir=CACHE_DIR,
        max_basins=MAX_BASINS
    )
    
    # ==================== 5. 与CAMELSH数据合并 ====================
    if len(qualifiers_data) == 0:
        print("\n错误: 未获取到任何qualifiers数据")
        return
    
    print(f"\n{'='*80}")
    print("与CAMELSH数据合并")
    print(f"{'='*80}")

    # Read CAMELSH path from project config (same source as multi_task_lstm.py)
    import os
    try:
        from config import CAMELSH_DATA_PATH
    except Exception as e:
        print(f"\n错误: 无法从项目 config.py 读取 CAMELSH_DATA_PATH - {e}")
        return

    if not os.path.exists(CAMELSH_DATA_PATH):
        print(f"\n错误: CAMELSH_DATA_PATH 不存在: {CAMELSH_DATA_PATH}")
        print("请在项目 config.py 中修正 CAMELSH_DATA_PATH 指向真实 CAMELSH 数据目录")
        return

    # Build per-basin time range mapping for slicing after loading
    # Use timezone-aware UTC timestamps to match fetcher's UTC-normalized indices.
    basin_time_ranges = {}
    basins_for_merge = []

    # Align with MAX_BASINS limitation (if enabled)
    basin_time_df_for_merge = basin_time_df.head(MAX_BASINS) if MAX_BASINS else basin_time_df

    for _, row in basin_time_df_for_merge.iterrows():
        basin_id = str(row["basin_id"]).zfill(8)

        if TIME_PERIOD == "train":
            start_raw = row["train_start"]
            end_raw = row["train_end"]
        elif TIME_PERIOD == "valid":
            start_raw = row["valid_start"]
            end_raw = row["valid_end"]
        elif TIME_PERIOD == "test":
            start_raw = row["test_start"]
            end_raw = row["test_end"]
        else:  # "all"
            start_raw = row["train_start"]
            end_raw = row["test_end"]

        start_ts = pd.to_datetime(start_raw)
        end_ts = pd.to_datetime(end_raw)

        if start_ts.tz is None:
            start_ts = start_ts.tz_localize("UTC")
        else:
            start_ts = start_ts.tz_convert("UTC")

        if end_ts.tz is None:
            end_ts = end_ts.tz_localize("UTC")
        else:
            end_ts = end_ts.tz_convert("UTC")

        basin_time_ranges[basin_id] = (start_ts, end_ts)
        basins_for_merge.append(basin_id)

    # Load a minimal global time range that covers all basins (date precision is enough for CAMELSH reader)
    global_start = min(v[0] for v in basin_time_ranges.values()).date().isoformat()
    global_end = max(v[1] for v in basin_time_ranges.values()).date().isoformat()

    merged_df = fetcher.merge_with_camelsh_dataset(
        camelsh_data_path=CAMELSH_DATA_PATH,
        gauge_ids=basins_for_merge,
        time_range=[global_start, global_end],
        qualifiers_data=qualifiers_data,
        add_weights=True,
        basin_time_ranges=basin_time_ranges,
    )
    
    # ==================== 6. 输出结果 ====================
    elapsed_time = time.time() - start_time
    
    print(f"\n{'='*80}")
    print("完成!")
    print(f"{'='*80}")
    print(f"总耗时: {elapsed_time / 60:.1f} 分钟")
    print(f"\n合并数据统计:")
    print(f"  总记录数: {len(merged_df):,}")
    print(f"  流域数: {merged_df['gauge_id'].nunique()}")
    if 'datetime' in merged_df.columns:
        print(f"  时间范围: {merged_df['datetime'].min()} 至 {merged_df['datetime'].max()}")
    
    # 统计qualifiers覆盖率
    if 'Q_flag' in merged_df.columns:
        q_coverage = (merged_df['Q_flag'] != 'missing').sum() / len(merged_df) * 100
        print(f"  径流qualifiers覆盖率: {q_coverage:.2f}%")
    
    if 'H_flag' in merged_df.columns:
        h_coverage = (merged_df['H_flag'] != 'missing').sum() / len(merged_df) * 100
        print(f"  水位qualifiers覆盖率: {h_coverage:.2f}%")
    
    print(f"\n输出文件:")
    print(f"  合并数据: {OUTPUT_DIR}/camelsh_with_qualifiers.csv")
    print(f"  统计报告: {OUTPUT_DIR}/qualifiers_report.txt")
    print(f"  缓存目录: {CACHE_DIR}")
    
    # 按流域统计覆盖率
    print(f"\n各流域qualifiers覆盖率:")
    for basin_id in merged_df['gauge_id'].unique()[:10]:  # 只显示前10个
        basin_data = merged_df[merged_df['gauge_id'] == basin_id]
        q_cov = (basin_data['Q_flag'] != 'missing').sum() / len(basin_data) * 100
        h_cov = (basin_data['H_flag'] != 'missing').sum() / len(basin_data) * 100
        print(f"  {basin_id}: 径流 {q_cov:.1f}%, 水位 {h_cov:.1f}%")
    
    if merged_df['gauge_id'].nunique() > 10:
        print(f"  ... (显示前10个，共{merged_df['gauge_id'].nunique()}个流域)")
    
    print(f"\n{'='*80}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n用户中断操作")
    except Exception as e:
        print(f"\n发生错误: {e}")
        import traceback
        traceback.print_exc()

