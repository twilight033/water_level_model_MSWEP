"""
按年份分批获取USGS qualifiers数据
避免长时间范围导致的API超时问题
"""

import sys
from pathlib import Path

# 添加项目路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(Path(__file__).parent))

from usgs_qualifiers_fetcher import USGSQualifiersFetcher
import pandas as pd
from tqdm import tqdm
import time


def merge_gauge_data(all_results):
    """
    合并多年份的数据
    
    Parameters
    ----------
    all_results : dict
        {gauge_id: {year: {'discharge': df, 'gage_height': df}}}
    
    Returns
    -------
    dict
        {gauge_id: {'discharge': df, 'gage_height': df}}
    """
    final_results = {}
    
    for gauge_id, year_data in all_results.items():
        final_results[gauge_id] = {}
        
        # 合并discharge数据
        discharge_dfs = []
        for year, data in year_data.items():
            if 'discharge' in data and data['discharge'] is not None and not data['discharge'].empty:
                discharge_dfs.append(data['discharge'])
        
        if discharge_dfs:
            final_results[gauge_id]['discharge'] = pd.concat(discharge_dfs, ignore_index=False).sort_index()
        
        # 合并gage_height数据
        gage_height_dfs = []
        for year, data in year_data.items():
            if 'gage_height' in data and data['gage_height'] is not None and not data['gage_height'].empty:
                gage_height_dfs.append(data['gage_height'])
        
        if gage_height_dfs:
            final_results[gauge_id]['gage_height'] = pd.concat(gage_height_dfs, ignore_index=False).sort_index()
    
    return final_results


def fetch_by_years(
    gauge_ids,
    start_year,
    end_year,
    output_dir="qualifiers_output_full",
    cache_dir="qualifiers_cache_full",
    use_cache=True
):
    """
    按年份批量获取qualifiers
    
    Parameters
    ----------
    gauge_ids : list
        站点ID列表
    start_year : int
        开始年份
    end_year : int
        结束年份
    output_dir : str
        输出目录
    cache_dir : str
        缓存目录
    use_cache : bool
        是否使用缓存
    
    Returns
    -------
    tuple
        (merged_results, fetcher)
    """
    
    fetcher = USGSQualifiersFetcher(
        output_dir=output_dir,
        cache_dir=cache_dir
    )
    
    # 存储所有年份的数据: {gauge_id: {year: {'discharge': df, 'gage_height': df}}}
    all_results = {}
    
    # 统计信息
    total_years = end_year - start_year + 1
    success_count = 0
    fail_count = 0
    
    print(f"\n{'='*80}")
    print(f"开始按年份获取qualifiers数据")
    print(f"{'='*80}")
    print(f"站点数: {len(gauge_ids)}")
    print(f"年份范围: {start_year} - {end_year} ({total_years}年)")
    print(f"预计请求数: {total_years * len(gauge_ids)}")
    print(f"预计耗时: 约 {total_years * len(gauge_ids) * 5 / 60:.1f} 分钟")
    print(f"使用缓存: {'是' if use_cache else '否'}")
    
    # 按年份循环
    for year in tqdm(range(start_year, end_year + 1), desc="年份进度", position=0):
        
        start_date = f"{year}-01-01"
        end_date = f"{year}-12-31"
        
        tqdm.write(f"\n正在获取 {year} 年的数据 ({start_date} 至 {end_date})...")
        
        try:
            # 获取该年度的数据
            year_results = fetcher.fetch_multiple_gauges(
                gauge_ids=gauge_ids,
                start_date=start_date,
                end_date=end_date,
                use_cache=use_cache
            )
            
            # 存储数据
            for gauge_id, data in year_results.items():
                if gauge_id not in all_results:
                    all_results[gauge_id] = {}
                
                all_results[gauge_id][year] = data
            
            # 统计该年度获取的数据量
            discharge_count = sum(1 for gid, data in year_results.items() 
                                 if 'discharge' in data and data['discharge'] is not None and not data['discharge'].empty)
            gage_height_count = sum(1 for gid, data in year_results.items() 
                                   if 'gage_height' in data and data['gage_height'] is not None and not data['gage_height'].empty)
            
            tqdm.write(f"  [OK] {year} 年完成 - 径流: {discharge_count}/{len(gauge_ids)} 站点, 水位: {gage_height_count}/{len(gauge_ids)} 站点")
            success_count += 1
            
        except Exception as e:
            tqdm.write(f"  [ERROR] {year} 年失败: {str(e)}")
            fail_count += 1
            continue
        
        # 短暂延迟，避免API限流
        time.sleep(0.5)
    
    # 合并所有年份的数据
    print(f"\n{'='*80}")
    print("合并各年份数据...")
    print(f"{'='*80}")
    
    final_results = merge_gauge_data(all_results)
    
    # 输出统计信息
    print(f"\n数据获取完成:")
    print(f"  成功: {success_count}/{total_years} 年")
    print(f"  失败: {fail_count}/{total_years} 年")
    print(f"\n站点数据统计:")
    for gauge_id, data in final_results.items():
        discharge_records = len(data['discharge']) if 'discharge' in data else 0
        gage_height_records = len(data['gage_height']) if 'gage_height' in data else 0
        print(f"  {gauge_id}:")
        print(f"    径流记录: {discharge_records:,}")
        print(f"    水位记录: {gage_height_records:,}")
    
    return final_results, fetcher


def main():
    """主函数"""
    
    # ==================== 配置参数 ====================
    
    # 1. 站点ID（可以从valid_waterlevel_basins.txt读取，或手动指定）
    import os
    import ast
    
    basin_file = "valid_waterlevel_basins.txt"
    
    if os.path.exists(basin_file):
        print(f"从 {basin_file} 读取流域ID...")
        with open(basin_file, 'r', encoding='utf-8') as f:
            content = f.read()
            start_idx = content.find('[')
            end_idx = content.rfind(']') + 1
            ALL_GAUGE_IDS = ast.literal_eval(content[start_idx:end_idx])
        print(f"读取了 {len(ALL_GAUGE_IDS)} 个流域ID")
        
        # 默认使用前10个站点作为测试
        print(f"\n注意: 为了测试，默认只使用前10个站点")
        print(f"如需处理全部站点，请修改下面的切片")
        GAUGE_IDS = ALL_GAUGE_IDS[:10]
    else:
        print(f"警告: {basin_file} 不存在，使用示例站点")
        GAUGE_IDS = ["01646500", "01434000"]
    
    # 2. 时间范围
    START_YEAR = 2001
    END_YEAR = 2024
    
    # 3. 输出配置
    OUTPUT_DIR = "qualifiers_output_full"
    CACHE_DIR = "qualifiers_cache_full"
    
    # 4. CAMELSH数据文件
    CAMELSH_FLOW_FILE = "camelsh_exported/flow_hourly.csv"
    CAMELSH_WATERLEVEL_FILE = "camelsh_exported/waterlevel_hourly.csv"
    
    # 检查CAMELSH数据是否存在
    if not os.path.exists(CAMELSH_FLOW_FILE):
        print("\n" + "!"*80)
        print("错误: CAMELSH数据文件不存在")
        print("!"*80)
        print(f"找不到: {CAMELSH_FLOW_FILE}")
        print(f"找不到: {CAMELSH_WATERLEVEL_FILE}")
        print("\n请先运行: uv run python qualifiers_fetcher/export_camelsh_data.py")
        print("!"*80)
        return
    
    print(f"\n{'='*80}")
    print("配置信息")
    print(f"{'='*80}")
    print(f"站点数: {len(GAUGE_IDS)}")
    print(f"站点ID: {GAUGE_IDS[:5]}{'...' if len(GAUGE_IDS) > 5 else ''}")
    print(f"时间范围: {START_YEAR} - {END_YEAR}")
    print(f"输出目录: {OUTPUT_DIR}")
    print(f"缓存目录: {CACHE_DIR}")
    
    # 询问用户确认
    print(f"\n{'='*80}")
    print("重要提示")
    print(f"{'='*80}")
    print(f"将要获取 {len(GAUGE_IDS)} 个站点，{END_YEAR - START_YEAR + 1} 年的数据")
    print(f"预计耗时: 约 {len(GAUGE_IDS) * (END_YEAR - START_YEAR + 1) * 5 / 60:.1f} 分钟")
    print(f"\n如果这是首次运行，建议先用少量站点测试（如修改为GAUGE_IDS[:2]）")
    
    response = input("\n确认开始获取? (y/n): ")
    if response.lower() != 'y':
        print("已取消")
        return
    
    # ==================== 执行获取 ====================
    
    start_time = time.time()
    
    # 按年份获取数据
    results, fetcher = fetch_by_years(
        gauge_ids=GAUGE_IDS,
        start_year=START_YEAR,
        end_year=END_YEAR,
        output_dir=OUTPUT_DIR,
        cache_dir=CACHE_DIR,
        use_cache=True
    )
    
    # ==================== 与CAMELSH数据合并 ====================
    
    print(f"\n{'='*80}")
    print("与CAMELSH数据合并")
    print(f"{'='*80}")
    
    merged_df = fetcher.merge_with_camelsh(
        camelsh_flow_file=CAMELSH_FLOW_FILE,
        camelsh_waterlevel_file=CAMELSH_WATERLEVEL_FILE,
        qualifiers_data=results,
        add_weights=True
    )
    
    # ==================== 输出结果 ====================
    
    elapsed_time = time.time() - start_time
    
    print(f"\n{'='*80}")
    print("完成!")
    print(f"{'='*80}")
    print(f"总耗时: {elapsed_time / 60:.1f} 分钟")
    print(f"\n合并数据统计:")
    print(f"  总记录数: {len(merged_df):,}")
    print(f"  站点数: {merged_df['gauge_id'].nunique()}")
    print(f"  时间范围: {merged_df.index.min()} 至 {merged_df.index.max()}")
    
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
    
    print(f"\n{'='*80}")


if __name__ == "__main__":
    main()

