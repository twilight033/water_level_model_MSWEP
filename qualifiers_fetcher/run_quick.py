"""
快速运行脚本 - 从实际项目配置获取qualifiers
根据你的valid_waterlevel_basins.txt和CAMELSH数据自动配置
"""

import sys
import os
from pathlib import Path

# 添加项目根目录和qualifiers_fetcher目录到Python路径
project_root = Path(__file__).parent.parent
qualifiers_dir = Path(__file__).parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(qualifiers_dir))

from usgs_qualifiers_fetcher import USGSQualifiersFetcher
import ast


def load_basin_ids():
    """从valid_waterlevel_basins.txt加载流域ID"""
    basin_file = "../valid_waterlevel_basins.txt"
    
    if not os.path.exists(basin_file):
        print(f"错误: 找不到文件 {basin_file}")
        print("请先运行项目中的其他脚本生成此文件")
        return []
    
    with open(basin_file, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # 提取列表
    start_idx = content.find('[')
    end_idx = content.rfind(']') + 1
    
    if start_idx == -1 or end_idx == 0:
        print("错误: 无法从文件中解析流域ID列表")
        return []
    
    basin_list = ast.literal_eval(content[start_idx:end_idx])
    return basin_list


def main():
    """主函数"""
    
    print("=" * 80)
    print("USGS Qualifiers Fetcher - 快速运行")
    print("=" * 80)
    
    # ==================== Step 1: 加载配置 ====================
    
    print("\nStep 1: 加载配置...")
    
    # 从项目中加载流域ID
    basin_ids = load_basin_ids()
    
    if not basin_ids:
        print("错误: 未能加载流域ID")
        return
    
    print(f"  加载了 {len(basin_ids)} 个流域ID")
    print(f"  示例: {basin_ids[:5]}")
    
    # 时间范围（根据项目配置）
    # 从multi_task_lstm.py中可以看到使用2001-2024的数据
    START_DATE = "2001-01-01"
    END_DATE = "2024-12-31"
    
    print(f"  时间范围: {START_DATE} 到 {END_DATE}")
    
    # ==================== Step 2: 选择要处理的流域 ====================
    
    print("\nStep 2: 选择要处理的流域...")
    
    # 选项1: 处理前N个流域（测试用）
    N_BASINS = 10  # 先测试10个流域
    selected_basins = basin_ids[:N_BASINS]
    
    # 选项2: 处理所有流域（生产用）
    # selected_basins = basin_ids
    
    print(f"  选择了 {len(selected_basins)} 个流域进行处理")
    print(f"  站点: {selected_basins}")
    
    # ==================== Step 3: 导出CAMELSH数据 ====================
    
    print("\nStep 3: 检查CAMELSH数据...")
    
    # 检查是否已经导出
    camelsh_exported_dir = "camelsh_exported"
    flow_file = os.path.join(camelsh_exported_dir, "flow_hourly.csv")
    wl_file = os.path.join(camelsh_exported_dir, "waterlevel_hourly.csv")
    
    if not os.path.exists(flow_file) or not os.path.exists(wl_file):
        print("  CAMELSH数据文件不存在，需要先导出...")
        print("  运行: uv run python export_camelsh_data.py")
        print("  或手动准备CSV文件")
        
        # 询问用户是否继续
        response = input("\n  是否继续？(输入'y'继续，其他键退出): ")
        if response.lower() != 'y':
            print("  用户取消")
            return
    else:
        print(f"  找到径流数据: {flow_file}")
        print(f"  找到水位数据: {wl_file}")
    
    # ==================== Step 4: 获取qualifiers ====================
    
    print("\nStep 4: 从USGS NWIS获取qualifiers...")
    
    fetcher = USGSQualifiersFetcher(
        output_dir="qualifiers_output",
        cache_dir="qualifiers_cache"
    )
    
    qualifiers_data = fetcher.fetch_multiple_gauges(
        gauge_ids=selected_basins,
        start_date=START_DATE,
        end_date=END_DATE,
        use_cache=True,
        delay=0.5
    )
    
    # ==================== Step 5: 合并数据 ====================
    
    print("\nStep 5: 与CAMELSH数据合并...")
    
    if os.path.exists(flow_file) and os.path.exists(wl_file):
        merged_df = fetcher.merge_with_camelsh(
            camelsh_flow_file=flow_file,
            camelsh_waterlevel_file=wl_file,
            qualifiers_data=qualifiers_data,
            output_file=None,
            add_weights=True
        )
        
        print("\n" + "=" * 80)
        print("完成!")
        print("=" * 80)
        print(f"\n输出文件:")
        print(f"  - 合并数据: qualifiers_output/camelsh_with_qualifiers.csv")
        print(f"  - 统计报告: qualifiers_output/qualifiers_report.txt")
        print(f"  - 缓存目录: qualifiers_cache/")
        
        # 显示数据预览
        print(f"\n数据预览（前5行）:")
        print(merged_df.head())
        
    else:
        print("\n警告: CAMELSH数据文件不存在，只保存了qualifiers数据到缓存")
    
    # ==================== Step 6: 下一步建议 ====================
    
    print("\n" + "=" * 80)
    print("下一步建议")
    print("=" * 80)
    print("\n1. 查看统计报告:")
    print("   cat qualifiers_output/qualifiers_report.txt")
    print("\n2. 如果测试成功，处理所有流域:")
    print("   修改此脚本中的 N_BASINS 或 selected_basins = basin_ids")
    print("\n3. 在训练中使用权重:")
    print("   根据 Q_weight 和 H_weight 列调整样本权重")
    print("\n4. 过滤低质量数据:")
    print("   例如: df[df['Q_weight'] > 0.7]")


if __name__ == "__main__":
    main()

