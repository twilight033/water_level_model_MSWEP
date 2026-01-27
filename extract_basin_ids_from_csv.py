"""
从 runtime_basins.csv 提取流域ID列表
（这个文件是之前运行 extract_per_basin_config.py 时生成的，包含了实际使用的100个流域）
"""

import pandas as pd
from datetime import datetime

def main():
    print("=" * 80)
    print("提取实际使用的流域ID列表")
    print("=" * 80)
    
    # 读取CSV文件
    input_file = "runtime_basins.csv"
    df = pd.read_csv(input_file, dtype={'basin_id': str})
    
    basin_ids = df['basin_id'].tolist()
    
    print(f"\n从 {input_file} 读取了 {len(basin_ids)} 个流域ID")
    
    # ==================== 1. 保存为Python格式 ====================
    output_txt = "actual_basin_ids.txt"
    with open(output_txt, 'w', encoding='utf-8') as f:
        f.write(f"# multi_task_lstm.py 实际使用的流域ID列表\n")
        f.write(f"# 来源: {input_file}\n")
        f.write(f"# 生成时间: {datetime.now()}\n")
        f.write(f"# 流域数量: {len(basin_ids)}\n")
        f.write(f"\n")
        f.write("ACTUAL_BASIN_IDS = [\n")
        for i, basin_id in enumerate(basin_ids):
            if i < len(basin_ids) - 1:
                f.write(f"    '{basin_id}',\n")
            else:
                f.write(f"    '{basin_id}'\n")
        f.write("]\n")
    
    print(f"\n已保存 Python 格式: {output_txt}")
    
    # ==================== 2. 保存为纯文本列表 ====================
    output_simple = "actual_basin_ids_simple.txt"
    with open(output_simple, 'w', encoding='utf-8') as f:
        for basin_id in basin_ids:
            f.write(f"{basin_id}\n")
    
    print(f"已保存 纯文本格式: {output_simple}")
    
    # ==================== 3. 保存详细报告 ====================
    output_report = "actual_basin_ids_report.txt"
    with open(output_report, 'w', encoding='utf-8') as f:
        f.write("=" * 80 + "\n")
        f.write("multi_task_lstm.py 实际使用的流域ID报告\n")
        f.write("=" * 80 + "\n\n")
        f.write(f"生成时间: {datetime.now()}\n")
        f.write(f"数据来源: {input_file}\n")
        f.write(f"流域数量: {len(basin_ids)}\n\n")
        f.write("流域ID列表:\n")
        f.write("-" * 80 + "\n")
        for i, basin_id in enumerate(basin_ids, 1):
            f.write(f"{i:3d}. {basin_id}\n")
        f.write("\n" + "=" * 80 + "\n")
    
    print(f"已保存 详细报告: {output_report}")
    
    # ==================== 4. 统计信息 ====================
    print(f"\n{'='*80}")
    print("流域ID统计")
    print(f"{'='*80}")
    print(f"总数量: {len(basin_ids)}")
    print(f"\n前10个流域:")
    for basin_id in basin_ids[:10]:
        print(f"  {basin_id}")
    print(f"\n后10个流域:")
    for basin_id in basin_ids[-10:]:
        print(f"  {basin_id}")
    
    # ==================== 5. 生成用于qualifiers_fetcher的格式 ====================
    output_for_fetcher = "basin_ids_for_qualifiers.txt"
    with open(output_for_fetcher, 'w', encoding='utf-8') as f:
        f.write("# 用于 qualifiers_fetcher 的流域ID列表\n")
        f.write("# 格式：每行一个流域ID（8位，带前导零）\n")
        f.write(f"# 总数: {len(basin_ids)}\n")
        f.write("#\n")
        f.write("# 使用方法:\n")
        f.write("# 在 qualifiers_fetcher 脚本中:\n")
        f.write("# with open('basin_ids_for_qualifiers.txt') as f:\n")
        f.write("#     basin_ids = [line.strip() for line in f if not line.startswith('#')]\n")
        f.write("\n")
        for basin_id in basin_ids:
            f.write(f"{basin_id}\n")
    
    print(f"\n已生成 qualifiers_fetcher 专用格式: {output_for_fetcher}")
    
    print(f"\n{'='*80}")
    print("完成！")
    print(f"{'='*80}")
    print(f"\n生成的文件:")
    print(f"  1. {output_txt} - Python列表格式")
    print(f"  2. {output_simple} - 纯文本列表（每行一个ID）")
    print(f"  3. {output_report} - 详细报告")
    print(f"  4. {output_for_fetcher} - qualifiers_fetcher专用格式")
    print(f"\n这些流域ID来自 {input_file}，")
    print(f"是 extract_per_basin_config.py 提取的实际训练流域。")


if __name__ == "__main__":
    main()
