"""
按照 multi_task_lstm.py 的实际逻辑提取流域ID列表
（不运行验证，直接从候选列表提取）

使用方法：
    python extract_basins_by_logic.py --num 100
    python extract_basins_by_logic.py --num 200
    python extract_basins_by_logic.py --num 1000
"""

import sys
import ast
import argparse
from datetime import datetime


def load_waterlevel_basins_from_file(file_path="valid_waterlevel_basins.txt"):
    """
    从文件中读取有水位数据的流域列表
    （与 multi_task_lstm.py 中的逻辑完全一致）
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # 查找 VALID_WATER_LEVEL_BASINS = [...]
        start_idx = content.find('VALID_WATER_LEVEL_BASINS = [')
        if start_idx == -1:
            raise ValueError(f"未在文件中找到 VALID_WATER_LEVEL_BASINS")
        
        # 找到列表开始位置
        list_start = content.find('[', start_idx)
        if list_start == -1:
            raise ValueError(f"未找到列表开始标记")
        
        # 找到匹配的结束括号
        bracket_count = 0
        list_end = list_start
        for i in range(list_start, len(content)):
            if content[i] == '[':
                bracket_count += 1
            elif content[i] == ']':
                bracket_count -= 1
                if bracket_count == 0:
                    list_end = i + 1
                    break
        
        if bracket_count != 0:
            raise ValueError(f"列表格式不完整，括号不匹配")
        
        # 提取列表字符串
        list_str = content[list_start:list_end]
        
        # 解析列表
        basin_list = ast.literal_eval(list_str)
        
        print(f"从文件 {file_path} 读取了 {len(basin_list)} 个候选流域")
        return basin_list
        
    except FileNotFoundError:
        print(f"错误: 文件 {file_path} 不存在")
        print("请先运行 scan_waterlevel_basins.py 生成流域列表文件")
        raise
    except Exception as e:
        print(f"读取文件 {file_path} 时出错: {e}")
        raise


def main():
    parser = argparse.ArgumentParser(
        description='按照 multi_task_lstm.py 的逻辑提取流域ID列表'
    )
    parser.add_argument(
        '--num',
        type=int,
        required=True,
        help='要提取的流域数量（例如：100, 200, 1000）'
    )
    parser.add_argument(
        '--input',
        type=str,
        default='valid_waterlevel_basins.txt',
        help='候选流域列表文件路径'
    )
    parser.add_argument(
        '--output-prefix',
        type=str,
        default='extracted_basin_ids',
        help='输出文件名前缀'
    )
    
    args = parser.parse_args()
    num_basins = args.num
    input_file = args.input
    output_prefix = args.output_prefix
    
    print("=" * 80)
    print("按照 multi_task_lstm.py 逻辑提取流域ID")
    print("=" * 80)
    print(f"\n配置:")
    print(f"  输入文件: {input_file}")
    print(f"  提取数量: {num_basins}")
    print(f"  输出前缀: {output_prefix}")
    
    # ==================== 1. 读取候选流域列表 ====================
    print(f"\n正在从文件读取候选流域列表...")
    candidate_basins = load_waterlevel_basins_from_file(input_file)
    
    print(f"\n候选流域统计:")
    print(f"  总候选数量: {len(candidate_basins)}")
    print(f"  前10个: {candidate_basins[:10]}")
    print(f"  后10个: {candidate_basins[-10:]}")
    
    # ==================== 2. 按照代码逻辑选择流域 ====================
    print(f"\n按照 multi_task_lstm.py 的逻辑:")
    print(f"  1. 从 {input_file} 读取候选流域")
    print(f"  2. 实际运行时会验证每个流域的数据有效性")
    print(f"  3. 选择验证通过的前 {num_basins} 个流域")
    print(f"\n当前操作:")
    print(f"  - 不运行数据验证（避免路径/环境问题）")
    print(f"  - 直接从候选列表提取前 {num_basins} 个流域")
    print(f"  - 假设: 候选列表中靠前的流域更可能通过验证")
    
    if num_basins > len(candidate_basins):
        print(f"\n警告: 请求数量 {num_basins} 超过候选流域总数 {len(candidate_basins)}")
        print(f"将提取全部 {len(candidate_basins)} 个流域")
        num_basins = len(candidate_basins)
    
    # 提取前 N 个流域
    selected_basins = candidate_basins[:num_basins]
    
    print(f"\n成功提取 {len(selected_basins)} 个流域ID")
    
    # ==================== 3. 保存结果 ====================
    # 3.1 保存为Python格式
    output_txt = f"{output_prefix}_{num_basins}.txt"
    with open(output_txt, 'w', encoding='utf-8') as f:
        f.write(f"# multi_task_lstm.py 逻辑提取的流域ID列表\n")
        f.write(f"# 来源: {input_file}\n")
        f.write(f"# 生成时间: {datetime.now()}\n")
        f.write(f"# 请求数量: {num_basins}\n")
        f.write(f"# 实际数量: {len(selected_basins)}\n")
        f.write(f"#\n")
        f.write(f"# 说明:\n")
        f.write(f"#   这些流域ID按照 multi_task_lstm.py 的选择逻辑提取\n")
        f.write(f"#   实际运行时会进行数据验证，只有验证通过的流域会被使用\n")
        f.write(f"#   此列表为候选流域的前 {num_basins} 个\n")
        f.write(f"\n")
        f.write(f"BASIN_IDS = [\n")
        for i, basin_id in enumerate(selected_basins):
            if i < len(selected_basins) - 1:
                f.write(f"    '{basin_id}',\n")
            else:
                f.write(f"    '{basin_id}'\n")
        f.write("]\n")
    
    print(f"\n已保存 Python 格式: {output_txt}")
    
    # 3.2 保存为纯文本
    output_simple = f"{output_prefix}_{num_basins}_simple.txt"
    with open(output_simple, 'w', encoding='utf-8') as f:
        for basin_id in selected_basins:
            f.write(f"{basin_id}\n")
    
    print(f"已保存 纯文本格式: {output_simple}")
    
    # 3.3 保存为CSV
    output_csv = f"{output_prefix}_{num_basins}.csv"
    with open(output_csv, 'w', encoding='utf-8') as f:
        f.write("basin_id,index\n")
        for i, basin_id in enumerate(selected_basins):
            f.write(f"{basin_id},{i}\n")
    
    print(f"已保存 CSV 格式: {output_csv}")
    
    # 3.4 保存详细报告
    output_report = f"{output_prefix}_{num_basins}_report.txt"
    with open(output_report, 'w', encoding='utf-8') as f:
        f.write("=" * 80 + "\n")
        f.write("multi_task_lstm.py 逻辑提取的流域ID报告\n")
        f.write("=" * 80 + "\n\n")
        f.write(f"生成时间: {datetime.now()}\n")
        f.write(f"数据来源: {input_file}\n")
        f.write(f"候选流域总数: {len(candidate_basins)}\n")
        f.write(f"请求提取数量: {num_basins}\n")
        f.write(f"实际提取数量: {len(selected_basins)}\n\n")
        
        f.write("提取逻辑说明:\n")
        f.write("-" * 80 + "\n")
        f.write("1. 从 valid_waterlevel_basins.txt 读取候选流域列表\n")
        f.write("2. multi_task_lstm.py 会验证每个流域的数据有效性:\n")
        f.write("   - 同时有有效的径流和水位数据\n")
        f.write("   - 有效数据比例 ≥ 10%\n")
        f.write("   - 数据不全为NaN\n")
        f.write("3. 从验证通过的流域中选择前 NUM_BASINS 个\n")
        f.write("\n")
        f.write("当前提取方式:\n")
        f.write("-" * 80 + "\n")
        f.write("- 不运行实际的数据验证（避免环境/路径问题）\n")
        f.write(f"- 直接从候选列表提取前 {num_basins} 个流域ID\n")
        f.write("- 假设候选列表已按可用性排序\n")
        f.write("\n\n")
        
        f.write("流域ID列表:\n")
        f.write("-" * 80 + "\n")
        for i, basin_id in enumerate(selected_basins, 1):
            f.write(f"{i:4d}. {basin_id}\n")
        f.write("\n" + "=" * 80 + "\n")
    
    print(f"已保存 详细报告: {output_report}")
    
    # ==================== 4. 显示统计信息 ====================
    print(f"\n{'='*80}")
    print("提取完成！")
    print(f"{'='*80}")
    print(f"\n统计信息:")
    print(f"  候选流域总数: {len(candidate_basins)}")
    print(f"  提取流域数量: {len(selected_basins)}")
    print(f"\n前10个流域:")
    for basin_id in selected_basins[:10]:
        print(f"  {basin_id}")
    
    if len(selected_basins) > 20:
        print(f"\n后10个流域:")
        for basin_id in selected_basins[-10:]:
            print(f"  {basin_id}")
    elif len(selected_basins) > 10:
        print(f"\n剩余流域:")
        for basin_id in selected_basins[10:]:
            print(f"  {basin_id}")
    
    print(f"\n生成的文件:")
    print(f"  1. {output_txt} - Python列表格式")
    print(f"  2. {output_simple} - 纯文本列表")
    print(f"  3. {output_csv} - CSV格式")
    print(f"  4. {output_report} - 详细报告")
    
    print(f"\n{'='*80}")
    print("重要说明")
    print(f"{'='*80}")
    print(f"这些流域ID是从候选列表中提取的前 {num_basins} 个。")
    print(f"实际运行 multi_task_lstm.py 时：")
    print(f"  - 会对这些流域进行数据验证")
    print(f"  - 只有验证通过的流域才会被使用")
    print(f"  - 最终使用的流域数量可能少于 {num_basins} 个")
    print(f"\n如需获取实际运行后的流域列表，请查看:")
    print(f"  - runtime_basins.csv")
    print(f"  - runtime_basin_time_ranges_simple.csv")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n用户中断操作")
    except Exception as e:
        print(f"\n发生错误: {e}")
        import traceback
        traceback.print_exc()
