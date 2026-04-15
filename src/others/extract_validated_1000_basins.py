"""
提取经过数据验证的前1000个流域

按照 multi_task_lstm_ablation_missing_labels.py 的逻辑验证流域
"""

import os
import sys
import pandas as pd
from datetime import datetime


def load_waterlevel_basins_from_file(file_path="valid_waterlevel_basins.txt"):
    """从文件中读取有水位数据的流域列表"""
    import ast
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        start_idx = content.find('VALID_WATER_LEVEL_BASINS = [')
        if start_idx == -1:
            raise ValueError(f"未在文件中找到 VALID_WATER_LEVEL_BASINS")
        
        list_start = content.find('[', start_idx)
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
        
        list_str = content[list_start:list_end]
        basin_list = ast.literal_eval(list_str)
        
        return basin_list
        
    except FileNotFoundError:
        print(f"错误: 文件 {file_path} 不存在")
        raise


def main():
    print("="*80)
    print("提取经过验证的前1000个流域")
    print("="*80)
    
    # 导入配置
    from config import CAMELSH_DATA_PATH, NUM_BASINS
    from improved_camelsh_reader import ImprovedCAMELSHReader
    
    # 读取候选流域列表
    print("\n[1] 读取候选流域列表...")
    VALID_WATER_LEVEL_BASINS = load_waterlevel_basins_from_file()
    print(f"  候选流域总数: {len(VALID_WATER_LEVEL_BASINS)}")
    
    # 加载CAMELSH数据
    print("\n[2] 加载CAMELSH数据...")
    camelsh_reader = ImprovedCAMELSHReader(CAMELSH_DATA_PATH, download=False, use_batch=True)
    camelsh = camelsh_reader.camelsh
    default_range = camelsh.default_t_range
    print(f"  数据集时间范围: {default_range}")
    
    # 验证流域 - 设置为1000个
    target_validated = 1000
    max_candidates = min(len(VALID_WATER_LEVEL_BASINS), target_validated * 3)
    
    print(f"\n[3] 验证流域数据...")
    print(f"  目标: 找到 {target_validated} 个有效流域")
    print(f"  将检查前 {max_candidates} 个候选流域")
    print(f"  这可能需要一些时间...")
    
    # 导入验证函数
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    
    # 从 multi_task_lstm.py 导入验证函数
    try:
        from multi_task_lstm import filter_basins_with_valid_data
    except ImportError:
        # 如果导入失败，使用 multi_task_lstm_ablation_missing_labels.py
        from multi_task_lstm_ablation_missing_labels import filter_basins_with_valid_data
    
    # 执行验证
    validated_basins = filter_basins_with_valid_data(
        camelsh_reader=camelsh_reader,
        basin_list=VALID_WATER_LEVEL_BASINS,
        time_range=default_range,
        max_basins_to_check=max_candidates,
        min_valid_ratio=0.1
    )
    
    print(f"\n  验证完成！")
    print(f"  通过验证的流域数量: {len(validated_basins)}")
    
    # 选择前1000个
    if len(validated_basins) >= target_validated:
        chosen_basins = validated_basins[:target_validated]
        print(f"  选择前 {target_validated} 个")
    else:
        chosen_basins = validated_basins
        print(f"  警告: 只有 {len(validated_basins)} 个流域通过验证")
        print(f"  将使用全部 {len(validated_basins)} 个流域")
    
    # 保存结果
    print(f"\n[4] 保存结果...")
    
    # 4.1 Python格式
    output_txt = f"validated_basin_ids_{len(chosen_basins)}.txt"
    with open(output_txt, 'w', encoding='utf-8') as f:
        f.write(f"# 经过数据验证的流域ID列表\n")
        f.write(f"# 来源: valid_waterlevel_basins.txt\n")
        f.write(f"# 验证时间: {datetime.now()}\n")
        f.write(f"# 验证标准:\n")
        f.write(f"#   - 同时有有效的径流和水位数据\n")
        f.write(f"#   - 有效数据比例 >= 10%\n")
        f.write(f"#   - 数据不全为NaN\n")
        f.write(f"# 候选流域总数: {len(VALID_WATER_LEVEL_BASINS)}\n")
        f.write(f"# 检查的流域数: {max_candidates}\n")
        f.write(f"# 通过验证数量: {len(validated_basins)}\n")
        f.write(f"# 实际使用数量: {len(chosen_basins)}\n")
        f.write(f"\n")
        f.write(f"VALIDATED_BASIN_IDS = [\n")
        for i, basin_id in enumerate(chosen_basins):
            if i < len(chosen_basins) - 1:
                f.write(f"    '{basin_id}',\n")
            else:
                f.write(f"    '{basin_id}'\n")
        f.write("]\n")
    
    print(f"  已保存 Python 格式: {output_txt}")
    
    # 4.2 纯文本格式
    output_simple = f"validated_basin_ids_{len(chosen_basins)}_simple.txt"
    with open(output_simple, 'w', encoding='utf-8') as f:
        for basin_id in chosen_basins:
            f.write(f"{basin_id}\n")
    
    print(f"  已保存 纯文本格式: {output_simple}")
    
    # 4.3 CSV格式
    output_csv = f"validated_basin_ids_{len(chosen_basins)}.csv"
    with open(output_csv, 'w', encoding='utf-8') as f:
        f.write("basin_id,index,in_candidate_list_rank\n")
        for i, basin_id in enumerate(chosen_basins):
            # 找到这个流域在候选列表中的位置
            candidate_rank = VALID_WATER_LEVEL_BASINS.index(basin_id) + 1
            f.write(f"{basin_id},{i},{candidate_rank}\n")
    
    print(f"  已保存 CSV 格式: {output_csv}")
    
    # 4.4 详细报告
    output_report = f"validated_basin_ids_{len(chosen_basins)}_report.txt"
    with open(output_report, 'w', encoding='utf-8') as f:
        f.write("="*80 + "\n")
        f.write("经过数据验证的流域ID报告\n")
        f.write("="*80 + "\n\n")
        f.write(f"生成时间: {datetime.now()}\n")
        f.write(f"数据来源: valid_waterlevel_basins.txt\n")
        f.write(f"数据集时间范围: {default_range}\n\n")
        
        f.write("验证标准:\n")
        f.write("-"*80 + "\n")
        f.write("1. 同时有有效的径流和水位数据\n")
        f.write("2. 有效数据比例 >= 10%\n")
        f.write("3. 数据不全为NaN\n\n")
        
        f.write("统计信息:\n")
        f.write("-"*80 + "\n")
        f.write(f"候选流域总数: {len(VALID_WATER_LEVEL_BASINS)}\n")
        f.write(f"检查的流域数: {max_candidates}\n")
        f.write(f"通过验证数量: {len(validated_basins)}\n")
        f.write(f"实际使用数量: {len(chosen_basins)}\n")
        f.write(f"验证通过率: {len(validated_basins)/max_candidates*100:.1f}%\n\n")
        
        # 分析候选列表位置分布
        candidate_ranks = [VALID_WATER_LEVEL_BASINS.index(b)+1 for b in chosen_basins]
        f.write("候选列表位置分布:\n")
        f.write("-"*80 + "\n")
        f.write(f"最小位置: {min(candidate_ranks)}\n")
        f.write(f"最大位置: {max(candidate_ranks)}\n")
        f.write(f"平均位置: {sum(candidate_ranks)/len(candidate_ranks):.1f}\n")
        f.write(f"中位数位置: {sorted(candidate_ranks)[len(candidate_ranks)//2]}\n\n")
        
        # 统计位置分布
        in_first_1000 = sum(1 for r in candidate_ranks if r <= 1000)
        in_1001_1500 = sum(1 for r in candidate_ranks if 1000 < r <= 1500)
        in_1501_plus = sum(1 for r in candidate_ranks if r > 1500)
        
        f.write("位置区间分布:\n")
        f.write(f"  前1000名: {in_first_1000} 个 ({in_first_1000/len(chosen_basins)*100:.1f}%)\n")
        f.write(f"  1001-1500名: {in_1001_1500} 个 ({in_1001_1500/len(chosen_basins)*100:.1f}%)\n")
        f.write(f"  1501名以后: {in_1501_plus} 个 ({in_1501_plus/len(chosen_basins)*100:.1f}%)\n\n")
        
        f.write("流域ID列表:\n")
        f.write("-"*80 + "\n")
        for i, basin_id in enumerate(chosen_basins, 1):
            candidate_rank = VALID_WATER_LEVEL_BASINS.index(basin_id) + 1
            f.write(f"{i:4d}. {basin_id} (候选列表第{candidate_rank}名)\n")
        f.write("\n" + "="*80 + "\n")
    
    print(f"  已保存 详细报告: {output_report}")
    
    # 显示摘要
    print(f"\n" + "="*80)
    print("完成！")
    print(f"="*80)
    print(f"\n统计摘要:")
    print(f"  候选流域总数: {len(VALID_WATER_LEVEL_BASINS)}")
    print(f"  检查的流域数: {max_candidates}")
    print(f"  通过验证数量: {len(validated_basins)}")
    print(f"  实际使用数量: {len(chosen_basins)}")
    print(f"  验证通过率: {len(validated_basins)/max_candidates*100:.1f}%")
    
    # 位置分布
    candidate_ranks = [VALID_WATER_LEVEL_BASINS.index(b)+1 for b in chosen_basins]
    in_first_1000 = sum(1 for r in candidate_ranks if r <= 1000)
    in_1001_1500 = sum(1 for r in candidate_ranks if 1000 < r <= 1500)
    
    print(f"\n候选列表位置分布:")
    print(f"  来自前1000名: {in_first_1000} 个 ({in_first_1000/len(chosen_basins)*100:.1f}%)")
    print(f"  来自1001-1500名: {in_1001_1500} 个 ({in_1001_1500/len(chosen_basins)*100:.1f}%)")
    
    print(f"\n前10个验证通过的流域:")
    for basin in chosen_basins[:10]:
        rank = VALID_WATER_LEVEL_BASINS.index(basin) + 1
        print(f"  {basin} (候选列表第{rank}名)")
    
    if len(chosen_basins) > 10:
        print(f"\n后10个验证通过的流域:")
        for basin in chosen_basins[-10:]:
            rank = VALID_WATER_LEVEL_BASINS.index(basin) + 1
            print(f"  {basin} (候选列表第{rank}名)")
    
    print(f"\n生成的文件:")
    print(f"  1. {output_txt} - Python列表格式")
    print(f"  2. {output_simple} - 纯文本列表")
    print(f"  3. {output_csv} - CSV格式（含候选列表位置）")
    print(f"  4. {output_report} - 详细报告")
    
    # MSWEP覆盖分析
    print(f"\n" + "="*80)
    print("MSWEP覆盖分析")
    print(f"="*80)
    
    try:
        # 读取MSWEP流域列表
        mswep_df = pd.read_csv("MSWEP/mswep_1000basins_mean_3hourly_1980_2024.csv", nrows=0)
        mswep_basins = mswep_df.columns.tolist()[1:]
        
        # 检查匹配
        matched = [b for b in chosen_basins if b in mswep_basins]
        missing = [b for b in chosen_basins if b not in mswep_basins]
        
        print(f"\nMSWEP数据覆盖:")
        print(f"  验证通过的流域: {len(chosen_basins)}")
        print(f"  MSWEP中存在: {len(matched)} ({len(matched)/len(chosen_basins)*100:.1f}%)")
        print(f"  MSWEP中缺失: {len(missing)} ({len(missing)/len(chosen_basins)*100:.1f}%)")
        
        if missing:
            print(f"\n缺失流域示例 (前10个):")
            for basin in missing[:10]:
                rank = VALID_WATER_LEVEL_BASINS.index(basin) + 1
                print(f"  {basin} (候选列表第{rank}名)")
            if len(missing) > 10:
                print(f"  ... 还有 {len(missing)-10} 个")
        
        # 保存MSWEP覆盖分析
        coverage_file = f"validated_mswep_coverage_{len(chosen_basins)}.csv"
        with open(coverage_file, 'w', encoding='utf-8') as f:
            f.write("basin_id,in_mswep,candidate_rank\n")
            for basin in chosen_basins:
                in_mswep = 'yes' if basin in mswep_basins else 'no'
                rank = VALID_WATER_LEVEL_BASINS.index(basin) + 1
                f.write(f"{basin},{in_mswep},{rank}\n")
        print(f"\n  已保存 MSWEP覆盖分析: {coverage_file}")
        
    except Exception as e:
        print(f"\n  无法分析MSWEP覆盖: {e}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n用户中断操作")
    except Exception as e:
        print(f"\n发生错误: {e}")
        import traceback
        traceback.print_exc()
