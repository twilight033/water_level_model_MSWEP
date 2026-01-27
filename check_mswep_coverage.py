"""
检查MSWEP数据覆盖情况

分析MSWEP数据文件中包含哪些流域，以及与候选流域的匹配情况
"""

import pandas as pd
import sys
from pathlib import Path


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


def check_mswep_coverage(
    mswep_file="MSWEP/mswep_1000basins_mean_3hourly_1980_2024.csv",
    num_basins_to_check=500
):
    """
    检查MSWEP数据覆盖情况
    
    Parameters
    ----------
    mswep_file : str
        MSWEP数据文件路径
    num_basins_to_check : int
        要检查的流域数量
    """
    print("="*80)
    print("MSWEP数据覆盖情况分析")
    print("="*80)
    
    # 1. 读取候选流域列表
    print("\n[1] 读取候选流域列表...")
    try:
        candidate_basins = load_waterlevel_basins_from_file()
        print(f"  候选流域总数: {len(candidate_basins)}")
    except Exception as e:
        print(f"  读取失败: {e}")
        return
    
    # 2. 选择前N个流域
    basins_to_check = candidate_basins[:num_basins_to_check]
    print(f"  将检查前 {len(basins_to_check)} 个流域")
    
    # 3. 读取MSWEP数据文件
    print(f"\n[2] 读取MSWEP数据文件...")
    print(f"  文件路径: {mswep_file}")
    
    if not Path(mswep_file).exists():
        print(f"  [FAIL] 错误: 文件不存在!")
        return
    
    try:
        # 只读取列名，不加载全部数据以节省内存
        df_header = pd.read_csv(mswep_file, nrows=0)
        mswep_basins = df_header.columns.tolist()
        
        # 去掉索引列（如果第一列是时间）
        if mswep_basins[0] in ['datetime', 'time', 'date', 'Unnamed: 0']:
            mswep_basins = mswep_basins[1:]
        
        print(f"  [OK] 读取成功")
        print(f"  MSWEP包含的流域数量: {len(mswep_basins)}")
        print(f"  前10个流域: {mswep_basins[:10]}")
        print(f"  后10个流域: {mswep_basins[-10:]}")
        
    except Exception as e:
        print(f"  [FAIL] 读取失败: {e}")
        return
    
    # 4. 匹配分析
    print(f"\n[3] 匹配分析...")
    
    # 转换为字符串格式进行匹配
    basins_to_check_str = [str(b) for b in basins_to_check]
    mswep_basins_str = [str(b) for b in mswep_basins]
    
    # 找出匹配和缺失的流域
    matched_basins = [b for b in basins_to_check_str if b in mswep_basins_str]
    missing_basins = [b for b in basins_to_check_str if b not in mswep_basins_str]
    
    match_rate = len(matched_basins) / len(basins_to_check) * 100
    
    print(f"\n匹配结果:")
    print(f"  检查的流域数量: {len(basins_to_check)}")
    print(f"  匹配的流域数量: {len(matched_basins)} ({match_rate:.1f}%)")
    print(f"  缺失的流域数量: {len(missing_basins)} ({100-match_rate:.1f}%)")
    
    # 5. 详细报告
    print(f"\n[4] 生成详细报告...")
    
    report_lines = []
    report_lines.append("="*80)
    report_lines.append("MSWEP数据覆盖情况详细报告")
    report_lines.append("="*80)
    report_lines.append("")
    report_lines.append(f"生成时间: {pd.Timestamp.now()}")
    report_lines.append(f"MSWEP文件: {mswep_file}")
    report_lines.append("")
    
    report_lines.append("统计摘要")
    report_lines.append("-"*80)
    report_lines.append(f"候选流域总数: {len(candidate_basins)}")
    report_lines.append(f"检查的流域数量: {len(basins_to_check)}")
    report_lines.append(f"MSWEP包含的流域总数: {len(mswep_basins)}")
    report_lines.append("")
    report_lines.append(f"匹配的流域数量: {len(matched_basins)} ({match_rate:.1f}%)")
    report_lines.append(f"缺失的流域数量: {len(missing_basins)} ({100-match_rate:.1f}%)")
    report_lines.append("")
    
    # 缺失流域详情
    if missing_basins:
        report_lines.append("")
        report_lines.append("MSWEP中缺失的流域详情")
        report_lines.append("-"*80)
        report_lines.append(f"共 {len(missing_basins)} 个流域在MSWEP数据中不存在")
        report_lines.append("")
        
        # 分组显示（每50个一组）
        for i in range(0, len(missing_basins), 50):
            batch = missing_basins[i:i+50]
            report_lines.append(f"第 {i+1}-{min(i+50, len(missing_basins))} 个缺失流域:")
            # 每行显示10个
            for j in range(0, len(batch), 10):
                line_basins = batch[j:j+10]
                report_lines.append("  " + ", ".join(line_basins))
            report_lines.append("")
    
    # 匹配流域样例
    report_lines.append("")
    report_lines.append("匹配成功的流域样例")
    report_lines.append("-"*80)
    report_lines.append(f"前20个匹配的流域:")
    for i in range(0, min(20, len(matched_basins)), 10):
        line_basins = matched_basins[i:i+10]
        report_lines.append("  " + ", ".join(line_basins))
    
    if len(matched_basins) > 20:
        report_lines.append(f"\n后20个匹配的流域:")
        for i in range(max(0, len(matched_basins)-20), len(matched_basins), 10):
            line_basins = matched_basins[i:i+10]
            report_lines.append("  " + ", ".join(line_basins))
    
    # 建议
    report_lines.append("")
    report_lines.append("")
    report_lines.append("建议")
    report_lines.append("-"*80)
    
    if match_rate >= 90:
        report_lines.append("[OK] 匹配率很高 (>=90%)，可以直接使用")
        report_lines.append(f"   建议: 将 NUM_BASINS 设置为 {len(matched_basins)} 或更少")
    elif match_rate >= 70:
        report_lines.append("[WARN] 匹配率中等 (70-90%)，建议评估是否可用")
        report_lines.append(f"   建议: 将 NUM_BASINS 设置为 {len(matched_basins)} 或更少")
        report_lines.append("   或考虑扩充MSWEP数据以包含缺失的流域")
    else:
        report_lines.append("[FAIL] 匹配率较低 (<70%)，建议扩充MSWEP数据")
        report_lines.append(f"   当前只能使用 {len(matched_basins)} 个流域")
        report_lines.append("   建议:")
        report_lines.append("   1. 更新MSWEP数据文件，包含更多流域")
        report_lines.append("   2. 或调整候选流域列表，优先选择MSWEP中存在的流域")
    
    report_lines.append("")
    report_lines.append("")
    report_lines.append("下一步操作")
    report_lines.append("-"*80)
    report_lines.append("1. 如果需要扩充MSWEP数据:")
    report_lines.append("   - 参考 mswep_coverage_missing_basins.txt 中的缺失流域列表")
    report_lines.append("   - 使用MSWEP数据处理工具提取这些流域的降雨数据")
    report_lines.append("")
    report_lines.append("2. 如果使用现有MSWEP数据:")
    report_lines.append("   - 在 config.py 中设置 NUM_BASINS 不超过匹配数量")
    report_lines.append(f"   - 推荐设置: NUM_BASINS = {len(matched_basins)}")
    report_lines.append("")
    report_lines.append("3. 如果需要优化流域选择:")
    report_lines.append("   - 参考 mswep_coverage_matched_basins.txt 中的匹配流域列表")
    report_lines.append("   - 优先使用这些流域进行训练")
    
    report_lines.append("")
    report_lines.append("="*80)
    
    # 保存报告
    report_file = "mswep_coverage_report.txt"
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write('\n'.join(report_lines))
    print(f"  [OK] 详细报告已保存: {report_file}")
    
    # 保存缺失流域列表
    if missing_basins:
        missing_file = "mswep_coverage_missing_basins.txt"
        with open(missing_file, 'w', encoding='utf-8') as f:
            f.write("# MSWEP数据中缺失的流域列表\n")
            f.write(f"# 共 {len(missing_basins)} 个流域\n\n")
            f.write("MISSING_BASINS = [\n")
            for basin in missing_basins:
                f.write(f"    '{basin}',\n")
            f.write("]\n")
        print(f"  [OK] 缺失流域列表已保存: {missing_file}")
    
    # 保存匹配流域列表
    matched_file = "mswep_coverage_matched_basins.txt"
    with open(matched_file, 'w', encoding='utf-8') as f:
        f.write("# MSWEP数据中匹配的流域列表\n")
        f.write(f"# 共 {len(matched_basins)} 个流域\n\n")
        f.write("MATCHED_BASINS = [\n")
        for basin in matched_basins:
            f.write(f"    '{basin}',\n")
        f.write("]\n")
    print(f"  [OK] 匹配流域列表已保存: {matched_file}")
    
    # 保存CSV格式
    csv_file = "mswep_coverage_analysis.csv"
    analysis_df = pd.DataFrame({
        'basin_id': basins_to_check_str,
        'in_mswep': [b in mswep_basins_str for b in basins_to_check_str],
        'status': ['matched' if b in mswep_basins_str else 'missing' for b in basins_to_check_str]
    })
    analysis_df.to_csv(csv_file, index=False)
    print(f"  [OK] CSV分析文件已保存: {csv_file}")
    
    # 打印部分报告到控制台
    print("\n" + "="*80)
    print("报告摘要")
    print("="*80)
    print(f"\n[OK] 匹配的流域: {len(matched_basins)} 个 ({match_rate:.1f}%)")
    print(f"[FAIL] 缺失的流域: {len(missing_basins)} 个 ({100-match_rate:.1f}%)")
    
    if missing_basins:
        print(f"\n缺失流域示例 (前20个):")
        for basin in missing_basins[:20]:
            print(f"  {basin}")
        if len(missing_basins) > 20:
            print(f"  ... 还有 {len(missing_basins)-20} 个")
    
    print(f"\n推荐配置:")
    print(f"  config.py -> NUM_BASINS = {len(matched_basins)}")
    
    print(f"\n生成的文件:")
    print(f"  1. {report_file} - 详细报告")
    print(f"  2. {missing_file} - 缺失流域列表" if missing_basins else "")
    print(f"  3. {matched_file} - 匹配流域列表")
    print(f"  4. {csv_file} - CSV分析文件")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='检查MSWEP数据覆盖情况')
    parser.add_argument(
        '--mswep-file',
        type=str,
        default='MSWEP/mswep_1000basins_mean_3hourly_1980_2024.csv',
        help='MSWEP数据文件路径'
    )
    parser.add_argument(
        '--num-basins',
        type=int,
        default=500,
        help='要检查的流域数量（默认500）'
    )
    
    args = parser.parse_args()
    
    try:
        check_mswep_coverage(
            mswep_file=args.mswep_file,
            num_basins_to_check=args.num_basins
        )
    except KeyboardInterrupt:
        print("\n\n用户中断操作")
    except Exception as e:
        print(f"\n发生错误: {e}")
        import traceback
        traceback.print_exc()
