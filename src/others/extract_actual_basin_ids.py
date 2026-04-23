"""
提取 multi_task_lstm.py 实际运行时选择的流域ID列表

这个脚本会模拟 multi_task_lstm.py 的流域选择逻辑，
输出实际会被用于训练的流域ID列表
"""

import sys
from pathlib import Path
from tqdm import tqdm

# 添加项目路径（others/ 及其父目录 src/ 都需要，config.py 在 src/ 下）
sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent))

from improved_camelsh_reader import ImprovedCAMELSHReader
from hydrodataset import StandardVariable


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
        if list_start == -1:
            raise ValueError(f"未找到列表开始标记")
        
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
        
        list_str = content[list_start:list_end]
        basin_list = ast.literal_eval(list_str)
        
        print(f"从文件 {file_path} 读取了 {len(basin_list)} 个候选流域")
        return basin_list
        
    except FileNotFoundError:
        print(f"错误: 文件 {file_path} 不存在")
        raise
    except Exception as e:
        print(f"读取文件 {file_path} 时出错: {e}")
        raise


def filter_basins_with_valid_data(camelsh_reader, basin_list, time_range, max_basins_to_check=None, min_valid_ratio=0.1):
    """
    验证流域列表，只保留同时有有效水位和径流数据的流域
    （与 multi_task_lstm.py 中的逻辑完全一致）
    """
    print(f"\n正在验证流域的水位和径流数据有效性...")
    print(f"候选流域数量: {len(basin_list)}")
    print(f"验证时间范围: {time_range}")
    print(f"最小有效数据比例: {min_valid_ratio:.1%}")
    
    basins_to_check = basin_list[:max_basins_to_check] if max_basins_to_check else basin_list
    print(f"将检查前 {len(basins_to_check)} 个流域...")
    
    valid_basins = []
    invalid_basins = []
    
    batch_size = 50
    for i in tqdm(range(0, len(basins_to_check), batch_size), desc="验证流域数据"):
        batch = basins_to_check[i:i+batch_size]
        
        try:
            waterlevel_ds = camelsh_reader.read_ts_xrdataset(
                gage_id_lst=batch,
                t_range=time_range,
                var_lst=[StandardVariable.WATER_LEVEL]
            )
            flow_ds = camelsh_reader.read_ts_xrdataset(
                gage_id_lst=batch,
                t_range=time_range,
                var_lst=[StandardVariable.STREAMFLOW]
            )
            
            waterlevel_df = None
            flow_df = None
            
            if StandardVariable.WATER_LEVEL in waterlevel_ds.data_vars:
                waterlevel_df = waterlevel_ds[StandardVariable.WATER_LEVEL].to_pandas().T
            else:
                for basin_id in batch:
                    invalid_basins.append((basin_id, "数据集缺少water_level变量"))
                continue
            
            if StandardVariable.STREAMFLOW in flow_ds.data_vars:
                flow_df = flow_ds[StandardVariable.STREAMFLOW].to_pandas().T
            else:
                for basin_id in batch:
                    invalid_basins.append((basin_id, "数据集缺少streamflow变量"))
                continue
            
            for basin_id in batch:
                waterlevel_valid = False
                flow_valid = False
                reasons = []
                
                if basin_id in waterlevel_df.columns:
                    wl_data = waterlevel_df[basin_id]
                    if wl_data.notna().any():
                        wl_valid_ratio = wl_data.notna().sum() / len(wl_data)
                        if wl_valid_ratio >= min_valid_ratio:
                            waterlevel_valid = True
                        else:
                            reasons.append(f"水位有效比例过低: {wl_valid_ratio:.2%}")
                    else:
                        reasons.append("水位数据全为NaN")
                else:
                    reasons.append("水位数据集中不存在")
                
                if basin_id in flow_df.columns:
                    flow_data = flow_df[basin_id]
                    if flow_data.notna().any():
                        flow_valid_ratio = flow_data.notna().sum() / len(flow_data)
                        if flow_valid_ratio >= min_valid_ratio:
                            flow_valid = True
                        else:
                            reasons.append(f"径流有效比例过低: {flow_valid_ratio:.2%}")
                    else:
                        reasons.append("径流数据全为NaN")
                else:
                    reasons.append("径流数据集中不存在")
                
                if waterlevel_valid and flow_valid:
                    valid_basins.append(basin_id)
                else:
                    reason_str = "; ".join(reasons) if reasons else "未知原因"
                    invalid_basins.append((basin_id, reason_str))
                    
        except Exception as e:
            print(f"\n警告: 批量加载失败 ({batch[0]} 到 {batch[-1]}): {e}")
            for basin_id in batch:
                invalid_basins.append((basin_id, f"加载失败: {str(e)[:50]}"))
    
    print(f"\n验证完成:")
    print(f"  有效流域: {len(valid_basins)} 个")
    print(f"  无效流域: {len(invalid_basins)} 个")
    
    return valid_basins


def main():
    """主函数：对CAMELSH全部流域筛选，有效数据比例阈值 >= 30%"""

    import datetime
    import pandas as pd
    from pathlib import Path

    MIN_VALID_RATIO = 0.3  # 有效数据比例阈值

    print("=" * 80)
    print("CAMELSH 全量流域筛选（Q 和 h 有效比例均 >= 30%）")
    print("=" * 80)

    # 导入配置
    from config import CAMELSH_DATA_PATH

    print(f"\n配置信息:")
    print(f"  CAMELSH数据路径: {CAMELSH_DATA_PATH}")
    print(f"  有效数据比例阈值: {MIN_VALID_RATIO:.0%}")

    # ==================== 1. 加载CAMELSH数据读取器 ====================
    print(f"\n正在初始化CAMELSH数据读取器...")
    camelsh_reader = ImprovedCAMELSHReader(CAMELSH_DATA_PATH, download=False, use_batch=True)

    camelsh = camelsh_reader.camelsh
    default_range = camelsh.default_t_range
    print(f"  默认时间范围: {default_range}")

    # ==================== 2. 获取全部候选流域（扫描 Hourly2 目录） ====================
    # 以 Hourly2 目录中存在文件的流域为候选集（必须有水位文件），
    # 径流是否有效由 filter 函数负责判断。
    hourly2_dir = Path(CAMELSH_DATA_PATH) / "CAMELSH" / "Hourly2" / "Hourly2"
    if not hourly2_dir.exists():
        raise FileNotFoundError(f"Hourly2 目录不存在: {hourly2_dir}")

    all_candidates = sorted(
        f.stem.replace("_hourly", "")
        for f in hourly2_dir.glob("*_hourly.nc")
    )
    print(f"\n从 Hourly2 目录扫描到 {len(all_candidates)} 个候选流域（全量，不限数目）")

    # ==================== 3. 验证流域数据（全量，阈值 30%） ====================
    validated_basins = filter_basins_with_valid_data(
        camelsh_reader=camelsh_reader,
        basin_list=all_candidates,
        time_range=default_range,
        max_basins_to_check=None,   # 不限制，检查全部
        min_valid_ratio=MIN_VALID_RATIO
    )

    if len(validated_basins) == 0:
        raise ValueError("未找到任何满足条件的流域！")

    print(f"\n{'='*80}")
    print(f"筛选结果")
    print(f"{'='*80}")
    print(f"候选流域数: {len(all_candidates)}")
    print(f"通过筛选数: {len(validated_basins)}")

    # ==================== 4. 保存结果（使用新文件名，不覆盖原文件） ====================
    # 输出到独立目录，不与代码混放
    out_dir = Path(__file__).parent / "output"
    out_dir.mkdir(parents=True, exist_ok=True)

    output_txt    = out_dir / "all_basins_valid30pct.txt"
    output_csv    = out_dir / "all_basins_valid30pct.csv"
    output_report = out_dir / "all_basins_valid30pct_report.txt"

    # TXT（Python 可直接 import 的格式）
    with open(output_txt, 'w', encoding='utf-8') as f:
        f.write(f"# CAMELSH 全量筛选：Q 和 h 有效比例均 >= {MIN_VALID_RATIO:.0%}\n")
        f.write(f"# 生成时间: {datetime.datetime.now()}\n")
        f.write(f"# 候选流域数: {len(all_candidates)}\n")
        f.write(f"# 通过筛选数: {len(validated_basins)}\n")
        f.write(f"# 数据时间范围: {default_range}\n\n")
        f.write("VALID_BASINS_30PCT = [\n")
        for i, basin_id in enumerate(validated_basins):
            comma = "," if i < len(validated_basins) - 1 else ""
            f.write(f"    '{basin_id}'{comma}\n")
        f.write("]\n")
    print(f"\n已保存流域列表（TXT）: {output_txt}")

    # CSV
    df = pd.DataFrame({
        'basin_id': validated_basins,
        'rank':     range(1, len(validated_basins) + 1)
    })
    df.to_csv(output_csv, index=False)
    print(f"已保存流域列表（CSV）: {output_csv}")

    # 文字报告
    with open(output_report, 'w', encoding='utf-8') as f:
        f.write("=" * 80 + "\n")
        f.write("CAMELSH 全量流域筛选报告\n")
        f.write("=" * 80 + "\n\n")
        f.write(f"生成时间     : {datetime.datetime.now()}\n")
        f.write(f"CAMELSH路径  : {CAMELSH_DATA_PATH}\n")
        f.write(f"数据时间范围 : {default_range}\n")
        f.write(f"有效比例阈值 : >= {MIN_VALID_RATIO:.0%}（Q 和 h 同时满足）\n\n")
        f.write(f"候选流域数   : {len(all_candidates)}\n")
        f.write(f"通过筛选数   : {len(validated_basins)}\n")
        f.write(f"筛选通过率   : {len(validated_basins)/len(all_candidates)*100:.1f}%\n\n")
        f.write("=" * 80 + "\n")
        f.write("通过筛选的流域列表\n")
        f.write("=" * 80 + "\n")
        for i, basin_id in enumerate(validated_basins, 1):
            f.write(f"  {i:4d}. {basin_id}\n")
        f.write("\n" + "=" * 80 + "\n")
    print(f"已保存筛选报告     : {output_report}")

    print(f"\n前10个流域: {validated_basins[:10]}")
    print(f"后10个流域: {validated_basins[-10:]}")

    print(f"\n{'='*80}")
    print("筛选完成!")
    print(f"{'='*80}")
    print(f"生成的文件（均不覆盖原有文件）:")
    print(f"  1. {output_txt}    - Python 可导入的流域列表")
    print(f"  2. {output_csv}    - CSV 格式流域列表")
    print(f"  3. {output_report} - 详细筛选报告")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n用户中断操作")
    except Exception as e:
        print(f"\n发生错误: {e}")
        import traceback
        traceback.print_exc()
