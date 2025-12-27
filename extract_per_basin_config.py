"""
提取multi_task_lstm.py实际使用的流域ID和每个流域各自的时间段
不执行训练，只提取配置信息
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
from tqdm import tqdm
from hydrodataset import StandardVariable
from improved_camelsh_reader import ImprovedCAMELSHReader

# 从multi_task_lstm.py导入必要的函数和常量
from multi_task_lstm import (
    load_waterlevel_basins_from_file,
    filter_basins_with_valid_data,
    TRAIN_RATIO,
    VALID_RATIO,
    TEST_RATIO,
    WINDOW_STEP
)

# 导入配置
from config import (
    CAMELSH_DATA_PATH,
    NUM_BASINS,
    SEQUENCE_LENGTH,
    FORCING_VARIABLES,
)

from mswep_loader import load_mswep_data, merge_forcing_with_mswep


def extract_per_basin_time_ranges(camelsh_reader, chosen_basins, default_range, forcings_ds, full_flow, full_waterlevel):
    """
    为每个流域提取其独立的时间划分
    
    模拟 _create_lookup_table 的逻辑，但只提取时间范围信息，不创建样本
    """
    
    print("\n" + "=" * 80)
    print("提取每个流域的独立时间划分")
    print("=" * 80)
    
    # 强迫数据的时间索引
    forcing_time_index = pd.DatetimeIndex(forcings_ds.time.values)
    target_index = full_flow.index
    
    basin_time_info = []
    
    for basin in tqdm(chosen_basins, desc="分析流域时间范围"):
        basin_info = {'basin_id': basin}
        
        try:
            # 找出flow和waterlevel都非NaN的位置
            flow_values = full_flow[basin].values
            wl_values = full_waterlevel[basin].values
            
            flow_valid_mask = ~np.isnan(flow_values)
            wl_valid_mask = ~np.isnan(wl_values)
            both_valid = flow_valid_mask & wl_valid_mask
            
            # 获取目标数据中所有有效的时间索引
            valid_target_times = set(target_index[both_valid])
            
            if len(valid_target_times) == 0:
                basin_info['status'] = 'no_valid_data'
                basin_time_info.append(basin_info)
                continue
            
            # 找出强迫数据时间点中，对应的目标数据也有效的时间点
            valid_forcing_times = []
            for ft in forcing_time_index:
                # 计算序列结束时刻
                end_time = ft + pd.Timedelta(hours=(SEQUENCE_LENGTH - 1) * 3)
                # 检查结束时刻的目标数据是否有效
                if end_time in valid_target_times:
                    valid_forcing_times.append(ft)
            
            if len(valid_forcing_times) < 1:
                basin_info['status'] = 'insufficient_data'
                basin_time_info.append(basin_info)
                continue
            
            # 找出有效时间的起止
            first_valid_time = valid_forcing_times[0]
            last_valid_time = valid_forcing_times[-1]
            
            # 在强迫数据时间索引中找到位置
            start_idx = forcing_time_index.get_loc(first_valid_time)
            end_idx = forcing_time_index.get_loc(last_valid_time)
            
            # 计算时间跨度
            total_span = end_idx - start_idx + 1
            
            # 按比例划分（与 _create_lookup_table 完全一致）
            train_end_idx = start_idx + int(total_span * TRAIN_RATIO)
            valid_end_idx = start_idx + int(total_span * (TRAIN_RATIO + VALID_RATIO))
            
            # 记录时间范围
            basin_info['status'] = 'success'
            basin_info['first_valid_time'] = str(first_valid_time)
            basin_info['last_valid_time'] = str(last_valid_time)
            basin_info['total_valid_steps'] = len(valid_forcing_times)
            basin_info['total_span_steps'] = total_span
            
            # 训练集
            basin_info['train_start'] = str(forcing_time_index[start_idx])
            basin_info['train_end'] = str(forcing_time_index[train_end_idx - 1]) if train_end_idx > start_idx else str(forcing_time_index[start_idx])
            basin_info['train_steps'] = train_end_idx - start_idx
            
            # 验证集
            basin_info['valid_start'] = str(forcing_time_index[train_end_idx])
            basin_info['valid_end'] = str(forcing_time_index[valid_end_idx - 1]) if valid_end_idx > train_end_idx else str(forcing_time_index[train_end_idx])
            basin_info['valid_steps'] = valid_end_idx - train_end_idx
            
            # 测试集
            basin_info['test_start'] = str(forcing_time_index[valid_end_idx])
            basin_info['test_end'] = str(forcing_time_index[end_idx])
            basin_info['test_steps'] = end_idx - valid_end_idx + 1
            
        except Exception as e:
            basin_info['status'] = f'error: {str(e)[:50]}'
        
        basin_time_info.append(basin_info)
    
    return pd.DataFrame(basin_time_info)


def extract_runtime_config():
    """提取运行时配置信息"""
    
    print("=" * 80)
    print("Multi-Task LSTM 实际运行配置提取（每个流域独立时间划分）")
    print("=" * 80)
    
    # ==================== 1. 读取流域列表 ====================
    print("\n[1] 读取流域列表...")
    try:
        VALID_WATER_LEVEL_BASINS = load_waterlevel_basins_from_file("valid_waterlevel_basins.txt")
        print(f"从文件读取了 {len(VALID_WATER_LEVEL_BASINS)} 个候选流域")
    except Exception as e:
        print(f"错误: 无法读取流域列表 - {e}")
        return
    
    # ==================== 2. 加载CAMELSH数据读取器 ====================
    print(f"\n[2] 加载CAMELSH数据读取器...")
    print(f"数据路径: {CAMELSH_DATA_PATH}")
    
    try:
        camelsh_reader = ImprovedCAMELSHReader(CAMELSH_DATA_PATH, download=False, use_batch=True)
        camelsh = camelsh_reader.camelsh
        default_range = camelsh.default_t_range
        print(f"CAMELSH默认时间范围: {default_range}")
    except Exception as e:
        print(f"错误: 无法加载CAMELSH数据 - {e}")
        return
    
    # ==================== 3. 过滤有效流域 ====================
    print(f"\n[3] 验证流域数据有效性...")
    max_candidates = min(len(VALID_WATER_LEVEL_BASINS), max(NUM_BASINS * 3, 200))
    
    try:
        validated_basins = filter_basins_with_valid_data(
            camelsh_reader=camelsh_reader,
            basin_list=VALID_WATER_LEVEL_BASINS,
            time_range=default_range,
            max_basins_to_check=max_candidates,
            min_valid_ratio=0.1
        )
        
        if len(validated_basins) < NUM_BASINS:
            chosen_basins = validated_basins
        else:
            chosen_basins = validated_basins[:NUM_BASINS]
        
        print(f"\n最终选择: {len(chosen_basins)} 个流域")
    except Exception as e:
        print(f"错误: 验证流域失败 - {e}")
        return
    
    # ==================== 4. 加载数据以模拟真实划分 ====================
    print(f"\n[4] 加载数据以提取每个流域的实际时间划分...")
    print("这可能需要几分钟...")
    
    try:
        # 加载气象数据
        print("  加载气象数据...")
        from hydrodataset import StandardVariable
        
        # 从配置文件读取需要的变量
        chosen_forcing_vars = []
        for var_name in FORCING_VARIABLES:
            if hasattr(StandardVariable, var_name.upper()):
                chosen_forcing_vars.append(getattr(StandardVariable, var_name.upper()))
            else:
                chosen_forcing_vars.append(var_name)
        
        # 分离降雨和其他气象变量
        chosen_forcing_vars_no_precip = [v for v in chosen_forcing_vars 
                                          if v != StandardVariable.PRECIPITATION]
        
        print(f"    非降雨变量: {[str(v) for v in chosen_forcing_vars_no_precip]}")
        
        forcings_ds_no_precip = camelsh_reader.read_ts_xrdataset(
            gage_id_lst=chosen_basins,
            t_range=default_range,
            var_lst=chosen_forcing_vars_no_precip
        )
        
        mswep_precip_df = load_mswep_data(
            file_path="MSWEP/mswep_220basins_mean_3hourly_1980_2024.csv",
            basin_ids=chosen_basins,
            time_range=default_range
        )
        
        forcings_ds = merge_forcing_with_mswep(forcings_ds_no_precip, mswep_precip_df)
        print(f"  气象数据形状: {forcings_ds.dims}")
        
        # 加载径流和水位数据
        print("  加载径流数据...")
        flow_ds = camelsh_reader.read_ts_xrdataset(
            gage_id_lst=chosen_basins,
            t_range=default_range,
            var_lst=[StandardVariable.STREAMFLOW]
        )
        
        print("  加载水位数据...")
        waterlevel_ds = camelsh_reader.read_ts_xrdataset(
            gage_id_lst=chosen_basins,
            t_range=default_range,
            var_lst=[StandardVariable.WATER_LEVEL]
        )
        
        full_flow = flow_ds[StandardVariable.STREAMFLOW].to_pandas().T
        full_waterlevel = waterlevel_ds[StandardVariable.WATER_LEVEL].to_pandas().T
        
        print(f"  径流数据形状: {full_flow.shape}")
        print(f"  水位数据形状: {full_waterlevel.shape}")
        
    except Exception as e:
        print(f"错误: 加载数据失败 - {e}")
        import traceback
        traceback.print_exc()
        return
    
    # ==================== 5. 提取每个流域的时间划分 ====================
    print(f"\n[5] 提取每个流域的独立时间划分...")
    
    basin_time_df = extract_per_basin_time_ranges(
        camelsh_reader, chosen_basins, default_range,
        forcings_ds, full_flow, full_waterlevel
    )
    
    # ==================== 6. 统计和输出 ====================
    print("\n" + "=" * 80)
    print("提取结果统计")
    print("=" * 80)
    
    success_basins = basin_time_df[basin_time_df['status'] == 'success']
    print(f"\n成功提取时间划分的流域: {len(success_basins)}/{len(chosen_basins)}")
    
    if len(success_basins) > 0:
        print(f"\n时间跨度统计:")
        print(f"  平均总步数: {success_basins['total_span_steps'].mean():.0f} 步")
        print(f"  平均训练步数: {success_basins['train_steps'].mean():.0f} 步")
        print(f"  平均验证步数: {success_basins['valid_steps'].mean():.0f} 步")
        print(f"  平均测试步数: {success_basins['test_steps'].mean():.0f} 步")
        
        print(f"\n时间范围示例（前5个流域）:")
        for idx, row in success_basins.head(5).iterrows():
            print(f"\n  流域 {row['basin_id']}:")
            print(f"    完整范围: {row['first_valid_time']} 至 {row['last_valid_time']}")
            print(f"    训练集: {row['train_start']} 至 {row['train_end']} ({row['train_steps']}步)")
            print(f"    验证集: {row['valid_start']} 至 {row['valid_end']} ({row['valid_steps']}步)")
            print(f"    测试集: {row['test_start']} 至 {row['test_end']} ({row['test_steps']}步)")
    
    # ==================== 7. 保存结果 ====================
    print(f"\n[6] 保存结果...")
    
    # 保存完整的流域时间划分表
    output_file = "runtime_basin_time_ranges.csv"
    basin_time_df.to_csv(output_file, index=False)
    print(f"  完整时间划分表已保存: {output_file}")
    
    # 保存只包含成功流域的简化版本
    if len(success_basins) > 0:
        simple_output = "runtime_basin_time_ranges_simple.csv"
        success_basins[['basin_id', 'train_start', 'train_end', 'valid_start', 
                       'valid_end', 'test_start', 'test_end']].to_csv(simple_output, index=False)
        print(f"  简化版时间划分表已保存: {simple_output}")
        
        # 保存文本格式的详细报告
        report_file = "runtime_basin_time_ranges_report.txt"
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write("=" * 80 + "\n")
            f.write("Multi-Task LSTM 每个流域的独立时间划分\n")
            f.write("=" * 80 + "\n\n")
            f.write(f"生成时间: {pd.Timestamp.now()}\n")
            f.write(f"流域总数: {len(chosen_basins)}\n")
            f.write(f"成功提取: {len(success_basins)}\n\n")
            
            f.write("时间划分说明:\n")
            f.write("  每个流域使用其自身的有效数据时间范围进行划分\n")
            f.write(f"  训练集比例: {TRAIN_RATIO:.0%}\n")
            f.write(f"  验证集比例: {VALID_RATIO:.0%}\n")
            f.write(f"  测试集比例: {TEST_RATIO:.0%}\n\n")
            
            f.write("=" * 80 + "\n")
            f.write("各流域详细时间划分\n")
            f.write("=" * 80 + "\n\n")
            
            for idx, row in success_basins.iterrows():
                f.write(f"流域 {row['basin_id']}:\n")
                f.write(f"  完整有效范围: {row['first_valid_time']} 至 {row['last_valid_time']}\n")
                f.write(f"  总有效步数: {row['total_valid_steps']} 步\n\n")
                f.write(f"  训练集:\n")
                f.write(f"    时间: {row['train_start']} 至 {row['train_end']}\n")
                f.write(f"    步数: {row['train_steps']} 步\n\n")
                f.write(f"  验证集:\n")
                f.write(f"    时间: {row['valid_start']} 至 {row['valid_end']}\n")
                f.write(f"    步数: {row['valid_steps']} 步\n\n")
                f.write(f"  测试集:\n")
                f.write(f"    时间: {row['test_start']} 至 {row['test_end']}\n")
                f.write(f"    步数: {row['test_steps']} 步\n\n")
                f.write("-" * 80 + "\n\n")
        
        print(f"  详细报告已保存: {report_file}")
    
    print("\n" + "=" * 80)
    print("配置提取完成！")
    print("=" * 80)
    print("\n重要说明:")
    print("  每个流域使用其自身的有效数据时间范围独立划分")
    print("  不同流域的训练/验证/测试时间段可能不同")
    print("  这是代码的设计行为，以适应不同流域的数据可用性")
    
    return basin_time_df


if __name__ == "__main__":
    try:
        config = extract_runtime_config()
    except Exception as e:
        print(f"\n发生错误: {e}")
        import traceback
        traceback.print_exc()

