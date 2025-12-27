"""
提取multi_task_lstm.py实际使用的流域ID和时间段
不执行训练，只提取配置信息
"""

import sys
from pathlib import Path
import pandas as pd
from hydrodataset import StandardVariable
from improved_camelsh_reader import ImprovedCAMELSHReader

# 从multi_task_lstm.py导入必要的函数和常量
from multi_task_lstm import (
    load_waterlevel_basins_from_file,
    filter_basins_with_valid_data,
    TRAIN_RATIO,
    VALID_RATIO,
    TEST_RATIO
)

# 导入配置
from config import (
    CAMELSH_DATA_PATH,
    NUM_BASINS,
    SEQUENCE_LENGTH,
    TRAIN_START, TRAIN_END,
    VALID_START, VALID_END,
    TEST_START, TEST_END
)


def extract_runtime_config():
    """提取运行时配置信息"""
    
    print("=" * 80)
    print("Multi-Task LSTM 实际运行配置提取")
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
        
        # 获取默认时间范围
        default_range = camelsh.default_t_range
        print(f"CAMELSH默认时间范围: {default_range}")
        
    except Exception as e:
        print(f"错误: 无法加载CAMELSH数据 - {e}")
        return
    
    # ==================== 3. 过滤有效流域 ====================
    print(f"\n[3] 验证流域数据有效性...")
    print(f"配置要求的流域数量: {NUM_BASINS}")
    print(f"将检查前 {min(len(VALID_WATER_LEVEL_BASINS), max(NUM_BASINS * 3, 200))} 个候选流域...")
    
    max_candidates = min(len(VALID_WATER_LEVEL_BASINS), max(NUM_BASINS * 3, 200))
    
    try:
        validated_basins = filter_basins_with_valid_data(
            camelsh_reader=camelsh_reader,
            basin_list=VALID_WATER_LEVEL_BASINS,
            time_range=default_range,
            max_basins_to_check=max_candidates,
            min_valid_ratio=0.1
        )
        
        if len(validated_basins) == 0:
            print("错误: 未找到任何有效流域！")
            return
        
        if len(validated_basins) < NUM_BASINS:
            print(f"\n警告: 只找到 {len(validated_basins)} 个有效流域，少于配置的 {NUM_BASINS} 个")
            chosen_basins = validated_basins
        else:
            chosen_basins = validated_basins[:NUM_BASINS]
        
    except Exception as e:
        print(f"错误: 验证流域失败 - {e}")
        return
    
    # ==================== 4. 计算时间划分 ====================
    print(f"\n[4] 计算时间划分...")
    
    # 将字符串时间转为pandas datetime
    start_time = pd.to_datetime(default_range[0])
    end_time = pd.to_datetime(default_range[1])
    total_duration = end_time - start_time
    
    # 按比例计算
    train_duration = total_duration * TRAIN_RATIO
    valid_duration = total_duration * VALID_RATIO
    test_duration = total_duration * TEST_RATIO
    
    train_start = start_time
    train_end = train_start + train_duration
    valid_start = train_end
    valid_end = valid_start + valid_duration
    test_start = valid_end
    test_end = end_time
    
    # ==================== 5. 输出结果 ====================
    print("\n" + "=" * 80)
    print("实际运行配置总结")
    print("=" * 80)
    
    print(f"\n【流域配置】")
    print(f"  配置要求: {NUM_BASINS} 个流域")
    print(f"  实际使用: {len(chosen_basins)} 个流域")
    print(f"  流域列表: {chosen_basins}")
    
    print(f"\n【时间配置】")
    print(f"  完整时间范围: {default_range[0]} 至 {default_range[1]}")
    print(f"  总时长: {total_duration.days} 天")
    
    print(f"\n  训练集 (Train):")
    print(f"    比例: {TRAIN_RATIO:.0%}")
    print(f"    时间段: {train_start.strftime('%Y-%m-%d')} 至 {train_end.strftime('%Y-%m-%d')}")
    print(f"    时长: {train_duration.days} 天")
    
    print(f"\n  验证集 (Valid):")
    print(f"    比例: {VALID_RATIO:.0%}")
    print(f"    时间段: {valid_start.strftime('%Y-%m-%d')} 至 {valid_end.strftime('%Y-%m-%d')}")
    print(f"    时长: {valid_duration.days} 天")
    
    print(f"\n  测试集 (Test):")
    print(f"    比例: {TEST_RATIO:.0%}")
    print(f"    时间段: {test_start.strftime('%Y-%m-%d')} 至 {test_end.strftime('%Y-%m-%d')}")
    print(f"    时长: {test_duration.days} 天")
    
    print(f"\n【模型配置】")
    print(f"  序列长度: {SEQUENCE_LENGTH} 步 (3小时分辨率)")
    print(f"  输入窗口时长: {SEQUENCE_LENGTH * 3} 小时 = {SEQUENCE_LENGTH * 3 / 24:.1f} 天")
    
    # ==================== 6. 保存到文件 ====================
    print(f"\n【保存配置文件】")
    
    # 保存流域列表
    basin_file = "runtime_basins.txt"
    with open(basin_file, 'w', encoding='utf-8') as f:
        f.write("# Multi-Task LSTM 实际使用的流域ID列表\n")
        f.write(f"# 生成时间: {pd.Timestamp.now()}\n")
        f.write(f"# 总数量: {len(chosen_basins)}\n\n")
        f.write("CHOSEN_BASINS = [\n")
        for i, basin in enumerate(chosen_basins):
            if i < len(chosen_basins) - 1:
                f.write(f"    '{basin}',\n")
            else:
                f.write(f"    '{basin}'\n")
        f.write("]\n")
    print(f"  流域列表已保存: {basin_file}")
    
    # 保存时间配置
    time_file = "runtime_time_config.txt"
    with open(time_file, 'w', encoding='utf-8') as f:
        f.write("# Multi-Task LSTM 实际使用的时间配置\n")
        f.write(f"# 生成时间: {pd.Timestamp.now()}\n\n")
        
        f.write("完整时间范围:\n")
        f.write(f"  开始: {default_range[0]}\n")
        f.write(f"  结束: {default_range[1]}\n")
        f.write(f"  总时长: {total_duration.days} 天\n\n")
        
        f.write("训练集 (Train):\n")
        f.write(f"  比例: {TRAIN_RATIO:.0%}\n")
        f.write(f"  开始: {train_start.strftime('%Y-%m-%d')}\n")
        f.write(f"  结束: {train_end.strftime('%Y-%m-%d')}\n")
        f.write(f"  时长: {train_duration.days} 天\n\n")
        
        f.write("验证集 (Valid):\n")
        f.write(f"  比例: {VALID_RATIO:.0%}\n")
        f.write(f"  开始: {valid_start.strftime('%Y-%m-%d')}\n")
        f.write(f"  结束: {valid_end.strftime('%Y-%m-%d')}\n")
        f.write(f"  时长: {valid_duration.days} 天\n\n")
        
        f.write("测试集 (Test):\n")
        f.write(f"  比例: {TEST_RATIO:.0%}\n")
        f.write(f"  开始: {test_start.strftime('%Y-%m-%d')}\n")
        f.write(f"  结束: {test_end.strftime('%Y-%m-%d')}\n")
        f.write(f"  时长: {test_duration.days} 天\n\n")
        
        f.write("模型配置:\n")
        f.write(f"  序列长度: {SEQUENCE_LENGTH} 步\n")
        f.write(f"  时间分辨率: 3 小时\n")
        f.write(f"  输入窗口: {SEQUENCE_LENGTH * 3} 小时 = {SEQUENCE_LENGTH * 3 / 24:.1f} 天\n")
    print(f"  时间配置已保存: {time_file}")
    
    # 保存为CSV格式（方便导入其他工具）
    csv_file = "runtime_basins.csv"
    basin_df = pd.DataFrame({
        'basin_id': chosen_basins,
        'index': range(len(chosen_basins))
    })
    basin_df.to_csv(csv_file, index=False)
    print(f"  流域列表CSV已保存: {csv_file}")
    
    time_df = pd.DataFrame({
        'dataset': ['Train', 'Valid', 'Test'],
        'start_date': [
            train_start.strftime('%Y-%m-%d'),
            valid_start.strftime('%Y-%m-%d'),
            test_start.strftime('%Y-%m-%d')
        ],
        'end_date': [
            train_end.strftime('%Y-%m-%d'),
            valid_end.strftime('%Y-%m-%d'),
            test_end.strftime('%Y-%m-%d')
        ],
        'duration_days': [
            train_duration.days,
            valid_duration.days,
            test_duration.days
        ],
        'ratio': [TRAIN_RATIO, VALID_RATIO, TEST_RATIO]
    })
    time_csv = "runtime_time_config.csv"
    time_df.to_csv(time_csv, index=False)
    print(f"  时间配置CSV已保存: {time_csv}")
    
    print("\n" + "=" * 80)
    print("配置提取完成！")
    print("=" * 80)
    
    return {
        'basins': chosen_basins,
        'time_range': default_range,
        'train_period': (train_start, train_end),
        'valid_period': (valid_start, valid_end),
        'test_period': (test_start, test_end)
    }


if __name__ == "__main__":
    try:
        config = extract_runtime_config()
    except Exception as e:
        print(f"\n发生错误: {e}")
        import traceback
        traceback.print_exc()

