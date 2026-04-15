"""
数据准备脚本 - 将原始数据转换为模型需要的格式

从data/目录读取原始的flow_stage_DV.csv文件
生成两个标准格式的文件：
- flow_data.csv: 径流数据
- waterlevel_data.csv: 水位数据
"""

import pandas as pd
import os

print("=" * 60)
print("数据准备脚本")
print("=" * 60)

# 配置
DATA_DIR = "data"
BASINS = ["01121000", "01118300"]  # 流域ID列表

# 读取原始数据
print("\n步骤 1: 读取原始数据文件...")
dataframes = {}
for basin in BASINS:
    file_path = os.path.join(DATA_DIR, f"{basin}_flow_stage_DV.csv")
    print(f"  读取: {file_path}")
    df = pd.read_csv(file_path)
    dataframes[basin] = df

# 找出所有流域都有完整数据（径流+水位）的时间范围
print("\n步骤 2: 确定有效时间范围...")
valid_ranges = []
for basin, df in dataframes.items():
    df['datetime'] = pd.to_datetime(df['datetime'])
    valid_data = df[df['stage_m'].notna() & df['discharge_m3s'].notna()].copy()
    if len(valid_data) > 0:
        start_date = valid_data['datetime'].min()
        end_date = valid_data['datetime'].max()
        valid_ranges.append((start_date, end_date, len(valid_data)))
        print(f"  {basin}: {start_date.date()} 至 {end_date.date()} ({len(valid_data)} 天)")

# 使用所有流域都有数据的最大交集时间范围
common_start = max([r[0] for r in valid_ranges])
common_end = min([r[1] for r in valid_ranges])
print(f"\n  共同时间范围: {common_start.date()} 至 {common_end.date()}")

# 创建流量和水位的DataFrame
print("\n步骤 3: 提取径流和水位数据...")
flow_data = pd.DataFrame()
waterlevel_data = pd.DataFrame()

for basin, df in dataframes.items():
    df['datetime'] = pd.to_datetime(df['datetime'])
    
    # 筛选共同时间范围的数据
    mask = (df['datetime'] >= common_start) & (df['datetime'] <= common_end)
    df_filtered = df[mask].copy()
    
    # 确保日期唯一且排序
    df_filtered = df_filtered.sort_values('datetime')
    df_filtered = df_filtered.drop_duplicates(subset=['datetime'])
    
    # 设置日期为索引
    df_filtered.set_index('datetime', inplace=True)
    
    # 提取流量数据（转换为mm/day需要流域面积，这里先用m3/s）
    # 注意：如果需要转换为mm/day，需要流域面积信息
    flow_data[basin] = df_filtered['discharge_m3s']
    
    # 提取水位数据（米）
    waterlevel_data[basin] = df_filtered['stage_m']
    
    print(f"  {basin}: 提取了 {len(df_filtered)} 天的数据")

# 检查缺失值
print("\n步骤 4: 检查数据质量...")
flow_missing = flow_data.isna().sum()
waterlevel_missing = waterlevel_data.isna().sum()

print("  径流数据缺失值:")
for basin in BASINS:
    print(f"    {basin}: {flow_missing[basin]} 个 ({flow_missing[basin]/len(flow_data)*100:.2f}%)")

print("  水位数据缺失值:")
for basin in BASINS:
    print(f"    {basin}: {waterlevel_missing[basin]} 个 ({waterlevel_missing[basin]/len(waterlevel_data)*100:.2f}%)")

# 处理缺失值（可选：插值或删除）
if flow_data.isna().any().any() or waterlevel_data.isna().any().any():
    print("\n  处理缺失值（使用线性插值）...")
    flow_data = flow_data.interpolate(method='linear', limit_direction='both')
    waterlevel_data = waterlevel_data.interpolate(method='linear', limit_direction='both')
    
    # 如果还有缺失值，使用前后填充
    flow_data = flow_data.fillna(method='ffill').fillna(method='bfill')
    waterlevel_data = waterlevel_data.fillna(method='ffill').fillna(method='bfill')

# 重置索引，使日期成为一列
flow_data.reset_index(inplace=True)
waterlevel_data.reset_index(inplace=True)

# 重命名日期列
flow_data.rename(columns={'datetime': 'date'}, inplace=True)
waterlevel_data.rename(columns={'datetime': 'date'}, inplace=True)

# 只保留日期部分（去掉时间和时区）
flow_data['date'] = pd.to_datetime(flow_data['date']).dt.date
waterlevel_data['date'] = pd.to_datetime(waterlevel_data['date']).dt.date

# 保存为CSV文件
print("\n步骤 5: 保存处理后的数据...")
flow_data.to_csv('flow_data.csv', index=False)
waterlevel_data.to_csv('waterlevel_data.csv', index=False)

print(f"  [保存] flow_data.csv")
print(f"  [保存] waterlevel_data.csv")

# 显示数据统计
print("\n步骤 6: 数据统计摘要")
print("\n径流数据统计 (m3/s):")
print(flow_data[BASINS].describe())

print("\n水位数据统计 (m):")
print(waterlevel_data[BASINS].describe())

# 显示样本数据
print("\n径流数据样本 (前5行):")
print(flow_data.head())

print("\n水位数据样本 (前5行):")
print(waterlevel_data.head())

print("\n" + "=" * 60)
print("数据准备完成！")
print("=" * 60)
print(f"\n数据详情:")
print(f"  - 流域数量: {len(BASINS)}")
print(f"  - 流域ID: {', '.join(BASINS)}")
print(f"  - 时间范围: {flow_data['date'].iloc[0]} 至 {flow_data['date'].iloc[-1]}")
print(f"  - 总天数: {len(flow_data)}")
print(f"\n生成的文件:")
print(f"  - flow_data.csv: 径流数据（单位：m3/s）")
print(f"  - waterlevel_data.csv: 水位数据（单位：m）")
print(f"\n下一步:")
print(f"  运行训练脚本: python train_multi_task_model.py")

