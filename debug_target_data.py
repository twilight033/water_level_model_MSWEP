from config import *
from improved_camelsh_reader import ImprovedCAMELSHReader
from hydrodataset import StandardVariable
import pandas as pd
import numpy as np

camelsh_reader = ImprovedCAMELSHReader(CAMELSH_DATA_PATH, download=False)
chosen_basins = VALID_WATER_LEVEL_BASINS[:4]

# 读取流量数据
flow_data = camelsh_reader.read_ts_xrdataset(
    gage_id_lst=chosen_basins,
    var_lst=[StandardVariable.STREAMFLOW],
    t_range=["2014-01-01", "2016-12-31"]
)
flow_df = flow_data[StandardVariable.STREAMFLOW].to_dataframe().unstack('basin')[StandardVariable.STREAMFLOW]

print("流量数据统计:")
print(flow_df.describe())
print("\n各流域流量数据范围:")
for basin in chosen_basins:
    if basin in flow_df.columns:
        data = flow_df[basin].dropna()
        print(f"流域 {basin}: 最小值={data.min():.4f}, 最大值={data.max():.4f}, 均值={data.mean():.4f}, 标准差={data.std():.4f}")
        print(f"  前10个值: {data.head(10).values}")
        print(f"  NaN数量: {flow_df[basin].isna().sum()}")
    else:
        print(f"流域 {basin}: 无数据")

# 检查归一化后的效果
print("\n=== 归一化测试 ===")
for basin in chosen_basins:
    if basin in flow_df.columns:
        data = flow_df[basin].dropna()
        if len(data) > 0:
            mean_val = data.mean()
            std_val = data.std()
            normalized = (data - mean_val) / std_val
            print(f"流域 {basin}:")
            print(f"  原始数据范围: [{data.min():.4f}, {data.max():.4f}]")
            print(f"  归一化参数: mean={mean_val:.4f}, std={std_val:.4f}")
            print(f"  归一化后范围: [{normalized.min():.4f}, {normalized.max():.4f}]")
            print(f"  归一化后前10个值: {normalized.head(10).values}")
            
            # 反归一化测试
            denormalized = normalized * std_val + mean_val
            print(f"  反归一化后前10个值: {denormalized.head(10).values}")
            print(f"  反归一化误差: {np.abs(denormalized - data).max():.10f}")





