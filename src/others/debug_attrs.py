from config import *
from improved_camelsh_reader import ImprovedCAMELSHReader
import pandas as pd
import numpy as np

camelsh_reader = ImprovedCAMELSHReader(CAMELSH_DATA_PATH, download=False)
chosen_basins = VALID_WATER_LEVEL_BASINS[:4]

# 读取属性数据
attrs = camelsh_reader.read_attr_xrdataset(
    gage_id_lst=chosen_basins, 
    var_lst=ATTRIBUTE_VARIABLES
)
attrs_df = attrs.to_dataframe().reset_index()
print("原始属性数据:")
print(attrs_df)
print("列:", list(attrs_df.columns))
print("数据类型:", attrs_df.dtypes)

# 确保有gauge_id列用于MultiTaskDataset
if 'basin' in attrs_df.columns:
    attrs_df['gauge_id'] = attrs_df['basin']
    attrs_df = attrs_df.set_index('basin')

print("\n设置索引后:")
print(attrs_df)
print("列:", list(attrs_df.columns))

# 为单任务模型创建只有数值列的版本
attrs_df_numeric = attrs_df.select_dtypes(include=[np.number])
print("\n数值列版本:")
print(attrs_df_numeric)
print("列:", list(attrs_df_numeric.columns))

# 为多任务模型保留gauge_id列，但只包含数值属性
attrs_df_multi = attrs_df_numeric.reset_index()
if 'gauge_id' not in attrs_df_multi.columns and 'basin' in attrs_df_multi.columns:
    attrs_df_multi['gauge_id'] = attrs_df_multi['basin']

print("\n多任务版本:")
print(attrs_df_multi)
print("列:", list(attrs_df_multi.columns))
print("数据类型:", attrs_df_multi.dtypes)

# 检查NaN
print("\nNaN检查:")
print("是否有NaN:", attrs_df_multi.isna().any().any())
print("每列NaN数量:")
print(attrs_df_multi.isna().sum())





