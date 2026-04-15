from config import *
from improved_camelsh_reader import ImprovedCAMELSHReader
import pandas as pd

camelsh_reader = ImprovedCAMELSHReader(CAMELSH_DATA_PATH, download=False)
chosen_basins = VALID_WATER_LEVEL_BASINS[:4]

print("选择的流域:", chosen_basins)
print("强迫变量:", FORCING_VARIABLES)
print("属性变量:", ATTRIBUTE_VARIABLES)

# 读取属性数据
try:
    attrs_data = camelsh_reader.read_attr_xrdataset(gage_id_lst=chosen_basins, var_lst=ATTRIBUTE_VARIABLES)
    attrs_df = attrs_data.to_dataframe().reset_index()
    print('属性数据列:', list(attrs_df.columns))
    print('属性数据形状:', attrs_df.shape)
    print('属性变量数量:', len(ATTRIBUTE_VARIABLES))
    print('强迫变量数量:', len(FORCING_VARIABLES))
    print('预期总维度:', len(FORCING_VARIABLES) + len(ATTRIBUTE_VARIABLES))
    
    # 检查单个流域的属性
    if 'basin' in attrs_df.columns:
        attrs_df = attrs_df.set_index('basin')
    
    for basin in chosen_basins:
        if basin in attrs_df.index:
            attrs = attrs_df.loc[basin].values
            print(f'流域 {basin} 属性维度:', attrs.shape)
        else:
            print(f'流域 {basin} 没有属性数据')
            
except Exception as e:
    print('错误:', e)
    import traceback
    traceback.print_exc()





