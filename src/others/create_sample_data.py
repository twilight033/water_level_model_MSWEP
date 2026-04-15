"""
生成示例数据文件（径流和水位数据）

此脚本用于生成示例格式的CSV文件，展示如何准备自定义的径流和水位数据。
在实际使用时，请将此脚本生成的数据替换为你的真实观测数据。
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta

def generate_sample_data(
    basins, 
    start_date="1990-09-01", 
    end_date="2010-08-31",
    output_flow="flow_data.csv",
    output_waterlevel="waterlevel_data.csv"
):
    """
    生成示例数据文件
    
    Parameters
    ----------
    basins : list
        流域ID列表
    start_date : str
        开始日期
    end_date : str
        结束日期
    output_flow : str
        径流数据输出文件名
    output_waterlevel : str
        水位数据输出文件名
    """
    
    # 创建日期范围
    date_range = pd.date_range(start=start_date, end=end_date, freq='D')
    n_days = len(date_range)
    
    print(f"生成数据: {n_days} 天，{len(basins)} 个流域")
    
    # 创建DataFrame
    flow_data = pd.DataFrame(index=date_range)
    waterlevel_data = pd.DataFrame(index=date_range)
    
    # 为每个流域生成模拟数据
    for i, basin in enumerate(basins):
        # 生成径流数据（模拟季节性变化 + 随机噪声）
        # 基础值 + 年度周期 + 随机波动
        t = np.arange(n_days)
        base_flow = 20 + 10 * i  # 不同流域有不同的基础流量
        seasonal = 15 * np.sin(2 * np.pi * t / 365.25)  # 年度季节性
        noise = np.random.randn(n_days) * 3  # 随机噪声
        flow = base_flow + seasonal + noise
        flow = np.maximum(flow, 0.1)  # 确保非负
        
        flow_data[basin] = flow
        
        # 生成水位数据（与径流相关，但有不同的尺度）
        # 水位 = f(径流) + 噪声
        # 使用幂律关系模拟流量-水位关系
        base_level = 2 + 0.5 * i  # 不同流域有不同的基准水位
        waterlevel = base_level + 0.1 * np.power(flow, 0.6)  # 幂律关系
        waterlevel += np.random.randn(n_days) * 0.05  # 小噪声
        waterlevel = np.maximum(waterlevel, 0.1)  # 确保非负
        
        waterlevel_data[basin] = waterlevel
    
    # 保存为CSV文件
    flow_data.index.name = 'date'
    waterlevel_data.index.name = 'date'
    
    flow_data.to_csv(output_flow)
    waterlevel_data.to_csv(output_waterlevel)
    
    print(f"\n已生成文件：")
    print(f"  - {output_flow}")
    print(f"  - {output_waterlevel}")
    
    # 显示数据统计信息
    print(f"\n径流数据统计：")
    print(flow_data.describe())
    print(f"\n水位数据统计：")
    print(waterlevel_data.describe())
    
    # 显示文件格式示例
    print(f"\n文件格式示例（前5行）：")
    print(f"\n{output_flow}:")
    print(flow_data.head())
    print(f"\n{output_waterlevel}:")
    print(waterlevel_data.head())


if __name__ == "__main__":
    # 示例：为CAMELS数据集的前2个流域生成数据
    # 在实际使用时，请根据你的流域ID修改
    
    from hydrodataset.camelsh import Camelsh
    from hydrodataset import StandardVariable
    import os
    
    # 加载CAMELSH数据获取流域ID
    try:
        # 指定CAMELSH数据路径
        camelsh_data_path = "camelsh_data"  # 修改为您的实际数据路径
        camelsh = Camelsh(camelsh_data_path, download=False)
        basin_ids = camelsh.read_object_ids()
        basins = basin_ids[:2].tolist()
        print(f"使用流域ID: {basins}")
    except:
        # 如果无法加载CAMELSH，使用默认流域ID
        print("无法加载CAMELSH数据，使用默认流域ID")
        basins = ["01013500", "01022500"]
    
    # 生成示例数据
    generate_sample_data(
        basins=basins,
        start_date="1990-09-01",
        end_date="2010-08-31",
        output_flow="flow_data.csv",
        output_waterlevel="waterlevel_data.csv"
    )
    
    print("\n注意：这些是模拟数据，仅用于演示格式。")
    print("在实际应用中，请使用你的真实观测数据替换这些文件。")


