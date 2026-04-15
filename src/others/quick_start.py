"""
快速入门脚本 - 多任务LSTM模型测试

这个脚本会：
1. 自动生成示例数据
2. 运行一个小规模的训练测试（2个epoch）
3. 验证模型是否正常工作

适合用于：
- 首次运行前的环境测试
- 验证代码是否正常工作
- 理解整个流程
"""

import os
import sys
from pathlib import Path

print("=" * 60)
print("多任务LSTM模型 - 快速入门测试")
print("=" * 60)

# 检查依赖包
print("\n步骤 1/5: 检查依赖包...")
required_packages = [
    'numpy', 'pandas', 'xarray', 'torch', 
    'tqdm', 'matplotlib', 'hydrodataset', 'HydroErr'
]

missing_packages = []
for package in required_packages:
    try:
        __import__(package if package != 'HydroErr' else 'HydroErr')
        print(f"  ✓ {package}")
    except ImportError:
        print(f"  ✗ {package} (缺失)")
        missing_packages.append(package)

if missing_packages:
    print(f"\n错误：缺少以下依赖包：{', '.join(missing_packages)}")
    print(f"请运行: pip install {' '.join(missing_packages)}")
    sys.exit(1)

print("  所有依赖包已安装！")

# 检查CAMELS数据
print("\n步骤 2/5: 检查CAMELS数据...")
camels_path = Path("camels/camels_us")
if not camels_path.exists():
    print(f"  ✗ CAMELS数据路径不存在: {camels_path}")
    print(f"  请下载CAMELS数据集并放置在正确位置")
    sys.exit(1)

try:
    from hydrodataset.camelsh import Camelsh
    # 指定CAMELSH数据路径
    camelsh_data_path = "camelsh_data"  # 修改为您的实际数据路径
    camelsh = Camelsh(camelsh_data_path, download=False)
    print(f"  ✓ CAMELSH数据加载成功")
    print(f"  数据路径: {camelsh.data_source_dir}")
    
    # 获取流域列表
    basin_ids = camelsh.read_object_ids()
    basins = basin_ids[:2].tolist()
    print(f"  测试流域: {basins}")
except Exception as e:
    print(f"  ✗ CAMELS数据加载失败: {e}")
    sys.exit(1)

# 生成示例数据
print("\n步骤 3/5: 生成示例数据...")
try:
    from create_sample_data import generate_sample_data
    
    generate_sample_data(
        basins=basins,
        start_date="1990-09-01",
        end_date="2010-08-31",
        output_flow="flow_data.csv",
        output_waterlevel="waterlevel_data.csv"
    )
    print("  ✓ 示例数据生成成功")
except Exception as e:
    print(f"  ✗ 示例数据生成失败: {e}")
    sys.exit(1)

# 运行快速测试
print("\n步骤 4/5: 运行模型训练测试...")
print("  (使用小规模参数进行快速测试)")

try:
    import random
    import numpy as np
    import pandas as pd
    import xarray as xr
    import torch
    from torch.utils.data import DataLoader
    import torch.nn as nn
    from tqdm import tqdm
    
    # 导入模型类
    from multi_task_lstm import (
        MultiTaskDataset, 
        MultiTaskLSTM, 
        train_epoch, 
        eval_model,
        set_random_seed,
        DEVICE
    )
    import HydroErr as he
    
    # 设置随机种子
    set_random_seed(1234)
    
    # 导入标准变量
    from hydrodataset import StandardVariable
    
    # 选择变量
    chosen_forcing_vars = [
        StandardVariable.PRECIPITATION,
        StandardVariable.TEMPERATURE_MEAN,
        StandardVariable.SOLAR_RADIATION,
        StandardVariable.POTENTIAL_EVAPOTRANSPIRATION
    ]
    chosen_attrs_vars = [
        "p_mean", "p_seasonality", "frac_snow", "aridity", "area"
    ]
    
    # 加载属性数据
    attrs = camelsh.read_attr_xrdataset(
        gage_id_lst=basins,
        var_lst=chosen_attrs_vars
    )
    print(f"  属性数据形状: {attrs.dims}")
    
    # 准备气象数据（使用较短的时间范围进行快速测试）
    train_times = ["2010-01-01", "2012-12-31"]
    valid_times = ["2013-01-01", "2014-12-31"]
    
    # 使用CAMELSH接口加载气象数据
    train_forcings = camelsh.read_ts_xrdataset(
        gage_id_lst=basins,
        t_range=train_times,
        var_lst=chosen_forcing_vars
    )
    valid_forcings = camelsh.read_ts_xrdataset(
        gage_id_lst=basins,
        t_range=valid_times,
        var_lst=chosen_forcing_vars
    )
    
    # 加载径流和水位数据
    train_flow_ds = camelsh.read_ts_xrdataset(
        gage_id_lst=basins,
        t_range=train_times,
        var_lst=[StandardVariable.STREAMFLOW]
    )
    train_waterlevel_ds = camelsh.read_ts_xrdataset(
        gage_id_lst=basins,
        t_range=train_times,
        var_lst=[StandardVariable.WATER_LEVEL]
    )
    valid_flow_ds = camelsh.read_ts_xrdataset(
        gage_id_lst=basins,
        t_range=valid_times,
        var_lst=[StandardVariable.STREAMFLOW]
    )
    valid_waterlevel_ds = camelsh.read_ts_xrdataset(
        gage_id_lst=basins,
        t_range=valid_times,
        var_lst=[StandardVariable.WATER_LEVEL]
    )
    
    # 转换为pandas DataFrame格式
    train_flow = train_flow_ds[StandardVariable.STREAMFLOW].to_pandas().T
    train_waterlevel = train_waterlevel_ds[StandardVariable.WATER_LEVEL].to_pandas().T
    valid_flow = valid_flow_ds[StandardVariable.STREAMFLOW].to_pandas().T
    valid_waterlevel = valid_waterlevel_ds[StandardVariable.WATER_LEVEL].to_pandas().T
    
    # 创建数据集（使用小参数）
    sequence_length = 50  # 较短的序列
    batch_size = 16       # 较小的批次
    
    ds_train = MultiTaskDataset(
        basins=basins,
        dates=train_times,
        data_attr=chosen_attrs,
        data_forcing=train_forcings,
        data_flow=train_flow,
        data_waterlevel=train_waterlevel,
        loader_type="train",
        seq_length=sequence_length,
    )
    tr_loader = DataLoader(ds_train, batch_size=batch_size, shuffle=True)
    
    means = ds_train.get_means()
    stds = ds_train.get_stds()
    
    ds_val = MultiTaskDataset(
        basins=basins,
        dates=valid_times,
        data_attr=chosen_attrs,
        data_forcing=valid_forcings,
        data_flow=valid_flow,
        data_waterlevel=valid_waterlevel,
        loader_type="valid",
        seq_length=sequence_length,
        means=means,
        stds=stds,
    )
    val_loader = DataLoader(ds_val, batch_size=100, shuffle=False)
    
    # 创建模型（使用小参数）
    input_size = len(chosen_attrs_vars) + len(chosen_forcing_vars)
    hidden_size = 32      # 较小的隐藏层
    dropout_rate = 0.1
    learning_rate = 1e-3
    
    task_weights = {'flow': 1.0, 'waterlevel': 1.0}
    
    model = MultiTaskLSTM(
        input_size=input_size,
        hidden_size=hidden_size,
        dropout_rate=dropout_rate,
        task_weights=task_weights
    ).to(DEVICE)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    loss_func = nn.MSELoss()
    
    print(f"  模型参数量: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")
    print(f"  训练样本数: {len(ds_train)}")
    print(f"  验证样本数: {len(ds_val)}")
    
    # 快速训练（仅2个epoch）
    n_epochs = 2
    print(f"\n  开始训练 ({n_epochs} epochs)...")
    
    for i in range(n_epochs):
        train_loss, _, _ = train_epoch(model, optimizer, tr_loader, loss_func, i + 1)
        
        # 验证
        obs_flow, obs_waterlevel, preds_flow, preds_waterlevel = eval_model(model, val_loader)
        
        preds_flow = ds_val.local_denormalization(preds_flow.cpu().numpy(), variable="flow")
        preds_waterlevel = ds_val.local_denormalization(preds_waterlevel.cpu().numpy(), variable="waterlevel")
        
        obs_flow = obs_flow.numpy().reshape(len(basins), -1)
        obs_waterlevel = obs_waterlevel.numpy().reshape(len(basins), -1)
        preds_flow = preds_flow.reshape(len(basins), -1)
        preds_waterlevel = preds_waterlevel.reshape(len(basins), -1)
        
        nse_flow = np.array([he.nse(preds_flow[j], obs_flow[j]) for j in range(len(basins))])
        nse_waterlevel = np.array([he.nse(preds_waterlevel[j], obs_waterlevel[j]) for j in range(len(basins))])
        
        print(f"  Epoch {i+1}: NSE(径流)={nse_flow.mean():.4f}, NSE(水位)={nse_waterlevel.mean():.4f}")
    
    print("\n  ✓ 模型训练测试成功！")
    
except Exception as e:
    print(f"\n  ✗ 模型训练测试失败: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# 总结
print("\n步骤 5/5: 测试完成总结")
print("=" * 60)
print("✓ 所有测试通过！")
print("\n环境配置正确，你可以：")
print("  1. 准备你的真实径流和水位数据")
print("     - 参考 flow_data.csv 和 waterlevel_data.csv 的格式")
print("  2. 运行完整训练：python multi_task_lstm.py")
print("  3. 查看详细文档：MULTI_TASK_README.md")
print("\n提示：")
print("  - 示例数据仅用于测试，请使用真实数据进行实际训练")
print("  - 完整训练建议使用更大的hidden_size和更多epochs")
print("=" * 60)
print("\n祝训练顺利！🎉")


