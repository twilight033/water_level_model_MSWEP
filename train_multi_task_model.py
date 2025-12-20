"""
多任务LSTM训练脚本 - 适配用户数据

使用准备好的flow_data.csv和waterlevel_data.csv进行训练
"""

import os
import random
import numpy as np
import pandas as pd
import xarray as xr
from matplotlib import pyplot as plt
from matplotlib import font_manager
import matplotlib
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import HydroErr as he

# 配置matplotlib中文字体
def setup_chinese_font():
    """设置matplotlib中文字体支持"""
    try:
        # 尝试使用系统中文字体
        chinese_fonts = [
            'SimHei',           # 黑体 (Windows)
            'Microsoft YaHei',  # 微软雅黑 (Windows)
            'DejaVu Sans',      # Linux
            'Arial Unicode MS', # macOS
            'PingFang SC',      # macOS
            'Hiragino Sans GB', # macOS
            'WenQuanYi Micro Hei', # Linux
        ]
        
        for font_name in chinese_fonts:
            try:
                # 检查字体是否可用
                font_path = font_manager.findfont(font_manager.FontProperties(family=font_name))
                if font_path and font_name.lower() in font_path.lower():
                    matplotlib.rcParams['font.sans-serif'] = [font_name]
                    matplotlib.rcParams['axes.unicode_minus'] = False
                    print(f"  ✓ 已设置中文字体: {font_name}")
                    return True
            except:
                continue
        
        # 如果没有找到合适的中文字体，使用默认字体并禁用unicode minus
        matplotlib.rcParams['axes.unicode_minus'] = False
        print("  ⚠ 未找到中文字体，使用默认字体")
        return False
        
    except Exception as e:
        print(f"  ⚠ 字体设置失败: {e}")
        matplotlib.rcParams['axes.unicode_minus'] = False
        return False

# 初始化中文字体
setup_chinese_font()

# 导入多任务模型组件
from multi_task_lstm import (
    MultiTaskDataset,
    MultiTaskLSTM,
    train_epoch,
    eval_model,
    set_random_seed,
    DEVICE
)

def load_custom_data(file_path, basins, time_range):
    """
    加载自定义的径流或水位数据
    
    Parameters
    ----------
    file_path : str
        数据文件路径
    basins : list
        流域ID列表（字符串格式）
    time_range : list
        时间范围 [start_date, end_date]
    
    Returns
    -------
    pd.DataFrame
        处理后的数据
    """
    df = pd.read_csv(file_path, parse_dates=['date'])
    df.set_index('date', inplace=True)
    
    # 确保basins是列表
    basins_list = list(basins)
    
    # 选择指定流域和时间范围
    selected_data = df.loc[time_range[0]:time_range[1], basins_list]
    
    return selected_data


if __name__ == "__main__":
    print("=" * 80)
    print("多任务LSTM模型训练 - 径流和水位联合预测")
    print("=" * 80)
    
    set_random_seed(1234)
    
    # ==================== 1. 配置参数 ====================
    print("\n步骤 1: 配置参数...")
    
    # 流域配置（使用实际的流域ID）
    chosen_basins = ["01121000", "01118300"]
    basins_num = len(chosen_basins)
    print(f"  选择的流域: {chosen_basins}")
    print(f"  流域数量: {basins_num}")
    
    # 时间范围配置（根据实际数据的可用范围）
    # 从1999-10-01到2014-12-31有完整数据
    # 划分为：训练集(1999-2008)，验证集(2009-2011)，测试集(2012-2014)
    train_times = ["1999-10-01", "2008-12-31"]
    valid_times = ["2009-01-01", "2011-12-31"]
    test_times = ["2012-01-01", "2014-12-31"]
    
    print(f"  训练期: {train_times[0]} 至 {train_times[1]}")
    print(f"  验证期: {valid_times[0]} 至 {valid_times[1]}")
    print(f"  测试期: {test_times[0]} 至 {test_times[1]}")
    
    # 模型超参数（基于fix_nan_final.py验证的稳定参数）
    sequence_length = 30     # 输入序列长度（天）- 验证稳定
    batch_size = 16          # 批次大小 - 验证稳定
    hidden_size = 32         # LSTM隐藏层大小
    dropout_rate = 0.1       # 降低Dropout比率避免过度正则化
    learning_rate = 1e-4     # 学习率 - 验证稳定
    n_epochs = 40            # 训练轮数
    grad_clip_norm = 1.0     # 梯度裁剪阈值 - 验证稳定
    
    # 任务权重
    task_weights = {
        'flow': 1.0,         # 径流任务权重
        'waterlevel': 1.0    # 水位任务权重
    }
    
    print(f"  序列长度: {sequence_length} 天")
    print(f"  批次大小: {batch_size}")
    print(f"  隐藏层大小: {hidden_size}")
    print(f"  Dropout率: {dropout_rate}")
    print(f"  学习率: {learning_rate}")
    print(f"  训练轮数: {n_epochs}")
    print(f"  梯度裁剪: {grad_clip_norm}")
    print(f"  任务权重: 径流={task_weights['flow']}, 水位={task_weights['waterlevel']}")
    
    # ==================== 2. 加载CAMELS数据（本地CAMELS/目录） ====================
    print("\n步骤 2: 加载CAMELS数据（气象强迫和流域属性，来自 CAMELS/）...")

    camels_dir = os.path.join("CAMELS")
    daymet_dir = os.path.join(camels_dir, "daymet")
    attrs_xlsx = os.path.join(camels_dir, "camels_attributes_v2.0.xlsx")

    if not os.path.isdir(camels_dir):
        raise FileNotFoundError(f"未找到 CAMELS 目录: {camels_dir}")
    if not os.path.isdir(daymet_dir):
        raise FileNotFoundError(f"未找到 Daymet 目录: {daymet_dir}")
    if not os.path.isfile(attrs_xlsx):
        raise FileNotFoundError(f"未找到属性表: {attrs_xlsx}")

    print(f"  CAMELS路径: {camels_dir}")
    print(f"  Daymet路径: {daymet_dir}")
    print(f"  属性文件: {attrs_xlsx}")

    # 读取属性数据（从多个分类文本文件）
    def load_camels_attributes(camels_dir_path: str) -> pd.DataFrame:
        """从 CAMELS 分类文本文件读取并合并属性数据"""
        attr_files = {
            'clim': 'camels_clim.txt',
            'geol': 'camels_geol.txt', 
            'hydro': 'camels_hydro.txt',
            'soil': 'camels_soil.txt',
            'topo': 'camels_topo.txt',
            'vege': 'camels_vege.txt'
        }
        
        dfs = []
        for category, filename in attr_files.items():
            file_path = os.path.join(camels_dir_path, filename)
            if os.path.exists(file_path):
                df = pd.read_csv(file_path, sep=';')
                dfs.append(df)
        
        if not dfs:
            raise FileNotFoundError(f"未找到任何属性文件在 {camels_dir_path}")
        
        # 以 gauge_id 为键合并所有属性
        attrs_merged = dfs[0]
        for df in dfs[1:]:
            attrs_merged = attrs_merged.merge(df, on='gauge_id', how='outer')
        
        return attrs_merged
    
    # 流域属性变量（基于实际 CAMELS 文本文件列名）
    chosen_attrs_vars = [
        # 气候属性
        "p_mean",
        "p_seasonality", 
        "frac_snow",
        "aridity",
        # 地质属性
        "geol_porostiy",
        "geol_permeability",
        # 土壤属性
        "soil_depth_statsgo",
        "soil_porosity",
        "soil_conductivity",
        # 地形属性
        "elev_mean",
        "slope_mean", 
        "area_gages2",
        # 植被属性
        "frac_forest",
        "lai_max",
    ]
    
    attrs_full = load_camels_attributes(camels_dir)
    
    # 只保留我们需要的属性变量和 gauge_id
    attrs_columns = ['gauge_id'] + chosen_attrs_vars
    attrs = attrs_full[attrs_columns].copy()
    
    print(f"  属性数据形状: {attrs.shape}")
    print(f"  属性数据列名: {list(attrs.columns)}")
    print(f"  包含 gauge_id: {'gauge_id' in attrs.columns}")

    # 辅助函数：从 Daymet 文本构建 forcing 的 xarray.Dataset
    def load_daymet_forcing(daymet_dir_path: str, basin_ids: list, forcing_vars: list) -> xr.Dataset:
        """从 CAMELS/daymet 文本读取指定流域的气象数据，返回 (time, basin) 维度的 xarray.Dataset。

        兼容常见Daymet列名：
        - year, y, Year / month, m, Mnth / day, d, Day
        - 变量名可能附带单位，例如 prcp(mm/day)、srad(W/m2)、tmax(C)、tmin(C)、vp(Pa)、dayl(s)
        """
        # 收集每个流域的 DataFrame（索引为日期）
        basin_to_df = {}

        for basin in basin_ids:
            # CAMELS Daymet 文件按 HUC 区域分目录，文件名格式：{gauge_id}_lump_cida_forcing_leap.txt
            # 流域 ID 前两位是 HUC 区域代码
            huc_region = basin[:2]
            expected_filename = f"{basin}_lump_cida_forcing_leap.txt"
            file_path = os.path.join(daymet_dir_path, huc_region, expected_filename)
            
            if not os.path.exists(file_path):
                raise FileNotFoundError(f"未找到 Daymet 文件: {file_path}")

            # 读取 Daymet 文本（跳过前3行头部信息，从第4行开始）
            df_raw = pd.read_csv(file_path, sep=r"\s+", engine="python", skiprows=3)

            # 规范列名（小写，去掉单位括号）
            def normalize_col(col: str):
                col_l = str(col).strip().lower()
                # 去掉括号中的单位
                if '(' in col_l and ')' in col_l:
                    col_l = col_l.split('(')[0].strip()
                return col_l

            df_raw.columns = [normalize_col(c) for c in df_raw.columns]

            # 推断日期列
            year_col = next((c for c in df_raw.columns if c in ["year", "yr", "y"]), None)
            month_col = next((c for c in df_raw.columns if c in ["month", "mnth", "m"]), None)
            day_col = next((c for c in df_raw.columns if c in ["day", "d"]), None)
            if year_col is None or month_col is None or day_col is None:
                raise ValueError(f"Daymet文件缺少日期列(year/month/day): {file_path}")

            df_raw["time"] = pd.to_datetime(dict(year=df_raw[year_col], month=df_raw[month_col], day=df_raw[day_col]))
            df_raw = df_raw.sort_values("time").set_index("time")

            # 建立变量映射（去单位名）
            # 常见列名：dayl, prcp, srad, tmax, tmin, vp
            col_map = {}
            for v in ["dayl", "prcp", "srad", "tmax", "tmin", "vp"]:
                if v in df_raw.columns:
                    col_map[v] = v
                else:
                    # 尝试匹配带单位的列名，比如 prcp(mm/day) -> prcp
                    matches = [c for c in df_raw.columns if c.startswith(v)]
                    if matches:
                        col_map[v] = matches[0]

            missing = [v for v in forcing_vars if v not in col_map]
            if missing:
                raise ValueError(f"Daymet缺少变量 {missing} 于 {file_path}")

            # 保留需要的列并重命名为目标变量名
            df_sel = df_raw[[col_map[v] for v in forcing_vars]].copy()
            df_sel.columns = forcing_vars
            basin_to_df[basin] = df_sel

        # 对齐所有流域的共同日期范围（内连接）
        common_index = None
        for df in basin_to_df.values():
            common_index = df.index if common_index is None else common_index.intersection(df.index)
        if common_index is None or len(common_index) == 0:
            raise ValueError("无法对齐Daymet时间索引（交集为空）")
        common_index = common_index.sort_values()

        # 组装到 xarray.Dataset，维度(time, basin)
        data_vars = {}
        for v in forcing_vars:
            # 构建二维数组: [time, basin]
            arr = np.stack([basin_to_df[b].reindex(common_index)[v].values for b in basin_ids], axis=1)
            data_vars[v] = (("time", "basin"), arr)

        ds = xr.Dataset(
            data_vars=data_vars,
            coords={
                "time": common_index,
                "basin": np.array(basin_ids, dtype=object),
            },
        )
        return ds
    
    # ==================== 3. 选择特征变量 ====================
    print("\n步骤 3: 选择特征变量...")
    
    # 气象强迫变量
    chosen_forcing_vars = ["dayl", "prcp", "srad", "tmax", "tmin", "vp"]
    print(f"  气象强迫变量 ({len(chosen_forcing_vars)}个): {', '.join(chosen_forcing_vars)}")
    
    print(f"  流域属性变量 ({len(chosen_attrs_vars)}个): {', '.join(chosen_attrs_vars)}")
    
    input_size = len(chosen_attrs_vars) + len(chosen_forcing_vars)
    print(f"  总输入特征维度: {input_size}")
    
    # 准备属性数据（Excel表中的 gauge_id 需为字符串以匹配 chosen_basins）
    if "gauge_id" in attrs.columns:
        attrs["gauge_id"] = attrs["gauge_id"].astype(str).str.zfill(8)
        print(f"  转换后属性数据包含 gauge_id: {'gauge_id' in attrs.columns}")
    else:
        raise KeyError("属性表缺少列 'gauge_id'")

    missing_attrs = [c for c in chosen_attrs_vars if c not in attrs.columns]
    if missing_attrs:
        raise KeyError(f"属性表缺少所需列: {missing_attrs}")

    chosen_attrs = attrs[attrs["gauge_id"].isin(chosen_basins)][["gauge_id"] + chosen_attrs_vars]
    
    print(f"  最终属性子集形状: {chosen_attrs.shape}")
    print(f"  最终属性子集列名: {list(chosen_attrs.columns)}")
    print(f"  最终子集包含 gauge_id: {'gauge_id' in chosen_attrs.columns}")
    
    # 不要设置索引，保留 gauge_id 列供 MultiTaskDataset 使用
    
    # 读取 Daymet 文本并构建 forcing 数据集
    forcing_ds_full = load_daymet_forcing(daymet_dir, chosen_basins, chosen_forcing_vars)

    # 根据时间范围切片
    train_forcings = forcing_ds_full[chosen_forcing_vars].sel(
        basin=chosen_basins, time=slice(train_times[0], train_times[1])
    )
    valid_forcings = forcing_ds_full[chosen_forcing_vars].sel(
        basin=chosen_basins, time=slice(valid_times[0], valid_times[1])
    )
    test_forcings = forcing_ds_full[chosen_forcing_vars].sel(
        basin=chosen_basins, time=slice(test_times[0], test_times[1])
    )
    
    # ==================== 4. 加载自定义径流和水位数据 ====================
    print("\n步骤 4: 加载自定义径流和水位数据...")
    
    try:
        train_flow = load_custom_data("flow_data.csv", chosen_basins, train_times)
        train_waterlevel = load_custom_data("waterlevel_data.csv", chosen_basins, train_times)
        print(f"  训练集 - 径流: {train_flow.shape}, 水位: {train_waterlevel.shape}")
        
        valid_flow = load_custom_data("flow_data.csv", chosen_basins, valid_times)
        valid_waterlevel = load_custom_data("waterlevel_data.csv", chosen_basins, valid_times)
        print(f"  验证集 - 径流: {valid_flow.shape}, 水位: {valid_waterlevel.shape}")
        
        test_flow = load_custom_data("flow_data.csv", chosen_basins, test_times)
        test_waterlevel = load_custom_data("waterlevel_data.csv", chosen_basins, test_times)
        print(f"  测试集 - 径流: {test_flow.shape}, 水位: {test_waterlevel.shape}")
        
        # 显示数据统计
        print(f"\n  数据统计:")
        print(f"    径流范围: {train_flow.min().min():.3f} - {train_flow.max().max():.3f} m3/s")
        print(f"    水位范围: {train_waterlevel.min().min():.3f} - {train_waterlevel.max().max():.3f} m")
        
    except FileNotFoundError:
        print("\n  [错误] 找不到数据文件！")
        print("  请先运行: python prepare_data.py")
        exit(1)
    except Exception as e:
        print(f"\n  [错误] 加载数据失败: {e}")
        exit(1)
    
    # ==================== 5. 创建数据集 ====================
    print("\n步骤 5: 创建数据集...")
    
    # 训练数据集
    ds_train = MultiTaskDataset(
        basins=chosen_basins,
        dates=train_times,
        data_attr=chosen_attrs,
        data_forcing=train_forcings,
        data_flow=train_flow,
        data_waterlevel=train_waterlevel,
        loader_type="train",
        seq_length=sequence_length,
    )
    tr_loader = DataLoader(ds_train, batch_size=batch_size, shuffle=True, num_workers=0)
    print(f"  训练集样本数: {len(ds_train)}")
    
    # 验证数据集
    means = ds_train.get_means()
    stds = ds_train.get_stds()
    ds_val = MultiTaskDataset(
        basins=chosen_basins,
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
    valid_batch_size = 1000
    val_loader = DataLoader(ds_val, batch_size=valid_batch_size, shuffle=False, num_workers=0)
    print(f"  验证集样本数: {len(ds_val)}")
    
    # 测试数据集
    ds_test = MultiTaskDataset(
        basins=chosen_basins,
        dates=test_times,
        data_attr=chosen_attrs,
        data_forcing=test_forcings,
        data_flow=test_flow,
        data_waterlevel=test_waterlevel,
        loader_type="test",
        seq_length=sequence_length,
        means=means,
        stds=stds,
    )
    test_batch_size = 1000
    test_loader = DataLoader(ds_test, batch_size=test_batch_size, shuffle=False, num_workers=0)
    print(f"  测试集样本数: {len(ds_test)}")
    
    # ==================== 6. 创建模型 ====================
    print("\n步骤 6: 创建多任务LSTM模型...")
    
    model = MultiTaskLSTM(
        input_size=input_size,
        hidden_size=hidden_size,
        dropout_rate=dropout_rate,
        task_weights=task_weights
    ).to(DEVICE)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    loss_func = nn.MSELoss()
    
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  模型参数量: {total_params:,}")
    print(f"  训练设备: {DEVICE}")
    
    # ==================== 7. 训练模型 ====================
    print("\n步骤 7: 开始训练...")
    print("=" * 80)
    
    train_losses = []
    val_nses_flow = []
    val_nses_waterlevel = []
    
    best_val_nse = -np.inf
    best_epoch = 0
    
    for i in range(n_epochs):
        # 训练
        train_loss, train_loss_flow, train_loss_waterlevel = train_epoch(
            model, optimizer, tr_loader, loss_func, i + 1
        )
        train_losses.append(train_loss)
        
        # 验证
        obs_flow, obs_waterlevel, preds_flow, preds_waterlevel = eval_model(model, val_loader)
        
        # 反归一化
        preds_flow = ds_val.local_denormalization(
            preds_flow.cpu().numpy(), variable="flow"
        )
        preds_waterlevel = ds_val.local_denormalization(
            preds_waterlevel.cpu().numpy(), variable="waterlevel"
        )
        
        obs_flow = obs_flow.numpy().reshape(basins_num, -1)
        obs_waterlevel = obs_waterlevel.numpy().reshape(basins_num, -1)
        preds_flow = preds_flow.reshape(basins_num, -1)
        preds_waterlevel = preds_waterlevel.reshape(basins_num, -1)
        
        # 计算NSE
        nse_flow = np.array([he.nse(preds_flow[j], obs_flow[j]) for j in range(basins_num)])
        nse_waterlevel = np.array([he.nse(preds_waterlevel[j], obs_waterlevel[j]) for j in range(basins_num)])
        
        val_nses_flow.append(nse_flow.mean())
        val_nses_waterlevel.append(nse_waterlevel.mean())
        
        # 计算综合NSE（用于选择最佳模型）
        avg_nse = (nse_flow.mean() + nse_waterlevel.mean()) / 2
        
        tqdm.write(
            f"Epoch {i+1}/{n_epochs} - "
            f"Loss: {train_loss:.4f} - "
            f"Val NSE (径流): {nse_flow.mean():.4f}, "
            f"(水位): {nse_waterlevel.mean():.4f}, "
            f"(平均): {avg_nse:.4f}"
        )
        
        # 保存最佳模型
        if avg_nse > best_val_nse:
            best_val_nse = avg_nse
            best_epoch = i + 1
            torch.save({
                'epoch': i + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'means': means,
                'stds': stds,
                'task_weights': task_weights,
                'val_nse_flow': nse_flow.mean(),
                'val_nse_waterlevel': nse_waterlevel.mean(),
            }, 'best_model.pth')
            tqdm.write(f"  → 保存最佳模型 (综合NSE: {avg_nse:.4f})")
    
    print("\n" + "=" * 80)
    print(f"训练完成！最佳模型: Epoch {best_epoch} (综合NSE: {best_val_nse:.4f})")
    
    # ==================== 8. 加载最佳模型并在测试集上评估 ====================
    print("\n步骤 8: 在测试集上评估最佳模型...")
    
    # 加载最佳模型
    checkpoint = torch.load('best_model.pth', weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    obs_flow, obs_waterlevel, preds_flow, preds_waterlevel = eval_model(model, test_loader)
    
    # 反归一化
    preds_flow = ds_test.local_denormalization(
        preds_flow.cpu().numpy(), variable="flow"
    )
    preds_waterlevel = ds_test.local_denormalization(
        preds_waterlevel.cpu().numpy(), variable="waterlevel"
    )
    
    obs_flow = obs_flow.numpy().reshape(basins_num, -1)
    obs_waterlevel = obs_waterlevel.numpy().reshape(basins_num, -1)
    preds_flow = preds_flow.reshape(basins_num, -1)
    preds_waterlevel = preds_waterlevel.reshape(basins_num, -1)
    
    # 计算测试集NSE
    nse_flow = np.array([he.nse(preds_flow[j], obs_flow[j]) for j in range(basins_num)])
    nse_waterlevel = np.array([he.nse(preds_waterlevel[j], obs_waterlevel[j]) for j in range(basins_num)])
    
    print("\n测试集结果：")
    print("=" * 80)
    for j in range(basins_num):
        print(f"流域 {chosen_basins[j]}:")
        print(f"  径流 NSE: {nse_flow[j]:.4f}")
        print(f"  水位 NSE: {nse_waterlevel[j]:.4f}")
        print()
    print(f"平均 NSE:")
    print(f"  径流: {nse_flow.mean():.4f}")
    print(f"  水位: {nse_waterlevel.mean():.4f}")
    print(f"  综合: {(nse_flow.mean() + nse_waterlevel.mean()) / 2:.4f}")
    print("=" * 80)
    
    # ==================== 9. 可视化结果 ====================
    print("\n步骤 9: 生成可视化图表...")
    
    # 准备日期范围
    start_date = pd.to_datetime(ds_test.dates[0], format="%Y-%m-%d") + pd.DateOffset(
        days=sequence_length - 1
    )
    end_date = pd.to_datetime(ds_test.dates[1], format="%Y-%m-%d")
    date_range = pd.date_range(start_date, end_date)
    
    # 为每个流域绘制结果
    for i in range(basins_num):
        fig, axes = plt.subplots(2, 1, figsize=(16, 10))
        
        # 径流预测图
        axes[0].plot(date_range, obs_flow[i], label="观测值", alpha=0.7, linewidth=1.5)
        axes[0].plot(date_range, preds_flow[i], label="预测值", alpha=0.7, linewidth=1.5)
        axes[0].legend(fontsize=12)
        axes[0].set_title(f"流域 {chosen_basins[i]} - 径流预测 (测试集 NSE: {nse_flow[i]:.4f})", fontsize=14, fontweight='bold')
        axes[0].set_ylabel("径流 (m³/s)", fontsize=12)
        axes[0].grid(True, alpha=0.3)
        
        # 水位预测图
        axes[1].plot(date_range, obs_waterlevel[i], label="观测值", alpha=0.7, linewidth=1.5)
        axes[1].plot(date_range, preds_waterlevel[i], label="预测值", alpha=0.7, linewidth=1.5)
        axes[1].legend(fontsize=12)
        axes[1].set_title(f"流域 {chosen_basins[i]} - 水位预测 (测试集 NSE: {nse_waterlevel[i]:.4f})", fontsize=14, fontweight='bold')
        axes[1].set_xlabel("日期", fontsize=12)
        axes[1].set_ylabel("水位 (m)", fontsize=12)
        axes[1].grid(True, alpha=0.3)
        
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        # 保存图片
        output_file = f"results_basin_{chosen_basins[i]}.png"
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"  [保存] {output_file}")
        plt.close()
    
    # 绘制训练曲线
    fig, axes = plt.subplots(1, 2, figsize=(16, 5))
    
    axes[0].plot(range(1, n_epochs + 1), train_losses, linewidth=2)
    axes[0].set_title("训练损失", fontsize=14, fontweight='bold')
    axes[0].set_xlabel("Epoch", fontsize=12)
    axes[0].set_ylabel("损失", fontsize=12)
    axes[0].grid(True, alpha=0.3)
    axes[0].axvline(x=best_epoch, color='r', linestyle='--', alpha=0.5, label=f'最佳模型 (Epoch {best_epoch})')
    axes[0].legend()
    
    axes[1].plot(range(1, n_epochs + 1), val_nses_flow, label="径流 NSE", linewidth=2)
    axes[1].plot(range(1, n_epochs + 1), val_nses_waterlevel, label="水位 NSE", linewidth=2)
    axes[1].set_title("验证集 NSE", fontsize=14, fontweight='bold')
    axes[1].set_xlabel("Epoch", fontsize=12)
    axes[1].set_ylabel("NSE", fontsize=12)
    axes[1].legend(fontsize=12)
    axes[1].grid(True, alpha=0.3)
    axes[1].axvline(x=best_epoch, color='r', linestyle='--', alpha=0.5)
    
    plt.tight_layout()
    plt.savefig("training_curves.png", dpi=300, bbox_inches='tight')
    print(f"  ✓ 已保存: training_curves.png")
    plt.close()
    
    print("\n" + "=" * 80)
    print("训练和评估完成！")
    print("=" * 80)
    print("\n生成的文件:")
    print("  - best_model.pth: 最佳模型权重")
    for basin in chosen_basins:
        print(f"  - results_basin_{basin}.png: 流域{basin}的预测结果")
    print("  - training_curves.png: 训练曲线")
    print("\n" + "=" * 80)

