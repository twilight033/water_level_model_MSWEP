import os
from pathlib import Path
import random
import numpy as np
import pandas as pd
import xarray as xr
from matplotlib import pyplot as plt
from matplotlib import font_manager
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader, Dataset
import torch.nn as nn
import torch.nn.functional as F
from hydrodataset.camelsh import Camelsh
from hydrodataset import StandardVariable
from improved_camelsh_reader import ImprovedCAMELSHReader
import HydroErr as he

DEVICE = torch.device(
    "cuda:0" if torch.cuda.is_available() else "cpu"
)  # check if GPU is available


def configure_chinese_font():
    """配置Matplotlib以支持中文显示"""
    preferred_fonts = [
        "SimHei",
        "Microsoft YaHei",
        "Source Han Sans SC",
        "Noto Sans CJK SC",
        "STSong",
    ]
    for font in preferred_fonts:
        try:
            font_path = font_manager.findfont(font, fallback_to_default=False)
            if font_path:
                plt.rcParams["font.family"] = font
                plt.rcParams["axes.unicode_minus"] = False
                print(f"使用字体: {font}")
                return
        except ValueError:
            continue
    print("警告: 未找到可用的中文字体，图表可能无法正常显示中文字符")
    plt.rcParams["axes.unicode_minus"] = False


class MultiTaskDataset(Dataset):
    """多任务数据集类，用于加载径流和水位数据（batch-first）"""

    def __init__(
        self,
        basins: list,
        dates: list,
        data_attr: pd.DataFrame,
        data_forcing: xr.Dataset,
        data_flow: pd.DataFrame,  # 径流数据（自定义）
        data_waterlevel: pd.DataFrame,  # 水位数据（自定义）
        loader_type: str = "train",
        seq_length: int = 100,
        means: dict = None,
        stds: dict = None,
    ):
        """
        初始化多任务数据集

        Parameters
        ----------
        basins : list
            流域ID列表
        dates : list
            时间范围 [start_date, end_date]
        data_attr : pd.DataFrame
            流域属性数据
        data_forcing : xr.Dataset
            气象强迫数据（来自CAMELS）
        data_flow : pd.DataFrame
            径流数据（自定义文件），格式：index为时间，columns为流域ID
        data_waterlevel : pd.DataFrame
            水位数据（自定义文件），格式：index为时间，columns为流域ID
        loader_type : str, optional
            数据集类型 "train", "valid", "test"
        seq_length : int, optional
            输入序列长度
        means : dict, optional
            归一化均值字典
        stds : dict, optional
            归一化标准差字典
        """
        super(MultiTaskDataset, self).__init__()
        if loader_type not in ["train", "valid", "test"]:
            raise ValueError(
                " 'loader_type' must be one of 'train', 'valid' or 'test' "
            )
        else:
            self.loader_type = loader_type
        self.basins = basins
        self.dates = dates

        self.seq_length = seq_length

        self.means = means
        self.stds = stds

        self.data_attr = data_attr
        self.data_forcing = data_forcing
        self.data_flow = data_flow
        self.data_waterlevel = data_waterlevel

        # 加载和预处理数据
        self._load_data()

    def __len__(self):
        return self.num_samples

    def __getitem__(self, item: int):
        basin, time_idx = self.lookup_table[item]
        seq_length = self.seq_length
        
        # 获取输入序列
        x = self.x[basin][time_idx : time_idx + seq_length]
        
        # 检查强迫数据
        if np.isnan(x).any():
            print(f"[错误] 样本 {item}: 强迫数据x包含NaN")
            print(f"  流域: {basin}, 时间索引: {time_idx}")
            print(f"  x形状: {x.shape}")
            raise ValueError("强迫数据x包含NaN")
        
        c = self.c.loc[basin].values
        
        # 检查属性数据
        if np.isnan(c).any():
            print(f"[错误] 样本 {item}: 属性数据c包含NaN")
            print(f"  流域: {basin}")
            print(f"  c值: {c}")
            raise ValueError("属性数据c包含NaN")
        
        c = np.tile(c, (seq_length, 1))
        xc = np.concatenate((x, c), axis=1)
        
        # 获取两个目标输出（序列最后一天的值）
        y_flow = self.y_flow[basin][time_idx + seq_length - 1]
        y_waterlevel = self.y_waterlevel[basin][time_idx + seq_length - 1]
        
        # 将两个目标合并为一个向量
        y = np.array([y_flow, y_waterlevel])
        
        # 最终NaN检查
        if np.isnan(xc).any():
            print(f"[错误] 样本 {item}: 输入特征包含NaN")
            print(f"  流域: {basin}, 时间索引: {time_idx}")
            raise ValueError("输入特征包含NaN")
        
        if np.isnan(y).any():
            print(f"[错误] 样本 {item}: 目标值包含NaN")
            print(f"  流域: {basin}, 时间索引: {time_idx}")
            print(f"  y_flow: {y_flow}, y_waterlevel: {y_waterlevel}")
            raise ValueError("目标值包含NaN")
        
        return torch.from_numpy(xc).float(), torch.from_numpy(y).float()

    def _load_data(self):
        """从文件加载数据 - 增强NaN检查和处理"""
        if self.loader_type == "train":
            train_mode = True
            # 计算归一化参数
            self.means = {}
            self.stds = {}
            
            # 检查原始数据中的NaN
            print("检查原始数据中的NaN...")
            
            # 检查强迫数据
            forcing_nan_count = self.data_forcing.isnull().sum().sum()
            print(f"强迫数据NaN统计: {forcing_nan_count}")
            if forcing_nan_count > 0:
                print(f"[警告] 强迫数据包含NaN值，将使用插值填充")
                # 使用简单的插值方法
                self.data_forcing = self.data_forcing.interpolate_na(dim='time', method='linear')
                # 如果还有NaN，用均值填充
                remaining_nan = self.data_forcing.isnull().sum().sum()
                if remaining_nan > 0:
                    self.data_forcing = self.data_forcing.fillna(self.data_forcing.mean())
            
            # 检查属性数据
            attr_nan_count = self.data_attr.isnull().sum().sum()
            if attr_nan_count > 0:
                print(f"[警告] 属性数据包含 {attr_nan_count} 个NaN值，将使用均值填充")
                numeric_cols = [col for col in self.data_attr.columns if col != 'gauge_id']
                self.data_attr[numeric_cols] = self.data_attr[numeric_cols].fillna(self.data_attr[numeric_cols].mean())
            
            # 检查径流数据
            flow_nan_count = self.data_flow.isnull().sum().sum()
            if flow_nan_count > 0:
                print(f"[警告] 径流数据包含 {flow_nan_count} 个NaN值，将使用前向填充和均值填充")
                self.data_flow = self.data_flow.ffill().bfill()
                # 如果仍有NaN，使用均值填充
                remaining_nan = self.data_flow.isnull().sum().sum()
                if remaining_nan > 0:
                    print(f"[警告] 前向/后向填充后仍有 {remaining_nan} 个NaN，使用均值填充")
                    self.data_flow = self.data_flow.fillna(self.data_flow.mean())
            
            # 检查水位数据
            wl_nan_count = self.data_waterlevel.isnull().sum().sum()
            if wl_nan_count > 0:
                print(f"[警告] 水位数据包含 {wl_nan_count} 个NaN值，将使用前向填充和均值填充")
                self.data_waterlevel = self.data_waterlevel.ffill().bfill()
                # 如果仍有NaN，使用均值填充
                remaining_nan = self.data_waterlevel.isnull().sum().sum()
                if remaining_nan > 0:
                    print(f"[警告] 前向/后向填充后仍有 {remaining_nan} 个NaN，使用均值填充")
                    self.data_waterlevel = self.data_waterlevel.fillna(self.data_waterlevel.mean())
            
            # 气象强迫数据的均值和标准差
            df_mean_forcings = self.data_forcing.mean().to_pandas()
            df_std_forcings = self.data_forcing.std().to_pandas()
            
            # 检查计算出的统计量
            if df_mean_forcings.isnull().any() or df_std_forcings.isnull().any():
                print("[错误] 强迫数据统计量包含NaN")
                raise ValueError("强迫数据统计量包含NaN")
            
            self.means['forcing'] = df_mean_forcings
            self.stds['forcing'] = df_std_forcings
            
            # 属性数据的均值和标准差（排除 gauge_id）
            numeric_cols = [col for col in self.data_attr.columns if col != 'gauge_id']
            numeric_attrs = self.data_attr[numeric_cols]
            df_mean_attr = numeric_attrs.mean()
            df_std_attr = numeric_attrs.std(ddof=0).fillna(0.0)
            
            # 检查计算出的统计量
            if df_mean_attr.isnull().any():
                print("[错误] 属性数据统计量均值包含NaN")
                raise ValueError("属性数据统计量包含NaN")
            
            self.means['attr'] = df_mean_attr
            self.stds['attr'] = df_std_attr
            
            # 径流数据的均值和标准差（排除日期列）
            flow_numeric = self.data_flow.select_dtypes(include=[np.number])
            flow_mean = flow_numeric.mean().mean()
            flow_std = flow_numeric.std().mean()
            
            # 检查计算出的统计量
            if np.isnan(flow_mean) or np.isnan(flow_std) or flow_std == 0:
                print(f"[错误] 径流数据统计量异常: mean={flow_mean}, std={flow_std}")
                raise ValueError("径流数据统计量异常")
            
            self.means['flow'] = flow_mean
            self.stds['flow'] = flow_std
            
            # 水位数据的均值和标准差（排除日期列）
            wl_numeric = self.data_waterlevel.select_dtypes(include=[np.number])
            waterlevel_mean = wl_numeric.mean().mean()
            waterlevel_std = wl_numeric.std().mean()
            
            # 检查计算出的统计量
            if np.isnan(waterlevel_mean) or np.isnan(waterlevel_std) or waterlevel_std == 0:
                print(f"[错误] 水位数据统计量异常: mean={waterlevel_mean}, std={waterlevel_std}")
                raise ValueError("水位数据统计量异常")
            
            self.means['waterlevel'] = waterlevel_mean
            self.stds['waterlevel'] = waterlevel_std
            
            # 最终NaN检查
            final_flow_nan = self.data_flow.isnull().sum().sum()
            final_wl_nan = self.data_waterlevel.isnull().sum().sum()
            final_attr_nan = self.data_attr[numeric_cols].isnull().sum().sum()
            
            if final_flow_nan > 0 or final_wl_nan > 0 or final_attr_nan > 0:
                print(f"[错误] 数据清洗后仍有NaN: 径流={final_flow_nan}, 水位={final_wl_nan}, 属性={final_attr_nan}")
                raise ValueError("数据清洗失败，仍存在NaN值")
            else:
                print("数据统计量计算完成，无NaN问题")
        else:
            train_mode = False

        # 归一化处理
        print("开始数据归一化...")
        self.x = self._normalize_forcing(self.data_forcing)
        print("强迫数据归一化完成")
        self.c = self._normalize_attr(self.data_attr)
        print("属性数据归一化完成")
        
        # 所有模式都需要归一化目标变量，保持一致性
        self.y_flow = self._normalize_flow(self.data_flow)
        self.y_waterlevel = self._normalize_waterlevel(self.data_waterlevel)
        
        # 同时保存原始数据用于评估
        if not train_mode:
            self.y_flow_raw = self._dataframe_to_dict(self.data_flow)
            self.y_waterlevel_raw = self._dataframe_to_dict(self.data_waterlevel)
        
        self.train_mode = train_mode
        self._create_lookup_table()

    def _dataframe_to_dict(self, df):
        """将DataFrame转换为字典格式，并处理NaN值"""
        result = {}
        for basin in self.basins:
            values = df[basin].values
            # 检查并处理NaN值
            if np.isnan(values).any():
                nan_count = np.isnan(values).sum()
                print(f"[警告] 流域 {basin} 的目标数据包含 {nan_count} 个NaN值，将使用前向填充")
                # 转换为pandas Series进行填充
                series = df[basin].ffill().bfill()
                # 如果仍有NaN，使用均值填充
                if series.isnull().any():
                    remaining_nan = series.isnull().sum()
                    print(f"[警告] 流域 {basin} 前向/后向填充后仍有 {remaining_nan} 个NaN，使用均值填充")
                    series = series.fillna(series.mean())
                values = series.values
            result[basin] = values
        return result

    def _normalize_forcing(self, data_forcing):
        """归一化气象强迫数据 - 增强NaN检查"""
        result = {}
        
        # 检查统计量
        print(f"强迫数据统计量检查:")
        print(f"  均值: {self.means['forcing'].values}")
        print(f"  标准差: {self.stds['forcing'].values}")
        
        # 处理标准差为0的情况（避免除零错误）
        std_values = self.stds['forcing'].values.copy()
        zero_std_mask = std_values < 1e-8
        if np.any(zero_std_mask):
            print(f"警告：发现标准差接近0的强迫变量，将设为1避免除零错误")
            std_values[zero_std_mask] = 1.0
        
        for basin in self.basins:
            basin_data = data_forcing.sel(basin=basin).to_array().to_numpy().T
            
            # 检查原始数据
            if np.isnan(basin_data).any():
                print(f"[错误] 流域 {basin} 强迫数据包含NaN")
                nan_count = np.isnan(basin_data).sum()
                print(f"  NaN数量: {nan_count}")
                raise ValueError(f"流域 {basin} 强迫数据包含NaN")
            
            normalized = (basin_data - self.means['forcing'].values) / std_values
            
            # 检查归一化后的数据
            if np.isnan(normalized).any():
                print(f"[错误] 流域 {basin} 归一化后强迫数据包含NaN")
                print(f"  原始数据形状: {basin_data.shape}")
                print(f"  均值: {self.means['forcing'].values}")
                print(f"  标准差: {std_values}")
                raise ValueError(f"流域 {basin} 归一化后强迫数据包含NaN")
            
            result[basin] = normalized
        return result

    def _normalize_attr(self, data_attr):
        """归一化属性数据（只处理数值列）- 增强NaN检查"""
        # 分离 gauge_id 和数值列
        gauge_ids = data_attr['gauge_id']
        numeric_cols = [col for col in data_attr.columns if col != 'gauge_id']
        numeric_attrs = data_attr[numeric_cols]
        
        print(f"属性数据归一化检查:")
        print(f"  原始属性数据形状: {numeric_attrs.shape}")
        print(f"  均值: {self.means['attr'].values}")
        print(f"  标准差: {self.stds['attr'].values}")
        
        # 检查是否有标准差为0的列
        zero_std_mask = self.stds['attr'] < 1e-8
        if zero_std_mask.any():
            print(f"[警告] 发现标准差接近0的属性列: {self.stds['attr'][zero_std_mask].index.tolist()}")
            # 对标准差为0的列，设置标准差为1（这样归一化后为0）
            std_values = self.stds['attr'].copy()
            std_values[zero_std_mask] = 1.0
        else:
            std_values = self.stds['attr']
        
        # 归一化数值列
        normalized = (numeric_attrs - self.means['attr']) / std_values
        
        # 检查归一化后是否有NaN
        if normalized.isnull().any().any():
            print(f"[错误] 属性数据归一化后包含NaN")
            print(f"  NaN位置:")
            for col in normalized.columns:
                if normalized[col].isnull().any():
                    print(f"    列 {col}: {normalized[col].isnull().sum()} 个NaN")
            raise ValueError("属性数据归一化后包含NaN")
        
        # 设置索引为 gauge_id，以便后续通过流域ID访问
        normalized.index = gauge_ids
        return normalized

    def _normalize_flow(self, data_flow):
        """归一化径流数据"""
        result = {}
        for basin in self.basins:
            values = data_flow[basin].values
            # 处理NaN值
            if np.isnan(values).any():
                print(f"[警告] 流域 {basin} 的径流数据在归一化前包含NaN，进行插值处理")
                values = pd.Series(values).ffill().bfill().fillna(self.means['flow']).values
            
            normalized = (values - self.means['flow']) / self.stds['flow']
            
            # 检查归一化后是否还有NaN
            if np.isnan(normalized).any():
                print(f"[错误] 流域 {basin} 的径流数据归一化后仍包含NaN")
                # 用0填充剩余的NaN
                normalized = np.nan_to_num(normalized, nan=0.0)
            
            result[basin] = normalized
        return result

    def _normalize_waterlevel(self, data_waterlevel):
        """归一化水位数据"""
        result = {}
        for basin in self.basins:
            values = data_waterlevel[basin].values
            # 处理NaN值
            if np.isnan(values).any():
                print(f"[警告] 流域 {basin} 的水位数据在归一化前包含NaN，进行插值处理")
                values = pd.Series(values).ffill().bfill().fillna(self.means['waterlevel']).values
            
            normalized = (values - self.means['waterlevel']) / self.stds['waterlevel']
            
            # 检查归一化后是否还有NaN
            if np.isnan(normalized).any():
                print(f"[错误] 流域 {basin} 的水位数据归一化后仍包含NaN")
                # 用0填充剩余的NaN
                normalized = np.nan_to_num(normalized, nan=0.0)
            
            result[basin] = normalized
        return result

    def _create_lookup_table(self):
        """创建索引表"""
        lookup = []
        seq_length = self.seq_length
        time_length = len(self.data_flow)
        
        for basin in tqdm(self.basins, desc="创建索引表"):
            for j in range(time_length - seq_length + 1):
                lookup.append((basin, j))
        
        self.lookup_table = {i: elem for i, elem in enumerate(lookup)}
        self.num_samples = len(self.lookup_table)

    def get_means(self):
        return self.means

    def get_stds(self):
        return self.stds

    def local_denormalization(self, feature, variable="flow"):
        """反归一化"""
        if variable == "flow":
            return feature * self.stds['flow'] + self.means['flow']
        elif variable == "waterlevel":
            return feature * self.stds['waterlevel'] + self.means['waterlevel']
        else:
            raise ValueError(f"Unknown variable: {variable}")


class MultiTaskLSTM(nn.Module):
    """双头多任务LSTM网络，同时预测径流和水位"""

    def __init__(
        self, 
        input_size: int, 
        hidden_size: int, 
        dropout_rate: float = 0.0,
        task_weights: dict = None
    ):
        """
        构建多任务LSTM模型

        Parameters
        ----------
        input_size : int
            输入特征维度
        hidden_size : int
            LSTM隐藏层大小
        dropout_rate : float, optional
            Dropout比率
        task_weights : dict, optional
            任务权重 {'flow': w1, 'waterlevel': w2}，默认均为1.0
        """
        super(MultiTaskLSTM, self).__init__()

        # 共享的LSTM层
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=2,
            bias=True,
            batch_first=True,
        )
        self.dropout = nn.Dropout(p=dropout_rate)
        
        # 两个独立的输出头
        self.fc_flow = nn.Linear(in_features=hidden_size, out_features=1)
        self.fc_waterlevel = nn.Linear(in_features=hidden_size, out_features=1)
        
        # 任务权重
        if task_weights is None:
            self.task_weights = {'flow': 1.0, 'waterlevel': 1.0}
        else:
            self.task_weights = task_weights

    def forward(self, x: torch.Tensor) -> tuple:
        """
        前向传播

        Returns
        -------
        tuple
            (径流预测, 水位预测)
        """
        output, (h_n, c_n) = self.lstm(x)

        # 使用最后一层的隐藏状态
        hidden = self.dropout(h_n[-1, :, :])
        
        # 两个独立的预测头
        pred_flow = self.fc_flow(hidden)
        pred_waterlevel = self.fc_waterlevel(hidden)
        
        return pred_flow, pred_waterlevel


def multi_task_loss(pred_flow, pred_waterlevel, target, loss_func, task_weights):
    """
    多任务损失函数

    Parameters
    ----------
    pred_flow : torch.Tensor
        径流预测值
    pred_waterlevel : torch.Tensor
        水位预测值
    target : torch.Tensor
        目标值 [batch_size, 2]，第一列是径流，第二列是水位
    loss_func : callable
        基础损失函数（如MSELoss）
    task_weights : dict
        任务权重

    Returns
    -------
    tuple
        (总损失, 径流损失, 水位损失)
    """
    loss_flow = loss_func(pred_flow, target[:, 0:1])
    loss_waterlevel = loss_func(pred_waterlevel, target[:, 1:2])
    
    total_loss = (
        task_weights['flow'] * loss_flow + 
        task_weights['waterlevel'] * loss_waterlevel
    )
    
    return total_loss, loss_flow, loss_waterlevel


def train_epoch(model, optimizer, loader, loss_func, epoch):
    """训练一个epoch - 增强NaN检查和错误处理"""
    model.train()
    pbar = tqdm(loader)
    pbar.set_description(f"Epoch {epoch}")
    
    epoch_loss = 0.0
    epoch_loss_flow = 0.0
    epoch_loss_waterlevel = 0.0
    
    for batch_idx, (xs, ys) in enumerate(pbar):
        # 检查输入数据
        if torch.isnan(xs).any() or torch.isnan(ys).any():
            print(f"[错误] Epoch {epoch}, 批次 {batch_idx}: 输入数据包含NaN")
            raise ValueError("输入数据包含NaN")
        
        optimizer.zero_grad()
        xs, ys = xs.to(DEVICE), ys.to(DEVICE)
        
        # 获取模型预测
        try:
            pred_flow, pred_waterlevel = model(xs)
        except Exception as e:
            print(f"[错误] Epoch {epoch}, 批次 {batch_idx}: 前向传播失败 - {e}")
            raise e
        
        # 检查预测结果
        if torch.isnan(pred_flow).any() or torch.isnan(pred_waterlevel).any():
            print(f"[错误] Epoch {epoch}, 批次 {batch_idx}: 预测结果包含NaN")
            raise ValueError("预测结果包含NaN")
        
        # 计算损失
        total_loss, loss_flow, loss_waterlevel = multi_task_loss(
            pred_flow, pred_waterlevel, ys, loss_func, model.task_weights
        )
        
        # 检查损失
        if torch.isnan(total_loss) or torch.isnan(loss_flow) or torch.isnan(loss_waterlevel):
            print(f"[错误] Epoch {epoch}, 批次 {batch_idx}: 损失包含NaN")
            print(f"  total_loss: {total_loss.item()}")
            print(f"  loss_flow: {loss_flow.item()}")
            print(f"  loss_waterlevel: {loss_waterlevel.item()}")
            raise ValueError("损失包含NaN")
        
        # 反向传播
        total_loss.backward()
        
        # 检查梯度
        grad_norm = 0.0
        for param in model.parameters():
            if param.grad is not None:
                param_norm = param.grad.data.norm(2)
                grad_norm += param_norm.item() ** 2
                if torch.isnan(param.grad).any():
                    print(f"[错误] Epoch {epoch}, 批次 {batch_idx}: 梯度包含NaN")
                    raise ValueError("梯度包含NaN")
        grad_norm = grad_norm ** 0.5
        
        # 梯度裁剪防止梯度爆炸
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        
        # 累计损失
        epoch_loss += total_loss.item()
        epoch_loss_flow += loss_flow.item()
        epoch_loss_waterlevel += loss_waterlevel.item()
        
        # 更新进度条
        pbar.set_postfix_str(
            f"Loss: {total_loss.item():.4f} "
            f"(Q: {loss_flow.item():.4f}, h: {loss_waterlevel.item():.4f}, "
            f"Grad: {grad_norm:.4f})"
        )
    
    # 检查epoch平均损失
    avg_loss = epoch_loss / len(loader)
    avg_loss_flow = epoch_loss_flow / len(loader)
    avg_loss_waterlevel = epoch_loss_waterlevel / len(loader)
    
    if np.isnan(avg_loss) or np.isnan(avg_loss_flow) or np.isnan(avg_loss_waterlevel):
        print(f"[错误] Epoch {epoch}: 平均损失包含NaN")
        raise ValueError("平均损失包含NaN")
    
    return avg_loss, avg_loss_flow, avg_loss_waterlevel


def eval_model(model, loader):
    """评估模型"""
    model.eval()
    obs_flow = []
    obs_waterlevel = []
    preds_flow = []
    preds_waterlevel = []
    
    with torch.no_grad():
        for xs, ys in loader:
            xs = xs.to(DEVICE)
            
            # 获取模型预测
            pred_flow, pred_waterlevel = model(xs)
            
            obs_flow.append(ys[:, 0])
            obs_waterlevel.append(ys[:, 1])
            preds_flow.append(pred_flow.squeeze())
            preds_waterlevel.append(pred_waterlevel.squeeze())

    return (
        torch.cat(obs_flow), 
        torch.cat(obs_waterlevel),
        torch.cat(preds_flow), 
        torch.cat(preds_waterlevel)
    )


def load_custom_data(file_path, basins, time_range):
    """
    加载自定义的径流或水位数据
    
    文件格式要求：
    - CSV文件
    - 第一列为日期（格式：YYYY-MM-DD）
    - 其余列为各流域的观测值，列名为流域ID
    
    Parameters
    ----------
    file_path : str
        数据文件路径
    basins : list
        流域ID列表
    time_range : list
        时间范围 [start_date, end_date]
    
    Returns
    -------
    pd.DataFrame
        处理后的数据
    """
    df = pd.read_csv(file_path, index_col=0, parse_dates=True)
    
    # 选择指定流域和时间范围
    selected_data = df.loc[time_range[0]:time_range[1], basins]
    
    return selected_data


def set_random_seed(seed):
    """设置随机种子以保证可复现性"""
    print("随机种子:", seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


if __name__ == "__main__":
    set_random_seed(1234)
    configure_chinese_font()
    
    # 导入配置
    from config import (
        CAMELSH_DATA_PATH, NUM_BASINS, SEQUENCE_LENGTH, BATCH_SIZE,
        TRAIN_START, TRAIN_END, VALID_START, VALID_END, TEST_START, TEST_END,
        FORCING_VARIABLES, ATTRIBUTE_VARIABLES, AVAILABLE_BASINS, VALID_WATER_LEVEL_BASINS
    )
    
    # ==================== 1. 加载CAMELSH数据 ====================
    print("\n正在加载CAMELSH数据...")
    print(f"数据路径: {CAMELSH_DATA_PATH}")
    
    # 使用改进的CAMELSH读取器
    camelsh_reader = ImprovedCAMELSHReader(CAMELSH_DATA_PATH, download=False, use_batch=True)
    
    # 获取数据概要
    summary = camelsh_reader.get_data_summary()
    print(f"数据概要:")
    print(f"  总流域数量: {summary['total_basins']}")
    print(f"  流域ID范围: {summary['basin_id_range']}")
    print(f"  批处理文件数量: {summary['batch_files']}")
    print(f"  NC文件数量: {summary['nc_files']}")
    print(f"  可用变量: {len(summary['available_variables'])} 个")
    
    # 获取流域列表
    basin_ids = camelsh_reader.read_object_ids()
    print(f"前10个流域: {basin_ids[:10]}")
    
    # 为了兼容性，保留原始camelsh对象用于属性数据读取
    camelsh = camelsh_reader.camelsh
    
    # ==================== 2. 选择流域和时间范围 ====================
    # 使用有有效水位数据的流域列表
    chosen_basins = VALID_WATER_LEVEL_BASINS[:NUM_BASINS]
    print(f"选择的流域: {chosen_basins}")
    print(f"注意：这些流域都有有效的水位数据")
    
    # 使用CAMELSH的默认时间范围
    default_range = camelsh.default_t_range
    print(f"数据集默认时间范围: {default_range}")
    
    # 使用配置文件中的时间范围
    train_times = [TRAIN_START, TRAIN_END]
    valid_times = [VALID_START, VALID_END]
    test_times = [TEST_START, TEST_END]
    print(f"训练时间: {train_times}")
    print(f"验证时间: {valid_times}")
    print(f"测试时间: {test_times}")
    
    # ==================== 3. 选择特征变量 ====================
    # 将配置文件中的字符串转换为StandardVariable
    chosen_forcing_vars = []
    for var_name in FORCING_VARIABLES:
        if hasattr(StandardVariable, var_name.upper()):
            chosen_forcing_vars.append(getattr(StandardVariable, var_name.upper()))
        else:
            # 如果没有对应的StandardVariable，直接使用字符串
            chosen_forcing_vars.append(var_name)
    
    chosen_attrs_vars = ATTRIBUTE_VARIABLES
    
    print(f"选择的气象变量: {FORCING_VARIABLES}")
    print(f"选择的属性变量: {ATTRIBUTE_VARIABLES}")
    
    # 准备属性数据
    print("\n正在加载属性数据...")
    attrs = camelsh.read_attr_xrdataset(
        gage_id_lst=chosen_basins,
        var_lst=chosen_attrs_vars
    )
    print(f"属性数据形状: {attrs.dims}")
    print(f"属性数据变量: {list(attrs.data_vars.keys())}")
    
    # 准备气象强迫数据 - 使用改进的读取器
    print("\n正在加载气象强迫数据...")
    train_forcings = camelsh_reader.read_ts_xrdataset(
        gage_id_lst=chosen_basins,
        t_range=train_times,
        var_lst=chosen_forcing_vars
    )
    valid_forcings = camelsh_reader.read_ts_xrdataset(
        gage_id_lst=chosen_basins,
        t_range=valid_times,
        var_lst=chosen_forcing_vars
    )
    test_forcings = camelsh_reader.read_ts_xrdataset(
        gage_id_lst=chosen_basins,
        t_range=test_times,
        var_lst=chosen_forcing_vars
    )
    print(f"训练集气象数据形状: {train_forcings.dims}")
    print(f"气象数据变量: {list(train_forcings.data_vars.keys())}")
    
    # ==================== 4. 加载径流和水位数据 ====================
    print("\n正在从CAMELSH数据集加载径流和水位数据...")
    
    # 使用改进的读取器加载径流数据
    print("加载径流数据...")
    train_flow_ds = camelsh_reader.read_ts_xrdataset(
        gage_id_lst=chosen_basins,
        t_range=train_times,
        var_lst=[StandardVariable.STREAMFLOW]
    )
    valid_flow_ds = camelsh_reader.read_ts_xrdataset(
        gage_id_lst=chosen_basins,
        t_range=valid_times,
        var_lst=[StandardVariable.STREAMFLOW]
    )
    test_flow_ds = camelsh_reader.read_ts_xrdataset(
        gage_id_lst=chosen_basins,
        t_range=test_times,
        var_lst=[StandardVariable.STREAMFLOW]
    )
    
    # 使用改进的读取器加载水位数据
    print("加载水位数据...")
    try:
        train_waterlevel_ds = camelsh_reader.read_ts_xrdataset(
            gage_id_lst=chosen_basins,
            t_range=train_times,
            var_lst=[StandardVariable.WATER_LEVEL]
        )
        valid_waterlevel_ds = camelsh_reader.read_ts_xrdataset(
            gage_id_lst=chosen_basins,
            t_range=valid_times,
            var_lst=[StandardVariable.WATER_LEVEL]
        )
        test_waterlevel_ds = camelsh_reader.read_ts_xrdataset(
            gage_id_lst=chosen_basins,
            t_range=test_times,
            var_lst=[StandardVariable.WATER_LEVEL]
        )
    except Exception as e:
        print(f"警告: 无法加载水位数据: {e}")
        print("将使用模拟水位数据进行演示...")
        # 创建模拟水位数据（基于径流数据）
        train_waterlevel_ds = train_flow_ds.copy()
        valid_waterlevel_ds = valid_flow_ds.copy()
        test_waterlevel_ds = test_flow_ds.copy()
        # 简单的水位模拟：水位 = 径流 * 0.1 + 随机噪声
        for ds in [train_waterlevel_ds, valid_waterlevel_ds, test_waterlevel_ds]:
            if StandardVariable.STREAMFLOW in ds.data_vars:
                flow_data = ds[StandardVariable.STREAMFLOW]
                # 模拟水位数据
                water_level_data = flow_data * 0.1 + np.random.normal(0, 0.01, flow_data.shape)
                ds[StandardVariable.WATER_LEVEL] = water_level_data
                # 移除径流数据，只保留水位
                ds = ds.drop_vars([StandardVariable.STREAMFLOW])
    
    # 转换为pandas DataFrame格式以兼容现有代码
    print("转换数据格式...")
    train_flow = train_flow_ds[StandardVariable.STREAMFLOW].to_pandas().T
    train_waterlevel = train_waterlevel_ds[StandardVariable.WATER_LEVEL].to_pandas().T
    
    valid_flow = valid_flow_ds[StandardVariable.STREAMFLOW].to_pandas().T
    valid_waterlevel = valid_waterlevel_ds[StandardVariable.WATER_LEVEL].to_pandas().T
    
    test_flow = test_flow_ds[StandardVariable.STREAMFLOW].to_pandas().T
    test_waterlevel = test_waterlevel_ds[StandardVariable.WATER_LEVEL].to_pandas().T
    
    print(f"径流数据形状: {train_flow.shape}")
    print(f"水位数据形状: {train_waterlevel.shape}")
    print(f"径流数据范围: {train_flow.min().min():.3f} - {train_flow.max().max():.3f}")
    print(f"水位数据范围: {train_waterlevel.min().min():.3f} - {train_waterlevel.max().max():.3f}")
    
    # ==================== 5. 创建数据集 ====================
    print("\n正在创建数据集...")
    sequence_length = SEQUENCE_LENGTH
    batch_size = BATCH_SIZE
    print(f"序列长度: {sequence_length}")
    print(f"批次大小: {batch_size}")
    
    # 转换属性数据为pandas DataFrame格式
    print("转换属性数据格式...")
    attrs_df = attrs.to_pandas()  # 不转置：流域为index，属性为columns
    attrs_df.index.name = 'gauge_id'
    attrs_df = attrs_df.reset_index()  # 将gauge_id作为列
    print(f"属性DataFrame形状: {attrs_df.shape}")
    print(f"属性DataFrame列: {list(attrs_df.columns)}")
    print("属性DataFrame样本:")
    print(attrs_df.head())
    
    # 训练数据集
    ds_train = MultiTaskDataset(
        basins=chosen_basins,
        dates=train_times,
        data_attr=attrs_df,
        data_forcing=train_forcings,
        data_flow=train_flow,
        data_waterlevel=train_waterlevel,
        loader_type="train",
        seq_length=sequence_length,
    )
    tr_loader = DataLoader(ds_train, batch_size=batch_size, shuffle=True)
    
    # 验证数据集
    means = ds_train.get_means()
    stds = ds_train.get_stds()
    ds_val = MultiTaskDataset(
        basins=chosen_basins,
        dates=valid_times,
        data_attr=attrs_df,
        data_forcing=valid_forcings,
        data_flow=valid_flow,
        data_waterlevel=valid_waterlevel,
        loader_type="valid",
        seq_length=sequence_length,
        means=means,
        stds=stds,
    )
    valid_batch_size = 1000
    val_loader = DataLoader(ds_val, batch_size=valid_batch_size, shuffle=False)
    
    # 测试数据集
    ds_test = MultiTaskDataset(
        basins=chosen_basins,
        dates=test_times,
        data_attr=attrs_df,
        data_forcing=test_forcings,
        data_flow=test_flow,
        data_waterlevel=test_waterlevel,
        loader_type="test",
        seq_length=sequence_length,
        means=means,
        stds=stds,
    )
    test_batch_size = 1000
    test_loader = DataLoader(ds_test, batch_size=test_batch_size, shuffle=False)
    
    # ==================== 6. 创建模型 ====================
    print("\n正在创建多任务LSTM模型...")
    input_size = len(chosen_attrs_vars) + len(chosen_forcing_vars)
    hidden_size = 64  # LSTM隐藏层大小
    dropout_rate = 0.2  # Dropout率
    learning_rate = 1e-3  # 学习率
    
    # 设置任务权重（可以根据需要调整）
    task_weights = {'flow': 1.0, 'waterlevel': 1.0}
    
    model = MultiTaskLSTM(
        input_size=input_size, 
        hidden_size=hidden_size, 
        dropout_rate=dropout_rate,
        task_weights=task_weights
    ).to(DEVICE)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    loss_func = nn.MSELoss()
    
    print(f"模型参数量: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")
    print(f"设备: {DEVICE}")
    
    # ==================== 7. 训练模型 ====================
    print("\n开始训练...")
    n_epochs = 5  # 训练轮数
    
    train_losses = []
    val_nses_flow = []
    val_nses_waterlevel = []
    
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
        obs_flow = ds_val.local_denormalization(
            obs_flow.cpu().numpy(), variable="flow"
        )
        obs_waterlevel = ds_val.local_denormalization(
            obs_waterlevel.cpu().numpy(), variable="waterlevel"
        )
        
        obs_flow = obs_flow.reshape(NUM_BASINS, -1)
        obs_waterlevel = obs_waterlevel.reshape(NUM_BASINS, -1)
        preds_flow = preds_flow.reshape(NUM_BASINS, -1)
        preds_waterlevel = preds_waterlevel.reshape(NUM_BASINS, -1)
        
        # 计算NSE
        nse_flow = np.array([he.nse(preds_flow[j], obs_flow[j]) for j in range(NUM_BASINS)])
        nse_waterlevel = np.array([he.nse(preds_waterlevel[j], obs_waterlevel[j]) for j in range(NUM_BASINS)])
        
        val_nses_flow.append(nse_flow.mean())
        val_nses_waterlevel.append(nse_waterlevel.mean())
        
        tqdm.write(
            f"Epoch {i+1} - "
            f"验证集 NSE (径流): {nse_flow.mean():.4f}, "
            f"NSE (水位): {nse_waterlevel.mean():.4f}"
        )
    
    # ==================== 8. 测试模型 ====================
    print("\n在测试集上评估...")
    obs_flow, obs_waterlevel, preds_flow, preds_waterlevel = eval_model(model, test_loader)
    
    # 反归一化
    preds_flow = ds_test.local_denormalization(
        preds_flow.cpu().numpy(), variable="flow"
    )
    preds_waterlevel = ds_test.local_denormalization(
        preds_waterlevel.cpu().numpy(), variable="waterlevel"
    )
    obs_flow = ds_test.local_denormalization(
        obs_flow.cpu().numpy(), variable="flow"
    )
    obs_waterlevel = ds_test.local_denormalization(
        obs_waterlevel.cpu().numpy(), variable="waterlevel"
    )
    
    obs_flow = obs_flow.reshape(NUM_BASINS, -1)
    obs_waterlevel = obs_waterlevel.reshape(NUM_BASINS, -1)
    preds_flow = preds_flow.reshape(NUM_BASINS, -1)
    preds_waterlevel = preds_waterlevel.reshape(NUM_BASINS, -1)
    
    # 计算测试集NSE
    nse_flow = np.array([he.nse(preds_flow[j], obs_flow[j]) for j in range(NUM_BASINS)])
    nse_waterlevel = np.array([he.nse(preds_waterlevel[j], obs_waterlevel[j]) for j in range(NUM_BASINS)])
    
    print(f"\n测试集结果：")
    for j in range(NUM_BASINS):
        print(f"流域 {chosen_basins[j]}:")
        print(f"  径流 NSE: {nse_flow[j]:.4f}")
        print(f"  水位 NSE: {nse_waterlevel[j]:.4f}")
    print(f"平均 NSE (径流): {nse_flow.mean():.4f}")
    print(f"平均 NSE (水位): {nse_waterlevel.mean():.4f}")
    
    # ==================== 9. 可视化结果 ====================
    print("\n正在生成可视化图表...")
    
    # 准备日期范围 - 根据实际预测数据长度
    start_date = pd.to_datetime(ds_test.dates[0], format="%Y-%m-%d") + pd.DateOffset(
        days=sequence_length - 1
    )
    # 使用实际预测数据的长度来创建日期范围
    actual_length = len(obs_flow[0])  # 获取第一个流域的数据长度
    date_range = pd.date_range(start_date, periods=actual_length, freq='H')  # 使用小时频率
    
    # 为每个流域绘制结果
    for i in range(NUM_BASINS):
        fig, axes = plt.subplots(2, 1, figsize=(14, 8))
        
        # 径流预测图
        axes[0].plot(date_range, obs_flow[i], label="观测值", alpha=0.7)
        axes[0].plot(date_range, preds_flow[i], label="预测值", alpha=0.7)
        axes[0].legend()
        axes[0].set_title(f"流域 {chosen_basins[i]} - 径流预测 (测试集 NSE: {nse_flow[i]:.3f})")
        axes[0].set_ylabel("径流 (mm/d)")
        axes[0].grid(True, alpha=0.3)
        
        # 水位预测图
        axes[1].plot(date_range, obs_waterlevel[i], label="观测值", alpha=0.7)
        axes[1].plot(date_range, preds_waterlevel[i], label="预测值", alpha=0.7)
        axes[1].legend()
        axes[1].set_title(f"流域 {chosen_basins[i]} - 水位预测 (测试集 NSE: {nse_waterlevel[i]:.3f})")
        axes[1].set_xlabel("日期")
        axes[1].set_ylabel("水位 (m)")
        axes[1].grid(True, alpha=0.3)
        
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        # 保存图片
        output_file = f"results_basin_{chosen_basins[i]}.png"
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"已保存图片: {output_file}")
    
    # 绘制训练曲线
    fig, axes = plt.subplots(1, 2, figsize=(14, 4))
    
    axes[0].plot(train_losses)
    axes[0].set_title("训练损失")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("损失")
    axes[0].grid(True, alpha=0.3)
    
    axes[1].plot(val_nses_flow, label="径流 NSE")
    axes[1].plot(val_nses_waterlevel, label="水位 NSE")
    axes[1].set_title("验证集 NSE")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("NSE")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig("training_curves.png", dpi=300, bbox_inches='tight')
    print("已保存训练曲线: training_curves.png")
    
    plt.show()
    
    # ==================== 10. 保存模型 ====================
    print("\n正在保存模型...")
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'means': means,
        'stds': stds,
        'task_weights': task_weights,
        'test_nse_flow': nse_flow.mean(),
        'test_nse_waterlevel': nse_waterlevel.mean(),
    }, 'multi_task_lstm_model.pth')
    print("模型已保存: multi_task_lstm_model.pth")
    
    print("\n训练完成！")

