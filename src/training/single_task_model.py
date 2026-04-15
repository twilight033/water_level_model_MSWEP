import os
from pathlib import Path
import random
import numpy as np
import pandas as pd
import xarray as xr
from matplotlib import pyplot as plt
import matplotlib.font_manager as fm
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

# 设置中文字体
font_path = "D:/code/TimesSong.ttf"  # 使用找到的字体文件
if os.path.exists(font_path):
    # 直接使用字体文件路径
    plt.rcParams['font.sans-serif'] = ['TimesSong']
    plt.rcParams['axes.unicode_minus'] = False
    # 注册字体
    fm.fontManager.addfont(font_path)
    print(f"使用字体: {font_path}")
else:
    # 使用系统中文字体
    plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False
    print("使用系统中文字体")


class SingleTaskDataset(Dataset):
    """单任务数据集类，用于加载径流或水位数据（batch-first）"""

    def __init__(
        self,
        basins: list,
        dates: list,
        data_attr: pd.DataFrame,
        data_forcing: xr.Dataset,
        data_target: pd.DataFrame,  # 目标数据（径流或水位）
        target_type: str = "flow",  # "flow" 或 "waterlevel"
        loader_type: str = "train",
        seq_length: int = 100,
        means: dict = None,
        stds: dict = None,
    ):
        """
        初始化单任务数据集

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
        data_target : pd.DataFrame
            目标数据（径流或水位），格式：index为时间，columns为流域ID
        target_type : str, optional
            目标类型 "flow" 或 "waterlevel"
        loader_type : str, optional
            数据集类型 "train", "valid", "test"
        seq_length : int, optional
            输入序列长度
        means : dict, optional
            归一化均值字典
        stds : dict, optional
            归一化标准差字典
        """
        super(SingleTaskDataset, self).__init__()
        if loader_type not in ["train", "valid", "test"]:
            raise ValueError(
                " 'loader_type' must be one of 'train', 'valid' or 'test' "
            )
        if target_type not in ["flow", "waterlevel"]:
            raise ValueError(
                " 'target_type' must be 'flow' or 'waterlevel' "
            )
        
        self.loader_type = loader_type
        self.target_type = target_type
        self.basins = basins
        self.dates = dates
        self.seq_length = seq_length
        self.means = means
        self.stds = stds

        self.data_attr = data_attr
        self.data_forcing = data_forcing
        self.data_target = data_target

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
        
        # 获取目标输出（序列最后一天的值）
        y = self.y[basin][time_idx + seq_length - 1]
        
        # 最终NaN检查
        if np.isnan(xc).any():
            print(f"[错误] 样本 {item}: 输入特征包含NaN")
            print(f"  流域: {basin}, 时间索引: {time_idx}")
            raise ValueError("输入特征包含NaN")
        
        if np.isnan(y):
            print(f"[错误] 样本 {item}: 目标值包含NaN")
            print(f"  流域: {basin}, 时间索引: {time_idx}")
            print(f"  y: {y}")
            raise ValueError("目标值包含NaN")
        
        return torch.from_numpy(xc).float(), torch.tensor(y).float()

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
            
            # 检查目标数据
            target_nan_count = self.data_target.isnull().sum().sum()
            if target_nan_count > 0:
                print(f"[警告] {self.target_type}数据包含 {target_nan_count} 个NaN值，将使用前向填充和均值填充")
                self.data_target = self.data_target.ffill().bfill()
                # 如果仍有NaN，使用均值填充
                remaining_nan = self.data_target.isnull().sum().sum()
                if remaining_nan > 0:
                    print(f"[警告] 前向/后向填充后仍有 {remaining_nan} 个NaN，使用均值填充")
                    self.data_target = self.data_target.fillna(self.data_target.mean())
            
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
            df_std_attr = numeric_attrs.std()
            
            # 检查计算出的统计量
            if df_mean_attr.isnull().any() or df_std_attr.isnull().any():
                print("[错误] 属性数据统计量包含NaN")
                raise ValueError("属性数据统计量包含NaN")
            
            self.means['attr'] = df_mean_attr
            self.stds['attr'] = df_std_attr
            
            # 目标数据的均值和标准差
            target_numeric = self.data_target.select_dtypes(include=[np.number])
            target_mean = target_numeric.mean().mean()
            target_std = target_numeric.std().mean()
            
            # 检查计算出的统计量
            if np.isnan(target_mean) or np.isnan(target_std) or target_std == 0:
                print(f"[错误] {self.target_type}数据统计量异常: mean={target_mean}, std={target_std}")
                raise ValueError(f"{self.target_type}数据统计量异常")
            
            self.means[self.target_type] = target_mean
            self.stds[self.target_type] = target_std
            
            # 最终NaN检查
            final_target_nan = self.data_target.isnull().sum().sum()
            final_attr_nan = self.data_attr[numeric_cols].isnull().sum().sum()
            
            if final_target_nan > 0 or final_attr_nan > 0:
                print(f"[错误] 数据清洗后仍有NaN: {self.target_type}={final_target_nan}, 属性={final_attr_nan}")
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
        self.y = self._normalize_target(self.data_target)
        
        # 在测试模式下，同时保存原始数据用于评估
        if not train_mode:
            self.y_raw = self._dataframe_to_dict(self.data_target)
        
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

    def _normalize_target(self, data_target):
        """归一化目标数据"""
        result = {}
        for basin in self.basins:
            values = data_target[basin].values
            # 处理NaN值
            if np.isnan(values).any():
                print(f"[警告] 流域 {basin} 的{self.target_type}数据在归一化前包含NaN，进行插值处理")
                values = pd.Series(values).ffill().bfill().fillna(self.means[self.target_type]).values
            
            normalized = (values - self.means[self.target_type]) / self.stds[self.target_type]
            
            # 检查归一化后是否还有NaN
            if np.isnan(normalized).any():
                print(f"[错误] 流域 {basin} 的{self.target_type}数据归一化后仍包含NaN")
                # 用0填充剩余的NaN
                normalized = np.nan_to_num(normalized, nan=0.0)
            
            result[basin] = normalized
        return result

    def _create_lookup_table(self):
        """创建索引表"""
        lookup = []
        seq_length = self.seq_length
        time_length = len(self.data_target)
        
        for basin in tqdm(self.basins, desc="创建索引表"):
            for j in range(time_length - seq_length + 1):
                lookup.append((basin, j))
        
        self.lookup_table = {i: elem for i, elem in enumerate(lookup)}
        self.num_samples = len(self.lookup_table)

    def get_means(self):
        return self.means

    def get_stds(self):
        return self.stds

    def local_denormalization(self, feature):
        """反归一化"""
        return feature * self.stds[self.target_type] + self.means[self.target_type]
    
    def get_raw_targets(self):
        """获取原始目标数据（仅测试模式）"""
        if hasattr(self, 'y_raw'):
            return self.y_raw
        else:
            # 如果没有原始数据，使用反归一化
            raw_data = {}
            for basin in self.y:
                raw_data[basin] = self.local_denormalization(self.y[basin])
            return raw_data


class SingleTaskLSTM(nn.Module):
    """单任务LSTM网络，预测径流或水位"""

    def __init__(
        self, 
        input_size: int, 
        hidden_size: int, 
        dropout_rate: float = 0.0
    ):
        """
        构建单任务LSTM模型

        Parameters
        ----------
        input_size : int
            输入特征维度
        hidden_size : int
            LSTM隐藏层大小
        dropout_rate : float, optional
            Dropout比率
        """
        super(SingleTaskLSTM, self).__init__()

        # LSTM层
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=2,
            bias=True,
            batch_first=True,
        )
        self.dropout = nn.Dropout(p=dropout_rate)
        
        # 输出层
        self.fc = nn.Linear(in_features=hidden_size, out_features=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播

        Returns
        -------
        torch.Tensor
            预测值
        """
        output, (h_n, c_n) = self.lstm(x)

        # 使用最后一层的隐藏状态
        hidden = self.dropout(h_n[-1, :, :])
        
        # 预测输出
        pred = self.fc(hidden)
        
        return pred


def train_epoch(model, optimizer, loader, loss_func, epoch):
    """训练一个epoch - 增强NaN检查和错误处理"""
    model.train()
    pbar = tqdm(loader)
    pbar.set_description(f"Epoch {epoch}")
    
    epoch_loss = 0.0
    
    for batch_idx, (xs, ys) in enumerate(pbar):
        # 检查输入数据
        if torch.isnan(xs).any() or torch.isnan(ys).any():
            print(f"[错误] Epoch {epoch}, 批次 {batch_idx}: 输入数据包含NaN")
            raise ValueError("输入数据包含NaN")
        
        optimizer.zero_grad()
        xs, ys = xs.to(DEVICE), ys.to(DEVICE)
        
        # 获取模型预测
        try:
            pred = model(xs)
        except Exception as e:
            print(f"[错误] Epoch {epoch}, 批次 {batch_idx}: 前向传播失败 - {e}")
            raise e
        
        # 检查预测结果
        if torch.isnan(pred).any():
            print(f"[错误] Epoch {epoch}, 批次 {batch_idx}: 预测结果包含NaN")
            raise ValueError("预测结果包含NaN")
        
        # 计算损失
        loss = loss_func(pred.squeeze(), ys)
        
        # 检查损失
        if torch.isnan(loss):
            print(f"[错误] Epoch {epoch}, 批次 {batch_idx}: 损失包含NaN")
            print(f"  loss: {loss.item()}")
            raise ValueError("损失包含NaN")
        
        # 反向传播
        loss.backward()
        
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
        epoch_loss += loss.item()
        
        # 更新进度条
        pbar.set_postfix_str(f"Loss: {loss.item():.4f}, Grad: {grad_norm:.4f}")
    
    # 检查epoch平均损失
    avg_loss = epoch_loss / len(loader)
    
    if np.isnan(avg_loss):
        print(f"[错误] Epoch {epoch}: 平均损失包含NaN")
        raise ValueError("平均损失包含NaN")
    
    return avg_loss


def eval_model(model, loader, dataset):
    """评估模型"""
    model.eval()
    obs = []
    preds = []
    
    with torch.no_grad():
        for xs, ys in loader:
            xs = xs.to(DEVICE)
            
            # 获取模型预测
            pred = model(xs)
            
            obs.append(ys.numpy())
            preds.append(pred.cpu().numpy())

    preds = np.concatenate(preds, axis=0)
    obs = np.concatenate(obs, axis=0)
    
    # 反归一化预测值
    preds_denorm = dataset.local_denormalization(preds)
    
    # 获取原始观测值（如果是测试模式）
    if hasattr(dataset, 'get_raw_targets'):
        raw_targets = dataset.get_raw_targets()
        # 重新构建观测值数组，使用原始数据
        obs_denorm = []
        for xs, ys in DataLoader(dataset, batch_size=256, shuffle=False, num_workers=0):
            for i, (x, y) in enumerate(zip(xs, ys)):
                basin, time_idx = dataset.lookup_table[len(obs_denorm)]
                obs_denorm.append(raw_targets[basin][time_idx + dataset.seq_length - 1])
        obs_denorm = np.array(obs_denorm).reshape(-1, 1)
    else:
        # 备用方案：使用反归一化
        obs_denorm = dataset.local_denormalization(obs)
    
    # 计算NSE
    try:
        nse = he.evaluator(he.nse, obs_denorm.flatten(), preds_denorm.flatten())[0]
    except:
        def calculate_nse(obs, sim):
            numerator = np.sum((obs - sim) ** 2)
            denominator = np.sum((obs - np.mean(obs)) ** 2)
            return 1 - (numerator / denominator)
        nse = calculate_nse(obs_denorm.flatten(), preds_denorm.flatten())
    
    return obs_denorm, preds_denorm, nse


def train_single_task_model(target_type, num_epochs=10):
    """训练单任务模型"""
    print(f"\n=== 训练单任务{target_type}模型 ===")
    
    # 导入配置
    from config import (
        CAMELSH_DATA_PATH, NUM_BASINS, SEQUENCE_LENGTH, BATCH_SIZE,
        TRAIN_START, TRAIN_END, VALID_START, VALID_END, TEST_START, TEST_END,
        FORCING_VARIABLES, ATTRIBUTE_VARIABLES, VALID_WATER_LEVEL_BASINS
    )
    
    # 设置随机种子
    random.seed(1234)
    np.random.seed(1234)
    torch.manual_seed(1234)
    
    # 使用改进的CAMELSH读取器
    camelsh_reader = ImprovedCAMELSHReader(CAMELSH_DATA_PATH, download=False, use_batch=True)
    
    # 选择流域（与多任务模型保持一致）
    chosen_basins = VALID_WATER_LEVEL_BASINS[:NUM_BASINS]
    print(f"选择的流域: {chosen_basins}")
    
    # 时间范围
    train_times = [TRAIN_START, TRAIN_END]
    valid_times = [VALID_START, VALID_END]
    test_times = [TEST_START, TEST_END]
    
    # 特征变量
    chosen_forcing_vars = []
    for var_name in FORCING_VARIABLES:
        if hasattr(StandardVariable, var_name.upper()):
            chosen_forcing_vars.append(getattr(StandardVariable, var_name.upper()))
        else:
            chosen_forcing_vars.append(var_name)
    
    chosen_attrs_vars = ATTRIBUTE_VARIABLES
    
    print(f"选择的气象变量: {FORCING_VARIABLES}")
    print(f"选择的属性变量: {ATTRIBUTE_VARIABLES}")
    
    # 加载属性数据
    camelsh = camelsh_reader.camelsh
    attrs = camelsh.read_attr_xrdataset(
        gage_id_lst=chosen_basins,
        var_lst=chosen_attrs_vars
    )
    
    # 加载气象强迫数据
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
    
    # 加载目标数据
    if target_type == "flow":
        target_var = StandardVariable.STREAMFLOW
    elif target_type == "waterlevel":
        target_var = StandardVariable.WATER_LEVEL
    else:
        raise ValueError("target_type must be 'flow' or 'waterlevel'")
    
    train_target_ds = camelsh_reader.read_ts_xrdataset(
        gage_id_lst=chosen_basins,
        t_range=train_times,
        var_lst=[target_var]
    )
    valid_target_ds = camelsh_reader.read_ts_xrdataset(
        gage_id_lst=chosen_basins,
        t_range=valid_times,
        var_lst=[target_var]
    )
    test_target_ds = camelsh_reader.read_ts_xrdataset(
        gage_id_lst=chosen_basins,
        t_range=test_times,
        var_lst=[target_var]
    )
    
    # 转换为pandas DataFrame格式
    train_target = train_target_ds[target_var].to_pandas().T
    valid_target = valid_target_ds[target_var].to_pandas().T
    test_target = test_target_ds[target_var].to_pandas().T
    
    # 转换属性数据为pandas DataFrame格式
    attrs_df = attrs.to_pandas()
    attrs_df.index.name = 'gauge_id'
    attrs_df = attrs_df.reset_index()
    
    print(f"{target_type}数据形状: {train_target.shape}")
    print(f"{target_type}数据范围: {train_target.min().min():.3f} - {train_target.max().max():.3f}")
    
    # 创建数据集
    sequence_length = SEQUENCE_LENGTH
    batch_size = BATCH_SIZE
    
    # 训练数据集
    ds_train = SingleTaskDataset(
        basins=chosen_basins,
        dates=train_times,
        data_attr=attrs_df,
        data_forcing=train_forcings,
        data_target=train_target,
        target_type=target_type,
        loader_type="train",
        seq_length=sequence_length,
    )
    tr_loader = DataLoader(ds_train, batch_size=batch_size, shuffle=True, num_workers=0)
    
    # 验证数据集
    means = ds_train.get_means()
    stds = ds_train.get_stds()
    ds_valid = SingleTaskDataset(
        basins=chosen_basins,
        dates=valid_times,
        data_attr=attrs_df,
        data_forcing=valid_forcings,
        data_target=valid_target,
        target_type=target_type,
        loader_type="valid",
        seq_length=sequence_length,
        means=means,
        stds=stds,
    )
    val_loader = DataLoader(ds_valid, batch_size=1000, shuffle=False, num_workers=0)
    
    # 测试数据集
    ds_test = SingleTaskDataset(
        basins=chosen_basins,
        dates=test_times,
        data_attr=attrs_df,
        data_forcing=test_forcings,
        data_target=test_target,
        target_type=target_type,
        loader_type="test",
        seq_length=sequence_length,
        means=means,
        stds=stds,
    )
    test_loader = DataLoader(ds_test, batch_size=1000, shuffle=False, num_workers=0)
    
    # 创建模型
    input_size = len(chosen_attrs_vars) + len(chosen_forcing_vars)
    hidden_size = 64  # 与多任务模型保持一致
    dropout_rate = 0.2
    learning_rate = 1e-3
    
    model = SingleTaskLSTM(
        input_size=input_size, 
        hidden_size=hidden_size, 
        dropout_rate=dropout_rate
    ).to(DEVICE)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    loss_func = nn.MSELoss()
    
    print(f"模型参数量: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")
    print(f"设备: {DEVICE}")
    
    # 训练模型
    print(f"\n开始训练{target_type}模型...")
    train_losses = []
    val_nses = []
    
    for i in range(num_epochs):
        # 训练
        train_loss = train_epoch(model, optimizer, tr_loader, loss_func, i + 1)
        train_losses.append(train_loss)
        
        # 验证
        obs_val, pred_val, nse_val = eval_model(model, val_loader, ds_valid)
        val_nses.append(nse_val)
        
        tqdm.write(f"Epoch {i+1} - 验证集 NSE: {nse_val:.4f}")
    
    # 测试
    print(f"\n在测试集上评估{target_type}模型...")
    obs_test, pred_test, nse_test = eval_model(model, test_loader, ds_test)
    
    print(f"测试集 NSE: {nse_test:.4f}")
    
    # 可视化结果
    print(f"\n正在生成{target_type}预测可视化图表...")
    
    # 准备日期范围
    start_date = pd.to_datetime(ds_test.dates[0], format="%Y-%m-%d") + pd.DateOffset(
        days=sequence_length - 1
    )
    actual_length = len(obs_test)
    date_range = pd.date_range(start_date, periods=actual_length, freq='H')
    
    # 只显示前1000个点以便观察
    plot_length = min(1000, actual_length)
    
    fig, ax = plt.subplots(figsize=(14, 6))
    
    ax.plot(date_range[:plot_length], obs_test[:plot_length], label="观测值", alpha=0.7)
    ax.plot(date_range[:plot_length], pred_test[:plot_length], label="预测值", alpha=0.7)
    ax.legend()
    
    target_name = "径流" if target_type == "flow" else "水位"
    unit = "(mm/d)" if target_type == "flow" else "(m)"
    
    ax.set_title(f"{target_name}预测结果 (前{plot_length}个点) - 测试集 NSE: {nse_test:.3f}")
    ax.set_xlabel("日期")
    ax.set_ylabel(f"{target_name} {unit}")
    ax.grid(True, alpha=0.3)
    
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    # 保存图片
    output_file = f"single_task_{target_type}_results.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"已保存图片: {output_file}")
    plt.show()
    
    # 保存模型
    model_file = f"single_task_{target_type}_model.pth"
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'means': means,
        'stds': stds,
        'nse_test': nse_test,
        'target_type': target_type,
    }, model_file)
    print(f"模型已保存: {model_file}")
    
    return model, means, stds, nse_test


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
    
    # 训练径流模型
    print("训练径流预测模型...")
    flow_model, flow_means, flow_stds, flow_nse = train_single_task_model("flow", num_epochs=10)
    
    # 训练水位模型
    print("\n训练水位预测模型...")
    waterlevel_model, wl_means, wl_stds, wl_nse = train_single_task_model("waterlevel", num_epochs=10)
    
    print(f"\n=== 训练完成 ===")
    print(f"径流模型测试集 NSE: {flow_nse:.4f}")
    print(f"水位模型测试集 NSE: {wl_nse:.4f}")





