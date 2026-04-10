"""
多任务LSTM消融实验（HL2Q 水位→径流级联架构）：标签缺失情况下的模型性能
测试场景：径流/水位数据在10%/30%/50%随机缺失情况下的模型表现

与 multi_task_lstm_ablation_hl2q.py 的区别：
1. create_missing_mask 使用局部随机生成器（np.random.default_rng），不污染全局状态
2. 每个缺失场景用多个不同的随机种子重复实验，结果以均值 ± 标准差表示
3. 图表加入误差棒；CSV 同时保存每次原始结果和聚合结果
"""
import os
from pathlib import Path
import random
from typing import Dict
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
from mswep_loader import load_mswep_data, merge_forcing_with_mswep

DEVICE = torch.device(
    "cuda:0" if torch.cuda.is_available() else "cpu"
)

# 时间划分比例
TRAIN_RATIO = 0.6
VALID_RATIO = 0.2
TEST_RATIO = 0.2
WINDOW_STEP = 3

# ── 重复实验配置 ────────────────────────────────────────────────────────────────
# 每个缺失场景重复的随机种子列表（用于生成不同的缺失 mask）
# 基线场景（0%缺失）不受 mask_seed 影响，但仍重复以评估模型训练的随机性
REPEAT_SEEDS = [42, 123, 456]


def print_device_info():
    """打印设备信息"""
    print("\n" + "=" * 60)
    print("设备信息")
    print("=" * 60)
    print(f"PyTorch版本: {torch.__version__}")
    print(f"CUDA可用: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA版本: {torch.version.cuda}")
        print(f"GPU数量: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            gpu_name = torch.cuda.get_device_name(i)
            gpu_memory = torch.cuda.get_device_properties(i).total_memory / 1024**3
            print(f"  GPU {i}: {gpu_name}")
            print(f"    显存: {gpu_memory:.2f} GB")
        print(f"当前使用设备: {DEVICE}")
    else:
        print("使用CPU进行训练")
        print(f"CPU核心数: {torch.get_num_threads()}")
    print("=" * 60 + "\n")


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


def create_missing_mask(data_df, missing_ratio, seed=42):
    """
    创建标签缺失mask（使用局部随机生成器，不影响全局 numpy 随机状态）

    Parameters
    ----------
    data_df : pd.DataFrame
        原始数据
    missing_ratio : float
        缺失比例 (0.0-1.0)，0表示不缺失（保持完整）
    seed : int
        随机种子（仅作用于本函数内的局部生成器）

    Returns
    -------
    pd.DataFrame
        带有人工缺失的数据（缺失部分设为NaN）
    """
    if missing_ratio == 0.0:
        print(f"  保持数据完整（无人工缺失）")
        return data_df.copy()

    # 局部随机生成器，不污染全局 np.random 状态
    rng = np.random.default_rng(seed)

    masked_data = data_df.copy()
    original_valid = masked_data.notna()

    for col in masked_data.columns:
        valid_indices = original_valid[col]
        valid_count = valid_indices.sum()

        if valid_count > 0:
            n_to_hide = int(valid_count * missing_ratio)
            if n_to_hide > 0:
                valid_positions = np.where(valid_indices)[0]
                hide_positions = rng.choice(valid_positions, size=n_to_hide, replace=False)
                masked_data.iloc[hide_positions, masked_data.columns.get_loc(col)] = np.nan

    original_valid_count = original_valid.sum().sum()
    final_valid_count = masked_data.notna().sum().sum()
    actual_missing_ratio = 1 - (final_valid_count / original_valid_count)

    print(f"  原始有效数据点: {original_valid_count}")
    print(f"  人工隐藏后有效数据点: {final_valid_count}")
    print(f"  实际缺失比例: {actual_missing_ratio:.2%}")

    return masked_data


class MultiTaskDatasetWithMissingLabels(Dataset):
    """支持标签缺失的多任务数据集类"""

    def __init__(
        self,
        basins: list,
        dates: list,
        data_attr: pd.DataFrame,
        data_forcing: xr.Dataset,
        data_flow: pd.DataFrame,
        data_waterlevel: pd.DataFrame,
        loader_type: str = "train",
        seq_length: int = 100,
        means: dict = None,
        stds: dict = None,
    ):
        super(MultiTaskDatasetWithMissingLabels, self).__init__()
        if loader_type not in ["train", "valid", "test"]:
            raise ValueError(
                " 'loader_type' must be one of 'train', 'valid' or 'test' "
            )
        else:
            self.loader_type = loader_type

        self.basins = [str(b) for b in basins]
        self.dates = dates
        self.seq_length = seq_length
        self.means = means if means is not None else {}
        self.stds = stds if stds is not None else {}
        self.data_attr = data_attr
        self.data_forcing = data_forcing
        self.data_flow = data_flow
        self.data_waterlevel = data_waterlevel

        self._load_data()

    def __len__(self):
        return self.num_samples

    def __getitem__(self, item: int):
        basin, time_idx = self.lookup_table[item]
        seq_length = self.seq_length

        if hasattr(self, 'forcing_time_index') and self.forcing_time_index is not None:
            if time_idx not in self.forcing_time_index:
                closest_idx = self.forcing_time_index.get_indexer([time_idx], method='nearest')[0]
                start_pos = closest_idx
            else:
                start_pos = self.forcing_time_index.get_loc(time_idx)
        else:
            start_pos = self.data_flow.index.get_loc(time_idx)

        x = self.x[basin][start_pos : start_pos + seq_length]
        c = self.c.loc[basin].values
        c = np.tile(c, (seq_length, 1))
        xc = np.concatenate((x, c), axis=1)

        target_time_idx = time_idx
        end_time = target_time_idx + pd.Timedelta(hours=(seq_length - 1) * 3)

        if end_time in self.data_flow.index:
            target_end_pos = self.data_flow.index.get_loc(end_time)
            y_flow = self.y_flow[basin][target_end_pos]
            y_waterlevel = self.y_waterlevel[basin][target_end_pos]
        else:
            nearest_end_idx = self.data_flow.index.get_indexer([end_time], method='nearest')[0]
            y_flow = self.y_flow[basin][nearest_end_idx]
            y_waterlevel = self.y_waterlevel[basin][nearest_end_idx]

        flow_mask = 0.0 if np.isnan(y_flow) else 1.0
        waterlevel_mask = 0.0 if np.isnan(y_waterlevel) else 1.0

        y_flow = 0.0 if np.isnan(y_flow) else y_flow
        y_waterlevel = 0.0 if np.isnan(y_waterlevel) else y_waterlevel

        y = np.array([y_flow, y_waterlevel])
        mask = np.array([flow_mask, waterlevel_mask])

        return (
            torch.from_numpy(xc).float(),
            torch.from_numpy(y).float(),
            torch.from_numpy(mask).float(),
            basin
        )

    def _load_data(self):
        """从文件加载数据"""
        if self.loader_type == "train":
            train_mode = True
            self.means = {}
            self.stds = {}

            print("检查原始数据中的NaN...")

            forcing_nan_count = self.data_forcing.isnull().sum().sum()
            print(f"强迫数据NaN统计: {forcing_nan_count}")
            if forcing_nan_count > 0:
                print(f"[警告] 强迫数据包含NaN值，将使用插值填充")
                self.data_forcing = self.data_forcing.interpolate_na(dim='time', method='linear')
                remaining_nan = self.data_forcing.isnull().sum().sum()
                if remaining_nan > 0:
                    self.data_forcing = self.data_forcing.fillna(self.data_forcing.mean())

            attr_nan_count = self.data_attr.isnull().sum().sum()
            if attr_nan_count > 0:
                print(f"[警告] 属性数据包含 {attr_nan_count} 个NaN值，将使用均值填充")
                numeric_cols = [col for col in self.data_attr.columns if col != 'gauge_id']
                self.data_attr[numeric_cols] = self.data_attr[numeric_cols].fillna(self.data_attr[numeric_cols].mean())

            flow_nan_count = self.data_flow.isnull().sum().sum()
            flow_total = self.data_flow.size
            flow_valid_ratio = (flow_total - flow_nan_count) / flow_total if flow_total > 0 else 0
            print(f"[信息] 径流数据统计（含人工缺失）:")
            print(f"  - 总数据点: {flow_total}")
            print(f"  - 有效数据点: {flow_total - flow_nan_count} ({flow_valid_ratio:.1%})")
            print(f"  - NaN数据点: {flow_nan_count} ({flow_nan_count/flow_total:.1%})")

            wl_nan_count = self.data_waterlevel.isnull().sum().sum()
            wl_total = self.data_waterlevel.size
            wl_valid_ratio = (wl_total - wl_nan_count) / wl_total if wl_total > 0 else 0
            print(f"[信息] 水位数据统计（含人工缺失）:")
            print(f"  - 总数据点: {wl_total}")
            print(f"  - 有效数据点: {wl_total - wl_nan_count} ({wl_valid_ratio:.1%})")
            print(f"  - NaN数据点: {wl_nan_count} ({wl_nan_count/wl_total:.1%})")

            df_mean_forcings = self.data_forcing.mean().to_pandas()
            df_std_forcings = self.data_forcing.std().to_pandas()
            self.means['forcing'] = df_mean_forcings
            self.stds['forcing'] = df_std_forcings

            numeric_cols = [col for col in self.data_attr.columns if col != 'gauge_id']
            numeric_attrs = self.data_attr[numeric_cols]
            df_mean_attr = numeric_attrs.mean()
            df_std_attr = numeric_attrs.std(ddof=0).fillna(0.0)
            self.means['attr'] = df_mean_attr
            self.stds['attr'] = df_std_attr

            flow_mean = {}
            flow_std = {}
            for basin in self.basins:
                series = self.data_flow[basin]
                valid_data = series.dropna()
                if len(valid_data) > 0:
                    flow_mean[basin] = valid_data.mean()
                    flow_std[basin] = valid_data.std() if valid_data.std() > 1e-6 else 1.0
                else:
                    flow_mean[basin] = 0.0
                    flow_std[basin] = 1.0

            self.means['flow'] = flow_mean
            self.stds['flow'] = flow_std

            waterlevel_mean = {}
            waterlevel_std = {}
            for basin in self.basins:
                series = self.data_waterlevel[basin]
                valid_data = series.dropna()
                if len(valid_data) > 0:
                    waterlevel_mean[basin] = valid_data.mean()
                    waterlevel_std[basin] = valid_data.std() if valid_data.std() > 1e-6 else 1.0
                else:
                    waterlevel_mean[basin] = 0.0
                    waterlevel_std[basin] = 1.0

            self.means['waterlevel'] = waterlevel_mean
            self.stds['waterlevel'] = waterlevel_std

            print("数据统计量计算完成")
        else:
            train_mode = False

        print("开始数据归一化...")
        self.forcing_time_index = pd.DatetimeIndex(self.data_forcing.time.values)
        self.x = self._normalize_forcing(self.data_forcing)

        successfully_normalized_basins = list(self.x.keys())
        if len(successfully_normalized_basins) < len(self.basins):
            skipped_count = len(self.basins) - len(successfully_normalized_basins)
            print(f"[信息] {skipped_count} 个流域因强迫数据问题被跳过")
            self.basins = successfully_normalized_basins
            print(f"归一化后实际可用的流域数量: {len(self.basins)} 个")

        if len(self.basins) == 0:
            raise ValueError("归一化后没有任何可用流域！")

        print("强迫数据归一化完成")
        self.c = self._normalize_attr(self.data_attr)
        print("属性数据归一化完成")

        self.y_flow = self._normalize_flow(self.data_flow)
        self.y_waterlevel = self._normalize_waterlevel(self.data_waterlevel)

        if not train_mode:
            self.y_flow_raw = self._dataframe_to_dict(self.data_flow)
            self.y_waterlevel_raw = self._dataframe_to_dict(self.data_waterlevel)

        self.train_mode = train_mode
        self._create_lookup_table()

    def _dataframe_to_dict(self, df):
        result = {}
        for basin in self.basins:
            result[basin] = df[basin].values
        return result

    def _normalize_forcing(self, data_forcing):
        result = {}
        std_values = self.stds['forcing'].values.copy()
        zero_std_mask = std_values < 1e-8
        if np.any(zero_std_mask):
            std_values[zero_std_mask] = 1.0

        available_basins_in_data = [str(b) for b in data_forcing.basin.values]

        for basin in self.basins:
            basin_str = str(basin)
            if basin_str not in available_basins_in_data:
                print(f"[警告] 流域 {basin_str} 不在强迫数据中，跳过")
                continue

            try:
                basin_data = data_forcing.sel(basin=basin_str).to_array().to_numpy().T
            except KeyError:
                try:
                    basin_data = data_forcing.sel(basin=basin).to_array().to_numpy().T
                except KeyError:
                    print(f"[警告] 流域 {basin_str} 无法从强迫数据中提取，跳过")
                    continue

            if np.isnan(basin_data).any():
                nan_ratio = np.isnan(basin_data).sum() / basin_data.size * 100
                print(f"[警告] 流域 {basin_str} 强迫数据包含NaN ({nan_ratio:.1f}%)，跳过该流域")
                continue

            normalized = (basin_data - self.means['forcing'].values) / std_values

            if np.isnan(normalized).any():
                print(f"[警告] 流域 {basin_str} 归一化后强迫数据包含NaN，跳过该流域")
                continue

            result[basin_str] = normalized
        return result

    def _normalize_attr(self, data_attr):
        gauge_ids = data_attr['gauge_id']
        numeric_cols = [col for col in data_attr.columns if col != 'gauge_id']
        numeric_attrs = data_attr[numeric_cols]

        zero_std_mask = self.stds['attr'] < 1e-8
        if zero_std_mask.any():
            std_values = self.stds['attr'].copy()
            std_values[zero_std_mask] = 1.0
        else:
            std_values = self.stds['attr']

        normalized = (numeric_attrs - self.means['attr']) / std_values
        normalized.index = gauge_ids
        return normalized

    def _normalize_flow(self, data_flow):
        result = {}
        for basin in self.basins:
            values = data_flow[basin].values
            if basin not in self.means.get('flow', {}):
                raise KeyError(f"流域 {basin} 的径流归一化参数不存在")
            mean = self.means['flow'][basin]
            std = self.stds['flow'][basin]
            result[basin] = (values - mean) / std
        return result

    def _normalize_waterlevel(self, data_waterlevel):
        result = {}
        for basin in self.basins:
            values = data_waterlevel[basin].values
            if basin not in self.means.get('waterlevel', {}):
                raise KeyError(f"流域 {basin} 的水位归一化参数不存在")
            mean = self.means['waterlevel'][basin]
            std = self.stds['waterlevel'][basin]
            result[basin] = (values - mean) / std
        return result

    def _create_lookup_table(self):
        """
        创建查找表，包含有至少一个有效标签的样本
        注意：训练时允许只有一个任务有标签，另一个缺失
        """
        lookup = []
        seq_length = self.seq_length
        forcing_times = self.forcing_time_index

        for basin in tqdm(self.basins, desc=f"创建 {self.loader_type} 索引表", disable=False):
            if basin not in self.y_flow or basin not in self.y_waterlevel:
                continue

            flow_values = self.y_flow[basin]
            wl_values = self.y_waterlevel[basin]
            target_index = self.data_flow.index

            flow_valid_mask = ~np.isnan(flow_values)
            wl_valid_mask = ~np.isnan(wl_values)
            at_least_one_valid = flow_valid_mask | wl_valid_mask

            valid_target_times = set(target_index[at_least_one_valid])

            valid_forcing_times = []
            for ft in forcing_times:
                end_time = ft + pd.Timedelta(hours=(seq_length - 1) * 3)
                if end_time in valid_target_times:
                    valid_forcing_times.append(ft)

            if len(valid_forcing_times) < 1:
                continue

            first_valid_time = valid_forcing_times[0]
            last_valid_time = valid_forcing_times[-1]
            start_idx = forcing_times.get_loc(first_valid_time)
            end_idx = forcing_times.get_loc(last_valid_time)
            total_span = end_idx - start_idx + 1

            train_end_idx = start_idx + int(total_span * TRAIN_RATIO)
            valid_end_idx = start_idx + int(total_span * (TRAIN_RATIO + VALID_RATIO))

            if self.loader_type == "train":
                range_start_idx = start_idx
                range_end_idx = train_end_idx
            elif self.loader_type == "valid":
                range_start_idx = train_end_idx
                range_end_idx = valid_end_idx
            else:
                range_start_idx = valid_end_idx
                range_end_idx = end_idx + 1

            if range_end_idx - range_start_idx < seq_length:
                continue

            for idx in range(range_start_idx, range_end_idx - seq_length + 1, WINDOW_STEP):
                window_start_time = forcing_times[idx]
                window_end_time = window_start_time + pd.Timedelta(hours=(seq_length - 1) * 3)
                if window_end_time in valid_target_times:
                    lookup.append((basin, window_start_time))

        self.lookup_table = {i: elem for i, elem in enumerate(lookup)}
        self.num_samples = len(self.lookup_table)
        print(f"\n{self.loader_type.upper()} 数据集统计:")
        print(f"  - 总样本数: {self.num_samples}")

        if self.num_samples == 0:
            raise ValueError(f"{self.loader_type} 数据集没有生成任何样本！")

    def get_means(self):
        return self.means

    def get_stds(self):
        return self.stds

    def local_denormalization(self, feature, basin, variable="flow"):
        basin = str(basin)
        if variable == "flow":
            mean = self.means['flow'][basin]
            std = self.stds['flow'][basin]
            return feature * std + mean
        elif variable == "waterlevel":
            mean = self.means['waterlevel'][basin]
            std = self.stds['waterlevel'][basin]
            return feature * std + mean
        else:
            raise ValueError(f"Unknown variable: {variable}")


class MultiTaskLSTM(nn.Module):
    """
    水位→径流级联多任务 LSTM 网络（HL2Q）—— 消融实验版

    Stage-1: 共享 LSTM + 水位预测头
    Stage-2: 径流预测头（接收共享隐层 + 水位预测经投影后的特征）
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        dropout_rate: float = 0.0,
        task_weights: dict = None,
        hl_proj_size: int = 16,
        stop_gradient: bool = False,
    ):
        super(MultiTaskLSTM, self).__init__()

        self.stop_gradient = stop_gradient

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=2,
            bias=True,
            batch_first=True,
        )
        self.dropout = nn.Dropout(p=dropout_rate)

        self.fc_waterlevel = nn.Linear(hidden_size, 1)

        self.hl_proj = nn.Sequential(
            nn.Linear(1, hl_proj_size),
            nn.ReLU(),
        )
        self.fc_flow = nn.Linear(hidden_size + hl_proj_size, 1)

        self.task_weights = task_weights if task_weights is not None else {'flow': 1.0, 'waterlevel': 1.0}

    def forward(self, x: torch.Tensor) -> tuple:
        _, (h_n, _) = self.lstm(x)
        hidden = self.dropout(h_n[-1])

        pred_waterlevel = self.fc_waterlevel(hidden)

        hl_input = pred_waterlevel.detach() if self.stop_gradient else pred_waterlevel
        hl_feat = self.hl_proj(hl_input)
        flow_input = torch.cat([hidden, hl_feat], dim=-1)
        pred_flow = self.fc_flow(flow_input)

        return pred_flow, pred_waterlevel


def multi_task_loss_with_mask(pred_flow, pred_waterlevel, target, mask, loss_func, task_weights):
    """带mask的多任务损失函数"""
    flow_mask = mask[:, 0:1]
    n_flow_valid = flow_mask.sum()
    if n_flow_valid > 0:
        loss_flow = (loss_func(pred_flow, target[:, 0:1]) * flow_mask).sum() / n_flow_valid
    else:
        loss_flow = torch.tensor(0.0, device=pred_flow.device)

    waterlevel_mask = mask[:, 1:2]
    n_waterlevel_valid = waterlevel_mask.sum()
    if n_waterlevel_valid > 0:
        loss_waterlevel = (loss_func(pred_waterlevel, target[:, 1:2]) * waterlevel_mask).sum() / n_waterlevel_valid
    else:
        loss_waterlevel = torch.tensor(0.0, device=pred_waterlevel.device)

    total_loss = (
        task_weights['flow'] * loss_flow +
        task_weights['waterlevel'] * loss_waterlevel
    )

    stats = {
        'n_flow_valid': int(n_flow_valid.item()),
        'n_waterlevel_valid': int(n_waterlevel_valid.item()),
    }

    return total_loss, loss_flow, loss_waterlevel, stats


def train_epoch(model, optimizer, loader, loss_func, epoch):
    """训练一个epoch（支持标签缺失）"""
    model.train()
    pbar = tqdm(loader)
    pbar.set_description(f"Epoch {epoch}")

    epoch_loss = 0.0
    epoch_loss_flow = 0.0
    epoch_loss_waterlevel = 0.0
    total_flow_samples = 0
    total_waterlevel_samples = 0

    for batch_idx, (xs, ys, masks, basins) in enumerate(pbar):
        optimizer.zero_grad()
        xs, ys, masks = xs.to(DEVICE), ys.to(DEVICE), masks.to(DEVICE)

        pred_flow, pred_waterlevel = model(xs)

        total_loss, loss_flow, loss_waterlevel, stats = multi_task_loss_with_mask(
            pred_flow, pred_waterlevel, ys, masks, loss_func, model.task_weights
        )

        if stats['n_flow_valid'] == 0 and stats['n_waterlevel_valid'] == 0:
            continue

        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        epoch_loss += total_loss.item()
        epoch_loss_flow += loss_flow.item()
        epoch_loss_waterlevel += loss_waterlevel.item()
        total_flow_samples += stats['n_flow_valid']
        total_waterlevel_samples += stats['n_waterlevel_valid']

        pbar.set_postfix_str(
            f"Loss: {total_loss.item():.4f} "
            f"(Q: {loss_flow.item():.4f}/{stats['n_flow_valid']}, "
            f"h: {loss_waterlevel.item():.4f}/{stats['n_waterlevel_valid']})"
        )

    avg_loss = epoch_loss / len(loader)
    avg_loss_flow = epoch_loss_flow / len(loader)
    avg_loss_waterlevel = epoch_loss_waterlevel / len(loader)

    print(f"  训练样本统计: 径流 {total_flow_samples}, 水位 {total_waterlevel_samples}")

    return avg_loss, avg_loss_flow, avg_loss_waterlevel


def eval_model(model, loader):
    """评估模型（只收集有有效标签的样本）"""
    model.eval()

    obs_flow = {}
    obs_waterlevel = {}
    preds_flow = {}
    preds_waterlevel = {}

    with torch.no_grad():
        for xs, ys, masks, basins in loader:
            xs = xs.to(DEVICE)
            pred_flow, pred_waterlevel = model(xs)

            ys_np = ys.numpy()
            masks_np = masks.numpy()
            pf_np = pred_flow.cpu().numpy().squeeze()
            pw_np = pred_waterlevel.cpu().numpy().squeeze()

            if pf_np.ndim == 0:
                pf_np = pf_np.reshape(1)
            if pw_np.ndim == 0:
                pw_np = pw_np.reshape(1)

            for i, basin in enumerate(basins):
                b = str(basin)

                if b not in obs_flow:
                    obs_flow[b] = []
                    obs_waterlevel[b] = []
                    preds_flow[b] = []
                    preds_waterlevel[b] = []

                if masks_np[i, 0] > 0.5:
                    obs_flow[b].append(ys_np[i, 0])
                    preds_flow[b].append(pf_np[i])

                if masks_np[i, 1] > 0.5:
                    obs_waterlevel[b].append(ys_np[i, 1])
                    preds_waterlevel[b].append(pw_np[i])

    for b in obs_flow.keys():
        obs_flow[b] = np.array(obs_flow[b])
        preds_flow[b] = np.array(preds_flow[b])

    for b in obs_waterlevel.keys():
        obs_waterlevel[b] = np.array(obs_waterlevel[b])
        preds_waterlevel[b] = np.array(preds_waterlevel[b])

    return obs_flow, obs_waterlevel, preds_flow, preds_waterlevel


def load_custom_data(file_path, basins, time_range):
    df = pd.read_csv(file_path, index_col=0, parse_dates=True)
    selected_data = df.loc[time_range[0]:time_range[1], basins]
    return selected_data


def set_random_seed(seed):
    print("随机种子:", seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_waterlevel_basins_from_file(file_path="valid_waterlevel_basins.txt"):
    import ast

    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()

        start_idx = content.find('VALID_WATER_LEVEL_BASINS = [')
        if start_idx == -1:
            raise ValueError(f"未在文件中找到 VALID_WATER_LEVEL_BASINS")

        list_start = content.find('[', start_idx)
        if list_start == -1:
            raise ValueError(f"未找到列表开始标记")

        bracket_count = 0
        list_end = list_start
        for i in range(list_start, len(content)):
            if content[i] == '[':
                bracket_count += 1
            elif content[i] == ']':
                bracket_count -= 1
                if bracket_count == 0:
                    list_end = i + 1
                    break

        if bracket_count != 0:
            raise ValueError(f"列表格式不完整，括号不匹配")

        list_str = content[list_start:list_end]
        basin_list = ast.literal_eval(list_str)

        print(f"从文件 {file_path} 读取了 {len(basin_list)} 个流域")
        return basin_list

    except FileNotFoundError:
        print(f"错误: 文件 {file_path} 不存在")
        raise
    except Exception as e:
        print(f"读取文件 {file_path} 时出错: {e}")
        raise


def filter_basins_with_valid_data(camelsh_reader, basin_list, time_range, max_basins_to_check=None, min_valid_ratio=0.1):
    """验证流域，只保留同时有有效水位和径流数据的流域"""
    print(f"\n正在验证流域的水位和径流数据有效性...")
    print(f"候选流域数量: {len(basin_list)}")

    basins_to_check = basin_list[:max_basins_to_check] if max_basins_to_check else basin_list
    print(f"将检查前 {len(basins_to_check)} 个流域...")

    valid_basins = []
    batch_size = 50

    for i in tqdm(range(0, len(basins_to_check), batch_size), desc="验证流域数据"):
        batch = basins_to_check[i:i+batch_size]

        try:
            waterlevel_ds = camelsh_reader.read_ts_xrdataset(
                gage_id_lst=batch,
                t_range=time_range,
                var_lst=[StandardVariable.WATER_LEVEL]
            )
            flow_ds = camelsh_reader.read_ts_xrdataset(
                gage_id_lst=batch,
                t_range=time_range,
                var_lst=[StandardVariable.STREAMFLOW]
            )

            if StandardVariable.WATER_LEVEL in waterlevel_ds.data_vars:
                waterlevel_df = waterlevel_ds[StandardVariable.WATER_LEVEL].to_pandas().T
            else:
                continue

            if StandardVariable.STREAMFLOW in flow_ds.data_vars:
                flow_df = flow_ds[StandardVariable.STREAMFLOW].to_pandas().T
            else:
                continue

            for basin_id in batch:
                waterlevel_valid = False
                flow_valid = False

                if basin_id in waterlevel_df.columns:
                    wl_data = waterlevel_df[basin_id]
                    if wl_data.notna().any():
                        wl_valid_ratio = wl_data.notna().sum() / len(wl_data)
                        if wl_valid_ratio >= min_valid_ratio:
                            waterlevel_valid = True

                if basin_id in flow_df.columns:
                    flow_data = flow_df[basin_id]
                    if flow_data.notna().any():
                        flow_valid_ratio_val = flow_data.notna().sum() / len(flow_data)
                        if flow_valid_ratio_val >= min_valid_ratio:
                            flow_valid = True

                if waterlevel_valid and flow_valid:
                    valid_basins.append(basin_id)

        except Exception as e:
            print(f"\n警告: 批量加载失败: {e}")

    print(f"\n验证完成: 有效流域 {len(valid_basins)} 个")
    return valid_basins


def run_ablation_experiment(flow_missing_ratio, waterlevel_missing_ratio,
                            flow_data_original, waterlevel_data_original,
                            experiment_name, mask_seed=42, **kwargs):
    """
    运行单个消融实验（单次，指定 mask_seed）

    Parameters
    ----------
    flow_missing_ratio : float
        径流标签缺失比例 (0.0表示完整)
    waterlevel_missing_ratio : float
        水位标签缺失比例 (0.0表示完整)
    flow_data_original : pd.DataFrame
        原始径流数据
    waterlevel_data_original : pd.DataFrame
        原始水位数据
    experiment_name : str
        实验名称
    mask_seed : int
        用于生成随机缺失 mask 的种子；径流用 mask_seed，水位用 mask_seed+1
    **kwargs : dict
        其他参数

    Returns
    -------
    dict
        实验结果
    model
        训练好的模型
    means, stds
        归一化参数
    """
    print(f"\n{'='*80}")
    print(f"实验: {experiment_name}  (mask_seed={mask_seed})")
    print(f"  径流标签缺失: {flow_missing_ratio:.0%}")
    print(f"  水位标签缺失: {waterlevel_missing_ratio:.0%}")
    print(f"{'='*80}")

    # 创建人工缺失数据（局部生成器，不影响全局状态）
    print(f"\n创建标签缺失数据...")
    print("径流数据:")
    flow_data_masked = create_missing_mask(
        flow_data_original, flow_missing_ratio, seed=mask_seed
    )
    print("水位数据:")
    waterlevel_data_masked = create_missing_mask(
        waterlevel_data_original, waterlevel_missing_ratio, seed=mask_seed + 1
    )

    # 创建数据集
    print("\n创建数据集...")
    ds_train = MultiTaskDatasetWithMissingLabels(
        basins=kwargs['chosen_basins'],
        dates=kwargs['default_range'],
        data_attr=kwargs['attrs_df'],
        data_forcing=kwargs['forcings_ds'],
        data_flow=flow_data_masked,
        data_waterlevel=waterlevel_data_masked,
        loader_type="train",
        seq_length=kwargs['sequence_length'],
    )
    tr_loader = DataLoader(ds_train, batch_size=kwargs['batch_size'], shuffle=True)

    means = ds_train.get_means()
    stds = ds_train.get_stds()

    ds_val = MultiTaskDatasetWithMissingLabels(
        basins=kwargs['chosen_basins'],
        dates=kwargs['default_range'],
        data_attr=kwargs['attrs_df'],
        data_forcing=kwargs['forcings_ds'],
        data_flow=flow_data_masked,
        data_waterlevel=waterlevel_data_masked,
        loader_type="valid",
        seq_length=kwargs['sequence_length'],
        means=means,
        stds=stds,
    )
    val_loader = DataLoader(ds_val, batch_size=1000, shuffle=False)

    # 测试集始终使用原始完整数据（无人工缺失）
    ds_test = MultiTaskDatasetWithMissingLabels(
        basins=kwargs['chosen_basins'],
        dates=kwargs['default_range'],
        data_attr=kwargs['attrs_df'],
        data_forcing=kwargs['forcings_ds'],
        data_flow=flow_data_original,
        data_waterlevel=waterlevel_data_original,
        loader_type="test",
        seq_length=kwargs['sequence_length'],
        means=means,
        stds=stds,
    )
    test_loader = DataLoader(ds_test, batch_size=1000, shuffle=False)

    # 创建模型
    print("\n创建模型（HL2Q）...")
    model = MultiTaskLSTM(
        input_size=kwargs['input_size'],
        hidden_size=kwargs['hidden_size'],
        dropout_rate=kwargs['dropout_rate'],
        task_weights=kwargs['task_weights'],
        hl_proj_size=kwargs.get('hl_proj_size', 16),
        stop_gradient=kwargs.get('stop_gradient', False),
    ).to(DEVICE)

    optimizer = torch.optim.Adam(model.parameters(), lr=kwargs['learning_rate'])
    loss_func = nn.MSELoss(reduction='none')

    # 训练（含早停）
    print("\n开始训练...")
    n_epochs = kwargs['n_epochs']
    best_val_nse_avg = -float('inf')
    best_epoch = 0
    patience = 5
    patience_counter = 0
    best_model_state = None

    train_losses = []
    val_nses_flow = []
    val_nses_waterlevel = []

    for i in range(n_epochs):
        train_loss, _, _ = train_epoch(model, optimizer, tr_loader, loss_func, i + 1)
        train_losses.append(train_loss)

        obs_flow_dict, obs_waterlevel_dict, preds_flow_dict, preds_waterlevel_dict = eval_model(
            model, val_loader
        )

        nse_flow_list = []
        nse_waterlevel_list = []

        for basin in kwargs['chosen_basins']:
            b = str(basin)
            if b in preds_flow_dict and len(preds_flow_dict[b]) > 0:
                pf = ds_val.local_denormalization(preds_flow_dict[b], b, variable="flow")
                of = ds_val.local_denormalization(obs_flow_dict[b], b, variable="flow")
                nse_flow_list.append(he.nse(pf, of))

            if b in preds_waterlevel_dict and len(preds_waterlevel_dict[b]) > 0:
                pw = ds_val.local_denormalization(preds_waterlevel_dict[b], b, variable="waterlevel")
                ow = ds_val.local_denormalization(obs_waterlevel_dict[b], b, variable="waterlevel")
                nse_waterlevel_list.append(he.nse(pw, ow))

        current_nse_flow = np.mean(nse_flow_list) if nse_flow_list else 0.0
        current_nse_waterlevel = np.mean(nse_waterlevel_list) if nse_waterlevel_list else 0.0
        current_nse_avg = (current_nse_flow + current_nse_waterlevel) / 2

        val_nses_flow.append(current_nse_flow)
        val_nses_waterlevel.append(current_nse_waterlevel)

        if current_nse_avg > best_val_nse_avg:
            best_val_nse_avg = current_nse_avg
            best_epoch = i + 1
            patience_counter = 0
            best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            print(f"Epoch {i+1} - NSE (径流): {current_nse_flow:.4f}, NSE (水位): {current_nse_waterlevel:.4f}, 平均: {current_nse_avg:.4f} ✓")
        else:
            patience_counter += 1
            print(f"Epoch {i+1} - NSE (径流): {current_nse_flow:.4f}, NSE (水位): {current_nse_waterlevel:.4f}, 平均: {current_nse_avg:.4f} ({patience_counter}/{patience})")

            if patience_counter >= patience:
                print(f"\n早停触发！最佳: Epoch {best_epoch}, NSE = {best_val_nse_avg:.4f}")
                model.load_state_dict({k: v.to(DEVICE) for k, v in best_model_state.items()})
                break

    # 测试
    print("\n在测试集上评估...")
    obs_flow_dict, obs_waterlevel_dict, preds_flow_dict, preds_waterlevel_dict = eval_model(
        model, test_loader
    )

    test_nse_flow_list = []
    test_nse_waterlevel_list = []

    for basin in kwargs['chosen_basins']:
        b = str(basin)
        if b in preds_flow_dict and len(preds_flow_dict[b]) > 10:
            pf = ds_test.local_denormalization(preds_flow_dict[b], b, variable="flow")
            of = ds_test.local_denormalization(obs_flow_dict[b], b, variable="flow")
            test_nse_flow_list.append(he.nse(pf, of))

        if b in preds_waterlevel_dict and len(preds_waterlevel_dict[b]) > 10:
            pw = ds_test.local_denormalization(preds_waterlevel_dict[b], b, variable="waterlevel")
            ow = ds_test.local_denormalization(obs_waterlevel_dict[b], b, variable="waterlevel")
            test_nse_waterlevel_list.append(he.nse(pw, ow))

    avg_nse_flow = np.mean(test_nse_flow_list) if test_nse_flow_list else 0.0
    avg_nse_waterlevel = np.mean(test_nse_waterlevel_list) if test_nse_waterlevel_list else 0.0

    print(f"\n测试集结果:")
    print(f"  平均 NSE (径流): {avg_nse_flow:.4f} (基于 {len(test_nse_flow_list)} 个流域)")
    print(f"  平均 NSE (水位): {avg_nse_waterlevel:.4f} (基于 {len(test_nse_waterlevel_list)} 个流域)")

    results = {
        'flow_missing_ratio': flow_missing_ratio,
        'waterlevel_missing_ratio': waterlevel_missing_ratio,
        'experiment_name': experiment_name,
        'mask_seed': mask_seed,
        'best_epoch': best_epoch,
        'best_val_nse_avg': best_val_nse_avg,
        'test_nse_flow': avg_nse_flow,
        'test_nse_waterlevel': avg_nse_waterlevel,
        'test_nse_flow_list': test_nse_flow_list,
        'test_nse_waterlevel_list': test_nse_waterlevel_list,
        'train_losses': train_losses,
        'val_nses_flow': val_nses_flow,
        'val_nses_waterlevel': val_nses_waterlevel,
        'n_basins_flow': len(test_nse_flow_list),
        'n_basins_waterlevel': len(test_nse_waterlevel_list),
    }

    return results, model, means, stds


def aggregate_repeat_results(seed_results_list):
    """
    将同一场景多次重复的结果聚合为均值 ± 标准差

    Parameters
    ----------
    seed_results_list : list[dict]
        每个 seed 的 results 字典列表

    Returns
    -------
    dict
        聚合后的统计量
    """
    nse_flows = [r['test_nse_flow'] for r in seed_results_list]
    nse_waterlevels = [r['test_nse_waterlevel'] for r in seed_results_list]
    best_epochs = [r['best_epoch'] for r in seed_results_list]

    return {
        'experiment_name': seed_results_list[0]['experiment_name'].rsplit('_seed', 1)[0],
        'flow_missing_ratio': seed_results_list[0]['flow_missing_ratio'],
        'waterlevel_missing_ratio': seed_results_list[0]['waterlevel_missing_ratio'],
        'n_repeats': len(seed_results_list),
        # 径流 NSE
        'test_nse_flow_mean': float(np.mean(nse_flows)),
        'test_nse_flow_std': float(np.std(nse_flows, ddof=1)) if len(nse_flows) > 1 else 0.0,
        'test_nse_flow_min': float(np.min(nse_flows)),
        'test_nse_flow_max': float(np.max(nse_flows)),
        # 水位 NSE
        'test_nse_waterlevel_mean': float(np.mean(nse_waterlevels)),
        'test_nse_waterlevel_std': float(np.std(nse_waterlevels, ddof=1)) if len(nse_waterlevels) > 1 else 0.0,
        'test_nse_waterlevel_min': float(np.min(nse_waterlevels)),
        'test_nse_waterlevel_max': float(np.max(nse_waterlevels)),
        # 其他
        'best_epoch_mean': float(np.mean(best_epochs)),
        'n_basins_flow': seed_results_list[0]['n_basins_flow'],
        'n_basins_waterlevel': seed_results_list[0]['n_basins_waterlevel'],
        # 保存每次原始值方便溯源
        'raw_nse_flows': nse_flows,
        'raw_nse_waterlevels': nse_waterlevels,
    }


if __name__ == "__main__":
    set_random_seed(1234)
    configure_chinese_font()
    print_device_info()

    from config import (
        CAMELSH_DATA_PATH, NUM_BASINS, SEQUENCE_LENGTH, BATCH_SIZE, EPOCHS,
        TRAIN_START, TRAIN_END, VALID_START, VALID_END, TEST_START, TEST_END,
        FORCING_VARIABLES, ATTRIBUTE_VARIABLES,
        IMAGES_SAVE_PATH, REPORTS_SAVE_PATH, MODEL_SAVE_PATH
    )

    os.makedirs(IMAGES_SAVE_PATH, exist_ok=True)
    os.makedirs(REPORTS_SAVE_PATH, exist_ok=True)
    os.makedirs(MODEL_SAVE_PATH, exist_ok=True)

    print(f"\n重复实验配置: 每个场景使用 {len(REPEAT_SEEDS)} 个随机种子: {REPEAT_SEEDS}")

    # 加载流域列表
    print("\n正在从文件读取流域列表...")
    VALID_WATER_LEVEL_BASINS = load_waterlevel_basins_from_file("valid_waterlevel_basins.txt")

    # 加载CAMELSH数据
    print("\n正在加载CAMELSH数据...")
    camelsh_reader = ImprovedCAMELSHReader(CAMELSH_DATA_PATH, download=False, use_batch=True)

    summary = camelsh_reader.get_data_summary()
    print(f"数据概要: 总流域数量 {summary['total_basins']}")

    basin_ids = camelsh_reader.read_object_ids()
    camelsh = camelsh_reader.camelsh
    default_range = camelsh.default_t_range
    print(f"数据集时间范围: {default_range}")

    max_candidates = min(len(VALID_WATER_LEVEL_BASINS), max(NUM_BASINS * 3, 200))
    validated_basins = filter_basins_with_valid_data(
        camelsh_reader=camelsh_reader,
        basin_list=VALID_WATER_LEVEL_BASINS,
        time_range=default_range,
        max_basins_to_check=max_candidates,
        min_valid_ratio=0.1
    )

    if len(validated_basins) == 0:
        raise ValueError("未找到有效流域！")

    chosen_basins = validated_basins[:NUM_BASINS]
    print(f"\n最终选择的流域 ({len(chosen_basins)} 个): {chosen_basins}")

    # 加载特征变量
    chosen_forcing_vars = []
    for var_name in FORCING_VARIABLES:
        if hasattr(StandardVariable, var_name.upper()):
            chosen_forcing_vars.append(getattr(StandardVariable, var_name.upper()))
        else:
            chosen_forcing_vars.append(var_name)

    chosen_attrs_vars = ATTRIBUTE_VARIABLES

    # 加载属性数据
    print("\n正在加载属性数据...")
    attrs = camelsh.read_attr_xrdataset(
        gage_id_lst=chosen_basins,
        var_lst=chosen_attrs_vars
    )

    # 加载气象数据
    print("\n正在加载气象强迫数据...")
    chosen_forcing_vars_no_precip = [v for v in chosen_forcing_vars
                                      if v != StandardVariable.PRECIPITATION]

    if chosen_forcing_vars_no_precip:
        forcings_ds_no_precip = camelsh_reader.read_ts_xrdataset(
            gage_id_lst=chosen_basins,
            t_range=default_range,
            var_lst=chosen_forcing_vars_no_precip
        )
    else:
        forcings_ds_no_precip = None

    print("\n从MSWEP加载降雨数据...")
    mswep_precip_df = load_mswep_data(
        file_path="MSWEP/mswep_1000basins_mean_3hourly_1980_2024.csv",
        basin_ids=chosen_basins,
        time_range=default_range
    )

    if forcings_ds_no_precip is not None:
        forcings_ds = merge_forcing_with_mswep(forcings_ds_no_precip, mswep_precip_df)
    else:
        from mswep_loader import convert_mswep_to_xarray
        forcings_ds = convert_mswep_to_xarray(mswep_precip_df, var_name='precipitation')

    # 加载径流和水位原始完整数据
    print("\n正在加载径流和水位数据...")
    flow_ds = camelsh_reader.read_ts_xrdataset(
        gage_id_lst=chosen_basins,
        t_range=default_range,
        var_lst=[StandardVariable.STREAMFLOW]
    )

    waterlevel_ds = camelsh_reader.read_ts_xrdataset(
        gage_id_lst=chosen_basins,
        t_range=default_range,
        var_lst=[StandardVariable.WATER_LEVEL]
    )

    full_flow_original = flow_ds[StandardVariable.STREAMFLOW].to_pandas().T
    full_waterlevel_original = waterlevel_ds[StandardVariable.WATER_LEVEL].to_pandas().T

    print(f"径流数据形状: {full_flow_original.shape}")
    print(f"水位数据形状: {full_waterlevel_original.shape}")

    attrs_df = attrs.to_pandas()
    attrs_df.index.name = 'gauge_id'
    attrs_df = attrs_df.reset_index()

    # 实验配置
    experiment_kwargs = {
        'chosen_basins': chosen_basins,
        'default_range': default_range,
        'attrs_df': attrs_df,
        'forcings_ds': forcings_ds,
        'sequence_length': SEQUENCE_LENGTH,
        'batch_size': BATCH_SIZE,
        'input_size': len(chosen_attrs_vars) + len(chosen_forcing_vars),
        'hidden_size': 64,
        'dropout_rate': 0.2,
        'learning_rate': 1e-3,
        'task_weights': {'flow': 1.0, 'waterlevel': 1.0},
        'n_epochs': EPOCHS,
        'hl_proj_size': 16,
        'stop_gradient': False,
    }

    # 消融实验场景定义（径流缺失比例, 水位缺失比例, 实验名称）
    experiments = [
        (0.0, 0.0, "baseline_both_complete"),
        (0.1, 0.0, "flow_missing_10pct_wl_complete"),
        (0.3, 0.0, "flow_missing_30pct_wl_complete"),
        (0.5, 0.0, "flow_missing_50pct_wl_complete"),
        (0.0, 0.1, "flow_complete_wl_missing_10pct"),
        (0.0, 0.3, "flow_complete_wl_missing_30pct"),
        (0.0, 0.5, "flow_complete_wl_missing_50pct"),
        (0.3, 0.3, "both_missing_30pct"),
        (0.5, 0.5, "both_missing_50pct"),
    ]

    total_runs = len(experiments) * len(REPEAT_SEEDS)
    print(f"\n将运行 {len(experiments)} 个场景 × {len(REPEAT_SEEDS)} 个种子 = {total_runs} 次实验")
    for i, (flow_miss, wl_miss, name) in enumerate(experiments, 1):
        print(f"  {i}. {name}: 径流缺失{flow_miss:.0%}, 水位缺失{wl_miss:.0%}")

    # ── 主实验循环 ────────────────────────────────────────────────────────────────
    all_raw_results = []       # 每次单次实验的原始结果
    all_aggregated_results = []  # 每个场景聚合后的结果

    for exp_idx, (flow_missing_ratio, waterlevel_missing_ratio, exp_name) in enumerate(experiments):
        print(f"\n\n{'#'*80}")
        print(f"场景 {exp_idx+1}/{len(experiments)}: {exp_name}")
        print(f"{'#'*80}")

        seed_results = []

        for seed_idx, mask_seed in enumerate(REPEAT_SEEDS):
            run_name = f"{exp_name}_seed{mask_seed}"
            print(f"\n--- 重复 {seed_idx+1}/{len(REPEAT_SEEDS)}，mask_seed={mask_seed} ---")

            results, model, means, stds = run_ablation_experiment(
                flow_missing_ratio=flow_missing_ratio,
                waterlevel_missing_ratio=waterlevel_missing_ratio,
                flow_data_original=full_flow_original,
                waterlevel_data_original=full_waterlevel_original,
                experiment_name=run_name,
                mask_seed=mask_seed,
                **experiment_kwargs
            )

            seed_results.append(results)
            all_raw_results.append(results)

            # 保存单次模型
            model_path = os.path.join(MODEL_SAVE_PATH, f'hl2q_ablation_{run_name}.pth')
            torch.save({
                'model_state_dict': model.state_dict(),
                'means': means,
                'stds': stds,
                'results': results,
            }, model_path)
            print(f"模型已保存: {model_path}")

        # 聚合该场景的多次结果
        agg = aggregate_repeat_results(seed_results)
        all_aggregated_results.append(agg)

        print(f"\n场景 [{exp_name}] 聚合结果 ({len(REPEAT_SEEDS)} 次重复):")
        print(f"  NSE (径流): {agg['test_nse_flow_mean']:.4f} ± {agg['test_nse_flow_std']:.4f}")
        print(f"  NSE (水位): {agg['test_nse_waterlevel_mean']:.4f} ± {agg['test_nse_waterlevel_std']:.4f}")

    # ── 汇总打印 ────────────────────────────────────────────────────────────────
    print("\n" + "="*90)
    print("消融实验汇总结果（均值 ± 标准差）")
    print("="*90)
    header = f"{'实验名称':<35} {'径流缺失':<8} {'水位缺失':<8} {'NSE径流(mean±std)':<22} {'NSE水位(mean±std)':<22}"
    print(header)
    print("-"*90)
    for agg in all_aggregated_results:
        print(
            f"{agg['experiment_name']:<35} "
            f"{agg['flow_missing_ratio']:<8.0%} "
            f"{agg['waterlevel_missing_ratio']:<8.0%} "
            f"{agg['test_nse_flow_mean']:.4f}±{agg['test_nse_flow_std']:.4f}        "
            f"{agg['test_nse_waterlevel_mean']:.4f}±{agg['test_nse_waterlevel_std']:.4f}"
        )

    # ── 保存 CSV ────────────────────────────────────────────────────────────────
    # 1. 原始每次结果
    raw_df = pd.DataFrame([{
        '实验名称': r['experiment_name'],
        '径流缺失比例': r['flow_missing_ratio'],
        '水位缺失比例': r['waterlevel_missing_ratio'],
        'mask_seed': r['mask_seed'],
        '最佳Epoch': r['best_epoch'],
        '测试NSE_径流': r['test_nse_flow'],
        '测试NSE_水位': r['test_nse_waterlevel'],
        '流域数_径流': r['n_basins_flow'],
        '流域数_水位': r['n_basins_waterlevel'],
    } for r in all_raw_results])

    raw_csv = os.path.join(REPORTS_SAVE_PATH, "hl2q_ablation_repeat_raw_results.csv")
    raw_df.to_csv(raw_csv, index=False, encoding='utf-8-sig')
    print(f"\n已保存原始结果CSV: {raw_csv}")

    # 2. 聚合结果
    agg_df = pd.DataFrame([{
        '实验名称': a['experiment_name'],
        '径流缺失比例': a['flow_missing_ratio'],
        '水位缺失比例': a['waterlevel_missing_ratio'],
        '重复次数': a['n_repeats'],
        'NSE径流_均值': a['test_nse_flow_mean'],
        'NSE径流_标准差': a['test_nse_flow_std'],
        'NSE径流_最小': a['test_nse_flow_min'],
        'NSE径流_最大': a['test_nse_flow_max'],
        'NSE水位_均值': a['test_nse_waterlevel_mean'],
        'NSE水位_标准差': a['test_nse_waterlevel_std'],
        'NSE水位_最小': a['test_nse_waterlevel_min'],
        'NSE水位_最大': a['test_nse_waterlevel_max'],
        '平均最佳Epoch': a['best_epoch_mean'],
        '流域数_径流': a['n_basins_flow'],
        '流域数_水位': a['n_basins_waterlevel'],
    } for a in all_aggregated_results])

    agg_csv = os.path.join(REPORTS_SAVE_PATH, "hl2q_ablation_repeat_aggregated_results.csv")
    agg_df.to_csv(agg_csv, index=False, encoding='utf-8-sig')
    print(f"已保存聚合结果CSV: {agg_csv}")

    # ── 可视化（带误差棒）────────────────────────────────────────────────────────
    print("\n正在生成对比图（含误差棒）...")

    baseline_agg = next(a for a in all_aggregated_results if a['experiment_name'] == 'baseline_both_complete')
    flow_missing_agg = [a for a in all_aggregated_results
                        if a['waterlevel_missing_ratio'] == 0 and a['flow_missing_ratio'] > 0]
    wl_missing_agg = [a for a in all_aggregated_results
                      if a['flow_missing_ratio'] == 0 and a['waterlevel_missing_ratio'] > 0]

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    # 图1：径流任务 NSE vs 缺失比例（含误差棒）
    ax1 = axes[0, 0]
    if flow_missing_agg:
        x = [a['flow_missing_ratio'] * 100 for a in flow_missing_agg]
        y = [a['test_nse_flow_mean'] for a in flow_missing_agg]
        yerr = [a['test_nse_flow_std'] for a in flow_missing_agg]
        ax1.errorbar(x, y, yerr=yerr, fmt='o-', linewidth=2, markersize=8,
                     capsize=5, capthick=1.5, label='径流缺失+水位完整', color='tab:blue')
    if wl_missing_agg:
        x = [a['waterlevel_missing_ratio'] * 100 for a in wl_missing_agg]
        y = [a['test_nse_flow_mean'] for a in wl_missing_agg]
        yerr = [a['test_nse_flow_std'] for a in wl_missing_agg]
        ax1.errorbar(x, y, yerr=yerr, fmt='s-', linewidth=2, markersize=8,
                     capsize=5, capthick=1.5, label='径流完整+水位缺失', color='tab:orange')
    ax1.axhline(y=baseline_agg['test_nse_flow_mean'], color='k', linestyle='--',
                alpha=0.6, label=f"基线 {baseline_agg['test_nse_flow_mean']:.3f}±{baseline_agg['test_nse_flow_std']:.3f}")
    ax1.fill_between(
        [0, 55],
        baseline_agg['test_nse_flow_mean'] - baseline_agg['test_nse_flow_std'],
        baseline_agg['test_nse_flow_mean'] + baseline_agg['test_nse_flow_std'],
        alpha=0.1, color='k'
    )
    ax1.set_xlabel('缺失比例 (%)', fontsize=11)
    ax1.set_ylabel('径流预测 NSE', fontsize=11)
    ax1.set_title(f'径流任务性能 vs 标签缺失（{len(REPEAT_SEEDS)}次重复）', fontsize=13)
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)

    # 图2：水位任务 NSE vs 缺失比例（含误差棒）
    ax2 = axes[0, 1]
    if flow_missing_agg:
        x = [a['flow_missing_ratio'] * 100 for a in flow_missing_agg]
        y = [a['test_nse_waterlevel_mean'] for a in flow_missing_agg]
        yerr = [a['test_nse_waterlevel_std'] for a in flow_missing_agg]
        ax2.errorbar(x, y, yerr=yerr, fmt='o-', linewidth=2, markersize=8,
                     capsize=5, capthick=1.5, label='径流缺失+水位完整', color='tab:blue')
    if wl_missing_agg:
        x = [a['waterlevel_missing_ratio'] * 100 for a in wl_missing_agg]
        y = [a['test_nse_waterlevel_mean'] for a in wl_missing_agg]
        yerr = [a['test_nse_waterlevel_std'] for a in wl_missing_agg]
        ax2.errorbar(x, y, yerr=yerr, fmt='s-', linewidth=2, markersize=8,
                     capsize=5, capthick=1.5, label='径流完整+水位缺失', color='tab:orange')
    ax2.axhline(y=baseline_agg['test_nse_waterlevel_mean'], color='k', linestyle='--',
                alpha=0.6, label=f"基线 {baseline_agg['test_nse_waterlevel_mean']:.3f}±{baseline_agg['test_nse_waterlevel_std']:.3f}")
    ax2.fill_between(
        [0, 55],
        baseline_agg['test_nse_waterlevel_mean'] - baseline_agg['test_nse_waterlevel_std'],
        baseline_agg['test_nse_waterlevel_mean'] + baseline_agg['test_nse_waterlevel_std'],
        alpha=0.1, color='k'
    )
    ax2.set_xlabel('缺失比例 (%)', fontsize=11)
    ax2.set_ylabel('水位预测 NSE', fontsize=11)
    ax2.set_title(f'水位任务性能 vs 标签缺失（{len(REPEAT_SEEDS)}次重复）', fontsize=13)
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)

    # 图3：跨任务影响——水位缺失对径流的影响（NSE下降%）
    ax3 = axes[1, 0]
    if wl_missing_agg:
        x = [a['waterlevel_missing_ratio'] * 100 for a in wl_missing_agg]
        baseline_flow = baseline_agg['test_nse_flow_mean']
        y_drop = [(baseline_flow - a['test_nse_flow_mean']) / max(abs(baseline_flow), 1e-6) * 100
                  for a in wl_missing_agg]
        # 误差传播：标准差的近似
        y_drop_err = [a['test_nse_flow_std'] / max(abs(baseline_flow), 1e-6) * 100
                      for a in wl_missing_agg]
        bars = ax3.bar(x, y_drop, color='tab:orange', alpha=0.7, width=5)
        ax3.errorbar(x, y_drop, yerr=y_drop_err, fmt='none', color='black',
                     capsize=4, capthick=1.5)
        ax3.set_xlabel('水位标签缺失比例 (%)', fontsize=11)
        ax3.set_ylabel('径流NSE相对下降 (%)', fontsize=11)
        ax3.set_title('跨任务影响：水位缺失 → 径流任务', fontsize=13)
        ax3.grid(True, alpha=0.3, axis='y')
        ax3.axhline(y=0, color='k', linestyle='-', linewidth=0.8)

    # 图4：跨任务影响——径流缺失对水位的影响（NSE下降%）
    ax4 = axes[1, 1]
    if flow_missing_agg:
        x = [a['flow_missing_ratio'] * 100 for a in flow_missing_agg]
        baseline_wl = baseline_agg['test_nse_waterlevel_mean']
        y_drop = [(baseline_wl - a['test_nse_waterlevel_mean']) / max(abs(baseline_wl), 1e-6) * 100
                  for a in flow_missing_agg]
        y_drop_err = [a['test_nse_waterlevel_std'] / max(abs(baseline_wl), 1e-6) * 100
                      for a in flow_missing_agg]
        ax4.bar(x, y_drop, color='tab:blue', alpha=0.7, width=5)
        ax4.errorbar(x, y_drop, yerr=y_drop_err, fmt='none', color='black',
                     capsize=4, capthick=1.5)
        ax4.set_xlabel('径流标签缺失比例 (%)', fontsize=11)
        ax4.set_ylabel('水位NSE相对下降 (%)', fontsize=11)
        ax4.set_title('跨任务影响：径流缺失 → 水位任务', fontsize=13)
        ax4.grid(True, alpha=0.3, axis='y')
        ax4.axhline(y=0, color='k', linestyle='-', linewidth=0.8)

    plt.tight_layout()
    comparison_file = os.path.join(
        IMAGES_SAVE_PATH, "hl2q_ablation_repeat_comparison.png"
    )
    plt.savefig(comparison_file, dpi=300, bbox_inches='tight')
    print(f"已保存对比图: {comparison_file}")

    print("\n消融实验（多次重复）完成！")
