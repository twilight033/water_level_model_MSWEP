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
)  # check if GPU is available

# 时间划分比例（经验值）：按每个流域自身完整时间序列划分
TRAIN_RATIO = 0.6
VALID_RATIO = 0.2
TEST_RATIO = 0.2

# 滑窗步长（经验值）：序列起点之间的时间步间隔，减小步长会增加样本数和内存占用
WINDOW_STEP = 3  # 对小时数据，相当于每隔 24 小时（1 天）取一个样本起点
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
        # 确保basins列表中的元素都是字符串类型，以保持一致性
        self.basins = [str(b) for b in basins]
        self.dates = dates

        self.seq_length = seq_length

        # 初始化均值和标准差（如果是训练模式，会在_load_data中计算；否则从参数传入）
        self.means = means if means is not None else {}
        self.stds = stds if stds is not None else {}

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
        # time_idx 是真实 datetime，需要找到整数位置
        # 注意：强迫数据和目标数据可能有不同的时间索引（3小时 vs 小时）
        if hasattr(self, 'forcing_time_index') and self.forcing_time_index is not None:
            # 使用强迫数据的时间索引
            if time_idx not in self.forcing_time_index:
                # 如果时间索引不存在，找最近的
                closest_idx = self.forcing_time_index.get_indexer([time_idx], method='nearest')[0]
                start_pos = closest_idx
            else:
                start_pos = self.forcing_time_index.get_loc(time_idx)
        else:
            # 降级方案：使用目标数据的时间索引
            start_pos = self.data_flow.index.get_loc(time_idx)
        
        x = self.x[basin][start_pos : start_pos + seq_length]
        
        # 检查强迫数据维度
        if len(x) == 0:
            print(f"[错误] 样本 {item}: 强迫数据x为空")
            print(f"  流域: {basin}, 时间索引: {time_idx}")
            print(f"  start_pos: {start_pos}, seq_length: {seq_length}")
            print(f"  强迫数据总长度: {len(self.x[basin])}")
            raise ValueError("强迫数据x为空")
        
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
        # 注意：目标数据使用原始时间索引，需要从lookup_table中获取的time_idx开始计算
        target_time_idx = time_idx
        if target_time_idx in self.data_flow.index:
            # 目标值对应的是序列结束时刻
            # 由于强迫数据是3小时分辨率，需要找到对应的目标数据位置
            end_time = target_time_idx + pd.Timedelta(hours=(seq_length - 1) * 3)
            if end_time in self.data_flow.index:
                target_end_pos = self.data_flow.index.get_loc(end_time)
                y_flow = self.y_flow[basin][target_end_pos]
                y_waterlevel = self.y_waterlevel[basin][target_end_pos]
            else:
                # 如果精确时间不存在，找最近的
                nearest_idx = self.data_flow.index.get_indexer([end_time], method='nearest')[0]
                y_flow = self.y_flow[basin][nearest_idx]
                y_waterlevel = self.y_waterlevel[basin][nearest_idx]
        else:
            # 如果时间索引不存在，找最近的
            nearest_idx = self.data_flow.index.get_indexer([target_time_idx], method='nearest')[0]
            end_time = target_time_idx + pd.Timedelta(hours=(seq_length - 1) * 3)
            nearest_end_idx = self.data_flow.index.get_indexer([end_time], method='nearest')[0]
            y_flow = self.y_flow[basin][nearest_end_idx]
            y_waterlevel = self.y_waterlevel[basin][nearest_end_idx]
        
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
            print(f"  目标时间: {end_time}")
            print(f"  y_flow: {y_flow}, y_waterlevel: {y_waterlevel}")
            raise ValueError("目标值包含NaN")
        
        return torch.from_numpy(xc).float(), torch.from_numpy(y).float(), basin

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
            
            # ========== 修复：不要对径流/水位数据做全局填充！==========
            # 只检查NaN数量，不做任何填充！
            # 后续在 _create_lookup_table 中会通过 dropna() 只使用有真实数据的时间点
            flow_nan_count = self.data_flow.isnull().sum().sum()
            flow_total = self.data_flow.size
            flow_valid_ratio = (flow_total - flow_nan_count) / flow_total if flow_total > 0 else 0
            print(f"[信息] 径流数据统计:")
            print(f"  - 总数据点: {flow_total}")
            print(f"  - 有效数据点: {flow_total - flow_nan_count} ({flow_valid_ratio:.1%})")
            print(f"  - NaN数据点: {flow_nan_count} ({flow_nan_count/flow_total:.1%})")
            
            wl_nan_count = self.data_waterlevel.isnull().sum().sum()
            wl_total = self.data_waterlevel.size
            wl_valid_ratio = (wl_total - wl_nan_count) / wl_total if wl_total > 0 else 0
            print(f"[信息] 水位数据统计:")
            print(f"  - 总数据点: {wl_total}")
            print(f"  - 有效数据点: {wl_total - wl_nan_count} ({wl_valid_ratio:.1%})")
            print(f"  - NaN数据点: {wl_nan_count} ({wl_nan_count/wl_total:.1%})")
            print(f"  注意：NaN数据将被自动跳过，只使用真实观测数据进行训练")
            
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
            
            # 径流数据的均值和标准差（每个流域独立计算）
            flow_mean = {}
            flow_std = {}
            for basin in self.basins:
                # self.basins已经是字符串列表了
                series = self.data_flow[basin]
                flow_mean[basin] = series.mean()
                flow_std[basin] = series.std() if series.std() > 1e-6 else 1.0
            
            self.means['flow'] = flow_mean
            self.stds['flow'] = flow_std
            
            # 水位数据的均值和标准差（每个流域独立计算）
            waterlevel_mean = {}
            waterlevel_std = {}
            for basin in self.basins:
                # self.basins已经是字符串列表了
                series = self.data_waterlevel[basin]
                waterlevel_mean[basin] = series.mean()
                waterlevel_std[basin] = series.std() if series.std() > 1e-6 else 1.0

            self.means['waterlevel'] = waterlevel_mean
            self.stds['waterlevel'] = waterlevel_std

            
            # 最终NaN检查（只检查强迫数据和属性数据，径流/水位数据允许有NaN）
            final_forcing_nan = int(self.data_forcing.isnull().to_array().sum().values)
            final_attr_nan = self.data_attr[numeric_cols].isnull().sum().sum()
            
            if final_forcing_nan > 0 or final_attr_nan > 0:
                print(f"[错误] 强迫/属性数据清洗后仍有NaN: 强迫={final_forcing_nan}, 属性={final_attr_nan}")
                raise ValueError("强迫/属性数据清洗失败，仍存在NaN值")
            else:
                print("数据统计量计算完成")
        else:
            train_mode = False

        # 归一化处理
        print("开始数据归一化...")
        
        # 保存强迫数据的时间索引，用于后续查找
        self.forcing_time_index = pd.DatetimeIndex(self.data_forcing.time.values)
        
        self.x = self._normalize_forcing(self.data_forcing)
        
        # 更新basins列表为实际归一化成功的流域
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
        """将DataFrame转换为字典格式，保持NaN不变"""
        result = {}
        for basin in self.basins:
            values = df[basin].values
            # 保持NaN不变，不做任何填充
            # NaN时间点会在 _create_lookup_table 中被自动跳过
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
        
        # 获取强迫数据中实际可用的流域列表
        available_basins_in_data = [str(b) for b in data_forcing.basin.values]
        
        for basin in self.basins:
            basin_str = str(basin)
            
            # 检查basin是否在数据中
            if basin_str not in available_basins_in_data:
                print(f"[警告] 流域 {basin_str} 不在强迫数据中，跳过")
                continue
            
            try:
                basin_data = data_forcing.sel(basin=basin_str).to_array().to_numpy().T
            except KeyError:
                # 如果sel失败，尝试使用原始basin值
                try:
                    basin_data = data_forcing.sel(basin=basin).to_array().to_numpy().T
                except KeyError:
                    print(f"[警告] 流域 {basin_str} 无法从强迫数据中提取，跳过")
                    continue
            
            # 检查原始数据是否包含NaN
            if np.isnan(basin_data).any():
                nan_count = np.isnan(basin_data).sum()
                total_count = basin_data.size
                nan_ratio = nan_count / total_count * 100
                print(f"[警告] 流域 {basin_str} 强迫数据包含NaN (数量: {nan_count}/{total_count}, {nan_ratio:.1f}%)，跳过该流域")
                continue
            
            normalized = (basin_data - self.means['forcing'].values) / std_values
            
            # 检查归一化后的数据是否包含NaN
            if np.isnan(normalized).any():
                print(f"[警告] 流域 {basin_str} 归一化后强迫数据包含NaN，跳过该流域")
                continue
            
            result[basin_str] = normalized
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
        """归一化径流数据（按流域独立归一化）- 保持NaN不变"""
        result = {}
        for basin in self.basins:
            # self.basins已经是字符串列表了
            values = data_flow[basin].values
            
            # 获取该流域的均值和标准差
            if basin not in self.means.get('flow', {}):
                raise KeyError(f"流域 {basin} 的径流归一化参数不存在")
            mean = self.means['flow'][basin]
            std = self.stds['flow'][basin]
            
            # 直接归一化，保持NaN不变
            normalized = (values - mean) / std
            
            # NaN是正常的，因为不是所有时间点都有观测
            # 这些NaN时间点会在 _create_lookup_table 中被 dropna() 自动跳过
            
            result[basin] = normalized
        return result

    def _normalize_waterlevel(self, data_waterlevel):
        """归一化水位数据（按流域独立归一化）- 保持NaN不变"""
        result = {}
        for basin in self.basins:
            # self.basins已经是字符串列表了
            values = data_waterlevel[basin].values
            
            # 获取该流域的均值和标准差
            if basin not in self.means.get('waterlevel', {}):
                raise KeyError(f"流域 {basin} 的水位归一化参数不存在")
            mean = self.means['waterlevel'][basin]
            std = self.stds['waterlevel'][basin]
            
            # 直接归一化，保持NaN不变
            normalized = (values - mean) / std
            
            # NaN是正常的，因为不是所有时间点都有观测
            # 这些NaN时间点会在 _create_lookup_table 中被 dropna() 自动跳过
            
            result[basin] = normalized
        return result

    def _create_lookup_table(self):
        """
        为每个流域独立构建滑窗索引，并按该流域自身完整时间范围做比例切分。

        说明：
        - 对每个流域，从归一化后的 flow/waterlevel 数据中找出非NaN时间索引
        - 按 (TRAIN_RATIO, VALID_RATIO, TEST_RATIO) 进行时间顺序切分
        - 在滑窗时，确保每个窗口内的数据是真正连续的（flow和waterlevel都非NaN）
        - **关键**：使用强迫数据的时间索引（3小时分辨率）作为基准
        """
        lookup = []
        seq_length = self.seq_length
        skipped_basins = []
        basins_with_samples = []
        basin_time_ranges = {}  # 记录每个流域的时间范围
        
        # 使用强迫数据的时间索引作为基准（3小时分辨率）
        forcing_times = self.forcing_time_index

        for basin in tqdm(self.basins, desc=f"创建 {self.loader_type} 索引表", disable=False):
            if basin not in self.y_flow or basin not in self.y_waterlevel:
                skipped_basins.append(basin)
                continue
            
            # 找出flow和waterlevel都非NaN的位置
            flow_values = self.y_flow[basin]
            wl_values = self.y_waterlevel[basin]
            target_index = self.data_flow.index
            
            flow_valid_mask = ~np.isnan(flow_values)
            wl_valid_mask = ~np.isnan(wl_values)
            both_valid = flow_valid_mask & wl_valid_mask
            
            # 获取目标数据中所有有效的时间索引
            valid_target_times = set(target_index[both_valid])
            
            # 找出强迫数据时间点中，对应的目标数据也有效的时间点
            valid_forcing_times = []
            for ft in forcing_times:
                # 计算序列结束时刻
                end_time = ft + pd.Timedelta(hours=(seq_length - 1) * 3)
                # 检查结束时刻的目标数据是否有效
                if end_time in valid_target_times:
                    valid_forcing_times.append(ft)

            if len(valid_forcing_times) < 1:
                skipped_basins.append(basin)
                continue

            # 找出有效时间的起止
            first_valid_time = valid_forcing_times[0]
            last_valid_time = valid_forcing_times[-1]
            
            # 在强迫数据时间索引中找到位置
            start_idx = forcing_times.get_loc(first_valid_time)
            end_idx = forcing_times.get_loc(last_valid_time)
            
            # 计算时间跨度
            total_span = end_idx - start_idx + 1
            
            # 按比例划分
            train_end_idx = start_idx + int(total_span * TRAIN_RATIO)
            valid_end_idx = start_idx + int(total_span * (TRAIN_RATIO + VALID_RATIO))
            
            # 根据loader_type确定范围
            if self.loader_type == "train":
                range_start_idx = start_idx
                range_end_idx = train_end_idx
            elif self.loader_type == "valid":
                range_start_idx = train_end_idx
                range_end_idx = valid_end_idx
            else:  # "test"
                range_start_idx = valid_end_idx
                range_end_idx = end_idx + 1
            
            # 检查范围内是否有足够样本
            if range_end_idx - range_start_idx < seq_length:
                continue
            
            # 在该范围内滑窗
            num_samples_this_basin = 0
            for idx in range(range_start_idx, range_end_idx - seq_length + 1, WINDOW_STEP):
                # 获取这个窗口的起始时间
                window_start_time = forcing_times[idx]
                # 计算结束时间
                window_end_time = window_start_time + pd.Timedelta(hours=(seq_length - 1) * 3)
                
                # 检查结束时间的目标数据是否有效
                if window_end_time in valid_target_times:
                    lookup.append((basin, window_start_time))
                    num_samples_this_basin += 1
            
            if num_samples_this_basin > 0:
                basins_with_samples.append(basin)
                # 记录该流域的时间范围信息
                basin_time_ranges[basin] = {
                    'first_valid': str(first_valid_time),
                    'last_valid': str(last_valid_time),
                    'total_valid_hours': len(valid_forcing_times),
                    'loader_start': str(forcing_times[range_start_idx]),
                    'loader_end': str(forcing_times[range_end_idx - 1]) if range_end_idx > 0 else str(forcing_times[-1]),
                    'num_samples': num_samples_this_basin
                }

        self.lookup_table = {i: elem for i, elem in enumerate(lookup)}
        self.num_samples = len(self.lookup_table)
        print(f"\n{self.loader_type.upper()} 数据集统计:")
        print(f"  - 总样本数: {self.num_samples}")
        print(f"  - 有样本的流域数: {len(basins_with_samples)}/{len(self.basins)}")
        if skipped_basins:
            print(f"  - 跳过的流域（不在数据中）: {len(skipped_basins)} 个")
            if len(skipped_basins) <= 10:
                print(f"    示例: {skipped_basins}")
        
        # 打印前几个流域的时间范围信息（验证每个流域有自己的时间区间）
        if basin_time_ranges and self.loader_type == "train":  # 只在训练集时打印，避免重复
            print(f"\n  各流域时间范围示例 (前5个):")
            for i, (basin_id, info) in enumerate(list(basin_time_ranges.items())[:5]):
                print(f"    流域 {basin_id}:")
                print(f"      完整数据范围: {info['first_valid']} 到 {info['last_valid']} (共{info['total_valid_hours']}个3小时步)")
                print(f"      {self.loader_type.upper()}期范围: {info['loader_start']} 到 {info['loader_end']}")
                print(f"      生成样本数: {info['num_samples']}")
        
        if self.num_samples == 0:
            raise ValueError(f"{self.loader_type} 数据集没有生成任何样本！请检查数据有效性。")


    def get_means(self):
        return self.means

    def get_stds(self):
        return self.stds

    def local_denormalization(self, feature, basin, variable="flow"):
        """按流域反归一化"""
        # 确保basin是字符串类型
        basin = str(basin)
        
        if variable == "flow":
            if basin not in self.means.get('flow', {}):
                raise KeyError(f"流域 {basin} 的径流归一化参数不存在")
            mean = self.means['flow'][basin]
            std = self.stds['flow'][basin]
            return feature * std + mean
        elif variable == "waterlevel":
            if basin not in self.means.get('waterlevel', {}):
                raise KeyError(f"流域 {basin} 的水位归一化参数不存在")
            mean = self.means['waterlevel'][basin]
            std = self.stds['waterlevel'][basin]
            return feature * std + mean
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
    
    for batch_idx, (xs, ys,basins) in enumerate(pbar):
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
    """评估模型：按流域分组返回观测和预测结果（不再返回一条长向量）"""
    model.eval()
    
    # 按流域存放结果：key = basin（流域ID），value = numpy数组
    obs_flow = {}
    obs_waterlevel = {}
    preds_flow = {}
    preds_waterlevel = {}
    
    with torch.no_grad():
        for xs, ys, basins in loader:
            xs = xs.to(DEVICE)
            pred_flow, pred_waterlevel = model(xs)
            
            # 统一搬到 CPU + numpy，方便后面处理
            ys_np = ys.numpy()  # [batch, 2]
            pf_np = pred_flow.cpu().numpy().squeeze()      # [batch]
            pw_np = pred_waterlevel.cpu().numpy().squeeze()  # [batch]
            
            # 有可能 squeeze 后变成标量，统一处理成一维
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
                
                obs_flow[b].append(ys_np[i, 0])
                obs_waterlevel[b].append(ys_np[i, 1])
                preds_flow[b].append(pf_np[i])
                preds_waterlevel[b].append(pw_np[i])
    
    # 列表转成 numpy 数组
        # --- 必须排序：否则 NSE 错！！ ---
    sorted_obs_flow = {}
    sorted_obs_waterlevel = {}
    sorted_preds_flow = {}
    sorted_preds_waterlevel = {}

    for b in obs_flow.keys():
        # 收集时间位置
        t = np.array([i for i in range(len(obs_flow[b]))])
        order = np.argsort(t)

        sorted_obs_flow[b] = np.array(obs_flow[b])[order]
        sorted_obs_waterlevel[b] = np.array(obs_waterlevel[b])[order]
        sorted_preds_flow[b] = np.array(preds_flow[b])[order]
        sorted_preds_waterlevel[b] = np.array(preds_waterlevel[b])[order]

    return sorted_obs_flow, sorted_obs_waterlevel, sorted_preds_flow, sorted_preds_waterlevel




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


def load_waterlevel_basins_from_file(file_path="valid_waterlevel_basins.txt"):
    """
    从文件中读取有水位数据的流域列表
    
    Parameters
    ----------
    file_path : str
        文件路径，默认为 "valid_waterlevel_basins.txt"
    
    Returns
    -------
    list
        流域ID列表
    """
    import ast
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # 使用 ast.literal_eval 安全地解析列表
        # 首先找到列表部分
        start_idx = content.find('VALID_WATER_LEVEL_BASINS = [')
        if start_idx == -1:
            raise ValueError(f"未在文件中找到 VALID_WATER_LEVEL_BASINS")
        
        # 找到列表开始位置
        list_start = content.find('[', start_idx)
        if list_start == -1:
            raise ValueError(f"未找到列表开始标记")
        
        # 找到匹配的结束括号
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
        
        # 提取列表字符串
        list_str = content[list_start:list_end]
        
        # 解析列表
        basin_list = ast.literal_eval(list_str)
        
        print(f"从文件 {file_path} 读取了 {len(basin_list)} 个有水位数据文件的流域")
        return basin_list
        
    except FileNotFoundError:
        print(f"错误: 文件 {file_path} 不存在")
        print("请先运行 scan_waterlevel_basins.py 生成流域列表文件")
        raise
    except Exception as e:
        print(f"读取文件 {file_path} 时出错: {e}")
        raise


def filter_basins_with_valid_data(camelsh_reader, basin_list, time_range, max_basins_to_check=None, min_valid_ratio=0.1):
    """
    验证流域列表，只保留同时有有效水位和径流数据（不全为NaN）的流域
    
    Parameters
    ----------
    camelsh_reader : ImprovedCAMELSHReader
        CAMELSH数据读取器
    basin_list : list
        候选流域ID列表
    time_range : list
        时间范围 [start_date, end_date]，用于验证数据有效性
    max_basins_to_check : int, optional
        最多检查的流域数量，如果为None则检查全部
    min_valid_ratio : float, optional
        最小有效数据比例，默认0.1（10%）
    
    Returns
    -------
    list
        过滤后的有效流域ID列表（同时有有效水位和径流数据）
    """
    print(f"\n正在验证流域的水位和径流数据有效性...")
    print(f"候选流域数量: {len(basin_list)}")
    print(f"验证时间范围: {time_range}")
    print(f"最小有效数据比例: {min_valid_ratio:.1%}")
    
    # 限制检查数量以提高效率
    basins_to_check = basin_list[:max_basins_to_check] if max_basins_to_check else basin_list
    print(f"将检查前 {len(basins_to_check)} 个流域...")
    
    valid_basins = []
    invalid_basins = []
    
    # 批量检查（每次检查一批以提高效率）
    batch_size = 50
    for i in tqdm(range(0, len(basins_to_check), batch_size), desc="验证流域数据"):
        batch = basins_to_check[i:i+batch_size]
        
        try:
            # 同时加载这批流域的水位和径流数据
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
            
            # 转换为pandas格式检查
            waterlevel_df = None
            flow_df = None
            
            if StandardVariable.WATER_LEVEL in waterlevel_ds.data_vars:
                waterlevel_df = waterlevel_ds[StandardVariable.WATER_LEVEL].to_pandas().T
            else:
                print(f"\n警告: 数据集缺少water_level变量")
                for basin_id in batch:
                    invalid_basins.append((basin_id, "数据集缺少water_level变量"))
                continue
            
            if StandardVariable.STREAMFLOW in flow_ds.data_vars:
                flow_df = flow_ds[StandardVariable.STREAMFLOW].to_pandas().T
            else:
                print(f"\n警告: 数据集缺少streamflow变量")
                for basin_id in batch:
                    invalid_basins.append((basin_id, "数据集缺少streamflow变量"))
                continue
            
            # 检查每个流域的数据
            for basin_id in batch:
                waterlevel_valid = False
                flow_valid = False
                reasons = []
                
                # 检查水位数据
                if basin_id in waterlevel_df.columns:
                    wl_data = waterlevel_df[basin_id]
                    if wl_data.notna().any():
                        wl_valid_ratio = wl_data.notna().sum() / len(wl_data)
                        if wl_valid_ratio >= min_valid_ratio:
                            waterlevel_valid = True
                        else:
                            reasons.append(f"水位有效比例过低: {wl_valid_ratio:.2%}")
                    else:
                        reasons.append("水位数据全为NaN")
                else:
                    reasons.append("水位数据集中不存在")
                
                # 检查径流数据
                if basin_id in flow_df.columns:
                    flow_data = flow_df[basin_id]
                    if flow_data.notna().any():
                        flow_valid_ratio = flow_data.notna().sum() / len(flow_data)
                        if flow_valid_ratio >= min_valid_ratio:
                            flow_valid = True
                        else:
                            reasons.append(f"径流有效比例过低: {flow_valid_ratio:.2%}")
                    else:
                        reasons.append("径流数据全为NaN")
                else:
                    reasons.append("径流数据集中不存在")
                
                # 只有两种数据都有效才加入有效列表
                if waterlevel_valid and flow_valid:
                    valid_basins.append(basin_id)
                else:
                    reason_str = "; ".join(reasons) if reasons else "未知原因"
                    invalid_basins.append((basin_id, reason_str))
                    
        except Exception as e:
            # 如果加载失败，这批流域都无效
            print(f"\n警告: 批量加载失败 ({batch[0]} 到 {batch[-1]}): {e}")
            for basin_id in batch:
                invalid_basins.append((basin_id, f"加载失败: {str(e)[:50]}"))
    
    print(f"\n验证完成:")
    print(f"  有效流域（同时有有效水位和径流数据）: {len(valid_basins)} 个")
    print(f"  无效流域: {len(invalid_basins)} 个")
    
    if len(invalid_basins) > 0 and len(invalid_basins) <= 10:
        print(f"\n无效流域示例 (前{min(10, len(invalid_basins))}个):")
        for basin_id, reason in invalid_basins[:10]:
            print(f"  {basin_id}: {reason}")
    
    return valid_basins


if __name__ == "__main__":
    set_random_seed(1234)
    configure_chinese_font()
    
    # 打印设备信息
    print_device_info()
    
    # 导入配置
    from config import (
        CAMELSH_DATA_PATH, NUM_BASINS, SEQUENCE_LENGTH, BATCH_SIZE, EPOCHS,
        TRAIN_START, TRAIN_END, VALID_START, VALID_END, TEST_START, TEST_END,
        FORCING_VARIABLES, ATTRIBUTE_VARIABLES,
        IMAGES_SAVE_PATH, REPORTS_SAVE_PATH, MODEL_SAVE_PATH
    )
    
    # 创建输出文件夹
    import os
    os.makedirs(IMAGES_SAVE_PATH, exist_ok=True)
    os.makedirs(REPORTS_SAVE_PATH, exist_ok=True)
    os.makedirs(MODEL_SAVE_PATH, exist_ok=True)
    
    # 从文件读取有水位数据的流域列表
    print("\n正在从文件读取有水位数据的流域列表...")
    VALID_WATER_LEVEL_BASINS = load_waterlevel_basins_from_file("valid_waterlevel_basins.txt")
    
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
    # 使用CAMELSH的默认时间范围，后续按比例划分 train/valid/test
    default_range = camelsh.default_t_range
    print(f"数据集默认时间范围: {default_range}")
    print(f"将按比例划分时间序列: 训练 {TRAIN_RATIO:.0%}, 验证 {VALID_RATIO:.0%}, 测试 {TEST_RATIO:.0%}")
    
    # 验证流域：从文件读取的流域列表中，过滤出同时有有效水位和径流数据的流域
    # 使用训练时间范围来验证数据有效性（需要足够的流域数量，所以先验证更多候选）
    print(f"\n从文件中读取了 {len(VALID_WATER_LEVEL_BASINS)} 个候选流域")
    
    # 验证流域：需要检查足够多的候选流域以确保有足够的有效流域
    # 考虑到一些流域可能全为NaN或只有一种数据有效，检查更多候选（最多检查3倍数量，或全部候选）
    max_candidates = min(len(VALID_WATER_LEVEL_BASINS), max(NUM_BASINS * 3, 200))
    print(f"将检查前 {max_candidates} 个候选流域以确保找到足够的有效流域...")
    
    validated_basins = filter_basins_with_valid_data(
        camelsh_reader=camelsh_reader,
        basin_list=VALID_WATER_LEVEL_BASINS,
        time_range=default_range,  # 使用完整时间范围检查数据有效性
        max_basins_to_check=max_candidates,
        min_valid_ratio=0.1
    )
    
    if len(validated_basins) == 0:
        raise ValueError("未找到任何同时有有效水位和径流数据的流域！请检查数据文件。")
    
    if len(validated_basins) < NUM_BASINS:
        print(f"\n警告: 只找到了 {len(validated_basins)} 个有效流域，少于请求的 {NUM_BASINS} 个")
        print(f"将使用所有找到的有效流域: {len(validated_basins)} 个")
    
    # 选择前NUM_BASINS个有效流域（或全部有效流域，如果不足NUM_BASINS个）
    chosen_basins = validated_basins[:NUM_BASINS]
    print(f"\n最终选择的流域 ({len(chosen_basins)} 个): {chosen_basins}")
    print(f"注意：这些流域都经过验证，同时有有效的水位和径流数据（不全为NaN，有效数据比例≥10%）")
    
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
    
    # 准备气象强迫数据 - 使用改进的读取器（完整时间范围，用于按比例切分）
    print("\n正在加载气象强迫数据（完整时间范围，用于按比例切分）...")
    
    # 分离降雨和其他气象变量
    chosen_forcing_vars_no_precip = [v for v in chosen_forcing_vars 
                                      if v != StandardVariable.PRECIPITATION]
    
    print(f"从CAMELSH加载的非降雨变量: {[str(v) for v in chosen_forcing_vars_no_precip]}")
    
    # 从CAMELSH加载非降雨气象变量
    if chosen_forcing_vars_no_precip:
        forcings_ds_no_precip = camelsh_reader.read_ts_xrdataset(
            gage_id_lst=chosen_basins,
            t_range=default_range,
            var_lst=chosen_forcing_vars_no_precip
        )
        print(f"非降雨气象数据形状: {forcings_ds_no_precip.dims}")
        print(f"非降雨气象数据变量: {list(forcings_ds_no_precip.data_vars.keys())}")
    else:
        forcings_ds_no_precip = None
        print("警告: 没有非降雨气象变量需要从CAMELSH加载")
    
    # 从MSWEP加载降雨数据
    print("\n从MSWEP加载降雨数据...")
    mswep_precip_df = load_mswep_data(
        file_path="MSWEP/mswep_220basins_mean_3hourly_1980_2024.csv",
        basin_ids=chosen_basins,
        time_range=default_range
    )
    
    # 合并MSWEP降雨与其他气象变量
    if forcings_ds_no_precip is not None:
        forcings_ds = merge_forcing_with_mswep(forcings_ds_no_precip, mswep_precip_df)
    else:
        # 如果没有其他气象变量，只使用MSWEP降雨
        print("\n只使用MSWEP降雨数据创建数据集...")
        from mswep_loader import convert_mswep_to_xarray
        forcings_ds = convert_mswep_to_xarray(mswep_precip_df, var_name='precipitation')
    
    print(f"最终气象数据形状: {forcings_ds.dims}")
    print(f"最终气象数据变量: {list(forcings_ds.data_vars.keys())}")
    
    # ==================== 4. 加载径流和水位数据 ====================
    print("\n正在从CAMELSH数据集加载径流和水位数据...")
    
    # 使用改进的读取器加载完整时间范围上的径流和水位数据
    print("加载径流数据（完整时间范围，用于按比例切分）...")
    flow_ds = camelsh_reader.read_ts_xrdataset(
        gage_id_lst=chosen_basins,
        t_range=default_range,
        var_lst=[StandardVariable.STREAMFLOW]
    )
    
    print("加载水位数据（完整时间范围，用于按比例切分）...")
    try:
        waterlevel_ds = camelsh_reader.read_ts_xrdataset(
            gage_id_lst=chosen_basins,
            t_range=default_range,
            var_lst=[StandardVariable.WATER_LEVEL]
        )
    except Exception as e:
        print(f"警告: 无法加载水位数据: {e}")
        print("将使用模拟水位数据进行演示...")
        waterlevel_ds = flow_ds.copy()
        if StandardVariable.STREAMFLOW in waterlevel_ds.data_vars:
            flow_data = waterlevel_ds[StandardVariable.STREAMFLOW]
            water_level_data = flow_data * 0.1 + np.random.normal(0, 0.01, flow_data.shape)
            waterlevel_ds[StandardVariable.WATER_LEVEL] = water_level_data
            waterlevel_ds = waterlevel_ds.drop_vars([StandardVariable.STREAMFLOW])
    
    # 转换为 pandas DataFrame 格式以兼容现有代码
    print("转换径流与水位数据格式...")
    full_flow = flow_ds[StandardVariable.STREAMFLOW].to_pandas().T
    full_waterlevel = waterlevel_ds[StandardVariable.WATER_LEVEL].to_pandas().T
    
    print(f"径流数据形状: {full_flow.shape}")
    print(f"水位数据形状: {full_waterlevel.shape}")
    print(f"径流数据范围: {full_flow.min().min():.3f} - {full_flow.max().max():.3f}")
    print(f"水位数据范围: {full_waterlevel.min().min():.3f} - {full_waterlevel.max().max():.3f}")
    
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
        dates=default_range,
        data_attr=attrs_df,
        data_forcing=forcings_ds,
        data_flow=full_flow,
        data_waterlevel=full_waterlevel,
        loader_type="train",
        seq_length=sequence_length,
    )
    tr_loader = DataLoader(ds_train, batch_size=batch_size, shuffle=True)
    
    # 验证数据集
    means = ds_train.get_means()
    stds = ds_train.get_stds()
    ds_val = MultiTaskDataset(
        basins=chosen_basins,
        dates=default_range,
        data_attr=attrs_df,
        data_forcing=forcings_ds,
        data_flow=full_flow,
        data_waterlevel=full_waterlevel,
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
        dates=default_range,
        data_attr=attrs_df,
        data_forcing=forcings_ds,
        data_flow=full_flow,
        data_waterlevel=full_waterlevel,
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
    
    # ==================== 7. 训练模型（带早停机制）====================
    print("\n开始训练...")
    n_epochs = EPOCHS  # 训练轮数
    
    train_losses = []
    val_nses_flow = []
    val_nses_waterlevel = []
    
    # 早停机制相关变量
    best_val_nse_avg = -float('inf')  # 最佳验证NSE（两个任务的平均）
    best_epoch = 0  # 最佳epoch
    patience = 5  # 容忍度：多少个epoch不提升就停止
    patience_counter = 0  # 计数器
    best_model_state = None  # 保存最佳模型状态
    
    print(f"早停机制已启用，patience = {patience}")
    print("早停指标：两个任务NSE的平均值")
    
    for i in range(n_epochs):
        # 训练
        train_loss, train_loss_flow, train_loss_waterlevel = train_epoch(
            model, optimizer, tr_loader, loss_func, i + 1
        )
        train_losses.append(train_loss)
        
        # ===== 验证：按流域计算 NSE，不再使用 reshape =====
        obs_flow_dict, obs_waterlevel_dict, preds_flow_dict, preds_waterlevel_dict = eval_model(
            model, val_loader
        )
        
        nse_flow_list = []
        nse_waterlevel_list = []
        
        # 只处理实际有预测结果的流域（可能在创建索引表时某些流域被跳过了）
        available_basins = set(preds_flow_dict.keys())
        
        for basin in chosen_basins:
            b = str(basin)
            
            # 跳过没有预测结果的流域
            if b not in available_basins:
                continue
            
            # 反归一化（注意：local_denormalization 是按变量全局均值/标准差来的）
            pf = ds_val.local_denormalization(preds_flow_dict[b],b, variable="flow")
            of = ds_val.local_denormalization(obs_flow_dict[b],b, variable="flow")
            pw = ds_val.local_denormalization(preds_waterlevel_dict[b],b, variable="waterlevel")
            ow = ds_val.local_denormalization(obs_waterlevel_dict[b],b, variable="waterlevel")
            
            # 计算每个流域的 NSE
            nse_flow_list.append(he.nse(pf, of))
            nse_waterlevel_list.append(he.nse(pw, ow))
        
        current_nse_flow = np.mean(nse_flow_list)
        current_nse_waterlevel = np.mean(nse_waterlevel_list)
        current_nse_avg = (current_nse_flow + current_nse_waterlevel) / 2
        
        val_nses_flow.append(current_nse_flow)
        val_nses_waterlevel.append(current_nse_waterlevel)
        
        # 早停逻辑
        if current_nse_avg > best_val_nse_avg:
            best_val_nse_avg = current_nse_avg
            best_epoch = i + 1
            patience_counter = 0
            # 保存最佳模型状态
            best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            tqdm.write(
                f"Epoch {i+1} - "
                f"验证集 NSE (径流): {current_nse_flow:.4f}, "
                f"NSE (水位): {current_nse_waterlevel:.4f}, "
                f"平均: {current_nse_avg:.4f} ✓ [新最佳]"
            )
        else:
            patience_counter += 1
            tqdm.write(
                f"Epoch {i+1} - "
                f"验证集 NSE (径流): {current_nse_flow:.4f}, "
                f"NSE (水位): {current_nse_waterlevel:.4f}, "
                f"平均: {current_nse_avg:.4f} "
                f"(无改进 {patience_counter}/{patience})"
            )
            
            # 如果超过patience，停止训练
            if patience_counter >= patience:
                print(f"\n早停触发！平均验证集NSE已连续 {patience} 个epoch未改进")
                print(f"最佳模型出现在 Epoch {best_epoch}，平均NSE = {best_val_nse_avg:.4f}")
                print("加载最佳模型...")
                # 加载最佳模型
                model.load_state_dict({k: v.to(DEVICE) for k, v in best_model_state.items()})
                # 截断训练曲线到实际训练的epoch数
                train_losses = train_losses[:i+1]
                val_nses_flow = val_nses_flow[:i+1]
                val_nses_waterlevel = val_nses_waterlevel[:i+1]
                break

    
    # ==================== 8. 测试模型 ====================
    print("\n在测试集上评估...")
    obs_flow_dict, obs_waterlevel_dict, preds_flow_dict, preds_waterlevel_dict = eval_model(
        model, test_loader
    )
    
    # 反归一化 + 计算每个流域 NSE
    # 只处理实际有预测结果的流域（可能在创建索引表时某些流域被跳过了）
    available_basins = set(preds_flow_dict.keys())
    
    # 同时准备一个反归一化后的结果，用于画图和计算NSE
    denorm_obs_flow = {}
    denorm_obs_waterlevel = {}
    denorm_preds_flow = {}
    denorm_preds_waterlevel = {}
    basin_nse_flow = {}  # 流域到NSE的映射
    basin_nse_waterlevel = {}  # 流域到NSE的映射
    evaluated_basins = []  # 实际评估的流域列表
    
    for basin in chosen_basins:
        b = str(basin)
        
        # 跳过没有预测结果的流域
        if b not in available_basins:
            continue
        
        denorm_preds_flow[b] = ds_test.local_denormalization(
            preds_flow_dict[b],b, variable="flow"
        )
        denorm_preds_waterlevel[b] = ds_test.local_denormalization(
            preds_waterlevel_dict[b],b, variable="waterlevel"
        )
        denorm_obs_flow[b] = ds_test.local_denormalization(
            obs_flow_dict[b],b, variable="flow"
        )
        denorm_obs_waterlevel[b] = ds_test.local_denormalization(
            obs_waterlevel_dict[b],b, variable="waterlevel"
        )
        
        nse_f = he.nse(denorm_preds_flow[b], denorm_obs_flow[b])
        nse_w = he.nse(denorm_preds_waterlevel[b], denorm_obs_waterlevel[b])
        
        basin_nse_flow[b] = nse_f
        basin_nse_waterlevel[b] = nse_w
        evaluated_basins.append(b)
    
    print(f"\n测试集结果：")
    for b in evaluated_basins:
        print(f"流域 {b}:")
        print(f"  径流 NSE: {basin_nse_flow[b]:.4f}")
        print(f"  水位 NSE: {basin_nse_waterlevel[b]:.4f}")
    if evaluated_basins:
        print(f"平均 NSE (径流): {np.mean(list(basin_nse_flow.values())):.4f}")
        print(f"平均 NSE (水位): {np.mean(list(basin_nse_waterlevel.values())):.4f}")
    
    # ==================== 9. 可视化结果 ====================
    print("\n正在生成可视化图表...")
    
    # 日期范围：按测试集时间 + 序列长度来推一推
    # 注意：使用3小时分辨率（freq='3H'）
    start_date = pd.to_datetime(ds_test.dates[0], format="%Y-%m-%d") + pd.Timedelta(
        hours=(sequence_length - 1) * 3  # 3小时分辨率
    )
    
    for b in evaluated_basins:
        basin = b
        
        of = denorm_obs_flow[b]
        pf = denorm_preds_flow[b]
        ow = denorm_obs_waterlevel[b]
        pw = denorm_preds_waterlevel[b]
        
        # 针对当前流域的序列长度生成时间轴
        # 使用3小时分辨率（freq='3H'）
        date_range = pd.date_range(start_date, periods=len(of), freq='3H')
        
        fig, axes = plt.subplots(2, 1, figsize=(14, 8))
        
        # 径流预测图
        axes[0].plot(date_range, of, label="观测值", alpha=0.7)
        axes[0].plot(date_range, pf, label="预测值", alpha=0.7)
        axes[0].legend()
        axes[0].set_title(f"流域 {basin} - 径流预测 (测试集 NSE: {basin_nse_flow[b]:.3f})")
        axes[0].set_ylabel("径流 (mm/d)")
        axes[0].grid(True, alpha=0.3)
        
        # 水位预测图
        axes[1].plot(date_range, ow, label="观测值", alpha=0.7)
        axes[1].plot(date_range, pw, label="预测值", alpha=0.7)
        axes[1].legend()
        axes[1].set_title(f"流域 {basin} - 水位预测 (测试集 NSE: {basin_nse_waterlevel[b]:.3f})")
        axes[1].set_xlabel("日期")
        axes[1].set_ylabel("水位 (m)")
        axes[1].grid(True, alpha=0.3)
        
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        # 保存图片
        output_file = os.path.join(IMAGES_SAVE_PATH, f"multi_task_results_basin_{basin}.png")
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
    training_curve_file = os.path.join(IMAGES_SAVE_PATH, "multi_task_training_curves.png")
    plt.savefig(training_curve_file, dpi=300, bbox_inches='tight')
    print(f"已保存训练曲线: {training_curve_file}")
    
    # ==================== 10. 保存模型 ====================
    print("\n正在保存模型...")
    model_path = os.path.join(MODEL_SAVE_PATH, 'multi_task_lstm_model.pth')
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'means': means,
        'stds': stds,
        'task_weights': task_weights,
        'test_nse_flow': np.mean(list(basin_nse_flow.values())) if basin_nse_flow else 0.0,
        'test_nse_waterlevel': np.mean(list(basin_nse_waterlevel.values())) if basin_nse_waterlevel else 0.0,
    }, model_path)
    print(f"模型已保存: {model_path}")
    
    print("\n训练完成！")

