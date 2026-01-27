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
# from improved_camelsh_reader import ImprovedCAMELSHReader  # CAMELSH专用

from hydrodataset import StandardVariable
import HydroErr as he
# from mswep_loader import load_mswep_data, merge_forcing_with_mswep  # CAMELSH+MSWEP专用


DEVICE = torch.device(
    "cuda:0" if torch.cuda.is_available() else "cpu"
)  # check if GPU is available

# 时间划分比例（经验值）：按每个流域自身完整时间序列划分
TRAIN_RATIO = 0.6
VALID_RATIO = 0.2
TEST_RATIO = 0.2

# 滑窗步长（经验值）：序列起点之间的时间步间隔，减小步长会增加样本数和内存占用
TIME_STEP_HOURS = 24  # CAMELS-US日尺度数据：1步=24小时
WINDOW_STEP = 1  # 日尺度：每隔1天取一个样本起点


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


def filter_basins_with_valid_data(camelsh_reader, basin_list, time_range, target_type, max_basins_to_check=None, min_valid_ratio=0.1):
    """
    验证流域列表，只保留有有效目标数据（不全为NaN）的流域
    
    Parameters
    ----------
    camelsh_reader : ImprovedCAMELSHReader
        CAMELSH数据读取器
    basin_list : list
        候选流域ID列表
    time_range : list
        时间范围 [start_date, end_date]，用于验证数据有效性
    target_type : str
        目标类型 "flow" 或 "waterlevel"
    max_basins_to_check : int, optional
        最多检查的流域数量，如果为None则检查全部
    min_valid_ratio : float, optional
        最小有效数据比例，默认0.1（10%）
    
    Returns
    -------
    list
        过滤后的有效流域ID列表
    """
    print(f"\n正在验证流域的{target_type}数据有效性...")
    print(f"候选流域数量: {len(basin_list)}")
    print(f"验证时间范围: {time_range}")
    print(f"最小有效数据比例: {min_valid_ratio:.1%}")
    
    # 限制检查数量以提高效率
    basins_to_check = basin_list[:max_basins_to_check] if max_basins_to_check else basin_list
    print(f"将检查前 {len(basins_to_check)} 个流域...")
    
    valid_basins = []
    invalid_basins = []
    
    # 确定目标变量
    if target_type == "flow":
        target_var = StandardVariable.STREAMFLOW
    elif target_type == "waterlevel":
        target_var = StandardVariable.WATER_LEVEL
    else:
        raise ValueError(f"未知的目标类型: {target_type}")
    
    # 批量检查（每次检查一批以提高效率）
    batch_size = 50
    for i in tqdm(range(0, len(basins_to_check), batch_size), desc="验证流域数据"):
        batch = basins_to_check[i:i+batch_size]
        
        try:
            # 加载这批流域的目标数据
            target_ds = camelsh_reader.read_ts_xrdataset(
                gage_id_lst=batch,
                t_range=time_range,
                var_lst=[target_var]
            )
            
            # 转换为pandas格式检查
            target_df = None
            
            if target_var in target_ds.data_vars:
                target_df = target_ds[target_var].to_pandas().T
                # 统一将列名转换为字符串，避免类型不匹配
                target_df.columns = [str(col) for col in target_df.columns]
            else:
                print(f"\n警告: 数据集缺少{target_type}变量")
                for basin_id in batch:
                    invalid_basins.append((basin_id, f"数据集缺少{target_type}变量"))
                continue
            
            # 检查每个流域的数据
            for basin_id in batch:
                target_valid = False
                reasons = []
                
                # 统一转换为字符串进行比较
                basin_id_str = str(basin_id)
                
                # 检查目标数据
                if basin_id_str in target_df.columns:
                    target_data = target_df[basin_id_str]
                    if target_data.notna().any():
                        target_valid_ratio = target_data.notna().sum() / len(target_data)
                        if target_valid_ratio >= min_valid_ratio:
                            target_valid = True
                        else:
                            reasons.append(f"{target_type}有效比例过低: {target_valid_ratio:.2%}")
                    else:
                        reasons.append(f"{target_type}数据全为NaN")
                else:
                    # 尝试查找相似的列名（调试信息）
                    available_cols = list(target_df.columns)
                    reasons.append(f"{target_type}数据集中不存在 (可用列: {available_cols[:5]}...)")
                
                # 只有数据有效才加入有效列表
                if target_valid:
                    valid_basins.append(basin_id_str)  # 统一使用字符串
                else:
                    reason_str = "; ".join(reasons) if reasons else "未知原因"
                    invalid_basins.append((basin_id_str, reason_str))
                
                    
        except Exception as e:
            # 如果加载失败，这批流域都无效
            print(f"\n警告: 批量加载失败 ({batch[0]} 到 {batch[-1]}): {e}")
            for basin_id in batch:
                invalid_basins.append((basin_id, f"加载失败: {str(e)[:50]}"))
    
    print(f"\n验证完成:")
    print(f"  有效流域（有有效{target_type}数据）: {len(valid_basins)} 个")
    print(f"  无效流域: {len(invalid_basins)} 个")
    
    if len(invalid_basins) > 0 and len(invalid_basins) <= 10:
        print(f"\n无效流域示例 (前{min(10, len(invalid_basins))}个):")
        for basin_id, reason in invalid_basins[:10]:
            print(f"  {basin_id}: {reason}")
    
    return valid_basins


class SingleTaskDataset(Dataset):
    """单任务数据集类，用于加载单一目标变量数据（径流或水位）"""

    def __init__(
        self,
        basins: list,
        dates: list,
        data_attr: pd.DataFrame,
        data_forcing: xr.Dataset,
        data_target: pd.DataFrame,  # 目标数据（径流或水位）
        target_type: str,  # "flow" 或 "waterlevel"
        loader_type: str = "train",
        seq_length: int = 100,
        means: dict = None,
        stds: dict = None,
        data_aux: pd.DataFrame = None,  # 辅助数据（用于筛选时间段）
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
        target_type : str
            目标类型 "flow" 或 "waterlevel"
        loader_type : str, optional
            数据集类型 "train", "valid", "test"
        seq_length : int, optional
            输入序列长度
        means : dict, optional
            归一化均值字典
        stds : dict, optional
            归一化标准差字典
        data_aux : pd.DataFrame, optional
            辅助数据（用于确保单任务和多任务模型使用相同的时间段）
            如果提供，则只使用target和aux都非NaN的时间段
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
        # 确保basins列表中的元素都是字符串类型，以保持一致性
        self.basins = [str(b) for b in basins]
        self.dates = dates
        self.seq_length = seq_length
        self.means = means
        self.stds = stds

        self.data_attr = data_attr
        self.data_forcing = data_forcing
        self.data_target = data_target
        self.data_aux = data_aux  # 辅助数据（另一个目标变量）

        self.time_index = {}

        # 加载和预处理数据
        self._load_data()

    def __len__(self):
        return self.num_samples

    def __getitem__(self, item: int):
        basin, time_idx = self.lookup_table[item]
        seq_length = self.seq_length
        
        # time_idx 是真实 datetime，需要找到整数位置
        # 注意：强迫数据和目标数据可能有不同的时间索引（日尺度(1天) vs 小时）
        # 需要在强迫数据的时间索引中查找
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
            start_pos = self.data_target.index.get_loc(time_idx)
        
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
        
        # 获取目标值（序列最后一天的值）
        # 注意：目标数据使用原始时间索引，需要从lookup_table中获取的time_idx开始计算
        # 找到目标数据中对应的位置
        target_time_idx = time_idx
        if target_time_idx in self.data_target.index:
            target_pos = self.data_target.index.get_loc(target_time_idx)
            # 目标值对应的是序列结束时刻
            # 由于强迫数据是日尺度(1天)分辨率，需要找到对应的目标数据位置
            # 使用lookup_table中的time_idx加上(seq_length-1)*日尺度(1天)
            end_time = target_time_idx + pd.Timedelta(days=(seq_length - 1))
            if end_time in self.data_target.index:
                target_end_pos = self.data_target.index.get_loc(end_time)
                y = self.y[basin][target_end_pos]
            else:
                # 如果精确时间不存在，找最近的
                nearest_idx = self.data_target.index.get_indexer([end_time], method='nearest')[0]
                y = self.y[basin][nearest_idx]
        else:
            # 如果时间索引不存在，找最近的
            nearest_idx = self.data_target.index.get_indexer([target_time_idx], method='nearest')[0]
            end_time = target_time_idx + pd.Timedelta(days=(seq_length - 1))
            nearest_end_idx = self.data_target.index.get_indexer([end_time], method='nearest')[0]
            y = self.y[basin][nearest_end_idx]
        
        # 最终NaN检查
        if np.isnan(xc).any():
            print(f"[错误] 样本 {item}: 输入特征包含NaN")
            print(f"  流域: {basin}, 时间索引: {time_idx}")
            raise ValueError("输入特征包含NaN")
        
        if np.isnan(y):
            print(f"[错误] 样本 {item}: 目标值包含NaN")
            print(f"  流域: {basin}, 时间索引: {time_idx}")
            print(f"  目标时间: {end_time}")
            raise ValueError("目标值包含NaN")
        
        return torch.from_numpy(xc).float(), torch.from_numpy(
            np.array([y], dtype=np.float32)
        )

    def _load_data(self):
        """加载和预处理数据 - 按流域独立归一化"""
        if self.loader_type == "train":
            train_mode = True
            # 计算归一化参数
            self.means = {}
            self.stds = {}
        else:
            train_mode = False

        # 准备数据字典
        self.x = {}  # 强迫数据
        self.y = {}  # 目标数据

        # 获取强迫变量名
        forcing_vars = [var for var in self.data_forcing.data_vars]
        print(f"强迫变量: {forcing_vars}")

        # 训练模式下，先检查和处理原始数据中的NaN，然后计算统计量
        if train_mode:
            # 检查原始数据中的NaN
            print("检查原始数据中的NaN...")
            
            # 检查强迫数据
            # 对于 xarray Dataset，转换为数组后计算 NaN 数量
            forcing_nan_count = int(self.data_forcing.isnull().to_array().sum().values)
            print(f"强迫数据NaN统计: {forcing_nan_count}")
            if forcing_nan_count > 0:
                print(f"[警告] 强迫数据包含NaN值，将使用插值填充")
                # 使用线性插值方法
                self.data_forcing = self.data_forcing.interpolate_na(dim='time', method='linear')
                # 如果还有NaN，用均值填充
                remaining_nan = int(self.data_forcing.isnull().to_array().sum().values)
                if remaining_nan > 0:
                    self.data_forcing = self.data_forcing.fillna(self.data_forcing.mean())
            
            # 检查属性数据
            attr_nan_count = self.data_attr.isnull().sum().sum()
            if attr_nan_count > 0:
                print(f"[警告] 属性数据包含 {attr_nan_count} 个NaN值，将使用均值填充")
                numeric_cols = [col for col in self.data_attr.columns if col != 'gauge_id']
                self.data_attr[numeric_cols] = self.data_attr[numeric_cols].fillna(self.data_attr[numeric_cols].mean())
            
            # ========== 修复：不要对目标数据做全局填充！==========
            # 统一将列名转换为字符串
            self.data_target.columns = [str(col) for col in self.data_target.columns]
            
            # 只检查NaN数量，不做任何填充！
            # 后续在 _create_lookup_table 中会通过 dropna() 只使用有真实数据的时间点
            target_nan_count = self.data_target.isnull().sum().sum()
            total_values = self.data_target.size
            valid_ratio = (total_values - target_nan_count) / total_values if total_values > 0 else 0
            print(f"[信息] {self.target_type}数据统计:")
            print(f"  - 总数据点: {total_values}")
            print(f"  - 有效数据点: {total_values - target_nan_count} ({valid_ratio:.1%})")
            print(f"  - NaN数据点: {target_nan_count} ({target_nan_count/total_values:.1%})")
            print(f"  注意：NaN数据将被自动跳过，只使用真实观测数据进行训练")
            
            # 检查强迫数据的 NaN 情况
            print(f"\n[信息] 强迫数据 NaN 统计:")
            for var in self.data_forcing.data_vars:
                nan_count = int(self.data_forcing[var].isnull().sum().values)
                total = int(self.data_forcing[var].size)
                print(f"  - {var}: {nan_count}/{total} ({nan_count/total*100:.1f}%)")
            
            # 气象强迫数据的均值和标准差（全局）
            # skipna=True 确保在有 NaN 的情况下仍能正确计算统计量
            df_mean_forcings = self.data_forcing.mean(skipna=True).to_pandas()
            df_std_forcings = self.data_forcing.std(skipna=True).to_pandas()
            
            # 检查计算出的统计量
            if df_mean_forcings.isnull().any() or df_std_forcings.isnull().any():
                print("[错误] 强迫数据统计量包含NaN")
                print(f"均值 NaN 的变量: {df_mean_forcings[df_mean_forcings.isnull()].index.tolist()}")
                print(f"标准差 NaN 的变量: {df_std_forcings[df_std_forcings.isnull()].index.tolist()}")
                # 如果标准差为0或NaN，设置为1.0以避免除零错误
                df_std_forcings = df_std_forcings.fillna(1.0)
                df_std_forcings[df_std_forcings == 0] = 1.0
                print("[警告] 已将 NaN 或零标准差替换为 1.0")
            
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
            
            # 目标数据的均值和标准差（每个流域独立计算）
            target_mean = {}
            target_std = {}
            for basin in self.basins:
                basin_str = str(basin)
                if basin_str in self.data_target.columns:
                    series = self.data_target[basin_str]
                    target_mean[basin_str] = series.mean()
                    target_std[basin_str] = series.std() if series.std() > 1e-6 else 1.0
            
            self.means[self.target_type] = target_mean
            self.stds[self.target_type] = target_std
            
            # 最终NaN检查（只检查强迫数据和属性数据，目标数据允许有NaN）
            # 对于 xarray Dataset，转换为数组后计算 NaN 数量
            final_forcing_nan = int(self.data_forcing.isnull().to_array().sum().values)
            final_attr_nan = self.data_attr[numeric_cols].isnull().sum().sum()
            
            if final_forcing_nan > 0 or final_attr_nan > 0:
                total_forcing = int(self.data_forcing.to_array().size)
                total_attr = self.data_attr[numeric_cols].size
                forcing_nan_pct = final_forcing_nan / total_forcing * 100 if total_forcing > 0 else 0
                attr_nan_pct = final_attr_nan / total_attr * 100 if total_attr > 0 else 0
                print(f"[警告] 原始数据包含NaN: 强迫={final_forcing_nan}/{total_forcing} ({forcing_nan_pct:.2f}%), 属性={final_attr_nan}/{total_attr} ({attr_nan_pct:.2f}%)")
                print("  注意：归一化时会自动填充或跳过含NaN的数据点")
            
            print("数据统计量计算完成")
        else:
            # 非训练模式也需要统一列名
            self.data_target.columns = [str(col) for col in self.data_target.columns]

        # 归一化处理
        print("开始数据归一化...")
        
        # 检查哪些 basin 实际存在于数据中
        available_basins_in_forcing = [str(b) for b in self.data_forcing.basin.values]
        available_basins_in_target = [str(col) for col in self.data_target.columns]
        
        # 只保留在所有数据中都存在的 basin
        valid_basins = []
        for basin in self.basins:
            basin_str = str(basin)
            if basin_str in available_basins_in_forcing and basin_str in available_basins_in_target:
                valid_basins.append(basin_str)
            else:
                missing_in = []
                if basin_str not in available_basins_in_forcing:
                    missing_in.append("强迫数据")
                if basin_str not in available_basins_in_target:
                    missing_in.append("目标数据")
                print(f"[警告] 流域 {basin_str} 不在以下数据中: {', '.join(missing_in)}，将跳过")
        
        # 更新 self.basins 为有效的 basin 列表
        self.basins = valid_basins
        print(f"实际可用的流域数量: {len(self.basins)} 个")
        
        if len(self.basins) == 0:
            raise ValueError("没有找到任何在所有数据中都存在的流域！")
        
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
        self.y = self._normalize_target(self.data_target)
        
        # 同时保存原始数据用于评估
        if not train_mode:
            self.y_raw = self._dataframe_to_dict(self.data_target)
        
        self.train_mode = train_mode
        self._create_lookup_table()


    def _dataframe_to_dict(self, df):
        """将DataFrame转换为字典格式，保持NaN不变"""
        result = {}
        for basin in self.basins:
            if basin in df.columns:
                values = df[basin].values
                # 保持NaN不变，不做任何填充
                # NaN时间点会在 _create_lookup_table 中被自动跳过
                result[basin] = values
        return result

    def _normalize_forcing(self, data_forcing):
        """归一化气象强迫数据 - 使用全局统计量"""
        result = {}
        
        # 处理标准差为0的情况（避免除零错误）
        std_values = self.stds['forcing'].values.copy()
        zero_std_mask = std_values < 1e-8
        if np.any(zero_std_mask):
            print(f"警告：发现标准差接近0的强迫变量，将设为1避免除零错误")
            std_values[zero_std_mask] = 1.0
        
        # 获取 data_forcing 中可用的 basin 列表，统一转换为字符串
        available_basins_in_data = [str(b) for b in data_forcing.basin.values]
        
        # 调试：记录跳过的原因
        skip_reasons = {'not_in_data': 0, 'extract_failed': 0, 'high_nan': 0, 'fill_failed': 0, 'norm_failed': 0}
        
        # 调试：检查第一个流域的数据
        if len(self.basins) > 0:
            first_basin = str(self.basins[0])
            if first_basin in available_basins_in_data:
                try:
                    sample_data = data_forcing.sel(basin=first_basin).to_array().to_numpy().T
                    nan_count = np.isnan(sample_data).sum()
                    total_size = sample_data.size
                    print(f"\n[DIAGNOSTIC] First basin {first_basin} data:")
                    print(f"  - Shape: {sample_data.shape} (timesteps x variables)")
                    print(f"  - NaN count: {nan_count} / {total_size} ({nan_count/total_size*100:.1f}%)")
                    print(f"  - Data range: [{np.nanmin(sample_data):.2f}, {np.nanmax(sample_data):.2f}]")
                    
                    # 保存到文件
                    with open("debug_forcing_data.txt", "a") as f:
                        f.write(f"\nFirst basin {first_basin}:\n")
                        f.write(f"Shape: {sample_data.shape}\n")
                        f.write(f"NaN count: {nan_count} / {total_size} ({nan_count/total_size*100:.1f}%)\n")
                        f.write(f"Data range: [{np.nanmin(sample_data):.2f}, {np.nanmax(sample_data):.2f}]\n")
                        # 检查每个变量的 NaN 情况
                        f.write("\nPer-variable NaN counts:\n")
                        for var_idx, var_name in enumerate(data_forcing.data_vars):
                            var_nan = np.isnan(sample_data[:, var_idx]).sum()
                            var_total = sample_data.shape[0]
                            f.write(f"  {var_name}: {var_nan}/{var_total} ({var_nan/var_total*100:.1f}%)\n")
                except Exception as e:
                    print(f"[WARNING] Cannot extract first basin data: {e}")
                    with open("debug_forcing_data.txt", "a") as f:
                        f.write(f"\nError extracting first basin: {e}\n")
        
        for basin in self.basins:
            basin_str = str(basin)
            
            # 检查 basin 是否在数据中
            if basin_str not in available_basins_in_data:
                skip_reasons['not_in_data'] += 1
                continue
            
            try:
                basin_data = data_forcing.sel(basin=basin_str).to_array().to_numpy().T
            except KeyError:
                # 如果 sel 失败，尝试使用原始 basin 值
                try:
                    basin_data = data_forcing.sel(basin=basin).to_array().to_numpy().T
                except KeyError:
                    skip_reasons['extract_failed'] += 1
                    continue
            
            # 检查原始数据是否包含NaN
            if np.isnan(basin_data).any():
                nan_count = np.isnan(basin_data).sum()
                total_count = basin_data.size
                nan_ratio = nan_count / total_count * 100
                
                # 如果 NaN 比例太高（>50%），则跳过该流域
                if nan_ratio > 50.0:
                    skip_reasons['high_nan'] += 1
                    if skip_reasons['high_nan'] <= 3:  # 只打印前3个
                        print(f"[警告] 流域 {basin_str} NaN比例过高 ({nan_ratio:.1f}%)，跳过")
                    continue
                
                # 否则，用均值填充 NaN（按变量）
                for var_idx in range(basin_data.shape[1]):
                    var_data = basin_data[:, var_idx]
                    if np.isnan(var_data).any():
                        var_mean = np.nanmean(var_data)
                        if np.isnan(var_mean):  # 如果整列都是NaN，用全局均值
                            var_mean = self.means['forcing'].values[var_idx]
                        basin_data[:, var_idx] = np.where(np.isnan(var_data), var_mean, var_data)
                
                # 验证填充后不再有 NaN
                if np.isnan(basin_data).any():
                    skip_reasons['fill_failed'] += 1
                    if skip_reasons['fill_failed'] <= 3:  # 只打印前3个
                        remaining_nan = np.isnan(basin_data).sum()
                        print(f"[警告] 流域 {basin_str} NaN填充后仍有 {remaining_nan} 个NaN，跳过")
                    continue
            
            normalized = (basin_data - self.means['forcing'].values) / std_values
            
            # 最终检查
            if np.isnan(normalized).any():
                skip_reasons['norm_failed'] += 1
                if skip_reasons['norm_failed'] <= 3:  # 只打印前3个
                    print(f"[警告] 流域 {basin_str} 归一化后含NaN，跳过")
                continue
            
            result[basin_str] = normalized
        
        # 打印跳过统计
        total_skipped = sum(skip_reasons.values())
        if total_skipped > 0:
            print(f"\n[统计] 归一化跳过流域原因:")
            for reason, count in skip_reasons.items():
                if count > 0:
                    print(f"  - {reason}: {count} 个流域")
        
        return result

    def _normalize_attr(self, data_attr):
        """归一化属性数据（只处理数值列）- 使用全局统计量"""
        # 分离 gauge_id 和数值列
        gauge_ids = data_attr['gauge_id']
        numeric_cols = [col for col in data_attr.columns if col != 'gauge_id']
        numeric_attrs = data_attr[numeric_cols]
        
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
            raise ValueError("属性数据归一化后包含NaN")
        
        # 设置索引为 gauge_id，以便后续通过流域ID访问
        normalized.index = gauge_ids
        return normalized

    def _normalize_target(self, data_target):
        """归一化目标数据（按流域独立归一化）- 保持NaN不变"""
        result = {}
        # 统一将列名转换为字符串
        data_target.columns = [str(col) for col in data_target.columns]
        
        for basin in self.basins:
            basin_str = str(basin)
            if basin_str not in data_target.columns:
                print(f"[警告] 流域 {basin_str} 不在目标数据列中，跳过")
                continue
                
            values = data_target[basin_str].values
            
            # 获取该流域的均值和标准差
            if basin_str not in self.means.get(self.target_type, {}):
                raise KeyError(f"流域 {basin_str} 的{self.target_type}归一化参数不存在")
            mean = self.means[self.target_type][basin_str]
            std = self.stds[self.target_type][basin_str]
            
            # 直接归一化，保持NaN不变（pandas会自动跳过NaN）
            normalized = (values - mean) / std
            
            # NaN是正常的，因为不是所有时间点都有观测
            # 这些NaN时间点会在 _create_lookup_table 中被 dropna() 自动跳过
            
            result[basin_str] = normalized
        return result

    def _create_lookup_table(self):
        """
        为每个流域独立构建滑窗索引，并按该流域自身完整时间范围做比例切分。

        说明：
        - 对每个流域，从归一化后的目标数据（self.y）中找出非NaN时间索引
        - 如果提供了辅助数据（data_aux），则确保target和aux都非NaN
        - 按 (TRAIN_RATIO, VALID_RATIO, TEST_RATIO) 进行时间顺序切分
        - 在滑窗时，确保每个窗口内的数据是真正连续的（没有被NaN断开）
        - **关键**：使用强迫数据的时间索引（日尺度(1天)分辨率）作为基准
        """
        lookup = []
        seq_length = self.seq_length

        skipped_basins = []
        basins_with_samples = []
        basin_time_ranges = {}  # 记录每个流域的时间范围
        
        # 判断是否需要同时检查辅助数据
        use_aux = self.data_aux is not None
        if use_aux:
            print(f"  注意：为确保与多任务模型公平对比，将同时检查flow和waterlevel数据")
        
        # 使用强迫数据的时间索引作为基准（日尺度(1天)分辨率）
        forcing_times = self.forcing_time_index
        
        for basin in tqdm(self.basins, desc=f"创建 {self.loader_type} 索引表", disable=False):
            basin_str = str(basin)
            if basin_str not in self.y:
                skipped_basins.append(basin_str)
                continue

            # 找出在强迫数据时间范围内，目标数据非NaN的时间点
            target_values = self.y[basin_str]
            target_index = self.data_target.index
            
            # 找出目标数据非NaN的位置
            target_valid_mask = ~np.isnan(target_values)
            
            # 如果提供了辅助数据，也要检查辅助数据
            if use_aux:
                if basin_str not in self.data_aux.columns:
                    skipped_basins.append(basin_str)
                    continue
                aux_values = self.data_aux[basin_str].values
                aux_valid_mask = ~np.isnan(aux_values)
                # 两者都非NaN才是有效的
                target_and_aux_valid = target_valid_mask & aux_valid_mask
            else:
                target_and_aux_valid = target_valid_mask
            
            # 获取目标数据中所有有效的时间索引
            valid_target_times = set(target_index[target_and_aux_valid])
            
            # 找出强迫数据时间点中，对应的目标数据也有效的时间点
            valid_forcing_times = []
            for ft in forcing_times:
                # 计算这个强迫数据时间点对应的目标时间范围
                # 序列结束时刻
                end_time = ft + pd.Timedelta(days=(seq_length - 1))
                # 检查结束时刻的目标数据是否有效
                if end_time in valid_target_times:
                    valid_forcing_times.append(ft)
            
            if len(valid_forcing_times) < 1:
                skipped_basins.append(basin_str)
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
                window_end_time = window_start_time + pd.Timedelta(days=(seq_length - 1))
                
                # 检查结束时间的目标数据是否有效
                if window_end_time in valid_target_times:
                    lookup.append((basin_str, window_start_time))
                    num_samples_this_basin += 1
            
            if num_samples_this_basin > 0:
                basins_with_samples.append(basin_str)
                # 记录该流域的时间范围信息
                basin_time_ranges[basin_str] = {
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
                print(f"      完整数据范围: {info['first_valid']} 到 {info['last_valid']} (共{info['total_valid_hours']}个日尺度(1天)步)")
                print(f"      {self.loader_type.upper()}期范围: {info['loader_start']} 到 {info['loader_end']}")
                print(f"      生成样本数: {info['num_samples']}")
        
        if self.num_samples == 0:
            raise ValueError(f"{self.loader_type} 数据集没有生成任何样本！请检查数据有效性。")

    def get_means(self):
        return self.means

    def get_stds(self):
        return self.stds

    def local_denormalization(self, feature, basin):
        """按流域反归一化"""
        # 确保basin是字符串类型
        basin = str(basin)
        
        if basin not in self.means.get(self.target_type, {}):
            raise KeyError(f"流域 {basin} 的{self.target_type}归一化参数不存在")
        mean = self.means[self.target_type][basin]
        std = self.stds[self.target_type][basin]
        return feature * std + mean
    
    def get_raw_targets(self):
        """获取原始目标数据（仅测试模式）"""
        if hasattr(self, 'y_raw'):
            return self.y_raw
        else:
            # 如果没有原始数据，使用反归一化
            raw_data = {}
            for basin in self.y:
                raw_data[basin] = self.local_denormalization(self.y[basin], basin)
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
        
        # 预测
        pred = self.fc(hidden)
        
        return pred


def train_epoch(model, optimizer, loader, loss_func, epoch, target_type):
    """训练一个epoch"""
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
        loss = loss_func(pred, ys)
        
        # 检查损失
        if torch.isnan(loss):
            print(f"[错误] Epoch {epoch}, 批次 {batch_idx}: 损失包含NaN")
            raise ValueError("损失包含NaN")
        
        # 反向传播
        loss.backward()
        
        # 梯度裁剪防止梯度爆炸
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        
        # 累计损失
        epoch_loss += loss.item()
        
        # 更新进度条
        pbar.set_postfix_str(f"Loss: {loss.item():.4f}")
    
    # 检查epoch平均损失
    avg_loss = epoch_loss / len(loader)
    
    if np.isnan(avg_loss):
        print(f"[错误] Epoch {epoch}: 平均损失包含NaN")
        raise ValueError("平均损失包含NaN")
    
    return avg_loss


def validate_epoch(model, loader, loss_func):
    """验证一个epoch，返回平均损失"""
    model.eval()
    total_loss = 0.0
    num_batches = 0
    
    with torch.no_grad():
        for xs, ys in loader:
            xs, ys = xs.to(DEVICE), ys.to(DEVICE)
            pred = model(xs)
            loss = loss_func(pred, ys)
            total_loss += loss.item()
            num_batches += 1
    
    return total_loss / num_batches if num_batches > 0 else 0.0


def eval_model(model, loader, target_type):
    """评估模型：按流域分组返回观测和预测结果（按时间排序）"""
    model.eval()
    
    # 按流域存放结果：key = basin（流域ID），value = list of (time, obs, pred)
    obs_by_basin = {}
    preds_by_basin = {}
    times_by_basin = {}
    sample_offset = 0
    
    with torch.no_grad():
        for xs, ys in tqdm(loader, desc="评估中"):
            batch_size = xs.size(0)
            xs = xs.to(DEVICE)
            pred = model(xs)
            
            # 统一搬到 CPU + numpy，方便后面处理
            ys_np = ys.numpy().squeeze()  # [batch]
            pred_np = pred.cpu().numpy().squeeze()  # [batch]
            
            # 有可能 squeeze 后变成标量，统一处理成一维
            if pred_np.ndim == 0:
                pred_np = pred_np.reshape(1)
            if ys_np.ndim == 0:
                ys_np = ys_np.reshape(1)
            
            # 获取对应的流域ID和时间
            dataset = loader.dataset
            
            for i in range(batch_size):
                # 从lookup_table获取basin和时间信息
                sample_idx = sample_offset + i
                if sample_idx < len(dataset.lookup_table):
                    basin, time_idx = dataset.lookup_table[sample_idx]
                    b = str(basin)
                    
                    if b not in obs_by_basin:
                        obs_by_basin[b] = []
                        preds_by_basin[b] = []
                        times_by_basin[b] = []
                    
                    obs_by_basin[b].append(ys_np[i])
                    preds_by_basin[b].append(pred_np[i])
                    times_by_basin[b].append(time_idx)
            
            sample_offset += batch_size
    
    # 按时间排序后转成 numpy 数组
    sorted_obs_by_basin = {}
    sorted_preds_by_basin = {}

    for b in obs_by_basin.keys():
        # 按时间排序
        times = np.array(times_by_basin[b])
        order = np.argsort(times)

        sorted_obs_by_basin[b] = np.array(obs_by_basin[b])[order]
        sorted_preds_by_basin[b] = np.array(preds_by_basin[b])[order]

    return sorted_obs_by_basin, sorted_preds_by_basin


def _target_label(target_type: str) -> str:
    return "径流" if target_type == "flow" else "水位"


def plot_training_curves(train_losses, val_nses, target_type):
    """绘制训练损失与验证集NSE曲线"""
    target_label = _target_label(target_type)
    fig, axes = plt.subplots(1, 2, figsize=(14, 4))

    axes[0].plot(train_losses)
    axes[0].set_title(f"{target_label}模型训练损失")
    axes[0].set_xlabel("轮次")
    axes[0].set_ylabel("损失")
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(val_nses, label="验证集 NSE")
    axes[1].set_title(f"{target_label}模型验证集NSE")
    axes[1].set_xlabel("轮次")
    axes[1].set_ylabel("NSE")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()

    plt.tight_layout()
    plt.savefig(
        f"single_task_{target_type}_training_curves.png",
        dpi=300,
        bbox_inches="tight",
    )


def plot_test_predictions(results, target_type):
    """绘制测试集预测图"""
    target_label = _target_label(target_type)
    obs = results["obs"]
    preds = results["pred"]
    dates = results["dates"]
    nse_per_basin = results["nse_per_basin"]

    for basin in obs:
        if basin not in preds or basin not in dates:
            continue
        if obs[basin].size == 0 or preds[basin].size == 0:
            continue

        fig, axes = plt.subplots(2, 1, figsize=(14, 8))

        axes[0].plot(dates[basin], obs[basin], label="观测值", alpha=0.7)
        axes[0].plot(dates[basin], preds[basin], label="预测值", alpha=0.7)
        axes[0].legend()
        axes[0].set_title(
            f"流域 {basin} - {target_label}预测 (测试集 NSE: {nse_per_basin.get(basin, float('nan')):.3f})"
        )
        axes[0].set_ylabel(target_label)
        axes[0].grid(True, alpha=0.3)

        axes[1].scatter(obs[basin], preds[basin], alpha=0.6, s=8)
        min_val = min(obs[basin].min(), preds[basin].min())
        max_val = max(obs[basin].max(), preds[basin].max())
        axes[1].plot([min_val, max_val], [min_val, max_val], "r--")
        axes[1].set_xlabel(f"观测{target_label}")
        axes[1].set_ylabel(f"预测{target_label}")
        axes[1].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(
            f"single_task_{target_type}_results_{basin}.png",
            dpi=300,
            bbox_inches="tight",
        )


def train_camelsus_flow(num_epochs=None):
    """
    使用 CAMELS-US（日尺度）径流数据训练单任务 LSTM（只做 flow）。

    说明：这是把原脚本的 CAMELSHour(flow+waterlevel) 版本，改成 CAMELS-US(flow) 版，
    方便你做“换数据集验证代码”的 sanity check。
    """
    print(f"\n=== 开始训练 CAMELS-US 单任务 flow 模型 ===")

    # ---- 读取配置（尽量兼容你现有 config.py 的命名）----
    from config import (
        FORCING_VARIABLES,
        ATTRIBUTE_VARIABLES,
        NUM_BASINS,
        SEQUENCE_LENGTH,
        BATCH_SIZE,
        EPOCHS,
        LEARNING_RATE,
        IMAGES_SAVE_PATH,
        REPORTS_SAVE_PATH,
        MODEL_SAVE_PATH,
    )

    # CAMELS-US 数据路径：尝试多种常见变量名，避免你还要改 config
    try:
        from config import CAMELSUS_DATA_PATH as _CAMELSUS_DATA_PATH
    except Exception:
        try:
            from config import CAMELS_US_DATA_PATH as _CAMELSUS_DATA_PATH
        except Exception:
            try:
                from config import CAMELS_DATA_PATH as _CAMELSUS_DATA_PATH
            except Exception as e:
                raise ImportError(
                    "在 config.py 里找不到 CAMELS-US 数据路径变量。\n"
                    "请新增其一：CAMELSUS_DATA_PATH 或 CAMELS_US_DATA_PATH（指向 CAMELS-US 根目录）"
                ) from e

    if num_epochs is None:
        num_epochs = EPOCHS

    os.makedirs(IMAGES_SAVE_PATH, exist_ok=True)
    os.makedirs(REPORTS_SAVE_PATH, exist_ok=True)
    os.makedirs(MODEL_SAVE_PATH, exist_ok=True)

    set_random_seed(1234)
    configure_chinese_font()
    print_device_info()

    # 模型参数（沿用原脚本）
    HIDDEN_SIZE = 64
    DROPOUT_RATE = 0.2
    learning_rate = LEARNING_RATE
    batch_size = BATCH_SIZE
    sequence_length = SEQUENCE_LENGTH
    num_basins = NUM_BASINS

    # ==================== 1. 加载 CAMELS-US 数据 ====================
    print("\n正在加载 CAMELS-US 数据...")
    print(f"数据路径: {_CAMELSUS_DATA_PATH}")

    # CAMELS-US 读取器：使用 hydrodataset 的 CamelsUs 类
    try:
        from hydrodataset import CamelsUs
        camels = CamelsUs(
            data_path=_CAMELSUS_DATA_PATH,
            download=False
        )
    except Exception as e:
        raise ImportError(
            "无法导入/初始化 CAMELS-US 读取器。\n"
            "请确认 hydrodataset 包已正确安装。\n"
            f"原始错误: {e}"
        )

    # 获取流域列表
    basin_ids = camels.read_object_ids()
    basin_ids = [str(b) for b in basin_ids]
    print(f"CAMELS-US 总流域数: {len(basin_ids)}")
    print(f"前10个流域: {basin_ids[:10]}")

    # ==================== 2. 选择流域与时间范围 ====================
    # CAMELS-US 默认是日尺度，直接使用数据集默认时间范围，后续仍按每个流域自身完整序列比例切分
    default_range = camels.default_t_range
    print(f"数据集默认时间范围: {default_range}")
    print(f"将按比例划分时间序列: 训练 {TRAIN_RATIO:.0%}, 验证 {VALID_RATIO:.0%}, 测试 {TEST_RATIO:.0%}")

    # 选择候选流域（这里不再做 waterlevel 相关筛选，只做 flow 有效性验证）
    candidate_basins = basin_ids
    max_candidates = min(len(candidate_basins), max(num_basins * 3, 200))
    print(f"将检查前 {max_candidates} 个候选流域，过滤出有有效 flow 观测的流域...")

    validated_basins = filter_basins_with_valid_data(
        camelsh_reader=camels,   # 复用原函数：要求 reader 具备 read_ts_xrdataset
        basin_list=candidate_basins[:max_candidates],
        time_range=default_range,
        target_type="flow",
        max_basins_to_check=max_candidates,
        min_valid_ratio=0.1
    )
    if len(validated_basins) == 0:
        raise ValueError("未找到任何有有效 flow 数据的流域！请检查 CAMELS-US 数据路径/格式。")

    chosen_basins = validated_basins[:num_basins]
    print(f"最终选择的流域 ({len(chosen_basins)} 个): {chosen_basins}")

    # ==================== 3. 选择特征变量 ====================
    # CAMELS-US 特殊处理：temperature_mean 不存在，需要用 max 和 min 代替
    camelsus_forcing_vars = []
    for var_name in FORCING_VARIABLES:
        if var_name == "temperature_mean":
            # CAMELS-US 用 temperature_max 和 temperature_min
            camelsus_forcing_vars.append(StandardVariable.TEMPERATURE_MAX)
            camelsus_forcing_vars.append(StandardVariable.TEMPERATURE_MIN)
        elif hasattr(StandardVariable, var_name.upper()):
            camelsus_forcing_vars.append(getattr(StandardVariable, var_name.upper()))
        else:
            camelsus_forcing_vars.append(var_name)
    
    chosen_forcing_vars = camelsus_forcing_vars
    chosen_attrs_vars = ATTRIBUTE_VARIABLES
    print(f"选择的气象变量: {FORCING_VARIABLES} (temperature_mean → max+min)")
    print(f"选择的属性变量: {ATTRIBUTE_VARIABLES}")

    # ==================== 4. 加载属性 / 强迫 / 目标 ====================
    print("\n正在加载属性数据...")
    attrs = camels.read_attr_xrdataset(
        gage_id_lst=chosen_basins,
        var_lst=chosen_attrs_vars
    )
    # 转成 DataFrame（和原脚本保持一致：index=gauge_id, columns=属性名）
    attrs_df = attrs.to_dataframe().reset_index()
    if 'gauge_id' not in attrs_df.columns:
        # 不同实现可能叫 basin / gage / station
        for alt in ['basin', 'gage', 'station', 'id']:
            if alt in attrs_df.columns:
                attrs_df = attrs_df.rename(columns={alt: 'gauge_id'})
                break
    attrs_df = attrs_df.set_index('gauge_id')
    # 把 gauge_id 也保留成列，适配你原脚本的 _normalize_attr
    attrs_df = attrs_df.reset_index()

    print("\n正在加载气象强迫数据（完整时间范围，用于按比例切分）...")
    forcings_ds = camels.read_ts_xrdataset(
        gage_id_lst=chosen_basins,
        t_range=default_range,
        var_lst=chosen_forcing_vars
    )
    
    # 诊断：打印加载的数据信息
    print(f"\n[DIAGNOSTIC] Forcing dataset info:")
    print(f"  - Dimensions: {dict(forcings_ds.dims)}")
    print(f"  - Variables: {list(forcings_ds.data_vars)}")
    print(f"  - Number of basins: {len(forcings_ds.basin)}")
    print(f"  - Number of timesteps: {len(forcings_ds.time)}")
    
    # 保存诊断信息到文件
    with open("debug_forcing_data.txt", "w") as f:
        f.write(f"Forcing dataset diagnostics:\n")
        f.write(f"Dimensions: {dict(forcings_ds.dims)}\n")
        f.write(f"Variables: {list(forcings_ds.data_vars)}\n")
        f.write(f"Basins: {len(forcings_ds.basin)}\n")
        f.write(f"Timesteps: {len(forcings_ds.time)}\n")
        f.write(f"Time range: {forcings_ds.time.values[0]} to {forcings_ds.time.values[-1]}\n")
    
    # 如果有 temperature_max 和 temperature_min，计算 temperature_mean
    if (StandardVariable.TEMPERATURE_MAX in forcings_ds.data_vars and 
        StandardVariable.TEMPERATURE_MIN in forcings_ds.data_vars):
        print("正在计算 temperature_mean = (max + min) / 2...")
        forcings_ds[StandardVariable.TEMPERATURE_MEAN] = (
            forcings_ds[StandardVariable.TEMPERATURE_MAX] + 
            forcings_ds[StandardVariable.TEMPERATURE_MIN]
        ) / 2.0
        # 移除 max 和 min，只保留 mean
        forcings_ds = forcings_ds.drop_vars([
            StandardVariable.TEMPERATURE_MAX, 
            StandardVariable.TEMPERATURE_MIN
        ])
        print(f"  - 计算后变量: {list(forcings_ds.data_vars)}")

    print("\n正在加载径流目标数据（完整时间范围）...")
    target_ds = camels.read_ts_xrdataset(
        gage_id_lst=chosen_basins,
        t_range=default_range,
        var_lst=[StandardVariable.STREAMFLOW]
    )
    if StandardVariable.STREAMFLOW not in target_ds.data_vars:
        raise KeyError("目标数据集中缺少 STREAMFLOW 变量，请检查 CAMELS-US reader 的变量映射。")
    flow_df = target_ds[StandardVariable.STREAMFLOW].to_pandas().T
    flow_df.columns = [str(c) for c in flow_df.columns]

    # ==================== 5. 构建 Dataset / DataLoader ====================
    train_set = SingleTaskDataset(
        basins=chosen_basins,
        dates=default_range,
        data_attr=attrs_df,
        data_forcing=forcings_ds,
        data_target=flow_df,
        target_type="flow",
        loader_type="train",
        seq_length=sequence_length
    )
    means = train_set.get_means()
    stds = train_set.get_stds()

    valid_set = SingleTaskDataset(
        basins=chosen_basins,
        dates=default_range,
        data_attr=attrs_df,
        data_forcing=forcings_ds,
        data_target=flow_df,
        target_type="flow",
        loader_type="valid",
        seq_length=sequence_length,
        means=means,
        stds=stds
    )

    test_set = SingleTaskDataset(
        basins=chosen_basins,
        dates=default_range,
        data_attr=attrs_df,
        data_forcing=forcings_ds,
        data_target=flow_df,
        target_type="flow",
        loader_type="test",
        seq_length=sequence_length,
        means=means,
        stds=stds
    )

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, drop_last=True)
    valid_loader = DataLoader(valid_set, batch_size=batch_size, shuffle=False, drop_last=False)
    test_loader  = DataLoader(test_set,  batch_size=batch_size, shuffle=False, drop_last=False)

    # ==================== 6. 定义模型 / 优化器 / Loss ====================
    input_size = next(iter(train_set.x.values())).shape[1] + train_set.c.shape[1]
    model = SingleTaskLSTM(input_size=input_size, hidden_size=HIDDEN_SIZE, dropout_rate=DROPOUT_RATE).to(DEVICE)

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    loss_func = nn.MSELoss()

    # ==================== 7. 训练（含早停） ====================
    best_val = float('inf')
    patience = 10
    patience_cnt = 0
    best_path = os.path.join(MODEL_SAVE_PATH, "camelsus_flow_best.pt")

    train_losses, val_losses = [], []
    for epoch in range(1, num_epochs + 1):
        tr_loss = train_epoch(model, optimizer, train_loader, loss_func, epoch, target_type="flow")
        val_loss = validate_epoch(model, valid_loader, loss_func)
        train_losses.append(tr_loss)
        val_losses.append(val_loss)

        print(f"Epoch {epoch:03d} | train={tr_loss:.4f} | val={val_loss:.4f}")

        if val_loss < best_val - 1e-6:
            best_val = val_loss
            patience_cnt = 0
            torch.save({'model': model.state_dict(), 'means': means, 'stds': stds}, best_path)
            print(f"  ✅ 更新 best 模型: {best_path}")
        else:
            patience_cnt += 1
            if patience_cnt >= patience:
                print(f"  ⏹ 早停触发：连续 {patience} 次 val 未提升")
                break

    # ==================== 8. 测试评估 ====================
    ckpt = torch.load(best_path, map_location=DEVICE)
    model.load_state_dict(ckpt['model'])

    # 这里沿用你原来的 eval_model / 画图 / 指标输出逻辑（如果你后面代码里有）
    results = eval_model(model, test_loader, target_type="flow")
    return results, best_path


# 兼容旧入口：你可以继续在命令行里跑 python xxx.py
def train_single_task_model(target_type="flow", num_epochs=None):
    if target_type != "flow":
        raise ValueError("CAMELS-US 版本只支持 target_type='flow'（不含 waterlevel）")
    return train_camelsus_flow(num_epochs=num_epochs)

if __name__ == "__main__":
    set_random_seed(1234)
    configure_chinese_font()
    
    # 打印设备信息
    print_device_info()
    
    # 训练流量预测模型
    print("训练流量预测模型...")
    flow_model, flow_means, flow_stds, flow_nse = train_single_task_model("flow")
    
    # 训练水位预测模型
    #print("\n训练水位预测模型...")
    #waterlevel_model, wl_means, wl_stds, wl_nse = train_single_task_model("waterlevel")
    
    print(f"\n=== 单任务模型训练完成 ===")
    print(f"流量模型NSE: {flow_nse:.4f}")
    #print(f"水位模型NSE: {wl_nse:.4f}")