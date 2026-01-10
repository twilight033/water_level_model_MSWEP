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
from improved_camelsh_reader import ImprovedCAMELSHReader
from hydrodataset import StandardVariable
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
        qualifiers_csv_path: str = None,  # 权重CSV路径（可选）
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
        # 注意：强迫数据和目标数据可能有不同的时间索引（3小时 vs 小时）
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
            # 由于强迫数据是3小时分辨率，需要找到对应的目标数据位置
            # 使用lookup_table中的time_idx加上(seq_length-1)*3小时
            end_time = target_time_idx + pd.Timedelta(hours=(seq_length - 1) * 3)
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
            end_time = target_time_idx + pd.Timedelta(hours=(seq_length - 1) * 3)
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
        
        # 查询权重（如果有）
        weight = self._get_sample_weight(basin, end_time)
        
        return torch.from_numpy(xc).float(), torch.from_numpy(
            np.array([y], dtype=np.float32)
        ), weight

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
            
            # 气象强迫数据的均值和标准差（全局）
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
                print(f"[错误] 强迫/属性数据清洗后仍有NaN: 强迫={final_forcing_nan}, 属性={final_attr_nan}")
                raise ValueError("强迫/属性数据清洗失败，仍存在NaN值")
            else:
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
        
        for basin in self.basins:
            basin_str = str(basin)
            
            # 检查 basin 是否在数据中
            if basin_str not in available_basins_in_data:
                print(f"[警告] 流域 {basin_str} 不在强迫数据中，跳过")
                continue
            
            try:
                basin_data = data_forcing.sel(basin=basin_str).to_array().to_numpy().T
            except KeyError:
                # 如果 sel 失败，尝试使用原始 basin 值
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
        - **关键**：使用强迫数据的时间索引（3小时分辨率）作为基准
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
        
        # 使用强迫数据的时间索引作为基准（3小时分辨率）
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
                end_time = ft + pd.Timedelta(hours=(seq_length - 1) * 3)
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
                window_end_time = window_start_time + pd.Timedelta(hours=(seq_length - 1) * 3)
                
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
                print(f"      完整数据范围: {info['first_valid']} 到 {info['last_valid']} (共{info['total_valid_hours']}个3小时步)")
                print(f"      {self.loader_type.upper()}期范围: {info['loader_start']} 到 {info['loader_end']}")
                print(f"      生成样本数: {info['num_samples']}")
        
        if self.num_samples == 0:
            raise ValueError(f"{self.loader_type} 数据集没有生成任何样本！请检查数据有效性。")

    def get_means(self):
        return self.means

    def get_stds(self):
        return self.stds

    def _load_qualifiers_weights(self, csv_path: str):
        """加载qualifiers权重CSV并建立索引"""
        import os
        if not os.path.exists(csv_path):
            print(f"[警告] 权重文件不存在: {csv_path}，将使用默认权重1.0")
            return
        
        print(f"正在加载权重数据: {csv_path}")
        try:
            # 只读取需要的列
            df = pd.read_csv(csv_path, usecols=['datetime', 'gauge_id', 'Q_weight', 'H_weight'])
            df['datetime'] = pd.to_datetime(df['datetime'])
            df['gauge_id'] = df['gauge_id'].astype(str)
            
            # 建立 (datetime, gauge_id) -> weight 索引
            # 根据 target_type 选择对应权重列
            weight_col = 'Q_weight' if self.target_type == 'flow' else 'H_weight'
            self.qualifiers_weights = {}
            for _, row in df.iterrows():
                key = (row['datetime'], row['gauge_id'])
                self.qualifiers_weights[key] = float(row[weight_col])
            
            print(f"  成功加载 {len(self.qualifiers_weights)} 条权重记录 (目标: {self.target_type})")
        except Exception as e:
            print(f"[警告] 加载权重文件失败: {e}，将使用默认权重1.0")
            self.qualifiers_weights = None
    
    def _get_sample_weight(self, basin: str, target_time):
        """获取样本权重"""
        if self.qualifiers_weights is None:
            return 1.0
        
        basin_str = str(basin)
        # 确保 target_time 是 pandas Timestamp
        if isinstance(target_time, pd.Timestamp):
            lookup_time = target_time
        else:
            lookup_time = pd.Timestamp(target_time)
        
        key = (lookup_time, basin_str)
        weight = self.qualifiers_weights.get(key, 1.0)
        return weight
    
    def _load_qualifiers_weights(self, csv_path: str):
        """加载qualifiers权重CSV并建立索引"""
        import os
        if not os.path.exists(csv_path):
            print(f"[警告] 权重文件不存在: {csv_path}，将使用默认权重1.0")
            return
        
        print(f"正在加载权重数据: {csv_path}")
        try:
            # 只读取需要的列
            df = pd.read_csv(csv_path, usecols=['datetime', 'gauge_id', 'Q_weight', 'H_weight'])
            df['datetime'] = pd.to_datetime(df['datetime'])
            df['gauge_id'] = df['gauge_id'].astype(str)
            
            # 建立 (datetime, gauge_id) -> weight 索引
            # 根据 target_type 选择对应权重列
            weight_col = 'Q_weight' if self.target_type == 'flow' else 'H_weight'
            self.qualifiers_weights = {}
            for _, row in df.iterrows():
                key = (row['datetime'], row['gauge_id'])
                self.qualifiers_weights[key] = float(row[weight_col])
            
            print(f"  成功加载 {len(self.qualifiers_weights)} 条权重记录 (目标: {self.target_type})")
        except Exception as e:
            print(f"[警告] 加载权重文件失败: {e}，将使用默认权重1.0")
            self.qualifiers_weights = None
    
    def _get_sample_weight(self, basin: str, target_time):
        """获取样本权重"""
        if self.qualifiers_weights is None:
            return 1.0
        
        basin_str = str(basin)
        # 确保 target_time 是 pandas Timestamp
        if isinstance(target_time, pd.Timestamp):
            lookup_time = target_time
        else:
            lookup_time = pd.Timestamp(target_time)
        
        key = (lookup_time, basin_str)
        weight = self.qualifiers_weights.get(key, 1.0)
        return weight

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
    
    for batch_idx, batch_data in enumerate(pbar):
        # 解包（兼容有权重和无权重两种情况）
        if len(batch_data) == 3:
            xs, ys, weights = batch_data
            weights_tensor = torch.tensor(weights, dtype=torch.float32).to(DEVICE)
        else:
            xs, ys = batch_data
            weights_tensor = None
        
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
        
        # 计算损失（应用样本权重）
        loss_per_sample = F.mse_loss(pred, ys, reduction='none')
        if weights_tensor is not None:
            loss_per_sample = loss_per_sample * weights_tensor.view(-1, 1)
        loss = loss_per_sample.mean()
        
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


def eval_model(model, loader, target_type):
    """评估模型：按流域分组返回观测和预测结果（按时间排序）"""
    model.eval()
    
    # 按流域存放结果：key = basin（流域ID），value = list of (time, obs, pred)
    obs_by_basin = {}
    preds_by_basin = {}
    times_by_basin = {}
    sample_offset = 0
    
    with torch.no_grad():
        for batch_data in tqdm(loader, desc="评估中"):
            # 解包（兼容有权重和无权重两种情况）
            if len(batch_data) == 3:
                xs, ys, _ = batch_data  # 忽略权重
            else:
                xs, ys = batch_data
            
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


def train_single_task_model(target_type="flow", num_epochs=None):
    """
    训练单任务模型
    
    Parameters
    ----------
    target_type : str
        目标类型 "flow" 或 "waterlevel"
    num_epochs : int
        训练轮数
    """
    print(f"\n=== 开始训练单任务{target_type}模型 ===")
    
    # 导入配置
    from config import (
        CAMELSH_DATA_PATH,
        FORCING_VARIABLES,
        ATTRIBUTE_VARIABLES,
        VALID_WATER_LEVEL_BASINS,
        NUM_BASINS,
        SEQUENCE_LENGTH,
        BATCH_SIZE,
        TRAIN_START,
        TRAIN_END,
        VALID_START,
        VALID_END,
        TEST_START,
        TEST_END,
        EPOCHS,
        LEARNING_RATE,
        IMAGES_SAVE_PATH,
        REPORTS_SAVE_PATH,
        MODEL_SAVE_PATH,
    )
    
    if num_epochs is None:
        num_epochs = EPOCHS
    
    # 创建输出文件夹
    import os
    os.makedirs(IMAGES_SAVE_PATH, exist_ok=True)
    os.makedirs(REPORTS_SAVE_PATH, exist_ok=True)
    os.makedirs(MODEL_SAVE_PATH, exist_ok=True)
    
    # 设置随机种子
    set_random_seed(1234)
    
    # 模型参数
    HIDDEN_SIZE = 64
    DROPOUT_RATE = 0.2
    learning_rate = LEARNING_RATE
    batch_size = BATCH_SIZE
    sequence_length = SEQUENCE_LENGTH
    num_basins = NUM_BASINS
    
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
    
    # 尝试从文件读取有目标数据的流域列表
    candidate_basins = None
    try:
        print("\n正在从文件读取有目标数据的流域列表...")
        candidate_basins = load_waterlevel_basins_from_file("valid_waterlevel_basins.txt")
        print(f"从文件读取了 {len(candidate_basins)} 个候选流域")
    except (FileNotFoundError, ValueError) as e:
        print(f"无法从文件读取流域列表: {e}")
        print("将从所有可用流域中自动筛选...")
        # 如果文件不存在或读取失败，使用所有可用流域
        candidate_basins = [str(b) for b in basin_ids]
        print(f"使用所有可用流域作为候选: {len(candidate_basins)} 个")
    
    # 如果从文件读取的流域太少，补充使用所有可用流域
    if candidate_basins and len(candidate_basins) < NUM_BASINS:
        print(f"\n从文件读取的流域数量 ({len(candidate_basins)}) 少于请求的 {NUM_BASINS} 个")
        print("将从所有可用流域中补充候选...")
        all_basins = [str(b) for b in basin_ids]
        # 合并并去重
        combined_basins = list(dict.fromkeys(candidate_basins + all_basins))
        candidate_basins = combined_basins
        print(f"合并后的候选流域数量: {len(candidate_basins)} 个")
    
    # 验证流域：从候选流域列表中，过滤出有有效目标数据的流域
    print(f"\n开始验证候选流域的{target_type}数据有效性...")
    print(f"候选流域数量: {len(candidate_basins)}")
    
    # 验证流域：需要检查足够多的候选流域以确保有足够的有效流域
    max_candidates = min(len(candidate_basins), max(NUM_BASINS * 3, 200))
    print(f"将检查前 {max_candidates} 个候选流域以确保找到足够的有效流域...")
    
    validated_basins = filter_basins_with_valid_data(
        camelsh_reader=camelsh_reader,
        basin_list=candidate_basins,
        time_range=default_range,  # 使用完整时间范围检查数据有效性
        target_type=target_type,
        max_basins_to_check=max_candidates,
        min_valid_ratio=0.1
    )
    
    if len(validated_basins) == 0:
        raise ValueError(f"未找到任何有有效{target_type}数据的流域！请检查数据文件。")
    
    if len(validated_basins) < NUM_BASINS:
        print(f"\n警告: 只找到了 {len(validated_basins)} 个有效流域，少于请求的 {NUM_BASINS} 个")
        print(f"将使用所有找到的有效流域: {len(validated_basins)} 个")
    
    # 选择前NUM_BASINS个有效流域（或全部有效流域，如果不足NUM_BASINS个）
    chosen_basins = validated_basins[:NUM_BASINS]
    print(f"\n最终选择的流域 ({len(chosen_basins)} 个): {chosen_basins}")
    print(f"注意：这些流域都经过验证，有有效的{target_type}数据（不全为NaN，有效数据比例≥10%）")
    
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
    
    # ==================== 4. 加载目标数据 ====================
    # 为了公平对比，单任务模型也要同时加载flow和waterlevel数据
    # 确保使用的是两者都存在的时间段
    print(f"\n正在从CAMELSH数据集加载目标数据...")
    print(f"注意：为了与多任务模型公平对比，将同时加载flow和waterlevel数据")
    print(f"      单任务模型只在两者都有数据的时间段上训练/测试")
    
    # 加载flow数据
    print("加载flow数据...")
    flow_ds = camelsh_reader.read_ts_xrdataset(
        gage_id_lst=chosen_basins,
        t_range=default_range,
        var_lst=[StandardVariable.STREAMFLOW]
    )
    full_flow = flow_ds[StandardVariable.STREAMFLOW].to_pandas().T
    full_flow.columns = [str(col) for col in full_flow.columns]
    
    # 加载waterlevel数据
    print("加载waterlevel数据...")
    wl_ds = camelsh_reader.read_ts_xrdataset(
        gage_id_lst=chosen_basins,
        t_range=default_range,
        var_lst=[StandardVariable.WATER_LEVEL]
    )
    full_waterlevel = wl_ds[StandardVariable.WATER_LEVEL].to_pandas().T
    full_waterlevel.columns = [str(col) for col in full_waterlevel.columns]
    
    # 确定当前任务的主要目标数据
    if target_type == "flow":
        full_target = full_flow
        full_aux = full_waterlevel  # 辅助数据，用于筛选时间段
    else:  # waterlevel
        full_target = full_waterlevel
        full_aux = full_flow  # 辅助数据，用于筛选时间段
    
    print(f"\n数据统计:")
    print(f"  Flow数据形状: {full_flow.shape}")
    print(f"  Waterlevel数据形状: {full_waterlevel.shape}")
    print(f"  主目标({target_type})数据范围: {full_target.min().min():.3f} - {full_target.max().max():.3f}")
    
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
    
    # 训练数据集（在数据集内部按比例切分时间）
    # 传入辅助数据，确保只使用flow和waterlevel都存在的时间段
    qualifiers_csv = "qualifiers_output/camelsh_with_qualifiers.csv"
    ds_train = SingleTaskDataset(
        basins=chosen_basins,
        dates=default_range,
        data_attr=attrs_df,
        data_forcing=forcings_ds,
        data_target=full_target,
        target_type=target_type,
        loader_type="train",
        seq_length=sequence_length,
        data_aux=full_aux,  # 辅助数据，用于筛选时间段
        qualifiers_csv_path=qualifiers_csv,
    )
    tr_loader = DataLoader(ds_train, batch_size=batch_size, shuffle=True)
    
    # 验证数据集
    means = ds_train.get_means()
    stds = ds_train.get_stds()
    ds_val = SingleTaskDataset(
        basins=chosen_basins,
        dates=default_range,
        data_attr=attrs_df,
        data_forcing=forcings_ds,
        data_target=full_target,
        target_type=target_type,
        loader_type="valid",
        seq_length=sequence_length,
        means=means,
        stds=stds,
        data_aux=full_aux,  # 辅助数据，用于筛选时间段
        qualifiers_csv_path=qualifiers_csv,
    )
    valid_batch_size = 1000
    val_loader = DataLoader(ds_val, batch_size=valid_batch_size, shuffle=False)
    
    # 测试数据集
    ds_test = SingleTaskDataset(
        basins=chosen_basins,
        dates=default_range,
        data_attr=attrs_df,
        data_forcing=forcings_ds,
        data_target=full_target,
        target_type=target_type,
        loader_type="test",
        seq_length=sequence_length,
        means=means,
        stds=stds,
        data_aux=full_aux,  # 辅助数据，用于筛选时间段
        qualifiers_csv_path=qualifiers_csv,
    )
    test_batch_size = 1000
    test_loader = DataLoader(ds_test, batch_size=test_batch_size, shuffle=False)
    
    # ==================== 6. 创建模型 ====================
    print("\n正在创建单任务LSTM模型...")
    input_size = len(chosen_attrs_vars) + len(chosen_forcing_vars)
    hidden_size = 64  # LSTM隐藏层大小
    dropout_rate = 0.2  # Dropout率
    learning_rate = 1e-3  # 学习率
    
    model = SingleTaskLSTM(
        input_size=input_size, 
        hidden_size=hidden_size, 
        dropout_rate=dropout_rate
    ).to(DEVICE)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    loss_func = nn.MSELoss()
    
    print(f"模型参数量: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")
    print(f"设备: {DEVICE}")
    
    # ==================== 7. 训练模型 ====================
    print("\n开始训练...")
    n_epochs = num_epochs
    
    train_losses = []
    val_nses = []
    
    for i in range(n_epochs):
        # 训练
        train_loss = train_epoch(
            model, optimizer, tr_loader, loss_func, i + 1, target_type
        )
        train_losses.append(train_loss)
        
        # ===== 验证：按流域计算 NSE =====
        obs_dict, preds_dict = eval_model(
            model, val_loader, target_type
        )
        
        nse_list = []
        
        # 只处理实际有预测结果的流域（可能在创建索引表时某些流域被跳过了）
        available_basins = set(preds_dict.keys())
        
        for basin in chosen_basins:
            b = str(basin)
            
            # 跳过没有预测结果的流域
            if b not in available_basins:
                continue
            
            # 反归一化（按流域独立归一化）
            pf = ds_val.local_denormalization(preds_dict[b], b)
            of = ds_val.local_denormalization(obs_dict[b], b)
            
            # 计算每个流域的 NSE
            nse_list.append(he.nse(pf, of))
        
        val_nses.append(np.mean(nse_list))
        
        tqdm.write(
            f"Epoch {i+1} - "
            f"训练损失: {train_loss:.6f}, "
            f"验证集 NSE: {np.mean(nse_list):.4f}"
        )

    
    # ==================== 8. 测试模型 ====================
    print("\n在测试集上评估...")
    obs_dict, preds_dict = eval_model(
        model, test_loader, target_type
    )
    
    # 反归一化 + 计算每个流域 NSE
    # 只处理实际有预测结果的流域（可能在创建索引表时某些流域被跳过了）
    available_basins = set(preds_dict.keys())
    
    # 同时准备一个反归一化后的结果，用于画图和计算NSE
    denorm_obs = {}
    denorm_preds = {}
    basin_nse = {}  # 流域到NSE的映射
    evaluated_basins = []  # 实际评估的流域列表
    
    for basin in chosen_basins:
        b = str(basin)
        
        # 跳过没有预测结果的流域
        if b not in available_basins:
            continue
        
        denorm_preds[b] = ds_test.local_denormalization(preds_dict[b], b)
        denorm_obs[b] = ds_test.local_denormalization(obs_dict[b], b)
        
        nse = he.nse(denorm_preds[b], denorm_obs[b])
        
        basin_nse[b] = nse
        evaluated_basins.append(b)
    
    print(f"\n测试集结果：")
    for b in evaluated_basins:
        print(f"流域 {b}: NSE = {basin_nse[b]:.4f}")
    if evaluated_basins:
        nse_test = np.mean(list(basin_nse.values()))
        print(f"平均 NSE: {nse_test:.4f}")
    else:
        nse_test = np.nan
        print("未找到有效评估结果")
    
    # ==================== 9. 可视化结果 ====================
    print("\n正在生成可视化图表...")
    
    # 日期范围：按测试集时间 + 序列长度来推一推
    # 注意：使用3小时分辨率（freq='3H'）
    start_date = pd.to_datetime(ds_test.dates[0], format="%Y-%m-%d") + pd.Timedelta(
        hours=(sequence_length - 1) * 3  # 3小时分辨率
    )
    
    target_label = _target_label(target_type)
    
    for b in evaluated_basins:
        basin = b
        
        of = denorm_obs[b]
        pf = denorm_preds[b]
        
        # 针对当前流域的序列长度生成时间轴
        # 使用3小时分辨率（freq='3H'）
        date_range = pd.date_range(start_date, periods=len(of), freq='3H')
        
        fig, axes = plt.subplots(2, 1, figsize=(14, 8))
        
        # 预测图
        axes[0].plot(date_range, of, label="观测值", alpha=0.7)
        axes[0].plot(date_range, pf, label="预测值", alpha=0.7)
        axes[0].legend()
        axes[0].set_title(f"流域 {basin} - {target_label}预测 (测试集 NSE: {basin_nse[b]:.3f})")
        axes[0].set_ylabel(target_label)
        axes[0].grid(True, alpha=0.3)
        
        # 散点图
        axes[1].scatter(of, pf, alpha=0.6, s=8)
        min_val = min(of.min(), pf.min())
        max_val = max(of.max(), pf.max())
        axes[1].plot([min_val, max_val], [min_val, max_val], "r--")
        axes[1].set_xlabel(f"观测{target_label}")
        axes[1].set_ylabel(f"预测{target_label}")
        axes[1].grid(True, alpha=0.3)
        
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        # 保存图片
        output_file = os.path.join(IMAGES_SAVE_PATH, f"single_task_{target_type}_results_basin_{basin}.png")
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"已保存图片: {output_file}")
        plt.close()  # 关闭图形，释放内存

    
    # 绘制训练曲线
    fig, axes = plt.subplots(1, 2, figsize=(14, 4))
    
    axes[0].plot(train_losses)
    axes[0].set_title("训练损失")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("损失")
    axes[0].grid(True, alpha=0.3)
    
    axes[1].plot(val_nses, label=f"验证集 NSE")
    axes[1].set_title(f"验证集 NSE ({target_label})")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("NSE")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    training_curve_file = os.path.join(IMAGES_SAVE_PATH, f"single_task_{target_type}_training_curves.png")
    plt.savefig(training_curve_file, dpi=300, bbox_inches='tight')
    print(f"已保存训练曲线: {training_curve_file}")
    plt.close()  # 关闭图形，释放内存
    
    # ==================== 10. 保存模型 ====================
    print("\n正在保存模型...")
    model_path = os.path.join(MODEL_SAVE_PATH, f"single_task_{target_type}_model.pth")
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'means': means,
        'stds': stds,
        'target_type': target_type,
        'test_nse': nse_test,
    }, model_path)
    print(f"模型已保存: {model_path}")
    
    return model, means, stds, nse_test


if __name__ == "__main__":
    set_random_seed(1234)
    configure_chinese_font()
    
    # 打印设备信息
    print_device_info()
    
    # 训练流量预测模型
    #print("训练流量预测模型...")
    #flow_model, flow_means, flow_stds, flow_nse = train_single_task_model("flow")
    
    # 训练水位预测模型
    print("\n训练水位预测模型...")
    waterlevel_model, wl_means, wl_stds, wl_nse = train_single_task_model("waterlevel")
    
    print(f"\n=== 单任务模型训练完成 ===")
    #print(f"流量模型NSE: {flow_nse:.4f}")
    print(f"水位模型NSE: {wl_nse:.4f}")