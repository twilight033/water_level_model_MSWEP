#!/usr/bin/env python3
"""
改进的CAMELSH数据读取器

整合了以下功能：
1. 直接从原始NC文件读取数据（适用于小规模数据）
2. 使用hydrodataset的批处理机制（适用于大规模数据）
3. 自动选择最优的读取策略
4. 与现有代码兼容的接口
"""

import os
import glob
import numpy as np
import pandas as pd
import xarray as xr
from pathlib import Path
from typing import List, Optional, Union, Dict, Any
from hydrodataset.camelsh import Camelsh
from hydrodataset import StandardVariable

class ImprovedCAMELSHReader:
    """改进的CAMELSH数据读取器"""
    
    def __init__(self, data_path: str, download: bool = False, use_batch: bool = True):
        """
        初始化读取器
        
        Args:
            data_path: CAMELSH数据路径
            download: 是否允许下载数据
            use_batch: 是否优先使用批处理机制
        """
        self.data_path = Path(data_path)
        self.use_batch = use_batch
        self.download = download
        
        # 初始化hydrodataset读取器
        self.camelsh = Camelsh(data_path, download=download)
        
        # 原始NC文件路径
        self.timeseries_dir = self.data_path / "CAMELSH" / "timeseries" / "Data" / "CAMELSH" / "timeseries"
        # Hourly2文件路径（包含水位数据）
        self.hourly2_dir = self.data_path / "CAMELSH" / "Hourly2" / "Hourly2"
        
        # 变量映射
        self.variable_mapping = {
            # 气象强迫变量
            'Rainf': StandardVariable.PRECIPITATION,
            'Tair': StandardVariable.TEMPERATURE_MEAN,
            'SWdown': StandardVariable.SOLAR_RADIATION,
            'PotEvap': StandardVariable.POTENTIAL_EVAPOTRANSPIRATION,
            'Qair': 'specific_humidity',
            'Wind_E': 'wind_east',
            'Wind_N': 'wind_north',
            'LWdown': 'longwave_radiation',
            'PSurf': 'surface_pressure',
            'CAPE': 'convective_energy',
            'CRainf_frac': 'convective_rain_fraction',
            
            # 水文变量
            'Streamflow': StandardVariable.STREAMFLOW,
            'streamflow': StandardVariable.STREAMFLOW,  # Hourly2文件中的变量名
            'water_level': StandardVariable.WATER_LEVEL,  # Hourly2文件中的水位变量
        }
        
        # 反向映射
        self.reverse_mapping = {}
        for nc_var, std_var in self.variable_mapping.items():
            if hasattr(std_var, 'value'):
                self.reverse_mapping[std_var.value] = nc_var
            else:
                self.reverse_mapping[str(std_var)] = nc_var
        
        # 手动添加StandardVariable的映射
        self.reverse_mapping[StandardVariable.STREAMFLOW] = 'Streamflow'  # timeseries文件中的变量名
        self.reverse_mapping[StandardVariable.WATER_LEVEL] = 'water_level'  # Hourly2文件中的变量名
    
    def read_object_ids(self) -> np.ndarray:
        """获取所有可用的流域ID"""
        try:
            # 优先使用hydrodataset接口
            return self.camelsh.read_object_ids()
        except Exception:
            # 备用方案：直接从NC文件获取
            if self.timeseries_dir.exists():
                nc_files = list(self.timeseries_dir.glob("*.nc"))
                basin_ids = [f.stem for f in nc_files]
                return np.array(sorted(basin_ids))
            else:
                raise FileNotFoundError("无法找到流域数据")
    
    def _check_batch_availability(self, basin_ids: List[str]) -> bool:
        """检查batch文件是否包含所需的流域"""
        batch_pattern = str(self.camelsh.cache_dir / "batch*_camelsh_timeseries.nc")
        batch_files = sorted(glob.glob(batch_pattern))
        
        if not batch_files:
            return False
        
        # 检查第一个batch文件中的流域
        try:
            ds = xr.open_dataset(batch_files[0])
            batch_basins = [str(b) for b in ds.basin.values]
            ds.close()
            
            # 检查是否有交集
            common_basins = set(basin_ids) & set(batch_basins)
            return len(common_basins) > 0
            
        except Exception:
            return False
    
    def _read_from_batch(self, gage_id_lst: List[str], t_range: List[str], 
                        var_lst: List[Union[str, StandardVariable]]) -> Optional[xr.Dataset]:
        """使用hydrodataset的批处理机制读取数据"""
        try:
            return self.camelsh.read_ts_xrdataset(
                gage_id_lst=gage_id_lst,
                t_range=t_range,
                var_lst=var_lst
            )
        except Exception as e:
            print(f"批处理读取失败: {e}")
            return None
    
    def _read_from_nc_files(self, gage_id_lst: List[str], t_range: List[str], 
                           var_lst: List[Union[str, StandardVariable]]) -> Optional[xr.Dataset]:
        """直接从NC文件读取数据"""
        try:
            # 转换变量名
            nc_vars = []
            for var in var_lst:
                if hasattr(var, 'value'):
                    std_name = var.value
                else:
                    std_name = str(var)
                
                if std_name in self.reverse_mapping:
                    nc_vars.append(self.reverse_mapping[std_name])
                else:
                    print(f"警告: 未找到变量 {std_name} 的映射")
            
            if not nc_vars:
                raise ValueError("没有可用的变量")
            
            # 读取每个流域的数据
            datasets = {}
            for basin_id in gage_id_lst:
                # 根据变量类型选择不同的数据源
                basin_data = {}
                
                # 检查是否需要水位数据
                need_water_level = any(
                    (hasattr(var, 'value') and var.value == 'water_level') or 
                    str(var) == 'water_level' 
                    for var in var_lst
                )
                
                # 从timeseries目录读取气象和径流数据
                timeseries_file = self.timeseries_dir / f"{basin_id}.nc"
                if timeseries_file.exists():
                    ts_ds = xr.open_dataset(timeseries_file)
                    basin_data['timeseries'] = ts_ds
                
                # 从Hourly2目录读取水位数据
                if need_water_level:
                    hourly2_file = self.hourly2_dir / f"{basin_id}_hourly.nc"
                    if hourly2_file.exists():
                        h2_ds = xr.open_dataset(hourly2_file)
                        basin_data['hourly2'] = h2_ds
                    else:
                        print(f"警告: 流域 {basin_id} 的Hourly2数据文件不存在")
                
                if not basin_data:
                    print(f"警告: 流域 {basin_id} 的数据文件不存在")
                    continue
                
                # 合并不同数据源的数据
                combined_ds = None
                
                for source_name, ds in basin_data.items():
                    # 时间筛选
                    if t_range:
                        start_time, end_time = t_range
                        if 'DateTime' in ds.coords:
                            ds_filtered = ds.sel(DateTime=slice(start_time, end_time))
                        elif 'time' in ds.coords:
                            ds_filtered = ds.sel(time=slice(start_time, end_time))
                        else:
                            print(f"警告: 数据集中未找到时间坐标")
                            ds_filtered = ds
                    else:
                        ds_filtered = ds
                    
                    # 变量筛选
                    available_vars = [v for v in nc_vars if v in ds_filtered.data_vars]
                    if available_vars:
                        ds_selected = ds_filtered[available_vars]
                        
                        # 统一时间坐标名称
                        if 'time' in ds_selected.coords and 'DateTime' not in ds_selected.coords:
                            ds_selected = ds_selected.rename({'time': 'DateTime'})
                        
                        if combined_ds is None:
                            combined_ds = ds_selected
                        else:
                            # 合并数据集
                            combined_ds = xr.merge([combined_ds, ds_selected])
                    
                    # 关闭原始数据集
                    ds.close()
                
                if combined_ds is not None:
                    datasets[basin_id] = combined_ds
                else:
                    print(f"警告: 流域 {basin_id} 中未找到所需变量")
            
            if not datasets:
                raise ValueError("未能读取任何流域的数据")
            
            # 合并数据集
            combined_data = {}
            
            for nc_var in nc_vars:
                data_arrays = []
                basin_list = []
                
                for basin_id, ds in datasets.items():
                    if nc_var in ds.data_vars:
                        data_arrays.append(ds[nc_var].values)
                        basin_list.append(basin_id)
                
                if data_arrays:
                    # 堆叠数据 (basin, time)
                    stacked_data = np.stack(data_arrays, axis=0)
                    
                    # 获取时间坐标（使用第一个数据集的时间）
                    time_coord = list(datasets.values())[0].DateTime
                    
                    # 创建DataArray
                    da = xr.DataArray(
                        stacked_data,
                        dims=['basin', 'time'],
                        coords={
                            'basin': basin_list,
                            'time': time_coord.values  # 使用.values避免坐标冲突
                        }
                    )
                    
                    # 使用标准变量名
                    std_name = None
                    for std_var, mapped_nc_var in self.reverse_mapping.items():
                        if mapped_nc_var == nc_var:
                            std_name = std_var
                            break
                    
                    if std_name:
                        combined_data[std_name] = da
            
            # 关闭所有数据集
            for ds in datasets.values():
                ds.close()
            
            if combined_data:
                return xr.Dataset(combined_data)
            else:
                raise ValueError("未能创建合并的数据集")
                
        except Exception as e:
            print(f"NC文件读取失败: {e}")
            return None
    
    def read_ts_xrdataset(self, gage_id_lst: List[str], t_range: List[str], 
                         var_lst: List[Union[str, StandardVariable]]) -> xr.Dataset:
        """
        读取时序数据（自动选择最优策略）
        
        Args:
            gage_id_lst: 流域ID列表
            t_range: 时间范围 [start, end]
            var_lst: 变量列表
            
        Returns:
            xr.Dataset: 时序数据集
        """
        print(f"读取 {len(gage_id_lst)} 个流域的时序数据...")
        print(f"时间范围: {t_range}")
        print(f"变量数量: {len(var_lst)}")
        
        # 策略1: 如果启用批处理且batch文件可用，优先使用批处理
        if self.use_batch:
            print("尝试使用批处理机制...")
            if self._check_batch_availability(gage_id_lst):
                dataset = self._read_from_batch(gage_id_lst, t_range, var_lst)
                if dataset is not None:
                    print("[成功] 批处理读取成功")
                    return dataset
                else:
                    print("[失败] 批处理读取失败，尝试直接读取NC文件")
            else:
                print("[警告] 批处理文件不包含所需流域，尝试直接读取NC文件")
        
        # 策略2: 直接从NC文件读取
        print("使用直接NC文件读取...")
        dataset = self._read_from_nc_files(gage_id_lst, t_range, var_lst)
        
        if dataset is not None:
            print("[成功] NC文件读取成功")
            return dataset
        else:
            raise ValueError("所有读取策略都失败了")
    
    def read_attr_xrdataset(self, gage_id_lst: List[str], 
                           var_lst: Optional[List[str]] = None) -> xr.Dataset:
        """读取属性数据（使用hydrodataset接口）"""
        return self.camelsh.read_attr_xrdataset(gage_id_lst=gage_id_lst, var_lst=var_lst)
    
    def get_data_summary(self) -> Dict[str, Any]:
        """获取数据集概要信息"""
        summary = {}
        
        # 流域信息
        basin_ids = self.read_object_ids()
        summary['total_basins'] = len(basin_ids)
        summary['basin_id_range'] = [basin_ids[0], basin_ids[-1]]
        
        # 批处理文件信息
        batch_pattern = str(self.camelsh.cache_dir / "batch*_camelsh_timeseries.nc")
        batch_files = sorted(glob.glob(batch_pattern))
        summary['batch_files'] = len(batch_files)
        
        # NC文件信息
        if self.timeseries_dir.exists():
            nc_files = list(self.timeseries_dir.glob("*.nc"))
            summary['nc_files'] = len(nc_files)
        else:
            summary['nc_files'] = 0
        
        # 可用变量
        summary['available_variables'] = list(self.variable_mapping.keys())
        
        return summary

def demonstrate_improved_reader():
    """演示改进的读取器"""
    
    print("=" * 80)
    print("改进的CAMELSH读取器演示")
    print("=" * 80)
    
    from config import CAMELSH_DATA_PATH
    
    # 创建读取器
    reader = ImprovedCAMELSHReader(CAMELSH_DATA_PATH, download=False, use_batch=True)
    
    # 获取数据概要
    print("\n1. 数据概要:")
    summary = reader.get_data_summary()
    for key, value in summary.items():
        print(f"  {key}: {value}")
    
    # 测试数据读取
    print("\n2. 测试数据读取:")
    
    # 获取测试流域（使用实际存在NC文件的流域）
    basin_ids = reader.read_object_ids()
    
    # 检查哪些流域有对应的NC文件
    available_basins = []
    for basin_id in basin_ids[:10]:  # 检查前10个
        nc_file = reader.timeseries_dir / f"{basin_id}.nc"
        if nc_file.exists():
            available_basins.append(basin_id)
        if len(available_basins) >= 3:  # 找到3个就够了
            break
    
    test_basins = available_basins[:3] if available_basins else basin_ids[:3].tolist()
    print(f"可用流域检查: 从前10个流域中找到 {len(available_basins)} 个有NC文件的流域")
    
    # 测试参数
    time_range = ["2010-01-01", "2010-01-07"]
    variables = [
        StandardVariable.PRECIPITATION,
        StandardVariable.TEMPERATURE_MEAN,
        StandardVariable.STREAMFLOW
    ]
    
    print(f"测试流域: {test_basins}")
    print(f"时间范围: {time_range}")
    print(f"变量: {[str(v) for v in variables]}")
    
    try:
        # 读取数据
        dataset = reader.read_ts_xrdataset(test_basins, time_range, variables)
        
        print(f"\n[成功] 数据读取成功!")
        print(f"数据维度: {dataset.dims}")
        print(f"变量列表: {list(dataset.data_vars.keys())}")
        
        # 显示数据统计
        for var_name in dataset.data_vars:
            data_array = dataset[var_name]
            print(f"  {var_name}: 形状={data_array.shape}")
            print(f"    数据范围: [{data_array.min().values:.3f}, {data_array.max().values:.3f}]")
        
        # 转换为pandas格式（兼容现有代码）
        print(f"\n3. 转换为pandas格式:")
        pandas_data = {}
        for var_name in dataset.data_vars:
            df = dataset[var_name].to_pandas().T  # 转置：时间为index，流域为columns
            pandas_data[var_name] = df
            print(f"  {var_name}: DataFrame形状={df.shape}")
        
        return reader, dataset, pandas_data
        
    except Exception as e:
        print(f"[失败] 数据读取失败: {e}")
        import traceback
        traceback.print_exc()
        return reader, None, None

if __name__ == "__main__":
    demonstrate_improved_reader()
