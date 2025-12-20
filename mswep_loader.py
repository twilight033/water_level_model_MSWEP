"""
MSWEP降雨数据加载工具

从本地CSV文件加载MSWEP 3小时分辨率降雨数据，并提供与xarray Dataset兼容的接口
"""

import pandas as pd
import numpy as np
import xarray as xr
from typing import List, Optional, Union


def load_mswep_data(
    file_path: str,
    basin_ids: List[str],
    time_range: Optional[List[str]] = None
) -> pd.DataFrame:
    """
    从CSV文件加载MSWEP降雨数据
    
    Parameters
    ----------
    file_path : str
        MSWEP数据文件路径 (CSV格式)
    basin_ids : List[str]
        需要加载的流域ID列表
    time_range : List[str], optional
        时间范围 [start_date, end_date]，格式如 ["2015-01-01", "2024-12-02"]
        如果为None，则加载所有时间
    
    Returns
    -------
    pd.DataFrame
        降雨数据，index为时间，columns为流域ID
        单位：mm/3h (根据MSWEP数据集单位)
    """
    print(f"\n正在从 {file_path} 加载MSWEP降雨数据...")
    
    # 读取CSV文件
    df = pd.read_csv(file_path, index_col=0, parse_dates=True)
    
    print(f"  原始数据形状: {df.shape}")
    print(f"  时间范围: {df.index.min()} 到 {df.index.max()}")
    print(f"  流域数量: {len(df.columns)}")
    
    # 确保basin_ids都是字符串
    basin_ids = [str(b) for b in basin_ids]
    
    # 筛选指定流域
    available_basins = [b for b in basin_ids if b in df.columns]
    missing_basins = [b for b in basin_ids if b not in df.columns]
    
    if missing_basins:
        print(f"  警告: 以下流域在MSWEP数据中不存在: {missing_basins[:10]}")
    
    if not available_basins:
        raise ValueError(f"所有请求的流域都不在MSWEP数据中！")
    
    df_selected = df[available_basins]
    print(f"  筛选后流域数量: {len(available_basins)}")
    
    # 筛选时间范围
    if time_range is not None:
        start_date, end_date = time_range
        df_selected = df_selected.loc[start_date:end_date]
        print(f"  筛选后时间范围: {df_selected.index.min()} 到 {df_selected.index.max()}")
        print(f"  筛选后数据形状: {df_selected.shape}")
    
    # 检查数据有效性
    nan_count = df_selected.isnull().sum().sum()
    total_values = df_selected.size
    if nan_count > 0:
        print(f"  数据中包含 {nan_count}/{total_values} ({nan_count/total_values:.2%}) 个NaN值")
    
    return df_selected


def resample_to_3hourly(ds: xr.Dataset, method: str = 'mean') -> xr.Dataset:
    """
    将xarray Dataset重采样到3小时分辨率
    
    Parameters
    ----------
    ds : xr.Dataset
        原始数据集（可能是小时或日分辨率）
    method : str
        重采样方法，'mean'（平均值）或 'sum'（求和）
    
    Returns
    -------
    xr.Dataset
        重采样后的3小时分辨率数据集
    """
    print(f"\n将数据重采样到3小时分辨率 (方法: {method})...")
    
    # 检查原始时间分辨率
    time_diff = pd.Series(ds.time.values).diff().dropna()
    avg_diff = time_diff.mean()
    print(f"  原始时间分辨率: {avg_diff}")
    
    # 重采样到3小时
    if method == 'mean':
        ds_resampled = ds.resample(time='3H').mean()
    elif method == 'sum':
        ds_resampled = ds.resample(time='3H').sum()
    else:
        raise ValueError(f"不支持的重采样方法: {method}")
    
    print(f"  重采样前数据点数: {len(ds.time)}")
    print(f"  重采样后数据点数: {len(ds_resampled.time)}")
    
    return ds_resampled


def merge_forcing_with_mswep(
    forcing_ds: xr.Dataset,
    mswep_df: pd.DataFrame,
    precip_var_name: str = 'precipitation'
) -> xr.Dataset:
    """
    将MSWEP降雨数据与其他气象变量合并
    
    Parameters
    ----------
    forcing_ds : xr.Dataset
        不含降雨的气象强迫数据（或需要替换降雨的数据）
    mswep_df : pd.DataFrame
        MSWEP降雨数据，index为时间，columns为流域ID
    precip_var_name : str
        降雨变量名称，默认为 'precipitation'
    
    Returns
    -------
    xr.Dataset
        合并后的完整强迫数据，包含MSWEP降雨
    """
    print(f"\n正在合并MSWEP降雨数据与其他气象变量...")
    
    # 确保forcing_ds也是3小时分辨率
    # 检查时间分辨率
    forcing_time_diff = pd.Series(forcing_ds.time.values).diff().dropna()
    avg_forcing_diff = forcing_time_diff.mean()
    
    # 如果forcing_ds不是3小时分辨率，需要重采样
    if avg_forcing_diff != pd.Timedelta('3h'):
        print(f"  检测到其他变量时间分辨率为 {avg_forcing_diff}，需要重采样到3小时")
        forcing_ds = resample_to_3hourly(forcing_ds, method='mean')
    
    # 将pandas DataFrame转换为xarray DataArray
    # 注意：mswep_df的columns是流域ID，index是时间
    # 需要转换为 (time, basin) 的格式
    
    # 获取流域列表（确保顺序与forcing_ds一致）
    if 'basin' in forcing_ds.dims:
        basin_order = forcing_ds.basin.values
    else:
        basin_order = mswep_df.columns.values
    
    # 确保mswep_df包含所有需要的流域
    available_basins = [b for b in basin_order if str(b) in mswep_df.columns]
    
    if len(available_basins) < len(basin_order):
        missing = [b for b in basin_order if str(b) not in mswep_df.columns]
        print(f"  警告: {len(missing)} 个流域在MSWEP数据中缺失，将用NaN填充")
    
    # 重新索引mswep_df以匹配basin_order
    basin_order_str = [str(b) for b in basin_order]
    mswep_df_reindexed = mswep_df.reindex(columns=basin_order_str)
    
    # 关键修复：去除mswep_df中的重复时间索引（如果有）
    if mswep_df_reindexed.index.duplicated().any():
        print(f"  警告: MSWEP数据存在重复时间索引，将保留第一个")
        mswep_df_reindexed = mswep_df_reindexed[~mswep_df_reindexed.index.duplicated(keep='first')]
    
    # 时间对齐 - 关键修复：去除重复的时间索引
    time_range = pd.DatetimeIndex(forcing_ds.time.values).unique()  # 去重
    
    # 找到mswep_df和time_range的交集，并去重
    mswep_times = pd.DatetimeIndex(mswep_df_reindexed.index).unique()
    common_times = mswep_times.intersection(time_range)
    
    print(f"  forcing_ds时间维度: {len(forcing_ds.time)}")
    print(f"  重采样后时间维度(去重): {len(time_range)}")
    print(f"  MSWEP时间维度(去重): {len(mswep_times)}")
    print(f"  共同时间点: {len(common_times)}")
    
    # 对齐数据
    if len(common_times) < len(time_range) * 0.9:  # 如果共同时间点少于90%
        print(f"  警告: 只有 {len(common_times)/len(time_range):.1%} 的时间点匹配")
    
    # 对forcing_ds和mswep_df都进行时间对齐
    forcing_ds = forcing_ds.sel(time=common_times)
    
    # 关键修复：确保索引common_times在mswep_df_reindexed中存在
    # 使用reindex而不是loc，这样可以处理不存在的索引
    mswep_df_aligned = mswep_df_reindexed.reindex(common_times)
    
    print(f"  对齐后的MSWEP数据形状: {mswep_df_aligned.shape}")
    print(f"  对齐后的forcing_ds时间维度: {len(forcing_ds.time)}")
    
    # 确保形状一致
    assert mswep_df_aligned.shape[0] == len(forcing_ds.time), \
        f"时间维度不匹配: MSWEP={mswep_df_aligned.shape[0]}, forcing_ds={len(forcing_ds.time)}"
    
    # 转换为xarray DataArray
    precip_da = xr.DataArray(
        mswep_df_aligned.values,
        coords={
            'time': forcing_ds.time,
            'basin': basin_order
        },
        dims=['time', 'basin'],
        name=precip_var_name
    )
    
    # 如果forcing_ds中已经有降雨变量，先删除
    if precip_var_name in forcing_ds.data_vars:
        print(f"  删除原有的 {precip_var_name} 变量")
        forcing_ds = forcing_ds.drop_vars([precip_var_name])
    
    # 添加MSWEP降雨到数据集
    forcing_ds[precip_var_name] = precip_da
    
    print(f"  合并完成！最终数据集变量: {list(forcing_ds.data_vars.keys())}")
    print(f"  最终数据集维度: {dict(forcing_ds.dims)}")
    
    # 检查NaN情况
    for var in forcing_ds.data_vars:
        nan_count = int(forcing_ds[var].isnull().sum().values)
        total = forcing_ds[var].size
        if nan_count > 0:
            print(f"  {var}: {nan_count}/{total} ({nan_count/total:.2%}) NaN值")
    
    return forcing_ds


def convert_mswep_to_xarray(
    mswep_df: pd.DataFrame,
    var_name: str = 'precipitation'
) -> xr.Dataset:
    """
    将MSWEP DataFrame转换为xarray Dataset格式
    
    Parameters
    ----------
    mswep_df : pd.DataFrame
        MSWEP降雨数据，index为时间，columns为流域ID
    var_name : str
        变量名称
    
    Returns
    -------
    xr.Dataset
        xarray格式的数据集
    """
    # 转换列名为basin维度
    basin_ids = mswep_df.columns.values
    time_index = mswep_df.index.values
    
    # 创建DataArray
    da = xr.DataArray(
        mswep_df.values,
        coords={
            'time': time_index,
            'basin': basin_ids
        },
        dims=['time', 'basin'],
        name=var_name
    )
    
    # 转换为Dataset
    ds = da.to_dataset()
    
    return ds


if __name__ == "__main__":
    # 测试代码
    print("测试MSWEP数据加载...")
    
    # 测试加载数据
    test_basins = ['01017000', '01017060', '01017290']
    test_time_range = ['2015-01-01', '2020-12-31']
    
    try:
        mswep_data = load_mswep_data(
            file_path="MSWEP/mswep_basin_mean_3hourly_2015_2024.csv",
            basin_ids=test_basins,
            time_range=test_time_range
        )
        
        print(f"\n测试成功！")
        print(f"加载的数据形状: {mswep_data.shape}")
        print(f"数据样本:\n{mswep_data.head()}")
        
        # 测试转换为xarray
        mswep_ds = convert_mswep_to_xarray(mswep_data)
        print(f"\n转换为xarray格式:")
        print(mswep_ds)
        
    except Exception as e:
        print(f"\n测试失败: {e}")
        import traceback
        traceback.print_exc()

