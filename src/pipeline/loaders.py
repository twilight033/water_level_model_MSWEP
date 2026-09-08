"""统一的缓存数据加载入口，供归一化、Dataset、评估共用。

所有数据都来自 data/camelsh_exported/ 下的 parquet 缓存，因此这些函数
不依赖 F 盘挂载状态，也不重复解析 NC / 1.4 GB 的 MSWEP CSV。
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

_ROOT = Path(__file__).resolve().parents[2]
for _p in (str(_ROOT / "src"), str(_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from pipeline.paths import FLOW_CACHE, WATERLEVEL_CACHE  # noqa: E402
from pipeline.export_forcing import MSWEP_CACHE, CAMELSH_FORCING_VARS, _cache_path  # noqa: E402

# 强迫变量顺序固定，模型输入通道顺序依赖它
FORCING_ORDER = ("precipitation",) + CAMELSH_FORCING_VARS
TASKS = ("flow", "waterlevel")


def load_forcing() -> tuple:
    """返回 (grid, forcing_array, basins)。

    forcing_array 形状 [n_basin, n_time, n_var]，float32，无 NaN。
    """
    frames = {"precipitation": pd.read_parquet(MSWEP_CACHE)}
    for var in CAMELSH_FORCING_VARS:
        frames[var] = pd.read_parquet(_cache_path(var))

    grid = frames["precipitation"].index
    for var in CAMELSH_FORCING_VARS:
        grid = grid.intersection(frames[var].index)
    grid = pd.DatetimeIndex(grid).sort_values()

    basins = list(frames["precipitation"].columns)
    stacked = np.stack(
        [frames[v].reindex(index=grid, columns=basins).to_numpy(dtype="float32")
         for v in FORCING_ORDER],
        axis=-1,
    )  # [n_time, n_basin, n_var]
    arr = np.ascontiguousarray(stacked.transpose(1, 0, 2))  # [n_basin, n_time, n_var]

    if np.isnan(arr).any():
        n_nan = int(np.isnan(arr).sum())
        raise ValueError(f"强迫数据含 {n_nan} 个 NaN，模型输入不允许缺失，请先检查缓存")
    return grid, arr, basins


def load_targets(grid: pd.DatetimeIndex, basins: list) -> dict:
    """返回 {task: [n_basin, n_time] float32 数组}，NaN 保留表示无观测。"""
    out = {}
    for task, cache in (("flow", FLOW_CACHE), ("waterlevel", WATERLEVEL_CACHE)):
        df = pd.read_parquet(cache).reindex(index=grid, columns=basins)
        out[task] = np.ascontiguousarray(df.to_numpy(dtype="float32").T)
    return out


def load_attributes(basins: list) -> tuple:
    """返回 (attr_array [n_basin, n_attr], 列名, one-hot 列名)。"""
    from pipeline.attributes import build_attributes

    df, onehot = build_attributes()
    df = df.reindex(basins)
    return df.to_numpy(dtype="float32"), list(df.columns), list(onehot)
