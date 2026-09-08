"""把 86 个流域的 3 小时气象强迫导出为 parquet 缓存。

降水来自 MSWEP（原生 3 小时累计，向前时间戳），其余变量来自 CAMELSH
小时数据并重采样为 3 小时均值（xarray 默认 label='left'，即 t 代表
[t, t+3h)）。聚合约定详见 docs/temporal_alignment.md。

MSWEP 原始 CSV 有 1.4 GB，每次训练重新解析代价太高，故一次性缓存。
"""

import sys
from pathlib import Path

import pandas as pd

_ROOT = Path(__file__).resolve().parents[2]
for _p in (str(_ROOT / "src"), str(_ROOT), str(_ROOT / "src" / "others")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from pipeline.paths import (  # noqa: E402
    EXPORT_DIR, load_basin_ids, resolve_mswep_csv, verify_camelsh_path,
)

# MSWEP CSV 的位置来自 config.MSWEP_CSV_PATH（可由同名环境变量覆盖），
# 不在此硬编码；只有重建缓存时才需要它
MSWEP_CACHE = EXPORT_DIR / "mswep_precip_3h_86.parquet"

# 从 CAMELSH 读取并重采样到 3 小时的变量（降水不在此列，来自 MSWEP）
CAMELSH_FORCING_VARS = ("temperature_mean", "solar_radiation")


def _cache_path(var_name: str) -> Path:
    return EXPORT_DIR / f"forcing_{var_name}_3h_86.parquet"


def export_mswep_precip(overwrite: bool = False) -> pd.DataFrame:
    """从 MSWEP CSV 中抽取 86 个流域的 3 小时降水并缓存。"""
    if not overwrite and MSWEP_CACHE.exists():
        print(f"MSWEP 缓存已存在: {MSWEP_CACHE.name}")
        return pd.read_parquet(MSWEP_CACHE)

    basins = load_basin_ids()
    csv_path = resolve_mswep_csv()
    print(f"解析 MSWEP CSV: {csv_path}")
    print(f"  文件大小 {csv_path.stat().st_size / 1e9:.2f} GB，仅取 {len(basins)} 个流域列...")
    df = pd.read_csv(
        csv_path,
        usecols=["time"] + basins,
        dtype={b: "float32" for b in basins},
        parse_dates=["time"],
    )
    df = df.set_index("time").sort_index()
    if df.index.duplicated().any():
        n_dup = int(df.index.duplicated().sum())
        print(f"  警告: 存在 {n_dup} 个重复时间戳，保留第一个")
        df = df[~df.index.duplicated(keep="first")]
    df = df.reindex(columns=basins)

    EXPORT_DIR.mkdir(parents=True, exist_ok=True)
    df.to_parquet(MSWEP_CACHE, compression="zstd")
    valid = int(df.notna().sum().sum())
    print(f"  形状: {df.shape}  时间: {df.index[0]} → {df.index[-1]}")
    print(f"  有效点: {valid} ({valid / df.size:.1%})")
    print(f"  已写入: {MSWEP_CACHE}")
    return df


def export_camelsh_forcing(overwrite: bool = False) -> dict:
    """读取 CAMELSH 小时气象变量，重采样为 3 小时均值后缓存。"""
    targets = {v: _cache_path(v) for v in CAMELSH_FORCING_VARS}
    if not overwrite and all(p.exists() for p in targets.values()):
        print("CAMELSH 强迫缓存已存在，直接读取")
        return {v: pd.read_parquet(p) for v, p in targets.items()}

    from hydrodataset import StandardVariable
    from improved_camelsh_reader import ImprovedCAMELSHReader
    from config import CAMELSH_DATA_PATH

    data_root = verify_camelsh_path(CAMELSH_DATA_PATH)
    basins = load_basin_ids()
    reader = ImprovedCAMELSHReader(str(data_root), download=False, use_batch=True)
    t_range = reader.camelsh.default_t_range

    std_vars = [getattr(StandardVariable, v.upper()) for v in CAMELSH_FORCING_VARS]
    print(f"\n读取 CAMELSH 小时气象变量: {list(CAMELSH_FORCING_VARS)}")
    ds = reader.read_ts_xrdataset(gage_id_lst=basins, t_range=t_range, var_lst=std_vars)

    # 小时 → 3 小时均值；xarray 默认 label='left'，t 代表 [t, t+3h)
    print("重采样到 3 小时均值（label='left'）...")
    ds_3h = ds.resample(time="3h").mean()

    EXPORT_DIR.mkdir(parents=True, exist_ok=True)
    out = {}
    for var_name, std_var in zip(CAMELSH_FORCING_VARS, std_vars):
        # 显式指定维度顺序，不能依赖 resample 之后的隐含 dim 顺序
        df = ds_3h[std_var].transpose("time", "basin").to_pandas()
        df.columns = [str(c) for c in df.columns]
        df.index = pd.DatetimeIndex(df.index)
        df.index.name = "time"
        df = df.reindex(columns=basins).astype("float32")
        cache = targets[var_name]
        df.to_parquet(cache, compression="zstd")
        valid = int(df.notna().sum().sum())
        print(f"  {var_name}: 形状 {df.shape}，有效点 {valid} ({valid / df.size:.1%}) → {cache.name}")
        out[var_name] = df
    return out


if __name__ == "__main__":
    overwrite = "--overwrite" in sys.argv
    export_mswep_precip(overwrite=overwrite)
    export_camelsh_forcing(overwrite=overwrite)
