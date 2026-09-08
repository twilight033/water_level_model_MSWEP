"""把 86 个流域的小时级径流与水位导出为 parquet 缓存。

后续所有步骤（冻结划分、覆盖率统计、训练）都读这份缓存，
既避免重复解析 NC 文件，也降低对 F 盘挂载状态的依赖。
"""

import sys
from pathlib import Path

import pandas as pd

_ROOT = Path(__file__).resolve().parents[2]
for _p in (str(_ROOT / "src"), str(_ROOT), str(_ROOT / "src" / "others")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from pipeline.paths import (  # noqa: E402
    EXPORT_DIR, FLOW_CACHE, WATERLEVEL_CACHE, load_basin_ids, verify_camelsh_path,
)


def export_targets(overwrite: bool = False) -> tuple:
    """导出径流与水位缓存，返回 (flow_df, waterlevel_df)。"""
    from hydrodataset import StandardVariable
    from improved_camelsh_reader import ImprovedCAMELSHReader
    from config import CAMELSH_DATA_PATH

    if not overwrite and FLOW_CACHE.exists() and WATERLEVEL_CACHE.exists():
        print(f"缓存已存在，直接读取: {FLOW_CACHE.name} / {WATERLEVEL_CACHE.name}")
        return pd.read_parquet(FLOW_CACHE), pd.read_parquet(WATERLEVEL_CACHE)

    data_root = verify_camelsh_path(CAMELSH_DATA_PATH)
    basins = load_basin_ids()
    print(f"数据根目录: {data_root}")
    print(f"流域数量: {len(basins)}")

    reader = ImprovedCAMELSHReader(str(data_root), download=False, use_batch=True)
    t_range = reader.camelsh.default_t_range
    print(f"时间范围: {t_range}")

    frames = {}
    for label, std_var, cache in (
        ("径流", StandardVariable.STREAMFLOW, FLOW_CACHE),
        ("水位", StandardVariable.WATER_LEVEL, WATERLEVEL_CACHE),
    ):
        print(f"\n正在读取{label}数据...")
        ds = reader.read_ts_xrdataset(
            gage_id_lst=basins, t_range=t_range, var_lst=[std_var]
        )
        # 显式指定维度顺序，避免依赖隐含的 dim 排列
        df = ds[std_var].transpose("time", "basin").to_pandas()
        df.columns = [str(c) for c in df.columns]
        df.index = pd.DatetimeIndex(df.index)
        df.index.name = "time"
        # 只保留请求的流域，缺失的流域整列为 NaN，便于覆盖率统计如实反映
        df = df.reindex(columns=basins)
        df = df.astype("float32")

        EXPORT_DIR.mkdir(parents=True, exist_ok=True)
        df.to_parquet(cache, compression="zstd")
        valid = int(df.notna().sum().sum())
        print(f"  形状: {df.shape}，有效点: {valid} ({valid / df.size:.1%})")
        print(f"  已写入: {cache}")
        frames[label] = df

    return frames["径流"], frames["水位"]


if __name__ == "__main__":
    export_targets(overwrite="--overwrite" in sys.argv)
