"""逐步诊断：定位测试或训练卡在哪一步。

用法（项目根目录）：

    python -X utf8 -u src/pipeline/diagnose.py

每一步都立刻 flush 输出，因此"最后打印的那一行"就是卡住的位置。所有路径探测
都带超时（见 pipeline.paths.path_reachable），不会像 ``Path.exists()`` 那样在
未挂载的映射盘上无限期阻塞。
"""

import sys
import time
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
for _p in (str(_ROOT / "src"), str(_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from pipeline.paths import (  # noqa: E402
    BASIN_SPLITS_FILE, EXPORT_DIR, NORM_STATS_FILE, path_reachable,
)


def step(label, fn):
    print(f"[{label}] 开始 ...", flush=True)
    started = time.time()
    try:
        value = fn()
    except Exception as exc:                       # noqa: BLE001 - 诊断需捕获一切
        print(f"[{label}] 失败 {time.time() - started:.1f}s  "
              f"{type(exc).__name__}: {exc}", flush=True)
        return None
    print(f"[{label}] 完成 {time.time() - started:.1f}s  -> {value}", flush=True)
    return value


def compare_cache_against_rebuild():
    """磁盘上的属性缓存与"直读 CSV 重建"是否一致，不一致时打印具体差异。

    缓存可能是早期用 hydrodataset 建的，与现在直读 CSV 的结果未必逐值相同
    （例如分类属性的类别码编号不同会让 one-hot 列名不同）。这里如实报告，
    因为第一轮 264 次实验用的正是磁盘上那份缓存。
    """
    import numpy as np
    import pandas as pd

    from pipeline.attribute_sources import build_attribute_table, cache_path
    from pipeline.paths import load_basin_ids

    path = cache_path("base")
    if not path.exists():
        return "基础缓存不存在，跳过比对"

    cached = pd.read_parquet(path)
    rebuilt, _ = build_attribute_table("base", load_basin_ids())
    rebuilt = rebuilt.reindex(cached.index)

    if list(cached.columns) != list(rebuilt.columns):
        only_cached = [c for c in cached.columns if c not in rebuilt.columns]
        only_rebuilt = [c for c in rebuilt.columns if c not in cached.columns]
        print(f"    列不一致\n"
              f"      缓存列 : {list(cached.columns)}\n"
              f"      重建列 : {list(rebuilt.columns)}\n"
              f"      仅缓存有: {only_cached}\n"
              f"      仅重建有: {only_rebuilt}", flush=True)
        return "列不一致（详见上方）"

    diffs = []
    for col in cached.columns:
        a = cached[col].to_numpy(dtype=float)
        b = rebuilt[col].to_numpy(dtype=float)
        if not np.allclose(a, b, rtol=1e-6, atol=1e-6, equal_nan=True):
            worst = int(np.nanargmax(np.abs(a - b)))
            diffs.append(f"{col}(最大差 {np.nanmax(np.abs(a - b)):.4g} @ "
                         f"{cached.index[worst]}: 缓存 {a[worst]:.6g} / 重建 {b[worst]:.6g})")
    if diffs:
        for line in diffs:
            print(f"    {line}", flush=True)
        return f"{len(diffs)}/{len(cached.columns)} 列取值不一致"
    return "完全一致"


def main():
    print(f"项目根目录: {_ROOT}", flush=True)

    step("1 缓存目录", lambda: EXPORT_DIR.is_dir())
    step("2 缓存文件清单",
         lambda: sorted(p.name for p in EXPORT_DIR.glob("*.parquet")))
    step("3 冻结划分", lambda: BASIN_SPLITS_FILE.exists())
    step("4 归一化统计量", lambda: NORM_STATS_FILE.exists())

    import pandas as pd
    for name in ("attributes_86.parquet", "attributes_86_extended.parquet"):
        path = EXPORT_DIR / name
        if path.exists():
            step(f"5 读 {name}", lambda p=path: pd.read_parquet(p).shape)
        else:
            print(f"[5 读 {name}] 跳过：文件不存在（需运行 "
                  f"src/pipeline/attribute_sources.py 重建）", flush=True)

    from config import CAMELSH_DATA_PATH, MSWEP_CSV_PATH
    print(f"[6] config.CAMELSH_DATA_PATH = {CAMELSH_DATA_PATH}", flush=True)
    print(f"[6] config.MSWEP_CSV_PATH    = {MSWEP_CSV_PATH}", flush=True)

    def probe():
        reachable = path_reachable(CAMELSH_DATA_PATH)
        return {None: "超时——很可能是未挂载的映射盘或断开的网络路径",
                True: "可达", False: "不存在"}[reachable]

    step("7 CAMELSH 路径可达性（带 5 秒超时）", probe)

    from pipeline.attribute_sources import get_attribute_table
    step("8 get_attribute_table('base')",
         lambda: get_attribute_table("base")[0].shape)
    step("9 get_attribute_table('extended')",
         lambda: get_attribute_table("extended")[0].shape)

    step("10 缓存与直读结果比对", compare_cache_against_rebuild)

    from pipeline.dataset import PreparedData
    step("11 PreparedData(attr_set='base')",
         lambda: (PreparedData().input_size,))

    print("\n若以上全部完成，说明数据侧正常，问题在别处；"
          "若在某一步停住，那一行即为卡点。", flush=True)


if __name__ == "__main__":
    main()
