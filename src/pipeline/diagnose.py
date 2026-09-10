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

    from pipeline.dataset import PreparedData
    step("10 PreparedData(attr_set='base')",
         lambda: (PreparedData().input_size,))

    print("\n若以上全部完成，说明数据侧正常，问题在别处；"
          "若在某一步停住，那一行即为卡点。", flush=True)


if __name__ == "__main__":
    main()
