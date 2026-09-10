"""项目路径常量与原始数据可用性校验。

原始数据路径全部来自 config.py（可由同名环境变量覆盖），本模块不硬编码
任何机器相关的路径。原始数据常放在可移动盘上，批量实验前先跑一次
preflight 确认路径可用，避免跑到一半才发现读不到文件：

    python -X utf8 src/pipeline/paths.py            # 含原始数据
    python -X utf8 src/pipeline/paths.py --no-raw   # 只查缓存与冻结划分
"""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

# 冻结划分、归一化统计量等"一次生成、长期复用"的产物
SPLITS_DIR = PROJECT_ROOT / "data" / "splits"
BASIN_SPLITS_FILE = SPLITS_DIR / "basin_splits.json"
NORM_STATS_FILE = SPLITS_DIR / "norm_stats.json"

# 覆盖率统计
COVERAGE_DIR = PROJECT_ROOT / "data" / "coverage"

# 目标变量缓存（避免每次重复解析 NC 文件）
EXPORT_DIR = PROJECT_ROOT / "data" / "camelsh_exported"
FLOW_CACHE = EXPORT_DIR / "flow_hourly_86.parquet"
WATERLEVEL_CACHE = EXPORT_DIR / "waterlevel_hourly_86.parquet"

# 实验输出根目录
RESULTS_ROOT = PROJECT_ROOT / "results"

# CAMELSH 数据集内部必须存在的子目录
_REQUIRED_SUBDIRS = (
    Path("CAMELSH") / "timeseries" / "Data" / "CAMELSH" / "timeseries",
    Path("CAMELSH") / "Hourly2" / "Hourly2",
    Path("CAMELSH") / "attributes",
)


def path_reachable(path, timeout_sec: float = 5.0):
    """带超时的路径可达性探测。

    返回 True/False 表示存在与否，返回 None 表示**超时**。

    为什么需要它：未挂载的映射盘或断开的网络路径上，``os.stat`` 可能阻塞几十秒
    甚至无限期，而不是立刻返回 False。直接调 ``Path.exists()`` 会让整个测试或
    批量训练静默挂死——实测中一次 ``unittest`` 就卡在属性缓存那条用例上没有
    任何输出。放到后台线程里探测，超时即判定为不可达并给出明确提示。
    """
    import threading

    result = {}

    def probe():
        try:
            result["value"] = Path(path).exists()
        except OSError as exc:
            result["error"] = exc

    worker = threading.Thread(target=probe, daemon=True)
    worker.start()
    worker.join(timeout_sec)
    if worker.is_alive():
        return None
    return result.get("value", False)


def verify_camelsh_path(data_path) -> Path:
    """校验 CAMELSH 数据根目录，缺任何一个子目录都立即报错。

    Parameters
    ----------
    data_path : str | Path
        config.CAMELSH_DATA_PATH，即包含 CAMELSH/ 的父目录。

    Returns
    -------
    Path
        校验通过的绝对路径。
    """
    root = Path(data_path)
    reachable = path_reachable(root)
    if reachable is None:
        raise FileNotFoundError(
            f"探测 CAMELSH 数据根目录超时: {root}\n"
            f"该路径很可能指向未挂载的映射盘或断开的网络位置。挂载后重试，"
            f"或改用本机实际路径：\n"
            f'    $env:CAMELSH_DATA_PATH = "<该机器上的实际路径>"'
        )
    if not reachable:
        raise FileNotFoundError(
            f"CAMELSH 数据根目录不存在: {root}\n"
            f"若数据在可移动盘上请先确认已挂载；否则修改 config.py 的 "
            f"CAMELSH_DATA_PATH，或设置同名环境变量：\n"
            f'    $env:CAMELSH_DATA_PATH = "<该机器上的实际路径>"'
        )
    missing = [str(sub) for sub in _REQUIRED_SUBDIRS if not (root / sub).is_dir()]
    if missing:
        raise FileNotFoundError(
            f"CAMELSH 数据目录结构不完整: {root}\n"
            f"缺少子目录: {missing}"
        )
    return root.resolve()


def preflight(require_raw: bool = True) -> bool:
    """启动前自检：打印并校验全部路径，缺什么一次说清。

    require_raw=True 时同时校验原始数据（重建缓存前用）；
    False 时只校验缓存与冻结划分（直接开跑训练时用）。
    """
    import sys as _sys

    ok = True
    print(f"项目根目录: {PROJECT_ROOT}")

    if require_raw:
        from config import CAMELSH_DATA_PATH, MSWEP_CSV_PATH
        print(f"config.CAMELSH_DATA_PATH = {CAMELSH_DATA_PATH}")
        print(f"config.MSWEP_CSV_PATH    = {MSWEP_CSV_PATH}")
        for label, fn in (("CAMELSH 数据集", lambda: verify_camelsh_path(CAMELSH_DATA_PATH)),
                          ("MSWEP 降水 CSV", resolve_mswep_csv)):
            try:
                print(f"  [OK] {label}: {fn()}")
            except FileNotFoundError as exc:
                ok = False
                print(f"  [缺失] {label}\n{exc}", file=_sys.stderr)

    print("缓存与冻结划分:")
    for label, path in (("目标 Q", FLOW_CACHE), ("目标 h", WATERLEVEL_CACHE),
                        ("冻结划分", BASIN_SPLITS_FILE), ("归一化统计量", NORM_STATS_FILE)):
        mark = "OK" if path.exists() else "缺失"
        if not path.exists() and not require_raw:
            ok = False
        print(f"  [{mark}] {label}: {path}")
    return ok


def resolve_mswep_csv(csv_path=None) -> Path:
    """解析并校验 MSWEP CSV 路径。

    支持绝对路径，也支持相对于项目根目录的相对路径。路径本身来自
    config.MSWEP_CSV_PATH（可由同名环境变量覆盖），不在代码里硬编码。
    """
    if csv_path is None:
        from config import MSWEP_CSV_PATH
        csv_path = MSWEP_CSV_PATH

    path = Path(csv_path)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    if not path.is_file():
        raise FileNotFoundError(
            f"MSWEP 降水 CSV 不存在: {path}\n"
            f"请修改 config.py 的 MSWEP_CSV_PATH，或设置同名环境变量："
            f'\n    $env:MSWEP_CSV_PATH = "<该机器上的实际路径>"'
        )
    return path.resolve()


def load_basin_ids(file_path=None) -> list:
    """读取 86basin_ids.txt 中的 VALID_WATER_LEVEL_BASINS 列表。"""
    import ast

    path = Path(file_path) if file_path else PROJECT_ROOT / "86basin_ids.txt"
    content = path.read_text(encoding="utf-8")
    marker = content.find("VALID_WATER_LEVEL_BASINS")
    if marker == -1:
        raise ValueError(f"{path} 中未找到 VALID_WATER_LEVEL_BASINS")
    start = content.find("[", marker)
    depth = 0
    for i in range(start, len(content)):
        if content[i] == "[":
            depth += 1
        elif content[i] == "]":
            depth -= 1
            if depth == 0:
                return [str(b) for b in ast.literal_eval(content[start:i + 1])]
    raise ValueError(f"{path} 中的列表括号不匹配")


if __name__ == "__main__":
    import sys as _sys

    _sys.path.insert(0, str(PROJECT_ROOT))
    _sys.exit(0 if preflight(require_raw="--no-raw" not in _sys.argv) else 1)
