"""项目路径常量与 CAMELSH 数据可用性校验。

F 盘是可移动盘，批量实验前必须先确认数据目录挂载正常，
否则会跑到一半才发现读不到文件。
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
    if not root.exists():
        raise FileNotFoundError(
            f"CAMELSH 数据根目录不存在: {root}\n"
            f"F 盘是可移动盘，请确认已挂载，或修改 config.py 的 CAMELSH_DATA_PATH。"
        )
    missing = [str(sub) for sub in _REQUIRED_SUBDIRS if not (root / sub).is_dir()]
    if missing:
        raise FileNotFoundError(
            f"CAMELSH 数据目录结构不完整: {root}\n"
            f"缺少子目录: {missing}"
        )
    return root.resolve()


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
