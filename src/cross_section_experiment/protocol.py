"""实验信息边界、原86限定、物理尺度与可重复运行指纹。"""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import numpy as np

PROTOCOL = "original86_cross_section_train_valid_q_hidden_v1"


def json_ready(value):
    """严格JSON不写NaN；缺失评价值保存为null。"""
    if isinstance(value, dict):
        return {str(k): json_ready(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(v) for v in value]
    if isinstance(value, np.ndarray):
        return json_ready(value.tolist())
    if isinstance(value, np.generic):
        return json_ready(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return None
    if isinstance(value, Path):
        return str(value)
    return value


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(json_ready(value), ensure_ascii=False, indent=2,
                               allow_nan=False), encoding="utf-8")


def fingerprint(value: dict) -> str:
    data = json.dumps(json_ready(value), sort_keys=True, ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(data).hexdigest()


def validate_population(prepared, expected_basins: list[str], expected_count: int = 86):
    """拒绝99流域对象或站号顺序不一致，不对缺断面的站静默删样本。"""
    if len(expected_basins) != expected_count or len(set(expected_basins)) != expected_count:
        raise ValueError(f"站号清单必须恰好含{expected_count}个不同站")
    if set(prepared.basins) != set(expected_basins):
        raise ValueError("缓存总体与原86站号清单不一致，不能混用99流域结果")
    times = prepared.grid.to_numpy(dtype="datetime64[ns]")
    delta = np.diff(times)
    step = np.timedelta64(3, "h")
    if len(times) < 2 or np.any(delta < step) or np.any(delta % step != np.timedelta64(0, "ns")):
        raise ValueError("原86输入网格须严格递增且与3小时网格对齐")
    # 缓存允许缺完整时刻，但不能把缺口前后的行当作相邻3小时输入。
    prepared.time_valid = np.r_[True, delta == step]


def heldout_split(basins: list[str], ratio: float, seed: int,
                  selection: str = "legacy_cross_section") -> tuple[list[str], list[str]]:
    if not 0 <= ratio < 1:
        raise ValueError("留出比例须满足0<=ratio<1")
    n = int(round(len(basins) * ratio))
    if selection == "original86":
        # 精确复用原86 masking.py 的子种子与站号顺序，不能只复用种子数字。
        from pipeline.masking import _subseed
        indices = np.random.default_rng(_subseed(seed, "holdout_flow", "all")).choice(
            len(basins), n, replace=False)
        selected = {basins[i] for i in indices}
    elif selection == "legacy_cross_section":
        # 保留首轮已完成断面实验的历史口径，不改写其结果。
        selected = set(np.random.default_rng(seed).choice(sorted(basins), n, replace=False))
    else:
        raise ValueError("未知的流域留出规则")
    held = [b for b in basins if b in selected]
    retained = [b for b in basins if b not in selected]
    if len(retained) < 3:
        raise ValueError("至少需要三个保留流域进行尺度拟合及内部验证")
    return held, retained


def q_masks(prepared, heldout: list[str]) -> tuple[dict, dict]:
    """只屏蔽所选站训练与验证Q，H和测试Q保持原始资料。"""
    train = np.zeros_like(prepared.targets_raw["flow"], dtype=bool)
    valid = np.zeros_like(train)
    for basin in heldout:
        bi = prepared.basin_index[basin]
        for split, array in (("train", train), ("valid", valid)):
            lo, hi = prepared.split_range(basin, split)
            array[bi, lo:hi + 1] = True
    return {"flow": train}, {"flow": valid}


def with_attributes(prepared, geometry: np.ndarray | None, columns: list[str]):
    """创建浅视图，仅替换静态输入，不改变冻结划分、目标或母对象。"""
    view = copy.copy(prepared)
    if geometry is not None:
        if geometry.shape[0] != len(prepared.basins):
            raise ValueError("断面输入的站数与缓存不一致")
        view.attrs = np.column_stack([prepared.attrs, geometry]).astype(np.float32)
        view.attr_columns = list(prepared.attr_columns) + list(columns)
        view.input_size = prepared.forcing.shape[-1] + view.attrs.shape[-1]
    return view


class WaterlevelOnlyPrepared:
    """水位模型完全不用Q标签、Q统计量或Q任务进行训练与选模。"""

    def __init__(self, prepared):
        self.base = prepared

    def __getattr__(self, key):
        return getattr(self.base, key)

    def make_scaling(self, scaling="physical", fit_basins=None):
        from pipeline.dataset import TargetScaling
        result = object.__new__(TargetScaling)
        result.scaling = "waterlevel_only"
        result.info = {"flow_information_used": False}
        result.mean = {"waterlevel": self.base.target_mean["waterlevel"]}
        result.std = {"waterlevel": self.base.target_std["waterlevel"]}
        result.normalized = {"waterlevel": self.base.targets["waterlevel"]}
        return result
