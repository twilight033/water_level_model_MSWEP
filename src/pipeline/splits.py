"""冻结每流域的训练/验证/测试边界。

设计要点
--------
1. 边界只算一次，写入 data/splits/basin_splits.json，之后所有模型、所有
   缺失场景、所有窗口长度都读这份文件，Dataset 内部不再推断边界。
2. 边界定义在**目标时刻轴**上，而不是窗口起点轴上。旧实现用
   ``first_valid_time = 首个有效目标 - seq_length*3h`` 反推起点，导致
   窗口长度一变边界就漂移，7/21/60 天敏感性实验无法在同一测试期上比较。
3. 参考有效期用 **Q 与 h 的交集**，训练段再向前扩展到并集起点（方案 C）。
   起因是实测发现 86 个流域中 65 个的水位记录从 2007 年才开始，相对径流
   平均滞后 10 年。若直接按并集有效期切 60/20/20，训练期 60% 大部分落在
   "只有径流"的年代，训练期水位标签覆盖率中位数仅 32%（42 个流域低于
   30%，2 个为 0），而验证/测试期高达 99.7%，水位任务训练监督严重不足。
   方案 C 用 Q∩h 共同有效期确定 60/20/20 的两个分界点，保证验证段与
   测试段两个任务都有充分标签；训练段则向前扩展到并集起点，把 2007 年
   之前只有径流的观测也用上（那些时刻的水位损失由逐任务掩膜屏蔽）。
   实测中位数：训练期 h 标签 29801 步、Q 标签 45023 步，均优于纯并集
   （12836 / 37642）与纯交集（29801 / 26524）方案。
   实际比例约 76/12/12，非严格 60/20/20，需在论文中说明。
4. 划分只依赖**原始未掩膜**标签，与人工缺失场景无关。
5. 验证段与测试段起始各留一段禁运期（embargo），保证任何测试样本的输入
   窗口都不会覆盖训练期的目标时刻。禁运长度取固定值（默认 60 天，即
   敏感性实验中最长的窗口），因此不随 seq_length 变化。
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

_ROOT = Path(__file__).resolve().parents[2]
for _p in (str(_ROOT / "src"), str(_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from pipeline.paths import (  # noqa: E402
    BASIN_SPLITS_FILE, COVERAGE_DIR, FLOW_CACHE, SPLITS_DIR, WATERLEVEL_CACHE,
    load_basin_ids,
)
from pipeline.export_forcing import MSWEP_CACHE, _cache_path, CAMELSH_FORCING_VARS  # noqa: E402

STEP_HOURS = 3
TRAIN_RATIO = 0.6
VALID_RATIO = 0.2
# 测试比例 = 1 - TRAIN_RATIO - VALID_RATIO
EMBARGO_DAYS = 60  # 与敏感性实验中最长窗口一致，保证边界不随 seq_length 变化


def load_forcing_index() -> pd.DatetimeIndex:
    """所有强迫变量共同覆盖的 3 小时时间轴。

    MSWEP 到 2024-12-02 12:00 结束，CAMELSH 强迫到 2024-12-31，
    取交集作为可用的窗口时间轴。
    """
    idx = pd.DatetimeIndex(pd.read_parquet(MSWEP_CACHE, columns=[]).index)
    for var in CAMELSH_FORCING_VARS:
        other = pd.DatetimeIndex(pd.read_parquet(_cache_path(var), columns=[]).index)
        idx = idx.intersection(other)
    return idx.sort_values()


def load_targets_on_grid(grid: pd.DatetimeIndex) -> tuple:
    """把小时级 Q/h 取到 3 小时目标网格上（瞬时值，不做聚合）。"""
    flow = pd.read_parquet(FLOW_CACHE).reindex(grid)
    wl = pd.read_parquet(WATERLEVEL_CACHE).reindex(grid)
    return flow, wl


def compute_basin_splits(
    ratios=(TRAIN_RATIO, VALID_RATIO),
    embargo_days: int = EMBARGO_DAYS,
) -> dict:
    """方案 C：分界点按 Q∩h 共同有效期算，训练段向前扩展到并集起点。"""
    basins = load_basin_ids()
    grid = load_forcing_index()
    flow, wl = load_targets_on_grid(grid)
    embargo_steps = int(embargo_days * 24 / STEP_HOURS)
    train_ratio, valid_ratio = ratios

    splits, skipped = {}, []
    for basin in basins:
        q_valid = flow[basin].notna().values
        h_valid = wl[basin].notna().values
        union = q_valid | h_valid
        intersect = q_valid & h_valid

        if not intersect.any():
            skipped.append((basin, "Q 与 h 无任何共同有效时刻，无法确定分界点"))
            continue

        inter_pos = np.flatnonzero(intersect)
        ref_start, ref_end = int(inter_pos[0]), int(inter_pos[-1])
        ref_span = ref_end - ref_start + 1
        if ref_span < 3 * embargo_steps:
            skipped.append((basin,
                            f"共同有效期过短（{ref_span} 步 < 3×禁运 {3 * embargo_steps} 步）"))
            continue

        # 分界点在共同有效期上按比例确定
        mark_train = ref_start + int(ref_span * train_ratio)
        mark_valid = ref_start + int(ref_span * (train_ratio + valid_ratio))
        # 训练段向前扩展到并集起点，用上 2007 年前只有径流的观测
        train_start = int(np.flatnonzero(union)[0])

        entry = {
            "train_start": str(grid[train_start]),
            "train_end": str(grid[mark_train - 1]),
            "valid_start": str(grid[mark_train + embargo_steps]),
            "valid_end": str(grid[mark_valid - 1]),
            "test_start": str(grid[mark_valid + embargo_steps]),
            "test_end": str(grid[ref_end]),
            "ref_start": str(grid[ref_start]),
            "ref_end": str(grid[ref_end]),
            "ref_span_steps": ref_span,
            "n_valid_union": int(union.sum()),
            "n_valid_intersect": int(intersect.sum()),
        }
        # 测试段不足一年的流域单独标注（按用户决定：保留但在结果中标出）
        test_steps = ref_end - (mark_valid + embargo_steps) + 1
        entry["test_steps"] = int(test_steps)
        entry["short_test"] = bool(test_steps < 365 * 24 / STEP_HOURS)
        splits[basin] = entry

    short = sorted(b for b, v in splits.items() if v["short_test"])
    meta = {
        "scheme": "C：分界点按 Q∩h 共同有效期的 60/20/20，训练段前扩至并集起点",
        "step_hours": STEP_HOURS,
        "ratios": {"train": train_ratio, "valid": valid_ratio,
                   "test": round(1 - train_ratio - valid_ratio, 6)},
        "embargo_days": embargo_days,
        "embargo_steps": embargo_steps,
        "reference": "Q 与 h 的交集有效期（原始未掩膜标签）确定分界点；训练段起点取并集起点",
        "grid_start": str(grid[0]),
        "grid_end": str(grid[-1]),
        "grid_steps": len(grid),
        "n_basins_kept": len(splits),
        "n_basins_skipped": len(skipped),
        "skipped": [{"basin": b, "reason": r} for b, r in skipped],
        "short_test_basins": short,
        "short_test_note": "测试段不足 1 年；按决定保留参与训练与评估，"
                           "但聚合结果需同时报告含/不含这些流域的版本",
    }
    return {"meta": meta, "splits": splits}


def save_splits(payload: dict, path: Path = BASIN_SPLITS_FILE) -> Path:
    SPLITS_DIR.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return path


def load_splits(path: Path = BASIN_SPLITS_FILE) -> dict:
    if not path.exists():
        raise FileNotFoundError(
            f"冻结划分文件不存在: {path}\n请先运行 python src/pipeline/splits.py"
        )
    return json.loads(path.read_text(encoding="utf-8"))


def split_bounds(basin: str, loader_type: str, payload: dict = None) -> tuple:
    """返回该流域该分段的目标时刻闭区间 (start, end)。"""
    payload = payload or load_splits()
    entry = payload["splits"][basin]
    key = {"train": ("train_start", "train_end"),
           "valid": ("valid_start", "valid_end"),
           "test": ("test_start", "test_end")}[loader_type]
    return pd.Timestamp(entry[key[0]]), pd.Timestamp(entry[key[1]])


if __name__ == "__main__":
    payload = compute_basin_splits()
    path = save_splits(payload)
    meta = payload["meta"]
    print(f"强迫时间轴: {meta['grid_start']} → {meta['grid_end']}（{meta['grid_steps']} 步 @3h）")
    print(f"保留流域: {meta['n_basins_kept']}，剔除: {meta['n_basins_skipped']}")
    for item in meta["skipped"]:
        print(f"  剔除 {item['basin']}: {item['reason']}")
    print(f"禁运期: {meta['embargo_days']} 天（{meta['embargo_steps']} 步）")
    print(f"已写入: {path}")

    rows = [{"basin": b, **v} for b, v in payload["splits"].items()]
    df = pd.DataFrame(rows)
    COVERAGE_DIR.mkdir(parents=True, exist_ok=True)
    out = COVERAGE_DIR / "basin_split_bounds.csv"
    df.to_csv(out, index=False, encoding="utf-8-sig")
    print(f"划分边界表: {out}")
    print("\n划分起止年份分布:")
    print(pd.DataFrame({
        "train_start": pd.to_datetime(df["train_start"]).dt.year,
        "test_start": pd.to_datetime(df["test_start"]).dt.year,
        "test_end": pd.to_datetime(df["test_end"]).dt.year,
    }).describe().loc[["min", "25%", "50%", "75%", "max"]].to_string())
