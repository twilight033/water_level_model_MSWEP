"""每流域 × 任务 × 分段的有效标签统计，以及最低样本门槛。

回应审稿意见 Major 4：原实现只用"整段记录有效比例 ≥10%"做一次全局粗筛
（multi_task_lstm_wl2d.py 的 filter_basins_with_valid_data），既没有报告
各分段的有效样本数，也没有保证每个流域在每个分段上都有足够标签。
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

from pipeline.paths import COVERAGE_DIR, SPLITS_DIR  # noqa: E402
from pipeline.splits import load_forcing_index, load_splits, load_targets_on_grid  # noqa: E402

TASKS = ("flow", "waterlevel")
SPLIT_NAMES = ("train", "valid", "test")

# 最低样本门槛：某流域在训练期某任务的有效标签数低于此值时，
# 该流域退出**该任务**的训练与评估，另一任务不受影响。
# 默认 2920 = 365 天 × 8 步/天，即至少一年的有效观测。
MIN_TRAIN_LABELS = 2920
ELIGIBILITY_FILE = SPLITS_DIR / "task_eligibility.json"


def _mask_for_split(grid: pd.DatetimeIndex, entry: dict, split: str) -> np.ndarray:
    start = pd.Timestamp(entry[f"{split}_start"])
    end = pd.Timestamp(entry[f"{split}_end"])
    return (grid >= start) & (grid <= end)


def build_coverage_table() -> pd.DataFrame:
    """逐流域 × 任务 × 分段统计有效标签数与有效比例。"""
    payload = load_splits()
    grid = load_forcing_index()
    flow, wl = load_targets_on_grid(grid)
    series = {"flow": flow, "waterlevel": wl}

    rows = []
    for basin, entry in payload["splits"].items():
        for split in SPLIT_NAMES:
            in_split = _mask_for_split(grid, entry, split)
            n_steps = int(in_split.sum())
            row = {"basin": basin, "split": split, "n_steps": n_steps,
                   "start": entry[f"{split}_start"], "end": entry[f"{split}_end"],
                   # 测试段不足 1 年的流域：保留参与训练与评估，但聚合时需单独标注
                   "short_test": entry.get("short_test", False)}
            for task in TASKS:
                valid = series[task][basin].notna().values & in_split
                n_valid = int(valid.sum())
                row[f"{task}_valid"] = n_valid
                row[f"{task}_ratio"] = (n_valid / n_steps) if n_steps else 0.0
            # 并集与交集，用于说明并集准入相对交集准入多保留多少样本
            union = ((series["flow"][basin].notna().values
                      | series["waterlevel"][basin].notna().values) & in_split)
            inter = ((series["flow"][basin].notna().values
                      & series["waterlevel"][basin].notna().values) & in_split)
            row["union_valid"] = int(union.sum())
            row["intersect_valid"] = int(inter.sum())
            rows.append(row)
    return pd.DataFrame(rows)


def build_eligibility(coverage: pd.DataFrame,
                      min_train_labels: int = MIN_TRAIN_LABELS) -> dict:
    """按训练期有效标签数决定每个流域参与哪些任务。"""
    train = coverage[coverage["split"] == "train"].set_index("basin")
    eligibility = {}
    for task in TASKS:
        keep = train.index[train[f"{task}_valid"] >= min_train_labels].tolist()
        eligibility[task] = sorted(keep)
    eligibility["_meta"] = {
        "min_train_labels": min_train_labels,
        "min_train_days": round(min_train_labels * 3 / 24, 1),
        "n_flow": len(eligibility["flow"]),
        "n_waterlevel": len(eligibility["waterlevel"]),
        "n_both": len(set(eligibility["flow"]) & set(eligibility["waterlevel"])),
    }
    return eligibility


def load_eligibility(path: Path = ELIGIBILITY_FILE) -> dict:
    if not path.exists():
        raise FileNotFoundError(
            f"任务资格文件不存在: {path}\n请先运行 python src/pipeline/coverage.py"
        )
    return json.loads(path.read_text(encoding="utf-8"))


if __name__ == "__main__":
    COVERAGE_DIR.mkdir(parents=True, exist_ok=True)
    SPLITS_DIR.mkdir(parents=True, exist_ok=True)

    cov = build_coverage_table()
    out = COVERAGE_DIR / "basin_task_split_counts.csv"
    cov.to_csv(out, index=False, encoding="utf-8-sig")
    print(f"覆盖率表: {out}  （{len(cov)} 行 = {cov['basin'].nunique()} 流域 × 3 分段）")

    print("\n各分段有效标签数汇总（步，1 步 = 3 小时）:")
    summary = cov.groupby("split")[
        ["n_steps", "flow_valid", "waterlevel_valid", "union_valid", "intersect_valid"]
    ].agg(["median", "min", "max"])
    print(summary.to_string())

    print("\n有效比例分位数:")
    for split in SPLIT_NAMES:
        sub = cov[cov["split"] == split]
        print(f"  {split}: Q  {sub['flow_ratio'].quantile([0, .25, .5, .75, 1]).round(3).tolist()}")
        print(f"  {split}: h  {sub['waterlevel_ratio'].quantile([0, .25, .5, .75, 1]).round(3).tolist()}")

    tr = cov[cov["split"] == "train"]
    gain = (tr["union_valid"] - tr["intersect_valid"]) / tr["intersect_valid"].replace(0, np.nan)
    print(f"\n训练期并集准入相对交集准入的样本增益（中位数）: {gain.median():.1%}")

    elig = build_eligibility(cov)
    ELIGIBILITY_FILE.write_text(json.dumps(elig, indent=2, ensure_ascii=False), encoding="utf-8")
    m = elig["_meta"]
    print(f"\n最低训练标签门槛: {m['min_train_labels']} 步（约 {m['min_train_days']} 天）")
    print(f"  参与径流任务: {m['n_flow']}/86    参与水位任务: {m['n_waterlevel']}/86    两者皆可: {m['n_both']}")
    short = sorted(cov.loc[cov["short_test"], "basin"].unique())
    print("")
    print(f"测试段不足 1 年的流域（保留，但聚合结果需同时报告含/不含版本）: {short}")
    dropped_q = sorted(set(cov['basin']) - set(elig['flow']))
    dropped_h = sorted(set(cov['basin']) - set(elig['waterlevel']))
    if dropped_q:
        print(f"  未达门槛（径流）: {dropped_q}")
    if dropped_h:
        print(f"  未达门槛（水位）: {dropped_h}")
    print(f"已写入: {ELIGIBILITY_FILE}")
