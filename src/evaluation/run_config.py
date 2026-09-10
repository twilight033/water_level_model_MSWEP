"""运行记录的“可比配置”口径。

只有**除架构与随机种子之外全部超参都相同**的两次运行才能互相比较。本模块把
一次运行的超参折成一个短标签（base / L480 / wh0.5 …），供 report、redundancy、
two_stage_baseline、attribute_diagnostics 共用同一套口径。

单独成模块而不是留在 report.py 里，是因为上述几个脚本都要用它——为了一个辅助
函数去 import 一个“生成表格的脚本”并不合适。
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

_ROOT = Path(__file__).resolve().parents[2]
for _p in (str(_ROOT / "src"), str(_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from pipeline.paths import RESULTS_ROOT  # noqa: E402

SUMMARY_DIR = RESULTS_ROOT / "summary"


# 一次运行的“可比配置”＝除架构与随机种子之外的全部超参。只有配置完全相同的
# 两次运行才能互相比较：把 7 天窗口与 60 天窗口、观测归一化与物理归一化混进
# 同一个均值是没有意义的。
#
# 此前各表一律只按 scenario == "complete" 过滤，该条件会把 4D（不同窗口长度）、
# 4F（不同流域子集）、4G（不同任务权重）以及 4H/4I/4J 一并混进来：
#   - 均值混了不同配置的运行；
#   - 同一流域出现多行，paired_differences 抛 ValueError，而 paired_tests_table
#     把 ValueError 吞掉 continue，导致“双头 vs 单任务”这组关键对比被静默丢弃。
#
# 标签不按 group 名推断，而是直接读 run.json 里记录的实参。group 只是排程用的
# 名字，同一分组内部也可能含多种配置（4D_window 就同时含 7 天与 60 天窗口，
# 按 group 归并会把 0.49 和 0.64 平均成一个无意义的 0.57）。
CONFIG_FIELDS = {
    # 字段: (TrainConfig 的默认值, 取非默认值时写进标签的模板)
    "seq_length": (168, "L{:g}"),
    "loss_norm": ("per_valid", "{}"),
    "strict_basins": (False, "strict"),
    "waterlevel_weight": (1.0, "wh{:g}"),
    "target_scaling": ("observed", "{}"),
    "attr_set": ("base", "{}"),
}


def _config_label(row) -> str:
    """把一次运行的超参折成一个短标签；全取默认值时为 base。"""
    parts = []
    for field, (default, template) in CONFIG_FIELDS.items():
        value = row.get(field)
        if value is None or (isinstance(value, float) and np.isnan(value)):
            continue
        if isinstance(default, bool):
            # 经 CSV 往返后可能是布尔也可能是字符串，而 bool("False") 为真，
            # 不能直接用 bool() 判断
            if str(value).lower() in ("true", "1"):
                parts.append(template)
        elif isinstance(default, (int, float)):
            if float(value) != float(default):
                parts.append(template.format(float(value)))
        elif value != default:
            parts.append(template.format(value))
    return "+".join(parts) if parts else "base"


def add_config_column(metrics: pd.DataFrame, runs: pd.DataFrame) -> pd.DataFrame:
    """给逐流域指标表标注所属的可比配置（按 run_key 从运行记录取超参）。"""
    labels = {rec["run_key"]: _config_label(rec) for _, rec in runs.iterrows()}
    out = metrics.copy()
    out["config"] = out["run_key"].map(labels)
    unknown = out["config"].isna()
    if unknown.any():
        # 缺少对应 run.json 的运行不能假定其配置，单独标出而非并入 base
        out.loc[unknown, "config"] = "未知配置"
        print(f"警告: {int(unknown.sum())} 行指标找不到运行记录，已标为“未知配置”")
    return out


def load_summary() -> tuple:
    metrics = pd.read_csv(SUMMARY_DIR / "all_metrics.csv", dtype={"basin": str},
                          low_memory=False)
    runs = pd.read_csv(SUMMARY_DIR / "all_runs.csv")
    return add_config_column(metrics, runs), runs


def select(metrics: pd.DataFrame, config: str = "base", scenario=None,
           task=None, held_out=None) -> pd.DataFrame:
    """筛出一批可以互相比较的结果。

    held_out 传 True/False 时只保留被留出/保留的流域；该列经 CSV 往返后可能是
    字符串，不能直接用布尔比较。
    """
    sub = metrics[metrics["config"] == config]
    if scenario is not None:
        sub = sub[sub["scenario"] == scenario]
    if task is not None:
        sub = sub[sub["task"] == task]
    if held_out is not None:
        flag = sub["held_out"].astype(str).str.lower().isin(("true", "1"))
        sub = sub[flag == bool(held_out)]
    return sub
