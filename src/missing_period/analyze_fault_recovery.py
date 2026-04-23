#!/usr/bin/env python3
"""
故障恢复时间统计脚本

目标：在经过初步筛选的流域（Q 和 h 有效比例均 ≥ 30%）中，
统计观测期内真实的故障恢复时间分布，避免将非观测期误算进样本。

观测期判定规则：
  1. 首尾 NaN 段 → 非观测期（站点尚未/已停止运营）
  2. 观测期内部，连续缺失 > LONG_GAP_THRESHOLD 天 → 也视为非观测期，
     将观测期切割为若干子期
  3. 每个子期内的所有缺失段 → 故障样本（故障恢复时间）

数据源：
  径流 Q：F:/data/CAMELSH/timeseries/Data/CAMELSH/timeseries/*.nc  变量 Streamflow
  水位 h：F:/data/CAMELSH/Hourly2/Hourly2/*_hourly.nc              变量 water_level
  流域列表：src/others/output/all_basins_valid30pct.txt

输出目录：src/missing_period/output/
"""

import sys
from pathlib import Path
import ast

import numpy as np
import pandas as pd
import xarray as xr
from tqdm import tqdm

# ==================== 路径配置 ====================
CAMELSH_DATA_PATH   = Path("F:/data")
TIMESERIES_DIR      = CAMELSH_DATA_PATH / "CAMELSH" / "timeseries" / "Data" / "CAMELSH" / "timeseries"
HOURLY2_DIR         = CAMELSH_DATA_PATH / "CAMELSH" / "Hourly2" / "Hourly2"

# 经过初步筛选（Q 和 h ≥30%）的流域列表
BASIN_LIST_FILE = Path(__file__).parent.parent / "others" / "output" / "all_basins_valid30pct.txt"

# 输出目录
OUTPUT_DIR = Path(__file__).parent / "output"

# ==================== 关键参数 ====================
# 观测期内超过此天数的缺失段视为"非观测期"，用于切割子期
LONG_GAP_THRESHOLD_DAYS = 180
# 折算为小时（数据分辨率 1 小时）
LONG_GAP_THRESHOLD_HOURS = LONG_GAP_THRESHOLD_DAYS * 24


# ==================== 核心算法 ====================

def get_gap_runs(arr: np.ndarray):
    """
    对布尔掩码（True=缺失）做 run-length encoding，返回每段缺失的起止索引。

    Returns
    -------
    starts, ends, lengths : 均为 np.ndarray
        starts[i] : 第 i 段缺失的起始索引（含）
        ends[i]   : 第 i 段缺失的结束索引（不含）
        lengths[i]: 段长（时间步数）
    """
    padded = np.concatenate(([False], arr, [False]))
    diff   = np.diff(padded.astype(np.int8))
    starts  = np.where(diff ==  1)[0]
    ends    = np.where(diff == -1)[0]
    return starts, ends, ends - starts


def analyze_basin(arr: np.ndarray, timestamps: np.ndarray,
                  long_gap_threshold: int):
    """
    分析单个流域的观测期与故障样本。

    Parameters
    ----------
    arr : np.ndarray
        原始时序（NaN = 缺失），1D，shape=(T,)
    timestamps : np.ndarray
        对应的时间戳数组（datetime64），shape=(T,)
    long_gap_threshold : int
        过长中断阈值（时间步数）

    Returns
    -------
    dict
        per-basin 统计信息
    """
    n = len(arr)
    missing_mask = np.isnan(arr)
    valid_mask   = ~missing_mask

    if not valid_mask.any():
        return None

    # ------ 1. 首尾 NaN = 非观测期 ------
    first_v = int(np.argmax(valid_mask))
    last_v  = int(n - 1 - np.argmax(valid_mask[::-1]))

    # ------ 2. 观测期内的缺失段 ------
    inner_missing = missing_mask[first_v : last_v + 1]
    if not inner_missing.any():
        # 观测期内无任何缺失
        return _build_record(
            timestamps, first_v, last_v,
            obs_subperiods=[(first_v, last_v)],
            fault_lengths=[],
            long_gap_indices=[]
        )

    inner_starts, inner_ends, inner_lengths = get_gap_runs(inner_missing)
    # 转回原始数组的绝对索引
    abs_starts  = inner_starts  + first_v
    abs_ends    = inner_ends    + first_v

    # ------ 3. 按过长缺失段切割观测子期 ------
    long_mask = inner_lengths > long_gap_threshold
    long_idx  = np.where(long_mask)[0]   # 过长段在内部段列表中的下标

    # 按过长段边界切割观测子期
    obs_subperiods = []   # (sub_start, sub_end) 原始索引
    fault_lengths  = []   # 故障段长度列表（时间步）

    seg_start = first_v
    for li in long_idx:
        gap_abs_start = abs_starts[li]
        gap_abs_end   = abs_ends[li]

        # 当前子期结束：gap 前最后一个有效点
        sub_end_candidates = np.where(valid_mask[seg_start : gap_abs_start])[0]
        if len(sub_end_candidates) > 0:
            sub_end = seg_start + sub_end_candidates[-1]
            obs_subperiods.append((seg_start, sub_end))
        # else: 子期内全是缺失，忽略

        # 收集当前子期内（seg_start ~ gap_abs_start）的短缺失段为故障
        for s, e, l in zip(abs_starts, abs_ends, inner_lengths):
            if s >= seg_start and e <= gap_abs_start and l <= long_gap_threshold:
                fault_lengths.append(int(l))

        # 下一子期从 gap 结束后首个有效点开始
        next_valids = np.where(valid_mask[gap_abs_end:])[0]
        if len(next_valids) == 0:
            seg_start = last_v + 1   # 无后续有效数据
            break
        seg_start = gap_abs_end + int(next_valids[0])

    # 最后一个子期
    if seg_start <= last_v:
        obs_subperiods.append((seg_start, last_v))
        # 收集该子期内的短缺失段
        for s, e, l in zip(abs_starts, abs_ends, inner_lengths):
            if s >= seg_start and e <= last_v + 1 and l <= long_gap_threshold:
                fault_lengths.append(int(l))

    return _build_record(
        timestamps, first_v, last_v,
        obs_subperiods=obs_subperiods,
        fault_lengths=fault_lengths,
        long_gap_indices=long_idx.tolist()
    )


def _build_record(timestamps, first_v, last_v, obs_subperiods, fault_lengths,
                  long_gap_indices):
    """整理统计字段。"""
    obs_duration_hours = last_v - first_v + 1

    # 各子期有效步数之和
    total_obs_hours = sum(e - s + 1 for s, e in obs_subperiods)
    # 故障总时长
    fault_total_hours = sum(fault_lengths)

    rec = {
        "obs_start":           str(timestamps[first_v])[:19],
        "obs_end":             str(timestamps[last_v])[:19],
        "obs_duration_days":   round(obs_duration_hours / 24, 1),
        "num_subperiods":      len(obs_subperiods),
        "obs_valid_hours":     total_obs_hours,
        "fault_count":         len(fault_lengths),
        "fault_total_hours":   fault_total_hours,
        "fault_rate_pct":      round(fault_total_hours / total_obs_hours * 100, 3)
                               if total_obs_hours > 0 else 0.0,
        "max_fault_hours":     int(max(fault_lengths)) if fault_lengths else 0,
        "mean_fault_hours":    round(float(np.mean(fault_lengths)), 2) if fault_lengths else 0.0,
        "median_fault_hours":  round(float(np.median(fault_lengths)), 2) if fault_lengths else 0.0,
        "p90_fault_hours":     round(float(np.percentile(fault_lengths, 90)), 2) if fault_lengths else 0.0,
        "p95_fault_hours":     round(float(np.percentile(fault_lengths, 95)), 2) if fault_lengths else 0.0,
        "p99_fault_hours":     round(float(np.percentile(fault_lengths, 99)), 2) if fault_lengths else 0.0,
        "fault_lengths_hours": ";".join(map(str, fault_lengths)),
    }
    return rec


# ==================== 数据读取 ==================

def read_streamflow_with_time(nc_path: Path):
    """读取径流数组和时间戳，失败返回 (None, None)。"""
    try:
        with xr.open_dataset(nc_path) as ds:
            arr = ds["Streamflow"].values.astype(float)
            ts  = ds["DateTime"].values
        return arr, ts
    except Exception as e:
        print(f"  [警告] 读取径流失败 {nc_path.name}: {e}")
        return None, None


def read_waterlevel_with_time(nc_path: Path):
    """读取水位数组和时间戳，失败返回 (None, None)。"""
    try:
        with xr.open_dataset(nc_path) as ds:
            if "water_level" not in ds.data_vars:
                return None, None
            arr = ds["water_level"].values.astype(float)
            ts  = ds["time"].values
        return arr, ts
    except Exception as e:
        print(f"  [警告] 读取水位失败 {nc_path.name}: {e}")
        return None, None


# ==================== 流域列表读取 ==================

def load_basin_list(file_path: Path) -> list:
    """从 all_basins_valid30pct.txt 读取流域ID列表。"""
    content = file_path.read_text(encoding="utf-8")
    start = content.find("VALID_BASINS_30PCT = [")
    if start == -1:
        raise ValueError(f"未在 {file_path} 中找到 VALID_BASINS_30PCT")
    list_start = content.find("[", start)
    depth, pos = 0, list_start
    for i in range(list_start, len(content)):
        if content[i] == "[":
            depth += 1
        elif content[i] == "]":
            depth -= 1
            if depth == 0:
                pos = i + 1
                break
    return ast.literal_eval(content[list_start:pos])


# ==================== 汇总报告 ==================

def generate_report(df_q: pd.DataFrame, df_h: pd.DataFrame,
                    threshold_days: int) -> str:
    lines = []
    lines.append("=" * 70)
    lines.append("CAMELSH 故障恢复时间统计报告")
    lines.append(f"过长缺失阈值（非观测期）: > {threshold_days} 天")
    lines.append("=" * 70)

    for label, df in [("径流 Q (Streamflow)", df_q), ("水位 h (Water Level)", df_h)]:
        lines.append(f"\n【{label}】")
        lines.append(f"  处理流域数: {len(df)}")

        no_fault = df[df["fault_count"] == 0]
        has_fault = df[df["fault_count"] > 0]
        lines.append(f"  观测期内无故障的流域: {len(no_fault)}")
        lines.append(f"  有故障记录的流域:     {len(has_fault)}")

        if len(has_fault) == 0:
            continue

        lines.append(f"\n  -- 观测期信息 --")
        lines.append(f"  观测期长度 均值:  {df['obs_duration_days'].mean():.0f} 天")
        lines.append(f"  观测子期数 均值:  {df['num_subperiods'].mean():.1f} 个/流域")

        lines.append(f"\n  -- 故障率（观测期内缺失占比）--")
        lines.append(f"  均值:   {has_fault['fault_rate_pct'].mean():.2f} %")
        lines.append(f"  中位数: {has_fault['fault_rate_pct'].median():.2f} %")
        lines.append(f"  最大值: {has_fault['fault_rate_pct'].max():.2f} %")

        # 汇总所有故障样本
        all_faults = []
        for row in has_fault["fault_lengths_hours"]:
            if pd.notna(row) and str(row).strip():
                all_faults.extend(int(x) for x in str(row).split(";") if x)
        all_faults = np.array(all_faults, dtype=float)
        total_samples = len(all_faults)

        lines.append(f"\n  -- 故障恢复时间分布（全部样本，共 {total_samples:,} 个）--")
        if total_samples > 0:
            for pct in [50, 75, 90, 95, 99]:
                v = np.percentile(all_faults, pct)
                lines.append(f"  p{pct:2d}: {v:.1f} 小时  ({v/24:.2f} 天)")
            lines.append(f"  最大值: {all_faults.max():.1f} 小时  ({all_faults.max()/24:.1f} 天)")
            lines.append(f"  均值:   {all_faults.mean():.1f} 小时  ({all_faults.mean()/24:.2f} 天)")

            # 按时长分档
            bins  = [0, 1, 3, 6, 12, 24, 72, 168, 720, 2160, float("inf")]
            lbls  = ["≤1h", "1-3h", "3-6h", "6-12h", "12h-1天",
                     "1-3天", "3-7天", "7-30天", "30-90天", ">90天"]
            lines.append(f"\n  故障时长分布：")
            counts, _ = np.histogram(all_faults, bins=bins)
            for lbl, cnt in zip(lbls, counts):
                pct_str = f"{cnt/total_samples*100:5.1f}%"
                bar = "█" * int(cnt / max(counts) * 25)
                lines.append(f"    {lbl:10s}: {cnt:8,} 次  {pct_str}  {bar}")

    lines.append("\n" + "=" * 70)
    return "\n".join(lines)


# ==================== 主流程 ====================

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("故障恢复时间统计")
    print(f"过长缺失阈值: > {LONG_GAP_THRESHOLD_DAYS} 天 → 视为非观测期")
    print("=" * 70)

    # 读取流域列表
    print(f"\n[1] 读取筛选后的流域列表: {BASIN_LIST_FILE}")
    basins = load_basin_list(BASIN_LIST_FILE)
    print(f"  流域数量: {len(basins)}")

    records_q = []
    records_h = []

    print(f"\n[2] 逐流域统计...")
    for basin_id in tqdm(basins, desc="处理进度", unit="流域"):

        # --- 径流 Q ---
        nc_q = TIMESERIES_DIR / f"{basin_id}.nc"
        if nc_q.exists():
            arr_q, ts_q = read_streamflow_with_time(nc_q)
            if arr_q is not None:
                rec = analyze_basin(arr_q, ts_q, LONG_GAP_THRESHOLD_HOURS)
                if rec is not None:
                    rec["basin_id"] = basin_id
                    records_q.append(rec)

        # --- 水位 h ---
        nc_h = HOURLY2_DIR / f"{basin_id}_hourly.nc"
        if nc_h.exists():
            arr_h, ts_h = read_waterlevel_with_time(nc_h)
            if arr_h is not None:
                rec = analyze_basin(arr_h, ts_h, LONG_GAP_THRESHOLD_HOURS)
                if rec is not None:
                    rec["basin_id"] = basin_id
                    records_h.append(rec)

    # 整理列顺序
    cols = ["basin_id", "obs_start", "obs_end", "obs_duration_days",
            "num_subperiods", "obs_valid_hours",
            "fault_count", "fault_total_hours", "fault_rate_pct",
            "max_fault_hours", "mean_fault_hours", "median_fault_hours",
            "p90_fault_hours", "p95_fault_hours", "p99_fault_hours",
            "fault_lengths_hours"]

    df_q = pd.DataFrame(records_q, columns=cols) if records_q else pd.DataFrame(columns=cols)
    df_h = pd.DataFrame(records_h, columns=cols) if records_h else pd.DataFrame(columns=cols)

    # 保存 CSV
    out_q = OUTPUT_DIR / "fault_recovery_Q.csv"
    out_h = OUTPUT_DIR / "fault_recovery_H.csv"
    df_q.to_csv(out_q, index=False, encoding="utf-8-sig")
    df_h.to_csv(out_h, index=False, encoding="utf-8-sig")
    print(f"\n  已保存径流故障统计: {out_q}")
    print(f"  已保存水位故障统计: {out_h}")

    # 生成报告
    report = generate_report(df_q, df_h, LONG_GAP_THRESHOLD_DAYS)
    print("\n" + report)
    out_report = OUTPUT_DIR / "fault_recovery_report.txt"
    out_report.write_text(report, encoding="utf-8")
    print(f"\n  文字报告已保存: {out_report}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n用户中断")
        sys.exit(1)
    except Exception as e:
        import traceback
        print(f"\n发生错误: {e}")
        traceback.print_exc()
        sys.exit(1)
