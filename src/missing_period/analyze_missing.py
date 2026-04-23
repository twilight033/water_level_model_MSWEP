#!/usr/bin/env python3
"""
CAMELSH 数据集缺失情况统计脚本

统计所有流域径流（Q）和水位（h）数据的：
- 缺失率
- 连续缺失段数量
- 连续缺失时长（最大值、均值、中位数、分布）

数据源：
  - 径流：F:/data/CAMELSH/timeseries/Data/CAMELSH/timeseries/*.nc  变量 Streamflow
  - 水位：F:/data/CAMELSH/Hourly2/Hourly2/*_hourly.nc              变量 water_level

时间分辨率：1小时
输出：src/missing_period/results/ 目录下的 CSV 和文本报告
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
from tqdm import tqdm

# ==================== 路径配置 ====================
CAMELSH_DATA_PATH = Path("F:/data")
TIMESERIES_DIR = CAMELSH_DATA_PATH / "CAMELSH" / "timeseries" / "Data" / "CAMELSH" / "timeseries"
HOURLY2_DIR     = CAMELSH_DATA_PATH / "CAMELSH" / "Hourly2" / "Hourly2"

# 输出目录（相对于本脚本所在位置）
OUTPUT_DIR = Path(__file__).parent / "results"

# 分辨率（小时）
TIME_STEP_HOURS = 1


# ==================== 核心计算函数 ====================

def compute_missing_stats(series: np.ndarray) -> dict:
    """
    对一维时序数组计算缺失统计量。

    Parameters
    ----------
    series : np.ndarray
        原始数值数组，NaN 表示缺失。

    Returns
    -------
    dict 包含：
        total_steps         : 总时间步数
        missing_steps       : 缺失步数
        missing_rate_pct    : 缺失率 (%)
        num_gaps            : 连续缺失段数量
        max_gap_hours       : 最长连续缺失（小时）
        mean_gap_hours      : 平均连续缺失（小时）
        median_gap_hours    : 中位连续缺失（小时）
        max_gap_days        : 最长连续缺失（天）
        mean_gap_days       : 平均连续缺失（天）
        gap_lengths_hours   : 每段连续缺失长度列表（小时）
    """
    total = len(series)
    missing_mask = np.isnan(series)
    missing_count = int(missing_mask.sum())

    stats = {
        "total_steps":      total,
        "missing_steps":    missing_count,
        "missing_rate_pct": round(missing_count / total * 100, 4) if total > 0 else 0.0,
        "num_gaps":         0,
        "max_gap_hours":    0,
        "mean_gap_hours":   0.0,
        "median_gap_hours": 0.0,
        "max_gap_days":     0.0,
        "mean_gap_days":    0.0,
        "gap_lengths_hours": [],
    }

    if missing_count == 0:
        return stats

    # 计算连续缺失段：对 missing_mask 做 run-length encoding
    # 找到每个连续段的起止索引
    padded = np.concatenate(([False], missing_mask, [False]))
    diff = np.diff(padded.astype(int))
    starts = np.where(diff == 1)[0]   # 段开始位置
    ends   = np.where(diff == -1)[0]  # 段结束位置（不含）
    gap_lengths = (ends - starts) * TIME_STEP_HOURS  # 转为小时

    stats["num_gaps"]         = len(gap_lengths)
    stats["max_gap_hours"]    = int(gap_lengths.max())
    stats["mean_gap_hours"]   = round(float(gap_lengths.mean()), 2)
    stats["median_gap_hours"] = round(float(np.median(gap_lengths)), 2)
    stats["max_gap_days"]     = round(stats["max_gap_hours"] / 24, 2)
    stats["mean_gap_days"]    = round(stats["mean_gap_hours"] / 24, 2)
    stats["gap_lengths_hours"] = gap_lengths.tolist()
    return stats


def read_streamflow(nc_path: Path) -> np.ndarray | None:
    """从 timeseries NC 文件读取 Streamflow 数组，失败返回 None。"""
    try:
        with xr.open_dataset(nc_path) as ds:
            return ds["Streamflow"].values.astype(float)
    except Exception as e:
        print(f"  [警告] 读取径流失败 {nc_path.name}: {e}")
        return None


def read_waterlevel(nc_path: Path) -> np.ndarray | None:
    """从 Hourly2 NC 文件读取 water_level 数组，失败返回 None。"""
    try:
        with xr.open_dataset(nc_path) as ds:
            if "water_level" not in ds.data_vars:
                return None
            return ds["water_level"].values.astype(float)
    except Exception as e:
        print(f"  [警告] 读取水位失败 {nc_path.name}: {e}")
        return None


# ==================== 主流程 ====================

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("CAMELSH 数据集缺失情况统计")
    print("=" * 70)

    # ---------- 1. 收集流域列表 ----------
    print("\n[1] 扫描数据文件...")

    ts_files   = sorted(TIMESERIES_DIR.glob("*.nc"))
    hl2_files  = sorted(HOURLY2_DIR.glob("*_hourly.nc"))

    ts_basins  = {f.stem: f for f in ts_files}                          # basin_id -> Path
    hl2_basins = {f.stem.replace("_hourly", ""): f for f in hl2_files}  # basin_id -> Path

    all_basins = sorted(set(ts_basins) | set(hl2_basins))

    print(f"  timeseries 目录：{len(ts_basins)} 个 NC 文件")
    print(f"  Hourly2    目录：{len(hl2_basins)} 个 NC 文件")
    print(f"  并集流域数量  ：{len(all_basins)} 个")

    # ---------- 2. 逐流域统计 ----------
    print("\n[2] 逐流域统计缺失情况（可能需要数分钟）...")

    records_q = []   # 径流统计
    records_h = []   # 水位统计

    for basin_id in tqdm(all_basins, desc="统计进度", unit="流域"):

        # --- 径流 ---
        if basin_id in ts_basins:
            arr_q = read_streamflow(ts_basins[basin_id])
            if arr_q is not None:
                st = compute_missing_stats(arr_q)
                records_q.append({
                    "basin_id":          basin_id,
                    "total_steps":       st["total_steps"],
                    "missing_steps":     st["missing_steps"],
                    "missing_rate_pct":  st["missing_rate_pct"],
                    "num_gaps":          st["num_gaps"],
                    "max_gap_hours":     st["max_gap_hours"],
                    "mean_gap_hours":    st["mean_gap_hours"],
                    "median_gap_hours":  st["median_gap_hours"],
                    "max_gap_days":      st["max_gap_days"],
                    "mean_gap_days":     st["mean_gap_days"],
                    # 将各段长度存为分号分隔字符串，节省列数
                    "gap_lengths_hours": ";".join(map(str, st["gap_lengths_hours"])),
                })

        # --- 水位 ---
        if basin_id in hl2_basins:
            arr_h = read_waterlevel(hl2_basins[basin_id])
            if arr_h is not None:
                st = compute_missing_stats(arr_h)
                records_h.append({
                    "basin_id":          basin_id,
                    "total_steps":       st["total_steps"],
                    "missing_steps":     st["missing_steps"],
                    "missing_rate_pct":  st["missing_rate_pct"],
                    "num_gaps":          st["num_gaps"],
                    "max_gap_hours":     st["max_gap_hours"],
                    "mean_gap_hours":    st["mean_gap_hours"],
                    "median_gap_hours":  st["median_gap_hours"],
                    "max_gap_days":      st["max_gap_days"],
                    "mean_gap_days":     st["mean_gap_days"],
                    "gap_lengths_hours": ";".join(map(str, st["gap_lengths_hours"])),
                })

    # ---------- 3. 整理为 DataFrame ----------
    df_q = pd.DataFrame(records_q)
    df_h = pd.DataFrame(records_h)

    # ---------- 4. 保存 CSV ----------
    out_q = OUTPUT_DIR / "missing_stats_streamflow.csv"
    out_h = OUTPUT_DIR / "missing_stats_waterlevel.csv"

    df_q.to_csv(out_q, index=False, encoding="utf-8-sig")
    df_h.to_csv(out_h, index=False, encoding="utf-8-sig")

    print(f"\n  已保存径流统计：{out_q}")
    print(f"  已保存水位统计：{out_h}")

    # ---------- 5. 打印汇总报告 ----------
    report_lines = generate_report(df_q, df_h, len(ts_basins), len(hl2_basins))
    report_text = "\n".join(report_lines)
    print("\n" + report_text)

    out_report = OUTPUT_DIR / "missing_stats_report.txt"
    with open(out_report, "w", encoding="utf-8") as f:
        f.write(report_text + "\n")
    print(f"\n  文字报告已保存：{out_report}")


def generate_report(df_q: pd.DataFrame, df_h: pd.DataFrame,
                    n_ts: int, n_hl2: int) -> list[str]:
    """生成汇总报告文本（返回行列表）。"""
    lines = []
    lines.append("=" * 70)
    lines.append("CAMELSH 缺失统计汇总报告")
    lines.append("=" * 70)

    for label, df, n_files in [("径流 (Streamflow)", df_q, n_ts),
                                ("水位 (Water Level)", df_h, n_hl2)]:
        lines.append(f"\n【{label}】")
        lines.append(f"  NC 文件数             : {n_files}")
        lines.append(f"  有效流域数            : {len(df)}")

        if len(df) == 0:
            lines.append("  (无有效数据)")
            continue

        # 全缺失流域（缺失率 100%）
        all_missing = df[df["missing_rate_pct"] >= 100.0]
        has_data    = df[df["missing_rate_pct"] < 100.0]

        lines.append(f"  全部缺失的流域数      : {len(all_missing)}")
        lines.append(f"  有有效数据的流域数    : {len(has_data)}")

        if len(has_data) == 0:
            continue

        lines.append(f"\n  -- 缺失率（仅含有有效数据的流域）--")
        lines.append(f"  均值                  : {has_data['missing_rate_pct'].mean():.2f} %")
        lines.append(f"  中位数                : {has_data['missing_rate_pct'].median():.2f} %")
        lines.append(f"  最小值                : {has_data['missing_rate_pct'].min():.2f} %")
        lines.append(f"  最大值                : {has_data['missing_rate_pct'].max():.2f} %")

        # 缺失率分布
        bins = [0, 5, 10, 20, 30, 50, 100]
        labels = ["0-5%", "5-10%", "10-20%", "20-30%", "30-50%", "50-100%"]
        has_data = has_data.copy()
        has_data["rate_bin"] = pd.cut(
            has_data["missing_rate_pct"], bins=bins, labels=labels,
            include_lowest=True, right=True
        )
        dist = has_data["rate_bin"].value_counts().reindex(labels, fill_value=0)
        lines.append(f"\n  缺失率分布：")
        for lbl, cnt in dist.items():
            bar = "█" * int(cnt / max(dist.max(), 1) * 30)
            lines.append(f"    {lbl:10s} : {cnt:5d} 流域  {bar}")

        # 连续缺失时长（仅有缺失段的流域）
        gapped = has_data[has_data["num_gaps"] > 0]
        lines.append(f"\n  -- 连续缺失时长（共 {len(gapped)} 个流域存在缺失段）--")
        if len(gapped) > 0:
            lines.append(f"  最大连续缺失 均值     : {gapped['max_gap_days'].mean():.1f} 天")
            lines.append(f"  最大连续缺失 中位数   : {gapped['max_gap_days'].median():.1f} 天")
            lines.append(f"  最大连续缺失 最大值   : {gapped['max_gap_days'].max():.1f} 天  "
                         f"(流域: {gapped.loc[gapped['max_gap_days'].idxmax(), 'basin_id']})")
            lines.append(f"  平均连续缺失 均值     : {gapped['mean_gap_days'].mean():.1f} 天")

            # 最大连续缺失时长分布（天）
            bins2  = [0, 1, 3, 7, 14, 30, 90, 180, 365, float("inf")]
            lbls2  = ["<1天", "1-3天", "3-7天", "7-14天", "14-30天",
                      "30-90天", "90-180天", "180-365天", ">365天"]
            gapped = gapped.copy()
            gapped["max_gap_bin"] = pd.cut(
                gapped["max_gap_days"], bins=bins2, labels=lbls2,
                include_lowest=True, right=True
            )
            dist2 = gapped["max_gap_bin"].value_counts().reindex(lbls2, fill_value=0)
            lines.append(f"\n  最大连续缺失分布（按天）：")
            for lbl, cnt in dist2.items():
                bar = "█" * int(cnt / max(dist2.max(), 1) * 30)
                lines.append(f"    {lbl:12s} : {cnt:5d} 流域  {bar}")

            # 缺失段数量分布
            lines.append(f"\n  缺失段数量统计：")
            lines.append(f"  均值                  : {gapped['num_gaps'].mean():.1f} 段/流域")
            lines.append(f"  中位数                : {gapped['num_gaps'].median():.1f} 段/流域")
            lines.append(f"  最多                  : {gapped['num_gaps'].max()} 段  "
                         f"(流域: {gapped.loc[gapped['num_gaps'].idxmax(), 'basin_id']})")

    lines.append("\n" + "=" * 70)
    return lines


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n用户中断操作")
        sys.exit(1)
    except Exception as e:
        import traceback
        print(f"\n发生错误: {e}")
        traceback.print_exc()
        sys.exit(1)
