"""检验无公开 STGQ 曲线候选站的 H→Q 关系是否弱于匹配的 STGQ 对照。

这不是模型预测实验，而是新增流域之前的资格核查。每个候选站按面积、气候和
地理位置匹配 5 个 STGQ 站；每个站均以时间前 70% 的成对 H/Q 拟合分箱率定表，
以后 30% 的真实 H 预测真实 Q，并计算 NSE。若候选站显著较低，说明其 H→Q
关系确实不同于普通水位-流量率定站，但仍不能单凭此确认具体的 USGS 测流方法。
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
from scipy import stats as sps

import sys

_ROOT = Path(__file__).resolve().parents[2]
for _path in (str(_ROOT / "src"), str(_ROOT)):
    if _path not in sys.path:
        sys.path.insert(0, _path)

from evaluation.rating_baseline import apply_rating, fit_rating, nse  # noqa: E402
from pipeline.paths import RESULTS_ROOT  # noqa: E402


SUMMARY_DIR = RESULTS_ROOT / "summary"
CANDIDATE_TABLE = SUMMARY_DIR / "nonrating_basin_candidates_run1.csv"
CACHE_DIR = _ROOT / "data" / "usgs_ratings_cache"
HOURLY_DIR = Path("F:/data/CAMELSH/Hourly2/Hourly2")
ATTR_DIR = Path("F:/data/CAMELSH/attributes")

N_CONTROLS = 5
N_NEAREST_TO_TRY = 40
N_BINS = 50
MIN_PAIRS = 1_000

_RE_NO_FILES = re.compile(r"NUMBER OF FILES RETURNED BY QUERY\s*=\s*(\d+)")

# 面积优先；其余变量用于避免把不同气候、不同地理背景的河流硬配在一起。
MATCH_COLUMNS = (
    "log_drain_sqkm", "pptavg_basin", "t_avg_basin", "snow_pct_precip",
    "precip_seas_ind", "lat_gage", "lng_gage",
)
MATCH_WEIGHTS = np.array((2.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0))


def _station_id(values: pd.Series) -> pd.Series:
    """保留前导零；9 位站号不截断。"""
    return values.astype(str).str.strip().str.replace(".0", "", regex=False).str.zfill(8)


def stgq_basins() -> set[str]:
    """从已缓存的 USGS base 文件识别有公开 STGQ 曲线的站。"""
    found = set()
    for path in CACHE_DIR.glob("*.base.rdb"):
        text = path.read_text(encoding="utf-8", errors="replace")
        no_files = _RE_NO_FILES.search(text)
        if no_files and no_files.group(1) == "0":
            continue
        if 'TYPE="STGQ"' in text:
            found.add(path.name.split(".")[0])
    return found


def load_matching_attributes() -> pd.DataFrame:
    """读取匹配所需的面积、气候和测站经纬度属性。"""
    basin = pd.read_csv(ATTR_DIR / "attributes_gageii_BasinID.csv", dtype={"STAID": str})
    climate = pd.read_csv(ATTR_DIR / "attributes_gageii_Climate.csv", dtype={"STAID": str})
    basin["basin"] = _station_id(basin["STAID"])
    climate["basin"] = _station_id(climate["STAID"])
    cols = ["basin", "DRAIN_SQKM", "LAT_GAGE", "LNG_GAGE"]
    climate_cols = ["basin", "PPTAVG_BASIN", "T_AVG_BASIN", "SNOW_PCT_PRECIP", "PRECIP_SEAS_IND"]
    out = basin[cols].merge(climate[climate_cols], on="basin", how="inner")
    out = out.rename(columns={
        "DRAIN_SQKM": "drain_sqkm", "LAT_GAGE": "lat_gage", "LNG_GAGE": "lng_gage",
        "PPTAVG_BASIN": "pptavg_basin", "T_AVG_BASIN": "t_avg_basin",
        "SNOW_PCT_PRECIP": "snow_pct_precip", "PRECIP_SEAS_IND": "precip_seas_ind",
    })
    out["log_drain_sqkm"] = np.log10(out["drain_sqkm"].where(out["drain_sqkm"] > 0))
    return out.dropna(subset=list(MATCH_COLUMNS)).set_index("basin")


def hq_nse(basin: str) -> tuple[float, int, int, str | None]:
    """按该站联合有效记录的时间跨度前 70% 拟合、后 30% 评估 H→Q 关系。"""
    path = HOURLY_DIR / f"{basin}_hourly.nc"
    if not path.exists():
        return float("nan"), 0, 0, "缺少 Hourly2 文件"
    try:
        with xr.open_dataset(path) as ds:
            q = ds["streamflow"].values.astype(float)
            h = ds["water_level"].values.astype(float)
    except Exception as exc:
        return float("nan"), 0, 0, f"读取失败：{type(exc).__name__}"

    paired = np.isfinite(q) & np.isfinite(h)
    if not paired.any():
        return float("nan"), 0, 0, "没有 H/Q 联合有效记录"

    # 新站的水位记录可能晚于 1980 年开始。以首末联合有效记录定义可评估时间跨度，
    # 与论文的“以 Q 与 H 同时有效时段为参考跨度”划分原则一致；中间缺口仍保留，
    # 不会把不相邻时段拼接起来。
    first = int(np.flatnonzero(paired)[0])
    last = int(np.flatnonzero(paired)[-1])
    cut = first + int(0.7 * (last - first + 1))
    train = paired[first:cut]
    test = paired[cut:last + 1]
    n_train, n_test = int(train.sum()), int(test.sum())
    if n_train < MIN_PAIRS or n_test < MIN_PAIRS:
        return float("nan"), n_train, n_test, "训练或测试配对不足"
    fit = fit_rating(h[first:cut][train], q[first:cut][train], n_bins=N_BINS)
    if fit is None:
        return float("nan"), n_train, n_test, "无法拟合分箱曲线"
    return nse(q[cut:last + 1][test], apply_rating(fit, h[cut:last + 1][test])), n_train, n_test, None


def choose_controls(candidate: str, attrs: pd.DataFrame, stgq: set[str],
                    cache: dict[str, tuple[float, int, int, str | None]]) -> list[str]:
    """按加权标准化距离找 5 个实际拥有可评估 H/Q 数据的 STGQ 对照。"""
    if candidate not in attrs.index:
        return []
    pool_ids = sorted(stgq.intersection(attrs.index))
    pool = attrs.loc[pool_ids, list(MATCH_COLUMNS)]
    center = attrs.loc[candidate, list(MATCH_COLUMNS)]
    mu, sd = pool.mean(), pool.std(ddof=0).replace(0, 1)
    distance = np.sqrt((((pool - center) / sd * MATCH_WEIGHTS) ** 2).sum(axis=1))

    selected = []
    for basin in distance.sort_values().head(N_NEAREST_TO_TRY).index:
        if basin not in cache:
            cache[basin] = hq_nse(basin)
        score, *_ = cache[basin]
        if np.isfinite(score):
            selected.append(basin)
        if len(selected) == N_CONTROLS:
            break
    return selected


def run() -> tuple[pd.DataFrame, pd.DataFrame]:
    if not CANDIDATE_TABLE.exists():
        raise FileNotFoundError(f"缺少候选表：{CANDIDATE_TABLE}")
    candidates = pd.read_csv(CANDIDATE_TABLE, dtype={"basin": str})
    candidate_ids = candidates.loc[candidates["selected"].fillna(False), "basin"].tolist()
    attrs = load_matching_attributes()
    stgq = stgq_basins()
    score_cache: dict[str, tuple[float, int, int, str | None]] = {}
    rows, control_rows = [], []

    for i, basin in enumerate(candidate_ids, 1):
        print(f"[{i:02d}/{len(candidate_ids)}] {basin}", flush=True)
        score_cache.setdefault(basin, hq_nse(basin))
        candidate_nse, n_train, n_test, error = score_cache[basin]
        controls = choose_controls(basin, attrs, stgq, score_cache)
        control_scores = [score_cache[c][0] for c in controls]
        median_control = float(np.median(control_scores)) if control_scores else float("nan")
        rows.append({
            "candidate_basin": basin, "candidate_hq_nse": candidate_nse,
            "candidate_train_pairs": n_train, "candidate_test_pairs": n_test,
            "n_controls": len(controls), "matched_control_median_hq_nse": median_control,
            "difference_candidate_minus_control": candidate_nse - median_control,
            "candidate_error": error,
        })
        for rank, control in enumerate(controls, 1):
            c_score, c_train, c_test, c_error = score_cache[control]
            control_rows.append({
                "candidate_basin": basin, "control_rank": rank, "control_basin": control,
                "candidate_hq_nse": candidate_nse, "control_hq_nse": c_score,
                "difference_candidate_minus_control": candidate_nse - c_score,
                "control_train_pairs": c_train, "control_test_pairs": c_test,
                "control_error": c_error,
            })
    return pd.DataFrame(rows), pd.DataFrame(control_rows)


def report(pairs: pd.DataFrame) -> None:
    valid = pairs.dropna(subset=["difference_candidate_minus_control"])
    diff = valid["difference_candidate_minus_control"].to_numpy()
    print("\n候选站 − 匹配 STGQ 对照中位数：")
    print(f"  可配对候选站：{len(valid)}/{len(pairs)}")
    print(f"  候选 H→Q NSE：中位 {valid.candidate_hq_nse.median():.4f}")
    print(f"  匹配对照中位 NSE：中位 {valid.matched_control_median_hq_nse.median():.4f}")
    print(f"  配对差值：均值 {diff.mean():+.4f}，中位 {np.median(diff):+.4f}")
    if len(diff) >= 5:
        two_sided = sps.wilcoxon(diff, alternative="two-sided").pvalue
        lower = sps.wilcoxon(diff, alternative="less").pvalue
        print(f"  Wilcoxon 双侧 p={two_sided:.4g}；候选更低（单侧）p={lower:.4g}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=SUMMARY_DIR)
    args = parser.parse_args()
    pairs, controls = run()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    pairs.to_csv(args.output_dir / "candidate_stgq_matched_pairs.csv", index=False, encoding="utf-8-sig")
    controls.to_csv(args.output_dir / "candidate_stgq_matched_controls.csv", index=False, encoding="utf-8-sig")
    report(pairs)
    print(f"\n已写入 {args.output_dir / 'candidate_stgq_matched_pairs.csv'}")
    print(f"已写入 {args.output_dir / 'candidate_stgq_matched_controls.csv'}")


if __name__ == "__main__":
    main()
