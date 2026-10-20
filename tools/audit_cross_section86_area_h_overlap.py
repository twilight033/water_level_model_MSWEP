"""只读核验原86历史面积与本地小时水位的配对覆盖，不读取Q或流速。"""

from pathlib import Path
import ast
import gzip
import json

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
HISTORY = ROOT / "research_notes/USGS 原86 历年断面核验/channel_history"
OUTPUT = ROOT / "reports/cross_section86_area_h_overlap"


def records(name):
    with gzip.open(HISTORY / name, "rt", encoding="utf-8") as stream:
        return [json.loads(line) for line in stream]


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    records_area = pd.DataFrame(records("channel_summary.jsonl.gz"))
    context = records("measurement_context.jsonl.gz")
    location = pd.DataFrame([{"record_id": item["record_id"],
        **{key: item["location"].get(key) for key in (
            "channel_name", "channel_location_distance", "channel_location_distance_unit", "channel_location_direction")},
        "measurement_type": item["quality_and_condition"].get("measurement_type"),
        "channel_measurement_type": item["quality_and_condition"].get("channel_measurement_type")}
        for item in context])
    area = records_area.merge(location, on="record_id", validate="one_to_one")
    area["area_numeric"] = pd.to_numeric(area.channel_area, errors="coerce")
    positive = np.isfinite(area.area_numeric) & (area.area_numeric > 0)
    area = area[positive].copy()
    factors = {"ft^2": 0.09290304, "m^2": 1.0}
    area["unit_factor"] = area.channel_area_unit.map(factors)
    unknown_units = int(area.unit_factor.isna().sum())
    area = area[area.unit_factor.notna()].copy()
    area["area_m2"] = area.area_numeric * area.unit_factor
    area["measurement_time"] = pd.to_datetime(area.time, utc=True, format="ISO8601").dt.tz_convert(None)
    h = pd.read_parquet(ROOT / "data/camelsh_exported/waterlevel_hourly_86.parquet")
    h.index = pd.DatetimeIndex(h.index)
    if not h.index.is_unique or not h.index.is_monotonic_increasing:
        raise ValueError("本地水位时间轴不唯一或未排序")
    splits = json.loads((ROOT / "data/splits/basin_splits.json").read_text(encoding="utf-8-sig"))["splits"]
    # 原站号文件保存的是Python列表赋值，不能按每行一个站号读取。
    ids = ast.literal_eval((ROOT / "86basin_ids.txt").read_text(encoding="utf-8-sig").split("=", 1)[1].strip())
    if len(ids) != 86 or set(ids) != set(h.columns) or set(area.basin) != set(ids):
        raise ValueError("面积或水位不是原86站总体")
    rows, station_rows = [], []
    for basin in ids:
        current = area[area.basin == basin].copy()
        times = pd.DatetimeIndex(current.measurement_time)
        values = h[basin].to_numpy(float)
        n = len(current)
        nearest = h.index.get_indexer(times, method="nearest", tolerance=pd.Timedelta("30min"))
        nearest_h = np.full(n, np.nan)
        nearest_dt = np.full(n, np.nan)
        found = nearest >= 0
        nearest_h[found] = values[nearest[found]]
        nearest_dt[found] = np.abs((h.index[nearest[found]] - times[found]).total_seconds())

        # 只允许准确整点或相邻两小时均有观测时插值，不跨缺口、不外推。
        exact = h.index.get_indexer(times)
        exact_h = np.full(n, np.nan)
        at_grid = exact >= 0
        exact_h[at_grid] = values[exact[at_grid]]
        interpolated = exact_h.copy()
        right = h.index.searchsorted(times, side="left")
        candidates = np.flatnonzero((exact < 0) & (right > 0) & (right < len(h)))
        if len(candidates):
            ri = right[candidates]
            li = ri - 1
            duration = (h.index[ri] - h.index[li]).total_seconds().to_numpy()
            left_delta = (times[candidates] - h.index[li]).total_seconds().to_numpy()
            valid = (duration == 3600) & np.isfinite(values[li]) & np.isfinite(values[ri])
            indices = candidates[valid]
            weight = left_delta[valid] / duration[valid]
            interpolated[indices] = values[li[valid]] * (1 - weight) + values[ri[valid]] * weight
        current["h_nearest_30min_m"] = nearest_h
        current["nearest_offset_sec"] = nearest_dt
        current["h_exact_hour_m"] = exact_h
        current["h_at_measurement_interpolated_m"] = interpolated
        period = np.full(n, "outside_model_splits_or_embargo", dtype=object)
        for split in ("train", "valid", "test"):
            lo, hi = pd.Timestamp(splits[basin][split + "_start"]), pd.Timestamp(splits[basin][split + "_end"])
            period[(times >= lo) & (times <= hi)] = split
        current["model_period"] = period
        current["pairing_method"] = np.where(np.isfinite(exact_h), "exact_hour",
            np.where(np.isfinite(interpolated), "linear_between_valid_adjacent_hours", "unpaired"))
        strict = current[np.isfinite(current.h_at_measurement_interpolated_m)]
        train = strict[strict.model_period == "train"]
        group_cols = ["channel_name", "channel_location_direction", "channel_location_distance",
                      "channel_location_distance_unit", "channel_measurement_type"]
        signatures = train[group_cols].fillna("").astype(str).drop_duplicates()
        row = {"basin": basin, "area_positive_records": n,
            "area_distinct_field_visits": current.field_visit_id.nunique(),
            "area_first_time": current.measurement_time.min().isoformat(),
            "area_last_time": current.measurement_time.max().isoformat(),
            "n_within_h_time_axis": int(((times >= h.index.min()) & (times <= h.index.max())).sum()),
            "n_nearest_h_30min": int(np.isfinite(nearest_h).sum()),
            "n_exact_hour_h": int(np.isfinite(exact_h).sum()),
            "n_strict_interpolated_h": len(strict),
            "n_unique_pair_times": strict.measurement_time.nunique(),
            "n_unique_paired_h": strict.h_at_measurement_interpolated_m.nunique(),
            "n_train_pairs": len(train), "n_train_field_visits": train.field_visit_id.nunique(),
            "n_train_unique_h": train.h_at_measurement_interpolated_m.nunique(),
            "n_train_location_method_signatures": len(signatures),
            "n_valid_pairs": int((strict.model_period == "valid").sum()),
            "n_test_pairs": int((strict.model_period == "test").sum()),
            "n_other_period_pairs": int((strict.model_period == "outside_model_splits_or_embargo").sum())}
        train_h = h[basin].loc[splits[basin]["train_start"]:splits[basin]["train_end"]].dropna()
        row["n_train_h_observations"] = len(train_h)
        row["train_h_min_m"] = train_h.min()
        row["train_h_max_m"] = train_h.max()
        if len(train):
            lo, hi = train.h_at_measurement_interpolated_m.min(), train.h_at_measurement_interpolated_m.max()
            row.update(train_area_paired_h_min_m=lo, train_area_paired_h_max_m=hi,
                fraction_train_h_inside_paired_stage_range=float(train_h.between(lo, hi).mean()))
        else:
            row.update(train_area_paired_h_min_m=np.nan, train_area_paired_h_max_m=np.nan,
                       fraction_train_h_inside_paired_stage_range=np.nan)
        rows.append(current)
        station_rows.append(row)
    paired = pd.concat(rows, ignore_index=True)
    station = pd.DataFrame(station_rows)
    paired.to_csv(OUTPUT / "area_h_pairs_inventory.csv", index=False, encoding="utf-8-sig")
    station.to_csv(OUTPUT / "station_coverage.csv", index=False, encoding="utf-8-sig")
    strict = paired[np.isfinite(paired.h_at_measurement_interpolated_m)]
    summary = {"n_basins": 86, "n_positive_area_records": int(positive.sum()),
        "n_unknown_area_units_excluded": unknown_units,
        "n_area_records_usable_units": len(paired),
        "n_area_records_within_h_time_axis": int(station.n_within_h_time_axis.sum()),
        "n_exact_hour_h_pairs": int(station.n_exact_hour_h.sum()),
        "n_nearest_h_30min_pairs": int(station.n_nearest_h_30min.sum()),
        "n_strict_interpolated_h_pairs": len(strict),
        "n_train_pairs": int(station.n_train_pairs.sum()),
        "n_valid_pairs": int(station.n_valid_pairs.sum()), "n_test_pairs": int(station.n_test_pairs.sum()),
        "n_other_period_pairs": int(station.n_other_period_pairs.sum()),
        "stations_with_area": int((station.area_positive_records > 0).sum()),
        "stations_with_any_strict_h_pair": int((station.n_strict_interpolated_h > 0).sum()),
        "stations_with_train_pairs": int((station.n_train_pairs > 0).sum()),
        "stations_with_5_train_pairs": int((station.n_train_pairs >= 5).sum()),
        "stations_with_20_train_pairs": int((station.n_train_pairs >= 20).sum()),
        "stations_with_5_unique_train_h": int((station.n_train_unique_h >= 5).sum()),
        "stations_with_20_unique_train_h": int((station.n_train_unique_h >= 20).sum()),
        "train_pairs_per_station_min_median_max": [int(station.n_train_pairs.min()),
            float(station.n_train_pairs.median()), int(station.n_train_pairs.max())],
        "paired_training_stage_range_coverage_min_median_max": [float(station.fraction_train_h_inside_paired_stage_range.min()),
            float(station.fraction_train_h_inside_paired_stage_range.median()),
            float(station.fraction_train_h_inside_paired_stage_range.max())],
        "stations_train_location_method_signatures_gt1": int((station.n_train_location_method_signatures > 1).sum()),
        "read_discharge_or_velocity_values": False,
        "interpolation_policy": "exact_hour_or_two_valid_adjacent_hourly_h_no_gap_no_extrapolation",
        "same_section_and_quality_verified": False,
        "records_are_not_independent_field_visits": True,
        "h_observations_are_in_meters": True,
        "curve_readiness_not_inferred_from_pair_counts": True}
    (OUTPUT / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    sections = ["# 原86历史面积与本地水位配对覆盖核验", "",
        "只读历史面积、位置元数据与小时水位；没有读取Q或流速，没有改动运行中的转换器。", "",
        "配对采用UTC时间。同一站准确整点可直接配对；非整点必须有相邻两小时的有效H，才进行线性插值。没有跨水位缺口或时间轴边界补值。", "",
        "```json", json.dumps(summary, ensure_ascii=False, indent=2), "```", "",
        "station_coverage.csv为逐站明细，area_h_pairs_inventory.csv保留每次面积的H匹配、日期、位置及方法。", "",
        "本核验仅证明时间和数值可以配对。尚未确认测量来自同一固定断面，也未完成测量质量、主河道/支汊、河床变化筛查，不能把配对数量直接等同于可靠A(H)曲线数量。", "",
        "训练期水位范围覆盖率只计算全部训练H中落在训练期H-A配对最小至最大H之间的比例；不同测线混合时，该比例不代表同一断面的有效覆盖。测试期面积没有被用来拟合曲线。"]
    (OUTPUT / "review.md").write_text("\n".join(sections) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print("\n训练配对最少的站：")
    print(station.sort_values("n_train_pairs").head(10)[
        ["basin", "area_positive_records", "n_strict_interpolated_h", "n_train_pairs", "n_train_unique_h"]].to_string(index=False))


if __name__ == "__main__":
    main()
