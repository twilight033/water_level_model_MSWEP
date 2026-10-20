"""核验稀疏实测面积对原86站训练时间步和输入窗口的覆盖，不拟合曲线。"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "reports/cross_section86_area_h_overlap/area_h_pairs_inventory.csv"
OUTPUT = ROOT / "reports/cross_section86_area_time_coverage"


def load_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    config = load_json(ROOT / "configs/cross_section86_strong_physical.json")
    splits = load_json(ROOT / "data/splits/basin_splits.json")["splits"]
    basins = list(load_json(ROOT / "results/cross_section86_survey_refit_v1/experiment.json")["basins"])
    sequence, stride = config["sequence_length"], config["train_stride"]
    assert sequence == 480 and stride == 8 and len(basins) == 86
    area = pd.read_csv(SOURCE, dtype={"basin": str, "field_visit_id": str}, low_memory=False)
    area = area[np.isfinite(area.area_m2) & (area.area_m2 > 0)].copy()
    area["instant"] = pd.to_datetime(area.time, utc=True, format="ISO8601").dt.tz_convert(None)
    grid = None
    for name in ("mswep_precip_3h_86.parquet", "forcing_temperature_mean_3h_86.parquet",
                 "forcing_solar_radiation_3h_86.parquet"):
        index = pd.read_parquet(ROOT / "data/camelsh_exported" / name, columns=[]).index
        grid = index if grid is None else grid.intersection(index)
    grid = pd.DatetimeIndex(grid).sort_values()
    h = pd.read_parquet(ROOT / "data/camelsh_exported/waterlevel_hourly_86.parquet").reindex(index=grid, columns=basins)
    continuous = np.r_[True, np.diff(grid.to_numpy()) == np.timedelta64(3, "h")]
    discontinuities = np.r_[0, np.cumsum(~continuous)]
    rows = []
    for basin in basins:
        start, end = pd.Timestamp(splits[basin]["train_start"]), pd.Timestamp(splits[basin]["train_end"])
        measurements = area[(area.basin == basin) & area.instant.between(start, end)].copy()
        assert measurements.model_period.eq("train").all()
        paired = measurements[np.isfinite(measurements.h_at_measurement_interpolated_m)]
        lo, hi = grid.searchsorted(start, "left"), grid.searchsorted(end, "right") - 1
        exact = grid.get_indexer(measurements.instant)
        nearest = grid.get_indexer(measurements.instant, method="nearest", tolerance=pd.Timedelta(minutes=90))
        exact = np.unique(exact[exact >= 0])
        nearest = np.unique(nearest[nearest >= 0])
        area_present = np.zeros(len(grid), bool)
        area_present[nearest] = True
        area_prefix = np.r_[0, np.cumsum(area_present)]
        h_valid = np.isfinite(h[basin].to_numpy())
        common = {"basin": basin, "train_start": start.isoformat(), "train_end": end.isoformat(),
            "n_area_records_in_train": len(measurements), "n_area_field_visits_in_train": measurements.field_visit_id.nunique(),
            "n_area_unique_instants_in_train": measurements.instant.nunique(), "n_train_h_area_pairs": len(paired),
            "n_train_paired_visits": paired.field_visit_id.nunique(), "n_area_days_in_train": measurements.instant.dt.floor("D").nunique()}
        for name, step in (("all_3h", 1), ("stride8", stride)):
            positions = np.arange(max(lo, sequence), hi + 1, step, dtype=np.int64)
            positions = positions[continuous[positions] & (discontinuities[positions] == discontinuities[positions - sequence])]
            for population, targets in (("all_inputs", positions), ("h_labeled", positions[h_valid[positions]])):
                suffix = f"{population}_{name}"
                exact_hits = np.intersect1d(targets, exact)
                nearest_hits = np.intersect1d(targets, nearest)
                counts_in_window = area_prefix[targets] - area_prefix[targets - sequence]
                common.update({f"n_targets_{suffix}": len(targets),
                    f"n_exact_area_targets_{suffix}": len(exact_hits),
                    f"n_nearest90min_area_targets_{suffix}": len(nearest_hits),
                    f"n_windows_with_any_measured_area_{suffix}": int((counts_in_window > 0).sum()),
                    f"n_windows_with_complete_measured_area_{suffix}": int((counts_in_window == sequence).sum())})
            if name == "all_3h":
                # 测量日覆盖只是宽松诊断，不代表整天的面积都有观测。
                days = grid[positions].floor("D").unique()
                measured_days = pd.DatetimeIndex(measurements.instant.dt.floor("D").unique())
                common["n_train_calendar_days_with_input_windows"] = len(days)
                common["n_train_calendar_days_with_any_area_measurement"] = len(days.intersection(measured_days))
        rows.append(common)
    table = pd.DataFrame(rows)
    assert table.basin.nunique() == 86 and table.n_train_h_area_pairs.sum() == 6618
    sums = {column: int(table[column].sum()) for column in table if column.startswith("n_")}
    summary = {"n_basins": 86, "stations_with_train_area": int(table.n_area_records_in_train.gt(0).sum()),
        "sequence_length": sequence, "step_hours": 3, "training_stride": stride,
        "nearest_time_alignment_minutes": 90, "totals": sums,
        "per_station_paired_records_min_median_max": [int(table.n_train_h_area_pairs.min()),
            float(table.n_train_h_area_pairs.median()), int(table.n_train_h_area_pairs.max())],
        "fractions": {}, "no_area_interpolation_or_curve_fitting": True, "no_q_values_read": True,
        "input_sha256": hashlib.sha256(SOURCE.read_bytes()).hexdigest()}
    for name in ("all_3h", "stride8"):
        for population in ("all_inputs", "h_labeled"):
            suffix = f"{population}_{name}"
            denominator = sums[f"n_targets_{suffix}"]
            summary["fractions"][suffix] = {
                "exact_measured_area": sums[f"n_exact_area_targets_{suffix}"] / denominator,
                "nearest90min_measured_area": sums[f"n_nearest90min_area_targets_{suffix}"] / denominator,
                "any_area_within_60day_input_window": sums[f"n_windows_with_any_measured_area_{suffix}"] / denominator,
                "complete_area_for_60day_input_window": sums[f"n_windows_with_complete_measured_area_{suffix}"] / denominator}
    summary["measurement_day_fraction"] = sums["n_train_calendar_days_with_any_area_measurement"] / sums["n_train_calendar_days_with_input_windows"]
    table.to_csv(OUTPUT / "station_time_coverage.csv", index=False, encoding="utf-8-sig")
    (OUTPUT / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    lines = ["# 原86站实测过水面积的训练时间覆盖", "",
        "本核验只读取面积、原冻结划分、气象缓存时间轴和H标签有效性；不读取Q，不拟合面积曲线，不填补面积序列。", "",
        "站点覆盖与时间覆盖分开：某站有历史面积，不表示该站每个训练时刻有同期面积。6618条是可与H配对的训练期记录，不是全部训练期面积记录。", "",
        f"86站中{summary['stations_with_train_area']}站有训练期面积。训练期面积共{sums['n_area_records_in_train']:,}条，"
        f"对应{sums['n_area_field_visits_in_train']:,}次访问，其中{sums['n_train_h_area_pairs']:,}条可配对H。", "",
        "## 时间步覆盖", "",
        "全3h目标时刻按原480步窗口及时间缺口规则生成，不要求Q/H标签有效；H有效子集单独报告。stride8对应原训练采样步长，H有效子集对应单任务H训练样本准入。双任务和Q单任务样本量另受Q标签/掩膜影响，不能把全时间轴称为某个Q情景的实际训练样本数。", ""]
    for suffix, values in summary["fractions"].items():
        lines += [f"- {suffix}：目标时刻{sums['n_targets_' + suffix]:,}个；面积时间严格相同{sums['n_exact_area_targets_' + suffix]:,}个（{values['exact_measured_area']:.4%}）；"
            f"允许与最近3h格点相差不超过90分钟时{sums['n_nearest90min_area_targets_' + suffix]:,}个（{values['nearest90min_measured_area']:.4%}）。"]
    lines += ["", "90分钟仅为宽松对齐诊断，不代表面积在这段时间内不变，也未混合不同测线的面积值；同站同时间格点重复记录仅计一次。", "",
        f"输入窗口内曾经出现一次面积的覆盖率，all_inputs_all_3h为{summary['fractions']['all_inputs_all_3h']['any_area_within_60day_input_window']:.2%}；"
        f"但480个输入时间步全部有面积的窗口为{sums['n_windows_with_complete_measured_area_all_inputs_all_3h']:,}个。不能将‘60天里测过一次’当成全窗口动态面积已知。", "",
        f"即使只按测量日统计，有面积测量的站点日占训练站点日{summary['measurement_day_fraction']:.2%}；这也不代表测量日内每个小时都有面积。", "",
        "## 方法含义", "",
        "不要求必须显式拟合A(H)。面积可以作为稀疏辅助监督或静态条件；若要求每个时间步都有动态面积，就需要A(H)、可靠断面几何计算或面积预测模型等估算机制。",
        "跨测量日直接插值/前向填充属于额外估算假设，不属于实测覆盖；后来的测量用于过去的输入还可能改变因果信息边界。",
        "原两阶段的预测端若只有气象及预测H，不能假设已知实测同期面积。", "",
        "复现：`python -X utf8 tools/audit_cross_section86_area_time_coverage.py`。", ""]
    (OUTPUT / "review.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
