"""保留测量时实际尺度的断面输入；不读取流量、流速或评级曲线。"""

from pathlib import Path
import json

import numpy as np
import pandas as pd

from .features import file_sha256


SCALE_COLUMNS = ["survey_log_area_m2", "survey_log_wet_length_m", "survey_log_width_m"]
STAGE_COLUMN = "survey_stage_z"


def measured_geometry(payload):
    """读取白名单几何字段；湿润长度只标记为湿周代理，不补造两岸。"""
    record = payload["record"]
    units = payload.get("units", {})
    expected = {"NXSDB_Area_sf": "ft^2", "NXSDB_WettedDistance_ft": "ft",
                "NXSDB_Width_ft": "ft", "NWIS_GageHeight_ft": "ft"}
    if any(units.get(k) != unit for k, unit in expected.items()):
        raise ValueError("测量几何字段缺少明确单位或单位不一致")
    area = float(record["NXSDB_Area_sf"]) * 0.09290304
    wet = float(record["NXSDB_WettedDistance_ft"]) * 0.3048
    width = float(record["NXSDB_Width_ft"]) * 0.3048
    stage = float(record["NWIS_GageHeight_ft"]) * 0.3048
    if not np.isfinite([area, wet, width, stage]).all() or min(area, wet, width) <= 0:
        raise ValueError("测量面积、湿润长度、宽度必须为有限正数，水位必须有限")
    points = pd.DataFrame(payload["points"])[["Distance_ft", "Depth_ft"]].apply(pd.to_numeric, errors="coerce")
    points = points[np.isfinite(points).all(axis=1) & (points.Depth_ft >= 0)]
    profile = points.groupby("Distance_ft").Depth_ft.median().sort_index()
    x = profile.index.to_numpy(float) * 0.3048
    depth = profile.to_numpy(float) * 0.3048
    if len(x) < 3 or x[-1] <= x[0]:
        raise ValueError("原始剖面不能独立核验几何尺度")
    # 面积积分使用原始深度，不能先减最小深度；仅作来源核验，不替换官方定义。
    return {"survey_log_area_m2": np.log(area), "survey_log_wet_length_m": np.log(wet),
            "survey_log_width_m": np.log(width), "survey_stage_m": stage,
            "area_m2": area, "sampled_wet_length_m": wet, "width_m": width,
            "radius_proxy_m": area / wet, "raw_min_depth_m": float(depth.min()),
            "raw_max_depth_m": float(depth.max()),
            "raw_profile_integral_m2": float(np.trapz(depth, x)),
            "raw_profile_wet_length_m": float(np.hypot(np.diff(x), np.diff(depth)).sum()),
            "left_endpoint_depth_m": float(depth[0]), "right_endpoint_depth_m": float(depth[-1])}


def load_survey_geometry(source, prepared, project_root):
    """复用已冻结的81个剖面选择，不按新结果重新挑选日期或测线。"""
    source = Path(source)
    audit = pd.read_csv(source / "geometry_audit.csv", dtype={"basin": str})
    audit = audit.set_index("basin").loc[prepared.basins]
    roots = [Path(project_root) / "data/cross_sections",
             Path(project_root) / "research_notes/USGS 原86 历年断面核验"]
    rows = []
    for basin, record in audit.iterrows():
        row = {"basin": basin, "geom_missing": int(not record.available),
               "profile_id": record.profile_id, "measurement_date": record.measurement_date,
               "location_code": record.location_code, "source_json": record.json_path,
               "source_sha256": record.source_sha256}
        if record.available:
            candidates = [root / record.json_path for root in roots]
            path = next((p for p in candidates if p.is_file()), None)
            if path is None or file_sha256(path) != record.source_sha256:
                raise ValueError(f"冻结剖面缺失或来源哈希变化: {basin}")
            payload = json.loads(path.read_text(encoding="utf-8-sig"))
            if str(payload["station"]) != basin:
                raise ValueError("剖面站号与冻结名单不一致")
            row.update(measured_geometry(payload))
            bi = prepared.basin_index[basin]
            # 测量时站点水位只作统计条件，不将其硬映射成上下游剖面的水面高程。
            row[STAGE_COLUMN] = ((row["survey_stage_m"] - prepared.target_mean["waterlevel"][bi])
                                 / prepared.target_std["waterlevel"][bi])
        rows.append(row)
    return pd.DataFrame(rows).set_index("basin")


def fit_survey_scaler(table, fit_basins, columns):
    """候选模型仅在内部训练站拟合填补和标准化，重拟合才使用全部保留站。"""
    fitted = table.loc[fit_basins, columns].to_numpy(float)
    fill, mean, std = [], [], []
    for values in fitted.T:
        values = values[np.isfinite(values)]
        fill.append(float(np.median(values)) if len(values) else 0.0)
        mean.append(float(values.mean()) if len(values) else 0.0)
        deviation = float(values.std()) if len(values) else 1.0
        std.append(deviation if deviation > 1e-8 else 1.0)
    return {"columns": list(columns), "fit_basins": list(fit_basins),
            "fill": fill, "mean": mean, "std": std}


def survey_inputs(table, basins, scaler, shuffle_seed=None):
    """连续几何与测量水位共同打乱；缺失站和缺失标记始终保持。"""
    values = table.loc[basins, scaler["columns"]].to_numpy(float)
    values = np.where(np.isfinite(values), values, np.asarray(scaler["fill"]))
    values = (values - np.asarray(scaler["mean"])) / np.asarray(scaler["std"])
    missing = table.loc[basins, "geom_missing"].to_numpy(float)
    if shuffle_seed is not None:
        valid = np.flatnonzero(missing == 0)
        order = np.random.default_rng(shuffle_seed).permutation(valid)
        if len(valid) > 1 and np.array_equal(order, valid):
            order = np.roll(valid, 1)
        values[valid] = values[order]
    result = np.column_stack([values, missing]).astype(np.float32)
    if not np.isfinite(result).all():
        raise ValueError("测量几何输入存在非有限值")
    return result
