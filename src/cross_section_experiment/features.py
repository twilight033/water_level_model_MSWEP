"""从白名单剖面字段提取断面快照特征，不读取流量或流速作为输入。"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

FEATURE_COLUMNS = (
    "geom_log_span_m",
    "geom_log_relief_m",
    "geom_shape_mean",
    "geom_shape_center",
    "geom_deepest_offset",
    "geom_shape_asymmetry",
)
ACCEPTED_QUALITY = {"EXCL", "GOOD", "FAIR"}


def file_sha256(path: Path) -> str:
    """用流式读取记录来源，避免大缓存一次进入内存。"""
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def profile_features(payload: dict) -> np.ndarray:
    """提取六个特征；深度减最小值以消除共同的垂向平移。

    跨向覆盖宽度仍受测量位置和当时水位影响。这些特征只描述已测剖面，
    不能解释为完整岸滩、绝对河床高程或任意水位下的过水面积。
    """
    units = payload.get("units", {})
    if units.get("Distance_ft", "ft") != "ft" or units.get("Depth_ft", "ft") != "ft":
        raise ValueError("剖面距离和深度必须明确为ft，不能猜测单位")
    pairs = []
    for point in payload["points"]:
        try:
            distance = float(point["Distance_ft"]) * 0.3048
            depth = float(point["Depth_ft"]) * 0.3048
        except (KeyError, TypeError, ValueError):
            continue
        if np.isfinite(distance) and np.isfinite(depth) and depth >= 0:
            pairs.append((distance, depth))
    if not pairs:
        raise ValueError("没有有效距离—深度点")
    frame = pd.DataFrame(pairs, columns=["x", "d"]).groupby("x", sort=True)["d"].median()
    x, depth = frame.index.to_numpy(float), frame.to_numpy(float)
    if x.size < 3 or x[-1] <= x[0] or depth.max() <= 0:
        raise ValueError("剖面至少需要三个不同距离、正宽度和至少一个正深度")
    span = float(x[-1] - x[0])
    relief = float(depth.max() - depth.min())
    u = (x - x[0]) / span
    grid = np.linspace(0, 1, 101)
    shape = np.interp(grid, u, depth - depth.min()) / max(relief, 1e-8)
    left = float(np.trapz(shape[:51], grid[:51]))
    right = float(np.trapz(shape[50:], grid[50:]))
    # 三个形状统计均不随剖面左右翻转改变，避免测线方向成为伪特征。
    maxima = grid[np.isclose(shape, shape.max())]
    deepest_offset = float(np.abs(maxima - 0.5).mean()) if relief > 1e-8 else 0.0
    values = np.array([
        np.log(span),
        np.log(max(relief, 0.01)),
        np.trapz(shape, grid),
        shape[50],
        deepest_offset,
        abs(left - right) / max(left + right, 1e-8),
    ], dtype=np.float32)
    if not np.isfinite(values).all():
        raise ValueError("断面特征出现非有限值")
    return values


def _archive_path(index_path: Path, relative: str) -> Path:
    """原存档与可移植包统一使用相对于存档根目录的路径。"""
    root = Path(index_path).resolve().parent.parent
    path = (root / relative).resolve()
    if not path.is_relative_to(root):
        raise ValueError(f"断面路径超出存档目录: {relative}")
    return path


def build_geometry(index_path: Path, basins: list[str], splits: dict,
                   mode: str = "snapshot") -> tuple[pd.DataFrame, pd.DataFrame]:
    """按预先规定的规则选择每站最早合格剖面，不按Q或模型表现挑选。

    snapshot是用户选定的固定断面回溯；causal只允许训练结束前的测量。
    不合并不同时期、不同位置和不同水位的断面。
    """
    if mode not in {"snapshot", "causal"}:
        raise ValueError(f"未知断面时间口径: {mode}")
    index_path = Path(index_path)
    source = pd.read_csv(index_path, dtype=str, keep_default_na=False)
    required = {"station", "mmt_ID", "measurement_date_local", "quality_code", "json_path"}
    if not required.issubset(source.columns):
        raise ValueError(f"断面索引缺少字段: {sorted(required - set(source.columns))}")
    rows, audits = [], []
    for basin in basins:
        station = source.loc[source.station == basin].copy()
        qualified = station.loc[station.quality_code.isin(ACCEPTED_QUALITY)].copy()
        if "valid_profile" in qualified:
            qualified = qualified.loc[qualified.valid_profile.str.lower() == "true"]
        dated = qualified.copy()
        if mode == "causal":
            cutoff = str(splits["splits"][basin]["train_end"])[:10]
            dated = dated.loc[dated.measurement_date_local <= cutoff]
        # 同日依次按质量和测量ID排序，排序不接触水位、Q或流速。
        dated["_quality_rank"] = dated.quality_code.map({"EXCL": 0, "GOOD": 1, "FAIR": 2})
        dated = dated.sort_values(["measurement_date_local", "_quality_rank", "mmt_ID"])
        values, chosen, errors = None, None, []
        for record in dated.to_dict("records"):
            path = _archive_path(index_path, record["json_path"])
            if not path.is_file():
                raise FileNotFoundError(f"索引中的剖面文件不存在: {path}")
            payload = json.loads(path.read_text(encoding="utf-8"))
            if str(payload.get("station")) != basin:
                raise ValueError(f"剖面站号与索引不一致: {path}")
            try:
                values = profile_features(payload)
            except (ValueError, KeyError) as exc:
                errors.append(f"{record['mmt_ID']}: {exc}")
                continue
            chosen = {**record, "file_sha256": file_sha256(path)}
            break
        available = values is not None
        row = {"basin": basin, **dict(zip(FEATURE_COLUMNS,
               values if available else [np.nan] * len(FEATURE_COLUMNS))),
               "geom_missing": int(not available)}
        rows.append(row)
        if available:
            reason = ""
        elif station.empty:
            reason = "no_point_profile"
        elif qualified.empty:
            reason = "no_accepted_quality"
        elif dated.empty:
            reason = "no_profile_before_train_end"
        else:
            reason = "invalid_numeric_profile"
        audit = {
            "basin": basin, "geometry_mode": mode, "available": available,
            "missing_reason": reason, "n_archived_profiles": len(station),
            "n_quality_profiles": len(qualified), "n_time_eligible_profiles": len(dated),
            "train_end": splits["splits"][basin]["train_end"],
            "profile_id": chosen["mmt_ID"] if chosen else "",
            "measurement_date": chosen["measurement_date_local"] if chosen else "",
            "quality_code": chosen["quality_code"] if chosen else "",
            "json_path": chosen["json_path"] if chosen else "",
            "source_sha256": chosen["file_sha256"] if chosen else "",
            "numeric_rejections": " | ".join(errors),
        }
        audit["post_train_geometry"] = bool(
            chosen and chosen["measurement_date_local"] > str(audit["train_end"])[:10])
        audit["location_code"] = chosen.get("location_code", "") if chosen else ""
        audits.append(audit)
    return pd.DataFrame(rows).set_index("basin"), pd.DataFrame(audits)


def fit_geometry_scaler(table: pd.DataFrame, fit_basins: list[str]) -> dict:
    """仅用Q保留流域的有效几何拟合填补值和尺度；缺失标记不标准化。"""
    if not fit_basins or not set(fit_basins).issubset(table.index):
        raise ValueError("几何标准化需要有效的保留流域列表")
    fit = table.loc[fit_basins, list(FEATURE_COLUMNS)].to_numpy(float)
    fill, means, stds = [], [], []
    for j in range(fit.shape[1]):
        vals = fit[:, j][np.isfinite(fit[:, j])]
        median = float(np.median(vals)) if vals.size else 0.0
        mean = float(vals.mean()) if vals.size else 0.0
        std = float(vals.std()) if vals.size else 1.0
        fill.append(median)
        means.append(mean)
        stds.append(std if std > 1e-8 else 1.0)
    return {"columns": list(FEATURE_COLUMNS), "fill": fill, "mean": means, "std": stds,
            "fit_basins": list(fit_basins), "missing_column": "geom_missing"}


def transform_geometry(table: pd.DataFrame, basins: list[str], scaler: dict) -> np.ndarray:
    values = table.loc[basins, scaler["columns"]].to_numpy(float)
    values = np.where(np.isfinite(values), values, np.asarray(scaler["fill"]))
    values = (values - np.asarray(scaler["mean"])) / np.asarray(scaler["std"])
    missing = table.loc[basins, "geom_missing"].to_numpy(float)[:, None]
    out = np.column_stack([values, missing]).astype(np.float32)
    if not np.isfinite(out).all():
        raise ValueError("几何输入填补后仍有非有限值")
    return out


def shuffled_geometry(values: np.ndarray, seed: int) -> np.ndarray:
    """打乱可用站之间的剖面对应关系；保留缺失站与缺失标记，精确匹配参数量。"""
    out = values.copy()
    valid = np.flatnonzero(values[:, -1] == 0)
    if valid.size > 1:
        rng = np.random.default_rng(seed)
        order = rng.permutation(valid)
        if np.array_equal(order, valid):
            order = np.roll(valid, 1)
        out[valid, :-1] = values[order, :-1]
    return out

