"""逐流域多指标评估，含高低流量分层与事件尺度指标。

回应审稿意见 Major 8：只报告平均 NSE 不足以评价 3 小时径流与水位预测。
本模块在物理量纲上计算 NSE、KGE 及其三个分量、RMSE、MAE、PBIAS、
Pearson r，并补充高流量段（前 2%）与低流量段（后 30%）的偏差，以及
事件尺度的洪峰时刻误差与峰值幅度误差。

聚合时同时给出均值、中位数、四分位数与失败流域（NSE<0）名单，而不是
只报一个总体平均值。
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

_ROOT = Path(__file__).resolve().parents[2]
for _p in (str(_ROOT / "src"), str(_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

STEP_HOURS = 3
# 事件识别：洪峰至少高于该分位数，且相邻峰间隔不少于 PEAK_MIN_SEPARATION 步
PEAK_QUANTILE = 0.98
PEAK_MIN_SEPARATION = 8       # 1 天
PEAK_MATCH_WINDOW = 8         # 在观测峰前后 ±1 天内寻找预测峰


def _safe(x):
    return float(x) if np.isfinite(x) else float("nan")


def nse(obs: np.ndarray, pred: np.ndarray) -> float:
    denom = ((obs - obs.mean()) ** 2).sum()
    if denom <= 0:
        return float("nan")
    return _safe(1.0 - ((pred - obs) ** 2).sum() / denom)


def kge(obs: np.ndarray, pred: np.ndarray) -> dict:
    """Gupta 等 (2009) 的 KGE 及三个分量。"""
    if obs.std() <= 0 or pred.std() <= 0:
        return {"kge": float("nan"), "kge_r": float("nan"),
                "kge_alpha": float("nan"), "kge_beta": float("nan")}
    r = float(np.corrcoef(obs, pred)[0, 1])
    alpha = float(pred.std() / obs.std())
    beta = float(pred.mean() / obs.mean()) if obs.mean() != 0 else float("nan")
    if not np.isfinite(beta):
        return {"kge": float("nan"), "kge_r": r, "kge_alpha": alpha, "kge_beta": beta}
    value = 1.0 - np.sqrt((r - 1) ** 2 + (alpha - 1) ** 2 + (beta - 1) ** 2)
    return {"kge": _safe(value), "kge_r": r, "kge_alpha": alpha, "kge_beta": beta}


def _percent_bias(obs: np.ndarray, pred: np.ndarray) -> float:
    total = obs.sum()
    if total == 0:
        return float("nan")
    return _safe(100.0 * (pred - obs).sum() / total)


def high_low_flow_bias(obs: np.ndarray, pred: np.ndarray) -> dict:
    """高流量段（前 2%）与低流量段（后 30%）的百分比偏差。

    对应 Yilmaz 等 (2008) 的 FHV / FLV 思路：按**观测**排序划分区段，
    再在该区段上比较预测与观测的总量偏差。
    """
    order = np.argsort(obs)[::-1]
    n = obs.size
    n_high = max(1, int(round(0.02 * n)))
    n_low = max(1, int(round(0.30 * n)))
    hi = order[:n_high]
    lo = order[-n_low:]
    return {
        "fhv_top2pct": _percent_bias(obs[hi], pred[hi]),
        "flv_bottom30pct": _percent_bias(obs[lo], pred[lo]),
        "n_high": int(n_high),
        "n_low": int(n_low),
    }


def peak_errors(obs: np.ndarray, pred: np.ndarray) -> dict:
    """事件尺度的洪峰时刻误差与峰值幅度误差。

    在**观测**序列上识别洪峰，再在预测序列的对应窗口内取最大值。
    时刻误差为正表示预测峰值出现得晚于观测。序列须为等间隔且无缺口
    （由 build_basin_series 用 NaN 补齐后在此按连续段处理）。
    """
    from scipy.signal import find_peaks

    valid = np.isfinite(obs) & np.isfinite(pred)
    if valid.sum() < PEAK_MIN_SEPARATION * 3:
        return {"n_peaks": 0, "peak_time_error_h": float("nan"),
                "peak_time_mae_h": float("nan"), "peak_magnitude_pct": float("nan")}

    threshold = np.nanquantile(obs[valid], PEAK_QUANTILE)
    peaks, _ = find_peaks(np.where(valid, obs, -np.inf), height=threshold,
                          distance=PEAK_MIN_SEPARATION)
    if peaks.size == 0:
        return {"n_peaks": 0, "peak_time_error_h": float("nan"),
                "peak_time_mae_h": float("nan"), "peak_magnitude_pct": float("nan")}

    time_errors, mag_errors = [], []
    for p in peaks:
        lo = max(0, p - PEAK_MATCH_WINDOW)
        hi = min(obs.size, p + PEAK_MATCH_WINDOW + 1)
        window = np.where(valid[lo:hi], pred[lo:hi], -np.inf)
        if not np.isfinite(window).any():
            continue
        q = int(np.argmax(window)) + lo
        time_errors.append((q - p) * STEP_HOURS)
        if obs[p] != 0:
            mag_errors.append(100.0 * (pred[q] - obs[p]) / obs[p])

    if not time_errors:
        return {"n_peaks": 0, "peak_time_error_h": float("nan"),
                "peak_time_mae_h": float("nan"), "peak_magnitude_pct": float("nan")}
    return {
        "n_peaks": int(len(time_errors)),
        "peak_time_error_h": _safe(np.mean(time_errors)),
        "peak_time_mae_h": _safe(np.mean(np.abs(time_errors))),
        "peak_magnitude_pct": _safe(np.mean(mag_errors)) if mag_errors else float("nan"),
    }


def basin_metrics(obs: np.ndarray, pred: np.ndarray,
                  series_obs: np.ndarray = None,
                  series_pred: np.ndarray = None) -> dict:
    """单个流域单个任务的全部指标。

    obs / pred 为有效样本上的一维数组（用于统计类指标）；
    series_obs / series_pred 为补齐到等间隔网格、缺失处为 NaN 的序列
    （用于事件尺度指标）。
    """
    if obs.size < 2:
        return {"n": int(obs.size)}
    residual = pred - obs
    out = {
        "n": int(obs.size),
        "nse": nse(obs, pred),
        "rmse": _safe(np.sqrt((residual ** 2).mean())),
        "mae": _safe(np.abs(residual).mean()),
        "pbias": _percent_bias(obs, pred),
        "pearson_r": _safe(np.corrcoef(obs, pred)[0, 1]) if obs.std() > 0 and pred.std() > 0 else float("nan"),
        "obs_mean": _safe(obs.mean()),
        "obs_std": _safe(obs.std()),
    }
    out.update(kge(obs, pred))
    out.update(high_low_flow_bias(obs, pred))
    if series_obs is not None and series_pred is not None:
        out.update(peak_errors(series_obs, series_pred))
    return out


def build_basin_series(prepared, result: dict, task: str, basin_idx: int,
                       scaling=None) -> pd.DataFrame:
    """把某流域的预测还原为按时间排序、缺失处为 NaN 的等间隔序列。"""
    sel = result["basin_idx"] == basin_idx
    pos = result["target_pos"][sel]
    order = np.argsort(pos)
    pos = pos[order]
    valid = result["mask"][task][sel][order] > 0.5

    obs_n = result["obs"][task][sel][order]
    pred_n = result["pred"][task][sel][order]
    # 用本次运行实际使用的尺度反归一化。observed 与 physical 两种模式下，
    # 观测都能被精确还原（同一仿射变换），因此指标始终是在物理量纲上算的
    obs = prepared.denormalize(task, basin_idx, obs_n, scaling)
    pred = prepared.denormalize(task, basin_idx, pred_n, scaling)

    full = np.arange(pos.min(), pos.max() + 1)
    frame = pd.DataFrame({"target_pos": full})
    frame["time"] = prepared.grid[full]
    lut = pd.DataFrame({
        "target_pos": pos,
        f"obs_{task}": np.where(valid, obs, np.nan),
        f"pred_{task}": np.where(valid, pred, np.nan),
    })
    return frame.merge(lut, on="target_pos", how="left")


def evaluate_run(prepared, result: dict, tasks, scaling=None) -> tuple:
    """对一次运行的测试集预测计算逐流域指标，并返回逐流域时序表。

    Returns
    -------
    (metrics_df, series_dict)
        metrics_df: 每行一个 (basin, task) 的全部指标；
        series_dict: {basin: DataFrame}，含真实时间戳与各任务观测/预测。
    """
    rows, series = [], {}
    for basin_idx in np.unique(result["basin_idx"]):
        basin = prepared.basins[int(basin_idx)]
        merged = None
        for task in tasks:
            frame = build_basin_series(prepared, result, task, int(basin_idx), scaling)
            merged = frame if merged is None else merged.merge(
                frame.drop(columns=["time"]), on="target_pos", how="outer")

            obs_series = frame[f"obs_{task}"].to_numpy()
            pred_series = frame[f"pred_{task}"].to_numpy()
            ok = np.isfinite(obs_series) & np.isfinite(pred_series)
            metrics = basin_metrics(obs_series[ok], pred_series[ok],
                                    obs_series, pred_series)
            rows.append({"basin": basin, "task": task, **metrics})
        series[basin] = merged.sort_values("target_pos").reset_index(drop=True)
    return pd.DataFrame(rows), series


def aggregate(metrics_df: pd.DataFrame, columns=("nse", "kge", "rmse", "mae", "pbias")) -> pd.DataFrame:
    """按任务聚合：均值、中位数、四分位数、失败流域数。"""
    out = []
    for task, sub in metrics_df.groupby("task"):
        row = {"task": task, "n_basins": len(sub)}
        for col in columns:
            if col not in sub:
                continue
            vals = sub[col].to_numpy(dtype=float)
            vals = vals[np.isfinite(vals)]
            if vals.size == 0:
                continue
            row[f"{col}_mean"] = float(vals.mean())
            row[f"{col}_median"] = float(np.median(vals))
            row[f"{col}_q25"] = float(np.percentile(vals, 25))
            row[f"{col}_q75"] = float(np.percentile(vals, 75))
        if "nse" in sub:
            failed = sub.loc[sub["nse"] < 0, "basin"].tolist()
            row["n_nse_negative"] = len(failed)
            row["nse_negative_basins"] = ";".join(failed)
        out.append(row)
    return pd.DataFrame(out)
