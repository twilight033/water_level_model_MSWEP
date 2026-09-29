"""用既有双头检查点在新流域上做零样本径流推理。

本脚本不训练、不修改 86 流域缓存或冻结划分。它读取用户提供的 MSWEP 降水表和
CAMELSH 原始数据，复用检查点中的物理径流尺度回归，对新流域输出逐时预测与 NSE。
结果仅用于检查跨流域接入与零样本泛化，不能替代后续重新训练的正式外部验证。

示例（PowerShell）：
    $env:CAMELSH_DATA_PATH = 'F:/data'
    python -X utf8 src/evaluation/zero_shot_candidate_inference.py `
      --mswep-csv 'F:/data/mswep_nonrating14_3hourly.csv'
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

_ROOT = Path(__file__).resolve().parents[2]
for _p in (str(_ROOT / "src"), str(_ROOT), str(_ROOT / "src" / "others")):
    if _p not in sys.path:
        sys.path.insert(0, _p)


DEFAULT_BASINS = (
    "01100561", "01189000", "02322800", "02365769", "02366996", "02407000",
    "03198000", "05422600", "06843500", "06890900", "06893620", "06893890",
    "07154500", "07230500",
)
DEFAULT_CHECKPOINT = (
    _ROOT / "results" / "runs"
    / "4K_best_physical_q_holdout70_dual_head_ms3_xs123_L480_physical_extended"
    / "model.pt"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="既有双头模型的零样本流域推理")
    parser.add_argument("--mswep-csv", type=Path, required=True,
                        help="包含 time 列和 14 个流域 ID 列的 3 小时 MSWEP CSV")
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT,
                        help="physical 双头模型的 model.pt")
    parser.add_argument("--out-dir", type=Path,
                        default=_ROOT / "results" / "zero_shot_nonrating14",
                        help="结果目录")
    parser.add_argument("--basins", nargs="+", default=list(DEFAULT_BASINS),
                        help="待推理流域 ID；默认使用筛出的 14 个")
    parser.add_argument("--start", default=None, help="可选起始时间，例如 2007-10-01")
    parser.add_argument("--end", default=None, help="可选结束时间，例如 2024-12-31")
    parser.add_argument("--batch-size", type=int, default=4096)
    return parser.parse_args()


def nse(obs: np.ndarray, pred: np.ndarray) -> float:
    """计算 NSE；有效样本不足或观测无变化时返回 NaN。"""
    valid = np.isfinite(obs) & np.isfinite(pred)
    if valid.sum() < 2:
        return float("nan")
    obs = obs[valid].astype("float64")
    pred = pred[valid].astype("float64")
    denom = float(np.square(obs - obs.mean()).sum())
    return float("nan") if denom <= 0 else float(1 - np.square(obs - pred).sum() / denom)


def read_mswep(path: Path, basins: list[str]) -> pd.DataFrame:
    """读取并校验用户生成的降水表。"""
    header = pd.read_csv(path, nrows=0).columns.astype(str).tolist()
    missing = [b for b in basins if b not in header]
    if missing:
        raise ValueError(f"MSWEP 文件缺少流域列: {missing}")
    if "time" not in header:
        raise ValueError("MSWEP 文件缺少 time 列")
    df = pd.read_csv(path, usecols=["time", *basins], parse_dates=["time"],
                     dtype={b: "float32" for b in basins})
    df = df.set_index("time").sort_index()
    if df.index.duplicated().any():
        raise ValueError(f"MSWEP 文件有 {int(df.index.duplicated().sum())} 个重复时间戳")
    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("MSWEP time 列无法解析为时间")
    return df.reindex(columns=basins)


def read_camelsh(basins: list[str]):
    """读取新流域的 CAMELSH 强迫和实测 Q，并重采样至 3 小时。"""
    from hydrodataset import StandardVariable
    from improved_camelsh_reader import ImprovedCAMELSHReader
    from config import CAMELSH_DATA_PATH
    from pipeline.paths import verify_camelsh_path

    root = verify_camelsh_path(CAMELSH_DATA_PATH)
    reader = ImprovedCAMELSHReader(str(root), download=False, use_batch=True)
    t_range = reader.camelsh.default_t_range
    variables = [StandardVariable.TEMPERATURE_MEAN, StandardVariable.SOLAR_RADIATION,
                 StandardVariable.STREAMFLOW]
    ds = reader.read_ts_xrdataset(gage_id_lst=basins, t_range=t_range, var_lst=variables)
    ds = ds.resample(time="3h").mean()

    def frame(var):
        df = ds[var].transpose("time", "basin").to_pandas()
        df.columns = [str(x) for x in df.columns]
        return df.reindex(columns=basins).astype("float32")

    return frame(StandardVariable.TEMPERATURE_MEAN), frame(StandardVariable.SOLAR_RADIATION), frame(StandardVariable.STREAMFLOW)


def build_attrs(basins: list[str], stats: dict) -> np.ndarray:
    """构造扩展属性并严格对齐到训练时属性列。"""
    from pipeline.attribute_sources import build_attribute_table

    raw, _ = build_attribute_table("extended", basins)
    columns = list(stats["meta"]["attr_columns"])
    unseen = [c for c in raw.columns if c not in columns]
    if unseen:
        print(f"警告：新流域出现训练集中未见的类别属性，已按全 0 编码: {unseen}")
    raw = raw.reindex(columns=columns, fill_value=0.0)
    if raw.isna().any().any():
        bad = raw.columns[raw.isna().any()].tolist()
        raise ValueError(f"属性缺失: {bad}")
    mean = np.array([stats["attr"][c]["mean"] for c in columns], dtype="float32")
    std = np.array([stats["attr"][c]["std"] for c in columns], dtype="float32")
    return ((raw.to_numpy(dtype="float32") - mean) / std).astype("float32"), raw


def physical_scale(raw_attrs: pd.DataFrame, blob: dict) -> tuple[np.ndarray, np.ndarray]:
    """用检查点保存的面积×降水回归恢复新流域的 Q 尺度，不使用新流域 Q。"""
    info = blob.get("scaling_info", {}).get("flow")
    if not info or "mean" not in info or "std" not in info:
        raise ValueError("检查点不含 physical 径流尺度信息；请使用 4K physical 模型")
    proxy = raw_attrs["area"].to_numpy(float) * raw_attrs["p_mean"].to_numpy(float)
    if np.any(proxy <= 0):
        raise ValueError("面积×年均降水必须为正")
    x = np.log(proxy)
    mean = np.exp(info["mean"]["intercept"] + info["mean"]["slope"] * x)
    std = np.exp(info["std"]["intercept"] + info["std"]["slope"] * x)
    return mean.astype("float32"), np.maximum(std, 1e-6).astype("float32")


@torch.no_grad()
def predict(model, forcing: np.ndarray, attrs: np.ndarray, seq_length: int,
            batch_size: int, device: str) -> tuple[np.ndarray, np.ndarray]:
    """仅在输入强迫完整的时刻生成归一化 Q 预测。"""
    n_basin, n_time, _ = forcing.shape
    valid = np.isfinite(forcing).all(axis=(1, 2))
    positions = []
    basin_idx = []
    for bi in range(n_basin):
        candidates = np.arange(seq_length, n_time, dtype=np.int64)
        # 每个窗口都必须完整；cumsum 比逐窗口循环快得多
        bad = (~valid[bi]).astype(np.int32)
        count = np.concatenate(([0], np.cumsum(bad)))
        keep = (count[candidates] - count[candidates - seq_length]) == 0
        positions.append(candidates[keep])
        basin_idx.append(np.full(int(keep.sum()), bi, dtype=np.int64))
    positions = np.concatenate(positions)
    basin_idx = np.concatenate(basin_idx)
    pred = np.empty(len(positions), dtype="float32")
    offsets = np.arange(-seq_length, 0, dtype=np.int64)
    for lo in range(0, len(positions), batch_size):
        hi = min(lo + batch_size, len(positions))
        bi = basin_idx[lo:hi]
        pos = positions[lo:hi]
        x = torch.from_numpy(np.ascontiguousarray(forcing[bi[:, None], pos[:, None] + offsets])).to(device)
        c = torch.from_numpy(np.ascontiguousarray(attrs[bi])).to(device)
        pred[lo:hi] = model(x, c)["flow"].squeeze(-1).cpu().numpy()
    return basin_idx, positions, pred


def main() -> None:
    args = parse_args()
    basins = [str(b).zfill(8) for b in args.basins]
    if not args.checkpoint.is_file():
        raise FileNotFoundError(f"检查点不存在: {args.checkpoint}")
    if not args.mswep_csv.is_file():
        raise FileNotFoundError(f"MSWEP 文件不存在: {args.mswep_csv}")

    from models.lstm_models import build_model
    from pipeline.normalization import load_norm_stats

    stats = load_norm_stats()
    blob = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    cfg = blob["model_config"]
    if cfg["architecture"] != "dual_head":
        raise ValueError(f"检查点不是双头模型: {cfg['architecture']}")
    if blob.get("config", {}).get("target_scaling") != "physical":
        raise ValueError("零样本流量推理必须使用 physical target scaling 的检查点")

    mswep = read_mswep(args.mswep_csv, basins)
    temp, solar, q_obs = read_camelsh(basins)
    grid = mswep.index.intersection(temp.index).intersection(solar.index).intersection(q_obs.index).sort_values()
    if args.start:
        grid = grid[grid >= pd.Timestamp(args.start)]
    if args.end:
        grid = grid[grid <= pd.Timestamp(args.end)]
    if len(grid) <= cfg.get("seq_length", 480):
        raise ValueError(f"共同时间轴仅 {len(grid)} 步，不足 L480 窗口")
    forcing = np.stack([
        mswep.reindex(grid, columns=basins).to_numpy("float32").T,
        temp.reindex(grid, columns=basins).to_numpy("float32").T,
        solar.reindex(grid, columns=basins).to_numpy("float32").T,
    ], axis=-1)
    attrs, raw_attrs = build_attrs(basins, stats)
    f_mean = np.array([stats["forcing"][v]["mean"] for v in ("precipitation", "temperature_mean", "solar_radiation")], dtype="float32")
    f_std = np.array([stats["forcing"][v]["std"] for v in ("precipitation", "temperature_mean", "solar_radiation")], dtype="float32")
    forcing = ((forcing - f_mean) / f_std).astype("float32")

    model = build_model("dual_head", forcing_size=cfg["forcing_size"], attr_size=cfg["attr_size"],
                        hidden_size=cfg["hidden_size"], dropout_rate=cfg["dropout_rate"]).to(
                            "cuda:0" if torch.cuda.is_available() else "cpu")
    model.load_state_dict(blob["state_dict"])
    model.eval()
    device = next(model.parameters()).device.type + (":0" if next(model.parameters()).device.type == "cuda" else "")
    bi, pos, q_norm = predict(model, forcing, attrs, 480, args.batch_size, device)
    q_mean, q_std = physical_scale(raw_attrs, blob)
    q_hat = q_norm * q_std[bi] + q_mean[bi]
    observed = q_obs.reindex(grid, columns=basins).to_numpy("float32").T[bi, pos]

    result = pd.DataFrame({
        "time": grid[pos], "basin": np.array(basins, dtype=object)[bi],
        "q_observed": observed, "q_predicted": q_hat,
    })
    rows = []
    for basin, part in result.groupby("basin", sort=True):
        rows.append({"basin": basin, "n_predictions": len(part),
                     "n_observed_q": int(part.q_observed.notna().sum()),
                     "nse": nse(part.q_observed.to_numpy(), part.q_predicted.to_numpy())})
    metrics = pd.DataFrame(rows).sort_values("basin")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    result.to_parquet(args.out_dir / "predictions_3hourly.parquet", index=False)
    metrics.to_csv(args.out_dir / "metrics_by_basin.csv", index=False, encoding="utf-8-sig")
    (args.out_dir / "run.json").write_text(json.dumps({
        "kind": "zero_shot_only", "checkpoint": str(args.checkpoint.resolve()),
        "mswep_csv": str(args.mswep_csv.resolve()), "basins": basins,
        "time_start": str(grid.min()), "time_end": str(grid.max()),
        "seq_length": 480, "target_scaling": "physical",
        "note": "结果仅用于零样本接入检查，非重新训练后的正式外部验证。",
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(metrics.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print(f"\n流域平均 NSE: {metrics.nse.mean():.4f}；中位 NSE: {metrics.nse.median():.4f}")
    print(f"已写入: {args.out_dir}")


if __name__ == "__main__":
    main()
