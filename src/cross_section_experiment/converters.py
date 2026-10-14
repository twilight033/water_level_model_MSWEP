"""独立区域转换器：整流域内部验证、流域平衡损失与冻结后的盲评价。"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch import nn


class RegionalConverter(nn.Module):
    """单隐层转换器，参数数目为hidden*(input_dim+2)+1。"""

    def __init__(self, input_dim: int, hidden: int):
        super().__init__()
        self.input_dim, self.hidden = input_dim, hidden
        self.net = nn.Sequential(nn.Linear(input_dim, hidden), nn.ReLU(), nn.Linear(hidden, 1))

    def forward(self, x):
        return self.net(x).squeeze(-1)

    def n_parameters(self):
        return sum(p.numel() for p in self.parameters())


@dataclass
class PairData:
    x: np.ndarray
    y: np.ndarray
    basin: np.ndarray
    weight: np.ndarray
    target_pos: np.ndarray


def internal_basin_split(retained: list[str], seed: int) -> tuple[list[str], list[str]]:
    """仅在Q保留流域内部留出整站，不能用外部Q留出站选参数。"""
    if len(retained) < 3:
        raise ValueError("内部转换验证至少需要三个保留流域")
    ordered = np.random.default_rng(seed).permutation(sorted(retained)).tolist()
    n_valid = max(1, min(len(ordered) - 2, int(round(0.2 * len(ordered)))))
    return ordered[n_valid:], ordered[:n_valid]


def collect_pairs(prepared, basins: list[str], split: str, scaling,
                  static: np.ndarray | None, max_per_basin: int = 4096) -> PairData:
    """只读取指定保留站的配对标签；按时间均匀抽样，不按流量大小抽样。"""
    xs, ys, ids, positions, counts = [], [], [], [], []
    for basin in basins:
        bi = prepared.basin_index[basin]
        lo, hi = prepared.split_range(basin, split)
        pos = np.arange(lo, hi + 1, dtype=np.int64)
        h = prepared.targets["waterlevel"][bi, pos]
        q = prepared.targets_raw["flow"][bi, pos]
        ok = np.isfinite(h) & np.isfinite(q)
        pos, h, q = pos[ok], h[ok], q[ok]
        if max_per_basin and len(pos) > max_per_basin:
            take = np.unique(np.linspace(0, len(pos) - 1, max_per_basin).astype(int))
            pos, h, q = pos[take], h[take], q[take]
        if not len(pos):
            continue
        x = h[:, None]
        if static is not None:
            x = np.column_stack([x, np.broadcast_to(static[bi], (len(pos), static.shape[1]))])
        y = (q - scaling.mean["flow"][bi]) / scaling.std["flow"][bi]
        xs.append(x.astype(np.float32))
        ys.append(y.astype(np.float32))
        ids.append(np.full(len(pos), bi, dtype=np.int64))
        positions.append(pos)
        counts.append(len(pos))
    if len(xs) < 1:
        raise ValueError(f"转换器{split}没有有效H—Q配对")
    size = sum(counts)
    # 全样本均值下，每站贡献相同权重；每批不重新归一化权重。
    weights = np.concatenate([np.full(n, size / (len(counts) * n), np.float32) for n in counts])
    return PairData(np.concatenate(xs), np.concatenate(ys), np.concatenate(ids),
                    weights, np.concatenate(positions))


def validation_score(data: PairData, pred: np.ndarray) -> float:
    """每站NSE等权平均；无方差站不参与选模，但不会被删出最终输出。"""
    scores = []
    for bi in np.unique(data.basin):
        selected = data.basin == bi
        obs, estimated = data.y[selected], pred[selected]
        denom = float(np.square(obs - obs.mean()).sum())
        if len(obs) >= 2 and denom > 1e-12:
            scores.append(1 - float(np.square(estimated - obs).sum()) / denom)
    if not scores:
        raise ValueError("转换器内部验证没有可计算NSE的流域")
    score = float(np.mean(scores))
    if not np.isfinite(score):
        raise ValueError("转换器内部验证分数不是有限值")
    return score


@torch.no_grad()
def predict_converter(model, x: np.ndarray, device: str, batch_size: int = 8192):
    model.eval()
    return np.concatenate([
        model(torch.from_numpy(x[i:i + batch_size]).to(device)).cpu().numpy()
        for i in range(0, len(x), batch_size)
    ])


def fit_network(train: PairData, hidden: int, epochs: int, seed: int, device: str,
                valid: PairData | None = None, batch_size: int = 2048,
                learning_rate: float = 1e-3) -> tuple[RegionalConverter, dict]:
    """验证仅用于确定候选宽度与轮数；最终重拟合不查看测试集。"""
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    model = RegionalConverter(train.x.shape[1], hidden).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    best_score, best_state, best_epoch = -float("inf"), None, epochs
    history = []
    for epoch in range(1, epochs + 1):
        model.train()
        losses = []
        order = rng.permutation(len(train.y))
        for start in range(0, len(order), batch_size):
            idx = order[start:start + batch_size]
            x = torch.from_numpy(train.x[idx]).to(device)
            y = torch.from_numpy(train.y[idx]).to(device)
            w = torch.from_numpy(train.weight[idx]).to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = (w * (model(x) - y).square()).mean()
            if not torch.isfinite(loss):
                raise ValueError("转换器损失出现非有限值")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        record = {"epoch": epoch, "loss": float(np.mean(losses))}
        if valid is not None:
            score = validation_score(valid, predict_converter(model, valid.x, device))
            record["valid_basin_nse"] = score
            if score > best_score:
                best_score, best_epoch = score, epoch
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        history.append(record)
    if valid is not None:
        if best_state is None:
            raise ValueError("转换器没有产生有效内部验证checkpoint")
        model.load_state_dict(best_state)
    return model, {"best_epoch": best_epoch, "best_score": best_score if valid is not None else None,
                   "history": history, "n_parameters": model.n_parameters()}


def fit_regional_converter(prepared, retained: list[str], static: np.ndarray | None,
                           widths: list[int], epochs: int, seed: int, device: str,
                           max_pairs: int = 4096, forced_width: int | None = None):
    """内部整站验证后在全部保留站重拟合，外部留出站Q不参与任何步骤。"""
    inner_train, inner_valid = internal_basin_split(retained, seed + 1000)
    # 内部伪无Q站的Q统计也不用于候选模型的尺度构建。
    inner_scale = prepared.make_scaling("physical", fit_basins=inner_train)
    train = collect_pairs(prepared, inner_train, "train", inner_scale, static, max_pairs)
    valid = collect_pairs(prepared, inner_valid, "valid", inner_scale, static, max_pairs)
    candidates = []
    for width in ([forced_width] if forced_width is not None else widths):
        _, info = fit_network(train, int(width), epochs, seed, device, valid)
        candidates.append({"width": int(width), **info})
    chosen = max(candidates, key=lambda v: v["best_score"])
    scale = prepared.make_scaling("physical", fit_basins=retained)
    final_train = collect_pairs(prepared, retained, "train", scale, static, max_pairs)
    model, final_info = fit_network(final_train, chosen["width"], chosen["best_epoch"],
                                    seed, device)
    audit = {
        "input_dim": final_train.x.shape[1], "hidden": chosen["width"],
        "n_parameters": model.n_parameters(), "epochs_refit": chosen["best_epoch"],
        "inner_train_basins": inner_train, "inner_validation_basins": inner_valid,
        "scale_fit_basins": retained, "n_train_pairs": len(final_train.y),
        "n_inner_train_pairs": len(train.y), "n_inner_validation_pairs": len(valid.y),
        "selected_by": "retained_basins_internal_whole_basin_validation",
        "training_h_source": "observed_training_h",
        "inference_h_source": "frozen_single_waterlevel_lstm_prediction",
        "input_error_distribution_shift": True,
        "candidates": candidates, "refit": final_info,
    }
    return model, scale, audit

