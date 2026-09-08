"""统一训练与评估循环，供全部架构共用。

统一的内容（回应审稿意见 Major 1 / Major 2）：所有模型使用同一份冻结
划分、同一份归一化统计量、同一套窗口、同一组超参与早停规则，因此模型
之间的差异只能来自架构本身。

如实记录的内容（回应 Major 2 / Major 3）：实际训练轮数、是否撞到轮数
上限、每个任务的标签曝光总量、有效更新步数。双头模型每轮的优化器步数
多于单任务（样本集是并集），这一差异必须报告而不是隐藏。
"""

import random
import sys
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path

import numpy as np
import torch
from torch import nn

_ROOT = Path(__file__).resolve().parents[2]
for _p in (str(_ROOT / "src"), str(_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from models.lstm_models import build_model  # noqa: E402
from pipeline.dataset import WindowDataset, make_loader  # noqa: E402

LOSS_NORMS = ("per_valid", "per_batch")


@dataclass
class TrainConfig:
    """一次训练运行的完整配置。"""

    architecture: str
    seq_length: int = 168             # 21 天 @3h
    window_step_train: int = 8        # 1 天；相邻窗口重叠 92%，已足够密
    window_step_eval: int = 8         # 验证用，早停判据
    window_step_test: int = 1         # 最终测试，逐步长评估以导出完整时序
    batch_size: int = 512
    hidden_size: int = 64
    dropout_rate: float = 0.2
    learning_rate: float = 1e-3
    # 固定 30 轮、不早停、余弦衰减。实测依据：
    # - patience=5 时 6 个种子的最佳轮落在 3–18，验证分与最佳轮几乎单调相关，
    #   种子标准差被抬到 0.0176，远大于待检测的效应量（约 0.008）；
    # - 去掉早停后曲线仍以 ±0.03 震荡，最佳轮顶到 28（未收敛）；
    # - 加余弦衰减后末 5 轮极差降到 0.004–0.007，最佳轮回落到 11–18，
    #   30 轮足够收敛，种子极差从 0.047 降到 0.024。
    max_epochs: int = 30
    patience: int = None             # None 表示跑满固定轮数，按验证分取最佳 checkpoint
    lr_schedule: str = "cosine"      # none | cosine
    lr_min_factor: float = 0.02      # cosine 末端学习率相对初值的比例
    proj_size: int = 16
    loss_norm: str = "per_valid"      # per_batch 为 4-E 归一化对照
    task_weights: dict = field(default_factory=lambda: {"flow": 1.0, "waterlevel": 1.0})
    model_seed: int = 1
    device: str = "cuda:0" if torch.cuda.is_available() else "cpu"

    def __post_init__(self):
        if self.loss_norm not in LOSS_NORMS:
            raise ValueError(f"loss_norm 必须是 {LOSS_NORMS} 之一，收到 {self.loss_norm!r}")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def masked_loss(preds: dict, y: torch.Tensor, m: torch.Tensor, tasks: tuple,
                task_weights: dict, loss_norm: str) -> tuple:
    """逐任务掩膜的加权 MSE。

    per_valid：除以 batch 内该任务的有效标签数，是多任务缺失标签的常规
    做法，损失尺度稳定。但它意味着删标签主要表现为"梯度估计方差变大"
    而非"梯度变小"——结合并集准入（删 Q 标签不减少样本数），这会系统性
    地削弱"缺失导致性能下降"的幅度，必须在论文中写明。
    per_batch：除以固定 batch size，缺失标签直接按比例缩小该任务梯度，
    作为对照证明结论不依赖归一化方式。
    """
    total = 0.0
    per_task = {}
    for k, task in enumerate(tasks):
        mask = m[:, k:k + 1]
        n_valid = mask.sum()
        sq = (preds[task] - y[:, k:k + 1]) ** 2 * mask
        if n_valid > 0:
            denom = n_valid if loss_norm == "per_valid" else mask.shape[0]
            loss = sq.sum() / denom
        else:
            loss = torch.zeros((), device=y.device)
        per_task[task] = loss
        total = total + task_weights.get(task, 1.0) * loss
    return total, per_task


def train_epoch(model, optimizer, loader, tasks, cfg) -> dict:
    model.train()
    device = cfg.device
    n_batches = 0
    n_updates = 0
    loss_sum = 0.0
    label_seen = {t: 0 for t in tasks}
    empty_batches = {t: 0 for t in tasks}

    for x, c, y, m, _, _ in loader:
        x = x.to(device, non_blocking=True)
        c = c.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        m = m.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        preds = model(x, c)
        total, _ = masked_loss(preds, y, m, tasks, cfg.task_weights, cfg.loss_norm)
        if not torch.isfinite(total):
            raise ValueError("训练损失出现非有限值，请检查输入与掩膜")
        total.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        loss_sum += float(total.detach())
        n_batches += 1
        n_updates += 1
        counts = m.sum(dim=0)
        for k, task in enumerate(tasks):
            label_seen[task] += int(counts[k])
            if counts[k] == 0:
                empty_batches[task] += 1

    return {
        "loss": loss_sum / max(n_batches, 1),
        "n_batches": n_batches,
        "n_updates": n_updates,
        "label_exposures": label_seen,
        "empty_task_batches": empty_batches,
    }


@torch.no_grad()
def predict(model, loader, tasks, device) -> dict:
    """返回归一化空间的预测与观测，并保留流域索引与目标时刻位置。"""
    model.eval()
    chunks = {"pred": {t: [] for t in tasks}, "obs": {t: [] for t in tasks},
              "mask": {t: [] for t in tasks}, "basin_idx": [], "target_pos": []}
    for x, c, y, m, bi, pos in loader:
        preds = model(x.to(device, non_blocking=True), c.to(device, non_blocking=True))
        for k, task in enumerate(tasks):
            chunks["pred"][task].append(preds[task].squeeze(-1).cpu().numpy())
            chunks["obs"][task].append(y[:, k].numpy())
            chunks["mask"][task].append(m[:, k].numpy())
        chunks["basin_idx"].append(bi.numpy())
        chunks["target_pos"].append(pos.numpy())

    out = {"basin_idx": np.concatenate(chunks["basin_idx"]),
           "target_pos": np.concatenate(chunks["target_pos"])}
    for key in ("pred", "obs", "mask"):
        out[key] = {t: np.concatenate(chunks[key][t]) for t in tasks}
    return out


def _nse_by_basin(result: dict, task: str) -> np.ndarray:
    """归一化空间的逐流域 NSE。NSE 对观测与预测的同一仿射变换不变，
    因此与在物理量纲上计算结果相同，可省去反归一化。"""
    valid = result["mask"][task] > 0.5
    if not valid.any():
        return np.array([])
    basins = result["basin_idx"][valid]
    obs = result["obs"][task][valid]
    pred = result["pred"][task][valid]
    scores = []
    for b in np.unique(basins):
        sel = basins == b
        o, p = obs[sel], pred[sel]
        denom = float(((o - o.mean()) ** 2).sum())
        if denom <= 0 or o.size < 2:
            continue
        scores.append(1.0 - float(((p - o) ** 2).sum()) / denom)
    return np.array(scores)


def train_model(prepared, cfg: TrainConfig, hidden: dict = None,
                basins: list = None, verbose: bool = True) -> dict:
    """完整训练一次并在测试集上评估，返回结果字典。"""
    set_seed(cfg.model_seed)
    model = build_model(cfg.architecture, forcing_size=prepared.forcing.shape[-1],
                        attr_size=prepared.attrs.shape[-1],
                        hidden_size=cfg.hidden_size, dropout_rate=cfg.dropout_rate,
                        proj_size=cfg.proj_size).to(cfg.device)
    tasks = model.tasks

    ds_train = WindowDataset(prepared, "train", cfg.seq_length, cfg.window_step_train,
                             tasks=tasks, basins=basins, hidden=hidden)
    ds_valid = WindowDataset(prepared, "valid", cfg.seq_length, cfg.window_step_eval,
                             tasks=tasks, basins=basins)
    ds_test = WindowDataset(prepared, "test", cfg.seq_length, cfg.window_step_test,
                            tasks=tasks, basins=basins)

    generator = torch.Generator().manual_seed(cfg.model_seed)
    tr_loader = make_loader(ds_train, cfg.batch_size, shuffle=True, generator=generator)
    va_loader = make_loader(ds_valid, 4096, shuffle=False)
    te_loader = make_loader(ds_test, 4096, shuffle=False)

    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.learning_rate)
    scheduler = None
    if cfg.lr_schedule == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=cfg.max_epochs,
            eta_min=cfg.learning_rate * cfg.lr_min_factor)
    elif cfg.lr_schedule != "none":
        raise ValueError(f"未知 lr_schedule: {cfg.lr_schedule!r}")

    best_score = -float("inf")
    best_epoch = 0
    best_state = None
    patience_left = cfg.patience
    history = []
    t_start = time.time()

    for epoch in range(1, cfg.max_epochs + 1):
        stats = train_epoch(model, optimizer, tr_loader, tasks, cfg)
        val = predict(model, va_loader, tasks, cfg.device)
        nse = {t: _nse_by_basin(val, t) for t in tasks}
        means = {t: (float(v.mean()) if v.size else float("nan")) for t, v in nse.items()}
        score = float(np.mean([means[t] for t in tasks]))

        history.append({"epoch": epoch, "train_loss": stats["loss"],
                        "val_nse": means, "score": score,
                        "lr": optimizer.param_groups[0]["lr"],
                        "n_updates": stats["n_updates"],
                        "label_exposures": dict(stats["label_exposures"])})
        if scheduler is not None:
            scheduler.step()
        improved = score > best_score
        if improved:
            best_score, best_epoch = score, epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            patience_left = cfg.patience or 0
        else:
            patience_left -= 1

        if verbose:
            flag = " *" if improved else (f"  ({patience_left})" if cfg.patience else "")
            detail = "  ".join(f"{t}={means[t]:.4f}" for t in tasks)
            print(f"  epoch {epoch:>2d}  loss={stats['loss']:.4f}  {detail}  "
                  f"score={score:.4f}{flag}")
        # patience 为 None/0 时不做早停：所有模型跑满固定轮数，再按验证分取最佳
        # checkpoint。早停会把"何时停"变成一个不受控的方差来源——实测 patience=5
        # 时 6 个种子的最佳轮在 3–18 之间，验证分与最佳轮几乎单调相关，
        # 种子标准差被抬到 0.0176，远大于待检测的效应量。
        if cfg.patience and patience_left <= 0:
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    test = predict(model, te_loader, tasks, cfg.device)
    elapsed = time.time() - t_start

    return {
        "config": asdict(cfg),
        "model_config": model.config(),
        "tasks": list(tasks),
        "best_epoch": best_epoch,
        "epochs_run": len(history),
        "hit_epoch_cap": len(history) >= cfg.max_epochs,
        "best_val_score": best_score,
        "history": history,
        "elapsed_sec": elapsed,
        "n_train_samples": len(ds_train),
        "n_valid_samples": len(ds_valid),
        "n_test_samples": len(ds_test),
        "train_label_counts": ds_train.label_counts(),
        "valid_label_counts": ds_valid.label_counts(),
        "test_label_counts": ds_test.label_counts(),
        "test_prediction": test,
        "state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()},
    }
