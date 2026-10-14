"""不依赖目标标签是否存在的完整时刻推理，保持各路径评价时间一致。"""

from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import Dataset

from models.lstm_models import build_model
from pipeline.dataset import make_loader
from training.trainer import predict


class InferenceWindows(Dataset):
    """只读强迫和静态输入；Q/H均不作为样本准入条件。"""

    def __init__(self, prepared, split: str, sequence: int, stride: int, tasks):
        self.prepared, self.sequence, self.tasks = prepared, sequence, tasks
        ids, positions = [], []
        continuous = getattr(prepared, "time_valid", np.ones(prepared.n_time, bool))
        prefix = np.r_[0, np.cumsum(~continuous)]
        for basin in prepared.basins:
            lo, hi = prepared.split_range(basin, split)
            pos = np.arange(max(lo, sequence), hi + 1, stride, dtype=np.int64)
            pos = pos[continuous[pos] & ((prefix[pos] - prefix[pos - sequence]) == 0)]
            if len(pos):
                ids.append(np.full(len(pos), prepared.basin_index[basin], np.int64))
                positions.append(pos)
        if not positions:
            raise ValueError(f"{split}没有可推理时刻")
        self.basin_idx, self.target_pos = np.concatenate(ids), np.concatenate(positions)
        self.offsets = np.arange(-sequence, 0, dtype=np.int64)

    def __len__(self):
        return len(self.basin_idx)

    def __getitem__(self, index):
        idx = np.atleast_1d(np.asarray(index, dtype=np.int64))
        bi, pos = self.basin_idx[idx], self.target_pos[idx]
        x = self.prepared.forcing[bi[:, None], pos[:, None] + self.offsets]
        c = self.prepared.attrs[bi]
        dummy = np.zeros((len(idx), len(self.tasks)), np.float32)
        return (torch.from_numpy(np.ascontiguousarray(x)),
                torch.from_numpy(np.ascontiguousarray(c)),
                torch.from_numpy(dummy), torch.from_numpy(dummy.copy()),
                torch.from_numpy(bi.copy()), torch.from_numpy(pos.copy()))


def infer(prepared, state_dict: dict, model_config: dict, split: str,
          sequence: int, stride: int, device: str) -> dict:
    model = build_model(model_config["architecture"], model_config["forcing_size"],
                        model_config["attr_size"], model_config["hidden_size"],
                        model_config["dropout_rate"])
    model.load_state_dict(state_dict)
    model.to(device)
    dataset = InferenceWindows(prepared, split, sequence, stride, model.tasks)
    result = predict(model, make_loader(dataset, 4096, False), model.tasks, device)
    del model
    if device.startswith("cuda"):
        torch.cuda.empty_cache()
    return result
