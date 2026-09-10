"""统一的滑窗数据集：冻结边界 + 并集准入 + 逐任务掩膜。

替代原来分散在 multi_task_lstm_wl2d.py、两个 ablation 脚本和单任务脚本
中的四份重复实现。关键差异：

1. 边界从 data/splits/basin_splits.json 读取，Dataset 内部不再推断，
   因此单任务/多任务、不同缺失比例、不同窗口长度得到完全相同的划分。
2. 样本准入用并集（至少一个任务标签有效），损失里按任务分别掩膜。
   单任务模式下只保留该任务标签有效的窗口，避免零梯度 batch。
3. 归一化统计量由 normalization.py 在训练段原始标签上算好后传入。
4. 每条样本携带真实目标时刻（网格整数位置），评估时据此还原时间轴，
   不再依赖 DataLoader 顺序或人为构造的等间隔 date_range。
5. lookup 表存整数位置，__getitem__ 不做任何 pandas 时间索引查找；
   支持一次取整批（__getitem__ 接受 list），配合 BatchSampler 使用。

目标时刻约定：输入窗口覆盖 [t0, t0+L*3h)，目标为 t0+L*3h 时刻的瞬时值，
即 target_pos = t0_pos + L，输入取 forcing[target_pos-L : target_pos]。
详见 docs/temporal_alignment.md。
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import (
    BatchSampler, DataLoader, Dataset, RandomSampler, SequentialSampler,
)

_ROOT = Path(__file__).resolve().parents[2]
for _p in (str(_ROOT / "src"), str(_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from pipeline.loaders import (  # noqa: E402
    FORCING_ORDER, TASKS, load_attributes, load_forcing, load_targets,
)
from pipeline.normalization import load_norm_stats  # noqa: E402
from pipeline.splits import load_splits  # noqa: E402

SPLIT_NAMES = ("train", "valid", "test")


class TargetScaling:
    """一次运行所用的目标归一化方案。

    ``observed``：逐流域用实测均值与标准差。默认方案，但对"该流域没有径流观测"
    的留出情景构成泄漏——模型仍间接知道流量量级与变幅。

    ``physical``：径流尺度改由 ``面积 × 日均降水`` 的 log–log 回归推出，回归只在
    仍有径流标签的流域上拟合，留出流域零径流观测。**水位仍用实测统计量**——这不是
    双标准：留出情景的前提就是该流域有水位记录，水位观测本来就是可得的。
    """

    def __init__(self, prepared, scaling: str = "observed", fit_basins=None):
        if scaling not in ("observed", "physical"):
            raise ValueError(f"scaling 必须是 observed 或 physical，收到 {scaling!r}")
        self.scaling = scaling
        self.info = {}
        basins = prepared.basins
        self.mean, self.std, self.normalized = {}, {}, {}

        for task in TASKS:
            if task == "flow" and scaling == "physical":
                from pipeline.normalization import physical_flow_scale
                mean, std, info = physical_flow_scale(
                    prepared.attrs_raw, prepared.stats[task],
                    fit_basins if fit_basins is not None else basins, basins)
                self.info[task] = info
            else:
                mean = np.array([prepared.stats[task][b]["mean"] for b in basins],
                                dtype="float32")
                std = np.array([prepared.stats[task][b]["std"] for b in basins],
                               dtype="float32")
            self.mean[task] = mean
            self.std[task] = std
            self.normalized[task] = (
                (prepared.targets_raw[task] - mean[:, None]) / std[:, None]
            ).astype("float32")

    def denormalize(self, task: str, basin_idx, values):
        return values * self.std[task][basin_idx] + self.mean[task][basin_idx]


class PreparedData:
    """一次性载入并归一化的共享容器，train/valid/test 三个数据集共用。"""

    def __init__(self, splits_payload: dict = None, norm_stats: dict = None,
                 attr_set: str = "base"):
        self.splits = splits_payload or load_splits()
        self.stats = norm_stats or load_norm_stats()

        grid, forcing, basins = load_forcing()
        self.grid = grid
        self.basins = basins
        self.basin_index = {b: i for i, b in enumerate(basins)}

        # 强迫归一化，统计量来自训练段
        f_mean = np.array(
            [self.stats["forcing"][v]["mean"] for v in FORCING_ORDER], dtype="float32")
        f_std = np.array(
            [self.stats["forcing"][v]["std"] for v in FORCING_ORDER], dtype="float32")
        self.forcing = ((forcing - f_mean) / f_std).astype("float32")

        # 静态属性归一化；one-hot 列的 mean=0/std=1，等于不做变换
        self.attr_set = attr_set
        attr_arr, attr_cols, _ = load_attributes(basins, attr_set=attr_set)
        # 属性缓存可能是早期用 hydrodataset 建的，分类属性的类别码编号不同会让
        # one-hot 列名对不上 norm_stats（后者随仓库分发、用当前代码生成）。
        # 逐值已验证两种来源完全相同，仅命名有别，因此直接按当前代码重建缓存是
        # 安全的；否则会以 KeyError 的形式在下一行崩掉，且提示不知所云。
        missing = [c for c in attr_cols if c not in self.stats["attr"]]
        if missing:
            from pipeline.attribute_sources import get_attribute_table
            print(f"[属性缓存不兼容] 以下列不在 norm_stats 中: {missing}\n"
                  f"  该缓存应为早期版本生成，正按当前代码重建（取值不变，仅统一命名）…")
            get_attribute_table(attr_set, overwrite=True, basins=basins)
            attr_arr, attr_cols, _ = load_attributes(basins, attr_set=attr_set)
            still_missing = [c for c in attr_cols if c not in self.stats["attr"]]
            if still_missing:
                raise KeyError(
                    f"重建后仍有属性列不在 norm_stats 中: {still_missing}\n"
                    f"请重新生成统计量：python -X utf8 src/pipeline/normalization.py")
        # 物理尺度回归需要未标准化的 area 与 p_mean，故原值一并保留
        self.attrs_raw = pd.DataFrame(attr_arr, index=basins, columns=attr_cols)
        a_mean = np.array([self.stats["attr"][c]["mean"] for c in attr_cols], dtype="float32")
        a_std = np.array([self.stats["attr"][c]["std"] for c in attr_cols], dtype="float32")
        self.attrs = ((attr_arr - a_mean) / a_std).astype("float32")
        self.attr_columns = attr_cols

        # 目标原值保留用于评估；默认按实测统计量归一化
        self.targets_raw = load_targets(grid, basins)
        self.default_scaling = TargetScaling(self, "observed")
        self.target_mean = self.default_scaling.mean
        self.target_std = self.default_scaling.std
        self.targets = self.default_scaling.normalized

        self.n_time = len(grid)
        self.input_size = self.forcing.shape[-1] + self.attrs.shape[-1]

    def make_scaling(self, scaling: str = "observed", fit_basins=None) -> TargetScaling:
        """按需构造目标归一化方案；observed 复用已建好的默认方案。"""
        if scaling == "observed":
            return self.default_scaling
        return TargetScaling(self, scaling, fit_basins)

    def split_range(self, basin: str, split: str) -> tuple:
        """返回该流域该分段目标时刻的整数位置闭区间 (start_pos, end_pos)。"""
        entry = self.splits["splits"][basin]
        start = pd.Timestamp(entry[f"{split}_start"])
        end = pd.Timestamp(entry[f"{split}_end"])
        start_pos = int(self.grid.searchsorted(start, "left"))
        end_pos = int(self.grid.searchsorted(end, "right")) - 1
        return start_pos, end_pos

    def denormalize(self, task: str, basin_idx, values, scaling: "TargetScaling" = None):
        """把归一化的预测或观测还原为物理量。"""
        return (scaling or self.default_scaling).denormalize(task, basin_idx, values)


class WindowDataset(Dataset):
    """按冻结边界在指定分段上滑窗。

    Parameters
    ----------
    prepared : PreparedData
        共享数据容器。
    split : {"train", "valid", "test"}
    seq_length : int
        输入窗口步数，1 步 = 3 小时。
    window_step : int
        相邻目标时刻之间的步长。训练与验证建议 8（1 天），最终测试用 1。
    tasks : tuple
        本次训练涉及的任务。单任务时只保留该任务标签有效的窗口。
    basins : list, optional
        参与的流域，默认取冻结划分中的全部流域。
    hidden : dict, optional
        {task: bool 数组 [n_basin, n_time]}，True 表示该标签被人工删除。
        只允许作用于训练段，由 masking.py 保证。
    """

    def __init__(self, prepared: PreparedData, split: str, seq_length: int,
                 window_step: int = 1, tasks=TASKS, basins=None, hidden=None,
                 scaling: TargetScaling = None):
        if split not in SPLIT_NAMES:
            raise ValueError(f"split 必须是 {SPLIT_NAMES} 之一，收到 {split!r}")
        self.prepared = prepared
        self.split = split
        self.seq_length = int(seq_length)
        self.window_step = int(window_step)
        self.tasks = tuple(tasks)
        self.hidden = hidden or {}
        # 目标归一化方案随运行而变（observed / physical），样本准入只看标签是否
        # 有效，与尺度无关，因此两种方案给出完全相同的样本集
        self.scaling = scaling or prepared.default_scaling
        self._targets = self.scaling.normalized

        pool = basins if basins is not None else list(prepared.splits["splits"].keys())
        self.basins = [b for b in pool if b in prepared.basin_index]

        basin_ids, positions = [], []
        self.per_basin_counts = {}
        for basin in self.basins:
            bi = prepared.basin_index[basin]
            start_pos, end_pos = prepared.split_range(basin, split)
            # 输入窗口必须完整落在时间轴内
            start_pos = max(start_pos, self.seq_length)
            if end_pos < start_pos:
                self.per_basin_counts[basin] = 0
                continue

            cand = np.arange(start_pos, end_pos + 1, self.window_step, dtype=np.int64)
            keep = np.zeros(cand.shape, dtype=bool)
            for task in self.tasks:
                valid = ~np.isnan(self._targets[task][bi, cand])
                if task in self.hidden:
                    valid &= ~self.hidden[task][bi, cand]
                keep |= valid          # 并集准入；单任务时等价于该任务有效
            cand = cand[keep]

            if cand.size:
                basin_ids.append(np.full(cand.shape, bi, dtype=np.int64))
                positions.append(cand)
            self.per_basin_counts[basin] = int(cand.size)

        if not positions:
            raise ValueError(
                f"{split} 分段没有生成任何样本，请检查冻结划分与标签有效性")
        self.basin_idx = np.concatenate(basin_ids)
        self.target_pos = np.concatenate(positions)
        self._offsets = np.arange(-self.seq_length, 0, dtype=np.int64)

    def __len__(self):
        return int(self.basin_idx.size)

    def label_counts(self) -> dict:
        """每个任务实际参与损失的标签数，用于报告监督总量。"""
        out = {}
        for task in self.tasks:
            valid = ~np.isnan(self._targets[task][self.basin_idx, self.target_pos])
            if task in self.hidden:
                valid &= ~self.hidden[task][self.basin_idx, self.target_pos]
            out[task] = int(valid.sum())
        return out

    def target_times(self) -> pd.DatetimeIndex:
        """本数据集全部样本的真实目标时刻。"""
        return self.prepared.grid[self.target_pos]

    def __getitem__(self, index):
        """支持单个索引与整批索引，后者绕开逐样本 Python 开销。"""
        idx = np.atleast_1d(np.asarray(index, dtype=np.int64))
        bi = self.basin_idx[idx]
        pos = self.target_pos[idx]

        time_idx = pos[:, None] + self._offsets[None, :]        # [B, L]
        x = self.prepared.forcing[bi[:, None], time_idx]        # [B, L, V]
        c = self.prepared.attrs[bi]                             # [B, A]

        y = np.empty((idx.size, len(self.tasks)), dtype="float32")
        m = np.zeros((idx.size, len(self.tasks)), dtype="float32")
        for k, task in enumerate(self.tasks):
            vals = self._targets[task][bi, pos]
            valid = ~np.isnan(vals)
            if task in self.hidden:
                valid &= ~self.hidden[task][bi, pos]
            # 先把 NaN 换成 0 再配掩膜，否则 0 * NaN = NaN 会污染整批梯度
            y[:, k] = np.where(valid, vals, 0.0)
            m[:, k] = valid.astype("float32")

        return (
            torch.from_numpy(np.ascontiguousarray(x)),
            torch.from_numpy(np.ascontiguousarray(c)),
            torch.from_numpy(y),
            torch.from_numpy(m),
            torch.from_numpy(bi.copy()),
            torch.from_numpy(pos.copy()),
        )


def make_loader(dataset: WindowDataset, batch_size: int, shuffle: bool,
                generator: torch.Generator = None, drop_last: bool = False) -> DataLoader:
    """用 BatchSampler 一次取整批，绕开逐样本 collate 的 Python 开销。"""
    base = (RandomSampler(dataset, generator=generator) if shuffle
            else SequentialSampler(dataset))
    sampler = BatchSampler(base, batch_size=batch_size, drop_last=drop_last)
    # batch_size=None 表示 sampler 已给出整批索引，dataset 直接返回整批张量
    return DataLoader(dataset, sampler=sampler, batch_size=None, collate_fn=lambda b: b)
