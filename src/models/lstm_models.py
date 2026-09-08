"""共享编码器的单任务 / 多任务 LSTM 家族。

与旧的 src/training/parallel_multitask.py 相比：
- 输入改为 (x, c) 两路。静态属性不在 Dataset 里 np.tile 成 [B, L, A]，
  而是在模型里广播后拼接，省掉大量内存带宽。
- 增加 CapacityMatchedMultiTaskLSTM，用于回答审稿意见 Major 1：
  它与 WL2D 的非线性深度和分支形状完全一致，唯一差别是那条分支的输入
  取自 hidden 的一个自由标量投影，而不是水位预测值。

为什么需要这个对照
------------------
WL2D 中 ``pred_wl = fc_waterlevel(hidden)`` 是 hidden 的线性函数，记为
标量 s；``wl_feat = ReLU(Linear(1→P)(pred_wl))`` 的 P 个通道全部是同一个
标量 s 的仿射变换过 ReLU。而径流头本来就直接看到完整的 hidden，因此这条
"水位→径流通路"并未提供任何独立信息，其真实作用是给径流头增加了一个
沿单一方向的分段线性特征，且该方向被水位损失约束。参数量上，这条通路
只有 2P + P = 48 个参数（P=16），占全模型约 0.09%。

要判断"级联是否真有作用"，就必须与"同样加一条分支、但方向不受水位损失
约束"的模型比较，这正是 CapacityMatchedMultiTaskLSTM。
"""

import torch
from torch import nn

TASKS = ("flow", "waterlevel")
ARCHITECTURES = ("single_flow", "single_waterlevel", "dual_head",
                 "capacity_matched", "wl2d", "wl2d_stopgrad", "wl2d_reverse")


class _SharedEncoder(nn.Module):
    """两层 LSTM 编码器，输入为强迫序列与广播后的静态属性。"""

    def __init__(self, forcing_size: int, attr_size: int, hidden_size: int,
                 dropout_rate: float, num_layers: int = 2):
        super().__init__()
        self.forcing_size = forcing_size
        self.attr_size = attr_size
        self.lstm = nn.LSTM(
            input_size=forcing_size + attr_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            bias=True,
            batch_first=True,
        )
        self.dropout = nn.Dropout(p=dropout_rate)

    def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        # 静态属性沿时间轴广播，避免在 Dataset 里物化 [B, L, A]
        c_expanded = c.unsqueeze(1).expand(-1, x.shape[1], -1)
        seq = torch.cat((x, c_expanded), dim=-1)
        _, (h_n, _) = self.lstm(seq)
        return self.dropout(h_n[-1])


class BaseModel(nn.Module):
    """统一接口：forward(x, c) -> {task: [B, 1]}。"""

    architecture = None
    tasks = ()

    def encode(self, x, c):
        return self.encoder(x, c)

    def n_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def config(self) -> dict:
        return {
            "architecture": self.architecture,
            "tasks": list(self.tasks),
            "forcing_size": self.encoder.forcing_size,
            "attr_size": self.encoder.attr_size,
            "hidden_size": self.encoder.lstm.hidden_size,
            "num_layers": self.encoder.lstm.num_layers,
            "dropout_rate": self.encoder.dropout.p,
            "n_parameters": self.n_parameters(),
        }


class SingleTaskLSTM(BaseModel):
    """单任务基线：共享编码器 + 一个线性头。"""

    def __init__(self, task: str, forcing_size: int, attr_size: int,
                 hidden_size: int = 64, dropout_rate: float = 0.2):
        super().__init__()
        if task not in TASKS:
            raise ValueError(f"task 必须是 {TASKS} 之一，收到 {task!r}")
        self.architecture = f"single_{task}"
        self.tasks = (task,)
        self.encoder = _SharedEncoder(forcing_size, attr_size, hidden_size, dropout_rate)
        self.head = nn.Linear(hidden_size, 1)

    def forward(self, x, c):
        return {self.tasks[0]: self.head(self.encode(x, c))}


class ParallelMultiTaskLSTM(BaseModel):
    """主线模型：共享编码器 + Q / h 两个独立线性头，头之间无前向连接。"""

    architecture = "dual_head"
    tasks = TASKS

    def __init__(self, forcing_size: int, attr_size: int,
                 hidden_size: int = 64, dropout_rate: float = 0.2):
        super().__init__()
        self.encoder = _SharedEncoder(forcing_size, attr_size, hidden_size, dropout_rate)
        self.fc_flow = nn.Linear(hidden_size, 1)
        self.fc_waterlevel = nn.Linear(hidden_size, 1)

    def forward(self, x, c):
        hidden = self.encode(x, c)
        return {"flow": self.fc_flow(hidden), "waterlevel": self.fc_waterlevel(hidden)}


class WL2DMultiTaskLSTM(BaseModel):
    """WL2D 级联：水位预测经投影后拼接进径流头。

    reverse=True 时改为 D→WL 反向级联，用于检验方向性主张。
    stop_gradient=True 时切断径流损失经该通路回传到水位头的路径。
    """

    tasks = TASKS

    def __init__(self, forcing_size: int, attr_size: int, hidden_size: int = 64,
                 dropout_rate: float = 0.2, proj_size: int = 16,
                 stop_gradient: bool = False, reverse: bool = False):
        super().__init__()
        self.architecture = ("wl2d_reverse" if reverse else
                             "wl2d_stopgrad" if stop_gradient else "wl2d")
        self.stop_gradient = stop_gradient
        self.reverse = reverse
        self.proj_size = proj_size
        self.encoder = _SharedEncoder(forcing_size, attr_size, hidden_size, dropout_rate)
        # upstream = 先预测的任务，downstream = 接收上游预测的任务
        self.up_task, self.down_task = ("flow", "waterlevel") if reverse else ("waterlevel", "flow")
        self.fc_up = nn.Linear(hidden_size, 1)
        self.proj = nn.Sequential(nn.Linear(1, proj_size), nn.ReLU())
        self.fc_down = nn.Linear(hidden_size + proj_size, 1)

    def forward(self, x, c):
        hidden = self.encode(x, c)
        pred_up = self.fc_up(hidden)
        signal = pred_up.detach() if self.stop_gradient else pred_up
        feat = self.proj(signal)
        pred_down = self.fc_down(torch.cat([hidden, feat], dim=-1))
        return {self.up_task: pred_up, self.down_task: pred_down}

    def config(self):
        cfg = super().config()
        cfg.update(proj_size=self.proj_size, stop_gradient=self.stop_gradient,
                   reverse=self.reverse)
        return cfg


class CapacityMatchedMultiTaskLSTM(BaseModel):
    """Major 1 对照：与 WL2D 分支形状相同，但分支输入不是水位预测。

    径流头前挂 ``ReLU(Linear(1→P))`` 分支，其输入是 hidden 的一个自由标量
    投影 ``aux_scalar(hidden)``，不接任何水位监督。相对 WL2D 多出
    ``aux_scalar`` 的 hidden_size+1 个参数（默认 65 个，约占全模型 0.12%）；
    训练脚本会如实记录并报告两者的精确参数量，不假装完全相等。
    """

    architecture = "capacity_matched"
    tasks = TASKS

    def __init__(self, forcing_size: int, attr_size: int, hidden_size: int = 64,
                 dropout_rate: float = 0.2, proj_size: int = 16):
        super().__init__()
        self.proj_size = proj_size
        self.encoder = _SharedEncoder(forcing_size, attr_size, hidden_size, dropout_rate)
        self.fc_waterlevel = nn.Linear(hidden_size, 1)
        self.aux_scalar = nn.Linear(hidden_size, 1)      # 自由标量，不受水位损失约束
        self.proj = nn.Sequential(nn.Linear(1, proj_size), nn.ReLU())
        self.fc_flow = nn.Linear(hidden_size + proj_size, 1)

    def forward(self, x, c):
        hidden = self.encode(x, c)
        feat = self.proj(self.aux_scalar(hidden))
        return {
            "flow": self.fc_flow(torch.cat([hidden, feat], dim=-1)),
            "waterlevel": self.fc_waterlevel(hidden),
        }

    def config(self):
        cfg = super().config()
        cfg.update(proj_size=self.proj_size)
        return cfg


def build_model(architecture: str, forcing_size: int, attr_size: int,
                hidden_size: int = 64, dropout_rate: float = 0.2,
                proj_size: int = 16) -> BaseModel:
    """按名称构造模型。所有架构共用同一编码器超参。"""
    common = dict(forcing_size=forcing_size, attr_size=attr_size,
                  hidden_size=hidden_size, dropout_rate=dropout_rate)
    if architecture == "single_flow":
        return SingleTaskLSTM("flow", **common)
    if architecture == "single_waterlevel":
        return SingleTaskLSTM("waterlevel", **common)
    if architecture == "dual_head":
        return ParallelMultiTaskLSTM(**common)
    if architecture == "capacity_matched":
        return CapacityMatchedMultiTaskLSTM(proj_size=proj_size, **common)
    if architecture == "wl2d":
        return WL2DMultiTaskLSTM(proj_size=proj_size, **common)
    if architecture == "wl2d_stopgrad":
        return WL2DMultiTaskLSTM(proj_size=proj_size, stop_gradient=True, **common)
    if architecture == "wl2d_reverse":
        return WL2DMultiTaskLSTM(proj_size=proj_size, reverse=True, **common)
    raise ValueError(f"未知架构 {architecture!r}，可选: {ARCHITECTURES}")
