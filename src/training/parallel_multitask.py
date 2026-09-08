"""共享 LSTM 编码器、Q / 水位独立预测头，以及实验架构选择。"""

import torch
from torch import nn


class ParallelMultiTaskLSTM(nn.Module):
    """两个任务共享表示和 dropout，各自通过线性头输出预测。

    返回顺序与现有训练循环一致：(pred_flow, pred_waterlevel)。
    两个头之间没有前向连接，但两个任务的损失都能更新共享编码器。
    """

    def __init__(self, input_size, hidden_size=64, dropout_rate=0.2,
                 task_weights=None):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=2,
            bias=True,
            batch_first=True,
        )
        self.dropout = nn.Dropout(dropout_rate)
        self.fc_flow = nn.Linear(hidden_size, 1)
        self.fc_waterlevel = nn.Linear(hidden_size, 1)
        self.task_weights = dict(
            task_weights if task_weights is not None
            else {"flow": 1.0, "waterlevel": 1.0}
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        _, (h_n, _) = self.lstm(x)
        hidden = self.dropout(h_n[-1])
        return self.fc_flow(hidden), self.fc_waterlevel(hidden)


def architecture_label(architecture):
    labels = {
        "parallel": "共享 LSTM + 两个独立预测头",
        "wl2d": "WL2D 水位→径流级联",
    }
    if architecture not in labels:
        raise ValueError(f"未知模型架构: {architecture!r}")
    return labels[architecture]


def build_multitask_model(architecture, wl2d_class, *, input_size,
                         hidden_size=64, dropout_rate=0.2, task_weights=None,
                         wl_proj_size=16, stop_gradient=False):
    """保留旧 WL2D 类，独立双头模型不创建任何级联层。"""
    architecture_label(architecture)
    common = dict(input_size=input_size, hidden_size=hidden_size,
                  dropout_rate=dropout_rate, task_weights=task_weights)
    if architecture == "parallel":
        if stop_gradient:
            raise ValueError("parallel 没有级联通路，不适用 stop_gradient")
        return ParallelMultiTaskLSTM(**common)
    return wl2d_class(**common, wl_proj_size=wl_proj_size,
                     stop_gradient=stop_gradient)


def model_metadata(model, architecture):
    """将实际架构和重建所需参数写入 checkpoint，避免混用旧模型。"""
    architecture_label(architecture)
    config = {
        "input_size": model.lstm.input_size,
        "hidden_size": model.lstm.hidden_size,
        "dropout_rate": model.dropout.p,
        "task_weights": dict(model.task_weights),
    }
    if architecture == "wl2d":
        config.update(wl_proj_size=model.wl_proj[0].out_features,
                      stop_gradient=model.stop_gradient)
    return {
        "architecture": architecture,
        "model_config": config,
        "num_parameters": sum(p.numel() for p in model.parameters()),
    }
