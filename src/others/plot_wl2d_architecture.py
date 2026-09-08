# -*- coding: utf-8 -*-
"""
绘制 WL2D 级联多任务 LSTM 模型架构图（时序展开风格）。

对应模型：src/training/multi_task_lstm_wl2d.py 中的 MultiTaskLSTM
- 共享 2 层 LSTM（hidden=64），单向 many-to-one（用末步隐状态预测）
- Stage-1 水位头 fc_wl(64->1) 先预测水位 h
- Stage-2 水位->径流级联：wl_proj(1->16, ReLU) 投影后与 hidden(64) 拼接为 80 维，
  送入径流头 fc_flow(80->1) 预测径流 Q
- 加权损失 loss = w1*loss_Q + w2*loss_h（默认 w1=w2=1.0，可调）

产出：outputs/ppt-assets/wl2d_architecture.png
"""

import os

import matplotlib
matplotlib.use("Agg")  # 无显示环境下也能保存图片
import matplotlib.pyplot as plt
from matplotlib import rcParams
from matplotlib.patches import FancyBboxPatch, Circle, FancyArrowPatch


# ==================== 中文字体配置 ====================
rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
rcParams["axes.unicode_minus"] = False

# ==================== 字号（整体放大，便于 PPT 观看） ====================
FS_TITLE = 22
FS_SUB = 14
FS_LSTM = 15       # LSTM 框
FS_BOX = 12.5      # 预测头 / 投影框
FS_VAR = 16        # 变量圆圈
FS_OBS = 13.5      # 观测圆圈
FS_SIDE = 13       # 左侧说明文字
FS_ANNO = 12       # 各类标注
FS_SMALL = 11      # 小号标注
FS_LOSS = 14       # loss 公式框
FS_LEG = 13.5      # 图例
FS_LEGT = 14.5     # 图例标题

# ==================== 配色（对齐参考图视觉语言） ====================
C_INPUT = "#D9EAD3"; C_INPUT_E = "#6AA84F"     # 变量圆圈：绿
C_LSTM = "#F4CCCC"; C_LSTM_E = "#CC6666"       # LSTM 框：粉
C_HEAD = "#CFE2F3"; C_HEAD_E = "#3D85C6"       # 预测头 / 投影框：蓝灰
C_SRC = "#D9D9D9"; C_SRC_E = "#999999"         # 数据源框：灰
C_ATTR = "#EFEFEF"; C_ATTR_E = "#999999"       # 属性框：浅灰
C_LOSS = "#FCE5CD"; C_LOSS_E = "#E69138"       # loss 框：橙
C_CAT = "#FFF2CC"; C_CAT_E = "#BF9000"         # 拼接节点：淡黄
C_OBS = "#FFFFFF"; C_OBS_E = "#666666"         # 观测圆圈：白
C_CASCADE = "#CC0000"                          # 级联通路箭头：红
C_ARROW = "#444444"                            # 普通箭头


# ==================== 绘图 helper ====================
def draw_box(ax, cx, cy, w, h, text, fc, ec, fontsize=FS_BOX, fontweight="normal",
             text_color="black", rounding=0.12):
    box = FancyBboxPatch(
        (cx - w / 2, cy - h / 2), w, h,
        boxstyle=f"round,pad=0.02,rounding_size={rounding}",
        facecolor=fc, edgecolor=ec, linewidth=1.8, zorder=3,
    )
    ax.add_patch(box)
    ax.text(cx, cy, text, ha="center", va="center", fontsize=fontsize,
            fontweight=fontweight, color=text_color, zorder=4)
    return box


def draw_circle(ax, cx, cy, r, text, fc, ec, fontsize=FS_VAR, fontweight="normal",
                text_color="black"):
    c = Circle((cx, cy), r, facecolor=fc, edgecolor=ec, linewidth=1.8, zorder=3)
    ax.add_patch(c)
    ax.text(cx, cy, text, ha="center", va="center", fontsize=fontsize,
            fontweight=fontweight, color=text_color, zorder=4)
    return c


def draw_arrow(ax, p0, p1, color=C_ARROW, lw=1.8, style="-|>", ls="-",
               shrinkA=3, shrinkB=3, connectionstyle="arc3,rad=0"):
    arr = FancyArrowPatch(
        p0, p1, arrowstyle=style, mutation_scale=18, color=color,
        linewidth=lw, linestyle=ls, shrinkA=shrinkA, shrinkB=shrinkB,
        connectionstyle=connectionstyle, zorder=2,
    )
    ax.add_patch(arr)
    return arr


def main():
    fig, ax = plt.subplots(figsize=(14.0, 9.1))
    ax.set_xlim(0, 14.35)
    ax.set_ylim(0, 9.6)
    ax.axis("off")

    # -------- 各行纵坐标 --------
    y_src = 0.62
    y_in = 1.95
    y_attr = 3.12
    y_lstm = 4.4
    y_box = 5.95
    y_pred = 7.2
    y_obs = 8.15

    # 时间步 x 位置：t=1, t=2, ⋯, t=T
    xs = [2.4, 4.5, 7.9]
    x_dots = 6.2
    xT = xs[-1]
    r_in, r_pred, r_obs = 0.48, 0.52, 0.5
    box_w, box_h = 1.45, 0.82

    # 预测头区域 x 位置（精简：只保留两个预测头，级联用一条红箭头表示）
    x_flow = 6.0      # 径流头 / Q̂ / Q_obs 列
    x_wl = 9.5        # 水位头 / ĥ / h_obs 列

    # ==================== 标题 ====================
    ax.text(7.0, 9.32, "多流域 水位→径流 级联多任务 LSTM（WL2D）",
            ha="center", va="center", fontsize=FS_TITLE, fontweight="bold")
    ax.text(6.6, 8.88, "共享 LSTM 编码器 · 水位→径流级联 · 加权多任务损失",
            ha="center", va="center", fontsize=13, color="#555555")

    # ==================== 1. 数据源框（底部） ====================
    draw_box(ax, 1.1, y_src, 2.0, 0.72, "MSWEP\n3h 降水", C_SRC, C_SRC_E, fontsize=FS_ANNO)
    draw_box(ax, 3.65, y_src, 2.2, 0.72, "CAMELSH 气象\n(气温 / 辐射)", C_SRC, C_SRC_E, fontsize=FS_ANNO)
    draw_arrow(ax, (1.1, y_src + 0.36), (xs[0] - 0.16, y_in - r_in),
               connectionstyle="arc3,rad=0.15")
    draw_arrow(ax, (3.65, y_src + 0.36), (xs[0] + 0.16, y_in - r_in),
               connectionstyle="arc3,rad=-0.15")

    # ==================== 2. 输入圆圈行（气象强迫） ====================
    in_labels = [r"$x^{(1)}$", r"$x^{(2)}$", r"$x^{(T)}$"]
    for x, lab in zip(xs, in_labels):
        draw_circle(ax, x, y_in, r_in, lab, C_INPUT, C_INPUT_E, fontsize=FS_VAR)
    ax.text(x_dots, y_in, r"$\cdots$", ha="center", va="center", fontsize=24)
    ax.text(xs[0] - 1.1, y_in, "气象强迫\nP / T / R\n(×3)",
            ha="right", va="center", fontsize=FS_SIDE, color="#3a7a30")

    # ==================== 3. Attributes 属性总线 ====================
    attr_cx = 1.6
    draw_box(ax, attr_cx, y_attr, 2.6, 0.76,
             "流域静态属性\nAttributes  ×12", C_ATTR, C_ATTR_E, fontsize=FS_ANNO)
    ax.plot([attr_cx + 1.3, xT + 0.95], [y_attr, y_attr], color=C_ATTR_E, lw=1.8, zorder=1)

    # ==================== 4. LSTM cell 行 ====================
    for x in xs:
        draw_box(ax, x, y_lstm, box_w, box_h, "LSTM", C_LSTM, C_LSTM_E,
                 fontsize=FS_LSTM, fontweight="bold")
        draw_arrow(ax, (x, y_in + r_in), (x, y_lstm - box_h / 2))
        draw_arrow(ax, (x, y_attr), (x, y_lstm - box_h / 2), color=C_ATTR_E,
                   lw=1.4, ls=(0, (4, 2)))
    draw_arrow(ax, (xs[0] + box_w / 2, y_lstm), (xs[1] - box_w / 2, y_lstm), lw=2.0)
    draw_arrow(ax, (xs[1] + box_w / 2, y_lstm), (x_dots - 0.32, y_lstm), lw=2.0)
    ax.text(x_dots, y_lstm, r"$\cdots$", ha="center", va="center", fontsize=24)
    draw_arrow(ax, (x_dots + 0.32, y_lstm), (xT - box_w / 2, y_lstm), lw=2.0)
    ax.text(xs[0] - 1.1, y_lstm, "共享 LSTM\n2 层\nhidden=64",
            ha="right", va="center", fontsize=FS_SIDE, color="#a03a3a")

    # 输入窗口标注
    ax.annotate("", xy=(xT + 0.7, 1.36), xytext=(xs[0] - 0.7, 1.36),
                arrowprops=dict(arrowstyle="<->", color="#888888", lw=1.4))
    ax.text(xT + 0.7, 1.06, "输入窗口 T = 168 步（21 天 @ 3h）",
            ha="right", va="center", fontsize=FS_ANNO, color="#666666")

    # ==================== 5. 末步预测头（WL2D 级联，核心，精简版） ====================
    y_top_lstm = y_lstm + box_h / 2
    y_junc = 5.15

    draw_arrow(ax, (xT, y_top_lstm), (xT, y_junc), lw=2.2, shrinkB=1)
    ax.text(xT + 0.22, (y_top_lstm + y_junc) / 2, r"$h_T$",
            ha="left", va="center", fontsize=FS_ANNO + 1, color="#333333")

    # h_T 分别送入两个预测头
    draw_arrow(ax, (xT, y_junc), (x_flow + 0.55, y_box - 0.41), lw=2.0, connectionstyle="arc3,rad=0.12")
    draw_arrow(ax, (xT, y_junc), (x_wl - 0.45, y_box - 0.41), lw=2.0, connectionstyle="arc3,rad=-0.12")

    # Stage-1：水位头 -> 水位预测
    draw_box(ax, x_wl, y_box, 1.95, 0.82, "① 水位头\nfc_wl (64→1)", C_HEAD, C_HEAD_E, fontsize=FS_BOX)
    draw_arrow(ax, (x_wl, y_box + 0.41), (x_wl, y_pred - r_pred))
    draw_circle(ax, x_wl, y_pred, r_pred, r"$\hat{h}_T$", C_INPUT, C_INPUT_E,
                fontsize=FS_VAR, fontweight="bold")

    # Stage-2：径流头（输入 = h_T 隐层 64 维 + 水位经 wl_proj 投影 16 维）
    draw_box(ax, x_flow, y_box, 2.15, 0.82, "② 径流头\nfc_flow (64+16→1)", C_HEAD, C_HEAD_E, fontsize=FS_BOX)
    draw_arrow(ax, (x_flow, y_box + 0.41), (x_flow, y_pred - r_pred))
    draw_circle(ax, x_flow, y_pred, r_pred, r"$\hat{Q}_T$", C_LSTM, C_LSTM_E,
                fontsize=FS_VAR, fontweight="bold")

    # 水位 → 径流 级联通路（红色：水位预测经 wl_proj 投影后拼接入径流头）
    draw_arrow(ax, (x_wl - r_pred - 0.05, y_pred - 0.15), (x_flow + 1.05, y_box + 0.36),
               color=C_CASCADE, lw=2.8, connectionstyle="arc3,rad=0.3")

    # ==================== 6. 观测与损失（顶部） ====================
    draw_circle(ax, x_flow, y_obs, r_obs, r"$Q_{obs}$", C_OBS, C_OBS_E, fontsize=FS_OBS)
    draw_circle(ax, x_wl, y_obs, r_obs, r"$h_{obs}$", C_OBS, C_OBS_E, fontsize=FS_OBS)
    draw_arrow(ax, (x_flow, y_pred + r_pred), (x_flow, y_obs - r_obs),
               color=C_LOSS_E, lw=2.1, style="<|-|>")
    draw_arrow(ax, (x_wl, y_pred + r_pred), (x_wl, y_obs - r_obs),
               color=C_LOSS_E, lw=2.1, style="<|-|>")
    ax.text(x_flow - r_pred - 0.12, (y_pred + y_obs) / 2 + 0.18, r"$loss_Q$",
            ha="right", va="center", fontsize=FS_ANNO + 1, color=C_LOSS_E)
    ax.text(x_wl - r_pred - 0.12, (y_pred + y_obs) / 2 + 0.18, r"$loss_h$",
            ha="right", va="center", fontsize=FS_ANNO + 1, color=C_LOSS_E)

    # loss 公式框（右上，精简为两行）
    loss_cx, loss_cy = 12.1, 7.75
    draw_box(ax, loss_cx, loss_cy, 3.85, 1.2,
             r"$loss = w_1 \cdot loss_Q + w_2 \cdot loss_h$"
             "\n" r"默认 $w_1{=}w_2{=}1.0$（可调）",
             C_LOSS, C_LOSS_E, fontsize=13.5)
    draw_arrow(ax, (x_wl + r_obs + 0.05, y_obs), (loss_cx - 1.95, loss_cy + 0.05),
               color=C_LOSS_E, lw=1.8, connectionstyle="arc3,rad=-0.15")

    # ==================== 级联通路说明（红色，取代整块图例） ====================
    cx0, cyy = 10.75, 5.25
    ax.annotate("", xy=(cx0 + 0.7, cyy), xytext=(cx0, cyy),
                arrowprops=dict(arrowstyle="-|>", color=C_CASCADE, lw=2.8))
    ax.text(cx0 + 0.92, cyy, "水位 → 径流 级联通路", ha="left", va="center",
            fontsize=FS_LEG, color=C_CASCADE, fontweight="bold")
    ax.text(cx0 + 0.92, cyy - 0.46, "经 wl_proj 投影后拼接入径流头", ha="left",
            va="center", fontsize=FS_SMALL, color=C_CASCADE)
    ax.text(cx0 + 0.92, cyy - 0.86, "端到端联合训练 (stop_grad=False)", ha="left",
            va="center", fontsize=FS_SMALL, color=C_CASCADE)

    # ==================== 保存 ====================
    out_dir = os.path.join("outputs", "ppt-assets")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "wl2d_architecture.png")
    fig.savefig(out_path, dpi=300, bbox_inches="tight", facecolor="white", pad_inches=0.15)
    print("[done] saved to:", out_path)


if __name__ == "__main__":
    main()
