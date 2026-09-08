# -*- coding: utf-8 -*-
"""
绘制"标签缺失鲁棒性消融实验"机制示意图（时间轴打洞 + 训练/评估管线）。

对应实验：
- src/training/multi_task_lstm_ablation_wl2d_repeat.py       （MCAR 逐点随机缺失）
- src/training/multi_task_lstm_ablation_realistic_missing.py （段状缺失，真实故障 ECDF）

机制要点（已核对代码）：
1. 完整的径流 Q、水位 h 标签序列；
2. 按目标缺失率人工注入缺失（置 NaN），两种模式：
   - MCAR：逐点随机挖空（create_missing_mask 随机选点）；
   - 段状：从真实故障段长 ECDF（fault_recovery_Q/H.csv）bootstrap 采样，挖连续缺失段；
3. 挖洞后的标签用于训练 / 验证，训练时 loss 自动跳过 NaN（缺失点不计损失）；
4. 测试集始终使用原始完整标签（wl2d_repeat.py:901）→ 在完整集上评估 NSE；
5. 缺失率档位 0/10/30/50%（只缺径流 / 只缺水位 / 两者都缺），每档 3 个随机种子重复取均值±方差。

产出：outputs/ppt-assets/ablation_mechanism.png
"""

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import rcParams
from matplotlib.patches import FancyBboxPatch, Rectangle, FancyArrowPatch


# ==================== 中文字体 ====================
rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
rcParams["axes.unicode_minus"] = False

# ==================== 字号 ====================
FS_TITLE = 20
FS_SUB = 12.5
FS_ZONE = 13.5     # 分区标题 ①②③…
FS_BOX = 12.5
FS_SMALL = 11
FS_STRIP = 14      # Q / h 行首标签

# ==================== 配色 ====================
C_Q = "#CFE2F3"; C_Q_E = "#3D85C6"     # 径流标签格：蓝
C_H = "#D9EAD3"; C_H_E = "#6AA84F"     # 水位标签格：绿
C_MISS = "#FFFFFF"; C_MISS_E = "#CC4444"  # 缺失格：白底红虚边
C_MODEL = "#F4CCCC"; C_MODEL_E = "#CC6666"
C_EVAL = "#FCE5CD"; C_EVAL_E = "#E69138"
C_ARROW = "#555555"


def draw_strip(ax, x0, yc, n, cw, ch, missing, valid_fc, valid_ec, label=None):
    """画一行标签格：filled=有效标签，white 虚边=缺失(NaN)。行竖直居中于 yc。"""
    for i in range(n):
        is_miss = i in missing
        rect = Rectangle(
            (x0 + i * cw, yc - ch / 2), cw * 0.82, ch,
            facecolor=C_MISS if is_miss else valid_fc,
            edgecolor=C_MISS_E if is_miss else valid_ec,
            linewidth=1.2, linestyle=(0, (2, 1.4)) if is_miss else "solid",
            zorder=3,
        )
        ax.add_patch(rect)
    if label is not None:
        ax.text(x0 - 0.16, yc, label, ha="right", va="center",
                fontsize=FS_STRIP, fontstyle="italic", color="#333333")


def draw_box(ax, cx, cy, w, h, text, fc, ec, fontsize=FS_BOX, fontweight="normal"):
    box = FancyBboxPatch(
        (cx - w / 2, cy - h / 2), w, h,
        boxstyle="round,pad=0.02,rounding_size=0.1",
        facecolor=fc, edgecolor=ec, linewidth=1.8, zorder=3,
    )
    ax.add_patch(box)
    ax.text(cx, cy, text, ha="center", va="center", fontsize=fontsize,
            fontweight=fontweight, zorder=4)


def draw_arrow(ax, p0, p1, color=C_ARROW, lw=2.0, connectionstyle="arc3,rad=0"):
    ax.add_patch(FancyArrowPatch(
        p0, p1, arrowstyle="-|>", mutation_scale=18, color=color,
        linewidth=lw, shrinkA=3, shrinkB=3, connectionstyle=connectionstyle, zorder=2,
    ))


def main():
    fig, ax = plt.subplots(figsize=(14.2, 8.4))
    ax.set_xlim(0, 14.2)
    ax.set_ylim(0, 8.5)
    ax.axis("off")

    # 条带参数
    n = 11
    cw, ch = 0.30, 0.36
    strip_w = n * cw

    # ==================== 标题 ====================
    ax.text(7.1, 8.15, "WL2D 多任务模型 · 标签缺失鲁棒性消融实验（机制示意）",
            ha="center", va="center", fontsize=FS_TITLE, fontweight="bold")
    ax.text(7.1, 7.72,
            "向标签人工注入缺失（两种模式）→ 训练时 loss 跳过缺失点 → 在完整测试集上评估",
            ha="center", va="center", fontsize=FS_SUB, color="#555555")

    # ==================== ① 完整标签序列（左中） ====================
    xc0 = 1.35
    yc_Q0, yc_h0 = 4.42, 3.9
    ax.text(xc0 + strip_w / 2, 5.15, "① 完整标签序列",
            ha="center", va="center", fontsize=FS_ZONE, fontweight="bold", color="#333333")
    ax.text(xc0 + strip_w / 2, 4.83, "(Q 径流 · h 水位)",
            ha="center", va="center", fontsize=FS_SMALL, color="#666666")
    draw_strip(ax, xc0, yc_Q0, n, cw, ch, [], C_Q, C_Q_E, label="Q")
    draw_strip(ax, xc0, yc_h0, n, cw, ch, [], C_H, C_H_E, label="h")

    # 格子图例（① 下方）
    lx, ly = xc0 + 0.1, 3.15
    ax.add_patch(Rectangle((lx, ly - 0.16), 0.34, 0.32, facecolor=C_Q, edgecolor=C_Q_E, lw=1.2))
    ax.text(lx + 0.46, ly, "有效标签", ha="left", va="center", fontsize=FS_SMALL, color="#444")
    ax.add_patch(Rectangle((lx + 2.0, ly - 0.16), 0.34, 0.32, facecolor=C_MISS,
                           edgecolor=C_MISS_E, lw=1.2, linestyle=(0, (2, 1.4))))
    ax.text(lx + 2.46, ly, "缺失 (置 NaN)", ha="left", va="center", fontsize=FS_SMALL, color="#444")

    # ==================== 分叉：注入人工缺失 ====================
    x_fork = xc0 + strip_w + 0.55       # 完整块右侧
    xb0 = x_fork + 0.95                 # 两个缺失块的左端
    ax.text(x_fork + 0.1, 4.18, "人工注入\n标签缺失", ha="center", va="center",
            fontsize=FS_SMALL, color="#333333")

    # ==================== ② MCAR 逐点随机缺失（上分支） ====================
    yc_Q1, yc_h1 = 6.42, 5.9
    ax.text(xb0 + strip_w / 2, 7.12, "② MCAR：逐点随机缺失",
            ha="center", va="center", fontsize=FS_ZONE, fontweight="bold", color=C_MISS_E)
    draw_strip(ax, xb0, yc_Q1, n, cw, ch, [2, 6, 9], C_Q, C_Q_E, label="Q")
    draw_strip(ax, xb0, yc_h1, n, cw, ch, [1, 5, 9], C_H, C_H_E, label="h")
    draw_arrow(ax, (x_fork + 0.05, 4.62), (xb0 - 0.12, yc_h1 - 0.05),
               connectionstyle="arc3,rad=0.18")

    # ==================== ③ 段状缺失（下分支，真实故障 ECDF） ====================
    yc_Q2, yc_h2 = 2.42, 1.9
    ax.text(xb0 + strip_w / 2, 3.12, "③ 段状缺失（真实故障段长 ECDF）",
            ha="center", va="center", fontsize=FS_ZONE, fontweight="bold", color=C_MISS_E)
    draw_strip(ax, xb0, yc_Q2, n, cw, ch, [3, 4, 5], C_Q, C_Q_E, label="Q")
    draw_strip(ax, xb0, yc_h2, n, cw, ch, [6, 7, 8], C_H, C_H_E, label="h")
    draw_arrow(ax, (x_fork + 0.05, 3.7), (xb0 - 0.12, yc_Q2 + 0.05),
               connectionstyle="arc3,rad=-0.18")

    # ==================== ④ WL2D 模型 ====================
    xb1 = xb0 + strip_w
    x_model = 11.35
    draw_box(ax, x_model, 4.3, 2.5, 1.35,
             "④ WL2D 模型\n共享 LSTM → 水位/径流双头\n(loss 跳过缺失标签)",
             C_MODEL, C_MODEL_E, fontsize=FS_BOX)
    # 挖洞标签（训练/验证）→ 模型
    draw_arrow(ax, (xb1 + 0.15, yc_h1), (x_model - 1.28, 4.72),
               connectionstyle="arc3,rad=-0.12")
    draw_arrow(ax, (xb1 + 0.15, yc_Q2), (x_model - 1.28, 3.88),
               connectionstyle="arc3,rad=0.12")
    ax.text(8.35, 4.55, "挖洞标签 → 训练 / 验证", ha="center",
            va="center", fontsize=FS_SMALL, color="#666666")

    # ==================== ⑤ 完整测试集评估 ====================
    draw_box(ax, x_model, 1.95, 2.9, 1.15,
             "⑤ 完整测试集评估\nNSE（3 seeds 取均值 ± 方差）",
             C_EVAL, C_EVAL_E, fontsize=FS_BOX)
    draw_arrow(ax, (x_model, 4.3 - 0.68), (x_model, 1.95 + 0.58), lw=2.2)
    ax.text(x_model + 1.72, 3.15, "测试集\n始终完整\n(无人工缺失)", ha="left", va="center",
            fontsize=FS_SMALL, color="#666666")

    # ==================== 底部：缺失率档位说明 ====================
    bar = FancyBboxPatch((0.7, 0.35), 12.8, 0.62,
                         boxstyle="round,pad=0.02,rounding_size=0.08",
                         facecolor="#F3F3F3", edgecolor="#BBBBBB", linewidth=1.4, zorder=1)
    ax.add_patch(bar)
    ax.text(7.1, 0.66,
            "缺失率档位  0% / 10% / 30% / 50%     ·     只缺径流 · 只缺水位 · 两者同时缺"
            "     ·     每档 3 个随机种子重复 → 均值 ± 方差",
            ha="center", va="center", fontsize=FS_SMALL, color="#333333")

    # ==================== 保存 ====================
    out_dir = os.path.join("outputs", "ppt-assets")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "ablation_mechanism.png")
    fig.savefig(out_path, dpi=300, bbox_inches="tight", facecolor="white", pad_inches=0.15)
    print("[done] saved to:", out_path)


if __name__ == "__main__":
    main()
