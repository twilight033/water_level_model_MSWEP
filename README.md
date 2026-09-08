# water_level_model

基于气象强迫与流域属性联合预测径流（Q）与水位（h）的多任务 LSTM。
主线模型为**共享 LSTM 编码器 + Q / h 两个独立预测头**；WL2D 级联降为
附录中的架构对照。

## 当前管线

数据管线已按审稿意见重建，所有模型共用同一份冻结划分、归一化统计量与
窗口集合，模型之间的差异只能来自架构本身。

```
data/camelsh_exported/     目标与强迫的 parquet 缓存（86 流域）
data/splits/               冻结划分、归一化统计量、掩膜缓存
data/coverage/             每流域 × 任务 × 分段的有效标签统计
results/runs/<run_key>/    每次运行的指标、训练历史、权重、时序
results/summary/           跨运行汇总与面向论文的表格
```

| 模块 | 职责 |
|---|---|
| `src/pipeline/splits.py` | 冻结每流域划分边界，与标签、掩膜、窗口长度完全解耦 |
| `src/pipeline/dataset.py` | 滑窗数据集：并集准入 + 逐任务掩膜 + 真实目标时刻 |
| `src/pipeline/masking.py` | 人工缺失（MCAR / 真实故障段 ECDF），仅作用于训练段 |
| `src/pipeline/normalization.py` | 统计量只用训练段原始标签，跨场景共用 |
| `src/models/lstm_models.py` | 单任务 / 双头 / 参数量匹配对照 / WL2D 及其变体 |
| `src/training/trainer.py` | 统一训练与评估循环，如实记录训练预算 |
| `src/training/run_matrix.py` | 实验矩阵编排，支持断点续跑 |
| `src/evaluation/` | 多指标、事件指标、配对检验与论文表格 |

## 换一台机器起步

`data/` 整体不入库，但 **`data/splits/` 与 `data/coverage/` 例外**——冻结划分、
归一化统计量和任务资格是全部模型可比性的契约，两台机器必须用同一份，不能各自
重算。这些是 0.3 MB 的纯文本，已随仓库分发。

```powershell
git pull
# 改 config.py 的 CAMELSH_DATA_PATH 为本机实际路径（本机为 F:/data）
uv venv; .\.venv\Scripts\activate; uv pip install -e .

# 重建 150 MB 的 parquet 缓存（约 5 分钟）。不要重跑 splits.py / normalization.py，
# 那会覆盖仓库里已冻结的划分与统计量。
.\.venv\Scripts\python.exe -X utf8 src/pipeline/export_targets.py
.\.venv\Scripts\python.exe -X utf8 src/pipeline/export_forcing.py
.\.venv\Scripts\python.exe -X utf8 src/pipeline/attributes.py

# 验证（30 条必须全过）
.\.venv\Scripts\python.exe -X utf8 -m unittest discover -s tests -v
```

缺失掩膜不入库，但由 `mask_seed` 确定性重建（子种子用 `zlib.crc32`，跨进程与
跨机器一致），首次用到时自动生成并缓存。

`results/` 不入库，跑完手动拷回。

## 训练协议（已固定）

固定 **30 轮**、**不早停**、**余弦学习率衰减**至初值 2%，batch 512。

实测依据：`patience=5` 早停时 6 个种子的最佳轮落在 3–18，验证分与最佳轮几乎
单调相关，种子标准差被抬到 0.0176——而待检测的架构差异只有约 0.008。去掉早停
后曲线仍以 ±0.03 震荡、最佳轮顶到第 28 轮（未收敛）；加余弦衰减后末 5 轮极差
降到 0.004–0.007，最佳轮回落到 11–18，种子极差从 0.047 降到 0.024。

**所有模型必须共用这一套协议**，否则架构之间不可比。

## 使用

```powershell
# 首次从零准备（需要 CAMELSH 原始数据集与 MSWEP CSV）
.\.venv\Scripts\python.exe -X utf8 src/pipeline/export_targets.py
.\.venv\Scripts\python.exe -X utf8 src/pipeline/export_forcing.py
.\.venv\Scripts\python.exe -X utf8 src/pipeline/splits.py
.\.venv\Scripts\python.exe -X utf8 src/pipeline/coverage.py
.\.venv\Scripts\python.exe -X utf8 src/pipeline/normalization.py

# 验证（跑正式实验前必须全部通过）
.\.venv\Scripts\python.exe -X utf8 -m unittest discover -s tests -v

# 批量实验
.\.venv\Scripts\python.exe -u -X utf8 src/training/run_matrix.py --scale smoke
.\.venv\Scripts\python.exe -u -X utf8 src/training/run_matrix.py

# 汇总为论文表格
.\.venv\Scripts\python.exe -X utf8 src/evaluation/report.py
```

CAMELSH 数据路径在 `config.py` 的 `CAMELSH_DATA_PATH` 配置（当前
`F:/data`，F 盘为可移动盘，运行前会自动校验目录结构）。

## 文档

- [时间聚合与标签对齐](docs/temporal_alignment.md) —— 逐变量区间定义与算例
- [双头多任务修订方案](docs/parallel_multitask_revision.md) —— 主线调整的由来

## 历史留档

`src/training/` 下的 `multi_task_lstm*.py`、`single_task_lstm*.py` 为旧实现，
其划分边界在 Dataset 构造时推断，会随标签、掩膜与窗口长度变化，产出的
数值不再用于论文。保留仅供回溯。
