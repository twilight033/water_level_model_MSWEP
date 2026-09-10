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
| `src/pipeline/attribute_sources.py` | 直读原始属性 CSV，不经 hydrodataset；base / extended 两套属性集 |
| `src/evaluation/` | 多指标、事件指标、配对检验与论文表格 |
| `src/evaluation/rating_baseline.py` | 率定关系稳定性、区域化率定曲线基线 |
| `src/evaluation/two_stage_baseline.py` | 两阶段基线：气象→预测水位→率定→径流 |
| `src/evaluation/attribute_diagnostics.py` | 逐流域性能与属性的关联诊断 |

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
.\.venv\Scripts\python.exe -X utf8 src/pipeline/attribute_sources.py

# 验证
.\.venv\Scripts\python.exe -X utf8 -m unittest discover -s tests -v
```

卡住或报错时，先跑逐步诊断定位（每步立即输出，最后打印的那行即卡点）：

```powershell
.\.venv\Scripts\python.exe -X utf8 -u src/pipeline/diagnose.py
```

**判据：无 ERROR、无 FAIL。** 未安装可选依赖 `hydrodataset` 时，
`test_parallel_multitask.py` 中 2 个历史留档脚本的兼容性用例会显示为
**skipped**，这是预期行为——那些脚本在模块顶层导入 hydrodataset，而新管线
建好 parquet 缓存后完全不需要它。真正要看的是 `test_pipeline.py` 的 32 条
（冻结划分、掩膜范围、样本集一致性、物理归一化无泄漏等）必须全过。

缺失掩膜不入库，但由 `mask_seed` 确定性重建（子种子用 `zlib.crc32`，跨进程与
跨机器一致），首次用到时自动生成并缓存。

`results/` 不入库，跑完手动拷回。

## 目标归一化与属性集

两个正交的配置维度，均由 `TrainConfig` 控制，并计入 `run_key`：

| `target_scaling` | 径流尺度来源 | 能主张什么 |
|---|---|---|
| `observed`（默认） | 逐流域实测均值/标准差 | 缺少连续时序时水位可替代——但留出流域仍间接知道自己的流量量级 |
| `physical` | `面积×降水` 的 log–log 回归，**只在仍有径流标签的流域上拟合** | 仅有属性与水位记录、**零流量观测**时水位监督依然有效 |

`physical` 模式下**水位仍用实测统计量**——留出情景的前提就是该流域有水位记录，
水位观测本来就是可得的。实测尺度精度：均值 R²=0.987/误差 12.5%，标准差
R²=0.953/误差 21.6%。

| `attr_set` | 列数 | 内容 |
|---|---|---|
| `base`（默认） | 15 | 项目既有的 12 个属性，与旧结果逐值一致 |
| `extended` | 27 | 追加高程、蒸散、水文土壤组、地下水位埋深、人口密度、城镇化率、坝密度等 |

**由实测径流导出的属性一律禁用**（`BFI_AVE`、`attributes_gageii_FlowRec.csv`
全部 114 列）——留出情景假设该流域无径流数据，用这些列等于把径流信息漏回去。
`attribute_sources.AttrSpec` 在构造时即拒绝。

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

原始数据路径在 `config.py` 的 `CAMELSH_DATA_PATH` 与 `MSWEP_CSV_PATH` 配置，
两者都优先读同名环境变量，换机器无需修改被跟踪的文件：

```powershell
$env:CAMELSH_DATA_PATH = "<包含 CAMELSH\ 的父目录>"
$env:MSWEP_CSV_PATH    = "<mswep_1000basins_mean_3hourly_1980_2024.csv 的路径>"
```

路径探测带 5 秒超时，指向未挂载的映射盘时会明确报错而不是静默挂死。
**这两个路径只在重建缓存时用到**，缓存建好后训练与评估都不再读取原始数据。

## 文档

- [时间聚合与标签对齐](docs/temporal_alignment.md) —— 逐变量区间定义与算例
- [双头多任务修订方案](docs/parallel_multitask_revision.md) —— 主线调整的由来

## 历史留档

`src/training/` 下的 `multi_task_lstm*.py`、`single_task_lstm*.py` 为旧实现，
其划分边界在 Dataset 构造时推断，会随标签、掩膜与窗口长度变化，产出的
数值不再用于论文。保留仅供回溯。
