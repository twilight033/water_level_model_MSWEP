# 多任务LSTM消融实验 - 标签缺失研究

## 快速开始

### 运行完整实验

```bash
uv run python multi_task_lstm_ablation_missing_labels.py
```

这将运行4个实验，测试标签缺失比例为 0%, 10%, 30%, 50% 的情况。

### 预期输出

1. **控制台输出**：每个实验的训练进度和结果
2. **模型文件**：`results/models/multi_task_ablation_missing_*.pth`
3. **结果图表**：`results/images/ablation_missing_labels_comparison.png`
4. **结果数据**：`results/reports/ablation_missing_labels_results.csv`

## 实验说明

### 目的
研究在训练数据标签（径流/水位观测值）随机缺失时，多任务学习模型的性能表现。

### 关键特性
- ✅ 支持任意缺失比例
- ✅ 径流和水位独立缺失
- ✅ 使用mask机制处理缺失标签
- ✅ 测试集保持完整（公平评估）
- ✅ 自动生成对比报告

### 实验场景

| 缺失比例 | 场景描述 | 预期影响 |
|---------|---------|---------|
| 0%      | 基线（无缺失） | 最佳性能 |
| 10%     | 轻度缺失 | 轻微性能下降 |
| 30%     | 中度缺失 | 中等性能下降 |
| 50%     | 重度缺失 | 显著性能下降 |

## 文件说明

### 核心文件

- **`multi_task_lstm_ablation_missing_labels.py`**  
  主程序，包含完整的消融实验框架

- **`multi_task_lstm - 早停.py`**  
  基础版本（无标签缺失）

### 文档文件

- **`results/reports/消融实验_标签缺失_说明.md`**  
  详细的实验设计和技术说明

- **`results/reports/消融实验_开发总结.md`**  
  开发总结和技术细节

- **`ABLATION_README.md`** (本文件)  
  快速参考指南

### 测试文件

- **`test_ablation_quick.py`**  
  快速测试脚本（用于验证代码）

## 调整实验参数

### 修改缺失比例

在 `multi_task_lstm_ablation_missing_labels.py` 的末尾：

```python
# 修改这行
missing_ratios = [0.0, 0.1, 0.3, 0.5]  # 可以添加或删除比例

# 例如：只测试0%和50%
missing_ratios = [0.0, 0.5]
```

### 修改模型参数

在 `config.py` 中调整：

```python
NUM_BASINS = 10      # 使用的流域数量
EPOCHS = 10          # 训练轮数
BATCH_SIZE = 64      # 批次大小
SEQUENCE_LENGTH = 168  # 序列长度
```

### 快速测试

如果只想快速验证代码是否能运行：

1. 修改 `config.py`：
   ```python
   NUM_BASINS = 5
   EPOCHS = 3
   ```

2. 修改实验场景：
   ```python
   missing_ratios = [0.0, 0.3]  # 只测试两个
   ```

3. 运行：
   ```bash
   uv run python multi_task_lstm_ablation_missing_labels.py
   ```

## 结果分析

### 查看数值结果

```bash
# CSV格式
cat results/reports/ablation_missing_labels_results.csv
```

### 查看可视化结果

打开 `results/images/ablation_missing_labels_comparison.png`

图表包含：
- **左图**：不同缺失比例下的测试NSE
- **右图**：相对基线的性能下降百分比

### 典型分析问题

1. **模型鲁棒性**  
   NSE下降是否平滑？还是在某个缺失比例处突然崩溃？

2. **任务差异**  
   径流和水位哪个更容易受缺失影响？

3. **多任务优势**  
   与单任务模型相比，多任务模型在缺失场景下是否更鲁棒？

## 常见问题

### Q1: 实验需要多长时间？

取决于配置：
- 5个流域，3个epoch，2个场景：约10-15分钟（GPU）
- 50个流域，10个epoch，4个场景：约2-4小时（GPU）

### Q2: 如何确保实验可重复？

代码使用固定随机种子：
- 总体随机种子：1234
- 径流缺失mask：42
- 水位缺失mask：43

### Q3: 测试集为什么不缺失？

为了公平评估不同缺失比例下训练的模型，测试集必须保持一致（完整）。

### Q4: 可以测试非对称缺失吗？

可以！修改 `run_ablation_experiment` 函数，为径流和水位使用不同的缺失比例。

### Q5: 如何与单任务模型对比？

可以运行 `single_task_lstm.py` 使用相同的缺失数据，然后对比结果。

## 扩展实验

### 1. 非对称缺失

修改 `run_ablation_experiment` 函数：

```python
# 径流缺失50%，水位缺失10%
flow_masked = create_missing_mask(flow_data, 0.5, seed=42)
wl_masked = create_missing_mask(wl_data, 0.1, seed=43)
```

### 2. 结构化缺失

实现季节性或连续时间段的缺失，更接近真实场景。

### 3. 对比单任务

在相同缺失比例下训练单任务模型，对比多任务的优势。

## 技术支持

如有问题，请查看：
1. **详细说明**: `results/reports/消融实验_标签缺失_说明.md`
2. **开发文档**: `results/reports/消融实验_开发总结.md`
3. **基础代码**: `multi_task_lstm - 早停.py`

## 版本信息

- **版本**: 1.0
- **创建日期**: 2025-12-22
- **基于**: `multi_task_lstm - 早停.py`
- **Python**: 3.8+
- **主要依赖**: PyTorch, xarray, pandas, HydroErr

---

**提示**: 首次运行建议使用较小的参数（5个流域，3个epoch）快速验证代码，然后再运行完整实验。

