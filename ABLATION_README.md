# 多任务LSTM消融实验 - 非对称标签缺失研究

## 快速开始

### 运行完整实验

```bash
uv run python multi_task_lstm_ablation_missing_labels.py
```

这将运行9个实验，测试**非对称标签缺失**的情况（一个任务缺失，另一个完整）。

### 预期输出

1. **控制台输出**：每个实验的训练进度和关键发现
2. **模型文件**：`results/models/multi_task_ablation_*.pth`（9个模型）
3. **结果图表**：`results/images/ablation_asymmetric_missing_comparison.png`（4个子图）
4. **结果数据**：`results/reports/ablation_asymmetric_missing_results.csv`

## 实验说明

### 核心问题
**当一个任务的标签缺失而另一个任务标签完整时，多任务学习模型的表现如何？**

### 关键创新
- ✅ **非对称缺失**：测试单个任务标签缺失的影响
- ✅ **跨任务影响分析**：研究一个任务缺失对另一个任务的影响
- ✅ **任务互补性**：验证多任务学习的鲁棒性优势
- ✅ **完整可视化**：4张图展示不同维度的结果

### 实验场景（共9个）

| 类型 | 径流标签 | 水位标签 | 场景描述 |
|-----|---------|---------|---------|
| 基线 | 完整(0%) | 完整(0%) | 两个任务都有完整标签 |
| **场景1** | 缺失10% | 完整(0%) | 研究径流缺失对两个任务的影响 |
| **场景1** | 缺失30% | 完整(0%) | 中度径流缺失 |
| **场景1** | 缺失50% | 完整(0%) | 重度径流缺失 |
| **场景2** | 完整(0%) | 缺失10% | 研究水位缺失对两个任务的影响 |
| **场景2** | 完整(0%) | 缺失30% | 中度水位缺失 |
| **场景2** | 完整(0%) | 缺失50% | 重度水位缺失 |
| **场景3** | 缺失30% | 缺失30% | 对比：两个任务同时缺失 |
| **场景3** | 缺失50% | 缺失50% | 对比：两个任务同时缺失 |

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

### 修改实验场景

在 `multi_task_lstm_ablation_missing_labels.py` 的主程序部分找到 `experiments` 列表：

```python
# 完整实验（9个场景）
experiments = [
    (0.0, 0.0, "baseline_both_complete"),
    (0.1, 0.0, "flow_missing_10pct_wl_complete"),
    (0.3, 0.0, "flow_missing_30pct_wl_complete"),
    (0.5, 0.0, "flow_missing_50pct_wl_complete"),
    (0.0, 0.1, "flow_complete_wl_missing_10pct"),
    (0.0, 0.3, "flow_complete_wl_missing_30pct"),
    (0.0, 0.5, "flow_complete_wl_missing_50pct"),
    (0.3, 0.3, "both_missing_30pct"),
    (0.5, 0.5, "both_missing_50pct"),
]

# 快速测试（只测试基线和30%缺失）
experiments = [
    (0.0, 0.0, "baseline"),
    (0.3, 0.0, "flow_missing_30pct"),
    (0.0, 0.3, "wl_missing_30pct"),
]
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
# CSV格式（包含9个实验的详细结果）
cat results/reports/ablation_asymmetric_missing_results.csv
```

### 查看可视化结果

打开 `results/images/ablation_asymmetric_missing_comparison.png`

**包含4个子图：**

1. **左上图**：径流任务性能 vs 标签缺失
   - 蓝线：径流缺失+水位完整
   - 橙线：径流完整+水位缺失
   - 虚线：基线性能

2. **右上图**：水位任务性能 vs 标签缺失
   - 蓝线：径流缺失+水位完整
   - 橙线：径流完整+水位缺失
   - 虚线：基线性能

3. **左下图**：跨任务影响 - 水位缺失对径流任务的影响
   - 柱状图显示径流NSE下降百分比

4. **右下图**：跨任务影响 - 径流缺失对水位任务的影响
   - 柱状图显示水位NSE下降百分比

### 关键分析问题

#### 1. 跨任务鲁棒性验证
**问题**：一个任务缺失，另一个任务受影响大吗？

**分析方法**：
```
查看左下和右下图：
- 如果柱状图接近0 → 跨任务影响小，模型鲁棒
- 如果柱状图较高 → 跨任务依赖强
```

#### 2. 任务依赖性分析
**问题**：哪个任务更依赖另一个任务？

**分析方法**：
```
对比两个跨任务影响图：
- 径流任务受水位缺失的影响 vs 水位任务受径流缺失的影响
- 哪个更大？说明哪个任务更依赖另一个
```

#### 3. 自身标签重要性
**问题**：任务主要靠自身标签还是另一个任务的标签？

**分析方法**：
```
对比上面两图中的蓝线和橙线：
- 蓝线大幅下降 → 任务主要依赖自身标签
- 橙线轻微下降 → 任务对另一个标签不敏感
```

#### 4. 多任务学习优势
**问题**：多任务比单任务更鲁棒吗？

**对比实验**：
```
运行单任务模型，对比相同缺失比例下的性能
预期：多任务模型在部分标签缺失时表现更好
```

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

### 1. 更多缺失比例

添加更细粒度的缺失比例（如5%, 15%, 40%）：

```python
experiments = [
    (0.0, 0.0, "baseline"),
    (0.05, 0.0, "flow_missing_5pct"),
    (0.15, 0.0, "flow_missing_15pct"),
    # ...更多场景
]
```

### 2. 极端非对称

测试更极端的非对称场景：

```python
# 一个任务完全缺失，另一个完整
experiments = [
    (0.9, 0.0, "flow_missing_90pct_wl_complete"),
    (0.0, 0.9, "flow_complete_wl_missing_90pct"),
]
```

### 3. 结构化缺失

实现更真实的缺失模式：

```python
def create_seasonal_missing(data_df, seasons_to_mask):
    """隐藏特定季节的数据"""
    # 例如：隐藏夏季径流数据（洪水期）
    pass

def create_continuous_missing(data_df, gap_length, n_gaps):
    """创建连续时间段的缺失"""
    # 例如：随机3段各30天的数据缺失
    pass
```

### 4. 对比单任务模型（重要）

运行单任务模型进行对比：

```bash
# 修改single_task_lstm.py支持标签缺失
# 然后对比：
# - 单任务模型（径流，缺失30%）
# - 多任务模型（径流缺失30%，水位完整）
```

**预期发现**：多任务模型应该表现更好（因为有水位任务辅助）

### 5. 不同缺失种子

测试随机性的影响：

```python
# 使用不同随机种子重复实验
for seed in [42, 123, 456]:
    flow_masked = create_missing_mask(flow_data, 0.3, seed=seed)
    # ...
```

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


