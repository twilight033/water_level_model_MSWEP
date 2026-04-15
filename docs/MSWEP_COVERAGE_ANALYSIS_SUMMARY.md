# MSWEP数据覆盖情况分析总结

## 问题描述

运行 `multi_task_lstm_ablation_missing_labels.py` 时，尽管请求了500个流域，但只使用了324个流域进行训练。

## 原因分析

### 数据流程

```
1. 候选流域列表 (valid_waterlevel_basins.txt)
   └─> 共5767个候选流域

2. 数据验证 (filter_basins_with_valid_data)
   └─> 验证径流和水位数据有效性
   └─> 通过验证的流域: 500个 ✓

3. MSWEP数据加载 (load_mswep_data)
   └─> MSWEP文件包含: 1000个流域
   └─> 与500个选择流域对比...
   └─> 匹配: 324个 ✓
   └─> 缺失: 176个 ✗
```

### 核心问题

**经过验证的500个流域中，有176个在MSWEP数据文件中不存在！**

```
实际选择的流域: 500个
MSWEP匹配的流域: 324个
MSWEP缺失的流域: 176个
匹配率: 64.8%
```

## 缺失流域分析

### 缺失的176个流域

完整列表见: `actual_missing_in_mswep.txt`

缺失流域主要分布在:
- `02xxxxx` 区域: 约70个流域 (主要是东南部流域)
- `03xxxxx` 区域: 约106个流域 (主要是俄亥俄河流域)

示例缺失流域:
```
02167450, 02197300, 02228500, 02231000, 02231342, ...
03010655, 03011800, 03015500, 03017500, 03021350, ...
```

### MSWEP文件包含的流域范围

当前MSWEP文件 (`mswep_1000basins_mean_3hourly_1980_2024.csv`) 包含:
- 总流域数: 1000个
- 主要覆盖: `01xxxxx` 区域（东北部和大西洋中部）
- 部分覆盖: `02xxxxx` 和 `03xxxxx` 区域

## 解决方案

### 方案1: 扩充MSWEP数据 (推荐用于完整研究)

**优点**: 可以使用全部500个验证通过的流域  
**缺点**: 需要额外下载和处理MSWEP数据

**操作步骤**:
1. 获取缺失流域列表:
   ```bash
   # 查看缺失的176个流域
   cat actual_missing_in_mswep.txt
   ```

2. 下载MSWEP原始数据并提取这176个流域的降雨数据

3. 合并到现有MSWEP文件中

4. 重新运行训练代码

### 方案2: 使用现有324个流域 (推荐用于快速测试)

**优点**: 无需额外数据，立即可用  
**缺点**: 流域数量减少，可能影响模型泛化性

**操作步骤**:
1. 修改 `config.py`:
   ```python
   NUM_BASINS = 324  # 从500改为324
   ```

2. 重新运行训练

3. 验证效果

### 方案3: 流域预筛选 (推荐用于最优配置)

**优点**: 确保选择的流域都有MSWEP数据  
**缺点**: 需要修改代码逻辑

**操作步骤**:
1. 修改 `multi_task_lstm_ablation_missing_labels.py`，在流域验证时增加MSWEP检查:

```python
def filter_basins_with_mswep(basins, mswep_file):
    """筛选MSWEP数据中存在的流域"""
    df = pd.read_csv(mswep_file, nrows=0)
    mswep_basins = df.columns.tolist()[1:]
    return [b for b in basins if b in mswep_basins]

# 在数据加载前添加
print("\n正在筛选MSWEP数据中存在的流域...")
mswep_file = "MSWEP/mswep_1000basins_mean_3hourly_1980_2024.csv"
validated_basins_with_mswep = filter_basins_with_mswep(
    validated_basins, 
    mswep_file
)
print(f"  MSWEP覆盖的流域数量: {len(validated_basins_with_mswep)}")

# 从MSWEP覆盖的流域中选择
chosen_basins = validated_basins_with_mswep[:NUM_BASINS]
```

2. 这样可以保证选择的流域都有MSWEP数据

## 性能影响评估

### 使用324个流域 vs 500个流域

| 指标 | 324个流域 | 500个流域 |
|------|-----------|-----------|
| 训练样本量 | 减少约35% | 完整 |
| 地理覆盖 | 主要集中在东北部 | 更广泛 |
| 模型泛化性 | 可能降低 | 更好 |
| 训练时间 | 更快 (约-35%) | 较慢 |
| 内存占用 | 更少 (约-35%) | 较多 |

### 推荐选择

- **研究发表**: 方案1 (扩充MSWEP数据)
- **快速实验**: 方案2 (使用324个流域)
- **最佳实践**: 方案3 (流域预筛选)

## 下一步操作

### 立即可行 (方案2)

```bash
# 1. 修改config.py
NUM_BASINS = 324

# 2. 重新运行
uv run python multi_task_lstm_ablation_missing_labels.py
```

### 最优方案 (方案3)

```bash
# 1. 使用提供的代码片段修改multi_task_lstm_ablation_missing_labels.py

# 2. 运行测试
uv run python multi_task_lstm_ablation_missing_labels.py

# 3. 如果需要更多流域，调整NUM_BASINS
# 最多可用: len(validated_basins_with_mswep)
```

### 完整方案 (方案1)

```bash
# 1. 提取缺失流域的MSWEP数据 (需要MSWEP原始数据)

# 2. 合并到现有文件

# 3. 使用全部500个流域训练
```

## 文件清单

本次分析生成的文件:

1. **check_mswep_coverage.py** - MSWEP覆盖检查工具
2. **mswep_coverage_report.txt** - 候选列表前500个的覆盖报告
3. **mswep_coverage_matched_basins.txt** - 匹配的流域列表
4. **mswep_coverage_analysis.csv** - CSV格式分析结果
5. **compare_actual_vs_mswep.py** - 实际选择流域的对比工具
6. **actual_missing_in_mswep.txt** - 实际缺失的176个流域列表
7. **MSWEP_COVERAGE_ANALYSIS_SUMMARY.md** - 本报告

## 关键发现

1. **MSWEP文件包含1000个流域**，但不是均匀分布，主要集中在某些区域

2. **验证后的500个流域**不是候选列表的前500个，而是经过数据完整性验证的

3. **176个流域在MSWEP中缺失**，导致只能使用324个流域

4. **建议**:
   - 短期: 使用324个流域快速测试
   - 长期: 扩充MSWEP数据或优化流域选择逻辑

---

**生成时间**: 2026-01-21  
**分析工具**: `check_mswep_coverage.py`, `compare_actual_vs_mswep.py`
