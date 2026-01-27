# 提取实际流域ID列表 - 完成总结

**完成时间**: 2026-01-17 14:45

## 任务概述

你要求提取 `multi_task_lstm.py` 实际运行时使用的流域ID列表。

## 数据来源

流域ID列表来自 **`runtime_basins.csv`** 文件，这个文件是之前运行 `extract_per_basin_config.py` 时生成的，包含了实际用于训练的100个流域。

## 生成的文件

### 1. `actual_basin_ids.txt`
**格式**: Python列表  
**用途**: 可以直接复制到Python代码中使用

```python
ACTUAL_BASIN_IDS = [
    '01017000',
    '01017060',
    '01017290',
    # ... 共100个
    '01123000'
]
```

### 2. `actual_basin_ids_simple.txt`
**格式**: 纯文本，每行一个ID  
**用途**: 便于其他程序读取

```
01017000
01017060
01017290
...
01123000
```

### 3. `actual_basin_ids_report.txt`
**格式**: 详细报告  
**用途**: 查看完整列表和统计信息

包含:
- 生成时间
- 数据来源
- 流域数量
- 完整的流域ID列表（1-100编号）

### 4. `basin_ids_for_qualifiers.txt`
**格式**: 专用于 qualifiers_fetcher  
**用途**: 在 qualifiers_fetcher 脚本中直接使用

```python
# 使用方法示例:
with open('basin_ids_for_qualifiers.txt') as f:
    basin_ids = [line.strip() for line in f if not line.startswith('#')]
```

## 流域统计

- **总数量**: 100个流域
- **ID格式**: 8位数字（带前导零）
- **范围**: 01017000 至 01123000

### 前10个流域:
```
01017000, 01017060, 01017290, 01017960, 01018000,
01018009, 01019000, 01021200, 01027200, 01029200
```

### 后10个流域:
```
01115770, 01117370, 01117468, 01117800, 01118300,
01119500, 01121000, 01122000, 01122500, 01123000
```

## 关键信息

### 这些流域ID是如何确定的？

根据 `multi_task_lstm.py` 的逻辑：

1. **候选流域来源**: `valid_waterlevel_basins.txt` 文件
2. **验证标准**: 
   - 同时有有效的径流和水位数据
   - 有效数据比例 ≥ 10%
   - 数据不全为NaN
3. **选择数量**: config.py 中 `NUM_BASINS = 100`
4. **选择方式**: 从验证通过的流域中选择前100个

### 与训练时间范围的关系

这100个流域各自有独立的训练/验证/测试时间范围，详见:
- `runtime_basin_time_ranges_simple.csv` - 简要信息
- `runtime_basin_time_ranges_report.txt` - 详细信息

每个流域按其自身的有效数据时间范围独立划分为:
- 训练期: 60%
- 验证期: 20%
- 测试期: 20%

## 使用建议

### 在 qualifiers_fetcher 中使用

如果要为这100个流域获取qualifiers数据：

```bash
# 方式1: 使用 basin_ids_for_qualifiers.txt
# 在脚本中读取这个文件

# 方式2: 直接使用 runtime_basins.csv
# 已经包含在 fetch_per_basin_timeranges.py 中
uv run python qualifiers_fetcher/fetch_per_basin_timeranges.py
```

`fetch_per_basin_timeranges.py` 已经配置为从 `runtime_basin_time_ranges_simple.csv` 读取流域ID和时间范围，会自动使用这100个流域。

### 在其他脚本中使用

```python
# 方式1: 直接复制列表
ACTUAL_BASIN_IDS = ['01017000', '01017060', ...]

# 方式2: 从文件读取
import pandas as pd
df = pd.read_csv('runtime_basins.csv', dtype={'basin_id': str})
basin_ids = df['basin_id'].tolist()

# 方式3: 读取纯文本文件
with open('actual_basin_ids_simple.txt') as f:
    basin_ids = [line.strip() for line in f if line.strip()]
```

## 与其他文件的关系

| 文件 | 内容 | 关系 |
|------|------|------|
| `runtime_basins.csv` | 100个流域ID | **源文件** |
| `runtime_basin_time_ranges_simple.csv` | 每个流域的时间范围 | 配套时间信息 |
| `runtime_basin_time_ranges_report.txt` | 详细时间范围报告 | 详细说明 |
| `actual_basin_ids.txt` | Python格式ID列表 | **本次生成** |
| `actual_basin_ids_simple.txt` | 纯文本ID列表 | **本次生成** |
| `actual_basin_ids_report.txt` | 详细报告 | **本次生成** |
| `basin_ids_for_qualifiers.txt` | qualifiers专用格式 | **本次生成** |

## 验证

可以通过以下方式验证这些流域ID：

```python
# 检查数量
assert len(basin_ids) == 100

# 检查格式（8位数字）
assert all(len(bid) == 8 for bid in basin_ids)
assert all(bid.isdigit() for bid in basin_ids)

# 检查是否有重复
assert len(basin_ids) == len(set(basin_ids))

# 检查是否与 runtime_basins.csv 一致
df = pd.read_csv('runtime_basins.csv', dtype={'basin_id': str})
assert basin_ids == df['basin_id'].tolist()
```

---

**状态**: ✅ 完成  
**输出文件数**: 4个  
**流域总数**: 100个  

现在你可以使用这些文件中的流域ID列表了！
