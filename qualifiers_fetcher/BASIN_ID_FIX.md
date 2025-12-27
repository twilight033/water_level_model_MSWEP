# Basin ID 前导零修复说明

**问题**: 2025-12-27 17:xx

## 问题描述

运行 `fetch_per_basin_timeranges.py` 时，所有USGS API请求都返回 400 错误：

```
警告: 请求失败 - 1017000: 400 Client Error
URL: ...&sites=1017000&...
```

## 根本原因

**Pandas读取CSV时丢失了basin_id的前导零**

CSV文件中：`01017000` (8位，正确) ✓  
Pandas读取后：`1017000` (7位，错误) ✗  
USGS API需要：`01017000` (8位) ✓

## 修复方案

### 1. 读取CSV时指定数据类型

```python
# 修复前
df = pd.read_csv(csv_file)

# 修复后
df = pd.read_csv(csv_file, dtype={'basin_id': str})
```

### 2. 使用时额外保护

```python
# 修复前
basin_id = str(row['basin_id'])

# 修复后
basin_id = str(row['basin_id']).zfill(8)  # 确保8位，前导零补齐
```

## 修复的文件

`qualifiers_fetcher/fetch_per_basin_timeranges.py`:
- Line ~28: 添加 `dtype={'basin_id': str}` 参数
- Line ~89: 添加 `.zfill(8)` 确保8位

## 验证

修复后测试：

```bash
# 读取CSV验证
uv run python -c "import pandas as pd; df = pd.read_csv('runtime_basin_time_ranges_simple.csv', dtype={'basin_id': str}); print(df['basin_id'].head().tolist())"

# 输出应该是
['01017000', '01017060', '01017290', '01017960', '01018000']
# ✓ 所有ID都是8位，保留前导零
```

## 现在可以重新运行

```bash
uv run python qualifiers_fetcher/fetch_per_basin_timeranges.py
```

API请求将使用正确的8位站点ID：
```
sites=01017000  ✓ (修复后)
而不是 sites=1017000  ✗ (修复前)
```

## 相关说明

这是一个常见的Pandas问题：
- CSV中 `01017000` 会被pandas自动转换为数字 `1017000`
- 解决方法：在 `read_csv()` 中指定 `dtype={'basin_id': str}`
- 或者使用 `.zfill(8)` 在使用时补齐前导零

---

**状态**: ✅ 已修复  
**测试**: ✅ 通过  
**可以继续**: ✅ 是

