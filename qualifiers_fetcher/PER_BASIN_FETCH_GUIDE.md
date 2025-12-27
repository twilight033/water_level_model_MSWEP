# 按每个流域独立时间范围获取Qualifiers - 使用指南

## 概述

`fetch_per_basin_timeranges.py` 脚本会为每个流域使用其独立的训练/验证/测试时间范围来获取USGS qualifiers数据，确保qualifiers数据与模型训练时使用的数据完全对应。

## 为什么需要这个脚本？

由于 `multi_task_lstm.py` 为每个流域独立划分时间（60%/20%/20%），不同流域的训练/测试时间段不同：

```
流域 01017000: 训练 2007-2018, 测试 2021-2024
流域 01018000: 训练 2007-2012, 测试 2014-2015
流域 01029200: 训练 2019-2022, 测试 2023-2024
```

因此需要为每个流域分别获取其对应时间段的qualifiers数据。

## 使用方法

### 前提条件

1. **已生成流域时间配置文件**
   ```bash
   uv run python extract_per_basin_config.py
   ```
   这会生成 `runtime_basin_time_ranges_simple.csv`

2. **已导出CAMELSH数据**
   ```bash
   uv run python qualifiers_fetcher/export_camelsh_data.py
   ```

### 基本使用

```bash
uv run python qualifiers_fetcher/fetch_per_basin_timeranges.py
```

脚本会：
1. 从 `runtime_basin_time_ranges_simple.csv` 读取每个流域的时间范围
2. 为每个流域使用其训练期时间范围获取qualifiers
3. 自动处理长时间跨度（按年份分批）
4. 合并到CAMELSH数据
5. 生成带qualifiers的完整数据集

### 配置选项

在脚本中可以修改以下参数：

```python
# 选择时间段
TIME_PERIOD = "train"  # 'train', 'valid', 'test', 'all'

# 测试模式：只处理前N个流域
MAX_BASINS = 5  # 设为None处理全部

# 输出目录
OUTPUT_DIR = f"qualifiers_output_per_basin_{TIME_PERIOD}"
CACHE_DIR = f"qualifiers_cache_per_basin_{TIME_PERIOD}"
```

## 时间段选项

### 1. `TIME_PERIOD = "train"` (推荐)
- 为每个流域获取其训练期的qualifiers
- 适合：分析训练数据质量

### 2. `TIME_PERIOD = "test"`
- 为每个流域获取其测试期的qualifiers
- 适合：分析测试数据质量

### 3. `TIME_PERIOD = "valid"`
- 为每个流域获取其验证期的qualifiers
- 适合：分析验证数据质量

### 4. `TIME_PERIOD = "all"`
- 获取从训练开始到测试结束的完整范围
- 适合：获取完整数据集的qualifiers

## 输出文件

### 1. 合并数据文件
```
qualifiers_output_per_basin_train/camelsh_with_qualifiers.csv
```

**列**：
- `datetime`: 时间戳
- `gauge_id`: 流域ID
- `Q`: 径流值
- `H`: 水位值
- `Q_flag`: 径流数据质量标记
- `H_flag`: 水位数据质量标记
- `Q_weight`: 径流数据权重
- `H_weight`: 水位数据权重

### 2. 统计报告
```
qualifiers_output_per_basin_train/qualifiers_report.txt
```

包含：
- 数据概览
- Qualifiers分布统计
- 权重统计
- 数据完整性

### 3. 缓存文件
```
qualifiers_cache_per_basin_train/
```

存储已获取的原始qualifiers数据，支持断点续传。

## 使用示例

### 示例1：获取训练期qualifiers（推荐）

```python
# 在脚本中设置
TIME_PERIOD = "train"
MAX_BASINS = None  # 处理全部流域

# 运行
# uv run python qualifiers_fetcher/fetch_per_basin_timeranges.py
```

**预期结果**：
- 为100个流域各自获取其训练期的qualifiers
- 耗时：约30-60分钟（取决于网络速度）
- 输出：包含训练期qualifiers的完整数据集

### 示例2：快速测试（5个流域）

```python
# 在脚本中设置
TIME_PERIOD = "train"
MAX_BASINS = 5  # 只处理前5个

# 运行
# uv run python qualifiers_fetcher/fetch_per_basin_timeranges.py
```

**预期结果**：
- 快速测试功能
- 耗时：约2-5分钟

### 示例3：获取完整时间范围

```python
# 在脚本中设置
TIME_PERIOD = "all"
MAX_BASINS = None

# 运行
# uv run python qualifiers_fetcher/fetch_per_basin_timeranges.py
```

**预期结果**：
- 获取每个流域从训练开始到测试结束的完整qualifiers
- 耗时：约60-120分钟

## 自动处理长时间跨度

脚本会自动检测时间跨度：
- **< 1年**：直接一次性获取
- **≥ 1年**：自动按年份分批获取，避免API超时

例如：
```
流域 01017000 (训练期: 2007-2018, 11年)
  → 自动分为: 2007, 2008, ..., 2018 (11次请求)

流域 01029200 (训练期: 2019-2022, 3年)
  → 自动分为: 2019, 2020, 2021, 2022 (4次请求)
```

## 缓存和断点续传

脚本使用缓存机制：

1. **首次运行**：从USGS API获取数据并缓存
2. **重复运行**：直接从缓存读取（秒级完成）
3. **中断恢复**：已获取的数据不会重复请求

**清除缓存**（如需重新获取）：
```bash
# Windows PowerShell
Remove-Item -Recurse -Force qualifiers_cache_per_basin_train

# Linux/Mac
rm -rf qualifiers_cache_per_basin_train
```

## 预期覆盖率

由于每个流域只获取其对应时期的qualifiers：

### 训练期 (TIME_PERIOD = "train")
```
流域 01017000 (2007-2018):
  CAMELSH总数据: 2007-2024 (17年)
  获取qualifiers: 2007-2018 (11年)
  覆盖率: 11/17 ≈ 65%

流域 01018000 (2007-2012):
  CAMELSH总数据: 2007-2015 (8年)
  获取qualifiers: 2007-2012 (5年)
  覆盖率: 5/8 ≈ 63%
```

这是**正常且正确的**，因为：
- 只获取训练期数据的qualifiers
- 验证期和测试期的数据没有qualifiers（因为没有获取）
- 如需全覆盖，使用 `TIME_PERIOD = "all"`

## 在模型训练中使用

获取到qualifiers数据后，可以在训练时使用：

```python
# 读取带qualifiers的数据
merged_data = pd.read_csv(
    'qualifiers_output_per_basin_train/camelsh_with_qualifiers.csv',
    index_col=0,
    parse_dates=True
)

# 方案1：只使用高质量数据
high_quality_mask = merged_data['Q_flag'] == 'A'
training_data = merged_data[high_quality_mask]

# 方案2：使用权重训练
# 在损失函数中使用 Q_weight 和 H_weight
weights = torch.tensor(merged_data['Q_weight'].values)
loss = (predictions - targets) ** 2 * weights

# 方案3：过滤特定条件的数据
# 例如：排除估计值
mask = ~merged_data['Q_flag'].str.contains('e', na=False)
filtered_data = merged_data[mask]
```

## 常见问题

### Q1: 为什么某些流域的覆盖率很低？

**A**: 可能原因：
1. 该流域的训练期较短（如只有3-5年）
2. USGS没有为该时期提供qualifiers
3. 该流域的数据记录较新，历史qualifiers较少

### Q2: 可以同时获取多个时间段吗？

**A**: 需要多次运行脚本：
```bash
# 第一次：训练期
# 修改 TIME_PERIOD = "train"
uv run python qualifiers_fetcher/fetch_per_basin_timeranges.py

# 第二次：测试期
# 修改 TIME_PERIOD = "test"
uv run python qualifiers_fetcher/fetch_per_basin_timeranges.py
```

### Q3: 脚本运行很慢，如何加速？

**A**: 
1. 使用缓存：重复运行会很快
2. 减少流域数量：设置 `MAX_BASINS = 10` 进行测试
3. 网络问题：在网络稳定时运行

### Q4: 某个流域获取失败怎么办？

**A**: 脚本会继续处理其他流域。失败原因可能：
1. 该流域在USGS系统中不存在
2. 网络临时中断
3. API限流

重新运行脚本即可，已成功的流域会从缓存读取。

## 与其他脚本的对比

| 脚本 | 时间范围 | 适用场景 |
|------|----------|----------|
| `test_quick_run.py` | 固定1个月 | 快速测试功能 |
| `test_batch_fetch.py` | 固定2年 | 测试分批逻辑 |
| `fetch_full_range.py` | 固定2001-2024 | 所有流域统一时间 |
| **`fetch_per_basin_timeranges.py`** | **每个流域独立** | **匹配训练数据** ✓ |

## 最佳实践

1. **首次使用**：先用 `MAX_BASINS = 5` 快速测试
2. **正式运行**：设置 `MAX_BASINS = None` 处理全部
3. **按需获取**：根据研究需要选择 `TIME_PERIOD`
4. **保留缓存**：不要频繁删除缓存目录
5. **定期运行**：USGS数据会更新，可定期重新获取

## 相关文件

- `runtime_basin_time_ranges_simple.csv`: 输入（流域时间配置）
- `extract_per_basin_config.py`: 生成时间配置
- `usgs_qualifiers_fetcher.py`: 核心Fetcher类
- `export_camelsh_data.py`: 导出CAMELSH数据

---

**版本**: 1.0  
**最后更新**: 2025-12-27  
**作者**: 根据multi_task_lstm.py的实际时间划分逻辑设计

