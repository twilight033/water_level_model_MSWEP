# 按年份分批获取USGS Qualifiers - 使用指南

## 简介

`fetch_full_range.py` 脚本用于按年份分批获取USGS qualifiers数据，避免长时间范围导致的API超时问题。

## 为什么需要分批获取？

USGS NWIS API 对单次请求有时间限制。当请求时间跨度过大（如23年）时，会导致：
- 请求超时 (Read timeout)
- 连接中断
- 数据丢失

通过按年份分批请求，可以：
- [OK] 避免超时问题
- [OK] 支持断点续传（使用缓存）
- [OK] 更好的进度追踪
- [OK] 失败年份不影响其他年份

## 使用步骤

### 1. 确保已导出CAMELSH数据

```bash
uv run python qualifiers_fetcher/export_camelsh_data.py
```

这会生成：
- `camelsh_exported/flow_hourly.csv`
- `camelsh_exported/waterlevel_hourly.csv`

### 2. 修改配置（可选）

打开 `fetch_full_range.py`，根据需要修改：

```python
# 站点数量（默认使用前10个站点）
GAUGE_IDS = ALL_GAUGE_IDS[:10]  # 改为 [:50] 或 [:] 使用全部

# 时间范围（默认2001-2024）
START_YEAR = 2001
END_YEAR = 2024

# 输出目录
OUTPUT_DIR = "qualifiers_output_full"
CACHE_DIR = "qualifiers_cache_full"
```

### 3. 运行脚本

```bash
uv run python qualifiers_fetcher/fetch_full_range.py
```

程序会：
1. 显示配置信息
2. 显示预计耗时
3. 要求确认
4. 按年份逐年获取数据
5. 自动合并所有年份数据
6. 与CAMELSH数据合并
7. 生成最终结果

## 输出文件

### 主要输出

1. **合并数据文件**: `qualifiers_output_full/camelsh_with_qualifiers.csv`
   - 包含列：`datetime`, `gauge_id`, `Q`, `H`, `Q_flag`, `H_flag`, `Q_weight`, `H_weight`

2. **统计报告**: `qualifiers_output_full/qualifiers_report.txt`
   - 数据概览
   - Qualifiers分布统计
   - 权重统计
   - 数据完整性分析

### 缓存文件

- 位置：`qualifiers_cache_full/`
- 作用：存储已获取的原始数据
- 好处：如果中断或失败，重新运行时会跳过已缓存的年份

## 耗时估算

| 站点数 | 年份数 | 预计耗时 |
|--------|--------|----------|
| 2      | 24     | 约 4 分钟 |
| 10     | 24     | 约 20 分钟 |
| 50     | 24     | 约 100 分钟 (1.7小时) |
| 100    | 24     | 约 200 分钟 (3.3小时) |

*实际耗时取决于网络速度和API响应时间*

## 使用建议

### 首次使用

1. **先测试少量站点**
   ```python
   GAUGE_IDS = ALL_GAUGE_IDS[:2]  # 只用2个站点测试
   ```

2. **测试短时间范围**
   ```python
   START_YEAR = 2023
   END_YEAR = 2024  # 只测试2年
   ```

3. **检查输出结果**
   - 查看 `qualifiers_report.txt`
   - 检查覆盖率是否符合预期

### 大规模获取

1. **分批处理**
   ```python
   # 第一批: 2001-2010
   START_YEAR = 2001
   END_YEAR = 2010
   OUTPUT_DIR = "qualifiers_output_2001_2010"
   
   # 第二批: 2011-2020
   START_YEAR = 2011
   END_YEAR = 2020
   OUTPUT_DIR = "qualifiers_output_2011_2020"
   ```

2. **使用缓存**
   - 如果中途中断，直接重新运行
   - 脚本会自动使用缓存，跳过已获取的数据

3. **在非高峰时段运行**
   - USGS API在美国工作时间可能较慢
   - 建议在美国非工作时间运行（北京时间晚上）

## 常见问题

### Q1: 某些年份获取失败怎么办？

**A**: 不用担心，失败的年份不影响其他年份。重新运行脚本即可，已成功的年份会从缓存读取。

### Q2: 如何清除缓存重新获取？

**A**: 删除缓存目录：
```bash
rm -rf qualifiers_cache_full
```
或在Windows PowerShell:
```powershell
Remove-Item -Recurse -Force qualifiers_cache_full
```

### Q3: Qualifiers覆盖率低怎么办？

**A**: 这可能是正常情况：
- USGS可能没有为所有时间段提供qualifiers
- 某些站点可能缺少历史qualifiers数据
- 早期数据（2001-2005）的qualifiers可能较少

检查具体某个站点某个时间段是否有数据：
```python
# 在脚本中添加调试输出
print(results[gauge_id]['discharge'].head())
```

### Q4: 运行时间太长，可以中断吗？

**A**: 可以！按 `Ctrl+C` 中断。重新运行时：
- 已缓存的年份会自动跳过
- 从中断的地方继续
- 不会重复请求

### Q5: 如何验证数据质量？

**A**: 查看生成的报告文件 `qualifiers_report.txt`：
- 检查qualifiers分布（A, P, e等）
- 查看缺失率
- 对比不同站点的覆盖率

Qualifier含义：
- `A`: Approved（已审核，最高质量）
- `P`: Provisional（临时数据）
- `e`: Estimated（估计值）

## 示例运行输出

```
================================================================================
配置信息
================================================================================
站点数: 10
站点ID: ['01646500', '01434000', '01042500', '01055000', '01057000']...
时间范围: 2001 - 2024
输出目录: qualifiers_output_full
缓存目录: qualifiers_cache_full

================================================================================
重要提示
================================================================================
将要获取 10 个站点，24 年的数据
预计耗时: 约 20.0 分钟

确认开始获取? (y/n): y

================================================================================
开始按年份获取qualifiers数据
================================================================================
站点数: 10
年份范围: 2001 - 2024 (24年)
预计请求数: 240
预计耗时: 约 20.0 分钟
使用缓存: 是

年份进度:   0%|                                          | 0/24 [00:00<?, ?it/s]

正在获取 2001 年的数据 (2001-01-01 至 2001-12-31)...
  ✓ 2001 年完成 - 径流: 10/10 站点, 水位: 10/10 站点

年份进度:   4%|█▋                                | 1/24 [00:05<02:00,  5.24s/it]
...
```

## 进阶功能

### 自定义获取逻辑

如需更灵活的控制，可以直接使用 `USGSQualifiersFetcher` 类：

```python
from usgs_qualifiers_fetcher import USGSQualifiersFetcher

fetcher = USGSQualifiersFetcher(
    output_dir="my_output",
    cache_dir="my_cache"
)

# 只获取特定月份
results = fetcher.fetch_multiple_gauges(
    gauge_ids=["01646500"],
    start_date="2023-06-01",
    end_date="2023-06-30",
    use_cache=True
)
```

### 过滤特定qualifiers

获取数据后，可以过滤出特定质量的数据：

```python
# 只保留 'A' (Approved) 质量的数据
df = pd.read_csv('qualifiers_output_full/camelsh_with_qualifiers.csv')
high_quality = df[df['Q_flag'] == 'A']
```

## 技术说明

- API: USGS NWIS Instantaneous Values (iv)
- 请求间隔: 0.5秒（避免限流）
- 超时设置: 30秒
- 数据格式: CSV（逗号分隔）
- 时区: UTC

## 相关文件

- `usgs_qualifiers_fetcher.py`: 核心功能实现
- `export_camelsh_data.py`: 导出CAMELSH数据
- `test_quick_run.py`: 快速测试脚本（1个月数据）
- `config.py`: 配置文件
- `README.md`: 完整文档

## 支持

如有问题，请检查：
1. 网络连接是否正常
2. USGS API是否在线：https://waterservices.usgs.gov/
3. 站点ID是否正确
4. 时间范围是否合理

