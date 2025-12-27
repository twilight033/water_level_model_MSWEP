# USGS Qualifiers Fetcher

为本地CAMELSH数据集添加USGS数据质量标签（qualifiers）。

## 功能特性

- ✅ 使用USGS NWIS Instantaneous Values API (iv)
- ✅ 批量查询多个站点的径流(00060)和水位(00065)数据
- ✅ 提取每条观测的qualifiers（数据质量标记）
- ✅ 自动处理时区，与CAMELSH数据对齐
- ✅ 智能缓存，避免重复请求API
- ✅ 生成包含Q, H, Q_flag, H_flag的合并数据
- ✅ 可选：基于qualifiers计算数据权重

## 文件说明

```
qualifiers_fetcher/
├── usgs_qualifiers_fetcher.py  # 主程序
├── config.py                    # 配置文件
├── export_camelsh_data.py      # CAMELSH数据导出工具
├── README.md                    # 本文件
├── requirements.txt             # 依赖包
├── qualifiers_cache/           # API响应缓存（自动创建）
└── qualifiers_output/          # 输出文件（自动创建）
```

## 快速开始

### 1. 安装依赖

```bash
cd qualifiers_fetcher
uv pip install -r requirements.txt
```

### 2. 准备CAMELSH数据

#### 方式A：如果已有CSV文件

确保你的CAMELSH数据格式为：
- **径流数据**: `flow_hourly.csv` (index=时间, columns=gauge_id)
- **水位数据**: `waterlevel_hourly.csv` (index=时间, columns=gauge_id)

#### 方式B：从CAMELSH Python包导出

```bash
# 编辑 export_camelsh_data.py 中的配置
uv run python export_camelsh_data.py
```

### 3. 配置参数

编辑 `config.py`：

```python
# 1. 指定站点ID
GAUGE_IDS = [
    "01646500",
    "01434000",
    # ...
]

# 或从CAMELSH文件自动读取
gauge_ids = load_gauge_ids_from_camelsh(
    flow_file="camelsh_exported/flow_hourly.csv"
)

# 2. 设置时间范围
START_DATE = "2001-01-01"
END_DATE = "2024-12-31"

# 3. 设置文件路径
CAMELSH_FLOW_FILE = "camelsh_exported/flow_hourly.csv"
CAMELSH_WATERLEVEL_FILE = "camelsh_exported/waterlevel_hourly.csv"
```

### 4. 运行程序

```bash
uv run python usgs_qualifiers_fetcher.py
```

## 输出文件

### 1. `camelsh_with_qualifiers.csv`

合并后的数据，包含以下列：

| 列名 | 说明 |
|-----|------|
| datetime | 时间戳（UTC） |
| gauge_id | 站点ID |
| Q | 径流值（来自CAMELSH） |
| H | 水位值（来自CAMELSH） |
| Q_flag | 径流数据质量标签 |
| H_flag | 水位数据质量标签 |
| Q_weight | 径流数据权重（0-1，可选） |
| H_weight | 水位数据权重（0-1，可选） |

### 2. `qualifiers_report.txt`

统计报告，包含：
- 数据概览
- Qualifier分布
- 权重统计
- 数据完整性分析

## USGS Qualifiers 说明

### 常见Qualifiers

| 代码 | 含义 | 权重建议 |
|-----|------|---------|
| A | Approved for publication（已批准发布） | 1.0 |
| P | Provisional（临时数据，可能修订） | 0.9 |
| e | Estimated（估计值） | 0.7 |
| < | Less than indicated value（小于标示值） | 0.6 |
| > | Greater than indicated value（大于标示值） | 0.6 |
| Ice | Ice affected（受冰冻影响） | 0.5 |
| Bkw | Backwater（回水影响） | 0.5 |
| Eqp | Equipment malfunction（设备故障） | 0.3 |
| missing | No qualifier data（无质量标记） | 0.0 |

## 高级用法

### 1. 自定义权重规则

在 `config.py` 中修改：

```python
CUSTOM_QUALIFIER_WEIGHTS = {
    'A': 1.0,
    'P': 0.85,  # 调整临时数据权重
    'e': 0.6,   # 调整估计值权重
    # ...
}
```

### 2. 批量处理多个时间段

```python
from usgs_qualifiers_fetcher import USGSQualifiersFetcher

fetcher = USGSQualifiersFetcher()

time_ranges = [
    ("2001-01-01", "2010-12-31"),
    ("2011-01-01", "2020-12-31"),
    ("2021-01-01", "2024-12-31"),
]

for start, end in time_ranges:
    qualifiers = fetcher.fetch_multiple_gauges(
        gauge_ids=GAUGE_IDS,
        start_date=start,
        end_date=end
    )
    # 处理数据...
```

### 3. 只获取特定参数

修改 `usgs_qualifiers_fetcher.py` 中的参数代码：

```python
# 只获取径流
params = {
    'parameterCd': "00060",  # 只要discharge
    # ...
}

# 只获取水位
params = {
    'parameterCd': "00065",  # 只要gage height
    # ...
}
```

## API限制和注意事项

### USGS NWIS API限制

1. **请求频率**: 建议每个请求间隔0.5秒以上
2. **数据量**: 单次请求最好不超过1年数据
3. **超时**: 默认30秒超时，可调整

### 缓存机制

- 首次请求会缓存到 `qualifiers_cache/`
- 缓存文件命名: `{gauge_id}_{start_date}_{end_date}.json`
- 删除缓存文件可强制重新请求

### 数据对齐

- USGS数据会自动转换为UTC时区
- 与CAMELSH数据按时间戳精确对齐
- 时间分辨率：保持与CAMELSH一致

## 故障排除

### 问题1: API请求失败

```
警告: 请求失败 - 01646500: Connection timeout
```

**解决方案**:
- 检查网络连接
- 增加 `REQUEST_DELAY`
- 检查站点ID是否正确

### 问题2: 缺少qualifiers

```
径流缺失qualifiers: 12345 (50.00%)
```

**可能原因**:
- 该站点在USGS系统中不存在
- 该时间段无instantaneous values数据
- 使用daily values而非instantaneous values

**解决方案**:
- 检查站点是否在USGS系统中
- 尝试调整时间范围
- 考虑使用USGS Daily Values API（需修改代码）

### 问题3: 时区不匹配

**解决方案**:
- 代码会自动处理时区转换为UTC
- 确保CAMELSH数据时区正确
- 检查 `merge_with_camelsh()` 中的时区处理逻辑

## 参考资料

### USGS NWIS文档

- [NWIS Web Services](https://waterservices.usgs.gov/)
- [Instantaneous Values API](https://waterservices.usgs.gov/rest/IV-Service.html)
- [Parameter Codes](https://help.waterdata.usgs.gov/parameter_cd?group_cd=PHY)
- [Qualifiers说明](https://help.waterdata.usgs.gov/codes-and-parameters/instantaneous-value-qualification-code-iv_rmk_cd)

### 相关工具

- [dataretrieval-python](https://github.com/DOI-USGS/dataretrieval-python) - USGS官方Python包
- [HyRiver](https://github.com/hyriver/hyriver) - 水文数据获取工具集

## 许可证

本工具遵循项目主许可证。

## 联系方式

如有问题，请在项目仓库提Issue。

---

**创建日期**: 2025-12-22  
**最后更新**: 2025-12-22

