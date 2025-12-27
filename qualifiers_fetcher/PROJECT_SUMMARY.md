# USGS Qualifiers Fetcher - 项目总结

## 📋 创建的文件

### 核心文件
1. **`usgs_qualifiers_fetcher.py`** (约600行)
   - 核心功能类 `USGSQualifiersFetcher`
   - USGS NWIS API交互
   - 数据解析和合并
   - 权重计算
   - 统计报告生成

2. **`config.py`** (约150行)
   - 配置参数管理
   - 站点ID加载工具
   - 时间范围配置
   - 权重规则自定义

3. **`export_camelsh_data.py`** (约100行)
   - 从CAMELSH Python包导出CSV
   - 批量导出径流和水位数据

4. **`run_quick.py`** (约150行)
   - 快速运行脚本
   - 自动从项目配置读取参数
   - 适合快速测试

5. **`usage_examples.py`** (约300行)
   - 6个使用示例
   - 训练中权重应用
   - 数据质量分析
   - 可视化工具

6. **`README.md`** (约400行)
   - 完整使用文档
   - API说明
   - 故障排除

7. **`requirements.txt`**
   - 依赖包列表

## 🎯 功能特性

### 1. 数据获取
- ✅ 使用USGS NWIS Instantaneous Values API
- ✅ 支持批量查询多个站点
- ✅ 查询径流(00060)和水位(00065)
- ✅ 提取每条观测的qualifiers
- ✅ 智能缓存避免重复请求

### 2. 数据处理
- ✅ 自动时区转换（UTC）
- ✅ 与CAMELSH数据按[gauge_id, time]对齐
- ✅ 生成Q, H, Q_flag, H_flag列
- ✅ 可选权重计算（Q_weight, H_weight）

### 3. 质量评估
- ✅ 支持所有USGS qualifier代码
- ✅ 可自定义权重规则
- ✅ 生成详细统计报告
- ✅ 数据完整性分析

### 4. 易用性
- ✅ 模块化设计
- ✅ 详细文档和示例
- ✅ 错误处理和日志
- ✅ 进度条显示

## 📊 输出数据格式

### camelsh_with_qualifiers.csv

| 列名 | 数据类型 | 说明 |
|-----|---------|------|
| datetime | datetime | 时间戳（UTC） |
| gauge_id | str | 站点ID |
| Q | float | 径流值（来自CAMELSH） |
| H | float | 水位值（来自CAMELSH） |
| Q_flag | str | 径流质量标签 |
| H_flag | str | 水位质量标签 |
| Q_weight | float | 径流权重（0-1） |
| H_weight | float | 水位权重（0-1） |

## 🚀 快速开始

### 最简单的使用方式

```bash
cd qualifiers_fetcher

# 1. 安装依赖
uv pip install -r requirements.txt

# 2. 导出CAMELSH数据（如果需要）
uv run python export_camelsh_data.py

# 3. 运行快速脚本
uv run python run_quick.py
```

### 自定义运行

```python
from usgs_qualifiers_fetcher import USGSQualifiersFetcher

# 初始化
fetcher = USGSQualifiersFetcher()

# 获取qualifiers
qualifiers = fetcher.fetch_multiple_gauges(
    gauge_ids=["01646500", "01434000"],
    start_date="2020-01-01",
    end_date="2024-12-31"
)

# 合并数据
merged_df = fetcher.merge_with_camelsh(
    camelsh_flow_file="flow.csv",
    camelsh_waterlevel_file="waterlevel.csv",
    qualifiers_data=qualifiers
)
```

## 💡 在训练中使用

### 方法1: 过滤低质量数据

```python
# 只使用高质量数据
df = pd.read_csv('camelsh_with_qualifiers.csv')
df_high_quality = df[
    (df['Q_weight'] >= 0.7) & 
    (df['H_weight'] >= 0.7)
]
```

### 方法2: 加权损失函数

```python
def weighted_mse_loss(pred, target, weight):
    return ((pred - target) ** 2 * weight).sum() / weight.sum()

# 在训练中
loss = weighted_mse_loss(predictions, targets, weights)
```

### 方法3: 加权采样

```python
from torch.utils.data import WeightedRandomSampler

sampler = WeightedRandomSampler(
    weights=df['Q_weight'].values,
    num_samples=len(df),
    replacement=True
)

dataloader = DataLoader(dataset, sampler=sampler)
```

## 📈 Qualifiers说明

### 常见代码

| 代码 | 含义 | 默认权重 | 建议处理 |
|-----|------|---------|---------|
| A | Approved（已批准） | 1.0 | 完全信任 |
| P | Provisional（临时） | 0.9 | 可使用 |
| e | Estimated（估计） | 0.7 | 降权使用 |
| Ice | Ice affected（冰冻） | 0.5 | 视情况使用 |
| Eqp | Equipment malfunction（故障） | 0.3 | 谨慎使用 |
| missing | 无qualifiers | 0.0 | 建议排除 |

### 权重计算逻辑

```python
# 多个qualifiers组合时
# 例如: "P,e" → weight = 0.9 * 0.7 = 0.63

qualifiers = "P,e".split(',')
weight = 1.0
for q in qualifiers:
    weight *= QUALIFIER_WEIGHTS[q]
```

## 🔧 高级配置

### 1. 修改权重规则

在 `config.py` 中：

```python
CUSTOM_QUALIFIER_WEIGHTS = {
    'A': 1.0,
    'P': 0.85,  # 调整临时数据权重
    'e': 0.6,   # 调整估计值权重
    # ...
}
```

### 2. 处理大量站点

```python
# 分批处理
batch_size = 50
for i in range(0, len(all_gauge_ids), batch_size):
    batch_ids = all_gauge_ids[i:i+batch_size]
    qualifiers = fetcher.fetch_multiple_gauges(batch_ids, ...)
    # 处理并保存...
```

### 3. 自定义时间分辨率

代码默认处理小时数据，如需改为日数据：

```python
# 在merge_with_camelsh中添加重采样
merged_df = merged_df.resample('D', on='datetime').mean()
```

## ⚠️ 注意事项

### API使用限制

1. **请求频率**: 建议≥0.5秒间隔
2. **数据量**: 单次请求不超过1年
3. **缓存**: 首次运行慢，后续使用缓存快

### 数据对齐

1. **时区**: 自动转换为UTC
2. **分辨率**: 保持与CAMELSH一致
3. **缺失值**: 用'missing'标记

### 质量控制

1. **验证站点**: 不是所有站点都有instantaneous values
2. **检查时间范围**: 部分站点数据不完整
3. **人工审核**: 建议抽查结果

## 🐛 故障排除

### 问题1: 请求超时

```python
# 增加timeout和delay
fetcher.fetch_qualifiers_for_gauge(..., timeout=60)
time.sleep(1.0)  # 增加延迟
```

### 问题2: 大量missing

可能原因：
- 站点无instantaneous values数据
- 使用了错误的参数代码
- 时间范围超出数据范围

解决：
- 检查USGS网站站点信息
- 尝试使用daily values API
- 调整时间范围

### 问题3: 内存不足

```python
# 分批处理和保存
for batch in batches:
    result = process_batch(batch)
    result.to_csv(f'output_{batch_id}.csv')
    del result  # 释放内存
```

## 📚 参考资源

### USGS文档
- [NWIS Web Services](https://waterservices.usgs.gov/)
- [Qualifiers说明](https://help.waterdata.usgs.gov/codes-and-parameters/instantaneous-value-qualification-code-iv_rmk_cd)

### Python工具
- [dataretrieval-python](https://github.com/DOI-USGS/dataretrieval-python) - 官方工具
- [HyRiver](https://github.com/hyriver/hyriver) - 综合水文工具

## 🎓 扩展建议

### 1. 添加更多参数

```python
# 除了00060和00065，还可以获取：
# 00010 - Temperature
# 00095 - Specific conductance
# 00300 - Dissolved oxygen
```

### 2. 时间序列分析

```python
# 分析qualifiers随时间的变化
# 识别数据质量趋势
# 检测异常时段
```

### 3. 站点质量评级

```python
# 基于qualifiers给站点打分
# 选择高质量站点进行训练
# 生成站点质量报告
```

## ✅ 测试清单

运行前检查：

- [ ] Python环境正确（推荐使用uv）
- [ ] 依赖包已安装
- [ ] CAMELSH数据文件存在
- [ ] 站点ID列表正确
- [ ] 时间范围合理
- [ ] 输出目录有写权限

运行后验证：

- [ ] 缓存文件生成
- [ ] 输出CSV格式正确
- [ ] 统计报告合理
- [ ] 权重值在0-1之间
- [ ] 时间对齐正确

## 📞 支持

如有问题：
1. 查看 `README.md` 详细文档
2. 运行 `usage_examples.py` 查看示例
3. 检查 `qualifiers_report.txt` 统计信息
4. 在项目仓库提Issue

---

**创建日期**: 2025-12-22  
**版本**: 1.0  
**作者**: AI Assistant

