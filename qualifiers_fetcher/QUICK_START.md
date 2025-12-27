# USGS Qualifiers Fetcher - 快速上手指南

## 📦 已创建的文件（共9个）

```
qualifiers_fetcher/
├── usgs_qualifiers_fetcher.py   # ⭐ 主程序（核心功能）
├── config.py                     # 配置文件
├── export_camelsh_data.py       # CAMELSH数据导出工具
├── run_quick.py                 # ⭐ 快速运行脚本（推荐）
├── usage_examples.py            # 使用示例（6个场景）
├── test_setup.py                # ⭐ 测试脚本（先运行这个）
├── README.md                    # 详细文档
├── PROJECT_SUMMARY.md           # 项目总结
└── requirements.txt             # 依赖包
```

## 🚀 5分钟快速开始

### Step 1: 测试环境（必需）

```bash
cd qualifiers_fetcher
uv run python test_setup.py
```

这将测试：
- ✅ Python依赖包
- ✅ 文件结构
- ✅ USGS API连接
- ✅ CAMELSH数据
- ✅ 流域ID列表

### Step 2: 准备CAMELSH数据（如需要）

如果测试显示缺少CAMELSH数据：

```bash
# 编辑 export_camelsh_data.py 配置
# 然后运行：
uv run python export_camelsh_data.py
```

### Step 3: 运行程序

```bash
# 方式A: 快速运行（推荐，自动配置）
uv run python run_quick.py

# 方式B: 完整运行（需手动配置）
uv run python usgs_qualifiers_fetcher.py
```

## 📊 输出结果

运行后会生成：

```
qualifiers_fetcher/
├── qualifiers_cache/           # API响应缓存
│   └── {gauge_id}_{start}_{end}.json
├── qualifiers_output/          # 输出文件
│   ├── camelsh_with_qualifiers.csv  # ⭐ 主要输出
│   └── qualifiers_report.txt         # 统计报告
└── camelsh_exported/          # 导出的CAMELSH数据
    ├── flow_hourly.csv
    └── waterlevel_hourly.csv
```

### 主要输出文件说明

**`camelsh_with_qualifiers.csv`** 包含：

| 列 | 说明 |
|----|------|
| datetime | 时间戳（UTC） |
| gauge_id | 站点ID |
| Q | 径流值（CAMELSH） |
| H | 水位值（CAMELSH） |
| Q_flag | 径流质量标签（USGS） |
| H_flag | 水位质量标签（USGS） |
| Q_weight | 径流权重（0-1） |
| H_weight | 水位权重（0-1） |

## 💡 在训练中使用

### 方法1: 过滤低质量数据

```python
import pandas as pd

df = pd.read_csv('qualifiers_output/camelsh_with_qualifiers.csv')

# 只使用高质量数据（权重≥0.7）
df_high = df[(df['Q_weight'] >= 0.7) & (df['H_weight'] >= 0.7)]

print(f"原始: {len(df)} → 过滤后: {len(df_high)}")
```

### 方法2: 在PyTorch中使用权重

```python
import torch

def weighted_mse_loss(pred, target, weight):
    """带权重的MSE损失"""
    return ((pred - target) ** 2 * weight).sum() / weight.sum()

# 在训练循环中
for xs, ys, weights in dataloader:
    predictions = model(xs)
    loss = weighted_mse_loss(predictions, ys, weights)
    loss.backward()
```

### 方法3: 数据质量分析

```bash
# 运行使用示例
uv run python usage_examples.py

# 将生成可视化和统计分析
```

## 🔧 配置选项

### 修改站点和时间范围

编辑 `run_quick.py`:

```python
# 选择要处理的流域数量
N_BASINS = 10  # 测试用
# N_BASINS = len(basin_ids)  # 处理全部

# 时间范围
START_DATE = "2001-01-01"
END_DATE = "2024-12-31"
```

### 修改权重规则

编辑 `config.py`:

```python
CUSTOM_QUALIFIER_WEIGHTS = {
    'A': 1.0,      # Approved - 完全可信
    'P': 0.9,      # Provisional - 临时数据  
    'e': 0.7,      # Estimated - 估计值
    'Ice': 0.5,    # Ice affected - 冰冻影响
    'Eqp': 0.3,    # Equipment malfunction - 设备故障
    'missing': 0.0 # No qualifier - 无标签
}
```

## 📝 USGS Qualifiers 快速参考

| 代码 | 含义 | 默认权重 | 建议 |
|-----|------|---------|------|
| A | Approved（已批准发布） | 1.0 | ✓ 完全信任 |
| P | Provisional（临时数据） | 0.9 | ✓ 可使用 |
| e | Estimated（估计值） | 0.7 | △ 降权使用 |
| < | Less than（小于标示值） | 0.6 | △ 谨慎使用 |
| Ice | Ice affected（冰冻影响） | 0.5 | △ 视情况 |
| Eqp | Equipment malfunction（设备故障） | 0.3 | ✗ 建议排除 |
| missing | 无qualifiers数据 | 0.0 | ✗ 排除 |

## ⚠️ 常见问题

### Q1: API请求很慢？

**A**: 这是正常的。首次运行需要从USGS下载数据。
- 使用缓存（默认启用）
- 后续运行会快很多
- 可以先测试少量站点

### Q2: 很多missing qualifiers？

**A**: 可能原因：
- 站点无instantaneous values数据
- 使用了daily values而非iv
- 时间范围超出数据范围

**解决**: 检查USGS网站站点信息

### Q3: 如何验证结果？

```bash
# 查看统计报告
cat qualifiers_output/qualifiers_report.txt

# 运行可视化
uv run python usage_examples.py
# 会生成 data_quality_analysis.png
```

## 📚 进阶使用

### 分析特定站点

```python
from usgs_qualifiers_fetcher import USGSQualifiersFetcher

fetcher = USGSQualifiersFetcher()

# 单个站点详细分析
discharge_df, gage_height_df = fetcher.fetch_qualifiers_for_gauge(
    gauge_id="01646500",
    start_date="2020-01-01",
    end_date="2024-12-31"
)

print(discharge_df.head())
```

### 批量处理多年数据

```python
years = range(2001, 2025)
for year in years:
    start = f"{year}-01-01"
    end = f"{year}-12-31"
    
    qualifiers = fetcher.fetch_multiple_gauges(
        gauge_ids=gauge_ids,
        start_date=start,
        end_date=end
    )
    # 处理...
```

### 自定义数据处理

```python
# 读取结果
df = pd.read_csv('qualifiers_output/camelsh_with_qualifiers.csv')

# 添加自定义列
df['is_high_quality'] = (df['Q_weight'] >= 0.8) & (df['H_weight'] >= 0.8)

# 按站点分组分析
for gauge_id in df['gauge_id'].unique():
    gauge_data = df[df['gauge_id'] == gauge_id]
    quality_pct = gauge_data['is_high_quality'].mean()
    print(f"{gauge_id}: {quality_pct:.1%} 高质量数据")
```

## 🎯 最佳实践

1. **先测试小数据量**: 用10个站点测试，确认无误后再处理全部
2. **使用缓存**: 避免重复请求USGS API
3. **检查报告**: 查看`qualifiers_report.txt`了解数据质量
4. **权重可视化**: 运行`usage_examples.py`生成图表
5. **增量处理**: 大量站点时分批处理

## 📞 获取帮助

1. **运行测试**: `uv run python test_setup.py`
2. **查看文档**: 阅读 `README.md`
3. **查看示例**: 运行 `usage_examples.py`
4. **检查报告**: 查看 `qualifiers_report.txt`

## ✅ 快速检查清单

开始前：
- [ ] 已安装依赖 (`requirements.txt`)
- [ ] 已运行测试 (`test_setup.py`)
- [ ] CAMELSH数据已准备
- [ ] 流域ID列表已配置

运行后：
- [ ] 缓存目录有文件
- [ ] 输出CSV格式正确
- [ ] 报告显示合理统计
- [ ] 权重值在0-1范围
- [ ] 时间对齐正确

## 🚀 现在就开始！

```bash
cd qualifiers_fetcher
uv run python test_setup.py      # 1. 测试
uv run python run_quick.py       # 2. 运行
uv run python usage_examples.py  # 3. 分析
```

---

**提示**: 如有任何问题，请查看 `README.md` 获取详细文档。

