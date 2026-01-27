# 为什么只验证前1500个候选流域？

## 代码逻辑

```python
# multi_task_lstm_ablation_missing_labels.py 第1102行
max_candidates = min(len(VALID_WATER_LEVEL_BASINS), max(NUM_BASINS * 3, 200))
```

### 计算过程

当 `NUM_BASINS = 500` 时：

```python
max_candidates = min(5767, max(500 * 3, 200))
               = min(5767, max(1500, 200))
               = min(5767, 1500)
               = 1500
```

## 设计原因

### 1. ⏱️ **效率考虑**

验证每个流域的数据需要：
- 读取径流数据文件
- 读取水位数据文件
- 检查数据有效性（是否全为NaN）
- 计算有效数据比例

**时间成本示例**：
```
假设验证1个流域需要 2秒
验证5767个流域 = 5767 × 2秒 = 11534秒 ≈ 3.2小时 ❌
验证1500个流域 = 1500 × 2秒 = 3000秒 ≈ 50分钟 ✓
```

### 2. 🎯 **足够性原则**

**目标**：找到500个有效流域

**假设**：
- 候选列表中约60-70%的流域数据有效
- 验证1500个 → 预期得到 900-1050个有效流域 ✓
- 从中选择前500个 → 完全够用

**公式**：
```
验证数量 = 需要数量 × 安全系数
1500 = 500 × 3
```

安全系数为**3倍**，是一个经验值。

### 3. 📊 **实际效果**

根据终端输出，验证效果很好：
```
验证前1500个候选流域
 ↓
得到足够多的有效流域（≥500个）
 ↓
选择前500个
 ↓
成功！
```

## 代码的灵活性

这个公式实际上很灵活：

```python
max_candidates = min(len(VALID_WATER_LEVEL_BASINS), max(NUM_BASINS * 3, 200))
```

### 不同场景下的行为

| NUM_BASINS | NUM_BASINS × 3 | max(NUM_BASINS×3, 200) | min(5767, ...) | 最终验证数量 |
|------------|----------------|------------------------|----------------|--------------|
| 50         | 150            | 200                    | 200            | **200**      |
| 100        | 300            | 300                    | 300            | **300**      |
| 500        | 1500           | 1500                   | 1500           | **1500**     |
| 2000       | 6000           | 6000                   | 5767           | **5767**     |

**规则**：
1. 至少验证200个（`max(..., 200)`）
2. 优先验证3倍需求量（`NUM_BASINS * 3`）
3. 不超过候选总数（`min(len(...), ...)`）

## 潜在问题

### 问题1：如果前1500个不够怎么办？

**假设场景**：
```
验证前1500个 → 只有400个通过验证
需要500个 → 不够！
```

**代码行为**：
```python
chosen_basins = validated_basins[:NUM_BASINS]
# 如果 len(validated_basins) < NUM_BASINS
# 则 chosen_basins 只有 len(validated_basins) 个
```

**结果**：
- 代码不会报错
- 只使用400个流域继续训练
- 但可能不是你期望的结果

**实际情况**：
- 当前配置下（NUM_BASINS=500），验证1500个足够
- 从终端输出看，成功获得了500个有效流域

### 问题2：为什么不验证全部5767个？

**对比**：

| 方案 | 验证数量 | 时间成本 | 流域质量 | 推荐 |
|------|---------|---------|---------|------|
| 验证全部 | 5767 | ~3.2小时 | 最优 | ❌ 太慢 |
| 验证3倍 | 1500 | ~50分钟 | 良好 | ✓ 平衡 |
| 验证2倍 | 1000 | ~33分钟 | 可能不够 | ⚠️ 风险 |

**权衡**：
- 验证全部：时间太长，收益不大（只需要前500个）
- 验证3倍：时间可接受，几乎总能找到足够的有效流域

### 问题3：候选列表的顺序重要吗？

**非常重要！**

如果候选列表是随机排序：
```
前1500个可能都是低质量流域 → 通过验证的很少 ✗
```

如果候选列表按质量排序：
```
前1500个包含高质量流域 → 通过验证的很多 ✓
```

**当前情况**：
- `valid_waterlevel_basins.txt` 由 `scan_waterlevel_basins.py` 生成
- 可能按流域ID顺序排列（字母数字顺序）
- 不是按数据质量排序

**建议改进**：
```python
# 可以考虑先对候选列表排序
# 例如按地理位置、数据完整性等
VALID_WATER_LEVEL_BASINS_SORTED = sort_by_data_quality(VALID_WATER_LEVEL_BASINS)
```

## 如果需要更多流域怎么办？

### 方案1：增加验证数量

```python
# 修改代码中的倍数
max_candidates = min(len(VALID_WATER_LEVEL_BASINS), max(NUM_BASINS * 5, 200))
                                                              # 3 → 5
```

### 方案2：验证全部候选流域

```python
# 移除限制
max_candidates = len(VALID_WATER_LEVEL_BASINS)  # 验证全部5767个
```

**建议**：
- 如果 NUM_BASINS ≤ 500：保持3倍（1500个）
- 如果 NUM_BASINS = 1000：使用4-5倍（4000-5000个）
- 如果需要所有可用流域：验证全部

### 方案3：只验证MSWEP中存在的流域

```python
# 先筛选MSWEP中存在的候选流域
mswep_df = pd.read_csv("MSWEP/mswep_1000basins_mean_3hourly_1980_2024.csv", nrows=0)
mswep_basins = mswep_df.columns.tolist()[1:]

# 从候选列表中筛选
candidates_with_mswep = [b for b in VALID_WATER_LEVEL_BASINS if b in mswep_basins]
print(f"MSWEP覆盖的候选流域: {len(candidates_with_mswep)}")

# 然后验证这些流域
max_candidates = len(candidates_with_mswep)  # 只验证MSWEP中的
validated_basins = filter_basins_with_valid_data(
    basin_list=candidates_with_mswep,  # 改为预筛选的列表
    max_basins_to_check=max_candidates
)
```

## 总结

### 为什么是1500个？

1. **效率**：验证5767个需要3小时，1500个只需50分钟
2. **足够性**：3倍安全系数（500×3=1500），几乎总能找到足够的有效流域
3. **经验法则**：在时间成本和流域质量之间找平衡

### 公式解析

```python
max_candidates = min(len(VALID_WATER_LEVEL_BASINS), max(NUM_BASINS * 3, 200))
                 │                                  │                  │
                 │                                  │                  └─ 最少验证200个
                 │                                  └─ 优先验证3倍需求量
                 └─ 不超过候选总数
```

### 风险

- ⚠️ 如果前1500个质量太差，可能得不到500个有效流域
- ⚠️ 没有考虑MSWEP数据存在性
- ⚠️ 假设候选列表前面的流域质量不会特别差

### 改进建议

```python
# 改进版本1：增加MSWEP预筛选
mswep_basins = load_mswep_basin_list()
candidates_filtered = [b for b in VALID_WATER_LEVEL_BASINS if b in mswep_basins]

# 改进版本2：动态调整验证数量
max_candidates = min(len(VALID_WATER_LEVEL_BASINS), NUM_BASINS * 4)  # 3→4提高安全性

# 改进版本3：增加验证失败检查
if len(validated_basins) < NUM_BASINS:
    print(f"警告：只验证到 {len(validated_basins)} 个有效流域，少于请求的 {NUM_BASINS} 个")
    print("建议：增加 max_candidates 或降低 NUM_BASINS")
```

---

**结论**：1500是一个基于经验的安全值，在大多数情况下能以合理的时间成本找到足够的有效流域。但这个设计没有考虑MSWEP数据的存在性，这是当前问题的根源。

**生成时间**: 2026-01-21
