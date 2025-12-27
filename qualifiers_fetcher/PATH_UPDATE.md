# 配置更新说明

## 已修复的路径问题

根据 `multi_task_lstm.py` 中的实际路径结构，已更新以下文件：

### 1. `config.py`

**更新内容**：
- ✅ 自动从项目主 `config.py` 读取 `CAMELSH_DATA_PATH`
- ✅ 如果无法读取，使用默认路径 `../camelsh_data/CAMELSH`
- ✅ CAMELSH导出文件路径更新为 `camelsh_exported/`

**新的路径结构**：
```python
# 自动读取项目配置
from config import CAMELSH_DATA_PATH  # 从主项目config.py

# 导出的CSV文件路径
CAMELSH_FLOW_FILE = "camelsh_exported/flow_hourly.csv"
CAMELSH_WATERLEVEL_FILE = "camelsh_exported/waterlevel_hourly.csv"
```

### 2. `export_camelsh_data.py`

**更新内容**：
- ✅ 从项目主配置读取 `CAMELSH_DATA_PATH`
- ✅ 从 `valid_waterlevel_basins.txt` 自动读取流域ID
- ✅ 时间范围更新为 2001-2024（与项目一致）
- ✅ 添加文件存在检查和错误处理

**使用方式**：
```bash
cd qualifiers_fetcher
uv run python export_camelsh_data.py
```

输出：
- `camelsh_exported/flow_hourly.csv` - 径流数据
- `camelsh_exported/waterlevel_hourly.csv` - 水位数据

### 3. `run_quick.py`

**更新内容**：
- ✅ 时间范围注释更清晰
- ✅ 与项目配置保持一致

## 📁 项目目录结构

```
water_level_model_MSWEP/
├── config.py                      # 主配置（包含CAMELSH_DATA_PATH）
├── valid_waterlevel_basins.txt    # 流域ID列表
├── camelsh_data/
│   └── CAMELSH/                   # CAMELSH数据集
│       ├── attributes/
│       └── timeseries/
├── qualifiers_fetcher/            # Qualifiers工具
│   ├── config.py                  # ✅ 已更新：自动读取主配置
│   ├── export_camelsh_data.py    # ✅ 已更新：自动读取配置
│   ├── run_quick.py               # ✅ 已更新
│   └── camelsh_exported/          # 导出的CSV文件（自动创建）
│       ├── flow_hourly.csv
│       └── waterlevel_hourly.csv
└── ...
```

## 🚀 使用流程

### Step 1: 导出CAMELSH数据

```bash
cd qualifiers_fetcher
uv run python export_camelsh_data.py
```

这会：
1. 从 `../config.py` 读取 `CAMELSH_DATA_PATH`
2. 从 `../valid_waterlevel_basins.txt` 读取流域ID
3. 导出前50个流域的数据到 `camelsh_exported/`

### Step 2: 运行qualifiers获取

```bash
uv run python run_quick.py
```

这会：
1. 从 `../valid_waterlevel_basins.txt` 读取流域ID
2. 从USGS获取qualifiers
3. 与导出的CAMELSH数据合并
4. 输出到 `qualifiers_output/camelsh_with_qualifiers.csv`

## 🔧 配置说明

### 如果你的CAMELSH路径不同

编辑主项目的 `config.py`：

```python
# 在项目根目录的 config.py 中
CAMELSH_DATA_PATH = "你的实际路径"  # 例如: "F:/data"
```

`qualifiers_fetcher` 会自动读取这个配置。

### 如果无法读取主配置

`qualifiers_fetcher/config.py` 会使用默认路径：

```python
CAMELSH_DATA_PATH = "../camelsh_data/CAMELSH"
```

你可以直接在 `qualifiers_fetcher/export_camelsh_data.py` 中修改：

```python
CAMELSH_DATA_PATH = "你的路径"
```

## ⚠️ 注意事项

1. **CAMELSH数据格式**：
   - 导出的CSV格式：index=时间，columns=gauge_id
   - 径流：小时分辨率
   - 水位：小时分辨率

2. **时间范围**：
   - 项目使用 2001-2024 数据
   - MSWEP使用 3小时分辨率
   - CAMELSH使用 小时分辨率

3. **流域ID**：
   - 从 `valid_waterlevel_basins.txt` 读取
   - 该文件由主项目生成
   - 包含有水位数据的流域

## ✅ 验证配置

运行测试脚本：

```bash
cd qualifiers_fetcher
uv run python test_setup.py
```

这会检查：
- ✅ 依赖包
- ✅ 文件结构
- ✅ USGS API连接
- ✅ CAMELSH数据
- ✅ 流域ID列表

## 📝 总结

所有路径配置已更新为：

1. ✅ **自动从主项目配置读取**
2. ✅ **与 multi_task_lstm.py 保持一致**
3. ✅ **支持相对路径和绝对路径**
4. ✅ **添加了完善的错误处理**
5. ✅ **提供清晰的日志输出**

现在可以直接运行，无需手动修改路径！

---

**更新日期**: 2025-12-22  
**基于**: multi_task_lstm.py 的路径结构

