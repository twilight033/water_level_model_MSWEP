"""
CAMELSH数据集配置文件

在这里修改您的CAMELSH数据路径和其他配置
"""

# ==================== CAMELSH数据路径配置 ====================
# 修改为您的实际CAMELSH数据路径
# 支持相对路径和绝对路径
CAMELSH_DATA_PATH = "F:/data"

# 注意：确保路径下有CAMELSH目录，包含以下结构：
# F:/data/CAMELSH/
# ├── attributes/
# ├── timeseries/
# ├── shapefiles/
# └── 其他相关文件

# 示例路径（取消注释并修改为您的路径）:
# CAMELSH_DATA_PATH = "D:/data/camelsh"
# CAMELSH_DATA_PATH = "/home/user/data/camelsh"
# CAMELSH_DATA_PATH = "../camelsh_dataset"

# ==================== 模型训练配置 ====================
# 流域数量（用于测试）
NUM_BASINS = 100

# 可用流域列表（经过验证有完整数据文件的流域）
AVAILABLE_BASINS = [
    '01011000', '01017000', '01017060', '01017290', '01017960',
    '01018000', '01018009'  # 可以根据需要添加更多
]

# 有有效水位数据的流域列表
VALID_WATER_LEVEL_BASINS = [
    '01017000', '01017060', '01017290', '01017960'
]

# 序列长度
SEQUENCE_LENGTH = 100

# 批次大小
BATCH_SIZE = 32

# 训练轮数
EPOCHS = 10

# 学习率
LEARNING_RATE = 0.001

# ==================== 时间范围配置 ====================
# 训练时间范围
TRAIN_START = "2010-01-01"
TRAIN_END = "2015-12-31"

# 验证时间范围
VALID_START = "2016-01-01"
VALID_END = "2018-12-31"

# 测试时间范围
TEST_START = "2019-01-01"
TEST_END = "2020-12-31"

# ==================== 特征变量配置 ====================
# 气象强迫变量（使用StandardVariable）
FORCING_VARIABLES = [
    "precipitation",
    "temperature_mean", 
    "solar_radiation",
    "potential_evapotranspiration"
]

# 流域属性变量（CAMELSH数据集中确实可用的变量）
ATTRIBUTE_VARIABLES = [
    "area",
    "p_mean",
    "p_seasonality", 
    "frac_snow",
    "aridity",
    #"elev_mean",
    "slope_mean",
    "frac_forest",
    "dom_land_cover",
    "soil_depth_statgso",
    "swc_pc_syr",
    "geol_class_1st",
    "geol_permeability"
    # 注意：只使用经过验证的属性变量
    # 如需添加更多变量，请先通过测试确认其可用性
]

# ==================== 输出配置 ====================
# 模型保存路径
MODEL_SAVE_PATH = "results"

# 结果保存路径
RESULTS_SAVE_PATH = "results"

# 日志保存路径
LOGS_SAVE_PATH = "results/logs"

# ==================== 设备配置 ====================
# 是否使用GPU（如果可用）
USE_GPU = True

# GPU设备ID（如果有多个GPU）
GPU_DEVICE_ID = 0

