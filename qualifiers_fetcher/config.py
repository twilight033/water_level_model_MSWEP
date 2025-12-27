"""
配置文件 - USGS Qualifiers Fetcher
根据你的实际情况修改这些配置
"""

# ==================== 站点配置 ====================

# 方式1: 手动指定站点ID列表
GAUGE_IDS = [
    "01646500",  # Potomac River
    "01434000",  # Delaware River
    # 添加更多站点...
]

# 方式2: 从文件读取站点ID（推荐用于大量站点）
def load_gauge_ids_from_file(file_path: str) -> list:
    """
    从文件加载站点ID列表
    
    文件格式：
    - 每行一个站点ID
    - 或CSV文件，第一列为站点ID
    """
    import pandas as pd
    
    if file_path.endswith('.csv'):
        df = pd.read_csv(file_path)
        # 假设第一列是gauge_id
        return df.iloc[:, 0].astype(str).tolist()
    else:
        # 纯文本文件，每行一个ID
        with open(file_path, 'r') as f:
            return [line.strip() for line in f if line.strip()]


# 方式3: 从CAMELSH数据文件读取站点ID
def load_gauge_ids_from_camelsh(flow_file: str, waterlevel_file: str = None) -> list:
    """从CAMELSH数据文件的列名中提取站点ID"""
    import pandas as pd
    
    flow_df = pd.read_csv(flow_file, index_col=0, nrows=0)  # 只读表头
    gauge_ids = set(flow_df.columns.astype(str))
    
    if waterlevel_file:
        wl_df = pd.read_csv(waterlevel_file, index_col=0, nrows=0)
        gauge_ids.update(wl_df.columns.astype(str))
    
    return sorted(list(gauge_ids))


# ==================== 时间范围配置 ====================

# 方式1: 固定时间范围
START_DATE = "2001-01-01"
END_DATE = "2024-12-31"

# 方式2: 从CAMELSH数据文件自动推断时间范围
def get_time_range_from_camelsh(flow_file: str) -> tuple:
    """从CAMELSH数据文件推断时间范围"""
    import pandas as pd
    
    df = pd.read_csv(flow_file, index_col=0, parse_dates=True)
    start_date = df.index.min().strftime("%Y-%m-%d")
    end_date = df.index.max().strftime("%Y-%m-%d")
    
    return start_date, end_date


# ==================== 文件路径配置 ====================

# CAMELSH数据文件路径（根据你的实际路径修改）
# 注意：这些文件需要先通过 export_camelsh_data.py 从CAMELSH数据集导出

# 选项1: 使用导出的CSV文件（推荐）
CAMELSH_FLOW_FILE = "camelsh_exported/flow_hourly.csv"
CAMELSH_WATERLEVEL_FILE = "camelsh_exported/waterlevel_hourly.csv"

# 选项2: 如果你的CAMELSH数据已经是CSV格式，指定实际路径
# CAMELSH_FLOW_FILE = "../path/to/your/flow_data.csv"
# CAMELSH_WATERLEVEL_FILE = "../path/to/your/waterlevel_data.csv"

# CAMELSH数据集路径（用于export_camelsh_data.py）
# 从项目主config.py读取，或在这里指定
try:
    import sys
    sys.path.append('..')
    from config import CAMELSH_DATA_PATH
    print(f"使用项目配置的CAMELSH路径: {CAMELSH_DATA_PATH}")
except ImportError:
    # 如果无法从主配置读取，使用默认值
    CAMELSH_DATA_PATH = "../camelsh_data/CAMELSH"
    print(f"使用默认CAMELSH路径: {CAMELSH_DATA_PATH}")


# ==================== 输出配置 ====================

# 输出目录
OUTPUT_DIR = "qualifiers_output"

# 缓存目录（避免重复请求USGS API）
CACHE_DIR = "qualifiers_cache"

# 输出文件名
OUTPUT_FILE = "camelsh_with_qualifiers.csv"


# ==================== API配置 ====================

# 是否使用缓存（强烈推荐，避免重复请求）
USE_CACHE = True

# 请求之间的延迟（秒）- 避免被USGS API限流
REQUEST_DELAY = 0.5

# 是否添加权重列（基于qualifiers计算数据质量权重）
ADD_WEIGHTS = True


# ==================== 高级配置 ====================

# 自定义qualifier权重映射
CUSTOM_QUALIFIER_WEIGHTS = {
    'A': 1.0,      # Approved - 完全可信
    'P': 0.9,      # Provisional - 临时数据
    'e': 0.7,      # Estimated - 估计值
    '<': 0.6,      # Less than - 小于某值
    '>': 0.6,      # Greater than - 大于某值
    'Ice': 0.5,    # Ice affected - 冰冻影响
    'Bkw': 0.5,    # Backwater - 回水影响
    'Eqp': 0.3,    # Equipment malfunction - 设备故障
    'missing': 0.0 # No qualifier data - 无质量标记
}


# ==================== 使用示例 ====================

if __name__ == "__main__":
    """
    这个配置文件可以直接运行以验证配置是否正确
    """
    
    print("=" * 80)
    print("配置验证")
    print("=" * 80)
    
    # 1. 验证站点ID
    print(f"\n站点配置:")
    print(f"  方式1 - 手动指定: {len(GAUGE_IDS)} 个站点")
    print(f"  示例: {GAUGE_IDS[:3]}")
    
    # 2. 验证时间范围
    print(f"\n时间范围配置:")
    print(f"  开始日期: {START_DATE}")
    print(f"  结束日期: {END_DATE}")
    
    # 3. 验证文件路径
    import os
    print(f"\n文件路径验证:")
    print(f"  径流数据: {CAMELSH_FLOW_FILE}")
    print(f"    存在: {os.path.exists(CAMELSH_FLOW_FILE)}")
    print(f"  水位数据: {CAMELSH_WATERLEVEL_FILE}")
    print(f"    存在: {os.path.exists(CAMELSH_WATERLEVEL_FILE)}")
    
    # 4. 如果文件存在，尝试读取站点ID
    if os.path.exists(CAMELSH_FLOW_FILE):
        try:
            gauge_ids_from_file = load_gauge_ids_from_camelsh(
                CAMELSH_FLOW_FILE,
                CAMELSH_WATERLEVEL_FILE if os.path.exists(CAMELSH_WATERLEVEL_FILE) else None
            )
            print(f"\n  从CAMELSH文件检测到 {len(gauge_ids_from_file)} 个站点")
            print(f"  示例: {gauge_ids_from_file[:5]}")
        except Exception as e:
            print(f"\n  警告: 无法从CAMELSH文件读取站点ID - {e}")
    
    print("\n" + "=" * 80)
    print("配置验证完成!")
    print("如果所有路径都存在，可以运行 usgs_qualifiers_fetcher.py")
    print("=" * 80)

