"""
辅助脚本：从CAMELSH数据集导出径流和水位数据
如果你的CAMELSH数据在Python包中，使用此脚本导出为CSV
"""

import sys
import os
from pathlib import Path

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import pandas as pd
import xarray as xr
from hydrodataset import StandardVariable
from improved_camelsh_reader import ImprovedCAMELSHReader


def export_camelsh_data(
    camelsh_data_path: str,
    gauge_ids: list,
    time_range: list,
    output_dir: str = "camelsh_exported"
):
    """
    从CAMELSH数据集导出径流和水位数据为CSV
    
    Parameters
    ----------
    camelsh_data_path : str
        CAMELSH数据路径
    gauge_ids : list
        要导出的站点ID列表
    time_range : list
        时间范围 [start_date, end_date]
    output_dir : str
        输出目录
    """
    os.makedirs(output_dir, exist_ok=True)
    
    print("=" * 80)
    print("从CAMELSH数据集导出数据")
    print("=" * 80)
    
    # 初始化CAMELSH读取器
    print(f"\n加载CAMELSH数据: {camelsh_data_path}")
    reader = ImprovedCAMELSHReader(camelsh_data_path, download=False, use_batch=True)
    
    # 导出径流数据
    print(f"\n导出径流数据...")
    print(f"  站点数: {len(gauge_ids)}")
    print(f"  时间范围: {time_range}")
    
    try:
        flow_ds = reader.read_ts_xrdataset(
            gage_id_lst=gauge_ids,
            t_range=time_range,
            var_lst=[StandardVariable.STREAMFLOW]
        )
        
        # 转换为DataFrame
        flow_df = flow_ds[StandardVariable.STREAMFLOW].to_pandas().T
        
        # 保存
        flow_output = os.path.join(output_dir, "flow_hourly.csv")
        flow_df.to_csv(flow_output)
        print(f"  已保存: {flow_output}")
        print(f"  形状: {flow_df.shape}")
        
    except Exception as e:
        print(f"  警告: 导出径流数据失败 - {e}")
        flow_df = None
    
    # 导出水位数据
    print(f"\n导出水位数据...")
    
    try:
        waterlevel_ds = reader.read_ts_xrdataset(
            gage_id_lst=gauge_ids,
            t_range=time_range,
            var_lst=[StandardVariable.WATER_LEVEL]
        )
        
        # 转换为DataFrame
        waterlevel_df = waterlevel_ds[StandardVariable.WATER_LEVEL].to_pandas().T
        
        # 保存
        waterlevel_output = os.path.join(output_dir, "waterlevel_hourly.csv")
        waterlevel_df.to_csv(waterlevel_output)
        print(f"  已保存: {waterlevel_output}")
        print(f"  形状: {waterlevel_df.shape}")
        
    except Exception as e:
        print(f"  警告: 导出水位数据失败 - {e}")
        waterlevel_df = None
    
    print("\n" + "=" * 80)
    print("导出完成!")
    print("=" * 80)
    
    return flow_df, waterlevel_df


if __name__ == "__main__":
    # ==================== 配置 ====================
    
    # CAMELSH数据路径（从项目主配置读取）
    try:
        import sys
        sys.path.append('..')
        from config import CAMELSH_DATA_PATH
        print(f"从项目配置读取CAMELSH路径: {CAMELSH_DATA_PATH}")
    except ImportError:
        # 如果无法读取主配置，使用默认路径
        CAMELSH_DATA_PATH = "../camelsh_data/CAMELSH"
        print(f"使用默认路径: {CAMELSH_DATA_PATH}")
    
    # 站点ID（从valid_waterlevel_basins.txt读取）
    import ast
    basin_file = "../valid_waterlevel_basins.txt"
    
    if os.path.exists(basin_file):
        print(f"从 {basin_file} 读取流域ID...")
        with open(basin_file, 'r', encoding='utf-8') as f:
            content = f.read()
            start_idx = content.find('[')
            end_idx = content.rfind(']') + 1
            GAUGE_IDS = ast.literal_eval(content[start_idx:end_idx])
        print(f"读取了 {len(GAUGE_IDS)} 个流域ID")
    else:
        print(f"警告: {basin_file} 不存在，使用示例流域ID")
        GAUGE_IDS = [
            '01646500', '01434000', '01042500', '01055000', '01057000'
        ]
    
    # 时间范围（与项目保持一致：2001-2024）
    TIME_RANGE = ["2001-01-01", "2024-12-31"]
    
    # 输出目录
    OUTPUT_DIR = "camelsh_exported"
    
    # ==================== 执行导出 ====================
    
    print("\n提示：首次导出可能需要较长时间，建议先测试少量流域")
    print(f"当前将导出前50个流域（共{len(GAUGE_IDS)}个可用）")
    print("如需导出全部，请修改下面的切片")
    
    flow_df, waterlevel_df = export_camelsh_data(
        camelsh_data_path=CAMELSH_DATA_PATH,
        gauge_ids=GAUGE_IDS[:50],  # 导出前50个流域
        time_range=TIME_RANGE,
        output_dir=OUTPUT_DIR
    )
    
    print(f"\n导出的文件可用于 usgs_qualifiers_fetcher.py")
    print(f"  径流数据: {OUTPUT_DIR}/flow_hourly.csv")
    print(f"  水位数据: {OUTPUT_DIR}/waterlevel_hourly.csv")

