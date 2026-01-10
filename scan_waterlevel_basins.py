#!/usr/bin/env python3
"""
扫描所有有水位数据文件的流域ID

方法1: 直接从Hourly2目录扫描所有NC文件
方法2: 从hydrodataset获取所有流域ID，然后检查Hourly2目录中哪些有对应文件
"""

from pathlib import Path
from config import CAMELSH_DATA_PATH
import sys

def scan_from_directory():
    """方法1: 直接从Hourly2目录扫描文件"""
    base_path = Path(CAMELSH_DATA_PATH)
    possible_paths = [
        base_path / "CAMELSH" / "Hourly2" / "Hourly2",
        base_path / "CAMELSH" / "Hourly2",
        base_path / "Hourly2" / "Hourly2",
        base_path / "Hourly2",
    ]
    
    hourly2_dir = None
    for path in possible_paths:
        if path.exists():
            hourly2_dir = path
            break
    
    if hourly2_dir is None:
        return None, None
    
    # 查找所有 *_hourly.nc 文件
    hourly_files = list(hourly2_dir.glob("*_hourly.nc"))
    
    # 如果没找到，尝试所有.nc文件
    if len(hourly_files) == 0:
        hourly_files = list(hourly2_dir.glob("*.nc"))
    
    if len(hourly_files) == 0:
        return hourly2_dir, []
    
    # 提取流域ID
    valid_basins = []
    for file_path in sorted(hourly_files):
        basin_id = file_path.stem.replace('_hourly', '')
        if basin_id not in valid_basins:
            valid_basins.append(basin_id)
    
    return hourly2_dir, valid_basins

def scan_from_basins(basin_ids):
    """方法2: 检查已知流域ID是否有Hourly2文件"""
    base_path = Path(CAMELSH_DATA_PATH)
    possible_paths = [
        base_path / "CAMELSH" / "Hourly2" / "Hourly2",
        base_path / "CAMELSH" / "Hourly2",
        base_path / "Hourly2" / "Hourly2",
        base_path / "Hourly2",
    ]
    
    hourly2_dir = None
    for path in possible_paths:
        if path.exists():
            hourly2_dir = path
            break
    
    if hourly2_dir is None:
        return None, []
    
    valid_basins = []
    for basin_id in basin_ids:
        # 检查是否有 {basin_id}_hourly.nc 文件
        hourly_file = hourly2_dir / f"{basin_id}_hourly.nc"
        if hourly_file.exists():
            valid_basins.append(str(basin_id))
    
    return hourly2_dir, valid_basins

def scan_waterlevel_basins():
    """扫描所有有水位数据文件的流域"""
    
    print("=" * 80)
    print("扫描有水位数据文件的流域")
    print("=" * 80)
    
    # 首先尝试方法1: 直接从目录扫描
    print("\n方法1: 从Hourly2目录扫描所有NC文件...")
    hourly2_dir, valid_basins_dir = scan_from_directory()
    
    if hourly2_dir is None:
        print("错误: 未找到Hourly2目录!")
        print(f"请检查配置: CAMELSH_DATA_PATH = {CAMELSH_DATA_PATH}")
        return []
    
    if valid_basins_dir is not None and len(valid_basins_dir) > 0:
        print(f"找到Hourly2目录: {hourly2_dir}")
        print(f"从目录扫描到 {len(valid_basins_dir)} 个有水位数据文件的流域")
        valid_basins = valid_basins_dir
    else:
        print("目录中没有找到NC文件")
        print("\n方法2: 从hydrodataset获取所有流域ID，然后检查Hourly2文件...")
        
        try:
            from improved_camelsh_reader import ImprovedCAMELSHReader
            reader = ImprovedCAMELSHReader(CAMELSH_DATA_PATH, download=False)
            all_basin_ids = reader.read_object_ids()
            print(f"从hydrodataset获取到 {len(all_basin_ids)} 个流域ID")
            
            hourly2_dir, valid_basins = scan_from_basins(all_basin_ids)
            print(f"检查后，找到 {len(valid_basins)} 个有Hourly2文件的流域")
        except Exception as e:
            print(f"从hydrodataset获取流域ID失败: {e}")
            print("尝试使用config.py中的AVAILABLE_BASINS...")
            
            from config import AVAILABLE_BASINS
            hourly2_dir, valid_basins = scan_from_basins(AVAILABLE_BASINS)
            print(f"从AVAILABLE_BASINS检查后，找到 {len(valid_basins)} 个有Hourly2文件的流域")
    
    if len(valid_basins) == 0:
        print("\n警告: 未找到任何有水位数据文件的流域!")
        return []
    
    print(f"\n" + "=" * 80)
    print(f"扫描完成! 找到 {len(valid_basins)} 个有水位数据文件的流域")
    print("=" * 80)
    
    if len(valid_basins) > 0:
        print(f"\n有效流域ID列表 (共{len(valid_basins)}个):")
        print(f"VALID_WATER_LEVEL_BASINS = [")
        # 每10个一行打印
        for i in range(0, len(valid_basins), 10):
            batch = valid_basins[i:i+10]
            basin_str = ", ".join([f"'{b}'" for b in batch])
            if i + 10 < len(valid_basins):
                print(f"    {basin_str},")
            else:
                print(f"    {basin_str}")
        print(f"]")
        
        # 显示前30个作为示例
        print(f"\n前30个流域ID示例:")
        for i, basin_id in enumerate(valid_basins[:30], 1):
            print(f"  {i:2d}. {basin_id}")
        if len(valid_basins) > 30:
            print(f"  ... 还有 {len(valid_basins) - 30} 个")
        
        # 保存到文件
        output_file = "valid_waterlevel_basins.txt"
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write("# 有水位数据文件的流域ID列表\n")
            f.write("# 自动扫描生成\n\n")
            f.write("VALID_WATER_LEVEL_BASINS = [\n")
            for i in range(0, len(valid_basins), 10):
                batch = valid_basins[i:i+10]
                basin_str = ", ".join([f"'{b}'" for b in batch])
                if i + 10 < len(valid_basins):
                    f.write(f"    {basin_str},\n")
                else:
                    f.write(f"    {basin_str}\n")
            f.write("]\n")
        print(f"\n结果已保存到: {output_file}")
    
    return valid_basins

if __name__ == "__main__":
    try:
        valid_basins = scan_waterlevel_basins()
    except KeyboardInterrupt:
        print("\n\n扫描被用户中断")
        sys.exit(1)
    except Exception as e:
        print(f"\n错误: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
