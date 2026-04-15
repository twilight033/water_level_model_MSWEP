"""调试 aqua_fetch 检查的文件路径"""
import os

# 模拟 aqua_fetch 的路径检查逻辑
data_path = "D:/download/camels/camels_us"

# aqua_fetch 期望的文件
expected_files = {
    'basin_timeseries_v1p2_metForcing_obsFlow.zip': 'https://zenodo.org/records/15529996/files/',
    'basin_set_full_res.zip': 'https://zenodo.org/records/15529996/files/',
}

print("=" * 70)
print("调试 aqua_fetch 路径检查")
print("=" * 70)
print(f"\n数据路径: {data_path}")
print(f"路径存在: {os.path.exists(data_path)}")
print(f"绝对路径: {os.path.abspath(data_path)}")

print("\n" + "=" * 70)
print("检查文件:")
print("=" * 70)

for fname in expected_files.keys():
    fpath = os.path.join(data_path, fname)
    exists = os.path.exists(fpath)
    
    print(f"\n文件名: {fname}")
    print(f"  完整路径: {fpath}")
    print(f"  存在检查: {exists}")
    
    if exists:
        size = os.path.getsize(fpath)
        print(f"  文件大小: {size:,} 字节 ({size/1024/1024:.2f} MB)")
    else:
        print(f"  ⚠ 文件不存在！")
        
        # 检查是否有大小写问题
        dir_path = os.path.dirname(fpath)
        if os.path.exists(dir_path):
            print(f"  目录内容:")
            for item in os.listdir(dir_path):
                if fname.lower() in item.lower():
                    print(f"    - {item} (可能的匹配)")

print("\n" + "=" * 70)
print("完成")
print("=" * 70)
