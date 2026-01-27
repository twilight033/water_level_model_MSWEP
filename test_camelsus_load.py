"""测试 CAMELS-US 数据加载，显示详细进度"""
import sys
import time
import io

def main():
    # 设置输出编码为 UTF-8
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

    print("=" * 60)
    print("测试 CAMELS-US 数据加载")
    print("=" * 60)

    print("\n步骤1: 导入模块...")
    start = time.time()
    from hydrodataset import CamelsUs
    print(f"[OK] 导入完成 ({time.time()-start:.2f}秒)")

    print("\n步骤2: 初始化 CamelsUs 对象...")
    print("数据路径: D:/download/camels/camels_us")
    start = time.time()

    try:
        camels = CamelsUs(
            data_path="D:/download/camels/camels_us",
            download=False
        )
        print(f"[OK] 初始化完成 ({time.time()-start:.2f}秒)")
        
        print("\n步骤3: 获取流域列表...")
        start = time.time()
        basin_ids = camels.read_object_ids()
        print(f"[OK] 获取到 {len(basin_ids)} 个流域 ({time.time()-start:.2f}秒)")
        print(f"  前10个流域: {basin_ids[:10].tolist()}")
        
    except Exception as e:
        print(f"[ERROR] 错误: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    print("\n" + "=" * 60)
    print("测试完成！")
    print("=" * 60)

if __name__ == '__main__':
    main()
