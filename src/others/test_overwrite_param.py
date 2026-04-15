"""测试 overwrite 参数是否正确传递"""
from aqua_fetch import CAMELS_US

# 测试1: 不传 overwrite (默认应该是 False)
print("=" * 70)
print("测试1: 不传 overwrite 参数")
print("=" * 70)
try:
    c1 = CAMELS_US.__new__(CAMELS_US)
    c1.__dict__['overwrite'] = None  # 预设
    print(f"准备初始化...")
except Exception as e:
    print(f"创建失败: {e}")

# 测试2: 显式传 overwrite=False
print("\n" + "=" * 70)
print("测试2: 传递 overwrite=False")
print("=" * 70)

# 使用 monkey patch 来拦截下载
original_init = CAMELS_US.__init__

def patched_init(self, path=None, data_source='daymet', **kwargs):
    print(f"CAMELS_US.__init__ 被调用:")
    print(f"  path={path}")
    print(f"  data_source={data_source}")
    print(f"  kwargs={kwargs}")
    
    # 调用父类初始化，但跳过下载部分
    from aqua_fetch.rr._camels import _RainfallRunoff
    _RainfallRunoff.__init__(self, path=path, name="CAMELS_US", **kwargs)
    
    print(f"\n初始化后的属性:")
    print(f"  self.overwrite={getattr(self, 'overwrite', 'NOT SET')}")
    print(f"  self.verbosity={getattr(self, 'verbosity', 'NOT SET')}")
    print(f"  self.path={getattr(self, 'path', 'NOT SET')}")
    
    # 不执行下载逻辑
    raise Exception("测试完成，跳过下载")

CAMELS_US.__init__ = patched_init

try:
    c2 = CAMELS_US("D:/download/camels/camels_us", overwrite=False, verbosity=1)
except Exception as e:
    if "测试完成" not in str(e):
        print(f"错误: {e}")

# 恢复原始方法
CAMELS_US.__init__ = original_init

print("\n" + "=" * 70)
print("测试完成")
print("=" * 70)
