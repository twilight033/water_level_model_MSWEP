import torch
import platform
import sys
import subprocess

print("="*70)
print("系统信息")
print("="*70)
print(f"操作系统: {platform.system()} {platform.release()}")
print(f"Python版本: {sys.version}")
print()

print("="*70)
print("PyTorch / CUDA 检查")
print("="*70)
print(f"PyTorch版本: {torch.__version__}")
print(f"torch.cuda.is_available(): {torch.cuda.is_available()}")
print(f"torch.version.cuda: {torch.version.cuda}")
print(f"GPU数量: {torch.cuda.device_count()}")

if torch.cuda.is_available():
    print("\n检测到可用GPU：")
    for i in range(torch.cuda.device_count()):
        print(f"  [{i}] {torch.cuda.get_device_name(i)}")
        props = torch.cuda.get_device_properties(i)
        print(f"      显存: {props.total_memory/1024**3:.2f} GB")
else:
    print("\n⚠️ PyTorch 没检测到 GPU\n")

print()

print("="*70)
print("显卡驱动 / NVIDIA 检查")
print("="*70)
# 检查 nvidia-smi
try:
    result = subprocess.run(["nvidia-smi"], capture_output=True, text=True)
    if result.returncode == 0:
        print(result.stdout)
    else:
        print("⚠️ 找不到 nvidia-smi，可能是：未安装驱动 / 无独立GPU / WSL未配置")
except FileNotFoundError:
    print("⚠️ 系统没有 nvidia-smi（驱动未安装 或 未配置）")

print()

# 进一步 GPU 张量移动测试
print("="*70)
print("GPU 张量测试")
print("="*70)

try:
    if torch.cuda.is_available():
        x = torch.rand(1000, 1000).to("cuda")
        print("✔️ 张量成功创建在 GPU 上")
    else:
        print("❌ 因 torch.cuda.is_available() = False，跳过 GPU 张量测试")
except Exception as e:
    print("❌ GPU 张量创建失败：", e)

print("\n检查完成！")
