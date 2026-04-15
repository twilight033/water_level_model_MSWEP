"""
一键运行脚本 - CAMELSH数据集多任务LSTM模型训练流程

这个脚本会自动执行：
1. 环境检查：验证CAMELSH数据集和依赖包
2. 数据集成测试：验证CAMELSH数据加载
3. 模型训练：训练多任务LSTM模型
4. 结果可视化：生成预测结果图表

使用方法:
    python run_all.py              # 完整流程
    python run_all.py --test-only  # 仅运行测试
    python run_all.py --skip-test  # 跳过测试直接训练
"""

import subprocess
import sys
import os
import argparse
from pathlib import Path
import time

def run_command(cmd, description, allow_failure=False):
    """运行命令并显示进度"""
    print("\n" + "=" * 80)
    print(f"{description}")
    print("=" * 80)
    print(f"执行命令: {cmd}")
    
    start_time = time.time()
    result = subprocess.run(cmd, shell=True)
    end_time = time.time()
    
    duration = end_time - start_time
    
    if result.returncode != 0:
        print(f"\n[错误] {description}失败 (耗时: {duration:.1f}秒)")
        if not allow_failure:
            sys.exit(1)
        return False
    
    print(f"\n[完成] {description} (耗时: {duration:.1f}秒)")
    return True

def check_camelsh_environment():
    """检查CAMELSH环境和依赖"""
    print("\n" + "=" * 60)
    print("环境检查")
    print("=" * 60)
    
    all_checks_passed = True
    
    # 检查Python版本
    python_version = sys.version_info
    if python_version >= (3, 10):
        print(f"  ✓ Python版本: {python_version.major}.{python_version.minor}.{python_version.micro}")
    else:
        print(f"  ✗ Python版本过低: {python_version.major}.{python_version.minor}.{python_version.micro}")
        print("  需要Python 3.10或更高版本")
        all_checks_passed = False
    
    # 检查必需的Python包
    required_packages = [
        ('hydrodataset', 'hydrodataset'),
        ('torch', 'torch'),
        ('numpy', 'numpy'),
        ('pandas', 'pandas'),
        ('xarray', 'xarray'),
        ('matplotlib', 'matplotlib'),
        ('tqdm', 'tqdm'),
        ('HydroErr', 'HydroErr')
    ]
    
    print("\n检查Python依赖包:")
    for package_name, import_name in required_packages:
        try:
            __import__(import_name)
            print(f"  ✓ {package_name}")
        except ImportError:
            print(f"  ✗ {package_name} (未安装)")
            print(f"    请运行: pip install {package_name}")
            all_checks_passed = False
    
    # 检查CAMELSH数据路径
    print("\n检查数据路径:")
    try:
        # 尝试初始化CAMELSH数据集
        from hydrodataset.camelsh import Camelsh
        from config import CAMELSH_DATA_PATH
        camelsh = Camelsh(CAMELSH_DATA_PATH, download=False)
        print(f"  ✓ CAMELSH数据路径: {camelsh.data_source_dir}")
        
        # 检查流域数量
        try:
            basin_ids = camelsh.read_object_ids()
            print(f"  ✓ 可用流域数量: {len(basin_ids)}")
        except Exception as e:
            print(f"  ⚠ 无法读取流域列表: {e}")
            
    except Exception as e:
        try:
            from config import CAMELSH_DATA_PATH
            data_path_msg = f"    1. CAMELSH数据路径正确: {CAMELSH_DATA_PATH}"
        except:
            data_path_msg = "    1. CAMELSH数据路径配置正确"
            
        print(f"  ✗ CAMELSH数据集初始化失败: {e}")
        print("  请确保:")
        print(data_path_msg)
        print("    2. 数据文件格式正确")
        print("    3. 参考: CAMELSH_使用说明.md")
        all_checks_passed = False
    
    # 检查hydro_setting.yml配置
    print("\n检查hydrodataset配置:")
    setting_file = Path.home() / "hydro_setting.yml"
    if setting_file.exists():
        print(f"  ✓ 配置文件存在: {setting_file}")
        
        # 尝试读取配置内容
        try:
            try:
                import yaml
            except ImportError:
                print("  ⚠ PyYAML未安装，无法验证配置文件内容")
                return all_checks_passed
                
            with open(setting_file, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
            
            if 'local_data_path' in config:
                print("  ✓ 配置文件格式正确")
                if 'root' in config['local_data_path']:
                    root_path = Path(config['local_data_path']['root'])
                    print(f"  ✓ 数据根目录: {root_path}")
                if 'cache' in config['local_data_path']:
                    cache_path = Path(config['local_data_path']['cache'])
                    print(f"  ✓ 缓存目录: {cache_path}")
            else:
                print("  ⚠ 配置文件格式可能不正确")
        except Exception as e:
            print(f"  ⚠ 配置文件读取警告: {e}")
    else:
        print(f"  ✗ 配置文件不存在: {setting_file}")
        print("  请创建配置文件，参考CAMELSH_使用说明.md")
        all_checks_passed = False
    
    # 检查输出目录
    print("\n检查输出目录:")
    output_dirs = ['models', 'results', 'logs']
    for dir_name in output_dirs:
        dir_path = Path(dir_name)
        if not dir_path.exists():
            dir_path.mkdir(parents=True, exist_ok=True)
            print(f"  ✓ 创建目录: {dir_name}/")
        else:
            print(f"  ✓ 目录存在: {dir_name}/")
    
    return all_checks_passed

def check_gpu_availability():
    """检查GPU可用性"""
    try:
        import torch
        if torch.cuda.is_available():
            gpu_count = torch.cuda.device_count()
            gpu_name = torch.cuda.get_device_name(0)
            print(f"  ✓ GPU可用: {gpu_name} (共{gpu_count}个GPU)")
            return True
        else:
            print("  ⚠ GPU不可用，将使用CPU训练")
            return False
    except:
        print("  ⚠ 无法检查GPU状态")
        return False

def parse_arguments():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(
        description="CAMELSH数据集多任务LSTM模型训练流程",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例用法:
  python run_all.py                    # 完整流程（推荐）
  python run_all.py --test-only        # 仅运行环境检查和数据测试
  python run_all.py --skip-test        # 跳过测试直接训练
  python run_all.py --quick-only       # 仅运行快速验证
  python run_all.py --no-interaction   # 非交互模式（自动跳过可选步骤）
        """
    )
    
    parser.add_argument('--test-only', action='store_true',
                        help='仅运行环境检查和数据集成测试')
    parser.add_argument('--skip-test', action='store_true',
                        help='跳过数据集成测试直接进行训练')
    parser.add_argument('--quick-only', action='store_true',
                        help='仅运行快速验证测试')
    parser.add_argument('--no-interaction', action='store_true',
                        help='非交互模式，自动跳过可选步骤')
    parser.add_argument('--verbose', '-v', action='store_true',
                        help='显示详细输出')
    
    return parser.parse_args()

if __name__ == "__main__":
    # 解析命令行参数
    args = parse_arguments()
    
    print("=" * 80)
    print("多任务LSTM模型 - 一键运行脚本 (CAMELSH版本)")
    print("=" * 80)
    print(f"开始时间: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    
    total_start_time = time.time()
    
    # 步骤1: 检查CAMELSH环境
    print("\n🔍 步骤 1: 环境检查")
    if not check_camelsh_environment():
        print("\n❌ 环境检查失败")
        print("\n📋 解决方案:")
        print("  1. 检查Python版本是否 >= 3.10")
        print("  2. 安装缺失的依赖包: pip install -r requirements.txt")
        print("  3. 下载CAMELSH数据集到 camelsh/camelsh_data/ 目录")
        print("  4. 配置 ~/hydro_setting.yml 文件")
        print("  5. 参考 CAMELSH_使用说明.md 获取详细指导")
        sys.exit(1)
    
    # 检查GPU
    print("\n🖥️  GPU检查:")
    check_gpu_availability()
    
    # 步骤2: 运行CAMELSH集成测试
    if not args.skip_test and not args.quick_only:
        print("\n🧪 步骤 2: CAMELSH数据集成测试")
        test_success = run_command(
            "python test_camelsh_integration.py",
            "CAMELSH数据集成测试",
            allow_failure=True
        )
        
        if not test_success:
            print("\n⚠️  数据集成测试失败，但可以继续训练")
            if not args.no_interaction:
                print("是否继续进行训练？(y/n): ", end="")
                user_input = input().strip().lower()
                if user_input not in ['y', 'yes', '是']:
                    print("用户取消操作")
                    sys.exit(1)
    
    # 步骤3: 快速验证（可选）
    if args.quick_only or (not args.test_only and not args.skip_test):
        if args.quick_only or not args.no_interaction:
            if args.quick_only:
                run_quick = True
            else:
                print("\n🚀 是否运行快速验证测试？(推荐用于首次运行) (y/n): ", end="")
                user_input = input().strip().lower()
                run_quick = user_input in ['y', 'yes', '是']
            
            if run_quick:
                print("\n⚡ 步骤 3: 快速验证测试")
                quick_success = run_command(
                    "python quick_start.py",
                    "快速验证测试",
                    allow_failure=True
                )
                
                if args.quick_only:
                    if quick_success:
                        print("\n✅ 快速验证测试完成！")
                    else:
                        print("\n❌ 快速验证测试失败！")
                    sys.exit(0 if quick_success else 1)
    
    # 步骤4: 完整模型训练
    if not args.test_only and not args.quick_only:
        print("\n🎯 步骤 4: 完整模型训练")
        
        if not args.no_interaction:
            print("准备开始完整模型训练，这可能需要较长时间。")
            print("继续？(y/n): ", end="")
            user_input = input().strip().lower()
            if user_input not in ['y', 'yes', '是']:
                print("用户取消训练")
                sys.exit(0)
        
        training_success = run_command(
            "python multi_task_lstm.py",
            "完整模型训练"
        )
        
        if training_success:
            print("\n🎉 模型训练完成！")
        else:
            print("\n❌ 模型训练失败！")
            sys.exit(1)
    
    # 显示总结信息
    total_end_time = time.time()
    total_duration = total_end_time - total_start_time
    
    print("\n" + "=" * 80)
    if args.test_only:
        print("✅ 环境检查和数据测试完成！")
    elif args.quick_only:
        print("⚡ 快速验证测试完成！")
    else:
        print("🎉 CAMELSH多任务LSTM模型训练流程完成！")
    print("=" * 80)
    print(f"总耗时: {total_duration:.1f}秒 ({total_duration/60:.1f}分钟)")
    print(f"结束时间: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    
    # 检查生成的文件
    print("\n📁 检查生成的文件:")
    
    # 模型文件
    model_files = [
        ("best_model.pth", "最佳模型权重"),
        ("training_log.txt", "训练日志"),
        ("model_config.json", "模型配置")
    ]
    
    print("  🔧 模型文件:")
    for filename, description in model_files:
        if Path(filename).exists():
            file_size = Path(filename).stat().st_size / (1024*1024)  # MB
            print(f"    ✓ {filename} ({description}) - {file_size:.1f}MB")
        else:
            print(f"    ✗ {filename} ({description}) - 未生成")
    
    # 可视化文件
    viz_files = [
        ("training_curves.png", "训练曲线"),
        ("prediction_results.png", "预测结果"),
        ("model_performance.png", "模型性能"),
        ("data_overview.png", "数据概览")
    ]
    
    print("  📊 可视化文件:")
    for filename, description in viz_files:
        if Path(filename).exists():
            print(f"    ✓ {filename} ({description})")
        else:
            print(f"    ✗ {filename} ({description}) - 未生成")
    
    # 输出目录
    output_dirs = ["models/", "results/", "logs/"]
    print("  📂 输出目录:")
    for dirname in output_dirs:
        dir_path = Path(dirname)
        if dir_path.exists():
            file_count = len(list(dir_path.glob("*")))
            print(f"    ✓ {dirname} ({file_count} 个文件)")
        else:
            print(f"    ✗ {dirname} - 不存在")
    
    # 缓存文件
    try:
        try:
            from hydrodataset import CACHE_DIR
        except ImportError:
            print("  💾 CAMELSH缓存: hydrodataset未安装，无法检查")
        else:
            cache_path = Path(CACHE_DIR)
            if cache_path.exists():
                cache_files = list(cache_path.glob("*camelsh*"))
                print(f"  💾 CAMELSH缓存: {len(cache_files)} 个文件 (位置: {cache_path})")
            else:
                print("  💾 CAMELSH缓存: 未找到")
    except Exception as e:
        print(f"  💾 CAMELSH缓存: 检查时出错 ({e})")
    
    if not args.test_only:
        print("\n📈 下一步建议:")
        print("  1. 查看训练曲线图 (training_curves.png) 评估模型收敛情况")
        print("  2. 检查预测结果图 (prediction_results.png) 评估模型性能")
        print("  3. 查看训练日志 (training_log.txt) 了解详细训练信息")
        print("  4. 根据需要调整超参数重新训练")
        print("  5. 使用训练好的模型进行新数据预测")
    
    print("\n📚 参考文档:")
    docs = [
        ("CAMELSH_使用说明.md", "CAMELSH数据集使用指南"),
        ("MULTI_TASK_README.md", "项目总体说明"),
        ("更改总结.md", "本次更新的详细说明"),
        ("test_camelsh_integration.py", "数据集成测试脚本")
    ]
    
    for filename, description in docs:
        if Path(filename).exists():
            print(f"  ✓ {filename} - {description}")
        else:
            print(f"  ✗ {filename} - {description} (文件不存在)")
    
    print("\n🔧 故障排除:")
    print("  如果遇到问题，请:")
    print("  1. 重新运行: python run_all.py --test-only")
    print("  2. 检查环境配置: python test_camelsh_integration.py")
    print("  3. 查看详细日志文件")
    print("  4. 参考CAMELSH_使用说明.md")
    
    print("=" * 80)

