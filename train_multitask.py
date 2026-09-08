"""双头多任务模型入口；复用现有主实验和缺失实验训练流程。"""

import argparse
from datetime import datetime
from importlib import import_module
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parent
EXPERIMENT_MODULES = {
    "main": "multi_task_lstm_wl2d",
    "mcar": "multi_task_lstm_ablation_wl2d_repeat",
    "segment": "multi_task_lstm_ablation_realistic_missing",
}


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="共享 LSTM 编码器 + Q / 水位两个独立线性预测头",
        epilog="当前复用的旧数据流程仍需完成划分、归一化和评估修复，"
               "才能用于论文正式重跑。",
    )
    parser.add_argument("--experiment", choices=EXPERIMENT_MODULES, default="main")
    parser.add_argument("--model-seed", type=int, default=1234)
    parser.add_argument("--output-dir", type=Path,
                        help="本次运行的输出目录，必须为新目录")
    args = parser.parse_args(argv)

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    output_root = (args.output_dir or PROJECT_ROOT / "results" / "parallel"
                   / args.experiment / f"seed{args.model_seed}_{run_id}").resolve()
    if output_root.exists():
        parser.error(f"输出目录已经存在，请使用新目录: {output_root}")

    # 旧脚本依赖 src/others 中的读取器；配置明确使用项目根目录版本。
    for location in (PROJECT_ROOT / "src" / "others",
                     PROJECT_ROOT / "src" / "training", PROJECT_ROOT):
        sys.path.insert(0, str(location))
    experiment = import_module(EXPERIMENT_MODULES[args.experiment])
    print("模型：共享 LSTM + Q / 水位两个独立预测头")
    print(f"输出目录：{output_root}")
    print("说明：本次仅切换架构，旧数据流程尚未完成审稿要求的修复。")
    experiment.main(architecture="parallel", output_root=output_root,
                    model_seed=args.model_seed)


if __name__ == "__main__":
    main()
