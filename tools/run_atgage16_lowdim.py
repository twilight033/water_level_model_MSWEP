"""一键运行16站H与H+A低维对照；--estimator区分岭回归和神经网络。"""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from cross_section_experiment.area_lowdim16 import main


if __name__ == "__main__":
    raise SystemExit(main())
