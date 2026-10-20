"""一键运行仅16站参与拟合的面积阶段1；--preflight只检查，不训练。"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from cross_section_experiment.area_stage1_only16 import main


if __name__ == "__main__":
    raise SystemExit(main())
