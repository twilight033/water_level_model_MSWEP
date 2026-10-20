"""一键运行原86阶段1测站处16站面积实验；--preflight只检查，不训练。"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from cross_section_experiment.area_stage1 import main


if __name__ == "__main__":
    raise SystemExit(main())
