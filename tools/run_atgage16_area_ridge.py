"""一键在CPU运行16站的有面积、无面积岭回归对照。"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from cross_section_experiment.area_ridge16 import main


if __name__ == "__main__":
    raise SystemExit(main())
