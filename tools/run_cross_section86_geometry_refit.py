"""重拟合原86阶段1：冻结H预测，使用测量几何尺度并保留完整对照。"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from cross_section_experiment.survey_refit import main


if __name__ == "__main__":
    raise SystemExit(main())
