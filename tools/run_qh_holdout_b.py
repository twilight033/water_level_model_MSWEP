"""顺序运行B及按Q选模B/C的入口；不带--run时仅核对已有C记录。"""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from training.qh_holdout_b import main


if __name__ == "__main__":
    raise SystemExit(main())
