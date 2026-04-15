from pathlib import Path
from typing import Set

import pandas as pd

from config import CAMELSH_DATA_PATH


def main() -> None:
    """List all available attribute variable names from the raw CAMELSH CSV files."""
    root = Path(CAMELSH_DATA_PATH) / "CAMELSH" / "attributes"

    if not root.exists():
        raise SystemExit(f"Attribute directory does not exist: {root}")

    csv_files = sorted(root.glob("*.csv"))
    if not csv_files:
        raise SystemExit(f"No CSV attribute files found under: {root}")

    print("Found attribute CSV files:")
    for f in csv_files:
        print(" -", f.name)

    all_columns: Set[str] = set()

    print("\nPer-file columns (from headers only):")
    for f in csv_files:
        # Only read headers to avoid loading full data
        df = pd.read_csv(f, nrows=0)
        cols = list(df.columns)
        print(f"File {f.name}:")
        print("  ", ", ".join(cols))
        all_columns.update(cols)

    print("\nUnion of all attribute column names:")
    for name in sorted(all_columns):
        print(name)


if __name__ == "__main__":
    main()



