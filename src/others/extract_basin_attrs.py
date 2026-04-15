from pathlib import Path

import pandas as pd

from config import CAMELSH_DATA_PATH
from improved_camelsh_reader import ImprovedCAMELSHReader


def main() -> None:
    """
    Extract selected attribute variables for the basins listed in basin_ids.csv.

    Attributes to extract (from config.py, ATTRIBUTE_VARIABLES lines 79–81):
      - p_mean
      - p_seasonality
      - frac_snow
    """
    project_root = Path(__file__).parent
    basin_ids_path = project_root / "basin_ids.csv"

    if not basin_ids_path.exists():
        raise SystemExit(f"Cannot find basin_ids.csv at: {basin_ids_path}")

    # Read basin list (as strings to preserve leading zeros)
    basin_df = pd.read_csv(basin_ids_path, dtype={"basin_id": str})
    basin_ids = basin_df["basin_id"].astype(str).tolist()

    print(f"Loaded {len(basin_ids)} basin ids from {basin_ids_path}")

    # Initialise CAMELSH reader
    reader = ImprovedCAMELSHReader(CAMELSH_DATA_PATH, download=False)

    # Intersect requested basins with those actually available in CAMELSH
    available_basins = set(reader.camelsh.read_object_ids().astype(str).tolist())
    filtered_basin_ids = [b for b in basin_ids if b in available_basins]

    if not filtered_basin_ids:
        raise SystemExit("None of the basin_ids from basin_ids.csv are present in CAMELSH attributes.")

    missing = sorted(set(basin_ids) - available_basins)
    if missing:
        print(f"Warning: {len(missing)} basin ids are not present in CAMELSH and will be skipped.")
        print("Missing examples:", missing[:10])

    # Attributes to extract
    attr_vars = ["p_mean", "p_seasonality", "frac_snow"]
    print(f"Requesting attribute variables: {attr_vars}")

    attrs = reader.read_attr_xrdataset(gage_id_lst=filtered_basin_ids, var_lst=attr_vars)

    # Convert to DataFrame: basin as a column, attributes as columns
    attrs_df = attrs.to_dataframe().reset_index()

    # Standardise basin id column name
    if "basin" in attrs_df.columns:
        attrs_df.rename(columns={"basin": "basin_id"}, inplace=True)

    # Keep only unique rows per basin_id and selected attributes
    keep_cols = ["basin_id"] + attr_vars
    attrs_df = attrs_df[keep_cols].drop_duplicates(subset=["basin_id"])

    print("Extracted attributes for basins:")
    print(attrs_df.head())

    # Merge back to original basin_ids.csv (left join to keep original order/metrics)
    merged = basin_df.merge(attrs_df, on="basin_id", how="left")

    out_path = project_root / "basin_ids_with_attrs.csv"
    merged.to_csv(out_path, index=False)

    print(f"\nSaved merged attributes to: {out_path}")


if __name__ == "__main__":
    main()


