from config import CAMELSH_DATA_PATH
from improved_camelsh_reader import ImprovedCAMELSHReader


def main() -> None:
    """Check which attribute variables are actually available via Camelsh.read_attr_xrdataset."""
    reader = ImprovedCAMELSHReader(CAMELSH_DATA_PATH, download=False)
    camelsh = reader.camelsh

    basin_ids = camelsh.read_object_ids()
    if len(basin_ids) == 0:
        raise SystemExit("No basins found in CAMELSH.")

    test_basin = [str(basin_ids[0])]
    print(f"Using basin for test: {test_basin[0]}")

    vars_to_test = ["aridity_index", "snw_pc_syr", "SLOPE_PCT"]
    for v in vars_to_test:
        print(f"\n=== Testing variable: {v} ===")
        try:
            ds = camelsh.read_attr_xrdataset(gage_id_lst=test_basin, var_lst=[v])
            print("Success. Dataset variables:", list(ds.data_vars.keys()))
            print(ds)
        except Exception as e:
            print(f"Failed to read variable '{v}': {e!r}")


if __name__ == "__main__":
    main()


