from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def main() -> None:
    """
    Plot scatter correlations between NSE (flow/stage) and three attributes:
    - p_mean
    - p_seasonality
    - frac_snow

    This script reads `basin_ids_with_attrs.csv` in the project root and
    generates six figures (flow vs each attribute, stage vs each attribute),
    only including basins where the corresponding NSE (flow or stage) is
    non‑negative.
    """
    project_root = Path(__file__).parent
    csv_path = project_root / "basin_ids_with_attrs.csv"
    if not csv_path.exists():
        raise SystemExit(f"Cannot find basin_ids_with_attrs.csv at: {csv_path}")

    df = pd.read_csv(csv_path)

    # Ensure numeric dtypes (ignore errors due to possible empty cells)
    num_cols = ["flow", "stage", "p_mean", "p_seasonality", "frac_snow"]
    for col in num_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Output directory for figures
    out_dir = project_root / "results"
    out_dir.mkdir(parents=True, exist_ok=True)

    attr_cols = ["p_mean", "p_seasonality", "frac_snow"]

    # Helper to create a single scatter plot
    def plot_scatter(
        x: pd.Series,
        y: pd.Series,
        x_label: str,
        y_label: str,
        title_prefix: str,
        out_name: str,
    ) -> None:
        # Drop NaNs pairwise
        valid = x.notna() & y.notna()
        x_valid = x[valid]
        y_valid = y[valid]

        if len(x_valid) == 0:
            print(f"Skip {out_name}: no valid data after filtering.")
            return

        # Compute Pearson correlation
        if len(x_valid) > 1:
            r = np.corrcoef(x_valid, y_valid)[0, 1]
        else:
            r = np.nan

        fig, ax = plt.subplots(figsize=(6, 5))
        ax.scatter(x_valid, y_valid, alpha=0.7, edgecolor="k")
        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)

        if not np.isnan(r):
            title = f"{title_prefix} (r = {r:.2f}, n = {len(x_valid)})"
        else:
            title = f"{title_prefix} (n = {len(x_valid)})"
        ax.set_title(title)
        ax.grid(True, linestyle="--", alpha=0.3)

        out_path = out_dir / out_name
        fig.tight_layout()
        fig.savefig(out_path, dpi=200)
        plt.close(fig)

        print(f"Saved figure: {out_path}")

    # 1) flow vs attributes (only basins with NSE_flow >= 0)
    if "flow" in df.columns:
        df_flow = df[df["flow"] >= 0].copy()
        for attr in attr_cols:
            if attr not in df_flow.columns:
                continue
            plot_scatter(
                x=df_flow[attr],
                y=df_flow["flow"],
                x_label=attr,
                y_label="flow NSE",
                title_prefix=f"Flow NSE vs {attr}",
                out_name=f"scatter_flow_vs_{attr}.png",
            )
    else:
        print("Column 'flow' not found in CSV; skip flow plots.")

    # 2) stage vs attributes (only basins with NSE_stage >= 0)
    if "stage" in df.columns:
        df_stage = df[df["stage"] >= 0].copy()
        for attr in attr_cols:
            if attr not in df_stage.columns:
                continue
            plot_scatter(
                x=df_stage[attr],
                y=df_stage["stage"],
                x_label=attr,
                y_label="stage NSE",
                title_prefix=f"Stage NSE vs {attr}",
                out_name=f"scatter_stage_vs_{attr}.png",
            )
    else:
        print("Column 'stage' not found in CSV; skip stage plots.")


if __name__ == "__main__":
    main()


