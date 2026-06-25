# AGENTS.md

This file provides guidance to Codex (Codex.ai/code) when working with code in this repository.

## Project Overview

Multi-task LSTM model for jointly predicting streamflow (Q) and water level (h) from meteorological forcing and basin attributes. Uses the CAMELSH dataset (Chinese basins) with MSWEP 3-hour precipitation as an additional forcing source.

## 语言偏好

- 所有对话请使用简体中文回复。

## Setup

This project uses `uv` as the package manager.

```bash
# Install dependencies (CPU)
uv pip install -e .

# Install with local hydrodataset (required for CAMELSH access)
uv pip install -e .[hydro]
```

PyTorch GPU (CUDA 12.1) is configured via the `pytorch-gpu` index in `pyproject.toml`. The `hydrodataset` local path is set in `[tool.uv.sources]` — update it to match your machine (`D:/code/hydrodataset` by default).

## Running the Model

```bash
# Full pipeline (data prep → train → evaluate)
python run_all.py

# Step-by-step
python prepare_data.py            # generates flow_data.csv and waterlevel_data.csv
python train_multi_task_model.py  # trains model, saves to results/

# Quick environment check (2-epoch test run)
python quick_start.py

# Check GPU availability
python check_GPU.py

# Evaluate a trained model
python multi_task_evaluation.py

# Compare models
python model_comparison.py
```

## Configuration

All paths and hyperparameters are centralized in **`config.py`**:

- `CAMELSH_DATA_PATH` — path to CAMELSH dataset (default `F:/data`)
- `CAMELSUS_DATA_PATH` — path to CAMELS-US dataset
- `TRAIN_START/END`, `VALID_START/END`, `TEST_START/END` — fixed calendar splits (2001–2017 / 2018–2020 / 2021–2024), constrained to MSWEP availability
- `SEQUENCE_LENGTH` — 168 steps = 21 days at 3-hour resolution
- `FORCING_VARIABLES`, `ATTRIBUTE_VARIABLES` — lists of feature variables used

**Important split duality**: `multi_task_lstm.py` and `multi_task_lstm_wl2d.py` each define their own `TRAIN_RATIO/VALID_RATIO/TEST_RATIO` (60/20/20) at the top of the file. These scripts split each basin's own time range proportionally, independent of the calendar dates in `config.py`. Update both places when changing split logic.

**`WINDOW_STEP`** (default 3): sliding-window stride in time steps between sample start points. Smaller values increase sample count and memory usage.

## Architecture

### Data Flow

```
CAMELSH (hourly NC files)       → ImprovedCAMELSHReader → xr.Dataset (forcing)
MSWEP (CSV, 3h resolution)      → mswep_loader.py       → pd.DataFrame (precipitation)
CAMELSH attributes              → hydrodataset.Camelsh  → pd.DataFrame (static attrs)
flow_data.csv / waterlevel_data.csv                     → pd.DataFrame (targets Q, h)
USGS qualifiers (optional)      → qualifiers_fetcher/   → loss weights per timestep
```

All forcing is merged and resampled to 3-hour resolution via `merge_forcing_with_mswep()` before entering `MultiTaskDataset`. CAMELSH NC files are at hourly resolution; the reader resamples them to 3h. MSWEP CSVs are already at 3h.

### Key Classes / Functions (`multi_task_lstm.py`)

- **`MultiTaskDataset`** — PyTorch Dataset; builds sliding-window lookup tables per basin. Each sample is `(xc, y_flow, y_waterlevel, w_flow, w_waterlevel)` where `xc = [forcing | static_attrs]` concatenated along the feature axis. Normalization statistics are computed on the training split and passed (via `means`/`stds` dicts) to validation/test datasets. NaN in Q/h targets is allowed and skipped in loss; NaN in forcing/attrs raises an error.
- **`MultiTaskLSTM`** — shared LSTM encoder with two linear output heads (flow head, water-level head).
- **`train_epoch` / `eval_model`** — training and evaluation loops with optional qualifier-weighted NSE loss.

### `ImprovedCAMELSHReader` (`improved_camelsh_reader.py`)

Wraps `hydrodataset.Camelsh` with direct NC file access as a fallback. Reads from two directories:
- `CAMELSH/timeseries/Data/CAMELSH/timeseries/` — standard hourly forcing
- `CAMELSH/Hourly2/Hourly2/` — water level data

### Script Variants

| Script | Description |
|--------|-------------|
| `single_task_lstm - 早停+正则.py` | Single-task flow-only LSTM with early stopping and regularization |
| `multi_task_lstm_wl2d.py` | Multi-task WL2D LSTM (Q + h with wl→D cascade) |
| `multi_task_lstm_ablation_wl2d_repeat.py` | WL2D ablation with MCAR missing (multi-seed repeat) |
| `multi_task_lstm_ablation_realistic_missing.py` | WL2D ablation with segment-based missing from real fault ECDF |
| `multi_task_lstm.py` | [历史留档] Baseline multi-task LSTM (no early stopping) |
| `multi_task_lstm - 早停.py` | [历史留档] Multi-task LSTM with early stopping |
| `single_task_lstm.py` | [历史留档] Single-task flow-only LSTM (no early stopping) |
| `multi_task_lstm_ablation_*.py` | [历史留档] Other ablation study variants |

### USGS Qualifier Weights

Set `USE_QUALIFIER_WEIGHTS = True` in `multi_task_lstm.py` to weight the loss by USGS measurement quality flags. Weights are loaded from `qualifiers_output_full/camelsh_with_qualifiers.csv`. The fetcher scripts in `qualifiers_fetcher/` download these from the USGS NWIS API.

### MSWEP Data

Located in `MSWEP/`:
- `mswep_1000basins_mean_3hourly_1980_2024.csv` — 1000-basin dataset
- `mswep_220basins_mean_3hourly_1980_2024.csv` — 220-basin subset

Loaded via `mswep_loader.load_mswep_data()` and merged with CAMELSH forcing via `merge_forcing_with_mswep()`. The merge handles duplicate timestamps, basin reindexing, and time alignment automatically.

## Output Structure

```
results/
  models/    # saved .pth checkpoints
  images/    # per-basin prediction plots
  reports/   # markdown evaluation reports
  logs/      # training logs
```

Model checkpoints store both `model_state_dict` and normalization statistics (`means`, `stds`) so evaluation can reconstruct the exact same preprocessing.
