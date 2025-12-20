import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from multi_task_lstm import MultiTaskLSTM, MultiTaskDataset
from improved_camelsh_reader import ImprovedCAMELSHReader
from hydrodataset import StandardVariable

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def evaluate_multi_task_model_fixed():
    """修复后的多任务模型评估"""
    print("=== Fixed Multi-Task Model Evaluation ===")
    
    # Import configuration
    from config import (
        CAMELSH_DATA_PATH, 
        FORCING_VARIABLES, 
        ATTRIBUTE_VARIABLES, 
        VALID_WATER_LEVEL_BASINS
    )
    
    # Check model file
    model_path = "multi_task_lstm_model.pth"
    if not os.path.exists(model_path):
        print(f"Multi-task model file not found: {model_path}")
        return
    
    # Load checkpoint to get parameters
    checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
    
    # Infer hidden size from weights
    state_dict = checkpoint['model_state_dict']
    for key, tensor in state_dict.items():
        if 'fc_flow.weight' in key:
            HIDDEN_SIZE = tensor.shape[1]
            break
    else:
        HIDDEN_SIZE = 64  # default value
    
    print(f"Inferred hidden size from model file: {HIDDEN_SIZE}")
    SEQUENCE_LENGTH = 100
    NUM_BASINS = 4
    
    # Test data time range
    TEST_START_DATE = "2014-01-01"
    TEST_END_DATE = "2016-12-31"
    
    # Select basins
    chosen_basins = VALID_WATER_LEVEL_BASINS[:NUM_BASINS]
    print(f"Test basins: {chosen_basins}")
    
    # Initialize data reader
    camelsh_reader = ImprovedCAMELSHReader(CAMELSH_DATA_PATH, download=False)
    
    # Read data
    print("Reading test data...")
    attrs = camelsh_reader.read_attr_xrdataset(
        gage_id_lst=chosen_basins, 
        var_lst=ATTRIBUTE_VARIABLES
    )
    # Convert to DataFrame and add gauge_id column
    attrs_df = attrs.to_dataframe().reset_index()
    attrs_df = attrs_df.rename(columns={'basin': 'gauge_id'})
    
    forcing_data = camelsh_reader.read_ts_xrdataset(
        gage_id_lst=chosen_basins,
        var_lst=FORCING_VARIABLES,
        t_range=[TEST_START_DATE, TEST_END_DATE]
    )
    
    flow_data = camelsh_reader.read_ts_xrdataset(
        gage_id_lst=chosen_basins,
        var_lst=[StandardVariable.STREAMFLOW],
        t_range=[TEST_START_DATE, TEST_END_DATE]
    )
    flow_df = flow_data[StandardVariable.STREAMFLOW].to_dataframe().unstack('basin')[StandardVariable.STREAMFLOW]
    
    waterlevel_data = camelsh_reader.read_ts_xrdataset(
        gage_id_lst=chosen_basins,
        var_lst=[StandardVariable.WATER_LEVEL],
        t_range=[TEST_START_DATE, TEST_END_DATE]
    )
    waterlevel_df = waterlevel_data[StandardVariable.WATER_LEVEL].to_dataframe().unstack('basin')[StandardVariable.WATER_LEVEL]
    
    # Load model
    print("Loading multi-task model...")
    
    input_size = len(FORCING_VARIABLES) + len(ATTRIBUTE_VARIABLES)
    model = MultiTaskLSTM(
        input_size=input_size,
        hidden_size=HIDDEN_SIZE,
        dropout_rate=0.0
    ).to(DEVICE)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    # Create test dataset with correct normalization
    print("Creating test dataset with fixed normalization...")
    ds_test = MultiTaskDataset(
        basins=chosen_basins,
        dates=[TEST_START_DATE, TEST_END_DATE],
        data_attr=attrs_df,
        data_forcing=forcing_data,
        data_flow=flow_df,
        data_waterlevel=waterlevel_df,
        loader_type="test",
        seq_length=SEQUENCE_LENGTH,
        means=checkpoint['means'],
        stds=checkpoint['stds'],
    )
    
    # Evaluate model
    dataloader = DataLoader(ds_test, batch_size=256, shuffle=False, num_workers=0)
    
    preds_flow = []
    preds_waterlevel = []
    obs_flow = []
    obs_waterlevel = []
    
    print("Model prediction in progress...")
    with torch.no_grad():
        for xs, ys in tqdm(dataloader, desc="Evaluating fixed multi-task model"):
            xs = xs.to(DEVICE)
            pred_flow, pred_waterlevel = model(xs)
            
            preds_flow.append(pred_flow.cpu().numpy())
            preds_waterlevel.append(pred_waterlevel.cpu().numpy())
            obs_flow.append(ys[:, 0:1].numpy())
            obs_waterlevel.append(ys[:, 1:2].numpy())
    
    # Concatenate prediction results
    preds_flow = np.concatenate(preds_flow, axis=0)
    preds_waterlevel = np.concatenate(preds_waterlevel, axis=0)
    obs_flow = np.concatenate(obs_flow, axis=0)
    obs_waterlevel = np.concatenate(obs_waterlevel, axis=0)
    
    print(f"Prediction shapes - Flow: {preds_flow.shape}, Water Level: {preds_waterlevel.shape}")
    print(f"Observation shapes - Flow: {obs_flow.shape}, Water Level: {obs_waterlevel.shape}")
    
    # Denormalize predictions and observations
    print("Denormalizing data...")
    preds_flow_denorm = ds_test.local_denormalization(preds_flow, "flow")
    preds_waterlevel_denorm = ds_test.local_denormalization(preds_waterlevel, "waterlevel")
    obs_flow_denorm = ds_test.local_denormalization(obs_flow, "flow")
    obs_waterlevel_denorm = ds_test.local_denormalization(obs_waterlevel, "waterlevel")
    
    print(f"Denormalized ranges:")
    print(f"  Flow - Obs: [{obs_flow_denorm.min():.2f}, {obs_flow_denorm.max():.2f}], Pred: [{preds_flow_denorm.min():.2f}, {preds_flow_denorm.max():.2f}]")
    print(f"  Water Level - Obs: [{obs_waterlevel_denorm.min():.2f}, {obs_waterlevel_denorm.max():.2f}], Pred: [{preds_waterlevel_denorm.min():.2f}, {preds_waterlevel_denorm.max():.2f}]")
    
    # Calculate NSE
    def calculate_nse(obs, pred):
        """Calculate Nash-Sutcliffe Efficiency"""
        obs_flat = obs.flatten()
        pred_flat = pred.flatten()
        
        # Remove NaN values
        mask = ~(np.isnan(obs_flat) | np.isnan(pred_flat))
        obs_clean = obs_flat[mask]
        pred_clean = pred_flat[mask]
        
        if len(obs_clean) == 0:
            return np.nan
        
        numerator = np.sum((obs_clean - pred_clean) ** 2)
        denominator = np.sum((obs_clean - np.mean(obs_clean)) ** 2)
        
        if denominator == 0:
            return np.nan
        
        return 1 - (numerator / denominator)
    
    nse_flow = calculate_nse(obs_flow_denorm, preds_flow_denorm)
    nse_waterlevel = calculate_nse(obs_waterlevel_denorm, preds_waterlevel_denorm)
    
    print(f"\nFixed Multi-task model results:")
    print(f"Flow NSE: {nse_flow:.4f}")
    print(f"Water Level NSE: {nse_waterlevel:.4f}")
    
    # Calculate additional metrics
    def calculate_metrics(obs, pred):
        obs_flat = obs.flatten()
        pred_flat = pred.flatten()
        
        # Remove NaN values
        mask = ~(np.isnan(obs_flat) | np.isnan(pred_flat))
        obs_clean = obs_flat[mask]
        pred_clean = pred_flat[mask]
        
        if len(obs_clean) == 0:
            return {'rmse': np.nan, 'mae': np.nan, 'correlation': np.nan}
        
        rmse = np.sqrt(np.mean((obs_clean - pred_clean) ** 2))
        mae = np.mean(np.abs(obs_clean - pred_clean))
        correlation = np.corrcoef(obs_clean, pred_clean)[0, 1] if len(obs_clean) > 1 else np.nan
        
        return {'rmse': rmse, 'mae': mae, 'correlation': correlation}
    
    flow_metrics = calculate_metrics(obs_flow_denorm, preds_flow_denorm)
    wl_metrics = calculate_metrics(obs_waterlevel_denorm, preds_waterlevel_denorm)
    
    # Plot detailed results
    fig, axes = plt.subplots(3, 2, figsize=(16, 18))
    
    # Flow time series (first 2000 points)
    n_points = min(2000, len(obs_flow_denorm))
    axes[0, 0].plot(obs_flow_denorm[:n_points], label='Observed', alpha=0.8, linewidth=1)
    axes[0, 0].plot(preds_flow_denorm[:n_points], label='Predicted', alpha=0.8, linewidth=1)
    axes[0, 0].set_title(f'Flow Prediction Time Series (NSE: {nse_flow:.3f})')
    axes[0, 0].set_xlabel('Time Step')
    axes[0, 0].set_ylabel('Flow (m³/s)')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Water level time series (first 2000 points)
    axes[0, 1].plot(obs_waterlevel_denorm[:n_points], label='Observed', alpha=0.8, linewidth=1)
    axes[0, 1].plot(preds_waterlevel_denorm[:n_points], label='Predicted', alpha=0.8, linewidth=1)
    axes[0, 1].set_title(f'Water Level Prediction Time Series (NSE: {nse_waterlevel:.3f})')
    axes[0, 1].set_xlabel('Time Step')
    axes[0, 1].set_ylabel('Water Level (m)')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Flow scatter plot
    axes[1, 0].scatter(obs_flow_denorm, preds_flow_denorm, alpha=0.6, s=2)
    axes[1, 0].plot([obs_flow_denorm.min(), obs_flow_denorm.max()], 
                    [obs_flow_denorm.min(), obs_flow_denorm.max()], 'r--', linewidth=2)
    axes[1, 0].set_xlabel('Observed Flow (m³/s)')
    axes[1, 0].set_ylabel('Predicted Flow (m³/s)')
    axes[1, 0].set_title(f'Flow Scatter Plot (NSE: {nse_flow:.3f})')
    axes[1, 0].grid(True, alpha=0.3)
    
    # Water level scatter plot
    axes[1, 1].scatter(obs_waterlevel_denorm, preds_waterlevel_denorm, alpha=0.6, s=2)
    axes[1, 1].plot([obs_waterlevel_denorm.min(), obs_waterlevel_denorm.max()], 
                    [obs_waterlevel_denorm.min(), obs_waterlevel_denorm.max()], 'r--', linewidth=2)
    axes[1, 1].set_xlabel('Observed Water Level (m)')
    axes[1, 1].set_ylabel('Predicted Water Level (m)')
    axes[1, 1].set_title(f'Water Level Scatter Plot (NSE: {nse_waterlevel:.3f})')
    axes[1, 1].grid(True, alpha=0.3)
    
    # Flow residual plot
    residuals_flow = preds_flow_denorm.flatten() - obs_flow_denorm.flatten()
    axes[2, 0].scatter(obs_flow_denorm, residuals_flow, alpha=0.6, s=2)
    axes[2, 0].axhline(y=0, color='r', linestyle='--', linewidth=2)
    axes[2, 0].set_xlabel('Observed Flow (m³/s)')
    axes[2, 0].set_ylabel('Residuals (m³/s)')
    axes[2, 0].set_title('Flow Residual Plot')
    axes[2, 0].grid(True, alpha=0.3)
    
    # Water level residual plot
    residuals_waterlevel = preds_waterlevel_denorm.flatten() - obs_waterlevel_denorm.flatten()
    axes[2, 1].scatter(obs_waterlevel_denorm, residuals_waterlevel, alpha=0.6, s=2)
    axes[2, 1].axhline(y=0, color='r', linestyle='--', linewidth=2)
    axes[2, 1].set_xlabel('Observed Water Level (m)')
    axes[2, 1].set_ylabel('Residuals (m)')
    axes[2, 1].set_title('Water Level Residual Plot')
    axes[2, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('fixed_multi_task_results.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Print detailed statistical information
    print("\n=== Detailed Statistical Information (Fixed) ===")
    print(f"Flow Prediction:")
    print(f"  NSE: {nse_flow:.4f}")
    print(f"  RMSE: {flow_metrics['rmse']:.4f}")
    print(f"  MAE: {flow_metrics['mae']:.4f}")
    print(f"  Correlation: {flow_metrics['correlation']:.4f}")
    
    print(f"Water Level Prediction:")
    print(f"  NSE: {nse_waterlevel:.4f}")
    print(f"  RMSE: {wl_metrics['rmse']:.4f}")
    print(f"  MAE: {wl_metrics['mae']:.4f}")
    print(f"  Correlation: {wl_metrics['correlation']:.4f}")
    
    return {
        'flow_nse': nse_flow,
        'waterlevel_nse': nse_waterlevel,
        'obs_flow': obs_flow_denorm,
        'pred_flow': preds_flow_denorm,
        'obs_waterlevel': obs_waterlevel_denorm,
        'pred_waterlevel': preds_waterlevel_denorm
    }

if __name__ == "__main__":
    results = evaluate_multi_task_model_fixed()
