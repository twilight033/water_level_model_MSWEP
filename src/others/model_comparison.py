import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import HydroErr as he

# 设置中文字体
font_path = "D:/code/TimesSong.ttf"
if os.path.exists(font_path):
    # 直接使用字体文件路径
    plt.rcParams['font.sans-serif'] = ['TimesSong']
    plt.rcParams['axes.unicode_minus'] = False
    # 注册字体
    fm.fontManager.addfont(font_path)
    print(f"使用字体: {font_path}")
else:
    # 使用系统中文字体
    plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False
    print("使用系统中文字体")

from multi_task_lstm import MultiTaskLSTM, MultiTaskDataset
from single_task_lstm import SingleTaskLSTM, SingleTaskDataset
from improved_camelsh_reader import ImprovedCAMELSHReader
from hydrodataset import StandardVariable

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def load_single_task_model(model_path, input_size, hidden_size):
    """加载单任务模型"""
    checkpoint = torch.load(model_path, map_location=DEVICE, weights_only=False)
    
    model = SingleTaskLSTM(
        input_size=input_size,
        hidden_size=hidden_size,
        dropout_rate=0.0  # 评估时不使用dropout
    ).to(DEVICE)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    return model, checkpoint['means'], checkpoint['stds'], checkpoint['nse_test']


def load_multi_task_model(model_path, input_size, hidden_size):
    """加载多任务模型"""
    checkpoint = torch.load(model_path, map_location=DEVICE, weights_only=False)
    
    model = MultiTaskLSTM(
        input_size=input_size,
        hidden_size=hidden_size,
        dropout_rate=0.0  # 评估时不使用dropout
    ).to(DEVICE)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    return model, checkpoint['means'], checkpoint['stds']


def evaluate_single_task_model(model, dataset, target_type):
    """评估单任务模型"""
    dataloader = DataLoader(dataset, batch_size=256, shuffle=False, num_workers=0)
    
    preds = []
    obs = []
    
    model.eval()
    with torch.no_grad():
        for xs, ys in tqdm(dataloader, desc=f"评估{target_type}模型"):
            xs = xs.to(DEVICE)
            pred = model(xs)
            
            preds.append(pred.cpu().numpy())
            obs.append(ys.numpy())
    
    preds = np.concatenate(preds, axis=0)
    obs = np.concatenate(obs, axis=0)
    
    # 反归一化预测值
    preds_denorm = dataset.local_denormalization(preds)
    
    # 获取原始观测值（如果是测试模式）
    if hasattr(dataset, 'get_raw_targets'):
        raw_targets = dataset.get_raw_targets()
        # 重新构建观测值数组，使用原始数据
        obs_denorm = []
        for xs, ys in DataLoader(dataset, batch_size=256, shuffle=False, num_workers=0):
            for i, (x, y) in enumerate(zip(xs, ys)):
                basin, time_idx = dataset.lookup_table[len(obs_denorm)]
                obs_denorm.append(raw_targets[basin][time_idx + dataset.seq_length])
        obs_denorm = np.array(obs_denorm).reshape(-1, 1)
    else:
        # 备用方案：使用反归一化
        obs_denorm = dataset.local_denormalization(obs)
    
    # 计算NSE
    try:
        nse = he.evaluator(he.nse, obs_denorm.flatten(), preds_denorm.flatten())[0]
    except:
        # 备用NSE计算
        def calculate_nse(obs, sim):
            numerator = np.sum((obs - sim) ** 2)
            denominator = np.sum((obs - np.mean(obs)) ** 2)
            return 1 - (numerator / denominator)
        nse = calculate_nse(obs_denorm.flatten(), preds_denorm.flatten())
    
    return obs_denorm, preds_denorm, nse


def load_single_task_model(model_path, input_size, hidden_size=None):
    """加载单任务模型"""
    checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
    
    # 从模型状态字典推断隐藏层大小
    if hidden_size is None:
        # 从fc层的输入维度推断隐藏层大小
        fc_weight_shape = checkpoint['model_state_dict']['fc.weight'].shape
        inferred_hidden_size = fc_weight_shape[1]  # fc层的输入维度
        print(f"从模型文件推断隐藏层大小: {inferred_hidden_size}")
    else:
        inferred_hidden_size = hidden_size
    
    model = SingleTaskLSTM(input_size=input_size, hidden_size=inferred_hidden_size)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    means = checkpoint['means']
    stds = checkpoint['stds']
    nse_test = checkpoint.get('nse_test', None)
    
    return model, means, stds, nse_test


def load_multi_task_model(model_path, input_size, hidden_size):
    """加载多任务模型"""
    checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
    
    model = MultiTaskLSTM(input_size=input_size, hidden_size=hidden_size)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    means = checkpoint['means']
    stds = checkpoint['stds']
    
    return model, means, stds


def evaluate_multi_task_model(model, dataset):
    """评估多任务模型"""
    dataloader = DataLoader(dataset, batch_size=256, shuffle=False, num_workers=0)
    
    preds_flow = []
    preds_waterlevel = []
    obs_flow = []
    obs_waterlevel = []
    
    model.eval()
    with torch.no_grad():
        for xs, ys in tqdm(dataloader, desc="评估多任务模型"):
            xs = xs.to(DEVICE)
            pred_flow, pred_waterlevel = model(xs)
            
            preds_flow.append(pred_flow.cpu().numpy())
            preds_waterlevel.append(pred_waterlevel.cpu().numpy())
            obs_flow.append(ys[:, 0:1].numpy())
            obs_waterlevel.append(ys[:, 1:2].numpy())
    
    preds_flow = np.concatenate(preds_flow, axis=0)
    preds_waterlevel = np.concatenate(preds_waterlevel, axis=0)
    obs_flow = np.concatenate(obs_flow, axis=0)
    obs_waterlevel = np.concatenate(obs_waterlevel, axis=0)
    
    # 反归一化
    preds_flow_denorm = dataset.local_denormalization(preds_flow, "flow")
    preds_waterlevel_denorm = dataset.local_denormalization(preds_waterlevel, "waterlevel")
    obs_flow_denorm = dataset.local_denormalization(obs_flow, "flow")
    obs_waterlevel_denorm = dataset.local_denormalization(obs_waterlevel, "waterlevel")
    
    # 计算NSE
    try:
        nse_flow = he.evaluator(he.nse, obs_flow_denorm.flatten(), preds_flow_denorm.flatten())[0]
        nse_waterlevel = he.evaluator(he.nse, obs_waterlevel_denorm.flatten(), preds_waterlevel_denorm.flatten())[0]
    except:
        # 备用NSE计算
        def calculate_nse(obs, sim):
            numerator = np.sum((obs - sim) ** 2)
            denominator = np.sum((obs - np.mean(obs)) ** 2)
            return 1 - (numerator / denominator)
        nse_flow = calculate_nse(obs_flow_denorm.flatten(), preds_flow_denorm.flatten())
        nse_waterlevel = calculate_nse(obs_waterlevel_denorm.flatten(), preds_waterlevel_denorm.flatten())
    
    return (obs_flow_denorm, preds_flow_denorm, nse_flow), \
           (obs_waterlevel_denorm, preds_waterlevel_denorm, nse_waterlevel)


def compare_models():
    """比较多任务模型和单任务模型的性能"""
    print("=== 模型性能对比 ===")
    
    # 导入配置
    from config import (
        CAMELSH_DATA_PATH, 
        FORCING_VARIABLES, 
        ATTRIBUTE_VARIABLES, 
        VALID_WATER_LEVEL_BASINS
    )
    
    # 参数设置
    HIDDEN_SIZE = 64  # 与现有多任务模型保持一致
    SEQUENCE_LENGTH = 100
    NUM_BASINS = 2  # 与单任务模型训练时保持一致
    
    # 测试数据时间范围
    TEST_START_DATE = "2014-01-01"
    TEST_END_DATE = "2016-12-31"
    
    # 选择流域 - 使用与训练时相同的流域数量和顺序
    # 多任务模型使用了4个流域，单任务模型也应该使用相同的流域
    chosen_basins = VALID_WATER_LEVEL_BASINS[:4]  # 使用4个流域，与训练时一致
    print(f"测试流域: {chosen_basins}")
    
    # 确保使用相同的属性变量数量
    print(f"使用的属性变量: {ATTRIBUTE_VARIABLES}")
    print(f"属性变量数量: {len(ATTRIBUTE_VARIABLES)}")
    print(f"预期输入维度: {len(FORCING_VARIABLES)} (强迫) + {len(ATTRIBUTE_VARIABLES)} (属性) = {len(FORCING_VARIABLES) + len(ATTRIBUTE_VARIABLES)}")
    
    # 模型文件路径
    model_files = {
        'multi_task': 'multi_task_lstm_model.pth',
        'flow': 'single_task_flow_model.pth',
        'waterlevel': 'single_task_waterlevel_model.pth'
    }
    
    # 检查模型文件中保存的流域信息
    if os.path.exists(model_files['flow']):
        checkpoint = torch.load(model_files['flow'], map_location='cpu', weights_only=False)
        print(f"流量模型训练时使用的流域数量: 从模型文件推断")
    
    # 初始化数据读取器
    camelsh_reader = ImprovedCAMELSHReader(CAMELSH_DATA_PATH, download=False)
    
    # 读取数据
    print("读取测试数据...")
    attrs = camelsh_reader.read_attr_xrdataset(
        gage_id_lst=chosen_basins, 
        var_lst=ATTRIBUTE_VARIABLES
    )
    attrs_df = attrs.to_dataframe().reset_index()
    # 确保有gauge_id列用于MultiTaskDataset
    if 'basin' in attrs_df.columns:
        attrs_df['gauge_id'] = attrs_df['basin']
        attrs_df = attrs_df.set_index('basin')
    
    # 为单任务模型创建只有数值列的版本
    attrs_df_numeric = attrs_df.select_dtypes(include=[np.number])
    
    # 为多任务模型保留gauge_id列，但只包含数值属性
    attrs_df_multi = attrs_df_numeric.reset_index()
    if 'gauge_id' not in attrs_df_multi.columns and 'basin' in attrs_df_multi.columns:
        attrs_df_multi['gauge_id'] = attrs_df_multi['basin']
        # 移除basin列，只保留数值列和gauge_id
        attrs_df_multi = attrs_df_multi.drop('basin', axis=1)
    
    print(f"属性数据最终形状: {attrs_df.shape}")
    print(f"属性数据列: {list(attrs_df.columns)}")
    
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
    
    input_size = len(FORCING_VARIABLES) + len(ATTRIBUTE_VARIABLES)
    
    # 加载模型
    print("加载模型...")
    
    # 检查模型文件是否存在
    model_files = {
        'flow': 'single_task_flow_model.pth',
        'waterlevel': 'single_task_waterlevel_model.pth',
        'multi_task': 'multi_task_lstm_model.pth'
    }
    
    results = {}
    
    # 1. 评估单任务流量模型
    if os.path.exists(model_files['flow']):
        print("评估单任务流量模型...")
        flow_model, flow_means, flow_stds, _ = load_single_task_model(
            model_files['flow'], input_size
        )
        
        # 创建测试数据集（使用保存的统计量）
        ds_flow_test = SingleTaskDataset(
            basins=chosen_basins,
            dates=[TEST_START_DATE, TEST_END_DATE],
            data_attr=attrs_df_numeric,
            data_forcing=forcing_data,
            data_target=flow_df,
            target_type="flow",
            loader_type="test",
            seq_length=SEQUENCE_LENGTH,
            means=flow_means,
            stds=flow_stds,
        )
        
        print(f"使用的归一化参数:")
        print(f"  流量均值: {flow_means['flow']:.6f}")
        print(f"  流量标准差: {flow_stds['flow']:.6f}")
        
        obs_flow_single, pred_flow_single, nse_flow_single = evaluate_single_task_model(
            flow_model, ds_flow_test, "flow"
        )
        results['single_flow'] = {
            'obs': obs_flow_single,
            'pred': pred_flow_single,
            'nse': nse_flow_single
        }
        print(f"单任务流量模型NSE: {nse_flow_single:.4f}")
    else:
        print(f"未找到单任务流量模型文件: {model_files['flow']}")
    
    # 2. 评估单任务水位模型
    if os.path.exists(model_files['waterlevel']):
        print("评估单任务水位模型...")
        wl_model, wl_means, wl_stds, _ = load_single_task_model(
            model_files['waterlevel'], input_size, HIDDEN_SIZE
        )
        
        # 创建测试数据集
        ds_wl_test = SingleTaskDataset(
            basins=chosen_basins,
            dates=[TEST_START_DATE, TEST_END_DATE],
            data_attr=attrs_df_numeric,
            data_forcing=forcing_data,
            data_target=waterlevel_df,
            target_type="waterlevel",
            loader_type="test",
            seq_length=SEQUENCE_LENGTH,
            means=wl_means,
            stds=wl_stds,
        )
        
        obs_wl_single, pred_wl_single, nse_wl_single = evaluate_single_task_model(
            wl_model, ds_wl_test, "waterlevel"
        )
        results['single_waterlevel'] = {
            'obs': obs_wl_single,
            'pred': pred_wl_single,
            'nse': nse_wl_single
        }
        print(f"单任务水位模型NSE: {nse_wl_single:.4f}")
    else:
        print(f"未找到单任务水位模型文件: {model_files['waterlevel']}")
    
    # 3. 评估多任务模型
    if os.path.exists(model_files['multi_task']):
        print("评估多任务模型...")
        multi_model, multi_means, multi_stds = load_multi_task_model(
            model_files['multi_task'], input_size, HIDDEN_SIZE
        )
        
        # 创建测试数据集
        ds_multi_test = MultiTaskDataset(
            basins=chosen_basins,
            dates=[TEST_START_DATE, TEST_END_DATE],
            data_attr=attrs_df_multi,
            data_forcing=forcing_data,
            data_flow=flow_df,
            data_waterlevel=waterlevel_df,
            loader_type="test",
            seq_length=SEQUENCE_LENGTH,
            means=multi_means,
            stds=multi_stds,
        )
        
        (obs_flow_multi, pred_flow_multi, nse_flow_multi), \
        (obs_wl_multi, pred_wl_multi, nse_wl_multi) = evaluate_multi_task_model(
            multi_model, ds_multi_test
        )
        
        results['multi_flow'] = {
            'obs': obs_flow_multi,
            'pred': pred_flow_multi,
            'nse': nse_flow_multi
        }
        results['multi_waterlevel'] = {
            'obs': obs_wl_multi,
            'pred': pred_wl_multi,
            'nse': nse_wl_multi
        }
        print(f"多任务模型流量NSE: {nse_flow_multi:.4f}")
        print(f"多任务模型水位NSE: {nse_wl_multi:.4f}")
    else:
        print(f"未找到多任务模型文件: {model_files['multi_task']}")
    
    # 绘制对比结果
    if results:
        plot_comparison_results(results)
    
    # 打印性能总结
    print("\n=== 性能对比总结 ===")
    if 'single_flow' in results and 'multi_flow' in results:
        print(f"流量预测:")
        print(f"  单任务模型NSE: {results['single_flow']['nse']:.4f}")
        print(f"  多任务模型NSE: {results['multi_flow']['nse']:.4f}")
        improvement_flow = results['multi_flow']['nse'] - results['single_flow']['nse']
        print(f"  多任务改进: {improvement_flow:+.4f}")
    
    if 'single_waterlevel' in results and 'multi_waterlevel' in results:
        print(f"水位预测:")
        print(f"  单任务模型NSE: {results['single_waterlevel']['nse']:.4f}")
        print(f"  多任务模型NSE: {results['multi_waterlevel']['nse']:.4f}")
        improvement_wl = results['multi_waterlevel']['nse'] - results['single_waterlevel']['nse']
        print(f"  多任务改进: {improvement_wl:+.4f}")
    
    return results


def plot_comparison_results(results):
    """绘制对比结果"""
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # 流量对比
    if 'single_flow' in results and 'multi_flow' in results:
        # 时间序列对比（前1000个点）
        n_points = min(1000, len(results['single_flow']['obs']))
        
        ax = axes[0, 0]
        ax.plot(results['single_flow']['obs'][:n_points], label='观测值', alpha=0.7, color='black')
        ax.plot(results['single_flow']['pred'][:n_points], label=f"单任务 (NSE: {results['single_flow']['nse']:.3f})", alpha=0.7)
        ax.plot(results['multi_flow']['pred'][:n_points], label=f"多任务 (NSE: {results['multi_flow']['nse']:.3f})", alpha=0.7)
        ax.set_title('流量预测时间序列对比')
        ax.set_xlabel('时间步')
        ax.set_ylabel('流量')
        ax.legend()
        ax.grid(True)
        
        # 散点图对比 - 单任务
        ax = axes[0, 1]
        ax.scatter(results['single_flow']['obs'], results['single_flow']['pred'], alpha=0.5, s=1)
        ax.plot([results['single_flow']['obs'].min(), results['single_flow']['obs'].max()], 
                [results['single_flow']['obs'].min(), results['single_flow']['obs'].max()], 'r--')
        ax.set_xlabel('观测流量')
        ax.set_ylabel('预测流量')
        ax.set_title(f'单任务流量模型 (NSE: {results["single_flow"]["nse"]:.3f})')
        ax.grid(True)
        
        # 散点图对比 - 多任务
        ax = axes[0, 2]
        ax.scatter(results['multi_flow']['obs'], results['multi_flow']['pred'], alpha=0.5, s=1)
        ax.plot([results['multi_flow']['obs'].min(), results['multi_flow']['obs'].max()], 
                [results['multi_flow']['obs'].min(), results['multi_flow']['obs'].max()], 'r--')
        ax.set_xlabel('观测流量')
        ax.set_ylabel('预测流量')
        ax.set_title(f'多任务流量模型 (NSE: {results["multi_flow"]["nse"]:.3f})')
        ax.grid(True)
    
    # 水位对比
    if 'single_waterlevel' in results and 'multi_waterlevel' in results:
        # 时间序列对比（前1000个点）
        n_points = min(1000, len(results['single_waterlevel']['obs']))
        
        ax = axes[1, 0]
        ax.plot(results['single_waterlevel']['obs'][:n_points], label='观测值', alpha=0.7, color='black')
        ax.plot(results['single_waterlevel']['pred'][:n_points], label=f"单任务 (NSE: {results['single_waterlevel']['nse']:.3f})", alpha=0.7)
        ax.plot(results['multi_waterlevel']['pred'][:n_points], label=f"多任务 (NSE: {results['multi_waterlevel']['nse']:.3f})", alpha=0.7)
        ax.set_title('水位预测时间序列对比')
        ax.set_xlabel('时间步')
        ax.set_ylabel('水位')
        ax.legend()
        ax.grid(True)
        
        # 散点图对比 - 单任务
        ax = axes[1, 1]
        ax.scatter(results['single_waterlevel']['obs'], results['single_waterlevel']['pred'], alpha=0.5, s=1)
        ax.plot([results['single_waterlevel']['obs'].min(), results['single_waterlevel']['obs'].max()], 
                [results['single_waterlevel']['obs'].min(), results['single_waterlevel']['obs'].max()], 'r--')
        ax.set_xlabel('观测水位')
        ax.set_ylabel('预测水位')
        ax.set_title(f'单任务水位模型 (NSE: {results["single_waterlevel"]["nse"]:.3f})')
        ax.grid(True)
        
        # 散点图对比 - 多任务
        ax = axes[1, 2]
        ax.scatter(results['multi_waterlevel']['obs'], results['multi_waterlevel']['pred'], alpha=0.5, s=1)
        ax.plot([results['multi_waterlevel']['obs'].min(), results['multi_waterlevel']['obs'].max()], 
                [results['multi_waterlevel']['obs'].min(), results['multi_waterlevel']['obs'].max()], 'r--')
        ax.set_xlabel('观测水位')
        ax.set_ylabel('预测水位')
        ax.set_title(f'多任务水位模型 (NSE: {results["multi_waterlevel"]["nse"]:.3f})')
        ax.grid(True)
    
    plt.tight_layout()
    plt.savefig('model_comparison_results.png', dpi=300, bbox_inches='tight')
    plt.show()


if __name__ == "__main__":
    results = compare_models()
