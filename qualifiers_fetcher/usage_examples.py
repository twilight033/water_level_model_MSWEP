"""
使用示例 - 如何在训练中使用qualifiers数据
"""

import pandas as pd
import numpy as np


# ==================== 示例1: 基于权重过滤数据 ====================

def filter_by_quality(data_file: str, min_weight: float = 0.7) -> pd.DataFrame:
    """
    过滤低质量数据
    
    Parameters
    ----------
    data_file : str
        包含qualifiers的数据文件
    min_weight : float
        最小权重阈值
    
    Returns
    -------
    pd.DataFrame
        过滤后的数据
    """
    df = pd.read_csv(data_file)
    
    # 过滤低质量数据
    df_filtered = df[
        (df['Q_weight'] >= min_weight) & 
        (df['H_weight'] >= min_weight)
    ].copy()
    
    print(f"原始数据: {len(df)} 行")
    print(f"过滤后: {len(df_filtered)} 行 ({len(df_filtered)/len(df)*100:.1f}%)")
    
    return df_filtered


# ==================== 示例2: 在PyTorch DataLoader中使用权重 ====================

def create_weighted_sampler(df: pd.DataFrame, weight_column: str = 'Q_weight'):
    """
    创建基于权重的采样器
    
    Parameters
    ----------
    df : pd.DataFrame
        包含权重列的数据
    weight_column : str
        权重列名
    
    Returns
    -------
    torch.utils.data.WeightedRandomSampler
    """
    import torch
    from torch.utils.data import WeightedRandomSampler
    
    weights = df[weight_column].fillna(0).values
    
    # 确保权重为正
    weights = np.maximum(weights, 0.01)
    
    sampler = WeightedRandomSampler(
        weights=weights,
        num_samples=len(weights),
        replacement=True
    )
    
    return sampler


# ==================== 示例3: 在损失函数中使用权重 ====================

def weighted_mse_loss(predictions, targets, weights):
    """
    带权重的MSE损失
    
    Parameters
    ----------
    predictions : torch.Tensor
        预测值
    targets : torch.Tensor
        目标值
    weights : torch.Tensor
        样本权重
    
    Returns
    -------
    torch.Tensor
        加权损失
    """
    import torch
    
    # 计算平方误差
    squared_errors = (predictions - targets) ** 2
    
    # 应用权重
    weighted_errors = squared_errors * weights
    
    # 返回平均损失
    return weighted_errors.sum() / weights.sum()


# 使用示例
"""
# 在训练循环中
for xs, ys, weights in dataloader:
    predictions = model(xs)
    loss = weighted_mse_loss(predictions, ys, weights)
    loss.backward()
    optimizer.step()
"""


# ==================== 示例4: 根据qualifiers分组分析 ====================

def analyze_by_qualifier(data_file: str):
    """
    按qualifier类型分析模型性能
    """
    df = pd.read_csv(data_file)
    
    # 添加模型预测列（示例）
    # df['Q_pred'] = ...  # 从模型预测结果加载
    
    # 按qualifier分组
    print("\n径流数据质量分布:")
    q_flag_stats = df.groupby('Q_flag').agg({
        'Q': 'count',
        'Q_weight': 'mean'
    })
    print(q_flag_stats)
    
    print("\n水位数据质量分布:")
    h_flag_stats = df.groupby('H_flag').agg({
        'H': 'count',
        'H_weight': 'mean'
    })
    print(h_flag_stats)


# ==================== 示例5: 创建自定义Dataset类 ====================

class WeightedHydroDataset(torch.utils.data.Dataset):
    """
    支持样本权重的水文数据集
    """
    
    def __init__(self, data_file: str, use_weights: bool = True):
        """
        初始化
        
        Parameters
        ----------
        data_file : str
            包含qualifiers的数据文件
        use_weights : bool
            是否使用权重
        """
        import torch
        
        self.df = pd.read_csv(data_file)
        self.use_weights = use_weights
        
        # 预处理
        self.df = self.df.dropna(subset=['Q', 'H'])
        
        print(f"加载了 {len(self.df)} 条记录")
        
        if use_weights:
            print(f"平均权重 - 径流: {self.df['Q_weight'].mean():.3f}, "
                  f"水位: {self.df['H_weight'].mean():.3f}")
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        """
        获取单个样本
        
        Returns
        -------
        tuple
            (features, targets, weights)
        """
        import torch
        
        row = self.df.iloc[idx]
        
        # 特征（这里简化，实际应该包含气象强迫等）
        features = torch.tensor([row['Q'], row['H']], dtype=torch.float32)
        
        # 目标（这里简化）
        targets = torch.tensor([row['Q'], row['H']], dtype=torch.float32)
        
        # 权重
        if self.use_weights:
            weights = torch.tensor([row['Q_weight'], row['H_weight']], dtype=torch.float32)
        else:
            weights = torch.ones(2, dtype=torch.float32)
        
        return features, targets, weights


# 使用示例
"""
from torch.utils.data import DataLoader

dataset = WeightedHydroDataset('qualifiers_output/camelsh_with_qualifiers.csv')
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

for features, targets, weights in dataloader:
    # 训练代码...
    pass
"""


# ==================== 示例6: 数据质量可视化 ====================

def visualize_data_quality(data_file: str, gauge_id: str = None):
    """
    可视化数据质量
    
    Parameters
    ----------
    data_file : str
        数据文件
    gauge_id : str, optional
        指定站点ID，如果为None则分析所有站点
    """
    import matplotlib.pyplot as plt
    
    df = pd.read_csv(data_file)
    
    if gauge_id:
        df = df[df['gauge_id'] == gauge_id]
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # 1. 权重分布直方图
    axes[0, 0].hist(df['Q_weight'].dropna(), bins=50, alpha=0.7, label='径流')
    axes[0, 0].hist(df['H_weight'].dropna(), bins=50, alpha=0.7, label='水位')
    axes[0, 0].set_xlabel('权重')
    axes[0, 0].set_ylabel('频数')
    axes[0, 0].set_title('权重分布')
    axes[0, 0].legend()
    
    # 2. Qualifier分布饼图
    q_flag_counts = df['Q_flag'].value_counts().head(5)
    axes[0, 1].pie(q_flag_counts.values, labels=q_flag_counts.index, autopct='%1.1f%%')
    axes[0, 1].set_title('径流Qualifier分布（Top 5）')
    
    # 3. 时间序列 - 权重变化
    df['datetime'] = pd.to_datetime(df['datetime'])
    df_sorted = df.sort_values('datetime')
    
    axes[1, 0].plot(df_sorted['datetime'], df_sorted['Q_weight'], alpha=0.5, label='径流')
    axes[1, 0].plot(df_sorted['datetime'], df_sorted['H_weight'], alpha=0.5, label='水位')
    axes[1, 0].set_xlabel('时间')
    axes[1, 0].set_ylabel('权重')
    axes[1, 0].set_title('权重时间序列')
    axes[1, 0].legend()
    axes[1, 0].tick_params(axis='x', rotation=45)
    
    # 4. 权重 vs 数据值散点图
    axes[1, 1].scatter(df['Q'], df['Q_weight'], alpha=0.3, s=1, label='径流')
    axes[1, 1].scatter(df['H'], df['H_weight'], alpha=0.3, s=1, label='水位')
    axes[1, 1].set_xlabel('观测值')
    axes[1, 1].set_ylabel('权重')
    axes[1, 1].set_title('权重 vs 观测值')
    axes[1, 1].legend()
    
    plt.tight_layout()
    plt.savefig('data_quality_analysis.png', dpi=300)
    print("可视化已保存: data_quality_analysis.png")


# ==================== 主函数 ====================

if __name__ == "__main__":
    """运行所有示例"""
    
    DATA_FILE = "qualifiers_output/camelsh_with_qualifiers.csv"
    
    print("=" * 80)
    print("Qualifiers使用示例")
    print("=" * 80)
    
    # 检查文件是否存在
    import os
    if not os.path.exists(DATA_FILE):
        print(f"\n错误: 找不到文件 {DATA_FILE}")
        print("请先运行 usgs_qualifiers_fetcher.py")
        exit(1)
    
    # 示例1: 过滤数据
    print("\n示例1: 过滤低质量数据")
    print("-" * 40)
    filtered_df = filter_by_quality(DATA_FILE, min_weight=0.7)
    
    # 示例4: 分组分析
    print("\n示例4: 按qualifier分析")
    print("-" * 40)
    analyze_by_qualifier(DATA_FILE)
    
    # 示例6: 可视化
    print("\n示例6: 数据质量可视化")
    print("-" * 40)
    visualize_data_quality(DATA_FILE)
    
    print("\n" + "=" * 80)
    print("示例运行完成!")
    print("=" * 80)
    print("\n更多示例请查看代码注释")

