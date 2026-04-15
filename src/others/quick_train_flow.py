#!/usr/bin/env python3
"""
快速训练单任务流量模型
"""

from config import *
from improved_camelsh_reader import ImprovedCAMELSHReader
from single_task_lstm import SingleTaskDataset, SingleTaskLSTM, train_single_task_model
from hydrodataset import StandardVariable
import torch

# 设置设备
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"使用设备: {DEVICE}")

def main():
    print("=== 快速训练单任务流量模型 ===")
    
    # 训练参数
    NUM_EPOCHS = 20  # 减少训练轮数以加快速度
    LEARNING_RATE = 0.001
    BATCH_SIZE = 256
    
    print(f"训练参数:")
    print(f"  训练轮数: {NUM_EPOCHS}")
    print(f"  学习率: {LEARNING_RATE}")
    print(f"  批次大小: {BATCH_SIZE}")
    print(f"  序列长度: {SEQUENCE_LENGTH}")
    print(f"  隐藏层大小: {HIDDEN_SIZE}")
    
    try:
        # 训练模型
        model, means, stds, nse = train_single_task_model(
            target_type="flow", 
            num_epochs=NUM_EPOCHS
        )
        
        print(f"\n=== 训练完成 ===")
        print(f"最终NSE: {nse:.4f}")
        print(f"模型已保存到: single_task_flow_model.pth")
        
        if nse > 0:
            print("✅ 训练成功！NSE为正值")
        else:
            print("⚠️ NSE为负值，模型性能较差")
            
    except Exception as e:
        print(f"❌ 训练失败: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()





