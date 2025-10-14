"""
主训练脚本 - DeepONet for Vlasov-Poisson System
Main Training Script
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pickle
import os
import json
from datetime import datetime
from tqdm import tqdm

from vp_operator import VlasovPoissonOperator
from visualization import (
    plot_prediction_comparison,
    plot_time_evolution,
    plot_electric_field_comparison,
    plot_loss_history,
    plot_error_distribution,
    plot_generalization_test
)


class VPDataset(Dataset):
    """
    Vlasov-Poisson 数据集
    
    将生成的数据转换为 PyTorch Dataset
    """
    
    def __init__(self, data_dict, n_time_samples=10):
        """
        初始化数据集
        
        Args:
            data_dict: 数据字典（从 pickle 加载）
            n_time_samples: 每个样本采样的时间点数
        """
        self.f0 = torch.tensor(data_dict['initial_conditions'], dtype=torch.float32)
        self.solutions = torch.tensor(data_dict['solutions'], dtype=torch.float32)
        self.x = torch.tensor(data_dict['x'], dtype=torch.float32)
        self.v = torch.tensor(data_dict['v'], dtype=torch.float32)
        self.t = torch.tensor(data_dict['t'], dtype=torch.float32)
        
        self.n_samples = len(self.f0)
        self.n_time_samples = n_time_samples
        self.nt, self.nx, self.nv = self.solutions.shape[1:]
    
    def __len__(self):
        return self.n_samples
    
    def __getitem__(self, idx):
        """
        获取一个训练样本
        
        Returns:
            f0: 初始条件 [nx, nv]
            t_samples: 采样的时间点 [n_time_samples]
            f_targets: 对应的目标分布 [n_time_samples, nx, nv]
        """
        f0 = self.f0[idx]
        
        # 随机采样时间点（不包括 t=0）
        t_indices = torch.randint(1, self.nt, (self.n_time_samples,))
        t_samples = self.t[t_indices]
        f_targets = self.solutions[idx, t_indices]
        
        return f0, t_samples, f_targets


def load_dataset(data_path):
    """
    加载数据集
    
    Args:
        data_path: 数据文件路径
        
    Returns:
        data_dict: 数据字典
    """
    print(f"加载数据集: {data_path}")
    with open(data_path, 'rb') as f:
        data_dict = pickle.load(f)
    
    print(f"  样本数: {len(data_dict['initial_conditions'])}")
    print(f"  网格大小: {data_dict['solutions'].shape[1:]}  (nt, nx, nv)")
    
    return data_dict


def train_epoch(model, dataloader, optimizer, device):
    """
    训练一个 epoch
    
    Args:
        model: 模型
        dataloader: 数据加载器
        optimizer: 优化器
        device: 计算设备
        
    Returns:
        avg_loss: 平均损失
    """
    model.train()
    total_loss = 0.0
    n_batches = 0
    
    for f0, t_samples, f_targets in tqdm(dataloader, desc="Training", leave=False):
        f0 = f0.to(device)
        t_samples = t_samples.to(device)
        f_targets = f_targets.to(device)
        
        batch_size = f0.shape[0]
        n_time_samples = t_samples.shape[1]
        
        optimizer.zero_grad()
        
        # 对每个时间点进行预测
        loss = 0.0
        for i in range(n_time_samples):
            t_i = t_samples[:, i]
            f_target_i = f_targets[:, i]
            
            # 预测
            f_pred_i = model(f0, t_i[0])  # 假设 batch 内的时间相同
            
            # MSE 损失
            loss_i = torch.mean((f_pred_i - f_target_i)**2)
            loss += loss_i
        
        loss = loss / n_time_samples
        
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        n_batches += 1
    
    avg_loss = total_loss / n_batches
    return avg_loss


def validate(model, dataloader, device):
    """
    验证模型
    
    Args:
        model: 模型
        dataloader: 数据加载器
        device: 计算设备
        
    Returns:
        avg_loss: 平均损失
    """
    model.eval()
    total_loss = 0.0
    n_batches = 0
    
    with torch.no_grad():
        for f0, t_samples, f_targets in tqdm(dataloader, desc="Validation", leave=False):
            f0 = f0.to(device)
            t_samples = t_samples.to(device)
            f_targets = f_targets.to(device)
            
            n_time_samples = t_samples.shape[1]
            
            # 对每个时间点进行预测
            loss = 0.0
            for i in range(n_time_samples):
                t_i = t_samples[:, i]
                f_target_i = f_targets[:, i]
                
                # 预测
                f_pred_i = model(f0, t_i[0])
                
                # MSE 损失
                loss_i = torch.mean((f_pred_i - f_target_i)**2)
                loss += loss_i
            
            loss = loss / n_time_samples
            
            total_loss += loss.item()
            n_batches += 1
    
    avg_loss = total_loss / n_batches
    return avg_loss


def evaluate_test_set(model, test_data, device, save_dir='results'):
    """
    在测试集上评估模型
    
    Args:
        model: 训练好的模型
        test_data: 测试数据字典
        device: 计算设备
        save_dir: 保存目录
    """
    print("\n" + "="*70)
    print("测试集评估")
    print("="*70)
    
    model.eval()
    
    f0_all = torch.tensor(test_data['initial_conditions'], dtype=torch.float32).to(device)
    solutions_all = test_data['solutions']
    t_all = test_data['t']
    x = test_data['x']
    v = test_data['v']
    
    n_samples = len(f0_all)
    nt = len(t_all)
    
    # 存储所有误差
    all_errors = []
    
    # 对每个测试样本进行评估
    print(f"\n评估 {n_samples} 个测试样本...")
    
    with torch.no_grad():
        for i in tqdm(range(n_samples), desc="Testing"):
            f0 = f0_all[i:i+1]
            f_true_all = solutions_all[i]
            
            # 预测所有时间点
            for t_idx in range(1, nt):  # 跳过 t=0
                t = torch.tensor([t_all[t_idx]], dtype=torch.float32).to(device)
                f_true = f_true_all[t_idx]
                
                # 预测
                f_pred = model(f0, t).squeeze(0).cpu().numpy()
                
                # 计算误差
                error = np.abs(f_pred - f_true)
                all_errors.append(error)
    
    all_errors = np.array(all_errors)
    
    # 统计误差
    mean_error = all_errors.mean()
    max_error = all_errors.max()
    median_error = np.median(all_errors)
    
    print(f"\n测试集误差统计:")
    print(f"  平均误差: {mean_error:.4e}")
    print(f"  中位误差: {median_error:.4e}")
    print(f"  最大误差: {max_error:.4e}")
    
    # 绘制误差分布
    error_path = os.path.join(save_dir, 'test_error_distribution.png')
    plot_error_distribution(all_errors, save_path=error_path)
    
    # 可视化泛化测试
    plot_generalization_test(model, test_data, device, save_dir=os.path.join(save_dir, 'generalization'))
    
    return {
        'mean_error': mean_error,
        'median_error': median_error,
        'max_error': max_error
    }


def save_config(config, save_dir):
    """
    保存配置文件
    
    Args:
        config: 配置字典
        save_dir: 保存目录
    """
    os.makedirs(save_dir, exist_ok=True)
    
    # 保存 JSON
    config_path = os.path.join(save_dir, 'training_config.json')
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=4)
    
    # 保存可读文本
    txt_path = os.path.join(save_dir, 'training_config.txt')
    with open(txt_path, 'w') as f:
        f.write("="*70 + "\n")
        f.write("DeepONet Training Configuration\n")
        f.write("="*70 + "\n\n")
        
        for key, value in config.items():
            f.write(f"{key}: {value}\n")
    
    print(f"配置已保存:")
    print(f"  {config_path}")
    print(f"  {txt_path}")


def main():
    """
    主训练函数
    """
    print("="*70)
    print("DeepONet for Vlasov-Poisson System")
    print("="*70)
    
    # ======================= 配置参数 =======================
    config = {
        # 数据参数
        'train_data_path': 'data/train/vp_dataset.pkl',
        'val_data_path': 'data/val/vp_dataset.pkl',
        'test_data_path': 'data/test/vp_dataset.pkl',
        
        # 网络参数
        'branch_dim': 128,
        'trunk_dim': 128,
        'p': 100,  # 基函数数量
        
        # 训练参数
        'batch_size': 8,
        'n_epochs': 100,
        'learning_rate': 1e-3,
        'lr_scheduler': 'cosine',  # 'cosine' or 'step'
        'n_time_samples': 10,  # 每个样本采样的时间点数
        
        # 设备
        'device': 'cuda' if torch.cuda.is_available() else 'cpu',
        
        # 保存路径
        'save_dir': 'results',
        'checkpoint_dir': 'checkpoints',
        
        # 其他
        'seed': 42,
        'num_workers': 0,
        
        # 记录时间
        'start_time': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }
    
    # 设置随机种子
    torch.manual_seed(config['seed'])
    np.random.seed(config['seed'])
    
    device = torch.device(config['device'])
    print(f"\n使用设备: {device}")
    
    # ======================= 加载数据 =======================
    print("\n" + "="*70)
    print("加载数据集")
    print("="*70)
    
    train_data = load_dataset(config['train_data_path'])
    val_data = load_dataset(config['val_data_path'])
    test_data = load_dataset(config['test_data_path'])
    
    # 从数据中获取网格信息
    config['nx'] = train_data['x'].shape[0]
    config['nv'] = train_data['v'].shape[0]
    config['t_max'] = float(train_data['t'][-1])
    config['x_max'] = float(train_data['x'][-1])
    config['v_max'] = float(train_data['v'][-1])
    
    # 创建数据集和数据加载器
    train_dataset = VPDataset(train_data, n_time_samples=config['n_time_samples'])
    val_dataset = VPDataset(val_data, n_time_samples=config['n_time_samples'])
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['batch_size'],
        shuffle=True,
        num_workers=config['num_workers']
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        num_workers=config['num_workers']
    )
    
    # ======================= 创建模型 =======================
    print("\n" + "="*70)
    print("创建模型")
    print("="*70)
    
    model = VlasovPoissonOperator(config).to(device)
    
    # ======================= 优化器和调度器 =======================
    optimizer = optim.Adam(model.parameters(), lr=config['learning_rate'])
    
    if config['lr_scheduler'] == 'cosine':
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=config['n_epochs'],
            eta_min=1e-6
        )
    elif config['lr_scheduler'] == 'step':
        scheduler = optim.lr_scheduler.StepLR(
            optimizer,
            step_size=config['n_epochs']//3,
            gamma=0.1
        )
    else:
        scheduler = None
    
    # ======================= 训练循环 =======================
    print("\n" + "="*70)
    print("开始训练")
    print("="*70)
    
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    
    # 创建保存目录
    os.makedirs(config['save_dir'], exist_ok=True)
    os.makedirs(config['checkpoint_dir'], exist_ok=True)
    
    for epoch in range(config['n_epochs']):
        print(f"\nEpoch {epoch+1}/{config['n_epochs']}")
        print("-" * 70)
        
        # 训练
        train_loss = train_epoch(model, train_loader, optimizer, device)
        train_losses.append(train_loss)
        
        # 验证
        val_loss = validate(model, val_loader, device)
        val_losses.append(val_loss)
        
        # 更新学习率
        if scheduler is not None:
            scheduler.step()
            current_lr = scheduler.get_last_lr()[0]
        else:
            current_lr = config['learning_rate']
        
        print(f"Train Loss: {train_loss:.4e} | Val Loss: {val_loss:.4e} | LR: {current_lr:.2e}")
        
        # 保存最佳模型
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            checkpoint_path = os.path.join(config['checkpoint_dir'], 'best_model.pth')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'config': config
            }, checkpoint_path)
            print(f"  ✓ 最佳模型已保存: {checkpoint_path}")
        
        # 定期保存检查点
        if (epoch + 1) % 20 == 0:
            checkpoint_path = os.path.join(config['checkpoint_dir'], f'checkpoint_epoch_{epoch+1}.pth')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_loss,
                'val_loss': val_loss,
                'config': config
            }, checkpoint_path)
            print(f"  ✓ 检查点已保存: {checkpoint_path}")
    
    # 记录结束时间
    config['end_time'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    config['best_val_loss'] = float(best_val_loss)
    
    # ======================= 保存训练历史 =======================
    print("\n" + "="*70)
    print("保存训练结果")
    print("="*70)
    
    # 保存配置
    save_config(config, config['save_dir'])
    
    # 保存损失历史
    loss_path = os.path.join(config['save_dir'], 'loss_history.png')
    plot_loss_history(train_losses, val_losses, save_path=loss_path)
    
    # 保存损失数据
    np.savez(
        os.path.join(config['save_dir'], 'losses.npz'),
        train_losses=train_losses,
        val_losses=val_losses
    )
    
    # ======================= 测试集评估 =======================
    # 加载最佳模型
    checkpoint = torch.load(os.path.join(config['checkpoint_dir'], 'best_model.pth'))
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # 评估
    test_metrics = evaluate_test_set(model, test_data, device, save_dir=config['save_dir'])
    
    # 保存测试结果
    config['test_metrics'] = test_metrics
    save_config(config, config['save_dir'])
    
    print("\n" + "="*70)
    print("训练完成！")
    print("="*70)
    print(f"\n最佳验证损失: {best_val_loss:.4e}")
    print(f"测试集平均误差: {test_metrics['mean_error']:.4e}")
    print(f"\n结果保存在: {config['save_dir']}")


if __name__ == '__main__':
    main()
