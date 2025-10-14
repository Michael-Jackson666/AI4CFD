"""
可视化模块 - DeepONet 结果展示
Visualization Module for DeepONet Results
"""

import numpy as np
import matplotlib.pyplot as plt
import torch
import os


def plot_prediction_comparison(f0, f_true, f_pred, x, v, t, save_path=None):
    """
    对比真实解和预测解
    
    Args:
        f0: 初始条件 [nx, nv]
        f_true: 真实解 [nx, nv]
        f_pred: 预测解 [nx, nv]
        x: 空间网格
        v: 速度网格
        t: 时间
        save_path: 保存路径
    """
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # 转换为 numpy
    if torch.is_tensor(f0):
        f0 = f0.cpu().numpy()
    if torch.is_tensor(f_true):
        f_true = f_true.cpu().numpy()
    if torch.is_tensor(f_pred):
        f_pred = f_pred.cpu().numpy()
    if torch.is_tensor(x):
        x = x.cpu().numpy()
    if torch.is_tensor(v):
        v = v.cpu().numpy()
    
    # 初始条件
    im0 = axes[0, 0].contourf(x, v, f0.T, levels=20, cmap='viridis')
    axes[0, 0].set_xlabel('x')
    axes[0, 0].set_ylabel('v')
    axes[0, 0].set_title(f'Initial Condition (t=0)')
    plt.colorbar(im0, ax=axes[0, 0])
    
    # 真实解
    vmin = min(f_true.min(), f_pred.min())
    vmax = max(f_true.max(), f_pred.max())
    
    im1 = axes[0, 1].contourf(x, v, f_true.T, levels=20, cmap='viridis', vmin=vmin, vmax=vmax)
    axes[0, 1].set_xlabel('x')
    axes[0, 1].set_ylabel('v')
    axes[0, 1].set_title(f'True Solution (t={t:.2f})')
    plt.colorbar(im1, ax=axes[0, 1])
    
    # 预测解
    im2 = axes[1, 0].contourf(x, v, f_pred.T, levels=20, cmap='viridis', vmin=vmin, vmax=vmax)
    axes[1, 0].set_xlabel('x')
    axes[1, 0].set_ylabel('v')
    axes[1, 0].set_title(f'Predicted Solution (t={t:.2f})')
    plt.colorbar(im2, ax=axes[1, 0])
    
    # 误差
    error = np.abs(f_true - f_pred)
    im3 = axes[1, 1].contourf(x, v, error.T, levels=20, cmap='hot')
    axes[1, 1].set_xlabel('x')
    axes[1, 1].set_ylabel('v')
    axes[1, 1].set_title(f'Absolute Error (max={error.max():.2e})')
    plt.colorbar(im3, ax=axes[1, 1])
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"图像已保存: {save_path}")
    else:
        plt.show()
    
    plt.close()


def plot_time_evolution(f0, f_true_list, f_pred_list, x, v, t_list, save_path=None):
    """
    绘制时间演化序列
    
    Args:
        f0: 初始条件 [nx, nv]
        f_true_list: 真实解列表 [n_times, nx, nv]
        f_pred_list: 预测解列表 [n_times, nx, nv]
        x: 空间网格
        v: 速度网格
        t_list: 时间列表
        save_path: 保存路径
    """
    n_times = len(t_list)
    fig, axes = plt.subplots(3, n_times, figsize=(4*n_times, 10))
    
    # 转换为 numpy
    if torch.is_tensor(f0):
        f0 = f0.cpu().numpy()
    if torch.is_tensor(x):
        x = x.cpu().numpy()
    if torch.is_tensor(v):
        v = v.cpu().numpy()
    
    # 统一颜色范围
    all_true = [f.cpu().numpy() if torch.is_tensor(f) else f for f in f_true_list]
    all_pred = [f.cpu().numpy() if torch.is_tensor(f) else f for f in f_pred_list]
    vmin = min(f0.min(), min([f.min() for f in all_true]), min([f.min() for f in all_pred]))
    vmax = max(f0.max(), max([f.max() for f in all_true]), max([f.max() for f in all_pred]))
    
    for i, (f_true, f_pred, t) in enumerate(zip(all_true, all_pred, t_list)):
        # 真实解
        im1 = axes[0, i].contourf(x, v, f_true.T, levels=20, cmap='viridis', vmin=vmin, vmax=vmax)
        axes[0, i].set_xlabel('x')
        axes[0, i].set_ylabel('v')
        axes[0, i].set_title(f'True (t={t:.2f})')
        
        # 预测解
        im2 = axes[1, i].contourf(x, v, f_pred.T, levels=20, cmap='viridis', vmin=vmin, vmax=vmax)
        axes[1, i].set_xlabel('x')
        axes[1, i].set_ylabel('v')
        axes[1, i].set_title(f'Predicted (t={t:.2f})')
        
        # 误差
        error = np.abs(f_true - f_pred)
        im3 = axes[2, i].contourf(x, v, error.T, levels=20, cmap='hot')
        axes[2, i].set_xlabel('x')
        axes[2, i].set_ylabel('v')
        axes[2, i].set_title(f'Error (max={error.max():.2e})')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"图像已保存: {save_path}")
    else:
        plt.show()
    
    plt.close()


def plot_electric_field_comparison(E_true_list, E_pred_list, x, t_list, save_path=None):
    """
    对比电场演化
    
    Args:
        E_true_list: 真实电场列表 [n_times, nx]
        E_pred_list: 预测电场列表 [n_times, nx]
        x: 空间网格
        t_list: 时间列表
        save_path: 保存路径
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # 转换为 numpy
    if torch.is_tensor(x):
        x = x.cpu().numpy()
    
    E_true_array = np.array([E.cpu().numpy() if torch.is_tensor(E) else E for E in E_true_list])
    E_pred_array = np.array([E.cpu().numpy() if torch.is_tensor(E) else E for E in E_pred_list])
    t_array = np.array(t_list)
    
    # 真实电场时空图
    T, X = np.meshgrid(t_array, x, indexing='ij')
    vmin = min(E_true_array.min(), E_pred_array.min())
    vmax = max(E_true_array.max(), E_pred_array.max())
    
    im1 = axes[0, 0].contourf(X, T, E_true_array, levels=20, cmap='RdBu_r', vmin=vmin, vmax=vmax)
    axes[0, 0].set_xlabel('x')
    axes[0, 0].set_ylabel('t')
    axes[0, 0].set_title('True Electric Field')
    plt.colorbar(im1, ax=axes[0, 0])
    
    # 预测电场时空图
    im2 = axes[0, 1].contourf(X, T, E_pred_array, levels=20, cmap='RdBu_r', vmin=vmin, vmax=vmax)
    axes[0, 1].set_xlabel('x')
    axes[0, 1].set_ylabel('t')
    axes[0, 1].set_title('Predicted Electric Field')
    plt.colorbar(im2, ax=axes[0, 1])
    
    # 误差时空图
    error = np.abs(E_true_array - E_pred_array)
    im3 = axes[1, 0].contourf(X, T, error, levels=20, cmap='hot')
    axes[1, 0].set_xlabel('x')
    axes[1, 0].set_ylabel('t')
    axes[1, 0].set_title(f'Absolute Error (max={error.max():.2e})')
    plt.colorbar(im3, ax=axes[1, 0])
    
    # 不同时刻的电场曲线
    n_snapshots = min(5, len(t_list))
    indices = np.linspace(0, len(t_list)-1, n_snapshots, dtype=int)
    
    for idx in indices:
        axes[1, 1].plot(x, E_true_array[idx], '-', label=f't={t_array[idx]:.2f} (true)', alpha=0.7)
        axes[1, 1].plot(x, E_pred_array[idx], '--', label=f't={t_array[idx]:.2f} (pred)', alpha=0.7)
    
    axes[1, 1].set_xlabel('x')
    axes[1, 1].set_ylabel('E')
    axes[1, 1].set_title('Electric Field Profiles')
    axes[1, 1].legend(fontsize=8)
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"图像已保存: {save_path}")
    else:
        plt.show()
    
    plt.close()


def plot_loss_history(train_losses, val_losses=None, save_path=None):
    """
    绘制训练损失历史
    
    Args:
        train_losses: 训练损失列表
        val_losses: 验证损失列表（可选）
        save_path: 保存路径
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    epochs = range(1, len(train_losses) + 1)
    ax.plot(epochs, train_losses, 'b-', label='Training Loss', linewidth=2)
    
    if val_losses is not None:
        ax.plot(epochs, val_losses, 'r-', label='Validation Loss', linewidth=2)
    
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Loss', fontsize=12)
    ax.set_title('Training History', fontsize=14)
    ax.set_yscale('log')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"图像已保存: {save_path}")
    else:
        plt.show()
    
    plt.close()


def plot_error_distribution(errors, save_path=None):
    """
    绘制误差分布直方图
    
    Args:
        errors: 误差数组 [n_samples, nx, nv] or flattened
        save_path: 保存路径
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # 转换为 numpy
    if torch.is_tensor(errors):
        errors = errors.cpu().numpy()
    
    errors_flat = errors.flatten()
    
    # 直方图
    axes[0].hist(errors_flat, bins=50, edgecolor='black', alpha=0.7)
    axes[0].set_xlabel('Absolute Error', fontsize=12)
    axes[0].set_ylabel('Frequency', fontsize=12)
    axes[0].set_title('Error Distribution', fontsize=14)
    axes[0].grid(True, alpha=0.3)
    
    # 累积分布
    sorted_errors = np.sort(errors_flat)
    cumulative = np.arange(1, len(sorted_errors) + 1) / len(sorted_errors)
    axes[1].plot(sorted_errors, cumulative, linewidth=2)
    axes[1].set_xlabel('Absolute Error', fontsize=12)
    axes[1].set_ylabel('Cumulative Probability', fontsize=12)
    axes[1].set_title('Cumulative Error Distribution', fontsize=14)
    axes[1].grid(True, alpha=0.3)
    
    # 添加统计信息
    mean_error = errors_flat.mean()
    max_error = errors_flat.max()
    median_error = np.median(errors_flat)
    
    stats_text = f'Mean: {mean_error:.2e}\nMedian: {median_error:.2e}\nMax: {max_error:.2e}'
    axes[1].text(0.6, 0.3, stats_text, transform=axes[1].transAxes,
                fontsize=10, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"图像已保存: {save_path}")
    else:
        plt.show()
    
    plt.close()


def plot_generalization_test(model, test_dataset, device, save_dir='visualizations'):
    """
    测试模型泛化能力
    
    Args:
        model: 训练好的模型
        test_dataset: 测试数据集
        device: 计算设备
        save_dir: 保存目录
    """
    os.makedirs(save_dir, exist_ok=True)
    
    model.eval()
    
    # 随机选择几个测试样本
    n_samples = min(5, len(test_dataset['initial_conditions']))
    indices = np.random.choice(len(test_dataset['initial_conditions']), n_samples, replace=False)
    
    print(f"\n测试 {n_samples} 个样本的泛化能力...")
    
    for i, idx in enumerate(indices):
        f0 = torch.tensor(test_dataset['initial_conditions'][idx], dtype=torch.float32).to(device)
        f_true_all = test_dataset['solutions'][idx]  # [nt, nx, nv]
        params = test_dataset['parameters'][idx]
        
        # 选择几个时间点
        t_indices = [len(test_dataset['t'])//4, len(test_dataset['t'])//2, -1]
        t_list = [test_dataset['t'][ti] for ti in t_indices]
        
        f_true_list = []
        f_pred_list = []
        
        with torch.no_grad():
            for ti in t_indices:
                t = test_dataset['t'][ti]
                f_true = f_true_all[ti]
                
                # 预测
                f_pred = model(f0.unsqueeze(0), torch.tensor([t], dtype=torch.float32).to(device))
                f_pred = f_pred.squeeze(0)
                
                f_true_list.append(f_true)
                f_pred_list.append(f_pred)
        
        # 绘制时间演化
        save_path = os.path.join(save_dir, f'generalization_test_{i+1}.png')
        plot_time_evolution(
            f0.cpu().numpy(),
            f_true_list,
            f_pred_list,
            test_dataset['x'],
            test_dataset['v'],
            t_list,
            save_path=save_path
        )
        
        print(f"  样本 {i+1}: beam_v={params[0]:.2f}, thermal_v={params[1]:.3f}, "
              f"perturb_amp={params[2]:.3f}, k_mode={int(params[3])}")


if __name__ == '__main__':
    # 测试可视化函数
    print("测试可视化模块...")
    
    # 创建虚拟数据
    nx, nv = 64, 64
    x = np.linspace(0, 10, nx)
    v = np.linspace(-5, 5, nv)
    
    X, V = np.meshgrid(x, v, indexing='ij')
    
    # 初始条件
    f0 = np.exp(-(V**2)/2) * (1 + 0.1*np.cos(2*np.pi*X/10))
    
    # 模拟演化
    t = 25.0
    f_true = np.exp(-(V**2)/2) * (1 + 0.05*np.cos(2*np.pi*X/10 - t))
    f_pred = f_true + np.random.randn(*f_true.shape) * 0.01
    
    # 测试对比图
    print("\n生成对比图...")
    plot_prediction_comparison(f0, f_true, f_pred, x, v, t, save_path='test_comparison.png')
    
    # 测试损失历史
    print("\n生成损失历史图...")
    train_losses = [1.0 / (i+1)**0.5 for i in range(100)]
    val_losses = [1.2 / (i+1)**0.5 for i in range(100)]
    plot_loss_history(train_losses, val_losses, save_path='test_loss.png')
    
    # 测试误差分布
    print("\n生成误差分布图...")
    errors = np.abs(f_true - f_pred)
    plot_error_distribution(errors, save_path='test_error_dist.png')
    
    print("\n可视化测试完成！")
