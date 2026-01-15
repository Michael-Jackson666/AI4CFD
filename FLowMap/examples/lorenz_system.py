"""
Lorenz System Prediction using Flow Map Learning

This example demonstrates how to use Flow Map Learning to predict
the chaotic Lorenz system trajectory.

The Lorenz system:
    dx/dt = σ(y - x)
    dy/dt = x(ρ - z) - y
    dz/dt = xy - βz

With parameters σ = 10, ρ = 28, β = 8/3 (chaotic regime)

Author: AI4CFD Project
Date: 2026-01
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import sys
import os

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models import FlowMapMLP, FlowMapResNet
from utils import (lorenz_system, generate_trajectory, create_training_pairs,
                  train_flowmap, plot_trajectory_3d, plot_time_series,
                  plot_error_analysis)


def main():
    """Main function to run Lorenz system prediction."""
    
    # Set random seed
    torch.manual_seed(42)
    np.random.seed(42)
    
    # ========================================================================
    # 1. 生成训练数据
    # ========================================================================
    print("=" * 70)
    print("Lorenz System - Flow Map Learning".center(70))
    print("=" * 70)
    print("\n[1] Generating training data...")
    
    # Lorenz 参数
    sigma, rho, beta = 10.0, 28.0, 8.0/3.0
    
    # 生成训练轨迹
    x0_train = np.array([1.0, 1.0, 1.0])
    t_span_train = (0, 20)
    dt = 0.01
    
    t_train, traj_train = generate_trajectory(
        lorenz_system, x0_train, t_span_train, dt,
        sigma=sigma, rho=rho, beta=beta
    )
    
    print(f"  Training trajectory shape: {traj_train.shape}")
    print(f"  Time span: {t_span_train}")
    print(f"  Time step: {dt}")
    print(f"  Number of steps: {len(t_train)}")
    
    # 创建训练对
    x_train, x_target, dt_tensor = create_training_pairs(traj_train, dt, stride=1)
    
    print(f"  Training pairs: {len(x_train)}")
    
    # ========================================================================
    # 2. 创建 Flow Map 模型
    # ========================================================================
    print("\n[2] Creating Flow Map model...")
    
    model = FlowMapMLP(
        state_dim=3,
        hidden_dims=[128, 128, 128],
        activation='tanh',
        use_residual=True,
        time_encoding=True
    )
    
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Model architecture: FlowMapMLP")
    print(f"  State dimension: 3")
    print(f"  Hidden dimensions: [128, 128, 128]")
    print(f"  Total parameters: {n_params}")
    print(f"  Use residual: True")
    print(f"  Time encoding: True")
    
    # ========================================================================
    # 3. 训练模型
    # ========================================================================
    print("\n[3] Training Flow Map model...")
    
    history = train_flowmap(
        model, x_train, x_target, dt_tensor,
        epochs=2000,
        lr=1e-3,
        print_every=200
    )
    
    # 绘制训练曲线
    plt.figure(figsize=(10, 4))
    plt.semilogy(history['loss'], 'b-', linewidth=2)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Loss (log scale)', fontsize=12)
    plt.title('Training Loss', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('lorenz_training_loss.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # ========================================================================
    # 4. 测试预测
    # ========================================================================
    print("\n[4] Testing prediction...")
    
    # 使用不同的初始条件进行测试
    x0_test = np.array([1.0, 0.0, 0.0])
    t_span_test = (0, 25)
    
    t_test, traj_test = generate_trajectory(
        lorenz_system, x0_test, t_span_test, dt,
        sigma=sigma, rho=rho, beta=beta
    )
    
    print(f"  Test trajectory: {len(t_test)} steps")
    
    # 多步预测
    model.eval()
    with torch.no_grad():
        x0_tensor = torch.FloatTensor(x0_test).unsqueeze(0)
        n_steps = len(t_test) - 1
        
        pred_trajectory = model.multi_step_predict(x0_tensor, dt, n_steps)
        pred_trajectory = pred_trajectory.squeeze().numpy()
    
    print(f"  Predicted trajectory shape: {pred_trajectory.shape}")
    
    # ========================================================================
    # 5. 可视化结果
    # ========================================================================
    print("\n[5] Visualizing results...")
    
    # 3D 相空间
    fig = plt.figure(figsize=(16, 6))
    
    ax1 = fig.add_subplot(121, projection='3d')
    ax1.plot(traj_test[:, 0], traj_test[:, 1], traj_test[:, 2],
             'b-', linewidth=0.8, label='True', alpha=0.7)
    ax1.set_xlabel('x', fontsize=11)
    ax1.set_ylabel('y', fontsize=11)
    ax1.set_zlabel('z', fontsize=11)
    ax1.set_title('True Lorenz Attractor', fontsize=14, fontweight='bold')
    ax1.legend()
    
    ax2 = fig.add_subplot(122, projection='3d')
    ax2.plot(pred_trajectory[:, 0], pred_trajectory[:, 1], pred_trajectory[:, 2],
             'r-', linewidth=0.8, label='Predicted', alpha=0.7)
    ax2.set_xlabel('x', fontsize=11)
    ax2.set_ylabel('y', fontsize=11)
    ax2.set_zlabel('z', fontsize=11)
    ax2.set_title('Flow Map Prediction', fontsize=14, fontweight='bold')
    ax2.legend()
    
    plt.tight_layout()
    plt.savefig('lorenz_3d_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # 时间序列对比
    fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)
    labels = ['x', 'y', 'z']
    
    for i, (ax, label) in enumerate(zip(axes, labels)):
        ax.plot(t_test, traj_test[:, i], 'b-', linewidth=2, 
                label='True', alpha=0.7)
        ax.plot(t_test, pred_trajectory[:, i], 'r--', linewidth=2,
                label='Predicted', alpha=0.7)
        ax.set_ylabel(f'${label}$', fontsize=12)
        ax.legend(loc='upper right', fontsize=10)
        ax.grid(True, alpha=0.3)
    
    axes[-1].set_xlabel('Time', fontsize=12)
    plt.suptitle('Lorenz System: Time Series Comparison', 
                fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('lorenz_time_series.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # ========================================================================
    # 6. 误差分析
    # ========================================================================
    print("\n[6] Error analysis...")
    
    # 计算误差
    error = np.abs(traj_test - pred_trajectory)
    total_error = np.linalg.norm(error, axis=1)
    
    # 相对误差
    true_norm = np.linalg.norm(traj_test, axis=1) + 1e-10
    relative_error = total_error / true_norm * 100
    
    print(f"  Mean Absolute Error: {np.mean(error):.6e}")
    print(f"  Max Absolute Error:  {np.max(error):.6e}")
    print(f"  Mean Relative Error: {np.mean(relative_error):.4f}%")
    
    # 找到误差超过阈值的时间（混沌预测的有效时间）
    threshold = 10  # 10% 相对误差
    valid_time_idx = np.where(relative_error > threshold)[0]
    if len(valid_time_idx) > 0:
        valid_time = t_test[valid_time_idx[0]]
        print(f"  Valid prediction time (< {threshold}% error): {valid_time:.2f}")
    else:
        print(f"  Valid prediction time: > {t_test[-1]:.2f} (entire trajectory)")
    
    # 误差演化图
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # 绝对误差
    for i, label in enumerate(['x', 'y', 'z']):
        axes[0, 0].plot(t_test, error[:, i], label=f'${label}$', 
                       linewidth=2, alpha=0.8)
    axes[0, 0].set_xlabel('Time', fontsize=12)
    axes[0, 0].set_ylabel('Absolute Error', fontsize=12)
    axes[0, 0].set_title('Error per Component', fontsize=12, fontweight='bold')
    axes[0, 0].legend(fontsize=10)
    axes[0, 0].grid(True, alpha=0.3)
    
    # 总误差（对数）
    axes[0, 1].semilogy(t_test, total_error, 'r-', linewidth=2)
    axes[0, 1].set_xlabel('Time', fontsize=12)
    axes[0, 1].set_ylabel('Total Error (log scale)', fontsize=12)
    axes[0, 1].set_title('Total Error Evolution', fontsize=12, fontweight='bold')
    axes[0, 1].grid(True, alpha=0.3)
    
    # 相对误差
    axes[1, 0].plot(t_test, relative_error, 'g-', linewidth=2)
    axes[1, 0].axhline(y=threshold, color='k', linestyle='--', alpha=0.5,
                      label=f'{threshold}% threshold')
    axes[1, 0].set_xlabel('Time', fontsize=12)
    axes[1, 0].set_ylabel('Relative Error (%)', fontsize=12)
    axes[1, 0].set_title('Relative Error', fontsize=12, fontweight='bold')
    axes[1, 0].legend(fontsize=10)
    axes[1, 0].grid(True, alpha=0.3)
    
    # 短期预测放大
    short_time_idx = int(5 / dt)  # 前5秒
    axes[1, 1].plot(t_test[:short_time_idx], traj_test[:short_time_idx, 0],
                   'b-', linewidth=2, label='True x')
    axes[1, 1].plot(t_test[:short_time_idx], pred_trajectory[:short_time_idx, 0],
                   'r--', linewidth=2, label='Predicted x')
    axes[1, 1].set_xlabel('Time', fontsize=12)
    axes[1, 1].set_ylabel('x', fontsize=12)
    axes[1, 1].set_title('Short-term Prediction (first 5s)', fontsize=12, fontweight='bold')
    axes[1, 1].legend(fontsize=10)
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('lorenz_error_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # ========================================================================
    # 7. 保存模型
    # ========================================================================
    print("\n[7] Saving model...")
    
    torch.save({
        'model_state_dict': model.state_dict(),
        'sigma': sigma,
        'rho': rho,
        'beta': beta,
        'dt': dt,
        'history': history
    }, 'lorenz_flowmap_model.pth')
    
    print("  Model saved to: lorenz_flowmap_model.pth")
    
    print("\n" + "=" * 70)
    print("Complete!".center(70))
    print("=" * 70)
    print("\nGenerated files:")
    print("  - lorenz_training_loss.png")
    print("  - lorenz_3d_comparison.png")
    print("  - lorenz_time_series.png")
    print("  - lorenz_error_analysis.png")
    print("  - lorenz_flowmap_model.pth")


if __name__ == "__main__":
    main()
