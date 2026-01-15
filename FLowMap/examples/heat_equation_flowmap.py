"""
Heat Equation Prediction using Flow Map Learning

This example demonstrates how to use Flow Map Learning to predict
the time evolution of the 1D heat equation.

The 1D heat equation:
    ∂u/∂t = α * ∂²u/∂x²

with Dirichlet boundary conditions: u(0,t) = u(L,t) = 0
and initial condition: u(x,0) = sin(πx)

Author: AI4CFD Project
Date: 2026-01
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
import sys
import os

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models import FlowMapCNN
from utils import heat_equation_1d, plot_pde_evolution


def analytical_solution(x: np.ndarray, t: np.ndarray, alpha: float = 0.01) -> np.ndarray:
    """
    热传导方程的解析解
    
    对于初始条件 u(x,0) = sin(πx)，解析解为：
    u(x,t) = sin(πx) * exp(-α * π² * t)
    
    Args:
        x: 空间网格 [nx]
        t: 时间网格 [nt]
        alpha: 热扩散系数
    
    Returns:
        solution: [nt, nx]
    """
    X, T = np.meshgrid(x, t)
    return np.sin(np.pi * X) * np.exp(-alpha * np.pi**2 * T)


def generate_training_data(nx: int = 64, 
                          n_train_steps: int = 500,
                          alpha: float = 0.01) -> tuple:
    """
    生成热传导方程的训练数据
    
    Args:
        nx: 空间网格点数
        n_train_steps: 训练时间步数
        alpha: 热扩散系数
    
    Returns:
        x: 空间网格
        t: 时间网格
        solution: 数值解
        dx, dt: 网格步长
    """
    # 空间网格
    L = 1.0
    x = np.linspace(0, L, nx)
    dx = x[1] - x[0]
    
    # 时间步长（满足 CFL 条件）
    dt = 0.4 * dx**2 / alpha  # r = α*dt/dx² < 0.5
    
    # 初始条件
    u0 = np.sin(np.pi * x)
    
    # 数值求解
    solution = heat_equation_1d(u0, alpha, dx, dt, n_train_steps)
    
    # 时间网格
    t = np.arange(n_train_steps + 1) * dt
    
    return x, t, solution, dx, dt


def create_cnn_training_pairs(solution: np.ndarray, 
                              dt: float) -> tuple:
    """
    创建 CNN 训练对
    
    Args:
        solution: [n_steps + 1, nx]
        dt: 时间步长
    
    Returns:
        u_current: [n_pairs, 1, nx]
        u_next: [n_pairs, 1, nx]
        dt_tensor: [n_pairs, 1]
    """
    n_pairs = len(solution) - 1
    
    u_current = solution[:-1]  # [n_pairs, nx]
    u_next = solution[1:]      # [n_pairs, nx]
    
    # 添加通道维度
    u_current = u_current[:, np.newaxis, :]  # [n_pairs, 1, nx]
    u_next = u_next[:, np.newaxis, :]        # [n_pairs, 1, nx]
    
    dt_tensor = np.ones((n_pairs, 1)) * dt
    
    return (torch.FloatTensor(u_current),
            torch.FloatTensor(u_next),
            torch.FloatTensor(dt_tensor))


def train_cnn_flowmap(model, u_train, u_target, dt_tensor,
                      epochs=1000, lr=1e-3, print_every=100):
    """训练 CNN Flow Map 模型"""
    
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=100, verbose=False
    )
    
    history = {'loss': []}
    
    print("=" * 60)
    print("Training CNN Flow Map Model".center(60))
    print("=" * 60)
    
    for epoch in range(epochs):
        optimizer.zero_grad()
        
        # 前向传播
        u_pred = model(u_train, dt_tensor)
        
        # 损失
        loss = torch.mean((u_pred - u_target) ** 2)
        
        # 反向传播
        loss.backward()
        optimizer.step()
        scheduler.step(loss)
        
        history['loss'].append(loss.item())
        
        if (epoch + 1) % print_every == 0 or epoch == 0:
            print(f"Epoch {epoch+1:5d} | Loss: {loss.item():.6e}")
    
    print("=" * 60)
    print("Training completed!".center(60))
    print("=" * 60)
    
    return history


def main():
    """Main function to run heat equation prediction."""
    
    # Set random seed
    torch.manual_seed(42)
    np.random.seed(42)
    
    # ========================================================================
    # 1. 生成训练数据
    # ========================================================================
    print("=" * 70)
    print("Heat Equation - Flow Map Learning".center(70))
    print("=" * 70)
    print("\n[1] Generating training data...")
    
    # 参数
    nx = 64            # 空间网格点数
    alpha = 0.01       # 热扩散系数
    n_train_steps = 500  # 训练时间步数
    
    x, t_train, solution_train, dx, dt = generate_training_data(
        nx, n_train_steps, alpha
    )
    
    print(f"  Spatial points: {nx}")
    print(f"  Spatial step (dx): {dx:.6f}")
    print(f"  Time step (dt): {dt:.6f}")
    print(f"  Diffusion coefficient (α): {alpha}")
    print(f"  CFL number (r): {alpha * dt / dx**2:.3f}")
    print(f"  Training steps: {n_train_steps}")
    print(f"  Solution shape: {solution_train.shape}")
    
    # 可视化训练数据
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # 等高线图
    X, T = np.meshgrid(x, t_train)
    contour = axes[0].contourf(X, T, solution_train, levels=30, cmap='viridis')
    axes[0].set_xlabel('x', fontsize=12)
    axes[0].set_ylabel('t', fontsize=12)
    axes[0].set_title('Training Data: Heat Equation', fontsize=12, fontweight='bold')
    plt.colorbar(contour, ax=axes[0], label='u(x,t)')
    
    # 时间快照
    snapshot_indices = [0, n_train_steps//4, n_train_steps//2, 
                       3*n_train_steps//4, n_train_steps]
    colors = plt.cm.viridis(np.linspace(0, 1, len(snapshot_indices)))
    
    for idx, color in zip(snapshot_indices, colors):
        axes[1].plot(x, solution_train[idx], color=color,
                    label=f't = {t_train[idx]:.3f}', linewidth=2)
    
    axes[1].set_xlabel('x', fontsize=12)
    axes[1].set_ylabel('u(x, t)', fontsize=12)
    axes[1].set_title('Time Snapshots', fontsize=12, fontweight='bold')
    axes[1].legend(fontsize=9)
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('heat_training_data.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # 创建训练对
    u_train, u_target, dt_tensor = create_cnn_training_pairs(solution_train, dt)
    print(f"  Training pairs: {len(u_train)}")
    
    # ========================================================================
    # 2. 创建 CNN Flow Map 模型
    # ========================================================================
    print("\n[2] Creating CNN Flow Map model...")
    
    model = FlowMapCNN(
        in_channels=1,
        hidden_channels=32,
        num_layers=4,
        kernel_size=5,
        use_residual=True
    )
    
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Model architecture: FlowMapCNN")
    print(f"  Hidden channels: 32")
    print(f"  Number of layers: 4")
    print(f"  Kernel size: 5")
    print(f"  Total parameters: {n_params}")
    
    # ========================================================================
    # 3. 训练模型
    # ========================================================================
    print("\n[3] Training Flow Map model...")
    
    history = train_cnn_flowmap(
        model, u_train, u_target, dt_tensor,
        epochs=1000,
        lr=1e-3,
        print_every=100
    )
    
    # 绘制训练曲线
    plt.figure(figsize=(10, 4))
    plt.semilogy(history['loss'], 'b-', linewidth=2)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Loss (log scale)', fontsize=12)
    plt.title('Training Loss', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('heat_training_loss.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # ========================================================================
    # 4. 测试长期预测
    # ========================================================================
    print("\n[4] Testing long-term prediction...")
    
    # 预测更长时间
    n_test_steps = 1000  # 预测1000步
    
    model.eval()
    with torch.no_grad():
        # 初始条件
        u0 = torch.FloatTensor(solution_train[0:1, np.newaxis, :])  # [1, 1, nx]
        
        # 多步预测
        predictions = [u0.squeeze().numpy()]
        u = u0
        
        for _ in range(n_test_steps):
            u = model(u, dt)
            predictions.append(u.squeeze().numpy())
        
        predictions = np.array(predictions)  # [n_steps + 1, nx]
    
    # 生成真实解进行对比
    t_test = np.arange(n_test_steps + 1) * dt
    solution_exact = analytical_solution(x, t_test, alpha)
    
    print(f"  Prediction steps: {n_test_steps}")
    print(f"  Prediction time: {t_test[-1]:.3f}")
    print(f"  Predictions shape: {predictions.shape}")
    
    # ========================================================================
    # 5. 可视化结果
    # ========================================================================
    print("\n[5] Visualizing results...")
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    
    # 1. 预测结果等高线
    X, T = np.meshgrid(x, t_test)
    contour1 = axes[0, 0].contourf(X, T, predictions, levels=30, cmap='viridis')
    axes[0, 0].set_xlabel('x', fontsize=12)
    axes[0, 0].set_ylabel('t', fontsize=12)
    axes[0, 0].set_title('Flow Map Prediction', fontsize=12, fontweight='bold')
    plt.colorbar(contour1, ax=axes[0, 0])
    
    # 2. 解析解等高线
    contour2 = axes[0, 1].contourf(X, T, solution_exact, levels=30, cmap='viridis')
    axes[0, 1].set_xlabel('x', fontsize=12)
    axes[0, 1].set_ylabel('t', fontsize=12)
    axes[0, 1].set_title('Analytical Solution', fontsize=12, fontweight='bold')
    plt.colorbar(contour2, ax=axes[0, 1])
    
    # 3. 误差等高线
    error = np.abs(predictions - solution_exact)
    contour3 = axes[0, 2].contourf(X, T, error, levels=30, cmap='hot')
    axes[0, 2].set_xlabel('x', fontsize=12)
    axes[0, 2].set_ylabel('t', fontsize=12)
    axes[0, 2].set_title('Absolute Error', fontsize=12, fontweight='bold')
    plt.colorbar(contour3, ax=axes[0, 2])
    
    # 4. 时间快照对比
    test_snapshots = [0, n_test_steps//4, n_test_steps//2, 
                     3*n_test_steps//4, n_test_steps]
    colors = plt.cm.viridis(np.linspace(0, 1, len(test_snapshots)))
    
    for idx, color in zip(test_snapshots, colors):
        axes[1, 0].plot(x, predictions[idx], '--', color=color,
                       label=f't={t_test[idx]:.3f} (pred)', linewidth=2)
        axes[1, 0].plot(x, solution_exact[idx], '-', color=color,
                       alpha=0.5, linewidth=3)
    
    axes[1, 0].set_xlabel('x', fontsize=12)
    axes[1, 0].set_ylabel('u(x, t)', fontsize=12)
    axes[1, 0].set_title('Snapshots Comparison (solid=exact, dashed=pred)', 
                        fontsize=11, fontweight='bold')
    axes[1, 0].grid(True, alpha=0.3)
    
    # 5. 特定位置的时间演化
    x_indices = [nx//4, nx//2, 3*nx//4]
    
    for i, xi in enumerate(x_indices):
        axes[1, 1].plot(t_test, predictions[:, xi], '--', 
                       label=f'x={x[xi]:.2f} (pred)', linewidth=2)
        axes[1, 1].plot(t_test, solution_exact[:, xi], '-',
                       alpha=0.5, linewidth=3)
    
    axes[1, 1].set_xlabel('t', fontsize=12)
    axes[1, 1].set_ylabel('u(x, t)', fontsize=12)
    axes[1, 1].set_title('Time Evolution at Fixed x', fontsize=12, fontweight='bold')
    axes[1, 1].legend(fontsize=9)
    axes[1, 1].grid(True, alpha=0.3)
    
    # 6. L2误差随时间演化
    l2_error = np.sqrt(np.mean(error**2, axis=1))
    relative_l2 = l2_error / (np.sqrt(np.mean(solution_exact**2, axis=1)) + 1e-10)
    
    axes[1, 2].semilogy(t_test, l2_error, 'r-', linewidth=2, label='L2 Error')
    axes[1, 2].semilogy(t_test, relative_l2, 'b-', linewidth=2, label='Relative L2')
    axes[1, 2].set_xlabel('t', fontsize=12)
    axes[1, 2].set_ylabel('Error (log scale)', fontsize=12)
    axes[1, 2].set_title('Error Evolution', fontsize=12, fontweight='bold')
    axes[1, 2].legend(fontsize=10)
    axes[1, 2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('heat_flowmap_results.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # ========================================================================
    # 6. 误差统计
    # ========================================================================
    print("\n[6] Error statistics...")
    
    print(f"  Mean L2 Error: {np.mean(l2_error):.6e}")
    print(f"  Max L2 Error:  {np.max(l2_error):.6e}")
    print(f"  Final L2 Error: {l2_error[-1]:.6e}")
    print(f"  Mean Relative Error: {np.mean(relative_l2)*100:.4f}%")
    print(f"  Final Relative Error: {relative_l2[-1]*100:.4f}%")
    
    # ========================================================================
    # 7. 保存模型
    # ========================================================================
    print("\n[7] Saving model...")
    
    torch.save({
        'model_state_dict': model.state_dict(),
        'alpha': alpha,
        'dx': dx,
        'dt': dt,
        'nx': nx,
        'history': history
    }, 'heat_flowmap_model.pth')
    
    print("  Model saved to: heat_flowmap_model.pth")
    
    print("\n" + "=" * 70)
    print("Complete!".center(70))
    print("=" * 70)
    print("\nGenerated files:")
    print("  - heat_training_data.png")
    print("  - heat_training_loss.png")
    print("  - heat_flowmap_results.png")
    print("  - heat_flowmap_model.pth")


if __name__ == "__main__":
    main()
