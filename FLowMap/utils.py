"""
Flow Map Learning Utilities

Helper functions for data generation, training, and visualization.

Author: AI4CFD Project
Date: 2026-01
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from typing import Tuple, List, Optional, Callable
from scipy.integrate import solve_ivp


# ============================================================================
# 动力系统定义
# ============================================================================

def lorenz_system(t: float, state: np.ndarray, 
                  sigma: float = 10.0, 
                  rho: float = 28.0, 
                  beta: float = 8.0/3.0) -> np.ndarray:
    """
    Lorenz 混沌系统
    
    dx/dt = σ(y - x)
    dy/dt = x(ρ - z) - y
    dz/dt = xy - βz
    
    Args:
        t: 时间（未使用，自治系统）
        state: [x, y, z]
        sigma, rho, beta: Lorenz 参数
    
    Returns:
        dstate/dt
    """
    x, y, z = state
    return np.array([
        sigma * (y - x),
        x * (rho - z) - y,
        x * y - beta * z
    ])


def van_der_pol(t: float, state: np.ndarray, mu: float = 1.0) -> np.ndarray:
    """
    Van der Pol 振荡器
    
    dx/dt = y
    dy/dt = μ(1 - x²)y - x
    """
    x, y = state
    return np.array([y, mu * (1 - x**2) * y - x])


def pendulum(t: float, state: np.ndarray, 
             g: float = 9.81, L: float = 1.0) -> np.ndarray:
    """
    单摆系统
    
    dθ/dt = ω
    dω/dt = -(g/L)sin(θ)
    """
    theta, omega = state
    return np.array([omega, -(g/L) * np.sin(theta)])


def duffing_oscillator(t: float, state: np.ndarray,
                       alpha: float = 1.0,
                       beta: float = 1.0,
                       delta: float = 0.1) -> np.ndarray:
    """
    Duffing 振荡器（非线性弹簧）
    
    dx/dt = y
    dy/dt = -δy - αx - βx³
    """
    x, y = state
    return np.array([y, -delta * y - alpha * x - beta * x**3])


# ============================================================================
# 数据生成
# ============================================================================

def generate_trajectory(system_func: Callable,
                       x0: np.ndarray,
                       t_span: Tuple[float, float],
                       dt: float,
                       method: str = 'RK45',
                       **system_params) -> Tuple[np.ndarray, np.ndarray]:
    """
    生成动力系统轨迹
    
    Args:
        system_func: 动力系统函数 f(t, x, **params)
        x0: 初始条件
        t_span: 时间范围 (t0, tf)
        dt: 时间步长
        method: 积分方法
        **system_params: 系统参数
    
    Returns:
        t_eval: 时间点
        trajectory: 轨迹 [n_steps, state_dim]
    """
    t_eval = np.arange(t_span[0], t_span[1] + dt, dt)
    
    # 包装系统函数以包含参数
    def wrapped_func(t, x):
        return system_func(t, x, **system_params)
    
    sol = solve_ivp(wrapped_func, t_span, x0, method=method, 
                    t_eval=t_eval, dense_output=True)
    
    return sol.t, sol.y.T


def generate_multiple_trajectories(system_func: Callable,
                                  x0_list: List[np.ndarray],
                                  t_span: Tuple[float, float],
                                  dt: float,
                                  **system_params) -> Tuple[np.ndarray, np.ndarray]:
    """
    生成多条轨迹
    
    Returns:
        all_t: 时间点
        all_trajectories: [n_traj, n_steps, state_dim]
    """
    trajectories = []
    
    for x0 in x0_list:
        t, traj = generate_trajectory(system_func, x0, t_span, dt, **system_params)
        trajectories.append(traj)
    
    return t, np.stack(trajectories)


def create_training_pairs(trajectory: np.ndarray, 
                         dt: float,
                         stride: int = 1) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    从轨迹创建训练对 (x_n, x_{n+1})
    
    Args:
        trajectory: 轨迹数据 [n_steps, state_dim]
        dt: 时间步长
        stride: 采样步长
    
    Returns:
        x_current: 当前状态 [n_pairs, state_dim]
        x_next: 下一状态 [n_pairs, state_dim]
        dt_tensor: 时间步长 [n_pairs, 1]
    """
    n_steps = len(trajectory)
    indices = range(0, n_steps - 1, stride)
    
    x_current = trajectory[list(indices)]
    x_next = trajectory[list(i + 1 for i in indices)]
    dt_tensor = np.ones((len(indices), 1)) * dt
    
    return (torch.FloatTensor(x_current),
            torch.FloatTensor(x_next),
            torch.FloatTensor(dt_tensor))


def create_multistep_training_pairs(trajectory: np.ndarray,
                                   dt: float,
                                   max_steps: int = 5) -> Tuple[torch.Tensor, ...]:
    """
    创建多步训练对，用于多尺度训练
    
    Args:
        trajectory: 轨迹
        dt: 基础时间步长
        max_steps: 最大步数
    
    Returns:
        多步训练对
    """
    pairs = []
    
    for k in range(1, max_steps + 1):
        n_pairs = len(trajectory) - k
        if n_pairs <= 0:
            break
        
        x_current = trajectory[:n_pairs]
        x_next = trajectory[k:k + n_pairs]
        dt_k = np.ones((n_pairs, 1)) * k * dt
        
        pairs.append({
            'x_current': torch.FloatTensor(x_current),
            'x_next': torch.FloatTensor(x_next),
            'dt': torch.FloatTensor(dt_k)
        })
    
    return pairs


# ============================================================================
# PDE 数据生成
# ============================================================================

def heat_equation_1d(u0: np.ndarray, 
                     alpha: float,
                     dx: float,
                     dt: float,
                     n_steps: int) -> np.ndarray:
    """
    1D 热传导方程有限差分求解
    
    ∂u/∂t = α * ∂²u/∂x²
    
    使用 FTCS (Forward Time Central Space) 格式
    
    Args:
        u0: 初始条件 [nx]
        alpha: 热扩散系数
        dx: 空间步长
        dt: 时间步长
        n_steps: 时间步数
    
    Returns:
        solution: [n_steps + 1, nx]
    """
    nx = len(u0)
    r = alpha * dt / dx**2
    
    if r > 0.5:
        print(f"Warning: CFL condition violated (r = {r:.3f} > 0.5)")
    
    solution = np.zeros((n_steps + 1, nx))
    solution[0] = u0.copy()
    
    for n in range(n_steps):
        u = solution[n].copy()
        # 内部点
        solution[n+1, 1:-1] = u[1:-1] + r * (u[2:] - 2*u[1:-1] + u[:-2])
        # 边界条件 (Dirichlet: u = 0)
        solution[n+1, 0] = 0
        solution[n+1, -1] = 0
    
    return solution


def advection_equation_1d(u0: np.ndarray,
                         c: float,
                         dx: float,
                         dt: float,
                         n_steps: int) -> np.ndarray:
    """
    1D 对流方程有限差分求解
    
    ∂u/∂t + c * ∂u/∂x = 0
    
    使用迎风格式
    """
    nx = len(u0)
    solution = np.zeros((n_steps + 1, nx))
    solution[0] = u0.copy()
    
    CFL = abs(c) * dt / dx
    if CFL > 1:
        print(f"Warning: CFL condition violated (CFL = {CFL:.3f} > 1)")
    
    for n in range(n_steps):
        u = solution[n].copy()
        if c > 0:
            # 迎风格式（左）
            solution[n+1, 1:] = u[1:] - c * dt / dx * (u[1:] - u[:-1])
            solution[n+1, 0] = u[0]  # 边界
        else:
            # 迎风格式（右）
            solution[n+1, :-1] = u[:-1] - c * dt / dx * (u[1:] - u[:-1])
            solution[n+1, -1] = u[-1]  # 边界
    
    return solution


def burgers_equation_1d(u0: np.ndarray,
                       nu: float,
                       dx: float,
                       dt: float,
                       n_steps: int) -> np.ndarray:
    """
    1D Burgers 方程有限差分求解
    
    ∂u/∂t + u * ∂u/∂x = ν * ∂²u/∂x²
    """
    nx = len(u0)
    solution = np.zeros((n_steps + 1, nx))
    solution[0] = u0.copy()
    
    for n in range(n_steps):
        u = solution[n].copy()
        
        # 对流项（迎风）
        u_pos = np.maximum(u, 0)
        u_neg = np.minimum(u, 0)
        
        conv = np.zeros_like(u)
        conv[1:-1] = (u_pos[1:-1] * (u[1:-1] - u[:-2]) / dx +
                     u_neg[1:-1] * (u[2:] - u[1:-1]) / dx)
        
        # 扩散项
        diff = np.zeros_like(u)
        diff[1:-1] = nu * (u[2:] - 2*u[1:-1] + u[:-2]) / dx**2
        
        solution[n+1] = u - dt * conv + dt * diff
        solution[n+1, 0] = 0
        solution[n+1, -1] = 0
    
    return solution


# ============================================================================
# 训练工具
# ============================================================================

def train_flowmap(model,
                 x_train: torch.Tensor,
                 x_target: torch.Tensor,
                 dt: torch.Tensor,
                 epochs: int = 1000,
                 lr: float = 1e-3,
                 print_every: int = 100,
                 device: str = 'cpu') -> dict:
    """
    训练 Flow Map 模型
    
    Args:
        model: Flow Map 模型
        x_train: 训练输入
        x_target: 目标输出
        dt: 时间步长
        epochs: 训练轮数
        lr: 学习率
        print_every: 打印频率
        device: 设备
    
    Returns:
        history: 训练历史
    """
    model = model.to(device)
    x_train = x_train.to(device)
    x_target = x_target.to(device)
    dt = dt.to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=200, verbose=False
    )
    
    history = {'loss': []}
    
    print("=" * 60)
    print("Training Flow Map Model".center(60))
    print("=" * 60)
    
    for epoch in range(epochs):
        optimizer.zero_grad()
        
        # 前向传播
        x_pred = model(x_train, dt)
        
        # 损失
        loss = torch.mean((x_pred - x_target) ** 2)
        
        # 反向传播
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step(loss)
        
        history['loss'].append(loss.item())
        
        if (epoch + 1) % print_every == 0 or epoch == 0:
            print(f"Epoch {epoch+1:5d} | Loss: {loss.item():.6e}")
    
    print("=" * 60)
    print("Training completed!".center(60))
    print("=" * 60)
    
    return history


def train_multistep(model,
                   trajectory: np.ndarray,
                   dt: float,
                   epochs: int = 1000,
                   k_max: int = 5,
                   lr: float = 1e-3,
                   device: str = 'cpu') -> dict:
    """
    多步训练（同时优化单步和多步预测）
    
    这可以提高长期预测的稳定性
    """
    model = model.to(device)
    
    # 创建多步训练对
    pairs = create_multistep_training_pairs(trajectory, dt, k_max)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    history = {'loss': [], 'loss_by_k': {k: [] for k in range(1, k_max + 1)}}
    
    print("=" * 60)
    print("Multi-step Training".center(60))
    print("=" * 60)
    
    for epoch in range(epochs):
        optimizer.zero_grad()
        total_loss = 0
        
        for k, pair in enumerate(pairs, 1):
            x_curr = pair['x_current'].to(device)
            x_next_true = pair['x_next'].to(device)
            dt_k = pair['dt'].to(device)
            
            # 迭代 k 步预测
            x_pred = x_curr
            for _ in range(k):
                x_pred = model(x_pred, dt)
            
            loss_k = torch.mean((x_pred - x_next_true) ** 2)
            total_loss = total_loss + loss_k / k  # 加权
            
            history['loss_by_k'][k].append(loss_k.item())
        
        total_loss.backward()
        optimizer.step()
        history['loss'].append(total_loss.item())
        
        if (epoch + 1) % 200 == 0:
            print(f"Epoch {epoch+1:5d} | Total Loss: {total_loss.item():.6e}")
    
    return history


# ============================================================================
# 可视化
# ============================================================================

def plot_trajectory_2d(trajectory: np.ndarray,
                      pred_trajectory: Optional[np.ndarray] = None,
                      title: str = 'Phase Space',
                      dim1: int = 0, dim2: int = 1):
    """绘制2D相空间轨迹"""
    plt.figure(figsize=(10, 8))
    
    plt.plot(trajectory[:, dim1], trajectory[:, dim2], 
             'b-', linewidth=1.5, label='True', alpha=0.7)
    
    if pred_trajectory is not None:
        plt.plot(pred_trajectory[:, dim1], pred_trajectory[:, dim2],
                'r--', linewidth=1.5, label='Predicted', alpha=0.7)
    
    plt.xlabel(f'$x_{dim1+1}$', fontsize=12)
    plt.ylabel(f'$x_{dim2+1}$', fontsize=12)
    plt.title(title, fontsize=14, fontweight='bold')
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


def plot_trajectory_3d(trajectory: np.ndarray,
                      pred_trajectory: Optional[np.ndarray] = None,
                      title: str = '3D Phase Space'):
    """绘制3D相空间轨迹"""
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    ax.plot(trajectory[:, 0], trajectory[:, 1], trajectory[:, 2],
           'b-', linewidth=1, label='True', alpha=0.7)
    
    if pred_trajectory is not None:
        ax.plot(pred_trajectory[:, 0], pred_trajectory[:, 1], pred_trajectory[:, 2],
               'r--', linewidth=1, label='Predicted', alpha=0.7)
    
    ax.set_xlabel('$x$', fontsize=12)
    ax.set_ylabel('$y$', fontsize=12)
    ax.set_zlabel('$z$', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    plt.tight_layout()
    plt.show()


def plot_time_series(t: np.ndarray,
                    trajectory: np.ndarray,
                    pred_trajectory: Optional[np.ndarray] = None,
                    labels: Optional[List[str]] = None):
    """绘制时间序列"""
    n_dims = trajectory.shape[1]
    
    if labels is None:
        labels = [f'$x_{i+1}$' for i in range(n_dims)]
    
    fig, axes = plt.subplots(n_dims, 1, figsize=(12, 3*n_dims), sharex=True)
    if n_dims == 1:
        axes = [axes]
    
    for i, ax in enumerate(axes):
        ax.plot(t, trajectory[:, i], 'b-', linewidth=2, label='True', alpha=0.7)
        if pred_trajectory is not None:
            ax.plot(t[:len(pred_trajectory)], pred_trajectory[:, i],
                   'r--', linewidth=2, label='Predicted', alpha=0.7)
        ax.set_ylabel(labels[i], fontsize=12)
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
    
    axes[-1].set_xlabel('Time', fontsize=12)
    plt.tight_layout()
    plt.show()


def plot_pde_evolution(solution: np.ndarray,
                      x: np.ndarray,
                      t: np.ndarray,
                      title: str = 'PDE Solution'):
    """绘制PDE时空演化"""
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    
    # 1. 等高线图
    X, T = np.meshgrid(x, t)
    contour = axes[0].contourf(X, T, solution, levels=30, cmap='viridis')
    axes[0].set_xlabel('x', fontsize=12)
    axes[0].set_ylabel('t', fontsize=12)
    axes[0].set_title('Solution Contour', fontsize=12, fontweight='bold')
    plt.colorbar(contour, ax=axes[0])
    
    # 2. 时间快照
    n_snapshots = 5
    indices = np.linspace(0, len(t) - 1, n_snapshots, dtype=int)
    colors = plt.cm.viridis(np.linspace(0, 1, n_snapshots))
    
    for idx, color in zip(indices, colors):
        axes[1].plot(x, solution[idx], color=color, 
                    label=f't = {t[idx]:.2f}', linewidth=2)
    
    axes[1].set_xlabel('x', fontsize=12)
    axes[1].set_ylabel('u(x, t)', fontsize=12)
    axes[1].set_title('Time Snapshots', fontsize=12, fontweight='bold')
    axes[1].legend(fontsize=9)
    axes[1].grid(True, alpha=0.3)
    
    # 3. 3D 表面图
    ax3d = fig.add_subplot(133, projection='3d')
    ax3d.plot_surface(X, T, solution, cmap='viridis', alpha=0.9)
    ax3d.set_xlabel('x', fontsize=11)
    ax3d.set_ylabel('t', fontsize=11)
    ax3d.set_zlabel('u', fontsize=11)
    ax3d.set_title('3D Surface', fontsize=12, fontweight='bold')
    
    # 移除原来的 axes[2]
    axes[2].remove()
    
    plt.tight_layout()
    plt.show()


def plot_error_analysis(true_traj: np.ndarray,
                       pred_traj: np.ndarray,
                       t: np.ndarray):
    """误差分析可视化"""
    error = np.abs(true_traj - pred_traj)
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # 1. 逐维度误差
    for i in range(error.shape[1]):
        axes[0, 0].plot(t[:len(error)], error[:, i], 
                       label=f'Dim {i+1}', linewidth=2, alpha=0.8)
    axes[0, 0].set_xlabel('Time', fontsize=12)
    axes[0, 0].set_ylabel('Absolute Error', fontsize=12)
    axes[0, 0].set_title('Error per Dimension', fontsize=12, fontweight='bold')
    axes[0, 0].legend(fontsize=10)
    axes[0, 0].grid(True, alpha=0.3)
    
    # 2. 总误差
    total_error = np.linalg.norm(error, axis=1)
    axes[0, 1].semilogy(t[:len(error)], total_error, 'r-', linewidth=2)
    axes[0, 1].set_xlabel('Time', fontsize=12)
    axes[0, 1].set_ylabel('Total Error (log)', fontsize=12)
    axes[0, 1].set_title('Total Error Evolution', fontsize=12, fontweight='bold')
    axes[0, 1].grid(True, alpha=0.3)
    
    # 3. 相对误差
    true_norm = np.linalg.norm(true_traj[:len(pred_traj)], axis=1) + 1e-10
    relative_error = total_error / true_norm * 100
    axes[1, 0].plot(t[:len(error)], relative_error, 'g-', linewidth=2)
    axes[1, 0].set_xlabel('Time', fontsize=12)
    axes[1, 0].set_ylabel('Relative Error (%)', fontsize=12)
    axes[1, 0].set_title('Relative Error', fontsize=12, fontweight='bold')
    axes[1, 0].grid(True, alpha=0.3)
    
    # 4. 误差统计
    axes[1, 1].boxplot([error[:, i] for i in range(error.shape[1])],
                      labels=[f'Dim {i+1}' for i in range(error.shape[1])])
    axes[1, 1].set_ylabel('Error', fontsize=12)
    axes[1, 1].set_title('Error Distribution', fontsize=12, fontweight='bold')
    axes[1, 1].grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.show()
    
    # 打印统计信息
    print("\n" + "=" * 50)
    print("Error Statistics".center(50))
    print("=" * 50)
    print(f"Mean Absolute Error: {np.mean(error):.6e}")
    print(f"Max Absolute Error:  {np.max(error):.6e}")
    print(f"Mean Relative Error: {np.mean(relative_error):.4f}%")
    print(f"Final Relative Error: {relative_error[-1]:.4f}%")
    print("=" * 50)
