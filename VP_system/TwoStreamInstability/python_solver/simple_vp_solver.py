#!/usr/bin/env python3
"""
简化的 Vlasov-Poisson 求解器
使用算子分裂法求解双流不稳定性问题
可以替代 HyPar 快速查看结果
"""

import numpy as np
import struct
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.animation import FuncAnimation
import os

print("="*70)
print("简化版 Vlasov-Poisson 求解器")
print("="*70)

# ==================== 读取配置 ====================
print("\n1. 读取配置文件...")

# 默认参数
nx, nv = 128, 128
n_iter = 25
dt = 0.2

# 从 solver.inp 读取
try:
    with open('solver.inp', 'r') as f:
        for line in f:
            if 'size' in line:
                parts = line.split()
                nx = int(parts[1])
                nv = int(parts[2])
            elif 'n_iter' in line:
                parts = line.split()
                n_iter = int(parts[1])
            elif 'dt' in line:
                parts = line.split()
                dt = float(parts[1])
    print(f"   网格: {nx} × {nv}")
    print(f"   时间步数: {n_iter}")
    print(f"   时间步长: {dt}")
except:
    print("   警告: 无法读取 solver.inp, 使用默认值")

# ==================== 读取初始条件 ====================
print("\n2. 读取初始条件...")

try:
    with open('initial.inp', 'rb') as f:
        # 读取网格
        x = np.array(struct.unpack('d' * nx, f.read(8 * nx)))
        v = np.array(struct.unpack('d' * nv, f.read(8 * nv)))
        
        # 读取初始分布
        f0_data = np.array(struct.unpack('d' * (nx * nv), f.read(8 * nx * nv)))
        f0 = f0_data.reshape((nv, nx))
    
    print(f"   ✓ 初始条件加载成功")
    print(f"   x 范围: [{x.min():.2f}, {x.max():.2f}]")
    print(f"   v 范围: [{v.min():.2f}, {v.max():.2f}]")
    print(f"   f 范围: [{f0.min():.2e}, {f0.max():.2e}]")
except Exception as e:
    print(f"   ✗ 错误: 无法读取初始条件 - {e}")
    exit(1)

dx = x[1] - x[0]
dv = v[1] - v[0]

# ==================== 定义求解器 ====================
print("\n3. 初始化求解器...")

def compute_electric_field(f, v, x):
    """计算电场"""
    # 计算电子密度: n_e = ∫ f dv
    n_e = np.trapz(f, v, axis=0)
    
    # 电荷密度偏差: ρ = n_e - 1
    rho = n_e - 1.0
    
    # 通过积分求解电场: dE/dx = ρ
    E = np.zeros_like(n_e)
    E[0] = 0  # 边界条件
    for i in range(1, len(E)):
        E[i] = E[i-1] + rho[i-1] * (x[i] - x[i-1])
    
    # 周期性边界：去除平均值
    E = E - np.mean(E)
    
    return E

def solve_vlasov_step(f, E, v, x, dt):
    """
    使用算子分裂法求解一个时间步
    
    Step 1: 对流步 ∂f/∂t + v·∂f/∂x = 0
    Step 2: 加速步 ∂f/∂t - E·∂f/∂v = 0
    """
    nv, nx = f.shape
    dx = x[1] - x[0]
    dv = v[1] - v[0]
    
    # Step 1: 对流步 (上风格式)
    f_temp = np.zeros_like(f)
    for j in range(nv):
        v_j = v[j]
        if v_j > 0:
            # 向右传播
            f_temp[j, :] = f[j, :] - v_j * dt / dx * (f[j, :] - np.roll(f[j, :], 1))
        else:
            # 向左传播
            f_temp[j, :] = f[j, :] - v_j * dt / dx * (np.roll(f[j, :], -1) - f[j, :])
    
    # Step 2: 加速步 (上风格式)
    f_new = np.zeros_like(f)
    for i in range(nx):
        E_i = E[i]
        if E_i > 0:
            # 速度增加
            for j in range(nv):
                if j > 0:
                    f_new[j, i] = f_temp[j, i] + E_i * dt / dv * (f_temp[j, i] - f_temp[j-1, i])
                else:
                    f_new[j, i] = f_temp[j, i]  # 边界
        else:
            # 速度减少
            for j in range(nv):
                if j < nv - 1:
                    f_new[j, i] = f_temp[j, i] + E_i * dt / dv * (f_temp[j+1, i] - f_temp[j, i])
                else:
                    f_new[j, i] = f_temp[j, i]  # 边界
    
    # 确保非负
    f_new = np.maximum(f_new, 0)
    
    return f_new

# ==================== 时间演化 ====================
print(f"\n4. 开始时间演化 ({n_iter} 步)...")

f = f0.copy()
f_history = [f.copy()]
E_history = []
times = [0]

print("\n进度:")
for n in range(n_iter):
    # 计算电场
    E = compute_electric_field(f, v, x)
    E_history.append(E)
    
    # 演化一步
    f = solve_vlasov_step(f, E, v, x, dt)
    
    # 保存
    f_history.append(f.copy())
    times.append((n+1) * dt)
    
    # 显示进度
    if (n + 1) % 5 == 0 or n == n_iter - 1:
        print(f"   步骤 {n+1}/{n_iter} (t={times[-1]:.2f})")

print("   ✓ 时间演化完成")

# ==================== 创建动画 ====================
print("\n5. 创建动画...")

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# 初始化
X, V = np.meshgrid(x, v)

def update(frame):
    """更新动画帧"""
    t = times[frame]
    f_current = f_history[frame]
    
    # 清除所有子图
    for ax in axes.flat:
        ax.clear()
    
    # 1. 相空间图
    im1 = axes[0, 0].contourf(X, V, f_current, levels=50, cmap='jet')
    axes[0, 0].set_xlabel('Position x', fontsize=12)
    axes[0, 0].set_ylabel('Velocity v', fontsize=12)
    axes[0, 0].set_title(f'Phase Space (t={t:.2f})', fontsize=14)
    
    # 2. 密度图
    density = np.trapz(f_current, v, axis=0)
    axes[0, 1].plot(x, density, 'b-', linewidth=2)
    axes[0, 1].set_xlabel('Position x', fontsize=12)
    axes[0, 1].set_ylabel('Density ρ(x)', fontsize=12)
    axes[0, 1].set_title(f'Spatial Density (t={t:.2f})', fontsize=14)
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].set_ylim([0, 2.5])
    
    # 3. 电场
    if frame > 0:
        E = E_history[frame-1]
        axes[1, 0].plot(x, E, 'r-', linewidth=2)
        axes[1, 0].set_xlabel('Position x', fontsize=12)
        axes[1, 0].set_ylabel('Electric Field E(x)', fontsize=12)
        axes[1, 0].set_title(f'Electric Field (t={t:.2f})', fontsize=14)
        axes[1, 0].grid(True, alpha=0.3)
        axes[1, 0].axhline(y=0, color='k', linestyle='--', alpha=0.3)
    
    # 4. 速度分布（在 x=L/2 处）
    x_mid_idx = nx // 2
    f_v = f_current[:, x_mid_idx]
    axes[1, 1].plot(v, f_v, 'g-', linewidth=2)
    axes[1, 1].set_xlabel('Velocity v', fontsize=12)
    axes[1, 1].set_ylabel(f'f(x={x[x_mid_idx]:.2f}, v)', fontsize=12)
    axes[1, 1].set_title(f'Velocity Distribution at x=L/2 (t={t:.2f})', fontsize=14)
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    return []

# 创建动画
print("   创建 GIF 动画...")
anim = FuncAnimation(fig, update, frames=len(times), 
                    interval=200, blit=False, repeat=True)

# 保存动画
gif_path = 'two_stream_instability.gif'
anim.save(gif_path, writer='pillow', fps=5, dpi=100)
print(f"   ✓ 动画已保存: {gif_path}")

plt.close()

# ==================== 创建诊断图 ====================
print("\n6. 创建诊断图...")

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# 计算物理量
electric_energy = []
kinetic_energy = []
total_mass = []

for i, f_snap in enumerate(f_history):
    density = np.trapz(f_snap, v, axis=0)
    rho = density - 1.0
    
    # 电场能量
    E_energy = np.trapz(rho**2, x)
    electric_energy.append(E_energy)
    
    # 动能
    V_mesh, X_mesh = np.meshgrid(v, x)
    K_energy = np.trapz(np.trapz(f_snap * V_mesh.T**2, v, axis=0), x)
    kinetic_energy.append(K_energy)
    
    # 总质量
    mass = np.trapz(density, x)
    total_mass.append(mass)

# 电场能量演化
axes[0, 0].semilogy(times, electric_energy, 'b-', linewidth=2)
axes[0, 0].set_xlabel('Time', fontsize=12)
axes[0, 0].set_ylabel('Electric Field Energy', fontsize=12)
axes[0, 0].set_title('Electric Field Energy Evolution', fontsize=14)
axes[0, 0].grid(True, alpha=0.3)

# 动能演化
axes[0, 1].plot(times, kinetic_energy, 'r-', linewidth=2)
axes[0, 1].set_xlabel('Time', fontsize=12)
axes[0, 1].set_ylabel('Kinetic Energy', fontsize=12)
axes[0, 1].set_title('Kinetic Energy Evolution', fontsize=14)
axes[0, 1].grid(True, alpha=0.3)

# 质量守恒
axes[1, 0].plot(times, total_mass, 'g-', linewidth=2)
axes[1, 0].set_xlabel('Time', fontsize=12)
axes[1, 0].set_ylabel('Total Mass', fontsize=12)
axes[1, 0].set_title('Mass Conservation Check', fontsize=14)
axes[1, 0].grid(True, alpha=0.3)

# 最终相空间
axes[1, 1].contourf(X, V, f_history[-1], levels=50, cmap='jet')
axes[1, 1].set_xlabel('Position x', fontsize=12)
axes[1, 1].set_ylabel('Velocity v', fontsize=12)
axes[1, 1].set_title(f'Final Phase Space (t={times[-1]:.2f})', fontsize=14)

plt.tight_layout()
diag_path = 'diagnostics.png'
plt.savefig(diag_path, dpi=150, bbox_inches='tight')
print(f"   ✓ 诊断图已保存: {diag_path}")

plt.close()

# ==================== 完成 ====================
print("\n" + "="*70)
print("✓ 所有任务完成！")
print("="*70)
print("\n生成的文件:")
print(f"  - {gif_path}  (相空间演化动画)")
print(f"  - {diag_path}      (物理量诊断图)")
print("\n查看结果:")
print(f"  $ open {gif_path}")
print(f"  $ open {diag_path}")
print("\n提示: 要看到更完整的双涡旋演化，可以:")
print("  1. 修改 solver.inp 中的 n_iter 为 100")
print("  2. 重新运行: python3 simple_vp_solver.py")
print()
