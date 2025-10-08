#!/usr/bin/env python3
"""
基于物理信息神经网络(PINNs)的Vlasov-Poisson系统求解器
"""

import numpy as np
import matplotlib.pyplot as plt
import time
import warnings
warnings.filterwarnings('ignore')

import deepxde as dde

# 设置随机种子确保结果可重复
np.random.seed(42)
dde.config.set_random_seed(42)

# 配置matplotlib生成科学图表
plt.rcParams.update({
    'figure.figsize': (10, 6),
    'savefig.dpi': 300,
    'axes.grid': True,
    'grid.alpha': 0.3
})

# 物理参数(无量纲化)
L = 2.0 * np.pi          # 空间域长度
T = 20.0                 # 时间域长度
v_max = 6.0              # 最大速度
epsilon_0 = 1.0          # 真空介电常数
q = -1.0                 # 粒子电荷(电子)
m = 1.0                  # 粒子质量
k_mode = 1.0             # 扰动波数
amplitude = 0.1          # 扰动幅度
gamma_landau_theory = 0.1533  # 理论Landau阻尼率

# 计算域边界
x_min, x_max = 0.0, L
v_min, v_max = -v_max, v_max
t_min, t_max = 0.0, T

# 默认神经网络架构(可在main函数中修改)
default_layer_sizes = [3] + [50] * 4 + [1]  # 3输入 -> 4个隐藏层(每层50个神经元) -> 1输出
activation = "tanh"
initializer = "Glorot uniform"

def initial_distribution(x, v):
    """初始分布函数：带小幅扰动的Maxwellian分布"""
    f_maxwell = (1.0 / np.sqrt(2 * np.pi)) * np.exp(-v**2 / 2)
    perturbation = amplitude * np.cos(k_mode * x)
    return f_maxwell * (1.0 + perturbation)

def analytical_electric_field(x, t):
    """Landau阻尼的解析电场"""
    gamma = 0.153
    E0 = amplitude * k_mode
    
    # 检查是否使用TensorFlow后端
    if hasattr(x, 'dtype') and 'tensorflow' in str(type(x)):
        import tensorflow as tf
        return E0 * tf.exp(-gamma * t) * tf.sin(k_mode * x)
    elif hasattr(x, 'dtype') and 'torch' in str(type(x)):
        import torch
        return E0 * torch.exp(-gamma * t) * torch.sin(k_mode * x)
    else:
        return E0 * np.exp(-gamma * t) * np.sin(k_mode * x)

def vlasov_poisson_system(x, y):
    """Vlasov方程PDE系统"""
    x_pos, v_vel, t_time = x[:, 0:1], x[:, 1:2], x[:, 2:3]
    
    df_dt = dde.grad.jacobian(y, x, i=0, j=2)  # ∂f/∂t
    df_dx = dde.grad.jacobian(y, x, i=0, j=0)  # ∂f/∂x
    df_dv = dde.grad.jacobian(y, x, i=0, j=1)  # ∂f/∂v
    
    E = analytical_electric_field(x_pos, t_time)
    vlasov_residual = df_dt + v_vel * df_dx + (q * E / m) * df_dv
    
    return vlasov_residual

def initial_condition_f(x):
    """分布函数的初始条件"""
    return initial_distribution(x[:, 0:1], x[:, 1:2])

def create_model(layer_sizes=None):
    """创建和配置PINN模型"""
    if layer_sizes is None:
        layer_sizes = default_layer_sizes
    
    # 定义几何和时间域
    geom_phase = dde.geometry.Rectangle([x_min, v_min], [x_max, v_max])
    timedomain = dde.geometry.TimeDomain(t_min, t_max)
    geomtime_phase = dde.geometry.GeometryXTime(geom_phase, timedomain)
    
    # 定义初始条件
    ic_f = dde.icbc.IC(geomtime_phase, initial_condition_f, lambda _, on_initial: on_initial)
    
    # 创建PDE问题，精确控制采样点数量以避免警告
    pde_vlasov = dde.data.TimePDE(
        geomtime_phase,
        vlasov_poisson_system,
        [ic_f],
        num_domain=1400,    # 域内点数
        num_boundary=62,    # 边界点数(精确匹配以避免警告)
        num_initial=280,    # 初始条件点数
        num_test=1000,      # 测试点数(精确匹配)
        train_distribution="uniform"  # 使用均匀分布采样
    )
    
    # 创建神经网络和模型
    net = dde.nn.FNN(layer_sizes, activation, initializer)
    model = dde.Model(pde_vlasov, net)
    
    print(f"神经网络架构: {layer_sizes}")
    print(f"总参数数量: {sum(layer_sizes[i] * layer_sizes[i+1] + layer_sizes[i+1] for i in range(len(layer_sizes)-1)):,}")
    
    return model

def train_model(model, iterations=8000):
    """训练PINN模型"""
    print("开始训练PINN模型...")
    start_time = time.time()
    
    model.compile("adam", lr=0.001)
    losshistory, train_state = model.train(iterations=iterations)
    
    train_time = time.time() - start_time
    print(f"训练完成！耗时: {train_time:.1f}秒")
    
    return losshistory, train_state

def plot_training_loss(losshistory):
    """绘制训练损失收敛图"""
    plt.figure(figsize=(10, 6))
    plt.semilogy(losshistory.loss_train, 'b-', linewidth=2, label='Training Loss')
    if hasattr(losshistory, 'loss_test') and losshistory.loss_test is not None:
        plt.semilogy(losshistory.loss_test, 'r--', linewidth=2, label='Test Loss')
    plt.xlabel('Iterations')
    plt.ylabel('Loss (log scale)')
    plt.title('Training Loss Convergence')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig('training_loss.png', dpi=300, bbox_inches='tight')
    plt.show()

def predict_evolution(model, nt_snapshots=6):
    """预测和可视化分布函数演化"""
    nx_pred, nv_pred = 64, 64
    x_pred = np.linspace(x_min, x_max, nx_pred)
    v_pred = np.linspace(v_min, v_max, nv_pred)
    t_pred = np.linspace(t_min, t_max, nt_snapshots)
    
    X_pred, V_pred = np.meshgrid(x_pred, v_pred)
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()
    
    for i, t_snap in enumerate(t_pred):
        if i >= 6:
            break
            
        T_pred = np.full_like(X_pred, t_snap)
        points_pred = np.column_stack([
            X_pred.flatten(), 
            V_pred.flatten(), 
            T_pred.flatten()
        ])
        
        f_pred = model.predict(points_pred).reshape(X_pred.shape)
        
        ax = axes[i]
        im = ax.contourf(X_pred, V_pred, f_pred, levels=20, cmap='viridis')
        ax.set_xlabel('Position x (dimensionless)')
        ax.set_ylabel('Velocity v (dimensionless)')
        ax.set_title(f'Distribution f(x,v) at t = {t_snap:.2f}')
        ax.grid(True, alpha=0.3)
        
        cbar = plt.colorbar(im, ax=ax, shrink=0.8)
        cbar.set_label('f(x,v,t)')
    
    plt.tight_layout()
    plt.savefig('vlasov_evolution.png', dpi=300, bbox_inches='tight')
    plt.show()

def check_conservation_laws(model):
    """检查Vlasov-Poisson系统的守恒定律"""
    print("\n=== 守恒定律验证 ===")
    
    # 创建预测网格
    nx, nv = 64, 64
    x_pred = np.linspace(x_min, x_max, nx)
    v_pred = np.linspace(v_min, v_max, nv)
    X_pred, V_pred = np.meshgrid(x_pred, v_pred)
    
    # 检查守恒的时间点
    time_points = [0.0, T/4, T/2, 3*T/4, T]
    
    particle_numbers = []
    total_energies = []
    total_momenta = []
    
    dx = x_pred[1] - x_pred[0]
    dv = v_pred[1] - v_pred[0]
    
    for t_check in time_points:
        # 为此时间创建输入点
        T_pred = np.full_like(X_pred, t_check)
        points = np.column_stack([
            X_pred.flatten(),
            V_pred.flatten(), 
            T_pred.flatten()
        ])
        
        # 预测分布函数
        f_pred = model.predict(points).reshape(X_pred.shape)
        
        # 1. 粒子数守恒: N = ∫∫ f(x,v,t) dx dv
        particle_number = np.trapz(np.trapz(f_pred, v_pred, axis=0), x_pred)
        particle_numbers.append(particle_number)
        
        # 2. 总动量: P = ∫∫ m*v*f(x,v,t) dx dv
        momentum = m * np.trapz(np.trapz(f_pred * V_pred, v_pred, axis=0), x_pred)
        total_momenta.append(momentum)
        
        # 3. 动能: E_kinetic = ∫∫ (1/2)*m*v²*f(x,v,t) dx dv
        kinetic_energy = 0.5 * m * np.trapz(np.trapz(f_pred * V_pred**2, v_pred, axis=0), x_pred)
        
        # 4. 电场能量(简化近似): E_field = (1/2)*ε₀*∫ E²(x,t) dx
        x_field = x_pred
        E_field = analytical_electric_field(x_field, t_check)
        electric_energy = 0.5 * epsilon_0 * np.trapz(E_field**2, x_field)
        
        total_energy = kinetic_energy + electric_energy
        total_energies.append(total_energy)
        
        print(f"t = {t_check:6.2f}: N = {particle_number:8.6f}, P = {momentum:8.6f}, E = {total_energy:8.6f}")
    
    # 计算守恒误差
    initial_N = particle_numbers[0]
    initial_P = total_momenta[0]
    initial_E = total_energies[0]
    
    N_error = max(abs(np.array(particle_numbers) - initial_N)) / abs(initial_N) * 100
    P_error = max(abs(np.array(total_momenta) - initial_P)) / (abs(initial_P) + 1e-10) * 100
    E_error = max(abs(np.array(total_energies) - initial_E)) / abs(initial_E) * 100
    
    print(f"\n守恒误差:")
    print(f"粒子数: {N_error:.3f}%")
    print(f"总动量: {P_error:.3f}%")
    print(f"总能量: {E_error:.3f}%")
    
    # 绘制守恒量图
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))
    
    ax1.plot(time_points, particle_numbers, 'bo-', linewidth=2, markersize=6)
    ax1.set_xlabel('Time t')
    ax1.set_ylabel('Particle Number N')
    ax1.set_title('Particle Number Conservation')
    ax1.grid(True, alpha=0.3)
    
    ax2.plot(time_points, total_momenta, 'ro-', linewidth=2, markersize=6)
    ax2.set_xlabel('Time t')
    ax2.set_ylabel('Total Momentum P')
    ax2.set_title('Momentum Conservation')
    ax2.grid(True, alpha=0.3)
    
    ax3.plot(time_points, total_energies, 'go-', linewidth=2, markersize=6)
    ax3.set_xlabel('Time t')
    ax3.set_ylabel('Total Energy E')
    ax3.set_title('Energy Conservation')
    ax3.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('conservation_laws.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # 返回守恒质量评估
    if N_error < 1.0 and P_error < 1.0 and E_error < 5.0:
        print("\n✅ 守恒定律保持良好!")
        return True
    else:
        print("\n⚠️  检测到守恒定律违背!")
        print("考虑调整网络架构或训练参数。")
        return False

def main(iterations=8000, layer_sizes=None):
    """
    主函数: 求解Vlasov-Poisson方程
    
    参数:
        iterations (int): 训练迭代次数 (默认: 8000)
        layer_sizes (list): 神经网络层数配置 (默认: [3] + [100]*4 + [1])
    """
    print("=" * 60)
    print("       PINN求解Vlasov-Poisson方程系统")
    print("=" * 60)
    print(f"物理参数设置:")
    print(f"空间域: x ∈ [0, {L:.2f}], 速度域: v ∈ [-{v_max}, {v_max}], 时间域: t ∈ [0, {T}]")
    print(f"扰动幅度: {amplitude}")
    print(f"波数: {k_mode}")
    print(f"训练迭代次数: {iterations}")
    
    # 创建模型
    if layer_sizes is None:
        layer_sizes = [3] + [100]*4 + [1]
    
    model = create_model(layer_sizes=layer_sizes)
    print("模型创建成功")
    
    # 训练模型
    losshistory, train_state = train_model(model, iterations=iterations)
    
    # 绘制训练结果
    plot_training_loss(losshistory)
    
    # 检查守恒定律
    conservation_ok = check_conservation_laws(model)
    
    # 预测演化
    predict_evolution(model)
    
    if conservation_ok:
        print("\n🎉 仿真成功完成，守恒定律保持良好!")
    else:
        print("\n⚠️  仿真完成但守恒定律可能被违背!")
        print("建议使用不同参数重新训练。")

if __name__ == "__main__":
    import sys
    
    # 参数设置
    iterations = 8000  
    layer_sizes = [3] + [300]*12 + [1]  # 5层隐藏层，每层300个神经元
    
    print("\n可用的自定义选项:")
    print("1. 修改 iterations 变量来调整训练迭代次数")
    print("2. 修改 layer_sizes 变量来自定义神经网络架构")
    print("3. 或通过命令行传递迭代次数: python vlasov_poisson_solver.py <iterations>")
    print(f"\n当前设置: iterations={iterations}, layer_sizes={layer_sizes or '[3, 100, 100, 100, 100, 1]'}")
    
    main(iterations=iterations, layer_sizes=layer_sizes)