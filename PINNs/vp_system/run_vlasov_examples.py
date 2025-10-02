#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Vlasov-Poisson方程组求解器使用示例

这个脚本演示了如何使用VlasovPoissonSolver求解6维Vlasov-Poisson系统
包含多个经典等离子体物理案例
"""

from vlasov_poisson_solver import VlasovPoissonSolver
import numpy as np
import matplotlib.pyplot as plt

def example_landau_damping():
    """Landau阻尼示例 - 等离子体物理经典问题"""
    print("=" * 70)
    print("Landau阻尼示例：研究等离子体中的波阻尼现象")
    print("=" * 70)
    
    # 创建Landau阻尼求解器
    solver = VlasovPoissonSolver(
        x_domain=(-np.pi, np.pi),  # 使用2π周期域
        v_domain=(-4.0, 4.0),
        time_domain=(0.0, 10.0),   # 较长时间观察阻尼
        case="landau_damping"
    )
    
    # 设置问题
    solver.setup_geometry_and_conditions()
    
    # 可视化初始条件
    solver.visualize_initial_conditions()
    
    # 创建模型 (使用较少参数以减少计算时间)
    solver.create_model(
        num_domain=5000,
        num_boundary=400,
        num_initial=400,
        layer_sizes=[3, 100, 100, 100, 2],
        activation="tanh"
    )
    
    # 训练模型
    solver.train(
        adam_iterations=8000,
        adam_lr=0.001,
        use_lbfgs=True,
        weights_pde=[1.0, 0.1]  # Vlasov方程权重更大
    )
    
    # 分析结果
    solver.visualize_phase_space_evolution(times=[0, 2, 5, 8])
    solver.analyze_plasma_dynamics()
    
    return solver

def example_two_stream_instability():
    """双流不稳定性示例"""
    print("=" * 70)
    print("双流不稳定性示例：两束反向电子束的不稳定性")
    print("=" * 70)
    
    # 创建双流不稳定性求解器
    solver = VlasovPoissonSolver(
        x_domain=(-1.0, 1.0),
        v_domain=(-3.0, 3.0),
        time_domain=(0.0, 5.0),
        case="two_stream"
    )
    
    solver.setup_geometry_and_conditions()
    solver.visualize_initial_conditions()
    
    # 创建模型
    solver.create_model(
        num_domain=4000,
        num_boundary=300,
        num_initial=300,
        layer_sizes=[3, 80, 80, 80, 2]
    )
    
    # 训练
    solver.train(
        adam_iterations=6000,
        adam_lr=0.0012,
        use_lbfgs=True
    )
    
    # 可视化结果
    solver.visualize_phase_space_evolution(times=[0, 1, 2.5, 4])
    solver.analyze_plasma_dynamics(time_points=15)
    
    return solver

def example_bump_on_tail():
    """Bump-on-tail不稳定性示例"""
    print("=" * 70)
    print("Bump-on-tail不稳定性示例：高能电子尾部驱动的不稳定性")
    print("=" * 70)
    
    solver = VlasovPoissonSolver(
        x_domain=(-2.0, 2.0),
        v_domain=(-2.0, 5.0),  # 扩展速度域以包含高能尾部
        time_domain=(0.0, 8.0),
        case="bump_on_tail"
    )
    
    solver.setup_geometry_and_conditions()
    solver.visualize_initial_conditions()
    
    solver.create_model(
        num_domain=4500,
        num_boundary=350,
        num_initial=350,
        layer_sizes=[3, 90, 90, 90, 2]
    )
    
    solver.train(
        adam_iterations=7000,
        adam_lr=0.0008,
        use_lbfgs=True
    )
    
    solver.visualize_phase_space_evolution(times=[0, 2, 4, 6])
    solver.analyze_plasma_dynamics()
    
    return solver

def example_parameter_study():
    """参数研究：不同波数k对Landau阻尼的影响"""
    print("=" * 70)
    print("参数研究：不同波数k对Landau阻尼率的影响")
    print("=" * 70)
    
    k_values = [0.3, 0.5, 0.8]  # 不同的波数
    damping_rates = []
    
    plt.figure(figsize=(15, 5))
    
    for i, k_mode in enumerate(k_values):
        print(f"\n--- 求解波数 k = {k_mode} 的情况 ---")
        
        # 创建自定义求解器
        class CustomLandauSolver(VlasovPoissonSolver):
            def __init__(self, k_mode_custom):
                super().__init__(
                    x_domain=(-np.pi, np.pi),
                    v_domain=(-3.0, 3.0),
                    time_domain=(0.0, 6.0),
                    case="landau_damping"
                )
                self.k_mode = k_mode_custom  # 覆盖默认波数
        
        solver = CustomLandauSolver(k_mode)
        solver.setup_geometry_and_conditions()
        
        # 快速训练
        solver.create_model(
            num_domain=3000,
            num_boundary=250,
            num_initial=250,
            layer_sizes=[3, 70, 70, 2]
        )
        
        solver.train(
            adam_iterations=4000,
            adam_lr=0.001,
            use_lbfgs=False  # 跳过L-BFGS以节省时间
        )
        
        # 分析电场演化
        times = np.linspace(0, 6, 30)
        electric_fields = []
        
        for t in times:
            # 计算中心点电场
            x_center, v_center = 0.0, 0.0
            dx = 0.01
            
            phi_left = solver.predict(np.array([[x_center - dx, v_center, t]]))[0, 1]
            phi_right = solver.predict(np.array([[x_center + dx, v_center, t]]))[0, 1]
            E_field = -(phi_right - phi_left) / (2 * dx)
            electric_fields.append(E_field)
        
        electric_fields = np.array(electric_fields)
        
        # 绘制电场演化
        plt.subplot(1, 3, i+1)
        plt.semilogy(times, np.abs(electric_fields), 'b-', linewidth=2, label='|E(t)|')
        plt.xlabel('时间 t')
        plt.ylabel('|电场|')
        plt.title(f'k = {k_mode}')
        plt.grid(True, alpha=0.3)
        plt.legend()
        
        # 估算阻尼率 (简单线性拟合)
        if len(times) > 10:
            mid_start = len(times) // 4
            mid_end = 3 * len(times) // 4
            log_E = np.log(np.abs(electric_fields[mid_start:mid_end]) + 1e-10)
            time_fit = times[mid_start:mid_end]
            
            # 线性拟合 log|E| = -γt + const
            poly_fit = np.polyfit(time_fit, log_E, 1)
            damping_rate = -poly_fit[0]
            damping_rates.append(damping_rate)
            
            plt.plot(time_fit, np.exp(poly_fit[1] + poly_fit[0] * time_fit), 
                    'r--', alpha=0.8, label=f'γ≈{damping_rate:.3f}')
            plt.legend()
    
    plt.tight_layout()
    plt.show()
    
    # 总结参数研究结果
    print("\n📊 参数研究总结:")
    print("波数 k\t阻尼率 γ")
    print("-" * 20)
    for k, gamma in zip(k_values, damping_rates):
        print(f"{k:.1f}\t{gamma:.4f}")

def example_advanced_analysis():
    """高级分析示例：深入研究相空间结构"""
    print("=" * 70)
    print("高级分析示例：深入研究Vlasov-Poisson系统的相空间结构")
    print("=" * 70)
    
    # 创建求解器
    solver = VlasovPoissonSolver(
        x_domain=(-1.0, 1.0),
        v_domain=(-2.5, 2.5),
        time_domain=(0.0, 4.0),
        case="landau_damping"
    )
    
    solver.setup_geometry_and_conditions()
    
    # 训练模型
    solver.create_model(
        num_domain=3500,
        num_boundary=280,
        num_initial=280,
        layer_sizes=[3, 80, 80, 80, 2]
    )
    
    solver.train(adam_iterations=5000, adam_lr=0.001, use_lbfgs=True)
    
    # 高级分析
    print("\n🔍 进行高级相空间分析...")
    
    # 1. 相空间轨迹跟踪
    plt.figure(figsize=(18, 12))
    
    # 选择几个代表性的初始相空间点
    initial_points = [
        (-0.5, -1.5), (-0.5, 0.0), (-0.5, 1.5),
        (0.0, -1.0), (0.0, 0.0), (0.0, 1.0),
        (0.5, -1.5), (0.5, 0.0), (0.5, 1.5)
    ]
    
    times = np.linspace(0, 4, 40)
    
    # 2. 速度分布演化
    plt.subplot(2, 3, 1)
    x_fixed = 0.0  # 固定位置
    v_points = np.linspace(-2.5, 2.5, 50)
    
    for i, t in enumerate([0, 1, 2, 3]):
        phase_points = np.array([[x_fixed, v, t] for v in v_points])
        predictions = solver.predict(phase_points)
        f_values = predictions[:, 0]
        
        plt.plot(v_points, f_values, linewidth=2, 
                label=f't={t}', alpha=0.8)
    
    plt.xlabel('速度 v')
    plt.ylabel('分布函数 f')
    plt.title(f'速度分布演化 (x={x_fixed})')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 3. 密度分布演化
    plt.subplot(2, 3, 2)
    x_points = np.linspace(-1, 1, 30)
    
    for t in [0, 1, 2, 3]:
        macro_quantities = solver.compute_macroscopic_quantities(x_points, t)
        plt.plot(x_points, macro_quantities['density'], 
                linewidth=2, label=f't={t}', alpha=0.8)
    
    plt.xlabel('位置 x')
    plt.ylabel('密度 n')
    plt.title('密度分布演化')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 4. 电势演化
    plt.subplot(2, 3, 3)
    for t in [0, 1, 2, 3]:
        v_center = 0.0
        phi_values = []
        for x in x_points:
            phi = solver.predict(np.array([[x, v_center, t]]))[0, 1]
            phi_values.append(phi)
        
        plt.plot(x_points, phi_values, linewidth=2, 
                label=f't={t}', alpha=0.8)
    
    plt.xlabel('位置 x')
    plt.ylabel('电势 φ')
    plt.title('电势演化')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 5. 相空间涡旋检测
    plt.subplot(2, 3, 4)
    t_fixed = 2.0  # 固定时间
    x_fine = np.linspace(-1, 1, 25)
    v_fine = np.linspace(-2.5, 2.5, 25)
    X_fine, V_fine = np.meshgrid(x_fine, v_fine)
    
    phase_fine = np.stack([X_fine.flatten(), V_fine.flatten(), 
                          np.full_like(X_fine.flatten(), t_fixed)], axis=1)
    f_fine = solver.predict(phase_fine)[:, 0].reshape(X_fine.shape)
    
    contour_plot = plt.contour(X_fine, V_fine, f_fine, levels=15, alpha=0.8)
    plt.clabel(contour_plot, inline=True, fontsize=8)
    plt.xlabel('位置 x')
    plt.ylabel('速度 v')
    plt.title(f'相空间等值线 (t={t_fixed})')
    
    # 6. 能量分析
    plt.subplot(2, 3, 5)
    kinetic_energies = []
    potential_energies = []
    
    for t in times:
        # 计算动能密度 (简化)
        x_sample = np.linspace(-1, 1, 20)
        v_sample = np.linspace(-2.5, 2.5, 20)
        
        kinetic_energy = 0.0
        potential_energy = 0.0
        
        for x in x_sample:
            for v in v_sample:
                pred = solver.predict(np.array([[x, v, t]]))
                f_val = pred[0, 0]
                phi_val = pred[0, 1]
                
                kinetic_energy += 0.5 * v**2 * f_val
                potential_energy += 0.5 * phi_val**2
        
        kinetic_energies.append(kinetic_energy)
        potential_energies.append(potential_energy)
    
    plt.plot(times, kinetic_energies, 'b-', linewidth=2, label='动能')
    plt.plot(times, potential_energies, 'r-', linewidth=2, label='势能')
    plt.plot(times, np.array(kinetic_energies) + np.array(potential_energies), 
            'g--', linewidth=2, label='总能量')
    plt.xlabel('时间 t')
    plt.ylabel('能量')
    plt.title('能量守恒分析')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 7. 熵演化 (简化)
    plt.subplot(2, 3, 6)
    entropies = []
    
    for t in times:
        entropy = 0.0
        for x in x_sample:
            for v in v_sample:
                f_val = solver.predict(np.array([[x, v, t]]))[0, 0]
                if f_val > 1e-10:  # 避免log(0)
                    entropy -= f_val * np.log(f_val + 1e-10)
        entropies.append(entropy)
    
    plt.plot(times, entropies, 'purple', linewidth=2)
    plt.xlabel('时间 t')
    plt.ylabel('熵 S')
    plt.title('熵演化 (H定理)')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    print("📈 高级分析完成!")
    print(f"能量变化: {(max(kinetic_energies) - min(kinetic_energies)):.6f}")
    print(f"熵变化: {(max(entropies) - min(entropies)):.6f}")

if __name__ == "__main__":
    # 设置matplotlib
    plt.rcParams['font.sans-serif'] = ['Arial', 'SimHei']
    plt.rcParams['axes.unicode_minus'] = False
    
    print("⚡ Vlasov-Poisson方程组求解器示例集")
    print("请选择要运行的示例：")
    print("1. Landau阻尼示例")
    print("2. 双流不稳定性示例")
    print("3. Bump-on-tail不稳定性示例")
    print("4. 参数研究示例")
    print("5. 高级分析示例")
    print("6. 运行所有示例")
    
    choice = input("\n请输入选择 (1-6): ").strip()
    
    if choice == "1":
        solver = example_landau_damping()
    elif choice == "2":
        solver = example_two_stream_instability()
    elif choice == "3":
        solver = example_bump_on_tail()
    elif choice == "4":
        example_parameter_study()
    elif choice == "5":
        example_advanced_analysis()
    elif choice == "6":
        print("\n🚀 运行所有示例...")
        print("⚠️  注意：这将需要很长时间！")
        confirm = input("确认运行所有示例？(y/n): ").strip().lower()
        if confirm == 'y':
            solver1 = example_landau_damping()
            solver2 = example_two_stream_instability()
            solver3 = example_bump_on_tail()
            example_parameter_study()
            example_advanced_analysis()
            print("\n✅ 所有示例运行完成！")
        else:
            print("取消运行所有示例")
    else:
        print("❌ 无效选择，运行Landau阻尼示例...")
        solver = example_landau_damping()
    
    print("\n🎉 Vlasov-Poisson示例运行完成！")