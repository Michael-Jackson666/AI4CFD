"""
运行脚本 - Vlasov-Poisson系统Landau阻尼模拟
展示经典的Landau阻尼现象及相关物理量的演化
"""

import numpy as np
import matplotlib.pyplot as plt
from vp_solver import VlasovPoissonSolver, compute_landau_damping_rate


def plot_landau_damping_results(solver, history, gamma_theory, save_fig=True):
    """
    绘制Landau阻尼结果
    
    参数:
        solver: VP求解器实例
        history: 时间历史数据
        gamma_theory: 理论阻尼率
        save_fig: 是否保存图像
    """
    fig = plt.figure(figsize=(18, 12))
    
    # 1. 电场模式随时间的演化（Landau阻尼）
    ax1 = plt.subplot(3, 3, 1)
    ax1.semilogy(history['time'], history['electric_field_mode'], 
                 'b-', linewidth=2, label='数值模拟')
    
    # 理论阻尼曲线
    E0 = history['electric_field_mode'][0]
    t_theory = history['time']
    E_theory = E0 * np.exp(gamma_theory * t_theory)
    ax1.semilogy(t_theory, E_theory, 'r--', linewidth=2, 
                 label=f'理论 (γ={gamma_theory:.6f})')
    
    ax1.set_xlabel('时间 t', fontsize=11)
    ax1.set_ylabel('电场第一模式幅值 |E₁|', fontsize=11)
    ax1.set_title('Landau阻尼现象', fontsize=13, fontweight='bold')
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.3)
    
    # 2. 能量守恒
    ax2 = plt.subplot(3, 3, 2)
    total_energy = history['energy']
    energy_normalized = (total_energy - total_energy[0]) / total_energy[0]
    ax2.plot(history['time'], energy_normalized * 100, 'g-', linewidth=2)
    ax2.set_xlabel('时间 t', fontsize=11)
    ax2.set_ylabel('总能量相对变化 (%)', fontsize=11)
    ax2.set_title('能量守恒检查', fontsize=13, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.axhline(y=0, color='k', linestyle='--', alpha=0.5)
    
    # 3. 电场能量和动能
    ax3 = plt.subplot(3, 3, 3)
    ax3.semilogy(history['time'], history['electric_energy'], 
                'b-', linewidth=2, label='电场能量')
    ax3.semilogy(history['time'], history['kinetic_energy'], 
                'r-', linewidth=2, label='动能')
    ax3.set_xlabel('时间 t', fontsize=11)
    ax3.set_ylabel('能量', fontsize=11)
    ax3.set_title('能量分量演化', fontsize=13, fontweight='bold')
    ax3.legend(fontsize=9)
    ax3.grid(True, alpha=0.3)
    
    # 4. 初始分布函数
    ax4 = plt.subplot(3, 3, 4)
    state_initial = solver.get_state()
    im1 = ax4.contourf(solver.x, solver.v, state_initial['f'].T, 
                      levels=20, cmap='viridis')
    ax4.set_xlabel('位置 x', fontsize=11)
    ax4.set_ylabel('速度 v', fontsize=11)
    ax4.set_title('初始分布函数 f(x,v,t=0)', fontsize=13, fontweight='bold')
    plt.colorbar(im1, ax=ax4, label='f')
    
    # 5. 最终分布函数
    ax5 = plt.subplot(3, 3, 5)
    im2 = ax5.contourf(solver.x, solver.v, solver.f.T, 
                      levels=20, cmap='viridis')
    ax5.set_xlabel('位置 x', fontsize=11)
    ax5.set_ylabel('速度 v', fontsize=11)
    ax5.set_title(f'最终分布函数 f(x,v,t={history["time"][-1]:.1f})', 
                 fontsize=13, fontweight='bold')
    plt.colorbar(im2, ax=ax5, label='f')
    
    # 6. 电场分布
    ax6 = plt.subplot(3, 3, 6)
    ax6.plot(solver.x, solver.E, 'b-', linewidth=2)
    ax6.set_xlabel('位置 x', fontsize=11)
    ax6.set_ylabel('电场 E(x)', fontsize=11)
    ax6.set_title(f'电场分布 (t={history["time"][-1]:.1f})', 
                 fontsize=13, fontweight='bold')
    ax6.grid(True, alpha=0.3)
    ax6.axhline(y=0, color='k', linestyle='--', alpha=0.5)
    
    # 7. 速度分布演化
    ax7 = plt.subplot(3, 3, 7)
    v_dist_initial = np.trapz(state_initial['f'], solver.x, axis=0)
    v_dist_final = np.trapz(solver.f, solver.x, axis=0)
    ax7.plot(solver.v, v_dist_initial, 'b-', linewidth=2, label='初始')
    ax7.plot(solver.v, v_dist_final, 'r--', linewidth=2, label='最终')
    ax7.set_xlabel('速度 v', fontsize=11)
    ax7.set_ylabel('∫f dx', fontsize=11)
    ax7.set_title('速度分布演化', fontsize=13, fontweight='bold')
    ax7.legend(fontsize=9)
    ax7.grid(True, alpha=0.3)
    
    # 8. 空间密度演化
    ax8 = plt.subplot(3, 3, 8)
    rho_initial = np.trapz(state_initial['f'], solver.v, axis=1)
    rho_final = np.trapz(solver.f, solver.v, axis=1)
    ax8.plot(solver.x, rho_initial, 'b-', linewidth=2, label='初始')
    ax8.plot(solver.x, rho_final, 'r--', linewidth=2, label='最终')
    ax8.set_xlabel('位置 x', fontsize=11)
    ax8.set_ylabel('密度 ρ(x)', fontsize=11)
    ax8.set_title('空间密度演化', fontsize=13, fontweight='bold')
    ax8.legend(fontsize=9)
    ax8.grid(True, alpha=0.3)
    ax8.axhline(y=1.0, color='k', linestyle='--', alpha=0.5)
    
    # 9. L2范数
    ax9 = plt.subplot(3, 3, 9)
    ax9.plot(history['time'], history['L2_norm'], 'purple', linewidth=2)
    ax9.set_xlabel('时间 t', fontsize=11)
    ax9.set_ylabel('||f||₂', fontsize=11)
    ax9.set_title('分布函数L2范数', fontsize=13, fontweight='bold')
    ax9.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_fig:
        plt.savefig('landau_damping_results.png', dpi=200, bbox_inches='tight')
        print("\n结果图已保存: landau_damping_results.png")
    
    plt.show()


def plot_damping_comparison(history, gamma_theory):
    """
    单独绘制阻尼对比图（更清晰）
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # 对数坐标
    ax1 = axes[0]
    ax1.semilogy(history['time'], history['electric_field_mode'], 
                 'b-', linewidth=2.5, label='数值模拟', marker='o', 
                 markersize=4, markevery=10)
    
    E0 = history['electric_field_mode'][0]
    t_theory = history['time']
    E_theory = E0 * np.exp(gamma_theory * t_theory)
    ax1.semilogy(t_theory, E_theory, 'r--', linewidth=2.5, 
                 label=f'理论预测 (γ={gamma_theory:.6f})')
    
    ax1.set_xlabel('时间 t', fontsize=13)
    ax1.set_ylabel('电场第一模式幅值 |E₁|', fontsize=13)
    ax1.set_title('Landau阻尼 - 对数坐标', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3, which='both')
    
    # 线性坐标
    ax2 = axes[1]
    ax2.plot(history['time'], history['electric_field_mode'], 
            'b-', linewidth=2.5, label='数值模拟', marker='o', 
            markersize=4, markevery=10)
    ax2.plot(t_theory, E_theory, 'r--', linewidth=2.5, 
            label=f'理论预测 (γ={gamma_theory:.6f})')
    
    ax2.set_xlabel('时间 t', fontsize=13)
    ax2.set_ylabel('电场第一模式幅值 |E₁|', fontsize=13)
    ax2.set_title('Landau阻尼 - 线性坐标', fontsize=14, fontweight='bold')
    ax2.legend(fontsize=11)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('landau_damping_comparison.png', dpi=200, bbox_inches='tight')
    print("对比图已保存: landau_damping_comparison.png")
    plt.show()


def compute_damping_rate_from_simulation(history):
    """从模拟结果计算阻尼率"""
    time = history['time']
    E_mode = history['electric_field_mode']
    
    # 使用对数拟合
    log_E = np.log(E_mode)
    
    # 线性拟合 log(E) = log(E0) + γ*t
    # 只使用衰减阶段的数据（前80%）
    n_fit = int(len(time) * 0.8)
    coeffs = np.polyfit(time[:n_fit], log_E[:n_fit], 1)
    gamma_fitted = coeffs[0]
    
    return gamma_fitted


def print_analysis_results(history, gamma_theory, gamma_fitted):
    """打印分析结果"""
    print("\n" + "=" * 60)
    print("Landau阻尼分析结果")
    print("=" * 60)
    
    print(f"\n理论阻尼率: γ_theory = {gamma_theory:.8f}")
    print(f"数值阻尼率: γ_numeric = {gamma_fitted:.8f}")
    print(f"相对误差: {abs(gamma_fitted - gamma_theory) / abs(gamma_theory) * 100:.2f}%")
    
    print(f"\n能量守恒:")
    energy_change = (history['energy'][-1] - history['energy'][0]) / history['energy'][0]
    print(f"  总能量相对变化: {energy_change * 100:.4f}%")
    print(f"  能量标准差/平均值: {np.std(history['energy']) / np.mean(history['energy']) * 100:.4f}%")
    
    print(f"\nL2范数:")
    L2_change = (history['L2_norm'][-1] - history['L2_norm'][0]) / history['L2_norm'][0]
    print(f"  L2范数相对变化: {L2_change * 100:.4f}%")
    
    print(f"\n电场能量:")
    print(f"  初始: {history['electric_energy'][0]:.6e}")
    print(f"  最终: {history['electric_energy'][-1]:.6e}")
    print(f"  衰减因子: {history['electric_energy'][-1] / history['electric_energy'][0]:.6e}")
    
    print("\n" + "=" * 60)


def main():
    """主函数 - 运行完整的Landau阻尼模拟"""
    
    print("=" * 70)
    print(" " * 15 + "Vlasov-Poisson系统 - Landau阻尼模拟")
    print("=" * 70)
    
    # 参数设置
    nx = 64          # x方向网格数
    nv = 64          # v方向网格数
    Lx = 4 * np.pi   # x方向区域长度
    Lv = 6.0         # v方向区域长度
    dt = 0.1         # 时间步长
    T_final = 50.0   # 最终时间
    
    k_wave = 0.5     # 扰动波数
    alpha = 0.01     # 扰动幅度
    v_thermal = 1.0  # 热速度
    
    print(f"\n模拟参数:")
    print(f"  空间网格: {nx} × {nv}")
    print(f"  空间区域: x ∈ [0, {Lx:.2f}], v ∈ [{-Lv/2:.2f}, {Lv/2:.2f}]")
    print(f"  时间步长: dt = {dt}")
    print(f"  模拟时间: T = {T_final}")
    print(f"\nLandau阻尼参数:")
    print(f"  波数: k = {k_wave}")
    print(f"  扰动幅度: α = {alpha}")
    print(f"  热速度: vth = {v_thermal}")
    
    # 创建求解器
    print("\n初始化求解器...")
    solver = VlasovPoissonSolver(nx=nx, nv=nv, Lx=Lx, Lv=Lv, dt=dt)
    
    # 初始化Landau阻尼问题
    solver.initialize_landau_damping(k=k_wave, alpha=alpha, v_thermal=v_thermal)
    
    # 保存初始状态
    initial_state = solver.get_state()
    
    # 计算理论阻尼率
    gamma_theory = compute_landau_damping_rate(k_wave, v_thermal=v_thermal)
    print(f"\n理论Landau阻尼率: γ = {gamma_theory:.8f}")
    
    # 求解
    print("\n开始时间演化...")
    history = solver.solve(T_final=T_final, save_interval=5)
    
    # 从模拟结果计算阻尼率
    gamma_fitted = compute_damping_rate_from_simulation(history)
    
    # 打印分析结果
    print_analysis_results(history, gamma_theory, gamma_fitted)
    
    # 绘图
    print("\n生成可视化结果...")
    
    # 恢复初始状态用于绘图
    solver.f = initial_state['f']
    solver.E = initial_state['E']
    solver.rho = initial_state['rho']
    
    # 绘制完整结果
    plot_landau_damping_results(solver, history, gamma_theory, save_fig=True)
    
    # 绘制详细的阻尼对比
    plot_damping_comparison(history, gamma_theory)
    
    print("\n模拟完成!")
    print("生成的图像文件:")
    print("  - landau_damping_results.png (完整结果)")
    print("  - landau_damping_comparison.png (阻尼对比)")


if __name__ == "__main__":
    main()
