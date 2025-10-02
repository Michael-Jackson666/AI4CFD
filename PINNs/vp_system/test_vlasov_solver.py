#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Vlasov-Poisson求解器快速测试

这是一个简化的测试脚本，用于验证Vlasov-Poisson求解器的基本功能
使用较少的计算资源，适合初次测试和调试
"""

import numpy as np
import matplotlib.pyplot as plt
import sys
import os

# 添加当前目录到Python路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from vlasov_poisson_solver import VlasovPoissonSolver
    print("✅ VlasovPoissonSolver导入成功")
except ImportError as e:
    print(f"❌ 导入失败: {e}")
    print("请确保已安装 deepxde, numpy, matplotlib, scipy")
    sys.exit(1)

def quick_test():
    """快速功能测试"""
    print("🚀 开始Vlasov-Poisson求解器快速测试...")
    print("⚠️  这是一个简化测试，使用最小参数以快速验证功能")
    
    try:
        # 创建一个小规模的测试案例
        solver = VlasovPoissonSolver(
            x_domain=(-0.5, 0.5),     # 较小的空间域
            v_domain=(-2.0, 2.0),     # 较小的速度域  
            time_domain=(0.0, 0.5),   # 较短的时间
            case="landau_damping"
        )
        print("✅ 求解器创建成功")
        
        # 设置几何和条件
        solver.setup_geometry_and_conditions()
        print("✅ 几何和边界条件设置成功")
        
        # 可视化初始条件
        print("📊 可视化初始条件...")
        solver.visualize_initial_conditions(resolution=30)
        
        # 创建小型模型 (最小配置)
        solver.create_model(
            num_domain=1000,          # 最小采样点数
            num_boundary=80,
            num_initial=80,
            layer_sizes=[3, 40, 40, 2],  # 小网络
            activation="tanh"
        )
        print("✅ 神经网络模型创建成功")
        
        # 快速训练 (极少迭代，仅验证功能)
        print("🎯 开始快速训练 (仅验证功能)...")
        solver.train(
            adam_iterations=200,      # 极少迭代
            adam_lr=0.01,            # 较大学习率
            use_lbfgs=False          # 跳过L-BFGS
        )
        print("✅ 训练完成")
        
        # 测试预测功能
        print("🔍 测试预测功能...")
        test_points = np.array([
            [0.0, 0.0, 0.0],         # 原点
            [0.1, 0.5, 0.1],         # 随机点1
            [-0.1, -0.5, 0.2]        # 随机点2
        ])
        
        predictions = solver.predict(test_points)
        print(f"预测结果形状: {predictions.shape}")
        print(f"分布函数f范围: [{predictions[:, 0].min():.6f}, {predictions[:, 0].max():.6f}]")
        print(f"电势φ范围: [{predictions[:, 1].min():.6f}, {predictions[:, 1].max():.6f}]")
        
        # 简单可视化
        print("📈 生成简单可视化...")
        solver.visualize_phase_space_evolution(
            times=[0.0, 0.25, 0.5], 
            resolution=20
        )
        
        print("✅ 快速测试完成！")
        print("\n📋 测试总结:")
        print("- 求解器创建: ✅")
        print("- 几何设置: ✅") 
        print("- 模型构建: ✅")
        print("- 训练功能: ✅")
        print("- 预测功能: ✅")
        print("- 可视化功能: ✅")
        print("\n🎉 所有基本功能正常！可以进行完整的Vlasov-Poisson求解。")
        
        return solver
        
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        import traceback
        print("详细错误信息:")
        traceback.print_exc()
        return None

def minimal_landau_damping():
    """最小化的Landau阻尼示例"""
    print("\n" + "="*50)
    print("最小化Landau阻尼示例")
    print("="*50)
    
    solver = VlasovPoissonSolver(
        x_domain=(-1.0, 1.0),
        v_domain=(-2.0, 2.0),
        time_domain=(0.0, 1.0),
        case="landau_damping"
    )
    
    solver.setup_geometry_and_conditions()
    solver.visualize_initial_conditions(resolution=25)
    
    # 小规模训练
    solver.create_model(
        num_domain=1500,
        num_boundary=120,
        num_initial=120,
        layer_sizes=[3, 60, 60, 60, 2]
    )
    
    print("🚀 开始训练 (最小规模)...")
    solver.train(
        adam_iterations=1000,
        adam_lr=0.005,
        use_lbfgs=False
    )
    
    # 分析结果
    solver.visualize_phase_space_evolution(times=[0.0, 0.5, 1.0])
    
    # 简单的电场演化分析
    times = np.linspace(0, 1, 20)
    electric_fields = []
    
    for t in times:
        # 计算中心点电场
        dx = 0.01
        phi_left = solver.predict(np.array([[-dx, 0.0, t]]))[0, 1]
        phi_right = solver.predict(np.array([[dx, 0.0, t]]))[0, 1]
        E_field = -(phi_right - phi_left) / (2 * dx)
        electric_fields.append(E_field)
    
    plt.figure(figsize=(10, 4))
    
    plt.subplot(1, 2, 1)
    plt.semilogy(times, np.abs(electric_fields), 'b-', linewidth=2)
    plt.xlabel('时间 t')
    plt.ylabel('|电场|')
    plt.title('电场演化 (Landau阻尼)')
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 2, 2)
    plt.plot(solver.losshistory.steps, solver.losshistory.loss_train, 'b-', label='训练损失')
    plt.plot(solver.losshistory.steps, solver.losshistory.loss_test, 'r--', label='测试损失')
    plt.xlabel('训练步数')
    plt.ylabel('损失')
    plt.yscale('log')
    plt.title('训练历史')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    print(f"📊 电场初值: {abs(electric_fields[0]):.6f}")
    print(f"📊 电场终值: {abs(electric_fields[-1]):.6f}")
    
    if abs(electric_fields[-1]) < abs(electric_fields[0]):
        print("✅ 观察到电场衰减，符合Landau阻尼预期")
    else:
        print("⚠️  电场未显著衰减，可能需要更长训练或更好参数")
    
    return solver

def benchmark_test():
    """基准测试 - 评估计算性能"""
    print("\n" + "="*50)
    print("基准性能测试")
    print("="*50)
    
    import time
    
    # 测试不同规模的性能
    test_configs = [
        {"domain": 500, "boundary": 40, "initial": 40, "layers": [3, 30, 30, 2], "iter": 100},
        {"domain": 1000, "boundary": 80, "initial": 80, "layers": [3, 50, 50, 2], "iter": 200},
        {"domain": 2000, "boundary": 120, "initial": 120, "layers": [3, 70, 70, 2], "iter": 300},
    ]
    
    results = []
    
    for i, config in enumerate(test_configs):
        print(f"\n--- 测试配置 {i+1}: {config['domain']} 域内点 ---")
        
        try:
            start_time = time.time()
            
            solver = VlasovPoissonSolver(
                x_domain=(-0.5, 0.5),
                v_domain=(-1.5, 1.5),
                time_domain=(0.0, 0.3),
                case="landau_damping"
            )
            
            solver.setup_geometry_and_conditions()
            
            solver.create_model(
                num_domain=config["domain"],
                num_boundary=config["boundary"],
                num_initial=config["initial"],
                layer_sizes=config["layers"]
            )
            
            setup_time = time.time() - start_time
            
            # 训练
            train_start = time.time()
            solver.train(
                adam_iterations=config["iter"],
                adam_lr=0.01,
                use_lbfgs=False
            )
            train_time = time.time() - train_start
            
            total_time = time.time() - start_time
            
            results.append({
                "config": config,
                "setup_time": setup_time,
                "train_time": train_time,
                "total_time": total_time,
                "final_loss": solver.train_state.loss_train
            })
            
            print(f"✅ 设置时间: {setup_time:.2f}s")
            print(f"✅ 训练时间: {train_time:.2f}s")
            print(f"✅ 总时间: {total_time:.2f}s")
            print(f"✅ 最终损失: {solver.train_state.loss_train:.6f}")
            
        except Exception as e:
            print(f"❌ 配置 {i+1} 失败: {e}")
            results.append({"config": config, "error": str(e)})
    
    # 总结基准测试结果
    print("\n📊 基准测试总结:")
    print("-" * 60)
    print("域内点数\t训练时间\t总时间\t\t最终损失")
    print("-" * 60)
    
    for result in results:
        if "error" not in result:
            config = result["config"]
            print(f"{config['domain']}\t\t{result['train_time']:.1f}s\t\t{result['total_time']:.1f}s\t\t{result['final_loss']:.6f}")
        else:
            print(f"{result['config']['domain']}\t\t失败: {result['error'][:20]}...")
    
    print("-" * 60)

if __name__ == "__main__":
    # 设置matplotlib
    plt.rcParams['font.sans-serif'] = ['Arial', 'SimHei']
    plt.rcParams['axes.unicode_minus'] = False
    
    print("⚡ Vlasov-Poisson求解器测试套件")
    print("选择测试类型:")
    print("1. 快速功能测试 (推荐首次运行)")
    print("2. 最小化Landau阻尼示例")
    print("3. 基准性能测试")
    print("4. 运行所有测试")
    
    choice = input("\n请输入选择 (1-4): ").strip()
    
    if choice == "1":
        solver = quick_test()
    elif choice == "2":
        solver = minimal_landau_damping()
    elif choice == "3":
        benchmark_test()
    elif choice == "4":
        print("\n🚀 运行所有测试...")
        solver1 = quick_test()
        if solver1:
            solver2 = minimal_landau_damping()
            benchmark_test()
        print("\n✅ 所有测试完成！")
    else:
        print("❌ 无效选择，运行快速功能测试...")
        solver = quick_test()
    
    print("\n🎉 测试完成！如果所有测试通过，您可以安全地使用完整的Vlasov-Poisson求解器。")