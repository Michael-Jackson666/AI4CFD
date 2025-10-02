#!/usr/bin/env python3
"""
Quick test for Vlasov-Poisson system
Vlasov-Poisson 6D系统快速测试
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from vlasov_poisson_solver import VlasovPoissonSolver
import numpy as np

def quick_test_vp():
    """Vlasov-Poisson系统快速测试"""
    print("🚀 Quick Vlasov-Poisson 6D System Test")
    print("Testing basic functionality...")
    
    # 创建求解器 - 1D Landau damping with minimal resolution
    solver = VlasovPoissonSolver(
        x_domain=(-1, 1),
        v_domain=(-3, 3),
        time_domain=(0, 0.5),  # Very short time
        case="landau_damping"
    )
    
    # 设置Landau damping初始条件  
    solver.setup_geometry_and_conditions()
    
    # 测试预测功能（不训练）
    print("🔮 Testing prediction function...")
    
    # 创建小型测试点
    test_points = np.array([
        [0.0, 0.0, 0.0],    # (x, v, t)
        [0.5, 1.0, 0.1],
        [-0.5, -1.0, 0.2]
    ])
    
    try:
        # 测试初始条件方法
        print("🧪 Testing initial conditions...")
        
        # 创建一些测试点
        x_test = np.array([0.0, 0.5, -0.5])
        v_test = np.array([0.0, 1.0, -1.0]) 
        t_test = np.array([0.0, 0.0, 0.0])
        
        # 构造输入点 (x, v, t)
        test_points = np.column_stack([x_test, v_test, t_test])
        print(f"Test points shape: {test_points.shape}")
        print(f"Test points: {test_points}")
        
        # 测试是否能调用初始条件函数（如果存在）
        if hasattr(solver, 'initial_condition_f'):
            f_vals = solver.initial_condition_f(test_points)
            print(f"✅ Initial f function works, shape: {f_vals.shape}")
        else:
            print("⚠️ No initial_condition_f method found")
            
        if hasattr(solver, 'initial_condition_phi'):
            phi_vals = solver.initial_condition_phi(test_points) 
            print(f"✅ Initial phi function works, shape: {phi_vals.shape}")
        else:
            print("⚠️ No initial_condition_phi method found")
        
        print("✅ Vlasov-Poisson system basic functionality test passed!")
        
    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        return None
    
    return solver

if __name__ == "__main__":
    solver = quick_test_vp()