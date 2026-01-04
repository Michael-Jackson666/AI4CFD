"""
使用 KAN 求解 1D Poisson 方程

问题描述:
    -d²u/dx² = f(x),  x ∈ [-1, 1]
    边界条件: u(-1) = 0, u(1) = 0
    
本例中取 f(x) = sin(πx)
解析解: u(x) = sin(πx) / π²

特点: KAN 在求解此类光滑问题时表现优异
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import sys
import os

# 添加父目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models import KANPDE

# 设置随机种子
torch.manual_seed(42)
np.random.seed(42)

# 设备配置
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"🖥️  使用设备: {device}\n")


# ==============================================================================
# 1. 解析解
# ==============================================================================

def analytical_solution(x):
    """
    Poisson 方程解析解: u(x) = sin(πx) / π²
    """
    return np.sin(np.pi * x) / (np.pi ** 2)


def source_term(x):
    """
    源项: f(x) = sin(πx)
    """
    return torch.sin(np.pi * x)


# ==============================================================================
# 2. 数据准备
# ==============================================================================

def prepare_training_data(n_interior=100, n_boundary=2):
    """
    准备训练数据
    
    参数:
        n_interior: 内部配点数量
        n_boundary: 边界点数量
    
    返回:
        x_interior: 内部点
        x_boundary: 边界点
        u_boundary: 边界值
    """
    # 内部配点 (用于计算 PDE 残差)
    x_interior = torch.linspace(-1, 1, n_interior).unsqueeze(1).to(device)
    x_interior.requires_grad_(True)
    
    # 边界点
    x_boundary = torch.tensor([[-1.0], [1.0]], device=device)
    u_boundary = torch.tensor([[0.0], [0.0]], device=device)
    
    return x_interior, x_boundary, u_boundary


# ==============================================================================
# 3. 损失函数
# ==============================================================================

def compute_pde_loss(model, x_interior, x_boundary, u_boundary, lambda_reg=1e-5):
    """
    计算总损失
    
    L_total = L_pde + L_bc + λ * L_reg
    
    其中:
    - L_pde: PDE 残差损失 (内部点)
    - L_bc: 边界条件损失
    - L_reg: 正则化损失 (B-spline 系数)
    """
    # ========== PDE 残差损失 ==========
    # 计算 u 和 u''
    u = model(x_interior)
    
    # 一阶导数 du/dx
    u_x = torch.autograd.grad(
        u, x_interior,
        grad_outputs=torch.ones_like(u),
        create_graph=True,
        retain_graph=True
    )[0]
    
    # 二阶导数 d²u/dx²
    u_xx = torch.autograd.grad(
        u_x, x_interior,
        grad_outputs=torch.ones_like(u_x),
        create_graph=True,
        retain_graph=True
    )[0]
    
    # PDE: -u_xx = f(x)
    f = source_term(x_interior)
    pde_residual = -u_xx - f
    loss_pde = torch.mean(pde_residual ** 2)
    
    # ========== 边界条件损失 ==========
    u_b = model(x_boundary)
    loss_bc = torch.mean((u_b - u_boundary) ** 2)
    
    # ========== 正则化损失 ==========
    loss_reg = model.regularization_loss()
    
    # ========== 总损失 ==========
    total_loss = loss_pde + loss_bc + lambda_reg * loss_reg
    
    return total_loss, {
        'pde': loss_pde.item(),
        'bc': loss_bc.item(),
        'reg': loss_reg.item()
    }


# ==============================================================================
# 4. 训练函数
# ==============================================================================

def train_kan(model, x_interior, x_boundary, u_boundary, 
              epochs=5000, lr=1e-3, lambda_reg=1e-5):
    """
    训练 KAN 模型
    """
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.8, patience=500, verbose=False
    )
    
    history = {'total': [], 'pde': [], 'bc': [], 'reg': []}
    
    print("🎯 开始训练...")
    print(f"   Epochs: {epochs}, 学习率: {lr}, 正则化: {lambda_reg}\n")
    
    for epoch in range(epochs):
        optimizer.zero_grad()
        
        # 计算损失
        loss, loss_dict = compute_pde_loss(
            model, x_interior, x_boundary, u_boundary, lambda_reg
        )
        
        # 反向传播
        loss.backward()
        optimizer.step()
        scheduler.step(loss)
        
        # 记录历史
        history['total'].append(loss.item())
        history['pde'].append(loss_dict['pde'])
        history['bc'].append(loss_dict['bc'])
        history['reg'].append(loss_dict['reg'])
        
        # 打印进度
        if (epoch + 1) % 1000 == 0 or epoch == 0:
            print(f"Epoch {epoch+1:5d}/{epochs} | "
                  f"Loss: {loss.item():.2e} | "
                  f"PDE: {loss_dict['pde']:.2e} | "
                  f"BC: {loss_dict['bc']:.2e} | "
                  f"LR: {optimizer.param_groups[0]['lr']:.2e}")
    
    print("\n✅ 训练完成!")
    return history


# ==============================================================================
# 5. 评估和可视化
# ==============================================================================

def evaluate_and_plot(model, history):
    """
    评估模型并可视化结果
    """
    model.eval()
    
    # 测试点
    x_test = torch.linspace(-1, 1, 200).unsqueeze(1).to(device)
    
    with torch.no_grad():
        u_pred = model(x_test).cpu().numpy()
    
    x_test_np = x_test.cpu().numpy()
    u_exact = analytical_solution(x_test_np)
    
    # 计算误差
    error = np.abs(u_pred - u_exact)
    l2_error = np.linalg.norm(error) / np.linalg.norm(u_exact)
    max_error = np.max(error)
    
    print(f"\n📊 误差分析:")
    print(f"   相对 L2 误差: {l2_error:.4e}")
    print(f"   最大绝对误差: {max_error:.4e}")
    
    # 可视化
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # 1. 解的对比
    axes[0, 0].plot(x_test_np, u_exact, 'b-', linewidth=2, label='解析解')
    axes[0, 0].plot(x_test_np, u_pred, 'r--', linewidth=2, label='KAN 预测')
    axes[0, 0].set_xlabel('x')
    axes[0, 0].set_ylabel('u(x)')
    axes[0, 0].set_title('Poisson 方程解: KAN vs 解析解')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # 2. 绝对误差
    axes[0, 1].plot(x_test_np, error, 'g-', linewidth=2)
    axes[0, 1].set_xlabel('x')
    axes[0, 1].set_ylabel('|u_KAN - u_exact|')
    axes[0, 1].set_title(f'绝对误差 (Max: {max_error:.2e})')
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].set_yscale('log')
    
    # 3. 训练损失曲线
    epochs = range(1, len(history['total']) + 1)
    axes[1, 0].semilogy(epochs, history['total'], 'b-', label='总损失', alpha=0.8)
    axes[1, 0].semilogy(epochs, history['pde'], 'r-', label='PDE 损失', alpha=0.6)
    axes[1, 0].semilogy(epochs, history['bc'], 'g-', label='BC 损失', alpha=0.6)
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Loss (log scale)')
    axes[1, 0].set_title('训练损失曲线')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # 4. 误差分布
    axes[1, 1].hist(error.flatten(), bins=50, color='purple', alpha=0.7, edgecolor='black')
    axes[1, 1].set_xlabel('绝对误差')
    axes[1, 1].set_ylabel('频次')
    axes[1, 1].set_title('误差分布直方图')
    axes[1, 1].grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig('kan_poisson_results.png', dpi=150, bbox_inches='tight')
    print(f"\n💾 结果已保存到: kan_poisson_results.png")
    plt.show()
    
    return l2_error


# ==============================================================================
# 6. 主程序
# ==============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("KAN 求解 1D Poisson 方程")
    print("PDE: -u'' = sin(πx),  x ∈ [-1, 1]")
    print("BC:  u(-1) = u(1) = 0")
    print("=" * 70)
    
    # ========== 创建模型 ==========
    print("\n[1/4] 创建 KAN 模型...")
    model = KANPDE(
        layers=[1, 16, 16, 1],  # 1输入 -> 16 -> 16 -> 1输出
        grid_size=5,            # B-spline 网格大小
        spline_order=3,         # 三次样条
        grid_range=(-1, 1)      # 输入范围
    ).to(device)
    
    print(f"   模型结构: {[1, 16, 16, 1]}")
    print(f"   参数量: {model.count_parameters():,}")
    
    # ========== 准备数据 ==========
    print("\n[2/4] 准备训练数据...")
    x_interior, x_boundary, u_boundary = prepare_training_data(
        n_interior=100,
        n_boundary=2
    )
    print(f"   内部配点: {x_interior.shape[0]}")
    print(f"   边界点: {x_boundary.shape[0]}")
    
    # ========== 训练模型 ==========
    print("\n[3/4] 训练模型...")
    history = train_kan(
        model, x_interior, x_boundary, u_boundary,
        epochs=5000,
        lr=1e-3,
        lambda_reg=1e-5
    )
    
    # ========== 评估和可视化 ==========
    print("\n[4/4] 评估和可视化...")
    l2_error = evaluate_and_plot(model, history)
    
    print("\n" + "=" * 70)
    print(f"✅ 实验完成! 相对 L2 误差: {l2_error:.4e}")
    print("=" * 70)
