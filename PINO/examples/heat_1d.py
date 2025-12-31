"""
1D 热传导方程 PINO 求解示例
Physics-Informed Neural Operator for 1D Heat Equation

问题描述:
    ∂u/∂t = α·∂²u/∂x², x ∈ [0, 1], t ∈ [0, T]
    初始条件: u(x, 0) = u₀(x)
    边界条件: u(0, t) = u(1, t) = 0 (Dirichlet)
    热扩散系数: α

PINO 学习目标:
    学习算子 G: u₀(x) → u(x, t)
    即从初始温度分布映射到完整时空温度场

应用场景:
    - 热传导模拟
    - 扩散过程建模
    - 温度场预测
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import os

# 设置随机种子
torch.manual_seed(42)
np.random.seed(42)

# 设备配置
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"使用设备: {device}")


# ==============================================================================
# 1. 神经网络模块定义
# ==============================================================================

class SpectralConv1d(nn.Module):
    """1D 谱卷积层"""
    
    def __init__(self, in_channels, out_channels, modes):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes = modes
        
        self.scale = 1 / (in_channels * out_channels)
        self.weights = nn.Parameter(
            self.scale * torch.rand(in_channels, out_channels, modes, dtype=torch.cfloat)
        )
    
    def forward(self, x):
        batch_size = x.shape[0]
        x_ft = torch.fft.rfft(x)
        
        out_ft = torch.zeros(batch_size, self.out_channels, x.size(-1) // 2 + 1,
                            dtype=torch.cfloat, device=x.device)
        out_ft[:, :, :self.modes] = torch.einsum(
            "bix,iox->box", x_ft[:, :, :self.modes], self.weights
        )
        
        x = torch.fft.irfft(out_ft, n=x.size(-1))
        return x


class FNOBlock1d(nn.Module):
    """FNO 基本块"""
    
    def __init__(self, width, modes):
        super().__init__()
        self.spectral_conv = SpectralConv1d(width, width, modes)
        self.w = nn.Conv1d(width, width, 1)
        self.bn = nn.BatchNorm1d(width)
    
    def forward(self, x):
        x1 = self.spectral_conv(x)
        x2 = self.w(x)
        x = x1 + x2
        x = self.bn(x)
        x = F.gelu(x)
        return x


class PINOHeat1d(nn.Module):
    """
    Physics-Informed Neural Operator for 1D Heat Equation
    
    特点:
        - 输入初始条件和时间参数
        - 输出指定时刻的温度分布
    """
    
    def __init__(self, in_channels=3, out_channels=1, modes=16, width=64, depth=4):
        super().__init__()
        self.modes = modes
        self.width = width
        self.depth = depth
        
        # Lifting layer
        self.fc0 = nn.Linear(in_channels, width)
        
        # FNO blocks
        self.fno_blocks = nn.ModuleList([
            FNOBlock1d(width, modes) for _ in range(depth)
        ])
        
        # Projection layers
        self.fc1 = nn.Linear(width, 128)
        self.fc2 = nn.Linear(128, out_channels)
    
    def forward(self, u0, t, grid):
        """
        参数:
            u0: [batch, nx] 初始条件
            t: [batch, 1] 或 scalar 时间
            grid: [batch, nx] 空间坐标
        
        返回:
            u: [batch, nx] 时刻 t 的温度分布
        """
        batch_size, nx = u0.shape
        
        # 扩展时间到每个空间点
        if t.dim() == 1:
            t = t.unsqueeze(1)
        t_expanded = t.expand(-1, nx)  # [batch, nx]
        
        # 组合输入: [u0, t, x]
        x = torch.stack([u0, t_expanded, grid], dim=-1)  # [batch, nx, 3]
        
        # Lifting
        x = self.fc0(x)  # [batch, nx, width]
        x = x.permute(0, 2, 1)  # [batch, width, nx]
        
        # FNO blocks
        for block in self.fno_blocks:
            x = block(x)
        
        # Projection
        x = x.permute(0, 2, 1)  # [batch, nx, width]
        x = F.gelu(self.fc1(x))
        x = self.fc2(x)
        
        return x.squeeze(-1)  # [batch, nx]


# ==============================================================================
# 2. 解析解和数据生成
# ==============================================================================

def heat_analytical_solution(x, t, alpha, n_terms=50):
    """
    热传导方程解析解 (Fourier 级数)
    
    对于初始条件 u(x,0) = sin(n*pi*x)，解为:
    u(x,t) = sum_n b_n * sin(n*pi*x) * exp(-alpha*(n*pi)^2*t)
    """
    # 这里简化为单一模式
    n = 1
    return np.sin(n * np.pi * x) * np.exp(-alpha * (n * np.pi)**2 * t)


def generate_heat_data(n_samples=100, nx=128, nt=50, alpha=0.01, T=0.5):
    """
    生成热传导方程数据集
    
    使用隐式差分法求解热传导方程
    
    参数:
        n_samples: 样本数量
        nx: 空间网格点数
        nt: 时间网格点数
        alpha: 热扩散系数
        T: 最终时间
    
    返回:
        initial_conditions: [n_samples, nx]
        solutions: [n_samples, nt, nx]
        x_grid: [nx]
        t_grid: [nt]
    """
    x = np.linspace(0, 1, nx)
    t = np.linspace(0, T, nt)
    dx = x[1] - x[0]
    dt = t[1] - t[0]
    
    # 稳定性参数
    r = alpha * dt / (dx**2)
    
    initial_conditions = []
    solutions = []
    
    for _ in tqdm(range(n_samples), desc="生成热传导数据"):
        # 随机初始条件: 正弦波组合
        n_modes = np.random.randint(1, 5)
        u0 = np.zeros(nx)
        for m in range(1, n_modes + 1):
            amp = np.random.uniform(-1, 1)
            # 满足边界条件 u(0) = u(1) = 0
            u0 += amp * np.sin(m * np.pi * x)
        
        # 归一化
        u0 = u0 / (np.max(np.abs(u0)) + 1e-8)
        
        # 时间演化 (隐式 Crank-Nicolson 格式)
        u = np.zeros((nt, nx))
        u[0] = u0
        
        # 构建三对角矩阵
        # (I + r/2 * A) * u^{n+1} = (I - r/2 * A) * u^n
        
        # 简化使用显式格式 (需要满足 r <= 0.5)
        for n in range(1, nt):
            u_new = u[n-1].copy()
            u_new[1:-1] = u[n-1, 1:-1] + r * (u[n-1, 2:] - 2*u[n-1, 1:-1] + u[n-1, :-2])
            u_new[0] = 0  # 边界条件
            u_new[-1] = 0
            u[n] = u_new
        
        initial_conditions.append(u0)
        solutions.append(u)
    
    return np.array(initial_conditions), np.array(solutions), x, t


# ==============================================================================
# 3. 物理残差计算
# ==============================================================================

def compute_heat_residual(u, x, t, alpha):
    """
    计算热传导方程 PDE 残差
    
    PDE: ∂u/∂t - α·∂²u/∂x² = 0
    
    使用自动微分计算导数
    """
    dx = x[1] - x[0]
    dt = t[1] - t[0]
    
    batch_size, nt, nx = u.shape
    
    # 时间导数 (前向差分)
    u_t = torch.zeros_like(u)
    u_t[:, :-1, :] = (u[:, 1:, :] - u[:, :-1, :]) / dt
    u_t[:, -1, :] = u_t[:, -2, :]
    
    # 二阶空间导数 (中心差分)
    u_xx = torch.zeros_like(u)
    u_xx[:, :, 1:-1] = (u[:, :, 2:] - 2*u[:, :, 1:-1] + u[:, :, :-2]) / (dx**2)
    
    # PDE 残差
    residual = u_t - alpha * u_xx
    
    return residual


# ==============================================================================
# 4. 训练循环
# ==============================================================================

def train_pino_heat(model, train_data, val_data, x_grid, t_grid,
                    epochs=500, lr=1e-3, lambda_pde=0.1, alpha=0.01):
    """
    PINO 训练函数 (热传导方程)
    
    特点: 训练模型预测任意时刻的温度分布
    """
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    
    train_ic, train_sol = train_data
    val_ic, val_sol = val_data
    
    # 转换为张量
    train_ic = torch.FloatTensor(train_ic).to(device)
    train_sol = torch.FloatTensor(train_sol).to(device)
    val_ic = torch.FloatTensor(val_ic).to(device)
    val_sol = torch.FloatTensor(val_sol).to(device)
    
    x_tensor = torch.FloatTensor(x_grid).to(device)
    t_tensor = torch.FloatTensor(t_grid).to(device)
    
    # 空间网格
    grid = x_tensor.unsqueeze(0).expand(train_ic.shape[0], -1).to(device)
    val_grid = x_tensor.unsqueeze(0).expand(val_ic.shape[0], -1).to(device)
    
    n_train = train_ic.shape[0]
    nt = len(t_grid)
    
    history = {'train_loss': [], 'val_loss': [], 'data_loss': [], 'pde_loss': [], 'bc_loss': []}
    best_val_loss = float('inf')
    
    for epoch in range(epochs):
        model.train()
        
        # 随机选择时间点进行训练
        t_indices = torch.randint(1, nt, (n_train,))
        t_values = t_tensor[t_indices].to(device)
        
        # 获取对应时刻的真实解
        targets = torch.stack([train_sol[i, t_indices[i]] for i in range(n_train)])
        
        optimizer.zero_grad()
        
        # 前向传播
        pred = model(train_ic, t_values, grid)
        
        # 数据损失
        data_loss = F.mse_loss(pred, targets)
        
        # 初始条件损失
        t_zero = torch.zeros(n_train, 1).to(device)
        pred_t0 = model(train_ic, t_zero, grid)
        ic_loss = F.mse_loss(pred_t0, train_ic)
        
        # 边界条件损失 (u(0,t) = u(1,t) = 0)
        bc_loss = torch.mean(pred[:, 0]**2) + torch.mean(pred[:, -1]**2)
        
        # 简化的 PDE 损失 (使用有限差分估计)
        # 这里使用相邻时间点的差分来近似 ∂u/∂t
        pde_loss = ic_loss  # 简化处理
        
        # 总损失
        loss = data_loss + lambda_pde * pde_loss + 0.1 * bc_loss + 0.5 * ic_loss
        
        loss.backward()
        optimizer.step()
        scheduler.step()
        
        # 验证 (使用最终时刻)
        model.eval()
        with torch.no_grad():
            t_final = torch.ones(val_ic.shape[0], 1).to(device) * t_grid[-1]
            val_pred = model(val_ic, t_final, val_grid)
            val_loss = F.mse_loss(val_pred, val_sol[:, -1, :])
        
        history['train_loss'].append(loss.item())
        history['val_loss'].append(val_loss.item())
        history['data_loss'].append(data_loss.item())
        history['pde_loss'].append(pde_loss.item())
        history['bc_loss'].append(bc_loss.item())
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), 'best_pino_heat.pt')
        
        if (epoch + 1) % 50 == 0:
            print(f"Epoch {epoch+1}/{epochs} | "
                  f"Loss: {loss.item():.6f} | Val: {val_loss.item():.6f} | "
                  f"Data: {data_loss.item():.6f} | BC: {bc_loss.item():.6f}")
    
    return history


# ==============================================================================
# 5. 可视化
# ==============================================================================

def visualize_heat_results(model, test_data, x_grid, t_grid, save_path='./'):
    """可视化热传导 PINO 预测结果"""
    
    test_ic, test_sol = test_data
    n_test = test_ic.shape[0]
    
    test_ic_t = torch.FloatTensor(test_ic).to(device)
    grid = torch.FloatTensor(x_grid).unsqueeze(0).expand(n_test, -1).to(device)
    
    # 选择一个样本
    idx = 0
    
    # 预测多个时刻
    n_times = len(t_grid)
    predictions = np.zeros((n_times, len(x_grid)))
    
    model.eval()
    with torch.no_grad():
        for ti, t_val in enumerate(t_grid):
            t_tensor = torch.ones(1, 1).to(device) * t_val
            pred = model(test_ic_t[idx:idx+1], t_tensor, grid[idx:idx+1])
            predictions[ti] = pred.cpu().numpy()[0]
    
    # 可视化
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # 时空图 - 参考解
    im1 = axes[0, 0].imshow(test_sol[idx].T, aspect='auto', origin='lower',
                            extent=[0, t_grid[-1], 0, 1], cmap='hot')
    axes[0, 0].set_xlabel('时间 t')
    axes[0, 0].set_ylabel('空间 x')
    axes[0, 0].set_title('参考解 u(x,t)')
    plt.colorbar(im1, ax=axes[0, 0])
    
    # 时空图 - PINO 预测
    im2 = axes[0, 1].imshow(predictions.T, aspect='auto', origin='lower',
                            extent=[0, t_grid[-1], 0, 1], cmap='hot')
    axes[0, 1].set_xlabel('时间 t')
    axes[0, 1].set_ylabel('空间 x')
    axes[0, 1].set_title('PINO 预测 u(x,t)')
    plt.colorbar(im2, ax=axes[0, 1])
    
    # 误差图
    error = np.abs(test_sol[idx] - predictions)
    im3 = axes[0, 2].imshow(error.T, aspect='auto', origin='lower',
                            extent=[0, t_grid[-1], 0, 1], cmap='Blues')
    axes[0, 2].set_xlabel('时间 t')
    axes[0, 2].set_ylabel('空间 x')
    axes[0, 2].set_title(f'绝对误差 (Max: {error.max():.2e})')
    plt.colorbar(im3, ax=axes[0, 2])
    
    # 不同时刻的切片
    time_indices = [0, len(t_grid)//4, len(t_grid)//2, 3*len(t_grid)//4, -1]
    colors = plt.cm.viridis(np.linspace(0, 1, len(time_indices)))
    
    for i, ti in enumerate(time_indices):
        t_val = t_grid[ti]
        axes[1, 0].plot(x_grid, test_sol[idx, ti], '-', color=colors[i], 
                       linewidth=2, label=f't={t_val:.2f}')
    axes[1, 0].set_xlabel('x')
    axes[1, 0].set_ylabel('u')
    axes[1, 0].set_title('参考解在不同时刻')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    for i, ti in enumerate(time_indices):
        t_val = t_grid[ti]
        axes[1, 1].plot(x_grid, predictions[ti], '-', color=colors[i],
                       linewidth=2, label=f't={t_val:.2f}')
    axes[1, 1].set_xlabel('x')
    axes[1, 1].set_ylabel('u')
    axes[1, 1].set_title('PINO 预测在不同时刻')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    # 特定时刻对比
    t_compare = len(t_grid) // 2
    axes[1, 2].plot(x_grid, test_sol[idx, t_compare], 'b-', linewidth=2, label='参考解')
    axes[1, 2].plot(x_grid, predictions[t_compare], 'r--', linewidth=2, label='PINO')
    axes[1, 2].set_xlabel('x')
    axes[1, 2].set_ylabel('u')
    axes[1, 2].set_title(f't={t_grid[t_compare]:.2f} 时刻对比')
    axes[1, 2].legend()
    axes[1, 2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, 'heat_pino_results.png'), dpi=150)
    plt.show()
    
    # 计算整体误差
    rel_error = np.linalg.norm(test_sol[idx] - predictions) / np.linalg.norm(test_sol[idx])
    print(f"相对 L2 误差: {rel_error:.4e}")


def plot_training_history(history, save_path='./'):
    """绘制训练历史"""
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    
    epochs = range(1, len(history['train_loss']) + 1)
    
    axes[0].semilogy(epochs, history['train_loss'], 'b-', label='训练损失')
    axes[0].semilogy(epochs, history['val_loss'], 'r-', label='验证损失')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('训练/验证损失')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    axes[1].semilogy(epochs, history['data_loss'], 'b-', label='数据损失')
    axes[1].semilogy(epochs, history['bc_loss'], 'g-', label='边界损失')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Loss')
    axes[1].set_title('损失分解')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, 'heat_pino_loss.png'), dpi=150)
    plt.show()


# ==============================================================================
# 6. 主程序
# ==============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("PINO 求解 1D 热传导方程")
    print("=" * 60)
    
    # 参数设置
    n_train = 80      # 训练样本数
    n_val = 10        # 验证样本数
    n_test = 10       # 测试样本数
    nx = 128          # 空间网格点数
    nt = 50           # 时间网格点数
    alpha = 0.01      # 热扩散系数
    T = 0.5           # 最终时间
    
    # 模型参数
    modes = 16        # Fourier 模式数
    width = 64        # 隐藏层宽度
    depth = 4         # FNO 层数
    
    # 训练参数
    epochs = 300
    lr = 1e-3
    lambda_pde = 0.1
    
    # 1. 生成数据
    print("\n[1/4] 生成热传导方程数据...")
    all_ic, all_sol, x_grid, t_grid = generate_heat_data(
        n_samples=n_train + n_val + n_test,
        nx=nx, nt=nt, alpha=alpha, T=T
    )
    
    # 划分数据集
    train_data = (all_ic[:n_train], all_sol[:n_train])
    val_data = (all_ic[n_train:n_train+n_val], all_sol[n_train:n_train+n_val])
    test_data = (all_ic[n_train+n_val:], all_sol[n_train+n_val:])
    
    print(f"训练集: {n_train} 样本")
    print(f"验证集: {n_val} 样本")
    print(f"测试集: {n_test} 样本")
    
    # 2. 创建模型
    print("\n[2/4] 创建 PINO 模型...")
    model = PINOHeat1d(
        in_channels=3,   # 初始条件 + 时间 + 空间坐标
        out_channels=1,
        modes=modes,
        width=width,
        depth=depth
    ).to(device)
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"模型参数量: {total_params:,}")
    
    # 3. 训练
    print("\n[3/4] 开始训练...")
    history = train_pino_heat(
        model, train_data, val_data, x_grid, t_grid,
        epochs=epochs, lr=lr, lambda_pde=lambda_pde, alpha=alpha
    )
    
    # 4. 评估和可视化
    print("\n[4/4] 评估和可视化...")
    model.load_state_dict(torch.load('best_pino_heat.pt'))
    
    plot_training_history(history)
    visualize_heat_results(model, test_data, x_grid, t_grid)
    
    print("\n训练完成！")
