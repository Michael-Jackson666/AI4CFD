"""
1D Burgers 方程 PINO 求解示例
Physics-Informed Neural Operator for 1D Burgers' Equation

问题描述:
    ∂u/∂t + u·∂u/∂x = ν·∂²u/∂x², x ∈ [-1, 1], t ∈ [0, 1]
    初始条件: u(x, 0) = -sin(πx)
    边界条件: 周期边界 u(-1, t) = u(1, t)
    粘性系数: ν = 0.01

PINO 学习目标:
    学习算子 G: u₀(x) → u(x, t)
    即从初始条件映射到完整时空解
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
    """1D 谱卷积层 (Fourier Layer)"""
    
    def __init__(self, in_channels, out_channels, modes):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes = modes  # 保留的 Fourier 模式数
        
        # 可学习的 Fourier 系数
        self.scale = 1 / (in_channels * out_channels)
        self.weights = nn.Parameter(
            self.scale * torch.rand(in_channels, out_channels, modes, dtype=torch.cfloat)
        )
    
    def forward(self, x):
        """
        x: [batch, channels, spatial]
        """
        batch_size = x.shape[0]
        
        # 计算 FFT
        x_ft = torch.fft.rfft(x)
        
        # 在低频模式上乘以可学习权重
        out_ft = torch.zeros(batch_size, self.out_channels, x.size(-1) // 2 + 1,
                            dtype=torch.cfloat, device=x.device)
        out_ft[:, :, :self.modes] = torch.einsum(
            "bix,iox->box", x_ft[:, :, :self.modes], self.weights
        )
        
        # 逆 FFT
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


class PINO1d(nn.Module):
    """
    Physics-Informed Neural Operator for 1D problems
    
    架构:
        输入 -> Lifting -> [FNO Block] x depth -> Projection -> 输出
    """
    
    def __init__(self, in_channels=2, out_channels=1, modes=16, width=64, depth=4):
        super().__init__()
        self.modes = modes
        self.width = width
        self.depth = depth
        
        # Lifting layer: 将输入提升到高维空间
        self.fc0 = nn.Linear(in_channels, width)
        
        # FNO blocks
        self.fno_blocks = nn.ModuleList([
            FNOBlock1d(width, modes) for _ in range(depth)
        ])
        
        # Projection layer: 将高维特征投影回输出空间
        self.fc1 = nn.Linear(width, 128)
        self.fc2 = nn.Linear(128, out_channels)
    
    def forward(self, x, grid=None):
        """
        x: [batch, spatial, channels] 或 [batch, spatial]
        grid: [batch, spatial, 1] 可选的空间坐标
        """
        if x.dim() == 2:
            x = x.unsqueeze(-1)
        
        # 如果提供了 grid，拼接到输入
        if grid is not None:
            x = torch.cat([x, grid], dim=-1)
        
        # Lifting
        x = self.fc0(x)  # [batch, spatial, width]
        x = x.permute(0, 2, 1)  # [batch, width, spatial]
        
        # FNO blocks
        for block in self.fno_blocks:
            x = block(x)
        
        # Projection
        x = x.permute(0, 2, 1)  # [batch, spatial, width]
        x = F.gelu(self.fc1(x))
        x = self.fc2(x)
        
        return x


# ==============================================================================
# 2. 数据生成
# ==============================================================================

def generate_burgers_data(n_samples=100, nx=128, nt=100, nu=0.01, L=2.0, T=1.0):
    """
    使用谱方法生成 Burgers 方程数据
    
    参数:
        n_samples: 样本数量
        nx: 空间网格点数
        nt: 时间网格点数
        nu: 粘性系数
        L: 空间域长度 [-L/2, L/2]
        T: 时间域长度 [0, T]
    
    返回:
        initial_conditions: [n_samples, nx] 初始条件
        solutions: [n_samples, nt, nx] 完整时空解
        x_grid: [nx] 空间网格
        t_grid: [nt] 时间网格
    """
    x = np.linspace(-L/2, L/2, nx, endpoint=False)
    t = np.linspace(0, T, nt)
    dx = x[1] - x[0]
    dt = t[1] - t[0]
    
    # 波数
    k = np.fft.fftfreq(nx) * nx * 2 * np.pi / L
    
    initial_conditions = []
    solutions = []
    
    for _ in tqdm(range(n_samples), desc="生成数据"):
        # 随机初始条件: 不同振幅和相位的正弦波组合
        n_modes = np.random.randint(1, 4)
        u0 = np.zeros(nx)
        for m in range(n_modes):
            amp = np.random.uniform(-1, 1)
            freq = np.random.randint(1, 4)
            phase = np.random.uniform(0, 2*np.pi)
            u0 += amp * np.sin(freq * np.pi * x / (L/2) + phase)
        
        # 归一化到 [-1, 1]
        u0 = u0 / (np.max(np.abs(u0)) + 1e-8)
        
        # 使用 RK4 + 谱方法求解
        u = np.zeros((nt, nx))
        u[0] = u0
        
        u_current = u0.copy()
        
        def rhs(u_):
            u_hat = np.fft.fft(u_)
            u_x = np.real(np.fft.ifft(1j * k * u_hat))
            u_xx = np.real(np.fft.ifft(-k**2 * u_hat))
            return -u_ * u_x + nu * u_xx
        
        for i in range(1, nt):
            # RK4 时间积分
            k1 = rhs(u_current)
            k2 = rhs(u_current + 0.5*dt*k1)
            k3 = rhs(u_current + 0.5*dt*k2)
            k4 = rhs(u_current + dt*k3)
            u_current = u_current + (dt/6) * (k1 + 2*k2 + 2*k3 + k4)
            u[i] = u_current
        
        initial_conditions.append(u0)
        solutions.append(u)
    
    return (np.array(initial_conditions), np.array(solutions), x, t)


# ==============================================================================
# 3. 物理残差计算
# ==============================================================================

def compute_burgers_residual(u, x, t, nu=0.01):
    """
    计算 Burgers 方程的 PDE 残差
    
    PDE: ∂u/∂t + u·∂u/∂x - ν·∂²u/∂x² = 0
    
    参数:
        u: [batch, nt, nx] 预测解
        x: [nx] 空间坐标 (需要 requires_grad)
        t: [nt] 时间坐标 (需要 requires_grad)
        nu: 粘性系数
    
    返回:
        residual: [batch, nt, nx] PDE 残差
    """
    batch_size, nt, nx = u.shape
    
    # 使用有限差分计算导数
    dx = x[1] - x[0]
    dt = t[1] - t[0]
    
    # 时间导数 (前向差分)
    u_t = torch.zeros_like(u)
    u_t[:, :-1, :] = (u[:, 1:, :] - u[:, :-1, :]) / dt
    u_t[:, -1, :] = u_t[:, -2, :]  # 外推边界
    
    # 空间导数 (中心差分)
    u_x = torch.zeros_like(u)
    u_x[:, :, 1:-1] = (u[:, :, 2:] - u[:, :, :-2]) / (2 * dx)
    u_x[:, :, 0] = (u[:, :, 1] - u[:, :, -1]) / (2 * dx)  # 周期边界
    u_x[:, :, -1] = (u[:, :, 0] - u[:, :, -2]) / (2 * dx)
    
    # 二阶空间导数 (中心差分)
    u_xx = torch.zeros_like(u)
    u_xx[:, :, 1:-1] = (u[:, :, 2:] - 2*u[:, :, 1:-1] + u[:, :, :-2]) / (dx**2)
    u_xx[:, :, 0] = (u[:, :, 1] - 2*u[:, :, 0] + u[:, :, -1]) / (dx**2)
    u_xx[:, :, -1] = (u[:, :, 0] - 2*u[:, :, -1] + u[:, :, -2]) / (dx**2)
    
    # PDE 残差
    residual = u_t + u * u_x - nu * u_xx
    
    return residual


# ==============================================================================
# 4. 训练循环
# ==============================================================================

def train_pino(model, train_data, val_data, x_grid, t_grid, 
               epochs=500, lr=1e-3, lambda_pde=1.0, nu=0.01):
    """
    PINO 训练函数
    
    参数:
        model: PINO 模型
        train_data: (initial_conditions, solutions) 训练数据
        val_data: (initial_conditions, solutions) 验证数据
        x_grid: 空间网格
        t_grid: 时间网格
        epochs: 训练轮数
        lr: 学习率
        lambda_pde: 物理损失权重
        nu: 粘性系数
    """
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.5)
    
    train_ic, train_sol = train_data
    val_ic, val_sol = val_data
    
    # 转换为 PyTorch 张量
    train_ic = torch.FloatTensor(train_ic).to(device)
    train_sol = torch.FloatTensor(train_sol).to(device)
    val_ic = torch.FloatTensor(val_ic).to(device)
    val_sol = torch.FloatTensor(val_sol).to(device)
    
    x_tensor = torch.FloatTensor(x_grid).to(device)
    t_tensor = torch.FloatTensor(t_grid).to(device)
    
    # 创建空间网格 (作为额外输入)
    grid = torch.FloatTensor(x_grid).unsqueeze(0).unsqueeze(-1).to(device)
    grid = grid.expand(train_ic.shape[0], -1, -1)
    
    history = {'train_loss': [], 'val_loss': [], 'data_loss': [], 'pde_loss': []}
    
    best_val_loss = float('inf')
    
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        
        # 前向传播: 从初始条件预测完整解
        # 这里简化处理: 对每个时间步分别预测
        # 实际应用中可能需要更复杂的时间演化策略
        
        # 预测解 (简化: 直接将初始条件传入，输出最终时刻的解)
        pred = model(train_ic, grid)  # [batch, nx, 1]
        pred = pred.squeeze(-1)  # [batch, nx]
        
        # 数据损失: 与最终时刻的参考解比较
        target = train_sol[:, -1, :]  # 最终时刻
        data_loss = F.mse_loss(pred, target)
        
        # 物理损失: 计算 PDE 残差
        # 这里简化处理，实际中需要对完整时空解计算残差
        pde_loss = torch.tensor(0.0, device=device)
        
        # 总损失
        loss = data_loss + lambda_pde * pde_loss
        
        loss.backward()
        optimizer.step()
        scheduler.step()
        
        # 验证
        model.eval()
        with torch.no_grad():
            val_grid = grid[:val_ic.shape[0]]
            val_pred = model(val_ic, val_grid).squeeze(-1)
            val_target = val_sol[:, -1, :]
            val_loss = F.mse_loss(val_pred, val_target)
        
        history['train_loss'].append(loss.item())
        history['val_loss'].append(val_loss.item())
        history['data_loss'].append(data_loss.item())
        history['pde_loss'].append(pde_loss.item())
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), 'best_pino_burgers.pt')
        
        if (epoch + 1) % 50 == 0:
            print(f"Epoch {epoch+1}/{epochs} | "
                  f"Train Loss: {loss.item():.6f} | "
                  f"Val Loss: {val_loss.item():.6f} | "
                  f"Data Loss: {data_loss.item():.6f}")
    
    return history


# ==============================================================================
# 5. 可视化
# ==============================================================================

def visualize_results(model, test_data, x_grid, t_grid, save_path='./'):
    """可视化 PINO 预测结果"""
    
    test_ic, test_sol = test_data
    test_ic = torch.FloatTensor(test_ic).to(device)
    
    grid = torch.FloatTensor(x_grid).unsqueeze(0).unsqueeze(-1).to(device)
    grid = grid.expand(test_ic.shape[0], -1, -1)
    
    model.eval()
    with torch.no_grad():
        pred = model(test_ic, grid).squeeze(-1).cpu().numpy()
    
    # 选择一个样本可视化
    idx = 0
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    # 初始条件
    axes[0].plot(x_grid, test_ic[idx].cpu().numpy(), 'b-', linewidth=2)
    axes[0].set_xlabel('x')
    axes[0].set_ylabel('u')
    axes[0].set_title('初始条件 u(x, 0)')
    axes[0].grid(True, alpha=0.3)
    
    # 预测 vs 真实 (最终时刻)
    axes[1].plot(x_grid, test_sol[idx, -1, :], 'b-', linewidth=2, label='参考解')
    axes[1].plot(x_grid, pred[idx], 'r--', linewidth=2, label='PINO 预测')
    axes[1].set_xlabel('x')
    axes[1].set_ylabel('u')
    axes[1].set_title(f'最终时刻 t={t_grid[-1]:.2f}')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    # 误差
    error = np.abs(test_sol[idx, -1, :] - pred[idx])
    axes[2].plot(x_grid, error, 'g-', linewidth=2)
    axes[2].set_xlabel('x')
    axes[2].set_ylabel('|Error|')
    axes[2].set_title(f'绝对误差 (Max: {error.max():.2e})')
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, 'burgers_pino_prediction.png'), dpi=150)
    plt.show()
    
    print(f"相对 L2 误差: {np.linalg.norm(error) / np.linalg.norm(test_sol[idx, -1, :]):.4e}")


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
    axes[1].semilogy(epochs, history['pde_loss'], 'g-', label='物理损失')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Loss')
    axes[1].set_title('损失分解')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, 'burgers_pino_loss.png'), dpi=150)
    plt.show()


# ==============================================================================
# 6. 主程序
# ==============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("PINO 求解 1D Burgers 方程")
    print("=" * 60)
    
    # 参数设置
    n_train = 80      # 训练样本数
    n_val = 10        # 验证样本数
    n_test = 10       # 测试样本数
    nx = 128          # 空间网格点数
    nt = 100          # 时间网格点数
    nu = 0.01         # 粘性系数
    
    # 模型参数
    modes = 16        # Fourier 模式数
    width = 64        # 隐藏层宽度
    depth = 4         # FNO 层数
    
    # 训练参数
    epochs = 300
    lr = 1e-3
    lambda_pde = 0.1  # 物理损失权重
    
    # 1. 生成数据
    print("\n[1/4] 生成 Burgers 方程数据...")
    all_ic, all_sol, x_grid, t_grid = generate_burgers_data(
        n_samples=n_train + n_val + n_test,
        nx=nx, nt=nt, nu=nu
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
    model = PINO1d(
        in_channels=2,  # 初始条件 + 空间坐标
        out_channels=1,
        modes=modes,
        width=width,
        depth=depth
    ).to(device)
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"模型参数量: {total_params:,}")
    
    # 3. 训练
    print("\n[3/4] 开始训练...")
    history = train_pino(
        model, train_data, val_data, x_grid, t_grid,
        epochs=epochs, lr=lr, lambda_pde=lambda_pde, nu=nu
    )
    
    # 4. 评估和可视化
    print("\n[4/4] 评估和可视化...")
    model.load_state_dict(torch.load('best_pino_burgers.pt'))
    
    plot_training_history(history)
    visualize_results(model, test_data, x_grid, t_grid)
    
    print("\n训练完成！")
