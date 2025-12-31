"""
2D Darcy 流动 PINO 求解示例
Physics-Informed Neural Operator for 2D Darcy Flow

问题描述:
    -∇·(a(x,y)∇u(x,y)) = f(x,y), (x,y) ∈ [0,1]²
    边界条件: u = 0 on ∂Ω

    其中:
    - a(x,y): 渗透率场 (permeability field)
    - u(x,y): 压力场 (pressure field)
    - f(x,y): 源项 (source term)

PINO 学习目标:
    学习算子 G: a(x,y) → u(x,y)
    即从渗透率场映射到压力场

应用场景:
    - 地下水流动
    - 多孔介质渗流
    - 石油开采模拟
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
# 1. 2D 谱卷积层定义
# ==============================================================================

class SpectralConv2d(nn.Module):
    """2D 谱卷积层 (Fourier Layer)"""
    
    def __init__(self, in_channels, out_channels, modes1, modes2):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1  # x 方向 Fourier 模式数
        self.modes2 = modes2  # y 方向 Fourier 模式数
        
        self.scale = 1 / (in_channels * out_channels)
        
        # 可学习的 Fourier 系数 (复数)
        self.weights1 = nn.Parameter(
            self.scale * torch.rand(in_channels, out_channels, modes1, modes2, 
                                   dtype=torch.cfloat)
        )
        self.weights2 = nn.Parameter(
            self.scale * torch.rand(in_channels, out_channels, modes1, modes2,
                                   dtype=torch.cfloat)
        )
    
    def compl_mul2d(self, input, weights):
        """复数矩阵乘法"""
        # (batch, in_channel, x, y), (in_channel, out_channel, x, y) -> (batch, out_channel, x, y)
        return torch.einsum("bixy,ioxy->boxy", input, weights)
    
    def forward(self, x):
        """
        x: [batch, channels, height, width]
        """
        batch_size = x.shape[0]
        
        # 2D FFT
        x_ft = torch.fft.rfft2(x)
        
        # 在低频模式上进行线性变换
        out_ft = torch.zeros(batch_size, self.out_channels, x.size(-2), 
                            x.size(-1) // 2 + 1, dtype=torch.cfloat, device=x.device)
        
        # 四个角落的低频模式
        out_ft[:, :, :self.modes1, :self.modes2] = \
            self.compl_mul2d(x_ft[:, :, :self.modes1, :self.modes2], self.weights1)
        out_ft[:, :, -self.modes1:, :self.modes2] = \
            self.compl_mul2d(x_ft[:, :, -self.modes1:, :self.modes2], self.weights2)
        
        # 2D 逆 FFT
        x = torch.fft.irfft2(out_ft, s=(x.size(-2), x.size(-1)))
        
        return x


class FNOBlock2d(nn.Module):
    """2D FNO 基本块"""
    
    def __init__(self, width, modes1, modes2):
        super().__init__()
        self.spectral_conv = SpectralConv2d(width, width, modes1, modes2)
        self.w = nn.Conv2d(width, width, 1)
        self.bn = nn.BatchNorm2d(width)
    
    def forward(self, x):
        x1 = self.spectral_conv(x)
        x2 = self.w(x)
        x = x1 + x2
        x = self.bn(x)
        x = F.gelu(x)
        return x


class PINO2d(nn.Module):
    """
    Physics-Informed Neural Operator for 2D problems
    
    架构:
        输入 -> Lifting -> [FNO Block] x depth -> Projection -> 输出
    """
    
    def __init__(self, in_channels=3, out_channels=1, modes1=12, modes2=12, 
                 width=32, depth=4):
        super().__init__()
        self.modes1 = modes1
        self.modes2 = modes2
        self.width = width
        self.depth = depth
        
        # Padding for non-periodic boundaries
        self.padding = 8
        
        # Lifting layer
        self.fc0 = nn.Linear(in_channels, width)
        
        # FNO blocks
        self.fno_blocks = nn.ModuleList([
            FNOBlock2d(width, modes1, modes2) for _ in range(depth)
        ])
        
        # Projection layers
        self.fc1 = nn.Linear(width, 128)
        self.fc2 = nn.Linear(128, out_channels)
    
    def forward(self, x):
        """
        x: [batch, height, width, channels]
        输出: [batch, height, width, out_channels]
        """
        # Lifting
        x = self.fc0(x)
        x = x.permute(0, 3, 1, 2)  # [batch, width, height, width_spatial]
        
        # Padding for non-periodic BC
        x = F.pad(x, [0, self.padding, 0, self.padding])
        
        # FNO blocks
        for block in self.fno_blocks:
            x = block(x)
        
        # Remove padding
        x = x[..., :-self.padding, :-self.padding]
        
        # Projection
        x = x.permute(0, 2, 3, 1)  # [batch, height, width, channels]
        x = F.gelu(self.fc1(x))
        x = self.fc2(x)
        
        return x


# ==============================================================================
# 2. 数据生成
# ==============================================================================

def generate_grf(n_samples, nx, ny, alpha=2.0, tau=3.0, sigma=None):
    """
    生成 Gaussian Random Field (高斯随机场)
    用于模拟随机渗透率场 a(x,y)
    
    使用谱方法: a(x) ~ sum of random Fourier modes with decaying magnitude
    
    参数:
        n_samples: 样本数量
        nx, ny: 网格尺寸
        alpha: 衰减指数
        tau: 长度尺度
        sigma: 方差控制
    """
    if sigma is None:
        sigma = tau ** (2 - 1)
    
    # 波数网格
    kx = np.fft.fftfreq(nx) * nx
    ky = np.fft.fftfreq(ny) * ny
    kx, ky = np.meshgrid(kx, ky, indexing='ij')
    
    # 功率谱衰减
    k_norm = np.sqrt(kx**2 + ky**2) + 1e-8
    coef = sigma * (tau**2 * (2*np.pi)**2 + k_norm**2) ** (-alpha/2)
    coef[0, 0] = 0  # 去除零频分量
    
    samples = []
    for _ in range(n_samples):
        # 随机 Fourier 系数
        xi = np.random.randn(nx, ny) + 1j * np.random.randn(nx, ny)
        
        # 应用功率谱
        f_hat = coef * xi
        
        # 逆 FFT
        f = np.real(np.fft.ifft2(f_hat))
        
        # 转换到正值 (渗透率必须为正)
        f = np.exp(f)
        
        samples.append(f)
    
    return np.array(samples)


def solve_darcy_fd(a, f, nx, ny):
    """
    使用有限差分法求解 2D Darcy 方程
    
    -∇·(a∇u) = f, with u=0 on boundary
    
    参数:
        a: [nx, ny] 渗透率场
        f: [nx, ny] 源项
        nx, ny: 网格尺寸
    
    返回:
        u: [nx, ny] 压力场解
    """
    dx = 1.0 / (nx - 1)
    dy = 1.0 / (ny - 1)
    
    # 构建稀疏矩阵 (简化实现，使用迭代法)
    # 这里使用 Gauss-Seidel 迭代
    u = np.zeros((nx, ny))
    
    # 迭代求解
    for _ in range(1000):
        u_old = u.copy()
        
        for i in range(1, nx-1):
            for j in range(1, ny-1):
                # 5点差分格式
                ax = 0.5 * (a[i+1, j] + a[i, j])
                bx = 0.5 * (a[i, j] + a[i-1, j])
                ay = 0.5 * (a[i, j+1] + a[i, j])
                by = 0.5 * (a[i, j] + a[i, j-1])
                
                coef = ax + bx + ay + by
                
                u[i, j] = (ax * u[i+1, j] + bx * u[i-1, j] + 
                          ay * u[i, j+1] + by * u[i, j-1] + 
                          dx * dy * f[i, j]) / coef
        
        # 检查收敛
        if np.max(np.abs(u - u_old)) < 1e-6:
            break
    
    return u


def generate_darcy_data(n_samples=100, nx=64, ny=64):
    """
    生成 Darcy 流动数据集
    
    返回:
        permeability: [n_samples, nx, ny] 渗透率场
        pressure: [n_samples, nx, ny] 压力场
        grid_x, grid_y: 空间网格
    """
    print("生成高斯随机渗透率场...")
    permeability = generate_grf(n_samples, nx, ny)
    
    # 归一化渗透率到合理范围
    permeability = 3 + 9 * (permeability - permeability.min()) / \
                   (permeability.max() - permeability.min())
    
    # 常数源项
    f = np.ones((nx, ny))
    
    print("求解 Darcy 方程...")
    pressure = []
    for i in tqdm(range(n_samples)):
        u = solve_darcy_fd(permeability[i], f, nx, ny)
        pressure.append(u)
    
    pressure = np.array(pressure)
    
    # 空间网格
    x = np.linspace(0, 1, nx)
    y = np.linspace(0, 1, ny)
    grid_x, grid_y = np.meshgrid(x, y, indexing='ij')
    
    return permeability, pressure, grid_x, grid_y


# ==============================================================================
# 3. 物理残差计算
# ==============================================================================

def compute_darcy_residual(a, u, f=1.0, dx=1.0/63):
    """
    计算 Darcy 方程的 PDE 残差
    
    PDE: -∇·(a∇u) = f
    
    参数:
        a: [batch, nx, ny] 渗透率场
        u: [batch, nx, ny] 预测压力场
        f: 源项 (标量或 [batch, nx, ny])
        dx: 网格间距
    
    返回:
        residual: [batch, nx, ny] PDE 残差
    """
    batch_size, nx, ny = u.shape
    
    # 使用中心差分计算梯度
    u_x = torch.zeros_like(u)
    u_y = torch.zeros_like(u)
    
    u_x[:, 1:-1, :] = (u[:, 2:, :] - u[:, :-2, :]) / (2 * dx)
    u_y[:, :, 1:-1] = (u[:, :, 2:] - u[:, :, :-2]) / (2 * dx)
    
    # a * ∇u
    au_x = a * u_x
    au_y = a * u_y
    
    # ∇·(a∇u)
    div_au = torch.zeros_like(u)
    div_au[:, 1:-1, :] += (au_x[:, 2:, :] - au_x[:, :-2, :]) / (2 * dx)
    div_au[:, :, 1:-1] += (au_y[:, :, 2:] - au_y[:, :, :-2]) / (2 * dx)
    
    # PDE 残差
    residual = -div_au - f
    
    return residual


# ==============================================================================
# 4. 训练循环
# ==============================================================================

def train_pino_darcy(model, train_data, val_data, grid_x, grid_y,
                     epochs=500, lr=1e-3, lambda_pde=0.1, batch_size=16):
    """
    PINO 训练函数 (Darcy 流动)
    """
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    
    train_a, train_u = train_data
    val_a, val_u = val_data
    
    # 转换为 PyTorch 张量
    train_a = torch.FloatTensor(train_a).to(device)
    train_u = torch.FloatTensor(train_u).to(device)
    val_a = torch.FloatTensor(val_a).to(device)
    val_u = torch.FloatTensor(val_u).to(device)
    
    # 网格坐标
    grid_x_t = torch.FloatTensor(grid_x).unsqueeze(0).to(device)
    grid_y_t = torch.FloatTensor(grid_y).unsqueeze(0).to(device)
    
    n_train = train_a.shape[0]
    n_val = val_a.shape[0]
    
    history = {'train_loss': [], 'val_loss': [], 'data_loss': [], 'pde_loss': []}
    best_val_loss = float('inf')
    
    for epoch in range(epochs):
        model.train()
        
        # 随机打乱
        perm = torch.randperm(n_train)
        
        total_loss = 0
        total_data_loss = 0
        total_pde_loss = 0
        n_batches = 0
        
        for i in range(0, n_train, batch_size):
            idx = perm[i:min(i+batch_size, n_train)]
            
            a_batch = train_a[idx]  # [batch, nx, ny]
            u_batch = train_u[idx]  # [batch, nx, ny]
            
            # 准备输入: [batch, nx, ny, 3] = [a, x, y]
            bs = a_batch.shape[0]
            grid_x_batch = grid_x_t.expand(bs, -1, -1)
            grid_y_batch = grid_y_t.expand(bs, -1, -1)
            
            x_input = torch.stack([a_batch, grid_x_batch, grid_y_batch], dim=-1)
            
            optimizer.zero_grad()
            
            # 前向传播
            pred = model(x_input).squeeze(-1)  # [batch, nx, ny]
            
            # 数据损失
            data_loss = F.mse_loss(pred, u_batch)
            
            # 物理损失
            pde_residual = compute_darcy_residual(a_batch, pred)
            pde_loss = torch.mean(pde_residual[:, 1:-1, 1:-1]**2)  # 内部点
            
            # 边界损失 (Dirichlet BC: u=0)
            bc_loss = torch.mean(pred[:, 0, :]**2) + torch.mean(pred[:, -1, :]**2) + \
                     torch.mean(pred[:, :, 0]**2) + torch.mean(pred[:, :, -1]**2)
            
            # 总损失
            loss = data_loss + lambda_pde * pde_loss + 0.1 * bc_loss
            
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            total_data_loss += data_loss.item()
            total_pde_loss += pde_loss.item()
            n_batches += 1
        
        scheduler.step()
        
        # 验证
        model.eval()
        with torch.no_grad():
            grid_x_val = grid_x_t.expand(n_val, -1, -1)
            grid_y_val = grid_y_t.expand(n_val, -1, -1)
            val_input = torch.stack([val_a, grid_x_val, grid_y_val], dim=-1)
            val_pred = model(val_input).squeeze(-1)
            val_loss = F.mse_loss(val_pred, val_u)
        
        avg_loss = total_loss / n_batches
        avg_data_loss = total_data_loss / n_batches
        avg_pde_loss = total_pde_loss / n_batches
        
        history['train_loss'].append(avg_loss)
        history['val_loss'].append(val_loss.item())
        history['data_loss'].append(avg_data_loss)
        history['pde_loss'].append(avg_pde_loss)
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), 'best_pino_darcy.pt')
        
        if (epoch + 1) % 50 == 0:
            print(f"Epoch {epoch+1}/{epochs} | "
                  f"Train: {avg_loss:.6f} | Val: {val_loss.item():.6f} | "
                  f"Data: {avg_data_loss:.6f} | PDE: {avg_pde_loss:.6f}")
    
    return history


# ==============================================================================
# 5. 可视化
# ==============================================================================

def visualize_darcy_results(model, test_data, grid_x, grid_y, save_path='./'):
    """可视化 Darcy 流动 PINO 预测结果"""
    
    test_a, test_u = test_data
    test_a_t = torch.FloatTensor(test_a).to(device)
    
    n_test = test_a.shape[0]
    grid_x_t = torch.FloatTensor(grid_x).unsqueeze(0).expand(n_test, -1, -1).to(device)
    grid_y_t = torch.FloatTensor(grid_y).unsqueeze(0).expand(n_test, -1, -1).to(device)
    
    test_input = torch.stack([test_a_t, grid_x_t, grid_y_t], dim=-1)
    
    model.eval()
    with torch.no_grad():
        pred = model(test_input).squeeze(-1).cpu().numpy()
    
    # 选择一个样本可视化
    idx = 0
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # 渗透率场
    im1 = axes[0, 0].imshow(test_a[idx].T, origin='lower', cmap='viridis')
    axes[0, 0].set_title('渗透率场 a(x,y)')
    axes[0, 0].set_xlabel('x')
    axes[0, 0].set_ylabel('y')
    plt.colorbar(im1, ax=axes[0, 0])
    
    # 参考压力场
    im2 = axes[0, 1].imshow(test_u[idx].T, origin='lower', cmap='RdBu_r')
    axes[0, 1].set_title('参考压力场')
    axes[0, 1].set_xlabel('x')
    axes[0, 1].set_ylabel('y')
    plt.colorbar(im2, ax=axes[0, 1])
    
    # PINO 预测
    im3 = axes[0, 2].imshow(pred[idx].T, origin='lower', cmap='RdBu_r')
    axes[0, 2].set_title('PINO 预测压力场')
    axes[0, 2].set_xlabel('x')
    axes[0, 2].set_ylabel('y')
    plt.colorbar(im3, ax=axes[0, 2])
    
    # 误差场
    error = np.abs(test_u[idx] - pred[idx])
    im4 = axes[1, 0].imshow(error.T, origin='lower', cmap='hot')
    axes[1, 0].set_title(f'绝对误差 (Max: {error.max():.2e})')
    axes[1, 0].set_xlabel('x')
    axes[1, 0].set_ylabel('y')
    plt.colorbar(im4, ax=axes[1, 0])
    
    # 沿 x=0.5 的切片
    nx = test_a.shape[1]
    mid_x = nx // 2
    axes[1, 1].plot(grid_y[mid_x, :], test_u[idx, mid_x, :], 'b-', linewidth=2, label='参考解')
    axes[1, 1].plot(grid_y[mid_x, :], pred[idx, mid_x, :], 'r--', linewidth=2, label='PINO')
    axes[1, 1].set_xlabel('y')
    axes[1, 1].set_ylabel('u')
    axes[1, 1].set_title('x=0.5 处的压力分布')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    # 统计误差
    all_errors = np.abs(test_u - pred)
    rel_errors = np.linalg.norm(test_u - pred, axis=(1, 2)) / np.linalg.norm(test_u, axis=(1, 2))
    
    axes[1, 2].boxplot(rel_errors)
    axes[1, 2].set_ylabel('相对 L2 误差')
    axes[1, 2].set_title(f'测试集误差统计 (Mean: {rel_errors.mean():.4f})')
    axes[1, 2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, 'darcy_pino_results.png'), dpi=150)
    plt.show()
    
    print(f"平均相对 L2 误差: {rel_errors.mean():.4e}")
    print(f"最大相对 L2 误差: {rel_errors.max():.4e}")


# ==============================================================================
# 6. 主程序
# ==============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("PINO 求解 2D Darcy 流动方程")
    print("=" * 60)
    
    # 参数设置
    n_train = 80      # 训练样本数
    n_val = 10        # 验证样本数
    n_test = 10       # 测试样本数
    nx, ny = 64, 64   # 空间网格
    
    # 模型参数
    modes = 12        # Fourier 模式数
    width = 32        # 隐藏层宽度
    depth = 4         # FNO 层数
    
    # 训练参数
    epochs = 300
    lr = 1e-3
    lambda_pde = 0.1
    batch_size = 16
    
    # 1. 生成数据
    print("\n[1/4] 生成 Darcy 流动数据...")
    all_a, all_u, grid_x, grid_y = generate_darcy_data(
        n_samples=n_train + n_val + n_test,
        nx=nx, ny=ny
    )
    
    # 划分数据集
    train_data = (all_a[:n_train], all_u[:n_train])
    val_data = (all_a[n_train:n_train+n_val], all_u[n_train:n_train+n_val])
    test_data = (all_a[n_train+n_val:], all_u[n_train+n_val:])
    
    print(f"训练集: {n_train} 样本")
    print(f"验证集: {n_val} 样本")
    print(f"测试集: {n_test} 样本")
    print(f"网格尺寸: {nx} x {ny}")
    
    # 2. 创建模型
    print("\n[2/4] 创建 PINO 模型...")
    model = PINO2d(
        in_channels=3,   # 渗透率 + x坐标 + y坐标
        out_channels=1,
        modes1=modes,
        modes2=modes,
        width=width,
        depth=depth
    ).to(device)
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"模型参数量: {total_params:,}")
    
    # 3. 训练
    print("\n[3/4] 开始训练...")
    history = train_pino_darcy(
        model, train_data, val_data, grid_x, grid_y,
        epochs=epochs, lr=lr, lambda_pde=lambda_pde, batch_size=batch_size
    )
    
    # 4. 评估和可视化
    print("\n[4/4] 评估和可视化...")
    model.load_state_dict(torch.load('best_pino_darcy.pt'))
    
    visualize_darcy_results(model, test_data, grid_x, grid_y)
    
    print("\n训练完成！")
