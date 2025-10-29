"""
简单的Transformer训练脚本
用于快速开始Transformer求解PDE

使用方法:
    python train_simple.py --epochs 100 --batch_size 32 --lr 0.001
"""

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import argparse
import os
import sys

# 添加父目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 尝试导入模型
try:
    from models import SpatialTransformer
except ImportError:
    print("警告: 无法导入SpatialTransformer，将使用本地简化版本")
    SpatialTransformer = None


# ============= 数据生成 =============
def generate_heat_equation_data(num_samples=1000, nx=64, nt=50, alpha=0.01):
    """生成1D热传导方程数据"""
    x = np.linspace(0, 1, nx)
    t = np.linspace(0, 1, nt)
    dx = x[1] - x[0]
    dt = t[1] - t[0]
    
    r = alpha * dt / dx**2
    if r > 0.5:
        print(f"警告: r = {r:.4f} > 0.5, 可能不稳定!")
    
    data_input = []
    data_output = []
    
    for _ in range(num_samples):
        n_modes = np.random.randint(1, 4)
        u = np.zeros((nt, nx))
        
        for mode in range(1, n_modes + 1):
            amp = np.random.uniform(0.5, 1.5)
            u[0] += amp * np.sin(mode * np.pi * x)
        
        u[:, 0] = 0
        u[:, -1] = 0
        
        for n in range(0, nt-1):
            for i in range(1, nx-1):
                u[n+1, i] = u[n, i] + r * (u[n, i+1] - 2*u[n, i] + u[n, i-1])
        
        data_input.append(u[0])
        data_output.append(u[-1])
    
    return np.array(data_input), np.array(data_output), x


# ============= 数据集类 =============
class PDEDataset(Dataset):
    """PDE数据集"""
    
    def __init__(self, inputs, outputs, coords):
        self.inputs = torch.FloatTensor(inputs).unsqueeze(-1)
        self.outputs = torch.FloatTensor(outputs).unsqueeze(-1)
        self.coords = torch.FloatTensor(coords).unsqueeze(0).expand(len(inputs), -1, 1)
    
    def __len__(self):
        return len(self.inputs)
    
    def __getitem__(self, idx):
        return self.inputs[idx], self.coords[idx], self.outputs[idx]


# ============= 简化模型（如果无法导入） =============
class PhysicsPositionalEncoding(nn.Module):
    """基于物理坐标的位置编码"""
    
    def __init__(self, d_model, coord_dim=1):
        super().__init__()
        self.d_model = d_model
        self.coord_proj = nn.Linear(coord_dim, d_model)
        self.freq_bands = nn.Parameter(torch.randn(d_model // 2, coord_dim))
    
    def forward(self, coords):
        linear_encoding = self.coord_proj(coords)
        coords_expanded = coords.unsqueeze(-2)
        freq_expanded = self.freq_bands.unsqueeze(0).unsqueeze(0)
        freqs = torch.sum(coords_expanded * freq_expanded, dim=-1)
        sin_encoding = torch.sin(freqs)
        cos_encoding = torch.cos(freqs)
        freq_encoding = torch.cat([sin_encoding, cos_encoding], dim=-1)
        return linear_encoding + freq_encoding


class SimplePDETransformer(nn.Module):
    """简单的PDE求解Transformer"""
    
    def __init__(self, input_dim=1, output_dim=1, d_model=128, nhead=4, 
                 num_layers=3, coord_dim=1):
        super().__init__()
        
        self.input_proj = nn.Linear(input_dim, d_model)
        self.pos_encoding = PhysicsPositionalEncoding(d_model, coord_dim)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=4 * d_model,
            dropout=0.1,
            activation='gelu',
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)
        
        self.output_proj = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Linear(d_model // 2, output_dim)
        )
        
        self._reset_parameters()
    
    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
    
    def forward(self, x, coords):
        x = self.input_proj(x) + self.pos_encoding(coords)
        x = self.transformer(x)
        return self.output_proj(x)


# ============= 训练函数 =============
def train_epoch(model, dataloader, criterion, optimizer, device):
    """训练一个epoch"""
    model.train()
    total_loss = 0
    
    for inputs, coords, targets in dataloader:
        inputs = inputs.to(device)
        coords = coords.to(device)
        targets = targets.to(device)
        
        outputs = model(inputs, coords)
        loss = criterion(outputs, targets)
        
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        total_loss += loss.item()
    
    return total_loss / len(dataloader)


def evaluate(model, dataloader, criterion, device):
    """评估模型"""
    model.eval()
    total_loss = 0
    
    with torch.no_grad():
        for inputs, coords, targets in dataloader:
            inputs = inputs.to(device)
            coords = coords.to(device)
            targets = targets.to(device)
            
            outputs = model(inputs, coords)
            loss = criterion(outputs, targets)
            total_loss += loss.item()
    
    return total_loss / len(dataloader)


def plot_results(model, test_dataset, device, x_grid, save_path='results.png'):
    """绘制结果"""
    model.eval()
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    axes = axes.flatten()
    
    with torch.no_grad():
        for i in range(6):
            idx = np.random.randint(0, len(test_dataset))
            inputs, coords, targets = test_dataset[idx]
            
            inputs = inputs.unsqueeze(0).to(device)
            coords = coords.unsqueeze(0).to(device)
            
            outputs = model(inputs, coords)
            prediction = outputs.cpu().numpy()[0, :, 0]
            true_output = targets.numpy()[:, 0]
            initial = test_dataset.inputs[idx].numpy()[:, 0]
            
            axes[i].plot(x_grid, initial, 'b--', linewidth=2, label='初始', alpha=0.6)
            axes[i].plot(x_grid, true_output, 'g-', linewidth=2, label='真实')
            axes[i].plot(x_grid, prediction, 'r--', linewidth=2, label='预测')
            
            rel_error = np.linalg.norm(prediction - true_output) / np.linalg.norm(true_output)
            axes[i].set_title(f'样本 {idx} (误差: {rel_error:.4f})')
            axes[i].legend()
            axes[i].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"结果已保存到 {save_path}")
    plt.close()


# ============= 主函数 =============
def main():
    parser = argparse.ArgumentParser(description='训练Transformer求解PDE')
    parser.add_argument('--epochs', type=int, default=100, help='训练轮数')
    parser.add_argument('--batch_size', type=int, default=32, help='批次大小')
    parser.add_argument('--lr', type=float, default=0.001, help='学习率')
    parser.add_argument('--d_model', type=int, default=128, help='模型维度')
    parser.add_argument('--nhead', type=int, default=4, help='注意力头数')
    parser.add_argument('--num_layers', type=int, default=4, help='层数')
    parser.add_argument('--train_samples', type=int, default=800, help='训练样本数')
    parser.add_argument('--test_samples', type=int, default=200, help='测试样本数')
    parser.add_argument('--nx', type=int, default=64, help='空间离散点数')
    parser.add_argument('--save_dir', type=str, default='./results', help='保存目录')
    parser.add_argument('--seed', type=int, default=42, help='随机种子')
    
    args = parser.parse_args()
    
    # 设置随机种子
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    # 创建保存目录
    os.makedirs(args.save_dir, exist_ok=True)
    
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    # 生成数据
    print("\n生成训练数据...")
    X_train, y_train, x_grid = generate_heat_equation_data(
        num_samples=args.train_samples, nx=args.nx
    )
    X_test, y_test, _ = generate_heat_equation_data(
        num_samples=args.test_samples, nx=args.nx
    )
    print(f"训练集: {X_train.shape}, 测试集: {X_test.shape}")
    
    # 创建数据加载器
    train_dataset = PDEDataset(X_train, y_train, x_grid)
    test_dataset = PDEDataset(X_test, y_test, x_grid)
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
    
    # 创建模型
    print("\n创建模型...")
    if SpatialTransformer is not None:
        model = SpatialTransformer(
            input_dim=1,
            output_dim=1,
            d_model=args.d_model,
            nhead=args.nhead,
            num_layers=args.num_layers,
            coord_dim=1
        ).to(device)
    else:
        model = SimplePDETransformer(
            input_dim=1,
            output_dim=1,
            d_model=args.d_model,
            nhead=args.nhead,
            num_layers=args.num_layers,
            coord_dim=1
        ).to(device)
    
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"模型参数量: {num_params:,}")
    
    # 优化器和损失函数
    criterion = nn.MSELoss()
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=1e-6
    )
    
    # 训练
    print("\n开始训练...\n")
    train_losses = []
    test_losses = []
    best_test_loss = float('inf')
    
    for epoch in range(args.epochs):
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device)
        test_loss = evaluate(model, test_loader, criterion, device)
        
        train_losses.append(train_loss)
        test_losses.append(test_loss)
        
        scheduler.step()
        
        if test_loss < best_test_loss:
            best_test_loss = test_loss
            torch.save(model.state_dict(), 
                      os.path.join(args.save_dir, 'best_model.pth'))
        
        if (epoch + 1) % 10 == 0:
            print(f"Epoch [{epoch+1}/{args.epochs}] - "
                  f"Train Loss: {train_loss:.6f}, Test Loss: {test_loss:.6f}, "
                  f"LR: {optimizer.param_groups[0]['lr']:.6f}")
    
    print(f"\n训练完成! 最佳测试损失: {best_test_loss:.6f}")
    
    # 保存训练曲线
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Train Loss', linewidth=2)
    plt.plot(test_losses, label='Test Loss', linewidth=2)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Test Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(args.save_dir, 'loss_curve.png'), 
                dpi=150, bbox_inches='tight')
    print(f"损失曲线已保存到 {os.path.join(args.save_dir, 'loss_curve.png')}")
    plt.close()
    
    # 加载最佳模型并绘制结果
    model.load_state_dict(torch.load(os.path.join(args.save_dir, 'best_model.pth')))
    plot_results(model, test_dataset, device, x_grid, 
                os.path.join(args.save_dir, 'predictions.png'))
    
    # 计算测试集误差统计
    print("\n计算测试集误差...")
    relative_errors = []
    model.eval()
    
    with torch.no_grad():
        for i in range(len(test_dataset)):
            inputs, coords, targets = test_dataset[i]
            inputs = inputs.unsqueeze(0).to(device)
            coords = coords.unsqueeze(0).to(device)
            
            outputs = model(inputs, coords)
            prediction = outputs.cpu().numpy()[0, :, 0]
            true_output = targets.numpy()[:, 0]
            
            rel_error = np.linalg.norm(prediction - true_output) / np.linalg.norm(true_output)
            relative_errors.append(rel_error)
    
    print("\n=== 误差统计 ===")
    print(f"相对误差 - 平均: {np.mean(relative_errors):.6f}")
    print(f"相对误差 - 标准差: {np.std(relative_errors):.6f}")
    print(f"相对误差 - 最小: {np.min(relative_errors):.6f}")
    print(f"相对误差 - 最大: {np.max(relative_errors):.6f}")
    
    print(f"\n所有结果已保存到 {args.save_dir}/")


if __name__ == '__main__':
    main()
