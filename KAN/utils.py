"""
KAN 求解 PDE 的工具函数

包含数据生成、可视化、误差计算等辅助功能
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import warnings
warnings.filterwarnings('ignore')


# ==============================================================================
# 数据生成工具
# ==============================================================================

def generate_collocation_points(domain, n_points, method='uniform', device='cpu'):
    """
    生成配点（collocation points）
    
    参数:
        domain: 域范围，例如 [(xmin, xmax), (tmin, tmax)]
        n_points: 每个维度的点数
        method: 'uniform' 或 'random' 或 'lhs' (拉丁超立方采样)
        device: 'cpu' 或 'cuda'
    
    返回:
        points: [n_total, n_dim] 张量
    """
    ndim = len(domain)
    
    if method == 'uniform':
        # 均匀网格
        grids = [torch.linspace(d[0], d[1], n_points, device=device) for d in domain]
        meshgrid = torch.meshgrid(*grids, indexing='ij')
        points = torch.stack([m.flatten() for m in meshgrid], dim=1)
        
    elif method == 'random':
        # 随机采样
        n_total = n_points ** ndim
        points = torch.zeros(n_total, ndim, device=device)
        for i, (dmin, dmax) in enumerate(domain):
            points[:, i] = torch.rand(n_total, device=device) * (dmax - dmin) + dmin
            
    elif method == 'lhs':
        # 拉丁超立方采样 (Latin Hypercube Sampling)
        try:
            from scipy.stats import qmc
            sampler = qmc.LatinHypercube(d=ndim)
            sample = sampler.random(n=n_points ** ndim)
            
            # 缩放到域范围
            points = torch.zeros(n_points ** ndim, ndim, device=device)
            for i, (dmin, dmax) in enumerate(domain):
                points[:, i] = torch.tensor(
                    sample[:, i] * (dmax - dmin) + dmin,
                    device=device, dtype=torch.float32
                )
        except ImportError:
            print("⚠️  scipy 未安装，使用随机采样代替")
            return generate_collocation_points(domain, n_points, 'random', device)
    
    else:
        raise ValueError(f"未知的采样方法: {method}")
    
    return points


def generate_boundary_points(domain, n_points, device='cpu'):
    """
    生成边界点
    
    参数:
        domain: 域范围 [(xmin, xmax), ...]
        n_points: 每个边界的点数
        device: 设备
    
    返回:
        boundary_points: 边界点列表
    """
    ndim = len(domain)
    boundary_points = []
    
    for dim in range(ndim):
        for boundary_value in [domain[dim][0], domain[dim][1]]:
            # 在其他维度上采样
            other_dims = [d for i, d in enumerate(domain) if i != dim]
            
            if len(other_dims) > 0:
                other_points = generate_collocation_points(
                    other_dims, n_points, 'uniform', device
                )
                
                # 插入固定的边界值
                full_points = torch.zeros(
                    other_points.shape[0], ndim, device=device
                )
                
                j = 0
                for i in range(ndim):
                    if i == dim:
                        full_points[:, i] = boundary_value
                    else:
                        full_points[:, i] = other_points[:, j]
                        j += 1
                
                boundary_points.append(full_points)
    
    return boundary_points


# ==============================================================================
# 误差计算工具
# ==============================================================================

def compute_error_metrics(pred, exact):
    """
    计算误差指标
    
    参数:
        pred: 预测值 (numpy array)
        exact: 精确值 (numpy array)
    
    返回:
        metrics: 误差字典
    """
    error = pred - exact
    
    # 绝对误差
    abs_error = np.abs(error)
    max_error = np.max(abs_error)
    mean_error = np.mean(abs_error)
    
    # 相对误差
    rel_error = abs_error / (np.abs(exact) + 1e-10)
    max_rel_error = np.max(rel_error)
    mean_rel_error = np.mean(rel_error)
    
    # L2 范数
    l2_error = np.linalg.norm(error) / np.linalg.norm(exact)
    
    # L∞ 范数
    linf_error = max_error / np.max(np.abs(exact))
    
    metrics = {
        'max_abs_error': max_error,
        'mean_abs_error': mean_error,
        'max_rel_error': max_rel_error,
        'mean_rel_error': mean_rel_error,
        'l2_error': l2_error,
        'linf_error': linf_error
    }
    
    return metrics


def print_error_metrics(metrics):
    """打印误差指标"""
    print("\n" + "=" * 60)
    print("误差分析:")
    print("=" * 60)
    print(f"最大绝对误差:    {metrics['max_abs_error']:.4e}")
    print(f"平均绝对误差:    {metrics['mean_abs_error']:.4e}")
    print(f"相对 L2 误差:    {metrics['l2_error']:.4e}")
    print(f"相对 L∞ 误差:    {metrics['linf_error']:.4e}")
    print("=" * 60)


# ==============================================================================
# 可视化工具
# ==============================================================================

def plot_1d_solution(x, u_pred, u_exact=None, title='Solution', 
                     xlabel='x', ylabel='u', save_path=None):
    """
    绘制 1D 问题的解
    
    参数:
        x: 空间坐标 (numpy array)
        u_pred: KAN 预测解
        u_exact: 解析解 (可选)
        title: 图标题
        xlabel, ylabel: 坐标轴标签
        save_path: 保存路径 (可选)
    """
    plt.figure(figsize=(10, 6))
    
    if u_exact is not None:
        plt.plot(x, u_exact, 'b-', linewidth=2, label='解析解')
        plt.plot(x, u_pred, 'r--', linewidth=2, label='KAN 预测')
        
        # 误差图
        error = np.abs(u_pred - u_exact)
        plt.figure(figsize=(10, 4))
        plt.plot(x, error, 'g-', linewidth=2)
        plt.xlabel(xlabel)
        plt.ylabel('|误差|')
        plt.title('绝对误差')
        plt.grid(True, alpha=0.3)
        plt.yscale('log')
        
        if save_path:
            plt.savefig(save_path.replace('.png', '_error.png'), 
                       dpi=150, bbox_inches='tight')
    else:
        plt.plot(x, u_pred, 'r-', linewidth=2, label='KAN 预测')
    
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    plt.show()


def plot_2d_solution(x, t, u, title='Solution', cmap='RdBu_r', 
                     save_path=None):
    """
    绘制 2D 时空解
    
    参数:
        x: 空间坐标 [nx]
        t: 时间坐标 [nt]
        u: 解 [nt, nx]
        title: 标题
        cmap: 颜色映射
        save_path: 保存路径
    """
    plt.figure(figsize=(12, 5))
    
    # 时空图
    plt.subplot(1, 2, 1)
    im = plt.imshow(u.T, aspect='auto', origin='lower',
                    extent=[t[0], t[-1], x[0], x[-1]], cmap=cmap)
    plt.colorbar(im, label='u(x,t)')
    plt.xlabel('时间 t')
    plt.ylabel('空间 x')
    plt.title(title)
    
    # 不同时刻的切片
    plt.subplot(1, 2, 2)
    n_snapshots = 5
    indices = np.linspace(0, len(t) - 1, n_snapshots, dtype=int)
    
    for idx in indices:
        plt.plot(x, u[idx, :], label=f't={t[idx]:.2f}', alpha=0.8)
    
    plt.xlabel('空间 x')
    plt.ylabel('u')
    plt.title('不同时刻的解')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    plt.show()


def plot_training_history(history, log_scale=True, save_path=None):
    """
    绘制训练历史
    
    参数:
        history: 训练历史字典
        log_scale: 是否使用对数坐标
        save_path: 保存路径
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    epochs = range(1, len(history['total']) + 1)
    
    # 总损失
    axes[0].plot(epochs, history['total'], 'b-', linewidth=2, alpha=0.8)
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Total Loss')
    axes[0].set_title('训练总损失')
    axes[0].grid(True, alpha=0.3)
    if log_scale:
        axes[0].set_yscale('log')
    
    # 损失分解
    if 'pde' in history:
        axes[1].plot(epochs, history['pde'], 'r-', label='PDE 损失', alpha=0.7)
    if 'bc' in history:
        axes[1].plot(epochs, history['bc'], 'g-', label='BC 损失', alpha=0.7)
    if 'ic' in history:
        axes[1].plot(epochs, history['ic'], 'orange', label='IC 损失', alpha=0.7)
    if 'reg' in history:
        axes[1].plot(epochs, history['reg'], 'purple', label='正则化', alpha=0.7)
    
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Loss')
    axes[1].set_title('损失分解')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    if log_scale:
        axes[1].set_yscale('log')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    plt.show()


# ==============================================================================
# 模型工具
# ==============================================================================

def save_model(model, filepath):
    """
    保存模型
    
    参数:
        model: KAN 模型
        filepath: 保存路径
    """
    torch.save({
        'model_state_dict': model.state_dict(),
        'model_config': {
            'layers': model.layers,
        }
    }, filepath)
    print(f"✅ 模型已保存到: {filepath}")


def load_model(filepath, model_class, device='cpu'):
    """
    加载模型
    
    参数:
        filepath: 模型文件路径
        model_class: 模型类
        device: 设备
    
    返回:
        model: 加载的模型
    """
    checkpoint = torch.load(filepath, map_location=device)
    
    # 重建模型
    config = checkpoint['model_config']
    model = model_class(**config).to(device)
    
    # 加载参数
    model.load_state_dict(checkpoint['model_state_dict'])
    
    print(f"✅ 模型已从 {filepath} 加载")
    return model


# ==============================================================================
# 动画工具
# ==============================================================================

def create_animation(x, t, u, title='Time Evolution', interval=50, 
                     save_path=None):
    """
    创建时间演化动画
    
    参数:
        x: 空间坐标
        t: 时间坐标
        u: 解 [nt, nx]
        title: 标题
        interval: 帧间隔 (毫秒)
        save_path: 保存路径 (.gif 或 .mp4)
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    line, = ax.plot([], [], 'b-', linewidth=2)
    ax.set_xlim(x[0], x[-1])
    ax.set_ylim(u.min() * 1.1, u.max() * 1.1)
    ax.set_xlabel('x')
    ax.set_ylabel('u')
    ax.grid(True, alpha=0.3)
    
    time_text = ax.text(0.02, 0.95, '', transform=ax.transAxes)
    
    def init():
        line.set_data([], [])
        time_text.set_text('')
        return line, time_text
    
    def animate(i):
        line.set_data(x, u[i, :])
        time_text.set_text(f't = {t[i]:.3f}')
        ax.set_title(f'{title} (frame {i+1}/{len(t)})')
        return line, time_text
    
    anim = FuncAnimation(fig, animate, init_func=init,
                        frames=len(t), interval=interval,
                        blit=True, repeat=True)
    
    if save_path:
        if save_path.endswith('.gif'):
            anim.save(save_path, writer='pillow', fps=30)
        elif save_path.endswith('.mp4'):
            anim.save(save_path, writer='ffmpeg', fps=30)
        print(f"💾 动画已保存到: {save_path}")
    
    plt.show()


# ==============================================================================
# 调试工具
# ==============================================================================

def check_gradients(model, loss):
    """
    检查梯度
    
    参数:
        model: 模型
        loss: 损失
    """
    loss.backward(retain_graph=True)
    
    total_norm = 0
    for name, p in model.named_parameters():
        if p.grad is not None:
            param_norm = p.grad.data.norm(2)
            total_norm += param_norm.item() ** 2
            print(f"{name:30s} | Grad norm: {param_norm.item():.4e}")
    
    total_norm = total_norm ** 0.5
    print(f"\n总梯度范数: {total_norm:.4e}")


def visualize_kan_function(model, layer_idx=0, input_dim=0, output_dim=0, 
                           x_range=(-1, 1), n_points=100):
    """
    可视化 KAN 层学到的一元函数
    
    参数:
        model: KAN 模型
        layer_idx: 层索引
        input_dim: 输入维度索引
        output_dim: 输出维度索引
        x_range: 输入范围
        n_points: 采样点数
    """
    layer = model.kan_layers[layer_idx]
    
    x = torch.linspace(x_range[0], x_range[1], n_points).unsqueeze(1)
    
    # 计算 B-spline 基函数
    basis = layer.bsplines[input_dim].compute_basis_matrix(x)
    
    # 应用系数
    coeffs = layer.coeffs[input_dim, output_dim, :].detach().cpu()
    y = torch.matmul(basis.cpu(), coeffs).numpy()
    
    x_np = x.squeeze().numpy()
    
    plt.figure(figsize=(10, 6))
    plt.plot(x_np, y, 'b-', linewidth=2)
    plt.xlabel('输入')
    plt.ylabel('输出')
    plt.title(f'Layer {layer_idx}: 输入维度 {input_dim} → 输出维度 {output_dim}')
    plt.grid(True, alpha=0.3)
    plt.show()


if __name__ == "__main__":
    # 测试工具函数
    print("=" * 60)
    print("KAN 工具函数测试")
    print("=" * 60)
    
    # 测试配点生成
    print("\n测试配点生成:")
    domain = [(-1, 1), (0, 1)]
    points = generate_collocation_points(domain, 10, method='uniform')
    print(f"  生成配点形状: {points.shape}")
    
    # 测试误差计算
    print("\n测试误差计算:")
    pred = np.random.randn(100)
    exact = pred + 0.01 * np.random.randn(100)
    metrics = compute_error_metrics(pred, exact)
    print_error_metrics(metrics)
    
    print("\n✅ 所有测试通过!")
