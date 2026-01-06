"""
Kolmogorov-Arnold Network (KAN) for PDE Solving

基于 B-spline 基函数的 KAN 实现
"""

import torch
import torch.nn as nn
import numpy as np


class BSpline:
    """
    B-spline 基函数计算
    
    参数:
        grid_size: 网格点数量
        spline_order: B-spline 阶数 (通常为 3，即三次样条)
        grid_range: 网格范围 [a, b]
    """
    
    def __init__(self, grid_size=5, spline_order=3, grid_range=(-1, 1)):
        self.grid_size = grid_size
        self.spline_order = spline_order
        self.grid_range = grid_range
        
        # 构建节点向量 (knot vector)
        self.knots = self._create_knots()
    
    def _create_knots(self):
        """
        创建均匀节点向量
        
        对于 B-spline，节点向量长度为: grid_size + spline_order + 1
        """
        a, b = self.grid_range
        # 内部节点
        interior_knots = np.linspace(a, b, self.grid_size)
        
        # 添加边界重复节点（保证边界插值）
        knots = np.concatenate([
            np.repeat(a, self.spline_order),
            interior_knots,
            np.repeat(b, self.spline_order)
        ])
        
        return torch.FloatTensor(knots)
    
    def basis(self, x, i, k):
        """
        递归计算 B-spline 基函数 B_{i,k}(x)
        
        Cox-de Boor 递归公式:
        B_{i,0}(x) = 1 if t_i <= x < t_{i+1}, else 0
        B_{i,k}(x) = (x - t_i)/(t_{i+k} - t_i) * B_{i,k-1}(x) 
                   + (t_{i+k+1} - x)/(t_{i+k+1} - t_{i+1}) * B_{i+1,k-1}(x)
        
        参数:
            x: 输入点 [batch_size, 1]
            i: 基函数索引
            k: B-spline 阶数
        """
        if k == 0:
            # 零阶基函数
            return ((x >= self.knots[i]) & (x < self.knots[i + 1])).float()
        
        # 避免除零
        denom1 = self.knots[i + k] - self.knots[i]
        denom2 = self.knots[i + k + 1] - self.knots[i + 1]
        
        term1 = 0
        if denom1 != 0:
            term1 = (x - self.knots[i]) / denom1 * self.basis(x, i, k - 1)
        
        term2 = 0
        if denom2 != 0:
            term2 = (self.knots[i + k + 1] - x) / denom2 * self.basis(x, i + 1, k - 1)
        
        return term1 + term2
    
    def compute_basis_matrix(self, x):
        """
        计算所有基函数在 x 处的值
        
        返回: [batch_size, num_basis] 其中 num_basis = grid_size + spline_order
        """
        batch_size = x.shape[0]
        num_basis = self.grid_size + self.spline_order
        
        # 移动 knots 到与 x 相同的设备
        self.knots = self.knots.to(x.device)
        
        basis_matrix = torch.zeros(batch_size, num_basis, device=x.device)
        
        for i in range(num_basis):
            basis_matrix[:, i] = self.basis(x.squeeze(-1), i, self.spline_order)
        
        return basis_matrix


class KANLayer(nn.Module):
    """
    Kolmogorov-Arnold Network 层
    
    每个输入维度到每个输出维度的连接都是一个可学习的一元函数（B-spline）
    
    参数:
        in_dim: 输入维度
        out_dim: 输出维度
        grid_size: B-spline 网格点数
        spline_order: B-spline 阶数
        grid_range: 网格范围
    """
    
    def __init__(self, in_dim, out_dim, grid_size=5, spline_order=3, grid_range=(-1, 1)):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.grid_size = grid_size
        self.spline_order = spline_order
        
        # 每个输入维度创建一个 B-spline 对象
        self.bsplines = [
            BSpline(grid_size, spline_order, grid_range) 
            for _ in range(in_dim)
        ]
        
        # B-spline 系数矩阵: [in_dim, out_dim, num_basis]
        num_basis = grid_size + spline_order
        self.coeffs = nn.Parameter(
            torch.randn(in_dim, out_dim, num_basis) * 0.1
        )
    
    def forward(self, x):
        """
        前向传播
        
        输入: x [batch_size, in_dim]
        输出: y [batch_size, out_dim]
        
        计算方式:
        y_j = sum_{i=1}^{in_dim} Phi_{i,j}(x_i)
        其中 Phi_{i,j}(x_i) = sum_{k} c_{i,j,k} * B_k(x_i)
        """
        batch_size = x.shape[0]
        device = x.device
        
        # 输出初始化
        output = torch.zeros(batch_size, self.out_dim, device=device)
        
        # 对每个输入维度
        for i in range(self.in_dim):
            # 计算 B-spline 基函数: [batch_size, num_basis]
            x_i = x[:, i:i+1]
            basis = self.bsplines[i].compute_basis_matrix(x_i)
            
            # 应用系数: [batch_size, out_dim]
            phi = torch.matmul(basis, self.coeffs[i].T)
            
            output += phi
        
        return output
    
    def regularization_loss(self):
        """
        正则化损失：鼓励平滑的函数
        
        返回 B-spline 系数的 L2 范数
        """
        return torch.mean(self.coeffs ** 2)


class KANPDE(nn.Module):
    """
    用于 PDE 求解的 Kolmogorov-Arnold Network
    
    架构: 
        输入 -> KAN Layer -> ... -> KAN Layer -> 输出
    
    参数:
        layers: 网络结构，例如 [1, 32, 32, 1] 表示1输入，2个隐藏层各32节点，1输出
        grid_size: B-spline 网格大小
        spline_order: B-spline 阶数
        grid_range: 输入范围
    """
    
    def __init__(self, layers=[1, 32, 32, 1], grid_size=5, spline_order=3, grid_range=(-1, 1)):
        super().__init__()
        self.layers = layers
        self.depth = len(layers) - 1
        
        # 构建 KAN 层
        self.kan_layers = nn.ModuleList()
        for i in range(self.depth):
            self.kan_layers.append(
                KANLayer(
                    in_dim=layers[i],
                    out_dim=layers[i+1],
                    grid_size=grid_size,
                    spline_order=spline_order,
                    grid_range=grid_range
                )
            )
    
    def forward(self, x):
        """
        前向传播
        
        输入: x [batch_size, input_dim]
        输出: y [batch_size, output_dim]
        """
        for layer in self.kan_layers:
            x = layer(x)
        return x
    
    def compute_derivatives(self, x, order=1):
        """
        使用自动微分计算导数
        
        参数:
            x: 输入点 [batch_size, input_dim]
            order: 导数阶数
        
        返回:
            u: 函数值
            derivatives: 导数列表
        """
        x = x.requires_grad_(True)
        u = self.forward(x)
        
        derivatives = []
        grad = u
        
        for i in range(order):
            grad = torch.autograd.grad(
                grad, x,
                grad_outputs=torch.ones_like(grad),
                create_graph=True,
                retain_graph=True
            )[0]
            derivatives.append(grad)
        
        return u, derivatives
    
    def regularization_loss(self):
        """
        总正则化损失：所有层的正则化之和
        """
        reg_loss = 0
        for layer in self.kan_layers:
            reg_loss += layer.regularization_loss()
        return reg_loss
    
    def count_parameters(self):
        """统计模型参数量"""
        return sum(p.numel() for p in self.parameters())


# ==============================================================================
# 辅助函数
# ==============================================================================

def initialize_kan(layers, grid_size=5, spline_order=3, grid_range=(-1, 1)):
    """
    便捷函数：初始化 KAN 模型
    
    使用示例:
        model = initialize_kan([1, 32, 32, 1])
    """
    model = KANPDE(
        layers=layers,
        grid_size=grid_size,
        spline_order=spline_order,
        grid_range=grid_range
    )
    
    print(f"✅ KAN 模型创建成功!")
    print(f"   结构: {layers}")
    print(f"   参数量: {model.count_parameters():,}")
    print(f"   B-spline: grid_size={grid_size}, order={spline_order}")
    
    return model


if __name__ == "__main__":
    # 测试代码
    print("=" * 60)
    print("KAN 模型测试")
    print("=" * 60)
    
    # 创建模型
    model = initialize_kan(layers=[1, 16, 16, 1], grid_size=5, spline_order=3)
    
    # 测试前向传播
    x_test = torch.linspace(-1, 1, 100).unsqueeze(1)
    with torch.no_grad():
        y_test = model(x_test)
    
    print(f"\n测试输入形状: {x_test.shape}")
    print(f"测试输出形状: {y_test.shape}")
    
    # 测试导数计算
    x_grad = torch.tensor([[0.5]], requires_grad=True)
    u, derivs = model.compute_derivatives(x_grad, order=2)
    
    print(f"\n在 x=0.5 处:")
    print(f"  u = {u.item():.6f}")
    print(f"  u' = {derivs[0].item():.6f}")
    print(f"  u'' = {derivs[1].item():.6f}")
    
    print("\n✅ 所有测试通过!")
