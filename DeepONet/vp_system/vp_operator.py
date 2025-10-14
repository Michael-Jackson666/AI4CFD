"""
DeepONet 算子网络模块 - Vlasov-Poisson 系统
Deep Operator Network for Vlasov-Poisson System

DeepONet 架构:
- Branch Network: 处理初始条件 f(0,x,v)
- Trunk Network: 处理查询点 (t,x,v)
- 输出: f(t,x,v) = Σ b_k(f0) * t_k(t,x,v)
"""

import torch
import torch.nn as nn
import numpy as np


class BranchNetwork(nn.Module):
    """
    Branch Network: 编码初始条件
    
    输入: f(0,x,v) flatten 后的向量 [nx*nv]
    输出: 基函数系数 [p] (p 为基函数数量)
    """
    
    def __init__(self, input_dim, branch_dim, p):
        """
        Args:
            input_dim: 输入维度 (nx * nv)
            branch_dim: 隐藏层维度
            p: 输出维度 (基函数数量)
        """
        super(BranchNetwork, self).__init__()
        
        self.network = nn.Sequential(
            nn.Linear(input_dim, branch_dim),
            nn.Tanh(),
            nn.Linear(branch_dim, branch_dim),
            nn.Tanh(),
            nn.Linear(branch_dim, branch_dim),
            nn.Tanh(),
            nn.Linear(branch_dim, p)
        )
        
        # 初始化
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Xavier 初始化"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.zeros_(m.bias)
    
    def forward(self, f0):
        """
        前向传播
        
        Args:
            f0: 初始条件 [batch, nx*nv]
            
        Returns:
            branch_out: 基函数系数 [batch, p]
        """
        return self.network(f0)


class TrunkNetwork(nn.Module):
    """
    Trunk Network: 编码查询点坐标
    
    输入: (t, x, v) 归一化后的坐标
    输出: 基函数 [p]
    """
    
    def __init__(self, input_dim, trunk_dim, p):
        """
        Args:
            input_dim: 输入维度 (3: t, x, v)
            trunk_dim: 隐藏层维度
            p: 输出维度 (基函数数量)
        """
        super(TrunkNetwork, self).__init__()
        
        self.network = nn.Sequential(
            nn.Linear(input_dim, trunk_dim),
            nn.Tanh(),
            nn.Linear(trunk_dim, trunk_dim),
            nn.Tanh(),
            nn.Linear(trunk_dim, trunk_dim),
            nn.Tanh(),
            nn.Linear(trunk_dim, p)
        )
        
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Xavier 初始化"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.zeros_(m.bias)
    
    def forward(self, coords):
        """
        前向传播
        
        Args:
            coords: 查询点坐标 [batch, 3] (t, x, v)
            
        Returns:
            trunk_out: 基函数 [batch, p]
        """
        return self.network(coords)


class DeepONet(nn.Module):
    """
    DeepONet: 算子学习网络
    
    学习映射: G: f(0,x,v) → f(t,x,v)
    
    架构:
        f(t,x,v) ≈ Σ_k b_k(f0) * t_k(t,x,v) + bias
    """
    
    def __init__(self, config):
        """
        初始化 DeepONet
        
        Args:
            config: 配置字典
        """
        super(DeepONet, self).__init__()
        
        self.config = config
        
        # 网络参数
        self.nx = config['nx']
        self.nv = config['nv']
        self.input_dim = self.nx * self.nv  # Branch 输入维度
        self.branch_dim = config.get('branch_dim', 128)
        self.trunk_dim = config.get('trunk_dim', 128)
        self.p = config.get('p', 100)  # 基函数数量
        
        # 归一化参数
        self.t_max = config['t_max']
        self.x_max = config['x_max']
        self.v_max = config['v_max']
        
        # Branch Network
        self.branch_net = BranchNetwork(
            input_dim=self.input_dim,
            branch_dim=self.branch_dim,
            p=self.p
        )
        
        # Trunk Network
        self.trunk_net = TrunkNetwork(
            input_dim=3,  # (t, x, v)
            trunk_dim=self.trunk_dim,
            p=self.p
        )
        
        # Bias 项
        self.bias = nn.Parameter(torch.zeros(1))
        
        print(f"DeepONet 初始化:")
        print(f"  Branch 网络: {self.input_dim} → {self.branch_dim} → {self.p}")
        print(f"  Trunk 网络: 3 → {self.trunk_dim} → {self.p}")
        print(f"  总参数量: {self.count_parameters():,}")
    
    def count_parameters(self):
        """统计参数量"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def normalize_coords(self, t, x, v):
        """
        归一化坐标到 [-1, 1]
        
        Args:
            t: 时间 [batch] or scalar
            x: 空间 [batch] or scalar
            v: 速度 [batch] or scalar
            
        Returns:
            t_norm, x_norm, v_norm: 归一化坐标
        """
        t_norm = 2 * t / self.t_max - 1
        x_norm = 2 * x / self.x_max - 1
        v_norm = v / self.v_max  # v 已经是 [-v_max, v_max]
        
        return t_norm, x_norm, v_norm
    
    def forward(self, f0, t, x, v):
        """
        前向传播
        
        Args:
            f0: 初始条件 [batch_size, nx, nv]
            t: 时间查询点 [n_points]
            x: 空间查询点 [n_points]
            v: 速度查询点 [n_points]
            
        Returns:
            f_pred: 预测的分布函数 [batch_size, n_points]
        """
        batch_size = f0.shape[0]
        n_points = t.shape[0]
        
        # Flatten 初始条件
        f0_flat = f0.reshape(batch_size, -1)  # [batch_size, nx*nv]
        
        # Branch Network: [batch_size, p]
        branch_out = self.branch_net(f0_flat)
        
        # 归一化坐标
        t_norm, x_norm, v_norm = self.normalize_coords(t, x, v)
        coords = torch.stack([t_norm, x_norm, v_norm], dim=1)  # [n_points, 3]
        
        # Trunk Network: [n_points, p]
        trunk_out = self.trunk_net(coords)
        
        # 内积: [batch_size, n_points]
        # branch_out: [batch_size, p]
        # trunk_out: [n_points, p]
        f_pred = torch.matmul(branch_out, trunk_out.T) + self.bias
        
        # 使用 softplus 确保非负性
        f_pred = torch.nn.functional.softplus(f_pred)
        
        return f_pred
    
    def predict(self, f0, t, x, v):
        """
        预测函数（推理模式）
        
        Args:
            f0: 初始条件 [batch_size, nx, nv] or [nx, nv]
            t: 时间查询点 [n_points] or scalar
            x: 空间查询点 [n_points] or scalar
            v: 速度查询点 [n_points] or scalar
            
        Returns:
            f_pred: 预测值 [batch_size, n_points] or [n_points]
        """
        self.eval()
        with torch.no_grad():
            # 处理单个样本
            if f0.dim() == 2:
                f0 = f0.unsqueeze(0)
                single_sample = True
            else:
                single_sample = False
            
            # 处理标量查询点
            if not isinstance(t, torch.Tensor):
                t = torch.tensor([t], dtype=torch.float32, device=f0.device)
            if not isinstance(x, torch.Tensor):
                x = torch.tensor([x], dtype=torch.float32, device=f0.device)
            if not isinstance(v, torch.Tensor):
                v = torch.tensor([v], dtype=torch.float32, device=f0.device)
            
            # 确保在同一设备
            t = t.to(f0.device)
            x = x.to(f0.device)
            v = v.to(f0.device)
            
            # 前向传播
            f_pred = self.forward(f0, t, x, v)
            
            if single_sample:
                f_pred = f_pred.squeeze(0)
            
            return f_pred


class VlasovPoissonOperator(nn.Module):
    """
    Vlasov-Poisson 算子学习器
    
    扩展 DeepONet 以处理：
    1. 批量预测完整网格
    2. 电场计算
    3. 物理约束
    """
    
    def __init__(self, config):
        """
        初始化算子学习器
        
        Args:
            config: 配置字典
        """
        super(VlasovPoissonOperator, self).__init__()
        
        self.config = config
        self.deeponet = DeepONet(config)
        
        # 网格信息
        self.nx = config['nx']
        self.nv = config['nv']
        self.x_grid = torch.linspace(0, config['x_max'], self.nx)
        self.v_grid = torch.linspace(-config['v_max'], config['v_max'], self.nv)
    
    def forward(self, f0, t):
        """
        预测给定时刻完整网格上的分布函数
        
        Args:
            f0: 初始条件 [batch_size, nx, nv]
            t: 时间标量或张量 [1]
            
        Returns:
            f_pred: 预测分布 [batch_size, nx, nv]
        """
        batch_size = f0.shape[0]
        device = f0.device
        
        # 创建查询点网格
        X, V = torch.meshgrid(self.x_grid, self.v_grid, indexing='ij')
        x_query = X.flatten().to(device)  # [nx*nv]
        v_query = V.flatten().to(device)  # [nx*nv]
        
        # 时间
        if not isinstance(t, torch.Tensor):
            t = torch.tensor([t], dtype=torch.float32, device=device)
        t_query = t.expand(x_query.shape[0]).to(device)  # [nx*nv]
        
        # 预测
        f_pred_flat = self.deeponet(f0, t_query, x_query, v_query)  # [batch_size, nx*nv]
        
        # Reshape 到网格
        f_pred = f_pred_flat.reshape(batch_size, self.nx, self.nv)
        
        return f_pred
    
    def compute_electric_field(self, f):
        """
        从分布函数计算电场
        
        Args:
            f: 分布函数 [batch_size, nx, nv] or [nx, nv]
            
        Returns:
            E: 电场 [batch_size, nx] or [nx]
        """
        if f.dim() == 2:
            f = f.unsqueeze(0)
            single_sample = True
        else:
            single_sample = False
        
        batch_size = f.shape[0]
        device = f.device
        
        # 计算电子密度: n_e = ∫ f dv
        v_grid = self.v_grid.to(device)
        dv = v_grid[1] - v_grid[0]
        n_e = torch.trapz(f, v_grid, dim=2)  # [batch_size, nx]
        
        # 电荷密度: ρ = n_e - 1
        rho = n_e - 1.0
        
        # 积分求电场: E = ∫ ρ dx
        x_grid = self.x_grid.to(device)
        dx = x_grid[1] - x_grid[0]
        
        E = torch.zeros_like(rho)
        E[:, 0] = 0  # 边界条件
        for i in range(1, self.nx):
            E[:, i] = E[:, i-1] + rho[:, i-1] * dx
        
        # 去除平均值（周期边界）
        E = E - E.mean(dim=1, keepdim=True)
        
        if single_sample:
            E = E.squeeze(0)
        
        return E
    
    def count_parameters(self):
        """统计参数量"""
        return self.deeponet.count_parameters()


def test_networks():
    """
    测试网络架构
    """
    print("="*70)
    print("测试 DeepONet 架构")
    print("="*70)
    
    # 配置
    config = {
        'nx': 64,
        'nv': 64,
        't_max': 50.0,
        'x_max': 10.0,
        'v_max': 5.0,
        'branch_dim': 128,
        'trunk_dim': 128,
        'p': 100
    }
    
    # 创建网络
    model = VlasovPoissonOperator(config)
    
    # 测试数据
    batch_size = 4
    n_points = 100
    
    f0 = torch.randn(batch_size, config['nx'], config['nv'])
    t = torch.linspace(0, config['t_max'], n_points)
    x = torch.linspace(0, config['x_max'], n_points)
    v = torch.linspace(-config['v_max'], config['v_max'], n_points)
    
    print("\n输入形状:")
    print(f"  f0: {f0.shape}")
    print(f"  查询点数: {n_points}")
    
    # 测试 DeepONet 前向传播
    print("\n测试 DeepONet 前向传播...")
    f_pred = model.deeponet(f0, t, x, v)
    print(f"  输出形状: {f_pred.shape}")
    print(f"  输出范围: [{f_pred.min():.4f}, {f_pred.max():.4f}]")
    
    # 测试完整网格预测
    print("\n测试完整网格预测...")
    f_grid = model(f0, t[50])
    print(f"  输出形状: {f_grid.shape}")
    print(f"  输出范围: [{f_grid.min():.4f}, {f_grid.max():.4f}]")
    
    # 测试电场计算
    print("\n测试电场计算...")
    E = model.compute_electric_field(f_grid)
    print(f"  电场形状: {E.shape}")
    print(f"  电场范围: [{E.min():.4f}, {E.max():.4f}]")
    
    print("\n" + "="*70)
    print("所有测试通过！")
    print("="*70)


if __name__ == '__main__':
    test_networks()
