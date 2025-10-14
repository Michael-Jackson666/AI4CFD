"""
Transformer-based DeepONet 变体
Transformer-Enhanced Deep Operator Networks

将 Transformer 架构集成到 DeepONet 中，
以提高对长程依赖和复杂模式的建模能力。
"""

import torch
import torch.nn as nn
import numpy as np
import math


class PositionalEncoding(nn.Module):
    """
    位置编码模块
    
    为序列数据添加位置信息
    """
    
    def __init__(self, d_model, max_len=5000):
        """
        Args:
            d_model: 嵌入维度
            max_len: 最大序列长度
        """
        super(PositionalEncoding, self).__init__()
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        pe = pe.unsqueeze(0)  # [1, max_len, d_model]
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        """
        Args:
            x: [batch, seq_len, d_model]
            
        Returns:
            x with positional encoding added
        """
        return x + self.pe[:, :x.size(1), :]


class TransformerBranchNetwork(nn.Module):
    """
    Transformer-based Branch Network
    
    使用 Transformer 编码初始条件空间信息
    """
    
    def __init__(self, input_dim, d_model, nhead, num_layers, p):
        """
        Args:
            input_dim: 输入维度 (nx * nv)
            d_model: Transformer 嵌入维度
            nhead: 注意力头数
            num_layers: Transformer 层数
            p: 输出维度 (基函数数量)
        """
        super(TransformerBranchNetwork, self).__init__()
        
        self.d_model = d_model
        self.input_dim = input_dim
        
        # 输入投影
        self.input_proj = nn.Linear(1, d_model)  # 每个点单独编码
        
        # 位置编码
        self.pos_encoder = PositionalEncoding(d_model, max_len=input_dim)
        
        # Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=4*d_model,
            dropout=0.1,
            activation='gelu',
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # 输出投影
        self.output_proj = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, p)
        )
        
        self._initialize_weights()
    
    def _initialize_weights(self):
        """初始化权重"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def forward(self, f0):
        """
        前向传播
        
        Args:
            f0: 初始条件 [batch, nx*nv]
            
        Returns:
            branch_out: 基函数系数 [batch, p]
        """
        batch_size = f0.shape[0]
        
        # Reshape to sequence: [batch, seq_len, 1]
        x = f0.unsqueeze(-1)  # [batch, nx*nv, 1]
        
        # 输入投影
        x = self.input_proj(x)  # [batch, nx*nv, d_model]
        
        # 位置编码
        x = self.pos_encoder(x)
        
        # Transformer 编码
        x = self.transformer(x)  # [batch, nx*nv, d_model]
        
        # 全局池化 (取平均)
        x = x.mean(dim=1)  # [batch, d_model]
        
        # 输出投影
        out = self.output_proj(x)  # [batch, p]
        
        return out


class TransformerTrunkNetwork(nn.Module):
    """
    Transformer-based Trunk Network
    
    使用 Transformer 处理查询点坐标
    """
    
    def __init__(self, input_dim, d_model, nhead, num_layers, p):
        """
        Args:
            input_dim: 输入维度 (3: t, x, v)
            d_model: Transformer 嵌入维度
            nhead: 注意力头数
            num_layers: Transformer 层数
            p: 输出维度 (基函数数量)
        """
        super(TransformerTrunkNetwork, self).__init__()
        
        self.d_model = d_model
        
        # 输入投影
        self.input_proj = nn.Linear(input_dim, d_model)
        
        # 位置编码
        self.pos_encoder = PositionalEncoding(d_model, max_len=10000)
        
        # Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=4*d_model,
            dropout=0.1,
            activation='gelu',
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # 输出投影
        self.output_proj = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, p)
        )
        
        self._initialize_weights()
    
    def _initialize_weights(self):
        """初始化权重"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def forward(self, coords):
        """
        前向传播
        
        Args:
            coords: 查询点坐标 [n_points, 3] (t, x, v)
            
        Returns:
            trunk_out: 基函数 [n_points, p]
        """
        # 输入投影
        x = self.input_proj(coords)  # [n_points, d_model]
        
        # 添加 batch 维度用于 Transformer
        x = x.unsqueeze(0)  # [1, n_points, d_model]
        
        # 位置编码
        x = self.pos_encoder(x)
        
        # Transformer 编码
        x = self.transformer(x)  # [1, n_points, d_model]
        
        # 移除 batch 维度
        x = x.squeeze(0)  # [n_points, d_model]
        
        # 输出投影
        out = self.output_proj(x)  # [n_points, p]
        
        return out


class TransformerDeepONet(nn.Module):
    """
    Transformer-Enhanced DeepONet
    
    使用 Transformer 替代标准 MLP 的 DeepONet
    """
    
    def __init__(self, config):
        """
        初始化 Transformer DeepONet
        
        Args:
            config: 配置字典
        """
        super(TransformerDeepONet, self).__init__()
        
        self.config = config
        
        # 网络参数
        self.nx = config['nx']
        self.nv = config['nv']
        self.input_dim = self.nx * self.nv
        
        # Transformer 参数
        self.d_model = config.get('d_model', 128)
        self.nhead = config.get('nhead', 8)
        self.num_layers = config.get('num_layers', 4)
        self.p = config.get('p', 100)
        
        # 归一化参数
        self.t_max = config['t_max']
        self.x_max = config['x_max']
        self.v_max = config['v_max']
        
        # Transformer Branch Network
        self.branch_net = TransformerBranchNetwork(
            input_dim=self.input_dim,
            d_model=self.d_model,
            nhead=self.nhead,
            num_layers=self.num_layers,
            p=self.p
        )
        
        # Transformer Trunk Network
        self.trunk_net = TransformerTrunkNetwork(
            input_dim=3,  # (t, x, v)
            d_model=self.d_model,
            nhead=self.nhead,
            num_layers=self.num_layers,
            p=self.p
        )
        
        # Bias 项
        self.bias = nn.Parameter(torch.zeros(1))
        
        print(f"Transformer DeepONet 初始化:")
        print(f"  d_model: {self.d_model}, nhead: {self.nhead}, layers: {self.num_layers}")
        print(f"  Branch 网络: {self.input_dim} → Transformer → {self.p}")
        print(f"  Trunk 网络: 3 → Transformer → {self.p}")
        print(f"  总参数量: {self.count_parameters():,}")
    
    def count_parameters(self):
        """统计参数量"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def normalize_coords(self, t, x, v):
        """
        归一化坐标到 [-1, 1]
        
        Args:
            t: 时间
            x: 空间
            v: 速度
            
        Returns:
            t_norm, x_norm, v_norm: 归一化坐标
        """
        t_norm = 2 * t / self.t_max - 1
        x_norm = 2 * x / self.x_max - 1
        v_norm = v / self.v_max
        
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
        f_pred = torch.matmul(branch_out, trunk_out.T) + self.bias
        
        # 使用 softplus 确保非负性
        f_pred = torch.nn.functional.softplus(f_pred)
        
        return f_pred


class HybridDeepONet(nn.Module):
    """
    Hybrid DeepONet: MLP Branch + Transformer Trunk
    
    结合 MLP 的效率和 Transformer 的表达能力
    """
    
    def __init__(self, config):
        """
        初始化 Hybrid DeepONet
        
        Args:
            config: 配置字典
        """
        super(HybridDeepONet, self).__init__()
        
        self.config = config
        
        # 网络参数
        self.nx = config['nx']
        self.nv = config['nv']
        self.input_dim = self.nx * self.nv
        self.branch_dim = config.get('branch_dim', 128)
        
        # Transformer 参数
        self.d_model = config.get('d_model', 128)
        self.nhead = config.get('nhead', 8)
        self.num_layers = config.get('num_layers', 4)
        self.p = config.get('p', 100)
        
        # 归一化参数
        self.t_max = config['t_max']
        self.x_max = config['x_max']
        self.v_max = config['v_max']
        
        # MLP Branch Network
        self.branch_net = nn.Sequential(
            nn.Linear(self.input_dim, self.branch_dim),
            nn.Tanh(),
            nn.Linear(self.branch_dim, self.branch_dim),
            nn.Tanh(),
            nn.Linear(self.branch_dim, self.branch_dim),
            nn.Tanh(),
            nn.Linear(self.branch_dim, self.p)
        )
        
        # Transformer Trunk Network
        self.trunk_net = TransformerTrunkNetwork(
            input_dim=3,
            d_model=self.d_model,
            nhead=self.nhead,
            num_layers=self.num_layers,
            p=self.p
        )
        
        # Bias 项
        self.bias = nn.Parameter(torch.zeros(1))
        
        self._initialize_weights()
        
        print(f"Hybrid DeepONet 初始化:")
        print(f"  Branch: MLP ({self.input_dim} → {self.branch_dim} → {self.p})")
        print(f"  Trunk: Transformer (d_model={self.d_model}, nhead={self.nhead}, layers={self.num_layers})")
        print(f"  总参数量: {self.count_parameters():,}")
    
    def _initialize_weights(self):
        """初始化 MLP 权重"""
        for m in self.branch_net.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.zeros_(m.bias)
    
    def count_parameters(self):
        """统计参数量"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def normalize_coords(self, t, x, v):
        """归一化坐标"""
        t_norm = 2 * t / self.t_max - 1
        x_norm = 2 * x / self.x_max - 1
        v_norm = v / self.v_max
        return t_norm, x_norm, v_norm
    
    def forward(self, f0, t, x, v):
        """
        前向传播
        
        Args:
            f0: 初始条件 [batch_size, nx, nv]
            t, x, v: 查询点坐标 [n_points]
            
        Returns:
            f_pred: 预测值 [batch_size, n_points]
        """
        batch_size = f0.shape[0]
        
        # Flatten 初始条件
        f0_flat = f0.reshape(batch_size, -1)
        
        # Branch Network (MLP)
        branch_out = self.branch_net(f0_flat)  # [batch_size, p]
        
        # 归一化坐标
        t_norm, x_norm, v_norm = self.normalize_coords(t, x, v)
        coords = torch.stack([t_norm, x_norm, v_norm], dim=1)
        
        # Trunk Network (Transformer)
        trunk_out = self.trunk_net(coords)  # [n_points, p]
        
        # 内积
        f_pred = torch.matmul(branch_out, trunk_out.T) + self.bias
        f_pred = torch.nn.functional.softplus(f_pred)
        
        return f_pred


def test_transformer_networks():
    """
    测试 Transformer 网络架构
    """
    print("="*70)
    print("测试 Transformer DeepONet 架构")
    print("="*70)
    
    # 配置
    config = {
        'nx': 64,
        'nv': 64,
        't_max': 50.0,
        'x_max': 10.0,
        'v_max': 5.0,
        'd_model': 128,
        'nhead': 8,
        'num_layers': 4,
        'p': 100
    }
    
    # 测试数据
    batch_size = 4
    n_points = 100
    
    f0 = torch.randn(batch_size, config['nx'], config['nv'])
    t = torch.linspace(0, config['t_max'], n_points)
    x = torch.linspace(0, config['x_max'], n_points)
    v = torch.linspace(-config['v_max'], config['v_max'], n_points)
    
    # 测试 Transformer DeepONet
    print("\n测试 Transformer DeepONet...")
    model1 = TransformerDeepONet(config)
    f_pred1 = model1(f0, t, x, v)
    print(f"  输出形状: {f_pred1.shape}")
    print(f"  输出范围: [{f_pred1.min():.4f}, {f_pred1.max():.4f}]")
    
    # 测试 Hybrid DeepONet
    print("\n测试 Hybrid DeepONet...")
    config['branch_dim'] = 128
    model2 = HybridDeepONet(config)
    f_pred2 = model2(f0, t, x, v)
    print(f"  输出形状: {f_pred2.shape}")
    print(f"  输出范围: [{f_pred2.min():.4f}, {f_pred2.max():.4f}]")
    
    print("\n" + "="*70)
    print("所有测试通过！")
    print("="*70)


if __name__ == '__main__':
    test_transformer_networks()
