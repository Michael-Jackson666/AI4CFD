"""
Flow Map Learning Models

Implementation of neural networks for learning time integration operators.
Based on the work by Dongbin Xiu and collaborators.

Author: AI4CFD Project
Date: 2026-01
"""

import torch
import torch.nn as nn
import numpy as np
from typing import List, Optional, Tuple, Union


class FlowMapMLP(nn.Module):
    """
    Flow Map 多层感知机
    
    学习从当前状态到下一时刻状态的映射:
    x(t + Δt) = Φ(x(t), Δt)
    
    使用残差连接: x_next = x + NN(x, Δt)
    这种设计更容易学习小的状态变化
    """
    
    def __init__(self, 
                 state_dim: int,
                 hidden_dims: List[int] = [64, 64, 64],
                 activation: str = 'tanh',
                 use_residual: bool = True,
                 time_encoding: bool = True):
        """
        Args:
            state_dim: 状态空间维度
            hidden_dims: 隐藏层维度列表
            activation: 激活函数 ('tanh', 'relu', 'gelu', 'silu')
            use_residual: 是否使用残差连接
            time_encoding: 是否对时间步长进行编码
        """
        super(FlowMapMLP, self).__init__()
        
        self.state_dim = state_dim
        self.hidden_dims = hidden_dims
        self.use_residual = use_residual
        self.time_encoding = time_encoding
        
        # 激活函数
        activations = {
            'tanh': nn.Tanh(),
            'relu': nn.ReLU(),
            'gelu': nn.GELU(),
            'silu': nn.SiLU()
        }
        self.activation = activations.get(activation, nn.Tanh())
        
        # 时间编码维度
        if time_encoding:
            self.time_embed_dim = 16
            self.time_encoder = nn.Sequential(
                nn.Linear(1, self.time_embed_dim),
                nn.Tanh(),
                nn.Linear(self.time_embed_dim, self.time_embed_dim)
            )
            input_dim = state_dim + self.time_embed_dim
        else:
            input_dim = state_dim + 1  # 直接拼接时间
        
        # 构建网络
        layers = []
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(input_dim, hidden_dim))
            layers.append(self.activation)
            input_dim = hidden_dim
        
        layers.append(nn.Linear(input_dim, state_dim))
        self.net = nn.Sequential(*layers)
        
        # 初始化权重
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Xavier 初始化"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def forward(self, x: torch.Tensor, dt: Union[torch.Tensor, float]) -> torch.Tensor:
        """
        前向传播
        
        Args:
            x: 当前状态 [batch_size, state_dim]
            dt: 时间步长 [batch_size, 1] 或标量
        
        Returns:
            x_next: 下一时刻状态 [batch_size, state_dim]
        """
        batch_size = x.shape[0]
        
        # 处理时间输入
        if isinstance(dt, (float, int)):
            dt = torch.ones(batch_size, 1, device=x.device) * dt
        elif dt.dim() == 0:
            dt = dt.unsqueeze(0).unsqueeze(0).expand(batch_size, 1)
        elif dt.dim() == 1:
            dt = dt.unsqueeze(-1)
        
        # 时间编码
        if self.time_encoding:
            dt_encoded = self.time_encoder(dt)
            inputs = torch.cat([x, dt_encoded], dim=-1)
        else:
            inputs = torch.cat([x, dt], dim=-1)
        
        # 网络输出
        dx = self.net(inputs)
        
        # 残差连接
        if self.use_residual:
            return x + dx
        return dx
    
    def multi_step_predict(self, 
                          x0: torch.Tensor, 
                          dt: float, 
                          n_steps: int) -> torch.Tensor:
        """
        多步自回归预测
        
        Args:
            x0: 初始状态 [batch_size, state_dim] 或 [state_dim]
            dt: 时间步长
            n_steps: 预测步数
        
        Returns:
            trajectory: 完整轨迹 [n_steps + 1, batch_size, state_dim]
        """
        if x0.dim() == 1:
            x0 = x0.unsqueeze(0)
        
        trajectory = [x0]
        x = x0
        
        for _ in range(n_steps):
            x = self.forward(x, dt)
            trajectory.append(x)
        
        return torch.stack(trajectory, dim=0)


class FlowMapResNet(nn.Module):
    """
    Flow Map 残差网络
    
    使用残差块结构，适合更深的网络
    """
    
    def __init__(self,
                 state_dim: int,
                 hidden_dim: int = 64,
                 num_blocks: int = 3,
                 use_residual: bool = True):
        """
        Args:
            state_dim: 状态空间维度
            hidden_dim: 隐藏层维度
            num_blocks: 残差块数量
            use_residual: 是否使用外部残差连接
        """
        super(FlowMapResNet, self).__init__()
        
        self.state_dim = state_dim
        self.use_residual = use_residual
        
        # 输入投影
        self.input_proj = nn.Linear(state_dim + 1, hidden_dim)
        
        # 残差块
        self.blocks = nn.ModuleList([
            ResidualBlock(hidden_dim) for _ in range(num_blocks)
        ])
        
        # 输出投影
        self.output_proj = nn.Linear(hidden_dim, state_dim)
        
        self._initialize_weights()
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def forward(self, x: torch.Tensor, dt: Union[torch.Tensor, float]) -> torch.Tensor:
        batch_size = x.shape[0]
        
        if isinstance(dt, (float, int)):
            dt = torch.ones(batch_size, 1, device=x.device) * dt
        elif dt.dim() == 0:
            dt = dt.unsqueeze(0).unsqueeze(0).expand(batch_size, 1)
        elif dt.dim() == 1:
            dt = dt.unsqueeze(-1)
        
        # 拼接输入
        inputs = torch.cat([x, dt], dim=-1)
        
        # 前向传播
        h = self.input_proj(inputs)
        for block in self.blocks:
            h = block(h)
        dx = self.output_proj(h)
        
        if self.use_residual:
            return x + dx
        return dx
    
    def multi_step_predict(self, x0, dt, n_steps):
        if x0.dim() == 1:
            x0 = x0.unsqueeze(0)
        
        trajectory = [x0]
        x = x0
        
        for _ in range(n_steps):
            x = self.forward(x, dt)
            trajectory.append(x)
        
        return torch.stack(trajectory, dim=0)


class ResidualBlock(nn.Module):
    """残差块"""
    
    def __init__(self, dim: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Linear(dim, dim),
            nn.Tanh(),
            nn.Linear(dim, dim)
        )
    
    def forward(self, x):
        return x + self.block(x)


class FlowMapCNN(nn.Module):
    """
    Flow Map CNN for PDEs
    
    用于学习场量（如温度场、速度场）的时间演化
    适用于网格化数据
    """
    
    def __init__(self,
                 in_channels: int = 1,
                 hidden_channels: int = 32,
                 num_layers: int = 4,
                 kernel_size: int = 3,
                 use_residual: bool = True):
        """
        Args:
            in_channels: 输入通道数（物理场数量）
            hidden_channels: 隐藏层通道数
            num_layers: 卷积层数量
            kernel_size: 卷积核大小
            use_residual: 是否使用残差连接
        """
        super(FlowMapCNN, self).__init__()
        
        self.in_channels = in_channels
        self.use_residual = use_residual
        padding = kernel_size // 2
        
        # 输入层（包含时间信息的通道）
        self.input_conv = nn.Conv1d(in_channels + 1, hidden_channels, 
                                    kernel_size, padding=padding)
        
        # 中间卷积层
        self.conv_layers = nn.ModuleList()
        for _ in range(num_layers - 2):
            self.conv_layers.append(
                nn.Sequential(
                    nn.Conv1d(hidden_channels, hidden_channels, 
                             kernel_size, padding=padding),
                    nn.BatchNorm1d(hidden_channels),
                    nn.ReLU()
                )
            )
        
        # 输出层
        self.output_conv = nn.Conv1d(hidden_channels, in_channels, 
                                     kernel_size, padding=padding)
    
    def forward(self, u: torch.Tensor, dt: Union[torch.Tensor, float]) -> torch.Tensor:
        """
        Args:
            u: 当前场量 [batch, channels, nx] (1D) 或 [batch, channels, nx, ny] (2D)
            dt: 时间步长
        
        Returns:
            u_next: 下一时刻场量
        """
        batch_size = u.shape[0]
        spatial_shape = u.shape[2:]
        
        # 创建时间通道
        if isinstance(dt, (float, int)):
            dt_tensor = torch.ones(batch_size, 1, *spatial_shape, device=u.device) * dt
        else:
            dt_tensor = dt.view(batch_size, 1, *([1] * len(spatial_shape))).expand(
                batch_size, 1, *spatial_shape
            )
        
        # 拼接时间通道
        x = torch.cat([u, dt_tensor], dim=1)
        
        # 前向传播
        x = torch.relu(self.input_conv(x))
        for conv in self.conv_layers:
            x = conv(x)
        du = self.output_conv(x)
        
        if self.use_residual:
            return u + du
        return du
    
    def multi_step_predict(self, u0, dt, n_steps):
        trajectory = [u0]
        u = u0
        
        for _ in range(n_steps):
            u = self.forward(u, dt)
            trajectory.append(u)
        
        return torch.stack(trajectory, dim=0)


class FlowMapGRU(nn.Module):
    """
    Flow Map with GRU Memory
    
    使用 GRU 捕捉历史信息，适用于非马尔可夫系统
    或需要记忆的降阶模型
    """
    
    def __init__(self,
                 state_dim: int,
                 hidden_dim: int = 64,
                 num_layers: int = 2,
                 use_residual: bool = True):
        """
        Args:
            state_dim: 状态维度
            hidden_dim: GRU 隐藏维度
            num_layers: GRU 层数
            use_residual: 是否使用残差连接
        """
        super(FlowMapGRU, self).__init__()
        
        self.state_dim = state_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.use_residual = use_residual
        
        # 输入编码
        self.input_encoder = nn.Linear(state_dim + 1, hidden_dim)
        
        # GRU
        self.gru = nn.GRU(hidden_dim, hidden_dim, num_layers, batch_first=True)
        
        # 输出解码
        self.output_decoder = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, state_dim)
        )
    
    def forward(self, x, dt, hidden=None):
        """
        Args:
            x: 当前状态 [batch, state_dim]
            dt: 时间步长
            hidden: GRU 隐藏状态
        
        Returns:
            x_next: 下一状态
            hidden: 更新的隐藏状态
        """
        batch_size = x.shape[0]
        
        if isinstance(dt, (float, int)):
            dt = torch.ones(batch_size, 1, device=x.device) * dt
        elif dt.dim() == 0:
            dt = dt.unsqueeze(0).unsqueeze(0).expand(batch_size, 1)
        elif dt.dim() == 1:
            dt = dt.unsqueeze(-1)
        
        # 编码输入
        inputs = torch.cat([x, dt], dim=-1)
        h = self.input_encoder(inputs).unsqueeze(1)  # [batch, 1, hidden]
        
        # GRU
        out, hidden = self.gru(h, hidden)
        
        # 解码输出
        dx = self.output_decoder(out.squeeze(1))
        
        if self.use_residual:
            return x + dx, hidden
        return dx, hidden
    
    def multi_step_predict(self, x0, dt, n_steps):
        if x0.dim() == 1:
            x0 = x0.unsqueeze(0)
        
        trajectory = [x0]
        x = x0
        hidden = None
        
        for _ in range(n_steps):
            x, hidden = self.forward(x, dt, hidden)
            trajectory.append(x)
        
        return torch.stack(trajectory, dim=0)


class HamiltonianFlowMap(nn.Module):
    """
    Hamiltonian Flow Map
    
    保辛结构的 Flow Map，适用于哈密顿系统
    保证能量在长期预测中近似守恒
    """
    
    def __init__(self, 
                 dim: int,
                 hidden_dims: List[int] = [64, 64]):
        """
        Args:
            dim: 位置/动量空间维度 (总状态维度 = 2*dim)
            hidden_dims: 隐藏层维度
        """
        super(HamiltonianFlowMap, self).__init__()
        
        self.dim = dim
        
        # 学习哈密顿量 H(q, p)
        layers = []
        input_dim = 2 * dim
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(input_dim, hidden_dim),
                nn.Tanh()
            ])
            input_dim = hidden_dim
        layers.append(nn.Linear(input_dim, 1))
        
        self.hamiltonian = nn.Sequential(*layers)
    
    def forward(self, qp: torch.Tensor, dt: float) -> torch.Tensor:
        """
        使用辛欧拉方法积分
        
        Args:
            qp: [batch, 2*dim] 前半部分是 q，后半部分是 p
            dt: 时间步长
        
        Returns:
            qp_next: 下一时刻的 (q, p)
        """
        q = qp[:, :self.dim]
        p = qp[:, self.dim:]
        
        # 计算 ∂H/∂p 和 ∂H/∂q
        qp.requires_grad_(True)
        H = self.hamiltonian(qp)
        
        grad_H = torch.autograd.grad(H.sum(), qp, create_graph=True)[0]
        dH_dq = grad_H[:, :self.dim]
        dH_dp = grad_H[:, self.dim:]
        
        # 辛欧拉积分
        # q_next = q + dt * ∂H/∂p
        # p_next = p - dt * ∂H/∂q
        q_next = q + dt * dH_dp
        p_next = p - dt * dH_dq
        
        return torch.cat([q_next, p_next], dim=-1)
    
    def compute_energy(self, qp: torch.Tensor) -> torch.Tensor:
        """计算系统能量"""
        return self.hamiltonian(qp)


def count_parameters(model: nn.Module) -> int:
    """统计模型参数数量"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# 便捷函数
def create_flowmap_model(model_type: str = 'mlp', **kwargs) -> nn.Module:
    """
    创建 Flow Map 模型
    
    Args:
        model_type: 模型类型 ('mlp', 'resnet', 'cnn', 'gru', 'hamiltonian')
        **kwargs: 模型参数
    
    Returns:
        Flow Map 模型
    """
    models = {
        'mlp': FlowMapMLP,
        'resnet': FlowMapResNet,
        'cnn': FlowMapCNN,
        'gru': FlowMapGRU,
        'hamiltonian': HamiltonianFlowMap
    }
    
    if model_type not in models:
        raise ValueError(f"Unknown model type: {model_type}. "
                        f"Available: {list(models.keys())}")
    
    return models[model_type](**kwargs)
