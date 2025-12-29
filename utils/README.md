# AI4CFD 工具库 (Utils)

[![Version](https://img.shields.io/badge/version-2.0.0-blue.svg)](https://github.com/Michael-Jackson666/AI4CFD)
[![Python](https://img.shields.io/badge/Python-3.8+-brightgreen.svg)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.10+-orange.svg)](https://pytorch.org)

本目录包含 AI4CFD 项目的**综合工具库**，提供快速实现 PINNs、DeepONet、FNO、TNN、Transformer 等 AI4CFD 算法所需的全部组件。

## 🚀 快速开始

```python
# 一行导入所有工具
from utils import (
    # 快速创建模型
    create_pinn, create_deeponet, create_fno, create_tnn, create_pde_transformer,
    # 训练工具
    train_model, PINNTrainer, FNOTrainer,
    # 数据生成
    generate_burgers_data, generate_navier_stokes_data,
    # 评估和可视化
    relative_l2_error, plot_2d_solution
)

# 快速创建 PINN 模型
model = create_pinn(input_dim=2, output_dim=1, hidden_dims=[64, 64, 64])

# 快速创建 FNO 模型
fno = create_fno(modes=16, width=64, dim=2)

# 快速创建 DeepONet
deeponet = create_deeponet(branch_input_dim=100, trunk_input_dim=1)
```

## 📁 模块结构

```
utils/
├── __init__.py       # 统一导出接口 + 快速创建函数
├── data_utils.py     # 数据生成、加载、预处理
├── nn_blocks.py      # 神经网络构建模块
├── training.py       # 训练工具、损失函数、优化器
├── trainers.py       # 各类方法的专用 Trainer
├── metrics.py        # 评估指标
├── plotting.py       # 可视化工具
└── README.md         # 本文档
```

---

## 📚 详细文档

### 1️⃣ `nn_blocks.py` - 神经网络模块

提供所有 AI4CFD 方法的核心网络组件：

#### 基础模块

| 类名 | 说明 | 使用场景 |
|------|------|----------|
| `MLP` | 多层感知机 | 通用基础网络 |
| `FourierFeatures` | 傅里叶特征编码 | 捕获高频信息 |
| `ModifiedMLP` | 改进版 MLP | 更好的表达能力 |
| `ResidualBlock` | 残差块 | 深层网络训练 |
| `ResMLP` | 残差 MLP | 避免梯度消失 |

#### PINNs 模块

| 类名 | 说明 |
|------|------|
| `PINN` | 标准物理信息神经网络 |
| `AdaptiveWeightPINN` | 自适应权重 PINN（自动平衡损失项） |

```python
from utils import PINN, AdaptiveWeightPINN

# 标准 PINN
pinn = PINN(input_dim=2, output_dim=1, hidden_dims=[64, 64, 64, 64])

# 自适应权重 PINN
adaptive_pinn = AdaptiveWeightPINN(input_dim=2, output_dim=1, hidden_dims=[64]*4)
```

#### DeepONet 模块

| 类名 | 说明 |
|------|------|
| `DeepONet` | 标准 DeepONet |
| `StackedDeepONet` | 多层堆叠 DeepONet |

```python
from utils import DeepONet, StackedDeepONet

# 学习算子: u(x) -> G(u)(y)
deeponet = DeepONet(
    branch_input_dim=100,   # 传感器点数
    trunk_input_dim=1,      # 查询点维度
    branch_layers=[100, 100],
    trunk_layers=[100, 100],
    p=50                    # 输出维度
)

# 堆叠版本（更强表达能力）
stacked = StackedDeepONet(branch_input_dim=100, trunk_input_dim=1, num_layers=3)
```

#### FNO 模块

| 类名 | 说明 |
|------|------|
| `SpectralConv1d` | 1D 谱卷积层 |
| `SpectralConv2d` | 2D 谱卷积层 |
| `FNO1d` | 1D 傅里叶神经算子 |
| `FNO2d` | 2D 傅里叶神经算子 |

```python
from utils import FNO1d, FNO2d

# 1D FNO（如 Burgers 方程）
fno_1d = FNO1d(in_channels=1, out_channels=1, modes=16, width=64)

# 2D FNO（如 Navier-Stokes）
fno_2d = FNO2d(in_channels=1, out_channels=1, modes1=12, modes2=12, width=32)
```

#### TNN 模块

| 类名 | 说明 |
|------|------|
| `TensorLayer` | 张量分解层 |
| `TNN` | 张量神经网络 |
| `TuckerTNN` | Tucker 分解 TNN |

```python
from utils import TNN, TuckerTNN

# 标准 TNN
tnn = TNN(input_dim=3, output_dim=1, rank=20)

# Tucker 分解版本
tucker_tnn = TuckerTNN(input_dim=3, output_dim=1, rank=15)
```

#### Transformer 模块

| 类名 | 说明 |
|------|------|
| `PositionalEncoding` | 位置编码 |
| `PDETransformer` | PDE 求解 Transformer |
| `SpatioTemporalTransformer` | 时空 Transformer |

```python
from utils import PDETransformer, SpatioTemporalTransformer

# PDE Transformer
pde_trans = PDETransformer(
    input_dim=2, output_dim=1,
    d_model=64, nhead=4, num_layers=4
)

# 时空 Transformer（适合时间演化问题）
st_trans = SpatioTemporalTransformer(
    spatial_dim=2, temporal_dim=1, output_dim=1
)
```

---

### 2️⃣ `data_utils.py` - 数据工具

#### PDE 数据生成

| 函数 | 说明 |
|------|------|
| `generate_1d_poisson_data()` | 1D Poisson 方程 |
| `generate_2d_poisson_data()` | 2D Poisson 方程 |
| `generate_heat_equation_data()` | 热传导方程 |
| `generate_burgers_data()` | Burgers 方程（谱方法求解） |
| `generate_navier_stokes_data()` | 2D Navier-Stokes（涡量-流函数） |
| `generate_wave_equation_data()` | 波动方程 |

```python
from utils import generate_burgers_data, generate_navier_stokes_data

# Burgers 方程数据
x, t, u = generate_burgers_data(n_x=256, n_t=100, nu=0.01)

# Navier-Stokes 数据
x, y, t, omega = generate_navier_stokes_data(n_x=64, n_y=64, n_t=20, Re=1000)
```

#### 算子学习数据

| 函数 | 说明 |
|------|------|
| `generate_operator_data()` | DeepONet 算子学习数据 |
| `generate_parametric_pde_data()` | 参数化 PDE 数据 |
| `generate_fno_data()` | FNO 训练数据 |

```python
from utils import generate_operator_data, generate_fno_data

# DeepONet: 学习反导数算子
u_sensors, y_query, G_u = generate_operator_data(
    operator_type='antiderivative', n_samples=1000
)

# FNO: Darcy 流或 Navier-Stokes
train_data, test_data = generate_fno_data(pde_type='darcy', n_samples=1000)
```

#### 边界/初始条件

| 函数 | 说明 |
|------|------|
| `create_boundary_conditions()` | 创建边界条件点 |
| `create_initial_conditions()` | 创建初始条件点 |
| `create_mesh_grid()` | 创建多维网格 |

```python
from utils import create_boundary_conditions, create_mesh_grid

# 2D 边界条件
bc_coords, bc_vals = create_boundary_conditions(
    domain=[(-1, 1), (-1, 1)], n_points=100, bc_type='dirichlet', bc_value=0
)

# 创建网格
coords = create_mesh_grid(domain=[(0, 1), (0, 1)], n_points=[50, 50])
```

#### DataLoader 工具

| 类/函数 | 说明 |
|---------|------|
| `PDEDataset` | 通用 PDE 数据集 |
| `DeepONetDataset` | DeepONet 数据集 |
| `create_training_dataloader()` | 创建训练 DataLoader |
| `create_fno_dataloader()` | 创建 FNO DataLoader |
| `create_deeponet_dataloader()` | 创建 DeepONet DataLoader |

---

### 3️⃣ `training.py` - 训练工具

#### 损失函数

| 类名 | 说明 |
|------|------|
| `PINNLoss` | PINN 复合损失（PDE + BC + IC） |
| `WeightedMSELoss` | 加权 MSE 损失 |
| `RelativeMSELoss` | 相对 MSE 损失 |
| `SobolevLoss` | Sobolev 范数损失（含导数项） |
| `SpectralLoss` | 谱空间损失 |

```python
from utils import PINNLoss, SobolevLoss

# PINN 损失
loss_fn = PINNLoss(pde_weight=1.0, bc_weight=100.0, ic_weight=100.0)

# Sobolev 损失（考虑梯度匹配）
sobolev = SobolevLoss(order=1, weight=0.1)
```

#### PDE 残差计算

| 函数 | 说明 |
|------|------|
| `compute_pde_residual()` | 计算 PDE 残差（支持多种方程） |
| `compute_derivative()` | 计算任意阶导数 |
| `compute_laplacian()` | 计算拉普拉斯算子 |
| `compute_gradient()` | 计算梯度 |
| `compute_divergence()` | 计算散度 |

```python
from utils import compute_pde_residual, compute_laplacian

# 计算 Burgers 方程残差
residual = compute_pde_residual(coords, u, pde_type='burgers', nu=0.01)

# 计算拉普拉斯
laplacian = compute_laplacian(coords, u)
```

#### 优化器和调度器

| 函数 | 说明 |
|------|------|
| `get_optimizer()` | 获取优化器（Adam, SGD, LBFGS等） |
| `get_scheduler()` | 获取学习率调度器 |
| `WarmupCosineScheduler` | 预热+余弦衰减 |
| `train_with_lbfgs()` | L-BFGS 精细化训练 |

```python
from utils import get_optimizer, get_scheduler, train_with_lbfgs

# 获取优化器
optimizer = get_optimizer(model, name='adam', lr=1e-3, weight_decay=1e-4)

# 获取调度器
scheduler = get_scheduler(optimizer, name='cosine', T_max=1000)

# L-BFGS 精细化
model = train_with_lbfgs(model, loss_fn, data, max_iter=500)
```

#### 训练辅助工具

| 类/函数 | 说明 |
|---------|------|
| `EarlyStopping` | 早停机制 |
| `GradientBalancer` | 梯度平衡（多任务学习） |
| `adaptive_sampling()` | 自适应采样（基于残差） |
| `gradient_clipping()` | 梯度裁剪 |

---

### 4️⃣ `trainers.py` - 专用训练器

提供各类方法的专用 Trainer：

| 类名 | 用于 |
|------|------|
| `BaseTrainer` | 基础训练器 |
| `PINNTrainer` | PINNs（支持 L-BFGS） |
| `DeepONetTrainer` | DeepONet |
| `FNOTrainer` | FNO |
| `TNNTrainer` | TNN |

```python
from utils import PINNTrainer, FNOTrainer

# PINN 训练器
pinn_trainer = PINNTrainer(
    model, 
    pde_loss_fn=burgers_residual,
    bc_data=bc_data,
    ic_data=ic_data
)
history = pinn_trainer.train(train_data, epochs=10000, lr=1e-3)

# FNO 训练器
fno_trainer = FNOTrainer(model)
history = fno_trainer.train(train_loader, epochs=500, lr=1e-3)
```

---

### 5️⃣ `metrics.py` - 评估指标

| 函数 | 说明 |
|------|------|
| `mse_loss()` | 均方误差 |
| `mae_loss()` | 平均绝对误差 |
| `relative_l2_error()` | 相对 L² 误差：$\frac{\|\|u - u_{exact}\|\|_2}{\|\|u_{exact}\|\|_2}$ |
| `relative_linf_error()` | 相对 L∞ 误差 |
| `physics_residual_l2()` | 物理残差 L² 范数 |
| `conservation_error()` | 守恒律误差 |
| `energy_error()` | 能量误差 |
| `evaluate_model_performance()` | 综合性能评估 |

```python
from utils import relative_l2_error, evaluate_model_performance

# 单个指标
l2_err = relative_l2_error(u_pred, u_exact)
print(f"Relative L2 error: {l2_err:.4e}")

# 综合评估
metrics = evaluate_model_performance(u_pred, u_exact, coords, model)
print(metrics)
```

---

### 6️⃣ `plotting.py` - 可视化

| 函数 | 说明 |
|------|------|
| `plot_1d_solution()` | 1D 解对比图 |
| `plot_2d_solution()` | 2D 解等高线+3D 曲面 |
| `plot_2d_comparison()` | 预测/真实/误差三合一 |
| `plot_training_history()` | 训练历史曲线 |
| `plot_burgers_evolution()` | Burgers 方程时间演化 |
| `plot_residuals()` | 物理残差分布 |
| `save_animation_frames()` | 保存动画帧 |

```python
from utils import plot_2d_comparison, plot_training_history

# 2D 解对比
plot_2d_comparison(X, Y, u_pred, u_exact, title="Poisson Solution")

# 训练历史
plot_training_history(history, metrics=['loss', 'l2_error'])
```

---

## 🎯 完整示例

### 示例 1: 使用 PINN 求解 Burgers 方程

```python
import torch
from utils import (
    create_pinn, generate_burgers_data, 
    create_boundary_conditions, create_initial_conditions,
    PINNTrainer, compute_pde_residual,
    plot_2d_comparison, relative_l2_error
)

# 1. 准备数据
x, t, u_exact = generate_burgers_data(n_x=256, n_t=100, nu=0.01)

# 2. 创建模型
model = create_pinn(input_dim=2, output_dim=1, hidden_dims=[64, 64, 64, 64])

# 3. 定义 PDE 残差
def burgers_residual(coords, u):
    return compute_pde_residual(coords, u, pde_type='burgers', nu=0.01)

# 4. 准备边界和初始条件
bc_data = create_boundary_conditions(domain=[(-1, 1), (0, 1)], n_points=100)
ic_data = create_initial_conditions(domain=[(-1, 1)], n_points=100, 
                                    ic_function=lambda x: -np.sin(np.pi * x))

# 5. 训练
trainer = PINNTrainer(model, pde_loss_fn=burgers_residual, 
                      bc_data=bc_data, ic_data=ic_data)
history = trainer.train(epochs=10000, lr=1e-3)

# 6. 评估
u_pred = model(test_coords)
print(f"Relative L2 error: {relative_l2_error(u_pred, u_exact):.4e}")
```

### 示例 2: 使用 FNO 求解 Darcy 流

```python
from utils import (
    create_fno, generate_fno_data, create_fno_dataloader,
    FNOTrainer, relative_l2_error
)

# 1. 生成数据
train_data, test_data = generate_fno_data(pde_type='darcy', n_samples=1000)
train_loader = create_fno_dataloader(train_data, batch_size=20)

# 2. 创建 FNO
fno = create_fno(modes=12, width=32, dim=2)

# 3. 训练
trainer = FNOTrainer(fno)
history = trainer.train(train_loader, epochs=500)

# 4. 评估
with torch.no_grad():
    pred = fno(test_data['input'])
print(f"Test L2 error: {relative_l2_error(pred, test_data['output']):.4e}")
```

### 示例 3: 使用 DeepONet 学习算子

```python
from utils import (
    create_deeponet, generate_operator_data, create_deeponet_dataloader,
    DeepONetTrainer
)

# 1. 生成算子数据（学习反导数）
u_sensors, y_query, G_u = generate_operator_data(
    operator_type='antiderivative', n_samples=1000
)

# 2. 创建 DeepONet
deeponet = create_deeponet(
    branch_input_dim=100, trunk_input_dim=1,
    hidden_dim=100, p=50
)

# 3. 训练
loader = create_deeponet_dataloader(u_sensors, y_query, G_u)
trainer = DeepONetTrainer(deeponet)
history = trainer.train(loader, epochs=1000)
```

---

## 📖 API 速查表

### 快速创建函数

```python
model = create_pinn(input_dim, output_dim, hidden_dims, activation, use_fourier, use_adaptive_weights)
model = create_deeponet(branch_input_dim, trunk_input_dim, hidden_dim, p, branch_layers, trunk_layers)
model = create_fno(in_channels, out_channels, modes, width, dim, depth)
model = create_tnn(input_dim, output_dim, rank, layers_per_dim, hidden_dim, use_tucker)
model = create_pde_transformer(input_dim, output_dim, d_model, nhead, num_layers, dim_feedforward)
```

### 通用训练函数

```python
model, history = train_model(
    model, train_data, 
    epochs=1000, lr=1e-3, 
    method='pinn',           # 'pinn', 'deeponet', 'fno', 'tnn'
    pde_loss_fn=residual_fn, # for PINNs
    bc_data=bc_data,         # boundary conditions
    ic_data=ic_data,         # initial conditions
    device='cuda',
    verbose=True,
    save_path='model.pt'
)
```

---

## 📦 导入方式

```python
# 方式 1: 完整导入
from utils import *

# 方式 2: 选择性导入
from utils import MLP, PINN, FNO2d, DeepONet
from utils import generate_burgers_data, create_boundary_conditions
from utils import PINNTrainer, relative_l2_error

# 方式 3: 使用快速创建函数
from utils import create_pinn, create_fno, train_model
```

---

## 📝 版本历史

- **v2.0.0** (2024-12): 重大更新
  - 新增 `nn_blocks.py`: 完整的神经网络模块库
  - 新增 `training.py`: 损失函数、PDE 残差、优化器工具
  - 新增 `trainers.py`: 各方法专用训练器
  - 更新 `data_utils.py`: 增加 Navier-Stokes、波动方程数据生成
  - 更新 `__init__.py`: 统一接口 + 快速创建函数

- **v1.0.0** (2024-01): 初始版本
  - 基础数据工具、指标、可视化
|------|------|
| `setup_plotting_style()` | 设置统一绘图风格 |
| `plot_1d_solution()` | 绘制 1D 解对比图 |
| `plot_2d_solution()` | 绘制 2D 解热力图 |
| `plot_3d_surface()` | 绘制 3D 表面图 |
| `plot_error_distribution()` | 绘制误差分布图 |
| `plot_training_history()` | 绘制训练损失曲线 |

**示例**：
```python
from utils.plotting import plot_1d_solution, plot_training_history

# 绘制解对比
plot_1d_solution(x, u_pred, u_exact, title="Poisson Solution")

# 绘制训练曲线
plot_training_history(loss_history, title="Training Loss")
```

## 使用方法

### 导入方式

```python
# 导入单个函数
from utils.metrics import relative_l2_error

# 导入整个模块
from utils import data_utils, metrics, plotting
```

### 依赖库

```
numpy
torch
matplotlib
seaborn
scipy
```

## 兼容性

- 支持 NumPy 数组和 PyTorch 张量
- 自动检测输入类型并选择对应实现
- GPU 张量会自动转移到 CPU 进行可视化
