# AI4CFD Utility Package (Utils)

[![Version](https://img.shields.io/badge/version-2.0.0-blue.svg)](https://github.com/Michael-Jackson666/AI4CFD)
[![Python](https://img.shields.io/badge/Python-3.8+-brightgreen.svg)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.10+-orange.svg)](https://pytorch.org)

This directory contains the **comprehensive utility library** for the AI4CFD project, providing all necessary components for quickly implementing AI4CFD algorithms such as PINNs, DeepONet, FNO, TNN, and Transformers.

## 🚀 Quick Start

```python
# Import all tools in one line
from utils import (
    # Quickly create models
    create_pinn, create_deeponet, create_fno, create_tnn, create_pde_transformer,
    # Training tools
    train_model, PINNTrainer, FNOTrainer,
    # Data generation
    generate_burgers_data, generate_navier_stokes_data,
    # Evaluation and visualization
    relative_l2_error, plot_2d_solution
)

# Quickly create PINN model
model = create_pinn(input_dim=2, output_dim=1, hidden_dims=[64, 64, 64])

# Quickly create FNO model
fno = create_fno(modes=16, width=64, dim=2)

# Quickly create DeepONet
deeponet = create_deeponet(branch_input_dim=100, trunk_input_dim=1)
```

## 📁 Module Structure

```
utils/
├── __init__.py       # Unified export interface + quick creation functions
├── data_utils.py     # Data generation, loading, preprocessing
├── nn_blocks.py      # Neural network building blocks
├── training.py       # Training tools, loss functions, optimizers
├── trainers.py       # Specialized Trainers for various methods
├── metrics.py        # Evaluation metrics
├── plotting.py       # Visualization tools
└── README.md         # This document
```

---

## 📚 Detailed Documentation

### 1️⃣ `nn_blocks.py` - Neural Network Modules

Provides core network components for all AI4CFD methods:

#### Basic Modules

| Class Name | Description | Usage Scenario |
|------|------|----------|
| `MLP` | Multi-Layer Perceptron | General basic network |
| `FourierFeatures` | Fourier Feature Encoding | Capturing high-frequency information |
| `ModifiedMLP` | Modified MLP | Better expressivity |
| `ResidualBlock` | Residual Block | Deep network training |
| `ResMLP` | Residual MLP | Avoiding gradient vanishing |

#### PINNs Modules

| Class Name | Description |
|------|------|
| `PINN` | Standard Physics-Informed Neural Network |
| `AdaptiveWeightPINN` | Adaptive Weight PINN (Auto-balancing loss terms) |

```python
from utils import PINN, AdaptiveWeightPINN

# Standard PINN
pinn = PINN(input_dim=2, output_dim=1, hidden_dims=[64, 64, 64, 64])

# Adaptive Weight PINN
adaptive_pinn = AdaptiveWeightPINN(input_dim=2, output_dim=1, hidden_dims=[64]*4)
```

#### DeepONet Modules

| Class Name | Description |
|------|------|
| `DeepONet` | Standard DeepONet |
| `StackedDeepONet` | Stacked DeepONet |

```python
from utils import DeepONet, StackedDeepONet

# Learn operator: u(x) -> G(u)(y)
deeponet = DeepONet(
    branch_input_dim=100,   # Number of sensor points
    trunk_input_dim=1,      # Query point dimension
    branch_layers=[100, 100],
    trunk_layers=[100, 100],
    p=50                    # Output dimension
)

# Stacked version (Stronger expressivity)
stacked = StackedDeepONet(branch_input_dim=100, trunk_input_dim=1, num_layers=3)
```

#### FNO Modules

| Class Name | Description |
|------|------|
| `SpectralConv1d` | 1D Spectral Convolution Layer |
| `SpectralConv2d` | 2D Spectral Convolution Layer |
| `FNO1d` | 1D Fourier Neural Operator |
| `FNO2d` | 2D Fourier Neural Operator |

```python
from utils import FNO1d, FNO2d

# 1D FNO (e.g., Burgers equation)
fno_1d = FNO1d(in_channels=1, out_channels=1, modes=16, width=64)

# 2D FNO (e.g., Navier-Stokes)
fno_2d = FNO2d(in_channels=1, out_channels=1, modes1=12, modes2=12, width=32)
```

#### TNN Modules

| Class Name | Description |
|------|------|
| `TensorLayer` | Tensor Decomposition Layer |
| `TNN` | Tensor Neural Network |
| `TuckerTNN` | Tucker Decomposition TNN |

```python
from utils import TNN, TuckerTNN

# Standard TNN
tnn = TNN(input_dim=3, output_dim=1, rank=20)

# Tucker Decomposition version
tucker_tnn = TuckerTNN(input_dim=3, output_dim=1, rank=15)
```

#### Transformer Modules

| Class Name | Description |
|------|------|
| `PositionalEncoding` | Positional Encoding |
| `PDETransformer` | PDE Solving Transformer |
| `SpatioTemporalTransformer` | Spatio-Temporal Transformer |

```python
from utils import PDETransformer, SpatioTemporalTransformer

# PDE Transformer
pde_trans = PDETransformer(
    input_dim=2, output_dim=1,
    d_model=64, nhead=4, num_layers=4
)

# Spatio-Temporal Transformer (Suitable for time evolution problems)
st_trans = SpatioTemporalTransformer(
    spatial_dim=2, temporal_dim=1, output_dim=1
)
```

---

### 2️⃣ `data_utils.py` - Data Utilities

#### PDE Data Generation

| Function | Description |
|------|------|
| `generate_1d_poisson_data()` | 1D Poisson Equation |
| `generate_2d_poisson_data()` | 2D Poisson Equation |
| `generate_heat_equation_data()` | Heat Equation |
| `generate_burgers_data()` | Burgers' Equation (Spectral Method) |
| `generate_navier_stokes_data()` | 2D Navier-Stokes (Vorticity-Stream Function) |
| `generate_wave_equation_data()` | Wave Equation |

```python
from utils import generate_burgers_data, generate_navier_stokes_data

# Burgers equation data
x, t, u = generate_burgers_data(n_x=256, n_t=100, nu=0.01)

# Navier-Stokes data
x, y, t, omega = generate_navier_stokes_data(n_x=64, n_y=64, n_t=20, Re=1000)
```

#### Operator Learning Data

| Function | Description |
|------|------|
| `generate_operator_data()` | DeepONet Operator Learning Data |
| `generate_parametric_pde_data()` | Parametric PDE Data |
| `generate_fno_data()` | FNO Training Data |

```python
from utils import generate_operator_data, generate_fno_data

# DeepONet: Learn antiderivative operator
u_sensors, y_query, G_u = generate_operator_data(
    operator_type='antiderivative', n_samples=1000
)

# FNO: Darcy Flow or Navier-Stokes
train_data, test_data = generate_fno_data(pde_type='darcy', n_samples=1000)
```

#### Boundary/Initial Conditions

| Function | Description |
|------|------|
| `create_boundary_conditions()` | Create Boundary Condition Points |
| `create_initial_conditions()` | Create Initial Condition Points |
| `create_mesh_grid()` | Create Multi-dimensional Mesh Grid |

```python
from utils import create_boundary_conditions, create_mesh_grid

# 2D Boundary Conditions
bc_coords, bc_vals = create_boundary_conditions(
    domain=[(-1, 1), (-1, 1)], n_points=100, bc_type='dirichlet', bc_value=0
)

# Create Mesh Grid
coords = create_mesh_grid(domain=[(0, 1), (0, 1)], n_points=[50, 50])
```

#### DataLoader Tools

| Class/Function | Description |
|---------|------|
| `PDEDataset` | Generic PDE Dataset |
| `DeepONetDataset` | DeepONet Dataset |
| `create_training_dataloader()` | Create Training DataLoader |
| `create_fno_dataloader()` | Create FNO DataLoader |
| `create_deeponet_dataloader()` | Create DeepONet DataLoader |

---

### 3️⃣ `training.py` - Training Tools

#### Loss Functions

| Class Name | Description |
|------|------|
| `PINNLoss` | PINN Composite Loss (PDE + BC + IC) |
| `WeightedMSELoss` | Weighted MSE Loss |
| `RelativeMSELoss` | Relative MSE Loss |
| `SobolevLoss` | Sobolev Norm Loss (Includes derivative terms) |
| `SpectralLoss` | Spectral Space Loss |

```python
from utils import PINNLoss, SobolevLoss

# PINN Loss
loss_fn = PINNLoss(pde_weight=1.0, bc_weight=100.0, ic_weight=100.0)

# Sobolev Loss (Consider gradient matching)
sobolev = SobolevLoss(order=1, weight=0.1)
```

#### PDE Residual Calculation

| Function | Description |
|------|------|
| `compute_pde_residual()` | Compute PDE Residual (Supports multiple equations) |
| `compute_derivative()` | Compute Arbitrary Order Derivative |
| `compute_laplacian()` | Compute Laplacian Operator |
| `compute_gradient()` | Compute Gradient |
| `compute_divergence()` | Compute Divergence |

```python
from utils import compute_pde_residual, compute_laplacian

# Compute Burgers equation residual
residual = compute_pde_residual(coords, u, pde_type='burgers', nu=0.01)

# Compute Laplacian
laplacian = compute_laplacian(coords, u)
```

#### Optimizers and Schedulers

| Function | Description |
|------|------|
| `get_optimizer()` | Get Optimizer (Adam, SGD, LBFGS, etc.) |
| `get_scheduler()` | Get Learning Rate Scheduler |
| `WarmupCosineScheduler` | Warmup + Cosine Decay |
| `train_with_lbfgs()` | L-BFGS Refined Training |

```python
from utils import get_optimizer, get_scheduler, train_with_lbfgs

# Get Optimizer
optimizer = get_optimizer(model, name='adam', lr=1e-3, weight_decay=1e-4)

# Get Scheduler
scheduler = get_scheduler(optimizer, name='cosine', T_max=1000)

# L-BFGS Refinement
model = train_with_lbfgs(model, loss_fn, data, max_iter=500)
```

#### Training Helper Tools

| Class/Function | Description |
|---------|------|
| `EarlyStopping` | Early Stopping Mechanism |
| `GradientBalancer` | Gradient Balancing (Multi-task Learning) |
| `adaptive_sampling()` | Adaptive Sampling (Based on Residuals) |
| `gradient_clipping()` | Gradient Clipping |

---

### 4️⃣ `trainers.py` - Specialized Trainers

Provides specialized Trainers for various methods:

| Class Name | Used For |
|------|------|
| `BaseTrainer` | Base Trainer |
| `PINNTrainer` | PINNs (Supports L-BFGS) |
| `DeepONetTrainer` | DeepONet |
| `FNOTrainer` | FNO |
| `TNNTrainer` | TNN |

```python
from utils import PINNTrainer, FNOTrainer

# PINN Trainer
pinn_trainer = PINNTrainer(
    model, 
    pde_loss_fn=burgers_residual,
    bc_data=bc_data,
    ic_data=ic_data
)
history = pinn_trainer.train(train_data, epochs=10000, lr=1e-3)

# FNO Trainer
fno_trainer = FNOTrainer(model)
history = fno_trainer.train(train_loader, epochs=500, lr=1e-3)
```

---

### 5️⃣ `metrics.py` - Evaluation Metrics

| Function | Description |
|------|------|
| `mse_loss()` | Mean Squared Error |
| `mae_loss()` | Mean Absolute Error |
| `relative_l2_error()` | Relative L² Error: $\frac{\|\|u - u_{exact}\|\|_2}{\|\|u_{exact}\|\|_2}$ |
| `relative_linf_error()` | Relative L∞ Error |
| `physics_residual_l2()` | Physics Residual L² Norm |
| `conservation_error()` | Conservation Law Error |
| `energy_error()` | Energy Error |
| `evaluate_model_performance()` | Comprehensive Performance Evaluation |

```python
from utils import relative_l2_error, evaluate_model_performance

# Single Metric
l2_err = relative_l2_error(u_pred, u_exact)
print(f"Relative L2 error: {l2_err:.4e}")

# Comprehensive Evaluation
metrics = evaluate_model_performance(u_pred, u_exact, coords, model)
print(metrics)
```

---

### 6️⃣ `plotting.py` - Visualization

| Function | Description |
|------|------|
| `plot_1d_solution()` | 1D Solution Comparison Plot |
| `plot_2d_solution()` | 2D Solution Contour + 3D Surface |
| `plot_2d_comparison()` | Prediction/Ground Truth/Error Comparison |
| `plot_training_history()` | Training History Curve |
| `plot_burgers_evolution()` | Burgers' Equation Time Evolution |
| `plot_residuals()` | Physics Residual Distribution |
| `save_animation_frames()` | Save Animation Frames |

```python
from utils import plot_2d_comparison, plot_training_history

# 2D Solution Comparison
plot_2d_comparison(X, Y, u_pred, u_exact, title="Poisson Solution")

# Training History
plot_training_history(history, metrics=['loss', 'l2_error'])
```

---

## 🎯 Complete Examples

### Example 1: Solving Burgers' Equation with PINN

```python
import torch
from utils import (
    create_pinn, generate_burgers_data, 
    create_boundary_conditions, create_initial_conditions,
    PINNTrainer, compute_pde_residual,
    plot_2d_comparison, relative_l2_error
)

# 1. Prepare Data
x, t, u_exact = generate_burgers_data(n_x=256, n_t=100, nu=0.01)

# 2. Create Model
model = create_pinn(input_dim=2, output_dim=1, hidden_dims=[64, 64, 64, 64])

# 3. Define PDE Residual
def burgers_residual(coords, u):
    return compute_pde_residual(coords, u, pde_type='burgers', nu=0.01)

# 4. Prepare Boundary and Initial Conditions
bc_data = create_boundary_conditions(domain=[(-1, 1), (0, 1)], n_points=100)
ic_data = create_initial_conditions(domain=[(-1, 1)], n_points=100, 
                                    ic_function=lambda x: -np.sin(np.pi * x))

# 5. Train
trainer = PINNTrainer(model, pde_loss_fn=burgers_residual, 
                      bc_data=bc_data, ic_data=ic_data)
history = trainer.train(epochs=10000, lr=1e-3)

# 6. Evaluate
u_pred = model(test_coords)
print(f"Relative L2 error: {relative_l2_error(u_pred, u_exact):.4e}")
```

### Example 2: Solving Darcy Flow with FNO

```python
from utils import (
    create_fno, generate_fno_data, create_fno_dataloader,
    FNOTrainer, relative_l2_error
)

# 1. Generate Data
train_data, test_data = generate_fno_data(pde_type='darcy', n_samples=1000)
train_loader = create_fno_dataloader(train_data, batch_size=20)

# 2. Create FNO
fno = create_fno(modes=12, width=32, dim=2)

# 3. Train
trainer = FNOTrainer(fno)
history = trainer.train(train_loader, epochs=500)

# 4. Evaluate
with torch.no_grad():
    pred = fno(test_data['input'])
print(f"Test L2 error: {relative_l2_error(pred, test_data['output']):.4e}")
```

### Example 3: Learning Operators with DeepONet

```python
from utils import (
    create_deeponet, generate_operator_data, create_deeponet_dataloader,
    DeepONetTrainer
)

# 1. Generate Operator Data (Learn Antiderivative)
u_sensors, y_query, G_u = generate_operator_data(
    operator_type='antiderivative', n_samples=1000
)

# 2. Create DeepONet
deeponet = create_deeponet(
    branch_input_dim=100, trunk_input_dim=1,
    hidden_dim=100, p=50
)

# 3. Train
loader = create_deeponet_dataloader(u_sensors, y_query, G_u)
trainer = DeepONetTrainer(deeponet)
history = trainer.train(loader, epochs=1000)
```

---

## 📖 API Cheat Sheet

### Quick Creation Functions

```python
model = create_pinn(input_dim, output_dim, hidden_dims, activation, use_fourier, use_adaptive_weights)
model = create_deeponet(branch_input_dim, trunk_input_dim, hidden_dim, p, branch_layers, trunk_layers)
model = create_fno(in_channels, out_channels, modes, width, dim, depth)
model = create_tnn(input_dim, output_dim, rank, layers_per_dim, hidden_dim, use_tucker)
model = create_pde_transformer(input_dim, output_dim, d_model, nhead, num_layers, dim_feedforward)
```

### Universal Training Function

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

## 📦 Import Methods

```python
# Method 1: Full Import
from utils import *

# Method 2: Selective Import
from utils import MLP, PINN, FNO2d, DeepONet
from utils import generate_burgers_data, create_boundary_conditions
from utils import PINNTrainer, relative_l2_error

# Method 3: Using Quick Creation Functions
from utils import create_pinn, create_fno, train_model
```

---

## 📝 Version History

- **v2.0.0** (2024-12): Major Update
  - Added `nn_blocks.py`: Complete neural network module library
  - Added `training.py`: Loss functions, PDE residuals, optimizer tools
  - Added `trainers.py`: Specialized trainers for each method
  - Updated `data_utils.py`: Added Navier-Stokes, Wave equation data generation
  - Updated `__init__.py`: Unified interface + quick creation functions

- **v1.0.0** (2024-01): Initial Version
  - Basic data tools, metrics, visualization
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
