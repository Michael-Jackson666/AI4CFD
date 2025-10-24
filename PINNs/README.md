# Physics-Informed Neural Networks (PINNs) for CFD

Physics-Informed Neural Networks (PINNs) 是一类将物理规律（偏微分方程 PDEs）直接融入神经网络训练过程的深度学习方法。本项目提供了 PINNs 在计算流体力学（CFD）中的完整实现，包括从基础教程到高级应用（如 Vlasov-Poisson 系统）的全套资源。

PINNs are a class of neural networks that incorporate physical laws (PDEs) directly into the training process by encoding PDEs as regularization terms in the loss function using automatic differentiation.

## 📦 安装 (Installation)

### Prerequisites
- Python 3.8+
- CUDA (optional, for GPU acceleration)

### Basic Installation
```bash
# Clone the repository
git clone https://github.com/your-repo/AI4CFD.git
cd AI4CFD/PINNs

# Install PyTorch (CPU version)
pip install torch torchvision torchaudio

# For GPU support, visit: https://pytorch.org/get-started/locally/
# Example for CUDA 11.8:
# pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Install other dependencies
pip install numpy scipy matplotlib jupyter
```

### DeepXDE Installation (for examples/)
```bash
pip install deepxde
```

### Verify Installation
```bash
python -c "import torch; print('PyTorch version:', torch.__version__)"
python -c "import torch; print('CUDA available:', torch.cuda.is_available())"
```

## 🎯 Key Concepts

### What are PINNs?
PINNs solve PDEs by:
1. **Neural Network Approximation**: Using a neural network to approximate the solution u(x,t)
2. **Physics Constraints**: Incorporating PDE residuals as loss terms
3. **Automatic Differentiation**: Computing derivatives needed for PDEs automatically
4. **Multi-objective Training**: Balancing data fitting and physics compliance

### Advantages
- Can solve PDEs with limited or noisy data
- Naturally handle complex geometries
- Incorporate prior physical knowledge
- Provide smooth, differentiable solutions

### Applications
- Fluid dynamics (Navier-Stokes equations)
- Heat transfer problems
- Wave propagation
- Inverse problems and parameter estimation

## 📁 目录结构 (Directory Structure)

```
PINNs/
├── README.md                      # 本文档
├── models.py                      # 神经网络架构定义
├── pde_definitions.py             # 常见 PDE 定义和残差函数
├── train.py                       # 通用训练脚本
│
├── tutorial/                      # 📚 教程笔记本
│   ├── tutorial_chinese.ipynb     # ⭐ 完整中文 PINNs 教程
│   ├── tutorial_eng.ipynb         # ⭐ Complete English PINNs Tutorial
│   ├── possion_1d.ipynb          # 1D 泊松方程教程
│   ├── heat_2d.ipynb             # 2D 热传导方程示例
│   ├── ns_basic.ipynb            # Navier-Stokes 基础教程
│   ├── ns_advanced.ipynb         # Navier-Stokes 高级应用
│   ├── system_pde.ipynb          # 耦合 PDE 系统教程
│   └── vlasov_poisson.ipynb      # Vlasov-Poisson 系统教程
│
├── examples/                      # 💡 示例代码
│   ├── possion_dirichlet_1d.py   # 使用 DeepXDE 的 1D 泊松方程
│   └── *.dat                      # 训练数据和结果
│
└── vp_system/                     # 🌌 Vlasov-Poisson 系统专题
    ├── README.md                  # VP 系统详细文档
    ├── main.py                    # 主训练脚本
    ├── vp_pinn.py                 # VP-PINNs 实现
    ├── mlp.py                     # 多层感知机模型
    ├── transformer.py             # Transformer 模型
    ├── visualization.py           # 可视化工具
    ├── compare_models.py          # 模型对比脚本
    ├── comparison/                # 模型对比结果
    ├── beihang_papper/            # 北航论文相关代码
    ├── true_test/                 # 真实测试数据
    └── 2025/                      # 2025年最新研究成果
```

## � 核心文件说明 (Core Files Description)

### `train.py`
通用训练脚本，支持多种 PDE 问题的求解。

**功能**:
- 支持多种 PDE 类型（Poisson、Heat、Burgers、Navier-Stokes 等）
- 灵活的命令行参数配置
- 自动保存训练历史和模型检查点
- 集成 TensorBoard 可视化

**使用方法**:
```bash
python train.py --problem poisson --epochs 10000 --lr 0.001 --layers 4 --neurons 50
```

### `models.py`
神经网络架构定义。

**包含的模型**:
- **FullyConnectedNN**: 标准全连接网络（tanh 激活）
- **ResidualNN**: 残差连接网络（提高训练稳定性）
- **FourierNet**: 傅里叶特征网络（高频信息捕获）
- **MultiScaleNet**: 多尺度网络（适用于多尺度问题）

### `pde_definitions.py`
常见 PDE 的定义和残差函数计算。

**支持的方程**:
- **Poisson 方程**: $\nabla^2 u = f$
- **热传导方程**: $\frac{\partial u}{\partial t} = \alpha \nabla^2 u$
- **Burgers 方程**: $\frac{\partial u}{\partial t} + u\frac{\partial u}{\partial x} = \nu \frac{\partial^2 u}{\partial x^2}$
- **Navier-Stokes 方程**: 不可压缩流体运动方程
- **Vlasov-Poisson 系统**: 等离子体物理方程组

## 📚 教程笔记本详解 (Tutorial Notebooks)

### 入门教程 (Beginner Level)

#### `tutorial_chinese.ipynb` ⭐
**完整的中文 PINNs 教程**
- PINNs 基本概念和原理
- PyTorch 自动微分基础
- 从零开始实现简单的 1D PDE 求解器
- 损失函数设计和训练策略
- 结果可视化和误差分析

#### `tutorial_eng.ipynb` ⭐
**Complete English PINNs Tutorial**
- Fundamental concepts of PINNs
- PyTorch automatic differentiation basics
- Step-by-step implementation of a simple 1D PDE solver
- Loss function design and training strategies
- Visualization and error analysis

### 基础 PDE 教程 (Basic PDEs)

#### `possion_1d.ipynb`
**1D 泊松方程**: $-\frac{d^2u}{dx^2} = f(x)$
- 边界条件处理（Dirichlet/Neumann）
- 源项处理
- 精确解对比

#### `heat_2d.ipynb`
**2D 热传导方程**: $\frac{\partial u}{\partial t} = \alpha \left(\frac{\partial^2 u}{\partial x^2} + \frac{\partial^2 u}{\partial y^2}\right)$
- 初始条件设置
- 时空域采样策略
- 瞬态和稳态求解

### 高级流体动力学 (Advanced Fluid Dynamics)

#### `ns_basic.ipynb`
**Navier-Stokes 基础教程**
- 2D 方腔驱动流（Lid-driven cavity flow）
- 速度-压力耦合处理
- 不可压缩条件 $\nabla \cdot \mathbf{u} = 0$

#### `ns_advanced.ipynb`
**Navier-Stokes 高级应用**
- 圆柱绕流（Flow past a cylinder）
- 卡门涡街（Karman vortex street）
- 雷诺数效应分析

### 复杂系统 (Complex Systems)

#### `system_pde.ipynb`
**耦合 PDE 系统教程**
- 多物理场耦合
- 反应-扩散系统
- 多变量网络设计

#### `vlasov_poisson.ipynb`
**Vlasov-Poisson 系统教程**
- 等离子体动理学方程
- 电场自洽求解
- 双流不稳定性（Two-stream instability）仿真
- 相空间演化分析

## �🚀 快速开始 (Quick Start)

### Running the Tutorials
```bash
# Chinese tutorial (recommended for beginners)
jupyter notebook tutorial/tutorial_chinese.ipynb

# English tutorial
jupyter notebook tutorial/tutorial_eng.ipynb

# Specific PDE examples
jupyter notebook tutorial/possion_1d.ipynb
jupyter notebook tutorial/heat_2d.ipynb
```

### Running DeepXDE Example
```bash
cd examples/
python possion_dirichlet_1d.py
```

**说明**: 该示例使用 DeepXDE 库求解带 Dirichlet 边界条件的 1D Poisson 方程。训练完成后会生成以下文件:
- `loss.dat`: 训练损失历史
- `train.dat`: 训练点数据
- `test.dat`: 测试点预测结果

### Training Custom Models
```bash
python train.py --problem poisson --epochs 10000 --lr 0.001
```

### Available Problems
- `poisson` - 1D/2D Poisson equation
- `heat` - Heat equation (1D/2D)
- `burgers` - Burgers' equation
- `navier_stokes` - Navier-Stokes equations
- `vlasov_poisson` - Vlasov-Poisson system

### 🌌 Vlasov-Poisson System (vp_system/)

For advanced Vlasov-Poisson plasma physics applications, we provide a dedicated research-grade implementation:

```bash
cd vp_system/
python main.py  # Train VP-PINNs model
```

**Features**:
- **Multiple Architectures**: MLP and Transformer-based models
- **Model Comparison**: Automated performance benchmarking (`compare_models.py`)
- **Visualization Suite**: Comprehensive plotting tools (`visualization.py`)
- **Research Materials**: Includes code from published research papers (`beihang_papper/`)
- **Validation Tests**: True solution comparisons (`true_test/`)

**Key Files**:
- `vp_pinn.py`: Core PINNs implementation for Vlasov-Poisson equations
- `mlp.py`: Multi-layer perceptron architecture optimized for VP systems
- `transformer.py`: Attention-based architecture for capturing long-range dependencies
- `compare_models.py`: Benchmark different model architectures
- `visualization.py`: Phase space, electric field, and distribution function plots

See `vp_system/README.md` for detailed documentation on the VP-PINNs implementation.

## � 示例代码说明 (Examples Directory)

`examples/` 目录包含使用 DeepXDE 库的独立示例：

### `possion_dirichlet_1d.py`
使用 DeepXDE 求解 1D Poisson 方程：

$$-\frac{d^2u}{dx^2} = -\pi^2 \sin(\pi x), \quad x \in [0,1]$$

**边界条件**: $u(0) = u(1) = 0$

**精确解**: $u(x) = \sin(\pi x)$

**特点**:
- 使用 DeepXDE 的高级 API 简化代码
- 自动处理采样和训练
- 生成损失曲线和预测结果对比图

**输出文件**:
- `loss.dat`: 训练过程中的损失值记录
- `train.dat`: 训练数据点
- `test.dat`: 测试点的预测结果

## �📊 Example Results

The tutorials demonstrate solving:

### Basic PDEs
1. **1D Poisson Equation**: ∂²u/∂x² = f(x)
2. **2D Heat Equation**: ∂u/∂t = α∇²u
3. **1D Burgers' Equation**: ∂u/∂t + u∂u/∂x = ν∂²u/∂x²

### Advanced Applications
4. **2D Navier-Stokes**: Fluid flow simulations
5. **Vlasov-Poisson System**: Plasma physics applications
6. **System of PDEs**: Coupled equation systems

### Implementation Approaches
- **Pure PyTorch**: From-scratch implementations with manual gradient computation
- **DeepXDE**: Using the DeepXDE library for rapid prototyping
- **Comparative Analysis**: Performance comparison between different approaches

## 🔧 Implementation Details

### Network Architecture
- Fully connected neural networks with tanh activations
- Input: spatial/temporal coordinates
- Output: solution values

### Loss Function
```
L = L_data + λ_pde * L_pde + λ_bc * L_bc
```

Where:
- `L_data`: Data fitting loss (if training data available)
- `L_pde`: PDE residual loss
- `L_bc`: Boundary condition loss

### Training Strategy
1. Initialize network with Xavier/He initialization
2. Sample collocation points in domain
3. Compute PDE residuals using automatic differentiation
4. Optimize combined loss using L-BFGS or Adam

## 📚 Mathematical Background

For a general PDE:
```
F(x, u, ∂u/∂x, ∂²u/∂x², ...) = 0
```

PINNs minimize:
```
min Σ |F(x_i, u_θ(x_i), ∇u_θ(x_i), ...)|² + BC_loss + Data_loss
```

The neural network u_θ(x) with parameters θ learns to satisfy both the PDE and boundary conditions.

## 🔗 References

1. Raissi, M., Perdikaris, P., & Karniadakis, G. E. (2019). Physics-informed neural networks: A deep learning framework for solving forward and inverse problems involving nonlinear partial differential equations. Journal of Computational Physics, 378, 686-707.

2. Karniadakis, G. E., Kevrekidis, I. G., Lu, L., Perdikaris, P., Wang, S., & Yang, L. (2021). Physics-informed machine learning. Nature Reviews Physics, 3(6), 422-440.

3. Lu, L., Meng, X., Mao, Z., & Karniadakis, G. E. (2021). DeepXDE: A deep learning library for solving differential equations. SIAM review, 63(1), 208-228.

## 🌟 Features

### Comprehensive Tutorials
- **Bilingual Support**: Both Chinese and English tutorials available
- **Progressive Learning**: From basic concepts to advanced applications
- **Interactive Examples**: Jupyter notebooks with step-by-step explanations
- **Visualization Tools**: Comprehensive plotting and analysis functions

### Multiple Implementation Styles
- **Educational**: Pure PyTorch implementations for learning
- **Production**: DeepXDE-based examples for practical applications
- **Comparative**: Side-by-side performance analysis

### Advanced Topics Covered
- Multi-scale neural networks
- Adaptive weighting strategies
- Uncertainty quantification
- Inverse problem solving
- Parameter identification