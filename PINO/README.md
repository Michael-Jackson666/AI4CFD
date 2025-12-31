# Physics-Informed Neural Operators (PINO)

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.10+-ee4c2c.svg)](https://pytorch.org/)

Physics-Informed Neural Operators (PINO) 是一种结合了**物理信息神经网络 (PINNs)** 和**神经算子 (Neural Operators)** 优点的深度学习方法。PINO 既可以学习参数化 PDE 的解算子，又在训练过程中融入物理约束，从而减少对标签数据的依赖。

## 🧠 核心思想

PINO 的核心是将两种范式相结合：

### 1. 神经算子 (Neural Operator)
学习从输入函数空间到输出函数空间的映射：

$$
\mathcal{G}_\theta: \mathcal{A} \to \mathcal{U}, \quad a \mapsto u
$$

其中 $a$ 是输入函数（如初始条件、边界条件、PDE 系数），$u$ 是 PDE 的解。

### 2. 物理信息约束 (Physics-Informed)
在训练损失中加入 PDE 残差项：

$$
\mathcal{L} = \underbrace{\mathcal{L}_{data}}_{\text{数据损失}} + \lambda \underbrace{\mathcal{L}_{PDE}}_{\text{物理残差}}
$$

其中物理残差为：

$$
\mathcal{L}_{PDE} = \frac{1}{N}\sum_{i=1}^{N} \|\mathcal{N}[\mathcal{G}_\theta(a)](x_i, t_i)\|^2
$$

$\mathcal{N}[\cdot]$ 表示 PDE 算子（如 $\frac{\partial u}{\partial t} + u\frac{\partial u}{\partial x} - \nu\frac{\partial^2 u}{\partial x^2}$ 对于 Burgers 方程）。

## 📊 PINO vs PINNs vs FNO

| 特性 | PINNs | FNO | PINO |
|------|-------|-----|------|
| **学习目标** | 单个 PDE 解 | 参数化 PDE 解算子 | 参数化 PDE 解算子 |
| **数据需求** | 低（物理约束） | 高（需要大量解） | 中（物理约束+少量数据） |
| **推理速度** | 慢（需重新训练） | 快（单次前向传播） | 快（单次前向传播） |
| **泛化能力** | 低（单个问题） | 高（多参数泛化） | 高（多参数泛化） |
| **物理一致性** | 高（显式约束） | 中（隐式学习） | 高（显式约束） |

## 📁 目录结构

```
PINO/
├── README.md                      # 本文档
├── models.py                      # PINO 神经网络架构
├── pino_core.py                   # PINO 核心算法实现
├── train.py                       # 通用训练脚本
│
├── tutorial/                      # 📚 教程
│   ├── README.md                  # 教程目录说明
│   ├── pino_overview.ipynb        # ⭐ PINO 概述教程（入门必读）
│   ├── burgers_pino.ipynb         # Burgers 方程 PINO 求解
│   ├── darcy_pino.ipynb           # Darcy 流 PINO 求解
│   └── ns_pino.ipynb              # Navier-Stokes PINO 求解
│
└── examples/                      # 💡 示例代码
    ├── README.md                  # 示例说明
    ├── burgers_1d.py              # 1D Burgers 方程示例
    ├── darcy_2d.py                # 2D Darcy 流示例
    └── heat_1d.py                 # 1D 热传导方程示例
```

## 🚀 快速开始

### 安装依赖

```bash
# 基础依赖
pip install torch numpy scipy matplotlib jupyter

# 可选：GPU 加速（CUDA 11.8）
pip install torch --index-url https://download.pytorch.org/whl/cu118
```

### 运行示例

```bash
cd PINO/examples
python burgers_1d.py
```

### 启动教程

```bash
jupyter notebook PINO/tutorial/pino_overview.ipynb
```

## 🎯 典型应用场景

1. **参数化 PDE 求解**：不同初始条件/边界条件/PDE 系数的快速求解
2. **数据稀疏场景**：有限标签数据 + 物理约束
3. **实时预测**：训练完成后毫秒级推理
4. **逆问题**：从观测数据反演 PDE 参数

## 📖 进一步阅读

1. Li, Z., et al. (2021). "Physics-Informed Neural Operator for Learning Partial Differential Equations." arXiv:2111.03794
2. Raissi, M., et al. (2019). "Physics-informed neural networks." Journal of Computational Physics
3. Li, Z., et al. (2020). "Fourier Neural Operator for Parametric Partial Differential Equations." arXiv:2010.08895