# PINO 教程目录

本目录包含 Physics-Informed Neural Operators (PINO) 的学习教程。

## 📚 教程列表

| 教程 | 描述 | 难度 |
|------|------|------|
| [pino_overview.ipynb](pino_overview.ipynb) | **PINO 概述教程** - 核心概念与原理介绍 | ⭐ 入门 |
| [burgers_pino.ipynb](burgers_pino.ipynb) | 1D Burgers 方程 PINO 实战 | ⭐⭐ 基础 |
| [darcy_pino.ipynb](darcy_pino.ipynb) | 2D Darcy 流动 PINO 求解 | ⭐⭐⭐ 进阶 |
| [ns_pino.ipynb](ns_pino.ipynb) | Navier-Stokes 方程 PINO 应用 | ⭐⭐⭐⭐ 高级 |

## 🎯 学习路径

```
1. PINO 概述教程 (pino_overview.ipynb)
   ├── 了解 PINO 基本概念
   ├── 理解算子学习思想
   └── 掌握 FNO + Physics 结合方式
                │
                ▼
2. Burgers 方程实战 (burgers_pino.ipynb)
   ├── 1D 问题入门
   ├── 数据生成与预处理
   └── 模型训练与评估
                │
                ▼
3. Darcy 流动求解 (darcy_pino.ipynb)
   ├── 2D 问题扩展
   ├── 椭圆型 PDE 处理
   └── 边界条件施加
                │
                ▼
4. Navier-Stokes 应用 (ns_pino.ipynb)
   ├── 复杂流体力学问题
   ├── 多物理场耦合
   └── 高级训练技巧
```

## 📋 前置知识

在开始学习之前，建议您具备以下基础：

1. **数学基础**
   - 偏微分方程 (PDE) 基本概念
   - Fourier 变换与谱方法
   - 数值分析基础

2. **深度学习基础**
   - PyTorch 框架使用
   - 神经网络训练流程
   - 损失函数设计

3. **相关方法了解**
   - PINNs (物理信息神经网络)
   - FNO (Fourier Neural Operator)
   - 算子学习基本思想

## 💻 环境配置

```bash
# 创建虚拟环境
conda create -n pino python=3.9
conda activate pino

# 安装依赖
pip install torch>=1.10
pip install numpy scipy matplotlib
pip install tqdm jupyter
```

## 📖 推荐阅读顺序

### 初学者

如果您是 AI4PDE 的新手：

1. 先阅读 [PINO 概述教程](pino_overview.ipynb) 了解整体框架
2. 运行 [Burgers 教程](burgers_pino.ipynb) 进行实践
3. 对比 PINNs 和 FNO 的差异

### 进阶学习

如果您已有 PINNs 或 FNO 经验：

1. 快速浏览概述教程的核心公式
2. 直接进入 [Darcy 教程](darcy_pino.ipynb) 学习 2D 问题
3. 尝试 [NS 教程](ns_pino.ipynb) 挑战复杂问题

## 🔬 核心概念预览

### PINO 损失函数

$$\mathcal{L}_{PINO} = \mathcal{L}_{data} + \lambda \mathcal{L}_{PDE}$$

其中：
- $\mathcal{L}_{data}$: 数据拟合损失
- $\mathcal{L}_{PDE}$: 物理残差损失
- $\lambda$: 物理约束权重

### PINO vs PINNs vs FNO

| 特性 | PINNs | FNO | PINO |
|------|-------|-----|------|
| 学习目标 | 单一解 | 解算子 | 解算子 |
| 物理约束 | ✅ 强 | ❌ 无 | ✅ 有 |
| 泛化能力 | 弱 | 强 | 强 |
| 数据需求 | 低 | 高 | 中 |

## 📁 目录结构

```
tutorial/
├── README.md           # 本文件
├── pino_overview.ipynb # 概述教程 (必读)
├── burgers_pino.ipynb  # Burgers 方程教程
├── darcy_pino.ipynb    # Darcy 流动教程
└── ns_pino.ipynb       # Navier-Stokes 教程
```

## 🔗 参考资源

1. **论文**
   - Li et al., "Physics-Informed Neural Operator for Learning Partial Differential Equations"
   - Li et al., "Fourier Neural Operator for Parametric Partial Differential Equations"

2. **代码仓库**
   - [neuraloperator](https://github.com/neuraloperator/neuraloperator)
   - [physics_informed_neural_operator](https://github.com/zongyi-li/physics_informed_neural_operator)

3. **相关教程**
   - 本项目 FNO 教程: `../FNO/tutorial/`
   - 本项目 PINNs 教程: `../PINNs/tutorial/`

---

> 💡 **提示**: 建议按照学习路径顺序学习，每个教程都包含理论讲解和代码实践两部分。
