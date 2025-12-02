# 张量神经网络 (TNN)

## 简介

**张量神经网络（Tensor Neural Network, TNN）** 是一种专门用于求解高维偏微分方程（PDE）的神经网络架构。通过张量分解技术，TNN 有效克服了传统方法面临的**维数灾难**问题。

## 核心思想

TNN 将高维函数表示为一维函数的乘积和：

$$
u(x_1, x_2, \ldots, x_d) = \sum_{i=1}^{r} C_i \prod_{k=1}^{d} \phi_k^{(i)}(x_k)
$$

其中：
- $r$：张量秩（分解项数）
- $\phi_k^{(i)}$：第 $k$ 个维度的一维神经网络
- $C_i$：组合系数

**参数量对比**：
| 方法 | 参数量 | 5维示例 (N=100) |
|------|--------|-----------------|
| 传统网格 | $O(N^d)$ | $10^{10}$ |
| TNN | $O(d \cdot r \cdot n)$ | $\sim 25,000$ |

## 求解 PDE 的方法

### 1. 变分形式

以 Poisson 方程为例：$-\Delta u = f$

将 PDE 转化为能量泛函最小化问题：

$$
\mathcal{E}(u) = \frac{1}{2} \int_\Omega |\nabla u|^2 \, dx - \int_\Omega f \cdot u \, dx
$$

### 2. 数值积分

使用 **Gauss-Legendre 积分**离散化：
- 每个维度独立生成积分点和权重
- 利用张量积结构，将 $d$ 维积分分解为 $d$ 个一维积分的乘积

### 3. 线性系统求解

将 TNN 展开代入变分形式，得到线性系统 $AC = B$：
- $A$：刚度矩阵（$H^1$ 内积）
- $B$：载荷向量（$L^2$ 内积）
- $C$：待求系数

### 4. 优化训练

两阶段优化策略：
1. **Adam**：快速探索参数空间
2. **L-BFGS**：精细优化至高精度

### 5. 边界条件

通过乘以边界函数自动满足：

$$
u_{\text{TNN}} = \text{bd}(x) \cdot \sum_i C_i \prod_k \phi_k^{(i)}(x_k)
$$

其中 $\text{bd}(x) = \prod_k (x_k - a)(b - x_k)$ 在边界自动为零。

## 目录结构

```
TNN/
├── README.md          # 本文件
├── tutorial/          # 入门教程（2D Poisson 方程）
├── train/             # 训练脚本
│   └── dim5/          # 5维示例
│       ├── quadrature.py    # 高斯积分
│       ├── tnn.py           # TNN 模型
│       ├── integration.py   # 张量积分
│       └── ex_5_1_dim5.py   # 主训练程序
└── example/           # 更多示例
```

## 快速开始

### 入门教程
```bash
cd tutorial
jupyter notebook TNN_tutorial.ipynb
```

### 运行 5D 示例
```bash
cd train/dim5
python ex_5_1_dim5.py
```

## 适用问题

- ✅ 高维 Poisson 方程
- ✅ 椭圆型 PDE
- ✅ 特征值问题
- ✅ Dirichlet / Neumann 边界条件
- ✅ 有界域 / 无界域

## 参考文献

TNN 方法基于张量分解理论，结合深度学习技术求解高维 PDE。更多理论细节请参阅相关论文。

## 许可证

见项目根目录 LICENSE 文件。
