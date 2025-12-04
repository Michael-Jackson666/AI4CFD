# TNN 示例集

本目录包含使用张量神经网络（TNN）求解各类高维 PDE 问题的示例代码。

## 示例列表

| 示例 | 问题类型 | 边界条件 | 维度 |
|------|----------|----------|------|
| ex_5_1 | Poisson 方程 | 齐次 Dirichlet | 5D, 10D, 20D |
| ex_5_2 | Poisson 方程 | 非齐次 Dirichlet | 5D, 10D, 20D |
| ex_5_3 | Poisson 方程 | Neumann | 5D, 10D, 20D |
| ex_5_4 | 特征值问题 | Dirichlet | 5D, 10D, 20D |
| ex_5_5 | 无界域问题 | 衰减条件 | 5D, 10D, 20D |

## 示例详情

### ex_5_1：齐次 Dirichlet 边界条件

$$
\begin{cases}
-\Delta u = f & \text{in } \Omega = [-1,1]^d \\
u = 0 & \text{on } \partial\Omega
\end{cases}
$$

通过边界函数 $\text{bd}(x) = \prod_k (x_k - a)(b - x_k)$ 自动满足边界条件。

### ex_5_2：非齐次 Dirichlet 边界条件

$$
\begin{cases}
-\Delta u = f & \text{in } \Omega \\
u = g & \text{on } \partial\Omega
\end{cases}
$$

使用双 TNN 分解：$u = u_0 + \text{bd}(x) \cdot u_1$，其中 $u_0$ 满足边界条件。

### ex_5_3：Neumann 边界条件

$$
\begin{cases}
-\Delta u = f & \text{in } \Omega \\
\frac{\partial u}{\partial n} = g & \text{on } \partial\Omega
\end{cases}
$$

采用弱形式处理，边界积分项显式计算。

### ex_5_4：特征值问题

$$
\begin{cases}
-\Delta u = \lambda u & \text{in } \Omega \\
u = 0 & \text{on } \partial\Omega
\end{cases}
$$

求解 Laplace 算子的特征值和特征函数。

### ex_5_5：无界域

$$
\begin{cases}
-\Delta u + u = f & \text{in } \mathbb{R}^d \\
u \to 0 & \text{as } |x| \to \infty
\end{cases}
$$

使用 **Hermite-Gauss 积分**处理无界域，适用于衰减边界条件。

## 目录结构

```
example/
├── README.md
├── ex_5_1/           # 齐次 Dirichlet
│   ├── dim5/
│   ├── dim10/
│   └── dim20/
├── ex_5_2/           # 非齐次 Dirichlet
│   ├── dim5/
│   ├── dim10/
│   └── dim20/
├── ex_5_3/           # Neumann
│   ├── dim5/
│   ├── dim10/
│   └── dim20/
├── ex_5_4/           # 特征值问题
│   ├── dim5/
│   ├── dim10/
│   └── dim20/
└── ex_5_5/           # 无界域
    ├── dim5/
    ├── dim10/
    └── dim20/
```

## 运行示例

```bash
# 运行 5D Poisson 方程（齐次 Dirichlet）
cd ex_5_1/dim5
python ex_5_1_dim5.py

# 运行 10D 特征值问题
cd ex_5_4/dim10
python ex_5_4_dim10.py
```

## 共用模块

每个示例目录下包含以下核心文件：
- `quadrature.py`：Gauss-Legendre / Hermite-Gauss 积分
- `tnn.py`：TNN 模型定义
- `integration.py`：张量积分函数

## 预期结果

| 问题 | 维度 | L² 误差 | 训练时间 |
|------|------|---------|----------|
| ex_5_1 | 5D | ~10⁻⁵ | ~30 min |
| ex_5_1 | 10D | ~10⁻⁴ | ~1 hour |
| ex_5_1 | 20D | ~10⁻³ | ~2 hours |

*注：时间基于 CUDA GPU 估计，CPU 会更慢。*
