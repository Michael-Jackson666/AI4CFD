# ex_5_3：Neumann 边界条件

## 问题描述

求解高维 Poisson 方程，带 Neumann 边界条件：

$$
\begin{cases}
-\Delta u = f & \text{in } \Omega = [0,1]^d \\
\frac{\partial u}{\partial n} = g & \text{on } \partial\Omega
\end{cases}
$$

本例中：

$$
f(x) = d\pi^2 F(x), \quad F(x) = \sum_{i=1}^{d} \sin(\pi x_i)
$$

精确解：$u(x) = F(x) = \sum_{i=1}^{d} \sin(\pi x_i)$

## 边界条件处理

Neumann 边界条件通过**弱形式**处理，边界积分项显式计算：

$$
\int_\Omega \nabla u \cdot \nabla v \, dx = \int_\Omega f v \, dx + \int_{\partial\Omega} g v \, ds
$$

与 Dirichlet 问题不同，这里**不使用边界函数** `bd=None`。

## 目录结构

```
ex_5_3/
├── README.md
├── dim5/           # 5 维问题
│   ├── quadrature.py
│   ├── tnn.py
│   ├── integration.py
│   └── ex_5_3_dim5.py
├── dim10/          # 10 维问题
│   └── ...
└── dim20/          # 20 维问题
    └── ...
```

## 参数配置

| 参数 | 值 |
|------|------|
| 计算域 | $[0,1]^d$ |
| 张量秩 $p$ | 100 |
| 积分点数 $N$ | 1600 (16×100) |
| 网络结构 | [1,100,100,100,100] |

## 运行方法

```bash
# 5 维
cd dim5
python ex_5_3_dim5.py

# 10 维
cd dim10
python ex_5_3_dim10.py

# 20 维
cd dim20
python ex_5_3_dim20.py
```

## 关键技术

### 边界梯度计算

在边界 $x_i = 0$ 和 $x_i = 1$ 处计算法向导数：

```python
grad_F0[i,i,:] = -π   # x_i = 0 边界，法向向内
grad_F1[i,i,:] = -π   # x_i = 1 边界，法向向外
```

### 弱形式积分

损失函数包含边界积分项，需要在每个维度的边界面上积分：
- $d$ 个维度 × 2 个边界面 = $2d$ 个边界积分

## 预期结果

| 维度 | L² 误差 | 训练时间 (GPU) |
|------|---------|----------------|
| 5D | ~10⁻⁴ | ~25 min |
| 10D | ~10⁻³ | ~50 min |
| 20D | ~10⁻² | ~2 hours |

## 注意事项

- Neumann 问题的解不唯一（差一个常数），通常需要附加条件如 $\int_\Omega u \, dx = 0$
- 本例的精确解满足特定的归一化条件
