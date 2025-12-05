# ex_5_1：齐次 Dirichlet 边界条件

## 问题描述

求解高维 Poisson 方程，带齐次 Dirichlet 边界条件：

$$
\begin{cases}
-\Delta u = f & \text{in } \Omega = [-1,1]^d \\
u = 0 & \text{on } \partial\Omega
\end{cases}
$$

右端项：

$$
f(x_1, \ldots, x_d) = \sum_{k=1}^{d} \sin(2\pi x_k) \prod_{i \neq k} \sin(\pi x_i)
$$

## 边界条件处理

通过乘以边界函数自动满足齐次 Dirichlet 条件：

$$
\text{bd}(x_k) = (x_k - a)(b - x_k)
$$

在 $x_k = a$ 或 $x_k = b$ 处自动为零。

## 目录结构

```
ex_5_1/
├── README.md
├── dim5/           # 5 维问题
│   ├── quadrature.py
│   ├── tnn.py
│   ├── integration.py
│   └── ex_5_1_dim5.py
├── dim10/          # 10 维问题
│   └── ...
└── dim20/          # 20 维问题
    └── ...
```

## 参数配置

| 参数 | dim5 | dim10 | dim20 |
|------|------|-------|-------|
| 张量秩 $p$ | 50 | 50 | 50 |
| 积分点数 $N$ | 3200 | 3200 | 3200 |
| 网络结构 | [1,100,100,100,50] | 同左 | 同左 |
| Adam epochs | 50,000 | 50,000 | 50,000 |
| L-BFGS epochs | 10,000 | 10,000 | 10,000 |

## 运行方法

```bash
# 5 维
cd dim5
python ex_5_1_dim5.py

# 10 维
cd dim10
python ex_5_1_dim10.py

# 20 维
cd dim20
python ex_5_1_dim20.py
```

## 预期结果

| 维度 | L² 误差 | H¹ 误差 | 训练时间 (GPU) |
|------|---------|---------|----------------|
| 5D | ~10⁻⁵ | ~10⁻⁴ | ~30 min |
| 10D | ~10⁻⁴ | ~10⁻³ | ~1 hour |
| 20D | ~10⁻³ | ~10⁻² | ~2 hours |

## 输出

- 终端打印：每轮 loss、L² 误差、H¹ 误差
- 可视化：2D 切片对比图（如有）
