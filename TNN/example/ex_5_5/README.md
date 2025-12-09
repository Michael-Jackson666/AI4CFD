# ex_5_5：无界域问题

## 问题描述

求解无界域上的 Schrödinger 特征值问题（量子谐振子）：

$$
\begin{cases}
-\Delta u + V(x) u = \lambda u & \text{in } \mathbb{R}^d \\
u \to 0 & \text{as } |x| \to \infty
\end{cases}
$$

势能函数：

$$
V(x) = \sum_{i=1}^{d} x_i^2
$$

## 理论解

量子谐振子的基态能量和波函数：

$$
\lambda_1 = d, \quad u_1(x) = \exp\left(-\frac{1}{2}\sum_{i=1}^d x_i^2\right)
$$

| 维度 | 理论基态能量 $\lambda_1$ |
|------|--------------------------|
| 5D | 5 |
| 10D | 10 |
| 20D | 20 |

## 数值积分

使用 **Hermite-Gauss 积分**处理无界域：

$$
\int_{-\infty}^{+\infty} f(x) e^{-x^2} dx \approx \sum_{i=1}^{N} w_i f(x_i)
$$

Hermite 积分点和权重自然适应高斯衰减函数。

## 目录结构

```
ex_5_5/
├── README.md
├── dim5/           # 5 维问题
│   ├── quadrature.py
│   ├── tnn.py
│   ├── integration.py
│   └── ex_5_5_dim5.py
├── dim10/          # 10 维问题
│   └── ...
└── dim20/          # 20 维问题
    └── ...
```

## 参数配置

| 参数 | 值 |
|------|------|
| 计算域 | $\mathbb{R}^d$ |
| 张量秩 $p$ | 50 |
| Hermite 积分点 $N$ | 200 |
| 网络结构 | [1,100,100,100,50] |
| 激活函数 | sin(x) |
| 边界函数 | 无（自然衰减） |

## 运行方法

```bash
# 5 维
cd dim5
python ex_5_5_dim5.py

# 10 维
cd dim10
python ex_5_5_dim10.py

# 20 维
cd dim20
python ex_5_5_dim20.py
```

## 关键技术

### Hermite-Gauss 积分

```python
z, w = Hermite_Gauss_Quad(200, device=device, dtype=dtype, modified=False)
```

### 梯度修正

由于 Hermite 积分包含 $e^{-x^2}$ 权重，梯度需要修正：

```python
grad_phi = grad_phi - z * phi
```

### Rayleigh-Ritz 方法

求解广义特征值问题 $A C = \lambda M C$：
- $A$：刚度矩阵 + 势能矩阵
- $M$：质量矩阵

```python
A = part1 + part2  # H^1 内积 + 势能项
M = part0          # L^2 内积
# Cholesky 分解求解广义特征值
L = torch.linalg.cholesky(M)
...
E, U = torch.linalg.eigh(D)
lambda_min = E[torch.argmin(E)]
```

## 预期结果

| 维度 | 计算能量 | 相对误差 | 训练时间 (GPU) |
|------|----------|----------|----------------|
| 5D | ~5.00 | ~10⁻⁴ | ~20 min |
| 10D | ~10.00 | ~10⁻³ | ~40 min |
| 20D | ~20.00 | ~10⁻² | ~1.5 hours |

## 物理背景

量子谐振子是量子力学的基础模型：
- 描述粒子在二次势阱中的运动
- 基态能量 $E_0 = \frac{d}{2}\hbar\omega$（本例取 $\hbar = \omega = 1$）
- 高维谐振子在分子振动、量子场论中有重要应用
