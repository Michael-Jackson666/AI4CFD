# ex_5_4：特征值问题

## 问题描述

求解高维 Laplace 特征值问题：

$$
\begin{cases}
-\Delta u = \lambda u & \text{in } \Omega = [0,1]^d \\
u = 0 & \text{on } \partial\Omega
\end{cases}
$$

目标是找到最小特征值 $\lambda_1$ 及其对应的特征函数 $u_1$。

## 理论解

对于 $[0,1]^d$ 上的齐次 Dirichlet 问题，第一特征值和特征函数为：

$$
\lambda_1 = d\pi^2, \quad u_1(x) = \prod_{i=1}^d \sin(\pi x_i)
$$

| 维度 | 理论特征值 $\lambda_1$ |
|------|------------------------|
| 5D | $5\pi^2 \approx 49.35$ |
| 10D | $10\pi^2 \approx 98.70$ |
| 20D | $20\pi^2 \approx 197.39$ |

## 变分形式

利用 Rayleigh 商：

$$
\lambda = \frac{\int_\Omega |\nabla u|^2 \, dx}{\int_\Omega u^2 \, dx}
$$

最小化 Rayleigh 商即可得到最小特征值。

## 目录结构

```
ex_5_4/
├── README.md
├── dim5/           # 5 维问题
│   ├── quadrature.py
│   ├── tnn.py
│   ├── integration.py
│   └── ex_5_4_dim5.py
├── dim10/          # 10 维问题
│   └── ...
└── dim20/          # 20 维问题
    └── ...
```

## 参数配置

| 参数 | 值 |
|------|------|
| 计算域 | $[0,1]^d$ |
| 张量秩 $p$ | 50 |
| 积分点数 $N$ | 1600 (16×100) |
| 网络结构 | [1,100,100,100,50] |
| 激活函数 | sin(x) |

## 运行方法

```bash
# 5 维
cd dim5
python ex_5_4_dim5.py

# 10 维
cd dim10
python ex_5_4_dim10.py

# 20 维
cd dim20
python ex_5_4_dim20.py
```

## 关键技术

### 特征值计算

通过 TNN 计算 $H^1$ 和 $L^2$ 内积：

```python
# H^1 半范数 (分子)
inner1 = Int2TNN_amend_1d(w, w, alpha, phi, alpha, phi, grad_phi, grad_phi)

# L^2 范数 (分母)
inner0 = Int2TNN(w, alpha, phi, alpha, phi)

# Rayleigh 商
lambda_approx = inner1 / inner0
```

### 边界条件

使用边界函数 $\text{bd}(x) = (x-a)(b-x)$ 自动满足齐次 Dirichlet 条件。

## 预期结果

| 维度 | 计算特征值 | 相对误差 | 训练时间 (GPU) |
|------|------------|----------|----------------|
| 5D | ~49.35 | ~10⁻⁴ | ~25 min |
| 10D | ~98.70 | ~10⁻³ | ~50 min |
| 20D | ~197.39 | ~10⁻² | ~2 hours |

## 注意事项

- 特征函数需要归一化：$\|u\|_{L^2} = 1$
- 优化目标是最小化 Rayleigh 商
- 高阶特征值需要额外的正交约束
