# ex_5_2：非齐次 Dirichlet 边界条件

## 问题描述

求解高维 Poisson 方程，带非齐次 Dirichlet 边界条件：

$$
\begin{cases}
-\Delta u = f & \text{in } \Omega = [0,1]^d \\
u = g & \text{on } \partial\Omega
\end{cases}
$$

本例中：

$$
f(x) = \frac{\pi^2}{4} F(x), \quad F(x) = \sum_{i=1}^{d} \sin\left(\frac{\pi x_i}{2}\right)
$$

精确解：$u(x) = F(x)$

## 边界条件处理

使用**双 TNN 分解**处理非齐次边界：

$$
u = u_0 + \text{bd}(x) \cdot u_1
$$

其中：
- $u_0$：满足边界条件 $u_0|_{\partial\Omega} = g$ 的 TNN
- $u_1$：满足齐次边界条件的 TNN
- $\text{bd}(x)$：边界距离函数

## 目录结构

```
ex_5_2/
├── README.md
├── dim5/           # 5 维问题
│   ├── quadrature.py
│   ├── tnn.py
│   ├── integration.py
│   └── ex_5_2_dim5.py
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
| 积分点数 $N$ | 160 (16×10) |
| 网络结构 | [1,100,100,100,50] |

## 运行方法

```bash
# 5 维
cd dim5
python ex_5_2_dim5.py

# 10 维
cd dim10
python ex_5_2_dim10.py

# 20 维
cd dim20
python ex_5_2_dim20.py
```

## 关键技术

### 边界值处理

在边界 $x_i = 0$ 和 $x_i = 1$ 处分别计算 $F$ 的值：

```python
F_bd0[i,i,:] = sin(π/2 * 0) = 0      # x_i = 0 边界
F_bd1[i,i,:] = sin(π/2 * 1) = 1      # x_i = 1 边界
```

### 双 TNN 耦合

损失函数同时优化两个 TNN 网络，确保：
1. 边界条件精确满足
2. 内部 PDE 残差最小化

## 预期结果

| 维度 | L² 误差 | 训练时间 (GPU) |
|------|---------|----------------|
| 5D | ~10⁻⁴ | ~20 min |
| 10D | ~10⁻³ | ~40 min |
| 20D | ~10⁻² | ~1.5 hours |
