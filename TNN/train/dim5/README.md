# TNN 5D PDE Solver

## 问题描述

求解五维 Poisson 方程：

$$
-\Delta u = f \quad \text{in } \Omega = [-1,1]^5, \quad u = 0 \quad \text{on } \partial\Omega
$$

右端项定义为：

$$
f(x_1, x_2, x_3, x_4, x_5) = \sum_{k=1}^{5} \sin(2\pi x_k) \prod_{i \neq k} \sin(\pi x_i)
$$

## TNN 核心思想

使用**张量分解**克服维数灾难，将高维函数表示为低维函数的乘积和：

$$
u(x_1, \ldots, x_5) \approx \sum_{i=1}^{50} \alpha_i \prod_{k=1}^{5} \phi_k^{(i)}(x_k)
$$

**参数量对比**：
  1.传统网格方法：$O(N^5) \approx 10^{10}$ 个自由度
  2.TNN 方法：$O(5 \times 100 \times 50) \approx 25,000$ 个参数

**网络结构**：每个一维基函数 $\phi_k$ 由独立的神经网络表示
```
输入(1D) → FC(100) → FC(100) → FC(100) → 输出(50)
激活函数: sin(x)
边界条件: 乘以 sin(πx) 强制边界为零
```

## 用 TNN 求解 PDE 的步骤

1. **定义能量泛函**  
  利用五维 Poisson 方程的变分形式，将问题转化为最小化能量
  
  $$
  \mathcal{E}(u) = \frac{1}{2} \int_{\Omega} |\nabla u|^2 \mathrm{d}x - \int_{\Omega} f(x) u(x) \mathrm{d}x
  $$

2. **构造张量分解近似**  
  通过分离变量得到系数向量 $C$ 与一维基函数的乘积形式
  
  $$
  u_C(x) = \sum_{i=1}^{p} C_i \prod_{k=1}^{5} \phi_k^{(i)}(x_k)
  $$

3. **数值积分与刚度矩阵**  
  使用 `quadrature.py` 中的高斯-勒让德积分在一维上计算所有需要的
  $L^2$ 与 $H^1$ 积分，再通过张量积组合得到高维积分，组装线性系统
  
  $$
  A(C) = \int_\Omega \nabla u_C \cdot \nabla \phi_j \mathrm{d}x,
  \qquad B = \int_\Omega f \phi_j \mathrm{d}x
  $$

4. **最小化能量**  
  通过 Adam 与 L-BFGS 对 TNN 参数求解，使得 $A C = B$ 成立并最小化
  $\mathcal{E}(u_C)$。梯度由 `TNN.forward(..., need_grad=2)` 自动计算，确保二阶导数可用于能量项。

5. **误差评估与可视化**  
  训练结束后调用 `error0_estimate` 与 `error1_estimate` 评估 $L^2$ 与 $H^1$
  误差，并在固定三个坐标切片上绘制 TNN 预测与精确解。


## 代码文件说明

### 1. `quadrature.py` - 数值积分
提供高斯-勒让德积分点和权重：
- `quadrature_1d(N)`: 返回 N 点（1-16）高斯积分规则
- `composite_quadrature_1d(N, a, b, M)`: 复合积分，将 [a,b] 分成 M 个子区间

**本例配置**：N=16 点/子区间，M=200 子区间 → 总共 3200 个积分点

### 2. `tnn.py` - TNN 神经网络
核心类：
- **`TNN_Linear`**: 批量线性层，形状 [dim, n_out, n_in]，同时处理 5 个维度
- **`TNN_Sin`**: 正弦激活函数，包含 `forward()`, `grad()`, `grad_grad()` 方法
- **`TNN`**: 主模型类
  ```python
  forward(w, x, need_grad=0)  # 仅计算 φ
  forward(w, x, need_grad=1)  # 计算 φ 和 ∂φ/∂x
  forward(w, x, need_grad=2)  # 计算 φ, ∂φ/∂x, ∂²φ/∂x²
  ```

### 3. `integration.py` - 张量积分
利用张量积结构高效计算高维积分：
- **`Int1TNN(w, α, φ)`**: 单个 TNN 的积分
- **`Int2TNN(w, α₁, φ₁, α₂, φ₂)`**: 两个 TNN 的 L² 内积
- **`Int2TNN_amend_1d(...)`**: 两个 TNN 的 H¹ 内积（含导数项）
  
**关键优化**：不计算 N<sup>5</sup> 个网格点，而是利用张量积将复杂度从 O(N<sup>5</sup>) 降到 O(5N)

### 4. `ex_5_1_dim5.py` - 主训练程序
完整的训练流程：

**模型配置**：
```python
dim = 5                          # 问题维度
size = [1, 100, 100, 100, 50]   # 网络结构
p = 50                           # 张量秩
```

**损失函数**（变分形式）：
```python
# 求解系数 C，使其满足变分方程
A @ C = B  # A: 刚度矩阵, B: 载荷向量
loss = ∫|ΔC·φ|² + (d+3)²π⁴∫f² + 2(d+3)π²∫(ΔC·φ)·f
```

**两阶段优化**：
1. **Adam**: 50,000 epochs, lr=0.003 (快速探索)
2. **L-BFGS**: 10,000 epochs (精细优化)

**可视化**：绘制两个优化器的Loss曲线

## 快速开始

### 环境要求
```bash
Python >= 3.9
PyTorch >= 1.10
NumPy, Matplotlib
```

### 运行训练
```bash
cd TNN/train/dim5
python ex_5_1_dim5.py
```

### 主要参数
```python
quad = 16       # 每子区间的高斯点数
n = 200         # 子区间数量
p = 50          # 张量秩
size = [1, 100, 100, 100, p]  # 网络层数
epochs_adam = 50000
epochs_lbfgs = 10000
```

## 核心技术

### 1. 强制边界条件
```python
def bd(x):
    return (x - a) * (b - x)  # 自动满足边界为零
```

### 2. 批量梯度计算
通过 `need_grad` 参数在前向传播中同时计算导数（链式法则），避免重复计算

### 3. 张量积积分
利用可分离性：

$$
\int_{\Omega} \prod_{k=1}^5 f_k(x_k) dx = \prod_{k=1}^5 \int f_k(x_k) dx_k
$$

将 5 维积分分解为 5 个 1 维积分的乘积

## 输出结果

**终端输出**：
```
epoch = 0
loss = 2.34567e-02
error0 = 1.23e-03
error1 = 4.56e-03
****************************************
...
Done!
Training took: 2345.67s
```

**保存文件**：
- `solution_2d_slice.png`: 2D 切片可视化（预测/精确解/误差）
- `model/*.pkl`: 训练的模型权重（如果 save=True）

---

**更新时间**: 2025年1月  
**许可证**: 见项目根目录 LICENSE 文件

