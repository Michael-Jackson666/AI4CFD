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

- 传统网格方法：$O(N^5) \approx 10^{10}$ 个自由度
- TNN 方法：$O(5 \times 100 \times 50) \approx 25{,}000$ 个参数

**网络结构**：每个一维基函数 $\phi_k$ 由独立的神经网络表示
```
输入(1D) → FC(100) → FC(100) → FC(100) → 输出(50)
激活函数: sin(x)
边界条件: 乘以 sin(πx) 强制边界为零
```

## 求解流程（与代码完全对应）

1. **高维积分离散**  
  `quadrature.py` 中的 `composite_quadrature_1d(quad=16, n=200)` 先在一维区间上生成 16×200=3200 个 Gauss-Legendre 积分点，再对五个维度执行张量积以形成高维积分权重。`ex_5_1_dim5.py` 中的所有积分函数如 `Int2TNN`、`Int2TNN_amend_1d` 都只依赖这些一维节点，避免构造 $3200^5$ 的网格。

2. **构造张量基与其导数**  
  `TNN.forward(w, x, need_grad=2)` 会输出 
  $$
  \phi_{d,i}(x_n),\quad \partial_{x_d} \phi_{d,i}(x_n),\quad \partial_{x_d}^2 \phi_{d,i}(x_n)
  $$
  其中 $d=1\ldots5,\ i=1\ldots p,\ n=1\ldots N$。
  - 边界条件通过 `bd(x)=(x-a)(b-x)` 在每个一维网络的输出处强制。
  - 所有基函数会进行 $\|\phi_{d,i}\|_{L^2(w)}=1$ 的归一化（`normalization(...)`）。

3. **求解能量泛函的最优系数**  
  目标能量为
  $$
  \mathcal{E}(u)=\tfrac12\int_{\Omega}|\nabla u|^2\,\mathrm{d}x-\int_{\Omega}f u\,\mathrm{d}x.
  $$
  代码中以张量基展开 $u=\sum_i C_i \prod_d \phi_{d,i}(x_d)$ 后，利用 
  ```python
  part1 = Int2TNN_amend_1d(...)
  part2 = Int2TNN(...)
  C = torch.linalg.solve(part1, (dim+3)*pi**2*part2)
  ```
  建立线性系统 $A C = B$，其中 `part1` 是 $\langle \nabla \phi_i, \nabla \phi_j \rangle$，`part2` 是 $\langle f, \phi_j \rangle$。

4. **Laplace 项与损失构造**  
  为了最小化 $\| -\Delta u - f\|_{L^2}$，脚本把二阶导数张量 `grad_grad_phi` 替换到每个维度上，得到
  $$
  \Delta u = \sum_i C_i \sum_{d=1}^5 \partial_{x_d}^2 \phi_{d,i}(x_d) \prod_{k\ne d}\phi_{k,i}(x_k).
  $$
  最终损失由三部分组成：
  $$
  	ext{loss} = \langle \Delta u, \Delta u \rangle + (d+3)^2 \pi^4 \langle f,f \rangle + 2(d+3)\pi^2 \langle \Delta u, f \rangle,
  $$
  与代码 `part1 + (dim+3)**2*np.pi**4*part2 + 2*(dim+3)*np.pi**2*part3` 对应。

5. **优化与可视化**  
  - **Adam 阶段**：500 epoch，学习率 0.003（`loss_history_adam`）。
  - **L-BFGS 阶段**：100 epoch，学习率 1。`loss_history_lbfgs` 接在 Adam 之后绘制，图中还画出“Optimizer Switch”竖线。
  训练完成后，用 `error0_estimate`、`error1_estimate` 给出 $L^2$ 与 $H^1$ 误差，最后在切片 $(x_3,x_4,x_5)=(0,0,0)$ 上可视化 TNN 预测与精确解。


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

### 教程
推荐先阅读 **[tnn_tutorial.ipynb](tnn_tutorial.ipynb)**，该 Jupyter 教程按照代码逻辑逐步讲解求解流程。

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

