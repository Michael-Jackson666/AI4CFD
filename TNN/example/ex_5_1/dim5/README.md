# TNN (Tensor Neural Network) 5D PDE Solver Example

## 概述 (Overview)

本示例展示了如何使用张量神经网络（Tensor Neural Network, TNN）求解五维偏微分方程问题。TNN是一种专门用于高维PDE求解的深度学习方法，通过张量分解的思想将高维问题分解为多个低维问题，从而有效克服维数灾难。

This example demonstrates how to use Tensor Neural Networks (TNN) to solve a five-dimensional partial differential equation (PDE). TNN is a deep learning method specifically designed for high-dimensional PDE solving, which decomposes high-dimensional problems into multiple low-dimensional problems through tensor decomposition, effectively overcoming the curse of dimensionality.

## 问题描述 (Problem Formulation)

### 目标函数 (Objective Function)

求解使能量泛函最小化的函数 $u: [-1,1]^5 \to \mathbb{R}$：

$$
\min_{u \in H^1_0([-1,1]^5)} E(u) = \frac{1}{2}\int_{[-1,1]^5} |\nabla u|^2 dx - \int_{[-1,1]^5} f \cdot u \, dx
$$

其中右端项为：

$$
f(x_1, x_2, x_3, x_4, x_5) = \sum_{k=1}^{5} \sin(2\pi x_k) \prod_{i \neq k} \sin(\pi x_i)
$$

### 边界条件 (Boundary Conditions)

使用强制边界条件：
$$
u(x_1, \ldots, x_5) = \prod_{k=1}^{5} \sin(\pi x_k) \cdot \text{NN}(x_1, \ldots, x_5)
$$

这保证了 $u$ 在边界 $\partial[-1,1]^5$ 上自动为零。

## TNN 架构 (TNN Architecture)

### 核心思想 (Core Idea)

TNN使用张量分解来表示高维函数：

$$
u(x_1, \ldots, x_d) \approx \sum_{i=1}^{p} \alpha_i \prod_{k=1}^{d} \phi_k^{(i)}(x_k)
$$

其中：
- $d = 5$ 是问题维度
- $p = 50$ 是张量秩（rank），控制表示能力
- $\alpha_i$ 是可学习的缩放参数
- $\phi_k^{(i)}(x_k)$ 是第 $k$ 维上的第 $i$ 个基函数，由一维神经网络表示

### 网络结构 (Network Structure)

每个一维基函数由全连接神经网络构建：

```
输入层: 1维 → 隐藏层1: 100神经元 → 隐藏层2: 100神经元 → 隐藏层3: 100神经元 → 输出层: 50维
激活函数: sin(x)
```

- **输入维度**: 1 (单变量 $x_k$)
- **隐藏层**: 3层，每层100个神经元
- **输出维度**: 50 (对应50个基函数)
- **激活函数**: $\sin(x)$（可微且周期性）
- **边界条件强制**: $\sin(\pi x_k)$ 乘子

## 代码结构 (Code Structure)

### 主要文件 (Main Files)

1. **`ex_5_1_dim5.py`** - 主程序文件
   - 定义5D PDE问题
   - 配置TNN模型
   - 实现训练流程
   - 可视化和结果保存

2. **`tnn.py`** - TNN模型实现
   - `TNN_Linear`: 批量线性层，处理 [dim, n_out, n_in] 形状的张量
   - `TNN_Scaling`: 缩放参数层，存储 $\alpha$ 参数
   - `TNN_Sin`: 正弦激活函数及其导数
   - `TNN`: 完整的TNN模型类
     - `forward()`: 前向传播，计算基函数值
     - `need_grad=1`: 同时计算一阶导数
     - `need_grad=2`: 同时计算一阶和二阶导数

3. **`integration.py`** - 积分运算工具
   - `Int1TNN()`: 单个TNN的积分
   - `Int2TNN()`: 两个TNN的L²内积
   - `Int2TNN_amend_1d()`: 两个TNN的H¹内积（含导数项）
   - `Int3TNN()`, `Int4TNN()`: 三个/四个TNN的乘积积分
   - `error0_estimate()`, `error1_estimate()`: L²和H¹误差估计

4. **`quadrature.py`** - 数值积分规则
   - `quadrature_1d()`: 1D高斯-勒让德积分（1-16点）
   - `composite_quadrature_1d()`: 复合积分规则，将区间分为M个子区间
   - `composite_quadrature_2d()`: 2D张量积积分规则

### 积分方案 (Quadrature Scheme)

```python
N = 16    # 每个子区间的积分点数（16点高斯-勒让德）
M = 200   # 区间[-1,1]的子区间数量
```

总积分点数：16 × 200 = 3200 points per dimension

## 训练策略 (Training Strategy)

### 两阶段优化 (Two-Stage Optimization)

#### 第一阶段: Adam优化器
```python
epochs: 50,000
learning rate: 0.001
optimizer: Adam
purpose: 快速找到较好的初始解
```

#### 第二阶段: L-BFGS优化器
```python
epochs: 10,000
max iterations per epoch: 1
optimizer: L-BFGS
purpose: 精细优化，找到更精确的解
```

### 损失函数 (Loss Function)

基于变分形式的能量泛函：

$$
\text{Loss} = \frac{1}{2}\int_{[-1,1]^5} |\nabla u|^2 dx - \int_{[-1,1]^5} f \cdot u \, dx
$$

在代码中实现为：
```python
loss = 0.5 * Int2TNN_amend_1d(w, w, alpha, phi, alpha, phi, grad_phi, grad_phi) \
       - Int2TNN(w, alpha_F, F, alpha, phi)
```

其中：
- 第一项: $\frac{1}{2}||u||_{H^1}^2$ - 正则化项（能量项）
- 第二项: $-(f, u)_{L^2}$ - 右端项的作用

## 使用方法 (Usage)

### 环境要求 (Requirements)

```bash
Python >= 3.9
PyTorch >= 1.10
NumPy
SciPy (for advanced quadrature rules)
```

### 运行示例 (Run Example)

```bash
cd /path/to/dim5
python ex_5_1_dim5.py
```

### 主要参数设置 (Key Parameters)

在 `ex_5_1_dim5.py` 中可以调整以下参数：

```python
# TNN架构参数
dim = 5                    # 问题维度
size = [1, 100, 100, 100, 50]  # 网络结构: [输入, 隐藏层1, 隐藏层2, 隐藏层3, 输出]

# 积分参数
N = 16                     # 每个子区间的高斯点数
M = 200                    # 子区间数量

# 训练参数
epoch_Adam = 50000         # Adam优化器迭代次数
epoch_LBFGS = 10000        # L-BFGS优化器迭代次数
lr = 0.001                 # Adam学习率
```

## 关键技术细节 (Key Technical Details)

### 1. 强制边界条件 (Forced Boundary Conditions)

使用乘积形式的边界条件函数：

```python
def bd(x):
    """边界条件: sin(π*x)"""
    return torch.sin(np.pi * x)

def grad_bd(x):
    """边界条件的梯度: π*cos(π*x)"""
    return np.pi * torch.cos(np.pi * x)
```

在TNN前向传播中：
```python
phi = neural_network_output * bd_value
grad_phi = neural_network_output * grad_bd_value + neural_network_grad * bd_value
```

### 2. 批量矩阵运算 (Batch Matrix Operations)

TNN_Linear层处理形状为 `[dim, n_out, n_in]` 的权重：

```python
# 输入: x [dim, n_in, N]  (N个积分点)
# 权重: W [dim, n_out, n_in]
# 输出: y [dim, n_out, N]
y = W @ x  # 批量矩阵乘法
```

这使得5个维度的神经网络可以并行计算。

### 3. 同时计算函数值和导数 (Simultaneous Value and Gradient Computation)

通过链式法则高效计算梯度：

```python
# 前向传播同时记录中间激活值
for layer in layers:
    x = layer(x)
    grad_x = activation.grad(x) * grad_x
    grad_x = next_layer.weight @ grad_x
    x = activation(x)
```

### 4. 正交初始化 (Orthogonal Initialization)

```python
# 对每个维度的权重矩阵进行正交初始化
for j in range(dim):
    nn.init.orthogonal_(weight[j, :, :])
```

这有助于训练稳定性和收敛速度。

## 输出结果 (Output Results)

### 终端输出 (Console Output)

```
Epoch: 0, Loss: 1.234567e-02, Time: 0.123s
Epoch: 1000, Loss: 5.678901e-03, Time: 0.098s
...
Adam optimization finished!
Switching to L-BFGS...
Epoch: 50000, Loss: 1.234567e-05, Time: 0.156s
...
Training completed!
Final loss: 1.234567e-07
```

### 保存文件 (Saved Files)

- 模型权重会自动保存
- 损失历史可用于收敛性分析
- 可以添加可视化代码来绘制解的切片

## 数学背景 (Mathematical Background)

### 变分原理 (Variational Principle)

原PDE问题：
$$
-\Delta u = f \quad \text{in } \Omega = [-1,1]^5
$$
$$
u = 0 \quad \text{on } \partial\Omega
$$

等价于最小化能量泛函：
$$
E(u) = \frac{1}{2}\int_\Omega |\nabla u|^2 dx - \int_\Omega f u \, dx
$$

### TNN的优势 (Advantages of TNN)

1. **克服维数灾难**: 参数量随维度线性增长而非指数增长
   - 传统网格方法: $O(n^d)$ 个自由度
   - TNN方法: $O(d \cdot n \cdot p)$ 个参数

2. **自适应表示**: 通过调整秩 $p$ 控制表示能力

3. **物理约束**: 容易强制边界条件和对称性

4. **高效积分**: 利用张量积结构简化高维积分计算

### 收敛性分析 (Convergence Analysis)

可以使用H¹和L²范数监控收敛：

```python
# L²误差
error_L2 = error0_estimate(w, alpha_exact, u_exact, alpha, phi)

# H¹误差
error_H1 = error1_estimate(w, alpha_exact, u_exact, alpha, phi, 
                           grad_u_exact, grad_phi)
```

## 扩展方向 (Extensions)

### 1. 更高维问题
修改 `dim` 参数即可扩展到更高维度（如6D, 7D等）

### 2. 不同的PDE
修改右端项 `F` 和边界条件 `bd` 可求解其他PDE

### 3. 自适应秩选择
在训练过程中动态调整张量秩 `p`

### 4. 其他优化算法
尝试其他优化器如AdamW、SGD with momentum等

### 5. 物理信息约束
在损失函数中添加PDE残差项（类似PINN）

## 参考文献 (References)

1. **Tensor Neural Networks**: 基础理论和算法
2. **Physics-Informed Neural Networks (PINNs)**: 物理约束的深度学习方法
3. **Variational Methods for PDEs**: 变分原理和弱形式
4. **Numerical Integration**: 高斯积分规则和复合积分

## 常见问题 (FAQ)

### Q1: 为什么使用sin激活函数？
A: sin函数具有无限次可微性，且其导数容易计算，适合求解需要高阶导数的PDE问题。

### Q2: 如何选择张量秩p？
A: 秩p控制解的表示能力。通常从较小的值开始（如20-50），逐步增大直到损失收敛。秩越大，表示能力越强，但参数量和计算成本也越高。

### Q3: 为什么需要两阶段优化？
A: Adam快速但可能陷入局部最优；L-BFGS虽慢但能找到更精确的解。两者结合可以兼顾速度和精度。

### Q4: 如何处理更复杂的边界条件？
A: 修改 `bd()` 和 `grad_bd()` 函数。对于非齐次边界条件，可以将解分解为 $u = u_0 + u_h$，其中$u_0$满足边界条件，$u_h$为齐次边界条件的修正项。

### Q5: 如何可视化5D结果？
A: 可以固定部分变量，绘制低维切片。例如固定 $x_3=x_4=x_5=0$，绘制 $u(x_1, x_2, 0, 0, 0)$ 的2D图像。

## 联系与支持 (Contact & Support)

如有问题或建议，请参考：
- 项目主README文档
- 相关学术论文
- 提交Issue到代码仓库

---

**最后更新**: 2025年1月
**许可证**: 参见项目根目录LICENSE文件
