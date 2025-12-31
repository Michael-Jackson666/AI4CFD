# PINO 示例代码

本文件夹包含 Physics-Informed Neural Operators (PINO) 的示例代码和相关数据。

## 📁 文件列表

| 文件名 | 描述 | PDE 类型 |
|--------|------|----------|
| `burgers_1d.py` | 1D Burgers 方程 PINO 求解 | 非线性对流扩散 |
| `darcy_2d.py` | 2D Darcy 流 PINO 求解 | 椭圆型 PDE |
| `heat_1d.py` | 1D 热传导方程 PINO 求解 | 抛物型 PDE |

## 🔬 问题描述

### 1. Burgers 方程 (burgers_1d.py)

求解 1D 粘性 Burgers 方程：

$$
\frac{\partial u}{\partial t} + u \frac{\partial u}{\partial x} = \nu \frac{\partial^2 u}{\partial x^2}, \quad x \in [-1, 1], \, t \in [0, 1]
$$

- **初始条件**：$u(x, 0) = -\sin(\pi x)$
- **边界条件**：$u(-1, t) = u(1, t) = 0$ (周期边界)
- **粘性系数**：$\nu = 0.01$

PINO 学习的算子：$\mathcal{G}: u_0(x) \mapsto u(x, t)$，即从初始条件映射到完整时空解。

### 2. Darcy 流 (darcy_2d.py)

求解 2D 稳态 Darcy 流方程：

$$
-\nabla \cdot (a(x, y) \nabla u) = f(x, y), \quad (x, y) \in [0, 1]^2
$$

- **边界条件**：$u|_{\partial\Omega} = 0$ (Dirichlet)
- **渗透率场**：$a(x, y)$ 为随机生成的对数正态场

PINO 学习的算子：$\mathcal{G}: a(x, y) \mapsto u(x, y)$，即从渗透率场映射到压力场。

### 3. 热传导方程 (heat_1d.py)

求解 1D 热传导方程：

$$
\frac{\partial u}{\partial t} = \alpha \frac{\partial^2 u}{\partial x^2}, \quad x \in [0, 1], \, t \in [0, 1]
$$

- **初始条件**：$u(x, 0) = \sin(\pi x)$
- **边界条件**：$u(0, t) = u(1, t) = 0$
- **热扩散系数**：$\alpha = 0.1$

## 🚀 运行方式

### 环境准备

```bash
# 安装依赖
pip install torch numpy scipy matplotlib tqdm

# GPU 支持（可选）
pip install torch --index-url https://download.pytorch.org/whl/cu118
```

### 运行示例

```bash
# 进入示例目录
cd PINO/examples

# 运行 Burgers 方程示例
python burgers_1d.py

# 运行 Darcy 流示例
python darcy_2d.py

# 运行热传导方程示例
python heat_1d.py
```

## 📊 输出说明

运行示例后，会生成以下输出：

1. **训练日志**：控制台输出训练进度和损失
2. **损失曲线**：`*_loss.png` 训练/物理损失随 epoch 变化
3. **预测结果**：`*_prediction.png` PINO 预测 vs 参考解对比
4. **误差分析**：`*_error.png` 误差分布热力图

## 💡 关键代码结构

每个示例都遵循以下结构：

```python
# 1. 导入模块
import torch
from pino_core import PINO, FNOBlock

# 2. 定义 PDE 残差
def pde_residual(u, x, t, params):
    """计算 PDE 残差用于物理约束"""
    u_t = torch.autograd.grad(u, t, ...)
    u_x = torch.autograd.grad(u, x, ...)
    u_xx = torch.autograd.grad(u_x, x, ...)
    return u_t + u * u_x - nu * u_xx

# 3. 创建 PINO 模型
model = PINO(
    input_dim=...,
    output_dim=...,
    modes=12,
    width=32
)

# 4. 定义损失函数
def loss_fn(model, data):
    pred = model(data['input'])
    data_loss = F.mse_loss(pred, data['output'])
    pde_loss = pde_residual(pred, ...).pow(2).mean()
    return data_loss + lambda_pde * pde_loss

# 5. 训练循环
for epoch in range(epochs):
    loss = loss_fn(model, batch)
    loss.backward()
    optimizer.step()
```

## 🔧 超参数调优建议

| 参数 | 推荐范围 | 说明 |
|------|----------|------|
| `modes` | 8-20 | Fourier 模式数，越大越精确但计算量增加 |
| `width` | 20-64 | 隐藏层宽度 |
| `depth` | 4-6 | FNO 层数 |
| `lambda_pde` | 0.1-10.0 | 物理损失权重，需根据问题调整 |
| `lr` | 1e-3 - 1e-4 | 学习率 |
| `batch_size` | 4-20 | 批量大小 |

## ⚠️ 常见问题

1. **内存不足**：减小 `modes`、`width` 或 `batch_size`
2. **训练不收敛**：降低学习率，增加训练 epoch
3. **物理残差过大**：增大 `lambda_pde` 权重
4. **过拟合**：增加训练数据或添加正则化

## 📖 参考资料

- Li et al., "Physics-Informed Neural Operator for Learning Partial Differential Equations" (2021)
- 更多理论细节请参阅 `PINO/tutorial/pino_overview.ipynb`
