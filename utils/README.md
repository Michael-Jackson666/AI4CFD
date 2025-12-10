# 工具函数库 (utils)

本目录包含 AI4CFD 项目的通用工具函数，可供各模块共享使用。

## 文件说明

### `data_utils.py` - 数据工具

数据生成、加载和预处理函数：

| 函数 | 说明 |
|------|------|
| `generate_1d_poisson_data()` | 生成 1D Poisson 方程训练数据 |
| `generate_2d_poisson_data()` | 生成 2D Poisson 方程训练数据 |
| `normalize_data()` | 数据归一化 |
| `split_data()` | 训练/验证/测试集划分 |

**示例**：
```python
from utils.data_utils import generate_1d_poisson_data

x, u_exact, f = generate_1d_poisson_data(n_points=1000, domain=(-1, 1))
```

### `metrics.py` - 评估指标

PDE 求解的误差度量函数：

| 函数 | 说明 |
|------|------|
| `mse_loss()` | 均方误差 (MSE) |
| `mae_loss()` | 平均绝对误差 (MAE) |
| `relative_l2_error()` | 相对 L² 误差：$\frac{\|u - u_{exact}\|_2}{\|u_{exact}\|_2}$ |
| `relative_h1_error()` | 相对 H¹ 误差（含梯度项） |
| `max_error()` | 最大误差：$\max |u - u_{exact}|$ |

**示例**：
```python
from utils.metrics import relative_l2_error, max_error

l2_err = relative_l2_error(u_pred, u_exact)
max_err = max_error(u_pred, u_exact)
print(f"L² error: {l2_err:.4e}, Max error: {max_err:.4e}")
```

### `plotting.py` - 可视化工具

PDE 解和训练过程的可视化函数：

| 函数 | 说明 |
|------|------|
| `setup_plotting_style()` | 设置统一绘图风格 |
| `plot_1d_solution()` | 绘制 1D 解对比图 |
| `plot_2d_solution()` | 绘制 2D 解热力图 |
| `plot_3d_surface()` | 绘制 3D 表面图 |
| `plot_error_distribution()` | 绘制误差分布图 |
| `plot_training_history()` | 绘制训练损失曲线 |

**示例**：
```python
from utils.plotting import plot_1d_solution, plot_training_history

# 绘制解对比
plot_1d_solution(x, u_pred, u_exact, title="Poisson Solution")

# 绘制训练曲线
plot_training_history(loss_history, title="Training Loss")
```

## 使用方法

### 导入方式

```python
# 导入单个函数
from utils.metrics import relative_l2_error

# 导入整个模块
from utils import data_utils, metrics, plotting
```

### 依赖库

```
numpy
torch
matplotlib
seaborn
scipy
```

## 兼容性

- 支持 NumPy 数组和 PyTorch 张量
- 自动检测输入类型并选择对应实现
- GPU 张量会自动转移到 CPU 进行可视化
