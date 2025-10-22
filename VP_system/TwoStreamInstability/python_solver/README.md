# Python 求解器

## 简介

本文件夹包含双流不稳定性（Two-Stream Instability）Vlasov-Poisson 系统的 Python 实现，提供多个不同复杂度的求解器版本。

## 文件说明

### 求解器实现

1. **stable_vp_solver.py** - 改进的稳定求解器 ⭐
   - 5 阶 WENO 空间离散
   - RK4 时间积分
   - 非负性保持
   - 数值耗散项
   - 自动从 HyPar 格式读取初始条件

2. **simple_vp_solver.py** - 简单版本
   - 2 阶中心差分
   - RK4 时间积分
   - 基础实现，适合学习

3. **quick_start.py** - 快速演示脚本
   - 预设参数
   - 快速运行
   - 适合快速查看结果

4. **run_simulation.py** - 通用运行脚本
   - 命令行参数
   - 灵活配置

### 输出文件

- **two_stream_stable.png**: 4 面板诊断图
  - 相空间演化
  - 最终相空间
  - 守恒量
  - 密度演化
  
- **two_stream_stable.gif**: 相空间动画

- **two_stream_instability.gif**: 简单求解器输出动画

- **diagnostics.png**: 诊断图

## 快速开始

### 方法 1: 使用改进的稳定求解器（推荐）

```bash
python stable_vp_solver.py
```

**自定义参数**:
```python
from stable_vp_solver import StableVPSolver
import numpy as np

# 创建求解器
solver = StableVPSolver(
    nx=128,           # x 网格点数
    nv=128,           # v 网格点数
    Lx=2*np.pi,       # x 域大小
    Lv=12.0,          # v 域大小
    dt=0.1,           # 时间步长
    T=20.0,           # 总时间
    epsilon=1e-4      # 数值耗散
)

# 设置初始条件（自动或手动）
solver.load_initial_condition('initial.inp')  # 从 HyPar 文件
# 或
solver.set_default_initial_condition()        # 默认双 Maxwellian

# 求解
snapshots = solver.solve(save_interval=10)

# 可视化
solver.plot_results(snapshots, 'result.png')
solver.create_animation(snapshots, 'animation.gif')
```

### 方法 2: 快速演示

```bash
python quick_start.py
```

### 方法 3: 简单求解器

```bash
python simple_vp_solver.py
```

## 物理参数

### 默认配置
- **网格**: 128×128
- **域**: x ∈ [0, 2π], v ∈ [-6, 6]
- **时间步**: dt = 0.1
- **总时间**: T = 20.0

### 初始条件（双 Maxwellian）
```python
# 两束反向运动的等离子体
v1 = +2.0              # 右移束流速度
v2 = -2.0              # 左移束流速度
T = 1.0                # 温度
amplitude = 0.1        # 扰动幅度
k = 1.0                # 波数

# 分布函数
M1 = (2πT)^(-1/2) * exp(-(v-v1)²/(2T))
M2 = (2πT)^(-1/2) * exp(-(v-v2)²/(2T))
f(x,v,t=0) = 0.5(M1 + M2) * [1 + amplitude*cos(kx)]
```

## 数值方法对比

| 特性 | stable_vp_solver.py | simple_vp_solver.py |
|------|---------------------|---------------------|
| 空间离散 | WENO5 | 2阶中心差分 |
| 时间积分 | RK4 | RK4 |
| 非负性 | ✓ | ✗ |
| 数值耗散 | ✓ | ✗ |
| 稳定性 | 较好 | 较差 |
| 速度 | 慢 | 快 |

## 预期结果

### 时间演化
1. **t = 0-5**: 初始扰动线性增长
2. **t = 5-10**: 非线性效应增强
3. **t = 10-15**: 形成涡旋结构
4. **t = 15-20**: 双涡旋稳定状态

### 相空间特征
- **初始**: 两个分离的 Gaussian 峰 (v = ±2)
- **线性阶段**: 波状扰动在 x 方向传播
- **非线性阶段**: 粒子捕获，形成涡旋
- **饱和状态**: 清晰的**双涡旋结构** (double vortex)

### 守恒量
- **质量**: M = ∫∫f dx dv ≈ 常数
- **能量**: E = ∫∫(v²/2)f dx dv + ∫E²/2 dx ≈ 常数（略微增长）
- **熵**: S = -∫∫f ln(f) dx dv（单调递增）

## 已知问题与限制

### 1. 数值不稳定性 ⚠️

**症状**: `NaN` 或 `overflow` 错误

**原因**:
- Vlasov-Poisson 系统的刚性
- 高速梯度导致数值爆炸
- WENO 权重计算中的极小量

**解决方法**:
```python
# 减小时间步长
dt = 0.05  # 而不是 0.1

# 增加数值耗散
epsilon = 1e-3  # 而不是 1e-4

# 限制最大速度范围
Lv = 10.0  # 而不是 12.0
```

### 2. 精度限制

Python 实现无法达到专业求解器（HyPar）的精度，因为：
- 缺乏自适应时间步长
- WENO 实现简化
- 没有保守性修正
- 边界处理不够精细

### 3. 性能

- **128×128 网格**: ~5-10 分钟（Python）
- **256×256 网格**: ~30-60 分钟（Python）
- HyPar（C++）快 10-100 倍

## 调试技巧

### 检查守恒量
```python
# 在 solve() 中每步检查
print(f"Mass: {self.diagnostics['mass'][-1]:.6f}")
print(f"Energy: {self.diagnostics['energy'][-1]:.6f}")

# 质量应该接近 2π×12×1 ≈ 75.4
# 如果偏差 >1%，说明有问题
```

### 可视化中间步骤
```python
# 增加保存频率
snapshots = solver.solve(save_interval=5)  # 每5步保存

# 检查电场
solver.compute_electric_field()
plt.plot(solver.x, solver.E)
plt.show()
```

### 降低分辨率测试
```python
# 快速测试
solver = StableVPSolver(64, 64, 2*np.pi, 12.0, 0.2, 10.0)
```

## 进阶使用

### 读取 HyPar 输出

```python
import numpy as np

def read_hypar_output(filename):
    """读取 HyPar 输出文件 op_*.dat"""
    with open(filename, 'rb') as f:
        ndims = np.fromfile(f, dtype=np.int32, count=1)[0]
        sizes = np.fromfile(f, dtype=np.int32, count=ndims)
        data = np.fromfile(f, dtype=np.float64)
    
    nv, nx = sizes
    f = data.reshape((nv, nx), order='C')
    return f

# 使用
f_hypar = read_hypar_output('../hypar_solver/op_00100.dat')
```

### 自定义初始条件

```python
def landau_damping(self):
    """Landau 阻尼初始条件"""
    X, V = np.meshgrid(self.x, self.v, indexing='ij')
    X, V = X.T, V.T
    
    # Maxwellian 背景
    M = (2*np.pi)**(-0.5) * np.exp(-0.5*V**2)
    
    # 空间扰动
    alpha = 0.01
    k = 0.5
    
    self.f = M * (1 + alpha*np.cos(k*X))
```

## 依赖项

```bash
pip install numpy scipy matplotlib
```

**版本要求**:
- Python ≥ 3.7
- NumPy ≥ 1.18
- SciPy ≥ 1.5
- Matplotlib ≥ 3.2

## 参考文献

1. **数值方法**:
   - Filbet & Sonnendrücker (2003) "Comparison of Eulerian Vlasov solvers"
   - Shu (2009) "High Order Weighted Essentially Nonoscillatory Schemes"

2. **物理背景**:
   - Bittencourt, "Fundamentals of Plasma Physics" (2004)
   - Chen, "Introduction to Plasma Physics" (1984)

3. **双流不稳定性**:
   - Pierce (1948) "Theory of the Beam-Type Traveling-Wave Tube"
   - Buneman (1958) "Instability, Turbulence, and Conductivity"

## 贡献

欢迎改进和优化！特别是：
- 更稳定的 WENO 实现
- 自适应时间步长
- 并行化（NumPy → CuPy）
- 保守性修正算法

## 许可证

MIT License
