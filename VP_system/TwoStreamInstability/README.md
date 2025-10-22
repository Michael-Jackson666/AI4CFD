# 双流不稳定性模拟 (Two-Stream Instability)

## 📋 简介

本目录包含双流不稳定性 **Vlasov-Poisson 系统**的模拟代码，提供 **C 语言（HyPar）** 和 **Python** 两种实现。

### 物理背景

双流不稳定性是等离子体物理中的经典现象：
- 两束速度相反的电子束相遇
- 由于库仑相互作用产生电场扰动
- 扰动指数增长，形成**双涡旋结构** (double vortex)
- 最终达到非线性饱和状态

### Vlasov-Poisson 方程组

$$
\frac{\partial f}{\partial t} + v \cdot \frac{\partial f}{\partial x} - E \cdot \frac{\partial f}{\partial v} = 0 \quad \text{(Vlasov 方程)}
$$

$$
\frac{\partial E}{\partial x} = \int f \, dv - 1 \quad \text{(Poisson 方程)}
$$

其中：
- $f(x,v,t)$ 是分布函数
- $E(x,t)$ 是自洽电场
- $x$ 是空间坐标，$v$ 是速度坐标

## 📁 文件夹结构

```
TwoStreamInstability/
├── README.md                  # 本文档（总览）
│
├── hypar_solver/              # ⚡ HyPar C 语言求解器（高性能）
│   ├── README.md              # C 求解器详细文档
│   ├── main.c                 # 初始条件生成器
│   ├── main                   # 编译后的可执行文件
│   ├── solver.inp             # 求解器配置（网格、时间步等）
│   ├── physics.inp            # 物理模型配置
│   ├── boundary.inp           # 边界条件
│   ├── librom.inp             # libROM 配置（可选）
│   ├── initial.inp            # 初始条件（二进制格式）
│   ├── plot.gp                # Gnuplot 可视化脚本
│   ├── op.gif                 # 输出动画
│   ├── install_hypar.sh       # HyPar 完整安装脚本
│   └── compile_hypar.sh       # HyPar 简化编译脚本
│
└── python_solver/             # 🐍 Python 求解器（易用）
    ├── README.md              # Python 求解器详细文档
    ├── stable_vp_solver.py    # ⭐ 改进的稳定求解器（WENO5+RK4）
    ├── simple_vp_solver.py    # 简单版求解器（2阶中心差分）
    ├── quick_start.py         # 快速演示脚本
    ├── run_simulation.py      # 通用运行脚本
    ├── two_stream_stable.png  # 4 面板诊断图
    └── two_stream_stable.gif  # 相空间演化动画
```

## 🚀 快速开始

### 选项 A: Python 求解器（推荐新手）✅

**优点**: 易于安装、快速上手、结果可视化丰富

```bash
# 进入 Python 求解器目录
cd python_solver

# 快速演示
python quick_start.py

# 或使用改进的稳定求解器
python stable_vp_solver.py
```

**预期输出**:
- `two_stream_stable.png` - 4 面板诊断图（相空间、守恒量、密度）
- `two_stream_stable.gif` - 动画

**详细文档**: 查看 [`python_solver/README.md`](python_solver/README.md)

---

### 选项 B: HyPar C 语言求解器（高性能）⚡

**优点**: 专业级精度、快速计算、工业标准

```bash
# 进入 HyPar 求解器目录
cd hypar_solver

# 步骤 1: 生成初始条件
./main

# 步骤 2: 运行 HyPar（需要先安装）
HyPar

# 步骤 3: 生成动画
gnuplot plot.gp
```

**预期输出**:
- `op_00000.dat` ~ `op_00100.dat` - 时间快照（二进制）
- `op.gif` - Gnuplot 生成的动画

**HyPar 安装**: 查看 [`hypar_solver/README.md`](hypar_solver/README.md)

**详细文档**: 查看 [`hypar_solver/README.md`](hypar_solver/README.md)

## 🎯 方法对比

| 特性 | Python 求解器 | HyPar C 求解器 |
|------|--------------|---------------|
| **安装难度** | ⭐ 简单 (pip install) | ⭐⭐⭐ 复杂 (源码编译) |
| **运行速度** | 慢 (~10分钟) | 快 (~1分钟) |
| **数值精度** | 中等 | 高（WENO5） |
| **稳定性** | 一般（可能出现NaN） | 优秀 |
| **易用性** | ⭐⭐⭐ 友好 | ⭐⭐ 需要C/配置文件知识 |
| **可视化** | 丰富（Matplotlib） | 基础（Gnuplot） |
| **推荐场景** | 学习、快速验证 | 研究、发表 |

## 📊 预期结果

### 相空间演化

观察生成的动画（`.gif` 文件），应看到：

1. **t = 0-5**: 初始双 Maxwellian 分布（v = ±2）
   - 两个高斯峰分离
   - 空间周期性扰动

2. **t = 5-15**: 线性不稳定性阶段
   - 电场扰动指数增长
   - 相空间出现波状结构
   - **双涡旋开始形成**

3. **t = 15-25**: 非线性饱和
   - 清晰的**双涡旋结构** 🌀🌀
   - 涡旋可能合并或分裂
   - 达到准稳态

### 守恒量

- **质量**: $M = \iint f \, dx \, dv \approx \text{常数}$（应该精确守恒）
- **能量**: $E = \iint \frac{v^2}{2} f \, dx \, dv + \int \frac{E^2}{2} \, dx$（轻微增长是正常的）
- **熵**: $S = -\iint f \ln(f) \, dx \, dv$（应单调递增）

如果质量偏差 >1%，说明数值方法有问题。

## ⚙️ 参数配置

### 关键物理参数

在 `hypar_solver/main.c` 或 Python 代码中：

```c
double T = 1.0;          // 温度（热速度）
double k = 1.0;          // 波数（扰动波长 = 2π/k）
double beam_v = 2.0;     // 束流速度（±beam_v）
double perturb = 0.1;    // 扰动幅度
```

### 数值参数

在 `hypar_solver/solver.inp` 中：

```plaintext
size               128 128   # 网格：x × v
n_iter             100       # 时间步数
dt                 0.2       # 时间步长
```

**推荐设置**:
- 快速测试: $64 \times 64$, n_iter=50
- 标准: $128 \times 128$, n_iter=100
- 高精度: $256 \times 256$, n_iter=200

## 🐛 常见问题

### Q1: Python 求解器出现 NaN 或 overflow

**原因**: Vlasov-Poisson 系统数值刚性强

**解决方案**:
```python
# 减小时间步长
dt = 0.05  # 而不是 0.1

# 增加数值耗散
epsilon = 1e-3  # 而不是 1e-4
```

### Q2: HyPar 编译失败（macOS）

**原因**: macOS 构建系统兼容性问题（`version` 文件冲突）

**解决方案**:
1. 使用 Linux 虚拟机或 HPC 集群
2. 寻找预编译二进制
3. 使用 Python 求解器作为替代

详见 [`hypar_solver/README.md`](hypar_solver/README.md)

### Q3: 看不到双涡旋结构

**可能原因**:
- 时间步数太少（增加 `n_iter` 到 100-200）
- 扰动幅度太小（增加 `perturb` 到 0.2）
- 网格分辨率不够（使用 $256 \times 256$）

### Q4: 动画太快/太慢

**Gnuplot** (`plot.gp`):
```gnuplot
set term gif animate delay 10  # 改为 20 更慢，5 更快
```

**Python** (`stable_vp_solver.py`):
```python
anim = FuncAnimation(..., interval=100, ...)  # 改为 200 更慢
```

## 📚 理论背景

### 线性增长率

小扰动近似下，双流不稳定性的增长率为：

$$
\gamma(k) \approx \sqrt{\frac{3}{2}} k v_{th}
$$

其中 $k$ 是波数，$v_{th} = \sqrt{T}$ 是热速度。

对于默认参数（$k=1$, $T=1$），理论增长率 $\gamma \approx 1.22$。

### 数值方法

**HyPar**:
- 空间离散: 5阶 WENO（加权本质无振荡）
- 时间积分: 4阶 Runge-Kutta
- 并行: MPI 支持

**Python 稳定求解器**:
- 空间离散: 5阶 WENO（简化版）
- 时间积分: 4阶 Runge-Kutta
- 非负性保持: max(f, 0)

## 📖 参考文献

1. **Vlasov 求解器综述**:
   Filbet & Sonnendrücker (2003) "Comparison of Eulerian Vlasov solvers"

2. **双流不稳定性经典论文**:
   O'Neil (1965) "Collisionless damping of nonlinear plasma oscillations"

3. **WENO 格式**:
   Shu (2009) "High Order Weighted Essentially Nonoscillatory Schemes"

4. **HyPar 文档**:
   https://github.com/debog/hypar

## 🤝 贡献与支持

- **问题反馈**: 提交 Issue
- **代码贡献**: 提交 Pull Request
- **讨论交流**: Discussions

## 📄 许可证

- HyPar: BSD-3-Clause
- Python 代码: MIT License

---

**🎉 祝您成功观察到双涡旋现象！**

如有问题，请先查看各子文件夹的 `README.md` 以获取详细文档。
