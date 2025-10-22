# HyPar C 语言求解器

## 简介

本文件夹包含基于 HyPar（高性能偏微分方程求解器）的双流不稳定性（Two-Stream Instability）Vlasov-Poisson 系统的 C 语言实现。

## 文件说明

### 核心文件
- **main.c**: C 语言主程序，生成双 Maxwellian 初始条件
- **main**: 编译后的可执行文件
- **main.dSYM/**: macOS 调试符号文件

### 配置文件
- **solver.inp**: 求解器配置
  - 网格: 128×128 (x, v)
  - 时间步: n_iter=100, dt=0.2
  - 总时间: T=20.0
  - 时间积分: RK4
  
- **physics.inp**: 物理模型配置
  - 模型: vlasov (Vlasov-Poisson)
  - 自洽电场: 启用
  
- **boundary.inp**: 边界条件
  - x 方向: 周期边界 (periodic)
  - v 方向: Dirichlet 边界 (absorbing)
  
- **librom.inp**: libROM 配置（模型降阶，可选）

### 初始条件
- **initial.inp**: 二进制格式的初始条件文件（128×128 网格）
  - 双 Maxwellian 分布: f = 0.5(M₁ + M₂)
  - 束流速度: v₁ = +2.0, v₂ = -2.0
  - 温度: T = 1.0
  - 扰动: amplitude = 0.1, 波数 k = 1.0

### 可视化
- **plot.gp**: Gnuplot 脚本，用于生成动画
- **op.gif**: 输出的动画文件

### 安装脚本
- **compile_hypar.sh**: HyPar 简化编译脚本
- **install_hypar.sh**: HyPar 完整安装脚本（384行）

## 使用方法

### 1. 生成初始条件

```bash
# 编译 C 程序（如果需要）
gcc -o main main.c -lm

# 运行生成初始条件
./main
```

输出：`initial.inp` (二进制格式)

### 2. 运行 HyPar 模拟

```bash
# 需要先安装 HyPar
# 参考 install_hypar.sh 或 compile_hypar.sh

# 运行模拟
HyPar
```

输出：`op_00000.dat` 到 `op_00100.dat` (101 个时间快照)

### 3. 生成动画

```bash
gnuplot plot.gp
```

输出：`op.gif`

## HyPar 安装

### 依赖项
- gcc / g++
- autoconf, automake, libtool
- (可选) MPI, PETSc, libROM

### 安装步骤

**方法 1: 使用安装脚本**
```bash
bash install_hypar.sh
```

**方法 2: 手动安装**
```bash
# 下载源码
git clone https://github.com/debog/hypar.git ~/hypar_install/hypar

# 编译安装
cd ~/hypar_install/hypar
autoreconf -i
./configure --prefix=$HOME/.local
make -j8
make install

# 添加到 PATH
export PATH="$HOME/.local/bin:$PATH"
```

### 已知问题

**编译错误**: `version` 文件冲突
- **症状**: C++ 编译器将 `version` 文件当作头文件
- **临时解决**: 删除 `version` 文件后重新配置
- **根本原因**: HyPar 构建系统在 macOS 上的兼容性问题

**建议替代方案**:
1. 使用 Linux 虚拟机或 HPC 集群
2. 寻找预编译的 HyPar 二进制文件
3. 使用 Python 求解器（见 `../python_solver/`）

## 物理背景

**双流不稳定性** (Two-Stream Instability)
- 两束反向运动的带电粒子流相互作用
- 初始状态: 双 Maxwellian 分布 (v = ±2.0)
- 不稳定性增长: 电场扰动指数增长
- 最终状态: 相空间出现**双涡旋结构** (double vortex)

**Vlasov-Poisson 系统**:
```
∂f/∂t + v·∂f/∂x + E·∂f/∂v = 0  (Vlasov 方程)
∂E/∂x = ∫f dv - 1               (Poisson 方程)
```

## 数值方法

- **空间离散**: 5 阶 WENO (加权本质无振荡)
- **时间积分**: 4 阶 Runge-Kutta (RK4)
- **网格**: 128×128 均匀网格
- **域大小**: x ∈ [0, 2π], v ∈ [-6, 6]

## 参考资料

- HyPar 官网: http://hypar.github.io/
- HyPar GitHub: https://github.com/debog/hypar
- 论文: Ghosh & Baeder (2012) "Compact Reconstruction Schemes with Weighted ENO Limiting"
- 双流不稳定性: Bittencourt, "Fundamentals of Plasma Physics"

## 许可证

本代码遵循 HyPar 的许可证（BSD-3-Clause）。
