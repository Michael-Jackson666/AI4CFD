# VP_system - Vlasov–Poisson Solver Collection

## 简介

`VP_system` 包含一组用于求解 **Vlasov–Poisson** 方程的演示代码和工具，覆盖经典的等离子体物理问题（例如 Landau 阻尼、双流不稳定性等）。这些示例包括 Python 实现（易用、可视化丰富）和基于 HyPar 的 C 语言实现（高性能，适用于大规模计算）。

## 方程与物理背景

Vlasov–Poisson 系统通常表示为：

$$
\frac{\partial f}{\partial t} + v \cdot \nabla_x f + a(x,t) \cdot \nabla_v f = 0
$$

$$
\nabla_x \cdot E = \int f\,dv - n_0, \quad E = -\nabla_x \phi
$$

其中：
- $f(x,v,t)$ 表示相空间中的分布函数
- $E(x,t)$ 为自洽电场
- $n_0$ 为背景密度（通常归一化为 1）

常见目标：模拟电场扰动、阻尼/增长、相空间结构演化，并验证质量、能量守恒与数值稳定性。

## 目录结构

```
VP_system/
├── README.md                    # 本文件
├── Landou/                      # Landau 阻尼示例（Python 实现）
│   ├── README.md
│   ├── vp_solver.py
│   └── run_simulation.py
└── TwoStreamInstability/        # 双流不稳定性示例（Python + HyPar C）
    ├── README.md
    ├── hypar_solver/            # HyPar C 语言实现（高性能）
    └── python_solver/           # Python 实现（WENO5、Strang split 等）
```

## 子模块说明

- `Landou/` (Landau 阻尼) — Python 实现，使用半拉格朗日（semi-Lagrangian）或谱方法，配有可视化脚本。常见方法包括时间分裂（Strang splitting），并用谱法或快速 Poisson 求解（FFT）获得 $E(x,t)$。

- `TwoStreamInstability/` — 提供两个实现：Python（可读性强，适合教学与小规模实验）和 HyPar C（更快、更精确，适合大规模仿真）。HyPar 版本依赖编译与运行脚本，Python 版本则便于交互式调试与可视化（Matplotlib）。

## 数值方法要点

- 半拉格朗日（Semi-Lagrangian）方法：沿特征线插值，适用于 Vlasov 方程的对流项。
- WENO / 高阶格式：用于求解对流方程保证抗振荡性。
- 时间分裂（Strang Splitting）：由于 Vlasov 方程的对流形式，常用分裂法减少数值误差。典型分裂序列：

```python
S = S_v(dt/2) ∘ S_x(dt) ∘ S_v(dt/2)
```

- Poisson 方程求解：通常在周期域使用谱方法（FFT）求解 $
abla_x
the E$: 

$$
ik E_k = \rho_k, \quad E_k = -\frac{i \rho_k}{k}
$$

- 质量/能量守恒：数值方法要尽量保持如下不变量以保证物理逼真性：

$$
M = \iint f \, dx \, dv, \qquad E_{total} = \iint \frac{v^2}{2} f \, dx \, dv + \int \frac{|E|^2}{2} \, dx
$$

## 依赖与环境

建议在 Python 环境中运行以下库：

```bash
pip install numpy scipy matplotlib
```

HyPar（C 语言实现）需要从源码构建，并依赖 Gnuplot/编译器环境，详见 `TwoStreamInstability/hypar_solver/README.md`。

## 快速开始

### 1) 运行 Landau 阻尼（Python）

```bash
cd VP_system/Landou
python run_simulation.py
```

### 2) 运行 Two-Stream Python 版本

```bash
cd VP_system/TwoStreamInstability/python_solver
python quick_start.py
# 或运行 `stable_vp_solver.py` 获取更稳定的求解器
python stable_vp_solver.py
```

### 3) 使用 HyPar C 求解器（Two-Stream — 高性能）

```bash
cd VP_system/TwoStreamInstability/hypar_solver
chmod +x compile_hypar.sh install_hypar.sh
./compile_hypar.sh
./install_hypar.sh   # 若需要安装
./main                # 生成初始数据、运行 HyPar 程序
# 使用 gnuplot 绘图（必要时）
gnuplot plot.gp
```

## 可视化与结果

运行 Python 脚本后，将会生成：
- 相空间动画（`.gif`）
- 能量/质量守恒图和对比图
- 局部诊断图（电场、密度、速度分布等）

示例脚本 `run_simulation.py` 会保存绘图文件（如 `two_stream_stable.png`, `landau_damping_results.png`）。

## 调参建议（提高精度 / 效率）

- 增加网格：`nx`, `nv` 提高空间/速度解析度
- 减小时间步：`dt` 降低时间截断误差
- 使用 HyPar：需要更复杂的安装但可以显著提升运行速度

## 常见问题 (FAQ)

- **NaN/overflow**：尝试减小时间步 `dt`，或在数值格式中加入耗散项。
- **HyPar 编译失败**：请优先在 Linux 或 HPC 集群上构建，或使用 Python 版本作为替代。

## 贡献与扩展

欢迎贡献新的数值方法、边界条件处理或更高维实现。提交 PR 时请包含：
- 复现数据或脚本
- 运行指令（README 更新）
- 结果示例或测试集

## 参考与致谢

本模块参考了等离子体物理与数值 PDE 社区的多篇经典论文和代码实现。感谢开源软件贡献者。