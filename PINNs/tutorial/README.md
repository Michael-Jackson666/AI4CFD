# PINNs Tutorial

本教程位于 `PINNs/tutorial`，包含一系列 Jupyter notebooks，演示如何使用物理约束神经网络（Physics-Informed Neural Networks, PINNs）求解常见偏微分方程（PDE）。数学公式均采用 Markdown 的 `$...$` 或 `$$...$$` 语法。

## 教程列表

- `possion_1d.ipynb` — 1D Poisson 方程（Dirichlet 边界）示例。
- `heat_2d.ipynb` — 2D Heat 方程（时间依赖）教程。
- `ns_basic.ipynb` — 基础 Navier–Stokes 示例（入门）。
- `ns_advanced.ipynb` — 进阶 Navier–Stokes 实例（更复杂边界与参数）。
- `system_pde.ipynb` — 具有耦合 PDE 系统的示例（多方程耦合）。
- `vlasov_poisson.ipynb` — Vlasov–Poisson 耦合问题与 PINNs 解法。
- `tutorial_eng.ipynb` — 英文版综合教程（可直接用于教学）。
- `tutorial_chinese.ipynb` — 中文版综合教程（注释与说明为中文）。

## 目标与学习成果

完成这些笔记本后，你将：

- 理解 PINNs 的基本原理：如何将 PDE 的物理残差作为训练损失的一部分。
- 掌握如何为 PDE 构造损失项，例如：

  - 对于 Poisson 方程 $-\Delta u = f$，物理残差项为 $r(x) = -\Delta u(x) - f(x)$，训练最小化 $\|r\|_2^2$；
  - 对于时间依赖问题（例如 Heat 方程 $u_t - \Delta u = s$），损失包括时间导数项 $u_t$ 和空间导数项；

- 学会对边界条件（Dirichlet/Neumann）进行弱约束或强制处理（boundary function / penalty term）。

## 运行环境

建议创建 Python 虚拟环境并安装依赖：

```bash
pip install torch numpy scipy matplotlib jupyter
```

如果需要使用 GPU，请确保已安装对应版本的 CUDA 与 PyTorch。

## 快速开始

1. 打开 Jupyter：

```bash
jupyter notebook PINNs/tutorial
```

2. 打开想运行的 notebook（如 `possion_1d.ipynb`），按顺序运行单元格。通常 notebook 会包含：
   - 问题定义与解析解（如果有）
   - 网络结构与超参数（学习率、层数、激活函数）
   - 损失构造（物理残差、边界损失、MSE 数据误差）
   - 训练循环与可视化

## 常见技巧

- 使用自动求导计算高阶导数（PyTorch 的 `autograd.grad`）；
- 对 PDE 的残差使用归一化或加权来平衡不同物理项的尺度；
- 对时间依赖问题可以使用时间分块训练或序列式训练策略。

## 输出与可视化

通常 notebooks 会生成：
- 训练损失曲线（$L^2$ 残差随迭代的变化），
- 解的空间/时间分布图（热力图或表面图），
- 精确解与预测解的对比与误差图（绝对误差/相对误差）。

## 进一步阅读与扩展

- Raissi, Perdikaris & Karniadakis, "Physics-informed neural networks" (JCP, 2019)。
- 将 PINNs 与其他算子学习方法（如 DeepONet / FNO / TNN）进行比较，探索在高维或稀疏数据场景下的表现差异。

---

如果你希望我把每个 notebook 的关键超参数（例如 network size、training epochs、learning rate）摘录到 README 中以便快速参考，我可以继续为每个 notebook 添加一个小节并推送变更。