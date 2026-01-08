# KAN 求解 PDE 教程

本教程将带你从零开始学习如何使用 **Kolmogorov-Arnold Networks (KAN)** 求解偏微分方程。

## 📚 教程列表

| 教程 | 内容 | 难度 | 时长 |
|------|------|------|------|
| [kan_pde_tutorial.ipynb](kan_pde_tutorial.ipynb) | **KAN 求解 PDE 完整教程** | ⭐⭐ 基础 | 60 分钟 |

## 🎯 学习目标

完成本教程后，你将能够：

1. ✅ 理解 Kolmogorov-Arnold 表示定理的核心思想
2. ✅ 掌握 B-spline 基函数在 KAN 中的作用
3. ✅ 实现完整的 KAN 网络用于 PDE 求解
4. ✅ 对比 KAN 与传统 MLP (PINNs) 的差异
5. ✅ 调优 KAN 超参数以获得最佳性能

## 📖 教程内容概览

### [kan_pde_tutorial.ipynb](kan_pde_tutorial.ipynb)

**完整的 KAN 求解 PDE 教程（中文）**

#### 章节结构：

1. **引言** (10 分钟)
   - Kolmogorov-Arnold 表示定理
   - KAN vs MLP 的本质区别
   - 为什么 KAN 适合 PDE 求解

2. **B-spline 基础** (15 分钟)
   - B-spline 基函数数学定义
   - Cox-de Boor 递归公式
   - 代码实现和可视化

3. **KAN 层实现** (15 分钟)
   - KAN 层的数学表达
   - PyTorch 实现
   - 前向传播详解

4. **求解 Poisson 方程** (15 分钟)
   - 问题定义: $-u'' = \sin(\pi x)$
   - 损失函数设计
   - 训练和可视化

5. **进阶技巧** (5 分钟)
   - 自适应网格细化
   - 网格大小的选择
   - 正则化策略

## 🚀 快速开始

### 运行教程

```bash
cd KAN/tutorial
jupyter notebook kan_pde_tutorial.ipynb
```

### 环境要求

```bash
# 基础依赖
pip install torch numpy scipy matplotlib

# Jupyter 支持
pip install jupyter ipykernel

# 可选（更好的可视化）
pip install seaborn plotly
```

## 📊 教程特色

### 1. **交互式学习**

- 所有代码单元格可直接运行
- 实时可视化 B-spline 函数
- 对比不同超参数的效果

### 2. **详细注释**

- 每行关键代码都有中文注释
- 数学公式与代码一一对应
- 常见错误和调试技巧

### 3. **实战演练**

- 从零实现 B-spline 层
- 构建完整 KAN 网络
- 求解真实 PDE 问题

### 4. **性能对比**

- KAN vs MLP 精度对比
- 参数量和计算时间分析
- 收敛速度对比

## 💡 学习建议

### 初学者路径

1. **先浏览理论部分**
   - 理解 Kolmogorov-Arnold 定理
   - 了解 B-spline 基本概念
   
2. **运行示例代码**
   - 逐个单元格执行
   - 观察输出结果
   - 修改参数观察变化

3. **完成练习题**
   - 教程中的编程练习
   - 尝试求解新的 PDE

### 进阶学习

1. **深入理论**
   - 阅读 KAN 原论文
   - 研究 B-spline 数学性质
   - 探索其他基函数（小波、Fourier）

2. **扩展应用**
   - 二维 PDE 问题
   - 时间依赖问题
   - 非线性 PDE

3. **性能优化**
   - 自适应网格实现
   - 并行计算
   - GPU 加速

## 🔬 实验建议

在学习过程中，尝试以下实验：

### 实验 1: 网格大小影响

```python
# 测试不同 grid_size
for grid_size in [3, 5, 7, 10]:
    model = KANPDE(layers=[1, 16, 16, 1], grid_size=grid_size)
    # 训练并记录误差
```

**问题**: 更大的网格是否总是更好？

### 实验 2: B-spline 阶数

```python
# 测试不同阶数
for order in [2, 3, 4, 5]:
    # 比较收敛速度和最终精度
```

**问题**: 高阶 B-spline 有什么优缺点？

### 实验 3: 网络深度

```python
# 测试不同网络结构
architectures = [
    [1, 32, 1],
    [1, 16, 16, 1],
    [1, 16, 16, 16, 1]
]
```

**问题**: KAN 需要多深的网络？

## 📚 补充资源

### 必读论文

1. **KAN 原论文**:
   - Liu et al. (2024). "KAN: Kolmogorov-Arnold Networks"
   - [论文链接](https://arxiv.org/abs/2404.19756)

2. **Kolmogorov 定理**:
   - Kolmogorov (1957). "On the representation of continuous functions"

3. **B-spline 理论**:
   - de Boor (2001). "A Practical Guide to Splines"

### 在线资源

- [B-spline 可视化工具](https://www.ibiblio.org/e-notes/Splines/Intro.htm)
- [Kolmogorov-Arnold 定理讲解](https://en.wikipedia.org/wiki/Kolmogorov%E2%80%93Arnold_representation_theorem)

### 相关项目教程

- 本项目 PINNs 教程: `../../PINNs/tutorial/`
- 本项目 DeepONet 教程: `../../DeepONet/tutorial/`
- 本项目 FNO 教程: `../../FNO/tutorial/`

## ⚠️ 常见问题 FAQ

### Q1: KAN 比 MLP 慢多少？

**A**: B-spline 计算比简单激活函数慢 2-3 倍，但由于参数更少，整体训练时间接近。

### Q2: 什么时候不应该用 KAN？

**A**: 
- 解不光滑（如激波）
- 输入维度很高（>10D）
- 对速度要求极高的场景

### Q3: 如何选择 grid_size？

**A**: 
- 光滑问题: 5-7 足够
- 复杂问题: 10-15
- 先小后大，逐步细化

### Q4: KAN 能用于逆问题吗？

**A**: 可以！KAN 可以像 PINNs 一样用于参数反演，且通常精度更高。

### Q5: 如何可视化学到的函数？

**A**: 使用 `utils.py` 中的 `visualize_kan_function()` 函数。

## 🎓 课后练习

完成教程后，尝试以下练习巩固知识：

### 练习 1: 修改 Poisson 方程

求解: $-u'' = x^2$, $u(-1) = u(1) = 0$

提示: 修改 `source_term` 函数。

### 练习 2: 添加 Neumann 边界

求解: $-u'' = \sin(\pi x)$, $u'(-1) = u'(1) = 0$

提示: 边界损失用导数约束。

### 练习 3: 2D Poisson

求解: $-\Delta u = 1$, $(x,y) \in [0,1]^2$, Dirichlet BC

提示: 扩展 KAN 到二维输入。

## 🔗 相关资源

- [KAN GitHub 仓库](https://github.com/KindXiaoming/pykan)
- [PINNs vs KAN 对比](https://arxiv.org/abs/2404.19756)
- [B-spline 教程](https://pages.mtu.edu/~shene/COURSES/cs3621/NOTES/)

---

**Happy Learning! 🎉**

如有问题，请查看主文档 [../README.md](../README.md) 或提交 Issue。
