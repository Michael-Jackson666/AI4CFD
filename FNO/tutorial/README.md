# FNO Tutorial

本教程位于 `FNO/tutorial`，包含基于 Fourier Neural Operator（FNO）实现的交互式 Jupyter 教程：

- `FNO_tutorial.ipynb`：以 PyTorch 为后端的示例 notebook，展示如何使用 FNO 求解带参数或函数右端项的 PDE。

## 教程目标

1. 介绍 FNO 的背景与理论：如何通过频谱（Fourier）层对算子进行学习。
2. 演示数据准备、模型构建（Fourier layers、MLP）、训练与评估流程。
3. 展示 FNO 在 PDE（如 Poisson / Burgers / Darcy 等）问题上的应用与可视化结果。

## 算法背景（核心思想）

FNO 通过在频谱域进行卷积来学习从输入函数 $a(x)$ 到解函数 $u(x)$ 的算子 $\mathcal{G}$：

$$
\mathcal{G}(a)(x) = \mathcal{F}^{-1}\left( W(\xi) \cdot \mathcal{F}[v](\xi) \right)(x) + \text{local terms},
$$

其中：
- $\mathcal{F}$ / $\mathcal{F}^{-1}$ 为 Fourier 变换与逆变换；
- $W(\xi)$ 是在频率域上学习到的可训练权重（按分段或通道）；
- $v$ 是网络中间表示（embedding）向量。

FNO 的优势：
- 更少的参数能学习全局算子依赖；
- 对输入函数的平滑性有较好泛化表现；
- 适用于高维或具有复杂参数化输入的 PDE。

## 教程文件概览

- `FNO_tutorial.ipynb`：包含以下步骤
  1. 导入依赖与数据生成/加载（包括训练/测试划分）
  2. FNO 模型结构：Fourier layers + Pointwise MLP
  3. 损失函数定义（MSE、相对误差等）
  4. 训练循环（Adam、LR scheduler）与评估
  5. 结果可视化：解对比图、误差热力图、训练损失曲线

## 运行教程（快速开始）

### 环境要求

```bash
pip install torch numpy matplotlib jupyter scipy
```

### 运行 Notebook

```bash
jupyter notebook FNO/tutorial/FNO_tutorial.ipynb
```

按顺序运行单元格来重现实验。可在 notebook 中调整如下超参数：
- `modes`（保留的频率数）
- `width`（通道数）
- `learning_rate`、`batch_size`、`epochs` 等

## 常见问题与建议

- FNO 在非周期域问题中需要小心处理边界（可以通过扩展/填充或局部修正）。
- 对于较高频的解，建议保留更多 `modes` 或更高的 `width`。
- 若训练出现不稳定，可尝试减小学习率或增加 L2 正则化。

## 扩展方向

- 将 FNO 与 PINN 或 TNN 等方法比较：在样本成本、参数量和泛化性能方面的差异。 
- 试验不同的频谱算子参数化方式（如低频/高频分段权重）。
- 将 FNO 应用到随机 PDE 或不规则网格数据（结合图神经网络/插值）。

## 参考文献

- Zongyi Li et al., "Fourier Neural Operator for Parametric Partial Differential Equations", ICLR 2021.

## 作者

AI4CFD 项目组
