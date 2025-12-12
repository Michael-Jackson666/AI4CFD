# DeepONet Tutorial

本教程位于 `DeepONet/tutorial`，核心文件：

- `operator_learning_torch.ipynb`：一个基于 PyTorch 的交互式 Jupyter 教程，展示如何训练与评估 DeepONet，用于算子学习（operator learning）。

## 教程目标

本教程演示：

1. DeepONet 的基本思想和结构（branch / trunk 网络）
2. 如何准备训练数据并对算子进行训练
3. 损失函数、训练流程（Adam / LR 调整等）以及评估指标
4. 推理（inference）和结果可视化

## 算法背景（简要）

DeepONet 将一个算子 $\mathcal{G}: u(\cdot) \mapsto v(\cdot)$ 近似为：

$$
\mathcal{G}(u)(y) \approx \sum_{k=1}^{p} b_k(u) \, t_k(y)
$$

其中：
- $b_k(u)$ 由 `branch` 网络从函数 $u$ 的离散表示中学习得到；
- $t_k(y)$ 由 `trunk` 网络以 $y$ 为输入学习得到；
- $p$ 为输出的合成秩（branch-trunk 投影维度）。

典型训练目标是最小化平方残差：

$$
\min_{\theta} \, \frac{1}{N} \sum_{i=1}^N \|\mathcal{G}(u_i)(\cdot) - v_i(\cdot)\|_2^2.
$$

## 使用说明

### 依赖

```bash
pip install torch numpy matplotlib jupyter
```

### 运行教程

1. 打开 Jupyter Notebook：

```bash
jupyter notebook DeepONet/tutorial/operator_learning_torch.ipynb
```

2. 按顺序运行单元格，必要时调整超参数（学习率、batch size、网络结构等）。

## 输出与可视化

- 训练过程的 loss 曲线
- 测试集上的预测 vs 参考解对比图
- 相对误差与 L2 误差统计

## 建议与扩展

- 可将教程中的简单算子替换为更复杂的 PDE 解算子，检验泛化能力
- 将 `branch` 与 `trunk` 网络替换为不同架构（CNN、ResNet）以提升表示能力
- 使用更大规模的数据集与更长训练时间以评估收敛性

## 作者

本教程由 AI4CFD 项目提供，若需贡献或改进教程，请提交 PR 并附上可复现的结果和说明。
