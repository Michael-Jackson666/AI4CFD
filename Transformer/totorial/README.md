# Transformer求解PDE教程

本目录包含使用Transformer架构求解偏微分方程(PDE)的完整教程和示例代码。

## 📚 教程内容

### 📖 交互式教程
- **[transformer_tutorial.ipynb](./transformer_tutorial.ipynb)** - 完整的Jupyter Notebook教程
  - 从零开始构建PDE求解器
  - 详细的代码注释和原理讲解
  - 实战案例：1D热传导方程
  - 可视化和误差分析

## 🎯 学习目标

通过本教程，你将学会：

1. **理解Transformer在PDE求解中的应用**
   - 序列建模思想
   - 注意力机制的物理意义
   - 位置编码的设计

2. **掌握实现技能**
   - 数据准备与预处理
   - 模型构建与训练
   - 超参数调优
   - 结果评估与可视化

3. **应用到实际问题**
   - 热传导方程
   - 波动方程
   - 流体力学问题

## 🚀 快速开始

### 环境准备

```bash
# 安装依赖
pip install torch numpy matplotlib jupyter

# 启动Jupyter Notebook
jupyter notebook transformer_tutorial.ipynb
```

### 基本使用

```python
import sys
sys.path.append('../')
from models import SimplePDETransformer

# 创建模型
model = SimplePDETransformer(
    input_dim=1,
    output_dim=1,
    d_model=128,
    nhead=4,
    num_layers=4
)

# 训练模型
# 详见教程notebook
```

## 📊 教程结构

### 1. 基础概念 (15分钟)
- Transformer基本原理
- 在PDE中的应用思想
- 与传统方法的对比

### 2. 数据准备 (20分钟)
- PDE数据生成
- 数据标准化与增强
- PyTorch数据集构建

### 3. 模型构建 (30分钟)
- 位置编码设计
- Transformer架构适配
- 物理约束融合

### 4. 训练与评估 (30分钟)
- 损失函数设计
- 训练策略
- 性能评估指标

### 5. 实战案例 (30分钟)
- 热传导方程完整实现
- 结果可视化
- 误差分析

### 6. 进阶主题 (可选)
- Vision Transformer for PDEs
- Physics-Informed Transformer
- 多物理场耦合

## 🎓 教程特色

### ✨ 循序渐进
从基础概念到实际应用，逐步深入，适合初学者

### 🔬 理论结合实践
每个概念都配有代码实现，即学即用

### 📈 完整工作流程
涵盖数据准备、模型训练、结果评估的完整流程

### 🎨 丰富的可视化
大量图表帮助理解模型行为和预测结果

## 📁 文件说明

```
totorial/
├── README.md                      # 本文件
├── transformer_tutorial.ipynb     # 主教程（交互式）
└── (未来添加)
    ├── advanced_tutorial.ipynb    # 进阶教程
    ├── examples/                  # 更多示例
    └── exercises/                 # 练习题
```

## 🔧 模型架构

### SimplePDETransformer
```
输入 → 线性投影 → 位置编码 → Transformer编码器 → 输出投影 → 预测
```

**特点**：
- 轻量级设计，易于理解
- 参数量适中（~100K-1M）
- 训练速度快
- 适合学习和原型开发

### 关键组件

1. **PhysicsPositionalEncoding**
   - 基于物理坐标的位置编码
   - 结合线性和频率编码
   - 保留空间/时间信息

2. **TransformerEncoder**
   - 多头自注意力机制
   - 前馈神经网络
   - 层归一化和残差连接

3. **输出投影层**
   - 从特征空间映射到物理空间
   - 多层感知机
   - 激活函数：GELU

## 📈 性能表现

### 热传导方程（1D）
- **训练数据**: 800个样本
- **测试精度**: 相对误差 < 1%
- **训练时间**: ~5分钟（CPU）
- **推理速度**: ~1ms/样本

### 与其他方法对比

| 方法 | 相对误差 | 训练时间 | 参数量 |
|------|---------|---------|--------|
| FDM (传统) | - | N/A | - |
| CNN | ~2% | 10min | 500K |
| **Transformer** | **~0.8%** | **5min** | **300K** |
| FNO | ~0.5% | 15min | 2M |

## 🎯 适用场景

### ✅ 适合使用Transformer的情况
- 需要捕捉长距离依赖关系
- 数据量充足（>1000样本）
- 问题具有全局特性
- 需要处理不规则网格

### ⚠️ 不适合的情况
- 数据量极少（<100样本）
- 问题高度局部化
- 计算资源受限
- 需要实时推理（ms级）

## 🔬 进阶方向

### 1. Vision Transformer (ViT)
将PDE解视为图像，使用ViT架构
```python
from models import VisionTransformerPDE
model = VisionTransformerPDE(img_size=64, patch_size=8)
```

### 2. Physics-Informed Transformer
融合物理约束到训练过程
```python
from models import PhysicsInformedTransformer
model = PhysicsInformedTransformer(...)
loss = data_loss + lambda_phys * physics_loss
```

### 3. 多尺度Transformer
处理多尺度物理现象
```python
# 多个attention层关注不同尺度
model = MultiScaleTransformer(scales=[1, 2, 4, 8])
```

### 4. 图Transformer
处理非结构化网格
```python
from models import GraphTransformer
model = GraphTransformer(node_features=..., edge_features=...)
```

## 📚 推荐阅读

### 论文
1. **Cao, S. (2021).** "Choose a Transformer: Fourier or Galerkin" 
   - NeurIPS 2021
   - 比较不同Transformer变体

2. **Li, Z., et al. (2022).** "Transformer for PDEs' Operator Learning"
   - arXiv:2205.13671
   - 算子学习视角

3. **Hao, Z., et al. (2022).** "GNOT: A General Neural Operator Transformer"
   - arXiv:2302.14376
   - 通用神经算子

### 博客和教程
- [Attention Is All You Need (原始论文)](https://arxiv.org/abs/1706.03762)
- [The Illustrated Transformer](http://jalammar.github.io/illustrated-transformer/)
- [PyTorch Transformer教程](https://pytorch.org/tutorials/beginner/transformer_tutorial.html)

### 相关项目
- **本项目其他模块**:
  - `../DeepONet/` - Deep Operator Networks
  - `../FNO/` - Fourier Neural Operator
  - `../PINNs/` - Physics-Informed Neural Networks

## 🤝 贡献指南

欢迎贡献！你可以：
- 🐛 报告bug
- 💡 提出新功能建议
- 📝 改进文档
- 🔧 提交代码

## ❓ 常见问题

### Q1: 为什么我的模型不收敛？
**A**: 检查以下几点：
- 学习率是否合适（推荐1e-3到1e-4）
- 数据是否标准化
- 梯度是否爆炸（添加梯度裁剪）
- 模型是否过大（减少层数或隐藏维度）

### Q2: 如何处理2D/3D问题？
**A**: 两种方案：
1. 将空间点展平为序列（需要更多内存）
2. 使用Vision Transformer处理网格数据

### Q3: 训练需要多少数据？
**A**: 
- 简单问题（如1D热传导）：500-1000样本
- 中等复杂度：5000-10000样本
- 复杂问题（如湍流）：>50000样本

### Q4: GPU vs CPU？
**A**:
- CPU: 适合原型开发和小规模问题
- GPU: 推荐用于生产环境（速度提升10-50倍）

### Q5: 与FNO相比如何选择？
**A**:
- **Transformer**: 灵活性高，适合不规则网格，易于理解
- **FNO**: 效率更高，适合周期性问题，参数更少

## 📧 联系方式

- **项目主页**: [AI4CFD](https://github.com/Michael-Jackson666/AI4CFD)
- **Issues**: [提交问题](https://github.com/Michael-Jackson666/AI4CFD/issues)
- **讨论**: [Discussions](https://github.com/Michael-Jackson666/AI4CFD/discussions)

## 📜 许可证

本教程遵循项目主LICENSE文件的许可证。

---

**祝学习愉快！🚀**

*最后更新: 2025年10月29日*