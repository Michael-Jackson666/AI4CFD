# Transformer for PDE - 快速开始指南

欢迎使用Transformer求解偏微分方程！本指南将帮助你在5分钟内运行你的第一个示例。

## 🚀 三步开始

### 第一步：安装依赖

```bash
pip install torch numpy matplotlib jupyter
```

### 第二步：选择你的学习方式

#### 方式A：交互式教程（推荐）

```bash
# 启动Jupyter Notebook
jupyter notebook transformer_tutorial.ipynb
```

按照notebook中的步骤，逐步学习和实验。

#### 方式B：命令行快速训练

```bash
# 使用默认参数训练
python train_simple.py

# 或者自定义参数
python train_simple.py --epochs 100 --batch_size 32 --lr 0.001
```

### 第三步：查看结果

训练完成后，在`results/`目录下查看：
- `loss_curve.png` - 训练曲线
- `predictions.png` - 预测结果
- `best_model.pth` - 最佳模型

## 📊 示例输出

训练过程输出示例：
```
使用设备: cpu
生成训练数据...
训练集: (800, 64), 测试集: (200, 64)

创建模型...
模型参数量: 387,329

开始训练...

Epoch [10/100] - Train Loss: 0.002456, Test Loss: 0.002678, LR: 0.000951
Epoch [20/100] - Train Loss: 0.000854, Test Loss: 0.000932, LR: 0.000809
Epoch [30/100] - Train Loss: 0.000432, Test Loss: 0.000487, LR: 0.000588
...

训练完成! 最佳测试损失: 0.000312
```

## 🎯 参数说明

### 训练参数

```bash
python train_simple.py \
    --epochs 100 \          # 训练轮数
    --batch_size 32 \       # 批次大小
    --lr 0.001 \           # 学习率
    --d_model 128 \        # 模型维度
    --nhead 4 \            # 注意力头数
    --num_layers 4 \       # Transformer层数
    --train_samples 800 \  # 训练样本数
    --test_samples 200 \   # 测试样本数
    --nx 64 \              # 空间离散点数
    --save_dir ./results   # 结果保存目录
```

### 推荐配置

**快速测试**（1分钟）:
```bash
python train_simple.py --epochs 20 --train_samples 200 --d_model 64
```

**标准训练**（5分钟）:
```bash
python train_simple.py --epochs 100 --train_samples 800 --d_model 128
```

**高精度**（15分钟）:
```bash
python train_simple.py --epochs 200 --train_samples 2000 --d_model 256 --num_layers 6
```

## 💡 常见问题

### Q: 如何使用GPU？
A: 代码会自动检测GPU。确保安装了CUDA版本的PyTorch：
```bash
pip install torch --index-url https://download.pytorch.org/whl/cu118
```

### Q: 内存不够怎么办？
A: 减小batch_size或d_model：
```bash
python train_simple.py --batch_size 16 --d_model 64
```

### Q: 如何提高精度？
A: 增加模型容量和训练时间：
```bash
python train_simple.py --epochs 200 --d_model 256 --num_layers 6 --train_samples 2000
```

### Q: 如何可视化中间结果？
A: 使用交互式notebook：
```bash
jupyter notebook transformer_tutorial.ipynb
```

## 📚 下一步

- ✅ 完成快速开始
- 📖 阅读完整教程：`transformer_tutorial.ipynb`
- 🔬 查看详细文档：`README.md`
- 🚀 探索父目录的高级模型：`../models.py`
- 💡 尝试其他PDE问题（修改数据生成函数）

## 🎓 学习路径

```
快速开始（你在这里）
    ↓
完整教程（transformer_tutorial.ipynb）
    ↓
高级特性（Vision Transformer, Physics-Informed）
    ↓
实际应用（自己的PDE问题）
```

## 📁 文件结构

```
totorial/
├── QUICKSTART.md              ← 你在这里
├── README.md                  ← 详细文档
├── transformer_tutorial.ipynb ← 交互式教程
├── train_simple.py            ← 快速训练脚本
└── results/                   ← 训练结果（自动创建）
    ├── best_model.pth
    ├── loss_curve.png
    └── predictions.png
```

## 🎉 成功标志

如果你看到类似下面的输出，恭喜你成功了！

```
训练完成! 最佳测试损失: 0.000312

=== 误差统计 ===
相对误差 - 平均: 0.008234
相对误差 - 标准差: 0.004156
相对误差 - 最小: 0.001234
相对误差 - 最大: 0.023456

所有结果已保存到 ./results/
```

## 🆘 获取帮助

- 查看详细文档：`README.md`
- 运行帮助命令：`python train_simple.py --help`
- 提交Issue：[GitHub Issues](https://github.com/Michael-Jackson666/AI4CFD/issues)

---

**现在就开始你的Transformer PDE之旅吧！🚀**
