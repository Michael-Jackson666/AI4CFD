# 模型架构选择指南
# Model Architecture Selection Guide

## 🎯 快速开始

在 `main.py` 中，只需要修改一行代码即可切换模型：

```python
'model_type': 'mlp',  # 改成你想要的模型类型
```

## 📦 可用的模型架构

### 1. **MLP** (Multi-Layer Perceptron)
传统的多层感知机，PINN 的经典架构。

**使用方式：**
```python
configuration = {
    'model_type': 'mlp',
    'nn_layers': 8,      # 隐藏层数
    'nn_neurons': 128,   # 每层神经元数
    # ... 其他配置
}
```

**特点：**
- ✅ 训练速度快
- ✅ 稳定可靠
- ✅ 参数量较小
- ✅ 适合大多数问题

**适用场景：**
- 快速原型开发
- 标准 PINN 问题
- 计算资源有限

**参数量示例：**
- 8 层 × 128 神经元 ≈ 133K 参数
- 12 层 × 256 神经元 ≈ 788K 参数

---

### 2. **Transformer**
基于注意力机制的 Transformer 架构。

**使用方式：**
```python
configuration = {
    'model_type': 'transformer',
    'd_model': 256,                    # 嵌入维度
    'nhead': 8,                        # 注意力头数
    'num_transformer_layers': 6,       # Transformer 层数
    'dim_feedforward': 1024,           # 前馈网络维度
    'dropout': 0.1,                    # Dropout 率
    # ... 其他配置
}
```

**特点：**
- ✅ 能捕捉长程依赖关系
- ✅ 自注意力机制
- ✅ 适合复杂模式
- ⚠️ 参数量大
- ⚠️ 训练较慢

**适用场景：**
- 复杂的物理现象
- 需要捕捉全局相关性
- 有足够的计算资源

**参数量示例：**
- 标准配置 (d=256, h=8, l=6) ≈ 2.5M 参数

---

### 3. **Lightweight Transformer**
轻量级 Transformer，参数量更少，训练更快。

**使用方式：**
```python
configuration = {
    'model_type': 'lightweight_transformer',
    'd_model': 128,                    # 较小的嵌入维度
    'nhead': 4,                        # 较少的注意力头
    'num_transformer_layers': 3,       # 较少的层数
    'dim_feedforward': 512,            # 较小的前馈维度
    'dropout': 0.1,
    # ... 其他配置
}
```

**特点：**
- ✅ Transformer 的优点
- ✅ 训练速度更快
- ✅ 参数量适中
- ⚠️ 表达能力略弱于标准 Transformer

**适用场景：**
- 想尝试 Transformer 但资源有限
- 快速实验
- 中等复杂度问题

**参数量示例：**
- 轻量配置 (d=128, h=4, l=3) ≈ 600K 参数

---

### 4. **Hybrid Transformer**
混合架构，结合 Transformer 和 MLP 的优势。

**使用方式：**
```python
configuration = {
    'model_type': 'hybrid_transformer',
    'd_model': 256,                    # Transformer 嵌入维度
    'nhead': 8,                        # 注意力头数
    'num_transformer_layers': 4,       # Transformer 层数
    'num_mlp_layers': 4,               # MLP 分支层数
    'mlp_neurons': 512,                # MLP 每层神经元数
    'dropout': 0.1,
    # ... 其他配置
}
```

**特点：**
- ✅ 结合两种架构的优势
- ✅ Transformer 捕捉全局特征
- ✅ MLP 捕捉局部特征
- ⚠️ 参数量最大
- ⚠️ 训练最慢

**适用场景：**
- 最复杂的问题
- 需要同时捕捉全局和局部特征
- 追求最高精度
- 有充足的计算资源

**参数量示例：**
- 混合配置 ≈ 3M+ 参数

---

## 📊 模型对比表

| 模型类型 | 参数量 | 训练速度 | 表达能力 | 内存占用 | 推荐场景 |
|---------|-------|---------|---------|---------|---------|
| MLP | ⭐ 小 | ⭐⭐⭐ 快 | ⭐⭐ 中 | ⭐ 低 | 快速原型、标准问题 |
| Lightweight Transformer | ⭐⭐ 中 | ⭐⭐ 中 | ⭐⭐⭐ 强 | ⭐⭐ 中 | 实验、中等复杂度 |
| Transformer | ⭐⭐⭐ 大 | ⭐ 慢 | ⭐⭐⭐⭐ 很强 | ⭐⭐⭐ 高 | 复杂问题、全局相关性 |
| Hybrid Transformer | ⭐⭐⭐⭐ 很大 | ⭐ 很慢 | ⭐⭐⭐⭐⭐ 最强 | ⭐⭐⭐⭐ 很高 | 最复杂问题、高精度 |

---

## 💡 使用建议

### 第一次使用
```python
'model_type': 'mlp',
'nn_layers': 8,
'nn_neurons': 128,
'epochs': 500,  # 快速测试
```

### 标准训练
```python
'model_type': 'mlp',
'nn_layers': 12,
'nn_neurons': 256,
'epochs': 2000,
```

### 尝试 Transformer
```python
'model_type': 'lightweight_transformer',
'd_model': 128,
'nhead': 4,
'num_transformer_layers': 3,
'epochs': 1000,
```

### 高精度训练
```python
'model_type': 'transformer',  # 或 'hybrid_transformer'
'd_model': 256,
'nhead': 8,
'num_transformer_layers': 6,
'epochs': 5000,
'learning_rate': 5e-5,  # 降低学习率
```

---

## 🔧 常见配置组合

### 配置 1: 快速测试 (5-10 分钟)
```python
configuration = {
    'model_type': 'mlp',
    'nn_layers': 6,
    'nn_neurons': 64,
    'epochs': 500,
    'plot_dir': 'quick_test'
}
```

### 配置 2: 标准 MLP (30 分钟)
```python
configuration = {
    'model_type': 'mlp',
    'nn_layers': 8,
    'nn_neurons': 128,
    'epochs': 2000,
    'plot_dir': 'mlp_standard'
}
```

### 配置 3: 轻量 Transformer (45 分钟)
```python
configuration = {
    'model_type': 'lightweight_transformer',
    'd_model': 128,
    'nhead': 4,
    'num_transformer_layers': 3,
    'epochs': 1500,
    'plot_dir': 'transformer_light'
}
```

### 配置 4: 标准 Transformer (2 小时)
```python
configuration = {
    'model_type': 'transformer',
    'd_model': 256,
    'nhead': 8,
    'num_transformer_layers': 6,
    'epochs': 2000,
    'learning_rate': 5e-5,
    'plot_dir': 'transformer_standard'
}
```

### 配置 5: 混合模型 - 高精度 (4+ 小时)
```python
configuration = {
    'model_type': 'hybrid_transformer',
    'd_model': 256,
    'nhead': 8,
    'num_transformer_layers': 4,
    'num_mlp_layers': 4,
    'mlp_neurons': 512,
    'epochs': 5000,
    'learning_rate': 5e-5,
    'plot_dir': 'hybrid_high_precision'
}
```

---

## 📈 性能调优建议

### 如果训练不稳定：
1. 降低学习率：`'learning_rate': 5e-5` 或 `1e-5`
2. 增加 dropout：`'dropout': 0.2`
3. 使用梯度裁剪（已内置）

### 如果过拟合：
1. 增加 dropout：`'dropout': 0.2` 或 `0.3`
2. 减少模型复杂度（减少层数或神经元数）
3. 增加正则化权重

### 如果欠拟合：
1. 增加模型容量（更多层或神经元）
2. 增加训练轮数
3. 降低学习率，训练更久

---

## 🚀 快速切换示例

**在 `main.py` 中，找到配置部分，取消注释相应的预设：**

```python
# ============================================================
# QUICK CONFIGURATION PRESETS (uncomment to use)
# ============================================================

# Preset 1: Standard MLP (default, fast training)
configuration['model_type'] = 'mlp'
configuration['nn_layers'] = 8
configuration['nn_neurons'] = 128

# # Preset 3: Standard Transformer (good for complex patterns)
# configuration['model_type'] = 'transformer'
# configuration['d_model'] = 256
# configuration['nhead'] = 8
# configuration['num_transformer_layers'] = 6

# # Preset 5: Hybrid Model (combines both approaches)
# configuration['model_type'] = 'hybrid_transformer'
# configuration['d_model'] = 256
# configuration['nhead'] = 8
# configuration['num_transformer_layers'] = 4
# configuration['num_mlp_layers'] = 4
```

只需要注释掉当前的，取消注释你想用的即可！

---

## 📝 运行命令

```bash
cd /Users/jack/Desktop/ML/AI4CFD/PINNs/vp_system
python main.py
```

模型会自动根据 `model_type` 选择对应的架构并开始训练！

---

**提示**: 建议先用 MLP 快速测试，确保代码正常运行，然后再尝试 Transformer 架构！
