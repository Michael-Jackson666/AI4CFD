# Vlasov-Poisson PINN 求解器

基于物理信息神经网络(PINNs)的 1D Vlasov-Poisson 系统求解器，支持多种神经网络架构（MLP、Transformer 等），并具有完整的配置追踪功能。

## ✨ 主要特性

- 🎯 **多架构支持**: MLP、Transformer、Lightweight Transformer、Hybrid Transformer
- 💾 **配置自动保存**: 每次训练自动保存完整配置（JSON + TXT）
- 🔍 **配置对比工具**: 轻松对比不同实验的参数设置
- 📊 **归一化输入**: 改进的训练稳定性
- 📈 **可视化**: 自动生成相空间演化图和损失曲线
- 🚀 **简单易用**: 一行代码切换模型架构

---

## 🚀 快速开始

### 1. 安装依赖

```bash
pip install torch numpy matplotlib
```

### 2. 运行训练

```bash
cd /Users/jack/Desktop/ML/AI4CFD/PINNs/vp_system
python main.py
```

### 3. 切换模型架构

在 `main.py` 中修改一行代码：

```python
'model_type': 'mlp',  # 改成 'transformer', 'lightweight_transformer', 或 'hybrid_transformer'
```

就这么简单！🎉

---

## 📦 可用的模型架构

### 1. MLP (默认) - 快速稳定

```python
configuration = {
    'model_type': 'mlp',
    'nn_layers': 8,      # 隐藏层数
    'nn_neurons': 128,   # 每层神经元数
}
```

**特点**: 
- ✅ 训练快（~20分钟/1000 epochs）
- ✅ 参数少（~133K）
- ✅ 稳定可靠

**适用**: 快速原型、标准问题

---

### 2. Lightweight Transformer - 平衡选择

```python
configuration = {
    'model_type': 'lightweight_transformer',
    'd_model': 128,
    'nhead': 4,
    'num_transformer_layers': 3,
}
```

**特点**:
- ✅ Transformer 优势
- ✅ 训练较快（~30分钟）
- ✅ 参数适中（~600K）

**适用**: 实验探索、中等复杂度

---

### 3. Standard Transformer - 高性能

```python
configuration = {
    'model_type': 'transformer',
    'd_model': 256,
    'nhead': 8,
    'num_transformer_layers': 6,
}
```

**特点**:
- ✅ 强大表达能力
- ✅ 捕捉全局特征
- ⚠️ 训练慢（~45分钟）
- ⚠️ 参数多（~2.5M）

**适用**: 复杂问题、高精度需求

---

### 4. Hybrid Transformer - 最强组合

```python
configuration = {
    'model_type': 'hybrid_transformer',
    'd_model': 256,
    'nhead': 8,
    'num_transformer_layers': 4,
    'num_mlp_layers': 4,
}
```

**特点**:
- ✅ 全局+局部特征
- ✅ 最高精度
- ⚠️ 最慢（~60分钟+）
- ⚠️ 参数最多（~3M+）

**适用**: 最复杂问题、追求极致精度

---

## 📊 模型对比表

| 模型 | 参数量 | 训练时间 | GPU内存 | 推荐场景 |
|-----|-------|---------|---------|----------|
| **MLP** | 133K | ~20分钟 | 低 | 日常使用、快速测试 |
| **Lightweight Transformer** | 600K | ~30分钟 | 中 | 实验探索 |
| **Standard Transformer** | 2.5M | ~45分钟 | 高 | 复杂问题 |
| **Hybrid Transformer** | 3M+ | ~60分钟+ | 很高 | 最高精度 |

*基于 1000 epochs 的大致时间*

---

## 🔧 配置保存与追踪

### 自动保存配置

每次训练会自动生成：

```
plot_dir/
├── training_config.json    # JSON 格式（程序可读）
├── training_config.txt     # 文本格式（人类可读）
├── training_log.txt        # 训练日志
├── loss_history.png        # 损失曲线
└── results_epoch_*.png     # 相空间演化图
```

### 查看配置

```bash
# 查看文本配置
cat 2025/10/13/2/training_config.txt

# 查看 JSON 配置
cat 2025/10/13/2/training_config.json
```

### 对比不同实验

```bash
# 交互式对比工具
python compare_configs.py

# 列出所有配置
python compare_configs.py list

# 对比两个配置
python compare_configs.py compare config1.json config2.json
```

**示例输出**:

```
[Model]
  model_type          | mlp              | transformer          
  nn_layers           | 8                | N/A                  
  d_model             | N/A              | 256                  

[Training]
  epochs              | 2000             | 5000                 
  learning_rate       | 0.0001           | 5e-05                
```

---

## 📖 完整配置示例

### 示例 1: 快速测试（5-10分钟）

```python
configuration = {
    # 模型
    'model_type': 'mlp',
    'nn_layers': 6,
    'nn_neurons': 64,
    
    # 训练
    'epochs': 500,
    'learning_rate': 1e-4,
    
    # 输出
    'plot_dir': 'quick_test',
    'log_frequency': 50,
    'plot_frequency': 100,
}
```

### 示例 2: 标准训练（30分钟）

```python
configuration = {
    # 模型
    'model_type': 'mlp',
    'nn_layers': 8,
    'nn_neurons': 128,
    
    # 训练
    'epochs': 2000,
    'learning_rate': 1e-4,
    'n_pde': 70000,
    'n_ic': 1100,
    'n_bc': 1100,
    
    # 损失权重
    'weight_pde': 1.0,
    'weight_ic': 5.0,
    'weight_bc': 10.0,
    
    # 输出
    'plot_dir': 'experiments/mlp_standard',
}
```

### 示例 3: Transformer 高精度（2-4小时）

```python
configuration = {
    # 模型
    'model_type': 'transformer',
    'd_model': 256,
    'nhead': 8,
    'num_transformer_layers': 6,
    'dim_feedforward': 1024,
    'dropout': 0.1,
    
    # 训练
    'epochs': 5000,
    'learning_rate': 5e-5,  # 更小的学习率
    'n_pde': 100000,        # 更多采样点
    
    # 输出
    'plot_dir': 'experiments/transformer_high_precision',
}
```

---

## 📁 文件结构

```
vp_system/
├── main.py                     # 主训练脚本 ⭐
├── vp_pinn.py                  # PINN 求解器核心
├── mlp.py                      # MLP 模型定义
├── transformer.py              # Transformer 模型定义
├── visualization.py            # 可视化函数
├── compare_configs.py          # 配置对比工具
├── compare_models.py           # 模型对比实验脚本
└── README.md                   # 本文档
```

---

## 🎯 使用工作流

### 1. 快速测试

```bash
# 使用默认 MLP，500 epochs
python main.py
```

修改 `main.py` 中的 `epochs` 为 500 进行快速测试。

### 2. 标准训练

```python
# main.py 中保持默认配置
'model_type': 'mlp',
'epochs': 2000,
```

```bash
python main.py
```

### 3. 尝试 Transformer

```python
# main.py 中修改
'model_type': 'lightweight_transformer',
'epochs': 1500,
```

```bash
python main.py
```

### 4. 对比实验

```bash
# 运行对比脚本
python compare_models.py

# 选择：
# 1. 对比不同架构
# 2. 对比不同激活函数
# 3. 对比不同网络规模
```

### 5. 分析结果

```bash
# 查看训练日志
cat experiments/*/training_log.txt

# 对比配置
python compare_configs.py

# 查看可视化结果
# 打开 plot_dir 中的 PNG 图片
```

---

## 📚 方程组说明

### 1D Vlasov-Poisson 系统

**Vlasov 方程** (描述粒子分布演化):
$$\frac{\partial f}{\partial t} + v \frac{\partial f}{\partial x} - E(x,t) \frac{\partial f}{\partial v} = 0$$

**Poisson 方程** (电场与密度关系):
$$\frac{\partial E}{\partial x} = n_e(x,t) - 1$$

其中:
- $f(t,x,v)$: 粒子分布函数
- $E(x,t)$: 电场
- $n_e(x,t) = \int f(t,x,v) dv$: 电子密度

### 初始条件: 双流不稳定性

$$f(0,x,v) = \frac{1}{2\sigma\sqrt{2\pi}} \left[e^{-(v-v_b)^2/(2\sigma^2)} + e^{-(v+v_b)^2/(2\sigma^2)}\right] \times [1 + A\cos(kx)]$$

参数:
- $v_b = 1.0$: 束流速度
- $\sigma = 0.5$: 热速度
- $A = 0.1$: 扰动幅度
- $k = 2\pi/L_x$: 波数

---

## 🔍 配置参数说明

### 域参数

```python
't_max': 62.5,      # 最大时间
'x_max': 10.0,      # 空间域长度  
'v_max': 5.0,       # 最大速度
```

### 物理参数

```python
'beam_v': 1.0,          # 束流速度
'thermal_v': 0.5,       # 热速度
'perturb_amp': 0.1,     # 扰动幅度
```

### 训练参数

```python
'epochs': 2000,         # 训练轮数
'learning_rate': 1e-4,  # 学习率
'n_pde': 70000,         # PDE采样点数
'n_ic': 1100,           # 初始条件点数
'n_bc': 1100,           # 边界条件点数
```

### 损失权重

```python
'weight_pde': 1.0,      # PDE损失权重
'weight_ic': 5.0,       # 初始条件权重
'weight_bc': 10.0,      # 边界条件权重
```

---

## 🛠️ 常见问题

### Q1: 训练不稳定怎么办？

**方案**:
1. 降低学习率: `'learning_rate': 5e-5`
2. 增加 dropout (Transformer): `'dropout': 0.2`
3. 减少模型规模

### Q2: 如何提高精度？

**方案**:
1. 增加训练轮数: `'epochs': 5000`
2. 使用更大模型: `'transformer'` 或 `'hybrid_transformer'`
3. 增加采样点: `'n_pde': 100000`
4. 调整损失权重

### Q3: 训练太慢怎么办？

**方案**:
1. 使用 MLP: `'model_type': 'mlp'`
2. 减少采样点: `'n_pde': 50000`
3. 使用 GPU 加速
4. 降低可视化频率: `'plot_frequency': 1000`

### Q4: 如何对比不同模型？

**方案**:
```bash
python compare_models.py
```
选择"对比不同架构"，自动运行并对比结果。

---

## 📈 性能调优建议

### 过拟合
- 增加 dropout
- 减少模型复杂度
- 增加正则化权重

### 欠拟合
- 增加模型容量
- 增加训练轮数
- 降低学习率，训练更久

### 不稳定
- 降低学习率
- 使用梯度裁剪（已内置）
- 检查初始条件

---

## 📝 输出文件说明

每次训练生成的文件：

```
plot_dir/
├── training_config.json        # 配置（JSON格式）
├── training_config.txt         # 配置（文本格式）
├── training_log.txt           # CSV格式训练日志
│   格式: Epoch,Total_Loss,PDE_Loss,IC_Loss,BC_Loss,Time_s
├── loss_history.png           # 损失曲线图
└── results_epoch_XXXX.png     # 周期性结果图
    ├── 相空间演化（3个时间步）
    ├── 初始条件对比
    ├── 密度分布
    └── 电场分布
```

---

## 🎓 使用示例

### 示例 1: 基础使用

```python
# main.py 中使用默认配置
python main.py
```

### 示例 2: 切换到 Transformer

```python
# 修改 main.py
configuration['model_type'] = 'transformer'
configuration['d_model'] = 256
configuration['nhead'] = 8
configuration['num_transformer_layers'] = 6
```

### 示例 3: 批量实验

```bash
# 使用对比脚本
python compare_models.py

# 选择选项 1: 对比不同架构
# 会自动运行 MLP、Lightweight Transformer、Standard Transformer
```

### 示例 4: 查看和对比结果

```bash
# 列出所有实验
python compare_configs.py list

# 对比两个实验
python compare_configs.py
# 选择 2: 对比两个配置
```

---

## 🌟 快速参考

| 任务 | 命令/操作 |
|-----|----------|
| 运行训练 | `python main.py` |
| 切换模型 | 修改 `'model_type'` |
| 对比模型 | `python compare_models.py` |
| 查看配置 | `python compare_configs.py` |
| 快速测试 | 设置 `'epochs': 500` |

---

## 💡 最佳实践

1. **首次使用**: 先用 MLP + 500 epochs 快速测试
2. **实验命名**: 使用有意义的 `plot_dir` 名称
3. **记录笔记**: 在输出目录添加 `notes.txt`
4. **定期对比**: 用 `compare_configs.py` 追踪改进
5. **保存好结果**: 备份效果好的配置和模型

---

## 🔗 相关文档

- [模型架构详细说明](MODEL_SELECTION_GUIDE.md) - 已删除，内容已整合
- [配置追踪指南](CONFIG_TRACKING_GUIDE.md) - 已删除，内容已整合

---

## 📧 支持

遇到问题？
1. 查看本 README
2. 运行 `python compare_configs.py` 检查配置
3. 查看训练日志 `training_log.txt`

---

**开始你的 Vlasov-Poisson PINN 训练之旅！** 🚀

```bash
python main.py
```
