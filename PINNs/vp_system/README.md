# Vlasov-Poisson PINN 求解器

基于物理信息神经网络(PINNs)的 1D Vlasov-Poisson 系统求解器，支持多种神经网络架构（MLP、Transformer 等），并具有完整的配置追踪功能。

## ✨ 主要特性

- 🎯 **多架构支持**: MLP、Transformer、Lightweight Transformer、Hybrid Transformer
- � **可配置初始条件**: 支持 Two-Stream、Landau Damping、Single Beam 等多种物理场景
- �💾 **配置自动保存**: 每次训练自动保存完整配置（JSON + TXT）
- 🔍 **配置对比工具**: 轻松对比不同实验的参数设置
- 📊 **归一化输入**: 改进的训练稳定性
- 📈 **可视化**: 自动生成相空间演化图和损失曲线
- 🚀 **简单易用**: 通过配置文件轻松切换模型和初始条件

## 🆕 最新更新 (2025-11-02)

### 初始条件完全配置化 ✨
初始条件已完全移至 `config.py`，实现真正的科学实验工作流程！

**支持的物理场景**:
- ✅ **Two-Stream Instability** (双流不稳定性): 两束反向电子束的不稳定增长
- ✅ **Landau Damping** (Landau阻尼): 等离子体波的动理学阻尼
- ✅ **Single Beam** (单束流): 单个电子束传播
- ✅ **Custom** (自定义): 完全自定义的初始条件函数

**快速切换示例**:
```python
# 在 config.py 中取消注释即可切换
use_ic_preset('landau_damping')      # Landau 阻尼
use_ic_preset('two_stream_strong')   # 强双流不稳定性
use_ic_preset('two_stream_weak')     # 弱双流不稳定性
use_ic_preset('single_beam')         # 单束流
```

**测试初始条件**:
```bash
python test_initial_conditions.py  # 生成所有初始条件的可视化
```

---

## 🚀 快速开始

### 1. 安装依赖

```bash
pip install torch numpy matplotlib
```

### 2. 配置训练参数

编辑 `config.py` 文件，所有参数都在这里配置：

```python
# 选择模型架构
MODEL_TYPE = 'hybrid_transformer'  # 或 'mlp', 'transformer', 'lightweight_transformer'

# 选择初始条件（取消注释使用预设）
# use_ic_preset('two_stream_strong')   # 强双流不稳定性
# use_ic_preset('landau_damping')      # Landau 阻尼

# 训练参数
TRAINING = {
    'epochs': 20000,
    'learning_rate': 1e-4,
    'n_pde': 16000,
    'n_ic': 1000,
    'n_bc': 1000,
}

# 输出目录
LOGGING = {
    'plot_dir': '2025/11/02/1'  # 结果保存路径
}
```

### 3. 验证配置

```bash
python config.py  # 测试配置是否正确
```

输出示例：
```
======================================================================
CONFIGURATION TEST
======================================================================

✓ Configuration is valid!

Model Type: hybrid_transformer
Initial Condition: two_stream
Domain: t∈[0,62.5], x∈[0,10.0], v∈[±5.0]
Training: 20000 epochs, LR=0.0001
Sampling: PDE=16000, IC=1000, BC=1000
```

### 4. 运行训练

```bash
python main.py
```

**就这么简单！** 🎉 所有配置都在 `config.py` 中，无需修改主代码。

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

## � 初始条件配置详解

### 可用的初始条件类型

在 `config.py` 中的 `INITIAL_CONDITION` 字典配置初始条件：

#### 1. Two-Stream Instability (双流不稳定性) - 默认

```python
INITIAL_CONDITION = {
    'type': 'two_stream',
    'beam_v': 1.0,          # 束流速度 (±v_b)
    'thermal_v': 0.5,       # 热速度展宽
    'perturb_amp': 0.1,     # 扰动幅度 (0-1)
    'perturb_mode': 1,      # 波数模式 k = 2π*mode/L_x
}
```

**物理意义**: 两束反向传播的电子束相互作用产生不稳定性
**数学形式**: $f(0,x,v) = \frac{1}{2}[M(v-v_b) + M(v+v_b)] \times [1 + A\cos(kx)]$

#### 2. Landau Damping (Landau阻尼)

```python
INITIAL_CONDITION = {
    'type': 'landau',
    'landau_v_thermal': 1.0,     # 热速度
    'landau_perturb_amp': 0.01,  # 小扰动幅度
    'landau_mode': 1,            # 波模数
}
```

**物理意义**: 等离子体波的动理学阻尼
**数学形式**: $f(0,x,v) = M(v) \times [1 + A\cos(kx)]$

#### 3. Single Beam (单束流)

```python
INITIAL_CONDITION = {
    'type': 'single_beam',
    'single_v_center': 0.0,      # 束流中心速度
    'single_v_thermal': 0.5,     # 热展宽
    'single_perturb_amp': 0.05,  # 扰动幅度
    'single_mode': 1,            # 波模数
}
```

**物理意义**: 单个电子束传播
**数学形式**: $f(0,x,v) = M(v-v_c) \times [1 + A\cos(kx)]$

#### 4. Custom (自定义)

```python
def CUSTOM_IC(x, v, config):
    import torch
    # 你的自定义公式
    v_th = 0.5
    norm = 1.0 / (v_th * torch.sqrt(torch.tensor(2 * torch.pi)))
    return norm * torch.exp(-v**2 / (2 * v_th**2))

INITIAL_CONDITION = {
    'type': 'custom',
}

CUSTOM_IC = CUSTOM_IC  # 设置自定义函数
```

### 使用预设配置

在 `config.py` 中取消注释以下任一行：

```python
use_ic_preset('two_stream_strong')   # 强双流不稳定性 (快速增长)
use_ic_preset('two_stream_weak')     # 弱双流不稳定性 (慢增长)
use_ic_preset('landau_damping')      # 标准 Landau 阻尼
use_ic_preset('single_beam')         # 单束流
```

### 可视化初始条件

```bash
python test_initial_conditions.py
```

生成的图像：
- `test_ic_two_stream_strong.png`: 强双流初始条件
- `test_ic_landau_damping.png`: Landau阻尼初始条件
- `compare_initial_conditions.png`: 所有初始条件对比

## 📖 完整配置示例

### 示例 1: 研究 Two-Stream 不稳定性

```python
# config.py
MODEL_TYPE = 'hybrid_transformer'

use_ic_preset('two_stream_strong')  # 使用预设

TRAINING = {
    'epochs': 20000,
    'learning_rate': 1e-4,
}

LOGGING = {
    'plot_dir': 'experiments/two_stream'
}
```

### 示例 2: 验证 Landau 阻尼

```python
# config.py
MODEL_TYPE = 'mlp'

INITIAL_CONDITION = {
    'type': 'landau',
    'landau_v_thermal': 1.0,
    'landau_perturb_amp': 0.01,
    'landau_mode': 1,
}

TRAINING = {
    'epochs': 15000,
}

LOGGING = {
    'plot_dir': 'experiments/landau_damping'
}
```

### 示例 3: 快速测试新想法

```python
# config.py
MODEL_TYPE = 'mlp'

MLP_CONFIG = {
    'nn_layers': 6,
    'nn_neurons': 64,
}

TRAINING = {
    'epochs': 500,
    'n_pde': 8000,
}

LOGGING = {
    'plot_dir': 'quick_test',
    'log_frequency': 50,
}
```

---

## 📁 文件结构

```
vp_system/
├── config.py                        # 配置文件 ⭐ (所有参数在这里设置)
├── main.py                          # 主训练脚本
├── vp_pinn.py                       # PINN 求解器核心
├── mlp.py                           # MLP 模型定义
├── transformer.py                   # Transformer 模型定义
├── visualization.py                 # 可视化函数
├── test_initial_conditions.py       # 初始条件测试和可视化
├── compare_models.py                # 模型对比实验脚本
├── README.md                        # 本文档
└── 2025/                            # 训练结果输出目录
    └── 11/02/1/
        ├── training_config.json     # 保存的配置 (JSON)
        ├── training_config.txt      # 保存的配置 (文本)
        ├── training_log.txt         # 训练日志
        ├── loss_history.png         # 损失曲线
        └── results_epoch_*.png      # 周期性结果图
```

---

## 🎯 使用工作流

### 1. 快速测试（~5分钟）

```python
# 编辑 config.py
MODEL_TYPE = 'mlp'

TRAINING = {
    'epochs': 500,          # 减少训练轮数
    'n_pde': 8000,          # 减少采样点
}

LOGGING = {
    'plot_dir': 'quick_test'
}
```

```bash
python config.py  # 验证配置
python main.py    # 开始训练
```

### 2. 标准训练（~30分钟）

```python
# config.py 使用默认配置
MODEL_TYPE = 'hybrid_transformer'

TRAINING = {
    'epochs': 20000,
    'learning_rate': 1e-4,
}
```

```bash
python main.py
```

### 3. 切换初始条件

```python
# 在 config.py 中取消注释
use_ic_preset('landau_damping')  # 切换到 Landau 阻尼

# 或直接修改参数
INITIAL_CONDITION = {
    'type': 'landau',
    'landau_v_thermal': 1.0,
    'landau_perturb_amp': 0.01,
    'landau_mode': 1,
}
```

```bash
python test_initial_conditions.py  # 可视化初始条件
python main.py                      # 开始训练
```

### 4. 参数扫描实验

```python
# 创建实验脚本 run_experiments.py
from config import INITIAL_CONDITION, LOGGING, get_configuration
import subprocess

for amp in [0.05, 0.10, 0.15, 0.20]:
    INITIAL_CONDITION['perturb_amp'] = amp
    LOGGING['plot_dir'] = f'experiments/amp_{amp}'
    
    # 保存配置并运行
    subprocess.run(['python', 'main.py'])
```

### 5. 查看和分析结果

```bash
# 查看训练日志
cat 2025/11/02/1/training_log.txt

# 查看配置
cat 2025/11/02/1/training_config.txt

# 可视化结果
open 2025/11/02/1/*.png  # macOS
# 或使用任何图片查看器
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

## 🔍 config.py 参数说明

所有参数都在 `config.py` 中配置，分为以下几个部分：

### 1. 域参数 (DOMAIN)

```python
DOMAIN = {
    't_max': 62.5,      # 最大时间 (单位: ω_p^-1)
    'x_max': 10.0,      # 空间域长度
    'v_max': 5.0,       # 最大速度
}
```

### 2. 物理参数 (PHYSICS) - 已废弃

原有的 `PHYSICS` 参数现在整合到 `INITIAL_CONDITION` 中。

### 3. 初始条件 (INITIAL_CONDITION) ⭐ 新增

```python
INITIAL_CONDITION = {
    'type': 'two_stream',   # 初始条件类型
    
    # Two-stream 参数
    'beam_v': 1.0,          # 束流速度
    'thermal_v': 0.5,       # 热速度
    'perturb_amp': 0.1,     # 扰动幅度
    'perturb_mode': 1,      # 波数模式
    
    # Landau 参数
    'landau_v_thermal': 1.0,
    'landau_perturb_amp': 0.01,
    'landau_mode': 1,
    
    # Single beam 参数
    'single_v_center': 0.0,
    'single_v_thermal': 0.5,
    'single_perturb_amp': 0.05,
    'single_mode': 1,
}
```

### 4. 模型架构 (MODEL_TYPE, MLP_CONFIG, TRANSFORMER_CONFIG)

```python
MODEL_TYPE = 'hybrid_transformer'  # 'mlp', 'transformer', 'hybrid_transformer', 'lightweight_transformer'

# MLP 配置
MLP_CONFIG = {
    'nn_layers': 8,         # 隐藏层数
    'nn_neurons': 128,      # 每层神经元数
}

# Transformer 配置
TRANSFORMER_CONFIG = {
    'd_model': 256,                 # 嵌入维度
    'nhead': 8,                     # 注意力头数
    'num_transformer_layers': 4,    # Transformer 层数
    'dim_feedforward': 512,         # 前馈网络维度
    'dropout': 0.1,                 # Dropout 率
}
```

### 5. 训练参数 (TRAINING)

```python
TRAINING = {
    'epochs': 20000,         # 训练轮数
    'learning_rate': 1e-4,   # 学习率
    'n_pde': 16000,          # PDE 采样点数
    'n_ic': 1000,            # 初始条件点数
    'n_bc': 1000,            # 边界条件点数
}
```

### 6. 损失权重 (LOSS_WEIGHTS)

```python
LOSS_WEIGHTS = {
    'weight_pde': 7.0,      # PDE 损失权重
    'weight_ic': 3.0,       # 初始条件权重
    'weight_bc': 2.0,       # 边界条件权重
}
```

### 7. 数值和日志参数 (NUMERICAL, LOGGING)

```python
NUMERICAL = {
    'v_quad_points': 128,   # 速度积分的求积点数
}

LOGGING = {
    'log_frequency': 200,    # 每 N 轮记录一次
    'plot_frequency': 2000,  # 每 N 轮绘图一次
    'plot_dir': '2025/11/02/1'  # 输出目录
}
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
| 验证配置 | `python config.py` |
| 可视化初始条件 | `python test_initial_conditions.py` |
| 运行训练 | `python main.py` |
| 切换模型 | 在 `config.py` 中修改 `MODEL_TYPE` |
| 切换初始条件 | 在 `config.py` 中调用 `use_ic_preset()` |
| 快速测试 | 在 `config.py` 中设置 `'epochs': 500` |
| 查看结果 | 打开 `LOGGING['plot_dir']` 中的图像 |

---

## 💡 最佳实践

### 实验工作流程建议

1. **首次使用**: 
   ```bash
   python config.py                    # 验证配置
   python test_initial_conditions.py   # 可视化初始条件
   python main.py                      # 开始训练（先用少量epochs测试）
   ```

2. **配置与代码分离**: 
   - ✅ 所有参数在 `config.py` 中修改
   - ✅ 不要修改 `main.py` 或 `vp_pinn.py`
   - ✅ 这样可以轻松追踪和复现实验

3. **实验组织**:
   ```python
   # 使用有意义的输出目录名
   LOGGING['plot_dir'] = 'experiments/landau_vth_1.0'
   LOGGING['plot_dir'] = 'experiments/two_stream_strong'
   LOGGING['plot_dir'] = 'scan/perturb_amp_0.15'
   ```

4. **参数调整顺序**:
   - 先选择初始条件类型
   - 再选择模型架构  
   - 最后调整训练参数
   - 使用 `test_initial_conditions.py` 验证初始条件

5. **调试技巧**:
   ```python
   # 快速调试：减少训练量
   TRAINING['epochs'] = 500
   TRAINING['n_pde'] = 8000
   LOGGING['plot_frequency'] = 100
   ```

6. **记录实验**:
   - 配置自动保存为 JSON 和 TXT
   - 在输出目录添加 `notes.txt` 记录实验目的
   - 使用版本控制（git）管理 `config.py`

### 性能优化建议

- **训练慢**: 使用 MLP 或减少采样点
- **精度低**: 增加 epochs、使用 Transformer、增加采样点
- **不稳定**: 降低学习率、增加 dropout
- **过拟合**: 减少模型复杂度、增加正则化权重

---

## 📧 故障排查

### 问题：配置验证失败

```bash
python config.py  # 查看具体错误信息
```

常见错误：
- 初始条件类型拼写错误
- 参数值不合理（如负数、零）
- 自定义初始条件函数未定义

### 问题：训练不收敛

检查：
1. 初始条件是否合理（运行 `test_initial_conditions.py`）
2. 学习率是否过大（尝试 `1e-5`）
3. 损失权重是否平衡
4. 采样点是否足够

### 问题：找不到输出文件

检查：
```python
# config.py 中的输出目录
print(LOGGING['plot_dir'])  # 确认路径
```

输出在当前目录的相对路径下，例如：`./2025/11/02/1/`

---

## 🎓 学习路径

1. **第一周**: 熟悉基本操作
   - 运行默认配置
   - 尝试不同初始条件
   - 理解输出结果

2. **第二周**: 探索模型架构
   - 对比 MLP vs Transformer
   - 调整网络规模
   - 观察训练时间和精度

3. **第三周**: 深入物理场景
   - 研究 Two-Stream 不稳定性增长率
   - 验证 Landau 阻尼理论
   - 探索参数空间

4. **第四周**: 高级应用
   - 参数扫描实验
   - 自定义初始条件
   - 发表级别的结果

---

## 🚀 开始你的研究

### 完整工作流程

```bash
# 1. 配置参数
vim config.py  # 或使用任何编辑器

# 2. 验证配置
python config.py

# 3. 可视化初始条件
python test_initial_conditions.py

# 4. 开始训练
python main.py

# 5. 查看结果
open 2025/11/02/1/*.png  # macOS
```

### 三个命令快速开始

```bash
python config.py                    # 验证配置 ✓
python test_initial_conditions.py   # 可视化初始条件 ✓
python main.py                      # 开始训练 🚀
```

**祝研究顺利！** 🎓✨

---

## 📝 更新日志

### 2025-11-02
- ✨ 初始条件完全配置化
- ✨ 支持 4 种物理场景（Two-Stream, Landau, Single Beam, Custom）
- ✨ 新增初始条件预设系统
- ✨ 新增 `test_initial_conditions.py` 可视化工具
- 📚 文档整合和简化

### 2024
- ✨ 支持多种神经网络架构（MLP, Transformer, Hybrid）
- ✨ 配置自动保存和追踪
- ✨ 归一化输入改进训练稳定性
- 📊 完整的可视化系统
