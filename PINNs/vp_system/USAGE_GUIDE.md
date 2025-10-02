# Vlasov-Poisson求解器使用指南

## 🎯 使用前准备

### 检查系统配置
在开始使用Vlasov-Poisson求解器之前，请确认您的系统满足以下要求：

#### 最低配置
- **CPU**: 多核处理器 (推荐8核以上)
- **内存**: 16GB RAM (推荐32GB+)
- **GPU**: 可选但强烈推荐 (RTX 3070或更好)
- **存储**: 5GB可用空间

#### 软件环境
```bash
python >= 3.8
deepxde >= 1.0
numpy >= 1.20
matplotlib >= 3.3
scipy >= 1.7
```

### 安装步骤
```bash
# 1. 安装DeepXDE (选择一个后端)
pip install deepxde[tensorflow]  # TensorFlow后端
# 或
pip install deepxde[pytorch]     # PyTorch后端
# 或  
pip install deepxde[jax]         # JAX后端

# 2. 安装其他依赖
pip install numpy matplotlib scipy

# 3. (推荐) 安装GPU支持
pip install tensorflow-gpu  # TensorFlow GPU
# 或
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118  # PyTorch GPU
```

## 🚀 快速入门

### 步骤1: 运行测试
首次使用前，强烈建议运行测试脚本验证安装：

```bash
cd vlasov_poisson_system
python test_vlasov_solver.py
```

选择"1. 快速功能测试"，如果看到所有"✅"，说明安装成功。

### 步骤2: 选择合适的案例
根据您的研究目标和计算资源选择案例：

| 案例 | 计算难度 | 物理现象 | 推荐用途 |
|------|----------|----------|----------|
| Landau阻尼 | ⭐⭐⭐ | 波阻尼 | 入门学习 |
| 双流不稳定 | ⭐⭐⭐⭐ | 束流不稳定 | 进阶研究 |
| Bump-on-tail | ⭐⭐⭐⭐⭐ | 高能粒子效应 | 专业研究 |

### 步骤3: 参数配置
根据计算资源调整参数：

#### 快速测试配置 (30分钟内)
```python
solver.create_model(
    num_domain=1500,      # 域内点
    num_boundary=120,     # 边界点
    num_initial=120,      # 初始点
    layer_sizes=[3, 60, 60, 60, 2]  # 网络结构
)

solver.train(
    adam_iterations=2000,  # 迭代次数
    adam_lr=0.001,        # 学习率
    use_lbfgs=False       # 跳过L-BFGS
)
```

#### 标准配置 (2-4小时)
```python
solver.create_model(
    num_domain=5000,
    num_boundary=400,
    num_initial=400,
    layer_sizes=[3, 100, 100, 100, 2]
)

solver.train(
    adam_iterations=8000,
    adam_lr=0.0008,
    use_lbfgs=True
)
```

#### 高精度配置 (6-12小时)
```python
solver.create_model(
    num_domain=10000,
    num_boundary=800,
    num_initial=800,
    layer_sizes=[3, 128, 128, 128, 128, 2]
)

solver.train(
    adam_iterations=15000,
    adam_lr=0.0005,
    use_lbfgs=True
)
```

## 📊 结果解读指南

### 1. 训练损失分析
- **目标损失**: 通常应达到1e-4或更低
- **收敛判据**: 损失曲线应单调下降并趋于平稳
- **异常情况**: 如果损失震荡或发散，需要调整学习率

### 2. 物理量检验
验证结果的物理合理性：

#### 质量守恒
```python
# 计算总粒子数
total_particles = ∫∫ f(x,v,t) dx dv
# 应该在时间演化中保持常数
```

#### 能量演化
```python
# 动能 + 势能应该守恒(或按已知规律变化)
kinetic_energy = ∫∫ (v²/2) f(x,v,t) dx dv
potential_energy = (ε₀/2) ∫ E²(x,t) dx
```

#### Landau阻尼率
对于Landau阻尼案例，电场应按`E(t) ∝ exp(-γt)`衰减：
```python
# 拟合电场衰减率
import numpy as np
times = np.linspace(0, t_max, 50)
E_fields = [compute_electric_field(t) for t in times]
gamma = -np.polyfit(times[10:40], np.log(np.abs(E_fields[10:40])), 1)[0]
print(f"阻尼率: γ = {gamma:.4f}")
```

### 3. 相空间结构
检查相空间演化的合理性：
- **初期**: 应接近设定的初始分布
- **演化**: 应显示预期的物理现象(如束流混合、相空间扭曲)
- **长期**: 应趋向某种平衡态或周期性行为

## 🔧 故障排除

### 常见问题及解决方案

#### 1. 内存不足错误
```
OutOfMemoryError: GPU/CPU memory exceeded
```

**解决方案**:
- 减少采样点数 (`num_domain`, `num_boundary`, `num_initial`)
- 缩小网络规模 (`layer_sizes`)
- 使用更小的批处理大小
- 关闭不必要的程序释放内存

#### 2. 训练不收敛
```
Loss不下降或发散
```

**解决方案**:
- 降低学习率 (从0.001降到0.0005或0.0001)
- 增加网络深度或宽度
- 调整PDE方程权重 `weights_pde=[1.0, 0.1]`
- 检查初始条件是否合理

#### 3. 结果不物理
```
质量不守恒、能量发散等
```

**解决方案**:
- 增加训练时间
- 提高采样点密度
- 检查边界条件设置
- 缩短时间域范围
- 验证初始条件的物理合理性

#### 4. 计算太慢
```
训练时间过长
```

**解决方案**:
- 启用GPU加速
- 减少采样点数进行快速测试
- 使用更小的时间域
- 考虑使用云计算资源

### 性能优化技巧

#### GPU加速配置
```python
# TensorFlow
import tensorflow as tf
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    tf.config.experimental.set_memory_growth(gpus[0], True)

# PyTorch
import torch
if torch.cuda.is_available():
    device = torch.device('cuda')
    print(f"Using GPU: {torch.cuda.get_device_name()}")
```

#### 内存优化
```python
# 设置更小的批处理大小
import deepxde as dde
dde.config.set_default_float("float32")  # 使用单精度
```

#### 自适应训练策略
```python
# 分阶段训练
# 阶段1: 快速预训练
solver.train(adam_iterations=2000, adam_lr=0.01, use_lbfgs=False)

# 阶段2: 精细调优
solver.train(adam_iterations=5000, adam_lr=0.001, use_lbfgs=False)

# 阶段3: 最终优化
solver.train(adam_iterations=0, use_lbfgs=True)
```

## 📚 进阶使用

### 自定义物理案例
```python
class MyCustomVlasovSolver(VlasovPoissonSolver):
    def __init__(self):
        super().__init__(
            x_domain=(-2.0, 2.0),
            v_domain=(-4.0, 4.0),
            time_domain=(0.0, 5.0),
            case="custom"
        )
    
    def initial_condition_f(self, inputs):
        x, v, t = inputs[:, 0:1], inputs[:, 1:2], inputs[:, 2:3]
        # 实现您的自定义初始分布
        return your_custom_distribution(x, v)
    
    def initial_condition_phi(self, inputs):
        # 实现您的自定义初始电势
        return your_custom_potential(inputs)
```

### 多参数扫描
```python
# 参数扫描示例
parameters = {
    'amplitude': [0.01, 0.05, 0.1],
    'k_mode': [0.3, 0.5, 0.8],
    'thermal_velocity': [0.8, 1.0, 1.2]
}

results = {}
for amp in parameters['amplitude']:
    for k in parameters['k_mode']:
        for vth in parameters['thermal_velocity']:
            # 创建求解器并运行
            # 保存结果到results字典
            pass
```

### 结果后处理
```python
# 计算高阶矩
def compute_higher_moments(solver, x_points, t):
    """计算密度、流速、温度、偏度、峰度等"""
    moments = {}
    # 实现计算逻辑
    return moments

# 相空间分析
def analyze_phase_space_structure(solver, t):
    """分析相空间中的涡旋、纤维化等结构"""
    # 实现分析逻辑
    pass

# 频谱分析
def spectral_analysis(time_series):
    """对时间序列进行频谱分析"""
    from scipy.fft import fft, fftfreq
    # 实现FFT分析
    pass
```

## 🎓 学习路径建议

### 初学者 (第1-2周)
1. 运行快速测试验证安装
2. 学习Landau阻尼基础案例
3. 理解相空间可视化
4. 掌握基本参数调整

### 进阶用户 (第3-4周)
1. 尝试双流不稳定性案例
2. 学习参数研究方法
3. 掌握结果物理性检验
4. 自定义初始条件

### 专业用户 (第5-8周)
1. 研究Bump-on-tail等复杂案例
2. 开发自定义物理模型
3. 进行多参数优化研究
4. 发表研究成果

## 📞 获取帮助

### 社区资源
- **GitHub Issues**: 报告bug和功能请求
- **DeepXDE论坛**: 技术讨论和经验分享
- **等离子体物理论坛**: 物理问题讨论

### 学术支持
- **论文参考**: 查阅相关的Vlasov-Poisson数值方法论文
- **专业会议**: 参加等离子体物理和计算物理会议
- **合作研究**: 寻找相关领域的合作者

---

**🎯 记住**: Vlasov-Poisson系统是极其复杂的6维问题，需要足够的耐心和计算资源。建议从简单案例开始，逐步提高复杂度，并始终验证结果的物理合理性！