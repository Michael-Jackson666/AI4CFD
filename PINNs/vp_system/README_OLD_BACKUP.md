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

## 📚 方程组介绍

Vlasov-Poisson系统是等离子体物理中的基本方程组，描述了无碰撞等离子体中粒子分布函数的演化：

### Vlasov方程 (6D相空间)
$$\frac{\partial f}{\partial t} + v \cdot \nabla_x f + \frac{q}{m} E \cdot \nabla_v f = 0$$

### Poisson方程
$$\nabla^2 \phi = -\frac{\rho}{\varepsilon_0} = -\frac{q}{\varepsilon_0} \int f \, dv$$

### 电场
$$E = -\nabla \phi$$

其中：
- $f(x,v,t)$: 粒子分布函数 (6D相空间密度)
- $\phi(x,t)$: 电势
- $E(x,t)$: 电场
- $\rho(x,t)$: 电荷密度
- $q/m$: 电荷质量比
- $\varepsilon_0$: 真空介电常数

## 📁 文件结构

```
vlasov_poisson_system/
├── vlasov_poisson_solver.py    # 主求解器类
├── run_vlasov_examples.py      # 使用示例和演示
└── README.md                   # 说明文档
```

## 🚀 快速开始

### 1. 环境要求

确保已安装以下Python包：

```bash
pip install deepxde numpy matplotlib scipy
```

**注意**: Vlasov-Poisson系统是6维问题，对计算资源要求极高，建议使用GPU加速。

### 2. 基础使用

```python
from vlasov_poisson_solver import VlasovPoissonSolver

# 创建Landau阻尼求解器
solver = VlasovPoissonSolver(
    x_domain=(-1.0, 1.0),      # 空间域
    v_domain=(-3.0, 3.0),      # 速度域
    time_domain=(0.0, 1.0),    # 时间域
    case="landau_damping"      # 预设案例
)

# 设置问题
solver.setup_geometry_and_conditions()

# 创建神经网络模型
solver.create_model()

# 训练模型
solver.train()

# 可视化结果
solver.visualize_phase_space_evolution()
solver.analyze_plasma_dynamics()
```

### 3. 运行示例

```bash
python run_vlasov_examples.py
```

程序提供5个经典等离子体物理示例：
1. **Landau阻尼** - 等离子体波的无碰撞阻尼
2. **双流不稳定性** - 两束反向电子束的不稳定性
3. **Bump-on-tail不稳定性** - 高能电子尾部驱动的不稳定性
4. **参数研究** - 不同波数对阻尼率的影响
5. **高级分析** - 深入的相空间结构分析

## 🔧 主要功能

### VlasovPoissonSolver类

#### 核心方法
- **初始化**: `__init__(x_domain, v_domain, time_domain, case)`
- **设置问题**: `setup_geometry_and_conditions()`
- **创建模型**: `create_model(num_domain, num_boundary, num_initial, layer_sizes)`
- **模型训练**: `train(adam_iterations, adam_lr, weights_pde)`
- **预测**: `predict(inputs)` - 返回$[f, \phi]$

#### 分析工具
- **宏观量计算**: `compute_macroscopic_quantities()` - 密度、平均速度、温度
- **相空间可视化**: `visualize_phase_space_evolution()`
- **等离子体动力学分析**: `analyze_plasma_dynamics()`
- **初始条件可视化**: `visualize_initial_conditions()`

#### 模型管理
- **保存模型**: `save_model(filename)`
- **加载模型**: `load_model(filename)`

### 预设案例

| 案例 | 物理现象 | 特点 |
|------|----------|------|
| `landau_damping` | Landau阻尼 | 等离子体波的无碰撞阻尼 |
| `two_stream` | 双流不稳定性 | 两束反向流的相互作用 |
| `bump_on_tail` | 尾部不稳定性 | 高能粒子驱动的不稳定性 |

## 📊 使用技巧

### 1. 网络设计要点
- **6D复杂性**: 比低维PDE复杂数千倍，需要大型网络
- **推荐结构**: `[3, 128, 128, 128, 128, 2]` 或更大
- **激活函数**: `tanh`对等离子体系统效果最好
- **输出**: 2维 (分布函数$f$, 电势$\phi$)

### 2. 训练策略
- **大量采样点**: 6D系统需要8000+域内采样点
- **小学习率**: 建议0.0005-0.001
- **长时间训练**: 15000+ Adam迭代 + L-BFGS
- **方程权重**: Vlasov权重 > Poisson权重 (如$[1.0, 0.1]$)

### 3. 计算资源优化
- **GPU加速**: 强烈推荐使用CUDA
- **内存管理**: 6D系统内存需求巨大
- **批处理**: 合理设置batch size避免内存溢出

### 4. 物理参数调优
- **时间域**: 不要设置太长，避免数值不稳定
- **速度域**: 必须包含主要粒子速度分布
- **空间域**: 通常使用周期性边界条件

## 🎯 经典应用案例

### 1. Landau阻尼研究
```python
solver = VlasovPoissonSolver(
    x_domain=(-np.pi, np.pi),
    v_domain=(-4.0, 4.0),
    time_domain=(0.0, 10.0),
    case="landau_damping"
)
# 研究不同波数$k$的阻尼率$\gamma$
```

### 2. 双流不稳定性
```python
solver = VlasovPoissonSolver(
    v_domain=(-3.0, 3.0),  # 覆盖两束流速度
    case="two_stream"
)
# 观察束流间的能量交换和相空间混合
```

### 3. 等离子体加热
```python
solver = VlasovPoissonSolver(
    v_domain=(-2.0, 5.0),  # 扩展到高能区域
    case="bump_on_tail"
)
# 研究高能粒子对等离子体加热的影响
```

## 🔬 高级功能

### 自定义初始分布
```python
class CustomVlasovSolver(VlasovPoissonSolver):
    def initial_condition_f(self, inputs):
        x, v, t = inputs[:, 0:1], inputs[:, 1:2], inputs[:, 2:3]
        # 自定义分布函数
        return custom_distribution(x, v)
    
    def initial_condition_phi(self, inputs):
        # 自定义初始电势
        return custom_potential(inputs)
```

### 宏观量时间演化
宏观量的计算公式：

- **粒子密度**: $n(x,t) = \int f(x,v,t) \, dv$
- **平均速度**: $\langle v \rangle(x,t) = \frac{1}{n(x,t)} \int v \cdot f(x,v,t) \, dv$  
- **温度**: $T(x,t) \propto \frac{1}{n(x,t)} \int (v - \langle v \rangle)^2 f(x,v,t) \, dv$

```python
# 计算密度、流速、温度的时空演化
times = np.linspace(0, t_max, 50)
x_points = np.linspace(x_min, x_max, 30)

for t in times:
    macro_quantities = solver.compute_macroscopic_quantities(x_points, t)
    density = macro_quantities['density']
    mean_velocity = macro_quantities['mean_velocity']
    temperature = macro_quantities['temperature']
```

### 能量守恒检验
```python
# 检验总能量守恒
kinetic_energy = ∫∫ (v²/2) f(x,v,t) dx dv
potential_energy = (ε₀/2) ∫ E²(x,t) dx
total_energy = kinetic_energy + potential_energy
```

更精确的数学表达式：

$$\text{动能} = \iint \frac{v^2}{2} f(x,v,t) \, dx \, dv$$

$$\text{势能} = \frac{\varepsilon_0}{2} \int E^2(x,t) \, dx$$

$$\text{总能量} = \text{动能} + \text{势能}$$

## ⚠️ 重要注意事项

### 1. 计算复杂性
- **维度诅咒**: 6D系统比2D复杂10³-10⁶倍
- **训练时间**: 可能需要数小时到数天
- **内存需求**: 通常需要16GB+内存
- **GPU必需**: CPU训练几乎不可行

### 2. 数值稳定性
- **时间步长**: Vlasov方程对时间步长敏感
- **速度边界**: 必须包含所有重要的速度成分
- **周期性**: 空间边界条件影响结果准确性

### 3. 物理有效性
- **质量守恒**: $\int f \, dv$ 应该守恒
- **能量守恒**: 总能量应该守恒(或按已知规律变化)
- **熵增**: H定理要求熵单调增加

### 4. 收敛性检验
- **网格无关性**: 增加采样点数验证结果
- **时间收敛**: 延长时间域检查长期行为
- **参数敏感性**: 测试不同初始条件的鲁棒性

## 📈 性能基准

| 案例 | 域内点数 | 训练时间 | GPU内存 | 推荐配置 |
|------|----------|----------|---------|----------|
| Landau阻尼 | 8000 | ~2小时 | 8GB | RTX 3080+ |
| 双流不稳定 | 6000 | ~1.5小时 | 6GB | RTX 3070+ |
| Bump-on-tail | 7000 | ~2.5小时 | 10GB | RTX 3090+ |

## 🔍 调试指南

### 常见问题
1. **训练不收敛**
   - 减小学习率
   - 增加网络深度
   - 调整PDE权重比例
   - 检查初始条件合理性

2. **内存不足**
   - 减少采样点数
   - 降低网络复杂度
   - 使用梯度累积

3. **结果不物理**
   - 检查边界条件设置
   - 验证初始条件
   - 增加训练时间
   - 调整时间域范围

## 📖 理论背景

### Landau阻尼
Landau阻尼是等离子体物理中的重要现象，由于粒子与波的共振相互作用导致波的无碰撞阻尼。阻尼率由Landau积分给出：

$$\gamma = -\pi \frac{\omega_p^2}{k^2} \left.\frac{\partial f_0}{\partial v}\right|_{v=\omega/k}$$

### 双流不稳定性
当两束粒子流相对运动时，会发生静电不稳定性，增长率由：

$$\gamma = \frac{\omega_p}{2}\left(\frac{\omega}{kv_0}\right)^{1/3}$$

其中$v_0$是束流相对速度。

## 📚 参考资料

### 经典教材
- Birdsall & Langdon: "Plasma Physics via Computer Simulation"
- Krall & Trivelpiece: "Principles of Plasma Physics"
- Boyd & Sanderson: "The Physics of Plasmas"

### 重要论文
- Landau (1946): "On the vibrations of the electronic plasma"
- Vlasov (1938): "On vibration properties of electron gas"
- Dawson (1983): "Particle simulation of plasmas"

### 数值方法
- Cheng & Knorr (1976): "The integration of the Vlasov equation"
- Filbet & Sonnendrücker (2003): "Comparison of Eulerian Vlasov solvers"

## 🤝 贡献

欢迎贡献代码和改进！可以考虑的扩展方向：
- 多物种等离子体 (电子+离子)
- 磁化等离子体 (添加磁场)
- 相对论性效应
- 碰撞项 (Fokker-Planck)
- 2D/3D空间扩展

## 📄 许可证

MIT License

---

**⚠️ 免责声明**: Vlasov-Poisson系统是极其复杂的6维问题，本求解器主要用于研究和教学目的。对于实际等离子体物理应用，请谨慎验证结果的物理合理性。