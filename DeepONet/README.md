# Deep Operator Networks (DeepONet)

Deep Operator Networks (DeepONet) are neural networks designed to learn operators that map between infinite-dimensional function spaces. Unlike traditional neural networks that map finite-dimensional vectors to vectors, DeepONet can learn mappings from functions to functions, making them particularly powerful for solving families of PDEs and operator learning problems.

DeepONet 是用于学习无穷维函数空间之间算子映射的神经网络。与传统神经网络不同，DeepONet 可以学习从函数到函数的映射，特别适用于求解 PDE 族和算子学习问题。

## 📦 安装 (Installation)

### Prerequisites
- Python 3.8+
- PyTorch 1.10+
- CUDA (optional, for GPU acceleration)

### Basic Installation
```bash
# Clone the repository
git clone https://github.com/your-repo/AI4CFD.git
cd AI4CFD/DeepONet

# Install PyTorch (CPU version)
pip install torch torchvision torchaudio

# For GPU support, visit: https://pytorch.org/get-started/locally/

# Install other dependencies
pip install numpy scipy matplotlib jupyter
```

### Verify Installation
```bash
python -c "import torch; print('PyTorch version:', torch.__version__)"
python operator_learning_pure.py  # Run simple example
```

## 🎯 Key Concepts

### What is DeepONet?
DeepONet learns operator mappings of the form:
```
G: U → V
```
where U and V are function spaces. For example:
- **PDE Solution Operator**: G maps initial/boundary conditions to PDE solutions
- **Parameter-to-Solution Map**: G maps PDE parameters to solutions
- **Time Evolution Operator**: G maps current state to future state

### Architecture
DeepONet consists of two sub-networks:
1. **Branch Network**: Encodes the input function at sensor locations
2. **Trunk Network**: Encodes the query locations where we want to evaluate the output
3. **Combination**: Outputs are combined via inner product: `G(u)(y) = Σᵢ bᵢ(u) * tᵢ(y)`

### Key Advantages
- **Universal approximation**: Can approximate any continuous operator
- **Generalization**: Trained on one set of functions, generalizes to new functions
- **Efficiency**: No need to retrain for new input functions
- **Theoretical foundation**: Based on universal approximation theorem for operators

## 📁 目录结构 (Directory Structure)

```
DeepONet/
├── README.md                          # 本文档
├── models.py                          # DeepONet 架构和变体
├── operators.py                       # 常见算子定义和示例
├── operator_learning_pure.py         # 纯 PyTorch 实现的算子学习
│
├── tutorial/                          # 📚 教程笔记本
│   └── operator_learning_torch.ipynb # PyTorch DeepONet 完整教程
│
└── vp_system/                         # 🌌 Vlasov-Poisson 算子学习
    ├── README.md                      # VP 算子学习详细文档
    ├── main.py                        # 主训练脚本
    ├── vp_operator.py                 # VP 系统算子定义
    ├── data_generate.py               # VP 训练数据生成
    ├── transformer.py                 # Transformer 架构
    └── visualization.py               # 结果可视化工具
```

## 📝 核心文件说明 (Core Files Description)

### `models.py`
DeepONet 架构和变体实现。

**包含的模型**:
- **StandardDeepONet**: 标准 DeepONet（Branch + Trunk 网络）
- **PODDeepONet**: 基于 POD（本征正交分解）的 DeepONet
- **DeepONet with Attention**: 注意力机制增强版本
- **Residual DeepONet**: 带残差连接的 DeepONet

### `operators.py`
常见算子定义和数据生成函数。

**支持的算子**:
- **反导数算子**: $G: f(x) \rightarrow \int f(x)dx$
- **热方程算子**: $G: u_0(x) \rightarrow u(x,t)$
- **Darcy 流算子**: $G: \kappa(x,y) \rightarrow p(x,y)$
- **Burgers 方程算子**: 非线性对流扩散算子
- **泊松方程算子**: $G: f(x) \rightarrow u(x)$ 其中 $\nabla^2 u = f$

### `operator_learning_pure.py`
使用纯 PyTorch 实现的算子学习完整示例。

**功能**:
- 从零开始构建 DeepONet
- 数据生成和预处理
- 训练循环和验证
- 结果可视化和误差分析
- 适合学习 DeepONet 的内部机制

## � 教程笔记本 (Tutorial Notebook)

### `tutorial/operator_learning_torch.ipynb`
**完整的 PyTorch DeepONet 教程**

**内容涵盖**:
- 📖 **DeepONet 理论基础**: 算子学习的数学原理和通用逼近定理
- 🏗️ **架构设计**: Branch 网络和 Trunk 网络的详细解析
- 💻 **代码实现**: 从零开始用 PyTorch 构建 DeepONet
- 📊 **实际案例**: 
  - 反导数算子学习
  - 热方程解算子
  - Darcy 流算子
- 🎯 **训练技巧**: 数据生成策略、损失函数设计、超参数调优
- 📈 **结果分析**: 误差评估、泛化能力测试、可视化技术

**适合人群**: 
- 希望深入理解 DeepONet 原理的研究者
- 需要自定义算子学习模型的开发者
- 学习科学机器学习（SciML）的学生

## �🚀 快速开始 (Quick Start)

### Running the Tutorial
```bash
# PyTorch DeepONet 完整教程
jupyter notebook tutorial/operator_learning_torch.ipynb
```

### Running Pure PyTorch Implementation
```bash
# 运行纯 PyTorch 实现示例
python operator_learning_pure.py
```

### 🌌 Vlasov-Poisson Operator Learning (vp_system/)

For Vlasov-Poisson operator learning applications (学习 VP 系统的解算子):

```bash
cd vp_system/

# 1. 生成训练数据
python data_generate.py

# 2. 训练 DeepONet 模型
python main.py

# 3. 可视化结果
python visualization.py
```

**功能特点**:
- **算子学习**: 学习从初始分布 $f_0(x,v)$ 到演化后分布 $f(x,v,t)$ 的映射
- **数据生成**: 自动生成多组不同初始条件的 VP 系统求解数据
- **Transformer 架构**: 支持基于注意力机制的 DeepONet 变体
- **可视化工具**: 相空间、电场、分布函数的综合绘图

**Key Files**:
- `vp_operator.py`: VP 系统算子定义和 DeepONet 实现
- `data_generate.py`: 生成训练数据（多组初始条件的 VP 演化）
- `main.py`: 完整的训练和测试流程
- `transformer.py`: Transformer-based DeepONet 架构
- `visualization.py`: 结果可视化（相空间动画、误差分析等）

详见 `vp_system/README.md` 了解更多细节。

**技术亮点**:
- ⚡ **高效数据生成**: 自动化生成多种初始条件的 VP 演化轨迹
- 🧠 **先进架构**: 支持传统 MLP 和 Transformer 两种 DeepONet 实现
- 🎨 **丰富可视化**: 相空间密度、电场分布、时间演化动画
- 📊 **定量评估**: L2 误差、相对误差、能量守恒检验
- 🔄 **端到端流程**: 从数据生成到模型训练到结果分析的完整管道

**与传统求解器对比**:
| 特性 | 传统数值求解器 | DeepONet 算子学习 |
|------|---------------|------------------|
| 单次求解时间 | 分钟-小时 | 毫秒级 |
| 参数扫描 | 每组参数重新计算 | 一次前向传播 |
| 内存占用 | 高（网格数据） | 低（网络参数） |
| 适用场景 | 高精度单次计算 | 快速多次预测 |

## 📊 示例应用 (Example Applications)

### 1. 热方程算子 (Heat Equation Operator)
学习从初始温度分布到任意时刻温度分布的算子映射：
```
G: u₀(x) → u(x,t)
```
**应用**: 瞬态热传导、温度场预测

### 2. Darcy 流算子 (Darcy Flow Operator)
学习从渗透率场到压力/速度场的映射：
```
G: κ(x,y) → p(x,y)
```
**应用**: 多孔介质流动、地下水模拟、油藏工程

### 3. 反导数算子 (Antiderivative Operator)
学习从函数到其不定积分的映射：
```
G: f(x) → ∫f(x)dx
```
**应用**: 数值积分、符号计算的神经网络替代

### 4. Burgers 方程算子 (Burgers' Equation Operator)
学习非线性对流扩散方程的解算子：
```
G: u₀(x) → u(x,t)  其中  ∂u/∂t + u∂u/∂x = ν∂²u/∂x²
```
**应用**: 激波传播、交通流模拟、湍流建模

### 5. **Vlasov-Poisson 算子** 🌌 (vp_system/)
学习等离子体动理学方程的演化算子：
```
G: f₀(x,v) → f(x,v,t)
```
**应用**: 
- 双流不稳定性（Two-stream instability）模拟
- 等离子体动力学预测
- 相空间演化学习
- 电场自洽计算

这是本项目的**高级应用示例**，展示了 DeepONet 在复杂物理系统中的强大能力。

## 🔧 Implementation Details

### Training Data Generation
DeepONet requires:
1. **Input functions**: Sampled at sensor locations
2. **Output functions**: Evaluated at query locations
3. **Sensor locations**: Fixed points where input functions are observed
4. **Query locations**: Points where output is evaluated

### Network Architecture
```python
# Branch network: processes input function values
branch_net = MLP(input_dim=sensor_size, output_dim=p)

# Trunk network: processes query coordinates  
trunk_net = MLP(input_dim=coord_dim, output_dim=p)

# Output: inner product
output = torch.sum(branch_output * trunk_output, dim=-1)
```

### Loss Function
```python
loss = MSE(predicted_output, true_output) + regularization_terms
```

## 📚 Mathematical Background

For a continuous operator G: U → V between Banach spaces, DeepONet approximates:

$$G(u)(y) = \sum_{i=1}^p b_i(u) \cdot t_i(y) + b_0$$

where:
- $b_i(u)$ are the branch network outputs (depend on input function u)
- $t_i(y)$ are the trunk network outputs (depend on query location y)
- $p$ is the dimension of the latent space

### Universal Approximation
**Theorem**: If the branch network can approximate any continuous functional and the trunk network can approximate any continuous function, then DeepONet can approximate any continuous operator.

## 🎯 Training Strategies

### 1. Standard Training
- Generate diverse input functions
- Sample query points uniformly
- Use standard MSE loss

### 2. Physics-Informed Training
- Add PDE residual loss
- Enforce boundary conditions
- Include conservation laws

### 3. Multi-Fidelity Training
- Use data from multiple resolutions
- Transfer learning between fidelities
- Adaptive sampling strategies

### 4. Residual Learning
- Learn residuals from simple operators
- Improve accuracy on complex problems
- Reduce training time

## 🔗 在 CFD 中的应用 (Applications in CFD)

### 流体流动算子 (Fluid Flow Operators)
- **Navier-Stokes 算子**: 边界条件 → 速度/压力场
  - 应用: 快速流场预测、形状优化、实时模拟
- **湍流建模**: 学习亚网格尺度模型算子
  - 应用: LES/RANS 模型的数据驱动替代
- **形状优化**: 几何参数 → 流动特性
  - 应用: 翼型设计、管道优化

### 传热算子 (Heat Transfer)
- **热传导算子**: 热物性参数 → 温度场
  - 应用: 材料设计、热管理系统
- **对流传热**: 学习换热系数算子
  - 应用: 冷却系统设计、热交换器优化
- **辐射传热**: 复杂辐射传递算子建模
  - 应用: 高温系统、燃烧模拟

### 多相流算子 (Multiphase Flows)
- **界面追踪**: 学习界面演化算子
  - 应用: 自由表面流、气液两相流
- **相变过程**: 熔化/凝固算子建模
  - 应用: 铸造工艺、增材制造
- **多孔介质**: 学习有效物性算子
  - 应用: 油藏模拟、地下水流动

### 等离子体物理 (Plasma Physics) 🌌
- **Vlasov-Poisson 系统**: 初始分布 → 相空间演化（见 `vp_system/`）
  - 应用: 双流不稳定性、等离子体振荡、束流传输
- **电磁场演化**: 学习自洽场算子
  - 应用: 核聚变、等离子体加速器

### 相比传统 CFD 的优势
✅ **快速预测**: 训练后毫秒级求解（传统 CFD: 小时/天）  
✅ **参数扫描**: 无需重新计算，适合优化和不确定性量化  
✅ **泛化能力**: 一次训练，处理相似问题族  
✅ **可微分**: 支持梯度优化和逆问题求解  

## 📖 参考文献 (References)

1. Lu, L., Jin, P., Pang, G., Zhang, Z., & Karniadakis, G. E. (2021). Learning nonlinear operators via DeepONet based on the universal approximation theorem of operators. Nature Machine Intelligence, 3(3), 218-229.

2. Wang, S., Wang, H., & Perdikaris, P. (2021). Learning the solution operator of parametric partial differential equations with physics-informed DeepONets. Science Advances, 7(40), eabi8605.

3. Lin, C., Li, Z., Lu, L., Cai, S., Maxey, M., & Karniadakis, G. E. (2021). Operator learning for predicting multiscale bubble growth dynamics. The Journal of Chemical Physics, 154(10), 104118.