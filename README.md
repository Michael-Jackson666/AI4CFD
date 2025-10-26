# AI4CFD: AI Methods for Computational Fluid Dynamics

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.10+-ee4c2c.svg)](https://pytorch.org/)

A comprehensive collection of state-of-the-art deep learning methods for solving Partial Differential Equations (PDEs) in computational fluid dynamics and plasma physics. This repository provides complete implementations with tutorials, training code, and real-world applications.

## 🎯 Overview

This repository bridges the gap between AI research and practical PDE solving by providing:

- **✅ Complete Implementations**: Production-ready code for 4 major AI-PDE methods
- **📚 Interactive Tutorials**: Jupyter notebooks with step-by-step explanations
- **🌌 Real Applications**: Including plasma physics (Vlasov-Poisson systems)
- **🔧 Utility Tools**: Shared functions for data processing, visualization, and evaluation
- **📖 Comprehensive Documentation**: Detailed READMEs for each method

## 🧠 Methods Implemented

## 🧠 Methods Implemented

### 1. Physics-Informed Neural Networks (PINNs) 🔬
**Neural networks that encode physical laws directly into the loss function**

- ✨ Solve forward and inverse PDE problems
- ✨ Handle complex geometries and boundary conditions
- ✨ Work with limited or noisy data
- � **Location**: `PINNs/`
- 📚 **Tutorials**: 8 comprehensive Jupyter notebooks (English & Chinese)
- 🎯 **Applications**: Poisson, Heat, Burgers, Navier-Stokes, Vlasov-Poisson

**Key Features**:
- Multiple neural network architectures (MLP, Residual, Fourier, Multi-scale)
- Physics-informed loss functions with automatic differentiation
- Educational tutorials from basics to advanced topics
- Research-grade VP-PINNs implementation with Transformer architecture

### 2. Deep Operator Networks (DeepONet) 🎯
**Learn mappings between infinite-dimensional function spaces**

- ✨ Learn entire families of PDE solutions at once
- ✨ Fast inference after training (milliseconds vs hours)
- ✨ Generalize to unseen parameters and initial conditions
- 📂 **Location**: `DeepONet/`
- 📚 **Tutorial**: PyTorch implementation with detailed examples
- 🎯 **Applications**: Antiderivative, Heat equation, Darcy flow, Burgers, VP systems

**Key Features**:
- Branch-Trunk architecture for operator learning
- Pure PyTorch implementation for learning and customization
- Transformer-based DeepONet variants
- VP operator learning for plasma physics

### 3. Fourier Neural Operators (FNO) 🌊
**Neural operators in frequency domain for efficient PDE solving**

- ✨ Discretization-invariant: train once, apply to any resolution
- ✨ Excellent for periodic and quasi-periodic problems
- ✨ State-of-the-art performance on benchmark datasets
- 📂 **Location**: `FNO/`
- 🎯 **Applications**: Navier-Stokes, Darcy flow, turbulence modeling

**Key Features**:
- Fourier layer implementation with efficient FFT
- Multi-scale frequency mixing
- Resolution-independent architecture

### 4. Transformer-based Methods 🤖
**Sequence-to-sequence models adapted for PDE solving**

- ✨ Capture long-range dependencies in spatial/temporal domains
- ✨ Attention mechanisms for adaptive feature learning
- ✨ Flexible architecture for various PDE types
- 📂 **Location**: `Transformer/`
- 🎯 **Applications**: Time-series prediction, multi-physics coupling

**Key Features**:
- Self-attention for spatial features
- Temporal attention for time evolution
- Hybrid physics-attention architectures

**Key Features**:
- Self-attention for spatial features
- Temporal attention for time evolution
- Hybrid physics-attention architectures

## 📁 Repository Structure

```
AI4CFD/
├── 📂 PINNs/                          # Physics-Informed Neural Networks
│   ├── models.py                      # Network architectures
│   ├── pde_definitions.py             # PDE residual functions
│   ├── train.py                       # Training pipeline
│   ├── tutorial/                      # 8 Jupyter notebooks
│   │   ├── tutorial_chinese.ipynb     # Complete Chinese tutorial
│   │   ├── tutorial_eng.ipynb         # Complete English tutorial
│   │   ├── possion_1d.ipynb          # 1D Poisson equation
│   │   ├── heat_2d.ipynb             # 2D heat equation
│   │   ├── ns_basic.ipynb            # Navier-Stokes basics
│   │   ├── ns_advanced.ipynb         # Advanced Navier-Stokes
│   │   ├── system_pde.ipynb          # Coupled PDEs
│   │   └── vlasov_poisson.ipynb      # Plasma physics
│   ├── examples/                      # DeepXDE examples
│   ├── vp_system/                     # Research-grade VP-PINNs
│   │   ├── mlp.py, transformer.py    # Model architectures
│   │   ├── compare_models.py         # Benchmarking tools
│   │   └── visualization.py          # Plotting utilities
│   └── README.md                      # Detailed documentation
│
├── 📂 DeepONet/                       # Deep Operator Networks
│   ├── models.py                      # DeepONet architectures
│   ├── operators.py                   # Operator definitions
│   ├── operator_learning_pure.py     # Pure PyTorch implementation
│   ├── tutorial/                      
│   │   └── operator_learning_torch.ipynb  # Complete tutorial
│   ├── vp_system/                     # VP operator learning
│   │   ├── vp_operator.py            # VP-specific operators
│   │   ├── data_generate.py          # Training data generation
│   │   ├── transformer.py            # Transformer architecture
│   │   └── visualization.py          # Result visualization
│   └── README.md
│
├── 📂 FNO/                            # Fourier Neural Operators
│   ├── layers.py                      # Fourier layers
│   ├── models.py                      # FNO architectures
│   └── README.md
│
├── 📂 Transformer/                    # Transformer-based Methods
│   ├── models.py                      # Transformer architectures
│   └── README.md
│
├── 📂 VP_system/                      # Vlasov-Poisson Applications
│   └── TwoStreamInstability/          # Two-stream instability simulation
│       ├── hypar_solver/              # HyPar C solver implementation
│       ├── python_solver/             # Python-based solvers
│       └── README.md                  # Complete documentation
│
├── 📂 utils/                          # Shared Utilities
│   ├── __init__.py
│   ├── data_utils.py                  # Data loading & preprocessing
│   ├── plotting.py                    # Visualization functions
│   └── metrics.py                     # Evaluation metrics
│
├── requirements.txt                   # Python dependencies
├── LICENSE                            # MIT License
└── README.md                          # This file
```

## 🚀 Quick Start

### Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/Michael-Jackson666/AI4CFD.git
   cd AI4CFD
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Install PyTorch (if not already installed):**
   ```bash
   # CPU version
   pip install torch torchvision torchaudio
   
   # GPU version (CUDA 11.8)
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
   ```

### Getting Started with Each Method

#### PINNs - Start Here for Beginners 🎓
```bash
cd PINNs/
jupyter notebook tutorial/tutorial_eng.ipynb  # English tutorial
# or
jupyter notebook tutorial/tutorial_chinese.ipynb  # 中文教程
```

#### DeepONet - Learn Operator Learning 🎯
```bash
cd DeepONet/
jupyter notebook tutorial/operator_learning_torch.ipynb
# or run pure PyTorch example
python operator_learning_pure.py
```

#### FNO - Frequency Domain Methods 🌊
```bash
cd FNO/
# See README.md for specific examples
```

#### Vlasov-Poisson System - Plasma Physics Application 🌌
```bash
cd VP_system/TwoStreamInstability/

# Option 1: Python solver (quick start)
cd python_solver/
python quick_start.py

# Option 2: HyPar C solver (high performance)
cd hypar_solver/
./compile_hypar.sh
```

## 🎯 Key Applications

### Fluid Dynamics
- **Navier-Stokes Equations**: Incompressible flow, lid-driven cavity, flow past cylinder
- **Turbulence Modeling**: Reynolds-averaged and large-eddy simulations
- **Shape Optimization**: Airfoil design, pipe flow optimization

### Heat Transfer
- **Conduction & Convection**: 1D/2D heat equations with various boundary conditions
- **Multi-physics Coupling**: Thermal-fluid interaction problems

### Plasma Physics 🌌
- **Vlasov-Poisson Systems**: Kinetic plasma simulations
- **Two-Stream Instability**: Classic plasma instability phenomenon
- **Phase Space Dynamics**: Distribution function evolution

### General PDEs
- **Poisson Equation**: Electrostatics, steady-state diffusion
- **Burgers' Equation**: Shock wave formation, nonlinear transport
- **Darcy Flow**: Porous media, subsurface flow

### General PDEs
- **Poisson Equation**: Electrostatics, steady-state diffusion
- **Burgers' Equation**: Shock wave formation, nonlinear transport
- **Darcy Flow**: Porous media, subsurface flow

## 📊 Comparison: AI Methods vs Traditional Solvers

| Feature | Traditional CFD | PINNs | DeepONet | FNO |
|---------|----------------|-------|----------|-----|
| **Training Required** | ❌ No | ✅ Yes | ✅ Yes | ✅ Yes |
| **Single Solve Time** | Hours | Seconds | Milliseconds | Milliseconds |
| **Parameter Sweep** | Re-solve each | Re-train or transfer | One forward pass | One forward pass |
| **Data Requirements** | N/A | Low (physics-informed) | High (need solutions) | High (need solutions) |
| **Accuracy** | Very High | Medium-High | Medium-High | High |
| **Mesh Independence** | ❌ No | ✅ Yes | ✅ Yes | ✅ Yes |
| **Best For** | High-accuracy single solve | Limited data, complex BC | Fast multi-query | Turbulence, high-res |

**When to use each method:**
- **PINNs**: Limited data, complex geometries, inverse problems, parameter identification
- **DeepONet**: Multi-query problems, real-time prediction, parametric studies
- **FNO**: High-resolution periodic problems, turbulence, weather prediction
- **Traditional CFD**: Highest accuracy requirements, single high-fidelity simulation

## 🔧 Requirements

### Core Dependencies
- **Python**: 3.8 or higher
- **PyTorch**: 1.10 or higher (some implementations support 1.8+)
- **NumPy**: 1.20+
- **SciPy**: 1.7+
- **Matplotlib**: 3.3+
- **Jupyter**: For interactive tutorials

### Optional Dependencies
- **DeepXDE**: For PINNs examples (install: `pip install deepxde`)
- **CUDA**: For GPU acceleration (highly recommended)
- **TensorBoard**: For training visualization

### Installation
All dependencies can be installed via:
```bash
pip install -r requirements.txt
```

For detailed requirements, see `requirements.txt`.

## 📖 Documentation

Each method has comprehensive documentation:

- **PINNs/README.md**: Complete guide to PINNs with 8 tutorials, installation, theory, and VP system
- **DeepONet/README.md**: Operator learning guide with architecture details, VP operator learning
- **FNO/README.md**: Fourier neural operators with implementation details
- **Transformer/README.md**: Transformer-based methods for PDEs
- **VP_system/TwoStreamInstability/README.md**: Plasma physics applications with both C and Python solvers

## 🌟 Highlights

### Comprehensive Tutorials
- **Bilingual Support**: English and Chinese tutorials for PINNs
- **Progressive Learning**: From basic concepts to advanced applications
- **Interactive**: Jupyter notebooks with executable code cells
- **Well-commented**: Detailed explanations of each step

### Research-Grade Implementations
- **VP-PINNs**: Advanced plasma physics with MLP and Transformer architectures
- **VP-DeepONet**: Operator learning for Vlasov-Poisson systems
- **Model Comparison**: Tools to benchmark different architectures
- **Visualization Suite**: Comprehensive plotting for phase space, fields, and distributions

### Production-Ready Code
- **Modular Design**: Easy to adapt for custom problems
- **Well-tested**: Validated on benchmark problems
- **Documented**: Clear code structure with docstrings
- **Extensible**: Easy to add new PDEs and architectures

### Production-Ready Code
- **Modular Design**: Easy to adapt for custom problems
- **Well-tested**: Validated on benchmark problems
- **Documented**: Clear code structure with docstrings
- **Extensible**: Easy to add new PDEs and architectures

## 📚 Learning Path

### Beginner Track 🌱
1. Start with **PINNs/tutorial/tutorial_eng.ipynb** (or Chinese version)
2. Try simple examples: Poisson 1D, Heat 1D
3. Understand automatic differentiation and physics loss
4. Experiment with different network architectures

### Intermediate Track 🌿
1. Explore **DeepONet/tutorial/operator_learning_torch.ipynb**
2. Learn operator learning concepts
3. Try 2D problems: Heat 2D, Navier-Stokes
4. Compare PINNs vs DeepONet performance

### Advanced Track 🌳
1. Study **VP system implementations** in both PINNs and DeepONet
2. Understand Transformer architectures for PDEs
3. Explore FNO for high-resolution problems
4. Implement custom PDEs and compare all methods

### Research Track 🚀
1. Dive into **vp_system/** directories
2. Study model comparison tools
3. Experiment with hybrid architectures
4. Contribute new methods or improvements

## 🎓 Citation

If you use this repository in your research, please cite:

```bibtex
@software{ai4cfd2025,
  author = {Michael-Jackson666},
  title = {AI4CFD: AI Methods for Computational Fluid Dynamics},
  year = {2025},
  url = {https://github.com/Michael-Jackson666/AI4CFD}
}
```

### Key Papers

**PINNs:**
- Raissi, M., Perdikaris, P., & Karniadakis, G. E. (2019). Physics-informed neural networks: A deep learning framework for solving forward and inverse problems involving nonlinear partial differential equations. *Journal of Computational Physics*, 378, 686-707.

**DeepONet:**
- Lu, L., Jin, P., Pang, G., Zhang, Z., & Karniadakis, G. E. (2021). Learning nonlinear operators via DeepONet based on the universal approximation theorem of operators. *Nature Machine Intelligence*, 3(3), 218-229.

**FNO:**
- Li, Z., Kovachki, N., Azizzadenesheli, K., Liu, B., Bhattacharya, K., Stuart, A., & Anandkumar, A. (2020). Fourier neural operator for parametric partial differential equations. *arXiv preprint arXiv:2010.08895*.

## 🤝 Contributing

Contributions are welcome! Here's how you can help:

### Ways to Contribute
- 🐛 **Bug Reports**: Open an issue describing the bug
- 💡 **Feature Requests**: Suggest new methods or improvements
- 📝 **Documentation**: Improve or translate documentation
- 🔧 **Code**: Submit pull requests with new features or fixes
- 🎓 **Tutorials**: Add new tutorial notebooks
- 🧪 **Examples**: Contribute new application examples

### Contribution Guidelines
1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

Please ensure:
- Code follows existing style conventions
- New features include tests and documentation
- Tutorials are well-commented and executable
- README files are updated accordingly

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 📧 Contact & Support

### Questions or Issues?
- 🐛 **Bug Reports**: [Open an issue](https://github.com/Michael-Jackson666/AI4CFD/issues)
- 💬 **Discussions**: Use GitHub Discussions for general questions
- 📧 **Email**: Contact the maintainers through GitHub

### Stay Updated
- ⭐ **Star** this repository to follow updates
- 👀 **Watch** for new releases and features
- 🔔 **Follow** the repository for notifications

## 🙏 Acknowledgments

This repository builds upon the foundational work of many researchers in the field of scientific machine learning. Special thanks to:

- The Karniadakis group at Brown University for pioneering PINNs and DeepONet
- The Anandkumar group at Caltech for developing FNO
- The open-source community for PyTorch, NumPy, and SciPy
- All contributors and users providing feedback

## 📊 Repository Stats

- **Methods Implemented**: 4 major AI-PDE methods
- **Tutorial Notebooks**: 9+ interactive Jupyter notebooks
- **Supported PDEs**: 10+ different equation types
- **Languages**: Bilingual (English & Chinese)
- **Applications**: Fluid dynamics, heat transfer, plasma physics, and more

---

<div align="center">

**⭐ Star this repository if you find it useful! ⭐**

*Bridging AI research and practical PDE solving for computational fluid dynamics*

[Report Bug](https://github.com/Michael-Jackson666/AI4CFD/issues) • [Request Feature](https://github.com/Michael-Jackson666/AI4CFD/issues) • [Documentation](https://github.com/Michael-Jackson666/AI4CFD)

</div>
