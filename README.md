# AI4CFD: AI Methods for Computational Fluid Dynamics

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.10+-ee4c2c.svg)](https://pytorch.org/)

Complete implementations of deep learning methods for solving Partial Differential Equations (PDEs), including tutorials, training code, and real-world applications.

## 🧠 Core Methods

### 1. Physics-Informed Neural Networks (PINNs)
**Neural networks that encode physical laws directly into the loss function**

Core Idea: Construct physics-constrained loss using automatic differentiation:
```
Loss = MSE(Boundary) + MSE(Initial) + MSE(PDE Residual)
```

- **Advantages**: Low data requirements, handles complex boundaries, suitable for inverse problems
- **Location**: `PINNs/`
- **Tutorials**: 8 Jupyter notebooks (English & Chinese), from basics to advanced
- **Applications**: Poisson, Heat, Navier-Stokes, Vlasov-Poisson systems

### 2. Deep Operator Networks (DeepONet)
**Neural networks learning mappings between infinite-dimensional function spaces**

Core Idea: Branch-Trunk architecture for learning operators G: u → G(u):
```
DeepONet(u)(y) = Σᵢ bᵢ(u) · tᵢ(y)
Branch network bᵢ: Encodes input function u
Trunk network tᵢ: Encodes output location y
```

- **Advantages**: Train once, fast inference (milliseconds), efficient for multi-query
- **Location**: `DeepONet/`
- **Tutorials**: Pure PyTorch implementation tutorial
- **Applications**: Heat equation, Burgers, Darcy flow, VP systems

### 3. Fourier Neural Operators (FNO)
**Neural operators solving PDEs in frequency domain**

Core Idea: Convolution in Fourier space for global information propagation:
```
v(x) = σ(W·u(x) + (K*u)(x))
where K*u computed in frequency domain: F⁻¹(R · F(u))
```

- **Advantages**: Resolution-invariant, excellent for periodic problems, high-resolution solving
- **Location**: `FNO/`
- **Applications**: Navier-Stokes, turbulence modeling, Darcy flow

### 4. Tensor Neural Networks (TNN)
**Neural networks using tensor decomposition for high-dimensional PDEs**

Core Idea: Decompose high-dimensional functions into products of low-dimensional functions:
```
u(x₁,...,xₐ) ≈ Σᵢ αᵢ · ∏ₖ φₖ⁽ⁱ⁾(xₖ)
```

- **Advantages**: Linear parameter growth (vs exponential), suitable for high-dimensional problems
- **Location**: `TNN/`
- **Tutorials**: Complete Jupyter tutorial and 5D examples
- **Applications**: 5D Poisson equation, high-dimensional PDE solving

### 5. Transformer-based Methods
**Sequence models using attention mechanisms for PDE solving**

Core Idea: Capture long-range dependencies in spatial/temporal domains via self-attention

- **Advantages**: Long-range dependency capture, flexible architecture design
- **Location**: `Transformer/`
- **Applications**: Time-series prediction, multi-physics coupling


## 📁 Project Structure

```
AI4CFD/
├── PINNs/              # Physics-Informed Neural Networks
│   ├── tutorial/       # 8 tutorials (English & Chinese)
│   ├── vp_system/      # Vlasov-Poisson system implementation
│   └── README.md
├── DeepONet/           # Deep Operator Networks
│   ├── tutorial/       # PyTorch implementation tutorial
│   ├── vp_system/      # VP operator learning
│   └── README.md
├── FNO/                # Fourier Neural Operators
│   └── README.md
├── TNN/                # Tensor Neural Networks
│   ├── tutorial/       # Complete tutorial
│   ├── train/dim5/     # 5D PDE solving example
│   └── README.md
├── Transformer/        # Transformer-based methods
│   └── README.md
├── VP_system/          # Vlasov-Poisson applications
│   └── TwoStreamInstability/  # Two-stream instability
└── utils/              # Shared utility functions
```

## 🚀 Quick Start

### Installation

```bash
git clone https://github.com/Michael-Jackson666/AI4CFD.git
cd AI4CFD
pip install -r requirements.txt
```

### Run Examples

**PINNs Tutorial (Recommended for Beginners)**:
```bash
cd PINNs/tutorial
jupyter notebook tutorial_eng.ipynb  # English tutorial
# or
jupyter notebook tutorial_chinese.ipynb  # Chinese tutorial
```

**TNN 5D Example**:
```bash
cd TNN/train/dim5
python ex_5_1_dim5.py
```

**DeepONet Tutorial**:
```bash
cd DeepONet/tutorial
jupyter notebook operator_learning_torch.ipynb
```

## 📊 Method Comparison

| Method | Training Data | Single Solve Speed | Parameter Query | Best For |
|--------|--------------|-------------------|-----------------|----------|
| **PINNs** | Low (physics-informed) | Seconds | Re-train needed | Complex boundaries, inverse problems, data scarcity |
| **DeepONet** | High (needs solutions) | Milliseconds | One forward pass | Multi-query, real-time prediction |
| **FNO** | High (needs solutions) | Milliseconds | One forward pass | Periodic problems, turbulence, high-resolution |
| **TNN** | Medium | Seconds | Re-train needed | High-dimensional problems (5D+) |
| **Transformer** | High (needs solutions) | Milliseconds | One forward pass | Time-series, long-range dependencies |

## 🎯 Typical Applications

- **Fluid Dynamics**: Navier-Stokes equations, turbulence modeling, shape optimization
- **Heat Transfer**: Heat equation, convection-diffusion, multi-physics coupling
- **Plasma Physics**: Vlasov-Poisson systems, two-stream instability
- **General PDEs**: Poisson equation, Burgers equation, Darcy flow

## 🔧 Dependencies

- Python >= 3.8
- PyTorch >= 1.10
- NumPy, SciPy, Matplotlib
- Jupyter (optional, for tutorials)

## 📖 Documentation

Each method has detailed README documentation:
- `PINNs/README.md`: Complete PINNs guide and tutorial index
- `DeepONet/README.md`: Operator learning detailed explanation
- `TNN/README.md`: Tensor neural network theory and implementation
- `TNN/train/dim5/README.md`: 5D PDE solving example guide

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

**⭐ Star this repository if you find it helpful! ⭐**
