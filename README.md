# AI4CFD: AI Methods for Computational Fluid Dynamics

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.10+-ee4c2c.svg)](https://pytorch.org/)

Complete implementations of deep learning methods for solving Partial Differential Equations (PDEs), including tutorials, training code, and real-world applications.

## 🧠 Core Methods

### 1. Physics-Informed Neural Networks (PINNs)
**Neural networks that encode physical laws directly into the loss function**

Core Idea: Construct physics-constrained loss using automatic differentiation:

$$
\begin{aligned}
\mathcal{L} = \mathcal{L}_{data} + \mathcal{L}_{PDE} = \frac{1}{N_b}\sum_{i=1}^{N_b}\left\|u(x_b^i) - u_b^i\right\|^2 + \frac{1}{N_f}\sum_{j=1}^{N_f}\left\|\mathcal{N} \left[u\right] (x_f^j)\right\|^2
\end{aligned}
$$

- **Advantages**: Low data requirements, handles complex boundaries, suitable for inverse problems
- **Location**: `PINNs/`
- **Tutorials**: 8 Jupyter notebooks (English & Chinese), from basics to advanced
- **Applications**: Poisson, Heat, Navier-Stokes, Vlasov-Poisson systems

### 2. Deep Operator Networks (DeepONet)
**Neural networks learning mappings between infinite-dimensional function spaces**

Core Idea: Branch-Trunk architecture for learning operators $G: u \to G(u)$:

$$
G(u)(y) \approx \sum_{k=1}^{p} \underbrace{b_k(u)}_{\text{Branch}} \cdot \underbrace{t_k(y)}_{\text{Trunk}}
$$

- **Advantages**: Train once, fast inference (milliseconds), efficient for multi-query
- **Location**: `DeepONet/`
- **Tutorials**: Pure PyTorch implementation tutorial
- **Applications**: Heat equation, Burgers, Darcy flow, VP systems

### 3. Fourier Neural Operators (FNO)
**Neural operators solving PDEs in frequency domain**

Core Idea: Convolution in Fourier space for global information propagation:

$$
v_{t+1}(x) = \sigma\left( W v_t(x) + \mathcal{F}^{-1}(R \cdot \mathcal{F}(v_t))(x) \right)
$$

- **Advantages**: Resolution-invariant, excellent for periodic problems, high-resolution solving
- **Location**: `FNO/`
- **Applications**: Navier-Stokes, turbulence modeling, Darcy flow

### 4. Physics-Informed Neural Operators (PINO)
**Neural operators combining FNO architecture with physics constraints**

Core Idea: Learn solution operators with both data fitting and PDE residuals:

$$
\mathcal{L}_{PINO} = \underbrace{\mathcal{L}_{data}}_{\text{Data Loss}} + \lambda \underbrace{\mathcal{L}_{PDE}}_{\text{Physics Loss}}
$$

- **Advantages**: Less data than FNO, better generalization than PINNs, fast inference, physics-consistent
- **Location**: `PINO/`
- **Tutorials**: Complete overview notebook, 3 example implementations (Burgers, Darcy, Heat)
- **Applications**: Burgers equation, Darcy flow, heat conduction, parametric PDEs

### 5. Tensor Neural Networks (TNN)
**Neural networks using tensor decomposition for high-dimensional PDEs**

Core Idea: Decompose high-dimensional functions into products of low-dimensional functions:

$$
u(x_1, \dots, x_d) \approx \sum_{r=1}^{R} \prod_{k=1}^{d} \phi_k^{(r)}(x_k)
$$

- **Advantages**: Linear parameter growth (vs exponential), suitable for high-dimensional problems
- **Location**: `TNN/`
- **Tutorials**: Complete Jupyter tutorial and 5D examples
- **Applications**: 5D Poisson equation, high-dimensional PDE solving

### 6. Kolmogorov-Arnold Networks (KAN)
**Neural networks with learnable activation functions inspired by Kolmogorov-Arnold representation theorem**

Core Idea: Replace fixed activation functions with learnable univariate functions (B-splines):

$$
f(x_1, \dots, x_n) = \sum_{q=0}^{2n} \Phi_q\left(\sum_{p=1}^{n} \phi_{q,p}(x_p)\right)
$$

- **Advantages**: Higher accuracy with fewer parameters, interpretable, excellent for smooth PDEs
- **Location**: `KAN/`
- **Tutorials**: Complete PDE solving tutorial with B-spline implementation
- **Applications**: Poisson, Heat, Burgers equations, smooth PDE problems

### 7. Transformer-based Methods
**Sequence models using attention mechanisms for PDE solving**

Core Idea: Capture long-range dependencies in spatial/temporal domains via self-attention:

$$
\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V
$$

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
├── PINO/               # Physics-Informed Neural Operators
│   ├── tutorial/       # Overview notebook
│   ├── examples/       # Burgers, Darcy, Heat examples
│   └── README.md
├── KAN/                # Kolmogorov-Arnold Networks
│   ├── tutorial/       # Complete PDE solving tutorial
│   ├── examples/       # Poisson, Heat, Burgers examples
│   ├── models.py       # KAN & B-spline implementation
│   └── utils.py        # Utility functions
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

**PINO Tutorial (Physics-Informed Operators)**:
```bash
cd PINO/tutorial
jupyter notebook pino_overview.ipynb
```

**DeepONet Tutorial**:
```bash
cd DeepONet/tutorial
jupyter notebook operator_learning_torch.ipynb
```

**KAN Tutorial (Kolmogorov-Arnold Networks)**:
```bash
cd KAN/tutorial
jupyter notebook kan_pde_tutorial.ipynb
```

**KAN Examples**:
```bash
cd KAN/examples
python poisson_1d.py   # 1D Poisson equation
python heat_1d.py      # 1D heat equation
python burgers_1d.py   # 1D Burgers equation
```

## 📊 Method Comparison

| Method | Training Data | Single Solve Speed | Parameter Query | Best For |
|--------|--------------|-------------------|-----------------|----------|
| **PINNs** | Low (physics-informed) | Seconds | Re-train needed | Complex boundaries, inverse problems, data scarcity |
| **DeepONet** | High (needs solutions) | Milliseconds | One forward pass | Multi-query, real-time prediction |
| **FNO** | High (needs solutions) | Milliseconds | One forward pass | Periodic problems, turbulence, high-resolution |
| **PINO** | Medium (data + physics) | Milliseconds | One forward pass | Parametric PDEs, less data scenarios, physics-consistent operators |
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
- `FNO/README.md`: Fourier Neural Operator implementation
- `PINO/README.md`: Physics-Informed Neural Operators guide
- `TNN/README.md`: Tensor neural network theory and implementation
- `TNN/train/dim5/README.md`: 5D PDE solving example guide

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

**⭐ Star this repository if you find it helpful! ⭐**
