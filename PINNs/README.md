# Physics-Informed Neural Networks (PINNs)

Physics-Informed Neural Networks (PINNs) are a class of neural networks that incorporate physical laws, described by partial differential equations (PDEs), directly into the training process. This is achieved by encoding the PDEs as regularization terms in the loss function using automatic differentiation.

## 🎯 Key Concepts

### What are PINNs?
PINNs solve PDEs by:
1. **Neural Network Approximation**: Using a neural network to approximate the solution u(x,t)
2. **Physics Constraints**: Incorporating PDE residuals as loss terms
3. **Automatic Differentiation**: Computing derivatives needed for PDEs automatically
4. **Multi-objective Training**: Balancing data fitting and physics compliance

### Advantages
- Can solve PDEs with limited or noisy data
- Naturally handle complex geometries
- Incorporate prior physical knowledge
- Provide smooth, differentiable solutions

### Applications
- Fluid dynamics (Navier-Stokes equations)
- Heat transfer problems
- Wave propagation
- Inverse problems and parameter estimation

## 📁 Files in this Directory

### Core Files
- `train.py` - Complete training script for various PDE problems
- `models.py` - Neural network architectures for PINNs
- `pde_definitions.py` - Common PDE definitions and residual functions

### Tutorial Notebooks (`tutorial/`)
- `tutorial_chinese.ipynb` - 完整的中文PINNs教程 (Complete Chinese PINNs Tutorial)
- `tutorial_eng.ipynb` - English PINNs Tutorial
- `possion_1d.ipynb` - 1D Poisson equation tutorial
- `heat_2d.ipynb` - 2D heat equation examples
- `ns_basic.ipynb` - Basic Navier-Stokes equations
- `ns_advanced.ipynb` - Advanced Navier-Stokes examples
- `system_pde.ipynb` - System of PDEs tutorial
- `vlasov_poisson.ipynb` - Vlasov-Poisson system

### Examples (`examples/`)
- `possion_dirichlet_1d.py` - 1D Poisson equation with Dirichlet BC using DeepXDE

### Additional Directories
- `vp_system/` - Vlasov-Poisson system implementations

## 🚀 Quick Start

### Running the Tutorials
```bash
# Chinese tutorial (recommended for beginners)
jupyter notebook tutorial/tutorial_chinese.ipynb

# English tutorial
jupyter notebook tutorial/tutorial_eng.ipynb

# Specific PDE examples
jupyter notebook tutorial/possion_1d.ipynb
jupyter notebook tutorial/heat_2d.ipynb
```

### Running DeepXDE Examples
```bash
cd examples/
python possion_dirichlet_1d.py
```

### Training Custom Models
```bash
python train.py --problem poisson --epochs 10000 --lr 0.001
```

### Available Problems
- `poisson` - 1D/2D Poisson equation
- `heat` - Heat equation (1D/2D)
- `burgers` - Burgers' equation
- `navier_stokes` - Navier-Stokes equations
- `vlasov_poisson` - Vlasov-Poisson system

## 📊 Example Results

The tutorials demonstrate solving:

### Basic PDEs
1. **1D Poisson Equation**: ∂²u/∂x² = f(x)
2. **2D Heat Equation**: ∂u/∂t = α∇²u
3. **1D Burgers' Equation**: ∂u/∂t + u∂u/∂x = ν∂²u/∂x²

### Advanced Applications
4. **2D Navier-Stokes**: Fluid flow simulations
5. **Vlasov-Poisson System**: Plasma physics applications
6. **System of PDEs**: Coupled equation systems

### Implementation Approaches
- **Pure PyTorch**: From-scratch implementations with manual gradient computation
- **DeepXDE**: Using the DeepXDE library for rapid prototyping
- **Comparative Analysis**: Performance comparison between different approaches

## 🔧 Implementation Details

### Network Architecture
- Fully connected neural networks with tanh activations
- Input: spatial/temporal coordinates
- Output: solution values

### Loss Function
```
L = L_data + λ_pde * L_pde + λ_bc * L_bc
```

Where:
- `L_data`: Data fitting loss (if training data available)
- `L_pde`: PDE residual loss
- `L_bc`: Boundary condition loss

### Training Strategy
1. Initialize network with Xavier/He initialization
2. Sample collocation points in domain
3. Compute PDE residuals using automatic differentiation
4. Optimize combined loss using L-BFGS or Adam

## 📚 Mathematical Background

For a general PDE:
```
F(x, u, ∂u/∂x, ∂²u/∂x², ...) = 0
```

PINNs minimize:
```
min Σ |F(x_i, u_θ(x_i), ∇u_θ(x_i), ...)|² + BC_loss + Data_loss
```

The neural network u_θ(x) with parameters θ learns to satisfy both the PDE and boundary conditions.

## 🔗 References

1. Raissi, M., Perdikaris, P., & Karniadakis, G. E. (2019). Physics-informed neural networks: A deep learning framework for solving forward and inverse problems involving nonlinear partial differential equations. Journal of Computational Physics, 378, 686-707.

2. Karniadakis, G. E., Kevrekidis, I. G., Lu, L., Perdikaris, P., Wang, S., & Yang, L. (2021). Physics-informed machine learning. Nature Reviews Physics, 3(6), 422-440.

3. Lu, L., Meng, X., Mao, Z., & Karniadakis, G. E. (2021). DeepXDE: A deep learning library for solving differential equations. SIAM review, 63(1), 208-228.

## 🌟 Features

### Comprehensive Tutorials
- **Bilingual Support**: Both Chinese and English tutorials available
- **Progressive Learning**: From basic concepts to advanced applications
- **Interactive Examples**: Jupyter notebooks with step-by-step explanations
- **Visualization Tools**: Comprehensive plotting and analysis functions

### Multiple Implementation Styles
- **Educational**: Pure PyTorch implementations for learning
- **Production**: DeepXDE-based examples for practical applications
- **Comparative**: Side-by-side performance analysis

### Advanced Topics Covered
- Multi-scale neural networks
- Adaptive weighting strategies
- Uncertainty quantification
- Inverse problem solving
- Parameter identification