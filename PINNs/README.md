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

- `tutorial.ipynb` - Interactive Jupyter notebook tutorial
- `train.py` - Complete training script for various PDE problems
- `models.py` - Neural network architectures for PINNs
- `pde_definitions.py` - Common PDE definitions and residual functions

## 🚀 Quick Start

### Running the Tutorial
```bash
jupyter notebook tutorial.ipynb
```

### Training a Model
```bash
python train.py --problem poisson --epochs 10000 --lr 0.001
```

### Available Problems
- `poisson` - 1D/2D Poisson equation
- `burgers` - Burgers' equation
- `heat` - Heat equation
- `wave` - Wave equation

## 📊 Example Results

The tutorial demonstrates solving:
1. **1D Poisson Equation**: ∂²u/∂x² = f(x)
2. **2D Poisson Equation**: ∇²u = f(x,y) 
3. **Burgers' Equation**: ∂u/∂t + u∂u/∂x = ν∂²u/∂x²

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