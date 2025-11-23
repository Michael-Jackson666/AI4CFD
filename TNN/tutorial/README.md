# TNN Tutorial

## Overview

This tutorial provides a comprehensive introduction to **Tensor Neural Networks (TNN)** for solving high-dimensional partial differential equations (PDEs). The tutorial implements TNN from scratch using PyTorch and demonstrates its application on a 2D Poisson equation.

## What is TNN?

Tensor Neural Network is a specialized neural network architecture designed for high-dimensional PDE problems. It leverages **tensor decomposition** to represent the solution as:

$$
u(x_1, x_2, \ldots, x_d) = \sum_{k=1}^{r} \prod_{i=1}^{d} \phi_i^{(k)}(x_i)
$$

where:
- $r$ is the **rank** (number of terms in decomposition)
- $\phi_i^{(k)}$ are 1D neural networks for each dimension
- This reduces parameters from $O(N^d)$ to $O(drN)$

## Tutorial Contents

### 1. Introduction
- TNN overview and advantages
- Comparison with standard neural networks
- Tutorial roadmap

### 2. Preparation
- Library imports (PyTorch, NumPy, Matplotlib)
- Chinese font configuration for plots

### 3. Core Components Implementation

#### 3.1 Boundary Functions
- `bd(x)`: Boundary distance function
- `grad_bd(x)`: Gradient of boundary function
- `grad_grad_bd(x)`: Hessian of boundary function
- Purpose: Enforce homogeneous Dirichlet boundary conditions

#### 3.2 TNN Linear Layer
- Custom linear transformation layer
- Batch processing for tensor decomposition
- Weight shape: `[dim, n_out, n_in]`

#### 3.3 Activation Function (TNN_Sin)
- Sine activation: `σ(x) = sin(x)`
- First derivative: `σ'(x) = cos(x)`
- Second derivative: `σ''(x) = -sin(x)`
- Smooth and periodic for PDE problems

#### 3.4 Complete TNN Model
- `SimpleTNN` class implementation
- Supports computing values, gradients, and Hessians
- Automatic boundary condition enforcement
- Orthogonal weight initialization

### 4. Application: 2D Poisson Equation

**Problem**: Solve the following PDE on $\Omega = [-1, 1]^2$:

$$
\begin{cases}
-\Delta u = f(x, y) & \text{in } \Omega \\
u = 0 & \text{on } \partial\Omega
\end{cases}
$$

where $f(x, y) = 2\pi^2 \sin(\pi x) \sin(\pi y)$

**Exact solution**: $u(x, y) = \sin(\pi x) \sin(\pi y)$

**Variational formulation**: Minimize energy functional

$$
E(u) = \frac{1}{2}\int_\Omega |\nabla u|^2 dx - \int_\Omega f \cdot u \, dx
$$

#### Implementation Steps:
1. **Quadrature setup**: Gauss-Legendre integration nodes (40×40)
2. **Loss function**: Energy functional with numerical integration
3. **Training**: Adam optimizer, 5000 epochs
4. **Visualization**: Solution comparison and error analysis

### 5. Results Visualization
- Training loss convergence curve
- Exact solution vs TNN prediction comparison
- Absolute error distribution heatmap
- Center-line profile comparison
- L2 and maximum error metrics

### 6. Summary and Extensions
- TNN advantages for high-dimensional problems
- Pointers to advanced topics:
  - Non-homogeneous boundary conditions (ex_5_2)
  - Neumann boundary conditions (ex_5_3)
  - Eigenvalue problems (ex_5_4)
  - Unbounded domains (ex_5_5)
  - Higher dimensions (5D, 10D, 20D examples)

## Requirements

```bash
pip install torch numpy matplotlib
```

## Usage

Open the Jupyter notebook and run cells sequentially:

```bash
jupyter notebook TNN_tutorial.ipynb
```

The tutorial is self-contained and includes:
- ✅ All necessary imports
- ✅ Complete code implementation
- ✅ Mathematical explanations
- ✅ Visualization examples
- ✅ Working 2D Poisson equation demo

## Key Features

1. **From Scratch Implementation**: Build TNN step-by-step with clear explanations
2. **Mathematical Rigor**: Detailed derivations and variational formulations
3. **Practical Example**: Complete workflow from problem definition to visualization
4. **Chinese Language**: All explanations in Chinese for better understanding
5. **Executable Code**: All cells can be run directly to reproduce results

## Expected Results

After training (5000 epochs with rank=50):
- **L2 Error**: ~10⁻⁴ to 10⁻⁵
- **Max Error**: ~10⁻³ to 10⁻⁴
- **Training Time**: ~2-5 minutes (CPU)

## Learning Outcomes

After completing this tutorial, you will:
- ✅ Understand tensor decomposition for neural networks
- ✅ Implement TNN from scratch in PyTorch
- ✅ Know how to enforce boundary conditions automatically
- ✅ Apply variational methods for PDE solving
- ✅ Use Gauss-Legendre quadrature for integration
- ✅ Visualize and analyze PDE solutions

## References

For more advanced examples, see the `../example/` directory:
- `ex_5_1/`: Homogeneous Dirichlet BC (dimensions 5, 10, 20)
- `ex_5_2/`: Non-homogeneous BC with two-TNN decomposition
- `ex_5_3/`: Neumann BC with weak formulation
- `ex_5_4/`: Eigenvalue problems
- `ex_5_5/`: Unbounded domains with Hermite-Gauss quadrature

## Author

Created as part of the AI4CFD project.

## License

See the main repository LICENSE file.
