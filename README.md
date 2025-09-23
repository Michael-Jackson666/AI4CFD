# AI4CFD

Various methods for solving PDEs using artificial intelligence

## Overview

AI4CFD is a comprehensive Python library that implements state-of-the-art artificial intelligence methods for solving partial differential equations (PDEs), with a focus on computational fluid dynamics (CFD) applications. The library provides implementations of several cutting-edge AI methods:

- **Physics-Informed Neural Networks (PINNs)** - Incorporates physics laws directly into neural network training
- **Deep Operator Networks (DeepONet)** - Learns operators that map functions to functions
- **Fourier Neural Operators (FNO)** - Efficient neural operators in Fourier space
- **Neural Ordinary Differential Equations (Neural ODEs)** - Continuous-time neural networks for dynamic systems

## Installation

### From Source

```bash
git clone https://github.com/Michael-Jackson666/AI4CFD.git
cd AI4CFD
pip install -e .
```

### Dependencies

The library requires:
- Python ≥ 3.8
- PyTorch ≥ 1.12.0
- NumPy ≥ 1.21.0
- SciPy ≥ 1.7.0
- Matplotlib ≥ 3.5.0
- scikit-learn ≥ 1.0.0

## Quick Start

### Physics-Informed Neural Networks (PINNs)

```python
import torch
from ai4cfd.pinns import PINNs, heat_equation_residual
from ai4cfd.utils import generate_collocation_points, train_pinn

# Define problem domain
domain_bounds = [(0.0, 1.0), (0.0, 1.0)]  # [x_min, x_max], [t_min, t_max]

# Generate collocation points
interior_points, boundary_points, initial_points = generate_collocation_points(
    domain_bounds=domain_bounds,
    num_interior=1000,
    num_boundary=200,
    num_initial=100
)

# Create PINN model
model = PINNs(input_dim=2, hidden_dim=50, output_dim=1, num_layers=4)

# Define PDE residual function
def pde_residual(x, u):
    return heat_equation_residual(x, u, alpha=0.01)

# Train the model
trained_model, loss_tracker = train_pinn(
    model=model,
    pde_func=pde_residual,
    interior_points=interior_points,
    boundary_points=boundary_points,
    boundary_values=torch.zeros(len(boundary_points), 1),
    num_epochs=2000
)
```

### Deep Operator Networks (DeepONet)

```python
from ai4cfd.deeponet import DeepONet

# Create DeepONet model
model = DeepONet(
    branch_input_dim=100,  # Number of sensors
    trunk_input_dim=1,     # Evaluation coordinates
    branch_hidden_dims=[128, 128],
    trunk_hidden_dims=[128, 128],
    output_dim=128
)

# Forward pass
branch_input = torch.randn(32, 100)  # Function values at sensors
trunk_input = torch.randn(32, 1)     # Evaluation points
output = model(branch_input, trunk_input)
```

### Fourier Neural Operators (FNO)

```python
from ai4cfd.fno import FNO2d

# Create FNO model for 2D problems
model = FNO2d(
    modes1=12,
    modes2=12,
    width=32,
    num_layers=4,
    input_dim=3,  # Initial condition + coordinates
    output_dim=1
)

# Forward pass
input_data = torch.randn(10, 64, 64, 3)  # Batch of 2D fields
output = model(input_data)
```

### Neural ODEs

```python
from ai4cfd.neural_ode import NeuralODE, ODEFunc

# Create ODE function network
ode_func = ODEFunc(input_dim=2, hidden_dims=[64, 64])

# Create Neural ODE
neural_ode = NeuralODE(ode_func)

# Solve ODE
y0 = torch.randn(10, 2)  # Initial conditions
t = torch.linspace(0, 1, 100)  # Time points
solution = neural_ode(y0, t)
```

## Examples

The `examples/` directory contains complete examples demonstrating each method:

- `heat_equation_pinn.py` - Solving 1D heat equation with PINNs
- `antiderivative_deeponet.py` - Learning antiderivative operator with DeepONet
- `burgers_2d_fno.py` - Solving 2D Burgers' equation with FNO

Run examples:
```bash
cd examples
python heat_equation_pinn.py
```

## Method Details

### Physics-Informed Neural Networks (PINNs)

PINNs embed physical laws, represented as PDEs, into the training process of neural networks. The method minimizes a loss function that includes:
- PDE residual loss
- Boundary condition loss
- Initial condition loss
- Data fitting loss (if available)

**Key Features:**
- Automatic differentiation for computing derivatives
- Flexible boundary and initial condition handling
- Support for various PDE types
- Built-in residual functions for common equations

### Deep Operator Networks (DeepONet)

DeepONet approximates nonlinear operators G that map input functions to output functions: G(u)(y) ≈ ∑ᵢ bᵢ(u) tᵢ(y), where:
- Branch network encodes the input function u
- Trunk network encodes the evaluation coordinates y
- The output is their dot product

**Key Features:**
- Branch-trunk architecture for operator learning
- POD-DeepONet variant for efficiency
- Handles function-to-function mappings
- Suitable for parametric PDEs

### Fourier Neural Operators (FNO)

FNO learns operators in Fourier space, making it efficient for problems with periodic boundary conditions or translation-invariant operators.

**Key Features:**
- Spectral convolutions in Fourier space
- 1D, 2D, and 3D implementations
- Efficient for large-scale problems
- Resolution-invariant operators

### Neural ODEs

Neural ODEs parameterize continuous-time dynamics using neural networks, solving dy/dt = f_θ(y, t).

**Key Features:**
- Continuous-time modeling
- Conservative variants for physics
- Fluid dynamics specialization
- Flexible ODE solvers

## Utilities

The library includes comprehensive utilities:

### Mesh Generation
- Uniform and random point generation
- Boundary point sampling
- Circular and complex geometries
- Adaptive mesh refinement

### Data Preprocessing
- Data scaling and normalization
- Train/validation/test splitting
- Noise addition for robustness
- Collocation point generation

### Training Utilities
- Early stopping
- Loss tracking and visualization
- Adaptive loss weighting
- Specialized training loops for each method

## Contributing

Contributions are welcome! Please see our contributing guidelines for details.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Citation

If you use AI4CFD in your research, please cite:

```bibtex
@software{ai4cfd2025,
  title={AI4CFD: Various methods for solving PDEs using artificial intelligence},
  author={Huang, Wenjie},
  year={2025},
  url={https://github.com/Michael-Jackson666/AI4CFD}
}
```

## References

1. Raissi, M., Perdikaris, P., & Karniadakis, G. E. (2019). Physics-informed neural networks: A deep learning framework for solving forward and inverse problems involving nonlinear partial differential equations. Journal of Computational Physics, 378, 686-707.

2. Lu, L., Jin, P., Pang, G., Zhang, Z., & Karniadakis, G. E. (2021). Learning nonlinear operators via DeepONet based on the universal approximation theorem of operators. Nature Machine Intelligence, 3(3), 218-229.

3. Li, Z., Kovachki, N., Azizzadenesheli, K., Liu, B., Bhattacharya, K., Stuart, A., & Anandkumar, A. (2020). Fourier neural operator for parametric partial differential equations. arXiv preprint arXiv:2010.08895.

4. Chen, R. T., Rubanova, Y., Bettencourt, J., & Duvenaud, D. K. (2018). Neural ordinary differential equations. Advances in neural information processing systems, 31.
