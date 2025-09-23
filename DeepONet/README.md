# Deep Operator Networks (DeepONet)

Deep Operator Networks (DeepONet) are neural networks designed to learn operators that map between infinite-dimensional function spaces. Unlike traditional neural networks that map finite-dimensional vectors to vectors, DeepONet can learn mappings from functions to functions, making them particularly powerful for solving families of PDEs and operator learning problems.

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

## 📁 Files in this Directory

- `tutorial.ipynb` - Interactive Jupyter notebook tutorial
- `train.py` - Complete training script for various operator learning problems
- `models.py` - DeepONet architectures and variants
- `operators.py` - Common operator definitions and examples

## 🚀 Quick Start

### Running the Tutorial
```bash
jupyter notebook tutorial.ipynb
```

### Training a Model
```bash
python train.py --operator heat --epochs 5000 --lr 0.001
```

### Available Operators
- `poisson` - Poisson equation solver operator
- `heat` - Heat equation solution operator
- `burgers` - Burgers' equation operator
- `darcy` - Darcy flow operator
- `advection` - Advection equation operator

## 📊 Example Applications

### 1. Heat Equation Operator
Learn the operator that maps initial temperature distributions to temperature at any time:
```
G: u₀(x) → u(x,t)
```

### 2. Darcy Flow Operator
Learn the operator that maps permeability fields to pressure/velocity fields:
```
G: κ(x,y) → p(x,y)
```

### 3. Antiderivative Operator
Learn the operator that maps functions to their antiderivatives:
```
G: f(x) → ∫f(x)dx
```

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

## 🔗 Applications in CFD

### Fluid Flow Operators
- **Navier-Stokes**: Map boundary conditions to velocity/pressure fields
- **Turbulence Modeling**: Learn subgrid-scale models
- **Shape Optimization**: Map geometries to flow characteristics

### Heat Transfer
- **Conduction**: Map thermal properties to temperature fields
- **Convection**: Learn heat transfer coefficients
- **Radiation**: Model complex radiative transfer

### Multiphase Flows
- **Interface Tracking**: Learn interface evolution operators
- **Phase Change**: Model melting/solidification operators
- **Porous Media**: Learn effective property operators

## 📖 References

1. Lu, L., Jin, P., Pang, G., Zhang, Z., & Karniadakis, G. E. (2021). Learning nonlinear operators via DeepONet based on the universal approximation theorem of operators. Nature Machine Intelligence, 3(3), 218-229.

2. Wang, S., Wang, H., & Perdikaris, P. (2021). Learning the solution operator of parametric partial differential equations with physics-informed DeepONets. Science Advances, 7(40), eabi8605.

3. Lin, C., Li, Z., Lu, L., Cai, S., Maxey, M., & Karniadakis, G. E. (2021). Operator learning for predicting multiscale bubble growth dynamics. The Journal of Chemical Physics, 154(10), 104118.