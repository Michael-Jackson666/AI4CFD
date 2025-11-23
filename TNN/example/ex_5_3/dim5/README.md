# TNN Example 5.3: Neumann Boundary Condition Problem

## Problem Description

This example solves a PDE with **Neumann (natural) boundary conditions** - prescribing derivatives on the boundary rather than values.

**Domain**: $[0,1]^{5}$

**PDE**:
$$
-\Delta u + \pi^2 u = 2\pi^2 F(x) \quad \text{in } \Omega
$$

**Neumann Boundary Condition**:
$$
\frac{\partial u}{\partial n} = \frac{\partial F}{\partial n} \quad \text{on } \partial\Omega
$$

where $n$ is the outward normal vector.

**Exact Solution**:
$$
F(x) = \sum_{i=1}^{5} \sin(\pi x_i)
$$

**Gradient on Boundary**:
- At $x_i = 0$: $\frac{\partial F}{\partial n} = -\pi$
- At $x_i = 1$: $\frac{\partial F}{\partial n} = -\pi$

## Key Innovation: Weak Enforcement of Neumann BC

Unlike Dirichlet conditions (ex_5_1, ex_5_2) which can be enforced strongly, Neumann conditions are incorporated into the variational formulation.

### Variational Formulation

The loss function includes:
1. **Interior PDE residual**: $\|\Delta u - \pi^2 u + 2\pi^2 F\|^2_\Omega$
2. **Boundary gradient matching**: $\|\nabla u \cdot n - \nabla F \cdot n\|^2_{\partial\Omega}$

This is a **mixed formulation** combining:
- Domain integration for the PDE
- Boundary integration for the Neumann condition

## TNN Architecture

**Single TNN model** without forced boundary conditions:
- **Rank**: 100 (higher than previous examples)
- **Architecture**: [1, 100, 100, 100, 100]
- **No boundary forcing**: `bd=None` (allows free boundary values)
- **Activation**: sin(x)

The network learns both interior values and boundary derivatives simultaneously.

## Loss Function Components

### Part 1: Interior Terms
$$
\mathcal{L}_{\Omega} = 4\pi^2\|F\|^2 + \pi^2\|\phi\|^2 + \frac{1}{\pi^2}\|\Delta\phi\|^2 - 4\pi^2(F,\phi) + 4(F,\Delta\phi) - 2(\phi,\Delta\phi)
$$

### Part 2: Boundary Terms
$$
\mathcal{L}_{\partial\Omega} = \|\nabla u \cdot n\|^2_{\partial\Omega \cap \{x_i=0\}} - 2(\nabla u \cdot n, \nabla F \cdot n)_{\partial\Omega} + \|\nabla F \cdot n\|^2_{\partial\Omega}
$$

### Total Loss
$$
\text{Loss} = \mathcal{L}_{\Omega} + \mathcal{L}_{\partial\Omega}
$$

## Configuration

- **Dimension**: 5
- **Domain**: [0, 1]^5
- **TNN rank**: 100
- **Quadrature**: 16 points, 100 partitions per dimension
- **Total quadrature points**: 1600 per dimension
- **Activation**: TNN_Sin

## Training Strategy

**Two-stage optimization**:

### Stage 1: Adam
- Epochs: 50,000
- Learning rate: 0.003
- Print interval: 100

### Stage 2: L-BFGS
- Epochs: 10,000
- Learning rate: 1.0
- Print interval: 100

## Key Technical Details

### 1. Boundary Derivative Computation

The code explicitly computes normal derivatives at boundaries:
```python
# At x=0: ∇u·n = -∂u/∂x
grad_phi0 = -grad_phi0 / norm

# At x=1: ∇u·n = +∂u/∂x
grad_phi1 = grad_phi1 / norm
```

### 2. Manual Normalization

Unlike some examples with `normed=True`, this uses manual normalization:
```python
norm = torch.sqrt(torch.sum(w*phi**2, dim=2)).unsqueeze(dim=-1)
phi = phi / norm
grad_phi = grad_phi / norm
```

### 3. Boundary Integration

Uses `Int2TNN_amend_1d` to compute boundary integrals involving gradients:
```python
nabla_phi_cdot_n0 = Int2TNN_amend_1d(w, w, C, phi, C, phi, grad_phi0, grad_phi0)
```

### 4. Higher Rank Requirement

Neumann problems typically require higher representation capacity (rank=100) compared to Dirichlet problems because:
- Less constraint from boundaries
- More complex solution structure
- Gradient matching adds difficulty

## Advantages of This Approach

1. **Natural BC handling**: No artificial boundary enforcement needed
2. **Variational consistency**: Follows natural weak formulation
3. **Physical accuracy**: Respects flux conservation on boundaries
4. **Flexibility**: Can handle mixed BC by combining techniques

## Usage

```bash
python ex_5_3_dim5.py
```

### Output

Reports L² and H¹ errors normalized by the exact solution energy.

## Comparison with Other Examples

| Feature | ex_5_1 | ex_5_2 | **ex_5_3** | ex_5_4 |
|---------|--------|--------|-----------|--------|
| BC Type | Dirichlet | Dirichlet | **Neumann** | Dirichlet |
| BC Value | u=0 | u=F | **∂u/∂n=∂F/∂n** | u=0 |
| Enforcement | Strong | Strong | **Weak** | Strong |
| # Models | 1 | 2 | **1** | 1 |
| Rank | 50 | 20+20 | **100** | 50 |

## Mathematical Background

Neumann problems have unique solutions only up to a constant. The additional $\pi^2 u$ term in the PDE ensures uniqueness by converting it to a **shifted Helmholtz equation**:

$$
-\Delta u + \pi^2 u = f
$$

This is elliptic and has a unique solution given Neumann BC.

The weak formulation naturally incorporates boundary integrals:
$$
\int_\Omega \nabla u \cdot \nabla v + \pi^2 \int_\Omega u v = \int_\Omega f v + \int_{\partial\Omega} (\nabla u \cdot n) v
$$

---

**Last updated**: January 2025
