# TNN Example 5.2: Non-homogeneous Boundary Value Problem

## Problem Description

This example solves a Poisson equation with **non-homogeneous boundary conditions** using a **two-TNN decomposition method**.

**Domain**: $[0,1]^{5}$

**PDE**:
$$
-\Delta u = \frac{\pi^2}{4} F(x) \quad \text{in } \Omega
$$

**Boundary Condition**:
$$
u = F(x) \quad \text{on } \partial\Omega
$$

**Source/Boundary Function**:
$$
F(x) = \sum_{i=1}^{5} \sin\left(\frac{\pi x_i}{2}\right)
$$

Note: In this problem, $F(x)$ is the exact solution.

## Key Innovation: Two-TNN Decomposition

Unlike ex_5_1 which handles homogeneous boundaries directly, this example uses **domain decomposition**:

$$
u(x) = u_0(x) + u_b(x)
$$

where:
- $u_b$: boundary correction function (enforces non-homogeneous BC)
- $u_0$: interior solution (satisfies modified PDE with homogeneous BC)

### Two Independent TNN Models

1. **`model_b`** (Boundary Model):
   - Rank: 20
   - Architecture: [1, 50, 50, 50, 20]
   - No forced boundary condition
   - Learns to match $F(x)$ on $\partial\Omega$

2. **`model_0`** (Interior Model):
   - Rank: 20
   - Architecture: [1, 50, 50, 50, 20]
   - Forced BC: $(x-a)(b-x)$ ensuring zero on boundary
   - Learns interior correction

## Two-Stage Training Strategy

### Stage 1: Train Boundary Model (`model_b`)

**Objective**: Minimize boundary matching error
$$
\text{Loss}_{\text{bd}} = \|u_b - F\|^2_{\partial\Omega}
$$

**Training**:
- Adam: 20,000 epochs, lr=0.003
- L-BFGS: 5,000 epochs, lr=0.1
- **Parameters frozen** after training

### Stage 2: Train Interior Model (`model_0`)

**Objective**: Minimize PDE residual
$$
\text{Loss} = \|\Delta u + \frac{\pi^2}{4}F\|^2_{\Omega}
$$

where $u = u_0 + u_b$ and $u_b$ is fixed.

**Training**:
- Adam: 20,000 epochs, lr=0.003
- L-BFGS: 5,000 epochs, lr=0.1
- Uses `retain_graph=True` for gradient computation

## Configuration

- **Dimension**: 5
- **Domain**: [0, 1]^5
- **Quadrature**: 16 points, 10 partitions per dimension
- **Total quadrature points**: 160 per dimension
- **Combined TNN rank**: 40 (20 + 20)

## Key Technical Details

### 1. Boundary Value Enforcement

Boundary values are sampled at domain boundaries:
```python
phi_bd0 = model_b(w, torch.zeros_like(x), ...)  # at x=0
phi_bd1 = model_b(w, torch.ones_like(x), ...)   # at x=1
```

### 2. Modified Right-Hand Side

The interior model solves:
$$
-\Delta u_0 = \frac{\pi^2}{4}F - \Delta u_b
$$

### 3. Gradient Retention

Uses `loss.backward(retain_graph=True)` because the computational graph involves fixed parameters from `model_b`.

### 4. Solution Reconstruction

Final solution combines both models:
```python
alpha = torch.cat((alpha_0, alpha_b), dim=0)
phi = torch.cat((phi_0, phi_b), dim=1)
```

## Advantages Over Single-TNN Approach

1. **Flexibility**: Can handle arbitrary boundary conditions
2. **Modularity**: Boundary and interior solved independently
3. **Accuracy**: Specialized models for different regions
4. **Reusability**: Same `model_b` for multiple interior problems

## Usage

```bash
python ex_5_2_dim5.py
```

### Output

The code prints:
- Stage 1: Boundary loss convergence
- Stage 2: Interior loss convergence
- Final L² and H¹ errors

## Comparison with Other Examples

| Example | Boundary Type | Method | Domain |
|---------|---------------|--------|--------|
| ex_5_1 | Homogeneous (u=0) | Single TNN | [-1,1]^5 |
| **ex_5_2** | **Non-homogeneous** | **Two TNNs** | **[0,1]^5** |
| ex_5_3 | Neumann BC | Single TNN | [0,1]^5 |
| ex_5_4 | Eigenvalue | Single TNN | [0,1]^5 |

## Mathematical Background

This approach is based on the **lifting method** for non-homogeneous BVPs:
1. Find $u_b$ satisfying boundary conditions
2. Solve for $u_0$ with homogeneous BC
3. Combine: $u = u_0 + u_b$

The variational formulation ensures weak enforcement of the PDE in the interior while strongly enforcing boundary conditions.

---

**Last updated**: January 2025
