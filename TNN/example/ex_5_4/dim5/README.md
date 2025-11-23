# TNN Example 5.4: Eigenvalue Problem

## Problem Description

This example solves a **Laplacian eigenvalue problem** using TNN to find the smallest eigenvalue and corresponding eigenfunction.

**Domain**: $[0,1]^{5}$

**Eigenvalue Problem**:
$$
-\Delta u = \lambda u \quad \text{in } \Omega
$$

**Boundary Condition**:
$$
u = 0 \quad \text{on } \partial\Omega
$$

**Exact Solution**:
- **Eigenvalue**: $\lambda = d\pi^2 = 5\pi^2 \approx 49.348$
- **Eigenfunction**: $u(x) = \prod_{i=1}^{5} \sin(\pi x_i)$

This is the **ground state** (fundamental mode) of the domain.

## Key Innovation: Rayleigh Quotient Minimization

Instead of solving the PDE directly, this approach minimizes the **Rayleigh quotient**:

$$
\lambda = \min_{u \in H^1_0(\Omega)} \frac{\int_\Omega |\nabla u|^2 \, dx}{\int_\Omega u^2 \, dx}
$$

The minimizer gives both the smallest eigenvalue and the corresponding eigenfunction.

## TNN Architecture

**Single TNN model** with homogeneous Dirichlet BC:
- **Rank**: 50
- **Architecture**: [1, 100, 100, 100, 50]
- **Forced BC**: $(x-a)(b-x)$ ensures $u=0$ on $\partial\Omega$
- **Activation**: sin(x)

## Computational Method: Generalized Eigenvalue Problem

The code solves a **generalized eigenvalue problem** in TNN coefficient space:

### Step 1: Assemble Matrices

**Stiffness matrix** (gradient):
$$
A_{ij} = \int_\Omega \nabla \phi_i \cdot \nabla \phi_j \, dx
$$

**Mass matrix**:
$$
M_{ij} = \int_\Omega \phi_i \phi_j \, dx
$$

### Step 2: Solve Generalized Eigenvalue Problem

$$
A \mathbf{u} = \lambda M \mathbf{u}
$$

Using Cholesky decomposition to convert to standard eigenvalue problem:
```python
L = torch.linalg.cholesky(M)
C = torch.linalg.solve(L, A.t()).t()
D = torch.linalg.solve(L, C)
E, V = torch.linalg.eigh(D)  # D = L^{-1} A L^{-T}
```

### Step 3: Extract Smallest Eigenvalue

```python
ind = torch.argmin(E)
lam = E[ind]  # smallest eigenvalue
u = torch.linalg.solve(L.t(), V[:, ind])  # corresponding coefficients
```

## Loss Function: Posterior Error Estimator

The loss is the **PDE residual norm**:

$$
\text{Loss} = \|\Delta u + \lambda u\|_{L^2(\Omega)} = \sqrt{\int_\Omega (\Delta u + \lambda u)^2 \, dx}
$$

Expanded form:
$$
\text{Loss} = \sqrt{\|\Delta u\|^2 + \lambda^2\|u\|^2 + 2\lambda(\Delta u, u)}
$$

This is a **posterior error estimator** that refines the eigenfunction approximation.

## Configuration

- **Dimension**: 5
- **Domain**: [0, 1]^5
- **TNN rank**: 50
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

### 1. Laplacian Operator

Computed using second derivatives and tensor manipulation:
```python
phi_expand = phi.expand(dim, -1, -1, -1).clone()
phi_expand[torch.arange(dim), torch.arange(dim), :, :] = grad_grad_phi
Delta_phi = phi_expand.transpose(0,1).flatten(1,2)
```

### 2. Three Error Metrics

The code reports:
- **Eigenvalue error**: $\frac{|\lambda - \lambda_{\text{exact}}|}{\lambda_{\text{exact}}}$
- **L² error**: $\|u - u_{\text{exact}}\|_{L^2} / \|u_{\text{exact}}\|_{L^2}$
- **H¹ error**: $\|u - u_{\text{exact}}\|_{H^1} / \|u_{\text{exact}}\|_{H^1}$

### 3. Projection-Based Error

Uses `projection=True` in error estimation:
```python
error0 = error0_estimate(w, alpha_F, F, u, phi, projection=True)
```

This computes the error in the best approximation sense rather than point-wise difference.

### 4. Alternative Activations Supported

Code includes commented options:
- `TNN_Tanh`
- `TNN_ReQU` (rectified quadratic unit)
- `TNN_Sigmoid`

## Mathematical Background

### Rayleigh Quotient Properties

1. **Variational principle**: The minimum of the Rayleigh quotient equals the smallest eigenvalue
2. **Orthogonality**: Higher eigenfunctions are orthogonal to lower ones
3. **Monotonicity**: Constraining the space increases the quotient

### Why This Works with TNN

TNN provides a **rich function space** for eigenfunction approximation:
- Tensor decomposition captures separable structure
- Forced BC ensures $u \in H^1_0(\Omega)$
- High rank (50) allows accurate representation

### Comparison to Traditional Methods

| Method | Complexity | Curse of Dimensionality | Accuracy |
|--------|-----------|------------------------|----------|
| Finite Difference | $O(N^d)$ | Severe | Good |
| Finite Element | $O(N^d)$ | Severe | Very Good |
| **TNN** | **$O(dNp)$** | **Mild** | **Excellent** |

## Usage

```bash
python ex_5_4_dim5.py
```

### Output Example

```
epoch = 0
loss = 1.234e-02
errorE = 2.345e-03
error0 = 3.456e-04
error1 = 4.567e-04
```

## Comparison with Other Examples

| Example | Problem Type | BC Type | Solution Type |
|---------|-------------|---------|---------------|
| ex_5_1 | PDE | Dirichlet | Direct |
| ex_5_2 | PDE | Non-homogeneous | Decomposition |
| ex_5_3 | PDE | Neumann | Weak form |
| **ex_5_4** | **Eigenvalue** | **Dirichlet** | **Rayleigh quotient** |
| ex_5_5 | Eigenvalue | Natural | Hermite space |

## Extensions

### Higher Eigenmodes

To find higher eigenvalues, add orthogonality constraints:
$$
\int_\Omega u_k u_j \, dx = 0 \quad \forall j < k
$$

### Different Operators

Can solve eigenproblems for:
- Biharmonic: $\Delta^2 u = \lambda u$
- Schrödinger: $-\Delta u + V(x)u = \lambda u$
- Stokes: $-\Delta \mathbf{u} + \nabla p = \lambda \mathbf{u}$

---

**Last updated**: January 2025
