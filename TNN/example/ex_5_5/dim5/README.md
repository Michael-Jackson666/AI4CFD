# TNN Example 5.5: Quantum Harmonic Oscillator (Unbounded Domain)

## Problem Description

This example solves the **quantum harmonic oscillator eigenvalue problem** on an **unbounded domain** $\mathbb{R}^d$ using Hermite-Gauss quadrature.

**Domain**: $\mathbb{R}^{5}$ (unbounded!)

**Schrödinger Eigenvalue Problem**:
$$
-\Delta u + V(x) u = \lambda u \quad \text{in } \mathbb{R}^5
$$

**Potential**:
$$
V(x) = \sum_{i=1}^{5} x_i^2
$$

**Natural Boundary Condition**:
$$
u(x) \to 0 \quad \text{as } |x| \to \infty
$$

**Exact Solution**:
- **Ground state eigenvalue**: $\lambda = d = 5$
- **Ground state eigenfunction**: $u(x) = e^{-\frac{1}{2}|x|^2}$ (Gaussian)

This is a fundamental problem in quantum mechanics!

## Key Innovation: Hermite-Gauss Quadrature

Unlike previous examples using Gauss-Legendre quadrature on bounded domains, this uses **Hermite-Gauss quadrature** for infinite domains:

### Weight Function

Integration with respect to weight $w(x) = e^{-x^2}$:
$$
\int_{\mathbb{R}} f(x) e^{-x^2} dx \approx \sum_{i=1}^{N} w_i f(z_i)
$$

### Modified Quadrature

Code uses `modified=False` to keep exponential weight:
```python
z, w = Hermite_Gauss_Quad(200, device=device, dtype=dtype, modified=False)
```

This means the weight $e^{-x^2}$ is included in quadrature weights $w_i$.

## TNN Architecture

**Single TNN model** without forced boundary conditions:
- **Rank**: 50
- **Architecture**: [1, 100, 100, 100, 50]
- **No forced BC**: Natural decay at infinity
- **Activation**: sin(x)

The exponential decay is enforced by the quadrature weight, not the network architecture!

## Two Loss Functions

### Loss 1: Rayleigh Quotient (Ritz Method)

Used in pre-training:
$$
\text{Loss}_{\text{Ritz}} = \frac{\int_{\mathbb{R}^d} |\nabla u|^2 + V|u|^2 \, dx}{\int_{\mathbb{R}^d} |u|^2 \, dx}
$$

This gives the eigenvalue directly.

### Loss 2: Residual Estimator (η-method)

Used in fine-tuning:
$$
\text{Loss}_{\eta} = \|\Delta u + Vu - \lambda u\|_{L^2(\mathbb{R}^d)}
$$

Expanded:
$$
= \sqrt{\lambda^2\|u\|^2 + \|\Delta u\|^2 + \|Vu\|^2 + 2\lambda(\Delta u, u) - 2(Vu, \Delta u) - 2\lambda(Vu, u)}
$$

## Modified Derivatives

Key technical detail: derivatives must account for weight function!

### Chain Rule with Weight

For $\phi(x) e^{-x^2/2}$:
- **First derivative**: $\phi'(x) e^{-x^2/2} = (\text{grad\_phi} - x \cdot \phi) e^{-x^2/2}$
- **Second derivative**: $\phi''(x) e^{-x^2/2} = (\text{grad\_grad\_phi} - 2x \cdot \text{grad\_phi} + (x^2-1)\phi) e^{-x^2/2}$

Code implementation:
```python
grad_phi = grad_phi - z * phi
grad_grad_phi = grad_grad_phi - 2*z*grad_phi + (z**2 - 1)*phi
```

## Configuration

- **Dimension**: 5
- **Domain**: $\mathbb{R}^5$ (unbounded)
- **TNN rank**: 50
- **Quadrature**: 200 Hermite-Gauss points per dimension
- **Total quadrature points**: 200 per dimension
- **Activation**: TNN_Sin

## Two-Stage Training Strategy

### Stage 1: Ritz Method with Adam
- **Loss**: Rayleigh quotient
- **Optimizer**: Adam
- **Epochs**: 10,000
- **Learning rate**: 0.01
- **Purpose**: Find good initial approximation

### Stage 2: η-method with L-BFGS
- **Loss**: PDE residual
- **Optimizer**: L-BFGS
- **Epochs**: 10,000
- **Learning rate**: 1.0
- **Purpose**: Refine eigenfunction and eigenvalue

## Key Technical Details

### 1. Generalized Eigenvalue Problem

Solves $A\mathbf{u} = \lambda M\mathbf{u}$ where:
```python
A = Int2TNN_amend_1d(w, w, alpha, phi, alpha, phi, grad_phi, grad_phi) 
    + Int3TNN(w, alpha_V, V, alpha, phi, alpha, phi)
M = Int2TNN(w, alpha, phi, alpha, phi)
```

### 2. Three-Body Integration

For potential term $V(x)u(x)u(x)$:
```python
Int3TNN(w, alpha_V, V, alpha, phi, alpha, phi)
```

### 3. Four-Body Integration

For $V(x)u(x)V(x)u(x)$:
```python
Int4TNN(w, alpha_V, V, u, phi, alpha_V, V, u, phi)
```

### 4. Exact Eigenfunction

Ground state is pure Gaussian (rank-1):
```python
F = torch.ones((dim, 1, N), device=device, dtype=dtype)  # Constant in z-space
alpha_F = torch.ones(1, device=device, dtype=dtype)
```

Because $e^{-|x|^2/2} = \prod_{i=1}^d e^{-x_i^2/2}$ is separable!

## Mathematical Background

### Quantum Harmonic Oscillator

The Hamiltonian is:
$$
\hat{H} = -\frac{\hbar^2}{2m}\Delta + \frac{1}{2}m\omega^2|x|^2
$$

With appropriate scaling ($\hbar = m = \omega = 1$):
$$
\hat{H} = -\Delta + |x|^2
$$

### Spectrum

Full spectrum (in 1D):
$$
\lambda_n = 2n + 1, \quad n = 0, 1, 2, \ldots
$$

In $d$ dimensions:
$$
\lambda_{n_1,\ldots,n_d} = 2(n_1 + \cdots + n_d) + d
$$

Ground state: $n_1 = \cdots = n_d = 0 \Rightarrow \lambda = d$

### Hermite Functions

Exact eigenfunctions are products of Hermite functions:
$$
u_{n_1,\ldots,n_d}(x) = \prod_{i=1}^d H_{n_i}(x_i) e^{-x_i^2/2}
$$

where $H_n$ are Hermite polynomials.

## Advantages of TNN for Unbounded Problems

1. **Natural framework**: Separable structure matches tensor decomposition
2. **Efficient quadrature**: Hermite-Gauss quadrature is optimal for Gaussians
3. **No truncation**: No need to truncate domain to $[-L, L]$
4. **Exact for low rank**: Ground state has rank 1, perfectly suited for TNN

## Usage

```bash
python ex_5_5_dim5.py
```

### Output Example

```
epoch = 0
loss = 1.234e-02
errorE = 2.345e-05    # Eigenvalue error
error0 = 3.456e-06    # L² error (projection)
error1 = 4.567e-06    # H¹ error (projection)
```

## Comparison with Other Examples

| Example | Domain | BC Type | Problem | Quadrature |
|---------|--------|---------|---------|------------|
| ex_5_1 | Bounded | Dirichlet | PDE | Gauss-Legendre |
| ex_5_2 | Bounded | Non-homogeneous | PDE | Gauss-Legendre |
| ex_5_3 | Bounded | Neumann | PDE | Gauss-Legendre |
| ex_5_4 | Bounded | Dirichlet | Eigenvalue | Gauss-Legendre |
| **ex_5_5** | **Unbounded** | **Natural** | **Eigenvalue** | **Hermite-Gauss** |

## Extensions

### Higher Energy States

Can find excited states by:
1. Adding orthogonality constraints to lower states
2. Deflation techniques
3. Minimizing Rayleigh quotient in orthogonal subspace

### Anharmonic Oscillators

Replace $V(x) = |x|^2$ with:
- $V(x) = |x|^4$ (quartic)
- $V(x) = |x|^2 + \epsilon|x|^4$ (anharmonic)

### Coupled Oscillators

Different frequencies per dimension:
$$
V(x) = \sum_{i=1}^d \omega_i^2 x_i^2
$$

---

**Last updated**: January 2025
