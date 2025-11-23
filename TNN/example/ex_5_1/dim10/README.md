# TNN 10D PDE Solver Example

## Problem Description

This example solves the same PDE problem as the 5D case, but in **10 dimensions**.

**Domain**: $[-1,1]^{10}$

**Objective**: Minimize the energy functional
$$
E(u) = \frac{1}{2}\int_{[-1,1]^{10}} |\nabla u|^2 dx - \int_{[-1,1]^{10}} f \cdot u \, dx
$$

**Source term**:
$$
f(x_1, \ldots, x_{10}) = \sum_{k=1}^{10} \sin(2\pi x_k) \prod_{i \neq k} \sin(\pi x_i)
$$

**Boundary condition**: $u = 0$ on $\partial[-1,1]^{10}$ (enforced by $\prod_{k=1}^{10}(x_k+1)(1-x_k)$)

## Configuration

- **Dimension**: 10
- **TNN rank**: 50
- **Network architecture**: [1, 100, 100, 100, 50] per dimension
- **Quadrature**: 16 points, 200 partitions per dimension
- **Total quadrature points**: 3200 per dimension

## Usage

```bash
python ex_5_1_dim10.py
```

## Notes

For detailed documentation on the TNN method, architecture, and implementation details, please refer to the [5D example README](../dim5/README.md).

The only difference from the 5D case is the problem dimension. All other components (TNN architecture, training strategy, loss function) remain the same.

---

**Last updated**: January 2025
