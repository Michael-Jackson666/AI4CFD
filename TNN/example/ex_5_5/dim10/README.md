# TNN 10D Quantum Harmonic Oscillator

## Problem Description

Same quantum harmonic oscillator problem as 5D case but in **10 dimensions**.

**Domain**: $\mathbb{R}^{10}$ (unbounded)

**Schrödinger Problem**: $-\Delta u + V(x)u = \lambda u$ where $V(x) = \sum_{i=1}^{10} x_i^2$

**Exact Solution**:
- **Ground state eigenvalue**: $\lambda = 10$
- **Ground state eigenfunction**: $u(x) = e^{-\frac{1}{2}\sum_{i=1}^{10} x_i^2}$

## Configuration

- **Dimension**: 10
- **Domain**: $\mathbb{R}^{10}$
- **TNN rank**: 50
- **Quadrature**: 200 Hermite-Gauss points per dimension

## Usage

```bash
python ex_5_5_dim10.py
```

## Notes

For detailed methodology on unbounded domain eigenvalue problems, see [5D example README](../dim5/README.md).

---

**Last updated**: January 2025
