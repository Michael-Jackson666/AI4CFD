# TNN 20D Eigenvalue Problem

## Problem Description

Same eigenvalue problem as 5D case but in **20 dimensions**.

**Domain**: $[0,1]^{20}$

**Eigenvalue Problem**: $-\Delta u = \lambda u$ in $\Omega$, $u = 0$ on $\partial\Omega$

**Exact Solution**:
- **Eigenvalue**: $\lambda = 20\pi^2 \approx 197.392$
- **Eigenfunction**: $u(x) = \prod_{i=1}^{20} \sin(\pi x_i)$

## Configuration

- **Dimension**: 20
- **TNN rank**: 50
- **Architecture**: [1, 100, 100, 100, 50]
- **Method**: Rayleigh quotient minimization

## Usage

```bash
python ex_5_4_dim20.py
```

## Notes

For detailed methodology on eigenvalue problems with TNN, see [5D example README](../dim5/README.md).

Demonstrates TNN's effectiveness for high-dimensional eigenvalue problems.

---

**Last updated**: January 2025
