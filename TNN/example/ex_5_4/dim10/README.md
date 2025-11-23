# TNN 10D Eigenvalue Problem

## Problem Description

Same eigenvalue problem as 5D case but in **10 dimensions**.

**Domain**: $[0,1]^{10}$

**Eigenvalue Problem**: $-\Delta u = \lambda u$ in $\Omega$, $u = 0$ on $\partial\Omega$

**Exact Solution**:
- **Eigenvalue**: $\lambda = 10\pi^2 \approx 98.696$
- **Eigenfunction**: $u(x) = \prod_{i=1}^{10} \sin(\pi x_i)$

## Configuration

- **Dimension**: 10
- **TNN rank**: 50
- **Architecture**: [1, 100, 100, 100, 50]
- **Method**: Rayleigh quotient minimization

## Usage

```bash
python ex_5_4_dim10.py
```

## Notes

For detailed methodology on eigenvalue problems with TNN, see [5D example README](../dim5/README.md).

---

**Last updated**: January 2025
