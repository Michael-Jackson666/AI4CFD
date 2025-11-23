# TNN 10D Non-homogeneous Boundary Value Problem

## Problem Description

Same as the 5D case but in **10 dimensions**.

**Domain**: $[0,1]^{10}$

**PDE**: $-\Delta u = \frac{\pi^2}{4} F(x)$ in $\Omega$

**Boundary**: $u = F(x)$ on $\partial\Omega$

**Source**: $F(x) = \sum_{i=1}^{10} \sin\left(\frac{\pi x_i}{2}\right)$

## Configuration

- **Dimension**: 10
- **Two TNN models**: `model_b` (boundary) + `model_0` (interior)
- **Rank**: 20 each
- **Architecture**: [1, 50, 50, 50, 20] per dimension

## Usage

```bash
python ex_5_2_dim10.py
```

## Notes

For detailed methodology and mathematical formulation, see [5D example README](../dim5/README.md).

The only difference is the problem dimension.

---

**Last updated**: January 2025
