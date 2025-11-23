# TNN 10D Neumann Boundary Condition Problem

## Problem Description

Same Neumann BC problem as 5D case but in **10 dimensions**.

**Domain**: $[0,1]^{10}$

**PDE**: $-\Delta u + \pi^2 u = 2\pi^2 F(x)$

**Neumann BC**: $\frac{\partial u}{\partial n} = \frac{\partial F}{\partial n}$ on $\partial\Omega$

**Exact Solution**: $F(x) = \sum_{i=1}^{10} \sin(\pi x_i)$

## Configuration

- **Dimension**: 10
- **TNN rank**: 100
- **Architecture**: [1, 100, 100, 100, 100]
- **Quadrature**: 16 points, 100 partitions

## Usage

```bash
python ex_5_3_dim10.py
```

## Notes

For detailed methodology on Neumann boundary conditions with TNN, see [5D example README](../dim5/README.md).

---

**Last updated**: January 2025
