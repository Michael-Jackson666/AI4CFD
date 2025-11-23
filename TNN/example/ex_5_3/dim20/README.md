# TNN 20D Neumann Boundary Condition Problem

## Problem Description

Same Neumann BC problem as 5D case but in **20 dimensions**.

**Domain**: $[0,1]^{20}$

**PDE**: $-\Delta u + \pi^2 u = 2\pi^2 F(x)$

**Neumann BC**: $\frac{\partial u}{\partial n} = \frac{\partial F}{\partial n}$ on $\partial\Omega$

**Exact Solution**: $F(x) = \sum_{i=1}^{20} \sin(\pi x_i)$

## Configuration

- **Dimension**: 20
- **TNN rank**: 100
- **Architecture**: [1, 100, 100, 100, 100]
- **Quadrature**: 16 points, 100 partitions

## Usage

```bash
python ex_5_3_dim20.py
```

## Notes

For detailed methodology on Neumann boundary conditions with TNN, see [5D example README](../dim5/README.md).

Demonstrates that the Neumann formulation scales to very high dimensions.

---

**Last updated**: January 2025
