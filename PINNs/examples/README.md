# PINNs Examples

This folder contains example scripts and data for the PINNs (Physics-Informed Neural Networks) module.

## Included files

- `possion_dirichlet_1d.py` : A simple 1D Poisson equation example with Dirichlet boundary conditions.
- `train.dat`, `test.dat`, `loss.dat` : Example data files produced by training runs (or used as sample input/output for plotting).

## Problem: 1D Poisson (Dirichlet)

Solve the 1D Poisson equation on the domain $\Omega = [-1,1]$:

$$
-\frac{d^2 u}{dx^2} = f(x), \qquad x \in [-1,1],
$$

with homogeneous Dirichlet boundary conditions:

$$
u(-1) = u(1) = 0.
$$

In the example `possion_dirichlet_1d.py`, the source term $f(x)$ and exact solution $u(x)$ are chosen so that an analytic solution is available for verification.

## How to run

1. Create a Python environment and install dependencies (suggested):

```bash
pip install torch numpy matplotlib scipy
```

2. Run the example:

```bash
cd PINNs/examples
python possion_dirichlet_1d.py
```

The script will train a PINN (or run a preset experiment), save training logs to `loss.dat`, and optionally write `train.dat` / `test.dat` containing predicted vs exact values for visualization.

## What to expect

- Terminal output: training progress and final loss
- Generated plots: comparison between PINN prediction and exact solution (if plotting enabled)
- Data files: `train.dat`, `test.dat`—can be loaded for plotting or further analysis

## Notes & Tips

- The example is minimal and intended for learning and quick tests. To adapt for other PDEs:
  - Modify `f(x)` and (optionally) the analytic solution for verification
  - Adjust network architecture and optimizer settings in the script
- For reproducible experiments, set random seeds and record hyperparameters (learning rate, epochs, network size)

## References

- Raissi, Perdikaris & Karniadakis, "Physics-informed neural networks: A deep learning framework for solving forward and inverse problems involving nonlinear partial differential equations" (JCP, 2019)

---

If you want, I can also add a small README in Chinese or add example plots and a script to reproduce figures automatically; tell me which you'd prefer.