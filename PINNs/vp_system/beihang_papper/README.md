# beihang_papper

📄 **Overview**

This folder contains a set of Physics-Informed Neural Network (PINN) implementations and experiments for the 1D Vlasov–Poisson system used to study the two-stream instability. The codebase contains multiple variants (original, normalized, conservation-aware, quick tests) and example result directories (e.g., `local_1000`, `remote_10000`, `remote_100000`, `reomote_ultra`).

---

## 🔬 Physics & Mathematical Background

We solve the 1D Vlasov–Poisson system for the electron distribution function $f(t,x,v)$ and self-consistent electric field $E(t,x)$:

- Vlasov equation (collisionless):

  $\displaystyle \frac{\partial f}{\partial t} + v\frac{\partial f}{\partial x} + E\frac{\partial f}{\partial v} = 0.$

- Poisson equation (1D, periodic domain):

  $\displaystyle \frac{\partial E}{\partial x} = n_e - 1, \quad n_e(t,x) = \int_{-V_{\max}}^{V_{\max}} f(t,x,v)\,\mathrm{d}v,$

  where the normalization has background ion density = 1.

Typical initial condition (two-stream) used in experiments:

$\displaystyle f(0,x,v)=\frac{1}{2}\left(N(v- v_b)+N(v + v_b)\right)\left(1 + A\cos(kx)\right),$

where $N(v)$ is a Gaussian with thermal width $v_{\text{th}}$, $v_b$ is beam velocity, $A$ is a perturbation amplitude, and $k = 2\pi/L_x$.

---

## 🧠 PINN Loss Components

The training uses a composite loss such as:

$\displaystyle L = \lambda_{\text{vlasov}}\lVert R_{\text{vlasov}}\rVert_2^2 + \lambda_{\text{poisson}}\lVert R_{\text{poisson}}\rVert_2^2 + \lambda_{\text{IC}}\lVert f - f_0\rVert_2^2 + \lambda_{\text{sym}}\lVert f(t,x,v)-f(t,x,-v)\rVert_2^2 + \lambda_{\text{cons}}(N_{\text{pred}}-N_{\text{true}})^2,$

where terms correspond to Vlasov residuals, Poisson residuals (on a grid), initial-condition mismatch, velocity symmetry, and total particle-number conservation (optional).

Network outputs enforce $f\ge 0$ by applying a Softplus activation at the network output.

---

## 📁 Files & Structure

- `main.py` — Baseline PINN implementation and training pipeline.
- `original.py` — Original implementation with classic 3-component loss (PDE, IC, BC).
- `normalized.py` / `original_normalized.py` — Versions that apply domain normalization for improved training stability.
- `conservation.py` — Adds an explicit conservation loss term to ensure total particle number is preserved.
- `quick_test.py` — Short-run configuration for fast validation or debugging.
- `quick_test/`, `local_1000/`, `remote_10000/`, `remote_100000/`, `reomote_ultra/`, `test_normalized/` — Output directories containing `results_epoch_*.png`, `loss_history.png`, TensorBoard events, and `training_log.txt`.

---

## ⚙️ Typical Config Keys (examples observed in scripts)

- Domain & discretization: `T_MAX`, `X_MAX`, `V_MAX`, `V_QUAD_POINTS`
- Model: `NN_LAYERS`, `NN_NEURONS`
- Optimizer: `LEARNING_RATE`, scheduler params
- Sampling / training: `EPOCHS`, `N_PHY`, `N_IC`
- Loss weights: `LAMBDA_VLASOV`, `LAMBDA_POISSON`, `LAMBDA_IC`, `LAMBDA_SYMM`, `LAMBDA_CONSERVATION`
- Logging & output: `PLOT_DIR`, TensorBoard logging

---

## ▶️ How to run

1. Install dependencies (example):

```bash
pip install torch numpy matplotlib tensorboard
```

2. Quick test (short run):

```bash
python quick_test.py
```

3. Full training / experiments:

```bash
python main.py          # or python normalized.py / original.py / conservation.py
```

4. Monitor training:

```bash
tensorboard --logdir=./local_1000   # or other output folder
```

5. Results & logs are saved in the output folders (`*_1000`, `*_10000`, `*_100000`, ...). Check `training_log.txt`, `loss_history.png`, and `results_epoch_*.png`.

---

## 🔍 Reproducing experiments

- Use the config section near the top of each script (or the script's default dict) to set domain parameters and hyperparameters.
- For longer experiments use the `remote_*` folders as examples of result layout.
- If you want to reproduce exact runs, inspect `training_log.txt` and TensorBoard event files in the corresponding folder for hyperparameters and run-time metrics.

---

## 📝 Notes & Tips

- The code uses numerical integration (trapezoidal rule) over velocity for computing electron density $n_e$; `V_QUAD_POINTS` controls quadrature resolution.
- Enforce `f \ge 0` through a Softplus at network output; this is essential for physical plausibility.
- For stability, try domain normalization (`normalized.py`) and lower learning rates or gradient clipping if training becomes unstable.

---

## 📚 Citation / Origin

This folder is named after a Beihang (BUAA) paper/experiment that inspired these implementations; please cite the original paper if you use these experiments in publications (citation details to be added here).

---

## 🤝 Contributing

Feel free to open issues or PRs for improvements (e.g., adding CLI args, saving full run-config YAMLs, adding unit tests for residuals, or adding reproducible experiment scripts).

---

License: see top-level `LICENSE` file in the repository.
