# Model Comparison — MLP vs Transformer (PINNs)

📊 **Overview**

This folder contains the results and configuration files for architecture comparison experiments that evaluate MLP and Transformer-based neural networks on the 1D Vlasov–Poisson PINN problem. The experiments were orchestrated by `compare_models.py` and each experiment's folder contains training logs, configuration dumps and result plots.

---

## 🚀 Purpose

- Compare convergence, stability, and qualitative solution accuracy between a standard MLP and two Transformer variants (lightweight & standard).
- Provide a reproducible experiment layout and saved hyperparameters so results can be inspected and re-run.

---

## 📁 Directory Structure

Each subfolder (`mlp_standard`, `transformer_light`, `transformer_standard`) contains:

- `training_config.json` — full run configuration (JSON, machine-readable) ✅
- `training_config.txt` — human-readable config summary ✅
- `training_log.txt` — per-epoch logging ✅
- `loss_history.png` — training loss vs iteration ✅
- `results_epoch_*.png` — snapshots of solution / diagnostics at recorded epochs ✅

---

## 🔎 Key Experimental Settings

| Experiment | Model | Description | lr | epochs | duration (s) | status |
|---|---:|---|---:|---:|---:|---:|
| `mlp_standard` | `mlp` | MLP Standard (8×128) | 1e-4 | 2000 | 41.65 | success |
| `transformer_light` | `lightweight_transformer` | Lightweight Transformer (d_model=128, nhead=4, layers=3) | 1e-4 | 2000 | 143.72 | success |
| `transformer_standard` | `transformer` | Compact Transformer (d_model=192, nhead=6, layers=4) | 8e-5 | 2000 | 261.54 | success |

> These values are read from each `training_config.json` in the corresponding experiment folder.

---

## ▶️ Reproducing the experiments

1. Ensure main PINN training utilities are available and dependencies are installed:

```bash
pip install torch numpy matplotlib
```

2. Run the comparison script and follow the prompt:

```bash
python compare_models.py
```

- Option `1` runs the full MLP vs Transformer comparison and writes outputs to `comparison/*`.
- Option `2` runs a quick test (200 epochs) for a single chosen model.

3. Inspect results (images, logs, and `training_config.json`) in the corresponding folder, or open TensorBoard if the run saved events.

---

## ✅ How to add a new model / experiment

- Copy one of the `training_config.json` files and modify architecture or hyperparameters.
- Or, edit `compare_models.py` to add a new configuration in `compare_mlp_vs_transformer()`.
- Run `python compare_models.py` or write a simple driver that calls `run_experiment(config_name, config, description)`.

---

## Tips & Notes

- Transformer-based models typically need smaller learning rates and longer training for stable results.
- Use `v_quad_points` to control quadrature resolution for computing electron density — increase to improve accuracy at the cost of runtime.
- Check `training_log.txt` and `loss_history.png` to diagnose stability or underfitting/overfitting.

---

## Contributing

Open an issue or PR to add more architectures, plot scripts, or automated summary tables for numerical metrics (e.g., L2 error or conservation violations).

---

License: see top-level `LICENSE`.
