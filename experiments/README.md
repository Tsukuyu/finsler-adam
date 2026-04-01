# Experiments

Reproducible benchmark experiments for Finsler-Adam.

## Quick Reproduce

All experiments can be reproduced without GPU (numpy-backed mock torch is used):

```bash
# Run synthetic benchmark suite (5 functions × 4 configs × 3 LRs × 3 seeds)
python experiments/scripts/run_benchmarks.py

# Generate publication figures
python experiments/scripts/plot_results.py

# Run hyperparameter robustness grid search (γ × α × LR)
python experiments/scripts/hp_robustness.py

# Generate compute cost comparison figures
python experiments/scripts/compute_cost_analysis.py
```

## Pre-computed Results

Results from our experiments are included in `results/` for immediate inspection:

- `summary.csv` — Final loss values for all configurations
- `full_results.json` — Complete step-by-step logs
- `hp_robustness.json` — γ×α grid search data (10×10×3 LRs×3 functions)

## Figures

All figures are in `../docs/figures/`:

| Figure | Description |
|--------|-------------|
| `fig1_convergence_curves.png` | Training loss trajectories (5 benchmarks) |
| `fig2_anna_clip_response.png` | Anna-Limit clipping response curve |
| `fig3_lr_sensitivity.png` | Learning rate sensitivity analysis |
| `fig4_mechanism.png` | Finsler-Adam architecture overview |
| `fig_hp_robustness.png` | HP robustness heatmaps (γ × α) |
| `fig_hp_sensitivity.png` | Marginal HP sensitivity analysis |
| `fig_compute_cost.png` | Theoretical compute cost comparison |
| `fig_optimizer_positioning.png` | Optimizer positioning scatter plot |
