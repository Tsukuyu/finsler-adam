#!/usr/bin/env python3
"""
Hyperparameter Robustness Analysis: γ × α × LR grid search on synthetic benchmarks.
Generates publication-quality heatmaps showing final loss across HP grid.
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import json
import os
import sys

# Add project root for mock torch
sys.path.insert(0, '/sessions/dreamy-funny-bell/zeta-harness')
import _torch_mock
import torch

RESULTS_DIR = '/sessions/dreamy-funny-bell/zeta-harness/experiments/results'


# ──────────── Benchmark Functions ────────────

def rosenbrock(x):
    return sum(100 * (x[i+1] - x[i]**2)**2 + (1 - x[i])**2 for i in range(len(x)-1))

def sphere(x):
    return sum(xi**2 for xi in x)

def steep_valley(x):
    return x[0]**2 + 10*x[1]**2 + 5*abs(x[0]*x[1])


# ──────────── Finsler-Adam (numpy-based) ────────────

def finsler_adam_optimize(func, dim, lr, gamma, alpha, steps=500,
                           beta1=0.9, beta2=0.999, eps=1e-8, wd=0.01):
    """Run Finsler-Adam on a function and return final loss."""
    np.random.seed(42)
    x = np.random.randn(dim) * 0.5
    m = np.zeros(dim)
    v = np.zeros(dim)

    for t in range(1, steps + 1):
        # Numerical gradient
        grad = np.zeros(dim)
        h = 1e-5
        f0 = func(x)
        for i in range(dim):
            x_plus = x.copy()
            x_plus[i] += h
            grad[i] = (func(x_plus) - f0) / h

        # Anna-Limit clipping
        if alpha > 0:
            abs_g = np.abs(grad)
            grad = grad / (1.0 + alpha * np.power(abs_g, 4.0/3.0))

        # Moment updates
        m = beta1 * m + (1 - beta1) * grad
        v = beta2 * v + (1 - beta2) * grad**2

        # Bias correction
        m_hat = m / (1 - beta1**t)
        v_hat = v / (1 - beta2**t)

        # Weight decay
        x *= (1 - lr * wd)

        # Finsler scaling
        if gamma > 0:
            finsler_scale = 1.0 + gamma * np.sign(v * m)
            x -= lr * finsler_scale * m_hat / (np.sqrt(v_hat) + eps)
        else:
            x -= lr * m_hat / (np.sqrt(v_hat) + eps)

    return func(x)


# ──────────── Grid Search ────────────

def run_grid_search():
    """Run γ×α grid for each LR and each function."""

    gammas = np.linspace(0.0, 0.9, 10)
    alphas = np.array([0.0, 0.01, 0.03, 0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0])
    lrs = [0.005, 0.01, 0.05]

    functions = {
        'Rosenbrock-2D': (rosenbrock, 2),
        'Sphere-5D': (sphere, 5),
        'SteepValley-2D': (steep_valley, 2),
    }

    all_results = {}

    for func_name, (func, dim) in functions.items():
        print(f"\n=== {func_name} ===")
        for lr in lrs:
            key = f"{func_name}_lr{lr}"
            grid = np.zeros((len(gammas), len(alphas)))
            for i, g in enumerate(gammas):
                for j, a in enumerate(alphas):
                    loss = finsler_adam_optimize(func, dim, lr, g, a, steps=500)
                    grid[i, j] = loss
                    if np.isnan(loss) or np.isinf(loss):
                        grid[i, j] = 1e6
                print(f"  lr={lr}, gamma={g:.1f}: {grid[i].min():.4f} - {grid[i].max():.4f}")
            all_results[key] = grid.tolist()

    return all_results, gammas, alphas, lrs, list(functions.keys())


def plot_heatmaps(all_results, gammas, alphas, lrs, func_names):
    """Generate publication-quality heatmap figures."""

    # Figure 1: 3×3 grid (3 functions × 3 LRs)
    fig, axes = plt.subplots(3, 3, figsize=(16, 14))
    fig.suptitle('Hyperparameter Robustness: Final Loss across γ × α Grid',
                 fontsize=16, fontweight='bold', y=0.98)

    for col, lr in enumerate(lrs):
        for row, func_name in enumerate(func_names):
            ax = axes[row, col]
            key = f"{func_name}_lr{lr}"
            grid = np.array(all_results[key])

            # Use log scale for better visualization
            grid_clipped = np.clip(grid, 1e-6, None)
            vmin = max(grid_clipped.min(), 1e-6)
            vmax = grid_clipped.max()

            if vmax / vmin > 100:
                im = ax.imshow(grid_clipped, aspect='auto', origin='lower',
                              norm=LogNorm(vmin=vmin, vmax=vmax),
                              cmap='RdYlGn_r')
            else:
                im = ax.imshow(grid_clipped, aspect='auto', origin='lower',
                              cmap='RdYlGn_r')

            ax.set_xticks(range(len(alphas)))
            ax.set_xticklabels([f'{a:.2f}' for a in alphas], fontsize=7, rotation=45)
            ax.set_yticks(range(len(gammas)))
            ax.set_yticklabels([f'{g:.1f}' for g in gammas], fontsize=7)

            if row == 2:
                ax.set_xlabel('α (Anna-Limit)', fontsize=10)
            if col == 0:
                ax.set_ylabel(f'{func_name}\nγ (Finsler)', fontsize=10)
            ax.set_title(f'LR = {lr}', fontsize=11, fontweight='bold')

            plt.colorbar(im, ax=ax, shrink=0.8)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    path1 = os.path.join(RESULTS_DIR, 'fig_hp_robustness.png')
    fig.savefig(path1, dpi=150, bbox_inches='tight')
    print(f"\nSaved: {path1}")
    plt.close(fig)

    # Figure 2: Marginal sensitivity (γ averaged over α, and vice versa)
    fig2, axes2 = plt.subplots(2, 3, figsize=(16, 8))
    fig2.suptitle('Marginal Hyperparameter Sensitivity', fontsize=14, fontweight='bold')

    colors_lr = {0.005: '#1f77b4', 0.01: '#ff7f0e', 0.05: '#d62728'}

    for col, func_name in enumerate(func_names):
        ax_g = axes2[0, col]
        ax_a = axes2[1, col]

        for lr in lrs:
            key = f"{func_name}_lr{lr}"
            grid = np.array(all_results[key])

            # Mean over α for each γ
            mean_over_alpha = grid.mean(axis=1)
            ax_g.plot(gammas, mean_over_alpha, 'o-', color=colors_lr[lr],
                     label=f'LR={lr}', linewidth=2, markersize=4)

            # Mean over γ for each α
            mean_over_gamma = grid.mean(axis=0)
            ax_a.plot(alphas, mean_over_gamma, 's-', color=colors_lr[lr],
                     label=f'LR={lr}', linewidth=2, markersize=4)

        ax_g.set_xlabel('γ (Finsler strength)')
        ax_g.set_ylabel('Mean final loss')
        ax_g.set_title(func_name, fontweight='bold')
        ax_g.legend(fontsize=8)
        ax_g.grid(True, alpha=0.3)

        ax_a.set_xlabel('α (Anna-Limit strength)')
        ax_a.set_ylabel('Mean final loss')
        ax_a.legend(fontsize=8)
        ax_a.grid(True, alpha=0.3)

    axes2[0, 0].set_ylabel('γ sensitivity\n(avg over α)\nMean final loss')
    axes2[1, 0].set_ylabel('α sensitivity\n(avg over γ)\nMean final loss')

    plt.tight_layout()
    path2 = os.path.join(RESULTS_DIR, 'fig_hp_sensitivity.png')
    fig2.savefig(path2, dpi=150, bbox_inches='tight')
    print(f"Saved: {path2}")
    plt.close(fig2)


if __name__ == '__main__':
    print("Running HP Robustness Grid Search...")
    all_results, gammas, alphas, lrs, func_names = run_grid_search()

    # Save raw data
    with open(os.path.join(RESULTS_DIR, 'hp_robustness.json'), 'w') as f:
        json.dump({
            'gammas': gammas.tolist(),
            'alphas': alphas.tolist(),
            'lrs': lrs,
            'functions': func_names,
            'results': all_results,
        }, f, indent=2)

    plot_heatmaps(all_results, gammas, alphas, lrs, func_names)
    print("\nHP Robustness analysis complete.")
