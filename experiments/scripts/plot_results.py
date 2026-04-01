"""Generate publication-quality plots from experiment results."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
try:
    import torch
except ImportError:
    import _torch_mock
import torch
import numpy as np
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import rcParams
from finsler_adam.anna_limit import anna_clip

rcParams['font.size'] = 10
rcParams['axes.labelsize'] = 11
rcParams['figure.dpi'] = 150

OUT = 'experiments/results'
os.makedirs(OUT, exist_ok=True)

# Load histories
with open(f'{OUT}/full_results.json') as f:
    data = json.load(f)

histories = data['histories']

# ============================================================
# Fig 1: Convergence curves (best LR per function)
# ============================================================
fig, axes = plt.subplots(2, 3, figsize=(15, 9))
axes = axes.flatten()

best_lr = {
    'Rosenbrock-2D': 'aggressive',
    'Sphere-5D': 'aggressive',
    'Ackley-5D': 'standard',
    'Rastrigin-5D': 'standard',
    'SteepValley-2D': 'standard',
}

colors = {'AdamW': '#2196F3', 'Finsler-Full': '#F44336',
          'Finsler-NoAnna': '#FF9800', 'Finsler-AnnaOnly': '#4CAF50'}
linestyles = {'AdamW': '-', 'Finsler-Full': '-', 'Finsler-NoAnna': '--', 'Finsler-AnnaOnly': '-.'}

for idx, (func_name, lr_name) in enumerate(best_lr.items()):
    ax = axes[idx]
    for opt_name in ['AdamW', 'Finsler-Full', 'Finsler-NoAnna', 'Finsler-AnnaOnly']:
        key = f"{func_name}_{lr_name}_{opt_name}"
        if key in histories:
            h = histories[key]
            # Subsample for clarity
            steps = range(0, len(h), max(1, len(h)//200))
            vals = [h[i] for i in steps]
            ax.plot(list(steps), vals, label=opt_name, color=colors[opt_name],
                    linestyle=linestyles[opt_name], linewidth=1.5, alpha=0.85)

    ax.set_title(f"{func_name} (lr={best_lr[func_name]})", fontsize=11, fontweight='bold')
    ax.set_xlabel('Step')
    ax.set_ylabel('Loss')
    ax.set_yscale('log')
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)

# Summary table in last subplot
ax = axes[5]
ax.axis('off')
conv_data = data['convergence']
table_data = []
headers = ['Function', 'AdamW\nFinal', 'Finsler-Full\nFinal', 'Δ%']
for func_name, lr_name in best_lr.items():
    adamw = conv_data[func_name][lr_name]['AdamW']['avg_final']
    finsler = conv_data[func_name][lr_name]['Finsler-Full']['avg_final']
    delta = ((finsler - adamw) / abs(adamw) * 100) if adamw != 0 else 0
    table_data.append([func_name.replace('-', '\n'), f"{adamw:.4f}", f"{finsler:.4f}",
                        f"{delta:+.1f}%"])

table = ax.table(cellText=table_data, colLabels=headers, loc='center', cellLoc='center')
table.auto_set_font_size(False)
table.set_fontsize(9)
table.scale(1.0, 1.6)
ax.set_title('Summary: Final Loss Comparison', fontsize=11, fontweight='bold', pad=20)

plt.tight_layout()
plt.savefig(f'{OUT}/fig1_convergence_curves.png', bbox_inches='tight')
plt.close()
print("Fig 1: Convergence curves saved.")

# ============================================================
# Fig 2: Anna-Clip Response Curves
# ============================================================
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

magnitudes = np.logspace(-2, 6, 500)
alphas = [0.01, 0.05, 0.1, 0.5, 1.0]
cmap = plt.cm.viridis(np.linspace(0.1, 0.9, len(alphas)))

for i, alpha in enumerate(alphas):
    outputs = []
    ratios = []
    for m in magnitudes:
        g = torch.tensor([m])
        c = anna_clip(g, alpha)
        cv = float(c._data[0])
        outputs.append(cv)
        ratios.append(cv / m if m > 0 else 1.0)

    ax1.plot(magnitudes, outputs, label=f'α={alpha}', color=cmap[i], linewidth=2)
    ax2.plot(magnitudes, ratios, label=f'α={alpha}', color=cmap[i], linewidth=2)

ax1.plot(magnitudes, magnitudes, 'k--', alpha=0.3, label='Identity (no clip)')
ax1.set_xscale('log')
ax1.set_yscale('log')
ax1.set_xlabel('Input Gradient Magnitude')
ax1.set_ylabel('Output Gradient Magnitude')
ax1.set_title('Anna-Limit: Input → Output', fontweight='bold')
ax1.legend()
ax1.grid(True, alpha=0.3)

ax2.set_xscale('log')
ax2.set_xlabel('Input Gradient Magnitude')
ax2.set_ylabel('Preservation Ratio (output/input)')
ax2.set_title('Anna-Limit: Preservation Ratio', fontweight='bold')
ax2.axhline(y=1.0, color='k', linestyle='--', alpha=0.3)
ax2.legend()
ax2.grid(True, alpha=0.3)
ax2.set_ylim(-0.05, 1.1)

plt.tight_layout()
plt.savefig(f'{OUT}/fig2_anna_clip_response.png', bbox_inches='tight')
plt.close()
print("Fig 2: Anna-Clip response saved.")

# ============================================================
# Fig 3: Learning Rate Sensitivity
# ============================================================
fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))

funcs = ['Rosenbrock-2D', 'Sphere-5D', 'Rastrigin-5D']
lr_names = ['conservative', 'standard', 'aggressive']
lr_vals = [0.005, 0.01, 0.05]

for idx, func_name in enumerate(funcs):
    ax = axes[idx]
    x = np.arange(len(lr_names))
    width = 0.2

    for j, opt_name in enumerate(['AdamW', 'Finsler-Full', 'Finsler-NoAnna', 'Finsler-AnnaOnly']):
        vals = []
        for lr_name in lr_names:
            v = conv_data[func_name][lr_name][opt_name]['avg_final']
            vals.append(v if np.isfinite(v) else 0)
        ax.bar(x + j*width, vals, width, label=opt_name if idx == 0 else '',
               color=colors[opt_name], alpha=0.8)

    ax.set_xticks(x + 1.5*width)
    ax.set_xticklabels([f"lr={v}" for v in lr_vals])
    ax.set_title(func_name, fontweight='bold')
    ax.set_ylabel('Final Loss')
    ax.grid(True, alpha=0.2, axis='y')

axes[0].legend(fontsize=8, loc='upper right')
plt.tight_layout()
plt.savefig(f'{OUT}/fig3_lr_sensitivity.png', bbox_inches='tight')
plt.close()
print("Fig 3: LR sensitivity saved.")

# ============================================================
# Fig 4: Finsler Scaling Mechanism Visualization
# ============================================================
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

# Left: Show M(v) as function of alignment
alignment = np.linspace(-1, 1, 100)
for gamma in [0.1, 0.3, 0.5, 0.7, 0.9]:
    M = 1 + gamma * np.sign(alignment)
    ax1.plot(alignment, M, label=f'γ={gamma}', linewidth=2)

ax1.axhline(y=1.0, color='k', linestyle='--', alpha=0.3, label='AdamW (γ=0)')
ax1.axvline(x=0, color='gray', linestyle=':', alpha=0.3)
ax1.set_xlabel('Alignment sgn(v·m)')
ax1.set_ylabel('Scaling Factor M(v)')
ax1.set_title('Finsler Scaling: Alignment → Step Size', fontweight='bold')
ax1.legend()
ax1.grid(True, alpha=0.3)

# Right: Combined effect diagram
categories = ['Aligned\n(accelerate)', 'Misaligned\n(brake)', 'Small grad\n(preserve)', 'Large grad\n(Anna clip)']
adamw_effect = [1.0, 1.0, 1.0, 1.0]
finsler_effect = [1.5, 0.5, 0.9, 0.1]

x = np.arange(len(categories))
ax2.bar(x - 0.2, adamw_effect, 0.35, label='AdamW', color='#2196F3', alpha=0.8)
ax2.bar(x + 0.2, finsler_effect, 0.35, label='Finsler-Adam', color='#F44336', alpha=0.8)
ax2.set_xticks(x)
ax2.set_xticklabels(categories, fontsize=9)
ax2.set_ylabel('Effective Step Scale')
ax2.set_title('Adaptive Behavior: AdamW vs Finsler-Adam', fontweight='bold')
ax2.legend()
ax2.grid(True, alpha=0.2, axis='y')

plt.tight_layout()
plt.savefig(f'{OUT}/fig4_mechanism.png', bbox_inches='tight')
plt.close()
print("Fig 4: Mechanism visualization saved.")

print("\nAll figures saved to experiments/results/")
