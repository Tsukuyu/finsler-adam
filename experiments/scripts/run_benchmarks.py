"""
Finsler-Adam vs AdamW: Comprehensive Empirical Comparison v2
==============================================================
Focused experiments with better hyperparameters and specific scenarios.
"""

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
import csv
import time
from finsler_adam.optimizer import FinslerAdam
from finsler_adam.anna_limit import anna_clip

# ============================================================
# Test Functions
# ============================================================

def rosenbrock(params):
    x, y = params[0], params[1]
    return (1 - x)**2 + 100 * (y - x**2)**2

def ackley_nd(params):
    n = len(params)
    sum_sq = sum(p**2 for p in params)
    sum_cos = sum(torch.tensor(np.cos(2*np.pi*p.item())) for p in params)
    return -20*torch.tensor(np.exp(-0.2*np.sqrt(sum_sq.item()/n))) - \
           torch.tensor(np.exp(sum_cos.item()/n)) + 20 + torch.tensor(np.e)

def rastrigin_nd(params):
    n = len(params)
    return torch.tensor(10.0*n) + sum(p**2 - 10*torch.tensor(np.cos(2*np.pi*p.item())) for p in params)

def sphere_nd(params):
    return sum(p**2 for p in params)

def steep_valley(params):
    """Custom: very steep valley that causes gradient explosions."""
    x, y = params[0], params[1]
    return (x**2 + y - 11)**2 + (x + y**2 - 7)**2 + 0.1*(x**4 + y**4)

EXPERIMENTS = {
    'Rosenbrock-2D': {'func': rosenbrock, 'dim': 2, 'lo': -2, 'hi': 2},
    'Sphere-5D': {'func': sphere_nd, 'dim': 5, 'lo': -5, 'hi': 5},
    'Ackley-5D': {'func': ackley_nd, 'dim': 5, 'lo': -3, 'hi': 3},
    'Rastrigin-5D': {'func': rastrigin_nd, 'dim': 5, 'lo': -3, 'hi': 3},
    'SteepValley-2D': {'func': steep_valley, 'dim': 2, 'lo': -4, 'hi': 4},
}

# ============================================================
# Numerical gradient helper
# ============================================================

def compute_grad(func, params, eps=1e-5):
    grads = []
    for i, p in enumerate(params):
        orig = p._data[0]
        p._data[0] = orig + eps
        fp = func(params)
        lp = float(fp._data.flat[0]) if hasattr(fp, '_data') else float(fp)
        p._data[0] = orig - eps
        fm = func(params)
        lm = float(fm._data.flat[0]) if hasattr(fm, '_data') else float(fm)
        p._data[0] = orig
        grads.append((lp - lm) / (2 * eps))
    return grads

# ============================================================
# Run optimization
# ============================================================

def optimize(func, dim, lo, hi, opt_factory, steps=1000, seed=42):
    np.random.seed(seed)
    params = [torch.tensor(np.random.uniform(lo, hi, 1).astype(np.float32), requires_grad=True)
              for _ in range(dim)]
    optimizer = opt_factory(params)
    history = []

    for step in range(steps):
        optimizer.zero_grad()
        loss = func(params)
        loss_val = float(loss._data.flat[0]) if hasattr(loss, '_data') else float(loss)
        history.append(loss_val)

        grads = compute_grad(func, params)
        for p, g in zip(params, grads):
            p.grad = torch.tensor([g])

        optimizer.step()

        # Check finite
        for p in params:
            if not np.isfinite(p._data[0]):
                history.extend([float('inf')] * (steps - step - 1))
                return history, False

    return history, True

# ============================================================
# Experiment Suite
# ============================================================

def main():
    print("=" * 70)
    print("FINSLER-ADAM vs AdamW: EMPIRICAL COMPARISON v2")
    print("=" * 70)

    LR_CONFIGS = {
        'conservative': 0.005,
        'standard': 0.01,
        'aggressive': 0.05,
    }

    OPT_CONFIGS = {
        'AdamW': lambda p, lr: torch.optim.AdamW(p, lr=lr),
        'Finsler-Full': lambda p, lr: FinslerAdam(p, lr=lr, gamma=0.5, anna_alpha=0.1, zeta_enabled=False),
        'Finsler-NoAnna': lambda p, lr: FinslerAdam(p, lr=lr, gamma=0.5, anna_alpha=0.0, zeta_enabled=False),
        'Finsler-AnnaOnly': lambda p, lr: FinslerAdam(p, lr=lr, gamma=0.0, anna_alpha=0.1, zeta_enabled=False),
    }

    NUM_RUNS = 3
    STEPS = 800

    all_data = {}
    summary_rows = []
    all_histories = {}

    for exp_name, exp_info in EXPERIMENTS.items():
        print(f"\n{'='*50}")
        print(f"  {exp_name} (dim={exp_info['dim']})")
        print(f"{'='*50}")

        all_data[exp_name] = {}
        all_histories[exp_name] = {}

        for lr_name, lr_val in LR_CONFIGS.items():
            print(f"\n  LR={lr_val} ({lr_name}):")

            for opt_name, opt_factory in OPT_CONFIGS.items():
                finals = []
                steps_to_thresh = []
                finite_count = 0
                run_histories = []

                threshold = {
                    'Rosenbrock-2D': 1.0,
                    'Sphere-5D': 0.1,
                    'Ackley-5D': 3.0,
                    'Rastrigin-5D': 20.0,
                    'SteepValley-2D': 1.0,
                }.get(exp_name, 1.0)

                for run in range(NUM_RUNS):
                    seed = 100 + run * 13
                    factory = lambda p, lv=lr_val, of=opt_factory: of(p, lv)
                    hist, finite = optimize(
                        exp_info['func'], exp_info['dim'],
                        exp_info['lo'], exp_info['hi'],
                        factory, steps=STEPS, seed=seed
                    )
                    run_histories.append(hist)

                    if finite:
                        finite_count += 1
                        finals.append(hist[-1])
                        conv = STEPS
                        for i, v in enumerate(hist):
                            if v < threshold:
                                conv = i
                                break
                        steps_to_thresh.append(conv)
                    else:
                        finals.append(float('inf'))
                        steps_to_thresh.append(STEPS)

                avg_final = np.mean([f for f in finals if np.isfinite(f)]) if any(np.isfinite(f) for f in finals) else float('inf')
                std_final = np.std([f for f in finals if np.isfinite(f)]) if sum(np.isfinite(f) for f in finals) > 1 else 0
                avg_conv = np.mean(steps_to_thresh)
                stability = finite_count / NUM_RUNS

                key = f"{lr_name}"
                if key not in all_data[exp_name]:
                    all_data[exp_name][key] = {}
                all_data[exp_name][key][opt_name] = {
                    'avg_final': float(avg_final),
                    'std_final': float(std_final),
                    'avg_conv_step': float(avg_conv),
                    'stability': stability,
                    'best_final': float(min(f for f in finals if np.isfinite(f))) if any(np.isfinite(f) for f in finals) else float('inf'),
                }

                # Store avg history for plotting
                hist_key = f"{exp_name}_{lr_name}_{opt_name}"
                avg_hist = np.mean(run_histories, axis=0)
                all_histories[hist_key] = avg_hist.tolist()

                print(f"    {opt_name:20s} | final={avg_final:12.4f} ± {std_final:8.4f} | conv={avg_conv:5.0f} | stable={stability*100:.0f}%")

                summary_rows.append({
                    'function': exp_name,
                    'lr_regime': lr_name,
                    'lr': lr_val,
                    'optimizer': opt_name,
                    'avg_final_loss': avg_final,
                    'std_final_loss': std_final,
                    'avg_conv_step': avg_conv,
                    'stability': stability,
                    'best_final': float(min(f for f in finals if np.isfinite(f))) if any(np.isfinite(f) for f in finals) else float('inf'),
                })

    # --- Experiment 2: Gradient Explosion Stress Test ---
    print(f"\n{'='*50}")
    print("  GRADIENT EXPLOSION STRESS TEST")
    print(f"{'='*50}")

    explosion_results = {}
    for opt_name in ['AdamW', 'Finsler-Full', 'Finsler-AnnaOnly']:
        survived = 0
        total = 30
        final_params = []

        for trial in range(total):
            np.random.seed(200 + trial)
            params = [torch.tensor([float(np.random.randn()) * 10], requires_grad=True)]

            if opt_name == 'AdamW':
                opt = torch.optim.AdamW(params, lr=0.01)
            elif opt_name == 'Finsler-Full':
                opt = FinslerAdam(params, lr=0.01, gamma=0.5, anna_alpha=0.1, zeta_enabled=False)
            else:
                opt = FinslerAdam(params, lr=0.01, gamma=0.0, anna_alpha=0.1, zeta_enabled=False)

            ok = True
            for step in range(200):
                opt.zero_grad()
                # Exponentially growing gradient
                scale = 10.0 ** (step / 40.0)
                direction = 1.0 if np.random.rand() > 0.5 else -1.0
                params[0].grad = torch.tensor([scale * direction])
                opt.step()

                if not np.isfinite(params[0]._data[0]):
                    ok = False
                    break

            if ok:
                survived += 1
                final_params.append(abs(float(params[0]._data[0])))

        avg_param = np.mean(final_params) if final_params else float('inf')
        explosion_results[opt_name] = {
            'survived': survived, 'total': total,
            'rate': survived/total, 'avg_final_param': avg_param
        }
        print(f"  {opt_name:20s} | survived: {survived}/{total} ({survived/total*100:.0f}%) | avg|param|={avg_param:.2f}")

    # --- Experiment 3: Anna-Clip Response Curve ---
    print(f"\n{'='*50}")
    print("  ANNA-CLIP RESPONSE CURVES")
    print(f"{'='*50}")

    magnitudes = np.logspace(-2, 6, 200)
    clip_curves = {}
    for alpha in [0.01, 0.05, 0.1, 0.5, 1.0]:
        ratios = []
        for m in magnitudes:
            g = torch.tensor([m])
            c = anna_clip(g, alpha)
            ratios.append(float(c._data[0]) / m)
        clip_curves[alpha] = ratios
        print(f"  alpha={alpha}: ratio@1={ratios[50]:.4f}, ratio@100={ratios[100]:.4f}, ratio@1e6={ratios[-1]:.8f}")

    # --- Save Everything ---
    os.makedirs('experiments/results', exist_ok=True)

    # Summary CSV
    with open('experiments/results/summary.csv', 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=summary_rows[0].keys())
        writer.writeheader()
        writer.writerows(summary_rows)

    # Full JSON
    output = {
        'convergence': all_data,
        'explosion': explosion_results,
        'anna_clip': {'magnitudes': magnitudes.tolist(), 'curves': {str(k): v for k, v in clip_curves.items()}},
        'histories': {k: v for k, v in all_histories.items()},
        'metadata': {'num_runs': NUM_RUNS, 'steps': STEPS, 'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')}
    }
    with open('experiments/results/full_results.json', 'w') as f:
        json.dump(output, f)

    print(f"\n{'='*70}")
    print("ALL EXPERIMENTS COMPLETE — results in experiments/results/")
    print(f"{'='*70}")
    return output

if __name__ == '__main__':
    main()
