"""
Finsler-Adam: Asymmetric-Metric Optimizer with Critical Scaling Gradient Clipping

A PyTorch optimizer that extends AdamW with:
  1. Finsler Scaling  — direction-dependent step size from Finsler geometry
  2. Anna-Limit       — smooth 4/3-exponent gradient clipping from Navier-Stokes theory
  3. Zeta Resonance   — structured exploration noise from Riemann zeta zeros (optional)

Quick start:
    from finsler_adam import FinslerAdam

    optimizer = FinslerAdam(model.parameters(), lr=1e-3)

Paper: https://github.com/Tsukuyu/finsler-adam
License: MIT
"""

from finsler_adam.optimizer import FinslerAdam
from finsler_adam.anna_limit import anna_clip

__version__ = "0.1.0"
__all__ = ["FinslerAdam", "anna_clip"]
