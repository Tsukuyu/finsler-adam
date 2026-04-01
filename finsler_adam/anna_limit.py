"""
Anna-Limit: Smooth Gradient Clipping with Navier-Stokes Critical Exponent

Standalone utility for applying Anna-Limit clipping to any gradient tensor.
This can be used independently of Finsler-Adam, e.g., as a drop-in replacement
for torch.nn.utils.clip_grad_norm_ in existing training pipelines.

Formula:
    g_out = g / (1 + α |g|^{4/3})

The 4/3 exponent comes from the critical scaling in 3D Navier-Stokes
regularity theory: for spatial dimension d=3, the critical exponent
is (d+1)/d = 4/3.
"""

import torch
from torch import Tensor, nn
from typing import Union


def anna_clip(grad: Tensor, alpha: float = 0.1) -> Tensor:
    """Apply Anna-Limit smooth gradient clipping.

    Args:
        grad: Input gradient tensor (any shape).
        alpha: Clipping strength. 0 disables clipping. Default: 0.1.

    Returns:
        Clipped gradient tensor (same shape, same device).

    Example::

        >>> g = torch.randn(100)
        >>> g_clipped = anna_clip(g, alpha=0.1)
        >>> # Small gradients preserved: |g|<1 → <10% distortion
        >>> # Large gradients clipped:   |g|=100 → ~98% reduction
    """
    if alpha == 0.0:
        return grad
    return grad / (1.0 + alpha * grad.abs().pow(4.0 / 3.0))


def anna_clip_grad_(model: nn.Module, alpha: float = 0.1) -> None:
    """Apply Anna-Limit clipping to all gradients of a model (in-place).

    Drop-in replacement for ``torch.nn.utils.clip_grad_norm_``.

    Args:
        model: The neural network model.
        alpha: Clipping strength. Default: 0.1.

    Example::

        >>> loss.backward()
        >>> anna_clip_grad_(model, alpha=0.1)  # replaces clip_grad_norm_
        >>> optimizer.step()
    """
    for p in model.parameters():
        if p.grad is not None:
            p.grad.data = anna_clip(p.grad.data, alpha)
