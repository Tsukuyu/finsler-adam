"""
FinslerAdam — Asymmetric-metric optimizer with critical scaling gradient clipping.

Architecture:
    1. Anna-Limit clipping:  g ← g / (1 + α|g|^{4/3})     (if α > 0)
    2. Adam moment updates:  m, v ← EMA(g), EMA(g²)
    3. Finsler scaling:      M_i = 1 + γ·sgn(v_i · m_i)    (if γ > 0)
    4. AdamW weight decay:   θ ← (1 − ηλ)θ
    5. Parameter update:     θ ← θ − η · M ⊙ m̂ / (√v̂ + ε)

When γ=0 and α=0, this reduces exactly to standard AdamW.
"""

import math
import torch
from torch import Tensor
from torch.optim.optimizer import Optimizer
from typing import List, Optional, Tuple


def _anna_clip(grad: Tensor, alpha: float) -> Tensor:
    """Smooth gradient clipping with Navier-Stokes 4/3 critical exponent.

    Formula: g_out = g / (1 + α|g|^{4/3})

    For |g| ≪ 1:  g_out ≈ g         (transparent — no distortion)
    For |g| ≫ 1:  g_out → 0         (smooth saturation)

    The 4/3 exponent comes from the critical scaling in 3D Navier-Stokes
    regularity theory: (d+1)/d for spatial dimension d=3.
    """
    if alpha == 0.0:
        return grad
    abs_grad = grad.abs()
    return grad / (1.0 + alpha * abs_grad.pow(4.0 / 3.0))


class FinslerAdam(Optimizer):
    r"""Finsler-Adam: AdamW extended with asymmetric metric scaling and smooth clipping.

    .. math::
        M_i &= 1 + \gamma \cdot \mathrm{sgn}(v_i \cdot m_i) \\
        g_{\mathrm{clip}} &= g \,/\, (1 + \alpha\,|g|^{4/3}) \\
        \theta_{t+1} &= (1-\eta\lambda)\,\theta_t
                        - \eta\, M \odot \hat{m}_t \,/\, (\sqrt{\hat{v}_t} + \epsilon)

    Args:
        params: Iterable of parameters to optimize.
        lr (float): Learning rate. Default: 1e-3.
        betas (Tuple[float, float]): Coefficients for first/second moment EMA.
            Default: (0.9, 0.999).
        eps (float): Denominator epsilon for numerical stability. Default: 1e-8.
        weight_decay (float): Decoupled weight decay (AdamW style). Default: 0.01.
        gamma (float): Finsler asymmetry strength in [0, 1). 0 disables scaling
            and recovers AdamW. Default: 0.5.
        anna_alpha (float): Anna-Limit clipping strength. 0 disables clipping.
            Default: 0.1.

    Example::

        >>> optimizer = FinslerAdam(model.parameters(), lr=1e-3, gamma=0.5, anna_alpha=0.1)
        >>> for data, target in dataloader:
        ...     optimizer.zero_grad()
        ...     loss = criterion(model(data), target)
        ...     loss.backward()
        ...     optimizer.step()

    Note:
        - ``gamma=0, anna_alpha=0`` → exact AdamW (drop-in replacement)
        - ``gamma=0, anna_alpha>0`` → AdamW + Anna-Limit safety (recommended first step)
        - ``gamma>0, anna_alpha>0`` → Full Finsler-Adam
    """

    def __init__(
        self,
        params,
        lr: float = 1e-3,
        betas: Tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.01,
        gamma: float = 0.5,
        anna_alpha: float = 0.1,
    ):
        if not 0.0 <= lr:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= eps:
            raise ValueError(f"Invalid epsilon value: {eps}")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 0: {betas[0]}")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 1: {betas[1]}")
        if not 0.0 <= gamma < 1.0:
            raise ValueError(f"Invalid gamma (Finsler strength): {gamma}")
        if not 0.0 <= anna_alpha:
            raise ValueError(f"Invalid anna_alpha: {anna_alpha}")

        defaults = dict(
            lr=lr, betas=betas, eps=eps,
            weight_decay=weight_decay,
            gamma=gamma, anna_alpha=anna_alpha,
        )
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        """Perform a single optimization step.

        Args:
            closure (callable, optional): A closure that re-evaluates the model
                and returns the loss.

        Returns:
            Optional loss from the closure.
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group["lr"]
            beta1, beta2 = group["betas"]
            eps = group["eps"]
            weight_decay = group["weight_decay"]
            gamma = group["gamma"]
            anna_alpha = group["anna_alpha"]

            for p in group["params"]:
                if p.grad is None:
                    continue

                grad = p.grad
                if grad.is_sparse:
                    raise RuntimeError("FinslerAdam does not support sparse gradients")

                # ---- Anna-Limit clipping (before moment updates) ----
                if anna_alpha > 0.0:
                    grad = _anna_clip(grad, anna_alpha)

                # ---- State initialization ----
                state = self.state[p]
                if len(state) == 0:
                    state["step"] = 0
                    state["exp_avg"] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    state["exp_avg_sq"] = torch.zeros_like(p, memory_format=torch.preserve_format)

                state["step"] += 1
                step = state["step"]
                exp_avg = state["exp_avg"]
                exp_avg_sq = state["exp_avg_sq"]

                # ---- Decoupled weight decay (AdamW) ----
                if weight_decay != 0.0:
                    p.mul_(1.0 - lr * weight_decay)

                # ---- Moment updates ----
                exp_avg.mul_(beta1).add_(grad, alpha=1.0 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1.0 - beta2)

                # ---- Bias correction ----
                bias_correction1 = 1.0 - beta1 ** step
                bias_correction2 = 1.0 - beta2 ** step
                step_size = lr / bias_correction1
                denom = (exp_avg_sq.sqrt() / math.sqrt(bias_correction2)).add_(eps)

                # ---- Finsler asymmetric scaling ----
                if gamma != 0.0:
                    # M_i = 1 + γ · sgn(v_i · m_i)
                    finsler_scale = (exp_avg_sq * exp_avg).sign().mul_(gamma).add_(1.0)
                    p.addcmul_(exp_avg / denom, finsler_scale, value=-step_size)
                else:
                    p.add_(exp_avg / denom, alpha=-step_size)

        return loss
