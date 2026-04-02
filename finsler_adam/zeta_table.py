"""
Zeta Resonance Table & Noise Generator

Uses the imaginary parts of the first 100 non-trivial zeros of the
Riemann zeta function as frequencies for structured exploration noise.

Key frequencies:
    rho_1 ≈ 14.1347 (ground state / stabilization)
    rho_2 ≈ 21.0220 (awakening transition)
    rho_3 ≈ 25.0109 (edge of chaos approach)

Noise formula:
    xi_d(t) = amplitude * sum_{n=1}^{N} sin(gamma_n * ln(t + 1) + d*pi/D) / sqrt(n)

This creates deterministic but non-periodic perturbation that helps
escape local minima through resonance with the arithmetic structure.
"""

import math
import torch
from torch import Tensor
from typing import List


# Imaginary parts of first 100 non-trivial zeros of the Riemann zeta function.
# Source: LMFDB / Andrew Odlyzko's tables (verified to 4+ decimal places).
_ZETA_ZEROS: List[float] = [
    14.1347, 21.0220, 25.0109, 30.4249, 32.9351,
    37.5862, 40.9187, 43.3271, 48.0052, 49.7738,
    52.9703, 56.4462, 59.3470, 60.8318, 65.1125,
    67.0798, 69.5464, 72.0672, 75.7047, 77.1448,
    79.3374, 82.9104, 84.7355, 87.4253, 88.8091,
    92.4919, 94.6513, 95.8706, 98.8312, 101.318,
    103.726, 105.447, 107.169, 111.030, 111.875,
    114.320, 116.227, 118.791, 121.370, 122.947,
    124.257, 127.517, 129.579, 131.088, 133.498,
    134.757, 138.116, 139.736, 141.124, 143.112,
    146.001, 147.423, 150.054, 150.925, 153.025,
    156.113, 157.598, 158.850, 161.189, 163.031,
    165.537, 167.184, 169.095, 169.912, 173.412,
    174.754, 176.441, 178.377, 179.916, 182.207,
    184.874, 185.599, 187.229, 189.416, 192.027,
    193.080, 195.265, 196.876, 198.015, 201.265,
    202.494, 204.190, 205.395, 207.906, 209.577,
    211.691, 213.348, 214.547, 216.170, 219.068,
    220.715, 221.431, 224.007, 224.983, 227.421,
    229.337, 231.250, 231.987, 233.693, 236.524,
]


def get_zeros(n: int = 100) -> List[float]:
    """Return the imaginary parts of the first n non-trivial zeros of zeta(s).

    Args:
        n: number of zeros to return (max 100)

    Returns:
        List of gamma_n values
    """
    n = min(n, len(_ZETA_ZEROS))
    return _ZETA_ZEROS[:n]


def zeta_noise(step_count: int, dim: int, amplitude: float = 1e-4,
               n_zeros: int = 20) -> Tensor:
    """Generate zeta resonance noise vector.

    Formula:
        xi_d(t) = amplitude * sum_{n=1}^{N} sin(gamma_n * ln(t+1) + d*pi/D) / sqrt(n)

    The noise is deterministic for a given step_count (no randomness),
    non-periodic (irrational frequency ratios from zeta zeros), and
    structured (harmonic content encodes arithmetic regularity).

    Args:
        step_count: current optimization step (t >= 0)
        dim: dimension of the noise vector
        amplitude: scaling factor for the noise
        n_zeros: number of zeta zeros to use

    Returns:
        Tensor of shape (dim,) containing structured noise
    """
    zeros = get_zeros(n_zeros)
    log_t = math.log(step_count + 1)  # +1 guards against ln(0)

    noise_data = []
    for d in range(dim):
        phase_shift = d * math.pi / max(dim, 1)
        val = 0.0
        for n_idx, gamma_n in enumerate(zeros):
            n = n_idx + 1
            val += math.sin(gamma_n * log_t + phase_shift) / math.sqrt(n)
        noise_data.append(amplitude * val)

    return torch.tensor(noise_data)
