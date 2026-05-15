"""AWR per-sample weight computation.

`w_i = clip(exp(advantage_i / beta), 0, max_weight)`.

The shift by `max(advantages / beta)` before `exp` keeps the computation
finite even when raw advantages are large; the relative scaling between
samples is preserved.
"""

import numpy as np


def awr_weights(
    advantages: np.ndarray,
    *,
    beta: float,
    max_weight: float = 20.0,
) -> np.ndarray:
    if beta <= 0:
        raise ValueError(f"beta must be > 0, got {beta}")
    adv = np.asarray(advantages, dtype=np.float32)
    scaled = adv / float(beta)
    # Subtract max for numerical stability — preserves relative order and ratios.
    scaled = scaled - scaled.max()
    w = np.exp(scaled).astype(np.float32)
    return np.clip(w, 0.0, max_weight).astype(np.float32)
