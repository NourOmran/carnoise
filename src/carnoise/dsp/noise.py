"""Stage 4 - broadband noise reduction (road, tyre, remaining wind and engine).

Noise estimate: minimum statistics. Per frequency bin, the power is smoothed recursively, its
minimum is taken over ~3.2 s and bias-corrected. This follows the slowly changing car noise floor
while ignoring street events that sit on top of it (a window this long also stops continuous speech
from being taken for noise).

Gain: decision-directed Wiener filter, smoothed 3x3 to tame "musical noise". GPS speed
(0-100 km/h) raises the over-subtraction (1.0 -> 1.4) and lowers the gain floor
(-12 -> -18 dB), because road and wind noise grow with speed. The floor is what keeps the
street sounding natural: no bin is ever cut by more than it.

Settings tuned on half of the QE evaluation's controlled mixtures and checked on the other half.
"""
from __future__ import annotations

import numpy as np
from scipy import ndimage, signal

FLOOR_DB = (-12.0, -18.0)    # gain floor at 0 km/h and at >= 100 km/h
OVERSUB = (1.0, 1.4)         # noise over-subtraction at 0 km/h and at >= 100 km/h
SPEED_FULL_KMH = 100.0
NOISE_WINDOW = 300           # frames of 512 samples (~3.2 s)
NOISE_BIAS = 1.4


def noise_psd(P: np.ndarray, win_frames: int = NOISE_WINDOW, alpha: float = 0.9, bias: float = NOISE_BIAS) -> np.ndarray:
    """Minimum-statistics noise power estimate for a (bins, frames) power array."""
    S = signal.lfilter([1 - alpha], [1, -alpha], P, axis=1)
    N = ndimage.minimum_filter1d(S, size=win_frames, axis=1, mode="nearest")
    N = ndimage.uniform_filter1d(N, size=win_frames // 2, axis=1, mode="nearest")
    N = ndimage.uniform_filter1d(N, size=3, axis=0, mode="nearest")
    return bias * N


def wiener_gain(P: np.ndarray, N: np.ndarray, speed: np.ndarray, floor_db=FLOOR_DB, beta=OVERSUB,
                a: float = 0.98) -> np.ndarray:
    """Decision-directed Wiener gain (bins, frames); ``speed`` is km/h per frame."""
    s = np.clip(speed / SPEED_FULL_KMH, 0, 1)
    b = beta[0] + (beta[1] - beta[0]) * s
    gmin = 10 ** ((floor_db[0] + (floor_db[1] - floor_db[0]) * s) / 20)
    Nb = N * b[None, :]
    gamma = P / np.maximum(Nb, 1e-30)
    G = np.empty_like(P)
    prev = np.ones(P.shape[0])
    for t in range(P.shape[1]):
        xi = a * prev + (1 - a) * np.maximum(gamma[:, t] - 1, 0)
        g = xi / (1 + xi)
        G[:, t] = g
        prev = g ** 2 * gamma[:, t]
    G = ndimage.median_filter(G, size=(3, 3), mode="nearest")
    return np.maximum(G, gmin[None, :])
