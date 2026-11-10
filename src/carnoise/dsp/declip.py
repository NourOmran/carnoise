"""Stage 0 - declipping.

In these recordings clipping is mostly driven by low-frequency rumble hitting full scale.
Each run of clipped samples is re-drawn with a cubic spline through the unclipped samples
around it; the result keeps the original sign and is never smaller than the clip level
(the true waveform went at least that far).
"""
from __future__ import annotations

import numpy as np
from scipy import interpolate

from ..constants import CLIP_LEVEL


def find_clip_runs(x: np.ndarray, thr: float = CLIP_LEVEL) -> tuple[np.ndarray, np.ndarray]:
    """Start (inclusive) and end (exclusive) indices of clipped runs in a 1-D signal."""
    cl = np.abs(x) >= thr
    d = np.diff(np.r_[0, cl.astype(np.int8), 0])
    return np.where(d == 1)[0], np.where(d == -1)[0]


def declip(x: np.ndarray, thr: float = CLIP_LEVEL) -> tuple[np.ndarray, int]:
    """Repair clipped runs of one channel.

    Returns the repaired signal and the number of clipped runs found.
    """
    y = x.copy()
    cl = np.abs(x) >= thr
    if not cl.any():
        return y, 0
    starts, ends = find_clip_runs(x, thr)
    for s, e in zip(starts, ends):
        ctx = max(16, e - s)
        left = np.arange(max(0, s - ctx), s)
        right = np.arange(e, min(len(x), e + ctx))
        anchors = np.r_[left[~cl[left]], right[~cl[right]]]
        if len(anchors) < 4:
            continue
        step = max(1, len(anchors) // 32)
        a = anchors[::step]
        fill = interpolate.CubicSpline(a, x[a])(np.arange(s, e))
        sign = np.sign(x[s:e])
        y[s:e] = sign * np.clip(np.abs(fill), thr, 4.0)
    return y, len(starts)


def declip_channels(x: np.ndarray, thr: float = CLIP_LEVEL) -> tuple[np.ndarray, int]:
    """:func:`declip` applied to every column of an (n, C) signal. Returns (signal, total runs)."""
    y = x.copy()
    runs = 0
    for c in range(y.shape[1]):
        y[:, c], r = declip(y[:, c], thr)
        runs += r
    return y, runs
