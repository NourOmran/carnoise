"""Channel layout handling.

Some recorders write the same signal to both channels. Those files are processed as mono
(and written back to every channel by the file runner); real stereo pairs keep both channels
and share one gain mask so the stereo image stays stable.
"""
from __future__ import annotations

import numpy as np

from ..constants import DUP_RATIO


def as_2d(x: np.ndarray) -> np.ndarray:
    """(n,) -> (n, 1); (n, C) unchanged."""
    return x[:, None] if x.ndim == 1 else x


def is_duplicate_pair(x: np.ndarray) -> bool:
    """True when a 2-channel signal's channels are copies of each other."""
    if x.ndim != 2 or x.shape[1] != 2:
        return False
    d = np.mean((x[:, 0] - x[:, 1]) ** 2) / max(np.mean(x[:, 0] ** 2), 1e-30)
    return bool(d < DUP_RATIO)


def as_channels(x: np.ndarray) -> np.ndarray:
    """(n,) or (n, C) -> (n, C) with a duplicate second channel dropped."""
    x = as_2d(x)
    return x[:, :1] if is_duplicate_pair(x) else x


def select_channels(x: np.ndarray, mono: bool) -> np.ndarray:
    """Apply a channel decision made elsewhere (e.g. by the per-file analysis pass)."""
    x = as_2d(x)
    return x[:, :1] if mono else x
