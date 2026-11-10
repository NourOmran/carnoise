"""Multichannel STFT helpers shared by the DSP stages."""
from __future__ import annotations

import numpy as np
from scipy import signal

from ..constants import FS, HOP, NFFT


def stft(x: np.ndarray, n: int = NFFT, hop: int = HOP) -> tuple[np.ndarray, np.ndarray]:
    """STFT of a (n_samples, C) signal.

    Returns ``(f, X)`` with ``X`` shaped ``(C, bins, frames)``.
    """
    f, _, X = signal.stft(x.T, FS, nperseg=n, noverlap=n - hop)
    return f, X


def istft(X: np.ndarray, length: int, n: int = NFFT, hop: int = HOP) -> np.ndarray:
    """Inverse of :func:`stft`, trimmed to ``length`` samples. Returns (length, C)."""
    _, y = signal.istft(X, FS, nperseg=n, noverlap=n - hop)
    return y[:, :length].T
