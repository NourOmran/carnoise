"""Stages 2 and 2c - per-file high-pass and low-pass, with automatic corner selection.

High-pass: the corner sits just above where the low-frequency excess (engine sub-harmonics,
road rumble) over the 200-500 Hz level ends, clamped to 70-200 Hz.

Low-pass: the lowest 1 kHz band edge (>= 10 kHz) above which every band's level stays flat
over time (p95 - p10 of its 0.5 s levels < 2 dB). Such bands hold no events, only the
recorder's own noise, so removing them loses nothing. ``None`` when every band moves.

The ``*_from_psd`` / ``*_from_spec`` variants take precomputed spectra so the streaming
per-file analysis (:mod:`carnoise.analysis`) can make the same decision without loading
the whole file.
"""
from __future__ import annotations

from typing import Optional

import numpy as np
from scipy import ndimage, signal

from ..constants import FS

HP_RANGE = (70.0, 200.0)
HP_EXCESS_DB = 10.0
LP_FROM_HZ = 10000.0
LP_BAND_HZ = 1000.0
LP_SPREAD_DB = 2.0


# ---- high-pass ---------------------------------------------------------------------------

def highpass_from_psd(f: np.ndarray, P: np.ndarray, lo: float = HP_RANGE[0], hi: float = HP_RANGE[1],
                      excess_db: float = HP_EXCESS_DB) -> float:
    """High-pass corner (Hz) from a Welch PSD of the channel mean."""
    Pd = ndimage.uniform_filter1d(10 * np.log10(P + 1e-30), 5)
    ref = np.median(Pd[(f >= 200) & (f < 500)])
    band = (f >= 40) & (f <= 250)
    loud = f[band][Pd[band] > ref + excess_db]
    return float(np.clip(1.2 * loud.max(), lo, hi)) if len(loud) else lo


def auto_highpass(x: np.ndarray, lo: float = HP_RANGE[0], hi: float = HP_RANGE[1],
                  excess_db: float = HP_EXCESS_DB) -> float:
    """High-pass corner (Hz) for an (n, C) signal held in memory."""
    f, P = signal.welch(x.mean(axis=1), FS, nperseg=8192)
    return highpass_from_psd(f, P, lo, hi, excess_db)


def highpass(x: np.ndarray, hz: float) -> np.ndarray:
    """4th-order Butterworth high-pass, zero phase, along axis 0."""
    sos = signal.butter(4, hz, "highpass", fs=FS, output="sos")
    return signal.sosfiltfilt(sos, x, axis=0)


# ---- low-pass ----------------------------------------------------------------------------

def lowpass_from_spec(f: np.ndarray, S: np.ndarray, lo: float = LP_FROM_HZ, band: float = LP_BAND_HZ,
                      spread_db: float = LP_SPREAD_DB) -> Optional[float]:
    """Low-pass corner (Hz) from a power spectrogram with ~0.5 s frames, or None."""
    edges = np.arange(lo, FS / 2 - band + 1, band)
    steady = []
    for e in edges:
        lev = 10 * np.log10(S[(f >= e) & (f < e + band)].sum(0) + 1e-30)
        steady.append(np.percentile(lev, 95) - np.percentile(lev, 10) < spread_db)
    for i, e in enumerate(edges):
        if all(steady[i:]):
            return float(e)
    return None


def level_spectrogram(m: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Power spectrogram of a 1-D signal in 4096-sample frames (no overlap)."""
    f, _, S = signal.spectrogram(m, FS, nperseg=4096, noverlap=0)
    return f, S


def to_half_second_frames(f: np.ndarray, S: np.ndarray) -> np.ndarray:
    """Sum groups of 6 consecutive 4096-sample frames (~0.5 s)."""
    return S[:, :(S.shape[1] // 6) * 6].reshape(len(f), -1, 6).sum(2)


def auto_lowpass(x: np.ndarray, lo: float = LP_FROM_HZ, band: float = LP_BAND_HZ,
                 spread_db: float = LP_SPREAD_DB) -> Optional[float]:
    """Low-pass corner (Hz) for an (n, C) signal held in memory, or None."""
    f, S = level_spectrogram(x.mean(axis=1))
    return lowpass_from_spec(f, to_half_second_frames(f, S), lo, band, spread_db)


def lowpass(x: np.ndarray, hz: float) -> np.ndarray:
    """8th-order Butterworth low-pass, zero phase, along axis 0."""
    sos = signal.butter(8, hz, "lowpass", fs=FS, output="sos")
    return signal.sosfiltfilt(sos, x, axis=0)
