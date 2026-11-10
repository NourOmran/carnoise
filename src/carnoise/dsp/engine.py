"""Stage 1 - engine-order removal.

There is no RPM signal, so the audio is the tachometer: the engine's firing frequency f0
(25-90 Hz) is tracked frame by frame as the strongest prominent peak in that band, and its
harmonics (2*f0 ... up to ~900 Hz) are pulled down to the level of the spectrum around them.

A fine STFT (8192 samples, ~5.9 Hz bins) is needed because the harmonics are only 30-75 Hz
apart. f0 is refined by parabolic interpolation, since a bin error is multiplied by the
harmonic number.
"""
from __future__ import annotations

import numpy as np
from scipy import ndimage

from .spectral import istft, stft

F0_BAND = (25.0, 90.0)
N_HARMONICS = 12
MAX_HZ = 900.0
PROMINENCE_DB = 6.0
MAX_CUT_DB = 20.0
N_FFT = 8192


def track_f0(f: np.ndarray, P: np.ndarray, f0_band=F0_BAND, prom_db: float = PROMINENCE_DB) -> np.ndarray:
    """Per-frame engine firing frequency (Hz) from a power spectrogram; NaN where no clear line."""
    df = f[1]
    band = np.where((f >= f0_band[0]) & (f <= f0_band[1]))[0]
    Pd = 10 * np.log10(P + 1e-30)
    ctx = ndimage.median_filter(Pd, size=(9, 1), mode="nearest")
    k = band[np.argmax(Pd[band], axis=0)]
    cols = np.arange(P.shape[1])
    prom = Pd[k, cols] - ctx[k, cols]
    a, b, c = Pd[k - 1, cols], Pd[k, cols], Pd[k + 1, cols]
    delta = 0.5 * (a - c) / np.where(np.abs(a - 2 * b + c) > 1e-9, a - 2 * b + c, 1e-9)
    f0 = np.where(prom >= prom_db, f[k] + np.clip(delta, -0.5, 0.5) * df, np.nan)
    good = ~np.isnan(f0)
    if good.sum() > 3:
        # Fill short gaps from neighbouring tracked frames. Interpolation is local on purpose:
        # a chunk-wide statistic here would make chunked output differ from whole-file output.
        idx = np.arange(len(f0))
        f0s = ndimage.median_filter(np.interp(idx, idx[good], f0[good]), size=7, mode="nearest")
        f0 = np.where(ndimage.binary_dilation(good, iterations=2), f0s, np.nan)
    return f0


def harmonic_gain(f: np.ndarray, P: np.ndarray, f0: np.ndarray, n_harm: int = N_HARMONICS,
                  max_hz: float = MAX_HZ, max_cut_db: float = MAX_CUT_DB) -> np.ndarray:
    """Gain mask (bins, frames) that notches each harmonic down to its neighbourhood level."""
    G = np.ones_like(P)
    for t in np.where(~np.isnan(f0))[0]:
        for h in range(2, n_harm + 1):
            fh = h * f0[t]
            if fh > max_hz:
                break
            # snap to the actual peak near h*f0 (search range grows with order), notch +-2 bins
            search = np.where(np.abs(f - fh) <= 4.0 + 0.03 * fh)[0]
            if len(search) == 0:
                continue
            pk = search[np.argmax(P[search, t])]
            on = slice(max(pk - 2, 0), pk + 3)
            off = np.r_[pk - 8:pk - 3, pk + 4:pk + 9]
            off = off[(off >= 0) & (off < len(f))]
            ref = np.median(P[off, t])
            G[on, t] = np.minimum(1.0, np.sqrt(ref / np.maximum(P[on, t], 1e-30)))
    return np.maximum(G, 10 ** (-max_cut_db / 20))


def remove_engine_orders(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Remove engine harmonics from an (n, C) signal.

    Returns the processed signal and the per-frame f0 track (NaN where no engine was found).
    One mask is computed from the mean power of all channels and applied to each.
    """
    n = N_FFT
    hop = n // 4
    f, X = stft(x, n, hop)
    P = np.mean(np.abs(X) ** 2, axis=0)
    f0 = track_f0(f, P)
    G = harmonic_gain(f, P, f0)
    return istft(X * G[None], len(x), n, hop), f0
