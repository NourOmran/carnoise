"""Stage 2b - persistent tonal lines (fans, alternator, recorder electronics).

These show up as narrow spectral lines between 150 Hz and 4 kHz that last for seconds and
may drift slowly (e.g. the 520-590 Hz cluster on Unit 4). They do not follow the engine, so
stage 1 does not catch them.

Detection, on a 32768-sample STFT (1.46 Hz bins, fine enough to split lines ~10 Hz apart):

1. average power over 0.7 s, so random noise peaks shrink while a steady line keeps its height;
2. a bin is a *peak* when it is a local maximum >= 6 dB above the 40th percentile of an
   ~90 Hz neighbourhood (a low percentile, because lines come in clusters);
3. a peak is a *line* when a peak recurs within +-2 bins in >= 60 % of the frames of a 2 s
   window.

Each line is notched +-2 bins / +-1 frame down to the neighbourhood level (at most -20 dB).
"""
from __future__ import annotations

import numpy as np
from scipy import ndimage

from ..constants import FS
from .spectral import istft, stft

N_FFT = 32768
HOP = 4096


def remove_tonal_lines(x: np.ndarray, lo: float = 150.0, hi: float = 4000.0, prom_db: float = 6.0,
                       min_s: float = 2.0, share: float = 0.6, drift_bins: int = 2, ctx_pct: float = 40,
                       smooth_s: float = 0.7) -> tuple[np.ndarray, float]:
    """Pull persistent narrow lines in an (n, C) signal down to the neighbouring level.

    Returns the processed signal and the share of lo-hi time-frequency cells that were notched.
    """
    f, X = stft(x, N_FFT, HOP)
    rows = np.where((f >= lo - 60) & (f <= hi + 60))[0]
    Xb = X[:, rows]
    P = np.mean(np.abs(Xb) ** 2, axis=0)
    P = ndimage.uniform_filter1d(P, max(1, int(smooth_s * FS / HOP)), axis=1, mode="nearest")
    Pd = 10 * np.log10(P + 1e-30)
    ctx = ndimage.percentile_filter(Pd, ctx_pct, size=(61, 1), mode="nearest")
    band = ((f[rows] >= lo) & (f[rows] <= hi))[:, None]
    peak = band & (Pd - ctx >= prom_db) & (Pd >= np.roll(Pd, 1, 0)) & (Pd >= np.roll(Pd, -1, 0))
    near = ndimage.binary_dilation(peak, structure=np.ones((2 * drift_bins + 1, 1), bool))
    win = max(3, int(min_s * FS / HOP))
    line = peak & (ndimage.uniform_filter1d(near.astype(float), win, axis=1, mode="nearest") >= share)
    mask = ndimage.binary_dilation(line, structure=np.ones((5, 3), bool))
    G = np.where(mask, np.minimum(1.0, np.sqrt(10 ** ((ctx - Pd) / 10))), 1.0)
    X[:, rows] = Xb * np.maximum(G, 0.1)[None]
    return istft(X, len(x), N_FFT, HOP), float(mask[band[:, 0]].mean())


def remove_tonal_lines_v3(x: np.ndarray, lo: float = 150.0, hi: float = 4000.0, prom_db: float = 6.0,
                          min_s: float = 10.0, share: float = 0.85, drift_bins: int = 6, ctx_pct: float = 40,
                          smooth_s: float = 0.35) -> tuple[np.ndarray, float]:
    """Horn-safe version (profile carnoise_v3).

    Same notch as :func:`remove_tonal_lines`, but a line has to be present in at least 85 % of a
    10 s window centred on the frame (v1: 60 % of 2 s), with +-6 bins (+-9 Hz) of drift allowed,
    and the spectrum is only smoothed over 0.35 s (v1: 0.7 s). Fans and alternators hum for minutes
    and still qualify; horns and alarms lasting a few seconds don't.

    On synthetic tests: horns of 0.5-4 s lose at most 0.6 dB (v1: 11-13 dB), while steady and slowly
    drifting (+-3 Hz) fan lines are still cut by about 16.5 dB, as in v1.
    """
    f, X = stft(x, N_FFT, HOP)
    rows = np.where((f >= lo - 60) & (f <= hi + 60))[0]
    Xb = X[:, rows]
    P = np.mean(np.abs(Xb) ** 2, axis=0)
    P = ndimage.uniform_filter1d(P, max(1, int(smooth_s * FS / HOP)), axis=1, mode="nearest")
    Pd = 10 * np.log10(P + 1e-30)
    ctx = ndimage.percentile_filter(Pd, ctx_pct, size=(61, 1), mode="nearest")
    band = ((f[rows] >= lo) & (f[rows] <= hi))[:, None]
    peak = band & (Pd - ctx >= prom_db) & (Pd >= np.roll(Pd, 1, 0)) & (Pd >= np.roll(Pd, -1, 0))
    near = ndimage.binary_dilation(peak, structure=np.ones((2 * drift_bins + 1, 1), bool))
    win = max(3, int(min_s * FS / HOP))
    persistence = ndimage.uniform_filter1d(near.astype(float), win, axis=1, mode="constant")
    line = peak & (persistence >= share)
    mask = ndimage.binary_dilation(line, structure=np.ones((5, 3), bool))
    G = np.where(mask, np.minimum(1.0, np.sqrt(10 ** ((ctx - Pd) / 10))), 1.0)
    X[:, rows] = Xb * np.maximum(G, 0.1)[None]
    return istft(X, len(x), N_FFT, HOP), float(mask[band[:, 0]].mean())
