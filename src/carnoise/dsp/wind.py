"""Stage 3w (profile carnoise_v3) - wind on the microphones, for two real microphones.

Wind is turbulence right at each microphone. That makes it different from street sound in two
ways: it's incoherent between the two mics, and it's often much louder at one mic than the other.
Sound from the street reaches both mics in step and at a steady level ratio.

Per frequency bin and STFT frame, below ``max_hz``:

* min-selection: each mic is pulled down to the other mic's level, after allowing for their normal
  level ratio (a running median over ~5 s, so short gusts don't shift it and chunked processing gives
  the same result as whole-file processing). This removes one-sided gusts;
* coherence gain: the square root of the short-time coherence, which removes wind hitting both mics.

The stage only switches on where wind is actually present, meaning the power-weighted 40-400 Hz
coherence (smoothed over ~0.5 s) is below ``gate_coh``. Elsewhere the gain is exactly 1. Duplicate-
channel and mono recordings skip this stage.
"""
from __future__ import annotations

import numpy as np
from scipy import ndimage, signal

MAX_HZ = 600.0
TAPER_HZ = 400.0
MIN_GAIN_DB = -12.0
GATE_COHERENCE = 0.6
RATIO_FRAMES = 469           # running-median window for the mics' normal level ratio (~5 s)


def wind_gain(X: np.ndarray, f: np.ndarray, max_hz: float = MAX_HZ, taper_hz: float = TAPER_HZ,
              alpha: float = 0.85, gmin_db: float = MIN_GAIN_DB, gate_coh: float = GATE_COHERENCE) -> np.ndarray:
    """Per-channel gains (2, bins, frames) for a (2, bins, frames) STFT."""
    A, B = X[0], X[1]
    sel = f < max_hz + taper_hz
    Pa, Pb = np.abs(A[sel]) ** 2, np.abs(B[sel]) ** 2
    Sab = A[sel] * np.conj(B[sel])

    def smooth(z):
        return ndimage.uniform_filter1d(signal.lfilter([1 - alpha], [1, -alpha], z, axis=1), 5, axis=0)

    sa, sb = smooth(Pa), smooth(Pb)
    sab = smooth(Sab.real) + 1j * smooth(Sab.imag)
    coh = np.clip(np.abs(sab) ** 2 / np.maximum(sa * sb, 1e-30), 0, 1)
    log_ratio = np.log(np.maximum(sa, 1e-30) / np.maximum(sb, 1e-30))
    r = np.exp(ndimage.median_filter(log_ratio, size=(1, RATIO_FRAMES), mode="nearest"))  # normal ratio a/b
    ga = np.sqrt(np.minimum(sa, r * sb) / np.maximum(sa, 1e-30))
    gb = np.sqrt(np.minimum(sb, sa / r) / np.maximum(sb, 1e-30))
    gc = np.sqrt(coh)
    gmin = 10 ** (gmin_db / 20)
    w = np.clip((max_hz + taper_hz - f[sel]) / taper_hz, 0, 1)[:, None]
    b = (f[sel] >= 40) & (f[sel] < 400)
    fc = np.sum(coh[b] * sa[b], 0) / np.maximum(np.sum(sa[b], 0), 1e-30)
    fc = ndimage.uniform_filter1d(fc, 47, mode="nearest")
    w = w * np.clip((gate_coh + 0.1 - fc) / 0.2, 0, 1)[None, :]           # soft switch around gate_coh
    G = np.ones((2,) + X.shape[1:])
    G[0, sel] = 1 - w * (1 - np.maximum(ga * gc, gmin))
    G[1, sel] = 1 - w * (1 - np.maximum(gb * gc, gmin))
    return G
