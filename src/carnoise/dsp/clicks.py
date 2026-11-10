"""Stage 3 - mount rattle / click repair.

A frame is a click when its 2-16 kHz energy (all channels together) is more than ``k`` times
the median of the surrounding ~0.5 s. Above 800 Hz such frames are clamped, per channel, to
the median magnitude of the non-click frames within +-8 frames (~85 ms). Phase is kept.

A higher ``k`` repairs fewer frames; real street transients (doors, footsteps) are usually
below the default threshold and are left alone.
"""
from __future__ import annotations

import numpy as np
from scipy import ndimage

DETECT_BAND = (2000.0, 16000.0)
REPAIR_FROM_HZ = 800.0
MEDIAN_FRAMES = 51
NEIGHBOUR_FRAMES = 8


def detect_clicks(mag: np.ndarray, f: np.ndarray, k: float) -> np.ndarray:
    """Boolean per-frame click flags from a (C, bins, frames) magnitude array."""
    hb = (f >= DETECT_BAND[0]) & (f < DETECT_BAND[1])
    e = np.sum(mag[:, hb] ** 2, axis=(0, 1))
    local = ndimage.median_filter(e, size=MEDIAN_FRAMES, mode="nearest")
    return ndimage.binary_dilation(e > k * local, iterations=1)


def repair_clicks(X: np.ndarray, f: np.ndarray, k: float = 4.0,
                  lo_hz: float = REPAIR_FROM_HZ) -> tuple[np.ndarray, np.ndarray]:
    """Repair click frames in a (C, bins, frames) STFT. Returns (repaired STFT, click flags)."""
    mag = np.abs(X)
    hit = detect_clicks(mag, f, k)
    sel = f >= lo_hz
    msel = mag[:, sel]          # taken once: copying it per hit made long files 20x slower
    out = mag.copy()
    n = mag.shape[2]
    for j in np.where(hit)[0]:
        nb = [i for i in range(max(0, j - NEIGHBOUR_FRAMES), min(n, j + NEIGHBOUR_FRAMES + 1)) if not hit[i]]
        if nb:
            ref = np.median(msel[:, :, nb], axis=2)
            out[:, sel, j] = np.minimum(msel[:, :, j], ref)
    return out * np.exp(1j * np.angle(X)), hit
