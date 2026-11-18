"""Stage 1 (optional, ``engine_notch = true``) - engine harmonics.

There is no RPM signal, so the audio is the tachometer. The engine's firing frequency f0
(25-90 Hz) is found frame by frame from its whole harmonic series, and every harmonic up to
1.5 kHz is pulled down to the level of the spectrum around it.

A fine STFT (8192 samples, ~5.9 Hz bins) is needed because the harmonics are only 25-90 Hz apart.

Off by default: engine lines are only 2-3 dB above the noise in this dataset, and notching them
costs a little overall cleanness (QE evaluation: SI-SDR improvement +13.4 dB -> +11.7 dB) in exchange
for about 11 dB more engine-line removal.
"""
from __future__ import annotations

import numpy as np
from scipy import ndimage

from .spectral import istft, stft

F0_BAND = (25.0, 90.0)
N_FFT = 8192


def track_f0(f: np.ndarray, P: np.ndarray, band=F0_BAND, n_h: int = 10, smooth: int = 4,
             min_contrast_db: float = 2.0, low_hz: float = 250.0, min_low_db: float = 3.0,
             max_jump_hz: float = 2.0, min_run: int = 6, max_hz: float = 1000.0) -> np.ndarray:
    """Engine f0 per frame from the whole harmonic series (NaN where there's no engine).

    On these microphones the fundamental is often weak, and the strongest low peak is often a voice
    or hum rather than the engine. So every candidate f0 (0.25 Hz steps) is scored by
    the mean prominence of its first ``n_h`` harmonics, on a spectrum smoothed over ~0.17 s so steady
    engine lines stand out and fast-moving voice harmonics blur. Then three gates:

    * contrast: the best candidate must beat the median candidate by ``min_contrast_db``;
    * low orders: the median prominence of the first four harmonics below ``low_hz`` must reach
      ``min_low_db``. A voice at 120 Hz only hits every second or third harmonic of a 40/60 Hz
      series, so it can't pass as an engine;
    * continuity: f0 moves at most ``max_jump_hz`` per frame for at least ``min_run`` frames.
    """
    df = f[1]
    top = int(max_hz / df) + 12
    Ps = ndimage.uniform_filter1d(P[:top], smooth, axis=1, mode="nearest") if smooth > 1 else P[:top]
    Ps = np.maximum(Ps, 0.0)               # running sums can dip a hair below 0 on silence
    Pd = 10 * np.log10(Ps + 1e-30)
    ctx = ndimage.median_filter(Pd, size=(15, 1), mode="nearest")
    Rm = ndimage.maximum_filter1d(Pd - ctx, 3, axis=0)
    cand = np.arange(band[0], band[1], 0.25)
    fh = cand[:, None] * np.arange(1, n_h + 1)[None, :]
    idx = np.clip(np.rint(fh / df).astype(int), 0, top - 1)
    S = np.clip(Rm[idx], -3, 20)                                    # (candidate, harmonic, frame)
    valid = (fh <= max_hz)[..., None]
    score = np.sum(S * valid, axis=1) / np.maximum(valid.sum(1), 1)
    low = np.where((fh <= low_hz)[..., None], S, np.nan)[:, :4]
    with np.errstate(all="ignore"):
        lscore = np.nanmedian(low, axis=1)
    best = np.argmax(score, axis=0)
    cols = np.arange(P.shape[1])
    ok = ((score[best, cols] - np.median(score, axis=0)) >= min_contrast_db) & (lscore[best, cols] >= min_low_db)
    f0 = np.where(ok, cand[best], np.nan)
    steady = np.r_[False, np.abs(np.diff(f0)) <= max_jump_hz]
    lab, _ = ndimage.label(steady | np.r_[steady[1:], False])
    keep = np.zeros(len(f0), bool)
    for i, sl in enumerate(ndimage.find_objects(lab), 1):
        if sl[0].stop - sl[0].start >= min_run:
            keep[sl] = lab[sl] == i
    f0 = np.where(keep, f0, np.nan)
    good = ~np.isnan(f0)
    if good.sum() > 3:
        i = np.arange(len(f0))
        f0s = ndimage.median_filter(np.interp(i, i[good], f0[good]), size=5, mode="nearest")
        f0 = np.where(ndimage.binary_dilation(good, iterations=2), f0s, np.nan)
    return f0


def harmonic_gain(f: np.ndarray, P: np.ndarray, f0: np.ndarray, n_h: int = 16, max_hz: float = 1500.0,
                  half: int = 3, max_cut_db: float = 30.0) -> np.ndarray:
    """Notch mask: every harmonic of f0 (including the fundamental) up to ``max_hz``, +-``half`` bins,
    pulled down to its neighbourhood (at most ``max_cut_db``). Where there's no line the gain stays 1."""
    df = f[1]
    nb = P.shape[0]
    G = np.ones_like(P)
    t = np.where(~np.isnan(f0))[0]
    if len(t) == 0:
        return G
    offs = np.arange(-10, 11)
    for h in range(1, n_h + 1):
        fh = h * f0[t]
        ok = fh <= max_hz
        if not ok.any():
            break
        tt, fh = t[ok], fh[ok]
        c = np.rint(fh / df).astype(int)
        idx = np.clip(c[:, None] + offs[None, :], 0, nb - 1)
        inwin = np.abs(idx * df - fh[:, None]) <= (4.0 + 0.03 * fh)[:, None]
        pk = idx[np.arange(len(tt)), np.argmax(np.where(inwin, P[idx, tt[:, None]], -1), axis=1)]
        off = np.clip(pk[:, None] + np.r_[-12:-half - 1, half + 2:13][None, :], 0, nb - 1)
        ref = np.median(P[off, tt[:, None]], axis=1)
        for d in range(-half, half + 1):
            b = np.clip(pk + d, 0, nb - 1)
            g = np.minimum(1.0, np.sqrt(ref / np.maximum(P[b, tt], 1e-30)))
            G[b, tt] = np.minimum(G[b, tt], g)
    G = np.maximum(G, 10 ** (-max_cut_db / 20))
    return ndimage.minimum_filter1d(G, 3, axis=1)                    # cover onset/offset smear


def remove_engine_orders(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Engine stage on an (n, C) signal. Returns the processed signal and the f0 track."""
    n = N_FFT
    hop = n // 4
    f, X = stft(x, n, hop)
    P = np.mean(np.abs(X) ** 2, axis=0)
    f0 = track_f0(f, P)
    G = harmonic_gain(f, P, f0)
    return istft(X * G[None], len(x), n, hop), f0
