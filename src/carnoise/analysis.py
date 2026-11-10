"""Per-file decisions, made once per recording before it is denoised in chunks.

Some choices (mono vs stereo, high-pass and low-pass corners) must be the same for every
chunk of a file, otherwise the sound would change at chunk boundaries. :func:`file_stats`
makes them in one streaming pass, so files of any length can be analysed in fixed memory.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional

import numpy as np
from scipy import signal

from .constants import CLIP_LEVEL, DUP_RATIO, FS
from .dsp.filters import highpass_from_psd, level_spectrogram, lowpass_from_spec, to_half_second_frames


@dataclass(frozen=True)
class FileDecisions:
    mono: bool               #: second channel is a copy of the first
    hp: float                #: automatic high-pass corner (Hz)
    lp: Optional[float]      #: automatic low-pass corner (Hz), None = keep the full band
    clip_runs: int           #: clipped runs in the file (all channels)


def file_stats(blocks: Iterable[np.ndarray]) -> FileDecisions:
    """Stream over a file's consecutive (n, C) blocks and return its :class:`FileDecisions`.

    Each block except the last must be a multiple of 24576 samples
    (see :data:`carnoise.constants.STATS_BLOCK`).
    """
    psd, nseg, spec = 0.0, 0, []
    dsum, xsum, runs, prev = 0.0, 0.0, 0, None
    f = fs_ = None
    for b in blocks:
        b = b[:, None] if b.ndim == 1 else b
        if b.shape[1] == 2:
            dsum += float(np.sum((b[:, 0] - b[:, 1]) ** 2))
            xsum += float(np.sum(b[:, 0] ** 2))
        cl = np.abs(b) >= CLIP_LEVEL
        edge = cl[:1] & prev if prev is not None else np.zeros((1, b.shape[1]), bool)
        runs += int(np.sum(np.diff(cl.astype(np.int8), axis=0) == 1) + np.sum(cl[:1] & ~edge))
        prev = cl[-1:]
        m = b.mean(axis=1)
        if len(m) >= 8192:
            f, P = signal.welch(m, FS, nperseg=8192)
            k = len(m) // 4096 - 1   # number of half-overlapping segments
            psd, nseg = psd + P * k, nseg + k
        if len(m) >= 4096:
            fs_, S = level_spectrogram(m)
            spec.append(S)
    mono = xsum > 0 and dsum / xsum < DUP_RATIO
    hp = highpass_from_psd(f, psd / nseg) if nseg else 70.0
    lp = None
    if spec:
        S = to_half_second_frames(fs_, np.concatenate(spec, axis=1))
        lp = lowpass_from_spec(fs_, S) if S.shape[1] >= 4 else None
    return FileDecisions(mono=bool(mono), hp=float(hp), lp=lp, clip_runs=runs)
