"""GPS speed track.

Each recording may have a per-second speed CSV (column ``speed_kmh``) next to it
(``<stem>_speed.csv``) or in a sibling ``speed/`` folder. Speed only sets how strong the
Wiener stage is; without it the lightest (stopped) setting is used.
"""
from __future__ import annotations

import csv
from pathlib import Path
from typing import Optional

import numpy as np

from .constants import FS, HOP


def find_speed_csv(audio_path: Path) -> Optional[Path]:
    """The speed CSV belonging to an audio file, or None."""
    for p in (audio_path.with_name(f"{audio_path.stem}_speed.csv"),
              audio_path.parent.parent / "speed" / f"{audio_path.stem}_speed.csv"):
        if p.is_file():
            return p
    return None


def read_speed_kmh(path: Optional[Path]) -> np.ndarray:
    """Per-second speed in km/h (NaN for blank rows); empty if there is no file."""
    if path is None or not Path(path).exists():
        return np.zeros(0)
    with open(path) as f:
        return np.array([float(r["speed_kmh"]) if r["speed_kmh"] else np.nan for r in csv.DictReader(f)],
                        dtype=float)


def speed_per_frame(path: Optional[Path], n_frames: int, t0: float = 0.0) -> np.ndarray:
    """Speed at each main-STFT frame. Gaps are interpolated; all zeros when unknown.

    ``t0`` is the time (s) of the first frame, for chunked processing.
    """
    v = read_speed_kmh(path)
    if len(v) == 0 or np.all(np.isnan(v)):
        return np.zeros(n_frames)
    idx = np.arange(len(v))
    ok = ~np.isnan(v)
    v = np.interp(idx, idx[ok], v[ok])
    t = t0 + np.arange(n_frames) * HOP / FS
    return np.interp(t, idx + 0.5, v)
