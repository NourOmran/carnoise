"""GPS speed track.

Source: the dataset's per-unit GPS log ``<data_root>/gps/<unit>_gps.csv``, with columns
``audio_file``, ``time_utc``, ``speed_kmh`` and ``fix_quality``. Rows for a recording are put into
one-second slots counted from the recording's start time (``start_utc`` in files.csv), and the
median of each slot is used.

Speed only sets how strong the Wiener stage is. Without it, the lightest (stopped) setting is used.
"""
from __future__ import annotations

import csv
from pathlib import Path
from datetime import datetime, timezone
from typing import Optional

import numpy as np

from .constants import FS, HOP


def _utc(text: str) -> datetime:
    text = text.strip().rstrip("Z")
    fmt = "%Y-%m-%dT%H:%M:%S.%f" if "." in text else "%Y-%m-%dT%H:%M:%S"
    return datetime.strptime(text, fmt).replace(tzinfo=timezone.utc)


def find_gps_csv(data_root: Path, unit: str) -> Optional[Path]:
    """The unit's GPS log in the dataset, or None."""
    p = Path(data_root) / "gps" / f"{unit}_gps.csv"
    return p if p.is_file() else None


def speed_from_gps(gps_csv: Optional[Path], audio_name: str, start_utc: str, seconds: float) -> np.ndarray:
    """Per-second speed (km/h, NaN where unknown) for one recording from a unit GPS log.

    Only rows with a fix (``fix_quality`` > 0) and a speed count. Empty if the log has none for this file.
    """
    if gps_csv is None or not start_utc:
        return np.zeros(0)
    n = int(np.ceil(seconds))
    t0 = _utc(start_utc)
    slots: dict[int, list[float]] = {}
    with open(gps_csv) as f:
        for r in csv.DictReader(f):
            if Path(r.get("audio_file", "")).name != audio_name or not r.get("speed_kmh"):
                continue
            if float(r.get("fix_quality") or 0) <= 0:
                continue
            k = int(np.floor((_utc(r["time_utc"]) - t0).total_seconds()))
            if 0 <= k < n:
                slots.setdefault(k, []).append(float(r["speed_kmh"]))
    if not slots:
        return np.zeros(0)
    v = np.full(n, np.nan)
    for k, vals in slots.items():
        v[k] = float(np.median(vals))
    return v


SpeedSource = Optional[np.ndarray]


def speed_per_frame(source: SpeedSource, n_frames: int, t0: float = 0.0) -> np.ndarray:
    """Speed at each main-STFT frame. Gaps are interpolated; all zeros when unknown.

    ``source`` is a per-second speed array (km/h, NaN for gaps) or None.
    ``t0`` is the time (s) of the first frame, for chunked processing.
    """
    if source is None:
        return np.zeros(n_frames)
    v = np.asarray(source, dtype=float)
    if len(v) == 0 or np.all(np.isnan(v)):
        return np.zeros(n_frames)
    idx = np.arange(len(v))
    ok = ~np.isnan(v)
    v = np.interp(idx, idx[ok], v[ok])
    t = t0 + np.arange(n_frames) * HOP / FS
    return np.interp(t, idx + 0.5, v)
