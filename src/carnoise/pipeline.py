"""The denoiser: stage order for one block of audio, and chunked streaming for whole files.

* :func:`denoise` runs all DSP stages on an in-memory (n, C) array.
* :func:`denoise_stream` runs a recording of any length chunk by chunk, with context on
  each side of every chunk, and hands the finished samples to a writer. It never holds more
  than one chunk (+ context) in memory and needs no files or ffmpeg, so it can be tested
  and reused on its own; :mod:`carnoise.batch.processor` connects it to decoding and FLAC output.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Iterator, Optional

import numpy as np

from .analysis import FileDecisions, file_stats
from .config import ChunkConfig, DSPConfig
from .constants import FS, GRID, STATS_BLOCK
from .dsp.channels import as_channels, select_channels
from .dsp.declip import declip_channels
from .dsp.engine import remove_engine_orders
from .dsp.filters import auto_highpass, auto_lowpass, highpass, lowpass
from .dsp.noise import FLOOR_DB, OVERSUB, noise_psd, wiener_gain
from .dsp.spectral import istft, stft
from .dsp.tonal import remove_tonal_lines
from .dsp.wind import wind_gain
from .speed import SpeedSource, speed_per_frame


@dataclass(frozen=True)
class DenoiseInfo:
    """What :func:`denoise` did to one block."""
    channels: int            #: channels processed (1 for a duplicate-channel file)
    clip_runs: int           #: clipped runs repaired
    hp: float                #: high-pass corner used (Hz)
    lp: Optional[float]      #: low-pass corner used (Hz), None = none
    tones: float             #: share of 150-4000 Hz time-frequency cells notched as tonal lines
    speed_mean: float        #: mean GPS speed (km/h)
    engine_tracked: float    #: share of frames where the engine was found


def _resolve_hp(setting, auto_value: Callable[[], float]) -> float:
    return auto_value() if setting == "auto" else float(setting)


def _resolve_lp(setting, auto_value: Callable[[], Optional[float]]) -> Optional[float]:
    if setting == "auto":
        return auto_value()
    return None if setting in ("none", None) else float(setting)


def denoise(x: np.ndarray, cfg: DSPConfig = DSPConfig(), speed: SpeedSource = None,
            decisions: Optional[FileDecisions] = None, t0: float = 0.0) -> tuple[np.ndarray, DenoiseInfo]:
    """Remove car noise from an (n,) or (n, C) float array at 48 kHz.

    Args:
        x: audio, float, full scale +-1.
        cfg: DSP settings.
        speed: GPS speed per second of the file (km/h, NaN for gaps), or None for unknown.
        decisions: per-file decisions from :func:`carnoise.analysis.file_stats`. Pass them when
            ``x`` is one chunk of a longer file; without them they are computed from ``x``.
        t0: start time of ``x`` within its file (s), to line up the speed track.

    Returns:
        ``(y, info)``. ``y`` is (n, C') where C' is 1 for a duplicate-channel input.
    """
    x = as_channels(x) if decisions is None else select_channels(x, decisions.mono)
    x, runs = declip_channels(x)                                        # 0 declip
    y, engine_share = x, 0.0
    if cfg.engine_notch:
        y, f0 = remove_engine_orders(x)                                 # 1 engine (optional)
        engine_share = float(np.mean(~np.isnan(f0)))
    hp = _resolve_hp(cfg.high_pass_hz, (lambda: decisions.hp) if decisions else (lambda: auto_highpass(x)))
    y = highpass(y, hp)                                                 # 2 high-pass
    tone_frac = 0.0
    if cfg.remove_tonal_lines:
        y, tone_frac = remove_tonal_lines(y)                            # 2b tonal lines (horn-safe)
    lp = _resolve_lp(cfg.low_pass_hz, (lambda: decisions.lp) if decisions else (lambda: auto_lowpass(x)))
    if lp:
        y = lowpass(y, lp)                                              # 2c low-pass
    f, X = stft(y)
    if X.shape[0] == 2:
        X = X * wind_gain(X, f)                                         # 3 wind (two real mics only)
    speed_frames = speed_per_frame(speed, X.shape[2], t0)
    P = np.mean(np.abs(X) ** 2, axis=0)
    floor = (FLOOR_DB[0] + cfg.floor_shift_db, FLOOR_DB[1] + cfg.floor_shift_db)
    G = wiener_gain(P, noise_psd(P), speed_frames, floor_db=floor, beta=OVERSUB)   # 4 noise
    out = istft(X * G[None], len(x))
    return out, DenoiseInfo(channels=x.shape[1], clip_runs=runs, hp=hp, lp=lp, tones=tone_frac,
                            speed_mean=float(speed_frames.mean()), engine_tracked=engine_share)


# ---- chunked streaming --------------------------------------------------------------------

@dataclass(frozen=True)
class ChunkPlan:
    """Chunk layout on the DSP grid, so STFT frames near a boundary see the same samples
    in both neighbouring chunks."""
    chunk_frames: int
    pad: int

    @classmethod
    def for_rate(cls, sr: int, chunking: ChunkConfig) -> "ChunkPlan":
        chunk = max(GRID, int(sr * chunking.block_seconds) // GRID * GRID)
        pad = -(-int(sr * chunking.context_seconds) // GRID) * GRID
        return cls(chunk, pad)

    def chunks(self, frames: int) -> Iterator[tuple[int, int, int, int]]:
        """Yield ``(start, stop, left, right)``: write [start, stop), process [left, right)."""
        for start in range(0, frames, self.chunk_frames):
            stop = min(start + self.chunk_frames, frames)
            yield start, stop, max(0, start - self.pad), min(stop + self.pad, frames)

    def count(self, frames: int) -> int:
        return -(-frames // self.chunk_frames)


@dataclass
class StreamResult:
    """Totals over a streamed file, for the run logs."""
    frames: int
    channels: int
    decisions: FileDecisions
    chunks: int
    peak: float = 0.0                 #: max |output| before PCM clipping
    clipped: int = 0                  #: output samples at or above full scale
    in_sq: float = 0.0
    out_sq: float = 0.0
    infos: list = field(default_factory=list)   #: (samples, DenoiseInfo) per full-DSP chunk

    def _rms_db(self, sq: float) -> float:
        return float(20 * np.log10(max(np.sqrt(sq / max(1, self.frames * self.channels)), 1e-12)))

    @property
    def input_rms_dbfs(self) -> float:
        return self._rms_db(self.in_sq)

    @property
    def output_rms_dbfs(self) -> float:
        return self._rms_db(self.out_sq)

    def _weighted(self, attr: str) -> float:
        w = sum(n for n, _ in self.infos) or 1
        return sum(n * getattr(i, attr) for n, i in self.infos) / w

    @property
    def engine_tracked_share(self) -> float:
        return self._weighted("engine_tracked")

    @property
    def tonal_cells_share(self) -> float:
        return self._weighted("tones")


def _denoise_chunk(x: np.ndarray, cfg: DSPConfig, decisions: FileDecisions, speed: SpeedSource,
                   t0: float) -> tuple[np.ndarray, Optional[DenoiseInfo]]:
    """One chunk with context: float32 (n, C) in, float32 (n, C) out (channel count kept)."""
    xf = np.asarray(x, dtype=np.float64)
    if len(x) < 2 * GRID:
        # too short for the spectral stages: high-pass only (still processed, not dropped)
        hp = _resolve_hp(cfg.high_pass_hz, lambda: decisions.hp)
        y, info = (highpass(xf, hp) if len(x) > 64 else xf.copy()), None
        y = y[:, None] if y.ndim == 1 else y
    else:
        y, info = denoise(xf, cfg, speed, decisions, t0)
    if y.shape[1] != x.shape[1]:
        y = np.repeat(y[:, :1], x.shape[1], axis=1)   # duplicate-channel file: write the copy back
    if y.shape != x.shape or not np.isfinite(y).all():
        raise RuntimeError("Denoising generated invalid audio")
    return np.asarray(y, dtype=np.float32), info


def denoise_stream(src: np.ndarray, sr: int, write: Callable[[np.ndarray], None],
                   cfg: DSPConfig = DSPConfig(), chunking: ChunkConfig = ChunkConfig(),
                   speed: SpeedSource = None) -> StreamResult:
    """Denoise a whole recording chunk by chunk.

    Args:
        src: (frames, C) float32 array-like supporting slicing, e.g. an ``np.memmap``.
        sr: sample rate; must be 48 kHz.
        write: called with each finished (n, C) float32 block, in order, clipped to +-0.999999.
        cfg, chunking: settings.
        speed: GPS speed per second of the recording (km/h, NaN for gaps), or None.
    """
    if sr != FS:
        raise ValueError(f"Car-noise DSP is built for {FS} Hz audio; this file is {sr} Hz")
    frames, ch = src.shape
    if not all(np.isfinite(src[i:i + STATS_BLOCK]).all() for i in range(0, frames, STATS_BLOCK)):
        raise ValueError("Source contains NaN/inf")
    decisions = file_stats(np.asarray(src[i:i + STATS_BLOCK], dtype=np.float64)
                           for i in range(0, frames, STATS_BLOCK))
    plan = ChunkPlan.for_rate(sr, chunking)
    res = StreamResult(frames=frames, channels=ch, decisions=decisions, chunks=plan.count(frames))
    written = 0
    for start, stop, left, right in plan.chunks(frames):
        x = np.asarray(src[left:right], dtype=np.float32).copy()
        y, info = _denoise_chunk(x, cfg, decisions, speed, left / sr)
        if info:
            res.infos.append((stop - start, info))
        original = x[start - left:stop - left]
        middle = y[start - left:stop - left]
        if len(middle) != stop - start:
            raise RuntimeError("Mismatch in chunk length")
        res.peak = max(res.peak, float(np.max(np.abs(middle))))
        res.clipped += int(np.count_nonzero(np.abs(middle) >= 1.0))
        res.in_sq += float(np.sum(original.astype(np.float64) ** 2))
        res.out_sq += float(np.sum(middle.astype(np.float64) ** 2))
        write(np.clip(middle, -0.999999, 0.999999))
        written += len(middle)
    if written != frames:
        raise RuntimeError(f"Output length changed: {written} vs {frames}")
    return res
