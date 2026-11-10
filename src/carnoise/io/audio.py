"""Audio decoding with FFmpeg into a memory-mapped float32 file.

FFmpeg is used rather than soundfile for input because it decodes every FLAC/WAV variant in
the dataset (including 32-bit FLAC) consistently. The decoded audio goes to local scratch
space and is read through ``np.memmap``, so a long recording is never fully loaded into RAM.
"""
from __future__ import annotations

import json
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np


@dataclass(frozen=True)
class AudioMeta:
    sample_rate: int
    channels: int


def require_ffmpeg() -> None:
    if not (shutil.which("ffmpeg") and shutil.which("ffprobe")):
        raise RuntimeError("FFmpeg/FFprobe required (Colab/Debian: apt-get -qq install ffmpeg; macOS: brew install ffmpeg)")


def probe(path: Path) -> AudioMeta:
    """Sample rate and channel count of the first audio stream (mono or stereo only)."""
    cmd = ["ffprobe", "-v", "error", "-select_streams", "a:0",
           "-show_entries", "stream=sample_rate,channels", "-of", "json", str(path)]
    out = subprocess.run(cmd, capture_output=True, text=True, check=True, timeout=60)
    streams = json.loads(out.stdout).get("streams", [])
    if not streams:
        raise ValueError("No audio stream")
    sr, ch = int(streams[0]["sample_rate"]), int(streams[0]["channels"])
    if sr < 1000 or ch not in (1, 2):
        raise ValueError(f"Unsupported sample rate/channels: {sr}/{ch}")
    return AudioMeta(sr, ch)


def decode_to_raw(path: Path, raw_path: Path, sample_rate: Optional[int] = None) -> None:
    """Decode to raw little-endian float32, keeping the channel count.

    ``sample_rate``: resample to this rate (FFmpeg's high-quality SoX resampler when available),
    or None to keep the source rate.
    """
    cmd = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-i", str(path), "-map", "0:a:0", "-vn"]
    if sample_rate:
        cmd += ["-af", f"aresample={sample_rate}:resampler=soxr:precision=28"]
    cmd += ["-f", "f32le", "-acodec", "pcm_f32le", str(raw_path)]
    p = subprocess.run(cmd, capture_output=True, text=True, timeout=900)
    if p.returncode and sample_rate and "soxr" in p.stderr:
        # FFmpeg built without libsoxr: fall back to its own resampler
        cmd[cmd.index("-af") + 1] = f"aresample={sample_rate}"
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=900)
    if p.returncode:
        raise RuntimeError("FFmpeg decode: " + p.stderr[-500:])


def open_raw(raw_path: Path, channels: int) -> np.memmap:
    """Memory-map a raw float32 file as (frames, channels)."""
    raw_bytes = raw_path.stat().st_size
    if raw_bytes == 0 or raw_bytes % (4 * channels) != 0:
        raise ValueError(f"Invalid decoded float32 size: {raw_bytes} bytes for {channels} channels")
    return np.memmap(raw_path, dtype="<f4", mode="r", shape=(raw_bytes // (4 * channels), channels))


def read_excerpt(path: Path, seconds: float, start: float = 0.0, channels: Optional[int] = None) -> np.ndarray:
    """Decode a short excerpt (for listening/QA) without decoding the whole file."""
    ch = channels or probe(path).channels
    cmd = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-ss", str(start), "-i", str(path),
           "-t", str(seconds), "-map", "0:a:0", "-f", "f32le", "-acodec", "pcm_f32le", "-"]
    raw = subprocess.check_output(cmd, timeout=90)
    return np.frombuffer(raw, dtype="<f4").reshape(-1, ch)
