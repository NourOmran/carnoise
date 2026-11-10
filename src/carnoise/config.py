"""User settings: DSP strength, chunking, and the batch run (paths, selection, output).

Settings are plain dataclasses so they can be built in code, or loaded from a TOML file
with :func:`load_config` (see ``config/colab.toml`` for a documented example).
"""
from __future__ import annotations

import sys
from dataclasses import dataclass, field, fields, replace
from pathlib import Path
from typing import Any, Optional, Union

if sys.version_info >= (3, 11):
    import tomllib
else:  # pragma: no cover
    import tomli as tomllib

HzSetting = Union[str, float]

#: Text stored in run_config.json; identical to the Colab notebook so its output folders resume.
PROCESSING_DESCRIPTION = (
    "FFmpeg float32 decode; car-noise DSP (declip, engine orders, auto HPF, tonal lines, "
    "auto LPF, click repair, min-statistics Wiener); preserve channels"
)


def _hz(value: HzSetting, name: str, allow_none: bool) -> HzSetting:
    if isinstance(value, str):
        v = value.strip().lower()
        if v == "auto" or (allow_none and v == "none"):
            return v
        try:
            return float(v)
        except ValueError:
            pass
        raise ValueError(f"{name} must be 'auto'{', none' if allow_none else ''} or a number, got {value!r}")
    return float(value)


@dataclass(frozen=True)
class DSPConfig:
    """How hard the denoiser works. Defaults are the tuned production values."""

    #: 'auto' (70-200 Hz, chosen per file) or a corner frequency in Hz.
    high_pass_hz: HzSetting = "auto"
    #: 'auto' (chosen per file), a corner frequency in Hz, or 'none'.
    low_pass_hz: HzSetting = "auto"
    #: Remove persistent narrow tones (fans, alternator, electronics).
    remove_tonal_lines: bool = True
    #: Click threshold vs the local median; higher = fewer repairs.
    click_k: float = 5.0
    #: dB added to the Wiener gain floor: negative = stronger removal, positive = more natural.
    floor_shift_db: float = 0.0

    def __post_init__(self):
        object.__setattr__(self, "high_pass_hz", _hz(self.high_pass_hz, "high_pass_hz", False))
        object.__setattr__(self, "low_pass_hz", _hz(self.low_pass_hz, "low_pass_hz", True))
        if self.click_k <= 1:
            raise ValueError("click_k must be > 1")


@dataclass(frozen=True)
class ChunkConfig:
    """How long files are split. Chunk and context sizes are rounded to the DSP grid."""

    #: Audio written per chunk (rounded down to the grid, ~44.4 s at 48 kHz).
    block_seconds: float = 45.0
    #: Extra audio processed on each side of a chunk (rounded up, ~6.8 s); noise tracking needs >= 5 s.
    context_seconds: float = 6.0

    def __post_init__(self):
        if not (self.context_seconds >= 5.0 and self.block_seconds >= self.context_seconds * 2):
            raise ValueError("need context_seconds >= 5 and block_seconds >= 2 * context_seconds")


@dataclass(frozen=True)
class BatchConfig:
    """A full-dataset run over ``<data_root>/files.csv``."""

    data_root: Path
    out_root: Path
    tmp_root: Path = Path("/tmp/carnoise_work")
    profile: str = "carnoise_v1"
    include_driving: bool = True
    include_office: bool = True
    #: None = every file; a number limits the run (smoke tests).
    max_files: Optional[int] = None
    #: soundfile subtype of the output FLAC.
    output_subtype: str = "PCM_24"
    #: 'resample' (convert to 48 kHz, output at 48 kHz) or 'error' (log and skip).
    other_sample_rates: str = "resample"
    resume: bool = True
    checkpoint_every: int = 5
    dsp: DSPConfig = field(default_factory=DSPConfig)
    chunking: ChunkConfig = field(default_factory=ChunkConfig)

    def __post_init__(self):
        for name in ("data_root", "out_root", "tmp_root"):
            object.__setattr__(self, name, Path(getattr(self, name)).expanduser())
        if not (self.include_driving or self.include_office):
            raise ValueError("select at least one of include_driving / include_office")
        if not self.profile or "/" in self.profile:
            raise ValueError("profile must be a non-empty name without '/'")
        if self.other_sample_rates not in ("resample", "error"):
            raise ValueError("other_sample_rates must be 'resample' or 'error'")
        if self.max_files is not None and self.max_files <= 0:
            object.__setattr__(self, "max_files", None)

    @property
    def groups(self) -> list[str]:
        return (["driving"] if self.include_driving else []) + (["office"] if self.include_office else [])

    def run_lock(self) -> dict[str, Any]:
        """Settings that must not change while an output folder is being filled (run_config.json).

        Same keys and values as the Colab notebook, so a notebook run can be resumed here.
        """
        d = self.dsp
        return {
            "include_driving": self.include_driving, "include_office": self.include_office,
            "full_recordings": True, "profile": self.profile,
            "high_pass_hz": d.high_pass_hz, "low_pass_hz": d.low_pass_hz,
            "remove_tonal_lines": d.remove_tonal_lines, "click_k": d.click_k,
            "floor_shift_db": d.floor_shift_db, "block_seconds": self.chunking.block_seconds,
            "context_seconds": self.chunking.context_seconds,
            "output_subtype": self.output_subtype,
            "processing": PROCESSING_DESCRIPTION,
        }


def _pick(cls, table: dict) -> dict:
    names = {f.name for f in fields(cls)}
    unknown = set(table) - names
    if unknown:
        raise ValueError(f"unknown {cls.__name__} settings: {', '.join(sorted(unknown))}")
    return dict(table)


def load_config(path: Union[str, Path], **overrides: Any) -> BatchConfig:
    """Load a :class:`BatchConfig` from TOML.

    Tables: ``[paths]``, ``[selection]``, ``[dsp]``, ``[chunking]``, ``[output]``, ``[run]``.
    Keyword overrides replace top-level BatchConfig fields (e.g. ``max_files=3``).
    """
    raw = tomllib.loads(Path(path).read_text())
    known = {"paths", "selection", "dsp", "chunking", "output", "run"}
    unknown = set(raw) - known
    if unknown:
        raise ValueError(f"unknown config tables: {', '.join(sorted(unknown))}")
    top: dict[str, Any] = {}
    for table in ("paths", "selection", "output", "run"):
        top.update(raw.get(table, {}))
    dsp_table = dict(raw.get("dsp", {}))
    if "profile" in dsp_table:          # profile is written under [dsp] for readability
        top["profile"] = dsp_table.pop("profile")
    if "subtype" in top:
        top["output_subtype"] = top.pop("subtype")
    if top.get("max_files") == 0:
        top["max_files"] = None
    cfg = BatchConfig(
        **_pick(BatchConfig, top),
        dsp=DSPConfig(**_pick(DSPConfig, dsp_table)),
        chunking=ChunkConfig(**_pick(ChunkConfig, raw.get("chunking", {}))),
    )
    overrides = {k: v for k, v in overrides.items() if v is not None}
    return replace(cfg, **overrides) if overrides else cfg
