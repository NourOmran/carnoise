"""Run logs in the output folder, in the same formats as the Colab notebook.

=========================  ==================================================================
``run_config.json``        settings lock: a folder is only ever filled with one configuration
``manifest.csv``           selected recordings, whether each source exists, its output path
``processed_metrics.csv``  one row per finished file (levels, peak, clipping, size)
``errors.csv``             one row per failed file (latest error only)
``dsp_details.csv``        per-file DSP decisions (high/low-pass, duplicate channels, ...)
``completion_by_unit.csv`` written by :func:`carnoise.batch.report.completion_by_unit`
=========================  ==================================================================
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

METRICS_COLUMNS = [
    "source_file", "group", "unit", "profile", "duration_seconds",
    "sample_rate_hz", "channels", "input_rms_dbfs", "enhanced_rms_dbfs",
    "rms_attenuation_db", "processed_peak_abs_before_clipping",
    "samples_clipped_to_pcm", "output_bytes", "output_path", "status",
]
ERROR_COLUMNS = ["source_file", "unit", "error"]
DSP_COLUMNS = [
    "source_file", "unit", "profile", "duplicate_channels", "high_pass_hz", "low_pass_hz",
    "clip_runs_repaired", "engine_tracked_share", "tonal_cells_share", "speed_file", "chunks",
    "source_sample_rate_hz",
]


def lock_run_config(out_root: Path, lock: dict[str, Any]) -> None:
    """Write run_config.json, or check that the existing one matches."""
    path = out_root / "run_config.json"
    if path.exists():
        if json.loads(path.read_text()) != lock:
            raise ValueError("Output folder contains a different processing configuration. "
                             "Change out_root/profile for this run or restore the earlier settings.")
    else:
        path.write_text(json.dumps(lock, indent=2))


def _read(path: Path) -> list[dict]:
    if path.exists() and path.stat().st_size:
        return pd.read_csv(path).to_dict("records")
    return []


class RunLogs:
    """In-memory view of the CSV logs, keyed by source file; :meth:`flush` writes them out."""

    def __init__(self, out_root: Path, profile: str):
        self.out_root = out_root
        self.profile = profile
        self.metrics_csv = out_root / "processed_metrics.csv"
        self.errors_csv = out_root / "errors.csv"
        self.dsp_csv = out_root / "dsp_details.csv"
        self.manifest_csv = out_root / "manifest.csv"
        self.metrics = {r["source_file"]: r for r in _read(self.metrics_csv) if r.get("profile") == profile}
        self.errors = {r["source_file"]: r for r in _read(self.errors_csv)}
        self.dsp = {r["source_file"]: r for r in _read(self.dsp_csv) if r.get("profile") == profile}

    def success(self, metrics: dict, dsp: dict) -> None:
        self.metrics[metrics["source_file"]] = metrics
        self.dsp[dsp["source_file"]] = dsp
        self.errors.pop(metrics["source_file"], None)

    def existing(self, src: str, row: dict, dest: Path) -> None:
        """A finished output from an earlier run that has no metrics row (e.g. logs were lost)."""
        if src not in self.metrics:
            self.metrics[src] = {"source_file": src, "group": str(row["group"]), "unit": str(row["unit"]),
                                 "profile": self.profile, "output_path": str(dest),
                                 "output_bytes": dest.stat().st_size,
                                 "status": "existing_file_metrics_unavailable"}

    def error(self, src: str, unit: str, exc: BaseException) -> None:
        self.errors[src] = {"source_file": src, "unit": unit, "error": type(exc).__name__ + ": " + str(exc)[:600]}

    def write_manifest(self, manifest: pd.DataFrame) -> None:
        manifest.to_csv(self.manifest_csv, index=False)

    def flush(self) -> None:
        pd.DataFrame(list(self.metrics.values()), columns=METRICS_COLUMNS).to_csv(self.metrics_csv, index=False)
        pd.DataFrame(list(self.errors.values()), columns=ERROR_COLUMNS).to_csv(self.errors_csv, index=False)
        pd.DataFrame(list(self.dsp.values()), columns=DSP_COLUMNS).to_csv(self.dsp_csv, index=False)
