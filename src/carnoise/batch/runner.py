"""Full-dataset run: every selected recording, sequentially, resumable after interruption."""
from __future__ import annotations

import logging
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from ..config import BatchConfig
from ..io.audio import require_ffmpeg
from ..io.storage import is_complete
from .logs import RunLogs, lock_run_config
from .manifest import build_manifest, safe_source
from ..speed import find_gps_csv
from .processor import FileOutcome, process_file

log = logging.getLogger("carnoise")


@dataclass
class RunStats:
    processed: int = 0
    skipped_existing: int = 0
    errors: int = 0


def metrics_row(row: dict, profile: str, out: FileOutcome) -> dict:
    s = out.stream
    return {"source_file": str(row["file"]), "group": str(row["group"]), "unit": str(row["unit"]),
            "profile": profile, "duration_seconds": round(out.duration_seconds, 3),
            "sample_rate_hz": out.sample_rate, "channels": s.channels,
            "input_rms_dbfs": s.input_rms_dbfs, "enhanced_rms_dbfs": s.output_rms_dbfs,
            "rms_attenuation_db": s.input_rms_dbfs - s.output_rms_dbfs,
            "processed_peak_abs_before_clipping": s.peak, "samples_clipped_to_pcm": s.clipped,
            "output_bytes": out.dest.stat().st_size, "output_path": str(out.dest), "status": "processed"}


def dsp_row(row: dict, cfg: BatchConfig, out: FileOutcome) -> dict:
    d, s = cfg.dsp, out.stream
    hp = s.decisions.hp if d.high_pass_hz == "auto" else float(d.high_pass_hz)
    lp = s.decisions.lp if d.low_pass_hz == "auto" else d.low_pass_hz
    return {"source_file": str(row["file"]), "unit": str(row["unit"]), "profile": cfg.profile,
            "duplicate_channels": s.decisions.mono, "high_pass_hz": round(hp, 1),
            "low_pass_hz": lp or "none", "clip_runs_repaired": s.decisions.clip_runs,
            "engine_tracked_share": round(s.engine_tracked_share, 3),
            "tonal_cells_share": round(s.tonal_cells_share, 4),
            "speed_file": str(out.speed_csv) if out.speed_csv else "", "chunks": s.chunks,
            "source_sample_rate_hz": out.source_sample_rate}


class BatchRunner:
    """Process every recording in the manifest into ``cfg.out_root``.

    Typical use::

        runner = BatchRunner(load_config("config/colab.toml"))
        runner.prepare()          # lock settings, build + save manifest
        stats = runner.run()      # process (skips finished outputs when resume is on)
    """

    def __init__(self, cfg: BatchConfig):
        self.cfg = cfg
        self.manifest: pd.DataFrame | None = None
        self.logs: RunLogs | None = None

    def prepare(self) -> pd.DataFrame:
        cfg = self.cfg
        require_ffmpeg()
        cfg.out_root.mkdir(parents=True, exist_ok=True)
        cfg.tmp_root.mkdir(parents=True, exist_ok=True)
        lock_run_config(cfg.out_root, cfg.run_lock())
        self.manifest = build_manifest(cfg)
        self.logs = RunLogs(cfg.out_root, cfg.profile)
        self.logs.write_manifest(self.manifest)
        return self.manifest

    def run(self) -> RunStats:
        if self.manifest is None:
            self.prepare()
        cfg, logs, manifest = self.cfg, self.logs, self.manifest
        stats = RunStats()
        total = len(manifest)
        for index, row in enumerate(manifest.to_dict("records"), start=1):
            src_key = str(row["file"])
            dest = Path(row["output_path"])
            if cfg.resume and is_complete(dest):
                stats.skipped_existing += 1
                logs.existing(src_key, row, dest)
            else:
                started = time.perf_counter()
                try:
                    with tempfile.TemporaryDirectory(prefix="ensense_", dir=str(cfg.tmp_root)) as scratch:
                        gps = find_gps_csv(cfg.data_root, str(row["unit"]))
                        start = str(row.get("start_utc", "") or "")
                        out = process_file(safe_source(cfg.data_root, src_key), dest, Path(scratch), cfg.dsp,
                                           cfg.chunking, cfg.output_subtype, cfg.other_sample_rates,
                                           gps_csv=gps, start_utc=start if start != "nan" else "")
                    logs.success(metrics_row(row, cfg.profile, out), dsp_row(row, cfg, out))
                    stats.processed += 1
                    log.info(f"[{index}/{total}] OK   {row['group']}/{row['unit']} {Path(src_key).name} "
                             f"({out.duration_seconds:.1f}s audio; {time.perf_counter() - started:.1f}s processing)")
                except Exception as e:  # one bad file must not stop the run
                    stats.errors += 1
                    logs.error(src_key, str(row["unit"]), e)
                    log.warning(f"[{index}/{total}] ERROR {row['unit']} {Path(src_key).name}: {e}")
            if index % cfg.checkpoint_every == 0:
                logs.flush()
        logs.flush()
        done = sum(is_complete(Path(p)) for p in manifest.output_path)
        log.info(f"RUN SUMMARY: {stats.__dict__}; outputs present {done}/{total}; output folder {cfg.out_root}")
        return stats
