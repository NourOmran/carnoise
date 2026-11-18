"""Command line: ``carnoise denoise | batch | report``.

    carnoise denoise IN... -o OUT_DIR       local files or folders -> OUT_DIR/<name>.flac
    carnoise batch  -c config/colab.toml    full dataset from files.csv (resumable)
    carnoise report -c config/colab.toml    completion table + failed files
"""
from __future__ import annotations

import argparse
import logging
import sys
import tempfile
from pathlib import Path

from . import __version__, configure_logging
from .batch import BatchRunner, build_manifest, completion_by_unit, failed_files, process_file
from .config import ChunkConfig, DSPConfig, load_config
from .io.audio import require_ffmpeg

AUDIO_EXT = {".wav", ".flac"}
log = logging.getLogger("carnoise")


def _inputs(paths: list[Path]) -> list[Path]:
    files: list[Path] = []
    for p in paths:
        if p.is_dir():
            files += sorted(f for f in p.rglob("*") if f.suffix.lower() in AUDIO_EXT)
        elif p.is_file():
            files.append(p)
        else:
            raise SystemExit(f"not found: {p}")
    return files


def cmd_denoise(a: argparse.Namespace) -> int:
    require_ffmpeg()
    dsp = DSPConfig(high_pass_hz=a.hp, low_pass_hz=a.lp, remove_tonal_lines=not a.no_tones,
                    floor_shift_db=a.floor_shift, engine_notch=a.engine_notch)
    a.out.mkdir(parents=True, exist_ok=True)
    failures = 0
    for src in _inputs(a.inputs):
        dest = a.out / f"{src.stem}.flac"
        try:
            with tempfile.TemporaryDirectory(prefix="carnoise_") as scratch:
                r = process_file(src, dest, Path(scratch), dsp, ChunkConfig(), a.subtype, "resample")
        except Exception as e:
            failures += 1
            log.warning(f"ERROR {src.name}: {e}")
            continue
        d, s = r.stream.decisions, r.stream
        lp = f"{d.lp:.0f} Hz" if d.lp else "none"
        log.info(f"{src.name}  {s.channels}ch{' (dup)' if d.mono else ''}  hp {d.hp:5.1f} Hz  lp {lp:>8}  "
                 f"tone cells {s.tonal_cells_share:5.1%}  clip runs {d.clip_runs:3d}  "
                 f"engine tracked {s.engine_tracked_share:.0%}  "
                 f"level {s.input_rms_dbfs:.1f} -> {s.output_rms_dbfs:.1f} dBFS  -> {dest}")
    return 1 if failures else 0


def cmd_batch(a: argparse.Namespace) -> int:
    cfg = load_config(a.config, max_files=a.max_files)
    runner = BatchRunner(cfg)
    manifest = runner.prepare()
    log.info(f"Input: {cfg.data_root}\nOutput: {cfg.out_root}\nProfile: {cfg.profile}\n"
             f"Recordings: {len(manifest)} ({int(manifest.source_found.sum())} found)")
    stats = runner.run()
    return 1 if stats.errors else 0


def cmd_report(a: argparse.Namespace) -> int:
    cfg = load_config(a.config)
    by_unit = completion_by_unit(build_manifest(cfg), cfg.out_root)
    print(by_unit.to_string(index=False))
    print(f"\nComplete: {int(by_unit.complete.sum())}/{int(by_unit.total.sum())}")
    errors = failed_files(cfg.out_root)
    if len(errors):
        print("\nFailed files:")
        print(errors.to_string(index=False))
    return 0


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="carnoise", description="Remove recording-car noise from street audio.")
    ap.add_argument("--version", action="version", version=f"carnoise {__version__}")
    sub = ap.add_subparsers(dest="command", required=True)

    d = sub.add_parser("denoise", help="denoise local files or folders")
    d.add_argument("inputs", nargs="+", type=Path, help="audio files or folders (wav/flac)")
    d.add_argument("-o", "--out", type=Path, required=True, help="output folder")
    d.add_argument("--hp", default="auto", help="high-pass corner in Hz, or 'auto'")
    d.add_argument("--lp", default="auto", help="low-pass corner in Hz, 'auto' or 'none'")
    d.add_argument("--no-tones", action="store_true", help="skip the tonal-line stage")
    d.add_argument("--floor-shift", type=float, default=0.0,
                   help="dB added to the gain floor: negative = stronger, positive = more natural")
    d.add_argument("--engine-notch", action="store_true", help="also notch engine harmonics")
    d.add_argument("--subtype", default="PCM_24", help="output FLAC subtype")
    d.set_defaults(func=cmd_denoise)

    b = sub.add_parser("batch", help="process the full dataset described by a config file")
    b.add_argument("-c", "--config", type=Path, required=True)
    b.add_argument("--max-files", type=int, help="limit the run (smoke test)")
    b.set_defaults(func=cmd_batch)

    r = sub.add_parser("report", help="completion table and failed files for a config's output folder")
    r.add_argument("-c", "--config", type=Path, required=True)
    r.set_defaults(func=cmd_report)
    return ap


def main(argv: list[str] | None = None) -> int:
    configure_logging()
    a = build_parser().parse_args(argv)
    return a.func(a)


if __name__ == "__main__":
    sys.exit(main())
