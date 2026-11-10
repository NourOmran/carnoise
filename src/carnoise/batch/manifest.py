"""Which recordings to process, and where each output goes.

The dataset is described by ``<data_root>/files.csv`` with at least the columns
``file`` (path relative to data_root), ``unit`` and ``group`` (``driving`` / ``office``).
Outputs mirror the source hierarchy: ``<out_root>/<file>.<profile>.flac``.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from ..config import BatchConfig

REQUIRED_COLUMNS = {"file", "unit", "group"}


def safe_source(data_root: Path, relpath: str) -> Path:
    """Resolve a catalog path, refusing anything that escapes data_root."""
    base = data_root.resolve()
    p = (data_root / relpath).resolve()
    if not p.is_relative_to(base):
        raise ValueError("Unsafe relative file path: " + str(relpath))
    return p


def destination(out_root: Path, profile: str, relpath: str) -> Path:
    """Output path for a source; keeps the source hierarchy and extension to avoid collisions."""
    rel = Path(str(relpath))
    return out_root / rel.parent / f"{rel.name}.{profile}.flac"


def build_manifest(cfg: BatchConfig) -> pd.DataFrame:
    """Selected recordings, sorted by group/unit/file, with ``source_found`` and ``output_path``."""
    catalog_path = cfg.data_root / "files.csv"
    if not catalog_path.is_file():
        raise FileNotFoundError(f"Missing {catalog_path}")
    catalog = pd.read_csv(catalog_path)
    missing = REQUIRED_COLUMNS - set(catalog.columns)
    if missing:
        raise ValueError("Missing columns in files.csv: " + ", ".join(sorted(missing)))
    m = catalog[catalog["group"].isin(cfg.groups)].copy()
    m["file"] = m["file"].astype(str)
    m = m.drop_duplicates(subset=["file"]).sort_values(["group", "unit", "file"]).reset_index(drop=True)
    if cfg.max_files is not None:
        m = m.iloc[:int(cfg.max_files)].copy()
    m["source_found"] = [safe_source(cfg.data_root, f).is_file() for f in m.file]
    m["output_path"] = [str(destination(cfg.out_root, cfg.profile, f)) for f in m.file]
    if len(m) == 0:
        raise ValueError("No recordings selected; check input folder and group flags")
    return m


def summarize(manifest: pd.DataFrame) -> pd.DataFrame:
    """Recordings per group and unit."""
    return manifest.groupby(["group", "unit"]).size().rename("recordings").reset_index()
