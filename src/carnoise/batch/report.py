"""Completion reporting for an output folder."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from ..io.storage import is_complete


def completion_by_unit(manifest: pd.DataFrame, out_root: Path) -> pd.DataFrame:
    """Total / complete / remaining recordings per group and unit; also saved as completion_by_unit.csv."""
    s = manifest[["group", "unit", "file", "output_path"]].copy()
    s["complete"] = s["output_path"].apply(lambda p: is_complete(Path(p)))
    by_unit = s.groupby(["group", "unit"]).agg(total=("file", "size"), complete=("complete", "sum")).reset_index()
    by_unit["remaining"] = by_unit.total - by_unit.complete
    by_unit["completion_percent"] = np.round(100 * by_unit.complete / by_unit.total, 1)
    by_unit.to_csv(out_root / "completion_by_unit.csv", index=False)
    return by_unit


def failed_files(out_root: Path) -> pd.DataFrame:
    """Rows of errors.csv (empty frame when there are none)."""
    path = out_root / "errors.csv"
    if path.exists() and path.stat().st_size:
        return pd.read_csv(path)
    return pd.DataFrame(columns=["source_file", "unit", "error"])
