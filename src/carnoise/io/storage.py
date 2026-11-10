"""Safe file placement on slow or network storage (Google Drive)."""
from __future__ import annotations

import os
import shutil
from pathlib import Path


def atomic_copy(local: Path, dest: Path) -> None:
    """Copy ``local`` to ``dest`` via ``dest.part`` and a rename, so an interrupted upload is
    never mistaken for a finished output."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_suffix(dest.suffix + ".part")
    shutil.copyfile(local, part)
    if part.stat().st_size != local.stat().st_size:
        raise IOError("Drive upload size mismatch")
    os.replace(part, dest)


def is_complete(path: Path, min_bytes: int = 100) -> bool:
    """An output counts as done when it exists and is not empty."""
    return path.exists() and path.stat().st_size > min_bytes
