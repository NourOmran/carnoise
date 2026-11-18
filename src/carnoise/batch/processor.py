"""One recording, end to end: probe -> decode -> denoise in chunks -> FLAC -> verify -> upload."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import soundfile as sf

from ..config import ChunkConfig, DSPConfig
from ..constants import FS
from ..io.audio import decode_to_raw, open_raw, probe
from ..io.storage import atomic_copy
from ..pipeline import StreamResult, denoise_stream
from ..speed import speed_from_gps


@dataclass
class FileOutcome:
    """Result of :func:`process_file`."""
    dest: Path
    sample_rate: int                 #: output sample rate (48 kHz after resampling)
    source_sample_rate: int
    speed_csv: Optional[Path]        #: GPS log the speed came from, None = no speed found
    stream: StreamResult

    @property
    def duration_seconds(self) -> float:
        return self.stream.frames / self.sample_rate


def process_file(src: Path, dest: Path, scratch: Path, dsp: DSPConfig = DSPConfig(),
                 chunking: ChunkConfig = ChunkConfig(), output_subtype: str = "PCM_24",
                 other_sample_rates: str = "resample", gps_csv: Optional[Path] = None,
                 start_utc: str = "") -> FileOutcome:
    """Denoise ``src`` into the FLAC ``dest``.

    The output is written to ``scratch`` first, checked, and only then copied to ``dest``
    (atomically), so ``dest`` either does not exist or is complete.

    Args:
        scratch: an empty local working directory (the decoded audio is stored here).
        other_sample_rates: 'resample' converts non-48 kHz input to 48 kHz (output is 48 kHz);
            'error' raises ValueError for them.
        gps_csv, start_utc: the unit's GPS log and the recording's start time (for the speed track).
    """
    if not src.is_file():
        raise FileNotFoundError(str(src))
    meta = probe(src)
    target = None
    if meta.sample_rate != FS:
        if other_sample_rates != "resample":
            raise ValueError(f"Car-noise DSP is built for {FS} Hz audio; this file is {meta.sample_rate} Hz")
        target = FS
    sr = target or meta.sample_rate
    raw_path = scratch / "decoded_audio.raw"
    local_out = scratch / "enhanced.flac"
    decode_to_raw(src, raw_path, target)
    src_mem = open_raw(raw_path, meta.channels)
    speed, speed_csv = None, None
    track = speed_from_gps(gps_csv, src.name, start_utc, src_mem.shape[0] / sr)
    if len(track) and not np.all(np.isnan(track)):
        speed, speed_csv = track, gps_csv
    with sf.SoundFile(str(local_out), "w", samplerate=sr, channels=meta.channels,
                      format="FLAC", subtype=output_subtype) as writer:
        result = denoise_stream(src_mem, sr, writer.write, dsp, chunking, speed)
    del src_mem
    check = sf.info(str(local_out))
    if check.frames != result.frames or check.channels != meta.channels or check.samplerate != sr:
        raise RuntimeError("Verification of output FLAC failed")
    atomic_copy(local_out, dest)
    return FileOutcome(dest=dest, sample_rate=sr, source_sample_rate=meta.sample_rate,
                       speed_csv=speed_csv, stream=result)
