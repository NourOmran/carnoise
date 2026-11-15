"""carnoise - remove a recording car's own noise from outside-microphone street audio.

Public API::

    from carnoise import denoise, DSPConfig            # in-memory arrays
    from carnoise import denoise_stream, ChunkConfig   # long recordings, chunk by chunk
    from carnoise import BatchRunner, load_config      # full dataset (files.csv)
"""
import logging

from .analysis import FileDecisions, file_stats
from .batch import BatchRunner, completion_by_unit, process_file
from .config import BatchConfig, ChunkConfig, DSPConfig, load_config
from .pipeline import DenoiseInfo, denoise, denoise_stream

__version__ = "1.1.0"

__all__ = ["denoise", "denoise_stream", "DenoiseInfo", "DSPConfig", "ChunkConfig", "BatchConfig",
           "load_config", "FileDecisions", "file_stats", "BatchRunner", "process_file",
           "completion_by_unit", "configure_logging", "__version__"]


def configure_logging(level: int = logging.INFO) -> None:
    """Print progress lines (``[i/n] OK ...``) to the console; used by the CLI and the notebook."""
    logger = logging.getLogger("carnoise")
    if not logger.handlers:
        h = logging.StreamHandler()
        h.setFormatter(logging.Formatter("%(message)s"))
        logger.addHandler(h)
    logger.setLevel(level)
