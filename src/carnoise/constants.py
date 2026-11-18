"""Fixed numbers of the pipeline.

These are not user settings: the DSP stages are tuned around them, and changing any of
them changes the output. User-facing settings live in :mod:`carnoise.config`.
"""

#: Sample rate the DSP is tuned for. Other rates are resampled (or rejected) before processing.
FS = 48000

#: Main STFT (wind + Wiener stages): 2048-sample window (42.7 ms), 75 % overlap.
NFFT = 2048
HOP = 512

#: Largest STFT window in the pipeline (tonal-line stage). Every hop divides it, so chunk
#: edges placed on multiples of GRID fall on a frame edge of every STFT in the pipeline.
GRID = 32768

#: Channel 2 counts as a copy of channel 1 when their difference is below -40 dB of channel 1.
DUP_RATIO = 1e-4

#: Samples at or above this magnitude are treated as clipped.
CLIP_LEVEL = 0.999

#: Block size for the streaming per-file analysis pass (~61 s). A multiple of 4096 * 6 so the
#: 0.5 s level frames of the low-pass decision line up across blocks.
STATS_BLOCK = 98304 * 30
