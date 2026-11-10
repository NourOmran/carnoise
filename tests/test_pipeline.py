import numpy as np
import pytest

from carnoise import ChunkConfig, DSPConfig, denoise, denoise_stream, file_stats
from carnoise.constants import GRID, STATS_BLOCK
from carnoise.pipeline import ChunkPlan
from conftest import FS, make_car_audio


def test_chunk_plan_covers_every_sample_once():
    plan = ChunkPlan.for_rate(FS, ChunkConfig(block_seconds=12, context_seconds=6))
    assert plan.chunk_frames % GRID == 0 and plan.pad % GRID == 0
    frames = 10 * plan.chunk_frames + 1234
    covered = np.zeros(frames, int)
    for start, stop, left, right in plan.chunks(frames):
        assert left <= start < stop <= right
        covered[start:stop] += 1
    assert (covered == 1).all()


def test_chunked_output_matches_whole_file():
    x, _ = make_car_audio(seconds=40, seed=3)
    x = x.astype(np.float32)
    out = []
    res = denoise_stream(x, FS, out.append, DSPConfig(), ChunkConfig(block_seconds=12, context_seconds=6))
    y_chunked = np.concatenate(out)
    assert res.chunks >= 3 and y_chunked.shape == x.shape

    decisions = file_stats(x[i:i + STATS_BLOCK].astype(np.float64) for i in range(0, len(x), STATS_BLOCK))
    y_whole, _ = denoise(x.astype(np.float64), DSPConfig(), decisions=decisions)
    err = np.mean((y_chunked - y_whole) ** 2) / np.mean(y_whole ** 2)
    assert 10 * np.log10(err) < -40          # chunk joins are inaudible


def test_duplicate_channel_file_keeps_two_channels():
    m, _ = make_car_audio(seconds=4, stereo=False)
    x = np.stack([m, m], axis=1).astype(np.float32)
    out = []
    res = denoise_stream(x, FS, out.append)
    y = np.concatenate(out)
    assert res.decisions.mono and y.shape == x.shape
    assert np.array_equal(y[:, 0], y[:, 1])


def test_short_file_is_still_processed():
    x = (np.random.default_rng(0).standard_normal((20000, 1)) * 0.05).astype(np.float32)
    out = []
    res = denoise_stream(x, FS, out.append)
    assert np.concatenate(out).shape == x.shape and res.infos == []     # high-pass only


def test_rejects_bad_input():
    with pytest.raises(ValueError, match="48000"):
        denoise_stream(np.zeros((1000, 1), np.float32), 44100, lambda b: None)
    x = np.zeros((100000, 1), np.float32)
    x[5] = np.nan
    with pytest.raises(ValueError, match="NaN"):
        denoise_stream(x, FS, lambda b: None)
