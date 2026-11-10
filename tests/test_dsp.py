import numpy as np
import pytest

from carnoise import DSPConfig, denoise
from carnoise.dsp import as_channels, declip, remove_engine_orders, remove_tonal_lines, repair_clicks, stft
from carnoise.dsp.engine import N_FFT as ENGINE_NFFT, track_f0
from conftest import FS


def level_db(x, f_lo, f_hi):
    X = np.fft.rfft(x * np.hanning(len(x)))
    f = np.fft.rfftfreq(len(x), 1 / FS)
    return 10 * np.log10(np.sum(np.abs(X[(f >= f_lo) & (f <= f_hi)]) ** 2) + 1e-30)


def test_duplicate_channels_become_mono():
    x = np.random.default_rng(1).standard_normal((48000, 1)) * 0.1
    assert as_channels(np.hstack([x, x])).shape[1] == 1
    assert as_channels(np.hstack([x, x * 0.5])).shape[1] == 2


def test_declip_restores_peaks():
    t = np.arange(4800) / FS
    true = 1.4 * np.sin(2 * np.pi * 100 * t)
    clipped = np.clip(true, -0.999, 0.999)
    fixed, runs = declip(clipped)
    assert runs > 0
    assert np.max(np.abs(fixed)) > 1.1                     # peaks re-drawn above the clip level
    assert np.mean((fixed - true) ** 2) < 0.5 * np.mean((clipped - true) ** 2)


def test_engine_f0_is_tracked_and_harmonics_reduced(car_audio):
    x, _ = car_audio
    f, X = stft(x, ENGINE_NFFT, ENGINE_NFFT // 4)
    f0 = track_f0(f, np.mean(np.abs(X) ** 2, axis=0))
    good = f0[~np.isnan(f0)]
    assert len(good) > 0.8 * len(f0)
    assert 35 < np.median(good) < 46
    y, _ = remove_engine_orders(x)
    assert level_db(y[:, 0], 150, 400) < level_db(x[:, 0], 150, 400) - 3


def test_tonal_line_removed_but_rest_kept(car_audio):
    x, _ = car_audio
    y, share = remove_tonal_lines(x)
    assert 0 < share < 0.2
    assert level_db(y[:, 0], 525, 535) < level_db(x[:, 0], 525, 535) - 6
    assert abs(level_db(y[:, 0], 2000, 4000) - level_db(x[:, 0], 2000, 4000)) < 1.0


@pytest.mark.xfail(strict=True, reason="known limitation of carnoise_v1 (docs/algorithms.md): the tonal "
                   "stage's 0.7 s smoothing + 0.68 s window let a steady-pitch event of ~0.2 s pass the "
                   "2 s persistence test, so horn-like sounds are cut by ~12 dB")
def test_short_steady_horn_is_kept():
    from conftest import band_noise
    n = 12 * FS
    x = 0.05 * band_noise(np.random.default_rng(0), n, 30, 8000)
    s, m = 4 * FS, int(0.5 * FS)
    t = np.arange(m) / FS
    x[s:s + m] += sum(0.03 / h * np.sin(2 * np.pi * 400 * h * t) for h in (1, 2, 3))
    y, _ = remove_tonal_lines(x[:, None])
    assert level_db(y[s:s + m, 0], 300, 1300) > level_db(x[s:s + m], 300, 1300) - 3


def test_clicks_detected(car_audio):
    x, click_at = car_audio
    f, X = stft(x)
    _, hit = repair_clicks(X, f, k=5.0)
    frames_hit = np.where(hit)[0] * 512
    found = sum(np.any(np.abs(frames_hit - c) < 2048) for c in click_at)
    assert found >= 0.8 * len(click_at)


def test_denoise_end_to_end(car_audio):
    x, _ = car_audio
    y, info = denoise(x)
    assert y.shape == x.shape and np.isfinite(y).all()
    assert info.channels == 2 and 70 <= info.hp <= 200
    assert info.engine_tracked > 0.8
    assert level_db(y[:, 0], 20, 20000) < level_db(x[:, 0], 20, 20000)       # quieter overall
    # a street event (bird-like glide, 2.5-5.5 kHz at 1.5 s) survives
    s = int(1.5 * FS)
    seg = slice(s, s + int(0.3 * FS))
    assert level_db(y[seg, 0], 2500, 5500) > level_db(x[seg, 0], 2500, 5500) - 6


def test_settings_change_strength(car_audio):
    x, _ = car_audio
    x = x[: 8 * FS]
    soft, _ = denoise(x, DSPConfig(floor_shift_db=+6))
    hard, _ = denoise(x, DSPConfig(floor_shift_db=-6))
    assert np.mean(hard ** 2) < np.mean(soft ** 2)
    _, info = denoise(x, DSPConfig(high_pass_hz=120, low_pass_hz="none", remove_tonal_lines=False))
    assert info.hp == 120 and info.lp is None and info.tones == 0
