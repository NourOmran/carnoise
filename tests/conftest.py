"""Synthetic test audio: car noise (engine harmonics, a fan tone, road noise, clicks) plus
"street" events, so every stage has something to find without shipping real recordings."""
import numpy as np
import pytest

FS = 48000
STREET_F0 = 2500.0       # street events glide from 2.5 kHz to ~5.5 kHz in 0.3 s


def band_noise(rng, n, lo, hi):
    X = np.fft.rfft(rng.standard_normal(n))
    f = np.fft.rfftfreq(n, 1 / FS)
    X[(f < lo) | (f > hi)] = 0
    X[f > 0] /= np.sqrt(np.maximum(f[f > 0], lo))      # pink-ish, like road noise
    y = np.fft.irfft(X, n)
    return y / np.max(np.abs(y))


def make_car_audio(seconds=20.0, seed=0, stereo=True):
    rng = np.random.default_rng(seed)
    n = int(seconds * FS)
    t = np.arange(n) / FS
    f0 = 40 + 3 * np.sin(2 * np.pi * t / 7)              # engine firing frequency, drifting
    phase = 2 * np.pi * np.cumsum(f0) / FS
    engine = sum(0.08 / h * np.sin(h * phase) for h in range(1, 13))
    fan = 0.02 * np.sin(2 * np.pi * 530 * t)
    road = 0.05 * band_noise(rng, n, 30, 8000)
    street = np.zeros(n)
    for k in range(int(seconds // 3)):                   # short gliding "bird" calls
        s = int((1.5 + 3 * k) * FS)
        m = int(0.3 * FS)
        street[s:s + m] += 0.05 * np.sin(2 * np.pi * (STREET_F0 + 5000 * t[:m]) * t[:m]) * np.hanning(m)
    clicks = np.zeros(n)
    click_at = (np.arange(1, int(seconds)) * FS + 1234)[: int(seconds) - 1]
    for c in click_at:
        clicks[c:c + 48] += 0.3 * rng.standard_normal(48)
    mono = engine + fan + road + street + clicks
    if not stereo:
        return mono, click_at
    other = engine + fan + 0.05 * band_noise(rng, n, 30, 8000) + street + clicks
    return np.stack([mono, other], axis=1), click_at


@pytest.fixture(scope="session")
def car_audio():
    return make_car_audio()
