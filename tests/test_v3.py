"""Behaviour the QE evaluation asked for: events kept, wind and fans removed, GPS speed used."""
import json
import shutil

import numpy as np
import pandas as pd
import pytest
import soundfile as sf

from carnoise import BatchConfig, BatchRunner, ChunkConfig, DSPConfig, denoise, denoise_stream, file_stats, load_config
from carnoise.constants import STATS_BLOCK
from carnoise.dsp import stft, istft
from carnoise.dsp.engine import N_FFT as ENGINE_NFFT, track_f0
from carnoise.dsp.tonal import remove_tonal_lines
from carnoise.dsp.wind import wind_gain
from carnoise.speed import speed_from_gps, speed_per_frame
from conftest import FS, STREET_F0, band_noise, make_car_audio

V3 = DSPConfig()


def level_db(x, lo, hi):
    X = np.fft.rfft(x * np.hanning(len(x)))
    f = np.fft.rfftfreq(len(x), 1 / FS)
    return 10 * np.log10(np.sum(np.abs(X[(f >= lo) & (f <= hi)]) ** 2) + 1e-30)


@pytest.mark.parametrize("horn_s", [0.5, 2.0, 4.0])
def test_horn_kept_while_fan_removed(horn_s):
    rng = np.random.default_rng(0)
    n = 30 * FS
    x = 0.05 * band_noise(rng, n, 30, 8000)
    t = np.arange(n) / FS
    x += 0.01 * np.sin(2 * np.pi * np.cumsum(555 + 3 * np.sin(2 * np.pi * t / 8)) / FS)   # drifting fan
    s, m = 4 * FS, int(horn_s * FS)
    th = np.arange(m) / FS
    x[s:s + m] += sum(0.03 / h * np.sin(2 * np.pi * 400 * h * th) for h in (1, 2, 3))
    y, _ = remove_tonal_lines(x[:, None])
    assert level_db(y[s:s + m, 0], 300, 1300) > level_db(x[s:s + m], 300, 1300) - 1.0     # horn kept
    fan = slice(12 * FS, 24 * FS)
    assert level_db(y[fan, 0], 545, 565) < level_db(x[fan], 545, 565) - 10                # fan removed


def test_v3_end_to_end_keeps_street_events(car_audio):
    x, _ = car_audio
    y, info = denoise(x, V3, np.full(20, 60.0))
    assert y.shape == x.shape and np.isfinite(y).all()
    assert info.speed_mean == pytest.approx(60.0)
    assert level_db(y[:, 0], 20, 20000) < level_db(x[:, 0], 20, 20000) - 6               # car noise removed
    s = int(1.5 * FS)
    seg = slice(s, s + int(0.3 * FS))
    assert level_db(y[seg, 0], STREET_F0, 5500) > level_db(x[seg, 0], STREET_F0, 5500) - 3


def test_v3_chunked_matches_whole_file():
    x, _ = make_car_audio(seconds=40, seed=3)
    x = x.astype(np.float32)
    out = []
    denoise_stream(x, FS, out.append, V3, ChunkConfig(block_seconds=12, context_seconds=6))
    y_chunked = np.concatenate(out)
    decisions = file_stats(x[i:i + STATS_BLOCK].astype(np.float64) for i in range(0, len(x), STATS_BLOCK))
    y_whole, _ = denoise(x.astype(np.float64), V3, decisions=decisions)
    err = np.mean((y_chunked - y_whole) ** 2) / np.mean(y_whole ** 2)
    assert 10 * np.log10(err) < -40


def test_wind_stage_leaves_calm_audio_and_cuts_gusts():
    rng = np.random.default_rng(1)
    n = 12 * FS
    street = 0.02 * band_noise(rng, n, 100, 4000)
    st = np.c_[street, np.r_[np.zeros(3), street[:-3]] * 0.9]          # one source, both mics
    wind = np.zeros((n, 2))
    gust = slice(6 * FS, 9 * FS)
    wind[gust, 0] = 0.3 * band_noise(rng, n, 20, 500)[gust]            # gust on mic 1 only
    f, X = stft(st + wind)
    G = wind_gain(X, f)
    calm_s = istft(stft(st)[1] * G, n)
    gust_w = istft(stft(wind)[1] * G, n)
    calm = slice(0, 5 * FS)
    assert abs(level_db(calm_s[calm, 0], 100, 4000) - level_db(st[calm, 0], 100, 4000)) < 0.2
    assert level_db(gust_w[gust, 0], 20, 500) < level_db(wind[gust, 0], 20, 500) - 6


def test_engine_tracker_finds_engine_but_not_a_voice():
    from carnoise.dsp.spectral import stft as sstft
    x, _ = make_car_audio(seconds=10, seed=2, stereo=False)
    f, X = sstft(x[:, None], ENGINE_NFFT, ENGINE_NFFT // 4)
    f0 = track_f0(f, np.abs(X[0]) ** 2)
    assert np.mean(~np.isnan(f0)) > 0.8 and np.nanmedian(np.abs(f0 - 40)) < 4
    # a voice-like series: 120 Hz with vibrato and syllable on/off, no car
    rng = np.random.default_rng(3)
    n = 10 * FS
    t = np.arange(n) / FS
    ph = 2 * np.pi * np.cumsum(120 + 8 * np.sin(2 * np.pi * 5 * t)) / FS
    env = (np.sin(2 * np.pi * 3 * t) > 0).astype(float)
    voice = env * sum(0.05 / h * np.sin(h * ph) for h in range(1, 15)) + 0.002 * rng.standard_normal(n)
    f, X = sstft(voice[:, None], ENGINE_NFFT, ENGINE_NFFT // 4)
    f0 = track_f0(f, np.abs(X[0]) ** 2)
    assert np.mean(~np.isnan(f0)) < 0.1


def test_speed_from_gps(tmp_path):
    p = tmp_path / "unit_01_gps.csv"
    p.write_text("audio_file,time_utc,lat,lon,alt_m,speed_kmh,course_deg,fix_quality,satellites,hdop,receiver\n"
                 "rec.flac,2026-01-01T10:00:00.00Z,0,0,0,10,0,1,5,1,a\n"
                 "rec.flac,2026-01-01T10:00:00.50Z,0,0,0,20,0,1,5,1,b\n"
                 "rec.flac,2026-01-01T10:00:02.00Z,0,0,0,99,0,0,5,1,a\n"      # no fix: ignored
                 "rec.flac,2026-01-01T10:00:03.00Z,0,0,0,40,0,1,5,1,a\n"
                 "other.flac,2026-01-01T10:00:01.00Z,0,0,0,70,0,1,5,1,a\n")
    v = speed_from_gps(p, "rec.flac", "2026-01-01T10:00:00Z", 4)
    assert v[0] == 15 and np.isnan(v[1]) and np.isnan(v[2]) and v[3] == 40
    frames = speed_per_frame(v, 400)
    assert np.all((frames >= 15) & (frames <= 40))


def test_settings_and_run_lock(tmp_path):
    with pytest.raises(ValueError):
        BatchConfig(data_root=tmp_path, out_root=tmp_path, profile="carnoise_v1")   # v1 is gone
    lock = BatchConfig(data_root=tmp_path, out_root=tmp_path).run_lock()
    # same lock as carnoise 1.1, so v3 output folders started then still resume
    assert lock["profile"] == "carnoise_v3" and lock["algorithm"] == "v3" and "click_k" not in lock
    assert lock["processing"].startswith("FFmpeg float32 decode; car-noise DSP v3")
    toml = tmp_path / "c.toml"
    toml.write_text('[paths]\ndata_root="d"\nout_root="o"\n[dsp]\nprofile="carnoise_v3"\nalgorithm="v3"\n'
                    'engine_notch=true\n')
    cfg = load_config(toml)                                              # 1.1-style config still loads
    assert cfg.profile == "carnoise_v3" and cfg.dsp.engine_notch
    toml.write_text('[paths]\ndata_root="d"\nout_root="o"\n[dsp]\nalgorithm="v1"\n')
    with pytest.raises(ValueError, match="only has v3"):
        load_config(toml)


@pytest.mark.skipif(not shutil.which("ffmpeg"), reason="ffmpeg not installed")
def test_v3_batch_reads_gps_speed(tmp_path):
    data = tmp_path / "ensense"
    a = data / "audio" / "driving" / "unit_01"
    a.mkdir(parents=True)
    (data / "gps").mkdir()
    x, _ = make_car_audio(seconds=6)
    sf.write(a / "rec1.flac", x * 0.5, FS, subtype="PCM_24")
    (data / "gps" / "unit_01_gps.csv").write_text(
        "audio_file,time_utc,speed_kmh,fix_quality\n" +
        "".join(f"rec1.flac,2026-01-01T10:00:0{i}.00Z,{50 + i},1\n" for i in range(6)))
    (data / "files.csv").write_text("file,unit,group,start_utc\n"
                                    "audio/driving/unit_01/rec1.flac,unit_01,driving,2026-01-01T10:00:00Z\n")
    cfg = BatchConfig(data_root=data, out_root=tmp_path / "out", tmp_root=tmp_path / "work",
                      profile="carnoise_v3", dsp=V3)
    stats = BatchRunner(cfg).run()
    assert (stats.processed, stats.errors) == (1, 0)
    dsp = pd.read_csv(cfg.out_root / "dsp_details.csv")
    assert dsp.speed_file.iloc[0].endswith("unit_01_gps.csv")
    assert json.loads((cfg.out_root / "run_config.json").read_text())["algorithm"] == "v3"
    assert sf.info(str(cfg.out_root / "audio/driving/unit_01/rec1.flac.carnoise_v3.flac")).channels == 2
