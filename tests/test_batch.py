"""End-to-end batch run on a tiny generated dataset (needs ffmpeg)."""
import json
import shutil
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import soundfile as sf
from scipy import signal

from carnoise import BatchConfig, BatchRunner, DSPConfig, load_config
from conftest import FS, make_car_audio

pytestmark = pytest.mark.skipif(not shutil.which("ffmpeg"), reason="ffmpeg not installed")

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture()
def dataset(tmp_path):
    data = tmp_path / "ensense"
    a = data / "audio" / "driving" / "unit_01"
    o = data / "audio" / "office" / "unit_02"
    a.mkdir(parents=True)
    o.mkdir(parents=True)
    x, _ = make_car_audio(seconds=6)
    sf.write(a / "rec1.flac", x * 0.5, FS, subtype="PCM_24")
    (a / "rec1_speed.csv").write_text("t,speed_kmh\n" + "".join(f"{i},{40 + i}\n" for i in range(7)))
    sf.write(o / "rec44.flac", signal.resample_poly(x[:, 0] * 0.5, 147, 160), 44100, subtype="PCM_24")
    (data / "files.csv").write_text("file,unit,group\n"
                                    "audio/driving/unit_01/rec1.flac,unit_01,driving\n"
                                    "audio/office/unit_02/rec44.flac,unit_02,office\n"
                                    "audio/office/unit_02/gone.flac,unit_02,office\n")
    return BatchConfig(data_root=data, out_root=tmp_path / "out", tmp_root=tmp_path / "work")


def test_full_run_logs_and_resume(dataset):
    stats = BatchRunner(dataset).run()
    assert (stats.processed, stats.errors) == (2, 1)
    out = dataset.out_root
    dest = out / "audio/driving/unit_01/rec1.flac.carnoise_v1.flac"
    info = sf.info(str(dest))
    assert info.samplerate == FS and info.channels == 2 and info.frames == 6 * FS
    assert sf.info(str(out / "audio/office/unit_02/rec44.flac.carnoise_v1.flac")).samplerate == FS

    metrics = pd.read_csv(out / "processed_metrics.csv")
    assert set(metrics.status) == {"processed"} and len(metrics) == 2
    dsp = pd.read_csv(out / "dsp_details.csv").set_index("source_file")
    assert dsp.loc["audio/driving/unit_01/rec1.flac", "speed_file"].endswith("rec1_speed.csv")
    assert dsp.loc["audio/office/unit_02/rec44.flac", "source_sample_rate_hz"] == 44100
    errors = pd.read_csv(out / "errors.csv")
    assert list(errors.source_file) == ["audio/office/unit_02/gone.flac"]
    assert json.loads((out / "run_config.json").read_text())["profile"] == "carnoise_v1"

    again = BatchRunner(dataset).run()                     # resume: nothing redone
    assert (again.processed, again.skipped_existing) == (0, 2)


def test_error_mode_for_other_rates(dataset):
    stats = BatchRunner(replace(dataset, other_sample_rates="error", include_office=True)).run()
    assert stats.errors == 2


def test_settings_are_locked(dataset):
    BatchRunner(replace(dataset, max_files=1)).run()
    changed = replace(dataset, dsp=DSPConfig(floor_shift_db=-3))
    with pytest.raises(ValueError, match="different processing configuration"):
        BatchRunner(changed).prepare()


def test_example_configs_load():
    cfg = load_config(ROOT / "config" / "colab.toml")
    assert cfg.profile == "carnoise_v1" and cfg.max_files is None
    assert cfg.run_lock()["high_pass_hz"] == "auto"
    local = load_config(ROOT / "config" / "local.toml", max_files=2)
    assert local.max_files == 2
