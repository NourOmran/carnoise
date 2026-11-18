# carnoise

Cleans up street recordings made with microphones mounted on a car. The car itself is loud: the
engine, tyres on the road, wind hitting the mics, a fan or alternator humming. This removes as much
of that as possible while leaving the street (people, traffic, birds, horns, sirens, the general
feel of the place) intact and natural.

It was written for the ensense dataset and runs as a batch job over the whole dataset, on Google
Colab or any machine with Python and FFmpeg. Output files are named with the profile `carnoise_v3`.

How it works, step by step, with the measured results: [docs/algorithms.md](docs/algorithms.md).

## Setup

You need Python 3.10 or newer and FFmpeg (`apt-get install ffmpeg` on Linux/Colab,
`brew install ffmpeg` on a Mac).

```bash
pip install .
```

For development (adds pytest): `pip install -e ".[dev]"`

## Running it

The main use is processing the whole dataset. It reads `files.csv` and the GPS logs in `gps/` from
the data folder and writes one cleaned FLAC per recording:

```bash
carnoise batch -c config/colab.toml
```

If it stops halfway (a Colab disconnect, for example), run the same command again. Files that are
already done get skipped.

A few other things you can do:

```bash
carnoise batch -c config/colab.toml --max-files 3    # quick test on 3 files
carnoise report -c config/colab.toml                 # how far along each unit is, and what failed
carnoise denoise recording.flac some_folder/ -o cleaned/   # just clean a few files (no GPS speed)
```

Use `batch` for real results: it's the path that reads GPS speed. `denoise` runs every file at the
"stopped" setting.

Or from Python:

```python
import soundfile as sf
from carnoise import DSPConfig, denoise

audio, sr = sf.read("recording.wav")   # needs to be 48 kHz
speed = None                           # or a per-second GPS speed array in km/h
cleaned, info = denoise(audio, DSPConfig(), speed)
print(info)   # what it decided: filter settings, hum removed, mean speed
```

On Colab, open `notebooks/colab_runner.ipynb`. It installs the package, runs everything and lets
you listen to the before/after.

## Settings worth knowing

Everything is in `config/colab.toml` with a comment next to it. The ones you're most likely to
touch:

- `floor_shift_db` controls how hard it works. The default is 0. Try -3 for more removal, or +3 if
  the street starts sounding thin or watery.
- `high_pass_hz` and `low_pass_hz` are picked automatically per file. You can force a number if
  you need to.
- `remove_tonal_lines` removes steady hums (fans, alternator). Horns, sirens and alarms are kept.
- `engine_notch` also notches the engine's harmonics. It removes more engine drone but costs a
  little clarity, so it's off by default. Decide by listening.
- `other_sample_rates` decides what happens to files that aren't 48 kHz. By default they're
  converted. Set it to `"error"` to skip them instead.

## More docs

- [docs/algorithms.md](docs/algorithms.md) explains each processing step and its numbers.
- [docs/architecture.md](docs/architecture.md) shows how the code is organised and why.
- [docs/operations.md](docs/operations.md) covers running it, what ends up in the output folder,
  and what to do when something fails.

## Tests

```bash
pytest
```

This takes about a minute. It uses generated test audio, so no real recordings are needed. It also
does a small end-to-end batch run, which needs FFmpeg installed.

## Things it doesn't do well yet

- Road and tyre noise sit in the same range as voices and many street sounds, so they're turned
  down, not removed. That's the main thing you'll still hear if you turn the volume right up.
  Going further would need a trained model.
- A passing vehicle that stays loud for more than about 3 seconds starts to be treated as
  background noise.
- Where a recording clipped badly, the lost part can only be guessed.
