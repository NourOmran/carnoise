# carnoise

Cleans up street recordings made with microphones mounted on a car. The car itself is loud: the
engine, tyres on the road, wind hitting the mics, the mount rattling, a fan or alternator humming.
This removes as much of that as possible while leaving the street (people, traffic, birds, horns,
the general feel of the place) sounding natural.

It was written for the ensense dataset and runs as a batch job over the whole dataset, on Google
Colab or any machine with Python and FFmpeg.

There are two versions of the processing:

- **v3 (recommended)**: the improved processing that came out of the QE evaluation. It separates
  street sound from the car much better than v1, doesn't damage horns, birds or knocks, removes
  wind with the two microphones, and actually uses GPS speed. Config: `config/colab_v3.toml` or
  `config/local_v3.toml`.
- **v1**: the original Colab notebook's processing, unchanged and checked sample for sample
  against the notebook, so folders it started can still be resumed. Config: `config/colab.toml`.

[docs/algorithms.md](docs/algorithms.md#v3-what-changed-and-why) explains what v3 changes and why.

![How it works](docs/algorithm_mindmap.png)

## Setup

You need Python 3.10 or newer and FFmpeg (`apt-get install ffmpeg` on Linux/Colab,
`brew install ffmpeg` on a Mac).

```bash
pip install .
```

For development (adds pytest): `pip install -e ".[dev]"`

## Running it

The main use is processing the whole dataset. It reads `files.csv` from the data folder and writes
one cleaned FLAC per recording:

```bash
carnoise batch -c config/colab_v3.toml
```

If it stops halfway (a Colab disconnect, for example), run the same command again. Files that are
already done get skipped.

A few other things you can do:

```bash
carnoise batch -c config/colab_v3.toml --max-files 3    # quick test on 3 files
carnoise report -c config/colab_v3.toml                 # how far along each unit is, and what failed
carnoise denoise recording.flac some_folder/ -o cleaned/ --algorithm v3   # just clean a few files
```

Or from Python:

```python
import soundfile as sf
from carnoise import DSPConfig, denoise

audio, sr = sf.read("recording.wav")   # needs to be 48 kHz
speed = None                           # or a per-second GPS speed array in km/h
cleaned, info = denoise(audio, DSPConfig(algorithm="v3"), speed)
print(info)   # what it found: filter settings, clicks, how well it tracked the engine
```

On Colab, open `notebooks/colab_runner.ipynb`. It installs the package, runs everything and lets
you listen to the before/after.

## Settings worth knowing

Everything is in `config/colab_v3.toml` with a comment next to it. The ones you're most likely to
touch:

- `floor_shift_db` controls how hard it works. The default is 0. Try -3 for more removal, or +3 if
  the street starts sounding thin or watery.
- `high_pass_hz` and `low_pass_hz` are picked automatically per file. You can force a number if
  you need to.
- `remove_tonal_lines` removes steady hums (fans, alternator). In v3 it leaves horns alone.
- `engine_notch` (v3) also notches the engine's harmonics. It removes more engine drone but costs a
  little clarity, so it's off by default. Decide by listening.
- `click_k` (v1 only) sets how sensitive the rattle repair is. v3 has no click repair, because it
  was removing bird calls and knocks.
- `other_sample_rates` decides what happens to files that aren't 48 kHz. By default they're
  converted. Set it to `"error"` to skip them instead, which is what the old notebook did.

## More docs

- [docs/algorithms.md](docs/algorithms.md) explains each processing step and its numbers.
- [docs/architecture.md](docs/architecture.md) shows how the code is organised and why.
- [docs/operations.md](docs/operations.md) covers running it, what ends up in the output folder,
  and what to do when something fails.

## Tests

```bash
pytest
```

This takes about two minutes. It uses generated test audio, so no real recordings are needed. It also
does a small end-to-end batch run, which needs FFmpeg installed.

## Things it doesn't do well yet

- Road and tyre noise sit in the same range as voices, so they're turned down, not removed. That's
  the main thing you'll still hear if you turn the volume right up. Going further would need a
  trained model.
- In v1 only: horns lose about 12 dB, and click repair removes bird calls. Both are fixed in v3.
- Where a recording clipped badly, the lost part can only be guessed.
