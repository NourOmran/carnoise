# How the code is organised

The code is split into layers, and each one only talks to the layer below it:

```
   carnoise CLI          Colab notebook
          \               /
           v             v
   batch/      the dataset run: which files, the loop, logs, resume
   io/         FFmpeg decoding and copying results to Drive
   pipeline.py runs the processing steps, splits long files into chunks
   analysis.py a few decisions made once per file
   dsp/        the actual signal processing, one file per step
```

The processing code in `dsp/` works on plain numpy arrays. It doesn't know about files, Drive or
FFmpeg. `pipeline.py` doesn't know about FFmpeg or Drive either. The `batch/` code does no signal
processing at all. Splitting it this way means each part can be tested on its own, and you can
reuse the processing somewhere else (in a real-time prototype, for example) without the batch
machinery.

## What happens to one recording

1. Look the file up in `files.csv` and make sure the path stays inside the data folder.
2. Ask FFmpeg for the sample rate and number of channels. If it isn't 48 kHz, FFmpeg converts it.
3. Decode the audio to a raw file on local disk. We read it from there with a memory map, so a
   long recording never has to fit in RAM.
4. Make one pass over the whole file to decide a few things up front: is it really stereo or just
   one channel copied twice, where should the high-pass and low-pass sit, and how much clipping is
   there.
5. Process the file in chunks of about 44 seconds. Each chunk gets about 7 seconds of extra audio
   on each side so the processing has context, and we keep only the middle.
6. Write the result to a FLAC on local disk, read it back to check it, then copy it to the output
   folder.
7. Add a row to the CSV logs.

## Why it's built this way

**Some decisions are made once per file.** If each chunk picked its own filter settings, you
could hear the sound change every 44 seconds. So those settings come from the single pass in
`analysis.py` and every chunk uses the same ones.

**Chunk edges line up with the processing frames.** Chunk lengths are multiples of 32768 samples,
the largest window any step uses. That way the audio near a chunk edge gets processed the same
way in both neighbouring chunks, and the joins can't be heard. The test suite checks that chunked
output matches processing the whole file in one go to within -40 dB. The engine tracker fills gaps
using only nearby frames, for the same reason.

**Memory use stays flat.** Only one chunk plus its context (under a minute of audio) is in memory
at a time, however long the recording is.

**You never get half-written files.** The output is written locally, checked, copied to Drive
under a `.part` name and then renamed. If a FLAC exists in the output folder, it's complete. That's
what makes resuming safe.

**One broken file doesn't stop the run.** Each file is processed inside its own try/except. A
failure is written to `errors.csv` and the run moves on to the next file.

**An output folder only ever holds one version of the processing.** The first run writes the
settings to `run_config.json`. Running again into the same folder with different settings is
refused. To try new settings, use a new output folder and a new profile name.

**It stays compatible with the old notebook.** File names, CSV columns and `run_config.json` are
exactly what the notebook wrote, so either one can continue a folder the other started. The only
addition is a `source_sample_rate_hz` column in `dsp_details.csv`.

## Where things are

| File | What's in it |
|---|---|
| `constants.py` | Fixed numbers: 48 kHz, window sizes, the chunk grid. Changing them changes the sound. |
| `config.py` | The settings you can change, and loading them from TOML |
| `dsp/declip.py` | Repairs clipped peaks |
| `dsp/engine.py` | Tracks the engine and removes its harmonics |
| `dsp/filters.py` | High-pass and low-pass, and picking them automatically |
| `dsp/tonal.py` | Removes steady hums (v1, and the horn-safe v3 version) |
| `dsp/clicks.py` | Repairs mount rattle (v1 only) |
| `dsp/wind.py` | Removes wind using the two microphones (v3) |
| `dsp/noise.py` | Estimates the background noise and reduces it |
| `dsp/spectral.py`, `dsp/channels.py` | Small helpers |
| `analysis.py` | The once-per-file decisions |
| `speed.py` | Reads GPS speed: per-file speed CSVs, or the unit's `gps/` log (v3) |
| `pipeline.py` | `denoise()` for one block of audio, `denoise_stream()` for whole files in chunks |
| `io/audio.py` | FFmpeg probing and decoding |
| `io/storage.py` | The safe copy to Drive |
| `batch/manifest.py` | Turns `files.csv` into the list of files to process |
| `batch/processor.py` | Handles one file from start to finish |
| `batch/runner.py` | The loop over all files, with resume |
| `batch/logs.py` | The CSV logs and the settings lock |
| `batch/report.py` | Completion per unit |
| `cli.py` | The `carnoise` command |

## Adding to it

- **A new processing step:** put it in its own file under `dsp/` as a function that takes and
  returns an array, call it from `denoise()` in `pipeline.py`, and add a test. If it needs a
  setting decided once per file, add that to `analysis.py`. Give the output a new profile name so
  it doesn't get mixed with old results.
- **An ML model on top:** wrap it as a function from array to array and call it at the end of
  `denoise()`. Chunking, logs and resume keep working as they are.
- **Different storage (S3, GCS):** only `io/storage.py` and the paths in the manifest touch
  storage, so those are the only places to change.
