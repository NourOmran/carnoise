# Running it

## What the data folder should look like

```
<data_root>/
  files.csv                 columns: file, unit, group, start_utc (others ignored)
  audio/<group>/<unit>/<name>.flac or .wav
  gps/<unit>_gps.csv        the unit's GPS log (audio_file, time_utc, speed_kmh, fix_quality)
```

**Where speed comes from.** Each recording's rows are taken from `gps/<unit>_gps.csv` (rows with a
GPS fix only), lined up using `start_utc` from `files.csv`. A recording with no GPS rows runs at the
gentlest ("stopped") setting. `dsp_details.csv` shows the GPS log each file used (`speed_file`;
empty means no speed was found).

`group` is either `driving` or `office`. The paths in `files.csv` are relative to the data folder,
and anything pointing outside it is refused.

## On Google Colab

1. Copy this repo folder to your Drive, for example `MyDrive/carnoise-pipeline`. The notebook uses
   `config/colab.toml`.
2. Open `notebooks/colab_runner.ipynb` in Colab and run the cells from top to bottom.
3. If Colab disconnects, reconnect and run them again. Files that are already done are skipped.

On Colab's CPU it runs at about 5.5 times real speed, so a 5 minute recording takes around a
minute. The full dataset (about 1,455 files) needs several sessions. FFmpeg is already installed on
Colab.

## On a server, or as a daily job

```bash
pip install /path/to/carnoise-pipeline
carnoise batch -c config/local.toml && carnoise report -c config/local.toml
```

If any file fails, the command exits with code 1, so a scheduler (cron or similar) can tell you.
New recordings added to `files.csv` get picked up on the next run, and the old ones are skipped.

## What ends up in the output folder

```
<out_root>/
  audio/<group>/<unit>/<name>.<ext>.<profile>.flac    the cleaned recording (24-bit FLAC)
  run_config.json          the settings this folder was made with
  manifest.csv             every selected recording, whether its source was found, where its output goes
  processed_metrics.csv    one row per finished file: length, channels, levels before and after,
                           peak, clipped samples, file size
  errors.csv               one row per file that failed, with the error
  dsp_details.csv          what the processing decided for each file: stereo or duplicate channels,
                           high-pass and low-pass, clip repairs, engine tracking (engine notch
                           only), hum removal, GPS log used, number of chunks, original sample rate
  completion_by_unit.csv   written by `carnoise report`
```

Every output is 48 kHz with the same number of channels as its source.

## Resuming, and changing settings

- With `resume = true` (the default), any recording whose output FLAC already exists is skipped.
  Outputs only show up once they've been checked, so a file that was interrupted halfway simply
  gets done again.
- The first run writes `run_config.json`. Running again into the same folder with different
  processing settings fails with "Output folder contains a different processing configuration". This
  is on purpose: one folder should never mix two versions of the processing. To try different
  settings, point `out_root` at a new folder and change `profile`.
- `max_files`, `other_sample_rates`, `checkpoint_every` and the paths can be changed freely between
  runs.
- Output folders started with carnoise 1.1 (profile `carnoise_v3`) can be continued with this
  version: the processing and the lock are the same.

## When something goes wrong

| What you see | What it means |
|---|---|
| `FFmpeg/FFprobe required` | FFmpeg isn't installed. |
| `Missing .../files.csv` | `data_root` is wrong, or Drive isn't mounted. |
| `FileNotFoundError` in errors.csv | The file is listed in `files.csv` but isn't on Drive. That's normal if only part of the data was copied. |
| `Car-noise DSP is built for 48000 Hz audio` | `other_sample_rates` is set to `"error"`. Change it to `"resample"` to process these files. |
| `Source contains NaN/inf` | The file is damaged. Open it in an audio editor to check. |
| `different processing configuration` | See the section above. |
| `Drive upload size mismatch` | Drive had a hiccup while copying. Run again and that file will be redone. |

## Privacy

These recordings can include people talking and where the car was. Keep Drive access limited to the
people who need it, and don't share clips without the data owner's OK.
