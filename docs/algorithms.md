# How the processing works

There's no trained model here. It's all classic signal processing, and it runs offline, so each
step can look both backwards and forwards in time. Input is 48 kHz audio. Output files carry the
profile name `carnoise_v3`.

The settings were chosen in a full evaluation against the client's QE request: controlled mixtures
with a known clean reference (half used for tuning, half held back for testing), plus 87 clips of
real driving from all 13 units. The results are [at the end of this page](#results).

**The core idea.** The car's own noise is *steady* and changes with *speed*. Street events (horns,
sirens, voices, animals, passing traffic) are *short rises* on top of it. Most steps follow the
steady part and leave the rises alone. Every step has a limit on how much it may remove, so a
street sound is never erased completely.

**Stereo.** Some recorders write the same signal to both channels. When the two channels are
near-identical (less than -40 dB apart), the file is processed once as mono and the result is copied
back to both channels. When they really are two microphones, both channels are kept, and the main
noise step works out a single correction from both channels together and applies it to each, so the
left/right picture doesn't wander.

The steps run in this order:

| # | Step | What it goes after | Code |
|---|---|---|---|
| 0 | Declip | peaks that hit the recorder's limit | `dsp/declip.py` |
| 1 | Engine notch (optional, off by default) | the engine's harmonics | `dsp/engine.py` |
| 2 | High-pass | deep rumble | `dsp/filters.py` |
| 2b | Tonal lines | fan, alternator, electronics hum | `dsp/tonal.py` |
| 2c | Low-pass | recorder hiss at the very top | `dsp/filters.py` |
| 3 | Wind | wind hitting the microphones | `dsp/wind.py` |
| 4 | Background noise | road, tyres, leftover wind and engine | `dsp/noise.py` |

## 0. Declip

In these recordings, clipping usually happens when low rumble pushes the signal past full scale.
For each run of clipped samples (|x| of 0.999 or more), a smooth curve (a cubic spline) is drawn
through the good samples on either side and used to fill the gap. The repaired samples keep their
sign and never come out smaller than the clip level, because the real waveform must have gone at
least that far.

This works well for short clips. Long, heavy clipping can only be estimated.

## 1. Engine notch (optional)

There's no RPM signal from the car, so the audio has to act as the rev counter. Off by default; turn
it on with `engine_notch = true`.

- The audio is analysed in 8192-sample windows (about 5.9 Hz resolution), because the harmonics are
  only 25 to 90 Hz apart.
- On these microphones the engine's fundamental is often weak, so the pitch is found from the whole
  harmonic series: every candidate between 25 and 90 Hz is scored by how far its first 10 harmonics
  stand above their surroundings.
- Three checks stop voices and hums from passing as an engine: the best candidate must clearly beat
  the rest, its low harmonics must be strong, and its pitch must move smoothly for at least six
  frames.
- Every harmonic, including the fundamental, up to 1.5 kHz is turned down to the level of the
  spectrum around it, by at most 30 dB.

**Why it's off by default.** Engine lines are only 2 to 3 dB above the noise in this dataset. The
notch takes them down about 11 dB further on real drives, but it also notches whatever street sound
happens to sit on an engine harmonic, so overall separation drops a little (median SI-SDR improvement
+13.4 dB -> +11.7 dB). Decide by ear. `engine_tracked_share` in `dsp_details.csv` shows how much of a
file had an engine found (0 when the notch is off).

## 2. High-pass

The step looks at the average spectrum of the whole file and compares the bass with the
200 to 500 Hz range. Wherever the bass is more than 10 dB louder than that, it's treated as
rumble, and the filter goes a little above the point where the rumble ends. The cutoff always
stays between 70 and 200 Hz. A file that gets a cutoff near 200 Hz was dominated by rumble.

## 2b. Tonal lines (horn-safe)

These are steady, narrow hums: fans, the alternator, recorder electronics. One example is the
cluster of slowly drifting tones between 520 and 590 Hz on Unit 4. Horns, sirens and alarms are
tonal too, but they last seconds, not minutes, and that difference is what this step uses.

- Very fine frequency resolution (about 1.5 Hz) between 150 Hz and 4 kHz, so tones only 10 Hz
  apart can be told apart.
- The spectrum is averaged over 0.35 s, so random noise evens out while a steady tone stays put.
- A frequency counts as a hum if it sticks out at least 6 dB above its neighbourhood and keeps
  showing up (allowing up to 9 Hz of drift) in at least **85% of a 10-second window**.
- Each hum found is turned down to the level around it, by at most 20 dB.

On test sounds, horns of 0.5 to 4 seconds lose at most 0.6 dB, while steady and slowly drifting fan
tones are cut by about 16.5 dB. `tonal_cells_share` tells you how much of the 150 Hz to 4 kHz range
got touched; it's usually well under 2%.

## 2c. Low-pass

Above roughly 15 kHz, many of these recordings hold nothing but the recorder's own hiss. To find
where that starts, the file is split into 1 kHz bands from 10 kHz upwards, and the step checks how
much each band's level moves over time. A band with real sound in it goes up and down. A band
holding only hiss stays flat, within 2 dB. The filter is placed at the lowest point above which
every band stays flat. If they all move, there's no low-pass. Typical cutoffs land between 14 and
21 kHz.

## 3. Wind

Wind is turbulence right at each microphone, so the two mics disagree: it's incoherent between them
and often much louder at one mic. Sound from the street reaches both mics in step and at a steady
level ratio.

- Below 600 Hz (fading out by 1 kHz), each mic is pulled down to the other mic's level, after
  allowing for their normal level ratio (a running median over about 5 s). That removes a gust on
  one mic.
- Anything incoherent between the two mics is reduced as well.
- At most 12 dB is removed.
- **It only switches on when there's wind**, meaning the mics' 40-400 Hz coherence drops below 0.6.
  The rest of the time it does nothing at all (calm audio changes by less than 0.2 dB in the tests).

Recordings with one mic, or with the same signal on both channels, skip this step.

## 4. Background noise

This is the main step. It deals with road and tyre noise and whatever is left of wind and engine.

- First it estimates the noise. For each frequency, it follows the quietest level over the last
  **3.2 seconds** or so (and adds a small correction, because the quietest level sits a little below
  the average noise). The car's noise floor changes slowly, while street sounds come and go on top
  of it, so the minimum mostly tracks the car. The window is long enough that continuous speech isn't
  taken for noise.
- Then it works out, for each frequency and moment, how much louder the signal is than that noise,
  and turns it down accordingly (a Wiener filter). A small amount of smoothing stops the
  "twinkling" artefacts this kind of filter can produce.
- **GPS speed** makes it work harder as the car goes faster, because road and wind noise grow with
  speed. Between 0 and 100 km/h the maximum reduction goes from **12 to 18 dB**. Speed comes from
  `gps/<unit>_gps.csv`; without it, the gentlest setting is used.
- `floor_shift_db` moves that maximum. Negative values allow more reduction; positive values keep
  more of the street.

The maximum reduction is the reason the street still sounds like a street: no frequency ever gets
cut by more than that.

## Chunked processing

Long files are processed in about 44-second chunks with about 7 seconds of context on both sides,
so memory use stays flat. Chunked output matches whole-file output to within -50 dB or better, far
below anything audible.

## What to expect

| Noise | Result |
|---|---|
| Recorder hiss at the top, steady hums | almost entirely gone |
| Wind on the mics (stereo recordings) | clearly reduced while it blows, untouched otherwise |
| Engine | reduced; more with `engine_notch` |
| Road and tyre noise | quieter, but it overlaps real street sound, so some always stays |

Be careful judging results by level alone. Taking out heavy rumble can drop the overall level a lot
without removing any street sound. Listen to a mix of units, speeds and weather before deciding
whether something is better, and compare at matched loudness.

## Results

From the QE evaluation. Mixtures are real car noise from the dataset plus known clean sounds;
"held-out" means noise recordings that were never used for tuning.

| Measure | High-pass only | carnoise_v3 |
|---|---|---|
| Median SI-SDR improvement, held-out mixtures | +7.2 dB | **+13.4 dB** |
| Event detection F1 | 0.74 | **0.80** |
| Level change of street events (horn, siren, birds, dog, passing car, knock) | | **0.0 dB, recall ≥ 0.99** |
| Damage on clean clips (no car noise) | | -0.09 dB |
| Street sounds standing out of the noise, real drives | | **+10 dB**, every unit improves |

SI-SDR ignores overall level, so making audio quieter scores zero: only real separation counts.
