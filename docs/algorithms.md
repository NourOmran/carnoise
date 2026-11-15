# How the processing works

There's no trained model here. It's all classic signal processing, and it runs offline, so each
step can look both backwards and forwards in time. Input is 48 kHz audio.

There are two versions, chosen with `algorithm` in the config:

- **v3 (profile `carnoise_v3`): use this one.** It came out of a full evaluation against the QE
  request: controlled mixtures with a known clean reference, half used for tuning and half held
  back for testing, plus 87 clips of real driving from all 13 units. What it changes, and why, is
  described [at the end of this page](#v3-what-changed-and-why).
- **v1 (profile `carnoise_v1`)** is the original processing, described step by step below. It's
  kept so folders already processed with it can still be resumed and compared.

A note on stereo first. Some of the older recorders write the same signal to both channels. When
the two channels are near-identical (less than -40 dB apart), the file is processed once as mono
and the result is copied back to both channels. When they really are two microphones, both
channels are kept. Every step then works out a single correction from both channels together and
applies it to each, so the left/right picture doesn't wander.

The steps run in this order:

| # | Step | What it goes after | Code |
|---|---|---|---|
| 0 | Declip | peaks that hit the recorder's limit | `dsp/declip.py` |
| 1 | Engine | the engine's tone and its harmonics | `dsp/engine.py` |
| 2 | High-pass | deep rumble | `dsp/filters.py` |
| 2b | Tonal lines | fan, alternator, electronics hum | `dsp/tonal.py` |
| 2c | Low-pass | recorder hiss at the very top | `dsp/filters.py` |
| 3 | Clicks | mount rattle | `dsp/clicks.py` |
| 4 | Background noise | road, tyres, wind, leftover engine | `dsp/noise.py` |

## 0. Declip

In these recordings, clipping usually happens when low rumble pushes the signal past full scale.
For each run of clipped samples (|x| of 0.999 or more), a smooth curve (a cubic spline) is drawn
through the good samples on either side and used to fill the gap. The repaired samples keep their
sign and never come out smaller than the clip level, because the real waveform must have gone at
least that far.

This works well for short clips. Long, heavy clipping can only be estimated.

## 1. Engine

There's no RPM signal from the car, so the audio has to act as the rev counter.

- The audio is analysed in 8192-sample windows, which gives about 5.9 Hz frequency resolution.
  That's needed because the harmonics are only 30 to 75 Hz apart.
- In each frame, the engine's base frequency is the strongest peak between 25 and 90 Hz, as long as
  it stands at least 6 dB above its surroundings. The estimate is refined between frequency bins,
  because any error gets multiplied at the higher harmonics.
- Short gaps are filled in from the neighbouring frames and the track is smoothed.
- Each harmonic from 2x up to 12x, below 900 Hz, is located (the search allows for a little
  drift) and turned down to the level of the spectrum around it, by at most 20 dB.

`engine_tracked_share` in `dsp_details.csv` is the share of the file where the engine was found.
While driving it's usually somewhere between 75 and 99%.

## 2. High-pass

The step looks at the average spectrum of the whole file and compares the bass with the
200 to 500 Hz range. Wherever the bass is more than 10 dB louder than that, it's treated as
rumble, and the filter goes a little above the point where the rumble ends. The cutoff always
stays between 70 and 200 Hz. A file that gets a cutoff near 200 Hz was dominated by rumble; Unit 4
lands at about 190 Hz.

## 2b. Tonal lines

These are steady, narrow hums that don't follow the engine: fans, the alternator, recorder
electronics. One example is the cluster of slowly drifting tones between 520 and 590 Hz on Unit 4.

- Very fine frequency resolution (about 1.5 Hz) between 150 Hz and 4 kHz, so tones only 10 Hz
  apart can be told apart.
- The spectrum is averaged over 0.7 s, so random noise evens out while a steady tone stays put.
- A frequency counts as a tone if it sticks out at least 6 dB above its neighbourhood and keeps
  showing up (allowing a little drift) in at least 60% of a 2 second window.
- Each tone found is turned down to the level around it, by at most 20 dB.

`tonal_cells_share` tells you how much of the 150 Hz to 4 kHz range got touched. It's usually 1
to 4%.

**Known problem in v1: horns (fixed in v3).** The 0.7 s averaging and the long analysis window stretch a short steady
sound out to about 1.5 s, which is enough to pass the "keeps showing up" test. So a car horn, or
anything else with a steady pitch, gets treated as a hum. I measured it with generated test sounds:

| Sound | 0.2 s long | 0.5 s | 1 s | 2 s |
|---|---|---|---|---|
| Horn-like (steady 400 Hz plus harmonics) | -11.5 dB | -12.9 dB | -12.9 dB | -13.1 dB |
| Bird-like (pitch sliding from 3 to 4.5 kHz) | 0 dB | 0 dB | 0 dB | 0 dB |

So horns and steady alarms come out roughly 12 dB quieter, while birds, sirens and voices aren't
touched, because their pitch keeps moving.

v3 fixes this by asking for much longer persistence (see below). The v1 test
`test_short_steady_horn_is_kept` in `tests/test_dsp.py` stays marked as an expected failure, because
v1 itself is left unchanged; `tests/test_v3.py` checks that v3 keeps horns.

## 2c. Low-pass

Above roughly 15 kHz, many of these recordings hold nothing but the recorder's own hiss. To find
where that starts, the file is split into 1 kHz bands from 10 kHz upwards, and the step checks how
much each band's level moves over time. A band with real sound in it goes up and down. A band
holding only hiss stays flat, within 2 dB. The filter is placed at the lowest point above which
every band stays flat. If they all move, there's no low-pass. Typical cutoffs land between 14 and
19 kHz.

## 3. Clicks (mount rattle)

The step looks for short moments where the 2 to 16 kHz range suddenly jumps to more than 5 times
(`click_k`) its typical level over the surrounding half second. In those moments, everything above
800 Hz is pulled back down to what the nearby clean moments look like (within about 85 ms either
side). Normal street sounds like doors and footsteps usually aren't sharp enough to set it off.

## 4. Background noise

This is the step that deals with road, tyre and wind noise.

- First it estimates the noise. For each frequency, it follows the quietest level over the last
  1.5 seconds or so. The car's noise floor changes slowly, while street sounds come and go on top
  of it, so the minimum mostly tracks the car.
- Then it works out, for each frequency and moment, how much louder the signal is than that noise,
  and turns it down accordingly (a Wiener filter). A small amount of smoothing stops the
  "twinkling" artefacts this kind of filter can produce.
- GPS speed makes it work harder as the car goes faster, because road and wind noise grow with
  speed. Between 0 and 100 km/h, the maximum reduction goes from 10 to 15 dB. Without a speed
  file, the gentlest setting is used.
- `floor_shift_db` moves that maximum. Negative values allow more reduction; positive values keep
  more of the street.

The maximum reduction is the reason the street still sounds like a street: no frequency ever gets
cut by more than that.

## What to expect

| Noise | Result |
|---|---|
| Recorder hiss at the top, steady hums | almost entirely gone |
| Engine, rattle, short clipping | much quieter |
| Road and tyre noise, wind on the mics | quieter, but it overlaps real street sound so some always stays |

Be careful judging results by level alone. Taking out heavy rumble can drop the overall level by
as much as 26 dB without removing any street sound. Listen to a mix of units, speeds and weather
before deciding whether something is better.

## v3: what changed and why

Every change below was measured on the same mixtures and clips, against two baselines: plain gain
and high-pass only. The full results are in the evaluation report.

**Click repair is gone.** It was the most damaging step in v1. Bird chirps, knocks and other short
street sounds look just like mount clicks to it, and on clean test events v1 took about 50 dB off
bird chirps and 8 dB off knocks. Without it, separation improved on 89% of the test mixtures.

**Noise suppression tracks the floor more slowly.** v1 followed the quietest level over about
1.5 s, which is shorter than a sentence, so during continuous speech it took the voice for noise.
v3 uses about 3.2 s and a smaller bias (1.4 instead of 1.8). It also really gets GPS speed now:
v1 looked for a `<name>_speed.csv` next to each recording, which this dataset doesn't have, so every
file ran at the "stopped" setting. v3 reads `gps/<unit>_gps.csv` and lowers the floor from -12 dB
when stopped to -18 dB at 100 km/h and above.

**Horn-safe tonal lines.** A line now has to be present for at least 85% of a 10-second window
(v1: 60% of 2 seconds), with up to 9 Hz of drift allowed. Fans and alternators hum for minutes and
are still cut by about 16 dB. Horns of up to 4 seconds lose at most 0.6 dB (v1: 11-13 dB).

**Wind removal with two microphones** (stage 3, `dsp/wind.py`). Wind is turbulence right at each
mic, so the two mics disagree, while sound from the street reaches both in step. Below 600 Hz, each
mic is pulled down to the other mic's level, and anything incoherent between them is reduced, by
12 dB at most. The stage only switches on while the mics' low-frequency coherence is low, meaning
when there's wind, and does nothing the rest of the time. On held-out wind clips it removes about
3 dB more car noise in 100 Hz-1 kHz. Recordings with one mic or copied channels skip it.

**Optional engine notch** (`engine_notch = true`). The engine's pitch is found from its whole
harmonic series rather than the strongest low peak, with checks so a voice can't pass as an engine,
and every harmonic including the fundamental is notched, up to 1.5 kHz. On real drives it takes the
engine lines down about 11 dB further. It costs some separation (median SI-SDR improvement on the
held-out mixtures goes from +13.4 to +11.7 dB), because it also notches whatever street sound happens to sit on an engine harmonic,
so it's off by default. Try it on the listening files and decide by ear.

**What stays the same:** declipping, the automatic high-pass and low-pass, and the way stereo and
copied-channel files are handled.

**Chunked processing.** Long files are still processed in about 44-second chunks with context on
both sides. For v3, chunked output matches whole-file output to within -50 dB or better (v1:
-150 dB), well below anything audible.

### Results in short

On held-out mixtures (not used for tuning), median SI-SDR improvement:

| | Median SI-SDR improvement |
|---|---|
| High-pass only | +7.2 dB |
| v1 | +6.2 dB |
| v3 | +13.4 dB |

On real drives, street sounds stand out of the car's noise floor by about 10 dB more with v3,
against under 2 dB with v1. Every type of test event (horn, siren, birds, dog, passing car, knock)
keeps its level.
