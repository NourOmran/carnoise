# How the processing works

There's no trained model here. It's all classic signal processing, and it runs offline, so each
step can look both backwards and forwards in time. Input is 48 kHz audio.

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

**Known problem: horns.** The 0.7 s averaging and the long analysis window stretch a short steady
sound out to about 1.5 s, which is enough to pass the "keeps showing up" test. So a car horn, or
anything else with a steady pitch, gets treated as a hum. I measured it with generated test sounds:

| Sound | 0.2 s long | 0.5 s | 1 s | 2 s |
|---|---|---|---|---|
| Horn-like (steady 400 Hz plus harmonics) | -11.5 dB | -12.9 dB | -12.9 dB | -13.1 dB |
| Bird-like (pitch sliding from 3 to 4.5 kHz) | 0 dB | 0 dB | 0 dB | 0 dB |

So horns and steady alarms come out roughly 12 dB quieter, while birds, sirens and voices aren't
touched, because their pitch keeps moving.

The fix I have in mind is to judge how long a tone lasts before the averaging, and to require it to
last at least 3 seconds. A fan or alternator hums for minutes, so those would still be caught. A
horn lasting a second or two wouldn't be. This changes the output, so it will go in as a new
profile. The test `test_short_steady_horn_is_kept` in `tests/test_dsp.py` is marked as an expected
failure for now and will start passing once it's fixed. Until then, if horns matter, turn this step
off with `remove_tonal_lines = false`.

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
