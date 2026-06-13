---
name: debug-no-signal
description: Use when a receiver runs but produces no output, noise, or the wrong result — systematically locate the fault instead of guessing, using render and measure to see the problem.
---

# Debug No Signal

A receiver that runs `status: ok` but produces silence or garbage has a DSP fault, not a crash. Find it systematically; do not randomly tweak parameters.

## Method (systematic-debugging applied to RF)

1. **Reproduce and name the symptom.** Re-run and state precisely what's wrong: silent audio, broadband noise, a tone at the wrong pitch, low level.
2. **See, don't guess.** Render a `spectrogram` (and/or `psd_plot`) of the *channelized* capture — the stage right after `freq_xlating_lowpass` — and `measure` it. Most faults are visible there.
3. **Check hypotheses cheapest-first, changing ONE thing per iteration:**
   - **Wrong offset or sign.** After channelizing, the target should sit at 0 Hz. If it's offset by ±X, your `center_offset` value or its sign is wrong (`center_offset = target_freq − capture_center_freq`).
   - **Over/under-decimation.** The post-decimation rate must be ≥ ~2× the signal bandwidth, and for FM must equal the demodulator's `quad_rate`. Check `sample_rate / decimation`.
   - **Filter too narrow/wide.** `cutoff` below the signal bandwidth clips it (muffled/silent); far too wide lets in neighbors (noisy).
   - **Levels / wrong channel.** `measure` the channelized capture: very low SNR means you channelized empty spectrum — re-check the target frequency from survey-spectrum.
   - **Demod/rate mismatch.** For `nbfm_rx`, `quad_rate` must be an integer multiple of `audio_rate`; a `wav_sink` `sample_rate` must match the audio stream rate.
4. **Re-verify after each change** the same way build-receiver verifies. Stop when the output is correct.
5. **If no vocabulary block fixes it** (e.g. the signal isn't FM at all), report the gap → escape-hatch.

## Judgment

- One change at a time. If you change three params and it works, you've learned nothing and it'll break next time.
- Trust the spectrogram over your mental model of the math.
