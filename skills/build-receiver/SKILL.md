---
name: build-receiver
description: Use when the user wants to receive or demodulate a specific signal — build, validate, run, and verify a receiver pipeline, then save it as a reusable, GRC-openable flowgraph.
---

# Build Receiver

Turn a target signal into a working receiver. The method below is general; FM voice and an unmodulated carrier are shown as two worked examples so the *shape* transfers to modulations the demo doesn't cover.

## General method (any modulation)

1. **Know the target.** Get its center frequency and occupied bandwidth from survey-spectrum / `measure`. Compute the offset from the capture's center: `center_offset = target_freq − capture_center_freq`.
2. **Discover the blocks.** Call `list_blocks`. Compose only from these types; check each block's input/output dtype ('c' complex, 'f' float) so connections match.
3. **Channelize.** Use `freq_xlating_lowpass` to shift the target to baseband and decimate. Choose `decimation` so the post-decimation rate is ~2–3× the signal bandwidth; set `cutoff ≈ signal_bandwidth` and `transition ≈ cutoff/2`.
4. **Demodulate to match the modulation** (see examples below).
5. **Validate.** `validate_pipeline(pipeline)` and fix *every* issue (each names the block and field) before running.
6. **Run.** `run_pipeline(pipeline)`; check `status == "ok"`.
7. **Verify before you claim success** (see Verify). Never report a working receiver without evidence.
8. **Persist.** `save_pipeline(pipeline)` and `export_grc(pipeline)` so the user keeps a reusable receiver they can open in GNU Radio Companion.

## Worked example A — NBFM voice

Channelize so the post-decimation rate equals the FM quad rate (a multiple of the audio rate), then demodulate to a WAV:

- `freq_xlating_lowpass` (decimation chosen so e.g. 2 MHz / 20 = 100 kHz = quad_rate; cutoff ≈ 8 kHz, transition ≈ 4 kHz)
- `nbfm_rx` with `audio_rate=25000`, `quad_rate=100000` (**quad_rate must be an integer multiple of audio_rate**)
- `wav_sink` (`sample_rate=25000`)

Connections: src → chan → demod → audio.

## Worked example B — unmodulated carrier / CW (non-FM)

There is *no* demodulator here — receiving a carrier means channelizing it to baseband and confirming it:

- `freq_xlating_lowpass` (narrow `cutoff`, e.g. a few kHz) → `file_sink`
- Then `measure(channelized_capture, center_freq=0)` (or render a `spectrogram`) to confirm the carrier sits at baseband with the expected SNR.

This proves the channelizer/offset logic independently of any demodulator — the same skeleton as example A with the demod stage removed.

## Other modulations

AM, SSB, and digital (OOK/PPM/FSK) demodulators are **not in the v1.0 vocabulary**. If the target needs one, report the gap and switch to the escape-hatch skill — do not force an FM demodulator onto a non-FM signal.

## Verify (mandatory)

- FM voice: read the recovered WAV and confirm energy in the audio band (e.g. a 1 kHz modulating tone shows up at 1 kHz).
- CW/other: `measure` or `spectrogram` the channelized output and confirm the carrier/structure is where you expect.
- If verification fails → switch to the debug-no-signal skill. Do not retry blindly.
