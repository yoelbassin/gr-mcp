---
name: survey-spectrum
description: Use when the user wants to know what signals are present — characterize a capture or a simulated device's spectrum (PSD, signal detection, spectrogram) and summarize what's on the air.
---

# Survey Spectrum

Characterize what's present in a capture, then summarize it for the user. This is the first step of almost every RF task ("what's here?").

## Method

1. **Get a capture.** From a simulated device: `capture(device_id, center_freq, sample_rate, duration)`. From a file the user brings: `load_capture(path)`. Use the returned `path` for everything below.
2. **Read the numbers.** `psd(capture_path)` → noise floor + strongest peaks. `find_signals(capture_path)` → candidate signals (center_freq, bandwidth, snr_db).
3. **Always look, don't just count.** Render `spectrogram(capture_path)` and read the image. `find_signals` is a first pass over a 1-D PSD; the spectrogram shows time/frequency structure (bursts, drift, multiple carriers) that the list can miss.
4. **Confirm wideband signals with `measure`.** A wide signal (e.g. an FM carrier) with a low noise floor can *fragment* into several `find_signals` detections. For any wide or clustered group, call `measure(capture_path, center_freq=...)` to get the true 99% occupied bandwidth, and trust the spectrogram over the raw detection count.
5. **Summarize.** Give the user a short table: frequency, bandwidth, SNR, and a guess at type (narrowband tone vs. wideband FM-like vs. bursty), plus the spectrogram path.

## Judgment

- **Seed a noise floor when simulating.** A noiseless tone has phantom sidebands that register as spurious detections. When you build a scene to survey (see simulate-scene), always include a small `noise` element (amplitude ≈ 0.005).
- **SNR gate.** Treat a `measure` result as a real signal only above ~8 dB SNR; below that you're likely measuring noise.
- **Bandwidth is threshold-crossing extent**, not −3 dB or OBW — use `measure` when you need a real bandwidth number.
