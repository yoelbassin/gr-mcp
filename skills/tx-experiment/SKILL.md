---
name: tx-experiment
description: Use when the user wants a closed-loop transmit-then-receive experiment in simulation — transmit a capture into a simulated device's scene and receive it back to confirm the link.
---

# TX/RX Experiment

Run a closed loop entirely in simulation: transmit a signal into a device's scene, then capture from that device and confirm the signal arrived. This is the basis for experiments (e.g. recover a transmitted tone, sanity-check a waveform).

## Method

1. **Prepare the payload capture.** Record one (`capture`), synthesize one (`render_scene` with inline elements), or `load_capture` a file. **Note its `sample_rate`** — v1.0 requires the receiving capture to use the *same* sample rate.
2. **Choose the receiving device.** Register one with simulate-scene (typically just a noise floor), or reuse an existing simulated device.
3. **Transmit — deliberately.** `transmit_capture(device_id, capture_path, freq, confirmed=True)`. `CONFIRM_TX` is on by default to catch wrong-frequency / wrong-device mistakes; set `confirmed=True` only after you've checked both. The payload becomes an `iq_file` element of the device's scene. (This change is session-scoped — it is not re-persisted to the scene file.)
4. **Receive.** `capture` the device at a center/sample_rate that covers the transmit frequency (sample rate must match the payload's), then run survey-spectrum (or build-receiver) and confirm the payload is present where you transmitted it.
5. **Report** what you transmitted (file, frequency, amplitude) and what you received (detected signal, SNR), so the experiment is reproducible.

## Judgment

- Frequency bookkeeping: the payload is placed at `freq − capture_center_freq` within the received band. Keep the transmit frequency inside the receive band.
- Match sample rates end to end, or the render will refuse with a clear error.
- TX gating is a mistake-catcher, not a licensing check — but still confirm the numbers before transmitting.
