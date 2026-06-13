---
name: simulate-scene
description: Use when the user wants a test signal or a simulated RF environment with no hardware — translate intent into a scene and register it as a persistent simulated device.
---

# Simulate Scene

Create a simulated device whose "on-air" contents you control. This is the no-hardware on-ramp, and a real workflow in its own right: prototype a receiver against known ground truth before going on-air, or reproduce a colleague's scenario without their antenna.

## Method

1. **Translate intent into elements.** Each element is `{kind, freq, amplitude, params}`:
   - `tone` — an unmodulated carrier at `freq`.
   - `fm_tone` — an NBFM voice carrier; `params.mod_freq` is the modulating audio tone (e.g. 1000 for 1 kHz). Renders only when the capture `sample_rate` is a multiple of 100000.
   - `noise` — a noise floor (`freq` ignored).
   - `iq_file` — replay a recorded capture (usually created via transmit, not by hand).
2. **Always add a noise floor.** Include a `noise` element (amplitude ≈ 0.005). A noiseless scene produces analysis artifacts (phantom sidebands) that confuse survey-spectrum.
3. **Register it.** `simulate_scene(device_id, elements)`. The scene is saved to `scenes/<device_id>.yaml` and the device reappears in later sessions — it's a shareable, git-friendly fixture.
4. **Verify the scene is what you intended.** `capture` the device and run survey-spectrum; confirm the signals land where you placed them.

## Judgment

- Keep amplitudes sane: a strong carrier ≈ 1.0, a weaker one ≈ 0.3–0.5, noise ≈ 0.005. Huge dynamic range makes weak signals invisible.
- Place signals comfortably inside the captured band — elements beyond ±45% of the sample rate from the center are skipped at render time.
- Name devices for what they represent (`fm_band`, `ism_test`), since the name becomes the scene filename.
