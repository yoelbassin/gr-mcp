# Marconi — Design Spec

**Date:** 2026-06-12
**Status:** Approved design, pre-implementation
**Working name:** Marconi (final name/PyPI availability to be checked before first release)

## Vision

An LLM-driven assistant for RF work — "Claude Code for RF." RF engineers and
researchers describe what they want in natural language; the agent builds,
runs, observes, and iterates on real signal chains, and leaves behind a
reproducible RF project the user owns.

Claude Code creates code; **Marconi creates reproducible RF experiments.**

This repo's previous content (gr-mcp) was the proof of concept. Marconi is the
product, built fresh in the same repository (which keeps its existing traffic
and stars). Fully open source.

## Product decisions

| Decision | Choice |
|---|---|
| First target user | SDR experimenters / researchers ("build → run → observe → iterate"); other RF domains (signal analysis, T&M/SCPI) follow later |
| Delivery vehicle | Claude Code plugin first; architecture must allow a future standalone platform without rewrites |
| Hardware scope | Simulation + RX + TX. **v1.0 is simulation-only**; hardware is additive later |
| Perception | Structured analysis tools (numbers) + rendered images read via vision |
| Execution engine | Engine-agnostic operations API; GNU Radio is the first backend. The backend interface is *extracted* from what the GNU Radio backend needs, never invented speculatively |
| Demo path | v1.0 sim "find the hidden signals" → v1.1 live-air full loop → v1.2 TX/RX experiment automation |
| License/distribution | Fully open source, this repo |

## Architecture

```
┌──────────────────────────────────────────────────────────────┐
│  LAYER 3 — Claude Code plugin (the agentic product)          │
│  skills: survey-spectrum, build-receiver, verify-signal-     │
│          chain, debug-no-signal, escape-hatch (last resort)  │
│  This is where the RF judgment lives — the differentiator.   │
└──────────────────────────────┬───────────────────────────────┘
                               │ invokes tools
┌──────────────────────────────▼───────────────────────────────┐
│  LAYER 2 — MCP server (FastMCP, deliberately thin)           │
│  ~18 tools, one per operation. Marshalling only, no logic.   │
│  Swappable frontend: a future standalone platform replaces   │
│  this layer and nothing else.                                │
└──────────────────────────────┬───────────────────────────────┘
                               │ imports
┌──────────────────────────────▼───────────────────────────────┐
│  LAYER 1 — marconi, the core Python package                  │
│  Engine-agnostic part (no gnuradio imports):                 │
│   • models.py — pydantic: Device, Capture, Signal,           │
│     Pipeline, Scene, RunResult                               │
│   • ops/ — devices, capture, analyze, render, pipeline,      │
│     simulate, transmit                                       │
│   • analyze/render implemented here (numpy/scipy/            │
│     matplotlib on IQ files — no DSP engine needed)           │
│   • workspace — the user's project directory (see below)    │
│   • vocabularies — curated block specs & scene elements      │
│  backends/base.py — the swap boundary:                       │
│    enumerate_devices · run_pipeline · capture_to_file ·      │
│    render_scene_to_file · transmit                           │
│  backends/gnuradio/ — first backend, the "sample engine"     │
└──────────────────────────────────────────────────────────────┘
```

### Layer roles

- **Layer 1 is the portability anchor.** All logic lives here. When the
  standalone platform is built, it imports `marconi` directly; only Layer 2
  is replaced.
- **Layer 3 is the product's soul.** Skills encode RF judgment (verify the
  PSD before declaring success, how to pick gain, how to debug a silent
  receiver). They are first-class code: versioned, and tested via eval
  scenarios (see Testing).
- **The backend boundary is "moves samples or touches devices."** Pipeline
  execution, capture, transmit, scene rendering, and device enumeration are
  backend work. Analysis, rendering, models, vocabularies, and the workspace
  sit above the boundary and never import GNU Radio.

### GNU Radio backend specifics

- Builds flowgraphs **directly via GNU Radio's Python API**
  (`gr.top_block()`, block construction, `connect()`). No `.grc` files or
  `grcc` compilation in the run loop — `.grc` is a *human export format*
  (`export_grc`) so users can open the agent's work in GNU Radio Companion.
- Hardware I/O uses GNU Radio's in-tree blocks (`gr-soapy`, `gr-uhd`);
  device enumeration uses the SoapySDR Python module that gr-soapy already
  depends on. **No dependency beyond GNU Radio itself**, and none of these
  names appear above the backend boundary.

### Simulation is a product feature, not test scaffolding

A `SimulatedDevice` is configured with a **scene** — a declarative
description of what is "on the air" (e.g., NBFM voice at 100.3 MHz at
−40 dBFS, GFSK bursts at 100.8 MHz, noise floor −90 dB, AWGN channel).
Every operation then works identically against simulated and real devices:
`capture()` from a simulated device produces real IQ of the scene;
`analyze`/`render` don't know it's synthetic; `transmit` to a simulated
device writes into its scene. The GNU Radio backend implements scene
rendering with signal-source/channel-model blocks.

Consequences:
- v1.0 ships with zero hardware and GNU Radio as the sole dependency.
- Real SDRs later are *additive* — more devices behind the same interface;
  no skill or workflow changes.
- Scenes with known ground truth double as the skill-eval harness, and
  tests consume simulation as a fixture factory (tests use the feature; the
  feature does not live in tests).
- Simulation is also a genuine RF workflow in its own right: prototype a
  receiver against a known synthetic signal before going on-air; reproduce
  a colleague's scenario without their antenna setup.

### The workspace is the user's project directory

Not a hidden cache. A session leaves behind a git-friendly RF project in
open formats, useful without Marconi:

- `pipelines/*.yaml` — engine-agnostic pipeline specs (human-readable,
  diffable) plus `.grc` exports that run on stock GNU Radio
- `captures/*` — IQ recordings with SigMF metadata (open standard; usable
  in inspectrum, GNU Radio, anyone's tooling)
- `scenes/*.yaml` — reproducible simulated environments, shareable like
  test fixtures
- `renders/`, reports — spectrograms, PSD plots; later, experiment sweeps
  with data + plots + write-up
- scripts — ordinary Python importing `marconi`; re-run standalone

Tools exchange **references** (paths/IDs + summary stats), never sample
blobs, keeping MCP responses small and sessions resumable.

### Escape hatch — last resort

When no operation fits, the agent may write a Python script that
`import marconi` and run it via Bash. Skills frame this explicitly as last
resort: try the operations API first; when escaping, report the gap. The
gaps are the roadmap for new operations.

## Operations surface (v1)

Eighteen MCP tools, one per operation:

| Family | Operations | Notes |
|---|---|---|
| devices | `list_devices` | Devices with capabilities (freq range, rates, TX?); includes `SimulatedDevice` |
| capture | `capture(device, center_freq, sample_rate, duration, gain)`, `load_capture(path)` | Returns `CaptureRef`. `load_capture` ingests `.cf32`/SigMF/`.wav` — analyzing someone else's file is a day-one workflow |
| analyze | `psd`, `find_signals`, `measure`, `detect_bursts` | Structured numbers only (peaks, noise floor, SNR, bandwidth, modulation hints) |
| render | `spectrogram`, `psd_plot`, `constellation` | PNGs for vision + the human |
| pipeline | `create_pipeline`, `validate`, `run`, `stop`, `export_grc` | `run` is bounded (duration/timeout) and returns `RunResult` |
| simulate | `create_scene`, `set_scene` | Scene vocabulary: tones, modulated audio, bursts, noise, channel impairments |
| transmit | `transmit_capture(device, capture, freq)` + TX pipelines | Gated by `confirm_tx: true` by default (catches agent mistakes, not licensing policing); user-switchable |

### Vocabulary: compositional, not protocol-enumerating

Three tiers:

1. **Primitives** — cleanly-schematized mappings of GNU Radio core blocks:
   sources, sinks, filters, resamplers, AGC, `quadrature_demod`, clock
   recovery, slicers. Protocol receivers are *composed* from these by the
   agent, guided by skill recipes.
2. **Named compositions** — sugar where GNU Radio ships a hier block
   (`nbfm_rx`, `wfm_rcv`, …): faster and less error-prone, never the only
   path.
3. **Everything else** — composed from primitives; missing primitives →
   escape hatch + reported gap.

A successful composition is a durable artifact: a working pipeline YAML in
the user's project, shareable and promotable into tier 2 (or a community
template collection) without new code. Protocol coverage grows by
accumulation, not by pre-building a function per protocol. Curation means
"which GR blocks get schemas the model can use reliably," not
one-tool-per-protocol. The curated set deliberately excludes most of GNU
Radio's ~400 blocks to keep the model's working vocabulary reliable.

## Data flow — v1.0 demo scenario

"Three unknown signals — find them, demodulate the FM one, play me the
audio":

1. `create_scene` — NBFM voice @ 100.3 MHz, GFSK bursts @ 100.8, bare
   carrier @ 99.5, noise floor −90 dB → `scenes/three_signals.yaml`
2. `list_devices` → `sim0` carrying that scene
3. `capture(sim0, center=100e6, rate=8e6, duration=2s)` → backend renders
   scene to IQ → `captures/scan_100M.sigmf-data` + sidecar → `CaptureRef`
   with summary stats
4. `psd` + `find_signals` → three carriers with freq/power/bandwidth;
   agent cross-checks via `spectrogram` (vision)
5. `measure(signal @ 100.3M)` → ~12 kHz FM-like → decide NBFM
6. `create_pipeline`: `sim_source → freq_xlating_filter → nbfm_demod →
   wav_sink`; `validate`; `run(duration=5s)`
7. `RunResult` → agent **verifies** (speech-band energy in the demodulated
   audio) before reporting; steps 4–7 loop on failure

The identical flow at v1.1 uses `rtl0` instead of `sim0` — that is the
point of the device abstraction.

## Error handling

- **Errors are for the agent first.** Validation failures return
  structured, actionable items with block ID, parameter, expected
  type/range (e.g. "`nbfm_demod.quad_rate` must be an integer multiple of
  `audio_rate`; got 48000/44100"). Backend exceptions are translated into
  vocabulary terms at the boundary; raw GR tracebacks are attached, never
  the primary message. The agent's self-correction loop is only as good as
  the error text.
- **Runs are bounded and supervised.** Every `run` has a timeout; runs
  yield exit status, stdout/stderr, and artifacts even on failure; `stop`
  always works (process-level kill as backstop). Device-busy and
  underrun/overrun become structured warnings, not silent sample loss.
- **TX gated by default** (`confirm_tx: true`, switchable) — catches agent
  mistakes (wrong frequency), not licensing enforcement (licensed users
  hold their own authority).

## Testing

1. **Unit (no GNU Radio):** models, vocabularies, workspace, analyze/render
   against golden IQ fixtures. Per-commit CI, runs anywhere.
2. **Backend integration (requires GNU Radio):** scene→IQ correctness,
   pipeline construction for every vocabulary block, run/stop/timeout.
3. **End-to-end:** operation chains through the MCP layer (FastMCP
   in-process client), sim-only.
4. **Skill evals:** scenes with ground truth + scored scenarios ("agent
   finds all 3 signals", "agent recovers the audio"). Release-time suite
   (needs API credits), not per-commit. This is what makes Layer 3
   first-class code.

## Phasing

| Phase | Scope | Demo |
|---|---|---|
| v1.0 | Simulation only; full ops surface minus hardware; GNU Radio sole dependency | "Find the hidden signals" (sim) |
| v1.1 | RX hardware via GR in-tree blocks (no new installs) | Live-air full loop: find strongest FM station, build receiver, play audio |
| v1.2 | TX on real hardware | TX/RX experiment automation (e.g., BER vs SNR sweep with report) |

Later domains (signal analysis/RE deep-dive, T&M/SCPI) arrive as additional
operation families / tool packs beside these — the platform emerges from
the wedge rather than being built first.

## Out of scope for v1

- Standalone platform / own agent UI (architecture allows it; not built now)
- SCPI/VISA instrument control
- Real-time agent interaction with a *running* flowgraph beyond run/stop
  (parameter tweaking mid-run is a candidate for v1.x)
- Import of arbitrary existing `.grc` files (export only in v1; import is a
  candidate later)
- Exposing GNU Radio's full block set (curated vocabulary only, by design)
