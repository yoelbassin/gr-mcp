# Marconi

**LLM-driven RF control and analysis — "Claude Code for RF."**

Marconi lets you drive software-defined radio workflows in natural language: describe what you want, and an AI agent builds the signal chain, runs it, looks at the spectrum, and iterates — leaving behind a reproducible RF project you own (SigMF captures, YAML pipelines, exported `.grc` flowgraphs).

> **Status: early development.** v1.0 is simulation-only and ships as a Python library, with a Claude Code plugin to follow. Hardware receive and transmit come after. Interfaces will change. This repository began as the `gr-mcp` proof of concept and is being rebuilt as Marconi.

## What it does today

- **Capture & ingest** IQ data — from simulated devices, plus SigMF / `.cf32` / `.wav` files via `load_capture`.
- **Analyze** — power spectral density, signal detection, windowed measurement (SNR, occupied bandwidth), burst detection. Structured numbers an agent can reason over.
- **Render** — spectrograms, PSD plots, and constellations as PNGs an agent reads with vision (and a human reads directly).
- **Build & run pipelines** — an engine-agnostic block vocabulary compiled to a GNU Radio flowgraph and run under a timeout watchdog, returning structured results.
- **Simulate** — describe a scene (an FM voice at 100.3 MHz, a carrier at 99.5 MHz, a noise floor) and capture from it exactly as you would from real hardware.
- **Export** any pipeline to a `.grc` file to open and tweak in GNU Radio Companion.

## Architecture

Marconi is an engine-agnostic operations layer (the `marconi` package) with GNU Radio as the first execution backend. The library is the product's portable core; a Claude Code plugin (MCP server + skills) is one front-end, and a standalone product is possible later without rewriting the core. See `docs/superpowers/specs/` for the full design.

## Requirements

- Python ≥ 3.13
- [`uv`](https://docs.astral.sh/uv/)
- [GNU Radio](https://www.gnuradio.org/) 3.10+ installed system-wide — needed to *run* pipelines and *simulate*. Analyzing existing IQ files does not require it.

## Quick start

```bash
git clone https://github.com/yoelbassin/gr-mcp
cd gr-mcp
uv sync
uv run pytest
```

```python
import marconi

ws = marconi.Workspace("my_project")

# Describe what's on the air, then capture from it like a real radio.
scene = marconi.SceneSpec(name="demo", elements=[
    marconi.SceneElement(kind="fm_tone", freq=100.3e6, params={"mod_freq": 1e3}),
    marconi.SceneElement(kind="tone", freq=99.5e6, amplitude=0.4),
    marconi.SceneElement(kind="noise", amplitude=0.01),
])
marconi.add_simulated_device("sim0", scene)

cap = marconi.capture("sim0", center_freq=100e6, sample_rate=2e6,
                      duration=0.25, workspace=ws)
for s in marconi.find_signals(cap):
    print(f"{s.center_freq/1e6:.3f} MHz  bw {s.bandwidth/1e3:.1f} kHz  SNR {s.snr_db:.0f} dB")
```

## Use it from Claude Code

Marconi ships as a Claude Code plugin: an MCP server exposing the operations as
tools, plus skills that carry the RF workflow.

```bash
# add this repo as a plugin marketplace, then install the marconi plugin
/plugin marketplace add yoelbassin/gr-mcp
/plugin install marconi
```

The plugin starts the `marconi-mcp` server (via `uv run --extra mcp marconi-mcp`)
and loads six skills: **survey-spectrum**, **build-receiver**, **debug-no-signal**,
**simulate-scene**, **tx-experiment**, and **escape-hatch** (last resort). Ask in
natural language — "simulate an FM station at 100.3 MHz and build me a receiver"
— and the agent surveys, builds, runs, verifies, and leaves the pipeline/capture
artifacts in your workspace.

The MCP server's workspace is its working directory; override with the
`MARCONI_WORKSPACE` environment variable.

## License

See [LICENSE](LICENSE).
