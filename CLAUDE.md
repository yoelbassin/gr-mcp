# Marconi Developer Guide

Marconi is an LLM-driven toolkit for RF control and analysis — "Claude Code for RF." It exposes RF operations (capture, analyze, render, build/run signal pipelines, simulate, transmit) as an engine-agnostic Python library, with GNU Radio as the first execution backend. The near-term delivery is a Claude Code plugin; the library is designed so a future standalone product can reuse it unchanged.

Full design: `docs/superpowers/specs/2026-06-12-marconi-design.md`. Implementation plans: `docs/superpowers/plans/`. (This repo began as the `gnuradio_mcp` proof of concept; that POC has been removed in favor of the Marconi rebuild.)

## Architecture

Three layers, strict dependency direction — each layer only knows the one below it:

1. **Core package `src/marconi/` (engine-agnostic — no `gnuradio` imports).** Models, operations, analysis/rendering, the workspace, the curated block vocabulary. This is the portability anchor: a future standalone product imports it directly.
2. **MCP server (thin adapter — Plan 3, not yet built).** One tool per operation; marshalling only, no logic.
3. **Claude Code plugin: skills + manifests (Plan 3, not yet built).** Encodes RF workflows and judgment.

The backend boundary is "moves samples or touches devices": everything under `src/marconi/backends/` may import GNU Radio; nothing else may, and even there the import is lazy (inside functions) so `import marconi` works on machines without GNU Radio.

## Module layout

```
src/marconi/
  models.py             # pydantic: CaptureRef, PSD/Signal results, Pipeline/Scene/Device/Run specs
  sigmf.py              # SigMF read/write; read_samples is the single sample reader
  workspace.py          # the user's project dir: captures/ renders/ pipelines/ scenes/
  specs.py              # YAML save/load for pipelines and scenes
  vocabulary.py         # curated block vocabulary + validate_pipeline
  config.py             # CONFIRM_TX toggle
  devices.py            # SimulatedDevice + registry + list_devices
  backends/
    base.py             # Backend ABC (the swap boundary) + BackendError
    gnuradio_backend.py # the ONLY module that imports gnuradio
  ops/
    capture.py          # load_capture (SigMF/cf32/wav); capture(device, ...)
    analyze.py          # psd, find_signals, measure, detect_bursts (numpy/scipy)
    render.py           # spectrogram, psd_plot, constellation (matplotlib, Agg)
    pipeline.py         # validate-then-run; save to workspace
    simulate.py         # scene_to_pipeline, render_scene
    transmit.py         # transmit_capture into simulated scenes
    export_grc.py       # PipelineSpec -> .grc for GNU Radio Companion
```

(Modules from `specs.py` onward are being built across Plan 2; not all exist yet at every commit.)

## Commands

```bash
uv sync --extra dev                  # install deps
uv run pytest tests/marconi          # the test suite
uv run pytest tests/marconi -v
uv run python -c "import marconi"     # must succeed WITHOUT gnuradio on the path
```

## Key constraints

- **Engine isolation.** Only `backends/gnuradio_backend.py` imports `gnuradio`, lazily. Analysis (`psd`, `find_signals`, …) and rendering run on plain numpy/scipy/matplotlib and need no DSP runtime. Running pipelines and simulating require GNU Radio 3.10+ installed system-wide (importable in the uv venv on this machine).
- **Artifacts, not blobs.** Ops exchange `CaptureRef`/paths into the workspace, never raw sample arrays across the API boundary — keeps responses small and sessions reproducible.
- **Open formats.** Captures are SigMF, pipelines/scenes are YAML, flowgraphs export to `.grc` — all usable without Marconi.
- **TX is gated** by `marconi.config.CONFIRM_TX` (default on). It catches agent mistakes (wrong frequency/device), not licensing.

## Development workflow

- TDD, frequent commits. Pre-commit hooks (black, isort, flake8, mypy) run on commit; if they reformat, `git add -u` and re-commit.
- v1.0 is **simulation-only** (no hardware). Hardware receive (v1.1) and transmit (v1.2) are additive behind the same device/backend interfaces — no workflow or skill changes.
