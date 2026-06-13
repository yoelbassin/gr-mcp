# Marconi 🤖

```
        o)))
        |
    .-------.
    | o   o |
    |   v   |
    '-------'

    Hi! I'm Marco.
```

LLM-driven RF for Claude Code. Describe what you want on the air and I survey the
spectrum, build and run the receiver, look at the signal, and leave a reproducible
RF project behind — SigMF captures, YAML pipelines, and `.grc` flowgraphs you own.

> Early development — v1.0 is simulation-only (no hardware yet). See `ROADMAP.md`.

> Proviously known as GNURadio/GR-MCP.

## Requirements

- [`uv`](https://docs.astral.sh/uv/) and Python ≥ 3.13
- [GNU Radio](https://www.gnuradio.org/) 3.10+ installed system-wide (conda-forge,
  your distro's package manager, or Homebrew) — needed to run pipelines and
  simulate. Analyzing existing IQ files doesn't require it.

## Install

In Claude Code:

```
/plugin marketplace add yoelbassin/gr-mcp
/plugin install marconi
```

That's it — `uv` starts the MCP server on first use and loads the skills. Nothing
else to configure.

## Use it

Ask in plain language:

> "Simulate an FM station at 100.3 MHz and build me a receiver."

> "Here's a capture, `mystery.wav` — survey what's on the air, then decode the strongest carrier."

I pick the right skill and drive the workflow end to end:

- **survey-spectrum** — what's present (PSD, signal detection, spectrogram)
- **simulate-scene** — register a simulated device from a description
- **build-receiver** — channelize, demodulate, validate, run, verify, save
- **debug-no-signal** — locate the fault when a receiver outputs nothing
- **tx-experiment** — closed-loop transmit-then-receive in simulation
- **escape-hatch** — drop to the Python library when no tool fits

Artifacts land under `artifacts/` in the server's working directory; set
`MARCONI_WORKSPACE` to put them elsewhere.

## License

[GPL-3.0-or-later](LICENSE).
