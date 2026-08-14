# Marconi

**"Claude Code for RF."** Marconi is an MCP server that gives an LLM agent
expert radio hands: it turns IQ into FEC-corrected bits — sync,
demodulation, symbol decisions, descrambling, deinterleaving, FEC — on top of
GNU Radio. You describe a signal in natural language; the agent surveys the
band, composes a modem spec, decodes your capture, and does the
framing/CRC/field work itself on the bits Marconi returns. Marconi is
**receive-only**: there is no transmit path and no signal generation.

## Requirements

- **GNU Radio 3.10** with Python bindings — `brew install gnuradio` (macOS)
  or `apt install gnuradio` (Debian/Ubuntu)
- **Python ≥ 3.11** (the interpreter GNU Radio was installed for)
- [`uv`](https://docs.astral.sh/uv/) is used if present, but is optional

Everything else is bootstrapped automatically on first launch: the launcher
finds the GNU Radio interpreter, builds a `--system-site-packages` venv, and
installs Marconi into it.

## Install

### Claude Code (plugin)

```
/plugin marketplace add yoelbassin/gr-mcp
/plugin install marconi@marconi
```

The first session start bootstraps the venv, so the `marconi` server can take
a minute to come up once; afterwards it starts instantly.

### Claude Code (manual)

```sh
git clone https://github.com/yoelbassin/gr-mcp
claude mcp add marconi -- /path/to/gr-mcp/scripts/marconi-mcp.sh
```

### Any other MCP client

Point your client at the launcher as a stdio server:

```json
{
  "mcpServers": {
    "marconi": { "command": "/path/to/gr-mcp/scripts/marconi-mcp.sh" }
  }
}
```

## Tools

| Tool | What it does |
| --- | --- |
| `explain` | How to read a survey or run result — the interpretation manual, fetched per topic |
| `describe_stages` | Marconi's stage vocabulary, generated live from the engine registry |
| `validate_modem` | Compile a modem spec without running it — the fast iteration loop |
| `run_rx` | Decode: run a modem spec over an IQ capture and return the full pipeline result |
| `read_stream` | Page a decoded stream back as bits/symbols/soft values for framing, CRC, and field parsing |
| `stream_stats` | Summarize a decoded stream's distribution shape and, on request, fitted modulation levels |
| `survey` | Characterize a raw-IQ capture — spectrum, envelope, symbol rate, instantaneous frequency, and bursts; pre-demod measurements, no interpretation |
| `capture` | Record IQ from an attached SDR (via gr-soapy) into a capture file for the tools above |

## Environment

- `MARCONI_WORKSPACE` — where run artifacts (`marconi-runs/`) are written;
  defaults to `~/.cache/marconi` (never the server's working directory,
  which as a plugin is your own project)
- `MARCONI_PYTHON` — interpreter to build the venv from, if the launcher
  shouldn't autodetect the one with GNU Radio

## Troubleshooting

- **"no Python with GNU Radio found"** — install GNU Radio (see
  Requirements), or set `MARCONI_PYTHON` to an interpreter where
  `python -c 'import gnuradio'` succeeds.
- **`.venv` exists but `gnuradio` won't import** — the venv was created
  without `--system-site-packages`. Delete it (`rm -rf .venv` in the repo or
  plugin directory) and relaunch; the bootstrap rebuilds it correctly.
- **Client times out on first launch** — the one-time bootstrap can outlast
  strict client timeouts; run `scripts/marconi-mcp.sh` once by hand, or raise
  the timeout (Claude Code: `MCP_TIMEOUT`, in ms).

## Development

```sh
uv venv --system-site-packages
uv sync
uv run pytest
```

GPL-3.0 — see [LICENSE](LICENSE).
