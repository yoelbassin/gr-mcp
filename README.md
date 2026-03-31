# GNU Radio MCP Server (`gr-mcp`)

[![Python Version](https://img.shields.io/badge/python-3.13%2B-blue.svg)](https://www.python.org/downloads/release/python-3130/)
[![Trust Score](https://archestra.ai/mcp-catalog/api/badge/quality/yoelbassin/gnuradioMCP)](https://archestra.ai/mcp-catalog/yoelbassin__gnuradiomcp)

`gr-mcp` is an MCP server that exposes GNU Radio flowgraph operations as tools for AI assistants and automation clients. It is built on FastMCP and designed for programmatic SDR workflow generation, editing, validation, and export.

## What It Provides

- Discovery of available GNU Radio blocks
- Programmatic block creation and removal
- Block parameter read/write operations
- Connection and disconnection between blocks
- Flowgraph validation and error inspection
- Flowgraph persistence to `.grc`

## Requirements

- Python `>=3.13`
- [GNU Radio](https://www.gnuradio.org/) installed and available to Python (tested with GNU Radio Companion `3.10.12.0`)
- [`uv`](https://docs.astral.sh/uv/)

## Installation

1. Clone the repository:

```bash
git clone https://github.com/yoelbassin/gr-mcp
cd gr-mcp
```

2. Install GNU Radio if it is not already installed:

- [GNU Radio installation guide](https://wiki.gnuradio.org/index.php/InstallingGR)

3. Create a virtual environment that can see system GNU Radio packages:

```bash
uv venv --system-site-packages
```

The `--system-site-packages` flag is required because GNU Radio is commonly installed as a system-level Python package.

## Run with an MCP Client

Add `gr-mcp` to your MCP client configuration (for example, Cursor or Claude Desktop):

```json
{
  "mcpServers": {
    "gr-mcp": {
      "command": "uv",
      "args": [
        "--directory",
        "/path/to/gr-mcp",
        "run",
        "main.py"
      ]
    }
  }
}
```

## Development

Install development dependencies and run tests:

```bash
pip install -e ".[dev]"
pytest
```

## Project Status

This project is under active development. Core functionality is available, and interfaces may evolve as the server matures.
