# GNURadio MCP Server

[![Python Version](https://img.shields.io/badge/python-3.13%2B-blue.svg)](https://www.python.org/downloads/release/python-3130/)


**GNURadio MCP Server** is a modern, extensible Machine Control Protocol (MCP) server for [GNURadio](https://www.gnuradio.org/), enabling programmatic, automated, and AI-driven creation of GNURadio flowgraphs. Designed for seamless integration with Large Language Models (LLMs), automation frameworks, and custom clients, it empowers you to generate `.grc` files and control SDR workflows at scale.

> **Why GNURadio MCP Server?**
> - Automate SDR workflows and flowgraph generation
> - Integrate with LLMs, bots, and custom tools
> - Build, modify, and validate flowgraphs programmatically
> - Save time and reduce manual errors in SDR prototyping


## Features
- 🌐 **MCP API**: Exposes a robust MCP interface for GNURadio
- 🛠️ **Programmatic Flowgraph Creation**: Build, edit, and save `.grc` files from code or automation
- 🤖 **LLM & Automation Ready**: Designed for AI and automation integration
- 🧩 **Extensible**: Modular architecture for easy extension and customization
- 📝 **Example Flowgraphs**: Includes ready-to-use `.grc` examples in the `misc/` directory
- 🧪 **Tested**: Comprehensive unit tests with `pytest`


## Quickstart

### Requirements
- Python >= 3.13
- GNURadio (Tested with GNURadio Companion v3.10.12.0)
- UV

### Usage
1. **Clone the repository**
```bash
git clone https://github.com/yoelbassin/gnuradioMCP
```

2. [**Install GNURadio**](https://wiki.gnuradio.org/index.php/InstallingGR)

3. **Set up a UV environment**
```bash
cd gnuradioMCP
uv venv --system-site-packages
```
   > The `--system-site-packages` flag is required because GNURadio installs the `gnuradio` Python package globally.

4. **Add the MCP server configuration to your client configuration.** For example, for Claude Desktop or Cursor:
```json
"mcpServers": {
    "GnuradioMCP": {
      "command": "uv",
      "args": [
        "--directory",
        "/path/to/gnuradioMCP",
        "run",
        "main.py"
      ]
    }
  }
```

## Development
Install development dependencies and run tests with:
```bash
pip install -e ".[dev]"
pytest
```


## Project Status
**In active development.** Core server functionality is available, but the API and features are evolving. Your feedback and contributions are highly valued!
