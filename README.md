# GNURadio MCP Server
The goal of this project is allowing LLMs to create GNURadio Flowcharts for GNURadio Companion. The MCP server works by creating a flowchart and saving it as a `.grc` file that can be opened in gnuradio-companion.

## Installation

Install GNURadio, follow the installation process in [InstallingGR](https://wiki.gnuradio.org/index.php/InstallingGR).
```bash
brew install gnuradio
```
Now create a virtual environment to run the project.
```bash
python3.13 -m venv --system-site-packages venv
source venv/bin/activate
pip install -e .
```
we use the `--system-site-packages` flag since GNURadio installs the `gnuradio` python package globally.

Currently, you can run the MCP server located on `main.py`
```
python main.py
```

and add the GnuradioMCP server to your LLM using
```
  "mcpServers": {
    "GnuradioMCP": {
      "url": "http://localhost:8000/sse"
    }
  }
```

## Current Status
In development, basic server have been created. Currently using GNURadio Companion (v3.10.12.0).
