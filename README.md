# GNURadio MCP Server
The goal of this project is allowing LLMs to create GNURadio Flowcharts for GNURadio Companion.

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

Run using 
```bash
python -m grc
```

## Current Status
Currently, the project is only a port of GNURadio Companion (v3.10.12.0)