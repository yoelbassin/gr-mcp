# gr-mcp Developer Guide

GNU Radio MCP server — exposes flowgraph operations as MCP tools for AI assistants.

## Architecture

```
main.py                          # FastMCP server entry point
src/gnuradio_mcp/
  models.py                      # Pydantic models (Block, Port, Connection, ExecutionResult, …)
  middlewares/
    base.py                      # ElementMiddleware wrapping gnuradio.grc.core.base.Element
    platform.py                  # PlatformMiddleware — block library, save/load flowgraphs
    flowgraph.py                 # FlowGraphMiddleware — add/remove blocks, connect, disconnect
    block.py                     # BlockMiddleware — params, ports
  providers/
    base.py                      # PlatformProvider — public API, 16 methods used as MCP tools
    mcp.py                       # McpPlatformProvider — registers all tools with FastMCP
```

## Commands

```bash
uv run main.py          # start the MCP server
uv run pytest           # run all tests
uv run pytest tests/unit/
uv run pytest tests/integration/
```

## Adding a New Tool

1. Add any new model to `models.py`
2. Implement the method on `PlatformProvider` in `providers/base.py`
3. Register it in `McpPlatformProvider.__init_tools` in `providers/mcp.py`
4. Document it in `SKILL.md` (tool reference table and workflow)

## Execution Flow

`execute_flowgraph` in `providers/base.py`:
1. Saves in-memory flowgraph to a temp `.grc` via `PlatformMiddleware.save_flowgraph`
2. Compiles with `grcc -o <tmpdir> <grc>` (system tool at `/opt/homebrew/bin/grcc`)
3. Runs the generated `flowgraph.py` with system `python3` and a configurable timeout
4. Returns `ExecutionResultModel` with stdout, stderr, exit code, compile errors, timed_out flag

## Key Constraints

- GNU Radio must be installed system-wide (not in the venv) — `main.py` injects its path via `sys.path`
- Flowgraph execution uses system `python3` (has gnuradio), not the venv Python
- `grcc` must be on PATH (`/opt/homebrew/bin/grcc` on macOS with Homebrew)
- Tests require a live GNU Radio installation — no mocking of the gnuradio package
