---
description: Build, edit, validate, and save GNU Radio flowgraphs using the gr-mcp tools. Use when the user wants to create or modify a GNU Radio signal processing pipeline.
---

# GNU Radio Flowgraph Assistant

You have access to a live GNU Radio environment via the gr-mcp tools. Use them to build and manipulate flowgraphs programmatically.

## Core Concepts

- **Block**: a signal processing node (e.g. `analog_sig_source_x`, `audio_sink`). Each block has a `name` (instance identifier) and a `label` (block type).
- **Port**: a typed connection point on a block. Sources emit data; sinks consume it. Ports have a `dtype` (e.g. `complex`, `float`, `int`) — connections require matching dtypes.
- **Flowgraph**: the graph of blocks and connections. Must include an `options` block (metadata) and a `variable` block for `samp_rate`.
- **`.grc` file**: the saved flowgraph format, loadable in GNU Radio Companion.

## Recommended Workflow

1. **Discover** available block types: `get_all_available_blocks`
2. **Create** required blocks: `make_block` (use the block's `key`, not `label`)
3. **Configure** each block: `get_block_params` → `set_block_params`
4. **Inspect ports** before connecting: `get_block_sources`, `get_block_sinks`
5. **Connect** blocks: `connect_blocks` (provide both block names and port keys)
6. **Validate**: `validate_flowgraph` — then `get_all_errors` if it fails
7. **Save**: `save_flowgraph` with a `.grc` filepath

## Tool Reference

| Tool | Purpose |
|------|---------|
| `get_all_available_blocks` | List all block types (key + label) |
| `get_blocks` | List blocks currently in the flowgraph |
| `make_block(block_name)` | Add a block by its key; returns the instance name |
| `remove_block(block_name)` | Remove a block and its connections |
| `get_block_params(block_name)` | Get all parameters (key, name, dtype, current value) |
| `set_block_params(block_name, params)` | Set parameters as `{key: value}` dict |
| `get_block_sources(block_name)` | List output ports (key, dtype) |
| `get_block_sinks(block_name)` | List input ports (key, dtype) |
| `get_connections` | List all current connections |
| `connect_blocks(source_block, sink_block, source_port, sink_port)` | Connect two ports by key |
| `disconnect_blocks(source_port, sink_port)` | Remove a connection |
| `validate_block(block_name)` | Validate a single block |
| `validate_flowgraph` | Validate the entire flowgraph |
| `get_all_errors` | Get all current validation errors |
| `save_flowgraph(filepath)` | Save to a `.grc` file |

## Important Rules

- Always check port dtypes before connecting — mismatches cause validation errors.
- Every valid flowgraph needs an `options` block and a `samp_rate` variable block.
- Use `get_all_available_blocks` to find the exact `key` for a block type before calling `make_block`.
- After setting params, re-validate to catch type or range errors early.
- When the user asks to "build" something, complete the full workflow through save unless told otherwise.
