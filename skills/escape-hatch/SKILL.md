---
name: escape-hatch
description: Use ONLY as a last resort when no Marconi tool or vocabulary block can express the task — write a Python script against the marconi library, run it, and report the capability gap.
---

# Escape Hatch (last resort)

When the operation tools and the block vocabulary genuinely cannot express what the user needs, you may drop to Python against the `marconi` library. This is the *last* resort, not a shortcut around learning the tools.

## Before you escape — STOP

1. Re-check `list_blocks` and the operation tools. Most "I need X" cases are a block or a composition you missed.
2. Confirm the task is truly outside the surface (a common real example: decoding a digital protocol such as ADS-B — OOK/PPM → bits → CRC → parse — which the v1.0 analog vocabulary doesn't cover).

## If you must escape

1. Write a Python script that `import marconi` and uses the library API directly — `Workspace`, the ops (`capture`, `find_signals`, `run_pipeline`, …), and the models. To read sample data, use `marconi.read_capture(path)` (returns `(samples, CaptureRef)`); stay inside the workspace.
2. Run it with `uv run python <script.py>`.
3. **Report the gap explicitly** to the user: which operation or block was missing. These gaps are the product roadmap — they're how the toolkit grows.
4. Leave the script in the workspace. Like pipelines and scenes, it's a durable, re-runnable artifact the user owns.

## Judgment

- Prefer extending the workflow with existing ops over hand-rolling DSP. If you find yourself reimplementing `find_signals` or a flowgraph in numpy, you've probably missed a tool.
- Keep escape scripts small and single-purpose; they document the gap.
