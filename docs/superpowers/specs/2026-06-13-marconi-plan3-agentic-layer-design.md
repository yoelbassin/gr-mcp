# Marconi Plan 3 — Agentic Layer Design

**Date:** 2026-06-13
**Status:** Approved design, pre-implementation
**Refines:** `docs/superpowers/specs/2026-06-12-marconi-design.md` (the overarching
design). This document settles the Layer-2 (MCP) and Layer-3 (plugin/skills)
specifics for v1.0 and reconciles the tool surface with the `marconi` library
actually shipped in Plans 1–2. Where it diverges from the overarching spec
(skill list, tool names), this document wins for Plan 3.

## Goal

Turn the finished engine-agnostic `marconi` library into the thing an RF
engineer talks to: a thin MCP server exposing the operations as tools, a Claude
Code plugin that packages it, and the RF workflow skills that carry the
judgment. This is the layer that delivers the product thesis — "Claude Code for
RF" — on top of the proven simulation core.

Plan 3 ships **v1.0: simulation-only**, with the FM receive loop as the reliable
end-to-end proof and a closed-loop TX/RX experiment as the second demo.

## Scope

**In:** the MCP server, the plugin manifests, six skills, error translation,
device-state persistence, and tests/evals through the tool surface.

**Explicitly deferred to v1.x** (not Plan 3):
- **Digital demodulation & ADS-B.** ADS-B is the canonical SDR demo and a
  compelling target, but it is a digital protocol decode (OOK/PPM → bits → CRC →
  field/CPR parse) that the current analog-oriented vocabulary does not cover.
  Closing that gap *generally* (OOK/PPM/FSK → bits primitives, without
  collapsing into one-tool-per-protocol) deserves its own brainstorm. v1.0 stays
  FM-first by deliberate wedge strategy. Note: `survey-spectrum` + `detect_bursts`
  will still let the agent *recognize and characterize* a burst signal like
  ADS-B ("~1 Mbit/s OOK bursts near 1090 MHz") even though it cannot yet decode
  it — the groundwork stays friendly to the v1.x build-out.
- Hardware (RX/TX) — additive behind the same device/backend interfaces (v1.1+).

## MCP server (Layer 2)

A deliberately thin FastMCP server: **one tool per operation, marshalling only,
no logic.** Tools import the `marconi` library and exchange `CaptureRef`s and
workspace paths — never sample arrays (the artifacts-not-blobs rule keeps tool
responses small and sessions resumable).

### Tool surface

Reconciled with the Plan 1–2 public API. ~18 operation tools plus vocabulary
discovery and run management:

| Family | Tools | Backs onto |
|---|---|---|
| devices | `list_devices`, `simulate_scene` | `list_devices`; build `SceneSpec` + `add_simulated_device` + persist scene YAML |
| capture | `load_capture`, `capture` | `load_capture`, `capture` |
| analyze | `psd`, `find_signals`, `measure`, `detect_bursts` | same-named ops |
| render | `spectrogram`, `psd_plot`, `constellation` | same-named ops |
| pipeline | `list_blocks`, `validate_pipeline`, `run_pipeline`, `save_pipeline`, `export_grc` | `vocabulary`, `validate_pipeline`, `run_pipeline`, `save_pipeline_to_workspace`, `export_grc` |
| simulate/tx | `render_scene`, `transmit_capture` | `render_scene`, `transmit_capture` |
| runs | `list_runs` | in-process run history (see State) |

Notes:
- **No `create_pipeline` tool.** A pipeline is a `PipelineSpec` (blocks +
  connections) the agent constructs as JSON and passes to
  `validate_pipeline`/`run_pipeline`. `list_blocks` exposes the curated
  vocabulary (block types, params, dtypes) so the agent composes pipelines
  against a known, reliable surface rather than guessing.
- **`simulate_scene`** is the merged `create_scene`/`set_scene` from the
  overarching spec: it takes a scene description, registers a `SimulatedDevice`,
  and persists the scene YAML (see State).

### Transport & workspace

- **Transport:** stdio, launched by the plugin's `.mcp.json` via
  `uv run --extra mcp marconi-mcp` (a new console entry point in `pyproject.toml`).
  `fastmcp` lives in an optional `mcp` extra rather than core dependencies, so the
  core library stays lean (the portability anchor); the `dev` extra pulls it in
  for tests.
- **Workspace:** the server's current working directory (the user's project
  dir), overridable with `MARCONI_WORKSPACE`. One `Workspace` instance per server
  process.

## State & lifecycle model (decision: A)

The `_REGISTRY` is currently process-global and ephemeral. Plan 3 adopts
**workspace-persisted devices + in-memory run handles:**

- **Simulated devices are derived from scene files.** `simulate_scene` writes the
  scene to `workspace/scenes/*.yaml` (already supported by `specs.py`). On
  startup the server scans that directory and re-registers a `SimulatedDevice`
  per scene. Devices become **reproducible and git-shareable** — directly serving
  the "workspace is the project" ethos. `list_devices` reflects the persisted
  scenes plus any backend-enumerated devices.
- **Runs are synchronous in v1.0, with an in-process history.** The GNU Radio
  backend's `run_pipeline` blocks until the flowgraph finishes (bounded by `head`
  blocks) or the timeout watchdog fires, so the `run_pipeline` tool blocks and
  returns a `RunResult`. Each run is recorded in an in-process history keyed by a
  generated run id (status, elapsed, artifacts); `list_runs` returns it for
  observability and artifact recall. **True background runs and `stop_run` are
  deferred to v1.x** — they are only meaningful once runs are non-blocking, which
  short simulation runs don't need.

## Error translation

A single boundary — a decorator applied to every tool — maps the library's
exception types into structured tool errors carrying a stable `code`, a
human-readable message, and the offending `block`/`field` where relevant. The
agent's self-correction loop is only as good as the error text, so raw
tracebacks are attached as detail, never the primary message.

Translated types: `PipelineValidationError`, `TransmitNotConfirmedError`,
`KeyError` (unknown device), `ValueError`, `PermissionError`, `BackendError`,
`RuntimeError`. Validation errors preserve the per-issue `ValidationIssue`
structure (block id, field, message) the library already produces.

## Skills (Layer 3 — the product's soul)

Six skills, each mapping to a stage of the general RF loop, not to a protocol.
The skill *set* is general by construction; the discipline below keeps each
skill's *content* general too.

| Skill | Encodes |
|---|---|
| **survey-spectrum** | *"What's here?"* `psd` → `find_signals` → `spectrogram` → written summary. Carries the FM-fragmentation workaround: seed/expect a noise floor; trust `measure`/`spectrogram` for wideband signals rather than treating each `find_signals` fragment as a distinct carrier. |
| **build-receiver** | *"Receive signal X."* General method: identify center freq + bandwidth → shift to baseband → decimate to ~2–3× the signal bandwidth → apply the demodulator matching the modulation → resample to the sink rate → **verify the output** (measure/render) before declaring success. FM is one worked example; the CW tone in the demo scene is a second, non-FM worked example (see Generality guardrails). |
| **debug-no-signal** | *"It's not working."* The systematic-debugging discipline applied to RF: check offset sign, decimation ratio, occupied BW vs. filter width, signal levels — using render/measure to *see* the failure rather than guessing. |
| **simulate-scene** | *"Make me a test signal."* Translate intent into a `SceneSpec` and register it via `simulate_scene`. The no-hardware on-ramp; also a genuine workflow (prototype against known ground truth). |
| **tx-experiment** | *"Transmit X and receive it back."* `transmit_capture` into a scene → `capture` it back → verify. Closed-loop, simulation-only; the second killer demo. Reuses the existing `transmit_capture` op and the `CONFIRM_TX` gate. |
| **escape-hatch** | Last resort: write Python importing `marconi` when the vocabulary genuinely can't express the task. Skills frame this explicitly as last resort and require reporting the gap — the gaps are the operations roadmap. |

(This refines the overarching spec's list: `verify-signal-chain` is folded into
`build-receiver`'s mandatory verify step; `simulate-scene` and `tx-experiment`
are added.)

## Generality guardrails

The chief risk to the platform vision is letting the demos harden the tools and
skills into demo-specific recipes. The architecture is structurally safe (tools
are 1:1 with general ops; the skill set is general workflow stages), so the risk
lives entirely in *how skills are written*. Four guardrails, baked into the spec
so future skill authors don't drift:

1. **Skills encode method, demos are worked examples.** Each skill leads with the
   general decision framework and shows a specific signal as *an* instance, with
   an explicit note on where another modulation diverges. Review test: *"Could an
   RF engineer follow this skill for AM or SSB, not just FM?"* If no, it is too
   specific and must be rewritten.
2. **A second, non-FM worked example — for free.** The demo scene already carries
   three signals (FM voice, an unmodulated tone, noise). `build-receiver`
   demonstrates receiving the **CW tone** as well as the FM signal — a genuinely
   different receive task on the same general skeleton, at zero extra scene cost.
3. **An anti-overfit eval.** At least one eval is a task *not* identical to the
   demo (a different scene, or "receive the carrier at 99.5 MHz"), to catch
   skills that only work on the rehearsed input.
4. **A stated generality principle** lives in the skills' authoring notes so the
   bar survives future contributors.

Content *breadth* (AM/SSB/digital demod, more analysis ops) grows deliberately
via the wedge roadmap and real usage — not under demo pressure. Narrow depth
(FM-first) is the wedge working as intended; narrow architecture or
demo-hardcoded skills is the failure we are designing against.

## Plugin packaging

A standard Claude Code plugin, with fresh Marconi manifests (the POC's were
deleted):
- `.claude-plugin/plugin.json` — metadata + skills.
- `.mcp.json` — declares the `marconi` MCP server (`uv run marconi-mcp`).
- `marketplace.json` — installable from this repo.
- `skills/` — the six skills above.
- `pyproject.toml` — `marconi-mcp` console entry point; add an optional `mcp`
  extra holding `fastmcp` (kept out of core deps so the library stays lean), and
  re-add `pytest-asyncio` (dev). The `dev` extra depends on `marconi[mcp]`.

## Testing & evals

1. **MCP layer (no subprocess):** in-process tests calling each tool function
   directly — assert it wraps the right op and that the error decorator
   translates each exception type into the right structured error.
2. **End-to-end through the tools:** the FM receive loop and the TX/RX loop
   driven *via the MCP tools* (not the library API), proving the tool surface is
   sufficient to complete both demos. Sim-only; requires GNU Radio.
3. **Skill evals (light for v1.0):** for each skill, an integration test that
   runs its documented tool sequence to the goal (proves the workflow is
   expressible). The two flagship demos are the headline evals; the anti-overfit
   eval (guardrail 3) is included. Heavier agent-transcript evals are a v1.x
   follow-up.

## Out of scope for Plan 3

- Digital demodulation / ADS-B decode (v1.x — own brainstorm).
- Hardware RX/TX (v1.1+).
- Background (non-blocking) runs and `stop_run` (v1.x — runs are synchronous in
  v1.0).
- Mid-run parameter tweaking beyond run/stop (v1.x candidate).
- `.grc` *import* (export only in v1).
- Agent-transcript skill evals (light tool-sequence evals only for v1.0).
