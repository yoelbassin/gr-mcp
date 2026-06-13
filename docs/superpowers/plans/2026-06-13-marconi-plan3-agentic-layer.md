# Marconi Plan 3 — Agentic Layer (MCP + Plugin + Skills) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Turn the finished `marconi` library into the product an RF engineer talks to: a thin FastMCP server exposing the operations as tools, a Claude Code plugin that packages it, and six RF workflow skills carrying the judgment.

**Architecture:** A new `src/marconi/mcp/` subpackage holds the Layer-2 adapter. Tool *logic* lives in plain, synchronous functions (`tools.py`) that marshal JSON-friendly args, call the existing `marconi` ops, and return dicts — so they are unit-testable without any async or FastMCP. `server.py` registers those functions with FastMCP and runs over stdio. A single error-translation decorator (`errors.py`) maps the library's exception types into structured, agent-actionable tool errors. Per-process `ServerState` (`state.py`) owns the workspace, persists simulated devices as scene YAML (re-registered on startup), and records a run history. Layer 3 is six `SKILL.md` files plus the plugin manifests, written to encode transferable RF method (not demo recipes).

**Tech Stack:** Python ≥3.13, FastMCP 2.x (stdio MCP server), pydantic v2, pytest (+ pytest-asyncio for one wiring test), `uv`. Runs that touch GNU Radio require it installed system-wide (already importable in this venv).

**Reference spec:** `docs/superpowers/specs/2026-06-13-marconi-plan3-agentic-layer-design.md`.

**Conventions:**
- TDD: write the failing test, see it fail, implement minimally, see it pass, commit.
- End every commit message with the repo's `Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>` trailer (see CLAUDE.md). Pre-commit hooks (black/isort/flake8/mypy) run on commit; if they reformat, `git add -u` and re-commit.
- The base import invariant still holds: `import marconi` must not import `marconi.mcp`, `fastmcp`, or `gnuradio`.
- Tests live in `tests/marconi/`. Tests that build/run a flowgraph (`capture`, `render_scene`, `run_pipeline`, transmit, end-to-end) require GNU Radio and follow the existing no-skip pattern (it is installed here).

---

## File Structure

**Created:**
- `src/marconi/mcp/__init__.py` — subpackage marker; no heavy imports (keeps `import marconi.mcp.errors` cheap).
- `src/marconi/mcp/errors.py` — `ToolError` re-export, `classify_error`, `to_tool_error`, `tool_error_boundary`.
- `src/marconi/mcp/state.py` — `ServerState`, module-level `get_state`/`set_state`/`reset_state`.
- `src/marconi/mcp/tools.py` — the 19 tool functions + `TOOLS` registry + `_ref` helper.
- `src/marconi/mcp/server.py` — `build_server()`, `main()`.
- `src/marconi/mcp/__main__.py` — `python -m marconi.mcp` → `main()`.
- `.claude-plugin/plugin.json`, `.claude-plugin/marketplace.json`, `.mcp.json` — plugin manifests.
- `skills/{survey-spectrum,build-receiver,debug-no-signal,simulate-scene,tx-experiment,escape-hatch}/SKILL.md`.
- Tests: `tests/marconi/test_mcp_errors.py`, `test_mcp_state.py`, `test_mcp_server.py`, `test_mcp_tools_devices.py`, `test_mcp_tools_analyze.py`, `test_mcp_tools_render.py`, `test_mcp_tools_pipeline.py`, `test_mcp_tools_simulate_tx.py`, `test_mcp_e2e.py`, `test_plugin_manifests.py`, `test_plugin_skills.py`, `test_skill_evals.py`.

**Modified:**
- `pyproject.toml` — add `fastmcp` dep, `pytest-asyncio` dev dep, `marconi-mcp` script, `asyncio_mode`.
- `tests/marconi/conftest.py` — add the `server_state` fixture.
- `README.md`, `CLAUDE.md` — plugin install + tool/skill surface + generality principle.

---

## Task 1: Dependencies, package skeleton, and entry point

**Files:**
- Modify: `pyproject.toml`
- Create: `src/marconi/mcp/__init__.py`, `src/marconi/mcp/__main__.py`
- Test: `tests/marconi/test_mcp_server.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/marconi/test_mcp_server.py
import marconi


def test_base_import_does_not_load_mcp_or_fastmcp():
    import sys

    # importing the top-level package must not pull in the MCP adapter or fastmcp
    assert "fastmcp" not in sys.modules
    assert "marconi.mcp.server" not in sys.modules


def test_mcp_subpackage_importable():
    import marconi.mcp  # noqa: F401
    import marconi.mcp.errors  # noqa: F401


def test_entry_point_target_exists():
    from marconi.mcp.server import main  # noqa: F401
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/marconi/test_mcp_server.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'marconi.mcp'` (and `main` import fails). The first test may pass already; that is fine.

- [ ] **Step 3: Add deps, script, and asyncio config to `pyproject.toml`**

Edit the three sections so they read exactly:

```toml
dependencies = [
    "pydantic",
    "pyyaml",
    "numpy>=2.4.6",
    "scipy>=1.17.1",
    "matplotlib>=3.11.0",
]

[project.optional-dependencies]
mcp = ["fastmcp>=2.0"]
dev = [
    "pytest >= 7.0",
    "pytest-asyncio",
    "pre-commit",
    "marconi[mcp]",
]

[project.scripts]
marconi-mcp = "marconi.mcp.server:main"

[tool.setuptools.packages.find]
where = ["src"]

[tool.pytest.ini_options]
# src on the path so `import marconi` resolves without an install step
pythonpath = ["src", "."]
asyncio_mode = "auto"
```

- [ ] **Step 4: Create the subpackage skeleton**

```python
# src/marconi/mcp/__init__.py
"""Marconi MCP server (Layer 2): a thin FastMCP adapter over the marconi ops.

Importing this package is deliberately cheap — submodules (server, tools) pull
in fastmcp and are imported explicitly where needed, so `import marconi` stays
free of fastmcp and gnuradio."""
```

```python
# src/marconi/mcp/__main__.py
from marconi.mcp.server import main

if __name__ == "__main__":
    main()
```

Note: `server.py` and `main` are created in Task 4. This task only needs the import in `test_entry_point_target_exists` to resolve *after* Task 4; to keep Task 1 green on its own, add a minimal placeholder `main` now and flesh it out in Task 4:

```python
# src/marconi/mcp/server.py  (minimal placeholder; completed in Task 4)
def main() -> None:
    raise NotImplementedError("completed in Task 4")
```

- [ ] **Step 5: Sync deps and run the test**

Run: `uv sync --extra dev && uv run pytest tests/marconi/test_mcp_server.py -v`
Expected: PASS (3 tests).

- [ ] **Step 6: Confirm the base-import invariant still holds**

Run: `uv run python -c "import marconi, sys; assert 'fastmcp' not in sys.modules and 'gnuradio' not in sys.modules; print('clean')"`
Expected: `clean`

- [ ] **Step 7: Commit**

```bash
git add pyproject.toml src/marconi/mcp/ tests/marconi/test_mcp_server.py uv.lock
git commit -m "Add marconi.mcp skeleton, fastmcp dep, and marconi-mcp entry point"
```

---

## Task 2: Error translation boundary

**Files:**
- Create: `src/marconi/mcp/errors.py`
- Test: `tests/marconi/test_mcp_errors.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/marconi/test_mcp_errors.py
import pytest
from fastmcp.exceptions import ToolError

from marconi.backends import BackendError
from marconi.mcp.errors import classify_error, tool_error_boundary
from marconi.models import ValidationIssue
from marconi.ops.transmit import TransmitNotConfirmedError
from marconi.vocabulary import PipelineValidationError


def test_classify_validation_error():
    exc = PipelineValidationError([ValidationIssue(block_id="x", field="freq", message="bad")])
    code, message = classify_error(exc)
    assert code == "validation_error"
    assert "bad" in message


def test_classify_tx_not_confirmed():
    assert classify_error(TransmitNotConfirmedError("need confirm"))[0] == "tx_not_confirmed"


def test_classify_permission():
    assert classify_error(PermissionError("cannot tx"))[0] == "tx_forbidden"


def test_classify_key_error_unwraps_quotes():
    code, message = classify_error(KeyError("unknown device 'sim0'"))
    assert code == "not_found"
    assert message.startswith("unknown device")  # no leading quote


def test_classify_backend_error():
    assert classify_error(BackendError("engine blew up"))[0] == "backend_error"


def test_classify_value_and_type_errors():
    assert classify_error(ValueError("bad arg"))[0] == "invalid_argument"
    assert classify_error(TypeError("wrong type"))[0] == "invalid_argument"


def test_classify_runtime_error():
    assert classify_error(RuntimeError("render failed"))[0] == "runtime_error"


def test_boundary_translates_and_reraises():
    @tool_error_boundary
    def boom():
        raise ValueError("nope")

    with pytest.raises(ToolError) as ei:
        boom()
    assert "[invalid_argument]" in str(ei.value)
    assert "nope" in str(ei.value)


def test_boundary_passes_through_success():
    @tool_error_boundary
    def ok():
        return {"answer": 42}

    assert ok() == {"answer": 42}


def test_boundary_does_not_double_wrap_tool_error():
    @tool_error_boundary
    def already():
        raise ToolError("[custom] already structured")

    with pytest.raises(ToolError) as ei:
        already()
    assert str(ei.value).count("[custom]") == 1
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/marconi/test_mcp_errors.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'marconi.mcp.errors'`.

- [ ] **Step 3: Write `errors.py`**

```python
# src/marconi/mcp/errors.py
"""Translate marconi's exception types into structured, agent-actionable tool
errors at the MCP boundary.

The message handed to the agent is always vocabulary-level and carries a stable
[code] prefix the agent can match on. PipelineValidationError already formats its
per-block, per-field issues into its message, so that detail survives."""

from __future__ import annotations

import functools
from collections.abc import Callable
from typing import Any, TypeVar

from fastmcp.exceptions import ToolError
from pydantic import ValidationError

from marconi.backends import BackendError
from marconi.ops.transmit import TransmitNotConfirmedError
from marconi.vocabulary import PipelineValidationError

F = TypeVar("F", bound=Callable[..., Any])


def classify_error(exc: Exception) -> tuple[str, str]:
    """Map an exception to a stable (code, message) pair. Pure and testable."""
    if isinstance(exc, PipelineValidationError):
        return "validation_error", str(exc)
    if isinstance(exc, TransmitNotConfirmedError):
        return "tx_not_confirmed", str(exc)
    if isinstance(exc, PermissionError):
        # v1.0: the only PermissionError through this boundary is the transmit
        # can_tx guard; revisit if file-I/O permission errors surface here.
        return "tx_forbidden", str(exc)
    if isinstance(exc, KeyError):
        # KeyError.__str__ wraps the message in repr quotes; strip a single
        # matching outer pair (str.strip would over-strip inner quotes).
        s = str(exc)
        if len(s) >= 2 and s[0] == s[-1] and s[0] in ("'", '"'):
            s = s[1:-1]
        return "not_found", s
    if isinstance(exc, FileNotFoundError):
        # a bad capture/file path is a common agent mistake
        return "not_found", str(exc)
    if isinstance(exc, BackendError):
        return "backend_error", str(exc)
    if isinstance(exc, ValidationError):
        # pydantic structural errors (malformed pipeline/scene dicts) — str(exc)
        # lists the offending fields, which is actionable for the agent
        return "invalid_argument", str(exc)
    if isinstance(exc, (ValueError, TypeError)):
        return "invalid_argument", str(exc)
    if isinstance(exc, RuntimeError):
        return "runtime_error", str(exc)
    return "internal_error", f"{type(exc).__name__}: {exc}"


def to_tool_error(exc: Exception) -> ToolError:
    code, message = classify_error(exc)
    return ToolError(f"[{code}] {message}")


def tool_error_boundary(fn: F) -> F:
    """Wrap a tool function so any marconi exception becomes a structured
    ToolError the MCP client (the agent) reliably sees."""

    @functools.wraps(fn)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        try:
            return fn(*args, **kwargs)
        except ToolError:
            raise
        except Exception as exc:  # noqa: BLE001 — the boundary translates everything
            raise to_tool_error(exc) from exc

    return wrapper  # type: ignore[return-value]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/marconi/test_mcp_errors.py -v`
Expected: PASS (10 tests).

- [ ] **Step 5: Commit**

```bash
git add src/marconi/mcp/errors.py tests/marconi/test_mcp_errors.py
git commit -m "Add MCP error-translation boundary"
```

---

## Task 3: Server state (workspace, device persistence, run history)

**Files:**
- Create: `src/marconi/mcp/state.py`
- Test: `tests/marconi/test_mcp_state.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/marconi/test_mcp_state.py
from pathlib import Path

import pytest

import marconi
from marconi.mcp.state import ServerState, get_state, reset_state, set_state
from marconi.models import RunResult, SceneElement, SceneSpec
from marconi.specs import save_scene
from marconi.workspace import Workspace


@pytest.fixture(autouse=True)
def _clean():
    reset_state()
    marconi.clear_devices()
    yield
    reset_state()
    marconi.clear_devices()


def test_get_state_before_init_raises():
    with pytest.raises(RuntimeError):
        get_state()


def test_set_and_get_state(tmp_path: Path):
    state = ServerState(Workspace(tmp_path))
    set_state(state)
    assert get_state() is state


def test_init_registers_persisted_scenes(tmp_path: Path):
    scenes = tmp_path / "scenes"
    scenes.mkdir()
    save_scene(SceneSpec(name="s", elements=[SceneElement(kind="noise", amplitude=0.01)]),
               scenes / "sim0.yaml")
    ServerState(Workspace(tmp_path))  # scans on init
    assert [d.id for d in marconi.list_devices()] == ["sim0"]


def test_init_is_idempotent_and_authoritative(tmp_path: Path):
    # a stale device in the global registry is cleared; the workspace is the source of truth
    marconi.add_simulated_device("ghost", SceneSpec(name="g"))
    scenes = tmp_path / "scenes"
    scenes.mkdir()
    save_scene(SceneSpec(name="s"), scenes / "real.yaml")
    ServerState(Workspace(tmp_path))
    ids = sorted(d.id for d in marconi.list_devices())
    assert ids == ["real"]


def test_run_history(tmp_path: Path):
    state = ServerState(Workspace(tmp_path))
    rid = state.next_run_id()
    rec = state.record_run(rid, "fm_rx",
                           RunResult(status="ok", elapsed_seconds=1.2,
                                     artifacts=[Path("a.wav")], error=None))
    assert rec["run_id"] == rid
    assert rec["status"] == "ok"
    assert rec["artifacts"] == ["a.wav"]
    assert state.runs[-1]["run_id"] == rid
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/marconi/test_mcp_state.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'marconi.mcp.state'`.

- [ ] **Step 3: Write `state.py`**

```python
# src/marconi/mcp/state.py
"""Per-process server state.

The workspace is the source of truth: on startup the simulated-device registry
is rebuilt from the scene YAML files under workspace/scenes/, so devices created
in a previous session reappear. Runs are synchronous in v1.0; each is recorded
in an in-process history for observability and artifact recall."""

from __future__ import annotations

from marconi.devices import add_simulated_device, clear_devices
from marconi.models import RunResult
from marconi.specs import load_scene
from marconi.workspace import Workspace


class ServerState:
    def __init__(self, workspace: Workspace) -> None:
        self.workspace = workspace
        self.runs: list[dict] = []
        self._run_counter = 0
        self._load_persisted_devices()

    def _load_persisted_devices(self) -> None:
        # The workspace is authoritative: clear any prior registry, then rebuild
        # from scene files (device id == scene file stem).
        clear_devices()
        scenes_dir = self.workspace.root / "scenes"
        if not scenes_dir.is_dir():
            return
        for path in sorted(scenes_dir.glob("*.yaml")):
            add_simulated_device(path.stem, load_scene(path))

    def next_run_id(self) -> str:
        self._run_counter += 1
        return f"run-{self._run_counter}"

    def record_run(self, run_id: str, pipeline_name: str, result: RunResult) -> dict:
        rec = {
            "run_id": run_id,
            "pipeline": pipeline_name,
            "status": result.status,
            "elapsed_seconds": result.elapsed_seconds,
            "artifacts": [str(p) for p in result.artifacts],
            "error": result.error,
        }
        self.runs.append(rec)
        return rec


_STATE: ServerState | None = None


def set_state(state: ServerState) -> None:
    global _STATE
    _STATE = state


def get_state() -> ServerState:
    if _STATE is None:
        raise RuntimeError("server state not initialized; call set_state() first")
    return _STATE


def reset_state() -> None:
    """Test helper: drop the state and empty the device registry."""
    global _STATE
    _STATE = None
    clear_devices()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/marconi/test_mcp_state.py -v`
Expected: PASS (5 tests).

- [ ] **Step 5: Commit**

```bash
git add src/marconi/mcp/state.py tests/marconi/test_mcp_state.py
git commit -m "Add MCP server state with workspace-persisted devices and run history"
```

---

## Task 4: Server skeleton, `list_blocks` tool, and wiring tests

**Files:**
- Create: `src/marconi/mcp/tools.py` (first tool + registry)
- Modify: `src/marconi/mcp/server.py` (replace the Task-1 placeholder)
- Modify: `tests/marconi/conftest.py` (add `server_state` fixture)
- Test: `tests/marconi/test_mcp_server.py` (extend)

- [ ] **Step 1: Add the `server_state` fixture to `conftest.py`**

Append to `tests/marconi/conftest.py`:

```python
import pytest

from marconi.mcp.state import ServerState, reset_state, set_state
from marconi.workspace import Workspace


@pytest.fixture
def server_state(tmp_path):
    """A fresh MCP ServerState rooted at a tmp workspace (also empties the
    global device registry via ServerState init / reset)."""
    state = ServerState(Workspace(tmp_path))
    set_state(state)
    yield state
    reset_state()
```

- [ ] **Step 2: Write the failing tests**

Append to `tests/marconi/test_mcp_server.py`:

```python
def test_list_blocks_tool_returns_vocabulary(server_state):
    from marconi.mcp.tools import list_blocks

    blocks = list_blocks()
    assert "nbfm_rx" in blocks
    assert blocks["nbfm_rx"]["inputs"] == ["c"]
    assert blocks["nbfm_rx"]["outputs"] == ["f"]
    names = {p["name"] for p in blocks["nbfm_rx"]["params"]}
    assert {"audio_rate", "quad_rate"} <= names


def test_tools_registry_count():
    from marconi.mcp.tools import TOOLS

    assert len(TOOLS) == 19
    assert "list_blocks" in TOOLS and "run_pipeline" in TOOLS


def test_build_server_registers_all_tools():
    from marconi.mcp.server import build_server
    from marconi.mcp.tools import TOOLS

    mcp = build_server()
    assert mcp is not None
    # FastMCP server builds without error; tool count matches the registry.
    # (Tool listing is asserted via the in-memory client below.)
    assert len(TOOLS) == 19


async def test_tools_listed_through_mcp_client(server_state):
    from fastmcp import Client

    from marconi.mcp.server import build_server
    from marconi.mcp.tools import TOOLS

    mcp = build_server()
    async with Client(mcp) as client:
        tools = await client.list_tools()
    names = {t.name for t in tools}
    assert names == set(TOOLS)


def test_main_initializes_state_from_env(tmp_path, monkeypatch):
    import marconi.mcp.server as server
    from marconi.mcp.state import get_state, reset_state

    monkeypatch.setenv("MARCONI_WORKSPACE", str(tmp_path))
    monkeypatch.setattr("fastmcp.FastMCP.run", lambda self: None)
    server.main()
    assert get_state().workspace.root == tmp_path
    reset_state()
```

- [ ] **Step 3: Run tests to verify they fail**

Run: `uv run pytest tests/marconi/test_mcp_server.py -v`
Expected: FAIL — `marconi.mcp.tools` missing; `build_server` missing.

- [ ] **Step 4: Write `tools.py` with the registry and `list_blocks`**

```python
# src/marconi/mcp/tools.py
"""The MCP tool functions: thin, synchronous marshalling over the marconi ops.

Each function takes JSON-friendly args, calls into the marconi library, and
returns plain dicts/lists. Captures are referenced by their .sigmf-data path
string (CaptureRef.path); _ref() reconstructs the CaptureRef from the sidecar.
Every function is wrapped by tool_error_boundary and registered in TOOLS."""

from __future__ import annotations

from collections.abc import Callable

import marconi
from marconi import sigmf
from marconi.mcp.errors import tool_error_boundary
from marconi.mcp.state import get_state
from marconi.models import CaptureRef, PipelineSpec, SceneElement, SceneSpec
from marconi.specs import save_scene
from marconi.vocabulary import VOCABULARY


def _ref(capture_path: str) -> CaptureRef:
    """Reconstruct a CaptureRef from a capture's .sigmf-data path."""
    return sigmf.read_meta(capture_path)


@tool_error_boundary
def list_blocks() -> dict:
    """The curated block vocabulary for composing pipelines. For each block
    type: input/output dtypes ('c'=complex, 'f'=float) and its parameters
    (name, type, required, default). Compose pipelines ONLY from these types."""
    out: dict = {}
    for name, d in VOCABULARY.items():
        out[name] = {
            "inputs": list(d.inputs),
            "outputs": list(d.outputs),
            "params": [
                {
                    "name": p.name,
                    "type": p.type.__name__,
                    "required": p.required,
                    "default": p.default,
                }
                for p in d.params
            ],
        }
    return out


# Tools are added to this registry as later tasks implement them.
TOOLS: dict[str, Callable] = {
    "list_blocks": list_blocks,
}
```

Note: `TOOLS` will reach 19 entries by the end of Task 9. The `test_tools_registry_count`/`test_tools_listed_through_mcp_client`/`test_build_server_registers_all_tools` assertions for `== 19` will FAIL until Task 9 completes. To keep this task green now, temporarily assert `>= 1` in those three tests, then tighten to `== 19` as the final step of Task 9 (the Task 9 commit includes that tightening). Apply that temporary relaxation now:

- in `test_tools_registry_count`: `assert len(TOOLS) >= 1`
- in `test_build_server_registers_all_tools`: `assert len(TOOLS) >= 1`
- in `test_tools_listed_through_mcp_client`: `assert set(TOOLS) <= names and "list_blocks" in names`

- [ ] **Step 5: Write the real `server.py`**

Replace the placeholder `src/marconi/mcp/server.py` with:

```python
# src/marconi/mcp/server.py
"""The FastMCP server: registers the tool functions and runs over stdio."""

from __future__ import annotations

import os
from pathlib import Path

from fastmcp import FastMCP

from marconi.mcp.state import ServerState, set_state
from marconi.mcp.tools import TOOLS
from marconi.workspace import Workspace


def build_server() -> FastMCP:
    mcp: FastMCP = FastMCP("marconi")
    for name, fn in TOOLS.items():
        mcp.tool(name=name)(fn)
    return mcp


def main() -> None:
    root = os.environ.get("MARCONI_WORKSPACE", ".")
    set_state(ServerState(Workspace(Path(root))))
    build_server().run()
```

- [ ] **Step 6: Run tests to verify they pass**

Run: `uv run pytest tests/marconi/test_mcp_server.py -v`
Expected: PASS (all, with the relaxed assertions). The async test exercises the real FastMCP in-memory client end-to-end.

- [ ] **Step 7: Commit**

```bash
git add src/marconi/mcp/server.py src/marconi/mcp/tools.py tests/marconi/conftest.py tests/marconi/test_mcp_server.py
git commit -m "Add FastMCP server, tool registry, and list_blocks tool"
```

---

## Task 5: Device and capture tools

**Files:**
- Modify: `src/marconi/mcp/tools.py`
- Test: `tests/marconi/test_mcp_tools_devices.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/marconi/test_mcp_tools_devices.py
from pathlib import Path

import pytest

from marconi.mcp import tools as T


def test_simulate_scene_registers_and_persists(server_state):
    info = T.simulate_scene("sim0", [{"kind": "noise", "amplitude": 0.01}])
    assert info["id"] == "sim0"
    assert info["kind"] == "simulated"
    assert (server_state.workspace.root / "scenes" / "sim0.yaml").exists()
    assert any(d["id"] == "sim0" for d in T.list_devices())


def test_devices_persist_across_restart(server_state):
    from marconi.mcp.state import ServerState, set_state
    from marconi.workspace import Workspace

    T.simulate_scene("sim0", [{"kind": "noise", "amplitude": 0.01}])
    # Simulate a new server process pointed at the same workspace.
    set_state(ServerState(Workspace(server_state.workspace.root)))
    assert any(d["id"] == "sim0" for d in T.list_devices())


def test_capture_returns_reference(server_state):
    T.simulate_scene("sim0", [
        {"kind": "tone", "freq": 100.05e6, "amplitude": 0.5},
        {"kind": "noise", "amplitude": 0.005},
    ])
    ref = T.capture("sim0", center_freq=100e6, sample_rate=1e6, duration=0.05)
    assert ref["sample_rate"] == 1e6
    assert ref["num_samples"] > 0
    assert Path(ref["path"]).exists()


def test_load_capture_cf32(server_state, tmp_path):
    import numpy as np

    raw = tmp_path / "x.cf32"
    np.arange(8, dtype=np.complex64).tofile(raw)
    ref = T.load_capture(str(raw), sample_rate=1e6)
    assert ref["num_samples"] == 8
    assert ref["sample_rate"] == 1e6
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/marconi/test_mcp_tools_devices.py -v`
Expected: FAIL — `AttributeError: module 'marconi.mcp.tools' has no attribute 'simulate_scene'`.

- [ ] **Step 3: Implement the device/capture tools**

Add to `src/marconi/mcp/tools.py` (before the `TOOLS` dict):

```python
@tool_error_boundary
def list_devices() -> list[dict]:
    """List available devices: simulated devices plus any backend hardware
    (none in v1.0). Each entry has id, kind, can_tx, description."""
    return [d.model_dump() for d in marconi.list_devices()]


@tool_error_boundary
def simulate_scene(
    device_id: str, elements: list[dict], scene_name: str | None = None
) -> dict:
    """Register a simulated device whose 'on-air' contents are `elements`.

    Each element: {kind: tone|noise|fm_tone|iq_file, freq: Hz (absolute,
    ignored for noise), amplitude: float, params: {...}}. fm_tone needs
    params.mod_freq and renders only when the capture sample_rate is a multiple
    of 100000. The scene is persisted to scenes/<device_id>.yaml so the device
    survives a restart. Always include a small noise element."""
    state = get_state()
    scene = SceneSpec(
        name=scene_name or device_id,
        elements=[SceneElement(**e) for e in elements],
    )
    dev = marconi.add_simulated_device(device_id, scene)
    save_scene(scene, state.workspace.root / "scenes" / f"{device_id}.yaml")
    return dev.info().model_dump()


@tool_error_boundary
def load_capture(
    path: str, sample_rate: float | None = None, center_freq: float | None = None
) -> dict:
    """Ingest an external IQ file (SigMF / .cf32 / .wav) into the workspace.
    .cf32 requires sample_rate. Returns a capture reference."""
    ref = marconi.load_capture(
        path, get_state().workspace, sample_rate=sample_rate, center_freq=center_freq
    )
    return ref.model_dump(mode="json")


@tool_error_boundary
def capture(
    device_id: str,
    center_freq: float,
    sample_rate: float,
    duration: float,
    name: str | None = None,
) -> dict:
    """Capture IQ from a device (v1.0: simulated) as seen at center_freq /
    sample_rate for `duration` seconds. Returns a capture reference whose
    'path' feeds the analyze/render tools."""
    ref = marconi.capture(
        device_id,
        center_freq=center_freq,
        sample_rate=sample_rate,
        duration=duration,
        workspace=get_state().workspace,
        name=name,
    )
    return ref.model_dump(mode="json")
```

Then extend the registry:

```python
TOOLS: dict[str, Callable] = {
    "list_blocks": list_blocks,
    "list_devices": list_devices,
    "simulate_scene": simulate_scene,
    "load_capture": load_capture,
    "capture": capture,
}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/marconi/test_mcp_tools_devices.py -v`
Expected: PASS (4 tests). `test_capture_returns_reference` runs GNU Radio.

- [ ] **Step 5: Commit**

```bash
git add src/marconi/mcp/tools.py tests/marconi/test_mcp_tools_devices.py
git commit -m "Add device and capture MCP tools"
```

---

## Task 6: Analyze tools

**Files:**
- Modify: `src/marconi/mcp/tools.py`
- Test: `tests/marconi/test_mcp_tools_analyze.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/marconi/test_mcp_tools_analyze.py
from marconi import sigmf
from marconi.mcp import tools as T


def _write_capture(server_state, make_iq):
    x = make_iq([(100e3, 1.0)], sample_rate=1e6, duration=0.05, noise_amplitude=0.01)
    ref = sigmf.write_capture(
        x, server_state.workspace.new_capture_path("t"),
        center_freq=100e6, sample_rate=1e6,
    )
    return str(ref.path)


def test_psd_returns_summary_not_arrays(server_state, make_iq):
    path = _write_capture(server_state, make_iq)
    out = T.psd(path)
    assert "noise_floor_db" in out and "peaks" in out
    assert "freqs" not in out and "psd_db" not in out  # arrays omitted
    assert out["num_bins"] > 0


def test_find_signals_returns_list(server_state, make_iq):
    path = _write_capture(server_state, make_iq)
    sigs = T.find_signals(path)
    assert isinstance(sigs, list) and sigs
    assert any(abs(s["center_freq"] - 100.1e6) < 5e3 for s in sigs)


def test_measure_returns_measurement(server_state, make_iq):
    path = _write_capture(server_state, make_iq)
    m = T.measure(path, center_freq=100.1e6)
    assert m["snr_db"] > 8
    assert m["occupied_bw_99"] >= 0


def test_detect_bursts_returns_list(server_state, make_iq):
    path = _write_capture(server_state, make_iq)
    bursts = T.detect_bursts(path)
    assert isinstance(bursts, list)  # a steady tone yields no bursts
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/marconi/test_mcp_tools_analyze.py -v`
Expected: FAIL — `psd` etc. not attributes of `tools`.

- [ ] **Step 3: Implement the analyze tools**

Add to `tools.py` (before the `TOOLS` dict; the `TOOLS` dict will be assembled fully in Task 9 — until then keep adding entries to the existing dict literal):

```python
@tool_error_boundary
def psd(capture_path: str, nperseg: int = 4096) -> dict:
    """Power-spectral-density summary: the noise floor (dB) and the strongest
    spectral peaks (freq Hz, power dB). The full PSD curve is intentionally
    omitted to keep responses small — render psd_plot and read the image for
    the shape."""
    r = marconi.psd(_ref(capture_path), nperseg=nperseg)
    return {
        "noise_floor_db": r.noise_floor_db,
        "peaks": [p.model_dump() for p in r.peaks],
        "num_bins": len(r.freqs),
    }


@tool_error_boundary
def find_signals(
    capture_path: str,
    threshold_db: float = 6.0,
    min_bandwidth: float = 500.0,
    nperseg: int = 4096,
) -> list[dict]:
    """Detect signals as contiguous PSD regions above the noise floor. Each:
    center_freq, bandwidth (threshold-crossing extent), peak_power_db, snr_db.
    Note: wideband signals (e.g. FM) with a low noise floor can fragment into
    several detections — cross-check with measure() and a spectrogram."""
    return [
        s.model_dump()
        for s in marconi.find_signals(
            _ref(capture_path),
            threshold_db=threshold_db,
            min_bandwidth=min_bandwidth,
            nperseg=nperseg,
        )
    ]


@tool_error_boundary
def measure(
    capture_path: str,
    center_freq: float,
    search_bandwidth: float = 200e3,
    nperseg: int = 4096,
) -> dict:
    """Measure the signal nearest center_freq within search_bandwidth:
    center_freq, occupied_bw_99 (99% power bandwidth), power_db, snr_db. Treat a
    signal as reliably present only above ~8 dB SNR."""
    return marconi.measure(
        _ref(capture_path),
        center_freq=center_freq,
        search_bandwidth=search_bandwidth,
        nperseg=nperseg,
    ).model_dump()


@tool_error_boundary
def detect_bursts(
    capture_path: str, window: float = 1e-3, threshold_db: float = 6.0
) -> list[dict]:
    """Detect on/off bursts from the smoothed power envelope. Each: start_time,
    duration (s), mean_power_db. An always-on signal yields no bursts; valid for
    duty cycles below ~75%."""
    return [
        b.model_dump()
        for b in marconi.detect_bursts(
            _ref(capture_path), window=window, threshold_db=threshold_db
        )
    ]
```

Add these four names to the `TOOLS` dict literal:

```python
    "psd": psd,
    "find_signals": find_signals,
    "measure": measure,
    "detect_bursts": detect_bursts,
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/marconi/test_mcp_tools_analyze.py -v`
Expected: PASS (4 tests). No GNU Radio needed.

- [ ] **Step 5: Commit**

```bash
git add src/marconi/mcp/tools.py tests/marconi/test_mcp_tools_analyze.py
git commit -m "Add analyze MCP tools (psd summary, find_signals, measure, detect_bursts)"
```

---

## Task 7: Render tools

**Files:**
- Modify: `src/marconi/mcp/tools.py`
- Test: `tests/marconi/test_mcp_tools_render.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/marconi/test_mcp_tools_render.py
from pathlib import Path

from marconi import sigmf
from marconi.mcp import tools as T


def _capture(server_state, make_iq):
    x = make_iq([(50e3, 1.0)], sample_rate=1e6, duration=0.05)
    ref = sigmf.write_capture(
        x, server_state.workspace.new_capture_path("r"),
        center_freq=10e6, sample_rate=1e6,
    )
    return str(ref.path)


def test_spectrogram_writes_png(server_state, make_iq):
    out = T.spectrogram(_capture(server_state, make_iq))
    assert out["kind"] == "spectrogram"
    assert Path(out["path"]).exists() and out["path"].endswith(".png")


def test_psd_plot_writes_png(server_state, make_iq):
    out = T.psd_plot(_capture(server_state, make_iq))
    assert out["kind"] == "psd"
    assert Path(out["path"]).exists()


def test_constellation_writes_png(server_state, make_iq):
    out = T.constellation(_capture(server_state, make_iq))
    assert out["kind"] == "constellation"
    assert Path(out["path"]).exists()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/marconi/test_mcp_tools_render.py -v`
Expected: FAIL — render tools not defined.

- [ ] **Step 3: Implement the render tools**

Add to `tools.py`:

```python
@tool_error_boundary
def spectrogram(capture_path: str, name: str = "spectrogram", nfft: int = 1024) -> dict:
    """Render a spectrogram PNG into the workspace. Returns {path, kind}; read
    the image with vision to see signals over time and frequency."""
    return marconi.spectrogram(
        _ref(capture_path), get_state().workspace, name=name, nfft=nfft
    ).model_dump(mode="json")


@tool_error_boundary
def psd_plot(capture_path: str, name: str = "psd") -> dict:
    """Render a PSD plot PNG (power vs frequency, noise floor, top peaks)."""
    return marconi.psd_plot(
        _ref(capture_path), get_state().workspace, name=name
    ).model_dump(mode="json")


@tool_error_boundary
def constellation(capture_path: str, name: str = "constellation", max_points: int = 5000) -> dict:
    """Render an I/Q constellation PNG (useful after channelizing/demodulating)."""
    return marconi.constellation(
        _ref(capture_path), get_state().workspace, name=name, max_points=max_points
    ).model_dump(mode="json")
```

Add to the `TOOLS` dict literal:

```python
    "spectrogram": spectrogram,
    "psd_plot": psd_plot,
    "constellation": constellation,
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/marconi/test_mcp_tools_render.py -v`
Expected: PASS (3 tests).

- [ ] **Step 5: Commit**

```bash
git add src/marconi/mcp/tools.py tests/marconi/test_mcp_tools_render.py
git commit -m "Add render MCP tools (spectrogram, psd_plot, constellation)"
```

---

## Task 8: Pipeline tools and run history

**Files:**
- Modify: `src/marconi/mcp/tools.py`
- Test: `tests/marconi/test_mcp_tools_pipeline.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/marconi/test_mcp_tools_pipeline.py
from pathlib import Path

import pytest
from fastmcp.exceptions import ToolError

from marconi.mcp import tools as T


def _tone_to_file_pipeline(out_path: str) -> dict:
    # tone_source -> head -> file_sink ; minimal runnable graph
    return {
        "name": "tone_dump",
        "sample_rate": 1e6,
        "blocks": [
            {"id": "src", "type": "tone_source", "params": {"freq": 1e5, "amplitude": 0.5}},
            {"id": "head", "type": "head", "params": {"num_samples": 4096}},
            {"id": "sink", "type": "file_sink", "params": {"path": out_path}},
        ],
        "connections": [
            {"src_block": "src", "dst_block": "head"},
            {"src_block": "head", "dst_block": "sink"},
        ],
    }


def test_validate_pipeline_ok(server_state, tmp_path):
    issues = T.validate_pipeline(_tone_to_file_pipeline(str(tmp_path / "o.bin")))
    assert issues == []


def test_validate_pipeline_reports_issues(server_state):
    bad = {"name": "bad", "sample_rate": 1e6,
           "blocks": [{"id": "x", "type": "nope", "params": {}}],
           "connections": []}
    issues = T.validate_pipeline(bad)
    assert any("unknown block type" in i["message"] for i in issues)


def test_run_pipeline_invalid_raises_tool_error(server_state):
    bad = {"name": "bad", "sample_rate": 1e6,
           "blocks": [{"id": "x", "type": "nope", "params": {}}],
           "connections": []}
    with pytest.raises(ToolError) as ei:
        T.run_pipeline(bad)
    assert "[validation_error]" in str(ei.value)


def test_run_pipeline_records_history(server_state, tmp_path):
    out = str(tmp_path / "o.bin")
    result = T.run_pipeline(_tone_to_file_pipeline(out))
    assert result["status"] == "ok"
    assert result["run_id"] == "run-1"
    assert Path(out).exists()
    runs = T.list_runs()
    assert runs[-1]["run_id"] == "run-1"


def test_save_pipeline_and_export_grc(server_state, tmp_path):
    spec = _tone_to_file_pipeline(str(tmp_path / "o.bin"))
    saved = T.save_pipeline(spec)
    grc = T.export_grc(spec)
    assert Path(saved["path"]).exists() and saved["path"].endswith(".yaml")
    assert Path(grc["path"]).exists() and grc["path"].endswith(".grc")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/marconi/test_mcp_tools_pipeline.py -v`
Expected: FAIL — pipeline tools not defined.

- [ ] **Step 3: Implement the pipeline tools**

Add to `tools.py`:

```python
@tool_error_boundary
def validate_pipeline(pipeline: dict) -> list[dict]:
    """Validate a pipeline spec against the vocabulary. Returns a list of issues
    (empty = valid); each issue names the block_id and field. Always validate
    and fix all issues before running."""
    spec = PipelineSpec.model_validate(pipeline)
    return [i.model_dump() for i in marconi.validate_pipeline(spec)]


@tool_error_boundary
def run_pipeline(pipeline: dict, timeout: float = 30.0) -> dict:
    """Validate then run a pipeline (blocks until it finishes or `timeout`
    seconds elapse). Returns {run_id, pipeline, status (ok|timeout|error),
    elapsed_seconds, artifacts, error}. Invalid specs raise a validation error."""
    state = get_state()
    spec = PipelineSpec.model_validate(pipeline)
    result = marconi.run_pipeline(spec, timeout=timeout)
    return state.record_run(state.next_run_id(), spec.name, result)


@tool_error_boundary
def save_pipeline(pipeline: dict) -> dict:
    """Save a pipeline spec as YAML under workspace/pipelines/. Returns {path}."""
    spec = PipelineSpec.model_validate(pipeline)
    path = marconi.save_pipeline_to_workspace(spec, get_state().workspace)
    return {"path": str(path)}


@tool_error_boundary
def export_grc(pipeline: dict, name: str | None = None) -> dict:
    """Export a pipeline as a GNU Radio Companion .grc file under
    workspace/pipelines/ so the user can open and tweak it in GRC. Returns {path}."""
    state = get_state()
    spec = PipelineSpec.model_validate(pipeline)
    out = state.workspace.root / "pipelines" / f"{name or spec.name}.grc"
    out.parent.mkdir(parents=True, exist_ok=True)
    return {"path": str(marconi.export_grc(spec, out))}


@tool_error_boundary
def list_runs() -> list[dict]:
    """The history of pipeline runs this session: each {run_id, pipeline,
    status, elapsed_seconds, artifacts, error}."""
    return list(get_state().runs)
```

Add to the `TOOLS` dict literal:

```python
    "validate_pipeline": validate_pipeline,
    "run_pipeline": run_pipeline,
    "save_pipeline": save_pipeline,
    "export_grc": export_grc,
    "list_runs": list_runs,
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/marconi/test_mcp_tools_pipeline.py -v`
Expected: PASS (5 tests). `test_run_pipeline_records_history` runs GNU Radio.

- [ ] **Step 5: Commit**

```bash
git add src/marconi/mcp/tools.py tests/marconi/test_mcp_tools_pipeline.py
git commit -m "Add pipeline MCP tools and run history"
```

---

## Task 9: Simulate and transmit tools (and finalize the registry)

**Files:**
- Modify: `src/marconi/mcp/tools.py`
- Modify: `tests/marconi/test_mcp_server.py` (tighten the relaxed assertions to `== 19`)
- Test: `tests/marconi/test_mcp_tools_simulate_tx.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/marconi/test_mcp_tools_simulate_tx.py
from pathlib import Path

import pytest
from fastmcp.exceptions import ToolError

from marconi.mcp import tools as T


def test_render_scene_adhoc(server_state):
    ref = T.render_scene(
        [{"kind": "tone", "freq": 100.02e6, "amplitude": 0.5},
         {"kind": "noise", "amplitude": 0.005}],
        center_freq=100e6, sample_rate=1e6, duration=0.05,
    )
    assert Path(ref["path"]).exists()
    assert ref["sample_rate"] == 1e6


def test_transmit_requires_confirmation(server_state):
    T.simulate_scene("sim0", [{"kind": "noise", "amplitude": 0.005}])
    payload = T.render_scene(
        [{"kind": "tone", "freq": 100e6, "amplitude": 0.3}],
        center_freq=100e6, sample_rate=1e6, duration=0.05,
    )
    with pytest.raises(ToolError) as ei:
        T.transmit_capture("sim0", payload["path"], freq=100e6)
    assert "[tx_not_confirmed]" in str(ei.value)


def test_transmit_confirmed_appends_element(server_state):
    T.simulate_scene("sim0", [{"kind": "noise", "amplitude": 0.005}])
    payload = T.render_scene(
        [{"kind": "tone", "freq": 100e6, "amplitude": 0.3}],
        center_freq=100e6, sample_rate=1e6, duration=0.05,
    )
    el = T.transmit_capture("sim0", payload["path"], freq=100e6, confirmed=True)
    assert el["kind"] == "iq_file"
    assert el["freq"] == 100e6
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/marconi/test_mcp_tools_simulate_tx.py -v`
Expected: FAIL — `render_scene`/`transmit_capture` not defined.

- [ ] **Step 3: Implement the simulate/transmit tools**

Add to `tools.py`:

```python
@tool_error_boundary
def render_scene(
    elements: list[dict],
    center_freq: float,
    sample_rate: float,
    duration: float,
    scene_name: str = "scene",
    name: str | None = None,
) -> dict:
    """Render an ad-hoc scene (inline `elements`, no device registered) to a
    capture in the workspace. Use simulate_scene + capture when you want a
    persistent device. Returns a capture reference."""
    scene = SceneSpec(
        name=scene_name, elements=[SceneElement(**e) for e in elements]
    )
    return marconi.render_scene(
        scene,
        center_freq=center_freq,
        sample_rate=sample_rate,
        duration=duration,
        workspace=get_state().workspace,
        name=name,
    ).model_dump(mode="json")


@tool_error_boundary
def transmit_capture(
    device_id: str,
    capture_path: str,
    freq: float,
    amplitude: float = 1.0,
    confirmed: bool = False,
) -> dict:
    """Replay a capture 'on the air' into a simulated device's scene (it becomes
    an iq_file element, audible to later captures at the SAME sample_rate).
    Gated by CONFIRM_TX: pass confirmed=True only after checking freq and
    device. The change is session-scoped (not re-persisted to the scene file)."""
    el = marconi.transmit_capture(
        device_id, _ref(capture_path), freq=freq, amplitude=amplitude, confirmed=confirmed
    )
    return el.model_dump(mode="json")
```

Add to the `TOOLS` dict literal:

```python
    "render_scene": render_scene,
    "transmit_capture": transmit_capture,
```

- [ ] **Step 4: Tighten the registry-count assertions**

In `tests/marconi/test_mcp_server.py`, revert the Task-4 relaxations to the strict forms:
- `test_tools_registry_count`: `assert len(TOOLS) == 19`
- `test_build_server_registers_all_tools`: `assert len(TOOLS) == 19`
- `test_tools_listed_through_mcp_client`: `assert names == set(TOOLS)`

- [ ] **Step 5: Run the full MCP suite**

Run: `uv run pytest tests/marconi/test_mcp_tools_simulate_tx.py tests/marconi/test_mcp_server.py -v`
Expected: PASS. The registry now holds all 19 tools.

- [ ] **Step 6: Commit**

```bash
git add src/marconi/mcp/tools.py tests/marconi/test_mcp_tools_simulate_tx.py tests/marconi/test_mcp_server.py
git commit -m "Add simulate/transmit MCP tools and finalize the 19-tool registry"
```

---

## Task 10: End-to-end through the tools (FM loop + TX/RX loop)

**Files:**
- Test: `tests/marconi/test_mcp_e2e.py`

These tests prove the *tool surface* (not the library API) is sufficient to complete both killer demos. They require GNU Radio.

- [ ] **Step 1: Write the failing test**

```python
# tests/marconi/test_mcp_e2e.py
import numpy as np
from scipy.io import wavfile

from marconi.mcp import tools as T


def test_fm_receive_loop_through_tools(server_state, tmp_path):
    # 1. The world
    T.simulate_scene("sim0", [
        {"kind": "fm_tone", "freq": 100.3e6, "amplitude": 1.0, "params": {"mod_freq": 1e3}},
        {"kind": "tone", "freq": 99.5e6, "amplitude": 0.4},
        {"kind": "noise", "amplitude": 0.005},
    ])
    # 2. Survey
    cap = T.capture("sim0", center_freq=100e6, sample_rate=2e6, duration=0.25)
    sigs = T.find_signals(cap["path"])
    assert len(sigs) >= 2
    fm = max(sigs, key=lambda s: s["bandwidth"])
    assert abs(fm["center_freq"] - 100.3e6) < 5e3
    # 3. Build + run a NBFM receiver via the tools
    wav = str(tmp_path / "audio.wav")
    pipeline = {
        "name": "fm_rx", "sample_rate": 2e6,
        "blocks": [
            {"id": "src", "type": "file_source", "params": {"path": cap["path"]}},
            {"id": "chan", "type": "freq_xlating_lowpass", "params": {
                "decimation": 20, "center_offset": fm["center_freq"] - 100e6,
                "cutoff": 8e3, "transition": 4e3}},
            {"id": "demod", "type": "nbfm_rx", "params": {"audio_rate": 25000, "quad_rate": 100000}},
            {"id": "audio", "type": "wav_sink", "params": {"path": wav, "sample_rate": 25000}},
        ],
        "connections": [
            {"src_block": "src", "dst_block": "chan"},
            {"src_block": "chan", "dst_block": "demod"},
            {"src_block": "demod", "dst_block": "audio"},
        ],
    }
    assert T.validate_pipeline(pipeline) == []
    run = T.run_pipeline(pipeline)
    assert run["status"] == "ok"
    # 4. Verify recovered audio peaks at 1 kHz
    rate, audio = wavfile.read(wav)
    audio = audio.astype(np.float32)
    audio = audio - audio.mean()
    spec = np.abs(np.fft.rfft(audio * np.hanning(len(audio))))
    peak = np.fft.rfftfreq(len(audio), 1 / rate)[int(np.argmax(spec))]
    assert abs(peak - 1e3) < 50


def test_tx_rx_loop_through_tools(server_state):
    # Receiver device: just a noise floor
    T.simulate_scene("rx0", [{"kind": "noise", "amplitude": 0.001}])
    # Payload: a tone 10 kHz above 100 MHz, recorded at 1 Msps
    payload = T.render_scene(
        [{"kind": "tone", "freq": 100.01e6, "amplitude": 0.5}],
        center_freq=100e6, sample_rate=1e6, duration=0.1,
    )
    # Transmit it into the receiver's scene (same rate), then receive it back
    el = T.transmit_capture("rx0", payload["path"], freq=100e6, confirmed=True)
    assert el["kind"] == "iq_file"
    cap = T.capture("rx0", center_freq=100e6, sample_rate=1e6, duration=0.1)
    sigs = T.find_signals(cap["path"])
    assert any(abs(s["center_freq"] - 100.01e6) < 5e3 for s in sigs)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/marconi/test_mcp_e2e.py -v`
Expected: initially FAIL only if a tool is missing; otherwise it should PASS once Tasks 5–9 are done. Run it to confirm the whole tool path works end-to-end.

- [ ] **Step 3: If anything fails, debug with systematic-debugging**

The library equivalents already pass (`test_api_v2.py`); a failure here means a marshalling bug in the tool layer (e.g. a param not forwarded). Fix in `tools.py`, not in the library.

- [ ] **Step 4: Commit**

```bash
git add tests/marconi/test_mcp_e2e.py
git commit -m "Add end-to-end MCP tool tests: FM receive loop and TX/RX loop"
```

---

## Task 11: Plugin manifests

**Files:**
- Create: `.claude-plugin/plugin.json`, `.claude-plugin/marketplace.json`, `.mcp.json`
- Test: `tests/marconi/test_plugin_manifests.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/marconi/test_plugin_manifests.py
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def test_plugin_json_valid():
    data = json.loads((ROOT / ".claude-plugin" / "plugin.json").read_text())
    assert data["name"] == "marconi"
    assert "version" in data and "description" in data


def test_marketplace_json_lists_marconi():
    data = json.loads((ROOT / ".claude-plugin" / "marketplace.json").read_text())
    names = [p["name"] for p in data["plugins"]]
    assert "marconi" in names


def test_mcp_json_declares_server():
    data = json.loads((ROOT / ".mcp.json").read_text())
    server = data["mcpServers"]["marconi"]
    assert server["command"] == "uv"
    assert "marconi-mcp" in server["args"]
    assert "${CLAUDE_PLUGIN_ROOT}" in server["args"]
    # fastmcp is an optional extra; the plugin must launch with it
    assert "--extra" in server["args"] and "mcp" in server["args"]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/marconi/test_plugin_manifests.py -v`
Expected: FAIL — manifest files do not exist.

- [ ] **Step 3: Create `.mcp.json`**

```json
{
  "mcpServers": {
    "marconi": {
      "command": "uv",
      "args": [
        "--directory",
        "${CLAUDE_PLUGIN_ROOT}",
        "run",
        "--extra",
        "mcp",
        "marconi-mcp"
      ]
    }
  }
}
```

- [ ] **Step 4: Create `.claude-plugin/plugin.json`**

```json
{
  "name": "marconi",
  "description": "LLM-driven RF control and analysis — \"Claude Code for RF\". Survey spectrum, build and run receivers, simulate scenes, and run closed-loop TX/RX experiments through natural language, leaving a reproducible RF project behind.",
  "version": "0.1.0",
  "author": {
    "name": "Yoel Bassin"
  },
  "repository": "https://github.com/yoelbassin/gr-mcp",
  "homepage": "https://github.com/yoelbassin/gr-mcp",
  "license": "MIT",
  "keywords": ["rf", "sdr", "gnuradio", "dsp", "signal-processing", "mcp"]
}
```

- [ ] **Step 5: Create `.claude-plugin/marketplace.json`**

```json
{
  "name": "marconi",
  "owner": {
    "name": "Yoel Bassin"
  },
  "plugins": [
    {
      "name": "marconi",
      "source": "./",
      "description": "LLM-driven RF control and analysis — \"Claude Code for RF\"."
    }
  ]
}
```

- [ ] **Step 6: Run test to verify it passes**

Run: `uv run pytest tests/marconi/test_plugin_manifests.py -v`
Expected: PASS (3 tests).

- [ ] **Step 7: Commit**

```bash
git add .claude-plugin/ .mcp.json tests/marconi/test_plugin_manifests.py
git commit -m "Add Marconi plugin manifests (plugin/marketplace/.mcp.json)"
```

---

## Task 12: Skills scaffold + frontmatter validator + survey-spectrum

**Files:**
- Create: `skills/survey-spectrum/SKILL.md`
- Test: `tests/marconi/test_plugin_skills.py`

- [ ] **Step 1: Write the failing test (validates every skill's frontmatter)**

```python
# tests/marconi/test_plugin_skills.py
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
SKILLS = ROOT / "skills"
EXPECTED = {
    "survey-spectrum", "build-receiver", "debug-no-signal",
    "simulate-scene", "tx-experiment", "escape-hatch",
}


def _frontmatter(path: Path) -> dict:
    text = path.read_text(encoding="utf-8")
    assert text.startswith("---\n"), f"{path} missing frontmatter"
    _, fm, _body = text.split("---\n", 2)
    return yaml.safe_load(fm)


def test_every_present_skill_has_valid_frontmatter():
    found = {p.parent.name for p in SKILLS.glob("*/SKILL.md")}
    assert found, "no skills found"
    for name in found:
        fm = _frontmatter(SKILLS / name / "SKILL.md")
        assert fm.get("name") == name
        assert isinstance(fm.get("description"), str) and len(fm["description"]) > 20


def test_survey_spectrum_skill_present():
    assert (SKILLS / "survey-spectrum" / "SKILL.md").exists()
```

Note: `test_every_present_skill_has_valid_frontmatter` globs, so it automatically covers each skill as Tasks 13–17 add them — no edits needed there.

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/marconi/test_plugin_skills.py -v`
Expected: FAIL — no skills directory yet.

- [ ] **Step 3: Write `skills/survey-spectrum/SKILL.md`**

```markdown
---
name: survey-spectrum
description: Use when the user wants to know what signals are present — characterize a capture or a simulated device's spectrum (PSD, signal detection, spectrogram) and summarize what's on the air.
---

# Survey Spectrum

Characterize what's present in a capture, then summarize it for the user. This is the first step of almost every RF task ("what's here?").

## Method

1. **Get a capture.** From a simulated device: `capture(device_id, center_freq, sample_rate, duration)`. From a file the user brings: `load_capture(path)`. Use the returned `path` for everything below.
2. **Read the numbers.** `psd(capture_path)` → noise floor + strongest peaks. `find_signals(capture_path)` → candidate signals (center_freq, bandwidth, snr_db).
3. **Always look, don't just count.** Render `spectrogram(capture_path)` and read the image. `find_signals` is a first pass over a 1-D PSD; the spectrogram shows time/frequency structure (bursts, drift, multiple carriers) that the list can miss.
4. **Confirm wideband signals with `measure`.** A wide signal (e.g. an FM carrier) with a low noise floor can *fragment* into several `find_signals` detections. For any wide or clustered group, call `measure(capture_path, center_freq=...)` to get the true 99% occupied bandwidth, and trust the spectrogram over the raw detection count.
5. **Summarize.** Give the user a short table: frequency, bandwidth, SNR, and a guess at type (narrowband tone vs. wideband FM-like vs. bursty), plus the spectrogram path.

## Judgment

- **Seed a noise floor when simulating.** A noiseless tone has phantom sidebands that register as spurious detections. When you build a scene to survey (see simulate-scene), always include a small `noise` element (amplitude ≈ 0.005).
- **SNR gate.** Treat a `measure` result as a real signal only above ~8 dB SNR; below that you're likely measuring noise.
- **Bandwidth is threshold-crossing extent**, not −3 dB or OBW — use `measure` when you need a real bandwidth number.
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/marconi/test_plugin_skills.py -v`
Expected: PASS (2 tests).

- [ ] **Step 5: Commit**

```bash
git add skills/survey-spectrum/SKILL.md tests/marconi/test_plugin_skills.py
git commit -m "Add survey-spectrum skill and skill-frontmatter validator"
```

---

## Task 13: build-receiver skill

**Files:**
- Create: `skills/build-receiver/SKILL.md`

- [ ] **Step 1: Write `skills/build-receiver/SKILL.md`**

```markdown
---
name: build-receiver
description: Use when the user wants to receive or demodulate a specific signal — build, validate, run, and verify a receiver pipeline, then save it as a reusable, GRC-openable flowgraph.
---

# Build Receiver

Turn a target signal into a working receiver. The method below is general; FM voice and an unmodulated carrier are shown as two worked examples so the *shape* transfers to modulations the demo doesn't cover.

## General method (any modulation)

1. **Know the target.** Get its center frequency and occupied bandwidth from survey-spectrum / `measure`. Compute the offset from the capture's center: `center_offset = target_freq − capture_center_freq`.
2. **Discover the blocks.** Call `list_blocks`. Compose only from these types; check each block's input/output dtype ('c' complex, 'f' float) so connections match.
3. **Channelize.** Use `freq_xlating_lowpass` to shift the target to baseband and decimate. Choose `decimation` so the post-decimation rate is ~2–3× the signal bandwidth; set `cutoff ≈ signal_bandwidth` and `transition ≈ cutoff/2`.
4. **Demodulate to match the modulation** (see examples below).
5. **Validate.** `validate_pipeline(pipeline)` and fix *every* issue (each names the block and field) before running.
6. **Run.** `run_pipeline(pipeline)`; check `status == "ok"`.
7. **Verify before you claim success** (see Verify). Never report a working receiver without evidence.
8. **Persist.** `save_pipeline(pipeline)` and `export_grc(pipeline)` so the user keeps a reusable receiver they can open in GNU Radio Companion.

## Worked example A — NBFM voice

Channelize so the post-decimation rate equals the FM quad rate (a multiple of the audio rate), then demodulate to a WAV:

- `freq_xlating_lowpass` (decimation chosen so e.g. 2 MHz / 20 = 100 kHz = quad_rate; cutoff ≈ 8 kHz, transition ≈ 4 kHz)
- `nbfm_rx` with `audio_rate=25000`, `quad_rate=100000` (**quad_rate must be an integer multiple of audio_rate**)
- `wav_sink` (`sample_rate=25000`)

Connections: src → chan → demod → audio.

## Worked example B — unmodulated carrier / CW (non-FM)

There is *no* demodulator here — receiving a carrier means channelizing it to baseband and confirming it:

- `freq_xlating_lowpass` (narrow `cutoff`, e.g. a few kHz) → `file_sink`
- Then `measure(channelized_capture, center_freq=0)` (or render a `spectrogram`) to confirm the carrier sits at baseband with the expected SNR.

This proves the channelizer/offset logic independently of any demodulator — the same skeleton as example A with the demod stage removed.

## Other modulations

AM, SSB, and digital (OOK/PPM/FSK) demodulators are **not in the v1.0 vocabulary**. If the target needs one, report the gap and switch to the escape-hatch skill — do not force an FM demodulator onto a non-FM signal.

## Verify (mandatory)

- FM voice: read the recovered WAV and confirm energy in the audio band (e.g. a 1 kHz modulating tone shows up at 1 kHz).
- CW/other: `measure` or `spectrogram` the channelized output and confirm the carrier/structure is where you expect.
- If verification fails → switch to the debug-no-signal skill. Do not retry blindly.
```

- [ ] **Step 2: Run the validator**

Run: `uv run pytest tests/marconi/test_plugin_skills.py -v`
Expected: PASS — the glob now covers `build-receiver` with valid frontmatter.

- [ ] **Step 3: Commit**

```bash
git add skills/build-receiver/SKILL.md
git commit -m "Add build-receiver skill (general method + FM and CW worked examples)"
```

---

## Task 14: debug-no-signal skill

**Files:**
- Create: `skills/debug-no-signal/SKILL.md`

- [ ] **Step 1: Write `skills/debug-no-signal/SKILL.md`**

```markdown
---
name: debug-no-signal
description: Use when a receiver runs but produces no output, noise, or the wrong result — systematically locate the fault instead of guessing, using render and measure to see the problem.
---

# Debug No Signal

A receiver that runs `status: ok` but produces silence or garbage has a DSP fault, not a crash. Find it systematically; do not randomly tweak parameters.

## Method (systematic-debugging applied to RF)

1. **Reproduce and name the symptom.** Re-run and state precisely what's wrong: silent audio, broadband noise, a tone at the wrong pitch, low level.
2. **See, don't guess.** Render a `spectrogram` (and/or `psd_plot`) of the *channelized* capture — the stage right after `freq_xlating_lowpass` — and `measure` it. Most faults are visible there.
3. **Check hypotheses cheapest-first, changing ONE thing per iteration:**
   - **Wrong offset or sign.** After channelizing, the target should sit at 0 Hz. If it's offset by ±X, your `center_offset` value or its sign is wrong (`center_offset = target_freq − capture_center_freq`).
   - **Over/under-decimation.** The post-decimation rate must be ≥ ~2× the signal bandwidth, and for FM must equal the demodulator's `quad_rate`. Check `sample_rate / decimation`.
   - **Filter too narrow/wide.** `cutoff` below the signal bandwidth clips it (muffled/silent); far too wide lets in neighbors (noisy).
   - **Levels / wrong channel.** `measure` the channelized capture: very low SNR means you channelized empty spectrum — re-check the target frequency from survey-spectrum.
   - **Demod/rate mismatch.** For `nbfm_rx`, `quad_rate` must be an integer multiple of `audio_rate`; a `wav_sink` `sample_rate` must match the audio stream rate.
4. **Re-verify after each change** the same way build-receiver verifies. Stop when the output is correct.
5. **If no vocabulary block fixes it** (e.g. the signal isn't FM at all), report the gap → escape-hatch.

## Judgment

- One change at a time. If you change three params and it works, you've learned nothing and it'll break next time.
- Trust the spectrogram over your mental model of the math.
```

- [ ] **Step 2: Run the validator**

Run: `uv run pytest tests/marconi/test_plugin_skills.py -v`
Expected: PASS.

- [ ] **Step 3: Commit**

```bash
git add skills/debug-no-signal/SKILL.md
git commit -m "Add debug-no-signal skill"
```

---

## Task 15: simulate-scene skill

**Files:**
- Create: `skills/simulate-scene/SKILL.md`

- [ ] **Step 1: Write `skills/simulate-scene/SKILL.md`**

```markdown
---
name: simulate-scene
description: Use when the user wants a test signal or a simulated RF environment with no hardware — translate intent into a scene and register it as a persistent simulated device.
---

# Simulate Scene

Create a simulated device whose "on-air" contents you control. This is the no-hardware on-ramp, and a real workflow in its own right: prototype a receiver against known ground truth before going on-air, or reproduce a colleague's scenario without their antenna.

## Method

1. **Translate intent into elements.** Each element is `{kind, freq, amplitude, params}`:
   - `tone` — an unmodulated carrier at `freq`.
   - `fm_tone` — an NBFM voice carrier; `params.mod_freq` is the modulating audio tone (e.g. 1000 for 1 kHz). Renders only when the capture `sample_rate` is a multiple of 100000.
   - `noise` — a noise floor (`freq` ignored).
   - `iq_file` — replay a recorded capture (usually created via transmit, not by hand).
2. **Always add a noise floor.** Include a `noise` element (amplitude ≈ 0.005). A noiseless scene produces analysis artifacts (phantom sidebands) that confuse survey-spectrum.
3. **Register it.** `simulate_scene(device_id, elements)`. The scene is saved to `scenes/<device_id>.yaml` and the device reappears in later sessions — it's a shareable, git-friendly fixture.
4. **Verify the scene is what you intended.** `capture` the device and run survey-spectrum; confirm the signals land where you placed them.

## Judgment

- Keep amplitudes sane: a strong carrier ≈ 1.0, a weaker one ≈ 0.3–0.5, noise ≈ 0.005. Huge dynamic range makes weak signals invisible.
- Place signals comfortably inside the captured band — elements beyond ±45% of the sample rate from the center are skipped at render time.
- Name devices for what they represent (`fm_band`, `ism_test`), since the name becomes the scene filename.
```

- [ ] **Step 2: Run the validator**

Run: `uv run pytest tests/marconi/test_plugin_skills.py -v`
Expected: PASS.

- [ ] **Step 3: Commit**

```bash
git add skills/simulate-scene/SKILL.md
git commit -m "Add simulate-scene skill"
```

---

## Task 16: tx-experiment skill

**Files:**
- Create: `skills/tx-experiment/SKILL.md`

- [ ] **Step 1: Write `skills/tx-experiment/SKILL.md`**

```markdown
---
name: tx-experiment
description: Use when the user wants a closed-loop transmit-then-receive experiment in simulation — transmit a capture into a simulated device's scene and receive it back to confirm the link.
---

# TX/RX Experiment

Run a closed loop entirely in simulation: transmit a signal into a device's scene, then capture from that device and confirm the signal arrived. This is the basis for experiments (e.g. recover a transmitted tone, sanity-check a waveform).

## Method

1. **Prepare the payload capture.** Record one (`capture`), synthesize one (`render_scene` with inline elements), or `load_capture` a file. **Note its `sample_rate`** — v1.0 requires the receiving capture to use the *same* sample rate.
2. **Choose the receiving device.** Register one with simulate-scene (typically just a noise floor), or reuse an existing simulated device.
3. **Transmit — deliberately.** `transmit_capture(device_id, capture_path, freq, confirmed=True)`. `CONFIRM_TX` is on by default to catch wrong-frequency / wrong-device mistakes; set `confirmed=True` only after you've checked both. The payload becomes an `iq_file` element of the device's scene. (This change is session-scoped — it is not re-persisted to the scene file.)
4. **Receive.** `capture` the device at a center/sample_rate that covers the transmit frequency (sample rate must match the payload's), then run survey-spectrum (or build-receiver) and confirm the payload is present where you transmitted it.
5. **Report** what you transmitted (file, frequency, amplitude) and what you received (detected signal, SNR), so the experiment is reproducible.

## Judgment

- Frequency bookkeeping: the payload is placed at `freq − capture_center_freq` within the received band. Keep the transmit frequency inside the receive band.
- Match sample rates end to end, or the render will refuse with a clear error.
- TX gating is a mistake-catcher, not a licensing check — but still confirm the numbers before transmitting.
```

- [ ] **Step 2: Run the validator**

Run: `uv run pytest tests/marconi/test_plugin_skills.py -v`
Expected: PASS.

- [ ] **Step 3: Commit**

```bash
git add skills/tx-experiment/SKILL.md
git commit -m "Add tx-experiment skill"
```

---

## Task 17: escape-hatch skill

**Files:**
- Create: `skills/escape-hatch/SKILL.md`

- [ ] **Step 1: Write `skills/escape-hatch/SKILL.md`**

```markdown
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

1. Write a Python script that `import marconi` and uses the library API directly — `Workspace`, the ops (`capture`, `find_signals`, `run_pipeline`, …), and the models. Read sample data only via `marconi.read_capture` / the library readers; stay inside the workspace.
2. Run it with `uv run python <script.py>`.
3. **Report the gap explicitly** to the user: which operation or block was missing. These gaps are the product roadmap — they're how the toolkit grows.
4. Leave the script in the workspace. Like pipelines and scenes, it's a durable, re-runnable artifact the user owns.

## Judgment

- Prefer extending the workflow with existing ops over hand-rolling DSP. If you find yourself reimplementing `find_signals` or a flowgraph in numpy, you've probably missed a tool.
- Keep escape scripts small and single-purpose; they document the gap.
```

- [ ] **Step 2: Run the validator (now covers all six skills)**

Run: `uv run pytest tests/marconi/test_plugin_skills.py -v`
Expected: PASS — six skills, all with valid frontmatter.

- [ ] **Step 3: Verify the full expected set is present**

Run: `uv run python -c "from pathlib import Path; s={p.parent.name for p in Path('skills').glob('*/SKILL.md')}; print(sorted(s)); assert s=={'survey-spectrum','build-receiver','debug-no-signal','simulate-scene','tx-experiment','escape-hatch'}"`
Expected: the sorted list of all six skills, no assertion error.

- [ ] **Step 4: Commit**

```bash
git add skills/escape-hatch/SKILL.md
git commit -m "Add escape-hatch skill (completes the six-skill set)"
```

---

## Task 18: Skill-sequence evals (incl. the anti-overfit eval)

**Files:**
- Test: `tests/marconi/test_skill_evals.py`

These encode each skill's documented tool sequence as an executable check that the workflow reaches its goal. The FM and TX/RX loops are already covered by Task 10; this task adds the survey, simulate, and — crucially — the **anti-overfit** eval that build-receiver works on a *non-FM* target (the CW carrier), which the FM demo never exercises. Requires GNU Radio.

- [ ] **Step 1: Write the failing test**

```python
# tests/marconi/test_skill_evals.py
from marconi.mcp import tools as T


def test_simulate_then_survey_sequence(server_state):
    # simulate-scene → survey-spectrum documented sequence
    T.simulate_scene("band", [
        {"kind": "tone", "freq": 100.2e6, "amplitude": 0.6},
        {"kind": "tone", "freq": 99.7e6, "amplitude": 0.5},
        {"kind": "noise", "amplitude": 0.005},
    ])
    cap = T.capture("band", center_freq=100e6, sample_rate=1e6, duration=0.1)
    sigs = T.find_signals(cap["path"])
    img = T.spectrogram(cap["path"])
    assert len(sigs) >= 2
    assert img["path"].endswith(".png")


def test_build_receiver_anti_overfit_cw_carrier(server_state):
    """build-receiver's CW worked example on a target the FM demo never uses:
    channelize the 99.5 MHz carrier to baseband and confirm it via measure."""
    T.simulate_scene("sim0", [
        {"kind": "fm_tone", "freq": 100.3e6, "amplitude": 1.0, "params": {"mod_freq": 1e3}},
        {"kind": "tone", "freq": 99.5e6, "amplitude": 0.5},
        {"kind": "noise", "amplitude": 0.005},
    ])
    cap = T.capture("sim0", center_freq=100e6, sample_rate=2e6, duration=0.1)
    # Channelize the CW carrier (offset = 99.5 - 100 = -0.5 MHz) down to baseband.
    chan_path = str(server_state.workspace.root / "captures" / "cw_chan.bin")
    pipeline = {
        "name": "cw_rx", "sample_rate": 2e6,
        "blocks": [
            {"id": "src", "type": "file_source", "params": {"path": cap["path"]}},
            {"id": "chan", "type": "freq_xlating_lowpass", "params": {
                "decimation": 40, "center_offset": -0.5e6, "cutoff": 10e3, "transition": 5e3}},
            {"id": "head", "type": "head", "params": {"num_samples": 5000}},
            {"id": "sink", "type": "file_sink", "params": {"path": chan_path}},
        ],
        "connections": [
            {"src_block": "src", "dst_block": "chan"},
            {"src_block": "chan", "dst_block": "head"},
            {"src_block": "head", "dst_block": "sink"},
        ],
    }
    assert T.validate_pipeline(pipeline) == []
    assert T.run_pipeline(pipeline)["status"] == "ok"
    # The channelized capture is at 2e6/40 = 50 kHz, centered on the carrier.
    from marconi import sigmf
    sigmf.write_meta_for(chan_path, center_freq=99.5e6, sample_rate=50e3)
    m = T.measure(chan_path, center_freq=99.5e6, search_bandwidth=40e3)
    assert m["snr_db"] > 8  # the carrier is clearly present at baseband
```

- [ ] **Step 2: Run test to verify it fails (or surfaces real issues)**

Run: `uv run pytest tests/marconi/test_skill_evals.py -v`
Expected: runs GNU Radio; if the CW eval's numbers need tuning (decimation/cutoff), adjust to make the carrier measurable — the *point* is that build-receiver's non-FM path works. Use systematic-debugging if `snr_db` is low (check the offset sign and that 99.5 MHz lands at baseband).

- [ ] **Step 3: Commit**

```bash
git add tests/marconi/test_skill_evals.py
git commit -m "Add skill-sequence evals incl. anti-overfit CW receiver eval"
```

---

## Task 19: Documentation — README, CLAUDE.md, generality principle

**Files:**
- Modify: `README.md`, `CLAUDE.md`

- [ ] **Step 1: Update `README.md`**

Add an "Install as a Claude Code plugin" section after "Quick start", and a "Skills" subsection. Insert this block:

```markdown
## Use it from Claude Code

Marconi ships as a Claude Code plugin: an MCP server exposing the operations as
tools, plus skills that carry the RF workflow.

```bash
# add this repo as a plugin marketplace, then install the marconi plugin
/plugin marketplace add yoelbassin/gr-mcp
/plugin install marconi
```

The plugin starts the `marconi-mcp` server (via `uv run --extra mcp marconi-mcp`)
and loads six skills: **survey-spectrum**, **build-receiver**, **debug-no-signal**,
**simulate-scene**, **tx-experiment**, and **escape-hatch** (last resort). Ask in
natural language — "simulate an FM station at 100.3 MHz and build me a receiver"
— and the agent surveys, builds, runs, verifies, and leaves the pipeline/capture
artifacts in your workspace.

The MCP server's workspace is its working directory; override with the
`MARCONI_WORKSPACE` environment variable.
```

- [ ] **Step 2: Update `CLAUDE.md`**

Under the Module layout, add the `mcp/` subpackage; and add a short "Skills authoring" note capturing the generality principle. Insert into the module-layout block:

```
  mcp/
    errors.py           # exception -> structured tool-error translation
    state.py            # ServerState: workspace + persisted devices + run history
    tools.py            # the 19 tool functions + TOOLS registry (thin marshalling)
    server.py           # build_server() + main() (stdio FastMCP)
```

And add this section after "Key constraints":

```markdown
## Skills authoring (the generality principle)

Skills under `skills/` are the product's judgment layer. Write them to encode
transferable *method*, not demo recipes: lead with the general decision
framework and show a specific signal only as a worked example. The review test
for any skill is **"could an RF engineer follow this for AM or SSB, not just
FM?"** — if not, it's too specific. Content breadth (new modulations, digital
demod) grows deliberately via the roadmap, not under demo pressure.
```

- [ ] **Step 3: Verify docs reference reality**

Run: `uv run python -c "import json,sys; sys.exit(0 if json.load(open('.mcp.json'))['mcpServers']['marconi']['args'][-1]=='marconi-mcp' else 1)"`
Expected: exit 0 (README's `uv run marconi-mcp` claim matches the manifest).

- [ ] **Step 4: Run the full suite one last time**

Run: `uv run pytest tests/marconi -q`
Expected: all tests pass (Plans 1–2 suite + the new MCP/plugin/skill tests).

- [ ] **Step 5: Confirm the base-import invariant**

Run: `uv run python -c "import marconi, sys; assert 'gnuradio' not in sys.modules and 'fastmcp' not in sys.modules; print('clean')"`
Expected: `clean`

- [ ] **Step 6: Commit**

```bash
git add README.md CLAUDE.md
git commit -m "Document the Claude Code plugin, MCP layer, and skills authoring principle"
```

---

## Self-Review

**1. Spec coverage** (against `2026-06-13-marconi-plan3-agentic-layer-design.md`):

| Spec section | Task(s) |
|---|---|
| MCP server, thin, one-tool-per-op | 4–9 |
| Tool surface: devices/capture | 5 |
| Tool surface: analyze | 6 |
| Tool surface: render | 7 |
| Tool surface: pipeline + list_blocks + list_runs | 4, 8 |
| Tool surface: simulate/tx | 9 |
| Transport stdio + workspace = CWD / `MARCONI_WORKSPACE` | 1, 4 |
| State model A: persisted devices + run history (synchronous runs) | 3, 5, 8 |
| Error translation boundary (all exception types) | 2 |
| Six skills | 12–17 |
| Generality guardrails: method-not-recipe; CW second example; anti-overfit eval; stated principle | 13 (CW), 18 (anti-overfit), 19 (principle) |
| Plugin packaging (plugin/marketplace/.mcp.json + entry point + deps) | 1, 11 |
| Testing: MCP layer in-process | 2–9 |
| Testing: end-to-end through tools (FM + TX/RX) | 10 |
| Testing: light skill evals | 18 |
| ADS-B / digital demod deferred | (out of scope — referenced in escape-hatch skill, Task 17) |

No gaps.

**2. Placeholder scan:** The only intentional staging is the `TOOLS` registry growing across Tasks 4–9, with the `== 19` assertions explicitly relaxed in Task 4 and tightened in Task 9 (Step 4). No "TBD"/"add error handling"/uncoded steps remain.

**3. Type consistency:** Tool functions reference captures by `capture_path: str` and reconstruct via `_ref` (= `sigmf.read_meta`) everywhere; pipeline tools take `pipeline: dict` validated via `PipelineSpec.model_validate`; `ServerState.record_run`/`next_run_id`/`runs` names match between Task 3 (definition) and Task 8 (use); `classify_error`/`to_tool_error`/`tool_error_boundary` names match between Task 2 and its consumers. `build_server`/`main`/`set_state`/`get_state`/`reset_state` are consistent across Tasks 1, 3, 4. The 19 tool names in the registry match the spec's tool table.
