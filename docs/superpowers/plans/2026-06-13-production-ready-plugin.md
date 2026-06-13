# Production-Ready Plugin Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make Marconi production-ready as a Claude Code plugin — add CI, confirm the plugin structure, and ensure simple marketplace installation — without adding product features.

**Architecture:** Three independent workstreams over the existing repo. (1) A pytest `gnuradio` marker plus a conftest auto-skip hook splits the suite into an engine-agnostic set and a GNU-Radio set, so a fast CI job and any GR-less machine stay green. (2) A two-job GitHub Actions workflow: a fast `core` gate (lint + engine-agnostic tests, no GNU Radio) and a `backend` job (full suite with conda-forge GNU Radio matched to Python 3.13). (3) Version alignment to 1.0.0 and verified install docs.

**Tech Stack:** Python 3.13, uv, pytest, pre-commit (black/isort/flake8/mypy), GitHub Actions, micromamba + conda-forge GNU Radio.

---

## Background the implementer needs

- **Engine isolation is a hard invariant.** Only `src/marconi/backends/gnuradio_backend.py` imports `gnuradio`, and only lazily inside functions (`_modules()`). `import marconi` must work with no GNU Radio installed. No test module imports GNU Radio at collection time (verified: collecting the whole suite with GNU Radio blocked yields 0 import errors).
- **GNU Radio is coupled to the Python interpreter.** Its compiled bindings must match the interpreter and its numpy ABI. This is why the backend CI job gets `gnuradio` + `numpy`/`scipy`/`matplotlib` from conda-forge (one consistent stack) and uses pip only for pure-Python packages.
- **The verification trick.** GNU Radio is installed on the dev machine, so to prove the marker set is complete you simulate its absence in-process: `sys.modules['gnuradio'] = None` makes any `import gnuradio` raise `ImportError`, exactly like the `core` CI job. The canonical check is:

  ```bash
  uv run python -c "import sys; sys.modules['gnuradio']=None; import pytest; raise SystemExit(pytest.main(['-m','not gnuradio','-q','tests']))"
  ```

  Expected: all selected tests pass, **zero errors, zero failures**. Any test that errors here with `BackendError`/`ImportError` is an unmarked GNU-Radio test — add the marker to it.

## File structure

| File | Change | Responsibility |
| --- | --- | --- |
| `pyproject.toml` | modify | Register the `gnuradio` pytest marker; bump version to `1.0.0`. |
| `tests/conftest.py` | modify | Auto-skip `gnuradio`-marked tests when GNU Radio is not importable. |
| `tests/marconi/backends/test_backend_gr.py` | modify | Mark the 4 flowgraph-executing tests. |
| `tests/marconi/ops/test_simulate.py` | modify | Mark the 1 render test. |
| `tests/marconi/ops/test_pipeline.py` | modify | Mark the 1 run test (validate/save tests stay unmarked). |
| `tests/marconi/ops/test_transmit.py` | modify | Module-level mark (all tests render/capture). |
| `tests/marconi/test_end_to_end_simulation.py` | modify | Mark the 1 full-loop test. |
| `tests/marconi/test_devices.py` | modify | Mark the 1 simulated-capture test. |
| `tests/marconi/mcp/test_e2e.py` | modify | Module-level mark. |
| `tests/marconi/mcp/test_tools_simulate_tx.py` | modify | Module-level mark. |
| `tests/marconi/mcp/test_tools_pipeline.py` | modify | Mark the 1 run test. |
| `tests/marconi/mcp/test_tools_devices.py` | modify | Mark the 1 capture test (add `import pytest`). |
| `tests/plugin/test_skill_evals.py` | modify | Module-level mark (add `import pytest`). |
| `tests/plugin/test_manifests.py` | modify | Guard that `plugin.json` and `pyproject.toml` versions stay in sync. |
| `.claude-plugin/plugin.json` | modify | Bump version to `1.0.0`. |
| `.github/workflows/ci.yml` | create | Two-job CI: `core` + `backend`. |
| `README.md` | modify | Make the install command explicit; fix typo. |

---

## Task 1: Register the `gnuradio` marker and add the auto-skip hook

**Files:**
- Modify: `pyproject.toml` (`[tool.pytest.ini_options]`)
- Modify: `tests/conftest.py`

- [ ] **Step 1: Register the marker in `pyproject.toml`**

In the existing `[tool.pytest.ini_options]` block, add a `markers` array. The block currently reads:

```toml
[tool.pytest.ini_options]
# src on the path so `import marconi` resolves without an install step
pythonpath = ["src", "."]
# tests/marconi/ mirrors src/marconi/ (ops/ mcp/ backends/); tests/plugin/ holds
# the Layer-3 (skills + manifests) tests. One root so a bare `pytest` finds both.
testpaths = ["tests"]
asyncio_mode = "auto"
```

Change it to:

```toml
[tool.pytest.ini_options]
# src on the path so `import marconi` resolves without an install step
pythonpath = ["src", "."]
# tests/marconi/ mirrors src/marconi/ (ops/ mcp/ backends/); tests/plugin/ holds
# the Layer-3 (skills + manifests) tests. One root so a bare `pytest` finds both.
testpaths = ["tests"]
asyncio_mode = "auto"
markers = [
    "gnuradio: test executes a GNU Radio flowgraph; auto-skipped when GNU Radio is unavailable.",
]
```

- [ ] **Step 2: Add the auto-skip hook to `tests/conftest.py`**

Insert this block immediately after the existing imports and the `MakeIQ = Callable[..., np.ndarray]` line, before the first `@pytest.fixture`:

```python
def _gnuradio_importable() -> bool:
    try:
        import gnuradio  # noqa: F401
    except Exception:
        return False
    return True


_GNURADIO = _gnuradio_importable()


def pytest_collection_modifyitems(
    config: pytest.Config, items: list[pytest.Item]
) -> None:
    """Auto-skip tests marked `gnuradio` when GNU Radio is not importable, so a
    bare `pytest` is green on machines (and CI jobs) without a DSP runtime."""
    if _GNURADIO:
        return
    skip = pytest.mark.skip(reason="GNU Radio is not installed")
    for item in items:
        if "gnuradio" in item.keywords:
            item.add_marker(skip)
```

- [ ] **Step 3: Verify the marker is registered (no unknown-mark warning)**

Run: `uv run pytest --markers | grep gnuradio`
Expected: a line `@pytest.mark.gnuradio: test executes a GNU Radio flowgraph; ...`

- [ ] **Step 4: Verify the full suite still passes (GNU Radio present on dev machine)**

Run: `uv run pytest -q`
Expected: all tests pass (nothing is marked yet, so this is unchanged behavior).

- [ ] **Step 5: Commit**

```bash
git add pyproject.toml tests/conftest.py
git commit -m "test: add gnuradio marker and auto-skip hook"
```

---

## Task 2: Mark every GNU-Radio-dependent test

**Files:**
- Modify (per-test marks): `tests/marconi/backends/test_backend_gr.py`, `tests/marconi/ops/test_simulate.py`, `tests/marconi/ops/test_pipeline.py`, `tests/marconi/test_end_to_end_simulation.py`, `tests/marconi/test_devices.py`, `tests/marconi/mcp/test_tools_pipeline.py`, `tests/marconi/mcp/test_tools_devices.py`
- Modify (module-level marks): `tests/marconi/ops/test_transmit.py`, `tests/marconi/mcp/test_e2e.py`, `tests/marconi/mcp/test_tools_simulate_tx.py`, `tests/plugin/test_skill_evals.py`

The set below was derived by tracing which tests reach `_modules()` (via `run_pipeline`, `build_top_block`, `render_scene`, or `capture()` on a simulated device — note `simulate_scene` only *registers* a device and does **not** run a flowgraph, so register-only tests stay unmarked). Step 4 is the source of truth — it catches anything missed.

- [ ] **Step 1: Per-test marks in mixed files**

Add the decorator `@pytest.mark.gnuradio` on the line directly above each of these `def`s. Most of these files already `import pytest`; `tests/marconi/mcp/test_tools_devices.py` does **not** — for that file, first add the import (its header is `from pathlib import Path` / blank / `from marconi.mcp import tools as T`; change it to):

```python
from pathlib import Path

import pytest

from marconi.mcp import tools as T
```

`tests/marconi/backends/test_backend_gr.py`:
- above `def test_build_top_block(tmp_path) -> None:`
- above `def test_build_error_carries_block_id(tmp_path) -> None:`
- above `def test_run_pipeline_produces_analyzable_capture(tmp_path) -> None:`
- above `def test_run_pipeline_timeout(tmp_path) -> None:`

`tests/marconi/ops/test_simulate.py`:
- above `def test_render_scene_produces_findable_signals(tmp_path: Path) -> None:`

`tests/marconi/ops/test_pipeline.py`:
- above `def test_run_executes_valid_pipeline(tmp_path: Path) -> None:`
  (leave `test_run_validates_first` and `test_save_pipeline_to_workspace` unmarked — they raise/validate/save without running)

`tests/marconi/test_end_to_end_simulation.py`:
- above `def test_killer_demo_full_loop(tmp_path: Path) -> None:`

`tests/marconi/test_devices.py`:
- above `def test_capture_from_simulated_device(tmp_path: Path) -> None:`

`tests/marconi/mcp/test_tools_pipeline.py`:
- above `def test_run_pipeline_records_history(server_state, tmp_path):`

`tests/marconi/mcp/test_tools_devices.py` (after adding the import above):
- above `def test_capture_returns_reference(server_state):`
  (leave the two `simulate_scene`-only tests and `test_load_capture_cf32` unmarked)

Example (the decorator goes immediately above the existing `def`, same indentation as `def`):

```python
@pytest.mark.gnuradio
def test_build_top_block(tmp_path) -> None:
    from marconi.backends.gnuradio_backend import build_top_block
    ...
```

- [ ] **Step 2: Module-level marks in fully-GNU-Radio files**

For `tests/marconi/ops/test_transmit.py` and `tests/marconi/mcp/test_tools_simulate_tx.py` (both already `import pytest`), add this line right after the import block, separated by a blank line:

```python
pytestmark = pytest.mark.gnuradio
```

For `tests/marconi/mcp/test_e2e.py`, which does **not** import pytest, change the top of the file from:

```python
import numpy as np
from scipy.io import wavfile

from marconi.mcp import tools as T
```

to:

```python
import numpy as np
import pytest
from scipy.io import wavfile

from marconi.mcp import tools as T

pytestmark = pytest.mark.gnuradio
```

For `tests/plugin/test_skill_evals.py`, which also does **not** import pytest, change the top of the file from:

```python
from marconi.mcp import tools as T
```

to:

```python
import pytest

from marconi.mcp import tools as T

pytestmark = pytest.mark.gnuradio
```

- [ ] **Step 3: Confirm the marked count (GNU Radio present)**

Run: `uv run pytest -m gnuradio --collect-only -q | tail -3`
Expected: **23 tests** collected. Breakdown — per-test: `test_backend_gr.py` 4, `test_simulate.py` 1, `test_pipeline.py` 1, `test_end_to_end_simulation.py` 1, `test_devices.py` 1, `test_tools_pipeline.py` 1, `test_tools_devices.py` 1, `test_block_coverage.py` 1 — `test_factories_cover_exactly_the_vocabulary` calls `_factories()` which imports GNU Radio; its sibling stays unmarked (= 11); module-level: `test_transmit.py` 5, `test_e2e.py` 2, `test_tools_simulate_tx.py` 3, `test_skill_evals.py` 2 (= 12).

- [ ] **Step 4: Verify completeness — run the engine-agnostic set with GNU Radio blocked**

Run:

```bash
uv run python -c "import sys; sys.modules['gnuradio']=None; import pytest; raise SystemExit(pytest.main(['-m','not gnuradio','-q','tests']))"
```

Expected: PASS — zero errors, zero failures. (This is exactly what the `core` CI job will run.)
If any test errors with `BackendError: GNU Radio is not importable` or an `ImportError` from `gnuradio`, it is an unmarked GNU-Radio test: add `@pytest.mark.gnuradio` above it (or `pytestmark` if the whole file needs it) and re-run until clean.

- [ ] **Step 5: Verify the auto-skip path (GNU Radio blocked, bare run)**

Run:

```bash
uv run python -c "import sys; sys.modules['gnuradio']=None; import pytest; raise SystemExit(pytest.main(['-q','tests']))"
```

Expected: PASS with **23 skipped** (reason: "GNU Radio is not installed"), zero failures. This proves a bare `pytest` is green without GNU Radio.

- [ ] **Step 6: Commit**

```bash
git add tests/
git commit -m "test: mark GNU Radio flowgraph tests for engine-agnostic CI"
```

---

## Task 3: Align version to 1.0.0 with a sync guard

**Files:**
- Modify: `tests/plugin/test_manifests.py`
- Modify: `.claude-plugin/plugin.json`
- Modify: `pyproject.toml`

- [ ] **Step 1: Add a version-sync test (red-green driver)**

Append this test to `tests/plugin/test_manifests.py`. It reads the version from both manifests and asserts they match. (`tomllib` is stdlib on Python ≥ 3.11; `requires-python` is ≥ 3.13.)

```python
import tomllib


def test_plugin_version_matches_pyproject():
    plugin = json.loads((ROOT / ".claude-plugin" / "plugin.json").read_text())
    pyproject = tomllib.loads((ROOT / "pyproject.toml").read_text())
    assert plugin["version"] == pyproject["project"]["version"]
```

Place the `import tomllib` with the other top-of-file imports (after `import json`), so the import block reads:

```python
import json
import tomllib
from pathlib import Path
```

- [ ] **Step 2: Run the sync test — it passes at the current matched version**

Run: `uv run pytest tests/plugin/test_manifests.py::test_plugin_version_matches_pyproject -v`
Expected: PASS (both manifests are currently `0.1.0`).

- [ ] **Step 3: Bump `.claude-plugin/plugin.json` only, to prove the guard fails on drift**

In `.claude-plugin/plugin.json`, change:

```json
  "version": "0.1.0",
```

to:

```json
  "version": "1.0.0",
```

- [ ] **Step 4: Run the sync test — it now fails (intentional)**

Run: `uv run pytest tests/plugin/test_manifests.py::test_plugin_version_matches_pyproject -v`
Expected: FAIL — `'1.0.0' == '0.1.0'` assertion error. This confirms the guard catches drift.

- [ ] **Step 5: Bump `pyproject.toml` to match**

In `pyproject.toml`, change the `[project]` version:

```toml
version = "0.1.0"
```

to:

```toml
version = "1.0.0"
```

- [ ] **Step 6: Run the manifest tests — all green**

Run: `uv run pytest tests/plugin/test_manifests.py -v`
Expected: all PASS, including the new sync test (both now `1.0.0`).

- [ ] **Step 7: Commit**

```bash
git add tests/plugin/test_manifests.py .claude-plugin/plugin.json pyproject.toml
git commit -m "chore: bump version to 1.0.0 with a manifest sync guard"
```

---

## Task 4: Add the CI workflow

**Files:**
- Create: `.github/workflows/ci.yml`

- [ ] **Step 1: Create `.github/workflows/ci.yml`**

```yaml
name: CI

on:
  push:
    branches: [main]
  pull_request:

jobs:
  core:
    name: core — lint + engine-agnostic tests (no GNU Radio)
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - name: Install uv
        uses: astral-sh/setup-uv@v6
        with:
          enable-cache: true

      - name: Sync dependencies
        run: uv sync --extra dev

      - name: Engine-isolation guard (import marconi without GNU Radio)
        run: uv run python -c "import marconi"

      - name: Cache pre-commit hooks
        uses: actions/cache@v4
        with:
          path: ~/.cache/pre-commit
          key: pre-commit-${{ hashFiles('.pre-commit-config.yaml') }}

      - name: Lint (pre-commit)
        run: uv run pre-commit run --all-files

      - name: Validate plugin (best-effort, non-blocking)
        continue-on-error: true
        run: |
          npm install -g @anthropic-ai/claude-code
          claude plugin validate .

      - name: Tests (engine-agnostic)
        run: uv run pytest -m "not gnuradio"

  backend:
    name: backend — full suite with GNU Radio
    runs-on: ubuntu-latest
    defaults:
      run:
        shell: bash -el {0}
    steps:
      - uses: actions/checkout@v4

      - name: Set up GNU Radio via conda-forge
        uses: mamba-org/setup-micromamba@v2
        with:
          environment-name: marconi-gr
          condarc: |
            channels:
              - conda-forge
          create-args: >-
            python=3.13
            gnuradio
            numpy
            scipy
            matplotlib
            pydantic
            pyyaml
            pytest
            pytest-asyncio
            pip

      - name: Install pure-Python deps and marconi
        run: |
          pip install fastmcp
          pip install -e . --no-deps

      - name: Sanity — GNU Radio and marconi import together
        run: python -c "import gnuradio, marconi, numpy; print('gnuradio', gnuradio.__file__)"

      - name: Full test suite
        run: pytest
```

Why the backend job installs the compiled stack (`numpy`/`scipy`/`matplotlib`) from conda-forge and uses `pip install -e . --no-deps`: GNU Radio's bindings link against conda-forge's numpy ABI. Letting pip resolve `marconi`'s deps could swap numpy for a PyPI build and break GNU Radio. `fastmcp` is the one runtime dep not reliably on conda-forge, so it comes from pip (pure Python, no numpy ABI coupling).

- [ ] **Step 2: Verify the workflow is valid YAML**

Run: `uv run python -c "import yaml; yaml.safe_load(open('.github/workflows/ci.yml')); print('ok')"`
Expected: prints `ok`.

- [ ] **Step 3: Verify pre-commit passes on the new file locally**

Run: `uv run pre-commit run --files .github/workflows/ci.yml`
Expected: `check-yaml`, `trailing-whitespace`, `end-of-file-fixer` all Passed (others Skipped — not Python).
If `end-of-file-fixer` or `trailing-whitespace` modifies the file, `git add -u` the change.

- [ ] **Step 4: Commit**

```bash
git add .github/workflows/ci.yml
git commit -m "ci: add core + backend GitHub Actions workflow"
```

---

## Task 5: Make the install command explicit and verify the flow

**Files:**
- Modify: `README.md`

The README already documents install and the GNU-Radio-optional requirement. This task makes the install reference explicit (`marconi@marconi` = plugin `marconi` from marketplace `marconi`, the name in `.claude-plugin/marketplace.json`) and fixes a typo found in passing.

- [ ] **Step 1: Make the install command explicit**

In `README.md`, change:

```
/plugin marketplace add yoelbassin/gr-mcp
/plugin install marconi
```

to:

```
/plugin marketplace add yoelbassin/gr-mcp
/plugin install marconi@marconi
```

- [ ] **Step 2: Fix the typo on the "Previously known as" line**

Change:

```
> Proviously known as GNURadio/GR-MCP.
```

to:

```
> Previously known as GNURadio/GR-MCP.
```

- [ ] **Step 3: Verify the marketplace/plugin names referenced in the README match the manifests**

Run:

```bash
uv run python -c "import json; mp=json.load(open('.claude-plugin/marketplace.json')); pl=json.load(open('.claude-plugin/plugin.json')); print('marketplace:', mp['name']); print('plugin:', pl['name']); assert mp['name']=='marconi' and pl['name']=='marconi'"
```

Expected: prints `marketplace: marconi` / `plugin: marconi` with no assertion error, confirming `marconi@marconi` is correct.

- [ ] **Step 4: Verify the plugin loads locally via --plugin-dir**

Run: `claude plugin validate .`
Expected: validation passes (no errors). If the `claude` CLI is unavailable in this environment, skip this step — the `core` CI job and `tests/plugin/test_manifests.py` cover manifest validity.

- [ ] **Step 5: Commit**

```bash
git add README.md
git commit -m "docs: explicit plugin install command and typo fix"
```

---

## Task 6: Full-suite green check

**Files:** none (verification only)

- [ ] **Step 1: Run the complete suite with GNU Radio present**

Run: `uv run pytest -q`
Expected: all tests pass (including the 23 GNU-Radio tests, since GNU Radio is installed on the dev machine).

- [ ] **Step 2: Run the engine-agnostic gate exactly as CI will**

Run:

```bash
uv run python -c "import sys; sys.modules['gnuradio']=None; import pytest; raise SystemExit(pytest.main(['-m','not gnuradio','-q','tests']))"
```

Expected: PASS, zero errors, zero failures.

- [ ] **Step 3: Run lint exactly as CI will**

Run: `uv run pre-commit run --all-files`
Expected: all hooks Passed. If any hook reformats, `git add -u` and amend/commit.

- [ ] **Step 4: Confirm the tree is clean**

Run: `git status --short`
Expected: empty (everything committed).

---

## Self-review notes (spec coverage)

- Spec §1 (structure conformance): Task 3 (version → 1.0.0 + sync guard); `claude plugin validate` in Task 4 (CI, non-blocking) and Task 5 Step 4 (local). Structure already conforms — no directory changes needed.
- Spec §2 (CI, two jobs + marker infra): Tasks 1, 2 (marker + auto-skip + marks), Task 4 (`core` + `backend` jobs).
- Spec §3 (simple installation): Task 5 (explicit install command, name-match verification, local validate) — README already covers prerequisites and the GNU-Radio-optional note.
- Spec risk (GNU Radio/Python coupling): handled in Task 4 via conda-forge for the compiled stack + `pip --no-deps`, with a sanity-import step.
- Spec success criteria: Task 6 covers the green-everywhere checks; the marker auto-skip (Task 2 Step 5) covers "bare pytest skips, not fails."
