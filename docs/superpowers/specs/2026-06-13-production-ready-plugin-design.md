# Marconi production-ready plugin (minimal)

**Date:** 2026-06-13
**Status:** Approved — ready for implementation planning

## Goal

Make Marconi production-ready as a Claude Code plugin with the minimal set of
changes. Three deliverables, nothing more:

1. **Conform to the plugin development guide's structure** — the layout is
   mostly already correct; verify and tidy.
2. **Add CI** — a fast quality gate plus a backend job that exercises GNU Radio.
3. **Simple plugin installation** — via the repo's self-hosted marketplace,
   documented and verified.

The codebase already builds all three architectural layers (core, MCP server,
plugin). This pass adds the operational scaffolding that makes it installable
and continuously verified, without adding product features.

## Non-goals (YAGNI)

Explicitly out of scope for this pass:

- Submitting to the public `anthropics/claude-plugins-community` marketplace.
- Publishing the Python package to PyPI.
- Release automation, changelogs, or semantic-release.
- Decode/hardware/bench features (these are the roadmap, not this pass).
- Coverage badges, Dependabot, or other repo-decoration.

## Context

Findings from the current repository that shape the design:

- The plugin scaffolding already matches the guide: `.claude-plugin/plugin.json`,
  `.claude-plugin/marketplace.json` (self-hosted, `source: "./"`), a root
  `.mcp.json` that launches the server with `${CLAUDE_PLUGIN_ROOT}`, and
  `skills/<name>/SKILL.md` directories. No stray dirs live under
  `.claude-plugin/`.
- There is **no CI** (`.github/` does not exist).
- Lint is already codified in `.pre-commit-config.yaml`: black, isort,
  flake8 (`--max-line-length=88`), mypy (with `types-PyYAML`), plus hygiene
  hooks (trailing-whitespace, end-of-file-fixer, check-yaml, debug-statements,
  name-tests-test).
- Version is inconsistent: `plugin.json` and `pyproject.toml` say `0.1.0`, while
  `ROADMAP.md` and `CLAUDE.md` describe "v1.0 shipped."
- `tests/plugin/test_manifests.py` already validates all three manifests in the
  suite.
- **GNU Radio is coupled to the Python interpreter version.** Locally the dev
  machine runs Python 3.14 with Homebrew's GNU Radio
  (`/opt/homebrew/.../python3.14/site-packages/gnuradio`). `requires-python` is
  `>=3.13`. Ubuntu's apt `gnuradio` is built for the distro's system Python
  (3.12 on 24.04), which would not match a 3.13 venv. This coupling is the one
  nontrivial CI risk and drives the backend-job design below.
- The test suite has **no mechanism today** to skip GNU-Radio-dependent tests
  when GNU Radio is absent; the flowgraph-executing tests hard-fail on import.

## Design

### 1. Plugin structure conformance (goal #1)

The directory layout already conforms to the guide. Concrete changes:

- **Bump version to `1.0.0`** in both `.claude-plugin/plugin.json` and
  `pyproject.toml`, aligning the manifests with `ROADMAP.md`/`CLAUDE.md` which
  already describe v1.0 as shipped. This is the production-ready signal.
- **Run `claude plugin validate`** (the official validator the community
  pipeline uses; a pure local check requiring no auth) and fix anything it
  flags.
- Keep `tests/plugin/test_manifests.py` as the in-suite manifest guard.

### 2. CI (goal #2)

A single workflow at `.github/workflows/ci.yml`, triggered on push to `main`
and on every pull request. Two jobs, both **required** (the backend job is
simply allowed to run slower):

**Job `core`** — fast gate, ubuntu-latest, no GNU Radio:

1. Install `uv`; `uv sync --extra dev`.
2. **Engine-isolation guard:** `uv run python -c "import marconi"` must succeed
   with no GNU Radio on the path — enforces the architecture's core invariant
   that `import marconi` works without a DSP runtime.
3. **Lint:** `uv run pre-commit run --all-files` (reuses the exact local hooks).
4. **Plugin validate:** `claude plugin validate` (Claude Code installed via npm).
   This step is optional — dropped if it turns out to require auth in CI.
5. **Tests:** `uv run pytest -m "not gnuradio"` — the engine-agnostic, MCP, and
   plugin tests.

**Job `backend`** — full coverage, ubuntu-latest + micromamba:

1. `setup-micromamba` with an environment containing `conda-forge::gnuradio`
   and `python=3.13`. conda-forge provides a GNU Radio build matched to the
   pinned Python, sidestepping the apt version-mismatch.
2. `pip install -e ".[dev]"` into that environment.
3. **Tests:** `pytest` — the full suite, including the GNU-Radio-marked tests.

Both jobs cache their package managers (uv cache; micromamba env) for speed.

**Marker infrastructure** — makes the core/backend split clean and keeps
GR-less dev machines green:

- Register a `gnuradio` marker in `pyproject.toml` (`[tool.pytest.ini_options]`
  `markers`).
- Add a `pytest_collection_modifyitems` hook in `tests/conftest.py` that
  auto-skips any `@pytest.mark.gnuradio` test when `import gnuradio` fails. This
  means a bare `pytest` does the right thing everywhere: full suite where GNU
  Radio is installed, GR tests skipped where it is not.
- Apply `@pytest.mark.gnuradio` to the flowgraph-executing tests. The candidate
  set is `backends/test_backend_gr.py`, `ops/test_simulate.py`,
  `ops/test_transmit.py`, `test_end_to_end_simulation.py`, `mcp/test_e2e.py`,
  `mcp/test_tools_simulate_tx.py`, and the run-path cases in
  `mcp/test_tools_pipeline.py`. The exact set is finalized during
  implementation by identifying which tests actually spin up a `top_block`
  (tests that only build/validate/export a pipeline do **not** need the marker).

### 3. Simple installation (goal #3)

- **README install section** documenting the self-hosted marketplace flow:

  ```
  /plugin marketplace add yoelbassin/gr-mcp
  /plugin install marconi@marconi
  ```

  With prerequisites stated plainly:
  - `uv` must be installed; it auto-provisions Python 3.13.
  - **GNU Radio 3.10+ is only needed for `run`/`simulate`/`transmit`.** Survey,
    analyze, render, and pipeline-building work without it.
  - The first MCP launch runs `uv sync` (needs network once).

- **Verify the flow locally** before claiming it works — via
  `claude plugin marketplace add ./` followed by install, or `--plugin-dir`.

## Files touched

- **Add:** `.github/workflows/ci.yml`
- **Edit:**
  - `tests/conftest.py` — `gnuradio` marker auto-skip hook.
  - `pyproject.toml` — register `gnuradio` marker; bump version to `1.0.0`.
  - `.claude-plugin/plugin.json` — bump version to `1.0.0`.
  - the flowgraph-executing test files — add `@pytest.mark.gnuradio`.
  - `README.md` — install section.

## Risk

The only nontrivial risk is the **GNU Radio / Python version coupling in CI**,
addressed by using conda-forge GNU Radio (matched to Python 3.13) for the
backend job rather than apt. If conda resolution proves slow or flaky, the
fallback is to relax the backend job to whatever Python version conda-forge's
current `gnuradio` is built against, keeping that job's environment internally
consistent.

## Success criteria

- `.github/workflows/ci.yml` runs on PRs and pushes to `main`; the `core` job
  is green without GNU Radio and the `backend` job exercises the full suite
  with GNU Radio.
- `import marconi` succeeds in CI with no GNU Radio installed.
- A bare `pytest` skips (does not fail) GNU-Radio tests when GNU Radio is
  absent.
- Manifest versions read `1.0.0` and `claude plugin validate` passes.
- A new user can install Marconi by adding the marketplace and running
  `/plugin install marconi@marconi`, following only the README.
