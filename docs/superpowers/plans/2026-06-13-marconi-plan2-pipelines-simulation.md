# Marconi v1.0 — Plan 2 of 3: Pipelines, Backend, Simulation

> **Historical — implementation plan, since executed.** Kept for provenance; for current status and sequencing see `ROADMAP.md`.

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make Marconi generate and run signals: engine-agnostic pipeline/scene/device models, a curated block vocabulary with validation, the backend interface, the GNU Radio backend (direct `gr.top_block` construction with bounded, timeout-supervised runs), scene rendering, SimulatedDevice + capture + transmit, and `.grc` export — culminating in the killer-demo as an automated test.

**Architecture:** Layer 1 of the design spec (`docs/superpowers/specs/2026-06-12-marconi-design.md`) gains its execution half. Everything above `backends/` stays GNU-Radio-free; the backend boundary is "moves samples or touches devices." Scene rendering is implemented as *generated pipelines* run through the same backend machinery (dogfooding). Plan 1's analyze ops verify Plan 2's signal generation in tests — the loop closes.

**Tech Stack:** Python ≥3.13, pydantic, pyyaml (already a dep), numpy, GNU Radio 3.10.12 (system, importable in the uv venv on this machine — verified), `grcc` at `/opt/homebrew/bin/grcc`.

---

## Verified GNU Radio facts (do NOT re-derive; these were tested against the installed 3.10.12)

- `import gnuradio` works directly under `uv run` on this machine; the root `tests/conftest.py` already imports it unconditionally, so all pytest runs require it anyway.
- Constructors verified working:
  - `analog.sig_source_c(samp_rate, analog.GR_COS_WAVE, freq, ampl, 0)`, same `_f`
  - `analog.noise_source_c(analog.GR_GAUSSIAN, amplitude, seed)`
  - `blocks.add_vcc(1)`, `blocks.multiply_const_cc(k)`, `blocks.rotator_cc(phase_inc)`
  - `blocks.head(gr.sizeof_gr_complex, n)`
  - `blocks.file_sink(gr.sizeof_gr_complex, path_str, False)`, `blocks.file_source(gr.sizeof_gr_complex, path_str, repeat_bool)`
  - `blocks.wavfile_sink(path_str, 1, rate_int, blocks.FORMAT_WAV, blocks.FORMAT_PCM_16, False)`
  - `analog.quadrature_demod_cf(gain)`
  - `analog.fm_deemph(fs, tau=7.5e-05)` (python hier block)
  - `analog.nbfm_rx(audio_rate, quad_rate, tau=7.5e-05, max_dev=5000.0)` — output rate = audio_rate
  - `analog.nbfm_tx(audio_rate, quad_rate, tau=7.5e-05, max_dev=5000.0, fh=-1.0)` — **fails with "insufficient extremals" at quad/audio = 20×; works at 4× (25000/100000)**. Scene generation must use audio=25_000, quad=100_000 and resample to the scene rate.
  - `gr_filter.freq_xlating_fir_filter_ccf(decim, taps, center_freq, samp_rate)` with `taps = firdes.low_pass(gain, sampling_freq, cutoff_freq, transition_width)`
  - `gr_filter.rational_resampler_fff(interpolation, decimation)` and `rational_resampler_ccc` — **direct attributes of `gnuradio.filter`; there is NO `gnuradio.filter.rational_resampler` submodule in 3.10.12**
- `tb = gr.top_block(); tb.connect(src, hd, snk); tb.run()` works; a +100 kHz tone landed in the expected FFT bin.
- mypy: add to `mypy.ini` if it complains: `[mypy-gnuradio.*]` / `ignore_missing_imports = True`.
- Pre-commit (black/isort/flake8/mypy) runs on commit; if reformatted, `git add -u` and retry.

## File structure

```
src/marconi/
  models.py            # MODIFY: + BlockSpec, ConnectionSpec, PipelineSpec, SceneElement,
                       #          SceneSpec, DeviceInfo, ValidationIssue, RunResult
  sigmf.py             # MODIFY: + read_samples, read_meta, write_meta_for (Plan-1 debt)
  specs.py             # NEW: YAML save/load for PipelineSpec & SceneSpec
  vocabulary.py        # NEW: curated BlockDefs + validate_pipeline + PipelineValidationError
  config.py            # NEW: CONFIRM_TX toggle
  devices.py           # NEW: SimulatedDevice + registry + list_devices
  backends/
    __init__.py        # NEW: get_backend registry (lazy import)
    base.py            # NEW: Backend ABC + BackendError
    gnuradio_backend.py# NEW: factories + build + run (single file, ~200 lines)
  ops/
    analyze.py         # MODIFY: use sigmf.read_samples
    render.py          # MODIFY: use sigmf.read_samples
    capture.py         # MODIFY: load_capture uses read_meta; + capture(device, ...)
    pipeline.py        # NEW: run/validate/save/load pipeline ops
    simulate.py        # NEW: scene_to_pipeline + render_scene
    transmit.py        # NEW: transmit_capture into simulated scenes
    export_grc.py      # NEW: PipelineSpec -> .grc YAML
  __init__.py          # MODIFY: export new API
tests/marconi/
  test_sigmf_meta.py, test_specs.py, test_vocabulary.py, test_backend_gr.py,
  test_pipeline_op.py, test_simulate.py, test_devices_capture.py,
  test_transmit.py, test_export_grc.py, test_api_v2.py
```

(Single-file GNU Radio backend: the factories table and the engine are cohesive and together fit in ~200 lines; split only if it outgrows that — report DONE_WITH_CONCERNS rather than restructuring.)

## Deliberate deviations from the spec (record, don't re-litigate)

- Backend interface starts minimal — `run_pipeline` + `enumerate_devices` (extracted from real need; `capture_to_file`/`render_scene_to_file`/`transmit` are core-level compositions of `run_pipeline` for the simulated world; hardware backends add methods in v1.1).
- `stop()`/background runs deferred to Plan 3 (bounded runs suffice for sim; MCP layer owns run handles).
- `audio_sink` excluded from the v1.0 vocabulary (headless tests; the demo writes a wav).
- Simulated devices live in core `devices.py` (they are scenes, not hardware); `backend.enumerate_devices()` is for hardware and returns `[]` in v1.0.

---

### Task 1: SigMF debt — read_samples / read_meta / write_meta_for

**Files:**
- Modify: `src/marconi/sigmf.py`
- Modify: `src/marconi/ops/analyze.py` (replace `_read_samples` body with delegation), `src/marconi/ops/render.py` (import from sigmf), `src/marconi/ops/capture.py` (sigmf branch uses read_meta)
- Test: `tests/marconi/test_sigmf_meta.py`

- [ ] **Step 1: Write the failing test**

`tests/marconi/test_sigmf_meta.py`:
```python
from pathlib import Path

import numpy as np

from marconi.sigmf import read_meta, read_samples, write_capture, write_meta_for


def test_read_meta_does_not_read_samples(tmp_path: Path, make_iq) -> None:
    samples = make_iq([])
    ref = write_capture(samples, tmp_path / "cap", center_freq=1e9, sample_rate=2e6)
    meta = read_meta(ref.path)
    assert meta == ref  # num_samples derived from data file size


def test_read_samples_matches_written(tmp_path: Path, make_iq) -> None:
    samples = make_iq([(10e3, 1.0)])
    ref = write_capture(samples, tmp_path / "cap", center_freq=0.0, sample_rate=1e6)
    np.testing.assert_array_equal(read_samples(ref), samples)


def test_write_meta_for_existing_raw_file(tmp_path: Path, make_iq) -> None:
    samples = make_iq([])
    data_path = tmp_path / "x.sigmf-data"
    samples.tofile(data_path)
    ref = write_meta_for(data_path, center_freq=433e6, sample_rate=1e6)
    assert ref.path == data_path
    assert ref.num_samples == len(samples)
    assert read_meta(data_path) == ref
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/marconi/test_sigmf_meta.py -v`
Expected: FAIL with `ImportError: cannot import name 'read_meta'`

- [ ] **Step 3: Implement**

Append to `src/marconi/sigmf.py` (and refactor `read_capture` to use the pieces):
```python
def read_meta(path: Path | str) -> CaptureRef:
    """Read only the SigMF metadata; num_samples comes from the data file size."""
    base = _base(Path(path))
    data_path = base.with_name(base.name + ".sigmf-data")
    meta_path = base.with_name(base.name + ".sigmf-meta")

    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    datatype = meta["global"]["core:datatype"]
    if datatype != "cf32_le":
        raise ValueError(f"unsupported SigMF datatype: {datatype}")
    if not meta.get("captures"):
        raise ValueError("SigMF meta has no captures")

    return CaptureRef(
        path=data_path,
        center_freq=float(meta["captures"][0].get("core:frequency", 0.0)),
        sample_rate=float(meta["global"]["core:sample_rate"]),
        num_samples=data_path.stat().st_size // 8,  # complex64 = 8 bytes
    )


def read_samples(capture: CaptureRef) -> np.ndarray:
    """The single reader for capture sample data."""
    return np.fromfile(capture.path, dtype=np.complex64)


def write_meta_for(
    data_path: Path | str, center_freq: float, sample_rate: float
) -> CaptureRef:
    """Create the .sigmf-meta sidecar for an existing raw cf32 data file."""
    data_path = Path(data_path)
    ref = CaptureRef(
        path=data_path,
        center_freq=center_freq,
        sample_rate=sample_rate,
        num_samples=data_path.stat().st_size // 8,
    )
    meta_path = _base(data_path).with_name(_base(data_path).name + ".sigmf-meta")
    meta = {
        "global": {
            "core:datatype": "cf32_le",
            "core:sample_rate": sample_rate,
            "core:version": SIGMF_VERSION,
        },
        "captures": [{"core:sample_start": 0, "core:frequency": center_freq}],
        "annotations": [],
    }
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return ref
```

Refactor `read_capture` body to: `ref = read_meta(path); return np.fromfile(ref.path, dtype=np.complex64), ref` (keeps its exact behavior; note `read_meta` derives num_samples from file size which equals len(samples) for files we wrote). Extract the meta-dict construction shared by `write_capture`/`write_meta_for` into a private `_meta_dict(center_freq, sample_rate)` helper if black/flake8 are happy — DRY but small.

In `src/marconi/ops/analyze.py`: change `_read_samples` to
```python
from marconi.sigmf import read_samples as _read_samples  # noqa: F401
```
(keeping the `_read_samples` name so `render.py`'s import keeps working), OR update both `analyze.py` and `render.py` to import `read_samples` from `marconi.sigmf` directly and delete `_read_samples`. Prefer the second (cleaner); update `render.py` accordingly.

In `src/marconi/ops/capture.py`, the SigMF branch of `load_capture` becomes:
```python
    if name.endswith((".sigmf-data", ".sigmf-meta")):
        return sigmf.read_meta(path)
```

- [ ] **Step 4: Run the full suite to verify nothing broke**

Run: `uv run pytest tests/marconi -v`
Expected: all pass (38 existing + 3 new = 41)

- [ ] **Step 5: Commit**

```bash
git add src/marconi tests/marconi/test_sigmf_meta.py
git commit -m "Add lazy SigMF metadata path and shared sample reader"
```

---

### Task 2: Pipeline/Scene/Device/Run models + YAML IO

**Files:**
- Modify: `src/marconi/models.py` (append)
- Create: `src/marconi/specs.py`
- Test: `tests/marconi/test_specs.py`

- [ ] **Step 1: Write the failing test**

`tests/marconi/test_specs.py`:
```python
from pathlib import Path

from marconi.models import (
    BlockSpec,
    ConnectionSpec,
    DeviceInfo,
    PipelineSpec,
    RunResult,
    SceneElement,
    SceneSpec,
)
from marconi.specs import load_pipeline, load_scene, save_pipeline, save_scene


def _pipeline() -> PipelineSpec:
    return PipelineSpec(
        name="tone_to_file",
        sample_rate=1e6,
        blocks=[
            BlockSpec(id="src", type="tone_source", params={"freq": 100e3}),
            BlockSpec(id="hd", type="head", params={"num_samples": 50000}),
            BlockSpec(id="snk", type="file_sink", params={"path": "out.cf32"}),
        ],
        connections=[
            ConnectionSpec(src_block="src", dst_block="hd"),
            ConnectionSpec(src_block="hd", dst_block="snk"),
        ],
    )


def test_pipeline_yaml_roundtrip(tmp_path: Path) -> None:
    spec = _pipeline()
    path = save_pipeline(spec, tmp_path / "p.yaml")
    assert path.exists()
    assert load_pipeline(path) == spec


def test_scene_yaml_roundtrip(tmp_path: Path) -> None:
    scene = SceneSpec(
        name="three_signals",
        elements=[
            SceneElement(kind="tone", freq=100.1e6, amplitude=0.5),
            SceneElement(kind="noise", amplitude=0.01),
            SceneElement(kind="fm_tone", freq=100.3e6, params={"mod_freq": 1e3}),
        ],
    )
    path = save_scene(scene, tmp_path / "s.yaml")
    assert load_scene(path) == scene


def test_run_result_and_device_info_construct() -> None:
    r = RunResult(status="ok", elapsed_seconds=1.5, artifacts=[Path("a.cf32")])
    assert r.status == "ok" and r.error is None
    d = DeviceInfo(id="sim0", kind="simulated", can_tx=True)
    assert d.can_tx


def test_connection_defaults_ports_to_zero() -> None:
    c = ConnectionSpec(src_block="a", dst_block="b")
    assert c.src_port == 0 and c.dst_port == 0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/marconi/test_specs.py -v`
Expected: FAIL with `ImportError: cannot import name 'BlockSpec'`

- [ ] **Step 3: Implement models** (append to `src/marconi/models.py`)

```python
class BlockSpec(BaseModel):
    """One block instance in a pipeline; type names come from marconi.vocabulary."""

    id: str
    type: str
    params: dict[str, float | int | str | bool] = {}


class ConnectionSpec(BaseModel):
    """Directed edge between block ports (port indices default to 0)."""

    src_block: str
    src_port: int = 0
    dst_block: str
    dst_port: int = 0


class PipelineSpec(BaseModel):
    """Engine-agnostic DSP graph. sample_rate in Hz is the default rate for
    rate-dependent blocks that don't set their own."""

    name: str = "pipeline"
    sample_rate: float
    blocks: list[BlockSpec]
    connections: list[ConnectionSpec]


class SceneElement(BaseModel):
    """One emitter in a simulated RF environment. freq is the absolute Hz
    position (ignored for kind='noise'); extra knobs go in params."""

    kind: str  # tone | noise | fm_tone | iq_file
    freq: float = 0.0
    amplitude: float = 1.0
    params: dict[str, float | int | str] = {}


class SceneSpec(BaseModel):
    """What is 'on the air' for a simulated device."""

    name: str = "scene"
    elements: list[SceneElement] = []


class DeviceInfo(BaseModel):
    id: str
    kind: str  # "simulated" (hardware kinds arrive in v1.1)
    can_tx: bool = False
    description: str = ""


class ValidationIssue(BaseModel):
    """One actionable pipeline validation error, addressed to the agent."""

    block_id: str | None = None
    field: str | None = None
    message: str


class RunResult(BaseModel):
    """Outcome of a pipeline run. status: ok | timeout | error."""

    status: str
    elapsed_seconds: float
    artifacts: list[Path] = []
    error: str | None = None
```

- [ ] **Step 4: Implement YAML IO**

`src/marconi/specs.py`:
```python
from pathlib import Path

import yaml

from marconi.models import PipelineSpec, SceneSpec


def save_pipeline(spec: PipelineSpec, path: Path | str) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        yaml.safe_dump(spec.model_dump(mode="json"), sort_keys=False),
        encoding="utf-8",
    )
    return path


def load_pipeline(path: Path | str) -> PipelineSpec:
    return PipelineSpec.model_validate(
        yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    )


def save_scene(scene: SceneSpec, path: Path | str) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        yaml.safe_dump(scene.model_dump(mode="json"), sort_keys=False),
        encoding="utf-8",
    )
    return path


def load_scene(path: Path | str) -> SceneSpec:
    return SceneSpec.model_validate(
        yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    )
```

- [ ] **Step 5: Run tests**

Run: `uv run pytest tests/marconi/test_specs.py -v`
Expected: 4 passed

- [ ] **Step 6: Commit**

```bash
git add src/marconi/models.py src/marconi/specs.py tests/marconi/test_specs.py
git commit -m "Add pipeline/scene/device/run models with YAML IO"
```

---

### Task 3: Vocabulary + validate_pipeline

**Files:**
- Create: `src/marconi/vocabulary.py`
- Test: `tests/marconi/test_vocabulary.py`

- [ ] **Step 1: Write the failing test**

`tests/marconi/test_vocabulary.py`:
```python
import pytest

from marconi.models import BlockSpec, ConnectionSpec, PipelineSpec
from marconi.vocabulary import (
    VOCABULARY,
    PipelineValidationError,
    validate_pipeline,
)


def _valid() -> PipelineSpec:
    return PipelineSpec(
        sample_rate=1e6,
        blocks=[
            BlockSpec(id="src", type="tone_source", params={"freq": 100e3}),
            BlockSpec(id="hd", type="head", params={"num_samples": 1000}),
            BlockSpec(id="snk", type="file_sink", params={"path": "o.cf32"}),
        ],
        connections=[
            ConnectionSpec(src_block="src", dst_block="hd"),
            ConnectionSpec(src_block="hd", dst_block="snk"),
        ],
    )


def test_valid_pipeline_has_no_issues() -> None:
    assert validate_pipeline(_valid()) == []


def test_unknown_block_type() -> None:
    p = _valid()
    p.blocks[0] = BlockSpec(id="src", type="warp_drive", params={})
    issues = validate_pipeline(p)
    assert any("warp_drive" in i.message and i.block_id == "src" for i in issues)


def test_missing_required_param() -> None:
    p = _valid()
    p.blocks[0] = BlockSpec(id="src", type="tone_source", params={})
    issues = validate_pipeline(p)
    assert any(i.field == "freq" and i.block_id == "src" for i in issues)


def test_unknown_param_rejected() -> None:
    p = _valid()
    p.blocks[0].params["warp"] = 9
    issues = validate_pipeline(p)
    assert any(i.field == "warp" for i in issues)


def test_dangling_connection_endpoint() -> None:
    p = _valid()
    p.connections.append(ConnectionSpec(src_block="ghost", dst_block="snk"))
    issues = validate_pipeline(p)
    assert any("ghost" in i.message for i in issues)


def test_dtype_mismatch() -> None:
    # quadrature_demod outputs float; file_sink expects complex
    p = PipelineSpec(
        sample_rate=1e6,
        blocks=[
            BlockSpec(id="src", type="tone_source", params={"freq": 1e3}),
            BlockSpec(id="qd", type="quadrature_demod", params={}),
            BlockSpec(id="snk", type="file_sink", params={"path": "o.cf32"}),
        ],
        connections=[
            ConnectionSpec(src_block="src", dst_block="qd"),
            ConnectionSpec(src_block="qd", dst_block="snk"),
        ],
    )
    issues = validate_pipeline(p)
    assert any("complex" in i.message and "float" in i.message for i in issues)


def test_unconnected_input_port() -> None:
    p = _valid()
    p.connections = p.connections[1:]  # head's input now dangles
    issues = validate_pipeline(p)
    assert any(i.block_id == "hd" for i in issues)


def test_duplicate_block_id() -> None:
    p = _valid()
    p.blocks.append(BlockSpec(id="src", type="tone_source", params={"freq": 1.0}))
    issues = validate_pipeline(p)
    assert any("duplicate" in i.message.lower() for i in issues)


def test_validation_error_formats_issues() -> None:
    p = _valid()
    p.blocks[0] = BlockSpec(id="src", type="tone_source", params={})
    err = PipelineValidationError(validate_pipeline(p))
    assert "src" in str(err) and "freq" in str(err)


def test_vocabulary_covers_spec_minimum() -> None:
    for t in (
        "tone_source", "audio_tone_source", "noise_source", "file_source",
        "head", "add", "multiply_const", "freq_shift", "freq_xlating_lowpass",
        "quadrature_demod", "rational_resampler_f", "rational_resampler_c",
        "fm_deemphasis", "nbfm_rx", "nbfm_tx", "file_sink", "wav_sink",
    ):
        assert t in VOCABULARY, t
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/marconi/test_vocabulary.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'marconi.vocabulary'`

- [ ] **Step 3: Implement**

`src/marconi/vocabulary.py`:
```python
"""The curated block vocabulary: compositional primitives plus a few named
compositions (nbfm_rx/nbfm_tx). Dtypes: "c" = complex64 stream, "f" = float32.

Rate-dependent blocks accept an optional `sample_rate` param that defaults to
the pipeline's sample_rate at build time.
"""

from dataclasses import dataclass, field

from marconi.models import PipelineSpec, ValidationIssue


@dataclass(frozen=True)
class Param:
    name: str
    type: type
    required: bool = False
    default: object = None


@dataclass(frozen=True)
class BlockDef:
    inputs: tuple[str, ...]
    outputs: tuple[str, ...]
    params: tuple[Param, ...] = field(default_factory=tuple)


_RATE = Param("sample_rate", float)

VOCABULARY: dict[str, BlockDef] = {
    "tone_source": BlockDef(
        (), ("c",),
        (Param("freq", float, required=True), Param("amplitude", float, default=1.0), _RATE),
    ),
    "audio_tone_source": BlockDef(
        (), ("f",),
        (Param("freq", float, required=True), Param("amplitude", float, default=0.5), _RATE),
    ),
    "noise_source": BlockDef(
        (), ("c",),
        (Param("amplitude", float, required=True), Param("seed", int, default=0)),
    ),
    "file_source": BlockDef(
        (), ("c",),
        (Param("path", str, required=True), Param("repeat", bool, default=False)),
    ),
    "head": BlockDef(("c",), ("c",), (Param("num_samples", int, required=True),)),
    "add": BlockDef(("c", "c"), ("c",)),
    "multiply_const": BlockDef(("c",), ("c",), (Param("value", float, required=True),)),
    "freq_shift": BlockDef(
        ("c",), ("c",), (Param("offset", float, required=True), _RATE)
    ),
    "freq_xlating_lowpass": BlockDef(
        ("c",), ("c",),
        (
            Param("decimation", int, required=True),
            Param("center_offset", float, required=True),
            Param("cutoff", float, required=True),
            Param("transition", float, required=True),
            _RATE,
        ),
    ),
    "quadrature_demod": BlockDef(("c",), ("f",), (Param("gain", float, default=1.0),)),
    "rational_resampler_f": BlockDef(
        ("f",), ("f",),
        (Param("interpolation", int, required=True), Param("decimation", int, required=True)),
    ),
    "rational_resampler_c": BlockDef(
        ("c",), ("c",),
        (Param("interpolation", int, required=True), Param("decimation", int, required=True)),
    ),
    "fm_deemphasis": BlockDef(("f",), ("f",), (Param("tau", float, default=75e-6), _RATE)),
    "nbfm_rx": BlockDef(
        ("c",), ("f",),
        (
            Param("audio_rate", int, required=True),
            Param("quad_rate", int, required=True),
            Param("tau", float, default=75e-6),
            Param("max_dev", float, default=5e3),
        ),
    ),
    "nbfm_tx": BlockDef(
        ("f",), ("c",),
        (
            Param("audio_rate", int, required=True),
            Param("quad_rate", int, required=True),
            Param("tau", float, default=75e-6),
            Param("max_dev", float, default=5e3),
        ),
    ),
    "file_sink": BlockDef(("c",), (), (Param("path", str, required=True),)),
    "wav_sink": BlockDef(
        ("f",), (),
        (Param("path", str, required=True), Param("sample_rate", int, required=True)),
    ),
}

_DTYPE_NAMES = {"c": "complex", "f": "float"}


class PipelineValidationError(Exception):
    """Raised when a pipeline fails validation; formats issues for the agent."""

    def __init__(self, issues: list[ValidationIssue]) -> None:
        self.issues = issues
        lines = []
        for i in issues:
            where = i.block_id or "<pipeline>"
            f = f".{i.field}" if i.field else ""
            lines.append(f"{where}{f}: {i.message}")
        super().__init__("pipeline validation failed:\n" + "\n".join(lines))


def _check_param_type(value: object, expected: type) -> bool:
    if expected is float:
        return isinstance(value, (int, float)) and not isinstance(value, bool)
    if expected is int:
        return isinstance(value, int) and not isinstance(value, bool)
    return isinstance(value, expected)


def validate_pipeline(spec: PipelineSpec) -> list[ValidationIssue]:
    issues: list[ValidationIssue] = []
    by_id: dict[str, str] = {}

    for b in spec.blocks:
        if b.id in by_id:
            issues.append(
                ValidationIssue(block_id=b.id, message=f"duplicate block id '{b.id}'")
            )
            continue
        by_id[b.id] = b.type
        d = VOCABULARY.get(b.type)
        if d is None:
            known = ", ".join(sorted(VOCABULARY))
            issues.append(
                ValidationIssue(
                    block_id=b.id,
                    message=f"unknown block type '{b.type}'; known types: {known}",
                )
            )
            continue
        defs = {p.name: p for p in d.params}
        for name in b.params:
            if name not in defs:
                issues.append(
                    ValidationIssue(
                        block_id=b.id,
                        field=name,
                        message=f"unknown parameter for {b.type}; "
                        f"accepted: {sorted(defs) or 'none'}",
                    )
                )
        for p in d.params:
            if p.required and p.name not in b.params:
                issues.append(
                    ValidationIssue(
                        block_id=b.id,
                        field=p.name,
                        message=f"required parameter missing ({p.type.__name__})",
                    )
                )
            elif p.name in b.params and not _check_param_type(b.params[p.name], p.type):
                issues.append(
                    ValidationIssue(
                        block_id=b.id,
                        field=p.name,
                        message=f"expected {p.type.__name__}, "
                        f"got {type(b.params[p.name]).__name__}",
                    )
                )

    connected_inputs: set[tuple[str, int]] = set()
    for c in spec.connections:
        for end, port_attr in ((c.src_block, "src"), (c.dst_block, "dst")):
            if end not in by_id:
                issues.append(
                    ValidationIssue(
                        message=f"connection references unknown block '{end}'"
                    )
                )
        if c.src_block in by_id and by_id[c.src_block] in VOCABULARY:
            d = VOCABULARY[by_id[c.src_block]]
            if c.src_port >= len(d.outputs):
                issues.append(
                    ValidationIssue(
                        block_id=c.src_block,
                        message=f"output port {c.src_port} out of range "
                        f"({len(d.outputs)} outputs)",
                    )
                )
        if c.dst_block in by_id and by_id[c.dst_block] in VOCABULARY:
            d = VOCABULARY[by_id[c.dst_block]]
            if c.dst_port >= len(d.inputs):
                issues.append(
                    ValidationIssue(
                        block_id=c.dst_block,
                        message=f"input port {c.dst_port} out of range "
                        f"({len(d.inputs)} inputs)",
                    )
                )
            else:
                key = (c.dst_block, c.dst_port)
                if key in connected_inputs:
                    issues.append(
                        ValidationIssue(
                            block_id=c.dst_block,
                            message=f"input port {c.dst_port} connected twice",
                        )
                    )
                connected_inputs.add(key)
        if (
            c.src_block in by_id
            and c.dst_block in by_id
            and by_id[c.src_block] in VOCABULARY
            and by_id[c.dst_block] in VOCABULARY
        ):
            src_d = VOCABULARY[by_id[c.src_block]]
            dst_d = VOCABULARY[by_id[c.dst_block]]
            if c.src_port < len(src_d.outputs) and c.dst_port < len(dst_d.inputs):
                out_t = src_d.outputs[c.src_port]
                in_t = dst_d.inputs[c.dst_port]
                if out_t != in_t:
                    issues.append(
                        ValidationIssue(
                            block_id=c.dst_block,
                            message=f"dtype mismatch: {c.src_block} outputs "
                            f"{_DTYPE_NAMES[out_t]} but {c.dst_block} input "
                            f"{c.dst_port} expects {_DTYPE_NAMES[in_t]}",
                        )
                    )

    for b in spec.blocks:
        d = VOCABULARY.get(b.type)
        if d is None:
            continue
        for port in range(len(d.inputs)):
            if (b.id, port) not in connected_inputs:
                issues.append(
                    ValidationIssue(
                        block_id=b.id,
                        message=f"input port {port} is not connected",
                    )
                )

    return issues
```

- [ ] **Step 4: Run tests**

Run: `uv run pytest tests/marconi/test_vocabulary.py -v`
Expected: 10 passed

- [ ] **Step 5: Commit**

```bash
git add src/marconi/vocabulary.py tests/marconi/test_vocabulary.py
git commit -m "Add curated block vocabulary with structured pipeline validation"
```

---

### Task 4: Backend interface + registry

**Files:**
- Create: `src/marconi/backends/__init__.py`, `src/marconi/backends/base.py`
- Test: `tests/marconi/test_backend_gr.py` (registry tests only at this stage)

- [ ] **Step 1: Write the failing test**

`tests/marconi/test_backend_gr.py` (initial content; later tasks append):
```python
import pytest

from marconi.backends import get_backend
from marconi.backends.base import Backend


def test_unknown_backend_rejected() -> None:
    with pytest.raises(ValueError, match="unknown backend"):
        get_backend("imaginary")


def test_gnuradio_backend_resolves() -> None:
    b = get_backend("gnuradio")
    assert isinstance(b, Backend)
    assert b.name == "gnuradio"
    assert b.enumerate_devices() == []  # no hardware support in v1.0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/marconi/test_backend_gr.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'marconi.backends'`

- [ ] **Step 3: Implement**

`src/marconi/backends/base.py`:
```python
from abc import ABC, abstractmethod

from marconi.models import DeviceInfo, PipelineSpec, RunResult


class BackendError(Exception):
    """A failure inside the sample engine, translated for the agent."""


class Backend(ABC):
    """The swap boundary: everything that moves samples or touches devices."""

    name: str

    @abstractmethod
    def run_pipeline(self, spec: PipelineSpec, timeout: float = 30.0) -> RunResult:
        """Run a validated pipeline to completion (bounded by head blocks),
        supervised by `timeout` seconds."""

    @abstractmethod
    def enumerate_devices(self) -> list[DeviceInfo]:
        """Hardware devices this backend can drive (none in v1.0)."""
```

`src/marconi/backends/__init__.py`:
```python
from marconi.backends.base import Backend, BackendError

__all__ = ["Backend", "BackendError", "get_backend"]


def get_backend(name: str = "gnuradio") -> Backend:
    """Resolve a backend by name; imports lazily so `import marconi` never
    pulls in an engine."""
    if name == "gnuradio":
        from marconi.backends.gnuradio_backend import GnuRadioBackend

        return GnuRadioBackend()
    raise ValueError(f"unknown backend '{name}'; available: gnuradio")
```

For this task, `gnuradio_backend.py` is a stub so the registry test passes:
```python
from marconi.backends.base import Backend
from marconi.models import DeviceInfo, PipelineSpec, RunResult


class GnuRadioBackend(Backend):
    """GNU Radio sample engine (build/run implemented in the next tasks)."""

    name = "gnuradio"

    def run_pipeline(self, spec: PipelineSpec, timeout: float = 30.0) -> RunResult:
        raise NotImplementedError  # Task 5/6

    def enumerate_devices(self) -> list[DeviceInfo]:
        return []
```

- [ ] **Step 4: Run tests**

Run: `uv run pytest tests/marconi/test_backend_gr.py -v`
Expected: 2 passed

- [ ] **Step 5: Commit**

```bash
git add src/marconi/backends tests/marconi/test_backend_gr.py
git commit -m "Add backend interface with lazy gnuradio registry"
```

---

### Task 5: GNU Radio backend — block factories and graph build

**Files:**
- Modify: `src/marconi/backends/gnuradio_backend.py` (replace stub internals)
- Test: append to `tests/marconi/test_backend_gr.py`

- [ ] **Step 1: Write the failing test** (append)

```python
from marconi.models import BlockSpec, ConnectionSpec, PipelineSpec


def _tone_pipeline(out_path: str, n: int = 50000) -> PipelineSpec:
    return PipelineSpec(
        name="tone_to_file",
        sample_rate=1e6,
        blocks=[
            BlockSpec(id="src", type="tone_source", params={"freq": 100e3}),
            BlockSpec(id="hd", type="head", params={"num_samples": n}),
            BlockSpec(id="snk", type="file_sink", params={"path": out_path}),
        ],
        connections=[
            ConnectionSpec(src_block="src", dst_block="hd"),
            ConnectionSpec(src_block="hd", dst_block="snk"),
        ],
    )


def test_build_top_block(tmp_path) -> None:
    from marconi.backends.gnuradio_backend import build_top_block

    spec = _tone_pipeline(str(tmp_path / "o.cf32"))
    tb, artifacts = build_top_block(spec)
    assert artifacts == [tmp_path / "o.cf32"]
    assert hasattr(tb, "run")  # it is a gr.top_block


def test_build_error_carries_block_id(tmp_path) -> None:
    from marconi.backends.base import BackendError
    from marconi.backends.gnuradio_backend import build_top_block

    spec = _tone_pipeline(str(tmp_path / "o.cf32"))
    # nbfm_tx with the verified-broken 20x ratio -> GR raises at construction
    spec.blocks.append(
        BlockSpec(
            id="bad_tx",
            type="nbfm_tx",
            params={"audio_rate": 50000, "quad_rate": 1000000},
        )
    )
    spec.blocks.append(
        BlockSpec(id="audio", type="audio_tone_source", params={"freq": 1e3})
    )
    spec.connections.append(ConnectionSpec(src_block="audio", dst_block="bad_tx"))
    import pytest as _pytest

    with _pytest.raises(BackendError, match="bad_tx"):
        build_top_block(spec)
```

- [ ] **Step 2: Run to verify failure**

Run: `uv run pytest tests/marconi/test_backend_gr.py -v`
Expected: 2 pass (registry), 2 FAIL with `ImportError: cannot import name 'build_top_block'`

- [ ] **Step 3: Implement**

Replace `src/marconi/backends/gnuradio_backend.py` with:
```python
"""GNU Radio sample engine. The only module that imports gnuradio.

All GNU Radio imports happen inside functions so that `import marconi`
works on machines without GNU Radio.
"""

import math
from collections.abc import Callable
from pathlib import Path
from typing import Any

from marconi.backends.base import Backend, BackendError
from marconi.models import DeviceInfo, PipelineSpec, RunResult

_SINK_TYPES = {"file_sink", "wav_sink"}


def _modules() -> Any:
    try:
        from gnuradio import analog, blocks, gr
        from gnuradio import filter as gr_filter
        from gnuradio.filter import firdes
    except ImportError as e:  # pragma: no cover
        raise BackendError(
            "GNU Radio is not importable. Install GNU Radio 3.10+ system-wide "
            "(https://wiki.gnuradio.org/index.php/InstallingGR)."
        ) from e
    return gr, blocks, analog, gr_filter, firdes


def _factories(rate: float) -> dict[str, Callable[[dict[str, Any]], Any]]:
    gr, blocks, analog, gr_filter, firdes = _modules()

    def r(p: dict[str, Any]) -> float:
        return float(p.get("sample_rate", rate))

    return {
        "tone_source": lambda p: analog.sig_source_c(
            r(p), analog.GR_COS_WAVE, p["freq"], p.get("amplitude", 1.0), 0
        ),
        "audio_tone_source": lambda p: analog.sig_source_f(
            r(p), analog.GR_COS_WAVE, p["freq"], p.get("amplitude", 0.5), 0
        ),
        "noise_source": lambda p: analog.noise_source_c(
            analog.GR_GAUSSIAN, p["amplitude"], p.get("seed", 0)
        ),
        "file_source": lambda p: blocks.file_source(
            gr.sizeof_gr_complex, p["path"], p.get("repeat", False)
        ),
        "head": lambda p: blocks.head(gr.sizeof_gr_complex, int(p["num_samples"])),
        "add": lambda p: blocks.add_vcc(1),
        "multiply_const": lambda p: blocks.multiply_const_cc(p["value"]),
        "freq_shift": lambda p: blocks.rotator_cc(
            2.0 * math.pi * p["offset"] / r(p)
        ),
        "freq_xlating_lowpass": lambda p: gr_filter.freq_xlating_fir_filter_ccf(
            int(p["decimation"]),
            firdes.low_pass(1.0, r(p), p["cutoff"], p["transition"]),
            p["center_offset"],
            r(p),
        ),
        "quadrature_demod": lambda p: analog.quadrature_demod_cf(
            p.get("gain", 1.0)
        ),
        "rational_resampler_f": lambda p: gr_filter.rational_resampler_fff(
            int(p["interpolation"]), int(p["decimation"])
        ),
        "rational_resampler_c": lambda p: gr_filter.rational_resampler_ccc(
            int(p["interpolation"]), int(p["decimation"])
        ),
        "fm_deemphasis": lambda p: analog.fm_deemph(
            fs=r(p), tau=p.get("tau", 75e-6)
        ),
        "nbfm_rx": lambda p: analog.nbfm_rx(
            audio_rate=int(p["audio_rate"]),
            quad_rate=int(p["quad_rate"]),
            tau=p.get("tau", 75e-6),
            max_dev=p.get("max_dev", 5e3),
        ),
        "nbfm_tx": lambda p: analog.nbfm_tx(
            audio_rate=int(p["audio_rate"]),
            quad_rate=int(p["quad_rate"]),
            tau=p.get("tau", 75e-6),
            max_dev=p.get("max_dev", 5e3),
        ),
        "file_sink": lambda p: blocks.file_sink(
            gr.sizeof_gr_complex, p["path"], False
        ),
        "wav_sink": lambda p: blocks.wavfile_sink(
            p["path"],
            1,
            int(p["sample_rate"]),
            blocks.FORMAT_WAV,
            blocks.FORMAT_PCM_16,
            False,
        ),
    }


def build_top_block(spec: PipelineSpec) -> tuple[Any, list[Path]]:
    """Instantiate a validated PipelineSpec as a gr.top_block.

    Returns (top_block, artifact_paths). Raises BackendError with the
    offending block id on construction failure.
    """
    gr, *_ = _modules()
    factories = _factories(spec.sample_rate)

    tb = gr.top_block(spec.name)
    instances: dict[str, Any] = {}
    artifacts: list[Path] = []

    for b in spec.blocks:
        factory = factories.get(b.type)
        if factory is None:
            raise BackendError(
                f"block '{b.id}': type '{b.type}' has no GNU Radio factory"
            )
        try:
            instances[b.id] = factory(dict(b.params))
        except Exception as e:
            raise BackendError(
                f"block '{b.id}' ({b.type}) failed to construct: {e}"
            ) from e
        if b.type in _SINK_TYPES:
            artifacts.append(Path(str(b.params["path"])))

    for c in spec.connections:
        try:
            tb.connect(
                (instances[c.src_block], c.src_port),
                (instances[c.dst_block], c.dst_port),
            )
        except Exception as e:
            raise BackendError(
                f"connecting {c.src_block}:{c.src_port} -> "
                f"{c.dst_block}:{c.dst_port} failed: {e}"
            ) from e

    return tb, artifacts


class GnuRadioBackend(Backend):
    """GNU Radio sample engine."""

    name = "gnuradio"

    def run_pipeline(self, spec: PipelineSpec, timeout: float = 30.0) -> RunResult:
        raise NotImplementedError  # Task 6

    def enumerate_devices(self) -> list[DeviceInfo]:
        return []
```

- [ ] **Step 4: Run tests**

Run: `uv run pytest tests/marconi/test_backend_gr.py -v`
Expected: 4 passed

- [ ] **Step 5: Commit**

```bash
git add src/marconi/backends/gnuradio_backend.py tests/marconi/test_backend_gr.py
git commit -m "Build gr.top_block graphs from pipeline specs"
```

---

### Task 6: GNU Radio backend — supervised run

**Files:**
- Modify: `src/marconi/backends/gnuradio_backend.py` (implement `run_pipeline`)
- Test: append to `tests/marconi/test_backend_gr.py`

- [ ] **Step 1: Write the failing test** (append; also add `import numpy as np` at the top of the test file — it is first used here)

```python
def test_run_pipeline_produces_analyzable_capture(tmp_path) -> None:
    """The loop closes: backend-generated samples are found by Plan-1 analysis."""
    import marconi
    from marconi.backends import get_backend

    out = tmp_path / "tone.cf32"
    result = get_backend("gnuradio").run_pipeline(_tone_pipeline(str(out)))
    assert result.status == "ok"
    assert result.artifacts == [out]
    assert result.error is None
    assert result.elapsed_seconds < 30

    ws = marconi.Workspace(tmp_path / "project")
    ref = marconi.load_capture(out, ws, sample_rate=1e6, center_freq=433e6)
    signals = marconi.find_signals(ref)
    assert len(signals) == 1
    assert abs(signals[0].center_freq - 433.1e6) < 2e3


def test_run_pipeline_timeout(tmp_path) -> None:
    """A never-ending flowgraph is stopped by the watchdog."""
    raw = tmp_path / "loop.cf32"
    np.zeros(1024, dtype=np.complex64).tofile(raw)
    spec = PipelineSpec(
        name="endless",
        sample_rate=1e6,
        blocks=[
            BlockSpec(
                id="src",
                type="file_source",
                params={"path": str(raw), "repeat": True},
            ),
            BlockSpec(
                id="snk", type="file_sink", params={"path": str(tmp_path / "o.cf32")}
            ),
        ],
        connections=[ConnectionSpec(src_block="src", dst_block="snk")],
    )
    from marconi.backends import get_backend

    result = get_backend("gnuradio").run_pipeline(spec, timeout=1.0)
    assert result.status == "timeout"
    assert result.elapsed_seconds >= 1.0
```

- [ ] **Step 2: Run to verify failure**

Run: `uv run pytest tests/marconi/test_backend_gr.py -v`
Expected: previous 4 pass; 2 new FAIL with `NotImplementedError`

- [ ] **Step 3: Implement** — replace `run_pipeline` in `GnuRadioBackend`:

```python
import threading
import time
import traceback
```
(top of file with other imports), then:

```python
    def run_pipeline(self, spec: PipelineSpec, timeout: float = 30.0) -> RunResult:
        start = time.monotonic()
        try:
            tb, artifacts = build_top_block(spec)
        except BackendError as e:
            return RunResult(
                status="error",
                elapsed_seconds=time.monotonic() - start,
                error=str(e),
            )

        failure: list[str] = []

        def _run() -> None:
            try:
                tb.run()
            except Exception:
                failure.append(traceback.format_exc())

        worker = threading.Thread(target=_run, daemon=True)
        worker.start()
        worker.join(timeout)

        timed_out = worker.is_alive()
        if timed_out:
            tb.stop()
            tb.wait()
            worker.join(5.0)

        elapsed = time.monotonic() - start
        if failure:
            return RunResult(
                status="error",
                elapsed_seconds=elapsed,
                artifacts=artifacts,
                error=f"flowgraph raised during run:\n{failure[0]}",
            )
        return RunResult(
            status="timeout" if timed_out else "ok",
            elapsed_seconds=elapsed,
            artifacts=artifacts,
            error="run exceeded timeout and was stopped" if timed_out else None,
        )
```

- [ ] **Step 4: Run tests**

Run: `uv run pytest tests/marconi/test_backend_gr.py -v`
Expected: 6 passed

- [ ] **Step 5: Commit**

```bash
git add src/marconi/backends/gnuradio_backend.py tests/marconi/test_backend_gr.py
git commit -m "Run pipelines with timeout watchdog and structured results"
```

---

### Task 7: pipeline ops (validate-then-run, save/load)

**Files:**
- Create: `src/marconi/ops/pipeline.py`
- Test: `tests/marconi/test_pipeline_op.py`

- [ ] **Step 1: Write the failing test**

`tests/marconi/test_pipeline_op.py`:
```python
from pathlib import Path

import pytest

from marconi.models import BlockSpec, ConnectionSpec, PipelineSpec
from marconi.ops.pipeline import run_pipeline, save_pipeline_to_workspace
from marconi.vocabulary import PipelineValidationError
from marconi.workspace import Workspace


def _spec(out: str) -> PipelineSpec:
    return PipelineSpec(
        name="tone",
        sample_rate=1e6,
        blocks=[
            BlockSpec(id="src", type="tone_source", params={"freq": 50e3}),
            BlockSpec(id="hd", type="head", params={"num_samples": 10000}),
            BlockSpec(id="snk", type="file_sink", params={"path": out}),
        ],
        connections=[
            ConnectionSpec(src_block="src", dst_block="hd"),
            ConnectionSpec(src_block="hd", dst_block="snk"),
        ],
    )


def test_run_validates_first(tmp_path: Path) -> None:
    bad = _spec(str(tmp_path / "o.cf32"))
    bad.blocks[0] = BlockSpec(id="src", type="tone_source", params={})
    with pytest.raises(PipelineValidationError, match="freq"):
        run_pipeline(bad)


def test_run_executes_valid_pipeline(tmp_path: Path) -> None:
    result = run_pipeline(_spec(str(tmp_path / "o.cf32")))
    assert result.status == "ok"
    assert (tmp_path / "o.cf32").exists()


def test_save_pipeline_to_workspace(tmp_path: Path) -> None:
    ws = Workspace(tmp_path)
    p = save_pipeline_to_workspace(_spec("o.cf32"), ws)
    assert p.parent == ws.root / "pipelines"
    assert p.suffix == ".yaml"
```

- [ ] **Step 2: Run to verify failure**

Run: `uv run pytest tests/marconi/test_pipeline_op.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'marconi.ops.pipeline'`

- [ ] **Step 3: Implement**

`src/marconi/ops/pipeline.py`:
```python
from pathlib import Path

from marconi.backends import get_backend
from marconi.models import PipelineSpec, RunResult
from marconi.specs import save_pipeline
from marconi.vocabulary import PipelineValidationError, validate_pipeline
from marconi.workspace import Workspace


def run_pipeline(
    spec: PipelineSpec, timeout: float = 30.0, backend: str = "gnuradio"
) -> RunResult:
    """Validate, then execute. Validation problems raise
    PipelineValidationError with per-block, per-field messages."""
    issues = validate_pipeline(spec)
    if issues:
        raise PipelineValidationError(issues)
    return get_backend(backend).run_pipeline(spec, timeout=timeout)


def save_pipeline_to_workspace(spec: PipelineSpec, workspace: Workspace) -> Path:
    return save_pipeline(spec, workspace.new_pipeline_path(spec.name))
```

- [ ] **Step 4: Run tests**

Run: `uv run pytest tests/marconi/test_pipeline_op.py -v`
Expected: 3 passed

- [ ] **Step 5: Commit**

```bash
git add src/marconi/ops/pipeline.py tests/marconi/test_pipeline_op.py
git commit -m "Add validate-then-run pipeline op"
```

---

### Task 8: simulate — scene_to_pipeline + render_scene

**Files:**
- Create: `src/marconi/ops/simulate.py`
- Test: `tests/marconi/test_simulate.py`

Scene-element semantics (engine-agnostic, encode exactly):
- `tone`: `tone_source(freq=el.freq - center, amplitude)`
- `noise`: `noise_source(amplitude, seed=el.params.get("seed", 0))` — band-wide
- `fm_tone`: NBFM-modulated audio tone. Fixed internal rates audio=25_000,
  quad=100_000 (the verified-working ratio). Chain: `audio_tone_source(freq=
  params["mod_freq"], sample_rate=25000)` → `nbfm_tx(25000, 100000)` →
  `rational_resampler_c(interpolation=int(sample_rate // 100_000), decimation=1)`
  → `freq_shift(offset=el.freq - center)` → adder. Amplitude applied via
  `multiply_const(value=el.amplitude)` after the resampler. Requires
  `sample_rate % 100_000 == 0` — else `ValueError`.
- `iq_file`: replayed capture. `file_source(path=params["path"])` →
  `multiply_const(el.amplitude)` → `freq_shift(offset=el.freq - center)` → adder.
  Requires `float(params["sample_rate"]) == sample_rate` — else `ValueError`
  ("transmit capture sample rate ... does not match ..." — used by Task 10).
- Elements (except noise) whose `|el.freq - center| > sample_rate * 0.45` are
  silently skipped (out of the rendered band).
- Adder chain: with k in-band element outputs, sum them pairwise with `add`
  blocks (k-1 adders); a single output connects straight to head. Zero in-band
  elements → `ValueError("scene has no elements within the rendered band")`.
- Tail: `head(num_samples=int(duration * sample_rate))` → `file_sink(path)`.

- [ ] **Step 1: Write the failing test**

`tests/marconi/test_simulate.py`:
```python
from pathlib import Path

import pytest

from marconi.models import SceneElement, SceneSpec
from marconi.ops.analyze import find_signals
from marconi.ops.simulate import render_scene, scene_to_pipeline
from marconi.vocabulary import validate_pipeline
from marconi.workspace import Workspace


def _scene() -> SceneSpec:
    return SceneSpec(
        name="two_tones",
        elements=[
            SceneElement(kind="tone", freq=100.1e6, amplitude=1.0),
            SceneElement(kind="tone", freq=99.8e6, amplitude=0.5),
            SceneElement(kind="noise", amplitude=0.01),
        ],
    )


def test_scene_to_pipeline_is_valid(tmp_path: Path) -> None:
    spec = scene_to_pipeline(
        _scene(), center_freq=100e6, sample_rate=1e6,
        duration=0.05, out_path=tmp_path / "s.cf32",
    )
    assert validate_pipeline(spec) == []
    types = [b.type for b in spec.blocks]
    assert types.count("tone_source") == 2
    assert types.count("noise_source") == 1
    assert types.count("add") == 2  # 3 sources -> 2 adders
    assert "head" in types and "file_sink" in types


def test_out_of_band_elements_skipped(tmp_path: Path) -> None:
    scene = _scene()
    scene.elements.append(SceneElement(kind="tone", freq=200e6))
    spec = scene_to_pipeline(
        scene, center_freq=100e6, sample_rate=1e6,
        duration=0.01, out_path=tmp_path / "s.cf32",
    )
    assert [b.type for b in spec.blocks].count("tone_source") == 2


def test_empty_band_rejected(tmp_path: Path) -> None:
    scene = SceneSpec(elements=[SceneElement(kind="tone", freq=200e6)])
    with pytest.raises(ValueError, match="no elements"):
        scene_to_pipeline(
            scene, center_freq=100e6, sample_rate=1e6,
            duration=0.01, out_path=tmp_path / "s.cf32",
        )


def test_fm_requires_divisible_rate(tmp_path: Path) -> None:
    scene = SceneSpec(
        elements=[SceneElement(kind="fm_tone", freq=100e6, params={"mod_freq": 1e3})]
    )
    with pytest.raises(ValueError, match="multiple of"):
        scene_to_pipeline(
            scene, center_freq=100e6, sample_rate=1.5e5,
            duration=0.01, out_path=tmp_path / "s.cf32",
        )


def test_render_scene_produces_findable_signals(tmp_path: Path) -> None:
    ws = Workspace(tmp_path / "project")
    ref = render_scene(
        _scene(), center_freq=100e6, sample_rate=1e6, duration=0.1, workspace=ws
    )
    assert ref.path.exists()
    assert ref.center_freq == 100e6
    assert ref.num_samples == 100000
    signals = find_signals(ref)
    assert len(signals) == 2
    found = sorted(s.center_freq for s in signals)
    assert abs(found[0] - 99.8e6) < 2e3
    assert abs(found[1] - 100.1e6) < 2e3
```

- [ ] **Step 2: Run to verify failure**

Run: `uv run pytest tests/marconi/test_simulate.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'marconi.ops.simulate'`

- [ ] **Step 3: Implement**

`src/marconi/ops/simulate.py`:
```python
from pathlib import Path

from marconi import sigmf
from marconi.models import (
    BlockSpec,
    CaptureRef,
    ConnectionSpec,
    PipelineSpec,
    SceneSpec,
)
from marconi.ops.pipeline import run_pipeline
from marconi.workspace import Workspace

_FM_AUDIO_RATE = 25_000
_FM_QUAD_RATE = 100_000


def scene_to_pipeline(
    scene: SceneSpec,
    center_freq: float,
    sample_rate: float,
    duration: float,
    out_path: Path | str,
) -> PipelineSpec:
    """Generate the pipeline that renders `scene` as seen by a receiver at
    center_freq/sample_rate for `duration` seconds, writing raw cf32 to
    out_path. Out-of-band elements (beyond ±45% of the rate) are skipped."""
    blocks: list[BlockSpec] = []
    connections: list[ConnectionSpec] = []
    outputs: list[str] = []  # block ids whose output feeds the adder chain

    for i, el in enumerate(scene.elements):
        offset = el.freq - center_freq
        if el.kind != "noise" and abs(offset) > sample_rate * 0.45:
            continue

        if el.kind == "tone":
            bid = f"tone{i}"
            blocks.append(
                BlockSpec(
                    id=bid,
                    type="tone_source",
                    params={"freq": offset, "amplitude": el.amplitude},
                )
            )
            outputs.append(bid)

        elif el.kind == "noise":
            bid = f"noise{i}"
            blocks.append(
                BlockSpec(
                    id=bid,
                    type="noise_source",
                    params={
                        "amplitude": el.amplitude,
                        "seed": int(el.params.get("seed", 0)),
                    },
                )
            )
            outputs.append(bid)

        elif el.kind == "fm_tone":
            if sample_rate % _FM_QUAD_RATE != 0:
                raise ValueError(
                    f"fm_tone requires sample_rate to be a multiple of "
                    f"{_FM_QUAD_RATE}, got {sample_rate}"
                )
            interp = int(sample_rate // _FM_QUAD_RATE)
            blocks += [
                BlockSpec(
                    id=f"fmaudio{i}",
                    type="audio_tone_source",
                    params={
                        "freq": float(el.params["mod_freq"]),
                        "sample_rate": float(_FM_AUDIO_RATE),
                    },
                ),
                BlockSpec(
                    id=f"fmtx{i}",
                    type="nbfm_tx",
                    params={
                        "audio_rate": _FM_AUDIO_RATE,
                        "quad_rate": _FM_QUAD_RATE,
                    },
                ),
                BlockSpec(
                    id=f"fmrr{i}",
                    type="rational_resampler_c",
                    params={"interpolation": interp, "decimation": 1},
                ),
                BlockSpec(
                    id=f"fmamp{i}",
                    type="multiply_const",
                    params={"value": el.amplitude},
                ),
                BlockSpec(
                    id=f"fmshift{i}",
                    type="freq_shift",
                    params={"offset": offset},
                ),
            ]
            connections += [
                ConnectionSpec(src_block=f"fmaudio{i}", dst_block=f"fmtx{i}"),
                ConnectionSpec(src_block=f"fmtx{i}", dst_block=f"fmrr{i}"),
                ConnectionSpec(src_block=f"fmrr{i}", dst_block=f"fmamp{i}"),
                ConnectionSpec(src_block=f"fmamp{i}", dst_block=f"fmshift{i}"),
            ]
            outputs.append(f"fmshift{i}")

        elif el.kind == "iq_file":
            file_rate = float(el.params["sample_rate"])
            if file_rate != sample_rate:
                raise ValueError(
                    f"transmit capture sample rate {file_rate} does not match "
                    f"scene render rate {sample_rate} (v1.0 requires equal rates)"
                )
            blocks += [
                BlockSpec(
                    id=f"iq{i}",
                    type="file_source",
                    params={"path": str(el.params["path"])},
                ),
                BlockSpec(
                    id=f"iqamp{i}",
                    type="multiply_const",
                    params={"value": el.amplitude},
                ),
                BlockSpec(
                    id=f"iqshift{i}",
                    type="freq_shift",
                    params={"offset": offset},
                ),
            ]
            connections += [
                ConnectionSpec(src_block=f"iq{i}", dst_block=f"iqamp{i}"),
                ConnectionSpec(src_block=f"iqamp{i}", dst_block=f"iqshift{i}"),
            ]
            outputs.append(f"iqshift{i}")

        else:
            raise ValueError(f"unknown scene element kind '{el.kind}'")

    if not outputs:
        raise ValueError("scene has no elements within the rendered band")

    # Pairwise adder chain
    current = outputs[0]
    for j, other in enumerate(outputs[1:]):
        adder = f"sum{j}"
        blocks.append(BlockSpec(id=adder, type="add", params={}))
        connections += [
            ConnectionSpec(src_block=current, dst_block=adder, dst_port=0),
            ConnectionSpec(src_block=other, dst_block=adder, dst_port=1),
        ]
        current = adder

    blocks += [
        BlockSpec(
            id="head",
            type="head",
            params={"num_samples": int(duration * sample_rate)},
        ),
        BlockSpec(id="sink", type="file_sink", params={"path": str(out_path)}),
    ]
    connections += [
        ConnectionSpec(src_block=current, dst_block="head"),
        ConnectionSpec(src_block="head", dst_block="sink"),
    ]

    return PipelineSpec(
        name=f"render_{scene.name}",
        sample_rate=sample_rate,
        blocks=blocks,
        connections=connections,
    )


def render_scene(
    scene: SceneSpec,
    center_freq: float,
    sample_rate: float,
    duration: float,
    workspace: Workspace,
    name: str | None = None,
    timeout: float = 60.0,
) -> CaptureRef:
    """Render a scene to a SigMF capture in the workspace."""
    base = workspace.new_capture_path(name or scene.name)
    data_path = base.with_name(base.name + ".sigmf-data")
    spec = scene_to_pipeline(scene, center_freq, sample_rate, duration, data_path)
    result = run_pipeline(spec, timeout=timeout)
    if result.status != "ok":
        raise RuntimeError(f"scene render failed ({result.status}): {result.error}")
    return sigmf.write_meta_for(data_path, center_freq, sample_rate)
```

- [ ] **Step 4: Run tests**

Run: `uv run pytest tests/marconi/test_simulate.py -v`
Expected: 5 passed

- [ ] **Step 5: Commit**

```bash
git add src/marconi/ops/simulate.py tests/marconi/test_simulate.py
git commit -m "Render simulated scenes through generated pipelines"
```

---

### Task 9: devices registry + capture op

**Files:**
- Create: `src/marconi/devices.py`
- Modify: `src/marconi/ops/capture.py` (append `capture`)
- Test: `tests/marconi/test_devices_capture.py`

- [ ] **Step 1: Write the failing test**

`tests/marconi/test_devices_capture.py`:
```python
from pathlib import Path

import pytest

from marconi.devices import (
    SimulatedDevice,
    add_simulated_device,
    clear_devices,
    get_device,
    list_devices,
)
from marconi.models import SceneElement, SceneSpec
from marconi.ops.analyze import find_signals
from marconi.ops.capture import capture
from marconi.workspace import Workspace


@pytest.fixture(autouse=True)
def _fresh_registry():
    clear_devices()
    yield
    clear_devices()


def _scene() -> SceneSpec:
    return SceneSpec(
        name="one_tone",
        elements=[
            SceneElement(kind="tone", freq=433.1e6, amplitude=1.0),
            SceneElement(kind="noise", amplitude=0.01),
        ],
    )


def test_registry_lists_simulated_devices() -> None:
    dev = add_simulated_device("sim0", _scene())
    assert isinstance(dev, SimulatedDevice)
    infos = list_devices()
    assert [d.id for d in infos] == ["sim0"]
    assert infos[0].kind == "simulated"
    assert infos[0].can_tx is True
    assert get_device("sim0") is dev


def test_duplicate_device_id_rejected() -> None:
    add_simulated_device("sim0", _scene())
    with pytest.raises(ValueError, match="already exists"):
        add_simulated_device("sim0", _scene())


def test_unknown_device_rejected() -> None:
    with pytest.raises(KeyError, match="nope"):
        get_device("nope")


def test_capture_from_simulated_device(tmp_path: Path) -> None:
    add_simulated_device("sim0", _scene())
    ws = Workspace(tmp_path)
    ref = capture(
        "sim0", center_freq=433e6, sample_rate=1e6, duration=0.05, workspace=ws
    )
    assert ref.path.is_relative_to(ws.root / "captures")
    assert ref.center_freq == 433e6
    signals = find_signals(ref)
    assert len(signals) == 1
    assert abs(signals[0].center_freq - 433.1e6) < 2e3
```

- [ ] **Step 2: Run to verify failure**

Run: `uv run pytest tests/marconi/test_devices_capture.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'marconi.devices'`

- [ ] **Step 3: Implement**

`src/marconi/devices.py`:
```python
"""Device registry. v1.0 knows only simulated devices (scenes behind a
device-shaped interface); hardware arrives via backends in v1.1."""

from dataclasses import dataclass

from marconi.backends import get_backend
from marconi.models import DeviceInfo, SceneSpec


@dataclass
class SimulatedDevice:
    id: str
    scene: SceneSpec
    can_tx: bool = True

    def info(self) -> DeviceInfo:
        return DeviceInfo(
            id=self.id,
            kind="simulated",
            can_tx=self.can_tx,
            description=f"simulated device with scene '{self.scene.name}'",
        )


_REGISTRY: dict[str, SimulatedDevice] = {}


def add_simulated_device(device_id: str, scene: SceneSpec) -> SimulatedDevice:
    if device_id in _REGISTRY:
        raise ValueError(f"device '{device_id}' already exists")
    dev = SimulatedDevice(id=device_id, scene=scene)
    _REGISTRY[device_id] = dev
    return dev


def get_device(device_id: str) -> SimulatedDevice:
    if device_id not in _REGISTRY:
        known = ", ".join(sorted(_REGISTRY)) or "none registered"
        raise KeyError(f"unknown device '{device_id}' (known: {known})")
    return _REGISTRY[device_id]


def list_devices(backend: str = "gnuradio") -> list[DeviceInfo]:
    infos = [d.info() for d in _REGISTRY.values()]
    infos.extend(get_backend(backend).enumerate_devices())
    return infos


def clear_devices() -> None:
    """Test helper: empty the registry."""
    _REGISTRY.clear()
```

Append to `src/marconi/ops/capture.py`:
```python
def capture(
    device: "SimulatedDevice | str",
    center_freq: float,
    sample_rate: float,
    duration: float,
    workspace: Workspace,
    name: str | None = None,
) -> CaptureRef:
    """Capture IQ from a device. v1.0: simulated devices only — renders the
    device's scene as seen at center_freq/sample_rate."""
    from marconi.devices import SimulatedDevice, get_device
    from marconi.ops.simulate import render_scene

    dev = get_device(device) if isinstance(device, str) else device
    if not isinstance(dev, SimulatedDevice):
        raise TypeError(f"unsupported device type: {type(dev).__name__}")
    return render_scene(
        dev.scene,
        center_freq=center_freq,
        sample_rate=sample_rate,
        duration=duration,
        workspace=workspace,
        name=name or f"{dev.id}_capture",
    )
```
(The lazy imports avoid a circular import: devices → backends, ops.capture → devices.)

- [ ] **Step 4: Run tests**

Run: `uv run pytest tests/marconi/test_devices_capture.py -v`
Expected: 4 passed

- [ ] **Step 5: Commit**

```bash
git add src/marconi/devices.py src/marconi/ops/capture.py tests/marconi/test_devices_capture.py
git commit -m "Add simulated device registry and device capture op"
```

---

### Task 10: transmit op

**Files:**
- Create: `src/marconi/config.py`, `src/marconi/ops/transmit.py`
- Test: `tests/marconi/test_transmit.py`

- [ ] **Step 1: Write the failing test**

`tests/marconi/test_transmit.py`:
```python
from pathlib import Path

import pytest

from marconi.devices import add_simulated_device, clear_devices
from marconi.models import SceneElement, SceneSpec
from marconi.ops.analyze import find_signals
from marconi.ops.capture import capture
from marconi.ops.simulate import render_scene
from marconi.ops.transmit import TransmitNotConfirmedError, transmit_capture
from marconi.workspace import Workspace


@pytest.fixture(autouse=True)
def _fresh_registry():
    clear_devices()
    yield
    clear_devices()


def _make_tone_capture(ws: Workspace):
    scene = SceneSpec(
        name="tx_payload",
        elements=[SceneElement(kind="tone", freq=433e6, amplitude=1.0)],
    )
    return render_scene(
        scene, center_freq=433e6, sample_rate=1e6, duration=0.05, workspace=ws
    )


def test_transmit_requires_confirmation(tmp_path: Path) -> None:
    dev = add_simulated_device(
        "sim0", SceneSpec(elements=[SceneElement(kind="noise", amplitude=0.01)])
    )
    ws = Workspace(tmp_path)
    payload = _make_tone_capture(ws)
    with pytest.raises(TransmitNotConfirmedError):
        transmit_capture(dev, payload, freq=433.2e6)


def test_transmit_places_capture_in_scene(tmp_path: Path) -> None:
    dev = add_simulated_device(
        "sim0", SceneSpec(elements=[SceneElement(kind="noise", amplitude=0.01)])
    )
    ws = Workspace(tmp_path)
    payload = _make_tone_capture(ws)

    transmit_capture(dev, payload, freq=433.2e6, confirmed=True)
    assert any(e.kind == "iq_file" for e in dev.scene.elements)

    rx = capture(
        "sim0", center_freq=433e6, sample_rate=1e6, duration=0.05, workspace=ws
    )
    signals = find_signals(rx)
    # payload tone was at baseband DC of the tx capture; placed at 433.2 MHz
    assert any(abs(s.center_freq - 433.2e6) < 3e3 for s in signals)


def test_transmit_to_non_tx_device_rejected(tmp_path: Path) -> None:
    dev = add_simulated_device(
        "sim0", SceneSpec(elements=[SceneElement(kind="noise", amplitude=0.01)])
    )
    dev.can_tx = False
    ws = Workspace(tmp_path)
    payload = _make_tone_capture(ws)
    with pytest.raises(PermissionError, match="cannot transmit"):
        transmit_capture(dev, payload, freq=433.2e6, confirmed=True)
```

- [ ] **Step 2: Run to verify failure**

Run: `uv run pytest tests/marconi/test_transmit.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'marconi.ops.transmit'`

- [ ] **Step 3: Implement**

`src/marconi/config.py`:
```python
"""Runtime configuration. CONFIRM_TX guards transmissions against agent
mistakes (wrong frequency/device) — not licensing enforcement. Licensed
users may set it to False."""

CONFIRM_TX: bool = True
```

`src/marconi/ops/transmit.py`:
```python
from marconi import config
from marconi.devices import SimulatedDevice, get_device
from marconi.models import CaptureRef, SceneElement


class TransmitNotConfirmedError(Exception):
    """Transmission attempted without confirmation while CONFIRM_TX is on."""


def transmit_capture(
    device: SimulatedDevice | str,
    capture: CaptureRef,
    freq: float,
    amplitude: float = 1.0,
    confirmed: bool = False,
) -> SceneElement:
    """Replay a capture 'on the air'. v1.0: simulated devices only — the
    capture becomes an iq_file element of the device's scene, audible to
    subsequent captures (requires matching sample rates)."""
    dev = get_device(device) if isinstance(device, str) else device
    if config.CONFIRM_TX and not confirmed:
        raise TransmitNotConfirmedError(
            "transmission requires confirmed=True (or set "
            "marconi.config.CONFIRM_TX = False); about to transmit "
            f"{capture.path.name} at {freq/1e6:.4f} MHz on '{dev.id}'"
        )
    if not dev.can_tx:
        raise PermissionError(f"device '{dev.id}' cannot transmit")
    element = SceneElement(
        kind="iq_file",
        freq=freq,
        amplitude=amplitude,
        params={"path": str(capture.path), "sample_rate": capture.sample_rate},
    )
    dev.scene.elements.append(element)
    return element
```

- [ ] **Step 4: Run tests**

Run: `uv run pytest tests/marconi/test_transmit.py -v`
Expected: 3 passed

- [ ] **Step 5: Commit**

```bash
git add src/marconi/config.py src/marconi/ops/transmit.py tests/marconi/test_transmit.py
git commit -m "Add confirmed transmit into simulated scenes"
```

---

### Task 11: export_grc

**Files:**
- Create: `src/marconi/ops/export_grc.py`
- Test: `tests/marconi/test_export_grc.py`

`.grc` files are YAML with `options` (a pseudo-block), `blocks`, `connections`,
and `metadata`. The compile test is the source of truth: `grcc` must accept the
output. If a mapping detail fails, run
`/opt/homebrew/bin/grcc -o <tmpdir> <file>` by hand and iterate on its error
message. To see a known-good reference file, generate one from the POC code:
`uv run python -c "..."` using `gnuradio_mcp.middlewares.platform.PlatformMiddleware.save_flowgraph`
(see `tests/integration` for usage) — or inspect any `.grc` from
`/opt/homebrew/share/gnuradio/examples`.

GRC block id mapping (vocab → GRC id, parameter dict):
- `tone_source` → `analog_sig_source_x` {type: complex, samp_rate, waveform: cos, freq, amp, offset: 0, phase: 0, showports: false}
- `audio_tone_source` → `analog_sig_source_x` {type: float, ...}
- `noise_source` → `analog_noise_source_x` {type: complex, noise_type: gaussian, amp, seed}
- `file_source` → `blocks_file_source` {file, type: complex, repeat: 'True'/'False', vlen: 1, begin_tag: pmt.PMT_NIL, offset: 0, length: 0}
- `head` → `blocks_head` {type: complex, num_items, vlen: 1}
- `add` → `blocks_add_xx` {type: complex, num_inputs: 2, vlen: 1}
- `multiply_const` → `blocks_multiply_const_vxx` {type: complex, const, vlen: 1}
- `freq_shift` → `blocks_rotator_cc` {phase_inc: <2*pi*offset/rate as float>, tag_inc_update: 'False'}
- `freq_xlating_lowpass` → `freq_xlating_fir_filter_xxx` {type: ccf, decim, center_freq, samp_rate, taps: firdes.low_pass(1.0, <rate>, <cutoff>, <transition>)}
- `quadrature_demod` → `analog_quadrature_demod_cf` {gain}
- `rational_resampler_f` → `rational_resampler_xxx` {type: fff, interp, decim, taps: '[]', fbw: 0}
- `rational_resampler_c` → `rational_resampler_xxx` {type: ccc, interp, decim, taps: '[]', fbw: 0}
- `fm_deemphasis` → `analog_fm_deemph` {samp_rate, tau}
- `nbfm_rx` → `analog_nbfm_rx` {audio_rate, quad_rate, tau, max_dev}
- `nbfm_tx` → `analog_nbfm_tx` {audio_rate, quad_rate, tau, max_dev, fh: -1.0}
- `file_sink` → `blocks_file_sink` {file, type: complex, unbuffered: 'False', append: 'False'}
- `wav_sink` → `blocks_wavfile_sink` {file, nchan: 1, samp_rate, format: wav, subformat: pcm_16, append: 'False'}

All parameter values must be **strings** in the YAML (GRC evaluates them as
Python expressions). Every block needs `states: {bus_sink: false, bus_source:
false, bus_structure: null, coordinate: [<x>, <y>], rotation: 0, state:
enabled}` — lay blocks out on a grid (e.g. x = 200*column, y = 100*row).
Connections are `[src_id, '0', dst_id, '0']` (port indices as strings).

- [ ] **Step 1: Write the failing test**

`tests/marconi/test_export_grc.py`:
```python
import shutil
import subprocess
from pathlib import Path

import pytest
import yaml

from marconi.models import BlockSpec, ConnectionSpec, PipelineSpec
from marconi.ops.export_grc import export_grc

GRCC = shutil.which("grcc") or "/opt/homebrew/bin/grcc"


def _rx_pipeline(tmp_path: Path) -> PipelineSpec:
    return PipelineSpec(
        name="nbfm_receiver",
        sample_rate=2e6,
        blocks=[
            BlockSpec(
                id="src",
                type="file_source",
                params={"path": str(tmp_path / "in.cf32")},
            ),
            BlockSpec(
                id="chan",
                type="freq_xlating_lowpass",
                params={
                    "decimation": 20,
                    "center_offset": 300e3,
                    "cutoff": 8e3,
                    "transition": 4e3,
                },
            ),
            BlockSpec(
                id="demod",
                type="nbfm_rx",
                params={"audio_rate": 25000, "quad_rate": 100000},
            ),
            BlockSpec(
                id="audio",
                type="wav_sink",
                params={"path": str(tmp_path / "out.wav"), "sample_rate": 25000},
            ),
        ],
        connections=[
            ConnectionSpec(src_block="src", dst_block="chan"),
            ConnectionSpec(src_block="chan", dst_block="demod"),
            ConnectionSpec(src_block="demod", dst_block="audio"),
        ],
    )


def test_export_structure(tmp_path: Path) -> None:
    out = export_grc(_rx_pipeline(tmp_path), tmp_path / "rx.grc")
    doc = yaml.safe_load(out.read_text(encoding="utf-8"))
    assert doc["options"]["parameters"]["id"] == "nbfm_receiver"
    ids = [b["id"] for b in doc["blocks"]]
    assert "analog_nbfm_rx" in ids and "blocks_file_source" in ids
    assert ["src", "0", "chan", "0"] in [list(c) for c in doc["connections"]]


def test_exported_grc_compiles(tmp_path: Path) -> None:
    if not Path(GRCC).exists():
        pytest.skip("grcc not available")
    out = export_grc(_rx_pipeline(tmp_path), tmp_path / "rx.grc")
    proc = subprocess.run(
        [GRCC, "-o", str(tmp_path), str(out)],
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert proc.returncode == 0, f"grcc failed:\n{proc.stdout}\n{proc.stderr}"
    assert (tmp_path / "nbfm_receiver.py").exists()
```

- [ ] **Step 2: Run to verify failure**

Run: `uv run pytest tests/marconi/test_export_grc.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'marconi.ops.export_grc'`

- [ ] **Step 3: Implement**

`src/marconi/ops/export_grc.py`:
```python
"""Export a PipelineSpec as a GNU Radio Companion .grc file.

.grc is the format for handing work to humans: open in GRC, tweak, keep.
Parameter values are strings because GRC evaluates them as Python.
"""

import math
from pathlib import Path
from typing import Any

import yaml

from marconi.models import BlockSpec, PipelineSpec


def _s(v: Any) -> str:
    return str(v)


def _map_block(b: BlockSpec, rate: float) -> tuple[str, dict[str, str]]:
    p = b.params
    r = float(p.get("sample_rate", rate))
    if b.type == "tone_source":
        return "analog_sig_source_x", {
            "type": "complex", "samp_rate": _s(r), "waveform": "analog.GR_COS_WAVE",
            "freq": _s(p["freq"]), "amp": _s(p.get("amplitude", 1.0)),
            "offset": "0", "phase": "0", "showports": "False",
        }
    if b.type == "audio_tone_source":
        return "analog_sig_source_x", {
            "type": "float", "samp_rate": _s(r), "waveform": "analog.GR_COS_WAVE",
            "freq": _s(p["freq"]), "amp": _s(p.get("amplitude", 0.5)),
            "offset": "0", "phase": "0", "showports": "False",
        }
    if b.type == "noise_source":
        return "analog_noise_source_x", {
            "type": "complex", "noise_type": "analog.GR_GAUSSIAN",
            "amp": _s(p["amplitude"]), "seed": _s(p.get("seed", 0)),
        }
    if b.type == "file_source":
        return "blocks_file_source", {
            "file": str(p["path"]), "type": "complex",
            "repeat": _s(bool(p.get("repeat", False))), "vlen": "1",
            "begin_tag": "pmt.PMT_NIL", "offset": "0", "length": "0",
        }
    if b.type == "head":
        return "blocks_head", {
            "type": "complex", "num_items": _s(int(p["num_samples"])), "vlen": "1"
        }
    if b.type == "add":
        return "blocks_add_xx", {"type": "complex", "num_inputs": "2", "vlen": "1"}
    if b.type == "multiply_const":
        return "blocks_multiply_const_vxx", {
            "type": "complex", "const": _s(p["value"]), "vlen": "1"
        }
    if b.type == "freq_shift":
        return "blocks_rotator_cc", {
            "phase_inc": _s(2.0 * math.pi * float(p["offset"]) / r),
            "tag_inc_update": "False",
        }
    if b.type == "freq_xlating_lowpass":
        return "freq_xlating_fir_filter_xxx", {
            "type": "ccf",
            "decim": _s(int(p["decimation"])),
            "center_freq": _s(p["center_offset"]),
            "samp_rate": _s(r),
            "taps": f"firdes.low_pass(1.0, {r}, {p['cutoff']}, {p['transition']})",
        }
    if b.type == "quadrature_demod":
        return "analog_quadrature_demod_cf", {"gain": _s(p.get("gain", 1.0))}
    if b.type in ("rational_resampler_f", "rational_resampler_c"):
        return "rational_resampler_xxx", {
            "type": "fff" if b.type.endswith("_f") else "ccc",
            "interp": _s(int(p["interpolation"])),
            "decim": _s(int(p["decimation"])),
            "taps": "[]", "fbw": "0",
        }
    if b.type == "fm_deemphasis":
        return "analog_fm_deemph", {"samp_rate": _s(r), "tau": _s(p.get("tau", 75e-6))}
    if b.type == "nbfm_rx":
        return "analog_nbfm_rx", {
            "audio_rate": _s(int(p["audio_rate"])),
            "quad_rate": _s(int(p["quad_rate"])),
            "tau": _s(p.get("tau", 75e-6)),
            "max_dev": _s(p.get("max_dev", 5e3)),
        }
    if b.type == "nbfm_tx":
        return "analog_nbfm_tx", {
            "audio_rate": _s(int(p["audio_rate"])),
            "quad_rate": _s(int(p["quad_rate"])),
            "tau": _s(p.get("tau", 75e-6)),
            "max_dev": _s(p.get("max_dev", 5e3)),
            "fh": "-1.0",
        }
    if b.type == "file_sink":
        return "blocks_file_sink", {
            "file": str(p["path"]), "type": "complex",
            "unbuffered": "False", "append": "False",
        }
    if b.type == "wav_sink":
        return "blocks_wavfile_sink", {
            "file": str(p["path"]), "nchan": "1",
            "samp_rate": _s(int(p["sample_rate"])),
            "format": "wav", "subformat": "pcm_16", "append": "False",
        }
    raise ValueError(f"no .grc mapping for block type '{b.type}'")


def export_grc(spec: PipelineSpec, path: Path | str) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    blocks = []
    for i, b in enumerate(spec.blocks):
        grc_id, params = _map_block(b, spec.sample_rate)
        blocks.append(
            {
                "name": b.id,
                "id": grc_id,
                "parameters": params,
                "states": {
                    "bus_sink": False,
                    "bus_source": False,
                    "bus_structure": None,
                    "coordinate": [200 + 250 * (i % 4), 100 + 150 * (i // 4)],
                    "rotation": 0,
                    "state": "enabled",
                },
            }
        )

    doc = {
        "options": {
            "parameters": {
                "id": spec.name.replace(" ", "_").replace("-", "_"),
                "title": spec.name,
                "author": "marconi",
                "output_language": "python",
                "generate_options": "no_gui",
                "run_options": "run",
                "category": "[GRC Hier Blocks]",
            },
            "states": {
                "bus_sink": False,
                "bus_source": False,
                "bus_structure": None,
                "coordinate": [8, 8],
                "rotation": 0,
                "state": "enabled",
            },
        },
        "blocks": blocks,
        "connections": [
            [c.src_block, str(c.src_port), c.dst_block, str(c.dst_port)]
            for c in spec.connections
        ],
        "metadata": {"file_format": 1, "grc_version": "3.10.12.0"},
    }
    path.write_text(yaml.safe_dump(doc, sort_keys=False), encoding="utf-8")
    return path
```

- [ ] **Step 4: Run tests; iterate against grcc errors**

Run: `uv run pytest tests/marconi/test_export_grc.py -v`
Expected: 2 passed. If `test_exported_grc_compiles` fails, read grcc's stderr
carefully — it names the block and parameter it rejects. Compare against a
reference `.grc` from `/opt/homebrew/share/gnuradio/examples` (e.g.
`grep -rl nbfm /opt/homebrew/share/gnuradio/examples` or any metadata example).
Fix the mapping table, not the test. If a block simply cannot be mapped after
3-4 iterations, report DONE_WITH_CONCERNS naming the block.

- [ ] **Step 5: Commit**

```bash
git add src/marconi/ops/export_grc.py tests/marconi/test_export_grc.py
git commit -m "Export pipelines as GNU Radio Companion .grc files"
```

---

### Task 12: Public API + killer-demo e2e

**Files:**
- Modify: `src/marconi/__init__.py`
- Test: `tests/marconi/test_api_v2.py`

- [ ] **Step 1: Write the failing test**

`tests/marconi/test_api_v2.py`:
```python
"""The killer demo as a test: three unknown signals — find them, identify
the FM one, demodulate it, verify the audio."""

from pathlib import Path

import numpy as np
import pytest
from scipy.io import wavfile

import marconi


@pytest.fixture(autouse=True)
def _fresh_registry():
    marconi.clear_devices()
    yield
    marconi.clear_devices()


def test_public_api_surface() -> None:
    for name in (
        "PipelineSpec", "SceneSpec", "SceneElement", "BlockSpec",
        "ConnectionSpec", "RunResult", "DeviceInfo",
        "validate_pipeline", "run_pipeline", "save_pipeline", "load_pipeline",
        "save_scene", "load_scene", "scene_to_pipeline", "render_scene",
        "add_simulated_device", "list_devices", "get_device", "clear_devices",
        "capture", "transmit_capture", "export_grc",
    ):
        assert hasattr(marconi, name), name


def test_killer_demo_full_loop(tmp_path: Path) -> None:
    ws = marconi.Workspace(tmp_path)

    # 1. The world: three signals the agent doesn't know about
    scene = marconi.SceneSpec(
        name="three_signals",
        elements=[
            marconi.SceneElement(
                kind="fm_tone", freq=100.3e6, amplitude=1.0,
                params={"mod_freq": 1e3},
            ),
            marconi.SceneElement(kind="tone", freq=99.5e6, amplitude=0.4),
            marconi.SceneElement(kind="noise", amplitude=0.005),
        ],
    )
    marconi.add_simulated_device("sim0", scene)
    assert [d.id for d in marconi.list_devices()] == ["sim0"]

    # 2. Survey the band
    cap = marconi.capture(
        "sim0", center_freq=100e6, sample_rate=2e6, duration=0.25, workspace=ws
    )
    signals = marconi.find_signals(cap)
    assert len(signals) >= 2
    fm = max(signals, key=lambda s: s.bandwidth)  # FM is the wide one
    assert abs(fm.center_freq - 100.3e6) < 5e3

    # 3. Build, save, and run the receiver
    rx = marconi.PipelineSpec(
        name="fm_rx",
        sample_rate=2e6,
        blocks=[
            marconi.BlockSpec(
                id="src", type="file_source", params={"path": str(cap.path)}
            ),
            marconi.BlockSpec(
                id="chan",
                type="freq_xlating_lowpass",
                params={
                    "decimation": 20,
                    "center_offset": fm.center_freq - 100e6,
                    "cutoff": 8e3,
                    "transition": 4e3,
                },
            ),
            marconi.BlockSpec(
                id="demod",
                type="nbfm_rx",
                params={"audio_rate": 25000, "quad_rate": 100000},
            ),
            marconi.BlockSpec(
                id="audio",
                type="wav_sink",
                params={"path": str(tmp_path / "audio.wav"), "sample_rate": 25000},
            ),
        ],
        connections=[
            marconi.ConnectionSpec(src_block="src", dst_block="chan"),
            marconi.ConnectionSpec(src_block="chan", dst_block="demod"),
            marconi.ConnectionSpec(src_block="demod", dst_block="audio"),
        ],
    )
    assert marconi.validate_pipeline(rx) == []
    result = marconi.run_pipeline(rx)
    assert result.status == "ok"

    # 4. Verify: the demodulated audio is dominated by the 1 kHz tone
    rate, audio = wavfile.read(tmp_path / "audio.wav")
    assert rate == 25000
    audio = audio.astype(np.float32)
    audio = audio - audio.mean()
    n = len(audio)
    assert n > 1000
    spectrum = np.abs(np.fft.rfft(audio * np.hanning(n)))
    peak_freq = np.fft.rfftfreq(n, 1 / rate)[int(np.argmax(spectrum))]
    assert abs(peak_freq - 1e3) < 50

    # 5. The durable artifacts
    from marconi.ops.pipeline import save_pipeline_to_workspace

    saved = save_pipeline_to_workspace(rx, ws)
    grc = marconi.export_grc(rx, ws.root / "pipelines" / "fm_rx.grc")
    assert saved.exists() and grc.exists()
    assert (ws.root / "captures").is_dir()
```

- [ ] **Step 2: Run to verify failure**

Run: `uv run pytest tests/marconi/test_api_v2.py -v`
Expected: FAIL with `AssertionError: PipelineSpec` (missing exports)

- [ ] **Step 3: Implement** — add to `src/marconi/__init__.py` (extend existing imports and `__all__`):

```python
from marconi.devices import (
    SimulatedDevice,
    add_simulated_device,
    clear_devices,
    get_device,
    list_devices,
)
from marconi.models import (
    BlockSpec,
    ConnectionSpec,
    DeviceInfo,
    PipelineSpec,
    RunResult,
    SceneElement,
    SceneSpec,
    ValidationIssue,
)
from marconi.ops.capture import capture
from marconi.ops.export_grc import export_grc
from marconi.ops.pipeline import run_pipeline, save_pipeline_to_workspace
from marconi.ops.simulate import render_scene, scene_to_pipeline
from marconi.ops.transmit import TransmitNotConfirmedError, transmit_capture
from marconi.specs import load_pipeline, load_scene, save_pipeline, save_scene
from marconi.vocabulary import (
    VOCABULARY,
    PipelineValidationError,
    validate_pipeline,
)
```
and extend `__all__` with every name above (keep alphabetical order; black/isort will enforce formatting).

- [ ] **Step 4: Run the full suite**

Run: `uv run pytest tests/marconi -v`
Expected: all pass (Plan 1's 41 post-Task-1 + this plan's ≈34 = ~75; exact count printed)

Also: `uv run pytest tests/unit -q` — Expected: 40 passed (POC untouched).

- [ ] **Step 5: Commit**

```bash
git add src/marconi/__init__.py tests/marconi/test_api_v2.py
git commit -m "Export full v1.0 simulation API; killer demo as e2e test"
```

---

## Self-review checklist (run after all tasks)

- `uv run pytest tests/marconi tests/unit -q` — everything green.
- `grep -rn "import gnuradio\|from gnuradio" src/marconi/ | grep -v gnuradio_backend` → empty (engine isolation holds).
- `uv run python -c "import marconi"` works and does not import gnuradio (check `'gnuradio' not in sys.modules`).
- The killer-demo test passes and leaves a readable project tree under its tmp workspace.
