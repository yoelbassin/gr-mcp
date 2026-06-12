# Marconi v1.0 — Plan 1 of 3: Core Package (Captures, Analysis, Rendering)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the engine-agnostic heart of the `marconi` package: capture models, SigMF read/write, the workspace (user project directory), `load_capture`, and the full `analyze` + `render` operation families — fully usable on any IQ file with zero GNU Radio dependency.

**Architecture:** This is Layer 1 of the Marconi design (see `docs/superpowers/specs/2026-06-12-marconi-design.md`), minus the backend. Everything here is pure numpy/scipy/matplotlib operating on IQ files. The existing gr-mcp POC code (`src/gnuradio_mcp/`, `main.py`) stays untouched until Plan 3 retires it. Plan 2 adds vocabularies, the backend interface, the GNU Radio backend, pipelines, and simulation; Plan 3 adds the MCP server, plugin, and skills.

**Tech Stack:** Python ≥3.13, pydantic, numpy, scipy, matplotlib (Agg), pytest, uv.

---

## Plan roadmap (context for the executor)

| Plan | Scope | Deliverable |
|---|---|---|
| **1 (this)** | models, sigmf, workspace, load_capture, analyze (psd/find_signals/measure/detect_bursts), render (spectrogram/psd_plot/constellation) | `import marconi` → load/analyze/render any IQ file |
| 2 | vocabularies, backend interface, GNU Radio backend, pipeline ops, scenes + SimulatedDevice, capture/transmit ops | full ops surface working from Python |
| 3 | FastMCP server (~18 tools), Claude Code plugin, 5 skills, e2e tests, skill evals, POC retirement | shippable plugin |

## File structure

```
src/marconi/
  __init__.py        # public API re-exports (Task 11)
  models.py          # CaptureRef + analysis result models (Task 2)
  sigmf.py           # minimal SigMF write/read, no external dep (Task 3)
  workspace.py       # Workspace: project-dir conventions, path allocation (Task 4)
  ops/
    __init__.py
    capture.py       # load_capture (Task 5)
    analyze.py       # psd, find_signals, measure, detect_bursts (Tasks 6-9)
    render.py        # spectrogram, psd_plot, constellation (Task 10)
tests/marconi/
  __init__.py
  conftest.py        # synthetic IQ fixture factory
  test_models.py
  test_sigmf.py
  test_workspace.py
  test_load_capture.py
  test_analyze_psd.py
  test_analyze_signals.py
  test_analyze_measure.py
  test_analyze_bursts.py
  test_render.py
  test_api.py        # public-API smoke + mini end-to-end (Task 11)
```

Notes for every task:

- Run tests with `uv run pytest tests/marconi -v` (the repo's pytest config already puts `src` on `pythonpath`).
- Pre-commit hooks (black, isort, flake8, mypy, trailing-whitespace) run on commit. If a hook reformats files, the commit aborts — `git add -u` and commit again. If mypy complains about missing stubs for `scipy`/`matplotlib`, add to `mypy.ini`:
  ```ini
  [mypy-scipy.*]
  ignore_missing_imports = True
  [mypy-matplotlib.*]
  ignore_missing_imports = True
  ```
- The repo's top-level `tests/conftest.py` injects the system GNU Radio path for the POC tests; it is harmless to the new tests. Do not modify it.

---

### Task 1: Package skeleton and dependencies

**Files:**
- Create: `src/marconi/__init__.py`, `src/marconi/ops/__init__.py`, `tests/marconi/__init__.py`
- Modify: `pyproject.toml` (via `uv add`)

- [ ] **Step 1: Add dependencies**

Run: `uv add numpy scipy matplotlib`
Expected: pyproject.toml `dependencies` gains the three packages; `uv.lock` updated.

- [ ] **Step 2: Create the package skeleton**

`src/marconi/__init__.py`:
```python
"""Marconi: LLM-driven RF control and analysis."""

__version__ = "0.1.0"
```

`src/marconi/ops/__init__.py`:
```python
```
(empty file)

`tests/marconi/__init__.py`:
```python
```
(empty file)

- [ ] **Step 3: Verify the package imports**

Run: `uv run python -c "import marconi; print(marconi.__version__)"`
Expected: `0.1.0`

- [ ] **Step 4: Commit**

```bash
git add pyproject.toml uv.lock src/marconi tests/marconi
git commit -m "Add marconi package skeleton with numpy/scipy/matplotlib"
```

---

### Task 2: Capture and analysis result models

**Files:**
- Create: `src/marconi/models.py`
- Test: `tests/marconi/test_models.py`

- [ ] **Step 1: Write the failing test**

`tests/marconi/test_models.py`:
```python
from pathlib import Path

from marconi.models import Burst, CaptureRef, DetectedSignal, PSDResult, SignalPeak


def test_capture_ref_duration() -> None:
    ref = CaptureRef(
        path=Path("captures/x.sigmf-data"),
        center_freq=100e6,
        sample_rate=1e6,
        num_samples=2_000_000,
    )
    assert ref.duration == 2.0
    assert ref.datatype == "cf32_le"


def test_capture_ref_roundtrip() -> None:
    ref = CaptureRef(
        path=Path("captures/x.sigmf-data"),
        center_freq=100e6,
        sample_rate=1e6,
        num_samples=10,
    )
    again = CaptureRef.model_validate_json(ref.model_dump_json())
    assert again == ref


def test_result_models_construct() -> None:
    psd = PSDResult(
        freqs=[1.0, 2.0],
        psd_db=[-90.0, -40.0],
        noise_floor_db=-90.0,
        peaks=[SignalPeak(freq=2.0, power_db=-40.0)],
    )
    assert psd.peaks[0].freq == 2.0
    sig = DetectedSignal(
        center_freq=100e6, bandwidth=12e3, peak_power_db=-40.0, snr_db=50.0
    )
    assert sig.bandwidth == 12e3
    burst = Burst(start_time=0.01, duration=0.005, mean_power_db=-30.0)
    assert burst.duration == 0.005
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/marconi/test_models.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'marconi.models'`

- [ ] **Step 3: Write the models**

`src/marconi/models.py`:
```python
from pathlib import Path

from pydantic import BaseModel


class CaptureRef(BaseModel):
    """Reference to an IQ capture on disk (.sigmf-data + .sigmf-meta sidecar)."""

    path: Path
    center_freq: float
    sample_rate: float
    num_samples: int
    datatype: str = "cf32_le"

    @property
    def duration(self) -> float:
        return self.num_samples / self.sample_rate


class SignalPeak(BaseModel):
    freq: float
    power_db: float


class PSDResult(BaseModel):
    freqs: list[float]
    psd_db: list[float]
    noise_floor_db: float
    peaks: list[SignalPeak]


class DetectedSignal(BaseModel):
    center_freq: float
    bandwidth: float
    peak_power_db: float
    snr_db: float


class SignalMeasurement(BaseModel):
    center_freq: float
    occupied_bw_99: float
    power_db: float
    snr_db: float


class Burst(BaseModel):
    start_time: float
    duration: float
    mean_power_db: float


class RenderResult(BaseModel):
    path: Path
    kind: str
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/marconi/test_models.py -v`
Expected: 3 passed

- [ ] **Step 5: Commit**

```bash
git add src/marconi/models.py tests/marconi/test_models.py
git commit -m "Add marconi capture and analysis result models"
```

---

### Task 3: Minimal SigMF write/read

SigMF is two files: raw samples in `<name>.sigmf-data` and JSON metadata in `<name>.sigmf-meta`. We hand-roll it (the format is trivial JSON; no new dependency).

**Files:**
- Create: `src/marconi/sigmf.py`
- Test: `tests/marconi/test_sigmf.py`
- Create: `tests/marconi/conftest.py`

- [ ] **Step 1: Write the shared IQ fixture factory**

`tests/marconi/conftest.py`:
```python
from collections.abc import Callable

import numpy as np
import pytest

MakeIQ = Callable[..., np.ndarray]


@pytest.fixture
def make_iq() -> MakeIQ:
    """Build complex64 baseband: noise plus tones at given (offset_hz, amplitude)."""

    def _make(
        tones: list[tuple[float, float]],
        sample_rate: float = 1e6,
        duration: float = 0.05,
        noise_amplitude: float = 0.01,
        seed: int = 0,
    ) -> np.ndarray:
        rng = np.random.default_rng(seed)
        n = int(sample_rate * duration)
        t = np.arange(n) / sample_rate
        x = rng.normal(0, noise_amplitude, n) + 1j * rng.normal(
            0, noise_amplitude, n
        )
        for freq, amp in tones:
            x = x + amp * np.exp(2j * np.pi * freq * t)
        return x.astype(np.complex64)

    return _make
```

- [ ] **Step 2: Write the failing test**

`tests/marconi/test_sigmf.py`:
```python
import json
from pathlib import Path

import numpy as np

from marconi.sigmf import read_capture, write_capture


def test_write_read_roundtrip(tmp_path: Path, make_iq) -> None:
    samples = make_iq([(100e3, 1.0)])
    ref = write_capture(
        samples, tmp_path / "cap", center_freq=433e6, sample_rate=1e6
    )

    assert ref.path == tmp_path / "cap.sigmf-data"
    assert ref.path.exists()
    assert (tmp_path / "cap.sigmf-meta").exists()
    assert ref.num_samples == len(samples)
    assert ref.center_freq == 433e6

    loaded, ref2 = read_capture(ref.path)
    np.testing.assert_array_equal(loaded, samples)
    assert ref2 == ref


def test_meta_is_valid_sigmf(tmp_path: Path, make_iq) -> None:
    write_capture(make_iq([]), tmp_path / "cap", center_freq=1e9, sample_rate=2e6)
    meta = json.loads((tmp_path / "cap.sigmf-meta").read_text())
    assert meta["global"]["core:datatype"] == "cf32_le"
    assert meta["global"]["core:sample_rate"] == 2e6
    assert meta["captures"][0]["core:frequency"] == 1e9


def test_read_accepts_meta_or_base_path(tmp_path: Path, make_iq) -> None:
    samples = make_iq([])
    write_capture(samples, tmp_path / "cap", center_freq=0.0, sample_rate=1e6)
    for p in (tmp_path / "cap.sigmf-meta", tmp_path / "cap"):
        loaded, ref = read_capture(p)
        assert ref.num_samples == len(samples)
```

- [ ] **Step 3: Run test to verify it fails**

Run: `uv run pytest tests/marconi/test_sigmf.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'marconi.sigmf'`

- [ ] **Step 4: Implement**

`src/marconi/sigmf.py`:
```python
import json
from pathlib import Path

import numpy as np

from marconi.models import CaptureRef

SIGMF_VERSION = "1.0.0"


def _base(path: Path) -> Path:
    name = path.name
    for suffix in (".sigmf-data", ".sigmf-meta"):
        if name.endswith(suffix):
            return path.with_name(name[: -len(suffix)])
    return path


def write_capture(
    samples: np.ndarray, path: Path, center_freq: float, sample_rate: float
) -> CaptureRef:
    """Write complex64 samples as a SigMF pair; `path` may omit the extension."""
    base = _base(path)
    base.parent.mkdir(parents=True, exist_ok=True)
    data_path = base.with_name(base.name + ".sigmf-data")
    meta_path = base.with_name(base.name + ".sigmf-meta")

    samples = np.asarray(samples, dtype=np.complex64)
    samples.tofile(data_path)

    meta = {
        "global": {
            "core:datatype": "cf32_le",
            "core:sample_rate": sample_rate,
            "core:version": SIGMF_VERSION,
        },
        "captures": [{"core:sample_start": 0, "core:frequency": center_freq}],
        "annotations": [],
    }
    meta_path.write_text(json.dumps(meta, indent=2))

    return CaptureRef(
        path=data_path,
        center_freq=center_freq,
        sample_rate=sample_rate,
        num_samples=len(samples),
    )


def read_capture(path: Path) -> tuple[np.ndarray, CaptureRef]:
    """Read a SigMF pair; `path` may be the data file, meta file, or base."""
    base = _base(Path(path))
    data_path = base.with_name(base.name + ".sigmf-data")
    meta_path = base.with_name(base.name + ".sigmf-meta")

    meta = json.loads(meta_path.read_text())
    datatype = meta["global"]["core:datatype"]
    if datatype != "cf32_le":
        raise ValueError(f"unsupported SigMF datatype: {datatype}")

    samples = np.fromfile(data_path, dtype=np.complex64)
    ref = CaptureRef(
        path=data_path,
        center_freq=float(meta["captures"][0].get("core:frequency", 0.0)),
        sample_rate=float(meta["global"]["core:sample_rate"]),
        num_samples=len(samples),
    )
    return samples, ref
```

- [ ] **Step 5: Run test to verify it passes**

Run: `uv run pytest tests/marconi/test_sigmf.py -v`
Expected: 3 passed

- [ ] **Step 6: Commit**

```bash
git add src/marconi/sigmf.py tests/marconi/test_sigmf.py tests/marconi/conftest.py
git commit -m "Add minimal SigMF capture write/read"
```

---

### Task 4: Workspace (the user's project directory)

**Files:**
- Create: `src/marconi/workspace.py`
- Test: `tests/marconi/test_workspace.py`

- [ ] **Step 1: Write the failing test**

`tests/marconi/test_workspace.py`:
```python
from pathlib import Path

from marconi.workspace import Workspace


def test_subdirs_created_on_demand(tmp_path: Path) -> None:
    ws = Workspace(tmp_path)
    cap = ws.new_capture_path("scan")
    assert cap == tmp_path / "captures" / "scan"
    assert cap.parent.is_dir()
    png = ws.new_render_path("spec")
    assert png == tmp_path / "renders" / "spec.png"
    assert png.parent.is_dir()


def test_paths_deduplicate(tmp_path: Path) -> None:
    ws = Workspace(tmp_path)
    first = ws.new_capture_path("scan")
    (first.parent / "scan.sigmf-data").touch()
    second = ws.new_capture_path("scan")
    assert second == tmp_path / "captures" / "scan-1"

    p1 = ws.new_render_path("spec")
    p1.touch()
    p2 = ws.new_render_path("spec")
    assert p2 == tmp_path / "renders" / "spec-1.png"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/marconi/test_workspace.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'marconi.workspace'`

- [ ] **Step 3: Implement**

`src/marconi/workspace.py`:
```python
from pathlib import Path


class Workspace:
    """The user's RF project directory.

    Layout: captures/ renders/ pipelines/ scenes/ — created on demand.
    Artifacts are exchanged as paths into this directory, never as blobs.
    """

    def __init__(self, root: Path | str = ".") -> None:
        self.root = Path(root)

    def _subdir(self, name: str) -> Path:
        d = self.root / name
        d.mkdir(parents=True, exist_ok=True)
        return d

    @staticmethod
    def _dedupe(directory: Path, stem: str, probe_suffix: str, suffix: str) -> Path:
        candidate = stem
        i = 0
        while list(directory.glob(candidate + probe_suffix + "*")) or (
            directory / (candidate + suffix)
        ).exists():
            i += 1
            candidate = f"{stem}-{i}"
        return directory / (candidate + suffix)

    def new_capture_path(self, name: str) -> Path:
        """Extension-less base path for a SigMF pair under captures/."""
        return self._dedupe(self._subdir("captures"), name, ".sigmf", "")

    def new_render_path(self, name: str) -> Path:
        return self._dedupe(self._subdir("renders"), name, ".png", ".png")

    def new_scene_path(self, name: str) -> Path:
        return self._dedupe(self._subdir("scenes"), name, ".yaml", ".yaml")

    def new_pipeline_path(self, name: str) -> Path:
        return self._dedupe(self._subdir("pipelines"), name, ".yaml", ".yaml")
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/marconi/test_workspace.py -v`
Expected: 2 passed

- [ ] **Step 5: Commit**

```bash
git add src/marconi/workspace.py tests/marconi/test_workspace.py
git commit -m "Add Workspace project-directory conventions"
```

---

### Task 5: load_capture op (SigMF, raw cf32, wav)

`load_capture` ingests external files. SigMF files are referenced in place; raw `.cf32` and `.wav` files are converted into SigMF inside the workspace (one canonical on-disk form for everything downstream).

**Files:**
- Create: `src/marconi/ops/capture.py`
- Test: `tests/marconi/test_load_capture.py`

- [ ] **Step 1: Write the failing test**

`tests/marconi/test_load_capture.py`:
```python
from pathlib import Path

import numpy as np
import pytest
from scipy.io import wavfile

from marconi.ops.capture import load_capture
from marconi.sigmf import read_capture, write_capture
from marconi.workspace import Workspace


def test_load_sigmf_in_place(tmp_path: Path, make_iq) -> None:
    ws = Workspace(tmp_path / "project")
    src = write_capture(
        make_iq([]), tmp_path / "ext", center_freq=433e6, sample_rate=1e6
    )
    ref = load_capture(src.path, ws)
    assert ref == src  # referenced in place, not copied


def test_load_raw_cf32(tmp_path: Path, make_iq) -> None:
    ws = Workspace(tmp_path / "project")
    samples = make_iq([(10e3, 1.0)])
    raw = tmp_path / "ext.cf32"
    samples.tofile(raw)

    ref = load_capture(raw, ws, sample_rate=1e6, center_freq=433e6)
    assert ref.path.is_relative_to(ws.root / "captures")
    loaded, _ = read_capture(ref.path)
    np.testing.assert_array_equal(loaded, samples)
    assert ref.sample_rate == 1e6


def test_load_raw_cf32_requires_sample_rate(tmp_path: Path, make_iq) -> None:
    ws = Workspace(tmp_path / "project")
    raw = tmp_path / "ext.cf32"
    make_iq([]).tofile(raw)
    with pytest.raises(ValueError, match="sample_rate"):
        load_capture(raw, ws)


def test_load_stereo_wav_as_iq(tmp_path: Path) -> None:
    ws = Workspace(tmp_path / "project")
    rate = 48000
    t = np.arange(4800) / rate
    i = (np.cos(2 * np.pi * 1000 * t) * 30000).astype(np.int16)
    q = (np.sin(2 * np.pi * 1000 * t) * 30000).astype(np.int16)
    wav = tmp_path / "ext.wav"
    wavfile.write(wav, rate, np.stack([i, q], axis=1))

    ref = load_capture(wav, ws)
    assert ref.sample_rate == rate
    loaded, _ = read_capture(ref.path)
    assert loaded.dtype == np.complex64
    assert len(loaded) == 4800
    # normalized to <= 1.0 magnitude
    assert np.abs(loaded).max() <= 1.0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/marconi/test_load_capture.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'marconi.ops.capture'`

- [ ] **Step 3: Implement**

`src/marconi/ops/capture.py`:
```python
from pathlib import Path

import numpy as np
from scipy.io import wavfile

from marconi import sigmf
from marconi.models import CaptureRef
from marconi.workspace import Workspace


def load_capture(
    path: Path | str,
    workspace: Workspace,
    sample_rate: float | None = None,
    center_freq: float | None = None,
) -> CaptureRef:
    """Ingest an external IQ file.

    SigMF files are referenced in place. Raw .cf32 (complex64 interleaved)
    and .wav (stereo = I/Q) files are converted to SigMF in the workspace.
    """
    path = Path(path)
    name = path.name

    if name.endswith((".sigmf-data", ".sigmf-meta")):
        _, ref = sigmf.read_capture(path)
        return ref

    if name.endswith(".cf32"):
        if sample_rate is None:
            raise ValueError("sample_rate is required for raw .cf32 files")
        samples = np.fromfile(path, dtype=np.complex64)
        return sigmf.write_capture(
            samples,
            workspace.new_capture_path(path.stem),
            center_freq=center_freq or 0.0,
            sample_rate=sample_rate,
        )

    if name.endswith(".wav"):
        rate, data = wavfile.read(path)
        if np.issubdtype(data.dtype, np.integer):
            data = data.astype(np.float32) / np.iinfo(data.dtype).max
        else:
            data = data.astype(np.float32)
        if data.ndim == 2 and data.shape[1] >= 2:
            samples = (data[:, 0] + 1j * data[:, 1]).astype(np.complex64)
        else:
            samples = data.reshape(-1).astype(np.complex64)
        return sigmf.write_capture(
            samples,
            workspace.new_capture_path(path.stem),
            center_freq=center_freq or 0.0,
            sample_rate=float(sample_rate or rate),
        )

    raise ValueError(f"unsupported capture format: {name}")
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/marconi/test_load_capture.py -v`
Expected: 4 passed

- [ ] **Step 5: Commit**

```bash
git add src/marconi/ops/capture.py tests/marconi/test_load_capture.py
git commit -m "Add load_capture for SigMF, raw cf32, and wav files"
```

---

### Task 6: analyze.psd

**Files:**
- Create: `src/marconi/ops/analyze.py`
- Test: `tests/marconi/test_analyze_psd.py`

- [ ] **Step 1: Write the failing test**

`tests/marconi/test_analyze_psd.py`:
```python
from pathlib import Path

import numpy as np

from marconi.ops.analyze import psd
from marconi.sigmf import write_capture


def test_psd_finds_tone_at_absolute_freq(tmp_path: Path, make_iq) -> None:
    # tone at +100 kHz offset, center 433 MHz -> absolute 433.1 MHz
    ref = write_capture(
        make_iq([(100e3, 1.0)]), tmp_path / "cap",
        center_freq=433e6, sample_rate=1e6,
    )
    result = psd(ref)

    assert len(result.freqs) == len(result.psd_db)
    assert len(result.peaks) >= 1
    strongest = max(result.peaks, key=lambda p: p.power_db)
    assert abs(strongest.freq - 433.1e6) < 1e3
    assert strongest.power_db > result.noise_floor_db + 20


def test_psd_noise_floor_sane(tmp_path: Path, make_iq) -> None:
    ref = write_capture(
        make_iq([], noise_amplitude=0.01), tmp_path / "cap",
        center_freq=0.0, sample_rate=1e6,
    )
    result = psd(ref)
    # pure noise: no strong peaks, floor near median power
    psd_arr = np.array(result.psd_db)
    assert abs(result.noise_floor_db - float(np.median(psd_arr))) < 1.0
    assert result.peaks == []
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/marconi/test_analyze_psd.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'marconi.ops.analyze'`

- [ ] **Step 3: Implement**

`src/marconi/ops/analyze.py`:
```python
import numpy as np
from scipy import signal as sp_signal

from marconi.models import (
    Burst,
    CaptureRef,
    DetectedSignal,
    PSDResult,
    SignalMeasurement,
    SignalPeak,
)


def _read_samples(capture: CaptureRef) -> np.ndarray:
    return np.fromfile(capture.path, dtype=np.complex64)


def _welch(
    capture: CaptureRef, nperseg: int = 4096
) -> tuple[np.ndarray, np.ndarray]:
    """Two-sided Welch PSD in dB, freqs absolute (Hz), ascending."""
    x = _read_samples(capture)
    nperseg = min(nperseg, len(x))
    freqs, p = sp_signal.welch(
        x,
        fs=capture.sample_rate,
        nperseg=nperseg,
        return_onesided=False,
        detrend=False,
    )
    freqs = np.fft.fftshift(freqs) + capture.center_freq
    p_db = 10 * np.log10(np.fft.fftshift(p) + 1e-30)
    return freqs, p_db


def psd(capture: CaptureRef, nperseg: int = 4096) -> PSDResult:
    freqs, p_db = _welch(capture, nperseg)
    noise_floor = float(np.median(p_db))
    peak_idx, _ = sp_signal.find_peaks(
        p_db, height=noise_floor + 10.0, distance=max(1, len(p_db) // 512)
    )
    peaks = [
        SignalPeak(freq=float(freqs[i]), power_db=float(p_db[i]))
        for i in peak_idx
    ]
    peaks.sort(key=lambda p: p.power_db, reverse=True)
    return PSDResult(
        freqs=[float(f) for f in freqs],
        psd_db=[float(v) for v in p_db],
        noise_floor_db=noise_floor,
        peaks=peaks,
    )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/marconi/test_analyze_psd.py -v`
Expected: 2 passed

- [ ] **Step 5: Commit**

```bash
git add src/marconi/ops/analyze.py tests/marconi/test_analyze_psd.py
git commit -m "Add analyze.psd with peak detection"
```

---

### Task 7: analyze.find_signals

**Files:**
- Modify: `src/marconi/ops/analyze.py` (append)
- Test: `tests/marconi/test_analyze_signals.py`

- [ ] **Step 1: Write the failing test**

`tests/marconi/test_analyze_signals.py`:
```python
from pathlib import Path

from marconi.ops.analyze import find_signals
from marconi.sigmf import write_capture


def test_finds_two_tones(tmp_path: Path, make_iq) -> None:
    ref = write_capture(
        make_iq([(100e3, 1.0), (-200e3, 0.5)]),
        tmp_path / "cap",
        center_freq=100e6,
        sample_rate=1e6,
    )
    signals = find_signals(ref)
    assert len(signals) == 2
    signals.sort(key=lambda s: s.center_freq)
    assert abs(signals[0].center_freq - 99.8e6) < 2e3
    assert abs(signals[1].center_freq - 100.1e6) < 2e3
    assert all(s.snr_db > 20 for s in signals)
    # stronger tone reported stronger
    assert signals[1].peak_power_db > signals[0].peak_power_db


def test_pure_noise_finds_nothing(tmp_path: Path, make_iq) -> None:
    ref = write_capture(
        make_iq([]), tmp_path / "cap", center_freq=0.0, sample_rate=1e6
    )
    assert find_signals(ref) == []
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/marconi/test_analyze_signals.py -v`
Expected: FAIL with `ImportError: cannot import name 'find_signals'`

- [ ] **Step 3: Implement** (append to `src/marconi/ops/analyze.py`)

```python
def find_signals(
    capture: CaptureRef,
    threshold_db: float = 6.0,
    min_bandwidth: float = 500.0,
    nperseg: int = 4096,
) -> list[DetectedSignal]:
    """Segment the PSD into contiguous regions above the noise floor."""
    freqs, p_db = _welch(capture, nperseg)
    noise_floor = float(np.median(p_db))
    bin_bw = float(freqs[1] - freqs[0])

    above = np.where(p_db > noise_floor + threshold_db)[0]
    if len(above) == 0:
        return []

    splits = np.where(np.diff(above) > 1)[0]
    groups = np.split(above, splits + 1)

    signals = []
    for g in groups:
        bandwidth = len(g) * bin_bw
        if bandwidth < min_bandwidth:
            continue
        p_lin = 10 ** (p_db[g] / 10)
        center = float(np.sum(freqs[g] * p_lin) / np.sum(p_lin))
        peak_db = float(p_db[g].max())
        signals.append(
            DetectedSignal(
                center_freq=center,
                bandwidth=float(bandwidth),
                peak_power_db=peak_db,
                snr_db=peak_db - noise_floor,
            )
        )
    signals.sort(key=lambda s: s.peak_power_db, reverse=True)
    return signals
```

Note: with nperseg=4096 at 1 Msps, bin width is ~244 Hz. A strong tone's window leakage puts several adjacent bins above threshold (~1 kHz group), so it survives `min_bandwidth=500.0`, while a single-bin noise spike (244 Hz) is rejected.

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/marconi/test_analyze_signals.py -v`
Expected: 2 passed

- [ ] **Step 5: Commit**

```bash
git add src/marconi/ops/analyze.py tests/marconi/test_analyze_signals.py
git commit -m "Add analyze.find_signals PSD segmentation"
```

---

### Task 8: analyze.measure

**Files:**
- Modify: `src/marconi/ops/analyze.py` (append)
- Test: `tests/marconi/test_analyze_measure.py`

- [ ] **Step 1: Write the failing test**

`tests/marconi/test_analyze_measure.py`:
```python
from pathlib import Path

from marconi.ops.analyze import measure
from marconi.sigmf import write_capture


def test_measure_tone(tmp_path: Path, make_iq) -> None:
    ref = write_capture(
        make_iq([(100e3, 1.0)]), tmp_path / "cap",
        center_freq=100e6, sample_rate=1e6,
    )
    m = measure(ref, center_freq=100.1e6, search_bandwidth=100e3)
    assert abs(m.center_freq - 100.1e6) < 1e3
    assert m.snr_db > 20
    # a pure tone occupies very little bandwidth
    assert m.occupied_bw_99 < 10e3


def test_measure_uses_search_window(tmp_path: Path, make_iq) -> None:
    # two tones; measuring around one must not report the other
    ref = write_capture(
        make_iq([(100e3, 0.5), (-200e3, 1.0)]),
        tmp_path / "cap",
        center_freq=100e6,
        sample_rate=1e6,
    )
    m = measure(ref, center_freq=100.1e6, search_bandwidth=100e3)
    assert abs(m.center_freq - 100.1e6) < 1e3
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/marconi/test_analyze_measure.py -v`
Expected: FAIL with `ImportError: cannot import name 'measure'`

- [ ] **Step 3: Implement** (append to `src/marconi/ops/analyze.py`)

```python
def measure(
    capture: CaptureRef,
    center_freq: float,
    search_bandwidth: float = 200e3,
    nperseg: int = 4096,
) -> SignalMeasurement:
    """Measure the signal nearest center_freq within the search window."""
    freqs, p_db = _welch(capture, nperseg)
    noise_floor = float(np.median(p_db))

    sel = (freqs >= center_freq - search_bandwidth / 2) & (
        freqs <= center_freq + search_bandwidth / 2
    )
    if not np.any(sel):
        raise ValueError("search window is outside the capture's spectrum")

    f_sel = freqs[sel]
    p_sel_db = p_db[sel]
    p_lin = 10 ** (p_sel_db / 10)
    total = float(np.sum(p_lin))

    csum = np.cumsum(p_lin) / total
    lo = float(f_sel[int(np.searchsorted(csum, 0.005))])
    hi = float(f_sel[min(int(np.searchsorted(csum, 0.995)), len(f_sel) - 1)])

    bin_bw = float(freqs[1] - freqs[0])
    return SignalMeasurement(
        center_freq=float(f_sel[int(np.argmax(p_sel_db))]),
        occupied_bw_99=hi - lo,
        power_db=10 * float(np.log10(total * bin_bw + 1e-30)),
        snr_db=float(p_sel_db.max()) - noise_floor,
    )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/marconi/test_analyze_measure.py -v`
Expected: 2 passed

- [ ] **Step 5: Commit**

```bash
git add src/marconi/ops/analyze.py tests/marconi/test_analyze_measure.py
git commit -m "Add analyze.measure for windowed signal measurement"
```

---

### Task 9: analyze.detect_bursts

**Files:**
- Modify: `src/marconi/ops/analyze.py` (append)
- Test: `tests/marconi/test_analyze_bursts.py`

- [ ] **Step 1: Write the failing test**

`tests/marconi/test_analyze_bursts.py`:
```python
from pathlib import Path

import numpy as np

from marconi.ops.analyze import detect_bursts
from marconi.sigmf import write_capture


def _bursty_signal(
    fs: float = 1e6, total: float = 0.05, on_start: float = 0.01, on_dur: float = 0.01
) -> np.ndarray:
    rng = np.random.default_rng(0)
    n = int(fs * total)
    x = (rng.normal(0, 0.01, n) + 1j * rng.normal(0, 0.01, n)).astype(np.complex64)
    t = np.arange(n) / fs
    tone = np.exp(2j * np.pi * 50e3 * t).astype(np.complex64)
    on = slice(int(on_start * fs), int((on_start + on_dur) * fs))
    x[on] += tone[on]
    return x


def test_detects_single_burst(tmp_path: Path) -> None:
    ref = write_capture(
        _bursty_signal(), tmp_path / "cap", center_freq=0.0, sample_rate=1e6
    )
    bursts = detect_bursts(ref)
    assert len(bursts) == 1
    b = bursts[0]
    assert abs(b.start_time - 0.01) < 2e-3
    assert abs(b.duration - 0.01) < 4e-3


def test_continuous_signal_is_not_bursty(tmp_path: Path, make_iq) -> None:
    ref = write_capture(
        make_iq([(50e3, 1.0)]), tmp_path / "cap", center_freq=0.0, sample_rate=1e6
    )
    # always-on signal: power never drops below threshold -> no distinct bursts
    assert detect_bursts(ref) == []
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/marconi/test_analyze_bursts.py -v`
Expected: FAIL with `ImportError: cannot import name 'detect_bursts'`

- [ ] **Step 3: Implement** (append to `src/marconi/ops/analyze.py`)

```python
def detect_bursts(
    capture: CaptureRef,
    window: float = 1e-3,
    threshold_db: float = 6.0,
) -> list[Burst]:
    """Detect on/off bursts from the smoothed power envelope.

    The threshold is median + threshold_db; an always-on signal therefore
    yields no bursts (its median power IS the signal).
    """
    x = _read_samples(capture)
    fs = capture.sample_rate
    win = max(1, int(window * fs))
    power = np.convolve(np.abs(x) ** 2, np.ones(win) / win, mode="same")
    p_db = 10 * np.log10(power + 1e-30)
    threshold = float(np.median(p_db)) + threshold_db

    above = np.where(p_db > threshold)[0]
    if len(above) == 0:
        return []

    splits = np.where(np.diff(above) > 1)[0]
    groups = np.split(above, splits + 1)

    bursts = []
    for g in groups:
        duration = len(g) / fs
        if duration < 2 * window:
            continue
        bursts.append(
            Burst(
                start_time=float(g[0] / fs),
                duration=float(duration),
                mean_power_db=float(np.mean(p_db[g])),
            )
        )
    return bursts
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/marconi/test_analyze_bursts.py -v`
Expected: 2 passed

- [ ] **Step 5: Commit**

```bash
git add src/marconi/ops/analyze.py tests/marconi/test_analyze_bursts.py
git commit -m "Add analyze.detect_bursts envelope segmentation"
```

---

### Task 10: render ops (spectrogram, psd_plot, constellation)

**Files:**
- Create: `src/marconi/ops/render.py`
- Test: `tests/marconi/test_render.py`

- [ ] **Step 1: Write the failing test**

`tests/marconi/test_render.py`:
```python
from pathlib import Path

from marconi.ops.render import constellation, psd_plot, spectrogram
from marconi.sigmf import write_capture
from marconi.workspace import Workspace

PNG_MAGIC = b"\x89PNG"


def _capture(tmp_path: Path, make_iq):
    return write_capture(
        make_iq([(100e3, 1.0)]), tmp_path / "cap",
        center_freq=100e6, sample_rate=1e6,
    )


def test_spectrogram_renders_png(tmp_path: Path, make_iq) -> None:
    ws = Workspace(tmp_path / "project")
    result = spectrogram(_capture(tmp_path, make_iq), ws)
    assert result.kind == "spectrogram"
    assert result.path.read_bytes()[:4] == PNG_MAGIC
    assert result.path.stat().st_size > 5000


def test_psd_plot_renders_png(tmp_path: Path, make_iq) -> None:
    ws = Workspace(tmp_path / "project")
    result = psd_plot(_capture(tmp_path, make_iq), ws)
    assert result.kind == "psd"
    assert result.path.read_bytes()[:4] == PNG_MAGIC


def test_constellation_renders_png(tmp_path: Path, make_iq) -> None:
    ws = Workspace(tmp_path / "project")
    result = constellation(_capture(tmp_path, make_iq), ws)
    assert result.kind == "constellation"
    assert result.path.read_bytes()[:4] == PNG_MAGIC
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/marconi/test_render.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'marconi.ops.render'`

- [ ] **Step 3: Implement**

`src/marconi/ops/render.py`:
```python
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from marconi.models import CaptureRef, RenderResult
from marconi.ops.analyze import _read_samples, psd
from marconi.workspace import Workspace


def _save(fig: "plt.Figure", workspace: Workspace, name: str, kind: str) -> RenderResult:
    out = workspace.new_render_path(name)
    fig.savefig(out, dpi=100, bbox_inches="tight")
    plt.close(fig)
    return RenderResult(path=out, kind=kind)


def spectrogram(
    capture: CaptureRef,
    workspace: Workspace,
    name: str = "spectrogram",
    nfft: int = 1024,
) -> RenderResult:
    x = _read_samples(capture)
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.specgram(
        x,
        NFFT=min(nfft, len(x)),
        Fs=capture.sample_rate,
        Fc=capture.center_freq,
        noverlap=min(nfft, len(x)) // 2,
    )
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Frequency (Hz)")
    ax.set_title(f"Spectrogram @ {capture.center_freq/1e6:.3f} MHz")
    return _save(fig, workspace, name, "spectrogram")


def psd_plot(
    capture: CaptureRef, workspace: Workspace, name: str = "psd"
) -> RenderResult:
    result = psd(capture)
    fig, ax = plt.subplots(figsize=(10, 6))
    freqs_mhz = np.array(result.freqs) / 1e6
    ax.plot(freqs_mhz, result.psd_db, linewidth=0.8)
    ax.axhline(
        result.noise_floor_db, linestyle="--", color="gray", label="noise floor"
    )
    for peak in result.peaks[:10]:
        ax.plot(peak.freq / 1e6, peak.power_db, "rv")
    ax.set_xlabel("Frequency (MHz)")
    ax.set_ylabel("PSD (dB/Hz)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    return _save(fig, workspace, name, "psd")


def constellation(
    capture: CaptureRef,
    workspace: Workspace,
    name: str = "constellation",
    max_points: int = 5000,
) -> RenderResult:
    x = _read_samples(capture)
    step = max(1, len(x) // max_points)
    pts = x[::step]
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(pts.real, pts.imag, s=2, alpha=0.4)
    ax.set_xlabel("I")
    ax.set_ylabel("Q")
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.3)
    return _save(fig, workspace, name, "constellation")
```

Note: `_read_samples` is intentionally reused from `analyze` — it is the single reader for capture data. If flake8 complains about the module-level `matplotlib.use("Agg")` before imports (E402), add `# noqa: E402` to the two imports below it.

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/marconi/test_render.py -v`
Expected: 3 passed

- [ ] **Step 5: Commit**

```bash
git add src/marconi/ops/render.py tests/marconi/test_render.py
git commit -m "Add render ops: spectrogram, psd_plot, constellation"
```

---

### Task 11: Public API and end-to-end smoke test

**Files:**
- Modify: `src/marconi/__init__.py`
- Test: `tests/marconi/test_api.py`

- [ ] **Step 1: Write the failing test**

`tests/marconi/test_api.py`:
```python
from pathlib import Path

import numpy as np

import marconi


def test_public_api_surface() -> None:
    for name in (
        "Workspace",
        "CaptureRef",
        "load_capture",
        "psd",
        "find_signals",
        "measure",
        "detect_bursts",
        "spectrogram",
        "psd_plot",
        "constellation",
        "write_capture",
        "read_capture",
    ):
        assert hasattr(marconi, name), name


def test_end_to_end_analysis_flow(tmp_path: Path) -> None:
    """Synthesize IQ -> workspace capture -> find -> measure -> render."""
    ws = marconi.Workspace(tmp_path)
    fs = 1e6
    t = np.arange(int(fs * 0.05)) / fs
    rng = np.random.default_rng(0)
    x = (
        np.exp(2j * np.pi * 100e3 * t)
        + rng.normal(0, 0.01, len(t))
        + 1j * rng.normal(0, 0.01, len(t))
    ).astype(np.complex64)

    ref = marconi.write_capture(
        x, ws.new_capture_path("synth"), center_freq=433e6, sample_rate=fs
    )
    signals = marconi.find_signals(ref)
    assert len(signals) == 1

    m = marconi.measure(ref, center_freq=signals[0].center_freq)
    assert m.snr_db > 20

    render = marconi.spectrogram(ref, ws)
    assert render.path.exists()
    # the workspace now holds a reusable RF project
    assert (ws.root / "captures").is_dir()
    assert (ws.root / "renders").is_dir()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/marconi/test_api.py -v`
Expected: FAIL with `AssertionError: Workspace` (attribute missing)

- [ ] **Step 3: Implement**

`src/marconi/__init__.py`:
```python
"""Marconi: LLM-driven RF control and analysis."""

from marconi.models import (
    Burst,
    CaptureRef,
    DetectedSignal,
    PSDResult,
    RenderResult,
    SignalMeasurement,
    SignalPeak,
)
from marconi.ops.analyze import detect_bursts, find_signals, measure, psd
from marconi.ops.capture import load_capture
from marconi.ops.render import constellation, psd_plot, spectrogram
from marconi.sigmf import read_capture, write_capture
from marconi.workspace import Workspace

__version__ = "0.1.0"

__all__ = [
    "Burst",
    "CaptureRef",
    "DetectedSignal",
    "PSDResult",
    "RenderResult",
    "SignalMeasurement",
    "SignalPeak",
    "Workspace",
    "constellation",
    "detect_bursts",
    "find_signals",
    "load_capture",
    "measure",
    "psd",
    "psd_plot",
    "read_capture",
    "spectrogram",
    "write_capture",
]
```

- [ ] **Step 4: Run the full marconi suite**

Run: `uv run pytest tests/marconi -v`
Expected: all 25 tests pass

- [ ] **Step 5: Verify the POC still works (no regressions)**

Run: `uv run pytest tests/unit -v`
Expected: same pass/fail state as before this plan started (these tests exercise the old gr-mcp POC and need system GNU Radio; we must not have broken them).

- [ ] **Step 6: Commit**

```bash
git add src/marconi/__init__.py tests/marconi/test_api.py
git commit -m "Export marconi public API with end-to-end smoke test"
```

---

## Self-review checklist (run after all tasks)

- All `tests/marconi` tests green.
- `uv run python -c "import marconi"` works without GNU Radio on the path.
- No file under `src/marconi/` imports `gnuradio` or anything from `src/gnuradio_mcp/`.
- The workspace directory layout matches the spec: `captures/`, `renders/`, `scenes/`, `pipelines/`.
