from __future__ import annotations

import inspect
import time
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from helpers._capture import WEDGED_SECONDS, wedged_probe

from marconi.capture import record as record_mod
from marconi.deadline import RunTimeout, bounded, set_deadline
from marconi.errors import classify_error
from marconi.mcp.tools import TOOLS, capture_tool, run_rx_tool, validate_modem
from marconi.mcp.workspace import discarded_if_unused, new_run_dir

# a (15,11) Hamming-like shape asking for 2-error correction: the validator
# enumerates every weight-1 and weight-2 pattern, which is where an unbounded
# compile used to sit
_HEAVY_SPEC: dict[str, Any] = {
    "symbol_rate": 1000.0,
    "path": [
        {
            "conv": "block_code",
            "code_bits": 200,
            "data_bits": 180,
            "parity_masks": [1 << (i % 180) for i in range(20)],
            "correct": 2,
        }
    ],
}


def test_bounded_raises_once_the_deadline_has_passed() -> None:
    with set_deadline(0.0):
        with pytest.raises(RunTimeout):
            list(bounded(range(4)))


def test_bounded_passes_items_through_without_a_deadline() -> None:
    assert list(bounded(range(4))) == [0, 1, 2, 3]


def test_validate_modem_compile_is_bounded() -> None:
    # the compile had no timeout at all while running the same validators that
    # enumerate syndrome tables. A timeout is not "your spec is invalid", so it
    # raises for the boundary to code as [deadline_exceeded] — the same shape
    # run_rx gives it — rather than returning valid=False
    with pytest.raises(RunTimeout):
        validate_modem(_HEAVY_SPEC, sample_rate=1e5, timeout=0.0)


def test_validate_modem_untimed_still_answers_normally() -> None:
    # the same spec with room to run reaches its real verdict (these masks
    # collide, so it is a spec error) instead of a deadline
    out = validate_modem(_HEAVY_SPEC, sample_rate=1e5)
    assert out["valid"] is False
    assert "deadline" not in str(out["errors"]).lower()


def test_run_rx_parses_the_spec_inside_its_own_deadline(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("MARCONI_WORKSPACE", str(tmp_path))
    cap = tmp_path / "iq.cf32"
    np.zeros(4096, np.complex64).tofile(cap)
    with pytest.raises(RunTimeout):
        run_rx_tool(
            spec=_HEAVY_SPEC,
            sample_rate=1e5,
            capture_path=str(cap),
            input_item_type=None,
            timeout=0.0,
        )


def test_every_tool_that_can_block_takes_a_timeout() -> None:
    """capture was the one tool with a hang path and the only blocking tool
    with no timeout at all: siblings offered 60/180/180 and it offered
    nothing, so an agent had no way to bound a wedged radio."""
    for name in ("validate_modem", "run_rx", "survey", "capture"):
        params = inspect.signature(TOOLS[name]).parameters
        assert "timeout" in params, name


def test_capture_bounds_a_wedged_radio_and_says_so(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The docstring's timeout paragraph, driven: a device that never answers
    is killed and reported as a deadline, not waited on."""
    doc = capture_tool.__doc__ or ""
    assert "timeout" in doc and "deadline_exceeded" in doc
    # the cap the prose quotes is the constant enforcing it, not a number that
    # can drift away from it
    assert f"{record_mod._PROBE_TIMEOUT_S:g} s" in doc
    monkeypatch.setenv("MARCONI_WORKSPACE", str(tmp_path))
    monkeypatch.setattr(record_mod, "_probe_device", wedged_probe)
    start = time.monotonic()
    with pytest.raises(RunTimeout) as excinfo:
        capture_tool(center_hz=100e6, duration_s=0.01, timeout=2.0)
    elapsed = time.monotonic() - start
    assert elapsed < WEDGED_SECONDS / 2, f"capture ran unbounded for {elapsed:.1f}s"
    assert classify_error(excinfo.value)[0] == "deadline_exceeded"


def test_capture_refuses_a_timeout_that_cannot_cover_the_recording(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A cap below the recording it is capping is a spec error the caller can
    retype, not a run to start and kill."""
    monkeypatch.setenv("MARCONI_WORKSPACE", str(tmp_path))
    with pytest.raises(ValueError) as excinfo:
        capture_tool(center_hz=100e6, duration_s=5.0, timeout=1.0)
    code, message = classify_error(excinfo.value)
    assert code == "invalid_argument"
    assert "timeout" in message and "duration_s" in message


def test_failed_run_leaves_no_empty_run_dir(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("MARCONI_WORKSPACE", str(tmp_path))
    runs = tmp_path / "marconi-runs"
    with pytest.raises(FileNotFoundError):
        run_rx_tool(
            spec={"symbol_rate": 1e3, "path": []},
            sample_rate=1e3,
            capture_path=str(tmp_path / "missing.cf32"),
        )
    assert list(runs.glob("rx-*")) == []


def test_run_dir_survives_when_it_holds_output(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # partial output is evidence: only an untouched directory is discarded
    monkeypatch.setenv("MARCONI_WORKSPACE", str(tmp_path))
    with discarded_if_unused(new_run_dir("rx")) as d:
        (d / "out.u8").write_bytes(b"\x01")
    assert d.is_dir()
