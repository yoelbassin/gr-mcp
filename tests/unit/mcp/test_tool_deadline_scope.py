from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pytest

from marconi.deadline import RunTimeout, bounded, set_deadline
from marconi.mcp.tools import run_rx_tool, run_tx_tool, validate_modem
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


def test_a_failed_run_tx_leaves_no_directory_behind(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """run_tx was the one run-dir path the cleanup did not reach; every failing
    call left an empty directory under marconi-runs/."""
    monkeypatch.setenv("MARCONI_WORKSPACE", str(tmp_path))
    runs = tmp_path / "marconi-runs"
    with pytest.raises(FileNotFoundError):
        run_tx_tool(
            spec={"symbol_rate": 1e3, "path": [{"conv": "slice"}]},
            sample_rate=1e3,
            bits_path=str(tmp_path / "missing.u8"),
        )
    assert list(runs.glob("tx-*")) == []


def test_run_tx_keeps_the_directory_holding_its_input_bits(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An inline `bits` string is written into the run dir, so a failure after
    that point still has something worth keeping."""
    monkeypatch.setenv("MARCONI_WORKSPACE", str(tmp_path))
    runs = tmp_path / "marconi-runs"
    with pytest.raises(FileNotFoundError):
        run_tx_tool(
            spec={"symbol_rate": 1e3, "path": [{"conv": "slice"}]},
            sample_rate=1e3,
            bits="0101",
            out_path=str(tmp_path / "nope" / "out.cf32"),
        )
    assert len(list(runs.glob("tx-*"))) == 1
