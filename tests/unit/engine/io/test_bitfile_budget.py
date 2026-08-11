"""bitfile's capture budget: an oversized read must name the size and the way
out, and a within-budget read must be untouched by the guard."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from marconi.engine.io import bitfile
from marconi.engine.io.bitfile import CaptureTooLarge, read_bits


def test_oversized_capture_raises_actionable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(bitfile, "_MAX_BITS", 64)
    path = tmp_path / "big.bits"
    bitfile.write_bits(path, np.zeros(65, np.uint8))
    with pytest.raises(CaptureTooLarge) as e:
        read_bits(path)
    msg = str(e.value)
    assert "65 items" in msg and "slice" in msg


def test_within_budget_reads_fine(tmp_path: Path) -> None:
    path = tmp_path / "ok.bits"
    bitfile.write_bits(path, np.array([0, 1, 1, 0], np.uint8))
    assert read_bits(path).tolist() == [0, 1, 1, 0]
