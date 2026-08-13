from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from pydantic import ValidationError

from marconi.engine.backends.gnuradio.runner import GnuRadioBackend, ensure_worker_warm
from marconi.engine.compile.compiler import compile_modem
from marconi.engine.compile.errors import CompileError
from marconi.engine.compile.ir import GrPipeline
from marconi.engine.modulation.ofdm.stages import CellSelectStep
from marconi.engine.stages.registry import stage_registry
from marconi.engine.types.descriptor import Carrier, Descriptor
from marconi.engine.types.enums import ItemType
from marconi.engine.types.levels import Level
from marconi.engine.types.models import Modem

SYM_C = Descriptor(Level.SYMBOLS, ItemType.C, Carrier.SOFT)


def _compile(
    modem: Modem, src: Path, snk: Path, start: Descriptor = SYM_C
) -> GrPipeline:
    return compile_modem(
        modem,
        stage_registry(),
        sample_rate=1.0,
        start=start,
        source_io={"path": str(src)},
        sink_io={"path": str(snk)},
    )


def test_cell_select_gathers_wanted_cells_to_front(tmp_path: Path) -> None:
    ensure_worker_warm()
    block, wanted = 12, [3, 7, 11]
    perm = wanted + [i for i in range(block) if i not in wanted]
    syms = (np.arange(3 * block, dtype=np.float32) + 1.0).astype(np.complex64)
    src = tmp_path / "in.cf32"
    syms.tofile(src)
    snk = tmp_path / "out.cf32"
    modem = Modem(
        symbol_rate=1.0,
        path=[CellSelectStep(select_perm=perm, keep=3)],
    )
    r = GnuRadioBackend().run_pipeline(_compile(modem, src, snk), timeout=30.0)
    assert r.status == "ok", r
    out = np.fromfile(snk, np.complex64)
    expect = np.concatenate([syms[b * block + np.asarray(wanted)] for b in range(3)])
    assert np.allclose(out, expect), (out.tolist(), expect.tolist())


def test_cell_select_pins_frame_len() -> None:
    out = stage_registry()["cell_select"].out_descriptor(
        SYM_C, CellSelectStep(select_perm=[1, 0, 2], keep=2)
    )
    assert out.frame_len == 2
    assert out.level is Level.SYMBOLS and out.item_type == "c"


def test_cell_select_rejects_non_permutation() -> None:
    with pytest.raises(ValidationError, match="permutation"):
        CellSelectStep(select_perm=[0, 0, 2], keep=1)


def test_cell_select_rejects_keep_beyond_block() -> None:
    with pytest.raises(ValidationError, match="keep"):
        CellSelectStep(select_perm=[0, 1], keep=3)


def test_cell_select_rejects_frame_len_mismatch(tmp_path: Path) -> None:
    framed = Descriptor(Level.SYMBOLS, ItemType.C, Carrier.SOFT, frame_len=5)
    modem = Modem(
        symbol_rate=1.0,
        path=[CellSelectStep(select_perm=list(range(12)), keep=3)],
    )
    with pytest.raises(CompileError, match="framed at 5"):
        _compile(modem, tmp_path / "a", tmp_path / "b", start=framed)


def test_cell_select_accepts_frame_len_whole_blocks_tile_frame(tmp_path: Path) -> None:
    # frame_len=12, select_perm len 3: whole gather blocks tile exactly into one
    # upstream frame (12 % 3 == 0, no straddle)
    framed = Descriptor(Level.SYMBOLS, ItemType.C, Carrier.SOFT, frame_len=12)
    modem = Modem(
        symbol_rate=1.0,
        path=[CellSelectStep(select_perm=[1, 0, 2], keep=2)],
    )
    # Should compile successfully without CompileError
    _compile(modem, tmp_path / "a", tmp_path / "b", start=framed)


def test_cell_select_accepts_frame_len_gather_spans_whole_frames(
    tmp_path: Path,
) -> None:
    # frame_len=3, select_perm len 12: gather span divides evenly by frame_len
    # (12 % 3 == 0, gather blocks span whole frames, no straddle)
    framed = Descriptor(Level.SYMBOLS, ItemType.C, Carrier.SOFT, frame_len=3)
    modem = Modem(
        symbol_rate=1.0,
        path=[CellSelectStep(select_perm=list(range(12)), keep=3)],
    )
    # Should compile successfully without CompileError
    _compile(modem, tmp_path / "a", tmp_path / "b", start=framed)
