from __future__ import annotations

from pathlib import Path
from typing import cast

import numpy as np
import pytest

from marconi.engine.backends.gnuradio.runner import GnuRadioBackend, ensure_worker_warm
from marconi.engine.compile.compiler import CompileError, compile_modem
from marconi.engine.stages.registry import stage_registry
from marconi.engine.types.descriptor import Carrier, Descriptor
from marconi.engine.types.levels import Level
from marconi.engine.types.models import ModemSpec, ModemStep

SYM_C = Descriptor(Level.SYMBOLS, "c", Carrier.SOFT)


def _compile(modem: ModemSpec, src: Path, snk: Path, start: Descriptor = SYM_C):
    return compile_modem(
        modem,
        stage_registry(),
        direction="rx",
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
    modem = ModemSpec(
        symbol_rate=1.0,
        path=[
            ModemStep(
                conv="cell_select",
                params={"select_perm": cast("list[float | int]", perm), "keep": 3},
            )
        ],
    )
    r = GnuRadioBackend().run_pipeline(_compile(modem, src, snk), timeout=30.0)
    assert r.status == "ok", r
    out = np.fromfile(snk, np.complex64)
    expect = np.concatenate([syms[b * block + np.asarray(wanted)] for b in range(3)])
    assert np.allclose(out, expect), (out.tolist(), expect.tolist())


def test_cell_select_pins_frame_len() -> None:
    out = stage_registry()["cell_select"].out_descriptor(
        SYM_C, {"select_perm": [1, 0, 2], "keep": 2}
    )
    assert out.frame_len == 2
    assert out.level is Level.SYMBOLS and out.item_type == "c"


def test_cell_select_rejects_non_permutation(tmp_path: Path) -> None:
    modem = ModemSpec(
        symbol_rate=1.0,
        path=[
            ModemStep(
                conv="cell_select",
                params={"select_perm": cast("list[float | int]", [0, 0, 2]), "keep": 1},
            )
        ],
    )
    with pytest.raises(Exception, match="permutation"):
        _compile(modem, tmp_path / "a", tmp_path / "b")


def test_cell_select_rejects_keep_beyond_block(tmp_path: Path) -> None:
    modem = ModemSpec(
        symbol_rate=1.0,
        path=[
            ModemStep(
                conv="cell_select",
                params={"select_perm": cast("list[float | int]", [0, 1]), "keep": 3},
            )
        ],
    )
    with pytest.raises(Exception, match="keep"):
        _compile(modem, tmp_path / "a", tmp_path / "b")


def test_cell_select_rejects_frame_len_mismatch(tmp_path: Path) -> None:
    framed = Descriptor(Level.SYMBOLS, "c", Carrier.SOFT, frame_len=5)
    modem = ModemSpec(
        symbol_rate=1.0,
        path=[
            ModemStep(
                conv="cell_select",
                params={
                    "select_perm": cast("list[float | int]", list(range(12))),
                    "keep": 3,
                },
            )
        ],
    )
    with pytest.raises(CompileError, match="framed at 5"):
        _compile(modem, tmp_path / "a", tmp_path / "b", start=framed)
