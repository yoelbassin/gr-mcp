"""The OFDM demod boundary carries its frame geometry across the seam.

ofdm_demod's own emit arithmetic fixes a frame at (data_syms + 1) * n_carriers
cells, and dqpsk_soft_demap re-declares that same geometry in its delay_cc /
keep_m_in_n line to drop the reference symbol — but the producer pinned
nothing, so the compiler had no frame to compare and a demap sized for a
different frame compiled clean: ofdm_demod(ds=2, nc=2) into
dqpsk_soft_demap(ds=5, nc=7), straight and through cell_select, both accepted.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from helpers import _lattice

from marconi.engine.compile.compiler import compile_modem
from marconi.engine.compile.errors import CompileError
from marconi.engine.modulation.ofdm.stages import (
    CellSelectStep,
    DqpskSoftDemapStep,
    OfdmCoherentSyncStep,
    OfdmDemodStep,
)
from marconi.engine.stages.registry import stage_registry
from marconi.engine.types.descriptor import Carrier, Descriptor
from marconi.engine.types.enums import ItemType
from marconi.engine.types.levels import Level
from marconi.engine.types.models import Modem
from marconi.engine.types.step import Step

IQ = Descriptor(Level.IQ, ItemType.C)
SYM_C = Descriptor(Level.SYMBOLS, ItemType.C, Carrier.SOFT)


def _demod(data_syms: int, n_carriers: int, fft_len: int = 8) -> OfdmDemodStep:
    sym_len = fft_len + 2
    return OfdmDemodStep(
        fft_len=fft_len,
        cp_len=2,
        sym_len=sym_len,
        null_len=4,
        frame_len=4 + (data_syms + 1) * sym_len,
        data_syms=data_syms,
        n_carriers=n_carriers,
        bin_perm=list(range(fft_len)),
    )


def _compile(path: list[Step], tmp_path: Path) -> None:
    compile_modem(
        Modem(symbol_rate=1000.0, path=path),
        stage_registry(),
        sample_rate=1e6,
        start=IQ,
        source_io={"path": str(tmp_path / "in.cf32")},
        sink_io={"path": str(tmp_path / "out.f32")},
    )


def test_demod_pins_the_frame_its_emit_arithmetic_fixes() -> None:
    stage = stage_registry()["ofdm_demod"]
    for data_syms, n_carriers in ((3, 4), (1, 16), (7, 2)):
        step = _demod(data_syms, n_carriers, fft_len=16)
        out = stage.out_descriptor(IQ, step)
        assert out.frame_len == (data_syms + 1) * n_carriers, (data_syms, n_carriers)


def test_coherent_sync_pins_the_frame_its_emit_arithmetic_fixes() -> None:
    stage = stage_registry()["ofdm_coherent_sync"]
    step = OfdmCoherentSyncStep(**_lattice.sync_params())
    out = stage.out_descriptor(IQ, step)
    assert out.frame_len == step.n_frame_syms * step.n_carriers


def test_a_demap_sized_for_another_frame_is_refused(tmp_path: Path) -> None:
    with pytest.raises(CompileError) as exc:
        _compile(
            [_demod(2, 2), DqpskSoftDemapStep(data_syms=5, n_carriers=7)],
            tmp_path,
        )
    message = str(exc.value)
    assert "42" in message and "6" in message, message


def test_the_matched_demap_compiles(tmp_path: Path) -> None:
    _compile([_demod(2, 2), DqpskSoftDemapStep(data_syms=2, n_carriers=2)], tmp_path)


def test_a_demap_behind_a_cell_gather_is_checked_against_the_gathered_frame(
    tmp_path: Path,
) -> None:
    # cell_select gathers per OFDM symbol here (span 4 = one symbol's carriers,
    # 3 kept), so the frame that reaches the demap is 3 cells per symbol across
    # all (data_syms + 1) symbols — the demap must be sized for THAT.
    gather = CellSelectStep(select_perm=[0, 1, 2, 3], keep=3)
    _compile(
        [
            _demod(2, 4, fft_len=4),
            gather,
            DqpskSoftDemapStep(data_syms=2, n_carriers=3),
        ],
        tmp_path,
    )
    with pytest.raises(CompileError, match="dqpsk_soft_demap"):
        _compile(
            [
                _demod(2, 4, fft_len=4),
                gather,
                DqpskSoftDemapStep(data_syms=2, n_carriers=4),
            ],
            tmp_path,
        )


def test_demap_output_frame_drops_the_reference_symbol() -> None:
    stage = stage_registry()["dqpsk_soft_demap"]
    step = DqpskSoftDemapStep(data_syms=3, n_carriers=8)
    framed = Descriptor(Level.SYMBOLS, ItemType.C, Carrier.SOFT, frame_len=(3 + 1) * 8)
    # 3 of every 4 symbols survive the reference drop, 2 LLRs per cell
    assert stage.out_descriptor(framed, step).frame_len == 3 * 8 * 2


def test_an_unframed_symbol_stream_still_pins_the_demap_geometry() -> None:
    stage = stage_registry()["dqpsk_soft_demap"]
    step = DqpskSoftDemapStep(data_syms=3, n_carriers=8)
    assert stage.out_descriptor(SYM_C, step).frame_len == 3 * 8 * 2
