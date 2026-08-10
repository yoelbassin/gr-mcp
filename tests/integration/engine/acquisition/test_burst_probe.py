"""burst_probe surfaces burst-start symbol offsets through RunResult
diagnostics: one mark at 0 for a single synced burst; no marks when nothing
upstream emits burst tags."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import numpy as np

from marconi.engine.backends.base import RunResult
from marconi.engine.backends.gnuradio.runner import GnuRadioBackend, ensure_worker_warm
from marconi.engine.compile.compiler import compile_modem
from marconi.engine.compile.ir import GrPipeline
from marconi.engine.io.bitfile import write_bits
from marconi.engine.modulation.css.stages import (
    ChirpSyncStep,
    CssDemapStep,
    DechirpStep,
)
from marconi.engine.stages.probes import BurstProbeStep
from marconi.engine.stages.registry import stage_registry
from marconi.engine.types.descriptor import Descriptor
from marconi.engine.types.enums import ItemType
from marconi.engine.types.levels import Level
from marconi.engine.types.models import Modem
from marconi.engine.types.step import Step

IQ = Descriptor(Level.IQ, ItemType.C)
_SF, _OVERSAMPLE, _ZERO_PAD = 7, 2, 4
_PREAMBLE_LEN, _SFD_SYMBOLS, _SYNC_SYMBOLS = 8, 2.25, 2
_SYM = 1.0
_RATE = 2 * (1 << _SF) * _SYM


def _compile(
    m: Modem, direction: Literal["rx", "tx"], src: Path, dst: Path
) -> GrPipeline:
    return compile_modem(
        m,
        stage_registry(),
        direction=direction,
        sample_rate=_RATE,
        start=IQ,
        source_io={"path": str(src)},
        sink_io={"path": str(dst)},
    )


def _spec(*, probe: bool, sync: bool) -> Modem:
    path: list[Step] = []
    if sync:
        path.append(
            ChirpSyncStep(
                sf=_SF,
                oversample=_OVERSAMPLE,
                zero_pad=_ZERO_PAD,
                preamble_len=_PREAMBLE_LEN,
                sfd_symbols=_SFD_SYMBOLS,
                sync_symbols=_SYNC_SYMBOLS,
            )
        )
    path.append(DechirpStep(sf=_SF, oversample=_OVERSAMPLE, zero_pad=_ZERO_PAD))
    if probe:
        path.append(BurstProbeStep())
    path.append(CssDemapStep(sf=_SF))
    return Modem(symbol_rate=_SYM, path=path)


def _bursts(r: RunResult) -> list[int] | None:
    for d in r.diagnostics:
        if d.key == "bursts" and d.marks is not None:
            return [int(m) for m in d.marks]
    return None


def test_synced_burst_marks_offset_zero(tmp_path: Path) -> None:
    ensure_worker_warm()
    be = GnuRadioBackend()
    rng = np.random.default_rng(5)
    bits = rng.integers(0, 2, 7 * 30).astype(np.uint8)
    pad = rng.integers(0, 2, 7 * 12).astype(np.uint8)
    src = tmp_path / "in.bits"
    write_bits(src, np.concatenate([bits, pad]))
    iq = tmp_path / "tx.iq"
    assert (
        be.run_pipeline(_compile(_spec(probe=False, sync=True), "tx", src, iq)).status
        == "ok"
    )
    out = tmp_path / "out.bits"
    r = be.run_pipeline(_compile(_spec(probe=True, sync=True), "rx", iq, out))
    assert r.status == "ok"
    assert _bursts(r) == [0], r.diagnostics


def test_untagged_stream_yields_no_marks(tmp_path: Path) -> None:
    ensure_worker_warm()
    be = GnuRadioBackend()
    bits = np.random.default_rng(6).integers(0, 2, 7 * 20).astype(np.uint8)
    src = tmp_path / "in.bits"
    write_bits(src, bits)
    iq = tmp_path / "tx.iq"
    assert (
        be.run_pipeline(_compile(_spec(probe=False, sync=False), "tx", src, iq)).status
        == "ok"
    )
    out = tmp_path / "out.bits"
    r = be.run_pipeline(_compile(_spec(probe=True, sync=False), "rx", iq, out))
    assert r.status == "ok"
    assert _bursts(r) == [], r.diagnostics
