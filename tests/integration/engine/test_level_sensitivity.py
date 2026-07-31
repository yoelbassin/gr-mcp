from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
import pytest
from helpers._dsp import aligned_ber, read_bits, write_bits

from marconi.engine.backends.gnuradio.runner import GnuRadioBackend, ensure_worker_warm
from marconi.engine.compile.compiler import CompileError, compile_modem
from marconi.engine.compile.ir import GrPipeline
from marconi.engine.modulation.fsk.stages import FskStep
from marconi.engine.modulation.psk.stages import PskDemapStep, PskDemodStep
from marconi.engine.stages.general import SliceStep
from marconi.engine.stages.registry import stage_registry
from marconi.engine.types.descriptor import Descriptor
from marconi.engine.types.enums import ItemType, PskOrder
from marconi.engine.types.levels import Level
from marconi.engine.types.models import Modem
from marconi.engine.types.step import Step

# Mirrors RunResult.status in marconi.engine.backends.base; base.py has `from
# __future__ import annotations`, so its Literal is stored unevaluated (a
# string) and isn't reachable as a reusable static alias without a source
# change -- this duplication is intentional and test-side only.
_Status = Literal["ok", "error", "timeout", "empty"]


@dataclass(frozen=True)
class GainResult:
    status: _Status
    ber: float


IQ = Descriptor(Level.IQ, ItemType.C)
_SR, _SYM = 8.0, 1.0
_GAINS = [1e-3, 1e-1, 1.0, 1e1, 1e3]

_CHAINS: dict[str, list[Step]] = {
    "fsk": [
        FskStep(deviation=1.0),
        SliceStep(),
    ],
    "psk": [
        PskDemodStep(order=PskOrder(2)),
        PskDemapStep(order=PskOrder(2)),
    ],
}


def _ber_at_gain(tmp_path: Path, name: str, gain: float) -> GainResult:
    be = GnuRadioBackend()
    spec = Modem(symbol_rate=_SYM, path=_CHAINS[name])
    bits = np.random.default_rng(0).integers(0, 2, 4096).astype(np.uint8)
    bp = write_bits(tmp_path / f"{name}_in.bits", bits)
    iq = tmp_path / f"{name}.iq"
    scaled = tmp_path / f"{name}_{gain}.iq"
    out = tmp_path / f"{name}_{gain}.bits"

    def _compile(direction: str, src: Path, snk: Path) -> GrPipeline:
        return compile_modem(
            spec,
            stage_registry(),
            direction=direction,
            sample_rate=_SR,
            start=IQ,
            source_io={"path": str(src)},
            sink_io={"path": str(snk)},
        )

    assert be.run_pipeline(_compile("tx", bp, iq)).status == "ok"
    (np.fromfile(iq, dtype=np.complex64) * np.complex64(gain)).tofile(scaled)
    rx = be.run_pipeline(_compile("rx", scaled, out))
    # A non-ok RX run (e.g. a hung loop) never decodes anything: worst-case BER,
    # but the status itself is kept so a crash is never mistaken for a decode.
    if rx.status != "ok":
        return GainResult(status=rx.status, ber=1.0)
    return GainResult(status="ok", ber=aligned_ber(read_bits(out), bits))


def _assert_level_invariant(name: str, table: dict[float, GainResult]) -> None:
    for gain, result in table.items():
        assert result.status == "ok", f"{name}@{gain} status {result.status}: {table}"
    assert all(
        r.ber == 0.0 for r in table.values()
    ), f"{name} is level-sensitive: {table}"


_EXPECT: dict[str, Callable[[str, dict[float, GainResult]], None]] = {
    "fsk": _assert_level_invariant,
}


@pytest.mark.parametrize("name", sorted(_EXPECT))
def test_v2_level_sensitivity(name: str, tmp_path: Path) -> None:
    ensure_worker_warm()
    table = {g: _ber_at_gain(tmp_path, name, g) for g in _GAINS}
    print(f"\nV2 {name} BER vs gain -> {table}")
    _EXPECT[name](name, table)


def test_psk_without_agc_rejected_at_compile() -> None:
    with pytest.raises(CompileError):
        compile_modem(
            Modem(symbol_rate=_SYM, path=_CHAINS["psk"]),
            stage_registry(),
            direction="rx",
            sample_rate=_SR,
            start=IQ,
            source_io={"path": "/dev/null"},
            sink_io={"path": "/dev/null"},
        )
