"""burst_probe surfaces burst-start symbol offsets through RunResult
diagnostics: one mark at 0 for a single synced burst; no marks when nothing
upstream emits burst tags."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from marconi.core.bitfile import write_bits
from marconi.core.descriptor import Descriptor
from marconi.core.levels import Level
from marconi.core.params import ParamValue
from marconi.phy.backends.gnuradio.runner import GnuRadioBackend, ensure_worker_warm
from marconi.phy.compiler import compile_modem
from marconi.phy.models import ModemSpec, ModemStep
from marconi.phy.stages import stage_registry

IQ = Descriptor(Level.IQ, "c")
_P: dict[str, ParamValue] = {"sf": 7, "oversample": 2, "zero_pad": 4, "preamble_len": 8}
_SYM = 1.0
_RATE = 2 * (1 << 7) * _SYM


def _compile(m: ModemSpec, direction: str, src: Path, dst: Path):
    return compile_modem(
        m,
        stage_registry(),
        direction=direction,
        sample_rate=_RATE,
        start=IQ,
        source_io={"path": str(src)},
        sink_io={"path": str(dst)},
    )


def _spec(*, probe: bool, sync: bool) -> ModemSpec:
    path = []
    if sync:
        path.append(ModemStep(conv="chirp_sync", params=_P))
    path.append(ModemStep(conv="dechirp", params=_P))
    if probe:
        path.append(ModemStep(conv="burst_probe", params={}))
    path.append(ModemStep(conv="css_demap", params=_P))
    return ModemSpec(symbol_rate=_SYM, path=path)


def _bursts(r) -> list[int] | None:
    for d in r.diagnostics.values():
        b = d.get("bursts")
        if isinstance(b, list):
            return [int(m) for m in b]
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
