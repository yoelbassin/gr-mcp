from pathlib import Path

import numpy as np
import pytest
from phy._dsp import aligned_ber, read_bits, write_bits

from marconi.core.descriptor import Descriptor
from marconi.core.levels import Level
from marconi.phy.backends.gnuradio.runner import GnuRadioBackend, ensure_worker_warm
from marconi.phy.compiler import compile_modem
from marconi.phy.ir import GrPipeline
from marconi.phy.models import ModemSpec, ModemStep
from marconi.phy.stages import stage_registry

IQ = Descriptor(Level.IQ, "c")
_SR, _SYM = 8.0, 1.0
_GAINS = [1e-3, 1e-1, 1.0, 1e1, 1e3]

_CHAINS = {
    "fsk": [
        ModemStep(conv="fsk", params={"deviation": 1.0}),
        ModemStep(conv="slice"),
    ],
    "psk": [
        ModemStep(conv="psk_demod", params={"order": 2}),
        ModemStep(conv="psk_demap", params={"order": 2}),
    ],
}


def _ber_at_gain(tmp_path: Path, name: str, gain: float) -> float:
    be = GnuRadioBackend()
    spec = ModemSpec(symbol_rate=_SYM, path=_CHAINS[name])
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
    # A non-ok RX run (e.g. a hung loop) never decodes anything: worst-case BER.
    if rx.status != "ok":
        return 1.0
    return aligned_ber(read_bits(out), bits)


def _assert_level_invariant(name: str, table: dict[float, float]) -> None:
    assert all(b == 0.0 for b in table.values()), f"{name} is level-sensitive: {table}"


def _assert_loop_gain_sensitive(name: str, table: dict[float, float]) -> None:
    assert table[1.0] == 0.0, f"{name} baseline is not BER-0: {table}"
    assert table[1e-3] > 0.01, f"{name} pattern changed: {table}"
    assert table[1e1] > 0.3, f"{name} pattern changed: {table}"
    assert table[1e3] > 0.3, f"{name} pattern changed: {table}"


_EXPECT = {
    "fsk": _assert_level_invariant,
    "psk": _assert_loop_gain_sensitive,
}


@pytest.mark.parametrize("name", sorted(_CHAINS))
def test_v2_level_sensitivity(name: str, tmp_path: Path) -> None:
    ensure_worker_warm()
    table = {g: _ber_at_gain(tmp_path, name, g) for g in _GAINS}
    print(f"\nV2 {name} BER vs gain -> {table}")
    _EXPECT[name](name, table)
