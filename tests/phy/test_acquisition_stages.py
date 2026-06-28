import numpy as np

from marconi.core.descriptor import Carrier, Descriptor
from marconi.core.levels import Level
from marconi.core.stages import validate_params
from marconi.phy.compile_context import CompileContext
from marconi.phy.stages import stage_registry
from marconi.phy.stages.acquisition import PreambleSync

_PRE = np.exp(1j * np.linspace(-3.0, 3.0, 64))
_P = {
    "preamble_i": _PRE.real.tolist(),
    "preamble_q": _PRE.imag.tolist(),
    "pad_symbols": 192,
    "threshold": 3.0,
}
SYM_C = Descriptor(Level.SYMBOLS, "c", carrier=Carrier.SOFT)


def test_registered_as_acquisition() -> None:
    assert stage_registry()["preamble_sync"].family == "acquisition"


def test_symbols_to_symbols_soft_unity_rate() -> None:
    s = PreambleSync()
    assert s.out_descriptor(SYM_C, _P) == SYM_C
    assert s.rate_factor(_P) == 1.0


def test_emit_rx_chains_sym_acquire_with_params() -> None:
    b = CompileContext(SYM_C, rate=4.0, symbol_rate=1.0)
    PreambleSync().emit_rx(b, _P)
    blk = next(x for x in b.build("t", 4.0).blocks if x.kind == "sym_acquire")
    assert blk.params["pad_symbols"] == 192
    assert blk.params["threshold"] == 3.0
    pi, pq = blk.params["preamble_i"], blk.params["preamble_q"]
    assert isinstance(pi, list) and isinstance(pq, list)
    assert len(pi) == 64 and len(pq) == 64


def test_emit_tx_chains_sym_prepend() -> None:
    b = CompileContext(SYM_C, rate=4.0, symbol_rate=1.0)
    PreambleSync().emit_tx(b, _P)
    assert [x.kind for x in b.build("t", 4.0).blocks] == ["sym_prepend"]


def test_param_validation() -> None:
    bad: list = []
    validate_params(
        "preamble_sync[0]",
        PreambleSync().params_model,
        {"preamble_i": [1.0, 0.0], "preamble_q": [0.0]},
        bad,
    )
    assert bad  # unequal length
    bad2: list = []
    validate_params(
        "preamble_sync[0]",
        PreambleSync().params_model,
        {"preamble_i": [], "preamble_q": []},
        bad2,
    )
    assert bad2  # empty preamble
