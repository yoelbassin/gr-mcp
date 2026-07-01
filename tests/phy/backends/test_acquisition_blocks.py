import numpy as np
import pytest

from marconi.phy.backends.gnuradio.blocks import GR_BLOCKS, _make_ctx

_PRE = np.exp(1j * np.linspace(-3.0, 3.0, 64)).astype(np.complex64)
_PI, _PQ = _PRE.real.tolist(), _PRE.imag.tolist()
_CASES = {
    "sym_prepend": {"preamble_i": _PI, "preamble_q": _PQ, "pad_symbols": 192},
    "corr_est_cc": {
        "preamble_i": _PI,
        "preamble_q": _PQ,
        "sps": 1,
        "mark_delay": 0,
        "threshold": 0.9,
    },
    "sym_strip": {"n_pre": 64},
}


@pytest.mark.parametrize("kind", list(_CASES))
def test_acquisition_factory_constructs_a_block(kind: str) -> None:
    ctx = _make_ctx(4.0)
    assert GR_BLOCKS[kind](ctx, _CASES[kind]) is not None
