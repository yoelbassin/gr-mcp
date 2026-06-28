import numpy as np
import pytest

from marconi.phy.backends.gnuradio.blocks import GR_BLOCKS, _make_ctx

_PRE = np.exp(1j * np.linspace(-3.0, 3.0, 64)).astype(np.complex64)
_PARAMS = {
    "preamble_i": _PRE.real.tolist(),
    "preamble_q": _PRE.imag.tolist(),
    "pad_symbols": 192,
    "threshold": 3.0,
}


@pytest.mark.parametrize("kind", ["sym_prepend", "sym_acquire"])
def test_acquisition_factory_constructs_a_block(kind: str) -> None:
    ctx = _make_ctx(4.0)
    assert GR_BLOCKS[kind](ctx, _PARAMS) is not None
