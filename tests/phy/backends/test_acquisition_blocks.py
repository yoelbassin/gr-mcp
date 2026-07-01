import numpy as np
import pytest

from marconi.phy.backends.gnuradio.blocks import GR_BLOCKS, _make_ctx
from marconi.phy.backends.gnuradio.embedded.preamble import _acquire_lock, _ramp

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


def test_acquire_lock_ignores_strong_payload_replica() -> None:
    # pad=192, preamble n=64 -> min_buf=320. A 2x-amplitude replica of the
    # preamble sits in the payload at lag 300 (beyond the acquisition window).
    # The unbounded argmax steals the lock onto it; the windowed search must
    # lock the true preamble at lag 192.
    n, pad, min_buf = 64, 192, 320
    pre = np.exp(1j * np.linspace(-3.0, 3.0, n)).astype(np.complex128)
    payload = np.zeros(256, dtype=np.complex128)
    payload[300 - (pad + n) : 300 - (pad + n) + n] = 2.0 * pre
    seg = np.concatenate([_ramp(pad).astype(np.complex128), pre, payload])
    lock = _acquire_lock(seg, pre, min_buf, threshold=3.0)
    assert lock is not None
    k, phase = lock
    assert k == pad
    assert abs(phase) < 1e-6
