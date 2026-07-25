import pytest

from marconi.engine.backends.base import BackendError
from marconi.engine.backends.gnuradio.blocks import GR_BLOCKS, _make_ctx

_KINDS = [
    ("constellation_receiver_cb", {"scheme": "qam", "order": 16}),
    ("constellation_receiver_cb", {"scheme": "qam", "order": 64}),
    ("chunks_to_symbols_bc", {"scheme": "qam", "order": 16}),
    ("chunks_to_symbols_bc", {"scheme": "qam", "order": 64}),
]


@pytest.mark.parametrize("kind,params", _KINDS)
def test_qam_factory_constructs_a_block(kind: str, params: dict) -> None:
    ctx = _make_ctx(4.0)
    assert GR_BLOCKS[kind](ctx, params) is not None


def test_qam_const_rejects_unknown_order() -> None:
    ctx = _make_ctx(4.0)
    with pytest.raises(BackendError):
        GR_BLOCKS["chunks_to_symbols_bc"](ctx, {"scheme": "qam", "order": 32})
