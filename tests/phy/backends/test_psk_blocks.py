import pytest

from marconi.phy.backends.base import BackendError
from marconi.phy.backends.gnuradio.blocks import GR_BLOCKS, _make_ctx

_KINDS = [
    (
        "rrc_filter_ccf",
        {"interpolation": 1, "rate": 4.0, "sps": 4.0, "alpha": 0.35, "span": 11},
    ),
    ("symbol_sync_cc", {"sps": 4.0, "loop_bw": 0.045}),
    ("costas_loop_cc", {"order": 4, "loop_bw": 0.045}),
    ("chunks_to_symbols_bc", {"scheme": "psk", "order": 4}),
    ("constellation_decoder_cb", {"scheme": "psk", "order": 4}),
    ("pack_k_bits_bb", {"k": 2}),
    ("unpack_k_bits_bb", {"k": 2}),
]


@pytest.mark.parametrize("kind,params", _KINDS)
def test_psk_factory_constructs_a_block(kind: str, params: dict) -> None:
    ctx = _make_ctx(4.0)
    block = GR_BLOCKS[kind](ctx, params)
    assert block is not None


def test_const_rejects_unknown_order() -> None:
    ctx = _make_ctx(4.0)
    with pytest.raises(BackendError):
        GR_BLOCKS["chunks_to_symbols_bc"](ctx, {"scheme": "psk", "order": 5})
