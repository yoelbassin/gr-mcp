import pytest

from marconi.phy.backends.gnuradio.blocks import GR_BLOCKS, _make_ctx

_KINDS = [
    (
        "freq_xlating_fir_filter_ccf",
        {"decim": 2, "center": 4.0, "cutoff": 2.0, "transition": 2.0, "rate": 16.0},
    ),
    ("conjugate_cc", {}),
]


@pytest.mark.parametrize("kind,params", _KINDS)
def test_conditioning_factory_constructs_a_block(kind: str, params: dict) -> None:
    ctx = _make_ctx(16.0)
    assert GR_BLOCKS[kind](ctx, params) is not None
