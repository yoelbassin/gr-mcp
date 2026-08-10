import pytest

from marconi.engine.backends.base import BackendError
from marconi.engine.backends.gnuradio.blocks import GR_BLOCKS, _factories, _modules


def test_unknown_kind_absent() -> None:
    assert "no_such_block" not in _factories()


def test_rrc_filter_accepts_fractional_sps() -> None:
    """A capture whose sample rate is not an integer multiple of the baud has
    fractional sps; the RRC matched filter's tap count must round it, not force
    an int and raise in the backend (sample_rate=10, symbol_rate=3 -> sps=3.33)."""
    ctx = _modules()
    block = GR_BLOCKS["rrc_filter_ccf"](
        ctx,
        {
            "interpolation": 1,
            "rate": 10.0,
            "sps": 10.0 / 3.0,
            "alpha": 0.35,
            "span": 11,
        },
    )
    assert block is not None


def test_const_rejects_unknown_order() -> None:
    ctx = _modules()
    with pytest.raises(BackendError):
        GR_BLOCKS["chunks_to_symbols_bc"](ctx, {"scheme": "psk", "order": 5})


def test_qam_const_rejects_unknown_order() -> None:
    ctx = _modules()
    with pytest.raises(BackendError):
        GR_BLOCKS["chunks_to_symbols_bc"](ctx, {"scheme": "qam", "order": 32})


def test_ctx_exposes_trellis() -> None:
    ctx = _modules()
    assert hasattr(ctx.trellis, "fsm") and hasattr(ctx.trellis, "viterbi_combined_fb")
