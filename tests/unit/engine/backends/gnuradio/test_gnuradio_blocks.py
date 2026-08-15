import pytest

from marconi.engine.backends.base import BackendError
from marconi.engine.backends.gnuradio.blocks import (
    GR_BLOCKS,
    BlockParams,
    _factories,
    _modules,
)


def test_unknown_kind_absent() -> None:
    assert "no_such_block" not in _factories()


def test_rrc_filter_accepts_fractional_sps() -> None:
    """A capture whose sample rate is not an integer multiple of the baud has
    fractional sps; the RRC matched filter's tap count must round it, not force
    an int and raise in the backend (sample_rate=10, symbol_rate=3 -> sps=3.33)."""
    ctx = _modules()
    block = GR_BLOCKS["rrc_filter_ccf"](
        ctx,
        BlockParams(
            {
                "interpolation": 1,
                "rate": 10.0,
                "sps": 10.0 / 3.0,
                "alpha": 0.35,
                "span": 11,
            }
        ),
    )
    assert block is not None


def test_const_rejects_unknown_order() -> None:
    ctx = _modules()
    with pytest.raises(BackendError):
        GR_BLOCKS["constellation_decoder_cb"](
            ctx, BlockParams({"scheme": "psk", "order": 5})
        )


def test_qam_const_rejects_unknown_order() -> None:
    ctx = _modules()
    with pytest.raises(BackendError):
        GR_BLOCKS["constellation_decoder_cb"](
            ctx, BlockParams({"scheme": "qam", "order": 32})
        )


def test_ctx_exposes_trellis() -> None:
    ctx = _modules()
    assert hasattr(ctx.trellis, "fsm") and hasattr(ctx.trellis, "viterbi_combined_fb")


def test_preamble_scale_sensitivity_is_a_documented_limitation() -> None:
    """THRESHOLD_ABSOLUTE compares against the ideal preamble's absolute
    energy, so detection depends on input scale: measured 0 detections at
    gain 0.1, exactly 1 (correct) at 1.0, and a flood at gain 10. Pinned so
    a threshold-method change gets a signal: THRESHOLD_DYNAMIC is scale-
    invariant but shifts the tag phase/timing calibration the strip lane is
    built on (measured: mis-derotated payloads, 128+ symbols lost), so the
    shipped contract is normalize-first - see the corr_est_cc factory note."""
    import numpy as np

    ctx = _modules()
    rng = np.random.default_rng(0)
    pre = np.exp(2j * np.pi * rng.random(32)).astype(np.complex64)
    payload = np.exp(2j * np.pi * rng.random(4000)).astype(np.complex64)
    payload[64 : 64 + 32] = pre
    counts = []
    for gain in (0.1, 1.0, 10.0):
        blk = GR_BLOCKS["corr_est_cc"](
            ctx,
            BlockParams(
                {
                    "preamble_i": pre.real.astype(float).tolist(),
                    "preamble_q": pre.imag.astype(float).tolist(),
                    "sps": 1,
                    "mark_delay": 0,
                    "threshold": 0.9,
                }
            ),
        )
        gr, blocks = ctx.gr, ctx.blocks
        tb = gr.top_block()
        src = blocks.vector_source_c((payload * gain).tolist(), False)
        dbg = blocks.tag_debug(gr.sizeof_gr_complex, "p")
        dbg.set_display(False)
        dbg.set_save_all(True)
        tb.connect(src, blk, dbg)
        tb.run()
        counts.append(sum(1 for t in dbg.current_tags() if "corr_est" in str(t.key)))
    assert counts[1] == 1, counts  # unit scale: exactly the one real preamble
    assert counts[0] == 0, counts  # under-scale: blind (the documented cost)
    assert counts[2] > 10, counts  # over-scale: floods (the documented cost)


def test_correlator_rejects_a_truncated_preamble_axis() -> None:
    """The IR-direct path skips the pydantic equal-length guard, and zip()
    truncates: an 8-sample preamble_i against a 4-sample preamble_q built a
    4-symbol correlator that GR then re-calibrated ("mark delay set to 3"),
    so the detection tag landed at the wrong phase instead of failing."""
    import numpy as np

    ctx = _modules()
    rng = np.random.default_rng(4)
    pre = np.exp(2j * np.pi * rng.random(8))
    with pytest.raises(BackendError, match="preamble"):
        GR_BLOCKS["corr_est_cc"](
            ctx,
            BlockParams(
                {
                    "preamble_i": pre.real.astype(float).tolist(),
                    "preamble_q": pre.imag.astype(float).tolist()[:4],
                    "sps": 1,
                    "mark_delay": 4,
                    "threshold": 0.9,
                }
            ),
        )
