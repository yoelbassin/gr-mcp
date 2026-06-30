def test_ctx_exposes_trellis():
    from marconi.phy.backends.gnuradio.blocks import _make_ctx

    ctx = _make_ctx(1.0)
    assert hasattr(ctx.trellis, "fsm") and hasattr(ctx.trellis, "viterbi_combined_fb")
