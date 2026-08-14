"""Per-sample rates DERIVED from (symbols x sps) at emit time: the step model
alone cannot bound them because b.sps is a compile-time fact. Unclamped, an
IIR pole above 1 turned agc(power) into a NaN factory (measured: 50% non-
finite under status ok at alpha 2), and symbol_sync amplified that NaN
stream to 3.4 GB before the deadline fired. squelch's alpha reached GR as a
backend construction error naming a block id. Both must fail the compile,
naming the parameter and its floor."""

from __future__ import annotations

import pytest

from marconi.engine.compile.compile_context import CompileContext
from marconi.engine.compile.errors import CompileError
from marconi.engine.stages.conditioning import Agc, AgcStep, Squelch, SquelchStep
from marconi.engine.types.descriptor import Descriptor
from marconi.engine.types.enums import AgcMode, ItemType
from marconi.engine.types.levels import Level


def _ctx(sps: float) -> CompileContext:
    b = CompileContext(
        Descriptor(Level.IQ, ItemType.C), rate=sps * 1000.0, symbol_rate=1000.0
    )
    b.chain("null_source_c")  # the power AGC branches, so it needs a tail
    return b


def test_agc_power_pole_above_one_fails_the_compile() -> None:
    step = AgcStep(conv="agc", mode=AgcMode.POWER, window_symbols=0.1)
    with pytest.raises(CompileError, match="window_symbols"):
        Agc().emit_rx(_ctx(2.0), step)


def test_agc_power_pole_at_one_sample_is_legal() -> None:
    step = AgcStep(conv="agc", mode=AgcMode.POWER, window_symbols=0.5)
    Agc().emit_rx(_ctx(2.0), step)  # alpha exactly 1.0: a one-sample window


def test_agc_feedback_rates_above_one_fail_the_compile() -> None:
    step = AgcStep(
        conv="agc", mode=AgcMode.FEEDBACK, attack_symbols=0.01, decay_symbols=16.0
    )
    with pytest.raises(CompileError, match="attack_symbols"):
        Agc().emit_rx(_ctx(2.0), step)


def test_squelch_alpha_above_one_fails_the_compile_not_the_backend() -> None:
    # GR rejects alpha > 1 itself, but as "block 'pwr_squelch_cc_0' failed to
    # construct" from inside the worker - a spec error wearing a backend cause
    step = SquelchStep(conv="squelch", threshold_db=-20.0, alpha_symbols=0.25)
    with pytest.raises(CompileError, match="alpha_symbols"):
        Squelch().emit_rx(_ctx(2.0), step)
