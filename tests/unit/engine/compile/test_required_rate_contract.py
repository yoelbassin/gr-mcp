"""The required-input-rate gate must not certify a rate it could not compute.

The gate's whole job is to refuse a spec whose pipeline rate does not match
what a stage needs. Its comparison is a RELATIVE tolerance — `abs(rate -
required) <= tol * required` — which reads `inf <= inf` as a match, so the one
value that makes the requirement unrepresentable is also the one value the gate
waves through.
"""

from __future__ import annotations

import pytest

from marconi.engine.compile.compiler import compile_modem
from marconi.engine.compile.errors import CompileError
from marconi.engine.modulation.css.stages import DechirpStep
from marconi.engine.stages.registry import stage_registry
from marconi.engine.types.descriptor import Descriptor
from marconi.engine.types.enums import ItemType
from marconi.engine.types.levels import Level
from marconi.engine.types.models import Modem

IQ = Descriptor(Level.IQ, ItemType.C)
# dechirp needs oversample * 2**sf samples per symbol, so its requirement is
# symbol_rate scaled by 2**17 - the largest multiplier any stage applies.
_WIDE = DechirpStep(sf=14, oversample=8, zero_pad=1)


def _compile(symbol_rate: float) -> None:
    compile_modem(
        Modem(symbol_rate=symbol_rate, path=[_WIDE]),
        stage_registry(),
        sample_rate=48_000.0,
        start=IQ,
        source_io={"path": "/dev/null"},
        sink_io={"path": "/dev/null"},
    )


def test_a_mismatched_rate_below_the_overflow_is_still_refused() -> None:
    """The ordinary path the gate exists for, kept as the control: at 1e300 the
    requirement is representable (1.31e305) and the mismatch is reported."""
    with pytest.raises(CompileError, match="requires input sample rate"):
        _compile(1e300)


@pytest.mark.parametrize("symbol_rate", [1e305, 1e308])
def test_a_requirement_too_large_to_represent_is_refused_not_waved_through(
    symbol_rate: float,
) -> None:
    """symbol_rate here is FINITE — the spec-level finiteness guard on Modem
    does not see it — but 2**17 times it overflows to inf, and the tolerance
    comparison then reads `abs(48000 - inf) <= tol * inf` as `inf <= inf`.
    Measured through the real validate_modem: symbol_rate 1e305 with this
    dechirp returned {"valid": true}, relabeling a 48 kHz IQ capture as a
    1e305-baud chirp stream."""
    with pytest.raises(CompileError, match="symbol_rate"):
        _compile(symbol_rate)
