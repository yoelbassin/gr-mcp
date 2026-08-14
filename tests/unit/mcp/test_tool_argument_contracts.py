"""Argument contracts at the MCP surface. Every one of these was a place where
the same mistake was told two different ways, or not told at all: an out-of-
range value silently accepted on one parameter and raised on its sibling, a
destructive write with no guard while every cost-sizing param in the tree is
bounded, a caller number surfacing as an engine bug.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest

from marconi.errors import classify_error
from marconi.mcp.streams import EmptyStreamError
from marconi.mcp.tools import (
    describe_stages,
    read_stream,
    stream_stats,
    survey,
    validate_modem,
)

_SPEC = {"symbol_rate": 1000.0, "path": [{"conv": "agc"}]}


@pytest.mark.parametrize("rate", [0.0, -48000.0, math.nan, math.inf])
def test_a_bad_sample_rate_is_the_callers_to_fix(rate: float) -> None:
    """Modem.symbol_rate is Field(gt=0); the quantity every rate check pairs it
    with had nothing at any tool entry but capture. inf was the sharp one — it
    reached round() inside the compiler and surfaced as [internal_error],
    "stop and report a bug", for a number the caller could have retyped."""
    result = validate_modem(_SPEC, sample_rate=rate)
    assert result["valid"] is False
    errors = result["errors"]
    assert isinstance(errors, list)
    assert errors[0]["code"] == "invalid_argument"
    assert "sample_rate" in str(errors[0]["message"])


def test_survey_names_sample_rate_rather_than_the_bounds_it_derived(
    tmp_path: Path,
) -> None:
    """The failure surfaced as "require 0 < min_symbol_rate < max_symbol_rate"
    — two parameters the caller had not passed, about a value that was not the
    problem."""
    cap = tmp_path / "c.cf32"
    rng = np.random.default_rng(0)
    n = 4096
    (rng.standard_normal(n) + 1j * rng.standard_normal(n)).astype(np.complex64).tofile(
        cap
    )
    with pytest.raises(ValueError, match="sample_rate"):
        survey(str(cap), sample_rate=0.0)


def test_read_stream_rejects_a_negative_count_like_it_rejects_an_offset(
    tmp_path: Path,
) -> None:
    p = tmp_path / "bits.u8"
    np.zeros(64, dtype=np.uint8).tofile(p)
    with pytest.raises(ValueError, match="offset"):
        read_stream(str(p), offset=-1)
    # ... and count<0 silently returned an empty page: the same mistake, told
    # two different ways
    with pytest.raises(ValueError, match="count"):
        read_stream(str(p), count=-1)


def test_stream_stats_on_an_empty_run_output_does_not_blame_the_call(
    tmp_path: Path,
) -> None:
    """The stream exists and the arguments were fine; a run decoded nothing.
    [invalid_argument] sent the agent back to re-check a correct call."""
    p = tmp_path / "out.f32"
    p.write_bytes(b"")
    with pytest.raises(EmptyStreamError) as exc:
        stream_stats(str(p))
    assert classify_error(exc.value)[0] == "failed_precondition"


def test_describe_stages_refuses_two_selectors_instead_of_dropping_one() -> None:
    with pytest.raises(ValueError, match="not both"):
        describe_stages(stage="agc", family="psk")


def test_malformed_step_entries_name_the_index_and_shape() -> None:
    # path=[null] raised TypeError past the boundary's net and surfaced as
    # "'NoneType' object is not iterable" - no index, no parameter, no shape
    import pytest

    from marconi.engine.stages.registry import step_models
    from marconi.engine.types.step import StepSpecError, steps_from_spec

    for bad in (None, 1, True, "agc"):
        with pytest.raises(StepSpecError, match=r"path\[0\].*object"):
            steps_from_spec([bad], step_models())


def test_gather_amplification_is_bounded() -> None:
    # perm=[0]*16384 validated and wrote 819 MB in 2.5 s (~57 GB at the
    # default timeout): the output stride needs a bound of its own, because
    # "the caller has to type it" stopped being true when the caller became
    # an LLM emitting JSON
    import pytest
    from pydantic import ValidationError

    from marconi.engine.coding.stages_bits import PermuteStep

    with pytest.raises(ValidationError, match="amplification"):
        PermuteStep(conv="permute", perm=[0] * 16384)
    # repetition-style depuncturing stays legal
    assert PermuteStep(conv="permute", perm=[0, 0, 1, 1]).perm == [0, 0, 1, 1]


def test_sync_pattern_length_is_bounded() -> None:
    # a 65,536-bit pattern validated and ran 14.7 s (one full stream pass
    # per pattern bit); the bounds sweep had no str arm to catch it
    import pytest
    from pydantic import ValidationError

    from marconi.engine.coding.stages_bits import SyncWordStep

    with pytest.raises(ValidationError, match="ceiling"):
        SyncWordStep(conv="sync_word", bits="01" * 40_000)
    assert SyncWordStep(conv="sync_word", bits="0110" * 16)
