from __future__ import annotations

from pydantic_core import PydanticCustomError

from marconi.engine.types.bounds import MAX_FRAME_ITEMS

# The widest legitimate repetition-style gather anyone has needed is a few
# copies per input bit (rate-1/n repetition depuncturing); 8 leaves margin.
_MAX_GATHER_AMPLIFICATION = 8

# Index lists are the one spec shape whose validity a length check cannot
# express, and every stage that takes one used to spell (or forget) its own
# rule: deinterleave and ofdm_demod validated nothing at all, so an
# out-of-range or duplicated index reached the GR constructor inside the
# worker and validate_modem — the "compile without running" surface — reported
# valid. The two rules live here, and test_vocabulary_completeness holds every
# *perm-shaped step field to one of them.


def check_block_permutation(perm: list[int], *, field: str) -> None:
    """A whole-block reorder: exactly one output slot per input slot, so the
    list must be a permutation of 0..len-1. The stock blockinterleaver blocks
    enforce this in C++ ("Expected N unique elements in the range [0, N)") and
    a negative index does not even survive their pybind signature, so a spec
    that skips the check simply fails later, in a worker, as a backend error."""
    if not perm:
        raise PydanticCustomError(
            "value_error",
            "{field} must be a non-empty permutation; an empty block "
            "reorders nothing and the block cannot size its stride",
            {"field": field},
        )
    if sorted(perm) != list(range(len(perm))):
        raise PydanticCustomError(
            "value_error",
            "{field} must be a permutation of 0..{last} (each index exactly "
            "once); the gather walks one whole block per output block, so a "
            "missing, repeated or out-of-range index has no slot to read",
            {"field": field, "last": len(perm) - 1},
        )


def check_gather_indices(perm: list[int], *, field: str) -> None:
    """A gather that MAY drop: output slot i takes input perm[i], the input
    stride is max(perm)+1, and repeats/omissions are legal (that is what makes
    it a depuncturing de-interleave). Only negatives are not: they wrap in
    numpy and would understate the stride."""
    if not perm:
        raise PydanticCustomError(
            "value_error", "{field} must be non-empty", {"field": field}
        )
    if any(i < 0 for i in perm):
        raise PydanticCustomError(
            "value_error",
            "{field} indices must be >= 0; a negative index would wrap and "
            "understate the input stride",
            {"field": field},
        )
    # max(perm)+1 IS the input stride ops_bits._perm_span reshapes by, so the
    # largest index sizes a per-block buffer.
    if max(perm) >= MAX_FRAME_ITEMS:
        raise PydanticCustomError(
            "value_error",
            "{field} index {worst} sets a {stride}-item input stride "
            "(max(perm)+1); {max} is the ceiling",
            {
                "field": field,
                "worst": max(perm),
                "stride": max(perm) + 1,
                "max": MAX_FRAME_ITEMS,
            },
        )
    # The OUTPUT stride needs its own bound: len(perm)/(max+1) is a stream
    # amplifier, and "the list's own length is bounded by the caller having
    # to type it" stopped being true when the caller became an LLM emitting
    # JSON - perm=[0]*16384 validated, amplified 16384x, and wrote 819 MB in
    # 2.5 s (~57 GB at the default timeout) into the workspace.
    if len(perm) > _MAX_GATHER_AMPLIFICATION * (max(perm) + 1):
        raise PydanticCustomError(
            "value_error",
            "{field} emits {out} items per {stride}-item input block - a "
            "{ratio}x stream amplification; {cap}x is the ceiling (a "
            "depuncturing gather repeats indices, it does not multiply the "
            "stream)",
            {
                "field": field,
                "out": len(perm),
                "stride": max(perm) + 1,
                "ratio": len(perm) // (max(perm) + 1),
                "cap": _MAX_GATHER_AMPLIFICATION,
            },
        )
