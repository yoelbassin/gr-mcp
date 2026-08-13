from __future__ import annotations

from pydantic_core import PydanticCustomError

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
