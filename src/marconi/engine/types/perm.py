from __future__ import annotations

from typing import Annotated

from pydantic import BeforeValidator
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


# The compact forms an index list may be written in, expanded before any
# validator sees it. Bounded by MAX_FRAME_ITEMS like the literal lists they
# stand in for, so a compact form cannot ask for an array a literal could not.
_MAX_EXPAND_DEPTH = 8


def _fail(msg: str, ctx: dict[str, object] | None = None) -> None:
    raise PydanticCustomError("value_error", msg, ctx or {})


def _expand(node: object, depth: int) -> list[int]:
    if depth > _MAX_EXPAND_DEPTH:
        _fail(
            "compact index list nests deeper than {max} levels",
            {"max": _MAX_EXPAND_DEPTH},
        )
    if isinstance(node, list):
        out: list[int] = []
        for item in node:
            if isinstance(item, bool) or not isinstance(item, int):
                _fail("a literal index list holds ints; got {bad!r}", {"bad": item})
            out.append(int(item))
        return out
    if not isinstance(node, dict):
        _fail(
            "an index list is a list of ints or one of the compact forms "
            "{{'range': [start, stop, step]}}, {{'repeat': <list>, "
            "'times': n}}, {{'concat': [<list>, ...]}}; got {bad!r}",
            {"bad": node},
        )
        return []
    keys = set(node)
    if keys == {"range"}:
        spec = node["range"]
        if not isinstance(spec, list) or not 2 <= len(spec) <= 3:
            _fail("'range' takes [start, stop] or [start, stop, step]")
        if any(isinstance(v, bool) or not isinstance(v, int) for v in spec):
            _fail("'range' bounds must be ints")
        start, stop = int(spec[0]), int(spec[1])
        step = int(spec[2]) if len(spec) == 3 else 1
        if step == 0:
            _fail("'range' step must not be 0")
        size = max(0, -(-(stop - start) // step)) if step else 0
        if size > MAX_FRAME_ITEMS:
            _fail(
                "'range' expands to {n} items; {cap} is the ceiling",
                {"n": size, "cap": MAX_FRAME_ITEMS},
            )
        return list(range(start, stop, step))
    if keys == {"repeat", "times"}:
        times = node["times"]
        if isinstance(times, bool) or not isinstance(times, int) or times < 0:
            _fail("'times' must be an int >= 0; got {bad!r}", {"bad": times})
        inner = _expand(node["repeat"], depth + 1)
        if len(inner) * int(times) > MAX_FRAME_ITEMS:
            _fail(
                "'repeat' expands to {n} items; {cap} is the ceiling",
                {"n": len(inner) * int(times), "cap": MAX_FRAME_ITEMS},
            )
        return inner * int(times)
    if keys == {"concat"}:
        parts = node["concat"]
        if not isinstance(parts, list):
            _fail("'concat' takes a list of index lists")
            return []
        out = []
        for part in parts:
            out.extend(_expand(part, depth + 1))
            if len(out) > MAX_FRAME_ITEMS:
                _fail("'concat' expands past {cap} items", {"cap": MAX_FRAME_ITEMS})
        return out
    _fail(
        "unknown compact index form {keys}; expected exactly one of "
        "'range', 'repeat'+'times', or 'concat'",
        {"keys": sorted(keys)},
    )
    return []


def expand_index_list(value: object) -> object:
    """Expand the compact index-list forms into a plain list of ints.

    Index lists are the largest thing a spec carries and the agent has to TYPE
    them: one entry per FFT bin for an OFDM bin_perm, one per block position
    for a de-interleave, one per mother-code bit for a depuncture mask —
    measured at ~8200 literals for a single real multicarrier spec. That is
    both the dominant token cost of the call and a transcription hazard: a
    3096-entry mask was miscounted twice, in opposite directions, and only the
    frame-multiple check caught it.

    Almost none of that length is information. A puncturing mask is one vector
    repeated; a de-interleave is strided ranges. So a list may also be written
    as {"range": [start, stop, step]}, {"repeat": <list>, "times": n}, or
    {"concat": [<list>, ...]}, nested freely — the same 3096-entry mask
    becomes three lines, and its length is then arithmetic rather than
    something to count by hand.

    Anything that is not one of those forms passes through untouched, so a
    literal list still validates exactly as before.
    """
    if isinstance(value, dict):
        return _expand(value, 0)
    if isinstance(value, list) and any(isinstance(v, dict) for v in value):
        return _expand({"concat": value}, 0)
    return value


# Every int-list spec field is declared with this, so "an index list may be
# written compactly" is one rule the agent learns once rather than a per-stage
# exception. The BeforeValidator runs ahead of the per-field checks, which
# therefore keep seeing a plain list and are unchanged.
IndexList = Annotated[list[int], BeforeValidator(expand_index_list)]
