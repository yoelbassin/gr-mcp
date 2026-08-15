"""Index lists may be written compactly.

They are the largest thing a spec carries and the agent has to TYPE them: a
real multicarrier spec needed ~8200 literal integers across bin_perm, a
de-interleave perm and a depuncture mask. That is the dominant token cost of
the call, and a transcription hazard — a 3096-entry mask was miscounted twice,
in opposite directions, and only the frame-multiple check caught it. Almost
none of that length is information.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from marconi.engine.modulation.coding.stages import DepunctureStep
from marconi.engine.types.perm import expand_index_list

# the puncturing geometry that could not be hand-typed: 21 blocks of one
# 32-bit vector, 3 of another, then a tail
_MASK = {
    "concat": [
        {"repeat": [1, 1, 1, 0], "times": 672},
        {
            "repeat": {"concat": [{"repeat": [1, 1, 1, 0], "times": 7}, [1, 1, 0, 0]]},
            "times": 12,
        },
        {"repeat": [1, 1, 0, 0], "times": 6},
    ]
}


def _ex(value: object) -> list[int]:
    out = expand_index_list(value)
    assert isinstance(out, list)
    return out


def test_a_literal_list_passes_through_unchanged() -> None:
    assert expand_index_list([3, 1, 2]) == [3, 1, 2]
    assert expand_index_list([]) == []


def test_range_repeat_and_concat_expand() -> None:
    assert expand_index_list({"range": [0, 8, 2]}) == [0, 2, 4, 6]
    assert expand_index_list({"range": [2, 5]}) == [2, 3, 4]
    assert expand_index_list({"repeat": [1, 0], "times": 3}) == [1, 0, 1, 0, 1, 0]
    assert expand_index_list({"concat": [[9], {"range": [0, 3]}]}) == [9, 0, 1, 2]


def test_a_list_may_mix_literals_and_compact_parts() -> None:
    assert expand_index_list([[7], {"repeat": [0], "times": 2}]) == [7, 0, 0]


def test_the_stride_two_deinterleave_is_two_ranges() -> None:
    perm = _ex({"concat": [{"range": [0, 3072, 2]}, {"range": [1, 3072, 2]}]})
    assert sorted(perm) == list(range(3072))
    assert perm[:3] == [0, 2, 4] and perm[1536:1539] == [1, 3, 5]


def test_the_hand_miscounted_mask_expands_to_the_right_geometry() -> None:
    mask = _ex(_MASK)
    assert len(mask) == 3096
    assert sum(mask) == 2304


def test_a_compact_mask_reaches_the_step_model_and_still_validates() -> None:
    """The expansion runs BEFORE the field's own checks, so a compact spec is
    held to exactly the rules a literal one is."""
    step = DepunctureStep.model_validate({"conv": "depuncture", "keep_mask": _MASK})
    assert len(step.keep_mask) == 3096
    assert sum(step.keep_mask) == 2304
    with pytest.raises(ValidationError, match="1 \\(keep\\) or 0"):
        DepunctureStep.model_validate(
            {"conv": "depuncture", "keep_mask": {"repeat": [1, 2], "times": 4}}
        )


@pytest.mark.parametrize(
    "bad",
    [
        {"range": [0]},
        {"range": [0, 10, 0]},
        {"repeat": [1]},
        {"repeat": [1], "times": -1},
        {"nope": [1]},
        {"range": [0, 10], "repeat": [1]},
    ],
)
def test_malformed_compact_forms_are_rejected(bad: object) -> None:
    with pytest.raises(Exception):
        expand_index_list(bad)


def test_a_non_compact_value_still_fails_at_the_field() -> None:
    """The expansion only claims dicts; anything else passes through to the
    field's own type check, so a junk value is rejected there and not turned
    into a confusing expansion error."""
    assert expand_index_list("seven") == "seven"
    with pytest.raises(ValidationError):
        DepunctureStep.model_validate({"conv": "depuncture", "keep_mask": "seven"})


def test_a_compact_form_cannot_outgrow_a_literal_one() -> None:
    """The bound a literal list gets from having to be typed does not exist
    here: {'repeat': [0], 'times': 10**9} is nine characters."""
    with pytest.raises(Exception):
        expand_index_list({"repeat": [0, 1], "times": 10**9})
    with pytest.raises(Exception):
        expand_index_list({"range": [0, 10**9]})
