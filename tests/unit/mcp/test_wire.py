"""The MCP responses are declared models, not dicts assembled key by key. What
that buys, pinned here: a typo'd or wrong-typed key cannot reach the agent, and
the null-omission rule the tool docstrings promise is one rule in one place."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from marconi.mcp.payload import (
    CappedIntList,
    CensusRow,
    DiagnosticRow,
    PipelinePayload,
    capped_int_list,
)
from marconi.mcp.streams import StreamPage, StreamStats
from marconi.mcp.wire import Payload, Ramp


class _Sample(Payload):
    required: int
    optional: str | None = None
    nested: Ramp | None = None


def test_an_unset_field_is_omitted() -> None:
    assert _Sample(required=1).as_payload() == {"required": 1}


def test_a_field_set_to_null_stays_on_the_wire() -> None:
    # "measured, and undefined" must not read the same as "not applicable"
    assert _Sample(required=1, optional=None).as_payload() == {
        "required": 1,
        "optional": None,
    }


def test_the_omission_rule_reaches_nested_models() -> None:
    payload = PipelinePayload.model_validate(
        {"status": "ok", "stream": None, "soft_stream": None}
    ).as_payload()
    assert payload["stream"] is None and payload["soft_stream"] is None


def test_an_unknown_key_cannot_reach_the_agent() -> None:
    with pytest.raises(ValidationError):
        _Sample.model_validate({"required": 1, "requried": 2})


def test_a_wrong_typed_value_cannot_reach_the_agent() -> None:
    with pytest.raises(ValidationError):
        StreamStats.model_validate(
            {
                "item_type": "f",
                "total_items": 4,
                "sampled_items": 4,
                "sampled": False,
                "centers": "not a list of floats",
            }
        )


@pytest.mark.parametrize(
    "seq,expected",
    [
        ([1, 2, 3], {"windows": [1, 2, 3]}),
        (
            list(range(0, 65 * 7, 7)),
            {
                "windows": [],
                "windows_total": 65,
                "windows_ramp": {"start": 0, "stride": 7, "count": 65},
            },
        ),
    ],
)
def test_capped_int_list_flattens_under_its_key(
    seq: list[int], expected: dict[str, object]
) -> None:
    assert capped_int_list("windows", seq) == expected


def test_the_capped_list_quartet_is_declared_by_every_consumer() -> None:
    # the four suffixed keys are a wire convention; a consumer that forgot one
    # would silently drop it, so both consumers declare all four and validate
    quartet = CappedIntList(
        values=[0], total=9, ramp=Ramp(start=0, stride=1, count=9), path="/tmp/x.i64"
    ).under("marks")
    assert set(quartet) == {"marks", "marks_total", "marks_ramp", "marks_path"}
    row = DiagnosticRow.model_validate({"block": "b0", "key": "bursts", **quartet})
    assert row.marks_total == 9
    for key in ("windows", "marks"):
        assert {f"{key}_total", f"{key}_ramp", f"{key}_path"} <= set(
            PipelinePayload.model_fields
        )


def test_census_kind_is_dropped_not_nulled_when_redundant() -> None:
    row = CensusRow.model_validate({"block": "agc[0]", "items_out": 5})
    assert "kind" not in row.as_payload()


def test_a_page_carries_only_its_own_rendered_field() -> None:
    page = StreamPage.model_validate(
        {"item_type": "b", "offset": 0, "count": 2, "total_items": 2, "bits": "10"}
    ).as_payload()
    assert page["bits"] == "10"
    assert not {"symbols", "values", "real", "imag"} & set(page)
