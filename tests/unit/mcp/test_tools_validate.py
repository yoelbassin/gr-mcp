from __future__ import annotations

from typing import cast

import pytest

from marconi.mcp.tools import describe_stages, validate_modem

_GOOD = {
    "symbol_rate": 1.0,
    "path": [{"conv": "fsk", "deviation": 1.0}, {"conv": "slice"}],
}


def test_good_spec_returns_trace() -> None:
    out = validate_modem(_GOOD, sample_rate=4.0)
    assert out["valid"] is True
    trace = cast(list[dict[str, object]], out["trace"])
    assert trace[0]["after"] == "<start>"
    assert trace[0]["level"] == "iq"
    assert trace[-1]["after"] == "slice[1]"
    assert trace[-1]["item_type"] == "b"
    assert trace[0]["sample_rate"] == 4.0


def test_unknown_conv_is_structured_error_not_exception() -> None:
    bad = {"symbol_rate": 1.0, "path": [{"conv": "warp_drive"}]}
    out = validate_modem(bad, sample_rate=4.0)
    assert out["valid"] is False
    errors = cast(list[dict[str, object]], out["errors"])
    assert errors[0]["code"] == "invalid_argument"
    assert "warp_drive" in cast(str, errors[0]["message"])


def test_multi_issue_spec_returns_one_error_per_issue() -> None:
    # two independent level faults (slice starts at symbols, not iq; a second
    # slice cannot follow the first's bits output) must arrive as an
    # addressable list with per-issue positions, not one newline-joined blob
    bad = {"symbol_rate": 1.0, "path": [{"conv": "slice"}, {"conv": "slice"}]}
    out = validate_modem(bad, sample_rate=4.0)
    assert out["valid"] is False
    errors = cast(list[dict[str, object]], out["errors"])
    assert len(errors) == 2
    assert [e["at"] for e in errors] == ["slice[0]", "slice[1]"]
    assert all(e["code"] == "invalid_argument" for e in errors)


def test_bad_input_item_type_raises_instead_of_structured_error() -> None:
    with pytest.raises(ValueError, match="input_item_type"):
        validate_modem(_GOOD, sample_rate=4.0, input_item_type="q")


def test_bad_input_level_raises_instead_of_structured_error() -> None:
    with pytest.raises(ValueError, match="input_level"):
        validate_modem(_GOOD, sample_rate=4.0, input_item_type="b", input_level="nope")


def test_sub_nyquist_rate_is_structured_error() -> None:
    out = validate_modem(_GOOD, sample_rate=1.5)
    assert out["valid"] is False
    errors = cast(list[dict[str, object]], out["errors"])
    assert any(
        "sps" in cast(str, e["message"]) or "rate" in cast(str, e["message"])
        for e in errors
    )


def test_validate_modem_reports_seeder_shadow_warning() -> None:
    spec = {
        "symbol_rate": 1000.0,
        "path": [
            {"conv": "sync_word", "bits": "10100001"},
            {"conv": "segment", "frame_body_len": 224},
        ],
    }
    out = validate_modem(
        spec, sample_rate=8000.0, input_item_type="b", input_level="bits"
    )
    assert out["valid"] is True
    warnings = cast(list[str], out["warnings"])
    assert any("segment" in w for w in warnings)


def test_validate_modem_warnings_empty_on_clean_spec() -> None:
    spec = {"symbol_rate": 1000.0, "path": [{"conv": "sync_word", "bits": "1010"}]}
    out = validate_modem(
        spec, sample_rate=8000.0, input_item_type="b", input_level="bits"
    )
    assert out["valid"] is True
    assert out["warnings"] == []


def test_describe_index_and_filters() -> None:
    idx = describe_stages()
    stages = cast(dict[str, list[dict[str, object]]], idx["stages"])
    assert any(e["name"] == "fsk" for e in stages["fsk"])
    assert "envelope" in idx
    one = describe_stages(stage="fsk")
    assert "envelope" not in one
    one_stages = cast(list[dict[str, object]], one["stages"])
    params_schema = cast(dict[str, object], one_stages[0]["params_schema"])
    properties = cast(dict[str, object], params_schema["properties"])
    assert properties["deviation"]
    fam = describe_stages(family="fsk")
    fam_stages = cast(list[dict[str, object]], fam["stages"])
    assert {cast(str, e["name"]) for e in fam_stages} >= {"fsk"}
