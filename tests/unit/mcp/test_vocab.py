from __future__ import annotations

from marconi.engine.stages.registry import stage_registry
from marconi.mcp.vocab import ENVELOPE, stage_details, stage_index


def test_index_covers_every_registry_stage() -> None:
    index = stage_index()
    assert {e["name"] for e in index} == set(stage_registry())
    entry = next(e for e in index if e["name"] == "fsk")
    assert entry["family"] == "fsk"
    assert entry["from_level"] == "iq"
    assert set(entry) >= {"name", "family", "from_level", "to_level", "directions"}


def test_details_carry_schema_and_contracts() -> None:
    (d,) = stage_details(["fsk"])
    assert d["params_schema"]["properties"]["deviation"]
    assert "min_input_sps" in d
    assert d["directions"] == ["rx", "tx"]


def test_every_stage_detail_builds() -> None:
    details = stage_details(sorted(stage_registry()))
    assert len(details) == len(stage_registry())
    for d in details:
        assert isinstance(d["params_schema"], dict)


def test_envelope_documents_the_spec_shape() -> None:
    assert "symbol_rate" in str(ENVELOPE)
    assert "path" in str(ENVELOPE)
