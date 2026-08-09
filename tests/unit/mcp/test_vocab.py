from __future__ import annotations

from marconi.engine.stages.registry import stage_registry
from marconi.mcp.vocab import ENVELOPE, stage_details, stage_index


def test_index_covers_every_registry_stage() -> None:
    index = stage_index()
    names = {e["name"] for rows in index.values() for e in rows}
    assert names == set(stage_registry())
    assert set(index) == {s.family for s in stage_registry().values()}
    entry = next(e for e in index["fsk"] if e["name"] == "fsk")
    assert entry["levels"] == "iq>symbols"
    assert set(entry) >= {"name", "levels", "dir"}


def test_details_carry_schema_and_contracts() -> None:
    (d,) = stage_details(["fsk"])
    assert d["params_schema"]["properties"]["deviation"]
    assert "min_input_sps" in d
    assert d["dir"] == "rx,tx"
    assert d["family"] == "fsk"


def test_fsk_loop_bw_documents_open_loop_mode() -> None:
    # the schema is the agent's only view of the stage, so the open-loop knob
    # for short bursts must be discoverable there
    (d,) = stage_details(["fsk"])
    loop_bw = d["params_schema"]["properties"]["loop_bw"]
    assert "open-loop" in loop_bw.get("description", "").lower()


def test_descramble_sequence_documents_hex_format() -> None:
    (d,) = stage_details(["descramble"])
    sequence = d["params_schema"]["properties"]["sequence"]
    assert "hex" in sequence.get("description", "").lower()


def test_every_stage_detail_builds() -> None:
    details = stage_details(sorted(stage_registry()))
    assert len(details) == len(stage_registry())
    for d in details:
        assert isinstance(d["params_schema"], dict)


def test_envelope_documents_the_spec_shape() -> None:
    assert "symbol_rate" in str(ENVELOPE)
    assert "path" in str(ENVELOPE)
