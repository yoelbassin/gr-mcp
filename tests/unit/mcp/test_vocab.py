from __future__ import annotations

import json

from marconi.engine.stages.registry import stage_registry
from marconi.engine.types.bounds import MAX_DECIM
from marconi.mcp.tools import describe_stages
from marconi.mcp.vocab import (
    ENVELOPE,
    MAX_DETAIL_BYTES,
    stage_details,
    stage_index,
)


def test_index_covers_every_registry_stage() -> None:
    index = stage_index()
    names = {e.name for rows in index.values() for e in rows}
    assert names == set(stage_registry())
    assert set(index) == {s.family for s in stage_registry().values()}
    entry = next(e for e in index["fsk"] if e.name == "fsk")
    assert entry.levels == "iq>symbols"
    assert set(entry.as_payload()) >= {"name", "levels"}


def test_details_carry_schema_and_contracts() -> None:
    ((d,), _) = stage_details(["fsk"])
    assert d.params_schema is not None
    assert d.params_schema["properties"]["deviation"]
    # the KEY must reach the wire even when the contract is null
    assert "min_input_sps" in d.as_payload()
    assert d.family == "fsk"


def test_fsk_loop_bw_documents_open_loop_mode() -> None:
    # the schema is the agent's only view of the stage, so the open-loop knob
    # for short bursts must be discoverable there
    ((d,), _) = stage_details(["fsk"])
    assert d.params_schema is not None
    loop_bw = d.params_schema["properties"]["loop_bw"]
    assert "open-loop" in loop_bw.get("description", "").lower()


def test_descramble_sequence_documents_hex_format() -> None:
    ((d,), _) = stage_details(["descramble"])
    assert d.params_schema is not None
    sequence = d.params_schema["properties"]["sequence"]
    assert "hex" in sequence.get("description", "").lower()


def test_every_stage_detail_builds() -> None:
    names = sorted(stage_registry())
    # a budget generous enough that nothing is dropped: this test is about the
    # rows building, and the budget has its own test below
    details, omitted = stage_details(names, budget=1 << 30)
    assert not omitted
    assert len(details) == len(names)
    for d in details:
        assert isinstance(d.params_schema, dict)


def test_envelope_documents_the_spec_shape() -> None:
    assert "symbol_rate" in str(ENVELOPE)
    assert "path" in str(ENVELOPE)


def test_the_envelope_states_the_extra_param_rule_the_schemas_dropped() -> None:
    # additionalProperties:false was identical in all ~58 per-stage schemas;
    # it is stated once here instead, so dropping it from them loses nothing
    assert "rejected" in ENVELOPE.spec_envelope.unknown_params
    ((d,), _) = stage_details(["fsk"])
    assert d.params_schema is not None
    assert "additionalProperties" not in d.params_schema


def test_a_detail_call_costs_no_more_than_its_budget() -> None:
    """Every other agent-facing surface prices itself (_MAX_PAGE_BYTES,
    _MAX_INLINE_LIST, _SPECTRUM_BINS); this one did not, and it is the biggest
    response the surface can produce. Measured before the budget: the coding
    family alone returned 13017 bytes — more than all eight tool docstrings
    together — and grew with the family."""
    for family in sorted({s.family for s in stage_registry().values()}):
        payload = describe_stages(family=family)
        size = len(json.dumps(payload))
        assert size <= MAX_DETAIL_BYTES, f"{family} detail call is {size} bytes"


def test_a_dropped_stage_is_named_so_it_can_be_fetched() -> None:
    names = sorted(stage_registry())
    rows, omitted = stage_details(names, budget=1)
    # one row always comes back, however big: a stage asked for by name must
    # never come back empty
    assert len(rows) == 1 and rows[0].name == names[0]
    assert omitted == names[1:]
    # and every omitted name is fetchable on its own
    ((only,), rest) = stage_details([omitted[0]])
    assert only.name == omitted[0] and not rest


def test_the_schema_keeps_what_the_agent_cannot_derive() -> None:
    # titles were pydantic restating the field name in Title Case on every
    # field of every stage; descriptions and constraints are the payload
    ((d,), _) = stage_details(["channelize"])
    assert d.params_schema is not None
    decim = d.params_schema["properties"]["decim"]
    assert "title" not in decim
    assert decim["maximum"] == MAX_DECIM
    assert "description" in decim


def test_a_step_conditional_contract_is_never_published_as_a_number() -> None:
    # Stage.accepts_amplitude_for's own docstring says the compiler consults
    # the METHOD, never the bare attribute. describe_stages published the
    # attribute, so symbol_sync advertised a 2.0 sps floor while enforcing
    # 4.0 open-loop and none closed-loop, and ook_envelope advertised an
    # amplitude contract its open-loop path drops — the exact trap
    # OOK_AGC_REMOVE_HINT exists to undo after a decode has degraded.
    from marconi.engine.stages.base import Stage
    from marconi.engine.stages.registry import stage_registry

    for name, stage in stage_registry().items():
        detail = stage_details([name])[0][0]
        conditional = detail.step_conditional or []
        for hook, key in (
            ("min_input_sps_for", "min_input_sps"),
            ("accepts_amplitude_for", "accepts_amplitude"),
        ):
            overrides = getattr(type(stage), hook, None) is not getattr(
                Stage, hook, None
            )
            if overrides:
                assert key in conditional, f"{name}: {key} published unconditionally"
                assert (
                    getattr(detail, key) is None
                ), f"{name}: {key} still carries a value"
