from __future__ import annotations

from marconi.engine.stages.registry import stage_registry
from marconi.mcp.vocab import ENVELOPE, stage_details, stage_index


def test_index_covers_every_registry_stage() -> None:
    index = stage_index()
    names = {e.name for rows in index.values() for e in rows}
    assert names == set(stage_registry())
    assert set(index) == {s.family for s in stage_registry().values()}
    entry = next(e for e in index["fsk"] if e.name == "fsk")
    assert entry.levels == "iq>symbols"
    assert set(entry.as_payload()) >= {"name", "levels", "dir"}


def test_details_carry_schema_and_contracts() -> None:
    (d,) = stage_details(["fsk"])
    assert d.params_schema is not None
    assert d.params_schema["properties"]["deviation"]
    # the KEY must reach the wire even when the contract is null
    assert "min_input_sps" in d.as_payload()
    assert d.dir == "rx,tx"
    assert d.family == "fsk"


def test_fsk_loop_bw_documents_open_loop_mode() -> None:
    # the schema is the agent's only view of the stage, so the open-loop knob
    # for short bursts must be discoverable there
    (d,) = stage_details(["fsk"])
    assert d.params_schema is not None
    loop_bw = d.params_schema["properties"]["loop_bw"]
    assert "open-loop" in loop_bw.get("description", "").lower()


def test_descramble_sequence_documents_hex_format() -> None:
    (d,) = stage_details(["descramble"])
    assert d.params_schema is not None
    sequence = d.params_schema["properties"]["sequence"]
    assert "hex" in sequence.get("description", "").lower()


def test_every_stage_detail_builds() -> None:
    details = stage_details(sorted(stage_registry()))
    assert len(details) == len(stage_registry())
    for d in details:
        assert isinstance(d.params_schema, dict)


def test_envelope_documents_the_spec_shape() -> None:
    assert "symbol_rate" in str(ENVELOPE)
    assert "path" in str(ENVELOPE)


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
        detail = stage_details([name])[0]
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
