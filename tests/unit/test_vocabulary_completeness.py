"""Every enum that owns a vocabulary is the ONE home for it. Where a lookup
table must be written out by hand, it is keyed by the enum and checked for
completeness at import — this pins that the checks exist and that they cover
every member, so adding a member fails loudly instead of raising KeyError
inside a tool call (which reaches the agent as an opaque [internal_error])."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from marconi.engine.stages.conditioning import _AGC_MODES
from marconi.engine.stages.registry import step_models
from marconi.engine.types.enums import AgcMode, ItemType
from marconi.engine.types.levels import Level
from marconi.mcp.payload import _TRACE_STAT_KEYS
from marconi.mcp.streams import _PAGE_SPECS, PageType, _check_page_specs
from marconi.mcp.tools import _DEFAULT_ENTRY_LEVEL, _ENTRY_LEVELS
from marconi.mcp.vocab import _ITEM_GLOSS, _LEVEL_GLOSS


def test_every_item_type_can_be_paged() -> None:
    assert {t.value for t in ItemType} <= {p.value for p in PageType}
    assert set(_PAGE_SPECS) == set(PageType)


def test_page_specs_agree_with_the_item_type_suffixes() -> None:
    for t in ItemType:
        assert _PAGE_SPECS[PageType(t.value)].suffix == t.suffix


def test_a_page_type_without_a_spec_fails_at_import(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    patched = dict(_PAGE_SPECS)
    del patched[PageType.L]
    monkeypatch.setattr("marconi.mcp.streams._PAGE_SPECS", patched)
    with pytest.raises(RuntimeError, match="no PageSpec"):
        _check_page_specs()


def test_every_item_type_has_a_default_entry_level_and_a_gloss() -> None:
    assert set(_DEFAULT_ENTRY_LEVEL) == set(ItemType)
    assert set(_ITEM_GLOSS) == set(ItemType)


def test_every_level_has_a_gloss_and_every_enterable_level_is_offered() -> None:
    assert set(_LEVEL_GLOSS) == set(Level)
    assert set(_ENTRY_LEVELS) == {lv.value for lv in Level if lv is not Level.AUDIO}


def test_every_item_type_has_trace_stat_keys() -> None:
    assert set(_TRACE_STAT_KEYS) == set(ItemType)


def test_every_agc_mode_has_one_spec() -> None:
    assert set(_AGC_MODES) == set(AgcMode)
    # each mode drives a DISTINCT statistic, or a consumer's amplitude contract
    # could be satisfied by a mode that normalizes something else
    amplitudes = [s.amplitude for s in _AGC_MODES.values()]
    assert len(set(amplitudes)) == len(amplitudes)


# A minimal VALID spec for every step that takes an index list, so the sweep
# below corrupts one field of a spec that otherwise compiles — a fixture that
# was already invalid would make every case pass for the wrong reason.
_INDEX_LIST_STEPS: dict[str, dict[str, object]] = {
    "deinterleave": {"perm": [0, 1]},
    "permute": {"perm": [0, 1]},
    "cell_select": {"select_perm": [0, 1], "keep": 1},
    "ofdm_demod": {
        "fft_len": 4,
        "cp_len": 1,
        "sym_len": 5,
        "null_len": 2,
        "frame_len": 100,
        "data_syms": 2,
        "n_carriers": 2,
        "bin_perm": [0, 1, 2, 3],
    },
}


def _index_list_fields() -> list[tuple[str, str]]:
    """Every step field whose name marks it as an index list. The name is the
    handle because the type alone cannot tell one from any other list[int]:
    what makes these special is that the value indexes a block's own stride."""
    out: list[tuple[str, str]] = []
    for conv, model in sorted(step_models().items()):
        for field, info in model.model_fields.items():
            if field.endswith("perm") or field.endswith("_indices"):
                assert info.annotation == list[int], (
                    f"{conv}.{field} is named like an index list but is typed "
                    f"{info.annotation}"
                )
                out.append((conv, field))
    return out


def _spec(conv: str, **overrides: object) -> dict[str, object]:
    base = _INDEX_LIST_STEPS.get(conv)
    assert base is not None, (
        f"'{conv}' takes an index list but has no minimal spec in "
        "_INDEX_LIST_STEPS; add one so the sweep below actually exercises it"
    )
    return {"conv": conv, **base, **overrides}


def test_every_index_list_field_is_covered_by_a_minimal_spec() -> None:
    # the sweep is vacuously green if a stage renames its field out of the
    # convention, so pin the ones we know about and that each fixture validates
    found = set(_index_list_fields())
    assert {
        ("deinterleave", "perm"),
        ("permute", "perm"),
        ("ofdm_demod", "bin_perm"),
        ("cell_select", "select_perm"),
    } <= found
    models = step_models()
    for conv, _ in found:
        models[conv].model_validate(_spec(conv))


@pytest.mark.parametrize("conv, field", _index_list_fields(), ids=str)
def test_every_index_list_field_rejects_a_negative_index(conv: str, field: str) -> None:
    """A negative index wraps in numpy and does not survive the stock
    blockinterleaver's pybind signature at all. deinterleave and ofdm_demod
    each shipped with no check whatsoever, so an invalid list reached the GR
    constructor inside the worker while validate_modem — whose whole purpose
    is catching this without running — reported valid."""
    valid = list(_INDEX_LIST_STEPS[conv][field])  # type: ignore[call-overload]
    with pytest.raises(ValidationError):
        step_models()[conv].model_validate(_spec(conv, **{field: [-1, *valid[1:]]}))


@pytest.mark.parametrize("conv, field", _index_list_fields(), ids=str)
def test_every_index_list_field_rejects_an_empty_list(conv: str, field: str) -> None:
    with pytest.raises(ValidationError):
        step_models()[conv].model_validate(_spec(conv, **{field: []}))


def test_str_enums_render_as_their_value() -> None:
    # (str, Enum) renders "Level.IQ" under f-string on 3.12+ while StrEnum
    # renders "iq". One idiom tree-wide, so an interpolated member is never
    # the repr in agent-facing text.
    for member in (Level.IQ, ItemType.C, AgcMode.POWER):
        assert f"{member}" == member.value
