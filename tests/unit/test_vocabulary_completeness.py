"""Every enum that owns a vocabulary is the ONE home for it. Where a lookup
table must be written out by hand, it is keyed by the enum and checked for
completeness at import — this pins that the checks exist and that they cover
every member, so adding a member fails loudly instead of raising KeyError
inside a tool call (which reaches the agent as an opaque [internal_error])."""

from __future__ import annotations

import pytest

from marconi.engine.stages.conditioning import _AGC_MODES
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


def test_str_enums_render_as_their_value() -> None:
    # (str, Enum) renders "Level.IQ" under f-string on 3.12+ while StrEnum
    # renders "iq". One idiom tree-wide, so an interpolated member is never
    # the repr in agent-facing text.
    for member in (Level.IQ, ItemType.C, AgcMode.POWER):
        assert f"{member}" == member.value
