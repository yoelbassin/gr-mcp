from __future__ import annotations

from marconi.engine.modulation.fsk.stages import FskStep, MskStep
from marconi.engine.modulation.ook.stages import (
    OOK_AGC_REMOVE_HINT as _OOK_AGC_REMOVE_HINT,
)
from marconi.engine.modulation.ook.stages import (
    OokEnvelopeStep,
)
from marconi.engine.run import _hints, composition_warnings
from marconi.engine.stages.conditioning import AgcStep
from marconi.engine.stages.general import SliceStep
from marconi.engine.stages.registry import stage_registry, step_models
from marconi.engine.types.descriptor import Descriptor
from marconi.engine.types.enums import ItemType
from marconi.engine.types.levels import Level
from marconi.engine.types.models import Modem

BITS = Descriptor(Level.BITS, ItemType.B)


def _modem(path: list[dict[str, object]]) -> Modem:
    return Modem.from_spec({"symbol_rate": 1000.0, "path": path}, step_models())


def test_msk_stage_is_tagged_polarity_ambiguous() -> None:
    reg = stage_registry()
    assert reg["msk"].polarity_ambiguous is True
    assert reg["fsk"].polarity_ambiguous is False  # discriminator has a fixed sense


def test_hints_flags_polarity_for_a_coherent_demod_path() -> None:
    modem = Modem(symbol_rate=2400.0, path=[MskStep(), SliceStep()])
    hints = _hints(modem, stage_registry(), BITS)
    assert any("polarity" in h.lower() for h in hints)


def test_hints_empty_for_a_non_ambiguous_demod() -> None:
    modem = Modem(symbol_rate=2400.0, path=[FskStep(deviation=1000.0), SliceStep()])
    assert _hints(modem, stage_registry(), BITS) == []


def test_hints_suggest_open_loop_retry_for_weak_closed_loop_fsk() -> None:
    modem = Modem(
        symbol_rate=1_000_000.0, path=[FskStep(deviation=250_000.0), SliceStep()]
    )
    hints = _hints(modem, stage_registry(), BITS, "no_signal")
    assert any("loop_bw" in h and "open-loop" in h.lower() for h in hints)


def test_hints_silent_when_fsk_decoded_or_already_open_loop() -> None:
    closed = Modem(
        symbol_rate=1_000_000.0, path=[FskStep(deviation=250_000.0), SliceStep()]
    )
    open_loop = Modem(
        symbol_rate=1_000_000.0,
        path=[FskStep(deviation=250_000.0, loop_bw=0.0), SliceStep()],
    )
    # a decoded run needs no retry; an already-open-loop run has none to offer
    assert _hints(closed, stage_registry(), BITS, "decoded") == []
    assert _hints(open_loop, stage_registry(), BITS, "no_signal") == []


def test_symbol_sync_closed_loop_hint_on_undecoded() -> None:
    # symbol_sync gained the same loop_bw=0 open-loop mode as fsk and
    # ook_envelope; a failed closed-loop run must get the same class of
    # retry hint — stage-owned, so a new open-loop mode cannot silently
    # miss the hint table again
    from marconi.engine.modulation.psk.stages import SymbolSyncStep

    closed = Modem(symbol_rate=1000.0, path=[SymbolSyncStep(sps=4)])
    hints = _hints(closed, stage_registry(), BITS, "no_signal")
    assert any("symbol_sync" in h and "loop_bw" in h for h in hints)
    open_loop = Modem(symbol_rate=1000.0, path=[SymbolSyncStep(sps=4, loop_bw=0.0)])
    assert _hints(open_loop, stage_registry(), BITS, "no_signal") == []
    assert _hints(closed, stage_registry(), BITS, "decoded") == []


def test_ook_closed_loop_hint_on_undecoded() -> None:
    modem = Modem(
        symbol_rate=2400.0, path=[OokEnvelopeStep(loop_bw=0.045), SliceStep()]
    )
    hints = _hints(modem, stage_registry(), BITS, verdict="uncertain")
    assert any("ook_envelope" in h and "open-loop" in h.lower() for h in hints)


def test_no_ook_hint_when_open_loop_or_decoded() -> None:
    open_loop = Modem(
        symbol_rate=2400.0, path=[OokEnvelopeStep(loop_bw=0.0), SliceStep()]
    )
    assert not any(
        "ook_envelope" in h
        for h in _hints(open_loop, stage_registry(), BITS, "uncertain")
    )
    closed = Modem(
        symbol_rate=2400.0, path=[OokEnvelopeStep(loop_bw=0.045), SliceStep()]
    )
    assert not any(
        "ook_envelope" in h for h in _hints(closed, stage_registry(), BITS, "decoded")
    )


def test_remove_agc_hint_for_open_loop_ook_with_agc_in_path() -> None:
    open_with_agc = Modem(
        symbol_rate=2400.0,
        path=[AgcStep(), OokEnvelopeStep(loop_bw=0.0), SliceStep()],
    )
    present = _hints(open_with_agc, stage_registry(), BITS, verdict="uncertain")
    assert _OOK_AGC_REMOVE_HINT in present
    assert all(s in _OOK_AGC_REMOVE_HINT for s in ("ook_envelope", "agc", "remove"))

    # open-loop with NO agc: nothing to remove, the hint stays silent
    open_no_agc = Modem(
        symbol_rate=2400.0, path=[OokEnvelopeStep(loop_bw=0.0), SliceStep()]
    )
    assert _OOK_AGC_REMOVE_HINT not in _hints(
        open_no_agc, stage_registry(), BITS, verdict="uncertain"
    )

    # closed-loop + agc is the CORRECT pairing: the remove-agc hint must stay
    # silent, and the existing open-loop-retry hint governs the closed-loop case
    closed_with_agc = Modem(
        symbol_rate=2400.0,
        path=[AgcStep(), OokEnvelopeStep(loop_bw=0.045), SliceStep()],
    )
    closed_hints = _hints(closed_with_agc, stage_registry(), BITS, verdict="uncertain")
    assert _OOK_AGC_REMOVE_HINT not in closed_hints
    assert any("ook_envelope" in h and "open-loop" in h.lower() for h in closed_hints)


def test_seeder_after_seeder_warns() -> None:
    modem = _modem(
        [
            {"conv": "sync_word", "bits": "10100001"},
            {"conv": "segment", "frame_body_len": 224},
            {"conv": "codebook", "code_bits": 2, "data_bits": 1, "table": [1, 2]},
        ]
    )
    warnings = composition_warnings(modem, stage_registry())
    assert len(warnings) == 1
    assert "segment" in warnings[0] and "sync_word" in warnings[0]
    assert "discard" in warnings[0].lower()


def test_single_seeder_and_gated_reseed_do_not_warn() -> None:
    assert (
        composition_warnings(
            _modem([{"conv": "sync_word", "bits": "10100001"}]), stage_registry()
        )
        == []
    )
    # sync_align GATES (seeds_windows=False) then segment re-tiles: the blessed combo
    assert (
        composition_warnings(
            _modem(
                [
                    {
                        "conv": "sync_align",
                        "access_code": "10100001",
                        "frame_len": 224,
                    },
                    {"conv": "harden"},
                    {"conv": "segment", "frame_body_len": 224},
                ]
            ),
            stage_registry(),
        )
        == []
    )
