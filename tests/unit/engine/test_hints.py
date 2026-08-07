from __future__ import annotations

from marconi.engine.modulation.fsk.stages import FskStep, MskStep
from marconi.engine.run import _hints
from marconi.engine.stages.general import SliceStep
from marconi.engine.stages.registry import stage_registry
from marconi.engine.types.descriptor import Descriptor
from marconi.engine.types.enums import ItemType
from marconi.engine.types.levels import Level
from marconi.engine.types.models import Modem

BITS = Descriptor(Level.BITS, ItemType.B)


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
