"""Floors whose values are measurements, pinned so a tidy-up cannot move
them without meeting the measurement that set them."""

from __future__ import annotations

from marconi.engine.stages.registry import stage_registry


def test_rectangular_ted_floors_sit_at_four() -> None:
    # Gardner's TED is degenerate at 2 sps on a RECTANGULAR waveform (a
    # discriminator output or an envelope has no excess-bandwidth shaping):
    # measured BER at sps 2 and 3 was 0.480 median over 10 seeds for BOTH
    # fsk and closed-loop ook_envelope, 0.000 at sps 4 - "confident garbage
    # with status ok", the exact case base.py says the floor exists for.
    # psk_demod at sps 2 is FINE (RRC-shaped; SER 0.0 measured) and keeps 2.
    reg = stage_registry()
    assert reg["fsk"].min_input_sps == 4.0
    ook = reg["ook_envelope"]
    closed = ook.step_model.model_validate({"conv": "ook_envelope", "loop_bw": 0.045})
    opened = ook.step_model.model_validate({"conv": "ook_envelope", "loop_bw": 0.0})
    assert ook.min_input_sps_for(closed) == 4.0
    assert ook.min_input_sps_for(opened) == 1.0
    assert reg["psk_demod"].min_input_sps == 2.0
