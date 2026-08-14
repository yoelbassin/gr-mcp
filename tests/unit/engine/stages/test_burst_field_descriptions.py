"""Each description documents a measured burst-signal trap from the ADS-B
dogfood; the schema is the agent-facing surface, so assert there."""

import re
from typing import Any

from marconi.engine.modulation.ook.stages import OokEnvelopeStep
from marconi.engine.stages.conditioning import AgcStep, SquelchStep
from marconi.engine.stages.registry import stage_registry


def _field_desc(model: type[Any], field: str) -> str:
    desc = model.model_json_schema()["properties"][field].get("description", "")
    assert isinstance(desc, str)
    return desc


def test_ook_loop_bw_names_open_loop() -> None:
    d = _field_desc(OokEnvelopeStep, "loop_bw")
    assert "open-loop" in d and "burst" in d and "deterministic" in d


def test_ook_loop_bw_quotes_the_closed_loop_floor_the_compiler_reads() -> None:
    """The field text offered "closed-loop still needs sps>=2" — a floor the
    compiler refuses (it reads min_input_sps_for, which is 4 from the measured
    BER) and the stage description in the same payload contradicts."""
    quoted = re.search(
        r"closed-loop still needs sps>=(\d+)", _field_desc(OokEnvelopeStep, "loop_bw")
    )
    assert quoted is not None
    stage = stage_registry()["ook_envelope"]
    closed = OokEnvelopeStep(conv="ook_envelope", loop_bw=0.045)
    assert float(quoted.group(1)) == stage.min_input_sps_for(closed)


def test_agc_window_documents_bursty_trap() -> None:
    d = _field_desc(AgcStep, "window_symbols")
    assert "burst" in d and "noise" in d


def test_squelch_documents_ppm_limit() -> None:
    assert "IIR" in _field_desc(SquelchStep, "threshold_db")
    d = _field_desc(SquelchStep, "alpha_symbols")
    assert "burst" in d and "ook_envelope" in d
