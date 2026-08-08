"""Each description documents a measured burst-signal trap from the ADS-B
dogfood; the schema is the agent-facing surface, so assert there."""

from typing import Any

from marconi.engine.modulation.ook.stages import OokEnvelopeStep
from marconi.engine.stages.conditioning import AgcStep, SquelchStep


def _field_desc(model: type[Any], field: str) -> str:
    return model.model_json_schema()["properties"][field].get("description", "")


def test_ook_loop_bw_names_open_loop() -> None:
    d = _field_desc(OokEnvelopeStep, "loop_bw")
    assert "open-loop" in d and "burst" in d and "deterministic" in d


def test_agc_window_documents_bursty_trap() -> None:
    d = _field_desc(AgcStep, "window_symbols")
    assert "burst" in d and "noise" in d


def test_squelch_documents_ppm_limit() -> None:
    assert "IIR" in _field_desc(SquelchStep, "threshold_db")
    d = _field_desc(SquelchStep, "alpha_symbols")
    assert "burst" in d and "ook_envelope" in d
