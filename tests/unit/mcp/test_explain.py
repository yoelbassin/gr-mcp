"""explain() holds the interpretation prose the tool docstrings used to carry
inline. Relocating agent-facing text is only safe if two things are gated: the
pointers resolve, and nothing was lost on the way.

The preamble cost is the reason it moved — the @tool docstrings load on every
session whether or not the tool is called, measured at ~5.5k tokens before the
agent did anything, survey alone ~2.2k of it. Almost none of that was needed to
CALL survey; it was needed to READ a survey result.
"""

from __future__ import annotations

import inspect
import re

import pytest

from marconi.mcp.explain import Topic, topic_index, topic_sections
from marconi.mcp.tools import TOOLS, explain

_POINTER = re.compile(r'explain\("([a-z_.]+)"\)')


def _docstrings() -> dict[str, str]:
    return {
        name: inspect.getdoc(getattr(fn, "__wrapped__", fn)) or ""
        for name, fn in TOOLS.items()
    }


def test_every_topic_a_docstring_points_at_resolves() -> None:
    """A pointer to a topic that does not exist is a dead end the agent cannot
    recover from — it has no other copy of the text."""
    pointed = {t for doc in _docstrings().values() for t in _POINTER.findall(doc)}
    assert pointed, "no explain() pointers found — the split is not wired up"
    for topic in sorted(pointed):
        assert topic_sections(topic), topic


def test_every_topic_is_reachable_from_a_docstring_pointer() -> None:
    """The other direction: a section nothing points at is a section the agent
    never learns exists. A bare tool prefix counts — explain("survey") returns
    every survey.* section."""
    pointed = {t for doc in _docstrings().values() for t in _POINTER.findall(doc)}
    reachable = {s.topic for p in pointed for s in topic_sections(p)}
    missing = sorted(t.value for t in Topic if t.value not in reachable)
    assert not missing, f"unreachable explain topics: {missing}"


def test_the_index_names_every_topic() -> None:
    assert set(topic_index()) == {t.value for t in Topic}
    assert all(summary.strip() for summary in topic_index().values())


def test_a_bare_prefix_returns_all_of_a_tools_sections() -> None:
    survey = {s.topic for s in topic_sections("survey")}
    assert survey == {t.value for t in Topic if t.value.startswith("survey.")}
    assert len(survey) > 1


def test_an_exact_topic_beats_a_prefix() -> None:
    assert [s.topic for s in topic_sections("survey.carrier")] == ["survey.carrier"]


def test_an_unknown_topic_lists_the_known_ones() -> None:
    with pytest.raises(ValueError, match="unknown topic"):
        topic_sections("survey.nonsense")


def test_the_index_call_and_a_topic_call_are_different_shapes() -> None:
    index = explain()
    assert "topics" in index and "sections" not in index
    section = explain("survey.carrier")
    assert "sections" in section and "topics" not in section


@pytest.mark.parametrize("tool", ["survey", "run_rx"])
def test_the_relocated_prose_kept_every_fact(tool: str) -> None:
    """Fact-token check across the move: every identifier, number and
    emphasised term the pre-split docstring carried must still be reachable —
    either still inline, or in the sections that docstring points at. Prose may
    be rewritten; a measured caveat may not be dropped."""
    before = _PRE_SPLIT[tool]
    doc = _docstrings()[tool]
    relocated = "\n".join(
        s.text for p in _POINTER.findall(doc) for s in topic_sections(p)
    )
    lost = sorted(_facts(before) - _facts(doc + "\n" + relocated))
    assert not lost, f"{tool}: facts lost in the split: {lost}"


def _facts(text: str) -> set[str]:
    return set(re.findall(r"\b[a-z][a-z0-9]*_[a-z0-9_]+\b|\b\d+(?:\.\d+)?\b", text))


# The docstrings as they stood before the split, kept verbatim so the check
# above measures against what shipped rather than against itself.
_PRE_SPLIT: dict[str, str] = {
    "survey": """
    Reads a bounded slice (capture_offset/capture_samples in complex samples,
    0 = to EOF; capture_dtype cf32/ci16/ci8/cu8, matching run_rx).
    WHICH SAMPLES: "bursts" streams the whole slice; every other block measures
    ONE contiguous window of at most 2^20 samples, relocated to the most active
    megasample. "analyzed_start" (with "analyzed_samples" and "span_samples")
    reports where it landed, the same frame as bursts.segments.
    SUB-BAND: center_hz and/or decim, bandwidth_hz sets the passband.
    "spectrum" — psd_db; bin i at freq_start_hz + i*freq_step_hz,
    center_offset_hz, peak_offset_hz, occupied_bw_hz, occupied_lo_hz,
    occupied_hi_hz.
    "carrier" — offset_hz, method "mpsk" or "spectral_centroid",
    offset_ambiguous, off_center, ppm = offset_hz / tuned_hz * 1e6, psk_order
    4 or 8, phase_concentration order_2/order_4/order_8.
    "envelope" — const_envelope_ratio, amplitude_kurtosis.
    "symbol_rate" — candidates_hz, strengths, eye_openness, eye_confirmed,
    search_lo_hz, search_hi_hz, min_symbol_rate, max_symbol_rate,
    clock_resolution_hz.
    "inst_freq" — hist_counts, hist_start_hz, hist_step_hz, peaks_hz,
    spread_hz.
    "bursts" — segments, duty_cycle, dominant_period_samples, segments_total,
    capture_scale offset_samples, decim; capture_offset = offset_samples +
    start*decim, capture_samples = length*decim. 512 cap.
    """,
    "run_rx": """
    capture_path, capture_dtype cf32/ci16/ci8/cu8, capture_offset,
    capture_samples, input_path, input_item_type, input_level.
    item_type "f" at level "symbols" slices POSITIVE to bit 1; at level "bits"
    it is an LLR where bit 1 is NEGATIVE. item_type "c" at level "symbols" is a
    constellation symbol, at level "iq" it is conditioned IQ.
    stream_stats item_type "c" constant_modulus_ratio is the false-lock check,
    not EVM.
    trace rows: mean_magnitude, std_magnitude, constant_modulus_ratio,
    ones_fraction, sample_rate.
    soft_stream {path, item_type, items, level, bit1_sign}.
    windows_ramp/marks_ramp {start, stride, count}; lists over 512 entries
    truncate with a _total count and a _path int64 sidecar.
    quality verdict decoded / uncertain / no_signal, quality.rationale.
    timeout, deadline_exceeded, status="timeout".
    """,
}
