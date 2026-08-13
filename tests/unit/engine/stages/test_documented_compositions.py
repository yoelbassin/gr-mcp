"""Every multi-stage recipe a stage description or tool docstring spells out,
compiled.

Stage descriptions and the @tool docstrings are functional product — they are
the only instructions the driving agent gets — but nothing executed the
compositions they name, so a description could advertise a path the compiler
refuses. It did: differential_demod's own text offered the pi/4-shifted
quaternary recipe, and the compiler rejected it with an alphabet-order error
because the stage inherited the base "order passes through" descriptor.

A recipe named in agent-facing text belongs here the day it is written.
"""

from __future__ import annotations

import math
from typing import Any

import pytest

from marconi.mcp.tools import validate_modem

# name -> (spec, sample_rate). Each must compile in the rx direction from a
# complex-IQ entry; the assertion is only that the compiler accepts it, since
# whether it DECODES is what the off-air gates measure.
_DOCUMENTED: dict[str, tuple[dict[str, Any], float]] = {
    # differential_demod: "for pi/4-shifted quaternary that is psk_demod order
    # 8 -> differential_demod {rotate: -pi/4, out_order: 4} -> an order-4 demap"
    "pi4_quaternary_differential": (
        {
            "symbol_rate": 4800.0,
            "path": [
                {"conv": "agc", "mode": "feedback"},
                {"conv": "psk_demod", "order": 8, "loop_bw": 0.02},
                {
                    "conv": "differential_demod",
                    "delay": 1,
                    "rotate": -math.pi / 4,
                    "out_order": 4,
                },
                {"conv": "psk_demap", "order": 4},
            ],
        },
        19200.0,
    ),
    # differential_demod: plain M-DPSK differences stay order M (the default)
    "plain_dpsk_keeps_its_order": (
        {
            "symbol_rate": 4800.0,
            "path": [
                {"conv": "agc", "mode": "feedback"},
                {"conv": "psk_demod", "order": 4, "loop_bw": 0.02},
                {"conv": "differential_demod", "delay": 1},
                {"conv": "psk_demap", "order": 4},
            ],
        },
        19200.0,
    ),
    # symbol_sync: "This is psk_demod's timing half split out: it stays at IQ,
    # so an equalizer can sit at the 1-sps seam before carrier recovery."
    "symbol_sync_then_equalizer_at_one_sps": (
        {
            "symbol_rate": 4800.0,
            "path": [
                {"conv": "agc", "mode": "feedback"},
                {"conv": "symbol_sync", "sps": 4, "loop_bw": 0.0, "alpha": 0.35},
                {"conv": "equalizer", "num_taps": 15},
                {"conv": "sample_symbols"},
            ],
        },
        19200.0,
    ),
    # ook_envelope: "Open-loop needs no agc stage ... runs at sps>=1, so a
    # capture already at the symbol rate decodes with no resample stage"
    "ook_open_loop_at_one_sps_without_agc": (
        {
            "symbol_rate": 4800.0,
            "path": [
                {"conv": "ook_envelope", "loop_bw": 0.0},
                {"conv": "slice"},
            ],
        },
        4800.0,
    ),
    # sync_align: "Output is a contiguous stream of frame bodies with no
    # windows - follow with segment to re-tile it for window-scoped stages."
    "sync_align_then_harden_then_segment": (
        {
            "symbol_rate": 4800.0,
            "path": [
                {"conv": "fsk", "deviation": 2400.0, "loop_bw": 0.0},
                {"conv": "mfsk_soft_demap", "levels": [-1.0, 1.0]},
                {"conv": "sync_align", "access_code": "0101100011", "frame_len": 64},
                {"conv": "harden"},
                {"conv": "segment", "frame_body_len": 64},
            ],
        },
        19200.0,
    ),
    # run_rx docstring: "A path may also end BEFORE any demod, at conditioned
    # IQ (level 'iq', the same 'c' .cf32 wire) — e.g. channelize->agc with no
    # demod stage."
    "conditioned_iq_terminal": (
        {
            "symbol_rate": 4800.0,
            "path": [
                {"conv": "channelize", "decim": 4, "bandwidth_hz": 20000.0},
                {"conv": "agc", "mode": "power"},
            ],
        },
        192000.0,
    ),
}


@pytest.mark.parametrize("name", sorted(_DOCUMENTED))
def test_a_documented_composition_compiles(name: str) -> None:
    spec, sample_rate = _DOCUMENTED[name]
    result = validate_modem(spec, sample_rate=sample_rate)
    assert result["valid"], f"{name}: {result.get('errors')}"


def test_the_pi4_recipe_needs_its_out_order() -> None:
    """The bug this module exists for, pinned from the failing side: without
    out_order the differential keeps the order-8 grid it was handed and the
    order-4 demap below it cannot compile."""
    spec, sample_rate = _DOCUMENTED["pi4_quaternary_differential"]
    path = [dict(step) for step in spec["path"]]
    del path[2]["out_order"]
    result = validate_modem({**spec, "path": path}, sample_rate=sample_rate)
    assert not result["valid"]
    assert "order-8" in str(result["errors"])
