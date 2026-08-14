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

import inspect
import math
import re
from typing import Any

import pytest

from marconi.engine.coding.stages_bits import DescrambleStep
from marconi.mcp.tools import describe_stages, validate_modem

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
    # msk: "slice then differential is the composition that decodes
    # differentially-encoded MSK polarity-free"
    "msk_slice_differential": (
        {
            "symbol_rate": 4800.0,
            "path": [
                {"conv": "msk"},
                {"conv": "slice"},
                {"conv": "differential"},
            ],
        },
        19200.0 * 5,
    ),
    # fsk: "follow with slice ... or mfsk_soft_demap (levels from
    # stream_stats centers)"
    "fsk_mfsk_soft_demap": (
        {
            "symbol_rate": 4800.0,
            "path": [
                {"conv": "fsk", "deviation": 2400.0},
                {"conv": "mfsk_soft_demap", "levels": [-1.0, 1.0]},
            ],
        },
        19200.0,
    ),
    # m_slice: "the 4-FSK/multi-level entry to symbol_map"; sync_symbols:
    # "marks survive the symbols->bits stage, and mark_frame then turns them
    # into windows THERE, at bits level"
    "sync_symbols_mslice_symbol_map_mark_frame": (
        {
            "symbol_rate": 4800.0,
            "path": [
                {"conv": "fsk", "deviation": 2400.0},
                {"conv": "sync_symbols", "pattern": [1, -1, 1, 1, -1, -1, 1, -1]},
                {
                    "conv": "m_slice",
                    "thresholds": [-1.0, 0.0, 1.0],
                    "levels": [0, 1, 2, 3],
                },
                {
                    "conv": "symbol_map",
                    "code_bits": 2,
                    "data_bits": 2,
                    "table": [0, 1, 2, 3],
                },
                {"conv": "mark_frame", "offset_bits": 0},
            ],
        },
        19200.0,
    ),
    # chirp_sync: "follow with dechirp"; dechirp: "run chirp_sync first";
    # css_demap: "matching dechirp's alphabet"
    "chirp_sync_dechirp_css_demap": (
        {
            "symbol_rate": 100.0,
            "path": [
                {
                    "conv": "chirp_sync",
                    "sf": 7,
                    "oversample": 2,
                    "zero_pad": 4,
                    "preamble_len": 8,
                    "sfd_symbols": 2.25,
                    "sync_symbols": 2,
                },
                {"conv": "dechirp", "sf": 7, "oversample": 2, "zero_pad": 4},
                {"conv": "css_demap", "sf": 7},
            ],
        },
        2 * (1 << 7) * 100.0,
    ),
    # qam_demod: "pair with agc mode='power'" -> qam_demap
    "agc_power_qam": (
        {
            "symbol_rate": 4800.0,
            "path": [
                {"conv": "agc", "mode": "power"},
                {"conv": "qam_demod", "order": 16},
                {"conv": "qam_demap", "order": 16},
            ],
        },
        19200.0,
    ),
    # ofdm_demod: "Pins the frame at (data_syms + 1) * n_carriers cells so the
    # downstream demap's geometry is checked" / dqpsk_soft_demap:
    # "data_syms/n_carriers ... the compiler checks them against the upstream
    # OFDM frame"
    "ofdm_demod_into_a_matched_dqpsk_demap": (
        {
            "symbol_rate": 1000.0,
            "path": [
                {
                    "conv": "ofdm_demod",
                    "fft_len": 8,
                    "cp_len": 2,
                    "sym_len": 10,
                    "null_len": 4,
                    "frame_len": 34,
                    "data_syms": 2,
                    "n_carriers": 2,
                    "bin_perm": list(range(8)),
                },
                {"conv": "dqpsk_soft_demap", "data_syms": 2, "n_carriers": 2},
            ],
        },
        1_000_000.0,
    ),
    # fm_demod: "follow with analytic, then translate to the subcarrier"
    "fm_analytic_translate": (
        {
            "symbol_rate": 1187.5,
            "path": [
                {"conv": "fm_demod", "deviation": 75000.0},
                {"conv": "analytic"},
                {"conv": "translate", "center_hz": 57000.0},
            ],
        },
        192000.0,
    ),
    # ook_envelope: "closed-loop needs normalized amplitude and sps>=4 - pair
    # it with an agc"
    "ook_closed_loop_at_its_floor": (
        {
            "symbol_rate": 4800.0,
            "path": [
                {"conv": "agc", "mode": "feedforward"},
                {"conv": "ook_envelope", "loop_bw": 0.045},
                {"conv": "slice"},
            ],
        },
        19200.0,
    ),
    # descramble: "supply the whitening sequence explicitly as hex (sequence)
    # or have the stage generate it from a Fibonacci LFSR (lfsr_mask/
    # lfsr_seed/lfsr_len), exactly one of the two"
    "descramble_from_an_explicit_sequence": (
        {
            "symbol_rate": 4800.0,
            "path": [
                {"conv": "fsk", "deviation": 2400.0},
                {"conv": "slice"},
                {"conv": "descramble", "sequence": "a5c3"},
            ],
        },
        19200.0,
    ),
    "descramble_from_an_lfsr": (
        {
            "symbol_rate": 4800.0,
            "path": [
                {"conv": "fsk", "deviation": 2400.0},
                {"conv": "slice"},
                {
                    "conv": "descramble",
                    "lfsr_mask": 0x41,
                    "lfsr_seed": 0x7F,
                    "lfsr_len": 7,
                },
            ],
        },
        19200.0,
    ),
}


def _symbol_sync_spec(sps: int, loop_bw: float) -> tuple[dict[str, Any], float]:
    return (
        {
            "symbol_rate": 4800.0,
            "path": [
                {"conv": "agc", "mode": "feedback"},
                {"conv": "symbol_sync", "sps": sps, "loop_bw": loop_bw, "alpha": 0.35},
                {"conv": "sample_symbols"},
            ],
        },
        4800.0 * sps,
    )


def _smallest_compiling_sps(loop_bw: float) -> int:
    for sps in range(1, 9):
        spec, sample_rate = _symbol_sync_spec(sps, loop_bw)
        if validate_modem(spec, sample_rate=sample_rate)["valid"]:
            return sps
    raise AssertionError(f"symbol_sync compiles at no sps 1..8 at loop_bw={loop_bw}")


_FLOOR_CLAIM = re.compile(r"(\d+)(?: samples per symbol)? (open-loop|closed-loop)")


def _claimed_floors(text: str) -> dict[str, int]:
    return {
        mode: int(value) for value, mode in _FLOOR_CLAIM.findall(" ".join(text.split()))
    }


@pytest.mark.parametrize("name", sorted(_DOCUMENTED))
def test_a_documented_composition_compiles(name: str) -> None:
    spec, sample_rate = _DOCUMENTED[name]
    result = validate_modem(spec, sample_rate=sample_rate)
    assert result["valid"], f"{name}: {result.get('errors')}"


def test_symbol_syncs_two_agent_texts_quote_the_floors_the_compiler_enforces() -> None:
    """describe_stages said "4 samples per symbol open-loop and no floor
    closed-loop" while the description it delegates to said "2 closed-loop, 4
    open-loop", and the compiler agreed with the second: closed-loop sps=1 is
    refused by the step model. Both texts are read back against the floors the
    compiler actually applies, so neither can drift alone."""
    measured = {
        "open-loop": _smallest_compiling_sps(0.0),
        "closed-loop": _smallest_compiling_sps(0.045),
    }
    rows = describe_stages(stage="symbol_sync")["stages"]
    assert isinstance(rows, list)
    published = str(rows[0]["description"])
    assert _claimed_floors(published) == measured, published
    assert _claimed_floors(inspect.getdoc(describe_stages) or "") == measured


def test_the_closed_loop_ook_floor_refuses_the_rate_one_step_below_it() -> None:
    """The field text offered sps>=2 closed-loop; the floor is 4, from the
    measurement that Gardner's TED is degenerate on a rectangular envelope."""
    spec, sample_rate = _DOCUMENTED["ook_closed_loop_at_its_floor"]
    result = validate_modem(spec, sample_rate=sample_rate / 2.0)
    assert not result["valid"]
    assert "4 samples per symbol" in str(result["errors"])


def test_the_descramble_description_names_every_input_its_validator_accepts() -> None:
    """The description described one branch and prescribed a workaround for
    the other ("for LFSR self-synchronizing scramblers, supply the expanded
    sequence") — wrong on its own terms, since the lfsr_* mode is additive and
    a multiplicative self-synchronizing descrambler is data-dependent with no
    fixed sequence. descramble's parameters are not tunables but its two
    exclusive MODES, and the index row is where the agent picks a stage."""
    rows = describe_stages(stage="descramble")["stages"]
    assert isinstance(rows, list)
    published = str(rows[0]["description"])
    modes = set(DescrambleStep.model_fields) - {"conv"}
    assert modes
    missing = sorted(name for name in modes if name not in published)
    assert not missing, f"{missing} not named in: {published}"


def test_descramble_takes_exactly_one_of_the_two_modes_it_advertises() -> None:
    """The description named only the sequence branch, so the mode the stage
    ships had to be discovered a level down in the schema. Both modes compile;
    neither both nor neither does."""
    spec, sample_rate = _DOCUMENTED["descramble_from_an_lfsr"]
    both = [dict(step) for step in spec["path"]]
    both[-1]["sequence"] = "a5c3"
    assert not validate_modem({**spec, "path": both}, sample_rate=sample_rate)["valid"]
    neither = [dict(step) for step in spec["path"]]
    neither[-1] = {"conv": "descramble"}
    result = validate_modem({**spec, "path": neither}, sample_rate=sample_rate)
    assert not result["valid"]
    assert "exactly one" in str(result["errors"])


def test_the_ofdm_demap_recipe_is_checked_against_the_frame_above_it() -> None:
    """The other half of the OFDM entry's claim: the check the description
    advertises has to REFUSE the mismatch, not merely pass the match. It did
    not — the demod pinned no frame, so a demap cutting 42-cell frames behind
    a 6-cell one compiled clean."""
    spec, sample_rate = _DOCUMENTED["ofdm_demod_into_a_matched_dqpsk_demap"]
    path = [dict(step) for step in spec["path"]]
    path[1]["data_syms"] = 5
    result = validate_modem({**spec, "path": path}, sample_rate=sample_rate)
    assert not result["valid"]
    assert "12" in str(result["errors"]) and "6" in str(result["errors"])


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
