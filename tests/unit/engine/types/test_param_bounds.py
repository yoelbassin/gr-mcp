"""Every step param that sizes work declares a ceiling.

`ge=` alone made each of these a spec that validates, compiles, and then either
allocates until the machine dies or grinds past the deadline with nothing to
show. Probing the registry at 2**40 found twenty-odd of them accepting it — the
class was much bigger than the three a review had named, which is why this is a
sweep and not three assertions.

Physical quantities are deliberately exempt and listed by name below: a wrong
Hz, dB, ppm or loop bandwidth is cheap, and the compiler's rate model already
checks the ones that matter. Everything else must refuse 2**40.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from marconi.engine.stages.registry import step_models

# A minimal valid spec per stage, so exactly one field varies at a time.
_BASE: dict[str, dict[str, object]] = {
    "agc": {},
    "am": {},
    "analytic": {},
    "block_code": {"code_bits": 7, "data_bits": 4, "parity_masks": [7, 11, 13]},
    "burst_probe": {},
    "cell_select": {"select_perm": [0, 1], "keep": 1},
    "channelize": {"decim": 2, "bandwidth_hz": 1000.0},
    "chirp_sync": {
        "sf": 7,
        "oversample": 1,
        "zero_pad": 1,
        "preamble_len": 8,
        "sfd_symbols": 2.0,
        "sync_symbols": 2,
    },
    "clock_correct": {"ppm": 10.0},
    "codebook": {"code_bits": 2, "data_bits": 1, "table": [1, 2]},
    "css_demap": {"sf": 7},
    "dechirp": {"sf": 7, "oversample": 1, "zero_pad": 1},
    "deinterleave": {"perm": [0, 1]},
    "depuncture": {"keep_mask": [1, 0]},
    "descramble": {"sequence": "ff"},
    "differential": {},
    "differential_demod": {},
    "dqpsk_soft_demap": {"data_syms": 2, "n_carriers": 2},
    "equalizer": {},
    "fec": {"scheme": "cc", "rate_inv": 2, "polys": [91, 121], "frame_bits": 8},
    "fll": {},
    "fm_demod": {"deviation": 5000.0},
    "fsk": {"deviation": 1200.0},
    "harden": {},
    "invert": {},
    "ldpc": {"block_size": 4, "check_nodes": [[0, 1]]},
    "m_slice": {"thresholds": [0.0], "levels": [0, 1]},
    "mark_frame": {},
    "mfsk_soft_demap": {"levels": [-1.0, 1.0]},
    "msk": {},
    "nibble_swap": {},
    "normalize": {"span_symbols": 8},
    "ofdm_coherent_sync": {
        "fft_len": 8,
        "cp_len": 2,
        "sym_len": 10,
        "n_frame_syms": 2,
        "n_carriers": 4,
        "kmin": -2,
        "dc_search": 0,
        "warmup_syms": 3,
        "pilot_lens": [1, 1],
        "pilot_carriers": [0, 1],
        "pilot_i": [1.0, 1.0],
        "pilot_q": [0.0, 0.0],
        "fp_carriers": [0],
        "fp_i": [1.0],
        "fp_q": [0.0],
    },
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
    "ofdm_frame_sync_probe": {
        "fft_len": 4,
        "cp_len": 1,
        "sym_len": 5,
        "null_len": 2,
        "frame_len": 100,
        "data_syms": 2,
    },
    "ook_envelope": {},
    "permute": {"perm": [0, 1]},
    "polar": {
        "block_size": 4,
        "info_bits": 2,
        "frozen_positions": [0, 1],
        "frozen_values": [0, 0],
    },
    "preamble_sync": {"preamble_i": [1.0, -1.0], "preamble_q": [0.0, 0.0]},
    "psk_demap": {"order": 4},
    "psk_demod": {"order": 4},
    "psk_soft_demap": {"order": 4},
    "qam_demap": {"order": 16},
    "qam_demod": {"order": 16},
    "realign": {"bit_offset": 0},
    "resample": {"interpolation": 1, "decimation": 1},
    "rs_code": {"symbol_bits": 8, "n": 255, "k": 223, "prim_poly": 285},
    "sample_symbols": {},
    "segment": {"frame_body_len": 8},
    "slice": {},
    "soft_demap": {"scheme": "psk", "order": 4},
    "squelch": {"threshold_db": -30.0},
    "symbol_map": {"code_bits": 2, "data_bits": 1, "table": [1, 2]},
    "symbol_sync": {"sps": 4},
    "sync_align": {"access_code": "0101", "frame_len": 8},
    "sync_symbols": {"pattern": [1, -1]},
    "sync_word": {"sync": "9162"},
    "translate": {"center_hz": 0.0},
}

# Fields exempt from the ceiling, each because the value costs nothing to be
# wrong about. Every entry is a physical quantity or a dimensionless loop/
# threshold constant that the compiler's rate model or the block itself checks.
# An entry here is a claim: "a huge value of this allocates nothing and
# iterates nothing."
_PHYSICAL: frozenset[str] = frozenset(
    {
        "agc.reference",
        "channelize.bandwidth_hz",
        "channelize.center_hz",
        "clock_correct.ppm",
        "differential_demod.rotate",
        "equalizer.modulus",
        "equalizer.step_size",
        "fll.loop_bw",
        "fm_demod.deviation",
        "fsk.deviation",
        "fsk.loop_bw",
        "mark_frame.offset_bits",
        "msk.loop_bw",
        "ofdm_coherent_sync.lock_min_ratio",
        "ofdm_coherent_sync.lock_min_score",
        "ook_envelope.loop_bw",
        "preamble_sync.threshold",
        "psk_demod.loop_bw",
        "qam_demod.loop_bw",
        "realign.bit_offset",
        "squelch.threshold_db",
        "symbol_sync.loop_bw",
        "translate.center_hz",
    }
)

_HUGE = 1 << 40


def _numeric_fields() -> list[tuple[str, str, bool]]:
    out: list[tuple[str, str, bool]] = []
    for conv, model in sorted(step_models().items()):
        for name, info in model.model_fields.items():
            if name == "conv":
                continue
            ann = str(info.annotation)
            if "list" in ann or "bool" in ann:
                continue
            is_int = "int" in ann
            if not is_int and "float" not in ann:
                continue
            out.append((conv, name, is_int))
    return out


def test_every_registered_stage_has_a_base_spec() -> None:
    """A stage missing from _BASE would silently drop out of the sweep."""
    missing = sorted(set(step_models()) - set(_BASE))
    assert not missing, f"stages with no base spec in this sweep: {missing}"
    models = step_models()
    for conv, base in _BASE.items():
        models[conv].model_validate({"conv": conv, **base})


@pytest.mark.parametrize("conv, field, is_int", _numeric_fields(), ids=lambda v: str(v))
def test_a_cost_sizing_param_refuses_an_absurd_value(
    conv: str, field: str, is_int: bool
) -> None:
    key = f"{conv}.{field}"
    probe: object = _HUGE if is_int else float(_HUGE)
    try:
        step_models()[conv].model_validate({"conv": conv, **_BASE[conv], field: probe})
    except ValidationError:
        assert key not in _PHYSICAL, (
            f"{key} is listed as a physical quantity but now refuses a large "
            "value; drop it from _PHYSICAL"
        )
        return
    assert key in _PHYSICAL, (
        f"{key} accepts 2**40. If it sizes an allocation, a filter, or an "
        "iteration count, give it a ceiling from engine/types/bounds.py. If a "
        "huge value genuinely costs nothing, add it to _PHYSICAL with that "
        "claim in mind."
    )


def test_a_correlator_tolerance_cannot_exceed_its_pattern() -> None:
    """Unbounded, this reached _chance_valid_rate's range(t+1) sphere-volume
    sum — a pure-Python loop with no deadline check — and the first term past
    the pattern length raised out of math.lgamma, so the agent got
    "expected a noninteger or positive integer, got 0.0" from inside the
    coding lane instead of a word about its own spec."""
    models = step_models()
    models["sync_word"].model_validate({"conv": "sync_word", "sync": "9162"})
    with pytest.raises(ValidationError, match="16-bit pattern"):
        models["sync_word"].model_validate(
            {"conv": "sync_word", "sync": "9162", "max_errors": 16}
        )
    with pytest.raises(ValidationError, match="pattern"):
        models["sync_symbols"].model_validate(
            {"conv": "sync_symbols", "pattern": [1, -1], "max_errors": 2}
        )


def test_the_dechirp_fft_is_bounded_by_its_product_not_its_fields() -> None:
    """sf and oversample are capped individually and zero_pad has no ceiling
    that would express the FFT the triple asks for."""
    models = step_models()
    models["dechirp"].model_validate(
        {"conv": "dechirp", "sf": 14, "oversample": 8, "zero_pad": 8}
    )
    with pytest.raises(ValidationError, match="point FFT per symbol window"):
        models["dechirp"].model_validate(
            {"conv": "dechirp", "sf": 14, "oversample": 8, "zero_pad": 64}
        )
