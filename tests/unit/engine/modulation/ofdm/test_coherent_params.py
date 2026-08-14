from __future__ import annotations

import pytest
from helpers import _lattice
from pydantic import ValidationError

from marconi.engine.compile.compile_context import CompileContext
from marconi.engine.modulation.ofdm.stages import (
    OfdmCoherentSync,
    OfdmCoherentSyncStep,
)
from marconi.engine.types.descriptor import Descriptor
from marconi.engine.types.enums import ItemType
from marconi.engine.types.levels import Level


def _good() -> dict[str, object]:
    return {
        "fft_len": 64,
        "cp_len": 16,
        "sym_len": 80,
        "n_frame_syms": 4,
        "n_carriers": 48,
        "kmin": -24,
        "dc_search": 2,
        "warmup_syms": 12,
        "pilot_lens": [1, 1, 1, 1],
        "pilot_carriers": [-24, -23, -22, -21],
        "pilot_i": [1.0, 1.0, 1.0, 1.0],
        "pilot_q": [0.0, 0.0, 0.0, 0.0],
        "fp_carriers": [3],
        "fp_i": [1.0],
        "fp_q": [0.0],
    }


def test_valid_params_pass() -> None:
    assert OfdmCoherentSyncStep.model_validate(_good()).fft_len == 64


def test_edge_pilot_within_dc_search_reach_rejected() -> None:
    # kmin=-fft_len/2 with any dc_search >= 1: the searched delta can push the
    # edge pilot's bin below 0 — reject at validation, not crash at frame 1
    bad = {
        **_good(),
        "kmin": -32,
        "n_carriers": 56,
        "pilot_carriers": [-32, -23, -22, -21],
    }
    with pytest.raises(ValidationError, match="FFT"):
        OfdmCoherentSyncStep.model_validate(bad)


def test_lock_thresholds_default_to_calibrated_values() -> None:
    step = OfdmCoherentSyncStep.model_validate(_good())
    assert step.lock_min_ratio == 2.0
    assert step.lock_min_score == 0.35


def test_emit_rx_passes_lock_thresholds_to_their_blocks() -> None:
    b = CompileContext(Descriptor(Level.IQ, ItemType.C), rate=1.0, symbol_rate=1.0)
    step = OfdmCoherentSyncStep(
        **_lattice.sync_params(), lock_min_ratio=1.5, lock_min_score=0.5
    )
    OfdmCoherentSync().emit_rx(b, step)
    p = b.build("t", 1.0)
    cp = next(x for x in p.blocks if x.kind == "cp_symbol_sync")
    assert cp.params["lock_min_ratio"] == 1.5
    eqz = next(x for x in p.blocks if x.kind == "pilot_lattice_equalizer")
    assert eqz.params["lock_min_score"] == 0.5


@pytest.mark.parametrize(
    "patch",
    [
        {"sym_len": 81},
        {"pilot_lens": [2, 1, 1]},  # sum stays 4: isolates the length check
        {"pilot_carriers": [-24, -23, -22]},
        {"pilot_i": [1.0, 1.0, 1.0]},
        {"fp_i": [1.0, 2.0]},
        {"n_carriers": 0},
        {"warmup_syms": 4},
        {"kmin": 5},  # span 5..53 excludes DC: emit grid gains a bin
        {"kmin": -60},  # span -60..-12 excludes DC
    ],
)
def test_inconsistent_geometry_rejected(patch: dict[str, object]) -> None:
    with pytest.raises(ValidationError):
        OfdmCoherentSyncStep.model_validate({**_good(), **patch})


def test_zero_lock_bars_are_rejected() -> None:
    # ge=0.0 admitted 0.0, and "ratio < 0.0" is never true: a block with a
    # zero bar locks unconditionally - on pure AWGN, on anything. The sibling
    # tag_gate raises for exactly this ("zero clears every significance bar").
    for field in ("lock_min_ratio", "lock_min_score"):
        with pytest.raises(ValidationError):
            OfdmCoherentSyncStep.model_validate({**_good(), field: 0.0})


def test_warmup_below_the_calibration_geometry_is_rejected() -> None:
    # LOCK_MIN_RATIO_DEFAULT = 2.0 was measured at warmup_syms >= 8; the
    # ratio is a max over sym_len offsets against the median of n_grid =
    # warmup_syms - 1 lattice sums, so its noise distribution moves with
    # warmup: measured false-lock rates on pure AWGN at the 2.0 bar were
    # 93/100 at warmup 2, 16/100 at 4, 0/100 at 8.
    with pytest.raises(ValidationError):
        OfdmCoherentSyncStep.model_validate({**_good(), "warmup_syms": 7})
    assert OfdmCoherentSyncStep.model_validate({**_good(), "warmup_syms": 8})
