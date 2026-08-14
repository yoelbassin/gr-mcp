from __future__ import annotations

import numpy as np
import pytest
from helpers import _lattice
from helpers._fakegr import FAKE_GR, drive
from pydantic import ValidationError

from marconi.engine.backends.gnuradio.embedded.pilot_lattice import (
    make_pilot_lattice_equalizer,
)
from marconi.engine.compile.compile_context import CompileContext
from marconi.engine.modulation.ofdm.stages import (
    OfdmCoherentSync,
    OfdmCoherentSyncStep,
)
from marconi.engine.types.bounds import MAX_FRAME_ITEMS
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


def test_negative_dc_search_is_rejected() -> None:
    # range(-dc_search, dc_search+1) is EMPTY for a negative value, so the
    # out-of-FFT guard inverted (accepting pilots the positive twin rejects)
    # and _try_lock crashed on argmin of an empty sequence
    with pytest.raises(ValidationError):
        OfdmCoherentSyncStep.model_validate({**_good(), "dc_search": -8})


def test_a_carrier_span_wider_than_its_fft_is_rejected() -> None:
    # This geometry was BLESSED here as "the legal full-FFT span". It is
    # structurally impossible: the emit grid runs kmin..kmin+n_carriers with
    # DC skipped, so 64 carriers from -32 need bin positions -32..+32 — 65 of
    # them, in an FFT that has 64. The blessing test validated the spec and
    # never ran the equalizer; driven, it locks at delta=0 and _equalize_frame
    # raises "index 64 is out of bounds" on frame 1. The synthetic side cannot
    # even build a spectrum for it.
    with pytest.raises(ValidationError, match="FFT"):
        OfdmCoherentSyncStep.model_validate(
            {**_good(), "kmin": -32, "n_carriers": 64, "dc_search": 0}
        )


def test_the_top_emitted_carrier_is_inside_the_dc_search_reach() -> None:
    # The bound used to stop at kmin + n_carriers - 1, one below the carrier
    # the block actually emits. At dc_search=1 this geometry validated, built,
    # locked on a genuinely +1-offset signal and then indexed bin 64 of a
    # 64-bin FFT — a worker IndexError from a spec validate_modem called valid.
    with pytest.raises(ValidationError, match="FFT"):
        OfdmCoherentSyncStep.model_validate(
            {**_good(), "kmin": -31, "n_carriers": 62, "dc_search": 1}
        )


_WIDEST = _lattice.Lattice(kmin=-32, n_carriers=63, dc_search=0)


def test_the_widest_admissible_span_locks_and_equalizes() -> None:
    # The replacement for the blessing above, and the other half of the same
    # claim: what the validator admits, the block survives. 63 carriers from
    # -32 occupy every bin of the 64-point FFT except DC — one more carrier or
    # one bin of DC search and the geometry is refused.
    step = OfdmCoherentSyncStep.model_validate(_WIDEST.sync_params())
    assert step.n_carriers == 63
    grid, spec = _WIDEST.make_spectra(60, theta=0.01, seed=3)
    blk = make_pilot_lattice_equalizer(FAKE_GR, **_WIDEST.eq_params())
    out = drive(blk, spec, chunk=5, out_dtype=np.complex64)
    assert blk.diagnostics["locks"] == 1
    eq = out.reshape(-1, step.n_carriers)
    assert len(eq) > 0
    err = np.mean(np.abs(eq - grid[: len(eq)]) ** 2) / np.mean(
        np.abs(grid[: len(eq)]) ** 2
    )
    assert err < 0.05


@pytest.mark.parametrize(
    "patch, reason",
    [
        # asks fft_vcc for a window measured in hundreds of GiB, with sym_len
        # co-varied so the equality check that used to answer for it passes
        ({"fft_len": 1 << 36, "sym_len": (1 << 36) + 16}, "acquisition"),
        # cp_symbol_sync's acquisition dies on a numpy broadcast mismatch
        ({"cp_len": -16, "sym_len": 48}, r"(?s)cp_len.*greater than or equal to 0"),
        # structurally dead block: the phase search loops over range(0), its
        # score stays -1.0, no lock_min_score is ever cleared, and nothing is
        # emitted at all under status ok
        (
            {
                "n_frame_syms": 0,
                "pilot_lens": [],
                "pilot_carriers": [],
                "pilot_i": [],
                "pilot_q": [],
            },
            r"(?s)n_frame_syms.*greater than or equal to 1",
        ),
    ],
)
def test_a_geometry_the_worker_cannot_survive_is_refused_at_the_spec(
    patch: dict[str, object], reason: str
) -> None:
    # Every one of these validated, compiled, and reached the GR worker, where
    # the run deadline — a parent-process contextvar — cannot reap it. The
    # sibling OfdmDemodStep refused all three at the spec.
    with pytest.raises(ValidationError, match=reason):
        OfdmCoherentSyncStep.model_validate({**_good(), **patch})


def test_the_acquisition_buffer_is_bounded_by_the_frame_cap() -> None:
    # cp_symbol_sync buffers warmup_syms * sym_len + fft_len + cp_len samples
    # before it can even attempt a lock; no single field's ceiling expresses
    # that product, and warmup_syms' own cap is far above what a large sym_len
    # can afford.
    sym = MAX_FRAME_ITEMS // 9
    geom = {**_good(), "fft_len": sym - 16, "sym_len": sym}
    assert OfdmCoherentSyncStep.model_validate({**geom, "warmup_syms": 8})
    with pytest.raises(ValidationError, match="acquisition"):
        OfdmCoherentSyncStep.model_validate({**geom, "warmup_syms": 9})
