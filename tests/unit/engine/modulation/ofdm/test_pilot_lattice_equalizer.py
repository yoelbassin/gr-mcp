from __future__ import annotations

import numpy as np
import pytest
from helpers import _lattice
from helpers._fakegr import FAKE_GR, FakeTag, drive

from marconi.engine.backends.gnuradio.embedded.pilot_lattice import (
    make_pilot_lattice_equalizer,
)
from marconi.engine.modulation.ofdm.stages import OfdmCoherentSyncStep
from marconi.engine.stages.registry import stage_registry
from marconi.engine.types.descriptor import Descriptor
from marconi.engine.types.enums import ItemType
from marconi.engine.types.levels import Level


def _evm(eq: np.ndarray, truth: np.ndarray) -> float:
    err = np.mean(np.abs(eq - truth) ** 2)
    return float(err / np.mean(np.abs(truth) ** 2))


def test_non_dc_straddling_span_rejected_at_construction() -> None:
    """The emit grid spans kmin..kmin+n_carriers minus exactly one DC skip;
    a span that misses DC would silently emit n_carriers+1 bins per symbol."""
    with pytest.raises(ValueError, match="straddle DC"):
        make_pilot_lattice_equalizer(FAKE_GR, **{**_lattice.eq_params(), "kmin": 4})


def test_pilot_bins_beyond_fft_reach_rejected_at_construction() -> None:
    """A dc_search wide enough to push an edge pilot's bin out of the FFT
    would starve that carrier's channel node forever (np.interp over an empty
    node kills the flowgraph on the first frame) and wrap negative indices to
    the opposite spectral edge inside the CFO estimate."""
    with pytest.raises(ValueError, match=r"bins \[-2, 66\] vs fft_len 64"):
        make_pilot_lattice_equalizer(
            FAKE_GR, **{**_lattice.eq_params(), "dc_search": 10}
        )


def test_the_carrier_past_the_widest_span_is_rejected_at_construction() -> None:
    """The TOP emitted carrier is kmin+n_carriers — emit skips DC, it does not
    stop short — and this seam's own copy of the bound stopped one bin below
    it. One carrier past the widest span a 64-bin FFT holds, the block built,
    locked, and then indexed bin 64 out of _equalize_frame. The stage
    validator refuses this too; both now ask the same expression."""
    # kmin=-31 puts the top carrier on an odd offset, which the pilot rule
    # never lands on: the emitted carrier is then the ONLY term that reaches
    # bin 64, so this fails the moment the bound stops one below it.
    past = _lattice.Lattice(kmin=-31, n_carriers=63, dc_search=0)
    with pytest.raises(ValueError, match=r"bins \[1, 64\] vs fft_len 64"):
        make_pilot_lattice_equalizer(FAKE_GR, **past.eq_params())
    widest = _lattice.Lattice(kmin=-32, n_carriers=63, dc_search=0)
    blk = make_pilot_lattice_equalizer(FAKE_GR, **widest.eq_params())
    _, spec = widest.make_spectra(40, seed=3)
    out = drive(blk, spec, chunk=5, out_dtype=np.complex64)
    assert out.size and out.size % widest.n_carriers == 0


def test_locks_and_equalizes_with_frame_phase_discovery() -> None:
    start_fs = 2
    grid, spec = _lattice.make_spectra(60, start_fs=start_fs, theta=0.01, seed=3)
    blk = make_pilot_lattice_equalizer(FAKE_GR, **_lattice.eq_params())
    out = drive(blk, spec, chunk=5, out_dtype=np.complex64)
    assert blk.diagnostics["locks"] == 1
    ncell = len(_lattice.EMIT)
    eq = out.reshape(-1, ncell)
    phi = (_lattice.NS - start_fs) % _lattice.NS
    truth = grid[phi : phi + len(eq)]
    assert _evm(eq, truth) < 0.05
    assert blk.diagnostics["frames_emitted"] * _lattice.NS == len(eq)
    # the stage's compile-time frame pin is this measured per-frame emit; a
    # downstream fixed-geometry consumer checks itself against that number
    pinned = stage_registry()["ofdm_coherent_sync"].out_descriptor(
        Descriptor(Level.IQ, ItemType.C),
        OfdmCoherentSyncStep(**_lattice.sync_params()),
    )
    assert pinned.frame_len == out.size // blk.diagnostics["frames_emitted"]


def test_noise_never_locks_or_emits() -> None:
    rng = np.random.default_rng(9)
    spec = (
        rng.standard_normal((60, _lattice.FFT_LEN))
        + 1j * rng.standard_normal((60, _lattice.FFT_LEN))
    ).astype(np.complex64)
    blk = make_pilot_lattice_equalizer(FAKE_GR, **_lattice.eq_params())
    out = drive(blk, spec, chunk=6, out_dtype=np.complex64)
    assert out.size == 0
    assert blk.diagnostics["locks"] == 0


def test_lock_min_score_param_gates_acquisition() -> None:
    # the synthetic-calibrated 0.35 gate is a caller param: an unreachable
    # threshold (score is a correlation <= 1) refuses to lock a clean signal.
    _, spec = _lattice.make_spectra(60, start_fs=2, theta=0.01, seed=3)
    blk = make_pilot_lattice_equalizer(
        FAKE_GR, **_lattice.eq_params(), lock_min_score=1.5
    )
    out = drive(blk, spec, chunk=5, out_dtype=np.complex64)
    assert blk.diagnostics["locks"] == 0
    assert out.size == 0


def test_sync_start_tag_forces_relock() -> None:
    _, a = _lattice.make_spectra(40, seed=4)
    _, b = _lattice.make_spectra(40, seed=5)
    spec = np.concatenate([a, b])
    blk = make_pilot_lattice_equalizer(FAKE_GR, **_lattice.eq_params())
    blk.in_tags = [FakeTag(40, "sync_start", 1)]
    out = drive(blk, spec, chunk=7, out_dtype=np.complex64)
    assert blk.diagnostics["locks"] == 2
    assert blk.diagnostics["relocks"] == 1
    assert out.size > 0


def test_relock_during_undrained_backlog_stays_sane() -> None:
    _, a = _lattice.make_spectra(30, seed=4)
    _, b = _lattice.make_spectra(60, seed=5)
    spec = np.concatenate([a, b])
    blk = make_pilot_lattice_equalizer(FAKE_GR, **_lattice.eq_params())
    blk.in_tags = [FakeTag(30, "sync_start", 1)]
    pos = 0
    while pos < 35:  # feed through the tag with NO output capacity
        out = np.zeros(0, np.complex64)
        before = blk.nitems_read(0)
        blk.general_work([spec[pos : pos + 5]], [out])
        pos += blk.nitems_read(0) - before
    got = []
    stalls = 0
    while pos < len(spec) and stalls < 64:
        out = np.zeros(1 << 15, np.complex64)
        before = blk.nitems_read(0)
        k = int(blk.general_work([spec[pos : pos + 5]], [out]))
        blk._nwritten += k
        got.append(out[:k].copy())
        consumed = blk.nitems_read(0) - before
        pos += consumed
        stalls = stalls + 1 if (consumed == 0 and k == 0) else 0
        assert blk._core._first <= blk._core._m
    k = 1
    while k:
        out = np.zeros(1 << 15, np.complex64)
        k = int(blk.general_work([spec[0:0]], [out]))
        blk._nwritten += k
        got.append(out[:k].copy())
        assert blk._core._first <= blk._core._m
    total = np.concatenate(got)
    assert blk.diagnostics["locks"] == 2
    assert blk.diagnostics["relocks"] == 1
    assert total.size > 0 and total.size % len(_lattice.EMIT) == 0


def test_memory_bounded_under_slow_consumer() -> None:
    _, spec = _lattice.make_spectra(600, seed=6)
    blk = make_pilot_lattice_equalizer(FAKE_GR, **_lattice.eq_params())
    pos = drained = stalls = peak = 0
    while pos < len(spec) and stalls < 4096:
        before = blk.nitems_read(0)
        out = np.zeros(64, np.complex64)
        k = int(blk.general_work([spec[pos : pos + 50]], [out]))
        blk._nwritten += k
        drained += k
        consumed = blk.nitems_read(0) - before
        pos += consumed
        stalls = stalls + 1 if (consumed == 0 and k == 0) else 0
        nodes = sum(len(ms) for ms in blk._core._node_ms)
        peak = max(peak, len(blk._core._vecs) + nodes + blk._out.size // 64)
    cap = 8 * _lattice.NS * (len(_lattice.EMIT) + _lattice.FFT_LEN)
    assert peak <= cap, f"state peaked at {peak} > {cap}"
    assert drained > 100 * len(_lattice.EMIT)


def test_lock_statistic_reaches_the_quality_verdict() -> None:
    # The equalizer computes a decode-grade lock statistic and gates
    # acquisition on it. Published under a name no extractor reads, it was
    # harvested, carried through the payload and silently dropped, leaving
    # ofdm_coherent_sync to score on cp_symbol_sync's evidence alone.
    from marconi.engine.backends.base import Diagnostic, DiagnosticKey
    from marconi.engine.quality import lock_evidence

    _, spec = _lattice.make_spectra(60, start_fs=2, theta=0.01, seed=3)
    blk = make_pilot_lattice_equalizer(FAKE_GR, **_lattice.eq_params())
    drive(blk, spec, chunk=5, out_dtype=np.complex64)

    rows = [
        Diagnostic(block="eq[0]", key=str(k), value=float(v))
        for k, v in blk.diagnostics.items()
        if k in (DiagnosticKey.LOCK_SCORE_BEST, DiagnosticKey.LOCK_SCORE_MIN)
    ]
    assert len(rows) == 2, sorted(blk.diagnostics)
    evidence = lock_evidence(rows)
    assert [(e.metric, e.assessment) for e in evidence] == [
        ("ofdm_pilot_score", "positive")
    ]
    # decode grade: the score is measured on the decoded lattice itself,
    # unlike cp_symbol_sync's pre-decision timing peak
    assert evidence[0].tier == "decode"
