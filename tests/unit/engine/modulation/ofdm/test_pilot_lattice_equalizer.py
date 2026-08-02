from __future__ import annotations

import numpy as np
from helpers import _lattice
from helpers._fakegr import FAKE_GR, FakeTag, drive

from marconi.engine.backends.gnuradio.embedded.pilot_lattice import (
    make_pilot_lattice_equalizer,
)


def _evm(eq: np.ndarray, truth: np.ndarray) -> float:
    err = np.mean(np.abs(eq - truth) ** 2)
    return float(err / np.mean(np.abs(truth) ** 2))


def test_non_dc_straddling_span_rejected_at_construction() -> None:
    """The emit grid spans kmin..kmin+n_carriers minus exactly one DC skip;
    a span that misses DC would silently emit n_carriers+1 bins per symbol."""
    import pytest

    with pytest.raises(ValueError, match="straddle DC"):
        make_pilot_lattice_equalizer(FAKE_GR, **{**_lattice.eq_params(), "kmin": 4})


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
        assert blk._first <= blk._m
    k = 1
    while k:
        out = np.zeros(1 << 15, np.complex64)
        k = int(blk.general_work([spec[0:0]], [out]))
        blk._nwritten += k
        got.append(out[:k].copy())
        assert blk._first <= blk._m
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
        nodes = sum(len(ms) for ms in blk._node_ms)
        peak = max(peak, len(blk._vecs) + nodes + blk._out.size // 64)
    cap = 8 * _lattice.NS * (len(_lattice.EMIT) + _lattice.FFT_LEN)
    assert peak <= cap, f"state peaked at {peak} > {cap}"
    assert drained > 100 * len(_lattice.EMIT)
