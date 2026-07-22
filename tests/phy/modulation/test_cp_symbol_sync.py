from __future__ import annotations

import _lattice
import numpy as np
from phy._fakegr import FAKE_GR, drive

from marconi.phy.backends.gnuradio.embedded.cp_sync import make_cp_symbol_sync


def _blk():
    return make_cp_symbol_sync(
        FAKE_GR,
        fft_len=_lattice.FFT_LEN,
        cp_len=_lattice.CP_LEN,
        warmup_syms=12,
    )


def _hit_frac(vecs: np.ndarray, clean: np.ndarray) -> float:
    hits = 0
    for v in vecs:
        c = max(
            abs(np.vdot(t, v)) / (np.linalg.norm(t) * np.linalg.norm(v) + 1e-9)
            for t in clean
        )
        hits += c > 0.7  # 0.4-sample STO alone caps the resultant near 0.86
    return hits / max(len(vecs), 1)


def test_locks_and_emits_phase_coherent_aligned_symbols() -> None:
    iq = _lattice.make_iq(60, cfo_hz=9.0, snr_db=28.0, sto_frac=0.4, seed=3)
    blk = _blk()
    # chunk must stay <= buf_cap - need: drive() drops what the block leaves
    out = drive(blk, iq, chunk=320, out_dtype=np.complex64)
    assert blk.diagnostics["locks"] == 1
    vecs = out.reshape(-1, _lattice.FFT_LEN)
    assert len(vecs) >= 55
    assert _hit_frac(vecs, _lattice.clean_symbols(60, seed=3)) >= 0.95
    x = np.fft.fftshift(np.fft.fft(vecs.astype(np.complex128), axis=1), axes=1)
    dc0 = _lattice.FFT_LEN // 2
    fpv = np.array(_lattice.FP_VALUES)
    pil = x[:, [k + dc0 for k in _lattice.FP_CARRIERS]] * np.conj(fpv)[None, :]
    d = pil[1:] * np.conj(pil[:-1])
    coherence = float(np.abs(d.sum()) / np.abs(d).sum())
    assert coherence >= 0.9, f"inter-symbol phase coherence {coherence:.3f}"


def test_noise_never_locks() -> None:
    rng = np.random.default_rng(5)
    iq = 0.3 * (rng.standard_normal(20000) + 1j * rng.standard_normal(20000)).astype(
        np.complex64
    )
    blk = _blk()
    out = drive(blk, iq, chunk=1024, out_dtype=np.complex64)
    assert out.size == 0
    assert blk.diagnostics["locks"] == 0


def test_dropout_emits_sync_start_and_relocks() -> None:
    a = _lattice.make_iq(30, seed=7)
    gap = np.zeros(4000, np.complex64)  # EMA(0.05) needs ~35 dead symbols to trip
    b = _lattice.make_iq(30, seed=8)
    blk = _blk()
    out = drive(blk, np.concatenate([a, gap, b]), chunk=811, out_dtype=np.complex64)
    assert blk.diagnostics["locks"] == 2
    starts = [t for t in blk.out_tags if t.key == "sync_start"]
    assert starts
    assert all(t.offset % _lattice.FFT_LEN == 0 for t in starts)
    assert out.size > 0


def test_memory_bounded_under_slow_consumer() -> None:
    iq = _lattice.make_iq(400, seed=9)
    blk = _blk()
    pos = drained = stalls = peak = 0
    while pos < iq.size and stalls < 4096:
        before = blk.nitems_read(0)
        out = np.zeros(256, np.complex64)
        k = int(blk.general_work([iq[pos : pos + 8192]], [out]))
        blk._nwritten += k
        drained += k
        consumed = blk.nitems_read(0) - before
        pos += consumed
        stalls = stalls + 1 if (consumed == 0 and k == 0) else 0
        peak = max(peak, blk._buf.size + blk._out.size)
    k = 1
    while k:  # EOF flush: drain the pending queue across zero-input wakeups
        out = np.zeros(256, np.complex64)
        k = int(blk.general_work([iq[0:0]], [out]))
        blk._nwritten += k
        drained += k
    need = 12 * _lattice.SYM_LEN + _lattice.FFT_LEN + _lattice.CP_LEN
    bound = (need + 8 * _lattice.SYM_LEN) + (1 << 15) + _lattice.FFT_LEN
    assert peak <= bound, f"internal state peaked at {peak} > {bound}"
    assert drained > 100 * _lattice.FFT_LEN
