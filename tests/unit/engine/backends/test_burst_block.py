"""burst_sampler: open-loop decimation with per-burst phase acquisition.
Scheduler-free via FAKE_GR/drive (sibling: test_msk_block.py) so chunk size
is fully controlled - chunk-independence is a property under direct test,
not scheduler luck. Synthetic envelopes below model the measured failure:
chips whose winning phase differs per burst, separated by noise the old
Gardner loop railed on. Fractional STO is injected via the phase argument
(OSR=1 trap rule)."""

from __future__ import annotations

import numpy as np
import pytest
from helpers._fakegr import FAKE_GR, drive

from marconi.engine.backends.gnuradio.embedded.burst import (
    _FLOOR_BLOCK,
    make_burst_sampler,
)


def _run_block(env: np.ndarray, sps: float, chunk: int | None = None) -> np.ndarray:
    blk = make_burst_sampler(FAKE_GR, sps=sps)
    return drive(
        blk,
        np.asarray(env, np.float32),
        chunk=chunk if chunk is not None else max(1, env.size),
        out_dtype=np.float32,
    )


def _ppm_burst(bits: str, sps: int, phase: int, amp: float = 1.0) -> np.ndarray:
    # bit 1 -> pulse-first chip pair (1,0); bit 0 -> (0,1); sps samples/chip
    chips = []
    for b in bits:
        chips += [1, 0] if b == "1" else [0, 1]
    sig = np.repeat(np.asarray(chips, np.float32) * amp, sps)
    return np.concatenate([np.zeros(phase, np.float32), sig])


def _noise(n: int, rng: np.random.Generator, level: float = 0.05) -> np.ndarray:
    return np.abs(rng.normal(0.0, level, n)).astype(np.float32)


def test_per_burst_phase_recovery() -> None:
    rng = np.random.default_rng(7)
    sps = 2
    bits = "10110010" * 8  # 64 bits = 128 chips per burst
    stream = [_noise(3000, rng)]
    for phase in (0, 1, 0, 1):  # adversarial: winning phase alternates per burst
        stream += [_ppm_burst(bits, sps, phase), _noise(2500, rng)]
    env = np.concatenate(stream)
    out = _run_block(env, float(sps))
    # each burst's chips must slice back to the transmitted chip pattern:
    # find each burst in the output by amplitude and compare the chip string
    hard = (out > 0.5).astype(np.uint8)
    want = "".join("10" if b == "1" else "01" for b in bits)
    found = "".join(map(str, hard))
    assert (
        found.count(want) == 4
    ), f"expected all 4 bursts recovered, got {found.count(want)}"


def test_single_burst_is_not_fragmented() -> None:
    # regression: the burst's OWN modulation transitions must never trip the
    # sustained-calm end-of-burst confirmation early. A PPM bit stream has
    # legitimate 2-chip low runs at bit junctions (e.g. a "1" bit's trailing
    # chip abutting a "0" bit's leading chip); a fall-run shorter than that
    # chops one burst into many low-sample, phase-unreliable flushes even
    # though the exact chip pattern can still (fragilely) reassemble on a
    # noiseless synthetic - this pins the mechanism, not just the output.
    rng = np.random.default_rng(9)
    bits = "10110010" * 8
    env = np.concatenate([_noise(2000, rng), _ppm_burst(bits, 2, 0), _noise(2000, rng)])
    blk = make_burst_sampler(FAKE_GR, sps=2.0)
    drive(blk, env, chunk=env.size, out_dtype=np.float32)
    assert blk.diagnostics["bursts_flushed"] == 1


def test_unfinished_burst_withheld_at_eof() -> None:
    # matches the docstring: "withholds an unfinished burst tail at EOF".
    # The burst's tail is placed flush with the end of input (total length
    # an exact _FLOOR_BLOCK multiple) so it is both fully SCANNED (unlike
    # test_eof_pending_remainder_bounded's raw un-scanned carryover) and
    # has no trailing calm run to confirm it complete - the state the
    # docstring's "unfinished burst tail" describes.
    rng = np.random.default_rng(31)
    bits = "1011" * 4
    burst = _ppm_burst(bits, 2, 0)
    prefix = _noise(_FLOOR_BLOCK - burst.size, rng)
    env = np.concatenate([prefix, burst])
    assert env.size == _FLOOR_BLOCK
    blk = make_burst_sampler(FAKE_GR, sps=2.0)
    out = drive(blk, env, chunk=env.size, out_dtype=np.float32)
    assert blk.diagnostics["bursts_flushed"] == 0
    want = "".join("10" if b == "1" else "01" for b in bits)
    hard = (out > 0.5).astype(np.uint8)
    assert want not in "".join(map(str, hard))


def test_deterministic_across_runs() -> None:
    rng = np.random.default_rng(11)
    env = np.concatenate(
        [_noise(5000, rng), _ppm_burst("1100" * 16, 2, 1), _noise(5000, rng)]
    )
    a = _run_block(env, 2.0)
    b = _run_block(env, 2.0)
    assert a.shape == b.shape and bool(np.array_equal(a, b))


@pytest.mark.parametrize("chunk", [1, 37, 997, 8191, 1 << 16])
def test_output_independent_of_chunk_size(chunk: int) -> None:
    # _FLOOR_BLOCK internal batching is the chunk-independence mechanism:
    # the SAME total input must produce byte-identical output regardless of
    # how the caller (GR's real scheduler in production) happens to slice
    # delivery. chunk=1 is the adversarial extreme (per-sample work calls).
    rng = np.random.default_rng(13)
    env = np.concatenate(
        [_noise(4000, rng), _ppm_burst("10110010" * 4, 2, 1), _noise(6000, rng)]
    )
    baseline = _run_block(env, 2.0, chunk=env.size)
    out = _run_block(env, 2.0, chunk=chunk)
    assert np.array_equal(out, baseline)


def test_output_rate_is_nominal() -> None:
    rng = np.random.default_rng(3)
    env = _noise(200_000, rng)
    out = _run_block(env, 2.0)
    assert abs(len(out) - 100_000) <= 200  # +-seam slack, no rail


def test_eof_pending_remainder_bounded() -> None:
    # the fixed _FLOOR_BLOCK batching that buys chunk-independence (above)
    # costs a trailing raw remainder shorter than one window that a true
    # EOF can never flush without more input (docstring: withholds an
    # unfinished tail; suite convention is sims pad past it). Pin the bound
    # so a future change can't silently widen the loss.
    rng = np.random.default_rng(21)
    env = _noise(50_000, rng)
    blk = make_burst_sampler(FAKE_GR, sps=2.0)
    drive(blk, env, chunk=env.size, out_dtype=np.float32)
    assert 0 <= blk._pending.size < _FLOOR_BLOCK


def test_rejects_sub_two_sps() -> None:
    with pytest.raises(ValueError):
        make_burst_sampler(FAKE_GR, sps=1.5)


def test_real_scheduler_runs_to_completion_without_deadlock() -> None:
    # FAKE_GR/drive is a hand-rolled approximation of the forecast/EOF
    # contract; this drives the same block through the actual GR C++
    # scheduler once to prove forecast_drain's [0]-while-pending announcement
    # never stalls it (float32 I/O only, so the uint8-main-thread SIGSEGV
    # trap - gr310-block-and-scheduler-quirks - does not apply here).
    from gnuradio import blocks as gb
    from gnuradio import gr

    from marconi.engine.backends.gnuradio.runner import ensure_worker_warm

    ensure_worker_warm()
    rng = np.random.default_rng(17)
    bits = "10110010" * 8
    env = np.concatenate([_noise(3000, rng), _ppm_burst(bits, 2, 1), _noise(3000, rng)])
    blk = make_burst_sampler(gr, sps=2.0)
    tb = gr.top_block()
    src = gb.vector_source_f(env.astype(np.float32).tolist(), False)
    snk = gb.vector_sink_f()
    tb.connect(src, blk, snk)
    tb.run()
    out = np.asarray(snk.data(), np.float32)
    hard = (out > 0.5).astype(np.uint8)
    want = "".join("10" if b == "1" else "01" for b in bits)
    assert want in "".join(map(str, hard))
