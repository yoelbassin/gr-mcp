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


def _chip_string(bits: str) -> str:
    return "".join("10" if b == "1" else "01" for b in bits)


def _ppm_burst_weak_head(
    bits: str, sps: int, weak_chips: int, weak_amp: float, full_amp: float = 1.0
) -> np.ndarray:
    # models a real pulse's own filter/AGC rise time: the burst's leading
    # weak_chips chips are measurably weaker than the rest (weak_amp), not
    # absent - a "0" chip stays 0 regardless of which amplitude multiplies it
    chips = []
    for b in bits:
        chips += [1, 0] if b == "1" else [0, 1]
    arr = np.asarray(chips, np.float32)
    scale = np.full(len(arr), full_amp, np.float32)
    scale[:weak_chips] = weak_amp
    return np.repeat(arr * scale, sps)


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
    want = _chip_string(bits)
    found = "".join(map(str, hard))
    assert (
        found.count(want) == 4
    ), f"expected all 4 bursts recovered, got {found.count(want)}"


def test_single_burst_is_not_fragmented() -> None:
    # regression (two escapes so far, both against this same test's payload):
    # (1) the burst's OWN chip-junction transitions must never trip the
    # sustained-calm end-of-burst confirmation - a PPM bit stream has
    # legitimate 2-chip low runs at bit junctions (e.g. a "1" bit's trailing
    # chip abutting a "0" bit's leading chip); a fall-run shorter than that
    # chops one burst into many low-sample, phase-unreliable flushes even
    # though the exact chip pattern can still (fragilely) reassemble on a
    # noiseless synthetic. (2) a real off-air capture then showed a
    # STRUCTURAL interior quiet run inside a single transmission (preamble-
    # scale, measured >=6 symbols/12 samples at sps=2) that a fall-run only
    # just past the chip-junction case (old: 4 chips) still fragmented on -
    # this is why the payload below now embeds a 12-symbol interior quiet
    # run explicitly, not just ordinary chip-junction gaps, and asserts
    # exact recovery spanning it, not just a flush count.
    rng = np.random.default_rng(9)
    sps = 2
    lead_bits, trail_bits = "10110010" * 4, "01001101" * 4
    gap_symbols = 12
    burst = np.concatenate(
        [
            _ppm_burst(lead_bits, sps, 0),
            np.zeros(gap_symbols * sps, np.float32),
            _ppm_burst(trail_bits, sps, 0),
        ]
    )
    env = np.concatenate([_noise(2000, rng), burst, _noise(2000, rng)])
    blk = make_burst_sampler(FAKE_GR, sps=float(sps))
    out = drive(blk, env, chunk=env.size, out_dtype=np.float32)
    assert blk.diagnostics["bursts_flushed"] == 1

    # the block emits ~1 item per symbol (decimated), so a 12-symbol raw gap
    # is 12 output zeros, not 12*sps
    want = _chip_string(lead_bits) + "0" * gap_symbols + _chip_string(trail_bits)
    hard = (out > 0.5).astype(np.uint8)
    assert want in "".join(map(str, hard))


def test_weak_leading_pulses_are_recovered() -> None:
    # A real pulse's own filter/AGC rise time can leave a burst's leading
    # chips measurably weaker than its steady-state body, even once the
    # receiver has locked on. Controlled reproduction of a live-capture
    # "head lost" read traced it to this: once rise-detection has locked
    # (fires on chip 0 here - the leading amplitude is far above the rise
    # threshold, just below full strength), burst_sampler preserves the
    # EXACT raw envelope value at every chip position, including the weak
    # ones; a hard-decision readout of a too-weak chip against ANY receiver
    # is a channel/SNR limit, not a phase or grid error. Amplitudes are
    # expressed as ratios to the noise level, not absolute values.
    noise_level = 0.05
    weak_ratio = 12.0  # comfortably above the 4x-floor rise threshold ...
    full_ratio = 20.0  # ... but well below the steady-state body
    rng = np.random.default_rng(43)
    bits = "10110010" * 8
    burst = _ppm_burst_weak_head(
        bits,
        2,
        weak_chips=8,
        weak_amp=noise_level * weak_ratio,
        full_amp=noise_level * full_ratio,
    )
    # noise-only false-positive flushes are an accepted, already-measured
    # residual (see test_output_rate_is_nominal's tolerance) and orthogonal
    # to what this test targets, so it checks content recovery directly
    # rather than an exact flush count
    env = np.concatenate(
        [_noise(3000, rng, noise_level), burst, _noise(3000, rng, noise_level)]
    )
    blk = make_burst_sampler(FAKE_GR, sps=2.0)
    out = drive(blk, env, chunk=env.size, out_dtype=np.float32)

    hard = (out > 0.5).astype(np.uint8)
    want = _chip_string(bits)
    found = "".join(map(str, hard))
    assert found.count(want) == 1, f"head not fully recovered: {found.count(want)}"


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
    want = _chip_string(bits)
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
    want = _chip_string(bits)
    assert want in "".join(map(str, hard))
