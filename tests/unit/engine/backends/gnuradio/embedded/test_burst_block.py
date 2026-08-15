"""burst_sampler: open-loop decimation with per-burst phase acquisition.
Scheduler-free via FAKE_GR/drive (sibling: test_msk_block.py) so chunk size
is fully controlled - chunk-independence is a property under direct test,
not scheduler luck. Synthetic envelopes below model the measured failure:
chips whose winning phase differs per burst, separated by noise the old
Gardner loop railed on. Fractional STO is injected via the phase argument
(OSR=1 trap rule)."""

from __future__ import annotations

import sys
from collections.abc import Callable
from types import FrameType, SimpleNamespace
from typing import Any

import numpy as np
import pytest
from helpers._fakegr import FAKE_GR, drive

from marconi.engine.backends.gnuradio.embedded.burst import (
    _FLOOR_BLOCK,
    BurstGeometry,
    BurstSamplerCore,
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


def _ppm_burst_ramped(
    bits: str, sps: int, low_amp: float, high_amp: float
) -> np.ndarray:
    # models a sliding-window agc stepping its gain mid-burst (measured on a
    # live capture: 1.9-2.3x within one ~120us frame as strong content enters
    # or exits the look-ahead) - a linear ramp across the chip sequence, not
    # a step, since the mechanism being modeled is continuous gain tracking
    chips = []
    for b in bits:
        chips += [1, 0] if b == "1" else [0, 1]
    ramp = np.linspace(low_amp, high_amp, len(chips)).astype(np.float32)
    return np.repeat(np.asarray(chips, np.float32) * ramp, sps)


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


def test_amplitude_immune_across_bursts_and_within_a_ramping_burst() -> None:
    # Round-3 finding on a live capture: burst_sampler is AGC-free by design
    # (no upstream stage establishes a shared amplitude reference), and the
    # sliding-window agc it replaces was independently found to step its own
    # gain mid-burst on pulsed signals (measured 1.9-2.3x within one ~120us
    # frame) - a fixed downstream slicing threshold cannot survive either
    # condition, only per-burst normalization can. Two bursts here sit at
    # baseline amplitudes 5x apart (no consistent global reference between
    # them, unlike a shared-gain AGC would provide), and one of the two ALSO
    # carries a linear gain ramp across its own duration (modeling exactly
    # what the sliding-window agc used to do to a single frame) - both must
    # still recover their exact chip pattern. Amplitudes are ratios to the
    # noise level, no protocol reference.
    noise_level = 0.05
    rng = np.random.default_rng(59)
    bits_ramped, bits_steady = "10110010" * 8, "01101001" * 8
    ramped = _ppm_burst_ramped(
        bits_ramped,
        2,
        low_amp=noise_level * 8.0,
        high_amp=noise_level * 14.0,
    )
    steady = _ppm_burst(bits_steady, 2, phase=0, amp=noise_level * 55.0)
    # steady's amplitude is ~5x ramped's midpoint ((8+14)/2 * 5 == 55)
    env = np.concatenate(
        [
            _noise(3000, rng, noise_level),
            ramped,
            _noise(3000, rng, noise_level),
            steady,
            _noise(3000, rng, noise_level),
        ]
    )
    out = _run_block(env, 2.0)
    hard = (out > 0.5).astype(np.uint8)
    found = "".join(map(str, hard))
    assert _chip_string(bits_ramped) in found
    assert _chip_string(bits_steady) in found


def test_scans_stay_vectorized() -> None:
    """The idle chip walk and the in-burst fall scan run inside GR's
    scheduler thread: boxed-scalar Python loops there turn a 60 s live
    capture into seconds of stall. A per-sample loop executes >= 3 traced
    Python lines per sample (12M+ here); the vectorized paths execute
    O(chunks + bursts) (measured: 980k idle, 1.71M dense). Counting
    executed lines is deterministic where a wall-clock ratio flaked under
    xdist contention — the clock measures the machine, the trace measures
    the code."""

    def traced_lines(fn: Callable[[], object]) -> int:
        n = 0

        def tracer(frame: FrameType, event: str, arg: Any) -> Any:
            nonlocal n
            if event == "line":
                n += 1
            return tracer

        # restore whatever was tracing, not None: under pytest --cov this
        # runs inside coverage's tracer, and evicting it leaves every later
        # test in this xdist worker recording nothing — a large phantom
        # coverage regression with no failing test pointing at it
        previous = sys.gettrace()
        sys.settrace(tracer)
        try:
            fn()
        finally:
            sys.settrace(previous)
        return n

    rng = np.random.default_rng(1)
    idle = np.abs(rng.normal(0.0, 0.05, 4_000_000)).astype(np.float32)
    parts = [np.abs(rng.normal(0.0, 0.05, 4096)).astype(np.float32)]
    for _ in range(2000):
        parts.append((1.0 + 0.3 * np.abs(rng.standard_normal(400))).astype(np.float32))
        parts.append(np.abs(rng.normal(0.0, 0.05, 1648)).astype(np.float32))
    bursty = np.concatenate(parts)

    for env, min_bursts in ((idle, 0), (bursty, 1500)):
        blk = make_burst_sampler(FAKE_GR, sps=2.0)
        lines = traced_lines(lambda: drive(blk, env, chunk=65536, out_dtype=np.float32))
        assert blk.diagnostics["bursts_flushed"] >= min_bursts
        assert lines < env.size, f"{lines=} vs {env.size} samples: per-item loop?"


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


def _finality_probe(total: int) -> SimpleNamespace:
    # what the build wiring hands a channelized burst_sampler: a probe whose
    # expected_items equals every input sample this block will read, so
    # nitems_read reaching it proves the whole (unpadded) capture is in hand.
    return SimpleNamespace(expected_items=total, exhausted=lambda: True)


def test_eof_flush_recovers_burst_whose_calm_tail_is_in_the_remainder() -> None:
    # The real fob failure: a burst ends, but its confirming calm run lands in
    # the trailing sub-_FLOOR_BLOCK remainder the block never processes, so the
    # burst stays open and is withheld at EOF (see the withheld test below for
    # the no-probe case). With a finality probe wired, _flush_tail processes
    # that remainder, the fall-detection fires normally, and the burst flushes
    # WHOLE - a natural flush, not a truncation.
    rng = np.random.default_rng(71)
    sps = 2
    bits = "10110010" * 8  # 128 chips -> 256 samples
    burst = _ppm_burst(bits, sps, 0)
    prefix = _noise(894, rng)  # push the burst across the 1024 boundary
    tail = _noise(250, rng)  # calm run (>fall_run) sitting in the remainder
    env = np.concatenate([prefix, burst, tail]).astype(np.float32)
    assert env.size % _FLOOR_BLOCK != 0  # a genuine sub-block remainder exists
    blk = make_burst_sampler(FAKE_GR, sps=float(sps))
    blk.eof_probe = _finality_probe(env.size)
    out = drive(blk, env, chunk=env.size, out_dtype=np.float32)
    assert blk.diagnostics["bursts_flushed"] == 1
    assert blk.diagnostics["bursts_truncated_at_eof"] == 0
    hard = "".join(map(str, (out > 0.5).astype(np.uint8)))
    assert _chip_string(bits) in hard, "burst not recovered from the EOF remainder"


def test_eof_flush_commits_a_truncated_burst_with_a_diagnostic() -> None:
    # A capture that truly ends mid-transmission (no calm tail to confirm the
    # burst complete): at proven finality the block commits what arrived rather
    # than dropping it as noise, and flags it so the run can surface the cut.
    # Same layout as test_unfinished_burst_withheld_at_eof, but WITH a probe.
    rng = np.random.default_rng(31)
    bits = "1011" * 4
    burst = _ppm_burst(bits, 2, 0)
    env = np.concatenate([_noise(_FLOOR_BLOCK - burst.size, rng), burst])
    assert env.size == _FLOOR_BLOCK
    blk = make_burst_sampler(FAKE_GR, sps=2.0)
    blk.eof_probe = _finality_probe(env.size)
    out = drive(blk, env, chunk=env.size, out_dtype=np.float32)
    assert blk.diagnostics["bursts_truncated_at_eof"] == 1
    hard = "".join(map(str, (out > 0.5).astype(np.uint8)))
    assert _chip_string(bits) in hard


def test_no_flush_without_a_finality_probe() -> None:
    # The flush is gated on a wired probe: without one (unknown rate, a live
    # source) the sim-pad withhold stands unchanged, so every FAKE_GR test above
    # that omits eof_probe keeps its original behavior.
    rng = np.random.default_rng(31)
    bits = "1011" * 4
    burst = _ppm_burst(bits, 2, 0)
    env = np.concatenate([_noise(_FLOOR_BLOCK - burst.size, rng), burst])
    blk = make_burst_sampler(FAKE_GR, sps=2.0)  # no eof_probe assigned
    out = drive(blk, env, chunk=env.size, out_dtype=np.float32)
    assert blk.diagnostics["bursts_truncated_at_eof"] == 0
    hard = "".join(map(str, (out > 0.5).astype(np.uint8)))
    assert _chip_string(bits) not in hard


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


def test_rejects_sub_one_sps() -> None:
    with pytest.raises(ValueError):
        make_burst_sampler(FAKE_GR, sps=0.5)


def test_construction_accepts_native_sps_one() -> None:
    make_burst_sampler(FAKE_GR, sps=1.0)


def test_native_sps_one_recovers_burst_chip_pattern() -> None:
    # stride=1.0, phases=1: the variance-max phase loop runs a single phase
    # and round(k*1.0) is the identity, so the block detects and normalizes
    # without decimating - this is the DSP claim under test, proven through
    # the real block, not just a non-raising construction.
    rng = np.random.default_rng(101)
    bits = "10110010" * 8
    burst = _ppm_burst(bits, 1, 0)
    env = np.concatenate([_noise(3000, rng), burst, _noise(3000, rng)])
    out = _run_block(env, 1.0)
    hard = (out > 0.5).astype(np.uint8)
    want = _chip_string(bits)
    assert want in "".join(map(str, hard))


def test_native_sps_one_deterministic_across_runs() -> None:
    rng = np.random.default_rng(103)
    env = np.concatenate(
        [_noise(5000, rng), _ppm_burst("1100" * 16, 1, 0), _noise(5000, rng)]
    )
    a = _run_block(env, 1.0)
    b = _run_block(env, 1.0)
    assert a.shape == b.shape and bool(np.array_equal(a, b))


@pytest.mark.parametrize("sps", [1, 2])
def test_output_grid_is_globally_continuous(sps: int) -> None:
    # The emission grid is ONE global chip index that never resets at a burst
    # boundary, so the total chip count depends only on the input length, not
    # on how many bursts split it: one chip per grid position round(k*stride)
    # below input_len, with zero per-burst slack. (The per-burst grid this
    # replaced re-emitted a pre-pad lead and re-anchored the cursor every
    # flush, inserting ~1 chip per burst boundary - this asserts that
    # insertion is gone.)
    rng = np.random.default_rng(1234)
    parts = [_noise(3000, rng)]
    for phase in (0, 1, 0):
        parts.append(_ppm_burst("10110010" * 6, sps, phase))
        parts.append(_noise(2500, rng))
    env = np.concatenate(parts).astype(np.float32)
    # pad the trailing idle so every input sample is processed (no sub-block
    # _pending remainder) and the last burst has confirmed complete (no
    # withheld tail): both preconditions for an exact count
    total = ((env.size // _FLOOR_BLOCK) + 1) * _FLOOR_BLOCK
    env = np.concatenate([env, _noise(total - env.size, rng)]).astype(np.float32)
    assert env.size % _FLOOR_BLOCK == 0
    out = _run_block(env, float(sps))
    assert len(out) == -(-env.size // sps)


@pytest.mark.parametrize("n,sps", [(4003, 4), (4097, 2), (2048, 2)])
def test_chip_count_covers_every_grid_slot_below_the_input_length(
    n: int, sps: int
) -> None:
    # A length that is not a multiple of the stride is the only one that tells
    # floor from ceil apart: the last grid slot still lands strictly below
    # input_len and is still owed a chip. The padded-to-a-multiple case above
    # cannot see the difference.
    rng = np.random.default_rng(n)
    core = BurstSamplerCore(BurstGeometry.build(sps=float(sps)))
    core.process_block(_noise(n, rng))
    core.finish()
    assert core.out.size == -(-n // sps), (core.out.size, n, sps)


def test_frame_straddling_burst_boundary_survives() -> None:
    # One logical chip sequence whose interior quiet run (longer than the fall
    # confirmation) makes the detector split it into TWO bursts. On the old
    # per-burst grid the second burst re-anchors (pre-pad prepend + cursor
    # reset) and the chip-pair grid slips mid-sequence; on the global continuous
    # grid both halves stay at their exact positions, so the whole sequence -
    # the two halves plus the EXACT number of idle chips the gap decimates to -
    # comes back contiguous. Models the real off-air frame lost at 6/7.
    rng = np.random.default_rng(2025)
    sps = 2
    a_bits, b_bits = "10110010" * 4, "11001010" * 4  # both start pulse-first
    gap = 128  # samples; > fall_run (64) so the detector splits into two bursts
    frame = np.concatenate(
        [
            _ppm_burst(a_bits, sps, 0),
            np.zeros(gap, np.float32),
            _ppm_burst(b_bits, sps, 0),
        ]
    )
    env = np.concatenate([_noise(2048, rng), frame, _noise(4096, rng)])
    total = ((env.size // _FLOOR_BLOCK) + 1) * _FLOOR_BLOCK
    env = np.concatenate([env, _noise(total - env.size, rng)]).astype(np.float32)
    blk = make_burst_sampler(FAKE_GR, sps=float(sps))
    out = drive(blk, env, chunk=env.size, out_dtype=np.float32)
    assert blk.diagnostics["bursts_flushed"] == 2
    hard = (out > 0.5).astype(np.uint8)
    found = "".join(map(str, hard))
    want = _chip_string(a_bits) + "0" * (gap // sps) + _chip_string(b_bits)
    assert want in found, "chip sequence slipped at the burst boundary"
    assert len(out) == env.size // sps  # global continuity: no insertion


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


@pytest.mark.parametrize("sps", [256, 1024, 1025, 2048])
def test_detects_bursts_at_every_sps_the_factory_accepts(sps: int) -> None:
    # The block-local detector windows scale with sps: the rise hunt needs
    # rise_run consecutive samples inside ONE processing block, and the fall
    # smoothing needs fall_smooth. A processing block pinned at _FLOOR_BLOCK
    # made both unreachable once they outgrew it, so no burst could ever be
    # detected and every chip fell through the idle path at noise scale.
    # _FALL_CHIPS of quiet must fit AFTER the burst for it to close normally.
    rng = np.random.default_rng(11)
    env = np.concatenate(
        [
            _noise(16 * sps, rng),
            np.ones(16 * sps, np.float32),
            _noise(128 * sps, rng),
        ]
    )
    blk = make_burst_sampler(FAKE_GR, sps=float(sps))
    out = drive(blk, env, chunk=_FLOOR_BLOCK, out_dtype=np.float32)
    assert blk.diagnostics["bursts_flushed"] == 1
    assert float(np.max(out)) > 0.5


def test_continuous_ook_with_a_loud_first_block_still_detects() -> None:
    # the floor seeded from np.median of the FIRST block: any capture whose
    # first block is >= 50% "on" (continuous OOK, Manchester chips, a capture
    # trimmed to the burst) made the median THE SIGNAL LEVEL, the rise bar
    # sat 4x above an envelope that never exceeds 1.0, and 12/16 clean
    # captures decoded to ALL ZEROS with status ok and no hint
    rng = np.random.default_rng(0)
    bits = "".join(rng.choice(list("01"), 400))
    chips = np.repeat(
        np.asarray([int(b) for b in bits], np.float32), 8
    )  # ~50% duty from sample 0
    blk = make_burst_sampler(FAKE_GR, sps=8.0)
    blk.eof_probe = _finality_probe(chips.size)
    out = drive(blk, chips + 0.02, chunk=4096, out_dtype=np.float32)
    # a continuous stream is one burst that runs to EOF: committed truncated
    assert (
        blk.diagnostics["bursts_flushed"] + blk.diagnostics["bursts_truncated_at_eof"]
        >= 1
    )
    assert out.size > 0
    hard = (out > 0.5).astype(np.uint8)
    want = np.asarray([int(b) for b in bits], np.uint8)
    n = min(hard.size, want.size)
    assert n > 300 and float((hard[:n] != want[:n]).mean()) < 0.02
