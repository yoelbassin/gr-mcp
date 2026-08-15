"""Per-block item counts, so an empty sink says which stage emptied it.

GR keeps exact per-block counters and they outlive the run, so the whole chain
is measurable without inserting a probe -- no topology change, no cost. Before
this a chain that acquired nothing returned status="ok" with a silent empty
file and the operator had no way to tell which stage stopped passing data.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from helpers import _synth as synth

from marconi.engine.backends.base import RunResult
from marconi.engine.backends.gnuradio.runner import GnuRadioBackend, ensure_worker_warm
from marconi.engine.compile.compiler import compile_modem
from marconi.engine.modulation.psk.stages import PskDemapStep, PskDemodStep
from marconi.engine.stages.acquisition import PreambleSyncStep
from marconi.engine.stages.conditioning import ChannelizeStep, ClockCorrectStep
from marconi.engine.stages.registry import stage_registry
from marconi.engine.types.descriptor import Amplitude, Descriptor
from marconi.engine.types.enums import ItemType, PskOrder
from marconi.engine.types.levels import Level
from marconi.engine.types.models import Modem
from marconi.engine.types.step import Step

IQ = Descriptor(Level.IQ, ItemType.C)
# a clean synthetic transmitter output, so the scale is known without an agc --
# which would otherwise add blocks to the very census under test
IQ_NORMALIZED = Descriptor(Level.IQ, ItemType.C, amplitude=Amplitude.PEAK_UNITY)
_SR, _SYM, _NBITS = 8.0, 1.0, 4096
_QPSK = PskOrder.QPSK
# Fraction of an upstream block's output its downstream must have read by EOF.
# Measured on this chain: 1.0 everywhere but symbol_sync_cc, which ends holding
# 31 of 16384 items (0.9981) -- the rrc history it never gets to consume. The
# floor sits at 0.98 so a longer filter (a full 89-tap history is 0.54% here)
# cannot false-fail it, while the failures it exists to catch -- a halved or
# mis-ported read counter, a chain that stalled mid-stream -- miss by ~50%.
_DRAIN_FLOOR = 0.98


def _run(path: list[Step], src: Path, snk: Path) -> RunResult:
    pipe = compile_modem(
        Modem(symbol_rate=_SYM, path=path),
        stage_registry(),
        sample_rate=_SR,
        start=IQ_NORMALIZED,
        source_io={"path": str(src)},
        sink_io={"path": str(snk)},
    )
    return GnuRadioBackend().run_pipeline(pipe, timeout=180.0)


def _signal(tmp_path: Path) -> Path:
    bits = np.random.default_rng(0).integers(0, 2, _NBITS).astype(np.uint8)
    tx: list[Step] = [PskDemodStep(order=_QPSK), PskDemapStep(order=_QPSK)]
    return synth.write(
        tmp_path / "sig.iq",
        synth.from_steps(tx, bits, sample_rate=_SR, symbol_rate=_SYM),
    )


def _by_kind(res: RunResult, kind: str) -> list[int | None]:
    return [c.items_out for c in res.census if c.kind == kind]


def test_census_covers_the_whole_pipeline_in_order(tmp_path: Path) -> None:
    ensure_worker_warm()
    res = _run(
        [PskDemodStep(order=_QPSK)],
        _signal(tmp_path),
        tmp_path / "out.cf32",
    )
    assert res.status == "ok"
    assert [c.kind for c in res.census] == [
        "iq_file_source",
        "rrc_filter_ccf",
        "symbol_sync_cc",
        "costas_loop_cc",
        "iq_file_sink",
    ]
    assert all(c.block for c in res.census)


def test_a_missing_port_reads_as_none_not_zero(tmp_path: Path) -> None:
    """A source has no input and a sink no output; that must not be reported as
    'consumed nothing', which is the failure signature."""
    ensure_worker_warm()
    res = _run(
        [PskDemodStep(order=_QPSK)],
        _signal(tmp_path),
        tmp_path / "out.cf32",
    )
    source, sink = res.census[0], res.census[-1]
    assert source.items_in is None
    assert source.items_out == 8 * _NBITS // 2
    assert sink.items_out is None and sink.items_in is not None


def test_census_measures_a_rate_change_exactly(tmp_path: Path) -> None:
    ensure_worker_warm()
    decim = 4
    res = _run(
        [ChannelizeStep(decim=decim, bandwidth_hz=1.0, center_hz=0.0)],
        _signal(tmp_path),
        tmp_path / "out.cf32",
    )
    assert res.status == "ok"
    (produced,) = _by_kind(res, "freq_xlating_fir_filter_ccf")
    arrived = res.census[0].items_out
    assert arrived is not None
    assert produced == arrived // decim


def test_census_names_the_stage_that_produced_nothing(tmp_path: Path) -> None:
    """A preamble that is not in the signal: acquisition withholds everything
    and the sink is empty, but the run is still 'ok'. The census is the only
    thing that says where the data stopped."""
    ensure_worker_warm()
    path: list[Step] = [
        PskDemodStep(order=_QPSK),
        PreambleSyncStep(
            preamble_i=[1.0, -1.0] * 16,
            preamble_q=[0.0] * 32,
            threshold=0.99,
        ),
    ]
    res = _run(path, _signal(tmp_path), tmp_path / "out.cf32")
    assert res.status == "empty"
    dead = [c for c in res.census if c.items_out == 0]
    assert [c.kind for c in dead] == ["sym_strip"]
    upstream = next(c for c in res.census if c.kind == "corr_est_cc")
    assert upstream.items_out is not None and upstream.items_out > 0


def test_a_block_without_counters_does_not_fail_the_run(tmp_path: Path) -> None:
    """pfb_arb_resampler exposes no nitems_read at all. A diagnostic must never
    be able to fail a run that otherwise worked -- it reports None instead."""
    ensure_worker_warm()
    res = _run(
        [ClockCorrectStep(ppm=5.0)],
        _signal(tmp_path),
        tmp_path / "out.cf32",
    )
    assert res.status == "ok", res.error
    arb = next(c for c in res.census if c.kind == "pfb_arb_resampler_ccf")
    assert arb.items_in is None


def test_an_all_stock_chain_is_observable_too(tmp_path: Path) -> None:
    """The blind spot this closes: diagnostics only ever carried counters that
    a Python block published, so a chain of nothing but stock GR blocks -- the
    common case, and the one the design prefers -- reported nothing at all."""
    ensure_worker_warm()
    res = _run(
        [PskDemodStep(order=_QPSK)],
        _signal(tmp_path),
        tmp_path / "out.cf32",
    )
    assert res.diagnostics == []
    assert len(res.census) == 5


def test_empty_sink_is_not_ok_and_names_the_stall(tmp_path: Path) -> None:
    """A run that reaches EOF but writes nothing to its terminal sink is NOT a
    success: status is 'empty', stalled_at names the first stage that produced
    zero, and error carries the human-readable gradient. status must stop lying
    'ok' when no signal was decoded."""
    ensure_worker_warm()
    path: list[Step] = [
        PskDemodStep(order=_QPSK),
        PreambleSyncStep(
            preamble_i=[1.0, -1.0] * 16,
            preamble_q=[0.0] * 32,
            threshold=0.99,
        ),
    ]
    res = _run(path, _signal(tmp_path), tmp_path / "out.cf32")
    assert res.status == "empty"
    stalled = next(c for c in res.census if c.block == res.stalled_at)
    assert stalled.kind == "sym_strip"
    assert res.error is not None and "sym_strip" in res.error


def test_a_run_that_produces_output_stays_ok(tmp_path: Path) -> None:
    """The honest-empty status must not fire on a normal decode: a chain that
    writes items to its sink is 'ok' with no stall."""
    ensure_worker_warm()
    res = _run(
        [PskDemodStep(order=_QPSK)],
        _signal(tmp_path),
        tmp_path / "out.cf32",
    )
    assert res.status == "ok"
    assert res.stalled_at is None


def test_adjacent_blocks_conserve_the_stream(tmp_path: Path) -> None:
    """items_out is pinned exactly above; items_in was pinned by nothing but
    "is not None" -- its only other gate asserted non-negativity, which GR's
    unsigned counters satisfy by construction, so it could not fail on any run
    this backend can produce. What IS contingent: a row's items_in must measure
    the same stream its upstream's items_out does. No block reads more than the
    block feeding it wrote, and a clean run at EOF leaves only a filter's
    history unread -- measured over 3 identical runs of this chain, every pair
    conserves exactly except symbol_sync_cc, which ends holding 31 of rrc's
    16384 items (99.81%). A read counter scaled by an item size, taken from the
    wrong port, or harvested off the wrong block fails here and nowhere else in
    this file: every other assertion keys on items_out."""
    ensure_worker_warm()
    res = _run(
        [PskDemodStep(order=_QPSK)],
        _signal(tmp_path),
        tmp_path / "out.cf32",
    )
    assert res.status == "ok"
    for up, down in zip(res.census, res.census[1:]):
        wrote, read = up.items_out, down.items_in
        assert wrote is not None and read is not None, (up, down)
        assert read <= wrote, f"{down.kind} read {read} of the {wrote} {up.kind} wrote"
        assert read >= _DRAIN_FLOOR * wrote, (
            f"{down.kind} left {wrote - read} of {up.kind}'s {wrote} items "
            "unread: the stream stopped moving mid-chain"
        )
