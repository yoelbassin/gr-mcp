"""The shared embedded-block lifecycle: one forecast/EOF/drain
discipline in lifecycle.py, and its two load-bearing properties — a large
pending backlog (built from one big input push) still fully drains across
many small output-window calls, and a slow consumer bounds chirp_sync's
memory via backpressure instead of unbounded buffering."""

from __future__ import annotations

import ast
from typing import Any

import numpy as np
from helpers._fakegr import FAKE_GR, drive
from helpers._paths import SRC_MARCONI

from marconi.engine.backends.gnuradio.embedded.chirp import (
    _modulate_symbol,
    chirp_prefix,
    make_chirp_sync,
)
from marconi.engine.backends.gnuradio.embedded.lifecycle import OutQueue, forecast_drain

_EMBEDDED = SRC_MARCONI / "engine" / "backends" / "gnuradio" / "embedded"


def _make_expander(gr: Any, block_in: int, block_out: int) -> Any:
    """Minimal OutQueue consumer: zero-pads each block_in items to block_out.
    The test-owned vehicle for exercising the shared drain discipline."""

    class _Expander(gr.basic_block):
        def __init__(self) -> None:
            gr.basic_block.__init__(
                self, name="expander", in_sig=[np.float32], out_sig=[np.float32]
            )
            self._buf = np.empty(0, dtype=np.float32)
            self._out = OutQueue(np.float32)

        def forecast(self, noutput_items: int, ninputs: int) -> list[int]:
            return forecast_drain(self._out.pending, ninputs)

        def general_work(self, input_items: Any, output_items: Any) -> int:
            inp = input_items[0]
            if inp.size:
                self._buf = np.concatenate([self._buf, np.asarray(inp, np.float32)])
                self.consume(0, len(inp))
            while self._buf.size >= block_in:
                blk = np.zeros(block_out, dtype=np.float32)
                blk[:block_in] = self._buf[:block_in]
                self._buf = self._buf[block_in:]
                self._out.push(blk)
            return self._out.drain(output_items[0])

    return _Expander()


def test_backlog_larger_than_output_window_drains_across_small_windows() -> None:
    """A backlog built from one large input push (far more pending output
    than fits in any single output window) must fully emerge across MANY
    SMALL output-window calls — the drain rides on forecast announcing
    drainability, never on a single oversized window that would hide a
    small-window drain bug."""
    block_in, block_out = 3, 4
    n_blocks = 514  # 514*4 = 2056 pending items, >> the out_len=64 window
    rng = np.random.default_rng(4)
    payload = rng.standard_normal(n_blocks * block_in).astype(np.float32)

    expected_blocks = np.zeros((n_blocks, block_out), dtype=np.float32)
    expected_blocks[:, :block_in] = payload.reshape(n_blocks, block_in)
    expected = expected_blocks.reshape(-1)

    big_out = drive(
        _make_expander(FAKE_GR, block_in, block_out),
        payload,
        chunk=payload.size,
        out_dtype=np.float32,
    )
    tiny_out = drive(
        _make_expander(FAKE_GR, block_in, block_out),
        payload,
        chunk=payload.size,
        out_len=64,  # 33 windows needed to drain the 2056-item backlog
        out_dtype=np.float32,
    )
    assert big_out.size == n_blocks * block_out
    assert np.array_equal(big_out, expected)
    assert np.array_equal(tiny_out, expected)


def test_chirp_sync_memory_bounded_under_slow_consumer() -> None:
    """A consumer draining 256 samples per call against 8192-sample input
    offers must not grow chirp_sync's internal buffers without bound: once
    pending output saturates, input is left unconsumed (GR backpressure)."""
    sf, osr, pl = 7, 2, 8
    sn = osr * (1 << sf)
    rng = np.random.default_rng(3)
    payload = 0.7 * (rng.standard_normal(400 * sn) + 1j * rng.standard_normal(400 * sn))
    sfd, sync = 2.25, 2
    sig = np.concatenate([chirp_prefix(sf, osr, pl, sfd), payload]).astype(np.complex64)

    blk = make_chirp_sync(FAKE_GR, sf, osr, 4, pl, float(1 << sf), sfd, sync)
    pos = 0
    drained = 0
    stalls = 0
    peak = 0
    while pos < sig.size and stalls < 4096:
        before = blk.nitems_read(0)
        out = np.zeros(256, np.complex64)
        k = int(blk.general_work([sig[pos : pos + 8192]], [out]))
        blk._nwritten += k
        drained += k
        consumed = blk.nitems_read(0) - before
        pos += consumed
        stalls = stalls + 1 if (consumed == 0 and k == 0) else 0
        peak = max(peak, blk._buf.size + blk._out.size)
    bound = (1 << 16) + 3 * 8192 + (pl + 6) * sn
    assert peak <= bound, f"internal buffering peaked at {peak} > {bound}"
    assert drained > 100 * sn  # the stream still flows under backpressure


def test_chirp_sync_eof_probe_flushes_withheld_tail() -> None:
    """Without EOF knowledge chirp_sync withholds its scan-claim margin
    forever (the scheduler kills it without a final call once forecast
    demands input). With an exhausted eof_probe it must announce
    drainability and flush the buffered payload tail in whole symbol
    windows on the termination sweep."""
    sf, osr, zp, pl = 7, 2, 4, 8
    sn = osr * (1 << sf)
    payload_syms = 24
    rng = np.random.default_rng(7)
    payload = np.concatenate(
        [
            _modulate_symbol(int(s), sf, osr)
            for s in rng.integers(1, 1 << sf, payload_syms)
        ]
    )
    sig = np.concatenate([chirp_prefix(sf, osr, pl, 2.25), payload]).astype(
        np.complex64
    )

    class _Exhausted:
        # exhausted() true from the first call — the small-capture reality
        # (the source finishes writing almost immediately); expected_items
        # is what licenses the irreversible FIR-tail pad, only once every
        # sample is provably consumed.
        expected_items = sig.size

        def exhausted(self) -> bool:
            return True

    def run(probe: _Exhausted | None) -> np.ndarray:
        blk = make_chirp_sync(FAKE_GR, sf, osr, zp, pl, float(1 << sf), 2.25, 2)
        blk.eof_probe = probe
        parts = [drive(blk, sig, chunk=sn, out_dtype=np.complex64)]
        # the scheduler's termination sweep: zero-input calls arrive only
        # while forecast announces drainability
        for _ in range(64):
            if blk.forecast(1 << 16, 1) != [0]:
                break
            out = np.zeros(1 << 16, np.complex64)
            k = int(blk.general_work([sig[0:0]], [out]))
            assert k, "announced drainability but emitted nothing (would spin)"
            blk._nwritten += k
            parts.append(out[:k].copy())
        return np.concatenate(parts)

    withheld = run(None)
    flushed = run(_Exhausted())
    assert flushed.size == payload_syms * sn, (
        f"flushed run emitted {flushed.size} of {payload_syms * sn} " f"payload samples"
    )
    assert flushed.size >= withheld.size + 4 * sn, (
        f"probe recovered only {flushed.size - withheld.size} samples over "
        f"the withholding run ({withheld.size})"
    )
    # an early-exhausted probe must never corrupt content: the flushed run
    # is a strict extension of the withholding run, sample for sample
    assert np.array_equal(flushed[: withheld.size], withheld)


def test_lifecycle_owned_by_shared_module() -> None:
    """The forecast/EOF/drain discipline lives in lifecycle.py alone: every
    embedded module with pending-output state binds it to OutQueue and gets its
    forecast from forecast_drain rather than a hand-rolled dialect. Checked on
    the parsed module, not its text, so the invariant is about what the code
    binds and imports — a renamed local or a reflowed line cannot fake it, and
    a new hand-rolled queue cannot slip past a substring. css_map/css_demap
    (fixed small-ratio converters) and sym_strip (streaming pass-through) are
    the documented exceptions with no pending state."""
    users = {"chirp.py", "msk.py", "ofdm.py", "cp_sync.py", "pilot_lattice.py"}
    for name in users:
        tree = ast.parse((_EMBEDDED / name).read_text())
        imported = {
            alias.name
            for node in ast.walk(tree)
            if isinstance(node, ast.ImportFrom)
            and (node.module or "").endswith("lifecycle")
            for alias in node.names
        }
        assert "forecast_drain" in imported, f"{name} does not share the forecast"

    for path in _EMBEDDED.glob("*.py"):
        if path.name == "lifecycle.py":
            continue
        tree = ast.parse(path.read_text())
        for node in ast.walk(tree):
            # every pending-output attribute must be constructed from OutQueue
            if not isinstance(node, ast.Assign):
                continue
            targets = [
                tgt
                for tgt in node.targets
                if isinstance(tgt, ast.Attribute) and tgt.attr == "_out"
            ]
            if not targets:
                continue
            call = node.value
            built_by = (
                call.func.id
                if isinstance(call, ast.Call) and isinstance(call.func, ast.Name)
                else None
            )
            assert built_by == "OutQueue", (
                f"{path.name}:{node.lineno} binds self._out to "
                f"{ast.unparse(node.value)} — the pending-output queue is "
                f"lifecycle.OutQueue, not a per-block dialect"
            )
