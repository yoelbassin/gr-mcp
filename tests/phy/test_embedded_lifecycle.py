"""The shared embedded-block lifecycle (issue 16): one forecast/EOF/drain
discipline in lifecycle.py, and its two load-bearing properties — a large
pending backlog (built from one big input push) still fully drains across
many small output-window calls, and a slow consumer bounds chirp_sync's
memory via backpressure instead of unbounded buffering."""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
from phy._fakegr import FAKE_GR, drive

from marconi.phy.backends.gnuradio.embedded.chirp import chirp_prefix, make_chirp_sync
from marconi.phy.backends.gnuradio.embedded.depuncture import make_depuncture

_EMBEDDED = (
    Path(__file__).resolve().parents[2]
    / "packages"
    / "marconi-phy"
    / "src"
    / "marconi"
    / "phy"
    / "backends"
    / "gnuradio"
    / "embedded"
)


def test_backlog_larger_than_output_window_drains_across_small_windows() -> None:
    """A backlog built from one large input push (far more pending output
    than fits in any single output window) must fully emerge across MANY
    SMALL output-window calls — the drain rides on forecast announcing
    drainability, never on a single oversized window that would hide a
    small-window drain bug."""
    keep_mask = [1, 1, 1, 0]
    block_in, block_out = 3, 4
    n_blocks = 514  # 514*4 = 2056 pending items, >> the out_len=64 window
    rng = np.random.default_rng(4)
    payload = rng.standard_normal(n_blocks * block_in).astype(np.float32)

    expected_blocks = np.zeros((n_blocks, block_out), dtype=np.float32)
    expected_blocks[:, :block_in] = payload.reshape(n_blocks, block_in)
    expected = expected_blocks.reshape(-1)

    big_out = drive(
        make_depuncture(FAKE_GR, keep_mask=keep_mask),
        payload,
        chunk=payload.size,
        out_dtype=np.float32,
    )
    tiny_out = drive(
        make_depuncture(FAKE_GR, keep_mask=keep_mask),
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


def test_lifecycle_owned_by_shared_module() -> None:
    """The forecast/EOF/drain discipline lives in lifecycle.py alone: every
    embedded module with pending-output state uses OutQueue/forecast_drain
    rather than a hand-rolled dialect. css_map/css_demap (fixed small-ratio
    converters) and sym_strip (streaming pass-through) are the documented
    exceptions with no pending state."""
    users = {"chirp.py", "depuncture.py", "msk.py", "ofdm.py"}
    for name in users:
        src = (_EMBEDDED / name).read_text()
        assert "forecast_drain" in src, f"{name} does not use the shared forecast"
    for path in _EMBEDDED.glob("*.py"):
        if path.name == "lifecycle.py":
            continue
        src = path.read_text()
        assert not re.search(
            r"np\.concatenate\(\[self\._out", src
        ), f"{path.name} re-implements the pending-output queue"
