"""The shared embedded-block lifecycle (issue 16): one forecast/EOF/drain
discipline in lifecycle.py, and its two load-bearing properties — a final
frame larger than one output window fully drains at EOF, and a slow consumer
bounds chirp_sync's memory via backpressure instead of unbounded buffering."""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
from phy._fakegr import FAKE_GR, drive
from phy.test_css_explicit_decode import _HEADER, _SF11_SYMBOLS

from marconi.phy.backends.gnuradio.embedded.chirp import chirp_prefix, make_chirp_sync
from marconi.phy.backends.gnuradio.embedded.coding import make_css_explicit_decode

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


def test_final_frame_larger_than_output_window_drains_at_eof() -> None:
    """css_explicit_decode's decoded frame (2056 bits) must fully emerge even
    when every granted output window is far smaller — the EOF flush rides on
    forecast announcing drainability, never on more input arriving."""
    blk = make_css_explicit_decode(FAKE_GR, sf=11, ldro=True, **_HEADER)
    big_out = drive(
        make_css_explicit_decode(FAKE_GR, sf=11, ldro=True, **_HEADER),
        np.asarray(_SF11_SYMBOLS, np.int16),
        chunk=1 << 16,
        out_dtype=np.uint8,
    )
    tiny_out = drive(
        blk,
        np.asarray(_SF11_SYMBOLS, np.int16),
        chunk=37,
        out_len=64,  # 33 windows needed for one frame
        out_dtype=np.uint8,
    )
    assert big_out.size == 2056
    assert np.array_equal(tiny_out, big_out)
    assert blk.diagnostics["frames_decoded"] == 1


def test_chirp_sync_memory_bounded_under_slow_consumer() -> None:
    """A consumer draining 256 samples per call against 8192-sample input
    offers must not grow chirp_sync's internal buffers without bound: once
    pending output saturates, input is left unconsumed (GR backpressure)."""
    sf, osr, pl = 7, 2, 8
    sn = osr * (1 << sf)
    rng = np.random.default_rng(3)
    payload = 0.7 * (rng.standard_normal(400 * sn) + 1j * rng.standard_normal(400 * sn))
    sig = np.concatenate([chirp_prefix(sf, osr, pl), payload]).astype(np.complex64)

    blk = make_chirp_sync(FAKE_GR, sf, osr, 4, pl, float(1 << sf))
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
    users = {"chirp.py", "coding.py", "depuncture.py", "ofdm.py"}
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
