"""sym_strip: the RX back half of preamble_sync, driven scheduler-free via
FAKE_GR/drive like every sibling embedded block. It was the ONE embedded
block with no in-process test (9% coverage), exercised only inside the
forkserver worker where the verdict is end-to-end BER."""

from __future__ import annotations

import numpy as np
from helpers._fakegr import FAKE_GR, FakeTag, drive

from marconi.engine.backends.gnuradio.embedded.preamble import make_sym_strip

_N_PRE = 8


def _tag(offset: int, phase: float) -> FakeTag:
    return FakeTag(offset, "phase_est", value=phase)


def test_unarmed_stream_emits_nothing() -> None:
    blk = make_sym_strip(FAKE_GR, n_pre=_N_PRE)
    out = drive(blk, np.ones(64, np.complex64), chunk=64, out_dtype=np.complex64)
    assert out.size == 0  # no detection ever armed the strip


def test_detection_strips_pad_preamble_and_derotates() -> None:
    rng = np.random.default_rng(0)
    payload = np.exp(2j * np.pi * rng.random(40)).astype(np.complex64)
    theta = 0.7
    pre = np.ones(_N_PRE, np.complex64)
    stream = (
        np.concatenate([0.1 * np.ones(16, np.complex64), pre, payload])
        * np.exp(1j * theta)
    ).astype(np.complex64)
    blk = make_sym_strip(FAKE_GR, n_pre=_N_PRE)
    # corr_est tags the DETECTION offset; the payload starts n_pre+1 later
    det = 16
    blk.in_tags = [_tag(det, theta)]
    out = drive(blk, stream, chunk=stream.size, out_dtype=np.complex64)
    want = stream[det + _N_PRE + 1 :] * np.exp(-1j * theta)
    assert out.size == want.size
    assert np.allclose(out, want, atol=1e-5)


def test_second_detection_rearms_with_its_own_phase() -> None:
    rng = np.random.default_rng(1)
    a = np.exp(2j * np.pi * rng.random(30)).astype(np.complex64)
    b = np.exp(2j * np.pi * rng.random(30)).astype(np.complex64)
    stream = np.concatenate([a, b]).astype(np.complex64)
    blk = make_sym_strip(FAKE_GR, n_pre=_N_PRE)
    blk.in_tags = [_tag(0, 0.3), _tag(30, -0.5)]
    out = drive(blk, stream, chunk=stream.size, out_dtype=np.complex64)
    first = stream[_N_PRE + 1 : 30] * np.exp(-1j * 0.3)
    second = stream[30 + _N_PRE + 1 :] * np.exp(1j * 0.5)
    assert np.allclose(out, np.concatenate([first, second]), atol=1e-5)


def test_chunk_boundaries_do_not_change_the_output() -> None:
    rng = np.random.default_rng(2)
    stream = np.exp(2j * np.pi * rng.random(200)).astype(np.complex64)
    tags = [_tag(10, 0.2), _tag(90, 1.1)]
    outs = []
    for chunk in (200, 64, 7):
        blk = make_sym_strip(FAKE_GR, n_pre=_N_PRE)
        blk.in_tags = list(tags)
        outs.append(drive(blk, stream, chunk=chunk, out_dtype=np.complex64))
    assert np.allclose(outs[0], outs[1], atol=1e-6)
    assert np.allclose(outs[0], outs[2], atol=1e-6)
