"""ofdm_frame_sync_probe has to MEASURE the geometry it was handed.

Its description offers the stage as the way to confirm fft_len/cp_len before
committing to a demod spec, but the block reported only a truncation counter:
a wrong cp_len and a right one produced byte-identical results, so the probe
could not answer the one question it exists for. cp_corr is the discriminator —
frames_found is not, because the null search locks the frame cadence whatever
fft_len and cp_len claim.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import numpy.typing as npt

from marconi.engine.backends.gnuradio.embedded.ofdm import (
    FrameSyncGeometry,
    OfdmFrameSyncCore,
)

FFT, CP, DATA_SYMS, NULL, FRAMES = 64, 16, 3, 32, 3
SYM = FFT + CP
FRAME = NULL + (DATA_SYMS + 1) * SYM


def _ofdm_stream(seed: int = 0) -> npt.NDArray[np.complex64]:
    """Textbook OFDM: a quiet null symbol, then symbols whose cyclic prefix is
    a copy of the tail of their own useful part. Written from the definition,
    not by inverting the block under test."""
    rng = np.random.default_rng(seed)
    out: list[npt.NDArray[np.complex64]] = []
    for _ in range(FRAMES + 1):
        out.append(np.zeros(NULL, np.complex64))
        for _ in range(DATA_SYMS + 1):
            useful = (rng.normal(0, 1, FFT) + 1j * rng.normal(0, 1, FFT)).astype(
                np.complex64
            )
            out.append(np.concatenate([useful[FFT - CP :], useful]))
    return np.concatenate(out).astype(np.complex64)


def _diagnostics(*, fft_len: int, cp_len: int) -> dict[str, Any]:
    geom = FrameSyncGeometry.build(
        fft_len=fft_len,
        cp_len=cp_len,
        sym_len=fft_len + cp_len,
        null_len=NULL,
        frame_len=FRAME,
        data_syms=DATA_SYMS,
    )
    core = OfdmFrameSyncCore(geom)
    core.feed(_ofdm_stream())
    core.pump()
    return dict(core.diagnostics)


def test_correct_geometry_reports_a_high_cp_correlation() -> None:
    d = _diagnostics(fft_len=FFT, cp_len=CP)
    assert int(d["frames_found"]) >= FRAMES
    assert float(d["cp_corr"]) > 0.95
    assert "null_offset" in d


def test_a_wrong_cp_len_collapses_the_cp_correlation() -> None:
    """The mutation guard: same capture, same null cadence, only cp_len wrong.
    frames_found stays put, so cp_corr has to be what moves."""
    right = _diagnostics(fft_len=FFT, cp_len=CP)
    wrong = _diagnostics(fft_len=FFT, cp_len=CP // 2)
    assert float(wrong["cp_corr"]) < 0.5 * float(right["cp_corr"])


def test_a_wrong_fft_len_collapses_the_cp_correlation() -> None:
    right = _diagnostics(fft_len=FFT, cp_len=CP)
    wrong = _diagnostics(fft_len=FFT // 2, cp_len=CP)
    assert float(wrong["cp_corr"]) < 0.5 * float(right["cp_corr"])


def test_a_clean_lock_reports_no_resync_drift() -> None:
    d = _diagnostics(fft_len=FFT, cp_len=CP)
    assert int(d.get("resync_drift_max", 0)) == 0
