"""CSS peak-dominance quality verdicts through the real engine.

A bare dechirp terminal used to produce no checkable evidence (hard symbols,
no soft stream), so every css decode read "uncertain". peak_decision now
tallies per-symbol peak/median dominance of its decision vector; these gates
pin the verdict behavior the calibration measured: decoded through the AWGN
envelope, no_signal on noise/tone, and conservative (never decoded) on
wrong-sf and at the symbol-error edge.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from helpers import _synth as synth
from helpers._dsp import channel

from marconi.engine.backends.gnuradio.runner import ensure_worker_warm
from marconi.engine.io.source import SourceSlice
from marconi.engine.modulation.css.stages import DechirpStep
from marconi.engine.run import PipelineResult, run_rx
from marconi.engine.stages.registry import stage_registry
from marconi.engine.types.descriptor import Descriptor
from marconi.engine.types.enums import ItemType
from marconi.engine.types.levels import Level
from marconi.engine.types.models import Modem

IQ = Descriptor(Level.IQ, ItemType.C)
_SF, _OS, _ZP = 7, 8, 1
_RATE = float(_OS * (1 << _SF))  # symbol_rate 1.0
_N_SYM = 60


def _tx_iq(tmp_path: Path, snr_db: float | None) -> Path:
    ensure_worker_warm()
    symbols = np.random.default_rng(0).integers(0, 1 << _SF, _N_SYM)
    clean = synth.write(
        tmp_path / "clean.cf32",
        synth.chirp_symbols(symbols, sf=_SF, oversample=_OS),
    )
    if snr_db is None:
        return clean
    return channel(clean, tmp_path / "noisy.cf32", snr_db=snr_db)


def _rx(
    tmp_path: Path, iq: Path, sf: int = _SF, oversample: int = _OS
) -> PipelineResult:
    return run_rx(
        Modem(
            symbol_rate=1.0,
            path=[DechirpStep(sf=sf, oversample=oversample, zero_pad=_ZP)],
        ),
        stage_registry(),
        sample_rate=_RATE,
        start=IQ,
        workdir=tmp_path,
        source=SourceSlice(path=iq),
        timeout=60.0,
    )


def test_clean_chirp_reads_decoded(tmp_path: Path) -> None:
    res = _rx(tmp_path, _tx_iq(tmp_path, snr_db=5.0))
    assert res.quality is not None
    assert res.quality.verdict == "decoded", res.quality
    assert [e.metric for e in res.quality.evidence] == ["peak_dominance"]


def test_deep_snr_chirp_still_reads_decoded(tmp_path: Path) -> None:
    # -12 dB per-sample: far below anything a non-spread demod survives, yet
    # measured dominance holds every symbol above the floor (SER 0)
    res = _rx(tmp_path, _tx_iq(tmp_path, snr_db=-12.0))
    assert res.quality is not None
    assert res.quality.verdict == "decoded", res.quality


def test_symbol_error_edge_is_never_decoded(tmp_path: Path) -> None:
    # -18 dB: symbol errors begin (~9% measured) and the dominant fraction
    # collapses to ~0.08 — conservative territory, whichever side of the
    # negative bar noise variation lands it on
    res = _rx(tmp_path, _tx_iq(tmp_path, snr_db=-18.0))
    assert res.quality is not None
    assert res.quality.verdict != "decoded", res.quality


def test_noise_reads_no_signal(tmp_path: Path) -> None:
    rng = np.random.default_rng(1)
    n = _N_SYM * int(_RATE)
    x = 0.1 * (rng.standard_normal(n) + 1j * rng.standard_normal(n))
    p = tmp_path / "noise.cf32"
    x.astype(np.complex64).tofile(p)
    res = _rx(tmp_path, p)
    assert res.quality is not None
    assert res.quality.verdict == "no_signal", res.quality


def test_tone_reads_no_signal(tmp_path: Path) -> None:
    # a CW carrier dechirps to a sweep — flat spectrum, nothing dominant
    n = _N_SYM * int(_RATE)
    x = np.exp(2j * np.pi * 0.01 * np.arange(n)).astype(np.complex64)
    p = tmp_path / "tone.cf32"
    x.tofile(p)
    res = _rx(tmp_path, p)
    assert res.quality is not None
    assert res.quality.verdict == "no_signal", res.quality


def test_wrong_sf_is_never_decoded(tmp_path: Path) -> None:
    # wrong sf at the SAME sample rate (the compiler rejects an sf whose
    # window disagrees with the rate outright, so the operator's realistic
    # mistake is a wrong sf/oversample split): the reference chirp's slope
    # is 2x off, the dechirped energy smears, and dominance must not vouch
    iq = _tx_iq(tmp_path, snr_db=10.0)
    res = _rx(tmp_path, iq, sf=_SF + 1, oversample=_OS // 2)
    assert res.quality is not None
    assert res.quality.verdict != "decoded", res.quality


@pytest.mark.parametrize("snr_db", [5.0, -12.0])
def test_dominance_tallies_cover_every_symbol(tmp_path: Path, snr_db: float) -> None:
    res = _rx(tmp_path, _tx_iq(tmp_path, snr_db=snr_db))
    dom = {
        d.key: d.count
        for d in res.diagnostics
        if d.block.startswith("peak_decision") and d.count is not None
    }
    assert dom["symbols_total"] == _N_SYM
    # threshold, not exact: at -12 dB a symbol's dominance can graze the floor
    assert dom["dominant_symbols"] >= int(0.9 * _N_SYM)
