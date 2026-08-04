from __future__ import annotations

from pathlib import Path

import numpy as np
from integration.engine.quality._capture import make_clean_capture

from marconi.engine.backends.gnuradio.runner import ensure_worker_warm
from marconi.engine.coding.stages_bits import BlockCodeStep, SyncWordStep
from marconi.engine.io.bitfile import write_bits
from marconi.engine.modulation.css.stages import ChirpSyncStep, DechirpStep
from marconi.engine.modulation.fsk.stages import FskStep, MfskSoftDemapStep
from marconi.engine.run import PipelineResult, run_rx
from marconi.engine.stages.probes import BurstProbeStep
from marconi.engine.stages.registry import stage_registry
from marconi.engine.types.descriptor import Descriptor
from marconi.engine.types.enums import ItemType
from marconi.engine.types.levels import Level
from marconi.engine.types.models import Bitstream, Modem

IQ = Descriptor(Level.IQ, ItemType.C)
BITS = Descriptor(Level.BITS, ItemType.B)
_SR, _SYM, _DEV = 4.0, 1.0, 1.0
_CSS_SF, _CSS_OSR, _CSS_ZP, _CSS_SYM = 7, 2, 4, 1.0
_CSS_PREAMBLE_LEN, _CSS_SFD_SYMBOLS, _CSS_SYNC_SYMBOLS = 8, 2.25, 2
_CSS_RATE = _CSS_OSR * (1 << _CSS_SF) * _CSS_SYM


def _css_marks_only_modem() -> Modem:
    return Modem(
        symbol_rate=_CSS_SYM,
        path=[
            ChirpSyncStep(
                sf=_CSS_SF,
                oversample=_CSS_OSR,
                zero_pad=_CSS_ZP,
                preamble_len=_CSS_PREAMBLE_LEN,
                sfd_symbols=_CSS_SFD_SYMBOLS,
                sync_symbols=_CSS_SYNC_SYMBOLS,
            ),
            DechirpStep(sf=_CSS_SF, oversample=_CSS_OSR, zero_pad=_CSS_ZP),
            BurstProbeStep(),
        ],
    )


def _soft_rx(symbol_rate: float) -> Modem:
    return Modem(
        symbol_rate=symbol_rate,
        path=[FskStep(deviation=_DEV), MfskSoftDemapStep(levels=[-1.0, 1.0])],
    )


def _run(modem: Modem, iq: Path, workdir: Path) -> PipelineResult:
    workdir.mkdir(exist_ok=True)
    return run_rx(
        modem,
        stage_registry(),
        sample_rate=_SR,
        start=IQ,
        workdir=workdir,
        source_io={"path": str(iq)},
    )


def _not_trusted(res: PipelineResult) -> bool:
    return res.status != "ok" or (
        res.quality is not None and res.quality.verdict != "decoded"
    )


def test_noise_only_capture_is_not_called_decoded(tmp_path: Path) -> None:
    rng = np.random.default_rng(3)
    noise = (rng.normal(0, 1, 16384) + 1j * rng.normal(0, 1, 16384)) / np.sqrt(2)
    iq = tmp_path / "noise.cf32"
    noise.astype(np.complex64).tofile(iq)
    res = _run(_soft_rx(_SYM), iq, tmp_path / "rx")
    assert _not_trusted(res), res.quality


def test_wrong_symbol_rate_is_not_called_decoded(tmp_path: Path) -> None:
    iq = make_clean_capture(tmp_path)
    res = _run(_soft_rx(1.6), iq, tmp_path / "rx")
    assert _not_trusted(res), res.quality


def test_clean_control_is_decoded(tmp_path: Path) -> None:
    iq = make_clean_capture(tmp_path)
    res = _run(_soft_rx(_SYM), iq, tmp_path / "rx")
    assert res.status == "ok"
    assert res.quality is not None and res.quality.verdict == "decoded"


def test_noisy_but_decodable_capture_is_decoded(tmp_path: Path) -> None:
    # 7 dB SNR: raw BER a few percent, squarely decodable - the old soft bar
    # (6.0, reachable only near-noiseless) starved this whole envelope into
    # "uncertain"; measured ratio here ~3.9 against the 2.0 bar
    clean = make_clean_capture(tmp_path)
    sig = np.fromfile(clean, np.complex64)
    rng = np.random.default_rng(5)
    snr_db = 7.0
    sigma = float(np.sqrt(np.mean(np.abs(sig) ** 2) / 10 ** (snr_db / 10) / 2))
    noisy = (
        sig
        + sigma * (rng.standard_normal(sig.size) + 1j * rng.standard_normal(sig.size))
    ).astype(np.complex64)
    iq = tmp_path / "noisy.cf32"
    noisy.tofile(iq)
    res = _run(_soft_rx(_SYM), iq, tmp_path / "rx_noisy")
    assert res.status == "ok"
    assert res.quality is not None and res.quality.verdict == "decoded", res.quality


def test_pure_tone_capture_is_not_called_decoded(tmp_path: Path) -> None:
    n = 16384
    t = np.arange(n)
    tone = np.exp(2j * np.pi * 0.05 * t).astype(np.complex64)
    iq = tmp_path / "tone.cf32"
    tone.tofile(iq)
    res = _run(_soft_rx(_SYM), iq, tmp_path / "rx")
    assert _not_trusted(res), res.quality


def _bits_rx(
    modem: Modem, bits: np.ndarray, tmp_path: Path, name: str
) -> PipelineResult:
    src = tmp_path / f"{name}.u8"
    write_bits(src, bits)
    workdir = tmp_path / f"{name}_rx"
    workdir.mkdir()
    return run_rx(
        modem,
        stage_registry(),
        sample_rate=1.0,
        start=BITS,
        workdir=workdir,
        input_stream=Bitstream(path=src, num_bits=int(bits.size)),
    )


def test_short_sync_on_random_bits_is_not_called_decoded(tmp_path: Path) -> None:
    # the C1 repro: an 8-bit sync over 60k random bits chance-matches ~234
    # times; those windows must never read as positive sync evidence
    rng = np.random.default_rng(7)
    bits = rng.integers(0, 2, 60_000).astype(np.uint8)
    modem = Modem(symbol_rate=1.0, path=[SyncWordStep(sync="a7")])
    res = _bits_rx(modem, bits, tmp_path, "short_sync")
    assert res.status == "ok", res
    assert _not_trusted(res), res.quality
    assert res.quality is not None and res.quality.verdict == "uncertain"


def test_long_sync_on_random_bits_is_no_signal(tmp_path: Path) -> None:
    rng = np.random.default_rng(8)
    bits = rng.integers(0, 2, 60_000).astype(np.uint8)
    modem = Modem(symbol_rate=1.0, path=[SyncWordStep(sync="7cd215d8")])
    res = _bits_rx(modem, bits, tmp_path, "long_sync")
    assert res.status == "ok", res
    assert res.quality is not None and res.quality.verdict == "no_signal"
    assert any(
        e.metric == "sync_matches" and e.assessment == "negative"
        for e in res.quality.evidence
    )


# a generic (15,7) single-error-correcting code built from distinct
# weight->=2 parity-column signatures; nothing protocol-shaped about it
_SIGS = (3, 5, 6, 9, 10, 12, 17)
_N_PARITY = 8
_MASKS = [
    sum(((sig >> (_N_PARITY - 1 - p)) & 1) << j for j, sig in enumerate(_SIGS))
    for p in range(_N_PARITY)
]


def _codewords(rng: np.random.Generator, n_words: int) -> np.ndarray:
    data = rng.integers(0, 2, (n_words, len(_SIGS))).astype(np.uint8)
    parity = np.stack(
        [
            np.bitwise_xor.reduce(
                data[:, [j for j in range(len(_SIGS)) if (m >> j) & 1]], axis=1
            )
            for m in _MASKS
        ],
        axis=1,
    )
    return np.concatenate([data, parity], axis=1).reshape(-1)


def _block_code_modem() -> Modem:
    return Modem(
        symbol_rate=1.0,
        path=[
            BlockCodeStep(
                code_bits=len(_SIGS) + _N_PARITY,
                data_bits=len(_SIGS),
                parity_masks=_MASKS,
                correct=1,
            )
        ],
    )


def test_block_code_on_random_bits_is_no_signal(tmp_path: Path) -> None:
    # the C2 gate: garbage words pass a (15,7) t=1 check at the ~6% chance
    # rate; the word-validity tally must land negative, not silently absent
    rng = np.random.default_rng(9)
    bits = rng.integers(0, 2, 60_000).astype(np.uint8)
    res = _bits_rx(_block_code_modem(), bits, tmp_path, "bc_garbage")
    assert res.status == "ok", res
    assert res.quality is not None and res.quality.verdict == "no_signal"
    assert any(
        e.metric == "word_validity" and e.assessment == "negative"
        for e in res.quality.evidence
    )


def test_block_code_on_real_codewords_is_decoded(tmp_path: Path) -> None:
    rng = np.random.default_rng(10)
    bits = _codewords(rng, 4000)
    res = _bits_rx(_block_code_modem(), bits, tmp_path, "bc_real")
    assert res.status == "ok", res
    assert res.quality is not None and res.quality.verdict == "decoded"
    assert any(
        e.metric == "word_validity" and e.assessment == "positive"
        for e in res.quality.evidence
    )


def test_two_valid_codewords_are_insufficient_evidence(tmp_path: Path) -> None:
    # even perfectly valid words carry no verdict without statistical mass: two
    # (15,7) words cannot be told from luck at 5-sigma odds, so the honest
    # answer is uncertain, not decoded
    rng = np.random.default_rng(12)
    bits = _codewords(rng, 2)
    res = _bits_rx(_block_code_modem(), bits, tmp_path, "bc_tiny")
    assert res.status == "ok", res
    assert res.quality is not None and res.quality.verdict == "uncertain"


def test_pure_noise_css_marks_path_is_not_called_decoded(tmp_path: Path) -> None:
    ensure_worker_warm()
    rng = np.random.default_rng(11)
    n = 256 * 200
    noise = ((rng.normal(0, 1, n) + 1j * rng.normal(0, 1, n)) / np.sqrt(2)).astype(
        np.complex64
    )
    iq = tmp_path / "css_noise.iq"
    noise.tofile(iq)
    workdir = tmp_path / "rx"
    workdir.mkdir()
    res = run_rx(
        _css_marks_only_modem(),
        stage_registry(),
        sample_rate=_CSS_RATE,
        start=IQ,
        workdir=workdir,
        source_io={"path": str(iq)},
    )
    assert _not_trusted(res), (res.status, res.error, res.quality, res.marks)
