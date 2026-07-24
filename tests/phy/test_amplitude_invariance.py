import sys
from collections.abc import Callable
from pathlib import Path
from typing import cast

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent / "modulation"))
import _lattice  # noqa: E402  (sibling helper, tests/phy/modulation/_lattice.py)
from phy._dsp import (  # noqa: E402
    aligned_ber,
    channel,
    read_bits,
    read_complex,
    resolved_ser_hard,
    tx_sym_indices,
    write_bits,
)

from marconi.core.descriptor import Carrier, Descriptor  # noqa: E402
from marconi.core.levels import Level  # noqa: E402
from marconi.core.params import ParamValue  # noqa: E402
from marconi.phy.backends.gnuradio.runner import (  # noqa: E402
    GnuRadioBackend,
    ensure_worker_warm,
)
from marconi.phy.compiler import CompileError, compile_modem  # noqa: E402
from marconi.phy.models import ModemSpec, ModemStep  # noqa: E402
from marconi.phy.stages import stage_registry  # noqa: E402

IQ = Descriptor(Level.IQ, "c")
SYM_C = Descriptor(Level.SYMBOLS, "c", carrier=Carrier.SOFT)
_SR, _SYM = 8.0, 1.0


def _agc(mode: str = "feedforward", **params: float) -> ModemStep:
    return ModemStep(conv="agc", params={"mode": mode, **params})


# CSS operates at its own sample-per-symbol scale (oversample*2^sf), not the
# modem's _SR/_SYM -- geometry mirrors tests/phy/test_css_roundtrip.py's
# cheapest mode (sf=7, oversample=2).
_CSS_SF, _CSS_OSR = 7, 2
_CSS_RATE = _CSS_OSR * (1 << _CSS_SF) * _SYM
_CSS_N_BITS = _CSS_SF * 40
_CSS_PARAMS: dict[str, ParamValue] = {
    "sf": _CSS_SF,
    "oversample": _CSS_OSR,
    "zero_pad": 4,
    "preamble_len": 8,
    "sfd_symbols": 2.25,
    "sync_symbols": 2,
}

# ofdm_demod is RX-only (no TX conv in the vocabulary), so its recipe's IQ is
# synthesized directly (IFFT+CP), geometry from tests/phy/test_ofdm_demod_stage.py.
_OFDM_FFT, _OFDM_CP, _OFDM_SYM, _OFDM_NULL, _OFDM_NC = 16, 4, 20, 24, 4
_OFDM_ACTIVE = [1, 2, 14, 15]
_OFDM_BIN_PERM = cast(
    "list[float | int]",
    _OFDM_ACTIVE + [b for b in range(_OFDM_FFT) if b not in _OFDM_ACTIVE],
)
_OFDM_FRAME_SYMS = 4
_OFDM_N_FRAMES = 4  # >=5 back-to-back frames trip an unrelated resync issue:
# artifacts/superpowers/issues/28-ofdm-demod-multiframe-resync.md
_OFDM_PARAMS: dict[str, ParamValue] = {
    "fft_len": _OFDM_FFT,
    "cp_len": _OFDM_CP,
    "sym_len": _OFDM_SYM,
    "null_len": _OFDM_NULL,
    "frame_len": _OFDM_NULL + _OFDM_FRAME_SYMS * _OFDM_SYM,
    "n_frame_syms": _OFDM_FRAME_SYMS,
    "data_syms": _OFDM_FRAME_SYMS - 1,
    "n_carriers": _OFDM_NC,
    "bin_perm": _OFDM_BIN_PERM,
}

_OFDMC_N_SYMS = 200
_OFDMC_SEED = 11
_QPSK_CORNERS = np.array([1 + 1j, 1 - 1j, -1 + 1j, -1 - 1j]) / np.sqrt(2)


_RECIPES: dict[str, dict[str, object]] = {
    "ook_envelope": {
        "agc": _agc(),
        "path": [ModemStep(conv="ook_envelope"), ModemStep(conv="slice")],
    },
    "qam_demod": {
        # order 16: the matrix's claim is GAIN-invariance, which order 16 shows
        # cleanly. Order 64's ring spacing entangles the documented power-mode
        # startup fragility (a window-robustness issue, not a gain issue) — its
        # SER-0 is already proven by the round-trip in Task 6.
        "agc": _agc("power", window_symbols=128.0),
        "path": [
            ModemStep(conv="qam_demod", params={"order": 16}),
            ModemStep(conv="qam_demap", params={"order": 16}),
        ],
    },
    "psk_demod": {
        # attack/decay raised 1.0/4.0 -> 16.0/16.0 (see report): agc2_cc's
        # additive gain update overshoots past zero on a 1000x static step at
        # the settled-fact time constant, entering a non-converging limit
        # cycle; 16/16 symbols is a verified stable plateau (14-24 all clean).
        "agc": _agc("feedback", attack_symbols=16.0, decay_symbols=16.0),
        "path": [
            ModemStep(conv="psk_demod", params={"order": 2}),
            ModemStep(conv="psk_demap", params={"order": 2}),
        ],
        "settle": 1536,
        "max_shift": 1856,
    },
    "fsk": {
        "agc": _agc(),
        "path": [
            ModemStep(conv="fsk", params={"deviation": 1.0}),
            ModemStep(conv="slice"),
        ],
    },
    "dechirp": {
        # window_symbols=1.0 (not the agc default of 16): a CSS "symbol" is
        # oversample*2^sf raw samples, so the default window spans nearly the
        # whole preamble and corrupts the very edge chirp_sync locks onto.
        "agc": _agc(window_symbols=1.0),
        "path": [
            ModemStep(conv="chirp_sync", params=_CSS_PARAMS),
            ModemStep(conv="dechirp", params=_CSS_PARAMS),
            ModemStep(conv="css_demap", params=_CSS_PARAMS),
        ],
        "sample_rate": _CSS_RATE,
        "n_bits": _CSS_N_BITS,
        "max_shift": 8 * _CSS_SF,
    },
    "msk": {
        "agc": _agc(),
        "path": [ModemStep(conv="msk"), ModemStep(conv="slice")],
        # msk is RX-only; TX is the fsk stage at h=0.5 (deviation=symbol_rate/4),
        # matching tests/phy/test_msk_roundtrip.py.
        "tx_path": [
            ModemStep(conv="fsk", params={"deviation": _SYM / 4.0}),
            ModemStep(conv="slice"),
        ],
        "sample_rate": 20.0,  # the ACARS operating point (test_msk_roundtrip.py)
        "n_bits": 2048,
    },
    "ofdm_demod": {
        "agc": _agc(),
        "path": [
            ModemStep(conv="ofdm_demod", params=_OFDM_PARAMS),
            ModemStep(conv="psk_demap", params={"order": 4}),
        ],
    },
    "ofdm_coherent_sync": {
        "agc": _agc(),
        "path": [ModemStep(conv="ofdm_coherent_sync", params=_lattice.sync_params())],
    },
}


def _recipe_path(name: str) -> list[ModemStep]:
    return list(cast("list[ModemStep]", _RECIPES[name]["path"]))


def _demod_stage_names() -> set[str]:
    return {
        name
        for name, stage in stage_registry().items()
        if stage.from_level is Level.IQ and stage.to_level is Level.SYMBOLS
    }


def test_every_demod_stage_has_an_amplitude_recipe() -> None:
    missing = _demod_stage_names() - set(_RECIPES)
    assert not missing, (
        f"demod stages with no amplitude-invariance recipe: {sorted(missing)}. "
        "Add a recipe so the new stage is covered by the invariance matrix."
    )


_GAINS = [1e-3, 1.0, 1e3]


def _tx_path(name: str) -> list[ModemStep]:
    raw = _RECIPES[name].get("tx_path", _RECIPES[name]["path"])
    return list(cast("list[ModemStep]", raw))


def _sample_rate(name: str) -> float:
    return float(cast(float, _RECIPES[name].get("sample_rate", _SR)))


def _n_bits(name: str) -> int:
    return int(cast(int, _RECIPES[name].get("n_bits", 4096)))


def _max_shift(name: str) -> int:
    return int(cast(int, _RECIPES[name].get("max_shift", 256)))


def _settle(name: str) -> int:
    return int(cast(int, _RECIPES[name].get("settle", 0)))


def _agc_for(name: str, agc_override: ModemStep | None) -> ModemStep:
    if agc_override is not None:
        return agc_override
    return cast(ModemStep, _RECIPES[name]["agc"])


def _apply(z: np.ndarray, gain: float, condition: str) -> np.ndarray:
    if condition == "static":
        return z * gain
    if condition == "fade":
        envelope = np.linspace(1.0, 0.05, z.size)
        return z * gain * envelope
    if condition == "burst":
        keep = (np.arange(z.size) // 512) % 2 == 0
        return z * gain * keep
    raise AssertionError(condition)


def _compile_pipeline(
    direction: str,
    path: list[ModemStep],
    sample_rate: float,
    symbol_rate: float,
    start: Descriptor,
    src: Path,
    snk: Path,
):
    return compile_modem(
        ModemSpec(symbol_rate=symbol_rate, path=path),
        stage_registry(),
        direction=direction,
        sample_rate=sample_rate,
        start=start,
        source_io={"path": str(src)},
        sink_io={"path": str(snk)},
    )


def _paths(
    name: str, with_agc: bool, agc_override: ModemStep | None = None
) -> tuple[list[ModemStep], list[ModemStep]]:
    rx = _recipe_path(name)
    tx = _tx_path(name)
    if not with_agc:
        return rx, tx
    return [_agc_for(name, agc_override)] + rx, tx


def _roundtrip_ber(
    tmp_path: Path,
    name: str,
    gain: float,
    with_agc: bool,
    condition: str,
    agc_override: ModemStep | None = None,
) -> float:
    be = GnuRadioBackend()
    rx_path, tx_path = _paths(name, with_agc, agc_override)
    rate = _sample_rate(name)
    bits = np.random.default_rng(0).integers(0, 2, _n_bits(name)).astype(np.uint8)
    bp = write_bits(tmp_path / f"{name}_in.bits", bits)
    iq = tmp_path / f"{name}.iq"
    scaled = tmp_path / f"{name}_{gain}_{with_agc}_{condition}.iq"
    out = tmp_path / f"{name}_{gain}_{with_agc}_{condition}.bits"

    assert (
        be.run_pipeline(_compile_pipeline("tx", tx_path, rate, _SYM, IQ, bp, iq)).status
        == "ok"
    )
    z = np.fromfile(iq, dtype=np.complex64)
    z = _apply(z, gain, condition)
    z.astype(np.complex64).tofile(scaled)
    assert (
        be.run_pipeline(
            _compile_pipeline("rx", rx_path, rate, _SYM, IQ, scaled, out)
        ).status
        == "ok"
    )
    return aligned_ber(
        read_bits(out), bits[_settle(name) :], max_shift=_max_shift(name)
    )


_QAM_ORDER = 16
_QAM_N_BITS = 24000
_QAM_STO = 1.5
_QAM_SETTLE_SYMS = 1500


def _qam_oracle(order: int) -> tuple[np.ndarray, int]:
    from gnuradio import digital  # in-process GR for oracle ground truth (allowed)

    if order == 16:
        c = digital.constellation_16qam()
    else:
        from gnuradio.digital import mod_codes, qam

        c = qam.qam_constellation(
            constellation_points=order,
            differential=False,
            mod_code=mod_codes.GRAY_CODE,
            large_ampls_to_corners=False,
        )
    return np.asarray(c.points()), int(c.bits_per_symbol())


def _qam_ser(
    tmp_path: Path,
    gain: float,
    with_agc: bool,
    condition: str,
    agc_override: ModemStep | None,
    order: int = _QAM_ORDER,
) -> float:
    be = GnuRadioBackend()
    rate = _sample_rate("qam_demod")
    bits = np.random.default_rng(0).integers(0, 2, _QAM_N_BITS).astype(np.uint8)
    bp = write_bits(tmp_path / "qam_in.bits", bits)
    clean = tmp_path / "qam_clean.iq"
    tx_path = [
        ModemStep(conv="qam_demod", params={"order": order}),
        ModemStep(conv="qam_demap", params={"order": order}),
    ]
    tx_pipe = _compile_pipeline("tx", tx_path, rate, _SYM, IQ, bp, clean)
    assert be.run_pipeline(tx_pipe).status == "ok"

    # constellation_receiver_cb's decision-directed loop needs SOME channel
    # dither to lock (see report): a bit-exact identity channel never
    # converges. A fixed, deterministic STO is held constant across the gain
    # sweep so it never confounds the gain-invariance claim.
    base = tmp_path / "qam_base.iq"
    channel(clean, base, sample_rate=rate, seed=0, sto=_QAM_STO)
    z = np.fromfile(base, dtype=np.complex64)
    z = _apply(z, gain, condition)
    scaled = tmp_path / f"qam_scaled_{gain}_{with_agc}_{condition}.iq"
    z.astype(np.complex64).tofile(scaled)

    rx_path = [ModemStep(conv="qam_demod", params={"order": order})]
    if with_agc:
        rx_path = [_agc_for("qam_demod", agc_override)] + rx_path
    out = tmp_path / f"qam_out_{gain}_{with_agc}_{condition}.u8"
    rx_pipe = _compile_pipeline("rx", rx_path, rate, _SYM, IQ, scaled, out)
    assert be.run_pipeline(rx_pipe).status == "ok"

    rx_idx = read_bits(out)
    points, k = _qam_oracle(order)
    tsi = tx_sym_indices(bits, k)
    return resolved_ser_hard(rx_idx, tsi, points, settle=_QAM_SETTLE_SYMS)


def _msk_best_ber(out: np.ndarray, bits: np.ndarray) -> float:
    h = out.astype(np.uint8)
    nrzi = (h[1:] ^ h[:-1]).astype(np.uint8)
    return min(aligned_ber(nrzi, bits), aligned_ber(1 - nrzi, bits))


def _msk_ber(
    tmp_path: Path,
    gain: float,
    with_agc: bool,
    condition: str,
    agc_override: ModemStep | None,
) -> float:
    be = GnuRadioBackend()
    rate = _sample_rate("msk")
    bits = np.random.default_rng(0).integers(0, 2, _n_bits("msk")).astype(np.uint8)
    bp = write_bits(tmp_path / "msk_in.bits", bits)
    clean = tmp_path / "msk_clean.iq"
    tx_pipe = _compile_pipeline("tx", _tx_path("msk"), rate, _SYM, IQ, bp, clean)
    assert be.run_pipeline(tx_pipe).status == "ok"
    z = np.fromfile(clean, dtype=np.complex64)
    z = _apply(z, gain, condition)
    scaled = tmp_path / f"msk_scaled_{gain}_{with_agc}_{condition}.iq"
    z.astype(np.complex64).tofile(scaled)

    rx_path = _recipe_path("msk")
    if with_agc:
        rx_path = [_agc_for("msk", agc_override)] + rx_path
    out = tmp_path / f"msk_out_{gain}_{with_agc}_{condition}.bits"
    rx_pipe = _compile_pipeline("rx", rx_path, rate, _SYM, IQ, scaled, out)
    assert be.run_pipeline(rx_pipe).status == "ok"
    return _msk_best_ber(read_bits(out), bits)


def _ofdm_demod_points(tmp_path: Path, bits: np.ndarray) -> np.ndarray:
    be = GnuRadioBackend()
    bp = write_bits(tmp_path / "ofdm_demod_bits.bits", bits)
    sym = tmp_path / "ofdm_demod_points.cf32"
    pipe = _compile_pipeline(
        "tx",
        [ModemStep(conv="psk_demap", params={"order": 4})],
        1.0,
        1.0,
        SYM_C,
        bp,
        sym,
    )
    assert be.run_pipeline(pipe).status == "ok"
    return read_complex(sym)


def _ofdm_demod_iq(points: np.ndarray) -> np.ndarray:
    n_syms = _OFDM_N_FRAMES * _OFDM_FRAME_SYMS
    points = points[: n_syms * _OFDM_NC].reshape(n_syms, _OFDM_NC)
    frames = []
    for f in range(_OFDM_N_FRAMES):
        frames.append(np.zeros(_OFDM_NULL, np.complex64))
        for s in range(_OFDM_FRAME_SYMS):
            spec = np.zeros(_OFDM_FFT, complex)
            spec[_OFDM_ACTIVE] = points[f * _OFDM_FRAME_SYMS + s]
            u = np.fft.ifft(spec)
            frames.append(np.concatenate([u[-_OFDM_CP:], u]).astype(np.complex64))
    frames.append(np.zeros(_OFDM_NULL, np.complex64))  # forward null: don't withhold
    return np.concatenate(frames).astype(np.complex64)


def _ofdm_demod_ber(
    tmp_path: Path,
    gain: float,
    with_agc: bool,
    condition: str,
    agc_override: ModemStep | None,
) -> float:
    be = GnuRadioBackend()
    n_bits = _OFDM_N_FRAMES * _OFDM_FRAME_SYMS * _OFDM_NC * 2
    bits = np.random.default_rng(0).integers(0, 2, n_bits).astype(np.uint8)
    points = _ofdm_demod_points(tmp_path, bits)
    z = _ofdm_demod_iq(points)
    z = _apply(z, gain, condition)
    scaled = tmp_path / f"ofdm_demod_{gain}_{with_agc}_{condition}.iq"
    z.astype(np.complex64).tofile(scaled)

    rx_path = _recipe_path("ofdm_demod")
    if with_agc:
        rx_path = [_agc_for("ofdm_demod", agc_override)] + rx_path
    out = tmp_path / f"ofdm_demod_out_{gain}_{with_agc}_{condition}.bits"
    pipe = _compile_pipeline("rx", rx_path, 1.0, 1.0, IQ, scaled, out)
    assert be.run_pipeline(pipe).status == "ok"
    return aligned_ber(read_bits(out), bits, max_shift=64)


def _ofdm_coherent_resolved_ser(eq: np.ndarray, n_syms: int, seed: int) -> float:
    mask = _lattice.data_mask()
    best = 1.0
    for phi in range(_lattice.NS):
        grid = _lattice.truth_grid(n_syms, phi, seed)
        errs, tot = 0, 0
        for i in range(min(len(eq), n_syms)):
            fs = i % _lattice.NS
            m = mask[fs]
            cell, truth = eq[i][m], grid[i][m]
            if not np.all(np.isfinite(cell)):
                return 1.0
            dec = np.argmin(np.abs(cell[:, None] - _QPSK_CORNERS[None, :]), axis=1)
            tdec = np.argmin(np.abs(truth[:, None] - _QPSK_CORNERS[None, :]), axis=1)
            errs += int(np.sum(dec != tdec))
            tot += int(m.sum())
        best = min(best, errs / tot if tot else 1.0)
    return best


def _ofdm_coherent_ser(
    tmp_path: Path,
    gain: float,
    with_agc: bool,
    condition: str,
    agc_override: ModemStep | None,
) -> float:
    be = GnuRadioBackend()
    z = _lattice.make_iq(_OFDMC_N_SYMS, snr_db=30.0, seed=_OFDMC_SEED)
    z = _apply(z, gain, condition)
    scaled = tmp_path / f"ofdm_coh_{gain}_{with_agc}_{condition}.iq"
    z.astype(np.complex64).tofile(scaled)

    rx_path = _recipe_path("ofdm_coherent_sync")
    if with_agc:
        rx_path = [_agc_for("ofdm_coherent_sync", agc_override)] + rx_path
    out = tmp_path / f"ofdm_coh_out_{gain}_{with_agc}_{condition}.cf32"
    pipe = _compile_pipeline(
        "rx", rx_path, _lattice.RATE, _lattice.RATE / _lattice.SYM_LEN, IQ, scaled, out
    )
    assert be.run_pipeline(pipe, timeout=120.0).status == "ok"
    ncell = len(_lattice.EMIT)
    eq = np.fromfile(out, np.complex64).reshape(-1, ncell)
    return _ofdm_coherent_resolved_ser(eq, _OFDMC_N_SYMS, _OFDMC_SEED)


_Measure = Callable[[Path, float, bool, str, ModemStep | None], float]
_BESPOKE: dict[str, _Measure] = {
    "qam_demod": _qam_ser,
    "msk": _msk_ber,
    "ofdm_demod": _ofdm_demod_ber,
    "ofdm_coherent_sync": _ofdm_coherent_ser,
}


def _measure(
    tmp_path: Path,
    name: str,
    gain: float,
    with_agc: bool,
    condition: str,
    agc_override: ModemStep | None = None,
) -> float:
    bespoke = _BESPOKE.get(name)
    if bespoke is not None:
        return bespoke(tmp_path, gain, with_agc, condition, agc_override)
    return _roundtrip_ber(tmp_path, name, gain, with_agc, condition, agc_override)


@pytest.mark.parametrize("name", sorted(_RECIPES))
@pytest.mark.parametrize("gain", _GAINS)
def test_static_offset_invariance_with_agc(
    name: str, gain: float, tmp_path: Path
) -> None:
    ensure_worker_warm()
    ber = _measure(tmp_path, name, gain, with_agc=True, condition="static")
    assert ber == 0.0, f"{name} at gain {gain}: BER {ber}"


@pytest.mark.parametrize("name", sorted(_RECIPES))
def test_slow_fade_invariance_with_agc(name: str, tmp_path: Path) -> None:
    ensure_worker_warm()
    ber = _measure(tmp_path, name, 1.0, with_agc=True, condition="fade")
    assert ber == 0.0, f"{name} under slow fade: BER {ber}"


def _requires_amplitude(name: str) -> bool:
    return stage_registry()[name].accepts_amplitude is not None


@pytest.mark.parametrize("name", sorted(_RECIPES))
def test_control_arm_without_agc(name: str, tmp_path: Path) -> None:
    ensure_worker_warm()
    if _requires_amplitude(name):
        with pytest.raises(CompileError):
            _measure(tmp_path, name, 1e-3, with_agc=False, condition="static")
        return
    bers = [
        _measure(tmp_path, name, g, with_agc=False, condition="static") for g in _GAINS
    ]
    print(f"\ncontrol {name} no-agc BER vs gain -> {dict(zip(_GAINS, bers))}")
    assert bers[_GAINS.index(1.0)] == 0.0, f"{name} baseline broken: {bers}"


_BURST_RECIPE = "ook_envelope"


def _burst_ber(tmp_path: Path, mode: str) -> float:
    params: dict[str, float] = {}
    if mode == "feedback":
        params = {"attack_symbols": 1.0, "decay_symbols": 4.0}
    override = ModemStep(conv="agc", params={"mode": mode, **params})
    return _roundtrip_ber(
        tmp_path,
        _BURST_RECIPE,
        1.0,
        with_agc=True,
        condition="burst",
        agc_override=override,
    )


def test_feedforward_is_no_worse_than_feedback_under_gaps(tmp_path: Path) -> None:
    ensure_worker_warm()
    ff = _burst_ber(tmp_path, "feedforward")
    fb = _burst_ber(tmp_path, "feedback")
    print(f"\nburst {_BURST_RECIPE} -> feedforward {ff}, feedback {fb}")
    assert ff <= fb + 0.05, f"feedforward {ff} worse than feedback {fb}"


def test_qam_wrong_agc_mode_mis_decodes(tmp_path: Path) -> None:
    ensure_worker_warm()
    ser = _qam_ser(
        tmp_path,
        gain=1.0,
        with_agc=True,
        condition="static",
        agc_override=_agc("feedforward"),
        order=64,
    )
    print(f"\nQAM-64, wrong agc mode (feedforward) -> SER {ser}")
    # Amplitude has a single NORMALIZED member by design, so [agc(mode=any),
    # qam_demod] type-checks regardless of mode -- only `power` mode delivers
    # the unit-RMS scale QAM's fixed-radius geometry needs. This substantiates
    # that a wrong-mode QAM recipe is still caught at the test layer, since the
    # type system intentionally does not distinguish normalization statistics.
    assert ser > 0.3, f"expected catastrophic SER for wrong-mode QAM, got {ser}"
