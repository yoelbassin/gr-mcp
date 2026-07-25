"""Stage-unit and SYMBOLS-terminal smoke tests for the CSS vertical.

Unit tests: registry presence, out_descriptor contracts, param validation,
direction support.

Smoke test: compile + run RX modem [chirp_sync, dechirp] (IQ→SYMBOLS) and
verify int16 symbols emerge that, when Gray-decoded, recover the sent bits.
This exercises the first SYMBOLS-terminal routing in the compiler.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from marconi.engine.stages.base import validate_params
from marconi.engine.stages.registry import stage_registry
from marconi.engine.types.descriptor import Carrier, Descriptor
from marconi.engine.types.levels import Level

IQ = Descriptor(Level.IQ, "c")
SYMBOLS_HARD = Descriptor(Level.SYMBOLS, "s", carrier=Carrier.HARD)
BITS_HARD = Descriptor(Level.BITS, "b", carrier=Carrier.HARD)

_SF7_PARAMS = {"sf": 7}


# ─── Registry ────────────────────────────────────────────────────────────────


def test_css_stages_registered() -> None:
    reg = stage_registry()
    assert reg["chirp_sync"].family == "css"
    assert reg["dechirp"].family == "css"
    assert reg["css_demap"].family == "css"


# ─── out_descriptor contracts ─────────────────────────────────────────────────


def test_chirp_sync_is_iq_to_iq_default_out_descriptor() -> None:
    from marconi.engine.modulation.css.stages import ChirpSync

    out = ChirpSync().out_descriptor(IQ, _SF7_PARAMS)
    # IQ→IQ: level, item_type, layout, carrier all unchanged from base default
    assert out == IQ


def test_dechirp_out_descriptor_is_symbols_hard() -> None:
    from marconi.engine.modulation.css.stages import Dechirp

    out = Dechirp().out_descriptor(IQ, _SF7_PARAMS)
    assert out == SYMBOLS_HARD


def test_css_demap_out_descriptor_is_bits_hard() -> None:
    from marconi.engine.modulation.css.stages import CssDemap

    out = CssDemap().out_descriptor(SYMBOLS_HARD, _SF7_PARAMS)
    assert out == BITS_HARD


# ─── Param validation ─────────────────────────────────────────────────────────


def test_css_params_reject_sf_4() -> None:
    from marconi.engine.modulation.css.stages import ChirpSync

    bad: list = []
    complete = {
        "sf": 4,
        "oversample": 2,
        "zero_pad": 4,
        "preamble_len": 8,
        "sfd_symbols": 2.25,
        "sync_symbols": 2,
    }
    validate_params("chirp_sync[0]", ChirpSync().params_model, complete, bad)
    assert bad, "sf=4 should be rejected"


def test_css_params_reject_sf_15() -> None:
    from marconi.engine.modulation.css.stages import Dechirp

    bad: list = []
    complete = {
        "sf": 15,
        "oversample": 2,
        "zero_pad": 4,
        "preamble_len": 8,
        "sfd_symbols": 2.25,
        "sync_symbols": 2,
    }
    validate_params("dechirp[0]", Dechirp().params_model, complete, bad)
    assert bad, "sf=15 should be rejected"


def test_css_params_accept_sf_11() -> None:
    from marconi.engine.modulation.css.stages import CssDemap

    ok: list = []
    complete = {
        "sf": 11,
        "oversample": 2,
        "zero_pad": 4,
        "preamble_len": 8,
        "sfd_symbols": 2.25,
        "sync_symbols": 2,
    }
    validate_params("css_demap[0]", CssDemap().params_model, complete, ok)
    assert not ok, "sf=11 should be accepted"


def test_css_params_require_sf() -> None:
    from marconi.engine.modulation.css.stages import ChirpSync

    issues: list = []
    validate_params("chirp_sync[0]", ChirpSync().params_model, {}, issues)
    assert {i.field for i in issues} >= {
        "sf",
        "oversample",
        "zero_pad",
        "preamble_len",
        "sfd_symbols",
        "sync_symbols",
    }


def test_css_params_bound_oversample() -> None:
    from marconi.engine.modulation.css.stages import Dechirp

    for osr, expect_ok in ((0, False), (1, True), (8, True), (9, False)):
        issues: list = []
        complete = {
            "sf": 7,
            "oversample": osr,
            "zero_pad": 4,
            "preamble_len": 8,
            "sfd_symbols": 2.25,
            "sync_symbols": 2,
        }
        validate_params("dechirp[0]", Dechirp().params_model, complete, issues)
        assert (not issues) is expect_ok, f"oversample={osr}: {issues}"


# ─── Direction support ────────────────────────────────────────────────────────


def test_all_css_stages_support_rx_and_tx() -> None:
    from marconi.engine.modulation.css.stages import ChirpSync, CssDemap, Dechirp

    for cls in (ChirpSync, Dechirp, CssDemap):
        s = cls()
        assert "rx" in s.directions, f"{cls.name} missing rx"
        assert "tx" in s.directions, f"{cls.name} missing tx"


# ─── SYMBOLS-terminal compile+run smoke ──────────────────────────────────────


def _gray_decode(g: int) -> int:
    n = g
    g >>= 1
    while g:
        n ^= g
        g >>= 1
    return n


def _symbols_to_bits(symbols: np.ndarray, sf: int) -> np.ndarray:
    """Gray-decode each int16 symbol and unpack sf bits MSB-first."""
    bits: list[int] = []
    for sym in symbols:
        s = _gray_decode(int(sym) & ((1 << sf) - 1))
        bits.extend((s >> (sf - 1 - j)) & 1 for j in range(sf))
    return np.asarray(bits, dtype=np.uint8)


def test_symbols_terminal_routing_smoke(tmp_path: Path) -> None:
    """Compile+run RX modem [chirp_sync, dechirp] → SYMBOLS sink (int16).

    This is the first SYMBOLS-terminal routing exercised through compile_modem.
    Verify the int16 symbols file is written and, after Gray-decode, the
    recovered bits match the transmitted payload bits.
    """
    from engine._dsp import write_bits

    from marconi.engine.backends.gnuradio.runner import (
        GnuRadioBackend,
        ensure_worker_warm,
    )
    from marconi.engine.compile.compiler import compile_modem
    from marconi.engine.stages.registry import stage_registry
    from marconi.engine.types.models import ModemSpec, ModemStep
    from marconi.engine.types.params import ParamValue

    SF, OS = 7, 2
    N_SYMS = 40
    SAMPLE_RATE = float(OS * (1 << SF))  # 256.0
    SYMBOL_RATE = 1.0  # bandwidth = symbol_rate * 2^SF = 128.0

    reg = stage_registry()
    bits = np.random.default_rng(42).integers(0, 2, SF * N_SYMS).astype(np.uint8)
    bits_path = write_bits(tmp_path / "in.bits", bits)
    iq_path = tmp_path / "frame.iq"
    sym_path = tmp_path / "out.sym"

    css_params: dict[str, ParamValue] = {
        "sf": SF,
        "oversample": OS,
        "zero_pad": 4,
        "preamble_len": 8,
        "sfd_symbols": 2.25,
        "sync_symbols": 2,
    }

    # --- TX: bits → IQ frame (via [chirp_sync, dechirp, css_demap]) ---
    tx_modem = ModemSpec(
        symbol_rate=SYMBOL_RATE,
        path=[
            ModemStep(conv="chirp_sync", params=css_params),
            ModemStep(conv="dechirp", params=css_params),
            ModemStep(conv="css_demap", params=css_params),
        ],
    )
    tx_pipe = compile_modem(
        tx_modem,
        reg,
        direction="tx",
        sample_rate=SAMPLE_RATE,
        start=IQ,
        source_io={"path": str(bits_path)},
        sink_io={"path": str(iq_path)},
    )

    # --- RX: IQ → SYMBOLS sink (int16) via [chirp_sync, dechirp] ---
    rx_modem = ModemSpec(
        symbol_rate=SYMBOL_RATE,
        path=[
            ModemStep(conv="chirp_sync", params=css_params),
            ModemStep(conv="dechirp", params=css_params),
        ],
    )
    rx_pipe = compile_modem(
        rx_modem,
        reg,
        direction="rx",
        sample_rate=SAMPLE_RATE,
        start=IQ,
        source_io={"path": str(iq_path)},
        sink_io={"path": str(sym_path)},
    )

    ensure_worker_warm()
    be = GnuRadioBackend()
    tx_result = be.run_pipeline(tx_pipe)
    assert tx_result.status == "ok", f"TX pipeline failed: {tx_result}"

    # chirp_sync's emission trails its always-on burst hunt, withholding a
    # bounded detector margin at EOF; pad with noise so the payload clears it
    # (real captures carry trailing noise anyway; zeros would pin the detector).
    sn = OS * (1 << SF)
    rng = np.random.default_rng(9)
    pad = 0.01 * (rng.standard_normal(12 * sn) + 1j * rng.standard_normal(12 * sn))
    frame = np.fromfile(iq_path, dtype=np.complex64)
    np.concatenate([frame, pad.astype(np.complex64)]).tofile(iq_path)

    rx_result = be.run_pipeline(rx_pipe)
    assert rx_result.status == "ok", f"RX pipeline failed: {rx_result}"

    # Read int16 symbols and verify non-trivial output
    symbols = np.fromfile(sym_path, dtype=np.int16)
    assert len(symbols) >= N_SYMS - 1, f"expected ~{N_SYMS} symbols, got {len(symbols)}"
    assert not np.all(symbols == 0), "all-zero symbols — likely a decode failure"

    # Gray-decode symbols → bits and compare to transmitted payload
    recovered_bits = _symbols_to_bits(symbols[:N_SYMS], SF)
    assert np.array_equal(
        recovered_bits, bits
    ), f"symbol→bit mismatch at indices {np.where(recovered_bits != bits)[0][:10]}"


def test_css_params_reject_preamble_len_below_5() -> None:
    from marconi.engine.modulation.css.stages import ChirpSync

    for bad in (3, 4):  # 3 = IndexError in _DetectScan.step, 4 = single-peak mislock
        issues: list = []
        # sync_symbols=1 (not baseline's 2): the _ok cross-field rule requires
        # sync_symbols < preamble_len - 2, so isolate the preamble-floor error.
        complete = {
            "sf": 7,
            "oversample": 2,
            "zero_pad": 4,
            "preamble_len": bad,
            "sfd_symbols": 2.25,
            "sync_symbols": 1,
        }
        validate_params(
            "chirp_sync[0]",
            ChirpSync().params_model,
            complete,
            issues,
        )
        assert issues, f"preamble_len={bad} should be rejected"


def test_css_params_accept_preamble_len_5() -> None:
    from marconi.engine.modulation.css.stages import ChirpSync

    ok: list = []
    complete = {
        "sf": 7,
        "oversample": 2,
        "zero_pad": 4,
        "preamble_len": 5,
        "sfd_symbols": 2.25,
        "sync_symbols": 2,
    }
    validate_params("chirp_sync[0]", ChirpSync().params_model, complete, ok)
    assert not ok, ok


def test_detect_scan_chunked_equals_oneshot() -> None:
    from marconi.engine.backends.gnuradio.embedded import chirp

    sf, os_, zp, pl = 7, 2, 4, 8
    grid = chirp._Grid(sf, os_, zp)
    sn = grid.sample_num
    up = chirp._base_upchirp(sf, os_)
    rng = np.random.default_rng(1)
    for pad_windows in (0, 3, 11):
        npad = pad_windows * sn + 37
        pad = 0.05 * (rng.standard_normal(npad) + 1j * rng.standard_normal(npad))
        tail = 0.05 * (rng.standard_normal(8 * sn) + 1j * rng.standard_normal(8 * sn))
        sig = np.concatenate(
            [pad, np.tile(up, pl), np.conj(np.tile(up, 3)), tail]
        ).astype(np.complex64)
        oneshot = chirp._DetectScan(grid, pl - 2).step(sig)
        chunked = chirp._DetectScan(grid, pl - 2)
        got = None
        for end in range(sn, len(sig) + 1, sn // 3):
            got = chunked.step(sig[:end])
            if got is not None:
                break
        assert oneshot is not None
        assert got == oneshot


def test_detect_scan_single_pass(monkeypatch) -> None:
    from marconi.engine.backends.gnuradio.embedded import chirp

    grid = chirp._Grid(7, 2, 4)
    sn = grid.sample_num
    calls = {"n": 0}
    orig = chirp._fine_peak

    def counting(*a, **k):  # untyped: mypy checks typed-def bodies even in tests
        calls["n"] += 1
        return orig(*a, **k)

    monkeypatch.setattr(chirp, "_fine_peak", counting)
    rng = np.random.default_rng(5)
    n_win = 200
    sig = (
        rng.standard_normal(n_win * sn) + 1j * rng.standard_normal(n_win * sn)
    ).astype(np.complex64)
    scan = chirp._DetectScan(grid, 6)
    for k in range(1, n_win + 1):
        assert scan.step(sig[: k * sn]) is None
    assert calls["n"] <= n_win


def test_sfd_sync_pending_vs_miss() -> None:
    from marconi.engine.backends.gnuradio.embedded import chirp

    sf, os_, zp, pl = 7, 2, 4, 8
    grid = chirp._Grid(sf, os_, zp)
    sn = grid.sample_num
    up = chirp._base_upchirp(sf, os_)
    sig = np.tile(up, pl).astype(np.complex64)  # preamble only, SFD not yet arrived
    assert chirp._sfd_sync(sig, 0, grid, (pl + 6) * sn, 2.25) is None
    partial = np.concatenate(
        [np.tile(up, pl), np.conj(np.tile(up, 2))[: int(1.6 * sn)]]
    ).astype(
        np.complex64
    )  # SFD found but its refinement window is still streaming in
    assert chirp._sfd_sync(partial, 0, grid, (pl + 6) * sn, 2.25) is None
    full = np.concatenate([np.tile(up, pl), np.conj(np.tile(up, 3))]).astype(
        np.complex64
    )
    ps = chirp._sfd_sync(full, 0, grid, (pl + 6) * sn, 2.25)
    assert ps is not None and ps > pl * sn
    long_up = np.tile(up, pl + 12).astype(np.complex64)  # no SFD within cap
    import pytest as _pytest

    with _pytest.raises(ValueError):
        chirp._sfd_sync(long_up, 0, grid, (pl + 6) * sn, 2.25)
