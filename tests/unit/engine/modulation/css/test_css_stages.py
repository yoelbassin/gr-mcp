"""Stage-unit and SYMBOLS-terminal smoke tests for the CSS vertical.

Unit tests: registry presence, out_descriptor contracts, param validation,
direction support.

Smoke test: compile + run RX modem [chirp_sync, dechirp] (IQ→SYMBOLS) and
verify int16 symbols emerge that, when Gray-decoded, recover the sent bits.
This exercises the first SYMBOLS-terminal routing in the compiler.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pytest
from pydantic import ValidationError

from marconi.engine.modulation.css.stages import (
    ChirpSyncStep,
    CssDemapStep,
    DechirpStep,
)
from marconi.engine.stages.registry import stage_registry
from marconi.engine.types.descriptor import Carrier, Descriptor
from marconi.engine.types.enums import ItemType
from marconi.engine.types.levels import Level

IQ = Descriptor(Level.IQ, ItemType.C)
SYMBOLS_HARD = Descriptor(Level.SYMBOLS, ItemType.S, carrier=Carrier.HARD)
BITS_HARD = Descriptor(Level.BITS, ItemType.B, carrier=Carrier.HARD)


# ─── Registry ────────────────────────────────────────────────────────────────


def test_css_stages_registered() -> None:
    reg = stage_registry()
    assert reg["chirp_sync"].family == "css"
    assert reg["dechirp"].family == "css"
    assert reg["css_demap"].family == "css"


# ─── out_descriptor contracts ─────────────────────────────────────────────────


def test_chirp_sync_is_iq_to_iq_default_out_descriptor() -> None:
    from marconi.engine.modulation.css.stages import ChirpSync

    step = ChirpSyncStep(
        sf=7, oversample=2, zero_pad=4, preamble_len=8, sfd_symbols=2.25, sync_symbols=2
    )
    out = ChirpSync().out_descriptor(IQ, step)
    # IQ→IQ: level, item_type, carrier all unchanged from base default
    assert out == IQ


def test_dechirp_out_descriptor_is_symbols_hard() -> None:
    from marconi.engine.modulation.css.stages import Dechirp

    out = Dechirp().out_descriptor(IQ, DechirpStep(sf=7, oversample=2, zero_pad=4))
    assert out == Descriptor(Level.SYMBOLS, ItemType.S, carrier=Carrier.HARD, order=128)


def test_css_demap_out_descriptor_is_bits_hard() -> None:
    from marconi.engine.modulation.css.stages import CssDemap

    out = CssDemap().out_descriptor(SYMBOLS_HARD, CssDemapStep(sf=7))
    assert out == BITS_HARD


# ─── Param validation ─────────────────────────────────────────────────────────


def test_css_params_reject_sf_4() -> None:
    with pytest.raises(ValidationError):
        ChirpSyncStep(
            sf=4,
            oversample=2,
            zero_pad=4,
            preamble_len=8,
            sfd_symbols=2.25,
            sync_symbols=2,
        )


def test_css_params_reject_sf_15() -> None:
    with pytest.raises(ValidationError):
        DechirpStep(sf=15, oversample=2, zero_pad=4)


def test_css_params_accept_sf_11() -> None:
    CssDemapStep(sf=11)


def test_css_params_require_sf() -> None:
    with pytest.raises(ValidationError) as exc:
        ChirpSyncStep()  # type: ignore[call-arg]
    fields = {err["loc"][0] for err in exc.value.errors()}
    assert fields >= {
        "sf",
        "oversample",
        "zero_pad",
        "preamble_len",
        "sfd_symbols",
        "sync_symbols",
    }


def test_css_params_bound_oversample() -> None:
    for osr, expect_ok in ((0, False), (1, True), (8, True), (9, False)):
        if expect_ok:
            DechirpStep(sf=7, oversample=osr, zero_pad=4)
        else:
            with pytest.raises(ValidationError):
                DechirpStep(sf=7, oversample=osr, zero_pad=4)


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
    from helpers._dsp import write_bits

    from marconi.engine.backends.gnuradio.runner import (
        GnuRadioBackend,
        ensure_worker_warm,
    )
    from marconi.engine.compile.compiler import compile_modem
    from marconi.engine.stages.registry import stage_registry
    from marconi.engine.types.models import Modem

    SF, OS = 7, 2
    N_SYMS = 40
    SAMPLE_RATE = float(OS * (1 << SF))  # 256.0
    SYMBOL_RATE = 1.0  # bandwidth = symbol_rate * 2^SF = 128.0

    reg = stage_registry()
    bits = np.random.default_rng(42).integers(0, 2, SF * N_SYMS).astype(np.uint8)
    bits_path = write_bits(tmp_path / "in.bits", bits)
    iq_path = tmp_path / "frame.iq"
    sym_path = tmp_path / "out.sym"

    chirp_sync = ChirpSyncStep(
        sf=SF,
        oversample=OS,
        zero_pad=4,
        preamble_len=8,
        sfd_symbols=2.25,
        sync_symbols=2,
    )
    dechirp = DechirpStep(sf=SF, oversample=OS, zero_pad=4)
    demap = CssDemapStep(sf=SF)

    # --- TX: bits → IQ frame (via [chirp_sync, dechirp, css_demap]) ---
    tx_modem = Modem(symbol_rate=SYMBOL_RATE, path=[chirp_sync, dechirp, demap])
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
    rx_modem = Modem(symbol_rate=SYMBOL_RATE, path=[chirp_sync, dechirp])
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
    for bad in (3, 4):  # 3 = IndexError in _DetectScan.step, 4 = single-peak mislock
        # sync_symbols=1 (not baseline's 2): the _ok cross-field rule requires
        # sync_symbols < preamble_len - 2, so isolate the preamble-floor error.
        with pytest.raises(ValidationError):
            ChirpSyncStep(
                sf=7,
                oversample=2,
                zero_pad=4,
                preamble_len=bad,
                sfd_symbols=2.25,
                sync_symbols=1,
            )


def test_css_params_accept_preamble_len_5() -> None:
    ChirpSyncStep(
        sf=7,
        oversample=2,
        zero_pad=4,
        preamble_len=5,
        sfd_symbols=2.25,
        sync_symbols=2,
    )


def test_css_params_accept_no_sfd_no_sync() -> None:
    # sfd_symbols=0 (no SFD) + sync_symbols=0 (no gap) is the minimal CSS
    # anatomy: a bare preamble run straight into payload. Anything less
    # parametric is LoRa's frame layout masquerading as the CSS family.
    ChirpSyncStep(
        sf=7,
        oversample=2,
        zero_pad=4,
        preamble_len=8,
        sfd_symbols=0.0,
        sync_symbols=0,
    )


def test_css_params_reject_sub_symbol_sfd() -> None:
    # (0, 1) is neither "no SFD" nor a detectable one: a window holding a
    # fraction of a down-chirp never dominates its up-chirp energy.
    with pytest.raises(ValidationError):
        ChirpSyncStep(
            sf=7,
            oversample=2,
            zero_pad=4,
            preamble_len=8,
            sfd_symbols=0.5,
            sync_symbols=0,
        )


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


def test_detect_scan_single_pass(monkeypatch: pytest.MonkeyPatch) -> None:
    from marconi.engine.backends.gnuradio.embedded import chirp

    grid = chirp._Grid(7, 2, 4)
    sn = grid.sample_num
    calls = {"n": 0}
    orig = chirp._fine_peak

    def counting(*a: Any, **k: Any) -> tuple[float, int]:
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


def test_preamble_end_pending_vs_miss() -> None:
    # the no-SFD analog of _sfd_sync: anchor on the first window that departs
    # the base-chirp bin, with the same pending/found/miss trichotomy
    from marconi.engine.backends.gnuradio.embedded import chirp

    sf, os_, zp, pl = 7, 2, 4, 8
    grid = chirp._Grid(sf, os_, zp)
    sn = grid.sample_num
    up = chirp._base_upchirp(sf, os_)
    cap = (pl + 2) * sn

    pending = np.tile(up, pl).astype(np.complex64)  # payload not yet arrived
    assert chirp._preamble_end(pending, 0, grid, cap) is None

    found = np.concatenate(
        [np.tile(up, pl), chirp._modulate_symbol(40, sf, os_), np.tile(up, 2)]
    ).astype(np.complex64)
    assert chirp._preamble_end(found, 0, grid, cap) == pl * sn

    long_up = np.tile(up, pl + 12).astype(np.complex64)  # run never departs
    import pytest as _pytest

    with _pytest.raises(ValueError):
        chirp._preamble_end(long_up, 0, grid, cap)
