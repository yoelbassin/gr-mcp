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

from marconi.core.descriptor import Carrier, Descriptor
from marconi.core.levels import Level
from marconi.core.stages import validate_params
from marconi.phy.stages import stage_registry

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
    from marconi.phy.modulation.css.stages import ChirpSync

    out = ChirpSync().out_descriptor(IQ, _SF7_PARAMS)
    # IQ→IQ: level, item_type, layout, carrier all unchanged from base default
    assert out == IQ


def test_dechirp_out_descriptor_is_symbols_hard() -> None:
    from marconi.phy.modulation.css.stages import Dechirp

    out = Dechirp().out_descriptor(IQ, _SF7_PARAMS)
    assert out == SYMBOLS_HARD


def test_css_demap_out_descriptor_is_bits_hard() -> None:
    from marconi.phy.modulation.css.stages import CssDemap

    out = CssDemap().out_descriptor(SYMBOLS_HARD, _SF7_PARAMS)
    assert out == BITS_HARD


# ─── Param validation ─────────────────────────────────────────────────────────


def test_css_params_reject_sf_4() -> None:
    from marconi.phy.modulation.css.stages import ChirpSync

    bad: list = []
    validate_params("chirp_sync[0]", ChirpSync().params_model, {"sf": 4}, bad)
    assert bad, "sf=4 should be rejected"


def test_css_params_reject_sf_15() -> None:
    from marconi.phy.modulation.css.stages import Dechirp

    bad: list = []
    validate_params("dechirp[0]", Dechirp().params_model, {"sf": 15}, bad)
    assert bad, "sf=15 should be rejected"


def test_css_params_accept_sf_11() -> None:
    from marconi.phy.modulation.css.stages import CssDemap

    ok: list = []
    validate_params("css_demap[0]", CssDemap().params_model, {"sf": 11}, ok)
    assert not ok, "sf=11 should be accepted"


def test_css_params_require_sf() -> None:
    from marconi.phy.modulation.css.stages import ChirpSync

    missing: list = []
    validate_params("chirp_sync[0]", ChirpSync().params_model, {}, missing)
    assert missing, "sf is required"


def test_explicit_params_reject_sf_reduction_ge_sf_when_ldro() -> None:
    from marconi.phy.modulation.css.stages import CssExplicitDecode

    bad: list = []
    validate_params(
        "css_explicit_decode[0]",
        CssExplicitDecode().params_model,
        {
            "sf": 7,
            "ldro": True,
            "sf_reduction": 7,
            "header_symbols": 8,
            "header_nibbles": 5,
            "header_data_bits": 12,
            "header_parity": [3840, 2273, 1178, 599, 303],
            "field_payload_len": [0, 8],
            "field_cr": [8, 3],
            "field_has_crc": [11, 1],
            "field_parity": [15, 5],
        },
        bad,
    )
    assert bad, "sf_reduction >= sf with ldro should be rejected"


# ─── Direction support ────────────────────────────────────────────────────────


def test_all_css_stages_support_rx_and_tx() -> None:
    from marconi.phy.modulation.css.stages import ChirpSync, CssDemap, Dechirp

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
    from phy._dsp import write_bits

    from marconi.core.params import ParamValue
    from marconi.phy.backends.gnuradio.runner import GnuRadioBackend, ensure_worker_warm
    from marconi.phy.compiler import compile_modem
    from marconi.phy.models import ModemSpec, ModemStep
    from marconi.phy.stages import stage_registry

    SF, OS = 7, 2
    N_SYMS = 40
    SAMPLE_RATE = float(OS * (1 << SF))  # 256.0
    SYMBOL_RATE = 1.0  # bandwidth = symbol_rate * 2^SF = 128.0

    reg = stage_registry()
    bits = np.random.default_rng(42).integers(0, 2, SF * N_SYMS).astype(np.uint8)
    bits_path = write_bits(tmp_path / "in.bits", bits)
    iq_path = tmp_path / "frame.iq"
    sym_path = tmp_path / "out.sym"

    css_params: dict[str, ParamValue] = {"sf": SF, "oversample": OS}

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
