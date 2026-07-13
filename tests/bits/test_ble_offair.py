"""Real off-air Bluetooth LE advertising, phy through the generic descramble + crc
stages, with BLE's own CRC-24 as the self-validating oracle.

The PHY (channelize/fsk/slice) closes with zero new production code — the bursts
are too short for Gardner timing (it rails, recovering 1/7 packets), so the fsk
runs open-loop (loop_bw=0), the same sampler ADS-B needs. descramble dewhitens each
PDU (BLE whitening is a data-independent per-channel LFSR sequence) and crc checks
the CRC-24 — both exercised as generic stages. BLE's batch-free framing (per-PDU
whitening reset, length at a byte offset, length-driven CRC) exceeds the generic
framing vocabulary, so the AA-anchored carve is thin test-side code, per the roadmap.

Capture: Zenodo SDR4IoT dataset (Rtone/imec, CC-BY) — Nordic nRF52 advertising
beacons + an ambient Apple device, channel 37. The oracle needs no external decoder:
CRC-24 self-validates, cross-checked once by an independent decoder that read the
advertised name "Nordic_Blinky". 6/7 decode CRC-valid (the Apple burst is marginal).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from marconi.bits.carriers import RxCarrier, _Frame
from marconi.bits.framing import bits_to_bytes, crc_rx, descramble_rx
from marconi.core.bitfile import read_bits
from marconi.core.descriptor import Descriptor
from marconi.core.levels import Level
from marconi.phy.backends.gnuradio.runner import GnuRadioBackend, ensure_worker_warm
from marconi.phy.compiler import compile_modem
from marconi.phy.models import ModemSpec, ModemStep
from marconi.phy.stages import stage_registry

IQ = Descriptor(Level.IQ, "c")
RATE = 5_000_000.0
_SLICE = (
    Path(__file__).resolve().parents[2] / "artifacts" / "assets" / "BLE" / "ble.cf32"
)

# BLE constants (caller data — a protocol lives in the fixture, not production).
CHANNEL = 37
AA = 0x8E89BED6  # advertising access address
AA_BITS = np.unpackbits(
    np.frombuffer(AA.to_bytes(4, "little"), np.uint8), bitorder="little"
)
CRC_POLY, CRC_INIT = 0x65B, 0x555555  # BLE CRC-24
MAX_PDU_BYTES = 2 + 37 + 3  # header + max advertising payload + CRC

# Oracle: BLE CRC-24 self-validates; these 5 distinct advertising payloads (AdvA + AD
# structures) were cross-checked by an independent decoder that read "Nordic_Blinky".
ORACLE_PAYLOADS = {
    "6e86df7c1fe2031900000201060e094e6f726469635f426c696e6b79",
    "dfd070a4036a02011a020a0c0bff4c001006131a6586b678",
    "b40a5c1da6f9110723d1bcea5f782315deef121223150000",
    "b40a5c1da6f9031900000201060e094e6f726469635f426c696e6b79",
    "592e2f7b7cc5031900000201060e094e6f726469635f426c696e6b79",
}


def _ble_whitening(nbytes: int) -> bytes:
    """BLE data-whitening sequence for CHANNEL: LFSR x^7+x^4+1, register = 1 then the
    6 channel-index bits MSB-first, output = position 6, LSB-first within each byte."""
    reg = [1] + [(CHANNEL >> (5 - i)) & 1 for i in range(6)]
    out = bytearray()
    for _ in range(nbytes):
        b = 0
        for bit in range(8):
            fb = reg[6]
            b |= fb << bit
            reg = [fb] + reg[:6]
            reg[4] ^= fb
        out.append(b)
    return bytes(out)


WHITEN = _ble_whitening(MAX_PDU_BYTES)


def _ble_modem() -> ModemSpec:
    return ModemSpec(
        symbol_rate=1_000_000.0,
        path=[
            ModemStep(
                conv="channelize",
                params={
                    "decim": 1,
                    "bandwidth_hz": 2_000_000.0,
                    "center_hz": -1_000_000.0,
                },
            ),
            ModemStep(conv="fsk", params={"deviation": 250_000.0, "loop_bw": 0.0}),
            ModemStep(conv="slice", params={}),
        ],
    )


def _frame(payload: bytes) -> RxCarrier:
    return RxCarrier(
        bits=np.zeros(0, np.uint8),
        frames=[_Frame(start=0, cursor=0, payload=payload)],
    )


def _aa_positions(bits: np.ndarray) -> list[int]:
    win = np.lib.stride_tricks.sliding_window_view(bits, 32)
    return np.flatnonzero((win != AA_BITS).sum(axis=1) == 0).tolist()


def _pdus(bits: np.ndarray) -> list[tuple[bool, str]]:
    """Thin AA-anchored carve; the generic descramble + crc stages do the work: per
    access address, dewhiten a max span, read the length, validate the exact PDU."""
    out: list[tuple[bool, str]] = []
    for aa in _aa_positions(bits):
        span = bits[aa + 32 : aa + 32 + MAX_PDU_BYTES * 8]
        nbytes = int(span.size) // 8
        if nbytes < 2:
            continue
        raw = bits_to_bytes(span[: nbytes * 8], "lsb")
        dw = (
            descramble_rx(_frame(raw), sequence=WHITEN[:nbytes].hex()).frames[0].payload
        )
        need = 2 + (dw[1] & 0x3F) + 3
        if nbytes < need:
            continue
        f = crc_rx(
            _frame(dw[:need]),
            poly=CRC_POLY,
            bits=24,
            init=CRC_INIT,
            reflected=True,
            checksum_le=True,
        ).frames[0]
        out.append((bool(f.crc_ok), f.payload[2:].hex()))
    return out


@pytest.mark.skipif(
    not _SLICE.exists(), reason="BLE slice absent — run tests/bits/make_ble_slice.py"
)
def test_ble_offair(tmp_path: Path) -> None:
    ensure_worker_warm()
    snk = tmp_path / "ble_bits.u8"
    pipe = compile_modem(
        _ble_modem(),
        stage_registry(),
        direction="rx",
        sample_rate=RATE,
        start=IQ,
        source_io={"path": str(_SLICE)},
        sink_io={"path": str(snk)},
    )
    r = GnuRadioBackend().run_pipeline(pipe, timeout=180.0)
    assert r.status == "ok", r

    valid = [payload for ok, payload in _pdus(read_bits(snk)) if ok]
    # Closure: BLE PDUs decode phy -> descramble -> crc to CRC-valid, matching the
    # independently-decoded oracle. >=5 of 7 (the marginal Apple burst carries a
    # stable 1-byte demod error); a mis-decode fails CRC or lands off-oracle.
    assert len(valid) >= 5, f"only {len(valid)} CRC-valid BLE PDUs: {valid}"
    assert (
        set(valid) <= ORACLE_PAYLOADS
    ), f"off-oracle payload: {set(valid) - ORACLE_PAYLOADS}"
    # more than one advertiser, so it is not one packet decoded repeatedly
    assert len({p[:12] for p in valid}) >= 2, f"too few distinct advertisers: {valid}"
    assert any(
        b"Nordic_Blinky" in bytes.fromhex(p) for p in valid
    ), "Nordic_Blinky name absent"
