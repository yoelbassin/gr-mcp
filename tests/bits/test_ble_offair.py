"""Real off-air Bluetooth LE advertising decoded by a composed codec — the proof
that a length-under-scrambling protocol fits the generic framing vocabulary.

The PHY (channelize/fsk/slice) closes with zero new production code — the bursts
are too short for Gardner timing (it rails, recovering 1/7 packets), so the fsk
runs open-loop (loop_bw=0), the same sampler ADS-B needs. The codec then carves
each PDU without protocol code: sync_word seeds on the access address,
fixed_frame carves the max span, frame_codebook bridges BLE's LSB-first on-air
byte order to the stages' MSB packing, descramble dewhitens (the per-PDU
whitening reset falls out of carved scope), length_frame re-carves by the
length byte read from the DEWHITENED payload, and crc checks the CRC-24.

Capture: Zenodo SDR4IoT dataset (Rtone/imec, CC-BY) — Nordic nRF52 advertising
beacons + an ambient Apple device, channel 37. The oracle needs no external decoder:
CRC-24 self-validates, cross-checked once by an independent decoder that read the
advertised name "Nordic_Blinky". 6/7 decode CRC-valid (the Apple burst is marginal).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from marconi.bits.models import CodecSpec, CodecStep
from marconi.bits.registry import registry
from marconi.bits.seam import parse_bitstream
from marconi.core.bitfile import read_bits
from marconi.core.descriptor import Descriptor
from marconi.core.levels import Level
from marconi.core.models import Bitstream
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
AA_HEX = np.packbits(AA_BITS).tobytes().hex()  # on-air bits in sync_word's MSB packing
CRC_POLY, CRC_INIT = 0x65B, 0x555555  # BLE CRC-24
MAX_PDU_BYTES = 2 + 37 + 3  # header + max advertising payload + CRC
# BLE is LSB-first on air; an 8-bit codebook expresses the per-byte bridge to the
# stages' MSB byte packing.
BIT_REVERSE = [int(f"{i:08b}"[::-1], 2) for i in range(256)]

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


def _ble_codec() -> CodecSpec:
    return CodecSpec(
        path=[
            CodecStep(conv="sync_word", params={"sync": AA_HEX, "max_errors": 0}),
            CodecStep(conv="fixed_frame", params={"payload_bits": MAX_PDU_BYTES * 8}),
            CodecStep(
                conv="frame_codebook",
                params={"code_bits": 8, "data_bits": 8, "table": BIT_REVERSE},
            ),
            CodecStep(conv="descramble", params={"sequence": WHITEN.hex()}),
            CodecStep(
                conv="length_frame",
                params={
                    "length_bits": 8,
                    "offset_bits": 8,
                    "base_bytes": 2 + 3,
                    "unit_bytes": 1,
                },
            ),
            CodecStep(
                conv="crc",
                params={
                    "poly": CRC_POLY,
                    "bits": 24,
                    "init": CRC_INIT,
                    "reflected": True,
                    "checksum_le": True,
                },
            ),
        ]
    )


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

    stream = Bitstream(path=snk, num_bits=read_bits(snk).size)
    result = parse_bitstream(stream, _ble_codec(), registry())
    valid = [fr.payload_hex[4:] for fr in result.frames if fr.crc_ok]
    # Closure: BLE PDUs decode phy -> composed codec to CRC-valid, matching the
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
