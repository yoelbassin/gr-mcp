"""Real off-air LoRa SF11 (IQ_2), the agent-loop shape end to end, CRC as the
oracle. The slice holds TWO complete frames; both must decode CRC-valid.

Run 1 is the product phy: resample/chirp_sync/dechirp/burst_probe in one
ModemSpec — chirp_sync re-arms per preamble and burst_probe surfaces both
burst marks in the same run (issue 03's multi-burst requirement), delivered
on the symbolstream. The header-driven carve is datasheet work and runs
test-side (helpers/css_explicit.py, demoted from a registered stage 2026-07):
it parses each marked burst's declared header and decodes the payload at its
declared code rate through core.coding's generic primitives. Run 2 feeds the
FEC-corrected bits back through the product's generic coding tail
(nibble_swap/segment) via input_stream. Dewhitening and the trailing CRC-16
are test-side helpers over run 2's per-frame windows, the same per-window
pattern as the POCSAG/BLE gates.

Capture: IQ_2 (LoPy4), 1 Msps, SF11/BW125/CR4-5/LDRO — cropped to a ~11.3 Msample
window holding two back-to-back frames (tests/e2e/make_lora_slice.py). Both
payloads read "RF fingerpring Project for Lora device..." — CRC-16/CCITT
(poly 0x1021, init 0) self-validates each 255-byte whitened payload.
"""

from __future__ import annotations

from pathlib import Path
from typing import cast

import pytest
from helpers import bitops, crc, framing
from helpers.css_explicit import css_explicit_decode

from marconi.engine.backends.gnuradio.runner import ensure_worker_warm
from marconi.engine.io.bitfile import read_bits, read_symbols, write_bits
from marconi.engine.run import run_rx
from marconi.engine.stages.registry import stage_registry
from marconi.engine.types.descriptor import Descriptor
from marconi.engine.types.levels import Level
from marconi.engine.types.models import Bitstream, ModemSpec, ModemStep
from marconi.engine.types.params import ParamValue

IQ = Descriptor(Level.IQ, "c")
BITS = Descriptor(Level.BITS, "b")
RATE = 1_000_000.0
_SLICE = (
    Path(__file__).resolve().parents[2]
    / "artifacts"
    / "assets"
    / "LoRa"
    / "iq2_frame.cf32"
)

# LoRa constants (caller data — a protocol lives in the fixture, not production).
# Hamming(4+cr, 4) parity masks per code rate 1..4, flattened low-cr-first
# (coding.parity_for_cr's expected layout).
_PARITY = {
    1: [15],
    2: [7, 14],
    3: [11, 13, 14],
    4: [7, 14, 11, 13],
}
_PARITY_MASKS = [m for cr in sorted(_PARITY) for m in _PARITY[cr]]

# css_explicit_decode geometry for LoRa's explicit header: 8 symbols, 5 header
# nibbles, reduced by 2, 12 data bits, checksum field masks; field (start,
# length) bit-positions sit inside the header nibbles.
_HEADER: dict[str, ParamValue] = {
    "sf": 11,
    "header_cr": 4,
    "reduced": True,
    "header_symbols": 8,
    "header_nibbles": 5,
    "sf_reduction": 2,
    "header_data_bits": 12,
    "header_parity": [3840, 2273, 1178, 599, 303],
    "field_payload_len": [0, 8],
    "field_cr": [8, 3],
    "field_has_crc": [11, 1],
    "field_parity": [15, 5],
    "data_bits": 4,
    "crc_bytes": 2,
    "parity_masks": cast("list[float | int]", _PARITY_MASKS),
    "reduced_offset": 0,
    "full_offset": 1,
}

# LoRa whitening sequence (255 bytes, LFSR period-255), plus two 0x00 bytes so
# the 2-byte CRC field appended after the 255-byte payload is not de-whitened
# (the LoRa TX writes the CRC into the encoded stream in unwhitened form).
_WHITEN = (
    "fffefcf8f0e1c2850b172f5ebc78f1e3c68d1a3468d0a04080010204081123478e1c3871e2c489"
    "12254b972e5cb870e0c08103060c193264c992244993264d9b376edcb972e4c890204182050a15"
    "2b56ad5bb66ddab56bd6ac59b265cb962c58b061c3870f1f3e7dfbf6eddbb76fdebd7af5ebd7ae"
    "5dba74e8d1a24488102143860d1b366cd8b163c78f1e3c79f3e7ce9c3973e6cc983162c58b162d"
    "5ab469d2a4489122458a142952a54a952a54a953a74e9d3b77eeddbb76ecd9b367cf9e3d7bf7ef"
    "dfbf7efdfaf4e9d3a64c993366cd9a356ad4a851a3468c183060c183070e1d3a75ead5aa55ab57"
    "af5fbe7cf9f2e5ca942850a142840913274f9f3f7f"
    "0000"
)

_FRAME_BODY_LEN = 2056  # 255 whitened payload bytes + 2 unwhitened CRC bytes


def _phy_modem() -> ModemSpec:
    return ModemSpec(
        name="lora_sf11_rx",
        symbol_rate=61.03515625,
        path=[
            ModemStep(conv="resample", params={"interpolation": 2, "decimation": 8}),
            ModemStep(
                conv="chirp_sync",
                params={
                    "sf": 11,
                    "oversample": 2,
                    "zero_pad": 10,
                    "preamble_len": 8,
                    "sfd_symbols": 2.25,
                    "sync_symbols": 2,
                },
            ),
            ModemStep(
                conv="dechirp",
                params={
                    "sf": 11,
                    "oversample": 2,
                    "zero_pad": 10,
                    "preamble_len": 8,
                    "sfd_symbols": 2.25,
                    "sync_symbols": 2,
                },
            ),
            ModemStep(conv="burst_probe", params={}),
        ],
    )


def _codec_modem() -> ModemSpec:
    return ModemSpec(
        name="lora_sf11_codec",
        symbol_rate=1.0,
        path=[
            ModemStep(conv="nibble_swap", params={}),
            ModemStep(conv="segment", params={"frame_body_len": _FRAME_BODY_LEN}),
        ],
    )


@pytest.mark.skipif(
    not _SLICE.exists(), reason="LoRa slice absent — run tests/e2e/make_lora_slice.py"
)
def test_lora_offair(tmp_path: Path) -> None:
    ensure_worker_warm()
    res = run_rx(
        _phy_modem(),
        stage_registry(),
        sample_rate=RATE,
        start=IQ,
        workdir=tmp_path,
        source_io={"path": str(_SLICE)},
    )
    assert res.status == "ok", res
    assert len(res.marks) >= 2, res.diagnostics
    assert res.symbolstream is not None and res.symbolstream.item_type == "s"
    symbols = read_symbols(res.symbolstream.path, "s")

    fec_bits = css_explicit_decode(symbols, res.symbolstream.marks, _HEADER)
    assert fec_bits.size, "no frame decoded from the marked bursts"
    fec_path = tmp_path / "fec.u8"
    write_bits(fec_path, fec_bits)

    res2 = run_rx(
        _codec_modem(),
        stage_registry(),
        sample_rate=1.0,
        start=BITS,
        workdir=tmp_path,
        input_stream=Bitstream(path=fec_path, num_bits=int(fec_bits.size)),
    )
    assert res2.status == "ok", res2
    assert res2.windows, "no LoRa frames segmented"
    assert res2.bitstream is not None
    bits = read_bits(res2.bitstream.path)

    decoded: list[bytes] = []
    for frame in framing.carve_fixed(bits, res2.windows, _FRAME_BODY_LEN):
        dewhitened = framing.xor_bits(frame, _WHITEN)
        ok, body = crc.crc_check(
            bitops.bits_to_bytes(dewhitened),
            poly=0x1021,
            bits=16,
            init=0,
            fold_tail=2,
            checksum_le=True,
        )
        if ok:
            decoded.append(body)
    assert len(decoded) >= 2, f"expected both frames CRC-valid, got {len(decoded)}"
    for payload in decoded[:2]:
        assert payload.startswith(b"RF fingerpring Project for Lora"), payload[:32]
