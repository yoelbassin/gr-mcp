"""LoRa's coding tail (nibble_swap -> segment) over de-FEC'd header-decode
output, deterministic — a golden bit array bypasses css_explicit_decode itself
(covered live by tests/e2e/test_lora_offair.py) to pin the pure-coding half of
run_rx: a Bitstream in, one segmented window out, dewhitened and CRC-16-valid.

The golden bits are css_explicit_decode(_SF11_SYMBOLS)'s output, captured once
(see data/lora_golden_bits.npy's provenance below) — 2056 bits of 255-byte
whitened LoRa payload + 2-byte CRC field (unwhitened), plus 12 bits of inert
legacy FEC-pad past index 2055 that segment's frame_body_len=2056 never touches.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from helpers import bitops, crc, framing

from marconi.engine.coding.stages_bits import NibbleSwapStep, SegmentStep
from marconi.engine.io.bitfile import read_bits, write_bits
from marconi.engine.run import run_rx
from marconi.engine.stages.registry import stage_registry
from marconi.engine.types.descriptor import Descriptor
from marconi.engine.types.enums import ItemType
from marconi.engine.types.levels import Level
from marconi.engine.types.models import Bitstream, Modem

BITS = Descriptor(Level.BITS, ItemType.B)
_FRAME_BODY_LEN = 2056

# Captured by:
#   cd /Users/joel/Clones/gr-mcp-rebuild
#   .venv/bin/python -c "
#     import sys; sys.path.insert(0, 'tests')
#     from helpers._css_synth import run as _run, SF11_SYMBOLS as _SF11_SYMBOLS
#     bits = _run(_SF11_SYMBOLS); print(len(bits)); print(list(bits))"
# then saved as data/lora_golden_bits.npy (uint8, shape (2068,)) — a giant
# Python literal doesn't belong in a test file.
_GOLDEN = Path(__file__).resolve().parent / "data" / "lora_golden_bits.npy"

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


def _golden_modem() -> Modem:
    return Modem(
        symbol_rate=1.0,
        path=[
            NibbleSwapStep(),
            SegmentStep(frame_body_len=_FRAME_BODY_LEN),
        ],
    )


def _golden_stream(tmp_path: Path) -> Bitstream:
    bits = np.load(_GOLDEN).astype(np.uint8)
    p = tmp_path / "golden.u8"
    write_bits(p, bits)
    return Bitstream(path=p, num_bits=int(bits.size))


@pytest.mark.skipif(not _GOLDEN.exists(), reason="golden bits absent")
def test_lora_golden_bits_decode_crc_valid(tmp_path: Path) -> None:
    res = run_rx(
        _golden_modem(),
        stage_registry(),
        sample_rate=1.0,
        start=BITS,
        workdir=tmp_path,
        input_stream=_golden_stream(tmp_path),
    )
    assert res.status == "ok", res
    assert res.windows == [0], res.windows
    assert res.bitstream is not None
    bits = read_bits(res.bitstream.path)

    decoded: list[bytes] = []
    for frame in framing.carve_fixed(bits, res.windows, _FRAME_BODY_LEN):
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
    assert decoded, "no CRC-valid frame"
    assert decoded[0].startswith(b"RF fingerpring Project for Lora"), decoded[0][:32]
