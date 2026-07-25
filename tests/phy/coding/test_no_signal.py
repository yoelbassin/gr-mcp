"""No signal present is the NORMAL case for a survey tool.

The correct answer to silence, to a stuck rail, and to noise is zero
decodes -- never an exception, and never fabricated bits. A tail whose seeder
found nothing reports status "empty" with no stream; a tail that passes bits
through reports "ok" and the decoder finds nothing valid in them. Every shipped
composition's coding tail is run against all three inputs: a decode that
crashes when the band is empty cannot be pointed at a band.

Regression this guards: POCSAG's codec raised IndexError on all three inputs.
permute's unseeded branch assumed perm was a permutation of range(len(perm)),
but POCSAG's is a DROPPING gather (496 indices spanning 511 bits), and that
branch is exactly where execution lands when sync_word seeds no windows.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import numpy as np
import pytest
from e2e import _dmr
from e2e.test_ais_offair import _AIS_CRC, _ais_modem
from e2e.test_ble_offair import (
    BIT_REVERSE,
)
from e2e.test_ble_offair import CRC_INIT as _BLE_CRC_INIT
from e2e.test_ble_offair import CRC_POLY as _BLE_CRC_POLY
from e2e.test_ble_offair import (
    MAX_PDU_BYTES,
)
from e2e.test_ble_offair import WHITEN as _BLE_WHITEN
from e2e.test_ble_offair import (
    _ble_modem,
)
from e2e.test_dab_codec import _FRAME_BODY_LEN as _DAB_FRAME_LEN
from e2e.test_dab_codec import _golden_modem as _dab_golden_modem
from e2e.test_dmr_offair import _dmr_modem
from e2e.test_pocsag_offair import (
    BATCH_CODEWORDS,
)
from e2e.test_pocsag_offair import DATA_BITS as _POCSAG_DATA_BITS
from e2e.test_pocsag_offair import (
    IDLE_DATA,
    _pocsag_modem,
)
from e2e.test_pocsag_offair import _word as _pocsag_word
from e2e.test_rds_offair import _codec_modem as _rds_codec_modem
from e2e.test_rds_offair import _decode_groups as _rds_decode_groups
from helpers import bitops, crc, framing
from phy.coding.test_lora_golden import _FRAME_BODY_LEN as _LORA_FRAME_LEN
from phy.coding.test_lora_golden import _WHITEN as _LORA_WHITEN
from phy.coding.test_lora_golden import _golden_modem as _lora_golden_modem
from pydantic import ValidationError

from marconi.core.bitfile import read_bits
from marconi.core.descriptor import Carrier, Descriptor
from marconi.core.levels import Level
from marconi.core.models import Bitstream, Symbolstream
from marconi.phy.coding.carrier import CodingCarrier, Window
from marconi.phy.coding.ops_bits import permute_rx
from marconi.phy.coding.stages_bits import Permute
from marconi.phy.engine import run_rx
from marconi.phy.models import ModemSpec
from marconi.phy.stages import stage_registry

BITS = Descriptor(Level.BITS, "b")
SOFT_SYMBOLS = Descriptor(Level.SYMBOLS, "f", carrier=Carrier.SOFT)
_N = 60_000

_Decoder = Callable[[np.ndarray, list[int]], int]


def _coding_tail(modem: ModemSpec) -> ModemSpec:
    """The post-PHY portion of a shipped modem: every step whose stage is
    engine=='coding', in order. This is what "no signal" exercises -- the
    same coding-tail composition the real off-air gate runs, fed synthetic
    bits/symbols instead of a GR front end."""
    reg = stage_registry()
    return ModemSpec(
        symbol_rate=modem.symbol_rate,
        path=[s for s in modem.path if reg[s.conv].engine == "coding"],
    )


def _decode_ais(bits: np.ndarray, _windows: list[int]) -> int:
    n = 0
    for _start, payload in framing.hdlc_frames(bits, bit_order="lsb"):
        ok, _body = crc.crc_check(
            payload,
            poly=int(_AIS_CRC["poly"]),
            bits=int(_AIS_CRC["bits"]),
            init=int(_AIS_CRC["init"]),
            reflected=bool(_AIS_CRC["reflected"]),
            xorout=int(_AIS_CRC["xorout"]),
            bit_order=str(_AIS_CRC["bit_order"]),
        )
        n += int(ok)
    return n


def _decode_ble(bits: np.ndarray, windows: list[int]) -> int:
    n = 0
    for raw in framing.carve_fixed(bits, windows, MAX_PDU_BYTES * 8):
        pdu = framing.xor_payload(
            bitops.translate_bytes(bitops.bits_to_bytes(raw), BIT_REVERSE),
            _BLE_WHITEN.hex(),
        )
        body = framing.carve_length(
            pdu, length_bits=8, offset_bits=8, base_bytes=2 + 3, unit_bytes=1
        )
        if body is None:
            continue
        ok, _payload = crc.crc_check(
            body,
            poly=_BLE_CRC_POLY,
            bits=24,
            init=_BLE_CRC_INIT,
            reflected=True,
            checksum_le=True,
        )
        n += int(ok)
    return n


def _decode_pocsag(bits: np.ndarray, windows: list[int]) -> int:
    n = 0
    batch_len = BATCH_CODEWORDS * _POCSAG_DATA_BITS
    for batch in framing.carve_fixed(bits, windows, batch_len):
        for d in batch.reshape(-1, _POCSAG_DATA_BITS):
            if int(d[0]) != 0:  # not an address codeword (flag bit 0)
                continue
            if _pocsag_word(d, 0, _POCSAG_DATA_BITS) == IDLE_DATA:
                continue
            n += 1
    return n


def _decode_dmr(bits: np.ndarray, windows: list[int]) -> int:
    n = 0
    for payload in framing.carve_fixed(bits, windows, 96):
        if _dmr.parse_payload(payload):
            n += 1
    return n


def _decode_lora(bits: np.ndarray, windows: list[int]) -> int:
    n = 0
    for frame in framing.carve_fixed(bits, windows, _LORA_FRAME_LEN):
        dewhitened = framing.xor_bits(frame, _LORA_WHITEN)
        ok, _body = crc.crc_check(
            bitops.bits_to_bytes(dewhitened),
            poly=0x1021,
            bits=16,
            init=0,
            fold_tail=2,
            checksum_le=True,
        )
        n += int(ok)
    return n


def _decode_dab(bits: np.ndarray, windows: list[int]) -> int:
    n = 0
    for window in framing.carve_fixed(bits, windows, _DAB_FRAME_LEN):
        ok, _body = crc.crc_check(
            bitops.bits_to_bytes(window),
            poly=0x1021,
            bits=16,
            init=0xFFFF,
            xorout=0xFFFF,
        )
        n += int(ok)
    return n


def _decode_rds(bits: np.ndarray, _windows: list[int]) -> int:
    return _rds_decode_groups(bits)[0]


def _compositions() -> dict[str, tuple[Callable[[], ModemSpec], Descriptor, _Decoder]]:
    return {
        "ais": (lambda: _coding_tail(_ais_modem(0.0)), BITS, _decode_ais),
        "ble": (lambda: _coding_tail(_ble_modem()), BITS, _decode_ble),
        "dab": (_dab_golden_modem, BITS, _decode_dab),
        "dmr": (lambda: _coding_tail(_dmr_modem()), SOFT_SYMBOLS, _decode_dmr),
        "lora": (_lora_golden_modem, BITS, _decode_lora),
        "pocsag": (lambda: _coding_tail(_pocsag_modem()), BITS, _decode_pocsag),
        "rds": (lambda: _rds_codec_modem(0), BITS, _decode_rds),
    }


def _bits_stream(kind: str, rng: np.random.Generator, tmp_path: Path) -> Bitstream:
    data = {
        "zeros": np.zeros(_N, np.uint8),
        "ones": np.ones(_N, np.uint8),
        "noise": rng.integers(0, 2, _N).astype(np.uint8),
    }[kind]
    p = tmp_path / f"{kind}.u8"
    data.tofile(p)
    return Bitstream(path=p, num_bits=int(data.size))


def _symbol_stream(kind: str, rng: np.random.Generator, tmp_path: Path) -> Symbolstream:
    data = {
        "zeros": np.zeros(_N, np.float32),
        "ones": np.ones(_N, np.float32),
        "noise": rng.normal(0, 1, _N).astype(np.float32),
    }[kind]
    p = tmp_path / f"{kind}.f32"
    data.tofile(p)
    return Symbolstream(path=p, num_symbols=int(data.size), item_type="f")


def _input_stream(
    start: Descriptor, kind: str, rng: np.random.Generator, tmp_path: Path
) -> Bitstream | Symbolstream:
    if start.level is Level.BITS:
        return _bits_stream(kind, rng, tmp_path)
    return _symbol_stream(kind, rng, tmp_path)


@pytest.mark.parametrize("name", sorted(_compositions()))
@pytest.mark.parametrize("kind", ["zeros", "ones", "noise"])
def test_shipped_composition_survives_no_signal(
    name: str, kind: str, tmp_path: Path
) -> None:
    build_modem, start, decode = _compositions()[name]
    rng = np.random.default_rng(0)
    stream = _input_stream(start, kind, rng, tmp_path)
    res = run_rx(
        build_modem(),
        stage_registry(),
        sample_rate=1.0,
        start=start,
        workdir=tmp_path,
        input_stream=stream,
    )
    assert res.status in ("ok", "empty"), res
    empty = np.zeros(0, np.uint8)
    bits = read_bits(res.bitstream.path) if res.bitstream is not None else empty
    if res.status == "empty":
        assert res.bitstream is None and res.symbolstream is None
    n = decode(bits, res.windows)
    assert n == 0, f"{name} found a valid frame in {kind}"


def test_dropping_gather_works_in_the_unseeded_scope() -> None:
    """The POCSAG shape: read 3 of every 4 bits, i.e. span 4 but length 3."""
    perm = [0, 1, 2, 4, 5, 6]
    bits = np.arange(16, dtype=np.uint8) % 2
    out = permute_rx(CodingCarrier(bits=bits), perm=perm)
    span = max(perm) + 1
    blocks = bits.size // span
    want = np.concatenate([bits[b * span + np.asarray(perm)] for b in range(blocks)])
    assert np.array_equal(out.bits, want)
    assert out.bits.size == blocks * len(perm)


def test_unseeded_and_seeded_scopes_agree_on_stride() -> None:
    """A gather must consume the same input span whichever scope it runs in."""
    perm = [0, 2, 5]
    bits = np.random.default_rng(1).integers(0, 2, 60).astype(np.uint8)
    unseeded = permute_rx(CodingCarrier(bits=bits), perm=perm)
    span = max(perm) + 1
    seeded = permute_rx(
        CodingCarrier(
            bits=bits,
            windows=[Window(start=b * span, cursor=b * span) for b in range(10)],
        ),
        perm=perm,
    )
    assert np.array_equal(unseeded.bits, seeded.bits)


def test_negative_perm_index_is_rejected() -> None:
    with pytest.raises(ValidationError):
        Permute._Params.model_validate({"perm": [0, -1, 2]})
    with pytest.raises(ValidationError):
        Permute._Params.model_validate({"perm": []})
