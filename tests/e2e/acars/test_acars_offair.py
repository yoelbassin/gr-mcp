"""Protocol #5: real off-air ACARS decodes end-to-end through one Modem
spanning phy through the coding tail, matched against an external decoder's
golden (issues 22/23).

Every ARINC-618 constant below is test-side caller data, per CLAUDE.md. External
oracle: golden_acarsdec.json (acarsdec 3.7 over the same capture; provenance
inside). Chain: AM-carried coherent MSK at 2400 baud, 7-bit ASCII chars with odd
parity in the MSB, ETX-terminated block, trailing CRC-16 block-check (BCS).

Front-end + demod (the msk stage): the committed conditioning chain
(channelize/resample/am/analytic/channelize@1800) hands the coherent `msk` demod
an analytic baseband; `slice` hard-decides one bit per symbol. That rail decision
IS the on-wire bit directly (ACARS is transmitter-precoded -- no differential),
first-received bit first, i.e. LSB-first within each 8-bit char. The demod is
clean here (0 bit errors on the 3 golden bursts), so the strict CRC -- which has
no error correction, unlike acarsdec's parity/CRC syndrome FEC -- is the oracle.

Framing note: an ACARS block is ETX-terminated and byte-aligned, but a bare 8-bit
ETX pattern also occurs bit-shifted mid-block (a char's odd-parity bit followed by
the next char's data), so a bit-sliding delimiter search locks onto a false early
terminator. With no byte-aligned terminator slicer in the vocabulary, framing is
instead a bounded CRC-validated length search: carve each sync'd burst at every
plausible block length and let the CRC accept only the true byte-aligned length.
The CRC is the oracle -- exactly the project's "estimators propose, oracle
confirms" idiom -- and the search carries no protocol magic (just a byte-length
range).

Sync + parity stripping are now product coding stages appended to the modem
after slice: `descramble` ({"sequence": "ff"}, an all-1s XOR = the old
descramble_bits global inversion) undoes the Costas loop's per-burst 180 deg
phase ambiguity, and `sync_word` ({"sync": "686880", ...}) finds "SYN SYN SOH"
bit-reversed to the slicer's LSB-first wire order, seeding a window per hit --
the algorithm is unchanged from the old bits-layer sync_word_rx. A burst arrives
either true or globally inverted, so both polarities are searched by running the
modem twice per offset, with and without the descramble step (mirroring the old
codec's `for invert in (True, False)` loop, now at the modem level since
descramble/sync_word run inside the same pipeline as the GR demod).

The per-length loop (old: fixed_frame -> crc -> frame_codebook -> parse) is now
test-side: framing.carve_fixed slices each window at every candidate frame_bytes
(matching the old fixed_frame_rx, which sliced the same raw post-sync_word bits),
crc.crc_check strips the 2-byte BCS with the same KERMIT-family params. Parity
stripping is where a helper's contract needed reconciling: bitops.translate_bytes
is a byte-for-byte table lookup (output stays one byte per input byte), while the
old frame_codebook step both looked up AND repacked 8-bit wire symbols into a
dense 7-bit-per-char bitstream (the layout parse_message's BitStruct fields
require -- reg is 49 bits = 7 chars * 7 bits, with no padding between chars).
_strip_parity below reproduces that exactly: translate_bytes performs the same
inverse-codebook lookup the old frame_codebook step did (each wire byte -> its
7-bit ASCII index, 0-127, still one byte per char), then the pad bit each lookup
byte carries (always 0, since values are <128) is dropped and the remaining 7
bits per char are repacked dense -- bit-for-bit what frame_codebook_rx's
_pack_symbols(data, 7) produced.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from helpers import bitops, crc, framing, parse

from marconi.engine.backends.gnuradio.runner import ensure_worker_warm
from marconi.engine.coding.stages_bits import DescrambleStep, SyncWordStep
from marconi.engine.io.bitfile import read_bits
from marconi.engine.io.source import SourceSlice
from marconi.engine.modulation.fsk.stages import MskStep
from marconi.engine.run import run_rx
from marconi.engine.stages.conditioning import (
    AgcStep,
    AmStep,
    AnalyticStep,
    ChannelizeStep,
    ResampleStep,
)
from marconi.engine.stages.general import SliceStep
from marconi.engine.stages.registry import stage_registry
from marconi.engine.types.descriptor import Descriptor
from marconi.engine.types.enums import ItemType
from marconi.engine.types.levels import Level
from marconi.engine.types.models import Modem
from marconi.engine.types.step import Step

IQ = Descriptor(Level.IQ, ItemType.C)
RATE = 2_000_000.0
_ASSETS = Path(__file__).resolve().parents[3] / "artifacts" / "assets" / "ACRAS"
_CF32 = _ASSETS / "acars.cf32"
_GOLDEN = _ASSETS / "golden_acarsdec.json"

# A block after SOH: header (mode + 7-char addr + ack + 2-char label + block-id +
# STX = 13 chars) + text + ETX + 2-byte BCS. 16 is the shortest possible block;
# 248 comfortably covers the ARINC-618 maximum. Only true byte-aligned lengths
# survive the CRC, so the bound is loose on purpose.
_MIN_FRAME_BYTES = 16
_MAX_FRAME_BYTES = 248


def _bitrev8(b: int) -> int:
    return int(format(b, "08b")[::-1], 2)


# ASCII code -> the bit-reversed on-wire byte for that char (forward table, 128
# entries: 7-bit ASCII index -> 8-bit wire byte with odd parity in the MSB, bit-
# reversed to the slicer's LSB-first wire order). The inverse (wire byte -> ASCII
# index, 0 for a parity-invalid byte) is what strips parity below -- the same
# 256-entry table the old frame_codebook step built internally.
_PARITY_STRIP = [_bitrev8(c | ((1 ^ (bin(c).count("1") & 1)) << 7)) for c in range(128)]
_INV_PARITY = [0] * 256
for _i, _w in enumerate(_PARITY_STRIP):
    _INV_PARITY[_w] = _i

# Plain 7-bit ASCII: index i decodes to chr(i). Faithful decode; acarsdec's
# cosmetic renderings (dot-padded addr, DEL->'d' label, #CFB sublabel dropped from
# text) are reconciled in the golden match below, not baked into the alphabet.
_ASCII7 = "".join(chr(i) for i in range(128))
_FIELDS: list[dict[str, object]] = [
    {"name": "mode", "bits": 7, "charset": _ASCII7, "char_bits": 7},
    {"name": "reg", "bits": 49, "charset": _ASCII7, "char_bits": 7},
    {"name": "ack", "bits": 7, "charset": _ASCII7, "char_bits": 7},
    {"name": "label", "bits": 14, "charset": _ASCII7, "char_bits": 7},
    {"name": "block_id", "bits": 7, "charset": _ASCII7, "char_bits": 7},
    {"name": "stx", "bits": 7, "charset": _ASCII7, "char_bits": 7},
    {"name": "msgno", "bits": 28, "charset": _ASCII7, "char_bits": 7},
    {"name": "flight", "bits": 42, "charset": _ASCII7, "char_bits": 7},
    {"name": "text", "bits": 0, "charset": _ASCII7, "char_bits": 7, "rest": True},
]

# old crc CodecStep params, verbatim: KERMIT-family over the 8-bit chars, seen
# bit-reversed here -> poly 0x1021 non-reflected.
_ACARS_CRC: dict[str, int | str] = {
    "poly": 0x1021,
    "bits": 16,
    "init": 0,
    "reflected": False,
    "xorout": 0,
    "bit_order": "msb",
}


def _acars_modem(offset_hz: float, invert: bool) -> Modem:
    path: list[Step] = [
        ChannelizeStep(decim=40, bandwidth_hz=24000.0, center_hz=offset_hz),
        ResampleStep(interpolation=24, decimation=25),
        AmStep(),
        AnalyticStep(),
        ChannelizeStep(decim=1, bandwidth_hz=2400.0, center_hz=1800.0),
        AgcStep(window_symbols=64.0),
        MskStep(),
        SliceStep(),
    ]
    if invert:
        path.append(DescrambleStep(sequence="ff"))
    path.append(SyncWordStep(sync="686880", max_errors=0))
    return Modem(symbol_rate=2400.0, path=path)


def _strip_parity(body: bytes) -> bytes:
    idx = bitops.translate_bytes(body, _INV_PARITY)
    bits8 = bitops.bytes_to_bits(idx, "msb").reshape(-1, 8)
    dense = bits8[:, 1:].reshape(-1)
    return bitops.bits_to_bytes(dense, "msb")


def _norm_reg(reg: object) -> str:
    return str(reg).strip().lstrip(".")  # addr is dot-padded to 7 chars


def _norm_label(label: object) -> str:
    return str(label).replace("\x7f", "d")  # acarsdec renders label DEL as 'd'


@pytest.mark.skipif(
    not (_CF32.exists() and _GOLDEN.exists()),
    reason="ACARS assets absent -- run tests/e2e/acars/make_acars_slice.py "
    "+ make_acars_golden.py",
)
def test_acars_offair_matches_acarsdec_golden(tmp_path: Path) -> None:
    ensure_worker_warm()
    golden = json.loads(_GOLDEN.read_text())
    assert golden["messages"], "empty golden -- regenerate"
    decoded: dict[tuple[float, int], dict[str, int | str]] = {}
    crc_valid: set[tuple[float, int]] = set()
    for raw_offset in golden["channels_hz_offset"]:
        offset = float(raw_offset)
        for invert in (True, False):
            workdir = tmp_path / f"{int(offset)}_{invert}"
            workdir.mkdir()
            res = run_rx(
                _acars_modem(offset, invert),
                stage_registry(),
                sample_rate=RATE,
                start=IQ,
                workdir=workdir,
                source=SourceSlice(path=_CF32),
                timeout=300.0,
            )
            assert res.status == "ok", res
            if not res.windows:
                continue
            assert res.bitstream is not None
            bits = read_bits(res.bitstream.path)
            for frame_bytes in range(_MIN_FRAME_BYTES, _MAX_FRAME_BYTES + 1):
                n = frame_bytes * 8
                starts = [s for s in res.windows if s + n <= bits.size]
                raws = framing.carve_fixed(bits, res.windows, n)
                for start, raw in zip(starts, raws):
                    payload = bitops.bits_to_bytes(raw, "msb")
                    ok, body = crc.crc_check(
                        payload,
                        poly=int(_ACARS_CRC["poly"]),
                        bits=int(_ACARS_CRC["bits"]),
                        init=int(_ACARS_CRC["init"]),
                        reflected=bool(_ACARS_CRC["reflected"]),
                        xorout=int(_ACARS_CRC["xorout"]),
                        bit_order=str(_ACARS_CRC["bit_order"]),
                    )
                    if not ok:
                        continue
                    key = (offset, start)
                    crc_valid.add(key)
                    if key in decoded:
                        continue
                    message = parse.parse_message(_strip_parity(body), _FIELDS)
                    if message is not None:
                        decoded[key] = message

    # PIN (knob #6): total_crc_ok measured over 3 serial GR runs = 4 / 4 / 4 (the
    # 3 golden bursts + a 4th genuine burst, block-7, that also passes strict CRC).
    # floor is pinned at 3, not the observed 4: the 3 golden bursts are 0-bit-error
    # AND independently pinned by the need=3 content match below (the real
    # invariant), whereas the 4th burst is not golden-anchored -- staking the suite
    # on it over a 3-run determinism sample is the fragile choice. floor=3 keeps the
    # count assert meaningful with a margin against a marginal 4th-burst wobble.
    floor = 3
    assert len(crc_valid) >= max(1, floor), f"crc-valid={len(crc_valid)}"

    by_key = {
        (_norm_reg(m["reg"]), str(m["block_id"]).strip()): m for m in decoded.values()
    }
    checked = 0
    for g in golden["messages"]:
        m = by_key.get((_norm_reg(g["reg"]), g["block_id"].strip()))
        if m is None:
            continue
        assert _norm_label(m["label"]) == g["label"].strip(), (g, m)
        # golden `text` drops acarsdec's #CFB sublabel and the 10-char
        # MsgNo+FlightID prefix, so it is a substring of the field-parsed text.
        assert g["text"] in str(m["text"]), (g, m)
        checked += 1

    # All 3 golden bursts demod at 0 bit errors (Gate B), so all 3 strict-decode.
    need = min(3, len(golden["messages"]))
    assert checked >= need, f"only {checked} golden messages matched (need {need})"
