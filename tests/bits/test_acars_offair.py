"""Protocol #5: real off-air ACARS decodes end-to-end through the generic
vocabulary, matched against an external decoder's golden (issues 22/23).

Every ARINC-618 constant below is test-side caller data, per CLAUDE.md. External
oracle: golden_acarsdec.json (acarsdec 3.7 over the same capture; provenance
inside). Chain: AM-carried coherent MSK at 2400 baud, 7-bit ASCII chars with odd
parity in the MSB, ETX-terminated block, trailing CRC-16 block-check (BCS).

Front-end + demod (the Task 3-5 msk stage): the committed conditioning chain
(channelize/resample/am/analytic/channelize@1800) hands the coherent `msk` demod
an analytic baseband; `slice` hard-decides one bit per symbol. That rail decision
IS the on-wire bit directly (ACARS is transmitter-precoded -- no differential),
first-received bit first, i.e. LSB-first within each 8-bit char. The demod is
clean here (0 bit errors on the 3 golden bursts), so the strict CRC -- which has
no error correction, unlike acarsdec's parity/CRC syndrome FEC -- is the oracle.

Framing note: an ACARS block is ETX-terminated and byte-aligned, but a bare 8-bit
ETX pattern also occurs bit-shifted mid-block (a char's odd-parity bit followed by
the next char's data), so `delimiter_frame`, which slides bit-by-bit, locks onto a
false early terminator (measured: first hit at bit 55 vs the real byte-aligned ETX
at 264 for the /CG frame). With no byte-aligned terminator slicer in the
vocabulary, framing is instead a bounded CRC-validated length search: carve each
sync'd burst at every plausible block length with `fixed_frame` and let the CRC
accept only the true byte-aligned length. The CRC is the oracle -- exactly the
project's "estimators propose, oracle confirms" idiom -- and the search carries no
protocol magic (just a byte-length range).
"""

from __future__ import annotations

import json
from pathlib import Path

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
RATE = 2_000_000.0
_ASSETS = Path(__file__).resolve().parents[2] / "artifacts" / "assets" / "ACRAS"
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


# ASCII code -> the bit-reversed on-wire byte for that char. On the wire a char is
# its 7-bit ASCII value in the low 7 bits plus an odd-parity bit in the MSB, sent
# LSB-first; `delimiter_frame`/`crc` pack the LSB-first stream MSB-first, so the
# byte the codec sees is that on-wire byte bit-reversed. `frame_codebook` inverts
# this table to recover the 7-bit ASCII value (dropping the parity bit); a
# parity-invalid byte maps to 0 and its frame fails CRC.
_PARITY_STRIP = [_bitrev8(c | ((1 ^ (bin(c).count("1") & 1)) << 7)) for c in range(128)]
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


def _acars_modem(offset_hz: float) -> ModemSpec:
    return ModemSpec(
        symbol_rate=2400.0,
        path=[
            ModemStep(
                conv="channelize",
                params={"decim": 40, "bandwidth_hz": 24000.0, "center_hz": offset_hz},
            ),
            ModemStep(conv="resample", params={"interpolation": 24, "decimation": 25}),
            ModemStep(conv="am", params={}),
            ModemStep(conv="analytic", params={}),
            ModemStep(
                conv="channelize",
                params={"decim": 1, "bandwidth_hz": 2400.0, "center_hz": 1800.0},
            ),
            ModemStep(conv="msk", params={}),
            ModemStep(conv="slice", params={}),
        ],
    )


def _acars_codec(invert: bool, frame_bytes: int) -> CodecSpec:
    # sync "686880" = SYN SYN SOH (0x16 0x16 0x01) bit-reversed to the LSB-first
    # wire order the slicer emits. The Costas loop resolves MSK's 180 deg phase
    # ambiguity per burst, so a burst arrives either true (invert=False) or
    # globally inverted (invert=True, undone by an all-ones descramble = bitwise
    # NOT); both polarities are searched so the decode is robust to run-to-run
    # phase acquisition. CRC (KERMIT-family over the 8-bit chars, seen bit-reversed
    # here -> poly 0x1021 non-reflected) strips the 2-byte BCS; frame_codebook then
    # drops parity to 7-bit ASCII; parse reads the ARINC-618 downlink fields.
    path: list[CodecStep] = []
    if invert:
        path.append(CodecStep(conv="descramble_bits", params={"sequence": "ff"}))
    path += [
        CodecStep(conv="sync_word", params={"sync": "686880", "max_errors": 0}),
        CodecStep(conv="fixed_frame", params={"payload_bits": frame_bytes * 8}),
        CodecStep(
            conv="crc",
            params={
                "poly": 0x1021,
                "bits": 16,
                "init": 0,
                "reflected": False,
                "xorout": 0,
                "bit_order": "msb",
            },
        ),
        CodecStep(
            conv="frame_codebook",
            params={"code_bits": 8, "data_bits": 7, "table": _PARITY_STRIP},
        ),
        CodecStep(conv="parse", params={"fields": _FIELDS}),
    ]
    return CodecSpec(name="protocol5", path=path)


def _norm_reg(reg: object) -> str:
    return str(reg).strip().lstrip(".")  # addr is dot-padded to 7 chars


def _norm_label(label: object) -> str:
    return str(label).replace("\x7f", "d")  # acarsdec renders label DEL as 'd'


@pytest.mark.skipif(
    not (_CF32.exists() and _GOLDEN.exists()),
    reason="ACARS assets absent -- run tests/bits/make_acars_slice.py "
    "+ make_acars_golden.py",
)
def test_acars_offair_matches_acarsdec_golden(tmp_path: Path) -> None:
    ensure_worker_warm()
    golden = json.loads(_GOLDEN.read_text())
    assert golden["messages"], "empty golden -- regenerate"
    reg = registry()
    decoded: dict[tuple[float, int], dict[str, int | str]] = {}
    crc_valid: set[tuple[float, int]] = set()
    for raw_offset in golden["channels_hz_offset"]:
        offset = float(raw_offset)
        snk = tmp_path / f"bits_{int(offset)}.u8"
        pipe = compile_modem(
            _acars_modem(offset),
            stage_registry(),
            direction="rx",
            sample_rate=RATE,
            start=IQ,
            source_io={"path": str(_CF32)},
            sink_io={"path": str(snk)},
        )
        r = GnuRadioBackend().run_pipeline(pipe, timeout=300.0)
        assert r.status == "ok", r
        bstream = Bitstream(
            path=snk, num_bits=int(read_bits(snk).size), source_capture=_CF32
        )
        for invert in (True, False):
            for frame_bytes in range(_MIN_FRAME_BYTES, _MAX_FRAME_BYTES + 1):
                res = parse_bitstream(bstream, _acars_codec(invert, frame_bytes), reg)
                for f in res.frames:
                    if f.crc_ok:
                        key = (offset, f.bit_offset)
                        crc_valid.add(key)
                        if f.message is not None and key not in decoded:
                            decoded[key] = f.message

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
