"""Real off-air FM broadcast -> RDS data bits: the coherent PSK lane's first
off-air gate, and the FM-subcarrier demod-then-demod composition in one
product path. Ten registered generic stages, zero product edits:

  fm_demod -> analytic -> channelize -> resample -> agc -> psk_demod ->
  psk_demap  (GR: discriminator, 57 kHz subcarrier extraction, RRC + Gardner +
  costas BPSK, hard bits)  then  realign -> codebook -> differential  (coding:
  biphase pair decode + differential decode).

The biphase pair phase is unknowable a priori (Gardner picks an arbitrary
symbol parity), so the GR half runs once to hard bits and the coding tail runs
per pair offset via input_stream; the checkword oracle picks the winner.

Oracle: RDS 26-bit blocks carry a 10-bit checkword (g(x) = x^10+x^8+x^7+x^5+
x^4+x^3+1 plus a per-block offset word) — group-synced blocks whose syndrome
matches count as valid, and groups of type 0A/0B carry the station's 8-char
Program Service name. The capture (PySDR's fm_rds_250k_1Msamples.iq, 4 s @
250 ksps) decodes 180 valid blocks with PS "Upliftin" — matched independently
by a numpy reference decode. Checkword math, offset words, and PS field
layout are protocol-datasheet work and live here, not in the product.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from marconi.core.bitfile import read_bits
from marconi.core.descriptor import Descriptor
from marconi.core.levels import Level
from marconi.core.models import Bitstream
from marconi.phy.backends.gnuradio.runner import ensure_worker_warm
from marconi.phy.engine import run_rx
from marconi.phy.models import ModemSpec, ModemStep
from marconi.phy.stages import stage_registry

IQ = Descriptor(Level.IQ, "c")
BITS = Descriptor(Level.BITS, "b")
RATE = 250_000.0
SYMBOL_RATE = 2375.0  # biphase symbols; data runs at half this
_CAPTURE = (
    Path(__file__).resolve().parents[2]
    / "artifacts"
    / "assets"
    / "RDS"
    / "fm_rds_250k_1Msamples.iq"
)

# RDS datasheet constants (caller data): checkword generator and the offset
# word each block position XORs onto its checkword.
G_POLY = 0b10110111001
OFFSETS = {0x0FC: "A", 0x198: "B", 0x168: "C", 0x350: "C'", 0x1B4: "D"}

MIN_VALID_BLOCKS = 120  # measured 180; margin for scheduler nondeterminism
STATION_PS = "Upliftin"


def _phy_modem() -> ModemSpec:
    return ModemSpec(
        name="rds_rx",
        symbol_rate=SYMBOL_RATE,
        path=[
            ModemStep(conv="fm_demod", params={"deviation": 75_000.0}),
            ModemStep(conv="analytic", params={}),
            ModemStep(
                conv="channelize",
                params={"decim": 5, "bandwidth_hz": 4800.0, "center_hz": 57_000.0},
            ),
            ModemStep(conv="resample", params={"interpolation": 19, "decimation": 50}),
            ModemStep(conv="agc", params={"mode": "power"}),
            ModemStep(conv="psk_demod", params={"order": 2, "alpha": 1.0}),
            ModemStep(conv="psk_demap", params={"order": 2}),
        ],
    )


def _codec_modem(bit_offset: int) -> ModemSpec:
    return ModemSpec(
        name="rds_codec",
        symbol_rate=1.0,
        path=[
            ModemStep(conv="realign", params={"bit_offset": bit_offset}),
            ModemStep(
                conv="codebook",
                params={"code_bits": 2, "data_bits": 1, "table": [1, 2]},
            ),
            ModemStep(conv="differential", params={}),
        ],
    )


def _crc10(info16: int) -> int:
    v = info16 << 10
    for i in range(15, -1, -1):
        if v & (1 << (i + 10)):
            v ^= G_POLY << i
    return v & 0x3FF


def _decode_groups(bits: np.ndarray) -> tuple[int, str]:
    """Group-sync the bit stream on checkword-valid A/B/C/D block sequences;
    return (valid block count, PS name assembled from 0A/0B groups)."""
    if bits.size < 104:
        return 0, ""
    vals = np.zeros(bits.size - 25, np.int64)
    for i in range(26):
        vals = (vals << 1) | bits[i : i + vals.size]

    def block_type(v: int) -> str | None:
        return OFFSETS.get((v & 0x3FF) ^ _crc10(v >> 10))

    types = [block_type(int(v)) for v in vals]
    n_valid, ps, i = 0, ["_"] * 8, 0
    while i < len(types) - 78:
        if (
            types[i] == "A"
            and types[i + 26] == "B"
            and types[i + 52] in ("C", "C'")
            and types[i + 78] == "D"
        ):
            b, d = int(vals[i + 26]) >> 10, int(vals[i + 78]) >> 10
            n_valid += 4
            if (b >> 12) == 0:
                seg = b & 0x3
                hi, lo = d >> 8, d & 0xFF
                ps[2 * seg] = chr(hi) if 32 <= hi < 127 else "?"
                ps[2 * seg + 1] = chr(lo) if 32 <= lo < 127 else "?"
            i += 104
        else:
            i += 1
    return n_valid, "".join(ps)


@pytest.mark.skipif(
    not _CAPTURE.exists(), reason="RDS capture absent — run tests/e2e/make_rds_asset.py"
)
def test_rds_offair(tmp_path: Path) -> None:
    ensure_worker_warm()
    res = run_rx(
        _phy_modem(),
        stage_registry(),
        sample_rate=RATE,
        start=IQ,
        workdir=tmp_path,
        source_io={"path": str(_CAPTURE)},
    )
    assert res.status == "ok", res
    assert res.bitstream is not None
    n_symbols = res.bitstream.num_bits
    assert n_symbols > 4000, f"only {n_symbols} biphase symbols demodulated"

    results = []
    for off in (0, 1):
        res2 = run_rx(
            _codec_modem(off),
            stage_registry(),
            sample_rate=1.0,
            start=BITS,
            workdir=tmp_path,
            input_stream=Bitstream(path=res.bitstream.path, num_bits=n_symbols),
        )
        assert res2.status == "ok", res2
        assert res2.bitstream is not None
        results.append(_decode_groups(read_bits(res2.bitstream.path)))

    n_valid, ps = max(results)
    assert n_valid >= MIN_VALID_BLOCKS, f"{n_valid} checkword-valid blocks: {results}"
    assert ps == STATION_PS, f"PS mismatch: {results}"
