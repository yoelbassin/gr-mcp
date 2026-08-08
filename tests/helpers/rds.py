from __future__ import annotations

import numpy as np

from marconi.engine.coding.stages_bits import (
    CodebookStep,
    DifferentialStep,
    RealignStep,
)
from marconi.engine.modulation.psk.stages import PskDemapStep, PskDemodStep
from marconi.engine.stages.conditioning import (
    AgcStep,
    AnalyticStep,
    ChannelizeStep,
    FmDemodStep,
    ResampleStep,
)
from marconi.engine.types.enums import AgcMode, PskOrder
from marconi.engine.types.models import Modem

SYMBOL_RATE = 2375.0  # biphase symbols; data runs at half this

# RDS datasheet constants (caller data): checkword generator and the offset
# word each block position XORs onto its checkword.
G_POLY = 0b10110111001
OFFSETS = {0x0FC: "A", 0x198: "B", 0x168: "C", 0x350: "C'", 0x1B4: "D"}


def phy_modem() -> Modem:
    return Modem(
        name="rds_rx",
        symbol_rate=SYMBOL_RATE,
        path=[
            FmDemodStep(deviation=75_000.0),
            AnalyticStep(),
            ChannelizeStep(decim=5, bandwidth_hz=4800.0, center_hz=57_000.0),
            ResampleStep(interpolation=19, decimation=50),
            AgcStep(mode=AgcMode.POWER),
            PskDemodStep(order=PskOrder.BPSK, alpha=1.0),
            PskDemapStep(order=PskOrder.BPSK),
        ],
    )


def codec_modem(bit_offset: int) -> Modem:
    return Modem(
        name="rds_codec",
        symbol_rate=1.0,
        path=[
            RealignStep(bit_offset=bit_offset),
            CodebookStep(code_bits=2, data_bits=1, table=[1, 2]),
            DifferentialStep(),
        ],
    )


def crc10(info16: int) -> int:
    v = info16 << 10
    for i in range(15, -1, -1):
        if v & (1 << (i + 10)):
            v ^= G_POLY << i
    return v & 0x3FF


def decode_groups(bits: np.ndarray) -> tuple[int, str]:
    """Group-sync the bit stream on checkword-valid A/B/C/D block sequences;
    return (valid block count, PS name assembled from 0A/0B groups)."""
    if bits.size < 104:
        return 0, ""
    vals = np.zeros(bits.size - 25, np.int64)
    for i in range(26):
        vals = (vals << 1) | bits[i : i + vals.size]

    def block_type(v: int) -> str | None:
        return OFFSETS.get((v & 0x3FF) ^ crc10(v >> 10))

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
