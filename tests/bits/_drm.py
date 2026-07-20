"""DRM (Mode B, spectrum occupancy 3) caller data for the FAC/SDC closure gate:
OFDM geometry, the scattered/frequency pilot generators, the FAC/SDC cell and
bit-interleave tables, the shared K=7 rate-1/4 mother code + puncture
patterns, energy-dispersal PRBS, the two CRCs, and the FAC/SDC field parsers.
Every value is ported verbatim from the Dream-verified scratch scripts
(save_fac_cells.py / fac_fec.py / sdc_decode.py); none of it lives in src/."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np
from pydantic import StrictInt

from marconi.bits.models import CodecSpec, CodecStep
from marconi.core.descriptor import Carrier, Descriptor
from marconi.core.levels import Level
from marconi.core.params import StageParams
from marconi.core.stages import RxStage, Stage
from marconi.phy.compile_context import CompileContext
from marconi.phy.models import ModemStep
from marconi.phy.stages.registry import stage_registry

FFT_LEN = 256
CP_LEN = 64
SYM_LEN = 320
NS = 15
KMIN = -103
KMAX = 103
N_CARRIERS = 206
RATE = 12000.0

_SCAT_W = [[512, 0, 512, 0, 512], [0, 512, 0, 512, 0], [512, 0, 512, 0, 512]]
_SCAT_Z = [[0, 57, 164, 64, 12], [168, 255, 161, 106, 118], [25, 232, 132, 233, 38]]
_SCAT_Q = 12
_BOOST_CARRIERS = {-103, -101, 101, 103}


def scat_ref(fs: int, k: int) -> complex:
    inn, im = fs % 3, fs // 3
    ip = (k - 1 - 2 * inn) // 6
    phase = 4 * _SCAT_Z[inn][im] + ip * _SCAT_W[inn][im] + ip * ip * (1 + fs) * _SCAT_Q
    amplitude = 2.0 if k in _BOOST_CARRIERS else np.sqrt(2.0)
    return complex(amplitude * np.exp(1j * 2 * np.pi * (phase % 1024) / 1024))


def _is_scattered_pilot(fs: int, k: int) -> bool:
    return k != 0 and (k - 1 - 2 * (fs % 3)) % 6 == 0


def scat_carriers(fs: int) -> list[int]:
    return [k for k in range(KMIN, KMAX + 1) if _is_scattered_pilot(fs, k)]


def pilot_tables() -> tuple[list[list[int]], list[list[complex]]]:
    carriers = [scat_carriers(fs) for fs in range(NS)]
    values = [[scat_ref(fs, k) for k in ks] for fs, ks in enumerate(carriers)]
    return carriers, values


FP_CARRIERS = [16, 48, 64]
FP_PHASES = [331, 651, 555]
FP_VALUES = [
    complex(np.sqrt(2.0) * np.exp(1j * 2 * np.pi * p / 1024)) for p in FP_PHASES
]

_DC_SEARCH = 4
_WARMUP_SYMS = 300


def sync_params() -> dict[str, Any]:
    carriers, values = pilot_tables()
    return {
        "fft_len": FFT_LEN,
        "cp_len": CP_LEN,
        "sym_len": SYM_LEN,
        "n_frame_syms": NS,
        "n_carriers": N_CARRIERS,
        "kmin": KMIN,
        "dc_search": _DC_SEARCH,
        "warmup_syms": _WARMUP_SYMS,
        "pilot_lens": [len(ks) for ks in carriers],
        "pilot_carriers": [int(k) for ks in carriers for k in ks],
        "pilot_i": [float(v.real) for vs in values for v in vs],
        "pilot_q": [float(v.imag) for vs in values for v in vs],
        "fp_carriers": [int(k) for k in FP_CARRIERS],
        "fp_i": [float(v.real) for v in FP_VALUES],
        "fp_q": [float(v.imag) for v in FP_VALUES],
    }


# fmt: off
# iTableFACRobModB: 65 fixed (frame_sym, carrier) positions, symbols 2..13.
FAC_CELLS: list[tuple[int, int]] = [
    (2, 13), (2, 25), (2, 43), (2, 55), (2, 67),
    (3, 15), (3, 27), (3, 45), (3, 57), (3, 69),
    (4, 17), (4, 29), (4, 47), (4, 59), (4, 71),
    (5, 19), (5, 31), (5, 49), (5, 61), (5, 73),
    (6, 9), (6, 21), (6, 33), (6, 51), (6, 63), (6, 75),
    (7, 11), (7, 23), (7, 35), (7, 53), (7, 65), (7, 77),
    (8, 13), (8, 25), (8, 37), (8, 55), (8, 67), (8, 79),
    (9, 15), (9, 27), (9, 39), (9, 57), (9, 69), (9, 81),
    (10, 17), (10, 29), (10, 41), (10, 59), (10, 71), (10, 83),
    (11, 19), (11, 31), (11, 43), (11, 61), (11, 73),
    (12, 21), (12, 33), (12, 45), (12, 63), (12, 75),
    (13, 23), (13, 35), (13, 47), (13, 65), (13, 77),
]
TIME_PILOT_CARRIERS = {
    14, 16, 18, 20, 24, 26, 32, 36, 42, 44, 48, 49, 50, 54, 56, 62, 64, 66, 68,
}
# fmt: on
assert len(FAC_CELLS) == 65


def _sdc_carriers(fs: int) -> list[int]:
    fp_carriers = set(FP_CARRIERS)
    out = []
    for k in range(KMIN, KMAX + 1):
        if k == 0 or _is_scattered_pilot(fs, k) or k in fp_carriers:
            continue
        if fs == 0 and k in TIME_PILOT_CARRIERS:
            continue
        out.append(k)
    return out


SDC_CELLS: list[tuple[int, int]] = [(0, k) for k in _sdc_carriers(0)] + [
    (1, k) for k in _sdc_carriers(1)
]
assert len(SDC_CELLS) == 322

ACTIVE_CARRIERS = [k for k in range(KMIN, KMAX + 1) if k != 0]
_CARRIER_INDEX = {k: i for i, k in enumerate(ACTIVE_CARRIERS)}


def gather_fac_cells(carr: np.ndarray) -> np.ndarray:
    """Select the 65 FAC cells per frame from the frame-aligned equalized
    carrier grid [n_symbols, N_CARRIERS] the coherent sync emits (symbol-major,
    ascending active-carrier index; output symbol 0 is frame-symbol 0)."""
    n_frames = carr.shape[0] // NS
    out = np.empty((n_frames, len(FAC_CELLS)), dtype=complex)
    for f in range(n_frames):
        for i, (s, c) in enumerate(FAC_CELLS):
            out[f, i] = carr[NS * f + s, _CARRIER_INDEX[c]]
    return out


# blockinterleaver_cc requires len(perm) == the repeating block it gathers
# over, so a cell-select perm spans a WHOLE frame/super-frame carrier block
# (symbol-major, then ascending active-carrier index) with the wanted cells
# gathered to the front in FAC_CELLS/SDC_CELLS order; a trailing keep_m_in_n
# stage then keeps just that front (plan: blockinterleaver_cc(fac_select_perm)
# + keep_m_in_n_c(65, FAC_SELECT_BLOCK)).
FAC_SELECT_BLOCK = NS * N_CARRIERS
SDC_SELECT_BLOCK = 3 * NS * N_CARRIERS


def _select_perm(cells: list[tuple[int, int]], block_size: int) -> list[int]:
    selected = [fs * N_CARRIERS + _CARRIER_INDEX[k] for fs, k in cells]
    rest = [i for i in range(block_size) if i not in set(selected)]
    return selected + rest


def fac_select_perm() -> list[int]:
    return _select_perm(FAC_CELLS, FAC_SELECT_BLOCK)


def sdc_select_perm() -> list[int]:
    return _select_perm(SDC_CELLS, SDC_SELECT_BLOCK)


# Standard (non-Dream-reversed) octal generators for the shared DAB/DRM K=7
# rate-1/4 mother code; bit-reversing each 7-bit value reproduces Dream's own
# ConvEncoder masks (0o155, 0o117, 0o123, 0o155) exactly. This is the exact
# polys/rate_inv already proven against real off-air DAB via the generic
# trellis "fec" stage (tests/phy/test_dab_fec_offair.py).
CONV_POLYS = [0o133, 0o171, 0o145, 0o133]
CONV_RATE_INV = 4
FAC_FRAME_BITS = 72
SDC_FRAME_BITS = 316
TAIL = 6

FAC_CODED_BITS = len(FAC_CELLS) * 2
SDC_CODED_BITS = len(SDC_CELLS) * 2
FAC_STEPS = FAC_FRAME_BITS + TAIL
SDC_STEPS = SDC_FRAME_BITS + TAIL

# Puncture pattern index 6 (FAC, rate 3/5) and index 4 (SDC, rate 1/2), as
# which mother rate-1/4 outputs {0,1,2,3} survive each trellis step.
FAC_PUNCTURE_PATTERN: tuple[tuple[int, ...], ...] = ((0, 1), (0,), (0, 1))
SDC_PUNCTURE_PATTERN: tuple[tuple[int, ...], ...] = ((0, 1),)


def _expand_puncture(pattern: tuple[tuple[int, ...], ...], n_steps: int) -> list[int]:
    keep: list[int] = []
    for step in range(n_steps):
        kept = pattern[step % len(pattern)]
        keep.extend(1 if j in kept else 0 for j in range(CONV_RATE_INV))
    return keep


FAC_PUNCTURE_KEEP: list[int] = _expand_puncture(FAC_PUNCTURE_PATTERN, FAC_STEPS)
SDC_PUNCTURE_KEEP: list[int] = _expand_puncture(SDC_PUNCTURE_PATTERN, SDC_STEPS)

INTERLEAVER_T0 = 21


def _interleave_table(frame_size: int, t0: int) -> list[int]:
    s = 1 << int(np.ceil(np.log2(frame_size)))
    q = s // 4 - 1
    table = [0] * frame_size
    for i in range(1, frame_size):
        v = (t0 * table[i - 1] + q) % s
        while v >= frame_size:
            v = (t0 * v + q) % s
        table[i] = v
    return table


def _inverse_perm(perm: list[int]) -> list[int]:
    inverse = [0] * len(perm)
    for i, p in enumerate(perm):
        inverse[p] = i
    return inverse


# Dream's table is a TX gather (sent[i] = coded[table[i]]); a receive chain
# built from gather-only generic stages (out[t] = in[perm[t]]) needs its
# functional inverse to reproduce Dream's own scatter-style RX deinterleave
# (out[table[i]] = received[i]) exactly -- confirmed against fac_fec.py's
# proven interleave_bits/deinterleave round trip.
def fac_bit_perm() -> list[int]:
    return _inverse_perm(_interleave_table(FAC_CODED_BITS, INTERLEAVER_T0))


def sdc_bit_perm() -> list[int]:
    return _inverse_perm(_interleave_table(SDC_CODED_BITS, INTERLEAVER_T0))


# gnuradio's constellation_soft_decoder for QPSK emits the imaginary-axis soft
# bit before the real-axis one, transposing every coded (Re, Im) pair relative to
# Dream's coded-bit order; xor-1 undoes that transpose. Composing it with the
# pure inverse-interleave gives the single gather that carries GR's soft output
# straight to mother-code bit order (verified: 109/109 CRC-8-valid FAC blocks).
def fac_deinterleave_perm() -> list[int]:
    return [p ^ 1 for p in fac_bit_perm()]


def energy_prbs(n: int) -> np.ndarray:
    reg = (1 << 32) - 1
    out = np.zeros(n, dtype=np.int64)
    for i in range(n):
        bit = ((reg >> 4) & 1) ^ ((reg >> 8) & 1)
        reg = ((reg << 1) | bit) & 0x1FF
        out[i] = bit
    return out


def crc8(bits: np.ndarray) -> int:
    reg = (1 << 32) - 1
    mask_out = 1 << 8
    poly = (1 << 2) | (1 << 3) | (1 << 4)
    for b in bits:
        reg <<= 1
        if reg & mask_out:
            reg |= 1
        if b:
            reg ^= 1
        if reg & 1:
            reg ^= poly
    reg = ~reg & 0xFFFFFFFF
    return reg & (mask_out - 1)


def crc16(bits: np.ndarray) -> int:
    reg = (1 << 32) - 1
    mask_out = 1 << 16
    poly = (1 << 5) | (1 << 12)
    for b in bits:
        reg <<= 1
        if reg & mask_out:
            reg |= 1
        if b:
            reg ^= 1
        if reg & 1:
            reg ^= poly
    reg = ~reg & 0xFFFFFFFF
    return reg & (mask_out - 1)


def _bits_to_uint(bits: np.ndarray) -> int:
    value = 0
    for b in bits:
        value = (value << 1) | int(b)
    return value


# Frame payloads travel as packed bytes (marconi.bits.framing.bits_to_bytes,
# msb-first) end to end -- Task 3/4 call parse_fac/parse_sdc_label directly on
# bytes.fromhex(frame.payload_hex), so unpacking msb-first here is load-bearing.
def _unpack_msb(data: bytes) -> np.ndarray:
    return np.unpackbits(np.frombuffer(data, dtype=np.uint8), bitorder="big")


def parse_fac(block72: bytes) -> dict[str, int | str]:
    bits = _unpack_msb(block72)
    idx = 0

    def take(n: int) -> int:
        nonlocal idx
        v = _bits_to_uint(bits[idx : idx + n])
        idx += n
        return v

    base_enh = take(1)
    identity = take(2)
    occupancy = take(4)
    interleaver_depth = take(1)
    msc_mode = take(2)
    sdc_mode = take(1)
    num_services = take(4)
    reconfig_index = take(3)
    rfu = take(2)
    return {
        "base_enh": base_enh,
        "identity": format(identity, "02b"),
        "occupancy": format(occupancy, "04b"),
        "interleaver_depth": interleaver_depth,
        "msc_mode": format(msc_mode, "02b"),
        "sdc_mode": sdc_mode,
        "num_services": num_services,
        "reconfig_index": reconfig_index,
        "rfu": rfu,
    }


def parse_sdc_label(block316: bytes) -> str:
    bits = _unpack_msb(block316)
    pos = 4
    total = 300
    while pos < total:
        length = _bits_to_uint(bits[pos : pos + 7])
        pos += 7
        if length == 0:
            break
        pos += 1
        etype = _bits_to_uint(bits[pos : pos + 4])
        pos += 4
        body_start = pos
        if etype == 1:
            cpos = body_start + 4
            raw = bytes(
                _bits_to_uint(bits[cpos + 8 * i : cpos + 8 * i + 8])
                for i in range(length)
            )
            return raw.decode("utf-8", errors="replace")
        pos = body_start + length * 8 + 4
    return ""


class _CellSelectDemapParams(StageParams):
    select_perm: list[int]
    keep: StrictInt
    scheme: str
    order: StrictInt


class CellSelectSoftDemap(RxStage[CompileContext]):
    """SYMBOLS(complex)->BITS(soft): gather the wanted cells to the front of each
    equalized carrier block (blockinterleaver_cc over a full-block select perm),
    keep them (keep_m_in_n_c), then soft-demap (constellation_soft_decoder). The
    select perm, keep count and constellation are caller data — FAC and SDC differ
    only there. Composed from stock GR blocks, so it lives test-side, never src/."""

    name = "cell_select_soft_demap"
    from_level = Level.SYMBOLS
    to_level = Level.BITS
    family = "ofdm"
    params_model = _CellSelectDemapParams
    accepts_item_type = "c"
    accepts_carrier = Carrier.SOFT

    def emit_rx(self, b: CompileContext, params: Mapping[str, Any]) -> None:
        p = _CellSelectDemapParams.model_validate(dict(params))
        b.chain("blockinterleaver_cc", perm=[int(x) for x in p.select_perm], mode=True)
        b.chain("keep_m_in_n_c", m=int(p.keep), n=len(p.select_perm))
        b.chain("constellation_soft_decoder", scheme=p.scheme, order=int(p.order))

    def out_descriptor(
        self, in_desc: Descriptor, params: Mapping[str, Any]
    ) -> Descriptor:
        return Descriptor(Level.BITS, "f", in_desc.layout, Carrier.SOFT)


def fac_stage_registry() -> dict[str, Stage[CompileContext]]:
    return {**stage_registry(), CellSelectSoftDemap.name: CellSelectSoftDemap()}


def fac_phy_steps() -> list[ModemStep]:
    return [
        ModemStep(conv="ofdm_coherent_sync", params=sync_params()),
        ModemStep(
            conv=CellSelectSoftDemap.name,
            params={
                "select_perm": list(fac_select_perm()),
                "keep": len(FAC_CELLS),
                "scheme": "psk",
                "order": 4,
            },
        ),
        ModemStep(conv="deinterleave", params={"perm": list(fac_deinterleave_perm())}),
        ModemStep(conv="depuncture", params={"keep_mask": list(FAC_PUNCTURE_KEEP)}),
        ModemStep(
            conv="fec",
            params={
                "scheme": "cc",
                "rate_inv": CONV_RATE_INV,
                "polys": list(CONV_POLYS),
                "frame_bits": FAC_FRAME_BITS,
                "tail": TAIL,
            },
        ),
    ]


def energy_prbs_hex(n_bits: int) -> str:
    return np.packbits(energy_prbs(n_bits).astype(np.uint8)).tobytes().hex()


# CRC-8 in the crc-library parameterization equivalent to crc8() above (Dream
# CRC.cpp G=0x11D, init all-ones, output complemented). The bits-layer `crc`
# stage checks the trailing byte over the 64 info bits of each 72-bit FAC block.
FAC_CRC_POLY = 0x1D
FAC_CRC_INIT = 0xFF
FAC_CRC_XOROUT = 0xFF


def fac_codec() -> CodecSpec:
    return CodecSpec(
        name="drm_fac",
        path=[
            CodecStep(
                conv="descramble_bits",
                params={"sequence": energy_prbs_hex(FAC_FRAME_BITS)},
            ),
            CodecStep(conv="segment", params={"frame_body_len": FAC_FRAME_BITS}),
            CodecStep(conv="fixed_frame", params={"payload_bits": FAC_FRAME_BITS}),
            CodecStep(
                conv="crc",
                params={
                    "poly": FAC_CRC_POLY,
                    "bits": 8,
                    "init": FAC_CRC_INIT,
                    "reflected": False,
                    "xorout": FAC_CRC_XOROUT,
                },
            ),
        ],
    )
