"""DRM (Mode B, spectrum occupancy 3) caller data for the FAC/SDC closure gate:
OFDM geometry, the scattered/frequency pilot generators, the FAC/SDC cell and
bit-interleave tables, the shared K=7 rate-1/4 mother code + puncture
patterns, energy-dispersal PRBS, the two CRCs, and the FAC/SDC field parsers.
Every value is ported verbatim from the Dream-verified scratch scripts
(save_fac_cells.py / fac_fec.py / sdc_decode.py); none of it lives in src/.

The FAC/SDC coding tail (descramble + segment) is now product (phy "coding"
family stages) riding inside fac_phy_steps/sdc_phy_steps, the same GR-chain-
then-coding-tail composition as the DAB gate. fac_check/sdc_check are the
test-side CRC checks over run_rx's per-window carve, replacing the old
CodecSpec-based fac_codec/sdc_codec/SdcCrc16."""

from __future__ import annotations

import math
from typing import Any

import numpy as np
from helpers import bitops, crc

from marconi.engine.coding.stages_bits import DescrambleStep, SegmentStep
from marconi.engine.modulation.coding.stages import (
    DeinterleaveStep,
    DepunctureStep,
    FecStep,
)
from marconi.engine.modulation.ofdm.stages import (
    CellSelectStep,
    OfdmCoherentSyncStep,
    SoftDemapStep,
)
from marconi.engine.types.step import Step

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
    # Dream's ported pilot phases (scat_ref/FP_VALUES, kept verbatim -- they're
    # golden-pinned against drm_phy_ref.md in test_drm_caller_data.py) are
    # pi-off vs pilot_lattice's equalizer convention -- same genus as the
    # bit-reversed ConvEncoder masks noted below. Negating them only here, at
    # the equalizer boundary, lands cells true-phased so the negating
    # soft_demap -> fec convention holds (bit1-negative, measured); both
    # scattered and frequency pilots flip together.
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
        "pilot_i": [float(-v.real) for vs in values for v in vs],
        "pilot_q": [float(-v.imag) for vs in values for v in vs],
        "fp_carriers": [int(k) for k in FP_CARRIERS],
        "fp_i": [float(-v.real) for v in FP_VALUES],
        "fp_q": [float(-v.imag) for v in FP_VALUES],
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


def _select_perm(
    cells: list[tuple[int, int]], block_size: int, offset: int = 0
) -> list[int]:
    selected = [offset + fs * N_CARRIERS + _CARRIER_INDEX[k] for fs, k in cells]
    rest = [i for i in range(block_size) if i not in set(selected)]
    return selected + rest


def fac_select_perm() -> list[int]:
    return _select_perm(FAC_CELLS, FAC_SELECT_BLOCK)


# The coherent sync emits FRAME-aligned symbols but not SUPER-frame-aligned: the
# SDC (symbols 0-1 of the super-frame's frame 0) sits at one of 3 frame positions
# within each 45-symbol (3-frame) super-frame block, unknown a priori. `phase`
# picks that position — offset the gather by phase whole frames so the select perm
# lifts the SDC cells from block-symbols {phase*15, phase*15+1}. The caller sweeps
# phase 0/1/2 and keeps the one that yields CRC-16-valid SDC (scratch: phase 2).
def sdc_select_perm(phase: int = 0) -> list[int]:
    return _select_perm(SDC_CELLS, SDC_SELECT_BLOCK, phase * NS * N_CARRIERS)


# Standard (non-Dream-reversed) octal generators for the shared DAB/DRM K=7
# rate-1/4 mother code; bit-reversing each 7-bit value reproduces Dream's own
# ConvEncoder masks (0o155, 0o117, 0o123, 0o155) exactly. This is the exact
# polys/rate_inv already proven against real off-air DAB via the generic
# trellis "fec" stage (tests/e2e/test_dab_offair.py).
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


def sdc_deinterleave_perm() -> list[int]:
    return [p ^ 1 for p in sdc_bit_perm()]


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


# Frame payloads travel as packed bytes (helpers.bitops.bits_to_bytes, msb-first)
# end to end, so parse_fac/parse_sdc_label read msb-first packed bytes directly.
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


def energy_prbs_hex(n_bits: int) -> str:
    # descramble tiles a byte-granular hex sequence over the bit stream, so a
    # PRBS whose period isn't a byte multiple (SDC's 316) would drift by its pad
    # bits every block. Repeat it to the least byte-aligned multiple (LCM(n, 8)) so
    # each block re-aligns on PRBS[0]. FAC's 72 is already byte-aligned (reps=1).
    reps = 8 // math.gcd(n_bits, 8)
    tiled = np.tile(energy_prbs(n_bits), reps).astype(np.uint8)
    return np.packbits(tiled).tobytes().hex()


def fac_phy_steps() -> list[Step]:
    return [
        OfdmCoherentSyncStep(**sync_params()),
        CellSelectStep(select_perm=list(fac_select_perm()), keep=len(FAC_CELLS)),
        SoftDemapStep(scheme="psk", order=4),
        DeinterleaveStep(perm=list(fac_deinterleave_perm())),
        DepunctureStep(keep_mask=list(FAC_PUNCTURE_KEEP)),
        FecStep(
            scheme="cc",
            rate_inv=CONV_RATE_INV,
            polys=list(CONV_POLYS),
            frame_bits=FAC_FRAME_BITS,
            tail=TAIL,
        ),
        DescrambleStep(sequence=energy_prbs_hex(FAC_FRAME_BITS)),
        SegmentStep(frame_body_len=FAC_FRAME_BITS),
    ]


def sdc_phy_steps(phase: int) -> list[Step]:
    return [
        OfdmCoherentSyncStep(**sync_params()),
        CellSelectStep(select_perm=list(sdc_select_perm(phase)), keep=len(SDC_CELLS)),
        SoftDemapStep(scheme="psk", order=4),
        DeinterleaveStep(perm=list(sdc_deinterleave_perm())),
        DepunctureStep(keep_mask=list(SDC_PUNCTURE_KEEP)),
        FecStep(
            scheme="cc",
            rate_inv=CONV_RATE_INV,
            polys=list(CONV_POLYS),
            frame_bits=SDC_FRAME_BITS,
            tail=TAIL,
        ),
        DescrambleStep(sequence=energy_prbs_hex(SDC_FRAME_BITS)),
        SegmentStep(frame_body_len=SDC_FRAME_BITS),
    ]


# CRC-8 in the crc-library parameterization equivalent to crc8() above (Dream
# CRC.cpp G=0x11D, init all-ones, output complemented). fac_check applies it
# over the 64 info bits of each 72-bit FAC block (the trailing byte is the
# check field, same trailing-scope convention as helpers.crc.crc_check).
FAC_CRC_POLY = 0x1D
FAC_CRC_INIT = 0xFF
FAC_CRC_XOROUT = 0xFF


def fac_check(window_bits: np.ndarray) -> bytes | None:
    ok, body = crc.crc_check(
        bitops.bits_to_bytes(window_bits),
        poly=FAC_CRC_POLY,
        bits=8,
        init=FAC_CRC_INIT,
        xorout=FAC_CRC_XOROUT,
    )
    return body if ok else None


# SDC's CRC-16 rides the SAME crc-library parameterization proven for FAC's CRC-8
# (poly 0x1021, init all-ones, output ones-complement), but DRM computes it over a
# zero-nibble-prefixed scope — zeros(4) ++ block[:300], the 16-bit check at
# [300:316]. That sub-byte framing can't ride helpers.crc.crc_check's byte-
# trailing convention, so sdc_check re-expresses the exact scope by hand over
# helpers.crc.crc_value and returns the raw 316-bit block for parse_sdc_label.
SDC_CRC_POLY = 0x1021
SDC_CRC_INIT = 0xFFFF
SDC_CRC_XOROUT = 0xFFFF
SDC_CRC_BITS = 16
SDC_CRC_PREFIX_ZEROS = 4
SDC_CRC_SCOPE_BITS = 300


def sdc_check(window_bits: np.ndarray) -> bytes | None:
    block = np.asarray(window_bits, dtype=np.uint8)
    total = SDC_CRC_SCOPE_BITS + SDC_CRC_BITS
    if block.size < total:
        return None
    weights = 1 << np.arange(SDC_CRC_BITS - 1, -1, -1, dtype=np.int64)
    prefix = np.zeros(SDC_CRC_PREFIX_ZEROS, dtype=np.uint8)
    scope = bitops.bits_to_bytes(np.concatenate([prefix, block[:SDC_CRC_SCOPE_BITS]]))
    computed = crc.crc_value(
        scope,
        poly=SDC_CRC_POLY,
        bits=SDC_CRC_BITS,
        init=SDC_CRC_INIT,
        xorout=SDC_CRC_XOROUT,
    )
    received = int(block[SDC_CRC_SCOPE_BITS:total].astype(np.int64).dot(weights))
    if computed != received:
        return None
    return bitops.bits_to_bytes(block)
