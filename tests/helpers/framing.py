from __future__ import annotations

import numpy as np
from helpers.bitops import bits_to_bytes, bytes_to_bits, read_uint


def carve_fixed(
    bits: np.ndarray, starts: list[int], payload_bits: int
) -> list[np.ndarray]:
    bits = np.asarray(bits, dtype=np.uint8)
    return [bits[s : s + payload_bits] for s in starts if s + payload_bits <= len(bits)]


def carve_length(
    payload: bytes,
    *,
    length_bits: int,
    base_bytes: int,
    unit_bytes: int,
    bit_order: str = "msb",
    offset_bits: int = 0,
) -> bytes | None:
    pbits = bytes_to_bits(payload, bit_order)
    if offset_bits + length_bits > pbits.size:
        return None
    length = read_uint(pbits, offset_bits, length_bits, bit_order)
    nbytes = base_bytes + unit_bytes * length
    if nbytes <= 0 or nbytes > len(payload):
        return None
    return payload[:nbytes]


def carve_delimiter(
    bits: np.ndarray, start: int, *, delimiter_hex: str, tail_bits: int = 0
) -> np.ndarray | None:
    pat = bytes_to_bits(bytes.fromhex(delimiter_hex))
    bits = np.asarray(bits, dtype=np.uint8)
    m = pat.size
    seg = bits[start:]
    if seg.size < m:
        return None
    windows = np.lib.stride_tricks.sliding_window_view(seg, m)
    hits = np.flatnonzero((windows != pat).sum(axis=1) == 0)
    if hits.size == 0:
        return None
    end = start + int(hits[0]) + m + tail_bits
    if end > bits.size:
        return None
    return bits[start:end]


def bitstuff(bits: np.ndarray) -> np.ndarray:
    out: list[int] = []
    ones = 0
    for bit in bits:
        out.append(int(bit))
        if bit:
            ones += 1
            if ones == 5:
                out.append(0)
                ones = 0
        else:
            ones = 0
    return np.array(out, dtype=np.uint8)


def _destuff(bits: np.ndarray) -> np.ndarray | None:
    out: list[int] = []
    ones = 0
    i = 0
    n = len(bits)
    while i < n:
        bit = int(bits[i])
        out.append(bit)
        if bit:
            ones += 1
            if ones == 5:
                if i + 1 >= n or int(bits[i + 1]) == 1:
                    return None
                i += 1
                ones = 0
        else:
            ones = 0
        i += 1
    return np.array(out, dtype=np.uint8)


def _find_flags(bits: np.ndarray) -> list[int]:
    # A flag 01111110 is a 0, six 1s (a length-6 window summing to 6), then a 0.
    # One cumsum + slice compares finds every one in a single pass -- no per-bit
    # loop, no O(8n) window materialization. The 0-bookends cannot fall inside
    # another flag's ones-run, so matches are always >=7 apart, exactly what the
    # old greedy i+=7 scan reported.
    n = bits.size
    if n < 8:
        return []
    cs = np.concatenate(([0], np.cumsum(bits, dtype=np.int32)))
    six_ones = (cs[7:n] - cs[1 : n - 6]) == 6
    match = six_ones & (bits[0 : n - 7] == 0) & (bits[7:n] == 0)
    return np.flatnonzero(match).tolist()


def hdlc_frames(bits: np.ndarray, bit_order: str = "msb") -> list[tuple[int, bytes]]:
    bits = np.asarray(bits, dtype=np.uint8)
    flags = _find_flags(bits)
    frames: list[tuple[int, bytes]] = []
    for a, b in zip(flags, flags[1:]):
        region = bits[a + 8 : b]
        if region.size == 0:
            continue
        content = _destuff(region)
        if content is None or content.size == 0 or content.size % 8 != 0:
            continue
        frames.append((a + 8, bits_to_bytes(content, bit_order)))
    return frames


def xor_payload(payload: bytes, sequence_hex: str) -> bytes:
    seq = bytes.fromhex(sequence_hex)
    if len(seq) < len(payload):
        raise ValueError(
            f"descramble sequence ({len(seq)} bytes) shorter than payload "
            f"({len(payload)} bytes)"
        )
    return bytes(b ^ seq[i] for i, b in enumerate(payload))


def xor_bits(bits: np.ndarray, sequence_hex: str) -> np.ndarray:
    seq = bytes_to_bits(bytes.fromhex(sequence_hex))
    bits = np.asarray(bits, dtype=np.uint8)
    if seq.size == 0 or bits.size == 0:
        return bits
    return np.bitwise_xor(bits, np.resize(seq, bits.size))
