"""THROWAWAY reality check (not for commit): can the rebuilt phy demod a real
off-air AIS capture end-to-end, verified by CRC?

Pipeline exercised: channelize -> fsk -> slice  (real use of the Plan-6
channelizer on a wideband multi-channel capture). The bits layer (NRZI +
HDLC de-stuff + CRC-16/X.25) is done inline here, since `bits` is deferred.
CRC validity is the oracle: we brute-force the NRZI convention and let the
HDLC FCS residue (0xF0B8) decide.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from phy._dsp import read_bits

from marconi.core.descriptor import Descriptor
from marconi.core.levels import Level
from marconi.phy.backends.gnuradio.runner import GnuRadioBackend, ensure_worker_warm
from marconi.phy.compiler import compile_modem
from marconi.phy.models import ModemSpec, ModemStep
from marconi.phy.stages import stage_registry

IQ = Descriptor(Level.IQ, "c")
RATE = 250000.0
SLICE = Path(
    "/private/tmp/claude-501/-Users-joel-Clones-gr-mcp/"
    "9e32fa17-090e-4ce1-909f-7c21ab84a067/scratchpad/ais_60s.cf32"
)
FLAG = np.array([0, 1, 1, 1, 1, 1, 1, 0], dtype=np.uint8)


def _ais_modem(center_hz: float) -> ModemSpec:
    return ModemSpec(
        symbol_rate=9600.0,
        path=[
            ModemStep(
                conv="channelize",
                params={"decim": 5, "bandwidth_hz": 14000.0, "center_hz": center_hz},
            ),
            ModemStep(conv="fsk", params={"deviation": 2400.0}),
            ModemStep(conv="slice", params={}),
        ],
    )


def _run(center_hz: float, snk: Path) -> np.ndarray:
    pipe = compile_modem(
        _ais_modem(center_hz),
        stage_registry(),
        direction="rx",
        sample_rate=RATE,
        start=IQ,
        source_io={"path": str(SLICE)},
        sink_io={"path": str(snk)},
    )
    r = GnuRadioBackend().run_pipeline(pipe)
    assert r.status == "ok", r
    return read_bits(snk)


def _nrzi(chan: np.ndarray, transition_is_zero: bool) -> np.ndarray:
    trans = (chan[1:] != chan[:-1]).astype(np.uint8)
    return (1 - trans) if transition_is_zero else trans


def _find_flags(bits: np.ndarray) -> np.ndarray:
    if len(bits) < 8:
        return np.array([], dtype=int)
    w = np.lib.stride_tricks.sliding_window_view(bits, 8)
    return np.where((w == FLAG).all(axis=1))[0]


def _destuff(bits: np.ndarray) -> np.ndarray:
    out = []
    ones = 0
    i = 0
    n = len(bits)
    while i < n:
        b = int(bits[i])
        out.append(b)
        if b == 1:
            ones += 1
            if ones == 5:
                i += 1  # drop the stuffed 0
                ones = 0
        else:
            ones = 0
        i += 1
    return np.array(out, dtype=np.uint8)


def _fcs_ok(frame: np.ndarray) -> bool:
    """CRC-16/X.25 bit-serial (reflected poly 0x8408, init 0xFFFF); a good HDLC
    frame (payload+FCS, transmission order) leaves residue 0xF0B8."""
    reg = 0xFFFF
    for b in frame:
        reg ^= int(b) & 1
        reg = (reg >> 1) ^ 0x8408 if (reg & 1) else (reg >> 1)
    return reg == 0xF0B8


def _ais_type(frame: np.ndarray) -> int:
    payload = frame[:-16]
    nb = len(payload) // 8
    if nb < 1:
        return -1
    octets = payload[: nb * 8].reshape(nb, 8)
    msb = octets[:, ::-1].reshape(-1)  # HDLC octets are LSB-first on the wire
    return int(msb[:6].dot(1 << np.arange(6)[::-1]))


def _decode(databits: np.ndarray) -> tuple[int, list[int], list[int]]:
    flags = _find_flags(databits)
    valid, types, lens = 0, [], []
    for a, b in zip(flags, flags[1:]):
        seg = databits[a + 8 : b]
        if not (24 <= len(seg) <= 2000):
            continue
        fr = _destuff(seg)
        if len(fr) >= 32 and _fcs_ok(fr):
            valid += 1
            lens.append(len(fr))
            types.append(_ais_type(fr))
    return valid, types, lens


def test_ais_offair_crc() -> None:
    assert SLICE.exists(), f"missing IQ slice {SLICE} — run make_ais_slice.py"
    ensure_worker_warm()
    tmp = SLICE.parent
    total = 0
    for center, name in [(-25000.0, "AIS1 (-25k)"), (25000.0, "AIS2 (+25k)")]:
        chan_bits = _run(center, tmp / f"ais_bits_{int(center)}.u8")
        print(f"\n=== {name}: {len(chan_bits)} demod bits ===")
        for tz in (True, False):
            data = _nrzi(chan_bits, transition_is_zero=tz)
            n_flags = len(_find_flags(data))
            valid, types, lens = _decode(data)
            from collections import Counter

            th = dict(sorted(Counter(types).items()))
            lh = dict(sorted(Counter(lens).items()))
            print(
                f"  NRZI(transition->{'0' if tz else '1'}): "
                f"flags={n_flags:5d}  CRC-valid frames={valid:4d}  "
                f"types={th}  lens={lh}"
            )
            total += valid
    print(f"\nTOTAL CRC-valid AIS frames across both channels: {total}")
    assert total >= 1, "no CRC-valid AIS frame recovered"
