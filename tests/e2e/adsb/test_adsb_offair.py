"""First pure-product burst-PPM gate: one spec (native-rate, no-resample,
no-agc open-loop ook_envelope) carries the PHY from IQ to the soft chip
stream on two real off-air 1090 MHz assets. The datasheet tail - Mode S
preamble constant, DF carve, CRC-24 - is test-side by design: framing and
CRC parameters are protocol-datasheet knowledge, not DSP judgment.

REVISED after the ceiling investigation (Task 12 of this batch): the
`resample` stage's anti-imaging low-pass smears the 0.5 us chips (measured:
more resample -> fewer frames), and a sliding-window agc steps its gain
mid-frame. The open-loop sampler now runs at native sps, so both assets
(2.4 Msps -> sps 1.2, 2 Msps -> sps 1.0) decode with no resample and no agc.

The product path ends at the SOFT chip stream (bare ook_envelope(loop_bw=0));
this test soft-pairs the 224 data chips into 112 bits itself - hard slicing
loses several dB on marginal chips that soft pairing still resolves. The
preamble detector is a power-ratio (not correlation) discriminator: pulse
energy over the preamble's 4 pulse chips must both dominate its 8 gap chips
and clear the stream's noise floor.

Every chip position passing that energy test is decoded and CRC-24 checked
directly, with no energy-ranked peak suppression in between: MEASURED, the
preamble's own pulse pattern autocorrelates with itself at a 7-chip shift
(two of the four pulse chips re-hit), producing a second, sometimes-higher-
energy candidate 7-14 chips from the true one. An energy-based "keep the
local max within a window" pass discarded correctly-aligned, CRC-valid
candidates in favor of that structural sidelobe - measured on the 20 s asset
against an independent brute-force CRC scan of every energy-mask hit (159
real frames exist in the capture; 153 of them clear the energy mask at least
once): a 16-chip suppression window recovered only 145 of those 153, a
3-chip window 152, and dropping suppression entirely recovers all 153.
CRC-24 is already close to a perfect discriminator (a random 24-bit space),
so it - not a proxy energy ranking - is what should decide which nearby
candidate is the real message; duplicate detections of the same physical
preamble collapse for free at the UNIQUE-payload count below."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from helpers.crc import crc_check_bits

from marconi.engine.backends.gnuradio.runner import ensure_worker_warm
from marconi.engine.io.bitfile import read_symbols
from marconi.engine.io.source import SourceSlice
from marconi.engine.modulation.ook.stages import OokEnvelopeStep
from marconi.engine.run import run_rx
from marconi.engine.stages.registry import stage_registry
from marconi.engine.types.descriptor import Descriptor
from marconi.engine.types.enums import ItemType
from marconi.engine.types.levels import Level
from marconi.engine.types.models import Modem

IQ = Descriptor(Level.IQ, ItemType.C)
_ASSET_DIR = Path(__file__).resolve().parents[3] / "artifacts" / "assets" / "ADS-B"
_CHIP_RATE = 2_000_000.0  # Mode S: fixed 0.5us chips regardless of capture rate

# Mode S preamble (datasheet, test-side): 8us / 16 chips, pulses at chips
# {0, 2, 7, 9}; gap chips well clear of any pulse edge (excludes 1, 8, 10, 15,
# each adjacent to a pulse transition).
_PULSE_CHIPS = (0, 2, 7, 9)
_GAP_CHIPS = (3, 4, 5, 6, 11, 12, 13, 14)
_PREAMBLE_CHIPS = 16
_DATA_BITS = 112  # DF17/18 extended squitter length; DF11 uses the first 56
_MSG_CHIPS = _PREAMBLE_CHIPS + 2 * _DATA_BITS  # 240

_CRC_POLY = 0xFFF409  # Mode S CRC-24 generator, width 24, init 0, xorout 0
_CRC_WIDTH = 24

# floor = max(2, int(min_observed * 0.8)) from 10 serial runs of
# `uv run pytest tests/e2e/adsb -q` (volk non-determinism -> thresholds, not
# exact counts; this simple envelope+sampler chain measured bit-identical
# 10/10 on both assets, but the floor still isn't pinned to the exact count).
# Observed unique-valid range over 10 runs:
#   adsb_20s.cf32:            min 153, max 153
#   adsb_live_2msps_5s.cf32:  min 7,   max 7
_CASES: dict[str, dict[str, float]] = {
    "adsb_20s.cf32": {"rate": 2_400_000.0, "floor": 122},
    "adsb_live_2msps_5s.cf32": {"rate": 2_000_000.0, "floor": 5},
}


def _modem() -> Modem:
    return Modem(symbol_rate=_CHIP_RATE, path=[OokEnvelopeStep(loop_bw=0.0)])


def _preamble_candidates(m: np.ndarray, valid_len: int) -> np.ndarray:
    pw = sum(m[off : off + valid_len] for off in _PULSE_CHIPS)
    gp = sum(m[off : off + valid_len] for off in _GAP_CHIPS)
    med = float(np.median(m))
    mask = (pw / 4.0 > 2.0 * gp / 8.0) & (pw / 4.0 > 3.0 * med)
    return np.flatnonzero(mask)


def _crc_valid_frame(bits: np.ndarray) -> np.ndarray | None:
    if not bits.any():  # CRC-24 of an all-zero body is 0 - the known blind spot
        return None
    df = int(bits[:5].dot([16, 8, 4, 2, 1]))
    if df in (17, 18):
        ok, _ = crc_check_bits(bits, poly=_CRC_POLY, width=_CRC_WIDTH)
        return bits if ok else None
    if df == 11:
        frame = bits[:56]
        ok, _ = crc_check_bits(frame, poly=_CRC_POLY, width=_CRC_WIDTH)
        return frame if ok else None
    return None


def _unique_valid(soft: np.ndarray) -> int:
    n = soft.size
    valid_len = n - _MSG_CHIPS + 1
    if valid_len <= 0:
        return 0
    m = (soft.astype(np.float32) + 1.0) / 2.0
    unique: set[tuple[int, ...]] = set()
    for k in _preamble_candidates(m, valid_len).tolist():
        data0 = k + _PREAMBLE_CHIPS
        offs = data0 + 2 * np.arange(_DATA_BITS)
        bits = (soft[offs] > soft[offs + 1]).astype(np.uint8)
        frame = _crc_valid_frame(bits)
        if frame is not None:
            unique.add(tuple(frame.tolist()))
    return len(unique)


_PARAMS = [
    pytest.param(
        name,
        marks=pytest.mark.skipif(
            not (_ASSET_DIR / name).exists(),
            reason=(
                "ADS-B asset absent - run tests/e2e/adsb/make_adsb_slice.py or "
                f"make_adsb_live_slice.py ({name})"
            ),
        ),
    )
    for name in sorted(_CASES)
]


@pytest.mark.parametrize("name", _PARAMS)
def test_adsb_offair(tmp_path: Path, name: str) -> None:
    ensure_worker_warm()
    case = _CASES[name]
    res = run_rx(
        _modem(),
        stage_registry(),
        sample_rate=case["rate"],
        start=IQ,
        workdir=tmp_path,
        source=SourceSlice(path=_ASSET_DIR / name),
    )
    assert res.status == "ok", res
    assert res.symbolstream is not None
    soft = read_symbols(res.symbolstream.path, "f")
    unique = _unique_valid(soft)
    assert (
        unique >= case["floor"]
    ), f"{name}: {unique} unique CRC-valid frames, floor {case['floor']}"
