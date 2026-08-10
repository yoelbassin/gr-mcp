"""Real off-air AIS, one Modem spanning phy through the coding tail, CRC as
the oracle.

Burst gating carries this decode. AIS is bursty, and a clock-recovery loop that
free-runs through the gaps arrives at the next burst mistimed, so most of the
capture was being thrown away. Measured, 3 runs per arm, counting only
CRC-valid frames with a non-zero payload (a zero payload passes a zero-init
CRC and is not evidence):

    baseline              14-19 messages, 10-15 vessels
    + agc only            10-21 messages, 10-16 vessels   <- agc alone does nothing
    + agc + squelch      251-330 messages, 29-31 vessels

Content check on the squelched arm: 824/824 decoded positions fall inside the
receiver's VHF footprint (lat 51.448..51.491, lon 0.174..0.379) and every
msg_type is in 1..27, so the extra frames are real traffic, not CRC collisions.
30 vessels over 60 s at class-A reporting rates is 200-900 messages, which is
the band this now lands in; the old chain was recovering about 5% of the
capture.

Run-to-run spread in the frame count remains wide (158..313 over 15 runs; the
root cause is volk kernel dispatch by buffer address, untouched by gating), so
the count is only a sanity floor. Distinct vessels is the stable statistic --
28..31 gated vs 10..16 ungated -- and carries the tight gate.

The NRZI decode (differential, invert=True) is now a product coding stage
appended to the modem after slice, the same as the old codec's leading step.
HDLC deframing/CRC/parse move to test-side helpers over the plain bitstream:
this gate seeds no windows (AIS has no sync_word -- hdlc_frames finds its own
flags), so framing.hdlc_frames walks res.bitstream directly, exactly as the
old hdlc_deframe_rx walked the differential-decoded carrier's bits.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from helpers import crc, framing, parse

from marconi.engine.backends.gnuradio.runner import ensure_worker_warm
from marconi.engine.coding.stages_bits import DifferentialStep
from marconi.engine.io.bitfile import read_bits
from marconi.engine.io.source import SourceSlice
from marconi.engine.modulation.fsk.stages import FskStep
from marconi.engine.run import run_rx
from marconi.engine.stages.conditioning import AgcStep, ChannelizeStep, SquelchStep
from marconi.engine.stages.general import SliceStep
from marconi.engine.stages.registry import stage_registry
from marconi.engine.types.descriptor import Descriptor
from marconi.engine.types.enums import AgcMode, ItemType
from marconi.engine.types.levels import Level
from marconi.engine.types.models import Modem

IQ = Descriptor(Level.IQ, ItemType.C)
RATE = 250000.0
_SLICE = (
    Path(__file__).resolve().parents[3]
    / "artifacts"
    / "assets"
    / "AIS"
    / "ais_60s.cf32"
)

# Every AIS type shares this header; the position-report body belongs only to
# types 1/2/3. Applying it to a type-4/5/18 frame would mis-parse it, so it
# rides a msg_type dispatch and other types decode to the header alone.
_AIS_COMMON: list[dict[str, object]] = [
    {"name": "msg_type", "bits": 6},
    {"name": "repeat", "bits": 2},
    {"name": "mmsi", "bits": 30},
]
_AIS_POSITION: list[dict[str, object]] = [
    {"name": "nav_status", "bits": 4},
    {"name": "rot", "bits": 8, "signed": True},
    {"name": "sog", "bits": 10},
    {"name": "pos_accuracy", "bits": 1},
    {"name": "lon", "bits": 28, "signed": True},
    {"name": "lat", "bits": 27, "signed": True},
    {"name": "cog", "bits": 12},
    {"name": "heading", "bits": 9},
    {"name": "timestamp", "bits": 6},
    {"name": "maneuver", "bits": 2},
    {"name": "spare", "bits": 3},
    {"name": "raim", "bits": 1},
    {"name": "radio", "bits": 19},
]
_AIS_CASES = [
    {"when": 1, "fields": _AIS_POSITION},
    {"when": 2, "fields": _AIS_POSITION},
    {"when": 3, "fields": _AIS_POSITION},
]

# old crc CodecStep params, verbatim.
_AIS_CRC: dict[str, int | str] = {
    "poly": 0x1021,
    "bits": 16,
    "init": 0xFFFF,
    "reflected": True,
    "xorout": 0xFFFF,
    "bit_order": "lsb",
}


def _ais_modem(center_hz: float) -> Modem:
    return Modem(
        symbol_rate=9600.0,
        path=[
            ChannelizeStep(decim=5, bandwidth_hz=14000.0, center_hz=center_hz),
            # window_symbols spans many burst gaps on purpose: a short window
            # lifts the inter-burst noise floor to meet the squelch threshold
            # and nothing gets muted (tests/unit/engine/stages/test_squelch.py
            # pins this).
            AgcStep(mode=AgcMode.POWER, window_symbols=4096.0),
            SquelchStep(threshold_db=-12.0, alpha_symbols=2.0),
            FskStep(deviation=2400.0),
            SliceStep(),
            DifferentialStep(invert=True),
        ],
    )


@pytest.mark.skipif(
    not _SLICE.exists(), reason="AIS slice absent — run tests/e2e/ais/make_ais_slice.py"
)
def test_ais_offair_crc(tmp_path: Path) -> None:
    ensure_worker_warm()
    total = 0
    msgs: list[dict[str, int | str]] = []
    for center in (-25000.0, 25000.0):
        workdir = tmp_path / str(int(center))
        workdir.mkdir()
        res = run_rx(
            _ais_modem(center),
            stage_registry(),
            sample_rate=RATE,
            start=IQ,
            workdir=workdir,
            source=SourceSlice(path=_SLICE),
        )
        assert res.status == "ok", res
        assert res.bitstream is not None
        bits = read_bits(res.bitstream.path)
        for _start, payload in framing.hdlc_frames(bits, bit_order="lsb"):
            ok, body = crc.crc_check(
                payload,
                poly=int(_AIS_CRC["poly"]),
                bits=int(_AIS_CRC["bits"]),
                init=int(_AIS_CRC["init"]),
                reflected=bool(_AIS_CRC["reflected"]),
                xorout=int(_AIS_CRC["xorout"]),
                bit_order=str(_AIS_CRC["bit_order"]),
            )
            if not ok:
                continue
            message = parse.parse_message(
                body,
                _AIS_COMMON,
                bit_order="msb",
                discriminator="msg_type",
                cases=_AIS_CASES,
            )
            if message is None:
                continue
            total += 1
            msgs.append(message)
    # The frame COUNT is the noisy statistic (158..313 over 15 runs), so it is
    # only a sanity floor: 3x the pre-gating chain's ~19, which a regression to
    # that chain cannot clear. The tight gate is the vessel count below.
    assert total >= 60, (
        f"expected >= 60 CRC-valid AIS frames (15-run range 158..313 with burst "
        f"gating; the pre-gating chain yielded ~19), got {total}"
    )
    assert msgs, "no CRC-valid AIS messages were parsed"
    types = [int(m["msg_type"]) for m in msgs]
    assert all(1 <= t <= 27 for t in types), f"out-of-range AIS msg_type: {types}"
    # dispatch, not mis-parse: only types 1/2/3 carry the position-report body
    for m in msgs:
        if "sog" in m:
            assert int(m["msg_type"]) in (1, 2, 3), f"body on wrong type: {m}"
    # content invariants against the capture's ground truth (CRC alone cannot
    # catch a parse-layer field misalignment): the 30-bit MMSI field must hold
    # a 9-digit value, distinct vessels rule out one lucky frame decoded
    # repeatedly, and every position report must sit within VHF range of the
    # one receiver.
    mmsis = {int(m["mmsi"]) for m in msgs}
    assert all(0 < v < 1_000_000_000 for v in mmsis), f"non-MMSI value: {mmsis}"
    # Distinct vessels is the stable statistic: 28..31 over 15 runs with burst
    # gating, against 10..16 without it. Tight enough that losing gating fails.
    assert len(mmsis) >= 22, f"too few distinct vessels: {sorted(mmsis)}"
    # every probed frame was a type-1/2/3 position report; a layout that no
    # longer fits the frame drops the body silently, so its absence is itself
    # the regression signal
    assert any("lat" in m for m in msgs), f"no position reports decoded: {msgs}"
    for m in msgs:
        if "lat" in m:
            lat, lon = int(m["lat"]) / 600000.0, int(m["lon"]) / 600000.0
            assert (
                50.9 < lat < 52.0 and -0.5 < lon < 1.2
            ), f"position outside the receiver's range: {m}"
