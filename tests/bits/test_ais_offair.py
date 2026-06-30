"""Real off-air AIS, end-to-end through phy -> bits, CRC as the oracle.

The GR demod is non-deterministic run-to-run, so this asserts a robust
threshold (num_crc_ok >= 4), not an exact count. Engine exactness is proven
deterministically in test_engine.py / test_framing.py.
"""

from __future__ import annotations

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
RATE = 250000.0
_SLICE = (
    Path(__file__).resolve().parents[2]
    / "artifacts"
    / "assets"
    / "AIS"
    / "ais_60s.cf32"
)

_AIS_FIELDS = [
    {"name": "msg_type", "bits": 6},
    {"name": "repeat", "bits": 2},
    {"name": "mmsi", "bits": 30},
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


def _ais_codec() -> CodecSpec:
    return CodecSpec(
        name="ais",
        path=[
            CodecStep(conv="differential", params={"invert": True}),
            CodecStep(conv="hdlc_deframe", params={"bit_order": "lsb"}),
            CodecStep(
                conv="crc",
                params={
                    "poly": 0x1021,
                    "bits": 16,
                    "init": 0xFFFF,
                    "reflected": True,
                    "xorout": 0xFFFF,
                    "bit_order": "lsb",
                },
            ),
            CodecStep(conv="parse", params={"bit_order": "msb", "fields": _AIS_FIELDS}),
        ],
    )


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


@pytest.mark.skipif(
    not _SLICE.exists(), reason="AIS slice absent — run tests/bits/make_ais_slice.py"
)
def test_ais_offair_crc(tmp_path: Path) -> None:
    ensure_worker_warm()
    reg = registry()
    total = 0
    for center in (-25000.0, 25000.0):
        snk = tmp_path / f"bits_{int(center)}.u8"
        pipe = compile_modem(
            _ais_modem(center),
            stage_registry(),
            direction="rx",
            sample_rate=RATE,
            start=IQ,
            source_io={"path": str(_SLICE)},
            sink_io={"path": str(snk)},
        )
        r = GnuRadioBackend().run_pipeline(pipe, timeout=120.0)
        assert r.status == "ok", r
        n = int(read_bits(snk).size)
        res = parse_bitstream(
            Bitstream(path=snk, num_bits=n, source_capture=_SLICE), _ais_codec(), reg
        )
        total += res.num_crc_ok
    assert total >= 4, f"expected >= 4 CRC-valid AIS frames, got {total}"
