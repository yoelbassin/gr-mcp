"""Real off-air LoRa SF11 (IQ_2), end-to-end through phy -> bits, CRC as the oracle.

The GR demod is non-deterministic run-to-run, so this asserts a robust
threshold (num_crc_ok >= 1), not an exact count.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from bits.test_lora_codec import _codec

from marconi.bits.registry import registry
from marconi.bits.seam import parse_bitstream
from marconi.core.bitfile import read_bits
from marconi.core.descriptor import Descriptor
from marconi.core.levels import Level
from marconi.core.models import Bitstream
from marconi.core.params import ParamValue
from marconi.phy.backends.gnuradio.runner import GnuRadioBackend, ensure_worker_warm
from marconi.phy.compiler import compile_modem
from marconi.phy.models import ModemSpec, ModemStep
from marconi.phy.stages import stage_registry

IQ = Descriptor(Level.IQ, "c")
_RATE = 1_000_000.0
_SLICE = (
    Path(__file__).resolve().parents[2]
    / "artifacts"
    / "assets"
    / "LoRa"
    / "iq2_frame.cf32"
)

_HEADER: dict[str, ParamValue] = {
    "sf": 11,
    "header_cr": 4,
    "ldro": True,
    "header_symbols": 8,
    "header_nibbles": 5,
    "sf_reduction": 2,
    "header_data_bits": 12,
    "header_parity": [3840, 2273, 1178, 599, 303],
    "field_payload_len": [0, 8],
    "field_cr": [8, 3],
    "field_has_crc": [11, 1],
    "field_parity": [15, 5],
}


def _modem() -> ModemSpec:
    return ModemSpec(
        name="lora_sf11_rx",
        symbol_rate=61.03515625,
        path=[
            ModemStep(conv="resample", params={"interpolation": 2, "decimation": 8}),
            ModemStep(
                conv="chirp_sync",
                params={"sf": 11, "oversample": 2, "zero_pad": 10, "preamble_len": 8},
            ),
            ModemStep(
                conv="dechirp", params={"sf": 11, "oversample": 2, "zero_pad": 10}
            ),
            ModemStep(conv="css_explicit_decode", params=_HEADER),
        ],
    )


@pytest.mark.skipif(
    not _SLICE.exists(), reason="LoRa slice absent — run tests/bits/make_lora_slice.py"
)
def test_iq2_decodes_crc_valid_rf_fingerpring(tmp_path: Path) -> None:
    ensure_worker_warm()
    snk = tmp_path / "lora_bits.u8"
    pipe = compile_modem(
        _modem(),
        stage_registry(),
        direction="rx",
        sample_rate=_RATE,
        start=IQ,
        source_io={"path": str(_SLICE)},
        sink_io={"path": str(snk)},
    )
    r = GnuRadioBackend().run_pipeline(pipe, timeout=180.0)
    assert r.status == "ok", r
    n = int(read_bits(snk).size)
    res = parse_bitstream(
        Bitstream(path=snk, num_bits=n, source_capture=_SLICE), _codec(), registry()
    )
    assert res.num_crc_ok >= 1, f"expected a CRC-valid LoRa frame, got {res.num_crc_ok}"
    decoded = bytes.fromhex(res.frames[0].payload_hex)
    assert decoded.startswith(b"RF fingerpring Project for Lora"), decoded[:32]
