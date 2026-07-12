"""Real off-air LoRa SF11 (IQ_2), end-to-end phy -> bits, CRC as the oracle.
The slice holds TWO complete frames; both must decode CRC-valid in one run —
chirp_sync re-arms per preamble, burst_probe surfaces both burst marks, and
the bits-layer css_explicit_decode carves one frame per mark (issue 03)."""

from __future__ import annotations

from pathlib import Path

import pytest
from bits.test_lora_codec import _codec
from phy._css_lora import HEADER as _LORA_HEADER

from marconi.bits.models import CodecSpec, CodecStep
from marconi.bits.registry import registry
from marconi.bits.seam import parse_bitstream
from marconi.core.bitfile import read_symbols
from marconi.core.descriptor import Descriptor
from marconi.core.levels import Level
from marconi.core.models import Symbolstream
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

_HEADER: dict[str, ParamValue] = dict(_LORA_HEADER)


def _modem() -> ModemSpec:
    return ModemSpec(
        name="lora_sf11_rx",
        symbol_rate=61.03515625,
        path=[
            ModemStep(conv="resample", params={"interpolation": 2, "decimation": 8}),
            ModemStep(
                conv="chirp_sync",
                params={
                    "sf": 11,
                    "oversample": 2,
                    "zero_pad": 10,
                    "preamble_len": 8,
                    "sfd_symbols": 2.25,
                    "sync_symbols": 2,
                },
            ),
            ModemStep(
                conv="dechirp",
                params={
                    "sf": 11,
                    "oversample": 2,
                    "zero_pad": 10,
                    "preamble_len": 8,
                    "sfd_symbols": 2.25,
                    "sync_symbols": 2,
                },
            ),
            ModemStep(conv="burst_probe", params={}),
        ],
    )


def _lora_codec() -> CodecSpec:
    return CodecSpec(
        name="lora_sf11",
        path=[CodecStep(conv="css_explicit_decode", params=dict(_HEADER))]
        + _codec().path,
    )


@pytest.mark.skipif(
    not _SLICE.exists(), reason="LoRa slice absent — run tests/bits/make_lora_slice.py"
)
def test_iq2_decodes_crc_valid_rf_fingerpring(tmp_path: Path) -> None:
    ensure_worker_warm()
    snk = tmp_path / "lora_syms.i16"
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
    marks: list[int] = []
    for d in r.diagnostics.values():
        b = d.get("bursts")
        if isinstance(b, list):
            marks = [int(m) for m in b]
    assert len(marks) >= 2, r.diagnostics
    n = int(read_symbols(snk).size)
    res = parse_bitstream(
        Symbolstream(path=snk, num_symbols=n, marks=marks, source_capture=_SLICE),
        _lora_codec(),
        registry(),
    )
    assert res.num_crc_ok >= 2, f"expected both frames CRC-valid, got {res.num_crc_ok}"
    for f in res.frames[:2]:
        decoded = bytes.fromhex(f.payload_hex)
        assert decoded.startswith(b"RF fingerpring Project for Lora"), decoded[:32]
