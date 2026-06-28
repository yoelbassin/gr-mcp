"""CSS off-air symbol oracle tests.

Two real captures verify that the rebuilt CSS receive chain (chirp_sync +
dechirp) recovers correct symbols from actual SDR hardware:

  Flinders — non-LoRa SF11 DOWN-chirp (SDRSharp, 2.048 Msps)
  LoRa SF7  — standard UP-chirp (USRP, 1 Msps)

Both tests are skipped when the asset files are absent so the suite remains
green on machines that only have the source tree.
"""

from __future__ import annotations

import wave
from pathlib import Path

import numpy as np
import pytest

from marconi.core.descriptor import Descriptor
from marconi.core.levels import Level
from marconi.core.params import ParamValue
from marconi.phy.backends.gnuradio.runner import GnuRadioBackend, ensure_worker_warm
from marconi.phy.compiler import compile_modem
from marconi.phy.models import ModemSpec, ModemStep

IQ = Descriptor(Level.IQ, "c")

_ASSET_ROOT = Path(__file__).parent.parent.parent / "artifacts" / "assets" / "CSS"

_FLINDERS_WAV = (
    _ASSET_ROOT
    / "SDRSharp_20160326_002438Z_433693kHz_IQ Flinders Uni Freezer Telemetry.wav"
)
_LORA_SF7_BIN = (
    _ASSET_ROOT
    / "LoRa"
    / "LoRA_SF_7_channel_1_dutycycle_long_time_2min_usrpgain_10dB_868.1_MHZ_1M_SPS.bin"
)

_FLINDERS_ORACLE = [
    1309,
    641,
    345,
    861,
    85,
    1749,
    1897,
    1481,
    873,
    393,
    1081,
    865,
    521,
    185,
    1853,
    1473,
    41,
    1933,
    829,
    1093,
]
_LORA_SF7_ORACLE = [
    85,
    1,
    1,
    113,
    85,
    97,
    101,
    77,
    27,
    79,
    64,
    91,
    56,
    38,
    91,
    50,
    54,
    13,
    55,
    120,
]


def _compile(modem: ModemSpec, rate: float, src: Path, snk: Path):
    from marconi.phy.stages import stage_registry

    return compile_modem(
        modem,
        stage_registry(),
        direction="rx",
        sample_rate=rate,
        start=IQ,
        source_io={"path": str(src)},
        sink_io={"path": str(snk)},
    )


@pytest.mark.skipif(not _FLINDERS_WAV.exists(), reason="Flinders asset absent")
def test_css_offair_flinders_sf11_downchirp(tmp_path: Path) -> None:
    """Non-LoRa SF11 DOWN-chirp off-air capture: first 20 symbols match oracle."""
    ensure_worker_warm()
    be = GnuRadioBackend()

    # Asset prep: stereo 16-bit WAV -> complex64
    w = wave.open(str(_FLINDERS_WAV), "rb")
    raw = np.frombuffer(w.readframes(w.getnframes()), dtype="<i2")
    w.close()
    iq = raw.astype(np.float32).reshape(-1, 2) / 32768.0
    x = (iq[:, 0] + 1j * iq[:, 1]).astype(np.complex64)
    src = tmp_path / "flinders.cf32"
    x.tofile(src)

    sf, os_, zp, pl = 11, 2, 4, 8
    p: dict[str, ParamValue] = {
        "sf": sf,
        "oversample": os_,
        "zero_pad": zp,
        "preamble_len": pl,
    }
    chan_p: dict[str, ParamValue] = {
        "decim": 4,
        "bandwidth_hz": 250_000.0,
        "center_hz": 302_000.0,
    }
    rs_p: dict[str, ParamValue] = {"interpolation": 125, "decimation": 256}

    modem = ModemSpec(
        symbol_rate=125_000.0 / (1 << sf),
        path=[
            ModemStep(conv="invert", params={}),
            ModemStep(conv="channelize", params=chan_p),
            ModemStep(conv="resample", params=rs_p),
            ModemStep(conv="chirp_sync", params=p),
            ModemStep(conv="dechirp", params=p),
        ],
    )
    snk = tmp_path / "flinders_syms.s16"
    result = be.run_pipeline(_compile(modem, 2_048_000.0, src, snk))
    assert result.status == "ok", result

    symbols = np.fromfile(snk, dtype=np.int16)
    assert len(symbols) >= 20, f"too few symbols: {len(symbols)}"
    assert symbols[:20].tolist() == _FLINDERS_ORACLE, (
        f"oracle mismatch:\n  got:    {symbols[:20].tolist()}\n"
        f"  expect: {_FLINDERS_ORACLE}"
    )


@pytest.mark.skipif(not _LORA_SF7_BIN.exists(), reason="LoRa SF7 asset absent")
def test_css_offair_lora_sf7_upchirp(tmp_path: Path) -> None:
    """Standard LoRa SF7 UP-chirp off-air capture: first 20 symbols match oracle."""
    ensure_worker_warm()
    be = GnuRadioBackend()

    # Asset prep: complex int16 binary, first 20 s -> complex64
    rate = 1_000_000.0
    n = int(rate * 20)
    raw = (
        np.fromfile(_LORA_SF7_BIN, dtype="<i2", count=2 * n).astype(np.float32)
        / 32768.0
    )
    x = (raw[0::2] + 1j * raw[1::2]).astype(np.complex64)
    src = tmp_path / "lora_sf7.cf32"
    x.tofile(src)

    sf, os_, zp, pl = 7, 2, 4, 8
    p: dict[str, ParamValue] = {
        "sf": sf,
        "oversample": os_,
        "zero_pad": zp,
        "preamble_len": pl,
    }
    chan_p: dict[str, ParamValue] = {
        "decim": 4,
        "bandwidth_hz": 200_000.0,
        "center_hz": 0.0,
    }

    modem = ModemSpec(
        symbol_rate=125_000.0 / (1 << sf),
        path=[
            ModemStep(conv="channelize", params=chan_p),
            ModemStep(conv="chirp_sync", params=p),
            ModemStep(conv="dechirp", params=p),
        ],
    )
    snk = tmp_path / "lora_sf7_syms.s16"
    result = be.run_pipeline(_compile(modem, rate, src, snk))
    assert result.status == "ok", result

    symbols = np.fromfile(snk, dtype=np.int16)
    assert len(symbols) >= 20, f"too few symbols: {len(symbols)}"
    assert symbols[:20].tolist() == _LORA_SF7_ORACLE, (
        f"oracle mismatch:\n  got:    {symbols[:20].tolist()}\n"
        f"  expect: {_LORA_SF7_ORACLE}"
    )
