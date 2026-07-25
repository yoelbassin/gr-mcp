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

from marconi.engine.backends.gnuradio.runner import GnuRadioBackend, ensure_worker_warm
from marconi.engine.compile.compiler import compile_modem
from marconi.engine.types.descriptor import Descriptor
from marconi.engine.types.levels import Level
from marconi.engine.types.models import ModemSpec, ModemStep
from marconi.engine.types.params import ParamValue

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

_IQ2_BIN = Path("/Users/joel/Clones/gr-mcp-rebuild/artifacts/captures/IQ_2.sigmf-data")
_IQ12_BIN = Path(
    "/Users/joel/Clones/gr-mcp-rebuild/artifacts/assets/CSS/LoRa/IQ_12/IQ_12.dat"
)

# Real WINeS SF11/BW125/CR4-5/LDRO LoPy4 frames, raw dechirp argmax bins (293).
# The bar is LDRO bin//4 (CRC-equivalent): a real long frame carries
# FEC-correctable +/-1 raw errors that //4 absorbs. IQ_12 is the same message as
# IQ_2 with ~+6 ppm TX clock drift (needs clock_correct).
_IQ2_ORACLE = [
    1813, 1225, 1085, 685, 281, 353, 81, 1445, 1713, 241, 1413, 1449, 573, 1213,
    1869, 345, 1657, 969, 1353, 1077, 785, 117, 1233, 1509, 1837, 1633, 1549, 1317,
    2041, 1309, 389, 281, 817, 1005, 1493, 645, 481, 1237, 1925, 1717, 1837, 189,
    1081, 133, 1529, 1029, 837, 1301, 1605, 1573, 1097, 1265, 1737, 1657, 1629, 837,
    445, 1117, 385, 1617, 857, 1321, 657, 445, 1577, 937, 97, 1541, 1593, 1753, 585,
    1269, 173, 1337, 1825, 581, 577, 517, 833, 977, 93, 217, 1077, 897, 597, 1873,
    1641, 1901, 1505, 361, 1325, 1473, 549, 1757, 237, 545, 881, 321, 325, 2013, 69,
    233, 1141, 581, 1573, 72, 1629, 1756, 965, 729, 589, 1273, 905, 821, 825, 1420,
    552, 1824, 692, 56, 1932, 128, 368, 692, 56, 112, 1580, 984, 1324, 244, 492, 952,
    1556, 1772, 140, 284, 1292, 832, 140, 1100, 152, 340, 1548, 892, 1108, 1876, 1944,
    1188, 864, 916, 1836, 572, 364, 860, 1000, 2000, 1896, 1012, 1336, 216, 1612, 768,
    1700, 1524, 320, 380, 1484, 972, 1032, 700, 640, 1688, 1284, 1848, 800, 1468, 948,
    1952, 312, 1824, 1604, 236, 528, 260, 1880, 328, 1440, 132, 260, 1880, 328, 264,
    1324, 264, 1860, 372, 1056, 1608, 364, 1928, 232, 436, 1016, 336, 2036, 20, 1820,
    156, 1832, 1272, 500, 904, 1620, 1832, 1276, 1540, 1336, 752, 1704, 2044, 2040,
    108, 1952, 1684, 1920, 248, 1728, 708, 144, 1136, 228, 596, 1128, 940, 1524, 2024,
    1816, 1776, 976, 756, 1516, 1088, 824, 44, 756, 1516, 744, 1964, 1584, 1736, 1428,
    1868, 632, 1776, 1868, 664, 512, 220, 1908, 1088, 132, 360, 1384, 1204, 1084, 1924,
    1988, 48, 1200, 972, 1948, 860, 1412, 1920, 592, 1884, 1652, 288, 1540, 1696, 1348,
    1296, 1936, 1332, 1852, 632, 960, 1224, 1640, 716, 56, 1976, 1772,
]  # fmt: skip

_IQ12_ORACLE = [
    1813, 1225, 1085, 685, 281, 353, 81, 1445, 1713, 241, 1413, 1449, 573, 1213,
    1869, 345, 1657, 969, 1353, 1077, 785, 117, 1233, 1509, 1837, 1633, 1549, 1317,
    2041, 1309, 389, 281, 817, 1005, 1493, 645, 481, 1237, 1925, 1717, 1837, 189,
    1081, 133, 1529, 1029, 837, 1301, 1605, 1573, 1097, 1265, 1737, 1657, 1629, 837,
    445, 1117, 385, 1617, 857, 1321, 657, 445, 1577, 937, 97, 1541, 1593, 1753,
    585, 1269, 173, 1337, 1825, 581, 577, 517, 833, 977, 93, 217, 1077, 897,
    597, 1873, 1641, 1901, 1505, 361, 1325, 1473, 549, 1757, 237, 545, 881, 321,
    325, 2013, 69, 233, 1141, 581, 1573, 73, 1629, 1757, 965, 729, 589, 1273,
    905, 821, 825, 1421, 553, 1825, 693, 57, 1933, 129, 369, 693, 58, 113,
    1581, 985, 1325, 245, 493, 953, 1557, 1773, 141, 285, 1293, 833, 142, 1101,
    153, 341, 1549, 893, 1109, 1878, 1946, 1189, 865, 917, 1838, 573, 365, 861,
    1001, 2002, 1898, 1013, 1337, 218, 1613, 769, 1701, 1525, 322, 381, 1485, 973,
    1033, 701, 641, 1690, 1285, 1850, 801, 1469, 949, 1954, 314, 1826, 1606, 238,
    529, 262, 1882, 330, 1441, 134, 262, 1882, 330, 266, 1325, 266, 1862, 374,
    1057, 1610, 366, 1930, 234, 438, 1017, 338, 2038, 22, 1822, 158, 1834, 1274,
    502, 906, 1622, 1834, 1278, 1542, 1338, 754, 1706, 2046, 2042, 110, 1954, 1686,
    1922, 250, 1730, 710, 146, 1138, 230, 598, 1130, 942, 1526, 2026, 1818, 1778,
    978, 758, 1518, 1090, 826, 46, 758, 1518, 746, 1966, 1586, 1738, 1430, 1870,
    634, 1778, 1870, 666, 514, 222, 1910, 1090, 134, 362, 1386, 1206, 1086, 1926,
    1990, 50, 1202, 974, 1950, 862, 1414, 1922, 594, 1886, 1654, 290, 1542, 1698,
    1350, 1298, 1938, 1334, 1854, 634, 962, 1226, 1642, 718, 58, 1978, 1774,
]  # fmt: skip


def _compile(modem: ModemSpec, rate: float, src: Path, snk: Path):
    from marconi.engine.stages.registry import stage_registry

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
        "sfd_symbols": 2.25,
        "sync_symbols": 2,
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
    # //4 (CRC-equivalent) bar: the joint CFO estimate lands Flinders' knife-edge
    # carrier-offset bin on the robust side, shifting a few marginal symbols by
    # +/-1 vs the oracle -- absorbed by //4, the bar the real SF11 captures use.
    assert [s // 4 for s in symbols[:20].tolist()] == [
        o // 4 for o in _FLINDERS_ORACLE
    ], (
        f"//4 oracle mismatch:\n  got:    {[s // 4 for s in symbols[:20].tolist()]}\n"
        f"  expect: {[o // 4 for o in _FLINDERS_ORACLE]}"
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
        "sfd_symbols": 2.25,
        "sync_symbols": 2,
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


@pytest.mark.skipif(not _IQ2_BIN.exists(), reason="IQ_2 asset absent")
def test_css_offair_iq2_sf11(tmp_path: Path) -> None:
    """Real SF11 LoRa (no SFO): production sync already recovers sub-sample timing
    -> bin//4 (CRC-equivalent) oracle exact over the full 293-symbol frame."""
    ensure_worker_warm()
    be = GnuRadioBackend()
    rate = 1_000_000.0
    x = np.fromfile(_IQ2_BIN, dtype=np.complex64, count=int(rate * 11))
    src = tmp_path / "iq2.cf32"
    x.tofile(src)
    p: dict[str, ParamValue] = {
        "sf": 11,
        "oversample": 2,
        "zero_pad": 10,
        "preamble_len": 8,
        "sfd_symbols": 2.25,
        "sync_symbols": 2,
    }
    modem = ModemSpec(
        symbol_rate=125_000.0 / 2048,
        path=[
            ModemStep(conv="resample", params={"interpolation": 2, "decimation": 8}),
            ModemStep(conv="chirp_sync", params=p),
            ModemStep(conv="dechirp", params=p),
        ],
    )
    snk = tmp_path / "iq2.s16"
    assert be.run_pipeline(_compile(modem, rate, src, snk)).status == "ok"
    syms = np.fromfile(snk, dtype=np.int16)
    assert len(syms) >= 293
    assert [int(s) // 4 for s in syms[:293]] == [o // 4 for o in _IQ2_ORACLE]


@pytest.mark.skipif(not _IQ12_BIN.exists(), reason="IQ_12 asset absent")
def test_css_offair_iq12_sf11_sfo(tmp_path: Path) -> None:
    """The trophy: long SF11 LoRa with ~+6 ppm TX clock drift. clock_correct(6)
    holds the timing to the tail -> bin//4 oracle exact over 293 symbols; ppm=0
    diverges in the tail."""
    ensure_worker_warm()
    be = GnuRadioBackend()
    rate = 1_000_000.0
    x = np.fromfile(
        _IQ12_BIN, dtype=np.complex64, count=5_700_000, offset=4_300_000 * 8
    )
    src = tmp_path / "iq12.cf32"
    x.tofile(src)
    p: dict[str, ParamValue] = {
        "sf": 11,
        "oversample": 2,
        "zero_pad": 10,
        "preamble_len": 8,
        "sfd_symbols": 2.25,
        "sync_symbols": 2,
    }

    def run(ppm: float, name: str) -> np.ndarray:
        path = [
            ModemStep(conv="resample", params={"interpolation": 2, "decimation": 8})
        ]
        if ppm:
            path.append(ModemStep(conv="clock_correct", params={"ppm": ppm}))
        path += [
            ModemStep(conv="chirp_sync", params=p),
            ModemStep(conv="dechirp", params=p),
        ]
        snk = tmp_path / name
        assert (
            be.run_pipeline(
                _compile(
                    ModemSpec(symbol_rate=125_000.0 / 2048, path=path), rate, src, snk
                )
            ).status
            == "ok"
        )
        return np.fromfile(snk, dtype=np.int16)

    corrected = run(6.0, "iq12_cc.s16")
    assert len(corrected) >= 293
    assert [int(s) // 4 for s in corrected[:293]] == [o // 4 for o in _IQ12_ORACLE]

    uncorrected = run(0.0, "iq12_raw.s16")
    n = min(293, len(uncorrected))
    errs = sum(int(uncorrected[i]) // 4 != _IQ12_ORACLE[i] // 4 for i in range(n))
    assert (
        errs > 20
    ), f"SFO must break the tail without clock_correct (got {errs} //4 errors)"
