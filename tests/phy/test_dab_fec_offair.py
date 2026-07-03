# tests/phy/test_dab_fec_offair.py
import sys
from pathlib import Path

import numpy as np
import pytest

from marconi.core.bitfile import read_bits
from marconi.core.descriptor import Descriptor
from marconi.core.levels import Level
from marconi.phy.backends.gnuradio.runner import GnuRadioBackend, ensure_worker_warm
from marconi.phy.compiler import compile_modem
from marconi.phy.models import ModemSpec, ModemStep
from marconi.phy.stages import stage_registry

sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_dab_ofdm_offair import _bin_perm  # noqa: E402

_SLICE = (
    Path(__file__).resolve().parents[1]
    / "artifacts"
    / "assets"
    / "DAB"
    / "bbc_slice.cf32"
)
_NC, _DS = 1536, 3


def _regroup():
    return [c * 2 + 1 for c in range(_NC)] + [c * 2 for c in range(_NC)]


def _keep_mask():
    def rep(pat, n):
        out = []
        while len(out) < n:
            out += pat
        return out[:n]

    pi16, pi15, pix = (
        rep([1, 1, 1, 0], 32),
        rep([1, 1, 1, 0], 28) + [1, 1, 0, 0],
        rep([1, 1, 0, 0], 24),
    )
    mask = []
    for _ in range(21):
        mask += [pi16[k % 32] for k in range(128)]
    for _ in range(3):
        mask += [pi15[k % 32] for k in range(128)]
    mask += pix
    assert len(mask) == 3096 and sum(mask) == 2304
    return mask


def _prbs():
    sr = [1] * 9
    out = []
    for _ in range(768):
        b = sr[8] ^ sr[4]
        out.append(b)
        sr = [b] + sr[:8]
    return np.array(out, np.uint8)


def _fib_crc_ok(fib32):
    crc = 0xFFFF
    for byte in fib32[:30]:
        crc ^= int(byte) << 8
        for _ in range(8):
            crc = (
                ((crc << 1) ^ 0x1021) & 0xFFFF if crc & 0x8000 else (crc << 1) & 0xFFFF
            )
    crc ^= 0xFFFF
    return crc == ((int(fib32[30]) << 8) | int(fib32[31]))


def _dab_phy_steps():
    return [
        ModemStep(
            conv="ofdm_demod",
            params={
                "fft_len": 2048,
                "cp_len": 504,
                "sym_len": 2552,
                "null_len": 2656,
                "frame_len": 196608,
                "n_frame_syms": 76,
                "data_syms": _DS,
                "n_carriers": _NC,
                "bin_perm": _bin_perm(),
            },
        ),
        ModemStep(
            conv="dqpsk_soft_demap",
            params={"data_syms": _DS, "n_carriers": _NC, "scheme": "psk", "order": 4},
        ),
        ModemStep(conv="deinterleave", params={"perm": _regroup()}),
        ModemStep(conv="depuncture", params={"keep_mask": _keep_mask()}),
        ModemStep(
            conv="fec",
            params={
                "scheme": "cc",
                "rate_inv": 4,
                "polys": [0o133, 0o171, 0o145, 0o133],
                "frame_bits": 768,
                "tail": 6,
            },
        ),
    ]


@pytest.mark.skipif(
    not _SLICE.exists(), reason="DAB slice absent — run tests/phy/make_dab_slice.py"
)
def test_dab_phy_decodes_crc_valid_fibs(tmp_path):
    ensure_worker_warm()
    snk = tmp_path / "fic.u8"
    modem = ModemSpec(
        name="dab_fec", symbol_rate=float(2_048_000 / 2552), path=_dab_phy_steps()
    )
    pipe = compile_modem(
        modem,
        stage_registry(),
        direction="rx",
        sample_rate=2_048_000.0,
        start=Descriptor(Level.IQ, "c"),
        source_io={"path": str(_SLICE)},
        sink_io={"path": str(snk)},
    )
    r = GnuRadioBackend().run_pipeline(pipe, timeout=180.0)
    assert r.status == "ok", r
    bits = read_bits(snk)
    prbs = _prbs()
    ok = 0
    for cif in range(bits.size // 768):
        dec = bits[cif * 768 : (cif + 1) * 768] ^ prbs
        for f in range(3):
            fib = np.packbits(dec[f * 256 : (f + 1) * 256])
            if _fib_crc_ok(fib):
                ok += 1
    assert ok >= 190, (
        "expected >=190 CRC-valid FIBs from the phy chain "
        f"(known-good 192, 10x var 0; issue 05), got {ok}"
    )
