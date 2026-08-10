# tests/e2e/dab/test_dab_fec_offair.py
from pathlib import Path

import numpy as np
import numpy.typing as npt
import pytest
from e2e.dab.test_dab_ofdm_offair import _bin_perm

from marconi.engine.backends.gnuradio.runner import GnuRadioBackend, ensure_worker_warm
from marconi.engine.compile.compiler import compile_modem
from marconi.engine.io.bitfile import read_bits
from marconi.engine.modulation.coding.stages import (
    DeinterleaveStep,
    DepunctureStep,
    FecStep,
)
from marconi.engine.modulation.ofdm.stages import DqpskSoftDemapStep, OfdmDemodStep
from marconi.engine.stages.registry import stage_registry
from marconi.engine.types.descriptor import Descriptor
from marconi.engine.types.enums import ItemType
from marconi.engine.types.levels import Level
from marconi.engine.types.models import Modem
from marconi.engine.types.step import Step

_SLICE = (
    Path(__file__).resolve().parents[2]
    / "artifacts"
    / "assets"
    / "DAB"
    / "bbc_slice.cf32"
)
_NC, _DS = 1536, 3


def _regroup() -> list[int]:
    return [c * 2 + 1 for c in range(_NC)] + [c * 2 for c in range(_NC)]


def _keep_mask() -> list[int]:
    def rep(pat: list[int], n: int) -> list[int]:
        out: list[int] = []
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


def _prbs() -> npt.NDArray[np.uint8]:
    sr = [1] * 9
    out: list[int] = []
    for _ in range(768):
        b = sr[8] ^ sr[4]
        out.append(b)
        sr = [b] + sr[:8]
    return np.array(out, np.uint8)


def _fib_crc_ok(fib32: npt.NDArray[np.uint8]) -> bool:
    crc = 0xFFFF
    for byte in fib32[:30]:
        crc ^= int(byte) << 8
        for _ in range(8):
            crc = (
                ((crc << 1) ^ 0x1021) & 0xFFFF if crc & 0x8000 else (crc << 1) & 0xFFFF
            )
    crc ^= 0xFFFF
    return crc == ((int(fib32[30]) << 8) | int(fib32[31]))


def _dab_phy_steps() -> list[Step]:
    return [
        OfdmDemodStep(
            fft_len=2048,
            cp_len=504,
            sym_len=2552,
            null_len=2656,
            frame_len=196608,
            data_syms=_DS,
            n_carriers=_NC,
            bin_perm=_bin_perm(),
        ),
        # DAB's D-QPSK mapping complements GR's stock qpsk constellation —
        # declared as explicit points (index = bit pattern, MSB-first)
        DqpskSoftDemapStep(
            data_syms=_DS,
            n_carriers=_NC,
            scheme="explicit",
            points_i=[x / np.sqrt(2) for x in (1.0, -1.0, 1.0, -1.0)],
            points_q=[x / np.sqrt(2) for x in (1.0, 1.0, -1.0, -1.0)],
        ),
        DeinterleaveStep(perm=_regroup()),
        DepunctureStep(keep_mask=_keep_mask()),
        FecStep(
            scheme="cc",
            rate_inv=4,
            polys=[0o133, 0o171, 0o145, 0o133],
            frame_bits=768,
            tail=6,
        ),
    ]


@pytest.mark.skipif(
    not _SLICE.exists(), reason="DAB slice absent — run tests/e2e/dab/make_dab_slice.py"
)
def test_dab_phy_decodes_crc_valid_fibs(tmp_path: Path) -> None:
    ensure_worker_warm()
    snk = tmp_path / "fic.u8"
    modem = Modem(
        name="dab_fec", symbol_rate=float(2_048_000 / 2552), path=_dab_phy_steps()
    )
    pipe = compile_modem(
        modem,
        stage_registry(),
        direction="rx",
        sample_rate=2_048_000.0,
        start=Descriptor(Level.IQ, ItemType.C),
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
        f"(known-good 192, 10x var 0), got {ok}"
    )
