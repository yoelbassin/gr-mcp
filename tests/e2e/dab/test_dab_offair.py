"""Real off-air DAB Mode I, one Modem spanning phy through the coding tail,
CRC as the oracle, plus the readable ensemble label. Known-good on this slice:
192 CRC-valid FIBs, measured 10x with byte-identical sink output (variance 0;
issue 05). The gate is known-good minus 2 — margin only for cross-machine float
drift, tight enough that the observed regression classes (issue 18's trim
192 -> 12; a lost tail frame) cannot pass green.

The PHY (ofdm_demod/dqpsk_soft_demap/deinterleave/depuncture/fec) and the
energy-dispersal descramble + 256-bit FIB segmentation compose in a single
Modem — descramble/segment are now product coding stages (family=coding),
the same GR-chain-then-coding-tail composition as the POCSAG/LoRa/BLE gates.
CRC-16 and the FIG-1/0 ensemble-label parse are test-side helpers
(helpers.crc.crc_check, _parse_ensemble) over run_rx's per-window carve.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from helpers import bitops, crc, framing

from marconi.engine.backends.gnuradio.runner import ensure_worker_warm
from marconi.engine.coding.stages_bits import DescrambleStep, SegmentStep
from marconi.engine.io.bitfile import read_bits
from marconi.engine.io.source import SourceSlice
from marconi.engine.modulation.coding.stages import (
    DeinterleaveStep,
    DepunctureStep,
    FecStep,
)
from marconi.engine.modulation.ofdm.stages import DqpskSoftDemapStep, OfdmDemodStep
from marconi.engine.run import run_rx
from marconi.engine.stages.registry import stage_registry
from marconi.engine.types.descriptor import Descriptor
from marconi.engine.types.enums import ItemType
from marconi.engine.types.levels import Level
from marconi.engine.types.models import Modem

IQ = Descriptor(Level.IQ, ItemType.C)
RATE = 2_048_000.0
_SLICE = (
    Path(__file__).resolve().parents[2]
    / "artifacts"
    / "assets"
    / "DAB"
    / "bbc_slice.cf32"
)

# DAB Mode I OFDM/FIC constants (caller data — a protocol lives in the fixture,
# not production): 2048-point FFT, 1536 active carriers, 3 data symbols per
# frame decoded here, mother-code K=7 rate-1/4 with the PI16/PI15/PIX puncture
# family, and the 9-bit energy-dispersal PRBS.
_FFT = 2048
_NC, _DS = 1536, 3
_FRAME_BODY_LEN = 256


def _carrier_bins() -> list[int]:
    tmp = [0]
    for _ in range(1, _FFT):
        tmp.append((13 * tmp[-1] + 511) % _FFT)
    off = [x - 1024 for x in tmp if x != 1024 and 256 <= x <= 1792]
    return [d if d > 0 else _FFT + d for d in off]


def _bin_perm() -> list[int]:
    bins = _carrier_bins()
    return bins + [b for b in range(_FFT) if b not in set(bins)]


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
    mask: list[int] = []
    for _ in range(21):
        mask += [pi16[k % 32] for k in range(128)]
    for _ in range(3):
        mask += [pi15[k % 32] for k in range(128)]
    mask += pix
    assert len(mask) == 3096 and sum(mask) == 2304
    return mask


def _prbs_hex() -> str:
    sr = [1] * 9
    prbs = []
    for _ in range(768):
        b = sr[8] ^ sr[4]
        prbs.append(b)
        sr = [b] + sr[:8]
    return np.packbits(np.array(prbs, np.uint8)).tobytes().hex()


def _dab_modem() -> Modem:
    return Modem(
        name="dab_rx",
        symbol_rate=float(2_048_000 / 2552),
        path=[
            OfdmDemodStep(
                fft_len=_FFT,
                cp_len=504,
                sym_len=2552,
                null_len=2656,
                frame_len=196608,
                data_syms=_DS,
                n_carriers=_NC,
                bin_perm=_bin_perm(),
            ),
            DqpskSoftDemapStep(
                data_syms=_DS,
                n_carriers=_NC,
                # DAB's D-QPSK bit mapping (bit 1 <-> negative coordinate,
                # first bit on Q) complements GR's stock qpsk constellation:
                # protocol mapping is caller data, declared as explicit points
                # whose index is the bit pattern (MSB-first)
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
            DescrambleStep(sequence=_prbs_hex()),
            SegmentStep(frame_body_len=_FRAME_BODY_LEN),
        ],
    )


def _parse_ensemble(payload30: bytes) -> str | None:
    i = 0
    while i < 30:
        h = payload30[i]
        if h == 0xFF:
            break
        typ, length = h >> 5, h & 0x1F
        fig = payload30[i + 1 : i + 1 + length]
        i += 1 + length
        if (
            typ == 1 and len(fig) >= 21 and (fig[0] & 0x07) == 0
        ):  # FIG 1/0 ensemble label
            return fig[3:19].decode("latin-1", "replace").strip()
    return None


@pytest.mark.skipif(
    not _SLICE.exists(), reason="DAB slice absent — run tests/e2e/dab/make_dab_slice.py"
)
def test_dab_offair(tmp_path: Path) -> None:
    ensure_worker_warm()
    res = run_rx(
        _dab_modem(),
        stage_registry(),
        sample_rate=RATE,
        start=IQ,
        workdir=tmp_path,
        source=SourceSlice(path=_SLICE),
    )
    assert res.status == "ok", res
    assert res.windows, "no DAB FIBs segmented"
    assert res.bitstream is not None
    bits = read_bits(res.bitstream.path)

    ok_bodies: list[bytes] = []
    for window in framing.carve_fixed(bits, res.windows, _FRAME_BODY_LEN):
        ok, body = crc.crc_check(
            bitops.bits_to_bytes(window),
            poly=0x1021,
            bits=16,
            init=0xFFFF,
            xorout=0xFFFF,
        )
        if ok:
            ok_bodies.append(body)
    num_ok = len(ok_bodies)
    assert num_ok >= 190, (
        "expected >=190 CRC-valid FIBs (known-good 192, 10x var 0), " f"got {num_ok}"
    )
    labels = {_parse_ensemble(body) for body in ok_bodies}
    assert "BBC National DAB" in labels, labels
