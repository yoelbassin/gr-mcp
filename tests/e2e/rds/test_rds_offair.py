"""Real off-air FM broadcast -> RDS data bits: the coherent PSK lane's first
off-air gate, and the FM-subcarrier demod-then-demod composition in one
product path. Ten registered generic stages, zero product edits:

  fm_demod -> analytic -> channelize -> resample -> agc -> psk_demod ->
  psk_demap  (GR: discriminator, 57 kHz subcarrier extraction, RRC + Gardner +
  costas BPSK, hard bits)  then  realign -> codebook -> differential  (coding:
  biphase pair decode + differential decode).

The biphase pair phase is unknowable a priori (Gardner picks an arbitrary
symbol parity), so the GR half runs once to hard bits and the coding tail runs
per pair offset via input_stream; the checkword oracle picks the winner.

Oracle: RDS 26-bit blocks carry a 10-bit checkword (g(x) = x^10+x^8+x^7+x^5+
x^4+x^3+1 plus a per-block offset word) — group-synced blocks whose syndrome
matches count as valid, and groups of type 0A/0B carry the station's 8-char
Program Service name. The capture (PySDR's fm_rds_250k_1Msamples.iq, 4 s @
250 ksps) decodes 180 valid blocks with PS "Upliftin" — matched independently
by a numpy reference decode. Checkword math, offset words, and PS field
layout are protocol-datasheet work and live in tests/helpers/rds.py, not in
the product.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from helpers import rds

from marconi.engine.backends.gnuradio.runner import ensure_worker_warm
from marconi.engine.io.bitfile import read_bits
from marconi.engine.run import run_rx
from marconi.engine.stages.registry import stage_registry
from marconi.engine.types.descriptor import Descriptor
from marconi.engine.types.enums import ItemType
from marconi.engine.types.levels import Level
from marconi.engine.types.models import Bitstream

IQ = Descriptor(Level.IQ, ItemType.C)
BITS = Descriptor(Level.BITS, ItemType.B)
RATE = 250_000.0
_CAPTURE = (
    Path(__file__).resolve().parents[3]
    / "artifacts"
    / "assets"
    / "RDS"
    / "fm_rds_250k_1Msamples.iq"
)

MIN_VALID_BLOCKS = 120  # measured 180; margin for scheduler nondeterminism
STATION_PS = "Upliftin"


@pytest.mark.skipif(
    not _CAPTURE.exists(),
    reason="RDS capture absent — run tests/e2e/rds/make_rds_asset.py",
)
def test_rds_offair(tmp_path: Path) -> None:
    ensure_worker_warm()
    res = run_rx(
        rds.phy_modem(),
        stage_registry(),
        sample_rate=RATE,
        start=IQ,
        workdir=tmp_path,
        source_io={"path": str(_CAPTURE)},
    )
    assert res.status == "ok", res
    assert res.bitstream is not None
    n_symbols = res.bitstream.num_bits
    assert n_symbols > 4000, f"only {n_symbols} biphase symbols demodulated"

    results = []
    for off in (0, 1):
        res2 = run_rx(
            rds.codec_modem(off),
            stage_registry(),
            sample_rate=1.0,
            start=BITS,
            workdir=tmp_path,
            input_stream=Bitstream(path=res.bitstream.path, num_bits=n_symbols),
        )
        assert res2.status == "ok", res2
        assert res2.bitstream is not None
        results.append(rds.decode_groups(read_bits(res2.bitstream.path)))

    n_valid, ps = max(results)
    assert n_valid >= MIN_VALID_BLOCKS, f"{n_valid} checkword-valid blocks: {results}"
    assert ps == STATION_PS, f"PS mismatch: {results}"
