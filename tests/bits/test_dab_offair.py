# tests/bits/test_dab_offair.py
"""Real off-air DAB Mode I, end-to-end phy -> bits, CRC as the oracle, plus the
readable ensemble label. Known-good on this slice is 192 CRC-valid FIBs; the gate
is >= 180 (GR demod wobbles run-to-run — never assert exact counts on GR output)
so the observed regression class (issue 18's trim, 192 -> 12) cannot pass green."""
import sys
from pathlib import Path

import pytest

from marconi.bits.registry import registry
from marconi.bits.seam import parse_bitstream
from marconi.core.bitfile import read_bits
from marconi.core.descriptor import Descriptor
from marconi.core.levels import Level
from marconi.core.models import Bitstream
from marconi.phy.backends.gnuradio.runner import GnuRadioBackend, ensure_worker_warm
from marconi.phy.compiler import compile_modem
from marconi.phy.models import ModemSpec
from marconi.phy.stages import stage_registry

sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_dab_codec import _codec  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "phy"))
from test_dab_fec_offair import (  # noqa: E402  (the canonical 5-step phy chain)
    _dab_phy_steps,
)

_SLICE = (
    Path(__file__).resolve().parents[1]
    / "artifacts"
    / "assets"
    / "DAB"
    / "bbc_slice.cf32"
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
    not _SLICE.exists(), reason="DAB slice absent — run tests/phy/make_dab_slice.py"
)
def test_dab_decodes_crc_valid_bbc(tmp_path):
    ensure_worker_warm()
    snk = tmp_path / "fic.u8"
    modem = ModemSpec(
        name="dab_rx", symbol_rate=float(2_048_000 / 2552), path=_dab_phy_steps()
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
    n = int(read_bits(snk).size)
    res = parse_bitstream(
        Bitstream(path=snk, num_bits=n, source_capture=_SLICE), _codec(), registry()
    )
    assert (
        res.num_crc_ok >= 180
    ), f"expected >=180 CRC-valid FIBs (known-good 192), got {res.num_crc_ok}"
    labels = {
        _parse_ensemble(bytes.fromhex(f.payload_hex)) for f in res.frames if f.crc_ok
    }
    assert "BBC National DAB" in labels, labels
