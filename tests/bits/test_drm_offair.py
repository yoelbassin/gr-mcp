"""Real off-air DRM (Deutsche Welle, Mode B, spectrum occupancy 3), end-to-end
phy -> bits, CRC-8 as the oracle plus the FAC channel-parameter invariants.
Known-good on this slice: 109/109 CRC-8-valid FAC blocks, every one occupancy
0011, identities cycling 01/10/11 across the super-frame. The gate is 109 minus
margin — enough for cross-machine float drift, far above any partial-decode
regression."""

from pathlib import Path

import pytest
from bits import _drm

from marconi.bits.registry import registry
from marconi.bits.seam import parse_bitstream
from marconi.core.bitfile import read_bits
from marconi.core.descriptor import Descriptor
from marconi.core.levels import Level
from marconi.core.models import Bitstream
from marconi.phy.backends.gnuradio.runner import GnuRadioBackend, ensure_worker_warm
from marconi.phy.compiler import compile_modem
from marconi.phy.models import ModemSpec

_SLICE = (
    Path(__file__).resolve().parents[1]
    / "artifacts"
    / "assets"
    / "DRM"
    / "dw_modeb.cf32"
)


@pytest.mark.skipif(
    not _SLICE.exists(), reason="DRM slice absent — run tests/bits/make_drm_slice.py"
)
def test_drm_fac(tmp_path: Path) -> None:
    ensure_worker_warm()
    snk = tmp_path / "fac.u8"
    modem = ModemSpec(
        name="drm_fac", symbol_rate=_drm.RATE / _drm.SYM_LEN, path=_drm.fac_phy_steps()
    )
    pipe = compile_modem(
        modem,
        _drm.fac_stage_registry(),
        direction="rx",
        sample_rate=_drm.RATE,
        start=Descriptor(Level.IQ, "c"),
        source_io={"path": str(_SLICE)},
        sink_io={"path": str(snk)},
    )
    assert GnuRadioBackend().run_pipeline(pipe, timeout=180.0).status == "ok"
    n = int(read_bits(snk).size)
    res = parse_bitstream(
        Bitstream(path=snk, num_bits=n, source_capture=_SLICE),
        _drm.fac_codec(),
        registry(),
    )
    assert res.num_crc_ok >= 90, (
        f"expected >=90 CRC-8-valid FAC blocks (scratch 109/109), "
        f"got {res.num_crc_ok}"
    )
    fields = [
        _drm.parse_fac(bytes.fromhex(f.payload_hex)) for f in res.frames if f.crc_ok
    ]
    assert all(f["occupancy"] == "0011" for f in fields)  # occ 3 / 10 kHz
    ids = {str(f["identity"]) for f in fields}
    assert {"01", "10", "11"} <= ids  # super-frame identity cycle
