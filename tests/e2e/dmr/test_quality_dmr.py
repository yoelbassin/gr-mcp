"""Real off-air DMR quality-verdict gate: a bare 4-FSK front end (channelize
+ fsk, no sync/coding tail) over the real capture reads as signal-present
under run_rx's M-ary soft-stream quality branch. The front end is sliced
straight out of test_dmr_offair's full-decode modem (same channelize/fsk
params that gate already validates end-to-end via CRC), so no deviation or
bandwidth literal is duplicated here.

BURST_SEGMENTS are the 7 activity windows survey_iq's own burst detector
finds in this capture (offset, length in complex samples) — measured once,
not re-derived per run. Every burst's soft-stream fit_levels order is 4
(genuine 4-level separation), with separation ~5.07-8.12 against the
quality module's 4.0 multilevel bar — all 7 currently clear it with
margin. The gate still checks all 7 and asserts any non-no_signal verdict,
rather than betting on one hardcoded window, as insurance against a future
demod or threshold change narrowing any single burst's margin."""

from __future__ import annotations

from pathlib import Path

import pytest
from e2e.dmr.test_dmr_offair import _SLICE, RATE
from e2e.dmr.test_dmr_offair import _dmr_modem as _dmr_full_modem

from marconi.engine.backends.gnuradio.runner import ensure_worker_warm
from marconi.engine.io.source import SourceSlice
from marconi.engine.run import run_rx
from marconi.engine.stages.registry import stage_registry
from marconi.engine.types.descriptor import Descriptor
from marconi.engine.types.enums import ItemType
from marconi.engine.types.levels import Level
from marconi.engine.types.models import Modem

IQ = Descriptor(Level.IQ, ItemType.C)

BURST_SEGMENTS = [
    (0, 91338),
    (161129, 117712),
    (366208, 134042),
    (560742, 134042),
    (757617, 134041),
    (952151, 134043),
    (1149028, 138732),
]


def _bare_fsk_modem() -> Modem:
    full = _dmr_full_modem()
    return Modem(symbol_rate=full.symbol_rate, path=full.path[:2])


@pytest.mark.skipif(
    not _SLICE.exists(),
    reason="DMR slice absent — run tests/e2e/dmr/make_dmr_slice.py",
)
def test_bare_fsk_reads_as_signal(tmp_path: Path) -> None:
    ensure_worker_warm()
    modem = _bare_fsk_modem()
    verdicts: list[str] = []
    for offset, length in BURST_SEGMENTS:
        res = run_rx(
            modem,
            stage_registry(),
            sample_rate=RATE,
            start=IQ,
            workdir=tmp_path,
            source=SourceSlice(path=_SLICE, offset=offset, length=length),
            timeout=60.0,
        )
        assert res.quality is not None
        verdicts.append(res.quality.verdict)
    assert any(v != "no_signal" for v in verdicts), verdicts
