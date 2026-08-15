"""Real off-air DMR quality gate: a bare 4-FSK front end (channelize + fsk, no
sync/coding tail) over the real capture must read as signal-PRESENT in every
burst window. The front end is sliced straight out of test_dmr_offair's
full-decode modem (same channelize/fsk params that gate already validates
end-to-end via CRC), so no deviation or bandwidth literal is duplicated here.

The external truth is the capture's proven content: test_dmr_offair decodes
CRC-valid frames out of this same file, so signal IS present in all of it and
every window must say so on its own — six regressions cannot hide behind one
survivor. BURST_SEGMENTS are the 7 activity windows survey_iq found (offset,
length in complex samples), measured once as fixtures.

What "say so" MEANS is the correction this file needed. It used to assert that
no window read no_signal, and no window can: this path taps soft SYMBOLS,
where _emit_soft drops every negative (a per-symbol eye attests presence,
never absence), and every other producer of a negative needs a stage a
demod-only path does not have — sync_search, validates_words, an OFDM lock, a
dechirp peak. Probed through the real run_rx on this exact modem shape:
gaussian noise, 1e-6 near-silence, a clean synthetic 4-level FM signal and a
3 dB one all return verdict "uncertain", so the old assertion held on every
possible input and could fail on none.

What this path CAN deliver is the positive, which noise cannot fake: each
burst's soft stream fits 4 levels with separation ~5.07-8.12 against the
quality module's 4.0 multilevel bar — the reading recorded when these windows
were chosen, and what this assertion now re-measures on every run instead of
describing. A forced 4-level fit on noise reaches only 3.20 through the same
path, which is where the bar's discrimination comes from. So every window must
carry a soft_eye POSITIVE at or above the bar, and a demod or threshold change
narrowing any single burst fails here. The same reading on synthetic captures
— 4-level signal positive, noise nothing — is gated without the asset by
tests/integration/engine/quality/test_quality_wiring.py."""

from __future__ import annotations

from pathlib import Path

import pytest
from e2e.dmr.test_dmr_offair import _SLICE, RATE
from e2e.dmr.test_dmr_offair import _dmr_modem as _dmr_full_modem

from marconi.engine.backends.gnuradio.runner import ensure_worker_warm
from marconi.engine.io.source import SourceSlice
from marconi.engine.quality import _SOFT_MULTILEVEL_SEPARATION
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
    eyes: list[tuple[int, float | None]] = []
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
        positive = [
            e.value
            for e in res.quality.evidence
            if e.metric == "soft_eye" and e.assessment == "positive"
        ]
        eyes.append((offset, max(positive, default=None)))
    weak = [(off, v) for off, v in eyes if v is None or v < _SOFT_MULTILEVEL_SEPARATION]
    assert not weak, (
        f"bursts whose 4-level eye no longer clears the "
        f"{_SOFT_MULTILEVEL_SEPARATION} bar: {weak} (all: {eyes})"
    )
