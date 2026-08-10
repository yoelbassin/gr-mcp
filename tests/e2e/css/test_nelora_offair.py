"""NELoRa-Bench off-air gates: real USRP N210 LoRa symbol captures with
per-symbol filename ground truth (see artifacts/assets/NELoRa/README.md).

The captures are concatenated symbol windows aligned from offset 0 — a bare
dechirp from sample 0 is the whole receive chain. Capture facts (LoRa SF7/SF10,
125 kHz bandwidth, 1 Msps => oversample 8) are the asset's SigMF ground truth
and live only in this e2e.
"""

from __future__ import annotations

from pathlib import Path
from typing import cast

import numpy as np
import pytest

from marconi.engine.backends.gnuradio.runner import ensure_worker_warm
from marconi.engine.io.source import SourceSlice
from marconi.engine.modulation.css.stages import DechirpStep
from marconi.engine.run import PipelineResult, run_rx
from marconi.engine.stages.registry import stage_registry
from marconi.engine.types.descriptor import Descriptor
from marconi.engine.types.enums import ItemType
from marconi.engine.types.levels import Level
from marconi.engine.types.models import Modem
from marconi.mcp.tools import survey

IQ = Descriptor(Level.IQ, ItemType.C)

_ASSET_ROOT = Path(__file__).resolve().parents[3] / "artifacts" / "assets" / "NELoRa"
_SF7 = _ASSET_ROOT / "sf7_pkt0.cf32"
_SF10 = _ASSET_ROOT / "sf10_pkt0.cf32"
_FS = 1_000_000.0
_OSR = 8
_SF7_RATE = 125_000.0 / (1 << 7)  # 976.5625 Hz — below the old fs/1000 floor


def _labels(slice_dir: Path) -> list[int]:
    # filename convention {position}_{symbol_value}_{x}_{SF}.mat; the
    # concatenated .cf32 is these slices in position order (README pins it)
    keyed = sorted(
        (int(f.name.split("_")[0]), int(f.name.split("_")[1]))
        for f in slice_dir.glob("*.mat")
    )
    return [value for _, value in keyed]


def _decode(tmp_path: Path, capture: Path, sf: int) -> PipelineResult:
    ensure_worker_warm()
    return run_rx(
        Modem(
            symbol_rate=125_000.0 / (1 << sf),
            path=[DechirpStep(sf=sf, oversample=_OSR, zero_pad=1)],
        ),
        stage_registry(),
        sample_rate=_FS,
        start=IQ,
        workdir=tmp_path,
        source=SourceSlice(path=capture),
        timeout=60.0,
    )


@pytest.mark.skipif(not _SF7.exists(), reason="NELoRa asset absent")
def test_sf7_bare_dechirp_reads_decoded_and_matches_labels(tmp_path: Path) -> None:
    # the dogfood gap this pins: a hard-symbol css terminal had no checkable
    # evidence and could never earn "decoded"; peak dominance is that evidence
    res = _decode(tmp_path, _SF7, sf=7)
    assert res.quality is not None
    assert res.quality.verdict == "decoded", res.quality
    assert "peak_dominance" in [e.metric for e in res.quality.evidence]
    labels = _labels(_SF7.with_suffix(""))
    assert res.symbolstream is not None
    symbols = np.fromfile(res.symbolstream.path, dtype=np.int16)
    assert len(symbols) == len(labels) == 78
    # 77/78: the final received symbol genuinely peaks one bin off its
    # transmit-truth label (the dataset's own dechirp validation matches
    # 77/78 too — see the asset README)
    matches = int(np.sum(symbols == np.asarray(labels)))
    assert matches >= 77, matches


@pytest.mark.skipif(not _SF10.exists(), reason="NELoRa asset absent")
def test_sf10_bare_dechirp_reads_decoded_and_matches_labels(tmp_path: Path) -> None:
    res = _decode(tmp_path, _SF10, sf=10)
    assert res.quality is not None
    assert res.quality.verdict == "decoded", res.quality
    labels = _labels(_SF10.with_suffix(""))
    assert res.symbolstream is not None
    symbols = np.fromfile(res.symbolstream.path, dtype=np.int16)
    assert len(symbols) == len(labels) == 62
    # 61/62: position 9's filename label is wrong (labeled 0, the slice
    # itself decodes to the second sync-word symbol; the dataset's own
    # validation also reads 61/62)
    matches = int(np.sum(symbols == np.asarray(labels)))
    assert matches >= 61, matches


@pytest.mark.skipif(not _SF7.exists(), reason="NELoRa asset absent")
@pytest.mark.parametrize("sf", [6, 8])
def test_adjacent_sf_reads_no_signal(tmp_path: Path, sf: int) -> None:
    # the hardest wrong-sf confusion: same 125 kHz bandwidth, so at sf+1 the
    # reference slope matches EXACTLY and each window just holds two symbols
    # (two half-energy tones) — dominance collapses either way, making the
    # verdict a clean discriminator for an agent sweeping SF hypotheses
    res = _decode(tmp_path, _SF7, sf=sf)
    assert res.quality is not None
    assert res.quality.verdict == "no_signal", res.quality


@pytest.mark.skipif(not _SF7.exists(), reason="NELoRa asset absent")
def test_survey_default_floor_reaches_sf7_symbol_rate() -> None:
    # the regression this gate pins: a fixed fs/1000 floor (1000 Hz) hid the
    # 976.56 Hz line entirely; the span-derived default must search below it
    # and surface a candidate within the clock spectrum's own quantization
    out = survey(str(_SF7), _FS)
    sr = cast(dict[str, object], out["symbol_rate"])
    assert cast(float, sr["search_lo_hz"]) < _SF7_RATE
    res = cast(float, sr["clock_resolution_hz"])
    cands = cast(list[float], sr["candidates_hz"])
    assert min(abs(c - _SF7_RATE) for c in cands) < 2.0 * res, (cands, res)
