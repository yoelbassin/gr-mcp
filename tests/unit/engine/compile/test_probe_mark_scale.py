"""Burst marks are recorded in item units at the probe's own position, and a
later GR stage can change those units. There are two exits and they need
different answers:

- at the CODING SEAM the marks are CONSUMED to seed windows, so wrong units
  silently decode the wrong spans — that is a compile error.
- at the OUTPUT boundary they are only REPORTED, so the composition stays legal
  (dechirp -> burst_probe -> css_demap wants the symbol marks AND the bits) and
  the run withholds them rather than indexing a stream they do not index.

Gated on the seam alone, the identical misplacement compiled clean all-GR and
the marks reached the agent mis-scaled.
"""

from __future__ import annotations

import pytest

from marconi.engine.coding.stages_bits import DescrambleStep
from marconi.engine.compile.compiler import CompiledPipeline, compile_pipeline
from marconi.engine.compile.errors import CompileError
from marconi.engine.modulation.css.stages import CssDemapStep, DechirpStep
from marconi.engine.stages.probes import BurstProbeStep
from marconi.engine.stages.registry import stage_registry
from marconi.engine.types.descriptor import Descriptor
from marconi.engine.types.enums import ItemType
from marconi.engine.types.levels import Level
from marconi.engine.types.models import Modem
from marconi.engine.types.step import Step

_SF, _OS = 7, 2
_RATE = 2.0 * (1 << _SF)
IQ = Descriptor(Level.IQ, ItemType.C)


def _compile(path: list[Step]) -> CompiledPipeline:
    return compile_pipeline(
        Modem(symbol_rate=1.0, path=path),
        stage_registry(),
        sample_rate=_RATE,
        start=IQ,
        source_io={"path": "in.cf32"},
        sink_io={"path": "seam.dat"},
    )


def _dechirp() -> Step:
    return DechirpStep(sf=_SF, oversample=_OS, zero_pad=1)


def test_marks_that_do_not_index_the_output_are_flagged() -> None:
    cp = _compile([_dechirp(), BurstProbeStep(), CssDemapStep(sf=_SF)])
    assert cp.unscaled_probe_marks == ("burst_probe[1]",)


def test_marks_that_do_index_the_output_are_not_flagged() -> None:
    cp = _compile([_dechirp(), BurstProbeStep()])
    assert cp.unscaled_probe_marks == ()


def test_the_same_misplacement_before_a_coding_seam_is_still_fatal() -> None:
    """The seam consumes them, so there is nothing safe to report."""
    with pytest.raises(CompileError, match="misread"):
        _compile(
            [
                _dechirp(),
                BurstProbeStep(),
                CssDemapStep(sf=_SF),
                DescrambleStep(sequence="ff"),
            ]
        )
