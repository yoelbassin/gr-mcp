from typing import Any

import pytest

from marconi.core.descriptor import Amplitude, Carrier, Descriptor, Layout
from marconi.core.levels import Level
from marconi.phy.stages import stage_registry

_NORMALIZED = Descriptor(
    Level.IQ, "c", Layout.STREAM, Carrier.HARD, Amplitude.NORMALIZED
)

_PARAMS: dict[str, dict[str, Any]] = {
    "channelize": {"decim": 1, "bandwidth_hz": 1.0},
    "resample": {"interpolation": 1, "decimation": 1},
    "clock_correct": {"ppm": 1.0},
}


@pytest.mark.parametrize("name", sorted(_PARAMS))
def test_scale_changing_stages_invalidate_amplitude(name: str) -> None:
    stage = stage_registry()[name]
    out = stage.out_descriptor(_NORMALIZED, _PARAMS[name])
    assert out.amplitude is Amplitude.UNKNOWN


def test_invert_preserves_amplitude() -> None:
    out = stage_registry()["invert"].out_descriptor(_NORMALIZED, {})
    assert out.amplitude is Amplitude.NORMALIZED


def test_analytic_yields_unknown_amplitude() -> None:
    audio = Descriptor(Level.AUDIO, "f", Layout.STREAM, Carrier.HARD)
    out = stage_registry()["analytic"].out_descriptor(audio, {})
    assert out.amplitude is Amplitude.UNKNOWN
