from pathlib import Path

import pytest

from marconi.models import SceneElement, SceneSpec
from marconi.ops.analyze import find_signals
from marconi.ops.simulate import render_scene, scene_to_pipeline
from marconi.vocabulary import validate_pipeline
from marconi.workspace import Workspace


def _scene() -> SceneSpec:
    return SceneSpec(
        name="two_tones",
        elements=[
            SceneElement(kind="tone", freq=100.1e6, amplitude=1.0),
            SceneElement(kind="tone", freq=99.8e6, amplitude=0.5),
            SceneElement(kind="noise", amplitude=0.01),
        ],
    )


def test_scene_to_pipeline_is_valid(tmp_path: Path) -> None:
    spec = scene_to_pipeline(
        _scene(),
        center_freq=100e6,
        sample_rate=1e6,
        duration=0.05,
        out_path=tmp_path / "s.cf32",
    )
    assert validate_pipeline(spec) == []
    types = [b.type for b in spec.blocks]
    assert types.count("tone_source") == 2
    assert types.count("noise_source") == 1
    assert types.count("add") == 2  # 3 sources -> 2 adders
    assert "head" in types and "file_sink" in types


def test_out_of_band_elements_skipped(tmp_path: Path) -> None:
    scene = _scene()
    scene.elements.append(SceneElement(kind="tone", freq=200e6))
    spec = scene_to_pipeline(
        scene,
        center_freq=100e6,
        sample_rate=1e6,
        duration=0.01,
        out_path=tmp_path / "s.cf32",
    )
    assert [b.type for b in spec.blocks].count("tone_source") == 2


def test_empty_band_rejected(tmp_path: Path) -> None:
    scene = SceneSpec(elements=[SceneElement(kind="tone", freq=200e6)])
    with pytest.raises(ValueError, match="no elements"):
        scene_to_pipeline(
            scene,
            center_freq=100e6,
            sample_rate=1e6,
            duration=0.01,
            out_path=tmp_path / "s.cf32",
        )


def test_fm_requires_divisible_rate(tmp_path: Path) -> None:
    scene = SceneSpec(
        elements=[SceneElement(kind="fm_tone", freq=100e6, params={"mod_freq": 1e3})]
    )
    with pytest.raises(ValueError, match="multiple of"):
        scene_to_pipeline(
            scene,
            center_freq=100e6,
            sample_rate=1.5e5,
            duration=0.01,
            out_path=tmp_path / "s.cf32",
        )


@pytest.mark.gnuradio
def test_render_scene_produces_findable_signals(tmp_path: Path) -> None:
    ws = Workspace(tmp_path / "project")
    ref = render_scene(
        _scene(), center_freq=100e6, sample_rate=1e6, duration=0.1, workspace=ws
    )
    assert ref.path.exists()
    assert ref.center_freq == 100e6
    assert ref.num_samples == 100000
    signals = find_signals(ref)
    assert len(signals) == 2
    found = sorted(s.center_freq for s in signals)
    assert abs(found[0] - 99.8e6) < 2e3
    assert abs(found[1] - 100.1e6) < 2e3
