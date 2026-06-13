from pathlib import Path

import pytest

from marconi.devices import (
    SimulatedDevice,
    add_simulated_device,
    clear_devices,
    get_device,
    list_devices,
)
from marconi.models import SceneElement, SceneSpec
from marconi.ops.analyze import find_signals
from marconi.ops.capture import capture
from marconi.workspace import Workspace


@pytest.fixture(autouse=True)
def _fresh_registry():
    clear_devices()
    yield
    clear_devices()


def _scene() -> SceneSpec:
    return SceneSpec(
        name="one_tone",
        elements=[
            SceneElement(kind="tone", freq=433.1e6, amplitude=1.0),
            SceneElement(kind="noise", amplitude=0.01),
        ],
    )


def test_registry_lists_simulated_devices() -> None:
    dev = add_simulated_device("sim0", _scene())
    assert isinstance(dev, SimulatedDevice)
    infos = list_devices()
    assert [d.id for d in infos] == ["sim0"]
    assert infos[0].kind == "simulated"
    assert infos[0].can_tx is True
    assert get_device("sim0") is dev


def test_duplicate_device_id_rejected() -> None:
    add_simulated_device("sim0", _scene())
    with pytest.raises(ValueError, match="already exists"):
        add_simulated_device("sim0", _scene())


def test_replace_redefines_device_scene() -> None:
    add_simulated_device("sim0", _scene())
    new_scene = SceneSpec(
        name="two_tone",
        elements=[SceneElement(kind="tone", freq=200e6, amplitude=1.0)],
    )
    dev = add_simulated_device("sim0", new_scene, replace=True)
    assert dev.scene.name == "two_tone"
    assert get_device("sim0").scene.name == "two_tone"
    assert [d.id for d in list_devices()] == ["sim0"]  # not duplicated


def test_unknown_device_rejected() -> None:
    with pytest.raises(KeyError, match="nope"):
        get_device("nope")


def test_capture_from_simulated_device(tmp_path: Path) -> None:
    add_simulated_device("sim0", _scene())
    ws = Workspace(tmp_path)
    ref = capture(
        "sim0", center_freq=433e6, sample_rate=1e6, duration=0.05, workspace=ws
    )
    assert ref.path.is_relative_to(ws.root / "captures")
    assert ref.center_freq == 433e6
    signals = find_signals(ref)
    assert len(signals) == 1
    assert abs(signals[0].center_freq - 433.1e6) < 2e3
