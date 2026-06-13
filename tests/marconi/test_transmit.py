from pathlib import Path

import pytest

from marconi.devices import add_simulated_device, clear_devices
from marconi.models import SceneElement, SceneSpec
from marconi.ops.analyze import find_signals
from marconi.ops.capture import capture
from marconi.ops.simulate import render_scene
from marconi.ops.transmit import TransmitNotConfirmedError, transmit_capture
from marconi.workspace import Workspace


@pytest.fixture(autouse=True)
def _fresh_registry():
    clear_devices()
    yield
    clear_devices()


def _make_tone_capture(ws: Workspace):
    scene = SceneSpec(
        name="tx_payload",
        elements=[SceneElement(kind="tone", freq=433e6, amplitude=1.0)],
    )
    return render_scene(
        scene, center_freq=433e6, sample_rate=1e6, duration=0.05, workspace=ws
    )


def test_transmit_requires_confirmation(tmp_path: Path) -> None:
    dev = add_simulated_device(
        "sim0", SceneSpec(elements=[SceneElement(kind="noise", amplitude=0.01)])
    )
    ws = Workspace(tmp_path)
    payload = _make_tone_capture(ws)
    with pytest.raises(TransmitNotConfirmedError):
        transmit_capture(dev, payload, freq=433.2e6)


def test_transmit_places_capture_in_scene(tmp_path: Path) -> None:
    dev = add_simulated_device(
        "sim0", SceneSpec(elements=[SceneElement(kind="noise", amplitude=0.01)])
    )
    ws = Workspace(tmp_path)
    payload = _make_tone_capture(ws)

    transmit_capture(dev, payload, freq=433.2e6, confirmed=True)
    assert any(e.kind == "iq_file" for e in dev.scene.elements)

    rx = capture(
        "sim0", center_freq=433e6, sample_rate=1e6, duration=0.05, workspace=ws
    )
    signals = find_signals(rx)
    # payload tone was at baseband DC of the tx capture; placed at 433.2 MHz
    assert any(abs(s.center_freq - 433.2e6) < 3e3 for s in signals)


def test_transmit_to_non_tx_device_rejected(tmp_path: Path) -> None:
    dev = add_simulated_device(
        "sim0", SceneSpec(elements=[SceneElement(kind="noise", amplitude=0.01)])
    )
    dev.can_tx = False
    ws = Workspace(tmp_path)
    payload = _make_tone_capture(ws)
    with pytest.raises(PermissionError, match="cannot transmit"):
        transmit_capture(dev, payload, freq=433.2e6, confirmed=True)
