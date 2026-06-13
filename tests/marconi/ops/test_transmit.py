from pathlib import Path

import pytest

from marconi.devices import add_simulated_device, clear_devices
from marconi.models import SceneElement, SceneSpec
from marconi.ops.analyze import find_signals
from marconi.ops.capture import capture
from marconi.ops.simulate import render_scene
from marconi.ops.transmit import (
    TransmitForbiddenError,
    TransmitNotConfirmedError,
    transmit_capture,
)
from marconi.workspace import Workspace

pytestmark = pytest.mark.gnuradio


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


def _make_short_tone_capture(ws: Workspace):
    """Render a 20000-sample (0.02 s at 1 MHz) tone payload."""
    scene = SceneSpec(
        name="short_payload",
        elements=[SceneElement(kind="tone", freq=433e6, amplitude=1.0)],
    )
    return render_scene(
        scene, center_freq=433e6, sample_rate=1e6, duration=0.02, workspace=ws
    )


def test_transmit_payload_loops_to_fill_longer_capture(tmp_path: Path) -> None:
    """A payload shorter than the capture duration must loop (repeat=True),
    filling the full requested number of samples rather than truncating."""
    dev = add_simulated_device(
        "sim0", SceneSpec(elements=[SceneElement(kind="noise", amplitude=0.01)])
    )
    ws = Workspace(tmp_path)

    # 20 000-sample payload
    payload = _make_short_tone_capture(ws)
    assert payload.num_samples == 20_000

    transmit_capture(dev, payload, freq=433.2e6, confirmed=True)

    # Capture for 0.1 s → 100 000 samples (5× longer than the payload)
    rx = capture("sim0", center_freq=433e6, sample_rate=1e6, duration=0.1, workspace=ws)
    assert rx.num_samples == 100_000, (
        f"expected 100000 samples (full capture), got {rx.num_samples} — "
        "payload probably did not loop"
    )


def test_transmit_to_non_tx_device_rejected(tmp_path: Path) -> None:
    dev = add_simulated_device(
        "sim0", SceneSpec(elements=[SceneElement(kind="noise", amplitude=0.01)])
    )
    dev.can_tx = False
    ws = Workspace(tmp_path)
    payload = _make_tone_capture(ws)
    with pytest.raises(TransmitForbiddenError, match="cannot transmit"):
        transmit_capture(dev, payload, freq=433.2e6, confirmed=True)


def test_transmit_forbidden_takes_priority_over_confirmation(tmp_path: Path) -> None:
    # capability is checked before confirmation: a device that can't transmit is
    # rejected as forbidden even when unconfirmed (not asked to confirm first).
    dev = add_simulated_device(
        "sim0", SceneSpec(elements=[SceneElement(kind="noise", amplitude=0.01)])
    )
    dev.can_tx = False
    ws = Workspace(tmp_path)
    payload = _make_tone_capture(ws)
    with pytest.raises(TransmitForbiddenError):
        transmit_capture(dev, payload, freq=433.2e6)  # confirmed defaults to False
