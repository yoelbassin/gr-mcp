from marconi import config
from marconi.devices import SimulatedDevice, get_device
from marconi.models import CaptureRef, SceneElement


class TransmitNotConfirmedError(Exception):
    """Transmission attempted without confirmation while CONFIRM_TX is on."""


class TransmitForbiddenError(Exception):
    """Transmission attempted on a device whose can_tx is False."""


def transmit_capture(
    device: SimulatedDevice | str,
    capture: CaptureRef,
    freq: float,
    amplitude: float = 1.0,
    confirmed: bool = False,
) -> SceneElement:
    """Replay a capture 'on the air'. v1.0: simulated devices only — the
    capture becomes an iq_file element of the device's scene, audible to
    subsequent captures (requires matching sample rates)."""
    dev = get_device(device) if isinstance(device, str) else device
    if config.CONFIRM_TX and not confirmed:
        raise TransmitNotConfirmedError(
            "transmission requires confirmed=True (or set "
            "marconi.config.CONFIRM_TX = False); about to transmit "
            f"{capture.path.name} at {freq/1e6:.4f} MHz on '{dev.id}'"
        )
    if not dev.can_tx:
        raise TransmitForbiddenError(f"device '{dev.id}' cannot transmit")
    element = SceneElement(
        kind="iq_file",
        freq=freq,
        amplitude=amplitude,
        params={"path": str(capture.path), "sample_rate": capture.sample_rate},
    )
    dev.scene.elements.append(element)
    return element
