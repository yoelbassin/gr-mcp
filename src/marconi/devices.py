"""Device registry. v1.0 knows only simulated devices (scenes behind a
device-shaped interface); hardware arrives via backends in v1.1."""

from dataclasses import dataclass

from marconi.backends import get_backend
from marconi.models import DeviceInfo, SceneSpec


class DeviceNotFoundError(Exception):
    """No device is registered under the requested id."""


@dataclass
class SimulatedDevice:
    id: str
    scene: SceneSpec
    can_tx: bool = True

    def info(self) -> DeviceInfo:
        return DeviceInfo(
            id=self.id,
            kind="simulated",
            can_tx=self.can_tx,
            description=f"simulated device with scene '{self.scene.name}'",
        )


_REGISTRY: dict[str, SimulatedDevice] = {}


def add_simulated_device(
    device_id: str, scene: SceneSpec, replace: bool = False
) -> SimulatedDevice:
    """Register a simulated device. Raises if `device_id` already exists unless
    `replace=True`, which redefines it (last-writer-wins) — the declarative
    semantics the simulate_scene tool wants so a scene can be re-issued."""
    if device_id in _REGISTRY and not replace:
        raise ValueError(f"device '{device_id}' already exists")
    dev = SimulatedDevice(id=device_id, scene=scene)
    _REGISTRY[device_id] = dev
    return dev


def get_device(device_id: str) -> SimulatedDevice:
    if device_id not in _REGISTRY:
        known = ", ".join(sorted(_REGISTRY)) or "none registered"
        raise DeviceNotFoundError(f"unknown device '{device_id}' (known: {known})")
    return _REGISTRY[device_id]


def list_devices(backend: str = "gnuradio") -> list[DeviceInfo]:
    infos = [d.info() for d in _REGISTRY.values()]
    infos.extend(get_backend(backend).enumerate_devices())
    return infos


def clear_devices() -> None:
    """Test helper: empty the registry."""
    _REGISTRY.clear()
