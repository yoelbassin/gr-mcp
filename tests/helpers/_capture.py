"""Device-probe stand-ins for the capture path.

capture runs its probe in a child process, so a stub monkeypatched over
`record._probe_device` is pickled BY REFERENCE and imported by that child: it
has to be a module-level function in an importable module. A lambda or a
function defined inside a test body cannot cross the boundary, which is why
these live here rather than beside their tests.
"""

from __future__ import annotations

import os
import time

from marconi.capture.record import CaptureError, DeviceReadback

# What a device that snaps the request to its own grid reports back. Any value
# the requested one is not, so a test can tell a readback that crossed back
# from the child from the requested value echoed by the no-bindings fallback.
SNAPPED_RATE = 1.92e6

# Longer than any probe budget under test, so a probe that is NOT interrupted
# fails its test on the threshold instead of hanging the suite.
WEDGED_SECONDS = 30.0


def stub_probe(device: str, rate: float, freq: float, ppm: float) -> DeviceReadback:
    return DeviceReadback(sample_rate=rate, center_hz=freq)


def snapping_probe(device: str, rate: float, freq: float, ppm: float) -> DeviceReadback:
    return DeviceReadback(sample_rate=SNAPPED_RATE, center_hz=freq)


def refusing_probe(device: str, rate: float, freq: float, ppm: float) -> DeviceReadback:
    raise CaptureError(f"no SDR device found (device={device!r})")


def crashing_probe(device: str, rate: float, freq: float, ppm: float) -> DeviceReadback:
    """A driver that takes its process down instead of raising — the reason the
    probe's own exit is read rather than assumed to be a timeout."""
    os._exit(70)


def wedged_probe(device: str, rate: float, freq: float, ppm: float) -> DeviceReadback:
    """A dongle whose driver never comes back — the hang this path had no bound
    for. Blocks in the child, so what the test measures is the parent's clock."""
    time.sleep(WEDGED_SECONDS)
    return DeviceReadback(sample_rate=rate, center_hz=freq)
