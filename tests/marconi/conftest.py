from collections.abc import Callable

import numpy as np
import pytest

MakeIQ = Callable[..., np.ndarray]


@pytest.fixture
def make_iq() -> MakeIQ:
    """Build complex64 baseband: noise plus tones at given (offset_hz, amplitude)."""

    def _make(
        tones: list[tuple[float, float]],
        sample_rate: float = 1e6,
        duration: float = 0.05,
        noise_amplitude: float = 0.01,
        seed: int = 0,
    ) -> np.ndarray:
        rng = np.random.default_rng(seed)
        n = int(sample_rate * duration)
        t = np.arange(n) / sample_rate
        x = rng.normal(0, noise_amplitude, n) + 1j * rng.normal(0, noise_amplitude, n)
        for freq, amp in tones:
            x = x + amp * np.exp(2j * np.pi * freq * t)
        return x.astype(np.complex64)

    return _make
