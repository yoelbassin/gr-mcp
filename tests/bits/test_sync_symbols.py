import numpy as np

from marconi.bits.carriers import RxCarrier
from marconi.bits.framing import sync_symbols_rx


def test_sync_symbols_marks_burst_start() -> None:
    # pattern of signs {+,+,-,-}; burst starts 3 symbols before the pattern.
    sym = np.array([0.0, 0.0, 0.0, 0.5, 0.4, -0.6, -0.7, 0.0], np.float32)
    out = sync_symbols_rx(
        RxCarrier(bits=np.zeros(0, np.uint8), symbols=sym),
        pattern=[1, 1, -1, -1],
        max_errors=0,
        pre_symbols=3,
    )
    assert out.marks == (0,)  # pattern at index 3, minus pre_symbols 3


def test_sync_symbols_tolerates_errors_and_drops_negative_offset() -> None:
    sym = np.array([0.5, 0.4, 0.9, -0.7], np.float32)  # 3rd sign wrong vs pattern
    out = sync_symbols_rx(
        RxCarrier(bits=np.zeros(0, np.uint8), symbols=sym),
        pattern=[1, 1, -1, -1],
        max_errors=1,
        pre_symbols=10,
    )
    assert out.marks == ()  # match at 0, minus 10 -> negative -> dropped


def test_sync_symbols_stage_registered() -> None:
    from marconi.bits.registry import registry

    assert "sync_symbols" in registry()
