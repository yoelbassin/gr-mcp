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


def test_sync_symbols_tolerates_one_error() -> None:
    sym = np.array([0.5, 0.4, 0.9, -0.7], np.float32)  # 3rd sign wrong vs pattern
    out = sync_symbols_rx(
        RxCarrier(bits=np.zeros(0, np.uint8), symbols=sym),
        pattern=[1, 1, -1, -1],
        max_errors=1,
        pre_symbols=0,
    )
    assert out.marks == (0,)


def test_sync_symbols_drops_negative_offset_on_exact_match() -> None:
    sym = np.array([0.0, 0.0, 0.0, 0.5, 0.4, -0.6, -0.7, 0.0], np.float32)
    out = sync_symbols_rx(
        RxCarrier(bits=np.zeros(0, np.uint8), symbols=sym),
        pattern=[1, 1, -1, -1],
        max_errors=0,
        pre_symbols=4,  # match is at index 3; pre_symbols exceeds it
    )
    assert out.marks == ()


def test_sync_symbols_dont_care_excluded_from_agreement() -> None:
    sym = np.array(
        [0.5, 0.0, -0.7], np.float32
    )  # middle sign matches neither +1 nor -1
    out = sync_symbols_rx(
        RxCarrier(bits=np.zeros(0, np.uint8), symbols=sym),
        pattern=[1, 0, -1],
        max_errors=0,
        pre_symbols=0,
    )
    assert out.marks == (0,)


def test_sync_symbols_stage_registered() -> None:
    from marconi.bits.registry import registry

    assert "sync_symbols" in registry()
