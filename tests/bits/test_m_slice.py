import numpy as np

from marconi.bits.carriers import RxCarrier
from marconi.bits.framing import m_slice_rx


def test_m_slice_maps_regions_to_levels() -> None:
    sym = np.array([-1.0, -0.5, 0.5, 1.0], np.float32)
    out = m_slice_rx(
        RxCarrier(bits=np.zeros(0, np.uint8), symbols=sym, marks=(2,)),
        thresholds=[-0.667, 0.0, 0.667],
        levels=[3, 2, 0, 1],
    )
    assert out.symbols is not None
    assert out.symbols.dtype == np.int16
    assert out.symbols.tolist() == [3, 2, 0, 1]
    assert out.marks == (2,)
