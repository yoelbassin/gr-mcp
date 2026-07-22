import numpy as np
import pytest
from pydantic import ValidationError

from marconi.bits.carriers import RxCarrier
from marconi.bits.framing import m_slice_rx
from marconi.bits.registry import registry
from marconi.bits.stages.symbol_ops import _MSliceParams


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


def test_m_slice_symbol_exactly_on_threshold_lands_in_lower_region() -> None:
    sym = np.array([0.0], np.float32)  # exactly thresholds[1]
    out = m_slice_rx(
        RxCarrier(bits=np.zeros(0, np.uint8), symbols=sym, marks=(0,)),
        thresholds=[-0.667, 0.0, 0.667],
        levels=[3, 2, 0, 1],
    )
    assert out.symbols is not None
    assert out.symbols.tolist() == [2]  # searchsorted(side='left') -> region 1


def test_m_slice_stage_registered() -> None:
    assert "m_slice" in registry()


def test_m_slice_params_length_mismatch_raises() -> None:
    with pytest.raises(ValidationError):
        _MSliceParams.model_validate(
            {"thresholds": [-0.667, 0.0, 0.667], "levels": [3, 2, 0]}
        )


def test_m_slice_params_non_ascending_thresholds_raises() -> None:
    with pytest.raises(ValidationError):
        _MSliceParams.model_validate(
            {"thresholds": [0.0, 0.0, 1.0], "levels": [3, 2, 0, 1]}
        )
