import pytest
from phy._css_lora import HEADER
from pydantic import ValidationError

from marconi.bits.builder import ProgramBuilder
from marconi.bits.registry import registry
from marconi.bits.stages.symbol_ops import CssExplicitDecode, _ExplicitParams
from marconi.core.levels import Level
from marconi.core.stages import StageDirectionError

_PARAMS = dict(HEADER)


def test_stage_levels_and_emit():
    s = CssExplicitDecode()
    assert (s.from_level, s.to_level) == (Level.SYMBOLS, Level.BITS)
    b = ProgramBuilder()
    s.emit_rx(b, _PARAMS)
    assert len(b.steps) == 1


def test_stage_is_rx_only():
    assert CssExplicitDecode().directions == frozenset({"rx"})
    with pytest.raises(StageDirectionError):
        CssExplicitDecode().emit_tx(ProgramBuilder(), _PARAMS)


def test_stage_in_registry():
    assert "css_explicit_decode" in registry()


def test_explicit_params_valid_sf11_cr4():
    p = _ExplicitParams.model_validate(_PARAMS)
    assert p.sf == 11 and p.header_cr == 4


def test_explicit_params_sf_too_low_raises():
    with pytest.raises(ValidationError):
        _ExplicitParams.model_validate({**_PARAMS, "sf": 4})


def test_explicit_params_sf_too_high_raises():
    with pytest.raises(ValidationError):
        _ExplicitParams.model_validate({**_PARAMS, "sf": 15})


def test_explicit_params_header_cr_absent_from_parity_table_raises():
    with pytest.raises(ValidationError):
        _ExplicitParams.model_validate({**_PARAMS, "header_cr": 5})


def test_explicit_params_reject_sf_reduction_ge_sf_when_reduced():
    with pytest.raises(ValidationError):
        _ExplicitParams.model_validate(
            {**_PARAMS, "sf": 7, "reduced": True, "sf_reduction": 7}
        )
