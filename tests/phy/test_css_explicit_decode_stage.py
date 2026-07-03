import pytest
from phy._css_lora import HEADER
from pydantic import ValidationError

from marconi.core.descriptor import Carrier, Descriptor
from marconi.core.levels import Level
from marconi.core.stages import StageDirectionError
from marconi.phy.compile_context import CompileContext
from marconi.phy.modulation.css.stages import CssExplicitDecode, _ExplicitParams


def _ctx():
    return CompileContext(
        Descriptor(Level.SYMBOLS, "s", "scalar", Carrier.HARD),
        rate=250000.0,
        symbol_rate=61.03515625,
    )


_PARAMS = dict(HEADER)


def test_stage_levels_and_emit():
    s = CssExplicitDecode()
    assert (s.from_level, s.to_level) == (Level.SYMBOLS, Level.BITS)
    ctx = _ctx()
    s.emit_rx(ctx, _PARAMS)
    pipe = ctx.build("t", 250000.0)
    assert any(b.kind == "css_explicit_decode" for b in pipe.blocks)


def test_stage_is_rx_only():
    with pytest.raises(StageDirectionError):
        CssExplicitDecode().emit_tx(_ctx(), _PARAMS)


def test_stage_in_registry():
    from marconi.phy.stages.registry import stage_registry

    assert "css_explicit_decode" in stage_registry()


# --- _ExplicitParams range guards ---


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
