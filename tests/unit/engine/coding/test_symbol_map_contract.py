"""symbol_map/codebook inverse-map contracts: symbol values must lie inside
the code_bits alphabet (numpy negative indexing silently decodes the wrong
entry otherwise), and a table with negative or duplicate codewords cannot
build an unambiguous inverse."""

import numpy as np
import pytest
from pydantic import ValidationError

from marconi.engine.coding.carrier import CodingCarrier
from marconi.engine.coding.ops_bits import codebook_rx
from marconi.engine.coding.stages_bits import CodebookStep
from marconi.engine.coding.stages_symbols import SymbolMapStep


def _symbol_carrier(values: list[int]) -> CodingCarrier:
    return CodingCarrier(
        bits=np.zeros(0, np.uint8), symbols=np.asarray(values, np.int64)
    )


def test_negative_symbol_values_are_rejected_not_wrapped() -> None:
    # m_slice(levels=[-1, 1]) into symbol_map: inv[-1] IS inv[1] via numpy
    # negative indexing — both symbol levels silently decode to the same bit
    with pytest.raises(ValueError, match=r"\[0, 2\)"):
        codebook_rx(
            _symbol_carrier([-1, 1, -1]),
            code_bits=1,
            data_bits=1,
            table=[0, 1],
            symbol_input=True,
        )


def test_out_of_alphabet_symbol_values_are_rejected() -> None:
    with pytest.raises(ValueError, match=r"\[0, 4\)"):
        codebook_rx(
            _symbol_carrier([0, 3, 4]),
            code_bits=2,
            data_bits=2,
            table=[0, 1, 2, 3],
            symbol_input=True,
        )


@pytest.mark.parametrize("step_cls", [CodebookStep, SymbolMapStep])
def test_table_rejects_negative_entries(step_cls: type) -> None:
    with pytest.raises(ValidationError, match="0, 2\\*\\*code_bits"):
        step_cls(code_bits=2, data_bits=1, table=[-1, 2])


@pytest.mark.parametrize("step_cls", [CodebookStep, SymbolMapStep])
def test_table_rejects_oversized_entries(step_cls: type) -> None:
    with pytest.raises(ValidationError, match="0, 2\\*\\*code_bits"):
        step_cls(code_bits=2, data_bits=1, table=[1, 4])


@pytest.mark.parametrize("step_cls", [CodebookStep, SymbolMapStep])
def test_table_rejects_duplicate_entries(step_cls: type) -> None:
    with pytest.raises(ValidationError, match="unique"):
        step_cls(code_bits=2, data_bits=1, table=[3, 3])


def test_symbol_map_declares_its_input_alphabet() -> None:
    # the one SYMBOLS->BITS demap with no required_input_order: validate_modem
    # blessed m_slice(levels=[0..3]) -> symbol_map(code_bits=1), and the worker
    # then threw "symbol values must lie in [0, 2)" from inside the coding
    # lane - the sibling psk_demod(8)->psk_demap(4) mismatch fails at compile
    from marconi.engine.stages.registry import stage_registry

    stage = stage_registry()["symbol_map"]
    step = stage.step_model.model_validate(
        {"conv": "symbol_map", "code_bits": 2, "data_bits": 1, "table": [1, 2]}
    )
    assert stage.required_input_order(step) == 4
