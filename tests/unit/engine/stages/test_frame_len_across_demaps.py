"""frame_len must cross the SYMBOLS->BITS seam or the downstream frame-
geometry compile check silently never fires: a mis-sized polar/ldpc/fec
codeword was REJECTED behind soft_demap (which rescales) and COMPILED behind
psk_soft_demap/dqpsk_soft_demap (which dropped the frame) - the documented
OFDM composition was the unguarded one. The sweep pins every current and
future demap; the per-stage factors are pinned separately below."""

from __future__ import annotations

import math

from unit.engine.types.test_param_bounds import _BASE

from marconi.engine.stages.registry import stage_registry
from marconi.engine.types.descriptor import Amplitude, Carrier, Descriptor
from marconi.engine.types.enums import ItemType
from marconi.engine.types.levels import Level


def _in_desc(stage: object, frame_len: int) -> Descriptor:
    item = getattr(stage, "accepts_item_type", None) or ItemType.F
    carrier = getattr(stage, "accepts_carrier", None) or Carrier.HARD
    return Descriptor(
        Level.SYMBOLS, item, carrier, Amplitude.UNKNOWN, frame_len=frame_len
    )


def test_every_symbols_to_bits_stage_carries_frame_len() -> None:
    checked = []
    for name, stage in sorted(stage_registry().items()):
        if stage.from_level is not Level.SYMBOLS or stage.to_level is not Level.BITS:
            continue
        step = stage.step_model.model_validate({"conv": name, **_BASE[name]})
        out = stage.out_descriptor(_in_desc(stage, 8), step)
        assert out.frame_len is not None, (
            f"{name} drops frame_len across the SYMBOLS->BITS seam, so a "
            "downstream codeword-geometry check can never fire behind it"
        )
        checked.append(name)
    assert len(checked) >= 8, checked  # the sweep found the demap family


def test_bit_expanding_demaps_scale_the_frame() -> None:
    reg = stage_registry()
    cases: dict[str, tuple[dict[str, object], int]] = {
        # k bits per symbol: frame_len_in * k
        "psk_demap": ({"order": 4}, 2),
        "psk_soft_demap": ({"order": 4}, 2),
        "qam_demap": ({"order": 16}, 4),
        "mfsk_soft_demap": ({"levels": [-3.0, -1.0, 1.0, 3.0]}, 2),
        "soft_demap": ({"scheme": "psk", "order": 4}, 2),
        "css_demap": ({"sf": 7}, 7),
        "slice": ({}, 1),
    }
    for name, (spec, k) in cases.items():
        stage = reg[name]
        step = stage.step_model.model_validate({"conv": name, **spec})
        out = stage.out_descriptor(_in_desc(stage, 8), step)
        assert out.frame_len == 8 * k, (name, out.frame_len)
    # dqpsk_soft_demap is not a per-item expander: it consumes one whole OFDM
    # frame and drops the reference SYMBOL inside it, so its output frame is
    # its own geometry (data_syms * n_carriers cells, k LLRs each), not the
    # input frame scaled. tests/unit/engine/modulation/ofdm/
    # test_ofdm_frame_geometry.py owns that seam.
    stage = reg["dqpsk_soft_demap"]
    step = stage.step_model.model_validate(
        {"conv": "dqpsk_soft_demap", "data_syms": 3, "n_carriers": 8}
    )
    assert stage.out_descriptor(_in_desc(stage, 32), step).frame_len == 3 * 8 * 2
    # symbol_map emits data_bits per input symbol
    stage = reg["symbol_map"]
    step = stage.step_model.model_validate(
        {"conv": "symbol_map", "code_bits": 2, "data_bits": 1, "table": [1, 2]}
    )
    assert stage.out_descriptor(_in_desc(stage, 8), step).frame_len == 8
    assert math.log2(4) == 2.0  # the k values above are log2(order)
