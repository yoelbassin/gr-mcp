from __future__ import annotations

import threading

import numpy as np
import pytest

from marconi.engine.backends.gnuradio.blocks import (
    BlockParams,
    _factories,
    _modules,
)

SPS = 4.0
SEG = 4096


def _agc2_output(attack_sym: float, decay_sym: float) -> np.ndarray:
    gr = _modules().gr
    from gnuradio import blocks

    step = np.concatenate(
        [
            0.05 * np.ones(SEG, dtype=np.complex64),
            5.0 * np.ones(SEG, dtype=np.complex64),
        ]
    )
    agc = _factories()["agc2_cc"](
        BlockParams(
            {
                "attack_rate": 1.0 / (attack_sym * SPS),
                "decay_rate": 1.0 / (decay_sym * SPS),
                "reference": 1.0,
                "max_gain": 0.0,
            }
        )
    )
    tb = gr.top_block("agc2_settle")
    src = blocks.vector_source_c(step.tolist(), False)
    snk = blocks.vector_sink_c()
    tb.connect(src, agc)
    tb.connect(agc, snk)
    t = threading.Thread(target=tb.run)
    t.start()
    t.join(60)
    assert not t.is_alive()
    return np.abs(np.array(snk.data()))


def _settles(attack_sym: float, decay_sym: float) -> bool:
    env = _agc2_output(attack_sym, decay_sym)
    tail = env[-SEG // 4 :]
    return bool(
        0.8 < float(tail.mean()) < 1.2 and float(tail.std()) < 0.1 * tail.mean()
    )


@pytest.mark.parametrize(
    "attack_sym,decay_sym",
    [
        (1.0, 1.0),
        (1.0, 16.0),
    ],
)
def test_agc2_settles_at_shipped_time_constants(
    attack_sym: float, decay_sym: float
) -> None:
    assert _settles(
        attack_sym, decay_sym
    ), f"limit cycle / no settle at attack={attack_sym}, decay={decay_sym} symbols"
