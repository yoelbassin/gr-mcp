from __future__ import annotations

import pytest
from pydantic import ValidationError

from marconi.phy.modulation.ofdm.stages import _CoherentParams


def _good() -> dict:
    return {
        "fft_len": 64,
        "cp_len": 16,
        "sym_len": 80,
        "n_frame_syms": 4,
        "n_carriers": 48,
        "kmin": -24,
        "dc_search": 2,
        "warmup_syms": 12,
        "pilot_lens": [1, 1, 1, 1],
        "pilot_carriers": [-24, -23, -22, -21],
        "pilot_i": [1.0, 1.0, 1.0, 1.0],
        "pilot_q": [0.0, 0.0, 0.0, 0.0],
        "fp_carriers": [3],
        "fp_i": [1.0],
        "fp_q": [0.0],
    }


def test_valid_params_pass() -> None:
    assert _CoherentParams.model_validate(_good()).fft_len == 64


@pytest.mark.parametrize(
    "patch",
    [
        {"sym_len": 81},
        {"pilot_lens": [1, 1, 1]},
        {"pilot_carriers": [-24, -23, -22]},
        {"pilot_i": [1.0, 1.0, 1.0]},
        {"fp_i": [1.0, 2.0]},
        {"n_carriers": 0},
        {"warmup_syms": 4},
    ],
)
def test_inconsistent_geometry_rejected(patch: dict) -> None:
    with pytest.raises(ValidationError):
        _CoherentParams.model_validate({**_good(), **patch})
