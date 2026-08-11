from __future__ import annotations

import pytest
from pydantic import ValidationError

from marconi.engine.modulation.ofdm.stages import (
    OfdmDemodStep,
    OfdmFrameSyncProbeStep,
)


def _good() -> dict[str, object]:
    return {
        "fft_len": 64,
        "cp_len": 16,
        "sym_len": 80,
        "null_len": 96,
        "frame_len": 1296,
        "data_syms": 14,
        "n_carriers": 48,
        "bin_perm": list(range(64)),
    }


def test_good_geometry_is_accepted() -> None:
    OfdmDemodStep(**_good())  # type: ignore[arg-type]
    probe = {k: v for k, v in _good().items() if k != "bin_perm"}
    probe.pop("n_carriers")
    OfdmFrameSyncProbeStep(**probe)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "field,value,match",
    [
        # frame_len 0 made the sync block's recovery branch a no-op, so its
        # buffer never trimmed and general_work spun forever on any capture
        # without the declared null: the run died on the wall-clock timeout
        # with nothing pointing at the parameter.
        ("frame_len", 0, "positive"),
        # np.convolve(p, np.ones(0)/0) — "v cannot be empty"
        ("null_len", 0, "positive"),
        ("null_len", -3, "positive"),
        # a negative count reached keep_m_in_n as a stride
        ("data_syms", -5, "positive"),
        ("fft_len", 0, "positive"),
        # accepted silently, then _frame_usefuls sliced at the wrong stride
        # and the demod emitted garbage under status ok
        ("sym_len", 999, "sym_len must equal fft_len \\+ cp_len"),
    ],
)
def test_degenerate_geometry_is_refused(field: str, value: int, match: str) -> None:
    spec = {**_good(), field: value}
    with pytest.raises(ValidationError, match=match):
        OfdmDemodStep(**spec)  # type: ignore[arg-type]


def test_carriers_must_fit_the_fft() -> None:
    with pytest.raises(ValidationError, match="n_carriers"):
        OfdmDemodStep(**{**_good(), "n_carriers": 999})  # type: ignore[arg-type]


def test_bin_perm_must_cover_the_fft() -> None:
    with pytest.raises(ValidationError, match="bin_perm"):
        OfdmDemodStep(**{**_good(), "bin_perm": [0, 1, 2]})  # type: ignore[arg-type]
