from __future__ import annotations

from pathlib import Path

import numpy as np

from marconi.engine.io.bitfile import write_llrs
from marconi.engine.quality import soft_evidence


def _llr_file(tmp_path: Path, values: np.ndarray) -> Path:
    p = tmp_path / "llrs.f32"
    write_llrs(p, values.astype(np.float32))
    return p


def test_bimodal_llrs_are_positive(tmp_path: Path) -> None:
    rng = np.random.default_rng(0)
    signs = rng.choice([-2.0, 2.0], size=4000)
    ev = soft_evidence(_llr_file(tmp_path, signs + rng.normal(0, 0.3, 4000)))
    assert [e.assessment for e in ev] == ["positive"]
    assert ev[0].metric == "soft_confidence"


def test_gaussian_noise_llrs_are_negative(tmp_path: Path) -> None:
    rng = np.random.default_rng(1)
    ev = soft_evidence(_llr_file(tmp_path, rng.normal(0, 1.0, 4000)))
    assert [e.assessment for e in ev] == ["negative"]


def test_all_zero_stream_is_negative(tmp_path: Path) -> None:
    ev = soft_evidence(_llr_file(tmp_path, np.zeros(4000)))
    assert [e.assessment for e in ev] == ["negative"]


def test_too_few_items_yield_nothing(tmp_path: Path) -> None:
    assert soft_evidence(_llr_file(tmp_path, np.ones(100))) == []


def test_none_and_missing_paths_yield_nothing(tmp_path: Path) -> None:
    assert soft_evidence(None) == []
    assert soft_evidence(tmp_path / "absent.f32") == []
