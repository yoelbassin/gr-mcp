from pathlib import Path

import pytest
from pydantic import ValidationError

from marconi.engine.types.models import Symbolstream


@pytest.mark.parametrize("marks", [[5, 3], [3, 3], [-1], [10]])
def test_symbolstream_rejects_non_position_marks(
    tmp_path: Path, marks: list[int]
) -> None:
    with pytest.raises(ValidationError, match="marks"):
        Symbolstream(path=tmp_path / "s.i16", num_symbols=10, marks=marks)


def test_symbolstream_accepts_ordered_marks(tmp_path: Path) -> None:
    s = Symbolstream(path=tmp_path / "s.i16", num_symbols=10, marks=[0, 5, 9])
    assert s.marks == [0, 5, 9]
