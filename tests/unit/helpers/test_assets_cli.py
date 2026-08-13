from __future__ import annotations

from pathlib import Path

import pytest
from helpers.assets.__main__ import main


def _manifest(tmp_path: Path, body: str) -> Path:
    p = tmp_path / "assets.toml"
    p.write_text(body)
    return p


def test_list_missing_prints_absent_assets(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    m = _manifest(
        tmp_path,
        """
[[asset]]
kind = "local"
path = "P/absent.bin"
""",
    )
    code = main(
        ["list", "--missing", "--manifest", str(m), "--root", str(tmp_path / "a")]
    )
    assert code == 0
    assert "P/absent.bin" in capsys.readouterr().out


def test_fetch_returns_nonzero_when_something_is_unresolvable(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    m = _manifest(
        tmp_path,
        """
[[asset]]
kind = "local"
path = "P/absent.bin"
""",
    )
    code = main(["fetch", "--manifest", str(m), "--root", str(tmp_path / "a")])
    assert code == 1
    assert "P/absent.bin" in capsys.readouterr().out


def test_fetch_succeeds_when_everything_resolves(
    local_server: tuple[str, Path], tmp_path: Path
) -> None:
    base, www = local_server
    (www / "r.bin").write_bytes(b"x" * 32)
    m = _manifest(
        tmp_path,
        f"""
[[asset]]
kind = "fetched"
path = "P/r.bin"
url = "{base}/r.bin"
""",
    )
    root = tmp_path / "a"
    assert main(["fetch", "--manifest", str(m), "--root", str(root)]) == 0
    assert (root / "P" / "r.bin").read_bytes() == b"x" * 32
