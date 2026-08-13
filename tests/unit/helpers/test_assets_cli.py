from __future__ import annotations

from pathlib import Path

import pytest
from helpers.assets.__main__ import main


def _manifest(tmp_path: Path, body: str) -> Path:
    p = tmp_path / "assets.toml"
    p.write_text(body)
    return p


def _mixed_manifest(tmp_path: Path, root: Path) -> Path:
    (root / "P").mkdir(parents=True)
    (root / "P" / "present.bin").write_bytes(b"x")
    return _manifest(
        tmp_path,
        """
[[asset]]
kind = "local"
path = "P/present.bin"

[[asset]]
kind = "local"
path = "P/absent.bin"
""",
    )


def test_list_prints_all_manifest_names(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    root = tmp_path / "a"
    m = _mixed_manifest(tmp_path, root)
    code = main(["list", "--manifest", str(m), "--root", str(root)])
    assert code == 0
    out = capsys.readouterr().out
    assert "P/present.bin" in out
    assert "P/absent.bin" in out


def test_list_missing_prints_only_absent_assets(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    root = tmp_path / "a"
    m = _mixed_manifest(tmp_path, root)
    code = main(["list", "--missing", "--manifest", str(m), "--root", str(root)])
    assert code == 0
    out = capsys.readouterr().out
    assert "P/absent.bin" in out
    assert "P/present.bin" not in out


def test_verify_reports_presence_per_asset(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    root = tmp_path / "a"
    m = _mixed_manifest(tmp_path, root)
    code = main(["verify", "--manifest", str(m), "--root", str(root)])
    assert code == 0
    lines = capsys.readouterr().out.splitlines()
    present_line = next(line for line in lines if "P/present.bin" in line)
    absent_line = next(line for line in lines if "P/absent.bin" in line)
    assert present_line.startswith("ok")
    assert absent_line.startswith("MISS")


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
