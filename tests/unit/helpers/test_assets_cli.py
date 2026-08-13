from __future__ import annotations

import hashlib
from pathlib import Path

import pytest
from helpers.assets.__main__ import main

PAYLOAD = b"0123456789" * 10
PAYLOAD_SHA = hashlib.sha256(PAYLOAD).hexdigest()


def _manifest(tmp_path: Path, body: str) -> Path:
    p = tmp_path / "assets.toml"
    p.write_text(body)
    return p


def _pinned(tmp_path: Path, root: Path, content: bytes, extra: str = "") -> Path:
    (root / "P").mkdir(parents=True, exist_ok=True)
    (root / "P" / "pinned.bin").write_bytes(content)
    return _manifest(
        tmp_path,
        f"""
[[asset]]
kind = "local"
path = "P/pinned.bin"
sha256 = "{PAYLOAD_SHA}"
bytes = {len(PAYLOAD)}
{extra}
""",
    )


def _line(out: str, name: str) -> str:
    return next(line for line in out.splitlines() if name in line)


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


def test_verify_re_hashes_and_flags_a_file_of_the_right_size_but_wrong_bytes(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    # exactly the pinned length, none of the pinned content: an existence-only
    # verify calls this ok, which is how a destroyed capture stayed green
    root = tmp_path / "a"
    m = _pinned(tmp_path, root, bytes(len(PAYLOAD)))
    code = main(["verify", "--manifest", str(m), "--root", str(root)])
    out = capsys.readouterr().out
    assert code == 1
    assert _line(out, "P/pinned.bin").startswith("BAD")
    assert "sha256" in out


def test_verify_accepts_a_file_that_matches_its_digest(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    root = tmp_path / "a"
    m = _pinned(tmp_path, root, PAYLOAD)
    code = main(["verify", "--manifest", str(m), "--root", str(root)])
    assert code == 0
    assert _line(capsys.readouterr().out, "P/pinned.bin").startswith("ok")


def test_verify_is_nonzero_when_a_required_asset_is_absent(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    m = _manifest(
        tmp_path,
        """
[[asset]]
kind = "local"
path = "P/needed.bin"
ci_required = true
""",
    )
    root = tmp_path / "a"
    code = main(["verify", "--manifest", str(m), "--root", str(root)])
    assert code == 1
    assert _line(capsys.readouterr().out, "P/needed.bin").startswith("MISS")


def test_verify_and_list_missing_do_not_report_a_consumed_intermediate(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    # the wav exists only to be carved and discarded, so its absence on a
    # healthy tree is the expected state, not a missing asset
    m = _manifest(
        tmp_path,
        """
[[asset]]
kind = "fetched"
path = "P/big.wav"
url = "https://example.invalid/big.zip"
strategy = "zip_member"
member = "big.wav"
discard_after_derive = true

[[asset]]
kind = "derived"
path = "P/leaf.bin"
derive_from = "P/big.wav"
derive = "pocsag_slice"
""",
    )
    root = tmp_path / "a"
    (root / "P").mkdir(parents=True)
    (root / "P" / "leaf.bin").write_bytes(PAYLOAD)
    assert main(["list", "--missing", "--manifest", str(m), "--root", str(root)]) == 0
    assert "P/big.wav" not in capsys.readouterr().out
    assert main(["verify", "--manifest", str(m), "--root", str(root)]) == 0
    assert _line(capsys.readouterr().out, "P/big.wav").startswith("used")


def test_root_defaults_to_the_environment_override(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    # the fetch hint printed by a skipping gate names no --root, so the CLI
    # default and require_asset must land on the same tree or the remedy the
    # message prescribes downloads into a directory nothing ever reads
    root = tmp_path / "elsewhere"
    m = _pinned(tmp_path, root, PAYLOAD)
    monkeypatch.setenv("MARCONI_ASSET_ROOT", str(root))
    assert main(["verify", "--manifest", str(m)]) == 0
    assert _line(capsys.readouterr().out, "P/pinned.bin").startswith("ok")


def test_fetch_returns_nonzero_when_something_is_unresolvable(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    m = _manifest(
        tmp_path,
        """
[[asset]]
kind = "fetched"
path = "P/r.bin"
url = "does-not-matter"

[[asset]]
kind = "derived"
path = "P/leaf.bin"
derive_from = "P/r.bin"
derive = "nope"
""",
    )
    root = tmp_path / "a"
    (root / "P").mkdir(parents=True)
    (root / "P" / "r.bin").write_bytes(b"x")
    code = main(["fetch", "--manifest", str(m), "--root", str(root)])
    out = capsys.readouterr().out
    assert code == 1
    # a bare "unresolved: <name>" is CI's whole fetch diagnostic; without the
    # cause there is nothing to distinguish a 403 from a digest mismatch
    assert "unresolved: P/leaf.bin" in out
    assert "nope" in _line(out, "P/leaf.bin")


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


def test_fetch_ignores_an_absent_local_asset_and_still_fetches_the_rest(
    local_server: tuple[str, Path],
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    base, www = local_server
    (www / "r.bin").write_bytes(b"x" * 32)
    m = _manifest(
        tmp_path,
        f"""
[[asset]]
kind = "local"
path = "P/absent.bin"

[[asset]]
kind = "fetched"
path = "P/r.bin"
url = "{base}/r.bin"
""",
    )
    root = tmp_path / "a"
    code = main(["fetch", "--manifest", str(m), "--root", str(root)])
    out = capsys.readouterr().out
    assert code == 0
    assert (root / "P" / "r.bin").read_bytes() == b"x" * 32
    assert "local (no upstream): P/absent.bin" in out
    assert not (root / "P" / "absent.bin").exists()


def test_fetch_exits_zero_for_a_ci_required_local_asset_that_is_absent(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """fetch reports only what it could have obtained; ci_required enforcement
    for an unobtainable local asset is the strict session hook's job, not
    this command's — a local asset has no upstream for fetch to fail against."""
    m = _manifest(
        tmp_path,
        """
[[asset]]
kind = "local"
path = "P/absent.bin"
ci_required = true
""",
    )
    root = tmp_path / "a"
    code = main(["fetch", "--manifest", str(m), "--root", str(root)])
    out = capsys.readouterr().out
    assert code == 0
    assert "local (no upstream): P/absent.bin" in out
