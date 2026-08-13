from __future__ import annotations

from pathlib import Path

import pytest
from helpers.assets.derive import DERIVERS, DeriveError
from helpers.assets.manifest import load_manifest
from helpers.assets.resolve import ResolveError, resolve, resolve_all

PAYLOAD = b"0123456789" * 100


def _manifest(tmp_path: Path, body: str) -> Path:
    p = tmp_path / "assets.toml"
    p.write_text(body)
    return p


@pytest.fixture
def doubling(monkeypatch: pytest.MonkeyPatch) -> None:
    def _double(src: Path, dst: Path) -> None:
        dst.parent.mkdir(parents=True, exist_ok=True)
        dst.write_bytes(src.read_bytes() * 2)

    monkeypatch.setitem(DERIVERS, "double", _double)


@pytest.fixture
def exploding(monkeypatch: pytest.MonkeyPatch) -> None:
    def _explode(src: Path, dst: Path) -> None:
        raise DeriveError("simulated derive failure")

    monkeypatch.setitem(DERIVERS, "explode", _explode)


def test_fetches_a_root_then_derives_its_leaf(
    local_server: tuple[str, Path], tmp_path: Path, doubling: None
) -> None:
    base, www = local_server
    (www / "r.bin").write_bytes(PAYLOAD)
    index = load_manifest(
        _manifest(
            tmp_path,
            f"""
[[asset]]
kind = "fetched"
path = "P/r.bin"
url = "{base}/r.bin"

[[asset]]
kind = "derived"
path = "P/leaf.bin"
derive_from = "P/r.bin"
derive = "double"
""",
        )
    )
    root = tmp_path / "artifacts"
    got = resolve("P/leaf.bin", index, root=root)
    assert got.read_bytes() == PAYLOAD * 2
    assert (root / "P" / "r.bin").exists()


def test_discard_after_derive_removes_the_intermediate(
    local_server: tuple[str, Path], tmp_path: Path, doubling: None
) -> None:
    base, www = local_server
    (www / "r.bin").write_bytes(PAYLOAD)
    index = load_manifest(
        _manifest(
            tmp_path,
            f"""
[[asset]]
kind = "fetched"
path = "P/r.bin"
url = "{base}/r.bin"
discard_after_derive = true

[[asset]]
kind = "derived"
path = "P/leaf.bin"
derive_from = "P/r.bin"
derive = "double"
""",
        )
    )
    root = tmp_path / "artifacts"
    assert resolve("P/leaf.bin", index, root=root).exists()
    assert not (root / "P" / "r.bin").exists()


def test_a_failed_derive_never_discards_the_parent(
    local_server: tuple[str, Path], tmp_path: Path, exploding: None
) -> None:
    base, www = local_server
    (www / "r.bin").write_bytes(PAYLOAD)
    index = load_manifest(
        _manifest(
            tmp_path,
            f"""
[[asset]]
kind = "fetched"
path = "P/r.bin"
url = "{base}/r.bin"
discard_after_derive = true

[[asset]]
kind = "derived"
path = "P/leaf.bin"
derive_from = "P/r.bin"
derive = "explode"
""",
        )
    )
    root = tmp_path / "artifacts"
    with pytest.raises(DeriveError):
        resolve("P/leaf.bin", index, root=root)
    assert (root / "P" / "r.bin").exists()


def test_an_existing_file_is_not_refetched(
    local_server: tuple[str, Path], tmp_path: Path
) -> None:
    base, www = local_server
    index = load_manifest(
        _manifest(
            tmp_path,
            f"""
[[asset]]
kind = "fetched"
path = "P/r.bin"
url = "{base}/absent.bin"
""",
        )
    )
    root = tmp_path / "artifacts"
    (root / "P").mkdir(parents=True)
    (root / "P" / "r.bin").write_bytes(PAYLOAD)
    assert resolve("P/r.bin", index, root=root).read_bytes() == PAYLOAD


def test_a_local_asset_that_is_absent_raises(tmp_path: Path) -> None:
    index = load_manifest(
        _manifest(
            tmp_path,
            """
[[asset]]
kind = "local"
path = "P/only.bin"
note = "ask the ops channel"
""",
        )
    )
    with pytest.raises(ResolveError, match="ask the ops channel"):
        resolve("P/only.bin", index, root=tmp_path / "artifacts")


def test_resolve_all_reports_the_names_it_could_not_get(tmp_path: Path) -> None:
    index = load_manifest(
        _manifest(
            tmp_path,
            """
[[asset]]
kind = "local"
path = "P/a.bin"

[[asset]]
kind = "local"
path = "P/b.bin"
""",
        )
    )
    assert sorted(resolve_all(index, root=tmp_path / "artifacts")) == [
        "P/a.bin",
        "P/b.bin",
    ]


def test_resolve_all_skips_discard_only_parents(
    local_server: tuple[str, Path], tmp_path: Path, doubling: None
) -> None:
    base, www = local_server
    (www / "r.bin").write_bytes(PAYLOAD)
    index = load_manifest(
        _manifest(
            tmp_path,
            f"""
[[asset]]
kind = "derived"
path = "P/leaf.bin"
derive_from = "P/r.bin"
derive = "double"

[[asset]]
kind = "fetched"
path = "P/r.bin"
url = "{base}/r.bin"
discard_after_derive = true
""",
        )
    )
    root = tmp_path / "artifacts"
    unresolved = resolve_all(index, root=root)
    assert "P/r.bin" not in unresolved
    assert (root / "P" / "leaf.bin").exists()
    assert not (root / "P" / "r.bin").exists()


def test_unknown_deriver_is_named(
    local_server: tuple[str, Path], tmp_path: Path
) -> None:
    base, www = local_server
    (www / "r.bin").write_bytes(PAYLOAD)
    index = load_manifest(
        _manifest(
            tmp_path,
            f"""
[[asset]]
kind = "fetched"
path = "P/r.bin"
url = "{base}/r.bin"

[[asset]]
kind = "derived"
path = "P/leaf.bin"
derive_from = "P/r.bin"
derive = "nope"
""",
        )
    )
    with pytest.raises(ResolveError, match="nope"):
        resolve("P/leaf.bin", index, root=tmp_path / "artifacts")
