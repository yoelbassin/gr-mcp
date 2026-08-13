from __future__ import annotations

import hashlib
from pathlib import Path

import pytest
from helpers.assets.derive import DERIVERS, DeriveError
from helpers.assets.manifest import load_manifest
from helpers.assets.resolve import ResolveError, resolve, resolve_all

PAYLOAD = b"0123456789" * 100
DOUBLED_SHA = hashlib.sha256(PAYLOAD * 2).hexdigest()


def _manifest(tmp_path: Path, body: str) -> Path:
    p = tmp_path / "assets.toml"
    p.write_text(body)
    return p


def _leftovers(root: Path) -> list[Path]:
    return sorted(root.rglob("*.part"))


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


@pytest.fixture
def interrupted(monkeypatch: pytest.MonkeyPatch) -> None:
    def _half(src: Path, dst: Path) -> None:
        dst.parent.mkdir(parents=True, exist_ok=True)
        dst.write_bytes(src.read_bytes()[:7])
        raise KeyboardInterrupt

    monkeypatch.setitem(DERIVERS, "half", _half)


@pytest.fixture
def silent(monkeypatch: pytest.MonkeyPatch) -> None:
    def _nothing(src: Path, dst: Path) -> None:
        return None

    monkeypatch.setitem(DERIVERS, "silent", _nothing)


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


def test_resolve_all_skips_absent_local_assets(tmp_path: Path) -> None:
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
    assert resolve_all(index, tmp_path / "artifacts") == []


def test_resolve_all_still_reports_a_real_fetch_failure(tmp_path: Path) -> None:
    index = load_manifest(
        _manifest(
            tmp_path,
            """
[[asset]]
kind = "local"
path = "P/absent.bin"

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
    )
    root = tmp_path / "artifacts"
    (root / "P").mkdir(parents=True)
    (root / "P" / "r.bin").write_bytes(PAYLOAD)
    unresolved = resolve_all(index, root)
    assert [u.name for u in unresolved] == ["P/leaf.bin"]
    # the cause is CI's only diagnostic: a bare name cannot be acted on
    assert "nope" in unresolved[0].cause


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
    unresolved = resolve_all(index, root)
    assert [u.name for u in unresolved] == []
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


def _derived_index(tmp_path: Path, base: str, deriver: str, pin: str) -> Path:
    return _manifest(
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
derive = "{deriver}"
{pin}
""",
    )


def test_a_derived_file_that_matches_its_digest_is_published(
    local_server: tuple[str, Path], tmp_path: Path, doubling: None
) -> None:
    base, www = local_server
    (www / "r.bin").write_bytes(PAYLOAD)
    index = load_manifest(
        _derived_index(tmp_path, base, "double", f'sha256 = "{DOUBLED_SHA}"')
    )
    root = tmp_path / "artifacts"
    assert resolve("P/leaf.bin", index, root).read_bytes() == PAYLOAD * 2
    assert _leftovers(root) == []


def test_a_derived_file_that_misses_its_digest_is_never_published(
    local_server: tuple[str, Path], tmp_path: Path, doubling: None
) -> None:
    # the deriver runs and produces real bytes; only the manifest digest says
    # they are the wrong ones. Publishing first and checking never (or not at
    # all) leaves the gate decoding whatever the deriver happened to write.
    base, www = local_server
    (www / "r.bin").write_bytes(PAYLOAD)
    index = load_manifest(
        _derived_index(tmp_path, base, "double", f'sha256 = "{"11" * 32}"')
    )
    root = tmp_path / "artifacts"
    with pytest.raises(ResolveError, match="sha256"):
        resolve("P/leaf.bin", index, root)
    assert not (root / "P" / "leaf.bin").exists()
    assert _leftovers(root) == []


def test_a_derived_file_of_the_wrong_size_is_never_published(
    local_server: tuple[str, Path], tmp_path: Path, doubling: None
) -> None:
    base, www = local_server
    (www / "r.bin").write_bytes(PAYLOAD)
    index = load_manifest(_derived_index(tmp_path, base, "double", "bytes = 7"))
    root = tmp_path / "artifacts"
    with pytest.raises(ResolveError, match="bytes"):
        resolve("P/leaf.bin", index, root)
    assert not (root / "P" / "leaf.bin").exists()


def test_an_interrupted_derive_leaves_no_short_file_behind(
    local_server: tuple[str, Path], tmp_path: Path, interrupted: None
) -> None:
    # a short file at the destination is the worst outcome: resolve returns it
    # forever, require_asset hands it to the gate, and verify calls it fine
    base, www = local_server
    (www / "r.bin").write_bytes(PAYLOAD)
    index = load_manifest(_derived_index(tmp_path, base, "half", ""))
    root = tmp_path / "artifacts"
    with pytest.raises(KeyboardInterrupt):
        resolve("P/leaf.bin", index, root)
    assert not (root / "P" / "leaf.bin").exists()
    assert _leftovers(root) == []


def test_a_deriver_that_writes_nothing_is_reported(
    local_server: tuple[str, Path], tmp_path: Path, silent: None
) -> None:
    base, www = local_server
    (www / "r.bin").write_bytes(PAYLOAD)
    index = load_manifest(_derived_index(tmp_path, base, "silent", ""))
    root = tmp_path / "artifacts"
    with pytest.raises(ResolveError, match="wrote nothing"):
        resolve("P/leaf.bin", index, root)
    assert not (root / "P" / "leaf.bin").exists()


def test_resolve_all_reports_a_derive_failure_instead_of_raising(
    local_server: tuple[str, Path], tmp_path: Path, exploding: None
) -> None:
    # a replacement upstream with a different sample width raises DeriveError:
    # uncaught, that is a raw traceback out of the CLI instead of a named asset
    base, www = local_server
    (www / "r.bin").write_bytes(PAYLOAD)
    index = load_manifest(_derived_index(tmp_path, base, "explode", ""))
    unresolved = resolve_all(index, tmp_path / "artifacts")
    assert [u.name for u in unresolved] == ["P/leaf.bin"]
    assert "simulated derive failure" in unresolved[0].cause
