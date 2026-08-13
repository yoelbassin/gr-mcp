from __future__ import annotations

from pathlib import Path

import pytest
from helpers.assets import SKIPPED, asset_path, require_asset, strict_failures
from helpers.assets.manifest import load_manifest


def _manifest(tmp_path: Path, body: str) -> Path:
    p = tmp_path / "assets.toml"
    p.write_text(body)
    return p


def test_asset_path_does_no_io(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MARCONI_ASSET_ROOT", str(tmp_path))
    p = asset_path("POCSAG/pocsag.cf32")
    assert p == tmp_path / "POCSAG" / "pocsag.cf32"
    assert not p.exists()


def test_require_asset_returns_an_existing_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("MARCONI_ASSET_ROOT", str(tmp_path))
    (tmp_path / "P").mkdir()
    (tmp_path / "P" / "x.bin").write_bytes(b"payload")
    assert require_asset("P/x.bin") == tmp_path / "P" / "x.bin"


def test_require_asset_skips_and_registers_when_absent(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # pytest.skip() raises Skipped, a BaseException (not Exception) by
    # design, so pytest.raises(Exception) cannot catch it: the exception
    # would propagate out of the test itself, marking it SKIPPED and
    # leaving the assertions below unreachable. pytest.skip.Exception is
    # pytest's own alias for Skipped, documented for exactly this case.
    monkeypatch.setenv("MARCONI_ASSET_ROOT", str(tmp_path))
    SKIPPED.clear()
    with pytest.raises(pytest.skip.Exception) as exc:
        require_asset("P/absent.bin")
    assert "P/absent.bin" in str(exc.value)
    assert "P/absent.bin" in SKIPPED


def test_require_asset_never_reaches_the_network(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # a fetchable asset pointed at a dead port must skip immediately, not
    # attempt a download: an e2e run has to be offline and deterministic
    monkeypatch.setenv("MARCONI_ASSET_ROOT", str(tmp_path))

    def _boom(*args: object, **kw: object) -> None:
        raise AssertionError("require_asset attempted a network call")

    monkeypatch.setattr("urllib.request.urlopen", _boom)
    SKIPPED.clear()
    with pytest.raises(Exception) as exc:
        require_asset("P/absent.bin")
    assert "Skipped" in type(exc.value).__name__


def test_strict_failures_names_only_ci_required_assets(tmp_path: Path) -> None:
    index = load_manifest(
        _manifest(
            tmp_path,
            """
[[asset]]
kind = "local"
path = "P/needed.bin"
ci_required = true

[[asset]]
kind = "local"
path = "P/optional.bin"
""",
        )
    )
    assert strict_failures(index, {"P/optional.bin"}) == []
    assert strict_failures(index, {"P/needed.bin", "P/optional.bin"}) == [
        "P/needed.bin"
    ]
