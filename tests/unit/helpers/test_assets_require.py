from __future__ import annotations

import importlib.util
from pathlib import Path
from types import ModuleType
from typing import cast

import pytest
from helpers.assets import SKIPPED, asset_path, require_asset, strict_failures
from helpers.assets.manifest import LocalAsset, load_manifest

_CONFTEST_PATH = Path(__file__).resolve().parents[2] / "conftest.py"


def _manifest(tmp_path: Path, body: str) -> Path:
    p = tmp_path / "assets.toml"
    p.write_text(body)
    return p


def _load_conftest() -> ModuleType:
    spec = importlib.util.spec_from_file_location(
        "assets_require_conftest_under_test", _CONFTEST_PATH
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class _StubReporter:
    def __init__(self) -> None:
        self.lines: list[str] = []

    def write_line(self, message: str, red: bool = False) -> None:
        self.lines.append(message)


class _StubPluginManager:
    def __init__(self, reporter: _StubReporter) -> None:
        self._reporter = reporter

    def get_plugin(self, name: str) -> _StubReporter | None:
        return self._reporter if name == "terminalreporter" else None


class _StubConfig:
    def __init__(self, reporter: _StubReporter) -> None:
        self.pluginmanager = _StubPluginManager(reporter)


class _StubSession:
    def __init__(self, reporter: _StubReporter) -> None:
        self.config = _StubConfig(reporter)
        self.exitstatus = 0


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
    # attempt a download: an e2e run has to be offline and deterministic.
    # A real network call would raise the patched urlopen's AssertionError,
    # not Skipped, so pytest.raises(pytest.skip.Exception) lets that escape
    # uncaught and fail the test instead of masking it as another skip.
    monkeypatch.setenv("MARCONI_ASSET_ROOT", str(tmp_path))

    def _boom(*args: object, **kw: object) -> None:
        raise AssertionError("require_asset attempted a network call")

    monkeypatch.setattr("urllib.request.urlopen", _boom)
    SKIPPED.clear()
    with pytest.raises(pytest.skip.Exception):
        require_asset("P/absent.bin")


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


def test_sessionfinish_is_a_noop_when_not_strict(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("MARCONI_ASSETS_STRICT", raising=False)

    def _must_not_be_called(*a: object, **k: object) -> dict[str, LocalAsset]:
        raise AssertionError("load_manifest called while not strict")

    monkeypatch.setattr("helpers.assets.manifest.load_manifest", _must_not_be_called)
    reporter = _StubReporter()
    session = _StubSession(reporter)
    conftest = _load_conftest()
    conftest.pytest_sessionfinish(cast(pytest.Session, session), 0)
    assert session.exitstatus == 0
    assert reporter.lines == []


def test_sessionfinish_is_a_noop_when_strict_and_nothing_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("MARCONI_ASSETS_STRICT", "1")
    monkeypatch.setattr(
        "helpers.assets.manifest.load_manifest",
        lambda *a, **k: {
            "P/optional.bin": LocalAsset(path="P/optional.bin", kind="local"),
        },
    )
    SKIPPED.clear()
    SKIPPED.add("P/optional.bin")
    reporter = _StubReporter()
    session = _StubSession(reporter)
    conftest = _load_conftest()
    conftest.pytest_sessionfinish(cast(pytest.Session, session), 0)
    assert session.exitstatus == 0
    assert reporter.lines == []


def test_sessionfinish_fails_the_run_when_strict_and_ci_required_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("MARCONI_ASSETS_STRICT", "1")
    monkeypatch.setattr(
        "helpers.assets.manifest.load_manifest",
        lambda *a, **k: {
            "P/needed.bin": LocalAsset(
                path="P/needed.bin", kind="local", ci_required=True
            ),
            "P/optional.bin": LocalAsset(path="P/optional.bin", kind="local"),
        },
    )
    SKIPPED.clear()
    SKIPPED.add("P/needed.bin")
    SKIPPED.add("P/optional.bin")
    reporter = _StubReporter()
    session = _StubSession(reporter)
    conftest = _load_conftest()
    conftest.pytest_sessionfinish(cast(pytest.Session, session), 0)
    assert session.exitstatus == 1
    assert reporter.lines == [
        "ASSETS STRICT: required assets absent, gates did not run:",
        "  P/needed.bin",
    ]
