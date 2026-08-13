from __future__ import annotations

import importlib.util
import os
import subprocess
import sys
from pathlib import Path
from types import ModuleType
from typing import cast

import pytest
from helpers.assets import asset_path, missing_required, require_asset
from helpers.assets.manifest import LocalAsset, load_manifest

_CONFTEST_PATH = Path(__file__).resolve().parents[2] / "conftest.py"
_REPO_ROOT = Path(__file__).resolve().parents[3]

# a cheap module that names no asset, so each invocation below would exit 0
# on its own merits: only the strict verdict can turn it red
_OTHER = "tests/unit/helpers/test_assets_manifest.py"

_REQUIRED = {
    "P/needed.bin": LocalAsset(path="P/needed.bin", kind="local", ci_required=True),
    "P/optional.bin": LocalAsset(path="P/optional.bin", kind="local"),
}


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
    def __init__(self, reporter: _StubReporter, exitstatus: int = 0) -> None:
        self.config = _StubConfig(reporter)
        self.exitstatus = exitstatus


def _finish(session: _StubSession) -> None:
    _load_conftest().pytest_sessionfinish(cast(pytest.Session, session), 0)


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


def test_require_asset_skips_when_absent(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # pytest.skip() raises Skipped, a BaseException (not Exception) by
    # design, so pytest.raises(Exception) cannot catch it: the exception
    # would propagate out of the test itself, marking it SKIPPED and
    # leaving the assertions below unreachable. pytest.skip.Exception is
    # pytest's own alias for Skipped, documented for exactly this case.
    monkeypatch.setenv("MARCONI_ASSET_ROOT", str(tmp_path))
    with pytest.raises(pytest.skip.Exception) as exc:
        require_asset("P/absent.bin")
    assert "P/absent.bin" in str(exc.value)


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
    with pytest.raises(pytest.skip.Exception):
        require_asset("P/absent.bin")


def test_missing_required_reads_the_filesystem_not_the_skips(tmp_path: Path) -> None:
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
    root = tmp_path / "assets"
    (root / "P").mkdir(parents=True)
    assert missing_required(index, root) == ["P/needed.bin"]
    (root / "P" / "needed.bin").write_bytes(b"x")
    assert missing_required(index, root) == []


def test_sessionfinish_is_a_noop_when_not_strict(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("MARCONI_ASSETS_STRICT", raising=False)

    def _must_not_be_called(*a: object, **k: object) -> dict[str, LocalAsset]:
        raise AssertionError("load_manifest called while not strict")

    monkeypatch.setattr("helpers.assets.manifest.load_manifest", _must_not_be_called)
    session = _StubSession(_StubReporter())
    _finish(session)
    assert session.exitstatus == 0


def test_sessionfinish_passes_when_the_required_asset_is_on_disk(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("MARCONI_ASSETS_STRICT", "1")
    monkeypatch.setenv("MARCONI_ASSET_ROOT", str(tmp_path))
    monkeypatch.setattr(
        "helpers.assets.manifest.load_manifest", lambda *a, **k: _REQUIRED
    )
    (tmp_path / "P").mkdir()
    (tmp_path / "P" / "needed.bin").write_bytes(b"x")
    reporter = _StubReporter()
    session = _StubSession(reporter)
    _finish(session)
    assert session.exitstatus == 0
    assert reporter.lines == []


def test_sessionfinish_fails_on_an_absent_asset_that_nothing_skipped_for(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # nothing in this process ever called require_asset, so a skip-driven
    # verdict has no evidence at all: only the filesystem check can fail here
    monkeypatch.setenv("MARCONI_ASSETS_STRICT", "1")
    monkeypatch.setenv("MARCONI_ASSET_ROOT", str(tmp_path))
    monkeypatch.setattr(
        "helpers.assets.manifest.load_manifest", lambda *a, **k: _REQUIRED
    )
    reporter = _StubReporter()
    session = _StubSession(reporter)
    _finish(session)
    assert session.exitstatus == 1
    assert reporter.lines == [
        f"ASSETS STRICT: required assets absent from {tmp_path}:",
        "  P/needed.bin",
    ]


def test_sessionfinish_never_downgrades_a_worse_exit_status(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("MARCONI_ASSETS_STRICT", "1")
    monkeypatch.setenv("MARCONI_ASSET_ROOT", str(tmp_path))
    monkeypatch.setattr(
        "helpers.assets.manifest.load_manifest", lambda *a, **k: _REQUIRED
    )
    reporter = _StubReporter()
    session = _StubSession(reporter, exitstatus=2)
    _finish(session)
    assert session.exitstatus == 2
    assert reporter.lines[0].startswith("ASSETS STRICT")


def _run_pytest(
    argv: list[str], root: Path, *, strict: bool
) -> subprocess.CompletedProcess[str]:
    env = dict(os.environ)
    env["MARCONI_ASSET_ROOT"] = str(root)
    if strict:
        env["MARCONI_ASSETS_STRICT"] = "1"
    else:
        env.pop("MARCONI_ASSETS_STRICT", None)
    return subprocess.run(
        [sys.executable, "-m", "pytest", *argv, "-q", "-p", "no:cacheprovider"],
        cwd=_REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
        timeout=600,
    )


def test_pocsag_gate_directory_exits_zero_when_lenient_and_asset_absent(
    tmp_path: Path,
) -> None:
    # A regression to a module-level `require_asset(...)` call (instead of
    # `asset_fixture(...)`) would skip the whole module at collection time,
    # collecting zero items and exiting 5 (NO_TESTS_COLLECTED) instead of 0 -
    # this asserts the exit code, not just "no failure", so it catches that.
    result = _run_pytest(["tests/e2e/pocsag", "-rs"], tmp_path, strict=False)
    assert result.returncode == 0, result.stdout + result.stderr
    assert "POCSAG/pocsag.cf32" in result.stdout


# Every one of these exited 0 while the strict verdict was driven by the skips
# a run happened to record: the xdist workers keep their skips to themselves,
# a deselected gate records none, and a selection that names no gate at all
# never reaches require_asset. The verdict is now the filesystem's.
@pytest.mark.parametrize(
    "argv",
    [
        pytest.param(["tests/e2e/pocsag"], id="plain"),
        pytest.param(["tests/e2e/pocsag", "-n", "2"], id="xdist"),
        pytest.param(["tests/e2e/pocsag", _OTHER], id="alongside-other-tests"),
        pytest.param(["tests/e2e/pocsag", _OTHER, "-k", "manifest"], id="deselected"),
        pytest.param([_OTHER], id="no-gate-selected"),
    ],
)
def test_strict_fails_whatever_the_invocation(argv: list[str], tmp_path: Path) -> None:
    result = _run_pytest(argv, tmp_path, strict=True)
    assert result.returncode == 1, result.stdout + result.stderr
    assert "ASSETS STRICT: required assets absent" in result.stdout
    assert "POCSAG/pocsag.cf32" in result.stdout
