import os

import pytest


def pytest_sessionfinish(session: pytest.Session, exitstatus: int) -> None:
    if os.environ.get("MARCONI_ASSETS_STRICT") != "1":
        return
    from helpers.assets import SKIPPED, strict_failures
    from helpers.assets.manifest import load_manifest

    missing = strict_failures(load_manifest(), SKIPPED)
    if not missing:
        return
    session.exitstatus = 1
    reporter = session.config.pluginmanager.get_plugin("terminalreporter")
    if reporter is not None:
        reporter.write_line(
            "ASSETS STRICT: required assets absent, gates did not run:", red=True
        )
        for name in missing:
            reporter.write_line(f"  {name}", red=True)
