import os

import pytest


def pytest_sessionfinish(session: pytest.Session, exitstatus: int) -> None:
    if os.environ.get("MARCONI_ASSETS_STRICT") != "1":
        return
    if hasattr(session.config, "workerinput"):
        return
    from helpers.assets import missing_required
    from helpers.assets.manifest import load_manifest
    from helpers.assets.root import asset_root

    root = asset_root()
    missing = missing_required(load_manifest(), root)
    if not missing:
        return
    if session.exitstatus == 0:
        session.exitstatus = 1
    reporter = session.config.pluginmanager.get_plugin("terminalreporter")
    if reporter is not None:
        reporter.write_line(
            f"ASSETS STRICT: required assets absent from {root}:", red=True
        )
        for name in missing:
            reporter.write_line(f"  {name}", red=True)
