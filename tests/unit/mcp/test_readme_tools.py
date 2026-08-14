"""The README's tool table is the first document a user reads, and it
survived the receive-only refactor advertising run_tx - a tool that does not
exist - while omitting two that do. The table is generated FROM the code's
TOOLS registry in spirit; this gate makes that literal."""

from __future__ import annotations

import re
from pathlib import Path

from marconi.mcp.tools import TOOLS

_ROOT = Path(__file__).resolve().parents[3]
_README = _ROOT / "README.md"
_LAUNCHER = _ROOT / "scripts" / "marconi-mcp.sh"

# The two causes of a failing `import gnuradio` the launcher hedges between.
# Naming only the second is the refuted diagnosis it carries a comment about:
# a numpy ABI mismatch survives a venv rebuild, so "delete .venv and relaunch"
# as the single remedy is an infinite loop, which is how it was found.
_CAUSES = ("numpy", "--system-site-packages")


def test_readme_tool_table_matches_the_registry() -> None:
    text = _README.read_text()
    rows = set(re.findall(r"^\| `(\w+)` \|", text, re.MULTILINE))
    registered = set(TOOLS)
    assert rows == registered, (
        f"README table vs TOOLS registry: phantom {sorted(rows - registered)}, "
        f"missing {sorted(registered - rows)}"
    )


def test_readme_makes_no_transmit_claim() -> None:
    text = _README.read_text().lower()
    for phrase in ("run_tx", "generates a signal", "and back"):
        assert phrase not in text, phrase


def _import_bullet() -> str:
    text = _README.read_text()
    start = text.index("`gnuradio` won't import")
    end = text.index("\n- ", start)
    return " ".join(text[start:end].split())


def _launcher_diagnosis() -> str:
    """What the launcher says when the venv it built cannot import gnuradio —
    the diagnosis the README's bullet is the prose copy of."""
    script = _LAUNCHER.read_text()
    guard = script.index("import numpy; import gnuradio")
    message = re.search(r'\bdie "(.*?)"', script[guard:], re.DOTALL)
    assert message is not None, "the launcher no longer dies on a failed import"
    return " ".join(message.group(1).replace("\\\n", "").split())


def test_the_readme_hedges_the_import_failure_the_way_the_launcher_does() -> None:
    """The launcher prints the REAL import error and hedges both causes,
    carrying a comment that guessing sent users into a delete-and-rebuild
    loop. The README asserted the guess, unconditionally, as the single cause
    — so a reader following it re-enters the loop the launcher was fixed to
    end. Every cause the shipped launcher names must be named here too."""
    diagnosis = _launcher_diagnosis()
    hedged = [cause for cause in _CAUSES if cause in diagnosis]
    assert hedged == list(_CAUSES), diagnosis
    bullet = _import_bullet()
    for cause in hedged:
        assert cause in bullet, f"{cause} missing from: {bullet}"
    # the rebuild remedy may only follow the one cause that justifies it
    assert bullet.index("--system-site-packages") < bullet.index("rm -rf"), bullet
