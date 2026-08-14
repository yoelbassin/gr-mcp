"""The README's tool table is the first document a user reads, and it
survived the receive-only refactor advertising run_tx - a tool that does not
exist - while omitting two that do. The table is generated FROM the code's
TOOLS registry in spirit; this gate makes that literal."""

from __future__ import annotations

import re
from pathlib import Path

from marconi.mcp.tools import TOOLS

_README = Path(__file__).resolve().parents[3] / "README.md"


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
