from pathlib import Path

TESTS_ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS = TESTS_ROOT / "artifacts"
SRC_MARCONI = TESTS_ROOT.parent / "src" / "marconi"
