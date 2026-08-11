from pathlib import Path

TESTS_ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS = TESTS_ROOT / "artifacts"
SRC_MARCONI = TESTS_ROOT.parent / "src" / "marconi"

# The repo-root artifact tree (the larger off-air captures); tests/artifacts
# holds the committed slices. Both are checkout-relative: an absolute path
# baked into a gate turns a moved or renamed checkout into a permanent skip
# whose reason still reads "asset absent".
REPO_ARTIFACTS = TESTS_ROOT.parent / "artifacts"
