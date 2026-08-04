#!/bin/sh
# Launches the Marconi MCP server (stdio), creating its venv on first run.
# GNU Radio only imports from a --system-site-packages venv built on the
# interpreter GNU Radio was installed for, so the bootstrap discovers that
# interpreter instead of trusting `python3`. stdout is the MCP protocol
# stream: every bootstrap message must go to stderr.
set -eu

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
VENV="$ROOT/.venv"

log() { printf '%s\n' "marconi-mcp: $*" >&2; }

die() {
    log "$*"
    exit 1
}

find_gr_python() {
    for p in "${MARCONI_PYTHON:-}" python3 python3.14 python3.13 python3.12 \
        python3.11 /opt/homebrew/bin/python3 /usr/local/bin/python3 \
        /usr/bin/python3; do
        [ -n "$p" ] || continue
        command -v "$p" >/dev/null 2>&1 || continue
        if "$p" -c 'import gnuradio' >/dev/null 2>&1; then
            command -v "$p"
            return 0
        fi
    done
    return 1
}

bootstrap() {
    log "first run: setting up $VENV"
    PY="$(find_gr_python)" || die "no Python with GNU Radio found. Install \
GNU Radio 3.10 (macOS: 'brew install gnuradio'; Debian/Ubuntu: 'apt install \
gnuradio') or point MARCONI_PYTHON at an interpreter that can 'import gnuradio'."
    log "using $PY"
    if command -v uv >/dev/null 2>&1; then
        uv venv --system-site-packages --python "$PY" "$VENV" >&2
        uv pip install --python "$VENV/bin/python" -e "$ROOT[mcp]" >&2
    else
        "$PY" -m venv --system-site-packages "$VENV" >&2
        "$VENV/bin/python" -m pip install -e "$ROOT[mcp]" >&2
    fi
}

[ -x "$VENV/bin/marconi-mcp" ] || bootstrap

"$VENV/bin/python" -c 'import gnuradio' >/dev/null 2>&1 || die "$VENV cannot \
import gnuradio (it was probably created without --system-site-packages). \
Delete it (rm -rf $VENV) and rerun."

exec "$VENV/bin/marconi-mcp"
