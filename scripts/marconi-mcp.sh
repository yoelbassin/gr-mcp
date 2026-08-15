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

create_venv() {
    PY="$(find_gr_python)" || die "no Python with GNU Radio found. Install \
GNU Radio 3.10 (macOS: 'brew install gnuradio'; Debian/Ubuntu: 'apt install \
gnuradio') or point MARCONI_PYTHON at an interpreter that can 'import gnuradio'."
    log "using $PY"
    if command -v uv >/dev/null 2>&1; then
        uv venv --system-site-packages --python "$PY" "$VENV" >&2
    else
        "$PY" -m venv --system-site-packages "$VENV" >&2
    fi
}

# Creating the venv and installing into it are separate because the refresh
# below runs only when the venv ALREADY exists, and `uv venv` refuses a
# directory that exists rather than reusing it - merging these back makes
# every refresh exit non-zero, which the MCP client reports as a bare
# CONNECTION_CLOSED. The venv is also frequently the checkout's own dev
# environment, so a refresh must never recreate it.
install_deps() {
    if command -v uv >/dev/null 2>&1; then
        uv pip install --python "$VENV/bin/python" -e "$ROOT[mcp]" >&2
    else
        "$VENV/bin/python" -m pip install -e "$ROOT[mcp]" >&2
    fi
    cp "$ROOT/pyproject.toml" "$VENV/.marconi-pyproject-stamp"
}

# Reinstall when pyproject changed: the install is editable, so source
# changes arrive free but NEW DEPENDENCIES do not - after a plugin update
# they ImportError'd inside a tool call with nothing pointing here. A missing
# stamp counts as changed: a venv built by anything other than this script
# (`uv sync`, a hand-rolled `uv venv --clear`) has marconi's deps unverified.
if [ ! -d "$VENV" ]; then
    log "first run: setting up $VENV"
    create_venv
    install_deps
elif [ ! -x "$VENV/bin/marconi-mcp" ]; then
    log "$VENV has no marconi install: installing"
    install_deps
elif ! cmp -s "$ROOT/pyproject.toml" "$VENV/.marconi-pyproject-stamp"; then
    log "pyproject.toml changed since install: refreshing dependencies"
    install_deps
fi

# Print the REAL import failure: discarding it and guessing at the cause
# ("probably created without --system-site-packages") sent users into an
# infinite delete-and-rebuild loop when the actual error was a numpy ABI
# mismatch between the venv's numpy and the one GNU Radio was built against.
if ! err="$("$VENV/bin/python" -c 'import numpy; import gnuradio' 2>&1)"; then
    log "$VENV cannot import gnuradio. The actual error:"
    printf '%s\n' "$err" >&2
    die "if this is a numpy ABI mismatch, align the venv's numpy with the \
one GNU Radio was built against; if gnuradio is missing entirely, the venv \
was probably created without --system-site-packages - delete it \
(rm -rf $VENV) and rerun."
fi

exec "$VENV/bin/marconi-mcp"
