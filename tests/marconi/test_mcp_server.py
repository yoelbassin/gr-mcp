def test_base_import_does_not_load_mcp_or_fastmcp():
    import subprocess
    import sys

    # Run in a fresh interpreter so other test modules' top-level imports of
    # fastmcp/marconi.mcp can't pollute this check.
    code = (
        "import marconi, sys; "
        "assert 'fastmcp' not in sys.modules, 'fastmcp leaked into base import'; "
        "assert 'gnuradio' not in sys.modules, 'gnuradio leaked into base import'; "
        "assert 'marconi.mcp.server' not in sys.modules"
    )
    subprocess.run([sys.executable, "-c", code], check=True)


def test_mcp_subpackage_importable():
    import marconi.mcp  # noqa: F401


def test_entry_point_target_exists():
    from marconi.mcp.server import main  # noqa: F401
