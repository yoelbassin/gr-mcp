import marconi  # noqa: F401


def test_base_import_does_not_load_mcp_or_fastmcp():
    import sys

    # importing the top-level package must not pull in the MCP adapter or fastmcp
    assert "fastmcp" not in sys.modules
    assert "marconi.mcp.server" not in sys.modules


def test_mcp_subpackage_importable():
    import marconi.mcp  # noqa: F401


def test_entry_point_target_exists():
    from marconi.mcp.server import main  # noqa: F401
