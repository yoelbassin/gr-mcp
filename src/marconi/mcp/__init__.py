"""Marconi MCP server (Layer 2): a thin FastMCP adapter over the marconi ops.

Importing this package is deliberately cheap — submodules (server, tools) pull
in fastmcp and are imported explicitly where needed, so `import marconi` stays
free of fastmcp and gnuradio."""
