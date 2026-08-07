"""MCP tool loader — discover tools from MCP servers over the protocol.

The agent doesn't hand-wire these tools; it discovers them at runtime from any
MCP server. get_tools() is async, so this wraps it in a synchronous call for the
framework's sync API. `config` follows MultiServerMCPClient's shape:
    {"name": {"command": "python", "args": ["server.py"], "transport": "stdio"}}
"""

import asyncio

from langchain_mcp_adapters.client import MultiServerMCPClient


def load_mcp_tools(config: dict) -> list:
    """Connect to the configured MCP server(s) and return their tools (synchronously)."""
    async def _get():
        client = MultiServerMCPClient(config)
        return await client.get_tools()

    return asyncio.run(_get())
