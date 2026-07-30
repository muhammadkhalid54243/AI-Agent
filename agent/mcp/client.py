"""A thin MCP client wrapper.

Connects to an MCP server over stdio, discovers its tools through the protocol,
and exposes two things the rest of the app needs:
  - openai_tools(): the discovered tools in OpenAI tool-schema format
  - call(name, args): execute a tool call THROUGH the protocol (not a local function)

The agent never imports the server. It learns what the server offers at runtime.
That runtime discovery is what makes MCP a *standard* rather than hand-wiring.
"""

from contextlib import asynccontextmanager

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client


class MCPClient:
    def __init__(self, session: ClientSession):
        self._session = session
        self._tools = []

    async def discover(self):
        """Ask the server what tools it has (the MCP handshake step)."""
        result = await self._session.list_tools()
        self._tools = result.tools
        return [t.name for t in self._tools]

    def openai_tools(self):
        """Convert MCP tool definitions → OpenAI tool-schema format.

        This is the adapter seam: MCP describes tools with `inputSchema`,
        the LLM APIs expect `parameters`. Same information, different shape.
        """
        return [
            {
                "type": "function",
                "function": {
                    "name": t.name,
                    "description": t.description or "",
                    "parameters": t.inputSchema,
                },
            }
            for t in self._tools
        ]

    async def call(self, name: str, args: dict) -> str:
        """Execute a tool call through the protocol and return its text result."""
        result = await self._session.call_tool(name, args)
        parts = []
        for block in result.content:
            text = getattr(block, "text", None)
            if text is not None:
                parts.append(text)
        return "\n".join(parts) if parts else "(no text content returned)"


@asynccontextmanager
async def connect(command: str, args: list[str]):
    """Spawn an MCP server subprocess and yield a ready MCPClient.

    Usage:
        async with connect("python", ["mcp_server/server.py"]) as client:
            names = await client.discover()
    """
    server_params = StdioServerParameters(command=command, args=args)
    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            client = MCPClient(session)
            yield client
